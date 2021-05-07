from .types import numpy_to_onnx
from . import array
from . import evaluator
from . import onnx_utils
from .exceptions import InternalException

import numpy as np
import networkx as nx
import onnx
import uuid

from collections import namedtuple
from collections.abc import Hashable
from typing import Tuple, List, Set, Dict


class Node(namedtuple("Node", "op_type node_name op_name attributes")):
    def __repr__(self):
        return f'Node({self.node_name})'


class Input(namedtuple("Input", "dtype shape")):
    def __repr__(self):
        return f'Input({self.shape}, dtype={self.dtype})'


class Output(namedtuple("Output", "dtype shape")):
    def __repr__(self):
        return f'Output({self.shape}, dtype={self.dtype})'


class TensorProtoInternal(
        namedtuple("TensorProtoInternal", "values dtype shape name")):
    def __repr__(self):
        return f'TensorProtoInternal({self.shape}, dtype={self.dtype})'


class HashableAttributes(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, list):
                kwargs[k] = tuple(v)
            elif not isinstance(v, Hashable):  # pragma: no cover
                raise TypeError(
                    f"Value has to be hashable, but got non-hashable type {type(v)}")
        super(HashableAttributes, self).__init__(**kwargs)
        self._hash = hash(frozenset(self.items()))

    def __hash__(self):
        # constant hash as this should be immutable
        return self._hash

    def __getitem__(self, name):
        value = super(HashableAttributes, self).__getitem__(name)
        if isinstance(value, array.Array):
            return onnx_utils.make_onnx_tensor(value._internal_name, value)
        return value

    def __iter__(self):  # pragma: no cover
        for value in super(HashableAttributes, self).__iter__():
            if isinstance(value, array.Array):
                yield onnx_utils.make_onnx_tensor(value._internal_name, value)
            yield value


# TODO
def build_node_from_onnx(onnx_proto):  # pragma: no cover
    return Node(tuple(onnx_proto.input), tuple(onnx_proto.output),
                onnx_proto.op_type,
                HashableAttributes(onnx_proto.attribute),
                onnx_proto.name)


def build_graph_from_onnx(onnx_graph, outputs):  # pragma: no cover
    # TODO
    raise NotImplementedError("FIXME")
    G = nx.DiGraph()
    node_map = {}
    for o in outputs:
        node_map[o] = Node((o,), "", "Output", HashableAttributes({}), o)
        G.add_node(node_map[o])

    for node in onnx_graph.node:
        node_obj = build_node_from_onnx(node)
        for input in node.input:
            node_map[input] = node_obj

    for i in onnx_graph.input:
        node_obj = node_map[i.name]
        G.add_edge(
            Node(
                "", (i.name,),
                "Input", HashableAttributes({}),
                i.name),
            node_obj)

    for node in onnx_graph.node:
        for output in node.output:
            node_obj = node_map[output]
            for input in node.input:
                G.add_edge(node_map[input], node_obj)

    return G


class Graph:
    def __init__(self):
        self._graph = nx.MultiDiGraph()

    def copy(self):
        g = Graph()
        g._graph = self._graph.copy()
        return g

    @property
    def nodes(self):
        return self._graph.nodes

    def toposort(self):
        return nx.topological_sort(self._graph)

    def reversed_toposort(self):
        return reversed(list(self.toposort()))

    def add_node(
            self, op_name: str, inputs: List["array.Array"],
            **attributes):

        op_id = uuid.uuid4()
        node_name = f"{op_name}_{op_id}"

        new_node = Node(
            None,  # op_type is added by register decorator
            node_name,
            op_name,
            HashableAttributes(**attributes))

        self._graph.add_node(node_name, node=new_node)

        for idx, i in enumerate(inputs):
            # nodes can have empty inputs
            if i is not None:
                if i._evaluator._parent_node is not None:
                    self._graph.add_edge(
                        i._evaluator._parent_node, node_name, index=idx,
                        array=i)
            else:
                self._graph.add_edge(None, node_name, index=idx, array=None)
        return node_name

    def add_initializer(
            self, name: str, dtype: np.dtype, dims: Tuple[int],
            vals):
        raise NotImplementedError()

    def add_input(self, array: "array.Array", feed_node_name: str):
        if feed_node_name not in self._graph.nodes:  # pragma: no cover
            raise InternalException("Invalid node to feed array to")

        new_node = Input(array.dtype, array.shape)
        self._graph.add_node(array._internal_name, node=new_node)
        self._graph.add_edge(array._internal_name,
                             feed_node_name,
                             index=0, array=array)
        return array._internal_name

    def cast_node_to_input(self, array, node_name):
        if node_name not in self._graph.nodes:
            raise InternalException(f"Node {node_name} not present")

        # remove all edges leading into the node
        # this will potentially create a dead end in the graph, but onnxruntime
        # should remove this at runtime
        in_edges = list(self._graph.in_edges(node_name))
        self._graph.remove_edges_from(in_edges)

        # now the node becomes an input
        new_node = Input(array.dtype, array.shape)
        self._graph.nodes[node_name]["node"] = new_node
        mapping = {node_name: array._internal_name}
        self._graph = nx.relabel_nodes(self._graph, mapping, copy=False)
        for out_edge in self._graph.out_edges(array._internal_name, data=True):
            out_edge[-1]["array"] = array

        return array._internal_name

    def add_output(self, array, from_node, idx):
        output_node = Output(array.dtype, array.shape)
        self._graph.add_node(array._internal_name,
                             node=output_node)
        # FIXME: when multiple outputs are supported
        self._graph.add_edge(
            from_node, array._internal_name, index=idx, array=array)
        return array._internal_name

    def add_empty_output(self, array, from_node, idx):
        output_node = Output(array.dtype, array.shape)
        self._graph.add_node(array._internal_name,
                             node=output_node)
        # FIXME: when multiple outputs are supported
        self._graph.add_edge(
            from_node, array._internal_name, index=idx, array=array)
        return array._internal_name

    def add_subgraph(self, other_graph):
        if self._graph is None:
            self._graph = other_graph._graph
        elif other_graph._graph is not None:
            self._graph = nx.compose(self._graph, other_graph._graph)


class ExecutableGraph:
    def __init__(
            self, graph: Graph,
            node_inputs: Dict[str, List["array.Array"]],
            node_outputs: Dict[str, List["array.Array"]],
            inputs: Dict[str, "array.Array"],
            outputs: Dict[str, "array.Array"],
            cached_results: "evaluator.IntermediateResultCache"):
        self._graph = graph.copy()

        self._input_names: Set[str] = set()

        inputs_to_add = list()

        cached_array_names = [a._internal_name
                              for a in cached_results._cache.values()]

        for array_name, input_array in inputs.items():
            if array_name not in cached_array_names:
                for node_name, node_input_arrays in node_inputs.items():
                    node_input_array_names = [
                        a._internal_name for a in node_input_arrays
                        if a is not None]
                    if node_name in self._graph.nodes and (
                            array_name in node_input_array_names):
                        inputs_to_add.append((input_array, node_name))

        for input_array, node_name in inputs_to_add:
            self._input_names.add(self._graph.add_input(
                input_array, node_name))

        if not cached_results.empty():
            for node_name, result in cached_results.to_dict().items():
                if node_name in self._graph.nodes:
                    self._input_names.add(self._graph.cast_node_to_input(
                        result, node_name))

        if len(outputs) != 1:
            raise NotImplementedError("Can only handle a single output value")

        self._output_names: Set[str] = set()
        # TODO: fix this loop if more than one output is supported
        # currently assumes that a single node output is used
        for parent_node, output_array in outputs.items():
            node_output_array_names = [a._internal_name
                                       for a in node_outputs[parent_node]]
            idx = node_output_array_names.index(output_array._internal_name)
            if idx == -1:
                raise InternalException(
                    "Could not find output array in output node")

            for node_output_idx, node_output_name in enumerate(
                    node_output_array_names):
                if node_output_idx == idx:
                    self._output_names.add(self._graph.add_output(
                        output_array, parent_node, idx))
                else:
                    a = node_outputs[parent_node][node_output_idx]
                    self._graph.add_output(a, parent_node, node_output_idx)

        if len(self._output_names) != 1:
            raise NotImplementedError("Can only handle a single output value")

    @classmethod
    def from_onnx(cls, onnx_graph, outputs=None):
        g = cls()
        if len(onnx_graph.output) == 0 and outputs is None:
            raise ValueError(
                "ONNX graph does not have an output and none was specified")
        elif outputs is None:
            outputs = onnx_graph.output

        g._graph = build_graph_from_onnx(onnx_graph, outputs)
        if not g._graph.is_directed():
            raise ValueError("Graph is not a DAG")

    def build_onnx_graph(self):
        g = onnx.GraphProto()
        template_graph = self._graph._graph

        # FIXME: make this independent of the number of outputs
        output_name = next(iter(self._output_names))
        ancestors = nx.ancestors(template_graph, output_name)
        if len(ancestors) == 0:
            raise InternalException("Could not find ancestor nodes")
        for node_name in ancestors:
            if node_name is None:
                continue

            node = template_graph.nodes[node_name]["node"]

            if isinstance(node, Input):
                g.input.append(
                    onnx.helper.make_tensor_value_info(
                        node_name, numpy_to_onnx(np.dtype(node.dtype)),
                        node.shape.tolist()))
            elif isinstance(node, Node):
                in_edges = template_graph.in_edges(node_name, data=True)
                out_edges = template_graph.out_edges(node_name, data=True)

                input_names = [
                    e[-1]["array"]._internal_name for e in in_edges
                    if e[-1]["array"] is not None]

                # ensures that empty inputs are still added to the graph as ""
                # which means empty input in ONNX
                # This makes sure that all inputs are in the corrent argument position
                for e in in_edges:
                    if e[-1]["array"] is None:
                        idx = e[-1]["index"]
                        input_names.insert(idx, "")

                output_names = [
                    e[-1]["array"]._internal_name for e in out_edges]

                n = onnx.helper.make_node(node.op_name,
                                          input_names,
                                          output_names,
                                          name=node_name, **node.attributes)
                g.node.append(n)
            else:  # pragma: no cover
                raise InternalException(f"Unhandled node type {type(node)}")

        output_node = template_graph.nodes[output_name]["node"]

        g.output.append(
            onnx.helper.make_tensor_value_info(
                output_name, numpy_to_onnx(np.dtype(output_node.dtype)),
                output_node.shape.tolist()))

        return g
