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
from typing import Tuple, Set, Dict, Optional


class Node(namedtuple("Node", "op_type node_name op_name attributes")):
    def __repr__(self):
        return f'Node({self.node_name})'


class Input(namedtuple("Input", "dtype shape name")):
    def __repr__(self):
        return f'Input({self.shape}, dtype={self.dtype}, name={self.name})'


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
        self._input_node_names: Set[str] = set()

    def copy(self):
        g = Graph()
        g._graph = self._graph.copy()
        g._input_node_names = self._input_node_names.copy()
        return g

    @property
    def nodes(self):
        return self._graph.nodes

    def toposort(self):
        return nx.topological_sort(self._graph)

    def reversed_toposort(self):
        return reversed(list(self.toposort()))

    def add_node(
            self, op_name: str,
            inputs: Tuple[Tuple[Optional[int], Optional["array.Array"]], ...],
            **attributes):
        op_id = uuid.uuid4()
        node_name = f"{op_name}_{op_id}"

        new_node = Node(
            None,  # op_type is added by register decorator
            node_name,
            op_name,
            HashableAttributes(**attributes))

        self._graph.add_node(node_name, node=new_node)

        for idx, (input_array_idx, input_array) in enumerate(inputs):
            # nodes can have empty inputs
            if input_array is not None:
                if input_array_idx is None:
                    raise InternalException(
                        "Input array does not have a index associated to it")
                parent_node = input_array._evaluator._parent_node
                if parent_node is not None:
                    tensor_id = uuid.uuid4()
                    edge_name = f"Tensor_{tensor_id}"
                    # adds edge between previous node (which provides an input) and
                    # this node
                    self._graph.add_edge(
                        parent_node, node_name, in_index=input_array_idx,
                        out_index=idx, name=edge_name)
            else:
                self._graph.add_edge(
                    None, node_name, in_index=input_array_idx, out_index=idx,
                    name=edge_name)
        return node_name

    def add_initializer(
            self, name: str, dtype: np.dtype, dims: Tuple[int],
            vals):
        raise NotImplementedError()

    def add_input(self, array: "array.Array"):
        input_id = uuid.uuid4()
        node_name = f"Input_{input_id}"
        new_node = Input(array.dtype, array.shape, node_name)
        self._graph.add_node(node_name, node=new_node)
        self._input_node_names.add(node_name)
        return node_name, array

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
            from_node, array._internal_name, in_index=idx,
            out_index=0, name=array._internal_name)
        return array._internal_name

    def add_subgraph(self, other_graph):
        if self._graph is None:
            raise InternalException()
        elif other_graph._graph is not None:
            self._graph = nx.compose(self._graph, other_graph._graph)
            self._input_node_names.update(other_graph._input_node_names)
            other_graph._input_node_names = self._input_node_names


class ExecutableGraph:
    def __init__(
            self, graph: Graph,
            node_inputs: Dict[str, Tuple["array.Array"]],
            node_outputs: Dict[str, Tuple["array.Array"]],
            outputs: Dict[str, "array.Array"],
            cached_results: "evaluator.IntermediateResultCache"):
        self._graph = graph.copy()
        self._output_names: Set[str] = set()
        self._input_node_mapping: Dict[str, str] = {}

        for output_node_name, output_array in outputs.items():
            output_arrays = node_outputs[output_node_name]
            output_array_names = [a._internal_name for a in output_arrays]
            output_array_idx = output_array_names.index(
                output_array._internal_name)
            if output_array_idx == -1:
                raise InternalException(
                    f"Could not find output array in output node {output_node_name}")

            self._graph.add_output(
                output_array, output_node_name, output_array_idx)
            self._output_names.add(output_array._internal_name)

            # this is needed since some node outputs are required
            # For example Unique has outputs [required, optional1, optional2, optional3]
            # if we just need optional1, we still need "required" to be an output of the
            # node
            # This assumes that only the one output of this node will be a graph output
            # that will be used. If that is not the case anything could happen, since
            # we don't add any of the remaining array names to self._output_names
            remaining_node_outputs = [
                (i, o) for i, o in enumerate(output_arrays)
                if o._internal_name != output_array._internal_name]

            for other_output in remaining_node_outputs:
                output_array_idx, output_array = other_output
                self._graph.add_output(
                    output_array, output_node_name, output_array_idx)

        for input_name, _ in node_inputs.items():
            if input_name in self._graph._input_node_names:
                input_out_edge = self._graph._graph.out_edges(
                    input_name, data=True)
                if len(input_out_edge) == 0:
                    raise InternalException()
                for e in input_out_edge:
                    edge_name = e[-1]["name"]
                    self._input_node_mapping[input_name] = edge_name

        # cached_array_names = [a._internal_name
        #                       for a in cached_results._cache.values()]

        # if not cached_results.empty():
        #     for node_name, result in cached_results.to_dict().items():
        #         if node_name in self._graph.nodes:
        #             self._input_names.add(self._graph.cast_node_to_input(
        #                 result, node_name))

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

    def get_input_node_mapping(self) -> Dict[str, str]:
        return self._input_node_mapping

    def build_onnx_graph(self) -> onnx.GraphProto:
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
                out_edges = template_graph.out_edges(node_name, data=True)
                if len(out_edges) == 0:
                    raise InternalException(len(out_edges))
                for e in out_edges:
                    out_node = e[1]
                    if out_node in ancestors:
                        g.input.append(
                            onnx.helper.make_tensor_value_info(
                                e[-1]["name"],
                                numpy_to_onnx(np.dtype(node.dtype)),
                                node.shape.tolist()))
            elif isinstance(node, Node):
                in_edges = template_graph.in_edges(node_name, data=True)
                out_edges = template_graph.out_edges(node_name, data=True)

                in_edges = list(in_edges)
                out_edges = list(out_edges)

                # input list can be empty.
                # For example, Constant only uses attribute values to generate output
                if len(in_edges) > 0:
                    idx = [e[-1]["out_index"] for e in in_edges]
                    input_names = [""] * (max(idx) + 1)
                    for e in in_edges:
                        if e[0] is not None:
                            input_names[e[-1]["out_index"]] = e[-1]["name"]
                else:
                    input_names = []

                # # ensures that empty inputs are still added to the graph as ""
                # # which means empty input in ONNX
                # # This makes sure that all inputs are in the correct argument position
                idx = [e[-1]["in_index"] for e in out_edges]
                output_names = [""] * (max(idx) + 1)
                for e in out_edges:
                    if e[0] is not None:
                        output_names[e[-1]["in_index"]] = e[-1]["name"]

                out_edges = list(out_edges)

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


def compile_graph(
        graph: Graph, node_inputs: Dict[str, Tuple["array.Array"]],
        node_outputs: Dict[str, Tuple["array.Array"]],
        outputs: Dict[str, "array.Array"],
        cached_results: "evaluator.IntermediateResultCache") -> ExecutableGraph:
    return ExecutableGraph(graph, node_inputs, node_outputs, outputs,
                           cached_results)
