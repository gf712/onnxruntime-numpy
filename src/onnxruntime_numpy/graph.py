from .types import numpy_to_onnx
from . import array
from . import onnx_utils

import numpy as np
import networkx as nx
import onnx

from collections import namedtuple, Hashable
from typing import Tuple, List


class Node(namedtuple("Node", "inputs outputs op_type attributes")):
    def __repr__(self):
        return f'Node({self.op_type})'


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
            elif not isinstance(v, Hashable):
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

    def __iter__(self):
        for value in super(HashableAttributes, self).__iter__():
            if isinstance(value, array.Array):
                return onnx_utils.make_onnx_tensor(value._internal_name, value)
            return value


def build_node_from_onnx(onnx_proto):
    return Node(tuple(onnx_proto.input), tuple(onnx_proto.output),
                onnx_proto.op_type,
                HashableAttributes(onnx_proto.attribute),
                onnx_proto.name)


def build_graph_from_onnx(onnx_graph, outputs):
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
        self._graph = nx.DiGraph()
        self._input_edges = {}

    def copy(self):
        g = Graph()
        g._graph = self._graph.copy()
        g._input_edges = {**self._input_edges}
        return g

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

    @property
    def nodes(self):
        return self._graph.nodes

    def toposort(self):
        return nx.topological_sort(self._graph)

    def reversed_toposort(self):
        return reversed(list(self.toposort()))

    def add_node(
            self, op_type: str, inputs: List["array.Array"],
            outputs: List["array.Array"],
            **attributes):
        new_node = Node(
            inputs,
            outputs,
            op_type,
            HashableAttributes(**attributes))

        node_name = str(hash(new_node))

        self._graph.add_node(node_name, node=new_node)

        for idx, i in enumerate(inputs):
            # nodes can have empty inputs
            if i is not None:
                # input nodes are not added. This way graphs do not have to be copied
                # around
                if i._evaluator._parent_node is not None:
                    self._graph.add_edge(
                        i._evaluator._parent_node, node_name, index=idx)
                else:
                    self._input_edges[i._internal_name] = node_name
        return node_name

    def add_initializer(
            self, name: str, dtype: np.dtype, dims: Tuple[int],
            vals):
        raise NotImplementedError()

    def add_input(self, array):
        if array._internal_name not in self._input_edges:
            raise ValueError("Invalid input array")

        new_node = Input(array.dtype, array.shape)
        self._graph.add_node(array._internal_name, node=new_node)
        self._graph.add_edge(array._internal_name,
                             self._input_edges[array._internal_name],
                             index=0)
        return array._internal_name

    def add_output(self, array, from_node):
        output_node = Output(array.dtype, array.shape)
        self._graph.add_node(array._internal_name,
                             node=output_node)
        # FIXME: when multiple outputs are supported
        self._graph.add_edge(from_node, array._internal_name, index=0)
        return array._internal_name

    def add_subgraph(self, other_graph):
        if self._graph is None:
            self._graph = other_graph._graph
            self._input_edges = {**other_graph._input_edges}
        elif other_graph._graph is not None:
            self._graph = nx.compose(self._graph, other_graph._graph)
            self._input_edges = {**self._input_edges,
                                 **other_graph._input_edges}

    def _build_graph_with_inputs_outputs(
            self, input_arrays, output_array, output):
        inputs = set()
        input_arrays = list(input_arrays)
        for a in input_arrays:
            inputs.add(self.add_input(a))

        output_name = self.add_output(output_array, output)

        return inputs, output_name

    def _remove_graph_inputs_outputs(self, input_names, output_name):
        for n in input_names:
            self._graph.remove_node(n)
        self._graph.remove_node(output_name)

    def build_onnx_graph(self, array, input_arrays, to_node):

        input_names, output_name = self._build_graph_with_inputs_outputs(
            input_arrays.values(), array, to_node)

        g = onnx.GraphProto()

        for node_name in nx.ancestors(self._graph, output_name):
            node = self._graph.nodes[node_name]["node"]
            if isinstance(node, Input):
                g.input.append(onnx.helper.make_tensor_value_info(
                    node_name, numpy_to_onnx(np.dtype(node.dtype)), node.shape))
            # elif "output" in node:
            #     node = node["output"]
            #     g.output.append(onnx.helper.make_tensor_value_info(
            #         node_name, numpy_to_onnx(np.dtype(node.dtype)), node.shape))
            elif isinstance(node, Node):
                n = onnx.helper.make_node(node.op_type,
                                          [n._internal_name
                                           if n is not None else ""
                                           for n in node.inputs],
                                          [n._internal_name
                                           if n is not None else ""
                                           for n in node.outputs],
                                          name=node_name, **node.attributes)
                g.node.append(n)
            else:
                raise ValueError("")

        output_node = self._graph.nodes[output_name]["node"]
        g.output.append(
            onnx.helper.make_tensor_value_info(
                output_name, numpy_to_onnx(np.dtype(output_node.dtype)),
                output_node.shape))

        # clean up
        self._remove_graph_inputs_outputs(input_names, output_name)

        return g
