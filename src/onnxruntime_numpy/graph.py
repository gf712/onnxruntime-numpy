from .types import numpy_to_onnx
from . import array
from . import onnx_utils

import numpy as np
import networkx as nx
import onnx

from collections import namedtuple, Hashable
from typing import Tuple


class Node(namedtuple("Node", "inputs outputs op_type attributes")):
    def __repr__(self):
        return f'{self.op_type}'


class Input(namedtuple("Input", "dtype shape")):
    def __repr__(self):
        return f'Input({self.shape}, dtype={self.dtype})'


class Output(namedtuple("Output", "dtype shape")):
    def __repr__(self):
        return f'Output({self.shape}, dtype={self.dtype})'


class TensorProtoInternal(namedtuple("TensorProtoInternal", "values dtype shape name")):
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
        G.add_edge(Node("", (i.name,), "Input", HashableAttributes({}), i.name), node_obj)

    for node in onnx_graph.node:
        for output in node.output:
            node_obj = node_map[output]
            for input in node.input:
                G.add_edge(node_map[input], node_obj)

    return G


class Graph:
    def __init__(self):
        self._graph = nx.DiGraph()

    @classmethod
    def from_onnx(cls, onnx_graph, outputs=None):
        if len(onnx_graph.output) == 0 and outputs is None:
            raise ValueError(
                "ONNX graph does not have an output and none was specified")
        elif outputs is None:
            outputs = onnx_graph.output

        self._graph = build_graph_from_onnx(onnx_graph, outputs)
        if not self._graph.is_directed():
            raise ValueError("Graph is not a DAG")

    def copy(self):
        g = Graph()
        g._graph = self._graph.copy()
        return g

    def toposort(self):
        return nx.topological_sort(self._graph)

    def reversed_toposort(self):
        return reversed(list(self.toposort()))

    def add_node(self, op_type, inputs, outputs, **attributes):
        new_node = Node(
            inputs,
            outputs,
            op_type,
            HashableAttributes(**attributes))
        node_name = str(hash(new_node))
        self._graph.add_node(node_name, node=new_node)
        for i in inputs:
            # nodes can have empty inputs
            if i is not None:
                self._graph.add_edge(i._evaluator._parent_node, node_name)
        return node_name

    def add_initializer(self, name: str, dtype: np.dtype, dims: Tuple[int], vals):
        raise NotImplementedError()

    def add_input(self, array):
        new_node = Input(array.dtype, array.shape)
        self._graph.add_node(array._internal_name, input=new_node)
        return array._internal_name

    def add_output(self, array):
        self._graph.add_node(array._internal_name,
                             output=Output(array.dtype, array.shape))
        return array._internal_name

    def add_subgraph(self, other_graph):
        self._graph = nx.compose(self._graph, other_graph._graph)

    def build_onnx_graph(self):

        g = onnx.GraphProto()

        for node_name, node in self._graph.nodes(data=True):
            if len(node) != 1:
                raise ValueError("")
            if "input" in node:
                node = node["input"]
                g.input.append(onnx.helper.make_tensor_value_info(
                    node_name, numpy_to_onnx(np.dtype(node.dtype)), node.shape))
            elif "output" in node:
                node = node["output"]
                g.output.append(onnx.helper.make_tensor_value_info(
                    node_name, numpy_to_onnx(np.dtype(node.dtype)), node.shape))
            elif "node" in node:
                node = node["node"]
                n = onnx.helper.make_node(node.op_type,
                                          [n._internal_name if n is not None else "" for n in node.inputs],
                                          [n._internal_name if n is not None else "" for n in node.outputs],
                                          name=node_name,
                                          **node.attributes)
                g.node.append(n)
            else:
                raise ValueError("")

        return g
