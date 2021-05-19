from .types import numpy_to_onnx
from . import array
from . import evaluator
from . import onnx_utils
from .exceptions import InternalException

import numpy as np
import networkx as nx
import onnx
import uuid

from collections import namedtuple, defaultdict
from collections.abc import Hashable
from typing import Tuple, Set, Dict, Optional, List, DefaultDict


class Node(namedtuple("Node", "op_type node_name op_name attributes")):
    def __repr__(self):
        return f'Node({self.node_name})'


class Input(namedtuple("Input", "dtype shape node_name")):
    def __repr__(self):
        return f'Input({self.shape}, dtype={self.dtype}, name={self.node_name})'


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
            return onnx_utils.make_onnx_tensor(value._internal_array._internal_name, value)
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
        node_map[o] = Node((o,), "", "Output",
                           HashableAttributes({}), o)
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
                "Input",
                HashableAttributes({}),
                i.name),
            node_obj)

    for node in onnx_graph.node:
        for output in node.output:
            node_obj = node_map[output]
            for input in node.input:
                G.add_edge(node_map[input], node_obj)

    return G


EdgeType = Set[Tuple[Optional[str], str, int]]
EdgeTableType = DefaultDict[str, EdgeType]


class Graph:
    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._input_node_names: Set[str] = set()
        self._edge_lookup_table: EdgeTableType = defaultdict(set)

    def copy(self):
        g = Graph()
        g._graph = self._graph.copy()
        g._input_node_names = self._input_node_names.copy()
        g._edge_lookup_table = self._edge_lookup_table
        return g

    @property
    def nodes(self):
        return self._graph.nodes

    def toposort(self):
        return nx.topological_sort(self._graph)

    def reversed_toposort(self):
        return reversed(list(self.toposort()))

    def add_edge(
            self, parent_node: Optional[str],
            node_name: str, in_index: Optional[int],
            out_index: int, name: str, required: bool):
        key: int = self._graph.add_edge(
            parent_node, node_name, in_index=in_index,
            out_index=out_index, name=name, required=required)
        e = self._graph.edges[
            (parent_node, node_name, key)]
        self._edge_lookup_table[e["name"]].add((parent_node, node_name, key))

    def add_node(
            self, op_name: str,
            inputs: Tuple[Optional["array.Array"], ...],
            outputs: Tuple["array.Array", ...],
            **attributes):
        op_id = uuid.uuid4()
        node_name = f"{op_name}_{op_id}"

        new_node = Node(
            None,  # op_type is added by register decorator
            node_name,
            op_name,
            HashableAttributes(**attributes))
        self._graph.add_node(node_name, node=new_node)

        for idx, input_array in enumerate(inputs):
            # nodes can have empty inputs
            if input_array is not None:
                inputs_output_node = input_array._internal_array._evaluator._output_node
                if inputs_output_node is None:
                    raise InternalException(
                        "Array must have an output node")
                inputs_output_edge = list(self._graph.in_edges(
                    inputs_output_node, data=True))
                if len(inputs_output_edge) != 1:
                    raise InternalException(
                        "Output node must have a single edge")
                inputs_output_index = inputs_output_edge[0][-1]["in_index"]
                connection_name = inputs_output_edge[0][-1]["name"]
                parent_node = input_array._internal_array._evaluator._parent_node
                if parent_node is None:
                    raise InternalException("Array must have a parent node")
                self.add_edge(
                    parent_node, node_name, in_index=inputs_output_index,
                    out_index=idx, name=connection_name, required=True)
            else:
                self.add_edge(
                    None, node_name, in_index=None, out_index=idx,
                    name="EmptyPlaceholder", required=True)

        output_node_names = []

        for output_array_idx, output in enumerate(outputs):
            output_id = uuid.uuid4()
            output_node_name = f"Output_{output_id}"
            output_node = Output(output.dtype, output.shape)
            self._graph.add_node(output_node_name, node=output_node)

            edge_name = f"Tensor_{output._internal_array._internal_name}"

            self.add_edge(
                node_name, output_node_name, in_index=output_array_idx,
                out_index=0, name=edge_name, required=True)

            output_node_names.append(output_node_name)

        return node_name, output_node_names

    def add_initializer(
            self, name: str, dtype: np.dtype, dims: Tuple[int],
            vals):
        raise NotImplementedError()

    def add_input(self, array: "array.Array"):
        input_id = uuid.uuid4()
        input_node_name = f"Input_{input_id}"
        input_node = Input(array.dtype, array.shape, input_node_name)
        self._graph.add_node(input_node_name, node=input_node)
        self._input_node_names.add(input_node_name)

        output_id = uuid.uuid4()
        output_node_name = f"Output_{output_id}"
        output_node = Output(array.dtype, array.shape)
        self._graph.add_node(output_node_name, node=output_node)

        edge_name = f"Tensor_{array._internal_array._internal_name}"

        self.add_edge(
            input_node_name, output_node_name, in_index=0,
            out_index=0, name=edge_name, required=False)

        return input_node_name, output_node_name

    def add_output(self, array, from_node, idx):
        output_node = Output(array.dtype, array.shape)
        self._graph.add_node(array._internal_name,
                             node=output_node)
        self.add_edge(
            from_node, array._internal_name, in_index=idx,
            out_index=0, name=array._internal_name, required=False)
        return array._internal_name

    def add_subgraph(self, other_graph):
        if self._graph is None:
            raise InternalException()
        elif other_graph._graph is not None:
            self._graph.add_nodes_from(other_graph._graph.nodes(data=True))
            self._graph.add_edges_from(other_graph._graph.edges(
                data=True, keys=True))

            self._input_node_names.update(other_graph._input_node_names)
            other_graph._input_node_names = self._input_node_names
            for k, v in other_graph._edge_lookup_table.items():
                self._edge_lookup_table[k].update(v)


class ExecutableGraph:
    def __init__(
            self, graph: Graph,
            node_inputs: Dict[str, Tuple["array.Array"]],
            node_outputs: Dict[str, Tuple["array.Array"]],
            output_names: List[str],
            cached_results: "evaluator.IntermediateResultCache"):
        self._output_node_info: Dict[str, Tuple[np.dtype, List[int]]] = {}
        self._input_node_mapping: Dict[str, str] = {}

        for output_node_name in output_names:
            output_edge = list(graph._graph.in_edges(
                output_node_name, data=True))

            if len(output_edge) != 1:
                raise InternalException("Output node must have one edge")

            output_node_parent = output_edge[0][0]
            output_array = node_outputs[output_node_parent][
                output_edge[0][-1]["in_index"]][1]

            self._output_node_info[output_node_name] = (
                numpy_to_onnx(np.dtype(output_array.dtype)),
                output_array.shape.tolist())

        for input_name, _ in node_inputs.items():
            if input_name in graph._input_node_names:
                input_out_edge = graph._graph.out_edges(
                    input_name, data=True)
                if len(input_out_edge) == 0:
                    # unused input node, should this be an error?
                    continue
                for e in input_out_edge:
                    edge_name = e[-1]["name"]
                    self._input_node_mapping[input_name] = edge_name

        if len(self._output_node_info) != 1:
            raise NotImplementedError("Can only handle a single output value")

        # FIXME: make this independent of the number of outputs
        output_name = next(iter(self._output_node_info.keys()))

        core_graph = graph._graph

        if not cached_results.empty():
            edges_to_remove: Set[Tuple[Optional[str], str, int]] = set()
            core_graph = core_graph.copy()
            for tensor_name, result in cached_results.get_all_cache_tensor_mappings().items():  # noqa
                cached_edge_names = graph._edge_lookup_table.get(tensor_name)
                if cached_edge_names is None:
                    continue
                for name in cached_edge_names:
                    in_node_name = name[0]
                    feed_node_name = name[1]
                    in_node = graph.nodes[in_node_name]["node"]
                    feed_node = graph.nodes[feed_node_name]["node"]
                    # this node has to stay, because it is needed to compute the
                    # output of a node
                    node_to_output_edge = isinstance(
                        in_node, Node) and isinstance(
                        feed_node, Output)
                    required = core_graph.edges[name]["required"]
                    if node_to_output_edge and required:
                        continue
                    node_to_node_edge = isinstance(
                        feed_node, Node) and isinstance(
                        in_node, Node)
                    if not node_to_node_edge:
                        continue
                    edges_to_remove.add(name)
                    input_id = uuid.uuid4()
                    input_node_name = f"Input_{input_id}"
                    input_node = Input(
                        result.dtype, result.shape, input_node_name)
                    core_graph.add_node(input_node_name, node=input_node)
                    graph._input_node_names.add(input_node_name)
                    self._input_node_mapping[input_node_name] = tensor_name
                    out_index = core_graph.edges[name]["out_index"]
                    tensor_name = core_graph.edges[name]["name"]
                    out_index = core_graph.edges[name]["out_index"]
                    core_graph.add_edge(
                        input_node_name, feed_node_name, in_index=0,
                        out_index=out_index, name=tensor_name, required=True)

            for edge_to_remove in edges_to_remove:
                core_graph.remove_edge(
                    *edge_to_remove[: 2],
                    key=edge_to_remove[2])

        self._ancestors = self._prune_graph(core_graph, output_name)

        if len(self._ancestors) == 0:
            raise InternalException("Could not find ancestor nodes")

    def _prune_graph(self, graph, output_name: str):
        nodes_of_interest = nx.ancestors(graph, output_name)
        nodes_of_interest -= set((None,))
        nodes_of_interest |= set((output_name,))

        self._input_node_mapping = {
            k: v for k, v in self._input_node_mapping.items()
            if k in nodes_of_interest}

        # need to add required outputs
        nodes_of_interest_to_add: Set[str] = set()
        for node_name in nodes_of_interest:
            node = graph.nodes[node_name]["node"]
            if isinstance(node, Input) or isinstance(node, Output):
                continue
            out_edges = list(graph.out_edges(node_name, data=True))
            for out_edge in out_edges:
                required = out_edge[-1]["required"]
                feed_node_name = out_edge[1]
                feed_node = graph.nodes[feed_node_name]["node"]
                if (required and feed_node not in nodes_of_interest
                        and isinstance(feed_node, Output)):
                    nodes_of_interest_to_add.add(feed_node_name)
        nodes_of_interest |= nodes_of_interest_to_add
        return graph.subgraph(nodes_of_interest)

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
        core_graph = self._ancestors

        # keep track of input names in order to avoid duplicate inputs
        graph_input_names: Set[str] = set()

        for node_name in core_graph:
            if node_name is None:
                continue

            if node_name in self._output_node_info:
                continue

            node = core_graph.nodes[node_name]["node"]

            if isinstance(node, Input):
                out_edges = core_graph.out_edges(node_name, data=True)
                if len(out_edges) == 0:
                    raise InternalException(len(out_edges))
                for e in out_edges:
                    out_node = e[1]
                    tensor_name = e[-1]["name"]
                    if (out_node in self._ancestors
                            and tensor_name not in graph_input_names):
                        g.input.append(
                            onnx.helper.make_tensor_value_info(
                                tensor_name,
                                numpy_to_onnx(np.dtype(node.dtype)),
                                node.shape.tolist()))
                        graph_input_names.add(tensor_name)
            elif isinstance(node, Node):
                in_edges = core_graph.in_edges(node_name, data=True)
                out_edges = core_graph.out_edges(node_name, data=True)

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
            elif isinstance(node, Output):
                continue
            else:  # pragma: no cover
                raise InternalException(f"Unhandled node type {type(node)}")

        for output_node_name, output_info in self._output_node_info.items():
            # already checked that there is only a single input in the output node
            output_edge = list(core_graph.in_edges(
                output_node_name, data=True))[0]

            g.output.append(
                onnx.helper.make_tensor_value_info(
                    output_edge[-1]["name"], *output_info))

        return g


def compile_graph(
        graph: Graph, node_inputs: Dict[str, Tuple["array.Array"]],
        node_outputs: Dict[str, Tuple["array.Array"]],
        output_names: List[str],
        cached_results: "evaluator.IntermediateResultCache") -> ExecutableGraph:
    return ExecutableGraph(graph, node_inputs, node_outputs, output_names,
                           cached_results)
