from .array import Array
from .graph import Graph
from typing import List
import logging


class OpTracer:
    def __init__(self, *t_args: Array, **t_kwargs: Array):
        raise NotImplementedError("Needs fixing")
        self._original_tensors: List[Array] = [*t_args, *t_kwargs.values()]
        # self._lazy_tensors = set()
        # this is a "tracker array"
        # we just want to know what ops were added from here on
        # this means that the array has the same shape/dtype/name as the original array
        # but it has its own evaluator (that will start empty)
        self._op_t_args = []
        for tensor in t_args:
            # if tensor._ort_value is None:
            # self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = Array(
                dims=tensor.shape, dtype=tensor.dtype,
                internal_name=tensor._internal_name)
            self._op_t_args.append(mock_tensor)

        self._op_t_kwargs = {}
        for kw, tensor in t_kwargs.items():
            # if tensor._ort_value is None:
            #     self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = Array(
                dims=tensor.shape, dtype=tensor.dtype,
                internal_name=tensor._internal_name)
            self._op_t_kwargs[kw] = mock_tensor

        self._graph = None

    def _merge_graphs(self, mock_array: Array):
        old_graph_output_nodes = {t._internal_name: t._evaluator._parent_node
                                  for t in self._original_tensors
                                  if t._evaluator._parent_node is not None}

        # first attach the input graphs to the new evaluator's graph
        for t in self._original_tensors:
            mock_array._evaluator.merge(t._evaluator)

        # and then "merge" the graphs by adding edges between output nodes
        # of old graphs and the new graph
        inputs_to_remove = set()
        new_graph_input_edges = mock_array._evaluator._input_values
        for input_name, input_array in mock_array._evaluator._input_values.items():
            if input_name in old_graph_output_nodes:
                output_node = old_graph_output_nodes[input_name]
                input_node = new_graph_input_edges[input_name]._evaluator._parent_node

                mock_array._evaluator._graph._graph.add_edge(
                    output_node, input_node, index=0, array=input_array)
                inputs_to_remove.add(input_name)
                logging.debug(
                    f"adding edge between: {input_name} and "
                    f"{new_graph_input_edges[input_name]}")

        for i in inputs_to_remove:
            mock_array._evaluator._input_values.pop(i)

    def trace_function_call(self, func):
        if self._graph is not None:
            raise ValueError("Can only track a function call once")
        mock_array = func(*self._op_t_args, **self._op_t_kwargs)
        if not isinstance(mock_array, Array):
            raise NotImplementedError(
                "Tracing only possible with functions with a single output of type "
                "Array")
        self._graph = Graph()
        self._graph.add_subgraph(mock_array._evaluator._graph.copy())
        self._merge_graphs(mock_array)

        return mock_array


class OpTracerContext(object):
    def __init__(self, graph, *t_args, **t_kwargs):
        self._graph_to_write_to = graph
        self._tracer = OpTracer(*t_args, **t_kwargs)

    def __enter__(self):
        return self._tracer

    def __exit__(self, type, value, traceback):
        self._graph_to_write_to.add_subgraph(self._tracer._graph)
