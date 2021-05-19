# TODO: fix types and formatting after refactor
# flake: noqa
# type: ignore
from .array import Array
from .graph import Graph
from .evaluator import merge_array_evaluators
from typing import List
import logging


class OpTracerImpl():
    def __init__(self):
        self._graph = None

    def trace(self, function, *args, **kwargs):
        # self._lazy_tensors = set()
        # this is a "tracker array"
        # we just want to know what ops were added from here on
        # this means that the array has the same shape/dtype/name as the original array
        # but it has its own evaluator (that will start empty)
        op_t_args = []
        for tensor in args:
            # if tensor._ort_value is None:
            # self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = Array(
                dims=tensor.shape, dtype=tensor.dtype,
                internal_name=tensor._internal_name)
            op_t_args.append(mock_tensor)

        op_t_kwargs = {}
        for kw, tensor in kwargs.items():
            # if tensor._ort_value is None:
            #     self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = Array(
                dims=tensor.shape, dtype=tensor.dtype,
                internal_name=tensor._internal_name)
            op_t_kwargs[kw] = mock_tensor

        results = function(*op_t_args, **op_t_kwargs)
        
        if not isinstance(results, tuple):
            results = (results,)

        for result in results:
            g = result._evaluator._graph

            mapping = {}
            evaluators = []
            for arg, mock_arg in zip(args, op_t_args):
                evaluators.append(mock_arg._evaluator)
                mapping[evaluators[-1]._parent_node] = arg._evaluator._parent_node

            for k, v in kwargs.items():
                evaluators.append(op_t_kwargs[k]._evaluator._parent_node)
                mapping[evaluators[-1]._parent_node] = v._evaluator._parent_node

            merge_array_evaluators(result._evaluator, *evaluators)

        return result

    def get_graph(self):
        return self._graph

    def get_compiled_graph(self):
        raise NotImplementedError("")


class OpTracer:
    """
        ```python
        with OpTracer() as tracer:
            expected = tracer.trace(add, x, y)
            g = tracer.get_graph()
            f = tracer.get_compiled_graph()
            result = f(x, y)
            assert (result.numpy() == expected.numpy()).all()
        ```    
    """

    def __init__(self):
        self._tracer = OpTracerImpl()

    def __enter__(self):
        return self._tracer

    def __exit__(self, type, value, traceback):
        self._graph_to_write_to.add_subgraph(self._tracer._graph)
