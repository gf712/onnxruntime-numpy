from . import array
from .evaluator import LazyEvaluator
from .types import onnx_to_numpy

import onnx
import numpy as np

from typing import Callable, NoReturn, Hashable, Tuple, Dict
from typing import Iterable as IterableType
from collections.abc import Iterable
from collections import defaultdict
import functools
import inspect
import uuid

# @functools.lru_cache


def make_graph_buffer(nodes, inputs, outputs, initializers):
    return onnx.helper.make_graph(nodes, "graph", inputs, outputs, initializers)


def merge_array_evaluators(evaluator_to_merge_into: LazyEvaluator, *evaluators):
    for evaluator in evaluators:
        evaluator_to_merge_into.merge(evaluator)


class OpTracker:
    def __init__(self, *t_args, **t_kwargs):
        self._original_evaluators = [a._evaluator for a in t_args]
        self._original_evaluators += [a._evaluator for a in t_kwargs.values()]
        # this is a "tracker array"
        # we just want to know what ops were added from here on
        self._op_t_args = []
        for tensor in t_args:
            mock_tensor = array.Array(
                dims=tensor.shape, dtype=tensor.dtype, internal_name=tensor._internal_name)
            self._op_t_args.append(mock_tensor)

        self._op_t_kwargs = {}
        for kw, tensor in t_kwargs.items():
            mock_tensor = array.Array(
                dims=tensor.shape, dtype=tensor.dtype, internal_name=tensor._internal_name)
            self._op_t_kwargs[kw] = mock_tensor

        self._graph = None

    def track_function_call(self, func):
        if self._graph is not None:
            raise ValueError("Can only track a function call once")
        mock_array = func(*self._op_t_args, **self._op_t_kwargs)
        self._graph = onnx.GraphProto()
        self._graph.CopyFrom(mock_array._evaluator._graph)
        # the mock_array becomes the actual array
        # after merging all the input evaluators
        merge_array_evaluators(mock_array._evaluator, *self._original_evaluators)

        return mock_array


class OpTrackerContext(object):
    def __init__(self, graph, *t_args, **t_kwargs):
        self._graph_to_write_to = graph
        self._tracker = OpTracker(*t_args, **t_kwargs)

    def __enter__(self):
        return self._tracker

    def __exit__(self, type, value, traceback):
        self._graph_to_write_to.MergeFrom(self._tracker._graph)


def hash_combine(seed: int, value: Hashable):
    return seed ^ hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2)


class TypeShapeID:
    def __init__(self, dtype: np.dtype, shape: Tuple[int]):
        self._dtype = dtype
        self._shape = shape

    def __hash__(self) -> int:
        seed = 0
        seed = hash_combine(seed, self._dtype)
        seed = hash_combine(seed, self._shape)
        return seed


class OptimizedGraph:
    def __init__(self, graph: onnx.GraphProto):
        self._graph = graph
        self._inputs = {}
        for input in self._graph.input:
            tensor_type_proto = input.type.tensor_type
            tensor_shape_proto = tensor_type_proto.shape
            dtype = onnx_to_numpy(tensor_type_proto.elem_type)
            shape = tuple(s.dim_value for s in tensor_type_proto.shape.dim)
            self._inputs[input.name] = TypeShapeID(dtype, shape)

        self._nodes_connect_to_input = defaultdict(list)
        
        for node in self._graph.node:
            inputs = [n.name for n in node.input]
            for input_name in self._input.keys():
                input_indices = [input_idx for input_idx, n in enumerate(inputs) if n == input_name]
                for idx in input_indices:
                    self._nodes_connect_to_input[inputs].append((node, idx))

    def input_names(self):
        return self._inputs.keys()

    def input_to_nodes_map(self):
        return self._nodes_connect_to_input

    def build_graph_for_tensors(self, tensors_map: Dict[str, "array.Array"]):
        if len(tensors_map) != len(self._inputs):
            raise ValueError(
                f"This graph has {len(self._inputs)} inputs, but {len(tensors_map)} were provided")
        new_input_names = {}
        for key_binding, t in tensors_map.items():
            if key_binding in self._inputs:
                type_shape_id = TypeShapeID(t.dtype, t.shape)
                if type_shape_id == self._inputs[key_binding]:
                    # build the map to update the graph
                    new_input_names[self._inputs[type_shape_id]] = t._internal_name
            else:
                # if one of the input types/shape does not match
                # we can't reuse this graph
                return None
            
        graph = onnx.GraphProto()
        graph.CopyFrom(self._graph)
        for input in graph.input:
            input.name = new_input_names[input.name]
        return graph


class MapFnArgsToInput:
    def __init__(self, fn_signature):
        self._fn_signature = fn_signature
        self._map_fn_parameters_to_graph_inputs = {}

    def get_arg_name_at_pos(self, arg_pos):
        parameters = list(self._fn_signature.parameters)
        return parameters[arg_pos]

    def bind_arg(self, arg_position, input_name):
        arg_name = self.get_arg_name_at_pos(arg_position)
        self._map_fn_parameters_to_graph_inputs[arg_name] = input_name

    def bind_kwarg(self, kwarg_name, input_name):
        self._map_fn_parameters_to_graph_inputs[kwarg_name] = input_name

    def adapt_graph_to_tensor_inputs(self, graph: OptimizedGraph, *tensors, **tensors_map):
        tensor_map = {}
        for arg_position, tensor in enumerate(tensors):
            arg_name = self.get_arg_name_at_pos(arg_position)
            original_tensor_name = self._map_fn_parameters_to_graph_inputs[arg_name]
            for input_node_name in graph.input_names():
                if input_node_name == original_tensor_name:
                    tensor_map[input_node_name] = tensor
                    break
                # nodes_and_input_index = input_to_nodes_map[original_tensor_name]
                # for (node, input_index) in nodes_and_input_index:
                #     for original_node in graph._graph.node:
                #         if node == original_node:

            
            # for node in graph.node:
            #     if original_tensor_name in 
            #     for node_input in node.input:
            #         if original_tensor_name == node_input:
            #             input_idx = list(node.input).index(original_tensor_name)
            #             node.input[input_idx] = tensor._internal_name

        for kwarg_name, tensor in tensors_map.items():
            original_tensor_name = self._map_fn_parameters_to_graph_inputs[kwarg_name]
            for input_node_name in graph.input_names():
                if input_node_name == original_tensor_name:
                    tensor_map[input_node_name] = tensor
                    break
            # for node in graph.node:
            #     for node_input in node.input:
            #         if original_tensor_name == node_input:
            #             input_idx = list(node.input).index(original_tensor_name)
            #             node.input[input_idx] = tensor._internal_name
        return graph.build_graph_for_tensors(tensor_map)

    # def build_input_map(self, *tensors, **tensors_map):
    #     inputs = {}
    #     for arg_position, tensor in enumerate(tensors):
    #         arg_name = self.get_arg_name_at_pos(arg_position)
    #         inputs[self._map_fn_parameters_to_graph_inputs[arg_name]] = tensor

    #     for kwarg_name, tensor in tensors_map.items():
    #         inputs[self._map_fn_parameters_to_graph_inputs[kwarg_name]] = tensor

    #     return inputs


global_function_graph_cache = {}


def jit(func):  # Callable[[IterableType[array.Array]], array.Array]
    def wrapper_jit(*array_objs, **array_obj_kwargs):  # IterableType[array.Array]
        # only Array objects are supported
        if any(map(lambda a: not isinstance(a, array.Array), array_objs)) or \
           any(map(lambda a: not isinstance(a, array.Array), array_obj_kwargs.values())):
            raise TypeError("Jit is currently only support with Array objects")

        if func not in global_function_graph_cache:

            graph = onnx.GraphProto(name=f"JIT_graph_{uuid.uuid4()}")

            with OpTrackerContext(graph, *array_objs, **array_obj_kwargs) as tracker:
                result_array = tracker.track_function_call(func)

            if isinstance(result_array, tuple) or result_array is None:
                raise ValueError(
                    "Jit only supports functions with a single return array")
            if not isinstance(result_array, array.Array):
                raise TypeError("Jit only supports Array object as a return value")

            fn_to_graph_input_map = MapFnArgsToInput(inspect.signature(func))
            for idx, tensor in enumerate(array_objs):
                fn_to_graph_input_map.bind_arg(idx, tensor._internal_name)

            for argname, tensor in array_obj_kwargs.items():
                fn_to_graph_input_map.bind_kwarg(argname, tensor._internal_name)

            # TODO: add graph optimisation before caching graph

            global_function_graph_cache[func] = (OptimizedGraph(graph), fn_to_graph_input_map,
                                                 result_array.shape, result_array.dtype)
        else:
            graph, fn_to_graph_input_map, output_shape, output_dtype = global_function_graph_cache[
                func]

            fn_to_graph_input_map.adapt_graph_to_tensor_inputs(
                graph, *array_objs, **array_obj_kwargs
            )
            input_evaluators = [t._evaluator for t in array_objs]
            input_evaluators += [t._evaluator for k, t in array_obj_kwargs.items()]

            result_array = array.Array(dims=output_shape, dtype=output_dtype)
            result_array._evaluator.add_subgraph(graph)
            merge_array_evaluators(result_array._evaluator, *input_evaluators)
        return result_array

    return wrapper_jit
