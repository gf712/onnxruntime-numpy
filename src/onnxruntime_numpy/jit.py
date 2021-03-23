from . import array
from .evaluator import LazyEvaluator
from .types import onnx_to_numpy
from .onnx_utils import make_onnx_tensor_value_info

import onnxoptimizer
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
        self._lazy_tensors = set()
        # this is a "tracker array"
        # we just want to know what ops were added from here on
        # this means that the array has the same shape/dtype/name as the original array
        # but it has its own evaluator (that will start empty)
        self._op_t_args = []
        for tensor in t_args:
            if tensor._ort_value is None:
                self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = array.Array(
                dims=tensor.shape, dtype=tensor.dtype, internal_name=tensor._internal_name)
            self._op_t_args.append(mock_tensor)

        self._op_t_kwargs = {}
        for kw, tensor in t_kwargs.items():
            if tensor._ort_value is None:
                self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = array.Array(
                dims=tensor.shape, dtype=tensor.dtype, internal_name=tensor._internal_name)
            self._op_t_kwargs[kw] = mock_tensor

        self._graph = None

    def track_function_call(self, func):
        if self._graph is not None:
            raise ValueError("Can only track a function call once")
        mock_array = func(*self._op_t_args, **self._op_t_kwargs)
        if not isinstance(mock_array, array.Array):
            raise NotImplementedError(
                "Tracing only possible with functions with a single output of type array.Array")
        self._graph = onnx.GraphProto()
        self._graph.CopyFrom(mock_array._evaluator._graph)
        self._graph.output.append(
            make_onnx_tensor_value_info(mock_array)
        )
        inputs_ = list(self._graph.input)
        outputs_ = list(self._graph.output)
        nodes_ = list(self._graph.node)
        # the mock_array becomes the actual array
        # after merging all the input evaluators
        merge_array_evaluators(mock_array._evaluator, *self._original_evaluators)

        for idx, input in enumerate(mock_array._evaluator._graph.input):
            # not an actual input
            if input.name in self._lazy_tensors:
                mock_array._evaluator._graph.input.pop(idx)

        inputs = list(mock_array._evaluator._graph.input)
        outputs = list(mock_array._evaluator._graph.output)
        nodes = list(mock_array._evaluator._graph.node)

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

    def __eq__(self, other: "TypeShapeID") -> bool:
        if self._dtype != other._dtype:
            return False
        if self._shape != other._shape:
            return False
        return True


class GraphSignature:
    def __init__(self, inputs_type_shape_id):  # , outputs):
        self._inputs = inputs_type_shape_id
        # self._outputs = outputs
        # this type should be treated as immutable, so can compute the hash once
        self._hash = self._hash_function()

    def _hash_function(self):
        seed = 0
        for input in self._inputs:
            seed = hash_combine(seed, input)

        # for output in self._outputs:
        #     seed = hash_combine(seed, output)

        return seed

    def __repr__(self) -> str:
        return f"{[(t._shape, t._dtype) for t in self._inputs]}"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: "GraphSignature") -> bool:
        return hash(self) == hash(other)


class JitFunctionSignature:
    def __init__(self, function: Callable, graph_signature: GraphSignature):
        self._function = function
        self._graph_signature = graph_signature
        # this type should be treated as immutable, so can compute the hash once
        self._hash = self._hash_function()

    def _hash_function(self):
        seed = 0
        seed = hash_combine(seed, self._function)
        seed = hash_combine(seed, self._graph_signature)

        return seed

    def __repr__(self):
        return f"JitFunctionSignature({self._function}, {self._graph_signature}, {self._hash})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: "JitFunctionSignature") -> bool:
        return hash(self) == hash(other)


class OptimizedGraph:
    def __init__(self, graph: onnx.GraphProto, *normalized_tensors):
        m = onnx.helper.make_model(graph)
        optimized_graph = onnxoptimizer.optimize(
            m, passes=["fuse_matmul_add_bias_into_gemm"]).graph

        self._graph = optimized_graph
        if len(self._graph.input) != len(normalized_tensors):
            raise ValueError(
                f"OptimizedGraph expected {len(self._graph.input)} inputs, but got {len(normalized_tensors)}")
        self._tensor_input_mapping = [None] * len(normalized_tensors)
        self._tensor_node_mapping = [[] for _ in range(len(normalized_tensors))]

        self._outputs = set()
        self._call_count = 0

        # map normalized_tensors to graph inputs
        for idx, t in enumerate(normalized_tensors):
            for input_node_idx, input_node in enumerate(self._graph.input):
                if t._internal_name == input_node.name:
                    self._tensor_input_mapping[idx] = input_node_idx

        for node_idx, node in enumerate(self._graph.node):
            node_input_names = [n for n in node.input]
            for t_idx, t in enumerate(normalized_tensors):
                input_indices = [input_idx for input_idx,
                                 n in enumerate(node_input_names) if n == t._internal_name]
                for idx in input_indices:
                    node_input_idx_pair = (node_idx, idx)
                    self._tensor_node_mapping[t_idx].append(node_input_idx_pair)

        self._output_type_shape_info = []
        for output in self._graph.output:
            self._outputs.add(output.name)
            tensor_type = output.type.tensor_type
            dtype = onnx_to_numpy(tensor_type.elem_type)
            shape = [s.dim_value for s in tensor_type.shape.dim]
            self._output_type_shape_info.append(TypeShapeID(dtype, shape))

    def get_output_type_shape_info(self):
        return self._output_type_shape_info

    def build_graph_for_tensors(self, *tensors: array.Array):
        if len(tensors) != len(self._tensor_input_mapping):
            raise ValueError(
                f"This graph has {len(self._inputs)} inputs, but {len(tensors)} were provided")

        graph = onnx.GraphProto(
            name=f"Generated_graph_from_{self._graph.name}_{self._call_count}")
        graph.CopyFrom(self._graph)

        for t_idx, t in enumerate(tensors):
            input_idx = self._tensor_input_mapping[t_idx]
            graph.input[input_idx].name = t._internal_name

            input_node_indices = self._tensor_node_mapping[t_idx]
            for node_idx, node_input_idx in input_node_indices:
                graph.node[node_idx].input[node_input_idx] = t._internal_name

        old_outputs = {}
        for node in graph.node:
            for output in node.output:
                if output not in old_outputs:
                    old_outputs[output] = str(uuid.uuid4())

        for node in graph.node:
            for idx, input in enumerate(node.input):
                if input in old_outputs:
                    node.input[idx] = old_outputs[input]
            for idx, output in enumerate(node.output):
                if output in old_outputs:
                    node.output[idx] = old_outputs[output]

        for idx, o in enumerate(graph.output):
            if o.name in old_outputs:
                graph.output[idx].name = old_outputs[output]

        self._call_count += 1
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

    def normalize_function_inputs(self, *tensors, **tensors_map):
        normalized_input = [None] * len(self._fn_signature.parameters)
        parameters = self._fn_signature.parameters

        for arg_position, tensor in enumerate(tensors):
            normalized_input[arg_position] = tensor

        for kwarg_name, tensor in tensors_map.items():
            for idx, (k, v) in enumerate(parameters.items()):
                if kwarg_name == k:
                    normalized_input[idx] = tensor
                    break

        if any(map(lambda el: el is None, normalized_input)):
            raise ValueError("Not all tensors were mapped to function input.")

        return normalized_input


def generate_graph_flow(func, *array_objs, **array_obj_kwargs):
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

    return graph, fn_to_graph_input_map, result_array


global_function_signature_cache = {}
global_function_graph_cache = {}


def generate_cached_graph(func, *array_objs, **array_obj_kwargs):
    graph, fn_to_graph_input_map, result_array = generate_graph_flow(
        func, *array_objs, **array_obj_kwargs)
    global_function_signature_cache[func] = fn_to_graph_input_map

    normalized_tensors = fn_to_graph_input_map.normalize_function_inputs(
        *array_objs, **array_obj_kwargs)
    inputs_type_shape_id = [TypeShapeID(t.dtype, t.shape) for t in normalized_tensors]

    # TODO: add graph optimisation before caching graph
    optimized_graph = OptimizedGraph(graph, *normalized_tensors)
    graph_signature = GraphSignature(inputs_type_shape_id)
    signature = JitFunctionSignature(func, graph_signature)

    global_function_graph_cache[signature] = optimized_graph
    return result_array


def jit(func):  # Callable[[IterableType[array.Array]], array.Array]
    def wrapper_jit(*array_objs, **array_obj_kwargs):  # IterableType[array.Array]
        # only Array objects are supported
        if any(map(lambda a: not isinstance(a, array.Array), array_objs)) or \
           any(map(lambda a: not isinstance(a, array.Array), array_obj_kwargs.values())):
            raise TypeError("Jit is currently only support with Array objects")

        if func not in global_function_signature_cache:
            result_array = generate_cached_graph(func, *array_objs, **array_obj_kwargs)
        else:
            fn_to_graph_input_map = global_function_signature_cache[func]
            normalized_tensors = fn_to_graph_input_map.normalize_function_inputs(
                *array_objs, **array_obj_kwargs)
            inputs_type_shape_id = [TypeShapeID(t.dtype, t.shape)
                                    for t in normalized_tensors]
            signature = JitFunctionSignature(func, GraphSignature(inputs_type_shape_id))

            if signature in global_function_graph_cache:
                cached_graph = global_function_graph_cache[signature]

                graph = cached_graph.build_graph_for_tensors(*normalized_tensors)

                input_evaluators = [t._evaluator for t in array_objs]
                input_evaluators += [t._evaluator for k, t in array_obj_kwargs.items()]

                if len(graph.output) != 1:
                    raise NotImplementedError(
                        "Currently JIT only supports single output graphs")
                else:
                    graph_output_name = graph.output[0].name
                    # we can remove output from the graph
                    # since the array will track it
                    graph.output.pop(0)

                    # get output dtype and shape
                    type_shape_info = cached_graph.get_output_type_shape_info()[0]
                    output_shape = tuple(type_shape_info._shape)
                    output_dtype = type_shape_info._dtype

                lazy_tensors = set(t._internal_name for t in normalized_tensors if t._ort_value is None)

                for idx, input in enumerate(graph.input):
                    # not an actual input
                    if input.name in lazy_tensors:
                        graph.input.pop(idx)

                evaluator = LazyEvaluator()
                evaluator.add_subgraph(graph)

                result_array = array.Array(
                    dims=output_shape, dtype=output_dtype, internal_name=graph_output_name, evaluator=evaluator)
                merge_array_evaluators(result_array._evaluator, *input_evaluators)

            else:
                # this is a cache miss because the function now takes a new parameter dtype/shape combination
                # so we generate a new cached graph for this unseen combination
                result_array = generate_cached_graph(
                    func, *array_objs, **array_obj_kwargs)

        return result_array

    return wrapper_jit
