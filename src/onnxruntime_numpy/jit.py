# FIXME: re-enable flake8 after refactoring this file
# flake8: noqa
from . import array
from .evaluator import LazyEvaluator
from .types import onnx_to_numpy
from .onnx_utils import make_onnx_tensor_value_info
from .graph import Graph, Node, Input, Output

import onnxoptimizer
import onnx
import numpy as np

from typing import Callable, NoReturn, Hashable, Tuple, Dict
from typing import Iterable as IterableType
from collections.abc import Iterable
from collections import defaultdict, namedtuple
import functools
import inspect
import uuid

# @functools.lru_cache


def make_graph_buffer(nodes, inputs, outputs, initializers):
    return onnx.helper.make_graph(nodes, "graph", inputs, outputs, initializers)


def merge_array_evaluators(evaluator_to_merge_into: LazyEvaluator, *evaluators):
    for evaluator in evaluators:
        evaluator_to_merge_into.merge(evaluator)


class TypeShapeID(namedtuple("TypeShapeID", "dtype shape")):
    def __repr__(self):
        return f'TypeShapeID({self.shape}, dtype={self.dtype}'


class GraphSignature(namedtuple("GraphSignature", "inputs_type_shape_id")):
    def __repr__(self) -> str:
        return f"{[(t._shape, t._dtype) for t in self.inputs_type_shape_id]}"


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
                dims=tensor.shape, dtype=tensor.dtype,
                internal_name=tensor._internal_name)
            self._op_t_args.append(mock_tensor)

        self._op_t_kwargs = {}
        for kw, tensor in t_kwargs.items():
            if tensor._ort_value is None:
                self._lazy_tensors.add(tensor._internal_name)
            mock_tensor = array.Array(
                dims=tensor.shape, dtype=tensor.dtype,
                internal_name=tensor._internal_name)
            self._op_t_kwargs[kw] = mock_tensor

        self._graph = None

    def track_function_call(self, func):
        if self._graph is not None:
            raise ValueError("Can only track a function call once")
        mock_array = func(*self._op_t_args, **self._op_t_kwargs)
        if not isinstance(mock_array, array.Array):
            raise NotImplementedError(
                "Tracing only possible with functions with a single output of type array.Array")
        self._graph = Graph()
        self._graph.add_subgraph(mock_array._evaluator._graph)
        self._graph.add_output(mock_array)
        # inputs_ = list(self._graph.input)
        # outputs_ = list(self._graph.output)
        # nodes_ = list(self._graph.node)
        # the mock_array becomes the actual array
        # after merging all the input evaluators
        merge_array_evaluators(mock_array._evaluator,
                               *self._original_evaluators)

        to_remove = set()

        for idx, input in enumerate(mock_array._evaluator._graph.inputs):
            # not an actual input
            if input.name in self._lazy_tensors:
                to_remove.add(input)

        mock_array._evaluator._graph.remove_inputs(to_remove)

        # inputs = list(mock_array._evaluator._graph.input)
        # outputs = list(mock_array._evaluator._graph.output)
        # nodes = list(mock_array._evaluator._graph.node)

        return mock_array


class OpTrackerContext(object):
    def __init__(self, graph, *t_args, **t_kwargs):
        self._graph_to_write_to = graph
        self._tracker = OpTracker(*t_args, **t_kwargs)

    def __enter__(self):
        return self._tracker

    def __exit__(self, type, value, traceback):
        self._graph_to_write_to.add_subgraph(self._tracker._graph)


class JitFunctionSignature(
    namedtuple(
        "JitFunctionSignature",
        "function graph_signature")):
    def __repr__(self):
        return f"JitFunctionSignature({self.function}, {self.graph_signature})"


class OptimizedGraph:
    def __init__(self, graph: Graph, *normalized_tensors):
        original_onnx_graph = graph.build_onnx_graph()
        m = onnx.helper.make_model(original_onnx_graph)
        optimized_graph = onnxoptimizer.optimize(
            m, passes=["fuse_matmul_add_bias_into_gemm"]).graph
        optimized_graph = Graph.from_onnx(optimized_graph)
        self._graph = optimized_graph
        if len(self._graph.inputs) != len(normalized_tensors):
            raise ValueError(
                f"OptimizedGraph expected {len(self._graph.input)} inputs, but got {len(normalized_tensors)}")
        self._tensor_input_mapping = [None] * len(normalized_tensors)
        self._tensor_node_mapping = [[] for _ in range(len(normalized_tensors))]

        self._outputs = set()
        self._call_count = 0

        # map normalized_tensors to graph inputs
        for idx, t in enumerate(normalized_tensors):
            for input_node_idx, input_node in enumerate(self._graph.inputs):
                if t._internal_name == input_node.name:
                    # dtype shape name
                    self._tensor_input_mapping[idx] = Input(
                        t.dtype, t.shape, t._internal_name)

        for node_idx, node in enumerate(self._graph._graph.nodes):
            # FIXME maybe
            if not isinstance(node, Node):
                continue
            node_input_names = node.inputs
            for t_idx, t in enumerate(normalized_tensors):
                input_indices = [input_idx for input_idx,
                                 n in enumerate(node_input_names)
                                 if n == t._internal_name]
                for idx in input_indices:
                    self._tensor_node_mapping[t_idx].append((node, idx))

        self._output_type_shape_info = []
        for output in self._graph.outputs:
            self._outputs.add(output.name)
            dtype = output.dtype
            shape = output.shape
            self._output_type_shape_info.append(TypeShapeID(dtype, shape))

    def get_output_type_shape_info(self):
        return self._output_type_shape_info

    def build_graph_for_tensors(self, *tensors: array.Array):
        if len(tensors) != len(self._tensor_input_mapping):
            raise ValueError(
                f"This graph has {len(self._inputs)} inputs, but {len(tensors)} were provided")

        graph = Graph()
        graph.add_subgraph(self._graph)

        for t_idx, t in enumerate(tensors):
            input_node = self._tensor_input_mapping[t_idx]
            new_input_node = Input(
                input_node.dtype, input_node.shape, t._internal_name)
            graph.replace_input(input_node, new_input_node)

            input_node_indices = self._tensor_node_mapping[t_idx]
            for node, node_input_idx in input_node_indices:
                # inputs outputs op_type attributes name
                new_inputs = (*node.inputs[:node_input_idx],
                              t._internal_name, *node.inputs[node_input_idx+1:])
                new_node = Node(new_inputs, node.outputs, node.op_type,
                                node.attributes, node.name)
                graph.replace_node(node, new_node)
                # graph.node[node_idx].input[node_input_idx] = t._internal_name

        old_outputs = {}
        for node in graph.nodes:
            if isinstance(node, Node):
                for output in node.outputs:
                    if output not in old_outputs:
                        old_outputs[output] = str(uuid.uuid4())

        nodes = []
        for node in graph.nodes:
            if isinstance(node, Node):
                new_inputs = []
                new_outputs = []
                for idx, input in enumerate(node.inputs):
                    if input in old_outputs:
                        new_inputs.append(old_outputs[input])
                    else:
                        new_inputs.append(input)
                for idx, output in enumerate(node.outputs):
                    if output in old_outputs:
                        new_outputs.append(old_outputs[output])
                    else:
                        new_outputs.append(output)

                nodes.append((node, Node(tuple(new_inputs), tuple(
                    new_outputs), node.op_type, node.attributes, node.name)))

        graph.replace_nodes(nodes)

        new_outputs = []
        for idx, o in enumerate(graph.outputs):
            if o.name in old_outputs:
                new_outputs.append((o, old_outputs[output]))
            else:
                new_outputs.append((o, o))
        graph.replace_outputs(new_outputs)

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
    graph = Graph()

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
    inputs_type_shape_id = tuple(TypeShapeID(t.dtype, t.shape)
                                 for t in normalized_tensors)

    graph.add_output(result_array)
    optimized_graph = OptimizedGraph(graph, *normalized_tensors)
    graph_signature = GraphSignature(inputs_type_shape_id)
    signature = JitFunctionSignature(func, graph_signature)

    global_function_graph_cache[signature] = optimized_graph
    return result_array


def jit(func):  # Callable[[IterableType[array.Array]], array.Array]
    # IterableType[array.Array]
    def wrapper_jit(*array_objs, **array_obj_kwargs):
        # only Array objects are supported
        if any(
            map(lambda a: not isinstance(a, array.Array),
                array_objs)) or any(
            map(
                lambda a: not isinstance(a, array.Array),
                array_obj_kwargs.values())):
            raise TypeError("Jit is currently only support with Array objects")

        if func not in global_function_signature_cache:
            result_array = generate_cached_graph(
                func, *array_objs, **array_obj_kwargs)
        else:
            fn_to_graph_input_map = global_function_signature_cache[func]
            normalized_tensors = fn_to_graph_input_map.normalize_function_inputs(
                *array_objs, **array_obj_kwargs)
            inputs_type_shape_id = tuple(TypeShapeID(t.dtype, t.shape)
                                         for t in normalized_tensors)
            signature = JitFunctionSignature(
                func, GraphSignature(inputs_type_shape_id))

            if signature in global_function_graph_cache:
                cached_graph = global_function_graph_cache[signature]

                graph = cached_graph.build_graph_for_tensors(
                    *normalized_tensors)

                input_evaluators = [t._evaluator for t in array_objs]
                input_evaluators += [t._evaluator for k,
                                     t in array_obj_kwargs.items()]

                if len(graph.outputs) != 1:
                    raise NotImplementedError(
                        "Currently JIT only supports single output graphs")
                else:
                    graph_output_name = next(iter(graph.outputs))
                    # we can remove output from the graph
                    # since the array will track it
                    graph.outputs.pop()

                    # get output dtype and shape
                    type_shape_info = cached_graph.get_output_type_shape_info()[
                        0]
                    output_shape = tuple(type_shape_info.shape)
                    output_dtype = type_shape_info.dtype

                lazy_tensors = set(
                    t._internal_name for t in normalized_tensors
                    if t._ort_value is None)

                for idx, input in enumerate(graph.inputs):
                    # not an actual input
                    if input.name in lazy_tensors:
                        graph.input.pop(idx)

                evaluator = LazyEvaluator()
                evaluator.add_subgraph(graph)

                result_array = array.Array(
                    dims=output_shape, dtype=output_dtype,
                    internal_name=graph_output_name, evaluator=evaluator)
                merge_array_evaluators(
                    result_array._evaluator, *input_evaluators)

            else:
                # this is a cache miss because the function now takes a new parameter dtype/shape combination
                # so we generate a new cached graph for this unseen combination
                result_array = generate_cached_graph(
                    func, *array_objs, **array_obj_kwargs)

        return result_array

    return wrapper_jit
