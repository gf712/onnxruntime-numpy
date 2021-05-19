import onnx
from typing import List, Tuple, Optional
import onnxruntime
import numpy as np
from . import core
from . import array
from .types import numpy_to_ort, ort_to_numpy
from .shapes import weak_shape_comparisson, as_shape
from .graph import Graph, ExecutableGraph, compile_graph
from .config import Config, get_ort_graph_optimization_level
from .exceptions import InternalException


class LazyEvaluator:
    def __init__(self):
        self._parent_node: Optional[str] = None
        self._output_node: Optional[str] = None
        self._array_to_node_map = core.ArrayNodeLookupTable()
        self._cached_results = core.IntermediateResultCache()
        self._graph: Graph = Graph()

    def copy(self) -> "LazyEvaluator":
        evaluator = LazyEvaluator()
        evaluator._graph = self._graph
        evaluator._cached_results = self._cached_results
        evaluator._array_to_node_map = self._array_to_node_map

        return evaluator

    def add_node(
            self, op_name: str, inputs: Tuple["array.Array"],
            outputs: Tuple["array.Array"], **attributes):

        self._parent_node, output_node_names = self._graph.add_node(
            op_name, inputs, outputs, **attributes)

        for output_array, output_node_name in zip(outputs, output_node_names):
            output_array._internal_array._evaluator._output_node = output_node_name
            output_array._internal_array._evaluator._parent_node = self._parent_node

        if self._parent_node is not None:
            # keep mypy happy because self._parent_node is Optional[str]
            self._array_to_node_map.add_input(self._parent_node, inputs)
            self._array_to_node_map.add_output(self._parent_node, outputs)
        else:
            raise InternalException("Parent node not set")
        return

    def add_initializer(
            self, name: str, dtype: np.dtype, dims: Tuple[int],
            vals):
        raise NotImplementedError("Initializers not implemented")

    def add_input(self, array: "array.Array"):
        dtype = array.dtype
        dims = array.shape
        default_values = array._internal_array._ort_value
        # FIXME
        if default_values is not None:
            if default_values.data_type() != numpy_to_ort(
                    np.dtype(dtype)):  # pragma: no cover
                raise TypeError("Input type does not match input node")
            default_shape = as_shape(default_values.shape())
            if not weak_shape_comparisson(
                    default_shape, dims):  # pragma: no cover
                raise ValueError(
                    f"Input tensor shape {default_shape} does not match input "
                    f"node shape {dims}")

        input_name, output_name = self._graph.add_input(array)
        self._parent_node = input_name
        self._output_node = output_name
        if self._parent_node is None:
            raise InternalException("Parent node not set")
        self._array_to_node_map.add_input(self._parent_node, (array,))
        self._array_to_node_map.add_output(self._parent_node, (array,))

    def add_subgraph(self, other_graph: Graph):
        if self._graph is None:
            self._graph = other_graph
        elif other_graph is not None:
            self._graph.add_subgraph(other_graph)

    def merge(self, other: "LazyEvaluator"):
        self.add_subgraph(other._graph)
        # share result cache
        self._cached_results.merge(other._cached_results)
        other._cached_results = self._cached_results

        self._array_to_node_map.update(other._array_to_node_map)
        other._array_to_node_map = self._array_to_node_map
        return

    def _build_executable_graph(self, array: "array.Array") -> ExecutableGraph:
        if self._parent_node is None:
            raise InternalException("Parent node not set")
        if self._output_node is None:
            raise InternalException("Output node not set")
        # FIXME: need to fix result caching
        return compile_graph(
            self._graph, self._array_to_node_map.get_input_map(),
            self._array_to_node_map.get_output_map(),
            [self._output_node],
            self._cached_results)

    def evaluate(self, output_array: "array.Array") -> List[np.ndarray]:
        if self._graph is None:  # pragma: no cover
            raise InternalException(
                "Graph is empty. "
                "This is an internal error. Please file a bug")

        output_node = self._array_to_node_map.get_output_map()[
            self._parent_node]
        output_idx = [o[1]._internal_name for o in output_node].index(
            output_array._internal_name)

        if output_idx == -1:
            raise InternalException(
                "Could not find index of output Array in output node")

        executable_graph = self._build_executable_graph(output_array)

        onnx_graph = executable_graph.build_onnx_graph()

        m = onnx.helper.make_model(onnx_graph)
        buffer = m.SerializeToString()

        output_name = list(self._graph._graph.in_edges(
            self._output_node, data=True))[0][-1]["name"]

        # TODO: maybe disable optimisations when graph has already been optimised
        # with jit?
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = get_ort_graph_optimization_level()

        try:
            onnx.save_model(m, "failed_model.onnx")

            session = onnxruntime.InferenceSession(
                buffer, providers=Config().get_providers(),
                sess_options=session_options)
        except Exception:  # pragma: no cover
            # dump failed model for debugging purposes
            onnx.save_model(m, "failed_model.onnx")
            raise
        io_binding = session.io_binding()
        session_input_names = [i.name for i in session.get_inputs()]

        graph_node_mapping = executable_graph.get_input_node_mapping()
        array_mapping = self._array_to_node_map.get_input_map()
        inputs = {
            **
            {input_output_name: array_mapping[input_node_name][0]
             for input_node_name, input_output_name in graph_node_mapping.items()
             # TODO: some input nodes, such as initializers, do not require an input
             # array. This should be cleaned up
             if input_node_name in array_mapping and len(array_mapping[input_node_name]) > 0},  # noqa
            **self._cached_results.get_all_cache_tensor_mappings()}
        inputs = {k: v for k, v in inputs.items() if k in session_input_names}

        if len(inputs) != len(session_input_names):
            raise InternalException(
                f"Expected {len(session_input_names)} inputs, but got {len(inputs)}")

        for input_name, input_array in inputs.items():
            if isinstance(input_array, tuple):
                ortvalue = input_array[1]._ort_value
            else:
                ortvalue = input_array._ort_value
            if ortvalue is None:
                raise ValueError(
                    "Internal bug. Array's Ortvalue is not set and can not be a model "
                    "input")
            ort_value_dtype = ort_to_numpy(ortvalue.data_type())
            # this will work 99% of the time in this century :D
            if ort_value_dtype == np.int64:
                ort_value_dtype = np.longlong
            if ort_value_dtype == np.uint64:
                ort_value_dtype = np.ulonglong

            shape = ortvalue.shape()

            io_binding.bind_input(
                name=input_name, device_type=ortvalue.device_name(),
                device_id=0, element_type=ort_value_dtype, shape=shape,
                buffer_ptr=ortvalue.data_ptr())

        if len(onnx_graph.output) != 1:
            raise NotImplementedError(
                "Only single output inference is supported")

        io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)

        result = io_binding.get_outputs()[0]

        if self._parent_node is not None:
            self._cached_results.add_entry(
                self._parent_node, output_array)
        else:
            raise InternalException("Evaluator does not have a parent node")

        return result


def merge_array_evaluators(
        evaluator_to_merge_into: LazyEvaluator, *evaluators: LazyEvaluator):
    for evaluator in evaluators:
        evaluator_to_merge_into.merge(evaluator)
