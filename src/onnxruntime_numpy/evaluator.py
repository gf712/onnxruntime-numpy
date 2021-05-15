import onnx
from typing import List, Tuple, Dict, Optional
import onnxruntime
import numpy as np
from .types import numpy_to_ort, ort_to_numpy
from .shapes import weak_shape_comparisson, as_shape
from .graph import Graph, ExecutableGraph, compile_graph
from .config import Config, get_ort_graph_optimization_level
from .exceptions import InternalException
from . import array


class IntermediateResultCache:
    def __init__(self):
        self._cache: Dict[str, "array.Array"] = {}

    def add_entry(self, node_name: str, array: "array.Array"):
        self._cache[node_name] = array

    def empty(self) -> bool:
        return len(self._cache) == 0

    def to_dict(self):
        return self._cache


class ArrayNodeLookupTable:
    def __init__(self):
        self._input_table: Dict[str, Tuple["array.Array"]] = {}
        self._output_table: Dict[str, Tuple["array.Array"]] = {}

    def add_input(self, node_name: str, arrays: Tuple["array.Array"]):
        self._input_table[node_name] = arrays

    def add_output(self, node_name: str, arrays: Tuple["array.Array"]):
        self._output_table[node_name] = arrays

    def get_input_map(self):
        return self._input_table

    def get_output_map(self):
        return self._output_table

    def update(self, other):
        self._input_table = {**self._input_table, **other._input_table}
        self._output_table = {**self._output_table, **other._output_table}


class LazyEvaluator:
    def __init__(self):
        # the name of the node that needs to be executed to get the data
        self._parent_node: Optional[str] = None
        self._array_to_node_map: ArrayNodeLookupTable = ArrayNodeLookupTable()
        self._cached_results: IntermediateResultCache = IntermediateResultCache()
        self._graph: Graph = Graph()

    def copy(self) -> "LazyEvaluator":
        evaluator = LazyEvaluator()
        evaluator._graph = self._graph
        evaluator._cached_results = self._cached_results
        evaluator._array_to_node_map = self._array_to_node_map

        return evaluator

    def _get_array_output_index(self, a: "array.Array"):
        output_map = self._array_to_node_map.get_output_map()
        arrays = output_map[
            a._evaluator._parent_node]
        array_names = [a._internal_name for a in arrays]
        idx = array_names.index(a._internal_name)
        if idx == -1:
            raise InternalException()
        return idx

    def add_node(
            self, op_name: str, inputs: Tuple["array.Array"],
            outputs: Tuple["array.Array"],
            **attributes):

        input_idx_pairs = tuple((None, None) if input_array is None else (
            self._get_array_output_index(input_array),
            input_array) for input_array in inputs)

        self._parent_node = self._graph.add_node(
            op_name, input_idx_pairs, **attributes)

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
        default_values = array._ort_value
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

        input_name, _ = self._graph.add_input(array)
        self._parent_node = input_name
        self._array_to_node_map.add_input(input_name, (array,))
        self._array_to_node_map.add_output(input_name, (array,))

    def add_subgraph(self, other_graph: Graph):
        if self._graph is None:
            self._graph = other_graph
        elif other_graph is not None:
            # if graph.name not in self._graph_names:
            self._graph.add_subgraph(other_graph)
            # self._graph_names.add(graph.name)

    def merge(self, other: "LazyEvaluator"):
        self.add_subgraph(other._graph)
        # share result cache
        for n, c in other._cached_results.to_dict().items():
            self._cached_results.add_entry(n, c)
        other._cached_results = self._cached_results

        self._array_to_node_map.update(other._array_to_node_map)
        other._array_to_node_map = self._array_to_node_map
        return
        # self._input_names.update(other._input_names)
        # self._node_names.update(other._node_names)
        # self._initializer_names.update(other._initializer_names)
        # self._graph_names.update(other._graph_names)
        # self._initializers = {**self._initializers, **other._initializers}

    def _build_executable_graph(self, array: "array.Array") -> ExecutableGraph:
        if self._parent_node is None:
            raise InternalException("Parent node not set")
        # FIXME: need to fix result caching
        return compile_graph(
            self._graph, self._array_to_node_map.get_input_map(),
            self._array_to_node_map.get_output_map(),
            {self._parent_node: array},
            IntermediateResultCache())  # self._cached_results)

    def evaluate(self, output_array: "array.Array") -> List[np.ndarray]:
        if self._graph is None:  # pragma: no cover
            raise InternalException(
                "Graph is empty. "
                "This is an internal error. Please file a bug")

        executable_graph = self._build_executable_graph(output_array)

        onnx_graph = executable_graph.build_onnx_graph()

        m = onnx.helper.make_model(onnx_graph)
        buffer = m.SerializeToString()

        output_name = output_array._internal_name

        # TODO: maybe disable optimisations when graph has already been optimised
        # with jit?
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = get_ort_graph_optimization_level()

        try:
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
             for input_node_name, input_output_name in graph_node_mapping.items()},
            **
            {a._internal_name: a
             for a in self._cached_results.to_dict().values()}}
        inputs = {k: v for k, v in inputs.items() if k in session_input_names}

        if len(inputs) != len(session_input_names):
            raise InternalException(
                f"Expected {len(session_input_names)} inputs, but got {len(inputs)}")

        for input_name, input_array in inputs.items():
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
                self._parent_node, array.Array(ort_value=result))
        else:
            raise InternalException("Evaluator does not have a parent node")

        return result


def merge_array_evaluators(
        evaluator_to_merge_into: LazyEvaluator, *evaluators: LazyEvaluator):
    for evaluator in evaluators:
        evaluator_to_merge_into.merge(evaluator)
