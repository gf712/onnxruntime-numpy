import onnx
from typing import List, Tuple, Dict
import onnxruntime
import numpy as np  # FIXME maybe
from .types import numpy_to_ort, ort_to_numpy
from .shapes import weak_shape_comparisson, as_shape
from .graph import Graph, ExecutableGraph
from collections.abc import Iterable
from . import array


class LazyEvaluator:
    def __init__(self):
        # the name of the node that needs to be executed to get the data
        self._parent_node = None
        # self._initializers = {}
        self._input_values: Dict[str, "array.Array"] = {}
        # graph_name = f"graph_{uuid.uuid4()}"
        # self._graph_names = set((graph_name,))
        self._graph = None

    def copy(self) -> "LazyEvaluator":
        evaluator = LazyEvaluator()
        evaluator._input_values = self._input_values
        evaluator._graph = self._graph
        # evaluator._input_names = copy.deepcopy(self._input_names)
        # evaluator._node_names = copy.deepcopy(self._node_names)
        # evaluator._initializer_names = copy.deepcopy(self._initializer_names)
        # evaluator._graph_names = copy.deepcopy(self._graph_names)

        return evaluator

    def add_node(self, op_type, inputs, outputs, **attributes):

        if self._graph is None:
            self._graph = Graph()

        if isinstance(inputs, Iterable):
            inputs = tuple(inputs)
        else:
            inputs = tuple((inputs,))

        if isinstance(outputs, Iterable):
            outputs = tuple(outputs)
        else:
            outputs = tuple((outputs,))

        # if node_name not in self._node_names:
        self._parent_node = self._graph.add_node(
            op_type, inputs, outputs, **attributes)
        # self._node_names.add(node_name)

    def add_initializer(
            self, name: str, dtype: np.dtype, dims: Tuple[int],
            vals):
        raise NotImplementedError("Initializers not implemented")

    def add_input(self, array: "array.Array"):
        name = array._internal_name
        dtype = array.dtype
        dims = array.shape
        default_values = array._ort_value
        # if name not in self._input_names:
        # onnx_type = numpy_to_onnx(dtype)
        # input_node = onnx.helper.make_tensor_value_info(name, onnx_type, dims)
        # FIXME
        if default_values is not None:
            if default_values.data_type() != numpy_to_ort(
                    np.dtype(dtype)):  # pragma: no cover
                raise TypeError("Input type does not match input node")
            default_shape = as_shape(default_values.shape())
            if not weak_shape_comparisson(
                    default_shape,
                    dims):  # pragma: no cover
                raise ValueError(
                    f"Input tensor shape {default_shape} does not match input "
                    f"node shape {dims}")
            self._input_values[name] = array
        else:
            # empty array
            self._input_values[name] = array
        # self._input_names.add(name)
        # self._parent_node = self._graph.add_input(array)

    def add_subgraph(self, other_graph: Graph):
        if self._graph is None:
            self._graph = other_graph
        elif other_graph is not None:
            # if graph.name not in self._graph_names:
            self._graph.add_subgraph(other_graph)
            # self._graph_names.add(graph.name)

    def merge(self, other: "LazyEvaluator"):
        self.add_subgraph(other._graph)
        self._input_values = {**self._input_values, **other._input_values}
        # self._input_names.update(other._input_names)
        # self._node_names.update(other._node_names)
        # self._initializer_names.update(other._initializer_names)
        # self._graph_names.update(other._graph_names)
        # self._initializers = {**self._initializers, **other._initializers}

    def _build_executable_graph(self, array: "array.Array") -> ExecutableGraph:
        return ExecutableGraph(
            self._graph, self._input_values.values(),
            {self._parent_node: array})

    def evaluate(self, array: "array.Array") -> List[np.ndarray]:
        if self._graph is None:  # pragma: no cover
            raise ValueError("Graph is empty. "
                             "This is an internal error. Please file a bug")

        executable_graph = self._build_executable_graph(array)

        onnx_graph = executable_graph.build_onnx_graph()
        m = onnx.helper.make_model(onnx_graph)
        buffer = m.SerializeToString()

        output_name = array._internal_name

        # TODO: how to handle multiple return values?
        try:
            # TODO: maybe disable optimisations when graph has already been optimised
            # with jit?
            session = onnxruntime.InferenceSession(buffer)
        except Exception:  # pragma: no cover
            # dump failed model for debugging purposes
            onnx.save_model(m, "failed_model.onnx")
            raise
        io_binding = session.io_binding()

        for input_name, array in self._input_values.items():
            ortvalue = array._ort_value
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

        return result


def merge_array_evaluators(
        evaluator_to_merge_into: LazyEvaluator, *evaluators: LazyEvaluator):
    for evaluator in evaluators:
        evaluator_to_merge_into.merge(evaluator)
