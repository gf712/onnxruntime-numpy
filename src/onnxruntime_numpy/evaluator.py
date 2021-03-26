import onnx
from typing import List, Any, Tuple
import onnxruntime
import numpy as np  # FIXME maybe
from .types import numpy_to_onnx, numpy_to_ort, ort_to_numpy
from .shapes import shapes_match
import uuid
from .graph import Graph
from collections.abc import Iterable

from onnxruntime.capi import _pybind_state as C


class LazyEvaluator:
    def __init__(self):
        # the name of the node that needs to be executed to get the data
        self._parent_node = None
        # self._initializers = {}
        self._input_values = {}
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

    def _reset(self):
        self._input_values = {}
        self._parent_node = None
        # self._input_names = set()
        # self._node_names = set()
        # self._initializer_names = set()

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
        self._parent_node = self._graph.add_node(op_type, inputs, outputs, **attributes)
        # self._node_names.add(node_name)

    def add_initializer(self, name: str, dtype: np.dtype, dims: Tuple[int], vals):
        # if name not in self._initializers:
        #     flat_values = flatten(vals)
        #     self._initializers[name] = onnx.helper.make_tensor(name, dtype, dims, flat_values)
        # if name not in self._initializer_names:
        raise NotImplementedError("Initializers not implemented")
        # self._graph.add_initializer()
        #     flat_values = flatten(vals, dtype=dtype)
        #     initializer = onnx.helper.make_tensor(name, dtype, dims, flat_values)
        #     self._graph.intializer.append(initializer)
        #     self._initializer_names.add(name)

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
            if default_values.data_type() != numpy_to_ort(dtype):
                raise TypeError("Input type does not match input node")
            if not shapes_match(default_values.shape(), dims):
                raise ValueError(
                    f"Input tensor shape {default_values.shape()} does not match input node shape {dims}")
            self._input_values[name] = array
        else:
            raise NotImplementedError()
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

    def evaluate(self, array: "array.Array") -> List[np.ndarray]:
        if self._graph is None:
            raise ValueError("Graph is empty. "
                             "This is an internal error. Please file a bug")

        onnx_graph = self._graph.build_onnx_graph(array, self._input_values, self._parent_node)
        m = onnx.helper.make_model(onnx_graph)
        buffer = m.SerializeToString()

        output_name = array._internal_name

        output_names = [output_name]

        inputs = list(onnx_graph.input)
        outputs = list(onnx_graph.output)
        nodes = list(onnx_graph.node)

        # TODO: how to handle multiple return values?
        try:
            # TODO: maybe disable optimisations when graph has already been optimised with jit?
            # session_options = C.get_default_session_options()
            # session_options.graph_optimization_level = session_options.graph_optimization_level.ORT_DISABLE_ALL
            # session = onnxruntime.InferenceSession(buffer, sess_options=session_options)
            session = onnxruntime.InferenceSession(buffer)
        except:
            onnx.save_model(m, "failed_model.onnx")
            raise
        io_binding = session.io_binding()

        for input_name, array in self._input_values.items():
            ortvalue = array._ort_value
            # this will work 99% of the time in this century :D
            ort_value_dtype = ortvalue._numpy_obj.dtype
            if ort_value_dtype == np.int64:
                ort_value_dtype = np.longlong
            if ort_value_dtype == np.uint64:
                ort_value_dtype = np.ulonglong

            shape = ortvalue.shape()

            io_binding.bind_input(name=input_name, device_type=ortvalue.device_name(),
                                  device_id=0,
                                  element_type=ort_value_dtype,
                                  shape=shape,
                                  buffer_ptr=ortvalue.data_ptr())

        if len(onnx_graph.output) != 1:
            raise NotImplementedError("Only single output inference is supported")

        io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)

        result = io_binding.get_outputs()[0]

        self._graph = None
        self._reset()

        return result
