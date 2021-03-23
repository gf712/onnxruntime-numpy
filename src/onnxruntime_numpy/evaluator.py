import onnx
from typing import List, Any, Tuple
import copy
import onnxruntime
import numpy as np  # FIXME maybe
from .types import numpy_to_onnx, numpy_to_ort, ort_to_numpy
from .shapes import shapes_match
import uuid

from onnxruntime.capi import _pybind_state as C

def flatten(vals, dtype) -> List[Any]:
    # TODO: implement this without numpy
    import numpy as np
    return np.array(vals, dtype=dtype).flatten().tolist()


class LazyEvaluator:
    def __init__(self):
        # self._nodes = {}
        # self._inputs = {}
        # self._outputs = []
        # self._initializers = {}
        self._input_values = {}
        self._input_names = set()
        self._node_names = set()
        self._initializer_names = set()
        graph_name = f"graph_{uuid.uuid4()}"
        self._graph_names = set((graph_name,))
        self._graph = onnx.GraphProto(name=graph_name)

    def copy(self) -> "LazyEvaluator":
        evaluator = LazyEvaluator()
        # evaluator._nodes = self._nodes
        # evaluator._initializers = self._initializers
        evaluator._input_values = self._input_values
        # evaluator._inputs = self._inputs
        # FIXME: is this copy necessary?
        evaluator._graph.CopyFrom(self._graph)
        evaluator._input_names = copy.deepcopy(self._input_names)
        evaluator._node_names = copy.deepcopy(self._node_names)
        evaluator._initializer_names = copy.deepcopy(self._initializer_names)
        evaluator._graph_names = copy.deepcopy(self._graph_names)

        return evaluator

    def _reset(self):
        self._input_values = {}
        self._input_names = set()
        self._node_names = set()
        self._initializer_names = set()

    def add_node(self, op_type, inputs, outputs, **attributes):
        node_name = op_type + ''.join(inputs) + ''.join(outputs) + \
            ''.join(str(k) + str(v) for k, v in attributes.items())
        if node_name not in self._node_names:
            # self._nodes[node_name] = onnx.helper.make_node(
            #     op_type, inputs, outputs, **attributes)
            node = onnx.helper.make_node(op_type, inputs, outputs, **attributes)
            self._graph.node.append(node)
            self._node_names.add(node_name)

    def add_initializer(self, name: str, dtype: np.dtype, dims: Tuple[int], vals):
        # if name not in self._initializers:
        #     flat_values = flatten(vals)
        #     self._initializers[name] = onnx.helper.make_tensor(name, dtype, dims, flat_values)
        if name not in self._initializer_names:
            flat_values = flatten(vals, dtype=dtype)
            initializer = onnx.helper.make_tensor(name, dtype, dims, flat_values)
            self._graph.intializer.append(initializer)
            self._initializer_names.add(name)

    def add_input(self, name: str, dtype: np.dtype, dims: Tuple[int], default_values: onnxruntime.OrtValue):
        # if name not in self._inputs:
        #     self._inputs[name] = onnx.helper.make_tensor_value_info(name, dtype, dims)
        #     # FIXME
        #     self._input_values[name] = np.array(default_values, dtype=onnx_to_numpy(dtype)).reshape(dims)
        #     self._graph.inputs.append()
        if name not in self._input_names:
            onnx_type = numpy_to_onnx(dtype)
            input_node = onnx.helper.make_tensor_value_info(name, onnx_type, dims)
            # FIXME
            if default_values is not None:
                if default_values.data_type() != numpy_to_ort(dtype):
                    raise TypeError("Input type does not match input node")
                if not shapes_match(default_values.shape(), dims):
                    raise ValueError(
                        f"Input tensor shape {default_values.shape()} does not match input node shape {dims}")
                self._input_values[name] = default_values
            self._graph.input.append(input_node)
            self._input_names.add(name)

    def add_subgraph(self, graph: onnx.GraphProto):
        if self._graph is None:
            self._graph = graph
        else:
            if graph.name not in self._graph_names:
                # only merge inputs and outputs since these must be unique
                for input in graph.input:
                    if input not in self._graph.input:
                        self._graph.input.append(input)
                for initializer in graph.initializer:
                    if initializer not in self._graph.initializer:
                        self._graph.initializer.append(initializer)
                for output in graph.output:
                    if output not in self._graph.output:
                        self._graph.output.append(output)
                self._graph.node.MergeFrom(graph.node)

                # self._graph.MergeFrom(graph)
                self._graph_names.add(graph.name)

    def merge(self, other: "LazyEvaluator"):
        # self._nodes = {**self._nodes, **other._nodes}
        # self._inputs = {**self._inputs, **other._inputs}
        self.add_subgraph(other._graph)
        self._input_values = {**self._input_values, **other._input_values}
        self._input_names.update(other._input_names)
        self._node_names.update(other._node_names)
        self._initializer_names.update(other._initializer_names)
        self._graph_names.update(other._graph_names)

        # self._initializers = {**self._initializers, **other._initializers}

    def evaluate(self, output_name: str, expected_output_type: np.dtype, expected_output_shape: Tuple[int]) -> List[np.ndarray]:
        if self._graph is None:
            raise ValueError("Graph is empty. "
                             "This is an internal error. Please file a bug")

        output = onnx.helper.make_tensor_value_info(
            output_name, numpy_to_onnx(np.dtype(expected_output_type)), expected_output_shape)
        outputs = [output]

        self._graph.output.extend(outputs)
        m = onnx.helper.make_model(self._graph)
        buffer = m.SerializeToString()

        feeds = self._input_values
        output_names = [output_name]

        inputs = list(self._graph.input)
        outputs = list(self._graph.output)
        nodes = list(self._graph.node)

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

        for input_name, ortvalue in self._input_values.items():
            # this will work 99% of the time in this century :D
            ort_value_dtype = ortvalue._numpy_obj.dtype
            if ort_value_dtype == np.int64:
                ort_value_dtype = np.longlong
            if ort_value_dtype == np.uint64:
                ort_value_dtype = np.ulonglong

            io_binding.bind_input(name=input_name, device_type=ortvalue.device_name(),
                                  device_id=0,
                                  element_type=ort_value_dtype,
                                  shape=ortvalue.shape(),
                                  buffer_ptr=ortvalue.data_ptr())

        if len(outputs) != 1:
            raise NotImplementedError("Only single output inference is supported")

        io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)

        result = io_binding.get_outputs()[0]

        self._graph = None
        self._reset()

        return result
