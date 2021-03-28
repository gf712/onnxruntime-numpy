import onnx
from . import array
from .types import numpy_to_onnx


def make_onnx_tensor(name: str, array: "array.Array") -> onnx.TensorProto:
    dims = array.shape
    data_type = numpy_to_onnx(array.dtype)
    return onnx.helper.make_tensor(
        name, data_type, dims, array.numpy().flatten())


def make_onnx_tensor_value_info(array: "array.Array") -> onnx.ValueInfoProto:
    dims = array.shape
    data_type = numpy_to_onnx(array.dtype)
    return onnx.helper.make_tensor_value_info(
        array._internal_name, data_type, dims)
