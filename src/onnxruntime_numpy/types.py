from onnx import TensorProto
import numpy as np
from typing import Union


float_types = [
    # np.float16,
    np.float32,
    np.double
]

unsigned_integer_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64
]

signed_integer_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]

integer_types = [
    *unsigned_integer_types,
    *signed_integer_types
]

bool_types = [
    np.bool_
]

numeric_types = [
    *float_types,
    *integer_types
]

all_types = [
    *numeric_types,
    *bool_types,
]


def is_float(dtype: Union[np.dtype, type]) -> bool:
    return np.dtype(dtype) in float_types


def is_unsigned_int(dtype: Union[np.dtype, type]) -> bool:
    return np.dtype(dtype) in unsigned_integer_types


def is_signed_int(dtype: Union[np.dtype, type]) -> bool:
    return np.dtype(dtype) in signed_integer_types


def is_integer(dtype: Union[np.dtype, type]) -> bool:
    return is_signed_int(dtype) or is_unsigned_int(dtype)


def is_numeric(dtype: Union[np.dtype, type]) -> bool:
    return is_float(dtype) or is_integer(dtype)


def is_bool(dtype: Union[np.dtype, type]) -> bool:
    return np.dtype(dtype) in bool_types


numpy_type_map = {
    np.dtype(np.bool_):    TensorProto.BOOL,
    np.dtype(np.float16):  TensorProto.FLOAT,
    np.dtype(np.float32):  TensorProto.FLOAT,
    np.dtype(np.float64):  TensorProto.DOUBLE,
    np.dtype(np.uint8):    TensorProto.UINT8,
    np.dtype(np.uint16):   TensorProto.UINT16,
    np.dtype(np.uint32):   TensorProto.UINT32,
    np.dtype(np.uint64):   TensorProto.UINT64,
    np.dtype(np.int8):     TensorProto.INT8,
    np.dtype(np.int16):    TensorProto.INT16,
    np.dtype(np.int32):    TensorProto.INT32,
    np.dtype(np.int64):    TensorProto.INT64
}

onnx_type_map = {
    TensorProto.BOOL:   np.dtype(np.bool_),
    TensorProto.FLOAT:  np.dtype(np.float32),
    TensorProto.DOUBLE: np.dtype(np.float64),
    TensorProto.UINT8:  np.dtype(np.uint8),
    TensorProto.UINT16: np.dtype(np.uint16),
    TensorProto.UINT32: np.dtype(np.uint32),
    TensorProto.UINT64: np.dtype(np.uint64),
    TensorProto.INT8:   np.dtype(np.int8),
    TensorProto.INT16:  np.dtype(np.int16),
    TensorProto.INT32:  np.dtype(np.int32),
    TensorProto.INT64:  np.dtype(np.int64)
}


def numpy_to_onnx(dtype: np.dtype) -> int:
    return numpy_type_map[dtype]


def onnx_to_numpy(dtype: int) -> np.dtype:
    return onnx_type_map[dtype]


python_type_map = {
    bool:  TensorProto.BOOL,
    float: TensorProto.FLOAT,
    int:   TensorProto.INT32,
}

onnx_str_map = {
    TensorProto.BOOL:   "bool",
    TensorProto.FLOAT:  "float32",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT8:  "uint8",
    TensorProto.UINT16: "uint16",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.INT8:   "int8",
    TensorProto.INT16:  "int16",
    TensorProto.INT32:  "int32",
    TensorProto.INT64:  "int64"
}

onnx_str_map = {
    TensorProto.BOOL:   bool,
    TensorProto.FLOAT:  float,
    TensorProto.DOUBLE: float,
    TensorProto.UINT8:  int,
    TensorProto.UINT16: int,
    TensorProto.UINT32: int,
    TensorProto.UINT64: int,
    TensorProto.INT8:   int,
    TensorProto.INT16:  int,
    TensorProto.INT32:  int,
    TensorProto.INT64:  int,
}


ort_to_numpy_map = {
    "tensor(bool)":    np.dtype(np.bool_),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)":   np.dtype(np.float32),
    "tensor(double)":  np.dtype(np.float64),
    "tensor(uint8)":   np.dtype(np.uint8),
    "tensor(uint16)":  np.dtype(np.uint16),
    "tensor(uint32)":  np.dtype(np.uint32),
    "tensor(uint64)":  np.dtype(np.uint64),
    "tensor(int8)":    np.dtype(np.int8),
    "tensor(int16)":   np.dtype(np.int16),
    "tensor(int32)":   np.dtype(np.int32),
    "tensor(int64)":   np.dtype(np.int64),
}


numpy_to_ort_map = {
    np.dtype(np.bool_):   "tensor(bool)",
    np.dtype(np.float16): "tensor(float16)",
    np.dtype(np.float32): "tensor(float)",
    np.dtype(np.float64): "tensor(double)",
    np.dtype(np.uint8):   "tensor(uint8)",
    np.dtype(np.uint16):  "tensor(uint16)",
    np.dtype(np.uint32):  "tensor(uint32)",
    np.dtype(np.uint64):  "tensor(uint64)",
    np.dtype(np.int8):    "tensor(int8)",
    np.dtype(np.int16):   "tensor(int16)",
    np.dtype(np.int32):   "tensor(int32)",
    np.dtype(np.int64):   "tensor(int64)",
}


# def python_to_onnx(dtype: type) -> int:
#     return python_type_map[dtype]


# def onnx_to_string(dtype: int) -> str:
#     return onnx_str_map[dtype]

# def onnx_to_python(dtype: int) -> type:
#     return onnx_str_map[dtype]

def ort_to_numpy(dtype: str) -> np.dtype:
    return ort_to_numpy_map[dtype]


def numpy_to_ort(dtype: np.dtype) -> str:
    return numpy_to_ort_map[dtype]


def python_to_numpy(dtype: type) -> np.dtype:
    if dtype == float:
        return np.dtype(np.float32)
    if dtype == int:
        return np.dtype(np.int32)
    if dtype == bool:
        return np.dtype(np.bool_)
    raise TypeError(f"Python type {dtype} not supported")
