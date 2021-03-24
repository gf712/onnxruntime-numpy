from .ops_utils import (unary_operator, binary_operator, nary_operator,
                        allowed_types, output_type_is_argn_position,
                        shapes_match_exactly, propagate_shape_from_argn_position,
                        output_checks_and_inference, propagate_shape_matmul,
                        check_input_shape_matmul, not_implemented_types,
                        concatenate_shapes, types_match_exactly,
                        initializer_operator, initializer_operator_from_shape,
                        reduction_axis, output_shape_from_einsum,
                        output_type, allow_broadcasting, array_is_square_matrix,
                        determinant_output_shape, broadcast_to,
                        flatten_shape, gather_check)
from .types import (bool_types, float_types, all_types, integer_types,
                    numeric_types, signed_integer_types, numpy_to_onnx,
                    unsigned_integer_types)
from . import array
from typing import List, Any, Union
from typing import Iterable as IterableType
from collections.abc import Iterable
import numpy as np


# Binary operators

def add(x, y):
    @not_implemented_types([np.uint32, np.uint64], [np.uint32, np.uint64])
    @allowed_types([*float_types, np.int32, np.int64],
                   [*float_types, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def add_helper(x, y):
        return binary_operator(x, y, "Add")

    return add_helper(x, y)


def logical_and(x, y):
    @allowed_types(bool_types, bool_types)
    @output_checks_and_inference(
        allow_broadcasting
    )
    def logical_and_helper(x, y):
        return binary_operator(x, y, "And")
    return logical_and_helper(x, y)


def logical_or(x, y):
    @allowed_types(bool_types, bool_types)
    @output_checks_and_inference(
        allow_broadcasting
    )
    def logical_or_helper(x, y):
        return binary_operator(x, y, "Or")
    return logical_or_helper(x, y)


def subtract(x, y):
    @not_implemented_types([np.uint32, np.uint64], [np.uint32, np.uint64])
    @allowed_types([*float_types, np.int32, np.int64],
                   [*float_types, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def subtract_helper(x, y):
        return binary_operator(x, y, "Sub")
    return subtract_helper(x, y)


def divide(x, y):
    @not_implemented_types([np.uint32, np.uint64], [np.uint32, np.uint64])
    @allowed_types([*float_types, np.int32, np.int64],
                   [*float_types, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def divide_helper(x, y):
        return binary_operator(x, y, "Div")
    return divide_helper(x, y)


def multiply(x, y):
    @not_implemented_types([np.uint32, np.uint64], [np.uint32, np.uint64])
    @allowed_types([*float_types, np.int32, np.int64],
                   [*float_types, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def multiply_helper(x, y):
        return binary_operator(x, y, "Mul")
    return multiply_helper(x, y)


def matmul(x, y):
    @allowed_types([*float_types, np.int32, np.int64, np.uint32, np.uint64],
                   [*float_types, np.int32, np.int64, np.uint32, np.uint64])
    @check_input_shape_matmul
    @output_checks_and_inference(
        propagate_shape_matmul()
    )
    def matmul_helper(x, y):
        return binary_operator(x, y, "MatMul")
    return matmul_helper(x, y)


def absolute(x):
    @allowed_types([*float_types, *integer_types])
    def absolute_helper(x):
        return unary_operator(x, "Abs")
    return absolute_helper(x)


def acos(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_acos(x):
        return unary_operator(x, "Acos")
    return helper_acos(x)


def acosh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_acos(x):
        return unary_operator(x, "Acosh")
    return helper_acos(x)


def asin(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_asin(x):
        return unary_operator(x, "Asin")
    return helper_asin(x)


def asinh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_asinh(x):
        return unary_operator(x, "Asinh")
    return helper_asinh(x)


def atan(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def atan_helper(x):
        return unary_operator(x, "Atan")
    return atan_helper(x)


def atanh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def atanh_helper(x):
        return unary_operator(x, "Atanh")
    return atanh_helper(x)


def ceil(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def ceil_helper(x):
        return unary_operator(x, "Ceil")
    return ceil_helper(x)


def clip(input, minimum=None, maximum=None):
    @not_implemented_types([np.uint16, np.uint32, np.int16, np.int32])
    @allowed_types([*numeric_types, *numeric_types, *numeric_types])
    def clip_helper(input, minimum, maximum):
        return nary_operator("Clip", input, minimum, maximum)
    return clip_helper(input, array.array(minimum, input.dtype), array.array(maximum, input.dtype))


def cast(array: "array.Array", to: type):
    @allowed_types([*all_types])
    @output_checks_and_inference(
        output_type_is_argn_position(1)
    )
    def cast_helper(array: "array.Array", to: type):
        return unary_operator(array, "Cast", to=numpy_to_onnx(np.dtype(to)))

    return cast_helper(array, to)


def compress(array, condition, axis=None):
    raise NotImplementedError()


def concat(arrays: List[array.Array], axis: int = -1) -> array.Array:
    @allowed_types(all_types)
    @types_match_exactly
    @output_checks_and_inference(
        concatenate_shapes("axis")
    )
    def concat_helper(*arrays, axis):
        return nary_operator("Concat", *arrays, axis=axis)

    return concat_helper(*arrays, axis=axis)


def constant(*, sparse_value=None,
             value: IterableType[Any] = None,
             value_float: float = None,
             value_floats: Union[array.Array, IterableType[float]] = None,
             value_int: int = None,
             value_ints: Union[array.Array, IterableType[int]] = None,
             value_string: str = None, value_strings=None) -> array.Array:

    def check_array_shape_and_type(values, expected_ndims, expected_dtype, argname):
        if isinstance(values, array.Array):
            shape = values.shape
            dtype = values.dtype
        else:
            shape = array.infer_shape_from_array(values)
            dtype = array.infer_dtype_from_array(values)
        if len(shape) != expected_ndims:
            raise ValueError(
                f"Argument {argname} expects a {expected_ndims}D array, but got {len(shape)}D array")
        if dtype != expected_dtype:
            raise ValueError(f"Argument {argname} expects an array of type {expected_dtype}, "
                             f"but got type {dtype}")

    if sparse_value:
        raise NotImplementedError("Sparse matrices are currently not supported")
    if value:
        return initializer_operator("Constant", value=value)
    if value_float:
        # TODO: should this be strict? i.e. if not isinstance(value, float) throw?
        return initializer_operator("Constant", value_float=array.array([float(value_float)]))
    if value_floats:
        check_array_shape_and_type(value_floats, 1, np.float32, "value_floats")
        return initializer_operator("Constant", value_floats=value_floats)
    if value_int:
        # TODO: should this be strict? i.e. if not isinstance(value, int) throw?
        return initializer_operator("Constant", value_int=array.array([int(value_int)]))
    if value_ints:
        check_array_shape_and_type(value_ints, 1, np.int32, "value_ints")
        return initializer_operator("Constant", value_ints=value_ints)
    if value_string:
        raise NotImplementedError("Strings are currently not implemented")
        return initializer_operator("Constant", value_string=value_string)
    if value_strings:
        raise NotImplementedError("Strings are currently not implemented")
        check_array_shape_and_type(value_strings, 1, np.STRING, "value_strings")
        return initializer_operator("Constant", value_strings=value_strings)
    else:
        raise ValueError("?")


def constant_of_shape(shape: IterableType[int], value=0.0) -> array.Array:
    if not isinstance(value, Iterable):
        value = array.array([value])
    if len(value) != 1:
        raise ValueError("Value must be a one-dim tensor with a single element")
    shape = array.array(shape, dtype=np.int64)
    return initializer_operator_from_shape("ConstantOfShape", shape, value)


def conv():
    raise NotImplementedError()


def relu(x):
    @not_implemented_types([np.float64, *signed_integer_types])
    @allowed_types([*float_types, *signed_integer_types])
    def relu_helper(x):
        return unary_operator(x, "Relu")
    return relu_helper(x)


def cos(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def cos_helper(x):
        return unary_operator(x, "Cos")
    return cos_helper(x)


def cosh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def cosh_helper(x):
        return unary_operator(x, "Cosh")
    return cosh_helper(x)


def cumsum(x, axis: int = -1, exclusive: bool = False, reverse: bool = False):
    @not_implemented_types([np.uint32, np.uint64])
    @allowed_types([*float_types, np.uint32, np.uint64, np.int32, np.int64])
    def cumsum_helper(x, axis, exclusive, reverse):
        return nary_operator("CumSum", x, array.array(axis, dtype=np.int32),
                             exclusive=int(exclusive), reverse=int(reverse))

    return cumsum_helper(x, axis, exclusive, reverse)


def det(x: "array.Array"):
    # TODO: add output shape inference
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    @array_is_square_matrix
    @output_checks_and_inference(
        determinant_output_shape
    )
    def det_helper(x):
        return unary_operator(x, "Det")
    return det_helper(x)


def einsum(*inputs, equation):
    @not_implemented_types([*unsigned_integer_types, np.int8, np.int16])
    @allowed_types([*float_types, *integer_types])
    @output_checks_and_inference(
        output_shape_from_einsum("equation")
    )
    def einsum_helper(*inputs, equation):
        return nary_operator("Einsum", *inputs, equation=equation)

    return einsum_helper(*inputs, equation=equation)


def elu(x, alpha=1.0):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def elu_helper(x, alpha):
        return unary_operator(x, "Elu", alpha=alpha)

    return elu_helper(x, alpha=float(alpha))


def equal(x, y):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16])
    @allowed_types(all_types, all_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def equal_helper(x, y):
        return binary_operator(x, y, "Equal")

    return equal_helper(x, y)


def erf(x):
    @not_implemented_types([*integer_types, np.float64])
    @allowed_types(numeric_types)
    def erf_helper(x):
        return unary_operator(x, "Erf")

    return erf_helper(x)


def exp(x):
    @allowed_types(float_types)
    def exp_helper(x):
        return unary_operator(x, "Exp")

    return exp_helper(x)


def expand(x, shape):
    # we have to evaluate the graph here (if necessary)
    numpy_array_shape = shape.numpy()
    if (len(numpy_array_shape.shape) != 1):
        raise ValueError("Shape must be a 1D tensor")

    @allowed_types(all_types, [np.int64])
    @output_checks_and_inference(
        broadcast_to(tuple(int(el) for el in numpy_array_shape))
    )
    def expand_helper(x, shape):
        return nary_operator("Expand", x, shape)

    return expand_helper(x, shape)


def eye_like(x: "array.Array", dtype: np.dtype = None, k=0):

    if x.ndims != 2:
        raise ValueError("Tensor must be 2D")

    if dtype is None:
        dtype = x.dtype
    if dtype not in all_types:
        raise TypeError(
            f"Output type {dtype} not supported. Supported types are {all_types}")
    if dtype not in [*float_types, np.uint64, np.int32, np.int64]:
        raise NotImplementedError(f"Output type {dtype} currently not implemented")

    @allowed_types(all_types)
    @not_implemented_types([np.int8, np.int16, np.uint8, np.uint16, np.uint32, np.bool_])
    @output_checks_and_inference(
        output_type(dtype)
    )
    def eye_like_helper(x: "array.Array", dtype: int, k: int):
        return unary_operator(x, "EyeLike", dtype=dtype, k=k)

    return eye_like_helper(x, dtype=numpy_to_onnx(np.dtype(dtype)), k=int(k))


def flatten(x: "Array.array", axis: int = 1):

    @allowed_types(all_types)
    @output_checks_and_inference(
        flatten_shape(int(axis))
    )
    def flatten_helper(x: "Array.array", axis: int):
        return unary_operator(x, "Flatten", axis=axis)

    return flatten_helper(x, axis=int(axis))


def floor(x: "array.Array"):
    @allowed_types(float_types)
    def floor_helper(x: "array.Array"):
        return unary_operator(x, "Floor")
    return floor_helper(x)


def gru(x: "array.Array", w: "array.Array", r: "array.Array", b: "array.Array", sequence_length: "array.Array", initial_h: "array.Array",
        hidden_size: int, activation_alpha: List[float] = None, activation_beta: List[float] = None, activations: List[str] = None, clip: float = 0.0,
        direction: str = "forward", layout: int = 0, linear_before_reset: bool = False):
    raise NotImplementedError()


def gather(x: "array.Array", indices: "array.Array", axis: int = 0):

    @allowed_types(all_types, [np.int32, np.int64])
    @output_checks_and_inference(
        gather_check(int(axis))
    )
    def gather_helper(x: "array.Array", indices: "array.Array", axis: int):
        return nary_operator("Gather", x, indices, axis=axis)

    return gather_helper(x, indices, axis=int(axis))