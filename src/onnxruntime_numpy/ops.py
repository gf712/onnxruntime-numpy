from .ops_utils import (
    unary_operator, binary_operator, nary_operator, allowed_types,
    output_type_is_argn_position, multi_output_nary_operator,
    output_checks_and_inference, propagate_shape_matmul,
    check_input_shape_matmul, not_implemented_types, concatenate_shapes,
    types_match_exactly, initializer_operator, reduce_axis,
    output_shape_from_einsum, output_type, allow_broadcasting,
    array_is_square_matrix, determinant_output_shape, broadcast_to,
    flatten_shape, check_input_shape_gemm, propagate_shape_gemm,
    force_evaluation, reshape_check, register, check_axis_is_valid,
    mark_as_optional)
from .types import (bool_types, float_types, all_types, integer_types,
                    numeric_types, signed_integer_types, numpy_to_onnx,
                    unsigned_integer_types, NumericType)
from .shapes import ShapeLike, as_shape, Shape, DynamicDimension, DynamicShape
from . import array
from .exceptions import InternalException
from typing import List, Any, Union, Optional
from typing import Iterable as IterableType
from collections.abc import Iterable
import numpy as np


@register
def absolute(x):
    @allowed_types([*float_types, *integer_types])
    def absolute_helper(x):
        return unary_operator(x, "Abs")
    return absolute_helper(x)


@register
def acos(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_acos(x):
        return unary_operator(x, "Acos")
    return helper_acos(x)


@register
def acosh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_acos(x):
        return unary_operator(x, "Acosh")
    return helper_acos(x)


@register
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


def argmax(x: "array.Array", axis: int = 0, keepdims: bool = True,
           select_last_index: bool = False):
    @allowed_types(numeric_types)
    @not_implemented_types(
        [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16,
         np.int64])
    @output_checks_and_inference(
        reduce_axis([axis], bool(keepdims))
    )
    def argmax_helper(x: "array.Array", axis: int, keepdims: int,
                      select_last_index: int):
        result = nary_operator(
            "ArgMax", x, axis=axis, keepdims=keepdims,
            select_last_index=select_last_index)
        result._dtype = np.int64
        return result
    return argmax_helper(x, axis, int(keepdims), int(select_last_index))


def argmin(x: "array.Array", axis: int = 0, keepdims: bool = True,
           select_last_index: bool = False):
    @allowed_types(numeric_types)
    @not_implemented_types(
        [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16,
         np.int64])
    @output_checks_and_inference(
        reduce_axis([axis], bool(keepdims))
    )
    def argmin_helper(x: "array.Array", axis: int, keepdims: int,
                      select_last_index: int):
        result = nary_operator(
            "ArgMin", x, axis=axis, keepdims=keepdims,
            select_last_index=select_last_index)
        result._dtype = np.int64
        return result
    return argmin_helper(x, axis, int(keepdims), int(select_last_index))


@register
def asin(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_asin(x):
        return unary_operator(x, "Asin")
    return helper_asin(x)


@register
def asinh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def helper_asinh(x):
        return unary_operator(x, "Asinh")
    return helper_asinh(x)


@register
def atan(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def atan_helper(x):
        return unary_operator(x, "Atan")
    return atan_helper(x)


@register
def atanh(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def atanh_helper(x):
        return unary_operator(x, "Atanh")
    return atanh_helper(x)


def bitshift(x: "array.Array", y: "array.Array", direction: str):

    if direction.upper() not in ["RIGHT", "LEFT"]:
        raise ValueError("Bitshift direction should be either RIGHT or LEFT")

    @allowed_types(unsigned_integer_types, unsigned_integer_types)
    @not_implemented_types([np.uint16], [np.uint16])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def helper_bitshift(x: "array.Array", y: "array.Array", direction: str):
        return nary_operator("BitShift", x, y, direction=direction)

    return helper_bitshift(x, y, direction=direction.upper())


def right_shift(x: "array.Array", y: "array.Array"):
    return bitshift(x, y, "RIGHT")


def left_shift(x: "array.Array", y: "array.Array"):
    return bitshift(x, y, "LEFT")


def cast(array: "array.Array", to: type):
    @allowed_types([*all_types])
    @output_checks_and_inference(
        output_type_is_argn_position(1)
    )
    def cast_helper(array: "array.Array", to: type):  # type: ignore
        return unary_operator(array, "Cast", to=numpy_to_onnx(np.dtype(to)))

    return cast_helper(array, to)


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
    return clip_helper(
        input, array.array(minimum, input.dtype),
        array.array(maximum, input.dtype))


def compress(
        x: "array.Array", condition: "array.Array", axis: Optional[int] = None):

    if axis is not None and (axis < - x.ndims or axis >= x.ndims):
        raise ValueError()

    if x.ndims < 1:
        raise ValueError("Expected Array of rank >= 1")
    if condition.ndims != 1:
        raise ValueError("Expected condition Array to be 1D")
    if condition.shape.can_be_static():
        if axis:
            input_dim = x.shape[axis]
            if input_dim.is_static():
                # check shape length is less than the input length along axis
                if int(condition.shape[0]) > int(input_dim):
                    raise ValueError(
                        f"condition array shape {condition.shape} is greater "
                        f"than input shape {x.shape} along axis {axis}")
        else:
            # check shape length is less than the flattened input size if axis is not
            # specified
            if x.shape.can_be_static():
                input_flat_size = x.shape.size()
                if int(condition.shape[0]) > input_flat_size:
                    raise ValueError(
                        f"condition array shape {condition.shape} is greater "
                        f"than input size {input_flat_size}")

    @allowed_types(all_types, bool_types)
    def compress_helper(
            x: "array.Array", condition: "array.Array", axis: Optional[int]):
        result = nary_operator("Compress", x, condition, axis=axis)
        # we don't know how many inputs will meet condition
        # so need to add a dynamic dimension where relevant
        if axis is not None:
            input_shape = x.shape.tolist()
            input_shape[axis] = -1
            result._dims = DynamicShape(*input_shape)
        else:
            result._dims = DynamicShape(-1)
        return result

    return compress_helper(x, condition, axis=axis)


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

    def check_array_shape_and_type(
            values: "array.Array", expected_ndims: int, expected_dtype: np.dtype,
            argname: str):
        shape = values.shape
        dtype = values.dtype
        if len(shape) != expected_ndims:
            raise ValueError(
                f"Argument {argname} expects a {expected_ndims}D array, "
                f"but got {len(shape)}D array")
        if dtype != expected_dtype:
            raise ValueError(
                f"Argument {argname} expects an array of type {expected_dtype}, "
                f"but got type {dtype}")
        return shape, dtype

    if sparse_value:
        raise NotImplementedError("Sparse matrices are currently not supported")
    if value:
        a = array.array(value)
        return initializer_operator("Constant", a.shape, a.dtype, value=a)
    if value_float:
        # TODO: should this be strict? i.e. if not isinstance(value, float) throw?
        a = array.array([float(value_float)])
        return initializer_operator("Constant", a.shape, a.dtype, value_float=a)
    if value_floats:
        shape, dtype = check_array_shape_and_type(
            array.array(value_floats),
            1, np.float32, "value_floats")
        return initializer_operator(
            "Constant", shape, dtype, value_floats=array.array(value_floats))
    if value_int:
        # TODO: should this be strict? i.e. if not isinstance(value, int) throw?
        a = array.array([int(value_int)])
        return initializer_operator("Constant", a.shape, a.dtype, value_int=a)
    if value_ints:
        shape, dtype = check_array_shape_and_type(
            array.array(value_ints),
            1, np.int32, "value_ints")
        return initializer_operator(
            "Constant", shape, dtype, value_ints=array.array(value_ints))
    if value_string:  # pragma: no cover
        raise NotImplementedError("Strings are currently not implemented")
        return initializer_operator(
            "Constant", len(value_string),
            np.string, value_string=value_string)
    if value_strings:  # pragma: no cover
        raise NotImplementedError("Strings are currently not implemented")
        shape, dtype = check_array_shape_and_type(
            value_strings, 1, np.STRING, "value_strings")
        return initializer_operator(
            "Constant", shape, dtype, value_strings=value_strings)
    else:  # pragma: no cover
        raise ValueError("?")


def constant_of_shape(shape: ShapeLike, value=0.0,
                      allow_shape_evaluation: bool = False) -> array.Array:
    """
    Generate a tensor with given value and shape.

    Args:
        shape (ShapeLike): The shape of the tensor.
        value (float, optional): The value of all elements of tensor. Defaults to 0.0.
        allow_shape_evaluation (bool, optional): Whether to explicitly evaluate
            shape if not explicitly known. A shape may not be known at runtime if it is
            the result of a previous operation. This can result in a potentially
            expensive computation to determine the values of shape. Defaults to False.

    Raises:
        ValueError: The shape is not a 1D tensor
        ValueError: The caller did not enable shape evaluation and the shape cannot be
            determined without explicit evaluation

    Returns:
        array.Array: Output tensor of shape specified by shape. If `value` is specified,
            the value and datatype of the output tensor is taken from `value`.
            If attribute `value` is not specified, the value in the output defaults to
            0, and the datatype defaults to `float32`.
    """
    if not isinstance(value, Iterable):
        value = array.array([value])
    if len(value) != 1:
        raise ValueError("Value must be a one-dim tensor with a single element")
    shape_ = as_shape(shape)
    if shape_.can_be_static():
        shape_array = array.array(shape_.to_static().tolist(), dtype=np.int64)
    elif allow_shape_evaluation:
        shape_array = force_evaluation(
            shape_.asarray(),
            "shape", allow_shape_evaluation)
    else:
        raise ValueError("Could not determine shape")
    output = nary_operator("ConstantOfShape", shape_array, value=value)
    output._dtype = value.dtype
    output._dims = DynamicShape(*shape_array.values())
    return output


@register
def cos(x):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def cos_helper(x):
        return unary_operator(x, "Cos")
    return cos_helper(x)


@register
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


@register
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


def einsum(*inputs, equation):
    @not_implemented_types([*unsigned_integer_types, np.int8, np.int16])
    @allowed_types([*float_types, *integer_types])
    @output_checks_and_inference(
        output_shape_from_einsum("equation")
    )
    def einsum_helper(*inputs, equation):
        return nary_operator("Einsum", *inputs, equation=equation)

    return einsum_helper(*inputs, equation=equation)


def equal(x, y):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @allowed_types(numeric_types, numeric_types)
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
        return nary_operator("Erf", x)

    return erf_helper(x)


@register
def exp(x):
    @allowed_types(float_types)
    def exp_helper(x):
        return unary_operator(x, "Exp")

    return exp_helper(x)


def expand(x, shape: ShapeLike):
    # we have to evaluate the graph here (if necessary)
    # TODO: do we?
    array_shape = as_shape(shape)

    @allowed_types(all_types, [np.int64])
    @output_checks_and_inference(
        broadcast_to(array_shape)
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
        raise NotImplementedError(
            f"Output type {dtype} currently not implemented")

    @allowed_types(all_types)
    @not_implemented_types([np.int8, np.int16, np.uint8, np.uint16, np.uint32,
                            np.bool_])
    @output_checks_and_inference(
        output_type(dtype)
    )
    def eye_like_helper(x: "array.Array", dtype: int, k: int):
        return unary_operator(x, "EyeLike", dtype=dtype, k=k)

    return eye_like_helper(x, dtype=numpy_to_onnx(np.dtype(dtype)), k=int(k))


def flatten(x: "array.Array", axis: int = 1):

    @allowed_types(all_types)
    @output_checks_and_inference(
        flatten_shape(int(axis))
    )
    def flatten_helper(x: "array.Array", axis: int):
        return unary_operator(x, "Flatten", axis=axis)

    return flatten_helper(x, axis=int(axis))


def floor(x: "array.Array"):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def floor_helper(x: "array.Array"):
        return unary_operator(x, "Floor")
    return floor_helper(x)


def gemm(a: "array.Array", b: "array.Array", c: Optional["array.Array"] = None,
         alpha: float = 1.0, beta: float = 1.0, transA: bool = False,
         transB: bool = False):

    @allowed_types([*float_types, np.int32, np.int64, np.uint32, np.uint64])
    @not_implemented_types([np.int32, np.int64, np.uint32, np.uint64])
    @check_input_shape_gemm
    @output_checks_and_inference(
        propagate_shape_gemm
    )
    def gemm_helper(
            a: "array.Array", b: "array.Array", c: "array.Array", alpha: float,
            beta: float, transA: int, transB: int):
        return nary_operator(
            "Gemm", a, b, c, alpha=alpha, beta=beta, transA=transA,
            transB=transB)

    if a.dtype != b.dtype and (c is not None and a.dtype != c.dtype):
        raise TypeError(
            f"Type of A ({a.dtype}) must match type of B ({b.dtype}) and C ({c.dtype})")

    return gemm_helper(
        a, b, c, alpha=float(alpha),
        beta=float(beta),
        transA=int(transA),
        transB=int(transB))


def greater(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_greater(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "Greater")
    return helper_greater(a, b)


def greater_equal(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_greater_equal(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "GreaterOrEqual")
    return helper_greater_equal(a, b)


def identity(x: "array.Array"):
    @allowed_types(all_types)
    def helper_identity(x: "array.Array"):
        return unary_operator(x, "Identity")

    return helper_identity(x)


def isinf(x: "array.Array", detect_negative: bool = True,
          detect_positive: bool = True):
    @allowed_types(float_types)
    @output_checks_and_inference(
        output_type(np.bool_)
    )
    def helper_isinf(
            x: "array.Array", detect_negative: int, detect_positive: int):
        return unary_operator(
            x, "IsInf", detect_negative=detect_negative,
            detect_positive=detect_positive)

    return helper_isinf(
        x, detect_negative=int(detect_negative),
        detect_positive=int(detect_positive))


def isneginf(x: "array.Array"):
    return isinf(x, detect_positive=False)


def isposinf(x: "array.Array"):
    return isinf(x, detect_negative=False)


def isnan(x: "array.Array"):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @output_checks_and_inference(
        output_type(np.bool_)
    )
    def helper_isnan(x: "array.Array"):
        return unary_operator(x, "IsNaN")

    return helper_isnan(x)


def less(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_less(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "Less")
    return helper_less(a, b)


def less_equal(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_less_equal(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "LessOrEqual")
    return helper_less_equal(a, b)


@register
def log(x: "array.Array"):
    @allowed_types(float_types)
    def helper_log(x: "array.Array"):
        return unary_operator(x, "Log")

    return helper_log(x)


@register
def matmul(x: "array.Array", y: "array.Array"):
    @allowed_types([*float_types, np.int32, np.int64, np.uint32, np.uint64],
                   [*float_types, np.int32, np.int64, np.uint32, np.uint64])
    @check_input_shape_matmul
    @output_checks_and_inference(
        propagate_shape_matmul()
    )
    def matmul_helper(x, y):
        return binary_operator(x, y, "MatMul")
    return matmul_helper(x, y)


def matmul_integer(A: "array.Array", B: "array.Array",
                   a_zero_point: "array.Array" = None,
                   b_zero_point: "array.Array" = None):
    if len(A.shape) > 1:
        A_cols = A.shape[-2]  # type: ignore
    elif len(A.shape) == 1:
        A_cols = A.shape[0]
    else:
        A_cols = DynamicDimension(0)
    B_rows = B.shape[0] if len(B.shape) > 0 else DynamicDimension(0)

    if a_zero_point is None:
        a_zero_point = array.array(0, dtype=A.dtype)
    elif a_zero_point.ndims == 1 and a_zero_point.shape[0] != A_cols:
        raise ValueError(
            "The A zero point 1D tensor must match the number of rows of A")
    elif a_zero_point.dtype != A.dtype:
        raise TypeError(
            f"The A zero points ({a_zero_point.dtype}) must have the same type as A "
            f"({A.dtype})")

    if b_zero_point is None:
        b_zero_point = array.array(0, dtype=B.dtype)
    elif b_zero_point.ndims == 1 and b_zero_point.shape[0] != B_rows:
        raise ValueError(
            "The B zero point 1D tensor must match the number of rows of B")
    elif b_zero_point.dtype != B.dtype:
        raise TypeError(
            f"The B zero points ({b_zero_point.dtype}) must have the same type as B "
            f"({B.dtype})")

    if (A.dtype == np.int8 and B.dtype == np.uint8) or \
       (A.dtype == np.int8 and B.dtype == np.int8):
        # uint8-int8 and int8-int8 known to not be implemented
        raise NotImplementedError(
            f"Combination A B matrix types int8-uint8 and int8-int8 are not "
            f"implemented. Got {A.dtype}-{B.dtype}")

    @allowed_types([np.uint8, np.int8],
                   [np.uint8, np.int8])
    @check_input_shape_matmul
    @output_checks_and_inference(
        propagate_shape_matmul(),
        output_type(np.int32)
    )
    def matmul_integer_helper(A, B, a_zero_point, b_zero_point):
        return nary_operator("MatMulInteger", A, B, a_zero_point, b_zero_point)

    return matmul_integer_helper(A, B, a_zero_point, b_zero_point)


def max(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_max(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceMax", x, axes=axes.values(), keepdims=keepdims)

    return helper_max(
        x, axes, keepdims=bool(keepdims))


def maximum(*arrays):
    @allowed_types(numeric_types)
    @not_implemented_types([np.uint8, np.uint16, np.int8, np.int16])
    @types_match_exactly
    @output_checks_and_inference(
        allow_broadcasting
    )
    def helper_maximum(*arrays):
        return nary_operator("Max", *arrays)

    return helper_maximum(*arrays)


def elementwise_mean(*inputs: "array.Array"):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @types_match_exactly
    @output_checks_and_inference(
        allow_broadcasting
    )
    def mean_helper(*inputs: "array.Array"):
        return nary_operator("Mean", *inputs)
    return mean_helper(*inputs)


def mean_variance_normalization(x: "array.Array", axes: List[int] = [0, 2, 3]):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def mean_variance_normalization_helper(x: "array.Array", axes: List[int]):
        return nary_operator("MeanVarianceNormalization", x, axes=axes)
    return mean_variance_normalization_helper(x, axes=axes)


def minimum(*arrays: "array.Array"):
    @allowed_types(numeric_types)
    @not_implemented_types([np.uint8, np.uint16, np.int8, np.int16])
    @types_match_exactly
    @output_checks_and_inference(
        allow_broadcasting
    )
    def helper_minimum(*arrays):
        return nary_operator("Min", *arrays)

    return helper_minimum(*arrays)


def min(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_min(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceMin", x, axes=axes.values(), keepdims=keepdims)

    return helper_min(
        x, axes, keepdims=bool(keepdims))


def mod(x: "array.Array", y: "array.Array", fmod: bool = False):
    if x.dtype in float_types:
        fmod = True

    @allowed_types(numeric_types, numeric_types)
    # @not_implemented_types([np.uint8, np.uint16, np.int8, np.int16])
    @types_match_exactly
    @output_checks_and_inference(
        allow_broadcasting
    )
    def helper_mod(x: "array.Array", y: "array.Array", fmod: bool):
        return nary_operator("Mod", x, y, fmod=fmod)

    return helper_mod(x, y, fmod=bool(fmod))


@register
def multiply(x: "array.Array", y: "array.Array"):
    @not_implemented_types([np.uint32, np.uint64], [np.uint32, np.uint64])
    @allowed_types([*float_types, np.int32, np.int64],
                   [*float_types, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def multiply_helper(x: "array.Array", y: "array.Array"):
        return binary_operator(x, y, "Mul")
    return multiply_helper(x, y)


def negative(x: "array.Array"):
    @allowed_types([*float_types, *signed_integer_types])
    @not_implemented_types([np.int16])
    def helper_negative(x):
        return nary_operator("Neg", x)

    return helper_negative(x)


def nonzero(x: "array.Array"):
    @allowed_types(all_types)
    @not_implemented_types([np.float64, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    def nonzero_helper(x: "array.Array"):
        result = unary_operator(x, "NonZero")
        result._dtype = np.int64
        result._dims = DynamicShape(x.ndims, -1)
        return result
    return nonzero_helper(x)


def not_(x: "array.Array"):
    @allowed_types(bool_types)
    def not_helper(x: "array.Array"):
        return unary_operator(x, "Not")
    return not_helper(x)


def one_hot(
        indices: "array.Array", depth: "array.Array", values: "array.Array",
        axis: int = -1):

    if axis < -indices.ndims and axis >= indices.ndims:
        raise ValueError(
            f"Axis must be in the range [-{indices.ndims}, {indices.ndims-1}]")

    IMPLEMENTED_COMBINATIONS = [[np.float32, np.float32, np.float32],
                                [np.int64, np.int64, np.float32],
                                [np.int64, np.int64, np.int64]]

    if [indices.dtype, depth.dtype, values.dtype] not in IMPLEMENTED_COMBINATIONS:
        raise ValueError(
            "Array type combination not implemented. Implemented combinations are: "
            f"{', '.join(str(comb) for comb in IMPLEMENTED_COMBINATIONS)}")

    @allowed_types(numeric_types, numeric_types, all_types)
    def one_hot_helper(
            indices: "array.Array", depth: "array.Array", values: "array.Array",
            axis: int):
        result = nary_operator("OneHot", indices, depth, values, axis=axis)
        output_shape: List[Optional[int]] = [None] * (len(indices.shape) + 1)
        dim_to_insert = -1 if array.is_lazy(depth) else int(depth.item())
        axis = len(indices.shape) + axis + 1 if axis < 0 else axis

        indices_idx = 0
        for idx in range(len(output_shape)):
            if idx == axis:
                output_shape[idx] = dim_to_insert
            else:
                output_shape[idx] = indices.shape[indices_idx]
                indices_idx += 1

        if any(map(lambda el: el is None, output_shape)):
            raise InternalException(
                f"Invalid output shape {output_shape} for one hot encoding!")
        # line below is ignored because mypy cannot figure out that output_shape cannot
        # have None at this point
        result._dims = DynamicShape(*output_shape)  # type: ignore
        result._dtype = values.dtype

        return result

    return one_hot_helper(indices, depth, values, axis=axis)


def logical_or(x, y):
    @allowed_types(bool_types, bool_types)
    @output_checks_and_inference(
        allow_broadcasting
    )
    def logical_or_helper(x, y):
        return binary_operator(x, y, "Or")
    return logical_or_helper(x, y)


def pad(x: "array.Array", pads: "array.Array",
        constant_value: Union["array.Array", str, int, bool] = 0):

    # TODO: fix this when dynamic shapes are implemented
    raise NotImplementedError()

    if isinstance(constant_value, str):
        # TODO
        raise NotImplementedError("")
    elif isinstance(constant_value, array.Array) and constant_value.ndims > 1:
        raise ValueError("Expected a scalar or 1D array")
    else:
        constant_value = array.array(constant_value)

    def pad_helper(x: "array.Array", pads: "array.Array",
                   constant_value: "array.Array"):
        result = nary_operator("Pad", x, pads, constant_value)
        result._dims = (-1) * result.ndims
        return result

    return pad_helper(x, pads, constant_value)


@register
def power(x: "array.Array", y: "array.Array"):
    @allowed_types([*float_types, np.int32, np.int64], numeric_types)
    @not_implemented_types([],
                           [np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def helper_power(x: "array.Array", y: "array.Array"):
        return binary_operator(x, y, "Pow")

    return helper_power(x, y)


def prod(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_prod(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceProd", x, axes=axes.values(), keepdims=keepdims)

    return helper_prod(
        x, axes, keepdims=bool(keepdims))


def reciprocal(x):
    @allowed_types(float_types)
    def reciprocal_helper(x):
        return unary_operator(x, "Reciprocal")
    return reciprocal_helper(x)


def reshape(x: "array.Array", shape: ShapeLike, allowzero: bool = False):

    @allowed_types(all_types)
    @output_checks_and_inference(
        reshape_check(as_shape(shape))
    )
    def helper_reshape(x: "array.Array", shape: Shape, allowzero: int):

        if allowzero == 1:
            import warnings
            warnings.warn("Ignoring currently not supported `allowzero`")

        # TODO: fixme when upgrading to opset 14
        # result = nary_operator("Reshape", x, shape.asarray(), allowzero=allowzero)
        result = nary_operator("Reshape", x, shape.asarray())
        result._dims = shape
        return result

    return helper_reshape(x, as_shape(shape), int(allowzero))


def sum(
        x: "array.Array", axes: Optional[Union[int, List[int],
                                               "array.Array"]] = None,
        keepdims: bool = True, noop_with_empty_axes: bool = False):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0) \
            and not noop_with_empty_axes:
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)
    elif isinstance(axes, Iterable) and not isinstance(axes, array.Array):
        axes = array.array(axes, np.int64)

    if noop_with_empty_axes:
        import warnings
        warnings.warn("option noop_with_empty_axes is currently unstable.")

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([],
                           [np.uint8, np.uint16, np.uint32, np.uint64, np.int8,
                            np.int16])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_sum(
            x: "array.Array", axes: "array.Array", keepdims: bool,
            noop_with_empty_axes: bool):
        if len(axes) == 0 and not noop_with_empty_axes:
            axes = None  # type: ignore
        return nary_operator(
            "ReduceSum", x, axes, keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes)

    return helper_sum(
        x, axes, keepdims=bool(keepdims),
        noop_with_empty_axes=bool(noop_with_empty_axes))


@register
def mean(
        x: "array.Array",
        axes: Optional[Union[int, List[int],
                             "array.Array"]] = None, keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)
    elif isinstance(axes, Iterable) and not isinstance(axes, array.Array):
        axes = array.array(axes, np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_mean(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceMean", x, axes=axes.values(), keepdims=keepdims)

    return helper_mean(x, axes, keepdims=bool(keepdims))


def sum_square(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_sum_square(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceSumSquare", x, axes=axes.values(), keepdims=keepdims)

    return helper_sum_square(
        x, axes, keepdims=bool(keepdims))


def arange(start: Union[NumericType, "array.Array"],
           limit: Union[NumericType, "array.Array"],
           delta: Union[NumericType, "array.Array"],
           allow_evaluation: bool = False):

    if not isinstance(start, array.Array):
        start = array.array(start)
    if not isinstance(limit, array.Array):
        limit = array.array(limit)
    if not isinstance(delta, array.Array):
        delta = array.array(delta)

    if len(start.shape) != 0:
        raise ValueError("Start value should be a scalar")
    if len(limit.shape) != 0:
        raise ValueError("Limit value should be a scalar")
    if len(delta.shape) != 0:
        raise ValueError("Delta value should be a scalar")

    @allowed_types([*float_types, np.int16, np.int32, np.int64],
                   [*float_types, np.int16, np.int32, np.int64],
                   [*float_types, np.int16, np.int32, np.int64])
    @not_implemented_types([np.int16], [np.int16], [np.int16])
    @types_match_exactly
    def arange_helper(start: "array.Array", limit: "array.Array",
                      delta: "array.Array"):
        result = nary_operator("Range", start, limit, delta)
        # FIXME: use dynamic shape?
        result._dtype = start.dtype
        # here we convert to float32 since int division truncates
        # abs(limit - start) / abs(delta)
        # in onp (10 - 6) / 3 == 1, but we want 2 (rounded up from 1.33)
        # so (10.0 - 6.0) / 3.0 == 1.33, then ceil(1.33) == 2.0
        # and in the end convert to int (since dimensions are always int)
        if allow_evaluation:
            result._dims = DynamicShape(
                int(ceil(
                    (abs(cast(limit, np.float32) - cast(start, np.float32))
                     / abs(cast(delta, np.float32)))).item()),)
        else:
            result._dims = DynamicShape(-1)
        return result

    return arange_helper(start, limit, delta)


def l1_norm(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_l1_norm(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceL1", x, axes=axes.values(), keepdims=keepdims)

    return helper_l1_norm(
        x, axes, keepdims=bool(keepdims))


def l2_norm(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_l2_norm(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceL2", x, axes=axes.values(), keepdims=keepdims)

    return helper_l2_norm(
        x, axes, keepdims=bool(keepdims))


def log_sum(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_log_sum(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceLogSum", x, axes=axes.values(), keepdims=keepdims)

    return helper_log_sum(
        x, axes, keepdims=bool(keepdims))


def log_sum_exp(
        x: "array.Array", axes: Optional[Union[int, List[int], "array.Array"]] = None,
        keepdims: bool = True):

    if axes is None or (isinstance(axes, Iterable) and len(axes) == 0):
        axes = array.array(range(x.ndims), np.int64)
    elif isinstance(axes, int):
        axes = array.array([axes], np.int64)

    @allowed_types([*float_types, np.int32, np.int64], [np.int64])
    @not_implemented_types([np.int64, np.float64], [])
    @output_checks_and_inference(
        reduce_axis(axes, bool(keepdims))
    )
    def helper_log_sum_exp(
            x: "array.Array", axes: "array.Array", keepdims: bool):
        return nary_operator(
            "ReduceLogSumExp", x, axes=axes.values(), keepdims=keepdims)

    return helper_log_sum_exp(
        x, axes, keepdims=bool(keepdims))


def interpolate(x: "array.Array", roi: Optional["array.Array"] = None,
                scales: Optional["array.Array"] = None,
                sizes: Optional["array.Array"] = None,
                coordinate_transformation_mode: str = "half_pixel",
                cubic_coeff_a: float = -0.75, exclude_outside: bool = False,
                extrapolation_value: float = 0.0, mode: str = "nearest",
                nearest_mode: str = "round_prefer_floor"):

    AVAILABLE_TRANSFORMATION_MODES = [
        "half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric",
        "tf_crop_and_resize"]

    if coordinate_transformation_mode.lower() not in AVAILABLE_TRANSFORMATION_MODES:
        raise ValueError(
            "coordinate_transformation_mode must be one of "
            f"{', '.join(AVAILABLE_TRANSFORMATION_MODES)}, but got "
            f"{coordinate_transformation_mode}")

    if mode.lower() not in ["nearest", "linear", "cubic"]:
        raise ValueError(
            f"mode must be either nearest, linear or cubic, but got {mode}")

    if nearest_mode.lower() not in [
            "round_prefer_floor", "round_prefer_ceil", "floor", "ceil"]:
        raise ValueError(
            "nearest_mode must be one of round_prefer_floor, round_prefer_ceil, "
            f"floor or ceil, but got {nearest_mode}")

    if roi is not None and roi.ndims != 1:
        raise ValueError(f"roi must be of rank 1, but got rank {roi.ndims}")

    if not ((scales is not None) ^ (sizes is not None)):
        raise ValueError("Either scales or sizes array must be specified")

    @allowed_types(all_types, float_types, [np.float32], [np.int64])
    @not_implemented_types(
        [np.float64, np.uint16, np.uint32, np.uint64, np.int8,
         np.int16, np.int64, np.bool_], [np.float64])
    def interpolate_helper(x: "array.Array", roi: Optional["array.Array"],
                              scales: Optional["array.Array"],
                              sizes: Optional["array.Array"],
                              coordinate_transformation_mode: str,
                              cubic_coeff_a: float,
                              exclude_outside: int, extrapolation_value: float,
                              mode: str, nearest_mode: str):

        if roi is not None and roi.shape[0].is_static() and len(roi) != x.ndims * 2:
            raise ValueError(
                f"Expected roi size to be {x.ndims * 2}, but got {len(roi)}")

        if scales is not None and \
                scales.shape[0].is_static() and len(scales) != x.ndims:
            raise ValueError(
                f"scales must have size equal to the rank as x, but got size "
                f"{scales.ndims}")

        if sizes is not None and sizes.shape[0].is_static() and len(sizes) != x.ndims:
            raise ValueError(
                f"sizes must have size equal to the rank as x, but got size "
                f"{sizes.ndims}")

        return nary_operator(
            "Resize", x, roi, scales, sizes,
            coordinate_transformation_mode=coordinate_transformation_mode,
            cubic_coeff_a=cubic_coeff_a, exclude_outside=exclude_outside,
            extrapolation_value=extrapolation_value, mode=mode,
            nearest_mode=nearest_mode)

    return interpolate_helper(
        x, roi, scales, sizes,
        coordinate_transformation_mode=coordinate_transformation_mode.lower(),
        cubic_coeff_a=float(cubic_coeff_a),
        exclude_outside=int(exclude_outside),
        extrapolation_value=float(extrapolation_value),
        mode=mode.lower(), nearest_mode=nearest_mode.lower())


def round(x: "array.Array"):
    @allowed_types(float_types)
    def round_helper(x: "array.Array"):
        return nary_operator("Round", x)
    return round_helper(x)


def shape(x: "array.Array") -> "array.Array":
    """Takes a tensor as input and outputs an 1D int64 tensor containing the shape of
    the input tensor.

    Note that Array.shape is more efficient (and should give the same result).
    The difference is that this `shape` free function adds the `Shape` node to the
    `ONNX` graph, which could improve runtime if the output is used in subsequent
    operations.

    Args:
        x (array.Array): Input tensor

    Returns:
        array.Array: Shape of the input tensor
    """
    @allowed_types(all_types)
    def shape_helper(x: "array.Array"):
        result = nary_operator("Shape", x)
        result._dims = DynamicShape(len(x.shape),)
        result._dtype = np.int64
        return result

    return shape_helper(x)


def sign(x: "array.Array") -> "array.Array":
    @allowed_types(numeric_types)
    def sign_helper(x: "array.Array") -> "array.Array":
        return nary_operator("Sign", x)
    return sign_helper(x)


@register
def sin(x) -> "array.Array":
    @allowed_types([*float_types])
    def helper_sin(x):
        return unary_operator(x, "Sin")
    return helper_sin(x)


@register
def sinh(x) -> "array.Array":
    @allowed_types([*float_types])
    @not_implemented_types([np.float64])
    def helper_sin(x):
        return unary_operator(x, "Sinh")
    return helper_sin(x)


def size(x: "array.Array") -> "array.Array":
    """Takes a tensor as input and outputs a int64 scalar that equals to the total
    number of elements of the input tensor.

    Note that len(x) is more efficient (and should give the same result).
    The difference is that this `size` free function adds the `Size` node to the
    `ONNX` graph, which could improve runtime if the output is used in subsequent
    operations. It will also know the tensor size at runtime (which may not be
    known when the graph is declared, i.e. when using len(x)).

    Args:
        x (array.Array): Input tensor

    Returns:
        array.Array: Size of the input tensor
    """
    @allowed_types(all_types)
    def size_helper(x: "array.Array"):
        result = nary_operator("Size", x)
        result._dims = DynamicShape()
        result._dtype = np.int64
        return result

    return size_helper(x)


def slice(x: "array.Array", starts: "array.Array", ends: "array.Array",
          axes: Optional["array.Array"] = None, steps: Optional["array.Array"] = None):

    if (starts.dtype != ends.dtype or (axes and starts.dtype != axes.dtype)
            or (steps and starts.dtype != steps.dtype)):
        raise ValueError("types of starts, ends, axes and steps must match")

    if starts.ndims != 1:
        raise ValueError("starts must be a 1D array")

    if ends.ndims != 1:
        raise ValueError("ends must be a 1D array")

    if axes and axes.ndims != 1:
        raise ValueError("axes must be a 1D array")
    # elif not axes:
    #     axes = array.array(list(range(starts)), dtype=starts.dtype)

    if steps and steps.ndims != 1:
        raise ValueError("steps must be a 1D array")
    # elif not steps:
    #     steps = array.array([1] * starts.ndims, dtype=starts.dtype)

    @allowed_types(all_types, [np.int32, np.int64],
                   [np.int32, np.int64],
                   [np.int32, np.int64],
                   [np.int32, np.int64])
    def slice_helper(
            x: "array.Array", starts: "array.Array", ends: "array.Array",
            axes: Optional["array.Array"], steps: Optional["array.Array"]):
        result = nary_operator("Slice", x, starts, ends, axes, steps)

        # TODO: infer shape
        # if (array.is_lazy(starts) or array.is_lazy(ends)
        #         or (axes and array.is_lazy(axes))
        #         or (steps and array.is_lazy(steps))):
        result._dims = DynamicShape(*[-1 for _ in range(x.ndims)])

        return result

    return slice_helper(x, starts, ends, axes, steps)


@register
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


def square(x: "array.Array") -> "array.Array":
    return x * x


@register
def sqrt(x: "array.Array") -> "array.Array":
    @allowed_types(float_types)
    def sqrt_helper(x: "array.Array") -> "array.Array":
        return nary_operator("Sqrt", x)
    return sqrt_helper(x)


@register
def tan(x: "array.Array") -> "array.Array":
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def tan_helper(x: "array.Array") -> "array.Array":
        return nary_operator("Tan", x)
    return tan_helper(x)


@register
def tanh(x: "array.Array") -> "array.Array":
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def tanh_helper(x: "array.Array") -> "array.Array":
        return nary_operator("Tanh", x)
    return tanh_helper(x)


def transpose(x: "array.Array", perm: Optional[List[int]] = None):
    if perm is None:
        perm = list(reversed(range(len(x.shape))))

    @allowed_types(all_types)
    def transpose_helper(x: "array.Array", perm: List[int]):
        result = nary_operator("Transpose", x, perm=perm)
        result._dims = DynamicShape(*map(lambda idx: x.shape[idx], perm))
        return result

    return transpose_helper(x, perm)


def logical_xor(a: "array.Array", b: "array.Array"):

    @allowed_types(bool_types, bool_types)
    @output_checks_and_inference(
        allow_broadcasting
    )
    def logical_xor_helper(a: "array.Array", b: "array.Array"):
        return nary_operator("Xor", a, b)

    return logical_xor_helper(a, b)


def squeeze(x: "array.Array", axes: "array.Array" = None):

    @allowed_types(all_types, [np.int64])
    def squeeze_helper(x: "array.Array", axes: "array.Array"):
        output_shape = x.shape
        if array.is_lazy(axes):
            output_shape = DynamicShape(
                *[DynamicDimension(-1)] * (x.ndims - int(axes.shape[0])))
        else:
            axis_to_remove = []
            for axis in axes:
                check_axis_is_valid(x, axis)
                if x.shape[axis] != 1:
                    raise ValueError(
                        f"cannot select axis ({axis}) to squeeze out "
                        "which has size not equal to one in shape "
                        f"{x.shape}")
                axis = axis if axis >= 0 else x.ndims + axis

                axis_to_remove.append(axis)
            output_shape = DynamicShape(
                *[s for idx, s in enumerate(output_shape)
                    if idx not in axis_to_remove])

        result = nary_operator("Squeeze", x, axes)
        result._dims = output_shape
        return result

    return squeeze_helper(x, axes)


def tile(x: "array.Array", repeats: "array.Array"):

    if repeats.ndims != 1:
        raise ValueError(
            f"repeats must be tensor of rank 1, but got tensor of rank {repeats.ndims}")

    @allowed_types(all_types, [np.int64])
    def tile_helper(x: "array.Array", repeats: "array.Array"):
        output_shape = None
        if array.is_lazy(repeats):
            output_shape = DynamicShape(*[-1] * x.ndims)
        else:
            if repeats.size != x.ndims:
                raise ValueError(
                    f"repeats array size {repeats.size} does not match rank of x "
                    f"{x.ndims}")
            repeats_ = repeats.numpy()
            output_shape_ = []
            for i in range(repeats.size):
                if x.shape[i].is_static():
                    if repeats_[i] < 1:
                        raise ValueError(
                            f"repeats must be greater than 0, but got {repeats}")
                    output_shape_.append(int(x.shape[i]) * repeats_[i])
                else:
                    output_shape_.append(-1)
            output_shape = DynamicShape(*output_shape_)

        result = nary_operator("Tile", x, repeats)
        result._dims = output_shape
        return result

    return tile_helper(x, repeats)


def topk(
        x: "array.Array", k: "array.Array", axis: int = -1, largest: bool = True,
        sorted: bool = True):

    @allowed_types(numeric_types, [np.int64])
    @not_implemented_types([*unsigned_integer_types, np.int8, np.int16, np.int32])
    def topk_helper(
            x: "array.Array", k: "array.Array", axis: int, largest: int,
            sorted: int):

        check_axis_is_valid(x, axis)

        values, indices = multi_output_nary_operator(2)(
            "TopK", x, k, axis=axis, largest=largest, sorted=sorted)

        output_shape_ = x.shape.tolist()

        output_shape_[axis] = -1 if array.is_lazy(k) else int(k.item())

        output_shape = DynamicShape(*output_shape_)

        values._dims = output_shape
        indices._dims = output_shape
        indices._dtype = np.int64

        return values, indices

    return topk_helper(
        x, k, axis=axis, largest=int(largest), sorted=int(sorted))


def trilu(
        x: "array.Array", k: Optional["array.Array"] = None, upper: bool = True):

    # TODO: update this when onnxruntime release ONNX opset 14 support
    raise NotImplementedError("Only available from ONNX opset 14")

    if x.ndims < 2:
        raise ValueError(
            "Expected x to be an array with rank greater than 1, but got rank"
            f"{x.ndims}")

    if k and k.ndims != 0:
        raise ValueError(
            f"Expected k to be an array with rank 0, but got rank {k.ndims}")

    @allowed_types(all_types, [np.int64])
    def trilu_helper(x: "array.Array", k: Optional["array.Array"], upper: int):
        return nary_operator("Trilu", x, k, upper=upper)

    return trilu_helper(x, k, upper=int(upper))


def tril(x: "array.Array", k: Optional["array.Array"] = None):
    return trilu(x, k, True)


def triu(x: "array.Array", k: Optional["array.Array"] = None):
    return trilu(x, k, False)


def unique(
        x: "array.Array", return_index: bool = False, return_inverse: bool = False,
        return_counts: bool = False, axis: Optional[int] = None, sorted: bool = True):

    @allowed_types(all_types)
    @not_implemented_types(
        [np.float64, *unsigned_integer_types, np.int16, np.int32, np.bool_])
    def unique_helper(x: "array.Array", axis: Optional[int], sorted: int):

        if axis:
            check_axis_is_valid(x, axis)

        y, indices, inverse_indices, counts = multi_output_nary_operator(
            4)("Unique", x, axis=axis, sorted=sorted)

        mark_as_optional(indices)
        mark_as_optional(inverse_indices)
        mark_as_optional(counts)

        if axis:
            output_shape = y.shape.tolist()
            output_shape[axis] = -1
            y._dims = DynamicShape(*output_shape)

        return_list = [y]
        if return_index:
            indices._dtype = np.int64
            indices._dims = DynamicShape(-1)
            return_list.append(indices)
        if return_inverse:
            inverse_indices._dtype = np.int64
            inverse_indices._dims = DynamicShape(-1)
            return_list.append(inverse_indices)
        if return_counts:
            counts._dtype = np.int64
            counts._dims = DynamicShape(-1)
            return_list.append(counts)

        return (*return_list,)

    return unique_helper(x, axis=axis, sorted=int(sorted))


def unsqueeze(x: "array.Array", axes: "array.Array"):

    @allowed_types(all_types, [np.int64])
    def unsqueeze_helper(x: "array.Array", axes: "array.Array"):
        output_shape = None
        if array.is_lazy(axes):
            output_shape = DynamicShape(
                *[DynamicDimension(-1)] * (x.ndims + int(axes.shape[0])))
        else:
            output_shape_: List[Optional[int]] = [
                None] * (x.ndims + int(axes.shape[0]))
            dims_to_insert = []
            for axis in axes:
                if (int(axis) < -len(output_shape_)
                        or int(axis) > len(output_shape_) - 1):
                    raise ValueError(
                        f"Axis must be in the range [-{len(output_shape_)}, "
                        f"{len(output_shape_)-1}]")
                axis = axis if axis >= 0 else len(output_shape_) + axis
                if axis in dims_to_insert:
                    raise ValueError(f"Repeated axis values in {axes}")
                dims_to_insert.append(axis)

            indices_idx = 0
            for idx in range(len(output_shape_)):
                if idx in dims_to_insert:
                    output_shape_[idx] = 1
                else:
                    output_shape_[idx] = x.shape[indices_idx]
                    indices_idx += 1

            if None in output_shape_:
                raise InternalException("output_shape cannot contain None")

            output_shape = DynamicShape(*output_shape_)  # type: ignore

        result = nary_operator("Unsqueeze", x, axes)
        result._dims = output_shape
        return result

    return unsqueeze_helper(x, axes)


def where(condition: "array.Array", x: "array.Array", y: "array.Array"):

    @allowed_types([np.bool_], all_types, all_types)
    @not_implemented_types(
        [],
        [np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.bool_],
        [np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.bool_])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def where_helper(
            condition: "array.Array", x: "array.Array", y: "array.Array"):
        result = nary_operator("Where", condition, x, y)
        result._dtype = x.dtype

        return result

    return where_helper(condition, x, y)
