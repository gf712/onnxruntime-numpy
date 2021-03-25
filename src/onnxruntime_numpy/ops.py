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
                        flatten_shape, gather_check, check_input_shape_gemm,
                        propagate_shape_gemm, propagate_shape_pool)
from .types import (bool_types, float_types, all_types, integer_types,
                    numeric_types, signed_integer_types, numpy_to_onnx,
                    unsigned_integer_types)
from . import array
from typing import List, Any, Union, Optional
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
    @not_implemented_types([np.float64])
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


def gather_elements(x: "array.Array", indices: "array.Array", axis: int = 0):

    if axis < -x.ndims or axis > x.ndims - 1:
        raise ValueError(
            f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")

    @allowed_types(all_types, [np.int32, np.int64])
    @output_checks_and_inference(
        propagate_shape_from_argn_position(1)
    )
    def gather_elements_helper(x: "array.Array", indices: "array.Array", axis: int):
        return nary_operator("GatherElements", x, indices, axis=axis)

    return gather_elements_helper(x, indices, axis=int(axis))


def gathernd(x: "array.Array", indices: "array.Array", batch_dims: int = 0):

    # if axis < -x.ndims or axis > x.ndims - 1:
    #     raise ValueError(
    #         f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")

    # @allowed_types(all_types, [np.int32, np.int64])
    # @output_checks_and_inference(
    #     propagate_shape_from_argn_position(1)
    # )
    # def gathernd_helper(x: "array.Array", indices: "array.Array", axis: int):
    #     return nary_operator("GatherND", x, indices, axis=axis)

    # return gathernd_helper(x, indices, axis=int(axis))
    raise NotImplementedError()


def gemm(a: "array.Array", b: "array.Array", c: Optional["array.Array"] = None,
         alpha: float = 1.0, beta: float = 1.0, transA: bool = False,
         transB: bool = False):

    @allowed_types([*float_types, np.int32, np.int64, np.uint32, np.uint64])
    @not_implemented_types([np.int32, np.int64, np.uint32, np.uint64])
    @check_input_shape_gemm
    @output_checks_and_inference(
        propagate_shape_gemm
    )
    def gemm_helper(a: "array.Array", b: "array.Array", c: "array.Array", alpha: float,
                    beta: float, transA: int, transB: int):
        return nary_operator("Gemm", a, b, c, alpha=alpha, beta=beta, transA=transA,
                             transB=transB)

    if a.dtype != b.dtype and a.dtype != c.dtype:
        raise TypeError(
            f"Type of A ({a.dtype}) must match type of B ({b.dtype}) and C ({c.dtype})")

    return gemm_helper(a, b, c, alpha=float(alpha), beta=float(beta), transA=int(transA),
                       transB=int(transB))


def global_average_pool(x: "array.Array"):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_pool
    )
    def helper_global_average_pool(x: "array.Array"):
        return unary_operator(x, "GlobalAveragePool")

    return helper_global_average_pool(x)


def global_lp_pool(x: "array.Array", p: int = 2):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    # @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_pool
    )
    def helper_global_lp_pool(x: "array.Array", p: int):
        return unary_operator(x, "GlobalLpPool", p=p)

    return helper_global_lp_pool(x, p=p)


def global_max_pool(x: "array.Array"):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    # @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_pool
    )
    def helper_global_max_pool(x: "array.Array"):
        return unary_operator(x, "GlobalMaxPool")

    return helper_global_max_pool(x)


def greater(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_greater(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "Greater")
    return helper_greater(a, b)


def greater_equal(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_greater_equal(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "GreaterOrEqual")
    return helper_greater_equal(a, b)


def hard_sigmoid(x: "array.Array", alpha: float = 0.2, beta: float = 0.5):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_hard_sigmoid(x: "array.Array", alpha: float, beta: float):
        return unary_operator(x, "HardSigmoid", alpha=alpha, beta=beta)

    return helper_hard_sigmoid(x, alpha=float(alpha), beta=float(beta))


def hardmax(x: "array.Array", axis: int = -1):

    if axis < -x.ndims or axis > x.ndims - 1:
        raise ValueError(
            f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_hardmax(x: "array.Array", axis: int):
        return unary_operator(x, "Hardmax", axis=axis)

    return helper_hardmax(x, axis=int(axis))


def identity(x: "array.Array"):
    @allowed_types(all_types)
    def helper_identity(x: "array.Array"):
        return unary_operator(x, "Identity")

    return helper_identity(x)


def instance_normalization(x: "array.Array", scale: "array.Array", bias: "array.Array", epsilon: float = 1e-05):

    if x.dtype != scale.dtype or x.dtype != bias.dtype:
        raise TypeError(
            f"The types of scale ({scale.dtype}) and B ({bias.dtype}) must match the one of x ({x.dtype})")
    if scale.ndims != 1:
        raise ValueError("scale must be a 1D tensor")
    if bias.ndims != 1:
        raise ValueError("bias must be a 1D tensor")

    @allowed_types(float_types, float_types, float_types)
    @not_implemented_types([np.float64], [np.float64], [np.float64])
    def helper_instance_normalization(x: "array.Array", scale: "array.Array", bias: "array.Array", epsilon: float):
        return nary_operator("InstanceNormalization", x, scale, bias, epsilon=epsilon)

    return helper_instance_normalization(x, scale, bias, epsilon=epsilon)


def isinf(x: "array.Array", detect_negative: bool = True, detect_positive: bool = True):
    @allowed_types(float_types)
    @output_checks_and_inference(
        output_type(np.bool_)
    )
    def helper_isinf(x: "array.Array", detect_negative: int, detect_positive: int):
        return unary_operator(x, "IsInf", detect_negative=detect_negative, detect_positive=detect_positive)

    return helper_isinf(x, detect_negative=int(detect_negative), detect_positive=int(detect_positive))


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


def lrn(x: "array.Array", size: int, alpha: float = 0.0001, beta: float = 0.75, bias: float = 1.0):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_lrn(x: "array.Array", size: int, alpha: float, beta: float, bias: float):
        return unary_operator(x, "LRN", size=size, alpha=alpha, beta=beta, bias=bias)

    return helper_lrn(x, size=size, alpha=alpha, beta=beta, bias=bias)


def lstm(x: "array.Array", w: "array.Array", r: "array.Array", b: "array.Array", sequence_length: "array.Array", initial_h: "array.Array",
         initial_c: "array.Array", P: "array.Array", hidden_size: int, activation_alpha: List[float] = None, activation_beta: List[float] = None,
         activations: List[str] = None, clip: float = 0.0, direction: str = "forward", layout: int = 0, input_forget: bool = False):
    raise NotImplementedError()


def leakyrelu(x: "array.Array", alpha: float = 0.01):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_leakyrelu(x, alpha: float):
        return unary_operator(x, "LeakyRelu", alpha=alpha)
    return helper_leakyrelu(x, alpha=alpha)


def less(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_less(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "Less")
    return helper_less(a, b)


def less_equal(a: "array.Array", b: "array.Array"):
    @not_implemented_types([np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16])
    @allowed_types(numeric_types, numeric_types)
    @output_checks_and_inference(
        output_type(np.bool_),
        allow_broadcasting
    )
    def helper_less_equal(a: "array.Array", b: "array.Array"):
        return binary_operator(a, b, "LessOrEqual")
    return helper_less_equal(a, b)


def log(x: "array.Array"):
    @allowed_types(float_types)
    def helper_log(x: "array.Array"):
        return unary_operator(x, "Log")

    return helper_log(x)


def logsoftmax(x: "array.Array", axis: int = -1):

    axis = int(axis)
    if axis < -x.ndims or axis > x.ndims - 1:
        raise ValueError(
            f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")

    @allowed_types(float_types)
    def helper_logsoftmax(x: "array.Array", axis: int):
        return unary_operator(x, "LogSoftmax", axis=axis)

    return helper_logsoftmax(x, axis=axis)


def lp_normalization(x: "array.Array", axis: int = -1, p: int = 2):

    axis = int(axis)
    if axis < -x.ndims or axis > x.ndims - 1:
        raise ValueError(
            f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")
    p = int(p)
    if p not in [1, 2]:
        raise ValueError(f"Normalization order has to be either 1 or 2, but got {p}")

    @allowed_types(float_types)
    def helper_lp_normalization(x: "array.Array", axis: int, p: int):
        return unary_operator(x, "LpNormalization", axis=axis, p=p)

    return helper_lp_normalization(x, axis=axis, p=p)


def lp_pool(x: "array.Array", kernel_shape: List[int], auto_pad: str = "NOTSET", p: int = 2, pads: List[int] = None, strides: List[int] = None):
    raise NotImplementedError()


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


def matmul_integer(A: "array.Array", B: "array.Array", a_zero_point: "array.Array" = None, b_zero_point: "array.Array" = None):

    if a_zero_point is None:
        a_zero_point = array.array(0, dtype=A.dtype)
    elif a_zero_point.ndims == 1 and a_zero_point.shape[0] != A.shape[-2]:
        raise ValueError(
            "The A zero point 1D tensor must match the number of rows of A")
    elif a_zero_point.dtype != A.dtype:
        raise TypeError(
            f"The A zero points ({a_zero_point.dtype}) must have the same type as A ({A.dtype})")

    if b_zero_point is None:
        b_zero_point = array.array(0, dtype=B.dtype)
    elif b_zero_point.ndims == 1 and b_zero_point.shape[0] != B.shape[-1]:
        raise ValueError(
            "The B zero point 1D tensor must match the number of rows of B")
    elif b_zero_point.dtype != B.dtype:
        raise TypeError(
            f"The B zero points ({b_zero_point.dtype}) must have the same type as B ({B.dtype})")

    if (A.dtype == np.int8 and B.dtype == np.uint8) or (A.dtype == np.int8 and B.dtype == np.int8):
        # uint8-int8 and int8-int8 known to not be implemented
        raise NotImplementedError(
            f"Combination A B matrix types int8-uint8 and int8-int8 are not implemented. Got {A.dtype}-{B.dtype}")

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


def maximum(*arrays):
    @allowed_types(numeric_types)
    @not_implemented_types([np.uint8, np.uint16, np.int8, np.int16])
    @types_match_exactly
    @shapes_match_exactly
    def helper_maximum(*arrays):
        return nary_operator("Max", *arrays)

    return helper_maximum(*arrays)


def maxpool(x: "array.Array", kernel_shape: List[int], auto_pad: str = "NOTSET", ceil_mode: bool = False,
            dilations: Optional[List[int]] = None, pads: Optional[List[int]] = None,
            storage_order: int = 0, strides: Optional[List[int]] = None):
    raise NotImplementedError("Operators with more than one output are not handled yet")


def maxroipool(x: "array.Array", rois: "array.Array", pooled_shape: List[int], spatial_scale: float = 1.0):
    raise NotImplementedError("Operators with more than one output are not handled yet")


def maxunpool(x: "array.Array", indices: "array.Array", kernel_shape: List[int], output_shape: Optional[List[int]] = None,
              pads: Optional[List[int]] = None, strides: Optional[List[int]] = None):

    if output_shape is None:
        raise NotImplementedError("Currently onnxruntime requires output_shape to be specified")
    
    @allowed_types(float_types, [np.int64], [np.int64])
    @not_implemented_types([np.float64])
    def helper_maxunpool(x: "array.Array", indices: "array.Array", output_shape: Optional[List[int]], kernel_shape: List[int],
                         pads: List[int], strides: List[int]):
        return nary_operator("MaxUnpool", x, indices, output_shape, kernel_shape=kernel_shape, pads=pads, strides=strides)

    return helper_maxunpool(x, indices, output_shape, kernel_shape=kernel_shape, pads=pads, strides=strides)



def power(x: "array.Array", y: "array.Array"):
    @allowed_types([*float_types, np.int32, np.int64], numeric_types)
    @not_implemented_types([], [np.uint8,np.uint16,np.uint32,np.uint64,np.int8,np.int16])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def helper_power(x: "array.Array", y: "array.Array"):
        return binary_operator(x, y, "Pow")
    
    return helper_power(x, y)