from .array import Array, array, is_lazy
from .ops_utils import (
    allowed_types, not_implemented_types, output_checks_and_inference,
    allow_broadcasting, nary_operator, propagate_shape_pool,
    force_evaluation)
from .types import (float_types, signed_integer_types, all_types)
from .shapes import ShapeLike, as_shape
import numpy as np
from typing import Union, Optional, List


def conv():
    # TODO
    raise NotImplementedError()


def relu(x):
    @not_implemented_types([np.float64, *signed_integer_types])
    @allowed_types([*float_types, *signed_integer_types])
    def relu_helper(x):
        return nary_operator("Relu", x)
    return relu_helper(x)


def elu(x, alpha=1.0):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def elu_helper(x, alpha):
        return nary_operator("Elu", x, alpha=alpha)

    return elu_helper(x, alpha=float(alpha))


def gru(
        x: Array, w: Array, r: Array, b: Array,
        sequence_length: Array, initial_h: Array,
        hidden_size: int, activation_alpha: List[float] = None,
        activation_beta: List[float] = None, activations: List[str] = None,
        clip: float = 0.0, direction: str = "forward", layout: int = 0,
        linear_before_reset: bool = False):
    # TODO
    raise NotImplementedError()


def global_average_pool(x: Array):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_pool
    )
    def helper_global_average_pool(x: Array):
        return nary_operator("GlobalAveragePool", x)

    return helper_global_average_pool(x)


def global_lp_pool(x: Array, p: int = 2):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    # @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_pool
    )
    def helper_global_lp_pool(x: Array, p: int):
        return nary_operator("GlobalLpPool", x, p=p)

    return helper_global_lp_pool(x, p=p)


def global_max_pool(x: Array):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    # @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_pool
    )
    def helper_global_max_pool(x: Array):
        return nary_operator("GlobalMaxPool", x)

    return helper_global_max_pool(x)


def hard_sigmoid(x: Array, alpha: float = 0.2, beta: float = 0.5):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_hard_sigmoid(x: Array, alpha: float, beta: float):
        return nary_operator("HardSigmoid", x, alpha=alpha, beta=beta)

    return helper_hard_sigmoid(x, alpha=float(alpha), beta=float(beta))


def hardmax(x: Array, axis: int = -1):

    if axis < -x.ndims or axis > x.ndims - 1:
        raise ValueError(
            f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_hardmax(x: Array, axis: int):
        return nary_operator("Hardmax", x, axis=axis)

    return helper_hardmax(x, axis=int(axis))


def instance_normalization(
        x: Array, scale: Array, bias: Array,
        epsilon: float = 1e-05):

    if x.dtype != scale.dtype or x.dtype != bias.dtype:
        raise TypeError(
            f"The types of scale ({scale.dtype}) and B ({bias.dtype}) must match the "
            f"one of x ({x.dtype})")
    if scale.ndims != 1:
        raise ValueError("scale must be a 1D tensor")
    if bias.ndims != 1:
        raise ValueError("bias must be a 1D tensor")

    @allowed_types(float_types, float_types, float_types)
    @not_implemented_types([np.float64], [np.float64], [np.float64])
    def helper_instance_normalization(
            x: Array, scale: Array, bias: Array,
            epsilon: float):
        return nary_operator(
            "InstanceNormalization", x, scale, bias, epsilon=epsilon)

    return helper_instance_normalization(x, scale, bias, epsilon=epsilon)


def lrn(
        x: Array, size: int, alpha: float = 0.0001, beta: float = 0.75,
        bias: float = 1.0):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_lrn(x: Array, size: int, alpha: float, beta: float,
                   bias: float):
        return nary_operator(
            "LRN", x, size=size, alpha=alpha, beta=beta, bias=bias)

    return helper_lrn(x, size=size, alpha=alpha, beta=beta, bias=bias)


def lstm(x: Array, w: Array, r: Array, b: Array, sequence_length: Array,
         initial_h: Array, initial_c: Array, P: Array, hidden_size: int,
         activation_alpha: List[float] = None, activation_beta: List[float] = None,
         activations: List[str] = None, clip: float = 0.0, direction: str = "forward",
         layout: int = 0, input_forget: bool = False):
    # TODO
    raise NotImplementedError()


def leakyrelu(x: Array, alpha: float = 0.01):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_leakyrelu(x, alpha: float):
        return nary_operator("LeakyRelu", x, alpha=alpha)
    return helper_leakyrelu(x, alpha=alpha)


def logsoftmax(x: Array, axis: int = -1):

    axis = int(axis)
    if axis < -x.ndims or axis > x.ndims - 1:
        raise ValueError(
            f"Axis must be in the range [-{x.ndims}, {x.ndims-1}]")

    @allowed_types(float_types)
    def helper_logsoftmax(x: Array, axis: int):
        return nary_operator("LogSoftmax", x, axis=axis)

    return helper_logsoftmax(x, axis=axis)


def maxpool(
        x: Array, kernel_shape: List[int],
        auto_pad: str = "NOTSET", ceil_mode: bool = False,
        dilations: Optional[List[int]] = None, pads: Optional[List[int]] = None,
        storage_order: int = 0, strides: Optional[List[int]] = None):
    # TODO
    raise NotImplementedError(
        "Operators with more than one output are not handled yet")


def maxroipool(
        x: Array, rois: Array, pooled_shape: List[int],
        spatial_scale: float = 1.0):
    # TODO
    raise NotImplementedError(
        "Operators with more than one output are not handled yet")


def maxunpool(
        x: Array, indices: Array, kernel_shape: List[int],
        output_shape: Optional[ShapeLike] = None, pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None, allow_evaluation: bool = False):

    # TODO: check this is still correct
    if output_shape is None:
        raise NotImplementedError(
            "Currently onnxruntime requires output_shape to be specified")

    output_shape = force_evaluation(
        as_shape(output_shape).asarray(),
        "output_shape", allow_evaluation)

    @allowed_types(float_types, [np.int64], [np.int64])
    @not_implemented_types([np.float64])
    def helper_maxunpool(
            x: Array, indices: Array,
            output_shape: Optional[Array],
            kernel_shape: List[int],
            pads: List[int],
            strides: List[int]):
        result = nary_operator(
            "MaxUnpool", x, indices, output_shape, kernel_shape=kernel_shape,
            pads=pads, strides=strides)
        if output_shape is not None:
            result._dims = as_shape(output_shape)
        return result

    return helper_maxunpool(
        x, indices, output_shape, kernel_shape=kernel_shape, pads=pads,
        strides=strides)


def negative_loglikelihood_loss(
        input: Array, target: Array,
        weight: Optional[Array] = None, ignore_index: Optional[int] = None,
        reduction: str = "mean"):
    def negative_loglikelihood_loss_helper(
            input: Array, target: Array,
            weight: Optional[Array],
            ignore_index: Optional[int],
            reduction: str):
        NotImplementedError("negative_loglikelihood_loss")
    return negative_loglikelihood_loss_helper(
        input, target, weight, ignore_index, reduction)


def prelu(x: Array, slope: Union[Array, float]):
    if isinstance(slope, float):
        slope = array([slope], dtype=x.dtype)

    @allowed_types([*float_types, np.uint32, np.uint64, np.int32, np.int64])
    @not_implemented_types([np.float64, np.uint32, np.uint64, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def prelu_helper(x: Array, slope: Array):
        return nary_operator("PRelu", x, slope)

    return prelu_helper(x, slope)


def scatter(
        data: Array, indices: Array, updates: Array,
        axis: int = 0):
    if axis < -len(data.shape) or axis >= len(data.shape):
        raise ValueError(
            f"Axis must be in the range [-{len(data.shape)}, {len(data.shape)-1}]")

    data_rank = len(data.shape)
    if data_rank == 0:
        raise ValueError("Data rank should be >= 1")
    if len(indices.shape) != data_rank:
        raise ValueError("Indices rank should be the same as that of data")
    if not is_lazy(indices):
        # TODO: could check indices are within bounds here
        pass
    if len(updates.shape) != data_rank:
        raise ValueError("Updates rank should be the same as that of data")

    @allowed_types(all_types, [np.int32, np.int64], all_types)
    def scatter_helper(
            data: Array, indices: Array, updates: Array,
            axis: int = 0):
        return nary_operator(
            "ScatterElements", data, indices, updates, axis=axis)

    return scatter_helper(data, indices, updates, axis=axis)


def scatter_nd(
        data: Array, indices: Array, updates: Array):

    q = len(data.shape)
    if q == 0:
        raise ValueError("Data rank should be >= 1")

    if indices.shape[-1] != -1:
        r = len(indices.shape)
        if len(updates.shape) != q + r - indices.shape[-1] - 1:
            raise ValueError(
                "Updates rank should of rank q + r - indices_shape[-1] - 1")

    @allowed_types(all_types, [np.int64], all_types)
    def scatter_nd_helper(
            data: Array, indices: Array, updates: Array):
        return nary_operator(
            "ScatterND", data, indices, updates)

    return scatter_nd_helper(data, indices, updates)


def selu(
        x: Array, alpha: Optional[float] = None, gamma: Optional[float] = None):

    alpha = float(alpha) if alpha is not None else None
    gamma = float(gamma) if gamma is not None else None

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def selu_helper(x: Array, alpha: float, gamma: float):
        return nary_operator("Selu", x, alpha=alpha, gamma=gamma)

    return selu_helper(x, alpha, gamma)


def softplus(x: Array):

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def softplus_helper(x: Array):
        return nary_operator("Softplus", x)

    return softplus_helper(x)


def softsign(x: Array):

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def softsign_helper(x: Array):
        return nary_operator("Softsign", x)

    return softsign_helper(x)
