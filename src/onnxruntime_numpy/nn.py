from .array import Array, array, is_lazy
from .ops_utils import (
    allowed_types, not_implemented_types, output_checks_and_inference,
    allow_broadcasting, nary_operator, propagate_shape_global_pool,
    force_evaluation, propagate_pool_shape, propagate_conv_shape,
    multi_output_nary_operator, check_axis_is_valid, gather_check,
    propagate_shape_from_argn_position)
from .types import (float_types, signed_integer_types, all_types, numeric_types)
from .shapes import ShapeLike, as_shape, DynamicShape, weak_shape_comparisson
import numpy as np
from typing import Union, Optional, List


def average_pool(
        x: Array, kernel_shape: List[int],
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        auto_pad: str = "NOTSET", ceil_mode: bool = False,
        count_include_pad: bool = False):

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_pool_shape(
            kernel_shape, pads, strides, auto_pad,
            ceil_mode))
    def average_pool_helper(x: Array, kernel_shape: List[int],
                            pads: Optional[List[int]],
                            strides: Optional[List[int]],
                            auto_pad: str, ceil_mode: int,
                            count_include_pad: int):
        return nary_operator(
            "AveragePool", x, kernel_shape=kernel_shape, pads=pads,
            strides=strides, auto_pad=auto_pad, ceil_mode=ceil_mode,
            count_include_pad=count_include_pad)

    return average_pool_helper(
        x, kernel_shape, pads, strides, auto_pad, int(ceil_mode),
        int(count_include_pad))


def batch_normalization(
        x: Array, scale: Array, bias: Array, input_mean: Array,
        input_var: Array, epsilon: float = 1e-5, momentum: float = 0.9,
        training_mode: bool = False):

    if x.dtype != scale.dtype or x.dtype != bias.dtype or x.dtype != input_var \
            or x.dtype != input_mean:
        raise ValueError(
            "Types of `x`, `scale` and `bias`, `input_mean` and `input_var` must match")

    @allowed_types(float_types, float_types, float_types)
    def batch_normalization_helper(
            x: Array, scale: Array, bias: Array, input_mean: Array,
            input_var: Array, epsilon: float, momentum: float,
            training_mode: int):
        if training_mode == 0:
            # eval mode
            return nary_operator(
                "BatchNormalization", x, scale, bias, input_mean, input_var,
                epsilon=epsilon, momentum=momentum)
        else:
            raise NotImplementedError()

    return batch_normalization_helper(
        x, scale, bias, input_mean, input_var, epsilon=epsilon,
        momentum=momentum, training_mode=int(training_mode))


def conv(
        x: Array, w: Array, bias: Optional[Array] = None,
        kernel_shape: Optional[List[int]] = None, pads: Optional[List[int]] = None,
        strides: List[int] = None, group: int = 1,
        dilations: Optional[List[int]] = None, auto_pad: str = "NOTSET"):

    if kernel_shape is None:
        kernel_shape = w.shape[2:].tolist()

    @allowed_types(float_types, float_types, float_types)
    @not_implemented_types([np.float64], [np.float64], [np.float64])
    @output_checks_and_inference(
        propagate_conv_shape(
            kernel_shape, pads, strides, group, dilations, auto_pad))
    def conv_helper(x: Array, w: Array, bias: Optional[Array],
                    kernel_shape: Optional[List[int]],
                    pads: Optional[int], strides: List[int], group: int,
                    dilations: Optional[List[int]], auto_pad: str):
        return nary_operator(
            "Conv", x, w, bias, kernel_shape=kernel_shape, pads=pads,
            strides=strides, group=group, dilations=dilations,
            auto_pad=auto_pad)

    return conv_helper(x, w, bias, kernel_shape=kernel_shape, pads=pads,
                       strides=strides, group=group, dilations=dilations,
                       auto_pad=auto_pad)


def conv_integer(
        x: Array, w: Array, x_zero_point: Optional[Array] = None,
        w_zero_point: Optional[Array] = None,
        kernel_shape: Optional[List[int]] = None, pads: Optional[List[int]] = None,
        strides: List[int] = None, group: int = 1,
        dilations: Optional[List[int]] = None, auto_pad: str = "NOTSET"):

    if kernel_shape is None:
        kernel_shape = w.shape[2:].tolist()

    if x_zero_point is not None and x.dtype != x_zero_point.dtype:
        raise ValueError(
            f"type of x {x.dtype} does not match corresponding zero point "
            f"{x_zero_point.dtype}")
    if w_zero_point is not None and w.dtype != w_zero_point.dtype:
        raise ValueError(
            f"type of x {w.dtype} does not match corresponding zero point "
            f"{w_zero_point.dtype}")

    @allowed_types([np.int8, np.uint8],
                   [np.int8, np.uint8],
                   [np.int8, np.uint8],
                   [np.int8, np.uint8])
    @not_implemented_types([np.int8], [np.int8], [np.int8], [np.int8])
    @output_checks_and_inference(
        propagate_conv_shape(
            kernel_shape, pads, strides, group, dilations, auto_pad))
    def conv_integer_helper(x: Array, w: Array, x_zero_point: Optional[Array],
                            w_zero_point: Optional[Array],
                            kernel_shape: Optional[List[int]],
                            pads: Optional[int], strides: List[int], group: int,
                            dilations: Optional[List[int]], auto_pad: str):
        result = nary_operator(
            "ConvInteger", x, w, x_zero_point, w_zero_point,
            kernel_shape=kernel_shape, pads=pads, strides=strides, group=group,
            dilations=dilations, auto_pad=auto_pad)
        result._dtype = np.int32
        return result

    return conv_integer_helper(
        x, w, x_zero_point, w_zero_point, kernel_shape=kernel_shape, pads=pads,
        strides=strides, group=group, dilations=dilations, auto_pad=auto_pad)


def conv_transpose(
        x: Array, w: Array, bias: Optional[Array] = None,
        kernel_shape: Optional[List[int]] = None, pads: Optional[List[int]] = None,
        strides: List[int] = None, group: int = 1,
        output_padding: Optional[List[int]] = None,
        output_shape: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None, auto_pad: str = "NOTSET"):

    if kernel_shape is None:
        kernel_shape = w.shape[2:].tolist()

    # I think these are flipped in the onnxruntime v1.7.0 implementation
    if auto_pad.upper() == "SAME_LOWER":
        auto_pad = "SAME_UPPER"
    elif auto_pad.upper() == "SAME_UPPER":
        auto_pad = "SAME_LOWER"

    @allowed_types(float_types, float_types, float_types)
    @not_implemented_types([np.float64], [np.float64], [np.float64])
    @output_checks_and_inference(
        propagate_conv_shape(
            kernel_shape, pads, strides, group, dilations, auto_pad,
            output_padding, output_shape, transpose=True))
    def conv_transpose_helper(
            x: Array, w: Array, bias: Optional[Array],
            kernel_shape: Optional[List[int]],
            pads: Optional[int],
            strides: List[int],
            group: int, dilations: Optional[List[int]],
            output_padding: Optional[List[int]],
            output_shape: Optional[List[int]],
            auto_pad: str):
        return nary_operator(
            "ConvTranspose", x, w, bias, kernel_shape=kernel_shape, pads=pads,
            strides=strides, group=group, dilations=dilations,
            auto_pad=auto_pad, output_shape=output_shape,
            output_padding=output_padding)

    return conv_transpose_helper(
        x, w, bias, kernel_shape=kernel_shape, pads=pads, strides=strides,
        group=group, dilations=dilations, auto_pad=auto_pad.upper(),
        output_shape=output_shape, output_padding=output_padding)


def depth_to_space(x: Array, blocksize: int, mode: str = "DCR"):

    if x.ndims != 4:
        raise ValueError(f"Rank of x has to be four, but got {x.ndims}")

    n, c, h, w = x.shape

    if c.is_static() and int(c) % (blocksize ** 2) != 0:
        raise ValueError(
            f"Input depth ({int(c)}) has to be a multiple of blocksize square "
            f"({blocksize ** 2})")

    @allowed_types(all_types)
    @not_implemented_types([np.float64, np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64, np.bool_])
    def depth_to_space_helper(x: Array, blocksize: int, mode: str):
        n, c, h, w = x.shape
        result = nary_operator(
            "DepthToSpace", x, blocksize=blocksize, mode=mode)
        new_c = int(c) // (blocksize ** 2) if c.is_static() else c
        new_h = int(h) * blocksize if h.is_static() else h
        new_w = int(w) * blocksize if w.is_static() else w
        result._dims = DynamicShape(n, new_c, new_h, new_w)

        return result

    return depth_to_space_helper(x, blocksize=blocksize, mode=mode)


def dequantize_linear(x: Array, x_scale: Array,
                      x_zero_point: Optional[Array] = None, axis: int = 1):

    if x.dtype == np.int32 and x_zero_point is not None:
        raise ValueError(
            "x_zero_point cannot be set when using x with type int32")

    # only check axis arg if one or both x_scale and x_zero_point aren't scalars
    if (len(x_scale.shape) > 0
            or (x_zero_point is not None and len(x_zero_point.shape) > 0)):
        check_axis_is_valid(x, axis)

    if len(x_scale.shape) > 0:
        if len(x_scale.shape) == 1 and x_scale.shape[0] != x.shape[axis]:
            raise ValueError(
                f"x_scale axis {axis} ({int(x_scale.shape[0])}) does not "
                f"match x {axis} ({int(x.shape[axis])})")
        if len(x_scale.shape) > 1:
            raise ValueError(
                "x_scale has to be a scalar or a 1D Array, but got "
                f"{x_scale.ndims}D Array")

    if x_zero_point is not None:
        if x.dtype != x_zero_point.dtype:
            raise ValueError("x and x_zero_point must have matching types")

        if len(
                x_zero_point.shape) == 1 and x_zero_point.shape[0] != x.shape[axis]:
            raise ValueError(
                f"x_zero_point axis {axis} ({int(x_zero_point.shape[0])}) does not "
                f"match x {axis} ({int(x.shape[axis])})")
        if len(x_zero_point.shape) > 1:
            raise ValueError(
                "x_zero_point has to be a scalar or a 1D Array, but got "
                f"{x_zero_point.ndims}D Array")

    @allowed_types([np.int8, np.uint8, np.int32],
                   [np.float32],
                   [np.int8, np.uint8, np.int32])
    def dequantize_linear_helper(
            x: Array, x_scale: Array, x_zero_point: Optional[Array] = None,
            axis: int = 1):
        result = nary_operator("DequantizeLinear", x,
                               x_scale, x_zero_point, axis=axis)
        result._dtype = np.float32
        return result

    return dequantize_linear_helper(x, x_scale, x_zero_point, axis=axis)


def dropout(
        x: Array, ratio: Optional[Array] = None,
        training_mode: Optional[Array] = None, seed: Optional[int] = None):

    if ratio and ratio.ndims != 0:
        raise ValueError("Ratio of Dropout must be a scalar.")

    if training_mode and training_mode.ndims != 0:
        raise ValueError("Training of Dropout must be a bool.")

    @allowed_types(float_types, float_types, [np.bool_])
    def dropout_helper(
            x: Array, ratio: Optional[Array],
            training_mode: Optional[Array],
            seed: Optional[int]):
        output, mask = multi_output_nary_operator(2)(
            "Dropout", x, ratio, training_mode, seed=seed)

        mask._dims = output.shape
        mask._dtype = np.bool_

        return output, mask

    return dropout_helper(x, ratio, training_mode, seed=seed)


def elu(x, alpha=1.0):
    @not_implemented_types([np.float64])
    @allowed_types([*float_types])
    def elu_helper(x, alpha):
        return nary_operator("Elu", x, alpha=alpha)

    return elu_helper(x, alpha=float(alpha))


def gather(x: Array, indices: Array, axis: int = 0):

    @allowed_types(all_types, [np.int32, np.int64])
    @output_checks_and_inference(
        gather_check(int(axis))
    )
    def gather_helper(x: Array, indices: Array, axis: int):
        return nary_operator("Gather", x, indices, axis=axis)

    return gather_helper(x, indices, axis=int(axis))


def gather_elements(x: Array, indices: Array, axis: int = 0):

    check_axis_is_valid(x, axis)

    @allowed_types(all_types, [np.int32, np.int64])
    @output_checks_and_inference(
        propagate_shape_from_argn_position(1)
    )
    def gather_elements_helper(
            x: Array, indices: Array, axis: int):
        return nary_operator("GatherElements", x, indices, axis=axis)

    return gather_elements_helper(x, indices, axis=int(axis))


def gathernd(x: Array, indices: Array, batch_dims: int = 0):

    if x.ndims < 1:
        raise ValueError(f"x must have rank >= 1, but got {x.ndims}")

    if indices.ndims < 1:
        raise ValueError(
            f"indices must have rank >= 1, but got {indices.ndims}")

    if batch_dims >= min(x.ndims, indices.ndims):
        raise ValueError(
            "batch_dims has be less than the rank of x and indices")

    @allowed_types(all_types, [np.int64])
    def gathernd_helper(x: Array, indices: Array, batch_dims: int):

        r = x.ndims

        for b in range(batch_dims):
            if indices.shape[b] != x.shape[b]:
                raise ValueError(
                    f"The first batch_dims dimensions ({batch_dims}) of the shape of "
                    f"indices tensor ({indices.shape}) and x ({x.shape}) tensor "
                    "must be equal.")

        if indices.shape[-1].is_static() \
                and indices.shape[-1] not in range(1, r - batch_dims):
            raise ValueError(
                "The last dimension of indices, should have a value in range [1, "
                f"{r-b}]")

        if not indices.shape[-1].is_static():
            # have to force evaluation
            _ = indices.numpy()

        last_indices_dimension = batch_dims + int(indices.shape[-1])
        output_shape = DynamicShape(
            *indices.shape[: -1],
            *x.shape[last_indices_dimension:])

        result = nary_operator("GatherND", x, indices, batch_dims=batch_dims)
        result._dims = output_shape

        return result

    return gathernd_helper(x, indices, batch_dims=batch_dims)


def global_average_pool(x: Array):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_global_pool
    )
    def helper_global_average_pool(x: Array):
        return nary_operator("GlobalAveragePool", x)

    return helper_global_average_pool(x)


def global_lp_pool(x: Array, p: int = 2):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    # TODO: @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_global_pool
    )
    def helper_global_lp_pool(x: Array, p: int):
        return nary_operator("GlobalLpPool", x, p=p)

    return helper_global_lp_pool(x, p=p)


def global_max_pool(x: Array):

    if x.ndims != 4:
        raise ValueError(
            f"Expected tensor to have 4 dimensions (N,C,H,W), but got {x.ndims}")

    @allowed_types(float_types)
    # TODO: @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_shape_global_pool
    )
    def helper_global_max_pool(x: Array):
        return nary_operator("GlobalMaxPool", x)

    return helper_global_max_pool(x)


def gru(
        x: Array, w: Array, r: Array, hidden_size: int, b: Optional[Array] = None,
        sequence_length: Optional[Array] = None, initial_h: Optional[Array] = None,
        activation_alpha: Optional[List[float]] = None,
        activation_beta: Optional[List[float]] = None,
        activations: Optional[List[str]] = None, clip: Optional[float] = None,
        direction: str = "forward", linear_before_reset: bool = False):

    if direction.lower() not in ["forward", "reverse", "bidirectional"]:
        raise ValueError(
            "direction has to be one of forward, reverse or bidirectional")

    def gru_helper(x: Array, w: Array, r: Array, b: Optional[Array],
                   sequence_length: Optional[Array], initial_h: Optional[Array],
                   hidden_size: int, activation_alpha: Optional[List[float]],
                   activation_beta: Optional[List[float]],
                   activations: Optional[List[str]],
                   clip: Optional[float], direction: str,
                   linear_before_reset: bool):

        y, yh = multi_output_nary_operator(2)(
            "GRU", x, w, r, b, sequence_length, initial_h,
            hidden_size=hidden_size, activation_alpha=activation_alpha,
            activation_beta=activation_beta, activations=activations, clip=clip,
            direction=direction, linear_before_reset=linear_before_reset)

        seq_length, batch_size, _ = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        y._dims = DynamicShape(seq_length, num_directions,
                               batch_size, hidden_size)
        yh._dims = DynamicShape(num_directions, batch_size, hidden_size)

        return y, yh

    return gru_helper(
        x, w, r, b, sequence_length, initial_h, hidden_size=hidden_size,
        activation_alpha=activation_alpha, activation_beta=activation_beta,
        activations=activations, clip=clip, direction=direction.lower(),
        linear_before_reset=linear_before_reset)


def hard_sigmoid(x: Array, alpha: float = 0.2, beta: float = 0.5):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_hard_sigmoid(x: Array, alpha: float, beta: float):
        return nary_operator("HardSigmoid", x, alpha=alpha, beta=beta)

    return helper_hard_sigmoid(x, alpha=float(alpha), beta=float(beta))


def hardmax(x: Array, axis: int = -1):

    check_axis_is_valid(x, axis)

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


def lp_normalization(x: Array, axis: int = -1, p: int = 2):

    axis = int(axis)

    check_axis_is_valid(x, axis)

    p = int(p)
    if p not in [1, 2]:
        raise ValueError(
            f"Normalization order has to be either 1 or 2, but got {p}")

    @allowed_types(float_types)
    def helper_lp_normalization(x: Array, axis: int, p: int):
        return nary_operator("LpNormalization", x, axis=axis, p=p)

    return helper_lp_normalization(x, axis=axis, p=p)


def lp_pool(
        x: Array, kernel_shape: List[int],
        auto_pad: str = "NOTSET", p: int = 2, pads: List[int] = None,
        strides: List[int] = None):

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    @output_checks_and_inference(
        propagate_pool_shape(
            kernel_shape, pads, strides, auto_pad, 1))
    def lp_pool_helper(
            x: Array, kernel_shape: List[int],
            auto_pad: str, p: int, pads: List[int],
            strides: List[int]):
        return nary_operator(
            "LpPool", x, kernel_shape=kernel_shape, auto_pad=auto_pad, p=p,
            pads=pads, strides=strides)

    return lp_pool_helper(
        x, kernel_shape=kernel_shape, auto_pad=auto_pad, p=p, pads=pads,
        strides=strides)


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


def lstm(
        x: Array, w: Array, r: Array, hidden_size: int, b: Optional[Array] = None,
        sequence_lengths: Optional[Array] = None, initial_h: Optional[Array] = None,
        initial_c: Optional[Array] = None, P: Optional[Array] = None,
        activation_alpha: Optional[List[float]] = None,
        activation_beta: Optional[List[float]] = None,
        activations: Optional[List[str]] = None, clip: Optional[float] = None,
        direction: str = "forward", input_forget: bool = False):

    if direction.lower() not in ["forward", "reverse", "bidirectional"]:
        raise ValueError(
            "direction has to be one of forward, reverse or bidirectional")

    @allowed_types(float_types, float_types, float_types, float_types,
                   [np.int32],
                   float_types, float_types, float_types)
    @not_implemented_types([np.float64],
                           [np.float64],
                           [np.float64],
                           [np.float64],
                           [],
                           [np.float64],
                           [np.float64],
                           [np.float64])
    def lstm_helper(
            x: Array, w: Array, r: Array, b: Optional[Array],
            sequence_lengths: Optional[Array],
            initial_h: Optional[Array],
            initial_c: Optional[Array], P: Optional[Array],
            hidden_size: int, activation_alpha: Optional[List[float]],
            activation_beta: Optional[List[float]],
            activations: Optional[List[str]],
            clip: Optional[float], direction: str, input_forget: int):
        seq_length, batch_size, input_size = x.shape
        num_directions = 2 if direction == "bidirectional" else 1

        expected_w_shape = DynamicShape(
            num_directions, 4 * hidden_size, input_size)

        expected_r_shape = DynamicShape(
            num_directions, 4 * hidden_size, hidden_size)

        if not weak_shape_comparisson(w.shape, expected_w_shape):
            raise ValueError(
                f"W expected to be of shape {expected_w_shape}, but got {w.shape}")

        if not weak_shape_comparisson(r.shape, expected_r_shape):
            raise ValueError(
                f"R expected to be of shape {expected_r_shape}, but got {r.shape}")

        if b is not None:
            expected_b_shape = DynamicShape(num_directions, 8 * hidden_size)
            if not weak_shape_comparisson(r.shape, expected_r_shape):
                raise ValueError(
                    f"bias expected to be of shape {expected_b_shape}, "
                    f"but got {b.shape}")

        if sequence_lengths:
            expected_sequence_lengths = DynamicShape(batch_size)
            if not weak_shape_comparisson(
                    sequence_lengths.shape, expected_sequence_lengths):
                raise ValueError(
                    "sequence_lengths expected to be of shape "
                    f"{expected_sequence_lengths}, but got {sequence_lengths.shape}")

        if initial_h:
            expected_initial_h = DynamicShape(
                num_directions, batch_size, hidden_size)
            if not weak_shape_comparisson(
                    initial_h.shape, expected_initial_h):
                raise ValueError(
                    "Initial hidden expected to be of shape "
                    f"{expected_initial_h}, but got {initial_h.shape}")

        if initial_c:
            expected_initial_c = DynamicShape(
                num_directions, batch_size, hidden_size)
            if not weak_shape_comparisson(
                    initial_c.shape, expected_initial_c):
                raise ValueError(
                    "Initial cell value expected to be of shape "
                    f"{expected_initial_c}, but got {initial_c.shape}")

        if P:
            expected_p = DynamicShape(num_directions, 3 * hidden_size)
            if not weak_shape_comparisson(P.shape, expected_p):
                raise ValueError(
                    "Initial cell value expected to be of shape "
                    f"{expected_p}, but got {P.shape}")

        y, yh, yc = multi_output_nary_operator(3)(
            "LSTM", x, w, r, b, sequence_lengths, initial_h, initial_c, P,
            hidden_size=hidden_size, activation_alpha=activation_alpha,
            activation_beta=activation_beta, activations=activations, clip=clip,
            direction=direction)

        y._dims = DynamicShape(seq_length, num_directions,
                               batch_size, hidden_size)
        yh._dims = DynamicShape(num_directions, batch_size, hidden_size)
        yc._dims = DynamicShape(num_directions, batch_size, hidden_size)

        return y, yh, yc

    return lstm_helper(
        x, w, r, b, sequence_lengths, initial_h, initial_c, P,
        hidden_size=hidden_size, activation_alpha=activation_alpha,
        activation_beta=activation_beta, activations=activations, clip=clip,
        direction=direction.lower(), input_forget=int(input_forget))


def leakyrelu(x: Array, alpha: float = 0.01):
    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def helper_leakyrelu(x, alpha: float):
        return nary_operator("LeakyRelu", x, alpha=alpha)
    return helper_leakyrelu(x, alpha=alpha)


def logsoftmax(x: Array, axis: int = -1):

    axis = int(axis)
    check_axis_is_valid(x, axis)

    @allowed_types(float_types)
    def helper_logsoftmax(x: Array, axis: int):
        return nary_operator("LogSoftmax", x, axis=axis)

    return helper_logsoftmax(x, axis=axis)


def maxpool(
        x: Array, kernel_shape: List[int],
        auto_pad: str = "NOTSET", ceil_mode: bool = False,
        dilations: Optional[List[int]] = None, pads: Optional[List[int]] = None,
        storage_order: int = 0, strides: Optional[List[int]] = None):

    @allowed_types([*float_types, np.int8, np.uint8])
    @output_checks_and_inference(
        propagate_pool_shape(
            kernel_shape, pads, strides, auto_pad,
            ceil_mode, dilations))
    def maxpool_helper(x: Array, kernel_shape: List[int],
                       pads: Optional[List[int]],
                       strides: Optional[List[int]],
                       dilations: Optional[List[int]],
                       auto_pad: str, ceil_mode: int,
                       storage_order: int):
        return multi_output_nary_operator(2)(
            "MaxPool", x, kernel_shape=kernel_shape, pads=pads,
            strides=strides, auto_pad=auto_pad, ceil_mode=ceil_mode,
            storage_order=storage_order, dilations=dilations)

    y, indices = maxpool_helper(
        x, kernel_shape, pads, strides, dilations, auto_pad, int(ceil_mode),
        storage_order)
    indices._dims = y.shape
    indices._dtype = np.int64

    return y, indices


def maxroipool(
        x: Array, rois: Array, pooled_shape: List[int],
        spatial_scale: float = 1.0):

    # if x.ndims < 4:
    #     raise ValueError()

    if rois.ndims != 2:
        raise ValueError(f"rois should have rank 2 but got rank {rois.ndims}")

    if rois.shape[1] != 5:
        raise ValueError(
            f"rois should have shape (num_rois, 5) but got {rois.shape}")

    def maxroipool_helper(x: Array, rois: Array, pooled_shape: List[int],
                          spatial_scale: float):

        batch_size, channels = x.shape[:2]
        num_rois = rois.shape[0]

        result = nary_operator(
            "MaxRoiPool", x, rois, pooled_shape=pooled_shape,
            spatial_scale=spatial_scale)

        result._dims = DynamicShape(
            num_rois, channels, pooled_shape[0],
            pooled_shape[1])

        return result

    return maxroipool_helper(
        x, rois, pooled_shape=pooled_shape, spatial_scale=spatial_scale)


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
        x: Array, target: Array,
        weight: Optional[Array] = None, ignore_index: Optional[int] = None,
        reduction: str = "mean"):

    import warnings
    warnings.warn("This function is currently unstable")

    if x.ndims < 2:
        raise ValueError(
            f"x must have rank greater than or equal to 2, but got rank {x.ndims}")

    if target.ndims < 1:
        raise ValueError(
            "target must have rank greater than or equal to 1, but got rank "
            f"{target.ndims}")

    if reduction.lower() not in ["none", "sum", "mean"]:
        raise ValueError(
            f"reduction must be one of none, sum or mean, but got {reduction}")

    def negative_loglikelihood_loss_helper(
            x: Array, target: Array,
            weight: Optional[Array],
            ignore_index: Optional[int],
            reduction: str = "mean"):

        if x.ndims > 2:
            x_n, x_c, x_ds = x.shape
            target_n, target_ds = target.shape

            if not weak_shape_comparisson(
                    DynamicShape(x_ds),
                    DynamicShape(target_ds)):
                raise ValueError(
                    f"x and target features shapes {x_ds} and {target_ds} are not "
                    "compatible")

        x_n = x.shape[0]
        target_n = target.shape[0]

        if x_n != target_n:
            raise ValueError(
                f"x and target batch sizes {x_n} and {target_n} are not compatible")

        if weight:
            if weight.ndims != 1:
                raise ValueError(
                    f"weight must have rank 1, but got rank {target.ndims}")

            x_c = x.shape[1]
            weight_c = weight.shape[0]

            if x_c != weight_c:
                raise ValueError(
                    f"x and target channel sizes {x_c} and {weight_c} are not "
                    "compatible")

        result = nary_operator(
            "NegativeLogLikelihoodLoss", x, target,
            ignore_index=ignore_index, reduction=reduction)

        result._dims = DynamicShape(
            x_n) if reduction == "none" else DynamicShape()

        return result

    return negative_loglikelihood_loss_helper(
        x, target, weight, ignore_index=ignore_index, reduction=reduction.lower())


def prelu(x: Array, slope: Union[Array, float]):
    if isinstance(slope, float):
        slope = array([slope], dtype=x.dtype)

    @allowed_types([*float_types, np.uint32, np.uint64, np.int32, np.int64])
    @not_implemented_types(
        [np.float64, np.uint32, np.uint64, np.int32, np.int64])
    @output_checks_and_inference(
        allow_broadcasting
    )
    def prelu_helper(x: Array, slope: Array):
        return nary_operator("PRelu", x, slope)

    return prelu_helper(x, slope)


def relu(x):
    @not_implemented_types([np.float64, *signed_integer_types])
    @allowed_types([*float_types, *signed_integer_types])
    def relu_helper(x):
        return nary_operator("Relu", x)
    return relu_helper(x)


def scatter(
        data: Array, indices: Array, updates: Array,
        axis: int = 0):
    check_axis_is_valid(data, axis)

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


def shrink(x: Array, bias: float = 0.0, lambd: float = 0.5):
    @allowed_types(numeric_types)
    def shrink_helper(x: Array, bias: float, lambd: float):
        return nary_operator("Shrink", x, bias=bias, lambd=lambd)

    return shrink_helper(x, bias=bias, lambd=lambd)


def sigmoid(x: Array):
    @allowed_types(float_types)
    def helper_sigmoid(x: Array):
        return nary_operator("Sigmoid", x)

    return helper_sigmoid(x)


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


def space_to_depth(x: Array, blocksize: int):

    if x.ndims != 4:
        raise ValueError(f"Rank of x must be four, but got {x.ndims}")

    n, c, h, w = x.shape

    if h.is_static() and int(h) % blocksize != 0:
        raise ValueError(
            f"Input height ({int(h)}) must be a multiple of blocksize ({blocksize})")

    if w.is_static() and int(w) % blocksize != 0:
        raise ValueError(
            f"Input width ({int(w)}) must be a multiple of blocksize ({blocksize})")

    @allowed_types(all_types)
    @not_implemented_types([np.float64, np.uint8, np.uint16, np.uint32, np.uint64,
                            np.int8, np.int16, np.int32, np.int64, np.bool_])
    def depth_to_space_helper(x: Array, blocksize: int):
        n, c, h, w = x.shape
        result = nary_operator(
            "SpaceToDepth", x, blocksize=blocksize)
        new_c = int(c) * (blocksize ** 2) if c.is_static() else c
        new_h = int(h) / blocksize if h.is_static() else h
        new_w = int(w) / blocksize if w.is_static() else w
        result._dims = DynamicShape(n, new_c, new_h, new_w)

        return result

    return depth_to_space_helper(x, blocksize=blocksize)


def thresholded_relu(x: Array, alpha: float = 1.0):

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def thresholded_relu_helper(x: Array, alpha: float):
        return nary_operator("ThresholdedRelu", x, alpha=alpha)

    return thresholded_relu_helper(x, alpha=float(alpha))
