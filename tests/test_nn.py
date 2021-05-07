import pytest
import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types, all_types
import numpy as np
from .utils import expect, GRU_Helper, dropout_reference, LSTM_Helper
import math
import itertools


def get_pool_pad_shape(auto_pad,
                       input_spatial_shape,
                       kernel_spatial_shape,
                       strides_spatial,
                       output_spatial_shape):
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                kernel_spatial_shape[i] - input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape


def get_pool_output_shape(auto_pad,
                          input_spatial_shape,
                          kernel_spatial_shape,
                          strides_spatial):
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(
                        input_spatial_shape[i])
                    / float(
                        strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(
                        input_spatial_shape[i] -
                        (kernel_spatial_shape[i] - 1)) /
                    float(strides_spatial[i])))
    return out_shape


def pool_reference(padded,
                   x_shape,
                   kernel_shape,
                   strides_shape,
                   out_shape,
                   pad_shape,
                   pooling_type,
                   count_include_pad=0):
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(
        range(x_shape[0]),
        range(x_shape[1]),
        *[
            range(
                int(
                 (x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) /
                    strides_shape[i] + 1)) for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        window_vals = np.array(
            [window[i]
             for i in list(
                 itertools.product(
                     *
                     [
                         range(
                             strides_shape[i] * shape[i + 2],
                             strides_shape[i] * shape[i + 2] + kernel_shape[i])
                         for i in range(spatial_size)]))])
        if pooling_type == 'AVG':
            f = np.average
        elif pooling_type == 'MAX':
            f = np.max
        else:
            raise NotImplementedError(
                'Pooling type {} does not support. Should be AVG, MAX'.format(
                    pooling_type))

        if count_include_pad == 1 and pooling_type == 'AVG':
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)


def reference_batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_1d_default(type_a):
    x = np.random.randn(1, 3, 32).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = [2]
    strides = [1]
    out_shape = get_pool_output_shape(
        'VALID', x_shape[2:],
        kernel_shape, strides)
    padded = x

    expected = pool_reference(
        padded, x_shape, kernel_shape, strides, out_shape, [0],
        'AVG').astype(type_a)
    result = onp.nn.average_pool(onp.array(x), kernel_shape=kernel_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_ceil(type_a):
    x = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]]).astype(type_a)
    kernel_shape = [3, 3]
    strides = [2, 2]
    ceil_mode = True

    expected = np.array([[[
        [6, 7.5],
        [12, 13.5]]]]).astype(type_a)
    result = onp.nn.average_pool(
        onp.array(x),
        kernel_shape=kernel_shape, strides=strides, ceil_mode=ceil_mode)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_default(type_a):
    x = np.random.randn(1, 3, 32, 32).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_pool_output_shape(
        'VALID', x_shape[2:],
        kernel_shape, strides)
    padded = x

    expected = pool_reference(padded, x_shape, kernel_shape,
                              strides, out_shape, (0, 0), 'AVG')

    result = onp.nn.average_pool(
        onp.array(x),
        kernel_shape=kernel_shape)

    expect(expected, result.numpy(), rtol=1.e-3)


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_pads(type_a):
    x = np.random.randn(1, 3, 28, 28).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = (3, 3)
    strides = (1, 1)
    pad_bottom = 2
    pad_top = 2
    pad_right = 2
    pad_left = 2
    pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
    out_shape = get_pool_output_shape('VALID', np.add(
        x_shape[2:], pad_shape), kernel_shape, strides)
    padded = np.pad(
        x, ((0, 0),
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right)),
        mode='constant', constant_values=np.nan)

    expected = pool_reference(padded, x_shape, kernel_shape,
                              strides, out_shape, pad_shape, 'AVG')

    result = onp.nn.average_pool(
        onp.array(x),
        kernel_shape=kernel_shape, pads=[2, 2, 2, 2])

    expect(expected, result.numpy(), rtol=1.e-3)


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_pads_count_include_pad(type_a):
    x = np.random.randn(1, 3, 28, 28).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = (3, 3)
    strides = (1, 1)
    pad_bottom = 2
    pad_top = 2
    pad_right = 2
    pad_left = 2
    pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
    out_shape = get_pool_output_shape('VALID', np.add(
        x_shape[2:], pad_shape), kernel_shape, strides)
    padded = np.pad(
        x, ((0, 0),
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right)),
        mode='constant', constant_values=0)

    expected = pool_reference(
        padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG',
        count_include_pad=1)

    result = onp.nn.average_pool(
        onp.array(x),
        kernel_shape=kernel_shape, pads=[2, 2, 2, 2],
        count_include_pad=True)

    expect(expected, result.numpy(), rtol=1.e-2)


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_same_upper(type_a):
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(type_a)

    expected = np.array([[[[4, 5.5, 7],
                           [11.5, 13, 14.5],
                           [19, 20.5, 22]]]]).astype(type_a)

    result = onp.nn.average_pool(
        onp.array(x),
        kernel_shape=[3, 3], strides=[2, 2],
        auto_pad="SAME_UPPER")

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_same_lower(type_a):
    x = np.random.randn(1, 3, 32, 32).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_pool_output_shape(
        'SAME_LOWER', x_shape[2:],
        kernel_shape, strides)
    pad_shape = get_pool_pad_shape(
        'SAME_LOWER', x_shape[2:],
        kernel_shape, strides, out_shape)
    pad_bottom = pad_shape[0] // 2
    pad_top = pad_shape[0] - pad_bottom
    pad_right = pad_shape[1] // 2
    pad_left = pad_shape[1] - pad_right
    padded = np.pad(
        x, ((0, 0),
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right)),
        mode='constant', constant_values=np.nan)
    expected = pool_reference(padded, x_shape, kernel_shape,
                              strides, out_shape, pad_shape, 'AVG')

    result = onp.nn.average_pool(
        onp.array(x), kernel_shape=kernel_shape, auto_pad="SAME_LOWER")

    expect(expected, result.numpy(), rtol=1e-03)


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_2d_strides(type_a):
    x = np.random.randn(1, 3, 32, 32).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = (5, 5)
    strides = (3, 3)
    out_shape = get_pool_output_shape(
        'VALID', x_shape[2:],
        kernel_shape, strides)
    padded = x
    expected = pool_reference(
        padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')

    result = onp.nn.average_pool(
        onp.array(x), kernel_shape=kernel_shape, strides=strides)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_average_pool_3d_default(type_a):
    x = np.random.randn(1, 3, 32, 32, 32).astype(type_a)
    x_shape = np.shape(x)
    kernel_shape = [2, 2, 2]
    strides = [1, 1, 1]
    out_shape = get_pool_output_shape(
        'VALID', x_shape[2:],
        kernel_shape, strides)
    padded = x
    expected = pool_reference(
        padded, x_shape, kernel_shape, strides, out_shape, (0, 0, 0), 'AVG')

    result = onp.nn.average_pool(
        onp.array(x), kernel_shape=kernel_shape)

    expect(expected, result.numpy(), rtol=1.e-2)


@pytest.mark.parametrize("type_a", float_types)
def test_batch_normalization_eval(type_a):
    x = np.random.randn(2, 3, 4, 5).astype(type_a)
    s = np.random.randn(3).astype(type_a)
    bias = np.random.randn(3).astype(type_a)
    mean = np.random.randn(3).astype(type_a)
    var = np.random.rand(3).astype(type_a)
    expected = reference_batchnorm_test_mode(
        x, s, bias, mean, var).astype(type_a)
    result = onp.nn.batch_normalization(
        onp.array(x),
        onp.array(s),
        onp.array(bias),
        onp.array(mean),
        onp.array(var))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv(type_a):
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[12., 21., 27., 33., 24.],
                           [33., 54., 63., 72., 51.],
                           [63., 99., 108., 117., 81.],
                           [93., 144., 153., 162., 111.],
                           [72., 111., 117., 123., 84.]]]]).astype(type_a)

    result = onp.nn.conv(
        onp.array(x),
        onp.array(W),
        kernel_shape=(3, 3),
        pads=(1, 1, 1, 1))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_without_padding(type_a):
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[54., 63., 72.],
                           [99., 108., 117.],
                           [144., 153., 162.]]]]).astype(type_a)

    result = onp.nn.conv(
        onp.array(x),
        onp.array(W),
        kernel_shape=(3, 3),
        pads=(0, 0, 0, 0))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_with_autopad_same(type_a):
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[12., 27., 24.],
                           [63., 108., 81.],
                           [72., 117., 84.]]]]).astype(type_a)

    result = onp.nn.conv(
        onp.array(x),
        onp.array(W),
        kernel_shape=(3, 3),
        strides=(2, 2),
        auto_pad="SAME_LOWER")

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_with_strides(type_a):
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[12., 27., 24.],
                           [63., 108., 81.],
                           [123., 198., 141.],
                           [112., 177., 124.]]]]).astype(type_a)

    result = onp.nn.conv(
        onp.array(x),
        onp.array(W),
        kernel_shape=(3, 3),
        strides=(2, 2),
        pads=(1, 1, 1, 1))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_with_strides_no_padding(type_a):
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[54., 72.],
                           [144., 162.],
                           [234., 252.]]]]).astype(type_a)

    result = onp.nn.conv(
        onp.array(x),
        onp.array(W),
        kernel_shape=(3, 3),
        strides=(2, 2),
        pads=(0, 0, 0, 0))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_with_strides_and_asymmetric_padding(type_a):
    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[21., 33.],
                           [99., 117.],
                           [189., 207.],
                           [171., 183.]]]]).astype(type_a)

    result = onp.nn.conv(
        onp.array(x),
        onp.array(W),
        kernel_shape=(3, 3),
        strides=(2, 2),
        pads=(1, 0, 1, 0))

    expect(expected, result.numpy())


# FIXME: Currently segfaults
# @pytest.mark.parametrize("type_a", [np.uint8])
# @pytest.mark.parametrize("type_b", [np.uint8])
# def test_conv_integer_with_padding(type_a, type_b):
#     x = np.array(
#         [2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(
#         type_a).reshape(
#         (1, 1, 3, 3))

#     x_zero_point = type_a(1)

#     w = np.array([1, 1, 1, 1]).astype(type_b).reshape((1, 1, 2, 2))

#     expected = np.array(
#         [1, 3, 5, 3, 5, 12, 16, 9, 11, 24, 28, 15, 7, 15, 17, 9]).astype(
#         np.int32).reshape(
#         (1, 1, 4, 4))

#     result = onp.nn.conv_integer(
#         onp.array(x),
#         onp.array(w),
#         onp.array(x_zero_point),
#         pads=(1, 1, 1, 1))

#     expect(expected, result.numpy())


# FIXME: Also segfaults
# @pytest.mark.parametrize("type_a", [np.uint8])
# @pytest.mark.parametrize("type_b", [np.uint8])
# def test_conv_integer_without_padding(type_a, type_b):
#     x = np.array(
#         [2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(
#         type_a).reshape(
#         (1, 1, 3, 3))

#     x_zero_point = type_a(1)

#     w = np.array([1, 1, 1, 1]).astype(type_b).reshape((1, 1, 2, 2))

#     expected = np.array([12, 16, 24, 28]).astype(np.int32).reshape(1, 1, 2, 2)

#     result = onp.nn.conv_integer(
#         onp.array(x),
#         onp.array(w),
#         onp.array(x_zero_point))

#     expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose(type_a):
    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                           [3., 8., 15., 12., 7.],
                           [9., 21., 36., 27., 15.],
                           [9., 20., 33., 24., 13.],
                           [6., 13., 21., 15., 8.]],

                          [[0., 1., 3., 3., 2.],
                           [3., 8., 15., 12., 7.],
                           [9., 21., 36., 27., 15.],
                           [9., 20., 33., 24., 13.],
                           [6., 13., 21., 15., 8.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x),
        onp.array(W))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_1D(type_a):
    x = np.array([[[0., 1., 2.]]]).astype(type_a)

    W = np.array([[[1., 1., 1.],  # (1, 2, 3)
                   [1., 1., 1.]]]).astype(type_a)

    expected = np.array([[[0., 1., 3., 3., 2.],  # (1, 2, 5)
                          [0., 1., 3., 3., 2.]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x),
        onp.array(W))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_3D(type_a):
    x = np.array([[[[[0., 1., 2., 3., 4.],  # (1, 1, 3, 4, 5)
                 [5., 6., 7., 8., 9.],
                 [10., 11., 12., 13., 14.],
                 [15., 16., 17., 18., 19.]],
        [[20., 21., 22., 23., 24.],
         [25., 26., 27., 28., 29.],
         [30., 31., 32., 33., 34.],
         [35., 36., 37., 38., 39.]],
        [[40., 41., 42., 43., 44.],
         [45., 46., 47., 48., 49.],
         [50., 51., 52., 53., 54.],
         [55., 56., 57., 58., 59.]]]]]).astype(type_a)

    W = np.array([[[[[1., 1., 1.],  # (1, 2, 3, 3, 3)
                 [1., 1., 1.],
                 [1., 1., 1.]],
        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]],
        [[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]]],
        [[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],
         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],
         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]]]]).astype(type_a)

    expected = np.array([[[[[0., 1., 3., 6., 9., 7., 4.],  # (1, 2, 5, 6, 7)
                            [5., 12., 21., 27., 33., 24., 13.],
                            [15., 33., 54., 63., 72., 51., 27.],
                            [30., 63., 99., 108., 117., 81., 42.],
                            [25., 52., 81., 87., 93., 64., 33.],
                            [15., 31., 48., 51., 54., 37., 19.]],

                           [[20., 42., 66., 72., 78., 54., 28.],
                            [50., 104., 162., 174., 186., 128., 66.],
                            [90., 186., 288., 306., 324., 222., 114.],
                            [120., 246., 378., 396., 414., 282., 144.],
                            [90., 184., 282., 294., 306., 208., 106.],
                            [50., 102., 156., 162., 168., 114., 58.]],

                           [[60., 123., 189., 198., 207., 141., 72.],
                            [135., 276., 423., 441., 459., 312., 159.],
                            [225., 459., 702., 729., 756., 513., 261.],
                            [270., 549., 837., 864., 891., 603., 306.],
                            [195., 396., 603., 621., 639., 432., 219.],
                            [105., 213., 324., 333., 342., 231., 117.]],

                           [[60., 122., 186., 192., 198., 134., 68.],
                            [130., 264., 402., 414., 426., 288., 146.],
                            [210., 426., 648., 666., 684., 462., 234.],
                            [240., 486., 738., 756., 774., 522., 264.],
                            [170., 344., 522., 534., 546., 368., 186.],
                            [90., 182., 276., 282., 288., 194., 98.]],

                           [[40., 81., 123., 126., 129., 87., 44.],
                            [85., 172., 261., 267., 273., 184., 93.],
                            [135., 273., 414., 423., 432., 291., 147.],
                            [150., 303., 459., 468., 477., 321., 162.],
                            [105., 212., 321., 327., 333., 224., 113.],
                            [55., 111., 168., 171., 174., 117., 59.]]],

                          [[[0., 1., 3., 6., 9., 7., 4.],
                            [5., 12., 21., 27., 33., 24., 13.],
                              [15., 33., 54., 63., 72., 51., 27.],
                              [30., 63., 99., 108., 117., 81., 42.],
                              [25., 52., 81., 87., 93., 64., 33.],
                              [15., 31., 48., 51., 54., 37., 19.]],

                           [[20., 42., 66., 72., 78., 54., 28.],
                              [50., 104., 162., 174., 186., 128., 66.],
                              [90., 186., 288., 306., 324., 222., 114.],
                              [120., 246., 378., 396., 414., 282., 144.],
                              [90., 184., 282., 294., 306., 208., 106.],
                              [50., 102., 156., 162., 168., 114., 58.]],

                           [[60., 123., 189., 198., 207., 141., 72.],
                              [135., 276., 423., 441., 459., 312., 159.],
                              [225., 459., 702., 729., 756., 513., 261.],
                              [270., 549., 837., 864., 891., 603., 306.],
                              [195., 396., 603., 621., 639., 432., 219.],
                              [105., 213., 324., 333., 342., 231., 117.]],

                           [[60., 122., 186., 192., 198., 134., 68.],
                              [130., 264., 402., 414., 426., 288., 146.],
                              [210., 426., 648., 666., 684., 462., 234.],
                              [240., 486., 738., 756., 774., 522., 264.],
                              [170., 344., 522., 534., 546., 368., 186.],
                              [90., 182., 276., 282., 288., 194., 98.]],

                           [[40., 81., 123., 126., 129., 87., 44.],
                              [85., 172., 261., 267., 273., 184., 93.],
                              [135., 273., 414., 423., 432., 291., 147.],
                              [150., 303., 459., 468., 477., 321., 162.],
                              [105., 212., 321., 327., 333., 224., 113.],
                              [55., 111., 168., 171., 174., 117., 59.]]]]]).astype(
                                  type_a)

    result = onp.nn.conv_transpose(
        onp.array(x),
        onp.array(W))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_with_output_shape(type_a):
    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]],

                          [[0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x),
        onp.array(W),
        strides=(3, 2),
        output_shape=(10, 8))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_with_output_padding(type_a):
    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]],

                          [[0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x),
        onp.array(W),
        strides=(3, 2),
        output_padding=(1, 1))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_with_output_padding_and_output_shape(type_a):
    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]],

                          [[0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [0., 0., 1., 1., 3., 2., 2., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [3., 3., 7., 4., 9., 5., 5., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [6., 6., 13., 7., 15., 8., 8., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x),
        onp.array(W),
        strides=(3, 2),
        output_shape=(10, 8),
        output_padding=(1, 1))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_autopad_same(type_a):
    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[0., 0., 1., 1., 3., 2.],
                           [0., 0., 1., 1., 3., 2.],
                           [3., 3., 8., 5., 12., 7.],
                           [3., 3., 7., 4., 9., 5.],
                           [9., 9., 20., 11., 24., 13.],
                           [6., 6., 13., 7., 15., 8.]],

                          [[0., 0., 1., 1., 3., 2.],
                           [0., 0., 1., 1., 3., 2.],
                           [3., 3., 8., 5., 12., 7.],
                           [3., 3., 7., 4., 9., 5.],
                           [9., 9., 20., 11., 24., 13.],
                           [6., 6., 13., 7., 15., 8.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x), onp.array(W),
        auto_pad="SAME_UPPER", strides=[2, 2])

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_dilations(type_a):
    x = np.array([[[[3., 8., 1.],  # (1, 1, 3, 3)
                    [9., 5., 7.],
                    [3., 2., 6.]]]]).astype(type_a)

    W = np.array([[[[7., 2.],  # (1, 1, 2, 2)
                    [1., 9.]]]]).astype(type_a)

    expected = np.array([[[[21., 56., 13., 16., 2.],  # [1, 1, 5, 5]
                           [63., 35., 67., 10., 14.],
                           [24., 22., 76., 76., 21.],
                           [9., 5., 88., 45., 63.],
                           [3., 2., 33., 18., 54.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x), onp.array(W),
        dilations=[2, 2])

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_conv_transpose_pads(type_a):
    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(type_a)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(type_a)

    expected = np.array([[[[1., 1., 3.],  # (1, 2, 7, 3)
                           [1., 1., 3.],
                           [7., 4., 9.],
                           [7., 4., 9.],
                           [7., 4., 9.],
                           [13., 7., 15.],
                           [13., 7., 15.]],

                          [[1., 1., 3.],
                           [1., 1., 3.],
                           [7., 4., 9.],
                           [7., 4., 9.],
                           [7., 4., 9.],
                           [13., 7., 15.],
                           [13., 7., 15.]]]]).astype(type_a)

    result = onp.nn.conv_transpose(
        onp.array(x), onp.array(W),
        strides=[3, 2],
        pads=[1, 2, 1, 2])

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_depth_to_space_crd_mode(type_a):
    # (1, 8, 2, 3) input tensor
    x = np.array([[[[0, 1, 2],
                    [3, 4, 5]],
                   [[9, 10, 11],
                    [12, 13, 14]],
                   [[18, 19, 20],
                    [21, 22, 23]],
                   [[27, 28, 29],
                    [30, 31, 32]],
                   [[36, 37, 38],
                    [39, 40, 41]],
                   [[45, 46, 47],
                    [48, 49, 50]],
                   [[54, 55, 56],
                    [57, 58, 59]],
                   [[63, 64, 65],
                    [66, 67, 68]]]]).astype(type_a)

    # (1, 2, 4, 6) output tensor
    expected = np.array([[[[0, 9, 1, 10, 2, 11],
                           [18, 27, 19, 28, 20, 29],
                           [3, 12, 4, 13, 5, 14],
                           [21, 30, 22, 31, 23, 32]],
                          [[36, 45, 37, 46, 38, 47],
                           [54, 63, 55, 64, 56, 65],
                           [39, 48, 40, 49, 41, 50],
                           [57, 66, 58, 67, 59, 68]]]]).astype(type_a)

    result = onp.nn.depth_to_space(onp.array(x), blocksize=2, mode="CRD")
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_depth_to_space_default(type_a):
    # (1, 8, 2, 3) input tensor
    x = np.array([[[[0, 1, 2],
                    [3, 4, 5]],
                   [[9, 10, 11],
                    [12, 13, 14]],
                   [[18, 19, 20],
                    [21, 22, 23]],
                   [[27, 28, 29],
                    [30, 31, 32]],
                   [[36, 37, 38],
                    [39, 40, 41]],
                   [[45, 46, 47],
                    [48, 49, 50]],
                   [[54, 55, 56],
                    [57, 58, 59]],
                   [[63, 64, 65],
                    [66, 67, 68]]]]).astype(type_a)

    # (1, 2, 4, 6) output tensor
    expected = np.array([[[[0, 18, 1, 19, 2, 20],
                           [36, 54, 37, 55, 38, 56],
                           [3, 21, 4, 22, 5, 23],
                           [39, 57, 40, 58, 41, 59]],
                          [[9, 27, 10, 28, 11, 29],
                           [45, 63, 46, 64, 47, 65],
                           [12, 30, 13, 31, 14, 32],
                           [48, 66, 49, 67, 50, 68]]]]).astype(type_a)

    result = onp.nn.depth_to_space(onp.array(x), blocksize=2)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.int8, np.uint8, np.int32])
def test_dequantize_linear(type_a):
    x = np.array([[[[3, 89],
                    [34, 200],
                    [74, 59]],
                   [[5, 24],
                    [24, 87],
                    [32, 13]],
                   [[245, 99],
                    [4, 142],
                    [121, 102]]]], dtype=type_a)
    x_scale = np.array([2, 4, 5], dtype=np.float32)
    if type_a == np.int32:
        expected = x.astype(np.float32) * x_scale.reshape(1, 3, 1, 1)
        result = onp.nn.dequantize_linear(
            onp.array(x),
            onp.array(x_scale))
    else:
        x_zero_point = np.array([84, 24, 196], dtype=type_a)
        expected = (x.astype(np.float32) - x_zero_point.reshape(1, 3, 1,
                    1).astype(np.float32)) * x_scale.reshape(1, 3, 1, 1)
        result = onp.nn.dequantize_linear(
            onp.array(x),
            onp.array(x_scale),
            onp.array(x_zero_point))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.int8, np.uint8, np.int32])
def test_dequantize_linear_scalar(type_a):
    x = np.array([[[[3, 89],
                    [34, 200],
                    [74, 59]],
                   [[5, 24],
                    [24, 87],
                    [32, 13]],
                   [[245, 99],
                    [4, 142],
                    [121, 102]]]], dtype=type_a)
    x_scale = np.array(2, dtype=np.float32)
    if type_a == np.int32:
        expected = x.astype(np.float32) * x_scale
        result = onp.nn.dequantize_linear(
            onp.array(x),
            onp.array(x_scale))
    else:
        x_zero_point = np.array(84, dtype=type_a)
        expected = (x.astype(np.float32) - x_zero_point.astype(np.float32)
                    ) * x_scale
        result = onp.nn.dequantize_linear(
            onp.array(x),
            onp.array(x_scale),
            onp.array(x_zero_point))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_dropout_default(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = dropout_reference(x)
    result, _ = onp.nn.dropout(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_dropout_default_mask(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected, expected_mask = dropout_reference(x, return_mask=True)

    result, mask = onp.nn.dropout(onp.array(x), seed=0)

    expect(expected, result.numpy())
    expect(expected_mask, mask.numpy())


@pytest.mark.parametrize("type_a", float_types)
@pytest.mark.parametrize("type_b", float_types)
def test_dropout_default_mask_ratio(type_a, type_b):
    x = np.random.randn(3, 4, 5).astype(type_a)
    r = type_b(0.1)
    expected, expected_mask = dropout_reference(x, r, return_mask=True)

    result, mask = onp.nn.dropout(onp.array(x), onp.array(r), seed=0)

    expect(expected, result.numpy())
    expect(expected_mask, mask.numpy())


@pytest.mark.parametrize("type_a", float_types)
@pytest.mark.parametrize("type_b", float_types)
@pytest.mark.parametrize("type_c", [np.bool_])
def test_dropout_training(type_a, type_b, type_c):
    x = np.random.randn(3, 4, 5).astype(type_a)
    r = type_b(0.75)
    t = type_c(True)
    mask = np.array([[[False, False, True, False, False],
                      [False, False, False, False, True],
                      [False, False, True, False, False],
                      [False, False, False, False, False]],
                     [[False, False, False, True, True],
                      [False, False, False, False, False],
                      [True, True, False, False, False],
                      [False, False, True, True, False]],
                     [[False, True, False, True, False],
                      [False, False, True, False, False],
                      [True, False, False, False, False],
                      [False, False, True, True, False]]])

    scale = 1. / (1. - r)
    expected = (scale * x * mask).astype(type_a)

    result, _ = onp.nn.dropout(
        onp.array(x),
        onp.array(r),
        onp.array(t),
        seed=0)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
@pytest.mark.parametrize("type_b", float_types)
@pytest.mark.parametrize("type_c", [np.bool_])
def test_dropout_training_default_ratio_mask(type_a, type_b, type_c):
    x = np.random.randn(3, 4, 5).astype(type_a)
    r = type_b(0.5)
    t = type_c(True)
    mask = np.array([[[False, False, True, False, True],
                      [False, False, True, True, True],
                      [False, True, True, False, False],
                      [True, True, False, False, False]],
                     [[False, True, True, True, True],
                      [True, False, True, False, True],
                      [True, True, False, False, True],
                      [False, True, True, True, False]],
                     [[False, True, True, True, True],
                      [False, True, True, False, False],
                      [True, False, False, False, False],
                      [False, False, True, True, False]]])

    scale = 1. / (1. - r)
    expected = (scale * x * mask).astype(type_a)

    result, m = onp.nn.dropout(
        onp.array(x),
        onp.array(r),
        onp.array(t),
        seed=0)

    expect(expected, result.numpy())
    expect(mask, m.numpy())


@pytest.mark.parametrize("type_a", float_types)
@pytest.mark.parametrize("type_b", float_types)
@pytest.mark.parametrize("type_c", [np.bool_])
def test_dropout_training_default_zero_ratio(type_a, type_b, type_c):
    x = np.random.randn(3, 4, 5).astype(type_a)
    r = type_b(0.)
    t = type_c(True)
    mask = np.array([[[True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True]],
                     [[True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True]],
                     [[True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True],
                      [True, True, True, True, True]]])

    scale = 1. / (1. - r)
    expected = (scale * x * mask).astype(type_a)

    result, m = onp.nn.dropout(
        onp.array(x),
        onp.array(r),
        onp.array(t),
        seed=0)

    expect(expected, result.numpy())
    expect(mask, m.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_elu(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([-1.2642411, 0., 1.], dtype=type_a)
    result = onp.nn.elu(a, alpha=2.0)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_elu_default(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([-0.63212055, 0., 1.], dtype=type_a)
    result = onp.nn.elu(a)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_relu(type_a):
    a = onp.array([[1., 2., -3.],
                   [-1, 0,  10.]], dtype=type_a)
    expected = onp.array([[1, 2, 0],
                          [0, 0, 10]], dtype=type_a)
    result = onp.nn.relu(a)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_average_pool(type_a):
    x = np.random.randn(1, 3, 5, 5).astype(type_a)
    spatial_shape = np.ndim(x) - 2
    expected = np.average(x, axis=tuple(
        range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        expected = np.expand_dims(expected, -1)

    result = onp.nn.global_average_pool(onp.array(x, dtype=type_a))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_average_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[5]]]]).astype(type_a)

    result = onp.nn.global_average_pool(onp.array(x, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_lp2_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[16.881943]]]]).astype(type_a)

    result = onp.nn.global_lp_pool(onp.array(x, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_lp1_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[45]]]]).astype(type_a)

    result = onp.nn.global_lp_pool(onp.array(x, dtype=type_a), p=1)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_lp_large_pool_precomputed(type_a):
    # this essentially becomes max pooling
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[9]]]]).astype(type_a)

    result = onp.nn.global_lp_pool(onp.array(x, dtype=type_a), p=40)

    expect(expected, result.numpy(), atol=0.1)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_max_pool(type_a):
    x = np.random.randn(1, 3, 5, 5).astype(type_a)
    spatial_shape = np.ndim(x) - 2
    expected = np.max(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        expected = np.expand_dims(expected, -1)

    result = onp.nn.global_max_pool(onp.array(x, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_max_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[9]]]]).astype(type_a)

    result = onp.nn.global_max_pool(onp.array(x, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_gru(type_a):
    x = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(type_a)

    input_size = 2
    hidden_size = 5
    weight_scale = 0.1
    number_of_gates = 3

    W = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               input_size)).astype(type_a)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               hidden_size)).astype(type_a)

    expected_y, expected_yh = GRU_Helper(
        X=x, W=W, R=R).step()

    y, yh = onp.nn.gru(onp.array(x), onp.array(W), onp.array(R),
                       hidden_size=hidden_size)

    expect(expected_y.astype(np.float32), y.numpy())
    expect(expected_yh.astype(np.float32), yh.numpy())


# TODO: fix test after upgrading to onnx opset version 14
# @pytest.mark.parametrize("type_a", [np.float32])
# def test_gru_batchwise(type_a):
#     x = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(type_a)

#     input_size = 2
#     hidden_size = 6
#     number_of_gates = 3
#     weight_scale = 0.2
#     layout = 1

#     W = weight_scale * np.ones((1, number_of_gates * hidden_size,
#                                input_size)).astype(type_a)
#     R = weight_scale * np.ones((1, number_of_gates * hidden_size,
#                                hidden_size)).astype(type_a)

#     expected_y, expected_yh = GRU_Helper(
#         X=x, W=W, R=R, layout=layout).step()

#     y, yh = onp.nn.gru(onp.array(x), onp.array(W), onp.array(R),
#                        hidden_size=hidden_size, layout=layout)

#     expect(expected_y, y.numpy())
#     expect(expected_yh, yh.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_hard_sigmoid(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([0.1, 0.6, 1.], dtype=type_a)
    result = onp.nn.hard_sigmoid(a, alpha=0.5, beta=0.6)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_hard_sigmoid_default(type_a):
    default_alpha = 0.2
    default_beta = 0.5
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x * default_alpha + default_beta, 0, 1)
    result = onp.nn.hard_sigmoid(onp.array(x))

    expect(result.numpy(), expected)


@pytest.mark.parametrize("type_a", [np.float32])
def test_hardmax(type_a):

    def hardmax(x, axis=-1):
        x_argmax = np.argmax(x, axis=axis)
        y = np.zeros_like(x)
        np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
        return y

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = hardmax(x, axis=0)
    result = onp.nn.hardmax(onp.array(x), axis=0)
    expect(result.numpy(), expected)

    expected = hardmax(x, axis=1)
    result = onp.nn.hardmax(onp.array(x), axis=1)
    expect(result.numpy(), expected)

    expected = hardmax(x)
    result = onp.nn.hardmax(onp.array(x))
    expect(result.numpy(), expected)


@pytest.mark.parametrize("type_a", [np.float32])
def test_instancenorm(type_a):

    def instancenorm_test_mode(x, s, bias, epsilon=1e-5):
        dims_x = len(x.shape)
        axis = tuple(range(2, dims_x))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        dim_ones = (1,) * (dims_x - 2)
        s = s.reshape(-1, *dim_ones)
        bias = bias.reshape(-1, *dim_ones)
        return s * (x - mean) / np.sqrt(var + epsilon) + bias

    x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(type_a)
    s = np.array([1.0, 1.5]).astype(type_a)
    bias = np.array([0, 1]).astype(type_a)
    expected = instancenorm_test_mode(x, s, bias).astype(type_a)

    expect(expected, onp.nn.instance_normalization(
        onp.array(x), onp.array(s), onp.array(bias)))

    x = np.random.randn(2, 3, 4, 5).astype(type_a)
    s = np.random.randn(3).astype(type_a)
    bias = np.random.randn(3).astype(type_a)
    epsilon = 1e-2
    expected = instancenorm_test_mode(x, s, bias, epsilon).astype(type_a)

    expect(
        expected, onp.nn.instance_normalization(
            onp.array(x),
            onp.array(s),
            onp.array(bias),
            epsilon=epsilon),
        rtol=1e-02)


@pytest.mark.parametrize("type_a", [np.float32])
def test_lrn_default(type_a):
    alpha = 0.0001
    beta = 0.75
    bias = 1.0
    nsize = 3
    x = np.random.randn(5, 5, 5, 5).astype(type_a)
    square_sum = np.zeros((5, 5, 5, 5)).astype(type_a)
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(
            x
            [n,
             max(0, c - int(math.floor((nsize - 1) / 2))):
             min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
             h, w] ** 2)
    expected = x / ((bias + (alpha / nsize) * square_sum) ** beta)

    expect(expected, onp.nn.lrn(onp.array(x), size=nsize))


@pytest.mark.parametrize("type_a", [np.float32])
def test_lrn(type_a):
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    nsize = 3
    x = np.random.randn(5, 5, 5, 5).astype(type_a)
    square_sum = np.zeros((5, 5, 5, 5)).astype(type_a)
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(
            x
            [n,
             max(0, c - int(math.floor((nsize - 1) / 2))):
             min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
             h, w] ** 2)
    expected = x / ((bias + (alpha / nsize) * square_sum) ** beta)

    expect(
        expected, onp.nn.lrn(
            onp.array(x),
            size=nsize, alpha=alpha, beta=beta, bias=bias))


@pytest.mark.parametrize("type_a", [np.float32])
def test_lstm(type_a):
    x = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(type_a)

    input_size = 2
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4

    W = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               input_size)).astype(type_a)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               hidden_size)).astype(type_a)

    expected_y, expected_yh = LSTM_Helper(X=x, W=W, R=R).step()

    y, yh, _ = onp.nn.lstm(onp.array(x), onp.array(W), onp.array(R),
                           hidden_size=hidden_size)

    expect(expected_y.astype(type_a), y.numpy())
    expect(expected_yh.astype(type_a), yh.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_lstm_initial_bias(type_a):
    x = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(type_a)

    input_size = 3
    hidden_size = 4
    weight_scale = 0.1
    custom_bias = 0.1
    number_of_gates = 4

    W = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               input_size)).astype(type_a)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               hidden_size)).astype(type_a)

    W_B = custom_bias * np.ones((1,
                                number_of_gates * hidden_size)).astype(type_a)
    R_B = np.zeros((1, number_of_gates * hidden_size)).astype(type_a)
    B = np.concatenate((W_B, R_B), 1)

    expected_y, expected_yh = LSTM_Helper(X=x, W=W, R=R, B=B).step()

    y, yh, _ = onp.nn.lstm(onp.array(x), onp.array(W), onp.array(R),
                           b=onp.array(B), hidden_size=hidden_size)

    expect(expected_y.astype(type_a), y.numpy())
    expect(expected_yh.astype(type_a), yh.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
@pytest.mark.parametrize("type_b", [np.int32])
def test_lstm_peepholes(type_a, type_b):
    x = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]).astype(type_a)

    input_size = 4
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4
    number_of_peepholes = 3

    W = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               input_size)).astype(type_a)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size,
                               hidden_size)).astype(type_a)
    B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(type_a)
    seq_lens = np.repeat(x.shape[0], x.shape[1]).astype(type_b)
    init_h = np.zeros((1, x.shape[1], hidden_size)).astype(type_a)
    init_c = np.zeros((1, x.shape[1], hidden_size)).astype(type_a)
    P = weight_scale * np.ones((1,
                               number_of_peepholes * hidden_size)).astype(type_a)

    expected_y, expected_yh = LSTM_Helper(
        X=x, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h).step()

    y, yh, _ = onp.nn.lstm(onp.array(x), onp.array(W), onp.array(R),
                           b=onp.array(B), P=onp.array(P),
                           sequence_lengths=onp.array(seq_lens),
                           hidden_size=hidden_size)

    expect(expected_y.astype(np.float32), y.numpy())
    expect(expected_yh.astype(np.float32), yh.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_leakyrelu(type_a):
    x = np.array([-1, 0, 1], dtype=type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    result = onp.nn.leakyrelu(onp.array(x), alpha=0.1)
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    result = onp.nn.leakyrelu(onp.array(x), alpha=0.1)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_leakyrelu_default(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.01
    result = onp.nn.leakyrelu(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_logsoftmax(type_a):
    x = np.array([[-1, 0, 1]]).astype(type_a)
    expected = np.array([[-2.4076061, -1.407606, -0.407606]]).astype(type_a)
    result = onp.nn.logsoftmax(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_logsoftmax_axis(type_a):

    def logsoftmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        tmp = np.exp(x - x_max)
        s = np.sum(tmp, axis=axis, keepdims=True)
        return (x - x_max) - np.log(s)

    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]
                 ).astype(type_a)
    expected = logsoftmax(x)
    result = onp.nn.logsoftmax(onp.array(x))
    expect(expected, result.numpy())

    x = np.abs(np.random.randn(3, 4, 5).astype(type_a))
    expected = logsoftmax(x, axis=0)
    result = onp.nn.logsoftmax(onp.array(x), axis=0)
    expect(expected, result.numpy())

    expected = logsoftmax(x, axis=1)
    result = onp.nn.logsoftmax(onp.array(x), axis=1)
    expect(expected, result.numpy())

    expected = logsoftmax(x, axis=2)
    result = onp.nn.logsoftmax(onp.array(x), axis=2)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_maxunpool_with_output_shape(type_a):
    xT = np.array([[[[5, 6],
                   [7, 8]]]], dtype=type_a)
    xI = np.array([[[[5, 7],
                   [13, 15]]]], dtype=np.int64)
    output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
    # FIXME: result expected in ONNX operators.md
    # expected = np.array([[[[0, 0, 0, 0, 0],
    #                        [0, 5, 0, 6, 0],
    #                        [0, 0, 0, 0, 0],
    #                        [0, 7, 0, 8, 0],
    #                        [0, 0, 0, 0, 0]]]], dtype=type_a)
    # result I am getting?
    expected = np.array([[[[0., 0., 0., 0., 0.],
                         [5., 0., 6., 0., 0.],
                         [0., 0., 0., 7., 0.],
                         [8., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]]]], dtype=type_a)
    result = onp.nn.maxunpool(onp.array(xT), onp.array(xI),
                              kernel_shape=[2, 2],
                              output_shape=onp.array(output_shape),
                              strides=[2, 2])
    expect(expected, result.numpy())


# @pytest.mark.parametrize("type_a", [np.float32])
# def test_maxunpool_without_output_shape(type_a):
#     xT = np.array([[[[5, 6],
#                      [7, 8]]]], dtype=type_a)
#     xI = np.array([[[[5, 7],
#                      [13, 15]]]], dtype=np.int64)
#     expected = np.array([[[[0, 0, 0, 0],
#                            [0, 1, 0, 2],
#                            [0, 0, 0, 0],
#                            [0, 3, 0, 4]]]], dtype=type_a)
#     result = onp.maxunpool(onp.array(xT), onp.array(
#         xI), kernel_shape=[2, 2], strides=[2, 2])
#     expect(expected, result.numpy())

@pytest.mark.parametrize("type_a", [np.float32])
def test_prelu(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    slope = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
    result = onp.nn.prelu(onp.array(x), onp.array(slope))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_prelu_broadcast(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    slope = np.random.randn(5).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
    result = onp.nn.prelu(onp.array(x), onp.array(slope))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_prelu_broadcast_scalar(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    slope = np.random.randn(1).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
    result = onp.nn.prelu(onp.array(x), onp.array(slope))
    expect(expected, result.numpy())

    result = onp.nn.prelu(onp.array(x), float(slope))
    expect(expected, result.numpy())


def scatter_elements(data, indices, updates, axis=0):  # type: ignore
    # taken from
    # https://github.com/onnx/onnx/blob/7e70e3dea265a958912b997a3fc8e8882d1e25c8/onnx/backend/test/case/node/scatterelements.py#L16
    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1:]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced indices for scattering of
    # updates param. in data
    idx = [[
        unpack(
            np.indices(idx_xsection_shape).reshape(
                indices.ndim - 1, -1)),
        indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]]
        for i in range(indices.shape[axis])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(
        axis, np.repeat(
            np.arange(indices.shape[axis]),
            np.prod(idx_xsection_shape)))

    scattered = np.copy(data)
    scattered[tuple(idx)] = updates[tuple(updates_idx)]
    return scattered


@pytest.mark.parametrize("type_a", all_types)
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_scatter_elements_with_axis(type_a, type_b):
    axis = 1
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=type_a)
    indices = np.array([[1, 3]], dtype=type_b)
    updates = np.array([[1.1, 2.1]], dtype=type_a)

    expected = scatter_elements(x, indices, updates, axis)
    result = onp.nn.scatter(
        onp.array(x),
        onp.array(indices),
        onp.array(updates),
        axis)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_scatter_elements_with_negative_indices(type_a, type_b):
    axis = 1
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=type_a)
    indices = np.array([[1, -3]], dtype=type_b)
    updates = np.array([[1.1, 2.1]], dtype=type_a)

    expected = scatter_elements(x, indices, updates, axis)
    result = onp.nn.scatter(
        onp.array(x),
        onp.array(indices),
        onp.array(updates),
        axis)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_scatter_elements_without_axis(type_a, type_b):
    x = np.zeros((3, 3), dtype=type_a)
    indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=type_b)
    updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=type_a)

    expected = scatter_elements(x, indices, updates)
    result = onp.nn.scatter(
        onp.array(x),
        onp.array(indices),
        onp.array(updates))
    expect(expected, result.numpy())


def scatter_nd_impl(data, indices, updates):
    # taken from https://github.com/onnx/onnx/blob/7e70e3dea265a958912b997a3fc8e8882d1e25c8/onnx/backend/test/case/node/scatternd.py#L15  # noqa

    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1]:]

    # Compute output
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        # NOTE: The order of iteration in this loop is not specified.
        # In particular, indices should not have duplicate entries:
        # that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
        # This ensures that the output value does not depend on the iteration order.
        output[indices[i]] = updates[i]
    return output


@pytest.mark.parametrize("type_a", all_types)
def test_scatter_nd(type_a):
    x = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
         [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
         [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
         [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=type_a)
    indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.array(
        [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
         [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=type_a)

    expected = scatter_nd_impl(x, indices, updates)
    result = onp.nn.scatter_nd(
        onp.array(x),
        onp.array(indices),
        onp.array(updates))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_selu(type_a):
    alpha = 2.0
    gamma = 3.0

    x = np.array([-1, 0, 1]).astype(type_a)

    expected = np.clip(
        x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
    result = onp.nn.selu(onp.array(x), alpha, gamma)
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(
        x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
    result = onp.nn.selu(onp.array(x), alpha, gamma)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_selu_default(type_a):
    default_alpha = 1.67326319217681884765625
    default_gamma = 1.05070102214813232421875

    x = np.array([-1, 0, 1]).astype(type_a)

    expected = np.clip(x, 0, np.inf) * default_gamma + (
        np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
    result = onp.nn.selu(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x, 0, np.inf) * default_gamma + (
        np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
    result = onp.nn.selu(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.float64])
def test_sigmoid(type_a):

    x = np.array([-1, 0, 1]).astype(type_a)
    expected = 1.0 / (1.0 + np.exp(np.negative(x)))
    result = onp.nn.sigmoid(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = 1.0 / (1.0 + np.exp(np.negative(x)))
    result = onp.nn.sigmoid(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_softplus(type_a):

    x = np.array([-1, 0, 1]).astype(type_a)

    expected = np.log(np.exp(x) + 1)
    result = onp.nn.softplus(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.log(np.exp(x) + 1)
    result = onp.nn.softplus(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_softsign(type_a):

    x = np.array([-1, 0, 1]).astype(type_a)

    expected = x / (1 + np.abs(x))
    result = onp.nn.softsign(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = x / (1 + np.abs(x))
    result = onp.nn.softsign(onp.array(x))
    expect(expected, result.numpy())
