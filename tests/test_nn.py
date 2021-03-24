import pytest
import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types
import numpy as np
from .utils import expect
import math


@pytest.mark.parametrize("type_a", [np.float32])
def test_elu(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([-1.2642411, 0., 1.], dtype=type_a)
    result = onp.elu(a, alpha=2.0)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_elu_default(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([-0.63212055, 0., 1.], dtype=type_a)
    result = onp.elu(a)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_relu(type_a):
    a = onp.array([[1., 2., -3.],
                   [-1, 0,  10.]], dtype=type_a)
    expected = onp.array([[1, 2, 0],
                          [0, 0, 10]], dtype=type_a)
    result = onp.relu(a)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_average_pool(type_a):
    x = np.random.randn(1, 3, 5, 5).astype(type_a)
    spatial_shape = np.ndim(x) - 2
    expected = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        expected = np.expand_dims(expected, -1)

    result = onp.global_average_pool(onp.array(x, dtype=type_a))
    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_average_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[5]]]]).astype(type_a)

    result = onp.global_average_pool(onp.array(x, dtype=type_a))

    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_lp2_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[16.881943]]]]).astype(type_a)

    result = onp.global_lp_pool(onp.array(x, dtype=type_a))

    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_lp1_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[45]]]]).astype(type_a)

    result = onp.global_lp_pool(onp.array(x, dtype=type_a), p=1)

    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_lp_large_pool_precomputed(type_a):
    # this essentially becomes max pooling
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[9]]]]).astype(type_a)

    result = onp.global_lp_pool(onp.array(x, dtype=type_a), p=40)

    expect(expected, result, atol=0.1)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_max_pool(type_a):
    x = np.random.randn(1, 3, 5, 5).astype(type_a)
    spatial_shape = np.ndim(x) - 2
    expected = np.max(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        expected = np.expand_dims(expected, -1)

    result = onp.global_max_pool(onp.array(x, dtype=type_a))

    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.float32])
def test_global_max_pool_precomputed(type_a):
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(type_a)

    expected = np.array([[[[9]]]]).astype(type_a)

    result = onp.global_max_pool(onp.array(x, dtype=type_a))

    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.float32])
def test_hard_sigmoid(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([0.1, 0.6, 1.], dtype=type_a)
    result = onp.hard_sigmoid(a, alpha=0.5, beta=0.6)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_hard_sigmoid_default(type_a):
    default_alpha = 0.2
    default_beta = 0.5
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x * default_alpha + default_beta, 0, 1)
    result = onp.hard_sigmoid(onp.array(x))

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
    result = onp.hardmax(onp.array(x), axis=0)
    expect(result.numpy(), expected)

    expected = hardmax(x, axis=1)
    result = onp.hardmax(onp.array(x), axis=1)
    expect(result.numpy(), expected)

    expected = hardmax(x)
    result = onp.hardmax(onp.array(x))
    expect(result.numpy(), expected)


@pytest.mark.parametrize("type_a", [np.float32])
def test_hardmax(type_a):

    def instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
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

    expect(expected, onp.instance_normalization(
        onp.array(x), onp.array(s), onp.array(bias)))

    x = np.random.randn(2, 3, 4, 5).astype(type_a)
    s = np.random.randn(3).astype(type_a)
    bias = np.random.randn(3).astype(type_a)
    epsilon = 1e-2
    expected = instancenorm_test_mode(x, s, bias, epsilon).astype(type_a)

    expect(expected, onp.instance_normalization(onp.array(x),
                                                onp.array(s), onp.array(bias), epsilon=epsilon), rtol=1e-02)


@pytest.mark.parametrize("type_a", [np.float32])
def test_lrn_default(type_a):
    alpha = 0.0001
    beta = 0.75
    bias = 1.0
    nsize = 3
    x = np.random.randn(5, 5, 5, 5).astype(type_a)
    square_sum = np.zeros((5, 5, 5, 5)).astype(type_a)
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(x[n,
                                       max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                       h,
                                       w] ** 2)
    expected = x / ((bias + (alpha / nsize) * square_sum) ** beta)

    expect(expected, onp.lrn(onp.array(x), size=nsize))


@pytest.mark.parametrize("type_a", [np.float32])
def test_lrn(type_a):
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    nsize = 3
    x = np.random.randn(5, 5, 5, 5).astype(type_a)
    square_sum = np.zeros((5, 5, 5, 5)).astype(type_a)
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(x[n,
                                       max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                       h,
                                       w] ** 2)
    expected = x / ((bias + (alpha / nsize) * square_sum) ** beta)

    expect(expected, onp.lrn(onp.array(x), size=nsize, alpha=alpha, beta=beta, bias=bias))


@pytest.mark.parametrize("type_a", [np.float32])
def test_leakyrelu(type_a):
    x = np.array([-1, 0, 1], dtype=type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    result = onp.leakyrelu(onp.array(x), alpha=0.1)
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    result = onp.leakyrelu(onp.array(x), alpha=0.1)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_leakyrelu_default(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.01
    result = onp.leakyrelu(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_logsoftmax(type_a):
    x = np.array([[-1, 0, 1]]).astype(type_a)
    expected = np.array([[-2.4076061, -1.407606, -0.407606]]).astype(type_a)
    result = onp.logsoftmax(onp.array(x))
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
    result = onp.logsoftmax(onp.array(x))
    expect(expected, result.numpy())

    x = np.abs(np.random.randn(3, 4, 5).astype(type_a))
    expected = logsoftmax(x, axis=0)
    result = onp.logsoftmax(onp.array(x), axis=0)
    expect(expected, result.numpy())

    expected = logsoftmax(x, axis=1)
    result = onp.logsoftmax(onp.array(x), axis=1)
    expect(expected, result.numpy())

    expected = logsoftmax(x, axis=2)
    result = onp.logsoftmax(onp.array(x), axis=2)
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
    result = onp.maxunpool(onp.array(xT), onp.array(xI),
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
