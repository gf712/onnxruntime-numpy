import onnxruntime_numpy as onp
# from onnxruntime_numpy.types import float_types
import numpy as np
# import pytest
from .utils import expect


def get_epsilon(dtype: np.dtype):
    if dtype == np.float32:
        return 1e-04
    elif dtype == np.float64:
        return 1e-07
    else:
        raise ValueError("")


def get_rtol(dtype: np.dtype):
    if dtype == np.float32:
        return 1e-01
    elif dtype == np.float64:
        return 1e-04
    else:
        raise ValueError("")


def gradient_check(f, x, result):
    epsilon = get_epsilon(x.dtype)
    step = onp.array(epsilon, dtype=x.dtype)
    delta = onp.array(epsilon * 2, dtype=x.dtype)
    gradient_approx = (f(x + step) - f(x - step)) / delta
    expect(gradient_approx.numpy(), result.numpy(), rtol=get_rtol(x.dtype))


def test_gradient_wrt_output():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.cos(x)
    g, c = onp.gradients(y, [x, y])
    gradient_check(onp.cos, x, g)
    expect(np.ones(y.shape.tolist(), dtype=np.float32), c.numpy())


def test_gradient_independent_variable():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.array([1, 2, 3], dtype=np.float32)
    g, = onp.gradients(
        y, [x],
        unconnected_gradients=onp.StopGradientsValue.NONE)
    assert g is None

    # TODO
    # g, = onp.gradients(
    #     y, [x],
    #     unconnected_gradients=onp.StopGradientsValue.Zero)
    # expect(np.zeros(y.shape.tolist(), dtype=np.float32), g.numpy())


def test_stop_gradient():
    a = onp.array([[1., 2, 3], [4, 5, 6]], dtype=np.float32)
    b = onp.sin(a)
    c = onp.cos(a)
    d = b * c
    e = onp.tan(b)
    f = d - e
    g = onp.exp(f)

    #     b --- e
    #   /   \    \
    # a      d -- f -- g
    #   \   /
    #     c
    # here we stop gradients from propagating through d

    grads = onp.gradients(g, [a, b, c, d, e, f, g], [d])

    # computed using tf.GradientTape.gradients
    expected_results = [
        np.array([[-0.6261989,  0.20907097,  0.7619696],
                  [5.2144356, -2.723631, -1.0587629]], dtype=np.float32),
        np.array([[-1.1589788, -0.5023971, -0.7696721],
                  [-7.9774904, -9.60167, -1.1026825]], dtype=np.float32),
        None,
        None,
        np.array([[-0.51463836, -0.18958703, -0.7544456],
                  [-4.216743, -3.16794, -1.0188099]], dtype=np.float32),
        np.array([[0.51463836, 0.18958703, 0.7544456],
                  [4.216743, 3.16794, 1.0188099]], dtype=np.float32),
        np.array([[1., 1., 1.],
                  [1., 1., 1.]], dtype=np.float32)
    ]

    for result, expected in zip(grads, expected_results):
        if result is None and expected is None:
            assert True
        else:
            expect(expected, result.numpy())


def test_abs_grad():
    x = onp.array([.1, -.2, -.3], dtype=np.float32)
    y = onp.absolute(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.absolute, x, g)


def test_acos_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.acos(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.acos, x, g)


def test_acosh_grad():
    x = onp.array([1.1, 2.2, 3.3], dtype=np.float32)
    y = onp.acosh(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.acosh, x, g)


def test_add_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.array([.1, .2, .3], dtype=np.float32)
    z = x + y
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: el + y, x, dx)
    gradient_check(lambda el: x + el, y, dy)

    z = onp.exp(x) + onp.cos(y)
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: onp.exp(el) + onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) + onp.cos(el), y, dy)


def test_asin_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.asin(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.asin, x, g)


def test_asihnh_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.asin(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.asinh, x, g)


def test_atan_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.atan(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.atan, x, g)


def test_atanh_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.atanh(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.atanh, x, g)


def test_cos_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.cos(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.cos, x, g)


def test_cosh_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.cosh(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.cosh, x, g)


def test_divide_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.array([.1, .332, .1], dtype=np.float32)
    z = x / y
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: el / y, x, dx)
    gradient_check(lambda el: x / el, y, dy)

    z = onp.exp(x) / onp.cos(y)
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: onp.exp(el) / onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) / onp.cos(el), y, dy)


def test_exp_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.exp(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.exp, x, g)


def test_log_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.log(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.log, x, g)


def test_matmul_grad1():
    a = onp.array([[1, 2, 3]], dtype=np.float32)
    b = onp.array([[1], [2], [3]], dtype=np.float32)

    y = onp.matmul(a, b)
    da, db = onp.gradients(y, [a, b])
    expect(a.numpy(), da.numpy())
    expect(b.numpy(), db.numpy())


def test_matmul_grad2():
    a = onp.array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=np.float32)
    b = onp.array([[1., 2., 3.],
                   [4., 5., 6.],
                   [7., 8., 9.]], dtype=np.float32)

    y = onp.matmul(a, b)
    da, db = onp.gradients(y, [a, b])
    da_expected = np.array([[6., 15., 24.],
                            [6., 15., 24.]], dtype=np.float32)
    db_expected = np.array([[5., 5., 5.],
                            [7., 7., 7.],
                            [9., 9., 9.]], dtype=np.float32)
    expect(da_expected, da.numpy())
    expect(db_expected, db.numpy())


def test_reduce_mean_grad_no_axis():
    a = onp.array([1, 2, 3, 4], dtype=np.float32)
    y = a.mean()
    expected_da = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    da, = onp.gradients(y, [a])
    expect(expected_da, da.numpy())

    a = onp.array([[1, 2, 3, 4]], dtype=np.float32)
    y = a.mean()
    expected_da = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    da, = onp.gradients(y, [a])
    expect(expected_da, da.numpy())

    a = onp.array([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=np.float32)
    y = a.mean()
    expected_da = np.array(
        [[0.125, 0.125, 0.125, 0.125],
         [0.125, 0.125, 0.125, 0.125]],
        dtype=np.float32)
    da, = onp.gradients(y, [a])
    expect(expected_da, da.numpy())


def test_reduce_mean_grad_axis_1():
    a = onp.array([[1, 2, 3, 4]], dtype=np.float32)
    y = a.mean(1)
    expected_da = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    da, = onp.gradients(y, [a])
    expect(expected_da, da.numpy())

    a = onp.array([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=np.float32)
    y = a.mean(1)
    expected_da = np.array(
        [[0.25, 0.25, 0.25, 0.25],
         [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32)
    da, = onp.gradients(y, [a])
    expect(expected_da, da.numpy())

    a = onp.array([[1, 2, 3, 4]], dtype=np.float32)
    y = a.mean(1)
    z = a + y
    expected_da = np.array([[2., 2., 2., 2.]], dtype=np.float32)
    expected_dy = np.array([[4.]], dtype=np.float32)

    da, dy = onp.gradients(z, [a, y])
    expect(expected_da, da.numpy())
    expect(expected_dy, dy.numpy())


def test_multiply_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.array([.1, .332, .1], dtype=np.float32)
    z = x * y
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: el * y, x, dx)
    gradient_check(lambda el: x * el, y, dy)

    z = onp.exp(x) * onp.cos(y)
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: onp.exp(el) * onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) * onp.cos(el), y, dy)


def test_power_grad():
    x = onp.array([8.1, 2.5, 3], dtype=np.float32)
    y = onp.array([1.3, 2, 3.4], dtype=np.float32)

    z = onp.power(x, y)
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: onp.power(el, y), x, dx)
    gradient_check(lambda el: onp.power(x, el), y, dy)

    z = x ** 2.
    dx, = onp.gradients(z, [x])
    expect(dx.numpy(), (x * onp.array(2.)).numpy())


def test_relu_grad():
    x = onp.array([-3, -2, -1, -.1, .1, 1, 2, 3], dtype=np.float32)
    y = onp.nn.relu(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.nn.relu, x, g)


def test_sin_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.sin(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.sin, x, g)


def test_sinh_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.sinh(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.sinh, x, g)


def test_square_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.square(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.square, x, g)

    y = onp.exp(onp.square(x))
    g, = onp.gradients(y, [x])
    gradient_check(lambda a: onp.exp(onp.square(a)), x, g)


def test_subtract_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.array([.1, .332, .1], dtype=np.float32)
    z = x - y
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: el - y, x, dx)
    gradient_check(lambda el: x - el, y, dy)

    z = onp.exp(x) - onp.cos(y)
    dx, dy = onp.gradients(z, [x, y])
    gradient_check(lambda el: onp.exp(el) - onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) - onp.cos(el), y, dy)


def test_sqrt_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.sqrt(x)
    dx, = onp.gradients(y, [x])
    gradient_check(onp.sqrt, x, dx)

    y = onp.exp(onp.sqrt(x)) - onp.cos(onp.sqrt(x))
    dx, = onp.gradients(y, [x])
    gradient_check(lambda el: onp.exp(
        onp.sqrt(el)) - onp.cos(onp.sqrt(el)), x, dx)

    y = onp.exp(x**.5) - onp.cos(x**.5)
    dx, = onp.gradients(y, [x])
    gradient_check(lambda el: onp.exp(el**.5) - onp.cos(el**.5), x, dx)


def test_tan_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.tan(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.tan, x, g)


def test_tanh_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.tanh(x)
    g, = onp.gradients(y, [x])
    gradient_check(onp.tanh, x, g)
