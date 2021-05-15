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
    g, c = onp.grad(y, x, y)
    gradient_check(onp.cos, x, g)
    expect(np.ones(y.shape.tolist(), dtype=np.float32), c.numpy())


def test_abs_grad():
    x = onp.array([.1, -.2, -.3], dtype=np.float32)
    y = onp.absolute(x)
    g, = onp.grad(y, x)
    gradient_check(onp.absolute, x, g)


def test_acos_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.acos(x)
    g, = onp.grad(y, x)
    gradient_check(onp.acos, x, g)


def test_acosh_grad():
    x = onp.array([1.1, 2.2, 3.3], dtype=np.float32)
    y = onp.acosh(x)
    g, = onp.grad(y, x)
    gradient_check(onp.acosh, x, g)


def test_add_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.array([.1, .2, .3], dtype=np.float32)
    z = x + y
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: el + y, x, dx)
    gradient_check(lambda el: x + el, y, dy)

    z = onp.exp(x) + onp.cos(y)
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: onp.exp(el) + onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) + onp.cos(el), y, dy)


def test_asin_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.asin(x)
    g, = onp.grad(y, x)
    gradient_check(onp.asin, x, g)


def test_asihnh_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.asin(x)
    g, = onp.grad(y, x)
    gradient_check(onp.asinh, x, g)


def test_atan_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.atan(x)
    g, = onp.grad(y, x)
    gradient_check(onp.atan, x, g)


def test_atanh_grad():
    x = onp.array([.1, .2, .3], dtype=np.float32)
    y = onp.atanh(x)
    g, = onp.grad(y, x)
    gradient_check(onp.atanh, x, g)


def test_cos_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.cos(x)
    g, = onp.grad(y, x)
    gradient_check(onp.cos, x, g)


def test_cosh_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.cosh(x)
    g, = onp.grad(y, x)
    gradient_check(onp.cosh, x, g)


def test_divide_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.array([.1, .332, .1], dtype=np.float32)
    z = x / y
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: el / y, x, dx)
    gradient_check(lambda el: x / el, y, dy)

    z = onp.exp(x) / onp.cos(y)
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: onp.exp(el) / onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) / onp.cos(el), y, dy)


def test_exp_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.exp(x)
    g, = onp.grad(y, x)
    gradient_check(onp.exp, x, g)


def test_log_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.log(x)
    g, = onp.grad(y, x)
    gradient_check(onp.log, x, g)


def test_matmul_grad():
    a = onp.array([[1, 2, 3]], dtype=np.float32)
    b = onp.array([[1], [2], [3]], dtype=np.float32)

    y = onp.matmul(a, b)
    da, db = onp.grad(y, a, b)
    expect(a.numpy(), da.numpy())
    expect(b.numpy(), db.numpy())


def test_multiply_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.array([.1, .332, .1], dtype=np.float32)
    z = x * y
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: el * y, x, dx)
    gradient_check(lambda el: x * el, y, dy)

    z = onp.exp(x) * onp.cos(y)
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: onp.exp(el) * onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) * onp.cos(el), y, dy)


def test_power_grad():
    x = onp.array([8.1, 2.5, 3], dtype=np.float32)
    y = onp.array([1.3, 2, 3.4], dtype=np.float32)

    z = onp.power(x, y)
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: onp.power(el, y), x, dx)
    gradient_check(lambda el: onp.power(x, el), y, dy)

    z = x ** 2.
    dx, = onp.grad(z, x)
    expect(dx.numpy(), (x * onp.array(2.)).numpy())


def test_sin_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.sin(x)
    g, = onp.grad(y, x)
    gradient_check(onp.sin, x, g)


def test_sinh_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.sinh(x)
    g, = onp.grad(y, x)
    gradient_check(onp.sinh, x, g)


# def test_square_grad():
#     x = onp.array([1, 2, 3], dtype=np.float32)
#     y = onp.square(x)
#     g, = onp.grad(y, x)
#     gradient_check(onp.square, x, g)

#     y = onp.exp(onp.square(x))
#     g, = onp.grad(y, x)
#     gradient_check(lambda a: onp.exp(onp.square(a)), x, g)


def test_subtract_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.array([.1, .332, .1], dtype=np.float32)
    z = x - y
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: el - y, x, dx)
    gradient_check(lambda el: x - el, y, dy)

    z = onp.exp(x) - onp.cos(y)
    dx, dy = onp.grad(z, x, y)
    gradient_check(lambda el: onp.exp(el) - onp.cos(y), x, dx)
    gradient_check(lambda el: onp.exp(x) - onp.cos(el), y, dy)


def test_sqrt_grad():
    x = onp.array([.3, .2,   .4], dtype=np.float32)
    y = onp.sqrt(x)
    dx, = onp.grad(y, x)
    gradient_check(onp.sqrt, x, dx)

    y = onp.exp(onp.sqrt(x)) - onp.cos(onp.sqrt(x))
    dx, = onp.grad(y, x)
    gradient_check(lambda el: onp.exp(
        onp.sqrt(el)) - onp.cos(onp.sqrt(el)), x, dx)

    y = onp.exp(x**.5) - onp.cos(x**.5)
    dx, = onp.grad(y, x)
    gradient_check(lambda el: onp.exp(el**.5) - onp.cos(el**.5), x, dx)


def test_tan_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.tan(x)
    g, = onp.grad(y, x)
    gradient_check(onp.tan, x, g)


def test_tanh_grad():
    x = onp.array([1, 2, 3], dtype=np.float32)
    y = onp.tanh(x)
    g, = onp.grad(y, x)
    gradient_check(onp.tanh, x, g)
