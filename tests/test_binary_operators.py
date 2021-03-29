import onnxruntime_numpy as onp
import numpy as np
import pytest
from onnxruntime_numpy.types import float_types
from .utils import expect


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_add(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    b = onp.array([1, 2, 3], dtype=type_a)

    expected = onp.array([2, 4, 6], dtype=type_a)
    result = onp.add(a, b)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sub(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    b = onp.array([3, 2, 1], dtype=type_a)

    expected = onp.array([-2, 0, 2], dtype=type_a)
    result = onp.subtract(a, b)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_divide(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    b = onp.array([3., 2., 1.], dtype=type_a)

    expected = onp.array([1./3., 1., 3.], dtype=type_a)
    result = onp.divide(a, b)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_multiply(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    b = onp.array([3., 2., 1.], dtype=type_a)

    expected = onp.array([3., 4., 3.], dtype=type_a)
    result = onp.multiply(a, b)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_equal(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    expected = np.equal(x, y)

    result = onp.array(x) == onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_equal_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(5) * 10).astype(type_a)
    expected = np.equal(x, y)

    result = onp.array(x) == onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_greater(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    expected = np.greater(x, y)

    result = onp.array(x) > onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_greater_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(5) * 10).astype(type_a)
    expected = np.greater(x, y)

    result = onp.array(x) > onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_greater_equal(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    expected = np.greater_equal(x, y)

    result = onp.array(x) >= onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_greater_equal_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(5) * 10).astype(type_a)
    expected = np.greater_equal(x, y)

    result = onp.array(x) >= onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_less(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    expected = np.less(x, y)

    result = onp.array(x) < onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_less_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(5) * 10).astype(type_a)
    expected = np.less(x, y)

    result = onp.array(x) < onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_less_equal(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    expected = np.less_equal(x, y)

    result = onp.array(x) <= onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_less_equal_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(5) * 10).astype(type_a)
    expected = np.less_equal(x, y)

    result = onp.array(x) <= onp.array(y)

    expect(expected, result.numpy())


@pytest.mark.parametrize(
    "type_a", [*float_types, np.int32, np.int64, np.uint32, np.uint64])
def test_matmul(type_a):
    A = onp.array([[[0,  1,  2,  3], [4,  5,  6,  7]],
                   [[8,  9, 10, 11], [12, 13, 14, 15]]], dtype=type_a)
    B = onp.array([[[0,  1],
                    [2,  3],
                    [4,  5],
                    [6,  7]],
                   [[8,  9],
                    [10, 11],
                    [12, 13],
                    [14, 15]]], dtype=type_a)
    expected = onp.array([[[28,  34],
                           [76,  98]],
                          [[428, 466],
                           [604, 658]]], dtype=type_a)

    result = A @ B

    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.uint8, np.int8])
@pytest.mark.parametrize("type_b", [np.uint8, np.int8])
def test_matmul_integer(type_a, type_b):
    if (type_a == np.int8 and type_b == np.uint8) or \
       (type_a == np.int8 and type_b == np.int8):
        return
    A = np.array([[11, 7, 3],
                  [10, 6, 2],
                  [9, 5, 1],
                  [8, 4, 0], ], dtype=type_a)

    a_zero_point = np.array(12, dtype=type_a)

    B = np.array([[1, 4],
                  [2, 5],
                  [3, 6], ], dtype=type_b)

    b_zero_point = np.array(0, dtype=type_b)

    expected = np.array([[-38, -83],
                         [-44, -98],
                         [-50, -113],
                         [-56, -128], ], dtype=np.int32)

    result = onp.matmul_integer(
        onp.array(A),
        onp.array(B),
        onp.array(a_zero_point),
        onp.array(b_zero_point))

    expect(expected, result.numpy())


@pytest.mark.parametrize(
    "type_a", [*float_types, np.uint32, np.uint64, np.int32, np.int64])
def test_maximum(type_a):
    data_0 = np.array([3, 2, 1]).astype(type_a)
    data_1 = np.array([1, 4, 4]).astype(type_a)
    data_2 = np.array([2, 5, 3]).astype(type_a)
    expected = np.array([3, 5, 4]).astype(type_a)
    result = onp.maximum(
        onp.array(data_0),
        onp.array(data_1),
        onp.array(data_2))
    expect(expected, result.numpy())

    result = onp.maximum(onp.array(data_0))
    expect(data_0, result.numpy())

    result = onp.maximum(onp.array(data_0), onp.array(data_1))
    expected = np.maximum(data_0, data_1)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
@pytest.mark.parametrize("type_b", [*float_types, np.int32, np.int64])
def test_pow(type_a, type_b):
    x = np.array([1, 2, 3]).astype(type_a)
    y = np.array([4, 5, 6]).astype(type_b)
    expected = np.power(x, y).astype(type_a)
    result = onp.power(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
@pytest.mark.parametrize("type_b", [*float_types, np.int32, np.int64])
def test_pow_operator(type_a, type_b):
    x = np.array([1, 2, 3]).astype(type_a)
    expected = np.power(x, 2).astype(type_a)
    result = onp.array(x) ** 2
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
@pytest.mark.parametrize("type_b", [*float_types, np.int32, np.int64])
def test_pow_broadcast(type_a, type_b):
    x = np.array([1, 2, 3]).astype(type_a)
    y = np.array(2).astype(type_b)
    expected = np.power(x, y).astype(type_a)
    result = onp.power(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(type_a)
    y = np.array([1, 2, 3]).astype(type_b)
    expected = np.power(x, y).astype(type_a)
    result = onp.power(onp.array(x), onp.array(y))
    expect(expected, result.numpy())
