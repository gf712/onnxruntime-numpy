import onnxruntime_numpy as onp
import numpy as np
import pytest
from onnxruntime_numpy.types import float_types, all_types


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_add(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    b = onp.array([1, 2, 3], dtype=type_a)

    expected = onp.array([2, 4, 6], dtype=type_a)
    result = onp.add(a, b)
    assert(np.allclose(expected.numpy(), result.numpy()))


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sub(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    b = onp.array([3, 2, 1], dtype=type_a)

    expected = onp.array([-2, 0, 2], dtype=type_a)
    result = onp.subtract(a, b)
    assert(np.allclose(expected.numpy(), result.numpy()))


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_divide(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    b = onp.array([3., 2., 1.], dtype=type_a)

    expected = onp.array([1./3., 1., 3.], dtype=type_a)
    result = onp.divide(a, b)
    assert(np.allclose(expected.numpy(), result.numpy()))


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_multiply(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    b = onp.array([3., 2., 1.], dtype=type_a)

    expected = onp.array([3., 4., 3.], dtype=type_a)
    result = onp.multiply(a, b)
    assert(np.allclose(expected.numpy(), result.numpy()))


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64, np.uint32, np.uint64])
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

    assert(np.allclose(expected.numpy(), result.numpy()))


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_equal(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    expected = np.equal(x, y)

    result = onp.array(x) == onp.array(y)

    assert np.allclose(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_equal_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) * 10).astype(type_a)
    y = (np.random.randn(5) * 10).astype(type_a)
    expected = np.equal(x, y)

    result = onp.array(x) == onp.array(y)

    assert np.allclose(expected, result.numpy())
