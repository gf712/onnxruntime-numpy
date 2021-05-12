import onnxruntime_numpy as onp
import numpy as np
import pytest
from onnxruntime_numpy.types import (
    float_types, numeric_types, bool_types, is_integer, all_types)
from .utils import expect


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_add(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    b = onp.array([1, 2, 3], dtype=type_a)

    expected = onp.array([2, 4, 6], dtype=type_a)
    result = onp.add(a, b)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", bool_types)
def test_and(type_a):
    x = (np.random.randn(3, 4) > 0).astype(type_a)
    y = (np.random.randn(3, 4) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", bool_types)
def test_and_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(5) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(4, 5) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(5, 6) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(4, 5, 6) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(3, 1, 5, 6) > 0).astype(type_a)
    expected = np.logical_and(x, y)
    result = onp.logical_and(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.uint8, np.uint32, np.uint64])
def test_left_shift(type_a):
    x = np.array([16, 4, 1]).astype(type_a)
    y = np.array([1, 2, 3]).astype(type_a)
    expected = x << y  # expected output [32, 16, 8]
    result = onp.array(x) << onp.array(y)
    expect(expected, result)


@pytest.mark.parametrize("type_a", [np.uint8, np.uint32, np.uint64])
def test_right_shift(type_a):
    x = np.array([16, 4, 1]).astype(type_a)
    y = np.array([1, 2, 3]).astype(type_a)
    expected = x >> y  # expected output [8, 1, 0]
    result = onp.array(x) >> onp.array(y)
    expect(expected, result)


@pytest.mark.parametrize("type_a", all_types)
def test_compress_axis_0(type_a):
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(type_a)
    condition = np.array([0, 1, 1])
    expected = np.compress(condition, x, axis=0)
    result = onp.compress(
        onp.array(x),
        onp.array(condition.astype(np.bool_)),
        axis=0)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_compress_axis_1(type_a):
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(type_a)
    condition = np.array([0, 1])
    expected = np.compress(condition, x, axis=1)
    result = onp.compress(
        onp.array(x),
        onp.array(condition.astype(np.bool_)),
        axis=1)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_compress_default_axis(type_a):
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(type_a)
    condition = np.array([0, 1, 0, 0, 1])
    expected = np.compress(condition, x)
    result = onp.compress(
        onp.array(x),
        onp.array(condition.astype(np.bool_)))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_compress_negative_axis(type_a):
    x = np.array([[1, 2], [3, 4], [5, 6]]).astype(type_a)
    condition = np.array([0, 1])
    expected = np.compress(condition, x, axis=-1)
    result = onp.compress(
        onp.array(x),
        onp.array(condition.astype(np.bool_)), axis=-1)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sub(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    b = onp.array([3, 2, 1], dtype=type_a)
    expected = onp.array([-2, 0, 2], dtype=type_a)
    result = onp.subtract(a, b)
    expect(expected.numpy(), result.numpy())

    a = np.random.randn(3, 4, 5).astype(type_a)
    b = np.random.randn(3, 4, 5).astype(type_a)
    expected = a - b
    result = onp.subtract(onp.array(a), onp.array(b))
    expect(expected, result.numpy())

    expect(expected, (onp.array(a) - onp.array(b)).numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sub_broadcast(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    y = np.random.randn(5).astype(type_a)
    expected = x - y
    result = onp.subtract(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_divide(type_a):
    x = np.array([3, 4]).astype(type_a)
    y = np.array([1, 2]).astype(type_a)
    expected = (x / y).astype(type_a)
    result = onp.divide(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    if is_integer(type_a):
        x = np.random.randint(1, 10, size=(3, 4, 5)).astype(type_a)
        y = np.random.randint(0, 10, size=(3, 4, 5)).astype(type_a) + 1
    else:
        x = np.random.randn(3, 4, 5).astype(type_a)
        y = np.random.randn(3, 4, 5).astype(type_a) + 1
    expected = (x / y).astype(type_a)
    result = onp.divide(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    expect(expected, (onp.array(x) / onp.array(y)).numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_divide_broadcast(type_a):
    if is_integer(type_a):
        x = np.random.randint(1, 10, size=(3, 4, 5)).astype(type_a)
        y = np.random.randint(1, 10, size=(5)).astype(type_a)
    else:
        x = np.random.randn(3, 4, 5).astype(type_a)
        y = np.random.randn(5).astype(type_a)
    expected = (x / y).astype(type_a)
    result = onp.divide(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


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


@pytest.mark.parametrize(
    "type_a", [np.float32])
def test_mean(type_a):
    data_0 = np.array([3, 0, 2]).astype(type_a)
    data_1 = np.array([1, 3, 4]).astype(type_a)
    data_2 = np.array([2, 6, 6]).astype(type_a)
    expected = np.array([2, 3, 4]).astype(type_a)
    result = onp.mean(
        onp.array(data_0),
        onp.array(data_1),
        onp.array(data_2))
    expect(expected, result.numpy())

    result = onp.mean(onp.array(data_0))
    expect(data_0, result.numpy())

    result = onp.mean(onp.array(data_0), onp.array(data_1))
    expected = np.divide(np.add(data_0, data_1), 2.).astype(type_a)
    expect(expected, result.numpy())


@pytest.mark.parametrize(
    "type_a", [*float_types, np.uint32, np.uint64, np.int32, np.int64])
def test_minimum(type_a):
    data_0 = np.array([3, 2, 1]).astype(type_a)
    data_1 = np.array([1, 4, 4]).astype(type_a)
    data_2 = np.array([2, 5, 3]).astype(type_a)
    expected = np.array([1, 2, 1]).astype(type_a)
    result = onp.minimum(
        onp.array(data_0),
        onp.array(data_1),
        onp.array(data_2))
    expect(expected, result.numpy())

    result = onp.minimum(onp.array(data_0))
    expect(data_0, result.numpy())

    result = onp.minimum(onp.array(data_0), onp.array(data_1))
    expected = np.minimum(data_0, data_1)
    expect(expected, result.numpy())


@pytest.mark.parametrize(
    "type_a", numeric_types)
def test_mod_broadcast(type_a):
    x = np.arange(0, 30).reshape([3, 2, 5]).astype(type_a)
    y = np.array([7]).astype(type_a)
    expected = np.mod(x, y)
    result = onp.mod(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


def test_mod_int64_fmod():
    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
    expected = np.fmod(x, y)
    result = onp.mod(onp.array(x), onp.array(y), fmod=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize(
    "type_a", numeric_types)
def test_mod_mixed_sign(type_a):
    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(type_a)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(type_a)
    expected = np.fmod(x, y) if type_a in float_types else np.mod(x, y)
    result = onp.mod(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_multiply(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    b = onp.array([3., 2., 1.], dtype=type_a)
    expected = onp.array([3., 4., 3.], dtype=type_a)
    result = onp.multiply(a, b)
    expect(expected.numpy(), result.numpy())

    a = np.random.uniform(low=0, high=10, size=(3, 4, 5)).astype(type_a)
    b = np.random.uniform(low=0, high=10, size=(3, 4, 5)).astype(type_a)
    expected = a * b
    result = onp.multiply(onp.array(a), onp.array(b))
    expect(expected, result.numpy())

    expect(expected, (onp.array(a) * onp.array(b)).numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_multiply_broadcast(type_a):
    x = np.random.uniform(low=0, high=10, size=(3, 4, 5)).astype(type_a)
    y = np.random.uniform(low=0, high=10, size=5).astype(type_a)
    expected = x * y

    result = onp.multiply(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", bool_types)
def test_or(type_a):
    x = (np.random.randn(3, 4) > 0).astype(type_a)
    y = (np.random.randn(3, 4) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", bool_types)
def test_or_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(5) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(4, 5) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(5, 6) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(4, 5, 6) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(3, 1, 5, 6) > 0).astype(type_a)
    expected = np.logical_or(x, y)
    result = onp.logical_or(onp.array(x), onp.array(y))
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


@pytest.mark.parametrize("type_a", all_types)
def test_tile(type_a):
    x = np.random.rand(2, 3, 4, 5).astype(type_a)
    repeats = np.random.randint(
        low=1, high=10, size=(np.ndim(x),)).astype(
        np.int64)

    expected = np.tile(x, repeats)
    result = onp.tile(onp.array(x), onp.array(repeats))
    expect(expected, result.numpy())


# TODO
# @pytest.mark.parametrize("type_a", all_types)
# def test_tile_lazy(type_a):
#     x = np.random.rand(2, 3, 4, 5).astype(type_a)
#     repeats = [4, 6, 12, 16]
#     expected = np.tile(x, repeats)

#     repeats = onp.array([2, 3, 6, 8], np.int64)
#     repeats += repeats
#     result = onp.tile(onp.array(x), repeats)
#     expect(expected, result.numpy())


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


@pytest.mark.parametrize("type_a", [np.bool_])
def test_xor(type_a):
    x = (np.random.randn(3, 4) > 0).astype(type_a)
    y = (np.random.randn(3, 4) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.bool_])
def test_xor_broadcast(type_a):
    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(5) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    y = (np.random.randn(4, 5) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(5, 6) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(4, 5, 6) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    y = (np.random.randn(3, 1, 5, 6) > 0).astype(type_a)
    expected = np.logical_xor(x, y)
    result = onp.logical_xor(onp.array(x), onp.array(y))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64, np.uint8])
def test_where(type_a):
    condition = np.array([[1, 0], [1, 1]], dtype=np.bool_)
    x = np.array([[1, 2], [3, 4]], dtype=type_a)
    y = np.array([[9, 8], [7, 6]], dtype=type_a)
    expected = np.where(condition, x, y)
    result = onp.where(onp.array(condition), onp.array(x), onp.array(y))

    expect(expected, result)


# TODO: fix broadcasting with more than two arrays
# @pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64, np.uint8])
# def test_where_broadcast(type_a):
#     condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
#     x = np.array([[1, 2], [3, 4]], dtype=type_a)
#     y = np.array([[9, 8], [7, 6]], dtype=type_a)
#     expected = np.where(condition, x, y)
#     result = onp.where(onp.array(condition), onp.array(x), onp.array(y))

#     expect(expected, result)
