import pytest
import onnxruntime_numpy as onp
from onnxruntime_numpy.types import all_types
import numpy as np
from .utils import expect


@pytest.mark.parametrize("type_a", all_types)
def test_slice(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    expected = x[0:3, 0:10]
    result = onp.array(x)[0:3, 0:10]

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_default_axes(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    expected = x[:, :, 3:4]
    result = onp.slice(onp.array(x), onp.array(starts), onp.array(ends))

    expect(expected, result.numpy())

    result = onp.array(x)[:, :, 3:4]
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_default_steps(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    expected = x[:, :, 3:4]
    result = onp.slice(
        onp.array(x),
        onp.array(starts),
        onp.array(ends),
        onp.array(axes))

    expect(expected, result.numpy())

    result = onp.array(x)[:, :, 3:4]
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_end_out_of_bounds(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    expected = x[:, 1:1000]
    result = onp.slice(
        onp.array(x),
        onp.array(starts),
        onp.array(ends),
        onp.array(axes),
        onp.array(steps))

    expect(expected, result.numpy())

    result = onp.array(x)[:, 1:1000]
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_negative(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    expected = x[:, 0:-1]
    result = onp.slice(
        onp.array(x),
        onp.array(starts),
        onp.array(ends),
        onp.array(axes),
        onp.array(steps))

    expect(expected, result.numpy())

    result = onp.array(x)[:, 0:-1]
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_neg_steps(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([20, 10, 4], dtype=np.int64)
    # -2**31 == INT32_MIN, which means end of array
    ends = np.array([0, -2**31, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2]).astype(np.int64)
    expected = x[20:0:-1, 10::-3, 4:1:-2]
    result = onp.slice(
        onp.array(x),
        onp.array(starts),
        onp.array(ends),
        onp.array(axes),
        onp.array(steps))

    expect(expected, result.numpy())

    result = onp.array(x)[20:0:-1, 10::-3, 4:1:-2]
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_negative_axes(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, -2, -1], dtype=np.int64)
    expected = x[:, :, 3:4]
    result = onp.slice(
        onp.array(x),
        onp.array(starts),
        onp.array(ends),
        onp.array(axes))

    expect(expected, result.numpy())

    result = onp.array(x)[:, :, 3:4]
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_slice_start_out_of_bounds(type_a):
    x = np.random.randn(20, 10, 5).astype(type_a)
    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    expected = x[:, 1000:1000]
    result = onp.slice(
        onp.array(x),
        onp.array(starts),
        onp.array(ends),
        onp.array(axes),
        onp.array(steps))

    expect(expected, result.numpy())

    result = onp.array(x)[:, 1000:1000]
    expect(expected, result.numpy())
