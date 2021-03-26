import pytest
import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types
import numpy as np
from .utils import expect


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sum_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(data, axis=axes, keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(data, axis=None, keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sum_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())


# FIXME: noop_with_empty_axes not working properly
# @pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
# def test_sum_empty_axes_input_noop(type_a):

#     shape = [3, 2, 2]
#     axes = np.array([], dtype=np.int64)
#     keepdims = True

#     data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=type_a)
#     expected = np.array(data).astype(type_a)
#     result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims, noop_with_empty_axes=True)
#     expect(expected, result.numpy())

#     data = np.random.uniform(-10, 10, shape).astype(type_a)
#     expected = np.array(data).astype(type_a)
#     result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims, noop_with_empty_axes=True)
#     expect(expected, result.numpy())

@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sum_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sum_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())