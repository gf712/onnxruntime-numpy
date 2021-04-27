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

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
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

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(
        data, axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(
        data, axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())


# FIXME: noop_with_empty_axes not working properly
# @pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
# def test_sum_empty_axes_input_noop(type_a):

#     shape = [3, 2, 2]
#     axes = np.array([], dtype=np.int64)
#     keepdims = True

#     data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
#                      [[9, 10], [11, 12]]], dtype=type_a)
#     expected = np.array(data).astype(type_a)
#     result = onp.sum(onp.array(data), axes=onp.array(
#         axes), keepdims=keepdims, noop_with_empty_axes=True)
#     expect(expected, result.numpy())

#     data = np.random.uniform(-10, 10, shape).astype(type_a)
#     expected = np.array(data).astype(type_a)
#     result = onp.sum(onp.array(data), axes=onp.array(
#         axes), keepdims=keepdims, noop_with_empty_axes=True)
#     expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sum_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(
        data, axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(
        data, axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_sum_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(
        data, axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(
        data, axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.sum(onp.array(data), axes=onp.array(axes), keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l1_norm_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(a=np.abs(data), axis=axes,
                      keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(a=np.abs(data), axis=None,
                      keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l1_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(
        a=np.abs(data), axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(
        a=np.abs(data), axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l1_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(
        np.abs(data), axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(
        np.abs(data), axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l1_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(
        np.abs(data), axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(
        np.abs(data), axis=tuple(axes.tolist()),
        keepdims=keepdims).astype(type_a)
    result = onp.l1_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l2_norm_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sqrt(np.sum(a=np.square(data), axis=axes,
                       keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sqrt(np.sum(a=np.square(data), axis=None,
                       keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l2_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sqrt(np.sum(
        a=np.square(data), axis=tuple(axes.tolist()),
        keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sqrt(np.sum(
        a=np.square(data), axis=tuple(axes.tolist()),
        keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l2_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sqrt(np.sum(
        np.square(data), axis=tuple(axes.tolist()),
        keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sqrt(np.sum(
        np.square(data), axis=tuple(axes.tolist()),
        keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_l2_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sqrt(np.sum(
        np.square(data), axis=tuple(axes.tolist()),
        keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sqrt(np.sum(
        np.square(data), axis=tuple(axes.tolist()),
        keepdims=keepdims)).astype(type_a)
    result = onp.l2_norm(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(data, axis=axes, keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(1, 10, shape).astype(type_a)
    expected = np.log(np.sum(data, axis=None,
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(data, axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(1, 10, shape).astype(type_a)
    expected = np.log(np.sum(data, axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(data, axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(1, 10, shape).astype(type_a)
    expected = np.log(np.sum(data, axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(data, axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(1, 10, shape).astype(type_a)
    expected = np.log(np.sum(data, axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_exp_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(np.exp(data), axis=axes,
                      keepdims=keepdims)).astype(type_a)
    result = onp.log_sum_exp(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    # FIXME: this causes issues with np.int32
    if type_a == np.float32:
        data = np.random.uniform(-10, 10, shape).astype(type_a)
        expected = np.log(np.sum(np.exp(data), axis=None,
                                 keepdims=keepdims)).astype(type_a)
        result = onp.log_sum_exp(onp.array(data), axes=axes, keepdims=keepdims)
        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_exp_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(np.exp(data), axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum_exp(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    # FIXME: this causes issues with np.int32
    if type_a == np.float32:
        data = np.random.uniform(-10, 10, shape).astype(type_a)
        expected = np.log(np.sum(np.exp(data), axis=tuple(axes.tolist()),
                                 keepdims=keepdims)).astype(type_a)
        result = onp.log_sum_exp(
            onp.array(data),
            axes=onp.array(axes),
            keepdims=keepdims)
        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_exp_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(np.exp(data), axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum_exp(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    # FIXME: this causes issues with np.int32
    if type_a == np.float32:
        data = np.random.uniform(-10, 10, shape).astype(type_a)
        expected = np.log(np.sum(np.exp(data), axis=tuple(axes.tolist()),
                                 keepdims=keepdims)).astype(type_a)
        result = onp.log_sum_exp(
            onp.array(data),
            axes=onp.array(axes),
            keepdims=keepdims)
        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_log_sum_exp_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.log(np.sum(np.exp(data), axis=tuple(axes.tolist()),
                             keepdims=keepdims)).astype(type_a)
    result = onp.log_sum_exp(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    # FIXME: this causes issues with np.int32
    if type_a == np.float32:
        data = np.random.uniform(-10, 10, shape).astype(type_a)
        expected = np.log(np.sum(np.exp(data), axis=tuple(axes.tolist()),
                                 keepdims=keepdims)).astype(type_a)
        result = onp.log_sum_exp(
            onp.array(data),
            axes=onp.array(axes),
            keepdims=keepdims)
        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_max_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.maximum.reduce(data, axis=axes,
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.maximum.reduce(data, axis=None,
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_max_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.maximum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.maximum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_max_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.maximum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.maximum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_max_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.maximum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.maximum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.max(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_min_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.minimum.reduce(data, axis=axes,
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.minimum.reduce(data, axis=None,
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_min_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.minimum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.minimum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_min_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.minimum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.minimum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_min_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.minimum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.minimum.reduce(data, axis=tuple(axes.tolist()),
                                 keepdims=keepdims).astype(type_a)
    result = onp.min(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_prod_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.prod(data, axis=axes,
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.prod(data, axis=None,
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_prod_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.prod(data, axis=tuple(axes.tolist()),
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.prod(data, axis=tuple(axes.tolist()),
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_prod_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.prod(data, axis=tuple(axes.tolist()),
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.prod(data, axis=tuple(axes.tolist()),
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_prod_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.prod(data, axis=tuple(axes.tolist()),
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.prod(data, axis=tuple(axes.tolist()),
                       keepdims=keepdims).astype(type_a)
    result = onp.prod(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_sum_square_default_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = None
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(np.square(data), axis=axes,
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(np.square(data), axis=None,
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(onp.array(data), axes=axes, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_sum_square_do_not_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = False

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(np.square(data), axis=tuple(axes.tolist()),
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(np.square(data), axis=tuple(axes.tolist()),
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_sum_square_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(np.square(data), axis=tuple(axes.tolist()),
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(np.square(data), axis=tuple(axes.tolist()),
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int32])
def test_sum_square_negative_axes_keepdims(type_a):

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = True

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]],
                     [[9, 10], [11, 12]]], dtype=type_a)
    expected = np.sum(np.square(data), axis=tuple(axes.tolist()),
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())

    data = np.random.uniform(-10, 10, shape).astype(type_a)
    expected = np.sum(np.square(data), axis=tuple(axes.tolist()),
                      keepdims=keepdims).astype(type_a)
    result = onp.sum_square(
        onp.array(data),
        axes=onp.array(axes),
        keepdims=keepdims)
    expect(expected, result.numpy())
