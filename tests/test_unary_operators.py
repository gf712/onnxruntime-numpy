import onnxruntime_numpy as onp
from onnxruntime_numpy.types import (
    float_types, integer_types, is_unsigned_int, all_types, is_bool,
    numeric_types, bool_types)
import pytest
import numpy as np
from .utils import expect
import itertools


def argmax_use_numpy(data, axis=0, keepdims=1):
    result = np.argmax(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


def argmax_use_numpy_select_last_index(data, axis=0, keepdims=True):
    data = np.flip(data, axis)
    result = np.argmax(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


def argmin_use_numpy(data, axis=0, keepdims=1):
    result = np.argmin(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


def argmin_use_numpy_select_last_index(data, axis=0, keepdims=True):
    data = np.flip(data, axis)
    result = np.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)


@pytest.mark.parametrize("type_a", [*float_types, *integer_types])
def test_abs(type_a):
    if is_unsigned_int(type_a):
        # it is invalid to use unsigned int type with negative values
        a = onp.array([1, 2, 3], dtype=type_a)
    else:
        a = onp.array([-1, -2, -3], dtype=type_a)
    expected = onp.array([1, 2, 3], dtype=type_a)
    result = onp.absolute(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_acos(type_a):
    a = onp.array([1., .5, .1], dtype=type_a)
    expected = onp.array([0., 1.04719755, 1.47062891], dtype=type_a)
    result = onp.acos(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_acosh(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    expected = onp.array([0., 1.3169579, 1.76274717], dtype=type_a)
    result = onp.acosh(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_default_axes_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    keepdims = True
    expected = argmax_use_numpy(x, keepdims=keepdims)
    result = onp.argmax(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_default_axes_keepdims_select_last_index(type_a):
    x = np.array([[2, 2], [3, 10]], dtype=type_a)
    keepdims = True
    expected = argmax_use_numpy_select_last_index(x, keepdims=keepdims)
    result = onp.argmax(onp.array(x), select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = True
    expected = argmax_use_numpy(x, axis=axis, keepdims=keepdims)
    result = onp.argmax(onp.array(x), axis=axis, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_keepdims_select_last_index(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = True
    expected = argmax_use_numpy_select_last_index(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmax(
        onp.array(x),
        axis=axis, keepdims=keepdims, select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_negative_axis_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = -1
    keepdims = True
    expected = argmax_use_numpy(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmax(
        onp.array(x),
        axis=axis, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_negative_axis_keepdims_select_last_index(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = -1
    keepdims = True
    expected = argmax_use_numpy_select_last_index(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmax(
        onp.array(x),
        axis=axis, keepdims=keepdims, select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_no_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = False
    expected = argmax_use_numpy(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmax(
        onp.array(x),
        axis=axis, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmax_no_keepdims_select_last_index(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = False
    expected = argmax_use_numpy_select_last_index(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmax(
        onp.array(x),
        axis=axis, keepdims=keepdims, select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_default_axes_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    keepdims = True
    expected = argmin_use_numpy(x, keepdims=keepdims)
    result = onp.argmin(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_default_axes_keepdims_select_last_index(type_a):
    x = np.array([[2, 2], [3, 10]], dtype=type_a)
    keepdims = True
    expected = argmin_use_numpy_select_last_index(x, keepdims=keepdims)
    result = onp.argmin(onp.array(x), select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = True
    expected = argmin_use_numpy(x, axis=axis, keepdims=keepdims)
    result = onp.argmin(onp.array(x), axis=axis, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_keepdims_select_last_index(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = True
    expected = argmin_use_numpy_select_last_index(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmin(
        onp.array(x),
        axis=axis, keepdims=keepdims, select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_negative_axis_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = -1
    keepdims = True
    expected = argmin_use_numpy(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmin(
        onp.array(x),
        axis=axis, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_negative_axis_keepdims_select_last_index(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = -1
    keepdims = True
    expected = argmin_use_numpy_select_last_index(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmin(
        onp.array(x),
        axis=axis, keepdims=keepdims, select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_no_keepdims(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = False
    expected = argmin_use_numpy(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmin(
        onp.array(x),
        axis=axis, keepdims=keepdims)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32])
def test_argmin_no_keepdims_select_last_index(type_a):
    x = np.array([[2, 1], [3, 10]], dtype=type_a)
    axis = 1
    keepdims = False
    expected = argmin_use_numpy_select_last_index(
        x, axis=axis, keepdims=keepdims)
    result = onp.argmin(
        onp.array(x),
        axis=axis, keepdims=keepdims, select_last_index=True)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_asin(type_a):
    a = onp.array([1., .2, .3], dtype=type_a)
    expected = onp.array([1.57079633, 0.20135792, 0.30469265], dtype=type_a)
    result = onp.asin(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_asinh(type_a):
    a = onp.array([1., .2, .3], dtype=type_a)
    expected = onp.array([0.88137359, 0.19869011, 0.29567305], dtype=type_a)
    result = onp.asinh(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_atan(type_a):
    a = onp.array([1., .2, .3], dtype=type_a)
    expected = onp.array([0.78539816, 0.19739556, 0.29145679], dtype=type_a)
    result = onp.atan(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_atanh(type_a):
    a = onp.array([0., .2, .3], dtype=type_a)
    expected = onp.array([0., 0.20273255, 0.3095196], dtype=type_a)
    result = onp.atanh(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*all_types])
@pytest.mark.parametrize("type_b", [*all_types])
def test_cast(type_a, type_b):
    a = onp.array([0, 1, 2], dtype=type_a)
    if is_bool(type_b) or is_bool(type_a):
        expected = onp.array([0, 1, 1], dtype=type_b)
    else:
        expected = onp.array([0, 1, 2], dtype=type_b)
    result = onp.cast(a, type_b)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_ceil(type_a):
    a = onp.array([-1.5, 2.49, -3.99], dtype=type_a)
    expected = onp.array([-1., 3., -3], dtype=type_a)
    result = onp.ceil(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*numeric_types])
def test_clip(type_a):
    if type_a in [np.int16, np.int32, np.uint16, np.uint32]:
        return
    a = onp.array([0, 1, 2], dtype=type_a)
    expected = onp.array([0, 1, 1], dtype=type_a)
    result = onp.clip(a, minimum=0, maximum=1)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_cos(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    expected = onp.array([0.54030231, -0.41614684, -0.9899925], dtype=type_a)
    result = onp.cos(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_cosh(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    expected = onp.array([1.54308063,  3.76219569, 10.067662], dtype=type_a)
    result = onp.cosh(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_det(type_a):
    a = onp.array([[1., 2.],
                   [3., 4.]], dtype=type_a)
    expected = onp.array(-2, dtype=type_a)
    result = onp.det(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_det_nd(type_a):
    a = onp.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]],
                   [[1, 3], [3, 1]]], dtype=type_a)
    expected = onp.array([-2., -3., -8.], dtype=type_a)
    result = onp.det(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_erf(type_a):
    a = onp.array([[1, 2, 3], [-1, -2, 0]], dtype=type_a)
    expected = onp.array([[0.84270079,  0.99532227,  0.99997791],
                          [-0.84270079, -0.99532227,  0.]],
                         dtype=type_a)
    result = onp.erf(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_exp(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([0.36787945, 1., 2.71828175],
                         dtype=type_a)
    result = onp.exp(a)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a",
                         [*float_types, np.uint64, np.int32, np.int64])
@pytest.mark.parametrize("type_b",
                         [*float_types, np.uint64, np.int32, np.int64])
def test_eyelike_populate_off_main_diagonal(type_a, type_b):
    shape = (4, 5)
    off_diagonal_offset = 1
    if type_a in integer_types:
        x = np.random.randint(0, 100, size=shape, dtype=type_a)
    elif type_a in float_types:
        x = np.random.randn(*shape).astype(type_a)
    else:
        raise ValueError(f"Invalid type {type_a}")

    expected = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=type_b)

    result = onp.eye_like(onp.array(x, dtype=type_a),
                          dtype=type_b, k=off_diagonal_offset)

    assert result.dtype == type_b
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a",
                         [*float_types, np.uint64, np.int32, np.int64])
@pytest.mark.parametrize("type_b",
                         [*float_types, np.uint64, np.int32, np.int64])
def test_eyelike_with_dtype(type_a, type_b):
    shape = (3, 4)
    if type_a in integer_types:
        x = np.random.randint(0, 100, size=shape, dtype=type_a)
    elif type_a in float_types:
        x = np.random.randn(*shape).astype(type_a)
    else:
        raise ValueError(f"Invalid type {type_a}")

    expected = np.eye(shape[0], shape[1], dtype=type_b)

    result = onp.eye_like(onp.array(x, dtype=type_a), dtype=type_b)

    assert result.dtype == type_b
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a",
                         [*float_types, np.uint64, np.int32, np.int64])
def test_eyelike_without_dtype(type_a):
    shape = (4, 4)
    if type_a in integer_types:
        x = np.random.randint(0, 100, size=shape, dtype=type_a)
    elif type_a in float_types:
        x = np.random.randn(*shape).astype(type_a)
    else:
        raise ValueError(f"Invalid type {type_a}")

    expected = np.eye(shape[0], shape[1], dtype=type_a)

    result = onp.eye_like(onp.array(x, dtype=type_a))

    assert result.dtype == type_a
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a",
                         [*float_types, np.uint64, np.int32, np.int64])
def test_eyelike_with_3d_tensor(type_a):
    shape = (4, 4, 1)
    if type_a in integer_types:
        x = np.random.randint(0, 100, size=shape, dtype=type_a)
    elif type_a in float_types:
        x = np.random.randn(*shape).astype(type_a)
    else:
        raise ValueError(f"Invalid type {type_a}")

    with pytest.raises(ValueError):
        _ = onp.eye_like(onp.array(x, dtype=type_a))


def test_eyelike_unsupported_type():
    shape = (4, 4)
    x = np.random.randint(0, 100, size=shape, dtype=np.int32)

    with pytest.raises(TypeError):
        _ = onp.eye_like(onp.array(x), dtype=np.str_)


@pytest.mark.parametrize("type_a", all_types)
def test_flatten(type_a):

    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(type_a)

    for i in range(len(shape)):
        new_shape = (1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
        expected = np.reshape(a, new_shape)

        result = onp.flatten(onp.array(a, dtype=type_a), axis=i)

        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_flatten_negativate_axis(type_a):

    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(type_a)

    for i in range(-len(shape), 0):
        new_shape = (np.prod(shape[0:i]).astype(int), -1)
        expected = np.reshape(a, new_shape)

        result = onp.flatten(onp.array(a, dtype=type_a), axis=i)

        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_flatten_with_default_axis(type_a):

    shape = (5, 4, 3, 2)
    a = np.random.random_sample(shape).astype(type_a)
    new_shape = (5, 24)
    expected = np.reshape(a, new_shape)

    result = onp.flatten(onp.array(a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_floor(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.floor(x)

    result = onp.floor(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_identity(type_a):
    x = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=type_a)

    expected = x

    result = onp.identity(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_isinf_infinity(type_a):
    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                 dtype=type_a)
    expected = np.isinf(x)
    result = onp.isinf(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_isinf_negative_infinity_only(type_a):
    x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf],
                 dtype=type_a)
    expected = np.isneginf(x)
    result = onp.isneginf(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_isinf_positive_infinity_only(type_a):
    x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf],
                 dtype=type_a)
    expected = np.isposinf(x)
    result = onp.isposinf(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_isnan(type_a):
    x = np.array([3.0, np.nan, 4.0, np.nan], dtype=type_a)
    expected = np.isnan(x)
    result = onp.isnan(onp.array(x))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_log(type_a):
    x = np.array([1, 10], dtype=type_a)
    expected = np.log(x)
    result = onp.log(onp.array(x))
    expect(expected, result.numpy())

    x = np.exp(np.random.randn(3, 4, 5).astype(type_a))
    expected = np.log(x)
    result = onp.log(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_mean_variance_normalization(type_a):
    input_data = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
                            [[0.02916367], [0.12964272], [0.5060197]],
                            [[0.79538304], [0.9411346], [0.9546573]]],
                           [[[0.17730942], [0.46192095], [0.26480448]],
                            [[0.6746842], [0.01665257], [0.62473077]],
                            [[0.9240844], [0.9722341], [0.11965699]]],
                           [[[0.41356155], [0.9129373], [0.59330076]],
                            [[0.81929934], [0.7862604], [0.11799799]],
                            [[0.69248444], [0.54119414], [0.07513223]]]], dtype=type_a)

    data_mean = np.mean(input_data, axis=(0, 2, 3), keepdims=1)
    data_mean_squared = np.power(data_mean, 2)
    data_squared = np.power(input_data, 2)
    data_squared_mean = np.mean(data_squared, axis=(0, 2, 3), keepdims=1)
    std = np.sqrt(data_squared_mean - data_mean_squared)
    expected = ((input_data - data_mean) / (std + 1e-9)).astype(type_a)

    result = onp.mean_variance_normalization(onp.array(input_data))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int8, np.int32, np.int64])
def test_negative(type_a):
    x = np.array([-4, 2]).astype(type_a)
    expected = np.negative(x)
    result = onp.negative(onp.array(x))
    expect(expected, result.numpy())

    result = -onp.array(x)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.uint8, np.int32, np.int64])
def test_nonzero(type_a):
    x = np.array([[1, 0], [1, 1]], dtype=type_a)
    expected = np.array(np.nonzero(x), dtype=np.int64)
    result = onp.nonzero(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*bool_types])
def test_not(type_a):
    x = (np.random.randn(3, 4) > 0).astype(type_a)
    expected = np.logical_not(x)
    result = onp.not_(onp.array(x))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5) > 0).astype(type_a)
    expected = np.logical_not(x)
    result = onp.not_(onp.array(x))
    expect(expected, result.numpy())

    x = (np.random.randn(3, 4, 5, 6) > 0).astype(type_a)
    expected = np.logical_not(x)
    result = onp.not_(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_reciprocal(type_a):
    x = np.array([-4, 2]).astype(type_a)

    expected = np.reciprocal(x)
    result = onp.reciprocal(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.rand(3, 4, 5).astype(type_a) + 0.5
    expected = np.reciprocal(x)
    result = onp.reciprocal(onp.array(x))
    expect(expected, result.numpy())


def reshape_reference_implementation(data, shape, allowzero=0):
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless
    # 'allowzero' is set
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_reordered_all_dims(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [4, 2, 3]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_reordered_last_dims(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [2, 4, 3]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_reduced_dims(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [2, 12]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_extended_dims(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [2, 3, 2, 2]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_one_dim(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [24]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_negative_dim(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [2, -1, 2]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_reshape_negative_extended_dims(type_a):
    original_shape = [2, 3, 4]
    expected_shape = [-1, 2, 3, 4]
    x = np.random.uniform(size=original_shape).astype(type_a)

    expected = reshape_reference_implementation(x, expected_shape)
    result = onp.array(x).reshape(expected_shape)
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_round(type_a):
    x = np.array([0.1, 0.5, 0.9, 1.2, 1.5,
                  1.8, 2.3, 2.5, 2.7, -1.1,
                  -1.5, -1.9, -2.2, -2.5, -2.8]).astype(type_a)

    expected = np.array([0., 0., 1., 1., 2.,
                         2., 2., 2., 3., -1.,
                         -2., -2., -2., -2., -3.]).astype(type_a)
    result = onp.round(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_shape(type_a):
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]).astype(type_a)
    expected = np.array([
        2, 3,
    ]).astype(np.int64)

    result = onp.shape(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.array(x.shape).astype(np.int64)
    result = onp.shape(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", numeric_types)
def test_sign(type_a):
    x = np.array(range(-5, 6)).astype(type_a)
    expected = np.sign(x)
    result = onp.sign(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_sin(type_a):
    x = np.array([-1, 0, 1]).astype(type_a)
    expected = np.sin(x)
    result = onp.sin(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.sin(x)
    result = onp.sin(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_sinh(type_a):
    x = np.array([-1, 0, 1]).astype(type_a)
    expected = np.sinh(x)
    result = onp.sinh(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.sinh(x)
    result = onp.sinh(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_size(type_a):
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]).astype(type_a)
    expected = np.array(6).astype(np.int64)

    result = onp.size(onp.array(x))
    expect(expected, result.numpy())

    x = np.random.randn(3, 4, 5).astype(type_a)
    expected = np.array(x.size).astype(np.int64)
    result = onp.size(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", float_types)
def test_sqrt(type_a):
    x = np.array([1, 4, 9]).astype(type_a)
    expected = np.sqrt(x)
    result = onp.sqrt(onp.array(x))
    expect(expected, result.numpy())

    x = np.abs(np.random.randn(3, 4, 5).astype(type_a))
    expected = np.sqrt(x)
    result = onp.sqrt(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_transpose_all_permutations(type_a):
    shape = (2, 3, 4)
    x = np.random.uniform(0, 1, size=shape).astype(type_a)
    permutations = list(itertools.permutations(np.arange(len(shape))))

    for i in range(len(permutations)):
        expected = np.transpose(x, permutations[i])
        result = onp.transpose(onp.array(x), permutations[i])
        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_transpose_default(type_a):
    shape = (2, 3, 4)
    x = np.random.uniform(0, 1, size=shape).astype(type_a)

    expected = x.T
    result = onp.array(x).T
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_tan(type_a):
    x = np.array([-1, 0, 1]).astype(type_a)

    expected = np.tan(x)
    result = onp.tan(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_tanh(type_a):
    x = np.array([-1, 0, 1]).astype(type_a)

    expected = np.tanh(x)
    result = onp.tanh(onp.array(x))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_squeeze(type_a):
    x = np.random.randn(1, 3, 4, 5).astype(type_a)
    axes = np.array([0], dtype=np.int64)
    expected = np.squeeze(x, axis=0)
    result = onp.squeeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_squeeze_negative_axes(type_a):
    x = np.random.randn(1, 3, 1, 5).astype(type_a)
    axes = np.array([-2], dtype=np.int64)
    expected = np.squeeze(x, axis=-2)
    result = onp.squeeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_squeeze_lazy(type_a):
    x = np.random.randn(1, 3, 1, 5).astype(type_a)
    axes = np.array([-1], dtype=np.int64)
    axes += axes  # -2
    expected = np.squeeze(x, axis=-2)
    result = onp.squeeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


# TODO: update this when onnxruntime release ONNX opset 14 support
# @pytest.mark.parametrize("type_a", all_types)
# def test_trilu_lower(type_a):
#     x = np.random.randint(10, size=(4, 5)).astype(type_a)
#     expected = np.tril(x, 0)
#     result = onp.tril(onp.array(x))
#     expect(expected, result.numpy())

@pytest.mark.parametrize("type_a", [*float_types, np.int64])
def test_topk(type_a):
    axis = 1
    largest = True
    k = 3

    X = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=type_a)
    K = np.array([k], dtype=np.int64)

    values_expected = np.array([[3, 2, 1],
                                [7, 6, 5],
                                [11, 10, 9]], dtype=type_a)
    indices_expected = np.array([[3, 2, 1],
                                 [3, 2, 1],
                                 [3, 2, 1]], dtype=np.int64)

    values, indices = onp.topk(
        onp.array(X),
        onp.array(K),
        axis=axis, largest=largest)

    expect(values_expected, values.numpy())
    expect(indices_expected, indices.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int64])
def test_topk_negative_axis(type_a):
    axis = -1
    largest = True
    k = 3

    X = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ], dtype=type_a)
    K = np.array([k], dtype=np.int64)

    values_expected = np.array([[3, 2, 1],
                                [7, 6, 5],
                                [11, 10, 9]], dtype=type_a)
    indices_expected = np.array([[3, 2, 1],
                                 [3, 2, 1],
                                 [3, 2, 1]], dtype=np.int64)

    values, indices = onp.topk(
        onp.array(X),
        onp.array(K),
        axis=axis, largest=largest)

    expect(values_expected, values.numpy())
    expect(indices_expected, indices.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int64])
def test_topk_smallest(type_a):
    axis = 1
    largest = False
    sorted = True
    k = 3

    X = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [11, 10, 9, 8],
    ], dtype=type_a)
    K = np.array([k], dtype=np.int64)

    values_expected = np.array([[0, 1, 2],
                                [4, 5, 6],
                                [8, 9, 10]], dtype=type_a)
    indices_expected = np.array([[0, 1, 2],
                                 [0, 1, 2],
                                 [3, 2, 1]], dtype=np.int64)

    values, indices = onp.topk(
        onp.array(X), onp.array(K),
        axis=axis, largest=largest, sorted=sorted)

    expect(values_expected, values.numpy())
    expect(indices_expected, indices.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int8, np.int64])
def test_unique_not_sorted_without_axis(type_a):
    x = np.array([2, 1, 1, 3, 4, 3], dtype=type_a)
    y, indices, inverse_indices, counts = np.unique(
        x, True, True, True)

    # prepare index mapping from sorted to unsorted
    argsorted_indices = np.argsort(indices)
    inverse_indices_map = {i: si for i, si in zip(
        argsorted_indices, np.arange(len(argsorted_indices)))}

    indices = indices[argsorted_indices]
    y_expected = np.take(x, indices, axis=0)
    inverse_indices = np.asarray([inverse_indices_map[i]
                                  for i in inverse_indices],
                                 dtype=np.int64)
    counts = counts[argsorted_indices]
    indices_expected = indices.astype(np.int64)
    inverse_indices_expected = inverse_indices.astype(np.int64)
    counts_expected = counts.astype(np.int64)

    y, indices, inverse_indices, counts = onp.unique(onp.array(
        x), return_index=True, return_inverse=True, return_counts=True, sorted=False)

    expect(y_expected, y.numpy())
    expect(indices_expected, indices.numpy())
    expect(inverse_indices_expected, inverse_indices.numpy())
    expect(counts_expected, counts.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int8, np.int64])
def test_unique_sorted_with_axis(type_a):
    x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]], dtype=type_a)
    y_expected, indices, inverse_indices, counts = np.unique(
        x, True, True, True, axis=0)

    indices_expected = indices.astype(np.int64)
    inverse_indices_expected = inverse_indices.astype(np.int64)
    counts_expected = counts.astype(np.int64)

    y, indices, inverse_indices, counts = onp.unique(
        onp.array(x),
        return_index=True, return_inverse=True, return_counts=True, sorted=True,
        axis=0)

    expect(y_expected, y.numpy())
    expect(indices_expected, indices.numpy())
    expect(inverse_indices_expected, inverse_indices.numpy())
    expect(counts_expected, counts.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int8, np.int64])
def test_unique_sorted_with_axis_3d(type_a):
    x = np.array([[[1, 1], [0, 1], [2, 1], [0, 1]],
                  [[1, 1], [0, 1], [2, 1], [0, 1]]], dtype=type_a)
    y_expected, indices, inverse_indices, counts = np.unique(
        x, True, True, True, axis=1)

    indices_expected = indices.astype(np.int64)
    inverse_indices_expected = inverse_indices.astype(np.int64)
    counts_expected = counts.astype(np.int64)

    y, indices, inverse_indices, counts = onp.unique(
        onp.array(x),
        return_index=True, return_inverse=True, return_counts=True, sorted=True,
        axis=1)

    expect(y_expected, y.numpy())
    expect(indices_expected, indices.numpy())
    expect(inverse_indices_expected, inverse_indices.numpy())
    expect(counts_expected, counts.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int8, np.int64])
def test_unique_negative_axis(type_a):
    x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 3]], dtype=type_a)
    y_expected, indices, inverse_indices, counts = np.unique(
        x, True, True, True, axis=-1)

    indices_expected = indices.astype(np.int64)
    inverse_indices_expected = inverse_indices.astype(np.int64)
    counts_expected = counts.astype(np.int64)

    y, indices, inverse_indices, counts = onp.unique(
        onp.array(x),
        return_index=True, return_inverse=True, return_counts=True, sorted=True,
        axis=-1)

    expect(y_expected, y.numpy())
    expect(indices_expected, indices.numpy())
    expect(inverse_indices_expected, inverse_indices.numpy())
    expect(counts_expected, counts.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.int8, np.int64])
def test_unique_without_axis(type_a):
    x = np.array([2, 1, 1, 3, 4, 3], dtype=type_a)
    y_expected, indices, inverse_indices, counts = np.unique(
        x, True, True, True)

    indices_expected = indices.astype(np.int64)
    inverse_indices_expected = inverse_indices.astype(np.int64)
    counts_expected = counts.astype(np.int64)

    y, indices, inverse_indices, counts = onp.unique(
        onp.array(x),
        return_index=True, return_inverse=True, return_counts=True, sorted=True)

    expect(y_expected, y.numpy())
    expect(indices_expected, indices.numpy())
    expect(inverse_indices_expected, inverse_indices.numpy())
    expect(counts_expected, counts.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze(type_a):
    x = np.random.randn(1, 3, 4, 5).astype(type_a)
    axes = np.array([0], dtype=np.int64)
    expected = np.expand_dims(x, axis=0)
    result = onp.unsqueeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze_negative_axes(type_a):
    x = np.random.randn(1, 3, 1, 5).astype(type_a)
    axes = np.array([-2], dtype=np.int64)
    expected = np.expand_dims(x, axis=-2)
    result = onp.unsqueeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze_lazy(type_a):
    x = np.random.randn(1, 3, 1, 5).astype(type_a)
    axes = np.array([-1], dtype=np.int64)
    axes += axes  # -2
    expected = np.expand_dims(x, axis=-2)
    result = onp.unsqueeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze_one_axis(type_a):
    x = np.random.randn(3, 4, 5).astype(np.float32)

    for i in range(x.ndim):
        axes = np.array([i]).astype(np.int64)
        expected = np.expand_dims(x, axis=i)
        result = onp.unsqueeze(onp.array(x), onp.array(axes))
        expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze_two_axis(type_a):
    x = np.random.randn(1, 3, 1, 5).astype(type_a)
    axes = np.array([1, 4], dtype=np.int64)
    expected = np.expand_dims(x, axis=1)
    expected = np.expand_dims(expected, axis=4)
    result = onp.unsqueeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze_three_axis(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    axes = np.array([2, 4, 5]).astype(np.int64)
    expected = np.expand_dims(x, axis=2)
    expected = np.expand_dims(expected, axis=4)
    expected = np.expand_dims(expected, axis=5)
    result = onp.unsqueeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_unsqueeze_unsorted(type_a):
    x = np.random.randn(3, 4, 5).astype(type_a)
    axes = np.array([5, 4, 2]).astype(np.int64)
    expected = np.expand_dims(x, axis=2)
    expected = np.expand_dims(expected, axis=4)
    expected = np.expand_dims(expected, axis=5)
    result = onp.unsqueeze(onp.array(x), onp.array(axes))
    expect(expected, result.numpy())
