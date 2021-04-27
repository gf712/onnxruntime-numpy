import onnxruntime_numpy as onp
from onnxruntime_numpy.types import (
    numeric_types, all_types, float_types)
import pytest
import numpy as np
from .utils import expect


@pytest.mark.parametrize("type_a", [*numeric_types])
def test_constant_value(type_a):
    a = onp.array([0, 1, 2], dtype=type_a)
    expected = onp.array([0, 1, 2], dtype=type_a)
    result = onp.constant(value=a)
    expect(expected.numpy(), result.numpy())

    a = onp.array([[[[0, 1, 2]]]], dtype=type_a)
    expected = onp.array([[[[0, 1, 2]]]], dtype=type_a)
    result = onp.constant(value=a)

    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_constant_value_float(type_a):
    a = 1.
    expected = 1.
    result = onp.constant(value_float=a)
    assert float(result) == expected


@pytest.mark.parametrize("type_a", [np.float32])
def test_constant_value_floats(type_a):
    a = onp.array([0, 1, 2], dtype=type_a)
    expected = onp.array([0, 1, 2], dtype=type_a)
    result = onp.constant(value=a)
    expect(result, expected)

    a = onp.array([[[[0, 1, 2]]]], dtype=type_a)

    with pytest.raises(Exception):
        # only works with 1D arrays
        result = onp.constant(value_floats=a)


def test_constant_value_floats_from_list():
    a = [0, 1, 2.]
    expected = onp.array([0, 1, 2], dtype=np.float32)
    result = onp.constant(value_floats=a)
    expect(result, expected)


@pytest.mark.parametrize("type_a", [np.int32])
def test_constant_value_int(type_a):
    a = 1
    expected = 1
    result = onp.constant(value_int=a)
    assert int(result) == expected


@pytest.mark.parametrize("type_a", [np.int32])
def test_constant_value_ints(type_a):
    a = onp.array([0, 1, 2], dtype=type_a)
    expected = onp.array([0, 1, 2], dtype=type_a)
    result = onp.constant(value_ints=a)
    expect(result, expected)

    a = onp.array([[[[0, 1, 2]]]], dtype=type_a)

    with pytest.raises(Exception):
        # only works with 1D arrays
        result = onp.constant(value_ints=a)


def test_constant_value_ints_from_list():
    a = [0, 1, 2]
    expected = onp.array([0, 1, 2], dtype=np.int32)
    result = onp.constant(value_ints=a)
    expect(result, expected)


@pytest.mark.parametrize("type_a", all_types)
def test_constant_value_of_shape(type_a):
    a = onp.array([1], dtype=type_a)
    shape = (1, 2, 3)
    expected = onp.array([[[1, 1, 1],
                           [1, 1, 1]]], dtype=type_a)
    result = onp.constant_of_shape(shape=shape, value=a)
    expect(expected.numpy(), result.numpy())


def test_constant_value_of_shape_default():
    shape = (1, 2, 3)
    expected = onp.array([[[0, 0, 0],
                           [0, 0, 0]]], dtype=np.float32)
    result = onp.constant_of_shape(shape=shape)
    expect(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_arange_positive_delta(type_a):
    start = type_a(1)
    limit = type_a(5)
    delta = type_a(2)

    expected = np.arange(start, limit, delta, dtype=type_a)
    result = onp.arange(onp.array(start), onp.array(limit), onp.array(delta))
    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_arange_negative_delta(type_a):
    start = type_a(10)
    limit = type_a(6)
    delta = type_a(-3)

    expected = np.arange(start, limit, delta, dtype=type_a)
    result = onp.arange(onp.array(start), onp.array(limit), onp.array(delta))
    expect(expected, result.numpy())
