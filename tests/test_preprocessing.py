import pytest
import onnxruntime_numpy as onp
import numpy as np
from .utils import expect, one_hot_reference

OHE_IMPLEMENTED_COMBINATIONS = [[np.float32, np.float32, np.float32],
                                [np.int64, np.int64, np.float32],
                                [np.int64, np.int64, np.int64]]


@pytest.mark.parametrize("type_comb", OHE_IMPLEMENTED_COMBINATIONS)
def test_one_hot_default(type_comb):
    axisValue = -1
    on_value = 3
    off_value = 1

    type_a, type_b, type_c = type_comb

    indices = np.array([[1, 9],
                        [2, 4]], dtype=type_a)
    depth = np.array([10], dtype=type_b)
    values = np.array([off_value, on_value], dtype=type_c)

    expected = one_hot_reference(
        indices, depth, axis=axisValue, dtype=type_c)
    expected = expected * (on_value - off_value) + off_value

    result = onp.one_hot(
        onp.array(indices),
        onp.array(depth),
        onp.array(values))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_comb", OHE_IMPLEMENTED_COMBINATIONS)
def test_one_hot_positive_axis(type_comb):
    axisValue = 1
    on_value = 3
    off_value = 1

    type_a, type_b, type_c = type_comb

    indices = np.array([[1, 9],
                        [2, 4]], dtype=type_a)
    depth = np.array([10], dtype=type_b)
    values = np.array([off_value, on_value], dtype=type_c)

    expected = one_hot_reference(
        indices, depth, axis=axisValue, dtype=type_c)
    expected = expected * (on_value - off_value) + off_value

    result = onp.one_hot(
        onp.array(indices), onp.array(depth), onp.array(values), axis=axisValue)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_comb", OHE_IMPLEMENTED_COMBINATIONS)
def test_one_hot_negative_axis(type_comb):
    axisValue = -2
    on_value = 3
    off_value = 1

    type_a, type_b, type_c = type_comb

    indices = np.array([[1, 9],
                        [2, 4]], dtype=type_a)
    depth = np.array([10], dtype=type_b)
    values = np.array([off_value, on_value], dtype=type_c)

    expected = one_hot_reference(
        indices, depth, axis=axisValue, dtype=type_c)
    expected = expected * (on_value - off_value) + off_value

    result = onp.one_hot(
        onp.array(indices), onp.array(depth), onp.array(values), axis=axisValue)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_comb", OHE_IMPLEMENTED_COMBINATIONS)
def test_one_hot_negative_indices(type_comb):
    axisValue = 1
    on_value = 3
    off_value = 1

    type_a, type_b, type_c = type_comb

    indices = np.array([0, -7, -8], dtype=type_a)
    depth = np.array([10], dtype=type_b)
    values = np.array([off_value, on_value], dtype=type_c)

    expected = one_hot_reference(
        indices, depth, axis=axisValue, dtype=type_c)
    expected = expected * (on_value - off_value) + off_value

    result = onp.one_hot(
        onp.array(indices), onp.array(depth), onp.array(values), axis=axisValue)

    expect(expected, result.numpy())
