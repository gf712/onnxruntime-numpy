import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types, all_types
import numpy as np
import pytest
from .utils import expect

@pytest.mark.parametrize("type_a", [*all_types])
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_gather_0(type_a, type_b):
    data = np.random.randn(5, 4, 3, 2).astype(type_a)
    indices = np.array([0, 1, 3])
    expected = np.take(data, indices, axis=0)

    result = onp.gather(onp.array(data), onp.array(indices, dtype=type_b), axis=0)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*all_types])
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_gather_1(type_a, type_b):
    data = np.random.randn(5, 4, 3, 2).astype(type_a)
    indices = np.array([0, 1, 3])
    expected = np.take(data, indices, axis=1)

    result = onp.gather(onp.array(data), onp.array(indices, dtype=type_b), axis=1)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*all_types])
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_gather_2d_indices(type_a, type_b):
    data = np.random.randn(3, 3).astype(type_a)
    indices = np.array([[0, 2]])
    expected = np.take(data, indices, axis=1)

    result = onp.gather(onp.array(data), onp.array(indices, dtype=type_b), axis=1)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*all_types])
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_gather_negative_indices(type_a, type_b):
    data = np.arange(10).astype(type_a)
    indices = np.array([0, -9, -10])
    expected = np.take(data, indices, axis=0)

    result = onp.gather(onp.array(data), onp.array(indices, dtype=type_b), axis=0)

    expect(expected, result.numpy())
