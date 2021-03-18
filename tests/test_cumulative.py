import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types
import numpy as np
import pytest


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_1d(type_a):

    x = onp.array([1., 2., 3., 4., 5.], dtype=type_a)
    axis = 0
    expected = onp.array([1., 3., 6., 10., 15.], dtype=type_a)

    result = onp.cumsum(x, axis)

    assert np.allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_1d_exclusive(type_a):

    x = onp.array([1., 2., 3., 4., 5.], dtype=type_a)
    axis = 0
    expected = onp.array([0., 1., 3., 6., 10.], dtype=type_a)

    result = onp.cumsum(x, axis, exclusive=True)

    assert np.allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_1d_reverse(type_a):

    x = onp.array([1., 2., 3., 4., 5.], dtype=type_a)
    axis = 0
    expected = onp.array([15., 14., 12., 9., 5.], dtype=type_a)

    result = onp.cumsum(x, axis, reverse=True)

    assert np.allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_1d_reverse_exclusive(type_a):

    x = onp.array([1., 2., 3., 4., 5.], dtype=type_a)
    axis = 0
    expected = onp.array([14., 12., 9., 5., 0.], dtype=type_a)

    result = onp.cumsum(x, axis, reverse=True, exclusive=True)

    assert np.allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_2d_axis_0(type_a):

    x = onp.array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=type_a)
    axis = 0
    expected = onp.array([[1., 2., 3.],
                          [5., 7., 9.]], dtype=type_a)

    result = onp.cumsum(x, axis)

    assert np.allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_2d_axis_1(type_a):

    x = onp.array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=type_a)
    axis = 1
    expected = onp.array([[1., 3., 6.],
                          [4., 9., 15.]], dtype=type_a)

    result = onp.cumsum(x, axis)

    assert np.allclose(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_cumsum_2d_negative_axis(type_a):

    x = onp.array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=type_a)
    axis = -1
    expected = onp.array([[1., 3., 6.],
                          [4., 9., 15.]], dtype=type_a)

    result = onp.cumsum(x, axis)

    assert np.allclose(result.numpy(), expected.numpy())
