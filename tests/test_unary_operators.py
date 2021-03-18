import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types, integer_types, is_unsigned_int, all_types, is_bool, numeric_types
import pytest
import numpy as np


@pytest.mark.parametrize("type_a", [*float_types, *integer_types])
def test_abs(type_a):
    if is_unsigned_int(type_a):
        # it is invalid to use unsigned int type with negative values
        a = onp.array([1, 2, 3], dtype=type_a)
    else:
        a = onp.array([-1, -2, -3], dtype=type_a)
    expected = onp.array([1, 2, 3], dtype=type_a)
    result = onp.absolute(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_acos(type_a):
    a = onp.array([1., .5, .1], dtype=type_a)
    expected = onp.array([0., 1.04719755, 1.47062891], dtype=type_a)
    result = onp.acos(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_acosh(type_a):
    a = onp.array([1., 2., 3.], dtype=type_a)
    expected = onp.array([0., 1.3169579, 1.76274717], dtype=type_a)
    result = onp.acosh(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_asin(type_a):
    a = onp.array([1., .2, .3], dtype=type_a)
    expected = onp.array([1.57079633, 0.20135792, 0.30469265], dtype=type_a)
    result = onp.asin(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_asinh(type_a):
    a = onp.array([1., .2, .3], dtype=type_a)
    expected = onp.array([0.88137359, 0.19869011, 0.29567305], dtype=type_a)
    result = onp.asinh(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_atan(type_a):
    a = onp.array([1., .2, .3], dtype=type_a)
    expected = onp.array([0.78539816, 0.19739556, 0.29145679], dtype=type_a)
    result = onp.atan(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_atanh(type_a):
    a = onp.array([0., .2, .3], dtype=type_a)
    expected = onp.array([0., 0.20273255, 0.3095196], dtype=type_a)
    result = onp.atanh(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*all_types])
@pytest.mark.parametrize("type_b", [*all_types])
def test_cast(type_a, type_b):
    a = onp.array([0, 1, 2], dtype=type_a)
    if is_bool(type_b) or is_bool(type_a):
        expected = onp.array([0, 1, 1], dtype=type_b)
    else:
        expected = onp.array([0, 1, 2], dtype=type_b)
    result = onp.cast(a, type_b)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_ceil(type_a):
    a = onp.array([-1.5, 2.49, -3.99], dtype=type_a)
    expected = onp.array([-1., 3., -3], dtype=type_a)
    result = onp.ceil(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*numeric_types])
def test_clip(type_a):
    if type_a in [np.int16, np.int32, np.uint16, np.uint32]:
        return
    a = onp.array([0, 1, 2], dtype=type_a)
    expected = onp.array([0, 1, 1], dtype=type_a)
    result = onp.clip(a, minimum=0, maximum=1)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [*numeric_types])
def test_constant_value(type_a):
    a = onp.array([0, 1, 2], dtype=type_a)
    expected = onp.array([0, 1, 2], dtype=type_a)
    result = onp.constant(value=a)
    assert np.allclose(expected.numpy(), result.numpy())

    a = onp.array([[[[0, 1, 2]]]], dtype=type_a)
    expected = onp.array([[[[0, 1, 2]]]], dtype=type_a)
    result = onp.constant(value=a)

    assert np.allclose(expected.numpy(), result.numpy())


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
    assert np.allclose(result, expected)

    a = onp.array([[[[0, 1, 2]]]], dtype=type_a)

    with pytest.raises(Exception) as e_info:
        # only works with 1D arrays
        result = onp.constant(value_floats=a)


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
    assert np.allclose(result, expected)

    a = onp.array([[[[0, 1, 2]]]], dtype=type_a)

    with pytest.raises(Exception) as e_info:
        # only works with 1D arrays
        result = onp.constant(value_ints=a)


@pytest.mark.parametrize("type_a", all_types)
def test_constant_value_of_shape(type_a):
    a = onp.array([1], dtype=type_a)
    shape = (1, 2, 3)
    expected = onp.array([[[1, 1, 1],
                           [1, 1, 1]]], dtype=type_a)
    result = onp.constant_of_shape(shape=shape, value=a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_cos(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    expected = onp.array([0.54030231, -0.41614684, -0.9899925], dtype=type_a)
    result = onp.cos(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_cosh(type_a):
    a = onp.array([1, 2, 3], dtype=type_a)
    expected = onp.array([1.54308063,  3.76219569, 10.067662], dtype=type_a)
    result = onp.cosh(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_det(type_a):
    a = onp.array([[1., 2.],
                   [3., 4.]], dtype=type_a)
    expected = onp.array([-2], dtype=type_a)
    result = onp.det(a)
    assert np.allclose(expected.numpy(), result.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_det_nd(type_a):
    a = onp.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]], dtype=type_a)
    expected = onp.array([-2., -3., -8.], dtype=type_a)
    result = onp.det(a)
    assert np.allclose(expected.numpy(), result.numpy())
