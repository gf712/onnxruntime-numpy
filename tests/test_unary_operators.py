import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types, integer_types, is_unsigned_int, all_types, is_bool, numeric_types
import pytest
import numpy as np
from .utils import expect


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
    expect(result, expected)

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
    a = onp.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]], dtype=type_a)
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


@pytest.mark.parametrize("type_a", [*float_types, np.uint64, np.int32, np.int64])
@pytest.mark.parametrize("type_b", [*float_types, np.uint64, np.int32, np.int64])
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


@pytest.mark.parametrize("type_a", [*float_types, np.uint64, np.int32, np.int64])
@pytest.mark.parametrize("type_b", [*float_types, np.uint64, np.int32, np.int64])
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


@pytest.mark.parametrize("type_a", [*float_types, np.uint64, np.int32, np.int64])
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


@pytest.mark.parametrize("type_a", float_types)
def test_floor(type_a):
    x = np.random.randn(3, 4, 5).astype(np.float32)
    expected = np.floor(x)

    result = onp.floor(onp.array(x))

    expect(expected, result.numpy())
