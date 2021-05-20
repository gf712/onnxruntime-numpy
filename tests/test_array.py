import onnxruntime_numpy as onp
from onnxruntime_numpy.array import is_lazy
from onnxruntime_numpy.types import all_types
import numpy as np
import pytest
from .utils import expect


def test_array_type_deduction_int():
    a = onp.array([1])
    assert(a.dtype == np.int32)
    nested = onp.array([[[[1], [2]], [[1], [2]]], [[[1], [2]], [[1], [2]]]])
    assert(nested.dtype == np.int32)
    nested = onp.array([[[[1], [2]], [[1], [2]]], [[[True], [2]], [[1], [2]]]])
    assert(nested.dtype == np.int32)


def test_array_type_deduction_float():
    a = onp.array([1.])
    assert(a.dtype == np.float32)
    nested = onp.array([[[[1], [2]], [[1], [2.]]], [[[1], [2]], [[1], [2]]]])
    assert(nested.dtype == np.float32)


def test_array_type_deduction_bool():
    a = onp.array([True])
    assert(a.dtype == np.bool_)
    nested = onp.array([[[[True], [True]], [[False], [False]]],
                        [[[True], [True]], [[True], [False]]]])
    assert(nested.dtype == np.bool_)


def test_array_type_deduction_string():

    with pytest.raises(Exception):
        # not implemented yet
        _ = onp.array(["test"])


def test_array_type_deduction_object():
    class A:
        pass

    with pytest.raises(Exception):
        # not implemented yet
        _ = onp.array([A()])


def test_array_with_same_type():
    a = onp.array([1.])
    b = onp.array(a)
    expect(a.numpy(), b.numpy())

    a = onp.array([1.])
    b = onp.array(a, dtype=a.dtype)
    expect(a.numpy(), b.numpy())


@pytest.mark.parametrize("from_type", all_types)
@pytest.mark.parametrize("to_type", all_types)
def test_new_array_with_cast(from_type, to_type):
    x = onp.array(0, dtype=from_type)
    y = onp.array(x, dtype=to_type)

    assert x.dtype == from_type
    assert y.dtype == to_type

    assert x.item() == y.item()
    assert x.shape == y.shape


@pytest.mark.parametrize("type_a", all_types)
def test_new_array_from_numpy_scalar(type_a):
    expected = type_a(0)
    result = onp.array(expected)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", all_types)
def test_array_item(type_a):
    expected = type_a(0)
    result = onp.array(expected)

    assert result.item() == 0


def test_array_is_not_lazy():
    expected = np.float32(0)
    result = onp.array(expected)

    assert not is_lazy(result)


def test_array_is_lazy():
    expected = np.float32(0)
    result = onp.array(expected) + onp.array(expected)

    assert is_lazy(result)
