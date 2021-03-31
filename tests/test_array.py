import onnxruntime_numpy as onp
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


def test_new_array_with_cast():
    a = onp.array([1.])
    # TODO
    with pytest.raises(NotImplementedError):
        b = onp.array(a, dtype=np.int32)

        assert b.dtype == np.int32
        assert a.shape == b.shape
        assert np.allclose(a, b)
