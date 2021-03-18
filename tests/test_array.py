import onnxruntime_numpy as onp
import numpy as np


def test_array_type_deduction_int():
    a = onp.array([1])
    assert(a.dtype == np.int32)
    nested = onp.array([[[[1], [2]], [[1], [2]]], [[[1], [2]], [[1], [2]]]])
    assert(nested.dtype == np.int32)


def test_array_type_deduction_float():
    a = onp.array([1.])
    assert(a.dtype == np.float32)
    nested = onp.array([[[[1], [2]], [[1], [2.]]], [[[1], [2]], [[1], [2]]]])
    assert(nested.dtype == np.float32)


def test_array_type_deduction_bool():
    a = onp.array([True])
    assert(a.dtype == np.bool)
    nested = onp.array([[[[True], [True]], [[False], [False]]],
                        [[[True], [True]], [[True], [False]]]])
    assert(nested.dtype == np.bool)
