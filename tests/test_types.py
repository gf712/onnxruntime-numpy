import pytest
from onnxruntime_numpy.types import is_float, is_unsigned_int, is_signed_int, is_integer, is_numeric
import numpy as np

def test_type_sets():
    assert(all(map(is_float, [#np.float16, 
                              np.float32, 
                              np.float64])))

    assert(all(map(is_unsigned_int, [np.uint8, 
                                     np.uint16, 
                                     np.uint32,
                                     np.uint64])))

    assert(all(map(is_signed_int, [np.int8, 
                                   np.int16, 
                                   np.int32,
                                   np.int64])))

    assert(all(map(is_integer, [np.int8, 
                                np.int16, 
                                np.int32,
                                np.int64,
                                np.uint8,
                                np.uint16,
                                np.uint32,
                                np.uint64])))

    assert(all(map(is_numeric, [#np.float16, 
                                np.float32, 
                                np.float64,
                                np.int8, 
                                np.int16, 
                                np.int32,
                                np.int64,
                                np.uint8,
                                np.uint16,
                                np.uint32,
                                np.uint64])))
