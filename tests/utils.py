import numpy as np


def expect(expected: np.ndarray, result: np.ndarray):
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape
    assert np.allclose(expected, result)
