import numpy as np


def expect(expected: np.ndarray, result: np.ndarray, rtol=1.e-4, **kwargs):
    assert expected.dtype == result.dtype
    assert expected.shape == result.shape
    assert np.allclose(expected, result, rtol=rtol, **kwargs)
