import pytest
import onnxruntime_numpy as onp
# from onnxruntime_numpy.types import float_types
import numpy as np
from .utils import expect


@pytest.mark.parametrize("type_a", [np.float32])
@pytest.mark.parametrize("type_b", [np.int32, np.int64])
def test_multinomial(type_a, type_b):
    batch_size = 32
    class_size = 10
    rng = np.random.default_rng(seed=1)

    outcomes = np.log(rng.random((batch_size, class_size), dtype=type_a))
    a = onp.multinomial(onp.array(outcomes), dtype=type_b, seed=1)

    expected = np.array([[1],
                         [3],
                         [1],
                         [6],
                         [9],
                         [5],
                         [0],
                         [4],
                         [0],
                         [0],
                         [6],
                         [9],
                         [5],
                         [7],
                         [7],
                         [7],
                         [1],
                         [3],
                         [6],
                         [3],
                         [9],
                         [7],
                         [1],
                         [8],
                         [4],
                         [5],
                         [2],
                         [1],
                         [9],
                         [1],
                         [5],
                         [2]], dtype=type_b)

    expect(expected, a.numpy())
