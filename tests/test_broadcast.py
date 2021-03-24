import onnxruntime_numpy as onp
import numpy as np
import pytest
from onnxruntime_numpy.types import float_types, all_types
from .utils import expect


@pytest.mark.parametrize("type_a", [*all_types])
def test_expand_dim_changed(type_a):

    shape = [3, 1]
    data = np.array([0,1,0], dtype=type_a).reshape(shape)
    new_shape = [2, 1, 6]
    expected = data * np.ones(new_shape, dtype=type_a)

    result = onp.expand(onp.array(data, dtype=type_a), 
                        onp.array(new_shape, dtype=np.int64))


    expect(expected, result.numpy())

@pytest.mark.parametrize("type_a", [*all_types])
def test_expand_dim_unchanged(type_a):

    shape = [3, 1]
    data = np.array([0,1,0], dtype=type_a).reshape(shape)
    new_shape = [3, 4]
    expected = np.tile(data, 4)

    result = onp.expand(onp.array(data, dtype=type_a), 
                        onp.array(new_shape, dtype=np.int64))


    expect(expected, result.numpy())
