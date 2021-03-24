import pytest
import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types
import numpy as np
from .utils import expect


@pytest.mark.parametrize("type_a", [np.float32])
def test_elu(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([-1.2642411, 0., 1.], dtype=type_a)
    result = onp.elu(a, alpha=2.0)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_elu_default(type_a):
    a = onp.array([-1, 0, 1], dtype=type_a)
    expected = onp.array([-0.63212055, 0., 1.], dtype=type_a)
    result = onp.elu(a)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [np.float32])
def test_relu(type_a):
    a = onp.array([[1., 2., -3.], 
                   [-1, 0,  10.]], dtype=type_a)
    expected = onp.array([[1, 2, 0],    
                          [0, 0, 10]], dtype=type_a)
    result = onp.relu(a)

    expect(result.numpy(), expected.numpy())
