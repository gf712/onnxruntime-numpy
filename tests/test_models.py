import pytest
import numpy as np
import onnxruntime_numpy as onp
from .utils import expect


@pytest.mark.integration
def test_integration_multilayer_perceptron():
    batch_size = 32
    n_features = 10

    def relu(x):
        return np.maximum(0, x)

    X = np.random.rand(batch_size, n_features).astype(np.float32)

    w1 = np.random.rand(n_features, 64).astype(np.float32)
    b1 = np.random.rand(64).astype(np.float32)
    expected = relu(X @ w1 + b1)

    w2 = np.random.rand(64, 1).astype(np.float32)
    b2 = np.random.rand(1).astype(np.float32)
    expected = expected @ w2 + b2

    result = onp.nn.relu(onp.array(X) @ onp.array(w1) + onp.array(b1))
    result = result @ onp.array(w2) + onp.array(b2)

    expect(expected, result.numpy())
