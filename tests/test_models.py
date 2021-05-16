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


@pytest.mark.integration
def test_integration_multilayer_perceptron_training():
    batch_size = 32
    n_features = 1
    # FIXME: actually check that the loss is going down over several epochs
    epochs = 1
    learning_rate = onp.array(0.01)

    def relu(x):
        return np.maximum(0, x)

    X = np.random.rand(batch_size, n_features).astype(np.float32)
    y = X + (np.random.rand(batch_size, 1).astype(np.float32) / 10.)

    w1 = np.random.rand(n_features, 64).astype(np.float32)
    b1 = np.random.rand(64).astype(np.float32)

    w2 = np.random.rand(64, 1).astype(np.float32)
    b2 = np.random.rand(1).astype(np.float32)

    X = onp.array(X)
    y = onp.array(y)
    w1 = onp.array(w1)
    b1 = onp.array(b1)
    w2 = onp.array(w2)
    b2 = onp.array(b2)

    for _ in range(epochs):
        result = onp.nn.relu(X @ w1 + b1)
        result = result @ w2 + b2

        loss = onp.square(result - y).mean()

        dw1, db1, dw2, db2 = onp.gradients(loss, [w1, b1, w2, b2])

        w1 += dw1 * learning_rate
        b1 += db1 * learning_rate
        w2 += dw2 * learning_rate
        b2 += db2 * learning_rate

        w1._eval()
        b2._eval()
        w2._eval()
        b2._eval()
