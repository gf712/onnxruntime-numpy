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
    a = onp.random.multinomial(onp.array(outcomes), dtype=type_b, seed=1)

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


@pytest.mark.parametrize("type_a", [np.float32, np.float64])
def test_normal(type_a):
    mean = 0.0
    scale = 1.0
    shape = [1, 2, 3]

    result = onp.random.normal(
        shape=shape, mean=mean, scale=scale, dtype=type_a, seed=42)
    if type_a == np.float32:
        expected = np.array(
            [[[-0.90124804, 0.8964087, -1.2027136],
              [-0.4901553, 0.01147013, 0.4431136]]],
            dtype=np.float32)
    else:
        expected = np.array(
            [[[-1.71411274,  0.17805683,  0.05717887],
              [-1.40979699,  0.75628399, -0.5822737]]],
            dtype=np.float64)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.float64])
def test_normal_like(type_a):
    mean = 0.0
    scale = 1.0
    x = onp.array(np.empty((1, 2, 3), dtype=type_a))

    result = onp.random.normal_like(
        x, mean=mean, scale=scale, dtype=type_a, seed=42)
    if type_a == np.float32:
        expected = np.array(
            [[[-0.90124804, 0.8964087, -1.2027136],
              [-0.4901553, 0.01147013, 0.4431136]]],
            dtype=np.float32)
    else:
        expected = np.array(
            [[[-1.71411274,  0.17805683,  0.05717887],
              [-1.40979699,  0.75628399, -0.5822737]]],
            dtype=np.float64)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.float64])
def test_uniform(type_a):
    low = 0.0
    high = 10.0
    shape = [1, 2, 3]

    result = onp.random.uniform(
        shape=shape, low=low, high=high, dtype=type_a, seed=42)
    if type_a == np.float32:
        expected = np.array(
            [[[3.2870704e-03, 5.2458711, 7.3542352],
              [2.6330554, 3.7622399, 1.9628583]]],
            dtype=np.float32)
    else:
        expected = np.array(
            [[[5.24587102, 2.63305541, 1.96285826],
              [5.12318109, 2.57101629, 8.15487627]]],
            dtype=np.float64)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [np.float32, np.float64])
def test_uniform_like(type_a):
    low = 0.0
    high = 10.0
    x = onp.array(np.empty((1, 2, 3), dtype=type_a))

    result = onp.random.uniform_like(
        x, low=low, high=high, dtype=type_a, seed=42)
    if type_a == np.float32:
        expected = np.array(
            [[[3.2870704e-03, 5.2458711, 7.3542352],
              [2.6330554, 3.7622399, 1.9628583]]],
            dtype=np.float32)
    else:
        expected = np.array(
            [[[5.24587102, 2.63305541, 1.96285826],
              [5.12318109, 2.57101629, 8.15487627]]],
            dtype=np.float64)

    expect(expected, result.numpy())
