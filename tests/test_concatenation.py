import onnxruntime_numpy as onp
from onnxruntime_numpy.types import all_types
import pytest
from .utils import expect


@pytest.mark.parametrize("type_a", [*all_types])
def test_concat(type_a):
    a = onp.array([0, 1, 1], dtype=type_a)
    b = onp.array([0, 1, 1], dtype=type_a)
    c = onp.array([0, 1, 1], dtype=type_a)

    expected = onp.array([0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=type_a)
    result = onp.concat([a, b, c])
    expect(expected.numpy(), result.numpy())
