import pytest
import onnxruntime_numpy as onp
from onnxruntime_numpy.config import get_ort_graph_optimization_level
from onnxruntime import GraphOptimizationLevel
from .utils import expect
import numpy as np


@pytest.mark.parametrize("optimization", list(onp.GraphOptimizationLevel))
def test_onnxruntime_graph_optimization(optimization):
    onp.Config().set_graph_optimization(optimization)
    a = onp.array([1., 2., 3.])
    a += a
    expected = np.array([2., 4., 6.], dtype=np.float32)

    expect(expected, a.numpy())

    if optimization == onp.GraphOptimizationLevel.DISABLED:
        onnxruntime_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    elif optimization == onp.GraphOptimizationLevel.BASIC:
        onnxruntime_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif optimization == onp.GraphOptimizationLevel.EXTENDED:
        onnxruntime_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    elif optimization == onp.GraphOptimizationLevel.ALL:
        onnxruntime_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        raise ValueError("Unknown GraphOptimizationLevel")

    assert get_ort_graph_optimization_level() == onnxruntime_optimization_level
