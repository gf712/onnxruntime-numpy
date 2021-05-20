import onnxruntime_numpy as onp
import numpy as np
from .utils import expect


def test_graph_cache():
    a = onp.array([1, 2, 3])
    b = onp.array([1, 2, 3])

    c = a + b
    expected = np.array([2, 4, 6], dtype=np.int32)

    g = c._internal_array._evaluator._graph
    #   Output_a
    #    /
    # Input_a
    #        \
    #         Add -- Output_c
    #        /
    # Input_b
    #   \
    #   Output_b
    #

    assert len(g.nodes) == 6

    exec_g = c._internal_array._evaluator._build_executable_graph()

    # Executable graph only keeps the Output nodes it needs
    # Input_a
    #        \
    #         Add -- Output_c
    #        /
    # Input_b
    assert len(exec_g._graph.nodes) == 4

    expect(expected, c.numpy())
    d = a + c

    # with caching d now has 4 nodes:
    # Input_a
    #        \
    #         Add -- Output_d
    #        /
    # Output_c
    #
    # without caching there are 6 nodes:
    # Input_a ----------- Add -- Output_d
    #        \           /
    #         Add -- Output_c
    #        /
    # Input_b

    exec_g = d._internal_array._evaluator._build_executable_graph()
    exec_g_no_cache = d._internal_array._evaluator._build_executable_graph(
        use_cache=False)

    assert len(exec_g._graph.nodes) == 4
    assert len(exec_g_no_cache._graph.nodes) == 6
