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


def test_graph_pruning():

    a = onp.array([1, 2, 3])
    b = onp.array([1, 2, 3])

    # the original a is now out of scope
    # and it is impossible to use
    a += b

    # however it (a1) is still needed to compute the new a (a2)
    # so the computational graph is
    #   Output_a1
    #    /
    # Input_a
    #        \
    #         Add -- Output_a2
    #        /
    # Input_b
    #   \
    #   Output_b
    #

    g = a._internal_array._evaluator._graph
    assert len(g.nodes) == 6

    # a (a2) is now evaluated, so a1 is not needed to compute anything
    a.eval()

    # So now a2 has graph
    #        Output_a1
    #        /
    # Unreachable_a
    #        \
    #         Add -- Output_a2
    #        /
    # Input_b
    #   \
    #   Output_b
    #
    # This won't break anything since Output_a2 is now cached
    # but to compute da2/db we may need a1, so this has to be cached

    # TODO: implement this!
    g = a._internal_array._evaluator._graph
    assert len(g.nodes) == 6

    b += a
    #        Output_a1
    #        /
    # Unreachable_a
    #        \
    #         Add -- Output_a2
    #         / \------------
    #        /                \
    # Unreachable_b ---------- Add -- Output_a3
    #       \
    #       Output_b
    #
    # which can now be:
    #     Output_a2
    #    /
    # Input_a2 ---------------
    #                         \
    # Unreachable_b --------- Add -- Output_a3
    #   \
    #   Output_b
    #
    # And the graph is the same as the one at the start
    b.eval()
    # TODO: implement this!
    g = b._internal_array._evaluator._graph
    assert len(g.nodes) == 6  # instead of 8
