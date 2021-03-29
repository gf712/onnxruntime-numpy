from onnxruntime_numpy.tracer import OpTracerContext
from onnxruntime_numpy.graph import Graph
import onnxruntime_numpy as onp
from .utils import expect


def test_optracer_new_inputs():

    def f(x, y):
        return x + y

    a = onp.array([1., 2., 3.])
    b = onp.array([1., 2., 3.])

    g = Graph()

    with OpTracerContext(g, a, b) as tracer:
        c = tracer.trace_function_call(f)

    assert len(g.nodes) == 1

    node_name = list(g.nodes)[0]
    assert g.nodes[node_name]["node"].op_type == "Add"

    assert len(g._input_edges) == 2
    assert g._input_edges[a._internal_name] == node_name
    assert g._input_edges[b._internal_name] == node_name

    expected = onp.array([2., 4., 6.])
    expect(expected.numpy(), c.numpy())


def test_optracer_inputs_with_history():

    def f(x, y):
        return x + y

    a = onp.array([1., 2., 3.])
    b = onp.array([1., 2., 3.])
    c = a + b

    g = Graph()

    with OpTracerContext(g, a, c) as tracer:
        d = tracer.trace_function_call(f)

    assert len(g.nodes) == 1

    node_name = list(g.nodes)[0]
    assert g.nodes[node_name]["node"].op_type == "Add"

    assert len(g._input_edges) == 2
    assert g._input_edges[a._internal_name] == node_name
    assert g._input_edges[c._internal_name] == node_name

    expected = onp.array([3., 6., 9.])
    expect(expected.numpy(), d.numpy())
