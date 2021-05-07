# from onnxruntime_numpy.tracer import OpTracerContext
# from onnxruntime_numpy.graph import Graph
# import onnxruntime_numpy as onp
# from .utils import expect


# FIXME: Graph refactor broke the tracer logic
# def test_optracer_new_inputs():

#     def f(x, y):
#         return x + y

#     a = onp.array([1., 2., 3.])
#     b = onp.array([1., 2., 3.])

#     g = Graph()

#     with OpTracerContext(g, a, b) as tracer:
#         c = tracer.trace_function_call(f)

#     assert len(g.nodes) == 1

#     node_name = list(g.nodes)[0]
#     assert g.nodes[node_name]["node"].op_name == "Add"

#     assert len(c._evaluator._input_values) == 2
#     assert a._internal_name in c._evaluator._input_values
#     assert b._internal_name in c._evaluator._input_values

#     expected = onp.array([2., 4., 6.])
#     expect(expected.numpy(), c.numpy())


# def test_optracer_inputs_with_history():

#     def f(x, y):
#         return x * y

#     a = onp.array([1., 2., 3.])
#     b = onp.array([1., 2., 3.])
#     c = a + b

#     g = Graph()

#     with OpTracerContext(g, a, c) as tracer:
#         d = tracer.trace_function_call(f)

#     assert len(g.nodes) == 1

#     node_name = list(g.nodes)[0]
#     assert g.nodes[node_name]["node"].op_name == "Mul"

#     assert len(c._evaluator._input_values) == 2
#     assert a._internal_name in d._evaluator._input_values
#     assert b._internal_name in d._evaluator._input_values

#     expected = onp.array([3., 6., 9.])
#     expect(expected.numpy(), d.numpy())
