from .array import Array, array
from .ops_utils import nary_operator
from . import ops
from .graph import Input
from typing import Dict, Tuple, Callable
from .config import HAS_ONNXRUNTIME_TRAINING
import networkx as nx

global_gradient_registry: Dict[str, Tuple[Callable, ...]] = {}


def gradient_factory(op):
    return global_gradient_registry.get(op)


def add_outgrads(prev_g, g):
    if prev_g is None:
        return g
    return prev_g + g


def register_gradient(op, *grad_funcs):
    global_gradient_registry[op.__qualname__] = tuple(grad_funcs)


def abs_grad(grad, output, x):
    return x / output


def acos_grad(grad, output, x):
    return -ops.reciprocal(
        ops.sqrt(grad - ops.square(x)))


def acosh_grad(grad, output, x):
    return ops.reciprocal(
        ops.sqrt(x - grad)
        * ops.sqrt(x + grad))


def add_grad_dx(grad, output, *args):
    return grad


def add_grad_dy(grad, output, *args):
    return grad


def asin_grad(grad, output, x):
    return ops.reciprocal(
        ops.sqrt(grad - ops.square(x)))


def asinh_grad(grad, output, x):
    return ops.reciprocal(
        ops.sqrt(grad + ops.square(x)))


def atan_grad(grad, output, x):
    return ops.reciprocal(
        grad + ops.square(x))


def atanh_grad(grad, output, x):
    return ops.reciprocal(
        grad - ops.square(x))


def cos_grad(grad, output, x):
    return grad * - ops.sin(x)


def cosh_grad(grad, output, x):
    return grad * ops.sinh(x)


def divide_grad_dx(grad, output, x, y):
    return grad / y


def divide_grad_dy(grad, output, x, y):
    return - grad * x / ops.square(y)


def exp_grad(grad, output, x):
    return grad * output


def log_grad(grad, output, x):
    return grad / x


def matmul_grad_dA(grad, output, A, B):
    if grad.ndims == 0:
        # output is a scalar
        return grad * B
    if len(A.shape) == 2 and len(B.shape) == 2:
        return ops.gemm(grad, B, transB=True)
    return ops.matmul(grad, B.T)


def matmul_grad_dB(grad, output, A, B):
    if grad.ndims == 0:
        # output is a scalar
        return grad * A
    # dispatch matmul of matrices to gemm
    if len(A.shape) == 2 and len(B.shape) == 2:
        return ops.gemm(A, grad, transA=True)
    # other tensor shapes are dispatched to matmul
    return ops.matmul(A.T, grad)


def multiply_grad_dx(grad, output, x, y):
    return y


def multiply_grad_dy(grad, output, x, y):
    return x


def power_grad_dbase(grad, output, base, exponent):
    return grad * exponent * base ** (
        exponent - array([1], dtype=exponent.dtype))


def power_grad_dexponent(grad, output, base, exponent):
    return grad * output * ops.log(base)


def sinh_grad(grad, output, x):
    return grad * ops.cosh(x)


def sin_grad(grad, output, x):
    if HAS_ONNXRUNTIME_TRAINING:
        return nary_operator("SinGrad", grad, x)
    else:
        return grad * ops.cos(x)


def square_grad(grad, output, x):
    return grad * array([2], dtype=x.dtype) * x


def sqrt_grad(grad, output, x):
    return grad / (array([2], dtype=x.dtype) * output)


def subtract_grad_dx(grad, output, *args):
    return grad


def subtract_grad_dy(grad, output, *args):
    return -grad


def tanh_grad(grad, output, x):
    return grad / ops.cosh(x) ** 2


def tan_grad(grad, output, x):
    return grad / ops.square(ops.cos(x))


register_gradient(ops.absolute, abs_grad)
register_gradient(ops.acos,     acos_grad)
register_gradient(ops.acosh,    acosh_grad)
register_gradient(ops.add,      add_grad_dx, add_grad_dy)
register_gradient(ops.asin,     asin_grad)
register_gradient(ops.asinh,    asinh_grad)
register_gradient(ops.atan,     atan_grad)
register_gradient(ops.atanh,    atanh_grad)
register_gradient(ops.cos,      cos_grad)
register_gradient(ops.cosh,     cosh_grad)
register_gradient(ops.divide,   divide_grad_dx, divide_grad_dy)
register_gradient(ops.exp,      exp_grad)
register_gradient(ops.log,      log_grad)
register_gradient(ops.matmul,   matmul_grad_dA, matmul_grad_dB)
register_gradient(ops.multiply, multiply_grad_dx, multiply_grad_dy)
register_gradient(ops.power,    power_grad_dbase, power_grad_dexponent)
register_gradient(ops.sin,      sin_grad)
register_gradient(ops.sinh,     sinh_grad)
register_gradient(ops.square,   square_grad)
register_gradient(ops.sqrt,   sqrt_grad)
register_gradient(ops.subtract, subtract_grad_dx, subtract_grad_dy)
register_gradient(ops.tan,      tan_grad)
register_gradient(ops.tanh,     tanh_grad)


def reversed_toposort(graph: nx.DiGraph):
    return reversed(list(nx.topological_sort(graph)))


def backward_pass(g, graph: nx.DiGraph, output_evaluator, inputs: Tuple
                  [Array, ...]) -> Tuple[Array, ...]:
    it = reversed_toposort(graph)
    node = graph.nodes[next(it)]["node"]
    outgrads = {node: g}

    topo_iterator = reversed_toposort(graph)

    for node_name in topo_iterator:
        node = graph.nodes[node_name]["node"]

        output_grad = outgrads[node]

        if isinstance(node, Input):
            continue

        grad_funcs = gradient_factory(node.op_type)
        if grad_funcs is None:
            raise NotImplementedError(
                f"Gradient for {node.op_name} not implemented")

        input_edges = graph._graph.in_edges(node_name)

        if len(input_edges) != len(grad_funcs):
            raise NotImplementedError()

        # output_edges = graph._graph.out_edges(node_name)

        output_mapping = output_evaluator._array_to_node_map.get_output_map()
        input_mapping = output_evaluator._array_to_node_map.get_input_map()

        for idx, (input_edge, grad_fn) in enumerate(
                zip(input_edges, grad_funcs)):
            parent_node = input_edge[0]
            node_outputs = output_mapping[node_name]
            node_inputs = input_mapping[node_name]
            parent_grad = grad_fn(output_grad, *node_outputs,
                                  *node_inputs, **node.attributes)
            parent = graph.nodes[parent_node]["node"]
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)

    input_mapping = output_evaluator._array_to_node_map.get_input_map()
    output_mapping = output_evaluator._array_to_node_map.get_output_map()

    outgrads_node_names = {k.node_name: v for k, v in outgrads.items()}

    return tuple(
        outgrads_node_names[i._evaluator._parent_node] for i in inputs)


# def grad_fn(func, argnum):

#     def grad_helper(*array_objs, **array_obj_kwargs):

#         inputs = [array_objs[argnum]]

#         grad_graph = Graph()

#         with OpTracerContext(grad_graph, *array_objs, **array_obj_kwargs) as tracker:
#             result_array = tracker.trace_function_call(func)

#         grad_graph = ExecutableGraph(
#             grad_graph, inputs,
#             {result_array._evaluator._parent_node: result_array})._graph

#         g = ops.constant_of_shape(result_array.shape, 1.0)

#         grad_result = backward_pass(g, grad_graph)

#         other_evaluators = [
#             a._evaluator for a in array_objs
#         ] + [
#             a._evaluator for a in array_obj_kwargs.values()
#         ]

#         merge_array_evaluators(grad_result._evaluator, *other_evaluators)

#         return grad_result

#     return grad_helper


def grad(output: Array, *inputs: Array):

    if output.ndims == 0:
        # TODO: constant_of_shape will cause issues when asked to
        # initialise a scalar
        g = array(1, dtype=output.dtype.type)
    else:
        g = ops.cast(ops.constant_of_shape(output.shape, 1.), output.dtype.type)

    output_graph = output._evaluator._graph._graph
    output_node_name = output._evaluator._parent_node

    ancestor_nodes = nx.ancestors(output_graph, output_node_name)

    # subgraph includes ancestor node and the output itself
    # output wrt itself will have gradient = 1, but add it anyway for completeness
    subgraph_nodes = [output._evaluator._parent_node, *ancestor_nodes]

    # view of the nodes of interest
    graph = output._evaluator._graph._graph.subgraph(subgraph_nodes)

    return backward_pass(g, graph, output._evaluator, inputs)
