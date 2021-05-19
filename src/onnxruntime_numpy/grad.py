from .array import Array, array
from .ops_utils import nary_operator
from . import ops
from . import nn
from .graph import Input
from .config import Config
from .exceptions import InternalException
from .core import _fetch_array
import networkx as nx
import numpy as np
from typing import Dict, Tuple, Callable, Optional, List, Set
from enum import Enum
import functools
import operator

global_gradient_registry: Dict[str, Tuple[Callable, ...]] = {}


class _StopGradient:
    pass


class StopGradientsValue(Enum):
    NONE = 0
    Zero = 1


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


def add_grad_dx(grad, output, x, y):
    return unbroadcast(grad, x)


def add_grad_dy(grad, output, x, y):
    return unbroadcast(grad, y)


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
    return unbroadcast(grad / y, x)


def divide_grad_dy(grad, output, x, y):
    return unbroadcast(- grad * x / ops.square(y), y)


def exp_grad(grad, output, x):
    return grad * output


def log_grad(grad, output, x):
    return grad / x


def matmul_grad_dA(grad, output, A, B):
    if grad.ndims == 0:
        # output is a scalar
        return unbroadcast(grad * B, A)
    if len(A.shape) == 2 and len(B.shape) == 2:
        return unbroadcast(ops.gemm(grad, B, transB=True), A)
    return unbroadcast(ops.matmul(grad, B.T), A)


def matmul_grad_dB(grad, output, A, B):
    if grad.ndims == 0:
        # output is a scalar
        return unbroadcast(grad * A, B)
    # dispatch matmul of matrices to gemm
    if len(A.shape) == 2 and len(B.shape) == 2:
        return unbroadcast(ops.gemm(A, grad, transA=True), B)
    # other tensor shapes are dispatched to matmul
    return unbroadcast(ops.matmul(A.T, grad), B)


def mean_grad_dx(grad: Array, output: Array, x: Array, **kwargs):
    keep_dims: bool = kwargs.get("keep_dims")  # type: ignore

    # code below adapted from tf
    input_shape = x.shape
    if keep_dims:
        output_shape = output.shape.tolist()
    else:
        axes: Optional[List[int]] = kwargs.get("axes")
        if axes is None:
            axes = list(range(x.ndims))
        else:
            # insert 1's in every axis that has been reduced
            output_shape = [0] * x.ndims
            dim_i = 0
            for idx in range(len(output_shape)):
                if idx in axes:
                    output_shape[idx] = 1
                else:
                    output_shape[idx] = int(output.shape[dim_i])
                    dim_i += 1

    tile_scaling = [
        int(i) / int(o) for i,
        o in zip(input_shape.tolist(), output_shape)]
    grad = ops.reshape(grad, output_shape)
    outgrad = ops.tile(grad, array(tile_scaling, dtype=np.int64))
    # then we need to normalize each axis (using the axis dimensionality)
    return outgrad / array(functools.reduce(operator.mul, tile_scaling, 1),
                           dtype=outgrad.dtype)


def multiply_grad_dx(grad, output, x, y):
    return unbroadcast(grad * y, x)


def multiply_grad_dy(grad, output, x, y):
    return unbroadcast(grad * x, y)


def power_grad_dbase(grad, output, base, exponent):
    return grad * exponent * base ** (
        exponent - array([1], dtype=exponent.dtype))


def power_grad_dexponent(grad, output, base, exponent):
    return grad * output * ops.log(base)


def relu_grad(grad, output, x):
    return ops.where(
        x < array([0],
                  dtype=x.dtype),
        array([0],
              dtype=x.dtype),
        array([1],
              dtype=x.dtype))


def sinh_grad(grad, output, x):
    return grad * ops.cosh(x)


def sin_grad(grad, output, x):
    if Config().onnxruntime_training_available():
        return nary_operator("SinGrad", grad, x)
    else:
        return grad * ops.cos(x)


# def square_grad(grad, output, x):
#     return grad * array([2], dtype=x.dtype) * x


def sqrt_grad(grad, output, x):
    return grad / (array([2], dtype=x.dtype) * output)


def subtract_grad_dx(grad, output, x, y):
    return unbroadcast(grad, x)


def subtract_grad_dy(grad, output, x, y):
    return unbroadcast(-grad, y)


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
register_gradient(ops.mean,     mean_grad_dx)
register_gradient(ops.multiply, multiply_grad_dx, multiply_grad_dy)
register_gradient(ops.power,    power_grad_dbase, power_grad_dexponent)
register_gradient(nn.relu,      relu_grad)
register_gradient(ops.sin,      sin_grad)
register_gradient(ops.sinh,     sinh_grad)
# FIXME: this should be just f'(x) = 2x
# but square is defined as x * x (since there is no square op in ONNX)
# so it is actually a binary operation with a single input/output
#  x                    x
#    \                 /  \
#     out  -> grad: out    + ---> 2x
#    /                 \  /
#  x                    x
# this means that the current framework will do dx/dout twice, even though
# we just want one pass to immediately get 2x
# register_gradient(ops.square,   square_grad)
register_gradient(ops.sqrt,     sqrt_grad)
register_gradient(ops.subtract, subtract_grad_dx, subtract_grad_dy)
register_gradient(ops.tan,      tan_grad)
register_gradient(ops.tanh,     tanh_grad)


def reversed_toposort(graph: nx.DiGraph):
    return reversed(list(nx.topological_sort(graph)))


def unbroadcast(output: Array, input: Array):
    """Given an output array undo the broadcasting so that it
       returns an array with input's shape.

    This code was taken from autograd.

    Args:
        output (Array): [description]
        input (Array): [description]
    """
    target_shape = input.shape

    # TODO: figure out what this does so it can be simplified :)
    if output.ndims > input.ndims:
        output = ops.sum(
            output, list(range(output.ndims - input.ndims)),
            keepdims=False)

    axes = []
    for axis, size in enumerate(target_shape):
        if size == 1:
            axes.append(axis)

    if len(axes) > 0:
        output = ops.sum(output, axes=axes, keepdims=True)

    return output


def backward_pass(g, graph: nx.DiGraph, output_evaluator,
                  inputs: Tuple[Array, ...],
                  stop_gradient_value: StopGradientsValue) -> Tuple[Array, ...]:
    it = reversed_toposort(graph)
    node = graph.nodes[next(it)]["node"]
    outgrads = {node: g}

    topo_iterator = reversed_toposort(graph)

    for node_name in topo_iterator:
        node = graph.nodes[node_name]["node"]

        output_grad = outgrads[node]
        input_edges = graph._graph.in_edges(node_name)

        if isinstance(node, Input):
            for input_edge in input_edges:
                parent_node = input_edge[0]
                parent_node["node"]["stop_gradient"] = True
            continue
        else:
            grad_funcs = gradient_factory(node.op_type)
            if grad_funcs is None:
                raise NotImplementedError(
                    f"Gradient for {node.op_name} not implemented")

            if len(input_edges) != len(grad_funcs):
                raise NotImplementedError()

            # output_edges = graph._graph.out_edges(node_name)

            output_mapping = output_evaluator._array_to_node_map.get_output_map()
            input_mapping = output_evaluator._array_to_node_map.get_input_map()

            for idx, (input_edge, grad_fn) in enumerate(
                    zip(input_edges, grad_funcs)):
                parent_node_name = input_edge[0]
                if parent_node_name not in graph.nodes:
                    # the node connecting to this node is not relevant
                    # to compute the gradient
                    #
                    # For example for dx/dbase we don't need dexponent/dout
                    #   base
                    #      \
                    #      Power -- out
                    #      /
                    # exponent
                    #
                    # Also, the graph passed to backward_pass only contains the
                    # nodes that are needed to compute the requested partials
                    continue
                parent_node = graph.nodes[parent_node_name]["node"]

                if "stop_gradient" in graph.nodes[parent_node_name] \
                        or output_grad is None:
                    # the output node in the forward graph is flagged
                    # as not participating in gradient, so ignore it
                    outgrads[parent_node] = outgrads.get(parent_node)
                    continue
                node_outputs = [_fetch_array(a[0])
                                for a in output_mapping[node_name]]
                node_inputs = [_fetch_array(a[0])
                               for a in input_mapping[node_name]]
                parent_grad = grad_fn(output_grad, *node_outputs,
                                      *node_inputs, **node.attributes)
                if isinstance(parent_grad, _StopGradient):
                    # TODO: not sure this is correct, but not used currently
                    parent_node["node"]["stop_gradient"] = True
                    outgrads[parent_node] = outgrads.get(parent_node)
                else:
                    outgrads[parent_node] = add_outgrads(
                        outgrads.get(parent_node), parent_grad)

    input_mapping = output_evaluator._array_to_node_map.get_input_map()
    output_mapping = output_evaluator._array_to_node_map.get_output_map()

    outgrads_node_names = {k.node_name: v for k, v in outgrads.items()}

    def get_result(output_name):
        if output_name in outgrads_node_names:
            return outgrads_node_names[output_name]
        elif stop_gradient_value == StopGradientsValue.NONE:
            return None
        elif stop_gradient_value == StopGradientsValue.Zero:
            raise NotImplementedError()

    return tuple(
        get_result(i._internal_array._evaluator._parent_node) for i in inputs)


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


def gradients(
        output: Array, inputs: List[Array],
        stop_gradients: Optional[List[Array]] = None,
        unconnected_gradients: StopGradientsValue = StopGradientsValue.NONE):

    if output.ndims == 0:
        # TODO: constant_of_shape will cause issues when asked to
        # initialise a scalar
        g = array(1, dtype=output.dtype.type)
    else:
        g = ops.cast(ops.constant_of_shape(output.shape, 1.), output.dtype.type)

    output_graph = output._internal_array._evaluator._graph._graph.copy()
    output_node_name = output._internal_array._evaluator._parent_node

    nodes_of_interest: Set[str] = set()
    for input_array in inputs:
        input_node_name = input_array._internal_array._evaluator._parent_node
        if input_node_name is None:
            raise InternalException("Input array has no internal name")
        nodes_of_interest.add(input_node_name)
        try:
            nodes_of_interest.update(
                *nx.all_simple_paths(output_graph, source=input_node_name,
                                     target=output_node_name))
        except Exception:
            pass

    # subgraph includes ancestor node and the output itself
    # output wrt itself will have gradient = 1, but add it anyway for completeness
    subgraph_nodes = [output_node_name, *nodes_of_interest]

    # view of the nodes of interest
    graph = output._internal_array._evaluator._graph._graph.subgraph(
        subgraph_nodes)

    if stop_gradients is not None:
        for stop_gradient in stop_gradients:
            node_name = stop_gradient._internal_array._evaluator._parent_node
            graph.nodes[node_name]["stop_gradient"] = True

    return backward_pass(
        g, graph, output._internal_array._evaluator, tuple(inputs),
        unconnected_gradients)
