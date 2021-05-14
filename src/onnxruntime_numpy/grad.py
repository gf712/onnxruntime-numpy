# TODO: fix types and formatting after refactor
# type: ignore
# flake: noqa

from .array import Array
from .evaluator import merge_array_evaluators, IntermediateResultCache
from .ops_utils import nary_operator
from . import ops
from .graph import Graph, Input
from .tracer import OpTracerContext
from typing import Dict, Tuple, Callable
from .graph import ExecutableGraph
from .config import HAS_ONNXRUNTIME_TRAINING

global_gradient_registry: Dict[str, Tuple[Callable, ...]] = {}


def gradient_factory(op):
    return global_gradient_registry.get(op)


def add_outgrads(prev_g, g):
    if prev_g is None:
        return g
    return prev_g + g


def register_gradient(op, *grad_funcs):
    global_gradient_registry[op.__qualname__] = tuple(grad_funcs)


def exp_grad(grad, output, x):
    return grad * output


def sinh_grad(grad, output, x):
    return grad * ops.cosh(x)


def sin_grad(grad, output, x):
    if HAS_ONNXRUNTIME_TRAINING:
        return nary_operator("SinGrad", grad, x)
    else:
        return grad * ops.cos(x)


def cos_grad(grad, output, x):
    return grad * ops.sin(x)


def cosh_grad(grad, output, x):
    return grad * ops.sinh(x)


def tanh_grad(grad, output, x):
    return grad / ops.cosh(x) ** 2


def log_grad(grad, output, x):
    return grad / x


def matmul_grad_wrtA(grad, output, A, B):
    if len(A.shape) == 2 and len(B.shape) == 2:
        return ops.gemm(grad, B, transB=True)
    return ops.matmul(grad, B.T)


def matmul_grad_wrtB(grad, output, A, B):
    # dispatch matmul of matrices to gemm
    if len(A.shape) == 2 and len(B.shape) == 2:
        return ops.gemm(A, grad, transA=True)
    # other tensor shapes are dispatched to matmul
    return ops.matmul(A.T, grad)


register_gradient(ops.exp,  exp_grad)
register_gradient(ops.sinh, sinh_grad)
register_gradient(ops.cosh, cosh_grad)
# register_gradient(ops.tanh, tanh_grad)
register_gradient(ops.log,  log_grad)
register_gradient(ops.sin,  sin_grad)
register_gradient(ops.cos,  cos_grad)
register_gradient(ops.matmul,  matmul_grad_wrtA, matmul_grad_wrtB)


def backward_pass(g, graph) -> Array:

    it = graph.reversed_toposort()
    next(it)
    node = graph._graph.nodes[next(it)]["node"]
    outgrads = {node: g}

    topo_iterator = graph.reversed_toposort()
    next(topo_iterator)

    for node_name in topo_iterator:
        node = graph.nodes[node_name]["node"]

        output_grad = outgrads.pop(node)

        if isinstance(node, Input):
            continue

        grad_funcs = gradient_factory(node.op_type)
        if grad_funcs is None:
            raise NotImplementedError(
                f"Gradient for {node.op_name} not implemented")

        if len(node.inputs) != len(grad_funcs):
            raise NotImplementedError()

        node_output = node.outputs[0]

        for idx, (output_array, grad_fn) in enumerate(
                zip(node.inputs, grad_funcs)):
            parent_node = [
                parent_node
                for(parent_node, this_node, data) in graph._graph.in_edges(
                    node_name, data=True) if data["index"] == idx]
            if len(parent_node) != 1:
                raise ValueError("Internal error. Ambiguous node input.")
            parent_grad = grad_fn(output_grad, node_output,
                                  *node.inputs, **node.attributes)
            parent = graph.nodes[parent_node[0]]["node"]
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)

    return output_grad


def grad_fn(func, argnum):

    def grad_helper(*array_objs, **array_obj_kwargs):

        inputs = [array_objs[argnum]]

        grad_graph = Graph()

        with OpTracerContext(grad_graph, *array_objs, **array_obj_kwargs) as tracker:
            result_array = tracker.trace_function_call(func)

        grad_graph = ExecutableGraph(
            grad_graph, inputs,
            {result_array._evaluator._parent_node: result_array})._graph

        g = ops.constant_of_shape(result_array.shape, 1.0)

        grad_result = backward_pass(g, grad_graph)

        other_evaluators = [
            a._evaluator for a in array_objs
        ] + [
            a._evaluator for a in array_obj_kwargs.values()
        ]

        merge_array_evaluators(grad_result._evaluator, *other_evaluators)

        return grad_result

    return grad_helper


def grad(output: Array, *inputs: Array):

    g = ops.constant_of_shape(output.shape, 1.0)

    grad_graph = ExecutableGraph(
        output._evaluator._graph, output._evaluator._array_to_node_map.get_input_map(),
        output._evaluator._array_to_node_map.get_output_map(),
        {input._internal_name: input for input in inputs},
        {output._evaluator._parent_node: output},
        IntermediateResultCache())._graph

    return backward_pass(g, grad_graph)
