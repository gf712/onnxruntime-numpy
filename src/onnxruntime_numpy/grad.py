# flake8: noqa

from . import array
from . import ops
from .graph import Graph, Input, Node, Output
from typing import List


global_gradient_registry = {}


def register_gradient(opname, *grad_funcs):
    global_gradient_registry[opname] = tuple(grad_funcs)


def exp_grad(grad, output, x):
    return grad * output


def sinh_grad(grad, output, x):
    return grad * ops.cosh(x)


def cosh_grad(grad, output, x):
    return grad * ops.sinh(x)


def tanh_grad(grad, output, x):
    return grad / ops.cosh(x) ** 2


def log_grad(grad, output, x):
    return grad / x


register_gradient("Exp", exp_grad)
register_gradient("Sinh", sinh_grad)
register_gradient("Cosh", cosh_grad)
register_gradient("Tanh", tanh_grad)
register_gradient("Log", log_grad)


def gradient_factory(op):
    return global_gradient_registry[op]


def add_outgrads(prev_g, g):
    if prev_g is None:
        return g
    return prev_g + g


def backward_pass(g, graph):

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

        # if not output_grad._requires_grad:
        #     continue

        grad_funcs = global_gradient_registry.get(node.op_type)
        if grad_funcs is None:
            raise NotImplementedError(f"Gradient for {node.op_type} not implemented")

        if len(node.inputs) != len(grad_funcs):
            raise NotImplementedError()

        value = node.outputs[0]

        for idx, (output_array, grad_fn) in enumerate(zip(node.inputs, grad_funcs)):
            parent_node = [parent_node for (parent_node, this_node, data) in graph._graph.in_edges(
                node_name, data=True) if data["index"] == idx]
            if len(parent_node) != 1:
                raise ValueError("Internal error. Ambiguous node input.")
            parent_grad = grad_fn(output_grad, value, output_array)
            parent = graph.nodes[parent_node[0]]["node"]
            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)

    return output_grad


def grad(x: "array.Array", with_respect_to: List["array.Array"]):
    grad_graph = x._evaluator._graph  # .copy()

    inputs, output_name = grad_graph._build_graph_with_inputs_outputs(
        with_respect_to, x, x._evaluator._parent_node)

    g = ops.constant_of_shape(x.shape, 1.0)
    # g._requires_grad = False

    grad_result = backward_pass(g, grad_graph)

    grad_graph._remove_graph_inputs_outputs(inputs, output_name)

    return grad_result
