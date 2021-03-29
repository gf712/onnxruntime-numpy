from . import array
from . import evaluator
from .einsum_helper import einsum_parse_and_resolve_equation
import functools
from typing import Tuple, Union
import numpy as np
from functools import reduce
import operator

STRICT_MODE = True


def add_node(evaluator, op_type, input_arrays, output_arrays, **attributes):
    attributes = {k: v for k, v in attributes.items() if v is not None}
    evaluator.add_node(op_type,
                       input_arrays,
                       output_arrays,
                       **attributes)


def nary_operator(op_name, *arrays, **attributes):
    # if len(arrays) > 0:
    #     arrays = list(filter(lambda a: a is not None, arrays))
    if len(arrays) == 0:
        raise ValueError("")
    if any(map(lambda a: a is not None, arrays)):
        first_not_none_idx = [idx for idx,
                              a in enumerate(arrays) if a is not None][0]
        new_evaluator = arrays[first_not_none_idx]._evaluator.copy()
        if len(arrays) > 1:
            for array_i in arrays[1:]:
                if array_i is not None:
                    new_evaluator.merge(array_i._evaluator)
    else:
        # if all inputs are none we create a new evaluator
        # and the graph starts from here
        # TODO: is this ever valid in the ONNX standard?
        new_evaluator = evaluator.LazyEvaluator()

    evaluators = [new_evaluator]
    n_return_values = 1
    if n_return_values > 1:
        for x in range(1, n_return_values):
            evaluators.append(new_evaluator.copy())
    new_arrays = []
    for e in evaluators:
        new_arrays.append(array.Array(evaluator=e))
        # by default assume that the output shape and dtype are the same as lhs
        new_arrays[-1]._dtype = arrays[0].dtype
        new_arrays[-1]._dims = arrays[0].shape

    add_node(new_evaluator, op_name, arrays, new_arrays, **attributes)

    # FIXME: assuming single output for now
    return new_arrays[0]


def unary_operator(array_obj: "array.Array", op_name: str, **attributes):
    return nary_operator(op_name, array_obj, **attributes)


def binary_operator(
        lhs: "array.Array", rhs: "array.Array", op_name: str, **attributes):
    return nary_operator(op_name, lhs, rhs, **attributes)


def initializer_operator(op_name: str, **attributes):
    if len(attributes) != 1:
        raise NotImplementedError("TODO")

    attribute, value = next(iter(attributes.items()))
    new_evaluator = evaluator.LazyEvaluator()
    new_array = array.Array(evaluator=new_evaluator)
    value._internal_name = new_array._internal_name
    attributes[attribute] = value
    new_array._dtype = value.dtype
    new_array._dims = value.shape

    add_node(new_evaluator, op_name, [], [new_array], **attributes)
    return new_array


def deduce_output_type(lhs: "array.Array", rhs: "array.Array"):
    if lhs._dtype != rhs._dtype:
        raise ValueError(
            f"Cannot handle differing types for lhs ({lhs._dtype}) and rhs "
            f"({rhs._dtype})")
    return lhs._dtype


def allowed_types(*expected_types):
    def decorator(func):
        @functools.wraps(func)
        def wrapper_allowed_types(*arrays, **kwargs):
            # type checks
            for idx, (array_obj, dtypes) in enumerate(
                    zip(arrays, expected_types)):
                if array_obj is not None and array_obj._dtype not in dtypes:
                    raise ValueError(
                        f"Unexpected type for argument {idx}, got {array_obj.dtype}, "
                        f"but expected one of {dtypes}!")
            return func(*arrays, **kwargs)
        return wrapper_allowed_types
    return decorator


def not_implemented_types(*expected_types):
    def decorator(func):
        @functools.wraps(func)
        def wrapper_allowed_types(*arrays, **kwargs):
            # type checks
            for idx, (array_obj, dtypes) in enumerate(
                    zip(arrays, expected_types)):
                if array_obj is not None and array_obj._dtype in dtypes:
                    raise NotImplementedError(
                        f"Operator not implemented for type \"{array_obj.dtype}\" "
                        f"(argument {idx})")
            return func(*arrays, **kwargs)
        return wrapper_allowed_types
    return decorator


def shapes_match_exactly(func):
    def wrapper_shapes_match_exactly(*array_objs, **kwargs):
        shapes = [o.shape for o in array_objs]
        if len(shapes) < 2:
            return func(*array_objs, **kwargs)
            # raise ValueError(f"Expected at least two arrays, but got {len(shapes)}")
        if not all(len(o.shape) == len(shapes[0]) for o in array_objs):
            raise ValueError(f"Not all shapes matched {shapes}")
        reference_shape = shapes[0]
        for shape in shapes:
            for s1, s2 in zip(reference_shape, shape):
                if s1 != s2:
                    raise ValueError("Shape mismatch")
        return func(*array_objs, **kwargs)
    return wrapper_shapes_match_exactly


def types_match_exactly(func):
    def wrapper_types_match_exactly(*array_objs, **kwargs):
        array_objs_filter = filter(
            lambda a: isinstance(a, array.Array),
            array_objs)
        if any(map(lambda a: a.dtype != array_objs[0].dtype, array_objs_filter)) \
           and STRICT_MODE:
            raise ValueError("Expected all arrays to be of the same type")
        return func(*array_objs, **kwargs)
    return wrapper_types_match_exactly


def check_input_shape_matmul(func):
    def wrapper_check_input_shape_matmul(*array_objs_and_inputs, **kwargs):
        A, B = array_objs_and_inputs[:2]
        a_shape = A.shape
        b_shape = B.shape
        if len(a_shape) < 2:
            if len(a_shape) == 0:
                raise ValueError(
                    "Matrix multiplication is not valid with scalars")
            else:
                a_shape = (1, *a_shape)
        if len(b_shape) < 2:
            if len(b_shape) == 0:
                raise ValueError(
                    "Matrix multiplication is not valid with scalars")
            else:
                b_shape = (*b_shape, 1)
        if len(a_shape) == 2 and len(b_shape) == 2:
            # (n,k),(k,m)->(n,m)
            n, ka = a_shape
            kb, m = b_shape
            if (ka != kb):
                raise ValueError(
                    f"Matrix multiplication invalid. Number of rows of A ({a_shape}) "
                    f"does not match number of columns of B ({b_shape})")
        else:
            n, ka = a_shape[-2:]
            kb, m = b_shape[-2:]
            if (ka != kb):
                raise ValueError(
                    f"Matrix multiplication invalid. Number of rows of A ({A.shape}) "
                    f"does not match number of columns of B ({B.shape})")
            rest_a = a_shape[:-2]
            rest_b = b_shape[:-2]
            if len(rest_a) != len(rest_b):
                raise ValueError("")
            for rest_a_i, rest_b_i in zip(rest_a, rest_b):
                if rest_a_i != rest_b_i:
                    raise ValueError("")

        return func(*array_objs_and_inputs, **kwargs)
    return wrapper_check_input_shape_matmul


def check_input_shape_gemm(func):
    def check_input_shape_gemm(*array_objs_and_inputs, **kwargs):
        A, B, C = array_objs_and_inputs
        transA = kwargs["transA"]
        transB = kwargs["transB"]
        a_shape = tuple(reversed(A.shape)) if transA else A.shape
        b_shape = tuple(reversed(B.shape)) if transB else B.shape
        # (n,k),(k,m)->(n,m)
        n, ka = a_shape
        kb, m = b_shape
        if (ka != kb):
            raise ValueError(
                f"Shape of A {a_shape} is incompatiable with shape of B {b_shape} "
                f"(after transposing)")

        # unidirectional broadcasting
        if C is not None and C.ndims > 0:
            output_shape = (n, m)
            c_shape = C.shape
            # Tensor A and B both have exactly the same shape.
            if c_shape == output_shape:
                pass
            # Tensor A and B all have the same number of dimensions
            # and the length of each dimensions is either a common
            # length or B's length is 1.
            elif len(c_shape) == len(output_shape):
                for s1, s2 in zip(output_shape, c_shape):
                    if s1 != s2 and s2 != 1:
                        raise ValueError(
                            f"C has incompatible shape {c_shape} with matrix "
                            f"multiplication result shape {output_shape}")
            # Tensor B has too few dimensions, and B can have
            # its shapes prepended with a dimension of length
            # 1 to satisfy property 2.
            elif len(c_shape) < len(output_shape):
                pad = len(output_shape) - len(c_shape)
                c_shape = (*(1,)*pad, *c_shape)
                for s1, s2 in zip(output_shape, c_shape):
                    if s1 != s2 and s2 != 1:
                        raise ValueError(
                            f"C has incompatible shape {c_shape} with matrix "
                            f"multiplication result shape {output_shape}")
            else:
                raise ValueError(
                    f"C has invalid shape {c_shape}")

        return func(*array_objs_and_inputs, **kwargs)
    return check_input_shape_gemm


def array_is_square_matrix(func):
    def wrapper_array_is_square_matrix(*array_objs_and_inputs, **kwargs):
        input_shape = array_objs_and_inputs[0].shape
        if len(input_shape) < 2:
            raise ValueError(
                "1-dimensional array given. Array must be at least two-dimensional")
        else:
            m, n = input_shape[-2:]
            if m != n and m < 2:
                raise ValueError(
                    "Last 2 dimensions of the array must be square")
        return func(*array_objs_and_inputs, **kwargs)
    return wrapper_array_is_square_matrix


def output_type_is_argn_position(arg_position):
    def wrapper_output_type_is_arg_position(
            return_array, *input_arrays_and_args, **kwargs):
        return_array._dtype = input_arrays_and_args[arg_position]
    return wrapper_output_type_is_arg_position


def propagate_shape_from_argn_position(arg_position):
    def wrapper_propagate_shape_from_argn_position(
            return_array, *input_arrays_and_args, **kwargs):
        return_array._dims = input_arrays_and_args[arg_position].shape
    return wrapper_propagate_shape_from_argn_position


def propagate_shape_matmul():
    def wrapper_propagate_shape_matmul(
            return_array, *input_arrays_and_args, **kwargs):
        A, B = input_arrays_and_args[:2]
        a_shape = A.shape
        b_shape = B.shape
        if len(a_shape) < 2:
            if len(a_shape) == 0:
                raise ValueError(
                    "Matrix multiplication is not valid with scalars")
            else:
                a_shape = (1, *a_shape)
        if len(b_shape) < 2:
            if len(b_shape) == 0:
                raise ValueError(
                    "Matrix multiplication is not valid with scalars")
            else:
                b_shape = (*b_shape, 1)
        if len(a_shape) == 2 and len(b_shape) == 2:
            # (n,k),(k,m)->(n,m)
            n, ka = a_shape
            kb, m = b_shape
            return_array._dims = (n, m)
        else:
            n, ka = a_shape[-2:]
            kb, m = b_shape[-2:]
            return_array._dims = (*a_shape[:-2], n, m)

        if A.ndims == 1:
            shape = list(return_array._dims)
            shape.pop(-2)
            return_array._dims = (*shape,)

        if B.ndims == 1:
            return_array._dims = return_array._dims[:-1]

    return wrapper_propagate_shape_matmul


def concatenate_shapes(axis_kwarg):
    def wrapper_concatenate_shapes(
            return_array, *input_arrays_and_args, **kwargs):
        axis = kwargs[axis_kwarg]
        # check all shapes are the same size
        shape_sizes = list(map(lambda a: len(a.shape), input_arrays_and_args))
        if any(
            map(
                lambda s: s != len(input_arrays_and_args[0].shape),
                shape_sizes)):
            raise ValueError(
                "All shapes must have the same number of dimensions")
        if axis < 0:
            if abs(axis) > shape_sizes[0]:
                raise ValueError(
                    f"Axis must be in the range [-{shape_sizes[0]}, "
                    f"{shape_sizes[0]-1}]")
            else:
                axis = len(input_arrays_and_args[0].shape) - 1

        new_shape = ()
        shapes = [a.shape for a in input_arrays_and_args]
        for shape_idx, shapes_at_idx in enumerate(zip(*shapes)):
            if shape_idx == axis:
                # concatenation axis
                new_shape = (*new_shape, sum(shapes_at_idx))
            elif any(map(lambda s: s != shapes_at_idx[0], shapes_at_idx)):
                raise ValueError("Dimension mismatch on axis {shape_idx}")
            else:
                new_shape = (*new_shape, shapes_at_idx[0])
        return_array._dims = new_shape
    return wrapper_concatenate_shapes


def reduction_axis(*, axis_arg=None, axis_kwarg=None):
    if not (bool(axis_kwarg) ^ bool(axis_arg)):
        raise ValueError("Either specify axis argument position or keyword. "
                         "This is an internal error, please file a bug.")

    def wrapper_reduction_axis(return_array, *input_arrays_and_args, **kwargs):
        if axis_kwarg:
            axis = kwargs[axis_kwarg]
        else:
            axis = input_arrays_and_args[axis_arg]
        shape = input_arrays_and_args[0].shape
        if axis < -len(shape) or axis > len(shape) - 1:
            raise ValueError(
                f"Axis must be in the range [-{len(shape)}, {len(shape)-1}]")
        new_shape_list = list(shape)
        new_shape_list.pop(axis)
        return_array._dims = tuple(new_shape_list)

    return wrapper_reduction_axis


def output_shape_from_einsum(equation_kwarg: str):
    def wrapper_output_shape_from_einsum(
            return_array, *input_arrays_and_args, **kwargs):
        if equation_kwarg not in kwargs:
            raise ValueError("Equation string not passed as kwarg. "
                             "Please file a bug report")
        equation = kwargs[equation_kwarg]
        input_shapes = [a.shape for a in input_arrays_and_args]
        input_axis_labels, output_axis_labels = einsum_parse_and_resolve_equation(
            equation, input_shapes)

        output_shape = ()

        for output_axis_label in output_axis_labels:
            for input_axis_label, input_shape in zip(
                    input_axis_labels, input_shapes):
                pos = input_axis_label.find(output_axis_label)
                if pos != -1:
                    output_shape = (*output_shape, input_shape[pos])
                    break

        return_array._dims = output_shape

    return wrapper_output_shape_from_einsum


def output_type(dtype: np.dtype):
    def wrapper_output_type(return_array, *input_arrays_and_args, **kwargs):
        return_array._dtype = dtype
    return wrapper_output_type


def allow_broadcasting(return_array, *input_arrays_and_args, **kwargs):
    array_shapes = [a.shape for a in input_arrays_and_args]
    if all(map(lambda a: a == array_shapes[0], array_shapes)):
        # no broadcasting needed
        shape = array_shapes[0]
    elif all(map(lambda a: len(a) == len(array_shapes[0]), array_shapes)):
        # The tensors all have the same number of dimensions and the length of each
        # dimensions is either a common length or 1.
        shape = ()
        for idx, dims_at_idx in enumerate(zip(*array_shapes)):
            s0 = dims_at_idx[0]
            s_set = set()
            for s in dims_at_idx:
                if s != s0 and s != 1:
                    raise ValueError(
                        f"Broadcasting not possible with shapes {array_shapes}")
                else:
                    s_set.add(s)
            shape = (*shape, max(s_set))
    else:
        # The tensors that have too few dimensions can have their shapes
        # prepended with a dimension of length 1 to satisfy property 2.
        max_array_shape = max(map(lambda a: len(a), array_shapes))
        def prepend_ones_to_shape(s, n): return (*(1,)*n, *s)
        array_shapes = [
            prepend_ones_to_shape(a, max_array_shape - len(a))
            if len(a) < max_array_shape else a for a in array_shapes]
        shape = ()
        for idx, dims_at_idx in enumerate(zip(*array_shapes)):
            s0 = dims_at_idx[0]
            s_set = set()
            for s in dims_at_idx:
                if s != s0 and s != 1:
                    raise ValueError(
                        f"Broadcasting not possible with shapes {array_shapes}")
                else:
                    s_set.add(s)
            shape = (*shape, max(s_set))

    return_array._dims = shape


def determinant_output_shape(return_array, *input_arrays_and_args, **kwargs):
    input_shape = input_arrays_and_args[0].shape
    if len(input_shape) == 2:
        return_array._dims = ()
    else:
        # we know at this point that the array is valid
        return_array._dims = input_shape[:-2]


def broadcast_to(shape_: Tuple[int]):
    def wrapper_broadcast_to(return_array, *input_arrays_and_args, **kwargs):
        input_array_shape = input_arrays_and_args[0].shape
        if len(input_array_shape) < len(shape_):
            pad = len(shape_) - len(input_array_shape)
            input_array_shape = (*(1,)*pad, *input_array_shape)
            shape = shape_
        elif len(input_array_shape) > len(shape_):
            pad = len(input_array_shape) - len(shape_)
            shape = (*(1,)*pad, *shape_)
        else:
            shape = shape_
        output_shape = ()
        for s1, s2 in zip(reversed(input_array_shape), reversed(shape)):
            if s1 != s2 and (1 not in [s1, s2]):
                raise ValueError(
                    f"Cannot broadcast array with shape {input_array_shape} to "
                    f"{shape}")
            output_shape = (max(s1, s2), *output_shape)

        return_array._dims = output_shape

    return wrapper_broadcast_to


def flatten_shape(axis: int):
    def wrapper_flatten_shape(return_array, *input_arrays_and_args, **kwargs):
        array_shape = input_arrays_and_args[0].shape
        if axis < -len(array_shape) or axis > len(array_shape) - 1:
            raise ValueError(
                f"Axis must be in the range [-{len(array_shape)}, "
                f"{len(array_shape)-1}]")

        def merge_axis(s): return reduce(operator.mul, s, 1)

        if axis == 0:
            output_shape = (1, merge_axis(array_shape))
        else:
            a = len(array_shape) + axis if axis < 0 else axis
            output_shape = (
                merge_axis(array_shape[: a]),
                merge_axis(array_shape[a:]))

        return_array._dims = output_shape

    return wrapper_flatten_shape


def gather_check(axis_: int):
    def wrapper_gather_check(return_array, *input_arrays_and_args, **kwargs):
        x = input_arrays_and_args[0]
        x_shape = x.shape
        indices = input_arrays_and_args[1]

        if x.ndims == 0:
            raise ValueError("Array cannot be a scalar")

        if axis_ < -len(x_shape) or axis_ > len(x_shape) - 1:
            raise ValueError(
                f"Axis must be in the range [-{len(x_shape)}, {len(x_shape)-1}]")
        axis = len(x_shape) + axis_ if axis_ < 0 else axis_

        if indices._ort_value is not None:
            # TODO: in this case we can perform a check of input values
            # it would be possible to force evaluation, but it might not be worth it?
            pass

        output_shape = (*x_shape[:axis], *indices.shape, *x_shape[axis+1:])

        return_array._dims = output_shape

    return wrapper_gather_check


def propagate_shape_gemm(return_array, *input_arrays_and_args, **kwargs):
    A, B, C = input_arrays_and_args
    transA = kwargs["transA"]
    transB = kwargs["transB"]
    a_shape = tuple(reversed(A.shape)) if transA else A.shape
    b_shape = tuple(reversed(B.shape)) if transB else B.shape
    # (n,k),(k,m)->(n,m)
    n, _ = a_shape
    _, m = b_shape
    return_array._dims = (n, m)


def propagate_shape_pool(return_array, *input_arrays_and_args, **kwargs):
    N, C, H, W = input_arrays_and_args[0].shape
    return_array._dims = (N, C, 1, 1)


def reduce_axis(axes: Union["array.Array", None], keepdims: bool):
    def output_reduce_axis(return_array, *input_arrays_and_args, **kwargs):
        if axes is None:
            # noop (this only happens in ReduceSum when noop_with_empty_axes is True
            # and axis is not specified)
            return

        x_shape = input_arrays_and_args[0].shape
        axes_np = axes.numpy()
        if any(map(lambda axis: axis < -len(x_shape) or axis > len(x_shape) - 1,
                   axes_np)):
            raise ValueError(
                f"Axis must be in the range [-{len(x_shape)}, {len(x_shape)-1}]")

        reduction_axes = [len(x_shape) + axis if axis <
                          0 else axis for axis in axes_np]

        output_shape = list(x_shape)

        if keepdims:
            for axis in reduction_axes:
                output_shape[axis] = 1
        else:
            for axis in reduction_axes:
                output_shape[axis] = None

        return_array._dims = tuple(
            filter(lambda s: s is not None, output_shape))

    return output_reduce_axis


def output_checks_and_inference(*output_checks):
    def decorator(func):
        @ functools.wraps(func)
        def wrapper_output_checks_and_inference(
                *input_arrays_and_args, **kwargs):
            return_array = func(*input_arrays_and_args, **kwargs)
            for output_check in output_checks:
                output_check(return_array, *input_arrays_and_args, **kwargs)
            return return_array
        return wrapper_output_checks_and_inference
    return decorator
