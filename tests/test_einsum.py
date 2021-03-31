import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types
from onnxruntime_numpy.einsum_helper import einsum_parse_and_resolve_equation
import numpy as np
import pytest
from .utils import expect


def test_einsum_incorrect_format():

    shapes = [(1, 2, 3)]
    equation = "...ii <->...i"

    with pytest.raises(ValueError):
        _ = einsum_parse_and_resolve_equation(equation, shapes)


def test_einsum_too_many_input_args():

    shapes = [(1, 2, 3)]
    equation = "bij, bjk -> bik"

    with pytest.raises(ValueError):
        _ = einsum_parse_and_resolve_equation(equation, shapes)


def test_einsum_too_many_ellipsis():

    shapes = [(1, 2, 3)]
    equation = "...i...i ->...i"

    with pytest.raises(ValueError):
        _ = einsum_parse_and_resolve_equation(equation, shapes)


def test_einsum_ellipses_do_not_match():

    shapes = [(1, 2, 3)]
    equation = "...ijkl ->...i"

    with pytest.raises(ValueError):
        _ = einsum_parse_and_resolve_equation(equation, shapes)


def test_einsum_period_output_ellipsis():

    shapes = [(1, 2, 3)]
    equation = "....ii ->...i"

    with pytest.raises(ValueError):
        _ = einsum_parse_and_resolve_equation(equation, shapes)

    equation = "...ii ->....i"

    with pytest.raises(ValueError):
        _ = einsum_parse_and_resolve_equation(equation, shapes)


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_einsum_batch_diagonal(type_a):

    x = onp.array(np.arange(2*3*3).reshape(2, 3, 3), dtype=type_a)
    equation = "...ii ->...i"
    expected = onp.array([[0,  4,  8],
                          [9, 13, 17]], dtype=type_a)

    result = onp.einsum(x, equation=equation)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_einsum_batch_matmul(type_a):
    x = onp.array(np.arange(3*1*2).reshape((3, 1, 2)), dtype=type_a)
    y = onp.array(np.arange(3*2*3).reshape(3, 2, 3), dtype=type_a)

    equation = "bij, bjk -> bik"
    expected = onp.array([[[3,   4,   5]],
                          [[39,  44,  49]],
                          [[123, 132, 141]]], dtype=type_a)

    result = onp.einsum(x, y, equation=equation)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_einsum_inner_prod(type_a):
    x = onp.array([1, 2, 3, 4, 5], dtype=type_a)
    y = onp.array([1, 2, 3, 4, 5], dtype=type_a)

    equation = "i,i"
    expected = onp.array(55, dtype=type_a)

    result = onp.einsum(x, y, equation=equation)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_einsum_sum(type_a):
    x = onp.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=type_a)

    equation = "ij->i"
    expected = onp.array([15, 15], dtype=type_a)

    result = onp.einsum(x, equation=equation)

    expect(result.numpy(), expected.numpy())


@pytest.mark.parametrize("type_a", [*float_types, np.int32, np.int64])
def test_einsum_transpose(type_a):
    x = onp.array([[1, 2, 3], [1, 2, 3]], dtype=type_a)

    equation = "ij->ji"
    expected = onp.array([[1, 1], [2, 2], [3, 3]], dtype=type_a)

    result = onp.einsum(x, equation=equation)

    expect(result.numpy(), expected.numpy())
