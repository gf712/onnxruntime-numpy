import onnxruntime_numpy as onp
from onnxruntime_numpy.types import float_types
import numpy as np
import pytest
from .utils import expect


def gemm_reference_implementation(A, B, C=None, alpha=1., beta=1., transA=0,
                                  transB=0):
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_all_attribute(type_a):
    a = np.random.ranf([4, 3]).astype(type_a)
    b = np.random.ranf([5, 4]).astype(type_a)
    c = np.random.ranf([1, 5]).astype(type_a)
    expected = gemm_reference_implementation(
        a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a),
                      transA=True,
                      transB=True,
                      alpha=0.25,
                      beta=0.35)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_alpha(type_a):
    a = np.random.ranf([3, 5]).astype(type_a)
    b = np.random.ranf([5, 4]).astype(type_a)
    c = np.zeros([1, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c, alpha=0.5)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a),
                      alpha=0.5)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_beta(type_a):
    a = np.random.ranf([3, 5]).astype(type_a)
    b = np.random.ranf([5, 4]).astype(type_a)
    c = np.zeros([1, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c, beta=0.5)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a),
                      beta=0.5)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_default_matrix_bias(type_a):
    a = np.random.ranf([3, 6]).astype(type_a)
    b = np.random.ranf([6, 4]).astype(type_a)
    c = np.random.ranf([3, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_default_no_bias(type_a):
    a = np.random.ranf([2, 10]).astype(type_a)
    b = np.random.ranf([10, 3]).astype(type_a)
    expected = gemm_reference_implementation(a, b)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_default_scalar_bias(type_a):
    a = np.random.ranf([2, 10]).astype(type_a)
    b = np.random.ranf([10, 3]).astype(type_a)
    c = np.array(3.14).astype(type_a)
    expected = gemm_reference_implementation(a, b, c)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_default_single_elem_vector_bias(type_a):
    a = np.random.ranf([2, 10]).astype(type_a)
    b = np.random.ranf([10, 3]).astype(type_a)
    c = np.random.ranf([1]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_vector_bias(type_a):
    a = np.random.ranf([2, 10]).astype(type_a)
    b = np.random.ranf([10, 4]).astype(type_a)
    c = np.random.ranf([1, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_default_zero_bias(type_a):
    a = np.random.ranf([2, 10]).astype(type_a)
    b = np.random.ranf([10, 4]).astype(type_a)
    c = np.zeros([1, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a))

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_transposeA(type_a):
    a = np.random.ranf([10, 2]).astype(type_a)
    b = np.random.ranf([10, 4]).astype(type_a)
    c = np.zeros([1, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c, transA=1)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a),
                      transA=True)

    expect(expected, result.numpy())


@pytest.mark.parametrize("type_a", [*float_types])
def test_gemm_transposeB(type_a):
    a = np.random.ranf([2, 10]).astype(type_a)
    b = np.random.ranf([4, 10]).astype(type_a)
    c = np.zeros([1, 4]).astype(type_a)
    expected = gemm_reference_implementation(a, b, c, transB=1)

    result = onp.gemm(onp.array(a, dtype=type_a),
                      onp.array(b, dtype=type_a),
                      onp.array(c, dtype=type_a),
                      transB=True)

    expect(expected, result.numpy())
