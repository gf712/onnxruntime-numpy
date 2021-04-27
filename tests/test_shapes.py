from onnxruntime_numpy.shapes import (
    StaticDimension, DynamicDimension, StaticShape, DynamicShape,
    strong_shape_comparisson, weak_shape_comparisson, as_shape)
import pytest
import numpy as np
import onnxruntime_numpy as onp


def test_compare_static_dimension():
    d1 = StaticDimension(1)
    d2 = StaticDimension(1)

    assert d1 == d2


def test_compare_dynamic_dimension():
    d1 = DynamicDimension(1)
    d2 = DynamicDimension(-1)

    assert d1 == d2


def test_invalid_static_dimension():
    with pytest.raises(ValueError):
        _ = StaticDimension(-1)
        _ = StaticDimension(-100)


def test_invalid_dynamic_dimension():
    with pytest.raises(ValueError):
        _ = DynamicDimension(-2)


def test_compare_static_shape():
    d1 = StaticShape(0, 1, 2, 3)
    d2 = StaticShape(0, 1, 2, 3)

    assert d1 == d2

    d2 = StaticShape(1, 1, 2, 3)

    assert d1 != d2

    d2 = StaticShape(0, 1, 2, 3, 2)

    assert d1 != d2


def test_invalid_static_shape():
    with pytest.raises(ValueError):
        _ = StaticShape(-1, 1, 2)
        _ = StaticShape(-2, 1, 2)


def test_invalid_dynamic_shape():
    with pytest.raises(ValueError):
        _ = DynamicShape(-2, 1, 2)


def test_compare_dynamic_shape():
    d1 = DynamicShape(0, -1, 2, 3)
    d2 = DynamicShape(0, 1, 2, 3)

    assert d1 == d2

    d2 = DynamicShape(1, 1, 2, 3)

    assert d1 != d2

    d2 = DynamicShape(0, 1, 2, 3, 2)

    assert d1 != d2


def test_static_to_dynamic_shape():
    d1 = StaticShape(1, 2, 3)
    d2 = d1.to_dynamic()
    expected = DynamicShape(1, 2, 3)

    assert d2 == expected


def test_dynamic_to_static_shape():
    d1 = DynamicShape(1, 2, 3)
    d2 = d1.to_static()
    expected = StaticShape(1, 2, 3)

    assert d2 == expected

    with pytest.raises(ValueError):
        d1 = DynamicShape(-1, 1, 2)
        _ = d1.to_static()


def test_strong_shape_comparisson():
    d1 = DynamicShape(1, 2, 3)
    d2 = StaticShape(1, 2, 3)

    assert strong_shape_comparisson(d1, d2)

    d3 = DynamicShape(-1, 2, 3)

    assert not strong_shape_comparisson(d1, d2, d3)

    d3 = DynamicShape(1, 2, 3, 4)

    assert not strong_shape_comparisson(d1, d2, d3)


def test_weak_shape_comparisson():
    d1 = DynamicShape(1, 2, 3)
    d2 = StaticShape(1, 2, 3)

    assert weak_shape_comparisson(d1, d2)

    d3 = DynamicShape(-1, 2, 3)

    assert weak_shape_comparisson(d1, d2, d3)

    d3 = DynamicShape(1, 2, 3, 4)

    assert not strong_shape_comparisson(d1, d2, d3)


def test_as_shape_from_iterable():
    a = (1, 2, 3)
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)

    a = [1, 2, 3]
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)

    a = [-1, 2, 3]
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)


def test_as_shape_from_array():
    a = onp.array([1, 2, 3], np.int32)
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)

    a = onp.array([-1, 2, 3], np.int32)
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)


def test_as_shape_from_shape():
    a = StaticShape(1, 2, 3)
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)

    a = DynamicShape(-1, 2, 3)
    s = as_shape(a)
    assert s == DynamicShape(1, 2, 3)
