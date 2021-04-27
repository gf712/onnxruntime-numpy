import numpy as np
from . import array
from .ops_utils import (allowed_types, not_implemented_types,
                        nary_operator, initializer_operator)
from .types import numpy_to_onnx, float_types
from .shapes import DynamicShape, ShapeLike, as_shape
from typing import Optional


def multinomial(
        x: "array.Array", dtype: np.dtype = np.int32, sample_size: int = 1,
        seed: Optional[float] = None):

    if dtype not in [np.int32, np.int64]:
        raise TypeError(
            f"Expected output type to be either int32 or int64, but got {dtype}")

    if x.ndims != 2:
        raise ValueError(
            f"Expected input array to have two dimensions, but got {x.ndims} "
            "dimensions")

    batch_size, _ = x.shape  # type: ignore

    @allowed_types(float_types)
    @not_implemented_types([np.float64])
    def multinomial_helper(
            x: "array.Array", dtype: np.dtype, sample_size: int,
            seed: Optional[float]):
        if seed is not None:
            seed = float(seed)
        result = nary_operator(
            "Multinomial", x, dtype=numpy_to_onnx(np.dtype(dtype)),
            sample_size=sample_size, seed=seed)
        result._dtype = dtype
        result._dims = DynamicShape(batch_size, 1)
        return result
    return multinomial_helper(
        x, dtype=dtype, sample_size=sample_size, seed=seed)


def normal(
        shape: ShapeLike,
        mean: float = 0.0, scale: float = 1.0, dtype: np.dtype = np.float32,
        seed: Optional[float] = None):
    if seed is not None:
        seed = float(seed)

    @allowed_types(float_types)
    def normal_helper(
            shape: ShapeLike,
            mean: float, scale: float, dtype: np.dtype, seed: Optional[float]):
        return initializer_operator(
            "RandomNormal", array_shape=as_shape(shape),
            array_dtype=np.dtype(dtype),
            shape=shape, dtype=numpy_to_onnx(np.dtype(dtype)),
            mean=mean, scale=scale, seed=seed)

    return normal_helper(
        shape=shape, mean=mean, scale=scale, dtype=dtype, seed=seed)


def normal_like(x="array.Array", mean: float = 0.0, scale: float = 1.0,
                dtype: Optional[np.dtype] = None, seed: Optional[float] = None):
    if seed is not None:
        seed = float(seed)

    @allowed_types(float_types)
    def normal_like_helper(
            x: "array.Array",
            mean: float, scale: float,
            dtype: Optional[np.dtype],
            seed: Optional[float]):
        result = nary_operator("RandomNormalLike", x, dtype=numpy_to_onnx(
            np.dtype(dtype)), mean=mean, scale=scale, seed=seed)
        result._dtype = dtype if dtype is not None else x.dtype
        result._dims = x.shape

        return result

    return normal_like_helper(x, mean=mean, scale=scale, dtype=dtype, seed=seed)


def uniform(
        shape: ShapeLike,
        low: float = 0.0, high: float = 1.0, dtype: np.dtype = np.float32,
        seed: Optional[float] = None):
    if seed is not None:
        seed = float(seed)

    @allowed_types(float_types)
    def uniform_helper(
            shape: ShapeLike,
            low: float, high: float, dtype: np.dtype, seed: Optional[float]):
        return initializer_operator(
            "RandomUniform", array_shape=as_shape(shape),
            array_dtype=np.dtype(dtype),
            shape=shape, dtype=numpy_to_onnx(np.dtype(dtype)),
            low=low, high=high, seed=seed)

    return uniform_helper(
        shape=shape, low=low, high=high, dtype=dtype, seed=seed)


def uniform_like(x="array.Array", low: float = 0.0, high: float = 1.0,
                 dtype: Optional[np.dtype] = None, seed: Optional[float] = None):
    if seed is not None:
        seed = float(seed)

    @allowed_types(float_types)
    def uniform_like_helper(
            x: "array.Array",
            low: float, high: float,
            dtype: Optional[np.dtype],
            seed: Optional[float]):
        result = nary_operator("RandomUniformLike", x, dtype=numpy_to_onnx(
            np.dtype(dtype)), low=low, high=high, seed=seed)
        result._dtype = dtype if dtype is not None else x.dtype
        result._dims = x.shape

        return result

    return uniform_like_helper(x, low=low, high=high, dtype=dtype, seed=seed)
