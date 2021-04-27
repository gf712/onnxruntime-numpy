from .evaluator import LazyEvaluator
from . import ops
from .types import numpy_to_ort, python_to_numpy, ort_to_numpy, all_types, AnyType
from .shapes import as_shape, Shape, ShapeLike, DynamicShape
from typing import Any, List, Union, Optional
from typing import Iterable as IterableType
from collections.abc import Iterable
import uuid
import numpy as np
import onnxruntime


class Array:
    def __init__(self, ort_value: onnxruntime.OrtValue = None,
                 dims: Optional[ShapeLike] = None, dtype: np.dtype = None, *,
                 evaluator: LazyEvaluator = None, internal_name: str = None):
        self._internal_name: str = str(
            uuid.uuid4()) if internal_name is None else internal_name  # FIXME
        self._ort_value: onnxruntime.OrtValue = ort_value
        if dims is None:
            self._dims = as_shape(tuple(ort_value.shape())
                                  ) if ort_value is not None else DynamicShape()
        else:
            self._dims = as_shape(dims)
        self._dtype = ort_to_numpy(ort_value.data_type(
        )) if ort_value is not None and dtype is None else dtype
        self._evaluator = evaluator if evaluator is not None else LazyEvaluator()
        self._treat_array_as_initializer = False
        if evaluator is None:
            self._initialize()

    def _initialize(self):
        if self._treat_array_as_initializer:
            self._evaluator.add_initializer(
                self._internal_name, self._dtype, self._dims, self._ort_value)
        else:
            self._evaluator.add_input(self)

    def _eval(self):
        if self._ort_value is None:
            self._ort_value = self._evaluator.evaluate(self)
            self._evaluator = LazyEvaluator()
            self._initialize()

    def ort_value(self) -> onnxruntime.OrtValue:
        if self._ort_value is None:
            self._eval()
        return self._ort_value

    def numpy(self) -> np.ndarray:
        return self.ort_value().numpy()

    def values(self) -> List[Any]:
        self._eval()
        if self._ort_value is None:
            return []
        return self._ort_value.numpy().tolist()

    @ property
    def shape(self) -> Shape:
        if self._dims is None:
            raise ValueError("Unevaluated shape.. This is a bug!")
        return self._dims  # type: ignore

    @ property
    def dtype(self) -> np.dtype:
        return self._dtype

    @ property
    def ndims(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape.size()

    def __getitem__(self, index) -> Any:
        return self.numpy()[index]

    def __int__(self) -> int:
        return int(self.numpy())

    def __float__(self) -> float:
        return float(self.numpy())

    def __setitem__(self, index, value):
        raise NotImplementedError(
            "Array value setter currently not implemented")

    def __add__(self, other: "Array") -> "Array":
        return ops.add(self, other)

    def __sub__(self, other: "Array") -> "Array":
        return ops.subtract(self, other)

    def __truediv__(self, other: "Array") -> "Array":
        return ops.divide(self, other)

    def __mul__(self, other: "Array") -> "Array":
        return ops.multiply(self, other)

    def __matmul__(self, other: "Array") -> "Array":
        return ops.matmul(self, other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Array):
            raise NotImplementedError(
                "Can only compare Array with another array")
        return ops.equal(self, other)

    def __gt__(self, other: "Array") -> "Array":
        return ops.greater(self, other)

    def __ge__(self, other: "Array") -> "Array":
        return ops.greater_equal(self, other)

    def __lt__(self, other: "Array") -> "Array":
        return ops.less(self, other)

    def __le__(self, other: "Array") -> "Array":
        return ops.less_equal(self, other)

    def __neg__(self) -> "Array":
        return ops.negative(self)

    def __pow__(self, other: Union["Array", int]) -> "Array":
        return ops.power(self, array(other))

    def __abs__(self) -> "Array":
        return ops.absolute(self)

    def item(self) -> AnyType:
        if len(
                self.shape) == 0 or (
                len(self.shape) == 1 and self.shape[0] == 1):
            return self.ort_value().numpy().item()
        else:
            raise ValueError(
                "can only convert an array of size 1 to a Python scalar")

    def __repr__(self) -> str:
        return self.numpy().__repr__()

    def __hash__(self):
        return hash(self._internal_name)

    def __iter__(self) -> Any:
        self._eval()
        yield from self.numpy()


def is_lazy(x: Array) -> bool:
    return x._ort_value is None


def merge_types(*types: IterableType[type]) -> type:
    type_set = set(*types)
    # if all the same don't neeed to do anything else
    if len(type_set) == 1:
        return next(iter(type_set))
    # else we check for float, int bool
    elif float in type_set:
        return float
    elif int in type_set:
        return int
    elif bool in type_set:
        return bool
    # if none worked then this type is invalid
    else:
        raise ValueError(
            f"Could not infer onnx type from python types {type_set}")


def infer_dtype_from_array(array: IterableType[Any]) -> int:
    def infer_dtype_from_array_helper(array: IterableType[Any]) -> type:
        if isinstance(array, Iterable):
            return merge_types(infer_dtype_from_array_helper(a) for a in array)
        else:
            return type(array)
    return python_to_numpy(infer_dtype_from_array_helper(array))


def array(values, dtype: np.dtype = None) -> Array:
    # it's already an Array
    if isinstance(values, Array):
        if dtype is None or values.dtype == dtype:
            return values
        else:
            # do cast
            raise NotImplementedError("")
            pass

    if not isinstance(values, Iterable) and dtype is None:
        if type(values) in all_types:
            dtype = all_types[all_types.index(type(values))]
        else:
            dtype = python_to_numpy(type(values))

    if dtype is None:
        if isinstance(values, np.ndarray):
            dtype = values.dtype
        else:
            # by default an empty array without a type is a float32
            dtype = numpy_to_ort(np.dtype(np.float32)) if len(
                values) == 0 else infer_dtype_from_array(values)

    if isinstance(values, np.ndarray):
        np_array = values
        np_array = np_array.astype(dtype)
    else:
        np_array = np.array(values, dtype=dtype)

    return Array(onnxruntime.OrtValue.ortvalue_from_numpy(np_array))
