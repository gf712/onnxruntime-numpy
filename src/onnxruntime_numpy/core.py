from .shapes import DynamicShape, StaticShape, as_shape, ShapeLike, Shape
from .exceptions import InternalException
from .evaluator import LazyEvaluator
from .types import ort_to_numpy
from . import array

import weakref
from weakref import ReferenceType
import uuid
import onnxruntime
import numpy as np
from typing import Optional, List, Any, Dict, Tuple, DefaultDict, Set
from collections import defaultdict


class InternalArray:
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

    def _eval(self):
        if self._ort_value is None:
            self._ort_value = self._evaluator.evaluate(self)
            self._dims = StaticShape(*self._ort_value.shape())

    def ort_value(self) -> onnxruntime.OrtValue:
        if self._ort_value is None:
            self._eval()
        return self._ort_value

    def numpy(self) -> np.ndarray:
        return self.ort_value().numpy()

    def values(self) -> List[Any]:
        return self.numpy().tolist()

    @property
    def shape(self) -> Shape:
        if self._dims is None:
            raise InternalException("Unevaluated shape.. This is a bug!")
        return self._dims  # type: ignore

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            raise InternalException("Unevaluated dtype.. This is a bug!")
        return self._dtype


def _array_finalizer_factory(x: "array.Array") -> ReferenceType:

    g = x._internal_array._evaluator._graph
    output_name = x._internal_array._evaluator._output_node
    parent_name = x._internal_array._evaluator._parent_node
    internal_name = x._internal_array._internal_name

    def _finalizer():
        with g._lock:
            n = g.nodes[output_name]
            n["node"]._can_be_dropped(True)
        # print(
        #     f"calling finalizer for {internal_name}, output node {output_name}, parent node: {parent_name}, graph {g}")

    f = weakref.finalize(x, _finalizer)
    f.atexit = False

    return weakref.ref(x)


def _fetch_array(x: ReferenceType) -> "array.Array":
    value = x()
    if value is None:
        raise InternalException("Array no longer available")
    return value


def as_array(x: InternalArray) -> "array.Array":
    return array.Array._from_internal_array(x)


class IntermediateResultCache:

    def __init__(self):
        self._cache: Dict[str, InternalArray] = dict()

    def merge(self, other):
        self._cache = {**self._cache, **other._cache}

    def add_entry(self, edge_name: str, array: InternalArray):
        self._cache[edge_name] = array

    def get_all_cache_tensor_mappings(self) -> Dict[str,
                                                    InternalArray]:
        return self._cache

    def empty(self) -> bool:
        return len(self._cache) == 0


class ArrayNodeLookupTable:
    def __init__(self):
        self._input_table: Dict[str,
                                List[Tuple[ReferenceType, InternalArray]]] = {}
        self._output_table: Dict[str,
                                 List[Tuple[ReferenceType, InternalArray]]] = {}

    def add_input(
            self, node_name: str, arrays: List["array.Array"]):
        self._input_table[node_name] = [(_array_finalizer_factory(
            a), a._internal_array) if a is not None else None for a in arrays]

    def add_output(
            self, node_name: str, arrays: List["array.Array"]):
        self._output_table[node_name] = [(_array_finalizer_factory(
            a), a._internal_array) if a is not None else None for a in arrays]

    def get_input_map(self):
        return self._input_table

    def get_output_map(self):
        return self._output_table

    def update(self, other):
        self._input_table = {**self._input_table, **other._input_table}
        self._output_table = {**self._output_table, **other._output_table}
