from typing import Collection, Union, List
from abc import ABC, abstractmethod
from . import array
from functools import reduce
from collections.abc import Iterable
import numpy as np


class Dimension(ABC):
    def __init__(self, dim: int):
        self._dim = dim

    def __eq__(self, other) -> bool:
        return self.equals(other)

    @abstractmethod
    def equals(self, other: Union["Dimension", int]) -> bool:
        pass

    @abstractmethod
    def hash(self) -> int:
        pass

    def is_static(self) -> bool:
        return self._dim != -1

    def __gt__(self, other) -> bool:
        return self._dim > other._dim

    def __hash__(self) -> int:
        return self.hash()

    def __int__(self) -> int:
        return self._dim


class DynamicDimension(Dimension):
    def __init__(self, dim: int):
        if dim < -1:
            raise ValueError("Dynamic dimension has to be greater or equal -1")
        super().__init__(dim)

    def equals(self, other: Union["Dimension", int]) -> bool:
        other_dim = other._dim if isinstance(other, Dimension) else other
        return self._dim == other_dim or -1 in [self._dim, other_dim]

    def hash(self) -> int:
        # too lazy to handle hash myself
        return hash(("DynamicDimension", self._dim))

    def __str__(self):
        return f"{'?' if self._dim == -1 else self._dim}"

    def __repr__(self):
        return f"DynamicDimension({self})"


class StaticDimension(Dimension):
    def __init__(self, dim: int):
        if dim < 0:
            raise ValueError(
                "Static dimension has to be known at compile time")
        super().__init__(dim)

    def equals(self, other: Union["Dimension", int]) -> bool:
        other_dim = other._dim if isinstance(other, Dimension) else other
        return self._dim == other_dim

    def hash(self) -> int:
        # too lazy to handle hash myself
        return hash(("StaticDimension", self._dim))

    def __str__(self):
        return f"{self._dim}"

    def __repr__(self):
        return f"StaticDimension({self})"


class Shape(ABC):
    def __init__(self, *shape: Dimension):
        self._shape = (*shape,)

    @property
    def ndims(self):
        return len(self._shape)

    def tolist(self) -> List[int]:
        return [int(s) for s in self._shape]

    def asarray(self) -> "array.Array":
        return array.array(self._shape, np.int64)

    def size(self) -> int:
        s = reduce(lambda lhs, rhs: lhs * int(rhs), self._shape, 1)
        return max(s, -1)

    def __getitem__(self, idx):
        slice_ = self._shape[idx]
        if not isinstance(slice_, Iterable):
            return slice_
        return DynamicShape(*(int(s) for s in slice_))

    def __iter__(self):
        yield from self._shape

    def __reversed__(self):
        return reversed(self._shape)

    def __eq__(self, other):
        return self.equals(other)

    def __len__(self) -> int:
        return len(self._shape)

    def __hash__(self) -> int:
        return hash(self._shape)

    @abstractmethod
    def can_be_static(self) -> bool:
        pass

    @abstractmethod
    def to_static(self) -> "StaticShape":
        pass

    @abstractmethod
    def to_dynamic(self) -> "DynamicShape":
        pass

    def equals(self, other):
        if len(self) != len(other):
            return False
        return all(s1.equals(s2) for s1, s2 in zip(self, other))


class StaticShape(Shape):
    def __init__(
            self, *shape: Union[Dimension, int]):
        super().__init__(*(StaticDimension(int(s)) for s in shape if s))

    @classmethod
    def from_shape(cls, shape: Shape) -> "StaticShape":
        return cls(*(int(s) for s in shape._shape))

    def can_be_static(self) -> bool:
        return True

    def to_static(self) -> "StaticShape":
        return StaticShape.from_shape(self)

    def to_dynamic(self) -> "DynamicShape":
        return DynamicShape.from_shape(self)

    def __repr__(self):
        return f"StaticShape({', '.join(str(x) for x in self._shape)})"


class DynamicShape(Shape):
    def __init__(self, *shape: Union[Dimension, int]):
        super().__init__(*(DynamicDimension(int(s)) for s in shape))

    @classmethod
    def from_shape(cls, shape: Shape) -> "DynamicShape":
        return cls(*(int(s) for s in shape._shape))

    def can_be_static(self) -> bool:
        return not any(int(d) == -1 for d in self._shape)

    def to_static(self) -> "StaticShape":
        if self.can_be_static():
            return StaticShape.from_shape(self)
        else:
            raise ValueError(
                f"Shape of {self} cannot be converted to StaticShape")

    def to_dynamic(self) -> "DynamicShape":
        return DynamicShape.from_shape(self)

    def __repr__(self):
        return f"DynamicShape({', '.join(str(x) for x in self._shape)})"


ShapeLike = Union[Shape, Collection[int], "array.Array"]


def as_shape(s: ShapeLike) -> Shape:
    if isinstance(s, Shape):
        return s
    elif isinstance(s, array.Array):
        if array.is_lazy(s):
            return DynamicShape(*(-1 for _ in range(s.ndims)))
        else:
            if len(s.shape) != 1:
                raise ValueError(
                    f"Shape has to be 1D array, but got {len(s.shape)}D array")
            return DynamicShape(*s.numpy())
    else:
        return DynamicShape(*s)


def strong_shape_comparisson(*shapes: Shape) -> bool:
    if not shapes[0].can_be_static():
        return False
    shape = StaticShape.from_shape(shapes[0])
    if len(shapes) > 1:
        for s in shapes[1:]:
            if not s.can_be_static():
                return False
            static_s = StaticShape.from_shape(s)
            if shape != static_s:
                return False

    return True


def weak_shape_comparisson(*shapes: Shape) -> bool:
    shape = DynamicShape.from_shape(shapes[0])

    if len(shapes) > 1:
        for s in shapes[1:]:
            static_s = DynamicShape.from_shape(s)
            if shape != static_s:
                return False

    return True


def shapes_match(rhs: Collection[int], lhs: Collection[int]) -> bool:

    if len(rhs) != len(lhs):
        return False

    for s1, s2 in zip(rhs, lhs):
        if s1 != s2:
            return False

    return True
