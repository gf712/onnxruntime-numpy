from typing import Collection
from abc import ABC, abstractmethod


class Shape(ABC):
    def __init__(self, shape: Collection[int]):
        self._shape = shape

    @property
    def ndims(self):
        return len(self._shape)

    def __eq__(self, other):
        return self.equals(other)

    @abstractmethod
    def equals(self, other):
        pass


class StaticShape(Shape):
    def __init__(self, shape: Collection[int]):
        super().__init__(shape)

    def equals(self, other):
        if self.ndims != other.ndis:
            return False

        for s1, s2 in zip(self._shape, other._shape):
            if s1 != s2:
                return False

        return True

    def __repr__(self):
        return f"StaticShape({self._shape})"


class DynamicShape(Shape):
    def __init__(self, shape: Collection[int]):
        super().__init__(shape)

    def equals(self, other):
        if self.ndims != other.ndis:
            return False

        for s1, s2 in zip(self._shape, other._shape):
            if s1 != s2 and -1 not in [s1, s2]:
                return False

        return True

    def __repr__(self):
        return f"DynamicShape({['?' if s==-1 else s for s in self._shape]})"


def shapes_match(rhs: Collection[int], lhs: Collection[int]) -> bool:

    if len(rhs) != len(lhs):
        return False

    for s1, s2 in zip(rhs, lhs):
        if s1 != s2:
            return False

    return True
