from typing import Iterable as IterableType

def shapes_match(rhs: IterableType[int], lhs: IterableType[int]) -> bool:
    
    if len(rhs) != len(lhs):
        return False

    for s1, s2 in zip(rhs, lhs):
        if s1 != s2:
            return False

    return True