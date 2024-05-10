"""
Permutations of `range(n) == [0, 1, ..., n-1]`, either as tuples in one-line
notation, or as indices into a list of $S_n$ sorted by Bruhat length.

>>> from pprint import pp
>>> pp(get_indexed_perms(3))
IndexedPerms(n=3,
             perms=[(0, (0, 1, 2)),
                    (1, (0, 2, 1)),
                    (1, (1, 0, 2)),
                    (2, (1, 2, 0)),
                    (2, (2, 0, 1)),
                    (3, (2, 1, 0))],
             perm_index={(0, 1, 2): 0,
                         (0, 2, 1): 1,
                         (1, 0, 2): 2,
                         (1, 2, 0): 3,
                         (2, 0, 1): 4,
                         (2, 1, 0): 5},
             inverse=[0, 1, 2, 4, 3, 5],
             mult=[[2, 1], [4, 0], [0, 3], [5, 2], [1, 5], [3, 4]])
"""

__all__ = [
    "BruhatLength", "PermOneLine", "PermIndex", "IndexedPerms",
    "get_perms", "get_inverse", "get_mult", "get_flip", "get_indexed_perms",
]

from dataclasses import dataclass
from typing import NewType

# the number of inversions in a permutation
BruhatLength = NewType('BruhatLength', int)

# a permutation of range(n), in one-line notation
PermOneLine = NewType('PermOneLine', tuple[int, ...])

# an index into the list returned by `get_perms`
PermIndex = NewType('PermIndex', int)


@dataclass
class IndexedPerms:
    """Precomputed values using PermIndex."""
    # the $n$ in $S_n$
    n: int

    # all permutations
    perms: list[tuple[BruhatLength, PermOneLine]]  # key: PermIndex
    perm_index: dict[PermOneLine, PermIndex]

    # pre-computed permutation group operations
    inverse: list[PermIndex]  # key: PermIndex
    mult: list[list[PermIndex]]  # key: PermIndex, $s_i$

    # allow unpacking
    def __iter__(self):
        return iter((
            self.n,
            self.perms,
            self.perm_index,
            self.inverse,
            self.mult,
        ))


def get_perms(n: int) -> list[tuple[BruhatLength, PermOneLine]]:
    """
    Return a list of all permutations of `range(n)`, sorted by Bruhat length.

    >>> from pprint import pp
    >>> pp(get_perms(3))
    [(0, (0, 1, 2)),
     (1, (0, 2, 1)),
     (1, (1, 0, 2)),
     (2, (1, 2, 0)),
     (2, (2, 0, 1)),
     (3, (2, 1, 0))]
    """
    _perm = list(range(n))  # mutated in the loop
    def loop(j=0, bru_len=0):
        """
        Variable depth nested loop.

        At depth `j`, we apply simple reflections to make the value `j` travel
        towards the front of the list, to position `i <= j`.
        """
        if j < n:
            # case 1: `j` stays where it is
            yield from loop(j+1, bru_len)
            for i in reversed(range(j)):
                # case 2: `j` moves one more step towards the front using $s_i$
                _perm[i], _perm[i+1] = _perm[i+1], _perm[i]
                yield from loop(j+1, bru_len + (j-i))
            # restore the permutation to its initial state
            _perm[:j+1] = _perm[1 : j+1] + _perm[:1]
        else:
            yield bru_len, tuple(_perm)
    return sorted(loop())


def get_inverse(perm: PermOneLine) -> PermOneLine:
    """
    Return the inverse permutation.

    >>> get_inverse((1, 2, 0))
    (2, 0, 1)
    """
    return tuple(perm.index(i) for i in range(len(perm)))


def get_mult(perm: PermOneLine, i: int) -> PermOneLine:
    """
    Multiply `perm` by $s_i$ on the right.

    >>> get_mult((2, 1, 0), 0)
    (1, 2, 0)
    """
    _perm = list(perm)
    _perm[i], _perm[i+1] = _perm[i+1], _perm[i]
    return tuple(_perm)


def get_flip(perm: PermOneLine) -> PermOneLine:
    """
    Conjugate `perm` by the long word.

    >>> get_flip((1, 0, 2))
    (0, 2, 1)
    """
    n = len(perm) - 1
    return tuple(n-i for i in reversed(perm))


def get_indexed_perms(n: int) -> IndexedPerms:
    """Precompute values using PermIndex."""
    perms = get_perms(n)
    perm_index = {perm: i for i, (_, perm) in enumerate(perms)}
    inverse = [perm_index[get_inverse(p)] for _, p in perms]
    mult = [[perm_index[get_mult(p, i)] for i in range(n-1)] for _, p in perms]
    return IndexedPerms(n, perms, perm_index, inverse, mult)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
