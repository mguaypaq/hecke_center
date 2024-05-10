"""
Cycle types and their Bruhat-minimum permutations.

A cycle type for a permutation in $S_n$ is represented as a tuple of length $n$,
where the $i$th entry is the number of cycles of length $i+1$.

For any permutation, the Bruhat length (which is the minimum length of a
factorization into _adjacent_ transpositions) is at least the reflection length
(which is the minimum length of a factorization into _any_ transpositions).
A permutation is Bruhat-minimum when the two lengths are equal. There exists at
least one Bruhat-minimum permutation of every cycle type.

>>> from pprint import pp
>>> from permutations import get_indexed_perms
>>> pp(get_indexed_cycle_types(get_indexed_perms(3)))
IndexedCycleTypes(cycle_types=[(3, 0, 0), (1, 1, 0), (0, 0, 1)],
                  cycle_type_index={(3, 0, 0): 0, (1, 1, 0): 1, (0, 0, 1): 2},
                  bru_min_perms=[[0], [1, 2], [3, 4]])
"""

__all__ = [
    "CycleType", "CycleTypeIndex", "IndexedCycleTypes",
    "get_cycle_type", "get_bru_min_perms", "get_indexed_cycle_types",
]

from dataclasses import dataclass
from typing import NewType

from permutations import PermOneLine, PermIndex, IndexedPerms

# a cycle type, where `cycle_type[i]` is the number of `i+1`-cycles
CycleType = NewType('CycleType', tuple[int, ...])

# an index into the list of cycle types
CycleTypeIndex = NewType('CycleTypeIndex', int)


@dataclass
class IndexedCycleTypes:
    """Precomputed values using CycleTypeIndex."""
    # all cycle types
    cycle_types: list[CycleType]  # key: CycleTypeIndex
    cycle_type_index: dict[CycleType, CycleTypeIndex]

    # Bruhat-minimum permutations for each cycle type
    bru_min_perms: list[list[PermIndex]]  # key: CycleTypeIndex

    # allow unpacking
    def __iter__(self):
        return iter((
            self.cycle_types,
            self.cycle_type_index,
            self.bru_min_perms,
        ))


def get_cycle_type(perm: PermOneLine) -> CycleType:
    """
    Return the cycle type of a permutation.

    >>> get_cycle_type((1, 2, 3, 0, 4, 5, 6))
    (3, 0, 0, 1, 0, 0, 0)
    """
    cycle_type = [0] * len(perm)
    edges = {i: j for i, j in enumerate(perm)}
    while edges:
        steps = 0
        i, j = edges.popitem()
        while j != i:
            steps += 1
            j = edges.pop(j)
        cycle_type[steps] += 1
    return tuple(cycle_type)


def get_bru_min_perms(n: int) -> dict[CycleType, list[PermOneLine]]:
    """
    Return the permutations with minimum Bruhat length within each cycle type.

    These are the permutations where each cycle:
    * permutes a consecutive subrange of `range(n)`; and
    * increases from min to max, followed by decreasing from max to min.

    So the supports of the cycles can be described exactly by an integer
    composition of `n`, and each cycle element between min and max can either
    be on the up side or the down side of its cycle.

    >>> from pprint import pp
    >>> pp(get_bru_min_perms(3))
    {(3, 0, 0): [(0, 1, 2)],
     (1, 1, 0): [(0, 2, 1), (1, 0, 2)],
     (0, 0, 1): [(1, 2, 0), (2, 0, 1)]}
    """
    _perm = [None] * n  # mutated in the loop
    _cycle_type = [0] * n  # mutated in the loop
    def loop(j=0, start=None, up=None, down=None):
        """
        Variable depth nested loop.

        At depth `j`, the current cycle being built (if any) has minimum
        element `start`, an up-chain ending at `up`, and a down chain starting
        from `down`. The loop body then decides whether `j` will be a fixed
        point, a min, a max, on the up-chain, or on the down-chain.
        """
        if j < n:
            if start is None:
                # case 1: `j` is a fixed point of `cur_perm`
                _perm[j] = j
                _cycle_type[0] += 1
                yield from loop(j+1)
                # restore the state
                _perm[j] = None
                _cycle_type[0] -= 1

                # case 2: `j` is the start of a cycle
                yield from loop(j+1, start=j, up=j, down=j)
                # the state is already restored
            else:
                # case 3: `j` is the end of a cycle
                _perm[up] = j
                _perm[j] = down
                _cycle_type[j-start] += 1
                yield from loop(j+1)
                # restore the state
                _perm[up] = None
                _perm[j] = None
                _cycle_type[j-start] -= 1

                # case 4: `j` is on the up side of the current cycle
                _perm[up] = j
                yield from loop(j+1, start=start, up=j, down=down)
                # restore the state
                _perm[up] = None

                # case 5: `j` is on the down side of the current cycle
                _perm[j] = down
                yield from loop(j+1, start=start, up=up, down=j)
                # restore the state
                _perm[j] = None
        else:
            # check that all cycles are finished
            if start is None:
                yield tuple(_cycle_type), tuple(_perm)

    # collect the permutations by cycle type
    result = {}
    for cycle_type, perm in loop():
        result.setdefault(cycle_type, []).append(perm)
    return result


def get_indexed_cycle_types(indexed_perms: IndexedPerms) -> IndexedCycleTypes:
    """Precompute values using CycleTypeIndex."""
    n = indexed_perms.n
    perm_index = indexed_perms.perm_index

    bru_min_dict = get_bru_min_perms(n)

    cycle_types = sorted(bru_min_dict.keys(), reverse=True)
    cycle_type_index = {ctype: i for i, ctype in enumerate(cycle_types)}

    bru_min_perms = [
        sorted(perm_index[p] for p in bru_min_dict[ctype])
        for ctype in cycle_types
    ]

    return IndexedCycleTypes(cycle_types, cycle_type_index, bru_min_perms)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
