"""
Coefficients of the Geck-Rouquier basis of the center of a Hecke algebra,
in terms of the natural basis of the entire Hecke algebra.

We're only dealing with type A (the symmetric group case), the ground ring
is Laurent polynomials in the variable $q$, and the normalization of the
natural basis is:
$$ (T_{s_i} - q)(T_{s_i} + 1) = 0 $$
where $s_i$ is the $i$th adjacent transposition, which swaps $i$ and $i+1$.

The Geck-Rouquier basis is indexed by cycle types. The basis element for a
given cycle type is uniquely determined by having coefficient 1 (as a constant
polynomial in $q$) on the Bruhat-minimum permutations of its cycle type, and
coefficient 0 (again constant) on the Bruhat-minimum permutations of other
cycle types.

Elements of the center typically have many equal coefficients in the natural
basis. We compute the coarsest set partition of the symmetric group such that
all elements of the center of the Hecke algebra have equal coefficients on each
block of the partition. For efficiency, we only store one coefficient per block
of this set partition.

>>> from pprint import pp
>>> indexed_perms = get_indexed_perms(3)
>>> indexed_cycle_types = get_indexed_cycle_types(indexed_perms)
>>> gr_coeffs = get_gr_coeffs(indexed_perms, indexed_cycle_types)
>>> pp(gr_coeffs.indexed_blocks)
IndexedBlocks(blocks=[[0], [1, 2], [3, 4], [5]], block_index=[0, 1, 1, 2, 2, 3])
>>> for i, cycle_type in enumerate(indexed_cycle_types.cycle_types):
...     pp(cycle_type)
...     pp(gr_coeffs.coeffs[:, :, i])
...     print('---')
(3, 0, 0)
array([[1, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]], dtype=object)
---
(1, 1, 0)
array([[0, 0, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0]], dtype=object)
---
(0, 0, 1)
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 0, 0, 0],
       [1, -1, 0, 0]], dtype=object)
---
"""

__all__ = ["GeckRouquierCoeffs", "get_gr_coeffs"]

from dataclasses import dataclass
import numpy as np

from permutations import (
    PermIndex, IndexedPerms,
    get_indexed_perms, get_flip,
)
from cycle_types import (
    CycleTypeIndex, IndexedCycleTypes,
    get_indexed_cycle_types,
)
from union_find import BlockIndex, IndexedBlocks, UnionFind


@dataclass
class GeckRouquierCoeffs:
    """Geck-Rouquier basis coefficients."""
    # the strict polynomial degree bound, `n*(n-1)//2 + 1`
    N: int

    # the GR-equivalence classes
    indexed_blocks: IndexedBlocks

    # The numpy array of GR coefficients.
    # The dtype is `object`, so the entries are regular Python bignums.
    # The shape is `(len(blocks), N, len(cycle_types))`.
    coeffs: np.ndarray

    # allow unpacking
    def __iter__(self):
        return iter((self.N, self.indexed_blocks, self.coeffs))


def get_gr_coeffs(
    indexed_perms: IndexedPerms,
    indexed_cycle_types: IndexedCycleTypes,
) -> GeckRouquierCoeffs:
    """
    Compute the Geck-Rouquier basis coefficients in the natural basis of the
    Hecke algebra.

    These coefficients are polynomials in $q^{-1}$.
    """
    # unpack precomputed values
    n, perms, perm_index, inverse, mult = indexed_perms
    cycle_types, cycle_type_index, bru_min_perms = indexed_cycle_types

    # start merging permutations which have the same GR basis coefficients
    uf = UnionFind(len(perms))

    # for each cycle type, Bruhat-minimum permutations are GR-equivalent
    for cluster in bru_min_perms:
        uf.merge(cluster)

    # for non-Bruhat-minimum permutations, we know some GR-equivalences
    for perm, (bru_len, one_line) in enumerate(perms):
        cluster = [perm]
        # horizontal conjugates are GR-equivalent
        for i in range(n-1):
            conjugate = inverse[mult[inverse[mult[perm][i]]][i]]
            if perms[conjugate][0] == bru_len:
                cluster.append(conjugate)
        # the inverse is GR-equivalent
        cluster.append(inverse[perm])
        # the flip is GR-equivalent
        cluster.append(perm_index[get_flip(one_line)])
        uf.merge(cluster)

    # get a first approximation of the GR-equivalence classes
    blocks, block_index = uf.get_indexed_blocks()

    # The GR coefficients can be computed by using the following recursion,
    # where $w$ is a permutation, $s$ is an adjacent transposition that is
    # neither a left nor a right descent of $w$, $sw \neq ws$, and $B$ is a
    # GR basis element:
    # $$ [T_{sws}]B = (1-q^{-1}) [T_{sw}]B + q^{-1} [T_w]B $$
    # The base case consists of the Bruhat-minimum permutations, which have
    # the constant polynomials 0 and 1 as GR coefficients.
    base_case: dict[BlockIndex, CycleTypeIndex] = {
        block_index[cluster[0]]: ctype_index
        for ctype_index, cluster in enumerate(bru_min_perms)
    }
    recursive_case: dict[BlockIndex, tuple[BlockIndex, BlockIndex]] = {}
    sws: BlockIndex
    block: Block
    for sws, block in enumerate(blocks):
        if sws in base_case: continue
        for sws_perm in block:
            for i in range(n-1):
                sw_perm = mult[sws_perm][i]
                if sw_perm < sws_perm:
                    w_perm = inverse[mult[inverse[sw_perm]][i]]
                    if w_perm < sw_perm: break
            else: continue
            break
        else: raise ValueError(f'Counter-example: {perms[sws_perm][1]}')
        recursive_case[sws] = (block_index[sw_perm], block_index[w_perm])

    # strict polynomial degree bound
    N = n*(n-1)//2 + 1

    # array to hold the GR coefficients
    coeffs = np.zeros((len(blocks), N, len(cycle_types)), dtype=object)

    # actually do the computation
    for bi, cti in base_case.items():
        coeffs[bi, 0, cti] = 1
    for sws, (sw, w) in recursive_case.items():
        # coeffs[sws] = (1-qi)*coeffs[sw] + qi*coeffs[w]
        coeffs[sws] = coeffs[sw]
        coeffs[sws, 1:] -= coeffs[sw, :-1]
        coeffs[sws, 1:] += coeffs[w, :-1]

    # now we can compute the true GR-equivalence classes
    groups: dict[tuple[int, ...], list[PermIndex]] = {}
    for subcoeffs, block in zip(coeffs, blocks):
        groups.setdefault(tuple(subcoeffs.flat), []).append(block[0])
    for cluster in groups.values():
        uf.merge(cluster)
    indexed_blocks = uf.get_indexed_blocks()

    # match up the new block indices with the old ones
    lookup = [block_index[b[0]] for b in indexed_blocks.blocks]

    return GeckRouquierCoeffs(N, indexed_blocks, coeffs[lookup])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
