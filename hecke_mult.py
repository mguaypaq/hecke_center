"""
Extracting certain coefficients in certain products in the Hecke algebra.

The goal is only to be able to compute the structure coefficients for the
center of the algebra, in the Geck-Rouquier basis, so we can make some
simplifying and optimizing assumptions.

One simple approach would be to pre-compute the product of every (ordered) pair
of natural basis elements for the whole Hecke algebra, expressed as a linear
combination of the natural basis elements, with coefficients being Laurent
polynomials in $q$. Unfortunately, this would require an $(n!)^3$ amount of
work, at least. (And also that much memory.)

A first simplification is that we know that certain pairs of natural basis
elements have equal coefficients for all elements of the center. So, we can
partition the elements of the symmetric group (which index the natural basis)
into blocks of elements which have equal coefficients for all elements of the
center, in the coarsest possible way. Then, we only need to multiply every
(ordered) pair of block sums.

A second simplification is that we don't need all coefficients of the result
in the natural basis. We only need enough of them to uniquely determine the
coefficients of the result in the Geck-Rouquier basis. In fact, a single
natural basis coefficient is enough for each Geck-Rouquier basis coefficient:
for each cycle type, an arbitrary Bruhat-minimum element will do the job.

Now, let's consider a naive loop to extract a single natural basis coefficient
from a product of two block sums, and see how many duplicated sub-computations
we can eliminate. For the left block sum, we can represent it as an $n!$-tuple
of coefficients in the natural basis, written as a column vector $b$. For the
right block sum, we can iterate over its summands, each of which can be
expressed as a product $T_1 T_2 ... T_k$ of generators for the Hecke algebra.
And we can represent the coefficient extraction operator as another $n!$-tuple
of coefficients in the natural basis, this time written as a row vector $c$.
For each $T_i$, write $M_i$ for the $n!$ by $n!$ square matrix which implements
the linear operation of multiplication on the right by $T_i$ in the Hecke
algebra. Then, the coefficient we're computing is:

$$ c M_k ... M_2 M_1 b $$  (1)

Since there are many fewer cycle types (which index the vectors $c$) than block
sums (which index the vectors $b$), it's more efficient to pre-compute the
products starting from $c$,

$$ c M_k ... M_2 M_1 $$  (2)

than the products starting from $b$,

$$ M_k ... M_2 M_1 b $$  (3)

Furthermore, each entry of the row vector (2) belongs to exactly one block sum,
so all of the entries are useful. And the partial products (when going from
left to right) are also useful. By constructing the words $T_k ... T_2 T_1$
using a Lehmer code, we can iterate through them so that:

1. Every permutation is produced exactly once.
2. Its reduced word is obtained by adding a single generator to a previously
   generated reduced word.
3. At most $n$ previously generated reduced words need to be kept in memory at
   once.

We can of course replace the single row vector $c$ by a matrix with several
rows, one for each cycle type. Then, we have the nested loops:

1. For every Lehmer code $T_k ... T_2 T_1$, of which there are $n!$.
2. Recall the product $c M_k ... M_2$, and apply $M_1$ on the right.
3. For every column of the result, of which there are $n!$, add its entries
   to the appropriate slice of a result array.

Finally, note that each matrix $M_i$ has a special shape: it's a 2x2 block
matrix. Specifically, if we reorder the natural basis elements so that:

1. The permutations *without* $T_i$ as a right descent come first.
2. The permutations with $T_i$ as a right descent come second, in the same
   relative order.

Then, the matrix for $M_i$ is:

[ 0 |   q*I   ]
[-------------]
[ I | (q-1)*I ]

So, both $M_i$ and its transpose can be implements as a few vector additions.

>>> indexed_perms = get_indexed_perms(3)
>>> indexed_cycle_types = get_indexed_cycle_types(indexed_perms)
>>> gr_coeffs = get_gr_coeffs(indexed_perms, indexed_cycle_types)
>>> bb_coeffs = get_block_block_coeffs(
...     indexed_perms,
...     indexed_cycle_types,
...     gr_coeffs.indexed_blocks,
... )
>>> print(bb_coeffs[2, 3, :, :])
[[0 0 0]
 [0 0 -2]
 [0 1 2]
 [0 0 0]]
"""

__all__ = ["get_block_block_coeffs"]

import numpy as np

from permutations import PermIndex, IndexedPerms, get_indexed_perms
from cycle_types import (
    CycleTypeIndex, IndexedCycleTypes,
    get_indexed_cycle_types,
)
from union_find import IndexedBlocks
from geck_rouquier import get_gr_coeffs


def get_block_block_coeffs(
    indexed_perms: IndexedPerms,
    indexed_cycle_types: IndexedCycleTypes,
    indexed_blocks: IndexedBlocks,
) -> np.ndarray:
    """
    Compute some representative coefficients in the natural basis for every
    ordered pair of block sums.

    Mathematically, each coefficient is a polynomial in $q$.

    The result is a numpy array.
    The dtype is `object`, so the entries are regular Python bignums.
    The shape is `(len(blocks), len(blocks), N, len(cycle_types))`,
    where `N` is the strict polynomial degree bound `n*(n-1)//2 + 1`.
    """
    # unpack precomputed values
    n, perms, perm_index, inverse, mult = indexed_perms
    cycle_types, cycle_type_index, bru_min_perms = indexed_cycle_types
    blocks, block_index = indexed_blocks

    # polynomial degree bound
    N = n*(n-1)//2 + 1

    # array to hold the polynomial coefficients
    coeffs = np.zeros(
        (len(blocks), len(blocks), N, len(cycle_types)),
        dtype=object,
    )

    # For each adjacent transposition $s$, we need the list of permutations $w$
    # which don't have $s$ as a right descent, and the corresponding list of
    # products $ws$ (which do have $s$ as a right descent) in the same order.
    splits: list[tuple[list[PermIndex], list[PermIndex]]] = [
        ([], [])
        for _ in range(n-1)
    ]
    perm: PermIndex
    products: list[PermIndex]  # key: range(n-1)
    for perm, products in enumerate(mult):
        prod: PermIndex
        for i, prod in enumerate(products):
            if perm < prod:
                w, ws = splits[i]
                w.append(perm)
                ws.append(prod)

    # For each cycle type, the coefficient extractor of a Bruhat-minimum
    # natural basis element of that cycle type in the Hecke algebra.
    initial_row_vec = np.zeros(
        (len(perms), N, len(cycle_types)),
        dtype=object,
    )
    cti: CycleTypeIndex
    ps: list[PermIndex]
    for cti, ps in enumerate(bru_min_perms):
        initial_row_vec[ps[0], 0, cti] = 1

    # The loop to generate permutations using a Lehmer code, used for
    # block sum being multiplied on the right.
    def loop(j=0, right_perm_inverse=0, row_vec=initial_row_vec):
        """Variable depth nested loop."""
        if j < n:
            # case 1: `j` stays where it is
            yield from loop(j+1, right_perm_inverse, row_vec)
            for i in reversed(range(j)):
                # case 2: `j` moves one more step towards the front using $s_i$
                right_perm_inverse = mult[right_perm_inverse][i]
                w, ws = splits[i]
                # row_vec[w] = old_row_vec[ws]
                # row_vec[ws] = q*old_row_vec[w] + (q-1)*old_row_vec[ws]
                old_row_vec_w = row_vec[w]
                old_row_vec_ws = row_vec[ws]
                row_vec = np.empty_like(row_vec)
                row_vec[w] = old_row_vec_ws
                row_vec[ws] = -old_row_vec_ws
                row_vec[ws, 1:] += old_row_vec_ws[:, :-1]
                row_vec[ws, 1:] += old_row_vec_w[:, :-1]
                yield from loop(j+1, right_perm_inverse, row_vec)
        else:
            yield right_perm_inverse, row_vec

    # The loop to collect each slice of coefficients in the correct slice of
    # the result array for the pair of block sums being multiplied.
    for right_perm_inverse, row_vec in loop():
        right_block = block_index[inverse[right_perm_inverse]]
        for left_block, subcoeffs in zip(block_index, row_vec):
            coeffs[left_block, right_block] += subcoeffs

    return coeffs


if __name__ == "__main__":
    import doctest
    doctest.testmod()
