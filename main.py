#!/usr/bin/env python
"""
Script to do the computation and output the result.

>>> print(to_json(3, *main(3)))
  "3": {
    "111x111": {
      "111" : {"e": 0, "c": [1]}
    },
    "111x21": {
      "21" : {"e": 0, "c": [1]}
    },
    "111x3": {
      "3" : {"e": 0, "c": [1]}
    },
    "21x21": {
      "111" : {"e": 1, "c": [3]},
      "21" : {"e": 0, "c": [-2, 2]},
      "3" : {"e": -1, "c": [1, 1, 1]}
    },
    "21x3": {
      "111" : {"e": 1, "c": [-1, 1]},
      "21" : {"e": 0, "c": [1, 0, 1]},
      "3" : {"e": -1, "c": [-1, -1, 1, 1]}
    },
    "3x3": {
      "111" : {"e": 1, "c": [1, 0, 1]},
      "21" : {"e": 0, "c": [-1, 0, 0, 1]},
      "3" : {"e": -1, "c": [1, 0, -1, 0, 1]}
    }
  }
"""

import argparse
import numpy as np

from permutations import *
from cycle_types import *
from union_find import *
from geck_rouquier import *
from hecke_mult import *


def main(n: int) -> tuple[list[CycleType], np.ndarray]:
    """
    Returns the list of cycle types for $S_n$, together with a numpy array of
    shape `(len(cycle_types), len(cycle_types), len(cycle_types), M)`, where
    `M == 3*n*(n-1)//2 + 1`. The dtype is `object`, so that each entry is a
    regular Python int.

    The first two cycle types indicate the Geck-Rouquier basis elements being
    multiplied. The third cycle type indicates which coefficient of the
    product. The fourth index is for the power of $q$ in the Laurent
    polynomial, starting at `-n*(n-1)`.
    """
    indexed_perms = get_indexed_perms(n)
    indexed_cycle_types = get_indexed_cycle_types(indexed_perms)
    gr_coeffs = get_gr_coeffs(indexed_perms, indexed_cycle_types)
    bb_coeffs = get_block_block_coeffs(
        indexed_perms,
        indexed_cycle_types,
        gr_coeffs.indexed_blocks,
    )

    blocks = gr_coeffs.indexed_blocks.blocks
    cycle_types = indexed_cycle_types.cycle_types
    N = gr_coeffs.N
    mult_coeffs = np.zeros((
        len(cycle_types),
        len(cycle_types),
        len(cycle_types),
        3*n*(n-1)//2 + 1,
    ), dtype=object)
    # such a terrible and slow loop (-_-)
    for left_cti in range(len(cycle_types)):
      for left_bi in range(len(blocks)):
        for left_deg, left_coeff in enumerate(
          gr_coeffs.coeffs[left_bi, :, left_cti]
        ):
          if left_coeff != 0:
            for right_cti in range(len(cycle_types)):
              for right_bi in range(len(blocks)):
                for right_deg, right_coeff in enumerate(
                  gr_coeffs.coeffs[right_bi, :, right_cti]
                ):
                  if right_coeff != 0:
                    for prod_cti in range(len(cycle_types)):
                      shift = n*(n-1) - left_deg - right_deg
                      mult_coeffs[
                        left_cti, right_cti, prod_cti, shift : shift+N,
                      ] += (left_coeff * right_coeff * bb_coeffs[
                        left_bi, right_bi, :, prod_cti
                      ])
    return cycle_types, mult_coeffs


def to_json(n, cycle_types, mult_coeffs) -> str:
    """
    Return a formatted JSON string for a multiplication table.
    """
    shift = n*(n-1)
    cycle_type_strings = [
        ''.join(reversed([str(i)*m for i, m in enumerate(c, 1)]))
        for c in cycle_types
    ]
    products = []
    for li, left_str in enumerate(cycle_type_strings):
        for ri, right_str in enumerate(cycle_type_strings[li:], li):
            gr_coeffs = []
            for ci, coeff_str in enumerate(cycle_type_strings):
                laurent = mult_coeffs[li, ri, ci]
                nonzero_mask = (laurent != 0)
                if nonzero_mask.any():
                    # how many polynomial coeffs to trim on the left
                    for first, val in enumerate(nonzero_mask):
                        if val: break
                    # how many polynomial coeffs to trim on the right
                    for last, val in enumerate(nonzero_mask[::-1]):
                        if val: break
                    # last may be zero, in which case -last would be wrong
                    trimmed = laurent[first : len(laurent)-last]
                    joined = ', '.join(map(str, trimmed))
                    gr_coeffs.append(
                        f'      "{coeff_str}" : {{'
                        f'"e": {first-shift}, '
                        f'"c": [{joined}]}}'
                    )
            joined = ',\n'.join(gr_coeffs)
            products.append(
                f'    "{left_str}x{right_str}": {{\n'
                f'{joined}\n'
                f'    }}'
            )
    joined = ',\n'.join(products)
    return (
        f'  "{n}": {{\n'
        f'{joined}\n'
        f'  }}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute Geck-Rouquier multiplication tables",
    )
    parser.add_argument(
        "n",
        help="which multiplication table to compute and display",
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "-t", "--test",
        help="run doctests for the script",
        action="store_true",
    )
    args = parser.parse_args()
    if args.test:
        import doctest
        doctest.testmod()
    else:
        if not args.n:
            parser.error("missing argument: n")
        for n in args.n:
            if n < 1:
                parser.error(f"invalid argument: {n=}")
        tables = [to_json(n, *main(n)) for n in args.n]
        joined = ',\n'.join(tables)
        print(f'{{\n{joined}\n}}')
