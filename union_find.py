"""
Union-find data structure, used for computing equivalence classes of an
equivalence relation on PermIndexes.

>>> from pprint import pp
>>> uf = UnionFind(8)
>>> uf.merge([0, 3])
>>> uf.merge([3, 4, 7])
>>> uf.merge([5, 6])
>>> pp(uf.get_indexed_blocks())
IndexedBlocks(blocks=[[0, 3, 4, 7], [1], [2], [5, 6]],
              block_index=[0, 1, 2, 0, 0, 3, 3, 0])
"""

__all__ = [
    "Block", "BlockIndex", "IndexedBlocks", "UnionFind",
]

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NewType

from permutations import PermIndex

# an equivalence class, represented as a sorted list
Block = NewType('Block', list[PermIndex])

# an index into a list of equivalence classes
BlockIndex = NewType('BlockIndex', int)


@dataclass
class IndexedBlocks:
    """The mapping between Blocks and PermIndexes, in both directions."""
    blocks: list[Block]  # key: BlockIndex
    block_index: list[BlockIndex]  # key: PermIndex

    # allow unpacking
    def __iter__(self):
        return iter((self.blocks, self.block_index))


class UnionFind:
    """Union-find data structure."""
    # `i` and `parents[i] <= i` are in the same equivalence class
    parents: list[PermIndex]  # key: PermIndex

    def __init__(self, length: int):
        """`length` should be $n!$"""
        self.parents = list(range(length))

    def merge(self, cluster: Iterable[PermIndex]):
        """Merge the equivalence classes of several permutations."""
        ancestry = set()
        for p in cluster:
            while p not in ancestry:
                ancestry.add(p)
                p = self.parents[p]
        m = min(ancestry)
        for p in ancestry:
            self.parents[p] = m

    def get_indexed_blocks(self) -> IndexedBlocks:
        """Return the current equivalence classes."""
        blocks: list[Block] = []
        block_index: list[BlockIndex] = []
        for perm, parent in enumerate(self.parents):
            if parent == perm:
                # new component
                index = len(blocks)
                blocks.append([])
            else:
                # existing component
                index = block_index[parent]
            blocks[index].append(perm)
            block_index.append(index)
        return IndexedBlocks(blocks, block_index)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
