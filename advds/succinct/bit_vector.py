"""Packed fixed length bit vector.

Amakobe, section 12.5, wants ``access`` in constant time and a bit vector that
does not spend a Python integer on every bit. Bits are stored in 64 bit words.
Rank and select live in ``rank_select.py`` and are attached by
``SuccinctBitVector``, which is the object the rest of the library queries.
"""

from __future__ import annotations

import array
from typing import Iterable, List, Sequence, Union


BitSource = Union[str, Sequence[int], Iterable[int]]


class BitVector:
    """Fixed length bit vector packed into 64 bit words.

    ``access`` is O(1). Space is O(n) bits, one word per 64 positions, plus
    the small Python object header. The length does not change after
    construction.
    """

    def __init__(self, bits: BitSource) -> None:
        data = _parse_bits(bits)
        self.n = len(data)
        self.words = array.array("Q")
        for start in range(0, self.n, 64):
            word = 0
            last = min(start + 64, self.n)
            for offset, bit in enumerate(data[start:last]):
                if bit:
                    word |= 1 << offset
            self.words.append(word)
        if self.n == 0:
            self.words = array.array("Q")

    def access(self, index: int) -> int:
        """Bit at ``index``, 0 or 1. O(1)."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        return (self.words[index >> 6] >> (index & 63)) & 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return self.access(index)

    def __iter__(self):
        for index in range(self.n):
            yield self.access(index)

    def to_list(self) -> List[int]:
        return [self.access(i) for i in range(self.n)]

    def __str__(self) -> str:
        return "".join(str(bit) for bit in self)

    def __repr__(self) -> str:
        preview = str(self)
        if len(preview) > 48:
            preview = preview[:45] + "..."
        return f"BitVector('{preview}', n={self.n})"


def _parse_bits(bits: BitSource) -> List[int]:
    if isinstance(bits, str):
        out: List[int] = []
        for char in bits:
            if char == "1":
                out.append(1)
            elif char == "0":
                out.append(0)
            elif char in " \n\t":
                continue
            else:
                raise ValueError(f"Bit strings may contain only 0 and 1, not {char!r}.")
        return out
    return [1 if bit else 0 for bit in bits]


class SuccinctBitVector(BitVector):
    """Bit vector plus the rank/select index from section 12.5.2.

    ``rank1(i)`` counts ones in ``bits[0:i+1]`` in O(1) after the directories
    are built. ``select1(k)`` returns the 0 based position of the k th one
    (k starts at 1), using the superblock directory, the block ranks, and the
    in block lookup table. ``rank`` and ``select`` are the names used in the
    chapter text and mean rank1 and select1.
    """

    def __init__(self, bits: BitSource) -> None:
        super().__init__(bits)
        # Local import keeps bit_vector and rank_select from importing each other at load time.
        from .rank_select import RankSelectIndex

        self.index = RankSelectIndex(self)

    def rank1(self, index: int) -> int:
        """Number of 1 bits in positions ``0`` through ``index`` inclusive. O(1)."""
        return self.index.rank1(index)

    def rank0(self, index: int) -> int:
        """Number of 0 bits in positions ``0`` through ``index`` inclusive. O(1)."""
        return self.index.rank0(index)

    def rank(self, index: int) -> int:
        """Chapter name for ``rank1``."""
        return self.rank1(index)

    def select1(self, k: int) -> int:
        """Position of the k th 1, with k starting at 1. Returns -1 if missing."""
        return self.index.select1(k)

    def select0(self, k: int) -> int:
        """Position of the k th 0, with k starting at 1. Returns -1 if missing."""
        return self.index.select0(k)

    def select(self, k: int) -> int:
        """Chapter name for ``select1``."""
        return self.select1(k)

    @property
    def total_ones(self) -> int:
        return self.index.total_ones

    def overhead_bytes(self) -> int:
        """Bytes in the superblock and block directories, not counting the bits."""
        return self.index.directory_bytes()

    def __repr__(self) -> str:
        return f"SuccinctBitVector(n={self.n}, ones={self.total_ones})"
