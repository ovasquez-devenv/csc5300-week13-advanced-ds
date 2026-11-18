"""Constant time rank, and select via superblocks, blocks, and lookup tables.

Amakobe, section 12.5.2, sketches a two level directory: a superblock every
about log squared n bits stores the number of ones before it, and a block
inside that superblock stores the ones since the superblock started. The
chapter's sample code then scans the leftover bits. This module replaces that
scan with a 16 bit lookup table, so ``rank1`` is a handful of array reads.

``select1`` binary searches the superblock directory (there are n / log^2 n of
them), walks the blocks of that one superblock, and finishes with the same
table. The in block step is O(1). The directory search is O(log n), which is
the bound the chapter gives before extra select sampling.
"""

from __future__ import annotations

import array
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bit_vector import BitVector


BLOCK_BITS = 16

# popcount[pattern] for every 16 bit pattern. Built once and shared.
_POP16 = array.array("B", (pattern.bit_count() for pattern in range(1 << BLOCK_BITS)))

# sel1[pattern * 16 + (k - 1)] is the bit offset of the k th 1, or 255 if none.
_SEL1 = array.array("B", b"\xff" * ((1 << BLOCK_BITS) * BLOCK_BITS))
for _pattern in range(1 << BLOCK_BITS):
    _rank = 0
    _base = _pattern << 4
    _bits = _pattern
    while _bits:
        _lsb = _bits & -_bits
        _SEL1[_base + _rank] = _lsb.bit_length() - 1
        _rank += 1
        _bits ^= _lsb


def popcount16(pattern: int) -> int:
    """Ones in a 16 bit pattern. O(1) table lookup."""
    return _POP16[pattern & 0xFFFF]


def select1_in_pattern(pattern: int, k: int) -> int:
    """Offset of the k th 1 inside a 16 bit pattern. k starts at 1. O(1)."""
    if not 1 <= k <= BLOCK_BITS:
        return -1
    found = _SEL1[((pattern & 0xFFFF) << 4) + (k - 1)]
    return -1 if found == 255 else found


def select0_in_pattern(pattern: int, k: int) -> int:
    """Offset of the k th 0, by selecting a 1 in the complemented pattern."""
    return select1_in_pattern((~pattern) & 0xFFFF, k)


class RankSelectIndex:
    """Superblock and block directories over a packed ``BitVector``.

    Block width is fixed at 16 so the lookup tables stay valid. Superblock
    width is the smallest multiple of 16 that is at least ``(log2 n)^2``,
    matching the chapter's two level split. Building the directories is O(n).
    ``rank1`` and ``rank0`` are O(1). ``select1`` and ``select0`` are O(log n)
    directory search plus an O(1) table probe inside the chosen block.
    """

    def __init__(self, bitvector: "BitVector") -> None:
        self.bits = bitvector
        self.n = bitvector.n
        log_n = max(4, int(math.log2(self.n))) if self.n > 1 else 4
        raw_super = max(BLOCK_BITS, log_n * log_n)
        self.super_size = ((raw_super + BLOCK_BITS - 1) // BLOCK_BITS) * BLOCK_BITS
        self.nblocks = (self.n + BLOCK_BITS - 1) // BLOCK_BITS if self.n else 0
        self.super_rank = array.array("Q")
        self.block_rank = array.array("I")
        self.total_ones = 0
        if self.n:
            self._build()

    def _pattern(self, block: int) -> int:
        start = block * BLOCK_BITS
        if start >= self.n:
            return 0
        word = self.bits.words[start >> 6]
        return (word >> (start & 63)) & 0xFFFF

    def _build(self) -> None:
        ones_before = 0
        super_count = (self.n + self.super_size - 1) // self.super_size
        block_cursor = 0
        for super_index in range(super_count):
            self.super_rank.append(ones_before)
            super_ones = 0
            block_end = min(self.nblocks, ((super_index + 1) * self.super_size) // BLOCK_BITS)
            while block_cursor < block_end:
                self.block_rank.append(super_ones)
                start = block_cursor * BLOCK_BITS
                width = min(BLOCK_BITS, self.n - start)
                pattern = self._pattern(block_cursor) & ((1 << width) - 1)
                count = popcount16(pattern)
                super_ones += count
                block_cursor += 1
            ones_before += super_ones
        self.total_ones = ones_before

    def rank1(self, index: int) -> int:
        """Ones in ``[0, index]`` inclusive. ``index < 0`` yields 0. O(1)."""
        if index < 0 or self.n == 0:
            return 0
        if index >= self.n:
            return self.total_ones
        super_index = index // self.super_size
        block = index // BLOCK_BITS
        total = self.super_rank[super_index] + self.block_rank[block]
        offset = index - block * BLOCK_BITS
        start = block * BLOCK_BITS
        width = min(BLOCK_BITS, self.n - start)
        pattern = self._pattern(block) & ((1 << width) - 1)
        if offset >= width - 1:
            return total + popcount16(pattern)
        mask = (1 << (offset + 1)) - 1
        return total + popcount16(pattern & mask)

    def rank0(self, index: int) -> int:
        """Zeros in ``[0, index]`` inclusive. O(1)."""
        if index < 0 or self.n == 0:
            return 0
        if index >= self.n:
            return self.n - self.total_ones
        return (index + 1) - self.rank1(index)

    def select1(self, k: int) -> int:
        """0 based index of the k th one. k starts at 1. O(log n)."""
        if k <= 0 or k > self.total_ones:
            return -1
        super_index = self._find_superblock(k, zeros=False)
        return self._select_in_superblock(super_index, k, zeros=False)

    def select0(self, k: int) -> int:
        """0 based index of the k th zero. k starts at 1. O(log n)."""
        total_zeros = self.n - self.total_ones
        if k <= 0 or k > total_zeros:
            return -1
        super_index = self._find_superblock(k, zeros=True)
        return self._select_in_superblock(super_index, k, zeros=True)

    def _find_superblock(self, k: int, zeros: bool) -> int:
        """Largest superblock whose prefix count of the requested bit is < k."""
        last = len(self.super_rank) - 1
        lo = 0
        hi = last
        while lo < hi:
            mid = (lo + hi + 1) // 2
            before = self._count_before_superblock(mid, zeros)
            if before < k:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _count_before_superblock(self, super_index: int, zeros: bool) -> int:
        ones = self.super_rank[super_index]
        if not zeros:
            return ones
        start = min(self.n, super_index * self.super_size)
        return start - ones

    def _select_in_superblock(self, super_index: int, k: int, zeros: bool) -> int:
        block_start = (super_index * self.super_size) // BLOCK_BITS
        block_end = min(self.nblocks, ((super_index + 1) * self.super_size) // BLOCK_BITS)
        for block in range(block_start, block_end):
            bit_start = block * BLOCK_BITS
            width = min(BLOCK_BITS, self.n - bit_start)
            pattern = self._pattern(block) & ((1 << width) - 1)
            ones_before = self.super_rank[super_index] + self.block_rank[block]
            if zeros:
                before = bit_start - ones_before
                inside = width - popcount16(pattern)
                if before + inside < k:
                    continue
                offset = select0_in_pattern(pattern, k - before)
            else:
                before = ones_before
                inside = popcount16(pattern)
                if before + inside < k:
                    continue
                offset = select1_in_pattern(pattern, k - before)
            if offset < 0 or offset >= width:
                continue
            return bit_start + offset
        return -1

    def directory_bytes(self) -> int:
        """Bytes occupied by the per vector directories (not the shared tables)."""
        return self.super_rank.itemsize * len(self.super_rank) + self.block_rank.itemsize * len(self.block_rank)

    def __repr__(self) -> str:
        return (
            f"RankSelectIndex(n={self.n}, super_size={self.super_size}, "
            f"blocks={self.nblocks}, ones={self.total_ones})"
        )
