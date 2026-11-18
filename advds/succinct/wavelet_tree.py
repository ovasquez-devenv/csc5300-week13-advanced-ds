"""Wavelet tree over a sequence, built on succinct rank and select.

Amakobe, section 12.5.3, splits the alphabet in half at every node and stores
a bit vector: 0 if the symbol went left, 1 if it went right. ``access``,
``rank``, and ``select`` each walk O(log sigma) nodes, where sigma is the
alphabet size. The bit vectors along a root to leaf path are the compressed
form of the text: about n log2(sigma) bits, before the rank directories.

Range count is a difference of ranks. The k th smallest symbol in a window
(a range quantile) follows the heavier side of the same bits, which is the
extra query the DNA index needs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .bit_vector import SuccinctBitVector


class WaveletTree:
    """Balanced wavelet tree.

    ``alphabet`` may be passed explicitly so a missing symbol still has a
    stable order (DNA uses A, C, G, T even if one base is absent). Symbols
    are ordered by their sorted alphabet, which is what ``range_quantile``
    means by smallest.
    """

    class Node:
        def __init__(self, alphabet: Sequence[object]) -> None:
            self.alphabet = list(alphabet)
            self.bitmap: Optional[SuccinctBitVector] = None
            self.left: Optional["WaveletTree.Node"] = None
            self.right: Optional["WaveletTree.Node"] = None

    def __init__(self, sequence: Sequence[object], alphabet: Optional[Iterable[object]] = None) -> None:
        self.sequence = list(sequence)
        self.n = len(self.sequence)
        if alphabet is None:
            self.alphabet = sorted(set(self.sequence), key=_sort_key)
        else:
            self.alphabet = sorted(set(alphabet), key=_sort_key)
            missing = [symbol for symbol in self.sequence if symbol not in set(self.alphabet)]
            if missing:
                raise ValueError(f"Symbol {missing[0]!r} is not in the alphabet.")
        self._alpha_index = {symbol: pos for pos, symbol in enumerate(self.alphabet)}
        self.root = self._build(self.alphabet, self.sequence)

    def _build(self, alphabet: Sequence[object], sequence: Sequence[object]) -> "WaveletTree.Node":
        node = WaveletTree.Node(alphabet)
        if len(alphabet) <= 1 or not sequence:
            return node
        mid = len(alphabet) // 2
        left_set = set(alphabet[:mid])
        bits: List[int] = []
        left_seq: List[object] = []
        right_seq: List[object] = []
        for symbol in sequence:
            if symbol in left_set:
                bits.append(0)
                left_seq.append(symbol)
            else:
                bits.append(1)
                right_seq.append(symbol)
        node.bitmap = SuccinctBitVector(bits)
        if left_seq:
            node.left = self._build(alphabet[:mid], left_seq)
        if right_seq:
            node.right = self._build(alphabet[mid:], right_seq)
        return node

    def access(self, index: int) -> object:
        """Symbol at ``index``. O(log sigma)."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        node = self.root
        while node.bitmap is not None:
            bit = node.bitmap.access(index)
            if bit == 0:
                index = node.bitmap.rank0(index) - 1
                if node.left is None:
                    break
                node = node.left
            else:
                index = node.bitmap.rank1(index) - 1
                if node.right is None:
                    break
                node = node.right
        return node.alphabet[0]

    def rank(self, symbol: object, index: int) -> int:
        """Occurrences of ``symbol`` in ``sequence[0:index+1]``. O(log sigma).

        ``index < 0`` returns 0. An unknown symbol returns 0.
        """
        if symbol not in self._alpha_index or self.n == 0:
            return 0
        if index < 0:
            return 0
        if index >= self.n:
            index = self.n - 1
        node = self.root
        target = self._alpha_index[symbol]
        while node.bitmap is not None:
            mid = len(node.alphabet) // 2
            if target < mid:
                index = node.bitmap.rank0(index) - 1
                if index < 0 or node.left is None:
                    return 0
                node = node.left
            else:
                target -= mid
                index = node.bitmap.rank1(index) - 1
                if index < 0 or node.right is None:
                    return 0
                node = node.right
        return index + 1

    def select(self, symbol: object, k: int) -> int:
        """0 based position of the k th ``symbol``. k starts at 1. O(log sigma).

        Returns -1 when the symbol does not occur k times.
        """
        if k <= 0 or symbol not in self._alpha_index:
            return -1
        if self.n == 0 or self.rank(symbol, self.n - 1) < k:
            return -1
        return self._select(self.root, symbol, k)

    def _select(self, node: "WaveletTree.Node", symbol: object, k: int) -> int:
        if node.bitmap is None:
            return k - 1
        mid = len(node.alphabet) // 2
        if symbol in set(node.alphabet[:mid]):
            assert node.left is not None and node.bitmap is not None
            child_pos = self._select(node.left, symbol, k)
            return node.bitmap.select0(child_pos + 1)
        assert node.right is not None and node.bitmap is not None
        child_pos = self._select(node.right, symbol, k)
        return node.bitmap.select1(child_pos + 1)

    def range_count(self, left: int, right: int, symbol: object) -> int:
        """Occurrences of ``symbol`` in the inclusive window. O(log sigma)."""
        self._check_window(left, right)
        if left == 0:
            return self.rank(symbol, right)
        return self.rank(symbol, right) - self.rank(symbol, left - 1)

    def frequency(self, symbol: object, left: int = 0, right: int | None = None) -> int:
        """How often ``symbol`` occurs in ``[left, right]``. O(log sigma)."""
        if self.n == 0:
            return 0
        if right is None:
            right = self.n - 1
        return self.range_count(left, right, symbol)

    def frequencies(self, left: int = 0, right: int | None = None) -> Dict[object, int]:
        """Count of every alphabet symbol in the window. O(sigma log sigma)."""
        if self.n == 0:
            return {symbol: 0 for symbol in self.alphabet}
        if right is None:
            right = self.n - 1
        return {symbol: self.range_count(left, right, symbol) for symbol in self.alphabet}

    def range_quantile(self, left: int, right: int, k: int) -> object:
        """The k th smallest symbol in ``[left, right]``. k starts at 1.

        O(log sigma). Order is the sorted alphabet, so for DNA that is
        A, then C, then G, then T.
        """
        self._check_window(left, right)
        width = right - left + 1
        if not 1 <= k <= width:
            raise ValueError(f"k={k} is outside a window of {width} symbols.")
        node = self.root
        while node.bitmap is not None:
            zeros = node.bitmap.rank0(right) - node.bitmap.rank0(left - 1)
            if k <= zeros:
                if node.left is None:
                    break
                left = node.bitmap.rank0(left - 1)
                right = node.bitmap.rank0(right) - 1
                node = node.left
            else:
                if node.right is None:
                    break
                k -= zeros
                left = node.bitmap.rank1(left - 1)
                right = node.bitmap.rank1(right) - 1
                node = node.right
        return node.alphabet[0]

    def bitmap_bits(self) -> int:
        """Total length of every node bit vector. About n log2(sigma) bits."""
        total = 0

        def walk(node: Optional[WaveletTree.Node]) -> None:
            nonlocal total
            if node is None or node.bitmap is None:
                return
            total += node.bitmap.n
            walk(node.left)
            walk(node.right)

        walk(self.root)
        return total

    def _check_window(self, left: int, right: int) -> None:
        if self.n == 0 or left > right or left < 0 or right >= self.n:
            raise IndexError((left, right))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"WaveletTree(n={self.n}, sigma={len(self.alphabet)})"


def _sort_key(symbol: object):
    try:
        return (0, symbol)
    except TypeError:
        return (1, str(symbol))
