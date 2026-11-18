"""Fenwick trees (binary indexed trees) for prefix and rectangle sums.

Amakobe, section 12.3, stores a prefix of ``lowbit(i) = i & -i`` elements in
slot ``i``. A prefix sum clears that bit repeatedly, and a point update walks
the other way, so both are O(log n). The table has n + 1 numbers, which is
O(n) space, tighter than the segment tree's 4n array. Range sums are the
difference of two prefixes (section 12.3.3). The 2D tree in section 12.3.5
does the same walk on rows and on columns.
"""

from __future__ import annotations

from typing import List, Sequence


class FenwickTree:
    """1-indexed Fenwick tree exposed with 0-based public indices.

    ``from_array`` builds the table in O(n). ``update``, ``prefix_sum``,
    ``range_sum``, and ``set`` are O(log n). Space is O(n).
    """

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non negative.")
        self.n = n
        self.tree: List[int] = [0] * (n + 1)

    @staticmethod
    def lowbit(index: int) -> int:
        """Isolate the least significant set bit. ``index`` is 1-based and positive."""
        return index & -index

    @classmethod
    def from_array(cls, arr: Sequence[int]) -> "FenwickTree":
        """Build in O(n) by pushing each slot into its parent once.

        After the copy, ``tree[i]`` holds ``arr[i - 1]``. Adding it into
        ``i + lowbit(i)`` folds in the range that slot is responsible for.
        """
        tree = cls(len(arr))
        for i, value in enumerate(arr, start=1):
            tree.tree[i] = value
        for i in range(1, tree.n + 1):
            parent = i + cls.lowbit(i)
            if parent <= tree.n:
                tree.tree[parent] += tree.tree[i]
        return tree

    def update(self, index: int, delta: int) -> None:
        """Add ``delta`` to the 0-based ``index``. O(log n)."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        index += 1
        while index <= self.n:
            self.tree[index] += delta
            index += self.lowbit(index)

    def prefix_sum(self, index: int) -> int:
        """Sum of elements ``0`` through ``index`` inclusive. O(log n).

        ``index < 0`` is the empty prefix and returns 0.
        """
        if index < 0:
            return 0
        if index >= self.n:
            raise IndexError(index)
        index += 1
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= self.lowbit(index)
        return total

    def range_sum(self, left: int, right: int) -> int:
        """Sum of the inclusive interval, as a difference of prefixes. O(log n)."""
        if left > right:
            raise ValueError(f"Empty range [{left}, {right}].")
        if left < 0 or right >= self.n:
            raise IndexError((left, right))
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    def set(self, index: int, value: int) -> None:
        """Replace the value at ``index``. O(log n)."""
        current = self.range_sum(index, index)
        self.update(index, value - current)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"FenwickTree(n={self.n})"


class FenwickTree2D:
    """Two dimensional Fenwick tree for point adds and rectangle sums.

    A point update and a prefix query are O(log rows * log cols). A rectangle
    sum is four prefixes (inclusion exclusion). Space is O(rows * cols).
    """

    def __init__(self, rows: int, cols: int) -> None:
        if rows < 0 or cols < 0:
            raise ValueError("Dimensions must be non negative.")
        self.rows = rows
        self.cols = cols
        self.tree: List[List[int]] = [[0] * (cols + 1) for _ in range(rows + 1)]

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[int]]) -> "FenwickTree2D":
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
        tree = cls(rows, cols)
        for r in range(rows):
            if len(matrix[r]) != cols:
                raise ValueError("Matrix rows must have equal length.")
            for c in range(cols):
                tree.update(r, c, matrix[r][c])
        return tree

    def update(self, row: int, col: int, delta: int) -> None:
        """Add ``delta`` to cell ``(row, col)``. O(log rows * log cols)."""
        if not 0 <= row < self.rows or not 0 <= col < self.cols:
            raise IndexError((row, col))
        r = row + 1
        while r <= self.rows:
            c = col + 1
            while c <= self.cols:
                self.tree[r][c] += delta
                c += c & -c
            r += r & -r

    def prefix_sum(self, row: int, col: int) -> int:
        """Sum of the rectangle from ``(0, 0)`` through ``(row, col)``. O(log * log)."""
        if row < 0 or col < 0:
            return 0
        if row >= self.rows or col >= self.cols:
            raise IndexError((row, col))
        total = 0
        r = row + 1
        while r > 0:
            c = col + 1
            while c > 0:
                total += self.tree[r][c]
                c -= c & -c
            r -= r & -r
        return total

    def rectangle_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Sum of cells with rows ``r1..r2`` and columns ``c1..c2``. O(log * log)."""
        if r1 > r2 or c1 > c2:
            raise ValueError("Empty rectangle.")
        if r1 < 0 or c1 < 0 or r2 >= self.rows or c2 >= self.cols:
            raise IndexError((r1, c1, r2, c2))
        total = self.prefix_sum(r2, c2)
        if r1 > 0:
            total -= self.prefix_sum(r1 - 1, c2)
        if c1 > 0:
            total -= self.prefix_sum(r2, c1 - 1)
        if r1 > 0 and c1 > 0:
            total += self.prefix_sum(r1 - 1, c1 - 1)
        return total

    def __repr__(self) -> str:
        return f"FenwickTree2D(rows={self.rows}, cols={self.cols})"
