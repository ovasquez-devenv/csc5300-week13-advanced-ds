"""Segment trees for range aggregates, including lazy range updates.

The layout follows Amakobe, Advanced Computational Algorithms (2nd ed., 2026),
section 12.2. A balanced binary tree stores one aggregate per segment. A query
interval breaks into O(log n) disjoint nodes, so both reads and point writes
are O(log n). The flat array uses the usual 4n slots, which is O(n) space.

Lazy propagation (section 12.2.5) tags a node when a whole segment is updated
and pushes that tag down only when a later query or update needs the children.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple, Union


Number = Union[int, float]
Combine = Callable[[Number, Number], Number]


def resolve_operation(operation: str) -> Tuple[Combine, Number]:
    """Return ``(combine, identity)`` for a named associative operation."""
    if operation == "sum":
        return (lambda a, b: a + b), 0
    if operation == "min":
        return min, math.inf
    if operation == "max":
        return max, -math.inf
    if operation == "gcd":
        return math.gcd, 0
    raise ValueError(
        f"Unknown operation {operation!r}. Expected 'sum', 'min', 'max', or 'gcd'."
    )


class SegmentTree:
    """Segment tree for range sum, min, max, or gcd with point updates.

    Build is O(n). ``query`` and ``update`` are O(log n). Space is O(n).
    Indices are 0 based and ranges are inclusive.
    """

    def __init__(self, arr: Sequence[Number], operation: str = "sum") -> None:
        if len(arr) == 0:
            raise ValueError("Segment tree needs at least one element.")
        self.operation = operation
        self.op, self.identity = resolve_operation(operation)
        self.n = len(arr)
        self.arr: List[Number] = list(arr)
        self.tree: List[Number] = [self.identity] * (4 * self.n)
        self._build(0, 0, self.n - 1)

    def _build(self, node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = self.arr[start]
            return
        mid = (start + end) // 2
        left = 2 * node + 1
        right = 2 * node + 2
        self._build(left, start, mid)
        self._build(right, mid + 1, end)
        self.tree[node] = self.op(self.tree[left], self.tree[right])

    def query(self, left: int, right: int) -> Number:
        """Aggregate of ``arr[left]`` through ``arr[right]`` inclusive. O(log n)."""
        self._check_range(left, right)
        return self._query(0, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> Number:
        if right < start or end < left:
            return self.identity
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        return self.op(
            self._query(left_child, start, mid, left, right),
            self._query(right_child, mid + 1, end, left, right),
        )

    def update(self, index: int, value: Number) -> None:
        """Set ``arr[index] = value`` and repair ancestors. O(log n)."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        self.arr[index] = value
        self._update(0, 0, self.n - 1, index, value)

    def _update(self, node: int, start: int, end: int, index: int, value: Number) -> None:
        if start == end:
            self.tree[node] = value
            return
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        if index <= mid:
            self._update(left_child, start, mid, index, value)
        else:
            self._update(right_child, mid + 1, end, index, value)
        self.tree[node] = self.op(self.tree[left_child], self.tree[right_child])

    def visualize(self) -> str:
        """Indent the tree as ``[start-end] = aggregate``, root first."""
        lines: List[str] = []

        def walk(node: int, start: int, end: int, depth: int) -> None:
            lines.append(f"{'  ' * depth}[{start}-{end}] = {self.tree[node]}")
            if start == end:
                return
            mid = (start + end) // 2
            walk(2 * node + 1, start, mid, depth + 1)
            walk(2 * node + 2, mid + 1, end, depth + 1)

        walk(0, 0, self.n - 1, 0)
        return "\n".join(lines)

    def _check_range(self, left: int, right: int) -> None:
        if left > right:
            raise ValueError(f"Empty range [{left}, {right}].")
        if left < 0 or right >= self.n:
            raise IndexError((left, right))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"SegmentTree(operation={self.operation!r}, n={self.n})"


class LazySegmentTree:
    """Segment tree with lazy range adds for sum, min, or max.

    ``update_range`` adds a constant to every element of an inclusive interval
    in O(log n). ``query_range`` is O(log n). A pending add is stored on the
    node and pushed to the children only when that node is opened. Space is
    O(n) for the aggregate array plus O(n) for the lazy tags.

    Range add distributes over sum (scaled by the segment length), min, and
    max (the tag is added once to the aggregate). Point assignment is also
    O(log n) and pushes tags along the edited path first.
    """

    def __init__(self, arr: Sequence[Number], operation: str = "sum") -> None:
        if operation not in ("sum", "min", "max"):
            raise ValueError("LazySegmentTree supports 'sum', 'min', and 'max'.")
        if len(arr) == 0:
            raise ValueError("Lazy segment tree needs at least one element.")
        self.operation = operation
        self.op, self.identity = resolve_operation(operation)
        self.n = len(arr)
        source = list(arr)
        self.tree: List[Number] = [self.identity] * (4 * self.n)
        self.lazy: List[Number] = [0] * (4 * self.n)
        self._build(0, 0, self.n - 1, source)

    def _build(self, node: int, start: int, end: int, source: Sequence[Number]) -> None:
        if start == end:
            self.tree[node] = source[start]
            return
        mid = (start + end) // 2
        left = 2 * node + 1
        right = 2 * node + 2
        self._build(left, start, mid, source)
        self._build(right, mid + 1, end, source)
        self.tree[node] = self.op(self.tree[left], self.tree[right])

    def _push(self, node: int, start: int, end: int) -> None:
        pending = self.lazy[node]
        if pending == 0:
            return
        if self.operation == "sum":
            self.tree[node] += pending * (end - start + 1)
        else:
            self.tree[node] += pending
        if start != end:
            self.lazy[2 * node + 1] += pending
            self.lazy[2 * node + 2] += pending
        self.lazy[node] = 0

    def update_range(self, left: int, right: int, value: Number) -> None:
        """Add ``value`` to every element in ``[left, right]``. O(log n)."""
        self._check_range(left, right)
        self._update_range(0, 0, self.n - 1, left, right, value)

    def _update_range(
        self,
        node: int,
        start: int,
        end: int,
        left: int,
        right: int,
        value: Number,
    ) -> None:
        self._push(node, start, end)
        if right < start or end < left:
            return
        if left <= start and end <= right:
            self.lazy[node] += value
            self._push(node, start, end)
            return
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        self._update_range(left_child, start, mid, left, right, value)
        self._update_range(right_child, mid + 1, end, left, right, value)
        self._push(left_child, start, mid)
        self._push(right_child, mid + 1, end)
        self.tree[node] = self.op(self.tree[left_child], self.tree[right_child])

    def update(self, index: int, value: Number) -> None:
        """Set one element to ``value``, pushing lazy tags on the way. O(log n)."""
        if not 0 <= index < self.n:
            raise IndexError(index)
        self._point(0, 0, self.n - 1, index, value)

    def _point(self, node: int, start: int, end: int, index: int, value: Number) -> None:
        self._push(node, start, end)
        if start == end:
            self.tree[node] = value
            return
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        if index <= mid:
            self._point(left_child, start, mid, index, value)
        else:
            self._point(right_child, mid + 1, end, index, value)
        self._push(left_child, start, mid)
        self._push(right_child, mid + 1, end)
        self.tree[node] = self.op(self.tree[left_child], self.tree[right_child])

    def query_range(self, left: int, right: int) -> Number:
        """Aggregate of the inclusive range. O(log n)."""
        self._check_range(left, right)
        return self._query_range(0, 0, self.n - 1, left, right)

    def query(self, left: int, right: int) -> Number:
        """Alias of ``query_range`` so callers can treat both trees alike."""
        return self.query_range(left, right)

    def _query_range(self, node: int, start: int, end: int, left: int, right: int) -> Number:
        self._push(node, start, end)
        if right < start or end < left:
            return self.identity
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        return self.op(
            self._query_range(left_child, start, mid, left, right),
            self._query_range(right_child, mid + 1, end, left, right),
        )

    def to_list(self) -> List[Number]:
        """Materialize every element. O(n log n)."""
        return [self.query_range(i, i) for i in range(self.n)]

    def visualize(self) -> str:
        """Push pending tags, then print ``[start-end] = aggregate``."""
        self._flush(0, 0, self.n - 1)
        lines: List[str] = []

        def walk(node: int, start: int, end: int, depth: int) -> None:
            lines.append(f"{'  ' * depth}[{start}-{end}] = {self.tree[node]}")
            if start == end:
                return
            mid = (start + end) // 2
            walk(2 * node + 1, start, mid, depth + 1)
            walk(2 * node + 2, mid + 1, end, depth + 1)

        walk(0, 0, self.n - 1, 0)
        return "\n".join(lines)

    def _flush(self, node: int, start: int, end: int) -> None:
        self._push(node, start, end)
        if start == end:
            return
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        self._flush(left_child, start, mid)
        self._flush(right_child, mid + 1, end)
        self.tree[node] = self.op(self.tree[left_child], self.tree[right_child])

    def _check_range(self, left: int, right: int) -> None:
        if left > right:
            raise ValueError(f"Empty range [{left}, {right}].")
        if left < 0 or right >= self.n:
            raise IndexError((left, right))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"LazySegmentTree(operation={self.operation!r}, n={self.n})"
