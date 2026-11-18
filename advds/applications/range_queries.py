"""One front door for range queries.

Section 12.7.3 picks a Fenwick tree when the problem is point updates and
sums, a lazy segment tree when the updates cover a whole interval, and a
plain segment tree for min, max, or gcd. This module adds the time travel
case from section 12.4: the same calls read or branch an older version of a
persistent segment tree.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..trees import FenwickTree, LazySegmentTree, SegmentTree, VersionStore


class RangeQuerySolver:
    """Chooses a backend and forwards query and update calls.

    Parameters
    ----------
    arr:
        Initial values.
    query_type:
        ``sum``, ``min``, ``max``, or ``gcd``.
    update_type:
        ``point`` or ``range``. Range updates are additive and require a
        lazy segment tree, so ``query_type`` must be sum, min, or max.
    time_travel:
        Keep every point update as a persistent version. Range updates and
        time travel are separate tools in the chapter, so they are not
        combined here.
    """

    def __init__(
        self,
        arr,
        query_type: str = "sum",
        update_type: str = "point",
        time_travel: bool = False,
    ) -> None:
        self.arr = list(arr)
        self.n = len(self.arr)
        if self.n == 0:
            raise ValueError("RangeQuerySolver needs at least one element.")
        self.query_type = query_type
        self.update_type = update_type
        self.time_travel = time_travel
        self.current_version = 0
        self.history: Optional[VersionStore] = None
        self.ds = None

        if time_travel:
            if update_type == "range":
                raise ValueError("Time travel here uses point updates on a persistent segment tree.")
            self.history = VersionStore(self.arr, operation=query_type)
            self.backend = "persistent"
            return

        if update_type == "range":
            if query_type not in ("sum", "min", "max"):
                raise ValueError("Range updates support sum, min, and max.")
            self.ds = LazySegmentTree(self.arr, operation=query_type)
            self.backend = "lazy_segment"
        elif update_type == "point" and query_type == "sum":
            self.ds = FenwickTree.from_array([int(v) for v in self.arr])
            self.backend = "fenwick"
        elif update_type == "point":
            self.ds = SegmentTree(self.arr, operation=query_type)
            self.backend = "segment"
        else:
            raise ValueError("update_type must be 'point' or 'range'.")

    def query(self, left: int, right: int, version: int | None = None):
        """Aggregate of the inclusive range. O(log n), or O(log n log m) is not used here.

        ``version`` is only meaningful for a time travel solver. Omit it to
        read the latest version.
        """
        if self.backend == "fenwick":
            if version is not None:
                raise ValueError("Fenwick backend has no history. Construct with time_travel=True.")
            return self.ds.range_sum(left, right)
        if self.backend == "lazy_segment":
            if version is not None:
                raise ValueError("Lazy segment tree has no history.")
            return self.ds.query_range(left, right)
        if self.backend == "segment":
            if version is not None:
                raise ValueError("Mutable segment tree has no history.")
            return self.ds.query(left, right)
        assert self.history is not None
        if version is None:
            version = self.current_version
        return self.history.query(left, right, version)

    def update(
        self,
        index: int | None = None,
        value=None,
        left: int | None = None,
        right: int | None = None,
        version: int | None = None,
    ):
        """Point set, or a range add when ``left`` and ``right`` are given.

        A time travel point update returns the new version id. Other backends
        return None. Branching is ``update(index, value, version=old_id)``.
        """
        if left is not None or right is not None:
            if left is None or right is None or value is None:
                raise ValueError("Range update needs left, right, and value.")
            if self.backend != "lazy_segment":
                raise ValueError("Range updates require update_type='range'.")
            self.ds.update_range(left, right, value)
            return None

        if index is None or value is None:
            raise ValueError("Point update needs index and value.")

        if self.backend == "fenwick":
            current = self.ds.range_sum(index, index)
            self.ds.update(index, int(value) - current)
            return None
        if self.backend == "lazy_segment":
            self.ds.update(index, value)
            return None
        if self.backend == "segment":
            self.ds.update(index, value)
            return None

        assert self.history is not None
        base = self.current_version if version is None else version
        self.current_version = self.history.update(index, value, version=base)
        return self.current_version

    def diff(self, left_version: int, right_version: int):
        """Leaf differences between two stored versions. Persistent backend only."""
        if self.history is None:
            raise ValueError("diff requires time_travel=True.")
        return self.history.diff(left_version, right_version)

    def get_structure_info(self) -> Dict[str, str]:
        """Short description of the chosen backend, in the style of section 12.7.3."""
        info = {
            "backend": self.backend,
            "size": str(self.n),
            "query_type": self.query_type,
            "update_type": "point" if self.time_travel else self.update_type,
        }
        if self.backend == "fenwick":
            info["space"] = f"O(n) = {self.n} elements"
            info["query_time"] = "O(log n)"
            info["update_time"] = "O(log n)"
        elif self.backend == "segment":
            info["space"] = f"O(4n) = {4 * self.n} slots"
            info["query_time"] = "O(log n)"
            info["update_time"] = "O(log n)"
        elif self.backend == "lazy_segment":
            info["space"] = f"O(4n) = {4 * self.n} slots plus lazy tags"
            info["query_time"] = "O(log n)"
            info["update_time"] = "O(log n) per range"
        else:
            versions = 0 if self.history is None else len(self.history)
            info["space"] = f"O(n + k log n) nodes across {versions} versions"
            info["query_time"] = "O(log n) on any stored version"
            info["update_time"] = "O(log n) new nodes per point update"
        return info

    def __repr__(self) -> str:
        return (
            f"RangeQuerySolver(backend={self.backend!r}, query_type={self.query_type!r}, n={self.n})"
        )
