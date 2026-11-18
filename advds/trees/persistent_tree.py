"""Persistent arrays and segment trees by path copying.

Amakobe, section 12.4, keeps every version. An update allocates a new node
only along the path from the root to the edited leaf and reuses the sibling
pointers, so the versions form a directed acyclic graph of shared nodes.
One update costs O(log n) time and O(log n) extra nodes. Older roots stay
valid because nodes are never rewritten.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .segment_tree import resolve_operation


class _ArrayNode:
    __slots__ = ("value", "left", "right")

    def __init__(self, value: int = 0, left: "_ArrayNode | None" = None, right: "_ArrayNode | None" = None) -> None:
        self.value = value
        self.left = left
        self.right = right


class PersistentArray:
    """Persistent array. ``set`` returns a new version and leaves this one alone.

    Reads and writes are O(log n). A write allocates O(log n) nodes.
    """

    def __init__(self, arr: Sequence[int] | None = None, size: int | None = None) -> None:
        if arr is not None:
            self.size = len(arr)
            self.root: Optional[_ArrayNode] = self._build(list(arr), 0, self.size - 1) if self.size else None
        else:
            self.size = size or 0
            self.root = self._build_empty(0, self.size - 1) if self.size else None

    def _build(self, arr: Sequence[int], left: int, right: int) -> _ArrayNode:
        if left == right:
            return _ArrayNode(value=arr[left])
        mid = (left + right) // 2
        return _ArrayNode(left=self._build(arr, left, mid), right=self._build(arr, mid + 1, right))

    def _build_empty(self, left: int, right: int) -> _ArrayNode:
        if left == right:
            return _ArrayNode(value=0)
        mid = (left + right) // 2
        return _ArrayNode(left=self._build_empty(left, mid), right=self._build_empty(mid + 1, right))

    def get(self, index: int) -> int:
        """Value at ``index``. O(log n)."""
        if self.root is None or not 0 <= index < self.size:
            raise IndexError(index)
        return self._get(self.root, 0, self.size - 1, index)

    def _get(self, node: _ArrayNode, left: int, right: int, index: int) -> int:
        if left == right:
            return node.value
        mid = (left + right) // 2
        if index <= mid:
            assert node.left is not None
            return self._get(node.left, left, mid, index)
        assert node.right is not None
        return self._get(node.right, mid + 1, right, index)

    def set(self, index: int, value: int) -> "PersistentArray":
        """Return a new array with ``index`` set to ``value``. O(log n) nodes."""
        if self.root is None or not 0 <= index < self.size:
            raise IndexError(index)
        created = PersistentArray(size=self.size)
        created.root = self._set(self.root, 0, self.size - 1, index, value)
        return created

    def _set(self, node: _ArrayNode, left: int, right: int, index: int, value: int) -> _ArrayNode:
        if left == right:
            return _ArrayNode(value=value)
        mid = (left + right) // 2
        if index <= mid:
            assert node.left is not None
            return _ArrayNode(left=self._set(node.left, left, mid, index, value), right=node.right)
        assert node.right is not None
        return _ArrayNode(left=node.left, right=self._set(node.right, mid + 1, right, index, value))

    def to_list(self) -> List[int]:
        """Copy the array out in order. O(n)."""
        result: List[int] = []
        self._to_list(self.root, result)
        return result

    def _to_list(self, node: Optional[_ArrayNode], result: List[int]) -> None:
        if node is None:
            return
        if node.left is None and node.right is None:
            result.append(node.value)
            return
        self._to_list(node.left, result)
        self._to_list(node.right, result)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"PersistentArray(size={self.size})"


class _SegNode:
    __slots__ = ("value", "left", "right")

    def __init__(
        self,
        value: int | float = 0,
        left: "_SegNode | None" = None,
        right: "_SegNode | None" = None,
    ) -> None:
        self.value = value
        self.left = left
        self.right = right


class PersistentSegmentTree:
    """One version of a persistent segment tree.

    ``update`` returns a new tree. The previous object still answers queries
    against the nodes it already points at. Sum, min, max, and gcd are the
    same associative operations as the mutable segment tree. Query is O(log n).
    A version shares every node that the edit did not touch.
    """

    def __init__(
        self,
        arr: Sequence[int | float] | None = None,
        n: int | None = None,
        operation: str = "sum",
        _root: _SegNode | None = None,
    ) -> None:
        self.operation = operation
        self.op, self.identity = resolve_operation(operation)
        if _root is not None:
            if n is None:
                raise ValueError("n is required when passing a root.")
            self.n = n
            self.root: Optional[_SegNode] = _root
            return
        if arr is None:
            arr = [0] * (n or 0)
        self.n = len(arr)
        self.root = self._build(list(arr), 0, self.n - 1) if self.n else None

    def _build(self, arr: Sequence[int | float], left: int, right: int) -> _SegNode:
        if left == right:
            return _SegNode(value=arr[left])
        mid = (left + right) // 2
        left_child = self._build(arr, left, mid)
        right_child = self._build(arr, mid + 1, right)
        return _SegNode(
            value=self.op(left_child.value, right_child.value),
            left=left_child,
            right=right_child,
        )

    def query(self, left: int, right: int) -> int | float:
        """Aggregate of the inclusive range on this version. O(log n)."""
        if self.n == 0 or self.root is None:
            raise ValueError("Empty tree.")
        if left > right or left < 0 or right >= self.n:
            raise IndexError((left, right))
        return self._query(self.root, 0, self.n - 1, left, right)

    def _query(self, node: _SegNode, left: int, right: int, query_left: int, query_right: int) -> int | float:
        if query_right < left or right < query_left:
            return self.identity
        if query_left <= left and right <= query_right:
            return node.value
        mid = (left + right) // 2
        assert node.left is not None and node.right is not None
        return self.op(
            self._query(node.left, left, mid, query_left, query_right),
            self._query(node.right, mid + 1, right, query_left, query_right),
        )

    def update(self, index: int, value: int | float) -> "PersistentSegmentTree":
        """Return a new version with ``index`` set to ``value``. O(log n) new nodes."""
        if self.root is None or not 0 <= index < self.n:
            raise IndexError(index)
        root = self._update(self.root, 0, self.n - 1, index, value)
        return PersistentSegmentTree(n=self.n, operation=self.operation, _root=root)

    def _update(self, node: _SegNode, left: int, right: int, index: int, value: int | float) -> _SegNode:
        if left == right:
            return _SegNode(value=value)
        mid = (left + right) // 2
        assert node.left is not None and node.right is not None
        if index <= mid:
            new_left = self._update(node.left, left, mid, index, value)
            new_right = node.right
        else:
            new_left = node.left
            new_right = self._update(node.right, mid + 1, right, index, value)
        return _SegNode(value=self.op(new_left.value, new_right.value), left=new_left, right=new_right)

    def to_list(self) -> List[int | float]:
        """Read every leaf. O(n)."""
        if self.root is None:
            return []
        out: List[int | float] = [0] * self.n

        def walk(node: _SegNode, left: int, right: int) -> None:
            if left == right:
                out[left] = node.value
                return
            mid = (left + right) // 2
            assert node.left is not None and node.right is not None
            walk(node.left, left, mid)
            walk(node.right, mid + 1, right)

        walk(self.root, 0, self.n - 1)
        return out

    def node_count(self) -> int:
        """Nodes reachable from this root, counting a shared node once per walk."""
        seen: set[int] = set()

        def walk(node: Optional[_SegNode]) -> None:
            if node is None or id(node) in seen:
                return
            seen.add(id(node))
            walk(node.left)
            walk(node.right)

        walk(self.root)
        return len(seen)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"PersistentSegmentTree(operation={self.operation!r}, n={self.n})"


def count_unique_nodes(trees: Sequence[PersistentSegmentTree]) -> int:
    """How many distinct node objects the versions share between them."""
    seen: set[int] = set()

    def walk(node: Optional[_SegNode]) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        walk(node.left)
        walk(node.right)

    for tree in trees:
        walk(tree.root)
    return len(seen)


class VersionStore:
    """Append only history of persistent segment tree roots.

    Version 0 is the array you pass in. ``update`` always appends a new
    version (it can branch off an older id) and returns that id. Every id
    stays queryable. Space across k updates is O(n + k log n) nodes.
    """

    def __init__(self, arr: Sequence[int | float], operation: str = "sum") -> None:
        self.operation = operation
        self.versions: List[PersistentSegmentTree] = [PersistentSegmentTree(arr, operation=operation)]

    def update(self, index: int, value: int | float, version: int | None = None) -> int:
        """Branch from ``version`` (default: latest) and return the new id. O(log n)."""
        source = self._pick(version)
        self.versions.append(source.update(index, value))
        return len(self.versions) - 1

    def query(self, left: int, right: int, version: int | None = None) -> int | float:
        """Range aggregate on one version. O(log n)."""
        return self._pick(version).query(left, right)

    def get_list(self, version: int | None = None) -> List[int | float]:
        return self._pick(version).to_list()

    def _pick(self, version: int | None) -> PersistentSegmentTree:
        if version is None:
            return self.versions[-1]
        if not 0 <= version < len(self.versions):
            raise IndexError(version)
        return self.versions[version]

    def __len__(self) -> int:
        return len(self.versions)

    @property
    def latest(self) -> int:
        return len(self.versions) - 1

    def unique_nodes(self) -> int:
        return count_unique_nodes(self.versions)

    def nodes_if_copied(self) -> int:
        """Node count if each version had been a private tree with no sharing."""
        return sum(tree.node_count() for tree in self.versions)

    def diff(self, left_version: int, right_version: int) -> List[tuple]:
        """Indices whose leaves differ. Shared subtrees are skipped.

        When the edited regions are few and the versions share nodes, the
        walk is O(k log n) for k differences rather than O(n).
        """
        left = self._pick(left_version)
        right = self._pick(right_version)
        if left.n != right.n or left.root is None or right.root is None:
            raise ValueError("Versions do not cover the same array.")
        changes: List[tuple] = []

        def walk(a: _SegNode, b: _SegNode, lo: int, hi: int) -> None:
            if a is b:
                return
            if lo == hi:
                if a.value != b.value:
                    changes.append((lo, a.value, b.value))
                return
            mid = (lo + hi) // 2
            assert a.left is not None and a.right is not None
            assert b.left is not None and b.right is not None
            walk(a.left, b.left, lo, mid)
            walk(a.right, b.right, mid + 1, hi)

        walk(left.root, right.root, 0, left.n - 1)
        return changes

    def __repr__(self) -> str:
        return f"VersionStore(versions={len(self.versions)}, operation={self.operation!r})"
