"""Van Emde Boas layout versus a level order array heap.

Section 12.6.3 stores a binary tree by recursively laying out the top half of
the levels, then each bottom subtree. A search that starts at the root and
walks down stays inside one contiguous chunk for as long as that chunk lasts,
instead of jumping to a new level of the heap array every step.

``CacheSimulator`` is a small fully associative cache with least recently used
replacement. It counts misses while the same logical node walk is replayed
through two different position maps: identity for the array heap, and the van
Emde Boas order for the recursive layout.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Sequence


class CacheSimulator:
    """Fully associative cache. One slot holds one aligned line of elements.

    ``line_size`` is how many consecutive array slots share a line. ``num_lines``
    is the capacity. An access is a hit when that line is already resident.
    Replacement drops the least recently used line. Counts are exact for this
    model. They are not hardware performance counters.
    """

    def __init__(self, line_size: int, num_lines: int) -> None:
        if line_size < 1 or num_lines < 1:
            raise ValueError("Cache line size and capacity must be positive.")
        self.line_size = line_size
        self.num_lines = num_lines
        self.resident: "OrderedDict[int, None]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, index: int) -> bool:
        """Touch the line containing ``index``. Returns True on a hit."""
        if index < 0:
            raise IndexError(index)
        line = index // self.line_size
        if line in self.resident:
            self.resident.move_to_end(line)
            self.hits += 1
            return True
        self.misses += 1
        self.resident[line] = None
        if len(self.resident) > self.num_lines:
            self.resident.popitem(last=False)
        return False

    def reset(self) -> None:
        self.resident.clear()
        self.hits = 0
        self.misses = 0

    def snapshot(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "accesses": self.hits + self.misses}


class VanEmdeBoasLayout:
    """Recursive layout of a binary heap's node ids.

    Node ids are the usual level order numbers: the left child of ``i`` is
    ``2i + 1`` and the right child is ``2i + 2``. ``build_layout(n)`` returns
    a permutation of ``0 .. n-1`` in the order those nodes should sit in
    memory. ``position_of(n)`` is the inverse map, from node id to memory slot.

    The height split follows section 12.6.3. The top ``ceil(height / 2)``
    levels are stored first. Each subtree that hangs off that top piece is
    stored next, and the same rule is applied inside every piece. Incomplete
    heaps simply skip ids that are ``>= n``.
    """

    @staticmethod
    def build_layout(n: int) -> List[int]:
        if n < 0:
            raise ValueError("n must be non negative.")
        if n == 0:
            return []
        layout: List[int] = []
        levels = int(math.floor(math.log2(n))) + 1

        def place(root: int, height: int) -> None:
            if root >= n or height <= 0:
                return
            if height == 1 or 2 * root + 1 >= n:
                layout.append(root)
                return
            top = (height + 1) // 2
            place(root, top)
            leaf_depth = top - 1
            first = root
            for _ in range(leaf_depth):
                first = 2 * first + 1
                if first >= n:
                    return
            bottom = height - top
            leaf_count = 1 << leaf_depth
            for offset in range(leaf_count):
                leaf = first + offset
                if leaf >= n:
                    break
                place(2 * leaf + 1, bottom)
                place(2 * leaf + 2, bottom)

        place(0, levels)
        return layout

    @staticmethod
    def position_of(n: int) -> List[int]:
        """``position[node_id]`` is the memory slot of that node. O(n) to build."""
        layout = VanEmdeBoasLayout.build_layout(n)
        if len(layout) != n or sorted(layout) != list(range(n)):
            raise RuntimeError("van Emde Boas layout is not a permutation of the heap ids.")
        position = [0] * n
        for slot, node in enumerate(layout):
            position[node] = slot
        return position


class ArrayHeapLayout:
    """Level order binary heap. Memory slot ``i`` stores node ``i``.

    Parent and child arithmetic is O(1), which is why heaps use this order.
    A root to leaf walk still touches a new region of the array at every
    level once the level is wider than a cache line.
    """

    @staticmethod
    def position_of(n: int) -> List[int]:
        if n < 0:
            raise ValueError("n must be non negative.")
        return list(range(n))

    @staticmethod
    def build_layout(n: int) -> List[int]:
        return ArrayHeapLayout.position_of(n)


def root_to_leaf_walks(n: int, walks: int, seed: int = 0) -> List[int]:
    """Node ids touched by ``walks`` random root to leaf descents, in order."""
    if n <= 0 or walks < 0:
        return []
    # A tiny deterministic generator keeps benchmarks repeatable without numpy.
    state = seed & 0xFFFFFFFF

    def rnd() -> int:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state

    touched: List[int] = []
    for _ in range(walks):
        node = 0
        touched.append(node)
        while 2 * node + 1 < n:
            right = 2 * node + 2
            if right < n and (rnd() & 1):
                node = right
            else:
                node = 2 * node + 1
            touched.append(node)
    return touched


def depth_first_walk(n: int) -> List[int]:
    """Preorder ids of the heap, left child before right child."""
    order: List[int] = []

    def visit(node: int) -> None:
        if node >= n:
            return
        order.append(node)
        visit(2 * node + 1)
        visit(2 * node + 2)

    if n > 0:
        visit(0)
    return order


def count_misses(node_ids: Sequence[int], position: Sequence[int], cache: CacheSimulator) -> int:
    """Replay logical node touches through a physical ``position`` map."""
    cache.reset()
    for node in node_ids:
        cache.access(position[node])
    return cache.misses


def compare_layouts(
    n: int,
    walks: int = 64,
    line_size: int = 8,
    num_lines: int = 4,
    seed: int = 0,
) -> Dict[str, int]:
    """Miss counts for the same walks under both layouts.

    Keys are ``veb_search_misses``, ``heap_search_misses``, ``veb_dfs_misses``,
    and ``heap_dfs_misses``.
    """
    search = root_to_leaf_walks(n, walks, seed)
    dfs = depth_first_walk(n)
    veb = VanEmdeBoasLayout.position_of(n)
    heap = ArrayHeapLayout.position_of(n)
    cache = CacheSimulator(line_size, num_lines)
    return {
        "n": n,
        "walks": walks,
        "line_size": line_size,
        "num_lines": num_lines,
        "veb_search_misses": count_misses(search, veb, cache),
        "heap_search_misses": count_misses(search, heap, cache),
        "veb_dfs_misses": count_misses(dfs, veb, cache),
        "heap_dfs_misses": count_misses(dfs, heap, cache),
        "search_accesses": len(search),
        "dfs_accesses": len(dfs),
    }
