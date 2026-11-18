"""Cache oblivious numerics and the van Emde Boas layout."""

import random

import numpy as np
import pytest

from advds.cache_oblivious import (
    ArrayHeapLayout,
    CacheObliviousOps,
    CacheSimulator,
    VanEmdeBoasLayout,
    compare_layouts,
)


def test_transpose_and_multiply_match_numpy():
    rng = np.random.default_rng(5)
    for rows, cols in ((1, 1), (1, 5), (4, 1), (3, 5), (8, 8), (7, 13)):
        matrix = rng.normal(size=(rows, cols))
        naive = CacheObliviousOps.matrix_transpose_naive(matrix)
        clever = CacheObliviousOps.matrix_transpose_cache_oblivious(matrix, leaf=4)
        assert np.allclose(naive, matrix.T)
        assert np.allclose(clever, matrix.T)
    for n, mid, p in ((1, 1, 1), (2, 3, 2), (5, 4, 6), (9, 9, 9)):
        left = rng.normal(size=(n, mid))
        right = rng.normal(size=(mid, p))
        naive = CacheObliviousOps.matrix_multiply_naive(left, right)
        clever = CacheObliviousOps.matrix_multiply_cache_oblivious(left, right, leaf=8)
        assert np.allclose(naive, left @ right)
        assert np.allclose(clever, left @ right)
    with pytest.raises(ValueError):
        CacheObliviousOps.matrix_multiply_naive(np.zeros((2, 2)), np.zeros((3, 2)))


def test_recursive_sum_matches_row_major():
    matrix = np.arange(30, dtype=float).reshape(5, 6)
    assert CacheObliviousOps.sum_row_major(matrix) == pytest.approx(matrix.sum())
    assert CacheObliviousOps.sum_recursive(matrix, leaf=3) == pytest.approx(matrix.sum())


def test_veb_layout_is_a_permutation():
    for n in list(range(0, 65)) + [100, 127, 128, 255, 1000]:
        layout = VanEmdeBoasLayout.build_layout(n)
        assert sorted(layout) == list(range(n))
        if n == 0:
            continue
        position = VanEmdeBoasLayout.position_of(n)
        assert [position[node] for node in layout] == list(range(n))
        assert ArrayHeapLayout.build_layout(n) == list(range(n))


def test_known_small_veb_order():
    # Height 4 perfect tree: top of 2 levels, then four bottom trees of 3 nodes.
    assert VanEmdeBoasLayout.build_layout(15) == [
        0, 1, 2, 3, 7, 8, 4, 9, 10, 5, 11, 12, 6, 13, 14,
    ]


def test_cache_simulator_and_layout_misses():
    cache = CacheSimulator(line_size=4, num_lines=1)
    assert cache.access(0) is False
    assert cache.access(3) is True
    assert cache.access(4) is False
    assert cache.access(1) is False  # line 0 was evicted
    assert cache.misses == 3

    stats = compare_layouts(4095, walks=40, line_size=8, num_lines=4, seed=2)
    assert stats["veb_search_misses"] > 0
    assert stats["heap_search_misses"] > 0
    # Depth first order jumps across the heap array. The recursive layout
    # keeps each bottom subtree together, so the same walk misses less.
    assert stats["veb_dfs_misses"] < stats["heap_dfs_misses"]


def test_search_misses_not_worse_on_a_wide_tree():
    stats = compare_layouts(8191, walks=80, line_size=4, num_lines=8, seed=7)
    assert stats["veb_search_misses"] <= stats["heap_search_misses"]
