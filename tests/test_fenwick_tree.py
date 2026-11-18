"""Fenwick tree prefix, range, and rectangle sums."""

import random

import pytest

from advds.trees import FenwickTree, FenwickTree2D


def test_lowbit_and_prefix_table():
    assert FenwickTree.lowbit(1) == 1
    assert FenwickTree.lowbit(6) == 2
    assert FenwickTree.lowbit(12) == 4
    arr = [1, 2, 3, 4, 5, 6, 7, 8]
    tree = FenwickTree.from_array(arr)
    for index in range(len(arr)):
        assert tree.prefix_sum(index) == sum(arr[: index + 1])
    assert tree.range_sum(2, 5) == 18
    assert tree.prefix_sum(-1) == 0


def test_point_updates_match_list():
    rng = random.Random(3)
    for n in (1, 8, 50, 300):
        arr = [rng.randint(-10, 10) for _ in range(n)]
        tree = FenwickTree.from_array(arr)
        for _ in range(60):
            index = rng.randint(0, n - 1)
            if rng.random() < 0.5:
                delta = rng.randint(-5, 5)
                arr[index] += delta
                tree.update(index, delta)
            else:
                value = rng.randint(-5, 5)
                arr[index] = value
                tree.set(index, value)
            left = rng.randint(0, n - 1)
            right = rng.randint(left, n - 1)
            assert tree.range_sum(left, right) == sum(arr[left : right + 1])


def test_rectangle_against_brute():
    rng = random.Random(4)
    matrix = [[rng.randint(0, 9) for _ in range(6)] for _ in range(5)]
    tree = FenwickTree2D.from_matrix(matrix)
    r1, c1, r2, c2 = 1, 1, 2, 2
    expected = sum(matrix[i][j] for i in range(r1, r2 + 1) for j in range(c1, c2 + 1))
    assert tree.rectangle_sum(r1, c1, r2, c2) == expected
    for _ in range(40):
        row = rng.randint(0, 4)
        col = rng.randint(0, 5)
        delta = rng.randint(-3, 3)
        matrix[row][col] += delta
        tree.update(row, col, delta)
        r1 = rng.randint(0, 4)
        r2 = rng.randint(r1, 4)
        c1 = rng.randint(0, 5)
        c2 = rng.randint(c1, 5)
        expected = sum(matrix[i][j] for i in range(r1, r2 + 1) for j in range(c1, c2 + 1))
        assert tree.rectangle_sum(r1, c1, r2, c2) == expected


def test_bounds():
    tree = FenwickTree.from_array([4])
    with pytest.raises(IndexError):
        tree.update(1, 1)
    with pytest.raises(ValueError):
        tree.range_sum(1, 0)
