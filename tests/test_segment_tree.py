"""Segment tree and lazy propagation against brute force."""

import math
import random

import pytest

from advds.trees import LazySegmentTree, SegmentTree


def test_book_sum_figure():
    arr = [1, 3, 5, 7, 9, 11]
    tree = SegmentTree(arr, operation="sum")
    assert tree.query(1, 4) == 24
    assert "[0-5] = 36" in tree.visualize()
    tree.update(2, 10)
    assert tree.query(1, 4) == 29
    # After the update the root is 1+3+10+7+9+11 = 41.
    assert tree.query(0, 5) == 41
    assert "[0-5] = 41" in tree.visualize()


def test_min_max_gcd_match_python():
    arr = [12, 7, 5, 15, 3, 9, 11, 18]
    left, right = 2, 6
    assert SegmentTree(arr, "min").query(left, right) == min(arr[left : right + 1])
    assert SegmentTree(arr, "max").query(left, right) == max(arr[left : right + 1])
    expected = 0
    for value in arr[left : right + 1]:
        expected = math.gcd(expected, value)
    assert SegmentTree(arr, "gcd").query(left, right) == expected


def test_random_point_updates_against_brute():
    rng = random.Random(12)
    for operation in ("sum", "min", "max", "gcd"):
        for n in (1, 2, 7, 64, 200):
            if operation == "gcd":
                arr = [rng.randint(0, 40) for _ in range(n)]
            else:
                arr = [rng.randint(-20, 40) for _ in range(n)]
            tree = SegmentTree(arr, operation)
            for _ in range(40):
                left = rng.randint(0, n - 1)
                right = rng.randint(left, n - 1)
                window = arr[left : right + 1]
                if operation == "sum":
                    assert tree.query(left, right) == sum(window)
                elif operation == "min":
                    assert tree.query(left, right) == min(window)
                elif operation == "max":
                    assert tree.query(left, right) == max(window)
                else:
                    got = window[0]
                    for value in window[1:]:
                        got = math.gcd(got, value)
                    assert tree.query(left, right) == got
                index = rng.randint(0, n - 1)
                value = rng.randint(0, 40)
                arr[index] = value
                tree.update(index, value)


def test_lazy_book_example():
    arr = [1, 2, 3, 4, 5, 6, 7, 8]
    tree = LazySegmentTree(arr)
    assert tree.query_range(2, 5) == 18
    tree.update_range(1, 4, 10)
    assert tree.query_range(2, 5) == 48
    assert tree.to_list() == [1, 12, 13, 14, 15, 6, 7, 8]


def test_lazy_random_sum_min_max():
    rng = random.Random(13)
    for operation in ("sum", "min", "max"):
        for n in (1, 5, 17, 80):
            arr = [rng.randint(-5, 15) for _ in range(n)]
            tree = LazySegmentTree(arr, operation)
            for _ in range(30):
                if rng.random() < 0.5:
                    left = rng.randint(0, n - 1)
                    right = rng.randint(left, n - 1)
                    delta = rng.randint(-3, 6)
                    for index in range(left, right + 1):
                        arr[index] += delta
                    tree.update_range(left, right, delta)
                else:
                    index = rng.randint(0, n - 1)
                    value = rng.randint(-5, 15)
                    arr[index] = value
                    tree.update(index, value)
                left = rng.randint(0, n - 1)
                right = rng.randint(left, n - 1)
                window = arr[left : right + 1]
                if operation == "sum":
                    assert tree.query(left, right) == sum(window)
                elif operation == "min":
                    assert tree.query(left, right) == min(window)
                else:
                    assert tree.query(left, right) == max(window)


def test_rejects_bad_ranges():
    tree = SegmentTree([1, 2, 3])
    with pytest.raises(ValueError):
        tree.query(2, 1)
    with pytest.raises(IndexError):
        tree.update(3, 1)
    with pytest.raises(ValueError):
        SegmentTree([], "sum")
    with pytest.raises(ValueError):
        LazySegmentTree([1, 2], "gcd")
