"""Range query solver and the tiny version control store."""

import random

from advds import (
    DNAIndex,
    FenwickTree,
    LazySegmentTree,
    PersistentSegmentTree,
    RangeQuerySolver,
    SegmentTree,
    VersionControl,
    WaveletTree,
)
from advds.applications import RangeQuerySolver as SolverFromPackage


def test_public_api_imports():
    assert SolverFromPackage is RangeQuerySolver
    assert SegmentTree([1], "sum").query(0, 0) == 1
    assert FenwickTree.from_array([2, 3]).range_sum(0, 1) == 5
    assert PersistentSegmentTree([1, 1]).query(0, 1) == 2
    assert LazySegmentTree([1, 1]).query_range(0, 1) == 2
    assert WaveletTree("ab").access(1) == "b"
    assert DNAIndex("A").count_base("A", 0, 0) == 1


def test_solver_picks_the_chapter_backends():
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    sums = RangeQuerySolver(arr, query_type="sum", update_type="point")
    assert sums.backend == "fenwick"
    assert sums.query(2, 5) == sum(arr[2:6])
    sums.update(2, 0)
    assert sums.query(2, 5) == 0 + 7 + 9 + 11

    mins = RangeQuerySolver(arr, query_type="min", update_type="point")
    assert mins.backend == "segment"
    assert mins.query(2, 5) == min(arr[2:6])

    ranges = RangeQuerySolver(arr, query_type="sum", update_type="range")
    assert ranges.backend == "lazy_segment"
    ranges.update(left=2, right=5, value=10)
    assert ranges.query(2, 5) == sum(arr[2:6]) + 40
    info = ranges.get_structure_info()
    assert info["backend"] == "lazy_segment"
    assert "O(log n)" in info["query_time"]


def test_time_travel_queries():
    arr = [4, 1, 6, 2]
    solver = RangeQuerySolver(arr, query_type="sum", time_travel=True)
    assert solver.backend == "persistent"
    assert solver.query(0, 3, version=0) == 13
    version = solver.update(1, 10)
    assert version == 1
    assert solver.query(0, 3, version=0) == 13
    assert solver.query(0, 3) == 4 + 10 + 6 + 2
    branch = solver.update(3, 0, version=0)
    assert solver.query(0, 3, version=branch) == 4 + 1 + 6 + 0
    assert solver.diff(0, branch) == [(3, 2, 0)]


def test_version_control_commit_diff_and_graph():
    repo = VersionControl()
    first = repo.commit("initial", {"main.py": "print(1)\n", "readme": "hello"})
    second = repo.commit("tweak print", {"main.py": "print(2)\n", "readme": "hello"})
    branch = repo.commit("side note", {"main.py": "print(1)\n", "readme": "hello!"}, parent=first)
    assert repo.checkout(first)["main.py"] == "print(1)\n"
    assert repo.checkout(second)["main.py"] == "print(2)\n"
    # The branch does not see the second commit, and readme changed one character.
    diff = {item["file"]: item for item in repo.diff(first, branch)}
    assert diff["main.py"]["status"] == "unchanged"
    # "hello" and "hello!" have different lengths, so the file is replaced.
    assert diff["readme"]["status"] == "replaced"
    assert diff["readme"]["old_len"] == 5
    assert diff["readme"]["new_len"] == 6
    same_len = repo.diff(first, second)
    main = next(item for item in same_len if item["file"] == "main.py")
    assert main["status"] == "modified"
    assert (6, "1", "2") in main["changes"]
    readme = next(item for item in same_len if item["file"] == "readme")
    assert readme["status"] == "unchanged"
    graph = repo.history_graph()
    assert "c0" in graph and "c2" in graph
    assert repo.head == branch
    assert len(repo.log()) == 3


def test_random_solver_sum_matches_list():
    rng = random.Random(19)
    arr = [rng.randint(0, 20) for _ in range(30)]
    solver = RangeQuerySolver(arr, "sum", "point")
    for _ in range(20):
        index = rng.randrange(30)
        value = rng.randint(0, 20)
        arr[index] = value
        solver.update(index, value)
        left = rng.randrange(30)
        right = rng.randint(left, 29)
        assert solver.query(left, right) == sum(arr[left : right + 1])
