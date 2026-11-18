"""Three range query setups from Amakobe, section 12.7.3, plus a time travel query.

Run from the repository root:

    python examples/range_queries_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advds import RangeQuerySolver


def main() -> None:
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print("Array:", arr)

    print("\nPoint updates and sums use a Fenwick tree.")
    sums = RangeQuerySolver(arr, query_type="sum", update_type="point")
    print(" ", sums.get_structure_info())
    print("  sum(2, 5) =", sums.query(2, 5))

    print("\nPoint updates and minimums use a segment tree.")
    mins = RangeQuerySolver(arr, query_type="min", update_type="point")
    print(" ", mins.get_structure_info())
    print("  min(2, 5) =", mins.query(2, 5))

    print("\nRange adds use a lazy segment tree.")
    ranges = RangeQuerySolver(arr, query_type="sum", update_type="range")
    print(" ", ranges.get_structure_info())
    ranges.update(left=2, right=5, value=10)
    print("  after adding 10 on [2, 5], sum(2, 5) =", ranges.query(2, 5))

    print("\nTime travel keeps every point update.")
    history = RangeQuerySolver([5, 3, 7, 2, 9, 1, 4, 8], query_type="sum", time_travel=True)
    print("  version 0 sum(1, 5) =", history.query(1, 5, version=0))
    v1 = history.update(2, 15)
    v2 = history.update(5, 10)
    print(f"  version {v1} sum(1, 5) =", history.query(1, 5, version=v1))
    print(f"  version {v2} sum(1, 5) =", history.query(1, 5, version=v2))
    print("  version 0 is unchanged:", history.query(1, 5, version=0))


if __name__ == "__main__":
    main()
