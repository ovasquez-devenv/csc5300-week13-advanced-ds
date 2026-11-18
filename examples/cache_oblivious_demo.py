"""Small transpose check and a van Emde Boas layout.

Run from the repository root:

    python examples/cache_oblivious_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from advds.cache_oblivious import CacheObliviousOps, VanEmdeBoasLayout, compare_layouts


def main() -> None:
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    naive = CacheObliviousOps.matrix_transpose_naive(matrix)
    recursive = CacheObliviousOps.matrix_transpose_cache_oblivious(matrix, leaf=4)
    print("Input:\n", matrix)
    print("Recursive transpose matches numpy:", np.allclose(recursive, matrix.T))
    print("Naive transpose matches too:", np.allclose(naive, recursive))

    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = np.array([[5.0, 6.0], [7.0, 8.0]])
    product = CacheObliviousOps.matrix_multiply_cache_oblivious(left, right, leaf=2)
    print("Product:\n", product)

    print("\nvan Emde Boas order for 15 nodes:")
    print(" ", VanEmdeBoasLayout.build_layout(15))
    stats = compare_layouts(1023, walks=40, line_size=8, num_lines=4, seed=1)
    print(
        "Search misses on 1023 nodes:",
        f"van Emde Boas {stats['veb_search_misses']}, array heap {stats['heap_search_misses']}",
    )


if __name__ == "__main__":
    main()
