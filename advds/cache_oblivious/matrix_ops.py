"""Cache oblivious matrix transpose and multiplication.

Amakobe, section 12.6, contrasts a tuned blocking factor (which depends on a
particular cache) with divide and conquer that stops at a small base case.
The recursion produces the same optimal cache miss bound for every cache that
is large enough to hold the base case, which is the cache oblivious claim.
Both routines here are the textbook recurrences. Naive loops are included so
a benchmark can compare wall clock time and a simulated cache.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class CacheObliviousOps:
    """Naive and cache oblivious dense linear algebra on NumPy arrays.

    Time is the usual arithmetic cost: O(n * m) to transpose and O(n * m * p)
    to multiply. The cache miss bounds quoted by Amakobe are O(n^2 / B) for a
    naive transpose that streams one matrix by columns, and
    O(n^2 / B + n^2 / sqrt(M)) for the recursive transpose, where B is the
    line length and M is the cache capacity.
    """

    @staticmethod
    def matrix_transpose_naive(matrix: np.ndarray) -> np.ndarray:
        """Transpose by writing ``B[j, i] = A[i, j]``. O(n * m) time.

        Column writes on ``B`` jump by a full row, which is the access the
        chapter calls out as cache unfriendly.
        """
        source = np.asarray(matrix, dtype=float)
        rows, cols = source.shape
        out = np.zeros((cols, rows), dtype=float)
        for i in range(rows):
            for j in range(cols):
                out[j, i] = source[i, j]
        return out

    @staticmethod
    def matrix_transpose_cache_oblivious(
        matrix: np.ndarray,
        out: Optional[np.ndarray] = None,
        i0: int = 0,
        j0: int = 0,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        leaf: int = 64,
    ) -> np.ndarray:
        """Recursive transpose. Splits the longer side until ``rows * cols <= leaf``.

        ``leaf`` is the base case threshold from section 12.6.2 (the text uses
        64). O(n * m) arithmetic. The recursion depth is O(log (n + m)).
        """
        source = np.asarray(matrix, dtype=float)
        if out is None:
            n_rows, n_cols = source.shape
            out = np.zeros((n_cols, n_rows), dtype=float)
            return CacheObliviousOps.matrix_transpose_cache_oblivious(
                source, out, 0, 0, n_rows, n_cols, leaf
            )
        assert rows is not None and cols is not None
        if rows <= 0 or cols <= 0:
            return out
        if rows * cols <= leaf or rows == 1 or cols == 1:
            for i in range(rows):
                for j in range(cols):
                    out[j0 + j, i0 + i] = source[i0 + i, j0 + j]
            return out
        if rows >= cols:
            mid = rows // 2
            CacheObliviousOps.matrix_transpose_cache_oblivious(source, out, i0, j0, mid, cols, leaf)
            CacheObliviousOps.matrix_transpose_cache_oblivious(
                source, out, i0 + mid, j0, rows - mid, cols, leaf
            )
        else:
            mid = cols // 2
            CacheObliviousOps.matrix_transpose_cache_oblivious(source, out, i0, j0, rows, mid, leaf)
            CacheObliviousOps.matrix_transpose_cache_oblivious(
                source, out, i0, j0 + mid, rows, cols - mid, leaf
            )
        return out

    @staticmethod
    def matrix_multiply_naive(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Triple loop in i, j, k order. O(n * m * p) arithmetic.

        The inner step reuses neither ``B``'s row nor a block of ``C``, so a
        small cache reloads the same entries many times.
        """
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes {a.shape} and {b.shape}.")
        rows, mid = a.shape
        cols = b.shape[1]
        out = np.zeros((rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                total = 0.0
                for k in range(mid):
                    total += a[i, k] * b[k, j]
                out[i, j] = total
        return out

    @staticmethod
    def matrix_multiply_cache_oblivious(
        left: np.ndarray,
        right: np.ndarray,
        out: Optional[np.ndarray] = None,
        i0: int = 0,
        j0: int = 0,
        k0: int = 0,
        rows: Optional[int] = None,
        mid: Optional[int] = None,
        cols: Optional[int] = None,
        leaf: int = 64,
    ) -> np.ndarray:
        """Recursive multiplication. ``out`` accumulates, it is not zeroed on subcalls.

        Multiplies the ``rows x mid`` block of ``left`` starting at ``(i0, k0)``
        by the ``mid x cols`` block of ``right`` starting at ``(k0, j0)`` and
        adds the product into ``out[i0:i0+rows, j0:j0+cols]``. The largest
        dimension is split, as in section 12.6.2. O(n * m * p) arithmetic.
        """
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        if out is None:
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"Incompatible shapes {a.shape} and {b.shape}.")
            out = np.zeros((a.shape[0], b.shape[1]), dtype=float)
            return CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, 0, 0, 0, a.shape[0], a.shape[1], b.shape[1], leaf
            )
        assert rows is not None and mid is not None and cols is not None
        if rows <= 0 or mid <= 0 or cols <= 0:
            return out
        if rows * mid * cols <= leaf or rows == 1 or mid == 1 or cols == 1:
            for i in range(rows):
                for j in range(cols):
                    total = out[i0 + i, j0 + j]
                    for k in range(mid):
                        total += a[i0 + i, k0 + k] * b[k0 + k, j0 + j]
                    out[i0 + i, j0 + j] = total
            return out
        if rows >= mid and rows >= cols:
            half = rows // 2
            CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, i0, j0, k0, half, mid, cols, leaf
            )
            CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, i0 + half, j0, k0, rows - half, mid, cols, leaf
            )
        elif mid >= rows and mid >= cols:
            half = mid // 2
            CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, i0, j0, k0, rows, half, cols, leaf
            )
            CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, i0, j0, k0 + half, rows, mid - half, cols, leaf
            )
        else:
            half = cols // 2
            CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, i0, j0, k0, rows, mid, half, leaf
            )
            CacheObliviousOps.matrix_multiply_cache_oblivious(
                a, b, out, i0, j0 + half, k0, rows, mid, cols - half, leaf
            )
        return out

    @staticmethod
    def sum_row_major(matrix: np.ndarray) -> float:
        """Add every entry in row major order. O(n * m)."""
        source = np.asarray(matrix, dtype=float)
        total = 0.0
        rows, cols = source.shape
        for i in range(rows):
            for j in range(cols):
                total += source[i, j]
        return total

    @staticmethod
    def sum_recursive(matrix: np.ndarray, leaf: int = 64) -> float:
        """Add every entry by recursively splitting quadrants. O(n * m).

        This is the traversal analogue of the transpose recurrence: submatrices
        are finished before the walk moves on, so a cache that holds one
        quadrant is reused.
        """
        source = np.asarray(matrix, dtype=float)

        def walk(i0: int, j0: int, rows: int, cols: int) -> float:
            if rows <= 0 or cols <= 0:
                return 0.0
            if rows * cols <= leaf or rows == 1 or cols == 1:
                total = 0.0
                for i in range(rows):
                    for j in range(cols):
                        total += source[i0 + i, j0 + j]
                return total
            if rows >= cols:
                half = rows // 2
                return walk(i0, j0, half, cols) + walk(i0 + half, j0, rows - half, cols)
            half = cols // 2
            return walk(i0, j0, rows, half) + walk(i0, j0 + half, rows, cols - half)

        return walk(0, 0, source.shape[0], source.shape[1])
