"""
Advanced Data Structures Library

Course library for CSC 5300 Week 13, following Amakobe, Advanced Computational
Algorithms (2nd ed., 2026), Chapter 12.

Provides:
- Segment trees: range queries with updates, including lazy range adds
- Fenwick trees: prefix sums in O(n) space
- Persistent structures: path copied versions that share nodes
- Succinct structures: rank/select bit vectors and wavelet trees
- Cache oblivious algorithms: recursive transpose, multiply, and van Emde Boas layout
"""

__version__ = "1.0.0"

from .trees import (
    FenwickTree,
    FenwickTree2D,
    LazySegmentTree,
    PersistentArray,
    PersistentSegmentTree,
    SegmentTree,
)
from .succinct import SuccinctBitVector, WaveletTree
from .cache_oblivious import CacheObliviousOps
from .applications import DNAIndex, RangeQuerySolver, VersionControl

__all__ = [
    "SegmentTree",
    "LazySegmentTree",
    "FenwickTree",
    "FenwickTree2D",
    "PersistentArray",
    "PersistentSegmentTree",
    "SuccinctBitVector",
    "WaveletTree",
    "CacheObliviousOps",
    "RangeQuerySolver",
    "VersionControl",
    "DNAIndex",
]
