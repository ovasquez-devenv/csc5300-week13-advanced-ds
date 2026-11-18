"""Benchmarks for the Chapter 12 library.

Writes time, memory, compression, and query efficiency tables as CSV, plus
PNG plots, into a results directory. Sizes stay modest so a full run finishes
in a few minutes on a normal machine. Pass ``quick=True`` for a smoke run.

Run from the repository root:

    python -m advds.benchmarks.performance
    python -m advds.benchmarks.performance --output results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..applications import DNAIndex
from ..cache_oblivious import CacheObliviousOps, CacheSimulator, compare_layouts
from ..succinct import SuccinctBitVector, WaveletTree
from ..trees import FenwickTree, PersistentSegmentTree, SegmentTree, VersionStore, count_unique_nodes


def deep_sizeof(obj, seen: set | None = None) -> int:
    """Approximate retained bytes, counting each object once.

    Shared persistent nodes are counted once, which is the point of the
    memory comparison. The shared 16 bit rank tables are module globals and
    are not attached to each bit vector, so they are not charged per query
    structure.
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    try:
        size = sys.getsizeof(obj)
    except TypeError:
        return 0
    if isinstance(obj, dict):
        size += sum(deep_sizeof(key, seen) + deep_sizeof(value, seen) for key, value in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(item, seen) for item in obj)
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        return size
    else:
        module = type(obj).__module__
        if module == "array":
            return size
        dct = getattr(obj, "__dict__", None)
        if isinstance(dct, dict):
            size += deep_sizeof(dct, seen)
        slots = getattr(type(obj), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if hasattr(obj, name):
                size += deep_sizeof(getattr(obj, name), seen)
    return size


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


class Benchmarks:
    """Timed comparisons requested in section 12.7.4, plus the later structures."""

    @staticmethod
    def run_all(output_dir: str | Path = "results", quick: bool = False, seed: int = 5300) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        np.random.seed(seed)

        if quick:
            range_sizes = [400]
            range_ops = 40
            persistent_n = 300
            persistent_versions = [10, 30]
            bit_sizes = [800]
            bit_queries = 40
            wave_sizes = [400]
            wave_queries = 40
            transpose_sizes = [24]
            multiply_sizes = [16]
            veb_sizes = [255]
            veb_walks = 20
        else:
            range_sizes = [2000, 8000, 20000]
            range_ops = 1500
            persistent_n = 4000
            persistent_versions = [25, 50, 100, 200]
            bit_sizes = [20000, 80000, 200000]
            bit_queries = 2000
            wave_sizes = [2000, 8000, 20000]
            wave_queries = 800
            transpose_sizes = [48, 96, 144]
            multiply_sizes = [24, 36, 48]
            veb_sizes = [1023, 4095, 16383]
            veb_walks = 200

        time_rows: List[dict] = []
        memory_rows: List[dict] = []
        compression_rows: List[dict] = []
        efficiency_rows: List[dict] = []

        print("Range queries: segment tree vs Fenwick tree")
        range_payload = _bench_range(range_sizes, range_ops, rng, time_rows, memory_rows, efficiency_rows)
        _plot_range(out / "time_range_queries.png", range_payload)

        print("Persistent segment tree versions")
        persistent_payload = _bench_persistent(
            persistent_n, persistent_versions, rng, time_rows, memory_rows, efficiency_rows
        )
        _plot_persistent(out / "persistent_versions.png", persistent_payload)

        print("Bit vector rank and select")
        bit_payload = _bench_bits(bit_sizes, bit_queries, rng, time_rows, memory_rows, efficiency_rows)
        _plot_bits(out / "bitvector_queries.png", bit_payload)

        print("Wavelet tree and DNA index")
        wave_payload = _bench_wavelet(
            wave_sizes, wave_queries, rng, time_rows, memory_rows, compression_rows, efficiency_rows
        )
        _plot_wavelet(out / "wavelet_compression.png", wave_payload)

        print("Matrix transpose and multiply")
        matrix_payload = _bench_matrix(
            transpose_sizes, multiply_sizes, time_rows, memory_rows, efficiency_rows, quick
        )
        _plot_matrix(out / "matrix_times.png", matrix_payload)

        print("van Emde Boas layout vs array heap")
        veb_payload = _bench_layouts(veb_sizes, veb_walks, efficiency_rows)
        _plot_layouts(out / "veb_cache_misses.png", veb_payload)

        _plot_memory(out / "memory_footprint.png", memory_rows)

        _write_csv(
            out / "time.csv",
            time_rows,
            ["structure", "operation", "n", "seconds", "ops", "notes"],
        )
        _write_csv(
            out / "memory_footprint.csv",
            memory_rows,
            ["structure", "n", "bytes", "notes"],
        )
        _write_csv(
            out / "compression_ratio.csv",
            compression_rows,
            ["structure", "n", "raw_bits", "stored_bits", "bit_ratio", "python_bytes", "notes"],
        )
        _write_csv(
            out / "query_efficiency.csv",
            efficiency_rows,
            ["structure", "operation", "n", "us_per_op", "baseline", "baseline_us_per_op", "speedup", "notes"],
        )

        summary = {
            "seed": seed,
            "quick": quick,
            "range": range_payload,
            "persistent": persistent_payload,
            "bitvector": bit_payload,
            "wavelet": wave_payload,
            "matrix": matrix_payload,
            "layouts": veb_payload,
            "memory": memory_rows,
            "compression": compression_rows,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote tables and plots to {out.resolve()}")
        return summary


def _bench_range(sizes, ops, rng, time_rows, memory_rows, efficiency_rows):
    payload = {"sizes": list(sizes), "segment": {"build": [], "query": [], "update": []}, "fenwick": {"build": [], "query": [], "update": []}}
    for n in sizes:
        arr = [rng.randint(1, 100) for _ in range(n)]
        queries = []
        for _ in range(ops):
            left = rng.randint(0, n - 1)
            right = rng.randint(0, n - 1)
            if left > right:
                left, right = right, left
            queries.append((left, right))
        updates = [(rng.randint(0, n - 1), rng.randint(1, 100)) for _ in range(ops)]

        start = time.perf_counter()
        segment = SegmentTree(arr, operation="sum")
        seg_build = time.perf_counter() - start
        start = time.perf_counter()
        for left, right in queries:
            segment.query(left, right)
        seg_query = time.perf_counter() - start
        start = time.perf_counter()
        for index, value in updates:
            segment.update(index, value)
        seg_update = time.perf_counter() - start

        start = time.perf_counter()
        fenwick = FenwickTree.from_array(arr)
        fen_build = time.perf_counter() - start
        start = time.perf_counter()
        for left, right in queries:
            fenwick.range_sum(left, right)
        fen_query = time.perf_counter() - start
        start = time.perf_counter()
        for index, value in updates:
            current = fenwick.range_sum(index, index)
            fenwick.update(index, value - current)
        fen_update = time.perf_counter() - start

        payload["segment"]["build"].append(seg_build * 1000)
        payload["segment"]["query"].append(seg_query * 1000)
        payload["segment"]["update"].append(seg_update * 1000)
        payload["fenwick"]["build"].append(fen_build * 1000)
        payload["fenwick"]["query"].append(fen_query * 1000)
        payload["fenwick"]["update"].append(fen_update * 1000)

        for structure, build, query, update, obj in (
            ("segment_tree", seg_build, seg_query, seg_update, segment),
            ("fenwick_tree", fen_build, fen_query, fen_update, fenwick),
        ):
            for operation, seconds in (("build", build), ("range_query", query), ("point_update", update)):
                time_rows.append(
                    {
                        "structure": structure,
                        "operation": operation,
                        "n": n,
                        "seconds": f"{seconds:.6f}",
                        "ops": 1 if operation == "build" else ops,
                        "notes": "milliseconds also stored in the range query plot",
                    }
                )
            memory_rows.append(
                {
                    "structure": structure,
                    "n": n,
                    "bytes": deep_sizeof(obj),
                    "notes": "Python object size after the timed updates",
                }
            )
        efficiency_rows.append(
            {
                "structure": "fenwick_tree",
                "operation": "range_query",
                "n": n,
                "us_per_op": f"{(fen_query / ops) * 1e6:.3f}",
                "baseline": "segment_tree",
                "baseline_us_per_op": f"{(seg_query / ops) * 1e6:.3f}",
                "speedup": f"{(seg_query / fen_query) if fen_query else 0:.3f}",
                "notes": "speedup is segment time divided by Fenwick time",
            }
        )
        print(
            f"  n={n}: segment build {seg_build*1000:.2f} ms, query {seg_query*1000:.2f} ms, "
            f"update {seg_update*1000:.2f} ms | fenwick build {fen_build*1000:.2f} ms, "
            f"query {fen_query*1000:.2f} ms, update {fen_update*1000:.2f} ms"
        )
    return payload


def _bench_persistent(n, version_counts, rng, time_rows, memory_rows, efficiency_rows):
    base = [rng.randint(0, 50) for _ in range(n)]
    payload = {
        "n": n,
        "versions": [],
        "unique_nodes": [],
        "copied_nodes": [],
        "update_ms": [],
        "query_ms": [],
        "copy_ms": [],
        "rebuild_ms": [],
    }
    # Time one batch of updates equal to the largest version count, recording prefixes.
    largest = max(version_counts)
    updates = [(rng.randint(0, n - 1), rng.randint(0, 50)) for _ in range(largest)]
    queries = []
    for _ in range(min(200, max(largest, 1))):
        left = rng.randint(0, n - 1)
        right = rng.randint(0, n - 1)
        if left > right:
            left, right = right, left
        queries.append((left, right))

    marks = set(version_counts)
    fresh = VersionStore(base, operation="sum")
    stamp = time.perf_counter()
    elapsed_at = {}
    for step, (index, value) in enumerate(updates, start=1):
        fresh.update(index, value)
        if step in marks:
            elapsed_at[step] = time.perf_counter() - stamp
    # Node counts are not part of the update timer. versions[0] is the original array.
    built_marks = {}
    for versions in version_counts:
        prefix = fresh.versions[: versions + 1]
        built_marks[versions] = (
            count_unique_nodes(prefix),
            sum(tree.node_count() for tree in prefix),
            deep_sizeof(prefix),
        )

    # Copy the whole list after every edit, which is the naive history cost.
    copy_times = {}
    naive = list(base)
    copy_stamp = time.perf_counter()
    for step, (index, value) in enumerate(updates, start=1):
        naive[index] = value
        snapshot = list(naive)
        if step in marks:
            copy_times[step] = time.perf_counter() - copy_stamp
        del snapshot

    # Rebuilding a private segment tree each version copies every node.
    rebuild_times = {}
    naive = list(base)
    rebuild_stamp = time.perf_counter()
    for step, (index, value) in enumerate(updates, start=1):
        naive[index] = value
        PersistentSegmentTree(naive)
        if step in marks:
            rebuild_times[step] = time.perf_counter() - rebuild_stamp

    for versions in version_counts:
        unique, copied, nbytes = built_marks[versions]
        update_s = elapsed_at[versions]
        copy_s = copy_times[versions]
        rebuild_s = rebuild_times[versions]
        start = time.perf_counter()
        for left, right in queries:
            fresh.query(left, right, version=versions)
        query_s = time.perf_counter() - start
        payload["versions"].append(versions)
        payload["unique_nodes"].append(unique)
        payload["copied_nodes"].append(copied)
        payload["update_ms"].append(update_s * 1000)
        payload["query_ms"].append(query_s * 1000)
        payload["copy_ms"].append(copy_s * 1000)
        payload["rebuild_ms"].append(rebuild_s * 1000)
        time_rows.append(
            {
                "structure": "persistent_segment_tree",
                "operation": "point_update_new_version",
                "n": n,
                "seconds": f"{update_s:.6f}",
                "ops": versions,
                "notes": "cumulative time to create this many versions",
            }
        )
        time_rows.append(
            {
                "structure": "full_array_copy",
                "operation": "point_update_new_version",
                "n": n,
                "seconds": f"{copy_s:.6f}",
                "ops": versions,
                "notes": "cumulative time to copy the whole list after each update",
            }
        )
        time_rows.append(
            {
                "structure": "rebuilt_segment_tree",
                "operation": "point_update_new_version",
                "n": n,
                "seconds": f"{rebuild_s:.6f}",
                "ops": versions,
                "notes": "cumulative time to build a fresh persistent tree after each update",
            }
        )
        memory_rows.append(
            {
                "structure": "persistent_segment_tree",
                "n": n,
                "bytes": nbytes,
                "notes": f"{versions} versions, {unique} unique nodes, {copied} nodes if each version were private",
            }
        )
        efficiency_rows.append(
            {
                "structure": "persistent_segment_tree",
                "operation": "versioned_update",
                "n": n,
                "us_per_op": f"{(update_s / versions) * 1e6:.3f}",
                "baseline": "rebuilt_segment_tree",
                "baseline_us_per_op": f"{(rebuild_s / versions) * 1e6:.3f}",
                "speedup": f"{(rebuild_s / update_s) if update_s else 0:.3f}",
                "notes": (
                    f"node sharing {unique}/{copied} after {versions} versions; "
                    f"list copy us/op {(copy_s / versions) * 1e6:.3f}"
                ),
            }
        )
        print(
            f"  versions={versions}: path copy {update_s*1000:.2f} ms, "
            f"list copy {copy_s*1000:.2f} ms, tree rebuild {rebuild_s*1000:.2f} ms, "
            f"nodes {unique} vs {copied}"
        )
    return payload


def _bench_bits(sizes, queries, rng, time_rows, memory_rows, efficiency_rows):
    payload = {"sizes": list(sizes), "rank_us": [], "select_us": [], "naive_rank_us": [], "naive_select_us": []}
    for n in sizes:
        bits = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
        vector = SuccinctBitVector(bits)
        ones = [index for index, bit in enumerate(bits) if bit]
        rank_at = [rng.randint(0, n - 1) for _ in range(queries)]
        select_at = [rng.randint(1, len(ones)) for _ in range(queries)] if ones else []

        start = time.perf_counter()
        for index in rank_at:
            vector.rank1(index)
        rank_s = time.perf_counter() - start
        start = time.perf_counter()
        for k in select_at:
            vector.select1(k)
        select_s = time.perf_counter() - start

        # Brute force on a shorter sample so a long bit vector still finishes.
        naive_queries = min(80, queries)
        naive_rank_idx = rank_at[:naive_queries]
        naive_select_k = select_at[:naive_queries]
        start = time.perf_counter()
        for index in naive_rank_idx:
            sum(bits[: index + 1])
        naive_rank_s = time.perf_counter() - start
        start = time.perf_counter()
        for k in naive_select_k:
            seen = 0
            for index, bit in enumerate(bits):
                seen += bit
                if seen == k:
                    break
        naive_select_s = time.perf_counter() - start

        rank_us = (rank_s / queries) * 1e6 if queries else 0
        select_us = (select_s / len(select_at)) * 1e6 if select_at else 0
        naive_rank_us = (naive_rank_s / naive_queries) * 1e6 if naive_queries else 0
        naive_select_us = (naive_select_s / max(1, len(naive_select_k))) * 1e6 if naive_select_k else 0
        payload["rank_us"].append(rank_us)
        payload["select_us"].append(select_us)
        payload["naive_rank_us"].append(naive_rank_us)
        payload["naive_select_us"].append(naive_select_us)

        time_rows.append({"structure": "succinct_bitvector", "operation": "rank1", "n": n, "seconds": f"{rank_s:.6f}", "ops": queries, "notes": ""})
        time_rows.append({"structure": "succinct_bitvector", "operation": "select1", "n": n, "seconds": f"{select_s:.6f}", "ops": len(select_at), "notes": ""})
        memory_rows.append(
            {
                "structure": "succinct_bitvector",
                "n": n,
                "bytes": deep_sizeof(vector),
                "notes": f"directories {vector.overhead_bytes()} bytes, data words separate",
            }
        )
        raw_bits = n
        directory_bits = vector.overhead_bytes() * 8
        efficiency_rows.append(
            {
                "structure": "succinct_bitvector",
                "operation": "rank1",
                "n": n,
                "us_per_op": f"{rank_us:.3f}",
                "baseline": "linear_scan",
                "baseline_us_per_op": f"{naive_rank_us:.3f}",
                "speedup": f"{(naive_rank_us / rank_us) if rank_us else 0:.3f}",
                "notes": f"select1 {select_us:.3f} us vs scan {naive_select_us:.3f} us",
            }
        )
        print(
            f"  n={n}: rank {rank_us:.2f} us (scan {naive_rank_us:.2f} us), "
            f"select {select_us:.2f} us (scan {naive_select_us:.2f} us), "
            f"overhead bits {directory_bits} on {raw_bits} data bits"
        )
    return payload


def _bench_wavelet(sizes, queries, rng, time_rows, memory_rows, compression_rows, efficiency_rows):
    bases = "ACGT"
    payload = {"sizes": list(sizes), "bit_ratio": [], "query_us": [], "python_bytes": []}
    for n in sizes:
        text = "".join(rng.choice(bases) for _ in range(n))
        index = DNAIndex(text)
        tree = index.wt
        windows = []
        quantiles = []
        for _ in range(queries):
            left = rng.randint(0, n - 1)
            right = rng.randint(0, n - 1)
            if left > right:
                left, right = right, left
            windows.append((left, right, rng.choice(bases)))
            quantiles.append(rng.randint(1, right - left + 1))
        start = time.perf_counter()
        for left, right, base in windows:
            tree.range_count(left, right, base)
        count_s = time.perf_counter() - start
        start = time.perf_counter()
        for (left, right, _base), k in zip(windows, quantiles):
            tree.range_quantile(left, right, k)
        quantile_s = time.perf_counter() - start

        stored_bits = tree.bitmap_bits()
        raw_bits = n * 8
        ratio = (raw_bits / stored_bits) if stored_bits else 0
        py_bytes = deep_sizeof(index)
        payload["bit_ratio"].append(ratio)
        payload["query_us"].append((count_s / queries) * 1e6)
        payload["python_bytes"].append(py_bytes)

        # Brute range count on a short sample.
        sample = windows[: min(40, len(windows))]
        start = time.perf_counter()
        for left, right, base in sample:
            text[left : right + 1].count(base)
        brute_s = time.perf_counter() - start
        brute_us = (brute_s / len(sample)) * 1e6
        count_us = (count_s / queries) * 1e6

        time_rows.append({"structure": "wavelet_tree", "operation": "range_count", "n": n, "seconds": f"{count_s:.6f}", "ops": queries, "notes": "DNA alphabet"})
        time_rows.append({"structure": "wavelet_tree", "operation": "range_quantile", "n": n, "seconds": f"{quantile_s:.6f}", "ops": queries, "notes": "DNA alphabet"})
        memory_rows.append({"structure": "dna_index", "n": n, "bytes": py_bytes, "notes": "wavelet plus the original Python string"})
        compression_rows.append(
            {
                "structure": "wavelet_tree_dna",
                "n": n,
                "raw_bits": raw_bits,
                "stored_bits": stored_bits,
                "bit_ratio": f"{ratio:.3f}",
                "python_bytes": py_bytes,
                "notes": "raw_bits counts 8 bits per ASCII base; stored_bits is the sum of node bit vectors",
            }
        )
        efficiency_rows.append(
            {
                "structure": "wavelet_tree",
                "operation": "range_count",
                "n": n,
                "us_per_op": f"{count_us:.3f}",
                "baseline": "str.count",
                "baseline_us_per_op": f"{brute_us:.3f}",
                "speedup": f"{(brute_us / count_us) if count_us else 0:.3f}",
                "notes": f"bit compression ratio {ratio:.3f} (ASCII bits / wavelet bits)",
            }
        )
        print(
            f"  n={n}: range count {count_us:.2f} us vs str.count {brute_us:.2f} us, "
            f"bit ratio {ratio:.2f}, python bytes {py_bytes}"
        )
    return payload


def _bench_matrix(transpose_sizes, multiply_sizes, time_rows, memory_rows, efficiency_rows, quick):
    payload = {
        "transpose_sizes": list(transpose_sizes),
        "transpose_naive_ms": [],
        "transpose_co_ms": [],
        "multiply_sizes": list(multiply_sizes),
        "multiply_naive_ms": [],
        "multiply_co_ms": [],
        "misses": [],
    }
    for n in transpose_sizes:
        matrix = np.random.randn(n, n)
        start = time.perf_counter()
        naive = CacheObliviousOps.matrix_transpose_naive(matrix)
        naive_s = time.perf_counter() - start
        start = time.perf_counter()
        clever = CacheObliviousOps.matrix_transpose_cache_oblivious(matrix, leaf=64)
        clever_s = time.perf_counter() - start
        if not np.allclose(naive, clever):
            raise RuntimeError("Transpose results diverged during the benchmark.")
        payload["transpose_naive_ms"].append(naive_s * 1000)
        payload["transpose_co_ms"].append(clever_s * 1000)
        time_rows.append({"structure": "transpose_naive", "operation": "transpose", "n": n, "seconds": f"{naive_s:.6f}", "ops": 1, "notes": ""})
        time_rows.append({"structure": "transpose_cache_oblivious", "operation": "transpose", "n": n, "seconds": f"{clever_s:.6f}", "ops": 1, "notes": "leaf 64"})
        memory_rows.append({"structure": "matrix_pair", "n": n, "bytes": int(matrix.nbytes * 2), "notes": "two float64 matrices, input plus transpose"})
        print(f"  transpose n={n}: naive {naive_s*1000:.2f} ms, cache oblivious {clever_s*1000:.2f} ms")

    for n in multiply_sizes:
        left = np.random.randn(n, n)
        right = np.random.randn(n, n)
        start = time.perf_counter()
        naive = CacheObliviousOps.matrix_multiply_naive(left, right)
        naive_s = time.perf_counter() - start
        start = time.perf_counter()
        clever = CacheObliviousOps.matrix_multiply_cache_oblivious(left, right, leaf=256)
        clever_s = time.perf_counter() - start
        if not np.allclose(naive, clever):
            raise RuntimeError("Multiply results diverged during the benchmark.")
        payload["multiply_naive_ms"].append(naive_s * 1000)
        payload["multiply_co_ms"].append(clever_s * 1000)
        time_rows.append({"structure": "multiply_naive", "operation": "multiply", "n": n, "seconds": f"{naive_s:.6f}", "ops": 1, "notes": "i j k loops"})
        time_rows.append({"structure": "multiply_cache_oblivious", "operation": "multiply", "n": n, "seconds": f"{clever_s:.6f}", "ops": 1, "notes": "leaf 256"})
        efficiency_rows.append(
            {
                "structure": "multiply_cache_oblivious",
                "operation": "multiply",
                "n": n,
                "us_per_op": f"{clever_s * 1e6:.3f}",
                "baseline": "multiply_naive",
                "baseline_us_per_op": f"{naive_s * 1e6:.3f}",
                "speedup": f"{(naive_s / clever_s) if clever_s else 0:.3f}",
                "notes": "wall clock, one multiply; speedup above 1 means the recursive code was faster",
            }
        )
        print(f"  multiply n={n}: naive {naive_s*1000:.2f} ms, cache oblivious {clever_s*1000:.2f} ms")

    # A cache that holds a tile but not a full column of the transpose.
    # Smaller caches in the quick run keep the smoke test short.
    miss_n = 64 if not quick else 12
    multiply_miss_n = 24 if not quick else 8
    t_line, t_lines = (8, 16) if not quick else (8, 4)
    m_line, m_lines = (8, 8) if not quick else (8, 4)
    misses = {
        "n": miss_n,
        "multiply_n": multiply_miss_n,
        "line_size": t_line,
        "num_lines": t_lines,
        "multiply_line_size": m_line,
        "multiply_num_lines": m_lines,
        "transpose_naive": _transpose_misses(miss_n, "naive", t_line, t_lines),
        "transpose_co": _transpose_misses(miss_n, "cache_oblivious", t_line, t_lines),
        "multiply_naive": _multiply_misses(multiply_miss_n, "naive", m_line, m_lines),
        "multiply_co": _multiply_misses(multiply_miss_n, "cache_oblivious", m_line, m_lines),
    }
    payload["misses"].append(misses)
    for name, count in misses.items():
        if name not in ("transpose_naive", "transpose_co", "multiply_naive", "multiply_co"):
            continue
        efficiency_rows.append(
            {
                "structure": name,
                "operation": "simulated_cache_misses",
                "n": miss_n if "transpose" in name else multiply_miss_n,
                "us_per_op": f"{count}",
                "baseline": "see notes",
                "baseline_us_per_op": "",
                "speedup": "",
                "notes": (
                    "us_per_op column holds miss count; "
                    f"transpose line {t_line} x {t_lines} lines; multiply line {m_line} x {m_lines} lines"
                ),
            }
        )
    print(
        f"  simulated misses: transpose n={miss_n} naive {misses['transpose_naive']}, "
        f"co {misses['transpose_co']}; multiply n={multiply_miss_n} naive {misses['multiply_naive']}, "
        f"co {misses['multiply_co']}"
    )
    return payload


def _transpose_misses(n: int, method: str, line_size: int, num_lines: int) -> int:
    cache = CacheSimulator(line_size, num_lines)
    base_b = n * n

    def read_a(i, j):
        cache.access(i * n + j)

    def write_b(i, j):
        cache.access(base_b + j * n + i)

    def naive():
        for i in range(n):
            for j in range(n):
                read_a(i, j)
                write_b(i, j)

    def clever(i0, j0, rows, cols):
        if rows * cols <= 16 or rows == 1 or cols == 1:
            for i in range(rows):
                for j in range(cols):
                    read_a(i0 + i, j0 + j)
                    write_b(i0 + i, j0 + j)
            return
        if rows >= cols:
            mid = rows // 2
            clever(i0, j0, mid, cols)
            clever(i0 + mid, j0, rows - mid, cols)
        else:
            mid = cols // 2
            clever(i0, j0, rows, mid)
            clever(i0, j0 + mid, rows, cols - mid)

    if method == "naive":
        naive()
    else:
        clever(0, 0, n, n)
    return cache.misses


def _multiply_misses(n: int, method: str, line_size: int, num_lines: int) -> int:
    cache = CacheSimulator(line_size, num_lines)
    base_b = n * n
    base_c = 2 * n * n

    def touch(kind, i, k, j):
        if kind == "a":
            cache.access(i * n + k)
        elif kind == "b":
            cache.access(base_b + k * n + j)
        else:
            cache.access(base_c + i * n + j)

    def naive():
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    touch("a", i, k, j)
                    touch("b", i, k, j)
                    touch("c", i, k, j)

    def clever(i0, j0, k0, rows, mid, cols):
        if rows * mid * cols <= 32 or rows == 1 or mid == 1 or cols == 1:
            for i in range(rows):
                for j in range(cols):
                    for k in range(mid):
                        touch("a", i0 + i, k0 + k, j0 + j)
                        touch("b", i0 + i, k0 + k, j0 + j)
                        touch("c", i0 + i, k0 + k, j0 + j)
            return
        if rows >= mid and rows >= cols:
            half = rows // 2
            clever(i0, j0, k0, half, mid, cols)
            clever(i0 + half, j0, k0, rows - half, mid, cols)
        elif mid >= rows and mid >= cols:
            half = mid // 2
            clever(i0, j0, k0, rows, half, cols)
            clever(i0, j0, k0 + half, rows, mid - half, cols)
        else:
            half = cols // 2
            clever(i0, j0, k0, rows, mid, half)
            clever(i0, j0 + half, k0, rows, mid, cols - half)

    if method == "naive":
        naive()
    else:
        clever(0, 0, 0, n, n, n)
    return cache.misses


def _bench_layouts(sizes, walks, efficiency_rows):
    payload = {"sizes": [], "veb_search": [], "heap_search": [], "veb_dfs": [], "heap_dfs": []}
    for n in sizes:
        stats = compare_layouts(n, walks=walks, line_size=8, num_lines=4, seed=5300)
        payload["sizes"].append(n)
        payload["veb_search"].append(stats["veb_search_misses"])
        payload["heap_search"].append(stats["heap_search_misses"])
        payload["veb_dfs"].append(stats["veb_dfs_misses"])
        payload["heap_dfs"].append(stats["heap_dfs_misses"])
        for key in ("veb_search_misses", "heap_search_misses", "veb_dfs_misses", "heap_dfs_misses"):
            efficiency_rows.append(
                {
                    "structure": "veb" if key.startswith("veb") else "array_heap",
                    "operation": key,
                    "n": n,
                    "us_per_op": str(stats[key]),
                    "baseline": "array_heap" if key.startswith("veb") else "veb",
                    "baseline_us_per_op": "",
                    "speedup": "",
                    "notes": f"misses, not microseconds; {walks} searches, line 8, 4 lines; accesses search {stats['search_accesses']} dfs {stats['dfs_accesses']}",
                }
            )
        print(
            f"  n={n}: search misses vEB {stats['veb_search_misses']} heap {stats['heap_search_misses']}; "
            f"dfs misses vEB {stats['veb_dfs_misses']} heap {stats['heap_dfs_misses']}"
        )
    return payload


def _plot_range(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = [("build", "Build time"), ("query", "Query time"), ("update", "Update time")]
    for ax, (key, title) in zip(axes, labels):
        ax.plot(payload["sizes"], payload["segment"][key], marker="o", label="Segment tree")
        ax.plot(payload["sizes"], payload["fenwick"][key], marker="s", label="Fenwick tree")
        _style_ax(ax, title, "Array length", "Time (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_persistent(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(payload["versions"], payload["update_ms"], marker="o", label="Path copying")
    axes[0].plot(payload["versions"], payload["rebuild_ms"], marker="s", label="Rebuild whole tree")
    axes[0].plot(payload["versions"], payload["copy_ms"], marker="^", label="Copy the list")
    _style_ax(axes[0], "Time to build versions", "Versions", "Time (ms)")
    axes[1].plot(payload["versions"], payload["unique_nodes"], marker="o", label="Shared nodes")
    axes[1].plot(payload["versions"], payload["copied_nodes"], marker="s", label="Nodes if copied")
    _style_ax(axes[1], "Nodes across versions", "Versions", "Nodes")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_bits(path: Path, payload: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(payload["sizes"], payload["rank_us"], marker="o", label="rank1")
    ax.plot(payload["sizes"], payload["select_us"], marker="s", label="select1")
    ax.plot(payload["sizes"], payload["naive_rank_us"], marker="^", label="scan rank")
    ax.plot(payload["sizes"], payload["naive_select_us"], marker="v", label="scan select")
    ax.set_yscale("log")
    _style_ax(ax, "Bit vector rank and select", "Length (bits)", "Microseconds per op")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_wavelet(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar([str(n) for n in payload["sizes"]], payload["bit_ratio"], color="#4C78A8", label="ASCII bits / wavelet bits")
    _style_ax(axes[0], "DNA wavelet bit ratio", "Sequence length", "Ratio")
    axes[1].plot(payload["sizes"], payload["query_us"], marker="o", label="range count")
    _style_ax(axes[1], "Wavelet range count", "Sequence length", "Microseconds per op")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_matrix(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x1 = np.arange(len(payload["transpose_sizes"]))
    width = 0.35
    axes[0].bar(x1 - width / 2, payload["transpose_naive_ms"], width, label="Naive")
    axes[0].bar(x1 + width / 2, payload["transpose_co_ms"], width, label="Cache oblivious")
    axes[0].set_xticks(x1, [str(n) for n in payload["transpose_sizes"]])
    _style_ax(axes[0], "Matrix transpose", "n", "Time (ms)")
    x2 = np.arange(len(payload["multiply_sizes"]))
    axes[1].bar(x2 - width / 2, payload["multiply_naive_ms"], width, label="Naive")
    axes[1].bar(x2 + width / 2, payload["multiply_co_ms"], width, label="Cache oblivious")
    axes[1].set_xticks(x2, [str(n) for n in payload["multiply_sizes"]])
    _style_ax(axes[1], "Matrix multiply", "n", "Time (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_layouts(path: Path, payload: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(payload["sizes"]))
    width = 0.35
    axes[0].bar(x - width / 2, payload["veb_search"], width, label="van Emde Boas")
    axes[0].bar(x + width / 2, payload["heap_search"], width, label="Array heap")
    axes[0].set_xticks(x, [str(n) for n in payload["sizes"]])
    _style_ax(axes[0], "Root to leaf search misses", "Nodes", "Misses")
    axes[1].bar(x - width / 2, payload["veb_dfs"], width, label="van Emde Boas")
    axes[1].bar(x + width / 2, payload["heap_dfs"], width, label="Array heap")
    axes[1].set_xticks(x, [str(n) for n in payload["sizes"]])
    _style_ax(axes[1], "Depth first walk misses", "Nodes", "Misses")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_memory(path: Path, memory_rows: Sequence[dict]) -> None:
    # One bar group per structure, using the largest n recorded for that structure.
    best: Dict[str, dict] = {}
    for row in memory_rows:
        name = row["structure"]
        if name in ("matrix_pair", "full_array_copy"):
            continue
        previous = best.get(name)
        if previous is None or int(row["n"]) >= int(previous["n"]):
            best[name] = row
    if not best:
        return
    names = list(best)
    values = [int(best[name]["bytes"]) / 1024 for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values, color="#4C78A8", label="Largest measured n")
    ax.set_title("Memory footprint")
    ax.set_ylabel("Kilobytes")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 12 benchmarks.")
    parser.add_argument("--output", default="results", help="Directory for CSV and PNG files.")
    parser.add_argument("--quick", action="store_true", help="Tiny sizes for a smoke run.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    Benchmarks.run_all(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
