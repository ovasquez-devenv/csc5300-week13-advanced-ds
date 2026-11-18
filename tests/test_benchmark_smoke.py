"""The benchmark script writes the tables and plots on a tiny input."""

from pathlib import Path

from advds.benchmarks.performance import Benchmarks


def test_quick_benchmark_writes_artifacts(tmp_path: Path):
    summary = Benchmarks.run_all(tmp_path, quick=True, seed=1)
    assert summary["range"]["sizes"] == [400]
    for name in (
        "time.csv",
        "memory_footprint.csv",
        "compression_ratio.csv",
        "query_efficiency.csv",
        "summary.json",
        "time_range_queries.png",
        "persistent_versions.png",
        "bitvector_queries.png",
        "wavelet_compression.png",
        "matrix_times.png",
        "veb_cache_misses.png",
        "memory_footprint.png",
    ):
        assert (tmp_path / name).stat().st_size > 0
