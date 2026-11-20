# CSC 5300 Week 13: Advanced Data Structures

Python library for Dr. Amakobe's Chapter 12 (Advanced Computational Algorithms, 2nd ed., 2026). It covers segment trees with lazy updates, Fenwick trees, persistent segment trees, rank/select bit vectors, wavelet trees, cache oblivious matrix routines, and a van Emde Boas tree layout, plus small applications for range queries, a toy version store, and a DNA index.

## Layout

```
advds/
  trees/            segment_tree.py, fenwick_tree.py, persistent_tree.py
  succinct/         bit_vector.py, rank_select.py, wavelet_tree.py
  cache_oblivious/  matrix_ops.py, layouts.py
  applications/     range_queries.py, version_control.py, dna_index.py
  benchmarks/       performance.py
tests/
examples/
docs/chapter12_report.md
results/            plots and CSV tables from the benchmark
setup.py
```

## Install

Python 3.9 or newer. The library needs NumPy and Matplotlib. Tests need pytest.

```bash
pip install -e ".[dev]"
```

From the repo root you can also run without installing, because the examples add the root to `sys.path` and pytest picks up `advds` on the path.

## Examples

```bash
python examples/range_queries_demo.py
python examples/version_control_demo.py
python examples/dna_index_demo.py
python examples/cache_oblivious_demo.py
```

`examples/sample.fasta` is the short sequence the DNA demo reads.

## Tests

```bash
python -m pytest
```

The suite checks each structure against a direct loop on random inputs. One test runs the benchmark in a tiny mode and checks that the CSV files and plots are written.

## Benchmarks

```bash
python -m advds.benchmarks.performance
python -m advds.benchmarks.performance --output results
python -m advds.benchmarks.performance --quick
```

The full run uses modest sizes (arrays up to 20,000, bit vectors up to 200,000, matrix products up to 48 by 48) so it should finish in a few minutes on a normal Windows computer. It writes:

- `results/time.csv`, `memory_footprint.csv`, `compression_ratio.csv`, `query_efficiency.csv`
- `results/summary.json`
- PNG plots for range queries, persistent versions, bit vectors, wavelet compression, matrix times, van Emde Boas misses, and memory

The files already in `results/`, and the numbers in `docs/chapter12_report.md`, are one run of that command on an AMD Ryzen AI 7 445 (12 logical cores, 15 GB of memory, Windows 11, Python 3.14.7). Clock times will differ if you run it again.
