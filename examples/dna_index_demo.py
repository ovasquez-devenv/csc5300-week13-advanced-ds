"""Count bases, quantiles, and a short pattern on a DNA string.

Run from the repository root:

    python examples/dna_index_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advds import DNAIndex


def main() -> None:
    path = Path(__file__).resolve().parent / "sample.fasta"
    index = DNAIndex.from_fasta(path)
    print(f"Indexed {len(index)} bases from {path.name}")
    print("Counts on [10, 20]:")
    for base, count in index.base_counts(10, 20).items():
        print(f"  {base}: {count}")
    print("3rd smallest base on [10, 20]:", index.kth_nucleotide(10, 20, 3))
    print(f"GC content: {index.gc_content(0, len(index) - 1):.1f}%")
    print("Occurrences of ACGT:", index.pattern_frequency("ACGT"))
    print("Wavelet bit vectors:", index.bitmap_bits(), "bits")
    print("ASCII would be:", len(index) * 8, "bits")


if __name__ == "__main__":
    main()
