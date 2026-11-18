"""Wavelet queries and the DNA index, checked against the raw string."""

import random

import pytest

from advds.applications import DNAIndex, read_fasta
from advds.succinct import WaveletTree


def test_access_rank_select_and_quantile():
    sequence = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    tree = WaveletTree(sequence)
    for index, value in enumerate(sequence):
        assert tree.access(index) == value
    for symbol in sorted(set(sequence)):
        for index in range(len(sequence)):
            assert tree.rank(symbol, index) == sequence[: index + 1].count(symbol)
        positions = [i for i, value in enumerate(sequence) if value == symbol]
        for k, pos in enumerate(positions, start=1):
            assert tree.select(symbol, k) == pos
    for left in range(len(sequence)):
        for right in range(left, len(sequence)):
            window = sorted(sequence[left : right + 1])
            for k, value in enumerate(window, start=1):
                assert tree.range_quantile(left, right, k) == value
            for symbol in set(sequence):
                assert tree.range_count(left, right, symbol) == window.count(symbol)


def test_random_strings_and_frequencies():
    rng = random.Random(8)
    alphabet = list("abcde")
    for n in (1, 3, 20, 150):
        text = [rng.choice(alphabet) for _ in range(n)]
        tree = WaveletTree(text, alphabet=alphabet)
        assert tree.frequencies()[text[0]] == text.count(text[0])
        assert tree.bitmap_bits() >= n
        for _ in range(15):
            index = rng.randrange(n)
            assert tree.access(index) == text[index]
            symbol = rng.choice(alphabet)
            assert tree.rank(symbol, index) == text[: index + 1].count(symbol)
            left = rng.randrange(n)
            right = rng.randint(left, n - 1)
            window = text[left : right + 1]
            k = rng.randint(1, len(window))
            assert tree.range_quantile(left, right, k) == sorted(window)[k - 1]
            assert tree.frequency(symbol, left, right) == window.count(symbol)


def test_dna_counts_quantile_pattern_and_fasta(tmp_path):
    dna = "ACGTACGTTAGCTAGCTAGCTAGCTACGTACGTT"
    index = DNAIndex(dna)
    for base in "ACGT":
        assert index.count_base(base, 10, 20) == dna[10:21].count(base)
    window = sorted(dna[10:21])
    assert index.kth_nucleotide(10, 20, 1) == window[0]
    assert index.kth_nucleotide(10, 20, len(window)) == window[-1]
    assert index.pattern_frequency("ACGT") == dna.count("ACGT")
    assert index.pattern_frequency("AAAA") == 0
    repeated = DNAIndex("AAAA")
    assert repeated.pattern_frequency("AA") == 3
    assert repeated.pattern_frequency("AAA", 1, 3) == 1
    gc = dna.count("G") + dna.count("C")
    assert index.gc_content(0, len(dna) - 1) == pytest.approx(100.0 * gc / len(dna))
    # Four symbols, two bitmap levels, so the bit vectors hold 2 bits per base.
    assert index.bitmap_bits() == 2 * len(dna)

    fasta = tmp_path / "sample.fasta"
    fasta.write_text(">frag one\nACGT\nAC\n>second\nTT\n", encoding="utf-8")
    records = read_fasta(fasta)
    assert records == [("frag one", "ACGTAC"), ("second", "TT")]
    joined = DNAIndex.from_fasta(fasta)
    assert len(joined) == 8
    assert joined.count_base("T", 0, 7) == 3
    second = DNAIndex.from_fasta(fasta, record=1)
    assert second.sequence == "TT"
    with pytest.raises(ValueError):
        DNAIndex("ACGN")
