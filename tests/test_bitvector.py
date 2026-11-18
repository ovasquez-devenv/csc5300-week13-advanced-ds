"""Rank and select against a linear scan, including the lookup tables."""

import random

import pytest

from advds.succinct import RankSelectIndex, SuccinctBitVector
from advds.succinct.rank_select import popcount16, select1_in_pattern


def test_lookup_tables_small_patterns():
    assert popcount16(0) == 0
    assert popcount16(0b1011) == 3
    assert select1_in_pattern(0b1011, 1) == 0
    assert select1_in_pattern(0b1011, 2) == 1
    assert select1_in_pattern(0b1011, 3) == 3
    assert select1_in_pattern(0b1011, 4) == -1


def test_book_bitstring():
    bits = "11010110101100001110"
    vector = SuccinctBitVector(bits)
    raw = [int(bit) for bit in bits]
    for index in range(len(raw)):
        assert vector.access(index) == raw[index]
        assert vector.rank1(index) == sum(raw[: index + 1])
        assert vector.rank0(index) == index + 1 - sum(raw[: index + 1])
    ones = [index for index, bit in enumerate(raw) if bit]
    for k, index in enumerate(ones, start=1):
        assert vector.select1(k) == index
    assert vector.select1(0) == -1
    assert vector.select1(len(ones) + 1) == -1
    assert vector.rank1(-1) == 0
    assert vector.rank1(10_000) == sum(raw)


def test_random_rank_select_including_edges():
    rng = random.Random(21)
    samples = ["0", "1", "0000", "1111", "1010", "0101"]
    samples.append("".join(rng.choice("01") for _ in range(1)))
    samples.append("".join(rng.choice("01") for _ in range(17)))
    samples.append("".join(rng.choice("01") for _ in range(200)))
    samples.append("".join(rng.choice("01") for _ in range(1000)))
    for bits in samples:
        raw = [int(bit) for bit in bits]
        vector = SuccinctBitVector(raw)
        assert isinstance(vector.index, RankSelectIndex)
        for index in range(len(raw)):
            assert vector.rank(index) == sum(raw[: index + 1])
        ones = [index for index, bit in enumerate(raw) if bit]
        zeros = [index for index, bit in enumerate(raw) if not bit]
        for k, index in enumerate(ones, start=1):
            assert vector.select(k) == index
        for k, index in enumerate(zeros, start=1):
            assert vector.select0(k) == index


def test_directory_grows_slower_than_the_bit_length():
    vector = SuccinctBitVector([1, 0] * 5000)
    # Directories are integers on top of the packed bits, but they should
    # not be a full Python int per bit.
    assert vector.overhead_bytes() < vector.n
    with pytest.raises(IndexError):
        vector.access(-1)
