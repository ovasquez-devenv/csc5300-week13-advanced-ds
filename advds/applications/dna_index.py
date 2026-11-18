"""DNA index on a wavelet tree.

Section 12.5.4 stores a deoxyribonucleic acid string over the alphabet
A, C, G, T. With four symbols the wavelet tree is two bits of bit vector per
base, plus the rank directories. ``count_base`` is a range count. The k th
smallest nucleotide in a window is a range quantile, with A before C before
G before T. Pattern frequency walks the select hits of the first base and
checks the following bases with ``access``, so a rare first base does not
scan the whole string.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from ..succinct import WaveletTree


ALPHABET = ("A", "C", "G", "T")


class DNAIndex:
    """Range counts, quantiles, and pattern counts on one DNA string."""

    def __init__(self, sequence: str) -> None:
        cleaned = _normalize(sequence)
        self.sequence = cleaned
        self.wt = WaveletTree(list(cleaned), alphabet=ALPHABET)

    def __len__(self) -> int:
        return len(self.sequence)

    def count_base(self, base: str, start: int, end: int) -> int:
        """How many times ``base`` occurs in the inclusive window. O(log sigma)."""
        base = _check_base(base)
        self._check_window(start, end)
        return self.wt.range_count(start, end, base)

    def kth_nucleotide(self, start: int, end: int, k: int) -> str:
        """k th smallest base in the window. k starts at 1. O(log sigma)."""
        self._check_window(start, end)
        symbol = self.wt.range_quantile(start, end, k)
        return str(symbol)

    def pattern_frequency(self, pattern: str, start: int = 0, end: int | None = None) -> int:
        """Overlapping occurrences of ``pattern`` inside the inclusive window.

        Occurrences may overlap. The search is O((1 + hits) * m * log sigma)
        for a pattern of length m, because each candidate is verified with
        ``access`` rather than a raw slice of the original string.
        """
        if end is None:
            end = len(self.sequence) - 1
        if len(self.sequence) == 0:
            return 0
        self._check_window(start, end)
        pattern = _normalize(pattern)
        if not pattern:
            raise ValueError("Pattern must be non empty.")
        last_start = end - len(pattern) + 1
        if last_start < start:
            return 0
        first = pattern[0]
        before = self.wt.rank(first, start - 1)
        total = self.wt.rank(first, last_start)
        found = 0
        for k in range(before + 1, total + 1):
            pos = self.wt.select(first, k)
            if pos < 0 or pos > last_start:
                break
            if self._matches(pos, pattern):
                found += 1
        return found

    def gc_content(self, start: int, end: int) -> float:
        """Percent of the window that is G or C."""
        self._check_window(start, end)
        width = end - start + 1
        gc = self.count_base("G", start, end) + self.count_base("C", start, end)
        return 100.0 * gc / width

    def base_counts(self, start: int, end: int) -> dict:
        self._check_window(start, end)
        return {base: self.count_base(base, start, end) for base in ALPHABET}

    def bitmap_bits(self) -> int:
        """Bits stored in wavelet node bit vectors, about 2 per base for DNA."""
        return self.wt.bitmap_bits()

    def _matches(self, pos: int, pattern: str) -> bool:
        for offset, base in enumerate(pattern):
            if self.wt.access(pos + offset) != base:
                return False
        return True

    def _check_window(self, start: int, end: int) -> None:
        if start > end or start < 0 or end >= len(self.sequence):
            raise IndexError((start, end))

    @classmethod
    def from_fasta(cls, path: Union[str, Path], record: int | None = None) -> "DNAIndex":
        """Index one FASTA record, or the concatenation of every record.

        ``record`` selects a single sequence (0 based). Headers are not part
        of the indexed text.
        """
        records = read_fasta(path)
        if not records:
            raise ValueError(f"No FASTA sequences in {path}.")
        if record is None:
            text = "".join(seq for _, seq in records)
        else:
            text = records[record][1]
        return cls(text)

    def __repr__(self) -> str:
        return f"DNAIndex(n={len(self.sequence)})"


def read_fasta(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Return ``(header, sequence)`` pairs. Sequence letters are uppercased.

    Blank lines are ignored. A header is the text after ``>``.
    """
    records: List[Tuple[str, str]] = []
    header: str | None = None
    chunks: List[str] = []
    text = Path(path).read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, _normalize("".join(chunks))))
            header = line[1:].strip() or f"record{len(records) + 1}"
            chunks = []
        else:
            if header is None:
                header = "record1"
            chunks.append(line)
    if header is not None:
        records.append((header, _normalize("".join(chunks))))
    return records


def _normalize(sequence: str) -> str:
    cleaned = "".join(sequence.split()).upper()
    bad = [char for char in cleaned if char not in ALPHABET]
    if bad:
        raise ValueError(f"DNA sequence contains {bad[0]!r}. Expected A, C, G, or T.")
    return cleaned


def _check_base(base: str) -> str:
    base = base.upper()
    if base not in ALPHABET:
        raise ValueError(f"Base must be one of A, C, G, T, not {base!r}.")
    return base
