# Executive summary

I built a Python library for Chapter 12 of Amakobe, Advanced Computational Algorithms (2nd ed., 2026), and timed it on one virtual machine. A Fenwick tree beat a segment tree on range sums and used less memory. A persistent segment tree kept two hundred versions of a four thousand element array in 10,593 nodes, not the 1,607,799 nodes of a private tree per version. Rank on a bit vector, with superblocks, blocks, and a 16 bit table, stayed under a microsecond. On deoxyribonucleic acid (DNA) the wavelet bit vectors hold two bits per base. The van Emde Boas layout cut simulated cache misses. Recursive matrix code did not win on the clock in this interpreter.

# Overview of advanced data structures

Amakobe starts from jobs arrays and hash tables do badly: a changing range aggregate, an old version, a long bit string, and a walk that misses the cache. Chapter 12 pairs those with a segment tree, a Fenwick tree (also a binary indexed tree), path copying, a succinct bit vector, a wavelet tree, and cache oblivious divide and conquer. I kept those shapes. The segment tree is a flat array of about 4n slots, the Fenwick tree steps by lowbit (`i & -i`), persistent nodes are never rewritten, and the bit vector is the two level directory in section 12.5.2 plus a lookup table. I did not use another textbook.

# Tree-based range queries

A leaf holds one array slot and an internal node holds the aggregate of its interval. A query is O(log n) disjoint nodes, so sum, min, and max share the walk. Build is linear, and a point update fixes one leaf and its ancestors. Lazy propagation, section 12.2.5, tags a whole segment add and pushes the tag only when a later call opens the node. A sum tag is scaled by the length. A min or max tag is added once.

The Fenwick tree only sums. Slot i covers lowbit(i) entries ending at i, and a range is the difference of two prefixes, as in section 12.3.3. At length 20,000, 1,500 queries took 7.35 ms in the segment tree and 1.86 ms in the Fenwick tree, 4.90 microseconds against 1.24, a factor of 3.96. Builds were 4.25 ms and 2.43 ms. Updates were closer, 4.30 ms and 3.36 ms, because a Fenwick write reads the old value first. Retained sizes were 973,237 and 248,203 bytes.

![Range query times](../results/time_range_queries.png)

![Memory footprint](../results/memory_footprint.png)

# Persistent structures

Section 12.4 copies the root to leaf path and shares the other child pointers. After a left side edit, the new root's right child is the same object as the old one, and the old root still returns the old sum. Two hundred updates on 4,000 elements left 10,593 nodes. Private copies would have been 1,607,799, about 152 times more, in 746,209 bytes of shared history.

Copying a Python list after each edit took 0.79 ms. Path copying took 1.65 ms, because new node objects cost more than a tight list copy. Rebuilding a whole segment tree every edit took 715.59 ms. Path copying wins against that rebuild, not against the list copy. An old version is still an O(log n) walk from its root.

![Persistent versions](../results/persistent_versions.png)

# Succinct and compressed structures

Section 12.5.2 wants constant time rank. Bits sit in 64 bit words. A superblock about (log n) squared bits wide stores the ones before it, a 16 bit block stores the count inside that superblock, and lookup tables finish the offset. rank1 is a few reads. select1 binary searches the superblocks and ends in the table, which is the chapter's binary search with an O(1) ending, so I call select O(log n).

On 200,000 bits, rank averaged 0.68 microseconds against 518.71 for a list scan, about 768 times, and select1 averaged 7.51 against 3,342.90. The object was 84,863 bytes, with 55,264 in the directories, because each block rank is a full 32 bit integer.

A wavelet tree (section 12.5.3) stores each alphabet split in a bit vector, so access, rank, select, a range count, and the k th symbol in a window are O(log sigma). DNA is A, C, G, T, two bits per base, ratio 4.00 against 8 bit characters: 40,000 stored bits on 20,000 bases. Range counts averaged 3.03 microseconds against 13.57 for a string count, though at 2,000 bases the string count won, 1.54 against 2.98. The index object was 203,809 bytes.

![Bit vector queries](../results/bitvector_queries.png)

![Wavelet compression](../results/wavelet_compression.png)

# Cache-oblivious methods

Section 12.6 recurses until a block is small enough to fit whatever cache is there. On the clock the plain loops won: a 144 by 144 transpose was 2.31 ms against 2.87 ms, and a 48 by 48 product was 19.07 ms against 24.30 ms. The miss model goes the other way. With least recently used replacement, 8 elements per line and 16 lines, a 64 by 64 transpose missed 4,608 times in plain order and 1,024 recursively. A 24 by 24 product with 8 lines missed 15,624 times in the triple loop and 6,528 in the recursion.

The van Emde Boas layout stores the top half of the levels, then each bottom subtree, while an array heap stores node i at index i. Two hundred root to leaf walks on 16,383 nodes, 4 lines of 8, missed 1,400 times in van Emde Boas order and 2,400 in the heap. A depth first walk missed 2,391 against 5,627. Both orders visit the same ids.

![Matrix times](../results/matrix_times.png)

![Layout misses](../results/veb_cache_misses.png)

# Applications (version control, DNA, range queries)

The front end in section 12.7.3 sends sums to the Fenwick tree, min or max to the segment tree, and a range add to the lazy tree. A time travel flag keeps a persistent segment tree so a query can name an older version.

Each file in the small version store is a persistent segment tree of character codes, one snapshot per commit. Equal length text is path copied, so an untouched file shares its root and the diff stops on that object. A new length is a replacement. The indented parent links are the history sketch from section 12.4.3, not a production tool.

The DNA index in section 12.5.4 counts a base with a range count and picks the k th smallest nucleotide with the quantile, ordered A, C, G, then T. Pattern frequency selects the first base and checks the rest with access, overlaps included. Tests compare those answers with direct counts on random strings.

# Benchmarks and analysis

## Methodology

This run was a virtual machine, Ubuntu 24.04, 4 Intel Xeon virtual processors, 16 GB of memory, Python 3.12.3. NumPy held the matrices and Matplotlib drew the plots. Seed 5300 fixes the inputs. I timed each batch and divided by the count, and memory charges each live object once so shared nodes are not billed twice. Miss counts come from that small cache, not from hardware. Lengths stay modest, so `python -m advds.benchmarks.performance` finished in a few seconds here and should stay under a few minutes on a normal Windows computer. Plots and comma separated value (CSV) tables are in `results/`.

## Results

At length 20,000 the Fenwick query was 3.96 times the segment query, at about a quarter of the bytes. Two hundred persistent versions took 1.65 ms and 10,593 nodes, against 715.59 ms and 1,607,799 nodes for full rebuilds. rank1 was 0.68 microseconds on 200,000 bits. The DNA bit ratio was 4, and a long range count was 4.47 times a string count. Recursive transpose misses were 1,024 against 4,608, and the van Emde Boas search missed 1,400 times against 2,400.

# Discussion

Amakobe counts traffic and nodes. In CPython a call and an object header often dominate, so the shorter Fenwick walk wins by a small factor even though both queries are O(log n). A list copy beats path copying on the clock and loses on nodes. The recursive product cuts misses from 15,624 to 6,528 and still takes longer in milliseconds. That measures the interpreter, not a failure of section 12.6. My block ranks are also fatter than the chapter's packed counters, a 32 bit integer per 16 bit block, which is why the directory outweighs the bits.

# Conclusion

The library does the chapter's jobs: range sum, min, and max with a lazy range add, old versions without a full clone, bit and DNA window counts, and a heap order that misses less. I would use the Fenwick tree for sums and the segment tree for min, max, or a range add. The node counts are the case for persistence. Cache oblivious recursion earned the misses and, here, not the milliseconds.

# Final remarks

Tests check each structure against a direct loop on random inputs, including the sum from index 1 through 4 on 1, 3, 5, 7, 9, 11. The scripts under `examples/` make those calls without the assertions. Where Python misses the chapter's space bound, the tables show the bytes I measured.
