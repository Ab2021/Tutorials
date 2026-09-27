"""Block allocation: contiguous vs paged, and the waste arithmetic.

The corpus's headline: naive contiguous allocation wastes ~60-80% of KV memory; paged
attention wastes <4% [R]. The difference is not a percentage of a bill -- it is the
difference between a concurrency limit of ~8 and one of ~40 on the same hardware.

Paging is table stakes. The interesting design work is everything built ON TOP of it:
sharing, prefix caching, tiering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


BLOCK_SIZE = 16          # [R] typical paged-attention block, in tokens


# --------------------------------------------------------------------------------------
# Paged allocator
# --------------------------------------------------------------------------------------

@dataclass
class BlockTable:
    """A sequence's mapping from logical token positions to physical blocks.

    This indirection IS paged attention: the sequence sees a contiguous address space; the
    allocator sees whatever physical blocks are free. Sharing is then just two tables
    pointing at the same block.
    """
    seq_id: str
    block_size: int = BLOCK_SIZE
    blocks: list[int] = field(default_factory=list)
    n_tokens: int = 0

    def needed_blocks(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def append(self, n_tokens: int, pool) -> int:
        """Extend the sequence. Returns the number of NEW blocks allocated."""
        want = self.needed_blocks(n_tokens)
        new = want - len(self.blocks)
        for _ in range(max(0, new)):
            self.blocks.append(pool.allocate())
        self.n_tokens = n_tokens
        return max(0, new)

    @property
    def capacity_tokens(self) -> int:
        return len(self.blocks) * self.block_size


class BlockPool:
    """A fixed pool of physical blocks, with refcounts so blocks can be SHARED.

    Refcounting is what makes prefix caching and best-of-N sharing possible: two sequences
    with the same prefix point at the same physical blocks, and the block is freed only when
    the last reference goes away.
    """

    def __init__(self, n_blocks: int, block_size: int = BLOCK_SIZE):
        if n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")
        self.block_size = block_size
        self.free = list(range(n_blocks))
        self.refcount = {i: 0 for i in range(n_blocks)}

    @property
    def total(self) -> int:
        return len(self.refcount)

    @property
    def used(self) -> int:
        return sum(1 for v in self.refcount.values() if v > 0)

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("KV pool exhausted -- this is the preemption trigger")
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def retain(self, b: int) -> None:
        """Share an existing block. This is the whole mechanism of prefix reuse."""
        if self.refcount[b] <= 0:
            raise ValueError(f"retain on a free block {b}")
        self.refcount[b] += 1

    def release(self, b: int) -> None:
        if self.refcount[b] <= 0:
            raise ValueError(f"release on a free block {b}")
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)

    def free_blocks(self) -> int:
        return len(self.free)


# --------------------------------------------------------------------------------------
# Waste arithmetic
# --------------------------------------------------------------------------------------

def paged_waste(seq_len: int, block_size: int = BLOCK_SIZE) -> float:
    """Internal fragmentation of one sequence, as a fraction of its allocated blocks."""
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    alloc = math.ceil(seq_len / block_size) * block_size
    return (alloc - seq_len) / alloc


def contiguous_waste(seq_len: int, max_seq_len: int) -> float:
    """Contiguous allocation must reserve the MAXIMUM length for every sequence.

    This is the ~60-80% the corpus reports [R]: not internal fragmentation, but reserving
    for the worst case that almost never arrives.
    """
    if seq_len <= 0 or max_seq_len <= 0:
        raise ValueError("lengths must be > 0")
    if seq_len > max_seq_len:
        raise ValueError("seq_len cannot exceed max_seq_len")
    return (max_seq_len - seq_len) / max_seq_len


def expected_waste(lengths: list[int], max_seq_len: int,
                   block_size: int = BLOCK_SIZE) -> dict:
    """Waste over a length distribution, reported BOTH ways.

    Mean-of-ratios and ratio-of-sums are different numbers, and confusing them is the standard
    way this arithmetic goes wrong:

      * ratio-of-sums  = total wasted tokens / total allocated tokens. This is a CAPACITY
        figure: it is how much of the pool is wasted, and it is what the corpus's "<4%" [R]
        refers to.
      * mean-of-ratios = average over sequences of each sequence's own waste fraction. Very
        short sequences round up to a whole block and show enormous waste individually, so
        this number is much larger -- and it is a FAIRNESS figure, not a capacity one.

    Both are reported because a team that only reads the mean-of-ratios will conclude paging
    does not work, and a team that only reads the ratio-of-sums will never notice that their
    short-request traffic is the expensive part.
    """
    if not lengths:
        raise ValueError("empty length distribution")

    # ratio-of-sums (capacity)
    tot_alloc_cp = sum(max_seq_len for _ in lengths)
    tot_used = sum(lengths)
    cp_agg = (tot_alloc_cp - tot_used) / tot_alloc_cp

    alloc_p = [math.ceil(s / block_size) * block_size for s in lengths]
    pp_agg = (sum(alloc_p) - tot_used) / sum(alloc_p)

    # mean-of-ratios (fairness)
    cp_mean = sum(contiguous_waste(s, max_seq_len) for s in lengths) / len(lengths)
    pp_mean = sum(paged_waste(s, block_size) for s in lengths) / len(lengths)

    return {"contiguous_agg": cp_agg, "paged_agg": pp_agg,
            "contiguous_mean": cp_mean, "paged_mean": pp_mean,
            "ratio_agg": cp_agg / pp_agg if pp_agg > 0 else float("inf"),
            "ratio_mean": cp_mean / pp_mean if pp_mean > 0 else float("inf")}


def make_length_distribution(rng, n: int = 4000, max_seq_len: int = 2048,
                             shape: float = 4.0) -> list[int]:
    """A skewed prompt-length distribution.

    Real traffic is skewed short. `shape` controls the skew: larger means most requests are
    much shorter than max_seq_len, which is exactly the regime where contiguous allocation
    wastes the most and therefore the case the corpus's 60-80% figure describes.
    """
    out = []
    for _ in range(n):
        u = rng.random()
        v = u ** shape
        out.append(max(1, int(1 + v * (max_seq_len - 1))))
    return out


def block_size_table(lengths: list[int], sizes=(4, 8, 16, 32, 64, 128)) -> list[dict]:
    """The block-size trade, at the AGGREGATE (capacity) measure.

    Internal fragmentation falls as blocks shrink, but the block TABLE grows (one entry per
    block) and per-block kernel overhead rises. Reporting the aggregate rather than the
    mean-of-ratios matters: the mean is dominated by very short sequences and would make every
    block size look bad, which hides the actual shape of the curve.
    """
    rows = []
    used = sum(lengths)
    for bs in sizes:
        alloc = sum(math.ceil(s / bs) * bs for s in lengths)
        waste = (alloc - used) / alloc
        entries = sum(math.ceil(s / bs) for s in lengths) / len(lengths)
        rows.append({"block_size": bs, "waste": waste, "mean_table_entries": entries})
    return rows
