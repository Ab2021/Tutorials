"""Paged KV allocator with static versus dynamic partitioning.

The mechanism this module exists to prove:

  Modern agent-era models are HYBRID -- full attention interleaved with a cheaper
  mechanism (sliding window, or linear attention such as KDA). Their memory behaviour is
  completely different: full-attention KV grows linearly with context, while a linear
  attention keeps a fixed-size state per sequence regardless of context length. [T]

  The naive answer is a STATIC split of the GPU memory pool between the two. The problem
  the corpus names is that the optimal split depends on batch size and context length,
  which are dynamic over time. The engine's answer is DYNAMIC partitioning: one shared
  pool, one allocator per attention type, drawing from the same space. [T]

This module models the difference as fragmentation: a static reservation is capacity held
for a type that is not using it. It measures nothing and no GPU is involved. [D]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

TOKENS_PER_BLOCK = 16      # paged KV block size; a tuning knob, not a constant [D]

# A linear-attention layer keeps a FIXED-SIZE state per sequence, whatever the context.
# That is what makes 1M context feasible at all [T]. It is fixed, not small: modelled
# here as a constant number of blocks per sequence, so a pool full of concurrent
# sequences still has to budget for it. [D]
LINEAR_BLOCKS_PER_SEQ = 24


@dataclass(frozen=True)
class Request:
    rid: int
    ctx_tokens: int
    hold_steps: int        # how many scheduling steps the request occupies its blocks

    @property
    def full_blocks(self) -> int:
        """Full attention grows linearly with context: ceil(ctx / tokens_per_block)."""
        return max(1, math.ceil(self.ctx_tokens / TOKENS_PER_BLOCK))

    @property
    def linear_blocks(self) -> int:
        """A fixed-size state per sequence, independent of context length. [T]"""
        return LINEAR_BLOCKS_PER_SEQ

    @property
    def total_blocks(self) -> int:
        return self.full_blocks + self.linear_blocks


@dataclass
class Outcome:
    admitted: int = 0
    rejected: int = 0
    peak_concurrent: int = 0
    admitted_ids: list[int] = field(default_factory=list)
    wait_steps: list[int] = field(default_factory=list)   # steps each admitted request waited
    # blocks reserved to an attention type but not in use, summed over steps
    idle_reserved_block_steps: int = 0
    used_block_steps: int = 0

    @property
    def pool_utilisation(self) -> float:
        total = self.used_block_steps + self.idle_reserved_block_steps
        return self.used_block_steps / total if total else 0.0

    @property
    def mean_wait_steps(self) -> float:
        return sum(self.wait_steps) / len(self.wait_steps) if self.wait_steps else 0.0

    @property
    def p95_wait_steps(self) -> int:
        if not self.wait_steps:
            return 0
        s = sorted(self.wait_steps)
        return s[min(len(s) - 1, int(0.95 * len(s)))]


class Pool:
    """A block pool. `split` is the fraction of blocks reserved for FULL attention.

    split is None  -> dynamic partitioning: one shared pool, no reservation.
    split is a float -> static partitioning: that share is reserved and cannot be lent.
    """

    def __init__(self, total_blocks: int, split: float | None,
                 max_seq_share: float | None = None):
        self.total = total_blocks
        self.split = split
        # A shared pool needs admission control. Without it a single enormous sequence
        # can take most of the pool and block everyone else -- which is how a dynamically
        # partitioned pool can lose to a well-chosen static split. This is the complement,
        # not a refinement. [D]
        self.max_seq_blocks = (int(total_blocks * max_seq_share)
                               if max_seq_share is not None else None)
        if split is None:
            self.full_cap = total_blocks
            self.linear_cap = total_blocks      # both draw from the same space
        else:
            self.full_cap = int(total_blocks * split)
            self.linear_cap = total_blocks - self.full_cap
        self.full_used = 0
        self.linear_used = 0

    # -- static partitioning keeps two separate ledgers; dynamic shares one --------

    def free_for(self, kind: str) -> int:
        if self.split is None:
            return self.total - self.full_used - self.linear_used
        if kind == "full":
            return self.full_cap - self.full_used
        return self.linear_cap - self.linear_used

    def can_admit(self, r: Request) -> bool:
        if self.max_seq_blocks is not None and r.full_blocks > self.max_seq_blocks:
            return False
        if self.split is None:
            return self.free_for("full") >= r.total_blocks
        # Static: each type is checked against its OWN reservation, so a request is
        # rejected while the other partition sits idle. That is the defect being modelled.
        return (self.free_for("full") >= r.full_blocks
                and self.free_for("linear") >= r.linear_blocks)

    def admit(self, r: Request) -> None:
        self.full_used += r.full_blocks
        self.linear_used += r.linear_blocks

    def release(self, r: Request) -> None:
        self.full_used -= r.full_blocks
        self.linear_used -= r.linear_blocks

    def idle_reserved(self) -> int:
        """Reserved-but-unused blocks. Zero by construction under dynamic partitioning."""
        if self.split is None:
            return 0
        return (self.full_cap - self.full_used) + (self.linear_cap - self.linear_used)


@dataclass
class Scenario:
    n_requests: int = 300
    total_blocks: int = 4096
    # Context lengths are heavy-tailed: most turns are short, some are enormous. An
    # agent session grows over hundreds of turns, so the mix shifts over time. [T]
    short_ctx: int = 2_000
    long_ctx: int = 64_000
    long_fraction: float = 0.25
    hold_steps: int = 8
    arrival_spread: int = 40     # requests arrive across this many steps
    seed: int = 11
    max_seq_share: float | None = 0.5   # cap any one sequence's share of the pool [D]


def make_requests(sc: Scenario) -> list[tuple[int, Request]]:
    """(arrival_step, request) pairs, sorted by arrival."""
    rng = random.Random(sc.seed)
    out = []
    for i in range(sc.n_requests):
        ctx = sc.long_ctx if rng.random() < sc.long_fraction else sc.short_ctx
        # jitter so the two cohorts do not move in lockstep
        ctx = int(ctx * rng.uniform(0.5, 1.5))
        out.append((rng.randint(0, sc.arrival_spread), Request(i, ctx, sc.hold_steps)))
    out.sort(key=lambda p: p[0])
    return out


def simulate(sc: Scenario, split: float | None) -> Outcome:
    """Step-driven. At each step: release finished, then try to admit waiting requests."""
    arrivals = make_requests(sc)
    arrivals_by_id = {r.rid: step for step, r in arrivals}
    pool = Pool(sc.total_blocks, split, sc.max_seq_share)
    out = Outcome()
    active: list[tuple[int, Request]] = []   # (finish_step, request)
    waiting: list[Request] = []
    idx = 0
    total_steps = sc.arrival_spread + sc.hold_steps * 6
    for step in range(total_steps):
        # 1. release
        still = []
        for finish, r in active:
            if finish <= step:
                pool.release(r)
            else:
                still.append((finish, r))
        active = still
        # 2. arrive
        while idx < len(arrivals) and arrivals[idx][0] <= step:
            waiting.append(arrivals[idx][1])
            idx += 1
        # 3. admit as many as fit
        remaining = []
        for r in waiting:
            if pool.can_admit(r):
                pool.admit(r)
                active.append((step + r.hold_steps, r))
                out.admitted += 1
                out.admitted_ids.append(r.rid)
                out.wait_steps.append(step - arrivals_by_id[r.rid])
            else:
                remaining.append(r)
        waiting = remaining
        # 4. account
        out.peak_concurrent = max(out.peak_concurrent, len(active))
        out.used_block_steps += pool.full_used + pool.linear_used
        out.idle_reserved_block_steps += pool.idle_reserved()
    out.rejected = len(waiting)
    return out


def sweep_split(sc: Scenario, splits) -> list[tuple[float, Outcome]]:
    return [(s, simulate(sc, s)) for s in splits]


def render_split_sweep(rows) -> str:
    lines = [f"  {'full reserve':>13} {'admitted':>9} {'rejected':>9} {'peak conc':>10} "
             f"{'pool util':>10}"]
    lines.append("  " + "-" * 56)
    for s, o in rows:
        lines.append(f"  {s:>12.0%} {o.admitted:>9} {o.rejected:>9} "
                     f"{o.peak_concurrent:>10} {o.pool_utilisation:>10.1%}")
    return "\n".join(lines)
