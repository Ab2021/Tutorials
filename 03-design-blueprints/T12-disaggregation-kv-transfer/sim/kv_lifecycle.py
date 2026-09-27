"""KV block lifecycle simulation — eviction, retention, and thrash.

This is the mechanism the blueprint exists to prove:

  * eviction is a NORMAL stage, not a failure;
  * *thrash* is the failure, and it is a rate, not a state;
  * a bounded retention lease converts "the session will come back" from a guess into a
    measurable hit-rate change;
  * and retention competes with active traffic for the same capacity, so more pinning is
    not monotonically better.

Deterministic: seeded, no wall-clock, no I/O. Every number is a simulation output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- model


@dataclass
class Block:
    key: tuple
    session: str
    last_used: float
    uses: int = 1
    pinned_until: float | None = None
    lease_redeemed: bool = False   # was this lease ever followed by an access?

    @property
    def pinned(self) -> bool:
        return self.pinned_until is not None


@dataclass
class Stats:
    hits: int = 0
    misses: int = 0
    creations: int = 0
    evictions: int = 0
    thrash_evictions: int = 0     # evicted after proving reusable (uses >= 2)
    pin_preemptions: int = 0      # a pinned block evicted anyway (quota exhausted)
    pin_grants: int = 0           # unpinned -> pinned transitions
    pin_hits: int = 0             # grants that were redeemed by a later access
    pin_denials: int = 0

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def thrash_rate(self) -> float:
        """Share of evictions that destroyed a block with a reuse history."""
        return self.thrash_evictions / self.evictions if self.evictions else 0.0

    @property
    def pin_efficiency(self) -> float:
        """Fraction of granted leases that were ever redeemed by a later access.

        This is the number that exposes over-pinning. A lease granted to a session that
        never comes back consumes capacity and returns nothing; it shows up here as
        efficiency falling while hit rate does not rise. [D]
        """
        return self.pin_hits / self.pin_grants if self.pin_grants else 0.0


# --------------------------------------------------------------------------- cache


class KvCache:
    """A capacity-bounded block store with a bounded retention lease.

    Policies:
      "recompute_only" -- a block is released the moment its session pauses.
      "retain"         -- a pausing session's blocks are pinned for a bounded TTL.
    """

    def __init__(self, capacity: int, policy: str, pin_fraction: float) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1 block")
        if policy not in ("recompute_only", "retain"):
            raise ValueError(f"unknown policy: {policy}")
        self.capacity = capacity
        self.policy = policy
        self.pin_capacity = int(capacity * pin_fraction)
        self.blocks: dict[tuple, Block] = {}
        self.stats = Stats()

    # ---------------------------------------------------------------- internals

    def _expire(self, now: float) -> None:
        for b in self.blocks.values():
            if b.pinned_until is not None and b.pinned_until <= now:
                b.pinned_until = None

    def _pinned_count(self) -> int:
        return sum(1 for b in self.blocks.values() if b.pinned)

    def _choose_victim(self, now: float) -> Block | None:
        """Deterministic order: expired lease, then unpinned LRU, then pinned LRU.

        The third tier only happens when the pin quota was already exceeded -- which is
        what `pin_preemptions` counts, and it should stay at zero.
        """
        expired = [b for b in self.blocks.values()
                   if b.pinned_until is not None and b.pinned_until <= now]
        if expired:
            return min(expired, key=lambda b: b.last_used)
        unpinned = [b for b in self.blocks.values() if not b.pinned]
        if unpinned:
            return min(unpinned, key=lambda b: b.last_used)
        if self.blocks:
            self.stats.pin_preemptions += 1
            return min(self.blocks.values(), key=lambda b: b.last_used)
        return None

    def _ensure_room(self, now: float) -> None:
        while len(self.blocks) >= self.capacity:
            victim = self._choose_victim(now)
            if victim is None:
                return
            self._evict(victim, now)

    def _evict(self, block: Block, now: float) -> None:
        self.blocks.pop(block.key, None)
        self.stats.evictions += 1
        # Thrash is evicting a block that HAD proven reusable. Evicting a never-reused
        # block is the policy working, not failing -- so it is excluded deliberately.
        if block.uses >= 2:
            self.stats.thrash_evictions += 1

    # ---------------------------------------------------------------- api

    def access(self, key: tuple, session: str, now: float) -> bool:
        """Return True on hit. A miss is not an error -- the caller recomputes."""
        self._expire(now)
        b = self.blocks.get(key)
        if b is not None and not (b.pinned_until is not None and b.pinned_until <= now):
            b.last_used = now
            b.uses += 1
            self.stats.hits += 1
            if b.pinned_until is not None and not b.lease_redeemed:
                b.lease_redeemed = True       # this lease earned its keep
                self.stats.pin_hits += 1
            return True
        self.stats.misses += 1
        self._ensure_room(now)
        self.blocks[key] = Block(key=key, session=session, last_used=now)
        self.stats.creations += 1
        return False

    def pause(self, session: str, now: float, ttl_s: float) -> None:
        """Session goes away for a while. This is where the two policies diverge."""
        owned = [b for b in self.blocks.values() if b.session == session]
        if self.policy == "recompute_only":
            for b in owned:
                self._evict(b, now)     # eviction here is normal, not a failure
            return
        for b in owned:
            already = b.pinned_until is not None and b.pinned_until > now
            if not already and self._pinned_count() >= self.pin_capacity:
                self.stats.pin_denials += 1
                # Denied a lease -> the block stays resident but unprotected. That is a
                # shorter effective retention, and it is reported, never hidden.
                continue
            if not already:
                self.stats.pin_grants += 1
                b.lease_redeemed = False      # a NEW lease starts unredeemed
            b.pinned_until = now + ttl_s      # refresh extends the same lease

    def resident(self) -> int:
        return len(self.blocks)


# --------------------------------------------------------------------------- scenario


@dataclass
class Scenario:
    n_sessions: int = 40
    turns: int = 6
    turn_gap_s: float = 2.0
    pause_s: float = 60.0
    capacity: int = 64
    pin_fraction: float = 0.10
    policy: str = "retain"
    seed: int = 7
    abandon_rate: float = 0.25   # share of sessions that leave and never come back
    abandon_lease_mult: float = 10.0   # how long a departed session's pin is held [D]


def simulate(sc: Scenario) -> Stats:
    """Each session's context GROWS: on turn k it re-reads blocks 0..k and adds block k.

    That is the agentic shape, and the prefix is the point. A session does work, calls a
    tool, and comes back to a context it already paid to prefill. On the return, blocks
    0..k-1 are cache HITS if they survived the pause -- which is exactly what a retention
    lease is for. A block per turn with a fresh key would make every access a miss and the
    model would prove nothing.
    """
    rng = random.Random(sc.seed)
    cache = KvCache(sc.capacity, sc.policy, sc.pin_fraction)
    for s in range(sc.n_sessions):
        session = f"s{s}"
        start = rng.uniform(0.0, sc.turn_gap_s)
        if rng.random() < sc.abandon_rate:
            # The session builds a little context and is never seen again. Under retain
            # it is pinned anyway for a long lease, occupying capacity that returns
            # nothing -- this is the cost side of the pin trade, and it is deliberately
            # in the model rather than assumed away.
            cache.access((session, 0), session, start)
            cache.pause(session, start, ttl_s=sc.pause_s * sc.abandon_lease_mult)
            continue
        for turn in range(sc.turns):
            at = start + turn * (sc.turn_gap_s + sc.pause_s)
            for j in range(turn + 1):          # re-read the prefix, then extend it
                cache.access((session, j), session, at)
            cache.pause(session, at, ttl_s=sc.pause_s * 1.5)
    return cache.stats


def sweep_capacity(base: Scenario, capacities) -> list[tuple[int, Stats, Stats]]:
    """Run both policies at each capacity, so the retention benefit is visible as a delta."""
    rows = []
    for c in capacities:
        a = Scenario(**{**base.__dict__, "capacity": c, "policy": "recompute_only"})
        b = Scenario(**{**base.__dict__, "capacity": c, "policy": "retain"})
        rows.append((c, simulate(a), simulate(b)))
    return rows


def render_sweep(rows) -> str:
    out = [f"  {'capacity':>9} {'hits(no-retain)':>16} {'hits(retain)':>13} "
           f"{'delta':>8} {'thrash(retain)':>15}"]
    out.append("  " + "-" * 66)
    for cap, a, b in rows:
        d = b.hit_rate - a.hit_rate
        out.append(f"  {cap:>9} {a.hit_rate:>15.1%} {b.hit_rate:>12.1%} "
                   f"{d:>+7.1%} {b.thrash_rate:>14.1%}")
    return "\n".join(out)
