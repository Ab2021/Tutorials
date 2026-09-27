"""KV tiering: HBM -> host DRAM -> SSD -> (recompute), plus the retention API.

The corpus's tiered-transfer ranking is explicit: pooled memory < RDMA << TCP/IP [T]. The
same talk reports ~5x TTFT improvement when a session's KV is restored from CPU rather than
recomputed on session return [T] (llm-d).

The design question this module answers is NOT "is offload faster than recompute" -- it usually
is. It is the narrower and much more useful one: **at what idle gap does offload stop paying for
itself**, and what happens to a request that is restored onto a DIFFERENT engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GB = 1e9


# --------------------------------------------------------------------------------------
# Transfer media
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Medium:
    """A KV transfer path.

    `rank` encodes the corpus's ordering for the three media it ranks explicitly -- pooled
    memory < RDMA << TCP/IP [T]. NVMe is not in the corpus's ranking and sits between RDMA and
    TCP here by bandwidth alone; a real deployment should rank by MEASURED round-trip time,
    because the fixed setup latency dominates for short transfers.
    """
    name: str
    bandwidth: float          # bytes/s, one direction
    latency: float            # seconds, fixed setup per transfer
    rank: int                 # 0 = fastest; ordering is the corpus's, values are [D]

    def transfer_time(self, n_bytes: float) -> float:
        return self.latency + n_bytes / self.bandwidth


# [T] ordering from the rack-scale pooled-KV talk; [D] figures are illustrative and MUST be
# replaced with measured numbers for the operator's own fabric.
POOLED_MEMORY = Medium("pooled-memory", bandwidth=900e9, latency=2e-6, rank=0)
RDMA = Medium("rdma", bandwidth=50e9, latency=5e-6, rank=1)
NVME_LOCAL = Medium("nvme-local", bandwidth=7e9, latency=20e-6, rank=2)
TCP = Medium("tcp", bandwidth=3e9, latency=200e-6, rank=3)


# --------------------------------------------------------------------------------------
# Tiering cost model
# --------------------------------------------------------------------------------------

@dataclass
class Tier:
    name: str
    medium: Medium
    capacity_bytes: float
    used_bytes: float = 0.0
    entries: int = 0

    @property
    def utilisation(self) -> float:
        return self.used_bytes / self.capacity_bytes if self.capacity_bytes else 0.0


def offload_cost(n_bytes: float, medium: Medium) -> float:
    """Wall-clock cost of keeping a session's KV in a tier across an idle gap.

    Counts BOTH directions. A model that counts only the restore hides the write that happens
    on the critical path of the *previous* request, which is where the tail latency appears.
    """
    return medium.transfer_time(n_bytes) * 2


def recompute_cost(seq_len: int, prefill_throughput: float) -> float:
    """Cost of dropping KV and re-prefilling on return.

    `prefill_throughput` is tokens/s for THIS sequence length -- it is not constant, because
    chunked prefill makes short re-prefills relatively cheaper. Callers passing a single
    headline number are making an approximation and should say so.
    """
    if prefill_throughput <= 0:
        raise ValueError("prefill_throughput must be > 0")
    return seq_len / prefill_throughput


def breakeven(seq_len: int, bytes_per_token: float, medium: Medium,
              prefill_throughput: float) -> dict:
    """Offload cost vs recompute cost -- the whole decision, in one comparison.

    Note what is NOT in this formula: the idle gap. A retained block costs nothing while it
    sits in a tier, so gap length does not enter the arithmetic; what the gap governs is
    whether the tier survives at all, which is the TTL's job (RetentionPolicy).

    What DOES enter is the medium. The corpus's ranking -- pooled memory < RDMA << TCP/IP [T]
    -- is not a nicety: on a slow medium the transfer exceeds the re-prefill and tiering is
    strictly worse than dropping. An operator who tiers everything and never computes this has
    added latency to their fastest traffic and recorded it as an optimisation.
    """
    n_bytes = seq_len * bytes_per_token
    off = offload_cost(n_bytes, medium)
    rec = recompute_cost(seq_len, prefill_throughput)
    return {"bytes": n_bytes, "offload_s": off, "recompute_s": rec,
            "offload_wins": off < rec, "ratio": off / rec if rec else float("inf")}


# --------------------------------------------------------------------------------------
# Preemption policy: recompute vs swap
# --------------------------------------------------------------------------------------

def preemption_choice(seq_len: int, bytes_per_token: float, medium: Medium,
                      prefill_throughput: float, expected_gap_s: float = 0.0) -> dict:
    """vLLM-style choice at the moment of preemption.

    Two exits from an out-of-memory condition:
      * RECOMPUTE -- drop the blocks; the sequence re-prefills when it resumes.
      * SWAP      -- move the blocks to a tier; the sequence resumes with KV intact.

    The COST comparison is gap-independent: 2 x transfer vs one re-prefill, exactly. The gap
    governs OCCUPANCY -- a long gap holds tier bytes for longer, so the tier fills and evicts.
    Two questions, two variables, and conflating them is how a fleet ends up swapping
    everything into a tier that evicts it before the sequence resumes.

    The corpus's ~5x TTFT win for the swap path [T] is a SESSION-RETURN figure, where the gap
    is seconds to minutes and occupancy pressure is real. Quoting it for a millisecond
    preemption gap is the most common misreading of that number.
    """
    r = breakeven(seq_len, bytes_per_token, medium, prefill_throughput)
    choice = "swap" if r["offload_wins"] else "recompute"
    return {**r, "expected_gap_s": expected_gap_s, "choice": choice,
            "note": ("cost is gap-independent; occupancy is not. A tier that evicts before "
                     "the sequence resumes is worse than recompute, because it pays both.")}


# --------------------------------------------------------------------------------------
# Retention API
# --------------------------------------------------------------------------------------

class RetentionPolicy:
    """Which sessions' KV is worth keeping, and for how long.

    The corpus states the mechanism plainly: the router consumes per-request create AND EVICT
    events, and an offload tier plus a retention API exist [T]. What it does not give is the
    policy. The policy below is [D].

    The key idea: retention length should be set by the session's RE-ARRIVAL distribution, not
    by a global TTL. A fleet-wide TTL sized for chat (seconds) will evict an agent's tool-call
    prefix (which returns in ~ms) and, worse, will keep a one-shot batch job's KV that will
    never be read again.
    """

    def __init__(self, max_retained_bytes: float, default_ttl_s: float = 300.0):
        self.max_retained_bytes = max_retained_bytes
        self.default_ttl_s = default_ttl_s
        self.used_bytes = 0.0
        self.sessions: dict[str, dict] = {}
        self.events: list[dict] = []       # the stream the router consumes [T]

    def create(self, session_id: str, bytes_kv: float, ttl_s: float | None = None) -> dict:
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        evicted: list[str] = []
        while (self.used_bytes + bytes_kv > self.max_retained_bytes
               and self.sessions):
            victim = min(self.sessions, key=lambda s: self.sessions[s]["expires_at"])
            evicted.append(victim)
            self.used_bytes -= self.sessions[victim]["bytes_kv"]
            del self.sessions[victim]
        self.sessions[session_id] = {"bytes_kv": bytes_kv, "ttl_s": ttl, "expires_at": ttl}
        self.used_bytes += bytes_kv
        ev = {"event": "create", "session_id": session_id, "bytes_kv": bytes_kv,
              "evicted_for_space": evicted}
        self.events.append(ev)
        return ev

    def evict(self, session_id: str, reason: str = "ttl") -> dict:
        s = self.sessions.pop(session_id, None)
        if s:
            self.used_bytes -= s["bytes_kv"]
        ev = {"event": "evict", "session_id": session_id, "reason": reason}
        self.events.append(ev)
        return ev

    def hit(self, session_id: str) -> bool:
        return session_id in self.sessions

    def value_of_retention(self, session_id: str, re_arrival_s: float,
                           bytes_per_token: float, seq_len: int,
                           medium: Medium, prefill_throughput: float) -> dict:
        """Is retaining THIS session worth the memory it occupies?

        Value = the transfer cost avoided on re-arrival, minus the opportunity cost of the
        blocks (which could be serving another sequence). The opportunity cost is stated as a
        rate rather than modelled, because the corpus supplies no concurrency price.
        """
        b = breakeven(seq_len, bytes_per_token, medium, prefill_throughput)
        keep = re_arrival_s < self.default_ttl_s and b["offload_wins"]
        return {"session_id": session_id, "re_arrival_s": re_arrival_s,
                "retain": keep, "reason":
                "re-arrival inside TTL and transfer beats re-prefill" if keep
                else "re-arrival too rare, or re-prefill cheaper"}
