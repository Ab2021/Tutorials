"""KV transfer cost model — tiers, break-even against recompute, and the P:D split.

Nothing here measures anything. The tier bandwidths are MODEL PARAMETERS chosen so the
ordering matches the corpus's description (HBM fastest, then peer, then pooled memory,
then DRAM); the absolute values are illustrative and marked [D]. The one qualitative
claim taken from a transcript -- that pooled-memory sharing was reported faster than RDMA
-- is recorded in the tier's `provenance` field and is a VENDOR REPORT, not a measurement
made here. [T]
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- KV size

LAYERS = 61          # representative large-MoE shape used across this blueprint [D]
KV_HEADS = 8
HEAD_DIM = 128
DTYPE_BYTES = 2      # fp16


def kv_bytes_per_token(layers: int = LAYERS, kv_heads: int = KV_HEADS,
                       head_dim: int = HEAD_DIM, dtype_bytes: int = DTYPE_BYTES) -> float:
    """2 (K and V) x layers x kv_heads x head_dim x dtype. [D]"""
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def kv_bytes(tokens: int, **kw) -> float:
    return tokens * kv_bytes_per_token(**kw)


# --------------------------------------------------------------------------- tiers


@dataclass(frozen=True)
class Tier:
    name: str
    bandwidth_gbps: float     # model parameter, NOT a benchmark [D]
    latency_us: float         # fixed per-transfer overhead [D]
    capacity_gb: float        # illustrative per-node/per-rack share [D]
    cost_rank: int            # ordering only
    provenance: str


TIERS: list[Tier] = [
    Tier("hbm       (local)", 2000.0, 1.0, 0.2, 0, "model parameter"),
    Tier("peer hbm  (p2p)  ", 100.0, 8.0, 1.6, 1, "model parameter"),
    Tier("pooled mem(rack) ", 50.0, 5.0, 8.0, 2,
         "vendor-reported faster than RDMA [T] - not measured here"),
    Tier("cpu dram  (offl) ", 25.0, 12.0, 64.0, 3, "model parameter"),
]


def tier(name_fragment: str) -> Tier:
    for t in TIERS:
        if name_fragment in t.name:
            return t
    raise KeyError(name_fragment)


def transfer_seconds(nbytes: float, t: Tier) -> float:
    """One direction. Bits over bandwidth, plus a fixed per-transfer overhead. [D]"""
    return (nbytes * 8.0) / (t.bandwidth_gbps * 1e9) + t.latency_us / 1e6


def round_trip_seconds(nbytes: float, t: Tier) -> float:
    """Offload then reload. This is what has to beat recompute to be worth doing."""
    return 2.0 * transfer_seconds(nbytes, t)


def recompute_seconds(prompt_tokens: int, prefill_tokens_per_s: float) -> float:
    """Re-running prefill instead of fetching the KV back. [D]"""
    if prefill_tokens_per_s <= 0:
        return float("inf")
    return prompt_tokens / prefill_tokens_per_s


def offload_wins(prompt_tokens: int, nbytes: float, t: Tier,
                 prefill_tokens_per_s: float) -> bool:
    return recompute_seconds(prompt_tokens, prefill_tokens_per_s) > round_trip_seconds(nbytes, t)


def crossover_bandwidth_gbps(prefill_tokens_per_s: float, **kw) -> float:
    """The tier bandwidth at which fetching KV and recomputing it cost the same. O(1).

    Per token, fetching costs 2 x 8 x bytes_per_token / bandwidth (both directions), and
    recomputing costs 1 / prefill_tokens_per_s. Setting them equal:

        bandwidth = 16 x bytes_per_token x prefill_tokens_per_s

    A tier faster than this wins; slower loses. Note that it does NOT depend on prompt
    length -- which is the counter-intuitive part: recompute and transfer both scale
    linearly with tokens, so the comparison is length-independent. What DOES depend on
    length is the fixed per-transfer latency's share, which is why small blocks are
    penalised. [D]
    """
    bpt = kv_bytes_per_token(**kw)
    return (16.0 * bpt * prefill_tokens_per_s) / 1e9


def latency_penalty_seconds(nbytes: float, t: Tier) -> float:
    """The part of the round trip that does not shrink with block size. [D]"""
    return 2.0 * (t.latency_us / 1e6)


def max_tolerable_pause_s(prompt_tokens: int, nbytes: float, t: Tier,
                          prefill_tokens_per_s: float) -> float:
    """Seconds of recompute avoided, minus the transport paid. Positive => fetch wins."""
    return (recompute_seconds(prompt_tokens, prefill_tokens_per_s)
            - round_trip_seconds(nbytes, t))


# --------------------------------------------------------------------------- P:D split


def per_request_work_s(prompt_tokens: int, output_tokens: int,
                       prefill_tps: float, decode_tps: float) -> tuple[float, float]:
    """Seconds of prefill-pool work and decode-pool work for one request. [D]

    NOTE THE UNITS. `prefill_tps` is a single-pass token rate for one worker;
    `decode_tps` is an AGGREGATE token rate for a batched decode worker, because decode
    only reaches a useful rate with a batch resident. They are not the same quantity and
    the model does not pretend they are.
    """
    return (prompt_tokens / max(prefill_tps, 1e-9),
            output_tokens / max(decode_tps, 1e-9))


def pd_split(total_workers: int, prompt_tokens: int, output_tokens: int,
             prefill_tps: float, decode_tps: float) -> tuple[int, int]:
    """How many prefill workers and how many decode workers for a given workload shape.

    A MODEL of the corpus's finding that the best prefill:decode ratio changes with the
    workload and that no single ratio wins. [T] -> encoded here as [D].

    Returns (prefill_workers, decode_workers), at least 1 of each, summing to
    `total_workers`.
    """
    if total_workers < 2:
        raise ValueError("need at least one prefill and one decode worker")
    p, d = per_request_work_s(prompt_tokens, output_tokens, prefill_tps, decode_tps)
    total = p + d
    if total <= 0:
        return total_workers // 2, total_workers - total_workers // 2
    p_count = int(round(total_workers * p / total))
    p_count = max(1, min(total_workers - 1, p_count))
    return p_count, total_workers - p_count


def imbalance(prefill_workers: int, decode_workers: int, prompt_tokens: int,
              output_tokens: int, prefill_tps: float, decode_tps: float) -> float:
    """Load per worker on the busier side, relative to the mean. 1.0 == balanced. [D]"""
    p, d = per_request_work_s(prompt_tokens, output_tokens, prefill_tps, decode_tps)
    p_load = p / max(prefill_workers, 1)
    d_load = d / max(decode_workers, 1)
    mean = (p_load + d_load) / 2.0
    if mean <= 0:
        return 1.0
    return max(p_load, d_load) / mean
