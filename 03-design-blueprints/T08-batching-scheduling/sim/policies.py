"""Chunked prefill, admission control, and goodput.

Three related decisions that all live at the same layer as batching:

  chunking   -- split a long prompt so it does not stall decode (trades TTFT for ITL stability)
  admission  -- refuse or queue when saturated, instead of degrading gracefully into uselessness
  goodput    -- the objective every one of these decisions is judged against
"""
from __future__ import annotations

import math

from .scheduler import Seq, run_continuous, completion_stats, throughput


# --------------------------------------------------------------------------------------
# Chunked prefill
# --------------------------------------------------------------------------------------

def prefill_chunk_size(max_num_batched_tokens: int, decode_seqs: int) -> int:
    """`prefill_chunk_size ~= max_num_batched_tokens - (decode_seqs x 1)`.

    A decode step consumes one token per running sequence; whatever budget remains can go to
    prefill. This is why the budget must be sized against the CONCURRENT decode batch rather than
    picked in isolation.

    Too small -> the prompt is chopped into many chunks and TTFT suffers.
    Too large -> the prefill chunk displaces decode work and ITL spikes.
    """
    return max(0, max_num_batched_tokens - decode_seqs)


def chunk_plan(prompt_tokens: int, chunk: int) -> dict:
    """How a prompt is split, and where the TTFT/ITL trade lands."""
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    n = math.ceil(prompt_tokens / chunk)
    return {"prompt_tokens": prompt_tokens, "chunk": chunk, "chunks": n,
            # Unchunked, the whole prompt is ONE step: everyone else waits for it.
            "unchunked_steps_blocked": 1,
            "chunked_steps_blocked": n,
            "note": ("unchunked: one very long step, so max ITL spike = full prefill.\n"
                     "chunked:   n short steps interleaved with decode, so the spike is one "
                     "chunk -- but the prompt finishes n steps later, which is the TTFT cost.")}


def chunking_effect(prompt_lens: list[int], decode_seqs: int,
                    budget: int, tps: int = 100) -> list[dict]:
    """The trade, per prompt length, both ways."""
    out = []
    for p in prompt_lens:
        unchunked_itl_spike = math.ceil(p / tps)
        chunk = prefill_chunk_size(budget, decode_seqs)
        plan = chunk_plan(p, chunk) if chunk else {"chunks": 1}
        out.append({"prompt": p, "chunk": chunk, "chunks": plan["chunks"],
                    "unchunked_itl_spike_steps": unchunked_itl_spike,
                    "chunked_itl_spike_steps": 1,
                    "ttft_steps_added": max(0, plan["chunks"] - 1)})
    return out


# --------------------------------------------------------------------------------------
# Admission control
# --------------------------------------------------------------------------------------

def saturation_gate(kv_usage: float, active_requests: int,
                    kv_threshold: float = 0.80, active_threshold: int = 8) -> dict:
    """The corpus's saturation test [T] (llm-d): KV ~80% full, or average active requests > 8.

    WHY IT IS PART OF SCHEDULING AND NOT PART OF AUTOSCALING: without a gate the system does not
    shed load, it degrades gracefully into uselessness -- every request gets slower, so nothing
    fails, so nothing alerts, so no capacity is added. A gate converts a latency collapse into a
    queue, which is a failure mode an operator can see and act on.
    """
    by_kv = kv_usage > kv_threshold
    by_active = active_requests > active_threshold
    return {"kv_usage": kv_usage, "active": active_requests,
            "saturated": by_kv or by_active,
            "reason": ("kv" if by_kv else "active" if by_active else "none")}


# --------------------------------------------------------------------------------------
# Goodput
# --------------------------------------------------------------------------------------

def goodput(latencies: list[float], slo: float) -> float:
    """Fraction of requests meeting the SLO. THE scheduling objective.

    Not throughput. Throughput rises monotonically with batch size, so it can never tell you that
    you have batched too far -- goodput peaks and falls, which is exactly the signal you need.
    """
    if not latencies:
        return 0.0
    return sum(1 for l in latencies if l <= slo) / len(latencies)


def batch_sweep(seqs: list[Seq], slot_options: list[int], slo: float,
                slos: tuple[float, ...] = (), tps: int = 100) -> list[dict]:
    """Throughput and goodput across batch sizes, so the DIVERGENCE is visible.

    Each row is the same traffic served at a different concurrency limit. Report goodput at
    SEVERAL SLOs, because the batch size that maximises it is a function of the SLO as much as of
    the system: a single-SLO column invites the reader to treat the peak as a property of the
    engine, which it is not.
    """
    probe = slos or (slo,)
    rows = []
    for slots in slot_options:
        work = [Seq(f"s{i}", s.prompt_tokens, s.output_tokens, session=s.session,
                    arrival=s.arrival) for i, s in enumerate(seqs)]
        st = run_continuous(work, slots, tps_per_slot=tps)
        stats = completion_stats(st)
        lat = [(s.finish or 0) - s.arrival for s in st.done]
        rows.append({"slots": slots, "throughput": throughput(st, tps),
                     "saturated_throughput": st.saturated_throughput(),
                     "p50": stats.get("p50", 0), "p99": stats.get("p99", 0),
                     "goodput": goodput(lat, slo),
                     "goodput_at": {s: goodput(lat, s) for s in probe},
                     "makespan": st.step, "n": len(lat),
                     "idle_fraction": st.idle_fraction()})
    return rows
