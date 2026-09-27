"""Latency decomposition and goodput.

    E2E = TTFT + ITL * N_out

TTFT is set by PREFILL; ITL is set by DECODE. The model's job is to say which of the two
levers actually moves a given budget -- and at long output lengths, TTFT is usually NOT it.

Goodput, not throughput, is the headline metric: a configuration can double throughput while
halving goodput by batching so aggressively that latency blows out.
"""
from __future__ import annotations


def ttft(n_prompt: int, prefill_throughput: float) -> float:
    """Seconds. `prefill_throughput` is prompt tokens/s for the deployment."""
    if n_prompt < 0:
        raise ValueError("n_prompt must be >= 0")
    if prefill_throughput <= 0:
        raise ValueError("prefill_throughput must be > 0")
    return n_prompt / prefill_throughput


def decode_throughput(hw_memory_bw: float, weights_bytes: float, batch: int = 1) -> float:
    """Tokens/s for the batch, under the memory-bound model.

    Decode reads `weights_bytes` once per step for the WHOLE batch, so:
        tokens/s (batch) = memory_bw / weights_bytes * batch
    This is the roofline result written as a throughput, and it is why batching helps decode
    so much: the weight read is shared.
    """
    if weights_bytes <= 0:
        raise ValueError("weights_bytes must be > 0")
    if batch <= 0:
        raise ValueError("batch must be > 0")
    return (hw_memory_bw / weights_bytes) * batch


def itl(hw_memory_bw: float, weights_bytes: float, batch: int = 1,
        kv_slope: float = 0.0) -> float:
    """Seconds between tokens.

    Under the PURE memory-bound model (kv_slope = 0) this is weights_bytes / bandwidth and is
    INDEPENDENT of batch -- the weight read is shared, so adding sequences adds throughput
    without adding latency. That is the clean statement of why batching is the win.

    `kv_slope` adds the effect the pure model omits: the KV read and the attention compute do
    grow with batch, so real ITL rises somewhat. Set it to 0 to see the roofline result in
    isolation, and non-zero to see the trade.
    """
    if weights_bytes <= 0:
        raise ValueError("weights_bytes must be > 0")
    if batch <= 0:
        raise ValueError("batch must be > 0")
    base = weights_bytes / hw_memory_bw
    return base * (1.0 + kv_slope * (batch - 1))


def queue_wait(batch: int, qps: float) -> float:
    """Mean wait for a batch to fill: a request arriving at random waits ~(B-1)/2 arrivals.

    This is the term that makes throughput tuning dangerous: a larger batch raises this, and
    it lands directly on TTFT.
    """
    if qps <= 0:
        raise ValueError("qps must be > 0")
    return max(0.0, (batch - 1) / (2.0 * qps))


def e2e(ttft_s: float, itl_s: float, n_out: int) -> float:
    return ttft_s + itl_s * n_out


def required_itl(budget_s: float, ttft_s: float, n_out: int) -> float:
    """The ITL a budget demands. Raises if the budget is unachievable -- returning a
    negative latency would let a modelling error propagate into a config file."""
    if n_out <= 0:
        raise ValueError("n_out must be > 0")
    v = (budget_s - ttft_s) / n_out
    if v <= 0:
        raise ValueError(
            f"budget {budget_s}s is already exceeded by TTFT {ttft_s}s alone; "
            f"no positive ITL satisfies it"
        )
    return v


def required_output_len(budget_s: float, ttft_s: float, itl_s: float) -> float:
    """The OTHER lever. Often the cheaper one."""
    if itl_s <= 0:
        raise ValueError("itl_s must be > 0")
    return max(0.0, (budget_s - ttft_s) / itl_s)


def goodput(requests: list[dict], ttft_slo: float, itl_slo: float) -> float:
    """|{r : TTFT(r) <= T_t AND ITL(r) <= T_i}| / |requests|.

    BOTH parts are required: a request with excellent TTFT and terrible ITL fails, and so
    does the reverse. A one-part SLO cannot express streaming quality.
    """
    if not requests:
        return 0.0
    ok = sum(1 for r in requests if r["ttft"] <= ttft_slo and r["itl"] <= itl_slo)
    return ok / len(requests)


def throughput_proxy(requests: list[dict]) -> float:
    """Total tokens delivered, regardless of whether the SLO was met.

    This is the metric that can improve while users suffer. It is here so the experiment can
    exhibit the divergence rather than assert it.
    """
    return sum(r.get("n_out", 0) for r in requests)
