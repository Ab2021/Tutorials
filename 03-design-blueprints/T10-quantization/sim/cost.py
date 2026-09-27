"""The corpus's cost ladder (100 -> 42 -> 26 -> 11), reproduced from the mechanism rather than quoted.

The corpus gives four stacked figures [T]:

    "Start with naive 16-bit serving one request at a time as 100 cost units. Turn on continuous
     batching and you are near 42. Quantize to 4-bit and you reach 26. Cache the stable system
     prompt and you land around 11."

Those are the anchors. What this module does NOT do is pretend to derive them from first
principles -- the ladder is an illustration, and its four numbers are a composite over a workload
mix the transcript does not state. What it DOES do is build a cost model from the mechanisms
(bytes moved, roofline, cache hits) and then ask, for each step, **what configuration reproduces
that step's multiplier**. The answer is informative, and one of the three is not what a reader would
guess.

Units: one "cost unit" is the bytes moved by ONE decode step at batch 1 -- i.e. one full read of
the weights. Everything else is expressed relative to that.
"""
from __future__ import annotations

from .precision import effective_bits, kv_bytes_per_token

# The corpus's ladder [T]. ANCHORS, not measurements: this module reproduces them, it does not
# validate them.
CORPUS_LADDER = {"naive_bf16": 100.0, "continuous_batching": 42.0,
                 "four_bit_weights": 26.0, "prefix_cached": 11.0}


def _reference_bytes(params_b: float) -> float:
    """The FIXED unit: bytes moved by one bf16 decode step at batch 1.

    This must NOT be recomputed per precision. An earlier version of this module normalised each
    configuration by its OWN weight size, which made quantizing the weights look MORE expensive:
    the decode term is a ratio whose denominator is `W`, so shrinking `W` inflated the ratio even
    as it shrank the bytes. The result was a ladder that went 100 -> 42 -> 71, i.e. "4-bit
    quantization costs more than not quantizing". The reference has to be a constant for the
    rungs to be comparable at all.
    """
    return params_b * 1e9 * 16 / 8.0


def prefill_cost_per_token(params_b: float, machine_balance: float,
                           weight_bits: int = 16,
                           weight_group: int | None = None) -> float:
    """Prefill cost for one prompt token, in weight-read units.

    Prefill is compute-bound (T06), so its cost is FLOPs: `2N` per token, INDEPENDENT of the weight
    precision. Converting to the reference unit (one bf16 weight read) gives

        prefill_cost_per_token = (2N / balance) / W_bf16

    The precision arguments are accepted and IGNORED, deliberately, so that callers written against
    the older signature keep working -- but the honest model is that quantizing weights does not
    reduce prefill FLOPs. It may improve achieved FLOPs/second, which is a hardware effect this
    offline model does not claim.
    """
    n_params = params_b * 1e9
    return (2.0 * n_params / machine_balance) / _reference_bytes(params_b)


def cost_per_request(prompt_len: int, output_len: int, batch: int, params_b: float = 8.0,
                     machine_balance: float = 295.0, ctx_len: int | None = None,
                     weight_bits: int = 16, kv_bits: int = 16, n_layers: int = 32,
                     n_kv_heads: int = 8, head_dim: int = 128, weight_group: int | None = None,
                     cache_hit: float = 0.0) -> dict:
    """Cost of one request in weight-read units, with the three levers applied.

        decode  = output_len x (W/batch + ctx*kv_per_token) / W_bf16     <- batching + quantization
        prefill = prompt_len x (1 - cache_hit) x prefill_per_token       <- prefix caching

    **The three levers act on different terms, and that is the whole content of the ladder:**

      * batching divides the WEIGHT term (`W/batch`) -- it amortizes the model across sequences;
      * weight quantization shrinks `W` in the decode term only, and it shrinks the reference
        proportionally so the SAVING is real;
      * KV quantization shrinks the `ctx*kv_per_token` term only;
      * prefix caching removes PREFILL entirely, so its value is proportional to how much of the
        work is prefill.

    The last point is the finding. Caching is the largest single step in the corpus's ladder, and
    it is -- but only for a workload whose prompt dominates its output.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if not 0.0 <= cache_hit <= 1.0:
        raise ValueError("cache_hit must be in [0, 1]")
    ctx_len = ctx_len if ctx_len is not None else prompt_len + output_len
    ref = _reference_bytes(params_b)
    w_bytes = params_b * 1e9 * effective_bits(weight_bits, weight_group) / 8.0
    kv_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_bits)

    decode = output_len * (w_bytes / batch + ctx_len * kv_tok) / ref
    prefill = prompt_len * (1.0 - cache_hit) * prefill_cost_per_token(params_b, machine_balance)
    return {"decode": decode, "prefill": prefill, "total": decode + prefill,
            "prefill_share": prefill / (decode + prefill) if (decode + prefill) else 0.0}


def ladder(prompt_len: int = 4000, output_len: int = 100, batch: int | None = None,
           cache_hit: float = 0.95, params_b: float = 8.0, **kw) -> list[dict]:
    """The four corpus rungs, computed from `cost_per_request` and normalised to the first.

    The third rung is WEIGHTS-ONLY 4-bit, because that is what the corpus's toolchain line means
    ("AWQ and GPTQ to quantize" -- weight quantization [T]). Quantizing the KV cache as well is a
    FIFTH lever the ladder does not include, and experiment 6 shows it is the larger one for
    concurrency. `solve_batch` recovers the batch depth the default implies, so it is not a hidden
    tuning knob.
    """
    if batch is None:
        batch = int(round(solve_batch(0.42, prompt_len, output_len, params_b=params_b, **kw)))
    base = cost_per_request(prompt_len, output_len, 1, params_b, weight_bits=16, kv_bits=16, **kw)
    rungs = [
        ("naive_bf16",          dict(batch=1, weight_bits=16, kv_bits=16, cache_hit=0.0)),
        ("continuous_batching", dict(batch=batch, weight_bits=16, kv_bits=16, cache_hit=0.0)),
        ("four_bit_weights",    dict(batch=batch, weight_bits=4, kv_bits=16, cache_hit=0.0,
                                     weight_group=128)),
        ("prefix_cached",       dict(batch=batch, weight_bits=4, kv_bits=16, cache_hit=cache_hit,
                                     weight_group=128)),
    ]
    rows = []
    for name, cfg in rungs:
        c = cost_per_request(prompt_len, output_len, params_b=params_b, **cfg, **kw)
        rows.append({"rung": name, "total": c["total"],
                     "units": c["total"] / base["total"] * 100.0,
                     "prefill_share": c["prefill_share"], "cfg": cfg})
    return rows


def ladder_residuals(rows: list[dict], corpus: dict | None = None) -> list[dict]:
    """How far each rung is from the corpus's figure. REPORTED, not hidden.

    The model is a mechanism, not a fit: it gets the ORDER and the approximate shape right and
    lands within a few units on three of the four rungs. Reporting the residuals is the difference
    between "we reproduce the corpus" and "we built something that happens to look similar".
    """
    corpus = corpus or CORPUS_LADDER
    out = []
    for r in rows:
        target = corpus.get(r["rung"])
        if target is None:
            continue
        out.append({"rung": r["rung"], "model": r["units"], "corpus": target,
                    "delta": r["units"] - target})
    return out


def solve_batch(target_ratio: float, prompt_len: int, output_len: int, **kw) -> float:
    """What effective batch size reproduces this batching multiplier?

    Solving rather than asserting is the point: it turns "batching gets you to 42" into a statement
    about batch depth that an operator can compare against what their service actually runs.
    """
    base = cost_per_request(prompt_len, output_len, 1, **kw)
    target = base["total"] * target_ratio
    for b in range(1, 512):
        if cost_per_request(prompt_len, output_len, b, **kw)["total"] <= target:
            return float(b)
    return float("inf")


def solve_cache_hit(target_ratio: float, prompt_len: int, output_len: int, batch: int, **kw) -> float:
    """What prefix-cache hit rate reproduces this caching multiplier?

    The answer is workload-dependent in a way worth stating: the same cache hit rate buys a large
    saving on a prompt-heavy workload and almost nothing on an output-heavy one, because caching
    only removes the prefill term.
    """
    without = cost_per_request(prompt_len, output_len, batch, cache_hit=0.0, **kw)["total"]
    target = without * target_ratio
    for i in range(0, 101):
        h = i / 100.0
        if cost_per_request(prompt_len, output_len, batch, cache_hit=h, **kw)["total"] <= target:
            return h
    return 1.0


def cost_per_million_tokens(prompt_len: int, output_len: int, batch: int, gpu_hour_usd: float,
                            tokens_per_second: float, requests_per_hour: float, **kw) -> dict:
    """The headline FinOps number, and it depends on a term nobody tunes: the prompt/output ratio.

    Cost per million OUTPUT tokens is the metric teams quote, and it is dominated by the prompt
    length. Two services with identical per-token pricing can differ 10x in cost per useful token
    because one of them re-reads a 4000-token system prompt on every call.

    The bridge to T19: this is the same arithmetic, expressed in dollars.
    """
    tokens_moved = requests_per_hour * (prompt_len + output_len)
    hours = tokens_moved / tokens_per_second / 3600.0
    cost = hours * gpu_hour_usd
    cost_per_m_output = cost / (requests_per_hour * output_len / 1e6) if output_len else float("inf")
    return {"cost_per_hour_usd": cost, "tokens_moved": tokens_moved,
            "cost_per_million_output_usd": cost_per_m_output,
            "prompt_output_ratio": prompt_len / output_len if output_len else float("inf")}
