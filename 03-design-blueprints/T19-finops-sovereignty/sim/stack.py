"""T19 -- the cost stack: eight layers, and the arithmetic of the one that dominates.

The supporting guide's decomposition [R] is the map, and its anchor statistic is the one to hold on
to: **system prompts are ~69% of input tokens and only ~28% of calls use prompt caching.** That is
the largest and cheapest lever in most stacks, sitting unused.

This module computes what a discount is worth, and the answer is never the discount. A cache read
"roughly 10 times cheaper than fresh tokens" [T] is a statement about the PRICE of one line of the
bill. What a programme can bank is `line_reduction x share_of_bill`, and the two are different
numbers by a factor of three.

Three things here are easy to get wrong and are computed rather than asserted:

  * the discount curve, and the fact that it has an ASYMPTOTE -- past ~50x the uncached floor
    dominates and further discount buys nothing;
  * the write premium, which means a cached prefix is not free even when it is never read, and
    which has a break-even number of READS that most caching rollouts never compute;
  * reasoning tokens, which are billed at the output rate and do not appear in the response.

Stdlib only. Every number reproducible from `python run.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------------------------
# The bill. Shares are of the TOTAL INFERENCE BILL, not of tokens, and they must sum to 1.00.
# The illustrative split is taken from the case study's own decomposition (section 8, step 1) and
# the system-prompt line is pinned to the corpus statistic: 69% of input tokens, input being 45%
# of the bill -> 0.69 x 0.45 = 0.3105.
# --------------------------------------------------------------------------------------------
PREFIX_SHARE_OF_INPUT = 0.69      # [R] Datadog 2026, via the guide
INPUT_SHARE_OF_BILL = 0.45        # [D] illustrative, from the case study's step 1


@dataclass(frozen=True)
class Layer:
    name: str
    share: float          # share of the total bill
    driver: str
    lever: str


BILL = (
    Layer("system_prompt",       0.3105, "fixed scaffolding, tool defs, few-shot",
          "prompt / prefix caching"),
    Layer("retrieved_context",   0.0795, "RAG chunks, injected documents",
          "context discipline, RAG vs long context"),
    Layer("conversation_memory", 0.0700, "chat history, agent scratchpad",
          "windowing, compaction"),
    Layer("model_tier",          0.2000, "frontier vs mid vs small / self-hosted",
          "right-sizing, routing, distillation"),
    Layer("output_length",       0.1600, "verbosity, format, max_tokens",
          "caps, terse output contracts"),
    Layer("reasoning_tokens",    0.1500, "extended thinking",
          "gate thinking by task complexity"),
    Layer("retry_overhead",      0.0300, "transient errors, guardrail re-runs",
          "bounded retries, circuit breakers"),
)

BILL_TOTAL = sum(l.share for l in BILL)
LAYER_BY_NAME = {l.name: l for l in BILL}

# --------------------------------------------------------------------------------------------
# Caching. `discount` is "how many times cheaper a cache read is than a fresh token": the corpus
# says "roughly 10 times cheaper" [T]; providers advertise ~50% (OpenAI), ~90% (Anthropic) and
# ~75% (Google) [R]. Those are DIFFERENT UNITS -- a 90% discount is a 10x discount -- and
# conflating them is the second most common arithmetic error in this topic.
# --------------------------------------------------------------------------------------------
CACHED_TODAY = 0.28               # [R] Datadog 2026: share of calls using prompt caching
CACHED_TARGET = 0.90              # [D] the target the case study sets


def percent_to_multiple(pct: float) -> float:
    """A '90% cheaper' discount is a 10x multiple, not a 90x one."""
    if not 0.0 <= pct < 1.0:
        raise ValueError("a discount percentage must be in [0, 1)")
    return 1.0 / (1.0 - pct)


def cache_read_price(discount: float) -> float:
    """Price of one cached-read token, as a fraction of a fresh token."""
    return 1.0 / discount


def prefix_line(cached_share: float, discount: float) -> float:
    """Cost of the prefix line per call, in units where an uncached call costs 1.00."""
    return cached_share * cache_read_price(discount) + (1.0 - cached_share)


def caching_saving(discount: float,
                   cached_today: float = CACHED_TODAY,
                   cached_target: float = CACHED_TARGET,
                   prefix_share: float = BILL[0].share) -> dict:
    """Both numbers, because they differ by a factor of three and only one is bankable."""
    now = prefix_line(cached_today, discount)
    fixed = prefix_line(cached_target, discount)
    line_reduction = (now - fixed) / now
    return {
        "discount": discount,
        "prefix_line_now": now,
        "prefix_line_fixed": fixed,
        "line_reduction": line_reduction,
        "prefix_share_of_bill": prefix_share,
        "blended_saving": line_reduction * prefix_share,
    }


def discount_sweep(discounts=(2.0, 5.0, 10.0, 20.0, 50.0, 100.0)) -> list:
    return [caching_saving(d) for d in discounts]


def asymptotic_reduction(cached_today: float = CACHED_TODAY,
                         cached_target: float = CACHED_TARGET) -> dict:
    """The ceiling of the caching lever: a cache read costs nothing at all.

    This is the bound nobody quotes, and it is the reason the discount curve flattens. At an
    infinite discount the cached share becomes free, but the UNCACHED share does not, and that
    floor is what sets the asymptote.
    """
    now = prefix_line(cached_today, float("inf"))    # every cached call free
    fixed = prefix_line(cached_target, float("inf"))  # the uncached 10% remains
    line_reduction = (now - fixed) / now
    return {
        "uncached_floor_now": 1.0 - cached_today,
        "uncached_floor_target": 1.0 - cached_target,
        "line_reduction": line_reduction,
        "blended_saving": line_reduction * BILL[0].share,
    }


def write_premium_break_even(discount: float, write_price: float) -> dict:
    """How many READS a cached prefix needs before the write premium pays for itself.

        no cache:   n
        cached:     write_price + n / discount
        break-even: n > write_price / (1 - 1/discount)

    `write_price` is the cost of the cache WRITE as a multiple of a fresh token (Anthropic's is
    ~1.25x, i.e. a 25% premium [R]). A rollout that turns caching on and never reads a prefix
    twice has made the bill worse, and this is the number that says so.
    """
    if discount <= 1.0:
        return {"discount": discount, "write_price": write_price, "n_reads": float("inf")}
    n = write_price / (1.0 - 1.0 / discount)
    return {"discount": discount, "write_price": write_price,
            "n_reads": n, "n_reads_ceil": math.ceil(n)}


def caching_npv(n_reads: int, discount: float, write_price: float) -> dict:
    """The actual comparison, for an integer number of reads."""
    uncached = float(n_reads)
    cached = write_price + n_reads * cache_read_price(discount)
    return {"n_reads": n_reads, "uncached": uncached, "cached": cached,
            "saving": uncached - cached,
            "worth_it": cached < uncached}


# --------------------------------------------------------------------------------------------
# Reasoning tokens: billed at the OUTPUT rate and absent from the response.
# --------------------------------------------------------------------------------------------
def reasoning_amplification(visible_output_tokens: int, reasoning_tokens: int) -> dict:
    """How much more a call costs than its visible output suggests.

    The corpus reports multipliers "anywhere from ~3x to ~15x depending on the task" [R]. The
    mechanism is that a 'short' answer from a thinking model is billed for the tokens you cannot
    see, so tokens-per-visible-output -- the metric the Token Raj speaker says teams must stop
    using [T] -- is not merely uninformative, it is systematically optimistic.
    """
    total = visible_output_tokens + reasoning_tokens
    return {
        "visible": visible_output_tokens,
        "reasoning": reasoning_tokens,
        "total_billed": total,
        "amplification": total / visible_output_tokens if visible_output_tokens else float("inf"),
    }


def reasoning_gating_saving(reasoning_share_of_bill: float, gated_fraction: float,
                            reduction_on_gated: float) -> float:
    """Gating thinking by task complexity, as a share of the total bill."""
    return reasoning_share_of_bill * gated_fraction * reduction_on_gated


def layer_table() -> list:
    return [{"name": l.name, "share": l.share, "driver": l.driver, "lever": l.lever}
            for l in BILL]
