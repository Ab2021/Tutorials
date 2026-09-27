"""The speedup formula, the optimal draft length, and the batch size at which it all stops working.

    tokens_per_step = 1 + sum_{i=1..gamma} prod_{j=1..i} alpha_j
    speedup        ~= tokens_per_step / (1 + gamma * c)

Two numbers decide everything: `alpha` (how often the drafter is right) and `c` (what the drafter
costs as a fraction of a target forward pass). Neither is a property of the hardware.

THE THIRD NUMBER IS THE ONE PEOPLE MISS, and it is not in the formula at all: speculative decoding
pays because a VERIFY PASS ON gamma+1 TOKENS COSTS THE SAME AS A ONE-TOKEN DECODE. That is true
exactly while decode is MEMORY-BOUND -- the weights are the bottleneck, so extra tokens ride along
on bandwidth that was going to be spent anyway. Once the batch is large enough that decode becomes
COMPUTE-BOUND, the verify pass costs `gamma+1` times as much and speculation is a net LOSS.

So the honest statement of the method is: it converts spare tensor cores into tokens, and the
amount of spare compute is a function of the batch size. This module computes where that runs out.
"""
from __future__ import annotations

from .acceptance import decay_profile, tokens_per_step


# --------------------------------------------------------------------------------------
# The speedup formula
# --------------------------------------------------------------------------------------

def speedup(alphas: list[float], draft_cost: float) -> float:
    """Tokens per target pass, divided by the relative cost of producing them.

    A value below 1.0 means speculation made the system SLOWER, and it is not an edge case: it is
    the expected outcome in the compute-bound regime, and it happens with a badly-chosen drafter
    at any batch size.
    """
    gamma = len(alphas)
    if draft_cost < 0:
        raise ValueError("draft_cost must be >= 0")
    return tokens_per_step(alphas) / (1.0 + gamma * draft_cost)


def gamma_sweep(alpha1: float, draft_cost: float, decay: float = 0.90,
                max_gamma: int = 16) -> list[dict]:
    """Speedup as a function of draft length, for a fixed acceptance profile.

    THE NON-MONOTONICITY IS THE POINT. Because acceptance decays with position, the numerator
    saturates while the denominator keeps growing linearly in `gamma`. Speedup therefore rises,
    peaks, and falls -- so "use more draft tokens" is wrong past a point, and the peak moves with
    BOTH `alpha` and `c`.

    A drafter with `c = 0` (n-gram) never pays a denominator penalty for length, so its curve
    flattens rather than falling. That is the structural difference between a free drafter and a
    cheap one.
    """
    rows = []
    for g in range(1, max_gamma + 1):
        alphas = decay_profile(alpha1, g, decay)
        rows.append({"gamma": g, "tokens_per_step": tokens_per_step(alphas),
                     "speedup": speedup(alphas, draft_cost),
                     "alpha_last": alphas[-1] if alphas else 0.0})
    return rows


def optimal_gamma(alpha1: float, draft_cost: float, decay: float = 0.90,
                  max_gamma: int = 16) -> dict:
    """The `gamma` that maximises speedup, and the flatness of the curve around it.

    The flatness matters more than the argmax. If the curve is flat between 4 and 8, an operator
    has latitude and should pick the smaller value (less draft work, less wasted compute on
    rejection). If it is sharply peaked, the tuning is real and is worth instrumenting.
    """
    rows = gamma_sweep(alpha1, draft_cost, decay, max_gamma)
    best = max(rows, key=lambda r: r["speedup"])
    top = best["speedup"]
    flat = [r["gamma"] for r in rows if r["speedup"] >= 0.98 * top]
    return {"best_gamma": best["gamma"], "best_speedup": top,
            "within_2pct": flat, "curve": rows}


def marginal_gamma_gain(alpha1: float, draft_cost: float, decay: float = 0.90,
                        max_gamma: int = 16) -> list[dict]:
    """The gain from one more draft token, so the stopping rule is explicit.

    `gamma` is raised until the marginal gain stops compensating for the marginal cost. Reporting
    the marginal series rather than only the optimum makes the stopping rule something an operator
    can apply to their own numbers.
    """
    rows = gamma_sweep(alpha1, draft_cost, decay, max_gamma)
    out = []
    for prev, cur in zip(rows, rows[1:]):
        out.append({"gamma": cur["gamma"],
                    "delta_speedup": cur["speedup"] - prev["speedup"],
                    "tokens_per_step": cur["tokens_per_step"]})
    return out


# --------------------------------------------------------------------------------------
# The regime crossover -- where speculation stops paying
# --------------------------------------------------------------------------------------

def arithmetic_intensity(batch: int, params_b: float, weight_bytes: int = 2,
                         kv_bytes_per_token: float = 0.33e6) -> float:
    """FLOPs per byte moved, for one decode step at this batch size. The roofline x-axis (T06).

        bytes moved  = W + batch * KV_per_token      (weights + the KV read for the whole batch)
        FLOPs        = 2 * N * batch

    WEIGHTS ARE A CONSTANT AND KV GROWS WITH BATCH. That asymmetry is the entire reason batch size
    changes the regime: at small batch the weights dominate and adding sequences is almost free;
    at large batch the KV traffic dominates and each additional sequence costs real bandwidth.
    """
    if batch <= 0:
        raise ValueError("batch must be > 0")
    n_params = params_b * 1e9
    w_bytes = n_params * weight_bytes
    moved = w_bytes + batch * kv_bytes_per_token
    flops = 2.0 * n_params * batch
    return flops / moved


def memory_bound_batch_limit(params_b: float, machine_balance_flops_per_byte: float,
                             weight_bytes: int = 2,
                             kv_bytes_per_token: float = 0.33e6) -> float:
    """The batch size at which decode stops being memory-bound. THE speculation crossover.

    Solve `arithmetic_intensity(batch) = machine_balance` for batch:

        batch* = (B * W) / (2N - B * kv_per_token)

    Below `batch*`, a `gamma+1`-token verify pass costs ONE decode step's worth of time, and
    speculative decoding converts idle tensor cores into extra tokens. Above it, verify costs
    `gamma+1` steps' worth and speculation is a pure loss.

    WORKED, for the figures used in this blueprint [D] -- a 70B at fp16 with grouped-query KV on
    an H100-class part:

        B  = 989 TFLOPS fp16 / 3.35 TB/s  ~ 295 flops/byte
        W  = 70e9 * 2                      = 140e9 bytes
        kv = 0.33e6 bytes/token            (80 layers x 8 kv heads x 128 dim x 2 x 2 bytes)

        batch* = (295 * 140e9) / (140e9 - 295 * 0.33e6)  ~ 295 sequences

    So the crossover sits at a few hundred concurrent sequences -- BELOW the concurrency a
    throughput-oriented deployment runs at, and above the concurrency a latency-oriented one runs
    at. That single sentence explains most of the disagreement about whether speculative decoding
    "works": the two deployments are on opposite sides of `batch*`.
    """
    n_params = params_b * 1e9
    w_bytes = n_params * weight_bytes
    denom = 2.0 * n_params - machine_balance_flops_per_byte * kv_bytes_per_token
    if denom <= 0:
        raise ValueError("this machine cannot be compute-bound at any batch size "
                         "(balance x kv_per_token exceeds 2N) -- speculation always pays")
    return (machine_balance_flops_per_byte * w_bytes) / denom


def step_cost(batch: int, gamma: int, draft_cost: float, batch_limit: float) -> float:
    """Relative cost of one speculative step at this batch size.

    TWO REGIMES, and the model is deliberately a hard switch at `batch_limit` rather than a smooth
    curve, because the qualitative behaviour is what matters:

      memory-bound  (batch <= batch_limit):  the verify pass is FREE-ish -> cost ~ 1 + gamma*c
      compute-bound (batch >  batch_limit):  verify does gamma+1 tokens' work -> gamma+1 + gamma*c
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    draft = gamma * draft_cost
    if batch <= batch_limit:
        return 1.0 + draft
    return (gamma + 1.0) + draft


def speedup_vs_batch(alphas: list[float], draft_cost: float,
                     batch_limit: float, batches: list[int]) -> list[dict]:
    """Speedup across batch sizes, showing the crossover and the loss beyond it.

    This is the curve that should be on the dashboard before speculative decoding is enabled. A
    single "we got 2x" number is a statement about one operating point and says nothing about the
    point the service runs at after the next traffic spike.
    """
    gamma = len(alphas)
    tps = tokens_per_step(alphas)
    rows = []
    for b in batches:
        c = step_cost(b, gamma, draft_cost, batch_limit)
        rows.append({"batch": b, "regime": "memory-bound" if b <= batch_limit else "compute-bound",
                     "speedup": tps / c, "cost": c})
    return rows


def deployment_verdict(alphas: list[float], draft_cost: float, batch_limit: float,
                       batch_p50: int, batch_p99: int) -> dict:
    """Is speculation a win at BOTH the median and the tail of this deployment's batch sizes?

    THE TAIL DECIDES. A deployment that is memory-bound at p50 and compute-bound at p99 will show
    a speedup in every average and a latency REGRESSION under load -- and the regression appears
    exactly when the system is least able to absorb it. Speculation must be judged at the batch
    size the service actually reaches, not the one it runs at on a quiet afternoon.
    """
    gamma = len(alphas)
    tps = tokens_per_step(alphas)
    s50 = tps / step_cost(batch_p50, gamma, draft_cost, batch_limit)
    s99 = tps / step_cost(batch_p99, gamma, draft_cost, batch_limit)
    return {"speedup_at_p50": s50, "speedup_at_p99": s99,
            "p50_regime": "memory-bound" if batch_p50 <= batch_limit else "compute-bound",
            "p99_regime": "memory-bound" if batch_p99 <= batch_limit else "compute-bound",
            "safe": s50 > 1.0 and s99 > 1.0,
            "note": ("safe" if (s50 > 1.0 and s99 > 1.0) else
                     "NOT SAFE -- speculation loses at one of these operating points. "
                     "Gate it on batch size, or reduce gamma until both are > 1.")}
