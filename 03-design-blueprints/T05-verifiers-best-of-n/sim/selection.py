"""The selection rule, the KL bound, and rejection sampling.

Two things this module is careful about:

1. `logprob` never enters the selection rule (see candidates.Candidate). Selecting on the
   generator's own score degenerates to greedy decoding with extra steps.
2. `kl_bound` is a CEILING, never a target. The corpus states it as the divergence created
   between the best-of-N output distribution and the TARGET (preference) distribution --
   not the base policy -- and notes explicitly that it is not tight [T] CMU lecture 12.
   The most common misuse is treating it as a quantity to maximise.
"""
from __future__ import annotations

import math
import random


def select(scores: list[dict], candidates: list, tie_break: str = "shorter"
           ) -> tuple[str, str | None, float]:
    """argmax over non-eliminated candidates. Returns (chosen, runner_up, margin).

    `tie_break="shorter"` is the default for a NON-ARBITRARY reason: judges prefer longer
    answers, so ties are disproportionately long. An unspecified tie-break inherits the
    verbosity bias.
    """
    by_id = {c.id: c for c in candidates}
    live = [s for s in scores if not s.get("eliminated")]
    if not live:
        return None, None, 0.0

    def key(s: dict):
        c = by_id[s["candidate_id"]]
        if tie_break == "shorter":
            return (s["score"], -c.length)
        if tie_break == "longer":
            return (s["score"], c.length)
        if tie_break == "first":
            return (s["score"], -int(s["candidate_id"].strip("c")))
        return (s["score"], 0.0)

    ranked = sorted(live, key=key, reverse=True)
    chosen = ranked[0]["candidate_id"]
    runner_up = ranked[1]["candidate_id"] if len(ranked) > 1 else None
    margin = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 0.0
    return chosen, runner_up, margin


def kl_bound(n: int) -> float:
    """log n - (n - 1)/n, transcribed from the corpus [T] CMU lecture 12.

    Both sides rise with n: more samples buy a more aggressive selection, and the bound on
    the divergence you have created rises with it. There is no monotonicity paradox.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return math.log(n) - (n - 1) / n


def acceptance_probability(p_x: float, q_x: float, C: float) -> float:
    """D(x) / (C * P(x)) in the classic rejection-sampling construction.

    Higher C means a looser envelope, lower acceptance, and more wasted samples. C must
    upper-bound the ratio D/P over the support or the sampler is not exact.
    """
    if q_x <= 0.0:
        return 0.0
    if C <= 0.0:
        raise ValueError("C must be positive")
    return min(1.0, p_x / (C * q_x))


def rejection_sample(pool: list, weights: dict, C: float, rng: random.Random):
    """Draw a candidate, accept with probability proportional to its weight.

    Returns a candidate id or None if the draw was rejected. The caller loops. `C` is
    supplied rather than inferred so the envelope is an explicit design choice.
    """
    c = rng.choice(pool)
    q = 1.0 / len(pool)                       # uniform proposal over the pool
    p = weights.get(c.id, 0.0)
    if rng.random() < acceptance_probability(p, q, C):
        return c.id
    return None


# --------------------------------------------------------------------------------------
# The KL experiment's machinery. Kept here because it is about the selection rule.
# --------------------------------------------------------------------------------------

def simulate_best_of_n(q: list[float], reward: list[float], n: int,
                       rng: random.Random, trials: int = 4000) -> list[float]:
    """Empirical distribution of the SELECTED outcome under best-of-n.

    `q` is the model's base distribution over K outcomes; `reward` is the target
    preference's per-outcome value. Draw n from q, keep the one with the highest reward
    draw. Returns the empirical selection frequencies.
    """
    K = len(q)
    counts = [0] * K
    cdf = []
    acc = 0.0
    for p in q:
        acc += p
        cdf.append(acc)

    def draw_index() -> int:
        u = rng.random()
        for i, c in enumerate(cdf):
            if u <= c:
                return i
        return K - 1

    for _ in range(trials):
        best_i, best_r = None, -1e18
        for _ in range(n):
            i = draw_index()
            r = reward[i] + rng.gauss(0.0, 0.05)   # noisy preference draw
            if r > best_r:
                best_i, best_r = i, r
        counts[best_i] += 1

    return [c / trials for c in counts]


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-4) -> float:
    """KL(p || q) with Laplace smoothing, so a zero in p does not produce infinity.

    Smoothing matters here: the empirical selection distribution can assign zero mass to
    an outcome the target prefers, and an unsmoothed KL would be infinite and useless for
    the comparison the experiment is making.
    """
    K = len(p)
    ps = [(x + eps) / (1.0 + eps * K) for x in p]
    qs = [(x + eps) / (1.0 + eps * K) for x in q]
    return sum(pi * math.log(pi / qi) for pi, qi in zip(ps, qs) if pi > 0.0)
