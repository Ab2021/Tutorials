"""The acceptance rule -- and the proof that speculative decoding is EXACT.

Speculative decoding is not an approximation. The output is distributed exactly as the target
model, and that property is not a happy accident of the method: it is bought by one specific
construction, the RESIDUAL resampling on rejection. Remove it and the sampler is subtly biased in
a way that looks completely normal -- fluent text, plausible latency, and a distribution that has
quietly drifted from the model you think you are serving.

This module implements the rule and then MEASURES the drift, so the claim is checked rather than
asserted.

    accept a draft token x with probability   min(1, p_target(x) / p_draft(x))
    on rejection, resample from                normalise(max(0, p_target - p_draft))
"""
from __future__ import annotations

import random

# --------------------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------------------

def normalise(v: list[float]) -> list[float]:
    s = sum(v)
    if s <= 0:
        raise ValueError("cannot normalise a distribution with zero mass")
    return [x / s for x in v]


def residual(p_target: list[float], p_draft: list[float]) -> list[float]:
    """`normalise(max(0, p_target - p_draft))` -- the correction that makes the method exact.

    Element-wise positive part, then normalised. Wherever the draft over-weighted a token
    relative to the target, the residual is zero; wherever the target puts more mass than the
    draft, that excess is what the rejection branch must supply.

    THE NORMALISER IS THE REJECTION PROBABILITY, and that identity is the whole proof:
        P(reject) = 1 - sum_y min(p_draft(y), p_target(y))
                  = sum_y max(0, p_target(y) - p_draft(y))
    so the residual's denominator is exactly the mass that will be resampled.
    """
    if len(p_target) != len(p_draft):
        raise ValueError("distributions must share a support")
    return normalise([max(0.0, t - d) for t, d in zip(p_target, p_draft)])


def rejection_mass(p_target: list[float], p_draft: list[float]) -> float:
    """Probability that one draft token is rejected."""
    return sum(max(0.0, t - d) for t, d in zip(p_target, p_draft))


def acceptance_rate(p_target: list[float], p_draft: list[float]) -> float:
    """Probability that one draft token is accepted: `sum_y min(p_target(y), p_draft(y))`.

    THE metric to monitor in production. It is a property of the draft/target PAIR, not of the
    batch shape or the load, which is why it is a better health signal than observed speedup --
    speedup is downstream of acceptance AND of the batch regime (see `regimes.py`).
    """
    return sum(min(t, d) for t, d in zip(p_target, p_draft))


def total_variation(p: list[float], q: list[float]) -> float:
    """Half the L1 distance. 0 = identical distributions."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


# --------------------------------------------------------------------------------------
# One speculative step
# --------------------------------------------------------------------------------------

def speculative_step(p_target: list[float], p_draft: list[float],
                     rng: random.Random) -> tuple[int, bool]:
    """Draw a draft token, accept or reject, and correct on rejection. Returns (token, accepted).

    Both branches are the same three lines of arithmetic; the difference between a correct and a
    broken implementation is entirely in WHICH distribution the rejection branch samples from.
    """
    x = _sample(p_draft, rng)
    if p_draft[x] > 0 and rng.random() < min(1.0, p_target[x] / p_draft[x]):
        return x, True
    return _sample(residual(p_target, p_draft), rng), False


def naive_step(p_target: list[float], p_draft: list[float],
               rng: random.Random) -> tuple[int, bool]:
    """THE BROKEN VARIANT, kept so the drift can be measured rather than described.

    On rejection it resamples from the TARGET -- which is the intuitive thing to do ("the draft
    was wrong, ask the real model"). It is wrong, and biased in a specific direction: it
    double-counts the mass where the target exceeds the draft, because the accepted branch
    already contributed `min(p_draft, p_target)` there.

        P(output = x) = min(p_d, p_t)(x) + R * p_t(x)      -- too much mass on high-p_t tokens

    The bias is largest exactly where the draft is worst, i.e. on the tokens the draft declined
    to propose. Nothing about the output looks wrong: it is fluent, it is on-topic, and it is
    not the model you deployed.
    """
    x = _sample(p_draft, rng)
    if p_draft[x] > 0 and rng.random() < min(1.0, p_target[x] / p_draft[x]):
        return x, True
    return _sample(p_target, rng), False


def _sample(p: list[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(p):
        acc += w
        if r < acc:
            return i
    return len(p) - 1


# --------------------------------------------------------------------------------------
# The exactness experiment
# --------------------------------------------------------------------------------------

def empirical_distribution(p_target: list[float], p_draft: list[float], n: int,
                           rng: random.Random, broken: bool = False) -> list[float]:
    """Run `n` speculative steps and return the empirical output distribution.

    This is the check that matters: not "does the sampler look right" but "does it converge to
    the target". `n` large and a fixed seed make the comparison deterministic.
    """
    counts = [0] * len(p_target)
    step = naive_step if broken else speculative_step
    for _ in range(n):
        tok, _acc = step(p_target, p_draft, rng)
        counts[tok] += 1
    return [c / n for c in counts]


def exactness_report(p_target: list[float], p_draft: list[float], n: int = 200_000,
                     seed: int = 7) -> dict:
    """The two-sided check: an exact sampler and a broken one, measured against the target."""
    ok = empirical_distribution(p_target, p_draft, n, random.Random(seed))
    bad = empirical_distribution(p_target, p_draft, n, random.Random(seed), broken=True)
    return {
        "n": n,
        "acceptance_rate": acceptance_rate(p_target, p_draft),
        "tv_exact": total_variation(ok, p_target),
        "tv_broken": total_variation(bad, p_target),
        "max_abs_err_exact": max(abs(a - b) for a, b in zip(ok, p_target)),
        "max_abs_err_broken": max(abs(a - b) for a, b in zip(bad, p_target)),
        "exact": ok,
        "broken": bad,
    }


# --------------------------------------------------------------------------------------
# Acceptance decay -- why long drafts stop paying
# --------------------------------------------------------------------------------------

def decay_profile(alpha1: float, gamma: int, decay: float = 0.90) -> list[float]:
    """Per-position acceptance `alpha_1 .. alpha_gamma`.

    Acceptance DECAYS with position, and this is the fact that kills naive scaling of the draft
    length. Each draft token is conditioned on the previous drafted tokens, which were themselves
    only guesses -- errors compound, so `alpha_j` falls roughly geometrically.

    Treating `alpha` as a single constant (the common simplification) makes `tokens_per_step`
    rise almost linearly in `gamma` and predicts that longer drafts are always better. They are
    not.
    """
    if not 0.0 < alpha1 <= 1.0:
        raise ValueError("alpha1 must be in (0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    return [alpha1 * (decay ** (j - 1)) for j in range(1, gamma + 1)]


def tokens_per_step(alphas: list[float]) -> float:
    """Expected tokens produced per target forward pass.

    `1 + sum_{i=1..gamma} prod_{j=1..i} alpha_j`

    THE 1 IS NOT THE DRAFT. It is the token the target produces anyway on its own -- a rejected
    first draft token always yields exactly one token from the verify pass. So the floor of this
    function is 1.0 (speculation can never produce less than plain decoding in tokens), and the
    reason a bad draft costs you is entirely in the denominator's draft term, never here.
    """
    total = 1.0
    run = 1.0
    for a in alphas:
        run *= a
        total += run
    return total
