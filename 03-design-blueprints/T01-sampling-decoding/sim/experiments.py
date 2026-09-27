"""The experiments run.py executes.

Every function returns plain data (lists of rows). run.py owns the formatting.
All randomness is seeded, so the output is reproducible run to run -- which is
itself one of the points the simulation makes.
"""

import math
from random import Random

from .distribution import (
    argmax,
    entropy,
    head_mass_distribution,
    jitter,
    margin_distribution,
    peaked_distribution,
    perplexity,
    softmax,
)
from . import samplers

VOCAB = 1000  # a toy vocabulary; the real one in the corpus is 128k


def exp_temperature_limits(seed=7):
    """T = 1 is the identity; T -> 0 is one-hot; T -> inf is uniform."""
    base = peaked_distribution(VOCAB, top_prob=0.4)
    rows = []
    for t in (0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 50.0):
        p = samplers.apply_temperature(base, t)
        rows.append(
            {
                "temperature": t,
                "max_prob": max(p),
                "entropy_nats": entropy(p),
                "argmax": argmax(p),
                "tied_at_max": sum(1 for x in p if abs(x - max(p)) < 1e-12),
            }
        )
    return rows


def exp_topk_drift(seed=7):
    """A fixed k is a different constraint at every step.

    The corpus reports 68% of the mass in the top 6 tokens after "the" and 99%
    after "the car" [T]. This builds two synthetic heads in that regime and
    reports what the model actually realises, so the two can be compared
    without pretending the simulation measured the lecture.
    """
    rows = []
    for label, head_mass in (("flat head (after 'the')", 0.68),
                             ("peaked head (after 'the car')", 0.99)):
        probs = head_mass_distribution(VOCAB, head_mass, head_size=6)
        top6 = sorted(probs, reverse=True)[:6]
        rows.append(
            {
                "step": label,
                "requested_head_mass": head_mass,
                "realised_top6_mass": sum(top6),
                "mass_beyond_top50": 1.0 - sum(sorted(probs, reverse=True)[:50]),
            }
        )
    return rows


def exp_truncation_compare(seed=7):
    """What each truncation rule keeps, on a peaked and on a flat step."""
    shapes = {
        "peaked": peaked_distribution(VOCAB, top_prob=0.55),
        "long-tail": head_mass_distribution(VOCAB, 0.30, head_size=3),
    }
    rules = [
        ("top_k(50)", lambda p: samplers.top_k(p, 50)),
        ("top_p(0.95)", lambda p: samplers.top_p(p, 0.95)),
        ("top_p(0.90)", lambda p: samplers.top_p(p, 0.90)),
        ("epsilon(1e-4)", lambda p: samplers.epsilon(p, 1e-4)),
        ("locally_typical(0.9)", lambda p: samplers.locally_typical(p, 0.9)),
    ]
    rows = []
    for shape_name, probs in shapes.items():
        for rule_name, fn in rules:
            cands, mass = fn(probs)
            rows.append(
                {
                    "shape": shape_name,
                    "rule": rule_name,
                    "survivors": len(cands),
                    "retained_mass": mass,
                    "max_kept_prob": max((p for _, p in cands), default=0.0),
                }
            )
    return rows


def exp_processor_order(seed=7):
    """Mask before truncate, or truncate before mask?

    The corpus's JSON-as-FSA observation is that at any single decoding step
    there may be ~10 valid continuations out of 100k [T]. This models that:
    the grammar permits 10 indices, all of them in the tail of the
    distribution. Truncating first is what produces the empty candidate set.
    """
    probs = head_mass_distribution(VOCAB, head_mass=0.999, head_size=3)
    allowed = list(range(VOCAB - 10, VOCAB))  # grammar-valid, all in the tail
    rows = []

    # Order A: mask first, then truncate what survives.
    masked, _ = samplers.apply_mask(probs, allowed)
    a_cands, a_mass = samplers.top_p(_expand(masked), 0.95)
    rows.append(
        {
            "order": "mask -> top_p(0.95)",
            "survivors": len(a_cands),
            "retained_mass": a_mass,
            "outcome": "drawable" if a_cands else "EMPTY CANDIDATE SET",
        }
    )

    # Order B: truncate first, then discover the grammar kept none of it.
    truncated, _ = samplers.top_p(probs, 0.95)
    b_cands, b_mass = samplers.apply_mask(_expand(truncated), allowed)
    rows.append(
        {
            "order": "top_p(0.95) -> mask",
            "survivors": len(b_cands),
            "retained_mass": b_mass,
            "outcome": "drawable" if b_cands else "EMPTY CANDIDATE SET",
        }
    )
    return rows


def _expand(candidates):
    """Turn a candidate set back into a probability vector over the vocab."""
    out = [0.0] * VOCAB
    for idx, p in candidates:
        out[idx] = p
    return out


def exp_diversity_vs_temperature(seed=11, samples=200, tokens=100):
    """Diversity rises with temperature, and so does surprise.

    distinct-1 is the ratio of unique tokens within a generation -- the
    deterministic diversity metric the corpus names [T]. Greedy collapses it.
    """
    probs = head_mass_distribution(VOCAB, head_mass=0.20, head_size=4)
    rng = Random(seed)
    rows = []
    for t in (0.0, 0.3, 0.7, 1.0, 1.2):
        shifted = samplers.apply_temperature(probs, t)
        cands, _ = samplers.top_p(shifted, 0.95)
        seq = [samplers.greedy(cands) if t == 0.0 else samplers.draw(cands, rng)
               for _ in range(tokens)]
        distinct1 = len(set(seq)) / len(seq)
        mean_ppl = sum(perplexity(shifted, i) for i in seq) / len(seq)
        rows.append(
            {
                "temperature": t,
                "distinct_1": distinct1,
                "mean_token_perplexity": mean_ppl,
                "survivors": len(cands),
            }
        )
    return rows


def _reduction_stats(probs, seed, repeats, magnitude):
    truth = argmax(probs)
    rng = Random(seed)
    results = [argmax(jitter(probs, magnitude, rng)) for _ in range(repeats)]
    agreements = sum(1 for r in results if r == truth)
    longest, current = 0, 0
    for r in results:
        if r == truth:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "repeats": repeats,
        "agreement_rate": agreements / repeats,
        "distinct_answers": len(set(results)),
        "longest_agreeing_streak": longest,
        "flips": repeats - agreements,
    }


def exp_reduction_order(seed=13, repeats=200, magnitude=1e-7):
    """Why temperature 0 is not a reproducibility contract.

    Each repeat re-runs the *same* argmax on a distribution perturbed by a
    reduction-order-sized jitter. Two step shapes are compared: one where the
    top two tokens are nearly tied, and one where the leader is clear. The
    interesting result is that the flip rate depends entirely on the margin --
    which is why the failure is intermittent and hard to reproduce on demand.
    The corpus describes the symptom as "you'll get it the same like five
    times, but then you'll get a different one" [T].
    """
    rows = []
    for label, margin in (("margin < perturbation", 1e-9),
                          ("margin >> perturbation", 0.05)):
        probs = margin_distribution(VOCAB, top1=0.30, margin=margin)
        stats = _reduction_stats(probs, seed, repeats, magnitude)
        stats["step"] = label
        stats["top1_minus_top2"] = margin
        rows.append(stats)
    return rows


def exp_mirostat(seed=17, steps=800, target_perplexity=12.0):
    """Target-perplexity control versus a fixed truncation threshold."""
    probs = head_mass_distribution(VOCAB, head_mass=0.35, head_size=4)
    rng = Random(seed)

    mio = samplers.Mirostat(target_perplexity=target_perplexity)
    observed = []
    for _ in range(steps):
        _, surprise = mio.step(probs, rng)
        observed.append(math.exp(surprise))

    fixed = samplers.top_p(probs, 0.95)[0]
    fixed_obs = []
    for _ in range(steps):
        idx = samplers.draw(fixed, rng)
        fixed_obs.append(perplexity(probs, idx))

    return {
        "target_perplexity": target_perplexity,
        "mirostat_mean_perplexity": sum(observed) / len(observed),
        "fixed_topp_mean_perplexity": sum(fixed_obs) / len(fixed_obs),
        "final_mu": mio.mu,
    }


def exp_tie_behaviour(seed=19):
    """Exactly tied logits give a uniform draw over the tied set, not argmax."""
    logits = [2.0, 2.0, 2.0, -1.0, -2.0]
    probs = softmax(logits)
    frozen = samplers.apply_temperature(probs, 0.0)
    rng = Random(seed)
    cands, _ = samplers.top_k(frozen, 5)
    draws = [samplers.draw(cands, rng) for _ in range(300)]
    from collections import Counter

    counts = Counter(draws)
    return {
        "tied_indices": [0, 1, 2],
        "frozen_max_prob": max(frozen),
        "share_of_draws_on_each_tied_token": {
            k: counts.get(k, 0) / 300 for k in (0, 1, 2)
        },
    }
