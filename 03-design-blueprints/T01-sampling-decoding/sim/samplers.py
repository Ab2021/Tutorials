"""The logit-processor chain: temperature, truncation, masking, and the draw.

Each function takes a probability vector and returns a *candidate set*: the
list of (index, probability) the sampler is allowed to draw from. Keeping the
set explicit (rather than mutating a copy of the vector) is what lets run.py
report survivors and retained mass per step, which is the diagnostic the
design's incident runbook depends on.
"""

import math
from random import Random

from .distribution import normalize


def apply_temperature(probs, temperature):
    """p_i^(1/T) renormalised.

    T = 1 is the identity (the only unbiased setting). T -> 0 is one-hot.
    T -> inf is uniform. Implemented in log space so large 1/T does not
    underflow silently.
    """
    if temperature <= 0.0:
        # The limit: all mass on the argmax. Exact ties are resolved by the
        # caller -- in the real system they yield a uniform draw over the
        # tied set, which greedy implementations routinely get wrong.
        best = max(probs)
        tied = [i for i, p in enumerate(probs) if p == best]
        out = [0.0] * len(probs)
        for i in tied:
            out[i] = 1.0 / len(tied)
        return out
    inv = 1.0 / temperature
    logs = []
    for p in probs:
        logs.append(math.log(p) * inv if p > 0.0 else float("-inf"))
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    return normalize(exps)


def top_k(probs, k):
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:k]
    return _as_set(probs, order)


def top_p(probs, threshold):
    """Nucleus: keep the shortest prefix (by descending probability) whose

    cumulative mass reaches `threshold`, renormalised.
    """
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    kept, acc = [], 0.0
    for i in order:
        kept.append(i)
        acc += probs[i]
        if acc >= threshold:
            break
    return _as_set(probs, kept)


def epsilon(probs, eps):
    """Drop every token below an absolute probability floor."""
    kept = [i for i, p in enumerate(probs) if p >= eps]
    if not kept:
        kept = [max(range(len(probs)), key=lambda i: probs[i])]
    return _as_set(probs, kept)


def locally_typical(probs, mass=0.9):
    """Sort by closeness to the distribution's entropy, then truncate.

    This is the rule that can cut a high-probability token: a token is kept for
    being *typical*, not for being likely. It therefore produces higher
    perplexity output than ancestral sampling -- by design.
    """
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)

    def score(i):
        p = probs[i]
        return abs(-math.log(p) - h) if p > 0.0 else float("inf")

    order = sorted(range(len(probs)), key=lambda i: (score(i), i))
    kept, acc = [], 0.0
    for i in order:
        kept.append(i)
        acc += probs[i]
        if acc >= mass:
            break
    return _as_set(probs, kept)


def apply_mask(probs, allowed):
    """Zero out every index the grammar forbids, renormalise the survivors."""
    allowed_set = set(allowed)
    kept = [i for i in range(len(probs)) if i in allowed_set and probs[i] > 0.0]
    if not kept:
        return [], 0.0
    return _as_set(probs, kept)


def _as_set(probs, indices):
    """Return ([(index, prob)], retained_mass) for a kept index list."""
    if not indices:
        return [], 0.0
    mass = sum(probs[i] for i in indices)
    if mass <= 0.0:
        return [], 0.0
    return [(i, probs[i] / mass) for i in indices], mass


def draw(candidates, rng: Random):
    """Sample an index from a candidate set produced by _as_set."""
    r = rng.random()
    acc = 0.0
    for idx, p in candidates:
        acc += p
        if r <= acc:
            return idx
    return candidates[-1][0]


def greedy(candidates):
    return max(candidates, key=lambda t: (t[1], -t[0]))[0]


class Mirostat:
    """Target-perplexity control.

    Instead of fixing a truncation threshold, hold a running estimate of the
    surprise and move the threshold to keep the *observed* perplexity near a
    target. The corpus notes this needs a per-model, per-task target and that
    it converges poorly on short outputs; the simulation shows both.
    """

    def __init__(self, target_perplexity=12.0, learning_rate=0.8):
        self.target = math.log(target_perplexity)
        self.rate = learning_rate
        self.mu = 2.0 * self.target

    def step(self, probs, rng: Random):
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        kept, acc = [], 0.0
        for i in order:
            surprise = -math.log(probs[i]) if probs[i] > 0.0 else float("inf")
            if surprise > self.mu:
                break
            kept.append(i)
            acc += probs[i]
        if not kept:
            kept = [order[0]]
        candidates, _ = _as_set(probs, kept)
        chosen = draw(candidates, rng)
        observed = -math.log(probs[chosen]) if probs[chosen] > 0.0 else self.mu
        self.mu -= self.rate * (observed - self.target)
        self.mu = max(self.mu, 0.05)
        return chosen, observed
