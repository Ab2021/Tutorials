"""Synthetic next-token distributions.

These are *models* of the shape a language model's output distribution takes.
They are not measurements of any real model. The real-world figures this
simulation illustrates (68% vs 99% top-6 mass, 128k vocabulary, long tail
holding half the mass) come from the corpus and are quoted and attributed in
HLD.md section 9; the numbers printed here are this model's own output.
"""

import math
from random import Random


def softmax(logits):
    """Numerically stable softmax over a list of logits."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def normalize(probs):
    total = sum(probs)
    if total <= 0.0:
        return list(probs)
    return [p / total for p in probs]


def entropy(probs):
    """Shannon entropy in nats."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return h


def perplexity(probs, sampled_index):
    """Per-token perplexity of a single draw: 1 / p(sampled)."""
    p = probs[sampled_index]
    if p <= 0.0:
        return float("inf")
    return 1.0 / p


def head_mass_distribution(vocab_size, head_mass, head_size, tail_slope=1.05, seed=0):
    """A distribution with `head_mass` of its probability on the first

    `head_size` tokens and a long Zipf-ish tail over the rest.

    The point of the two parameters is that the *same* top-k means different
    things at different steps: a flat head leaves the top 6 holding well under
    all the mass, a peaked head leaves them holding nearly all of it.
    """
    if head_size >= vocab_size:
        raise ValueError("head_size must be smaller than vocab_size")
    tail_n = vocab_size - head_size
    weights = [1.0 / ((i + 1) ** tail_slope) for i in range(tail_n)]
    tail_total = sum(weights)
    tail_budget = 1.0 - head_mass
    probs = [head_mass / head_size] * head_size
    probs += [tail_budget * w / tail_total for w in weights]
    return normalize(probs)


def margin_distribution(vocab_size, top1=0.30, margin=0.05, tail_slope=1.05):
    """A distribution whose top two tokens are separated by exactly `margin`.

    The size of that margin is what decides whether a reduction-order-sized
    floating-point perturbation can move the argmax. A narrow margin models a
    genuine near-tie step; a wide one models a confident step.
    """
    top2 = top1 - margin
    if top2 <= 0.0:
        raise ValueError("margin must be smaller than top1")
    tail_n = vocab_size - 2
    budget = 1.0 - top1 - top2
    weights = [1.0 / ((i + 1) ** tail_slope) for i in range(tail_n)]
    tail_total = sum(weights)
    return normalize([top1, top2] + [budget * w / tail_total for w in weights])


def flat_distribution(vocab_size):
    return [1.0 / vocab_size] * vocab_size


def peaked_distribution(vocab_size, top_prob=0.9):
    """A near-one-hot distribution: what a confident step looks like."""
    rest = (1.0 - top_prob) / (vocab_size - 1)
    return [top_prob] + [rest] * (vocab_size - 1)


def jitter(probs, magnitude, rng: Random):
    """Perturb a distribution the way an unordered floating-point reduction does.

    Real GPUs sum partial results in completion order, so the same logical
    computation yields slightly different logits run to run. This models that
    as a bounded relative perturbation. It is a model, not a hardware trace.
    """
    out = []
    for p in probs:
        delta = rng.uniform(-magnitude, magnitude)
        out.append(max(p * (1.0 + delta), 0.0))
    return normalize(out)


def argmax(probs):
    best = 0
    for i, p in enumerate(probs):
        if p > probs[best]:
            best = i
    return best


def top_indices(probs, n):
    return sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:n]


def mass_of(probs, indices):
    return sum(probs[i] for i in indices)
