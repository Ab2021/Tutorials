"""Adaptive self-consistency: a Dirichlet prior, a Beta stopping rule.

The corpus's account, which this module implements literally:

  * the posterior is proportional to the observed count plus alpha times the
    prior -- "count + alpha * P of prior";
  * alpha controls how much you lean on the prior: "If alpha is higher you rely
    on the prior probability more... If alpha is zero this is maximum likelihood
    estimation";
  * the worked example: counts 1,0,0 with alpha = 3 and a uniform prior gives
    pseudo-counts 2,1,1 and therefore "0.5 for A, 0.25 for B, 0.25 for C";
  * the Dirichlet is simplified to a Beta over the top-1 and top-2 outputs
    because the full-vocabulary Dirichlet "can be kind of expensive";
  * stop when the probability of the top output "continuing to be true" after
    more sampling exceeds 0.95, checked "after each batch of generations".

Everything here is arithmetic over counts. No model was run.
"""

import math

from . import cot


def dirichlet_posterior(counts, alpha, prior=None):
    """(count_i + alpha * prior_i) normalised.

    `counts` is a dict answer -> observed count; `prior` a dict answer ->
    probability (uniform if omitted). alpha = 0 reduces to maximum likelihood,
    which is the corpus's stated boundary case.
    """
    keys = sorted(counts)
    if prior is None:
        prior = {k: 1.0 / len(keys) for k in keys}
    pseudo = {k: counts[k] + alpha * prior.get(k, 0.0) for k in keys}
    total = sum(pseudo.values())
    if total == 0.0:
        return {k: 0.0 for k in keys}, pseudo
    return {k: pseudo[k] / total for k in keys}, pseudo


# ------------------------------------------------------------------- Beta

def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(x, a, b):
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x)
                    - _log_beta(a, b))


_BETA_CACHE = {}


def beta_leader_wins(a1, a2, steps=400):
    """P(p1 > p2) for the leader's share ~ Beta(a1, a2).

    The quantity the corpus describes as "the probability of number one output
    continuing to be true after you sampled more information": with two
    candidates in play, p1 > p2 is p1 > 0.5. Computed by Simpson's rule over
    [0.5, 1] on the Beta density.

    The corpus says of the derivation only "you can look at the paper for the
    full derivation", so the parameterisation here is this author's choice: the
    same alpha with a uniform prior over the two candidates in play, i.e.
    a1 = c1 + alpha/2 and a2 = c2 + alpha/2.

    Memoised: the counts are small integers, so a whole experiment touches only
    a few dozen distinct (a1, a2) pairs.
    """
    if a1 <= 0.0 or a2 <= 0.0:
        return 0.0
    key = (round(a1, 6), round(a2, 6))
    hit = _BETA_CACHE.get(key)
    if hit is not None:
        return hit

    h = 0.5 / steps
    total = beta_pdf(0.5, a1, a2) + beta_pdf(1.0, a1, a2)
    for i in range(1, steps):
        x = 0.5 + i * h
        total += (4.0 if i % 2 else 2.0) * beta_pdf(x, a1, a2)
    value = max(0.0, min(1.0, total * h / 3.0))
    _BETA_CACHE[key] = value
    return value


def leader_wins(counts, alpha=3.0):
    """Top-1 vs top-2 as a Beta, with the corpus's alpha semantics."""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    c1 = ordered[0][1]
    c2 = ordered[1][1] if len(ordered) > 1 else 0
    a1 = c1 + alpha / 2.0
    a2 = c2 + alpha / 2.0
    return ordered[0][0], beta_leader_wins(a1, a2), (a1, a2)


# ---------------------------------------------------- the sampling controller

def adaptive_self_consistency(problem, rng, threshold=0.95, batch=2, cap=16,
                              alpha=3.0, min_batches=1):
    """Sample in batches until the Beta rule says stop, or the cap is hit.

    Returns the answer, the samples actually spent, the confidence at the stop,
    and the full tally. `min_batches` exists because the corpus checks the rule
    "after each batch of generations" -- with a single sample there is no second
    candidate to compare the leader against.
    """
    tally = {a: 0 for a in cot.answers(problem)}
    spent = 0
    confidence = 0.0
    leader = None
    batches = 0
    while spent < cap:
        for _ in range(min(batch, cap - spent)):
            _, y = cot.sample(problem, rng)
            tally[y] += 1
            spent += 1
        batches += 1
        leader, confidence, _ = leader_wins(tally, alpha=alpha)
        if batches >= min_batches and confidence >= threshold:
            break
    return {
        "answer": leader,
        "samples": spent,
        "confidence": confidence,
        "tally": dict(tally),
        "stopped_early": spent < cap,
    }


def fixed_self_consistency(problem, rng, n):
    """The baseline adaptive sampling is measured against."""
    answer, tally = cot.self_consistency(problem, rng, n)
    return {"answer": answer, "samples": n, "tally": tally}


def cost_multiplier(samples, greedy_cost=1.0):
    """The corpus's cost statement: "100 samples... 100 times the cost"."""
    return samples * greedy_cost
