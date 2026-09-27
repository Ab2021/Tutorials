"""Chain of thought as a latent variable, and the three ways to read it.

The corpus's formulation: X is the problem, the chain Z is a LATENT variable,
Y is the answer, and the goal is p(y | x) obtained by "marginalizing over z. So
summing over all Z so that we can get a better prediction of Y".

Three decoders read that sum differently, and on the corpus's own counterexample
they do not agree:

  * GREEDY          argmax_z P(z|x), then argmax_y P(y|x,z)   -- mode-seeking
  * JOINT ARGMAX    argmax_{z,y} P(z|x)P(y|x,z)               -- the trap
  * MARGINAL        argmax_y SUM_z P(z|x)P(y|x,z)             -- the target

The corpus's counterexample is "how many inches are there in 3 ft": a model that
talks about centimetres with high probability "would give the best joint z-y
score but not the best p(y)". The toy problem below is built to have exactly
that shape, and the arithmetic is small enough to check by hand.

This is a MODEL. The probabilities are chosen, not measured; no language model
was run. The corpus's numbers (v^100 latent terms, 100 samples = 100x cost,
the 0.95 threshold) are quoted in HLD.md and are NOT reproduced here.
"""

import random

# --------------------------------------------------------------- the problems

PROBLEMS = {
    # The corpus's counterexample shape. The single highest-probability chain
    # is the centimetres one, and it is also the highest-probability JOINT pair
    # -- but the sum over chains prefers 36, because the two inch chains agree
    # with each other.
    "3ft_in_inches": {
        "answers": ["36", "91"],
        "correct": "36",
        "chains": {
            "cm_reasoning": (0.50, {"36": 0.40, "91": 0.60}),
            "inch_convert": (0.25, {"36": 0.90, "91": 0.10}),
            "unit_cancel":  (0.25, {"36": 0.90, "91": 0.10}),
        },
    },
    # The same shape with the wrong mass CONCENTRATED in one chain, so a short
    # unlucky run is unanimous and wrong. This is the fixture for the adaptive
    # rule's blind spot: the posterior measures agreement, not correctness.
    "trap": {
        "answers": ["36", "91"],
        "correct": "36",
        "chains": {
            "memorized_cm": (0.45, {"36": 0.0, "91": 1.0}),
            "derive_in":    (0.30, {"36": 1.0, "91": 0.0}),
            "cancel_units": (0.25, {"36": 1.0, "91": 0.0}),
        },
    },
    # A question where greedy is already right and the marginal agrees. Self-
    # consistency spends n samples here and changes nothing -- which is half the
    # cost argument in the capacity model.
    "2plus3": {
        "answers": ["5", "6"],
        "correct": "5",
        "chains": {
            "count_up": (0.80, {"5": 0.98, "6": 0.02}),
            "fingers":  (0.20, {"5": 0.90, "6": 0.10}),
        },
    },
}


def chains(problem):
    return PROBLEMS[problem]["chains"]


def answers(problem):
    return PROBLEMS[problem]["answers"]


def correct_answer(problem):
    return PROBLEMS[problem]["correct"]


def chain_prob(problem, chain):
    """P(z | x)."""
    return chains(problem)[chain][0]


def answer_given_chain(problem, chain, answer):
    """P(y | x, z)."""
    return chains(problem)[chain][1].get(answer, 0.0)


def marginal(problem):
    """P(y | x) = sum_z P(z|x) P(y|x,z). The quantity the corpus wants."""
    out = {}
    for a in answers(problem):
        out[a] = sum(chain_prob(problem, z) * answer_given_chain(problem, z, a)
                     for z in chains(problem))
    return out


def marginal_argmax(problem):
    m = marginal(problem)
    return max(sorted(m), key=lambda a: m[a])


def greedy(problem):
    """argmax_z P(z|x), then argmax_y P(y|x,z). Returns (chain, answer)."""
    z = max(sorted(chains(problem)), key=lambda c: chain_prob(problem, c))
    ys = chains(problem)[z][1]
    y = max(sorted(ys), key=lambda a: ys[a])
    return z, y


def joint_argmax(problem):
    """argmax_{z,y} P(z|x) P(y|x,z). Returns (chain, answer, probability)."""
    best = None
    for z in sorted(chains(problem)):
        for y, p in sorted(chains(problem)[z][1].items()):
            score = chain_prob(problem, z) * p
            if best is None or score > best[2]:
                best = (z, y, score)
    return best


def sample(problem, rng):
    """One ancestral draw: z ~ P(z|x), then y ~ P(y|x,z).

    The corpus's recommendation for the hard direction is exactly this --
    "ancestral sampling with a temperature of one is going to always be good
    enough uh in an auto regressive model".
    """
    zs = sorted(chains(problem))
    z = _draw(zs, [chain_prob(problem, c) for c in zs], rng)
    ys = sorted(chains(problem)[z][1])
    y = _draw(ys, [chains(problem)[z][1][a] for a in ys], rng)
    return z, y


def _draw(items, weights, rng):
    r = rng.random() * sum(weights)
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def self_consistency(problem, rng, n):
    """Sample n chains, majority-vote the answer. Returns (answer, tally)."""
    tally = {a: 0 for a in answers(problem)}
    for _ in range(n):
        _, y = sample(problem, rng)
        tally[y] += 1
    winner = max(sorted(tally), key=lambda a: tally[a])
    return winner, tally


def make_rng(seed):
    return random.Random(seed)
