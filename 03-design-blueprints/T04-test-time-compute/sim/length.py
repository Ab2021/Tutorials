"""Output length: the exceed rate, the clipped answer, and the cosine reward.

The corpus's failure story is specific and worth reproducing as arithmetic
rather than as a warning: models "would improve for a while and then they would
suddenly crash... they were exceeding the maximum output length", and "once they
started exceeding the maximum output length they would be getting all of the
problems wrong basically because the final answer was getting clipped". Training
"died". The metric is the EXCEED RATE.

The design's inference-time analogue is not a reward but a budget: never let a
clipped generation be returned as an answer. This module prices that.

The corpus's fix, for context: a reward "where we multiply it by the cosine...
where the cosine will converge to zero at the maximum output length. And so if
the answer is wrong we basically give a larger negative reward when the answer
is short... And then if you're getting it right, make it shorter." That shape is
computed here so both stated behaviours are visible.

This is a MODEL. The length distribution and the base accuracy are chosen to be
plausible, not measured; no model was run.
"""

import math

# A length distribution over reasoning traces, in tokens. Deliberately heavy-
# tailed: most traces are short and a minority run long, which is what makes the
# exceed rate sensitive to the budget in the region that matters.
LENGTH_PMF = [
    (200, 0.06), (400, 0.14), (600, 0.20), (800, 0.18), (1000, 0.14),
    (1200, 0.11), (1600, 0.09), (2000, 0.05), (2400, 0.03),
]

MAX_LENGTH = 2000


def mean_length():
    return sum(l * p for l, p in LENGTH_PMF)


def exceed_rate(budget):
    """Fraction of traces that would run past `budget` and be clipped."""
    return sum(p for l, p in LENGTH_PMF if l > budget)


def accuracy_under_budget(budget, base_accuracy, conclude_prob=0.0):
    """A clipped trace cannot emit its final answer, so it is counted wrong.

    `conclude_prob` is the probability that an over-budget trace obeys an
    instruction to conclude and emits its answer inside the budget anyway --
    the S1-style budget forcing the corpus describes ("they cut it off and they
    said... now answer and it answered").
    """
    clipped = exceed_rate(budget) * (1.0 - conclude_prob)
    return base_accuracy * (1.0 - clipped)


def cosine_term(length, max_length=MAX_LENGTH):
    """cos(pi/2 * L / Lmax): 1 at zero length, 0 at the maximum.

    The corpus's "cosine will converge to zero at the maximum output length".
    """
    frac = min(max(length, 0), max_length) / float(max_length)
    return math.cos((math.pi / 2.0) * frac)


def cosine_reward(correct, length, max_length=MAX_LENGTH):
    """correctness times the cosine, with correctness at +1 / -1.

    The classic reward is "one when you get it correct and a reward of zero when
    you get it wrong"; the corpus's variant multiplies by the cosine, and the
    stated consequences are that a wrong answer gets "a larger negative reward
    when the answer is short" and a correct one is pushed shorter. Scaling
    +/-1 by the cosine reproduces both; scaling the classic 1/0 by it would not,
    because zero times anything is still zero.
    """
    return (1.0 if correct else -1.0) * cosine_term(length, max_length)


def budget_table(budgets, base_accuracy, conclude_prob=0.0):
    rows = []
    for b in budgets:
        rows.append({
            "budget": b,
            "exceed_rate": exceed_rate(b),
            "clipped_wrong": exceed_rate(b) * (1.0 - conclude_prob),
            "accuracy": accuracy_under_budget(b, base_accuracy, conclude_prob),
        })
    return rows


def reward_table(lengths):
    return [{"length": l, "cosine": cosine_term(l),
             "reward_correct": cosine_reward(True, l),
             "reward_wrong": cosine_reward(False, l)} for l in lengths]
