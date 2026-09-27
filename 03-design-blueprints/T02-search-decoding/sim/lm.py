"""A toy bigram language model, plus the cost matrix the search operates on.

Vocabulary: a, b, c and EOS. The table is hand-built to contain the one
feature the topic turns on -- a *memorised sweet spot*, a state from which the
model marches to a high-probability continuation at near-zero cost. That is
the corpus's explanation for why no admissible heuristic exists over an
open-ended decoder: asked to recite something memorised, the model "would
generate like thousands of tokens with like probability of one" [T], and any
non-zero estimate of the remaining cost overestimates.

Costs are -log(probability) in nats, so lower is better and costs add along a
path.
"""

import math

TOKENS = ["a", "b", "c", "EOS"]
EOS = 3
START = 3

# P(next | prev), rows sum to 1.
#
# Two features are deliberate. EOS is very unlikely from the start state, so a
# one-token output is expensive and the model prefers to keep going -- without
# that, every length normalizer would agree trivially. And state b is a
# memorised sweet spot: b -> b costs almost nothing, which is what makes a
# longer sequence cheaper per token than a short one, and therefore what makes
# the choice of normalizer change which sequence gets shipped.
_TABLE = {
    3: [0.440, 0.250, 0.300, 0.010],   # start / after EOS
    0: [0.200, 0.300, 0.450, 0.050],
    1: [0.150, 0.655, 0.145, 0.050],   # <- the memorised sweet spot
    2: [0.100, 0.150, 0.300, 0.450],
}


def prob(prev, nxt):
    return _TABLE[prev][nxt]


def cost(prev, nxt):
    return -math.log(prob(prev, nxt))


def cost_row(prev, allow_eos=True):
    row = list(_TABLE[prev])
    if not allow_eos:
        row[EOS] = 0.0
        total = sum(row)
        row = [p / total for p in row]
    return [-math.log(p) if p > 0.0 else float("inf") for p in row]


def next_states(prev, allow_eos=True):
    return [t for t in range(len(TOKENS)) if t != EOS or allow_eos]


def row_min_cost(prev, allow_eos=True):
    """The minimum arc weight out of a state -- the lecture's row-minimum.

    "take the minimum arc weight in each row of the graph" [T]. Summed over
    the remaining steps it is a lower bound on the remaining cost, and
    therefore admissible.
    """
    return min(cost_row(prev, allow_eos))


def path_cost(path):
    """Total cost of a token path starting from START."""
    prev = START
    total = 0.0
    for t in path:
        total += cost(prev, t)
        prev = t
    return total


def render(path):
    return " ".join(TOKENS[t] for t in path)
