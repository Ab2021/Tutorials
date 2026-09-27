"""One priority-queue decoder, four configurations, plus recombination.

The corpus draws the unification explicitly: beam search "always prioritizes
things that are shorter... and then... if the output is the same length it
prioritizes based on score," best-first "always prioritizes based on score. Uh
but if the score is the same then it prioritizes based on length," K "is the
beam size for one and infinite for the other," and "beam search doesn't have
one [a heuristic]. Best search doesn't have one but a star search does" [T].

So: comparator, K, and h are the three parameters. Everything else is shared.

Costs are -log p in nats. LOWER IS BETTER.
"""

import heapq
import math

from .lm import EOS, START, cost, next_states, render, row_min_cost


def normalized(cost_value, length, alpha=1.0, normalize=True):
    """Length-normalized score -- HF's `len^alpha` length penalty family.

    Without it, EOS gets "an unfairly good chance" [T] and outputs shorten as
    width rises. With the wrong alpha, extreme values "explode" and invert the
    preference toward short sequences [T]. There is no alpha that makes all
    lengths comparable.
    """
    if not normalize or length == 0:
        return cost_value
    return cost_value / (length ** alpha)


# ---------------------------------------------------------------- greedy

def greedy(max_len=8):
    path, prev, expansions = [], START, 0
    for _ in range(max_len):
        options = next_states(prev)
        best = min(options, key=lambda t: (cost(prev, t), t))
        expansions += len(options)
        path.append(best)
        prev = best
        if best == EOS:
            break
    return {"path": tuple(path), "cost": sum(
        cost((path[i - 1] if i else START), path[i]) for i in range(len(path))
    ), "expansions": expansions}


# ----------------------------------------------------------- beam search

def beam_search(K=4, max_len=8, alpha=1.0, normalize=True, allow_eos=True,
                diverse_groups=1, diversity_penalty=0.0):
    """Classic beam: expand every beam, prune back to K by normalized score.

    Diverse beam search is the same loop with `diverse_groups` groups, each
    expanded with a penalty against tokens already chosen by earlier groups,
    and the two run in `t + g - 1` steps rather than `g * t` [T].
    """
    beams = [((), 0.0)]
    completed = []
    expansions = 0

    for _ in range(max_len):
        live = [b for b in beams if not b[0] or b[0][-1] != EOS]
        done = [b for b in beams if b[0] and b[0][-1] == EOS]
        completed.extend(done)
        if not live:
            break

        cands = []
        for path, c in live:
            prev = path[-1] if path else START
            for t in next_states(prev, allow_eos):
                expansions += 1
                cands.append((path + (t,), c + cost(prev, t)))

        if diverse_groups > 1 and diversity_penalty > 0.0:
            cands = _apply_diversity(cands, diversity_penalty)

        cands.sort(key=lambda pc: (normalized(pc[1], len(pc[0]), alpha, normalize),
                                   pc[0]))
        beams = cands[:K]

    completed.extend(b for b in beams if b[0] and b[0][-1] == EOS)
    # A beam that never emitted EOS is still a shippable output: hitting the
    # length cap is how a decoder terminates when the model will not stop.
    # Dropping these is the classic silent bug -- the candidate set silently
    # becomes "every EOS path the beam happened to keep", which is exactly the
    # short-sequence bias the normalizer exists to correct.
    completed.extend(b for b in beams if not b[0] or b[0][-1] != EOS)
    if not completed:
        completed = list(beams)
    completed.sort(key=lambda pc: (normalized(pc[1], len(pc[0]), alpha, normalize),
                                   pc[0]))
    best_path, best_cost = completed[0]
    return {
        "path": best_path,
        "cost": best_cost,
        "normalized": normalized(best_cost, len(best_path), alpha, normalize),
        "expansions": expansions,
        "candidates": completed[:K],
    }


def _apply_diversity(cands, penalty):
    """Diverse beam search's de-duplication: "encourage each new output we

    decode to be different from everything else that we've decoded up to that
    point" [T]. Modelled as a per-token penalty against tokens already used at
    the same time step by an earlier group -- which is the *cumulative* form
    the lecture prefers over raw Hamming diversity, since Hamming penalises
    "the" regardless of position [T].
    """
    seen = {}
    out = []
    for path, c in cands:
        step = len(path)
        tok = path[-1] if path else None
        used = seen.get(step, set())
        out.append((path, c + (penalty if tok in used else 0.0)))
        used.add(tok)
        seen[step] = used
    return out


# ----------------------------------------- priority-queue search family

def priority_search(K=float("inf"), heuristic=None, max_len=8, allow_eos=True,
                    alpha=1.0, normalize=True, per_length_cap=None):
    """Uniform-cost (K=inf, h=0), A* (K=inf, h>0) and best-first beam.

    `heuristic(state, steps_remaining) -> lower bound on remaining cost`.
    With an admissible heuristic the first completed path popped is optimal.
    """
    h = heuristic or (lambda state, remaining: 0.0)
    counter = 0
    start = ((), 0.0, START)
    pq = [(h(START, max_len), 0.0, counter, (), START)]
    expansions = 0
    best_complete = None
    per_length_count = {}

    while pq:
        f, g, _, path, prev = heapq.heappop(pq)

        if best_complete is not None and f >= best_complete[1]:
            break
        if len(path) >= max_len:
            if best_complete is None or g < best_complete[1]:
                best_complete = (path, g)
            continue

        expansions += 1
        for t in next_states(prev, allow_eos):
            npath = path + (t,)
            ng = g + cost(prev, t)
            if per_length_cap is not None:
                n = per_length_count.get(len(npath), 0)
                if n >= per_length_cap:
                    continue
                per_length_count[len(npath)] = n + 1
            if t == EOS:
                if best_complete is None or ng < best_complete[1]:
                    best_complete = (npath, ng)
                continue
            counter += 1
            remaining = max_len - len(npath)
            heapq.heappush(pq, (ng + h(t, remaining), ng, counter, npath, t))

    if best_complete is None:
        best_complete = ((), 0.0)
    return {
        "path": best_complete[0],
        "cost": best_complete[1],
        "expansions": expansions,
        # Only the h=0 setting carries a guarantee. Any non-zero heuristic must
        # be argued admissible, and over an open-ended decoder it is not.
        "optimal_guaranteed": heuristic is None,
    }


# ------------------------------------------------------- bounded A*

def bounded_astar(length, heuristic=None):
    """A* over a segment whose length is known in advance.

    This is the only place A* is usable: with the path length fixed there is no
    EOS absorbing state, and a row-minimum heuristic is computable. Over an
    open-ended decoder it is not constructible -- see experiments
    `exp_heuristic_breaks_on_memorised_span`.
    """
    h = heuristic or (lambda state, remaining: 0.0)
    counter = 0
    pq = [(0.0 + h(START, length), 0.0, counter, (), START)]
    expansions = 0
    best = None

    while pq:
        f, g, _, path, prev = heapq.heappop(pq)
        if len(path) == length:
            best = (path, g)
            break
        if best is not None and f >= best[1]:
            break
        expansions += 1
        for t in next_states(prev, allow_eos=False):
            npath = path + (t,)
            ng = g + cost(prev, t)
            remaining = length - len(npath)
            counter += 1
            heapq.heappush(
                pq, (ng + h(t, remaining), ng, counter, npath, t))
    return {"path": best[0], "cost": best[1], "expansions": expansions}


def row_min_heuristic(state, remaining):
    """h = remaining steps x that state's row minimum. Admissible."""
    if remaining <= 0:
        return 0.0
    return remaining * row_min_cost(state, allow_eos=False)


def constant_heuristic(per_step):
    """A fixed cost per remaining step. Harmless when every path has the same

    length -- a constant does not reorder nodes within a level -- so this is
    used as the h=0 baseline in bounded mode.
    """
    def h(state, remaining):
        return per_step * remaining
    return h


def pessimistic_heuristic(state, remaining):
    """h = remaining x the WORST arc out of the state. Inadmissible.

    This is the failure the corpus describes in the memorised case. A state
    the model is about to march through at near-zero cost is charged its most
    expensive arc for every remaining step, so the search deprioritises
    precisely the path it should take and commits to a worse one first.
    """
    if remaining <= 0:
        return 0.0
    from .lm import cost_row
    return remaining * max(cost_row(state, allow_eos=False))


# -------------------------------------------------------- optimality check

def brute_force_best(max_len=8, allow_eos=True):
    """Exhaustive optimum, for checking whether a search found it."""
    best = None
    stack = [((), START, 0.0)]
    while stack:
        path, prev, g = stack.pop()
        if path and prev == EOS:
            if best is None or g < best[1]:
                best = (path, g)
            continue
        if len(path) >= max_len:
            if best is None or g < best[1]:
                best = (path, g)
            continue
        for t in next_states(prev, allow_eos):
            stack.append((path + (t,), t, g + cost(prev, t)))
    return {"path": best[0], "cost": best[1]}


# ------------------------------------------------------------ recombination

def ngram_clusters(hypotheses, n):
    """Cluster by the most recent n tokens -- "very easy to implement in cache"."""
    groups = {}
    for h in hypotheses:
        key = h[-n:] if n > 0 else ()
        groups.setdefault(key, []).append(h)
    return groups


def kl_clusters(hypotheses, threshold, allow_eos=True):
    """Cluster by KL divergence between the next-token distributions.

    The corpus's mechanics: for each live hypothesis compute a distribution
    over the vocabulary, take pairwise KL, and prune "the ones where the kale
    divergence is like relatively low" [T]. The compared object is explicitly
    the distribution, not the hidden state -- "similarity in neural
    representations is not the correct term here" [T].
    """
    dists = {}
    for h in hypotheses:
        prev = h[-1] if h else START
        row = [math.exp(-c) for c in _row(prev, allow_eos)]
        s = sum(row)
        dists[h] = [p / s for p in row]

    pairs = 0
    merged = []
    used = set()
    for i, a in enumerate(hypotheses):
        if a in used:
            continue
        cluster = [a]
        used.add(a)
        for b in hypotheses[i + 1:]:
            if b in used:
                continue
            pairs += 1
            if _kl(dists[a], dists[b]) < threshold:
                cluster.append(b)
                used.add(b)
        merged.append(cluster)
    return {"clusters": merged, "pairwise_kl_computed": pairs}


def _row(prev, allow_eos):
    from .lm import cost_row
    return cost_row(prev, allow_eos)


def _kl(p, q):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0 and qi > 0.0:
            total += pi * math.log(pi / qi)
    return total


def enumerate_paths(max_len=5):
    """All EOS-free paths of a fixed length, for the recombination experiments."""
    out = []
    stack = [((), START)]
    while stack:
        path, prev = stack.pop()
        if len(path) == max_len:
            out.append(path)
            continue
        for t in next_states(prev, allow_eos=False):
            stack.append((path + (t,), t))
    return out


def show(path):
    return render(path) if path else "(empty)"
