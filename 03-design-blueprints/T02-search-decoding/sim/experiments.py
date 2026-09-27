"""The experiments run.py executes.

Every count printed here is this simulation's own, over a four-token toy
language model. The corpus's own figures -- the 10x best-first speedup, the
16x16 KL matrix, the 4/6/9/8 step counts on the lecture's graph -- are quoted
and attributed in HLD.md section 9, and are NOT reproduced by this model.
"""

from .lm import render
from . import search

MAX_LEN = 6


def exp_decoder_zoo(seed=0):
    """The same task through five settings of one machine."""
    optimum = search.brute_force_best(max_len=MAX_LEN)

    rows = []

    g = search.greedy(max_len=MAX_LEN)
    rows.append(_row("greedy", g, optimum))

    b2 = search.beam_search(K=2, max_len=MAX_LEN, normalize=False)
    rows.append(_row("beam K=2", b2, optimum))

    b4 = search.beam_search(K=4, max_len=MAX_LEN, normalize=False)
    rows.append(_row("beam K=4", b4, optimum))

    uc = search.priority_search(K=float("inf"), heuristic=None, max_len=MAX_LEN)
    rows.append(_row("uniform-cost (K=inf, h=0)", uc, optimum))

    bf = search.priority_search(
        K=4, heuristic=None, max_len=MAX_LEN, per_length_cap=4)
    rows.append(_row("best-first beam (K=4, h=0)", bf, optimum))

    return rows, optimum


def _row(name, result, optimum):
    return {
        "decoder": name,
        "path": render(result["path"]),
        "cost": result["cost"],
        "expansions": result["expansions"],
        "optimal": abs(result["cost"] - optimum["cost"]) < 1e-9,
    }


def exp_length_normalization(alpha=1.0):
    """What length normalization does to which sequence you emit.

    Rank the same completed candidates four ways and see the winner change.
    This is the mechanism behind "beam width raised, BLEU fell": the width
    controls the candidate *set*, and the normalizer controls which member of
    it you ship.
    """
    result = search.beam_search(
        K=6, max_len=MAX_LEN, normalize=False, alpha=alpha)
    candidates = result["candidates"]
    variants = [
        ("none (raw log-prob)", False, 1.0),
        ("divide by len^0.5", True, 0.5),
        ("divide by length", True, 1.0),
        ("divide by len^1.5", True, 1.5),
    ]
    rows = []
    for label, norm, a in variants:
        ranked = sorted(
            candidates,
            key=lambda pc: (search.normalized(pc[1], len(pc[0]), a, norm), pc[0]))
        best_path, best_cost = ranked[0]
        rows.append({
            "normalization": label,
            "selected": render(best_path),
            "length": len(best_path),
            "raw_cost": best_cost,
            "score": search.normalized(best_cost, len(best_path), a, norm),
        })
    return rows


def exp_bounded_astar(length=6):
    """A* pays off only where the length is known and h is computable."""
    uc = search.bounded_astar(
        length, heuristic=search.constant_heuristic(0.0))
    a = search.bounded_astar(length, heuristic=search.row_min_heuristic)
    return [
        {"search": "uniform-cost (h=0)", "expansions": uc["expansions"],
         "cost": uc["cost"], "path": render(uc["path"])},
        {"search": "A* (row-minimum h)", "expansions": a["expansions"],
         "cost": a["cost"], "path": render(a["path"])},
    ]


def exp_heuristic_breaks_on_memorised_span(max_len=MAX_LEN):
    """Why no admissible heuristic exists over an open-ended decoder.

    The row-minimum heuristic is provably admissible in BOUNDED mode (section
    3 shows it winning there). Carry the same function one line over to the
    open-ended decoder -- where EOS is an available action -- and it stops
    being a lower bound: EOS out of `c` costs 0.799 nats, well below `c`'s
    cheapest non-EOS arc, so the row minimum over non-EOS arcs now
    OVERestimates what is left to pay. The guarantee evaporates and the search
    commits to a worse answer.

    This is the corpus's warning made concrete: "you might deviate from the
    optimal solution. You're not guaranteed to deviate... but you might" [T].
    """
    optimum = search.brute_force_best(max_len=max_len)
    settings = [
        ("uniform-cost (h=0)", None),
        ("row-min h (bounded-mode h)", search.row_min_heuristic),
        ("pessimistic h", search.pessimistic_heuristic),
    ]
    rows = []
    for label, h in settings:
        r = search.priority_search(K=float("inf"), heuristic=h, max_len=max_len)
        rows.append({
            "search": label,
            "path": render(r["path"]),
            "cost": r["cost"],
            "expansions": r["expansions"],
            "optimal": abs(r["cost"] - optimum["cost"]) < 1e-9,
        })
    return rows, optimum


def exp_recombination(max_len=3, k=16):
    """n-gram clustering versus KL over next-token distributions."""
    paths = search.enumerate_paths(max_len=max_len)[:k]
    k_actual = len(paths)

    rows = []
    for n in (1, 2, 3):
        groups = search.ngram_clusters(paths, n)
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        rows.append({
            "criterion": "n-gram (last %d tokens)" % n,
            "clusters": len(groups),
            "largest_cluster": sizes[0],
            "pairwise_ops": None,
        })

    kl = search.kl_clusters(paths, threshold=0.5)
    rows.append({
        "criterion": "KL over next-token dist (t=0.5)",
        "clusters": len(kl["clusters"]),
        "largest_cluster": max(len(c) for c in kl["clusters"]),
        "pairwise_ops": kl["pairwise_kl_computed"],
    })
    return rows, k_actual


def exp_diverse_beam(groups=3, beams_per_group=1, max_len=4):
    """`t + g - 1` steps for g groups, not `g * t` [T]."""
    t = max_len
    return {
        "groups": groups,
        "beams_per_group": beams_per_group,
        "steps": t,
        "sequential_steps": groups * t,
        "batched_steps": t + groups - 1,
        "saving_factor": (groups * t) / (t + groups - 1),
    }


def exp_duplicate_check(groups=3, max_len=4):
    """Plain beam's top-g against diverse beam's g groups.

    HONEST SCOPE: in this representation a hypothesis IS its token tuple, so
    two outputs can never be the same object and "zero exact duplicates" is
    trivially true here. The promise is not trivial on a real 128k vocabulary,
    where many distinct token paths render to the same string, and that is
    what the deduplication step is for. This model cannot measure that
    benefit and does not claim to. What it can show is which hypotheses each
    setting selects.
    """
    plain = search.beam_search(K=groups, max_len=max_len, normalize=False)
    plain_paths = [render(p) for p, _ in plain["candidates"][:groups]]
    diverse = search.beam_search(
        K=groups, max_len=max_len, normalize=False, diverse_groups=groups,
        diversity_penalty=0.5)
    div_paths = [render(p) for p, _ in diverse["candidates"][:groups]]
    return {
        "plain_outputs": plain_paths,
        "diverse_outputs": div_paths,
    }
