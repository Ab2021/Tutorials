"""The verifier hierarchy. This module IS the system -- best-of-N is trivial, scoring is
where the value and every failure mode lives.

Every bias is an explicit parameter rather than something baked into the arithmetic, so an
experiment can set it to zero and observe what changes. The corpus's list of judge biases
(position, verbosity, self-preference -- Jung et al. 2023) [T] is reproduced here as three
independent knobs.
"""
from __future__ import annotations

from .candidates import Candidate


# --------------------------------------------------------------------------------------
# Tier 0 -- exact, programmatic. ALWAYS run first: it is free and it is correct.
# --------------------------------------------------------------------------------------

def programmatic_score(c: Candidate, rules: list) -> dict:
    """Run exact checks. A failing check ELIMINATES the candidate.

    `rules` is a list of callables `candidate -> (ok, reason)`. This is the tier the corpus
    says to always prefer when a check can be written [T]: for code that is a unit test, a
    compiler or a type checker -- "the strongest verifier that exists".
    """
    for rule in rules:
        ok, reason = rule(c)
        if not ok:
            return {"candidate_id": c.id, "tier": 0, "score": 0.0,
                    "eliminated": True, "reason": reason}
    return {"candidate_id": c.id, "tier": 0, "score": 1.0,
            "eliminated": False, "reason": None}


# --------------------------------------------------------------------------------------
# Tier 1 -- outcome reward model. One forward pass per candidate.
# --------------------------------------------------------------------------------------

def orm_score(c: Candidate, weights: dict | None = None) -> dict:
    """A learned, sequence-level score. Cheap relative to a judge: one forward, no decode.

    Reads `orm_quality` when present, so an experiment can give the ORM a NOISIER view of
    the candidate than the judge has. That is what makes the cascade's recall limitation
    (`top_k`) observable rather than assumed away.
    """
    w = weights or {"quality": 1.0}
    quality = float(c.meta.get("orm_quality", c.meta.get("quality", 0.0)))
    score = w.get("quality", 1.0) * quality
    return {"candidate_id": c.id, "tier": 1, "score": score,
            "eliminated": False, "reason": None}


# --------------------------------------------------------------------------------------
# Tier 2 -- process reward model. One forward pass per STEP.
# --------------------------------------------------------------------------------------

def prm_score(c: Candidate, steps: list[float], weights: dict | None = None) -> dict:
    """Per-step scoring. More expensive than an ORM, and worth it when the error is
    mid-trajectory rather than in the final answer: it can separate two candidates that
    agree on the answer but differ in how they got there."""
    w = weights or {"mean": 1.0}
    if not steps:
        return {"candidate_id": c.id, "tier": 2, "score": 0.0,
                "eliminated": True, "reason": "no steps to score"}
    score = w.get("mean", 1.0) * (sum(steps) / len(steps))
    return {"candidate_id": c.id, "tier": 2, "score": score,
            "eliminated": False, "reason": None}


# --------------------------------------------------------------------------------------
# Tier 3 -- generative judge. One GENERATION per call, so it can cost more than the
# candidate it is scoring.
# --------------------------------------------------------------------------------------

def judge_score(c: Candidate, pool_position: int = 0, pool_size: int = 1,
                length_bias: float = 0.0, position_bias: float = 0.0,
                judge_family: str | None = None) -> dict:
    """A prompted judge, with its three documented biases as parameters.

    raw = quality
        + length_bias    * (length / 1000)          verbosity bias
        + position_bias  * (1 - position / size)    position bias (earlier ranks higher)
        + FAMILY_BONUS   if judge_family == candidate family   self-preference

    Set every bias to 0 and this is an honest oracle; that is the CONTROL case, and the
    experiments assert the DIFFERENCE between the biased and unbiased runs rather than
    any absolute score.
    """
    FAMILY_BONUS = 0.15
    quality = float(c.meta.get("quality", 0.0))
    raw = quality
    raw += length_bias * (c.length / 1000.0)
    if pool_size > 1:
        raw += position_bias * (1.0 - pool_position / (pool_size - 1))
    if judge_family is not None and c.meta.get("family") == judge_family:
        raw += FAMILY_BONUS
    return {"candidate_id": c.id, "tier": 3, "score": raw,
            "eliminated": False, "reason": None}


# --------------------------------------------------------------------------------------
# The cascade -- the single biggest cost lever in the blueprint.
# --------------------------------------------------------------------------------------

def tiered_score(candidates: list[Candidate], rules: list, top_k: int = 4,
                 orm_weights: dict | None = None, judge_kwargs: dict | None = None
                 ) -> tuple[list[dict], dict]:
    """Tier 0 on all -> tier 1 on survivors -> tier 3 on the top k.

    Judging all n costs n judge generations; judging k costs k. The ORM's coarse ranking
    makes the judge's job a comparison among near-equals, which is what judges are best at.

    Returns (scores, usage) where usage is the COST RECORD: {"0": n, "1": n', "3": k}.
    A decision that cannot say how many candidates reached the judge cannot support a cost
    claim, so the usage dict is not optional.
    """
    jk = judge_kwargs or {}
    scores: dict[str, dict] = {}

    # tier 0 -- all
    for c in candidates:
        scores[c.id] = programmatic_score(c, rules)

    survivors = [c for c in candidates if not scores[c.id]["eliminated"]]

    # tier 1 -- survivors
    for c in survivors:
        scores[c.id] = orm_score(c, orm_weights)

    # tier 3 -- top k by tier-1 score
    ranked = sorted(survivors, key=lambda c: scores[c.id]["score"], reverse=True)
    judged = ranked[:max(0, top_k)]
    for pos, c in enumerate(judged):
        s = judge_score(c, pool_position=pos, pool_size=len(judged), **jk)
        s["tier_score1"] = scores[c.id]["score"]   # keep the coarse rank for audit
        scores[c.id] = s

    usage = {"0": len(candidates), "1": len(survivors), "3": len(judged)}
    return [scores[c.id] for c in candidates], usage


# --------------------------------------------------------------------------------------
# Trust -- a verifier you have not measured is not a verifier.
# --------------------------------------------------------------------------------------

def measure_agreement(scorer_verdicts: list[bool], human_verdicts: list[bool]) -> float:
    """|scorer agrees with human| / |gold set|.

    Re-measure whenever the scorer's model, prompt or version changes. There is no default
    here: an absent measurement is the absence of a verifier, not a neutral prior.
    """
    if not scorer_verdicts:
        return 0.0
    if len(scorer_verdicts) != len(human_verdicts):
        raise ValueError("gold set length mismatch")
    agree = sum(1 for s, h in zip(scorer_verdicts, human_verdicts) if s == h)
    return agree / len(scorer_verdicts)


def judge_flip_rate(judge_family: str | None, candidates: list[Candidate],
                    length_bias: float = 0.0, position_bias: float = 0.0) -> dict:
    """How often does the family bonus alone change the winner? Used by exp_judge_family."""
    from .selection import select

    scores, _ = tiered_score(
        candidates, rules=[],
        top_k=len(candidates),
        judge_kwargs={"judge_family": judge_family,
                      "length_bias": length_bias, "position_bias": position_bias},
    )
    winner_with = select(scores, candidates)[0]

    scores0, _ = tiered_score(
        candidates, rules=[],
        top_k=len(candidates),
        judge_kwargs={"judge_family": None,
                      "length_bias": length_bias, "position_bias": position_bias},
    )
    winner_without = select(scores0, candidates)[0]

    return {"with_family": winner_with, "without_family": winner_without,
            "changed": winner_with != winner_without}
