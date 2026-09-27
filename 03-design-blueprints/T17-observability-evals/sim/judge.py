"""LLM-as-a-judge: the three documented biases, and what correcting each one buys.

The corpus names all three, with the citation [T]:

    "the judge itself is biased. It tends to favor the first answer it sees. It rewards length even
     when length adds nothing, and it flatters outputs from its own model family. So, randomize the
     answer order, cap or normalize length, and never let a model be the sole judge of its own
     family."                                          -- How to Evaluate LLM Apps (Jung et al., 2023)

It also gives the two scoring modes and when each is right [T]:

    "Pointwise is scoring as the judge to rate one answer from one to five. It is simple but noisy
     because absolute scores wander. Pairwise scoring instead shows the judge two answers and asks
     which is better. Pairwise is far more reliable for close calls."

**This module does not measure a real judge.** It builds a judge with a *known* ground truth -- a
latent quality per answer -- injects each bias as a parameter, and measures how far the judge's
verdicts move from the truth. That is the only way to quantify a bias: with a real judge you can
measure agreement but never know which side was right.

The consequence for the design is that **agreement with a human-labelled gold set is the number
that validates an eval**, and it must be re-measured whenever the judge model or its prompt changes
[cheat sheet, D].

Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Answer:
    """An answer with a LATENT quality the judge cannot see, plus the two things that bias it."""

    aid: str
    quality: float          # 1..5, the human truth
    length: int             # tokens
    family: str             # which model family produced it
    text: str = ""


@dataclass
class Pair:
    a: Answer
    b: Answer

    @property
    def human_winner(self) -> str:
        """The ground truth: the higher-quality answer. Ties go to `a` by convention."""
        return "a" if self.a.quality >= self.b.quality else "b"

    @property
    def margin(self) -> float:
        """How close the call is. Small margins are where pointwise scoring fails [T]."""
        return abs(self.a.quality - self.b.quality)


@dataclass
class JudgeParams:
    """Each field is one documented bias. Zero means the bias is absent."""

    name: str
    family: str = "judge-fam"
    position_bias: float = 0.0      # added to whichever answer is shown FIRST
    verbosity_weight: float = 0.0   # x z-score of length; "rewards length even when it adds nothing"
    self_pref: float = 0.0          # added when the answer is from the judge's own family
    noise: float = 0.30             # irreducible per-call noise


@dataclass
class Correction:
    """The three fixes the corpus prescribes, each independently switchable."""

    randomize_order: bool = False   # "randomize the answer order"
    length_controlled: bool = False # "cap or normalize length"
    different_family: bool = False  # "never let a model be the sole judge of its own family"


# ----------------------------------------------------------------------------------------------
# The gold set
# ----------------------------------------------------------------------------------------------

@dataclass
class GoldSet:
    pairs: list[Pair]
    length_mean: float
    length_sd: float

    @property
    def n(self) -> int:
        return len(self.pairs)

    def truths(self) -> list[str]:
        return [p.human_winner for p in self.pairs]

    def label_balance(self) -> float:
        """Fraction of pairs whose truth is slot `a`. 0.5 is balanced.

        **This is a correctness check on the gold set, not a curiosity.** If slot `a` holds the
        better answer in nearly every pair, then a judge with a POSITION BIAS that happens to favour
        slot `a` scores high agreement by agreeing with an artefact of how the set was written --
        and Cohen's kappa collapses to 0.000, which is the only reason the problem is visible at all.
        A gold set assembled by always putting the good answer first is exactly this, and it is a
        very easy mistake to make when the set is built by hand from real traffic.
        """
        t = self.truths()
        return sum(1 for x in t if x == "a") / len(t) if t else 0.5

    def is_degenerate(self, tolerance: float = 0.15) -> bool:
        """True if the labels are too imbalanced for kappa to mean anything."""
        return abs(self.label_balance() - 0.5) > tolerance

    def close_pairs(self, margin: float = 0.5) -> list[Pair]:
        """Pairs where the true quality difference is small -- the corpus's "close calls" [T]."""
        return [p for p in self.pairs if p.margin <= margin]

    def family_mix(self) -> dict[str, int]:
        acc: dict[str, int] = {}
        for p in self.pairs:
            for a in (p.a, p.b):
                acc[a.family] = acc.get(a.family, 0) + 1
        return acc


def synthetic_gold_set(n_pairs: int = 400, seed: int = 17, margin: float = 1.2,
                       own_family_share: float = 0.5,
                       judge_family: str = "judge-fam") -> GoldSet:
    """A gold set drawn from REAL traffic in production [T]; synthetic here, and it says so.

    The corpus is explicit about where gold sets come from [T]: "They mine real production traffic
    instead of inventing toy questions because real users ask things you would never think to test.
    ... A curated 200 examples often beats a random 10,000."

    Three properties are modelled because each is load-bearing for the results:

      * **the better answer is placed in a RANDOM slot.** Without this the gold set's truth is
        ~always slot `a`, a position bias aimed at `a` scores as accuracy, and kappa reads 0.000
        because agreement with a constant is chance. This was a real defect in an earlier version and
        is the reason `is_degenerate()` exists.
      * quality differences are SMALL on average, so most pairs are close calls;
      * half the answers come from the judge's own family, so self-preference has something to act on
        -- and because the SLOT is random, own-family answers appear in both positions, so the bias
        cannot be confused with a position effect.
    """
    rng = random.Random(seed)
    pairs: list[Pair] = []
    lengths: list[int] = []
    for i in range(n_pairs):
        q_hi = rng.uniform(2.5, 5.0)
        q_lo = max(1.0, q_hi - rng.uniform(0.05, margin))
        # length is only weakly related to quality -- which is exactly why rewarding it is a bias
        l_hi = max(20, int(rng.gauss(320, 110)))
        l_lo = max(20, int(rng.gauss(290, 110)))
        fam_hi = judge_family if rng.random() < own_family_share else "other-fam"
        fam_lo = "other-fam" if fam_hi == judge_family else judge_family
        if rng.random() < 0.5:
            a = Answer(f"p{i}-a", q_hi, l_hi, fam_hi)
            b = Answer(f"p{i}-b", q_lo, l_lo, fam_lo)
        else:
            a = Answer(f"p{i}-a", q_lo, l_lo, fam_lo)
            b = Answer(f"p{i}-b", q_hi, l_hi, fam_hi)
        pairs.append(Pair(a, b))
        lengths += [l_hi, l_lo]
    mean = sum(lengths) / len(lengths)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return GoldSet(pairs, mean, math.sqrt(var) or 1.0)


# ----------------------------------------------------------------------------------------------
# The judge
# ----------------------------------------------------------------------------------------------

def judge_score(answer: Answer, p: JudgeParams, gold: GoldSet, *, shown_first: bool,
                correction: Correction, rng: random.Random) -> float:
    """The judge's score for one answer, in latent-quality units.

        score = quality
              + position_bias   x [shown first]
              + verbosity_weight x z(length)      <- zeroed when length is normalized
              + self_pref       x [own family]    <- zeroed when the family differs
              + noise
    """
    s = answer.quality
    if p.position_bias:
        s += p.position_bias if shown_first else 0.0
    if p.verbosity_weight and not correction.length_controlled:
        z = (answer.length - gold.length_mean) / gold.length_sd
        s += p.verbosity_weight * z
    if p.self_pref and not correction.different_family:
        if answer.family == p.family:
            s += p.self_pref
    if p.noise:
        s += rng.gauss(0.0, p.noise)
    return s


def judge_pair(pair: Pair, p: JudgeParams, gold: GoldSet, correction: Correction,
               rng: random.Random) -> tuple[str, float, float]:
    """A PAIRWISE verdict: which answer is better. Returns (verdict, score_a, score_b).

    Ordering is the whole content of `randomize_order`: with it off, `a` is always shown first, so
    the position bias has a fixed sign and becomes a systematic advantage for whichever slot the gold
    set puts the better answer in more often. With it on, the bias still fires per call but no longer
    correlates with the label, so it cancels in the aggregate.

    **The order draw is made in BOTH arms**, and only its USE differs. Skipping the draw when
    randomisation is off would make the two arms consume different positions in the RNG stream, so
    they would see different noise -- and the comparison between them would be confounded by that
    rather than measuring the bias. Drawing unconditionally makes the two runs a paired comparison
    over identical noise.
    """
    draw = rng.random()
    a_first = draw < 0.5 if correction.randomize_order else True
    # the judge model itself: a different family when the correction is on
    effective = JudgeParams(p.name, family="neutral-fam" if correction.different_family else p.family,
                            position_bias=p.position_bias, verbosity_weight=p.verbosity_weight,
                            self_pref=p.self_pref, noise=p.noise)
    sa = judge_score(pair.a, effective, gold, shown_first=a_first, correction=correction, rng=rng)
    sb = judge_score(pair.b, effective, gold, shown_first=not a_first, correction=correction, rng=rng)
    return ("a" if sa >= sb else "b"), sa, sb


def judge_pointwise(pair: Pair, p: JudgeParams, gold: GoldSet, correction: Correction,
                    rng: random.Random) -> tuple[str, float, float]:
    """A POINTWISE verdict: score each answer 1-5 independently, then compare.

    Modelled as pairwise with LARGER noise, because that is the mechanism behind the corpus's
    "absolute scores wander" [T]: on a 1-5 absolute scale the judge has to place an answer against a
    remembered rubric rather than against a visible alternative, so the per-call variance is larger.
    The bias terms are identical -- pointwise does not fix a bias, it only adds noise.
    """
    loud = JudgeParams(p.name, family=p.family, position_bias=0.0,   # no first/second in pointwise
                       verbosity_weight=p.verbosity_weight, self_pref=p.self_pref,
                       noise=p.noise * 2.5)
    sa = judge_score(pair.a, loud, gold, shown_first=True, correction=correction, rng=rng)
    sb = judge_score(pair.b, loud, gold, shown_first=True, correction=correction, rng=rng)
    return ("a" if sa >= sb else "b"), sa, sb


def both_orders_verdict(pair: Pair, p: JudgeParams, gold: GoldSet, correction: Correction,
                        rng: random.Random) -> tuple[str, float, float]:
    """Score the pair in BOTH orders and keep the verdict only if the two agree. [D]

    The corpus prescribes "randomize the answer order" [T]. Randomising removes the SIGN of the
    position bias -- no longer does the same slot always get the bonus -- but it does not remove the
    bias's contribution to VARIANCE: on any single call the bonus is still applied, at random, and on
    a close pair a random bonus of the same size as the margin is a coin flip. The measured
    consequence (experiment 5) is that randomising recovers only a point or two.

    The stronger fix, and the one this blueprint recommends operationally, is to run the pair in both
    orders and treat a disagreement as UNDECIDED. That cancels the bonus exactly rather than in
    expectation, and it turns the position bias from a silent error into a visible tie rate -- which
    is the number an eval owner can actually act on.

    The price is stated plainly: 2x judge calls, and a tie rate that grows with the bias. A tie is
    counted as WRONG by `evaluate`, not discarded, so the tie rate cannot be hidden by dropping ties.
    """
    c = Correction(randomize_order=False,
                   length_controlled=correction.length_controlled,
                   different_family=correction.different_family)
    v1, sa, sb = judge_pair(pair, p, gold, c, rng)
    swapped = Pair(pair.b, pair.a)
    v2, _, _ = judge_pair(swapped, p, gold, c, rng)
    if v2 == "a":
        v2 = "b"
    elif v2 == "b":
        v2 = "a"
    return (v1 if v1 == v2 else "tie"), sa, sb


# ----------------------------------------------------------------------------------------------
# Agreement: the number that validates the eval
# ----------------------------------------------------------------------------------------------

def agreement(verdicts: list[str], truths: list[str]) -> float:
    """Raw agreement between the judge and the human labels."""
    if not verdicts:
        return 0.0
    return sum(1 for v, t in zip(verdicts, truths) if v == t) / len(verdicts)


def cohen_kappa(verdicts: list[str], truths: list[str]) -> float:
    """Agreement corrected for chance.

    Raw agreement flatters a judge on an imbalanced gold set: if 85% of the truth is "a", a judge
    that always says "a" scores 85% agreement and knows nothing. Kappa is the fix, and it is the
    reason the blueprints gate on kappa as well as agreement.
    """
    if not verdicts:
        return 0.0
    n = len(verdicts)
    labels = sorted(set(verdicts) | set(truths))
    po = agreement(verdicts, truths)
    pe = 0.0
    for lab in labels:
        p_j = sum(1 for v in verdicts if v == lab) / n
        p_t = sum(1 for t in truths if t == lab) / n
        pe += p_j * p_t
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


def evaluate(gold: GoldSet, p: JudgeParams, correction: Correction, *, seed: int = 7,
             mode: str = "pairwise", pairs: list[Pair] | None = None) -> dict:
    """Run the judge over the gold set and report agreement, kappa, and the two slice results.

    **The slices are the point.** An overall agreement number hides the fact that a biased judge is
    accurate on the EASY pairs and near-random on the close calls -- and close calls are the only
    ones a release decision actually turns on.
    """
    rng = random.Random(seed)
    use = pairs if pairs is not None else gold.pairs
    fn = {"pairwise": judge_pair, "pointwise": judge_pointwise,
          "both_orders": both_orders_verdict}[mode]
    verdicts, truths = [], []
    for pair in use:
        v, _, _ = fn(pair, p, gold, correction, rng)
        verdicts.append(v)
        truths.append(pair.human_winner)
    close_idx = [i for i, pr in enumerate(use) if pr.margin <= 0.5]
    decided = [(v, t) for v, t in zip(verdicts, truths) if v in ("a", "b")]
    return {
        "mode": mode,
        "n": len(use),
        "agreement": agreement(verdicts, truths),
        "ties": sum(1 for v in verdicts if v == "tie"),
        "tie_rate": sum(1 for v in verdicts if v == "tie") / len(verdicts) if verdicts else 0.0,
        # `agreement` above counts a tie as WRONG, so the tie rate can never be hidden by dropping
        # ties. This second number is the accuracy over the pairs the judge was willing to decide,
        # and it is only meaningful READ NEXT TO the tie rate: a judge that abstains on half the set
        # and is right on most of the rest is not better than one that answers everything.
        "decided_n": len(decided),
        "decided_agreement": agreement([v for v, _ in decided], [t for _, t in decided])
        if decided else 0.0,
        "kappa": cohen_kappa(verdicts, truths),
        "label_balance": sum(1 for t in truths if t == "a") / len(truths) if truths else 0.5,
        "degenerate": abs(sum(1 for t in truths if t == "a") / len(truths) - 0.5) > 0.15
        if truths else False,
        "position_bias": p.position_bias,
        "verbosity_weight": p.verbosity_weight,
        "self_pref": p.self_pref,
        "noise": p.noise,
        "correction": correction,
        "close_n": len(close_idx),
        "close_agreement": agreement([verdicts[i] for i in close_idx],
                                     [truths[i] for i in close_idx]) if close_idx else 0.0,
    }


# ----------------------------------------------------------------------------------------------
# Sweeps used by the experiments
# ----------------------------------------------------------------------------------------------

DEFAULT_JUDGE = JudgeParams("biased-judge", family="judge-fam",
                            position_bias=0.45, verbosity_weight=0.40, self_pref=0.35, noise=0.30)


def correction_ladder(gold: GoldSet, p: JudgeParams = DEFAULT_JUDGE, seed: int = 7) -> list[dict]:
    """Each correction ALONE, then all three -- so the value of each is separable.

    A team that applies "the three fixes" as a bundle cannot tell which one is carrying the result,
    and the three cost very different amounts: randomising order is free, length control changes the
    prompt, and a second judge family is a second vendor and a second bill.
    """
    variants = [
        ("none (naive)",              Correction()),
        ("+ randomize order",         Correction(randomize_order=True)),
        ("+ length normalized",       Correction(length_controlled=True)),
        ("+ different family",        Correction(different_family=True)),
        ("+ all three",               Correction(True, True, True)),
    ]
    rows = []
    for label, corr in variants:
        r = evaluate(gold, p, corr, seed=seed)
        r["variant"] = label
        rows.append(r)
    return rows


def position_bias_sweep(gold: GoldSet, betas: list[float] | None = None,
                        p: JudgeParams | None = None, seed: int = 7) -> list[dict]:
    """Agreement as the position bias grows, under three handling strategies.

    The measured result is NOT the tidy story that "randomising is the fix". All three curves share an
    intercept at beta = 0; as beta grows, fixed order declines fastest, randomisation declines more
    slowly but still declines, and scoring BOTH ORDERS holds up best while accumulating ties.

    Why randomisation does not hold up: it decorrelates the bias from the label, so the SIGNED part
    of the error disappears -- but the bias is still applied, at random, on every single call. For a
    close pair, a random bonus the size of the margin is a coin flip, so randomising trades a
    systematic error for extra variance rather than removing the error. Both-orders cancels the bonus
    exactly instead of in expectation, which is why it is the operationally recommended fix and why
    the tie rate it produces is the honest cost of it.
    """
    betas = betas or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    base = p or DEFAULT_JUDGE
    rows = []
    for b in betas:
        variant = JudgeParams(base.name, base.family, position_bias=b,
                              verbosity_weight=base.verbosity_weight, self_pref=base.self_pref,
                              noise=base.noise)
        off = evaluate(gold, variant, Correction(), seed=seed)
        on = evaluate(gold, variant, Correction(randomize_order=True), seed=seed)
        both = evaluate(gold, variant, Correction(), seed=seed, mode="both_orders")
        rows.append({"beta": b,
                     "agreement_off": off["agreement"], "agreement_on": on["agreement"],
                     "agreement_both": both["agreement"],
                     "decided_both": both["decided_agreement"],
                     "kappa_off": off["kappa"], "kappa_on": on["kappa"],
                     "tie_rate": both["tie_rate"],
                     "recovered": on["agreement"] - off["agreement"],
                     "recovered_both": both["decided_agreement"] - off["agreement"]})
    return rows


def verbosity_effect(gold: GoldSet, p: JudgeParams | None = None, seed: int = 7) -> dict:
    """Does the judge's verdict track LENGTH rather than quality? Measured directly.

    The corpus's claim is "it rewards length even when length adds nothing" [T]. The gold set
    generates length independently of quality, so the TRUE population correlation between the length
    margin and the quality margin is zero -- **but the sample correlation is not exactly zero, and it
    is computed here rather than asserted**, because it is the baseline the biased judge's r must be
    compared against. A judge whose r matches the sample's r is not exhibiting bias at all; it is
    just reading length out of the quality signal that happens to be in the sample.
    """
    p = p or DEFAULT_JUDGE
    rng = random.Random(seed)
    xs, ys, qs = [], [], []
    for pair in gold.pairs:
        _, sa, sb = judge_pair(pair, p, gold, Correction(), rng)
        xs.append(pair.a.length - pair.b.length)
        ys.append(sa - sb)
        qs.append(pair.a.quality - pair.b.quality)
    r_biased = _pearson(xs, ys)
    r_truth = _pearson(xs, qs)

    rng2 = random.Random(seed)
    xs2, ys2 = [], []
    corr = Correction(length_controlled=True)
    for pair in gold.pairs:
        _, sa, sb = judge_pair(pair, p, gold, corr, rng2)
        xs2.append(pair.a.length - pair.b.length)
        ys2.append(sa - sb)
    r_fixed = _pearson(xs2, ys2)
    return {"r_length_score_biased": r_biased, "r_length_score_corrected": r_fixed,
            "r_length_quality_true": r_truth, "n": len(gold.pairs),
            "note": "length is generated independently of quality, so r_length_quality_true is the "
                    "sample's incidental correlation and is the baseline the biased judge's r is "
                    "compared against"}


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r. Stdlib, no numpy, and returns 0.0 rather than NaN for a constant input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else 0.0
