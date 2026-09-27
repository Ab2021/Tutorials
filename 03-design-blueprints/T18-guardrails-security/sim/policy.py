"""T18 -- policy: the false-refusal trade, which is a number with a cost on both sides.

The corpus states the trade as a warning [T]: *"Tune the rails too tight and you frustrate real users
with false refusals, which is its own kind of failure"* -- and then, one line later, the sentence that
turns it into arithmetic: *"a system that blocks everything is safe and useless."*

A warning cannot be tuned. This module makes the trade computable, and the computation produces three
results that a prose treatment cannot:

  1. **The cost-optimal threshold is not where intuition puts it.** It is where the marginal leak
     cost equals the marginal false-refusal cost, and that point moves by *orders of magnitude*
     between a regulated payer and a marketing bot with the same detector. There is no correct
     threshold; there is a correct threshold *given an L/F ratio you have to name out loud.*
  2. **For a regulated payer the optimum is operationally absurd** -- the arithmetic says block
     almost everything, at a false-refusal rate no product team would ship. That is not a bug in the
     arithmetic. It is proof that **one threshold is the wrong architecture**, and the fix is a
     cascade: a cheap high-recall filter followed by an expensive precise check.
  3. **A single global threshold is not uniformly harsh.** Legitimate traffic is not one
     distribution, so a threshold tuned to a 5% global false-refusal rate produces a much higher rate
     on the segment of users whose phrasing scores highest. The corpus's "balance safety against
     false refusals" [T] is an average over a distribution that is not uniform.

Scores are modelled as two unit-variance Gaussians. Nothing here is a benchmark -- the separation is
a parameter, and the point is how the *decision* moves with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Detector:
    """One rail described as a score distribution rather than as a catch rate.

    This is the honest description. A rail does not have "a 90% catch rate" -- it has a threshold,
    and a catch rate that is whatever the threshold happens to buy on the traffic it sees.
    """
    name: str
    mu_attack: float
    mu_benign: float = 0.0
    cost_units: float = 1.0
    note: str = ""

    @property
    def auc(self) -> float:
        """Area under the ROC -- the threshold-independent quality of the detector."""
        return normal_cdf((self.mu_attack - self.mu_benign) / math.sqrt(2.0))

    def catch(self, theta: float) -> float:
        return 1.0 - normal_cdf(theta - self.mu_attack)

    def fpr(self, theta: float) -> float:
        return 1.0 - normal_cdf(theta - self.mu_benign)


# Two detectors at very different price points. The cheap one is what you run on every chunk; the
# precise one is what you run when the cheap one is suspicious. The corpus's "two-pass intent
# extraction then act" [R] is this shape; it is presented there as a security technique and it is
# really a cost-architecture technique.
CHEAP_FILTER = Detector("cheap_filter", mu_attack=1.2, cost_units=1.0,
                        note="pattern + small classifier; runs on everything")
PRECISE_CHECK = Detector("precise_check", mu_attack=2.6, cost_units=20.0,
                         note="large classifier or two-pass intent extraction; runs on the flagged subset")


@dataclass(frozen=True)
class Scenario:
    name: str
    p_attack: float          # share of requests carrying an attack attempt
    leak_cost: float         # cost of one leak or one successful harmful action
    refusal_cost: float      # cost of one false refusal (support + churn increment)
    note: str = ""

    @property
    def ratio(self) -> float:
        """How many false refusals you should be willing to accept per leak avoided."""
        return self.leak_cost * self.p_attack / (self.refusal_cost * (1.0 - self.p_attack))


SCENARIOS = (
    Scenario("health_payer", p_attack=0.02, leak_cost=50_000.0, refusal_cost=50.0,
             note="regulated; the leak is an incident with notification and regulatory exposure"),
    Scenario("marketing_bot", p_attack=0.02, leak_cost=500.0, refusal_cost=40.0,
             note="brand risk on one side, conversion loss on the other"),
    Scenario("internal_tool", p_attack=0.005, leak_cost=2_000.0, refusal_cost=5.0,
             note="low stakes, few users; the false refusal is an annoyed colleague"),
)


def expected_cost(det: Detector, theta: float, sc: Scenario) -> dict:
    """Expected cost per request at a given threshold. Both terms are real money."""
    c = det.catch(theta)
    f = det.fpr(theta)
    residual_leak = (1.0 - c) * sc.p_attack * sc.leak_cost
    refusals = f * (1.0 - sc.p_attack) * sc.refusal_cost
    return {"theta": theta, "catch": c, "fpr": f,
            "leak_cost": residual_leak, "refusal_cost": refusals,
            "total": residual_leak + refusals}


def cost_curve(det: Detector, sc: Scenario, lo: float = -6.0, hi: float = 6.0, n: int = 121) -> list:
    step = (hi - lo) / (n - 1)
    return [expected_cost(det, lo + i * step, sc) for i in range(n)]


def optimal_threshold(det: Detector, sc: Scenario, lo: float = -6.0, hi: float = 6.0,
                      n: int = 241) -> dict:
    curve = cost_curve(det, sc, lo, hi, n)
    best = min(curve, key=lambda r: r["total"])
    allow_all = expected_cost(det, hi, sc)       # theta = +inf: never flag
    block_all = expected_cost(det, lo, sc)       # theta = -inf: always flag
    return {
        "scenario": sc.name, "detector": det.name,
        "optimum": best, "allow_all": allow_all, "block_all": block_all,
        "block_all_is_worse_than_allow_all": block_all["total"] > allow_all["total"],
        "saving_vs_allow_all": allow_all["total"] - best["total"],
        "saving_vs_block_all": block_all["total"] - best["total"],
        "ratio": sc.ratio,
        "closed_form_theta": closed_form_theta(det, sc),
    }


def closed_form_theta(det: Detector, sc: Scenario) -> float:
    """Where marginal leak cost equals marginal false-refusal cost.

    d/dtheta [ (1-c)pL + f(1-p)F ] = 0  ->  f_attack(theta) / f_benign(theta)
                                         = (1-p)F / (pL)
    For unit-variance Gaussians the left side is exp(mu_a*theta - mu_a^2/2) with mu_b = 0, so the
    balance point has a closed form. The experiments quote it as a check on the scan.
    """
    mu = det.mu_attack - det.mu_benign
    target = (1.0 - sc.p_attack) * sc.refusal_cost / (sc.p_attack * sc.leak_cost)
    if target <= 0.0 or target >= 1.0:
        return float("nan")
    return (math.log(target) + mu * mu / 2.0) / mu if mu else float("nan")


# ------------------------------------------------------------------------------------------------
# The cascade: why one threshold is the wrong architecture
# ------------------------------------------------------------------------------------------------

def cascade(cheap: Detector, precise: Detector, theta_cheap: float, theta_precise: float,
            sc: Scenario) -> dict:
    """Cheap high-recall filter first, expensive precise check on what it flags.

    Catch: an attack is caught if either stage flags it.
    False positives: BOTH stages must flag a benign request for it to be refused.
    Cost: the cheap detector on everything, the precise one on the flagged share.

    That is the entire reason a cascade dominates a single threshold: it multiplies the two
    detectors' *independent errors* the same way the rail stack does, while paying the expensive
    detector only on the fraction that needs it.
    """
    c = 1.0 - (1.0 - cheap.catch(theta_cheap)) * (1.0 - precise.catch(theta_precise))
    f = cheap.fpr(theta_cheap) * precise.fpr(theta_precise)
    run_share = cheap.fpr(theta_cheap) * (1.0 - sc.p_attack) + cheap.catch(theta_cheap) * sc.p_attack
    cost = cheap.cost_units + run_share * precise.cost_units
    leak = (1.0 - c) * sc.p_attack * sc.leak_cost
    refusals = f * (1.0 - sc.p_attack) * sc.refusal_cost
    return {"catch": c, "fpr": f, "precise_run_share": run_share, "detector_cost": cost,
            "leak_cost": leak, "refusal_cost": refusals, "total": leak + refusals + cost}


def find_operating_point(det: Detector, sc: Scenario, target_fpr: float) -> float:
    """The threshold that buys a given false-refusal rate -- the constraint a product team sets."""
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if det.fpr(mid) > target_fpr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cascade_vs_single(cheap: Detector = CHEAP_FILTER, precise: Detector = PRECISE_CHECK,
                      sc: Scenario = SCENARIOS[0], target_fpr: float = 0.02) -> dict:
    """Both architectures at the SAME false-refusal budget. The comparison is only fair that way."""
    t_cheap = find_operating_point(cheap, sc, target_fpr)
    single = expected_cost(cheap, t_cheap, sc)
    best_cascade = None
    for i in range(1, 60):
        t_precise = -3.0 + i * 0.1
        r = cascade(cheap, precise, t_cheap, t_precise, sc)
        if best_cascade is None or r["total"] < best_cascade["total"]:
            best_cascade = dict(r, theta_precise=t_precise)
    # A single-stage precise detector, priced the same way, for the third column.
    t_precise_only = find_operating_point(precise, sc, target_fpr)
    precise_only = expected_cost(precise, t_precise_only, sc)
    precise_only = dict(precise_only, detector_cost=precise.cost_units)
    precise_only["total"] += precise.cost_units
    return {
        "scenario": sc.name, "target_fpr": target_fpr,
        "cheap_theta": t_cheap, "single": dict(single, detector_cost=cheap.cost_units,
                                               total=single["total"] + cheap.cost_units),
        "precise_only": precise_only,
        "cascade": best_cascade,
        "catch_at_fixed_fpr": {"single": single["catch"],
                               "precise_only": precise_only["catch"],
                               "cascade": best_cascade["catch"]},
    }


# ------------------------------------------------------------------------------------------------
# The segment skew: one global threshold is not uniformly harsh
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    name: str
    share: float
    mu_offset: float     # how much higher this segment's LEGITIMATE scores sit
    note: str = ""


# Legitimate traffic scores higher for some users than others, for reasons that have nothing to do
# with safety: terseness, domain jargon, dialect, second-language phrasing. A classifier trained on
# an average is trained on the majority segment.
SEGMENTS = (
    Segment("majority", 0.70, 0.0, "the distribution the detector was tuned on"),
    Segment("terse", 0.10, 0.30, "short requests; few tokens to judge by"),
    Segment("domain_jargon", 0.12, 0.40, "plan and billing terminology the classifier reads as odd"),
    Segment("dialect_second_language", 0.08, 0.60, "phrasing that sits far from the training mean"),
)


def segment_fpr(det: Detector, theta: float, segments=SEGMENTS) -> list:
    rows = []
    for s in segments:
        fpr = 1.0 - normal_cdf(theta - (det.mu_benign + s.mu_offset))
        rows.append({"segment": s.name, "share": s.share, "fpr": fpr, "note": s.note})
    overall = sum(r["fpr"] * r["share"] for r in rows)
    worst = max(rows, key=lambda r: r["fpr"])
    return [dict(r, overall=overall, multiple_of_overall=r["fpr"] / overall,
                 worst=r["segment"] == worst["segment"]) for r in rows]
