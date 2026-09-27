"""T19 -- sovereignty: not a binary, not a scalar, and not four independent purchases.

The sovereignty talk's framing is the one to carry: "think about sovereignty as something like
binary -- it's a sovereign setup or non-sovereign setup. I think that's wrong... you need to start
looking at it as a system property, and this particular system property has various dimensions" [T].
The four dimensions are **control, trust, economics, continuity**.

This module takes that seriously in two ways the talk does not spell out:

  * **The dimensions have prerequisites.** Trust requires control -- you cannot cryptographically
    attest an environment whose model and accelerator you did not choose. Continuity requires
    control for the same reason. A posture can therefore score 4/4 on paper and 2/4 in effect.

  * **Continuity is a MINIMUM, not a mean.** Having two accelerators, two clouds and two model
    families but one serving engine is a single-vendor system, because the weakest layer is the
    binding constraint. The mean would call that 0.87; the min calls it zero. This is the same
    shape as T18's session trust -- a control that reads as an average and behaves as a floor.

And one thing that is arithmetic rather than judgement: **a residency rule has no error tolerance.**
It is a legal property, not a quality score, so 99.99% correct residency is 100% non-compliant.
That is why it must be a filter and never a score -- the argument T14 makes about routing is a
compliance requirement here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DIMENSIONS = ("control", "trust", "economics", "continuity")

DIMENSION_QUESTION = {
    "control":    "Where does inference run, on which model, on which accelerator?",
    "trust":      "What code and artifacts are actually executing?",
    "economics":  "Who controls the token cost?",
    "continuity": "Can you operate without a single vendor?",
}

# A dimension cannot exceed its prerequisite. Stated as data so it can be argued with rather than
# buried in a scoring function.
PREREQ = {
    "trust": "control",       # you cannot attest an environment you did not choose
    "continuity": "control",  # you cannot switch vendors if the vendor picks the model
}


@dataclass(frozen=True)
class Posture:
    name: str
    scores: dict          # dimension -> 0..1, as CLAIMED by whoever sells or owns the posture
    cost_index: float     # relative unit cost, 1.00 = a managed API at list price
    note: str


POSTURES = (
    Posture("compliance_only",
            {"control": 0.70, "trust": 0.10, "economics": 0.40, "continuity": 0.55},
            1.00,
            "contractual commitments, DPAs, a regional endpoint. The answer the Token Raj "
            "speaker dismisses: 'you can't legislate a memory dump' [T]."),
    Posture("regional_attested",
            {"control": 0.85, "trust": 0.90, "economics": 0.75, "continuity": 0.80},
            1.10,
            "in-region deployment plus TEEs with cryptographic attestation [T]. Costs a measured "
            "throughput penalty, not a modelled one."),
    Posture("own_accelerators",
            {"control": 0.95, "trust": 0.85, "economics": 0.85, "continuity": 0.90},
            1.35,
            "buy the silicon and run the fleet. Full control of the bottom layers at the price "
            "of an annual procurement cycle and no elasticity."),
    Posture("own_stack",
            {"control": 1.00, "trust": 0.95, "economics": 0.95, "continuity": 1.00},
            1.60,
            "silicon, models, weights, serving. Viable for a handful of organisations globally."),
    Posture("api_with_dpa",
            {"control": 0.55, "trust": 0.05, "economics": 0.30, "continuity": 0.25},
            0.85,
            "a frontier API, a signed agreement and a chosen region. The cheapest posture and "
            "the one with the widest gap between what it claims and what it can prove."),
    Posture("vendor_tee_lease",
            {"control": 0.50, "trust": 0.85, "economics": 0.35, "continuity": 0.30},
            1.05,
            "a vendor's confidential-computing tenant: the SILICON and the model are not yours, "
            "so the attestation proves a property of an environment you did not choose. This is "
            "the posture the prerequisite rule exists for -- it claims 0.85 trust and can "
            "evidence 0.50."),
)

POSTURE_BY_NAME = {p.name: p for p in POSTURES}


def effective(posture: Posture) -> dict:
    """Each dimension capped by its prerequisite. The gap is what a review must explain."""
    out = {}
    for d in DIMENSIONS:
        raw = posture.scores[d]
        pre = PREREQ.get(d)
        cap = posture.scores[pre] if pre else 1.0
        out[d] = min(raw, cap)
    return out


def claimed_vs_effective(posture: Posture) -> dict:
    eff = effective(posture)
    return {
        "posture": posture.name,
        "claimed": dict(posture.scores),
        "effective": eff,
        "capped": [d for d in DIMENSIONS if eff[d] < posture.scores[d]],
        "claimed_mean": sum(posture.scores.values()) / len(DIMENSIONS),
        "effective_mean": sum(eff.values()) / len(DIMENSIONS),
        "weakest_link": min(eff, key=lambda d: eff[d]),
    }


def posture_table() -> list:
    return [claimed_vs_effective(p) for p in POSTURES]


def mean_vs_min(posture: Posture) -> dict:
    """The two readings of the four dimensions, for every posture.

    A mean says a system with three strong dimensions is strong. A min says it is only as strong
    as its weakest one -- and only the second survives contact with an adversary or a regulator
    who gets to choose which dimension to attack.
    """
    eff = effective(posture)
    return {"posture": posture.name,
            "mean": sum(eff.values()) / len(DIMENSIONS),
            "min": min(eff.values()),
            "gap": sum(eff.values()) / len(DIMENSIONS) - min(eff.values())}


# --------------------------------------------------------------------------------------------
# The attestation arithmetic. "It's a trade-off. It's not going to run as fast as what it was
# running before. But it is going to run more securely" [T] -- and an unquantified trade-off is a
# trade-off nobody can approve.
# --------------------------------------------------------------------------------------------
def tee_uplift(throughput_penalty: float) -> float:
    """Cost uplift per token for a given throughput penalty.

    A 20% throughput penalty is a 25% cost uplift, not a 20% one: you need 1/(1-0.20) units of
    capacity to do the same work. Getting this wrong in the direction of the penalty undersizes
    every TEE budget by about a fifth.
    """
    if not 0.0 <= throughput_penalty < 1.0:
        raise ValueError("a throughput penalty must be in [0, 1)")
    return 1.0 / (1.0 - throughput_penalty) - 1.0


def attested_blended_uplift(throughput_penalty: float, attested_share: float) -> float:
    """The uplift on the TOTAL bill, given only part of the traffic must be attested."""
    return attested_share * tee_uplift(throughput_penalty)


def tee_budget_table(penalties=(0.05, 0.10, 0.20, 0.30, 0.40), shares=(0.10, 0.40, 1.00)) -> list:
    return [{"penalty": p, "attested_share": s,
             "uplift_per_token": tee_uplift(p),
             "blended_uplift": attested_blended_uplift(p, s)}
            for p in penalties for s in shares]


def attestation_breakeven_cost(uplift: float, current_bill: float) -> dict:
    """The money the sovereign posture costs, to sit next to the exposure it removes.

    The point is not the number. It is that the alternative is not 'free' -- it is an
    unquantified legal exposure, and a risk owner cannot approve an unquantified one.
    """
    return {"uplift": uplift, "current_bill": current_bill,
            "added_cost": current_bill * uplift,
            "new_bill": current_bill * (1.0 + uplift)}


# --------------------------------------------------------------------------------------------
# Residency: a filter, never a score.
# --------------------------------------------------------------------------------------------
def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def residency_score_leak(gap: float, noise: float) -> float:
    """The probability a SCORE-based residency router picks the wrong region.

    The in-region option must win by a margin `gap` against a scoring noise of `noise`. A filter
    makes this number exactly zero, because it removes the out-of-region option rather than
    discounting it. Any positive value here is the whole argument.
    """
    if noise <= 0.0:
        return 0.0 if gap > 0 else 0.5
    return 1.0 - normal_cdf(gap / noise)


def residency_leaks(gap: float, noise: float, requests: int) -> dict:
    rate = residency_score_leak(gap, noise)
    return {"gap": gap, "noise": noise, "requests": requests,
            "leak_rate": rate, "leaks": rate * requests, "filter_leaks": 0.0}


def residency_compliance(leak_rate: float) -> dict:
    """Residency has no tolerance. A 99.99% correct router is 100% non-compliant."""
    return {"leak_rate": leak_rate, "correct_rate": 1.0 - leak_rate,
            "compliant_as_filter": leak_rate == 0.0,
            "verdict": ("compliant" if leak_rate == 0.0
                        else "NON-COMPLIANT at any non-zero rate -- this is a legal property, "
                             "not a quality score")}


# --------------------------------------------------------------------------------------------
# Continuity: a minimum over layers, not a mean.
# --------------------------------------------------------------------------------------------
CONTINUITY_LAYERS = ("accelerator", "model_family", "serving_engine", "cloud_region")

CONTINUITY_NOTE = {
    "accelerator":   "'the choice of accelerator is extremely important... it should be part of "
                     "the implementation strategy' [T]",
    "model_family":  "'can you operate without a single vendor?' [T]",
    "serving_engine": "vLLM-class vs a vendor's runtime -- see T13",
    "cloud_region":  "'today you have a dependency on something and tomorrow your country says "
                     "you cannot use something from the other country' [T]",
}


def continuity(sources: dict) -> dict:
    """`sources` maps a layer to the number of INDEPENDENT options available at it.

    The reading returned is the minimum, because a chain is only as strong as its weakest layer
    and a regulator or a vendor gets to choose which layer to fail. The mean is returned beside it
    because the mean is what a slide will show.
    """
    per_layer = {k: (1.0 if sources.get(k, 0) >= 2 else 0.0) for k in CONTINUITY_LAYERS}
    floor = min(per_layer.values())
    mean = sum(per_layer.values()) / len(CONTINUITY_LAYERS)
    return {"sources": dict(sources), "per_layer": per_layer,
            "continuity": floor, "mean": mean, "gap": mean - floor,
            "binding_layer": min(per_layer, key=lambda k: per_layer[k]),
            "single_vendor": floor == 0.0}


def continuity_examples() -> list:
    return [
        continuity({"accelerator": 2, "model_family": 2, "serving_engine": 2, "cloud_region": 2}),
        continuity({"accelerator": 2, "model_family": 3, "serving_engine": 2, "cloud_region": 1}),
        continuity({"accelerator": 2, "model_family": 2, "serving_engine": 1, "cloud_region": 3}),
        continuity({"accelerator": 1, "model_family": 4, "serving_engine": 3, "cloud_region": 3}),
    ]


# --------------------------------------------------------------------------------------------
# The trust problem, and who can actually prove what.
# --------------------------------------------------------------------------------------------
THREE_WAY_TRUST = (
    ("model_owner", "silicon_owner",
     "putting weights on someone else's infrastructure exposes 'years of training and the millions "
     "of dollars they are spending' [T]"),
    ("infrastructure_provider", "model_owner",
     "a model may ship 'malicious code along with the model weights' -- 'the first time Qwen came "
     "out, they installed it, it was trying to do a netconnection back to the servers' [T]"),
    ("consumer", "both",
     "the model provider trains on their data, or the infrastructure provider builds a competing "
     "service from it"),
)


def trust_gap_closers() -> list:
    """What each party can do, and whether it is legal or cryptographic.

    The talk's own summary of the state of the art: 'right now the only frameworks which are
    protecting sovereignty are the legal frameworks... how can you prove cryptographically that
    none of the data stored in your CPUs and GPUs can be stolen? You need a cryptographic
    attestation' [T].
    """
    return [
        {"party": "consumer", "closer": "data-processing agreement", "kind": "legal",
         "proves": "an intention, not a state"},
        {"party": "consumer", "closer": "regional endpoint", "kind": "contractual",
         "proves": "where the request was routed, not what ran"},
        {"party": "consumer", "closer": "TEE + attestation", "kind": "cryptographic",
         "proves": "'only you as the data owner can see that in plain text, but you can attest "
                   "that' [T]"},
        {"party": "model_owner", "closer": "confidential computing on the lease",
         "kind": "cryptographic", "proves": "weights are not readable by the host"},
        {"party": "infrastructure_provider", "closer": "Kata containers, network isolation",
         "kind": "isolation", "proves": "'even if there's malicious code it is contained to that "
                                        "particular layer' [T]"},
        {"party": "infrastructure_provider", "closer": "model signature verification",
         "kind": "cryptographic", "proves": "'is the model I am running the model I evaluated'"},
    ]


def open_weights_spectrum() -> list:
    """'Open weights' is a position on a spectrum, not a checkbox.

    'When people say open source model they'll just publish open weights. They don't show the
    source code which was used to train, and they don't showcase the kind of silicon they used to
    train that, so that you can replicate it and reproduce it' [T].
    """
    return [
        {"name": "closed API", "weights": False, "training_code": False,
         "training_data": False, "reproducible": False},
        {"name": "open weights (the common case)", "weights": True, "training_code": False,
         "training_data": False, "reproducible": False},
        {"name": "open weights + training code", "weights": True, "training_code": True,
         "training_data": False, "reproducible": False},
        {"name": "open source (rare)", "weights": True, "training_code": True,
         "training_data": True, "reproducible": True},
    ]


def spectrum_position(name: str) -> dict:
    for r in open_weights_spectrum():
        if r["name"] == name:
            score = sum(1 for k in ("weights", "training_code", "training_data", "reproducible")
                        if r[k]) / 4.0
            return {**r, "openness": score,
                    "claimable_as_open_source": r["reproducible"]}
    raise KeyError(name)
