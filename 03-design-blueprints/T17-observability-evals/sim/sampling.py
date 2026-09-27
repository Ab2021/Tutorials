"""Trace sampling -- the cost/coverage frontier, and the blind spot in head sampling.

The corpus's rule [T]:

    "In development, trace everything. The detail is worth it. In high-traffic production, sample
     the successes, but always keep 100% of the errors. And redact personal data at the boundary
     before it is ever written, so your trace store never becomes a privacy problem of its own."
                                                    -- LLM Observability: Traces, Spans, OTel

That is four rules in two sentences, and this module quantifies all four:

  * **head sampling** (a uniform rate) is cheap and has a blind spot proportional to the incident's
    RARITY. A 1-in-1000 failure sampled at 5% is very likely to produce no trace at all in the
    window you needed it. This is the quantitative reason "always keep 100% of the errors" exists.
  * **tail sampling** keeps the decision until the request has finished, so it can select on
    outcome (status, latency) rather than on a coin flip. It costs a buffer and a decision point.
  * **the cost** is what you pay for coverage, and the frontier is not linear: capturing the last
    few percent of errors can cost more than the first 95%.
  * **redaction is not a sampling question** -- it happens at the boundary, before storage, at any
    rate. A 1% sample of unredacted prompts is still a privacy incident.

Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    """A sampling policy. Head policies use `baseline`; tail policies add outcome predicates."""

    name: str
    baseline: float = 0.0            # uniform keep-rate for ordinary requests
    error_rate: float = 0.0          # keep-rate for requests that errored
    slow_rate: float = 0.0           # keep-rate for requests slower than `slow_ms`
    slow_ms: float = 10_000.0        # the corpus's collector example uses 10 s [R]
    tail: bool = False               # is the decision made after the request completed?
    buffer_bytes: int = 0            # tail sampling must hold spans until the decision

    def describe(self) -> str:
        if not self.tail:
            return f"head: {self.baseline:.0%} uniform"
        return (f"tail: err {self.error_rate:.0%} / slow(>{self.slow_ms / 1000:.0f}s) "
                f"{self.slow_rate:.0%} / base {self.baseline:.0%}")


# The corpus's collector example, verbatim [R] (cheat sheet / OTel collector config):
#   errors -> 100%, latency > 10 s -> 100%, baseline -> probabilistic 5%
DEFAULT_POLICIES: list[Policy] = [
    Policy("trace_all",           baseline=1.00, error_rate=1.00, slow_rate=1.00),
    Policy("uniform_5pct",        baseline=0.05, error_rate=0.05, slow_rate=0.05),
    Policy("uniform_1pct",        baseline=0.01, error_rate=0.01, slow_rate=0.01),
    Policy("errors_only",         baseline=0.00, error_rate=1.00, slow_rate=0.00),
    Policy("tail_sample",         baseline=0.05, error_rate=1.00, slow_rate=1.00, tail=True,
           buffer_bytes=4096),
]


def expected_kept(policy: Policy, n_requests: int, error_rate: float, slow_rate: float) -> dict:
    """Spans kept, broken out by class, for a workload with these class rates.

    Returns the counts AND the per-class keep-rates, because the finding is in the second column:
    the *overall* keep-rate of a tail policy looks like head sampling and covers a completely
    different set of requests.
    """
    n_err = n_requests * error_rate
    n_slow = n_requests * slow_rate * (1.0 - error_rate)   # slow but not errored
    n_ok = n_requests - n_err - n_slow
    kept_err = n_err * policy.error_rate
    kept_slow = n_slow * (policy.slow_rate if policy.tail else policy.baseline)
    kept_ok = n_ok * policy.baseline
    kept = kept_err + kept_slow + kept_ok
    return {
        "kept": kept, "kept_pct": kept / n_requests * 100.0 if n_requests else 0.0,
        "kept_errors": kept_err,
        "error_coverage": kept_err / n_err if n_err else 1.0,
        "kept_slow": kept_slow,
        "slow_coverage": kept_slow / n_slow if n_slow else 1.0,
        "kept_ok": kept_ok,
    }


def storage_cost_gb(kept_spans: float, span_bytes: int = 4096) -> float:
    """Trace storage. The corpus warns about the failure mode directly [T]:
    "Trace every token at full volume, and the signal disappears under the noise.\""""
    return kept_spans * span_bytes / 1e9


def p_zero_captured(incident_rate: float, keep_rate: float, window_requests: int) -> float:
    """P(no trace at all for an incident class), for an incident of this rarity, in this window.

        p_capture = incident_rate * keep_rate      per request
        P(zero)   = (1 - p_capture) ** window_requests

    **This is the argument for tail sampling, and it is not about cost.** A head policy's blind
    spot is not "we have fewer traces"; it is "for a rare failure we have NO trace, exactly when we
    need one". The function is monotone in rarity, so the failures that matter most are the ones
    most likely to be invisible.
    """
    if not 0.0 <= keep_rate <= 1.0:
        raise ValueError("keep_rate must be in [0, 1]")
    if not 0.0 <= incident_rate <= 1.0:
        raise ValueError("incident_rate must be in [0, 1]")
    if window_requests < 0:
        raise ValueError("window_requests must be >= 0")
    p_capture = incident_rate * keep_rate
    if p_capture <= 0.0:
        return 1.0
    if p_capture >= 1.0:
        return 0.0
    return (1.0 - p_capture) ** window_requests


def requests_until_first_trace(incident_rate: float, keep_rate: float, confidence: float = 0.95) -> float:
    """Expected requests before you are `confidence` sure of having captured ONE trace.

    Reported in requests and in hours at a given rate, because "we will catch it eventually" is the
    reasoning that produces a six-week blind spot on a 1-in-10,000 failure.
    """
    p_capture = incident_rate * keep_rate
    if p_capture <= 0.0:
        return math.inf
    return math.log(1.0 - confidence) / math.log(1.0 - p_capture)


def frontier(n_requests: int, error_rate: float, slow_rate: float,
             policies: list[Policy] | None = None) -> list[dict]:
    """Every policy's cost AND both coverages, side by side. The frontier, not a recommendation.

    The reader is meant to see that `uniform_5pct` and `tail_sample` have nearly the same *overall*
    keep rate (a number a cost dashboard shows) and wildly different error coverage (the number
    that decides whether you can debug an incident).
    """
    policies = policies or DEFAULT_POLICIES
    rows = []
    for p in policies:
        r = expected_kept(p, n_requests, error_rate, slow_rate)
        r.update({"policy": p.name, "describe": p.describe(), "tail": p.tail,
                  "storage_gb": storage_cost_gb(r["kept"]), "buffer_mb": p.buffer_bytes / 1e6})
        rows.append(r)
    return rows


def redaction_is_orthogonal(policy: Policy) -> dict:
    """Sampling does not make a trace store safe, and this states it as a fact rather than advice.

    A 1% sample of unredacted prompts is still an incident -- just a smaller one. The corpus says
    this in the same breath as the sampling rule [T]: "redact personal data at the boundary before
    it is ever written".

    Returned as a struct so a config validator can refuse to load a non-redacting collector
    regardless of its sampling rate.
    """
    return {"sampling_rate": policy.baseline,
            "pii_risk_reduced_by_sampling": False,
            "requires_boundary_redaction": True,
            "note": "redaction happens before storage, at every sampling rate, including 1%"}
