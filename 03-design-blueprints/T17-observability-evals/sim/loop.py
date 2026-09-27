"""The feedback loop -- the only thing that makes tracing worth its storage bill.

The corpus states the whole design in one paragraph [T]:

    "The non-obvious part is that tracing only pays off when it is connected to something. On their
     own, traces are logs nobody opens. The magic is the feedback loop. Score live traces with a
     judge, alert when the score drops, and sample the failures into a dataset. That closes the
     circle from the evaluation video. Yesterday's production failure becomes today's test case, and
     your eval set gets stronger every week on its own."
                                                    -- LLM Observability: Traces, Spans, OTel

and names its failure mode just as plainly [T]:

    "Worst of all is the dashboard nobody owns. If no alert fires and no eval reads the traces, you
     have paid for storage, not for insight."

This module makes that claim quantitative. Two deployments run the same number of releases against
the same stream of production failures and differ in ONE respect: whether an observed failure is
written back into the eval set as a test case.

  * **open loop** (traces -> storage): every occurrence of every failure class escapes.
  * **closed loop** (traces -> judge -> dataset -> gate): the FIRST occurrence of a class creates a
    test case, and every later occurrence is caught by the gate before release.

The result is not "the closed loop is better" -- it is that the two diverge *super-linearly in the
number of distinct failure classes*, because the closed loop converts each class from a recurring
liability into a permanent asset. That is what "gets stronger every week on its own" means
mechanically.

Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class FailureClass:
    """A recurring way the system fails. The corpus's word for it is "a notable production failure".

    `rate` is its occurrence rate per release. Real failure classes are heavy-tailed: a few common
    ones and a long tail of rare, expensive ones [D]. `severity` is cost per escaped occurrence, in
    arbitrary units, so the escaped-defect count can be reported weighted as well as raw.
    """

    cid: str
    rate: float
    severity: float = 1.0
    first_seen_release: int | None = None
    covered: bool = False       # does the eval set now contain a test case for it?


def failure_taxonomy(n: int = 40, seed: int = 23, rate_alpha: float = 1.4,
                     mean_rate: float = 0.02) -> list[FailureClass]:
    """A heavy-tailed taxonomy. Zipf-ish rates: a few classes at a few percent, a long rare tail.

    The tail matters more than the head for this experiment. A rare class is exactly the one a
    closed loop rescues and an open loop never notices, because it is too infrequent to be caught by
    chance and too costly to ignore once it fires.
    """
    rng = random.Random(seed)
    classes = []
    for i in range(n):
        r = mean_rate / (i + 1) ** rate_alpha * rng.uniform(0.6, 1.6)
        classes.append(FailureClass(f"FC{i:02d}", rate=r,
                                    severity=1.0 + 9.0 * (i / max(1, n - 1))))  # rarer = costlier
    return classes


@dataclass
class LoopRun:
    """One deployment's trajectory over `releases` releases."""

    mode: str                       # "open" | "closed"
    per_release: list[dict] = field(default_factory=list)

    @property
    def escaped_total(self) -> float:
        return sum(r["escaped"] for r in self.per_release)

    @property
    def severity_weighted_total(self) -> float:
        return sum(r["escaped_weighted"] for r in self.per_release)

    @property
    def final_coverage(self) -> float:
        return self.per_release[-1]["coverage"] if self.per_release else 0.0

    @property
    def final_eval_size(self) -> int:
        return self.per_release[-1]["eval_size"] if self.per_release else 0

    def escaped_by_class(self) -> dict[str, float]:
        acc: dict[str, float] = {}
        for r in self.per_release:
            for cid, n in r["escaped_per_class"].items():
                acc[cid] = acc.get(cid, 0.0) + n
        return acc


def run_loop(classes: list[FailureClass], releases: int, *, closed: bool,
             sample_rate: float = 0.05, requests_per_release: int = 100_000,
             judge_sensitivity: float = 0.75, base_eval_size: int = 60,
             seed: int = 31) -> LoopRun:
    """Run the loop. `closed` is the only difference between the two arms.

    Three parameters carry the honesty of the model, and each is stated rather than hidden:

      * `sample_rate` -- the fraction of production traffic that is traced AND judged. In the closed
        loop this is what determines how quickly a failure class is *noticed*; it does not change
        whether it is eventually noticed, only how many occurrences escape first.
      * `judge_sensitivity` -- P(the judge flags one occurrence of a class it is capable of seeing).
        A judge that cannot see a failure class at all contributes nothing, which is why the corpus
        insists on measuring judge agreement rather than assuming it.
      * `requests_per_release` -- sets the absolute counts. The SHAPE of the result does not depend
        on it; the magnitudes do.
    """
    rng = random.Random(seed)
    run = LoopRun("closed" if closed else "open")
    eval_size = base_eval_size
    covered = 0
    for rel in range(1, releases + 1):
        escaped = 0.0
        escaped_w = 0.0
        per_class: dict[str, float] = {}
        for fc in classes:
            occ = fc.rate * requests_per_release
            if occ <= 0:
                continue
            if closed and fc.covered:
                # the eval set contains a test case -> the gate catches the regression pre-release
                continue
            escaped += occ
            escaped_w += occ * fc.severity
            per_class[fc.cid] = per_class.get(fc.cid, 0.0) + occ
            if fc.first_seen_release is None:
                fc.first_seen_release = rel
            if closed and not fc.covered:
                # did we trace AND judge an occurrence this release?
                traced = occ * sample_rate
                p_notice = 1.0 - (1.0 - judge_sensitivity) ** traced
                if rng.random() < p_notice:
                    fc.covered = True
                    covered += 1
                    eval_size += 1
        run.per_release.append({
            "release": rel, "escaped": escaped, "escaped_weighted": escaped_w,
            "escaped_per_class": per_class, "covered_classes": covered,
            "coverage": covered / len(classes), "eval_size": eval_size,
        })
    return run


def compare(releases: int = 24, seed: int = 31, **kw) -> dict:
    """Open vs closed. Reports the divergence, the crossover release, and the per-class split.

    **The number to look at is not the totals.** It is that the closed loop's per-release escape
    count *falls over time* while the open loop's is flat, so the ratio grows without bound -- and
    the last-release ratio is a divide by zero rather than a large number.

    The per-escape SEVERITY fields are the second finding, and they cut against the obvious
    expectation. The closed loop's residual escapes carry a HIGHER mean severity than the open
    loop's: the classes that resist closure longest are the rare, expensive ones, because a rare
    class produces few traced occurrences and the noticing probability falls with the rate. So the
    loop closes the head first and the tail last -- the tail-over-mean pattern this knowledge base
    finds in T08, T09 and T10, reproduced inside the loop built to fix it.
    """
    classes = failure_taxonomy(seed=seed)
    # each arm gets its own copy so `covered` state does not leak between them
    import copy
    open_run = run_loop(copy.deepcopy(classes), releases, closed=False, seed=seed, **kw)
    closed_run = run_loop(copy.deepcopy(classes), releases, closed=True, seed=seed, **kw)

    first, last = open_run.per_release[0], open_run.per_release[-1]
    c_first, c_last = closed_run.per_release[0], closed_run.per_release[-1]
    # cumulative
    cum_o, cum_c = 0.0, 0.0
    crossover = None
    for o, c in zip(open_run.per_release, closed_run.per_release):
        cum_o += o["escaped"]
        cum_c += c["escaped"]
        if crossover is None and c["escaped"] < o["escaped"] * 0.5:
            crossover = c["release"]
    return {
        "releases": releases,
        "n_classes": len(classes),
        "open": {"escaped_total": open_run.escaped_total,
                 "escaped_weighted": open_run.severity_weighted_total,
                 "first_release_escaped": first["escaped"],
                 "last_release_escaped": last["escaped"],
                 "final_coverage": open_run.final_coverage,
                 "final_eval_size": open_run.final_eval_size},
        "closed": {"escaped_total": closed_run.escaped_total,
                   "escaped_weighted": closed_run.severity_weighted_total,
                   "first_release_escaped": c_first["escaped"],
                   "last_release_escaped": c_last["escaped"],
                   "final_coverage": closed_run.final_coverage,
                   "final_eval_size": closed_run.final_eval_size},
        "escaped_ratio_total": (open_run.escaped_total / closed_run.escaped_total
                                if closed_run.escaped_total else math.inf),
        "escaped_ratio_weighted": (open_run.severity_weighted_total
                                   / closed_run.severity_weighted_total
                                   if closed_run.severity_weighted_total else math.inf),
        # mean severity per escaped occurrence, in each arm. If the closed loop's residual carries a
        # HIGHER mean severity than the open loop's, then the classes that resist closure longest are
        # the expensive ones -- the tail-over-mean pattern again, inside the loop itself.
        "open_severity_per_escape": (open_run.severity_weighted_total / open_run.escaped_total
                                     if open_run.escaped_total else 0.0),
        "closed_severity_per_escape": (closed_run.severity_weighted_total / closed_run.escaped_total
                                       if closed_run.escaped_total else 0.0),
        # which release first reaches full coverage -> all closed-loop escapes precede it
        "full_coverage_release": next((r["release"] for r in closed_run.per_release
                                       if r["coverage"] >= 1.0), None),
        "last_release_ratio": (last["escaped"] / c_last["escaped"] if c_last["escaped"] else math.inf),
        "crossover_release": crossover,
        "open_run": open_run, "closed_run": closed_run,
    }


def open_loop_cost(run: LoopRun, storage_gb_per_release: float, gpu_hours_per_release: float) -> dict:
    """What the open loop actually bought: a storage bill and the same defect rate.

    The corpus's "you have paid for storage, not for insight" [T], as arithmetic. The point is not
    that the storage is expensive in absolute terms -- it is that its RETURN is zero, so any price is
    too high.
    """
    n = len(run.per_release)
    return {"releases": n, "storage_gb_total": storage_gb_per_release * n,
            "gpu_hours_total": gpu_hours_per_release * n,
            "defects_prevented": 0.0,
            "coverage_gained": 0.0,
            "return_per_gb": 0.0,
            "note": "traces were collected, stored and never read: storage, not insight"}
