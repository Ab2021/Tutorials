"""The eval gate -- mean, percentile, or worst slice, and why the choice decides the release.

The corpus's rule [T]:

    "And a mean of 4.2 can still hide the 5% of answers that leak data or invent facts. Always
     inspect the worst cases, not just the average."
    "Most importantly, watch the fifth percentile, your worst cases. A great average with an ugly
     tail is exactly the profile that produces embarrassing screenshots."

and the mechanism it feeds into [T]:

    "run the whole thing in CI so a regression fails the build. Do that, and 'is it good?' stops
     being an argument and becomes a number you can point at release after release."

This module computes three gates over the same set of scores. They disagree, and the disagreement
is the finding -- **it is the same tail-over-mean pattern this knowledge base finds in T08's goodput,
T09's p99 and T10's SNR**, arrived at independently in the evaluation plane.

Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ScoreSet:
    """Per-example scores from ONE release, plus the class each example belongs to.

    The class label is what makes a worst-slice gate possible. A gate that only sees a list of
    numbers can compute a percentile but cannot ask "did the safety slice regress?", and the
    safety slice is the one that produces the embarrassing screenshot.
    """

    name: str
    scores: list[float]
    classes: list[str] = field(default_factory=list)   # "easy" | "hard" | "safety" ...

    def __post_init__(self) -> None:
        if self.classes and len(self.classes) != len(self.scores):
            raise ValueError("classes must be the same length as scores")
        if not self.classes:
            self.classes = ["all"] * len(self.scores)

    @property
    def n(self) -> int:
        return len(self.scores)

    def mean(self) -> float:
        return sum(self.scores) / self.n if self.n else 0.0

    def percentile(self, p: float) -> float:
        """Linear-interpolated percentile. p in [0, 100]."""
        if not self.scores:
            return 0.0
        xs = sorted(self.scores)
        if len(xs) == 1:
            return xs[0]
        k = (len(xs) - 1) * (p / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    def by_class(self) -> dict[str, list[float]]:
        acc: dict[str, list[float]] = {}
        for s, c in zip(self.scores, self.classes):
            acc.setdefault(c, []).append(s)
        return acc

    def worst_class_mean(self) -> tuple[str, float]:
        bc = self.by_class()
        if not bc:
            return ("all", 0.0)
        name = min(bc, key=lambda c: sum(bc[c]) / len(bc[c]))
        return name, sum(bc[name]) / len(bc[name])


def make_release(name: str, n_hard: int = 60, n_easy: int = 140, n_safety: int = 40,
                 easy_mu: float = 4.6, hard_mu: float = 3.6, safety_mu: float = 4.4,
                 safety_bad_frac: float = 0.03, safety_bad_mu: float = 1.6,
                 sigma: float = 0.55, seed: int = 5) -> ScoreSet:
    """Three slices, because one number cannot describe a release and the slices disagree.

    `safety_mu` is deliberately HIGH: the safety slice's *mean* looks healthy. The failures are a
    small fraction of it, so they are invisible in the aggregate and obvious in the worst cases --
    which is the corpus's "mean of 4.2 hiding the 5%" made concrete.
    """
    rng = random.Random(seed)
    scores: list[float] = []
    classes: list[str] = []
    for _ in range(n_easy):
        scores.append(min(5.0, max(1.0, rng.gauss(easy_mu, sigma))))
        classes.append("easy")
    for _ in range(n_hard):
        scores.append(min(5.0, max(1.0, rng.gauss(hard_mu, sigma))))
        classes.append("hard")
    n_bad = int(round(n_safety * safety_bad_frac))
    for i in range(n_safety):
        mu = safety_bad_mu if i < n_bad else safety_mu
        scores.append(min(5.0, max(1.0, rng.gauss(mu, sigma))))
        classes.append("safety")
    return ScoreSet(name, scores, classes)


@dataclass
class Gate:
    """One release criterion. `kind` is the whole design decision."""

    kind: str                 # "mean" | "percentile" | "worst_class"
    threshold: float
    percentile: float = 5.0
    target_class: str | None = None

    def measure(self, s: ScoreSet) -> float:
        if self.kind == "mean":
            return s.mean()
        if self.kind == "percentile":
            return s.percentile(self.percentile)
        if self.kind == "worst_class":
            _, v = s.worst_class_mean()
            return v
        raise ValueError(f"unknown gate kind: {self.kind}")

    def evaluate(self, s: ScoreSet) -> dict:
        v = self.measure(s)
        return {"gate": self.describe(), "kind": self.kind, "release": s.name,
                "value": v, "threshold": self.threshold, "pass": v >= self.threshold,
                "n": s.n}

    def describe(self) -> str:
        if self.kind == "mean":
            return f"mean >= {self.threshold}"
        if self.kind == "percentile":
            return f"p{self.percentile:g} >= {self.threshold}"
        return f"worst-class mean >= {self.threshold}"


def gate_matrix(releases: list[ScoreSet], gates: list[Gate]) -> list[dict]:
    """Every gate against every release -- and the cells where they disagree are the deliverable."""
    return [g.evaluate(r) for r in releases for g in gates]


def disagreement(releases: list[ScoreSet], gates: list[Gate]) -> dict:
    """How often the gates disagree, and in which direction.

    The interesting direction is mean-PASS while a tail gate FAILS. That is the release that ships a
    regression, and it is invisible to anyone watching the number the corpus warns about.
    """
    rows = gate_matrix(releases, gates)
    by_release: dict[str, list[dict]] = {}
    for r in rows:
        by_release.setdefault(r["release"], []).append(r)
    split = [name for name, rs in by_release.items()
             if len({r["pass"] for r in rs}) > 1]
    mean_pass_tail_fail = []
    for name, rs in by_release.items():
        m = next((r for r in rs if r["kind"] == "mean"), None)
        tails = [r for r in rs if r["kind"] != "mean"]
        if m and m["pass"] and tails and all(not t["pass"] for t in tails):
            mean_pass_tail_fail.append(name)
    return {"n_releases": len(releases), "n_gates": len(gates),
            "split_releases": split, "n_split": len(split),
            "mean_passes_tail_fails": mean_pass_tail_fail,
            "rates": {g.describe(): sum(1 for r in rows if r["gate"] == g.describe() and r["pass"])
                      / len(releases) for g in gates}}


def regression_hidden_by_mean(baseline: ScoreSet, candidate: ScoreSet,
                              sliver: str = "safety") -> dict:
    """The release where the mean IMPROVED and a slice got worse. The pattern to watch for.

    Returns the movement in both the aggregate and the named slice, so the two can be compared
    directly. A team that reviews only the aggregate sees an improvement and ships; the slice is
    where the user-visible harm went.
    """
    b_slice = baseline.by_class().get(sliver, [])
    c_slice = candidate.by_class().get(sliver, [])
    b_mu = sum(b_slice) / len(b_slice) if b_slice else 0.0
    c_mu = sum(c_slice) / len(c_slice) if c_slice else 0.0
    return {
        "baseline": baseline.name, "candidate": candidate.name, "slice": sliver,
        "mean_baseline": baseline.mean(), "mean_candidate": candidate.mean(),
        "mean_delta": candidate.mean() - baseline.mean(),
        "slice_baseline": b_mu, "slice_candidate": c_mu, "slice_delta": c_mu - b_mu,
        "p5_baseline": baseline.percentile(5.0), "p5_candidate": candidate.percentile(5.0),
        "p5_delta": candidate.percentile(5.0) - baseline.percentile(5.0),
        "mean_improved": candidate.mean() > baseline.mean(),
        "slice_worsened": c_mu < b_mu,
        "hidden_regression": candidate.mean() > baseline.mean() and c_mu < b_mu,
    }
