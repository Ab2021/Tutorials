"""T19 -- the unit: the decision that precedes every measurement, and the one nobody writes down.

"Optimization without measurement is just guessing" [T] is the corpus's rule, and it is half a rule.
Measurement requires a UNIT, and the unit is chosen before anything is measured -- which means the
unit decides the answer.

The Token Raj figures make this concrete and are the strongest two data points in the whole topic.
Between February and April, the same nominal task -- a 10,000-token knowledge artifact -- fell from
**30c to 16c**. In the same period, lines of code drafted fell from **630 to 91** and files touched
from **8.2 to 3.6**, while thinking tokens rose [T]. Three normalisations of those same two numbers
give three different stories, and they do not even share a sign:

    per artifact   0.53x   ->  a 47% saving    (the headline)
    per file       1.22x   ->  a 22% increase
    per line       3.69x   ->  a 270% increase (the delivered work)

So this module is about units, and it carries the corpus's own instruction on the subject: companies
using "tokens per output as one of the metrics -- you need to change the metrics. You need to start
looking at it as a business outcome rather than the token spend" [T].

The second half is the workload mix, where the same average-hides-the-tail shape appears with a time
axis: a small, expensive population that the average does not describe, and which the scenario's own
growth rate turns into the majority of the bill.
"""
from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------------------------
# Normalisation. All figures corpus [T]; the arithmetic is the module.
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Artifact:
    period: str
    cost: float          # in cents
    lines: float
    files: float


FEB = Artifact("Feb", 0.30, 630.0, 8.2)
APR = Artifact("Apr", 0.16, 91.0, 3.6)

WORKLOADS = {"per_artifact": "cost", "per_line": "lines", "per_file": "files"}
UNIT_NOTE = {
    "per_artifact": "what was asked for -- the unit the headline uses",
    "per_file":     "what changed -- a proxy for coupling and review load",
    "per_line":     "what was delivered -- the unit that is cheapest to inflate",
}
HEADLINE_UNIT = "per_artifact"


def normalised(a: Artifact, b: Artifact) -> dict:
    """Every unit, from the same two rows. Ratios above 1.0 mean MORE expensive per unit."""
    out = {
        "per_artifact": b.cost / a.cost,
        "per_file": (b.cost / b.files) / (a.cost / a.files),
        "per_line": (b.cost / b.lines) / (a.cost / a.lines),
    }
    return {
        "from": a.period, "to": b.period,
        "ratios": out,
        "headline_saving": 1.0 - out["per_artifact"],
        "per_unit_cost_before": {"artifact": a.cost, "file": a.cost / a.files,
                                 "line": a.cost / a.lines},
        "per_unit_cost_after": {"artifact": b.cost, "file": b.cost / b.files,
                                "line": b.cost / b.lines},
        "sign_disagreement": any(r > 1.0 for r in out.values()) and any(r < 1.0 for r in out.values()),
    }


def unit_ranking(a: Artifact = FEB, b: Artifact = APR) -> list:
    """The units, best-looking first. The ranking IS the incentive, so it is worth printing."""
    n = normalised(a, b)
    rows = [{"unit": u, "ratio": r, "reading": ("cheaper" if r < 1.0 else "MORE expensive"),
             "change_pct": (r - 1.0) * 100.0, "note": UNIT_NOTE[u]}
            for u, r in n["ratios"].items()]
    return sorted(rows, key=lambda r: r["ratio"])


def deflation_check(a: Artifact = FEB, b: Artifact = APR) -> dict:
    """Was the task the same task? The corpus says the cost fell because the artifact shrank.

    The test is not whether the price fell -- it did -- but whether the DELIVERABLE fell faster.
    """
    return {
        "cost_ratio": b.cost / a.cost,
        "lines_ratio": b.lines / a.lines,
        "files_ratio": b.files / a.files,
        "deflator_ratio": (b.lines / a.lines) if a.lines else float("nan"),
        "real_cost_change": (b.cost / b.lines) / (a.cost / a.lines),
        "verdict": ("the deliverable shrank faster than the price"
                    if (b.lines / a.lines) < (b.cost / a.cost) else
                    "the price fell faster than the deliverable"),
    }


def metric_families() -> list:
    """The metrics a cost programme reaches for, and what each one cannot see.

    The corpus's objection is specific: "companies which are doing tokens per output as one of the
    metrics -- you need to change the metrics" [T]. The general form is that every one of these is
    a ratio whose denominator is chosen by the team being measured.
    """
    return [
        {"metric": "cost per call", "denominator": "calls",
         "gameable_by": "splitting one task into more calls",
         "cannot_see": "the tail -- 1% of calls can be 29% of the bill"},
        {"metric": "tokens per output", "denominator": "visible output tokens",
         "gameable_by": "moving work into reasoning tokens, which are not visible",
         "cannot_see": "reasoning spend, billed at the output rate and absent from the response"},
        {"metric": "cost per task", "denominator": "tasks",
         "gameable_by": "doing less work per task, as the Feb->Apr figures show",
         "cannot_see": "whether the task got smaller"},
        {"metric": "cost per resolved case", "denominator": "resolutions",
         "gameable_by": "reclassifying resolutions",
         "cannot_see": "much -- this is the corpus's preferred direction [R]"},
        {"metric": "lines of code / PRs merged", "denominator": "commits",
         "gameable_by": "generation volume",
         "cannot_see": "value -- a model 'tends to make a shallow copy of the whole file' [T]"},
    ]


# --------------------------------------------------------------------------------------------
# The workload mix: a small expensive population, and the growth that makes it the majority.
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    name: str
    calls: int
    cost_per_call: float
    note: str = ""


# Illustrative, derived from the case study's own assumptions: a chat turn is a few thousand
# tokens at input/output list rates; an agentic task is ~12 steps of the plan-act-observe loop
# [R] with reasoning on, which the corpus puts at 10-100x non-agentic compute [T].
MIX = (
    Workload("chat", 990_000, 0.0030, "a chat turn is cents [R]"),
    Workload("agentic", 10_000, 0.1224,
             "an agentic multi-step task can run from tens of cents to several dollars [R]"),
)

GROWTH_FACTOR = 4.3       # [D] the scenario's YoY token growth at flat headcount


def mix_summary(mix=MIX) -> dict:
    total_calls = sum(w.calls for w in mix)
    total_cost = sum(w.calls * w.cost_per_call for w in mix)
    rows = [{"name": w.name, "calls": w.calls,
             "call_share": w.calls / total_calls,
             "cost": w.calls * w.cost_per_call,
             "cost_share": (w.calls * w.cost_per_call) / total_cost,
             "cost_per_call": w.cost_per_call,
             "note": w.note} for w in mix]
    return {"rows": rows, "total_calls": total_calls, "total_cost": total_cost,
            "mean_cost_per_call": total_cost / total_calls,
            "mean_cost_per_costly_call": max(w.cost_per_call for w in mix)}


def tail_ratio(mix=MIX) -> dict:
    m = mix_summary(mix)
    return {"mean_per_call": m["mean_cost_per_call"],
            "costly_call": m["mean_cost_per_costly_call"],
            "ratio": m["mean_cost_per_costly_call"] / m["mean_cost_per_call"]}


def growth_scenario(factor: float = GROWTH_FACTOR, grows: str = "agentic", mix=MIX) -> dict:
    """Grow one workload by `factor` and re-read the shares.

    The interesting output is not the new total. It is that a population which is 1% of volume and
    29% of spend becomes the majority of spend within one growth cycle -- so an optimisation order
    derived from TODAY's average is the wrong order for next year's bill.
    """
    before = mix_summary(mix)
    after_mix = tuple(
        Workload(w.name, int(w.calls * factor), w.cost_per_call, w.note) if w.name == grows else w
        for w in mix)
    after = mix_summary(after_mix)
    return {"before": before, "after": after, "factor": factor, "grew": grows,
            "before_share": {r["name"]: r for r in before["rows"]},
            "after_share": {r["name"]: r for r in after["rows"]}}


def cost_per_task_vs_call(mix=MIX, calls_per_task: int = 1) -> dict:
    """The corpus's unit instruction, as arithmetic: 'cost per task, not cost per call' [R]."""
    m = mix_summary(mix)
    return {"cost_per_call": m["mean_cost_per_call"],
            "cost_per_task": m["mean_cost_per_call"] * calls_per_task,
            "note": "the ratio between them is calls-per-task, which is the metric an agent "
                    "programme moves without anyone deciding to"}


def where_the_money_is(mix=MIX, cut: float = 0.20) -> list:
    """A uniform 20% cut applied to each workload, and what it is actually worth."""
    m = mix_summary(mix)
    return sorted(({"workload": r["name"], "cost": r["cost"],
                    "saving": r["cost"] * cut,
                    "cost_share": r["cost_share"]} for r in m["rows"]),
                  key=lambda r: -r["saving"])


def reasoning_invisibility(visible: int, reasoning: int) -> dict:
    """The same call, read as 'tokens per output' and read as cost."""
    total = visible + reasoning
    return {"visible": visible, "reasoning": reasoning,
            "tokens_per_output_metric": 1.0,
            "cost_multiple_vs_metric": total / visible if visible else float("inf"),
            "reasoning_share_of_output_billing": reasoning / total if total else 0.0}
