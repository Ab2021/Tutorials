"""T19 -- attribution: the precondition, and the metric that hides the team that is growing.

The guide's FinOps practice reduces to four steps and the first is non-negotiable: tag every call
by team, feature, tenant, model, route and environment, because "without attribution there is no
way to compute unit economics" [R]. Showback comes before chargeback, and only "once the tags are
trustworthy" [R].

Two things this module adds to that.

  * **Coverage is a fraction, and uncovered spend is invisible -- not neutral.** A 92% tag coverage
    figure is an average, and the teams that are growing fastest are the ones whose instrumentation
    was written last. The uncovered 8% is therefore not a random sample of the bill; it is
    concentrated exactly where the growth is. This is the same shape as every other tail in this
    knowledge base, applied to a governance metric.

  * **Chargeback can drive traffic off the gateway.** Once spend is billed to a team, the cheapest
    route for that team is a personal account, which is invisible to every metric above. The control
    is reconciliation: gateway spend against provider-side invoices for the same period, and the
    residual is the shadow total.

Stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    name: str
    spend: float            # this team's true spend on inference
    tagged_share: float     # fraction of that spend carrying a usable tag
    growth: float           # YoY growth multiple
    note: str = ""


# Illustrative. The design point is the CORRELATION between tagged_share and growth, which is what
# makes a coverage average misleading rather than merely incomplete.
TEAMS = (
    Team("platform_copilot", 420_000, 0.98, 1.4, "instrumented first, growth flat"),
    Team("customer_assist",  260_000, 0.95, 2.1, "instrumented second"),
    Team("data_science",     140_000, 0.85, 3.0, "self-serve notebooks"),
    Team("agent_pilot",      100_000, 0.55, 6.5, "the fastest-growing consumer, tagged last"),
    Team("finance_ops",       80_000, 0.90, 1.2, "steady"),
)

GROWTH_FACTOR = 4.3


def coverage(teams=TEAMS) -> dict:
    """Total coverage, and the two readings that disagree about it."""
    total = sum(t.spend for t in teams)
    tagged = sum(t.spend * t.tagged_share for t in teams)
    rows = [{"team": t.name, "spend": t.spend, "tagged_share": t.tagged_share,
             "untagged": t.spend * (1.0 - t.tagged_share), "growth": t.growth,
             "note": t.note} for t in teams]
    weighted = tagged / total
    simple = sum(t.tagged_share for t in teams) / len(teams)
    return {"rows": sorted(rows, key=lambda r: -r["untagged"]),
            "total_spend": total, "tagged_spend": tagged,
            "untagged_spend": total - tagged,
            "spend_weighted_coverage": weighted,
            "simple_mean_coverage": simple,
            "coverage_gap": simple - weighted,
            "worst_team": min(rows, key=lambda r: r["tagged_share"]),
            "largest_untagged": max(rows, key=lambda r: r["untagged"])}


def untagged_share_of_growth(teams=TEAMS, factor: float = GROWTH_FACTOR) -> dict:
    """Does the untagged spend live where the growth is?

    Each team's spend is scaled by its own growth rate RELATIVE TO THE SLOWEST-GROWING team, so no
    team shrinks and the total bill grows -- which is what a growing business looks like. Coverage is
    then recomputed. If uncovered spend were randomly distributed, coverage would be stable under
    growth. It is not: it falls, because the uncovered teams are the growing ones. (The coverage
    RATIO is invariant to which reference is used -- any common multiplier cancels -- but the untagged
    SPEND and its share of a growing bill are not, and those are the numbers worth quoting.)
    """
    ref = min(t.growth for t in teams)
    grown = tuple(
        Team(t.name, t.spend * (t.growth / ref), t.tagged_share, t.growth, t.note)
        for t in teams)
    before = coverage(teams)
    after = coverage(grown)
    return {"before": before, "after": after,
            "growth_reference": ref,
            "mean_growth": sum(t.growth for t in teams) / len(teams),
            "total_growth": after["total_spend"] / before["total_spend"],
            "coverage_before": before["spend_weighted_coverage"],
            "coverage_after": after["spend_weighted_coverage"],
            "coverage_delta": after["spend_weighted_coverage"] - before["spend_weighted_coverage"],
            "untagged_before": before["untagged_spend"],
            "untagged_after": after["untagged_spend"],
            "untagged_share_before": before["untagged_spend"] / before["total_spend"],
            "untagged_share_after": after["untagged_spend"] / after["total_spend"]}


def governance_ladder() -> list:
    """The four options, in the order the corpus implies, with what each one needs.

    "Showback before chargeback" [R] is a sequencing instruction, and the reason is political as
    well as technical: chargeback on untrusted tags produces a fight about the numbers instead of
    a plan to reduce them.
    """
    return [
        {"stage": "no attribution", "needs": "nothing", "behavioural_pressure": 0.0,
         "failure": "fails the first time the bill doubles"},
        {"stage": "showback", "needs": "tags", "behavioural_pressure": 0.3,
         "failure": "breaks when nobody acts on it"},
        {"stage": "chargeback", "needs": "trusted tags", "behavioural_pressure": 0.8,
         "failure": "breaks when teams route around the gateway"},
        {"stage": "hard per-run caps", "needs": "a gateway on the critical path",
         "behavioural_pressure": 1.0,
         "failure": "breaks if the cap is set from a mean rather than a tail"},
    ]


def shadow_gap(gateway_spend: float, provider_invoice: float) -> dict:
    """The reconciliation that detects traffic which left the gateway.

    A positive residual is spend the governance plane cannot see. It is also the only signal that
    fires on the failure mode chargeback introduces, which is why it belongs on the governance
    dashboard rather than in a quarterly audit.
    """
    return {"gateway_spend": gateway_spend, "provider_invoice": provider_invoice,
            "residual": provider_invoice - gateway_spend,
            "shadow_share": ((provider_invoice - gateway_spend) / provider_invoice
                             if provider_invoice else 0.0),
            "visible": gateway_spend >= provider_invoice * 0.99}


def cap_from_mean_vs_tail(mean_task_cost: float, p99_task_cost: float,
                          ceiling: float) -> dict:
    """A run ceiling set from the mean kills legitimate work; one set from the tail does not.

    "Runaway agents have burned tens of thousands of dollars over a single weekend" [R] is an
    argument for a ceiling. It is not an argument for a SMALL ceiling, and the difference is which
    statistic it is drawn from.
    """
    return {"mean_task_cost": mean_task_cost, "p99_task_cost": p99_task_cost,
            "ceiling": ceiling,
            "kills_legit_share": (1.0 if ceiling < p99_task_cost else 0.0),
            "catches_mean_runaway": ceiling > mean_task_cost * 10,
            "verdict": ("set from the tail -- legitimate long tasks survive"
                        if ceiling >= p99_task_cost else
                        "set from the mean -- it will terminate legitimate work")}
