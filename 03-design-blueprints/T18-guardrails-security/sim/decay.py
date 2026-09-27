"""T18 -- decay: "a rail you tested in March may be bypassed by June."

That sentence [T] is a schedule, and a schedule is a number. This module turns it into one.

Two populations of attack are living in the wild at all times: techniques the pattern library already
covers, and techniques it does not. Attackers rotate toward the second, because that is where the
return is. A red-team run converts some of the second population into the first. Between runs,
nothing does.

So the catch rate is a **sawtooth**, and the three numbers that matter are the mean (what the
corpus's headline metric -- *"track the injection catch rate over time"* [T] -- reports), the trough
(what the system is actually exposed to just before each red-team run), and the exposure, which is
the area between the sawtooth and what continuous red-teaming would have bought.

And the mean is the worst of the three to steer by, for a reason this knowledge base keeps
rediscovering: **the novel classes are a small share of traffic and a large share of consequence.**
The mean catch rate stays in the high seventies while the whole severity-5 population sits at the
residual. Averaging away the population that is both rare and dangerous is the same signature as
T08's goodput, T09's p99, T10's SNR and T17's gate -- here with a time axis and an attacker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .rails import ATTACK_MIX, measured_catch, FULL_STACK

# In-library catch and out-of-library catch, taken from the rail stack itself rather than assumed.
_LIB = tuple(a for a in ATTACK_MIX if a.technique != "novel")
_NOVEL = tuple(a for a in ATTACK_MIX if a.technique == "novel")

R_LIBRARY = measured_catch(FULL_STACK, _LIB)
R_RESIDUAL = measured_catch(FULL_STACK, _NOVEL)

W_LIBRARY = sum(a.weight for a in _LIB)
W_NOVEL = sum(a.weight for a in _NOVEL)

SEV_LIBRARY = sum(a.weight * a.severity for a in _LIB) / W_LIBRARY
SEV_NOVEL = sum(a.weight * a.severity for a in _NOVEL) / W_NOVEL


@dataclass
class DecayParams:
    novel_share_0: float = 0.18      # today's mix, from ATTACK_MIX
    rotation: float = 0.06           # share of the in-library population that rotates to novel per period
    discovery: float = 0.75          # share of the novel stock a red-team run actually finds
    periods: int = 24                # monthly releases over two years
    period_name: str = "month"


@dataclass
class Cadence:
    name: str
    every: int          # red-team every N periods; 0 means never


CADENCES = (
    Cadence("continuous (every release)", 1),
    Cadence("quarterly", 3),
    Cadence("semi-annual", 6),
    Cadence("annual", 12),
    Cadence("never (launch checkbox)", 0),
)


def run_decay(params: DecayParams = DecayParams()) -> dict:
    """One trajectory per cadence, plus the continuous baseline that defines the exposure."""
    series = {}
    for cad in CADENCES:
        novel = params.novel_share_0
        rows = []
        for t in range(1, params.periods + 1):
            # Between runs, attackers rotate: part of the in-library population becomes novel.
            novel = novel + params.rotation * (1.0 - novel)
            if cad.every and t % cad.every == 0:
                novel = novel * (1.0 - params.discovery)
            mean_catch = (1.0 - novel) * R_LIBRARY + novel * R_RESIDUAL
            sev_catch = ((1.0 - novel) * SEV_LIBRARY * R_LIBRARY
                         + novel * SEV_NOVEL * R_RESIDUAL) / (
                (1.0 - novel) * SEV_LIBRARY + novel * SEV_NOVEL)
            rows.append({"period": t, "novel_share": novel,
                         "mean_catch": mean_catch, "severity_catch": sev_catch,
                         "novel_catch": R_RESIDUAL, "red_team": bool(cad.every and t % cad.every == 0)})
        series[cad.name] = rows

    baseline = series[CADENCES[0].name]
    for name, rows in series.items():
        exposure = sum(b["mean_catch"] - a["mean_catch"] for a, b in zip(rows, baseline))
        sev_exposure = sum(b["severity_catch"] - a["severity_catch"] for a, b in zip(rows, baseline))
        mean = sum(r["mean_catch"] for r in rows) / len(rows)
        sev_mean = sum(r["severity_catch"] for r in rows) / len(rows)
        trough = min(r["mean_catch"] for r in rows)
        trough_row = min(rows, key=lambda r: r["mean_catch"])
        sev_trough = min(r["severity_catch"] for r in rows)
        series[name] = {
            "rows": rows, "cadence": name,
            "mean_catch": mean, "severity_mean_catch": sev_mean,
            "trough": trough, "trough_period": trough_row["period"],
            "severity_trough": sev_trough,
            "exposure": exposure,
            "exposure_pct_of_periods": exposure / len(rows) * 100.0,
            "severity_exposure": sev_exposure,
            "novel_catch_flat": R_RESIDUAL,
            "mean_minus_severity": mean - sev_mean,
        }
    return {"series": series, "baseline": CADENCES[0].name,
            "r_library": R_LIBRARY, "r_residual": R_RESIDUAL,
            "params": params}


def cadence_table(params: DecayParams = DecayParams()) -> list:
    out = run_decay(params)
    rows = []
    for cad in CADENCES:
        s = out["series"][cad.name]
        rows.append({
            "cadence": cad.name, "every": cad.every,
            "mean_catch": s["mean_catch"], "trough": s["trough"],
            "severity_mean": s["severity_mean_catch"], "severity_trough": s["severity_trough"],
            "exposure_pct": s["exposure_pct_of_periods"],
            "extra_exposure_vs_continuous": s["exposure"],
        })
    return rows


def cadence_for_trough(target: float, params: DecayParams = DecayParams()) -> dict:
    """The largest gap between red-team runs that keeps the trough above a target catch rate.

    This is the sentence "red team on a schedule" [T] with the schedule solved for. It is the only
    form of that instruction an engineering plan can be written against: the LARGEST gap between
    runs that still holds the trough above the target, so the answer is a budget rather than a
    superstition.
    """
    for cad in sorted([c for c in CADENCES if c.every], key=lambda c: -c.every):
        s = run_decay(params)["series"][cad.name]
        if s["trough"] >= target:
            return {"target": target, "cadence": cad.name, "every": cad.every,
                    "trough": s["trough"], "feasible": True}
    return {"target": target, "cadence": "infeasible at any cadence above 1 period",
            "every": 0, "trough": 0.0, "feasible": False}


def metric_blindness(params: DecayParams = DecayParams()) -> dict:
    """Why the metric the corpus prescribes cannot see the failure it is meant to catch.

    Three readings of the same decaying system. The published one is the mean. It moves a little.
    The severity-weighted one moves more. The novel-class one does not move AT ALL -- it sits at the
    residual from the first period to the last, because it was never above it. A metric that has
    never crossed a threshold cannot trigger an alert.
    """
    s = run_decay(params)["series"]["never (launch checkbox)"]["rows"]
    first, last = s[0], s[-1]
    return {
        "periods": len(s),
        "mean_first": first["mean_catch"], "mean_last": last["mean_catch"],
        "mean_drop": first["mean_catch"] - last["mean_catch"],
        "severity_first": first["severity_catch"], "severity_last": last["severity_catch"],
        "severity_drop": first["severity_catch"] - last["severity_catch"],
        "novel_first": first["novel_catch"], "novel_last": last["novel_catch"],
        "novel_drop": first["novel_catch"] - last["novel_catch"],
        "novel_share_first": first["novel_share"], "novel_share_last": last["novel_share"],
    }
