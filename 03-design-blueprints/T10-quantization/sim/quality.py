"""From reconstruction error to a decision: how much quality did this cost?

Two things happen here, and they are separate on purpose.

**1. A fitted degradation curve.** The corpus publishes a quality-loss column -- FP8 `< 1%`, 4-bit
`1-2%`, 2-bit `10-15%` [R]. Those three numbers are the only ground truth available offline, so the
curve is FITTED to them and its residuals are reported rather than hidden. The input to the curve is
the SNR that `quantize.py` actually measures, which closes the loop: measured error -> predicted
quality loss -> compared against the corpus's own figures.

**2. The reason the answer is not a table.** The corpus's operational warning is that 4-bit is
"nearly free on many models, but on some it quietly drops accuracy. So always rerun your evals
after quantizing" [T]. The fitted curve plus the outlier-severity sweep in `quantize.py` shows
WHY: at a fixed 4 bits, SNR varies with the tensor's dynamic range, and the same bit width therefore
lands at different points on the same curve. The degradation is not a property of the bit width.

**This is a model, not a measurement.** It is calibrated to the corpus's three anchors and it
reproduces them; it has not been validated against a real model's evals, and the guide says the
only way to know is to run your own [T].
"""
from __future__ import annotations

import math

from .precision import QUALITY_ANCHORS

# --------------------------------------------------------------------------------------
# SNR -> relative error
# --------------------------------------------------------------------------------------

def relative_error(snr: float) -> float:
    """RMS relative error from SNR in dB: `10 ** (-snr/20)`.

    The bridge between the two halves of this topic. Everything in `quantize.py` speaks SNR;
    everything a stakeholder cares about speaks quality loss. This is the conversion, and it is
    just the definition of dB.
    """
    if snr == float("inf"):
        return 0.0
    return 10.0 ** (-snr / 20.0)


# --------------------------------------------------------------------------------------
# The fitted degradation curve
# --------------------------------------------------------------------------------------

def _loglog_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of `y = A * x**B` on log-log axes. Returns (A, B)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    b = num / den if den else 0.0
    a = math.exp(my - b * mx)
    return a, b


def fit_degradation(anchors: dict[str, float] | None = None,
                    snrs: dict[str, float] | None = None) -> dict:
    """Fit `delta_ppl_pct = A * rel_err**B` to the corpus's quality anchors.

    `snrs` must be supplied by the caller -- the SNRs that `quantize.py` measured for each
    precision on a realistic tensor. Passing them in rather than hard-coding them is deliberate:
    it keeps the fit tied to the quantizer's real behaviour, so if the quantizer changes the fit
    moves with it.
    """
    anchors = anchors or QUALITY_ANCHORS
    if snrs is None:
        raise ValueError("snrs is required -- measure them with quantize.snr_db")
    xs, ys, labels = [], [], []
    for name, loss in anchors.items():
        if loss <= 0.0 or name not in snrs:
            continue
        xs.append(relative_error(snrs[name]))
        ys.append(loss)
        labels.append(name)
    if len(xs) < 2:
        raise ValueError("need at least two non-zero anchors to fit")
    A, B = _loglog_fit(xs, ys)
    preds = [A * x ** B for x in xs]
    return {
        "A": A, "B": B, "labels": labels,
        "fit_on": list(zip(labels, ys, preds)),
        "max_residual_pct": max(abs(p - y) / y * 100.0 for y, p in zip(ys, preds)),
        "note": ("delta_ppl_pct = A * rel_err**B, fitted to the corpus anchors; "
                 "a MODEL, not a measurement"),
    }


def predicted_degradation(snr: float, model: dict) -> float:
    """Predicted percent perplexity increase at this SNR, from the fitted curve."""
    return model["A"] * relative_error(snr) ** model["B"]


# --------------------------------------------------------------------------------------
# The cliff -- why the bit width does not decide the answer
# --------------------------------------------------------------------------------------

def degradation_across_models(model: dict, bits: int, severities: list[float],
                              quantize_fn, weights_fn) -> list[dict]:
    """Same bit width, different tensors, different outcomes. THE CORPUS'S WARNING, DEMONSTRATED.

    For each outlier severity the caller supplies, this quantizes at a FIXED bit width, measures
    SNR, and reads the degradation off the fitted curve. The output is the reason the guide says
    to rerun evals rather than consult a table: the spread across severities at 4 bits is wide
    enough to turn "nearly free" into "a visible regression" with no change in configuration.

    `quantize_fn` and `weights_fn` are injected so this module does not depend on `quantize.py`
    -- the same function works for any quantizer.
    """
    from .quantize import snr_db
    rows = []
    for sev in severities:
        w = weights_fn(severity=sev)
        q = quantize_fn(w, bits)
        s = snr_db(w, q)
        rows.append({"severity": sev, "bits": bits, "snr_db": s,
                     "predicted_ppl_pct": predicted_degradation(s, model),
                     "rel_err": relative_error(s)})
    return rows


def eval_gate(rows: list[dict], threshold_pct: float = 2.0) -> dict:
    """Should this quantization be merged? The corpus's rule as a gate.

    "always rerun your evals after quantizing" [T] is advice; this is the same advice as a CI
    check. It takes the WORST row, not the mean, because the mean is the number that lets a
    regression through -- the same tail-versus-mean discipline as T08's goodput and T09's p99.

    Returns a verdict plus the rows that breached, so a merge can be blocked with a reason.
    """
    worst = max(rows, key=lambda r: r["predicted_ppl_pct"])
    breached = [r for r in rows if r["predicted_ppl_pct"] > threshold_pct]
    return {
        "threshold_pct": threshold_pct,
        "worst_severity": worst["severity"],
        "worst_ppl_pct": worst["predicted_ppl_pct"],
        "breached": breached,
        "verdict": "REJECT" if breached else "ACCEPT",
        "note": ("gate on the WORST configuration, not the average -- the same rule as T08's "
                 "goodput and T09's p99 regression"),
    }
