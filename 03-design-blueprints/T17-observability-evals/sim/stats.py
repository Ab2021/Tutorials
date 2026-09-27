"""Eval set size: why "with 20 examples, one lucky run looks like real progress".

The corpus names the trap [T]:

    "A few traps quietly ruin evaluation. ... With 20 examples, one lucky run looks like real
     progress. With a single judge from one family, the score flatters itself. And a mean of 4.2
     can still hide the 5% of answers that leak data or invent facts. Always inspect the worst
     cases, not just the average."                     -- How to Evaluate LLM Apps

and gives the constructive version [T]:

    "They deliberately label the edge cases and the failures they have actually seen. And the set is
     a living thing. Every notable production failure becomes a new test case the following week. A
     curated 200 examples often beats a random 10,000."

Three separate statistical facts are hiding in those two paragraphs, and this module separates them:

  1. **Power.** A small set cannot DETECT a real regression, so a genuine break ships green.
  2. **False confidence.** A small set reports large improvements from a change that did nothing,
     because the noise floor is wider than the effect. This is the one that gets a bad change merged.
  3. **Tail representation.** A metric averaged over n examples cannot show you a failure that
     occurs in 5% of them unless the set is large enough to CONTAIN one. This is a different problem
     from power and it is the one the corpus's "mean of 4.2" sentence is about.

All three improve with n, at different rates, and (1) and (2) are two sides of the same standard
error. Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

import math
import random


def _erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26. Stdlib-only; accurate to ~1.5e-7, plenty for a design doc."""
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return 0.5 * (1.0 + _erf((x - mu) / (sigma * math.sqrt(2.0))))


def standard_error(sigma: float, n: int) -> float:
    """The width of the noise floor on a mean over n examples. Everything below follows from it."""
    if n <= 0:
        raise ValueError("n must be > 0")
    return sigma / math.sqrt(n)


def win_rate_sigma() -> float:
    """The per-example sd of a PAIRWISE win rate: a Bernoulli(0.5) has sd 0.5, its maximum.

    Worth stating because it is the worst case and it is the one teams actually hit: a judge that is
    at chance on close pairs produces a win rate whose per-example sd is 0.5, so a 100-pair eval has
    a standard error of 5 percentage points. **A three-point win-rate improvement on 100 pairs is
    inside the noise**, which is exactly the corpus's "one lucky run looks like real progress".
    """
    return 0.5


def minimum_detectable_effect(sigma: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """The smallest true effect this eval can detect, at this n, at this false-positive rate.

    Two-sided, two-sample (before vs after on independent draws), normal approximation:

        MDE = (z_{1-alpha/2} + z_{power}) * sigma * sqrt(2 / n)

    The z values are solved numerically from `normal_cdf` rather than looked up, so there is no
    table in the source to be wrong.
    """
    z_a = _z_for(1.0 - alpha / 2.0)
    z_p = _z_for(power)
    return (z_a + z_p) * sigma * math.sqrt(2.0 / n)


def _z_for(p: float) -> float:
    """Invert the normal CDF by bisection. Exact enough, and no magic constants."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def observed_gain_distribution(true_effect: float, sigma: float, n: int, trials: int = 20_000,
                               seed: int = 3) -> list[float]:
    """Simulate the measured improvement a release would report, given a TRUE effect of `true_effect`.

    Simulated rather than approximated because the two experiments below ask for tail probabilities,
    and the whole point of the exercise is that those tails are not small.
    """
    rng = random.Random(seed)
    se = standard_error(sigma, n)
    return [rng.gauss(true_effect, se) for _ in range(trials)]


def false_improvement_rate(sigma: float, n: int, threshold: float, trials: int = 20_000,
                           seed: int = 3) -> float:
    """P(report an improvement >= threshold | the change did NOTHING).

    This is the number behind the corpus's "one lucky run looks like real progress", and it depends
    only on `n` and on how big a number you consider worth shipping. **A team's shipping threshold is
    usually set from experience, not from `n`, which is how a 20-example eval ends up merging noise.**
    """
    draws = observed_gain_distribution(0.0, sigma, n, trials, seed)
    return sum(1 for d in draws if d >= threshold) / len(draws)


def detection_power(sigma: float, n: int, effect: float, threshold: float, trials: int = 20_000,
                    seed: int = 3) -> float:
    """P(report an improvement >= threshold | the change really did improve by `effect`).

    The mirror image, and the one that fails silently: a real regression of 0.2 is *invisible* at
    n = 20, so the fix ships and nobody notices for a release or two.
    """
    draws = observed_gain_distribution(effect, sigma, n, trials, seed)
    return sum(1 for d in draws if d >= threshold) / len(draws)


def tail_representation(n: int, failure_rate: float) -> dict:
    """Can this eval set even CONTAIN an example of a `failure_rate` failure class?

        P(at least one) = 1 - (1 - failure_rate) ** n

    A mean over n examples cannot report a problem in a class the set does not contain. At the
    corpus's 5% leak/invention rate [T], a 20-example set contains a leaking answer 64% of the time
    and a 200-example set contains one 99.997% of the time -- which is why "inspect the worst cases"
    needs a set large enough for a worst case to exist in.
    """
    if not 0.0 <= failure_rate <= 1.0:
        raise ValueError("failure_rate must be in [0, 1]")
    if n < 0:
        raise ValueError("n must be >= 0")
    p = 1.0 - (1.0 - failure_rate) ** n
    return {"n": n, "failure_rate": failure_rate, "p_at_least_one": p,
            "expected_count": n * failure_rate,
            "reliably_represented": p >= 0.95,
            "n_for_95pct": _n_for_representation(failure_rate, 0.95)}


def _n_for_representation(failure_rate: float, confidence: float) -> int:
    if failure_rate <= 0.0:
        return 0
    if failure_rate >= 1.0:
        return 1
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - failure_rate))


def eval_set_sizing(sigma: float = 0.5, threshold: float = 0.20,
                    sizes: list[int] | None = None) -> list[dict]:
    """The three effects side by side across n -- the table that answers "how big should it be?"

    The answer is not one number: the size that gives acceptable power against a 0.2 regression is
    larger than the size that keeps false improvements rare, and both are smaller than the size that
    reliably contains a 5% failure class. **The largest of the three binds, and it is usually the
    tail-representation one.**
    """
    sizes = sizes or [20, 50, 100, 200, 400, 1_000]
    rows = []
    for n in sizes:
        rows.append({
            "n": n,
            "se": standard_error(sigma, n),
            "mde_80": minimum_detectable_effect(sigma, n),
            "p_false_improve": false_improvement_rate(sigma, n, threshold),
            "power_0.2": detection_power(sigma, n, 0.20, threshold),
            "p_contains_5pct": tail_representation(n, 0.05)["p_at_least_one"],
        })
    return rows


def required_n(sigma: float, effect: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """The n that gives `power` against `effect`. The inverse of `minimum_detectable_effect`."""
    z_a = _z_for(1.0 - alpha / 2.0)
    z_p = _z_for(power)
    return math.ceil(2.0 * (sigma * (z_a + z_p) / effect) ** 2)
