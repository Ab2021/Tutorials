### 4.4 Limits and counterexamples

Each formula above has a region where it stops meaning anything. The boundaries are worth memorizing because they are not visible in the algebra.

**The Wald interval fails at small counts.** The normal approximation to the binomial requires enough successes and failures in each arm.

```text
Rule of thumb: at least 5 events and 5 non-events per arm.
Illustration: 40 per arm with 2 and 1 events.
  p1 = 2/40 = 0.050, p0 = 1/40 = 0.025, ITT = 0.025
  Var = 0.05*0.95/40 + 0.025*0.975/40 = 0.0011875 + 0.0006094
      = 0.0017969, SE = 0.04239
  Wald CI = 0.025 +/- 0.0831 = [-0.0581, 0.1081]
```

The interval's lower bound is below zero, which is impossible for a proportion difference that cannot be smaller than `-p0`, and the upper bound implies a treated rate of 13.3% when only 2 of 40 converted. With one arm at zero events the Wald interval collapses to a point, which is why exact or Agresti-Coull intervals are mandatory below the rule of thumb.

**Stratification fails when a stratum is degenerate.** If a stratum has zero events in one arm, its variance estimate is zero and its rate is 0% or 100%.

```text
Stratum with n1 = 30, y1 = 0, n0 = 30, y0 = 0:
  d_s = 0, v_s = 0  -> this stratum contributes weight but no variance
Two such strata plus one noisy large stratum produces an SE that is
too small and a weight structure that no longer reflects the population.
Correction: collapse until each arm in each stratum has >= 5 events.
```

**CACE is undefined when the compliance difference is zero.** If delivery is 60% in both arms — because the platform delivers the "control" campaign too — the denominator is zero and the receipt effect does not exist as a separate estimand. No amount of algebra recovers it; the correct statement is that the experiment identifies only the ITT of two nearly identical policies, and the design failed.

**DiD breaks under staggered adoption with heterogeneous effects.** When units adopt at different times and effects grow with exposure, the two-way fixed-effects estimator is a weighted average of all possible 2×2 comparisons, some of which use already-treated units as controls, and some of those weights are negative.

```text
Illustration of the sign problem: a later-treated cohort is used as the
control for an earlier-treated cohort. If the earlier cohort's effect is
still growing, its trend contaminates the "control" change and the
estimate can be biased with the wrong sign even though every 2×2
comparison in isolation is positive.
Diagnostic: run the estimator on each 2×2 pair; if the pairs disagree in
sign, the aggregate is not interpretable as an effect.
```

**Cluster standard errors break with few clusters.** With `G < 40` clusters, cluster-robust variance is biased downward and the normal reference distribution is wrong. The correction is a `t` distribution with `G - 2` degrees of freedom, and below about 10 clusters no correction rescues inference — the design, not the estimator, is the problem.

**Every formula above assumes the outcome is measured on assigned units.** If the outcome table contains one row per order and the assignment table one row per customer, the rates computed are per-order rates and the estimand silently changes to a purchase-weighted quantity. That error is invisible in the arithmetic and visible only in the grain check, which is why every readout should print `rows(assignment)`, `rows(outcome)`, and `distinct outcome units`.

**Validity condition for the whole of Section 4:** the formulas identify a causal quantity only when the assignment is recorded before outcomes, the analysis units match the assignment units, and the comparator is a real policy rather than an absence. Remove any one of those and the arithmetic still runs, the intervals still print, and the number no longer means anything.
