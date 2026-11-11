## 12. Quick Reference Card

| # | Item | What to do | Number / rule from this file |
|---:|---|---|---|
| 1 | Estimand | Write `ITT`, `CACE`, `ATT`, or `ATE` on the readout before computing anything | Weighting changes the answer: 0.688 / 0.656 / 0.717 pp from one table |
| 2 | ITT and interval | `p1 - p0` with `SE = sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)` | 1.028 pp, SE 0.390 pp, CI [0.26, 1.79] pp |
| 3 | Stratification | Expected counts and weights use `N_s * pi_s` and `N_s/N`, never realized arm shares | Naive 1.028 pp vs stratified 0.688 pp (33% bias) |
| 4 | Design effect | Multiply SE by `sqrt(1 + (m-1)rho)` for clustered assignment | `m=50, rho=0.02` → DEFF 1.98, SE 0.390 → 0.547 pp |
| 5 | CUPED | Pre-assignment covariate at correlation `rho` cuts SE by `sqrt(1-rho^2)` | `rho = 0.5` → factor 0.866; `rho = 0.7` → 0.714 |
| 6 | Compliance | `CACE = ITT / (delivery_treated - delivery_control)`; never delete undelivered units | 2.00 pp / (0.85 - 0.05) = 2.50 pp |
| 7 | MDE | `(1.959964 + 0.841621)*sqrt((p1(1-p1)+p0(1-p0))/n)` per arm | 1.115 pp at 5% baseline and 6,000/arm; 2.046 pp at 20% |
| 8 | Break-even | `effect* = cost_per_contact / contribution_per_purchase` | \$0.25/\$18 = 1.389 pp; compare to the interval, not the point |
| 9 | Sequential looks | Use OBF boundaries `2 - 2*Phi(1.959964/sqrt(t))`; unadjusted peeking multiplies error | 4 looks unadjusted → 0.127 FWER; OBF at `t=0.5` needs `|z|>2.7718` |
| 10 | Diagnosis | Post-assignment filters are the first suspect when a restricted readout is large | True ITT 0.000 can present as -1.50 pp among clickers |

The card in prose, for the three rows that matter most:

**Row 2 is the only arithmetic most readouts need.** Write it as a single expression so that no intermediate rounding happens: `(y1/n1 - y0/n0) ± 1.959964*sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)`. Compute the standard error from the unpooled rates; the pooled variant belongs in the hypothesis test, not the interval.

**Row 8 converts a statistical result into a business decision.** Compute it before the test launches, because its value determines whether the test can answer the question at all. If break-even is 1.389 pp and the MDE at the planned sample is 1.9 pp, the test is underpowered by construction and no analysis will repair that.

**Row 10 is the fastest diagnostic.** When a segmentation or exposure-restricted result is dramatically larger than the all-assigned result, the default hypothesis is post-treatment conditioning, and the check is comparing inclusion rates by arm. In the constructed example those rates were 0.600 versus 0.200 on a zero-effect treatment.
