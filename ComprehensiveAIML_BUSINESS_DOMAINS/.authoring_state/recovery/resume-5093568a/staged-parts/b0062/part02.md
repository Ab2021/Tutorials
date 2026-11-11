## 4. Core Formulas with Full Derivation

Every result below is presented in the same five-part shape, and each block closes with an explicit validity condition that says when the result stops applying.
FORMULA states the expression; WHERE defines every symbol with its unit; CONSTRAINT lists the assumptions the algebra silently buys; NUMERICAL EXAMPLE substitutes ILLUSTRATIVE numbers whose arithmetic was recomputed for this sheet; INTERPRETATION says what a practitioner may and may not conclude.
The running example is a checkout-flow experiment with a binary primary outcome, because it is the smallest setting in which every later complication is visible.
Continuous outcomes, ratio outcomes and clustered outcomes are treated as variants in 4.3, and each is a modification of the base result rather than a new science.

**Symbol table used throughout Section 4.**

| Symbol | Meaning | Units / type | Valid range |
|---|---|---|---|
| `Z_i` | assignment indicator, 1 for treatment | boolean per unit | {0,1} |
| `Y_i` | observed binary outcome in the window | boolean per unit | {0,1} |
| `Y_i(1)`, `Y_i(0)` | potential outcomes under each assignment | boolean per unit | {0,1} |
| `n_T`, `n_C` | assigned units in treatment, control | count | ≥1 |
| `p_T`, `p_C` | true outcome probabilities by arm | proportion | (0,1) |
| `p_hat_T`, `p_hat_C` | observed arm conversion rates | proportion | [0,1] |
| `d` | absolute effect `p_T − p_C` | proportion (pp after ×100) | (−1,1) |
| `alpha` | two-sided type I error | probability | (0,1) |
| `beta` | type II error; power is `1 − beta` | probability | (0,1) |
| `z_q` | standard normal quantile at cumulative `q` | standard deviations | real |
| `k` | treatment share of assigned traffic | proportion | (0,1) |
| `rho` | correlation of outcome with a pre-period covariate | dimensionless | [−1,1] |
| `ICC` | intra-cluster correlation of the outcome | dimensionless | [0,1) |
| `m` | mean cluster size | count | ≥1 |

### 4.1 Base case

**R1 — The arm difference is unbiased for the assignment contrast.**

```text
FORMULA
  d_hat = p_hat_T - p_hat_C
        = (1/n_T) * SUM_{i: Z_i=1} Y_i  -  (1/n_C) * SUM_{i: Z_i=0} Y_i

  E[d_hat] = E[Y(1) | Z=1] - E[Y(0) | Z=0]
           = E[Y(1)] - E[Y(0)]        under independent assignment
```

WHERE: the expectation is taken over repeated randomisations of the same eligible population; `SUM` runs over assigned units only; every assigned unit contributes exactly one `Y_i`, which must be a 0 or 1 after reduction to unit grain.

CONSTRAINT: three things must hold at once. Assignment is independent of the potential-outcome pair. There is exactly one version of each arm, so a mid-flight redesign changes the treatment being compared. And a unit's outcome does not depend on other units' assignments, which excludes shared inventory, social graphs and competitive auctions.

NUMERICAL EXAMPLE: control converts 13,428 of 118,944 assigned units and treatment converts 13,700 of 119,412.
`p_hat_C = 13,428 / 118,944 = 0.1128935`; `p_hat_T = 13,700 / 119,412 = 0.1147288`.
`d_hat = 0.1147288 − 0.1128935 = 0.0018354`, which is `0.1835 pp`, and the relative lift is `0.0018354 / 0.1128935 = 1.626%`.

INTERPRETATION: that single number estimates what would happen to the whole eligible population if the organisation adopted the new assignment policy and did nothing else differently. It does not estimate the effect on users who completed the flow, because conditioning on completion is itself a post-assignment filter.

**Validity condition for R1:** independent assignment, one version per arm, no interference, and a complete unit-level outcome. Break any one and `d_hat` still has a mean, but it is no longer the mean of a causal contrast.

**R2 — The variance of the difference and its standard error.**

```text
FORMULA
  Var(d_hat) = p_T (1 - p_T) / n_T  +  p_C (1 - p_C) / n_C

  SE_hat(d_hat) = sqrt( p_hat_T (1-p_hat_T)/n_T + p_hat_C (1-p_hat_C)/n_C )

  z = d_hat / SE_hat(d_hat)
```

WHERE: `p_hat(1-p_hat)` is the estimated Bernoulli variance in that arm, divided by the number of independent assigned units in that arm; `z` is in standard-deviation units and is compared to normal quantiles only when each arm has enough events.

CONSTRAINT: the two arm means must be independent, which holds only if the randomisation unit equals the analysis unit and no unit appears in both arms. Each arm needs enough successes and enough failures for the normal approximation to the binomial; a rough working floor is at least 5 expected events of each type per arm.

NUMERICAL EXAMPLE: `0.1147288 × 0.8852712 / 119,412 = 8.5064e-7` and `0.1128935 × 0.8871065 / 118,944 = 8.4211e-7`.
Sum `= 1.69275e-6`, `SE_hat = 0.00130106`.
`z = 0.0018354 / 0.00130106 = 1.4108`, giving a two-sided `p = 0.158`.

INTERPRETATION: the point estimate is less than one and a half standard errors from zero, so this trial neither establishes a benefit nor establishes the absence of the effect it was designed to find. The interval, not the p-value, carries the decision information.

**Validity condition for R2:** independent units within each arm, stable arm sizes, Bernoulli outcomes, and a normal approximation that is not being asked to represent a distribution with mass piled on 0 or 1.

**R3 — The interval and what "95%" means.**

```text
FORMULA
  CI_95 = [ d_hat - z_{0.975} * SE_hat ,  d_hat + z_{0.975} * SE_hat ]
        = [ d_hat - 1.95996 * SE_hat , d_hat + 1.95996 * SE_hat ]
```

WHERE: `z_{0.975} = 1.95996` is the standard normal quantile; the interval is symmetric in the proportion scale, and multiplying both ends by 100 expresses it in percentage points without changing its meaning.

CONSTRAINT: coverage is a property of the procedure across repeated experiments, and it holds only if the standard error is approximately unbiased. Symmetric Wald intervals misbehave when `p_hat` is near 0 or 1 or when counts are small; 4.3 gives the replacements.

NUMERICAL EXAMPLE: `0.0018354 ± 1.95996 × 0.00130106` gives `[−0.0007145, +0.0043852]`, reported as `[−0.071 pp, +0.439 pp]`.
The same interval as a relative effect divides by `p_hat_C`: `[−0.63%, +3.88%]`.

INTERPRETATION: the data are compatible with a small loss and with a gain under half a percentage point, and they are not compatible with the eight-tenths-of-a-point gain the team hoped for. That last statement is a statement about the interval's upper end, and it is the strongest defensible claim from this experiment.

**Validity condition for R3:** the same independence and approximation conditions as R2, plus a standard error that reflects the actual design rather than the row count of an event table.

**R4 — How many units the design needs.**

```text
FORMULA
  n_per_arm = (z_{1-alpha/2} + z_{1-beta})^2 * [ p_0(1-p_0) + p_1(1-p_1) ] / (p_1 - p_0)^2

WHERE
  p_0 = control rate, p_1 = treatment rate under the alternative,
  z_{1-0.025} = 1.95996, z_{0.80} = 0.84162, so the squared sum is 7.84888.
```

WHERE: `n_per_arm` is independent assigned units per arm, not page views, sessions or events. The bracket is the sum of Bernoulli variances at the two rates, and the denominator is the squared absolute effect.

CONSTRAINT: a two-sided test at `alpha`, a single analysis at the end, a normal approximation, and a true effect exactly equal to `p_1 − p_0`. Allowing interim looks, adding arms, or clustering raises the requirement; 4.3 quantifies each increase.

NUMERICAL EXAMPLE: `p_0 = 0.112`, `p_1 = 0.120`, so `p_0(1-p_0) = 0.099456`, `p_1(1-p_1) = 0.105600`, and the bracket is `0.205056`.
`n_per_arm = 7.84888 × 0.205056 / 0.000064 = 25,147.8`, rounded up to `25,148`.
A 0.4 pp target instead of 0.8 pp gives `7.84888 × 0.202 / 0.000016 = 99,092` per arm.

INTERPRETATION: halving the effect doubles the effect in the denominator squared, which is why the requirement goes from roughly twenty-five thousand to roughly ninety-nine thousand per arm for a change from 0.8 pp to 0.4 pp. Sample size is governed by the square of the smallest effect worth detecting, not by the traffic the team happens to have.

**Validity condition for R4:** one primary outcome, one analysis, no unplanned peeking, and a baseline rate estimated from a period that resembles the experiment window.

### 4.2 General case

**R5 — Minimum detectable effect, the inverse of R4.**

```text
FORMULA
  MDE(n) = (z_{1-alpha/2} + z_{1-beta}) * sqrt( (p_0(1-p_0) + p_1(1-p_1)) / n_per_arm )

  with p_1 = p_0 + MDE, solved by iteration, or approximated by
  MDE(n) ~ (z_{1-alpha/2} + z_{1-beta}) * sqrt( 2 p_0 (1-p_0) / n_per_arm )
```

WHERE: `MDE` returns an absolute difference in proportion units; multiply by 100 for percentage points and divide by `p_0` for the relative version. The approximation on the third line sets both arms' variances to the control variance and is accurate to a few per cent when the effect is small relative to `p_0`.

CONSTRAINT: the same design conditions as R4, plus the interpretation that MDE is the effect detected with probability `1 − beta`, not a threshold below which effects do not exist. An experiment can produce a significant result smaller than its MDE by chance and cannot produce a reliable estimate of an effect far below it.

NUMERICAL EXAMPLE: at `n = 118,944` per arm, `sqrt(2 × 0.112 × 0.888 / 118,944) = sqrt(1.6723e-6) = 0.0012932`.
`MDE = 2.8016 × 0.0012932 = 0.003622`, that is `0.362 pp` or `3.23%` relative.
At the planned `n = 99,092` the same arithmetic gives `0.004001`, matching the 0.4 pp target the design was built around.

INTERPRETATION: this trial could reliably detect a 0.36 pp move, so an observed interval ending at +0.44 pp is genuinely informative: the design was capable of finding the effect the team wanted and did not.

**Validity condition for R5:** the achieved per-arm count after exclusions must be used, not the planned count; a difference of a few per cent in `n` changes the MDE by about half as many per cent.

**R6 — Unequal allocation.**

```text
FORMULA
  N_total = (z_{1-alpha/2} + z_{1-beta})^2 * [ p_0(1-p_0)/(1-k) + p_1(1-p_1)/k ] / (p_1-p_0)^2

  with n_T = k * N_total and n_C = (1-k) * N_total
```

WHERE: `k` is the share of eligible traffic assigned to treatment; `1-k` is the control share. The brackets weight each arm's variance by the reciprocal of its share.

CONSTRAINT: `k` must be constant over the enrollment window, or the analysis must condition on enrollment phase. A ramp that changes `k` mid-flight makes the pooled arm rates incomparable to any single planned calculation.

NUMERICAL EXAMPLE: for the 0.8 pp target at `p_0 = 0.112`, `k = 0.5` needs 50,296 assigned units in total.
`k = 0.7` needs `7.84888 × (0.099456/0.3 + 0.1056/0.7) / 0.000064 = 59,158`, split 41,411 treatment and 17,747 control.
`k = 0.9` needs `136,361` in total, of which 122,725 are treated and only 13,636 are control.

INTERPRETATION: starving the control arm is expensive. Going from a 70/30 split to a 90/10 split more than doubles the total traffic requirement while leaving the control estimate so noisy that guardrail comparisons lose power. A 90/10 split is defensible only when the treatment is already believed safe or when the cost of exposing users to control is itself a business harm.

**Validity condition for R6:** the allocation share is fixed and recorded per unit; if the platform ramps, the plan must be re-derived at the final share using the total across all phases.
