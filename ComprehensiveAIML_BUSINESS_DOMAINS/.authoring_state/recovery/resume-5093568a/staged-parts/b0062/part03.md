**R7 — Multiple arms and the multiplicity cost.**

```text
FORMULA
  family-wise alpha split: alpha_c = alpha / (A - 1)          (Bonferroni, A arms vs control)

  n_per_arm(A) = (z_{1-alpha_c/2} + z_{1-beta})^2 * [p_0(1-p_0) + p_1(1-p_1)] / (p_1-p_0)^2
```

WHERE: `A` is the total number of arms including control, so `A − 1` comparisons are tested against the shared control; `alpha_c` is the per-comparison error rate that bounds the family error at `alpha`.

CONSTRAINT: Bonferroni controls the family-wise error under any dependence between the comparisons, which is why it is safe but conservative. It assumes each arm is independently randomised; a factorial design tests interactions on top and needs its own accounting.

NUMERICAL EXAMPLE: with `A = 3`, `alpha_c = 0.025`, so `z_{1-0.0125} = 2.2414` and the quantile sum becomes `2.2414 + 0.84162 = 3.08302`, whose square is `9.5050`.
`n_per_arm = 9.5050 × 0.205056 / 0.000064 = 30,454` per arm, so the three-arm design needs `91,362` assigned units against `50,296` for a two-arm design with the same target effect.

INTERPRETATION: the second treatment arm costs about 82% more total traffic than the two-arm design, not 50%. If the third arm is exploratory, the cheaper design is to run a second experiment sequentially and keep the family at size one.

**Validity condition for R7:** the shared control arm is not reused across experiments in a way that lets the same units be counted twice, and each arm's assignment is independent of the others.

**R8 — Sequential monitoring with alpha spending.**

```text
FORMULA  (Lan–DeMets alpha spending for an O'Brien–Fleming shape)
  alpha*(t) = 2 * [ 1 - Phi( z_{1-alpha/2} / sqrt(t) ) ]
  reject at look j if |Z_j| >= c_j, where c_j solves
  P_{H0}( |Z_1| < c_1, ..., |Z_{j-1}| < c_{j-1}, |Z_j| >= c_j ) = alpha*(t_j) - alpha*(t_{j-1})
```

WHERE: `t_j = n_j / n_max` is the information fraction at look `j`; `Z_j` is the standardised effect using all data up to that look; `c_j` are the boundary values solved sequentially from a multivariate normal recursion over the joint distribution of the `Z_j`.

CONSTRAINT: the number and timing of looks must be fixed in advance, and the total information `n_max` must be fixed, otherwise the spending function has no scale. The correlation between successive `Z_j` is `sqrt(t_i/t_j)`, which is what makes the recursion tractable and what makes naive per-look testing wrong.

NUMERICAL EXAMPLE: for four equally spaced looks at `t = 0.25, 0.50, 0.75, 1.00` with `alpha = 0.05`, the O'Brien–Fleming boundaries are approximately `4.333, 2.963, 2.359, 2.014`; a simulation of 400,000 Brownian paths crossing at those boundaries produced a type I error of `0.0499`.
Pocock-shaped flat boundaries for four looks require `c ≈ 2.355`; the same simulation at `c = 2.0` produced `0.115`, which is the cost of ignoring the looks entirely.

INTERPRETATION: the final-look boundary of 2.014 is barely more demanding than the fixed-sample 1.96, but the early boundaries are severe on purpose. That shape lets an early stop happen only for an effect so large that it survives the penalty, which is what a safety-driven stop needs.

**Validity condition for R8:** the analysis must be executed at the planned looks with the planned boundary; running the sequential test only when a dashboard looks good converts it back into an unadjusted peek.

### 4.3 Variants

**R9 — Variance reduction with a pre-period covariate (CUPED).**

```text
FORMULA
  theta = Cov(Y, X) / Var(X)
  Y_adj,i = Y_i - theta * (X_i - mean(X))
  d_hat_adj = mean(Y_adj | Z=1) - mean(Y_adj | Z=0)

  Var(d_hat_adj) / Var(d_hat) = 1 - rho^2
```

WHERE: `X_i` is a covariate measured strictly before assignment, on the same unit scale as the outcome or any linear transform of it; `rho` is the correlation between `Y` and `X`; `theta` is estimated from the pooled experiment data.

CONSTRAINT: `X` must be pre-treatment and must not be missing differentially by arm. Subtracting `theta * (X_i − mean(X))` removes variance but leaves the expected difference unchanged only because `mean(X)` is common to both arms; if `X` were treatment-affected, the adjustment would remove part of the effect itself.

NUMERICAL EXAMPLE: the plan needs 25,148 units per arm at `rho = 0`, so `Var(d_hat_adj)/Var(d_hat) = 1`.
At `rho = 0.45`, `1 − rho^2 = 0.7975` and the requirement falls to `25,148 × 0.7975 = 20,056` per arm, a 20% saving.
At `rho = 0.75`, `1 − 0.5625 = 0.4375` and the requirement falls to `11,002` per arm, a 56% saving.

INTERPRETATION: a good pre-period covariate is the cheapest sample size the team will ever buy, because it costs a join and no additional traffic. The saving is `rho^2`, so a covariate correlated at 0.2 with the outcome buys 4% and is not worth the operational risk.

**Validity condition for R9:** `X` is measured before assignment for every analysed unit, missingness is arm-independent, and `theta` is estimated from data that does not include treatment-induced variation in `X`.

**R10 — Clustered assignment and the design effect.**

```text
FORMULA
  DEFF = 1 + (m - 1) * ICC

  n_effective = n_units / DEFF
  clusters_per_arm = n_units * DEFF / m
```

WHERE: `m` is the mean number of analysis units per randomised cluster, `ICC` is the intra-cluster correlation of the outcome, and `n_units` is the number of analysis units (users, sessions) that the naive formula would have used.

CONSTRAINT: `ICC` must be estimated on comparable data, and `m` must be roughly constant; highly variable cluster sizes inflate the effective `m` and make the simple formula optimistic. The formula also assumes an exchangeable correlation structure within clusters, which fails in organisations with strongly differentiated roles.

NUMERICAL EXAMPLE: with 25 users per organisation and `ICC = 0.06`, `DEFF = 1 + 24 × 0.06 = 2.44`.
The 25,148 users per arm therefore behave like `25,148 / 2.44 = 10,307` independent users, and the design needs `25,148 × 2.44 / 25 = 2,454` organisations per arm.
If the analysis is run at the user level with 118,944 users per arm in 4,758 organisations, the effective sample is `118,944 / 2.44 = 48,747` users and the standard error is `sqrt(2.44) = 1.56` times too small.

INTERPRETATION: a user-level standard error in a cluster-randomised experiment is wrong by the square root of the design effect, which here turns a `z` of 3.8 into a `z` of 2.4. The point estimate can be right while the confidence interval is a fiction.

**Validity condition for R10:** the analysis variance must be computed at the randomisation grain, or a cluster-robust or cluster-bootstrap variance must be used; a mixed model works only if its random-effects structure matches the assignment structure.

**R11 — Ratio metrics and the delta method.**

```text
FORMULA
  R = SUM_i R_i / SUM_i S_i               (ratio of totals, e.g. revenue per session)

  Var(R_hat) ~ ( Var(R) - 2 R Cov(R,S) + R^2 Var(S) ) / ( n * mean(S)^2 )

  where Var and Cov are unit-level moments and n is the number of independent units
```

WHERE: `R_i` is the unit's numerator (revenue) and `S_i` its denominator (sessions); the moments are taken over units, not over sessions.

CONSTRAINT: units must be independent and the denominator must not cross zero at the unit level for the estimand to be defined. The delta method is a first-order Taylor expansion and degrades when the denominator's coefficient of variation is large.

NUMERICAL EXAMPLE: with `mean(S) = 4.2` sessions per user, `sd(S) = 6.1`, `mean(R) = $8.80145`, `sd(R) = $42`, `corr(R,S) = 0.55` and `n = 118,944`, the ratio is `2.09558` and the covariance term is `0.55 × 42 × 6.1 = 140.91`.
`Var = (1764 − 2 × 2.09558 × 140.91 + 4.39145 × 37.21) / (118,944 × 17.64) = 0.000637`, so `SE = 0.02524` and the relative standard error is `1.2%`.
Omitting the covariance term gives `0.03031`, inflating the standard error by 20% and discarding precision the data actually support.

INTERPRETATION: whether the covariance term is positive or negative matters more than its sign suggests: here it is favourable, and ignoring it makes the test look weaker than it is. The direction of the error is not knowable in advance, which is why ratio metrics need their variance derived rather than borrowed.

**Validity condition for R11:** independence at the unit grain, a stable unit definition for `S`, and enough units with nonzero denominators that the first-order expansion is not dominated by its second-order remainder.

**R12 — Small counts and zero cells.**

```text
FORMULA  (Clopper–Pearson exact interval for x successes in n trials)
  lower = Beta^{-1}(alpha/2; x, n - x + 1)        for x > 0, else 0
  upper = Beta^{-1}(1 - alpha/2; x + 1, n - x)    for x < n, else 1

  special case x = 0:  upper = 1 - (alpha/2)^(1/n)
```

WHERE: `Beta^{-1}` is the inverse beta distribution function; the interval is conservative in the sense that coverage is at least nominal.

CONSTRAINT: the exact interval assumes Bernoulli trials with a constant probability within the arm, which is itself an assumption; a heterogeneous population makes the exact interval slightly wider than necessary but does not break it. The formula gives no protection against the population being the wrong one.

NUMERICAL EXAMPLE: `3/400` gives `[0.001549, 0.021761]`; a Wald interval on the same data gives `[−0.00096, 0.01596]`, which extends below zero and is impossible for a proportion.
`0/50` gives `[0, 1 − 0.025^(1/50)] = [0, 0.0711]`; the Wald interval collapses to `[0, 0]`, which claims certainty from fifty observations.

INTERPRETATION: for a payment-failure guardrail with a handful of failures, report the exact interval and let the width do the arguing. A zero-width interval on a small sample is a defect, not a result.

**Validity condition for R12:** the counts are unit-level successes and trials with no duplicate units; with duplicated units the denominator is wrong and the exact interval inherits the error.

**R13 — Dilution from partial exposure.**

```text
FORMULA
  d_assigned = pi * d_exposed + (1 - pi) * d_unexposed

  with d_unexposed = 0:  d_exposed = d_assigned / pi
```

WHERE: `pi` is the fraction of assigned units in the treatment arm who are exposed; `d_exposed` is the effect among exposed units; `d_unexposed` is the effect among units who never reach the changed surface.

CONSTRAINT: the decomposition requires that the unexposed fraction be unaffected by treatment, or that its effect be genuinely zero. Both parts are assumptions about product behaviour, not consequences of randomisation.

NUMERICAL EXAMPLE: an effect of `0.8 pp` per assigned unit with `pi = 0.30` implies `0.8 / 0.30 = 2.67 pp` among exposed units, if the unexposed truly cannot respond.
If the treatment also raises exposure probability — here exposure is `74.2%` in control and `69.1%` in treatment, a difference of `−5.13 pp` with `z = −27.8` — then exposure is a treatment-affected variable and the decomposition shifts the estimand.

INTERPRETATION: dividing the assigned effect by the exposure rate is a diagnostic, not a licence to report the per-exposed number as the headline. When exposure itself moves, the per-exposed effect mixes the treatment's effect on who arrives with its effect on what they experience.

**Validity condition for R13:** exposure status is measured by an event that precedes the outcome, is not differentially missing by arm, and the unexposed contrast is credibly zero.
