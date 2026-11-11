
## 9. Complexity & Cost Analysis

### 9.1 Where the money and the time actually go

An experiment has four cost lines, and the largest one is almost never compute.

```text
cost line                     typical driver                                   illustrative monthly order
traffic opportunity cost      delay of a real win, or exposure to a bad change   dominant
engineering                   build, flag, instrument, analyse, ship              large
platform/compute              assignments, event volume, warehouse scans          small
organisational                reviews, disagreements, re-runs                     medium
```

The illustrative numbers below are constructed to show orders of magnitude, not a vendor quote.
Unit cost of 0.0009 USD per assignment covers flag evaluation, event logging, warehouse storage and
the analysis job; engineering at 3,200 USD per week.

```text
assignments in the paywall design        168,632
platform cost  168,632 * 0.0009       = $151.77
engineering    6 weeks * 3,200         = $19,200
total run cost                         = $19,351.77
```

The value side, ILLUSTRATIVE: 1.2 million visitors per month, an observed +0.5701 pp on trial start,
and a constructed $41 of expected subscription value per started trial.

```text
monthly trial starts attributable = 1,200,000 * 0.005701 = 6,841
monthly value                     = 6,841 * $41          = $280,481
break-even on the $19,352 run     = 19,352 / 280,481     = 0.069 months  (about 2 days)
```

Two consequences follow, and both change how you argue for experiment rigour.

First, the cost of *not* running the test is not the platform bill; it is the expected loss from
shipping a neutral or harmful change at full volume. If a change with a true effect of zero ships to
1.2M visitors, the loss is the implementation cost plus the opportunity cost of the next idea, and
if the change is mildly harmful the loss scales with volume.

Second, the marginal cost of *more* traffic in an online product is near zero, so the binding
constraint is calendar time, not sample size. A design that needs 42,158 per arm and gets 26,069 per
day finishes in 3.2 days; the same design at 5,000 per day takes 16.9 days, and the decision delay
costs more than the analysis.

### 9.2 Analysis-time complexity

```text
operation                                  complexity                paywall example
read assignment + event tables             O(rows scanned)           365k users, 1.2M events
per-user aggregation                       O(n)                      182,400 + 182,560 rows out
difference in proportions                  O(1)                      negligible
bootstrap CI at assignment unit            O(B * n)                  B=1,000 -> 365M row-draws
CUPED regression                           O(n * p)                  p=1-5 covariates
segment scan (S segments, M metrics)       O(S * M) tests            S=40, M=100 -> 4,000 tests
pre-period join for CUPED                  O(pre-rows)               98.5M rows for 90 days at 3/day
```

The only line that hurts at scale is the bootstrap over 365M row-draws, and it is worth it exactly
when clustering or a ratio metric makes the closed-form SE suspect. A middle path is a block
bootstrap over users with `B = 200`, which is enough to check whether the analytic SE is right to
within about 10%.

The segment scan is where multiplicity arrives without anyone deciding to introduce it. Four
thousand tests at alpha 0.05 produce 200 expected false positives, and a dashboard that sorts by
significance will show a headline built entirely from those 200.

### 9.3 Cost of the four precision levers

```text
lever                        mechanism                       cost                            when to use
more traffic                 run longer                      calendar delay                  default
better estimator (CUPED)      rho^2 variance reduction        pre-period data + pipeline      rho >= 0.3
coarser metric                fewer rare events               decision fidelity               base rate < 2%
design (stratify/pair)        remove between-stratum var      assignment complexity           strong pre-strata
```

Compute the exchange rate explicitly, because it is the argument that wins design reviews. In the
paywall example, `rho = 0.55` from prior-90-day trial activity cuts the required sample from 42,158
to 29,405 per arm, which is 1.43x the effective traffic — the same as adding 43% more visitors. The
cost is one join and one regression, roughly two engineer-days.

The coarser-metric lever is riskier because it changes the estimand. Moving from "trial start" to
"any paywall CTA click" multiplies the event rate by perhaps 4x and divides the required sample by
roughly 4, but the decision then rests on a proxy whose relationship to revenue is assumed, not
measured. Use it for screening many variants and confirm the winner on the real metric.

### 9.4 The cost of a wrong decision

A compact way to put numbers on Type I and Type II costs, which is what turns a statistical argument
into a product argument:

```text
expected cost of shipping a neutral change  = P(ship | neutral) * ship cost
expected cost of rejecting a good change    = P(reject | good) * value forgone

ILLUSTRATIVE, with ship cost $19,352 one-off and value $280,481 per month:
  false positive: pay $19,352 and forgo the next idea for ~2 weeks
  false negative: forgo ~$140,245 of half a month's value
ratio of costs ≈ 7.2 : 1 in favour of the false negative being worse
```

That ratio justifies 80% power over 95%, and it is the honest reason a 0.06 alpha or a 0.90 power is
sometimes appropriate. It does not justify a low alpha *and* low power, which is a design that costs
time and learns nothing.

### 9.5 Duration arithmetic, including the maturity tail

```text
n per arm                    42,158          design minimum
total assignments            84,316
eligible per day             26,069
assignment days             84,316 / 26,069         = 3.23
measurement window (14 d)   +14
calendar duration           ceil to whole weeks    = 21 days (3 weeks)
```

Rounding to whole weeks is not bureaucratic tidiness: partial weeks mix weekday and weekend
behaviour, and a cohort that is 60% weekend-heavy can move a conversion metric by several percent.
The paywall example's real run at 182,400 per arm took 14 days of assignment plus a 14-day maturity
tail, and the readout on day 14 (assignments complete) was preliminary.

## 10. Key Theorems / Results Reference

**T1 — Two-sample sample size (normal approximation).** For independent arms, the required `n` per
arm for power `1-beta` at two-sided `alpha` is
`(z_(1-alpha/2) + z_(1-beta))^2 * (sigma_1^2 + sigma_2^2) / delta^2`. Validity: independent units,
single analysis, known variances, `n * p >= 10` for proportions. This is the base case in §4.1.

**T2 — Variance reduction from a pre-period covariate.** For `Y_adj = Y - theta * (X - E[X])` with
`theta = Cov(Y,X)/Var(X)`, `Var(Y_adj) = Var(Y)(1 - rho^2)`. Validity: `X` measured before
randomisation; `theta` estimated consistently; linear relationship adequate. Equality holds for the
population `theta`; the sample-estimated version adds one degree of freedom of noise, negligible
above a few hundred units.

**T3 — Delta method for a ratio.** For `R = Y/X`, `Var(R) ~= Var(Y)/mu_X^2 + mu_Y^2 Var(X)/mu_X^4 -
2 mu_Y Cov(X,Y)/mu_X^3`. Validity: `mu_X` bounded away from zero, `X` concentration sufficient that
the first-order expansion is adequate (coefficient of variation of `X` below about 0.3).

**T4 — Design effect for clustering.** With equal cluster size `m` and intra-cluster correlation
`ICC`, `Var(mean) = sigma^2 * (1 + (m-1) ICC) / (K * m)` where `K` is the number of clusters.
Validity: exchangeable clusters, equal sizes; unequal sizes require a size-weighted ICC.

**T5 — Inflated Type I under repeated looks.** For a fixed boundary `c` applied at `K` looks on a
Brownian path, the crossing probability exceeds `alpha` and increases with `K`; simulation at
`c = 1.96` two-sided gives 0.050, 0.142, 0.194, 0.244 for `K = 1, 5, 10, 20` equally spaced looks.
Validity: independent increments, continuous monitoring, no alpha spending.

**T6 — Alpha spending boundary.** A boundary `c / sqrt(t)` applied at information fractions `t_k`
with `c` calibrated so total crossing probability under the null equals alpha. Calibrated values from
simulation: `c = 1.979` (K=2), `2.042` (K=5), `2.086` (K=10), giving final-look boundaries of
`1.979, 2.042, 2.086`. Validity: information fractions known and honoured; no unplanned looks.

**T7 — ITT to TOT conversion.** Under randomisation, exclusion restriction and monotonicity,
`TOT = ITT / P(exposed | assigned)`. Validity: the instrument (assignment) affects the outcome only
through exposure; no defiers. A 78% exposure share maps 0.5701 pp to 0.731 pp.

**T8 — Bonferroni bound.** For `k` tests, rejecting each at `alpha/k` controls family-wise error at
`alpha`. Conservative under positive dependence; exact under independence. Twelve tests at 0.05
require `p <= 0.00417` each, a `z` threshold of 2.86.

**T9 — Holm step-down.** Order p-values ascending and compare the `i`-th to `alpha/(k-i+1)`,
stopping at the first failure. Uniformly more powerful than Bonferroni and still controls family-wise
error under arbitrary dependence.

**T10 — Beta-Binomial conjugacy.** With `Beta(a, b)` prior and `x` successes in `n` trials, the
posterior is `Beta(a+x, b+n-x)`; the posterior mean is `(a+x)/(a+b+n)` and the posterior predictive
for the next `m` trials is Beta-Binomial. Validity: exchangeable trials; prior stated before
unblinding.

**T11 — Sample ratio mismatch test.** Under a correct allocation, assigned counts follow a
multinomial with planned probabilities; the chi-square statistic has `k-1` degrees of freedom. `k=2`
gives one degree of freedom, so `chi2 = 58.8` corresponds to `p < 1e-13`. Validity: counts taken
before any filtering; independence of assignments.

**T12 — Winsorisation trade-off.** Replacing the top `q` fraction of values with the `q`-quantile
reduces variance by roughly the tail's contribution but changes the estimand to a trimmed mean. The
bias is `E[Y | Y > Q] - Q` times `q`, which is exactly the quantity the decision may care about;
report both the trimmed and untrimmed effect.

## 11. Interview / Exam Questions

**Q1.** A test planned 50/50 has 152,100 control and 147,900 treatment assignments. A colleague says
the 2.8% difference is fine because "randomness". What is the chi-square statistic and what should
you do?

*Answer.* `E = 150,000` per arm. `chi2 = (2,100^2)/150,000 * 2 = 29.4 * 2 = 58.8` on one degree of
freedom, `p < 1e-13`. Do not analyse the metric. First re-run the test on pre-filter assignment
counts: if balanced, a post-assignment filter is the cause; if not, the bucketing service is. Then
fix and restart. *Reasoning:* at 300,000 assignments the standard error of the split is about 274
units, so a 4,200-unit gap is 15 standard errors — this is a mechanism, not luck. *Follow-up:* what
imbalance would be consistent with chance? *Red flag:* "the metric looks fine so the split does not
matter". *Scoring:* names the statistic, the degrees of freedom, and refuses to analyse.

**Q2.** Your design targets a 6% relative lift on a 9.6% base rate at 80% power. How many per arm,
and what happens to that number if the true base rate turns out to be 8.0%?

*Answer.* `n = 42,158` per arm. At `p1 = 0.080`, `p2 = 0.0848` and `d = 0.0048`, the same formula
gives 51,515 per arm — 22.2% more, not less. The reason is that at a fixed *relative* lift the
absolute effect shrinks proportionally to `p1` while the variance shrinks only as `p1(1-p1)`, so
`n` scales roughly with `(1-p1)/p1`: from 9.42 at `p1 = 0.096` to 11.5 at `p1 = 0.080`, a ratio of
1.22 that matches the computed 51,515/42,158 = 1.222. *Reasoning:* both numerator and denominator of
the effect move with the base rate, but not at the same rate. *Follow-up:* what happens at
`p1 = 0.048`? (88,942 per arm, 2.11x — the `(1-p)/p` scaling is now severe.) *Red flag:* "half the
base rate means half the sample". *Scoring:* computes both and identifies the `(1-p)/p` scaling.

**Q3.** You observe +0.57 pp with `p = 7e-9`. The product manager wants to know the probability the
change is actually better. What do you say?

*Answer.* The p-value does not answer that. Under flat `Beta(1,1)` priors the paywall data give
`P(p_T > p_C) = 1 - 1e-15` and the small 2,400-per-arm experiment gives 0.9585; both depend on the
prior, and the second one ships a change whose frequentist test has `p = 0.081`. What you can say
without a prior: the 95% interval on the relative effect is `[3.92%, 8.10%]`, so relative effects
below 3.92% are inconsistent with these data at that level. *Reasoning:* distinguish the
sampling-theory statement (interval coverage) from the belief statement (posterior mass), and note
the posterior is a function of the prior, not a property of the data alone. *Follow-up:* which prior
would push the posterior probability below 0.9 in the paywall case? *Red flag:* "there is a
99.99999% chance it works". *Scoring:* separates the two questions and refuses to conflate them.

**Q4.** A metric is revenue per session. Per-user aggregates give `mu_Y = 24.0`, `sd_Y = 26.0`,
`mu_X = 6.0`, `sd_X = 5.5`, `corr = 0.20`, `n = 20,000` per arm. What is the standard error, and
what does the naive formula give?

*Answer.* `R = 4.0`. Correct variance per unit is
`(26^2 + 16*5.5^2 - 2*4*0.20*5.5*26)/6^2 = (676 + 484 - 228.8)/36 = 931.2/36 = 25.867`, so
`SE = sqrt(25.867/20,000) = 0.0360`. Naive `SE = 26/(6*sqrt(20,000)) = 0.0306`, understating by
17.4%. *Reasoning:* the covariance term is 228.8 out of 931.2, a quarter of the variance, and it
matters. *Follow-up:* what correlation would make the naive formula conservative? *Red flag:*
reporting the naive SE and moving on. *Scoring:* computes the covariance term and states the
direction of the error.

**Q5.** You have 25,000 assignments per arm available in the window and a base rate of 2.1%. Your
ship threshold is a 10% relative lift. Is the test worth running?

*Answer.* The MDE at `n = 25,000` and `p = 0.021` is 0.375 pp, or 17.8% relative — larger than the
10% threshold (0.21 pp). Power at a true 0.21 pp effect is 0.359, so a real 10% improvement would be
missed about two times in three. Closing the gap needs 76,757 per arm (a 3.07x traffic increase) or a
coarser metric or a holdback with a months-long readout. *Reasoning:* MDE scales as `1/sqrt(n)`, so
closing a 1.78x gap in effect needs `1.78^2 = 3.17x` the sample. *Follow-up:* what sample would give
80% power at the threshold? *Red flag:* "let's just run it and see". *Scoring:* computes the MDE and
compares it to the threshold rather than to zero.

**Q6.** You checked the dashboard daily for two weeks and stopped the day the effect crossed
`p < 0.05`. What is the problem and what error rate did you actually run?

*Answer.* With a fixed 1.96 boundary and 14 looks, the simulated Type I rate is between the 0.194
(10 looks) and 0.244 (20 looks) points — call it roughly 0.22, four times the nominal 0.05. The fix
is either a pre-registered single look or an alpha-spending boundary computed for the planned number
of looks: at `K = 5`, the final-look boundary is 2.042, and the first look requires `z = 4.566`.
*Reasoning:* the walk's spread grows as `sqrt(t)` while the boundary is fixed, so crossing
probability accumulates. *Follow-up:* what is the cost of the correction in sample size? *Red flag:*
"but it was significant on the final day too". *Scoring:* quantifies the inflation and names the
correction.

**Q7.** Your ITT effect is 0.5701 pp and 78% of the treatment arm was exposed. What is the TOT effect
and what assumption does it need?

*Answer.* `TOT = 0.5701/0.78 = 0.731 pp`, or 7.68% relative to the 9.5197% control rate. It requires
the exclusion restriction (assignment affects the outcome only through exposure) and no defiers.
*Reasoning:* ITT is a weighted average of the exposed effect and a zero effect for the unexposed, so
dividing by the exposure share undoes the dilution. *Follow-up:* if exposure itself is caused by the
treatment (the treatment makes people more likely to reach the trigger), why is the ratio still
valid? *Red flag:* "the effect on exposed users is the real effect" without the assumption.
*Scoring:* computes both numbers and states both assumptions.

**Q8.** A guardrail shows 214/17,364 control versus 229/18,420 treatment refunds. Can you declare
the guardrail safe?

*Answer.* No. The difference is `+0.0108 pp` with `SE = 0.00117`, so `z = 0.09` — indistinguishable
from zero, and the MDE at this sample is 0.348 pp (28% relative). A true harmful increase of 0.10 pp
would be detected only 21% of the time. Either size the guardrail properly (51,593 per arm for
0.20 pp) or use a different assurance mechanism such as a longer holdback. *Reasoning:* absence of
evidence at an underpowered test is not evidence of absence, and the MDE quantifies how much absence
is being asserted. *Follow-up:* what minimum harmful effect would the business accept unmeasured?
*Red flag:* "not significant, so it is safe". *Scoring:* computes the MDE and refuses the safety
claim.

**Q9.** Twelve metrics were tested at alpha 0.05 and one moved at `p = 0.03`. What is the corrected
conclusion?

*Answer.* The per-test threshold under Bonferroni is `0.05/12 = 0.00417`, equivalent to `z = 2.86`;
`p = 0.03` does not clear it. Under Holm the smallest p-value is compared to `0.00417` as well, so
the same conclusion. The probability of at least one false positive across 12 independent tests is
`1 - 0.95^12 = 0.4596`, so a single nominal `p = 0.03` among twelve is unremarkable. *Reasoning:*
the family is the twelve tests, not the one that moved. *Follow-up:* when is a family-wise correction
the wrong choice? (When the metrics are a pre-registered hierarchy and only the primary gates
shipping.) *Red flag:* "but 0.03 is significant". *Scoring:* applies a correction and states the
family.

**Q10.** Cluster randomisation with 400 clusters per arm, 50 users per cluster, and `ICC = 0.02`.
What is the effective sample size and what else changes?

*Answer.* `DEFF = 1 + 49*0.02 = 1.98`, so 20,000 users behave like 10,101. Beyond that, the
comparison is at the cluster level with about 399 degrees of freedom — fine — but if there were 20
clusters per arm the critical value would be near 2.09 rather than 1.96, and one atypical cluster
could dominate. *Reasoning:* the design effect is about within-cluster correlation times cluster size,
and the degrees of freedom are about the number of clusters, not users. *Follow-up:* why does
increasing cluster size past a point stop helping? *Red flag:* "20,000 users is plenty".
*Scoring:* computes DEFF and mentions the degrees-of-freedom consequence.

**Q11.** The primary is flat, a secondary moved at `p = 0.04`, and the team wants to ship. What do
you check first?

*Answer.* Whether the secondary was pre-registered and whether its MDE is small enough that a 0.04
is interpretable. Then whether it is downstream of the primary (a moved secondary with a flat
primary often means the secondary is picking up the treatment's mechanism rather than value). If it
was not pre-registered, the defensible path is a new test with the secondary as primary, not a ship
decision. *Reasoning:* switching the decision metric after seeing the data converts the secondary's
error rate from controlled to uncontrolled. *Follow-up:* how would you design that second test
cheaply? *Red flag:* "the data support it either way". *Scoring:* names the pre-registration
violation and proposes the re-test.

**Q12.** Session-level analysis of a visitor-level randomisation gives `SE = 0.000647` and
`z = 8.81`, versus `SE = 0.000984` and `z = 5.79` at visitor level. Explain the difference and say
which one you report.

*Answer.* Sessions within a visitor are correlated, so the session-level sample contains fewer
independent observations than its row count suggests. The SE ratio, 1.52, is `sqrt(DEFF)`, giving
`DEFF = 2.31` — consistent with about 3.4 sessions per visitor and an intra-visitor ICC near 0.55.
Report the visitor-level analysis, which matches the randomisation unit; the session-level number is
wrong, not more precise. *Reasoning:* the variance formula assumes independent units and sessions
are not. *Follow-up:* what would justify session-level randomisation? *Red flag:* "more rows, more
power". *Scoring:* computes the design effect and reports the assignment-unit analysis.

**Q13.** Your treatment has a `rho = 0.55` pre-period covariate available. How much traffic does
CUPED save, and what could go wrong?

*Answer.* `Var` factor `1 - 0.55^2 = 0.6975`, so `42,158 * 0.6975 = 29,405` per arm, a 30% saving
equivalent to 1.43x the traffic. What goes wrong: using a covariate measured after assignment
(post-treatment bias), estimating `theta` on the same data that decides the test without accounting
for it (usually negligible but state it), and a covariate with a non-linear relationship where the
linear adjustment does not reduce variance. *Reasoning:* the adjustment removes the part of `Y` that
the pre-period predicts, and that part is balanced by randomisation already, so the estimand is
unchanged. *Follow-up:* why must `E[X]` be the pooled mean rather than the arm mean? *Red flag:*
"adjust for anything correlated with the outcome". *Scoring:* computes the factor and names
post-treatment bias.

**Q14.** You need a 14-day conversion window but you hit your sample size on day 10. Why can't you
stop?

*Answer.* The last cohort's 14-day window is still open, so its conversion count is incomplete while
earlier cohorts' are complete. If treatment accelerates conversion (a shorter time-to-trial), the
immature cohort is underweighted in the treatment arm relative to control, biasing the effect; if
treatment delays conversion, the bias reverses. Either wait `n/m + w` days or analyse only fully
mature cohorts and accept the reduced sample. *Reasoning:* the measurement window is part of the
design and truncating it is a form of differential censoring. *Follow-up:* how would you detect the
bias after the fact? *Red flag:* "we hit the number, we're done". *Scoring:* explains the censoring
mechanism and gives the `+ w` rule.

**Q15.** Your effect is significant and the CI is `[+0.0002, +0.0114]`. Ship cost is five
engineer-weeks. What is the decision?

*Answer.* Not enough information to ship; enough to know the effect is probably positive and probably
small. The lower bound, 0.02 pp, is far below the 0.4 pp needed to cover five engineer-weeks at
plausible volume, so the expected value of shipping is negative even though `p < 0.05`. The correct
output is "positive but below the practical threshold; revisit if the cost falls or the volume
rises". *Reasoning:* the decision is `E[effect] * volume * margin - cost`, not `p < 0.05`.
*Follow-up:* how would you re-scope the change to lower the cost? *Red flag:* "significant, ship
it". *Scoring:* separates statistical from practical significance and reasons about cost.

**Q16.** Why does changing the analysis unit from user to session raise `z` from 5.79 to 8.81 while
raising the risk of a wrong decision? Give the mechanism, not the slogan.

*Answer.* The `z` rises because the denominator of the SE uses a larger nominal `n` (431,900 instead
of 182,560), and the SE scales as `1/sqrt(n)`. But the outcomes of sessions belonging to one visitor
are positively correlated: a visitor who starts a trial does so in exactly one session, so the
per-session indicator is a rare event clustered within visitors. The effective sample size is closer
to the number of visitors, the naive SE is too small by `sqrt(DEFF) = 1.52`, and the test is
anti-conservative. *Reasoning:* independent-observation assumption violated, and the direction of the
error is predictable. *Follow-up:* name a metric where session-level analysis is legitimate.
*Red flag:* "session-level has more data so it is better". *Scoring:* gives the `sqrt(DEFF)` relation
and a concrete clustering mechanism.

## 12. Quick Reference Card

| Item | Rule | Number to remember |
|---|---|---|
| Sample size, proportions | `n = (z_a + z_b)^2 (sigma_1^2 + sigma_2^2) / d^2` with pooled null variance | 42,158 per arm for 6% on 9.6% at 80% power |
| Sample size, means | `n = 2 (z_a + z_b)^2 sigma^2 / delta^2` | 15.70 as the multiplier at alpha .05, power .80 |
| Effect and CI | `d_hat +/- 1.96 * sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)` | 0.5701 pp +/- 0.1930 pp |
| Relative lift | `ln(p_T/p_C)` with `SE = sqrt((1-p_T)/(p_T n_T) + (1-p_C)/(p_C n_C))` | 5.99%, CI [3.92%, 8.10%] |
| MDE at fixed n | invert the sample-size formula | 0.275 pp at 182,000 per arm, 9.6% base |
| SRM check | chi-square, `k-1` df, alpha 0.001 | `chi2 = 58.8` for 152,100 vs 147,900 |
| Ratio metric | delta method with covariance term | SE 0.0360 correct vs 0.0306 naive at `rho = 0.20` |
| CUPED | `Var` factor `1 - rho^2` | 0.6975 at `rho = 0.55`, 30% traffic saved |
| Peeking | alpha spending or single look | 0.1416 Type I at 5 looks with 1.96 |
| Sequential boundary | `c/sqrt(t)`, `c = 2.042` at 5 looks | final-look `z = 2.042`, first-look `z = 4.566` |
| Guardrail | report MDE next to the point estimate | 0.348 pp at 18,000 per arm, 1.23% base |
| Decision | `E[effect] * volume * margin > cost`, pre-registered | threshold 3% relative, not 0% |
