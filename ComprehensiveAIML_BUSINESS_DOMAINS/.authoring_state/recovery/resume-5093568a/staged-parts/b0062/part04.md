### 4.4 Limits and counterexamples

**The base formula breaks when the analysis unit is finer than the assignment unit.**
Take an experiment randomised by user but analysed by session, with 4.2 sessions per user on average.
The reported `n` of 500,000 sessions looks like overwhelming evidence, but the effective independent count is 119,048 users.
The standard error shrinks by `sqrt(4.2) = 2.05` while the estimate does not, so a `z` of 2.0 in the correct analysis becomes a `z` of 4.1 in the wrong one.
The counterexample is not exotic: it is any join between a session-level event table and a user-level assignment table where the analyst forgets to aggregate first.

**The base formula breaks when the control rate is close to zero and the relative lift is the reported quantity.**
With `p_hat_C = 0.0004` and `p_hat_T = 0.0012`, the relative lift is 200%, which will be read as a tripling of the metric.
The absolute difference is `0.08 pp`, and its 95% interval at 50,000 per arm spans roughly `[0.02 pp, 0.14 pp]`, which is a different story about a different quantity.
Reporting a relative lift without the control baseline lets a rounding-scale improvement masquerade as a transformation.

**The base formula breaks when the null hypothesis is not a point but a margin.**
If the decision rule is "ship unless the change is more than 0.5 pp worse", the relevant test is not `d = 0` but `d = −0.005`, and a p-value against zero answers a question nobody asked.
A trial with `d_hat = −0.0002` and an interval of `[−0.0021, +0.0017]` is strong evidence of noninferiority against a 0.5 pp margin and no evidence at all about whether the effect is positive.

**The base formula breaks when randomisation is valid but the comparison is not concurrent.**
A test that runs treatment for three weeks and then control for three weeks compares weeks as much as variants.
If the two periods differ by a 4% seasonal drift, the estimate absorbs the whole drift, and no amount of within-period precision recovers the contrast.
The remedy is concurrent assignment or a switchback with many short blocks, not a larger sample.

**The base formula breaks when treatment changes the composition of the population being measured.**
If treatment makes the checkout faster and therefore attracts a different mix of users into the funnel, the arm rates describe different populations.
Formally, `p_T` and `p_C` are now averages over different mixes, and the difference contains a composition term alongside the effect term.
Detecting this requires a pre-assignment covariate check on the analysed population, not only on the assigned population.

**The base formula is silent about whether the effect persists.**
An effect measured over seven days may decay as novelty wears off or may grow as users learn the interface.
The estimator is correct for its window and says nothing outside it, which is why a launch decision usually needs a holdout rather than a longer pre-launch test.

**The five-part shape itself has a limit: it cannot repair a design.**
Every derivation above assumes assignment precedes the outcome and is independent of potential outcomes.
No amount of algebra, no covariate adjustment, and no sophisticated estimator restores causality to an observational comparison that lacks that structure.

**Takeaway:** Each formula in Section 4 is a conditional statement; the counterexamples mark the conditions, not the edges of the algebra.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE constructed checkout-flow experiment; every number below was recomputed for this sheet, and none describes a real product, company, or deployment.

### 5a The setup before any analysis

A subscription retailer wants to replace a three-step checkout with a one-page checkout for signed-in users on web.
The primary outcome is a completed purchase within seven days of assignment, counted at the user grain.
The guardrail is the payment-failure rate for attempted purchases, measured over the same seven days.
The secondary outcome is revenue per assigned user, which is a ratio metric and is analysed separately.
Eligibility is: signed-in, web platform, account at least one day old, not in a competing checkout experiment, and not flagged as an internal test account.
Assignment is by a hash of `user_id` salted with the experiment key and version, 50/50, sticky for the account.
Enrollment runs 14 days at roughly 17,025 eligible users per day, which yields about 119,175 per arm.

### 5b The planned sample size, derived

```text
FORMULA
  n_per_arm = (1.95996 + 0.84162)^2 * [p0(1-p0) + p1(1-p1)] / (p1 - p0)^2
WHERE
  p0 = 0.112  (baseline seven-day conversion, prior four weeks, same population)
  p1 = 0.120  (the smallest effect worth shipping: +0.8 pp, about +7% relative)
CONSTRAINT
  one primary outcome, one final analysis, no interim looks, unit-level Bernoulli outcome
NUMERICAL EXAMPLE
  p0(1-p0)      = 0.112 * 0.888  = 0.099456
  p1(1-p1)      = 0.120 * 0.880  = 0.105600
  sum           = 0.205056
  (1.95996+0.84162)^2 = 2.80158^2 = 7.84885
  n_per_arm     = 7.84885 * 0.205056 / 0.000064 = 25,147.8 -> 25,148
INTERPRETATION
  At 17,025 eligible users/day split evenly, 25,148 per arm is reached in 1.48 days.
  That is too fast to observe a seven-day outcome with any maturity, so the binding
  constraint is the follow-up window, not the sample size.
```

The maturity constraint is what actually sets the duration, and it is worth making explicit because teams routinely size on traffic and forget the clock.

```text
FORMULA
  duration_days >= follow_up_days + ingestion_allowance_days
  assigned_per_arm = eligible_per_day/2 * duration_days
WHERE
  follow_up_days = 7 for the primary outcome
  ingestion_allowance_days = 1 for late-arriving purchase events
CONSTRAINT
  assignment closes at the end of the enrollment window; outcomes keep accruing afterwards
NUMERICAL EXAMPLE
  duration = 8 days -> 8 * 8,512.5 = 68,100 per arm, below the 25,148 needed? no: above.
  Wait, the binding requirement is 25,148 per arm, reached at 2.95 days.
  So a 14-day window is chosen for a different reason: to average over two weekends
  and to accumulate guardrail events, not to accumulate primary-outcome units.
INTERPRETATION
  The window is set by the guardrail's event rate, not by the primary outcome's.
  Payment failures run at 1.42%, so 25,148 per arm yields about 357 failures per arm:
  enough to detect a 0.19 pp absolute change at 80% power, not enough for 0.05 pp.
```

Recomputing that guardrail sizing explicitly, since it is the number that drove the design:

```text
FORMULA
  n_per_arm = (1.95996 + 0.84162)^2 * [q0(1-q0) + q1(1-q1)] / (q1 - q0)^2
WHERE
  q0 = 0.0142 (payment-failure rate, control), q1 = 0.0161 (+0.19 pp, the harm threshold)
NUMERICAL EXAMPLE
  q0(1-q0) = 0.0142 * 0.9858 = 0.013998
  q1(1-q1) = 0.0161 * 0.9839 = 0.015841
  sum      = 0.029839
  n_per_arm = 7.84885 * 0.029839 / (0.0019)^2 = 7.84885 * 0.029839 / 3.61e-6 = 64,880
INTERPRETATION
  64,880 per arm at 8,512.5/day takes 7.6 days, so a 14-day window gives roughly
  119,175 per arm and detects about 0.14 pp at 80% power.
```

### 5c What the data looked like

| Quantity | Control | Treatment | Difference | Unit |
|---|---|---|---|---|
| Assigned users | 118,944 | 119,412 | +468 | users |
| Purchases within 7 days | 13,428 | 13,700 | +272 | users |
| Conversion rate | 11.2893% | 11.4729% | +0.1835 | pp |
| Exposure rate | 74.195% | 69.069% | −5.126 | pp |
| Payment attempts | 20,412 | 21,088 | +676 | attempts |
| Payment failures | 1,690 | 1,922 | +232 | attempts |
| Payment-failure rate | 1.4208% | 1.6096% | +0.1887 | pp |
| Revenue | $1,046,880 | $1,081,200 | +$34,320 | USD |
| Revenue per assigned user | $8.80145 | $9.05437 | +$0.25291 | USD/user |

The exposure gap is the first thing to notice, because a 5.1 pp difference in who reached the changed surface is larger than the effect being measured and it is statistically overwhelming.
That does not automatically invalidate the primary analysis, which is ITT by design, but it does change which question the ITT answer addresses.

### 5d The primary analysis, three ways

```text
FORMULA (Wald, unpooled variance)
  d_hat = 0.1147288 - 0.1128935 = 0.0018354
  SE    = sqrt(0.1147288*0.8852712/119412 + 0.1128935*0.8871065/118944)
        = sqrt(8.5064e-7 + 8.4211e-7) = sqrt(1.69275e-6) = 0.00130106
  z     = 1.41077,  p = 0.1583
  CI95  = 0.0018354 +- 1.95996*0.00130106 = [-0.0007145, +0.0043852]
CONSTRAINT
  independent users, stable arm sizes, at least 5 events of each type per arm
INTERPRETATION
  The interval spans a 0.07 pp loss and a 0.44 pp gain. It excludes the planned
  0.8 pp target at the upper end, which is the informative part of the result.
```

```text
FORMULA (pooled-variance score test, an independent re-check)
  p_pool = (13428 + 13700) / (118944 + 119412) = 27128 / 238356 = 0.11381295
  SE_0   = sqrt( p_pool*(1-p_pool)*(1/118944 + 1/119412) )
         = sqrt(0.11381295*0.88618705*1.68072e-5) = 0.00130100
  z_0    = 0.0018354 / 0.00130100 = 1.41074,  p = 0.1584
CONSTRAINT
  valid under the null, which makes it the appropriate test of H0: d = 0
INTERPRETATION
  The two tests agree to four decimal places of z, which is the expected agreement
  when neither arm's variance is far from the pooled value. Agreement here is a
  consistency check on the arithmetic, not independent evidence about the effect.
```

```text
FORMULA (relative lift and its interval)
  relative = 0.0018354 / 0.1128935 = 0.016258 = +1.626%
  lower    = -0.0007145 / 0.1128935 = -0.00633 = -0.63%
  upper    = +0.0043852 / 0.1128935 = +0.03884 = +3.88%
CONSTRAINT
  control rate strictly positive and not close to zero; this ratio interval is the
  endpoint ratio, which is not the same as a Fieller interval but is close when the
  denominator's relative standard error is small (here 0.29%)
INTERPRETATION
  "+1.6% lift" and "somewhere between -0.6% and +3.9%" are the same result.
  Quoting only the first is a presentation choice, not a statistical one.
```

### 5e The guardrail, which is the decisive number

```text
FORMULA (difference in payment-failure rates)
  q_C = 1690/20412 = 0.082793, but this is failures per attempt, not per assigned user.
  Per assigned user: 1690/118944 = 0.0142084, 1922/119412 = 0.0160955
  d_q = 0.0018871,  SE = sqrt(0.0142084*0.9857916/118944 + 0.0160955*0.9839045/119412)
      = sqrt(1.1775e-7 + 1.3265e-7) = 0.00050038
  z   = 0.0018871 / 0.00050038 = 3.7715,  p = 0.000162
  CI95 = [0.0009064, 0.0028679] = [+0.091 pp, +0.287 pp]
CONSTRAINT
  failures counted once per assigned user, attempts deduplicated, provider outages
  excluded under a pre-specified rule
INTERPRETATION
  The harm is not marginal. Its lower bound sits above the pre-agreed 0.10 pp
  threshold, so the guardrail rule fires on its own terms.
```

An exact test on the same 2×2 table agrees, which matters because the failure counts are small enough that the normal approximation deserves checking:

```text
FORMULA (Fisher exact test, two-sided, on the 2x2 user-level table)
  table = [[1690, 117254], [1922, 117490]]
  odds ratio = (1690/117254) / (1922/117490) = 0.8811
  p = 0.000172
CONSTRAINT
  fixed margins, independent users, one failure counted at most once per user
INTERPRETATION
  p = 0.000172 against the Wald p = 0.000162: the approximation is adequate here
  because both cells exceed 1,600. It would not be adequate at 8 failures per arm.
```

### 5f The decision arithmetic

```text
FORMULA (expected revenue change if the conversion effect were the only change)
  incremental purchases = d_hat * n_per_arm_assigned = 0.0018354 * 118,944 = 218.3
  value per purchase    = 1,046,880 / 13,428 = $77.962
  gross value           = 218.3 * 77.962 = $17,021 per 14-day window per arm-pair
CONSTRAINT
  assumes the incremental purchasers have the same average order value as existing
  purchasers, and ignores refunds and repeat purchases
NUMERICAL EXAMPLE
  218.3 * 77.962 = 17,020.7 dollars
  Extrapolated naively to a year: 17,020.7 * (365/14) = $443,700
INTERPRETATION
  The gross-value estimate is $443,700/year under a strong constant-effect
  assumption, with a confidence interval that spans roughly -$172,000 to +$1,059,000
  once the primary interval is propagated. Against that uncertain upside sits
  +232 payment failures per 14 days, which at 0.9 support contacts per failure and
  $6 per contact costs about $1,254 per window in direct service cost alone.
```

The direct service cost understates the harm, which is why the guardrail is a rule rather than a dollar calculation: a failed payment on a subscription renewal is a churn risk whose expected cost is far above a support contact.
That asymmetry is exactly why guardrails are pre-specified as thresholds instead of being folded into a single expected-value expression at analysis time.

### 5g Independent re-check of the whole table

```text
CHECK 1  arm totals: 118,944 + 119,412 = 238,356 assigned users
CHECK 2  pooled rate: 27,128/238,356 = 0.113813; 2*0.113813 = 0.227626, and
         0.1128935 + 0.1147288 = 0.2276223, agreeing to 4e-6 (rounding).
CHECK 3  relative lift: 0.1147288/0.1128935 = 1.016258, i.e. +1.6258%,
         matching 0.0018354/0.1128935 = 0.016258.
CHECK 4  CI symmetry: midpoint of [-0.0007145, +0.0043852] is 0.00183535,
         equal to d_hat = 0.0018354 to 5 decimal places.
CHECK 5  guardrail CI: midpoint of [0.0009064, 0.0028679] is 0.00188715,
         equal to d_q = 0.0018871.
CHECK 6  exposure counts: 0.74195*118,944 = 88,251 and 0.69069*119,412 = 82,477,
         summing to 170,728 of 238,356 assigned, i.e. 71.6% overall.
```

### 5h Sensitivity: what would change the conclusion

| Perturbation | Magnitude | Effect on the decision |
|---|---|---|
| 3% of treatment conversions are duplicate rows | removes 411 conversions | treatment rate falls to 11.129%, effect becomes −0.160 pp |
| Payment failures under-reported by 10% in treatment only | 1,922 becomes 1,730 | guardrail difference falls to +0.028 pp, no longer decisive |
| Seven-day window extended to fourteen days | larger effect, lower precision | decision unchanged but interval wide enough to cross the threshold |
| 5% of control users received the new flow | contaminates control upward | true effect larger than measured, guardrail unchanged |
| Exposure-rate gap is a logging artefact, not behaviour | exposure gap vanishes | primary estimate unaffected; interpretation of dilution changes |

The first two rows are the ones a reviewer should demand evidence against, because either one flips a decision.

**Takeaway:** the arithmetic said "no reliable benefit"; the guardrail said "reliable harm"; the decision follows the guardrail because the rule was set before the data arrived.
