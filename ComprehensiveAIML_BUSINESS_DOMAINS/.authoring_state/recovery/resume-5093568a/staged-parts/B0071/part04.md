
## 7. Comparison / Selection Table

### 7.1 Choosing the design

| Design | Use when | Cost vs fixed-horizon | Main risk | Guard |
|---|---|---|---|---|
| Fixed-horizon two-arm | Decision date is fixed; traffic is ample | Baseline | Peeking, driven by curiosity | Lock the analysis script; log looks |
| Group-sequential (alpha spending) | Interim safety or business urgency | +3% (O'Brien-Fleming, 5 looks) to +22% (Pocock, 5 looks) sample at 80% power | Boundary misapplied (wrong information fractions) | Precompute boundaries from planned `K` |
| Sequential (always-valid) | Continuous monitoring by dashboards | Substantially more than group-sequential; the price is paid on every look | Wide intervals misread as "no effect" | Show the always-valid interval, not 1.96 |
| Cluster-randomised | Interference or ops constraint (store, city, team) | multiply units by `1 + (m-1) * ICC` | Few clusters breaks normal approximation | At least 20 clusters per arm, t-correction |
| Switchback | Marketplace or time-varying interference | Depends on block length | Carryover across blocks | Washout periods between blocks |
| Stratified / paired | Strong pre-period predictor available | Saves 10% to 50% | Mis-stratification at assignment | Check allocation within strata |
| CUPED-adjusted | Pre-period metric exists and correlates | Saves `rho^2` of variance | Post-treatment covariate creeps in | Covariate must predate assignment |
| Bayesian decision rule | Stakeholders need `P(better)` not p-value | Similar to frequentist | Prior opacity; threshold shopping | Publish prior and loss function |

### 7.2 Choosing the estimator

| Estimator | Estimand | Requires | Failure signature |
|---|---|---|---|
| Difference in means / proportions | ATE under randomisation | Independent assignments | SE too small under clustering |
| Regression with covariate adjustment | ATE (same, more precise) | Covariate pre-treatment | Post-treatment covariate biases effect |
| CUPED | ATE | Pre-period metric, `theta` from pooled or pre-data | No variance gain if `rho` near zero |
| Delta-method ratio | Ratio-of-sums ATE | Numerator/denominator covariance | Naive SE wrong either direction |
| Cluster-robust regression | ATE at cluster level | At least 20-40 clusters | t-degrees too few, or CR2 variance needed |
| Differences-in-differences on switchback | Time-block ATE | Parallel trends across blocks | Seasonality confounded with arm |
| Quantile / trimmed mean | Robust central effect | Tail dominance in metric | Effect on mean lost, decision unclear |

### 7.3 Choosing the metric

| Metric type | Sensitivity | Interpretability | Common trap |
|---|---|---|---|
| Binary conversion rate | Low (needs large n) | High, maps to a decision | Rare events make n infeasible |
| Count per user | Medium | Medium | Zero-inflation inflates variance |
| Revenue per user | Low (heavy tail) | High, maps to money | Top 1% dominates variance |
| Revenue per session (ratio) | Medium | Medium | Denominator variance ignored |
| Winsorised revenue | Higher | Medium, needs explanation | Trim level chosen after seeing data |
| Proxy metric (e.g. clicks) | High | Low, needs a link to value | Proxy moves without the outcome moving |
| Surrogate index (weighted combo) | Highest | Lowest | Composite hides a guardrail regression |

The selection column that matters most is the third one in 7.3. Teams pick a proxy because it is
sensitive, then ship on the proxy, then discover the proxy was not the outcome. A proxy is
defensible only with a stated, pre-registered mapping to the decision — "clicks per session" is not
a value statement, "clicked plan CTA, which is the only path to trial start" is.

### 7.4 Test type versus traffic reality

A useful selection rule expressed in traffic rather than statistics: compute the MDE your traffic
supports, then ask whether that MDE is smaller than the ship threshold.

```text
case                        p1      n/arm available    MDE (abs)    ship threshold   verdict
plan-modal copy             0.0960     182,000          0.275 pp       0.288 pp (3%)  viable
onboarding checklist        0.0210      25,000          0.375 pp       0.210 pp (10%) not viable at 2 arms
refund-policy wording       0.0123      18,000          0.348 pp       0.20 pp        not viable
pricing page layout         0.0960      60,000          0.482 pp       0.288 pp       not viable
```

The third row is the design trap that produces most "inconclusive" results: the threshold is
strictly larger than the detectable effect, so a true-and-valuable improvement will usually fail to
clear the bar. The fix is not a longer test on the same metric; it is a coarser metric (any refund
contact, not refund processed) or a different decision process (a staged rollout with a
holdback-based readout over months).

## 8. Failure Modes & Edge Cases

Each entry below names a mechanism, its competing causes, and a check that separates them. "Monitor
the metric" is not a check; a check produces a number and a threshold.

### 8.1 Sample ratio mismatch caused by post-assignment filtering

**Mechanism.** Assignment happens in the client or edge, and eligibility filters (bot exclusion,
VPN detection, logged-out users, ad-blockers) are applied downstream. If the filter's firing rate
differs by arm — because the treatment adds a client-side call, or loads a different script — the
analysis population is no longer randomised even though the assignment was.

**Competing causes.** (a) genuine randomisation bug in the hashing/bucketing service; (b) a
segment of traffic excluded only in one arm; (c) a logging pipeline that drops events for one arm's
client version; (d) chance.

**Discriminating check.** Chi-square on assigned counts at alpha 0.001, then the same test on
*pre-filter* assignment counts. If pre-filter is balanced and post-filter is not, the filter is the
cause and the fix is to analyse the assigned population with the filter applied symmetrically.

```text
observed 152,100 / 147,900 of 300,000, planned 50/50
E = 150,000
chi2 = (2100^2)/150000 + (2100^2)/150000 = 29.4 + 29.4 = 58.8
p < 1e-13
```

Compare with a healthy draw of 182,400 / 182,560: `chi2 = 0.07`, `p = 0.79`. The test is not
sensitive to small imbalances and that is deliberate — alpha 0.001 keeps the false alarm rate low
enough that a flag is worth investigating.

### 8.2 Assignment unit finer than the analysis unit

**Mechanism.** Sessions are randomised, visitors are analysed. One visitor's sessions are
correlated in outcome (a determined user starts a trial in whichever session they land in), so the
effective sample size is closer to the number of visitors than to the number of sessions.

**Competing causes.** (a) genuine large effect; (b) a metric definition change mid-test; (c) bot
traffic concentrated in one arm; (d) the correlation described above.

**Discriminating check.** Recompute the SE by bootstrapping at the *assignment* unit, or compute
the intra-visitor correlation of the outcome and apply `DEFF = 1 + (m-1) ICC`. In the paywall
example, session-level analysis gives `SE = 0.000647` and `z = 8.81` against a visitor-level
`SE = 0.000984` and `z = 5.79`. The ratio of SEs, 1.52, is the square root of the design effect, so
`DEFF = 2.31` — consistent with roughly 3.4 sessions per visitor and an ICC near 0.55.

### 8.3 Peeking at a fixed-horizon test

**Mechanism.** The analyst checks daily and reports when the result crosses 1.96. Every look is an
independent chance to cross, so the Type I rate is the probability of *any* crossing, not the
probability of crossing at the end.

**Competing causes.** (a) a real effect that accumulated monotonically; (b) a novelty spike that
decayed and was caught at its peak; (c) a metric definition or logging change on a specific date;
(d) the inflation itself.

**Discriminating check.** Count the looks in the query log or dashboard view history, then recompute
the decision with an alpha-spending boundary for that many looks. If the result is significant at
1.96 but not at the `K`-look boundary, the finding is an artefact of looking. Second check: plot the
daily z-statistic — a real effect wanders around a positive mean, while a peeked false positive
tends to be a single excursion with the path returning toward zero.

The measured inflation is large and monotone in the number of looks: 5.0% at one look, 14.2% at
five, 19.4% at ten, 24.4% at twenty, from a 40,000-replication simulation of a standard normal walk
with a fixed 1.96 boundary.

### 8.4 Novelty and primacy effects confounded with treatment

**Mechanism.** The treatment changes something visible. Early cohorts respond to the change, later
cohorts respond to the value (or stop noticing). The average over the window is a mix of a transient
and a persistent component, and the mix depends on the window length you happened to choose.

**Competing causes.** (a) genuine decay of a real benefit (users learn the feature is not useful);
(b) seasonality interacting with cohort composition; (c) a bug in the treatment fixed mid-test;
(d) instrumentation warming up.

**Discriminating check.** Split the window into weeks and compare control against treatment *within
each week's cohort* — not cumulative. A novelty effect produces a week-1 effect substantially larger
than the week-4 effect with the same sign; a real-but-decaying benefit produces a monotone decline
that stabilises above zero; seasonality produces a dip in both arms in the same week.

```text
ILLUSTRATIVE weekly revenue per visitor
          control   treatment   diff
week 1     9.61      10.14      +0.53
week 2     9.43      10.04      +0.61
```

The second week's difference is *larger* than the first, which rules out a simple novelty spike but
does not rule out novelty in the opposite direction (users needing time to find the new path). The
honest read: the effect is stable over two weeks, and the decision should be conditioned on a
post-launch readout at 4-6 weeks because two weeks cannot distinguish stability from a slow ramp.

### 8.5 Ratio metric analysed with the naive variance

**Mechanism.** `R = Y/X` is treated as if `X` were fixed. The variance of a ratio includes a term
in `sigma_X^2` and a covariance term, and the sign of the covariance decides whether the naive SE
is too small or too large.

**Competing causes.** (a) a real effect on the numerator; (b) a real effect on the denominator that
moves the ratio without moving value; (c) the variance error; (d) a change in the mix of units
contributing to the ratio.

**Discriminating check.** Compute the ratio three ways: naive, delta-method at the randomisation
unit, and a per-unit aggregate (`sum(Y)/sum(X)` over units, then a two-sample test on the aggregate
or a bootstrap over units). If naive and delta disagree by more than 10% in SE, the covariance
matters and only the delta or bootstrap result is usable.

```text
mu_X=6.0 sd_X=5.5  mu_Y=24.0 sd_Y=26.0  R=4.000   n=20,000
rho=0.20  correct SE 0.0360   naive 0.0306   naive understates by 17.4%
rho=0.90  correct SE 0.0135   naive 0.0306   naive overstates by 127%
```

The `rho = 0.90` row matters because it produces a *conservative* naive test, which nobody
investigates: an underpowered-looking result is attributed to a small effect rather than a variance
formula that left 127% of precision on the table.

### 8.6 Guardrail underpowered by construction

**Mechanism.** Guardrail events are rare by design — that is what makes them guardrails. The
experiment is sized on the primary metric's base rate, so the guardrail's sample is one to two
orders of magnitude too small, and the test has almost no chance of detecting the harm it exists to
detect.

**Competing causes.** (a) no harm; (b) harm smaller than the detectable floor; (c) harm in a subset
(one platform, one plan tier); (d) harm measured on the wrong denominator (per session instead of
per trial).

**Discriminating check.** Compute the guardrail's MDE at the realised sample before declaring
safety, and print it next to the point estimate. If `MDE > 2x` the smallest harmful effect the
business cares about, the guardrail is decorative. Second check: stratify by the platform or tier
with the highest event rate and test there, accepting a pre-registered multiplicity penalty.

```text
refund rate 1.2324% control, n/arm ~18,000
MDE (80% power, alpha .05 two-sided) = 0.348 pp  = 28.2% relative increase
smallest harmful effect the business cares about = 0.10 pp
verdict: guardrail cannot answer the question it was created for
```

### 8.7 Interference between units

**Mechanism.** Treatment changes one unit's behaviour in a way that changes another unit's outcome:
a referral prompt increases invites, which changes the invitee's experience; a discount reduces
inventory, which raises prices for others; a message send reduces a shared rate limit.

**Competing causes.** (a) spillover through a shared resource; (b) selection into the experiment
correlated with social graph position; (c) a genuine direct effect plus a market-level price
response; (d) measurement: the "other unit" outcome is logged under the treated unit.

**Discriminating check.** Compare the control group's outcome rate in high-treatment-density
clusters against low-density clusters. Under no interference the two are equal; under spillover the
control group in dense clusters moves in the treatment's direction. A second check is a
cluster-randomised replication on a small market set: if the cluster-level effect is much smaller
than the user-level effect, the user-level estimate includes a spillover that will not survive a
full launch.

### 8.8 Multiple comparisons across metrics and segments

**Mechanism.** The decision uses "did anything improve?" across twelve metrics and eight segments.
At alpha 0.05 per test, 96 tests produce about 4.8 expected false positives, and at least one
crossing is nearly certain.

**Competing causes.** (a) a real broad improvement; (b) a real effect in one segment only; (c)
chance crossings; (d) correlated metrics, where one true effect produces several correlated
"findings".

**Discriminating check.** Count the tests performed, then apply a family-wise or false-discovery
correction and report which findings survive. Bonferroni at `k=12` sets the per-test threshold to
`0.05/12 = 0.00417`, which is a `z` of 2.86 rather than 1.96. For correlated segment cuts, the
effective number of tests is lower than the nominal count; a permutation-based max-statistic
distribution handles the correlation properly.

```text
P(at least one false positive) with k=12 independent tests at alpha .05
= 1 - 0.95^12 = 1 - 0.5404 = 0.4596
```

A second, cheaper check that catches most of the real problem: ask whether the winning segment was
named in the pre-registration. If it was not, treat it as a hypothesis for the next test, not a
finding from this one.

### 8.9 Stopping on a secondary metric

**Mechanism.** The primary is flat, a secondary moved, and the team ships on the secondary. The
secondary was not the estimand, its base rate makes it noisier, and the switch happened after
seeing the result.

**Competing causes.** (a) the secondary is a better proxy for value and should have been primary;
(b) the primary is mis-specified; (c) the secondary move is noise amplified by a post-hoc choice;
(d) the secondary is downstream of the primary and moves for an unrelated reason.

**Discriminating check.** Ask what the pre-registered decision rule says, and whether the secondary
would have been examined had the primary been a loss. Then check the secondary's own MDE and its
correlation with the primary: if the secondary's MDE is 4x the observed move, the "win" is inside
the noise band. The defensible path is to re-run with the secondary as primary, sized properly, not
to ship on the switch.

### 8.10 Treatment leakage and cache contamination

**Mechanism.** A CDN, a client cache, or a shared service worker serves the treatment payload to
control users, or vice versa. The measured effect is diluted toward zero and the arms are partially
mixed.

**Competing causes.** (a) a small true effect; (b) partial rollout / low exposure; (c) leakage; (d)
a broken exposure log that mislabels which payload was served.

**Discriminating check.** Compare the *served* configuration hash against the assigned arm per
request, and count mismatches. A mismatch rate above roughly 2% makes the ITT interpretation
"effect of assignment, with documented contamination" and requires the TOT calculation with the
observed exposure share. A second check is a client-side canary: render a hidden marker in the
payload and count markers per arm.

### 8.11 Simpson's paradox from a composition shift

**Mechanism.** Treatment changes the mix of units reaching the metric: a faster onboarding flow
admits more low-intent users, whose conversion is higher in the flow but lower in the population
average. Aggregate treatment rate can fall while every stratum rises.

**Competing causes.** (a) genuine harm in the aggregate; (b) mix shift; (c) a metric that is a
ratio of two things the treatment both moved; (d) segment definitions changed mid-test.

**Discriminating check.** Compute the effect within each pre-treatment stratum (device class, signup
cohort, plan tier) and compare with the aggregate. If all strata are non-negative and the aggregate
is negative, the aggregate is a mix artefact and the decision should be made on a mix-adjusted
estimand — with the mix shift itself reported, because a mix that admits more low-intent users has
its own business meaning.

### 8.12 Metric definition drift during the run

**Mechanism.** A logging change mid-test adds an event type or changes a deduplication rule. Both
arms are affected, but not equally, because the change lands on a client version that is unevenly
distributed across arms by platform weighting.

**Competing causes.** (a) a step change from the instrumentation; (b) a real effect that happens to
start mid-window; (c) seasonality; (d) a bot sweep.

**Discriminating check.** Plot the metric by day for each arm separately and look for a level shift
that is shared (instrumentation) or arm-specific (instrumentation interacting with assignment).
Then re-run the analysis restricted to a window on one side of the change and compare. If the effect
survives on each side, the change is not the explanation.
