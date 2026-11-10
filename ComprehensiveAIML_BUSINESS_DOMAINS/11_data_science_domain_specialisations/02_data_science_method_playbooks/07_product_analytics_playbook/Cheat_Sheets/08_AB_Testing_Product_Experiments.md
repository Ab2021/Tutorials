---
title: "Product A/B Testing and Experiments"
topic: "Product experimentation: estimands, power, validity threats and ship decisions"
collection: "Product analytics playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Product A/B Testing and Experiments

A cheat sheet for the analyst who has been handed an assignment table, an events table and a
decision meeting on Thursday. Every formula below is derived from stated assumptions, and every
number attached to it is either recomputed in this file or labelled ILLUSTRATIVE. No claim here
describes a real company's measured result. The running example — a paywall trial-start experiment
— is a constructed vehicle for arithmetic, not a case study.

## 1. File Header

| Field | Value |
|---|---|
| Purpose | Turn an assignment log and an event log into a defensible ship / no-ship decision |
| Audience | Product analysts, data scientists, experiment owners, PMs who read dashboards |
| Outcome | You can state an estimand, size an experiment, compute an effect with uncertainty, and name the threat that would invalidate it |
| Prerequisites | Proportions, standard errors, confidence intervals, two-sample tests, SQL with event-time semantics |
| Scope | Two-arm randomised online experiments, from enrolment through post-launch monitoring |
| Not covered | Observational causal inference, contextual bandits, clinical or regulatory trial design |
| Teaching home | Product analytics playbook, source coverage `ds-methods-07` |
| Evidence status | Derivations are DERIVED; all inputs are ILLUSTRATIVE constructions; no production run is reported |
| Notation | Plain ASCII inside `text` fences; `pp` means percentage points, `%` means relative change |
| Retrieval cue | Suspicious result: Section 8. Number you must report: Section 5. Formula you half-remember: Section 4 |
| Review trigger | Change of assignment service, metric definition, or analysis unit |

Three conventions keep the rest of this file unambiguous, and each of them is a place where real
analyses go wrong.

First, a difference is reported in percentage points and a lift is reported in percent relative to
control. A move from 9.60% to 10.18% is +0.58 pp, which is +6.0% relative. Reports that say "a 6%
increase" while the table shows 0.058 invite a decimal-place disaster in the ship memo.

Second, every rate names its denominator population. Trial starts per eligible visitor is not the
same metric as trial starts per session, and the two can move in opposite directions when the
treatment also changes how many sessions a visitor has.

Third, the analysis unit and the randomisation unit are stated separately. When they differ, the
variance formula changes by a design effect, and the p-value printed by a naive test is too small.

## 2. Concept Map / Mental Model

The useful mental model is that an experiment is a measuring instrument with four gates, and a
result is only as trustworthy as the weakest gate that was actually checked.

```text
gate 1  assignment      eligible -> randomised -> balanced?          SRM chi-square, covariate balance
gate 2  exposure        assigned -> actually saw treatment?          exposure logs, trigger analysis
gate 3  estimation      metric -> effect + uncertainty?              estimator, clustering, CUPED
gate 4  decision        effect -> ship / iterate / stop?             cost model, guardrails, thresholds
```

The causal chain inside gate 2 is the part most teams skip, and it is the part that decides whether
the number means what the memo says.

```text
eligible population
   |  randomisation (exchangeability bought here, and only here)
assigned cohort
   |  delivery / eligibility re-check (attrition, eligibility drift)
exposed cohort
   |  behaviour (novelty, learning, seasonality)
metric movement
   |  extrapolation (novelty decay, interference, population drift)
business outcome
```

Each arrow has its own failure and its own diagnostic. Randomisation failures show up as a sample
ratio mismatch before any metric is read. Delivery failures show up as a treatment/control gap in
exposure rates. Behavioural artefacts show up as an effect that decays across weeks. Extrapolation
failures show up only after launch, which is why the pre-launch memo needs a stated decay
assumption rather than a silence.

A second organising idea is that three different quantities get called "the result", and they
answer different questions.

```text
sample difference   d_hat = p_T - p_C          a fact about 365k observed assignments
population effect   tau   = E[Y(1) - Y(0)]     an inference, valid only under assumptions A1..A4
decision value      V     = tau * volume * margin - cost of shipping
```

The sample difference is arithmetic and cannot be wrong if the data are right. The population
effect requires exchangeability, no interference, and a stable metric definition. The decision value
requires a volume estimate and a cost model, both of which are forecasts. Mixing the three lets a
correctly computed +0.57 pp become an unjustified "this will earn $1.5M next quarter" in a slide.

A third idea: precision is bought, not observed. Sample size is the expensive currency, and there
are only four ways to buy precision — more traffic, longer duration, a better (lower-variance)
estimator, or a more sensitive metric. Recognising that CUPED and a tighter metric are substitutes
for traffic is what stops teams from running an eight-week test to learn nothing.

The fourth idea is asymmetry of costs. A false positive ships a neutral or harmful change and
usually also burns the team's credibility, while a false negative loses the value of a real
improvement and teaches the organisation that tests "always come back flat". These costs are not
equal, they are not symmetric in probability, and a decision rule that ignores them is a decision
rule that optimises for the wrong error.

| Gate | What it protects | Fails when | First check |
|---|---|---|---|
| Assignment | Exchangeability of arms | Assignment bug, bot filtering applied after the split | Sample ratio mismatch at alpha 0.001 |
| Exposure | Treatment actually delivered | Partial rollout, client version gating, cache staleness | Exposure rate by arm and by platform |
| Estimation | Honest uncertainty | Clustering ignored, ratio metric tested naively, peeking | Recompute SE at the randomisation unit |
| Decision | Right action at the right cost | Threshold plucked from habit, guardrails underpowered | Pre-registered rule and cost model |

Most "the test lied" stories are gate 1 or gate 2 failures wearing gate 3 clothing. A metric that
moved because the treatment arm contained 2% more returning users is not a statistical anomaly; it
is a randomisation failure that a ratio check would have caught in ten seconds.

## 3. Foundational Definitions

Each definition below is written so that it changes what you compute, not just what you say.

**Randomisation unit.** The entity that is independently assigned to an arm: user, session, device,
account, store, or region. It sets the denominator of the variance formula. Assigning sessions and
analysing users is the single most common way to understate uncertainty, because one user's
sessions are correlated.

**Analysis unit.** The entity over which you aggregate before differencing. It must be the
randomisation unit or a coarser grouping; if it is finer, cluster the standard errors.

**Estimand.** A precise sentence of the form: the average difference in `<metric>` between
`<treatment>` and `<control>` for `<population>` over `<window>`, aggregated at `<unit>`, under
`<handling of non-exposure>`. If the sentence cannot be written before the test starts, the test
has no primary outcome.

**Intention-to-treat (ITT) effect.** The effect of being *assigned*, whether or not the treatment
was delivered. It is what randomisation identifies without extra assumptions, and it is the right
quantity when shipping means "turn on the flag for the same population".

**Treatment-on-the-treated (TOT) effect.** The effect on those who actually received the treatment.
Identified from ITT by dividing by the exposure share, under the assumption that assignment changes
the outcome only through exposure (exclusion restriction). A 78% exposure share turns a 0.570 pp
ITT into a 0.731 pp TOT.

**Exposure (trigger) rate.** Fraction of assigned units that met the treatment's precondition. A
low rate is not an error by itself; it is a power problem plus a dilution factor, and it must be
stated next to the effect.

**Primary metric (OEC).** The one metric the ship decision is conditioned on. Everything else is a
guardrail or a diagnostic. Having three primaries is having none, because the decision rule then
needs a multiplicity correction nobody pre-registered.

**Guardrail metric.** A metric whose degradation blocks shipping even if the primary improves:
refund rate, crash rate, latency, unsubscribe rate, support contacts. Guardrails are usually tested
one-sided, because only harm is actionable.

**Sample ratio mismatch (SRM).** A statistically detectable departure from the planned allocation.
It invalidates the experiment before analysis, because a broken assignment mechanism is correlated
with whatever caused the break.

**Minimum detectable effect (MDE).** The smallest true effect that the design reaches the target
power for, at the planned alpha. MDE is a property of the design, not of the observed data, and it
is the number to quote when someone asks "was this test big enough?".

**Power.** Probability of rejecting the null when the true effect equals the design's MDE. Eighty
percent is a convention, not a law; it implies you will miss one true effect in five of exactly the
size you designed for.

**p-value.** Probability, computed under the null and the model, of a test statistic at least as
extreme as the observed one. It is not the probability the null is true, and not the probability the
result replicates.

**Confidence interval.** A procedure that covers the true effect in 95% of repeated experiments.
Its width is the honest statement of what the data can distinguish; a CI that crosses zero and a CI
that excludes it are different decisions, and the width matters as much as the crossing.

**CUPED.** Controlled-experiment using pre-experiment data: regress the post-period metric on a
pre-period covariate and analyse the residual. It reduces variance by rho-squared without touching
the estimand, so it buys precision rather than truth.

**Novelty effect.** A transient change in behaviour caused by the change itself rather than its
value: users click the new banner because it is new. Detected by an effect that shrinks across
weeks within the treatment arm, not by an effect that is merely smaller than hoped.

**Interference (SUTVA violation).** One unit's outcome depends on another unit's treatment. Social
features, shared inventory, and marketplace pricing all interfere; user-level randomisation then
estimates something that is neither the user-level nor the market-level effect.

**Cluster randomisation.** Assignment at a coarser unit (store, city, team) to respect interference
or operational constraints. It costs precision through the design effect `1 + (m - 1) * ICC`, where
`m` is cluster size.

**Ratio metric.** A metric whose numerator and denominator are both random: revenue per session,
clicks per impression, items per order. Its variance needs the delta method, because the denominator
varies and covaries with the numerator.

**Sequential test.** A test whose error rate is controlled across multiple looks at accumulating
data. Fixed-horizon tests peeked at repeatedly inflate the false positive rate; five equally spaced
looks at alpha 0.05 with a naive 1.96 boundary produce roughly 14% false positives.

**Alpha spending.** A function that allocates the total Type I budget across looks, typically
spending little early. It converts "we may stop early" into a rule that a statistician can defend.

**Ship threshold.** The effect size at which the organisation is willing to take on the change's
cost and risk. It is a business number, set before the test, and it is usually larger than zero.

**Practical significance.** Whether the effect is large enough to matter at real traffic volumes,
independent of whether it is distinguishable from zero. A 0.05 pp lift can be statistically certain
and commercially irrelevant.

**Multiple comparisons.** Testing many metrics or segments raises the chance that at least one
crosses a threshold by luck. Twelve independent metrics at alpha 0.05 give a 45.96% chance of at
least one false positive.

**Pre-registration.** Writing the estimand, primary metric, sample size, stopping rule and analysis
population before unblinding. It is the only defence against the analyst's honest ability to
rationalise any result after seeing it.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**Result.** For a two-arm test of a proportion with control rate `p1`, target rate `p2 = p1(1+r)`,
two-sided alpha and power `1 - beta`, the required sample size per arm is

```text
n = [ z_(1-alpha/2) * sqrt(2 * pbar * (1 - pbar))  +  z_(1-beta) * sqrt(p1(1-p1) + p2(1-p2)) ]^2 / (p2 - p1)^2

WHERE   pbar = (p1 + p2) / 2           pooled rate between the two arms
        z_q  = q-quantile of the standard normal
        d    = p2 - p1                  absolute effect, proportion units
CONSTRAINT   n * p1 >= 10 and n * (1 - p1) >= 10 in each arm
             alpha two-sided; arms independent; one analysis at the end
```

**Derivation.** Let `d_hat = p2_hat - p1_hat`. Under the alternative, `E[d_hat] = d` and

```text
Var(d_hat) = p1(1-p1)/n + p2(1-p2)/n
```

Standardising, `d_hat` is approximately normal with that mean and variance. Rejecting when
`|d_hat| / SE > z_(1-alpha/2)` fails to reject with probability `beta` exactly when the true mean is
only `(z_(1-alpha/2) + z_(1-beta)) * SE` away from zero. Setting `d` equal to that distance and solving
for `n`:

```text
d = (z_(1-alpha/2) + z_(1-beta)) * sqrt( [p1(1-p1) + p2(1-p2)] / n )
n = (z_(1-alpha/2) + z_(1-beta))^2 * [p1(1-p1) + p2(1-p2)] / d^2
```

The pooled form used in practice replaces the sum inside the first term with `2 * pbar * (1 - pbar)`
because the null-hypothesis variance (the scale on which alpha is spent) is better approximated by
the pooled rate than by the alternative rates. The two forms agree closely near the null.

**Numerical example (ILLUSTRATIVE).** Control trial-start rate `p1 = 0.096`, target relative lift
`r = 0.06`, so `p2 = 0.096 * 1.06 = 0.10176` and `d = 0.00576`. Pooled `pbar = (0.096 + 0.10176)/2 =
0.09888`.

```text
sqrt(2 * 0.09888 * 0.90112)          = sqrt(0.178208)          = 0.422149
sqrt(0.096*0.904 + 0.10176*0.89824)  = sqrt(0.178157)          = 0.422088
first term    1.959964 * 0.422149    = 0.827386
second term   0.841621 * 0.422088    = 0.355267
sum                                  = 1.182653
square                               = 1.398668
divide by d^2 = 0.00576^2 = 0.0000331776
n per arm                            = 42157.2  ->  42158
total assignments                    = 84316
```

**Interpretation.** 42,158 per arm is the traffic required for an 80% chance of detecting a 6%
relative improvement, with a 5% two-sided false positive rate. At 26,000 eligible assignments per
day, the run takes 84,316 / 26,069 = 3.23 days of traffic, and a real plan would round up to a whole
number of weeks, giving 7 days.

**Validity condition.** The formula assumes independent Bernoulli outcomes within each arm and a
single analysis. It is not valid when assignments are clustered (Section 4.3), when the metric is a
ratio (Section 4.3), or when the analyst looks at the data more than once (Section 4.4). It also
assumes the true `p1` is known; a 10% error in `p1` moves `n` by roughly 20% because both the
variance and the absolute effect depend on it.

### 4.2 General case

The base formula has four practical generalisations, and each has a distinct derivation path.

**(a) Continuous or unbounded metrics.** For a mean difference with standard deviation `sigma`,

```text
n per arm = 2 * (z_(1-alpha/2) + z_(1-beta))^2 * sigma^2 / delta^2
```

With `alpha = 0.05` and power `0.80`, `2 * (1.959964 + 0.841621)^2 = 15.70`, the familiar
`16 * sigma^2 / delta^2`.

```text
revenue per visitor:  mu = $9.80, sigma = $52.00, target delta = 10% of mu = $0.98
z_sum = 1.959964 + 0.841621 = 2.801585
z_sum^2 = 7.84888 ; 2 * z_sum^2 = 15.69776
sigma^2 / delta^2 = 2704 / 0.9604 = 2815.49
n = 15.69776 * 2815.49 = 44197   (ILLUSTRATIVE)
```

The heavy tail is the story: a metric with a coefficient of variation of 5.31 needs 44,197 per arm
to resolve a 10% change, against 42,158 for a 6% relative change in a 9.6% proportion. Proportions
near the middle of the range are cheap to measure; spend metrics are expensive.

**(b) Unequal allocation.** If the control-to-treatment ratio is `k:1`, total sample size relative to
balanced allocation is multiplied by `(1 + k)^2 / (4k)`.

```text
k = 1   factor 1.000      balanced
k = 2   factor 1.125      total n +12.5% for the same power
k = 3   factor 1.333      total n +33.3%
k = 4   factor 1.5625     total n +56.3%
k = 9   factor 2.778      total n +177.8%
```

This is why "just hold back 10% as control" is not a 90/10 decision — it is a decision to buy the
same precision with 2.8x the traffic. Unequal splits are justified by risk containment (limit blast
radius on a payment flow change) rather than by precision.

**(c) Fixed-duration designs with traffic ramping.** If the eligible population is `m` per day and
the design requires `N` total, duration is `N / m` days, but the last cohort must be observed for
the full measurement window `w`:

```text
calendar days = N / m + w
```

Ignoring `+ w` is the classic "we hit the sample size so we can stop" error. Assignments stop, but
the newest cohort's 14-day conversion window is still filling, and comparing a partially mature
cohort against a mature one biases the effect toward whichever arm matures faster.

**(d) MDE from a fixed sample size.** Invert the same equation to answer "what can this test see?":

```text
d_min = (z_(1-alpha/2) + z_(1-beta)) * sqrt(2 * p * (1-p) / n)
```

At `p = 0.096` and `n = 182,000` per arm the binary search on the exact formula returns
`d_min = 0.00275`, i.e. 0.275 pp or 2.87% relative. Halving traffic to 50,000 per arm raises the
floor to 0.528 pp (5.5% relative); quartering to 25,000 gives 0.751 pp (7.82%). Precision degrades
as the square root: to see half the effect you need four times the traffic.

**Validity condition.** These inversions assume the same normal approximation and the same single
look. When the design is sequential, the MDE must be computed against the boundary that applies at
the analysis time, not against 1.96, which raises the MDE by 3% to 8% depending on the spending
function.

### 4.3 Variants

**Ratio metrics (delta method).** For `R = Y / X` computed from per-unit aggregates with means
`mu_Y, mu_X`, standard deviations `sigma_Y, sigma_X` and correlation `rho`:

```text
Var(R) ~= [ sigma_Y^2 + R^2 * sigma_X^2 - 2 * R * rho * sigma_X * sigma_Y ] / (n * mu_X^2)
```

The derivation is a first-order Taylor expansion of `Y/X` around `(mu_Y, mu_X)`; the gradient is
`(1/mu_X, -mu_Y/mu_X^2)`, and applying `g' Cov g` gives the expression above. The naive formula
`sigma_Y / (mu_X * sqrt(n))` silently assumes the denominator is a constant.

Two directions of error matter, and which one you get depends on the sign of `rho`.

```text
case A   mu_X=6.0 sd_X=5.5   mu_Y=24.0 sd_Y=26.0   R=4.000
  rho=0.20   correct SE 0.0360   naive SE 0.0306   naive understates by 17.4%
  rho=0.50   correct SE 0.0286   naive SE 0.0306   naive OVERstates by 6.7%
  rho=0.90   correct SE 0.0135   naive SE 0.0306   naive overstates by 127%
```

So the naive ratio test is not conservative; it is wrong in whichever direction the covariance
happens to point. Revenue per session with strong per-user correlation is genuinely more precise
than the naive formula suggests, and clicks per impression with a weak numerator/denominator link
is genuinely less precise. The only defensible move is to compute the covariance.

**CUPED.** Choose a pre-period covariate `X` (same metric before the experiment, or a correlated
proxy), estimate `theta = Cov(Y, X) / Var(X)` on pooled pre-period or in-experiment data, and
analyse `Y_adj = Y - theta * (X - E[X])`.

```text
Var(Y_adj) = Var(Y) * (1 - rho^2)
n_required = n_plain * (1 - rho^2)

rho = 0.30  ->  variance factor 0.9100  ->  n factor 0.910   (9% traffic saved)
rho = 0.50  ->  variance factor 0.7500  ->  n factor 0.750   (25% saved)
rho = 0.55  ->  variance factor 0.6975  ->  n factor 0.697   (30% saved)
rho = 0.70  ->  variance factor 0.5100  ->  n factor 0.510   (49% saved)
```

With `rho = 0.55` the paywall design needs `42158 * 0.6975 = 29,405` per arm instead of 42,158. The
estimand is unchanged because `X` is measured before randomisation and is therefore balanced in
expectation; a covariate measured *after* assignment is a post-treatment variable and adjusting for
it can bias the effect.

**Cluster randomisation.** When `m` units per cluster are assigned together and outcomes share
intra-cluster correlation `ICC`:

```text
DEFF = 1 + (m - 1) * ICC
effective n = total units / DEFF
```

```text
m = 50,  ICC = 0.02  ->  DEFF = 1.98   20,000 units behave like 10,101
m = 200, ICC = 0.01  ->  DEFF = 2.99   20,000 units behave like 6,689
m = 20,  ICC = 0.05  ->  DEFF = 1.95   20,000 units behave like 10,256
```

The number of *clusters*, not the number of users, usually binds: 20 clusters per arm gives a
t-distribution with about 19 degrees of freedom for the cluster-level comparison, so the critical
value rises from 1.96 to roughly 2.09 and the design is fragile to one atypical cluster.

**Sequential designs.** Spending alpha over `K` looks at information fractions `t_k` with an
O'Brien-Fleming-like boundary `c / sqrt(t_k)`:

```text
K = 5 looks, c = 2.042 (calibrated by simulation so total two-sided Type I = 0.05)
look 1  t=0.2  z boundary 4.566
look 2  t=0.4  z boundary 3.229
look 3  t=0.6  z boundary 2.636
look 4  t=0.8  z boundary 2.283
look 5  t=1.0  z boundary 2.042
```

A Pocock-like constant boundary is easier to communicate and slightly cheaper in expected sample
size when the effect is large, but more expensive at 80% power: `2.180` for two looks, `2.416` for
five, `2.556` for ten. The constant boundary makes early stopping plausible; the O'Brien-Fleming
shape makes it nearly impossible to stop early on a borderline result, which is usually the honest
arrangement. Both cost sample relative to a single look, and the cost is measurable: with five
looks the required noncentrality rises from 2.802 to 2.845 under O'Brien-Fleming (n inflation 3.1%)
and to 3.099 under Pocock (n inflation 22.4%). The O'Brien-Fleming penalty is small precisely
because it spends almost nothing early.

**Bayesian variant.** With `Beta(a, b)` priors and `x` successes in `n` trials, the posterior is
`Beta(a + x, b + n - x)`. The decision quantity becomes `P(p_T > p_C)` or `P(p_T > 1.03 * p_C)`,
computed by integrating the product of the two posteriors. Two worked comparisons show why the
number is not interchangeable with a p-value.

```text
small experiment, flat Beta(1,1) priors
  control   205 / 2,400     treatment 240 / 2,400
  frequentist z = 1.742, two-sided p = 0.081, 95% CI [-0.18 pp, +3.10 pp]
  posterior P(p_T > p_C)          = 0.9585   (Monte Carlo, 100,000 draws, +/-0.0006)
  posterior P(p_T > 1.05 * p_C)   = 0.8852

paywall experiment, flat priors
  control  17,364 / 182,400   treatment 18,420 / 182,560
  posterior P(p_T > p_C)          = 1 - 1e-15  (about 8 posterior standard deviations)
  posterior P(p_T > 1.03 * p_C)   = 0.9999
```

The small experiment is the instructive case: `p = 0.081` fails a 5% test, while `P(p_T > p_C) = 0.96`
looks like a confident win under a flat prior. Neither number is wrong; they answer different
questions, and a ship rule written as "posterior probability above 0.95" would ship a change that
the frequentist test cannot distinguish from zero. The memo must state which rule is in force, and
the same rule must apply to every experiment in the programme.

### 4.4 Limits and counterexamples

**Counterexample 1 — the formula is not the design.** A team computes `n = 42,158` per arm, runs
for exactly 42,158 assignments, and stops. Their observed effect is +0.11 pp with a CI of
`[-0.06, +0.28]`. The formula was satisfied; the design was not, because the 14-day measurement
window meant the final cohort contributed immature conversions, diluting the estimate by roughly the
fraction of the window still open. A sample-size calculation constrains the *variance at analysis
time*, not the calendar.

**Counterexample 2 — the p-value threshold is not the decision.** With `p = 0.03` and a CI of
`[+0.0002, +0.0114]`, the test is "significant" and the lower bound is 0.02 pp. If the ship cost
(five engineer-weeks plus a support load) requires 0.4 pp to break even, the correct action is
no-ship even though `p < 0.05`. Statistical significance answers "is it zero?", practical
significance answers "is it worth the cost?", and only the second one is a business question.

**Counterexample 3 — power calculation on the wrong variance.** Sizing a ratio metric with the
naive formula when `rho = 0.20` understates the required `n` by 17.4% in variance terms, which is
about 8.6% in linear terms. A design that "should" have 80% power delivers roughly 74%, and the
team concludes the feature does not work.

**Boundary where the approximation breaks.** The normal approximation to the binomial requires
`n * p >= 10` in both arms, and the sample-size formula inherits that. For `p1 = 0.004` and a 20%
relative lift, the exact formula gives `n = 4.5 million` per arm, which is not a design, it is a
statement that the metric cannot be measured at the available traffic. At that point the honest
options are a coarser metric (sessions with any event, rather than the rare event itself), a longer
window, or a different decision process — not a smaller `n` with a promise to "look carefully".

A second boundary: when the true effect is far from the design's MDE, the normal approximation to
the *power function* is poor. At 80% power the approximation is accurate to a few percent; at 99%
power the same formula understates the required `n` by around 10%, because the power curve is
convex in that region.

## 5. Worked Numerical Example

The example is a paywall experiment at a subscription content site. All inputs are ILLUSTRATIVE
constructions chosen so the arithmetic is checkable by hand at the printed precision; none of them
is a company measurement.

**Setup.** Visitors hitting an article paywall are randomised 50/50 at the visitor level. Control
sees the existing modal with two plan options; treatment sees the modal with a single recommended
plan plus a "compare plans" link. Primary metric: trial start within 14 days of first paywall view,
computed per visitor. Guardrail: refund within 30 days per trial. Secondary: revenue per visitor.

### 5.1 Design

```text
p1 (control trial-start rate)   = 0.0960
r  (target relative lift)       = 0.06
p2 = p1 * (1 + r)               = 0.10176
d  = p2 - p1                    = 0.00576
alpha two-sided 0.05 -> z = 1.959964
power 0.80          -> z = 0.841621
n per arm                       = 42158
total assignments               = 84316
```

### 5.2 Observed data

```text
arm        assigned    trial starts    rate
control    182,400     17,364          0.095197   (9.5197%)
treatment  182,560     18,420          0.100898   (10.0898%)
```

The design asked for 42,158 per arm and the run delivered 182,400 — 4.33x the design size. That is
not a mistake: the team ran the pre-registered 14-day window instead of stopping at the minimum,
and the extra traffic is what makes the guardrail testable at all.

### 5.3 Check 1 — absolute effect and its standard error

```text
p_C_hat = 17364 / 182400 = 0.0951974
p_T_hat = 18420 / 182560 = 0.1008981
d_hat   = 0.1008981 - 0.0951974 = 0.0057007        (+0.5701 pp)

Var(p_C) = 0.0951974 * 0.9048026 / 182400
         = 0.0861395 / 182400 = 4.72256e-7
Var(p_T) = 0.1008981 * 0.8991019 / 182560
         = 0.0907176 / 182560 = 4.96922e-7
Var(d)   = 9.69178e-7
SE(d)    = sqrt(9.69178e-7) = 0.00098447

z = 0.0057007 / 0.00098447 = 5.7907
two-sided p = 7.0e-9
95% CI = 0.0057007 +/- 1.959964 * 0.00098447 = [0.003771, 0.007630]
```

**Interpretation.** The effect is 0.570 pp with an interval from 0.377 pp to 0.763 pp. Every point
in that interval is positive, and the design's own MDE was 0.576 pp — the observed effect is almost
exactly the size the design was built to detect, which is a coincidence of the construction and a
reminder that a design at exactly its MDE has only 50% power in the region below that point.

### 5.4 Check 2 — relative effect and its interval

Reporting a lift as `d_hat / p_C_hat` and attaching a `+/- 1.96 * SE / p_C` interval is the common
shortcut. It is approximately right at this size, and it is worth showing why.

```text
naive relative: 0.0057007 / 0.0951974 = 0.05988  = +5.988%
naive CI:       [+0.003771/0.0951974, +0.007630/0.0951974] = [+3.962%, +8.015%]
```

The log-ratio approach is the better construction because the relative effect is a ratio and its
sampling distribution is skewed:

```text
log RR = ln(0.1008981 / 0.0951974) = 0.058160
SE(log RR) = sqrt( (1-p_T)/(p_T*n_T) + (1-p_C)/(p_C*n_C) )
           = sqrt( 0.8991019/(0.1008981*182560) + 0.9048026/(0.0951974*182400) )
           = sqrt( 4.8814e-5 + 5.2126e-5 ) = sqrt(1.00940e-4) = 0.0100469
95% CI on log scale = 0.058160 +/- 1.959964*0.0100469 = [0.038468, 0.077852]
95% CI on ratio     = [exp(0.038468), exp(0.077852)] = [1.03922, 1.08096]
```

**Interpretation.** "+5.99%, 95% CI [3.96%, 8.02%]" from the naive method and "[3.92%, 8.10%]" from
the log method agree to two decimal places here because the effect is small relative to the rate.
At a 40% relative lift on a 2% base rate the two diverge by more than a percentage point, and the
log method is the correct one.

### 5.5 Check 3 — the decision question, not the significance question

The pre-registered ship threshold was a 3% relative lift, chosen because a 3% lift on trial starts
covers the maintenance cost of the extra plan-comparison screen. The decision quantity is
`P(RR > 1.03)` under the log-normal approximation:

```text
z = (0.058160 - ln(1.03)) / 0.0100469
  = (0.058160 - 0.029559) / 0.0100469
  = 0.028601 / 0.0100469 = 2.8468
P(RR > 1.03) = 1 - Phi(2.8468) = 0.00221
```

That is the probability the true lift is *below* the threshold, so `P(RR > 1.03) = 0.9978`. The
result clears both the significance bar and the practical bar with room, and the memo can say so
without hedging.

Compare with a threshold of 8% instead: `z = (0.058160 - 0.076961)/0.0100469 = -1.8715`, giving
`P(RR > 1.08) = 0.031`. The same data supports a 3% ship decision and refuses an 8% one, which is
the point of stating the threshold before the test rather than after.

### 5.6 Check 4 — the guardrail

Refund rate per started trial: control 214 refunds in 17,364 trials (1.2324%), treatment 229 in
18,420 (1.2432%).

```text
p_C = 214/17364  = 0.0123243
p_T = 229/18420  = 0.0124321
diff = +0.0001078   (+0.0108 pp)
SE = sqrt(0.0123243*0.9876757/17364 + 0.0124321*0.9875679/18420)
   = sqrt(7.0081e-7 + 6.6644e-7) = sqrt(1.36725e-6) = 0.0011693
z (one-sided) = 0.0001078 / 0.0011693 = 0.0922
```

**Interpretation.** The point estimate moves in the wrong direction by 0.011 pp, with a standard
error twelve times larger. The design cannot distinguish this from zero, and more importantly it
cannot exclude a harmful increase. The correct statement is not "no guardrail impact" but "the
guardrail is underpowered: at 17-18k trials per arm, a true increase of 0.10 pp (8% relative) is
detected only 21.1% of the time at one-sided alpha 0.05". Sizing the guardrail properly, for a
0.20 pp absolute increase at 80% power, needs 51,593 trials per arm — 2.8x the available sample —
and a 0.10 pp increase would need 198,730 per arm, which is not a design but a statement that this
guardrail cannot be measured at the current trial volume.

### 5.7 Sensitivity

Recompute the primary result under three perturbations, because a conclusion that survives only one
analysis choice is not a conclusion.

```text
1. Exclude the first 24 hours (bot sweep + cache warm)
   control   17,201/180,900 = 0.0950857
   treatment 18,251/181,050 = 0.1008064
   diff = 0.0057207   SE = 0.0009881   z = 5.790     -> conclusion unchanged

2. Analysis at session level instead of visitor level
   sessions 412,300 C and 431,900 T, same per-session rates
   naive SE drops to 0.000647, z rises to 8.81 -- and the interval is now too narrow,
   because sessions within a visitor are correlated. Wrong, not better.

3. Trim the top 1% of visitors by pageviews
   rates move to 0.0948 and 0.1006, diff 0.005800, SE 0.000983, z = 5.90 -> unchanged
```

The second perturbation is included precisely because it looks like an improvement. A smaller
standard error obtained by analysing a finer unit is the most common way a correct design produces
an incorrect p-value.

## 6. Visual / Geometric Intuition

### 6.1 The two-distribution picture

Under the null, `d_hat` is centred at zero with standard error `SE_0`. Under the alternative at the
design's MDE, it is centred at `d` with standard error `SE_1`, which is very slightly larger. The
alpha region is the tail of the null beyond `1.96 * SE_0`; the power is the mass of the alternative
beyond the same threshold.

```text
                 null distribution              alternative at MDE
                        |                                |
   reject left          |          fail to reject       |          reject right
   -----|--------------[==]----------------------------[==]------------|-----
      -1.96SE_0          0                             d            1.96SE_0
```

Two geometric facts follow, and both have practical consequences.

The distance between the two centres, in units of `SE_0`, is `(z_a + z_b)` — 2.80 for the standard
80%-power design. That constant is where the `16` in `16 * sigma^2 / d^2` comes from: `2 * 2.8016^2 =
15.70`. Designs that quote `n = 16 sigma^2/d^2` are quoting a picture of two normal curves separated
by 2.8 standard errors.

Because separation scales as `1 / sqrt(n)` while the effect is fixed, doubling precision (halving
the SE) requires quadrupling `n`. The curve of `n` against target effect is a hyperbola, and the
practical consequence is that the last 20% of MDE reduction costs more than the first 80%.

### 6.2 The confidence-interval-as-ruler picture

A CI is a ruler whose length is fixed by the data and whose position is the point estimate. The
decision rule is "does the ruler clear the threshold?" — not "is the ruler on the right side of
zero?". Plotting the threshold as a vertical line and the interval as a horizontal bar makes the
four possible outcomes visible at once:

```text
                         0        threshold (3%)
   significant, below    |----[====]                    ship is a mistake
   significant, above    |        [====]----             ship
   not significant, above|   [==========]                 underpowered, promising
   not significant, below| [=====]                       stop
```

The third row is the expensive one. A wide interval that sits entirely above zero but crosses the
threshold looks like a near-miss and is usually a design failure: the interval's width, not its
position, is the thing to fix, and the fix is traffic or a better estimator.

### 6.3 The funnel picture of dilution

The ITT effect is a weighted average over exposure states, so a delivery gap flattens the observed
effect geometrically.

```text
assigned 100%
  |
  +-- exposed 78%  ---> effect tau_TOT
  |
  +-- not exposed 22% ---> effect 0 (by exclusion restriction assumption)

observed ITT = 0.78 * tau_TOT
tau_TOT = ITT / 0.78
```

With ITT `= 0.5701 pp`, `tau_TOT = 0.5701 / 0.78 = 0.731 pp` (0.7310 pp to four decimals), or 7.68%
relative against the 9.5197% control rate. The same shrinkage applies to power: a design powered for
a 6% ITT effect on 78% exposure is powered for a 7.7% treatment effect, and if the product team's
mental model is the 7.7% number, the design will look underpowered against their expectation while
being exactly right for the decision.

### 6.4 The variance budget as a stacked bar

Every experiment's variance has components, and knowing which dominates tells you what to buy.

```text
proportion metric, n per arm 42,158
  between-unit Bernoulli variance   100%      nothing else to remove
  (CUPED helps only via a pre-period correlate of the same Bernoulli)

revenue-per-visitor metric
  heavy tail (top 1% of visitors)   ~55%      trimming or winsorising, log-ish metrics
  session-count variance            ~30%      CUPED on prior-period sessions
  price/plan mix                    ~15%      stratify by plan tier at assignment
```

On the proportion metric there is nothing to buy except traffic, which is why product teams
gravitate to revenue metrics and then discover the tail. On the revenue metric, the largest single
lever is usually a pre-period covariate, not a larger sample.

### 6.5 The peeking picture

Under repeated looks, the test statistic traces a random walk in "z units", and each look is a
fresh chance to cross a fixed boundary. The walk's standard deviation grows as `sqrt(t)` while the
boundary is fixed, so the crossing probability accumulates.

```text
looks taken   naive 1.96 boundary   Type I error (simulated, 40,000 replications)
     1                1.96                     0.0500
     5                1.96                     0.1416
    10                1.96                     0.1939
    20                1.96                     0.2441
```

The geometric reading: a fixed boundary is a horizontal line and the walk spreads as a cone, so the
cone crosses the line early and often. A boundary of the form `c / sqrt(t)` is a hyperbola that
tracks the cone's edge, which is why alpha-spending boundaries are shaped that way rather than
being a constant.

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
