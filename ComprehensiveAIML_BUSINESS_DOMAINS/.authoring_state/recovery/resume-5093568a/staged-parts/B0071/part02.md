
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
until 42,158 assignments exist in each arm, and stops. Their observed effect is +0.11 pp with a CI of
`[-0.06, +0.28]`. The formula was satisfied; the design was not, because the 14-day measurement
window meant the final cohort contributed immature conversions. If conversion timing differs by arm
(a treatment that speeds up the path to trial concentrates conversions early), the immature tail
under-contributes in the faster arm and the estimated effect is pulled toward zero by roughly the
unfinished share of the window. A sample-size calculation constrains the *variance at analysis
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
