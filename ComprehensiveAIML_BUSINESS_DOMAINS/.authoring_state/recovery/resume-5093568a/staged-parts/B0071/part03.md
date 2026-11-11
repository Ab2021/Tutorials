
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
P(RR < 1.03) = Phi(2.8468) = 0.00221
P(RR > 1.03) = 1 - 0.00221 = 0.99779
```

The result clears both the significance bar and the practical bar with room: there is a 99.8% chance
under the normal approximation that the true lift exceeds the threshold, and the memo can say so
without hedging.

Compare with a threshold of 8% instead: `z = (0.058160 - 0.076961)/0.0100469 = -1.8713`, giving
`P(RR > 1.08) = 0.0306`. The same data supports a 3% ship decision and refuses an 8% one, which is
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
