# Scoring Probabilistic Forecasts — CRPS, Pinball Loss, Log Score, and Calibration Diagnostics

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 topics 08–09; Module 08 topics 07–08
> Version 2026.5

---

## 1. Why this topic exists

Two teams present a new probabilistic model. Team A says its continuous ranked probability
score (CRPS) improved by 0.04 cases per item-week. Team B says its 90% interval now covers 89%
of outcomes instead of 74%. Both claims are true, both are computed on the same data, and
neither tells the room whether to deploy the model, because they answer different questions and
the organisation has no rule for combining them.

The deeper problem is upstream. Somebody had to choose the score, and that choice silently
decided which model won. A score is a loss function, and a loss function encodes a claim about
what "better" means. Choose one that rewards concentration and the sharpest model wins even
when it is wrong in the tail. Choose one that rewards coverage and the widest model wins even
when it says nothing. Average over a portfolio and five catastrophic item–location combinations
outvote nineteen hundred ordinary ones. None of these choices is neutral, and almost none is
written down.

Three failures recur:

1. **The improper score that looks sensible.** "Mean absolute error of the median forecast" is
   computed on a probabilistic model, called its accuracy, and used to select between models
   that differ only in their tails. The metric cannot see the tails, so it selects the better
   centre and the worse tail, and the inventory policy consumes the tail.
2. **The calibration chart used as a verdict.** A flat probability integral transform (PIT)
   histogram appears on slide 6 and the room concludes the forecast is good. A calibrated
   forecast that is 400 units wide is calibrated and commercially useless.
3. **The average over a skewed portfolio.** Mean CRPS across 2,000 series falls by 8%, and the
   8% comes entirely from six series nobody replenishes at scale. The items that generate the
   service failures are unchanged.

**Where the surrounding topics sit.** Module 07 topic 09 treats point-error metrics — mean
absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE),
mean absolute scaled error (MASE), bias — and their cost interpretation; they are not re-taught
here, and section 5 explains only why none can score a distribution. Module 08 topic 07 treats
what a predictive distribution is for, topic 08 conformal prediction, and topic 10 the decision.
This topic is the bridge: whether the distribution is any good, and by what standard.

**Who is blocked.** The demand planning lead who cannot retire an incumbent because it wins on
the template metric. The data scientist whose better tail model loses a bake-off on mean absolute
error. The finance partner handed a percentage when asked to accrue for forecast risk.

> **Note:** Nothing here claims a single number can summarise a forecast. The argument is the
> opposite: a score answers exactly one question, the question is chosen by whoever picks the
> score, and professional practice is to state the question before the answer appears.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** propriety and strict propriety, and **demonstrate** that an improper score is
   maximised by a forecast the forecaster does not believe.
2. **Explain** why MAE, RMSE, MAPE, and MASE are at best consistent for a single functional and
   therefore cannot rank predictive distributions.
3. **Compute** the Brier score (squared probability error for an event), pinball loss, CRPS, and
   log score by hand, with units and sign conventions stated.
4. **Derive** CRPS as an integrated pinball loss, and **state** when a finite quantile grid
   approximates it and in which direction it errs.
5. **Choose** between marginal and joint scores for multi-horizon or multi-item forecasts, and
   justify the choice against the decision.
6. **Read** a PIT histogram, a reliability diagram, and a coverage table, and **separate** what
   each says from what the score says.
7. **Design** a comparison protocol — evaluation set, rolling origin, paired test, uncertainty —
   and state what would invalidate it.
8. **Audit** a published scoreboard against a ten-field reporting standard and identify which
   missing field most changes the conclusion.

Each outcome maps to an exercise in section 12 and to an answer in section 13.

## 3. The core mental model

Hold two pictures. The first is a landscape: every possible reported distribution is a point on
a surface, and the score is the height. Propriety says the true distribution sits at the bottom
of the valley. An improper score tilts the valley and the bottom moves to a lie. The second is
the distinction between a per-case score and an expected score. They are different objects, and
confusing them is the most common error in reading a scoreboard.

```text
   THE PROPRIETY LANDSCAPE — three-outcome demand, support 10 / 20 / 30 cases

   truthful F            sharpened G           point mass H
   10  0.20              10  0.05              10  0.00
   20  0.50              20  0.90              20  1.00
   30  0.30              30  0.05              30  0.00

   PANEL 1 — one realised outcome, y = 20          (lower is better except log)
   score                     F        G        H     who wins
   -----------------------   ------   ------   ----- ---------
   CRPS, cases                1.300    0.050   0.000 H
   Brier, 3 classes           0.380    0.015   0.000 H
   log score (higher better) -0.693   -0.105   0.000 H
   pinball at q = 0.8         2.000    0.000   0.000 H
   -> on a single case every proper score prefers the sharpened forecast.
      propriety is a property of expectations, not of individual outcomes.

   PANEL 2 — expected score under the TRUTHFUL F    (lower is better except log)
   score                     F        G        H      bottom of valley
   -----------------------   ------   ------   -----  ------------------
   CRPS, cases                3.700    4.550   5.000  F   proper
   Brier, 3 classes           0.620    0.865   1.000  F   proper
   log score (higher better) -1.030   -1.551   -inf   F   proper, unbounded
   pinball at q = 0.8         1.800    2.800   2.800  F   proper at that level
   max probability            0.500    0.900   1.000  H   IMPROPER
   interval width (80%)      20        0       0      H   IMPROPER
   -> the two improper scores put a confident liar at the optimum.
      no quantity of data repairs a score with the wrong valley in it.
```

Redraw both panels from memory and you can reconstruct most of this topic: panel 1 explains why a
demonstration on one series or one week is theatre, panel 2 why the score is a design decision
with a mathematical property attached rather than a neutral summary. Four consequences organise
sections 5 to 11.

**Propriety is a floor, not a ceiling.** It guarantees the truth is optimal *in expectation* and
nothing about variance: an honest forecast can lose to a lucky one over 40 origins.

**Strictness decides what can hide.** A strictly proper score is uniquely optimised by the truth;
a merely proper one admits a class of models that tie, and an uninformative forecast can live
inside that class. Averaging pinball over a finite grid is proper for those levels and not
strictly proper for the distribution between them.

**Score and calibration measure different objects.** The score combines calibration and
sharpness with weights it chooses; the chart shows one component. A forecast can improve its
score while getting worse calibrated if it gains enough sharpness, and can be perfectly
calibrated while scoring worse than climatology.

**Aggregation is a second loss function.** After choosing a proper score you choose how to
average it across series, horizons, and levels; the mean is dominated by a few members, which
makes it improper as an aggregator for a skewed portfolio. That choice is as consequential as
the first and is usually made by accident.

## 4. Definitions and vocabulary

Formulas appear in section 5. What follows is the vocabulary needed to read a scoreboard
critically, with the misuse that changes the answer and the unit that decides whether two
numbers are comparable.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Scoring rule | A function `S(F, y)` mapping a forecast distribution and a realisation to a number | Called a "metric" and assumed neutral | Depends on the rule |
| Proper scoring rule | A rule whose expectation is optimised by reporting the true distribution | Assumed to guarantee good performance on a finite sample | Same as the rule |
| Strictly proper | Proper with a unique optimum, so any other distribution scores worse in expectation | Confused with proper; strictness decides whether an uninformative forecast can tie | Property, not a number |
| Consistency | The rule's expectation is optimised by the true value of a named functional | Applied to a distribution when only a functional is scored | Property, not a number |
| Brier score | `(p − o)²` for a binary event with `o ∈ {0,1}`; multicategory form sums over classes | Quoted for a distribution rather than an event | Squared probability, 0 to 1 (binary) or 0 to 2 |
| Pinball loss | `q·e` for `e ≥ 0` and `(q − 1)·e` for `e < 0`, at quantile level `q` | Used once and called a distribution score | Units of the forecast variable |
| Quantile level | The probability below which the forecast claims the outcome falls | Treated as a confidence level, which reverses the reading | Dimensionless, 0 to 1 |
| CRPS | Integral of squared differences between the forecast CDF and the indicator at the realisation | Computed from a point forecast and called a distribution score | Units of the forecast variable |
| CRPSS | CRPS skill score, `1 − CRPS/CRPS_ref` | Reported without naming the reference | Dimensionless proportion |
| Log score | `ln f(y)` for a density, `ln P(y)` for a discrete forecast; higher is better | Reported as a positive loss without a sign convention | Natural-log units, unbounded below |
| Energy score | Multivariate generalisation of CRPS over a vector realisation | Assumed sensitive to correlation; it largely is not | Units of the vector norm |
| Variogram score | Sum over pairs of squared differences between forecast and observed pairwise absolute differences | Assumed strictly proper; it is proper but not strict | Depends on the power `p` |
| Ranked probability score | Sum of squared cumulative-probability errors over ordered categories | Applied to continuous variables without discretisation | Squared probability |
| Calibration | Agreement between stated probabilities and observed frequencies, averaged over the sample | Treated as equivalent to accuracy | Property, diagnosed graphically |
| Sharpness | Concentration of the predictive distribution | Rewarded on its own, which is gameable | Width in units, or entropy |
| Resolution | How much the forecast varies with the outcome it predicts | Confused with reliability | Component of a decomposition |
| Reliability | The part of a decomposed score attributable to miscalibration | Confused with dependability | Component, same units as the score |
| PIT | Probability integral transform, `u = F(y)`; uniform when calibrated | Applied to discrete forecasts without randomisation | Dimensionless, 0 to 1 |
| Randomized PIT | `F(y−) + U·(F(y) − F(y−))` with `U ~ Uniform(0,1)`; restores uniformity for discrete forecasts | Omitted, producing spurious spikes at atoms | Dimensionless, 0 to 1 |
| PIT histogram | Binned counts of PIT values across origins | Read as a formal test rather than a diagnostic | Counts per bin |
| Reliability diagram | Nominal probability against observed frequency, or nominal against empirical coverage | Plotted with too few cases to distinguish anything | Proportions |
| Empirical coverage | Share of outcomes inside a stated interval | Compared with nominal without a standard error | Proportion |
| Nominal coverage | The coverage the interval claims | Reported as if achieved | Proportion |
| Interval score | Penalty combining interval width with twice the distance of misses | Averaged across levels without weights | Units of the forecast variable |
| Skill score | Score expressed relative to a reference forecast | Reported without the reference's identity | Dimensionless proportion |
| Loss differential | `d_t = S_A,t − S_B,t`, the paired difference in score at one evaluation unit | Averaged without accounting for dependence | Units of the score |
| Diebold–Mariano test | Test of equal expected loss using the mean loss differential with a long-run variance estimate | Applied to nested models without adjustment | Test statistic, dimensionless |
| Model confidence set | The set of models not distinguishable from the best at a stated confidence | Reported as a ranking when it is a set | Set of model identifiers |
| Reference forecast | The benchmark a skill score uses, typically naive or climatological | Left unnamed, making the skill score uninterpretable | A forecast, not a number |
| WSPL | Weighted scaled pinball loss, the probabilistic metric of the M5 Accuracy competition | Compared across competitions with different quantile grids | Dimensionless, scaled |
| WRMSSE | Weighted root mean squared scaled error, the M5 point metric | Used to rank probabilistic models | Dimensionless, scaled |
| Aggregation weight | Weight applied to a series, level, or horizon when scores are pooled | Left implicit; equal weight is itself a claim | Dimensionless, sums to one |

Three pairs do the most damage when confused.

**Proper versus strictly proper.** Proper means the truth is *an* optimum; strict means it is
*the* optimum. Most textbook statements are about the strict version and most implementations
are the weak one. If your score is only proper, an uninformative forecast can sit at the same
height as the truth, and a selection loop will choose it when sampling noise favours it.

**Calibration versus accuracy.** Calibration is a property of the probabilities you issued;
accuracy is a property of outcomes relative to them. Climatology is calibrated by construction
and useless. A slightly overconfident but highly discriminating model beats climatology on every
proper score. Neither property implies the other, which is why a scoreboard needs both columns.

**Score versus skill score.** A CRPS of 1.62 cases means nothing without a reference. A CRPSS of
0.33 says the model removed a third of the reference's CRPS, and that statement survives a change
of item. Report both, and name the reference every time.

## 5. The mechanics, step by step

This section builds a scoring protocol in seven phases and twenty-five steps. The order is
deliberate — object, propriety, formula, calibration, comparison, aggregation — and
organisations routinely run it backwards, starting from the metric in the template, which is why
so many scoreboards are internally consistent and commercially empty.

### 5.1 Phase A — Fix what is being scored

**Step 1. Name the object: distribution, quantile set, interval, or point.** Four objects, four
scores. A distribution can be scored with CRPS or the log score; a set of nine quantiles with
pinball at nine levels, which claims nothing between the levels; an interval with the interval
score; a point with MAE, which is CRPS for a degenerate distribution. Almost every scoring
dispute is a dispute about which object is on the table.

**Step 2. Name the decision functional.** If the replenishment rule consumes the 0.95 quantile,
pinball at 0.95 is the selection criterion and CRPS is a diagnostic. If the policy evaluates
expected cost across the whole distribution, the distribution is the object. Scoring the
distribution when the decision uses one quantile selects the model that is better on average and
worse exactly where the money is.

**Step 3. Name the population, grain, horizon, and exclusions.** Which series, origins, lead
times, and vintages, and what is removed. Promotional weeks included or excluded, item level or
category level, are different questions. Write the six fields from Module 07 topic 09 into the
evaluation contract before running anything.

**Step 4. Name what a better score would change.** A model release, a retraining cadence, a
supplier conversation, a safety-stock parameter. If nothing follows from a score improving, the
measurement will drift toward whatever is easy to improve.

### 5.2 Phase B — Check propriety before anything else

**Step 5. Write the score's expectation and find its optimum.** For a forecast `F` and a true
distribution `G`, a score `S` is proper if `E_{Y∼G}[S(G, Y)] ≤ E_{Y∼G}[S(F, Y)]` for every `F`,
and strictly proper if equality holds only when `F = G`. It is a two-line check that most
organisations never perform, and it is the minimum requirement because without it the score has
a gradient pointing away from the truth.

**Step 6. See why the point-error family fails this test.** Each point metric is consistent for
one functional of the distribution and blind to everything else:

| Metric | Consistent for | Cannot see | Consequence for a distribution |
|---|---|---|---|
| MAE | The median | Tails, spread, shape | Distributions with the same median tie |
| RMSE | The mean | Tails beyond the second moment, asymmetry | A heavy right tail and a light one can tie |
| MAPE | No stable functional | Scale, zeros, the upper tail | Undefined or explosive at low actuals |
| MASE | The median, scaled by a naive benchmark | Everything the scaling removes | Ranks models, not distributions |
| Bias | The mean signed error | Magnitude and shape | A diagnostic, not a score |

The point is not that these metrics are bad; they are excellent at what they do, and Module 07
topic 09 maps them to decisions. The point is that they answer a question about one functional
while a probabilistic forecast is a claim about a whole distribution. Scoring a distribution by
its median is like reviewing a book by its page count: measurable, comparable, and unrelated to
the thing you care about.

**Step 7. Choose the score that matches the object.**

| Object | Score | Strictly proper | Note |
|---|---|---|---|
| Binary event | Brier score | Yes | Building block of CRPS |
| Multi-category | Multi-category Brier, ranked probability score | Yes | Ordinal outcomes favour the ranked version |
| Single quantile | Pinball at that level | Consistent, not strict for the distribution | The decision score when one quantile is used |
| Univariate distribution | CRPS | Yes, finite first moment | Reduces to MAE for a point forecast |
| Univariate density | Log score | Yes | Unbounded penalty; needs support coverage |
| Multivariate vector | Energy score | Yes, finite first moment | Weak on correlation |
| Multivariate vector | Variogram score | Proper, not strict | Sensitive to dependence, blind to common bias |
| Full path | Energy or variogram on the vector | As above | Marginal scores ignore dependence |

### 5.3 Phase C — Compute the univariate scores

**Step 8. The Brier score, in both forms.** For a binary event with forecast probability `p` and
outcome `o ∈ {0,1}`, and for a `K`-class outcome with one-hot vector `o`:

```text
   BS = (p - o)^2                              range 0 to 1, lower is better
   multicategory: BS = sum_k (p_k - o_k)^2     range 0 to 2, lower is better
```

It is strictly proper, it decomposes into reliability, resolution, and uncertainty, and it is
the discrete ancestor of CRPS: CRPS is the limit of the Brier score as the number of thresholds
grows without bound. That lineage explains why the reliability term in a CRPS decomposition is
read the same way.

**Step 9. Pinball loss, with the asymmetry made explicit.** For a quantile forecast `x` at level
`q` and a realisation `y`, with `e = y − x`:

```text
   L_q(y, x) = q * e            if e >= 0    (under-forecast)
   L_q(y, x) = (q - 1) * e      if e <  0    (over-forecast)
```

An under-forecast costs `q` per unit and an over-forecast `1 − q`; at `q = 0.8` the ratio is 4:1,
exactly the cost ratio a business with a critical fractile of 0.8 has already declared. At
`q = 0.5` the loss is `0.5|e|`, so doubled pinball at the median equals MAE. That is the bridge
from Module 07's metrics, and it is why MAE is not wrong for a median forecast, only
insufficient for a distribution.

Pinball is *consistent* for the `q`-quantile: the expected loss is minimised by the true
quantile and nothing else. It is not strictly proper for the distribution, because many
distributions share the same `q`-quantile. A set of nine pinball scores therefore says nine
quantiles are right and nothing about the probabilities between them.

**Step 10. CRPS, in two forms that must agree.** The integral form is the definition:

```text
   CRPS(F, y) = integral over x of (F(x) - 1{x >= y})^2 dx
```

It is the squared distance between the forecast cumulative distribution and the step function
that jumps from 0 to 1 at the realised value, and its units are the units of the forecast
variable: a CRPS of 1.62 cases is meaningful, a CRPS of 1.62 is not. For a degenerate forecast at
`m`, `F(x) = 1{x ≥ m}` and the integral collapses to `|y − m|`, so CRPS is MAE. That identity is
the cleanest statement of what CRPS adds: MAE extended to distributions, charging for shape as
well as centre. The computational form is the quantile integral:

```text
   CRPS(F, y) = 2 * integral over q in [0,1] of L_q(y, F^{-1}(q)) dq
```

With the doubling convention, CRPS is twice the average pinball loss over all levels. In
practice you approximate the integral on a grid, and the approximation is biased by an amount
that depends on the curvature of the quantile function and on where levels fall relative to
atoms. State the grid: two CRPS numbers computed on different grids are not the same number.

**Step 11. The log score, and the price of a zero.** For a density `f` the log score is `ln f(y)`
and the loss is `−ln f(y)`; for a discrete forecast, `ln P(y)`. It is strictly proper and
*local*: it depends only on the density at the realisation. Locality is both its virtue and its
vice. A forecast assigning zero density to an outcome that occurs scores negative infinity, and
a mean containing one such case is negative infinity whatever the other cases were — the score
reporting a falsified claim of impossibility, not a bug. The practical responses are to guarantee
support (mixture components, a floored density, full support on the plausible range) and to
report boundary cases separately. Never replace an infinity with a large finite number: that
converts an honest catastrophe into a fabricated statistic.

> **Note:** The unbounded penalty is why meteorological practice migrated to CRPS for continuous
> variables and kept the log score for events and for density forecasts with verified support.

**Step 12. Compute a skill score, not just a score.** For CRPS,
`CRPSS = 1 − CRPS_model / CRPS_reference`: positive beats the reference, zero matches it,
negative is worse than doing something simple. Name the reference — usually climatology, the
unconditional distribution of the series, or the incumbent's distribution, not the naive point
forecast.

### 5.4 Phase D — Multivariate and multi-horizon objects

**Step 13. Energy score for a vector.** For a forecast distribution `P` over `R^k` and a
realisation `y`, `ES(P, y) = E‖X − y‖ − 0.5·E‖X − X′‖`, with `X` and `X′` drawn independently
from `P`. It is strictly proper for distributions with finite first moment and reduces to CRPS
when `k = 1`. Its weakness is documented: it is poorly sensitive to misspecified correlation, so
a forecast with the right marginals and the wrong dependence can score well.

**Step 14. Variogram score for the same object.** The variogram score of order `p` compares
pairwise absolute differences:

```text
   VS_p(P, y) = sum over i,j of w_ij * ( |y_i - y_j|^p - E_P|X_i - X_j|^p )^2
```

It is proper but not strictly proper: a common bias applied to every component cancels inside
every difference and is invisible. Weights `w_ij` emphasise pairs that matter, such as adjacent
horizons or items in the same family. The original study found `p = 0.5` or `p = 1` more
discriminating than `p = 2`, and found the variogram score clearly better than the energy score
at detecting wrong correlation structure.

**Step 15. Decide how to treat horizons.** A four-week-ahead forecast is a four-dimensional
object, and three treatments are defensible but disagree:

- **Marginal sum.** Score each horizon with CRPS and average. Interpretable, blind to dependence:
  identical marginals with opposite dependence tie.
- **Joint score.** Score the four-vector with an energy or variogram score. Sensitive to
  dependence, harder to explain, dominated by the largest-scale component unless standardised.
- **Decision-weighted marginal.** Weight horizons by economic importance, often heaviest where
  review period plus lead time falls. Defensible if the weights are published.

There is no consensus: meteorology tends toward joint scores because dependence is the object;
supply chain practice toward marginal scores because planners consume one horizon at a time.
State which you used, and never compare a marginal sum with a joint score.

### 5.5 Phase E — Calibration as a separate instrument

**Step 16. Compute the PIT on data the model never saw.** For each origin, compute `u = F(y)`.
For discrete forecasts, randomise within the atom: `u = F(y−) + U·(F(y) − F(y−))`. Without
randomisation a count forecast produces spurious spikes at atom boundaries and the histogram
becomes unreadable.

**Step 17. Read the histogram shape.** Four canonical shapes cover most of what you will see.

| Shape | Reading | Typical cause | Effect on the score |
|---|---|---|---|
| Flat | Consistent with calibration | None | None; calibration contributes no penalty |
| U-shaped | Too many outcomes in both tails; intervals too narrow | Underestimated uncertainty, missing structural error | Raises CRPS through the reliability term |
| Humped | Too few outcomes in the tails; intervals too wide | Overdispersion, or a policy band added on top | Raises CRPS through lost resolution |
| Triangular or sloped | Outcomes systematically above or below the forecast | Bias, or a distribution shifted in one direction | Raises CRPS and will not be fixed by widening |

Module 08 topic 07 gives the fuller six-row version, including the spike at 0 or 1 that signals
a support error. The four above are the ones you will diagnose in a meeting.

**Step 18. Plot reliability and coverage, and put a standard error on the coverage.** A
reliability diagram plots nominal against empirical coverage; the number that matters is whether
the deviation exceeds sampling noise. For nominal coverage `p` on `n` independent origins the
standard error is `sqrt(p(1−p)/n)`. At `p = 0.90` and `n = 40` that is 0.047, so a
two-standard-error band is roughly ±9.5 percentage points: observing 82.5% is inside it and
proves nothing. At `n = 400` the band narrows to ±3.0 points and the same deviation becomes
evidence.

**Step 19. State the two-way independence of score and calibration.** A good score does not
prove calibration, and calibration does not prove a good score. The first is easy to
demonstrate: a sharp but biased forecast can gain resolution faster than it loses reliability
and post a respectable CRPS. The second is trivial: climatology is calibrated by construction,
has zero resolution, and its CRPS equals the climatological uncertainty. Report both, always,
and never let one stand in for the other.

### 5.6 Phase F — Compare honestly

**Step 20. Fix the evaluation set for every candidate.** Same origins, horizons, vintages,
actuals, treatment of stock-outs, and exclusions. A candidate evaluated on 38 origins against an
incumbent evaluated on 42 is described, not compared. Publish the count.

**Step 21. Pre-register the protocol and the rolling-origin scheme.** Write down the score,
weights, grid, horizon, exclusions, and reference before any candidate is scored; the mechanics
are Module 07 topic 08. What matters here is that origins cannot be tuned after the fact.

**Step 22. Use paired differences, not two independent means.** For each evaluation unit `t`,
compute `d_t = S_A,t − S_B,t`. The estimate is the mean of a paired series, far less noisy than
the difference of two separately estimated means because common variation cancels. Report the
mean difference, its standard error, and a confidence interval.

**Step 23. Handle dependence in the difference series.** Multi-step forecasts overlap, so `d_t`
is autocorrelated and the naive standard error is too small. The Diebold–Mariano test uses a
heteroskedasticity- and autocorrelation-consistent estimate of the long-run variance; the
Harvey–Leybourne–Newbold correction improves its size in small samples. It assumes covariance
stationarity of the loss differential and non-nested models; nested models need a different null
distribution and a test that ignores this over-rejects.

**Step 24. Control multiplicity, or do not run the family.** Comparing eight models pairwise is
twenty-eight tests, and at a 5% level about one false positive is expected by construction.
Either pre-specify the single comparison that decides the release, or use a model confidence set
and report the set that cannot be distinguished. "Model 3 beats model 7, p = 0.04" from a family
of eight is not a finding.

**Step 25. Report the uncertainty on the difference, not only the point estimate.** An
improvement of 0.02 cases on a baseline of 1.75 is a rounding error whose interval straddles
zero. An improvement of 0.40 on the twenty series carrying service risk may have a wide interval
and still justify the change. What goes in front of a decision maker is the interval and the
sentence that goes with it: "the difference is between a 2% and a 6% improvement, and it does not
change sign."

### 5.7 Phase G — Aggregate and report

**Step 26. Choose weights that match the decision, and publish them.** An unweighted mean asserts
every series matters equally: right for service reporting, wrong for model selection where revenue
is concentrated. Volume weights assert every unit matters equally, cost weights every euro of
consequence. None is neutral; each is a claim about which decision the scoreboard serves.

**Step 27. Report the distribution, not only its mean, and say whether the score is before or
after reconciliation.** Report the weighted mean with its weights, the unweighted median and 90th
percentile, the worst five series and their share of the total, and the share above a threshold.
Reconciliation changes errors by construction, so a leaf score computed before it is not
comparable with one computed after; section 6.8 lists the fields that make a scoreboard auditable.

## 6. Worked example

### 6.1 Setting

A distributor of industrial fasteners carries 2,000 item–warehouse series; 61% of item-weeks are
zero. Planning is weekly with a one-week review period and a three-week lead time, so the decision
horizon is four weeks. Money is euro at 2026 price levels and quantities are cases.

Three candidates. **Q1** is quantile regression on nine levels (0.005, 0.025, 0.165, 0.25, 0.5,
0.75, 0.835, 0.975, 0.995), fitted per item family with cross-learning. **Q2** is a hurdle model:
a Bernoulli occurrence component and a compound Poisson size component combined into a discrete
predictive distribution. **REF** is the incumbent's probabilistic layer — a seasonal naive point
forecast plus the empirical distribution of its historical errors.

The evaluation uses 40 rolling origins, a four-week horizon, and all 2,000 series. Stock-out weeks
are excluded from demand but counted and reported, because censored demand understates the truth.

> **Note:** The numbers below are constructed to be internally consistent and to isolate the
> effect of the scoring choice. They are not drawn from a published case, and no claim is made
> that they describe a real distributor.

### 6.2 The score table, and the ranking flip

Scores are computed per series-origin, then aggregated. CRPS is in cases; the scaled version
divides by the series' mean absolute deviation from the naive forecast.

| Metric | Q1 | Q2 | REF | Winner |
|---|---:|---:|---:|---|
| CRPS, mean over series (cases) | 1.75 | **1.62** | 2.40 | Q2 |
| CRPS, median over series (cases) | 0.80 | **0.74** | 1.30 | Q2 |
| CRPS, 5-series trimmed mean (cases) | 1.03 | **0.97** | 1.45 | Q2 |
| CRPS, 90th percentile over series | 4.9 | **4.2** | 6.8 | Q2 |
| Scaled CRPS, mean | 0.41 | **0.38** | 0.57 | Q2 |
| Pinball at q = 0.5 (cases) | **0.39** | 0.40 | 0.62 | Q1 |
| Pinball at q = 0.95 (cases) | **3.90** | 4.40 | 7.10 | Q1 |
| Brier, event "week above 10 cases" | 0.121 | **0.114** | 0.166 | Q2 |
| Log score, mean (nats) | **−1.84** | −1.91 | −2.35 | Q1 |

Read the table twice. Q2 wins on CRPS at every aggregation; Q1 wins on pinball at 0.95 — the level
the safety-stock policy consumes — by 0.50 cases, 11% of Q2's value there. The flip is not a
paradox: CRPS integrates over all levels and is dominated by the body, where most of the mass and
most of the squared error live, while pinball at 0.95 looks only at the tail.

The decision uses the 0.95 quantile, so pinball at 0.95 selects and Q1 is the correct deployment.
An organisation selecting on CRPS deploys Q2, improves its headline number, and worsens the
stock-out rate. That is the argument of this topic in one table.

### 6.3 Why the mean is dominated by a handful of series

Ten series are high-runner fasteners with CRPS between 18 and 48 cases; the other 1,990 sit
between 0.6 and 1.0.

| Component | Count | Sum of CRPS | Share of total | CRPS per series |
|---|---:|---:|---:|---:|
| Ten largest series | 10 | 198.0 | 56.6% | 19.8 |
| Remaining 1,990 | 1,990 | 152.0 | 43.4% | 0.076 |
| Total | 2,000 | 350.0 | 100% | 0.175 |

**5% of the portfolio carries 56.6% of the mean score.** A 5% improvement on the ten large
series moves the mean by 2.8%; a 50% improvement on the 1,990 small ones moves it by 21.7%. Both
are real improvements, and only one concerns the items generating service failures. The median
and the trimmed mean are the honest summaries; the mean flatters whichever model is good at
scale.

### 6.4 Calibration, separately

| Diagnostic | Q1 | Q2 | REF | Reading |
|---|---|---|---|---|
| 80% coverage, 40 origins | 31/40 = 77.5% | 33/40 = 82.5% | 26/40 = 65.0% | Inside the ±12.6-point band for all |
| 90% coverage, 40 origins | 33/40 = 82.5% | 36/40 = 90.0% | 28/40 = 70.0% | Q1 and REF outside the ±9.5-point band |
| Mean 80% width (cases) | 3.1 | 3.4 | 5.2 | Q1 sharper and slightly under-covering |
| PIT histogram shape | Mild U | Flat | Strong U | Q1 underdispersed, REF badly so |
| 0.95 quantile, empirical exceedance | 4.1% | 2.8% | 11.0% | Q1 slightly high, Q2 slightly low, REF far high |

The standard error for the 80% interval at `n = 40` is `sqrt(0.8 × 0.2 / 40) = 0.063`, so a
two-standard-error band is ±12.6 points; for the 90% interval it is ±9.5 points. The 80% row
therefore distinguishes nothing, and the 90% row distinguishes Q1 and REF from nominal but not
Q2. This is the arithmetic a coverage table must carry and almost never does.

Q1's histogram has nine values in the first bin and eight in the last against an expected four,
the U-shape of underdispersion, consistent with its 3.1 width against Q2's 3.4. Note what this
does not mean: it does not mean Q1 is the worse forecast. It means Q1's intervals claim slightly
more than they deliver, and the fix is to widen by roughly 10%, not to switch models.

### 6.5 The paired comparison

Per-series-origin differences `d_t = CRPS_Q1,t − CRPS_Q2,t` over 80,000 units:

| Quantity | Value |
|---|---:|
| Mean difference (cases) | +0.130 |
| Standard error, naive | 0.021 |
| Standard error, HAC with 4-week overlap | 0.050 |
| Diebold–Mariano statistic | 2.60 |
| Two-sided p-value | 0.009 |
| 95% confidence interval | [0.032, 0.228] |
| Q2 better on | 118 of 200 sampled series |
| Q1 better on | 62 of 200 |
| Tied | 20 of 200 |

The naive standard error is less than half the HAC one, because four-week-ahead forecasts from
consecutive origins share three weeks of outcome. This is the most common statistical error in
forecast scoreboards: the significance is real but overstated by a factor of five in the
variance, and the qualitative conclusion survives. The effect size does not — 0.13 cases on a
base of 1.75 is 7.4%, with an interval running from 1.8% to 13.0%.

Now the same comparison at the decision level. The paired difference in pinball at 0.95 is
`−0.50` cases in favour of Q1 with a HAC standard error of 0.14, a statistic of −3.57 and a
p-value below 0.001. **The two tests point in opposite directions, both are correct, and only
one concerns the decision.** That sentence is the deliverable of this topic.

### 6.6 The hierarchy and the reconciliation question

The 2,000 series roll up to 40 item families, 6 warehouses, and one total; MinT reconciliation
is applied at the family–warehouse level. Scores before and after:

| Level | Series | CRPS before | CRPS after | Change |
|---|---:|---:|---:|---:|
| Total | 1 | 41.2 | 38.9 | −5.6% |
| Warehouse | 6 | 12.4 | 12.1 | −2.4% |
| Family × warehouse | 240 | 3.8 | 3.95 | +3.9% |
| Item × warehouse | 2,000 | 1.75 | 1.82 | +4.0% |

Reconciliation improves the total and the warehouse and worsens the leaf by 4%. A weighting
giving equal weight to all four levels reports a net improvement; a weighting proportional to
series count reports a net deterioration; a weighting by decision importance, where the leaf
drives replenishment, reports a clear deterioration. **The reconciliation decision and the
aggregation decision are entangled, and whichever is made second determines the answer.**
Topics 02 and 03 treat the methods; the point here is that a scoreboard cannot be designed
independently of them.

> **Note:** M5 weighted each of its twelve levels equally, with weights summing to one within a
> level. That makes a level with three series count as much as a level with 30,490. It is
> defensible as a measure of hierarchical coherence and indefensible as a measure of
> replenishment quality. Cite the weighting whenever you cite the number.

### 6.7 What each audience hears

| Framing | Number | What the room hears |
|---|---|---|
| "CRPS improved 7%" | 1.75 to 1.62 cases | "A modest statistical gain; not worth a release" |
| "Scaled CRPS 0.38" | 0.38 against 0.57 for REF | "A third better than what we do now" |
| "Tail pinball 4.40 against 3.90" | 0.50 cases at q = 0.95 | "The new model will stock out more" |
| "5% of series carry 57% of the mean" | 10 of 2,000 | "The average is not about our problem" |
| "90% interval covers 82.5% on 40 origins" | ±9.5-point band | "We cannot tell yet" |
| "Difference 0.13 cases, interval 0.03 to 0.23" | HAC interval | "Real but small; what does it buy?" |

The last row is where a scoreboard becomes a decision document. A confidence interval forces the
conversation onto magnitude and relevance, where the money is, and away from significance, where
the argument usually lives.

### 6.8 The ten-field reporting standard

| # | Field | Content | What its absence hides |
|---:|---|---|---|
| 1 | Score and formula | Name, formula, sign and doubling conventions | Two incomparable numbers under one label |
| 2 | Units and scaling | Units of the series, or the scaling denominator | A mean CRPS compared across items of different scale |
| 3 | Evaluation set | Origins, horizons, vintages, series count, exclusions | A candidate scored on easier weeks |
| 4 | Aggregation | Weights, level handling, mean/median/trim, trim fraction | Domination by a few series |
| 5 | Uncertainty | Interval on the difference, test used, overlap handling | Significance invented by an underestimated standard error |
| 6 | Calibration evidence | PIT shape, coverage against nominal with `n`, interval widths | A score that hides a miscalibrated tail |
| 7 | Distribution of scores | Median, 90th percentile, worst five, worst-five share | A mean that conceals the failures |
| 8 | Reference | Named benchmark and its score | A skill score with no denominator |
| 9 | Decision link | Which functional or level drives the decision | A good score on the wrong question |
| 10 | Version and date | Model version, scoring code version, restatement notes | A metric that drifted until nobody could tell |

Four questions decide whether the scoreboard means anything: was the demand censored by
stock-outs, and how many weeks were affected? Is CRPS computed on one grid for all three
candidates, or does Q2's discrete distribution get a different treatment? Was reconciliation
applied before scoring for every candidate, or only for the new one? And are 40 origins enough to
say anything about a 0.95 tail that by construction contains about two exceedances?

## 7. Code, query, or calculation

The calculation below is the whole of CRPS and pinball on a three-outcome forecast. It is short
enough to do on paper, and doing it on paper is the point: the definitions are easy to
misremember and hard to misremember once you have integrated a step function by hand.

```text
CRPS AND PINBALL BY HAND -- three outcomes, one realised value

support x    = 10, 20, 30          truthful F   = 0.20, 0.50, 0.30
                                  sharpened G  = 0.05, 0.90, 0.05
realised y   = 20                 (all CDFs reach 1.0 above 30)

CRPS = sum over gaps of (F(x) - 1{x >= y})^2 * gap
  F: [10,20) (0.20 - 0)^2 * 10 = 0.40   [20,30) (0.70 - 1)^2 * 10 = 0.90  -> 1.300
  G: [10,20) (0.05 - 0)^2 * 10 = 0.025  [20,30) (0.95 - 1)^2 * 10 = 0.025 -> 0.050
  the two outer intervals contribute zero: F = 0 below 10 and F = 1 above 30

pinball at q = 0.5: medians are 20 and 20, so e = 0 and the loss is 0.000 for both
pinball at q = 0.8: q_F = 30 because F(20) = 0.70 < 0.8; q_G = 20 because G(20) = 0.95
  F: e = 20 - 30 = -10, loss = (0.8 - 1) * (-10) = +2.000
  G: e = 0,             loss = 0                   =  0.000

log score, higher is better: ln(0.50) = -0.693 ; ln(0.90) = -0.105
Brier, 3 classes:            F = .04 + .25 + .09 = 0.380 ; G = .0025 + .010 + .0025 = 0.015

read: on this ONE outcome the sharpened forecast wins every proper score.
      propriety is an expected-value property. Score many origins, then average.
```

Three checks before trusting any implementation. Confirm the indicator is `1{x ≥ y}` and not
`1{x > y}`: for continuous forecasts the difference is measure zero, but for discrete ones it
changes every gap. Confirm the doubling convention, because a grid approximation that omits the
factor of two reports exactly half the correct CRPS. Confirm the grid covers the tails, since
nine equally spaced levels stop at 0.005 and 0.995 and a distribution with substantial mass
beyond them is being scored only on its middle.

## 8. Data, standards, and regulatory context

No international standard specifies a scoring rule for forecast distributions. That absence is
informative: the choice is methodological and commercial, which is why it must be documented
rather than defaulted. What exists is a set of adjacent standards constraining vocabulary,
units, and reporting, and a few regimes constraining the process around the score.

### 8.1 Measurement and vocabulary

| Standard | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of probability, distribution, estimator | Any scoring rule | B |
| ISO 5725-1 | Trueness and precision of measurement | Accuracy as bias plus precision | Forecast evaluation, which has no repeated reference | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | "Accuracy" is qualitative, not a number | Anything specific to forecasts | B |
| ISO 80000-1 | Quantities and units | Unit conventions for money, time, ratios | Which currency or price year | B |
| ISO 8601 | Date and time representation | Week numbering and ordering | Which week an origin or outcome belongs to | B |

The VIM point transfers with force. "Forecast accuracy of 91%" is a conformance rate against an
implicit tolerance, exactly as "inventory accuracy of 98%" was in Module 01A topic 08: a
statistic about a definition and a sample, not a property of the forecast. CRPS is better
behaved because it is proper and has units, but "CRPS of 1.75" is still a sample statistic whose
meaning depends on population, horizon, and exclusions.

### 8.2 Operational meteorological practice, where CRPS is standard

CRPS entered supply chain forecasting from weather forecasting, and the operational conventions
are worth importing. In numerical weather prediction it has been used since the early 2000s
alongside rank histograms and reliability diagrams, with the canonical decomposition into
reliability, resolution, and uncertainty. The published application to the European Centre for
Medium-Range Weather Forecasts ensemble for 24-hour precipitation reports CRPS growing from about
0.98 mm at day 2 to about 1.31 mm at day 10, reliability small at both ranges and resolution
falling sharply with lead time. That pattern — reliability stable, resolution degrading — is what
a supply chain forecaster should look for when accuracy falls with horizon. (Grade A for method
and figures; the application is meteorological, not commercial.)

### 8.3 Model risk and internal control

Where forecast distributions feed inventory valuation, revenue guidance, or capital allocation
in a listed company, the *process* around the model falls inside internal control over financial
reporting under Section 404 of the Sarbanes-Oxley Act of 2002. The control governs who may change
a model, how the change is validated and documented, and whether the evaluation is reproducible.
It does not specify CRPS: a company can be fully compliant and still score its models with an
improper rule. Banking guidance such as the Federal Reserve's SR 11-7 is a useful template by
analogy — model inventory, validation, ongoing monitoring, documented assumptions — and does not
govern demand planning models.

### 8.4 Where a regulatory metric might bind

A forecasting system classified as high-risk under the EU Artificial Intelligence Act
(Regulation (EU) 2024/1689) would face Article 15: declared accuracy, robustness, and
cybersecurity metrics with testing against them. Most demand planning systems will not fall into a
high-risk category, so treat this as a design template rather than an obligation, and consult the
consolidated text. The template is useful: declare the metric, the test set, and the threshold,
and keep the evidence.

### 8.5 What is convention rather than requirement

- Nine quantile levels matching the 50%, 67%, 80%, 90%, 95%, and 99% intervals is a competition
  convention from M5 and meteorology, not an optimum; levels should follow the decision.
- The 5% trim in a trimmed mean, and the ten-bin PIT histogram, are conventions. State them.
- Equal weighting across hierarchy levels is a convention with large consequences, not a neutral
  default.
- "CRPS in cases" is local to a series; across a portfolio you need a scaled or weighted version,
  and which one is a decision.

An analyst who can distinguish a requirement from a convention can challenge a convention. In
scoring almost everything is a convention, which is why almost everything is arguable and why the
argument must be documented rather than won.

## 9. Industry practice and case evidence

### 9.1 What the competitions established

The M5 Accuracy competition is the largest public test of hierarchical probabilistic retail
forecasting, and its evaluation design is its most instructive artefact. It scored 42,840 series
across twelve aggregation levels, 30,490 of them at the bottom. Point accuracy used a weighted
root mean squared scaled error; the probabilistic track used a weighted scaled pinball loss
averaged over nine quantile levels corresponding to the median and the 50%, 67%, 80%, 90%, 95%,
and 99% intervals. Weights were cumulative dollar sales over the last 28 training days,
normalised to one within each level, with all twelve levels weighted equally.

Two findings matter here. Only 415 of 5,507 teams — about 7.5% — beat the bottom-up exponential
smoothing benchmark at all, so the distribution of skill is extremely skewed; that is a fact
about forecasting, not about the competition. And the winner was not best at every level,
ranking eleventh at the bottom level while winning overall. A single aggregate score, whatever
its weighting, conceals level-specific performance. (Grade A for the competition and reported
results; the interpretation is the author's.)

### 9.2 What the evaluation-set critique found

A subsequent analysis argued that scale-dependent measures let high-volume series dominate the
aggregate, effectively reducing a 30,000-series problem to a single-series problem, and
introduced a rank-stability measure: how well the model ranking survives resampling. It reported
rank stability as low as roughly 0.29 to 0.41 for the price-weighted squared-error metric,
against roughly 0.83 to 0.88 for scale-free measures such as weighted absolute percentage error
and symmetric mean absolute percentage error, with stability degrading as more hierarchical
levels were aggregated. It also reported that giving the top level a weight on the order of 5%
retained most of the stability.

Treat those figures as reported rather than replicated: the paper is a preprint and this course
has not re-run the experiment. The implication is robust either way. **A leaderboard whose
ranking depends on how the levels were weighted is measuring the weighting as much as the
models.** (Grade B.)

### 9.3 Probabilistic scoring on intermittent demand

The most directly relevant study evaluates nine probabilistic methods and seven combination
schemes on intermittent retail demand with a 28-day horizon and roughly 60% zero observations,
scoring on calibration, proper scoring rules, and inventory outcomes. Three findings transfer.
Simple averaging of probabilistic forecasts was the most robust combination scheme. Rankings under
the log score and the Brier score were inversely correlated with rankings under the discrete
ranked probability score, so the choice of proper score changed which method won. And the
best-scoring method was not the method with the lowest inventory cost — even a proper score,
correctly computed, is not the same object as the decision. (Grade A; not independently
replicated.)

Why the discrete ranked probability score rather than CRPS for count data: CRPS integrates a step
function and behaves awkwardly when the support is non-integer, while the ranked probability
score is defined on ordered categories. That is one reason the intermittent-demand literature has
not converged on CRPS the way meteorology has.

### 9.4 The hierarchical evaluation disagreement

Two papers in the same issue of the *International Journal of Forecasting* disagree about
evaluating hierarchical forecasts. One argues that averaging accuracy across levels has limited
benefit because each level serves a different decision, and proposes multi-objective evaluation:
Pareto frontiers across levels and metrics, scalarised only with importance weights from the
decision, since no general scalarisation exists. The other responds that for coherent point
forecasts the sensible measure is (root) mean squared error or a monotonic function of a weighted
sum of squared errors, and that MAE and MASE are problematic here. The disagreement is not about
arithmetic but about whether a hierarchy has a single objective. Anyone who says the question is
settled has not read both papers. (Grade A for both.)

### 9.5 What practitioners do, and the gap

Observed practice, as distinct from recommendation, stated as inference from working experience
rather than as a measured survey:

| Recommendation | Observed practice | Why the gap persists |
|---|---|---|
| Score with a proper rule matched to the decision | A single percentage accuracy in the template | The template predates the probabilistic layer |
| Report uncertainty on the difference | Point estimates compared across models | The statistical machinery sits outside the planning team |
| Report the per-series distribution | Mean across the portfolio | Means fit on a slide |
| Separate calibration from score | One chart, labelled "validation" | Different owners, different tools |
| Publish the weighting | Weights implicit in the system default | Nobody chose them, so nobody defends them |

> **Note:** The table is inference, not survey evidence. The course has no measured benchmark for
> how many organisations report CRPS, and any specific percentage quoted to you should be
> treated as vendor marketing until the sampling frame is shown.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Improper score selects an overconfident model | The winning model's 90% interval covers 60% of outcomes | The score rewards concentration regardless of truth | Tail failures in exactly the items that matter | Check propriety first; report coverage beside the score |
| Scoring the median and calling it the distribution | "CRPS" equals the MAE of a point forecast to three decimals | A degenerate forecast's CRPS is MAE by construction | No information about spread reaches selection | Score the distribution, or state that only the median is scored |
| Mean over a skewed portfolio | Ten series carry more than half the total score | Scale and skew dominate the arithmetic mean | Improvement driven by irrelevant series | Report median, trimmed mean, worst-five share |
| Score changed after the leaderboard | A footnote reads "metric revised in Q3" | Selection on the evaluation set | Inflated, unreproducible gains | Pre-register the score and its version |
| No uncertainty on the difference | "CRPS improved by 0.02" with no interval | A point estimate of a noisy quantity | Noise reported as progress | Paired differences, HAC standard error, interval |
| Naive standard error on overlapping horizons | A p-value five times smaller than the HAC one | Consecutive origins share outcomes | Significance manufactured by dependence | Long-run variance with bandwidth at least the horizon |
| Twenty-eight pairwise tests | "Model 3 beats model 7, p = 0.04" | Multiplicity | One false positive expected by construction | Pre-specify one comparison, or use a model confidence set |
| Log score with hard zeros | A mean log score of negative infinity, or a case silently excluded | A zero-probability outcome was realised | The comparison collapses to an artefact | Guarantee support; report boundary cases separately |
| Tail scored on too few origins | A 95% coverage claim from 40 points | About two exceedances expected in the sample | The claim is unfalsifiable | Report `n` and the binomial band; pool across series if defensible |
| Scoring before reconciliation, deploying after | Leaf scores from a different pipeline version | Reconciliation changes errors by construction | The scoreboard describes a model nobody runs | Score the version that ships; version both |

Two rows deserve expansion. **Improper score selects an overconfident model** looks most like
success: the score improves, the intervals narrow, the deck gets greener, and stock-outs rise in
the tail where the policy lives; the red flag is not in the score but in the coverage chart
nobody put on the same page. **Tail scored on too few origins** is the honest failure: a 0.95
coverage rate computed from 40 origins is numerically correct and inferentially empty, since the
standard error is `sqrt(0.95 × 0.05 / 40) = 0.034` and one exceedance moves the estimate 2.5 points.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The evaluation sample resembles the deployment distribution | A score estimates future loss | Regime change, promotions, new listings, a pandemic | Compare scores by sub-period; plot the rolling score |
| Origins are independent, or dependence is modelled | Standard errors and tests assume it | Overlapping horizons inflate significance | Autocorrelation of the loss differential; HAC estimate |
| The truth is observable and uncensored | Every score compares forecast with outcome | Stock-outs censor sales, understating demand | Flag stock-out weeks; estimate censored demand |
| The score is proper for the object compared | Otherwise the optimum is a lie | Selection drifts toward overconfidence | Check the expectation; construct a counterexample |
| Scores are comparable across series after scaling | Portfolios mix scales by orders of magnitude | The mean is dominated by the largest series | Compare mean, median, and trimmed mean |
| Evaluation and decision use the same functional | Selecting on CRPS while deciding on q0.95 | The better model is the wrong model for the decision | State the functional; score at that level |
| The aggregation weights reflect decision value | The pooled number is what the business acts on | A weighting that flatters reconciliation or scale | Recompute under a second weighting; check the sign |
| No information leaks from the evaluation window | Selection must be out of sample | Hyperparameters or model choice tuned on the scored origins | Pre-register; hold out a final window |
| The support covers plausible outcomes | The log score is unbounded below | Negative infinity from one zero-probability case | Count boundary cases; check the support |
| The code implements the stated formula | The number is only as good as the implementation | A missing factor of two, a sign flip, an inclusive indicator | Reproduce a hand calculation as in section 7 |

Three limits are worth stating plainly.

**A score is not a decision.** Every score reduces a distribution to one number using weights it
chooses. The decision uses different weights — the critical fractile, the holding cost, the
service promise. A model that wins on CRPS can lose on cost, and section 9.3 reports evidence
that it does. Use the score to select among models whose decisions are similar; use a cost model
to select among models whose decisions differ.

**Propriety protects against gaming the report, not against a bad model.** It guarantees the
truth is optimal in expectation and nothing about whether your model class contains the truth,
whether estimation is stable, or whether the sample is representative. A proper score on a bad
sample is a precise answer to the wrong question.

**The hierarchy has no canonical weighting.** The disagreement in section 9.4 is genuine. If you
report one number for a hierarchy you have made a decision no statistical theory will make for
you, and the honest response is to publish the weighting and a sensitivity to it.

## 12. Exercises

### Exercise 1 — Propriety by hand (calculation, 45 minutes)

Using the three-outcome example from section 7, compute expected CRPS, multicategory Brier score,
and log score under the truthful distribution for `F`, `G`, and `H`, then repeat under `G`.

**Deliverable:** a 3 × 3 table with each expectation shown term by term.
**Acceptance criteria:** all nine values correct; the row minimum identified under each truth; one
sentence on what changes when the truth is `G`.
**Time:** 45 minutes.

### Exercise 2 — CRPS as integrated pinball (derivation, 35 minutes)

Show numerically that CRPS equals twice the average pinball loss over `q = 0.1, 0.3, 0.5, 0.7,
0.9` for the truthful `F` with `y = 20`, and state why the five-level average misses the integral.

**Deliverable:** quantiles, pinball losses, the doubled average, and the exact CRPS.
**Acceptance criteria:** the quantile convention stated; the approximation error computed; one
sentence on how the grid should be chosen.
**Time:** 35 minutes.

### Exercise 3 — Design an improper score and break it (design, 40 minutes)

Propose three plausible-looking scores a planning team might adopt — mean interval width, maximum
class probability, negative entropy. For each, construct a forecast that maximises it while being
obviously wrong, and state the effect on a selection loop over 40 origins.

**Deliverable:** three counterexamples, each with its maximising forecast and the consequence.
**Acceptance criteria:** each counterexample is a valid distribution; the mechanism stated in one
sentence; at least one is a forecast a real system could produce.
**Time:** 40 minutes.

### Exercise 4 — Read three PIT histograms (analysis, 30 minutes)

Counts across ten bins, 200 series-origins each. A: 9, 8, 22, 24, 21, 23, 22, 24, 23, 24. B: 22,
23, 21, 20, 19, 20, 18, 19, 19, 19. C: 2, 3, 6, 12, 20, 28, 32, 33, 32, 32. Diagnose each, name
the likely cause, and say which is most dangerous for replenishment.

**Deliverable:** a diagnosis table with shape, cause, and consequence.
**Acceptance criteria:** shape named for each; expected count per bin stated; the recommendation
distinguishes "widen the interval" from "change the model".
**Time:** 30 minutes.

### Exercise 5 — Coverage arithmetic (calculation, 30 minutes)

For nominal coverage 0.80, 0.90, and 0.95, compute the standard error of empirical coverage at
`n = 20`, `n = 40`, and `n = 400`, and the smallest deviation distinguishable at two standard
errors.

**Deliverable:** two 3 × 3 tables — standard errors and detectable deviations.
**Acceptance criteria:** formula shown; nine values correct to three decimals; one sentence on the
0.95 tail at 40 origins.
**Time:** 30 minutes.

### Exercise 6 — Build the scoreboard (design, 50 minutes)

Complete the ten-field reporting standard from section 6.8 for any forecast evaluation you know,
marking unknown fields as unknown rather than inventing them.

**Deliverable:** a completed ten-field table plus a paragraph on which missing field most changes
the conclusion.
**Acceptance criteria:** at least seven fields completed; the unknowns justified by their effect;
one field named as this quarter's priority.
**Time:** 50 minutes.

### Exercise 7 — The paired comparison (calculation, 45 minutes)

Sixty evaluation units, two models. Per-unit score differences have mean 0.18, standard deviation
0.95, and lag-1 autocorrelation 0.55. Compute the naive standard error, the first-order
autocorrelation correction, the test statistic, and a 95% interval under each.

**Deliverable:** both standard errors, both intervals, and a decision sentence.
**Acceptance criteria:** the correction formula stated; arithmetic reproducible; the conclusion
phrased in terms of the interval, not the p-value.
**Time:** 45 minutes.

### Exercise 8 — Aggregation audit (analysis, 40 minutes)

Construct 100 series in which 4 carry 55% of total CRPS. Compute the mean, median, 4-series
trimmed mean, and 10% trimmed mean; then improve the four large series by 10% and, separately,
the 96 small ones by 10%, reporting the change in each summary.

**Deliverable:** the four summaries under the baseline and the two improvements.
**Acceptance criteria:** total and shares shown; the summary responding most to each improvement
identified; one sentence on which summary selection should use.
**Time:** 40 minutes.

### Exercise 9 — Hierarchy weighting (design and calculation, 55 minutes)

Using the four levels in section 6.6, compute the overall change from reconciliation under three
weightings — equal per level, proportional to series count, and leaf-only — and defend one for
replenishment and one for a coherence audit.

**Deliverable:** three weighted changes plus two defences.
**Acceptance criteria:** arithmetic shown for each; each defence names the decision it serves; one
sentence on why no weighting is neutral.
**Time:** 55 minutes.

## 13. Self-check answers

### 13.1 Exercise 1 arithmetic

Under truth `F = (0.2, 0.5, 0.3)`, with support 10, 20, 30.

```text
CRPS, outcome by outcome:
  10: [10,20) (0.20-1)^2*10 = 6.40 ; [20,30) (0.70-1)^2*10 = 0.90  ->  7.30
  20: [10,20) (0.20-0)^2*10 = 0.40 ; [20,30) (0.70-1)^2*10 = 0.90  ->  1.30
  30: [10,20) (0.20-0)^2*10 = 0.40 ; [20,30) (0.70-0)^2*10 = 4.90  ->  5.30
E[CRPS_F] = 0.2*7.30 + 0.5*1.30 + 0.3*5.30 = 1.46 + 0.65 + 1.59 = 3.700

for G = (0.05, 0.90, 0.05):
  10: [10,20) (0.05-1)^2*10 = 9.025 ; [20,30) (0.95-1)^2*10 = 0.025 -> 9.050
  20: [10,20) (0.05-0)^2*10 = 0.025 ; [20,30) (0.95-1)^2*10 = 0.025 -> 0.050
  30: [10,20) (0.05-0)^2*10 = 0.025 ; [20,30) (0.95-0)^2*10 = 9.025 -> 9.050
E[CRPS_G] = 0.2*9.05 + 0.5*0.05 + 0.3*9.05 = 1.81 + 0.025 + 2.715 = 4.550
for H (point mass at 20): E|20 - y| = 0.2*10 + 0.5*0 + 0.3*10 = 5.000
```

```text
Brier: 0.2*(0.64+0.25+0.09) + 0.5*(0.04+0.25+0.09) + 0.3*(0.04+0.25+0.49) = 0.620   [F]
       0.2*1.715 + 0.5*0.015 + 0.3*1.715                                 = 0.865   [G]
       0.2*2 + 0.5*0 + 0.3*2                                            = 1.000   [H]
log:   0.2 ln 0.2 + 0.5 ln 0.5 + 0.3 ln 0.3                              = -1.0297 [F]
       0.2 ln 0.05 + 0.5 ln 0.9 + 0.3 ln 0.05                            = -1.5505 [G]
       outcomes 10 and 30 have probability zero under H                 = -inf    [H]
```

Now repeat under truth `G`, where strict propriety requires `G` to win:

```text
CRPS: G under G = 0.05*9.050 + 0.90*0.050 + 0.05*9.050 = 0.950
      F under G = 0.05*7.300 + 0.90*1.300 + 0.05*5.300 = 1.800
      H under G = 0.05*10    + 0.90*0     + 0.05*10    = 1.000
```

`0.950 < 1.000 < 1.800`: the truth wins, as it must. The common error is scoring each forecast
against outcomes drawn from itself; the expectation must be taken under one `G` for all.

### 13.2 Exercise 2 arithmetic

Quantiles of `F` under the standard definition `inf{x : F(x) ≥ q}`: `q0.1 = 10`, `q0.3 = 20`,
`q0.5 = 20`, `q0.7 = 20` (since `F(20) = 0.70 ≥ 0.7`), `q0.9 = 30`. With `y = 20`:

```text
q=0.1, x=10: e = +10 -> 0.1*10         = 1.0
q=0.3, x=20: e = 0                    -> 0.0
q=0.5, x=20: e = 0                    -> 0.0
q=0.7, x=20: e = 0                    -> 0.0
q=0.9, x=30: e = -10 -> (0.9-1)*(-10) = 1.0
mean = 0.40 ; doubled = 0.800
```

The exact CRPS is 1.300, so the grid **understates** it by 0.500, an error of 38%. The direction
is not predictable in general: the quantile function is flat at 10 over `q ∈ (0, 0.2)` and at 30
over `q ∈ (0.7, 1)`, and five equally spaced levels put one point in each region. Because
`F(20) = 0.70` exactly, the 0.7 quantile is also ambiguous between 20 and 30; choosing 30 gives
3.0 there and a doubled mean of 1.4. State the convention or the number is not reproducible.

### 13.3 Exercise 3 rubric

Strong: three improper scores with maximisers — width broken by a zero-width interval, maximum
class probability and negative entropy both broken by a point mass — plus the observation that all
three are monotone in concentration, so the loop silently selects the most overconfident model and
the score improves every time. Weak: proper scores, or no constructed maximiser.

### 13.4 Exercise 4 model answer

A is a mild left-skewed U: slight underdispersion with a low bias. B is the classic U of
underdispersion, intervals too narrow. C rises monotonically from 2 to 32, the triangular shape of
a distribution shifted low.

B is most dangerous despite C's visible bias: B passes inspection at the centre and fails in the
tail, where safety stock is set. C's bias is correctable by a location shift; B needs its spread
re-estimated.

### 13.5 Exercise 5 arithmetic

Standard error `sqrt(p(1−p)/n)`:

| Nominal | `n = 20` | `n = 40` | `n = 400` |
|---|---:|---:|---:|
| 0.80 | 0.089 | 0.063 | 0.020 |
| 0.90 | 0.067 | 0.047 | 0.015 |
| 0.95 | 0.049 | 0.034 | 0.011 |

At `n = 40` and 0.95 the detectable deviation is `2 × 0.034 = 0.068`, so only coverage below
88.2% is distinguishable from noise. Knowing whether a 0.95 interval is within two points of
nominal takes roughly 500 independent origins, or a defensible pooling across series.

### 13.6 Exercise 6 rubric

Strong: seven or more fields completed, the rest marked unknown, with the aggregation or
uncertainty field named as priority because an unstated weighting can flip a ranking while a
missing interval only overstates confidence. Weak: every field filled with a plausible guess,
which is worse than leaving blanks.

### 13.7 Exercise 7 arithmetic

```text
n = 60, mean d = 0.18, sd = 0.95, lag-1 rho = 0.55
naive SE     = 0.95 / sqrt(60) = 0.1227
inflator     = (1 + rho) / (1 - rho) = 1.55 / 0.45 = 3.444
corrected SE = 0.1227 * sqrt(3.444) = 0.1227 * 1.856 = 0.2277
naive t      = 0.18 / 0.1227 = 1.47   -> 95% CI [-0.061, 0.421]
corrected t  = 0.18 / 0.2277 = 0.79   -> 95% CI [-0.278, 0.638]
```

The conclusion changes: marginally reportable under the naive standard error, clearly
indistinguishable from zero under the corrected one. The AR(1) inflator assumes first-order
dependence and a long series; the Diebold–Mariano test estimates the long-run variance directly,
preferable when the horizon overlap exceeds one period.

### 13.8 Exercise 8 model answer

Construct 96 series at 0.75 each (total 72) and 4 series at 22 each (total 88); the portfolio
total is 160 and the large-series share is 55%.

| Summary | Baseline | Large improved 10% | Small improved 10% |
|---|---:|---:|---:|
| Mean | 1.600 | 1.500 | 1.528 |
| Median | 0.750 | 0.750 | 0.675 |
| 4-series trimmed mean | 0.750 | 0.750 | 0.675 |
| 10% trimmed mean | 0.750 | 0.750 | 0.675 |

The mean responds to both, and reports the large-series improvement as the larger (6.25% against
4.5%) even though the small-series one is far more widely distributed. The median and trimmed
means are blind to the large-series improvement because those four series are trimmed away.
**Neither summary is right for every question:** the mean is right when a unit of CRPS costs the
same money everywhere, the median when the question is how the typical item is doing. Where a
stock-out on any item costs roughly the same, the unweighted median is the better default.

### 13.9 Exercise 9 arithmetic

Using the section 6.6 changes: total −5.6%, warehouse −2.4%, family–warehouse +3.9%,
item–warehouse +4.0%.

```text
equal per level   : (-5.6 - 2.4 + 3.9 + 4.0) / 4                  = -0.03%  ~ no change
proportional to n : (-5.6 - 14.4 + 936 + 8000) / 2247             = +3.96%
leaf only         : +4.0%
```

Three defensible weightings produce "no change", "+4%", and "+4%". Equal-per-level suits a
coherence audit, because coherence is a property of the hierarchy and each level tests it. Leaf
weighting suits replenishment, because the leaf generates the order. Proportional weighting is
the one to avoid: nobody chose it, and it equates "more series" with "more important". A
scoreboard reporting only the equal-per-level number approves a reconciliation that worsens the
item-level forecast.

## 14. Interview questions

**1. "What makes a scoring rule proper, and why does it matter?"**

Strong: the truth maximises expected score; without propriety the score's gradient points at a
lie. Distinguish proper from strictly proper. Weak: "it is a good metric", or conflating
propriety with accuracy.

**2. "Why can't you use MAE to evaluate a probabilistic forecast?"**

Strong: MAE is consistent for the median and blind to everything else, so distributions sharing a
median tie however different their tails; MAE is CRPS for a degenerate forecast. Weak: "MAE
ignores uncertainty", without saying which part.

**3. "Define CRPS and give its units."**

Strong: the integral of the squared difference between the forecast CDF and the indicator at the
realisation, in units of the forecast variable; twice the integrated pinball loss; MAE for a point
forecast. Weak: "a proper score for distributions", with no formula.

**4. "A model has better CRPS but worse pinball at the 0.95 quantile. Which do you deploy?"**

Strong: ask which functional the decision uses; CRPS is dominated by the body, so if the policy
consumes 0.95 then pinball at 0.95 selects and the CRPS advantage is secondary. Weak: pick CRPS
because it is more standard.

**5. "What does a U-shaped PIT histogram tell you?"**

Strong: underdispersion, intervals too narrow, too many outcomes in both tails; widen, and note
that the histogram says nothing about whether the centre is right. Weak: "the model is bad".

**6. "Can a forecast be well calibrated and useless?"**

Strong: yes — climatology, calibrated by construction, zero resolution, skill score zero; the
useful object is the sharpest forecast subject to calibration. Weak: no, calibration implies
quality.

**7. "How would you decide whether a 0.13-case CRPS improvement is real?"**

Strong: paired differences per evaluation unit, a long-run variance accounting for horizon
overlap, and a confidence interval reported instead of a p-value. Weak: compare two means and
declare a winner.

**8. "Why is the mean score across a portfolio misleading?"**

Strong: scale and skew let a few series dominate — 5% of series can carry half the total; report
the median, a trimmed mean, the worst-five share, and a scaled score. Weak: "outliers", naming the
symptom.

**9. "When would you use the log score rather than CRPS?"**

Strong: density forecasts with guaranteed support, events, and decisions sensitive to the density
at the outcome; the penalty is unbounded, so support must be engineered. Weak: "it is more
punishing".

**10. "What is the difference between the energy score and the variogram score?"**

Strong: both proper for multivariate forecasts; the energy score is the multivariate CRPS and
weak on correlation, the variogram score detects dependence but ignores common bias. Weak: treat
them as interchangeable.

**11. "How should a hierarchy be weighted when scoring?"**

Strong: no neutral weighting — equal-per-level makes a three-series level count as much as a
30,000-series level; weight by decision and publish a sensitivity. Weak: "weight by revenue".

**12. "Your scoreboard improved and service got worse. What happened?"**

Strong: the score measured a functional the decision does not use, the aggregation was dominated
by irrelevant series, or the scored version was pre-reconciliation. Weak: blame the data.

## 15. Key terms

- **Scoring rule** — a function of forecast distribution and outcome returning a number.
- **Proper** — the true distribution optimises the expected score.
- **Strictly proper** — the true distribution is the unique optimum.
- **Consistency** — a score optimised by the true value of a stated functional.
- **Brier score** — squared probability error for an event; multi-category form sums over classes.
- **Pinball loss** — asymmetric piecewise-linear loss consistent for a quantile.
- **Quantile level** — the probability at which a quantile forecast is issued.
- **CRPS** — continuous ranked probability score; integrated squared CDF error, in units of the
  variable.
- **CRPSS** — CRPS skill score against a named reference.
- **Log score** — log density or log probability at the outcome; strictly proper, unbounded.
- **Energy score** — multivariate CRPS; proper, weak on correlation.
- **Variogram score** — pairwise-difference score; proper, not strict, sensitive to dependence.
- **Ranked probability score** — cumulative-probability score for ordered categories.
- **Calibration** — agreement between stated probabilities and observed frequencies.
- **Sharpness** — concentration of the predictive distribution.
- **Resolution** — how much the forecast discriminates between outcomes.
- **Reliability** — the score component attributable to miscalibration.
- **PIT** — probability integral transform; uniform when calibrated.
- **Randomized PIT** — the discrete-data version that restores uniformity.
- **PIT histogram** — binned PIT counts; shape diagnoses dispersion and bias.
- **Reliability diagram** — nominal against empirical probability or coverage.
- **Empirical and nominal coverage** — observed share inside an interval, and the share claimed.
- **Interval score** — width plus twice the miss distance, at a stated level.
- **WIS** — weighted interval score; a discretised relative of CRPS.
- **Skill score** — score relative to a reference forecast.
- **Loss differential** — paired difference of scores at one evaluation unit.
- **Diebold–Mariano test** — test of equal expected loss with a long-run variance estimate.
- **Model confidence set** — models not distinguishable from the best at a stated confidence.
- **Trimmed mean** — mean after discarding a stated share of the largest values.
- **Aggregation weight** — the weight a series, level, or horizon receives when scores are pooled.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Gneiting, T. and Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *JASA*, 102(477), 359–378. DOI: 10.1198/016214506000001437. Propriety, energy score, CRPS.
- Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007). "Probabilistic Forecasts, Calibration and Sharpness." *JRSS-B*, 69(2), 243–268. The calibration–sharpness paradigm.
- Hersbach, H. (2000). "Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems." *Weather and Forecasting*, 15(5), 559–570. DOI: 10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2. The decomposition and the ECMWF figures in section 8.2.
- Candille, G. and Talagrand, O. (2005). "Evaluation of Probabilistic Prediction Systems for a Scalar Variable." *QJRMS*. DOI: 10.1256/qj.04.71.
- Scheuerer, M. and Hamill, T. M. (2015). "Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities." *Monthly Weather Review*, 143(4), 1321–1334. DOI: 10.1175/MWR-D-14-00269.1.
- Diebold, F. X. and Mariano, R. S. (1995). "Comparing Predictive Accuracy." *JBES*, 13(3), 253–263. DOI: 10.1080/07350015.1995.10524599.
- Harvey, D., Leybourne, S., and Newbold, P. (1997). "A Correction for the Diebold–Mariano Test." *International Journal of Forecasting*. Volume and pages not verified for this file.
- Gneiting, T., Wolffram, D., Resin, J., et al. (2023). "Model Diagnostics and Forecast Evaluation for Quantiles." *Annual Review of Statistics and Its Application*, 10, 597–621. DOI: 10.1146/annurev-statistics-032921-020240.
- Gneiting, T. and Resin, J. (2023). "Regression Diagnostics Meets Forecast Evaluation." *Electronic Journal of Statistics*, 17, 3226–3286. DOI: 10.1214/23-ejs2180. Randomized PIT.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). "The M5 Accuracy Competition: Results, Findings, and Conclusions." *International Journal of Forecasting*, 38(4).
- Wang, X., Kang, Y., and Petropoulos, F. (2024). "Combining Probabilistic Forecasts of Intermittent Demand." *European Journal of Operational Research*. DOI: 10.1016/j.ejor.2024.01.032.
- Athanasopoulos, G. and Kourentzes, N. (2023). "On the Evaluation of Hierarchical Forecasts." *International Journal of Forecasting*, 39(4), 1502–1511. DOI: 10.1016/j.ijforecast.2022.08.003.
- Kolassa, S. (2023). "Do We Want Coherent Hierarchical Forecasts, or Minimal MAPEs or MAEs? (We Won't Get Both!)." *International Journal of Forecasting*, 39(4), 1512–1517.
- Bjerregård, M. B., Møller, J. K., and Madsen, H. (2021). "An Introduction to Multivariate Probabilistic Forecast Evaluation." *Energy and AI*. DOI: 10.1016/j.egyai.2021.100058.
- Jordan, A., Krüger, F., and Lerch, S. (2019). "Evaluating Probabilistic Forecasts with scoringRules." *Journal of Statistical Software*, 90(12). DOI: 10.18637/jss.v090.i12.

**Preprints and working papers — Grade B**

- Hewamalage, H., Ackermann, K., and Bergmeir, C. "Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices." arXiv:2203.10716.
- Hewamalage, H., Montero-Manso, P., Bergmeir, C., and Hyndman, R. J. "A Look at the Evaluation Setup of the M5 Forecasting Competition." arXiv:2108.03588. Rank-stability figures in section 9.2 are as reported, not replicated.
- Athanasopoulos, G. and Kourentzes, N. (2020, revised 2021). "On the Evaluation of Hierarchical Forecasts." Monash Business School working papers 02/20 and 10/21.

**Standards, official, and regulatory — Grade B**

- ISO 3534-1, *Statistics — Vocabulary and Symbols — Part 1*. Current edition.
- ISO 5725-1, *Accuracy (Trueness and Precision) of Measurement Methods and Results — Part 1*.
- ISO 80000-1, *Quantities and Units — Part 1: General*.
- ISO 8601, *Date and Time — Representations for Information Interchange*.
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM).
- Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 15. Consult the consolidated text before relying on any requirement.
- Sarbanes-Oxley Act of 2002, Section 404, internal control over financial reporting.
- Federal Reserve Board, SR 11-7, *Guidance on Model Risk Management*. Cited by analogy only.

**Software documentation and competition material — Grade B or C**

- `scoringrules` Python and R packages (`properscoring`, `scoringRules`): CRPS, pinball, energy score, variogram score. Grade B for implementation, not for method choice.
- `scores` Python documentation, "The Probability Integral Transform (PIT)" tutorial; `statsmodels.tsa.stattools.diebold_mariano_test`, including the Harvey–Leybourne–Newbold adjustment. Grade B.
- M5 Competitors' Guide and the M5-methods repository (M Competitions). Official documentation. Grade B.

**Practitioner and vendor — Grade C or D**

- This file makes no vendor claims; section 9.5 is explicitly labelled inference. Any planning
  suite documentation quoted as evidence that "the system reports CRPS" is Grade D until the
  formula, the grid, and the weighting are shown.

**Note on citations.** Bibliographic details were checked against the sources returned by the
searches informing this file. Volume and page numbers are given where confirmed and flagged where
not. Numbers in sections 6 and 9 are constructed or reported as their sources state them; none
was independently replicated. The worked example is illustrative and internally consistent, not
drawn from a published case.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic, distributions and their purpose: [`07_probabilistic-forecasts.md`](07_probabilistic-forecasts.md)
- Topic 08, conformal prediction and its coverage guarantees: [`08_conformal-prediction.md`](08_conformal-prediction.md)
- Next topic, decision use of quantiles, is planned and not yet written: `10_quantiles-decisions.md` (planned)
- Intermittent demand and the compound Poisson route to a distribution: [`06_compound-poisson-hurdle.md`](06_compound-poisson-hurdle.md)
- Classification of intermittent series: [`05_intermittent-classification.md`](05_intermittent-classification.md)
- Reconciliation and aggregation: [`02_aggregation-consistency.md`](02_aggregation-consistency.md) and [`03_mint-reconciliation.md`](03_mint-reconciliation.md)
- The point-error metrics this topic does not re-teach: [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Rolling-origin evaluation: [`../07_time_series_forecasting/08_rolling-origin-backtesting.md`](../07_time_series_forecasting/08_rolling-origin-backtesting.md)
- Intermittent-demand measures: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Inventory policy that consumes the scored quantile: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- The forecasting module's survey: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Return to the part map: [`../_index.md`](../_index.md)
