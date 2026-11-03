> Module 07 — Time Series Forecasting Foundations
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topic 09; Module 04 topic 02; Module 07 topics 01–06
> Version 2026.5

# Forecast Combination — why averaging beats model selection more often than expected

---

## 1. Why this topic exists

A demand planning team has four forecasting methods in production: a seasonal exponential smoothing
model, an ARIMA model with calendar regressors, a regression on price and promotion flags, and a
gradient-boosted model built by the data science team. Every quarter the team compares the four on
the last twelve months, picks the one with the lowest error, and runs it for the next quarter. Every
quarter, the winner changes.

The pattern has a name: **the forecast combination puzzle**. The puzzle is not that averaging helps.
It is that the *simplest possible* average — equal weights, one over the number of forecasts —
routinely beats weights estimated to be optimal on the data available. The weights are optimal in
sample. Out of sample they are usually worse than doing nothing clever at all.

Three things make this more than a curiosity.

**The stakes are routine and large.** A one-point reduction in forecast error at item–location level
translates into safety stock, expediting, and obsolescence. It is not a result that stays in a paper;
it is a monthly planning decision repeated across thousands of series.

**Model selection has a hidden cost.** Choosing the best of *k* methods on a finite evaluation window
is a selection over noise. When the methods are close in accuracy, the winner is largely the method
that got lucky in that window, and the luck does not persist. Timmermann's survey of the field says
it plainly: choosing the single forecast with the best track record is often a bad idea, because
forecast performance is not persistent.

**Combination is nearly free.** It costs computer time and a line in the planning system. It does
not cost a new data source, a new hire, or a change to the planning calendar. Almost no other
intervention in forecasting has that ratio of benefit to organisational friction. The operational
point of this topic is therefore blunt: if you have two or more defensible forecasts, average them
before you spend a quarter deciding which one to trust.

There is a counter-current worth naming at the start. Combination reduces *variance* — the part of
error that comes from each method's idiosyncratic noise. It does not reduce *bias* that all the
methods share. If four models all ignore a promotion calendar, their average ignores it too, with a
smaller standard error and therefore more confidence. Averaging makes a shared blind spot look more
respectable. Section 11 treats this as an assumption with a name: errors must not be perfectly
correlated, and they are never independent in practice.

The topic also has a governance dimension. A forecasting competition and a combination benchmark are
both legitimate evaluation designs, but they answer different questions. A competition asks "which
single method is best on this dataset?" A benchmark asks "what does a portfolio of methods achieve?"
If the rule is chosen after the leaderboard is visible, the comparison is meaningless either way.

This topic sits after the five method families and after intermittent demand. It assumes you can
produce a defensible forecast from each and that you have a backtest on which to score them. What it
adds is the step most teams skip: constructing a portfolio deliberately, rather than searching
repeatedly for a champion.

> **Note:** Nothing here argues that combining is always the answer. On a series where one method is
> structurally correct and the others are misspecified in the same direction, the average can be
> worse than the best single model. The claim is empirical and conditional: on the populations where
> forecasting has been measured at scale, equal-weight combination is a hard benchmark.

## 2. Learning outcomes

By the end of this topic you can:

1. **Derive** the variance of an average of two forecasts from their error variances and the error
   correlation, and **state** in units what the correlation term contributes.
2. **Distinguish** combining forecasts from combining models, and **say** which one a production
   design implements.
3. **Compute** equal-weight, median, and trimmed-mean combinations by hand, and **choose** among them
   with a stated reason.
4. **Estimate** regression-based and variance-based weights, **identify** their estimation error on a
   short sample, and **explain** why shrinkage toward equal weights is the standard remedy.
5. **Design** a backtest in which a selection rule and a combination rule are compared fairly, both
   declared before results are seen.
6. **Diagnose** a combination that fails, attributing it to correlated errors, weight instability, a
   level shift, or a horizon mismatch rather than to "bad models".
7. **Explain** why the choice between a model-selection competition and a combination benchmark is a
   governance decision, and **write** the declaration that makes either defensible.
8. **Quantify** the operational cost of combination — compute and maintenance, not insight — and
   **state** when that cost is not worth paying.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is the arithmetic of averaging two errors. The second is the distinction
between the two things a team can do with a set of candidate models.

### 3.1 The averaging picture

```text
   Two forecasters, twelve periods, errors in units of demand

   period   1    2    3    4    5    6    7    8    9   10   11   12
   B       -10    4  -16    8   -6   12   -4   14   -8    6  -12   12
   D        14  -16   18  -12   16  -14   12  -16   14  -12   14  -18

   B alone:  MSE 101.3      D alone:  MSE 219.3
   Each carries a signal and a noise. The signal is shared; the noises
   mostly point in opposite directions.

   average  2   -6    1   -2    5   -1    4   -1    3   -3    1   -3
   average:  MSE   9.7

   The average beats either input, not because it models demand better,
   but because the errors cancel.
```

The cancellation is the whole mechanism. Averaging keeps the signal, which is shared, and shrinks
the noise, which is not. When the noises point in opposite directions the shrinkage is large; when
they point the same way it is small.

```text
   error of the average, two forecasts with equal variance sigma^2

   Var( (e1 + e2) / 2 )  =  sigma^2 * (1 + rho) / 2

   rho = +1.0   ->  sigma^2          no gain; the two are the same forecast
   rho =  0.0   ->  sigma^2 / 2      variance halved; standard error / 1.41
   rho = -0.5   ->  sigma^2 / 4      variance quartered; standard error halved
   rho = -1.0   ->  0                perfect offset, which never occurs

   General n, average pairwise correlation rho_bar:
   Var(mean) = sigma^2 / n * (1 + (n - 1) * rho_bar)
```

Two consequences organise the rest of the topic.

**Consequence 1 — diversity of error matters more than individual accuracy.** A mediocre model
whose errors are uncorrelated with the rest is worth more to a portfolio than an excellent model
that repeats everyone else's mistakes. Batchelor and Dua (1995) made this explicit, measuring the
benefit of combination as a probability of variance reduction and showing it rising with the
diversity of the component forecasters. When you screen candidates, look at the correlation of their
errors, not only at standalone error. A model that is 10% worse but 0.3 correlated with the incumbent
is often the more valuable addition.

**Consequence 2 — estimated weights carry their own error.** The formula uses the true variances and
the true correlation. In practice you estimate them from a finite sample. The estimate of a weight
is a random variable whose variance grows with the number of models and falls with the length of the
window. Combining many models on a short history is fitting noise with extra steps. Smith and Wallis
(2009) give the clean explanation; Claeskens, Magnus, Vasnev and Wang (2016) show that once weight
randomness is accounted for, there is no guarantee the "optimal" combination beats equal weights, or
even beats the individual forecasts.

### 3.2 The two procedures picture

```text
   Same four candidate forecasts, two legitimate procedures

   SELECTION                                COMBINATION
   ---------                                -----------
   score each on the backtest               score each on the backtest
            |                                        |
   pick the lowest error                    form a portfolio (equal, median,
            |                                trimmed, weighted)
   deploy one model                                  |
            |                               score the PORTFOLIO on the
   discard the other three                  same backtest
            |                                         |
   next quarter, the winner changes         deploy the portfolio
                                                      |
                                            all four stay in production

   Answers: "which is best?"                Answers: "what is the best
                                             portfolio I can build?"
```

The two are not rivals statistically. They are rival *policies*, and which one an evaluation is
entitled to claim depends on what was declared before the results were seen. A competition is
legitimate if entrants, metric, and selection window are fixed in advance. A combination benchmark is
legitimate if the portfolio rule is fixed in advance and scored on the same held-out period. What is
not legitimate is to run a competition, notice that a combination would have won, and report the
combination as the result. That is selection on the evaluation set wearing a different hat.

### 3.3 Four operations that get called "combining"

| Operation | What is averaged | What it fixes | What it does not fix |
|---|---|---|---|
| Combining forecasts | Output values of several models at one horizon | Idiosyncratic model noise | Shared bias; a missing driver |
| Combining models | Structure or parameters, e.g. averaged coefficients | Instability of one fitted structure | Misspecification common to all inputs |
| Combining across horizons | Separate portfolios at `h = 1`, `h = 2`, … | Horizon-dependent skill | Inconsistent paths across horizons |
| Combining across levels | Forecasts at item, family, total, reconciled | Disaggregation noise | Level-specific bias |

The first is what this topic is mostly about. The second is a different statistical object:
averaging coefficients is not averaging predictions, and it can be much worse, because the average of
two well-specified models need not be well specified. The third and fourth are refinements treated in
sections 5.7 and 5.8.

> **Note:** A "model ensemble" in machine learning usually means averaging predictions from many
> fits of the same algorithm with different seeds. That is variance reduction on one model family,
> not a combination of forecasts, and it is not evidence that combining *different* method families
> will help.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Forecast combination | One forecast built from two or more component forecasts of the same target | Used for any model averaging, including coefficient averaging | Target identifier plus horizon |
| Component forecast | An individual forecast entering a combination | Called a "model" when it may be a judgemental forecast | Units of the target |
| Combination weight | The share of a component in the combined value | Presented as a measure of model quality | Proportion, summing to 1 |
| Equal-weight average | Arithmetic mean of components, `1/n` each | Dismissed as naive; it is the standard to beat | Units |
| Simple average | Synonym for the equal-weight average | Confused with a mean forecast over history | Units |
| Median combination | The middle component value at each target | Treated as an average with different arithmetic | Units |
| Trimmed mean | Mean after removing the highest and lowest `100λ` per cent | Trim factor left unstated | Proportion `λ`, 0 to 0.5 |
| Winsorized mean | Mean after capping the extremes rather than deleting them | Confused with the trimmed mean | Proportion capped per tail |
| Trim factor | The fraction removed from each tail before averaging | Set to 0.5, which returns the median | Proportion |
| Optimal weights | Weights minimising expected squared error given the true covariance | Called optimal when estimated on a finite sample | Proportion |
| Estimated weights | Weights computed from sample variances and covariances | Reported without standard errors | Proportion |
| Bates–Granger weights | Inverse-variance weights using only the covariance diagonal | Assumed to handle correlated errors | `w_i ∝ 1/s_i²` |
| Inverse-variance weighting | Any scheme where weight rises as component variance falls | Applied to biased forecasts, where variance is the wrong criterion | Proportion |
| Regression-based weights | Coefficients from regressing the actual on component forecasts | Intercept left in, which makes it a calibration model | Regression coefficients |
| Constrained weights | Weights restricted to be non-negative and to sum to one | Applied without checking that the constraint binds | Proportion |
| Shrinkage | Pulling estimated weights toward equal weights | Confused with regularising the component models | Intensity, 0 to 1 |
| Combination puzzle | The finding that equal weights often beat estimated optimal weights | Described as a paradox rather than an estimation problem | None |
| Estimation error | Sampling error in weights estimated from a finite window | Ignored when weights are reported to four decimals | Variance or standard error |
| Error correlation | Correlation of two components' errors on the same targets | Computed across horizons instead of across periods | Dimensionless, −1 to 1 |
| Average pairwise correlation | Mean of all pairwise error correlations in a portfolio | Reported without the number of components | Dimensionless |
| Error covariance matrix | All variances and covariances of component errors | Estimated with fewer observations than parameters | Units squared |
| Diversity | The degree to which component errors differ in direction and timing | Equated with using different algorithms | Dimensionless |
| Portfolio effect | Variance reduction from combining imperfectly correlated forecasts | Assumed free and unlimited | Variance ratio |
| Variance reduction | Fall in error variance relative to a stated comparator | Reported without the comparator | Proportion or units squared |
| Shared bias | A bias common to all components, which averaging preserves | Mistaken for noise that combination removes | Units |
| Model selection | Choosing one model from a set on the basis of evaluation results | Called combination when the choice is made per period | Selection rule |
| Winner's curse | The tendency of the selected winner to underperform afterwards | Treated as bad luck | None |
| Ex ante | Chosen or fixed before the evaluation window | Used loosely for "before deployment" | Boolean |
| Ex post | Chosen with knowledge of the evaluation window | Reported as if it were ex ante | Boolean |
| Model averaging | Weighting several models' outputs by a criterion | Used as a synonym for Bayesian model averaging | Weight vector |
| Bayesian model averaging | Posterior-probability weights over models | Applied with equal priors and called BMA | Posterior probability |
| Meta-model | A model fitted on the outputs of component models | Fitted on the same data used to fit the components | Model identifier |
| Stacking | A meta-model with cross-validated component predictions | Leaking the target into the meta-features | Model identifier |
| Horizon-specific combination | A separate weight vector for each forecast horizon | Assumed unnecessary; the evidence is thin | Horizon index |
| Level-specific combination | A separate portfolio at each aggregation level | Confused with reconciliation | Level identifier |
| Forecast reconciliation | Adjusting forecasts so levels sum consistently | Treated as a combination method | Level hierarchy |
| Tracking signal | Cumulative error divided by mean absolute deviation | Used as a combination weight | Ratio |
| Break-even model count | The count beyond which added estimation error outweighs added diversity | Assumed unlimited | Count |
| Forecast value added | Error change attributable to one process step | Reported without a single-model comparator | Percent |
| Pre-registration | Writing the evaluation design before results are produced | Treated as paperwork rather than method | Document version |
| Forecast vintage | The version of a forecast as of a given origin | Mixed across components | Timestamp |
| Target alignment | Ensuring every component forecasts the same defined quantity | Assumed because the columns share a name | Definition identifier |
| Ensemble | A set of models whose outputs are pooled | Used for any group of models being compared | Model set |

Four of these repay a sentence, because the distinctions carry most of the practical weight.

**Equal-weight average versus mean forecast.** The mean forecast of Topic 01 averages the *history*
of one series and repeats it. The equal-weight average here averages several *forecasts* of the same
future target. Same arithmetic, different objects: one is a null model, the other a portfolio.
Confusing them produces the strange claim that "the mean forecast beat the model".

**Optimal versus estimated weights.** "Optimal" is a property of a weight vector relative to a known
covariance matrix. Once you estimate that matrix the weights are estimates, and the word optimal
becomes a claim about a population you cannot see. The literature's finding is not that optimal
weights are bad; it is that estimated weights, used as if optimal, are bad.

**Diversity versus decorrelation.** Diversity is the property you want; decorrelation is an
imperfect proxy. Two models can be uncorrelated and both be noise around the same wrong level. A
model built on a different information source can correlate at 0.6 with the incumbent and still add
value. Treat correlation as a screening statistic, not the objective.

**Combination versus calibration.** A regression of actual on forecast with an intercept does both:
the slope is a weight, the intercept a bias correction. If the intercept is materially non-zero you
have learned something about the component's bias, and that finding belongs in the report rather
than buried in a weight.

## 5. The mechanics, step by step

This section runs in the order a combination project should run: frame first, alignment second,
portfolio third, evaluation last. The order matters because the most common failure — choosing the
portfolio rule after seeing which one won — is prevented by nothing except sequence.

### 5.1 Fix the target, the horizon, and the population

**Step 1. Define the target once.** Every component must forecast the same quantity: the same
item–location key, unit, calendar bucket, and definition of demand (shipments, orders, or
consumption). *Output:* one versioned target definition. *Branch:* if one component forecasts
shipments and another point-of-sale (POS) scans, they are not combinable without a conversion, and
the conversion is a model in its own right.

**Step 2. Fix the horizon set.** Combination weights can differ by horizon, and component forecasts
certainly do. Decide whether you combine at each `h` separately or pool across horizons, and write
it down. *Branch:* pooling assumes the relative skill of components is stable across `h`, which is
often false — a regression with promotion flags typically wins at short horizons, a seasonal method
at long ones.

**Step 3. Fix the evaluation population.** Same series, same origins, same inclusion rules for new
and discontinued items, for every component and for the portfolio. *Branch:* if a component cannot
forecast some series, score the portfolio on the intersection and report the exclusion count.

**Step 4. Fix the forecast vintage.** Each component's forecast must be the one available at the
origin. A model refitted with today's revised history is not the forecast it would have made
eighteen months ago. *Branch:* if point-in-time history is unavailable, say so, and treat the result
as optimistic for all components equally.

### 5.2 Put every component on the same footing

Alignment is where most real combination projects quietly fail, because the errors are small enough
to survive a meeting and large enough to reverse a ranking.

**Step 5. Align the calendar.** ISO weeks, 4-4-5 retail periods, and calendar months do not map onto
each other without a rule. Pick one and convert every component's output to it.

**Step 6. Align the sign and the definition of error.** Error is `actual − forecast` throughout this
topic. A vendor that reports `forecast − actual` will produce a covariance matrix with every
off-diagonal sign flipped, and weights that look absurd for a reason nobody can find.

**Step 7. Check for level bias before weighting.** A component with a mean error of −4% of average
demand is biased low, and its variance understates its true error. *Branch:* correct the bias and
report the correction, or exclude the component. Do not let a regression intercept absorb it
silently; you will have combined a bias correction with a combination without noticing.

**Step 8. Check the scale.** Components forecasting in different units or at different aggregation
levels cannot be averaged. Convert to a common unit at a common level first.

### 5.3 Choose the combination family

Five families differ in how much they ask of the data.

**Family 1 — Equal weights.** `w_i = 1/n`. Requires nothing but the forecasts. It is the benchmark
every other family must beat, and in the published comparisons it frequently does not. Its weakness
is that it gives a broken component the same weight as a good one.

**Family 2 — Robust order statistics.** The median, and the trimmed mean with trim factor `λ`. Both
discount extremes without estimating anything. `λ = 0` returns the mean, `λ = 0.5` the median. Jose
and Winkler (2008) found that trimming 10 to 30% per tail slightly beat the simple average on M3
data and reduced the risk of large errors, with more trimming indicated when components disagree.

**Family 3 — Variance-based weights.** Bates–Granger weights `w_i ∝ 1/s_i²`, or the full-covariance
optimum. The diagonal version needs `n` variance estimates; the full version needs `n(n+1)/2`
covariances.

**Family 4 — Regression-based weights.** Regress realised values on the forecasts,
`y = a + Σ w_i f_i + e`, usually with `Σ w_i = 1` imposed and often with `a = 0`. The coefficients are
the weights. This is the encompassing test: if one weight is 1 and the rest 0, that component
encompasses the others and combination adds nothing.

**Family 5 — Shrinkage and constraints.** Estimate, then pull toward `1/n`:
`w_shrunk = (1 − k) w_est + k (1/n)`. Or constrain weights to be non-negative. Constraints prevent
the pathological negative weights that unconstrained regression produces when components are highly
collinear. Timmermann's survey lists trimming the worst models and shrinkage among the few
refinements that reliably improve on simple schemes.

| Family | Parameters | Data needed | Strength | Failure mode |
|---|---:|---|---|---|
| Equal weights | 0 | None | Cannot overfit; hard to beat | Weights a broken component equally |
| Median / trimmed | 0 or 1 | None | Robust to a wild component | Discards information; poor with `n = 2` |
| Variance (diagonal) | `n` | 20+ errors per component | Uses relative precision | Ignores correlation; unstable as `n` grows |
| Variance (full) | `n(n+1)/2` | Many times `n` errors | Theoretically efficient | Covariances from too little data |
| Regression | `n` (+ intercept) | Many times `n` errors | Detects encompassing | Negative weights; in-sample optimism |
| Shrinkage | `n` + 1 | As above | Controls weight variance | The intensity itself needs choosing |

### 5.4 Estimate weights without fooling yourself

**Step 9. Count parameters against observations.** With `n` components and `T` periods, the full
problem estimates `n(n+1)/2` quantities. For `n = 5` that is 15; a 24-period window gives 1.6
observations per parameter. *Rule of thumb:* if `T < 10n`, do not use full-covariance weights. Use
the diagonal version, shrinkage, or equal weights.

**Step 10. Estimate on one window, evaluate on another.** The windows must be disjoint. This is the
most important sentence in the section. A weight vector evaluated on the data used to estimate it
will look excellent and mean nothing.

**Step 11. Report the weight standard errors.** A weight of 0.62 with a standard error of 0.31 is not
evidence that the component deserves 62%. It is evidence that the data cannot distinguish 62% from
31% or 93%. Reporting weights without uncertainty is a regression coefficient without a standard
error.

**Step 12. Shrink, then check whether shrinkage helped.** Choose the intensity on a validation
window, not the evaluation window. If `k = 1` wins, the honest conclusion is that the data support
equal weights and the estimated weights were noise.

**Step 13. Test for encompassing.** Fit regression weights and test that one weight is 1 and the rest
0. If the hypothesis is not rejected, the single model is adequate and combination adds nothing on
this population. Report the test; do not suppress it because the portfolio is the preferred answer.

### 5.5 Score the portfolio, not just the components

**Step 14. Score every component and every candidate portfolio on the same backtest.** One row per
object, identical columns.

**Step 15. Report the best single component ex post, and label it.** This number makes the
portfolio's advantage visible and is unfair as a decision rule, because it could not be known in
advance. Put it beside the best selected ex ante — the component that won the *previous* window.

**Step 16. Report the win rate across series, not only the pooled error.** A portfolio 5% better on
average and worse on 55% of series is a different product decision from one better everywhere.
Topic 01, section 5.5, gives the reporting template.

**Step 17. Report error by horizon.** Combination gains are usually largest at short horizons, where
idiosyncratic noise dominates, and shrink at long horizons, where shared bias dominates. A flat gain
across horizons is a signal that the components are more similar than you think.

### 5.6 Declare the evaluation design before you look

**Step 18. Declare the design class.** Either a **model-selection competition** — "we will choose one
method from this fixed list on this metric and window" — or a **combination benchmark** — "we will
deploy a fixed portfolio rule and compare it to the components". Both are legitimate; they produce
different winners, and the winner of one is not evidence about the other.

**Step 19. Declare the portfolio rule if combination is the design.** Name the family, the estimation
window, the shrinkage rule, and the horizon treatment, before the evaluation window is scored.

**Step 20. Declare the comparison rule.** State which number the decision uses: the best ex-ante
component, the best ex-post component, or the portfolio. "Whichever is lowest" is a competition with
an unlimited entrant list, and the reported improvement becomes a maximum over many candidates.

**Step 21. Declare the pass rule.** A portfolio ships if it beats the best ex-ante component by more
than the larger of a stated margin and twice the standard error of the difference, on the
pre-registered metric and population. The margin pays for running `n` models instead of one.

**Step 22. Version and date the declaration.** A rule written after the leaderboard is visible is not
a rule.

```text
   Declaration, one page, dated before scoring

   design class      : combination benchmark
   components        : ES (damped), SARIMA, regression with promotion flags, GBM
   target            : demand in eaches, SKU x region, calendar month
   horizons          : 1..4, weights estimated separately per horizon
   portfolio rule    : equal weights; median and trimmed (lambda = 0.2) as sensitivity
   estimation window : rolling 36 months, weights re-estimated annually
   evaluation window : 24 months, rolling origin, 12 origins held out
   primary metric    : WAPE, volume-weighted, per horizon
   comparator        : best ex-ante component (selected on prior 12 months)
   pass rule         : portfolio WAPE <= comparator WAPE - max(0.03, 2 * SE_diff)
```

### 5.7 Combination across horizons

A common mistake is to estimate one weight vector and apply it at every horizon. Relative skill is
usually horizon-dependent: short-horizon forecasts are dominated by recent level and calendar
effects, long-horizon forecasts by trend and seasonality. A regression with promotion flags may be
the best component at `h = 1` and the worst at `h = 12`.

Two treatments are defensible. **Separate weights per horizon** estimate `w(h)` from errors at that
horizon, multiplying the parameter count by the number of horizons, so they need a long history and a
small component set. **Pooled weights with a horizon interaction** borrow strength across horizons at
the cost of assuming the ranking changes smoothly.

The third option — ignore the issue and pool — is defensible only if the backtest shows the ranking is
stable across horizons. Compute per-horizon error ratios; if the ranking flips, pooled weights average
contradictory recommendations.

> **Note:** The published evidence on horizon-specific combination weights is thinner than the
> confidence with which practitioners recommend it. The argument is structural — component skill
> varies with `h` — but direct empirical tests are scarce. Treat per-horizon weighting as a hypothesis
> to test on your own backtest, not an established result.

### 5.8 Combination across levels, and reconciliation

**Combining within a level** pools different methods forecasting the same aggregate; that is this
topic. **Reconciling across levels** adjusts forecasts so they add up; that is Module 08. They
interact: combine at the item level and then reconcile, and reconciliation overwrites part of the gain.
The rule is to combine where the decision is made, reconcile afterwards, and score the reconciled
object rather than the intermediate one.

### 5.9 Operate the portfolio

**Step 23. Keep every component running.** The portfolio needs `n` forecasts each cycle. A component
retired after losing a contest is unavailable when the regime changes and it would have helped.

**Step 24. Re-estimate weights on a schedule, not continuously.** Annual or semi-annual
re-estimation is typical. Continuous re-estimation chases noise and makes the portfolio impossible
to explain.

**Step 25. Monitor dispersion and correlation.** A jump in the standard deviation across component
forecasts usually means one component has seen a level shift the others have not. A rise in the mean
pairwise correlation toward 1 means the portfolio has become one model with extra cost.

**Step 26. Report the portfolio's FVA against the best single model, monthly.** A portfolio that
stops beating the best component is not necessarily broken, but the number belongs in the monthly
pack, not the annual review.

> **Note:** The operational cost of combination is small: `n` model runs, a weighting step, a
> monitoring line. The cost that is not small is explanatory. A planner who cannot see why the number
> moved will override it, and the override is usually worse than the portfolio.

## 6. Worked example

### 6.1 Setting

Larkfield Industrial Supply distributes fasteners to maintenance, repair, and operations (MRO)
customers. We follow one product family in one region, forecast monthly, one month ahead, in eaches.
The family averages about 1,200 eaches a month, with a mild autumn peak and no strong trend.

Four components are in production: **A**, damped exponential smoothing with `m = 12`; **B**, a
SARIMA with a seasonal difference and one autoregressive term; **C**, a regression on month dummies,
a promotion flag, and a customer-count index; **D**, a gradient-boosted model on 34 features,
refitted annually.

The team has 24 months of one-step-ahead errors, produced with the history available at each origin.
Months 1 to 12 are the weight-estimation window; months 13 to 24 are the evaluation window. The split
is arbitrary but declared in advance, which is the only property that matters. All figures are
constructed to be internally consistent; they are not taken from a company.

### 6.2 The estimation window, months 1 to 12

Errors in eaches, `e = actual − forecast`, so positive means the forecast was too low.

| Month | A | B | C | D |
|---:|---:|---:|---:|---:|
| 1 | 10 | 6 | −8 | 2 |
| 2 | −12 | 10 | 4 | −6 |
| 3 | 8 | −14 | 18 | 4 |
| 4 | −6 | 12 | −10 | −2 |
| 5 | 14 | −8 | 6 | 8 |
| 6 | −10 | 16 | −14 | −4 |
| 7 | 6 | −12 | 10 | 0 |
| 8 | −8 | 10 | 4 | −10 |
| 9 | 12 | −6 | −16 | 6 |
| 10 | −4 | 14 | 8 | −2 |
| 11 | 10 | −10 | −6 | 4 |
| 12 | −20 | −18 | 4 | 0 |
| **MSE** | **116.7** | **141.3** | **102.0** | **24.7** |

Every component is unbiased by construction, which isolates the variance question. D is far and away
the most accurate single model: MSE 24.7 against 102.0 for the next best. A selection rule applied at
month 12 would deploy D without hesitation, and that is what the team did.

The equal-weight average at each month, `(A + B + C + D) / 4`, is:

| Month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal weight | 2.5 | −1.0 | 4.0 | −1.5 | 5.0 | −3.0 | 1.0 | −1.0 | −1.0 | 4.0 | −0.5 | −8.5 |

The sum of squares is 151.0 over 12 months, so the equal-weight MSE is **12.58** — better than D's
24.7 and roughly eight times better than the worst component. In the very window used to estimate
weights, the unweighted average already won. This is the first warning, and it is routinely missed
because nobody computes the average once a "best model" has been selected.

### 6.3 Estimating weights on the same window

The team used the inverse-variance rule that practitioners usually call optimal, and that Bates and
Granger introduced in 1969:

```text
   w_i  proportional to  1 / s_i^2

   s^2:   A 116.67   B 141.33   C 102.00   D 24.67
   1/s^2: A 0.008571  B 0.007075  C 0.009804  D 0.040541
   sum   = 0.065991

   w:     A 0.1299    B 0.1072    C 0.1486    D 0.6143
```

D receives 61.4% because its variance is four to six times smaller than the others. That is a
defensible reading of the window. It is also, as the next table shows, a weight vector that does not
transport. Applied to the same twelve months, the MSE is **13.51** — worse than the equal-weight
average of 12.58 even in sample, because the diagonal rule ignores the covariances, which here are
large and negative. A rule that is optimal for a diagonal covariance matrix is not optimal for the
actual one.

> **Note:** The full-covariance optimum would use all six pairwise covariances: ten parameters from
> twelve observations. It is computable and it is not defensible. This is the point at which a
> practitioner should choose a rule with fewer parameters rather than a better formula.

### 6.4 The evaluation window, months 13 to 24

The environment shifted in month 13: a competitor exited the region, two large accounts changed
their ordering pattern, and the promotion calendar was rebuilt.

| Month | A | B | C | D |
|---:|---:|---:|---:|---:|
| 13 | 8 | −10 | 6 | 14 |
| 14 | −6 | 4 | −8 | −16 |
| 15 | 12 | −16 | 10 | 18 |
| 16 | −10 | 8 | −4 | −12 |
| 17 | 4 | −6 | 14 | 16 |
| 18 | 16 | 12 | −12 | −14 |
| 19 | −8 | −4 | 8 | 12 |
| 20 | 6 | 14 | −6 | −16 |
| 21 | −12 | −8 | 10 | 14 |
| 22 | 10 | 6 | −14 | −12 |
| 23 | −4 | −12 | 8 | 14 |
| 24 | −16 | 12 | −12 | −18 |
| **MSE** | **102.7** | **101.3** | **96.7** | **219.3** |
| **MAE** | **9.33** | **9.33** | **9.33** | **14.67** |

D, the selected model, is now the worst component by a factor of more than two. The other three are
indistinguishable on MAE. Nothing about D's construction changed; the relationship between its
features and demand did.

### 6.5 Every rule, scored on the evaluation window

| Object | Rule | MSE | MAE | Rank |
|---|---|---:|---:|---:|
| Equal weights | `1/4` each | **21.2** | **3.75** | 1 |
| Median | middle value of four | 46.9 | 5.25 | 3 |
| Trimmed mean, `λ = 0.2` | trim one per tail, average two | 46.9 | 5.25 | 3 |
| Inverse-variance weights | from months 1–12 | 93.1 | 9.40 | 4 |
| Best ex post component | C, chosen with hindsight | 96.7 | 9.33 | 5 |
| Best ex ante component | D, chosen at month 12 | 219.3 | 14.67 | 6 |

Four readings, and the third is the one most people get wrong.

**Equal weights win by a wide margin.** MSE 21.2 against 96.7 for the best component chosen with
hindsight and 219.3 for the component a selection rule would have deployed. The average is not a
compromise between good and bad models; it is a portfolio whose errors cancel.

**The estimated weights lose by a factor of 4.4 in MSE.** They were not wrong for months 1 to 12; they
were wrong for months 13 to 24, and there was no way to know at the time. That is estimation error, not
a modelling mistake, and it is why the puzzle exists.

**The median is worse here, and that is not a contradiction.** Robust rules discount a wild component;
here the wild component was D, whose errors were strongly negatively correlated with B's, so
discounting D threw away the cancellation. With four components a 20% trim equals the median; trimming
needs five or more to differ from both. Use robust rules when a component may be *wrong*, not when it
may be *unusual*.

**The best possible pair would have been better still.** B and D alone give MSE 9.7, but no rule
available at month 12 would have chosen that pair, and the advantage depends on a correlation invisible
in the estimation window. It is an upper bound on what diversity can buy.

### 6.6 The average pairwise correlation

| Pair | Correlation | Reading |
|---|---:|---|
| A–B | −0.07 | Nearly independent |
| A–C | −0.07 | Nearly independent |
| A–D | +0.08 | Nearly independent |
| B–C | −0.87 | Strong offset |
| B–D | −0.95 | Strong offset |
| C–D | +0.93 | Near-duplicate |
| **Mean** | **−0.16** | |

The mean is negative, which is why the equal-weight average beats every component. The variance
approximation gives

```text
   Var(mean)  ~  sigma^2 / n * (1 + (n - 1) * rho_bar)
              ~  130.0 / 4 * (1 + 3 * (-0.157))
              ~  32.5 * 0.529  ~  17.2
```

against an actual MSE of 21.2. The approximation assumes equal variances; D's is roughly twice the
others, which is why it understates the realised value. The direction is right and the magnitude is
close enough to be a sanity check.

### 6.7 What the team should have done at month 12

1. **Deploy the equal-weight average** and keep all four models running. No parameters, and on this
   evidence the best portfolio in both windows.
2. **Deploy a shrunk weight vector**, `0.5 × estimated + 0.5 × equal`, which would have given D 43%
   rather than 61% and roughly halved the damage.
3. **Deploy D but monitor dispersion**, with a rule that the average is used whenever the
   cross-model standard deviation exceeds a threshold.

The action taken — deploying D alone because it won the selection contest — was the worst of the four
options and the most natural one to take.

### 6.8 Across horizons

Larkfield also forecasts three months ahead for contract customers, and there the ranking changes.

| Horizon | Best component | Equal-weight skill vs best component | Best component's rank at `h = 1` |
|---:|---|---:|---:|
| 1 | D | +8% | 1 |
| 2 | C | +11% | 2 |
| 3 | C | +14% | 2 |
| 6 | B | +19% | 3 |

At short horizons the feature-rich model leads and the average adds modestly. At longer horizons the
seasonal and ARIMA components lead, the feature-rich model decays, and the average adds more because
the components disagree more. A pooled weight vector would average a recommendation to trust D with a
recommendation to distrust it. The table is constructed to illustrate the pattern, which is consistent
with the finding that combination gains are larger where component errors are less correlated.

## 7. Code, query, or calculation

The arithmetic below is meant to be done by hand once, because it makes the mechanism visible in a
way a library call does not. It uses components B and D from section 6, evaluation window only.

```text
   s_B^2 = 101.33     s_D^2 = 219.33     rho = -0.946
   s_B = 10.07        s_D = 14.81
   cov = rho * s_B * s_D = -0.946 * 149.08 = -141.03

   equal weights, w = 0.5 each:
   Var = 0.25*101.33 + 0.25*219.33 + 2*0.25*(-141.03)
       = 25.33 + 54.83 - 70.52 = 9.64            RMSE = 3.10

   optimal weight on D given these values:
   w*   = (s_B^2 - cov) / (s_B^2 + s_D^2 - 2*cov)
        = (101.33 + 141.03) / (101.33 + 219.33 + 282.06)
        = 242.36 / 602.72 = 0.402
   Var  = 0.598^2*101.33 + 0.402^2*219.33 + 0.4808*(-141.03)
        = 36.24 + 35.44 - 67.81 = 3.87           RMSE = 1.97

   w_D estimated on months 1-12 (two-model case) = 0.740
   Var at w_D = 0.740 applied to months 13-24    = 72.7    RMSE = 8.53
```

Three checks before trusting a calculation of this kind.

**Check the sign of the covariance.** A large positive covariance with a large weight is the common
case and produces a combination barely better than the best component. A large negative covariance
produces a spectacular result and should be interrogated: are the two components forecasting the
same target, or is one inverted?

**Check the weight against the data.** An optimal weight of 0.402 with a standard error of 0.25 is not
a precise instruction. On twelve observations, standard errors of that size are normal, and they are
why shrinkage exists.

**Check that the stationarity assumption holds.** The formula assumes errors are stationary over the
window. If the covariance structure changed between windows — as it did in section 6 — the formula
describes a population that no longer exists. That is not a reason to abandon it; it is the reason to
report the weight estimation window alongside the weight.

## 8. Data, standards, and regulatory context

No international standard prescribes how to combine forecasts. There is no ISO document that fixes a
weight vector, no regulation requiring an equal-weight average, and no accounting standard that says
whether a portfolio or a selected model is the right object to feed a plan. What exists is a
vocabulary standard, calendar and measurement conventions, governance practice that applies by
analogy, and — for listed companies — a disclosure regime that reaches forecasts where they feed
public guidance.

### 8.1 What the standards supply

| Standard or framework | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of error, bias, variance, estimator | Any combination rule or weight | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | "Accuracy" is not a quantity and has no numerical value | Whether a portfolio beats a component | B |
| ISO 5725-1 | Trueness and precision of measurement methods | Accuracy as bias plus precision | Forecast evaluation, which has no repeated reference | B |
| ISO 8601 | Date and time representation | Week numbering, ordering, durations | Which week a demand observation belongs to | B |
| SR 11-7 (U.S. Federal Reserve) | Model risk management for banking | Validation, documentation, monitoring | Demand planning models; it does not govern them | B |
| Sarbanes-Oxley Section 404 | Internal control over financial reporting | Controls over figures feeding reported financials | The statistical design of a combination | B |
| IFRS 15 / ASC 606 | Revenue recognition | When control transfers and revenue exists | Which forecast to use for planning | B |
| ASCM/APICS Dictionary | Supply chain terminology | Definitions of forecast, plan, consensus demand | Any method or weight | C |

The VIM point transfers most directly. "The combination is 8% more accurate" is a claim about a
sample, a metric, and a comparator; it is not a property of the combination the way a melting point is
a property of a metal. The standard does not forbid the sentence, only pretending it has a unique
meaning without the comparator attached. That is why every table here carries a comparator column.

### 8.2 Disclosure, where forecasts become public

A demand forecast rarely appears in a filing directly. It appears through inventory valuation,
through revenue guidance, and through statements about expected demand.

**Inventory valuation depends on the forecast.** A write-down for demand that did not materialise is
an accounting event with a documented basis. A company running a portfolio has a different — usually
smaller — exposure than one running a single selected model, and the difference is auditable if the
archive is kept.

**Guidance is a commitment, not a forecast.** A more accurate portfolio is not automatically more
useful for guidance, because guidance is managed for credibility and consistency rather than for
minimum mean squared error.

**Model risk expectations apply by analogy.** SR 11-7 requires a documented purpose, defined scope,
independent validation, and ongoing monitoring. It does not govern a distributor's demand plan, but it
maps onto a combination: state the purpose, define the component set, validate out of sample, monitor
the weights and the dispersion.

### 8.3 The data a combination requires

| Data object | Why it is needed | Minimum content | Common gap |
|---|---|---|---|
| Component forecast archive | To score every component on the same target and origin | Origin, target key, horizon, value, unit, model version | Only the consensus forecast is stored |
| Error archive | To estimate weights and correlations | Actual, forecast, origin, horizon, sign convention | Errors computed on revised actuals |
| Vintage history | To avoid look-ahead in the weight window | Snapshot of inputs at each origin | Warehouse stores current state only |
| Weight history | To explain and audit the portfolio | Weight vector, window, date, rule | Weights overwritten each cycle |
| Dispersion series | To monitor portfolio health | Standard deviation across components | Not computed |
| Promotion calendar | To interpret dispersion jumps | Event, date, affected items | Held in a spreadsheet |

The forecast archive is the load-bearing object. A team that stores only the consensus forecast cannot
reconstruct why it moved, cannot estimate weights without re-running every model, and cannot answer
"would the average have been better?" after the fact.

> **Note:** Storing component forecasts is cheap — a few columns per target per origin. Not storing
> them is expensive, because the counterfactual disappears. If you take one operational action from
> this topic, take this one.

### 8.4 Sign and revision conventions

Two conventions cause silent sign errors. **Vendor conventions:** several planning suites report
"bias" as forecast minus actual. Joining that to an internal error table flips every off-diagonal
sign and produces perverse weights. **Revision conventions:** an actual revised upward after the
origin changes the error attributed to the forecast, usually making the forecast look better. Record
the actual as first posted, or state that final actuals are used.

### 8.5 What is convention rather than requirement

- **Equal weights as the default.** A benchmark with a strong empirical record, not a requirement.
- **A 10 to 30% trim.** Jose and Winkler's empirical default, not a law; with `n = 4` it collapses to the median.
- **A 5% margin in the pass rule.** A governance convention that pays for migration cost.
- **A correlation threshold of 0.8 or 0.9 for "too similar".** Depends on `n` and on weight estimation error.
- **Annual re-estimation.** Inherited from planning cycles; monthly re-estimation is usually worse.
- **"Best fit" selection in planning software.** A vendor default that runs a selection competition
  whether or not the user intended one.

## 9. Industry practice and case evidence

### 9.1 The origin: two sets of airline forecasts

Bates and Granger published "The Combination of Forecasts" in 1969 in the *Journal of the Operational
Research Society*, then published as *Operational Research Quarterly*, volume 20, issue 4, pages 451
to 468. The paper combined two sets of forecasts of airline passenger data and found that the
composite had a lower mean squared error than either original. Weights were derived from past errors,
and several derivations were examined. (Grade A.)

Two things about the origin are worth keeping. The idea is a diversification argument borrowed from
portfolio theory: two imperfect predictors whose errors do not coincide can be worth more together
than either alone. And the paper's own method was the inverse-variance rule, which section 6 shows
losing to equal weights. The founding method did not survive best.

### 9.2 What the competitions found

| Competition | Year | Scale | Result relevant to combination | Grade |
|---|---|---:|---|---|
| M3 | 2000 | 3,003 series | Combinations were among the strongest entrants; the organisers concluded that statistically sophisticated methods do not necessarily beat simpler ones | A |
| M4 | 2018 | 100,000 series, 61 methods | Of the 17 most accurate methods, 12 were combinations. The winner was a hybrid exponential smoothing–recurrent neural network, reported as roughly 10% better than the "Comb" combination benchmark. All six pure ML entries failed to beat that benchmark | A |
| M5 | 2022 | 42,840 daily retail series | The winning entry was an equal-weight average of 220 LightGBM models across three pooling levels; the organisers list "the value of combining" as a finding, and report that blending top-down and bottom-up reconciliation beat either alone | A |

The pattern is consistent and slightly awkward for the puzzle narrative. In M4, combinations
dominated the leaderboard and the winner was a hybrid — itself a combination of a statistical model
and a neural network. In M5 the winner was a *simple equal-weight* average of many models. The most
sophisticated entrants arrived at combination, and the most successful ones arrived at unweighted
combination.

That is the empirical content of "averaging beats model selection more often than expected". Not
that averaging is clever, but that the alternatives, including clever weighting, are fragile in a way
a plain average is not.

> **Note:** The M5 result concerns 42,840 daily retail series from a single retailer, scored on a
> revenue-weighted hierarchical metric. Its gains shrink sharply at the lowest aggregation level,
> where replenishment decisions are made. It is a prior about method families, not a forecast of your
> gain; Topic 01, section 9.2, gives the transferability caveats in full.

### 9.3 What the surveys conclude

Timmermann's chapter in the *Handbook of Economic Forecasting* (2006) is the standard synthesis, and
its conclusions are unusually direct for a survey.

1. **Simple combination schemes are hard to beat.** Equal-weighted averages typically outperform the
   ex-ante best individual model, and simple schemes that ignore error correlations often dominate
   more refined schemes.
2. **Choosing the forecast with the best track record is often a bad idea**, because performance is
   not persistent.
3. **Trimming the worst models usually improves performance.**
4. **Shrinkage often improves performance** — pulling weights toward equal, and constraining them.
5. **Limited time-variation in weights may help**; elaborate time-varying models produce elusive
   gains.

The same survey reviews the record for macroeconomic survey forecasts, where combination gains are well
established, but that population is not SKU-level demand and the transfer should be cautious. The
direction matches what the competitions show.

### 9.4 The puzzle, stated precisely

The puzzle asks: why does `1/n` beat weights that are asymptotically optimal? Four explanations have
survived scrutiny, and they are not mutually exclusive.

**Estimation error.** Weights are estimated from finite samples and the estimates are noisy. Smith
and Wallis (2009) ground the explanation in the sampling variance of the estimated weights and show
it accounts for much of the observed gap. This is the mainstream answer.

**Small potential gains.** Elliott and Liao, in a working paper, argue that with similar component
variances and positive error correlations the population gain from optimal weights is itself small, so
the estimation error needed to lose it is small too. They propose subset averaging as a middle ground.

**Non-stationarity.** The covariance structure that made a weight vector optimal in one window does not
persist. Section 6 is a constructed example; the literature on structural breaks supports it.

**Invalid or unstable formulas.** Some "optimal" rules assume what the data violate, and some are
numerically unstable when components are collinear. Constrained and shrunk versions recover most of
the theoretical gain.

Claeskens, Magnus, Vasnev and Wang (2016) add the sharpest version: optimality is usually derived with
fixed weights, but practice estimates them. Once weight randomness is included, the combination is
biased even if every component is unbiased, its variance is larger, and it may not beat equal weights.

### 9.5 Diversity, measured

Batchelor and Dua (1995) is the canonical reference. They introduced a measure of the benefit of
combining — the probability of a reduction in error variance — and applied it to forecasts from a
panel of U.S. economists. The benefit rose with the number of forecasts combined and with the
diversity of theories and techniques among them. (Grade A.)

A 2025 preprint on infectious disease forecasting makes the same point with a striking number: under
independence, 28 mediocre models would have produced a much larger improvement than the ensemble
achieved, and the binding constraint was the average error correlation among components, around 0.6
and 0.82 to 0.91 within clusters. (Grade C: preprint, not peer reviewed, epidemiological setting.)

The commercial reading is direct. If four components are all gradient-boosted models on the same
feature set with different hyperparameters, their errors correlate at 0.9 and the portfolio adds
almost nothing. If one is a seasonal statistical model, one a regression on commercial drivers, and
one a judgemental forecast from sales, the correlations are lower and the portfolio has somewhere to
gain. Adding a fifth variant of the same model is not diversification; it is the same model with
extra compute.

### 9.6 What practice actually does

| Recommendation | Observed practice | Why the gap persists |
|---|---|---|
| Combine rather than select | Most suites default to "best fit" selection by information criterion | The feature is easy to switch on; combination requires configuration |
| Keep losing models running | Losing models are retired | Cost accounting treats model runs as cost, not portfolio assets |
| Report the best ex-ante comparator | The best ex-post model is reported | It produces the most impressive slide |
| Pre-register the design | Metric and comparison chosen at the results meeting | No process owner for evaluation design |
| Monitor error correlation | Not computed | Requires the component forecast archive |
| Shrink estimated weights | Weights used as estimated, to four decimals | Shrinkage is not exposed in most interfaces |

The last row produces the most damage per unit of effort to fix. A shrunk weight vector is one line of
arithmetic and recovers much of the gain that estimation error destroys.

### 9.7 What practitioners argue about

**"Averaging is for people who cannot choose."** The selection view. It is coherent when one method is
structurally correct and the others are known to be misspecified; it fails when methods are close in
accuracy and the choice is driven by noise, which the record says is the more common case.

**"Our models are too correlated for combination to help."** Often true, and a statement about the
component set rather than about combination. The response is to add a component built on different
information — a customer forecast, a promotion plan, a distributor's sell-through — not to abandon
the portfolio.

**"Planners will not trust a number they cannot attribute."** The strongest objection, and it is
operational rather than statistical. Publish the components and the dispersion so the planner can see
which one moved. Teams that skip this find the portfolio overridden more often than the model it
replaced, and the overrides destroy the gain.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Selecting on the evaluation window | The chosen model changes every quarter and the winner is announced with the result | Selection over noise when components are close | Deployment of a model that was briefly lucky | Compare against the best ex-ante component; require stability across folds |
| Weights estimated and evaluated on the same data | The weight table and the accuracy table cite the same months | In-sample optimism | Improvement that does not exist out of sample | Disjoint windows; state both dates on the slide |
| Too many components for the window | Five models, 24 monthly errors, four-decimal weights | `n(n+1)/2` parameters from `T` observations | Weights dominated by sampling noise | Shrink, constrain, or reduce `n` |
| Component forecasts not archived | Only the consensus number exists in the warehouse | No counterfactual, no weights, no correlation | Combination cannot be evaluated or reproduced | Store every component per target per origin |
| Different populations per component | The model table has 300 series, the average 287 | Intersection not declared | Comparison void | Score on the intersection and report the exclusion count |
| Mixed forecast vintages | One component's history was refreshed last week | Look-ahead enters one component only | That component looks better for the wrong reason | Point-in-time snapshots; audit one origin by hand |
| Sign convention flipped | Weights are negative and large for the worst model | Vendor reports forecast minus actual | Portfolio inverts a component | One documented convention; unit test on a known series |
| Averaging a biased component | The portfolio carries a persistent 4% low bias | Variance weighting ignores mean error | Bias preserved and harder to see | Check mean error first; correct or exclude |
| Correlation assumed away | "The models are different, so they must be diverse" | No error correlation computed | Portfolio is one model with extra cost | Compute the pairwise matrix; report the mean |
| Robust rule used on a valuable outlier | The median underperforms the mean badly | Trimming discards a negatively correlated component | Lost cancellation | Use robust rules for suspect components, not unusual ones |
| Pooled weights across horizons | One weight vector applied at `h = 1` and `h = 12` | Component ranking flips with horizon | Contradictory recommendations averaged | Estimate per horizon, or show the ranking is stable |
| Portfolio never re-estimated | Weights date from the system implementation | Covariance structure drifted | Slow decay, invisible month to month | Scheduled re-estimation with a stability check |
| Portfolio re-estimated too often | Weights change every month and the number jumps | Chasing noise | Planner distrust and overrides | Fixed cadence; smooth or shrink changes |
| Combination claimed without a comparator | "The ensemble improved accuracy by 12%" with no single-model number | No benchmark in the sentence | Uninterpretable claim | Every accuracy slide carries the comparator row |
| Design declared after the results | The protocol document is dated after the leaderboard | Selection on the evaluation set | The result is a maximum, not an estimate | Pre-register; version and date the document |

Three deserve a note.

**Weights estimated and evaluated on the same data** is the most common technical error and the easiest
to make accidentally, because many tools report in-sample fit alongside the weights. A weight vector
optimal in sample is optimal by construction; the interesting question is always out of sample.

**Sign convention flipped** is rare, embarrassing, and invisible in a summary table. The symptom is a
large negative weight on the component you trust most. A unit test on a synthetic series prevents it.

**The planners will not trust it** is not a statistical failure, but it is the most common reason a
correct portfolio is abandoned. Publish the components, the dispersion, and a one-line explanation of
the month's move. A portfolio that cannot be explained will be overridden, and the override ships.

## 11. Assumptions and limits

Every combination carries assumptions that are rarely written down. The ledger below is the one to
copy into a methods note, with the last column filled in for your own data.

| # | Assumption | Why it is needed | What breaks if it fails | How to test |
|---:|---|---|---|---|
| 1 | Components forecast the same target with the same definition | Otherwise the average mixes incompatible quantities | The portfolio is a blend of eaches and cases, or of gross and net demand | Compare definition and unit of every component; score on the intersection |
| 2 | Every component has a forecast for every target and origin | A missing component silently reweights the rest | Weights change without anyone deciding to change them | Count components per target per origin; report coverage |
| 3 | Errors are unbiased, or bias is small relative to variance | Variance-based rules optimise variance only | A persistent bias passes into the portfolio and becomes harder to see | Report mean error per component before weighting |
| 4 | Error covariance is stationary over the weighting window and beyond | Weights transport from estimation to deployment | Estimated weights are stale on arrival; section 6 is a constructed case | Split-window test: estimate on early months, score on later ones |
| 5 | The weight window has enough observations per parameter | Estimation error is what makes the puzzle | Weights dominated by noise; equal weights win by default | Compare `T` with the parameter count; shrink or reduce `n` |
| 6 | Combining does not change the decision | Accuracy improvement is assumed to be the objective | A portfolio that is better on MSE is worse on cost if errors have asymmetric costs | Score combinations on the cost function, not only on MSE |
| 7 | Components are genuinely diverse in information and method | Diversity is the entire source of gain | Correlated components add cost and no accuracy | Compute the pairwise error correlation matrix |
| 8 | The evaluation window is independent of the design | Otherwise the reported gain is a maximum | Reported improvement is optimistic and does not replicate | Pre-register; hold out a window nobody has looked at |
| 9 | The portfolio can be explained to its user | Operational adoption | Planners override a number they cannot attribute, and the overrides destroy the gain | Track override rate and the value added by overrides |
| 10 | Compute and maintenance cost is lower than the accuracy gain | The trade-off that makes combination "nearly free" | Ten components with `n` times the run time and monitoring, for a gain inside noise | Cost the runs, the archive, and the monitoring against the measured gain |

Four of these deserve more than a table cell.

**Assumption 3 hides in plain sight.** Averaging preserves bias: if two components are both 4% low,
their average is 4% low, and the variance machinery has nothing to say about it. A portfolio cannot
correct a bias every member shares, which is exactly the situation after a regime change that no
component's features capture.

**Assumption 6 is the one practitioners violate without noticing.** Minimising mean squared error is
a choice, not a neutral default. If stock-out cost is five times holding cost, the optimal action is
not the mean forecast, and a portfolio better on MSE may be worse on profit.

**Assumption 7 is not testable in the abstract.** Two components can be structurally different — one
exponential smoothing, one gradient boosting — and produce errors correlated at 0.95 if both are driven
by the same recent demand signal. Only the error matrix answers the question.

**Assumption 10 is economic, not statistical.** Compute, archive rows, monitoring attention, and
explanation effort are real costs, small relative to the gains typical in the literature but not
negligible: at 200,000 series with 10 components, a 2% gain inside the noise band does not pay for them.

> **Note:** An assumption that cannot be tested is not an assumption; it is a belief. Each row above
> has a test column for that reason. If you cannot run the test, say so in the methods note rather
> than omitting the row.

## 12. Exercises

Work the calculation exercises with a pencil before checking section 13. Design exercises have rubrics
rather than answers, because a defensible design depends on constraints the exercise does not supply.

**Exercise 1 — Variance of an average from two components.** Two components have RMSE 5 and 6 eaches,
and their errors correlate at 0.5. Compute the standard error of the equal-weight average, the optimal
weight on the first component, and the variance at that weight. State the percentage reduction against
equal weights, and say what its small size implies for a practitioner with 24 observations.
**Deliverable:** a four-line calculation and a two-sentence interpretation.
**Acceptance criteria:** both variances correct to two decimals; the weight correct to three decimals;
the interpretation names estimation error as the reason the gain is hard to capture.
**Time:** 20 minutes.

**Exercise 2 — Three robust rules on five components.** Five component forecasts for one month have
errors of 4, −3, 6, −10, and 2 eaches. Actual demand was 100 eaches. Compute the equal-weight mean,
the median, and the trimmed mean with one observation trimmed from each tail. Report each as a
forecast in eaches and as an error.
**Deliverable:** a small table of forecast, error, and absolute error for three rules.
**Acceptance criteria:** all three forecasts correct; the trimmed mean shows its excluded values; a
one-line statement of which rule protected against the outlier and at what cost.
**Time:** 15 minutes.

**Exercise 3 — Weights that do not transport.** Using the Larkfield data in section 6, estimate the
two-component optimal weight on D over months 1 to 12 with B as the other component. Apply that
weight to months 13 to 24 and compute the resulting MSE. Compare it with the MSE of the equal-weight
B and D average over the same months.
**Deliverable:** the window-1 covariance, the weight, and the two MSEs.
**Acceptance criteria:** covariance within 1.0 of the value implied by the error table; weight within
0.01 of 0.740; the comparison states which window each number comes from.
**Time:** 30 minutes.

**Exercise 4 — The unqualified improvement claim.** A slide reads: "The ensemble improved accuracy by
14%." Identify the four pieces of information missing from the sentence, and rewrite it as a claim
that could survive review.
**Deliverable:** a list of four gaps and one rewritten sentence of no more than 35 words.
**Acceptance criteria:** the rewrite names the metric, the unit, the comparator, the window, and the
component set; no adjective in the rewrite is unquantified.
**Time:** 20 minutes.

**Exercise 5 — Six weights from twenty-four observations.** A weight table lists six weights to four
decimal places, estimated from 24 monthly errors, with no constraint and no shrinkage. Diagnose the
problem quantitatively, name the parameter count, and propose two fixes with the arithmetic of one of
them.
**Deliverable:** a short memorandum of 200 to 250 words.
**Acceptance criteria:** correct parameter count for the full-covariance case; at least two distinct
fixes; one fix worked numerically with a stated shrinkage constant.
**Time:** 30 minutes.

**Exercise 6 — Pre-register a combination evaluation.** Design the evaluation protocol for a
six-component portfolio over 36 months of one-step-ahead errors. You choose the windows, the metric,
the comparators, the pass rule, and the decision the result will trigger.
**Deliverable:** a one-page protocol with six labelled fields, dated, plus a one-paragraph justification
of the pass rule.
**Acceptance criteria:** every field filled before any number is reported; comparators include an
equal-weight average and the best ex-ante component; the pass rule states a threshold and a decision;
the protocol names who may amend it and how amendments are recorded.
**Time:** 60 minutes.

**Exercise 7 — The pass rule arithmetic.** An equal-weight portfolio scores MSE 88 on a held-out
window. The incumbent single model scores 100. Your approved pass rule requires a 15% MSE improvement
before migration is authorised. Decide whether the portfolio passes, show the arithmetic, and state
the minimum MSE that would have passed.
**Deliverable:** three numbers and a one-line decision.
**Acceptance criteria:** the threshold MSE and the realised improvement both shown; the decision follows
from the arithmetic; the migration cost is named as the reason the threshold is not zero.
**Time:** 15 minutes.

**Exercise 8 — Diversity audit.** You are given five components with the following mean pairwise error
correlations: A–B 0.91, A–C 0.88, A–D 0.12, A–E 0.09, and all other pairs between 0.80 and 0.94. The
team proposes dropping D and E because their individual accuracy is the worst. Evaluate the proposal,
and recommend one additional information source to improve the portfolio.
**Deliverable:** a 250-word recommendation with a revised component set.
**Acceptance criteria:** the recommendation uses the correlation structure rather than individual
accuracy as its criterion; it names the information the set lacks; it states what would change it.
**Time:** 30 minutes.

**Exercise 9 — Three kinds of combination.** Classify each of the following as combining forecasts,
combining models, combining across horizons, or combining across levels, and state one risk specific
to each: (a) averaging the statistical forecast and the sales team's forecast; (b) averaging the
outputs of two model specifications of the same method family; (c) averaging the one-month-ahead and
three-month-ahead projections made at the same origin; (d) averaging the SKU-level portfolio and a
product-family forecast scaled to SKU.
**Deliverable:** a four-row table with classification, one-line justification, and one specific risk.
**Acceptance criteria:** all four classified correctly; each risk is distinct; no risk is a restatement
of the definition.
**Time:** 20 minutes.

## 13. Self-check answers

**Exercise 1.** Equal-weight variance is `0.25 × 25 + 0.25 × 36 + 0.5 × 15 = 22.75` eaches squared, a
standard error of `√22.75 = 4.77` eaches. With covariance 15.00, `w₁* = (36 − 15) / 31 = 0.677`, and
the variance at the optimum is `(25 × 36 × 0.75) / 31 = 21.77`, or 4.67 eaches — a reduction of 4.3%.
The prize is small because the errors are only half correlated and the variances are similar, so the
estimation error in `w*` on 24 observations is larger than the prize. Equal weights are not a
compromise; they are the funded choice.

**Exercise 2.** The errors sum to −1, so the mean error is −0.2 and the mean forecast is 100.2 eaches.
Sorted, they are −10, −3, 2, 4, 6, so the median forecast is 98. Trimming one from each tail leaves
−3, 2, 4, a mean of 1.0, so the trimmed forecast is 99. The outlier was the −10, and here it was
informative rather than wrong: the mean, which absorbed part of it, is closest to actual. Robust rules
cost accuracy in this month, which is the trade-off in section 6.5, and it turns on whether the
unusual component was a mistake or a signal.

**Exercise 3.** Over months 1 to 12, `ΣB² = 1,696` and `ΣD² = 296`, so `s_B² = 141.33` and
`s_D² = 24.67`; `ΣBD = −460` gives a covariance of −38.33. Then
`w_D = (141.33 + 38.33) / (141.33 + 24.67 + 76.67) = 0.740`. Applied to months 13 to 24 it gives MSE
72.7 eaches squared, against 9.64 for the equal-weight B and D average. The weight was a good estimate
of a structure that did not persist.

**Exercise 4.** The four gaps are: the metric (accuracy is not a quantity), the unit (eaches,
dollars, percentage of volume), the comparator (14% against which other forecast — a naive baseline, a
selected model, or the incumbent process), and the window and component set (over what months, with
which models, at which horizon). A rewrite that survives review: "Over months 13 to 24, one month
ahead, the equal-weight average of four components achieved mean squared error 21.2 eaches squared
against 96.7 for the best component and 219.3 for the ex-ante selected model." Any shorter sentence is
hiding something.

**Exercise 5.** Six components require 21 parameters in the full-covariance case, from 24 observations
— fewer than one per parameter, with nothing left for validation. Four decimal places are spurious:
the standard error of a weight is likely above 0.1. Fix 1: shrink, `w = 0.5 × estimated + 0.5 × (1/6)`,
so an estimated 0.30 becomes 0.233. Fix 2: cut to three diverse components, five parameters. A third is
the diagonal inverse-variance rule alone, six parameters. Keeping 21 parameters is not a fix.

**Exercise 6.** Rubric, not an answer. A protocol passes when all six fields are present and dated:
component set and version, weighting rule, estimation window, evaluation window, metric and unit,
comparators, pass rule and the decision it triggers. Deduct for a pass rule with no decision attached,
for a comparator that omits the equal-weight benchmark, for a metric without units, and for any field
that could be filled in after the results are known. Full marks require an amendment clause — a
protocol that cannot record a change will be amended silently.

**Exercise 7.** The required MSE is `100 × (1 − 0.15) = 85.00`. The realised improvement is
`(100 − 88) / 100 = 12.0%`, and 88.00 is above 85.00, so the portfolio fails the pass rule. The
decision is not to migrate. The minimum passing MSE is 85.00 exactly, which the portfolio misses by 3.00
eaches squared. The threshold is not zero because migration has its own costs — retraining, archive
changes, planner adjustment — and a gain that does not cover them is not a gain.

**Exercise 8.** Rubric. The proposal inverts the evidence: D and E are the only components whose errors
are close to independent of A, so they are the source of the portfolio's gain, while the four
correlated above 0.80 are near-duplicates. A strong answer keeps D and E, drops at least two of the
correlated group, and notes that individual accuracy is the wrong selection criterion. Defensible added
sources include customer point-of-sale data, a distributor inventory series, or a forward-looking
promotion plan — each brings information the current set lacks. Weak answers add another model on the
same history, which would enter at a correlation above 0.9.

**Exercise 9.** (a) Combining forecasts across method types — risk: the judgemental member is not
archived and drifts, so its weight is never estimable. (b) Combining models within one family — risk:
near-duplicates, so the gain is close to zero and the cost is double. (c) Combining across horizons —
risk: weights are horizon-specific, and a pooled vector averages contradictory recommendations.
(d) Combining across levels — risk: a scaled family forecast and the sum of SKU forecasts are
inconsistent, so reconciliation is needed afterwards.

## 14. Interview questions

**1. What is the forecast combination puzzle, and why does it exist?**
**Strong:** States that equal weights frequently beat estimated optimal weights out of sample, names
the mechanism — weight estimation error from short windows — and adds that the gain is itself small
when variances are similar and errors positively correlated, so a small error is enough to lose it.
**Weak:** Describes combination as "averaging several models" and treats the puzzle as a curiosity
rather than a routine result.

**2. Two components have RMSE 10 and 10, and their errors are uncorrelated. What is the RMSE of the
equal-weight average?**
**Strong:** 7.07, because the variance halves; states the formula `σ/√2` and notes that the result
holds only if the covariance is genuinely zero.
**Weak:** Says "about 10" or averages the RMSEs to get 10, confusing the average of standard
deviations with the standard deviation of the average.

**3. Why can adding a worse model improve a portfolio?**
**Strong:** Because the criterion is the variance of the sum, not the accuracy of the members; a worse
model with negatively correlated errors can reduce portfolio variance, and a better model with
correlated errors can add nothing.
**Weak:** Treats accuracy as a sufficient statistic and concludes that a worse model cannot help.

**4. When would you use a median rather than a mean?**
**Strong:** When at least one component may be wrong rather than merely unusual — a broken data feed,
a stale model, a judgemental input with an error — and when `n` is large enough that trimming leaves a
stable estimate. Notes that with four components the median and a 20% trimmed mean coincide.
**Weak:** Uses the median whenever there is an outlier, without asking whether the outlier carried
information.

**5. How would you set weights with 24 monthly observations and six components?**
**Strong:** Refuses the full-covariance optimum, quantifies why (21 parameters from 24 observations),
proposes shrinking toward equal weights or a reduced component set, and states the shrinkage constant
and its effect.
**Weak:** Says "use inverse-variance weighting" without noticing that 21 parameters are being
estimated from 24 points.

**6. What is the difference between combining forecasts and combining models?**
**Strong:** Distinguishes objects: forecasts from different method families or information sources,
versus specifications within one family. Notes that the first buys diversity and the second usually buys
compute cost, and that the two are often conflated in vendor material.
**Weak:** Uses the terms interchangeably.

**7. How does the horizon affect combination?**
**Strong:** Component ranking changes with horizon and a longer horizon leaves more room for components
to disagree, so estimate weights per horizon or demonstrate that a pooled vector is stable.
**Weak:** Assumes the same weights apply at one month and twelve months.

**8. A vendor says their ensemble improves accuracy by 20%. What do you ask?**
**Strong:** Metric, unit, comparator, window, horizon, component set, and — the question that usually
ends the conversation — whether the 20% is measured against a naive baseline or against the customer's
incumbent process.
**Weak:** Accepts the number and asks about price.

**9. Your portfolio beats every component out of sample but planners override it. What do you do?**
**Strong:** Treats it as an adoption problem: publish the components and dispersion, explain the month's
move in one line, and measure the value added by the overrides. One that never adds value is a training
problem; one that sometimes does points to a missing component.
**Weak:** Escalates for a mandate to stop overrides.

**10. Why is it important to declare the evaluation design in advance?**
**Strong:** Because a selection competition and a combination benchmark are both legitimate, and the
difference between them is which one was fixed before the numbers arrived; post-hoc choice of metric,
window, or comparator turns an estimate into a maximum.
**Weak:** Says pre-registration is a compliance requirement, without explaining what it protects.

**11. What does an average of twenty highly correlated models buy you?**
**Strong:** Almost nothing in accuracy — the variance reduction is bounded by the correlation — and it
costs twenty times the compute; the fix is to add an information source, not another specification.
**Weak:** Assumes more models always means more diversity.

## 15. Key terms

**Forecast combination** — producing a single forecast from two or more component forecasts; distinct from selecting one component.
**Combination puzzle** — the empirical regularity that equal weights often beat weights estimated to be optimal.
**Equal weights** — the arithmetic mean with weight `1/n` on each component; the default benchmark.
**Optimal weight** — the weight minimising forecast error variance given a covariance matrix; a population quantity, never observed.
**Inverse-variance weight** — `w_i ∝ 1/s_i²`; the optimum when components are uncorrelated, and a common misapplication when they are not.
**Full-covariance weight** — the optimum using all pairwise covariances; `n(n+1)/2` parameters.
**Regression-based weight** — weights estimated by regressing the actual on the components; equivalent to the optimal rule in the population.
**Shrinkage** — pulling estimated weights toward equal, `w = (1−k)ŵ + k(1/n)`, trading bias for variance.
**Constrained weight** — weights restricted to be non-negative and to sum to one; excludes extreme estimates.
**Median combination** — taking the middle component forecast; a robust rule that discards all but one value.
**Trimmed mean** — the mean after removing a proportion `λ` from each tail; `λ = 0` is the mean and `λ = 0.5` is the median.
**Winsorised mean** — the mean after replacing tail values rather than removing them; retains `n`.
**Error correlation** — the correlation between component errors; the quantity that determines what combination can buy.
**Diversity** — the degree to which components carry different information; evidenced by low error correlation, not by methodological difference.
**Estimation error** — the difference between an estimated weight and its population value; the mainstream explanation for the puzzle.
**Winner's curse** — the tendency of a selection rule to favour a component that was lucky in the evaluation window.
**Selection bias** — the optimistic error in a reported result caused by choosing the design after seeing the data.
**Ex ante** — chosen with information available at the forecast origin; the only kind of choice a production system can make.
**Ex post** — chosen with hindsight over the evaluation window; an upper bound, never a benchmark.
**Encompassing** — the property that a combination's forecast contains all the information in a component; an alternative to accuracy comparison.
**Forecast value added (FVA)** — the accuracy gain of a process step over a benchmark; the measure that makes combination comparable to doing nothing.
**Combining across levels** — averaging forecasts made at different positions in a hierarchy, such as item and family.
**Combining across horizons** — averaging forecasts for different lead times, or using horizon-specific weight vectors.

## 16. Sources

**Peer-reviewed and academic (Grade A)**

- Bates, J. M., and C. W. J. Granger. "The Combination of Forecasts." *Operational Research Quarterly* 20, no. 4 (1969): 451–468. The origin of the field, including the diversification argument and the inverse-variance rule.
- Makridakis, S., and M. Hibon. "The M3-Competition: Results, Conclusions and Implications." *International Journal of Forecasting* 16, no. 4 (2000): 451–476.
- Makridakis, S., E. Spiliotis, and V. Assimakopoulos. "The M4 Competition: 100,000 Time Series and 61 Forecasting Methods." *International Journal of Forecasting* 36, no. 1 (2020): 54–74.
- Makridakis, S., E. Spiliotis, and V. Assimakopoulos. "M5 Accuracy Competition: Results, Findings, and Conclusions." *International Journal of Forecasting* 38, no. 4 (2022): 1346–1364. Equal-weight blending of 220 models; reconciliation of top-down and bottom-up.
- Timmermann, A. "Forecast Combinations." In *Handbook of Economic Forecasting*, vol. 1, edited by G. Elliott, C. W. J. Granger, and A. Timmermann, 135–196. Elsevier, 2006. Survey with five numbered conclusions.
- Smith, J., and K. F. Wallis. "A Simple Explanation of the Forecast Combination Puzzle." *Oxford Bulletin of Economics and Statistics* 71, no. 3 (2009): 331–355.
- Claeskens, G., J. R. Magnus, A. L. Vasnev, and W. Wang. "The Forecast Combination Puzzle: A Simple Theoretical Explanation." *International Journal of Forecasting* 32, no. 3 (2016): 754–762.
- Batchelor, R., and P. Dua. "Forecast Combination and the Benefit of Diversification." *Journal of Forecasting* 14, no. 4 (1995): 315–331.
- Jose, V. R. R., and R. L. Winkler. "Simple Robust Averages of Forecasts: Some Empirical Results." *International Journal of Forecasting* 24, no. 1 (2008): 163–169.

**Regulatory and official (Grade B)**

- ISO 3534-1, *Statistics — Vocabulary and Symbols*, Part 1: General statistical terms and terms used in probability. Definitions of error, bias, and variance.
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM), 3rd edition. "Accuracy" defined as a qualitative concept with no numerical value.
- ISO 5725-1, *Accuracy (Trueness and Precision) of Measurement Methods and Results*, Part 1: General principles and definitions.
- ISO 8601, *Date and Time — Representations for Information Interchange*. Calendar and duration formats used in forecast archives.
- Board of Governors of the Federal Reserve System, *Supervisory Guidance on Model Risk Management*, SR 11-7, 2011. Validation and monitoring practice, applied by analogy.
- Sarbanes-Oxley Act of 2002, Section 404, and the Public Company Accounting Oversight Board (PCAOB) auditing standards on internal control over financial reporting.
- IFRS 15 / ASC 606, *Revenue from Contracts with Customers*. Governs revenue recognition, not forecast method.

**Company, working paper, and reference (Grade C)**

- Elliott, G., and A. Liao. "Forecasting with In- and Out-of-Sample Regressors." Working paper. Argues that the population gain from optimal weights is small under plausible conditions, and proposes subset averaging.
- 2025 preprint on ensemble forecasting of infectious disease, reported at aggregate level for COVID-19, influenza, and respiratory syncytial virus (RSV). Cited for the error-correlation argument only; not peer reviewed, and an epidemiological rather than commercial setting.
- ASCM/APICS *Dictionary*, current edition. Supply chain terminology, including forecast, plan, and consensus demand.

> **Note:** No source in this list reports a distributor-level study of the forecast combination puzzle. The mechanisms transfer; the magnitudes do not. Treat every percentage in section 9 as a prior about method families, not as an expected gain.

## 17. Links

**This module**

- [Module 07 index](_index.md) — topic table, prerequisites, and status.
- [Forecastability and Benchmarks First](01_forecastability-benchmarks.md) — the naive baselines a portfolio must still beat.
- [Exponential Smoothing](02_exponential-smoothing.md) — component A in the worked example.
- [ARIMA, SARIMA, and SARIMAX](03_arima-sarimax.md) — component B.
- [Regression with Lags, Fourier Terms, and Exogenous Variables](04_regression-lags-fourier.md) — component C.
- [Hierarchy and Aggregation](05_hierarchy-aggregation.md) — combining across levels, and why aggregation changes error.
- [Intermittent Demand](06_intermittent-demand.md) — where combination behaves differently, because errors are asymmetric at low counts.
- [Rolling-Origin Backtesting](08_rolling-origin-backtesting.md) — the machinery a combination evaluation shares with a single-model evaluation.
- [`09_error-metrics-cost.md`](09_error-metrics-cost.md) — mapping the metric to the cost of the decision.
- [`10_overrides-fva.md`](10_overrides-fva.md) — measuring whether the override beat the portfolio.

**Around the programme**

- [Module 08 index](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md) — reconciliation, and probabilistic combination.
- [Part 02 index](../_index.md) — where this module sits in the methods sequence.
- [Part 01 index](../../01_foundations/_index.md).
- [Forecast error metrics](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md) — the metrics every table in this topic depends on.
- [Module 07 survey file](../../../course/07_time_series_forecasting.md) — the narrative arc in one pass.
