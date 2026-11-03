# Conformal Prediction for Time Series

> Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 01A topics 10–11; Module 04 topic 02; Module 07 topics 08 and 09; Module 08 topics 06–07
> Version 2026.5

*Distribution-free coverage under dependence, and its limits.*

---

## 1. Why this topic exists

Conformal prediction is the most attractive guarantee in applied statistics, and for a reason that has
nothing to do with fashion. It says: if you can produce a number that scores how badly a prediction
missed, you can turn that score into an interval that covers the truth at least a stated fraction of
the time, in finite samples, without assuming a distribution. No normality, no likelihood, no variance
estimate, no asymptotics — ten calibration points and a sorting operation.

For supply chain forecasting this arrives as a gift, because the distributional assumptions that
textbook intervals need are exactly the ones demand data refuses to satisfy. Demand is intermittent,
skewed, bounded below at zero, spiked by promotions, censored by stock-outs, and heteroscedastic
across the year. A normality-based interval on a slow-moving spare part is not approximately right;
it is wrong in a direction nobody can predict without looking. Conformal needs none of that.

The gift comes with a condition printed in small type, and the small type is the whole subject of
this topic. The coverage statement holds when the calibration data and the point you are predicting
are **exchangeable** — when the joint distribution of the sample is unchanged by reordering it. A
demand series is one realisation of a dependent process observed in one order. The order carries
information, and a guarantee that averages over all orderings is a guarantee about a population your
series is not drawn from.

Three failures recur in practice, and each is expensive in a different way.

**One. The interval that was right on average and wrong when it mattered.** A team calibrates on
twenty weeks of a flat spring, deploys into the autumn peak, and reports 90.0% coverage for the
quarter. The number is correct. It is also an average in which the flat weeks covered at 100% and
the four peak weeks covered at 50%. The planner who consumed the interval during the peak
experienced something closer to a coin flip, and the dashboard that reported 90.0% made that
experience invisible. Section 6 works this case with arithmetic.

**Two. The calibration set that had already seen the future.** Split conformal requires a block of
data the model did not fit on. Teams under time pressure calibrate on the residual history that the
model was trained on, or on a block that overlaps the evaluation block, or on data revised after the
forecast was made. Each of these restores the appearance of a guarantee and removes the substance.
The interval narrows, coverage looks excellent in the backtest, and the first live quarter is a
surprise.

**Three. The horizon that was never calibrated.** A model is calibrated on one-week-ahead residuals
and used to set safety stock for a four-week lead time. The four-week error distribution is wider,
and for a seasonal product it is differently shaped. The conformal guarantee is a statement about
the score you computed, at the horizon you computed it, on the distribution you computed it on. A
one-step calibration transferred to a four-step decision is not a weaker version of the guarantee.
It is a different quantity wearing the same name.

**Who is blocked.** Three roles need different things from the same machinery. The inventory planner
needs a quantile of demand over the lead time, in the tail rather than on average. The sales and
operations (S&OP) lead needs a range to put in front of a consensus meeting that widens honestly when
the forecast is bad. The finance partner needs a downside case that survives an audit, meaning a
method describable in one paragraph and reproducible from a stored artefact. Conformal can serve all
three, but none of them if the coverage quoted is marginal and the decision is conditional.

**What this topic is not.** It is not a library tutorial, and it is not an argument that conformal
beats quantile regression or loses to it. It is an argument that the guarantee is precise, narrow, and
conditional on an assumption that time series violate, and that the professional skill is knowing
which repair restores enough of the assumption for the decision at hand.

Topics 07 and 09 of this module supply the surrounding machinery — predictive distributions and
calibration in 07, scoring rules in 09 — and this file does not re-teach them. Module 07 topic 08
supplies rolling-origin backtesting, the only honest way to test whether a conformal guarantee
survives deployment, and it is assumed here.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the split conformal coverage guarantee precisely, including the quantile index, the
   feasibility condition on the calibration-set size, and the two-sided coverage band, and
   **derive** the index rather than quoting it.
2. **Explain** what exchangeability buys, in what sense the guarantee is distribution-free, and why
   a demand series violates the assumption through drift, seasonality, autocorrelation, and regime
   change.
3. **Compute** a conformal interval by hand on a ten-point calibration set, and **identify** the
   three ways the arithmetic is commonly got wrong.
4. **Distinguish** marginal from conditional coverage, and **argue** why an operations decision
   usually needs the conditional statement that no distribution-free method can supply.
5. **Choose** among weighted, adaptive, ensemble, block, and horizon-specific conformal variants for
   a stated data situation, and **state** what each trades away.
6. **Diagnose** a coverage failure in production using the probability integral transform (PIT),
   coverage curves, and interval-width reports, and **attribute** it to a broken assumption.
7. **Compare** a conformal interval with a quantile-regression interval and with an
   empirical or simulation interval on the same data, and **justify** a choice.
8. **Write** a one-page interval specification that states the score, the calibration window, the
   horizon, the coverage target, the diagnostic, and the condition under which the interval should
   not be trusted.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three objects: a split on a timeline, a sorted list of scores, and a seam.

### 3.1 The machine, drawn once

```text
  SPLIT CONFORMAL, DRAWN ONCE, READ LEFT TO RIGHT
  ===============================================

  TIME  ----------------------------------------------------------->

  |<---- proper training ---->|<-- calibration -->|<---- test ----|
  week 1                    70 71                80 81           100
        fit f(x)                 score s_i            deploy C(x)
        freeze it  ------------->  no refit  -------->  no refit

  STEP 2  SCORE                    STEP 3  RANK
  +---------------------+          +------------------------------+
  | s_i = | y_i - f(x_i)|          | sort ascending               |
  | ten values, weeks   |  ----->  | s_(1) <= ... <= s_(10)       |
  | 71 to 80, units     |          | k = ceil((n+1)(1-alpha))     |
  +---------------------+          |   = ceil(11 * 0.90) = 10     |
                                   | q = s_(k) = s_(10) = 61 units|
                                   +---------------+--------------+
                                                   |
  STEP 4  FORM THE INTERVAL                        v
  +----------------------------------------------------------+
  |  C(x) = [ f(x) - q , f(x) + q ]                          |
  |  at f(x) = 240 units:  [ 179 , 301 ]  units              |
  +----------------------------------------------------------+

  ................... EXCHANGEABILITY SEAM ...................
  Everything above is valid only if the ten calibration scores
  and the next test score are exchangeable: their joint law is
  unchanged by any permutation.

     exchangeable     ( s_1 ... s_10 , s_test )  ~  any order
     time series      ( s_1 ... s_10 , s_test )  ~  THIS order
                                                   only

  The arrow of time is the one permutation the series will not
  let you perform. Section 5 is a sequence of attempts to
  restore a weaker version of the seam, or to admit it is gone.
```

Redraw that from memory before reading on. The four boxes are the whole algorithm. The seam is the
whole difficulty.

### 3.2 Why the order statistic is the right object

The argument is short and worth internalising, because it is the only piece of theory that
distinguishes conformal from a heuristic interval.

Assume the `n` calibration scores and the one test score are exchangeable. Then their relative
ordering is uniform: each of the `n + 1` scores is equally likely to be the largest, and by symmetry
the test score's rank among the `n + 1` is uniform on `{1, ..., n + 1}`. The interval covers the truth
exactly when the test score is no larger than the threshold `q`. If `q` is the `k`-th smallest
calibration score, the test score falls at or below it whenever its rank is at most `k`, which
happens with probability `k / (n + 1)`.

Set that to at least `1 - alpha` and solve:

```text
   k / (n + 1) >= 1 - alpha
   k >= (n + 1)(1 - alpha)
   k = ceil( (n + 1)(1 - alpha) )        smallest integer satisfying it
```

That is the whole derivation. Two consequences follow immediately and both are operational.

**The guarantee is exactly calibrated from above.** When scores are distinct, coverage is at most
`1 - alpha + 1/(n + 1)`. With ten calibration points, a nominal 90% interval covers somewhere
between 90.0% and 99.1%. The band is not a defect; it is the honest resolution of a ten-point
sample, and it is wide.

**The guarantee can be vacuous.** If `k > n` there is no calibration score large enough, and the
valid interval is the entire real line. Solving `k <= n` gives the feasibility condition
`n >= 1/alpha - 1`. A 90% interval needs nine calibration points. A 95% interval needs nineteen. A
99% interval needs ninety-nine. Teams routinely ask for 95% intervals from a dozen calibration
weeks and receive, from a well-behaved library, the maximum residual and a guarantee that is quietly
false.

### 3.3 Marginal is an average over test points

The probability in the coverage statement is over the draw of the calibration set **and** the draw
of the test point. It is not a statement about any particular `x`.

```text
  COVERAGE AS A MARGINAL AVERAGE, AND WHAT IT HIDES
  =================================================

  empirical coverage of a 90% conformal interval, by week

  1.00 |  x  x  x  x  x  x  x  x  x  x        <- plateau weeks
  0.90 |- - - - - - - - - - - - - - - - - -  nominal, marginal mean
  0.67 |                          x  x
  0.50 |                             x     x  <- ramp weeks
       +--|--|--|--|--|--|--|--|--|--|--|--> week
         1  4  7 10 13 16 19 22 25 28 31

  mean over all 20 test weeks = 18/20 = 0.900   exactly nominal
  mean over weeks 17 to 20    =  2/4  = 0.500   the planner's experience
```

Read the two summary lines together. A dashboard that prints one coverage number prints the first.
An operations team lives inside the second. This is not specific to conformal: every marginal
guarantee has the same shape. What is specific is that conformal is often sold as if the marginal
statement were conditional, and the pitch is seductive precisely because the marginal statement is
genuinely assumption-light.

### 3.4 What "distribution-free" actually buys

The phrase is load-bearing and usually misread. It does not mean assumption-free. It means coverage
does not depend on the conditional distribution of the response given the features. You may use a
nonsense model, a badly specified quantile regression, or a neural network trained on the wrong
objective, and still get coverage.

Three things it buys, each real: robustness to the shape of the error distribution (no normality, no
symmetry, no variance estimate, no homoscedasticity); finite-sample validity (`n = 10`, not `n` to
infinity); and model-agnosticism (the score wraps any point predictor, including one you cannot
explain).

Three things it does not buy, each routinely assumed: validity under dependence, because the seam is
an assumption and not a theorem about your data; conditional coverage, which is provably impossible
in general; and protection from a biased point forecast. Conformal recentres an interval on whatever
the model said, so if the model is biased, coverage can stay marginal-correct while the width grows
to absorb the bias. A conformal interval around a biased forecast is an honest statement about a bad
forecast, not a repair.

> **Note:** The guarantee also says nothing about the *shape* of the interval. A constant-width band
> around a heteroscedastic series is valid and useless. Conformalized quantile regression exists to
> fix exactly this.

### 3.5 Why conditional coverage is not available

Suppose you want coverage `1 - alpha` at every value of `x`, not on average. Lei and Wasserman (2014)
showed that any method with exact finite-sample conditional validity must produce prediction bands of
infinite width at non-atomic points, with probability one. Barber, Candès, Ramdas and Tibshirani
(2021) extended the result: even an approximate, subgroup-relaxed version cannot beat the trivial
solution of a wide marginal interval, except under restrictions on the class of sets considered.

The practical reading is blunt. You cannot buy conditional coverage with a cleverer score. You can
buy **local adaptivity** — intervals narrower where the model is precise and wider where it is not —
and local adaptivity is what people actually want when they ask for conditional coverage. It is a
different property, and it is achievable. Three routes, in ascending order of assumption:

| Route | Mechanism | What it assumes | Cost |
|---|---|---|---|
| Normalised score | Divide the residual by a predicted scale | The scale model is roughly right | A second model to fit and validate |
| Conformalized quantile regression | Conformalise two conditional quantiles | Quantile estimates are approximately ordered and roughly calibrated | Quantile loss training; crossing problems |
| Mondrian / group-conditional | Separate calibration per group | Groups are exchangeable internally | Groups must be defined in advance; small groups break feasibility |

All three improve the interval where it matters and none delivers conditional coverage. Say so in
the specification, because the alternative is that a reviewer assumes you claimed more than you did.

### 3.6 Where the time series breaks the seam

Four mechanisms, in increasing order of how badly they break the guarantee.

**Drift.** The level moves. A calibration window ending at week 80 describes a mean of 240 units; by
week 95 the mean is 300. Scores computed at week 80 are too small for week 95 by the drift
accumulated over fifteen weeks. Coverage decays smoothly and is easy to miss.

**Seasonality.** The score distribution is not stationary within the year. December residuals for a
toy retailer are larger and more skewed than April residuals. A calibration window that misses
December produces an interval that is too narrow every December, and an annual coverage report hides
it because the other forty-eight weeks over-cover.

**Autocorrelation.** Consecutive residuals are correlated. This does not by itself break marginal
coverage for a split-conformal interval on a stationary block: the scores remain identically
distributed and dependence affects the variance of empirical coverage, not its mean. What it breaks is
the *effective sample size*. Twenty highly correlated calibration scores carry the information of
five. The interval is valid and much noisier than `n = 20` suggests.

**Regime change.** A promotion calendar, a competitor exit, a network redesign, a new channel. The
post-change distribution is not a reordering of the pre-change one, and no weighting scheme recovers
what was never in the sample. This is where conformal fails outright and the honest answer is "the
interval is not valid, and here is the break date."

### 3.7 The horizon gap, which is the crux

A fifth mechanism is specific to forecasting rather than regression, and is the one most often missed
because it does not look like a distributional violation at all.

Split conformal calibrated on one-step residuals and applied at a four-step horizon has no guarantee,
even on a perfectly stationary series. The one-step and four-step residuals are different random
variables. For a random walk with innovation standard deviation `sigma`, the one-step error has
standard deviation `sigma` and the four-step error `2 sigma`. The calibration set contains no
information about the second quantity. The same argument applies to any horizon beyond the calibration
window in the temporal sense: a window that ends today says nothing about a forecast whose outcome
falls twelve weeks from today, because the process can move in ways the window never observed.

The repair is not statistical. Calibrate **at the horizon you will forecast**, on residuals from folds
whose outcome fell at that horizon. For an `H`-step decision that means `H` calibration sets, each
shorter than the `h = 1` set, each with a wider feasibility condition. This is where conformal and
rolling-origin backtesting meet, and why the prerequisite matters.

## 4. Definitions and vocabulary

The fourth column names the unit or coding convention. A term without one will be used two ways in
the same meeting.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Conformal prediction | A framework producing prediction sets with finite-sample coverage from a nonconformity score and an exchangeability assumption | Used to mean "any prediction interval" | Method family, no unit |
| Exchangeability | A joint distribution unchanged by permutation of its arguments | Confused with independence; independence implies it, not the reverse | Boolean assumption |
| Non-exchangeability | Any departure from exchangeability, including drift, seasonality, and dependence | Asserted from a time plot without a test | Boolean, with a named mechanism |
| Nonconformity score | A scalar measuring how badly a candidate prediction fits an observation; larger means worse | Called "the residual" when it is not one | Units of the response, or dimensionless if normalised |
| Conformity score | The negated nonconformity score, used in some implementations | Mixed with nonconformity in one codebase | Sign convention, documented |
| Calibration set | Held-out observations used only to compute scores and the quantile | Any data the model did not fit on, including data used for tuning | Count `n`, with a date range |
| Proper training set | Data used to fit the point predictor | Confused with the calibration set | Count, with a date range |
| Split conformal | Conformal using a single fixed split into proper training and calibration | Called "inductive conformal" in older literature | Method |
| Full conformal | Conformal refitting the model for every candidate response value | Assumed computationally impossible at scale | Method; cost `O(n)` refits |
| Jackknife+ | A leave-one-out conformal variant with a finite-sample guarantee | Used without noting it needs `n` refits | Method; cost `O(n)` refits |
| Quantile index `k` | `ceil((n + 1)(1 - alpha))`, the order statistic used as the threshold | Computed as `ceil(n(1 - alpha))`, which breaks the guarantee | Integer, `1 <= k <= n + 1` |
| Order statistic | The `k`-th smallest value of a sorted sample | Confused with an interpolated quantile | Units of the score |
| Empirical quantile | An interpolated estimate of a distributional quantile | Substituted for the order statistic | Units of the score |
| Marginal coverage | Probability of coverage averaged over calibration sets and test points | Reported as if it were per-prediction | Proportion in `[0, 1]` |
| Conditional coverage | Coverage at a fixed feature value, `P(Y in C(x) | X = x)` | Assumed to follow from marginal coverage | Proportion; unattainable distribution-free |
| Local validity | Coverage within a pre-specified partition cell or group | Sold as conditional coverage | Proportion per cell |
| Coverage band | The interval `[1 - alpha, 1 - alpha + 1/(n + 1)]` attainable at finite `n` | Reported as a point estimate | Proportion range |
| Undercoverage | Empirical coverage below nominal | Diagnosed as a bug when it is a broken assumption | Proportion gap |
| Interval width | Upper minus lower endpoint, at a stated `x` | Compared across methods with different coverage | Units of the response |
| Adaptive interval | An interval whose width varies with `x` | Confused with a conformal guarantee | Units, varying by `x` |
| Conformalized quantile regression (CQR) | Conformalising a pair of conditional quantile estimates | Described as "conformal plus quantile regression" without the ordering repair | Method |
| Pinball loss | The loss whose minimiser is a quantile | Confused with absolute error | Units of the response |
| Weighted conformal prediction | Conformal with recency or likelihood weights on calibration scores | Applied with weights that are not normalised | Weights summing to 1 |
| Adaptive conformal inference (ACI) | Online update of the working miscoverage level after each outcome | Described as giving per-step coverage | Method; one parameter `gamma` |
| EnbPI | Ensemble batch prediction intervals: bootstrap ensembles plus a rolling residual pool | Assumed to need no tuning | Method; ensemble size `B` |
| Block conformal | Conformal on blocks of consecutive observations rather than single points | Block length chosen without regard to dependence | Method; block length `b` |
| Mondrian conformal | Conformal run separately within each category or group | Used with groups too small to be feasible | Method; group count `G` |
| Score stationarity | The score distribution is constant over the calibration window and stable into deployment | Assumed from a stationary plot of the response | Boolean diagnostic |
| Calibration window | The date range supplying calibration scores | Set to "all history" by default | Date range, count `n` |
| Horizon-specific calibration | Separate calibration for each forecast horizon | Replaced by one-step calibration carried forward | Method; `H` calibration sets |
| Probability integral transform (PIT) | The forecast cumulative distribution evaluated at the realised value; uniform when calibrated | Computed on a predictive distribution you do not have | Dimensionless in `[0, 1]` |
| Coverage curve | Empirical coverage plotted against nominal level or against time | Replaced by a single coverage number | Proportion against level |
| Interval score | A scoring rule penalising width and penalising misses | Confused with coverage | Units of the response |
| Rolling calibration | Recomputing the threshold on a moving calibration window | Confused with refitting the model | Window length in periods |
| Regime shift | A persistent change in the data-generating process | Called an outlier when it persists | Date of break |
| Beta-mixing | A dependence condition under which split conformal retains near-nominal coverage | Assumed to hold without evidence | Mixing coefficient, decaying in lag |

Three of these carry most of the practical damage. **Quantile index versus empirical quantile:**
`numpy.quantile(s, 0.9)` and `s[ceil((n+1)*0.9) - 1]` are different numbers, and only the second
carries the guarantee; at `n = 10` they differ by the whole gap between the ninth and tenth order
statistics. **Marginal versus conditional:** every stakeholder who hears "90% coverage" hears "90% of
the time, for my item, in my season", and the method promises neither. **Calibration set versus
validation set:** the calibration set is not for model selection, because comparing three models on it
and keeping the best makes the winner's scores optimistically small and the interval too narrow.
A fourth distinction is testable: a normalised score divides by a predicted scale and a raw residual
does not, so plot interval width against the predicted scale — if it is flat you used a raw residual.

## 5. The mechanics, step by step

Six phases, thirty-four steps. Each step names what you do, what it produces, and what fails if you
skip it.

### Phase A — Decide whether conformal is the instrument

**Step 1. Name the decision and the functional it consumes.** "Safety stock for item 4471 at DC
Leeds, set from the 90th percentile of four-week demand." A coverage target is not a decision. The
functional sets the quantile, the quantile sets alpha, and alpha sets the calibration size. Output:
one sentence with an item, a location, a horizon, and a quantile. Failure: "we need intervals" and a
95% target chosen because it sounds serious.

**Step 2. Ask whether the decision needs a marginal or a conditional statement.** A pooled service
target across a category can be served by a marginal interval. A per-item safety stock cannot,
because a stock-out on one item is not compensated by overstock on another. Output: marginal,
conditional, or "conditional within a stated partition". Failure: a marginal interval used for a
per-item policy.

**Step 3. Ask whether the future is plausibly a draw from the calibration population.** Not "is the
series stationary" — nobody's series is stationary — but "is the next twelve weeks exchangeable with
the last twenty in the way that matters for this score". Output: a yes with a named risk, or a no
with a named break. Failure: the question is never asked.

**Step 4. Establish the score's unit and direction.** Units of demand per week; larger is worse.
Output: a definition a second analyst can implement identically. Failure: a score defined as a
percentage in one script and as units in another.

**Step 5. Fix the coverage target and the calibration size together.** From the quantile, derive
`alpha`, then `n >= 1/alpha - 1`, then check the data supply. Output: a feasible pair `(alpha, n)`.
Failure: a 95% target with `n = 12`, which yields a vacuous interval from any correct
implementation.

### Phase B — Build the split

**Step 6. Choose the split ratio.** Conventional and defensible: 60–70% proper training, 10–20%
calibration, 20% evaluation. Output: three date ranges. Failure: the calibration block is whatever
was left over, which usually means the least representative block in the series.

**Step 7. Order the split in time, never at random.** Exchangeability permits a random split in a
regression problem. In a time series a random split scatters calibration weeks through the training
period, and the model has seen the neighbours of every calibration point. Output: three contiguous
blocks. Failure: `train_test_split(shuffle=True)`, which is leakage with a respectable name.

**Step 8. Freeze the model before touching calibration.** Parameters, features, and preprocessing are
fixed at the end of the proper training block. Output: a model artefact with a hash and a fit date.
Failure: the model is retrained after calibration, so the scores describe a model that is no longer
deployed.

**Step 9. Record the vintage of every input.** Calibration residuals must be reproducible from data as
it existed at each calibration week. Output: a lineage record per input table. Failure: restated
demand or a retrospectively coded promotion calendar produces residuals the model could not have
made.

**Step 10. Fix the horizon strategy.** One model per horizon, or one model and per-horizon
calibration. Output: a stated `H` and which calibration set corresponds to which horizon. Failure:
one-step calibration reused at every horizon.

**Step 11. Check `n` against `alpha` separately for each horizon.** A four-horizon design has four
calibration sets, each smaller than the total. Output: a table of `(h, n_h, alpha, k_h, feasible)`.
Failure: the `h = 4` set has eleven points and the target is 95%, so the interval is the whole line
and nobody notices because the library returns a large finite number.

### Phase C — Compute the scores

**Step 12. Choose the score deliberately.** Four standard choices, in the order most teams should
consider them:

| Score | Definition | Right when | Fails when |
|---|---|---|---|
| Absolute residual | `|y - f(x)|` | Errors roughly homoscedastic; simplest defensible case | Variance varies with level or season |
| Normalised residual | `|y - f(x)| / s(x)` | A scale model exists and is roughly right | The scale model is worse than a constant |
| Signed residual pair | Lower and upper empirical quantiles of `y - f(x)` | Errors are skewed, as demand usually is | `n` too small for two tails |
| CQR score | `max(q_lo(x) - y, y - q_hi(x))` | Heteroscedastic and skewed at once | Quantile estimates cross or are badly fit |

Output: a named score with a formula. Failure: the default in a tutorial, applied to intermittent
demand where the absolute residual is dominated by a handful of large observations.

**Step 13. Compute scores with the inference-time code path.** The same feature pipeline, the same
preprocessing, the same model artefact. Output: a vector of `n` scores with dates. Failure: training
preprocessing applied to calibration data, a difference of one standardisation constant that shifts
every score.

**Step 14. Inspect the score distribution before sorting it.** Plot it, count atoms, and look for a
point mass. Intermittent demand produces many identical scores — zeros, or the same small integer —
and ties change what the order statistic means. Output: a histogram and a tie count. Failure: a
continuous-distribution assumption in code that silently interpolates between tied values.

**Step 15. Test score stationarity across the calibration window.** Split the window in half and
compare the score quantiles; or regress score on time. Output: a verdict with a test statistic or at
minimum a plot with a stated reading. Failure: a calibration window that spans a promotional period
at one end and a quiet period at the other, producing a threshold that is right for neither.

**Step 16. Audit for leakage one more time.** For each input to the score, ask when it became
knowable. Output: a signed statement. Failure: the calibration residual uses the realised value of a
driver that was not available at forecast time.

### Phase D — Rank and form the interval

**Step 17. Sort the scores ascending.** Output: `s_(1) <= ... <= s_(n)`. Failure: sorting descending,
which produces the smallest score as the threshold and an interval of near-zero width.

**Step 18. Compute `k = ceil((n + 1)(1 - alpha))`.** Do this on paper once per design, not in a helper
function you have not read. Output: an integer. Failure: `ceil(n(1 - alpha))`, which at `n = 10`,
`alpha = 0.10` gives `9` instead of `10` and under-covers by roughly ten percentage points.

**Step 19. Handle `k > n` explicitly.** Either widen `alpha`, enlarge `n`, or return the whole line
and say so. Output: a branch with a logged reason. Failure: a silent clamp to `s_(n)`, which is a
plausible-looking interval with no guarantee.

**Step 20. Set `q = s_(k)`.** Not an interpolated quantile. Output: a threshold in the score's units.
Failure: `numpy.quantile`, which interpolates and breaks the guarantee at small `n`.

**Step 21. Form the interval.** Symmetric: `[f(x) - q, f(x) + q]`. Asymmetric: `[f(x) + q_lo,
f(x) + q_hi]` from two order statistics. CQR: `[q_lo(x) - Q, q_hi(x) + Q]`. Output: two endpoints in
units of demand. Failure: a sign error in the asymmetric case, which produces a reversed interval
that still looks plausible in a table.

**Step 22. Translate the interval into the decision.** For a reorder point, the upper endpoint feeds
safety stock: `safety stock = upper - expected demand over lead time`. Output: a number in units.
Failure: the interval is reported and the decision continues to use a point forecast, which is the
most common way a correct method produces no value.

### Phase E — Diagnose the coverage

**Step 23. Evaluate on a block the calibration never touched.** Later in time, not randomly chosen.
Output: an evaluation error vector. Failure: evaluation on the calibration block, which measures
in-sample fit and reports it as coverage.

**Step 24. Compute empirical coverage with its own uncertainty.** `covered / total`, and a binomial
interval around it, because at `n = 20` test points the standard error on a 90% estimate is about
6.7 percentage points. Output: a proportion with an interval. Failure: 18/20 reported as "90%"
without noting that 90% is also consistent with 76%.

**Step 25. Compute the PIT and plot it.** The probability integral transform is `F(y)` evaluated at
the outcome, where `F` is the predictive cumulative distribution implied by the interval. Under
calibration it is uniform. Output: a histogram or an empirical cumulative curve. Failure: PIT
computed against a distribution the method never produced.

**Step 26. Compute coverage by segment and by horizon.** Segment by item class, by season, by
promotion flag, by horizon. This is where the marginal number breaks. Output: a coverage table.
Failure: one number for the quarter.

**Step 27. Report width, and width against a benchmark.** Mean width in units, median width, and the
ratio to a seasonal-naive empirical interval at the same coverage. Output: a width table. Failure:
coverage reported without width, which rewards an interval that covers by being enormous.

**Step 28. Compare against the intervals from topic 07.** An empirical residual interval, a
simulation interval, and a quantile-regression interval at the same nominal level. Output: a
four-row comparison. Failure: conformal adopted because it is new rather than because it is better on
this data.

### Phase F — Choose a variant when the base case fails

**Step 29. If the failure is drift, use rolling or weighted calibration.** Recompute the threshold on
a moving window, or weight recent scores more heavily. Output: a threshold series. Failure: a window
so short that `n` falls below feasibility.

**Step 30. If the failure is an unknown or abrupt shift, use adaptive conformal inference (ACI).**
Update the working miscoverage level after each outcome: `alpha_{t+1} = alpha_t + gamma(alpha -
err_t)`, where `err_t` is `1` on a miss and `0` on a cover. Output: an `alpha_t` path and its
intervals. Failure: `gamma` untuned to shift speed, or a pool too small for the discrete threshold to
move at all.

**Step 31. If you have a bootstrap ensemble, use EnbPI.** Aggregate the ensemble's fitted values and
maintain a rolling pool of out-of-bag residuals, so no data split is needed and the interval updates
as observations arrive. Output: a sequential interval. Failure: the ensemble is not actually
bootstrapped, so the residual pool is not what the method assumes.

**Step 32. If dependence dominates, use block conformal.** Calibrate on blocks of consecutive
observations, preserving within-block dependence. Output: a block length `b` and a threshold.
Failure: `b = 1`, which is the base case dressed up.

**Step 33. If heteroscedasticity dominates, use CQR.** Fit conditional quantiles at `alpha/2` and
`1 - alpha/2` with pinball loss, compute the CQR score on the calibration block, and widen the quantile
band by the appropriate order statistic. Output: an interval whose width follows the predicted scale.
Failure: quantile crossing left unrepaired, producing a negative-width interval.

**Step 34. If the horizon is long, calibrate per horizon and record it.** Nothing else substitutes.
Output: `H` thresholds, `H` coverage reports. Failure: one threshold, `H` horizons, and a coverage
claim that is true at the only horizon nobody uses.

### 5.1 Conformal against the intervals you already have

| Method | What it assumes | What it guarantees | Cost | Prefer when |
|---|---|---|---|---|
| Normal interval | Gaussian errors, constant variance | Asymptotic coverage under the assumption | One variance estimate | Never for demand; keep as a straw man |
| Empirical residual interval | Stationary error distribution | Approximate coverage, better as `n` grows | Sort the residuals | Stationary series with long history |
| Simulation interval | A correctly specified generative model | Coverage conditional on the model being right | Compute | The generative model is the deliverable |
| Quantile regression | The quantile model is well specified | Nothing finite-sample; good if right | Quantile loss fit | Heteroscedasticity with informative features |
| Split conformal | Exchangeability of scores | Finite-sample marginal coverage | One sort | You need a guarantee and the split is clean |
| CQR | Exchangeability plus a decent quantile model | Finite-sample marginal coverage with adaptive width | Two quantile fits plus a sort | Both skew and heteroscedasticity |

The last two rows are this topic; the empirical residual row above them is what most teams should try
first on a long stationary series, because it assumes less machinery and fails in ways that are easier
to see.

### 5.2 What each time-series variant trades away

| Variant | Restores | Gives up | Diagnostic that catches the trade |
|---|---|---|---|
| Rolling calibration | Recency of the score distribution | Sample size, hence resolution and the coverage band | Threshold volatility across windows |
| Weighted (Nex-CP) conformal | Recency, smoothly | A clean finite-sample bound; the bound now includes a total-variation term | Weight concentration; effective `n` |
| ACI | Long-run coverage under arbitrary shift | Any finite-sample, per-step guarantee | `alpha_t` path; oscillation amplitude |
| EnbPI | No data split; sequential updating | Requires a bootstrap ensemble; approximate rather than exact coverage | Ensemble diversity; residual pool age |
| Block conformal | Within-block dependence | Effective `n` falls by the block length | Block length sensitivity |
| Mondrian | Group-conditional validity | Feasibility within small groups | Group sizes against `1/alpha` |
| Horizon-specific | The horizon gap | `H` calibration sets, each smaller | Per-horizon `n` and width |

Read the third column as the price: every repair buys validity under a weaker assumption by spending
sample size, and sample size sets the width of the coverage band. There is no free repair.

## 6. Worked example

Meridian Home distributes replacement filter cartridges. Item 4471 at the Leeds distribution centre
(DC) sells about 240 units a week with an autumn ramp. The decision is a reorder point set every four
weeks from the 90th percentile of four-week demand. All figures are constructed for teaching and are
internally consistent.

### 6.1 The design as run

| Field | Value | Comment |
|---|---|---|
| History | 100 weeks | Everything in the warehouse |
| Proper training | Weeks 1–70 | Model fit and frozen |
| Calibration | Weeks 71–80 | `n = 10` |
| Evaluation | Weeks 81–100 | 20 weeks, never touched |
| Score | Absolute residual, units | Simplest defensible choice |
| Target | 90%, so `alpha = 0.10` | Set by the 90th-percentile policy |
| Model | Exponential smoothing with trend and seasonality | Frozen at week 70 |

The design is deliberately the textbook one, with one flaw built in: the calibration window, weeks
71 to 80, is a plateau in the annual cycle. That is not carelessness. It is what happens when the
split is chosen by date arithmetic rather than by inspection, which is how almost every real split
is chosen.

### 6.2 The ten calibration scores

Signed residuals `e_i = y_i - f_i` for weeks 71 to 80, in units:

```text
  week   71   72   73   74   75   76   77   78   79   80
  e_i   +18  -42   +7  -61  +25  -14  +38   -9  +52  -30
```

Their mean is `-1.6` units, so the model is very nearly unbiased over this window. Their absolute
values, sorted, are:

```text
  s_(1)=7   s_(2)=9   s_(3)=14  s_(4)=18  s_(5)=25
  s_(6)=30  s_(7)=38  s_(8)=42  s_(9)=52  s_(10)=61
```

### 6.3 The order statistic

```text
  n = 10,  alpha = 0.10
  k = ceil( (n + 1)(1 - alpha) ) = ceil( 11 * 0.90 ) = ceil(9.90) = 10
  k <= n, so the guarantee is attainable
  q = s_(10) = 61 units
```

At a forecast of 240 units the interval is `[179, 301]` units, a width of 122 units, or plus or minus
25% of the mean. At finite `n` the attainable coverage band is `[0.900, 0.991]`: the interval might
cover 99% of the time, and nothing in the sample can tell you which end you are near.

### 6.4 Deployment, weeks 81 to 100

**Table 6.1 — Twenty weeks against the frozen interval, units**

| Week | Forecast | Actual | Error | Covered |
|---:|---:|---:|---:|:--:|
| 81 | 238 | 250 | +12 | yes |
| 82 | 245 | 212 | −33 | yes |
| 83 | 231 | 236 | +5 | yes |
| 84 | 252 | 204 | −48 | yes |
| 85 | 240 | 261 | +21 | yes |
| 86 | 236 | 178 | −58 | yes |
| 87 | 244 | 253 | +9 | yes |
| 88 | 249 | 232 | −17 | yes |
| 89 | 233 | 277 | +44 | yes |
| 90 | 247 | 221 | −26 | yes |
| 91 | 241 | 272 | +31 | yes |
| 92 | 238 | 186 | −52 | yes |
| 93 | 250 | 265 | +15 | yes |
| 94 | 243 | 235 | −8 | yes |
| 95 | 246 | 285 | +39 | yes |
| 96 | 252 | 309 | +57 | yes |
| 97 | 258 | 332 | +74 | **no** |
| 98 | 262 | 310 | +48 | yes |
| 99 | 266 | 349 | +83 | **no** |
| 100 | 271 | 330 | +59 | yes |

Empirical coverage: `18 / 20 = 0.900`. Exactly nominal. A dashboard built on this table reports the
target being met, to the decimal.

### 6.5 What the same table says when you cut it differently

```text
  weeks 81-94 (plateau):   14 / 14 = 1.000
  weeks 95-100 (ramp):      4 /  6 = 0.667
  weeks 97-100 (peak):      2 /  4 = 0.500

  mean signed error, weeks 81-94:  (-12 - 33 - 48 + 21 - 58 - 17 - 26 - 52 - 8) / 9
                                 = -233 / 9 = -25.9 units  (over-forecast)
  mean signed error, weeks 95-100: (39 + 57 + 74 + 48 + 83 + 59) / 6
                                 = 360 / 6 = +60.0 units   (under-forecast)
```

The mechanism is now exact rather than suggestive. The interval half-width is 61 units and the mean
error during the ramp is +60 units. When the bias equals the half-width, a symmetric interval covers
about half the time, by construction. Coverage did not degrade because of a software fault or an
unlucky draw. It degraded because the score's location moved by exactly the amount the score's
spread was supposed to absorb.

The calibration window could not have seen this. Weeks 71 to 80 contained no ramp, so no residual in
it carried a positive seasonal component. The score distribution was stationary within the window
and non-stationary across the year, and a within-window stationarity test would have passed.

### 6.6 Three repairs, with their arithmetic

**Repair 1 — rolling calibration.** At week 90, recompute the threshold on weeks 81 to 90. Sorted
absolute errors: `8, 9, 12, 15, 17, 21, 26, 31, 33, 44`. With `n = 10`, `k = 10`, `q = 44`. That is
narrower than 61, because weeks 81 to 90 were quiet. At week 100, recompute on weeks 91 to 100:
`8, 15, 31, 39, 48, 52, 57, 59, 74, 83`, so `q = 83`. The threshold lags the shift by one window
length, which is the price, and the guarantee is now conditional on a random window rather than a
fixed one.

**Repair 2 — horizon-specific calibration.** Everything above is one-step error applied to a
four-step decision. A four-week-ahead residual is not a one-week-ahead residual; for a random walk it
is twice as wide. A correct design calibrates on residuals from folds whose outcome fell four weeks
after the origin. With 100 weeks of history and origins spaced four weeks apart, that leaves 26
folds at most, so `n` is smaller and the coverage band wider.

**Repair 3 — asymmetric calibration.** The ramp is one-sided. Using all twenty error values from
Table 6.1 as a pool (arithmetic, not evidence, because those values were not available at week 80):
sorted signed errors run from `-58` to `+83`. With `n = 20`, `alpha = 0.10`, the lower index is
`ceil(21 * 0.05) = 2` and the upper is `ceil(21 * 0.95) = 20`, giving `[-52, +83]`, a width of 135
units. A symmetric interval at the same pool would use `k = ceil(21 * 0.90) = 19`, giving `+/- 74`
and a width of 148 units. The asymmetric band is 8.8% narrower and puts the width where the risk is.

> **Note:** Repair 3 is the cheapest and most often skipped. Demand errors are skewed upward far
> more often than downward, because demand is bounded below and promotions are one-directional. A
> symmetric interval throws away that asymmetry and pays for it in width.

### 6.7 What the example establishes

**A marginal guarantee can be met exactly while the conditional experience fails completely.** The
90.0% figure is not a lie, not a rounding, and not a bug. It is the number the method promised,
computed correctly, and the planner at the peak still had a coin flip. **The failure has a date and a
magnitude.** The ramp begins around week 95 and the bias reaches the half-width by week 97, so a
coverage report by week would have shown it in week 97, not at quarter end. **None of the repairs is
free.** Rolling calibration costs sample size and lags, horizon-specific calibration costs three
quarters of the available folds, and asymmetric calibration costs the clean two-sided guarantee at
small `n`. State which cost you accepted.

## 7. Code, query, or calculation

One calculation, on paper, reproducing the arithmetic of section 6.3 and adding the two edge cases
that break implementations.

```text
Calibration residuals, units, weeks 71-80 (signed, e_i = y_i - f_i):
  +18  -42   +7  -61  +25  -14  +38   -9  +52  -30

Absolute scores s_i = |e_i|, sorted ascending:
  s_(1)=7  s_(2)=9  s_(3)=14  s_(4)=18  s_(5)=25
  s_(6)=30 s_(7)=38 s_(8)=42  s_(9)=52  s_(10)=61

n = 10, alpha = 0.10
k = ceil((n+1)(1-alpha)) = ceil(11 * 0.90) = ceil(9.90) = 10
k <= n, so the guarantee is attainable.  k/(n+1) = 10/11 = 0.9091 >= 0.90
q = s_(10) = 61 units
C(x) = [ f(x) - 61 , f(x) + 61 ]      at f(x)=240:  [179, 301] units

Coverage band, distinct scores:  0.900 <= P(Y in C) <= 0.900 + 1/11 = 0.991

EDGE CASE 1 -- target too tight for the sample:
  alpha = 0.05 -> k = ceil(11 * 0.95) = ceil(10.45) = 11 > n = 10
  The valid interval is the whole real line. Feasibility needs n >= 1/alpha - 1 = 19.

EDGE CASE 2 -- the wrong index:
  ceil(n(1-alpha)) = ceil(10 * 0.90) = 9  ->  q = 52, width 104, not 122
  Under nominal by roughly 52/61 of the intended tail. No guarantee.
```

Expected output, in units: `q = 61`, interval `[179, 301]`, coverage band `[0.900, 0.991]`, and two
edge cases that a library will not raise for you.

Three checks before trusting any implementation.

- **Recompute `k` by hand for every new `(n, alpha)` pair.** It is one line of arithmetic and it
  catches the two most common errors, the missing `+1` and the missing ceiling.
- **Confirm `k <= n`.** Print it. A threshold equal to the maximum calibration score is a warning
  sign when `alpha` is small, not a success.
- **Confirm the sort is ascending and the index is 1-based on paper, 0-based in code.** `s_(k)` in the
  mathematics is `s[k - 1]` in Python and `s(k-1)` in R. This off-by-one is worth a test.

## 8. Data, standards, and regulatory context

### 8.1 What the data must contain

| Requirement | Why it matters | What to record | If absent |
|---|---|---|---|
| Frozen forecast archive | The score must be against the forecast that was made, not a refit | Forecast value, origin, model hash | Scores describe a model that never ran |
| Outcome with revision history | Restated demand changes the residual | Observation date, value, revision date | Leakage through restatement |
| Calibration block isolated from training | The split is the method | Date ranges for all three blocks | In-sample calibration |
| Censoring flag | Stock-outs truncate demand | Boolean per item-location-week | Understated scores; intervals too narrow |
| Item and location master with validity dates | The series must mean the same thing across the window | Key with `valid_from` and `valid_to` | Silent discontinuities read as demand events |
| Promotion and event calendar as known at the origin | A retroactively coded promotion contaminates the score | Calendar vintage per origin | Scores that anticipate events |
| Stated horizon per score | One-step scores do not transfer | Integer `h` per calibration row | The horizon gap, invisible |

The first row is the one most organisations lack. Planning systems store the current forecast, not
the forecast as it stood at each origin. Without an archive, split conformal has no valid score to
compute, and any attempt to reconstruct one introduces hindsight.

### 8.2 Statistical standards that settle vocabulary

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Meanings of coverage, confidence, prediction interval, accuracy | B |
| ISO 5725-1:1994 | Accuracy, trueness, precision of measurement | Distinguishing systematic from random error | B |
| JCGM 100:2008 | Expression of uncertainty in measurement | The vocabulary of an uncertainty statement, including coverage factor and coverage interval | B |
| ISO 8601-1:2019, ISO 8601-2:2019 | Date and time representation | Week numbering, durations, offsets; where a split boundary falls | B |

None of these mentions conformal prediction. They matter because "90% confidence" without a statement
of whether it means a coverage probability, a confidence coefficient, or a tolerance interval will be
read three ways by three readers. JCGM 100:2008 is the most useful of the four: it separates the
*coverage interval* from the *coverage probability*, the distinction a marginal conformal statement
depends on.

### 8.3 Where regulation imposes adjacent obligations

| Regime | Domain | What it requires | Relevance | Grade |
|---|---|---|---|---|
| SR 11-7, Federal Reserve model risk guidance (2011) | Banking models | Validation including outcomes analysis, ongoing monitoring, documented conceptual soundness | The shape of the obligation, not the letter | B |
| Regulation (EU) 2024/1689 (AI Act), Article 15 | High-risk AI systems | Accuracy, robustness, cybersecurity, with declared metrics and levels | A declared metric and level is what a conformal specification is | B |
| FDA guidance on predetermined change control plans | AI-enabled medical devices | A pre-specified plan for model change with supporting evaluation | The protocol discipline, not the content | B |

A demand forecast scheduling production is not a high-risk AI system under the AI Act and a
distribution centre is not a bank. Do not describe an interval as "SR 11-7 compliant"; the phrase has
no meaning outside its domain. What transfers is the documentation obligation: state the metric, the
level, the calibration population, and the monitoring rule, and keep the artefact that lets someone
else re-run it.

### 8.4 What conformal does not fix

Four data defects survive every conformal variant, and the guarantee creates a false sense of security
around each. **Censoring:** if demand is measured as shipments, the largest true demands never appear,
so the scores come from a truncated sample and the interval covers future *shipments* rather than
future *demand* — the difference is exactly the stock-outs the interval is meant to prevent.
**Measurement error:** an interval around a forecast of a mismeasured quantity is valid for the
mismeasured quantity; a 3% warehouse count error enters the interval as 3%. **Master-data change:** an
item renumbered mid-window produces a discontinuity the score treats as a demand event, so the
interval widens, correctly, for the wrong reason. **Aggregation mismatch:** a score computed at
item-week level does not calibrate an interval at category-month level.

### 8.5 Convention rather than requirement

- **A 60/20/20 split** is convention. It leaves a calibration block of usable size and an evaluation
  block of usable length, and it is not optimal for any particular `n`.
- **`alpha = 0.10`** is convention inherited from the 90% service-level target most inventory policies
  use. It is not derived from a cost function; Module 08 topic 10 does that derivation.
- **A ten-point calibration set** is a common minimum because it is the smallest feasible set for a
  90% interval. It is far too small for a stable threshold, and its band spans nine percentage points.
- **"Coverage above 90%"** as a pass mark is convention. A method can pass by over-covering, which
  means the intervals are too wide and the inventory cost is too high.

An analyst who separates the requirement (finite-sample validity under a stated assumption) from the
convention (these particular numbers) can defend a design; one who cannot reproduces defaults and
calls them a standard.

## 9. Industry practice and case evidence

### 9.1 The lineage, because it explains the shape of the guarantee

Conformal prediction comes from a specific research programme, and knowing it explains why the
guarantee is narrow in exactly the way it is.

Vladimir Vovk's work in the 1980s and 1990s on algorithmic randomness and exchangeability, conducted
under Kolmogorov's programme, produced the transductive confidence machine. The 1998 paper with
Gammerman and Vapnik on learning by transduction, and the 1999 papers that followed, moved the idea
toward prediction. The canonical statement is Vovk, Gammerman and Shafer (2005), *Algorithmic
Learning in a Random World*, with a second edition in 2022.

Two consequences of that lineage are operational.

**The method is a test of randomness, not a model of the data.** That is why it needs no likelihood
and no distributional family: it asks whether the test point is a plausible member of the same
exchangeable sample as the calibration points. It is closer to a rank test than to a regression.

**The guarantee was always conditional on a stated model of randomness.** The 2005 book treats
exchangeability, Gaussian, and Markov models as three instances of on-line compression models, and
its chapter 8 generalises the construction to all of them. "Distribution-free" has always meant
distribution-free *within* a stated randomness model. A time series is a change of that model, not
an extension of the old one, and the book says so.

> **Note:** This is why the phrase "assumption-free" in vendor material is a reliable signal that
> the writer has not read the source. The assumption is exchangeability; everything else is
> downstream of it.

### 9.2 What the methodological record establishes

| Finding | Source | Implication | Grade |
|---|---|---|---|
| Exact finite-sample conditional validity requires infinite-width bands at non-atomic points | Lei and Wasserman (2014) | Conditional coverage is not a matter of a better score | A |
| Even approximate subgroup-conditional coverage cannot beat a wide marginal interval in general | Barber, Candès, Ramdas and Tibshirani (2021) | Local adaptivity is the achievable target, not conditional coverage | A |
| Coverage loss under non-exchangeability is bounded by a weighted total-variation distance; recency weights (Nex-CP) restore it | Barber, Candès, Ramdas and Tibshirani (2023) | A principled route to drift-robust intervals, at the cost of a bound rather than an equality | A |
| Split conformal remains valid with a small coverage penalty for stationary beta-mixing processes | Oliveira, Orenstein, Ramos and Romano (2024) | Dependence per se is less fatal than often claimed; non-stationarity is the real problem | A |
| ACI attains the target long-run coverage frequency under arbitrary distribution shift | Gibbs and Candès (2021) | Online correction is available when the shift is unknown | A |
| EnbPI gives approximately valid marginal coverage for strongly mixing errors without a data split | Xu and Xie (2021) | An ensemble route when splitting is expensive | A |
| CQR produces shorter intervals than split conformal at the same coverage across eleven benchmark datasets | Romano, Patterson and Candès (2019) | Adaptivity is worth the extra model when the data are heteroscedastic | A |
| Distributional conformal prediction recovers conditional validity asymptotically under a consistent conditional CDF estimate | Chernozhukov, Wüthrich and Zhu (2021) | The escape from the impossibility is an assumption, explicitly bought | A |

The pattern across the table is worth stating plainly. Every row that restores coverage under
dependence does so by adding an assumption — beta-mixing, a total-variation bound, a consistent
conditional CDF, a bootstrap ensemble — and every added assumption has to be checked on your data.
Conformal does not remove assumptions. It relocates them from the error distribution to the sampling
process, where they are harder to see and easier to ignore.

### 9.3 What the benchmarks show

| Study | Setting | Result | Grade |
|---|---|---|---|
| Conformal methods benchmark on roughly 3,000 monthly sales series at 90% nominal, AutoARIMA base | Multi-step split conformal, ACI, global conformal and AcMCP met nominal coverage; multi-step split conformal best on the interval score; EnbPI and SPCI under-covered | Horizon-specific calibration is the strongest simple defence on real sales data | B |
| Pooled M5 study with LightGBM and centered CQR, 80% nominal | Reported coverage around 0.38 to 0.40 | Pooling across series violates cross-sectional exchangeability; heavy tails compound it | C |
| Applied M5 write-up with a neural base model, 95% nominal | Marginal coverage 96.5%, but around 91% on smooth and erratic series | Marginal pass with conditional failure on exactly the segments that matter | D |
| Public benchmark repository, ETS conformal intervals at 80% nominal on a store-item dataset | 65.25% coverage; a gradient-boosted model over-covered at 85.20%; a foundation model's native quantiles were near nominal at 81.04% | Base-model misspecification dominates the conformal layer | E |

Read the grades before the numbers. The first row is a preprint benchmark and the most useful of the
four because it compares methods on one protocol with a stated base model. The last two are single
unreviewed exercises, useful as existence proofs of failure and not as estimates of a rate. The
middle row comes from a journal whose review status I could not establish; treat the figure as a
warning, not a measurement. Across all four, when conformal under-covers on real demand data the
cause is a violated exchangeability assumption rather than an implementation error — and pooling
items, pooling horizons, or calibrating on the wrong season each produce the same symptom while
requiring different repairs.

### 9.4 What practice looks like

Conformal now ships in general-purpose libraries — MAPIE, Fortuna, PUNCC, and language-specific
packages including a Julia implementation with a time-series guide — plus commercial forecasting
offerings with a conformal option. The arithmetic of section 7 is therefore no longer the barrier; the
barrier is the specification of which block is the calibration set, at which horizon, on which
population, with which diagnostic. Three patterns are visible. **The wrapper pattern** applies
conformal to an existing point forecast with a calibration block carved from recent history; it is the
common case and the most exposed to the horizon gap, because the point forecast is often multi-step
while the residuals are one-step. **The library-default pattern** calls a package's `conformal` method
with defaults, usually a random split, which leaks invisibly on a demand series. **The specified
pattern** states the score, window, horizon, target, diagnostic, and break condition; it is rare and
the only one that survives an audit.

### 9.5 What is missing from the public record

Searches for this topic returned no peer-reviewed study of conformal prediction adoption in
commercial supply chain planning, and no published distribution of coverage outcomes across deployed
systems; claims that "most companies now use conformal intervals" are impressions, and the vendor
material that makes them is graded D. What the record supports is narrower: the theory of when the
guarantee holds is settled and recent, the empirical record on sales data is thin and mostly
unreviewed, and it points one way. The gap between a theorem that says coverage degrades gracefully
and a benchmark showing 65% against an 80% target is the gap between a bound and a data set.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Calibration on the training residuals | The calibration date range overlaps the model fit range | In-sample scores are optimistically small | Interval too narrow; coverage collapses live | Three explicit date ranges; assert non-overlap in code |
| Random split on a time series | `shuffle=True` in a split call, or a calibration set scattered across years | Neighbouring weeks leak into training | Severe undercoverage with a clean-looking backtest | Chronological split, enforced by a date assertion |
| One-step calibration at an `h`-step horizon | One threshold; the decision horizon is four weeks | Different random variables | Coverage decays with horizon, silently | Per-horizon calibration; report `n_h` |
| Quantile index without the `+1` | Code contains `n * (1 - alpha)` | Wrong order statistic | Undercoverage of roughly one order-statistic gap | Hand-compute `k`; unit-test the function |
| Interpolated quantile | `numpy.quantile` in the threshold line | Interpolation breaks the rank argument | Guarantee void at small `n` | Use the order statistic; test at `n = 10` |
| `k > n` silently clamped | A 95% target with twelve calibration points | No feasible threshold exists | A plausible interval with no guarantee | Assert `k <= n`; fail loudly |
| Pooling items into one calibration set | One threshold across 4,000 stock-keeping units (SKUs) with different behaviour | Cross-sectional non-exchangeability | Marginal pass, per-item failure | Mondrian by class, or per-item where `n` allows |
| Marginal coverage reported as the only number | One percentage on a slide; no segmentation | Averaging over test points | The peak-season failure of section 6 | Coverage by segment and by week |
| Coverage reported without width | "90% coverage achieved" and no width table | Trivial route to coverage | Inventory overstock; cost not measured | Report mean and median width against a benchmark |
| Evaluation on the calibration block | The number is suspiciously close to nominal | In-sample | No information at all | Hold out a later block; state it |
| Calibration window shorter than the forecast horizon | 8-week calibration, 12-week horizon | The outcome distribution was never sampled | Unknown direction of error | Window at least as long as the horizon, ideally several times |
| Score not matched to structure | Symmetric interval on a series with a one-sided promotion effect | Skew absorbed into width, not location | Wide interval that still misses the tail | Asymmetric or CQR |
| ACI with `gamma` untuned and `n` small | The threshold never moves; `alpha_t` is flat | Discrete threshold has no resolution | The adaptive method is decorative | Check threshold movement; enlarge the pool |
| Bias mistaken for variance | The interval is wide and still misses | Conformal recentres on a biased point forecast | Coverage looks marginal-ok, cost is high | Diagnose bias separately; fix the model |

The two rows that matter most are **pooling items**, because a single threshold across a catalogue is
a marginal guarantee over the catalogue and a catalogue is not a decision unit, and **bias mistaken
for variance**, because a conformal interval around a biased forecast is still marginally valid while
being wide by roughly twice the bias. Report the mean signed score beside the width: if it is
materially non-zero, you have a model problem that no interval method repairs.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Exchangeability of calibration and test scores | The rank argument is the entire proof | The coverage statement has no basis | Compare score distributions across the calibration window; test for trend and seasonality |
| Calibration and deployment populations coincide | The threshold is a quantile of one distribution | Systematic under- or over-coverage | Coverage by segment, by week, by horizon |
| The model is frozen between calibration and deployment | Scores must describe the deployed predictor | Threshold mismatched to the model in production | Hash and date the artefact; diff backtest and production |
| The score is computable identically at inference | The score distribution must be the same object | Scores on a different scale; threshold meaningless | One code path; unit test on a frozen record |
| `n` is large enough for the target `alpha` | `k <= n` is a feasibility condition | Vacuous interval, or a silent clamp | Compute `k` and `n` per horizon |
| Scores are distinct, or ties are handled | The rank argument assumes a continuous score | Coverage band widens; exact upper bound fails | Count atoms in the score vector |
| The horizon is calibrated, not transferred | One-step and `h`-step errors differ | Coverage decays with `h` | Per-horizon coverage report |
| The calibration block is representative of the decision period | A quantile of the wrong season is the wrong quantile | The section 6 failure | Compare calibration and evaluation blocks on level, variance, seasonality |
| The outcome is not censored | Scores must reflect demand, not shipments | Interval covers the wrong quantity | Censoring rate per week; compare with unconstrained demand |
| Marginal coverage is the relevant target | The guarantee is marginal | A per-item decision is under-served | Segment the coverage report; if segments differ, the target is conditional |

**What no conformal variant covers.** A structural break after the last calibration point. A new
channel, a competitor exit, a change in the promotion calendar, a network redesign. These are not
estimable from the calibration sample, and no weighting scheme recovers what was never observed. The
honest response is a break register: a dated list of events that invalidate the interval, with the
instruction to recalibrate after each.

**The data floor.** A 90% interval with ten calibration points has a coverage band spanning nine
percentage points, and no amount of cleverness narrows it. Narrowing requires `n`, and `n` requires
history. A series younger than about two years cannot support a 95% per-horizon interval at a
four-week horizon, because the four-horizon calibration set will not reach nineteen points.

**The interpretability limit.** A conformal interval describes a score, not the data-generating
process, and it will not tell you why the interval is wide. If the width is driven by three
promotional weeks, the interval is correct and the diagnosis is missing; pair it with a decomposition.

## 12. Exercises

### Exercise 1 — Derive the index (calculation and proof sketch, 35 minutes)

Starting from the assumption that the test score's rank among `n + 1` exchangeable scores is uniform,
derive `k = ceil((n + 1)(1 - alpha))`, the feasibility condition, and the upper end of the coverage
band.

**Deliverable:** one page of algebra and a three-line proof sketch.
**Acceptance criteria:** the rank-uniformity step is stated, not assumed; the ceiling is justified as
the smallest integer meeting the inequality; the feasibility condition is derived rather than quoted;
the upper band `1 - alpha + 1/(n + 1)` is derived under a distinctness assumption that is stated.
**Time:** 35 minutes.

### Exercise 2 — Hand arithmetic on ten points (calculation, 30 minutes)

A calibration set has absolute scores `3, 11, 6, 24, 9, 17, 41, 2, 15, 28` in units, `n = 10`. Compute
the 80% and 90% intervals at a forecast of 150 units, and state whether a 95% interval is available.

**Deliverable:** the sorted vector, `k` for each level, `q`, both intervals, and the feasibility
verdict.
**Acceptance criteria:** `k` computed from `(n + 1)(1 - alpha)`; the 95% case identified as
infeasible with the required `n` stated; both intervals given in units with endpoints.
**Time:** 30 minutes.

### Exercise 3 — Find the seam in a real series (diagnosis, 45 minutes)

Take any demand series with at least 100 weekly observations. Split it 70/10/20 in time and compute
a split conformal interval at 90%. Then evaluate coverage by week.

**Deliverable:** a coverage-by-week plot and a one-paragraph attribution.
**Acceptance criteria:** the calibration window's position in the seasonal cycle is stated; at least
one week with coverage below 70% is identified, or its absence is explained; the attribution names
drift, seasonality, autocorrelation, regime change, or the horizon gap, with evidence.
**Time:** 45 minutes.

### Exercise 4 — Marginal versus conditional (critique, 40 minutes)

A report states "90.2% coverage achieved across the category for the quarter." Write the three
questions you would ask, and what each answer would change.

**Deliverable:** three questions with the decision each would alter.
**Acceptance criteria:** one question targets segmentation, one targets the horizon, one targets the
calibration window's representativeness; each answer is tied to a specific action.
**Time:** 40 minutes.

### Exercise 5 — Choose a variant (design, 50 minutes)

For each situation, name the variant and the parameter you would set, and state the cost you accept.

1. A stable series with 300 weeks of history and a gradual level drift.
2. A promotion-driven category where the shift is abrupt and undated.
3. A bootstrap ensemble already in production, with no appetite for a new split.
4. A highly seasonal item where the peak is four weeks long.
5. A catalogue of 3,000 items with wildly different volumes.

**Deliverable:** a five-row table: situation, variant, parameter, cost accepted, diagnostic.
**Acceptance criteria:** each parameter is a number or a rule, not "default"; each cost is stated in
sample size, assumption, or lag; each diagnostic would detect the variant failing.
**Time:** 50 minutes.

### Exercise 6 — CQR by hand (calculation, 45 minutes)

A quantile model gives `q_lo = 180` and `q_hi = 260` at a point where the outcome is `230`. Compute the
CQR score. Given five calibration scores `-12, +4, +9, +22, +31`, form the 80% interval at a point
where `q_lo = 200` and `q_hi = 290`.

**Deliverable:** the score definition, the five scores, the sorted vector, `k`, `Q`, and the final
interval.
**Acceptance criteria:** the `max` form of the score is used, not a residual; `k = ceil(6 * 0.8) = 5`;
the interval is `[q_lo - Q, q_hi + Q]` with `Q` stated; the width is compared with what a symmetric
absolute-residual interval would give.
**Time:** 45 minutes.

### Exercise 7 — Diagnose a coverage collapse (critique, 45 minutes)

A model reports 89% coverage in calibration and 64% in the first live quarter. The calibration window
was the previous twelve weeks; the horizon is six weeks. Give the two most likely causes, the
evidence that would separate them, and the repair for each.

**Deliverable:** two hypotheses, two discriminating measurements, two repairs.
**Acceptance criteria:** one hypothesis is the horizon gap; one is distributional; the measurements
are specific and cheap; the repairs are different.
**Time:** 45 minutes.

### Exercise 8 — Compare four interval methods (application, 60 minutes)

On one series, compute a normal interval, an empirical residual interval, a split conformal interval,
and a CQR interval at 90%. Evaluate all four on the same held-out block.

**Deliverable:** a four-row table of coverage, mean width, and width at the seasonal peak.
**Acceptance criteria:** identical evaluation block and nominal level; coverage reported with a
binomial interval; width reported at the peak separately from the average; one sentence on which you
would deploy and why.
**Time:** 60 minutes.

### Exercise 9 — Write the specification (design, 50 minutes)

Write a one-page interval specification for a named decision.

**Deliverable:** the specification, with a worked example row.
**Acceptance criteria:** states the decision and functional, the score and its formula, the three
date ranges, `n` and `k` per horizon, the coverage target and the attainable band, the diagnostic,
the segmentation for reporting, and the break condition that invalidates the interval.
**Time:** 50 minutes.

## 13. Self-check answers

**Exercise 1.** Let the `n + 1` scores be exchangeable. The test score's rank `R` is uniform on
`{1, ..., n + 1}` because exchangeability makes every ordering equally likely and the ranks are a
relabelling of the orderings. The interval covers when `s_test <= q`, and with `q = s_(k)` this holds
whenever `R <= k`, so `P(cover) = k/(n + 1)`. Requiring `k/(n+1) >= 1 - alpha` and taking the
smallest integer gives `k = ceil((n+1)(1-alpha))`. Feasibility is `k <= n`, which rearranges to
`n >= 1/alpha - 1`. With distinct scores, coverage is at most `1 - alpha + 1/(n+1)`, because
`k/(n+1) < 1 - alpha + 1/(n+1)`. The ceiling is what makes the guarantee an inequality rather than
an equality, and the gap it creates is the coverage band.

**Exercise 2.** Sorted: `2, 3, 6, 9, 11, 15, 17, 24, 28, 41`. At 80%, `k = ceil(11 * 0.80) =
ceil(8.8) = 9`, `q = 28`, interval `[122, 178]`. At 90%, `k = ceil(11 * 0.90) = 10`, `q = 41`,
interval `[109, 191]`. At 95%, `k = ceil(11 * 0.95) = 11 > 10`, so the interval is the whole line and
`n = 19` is required. Note the jump from 28 to 41 between the 80% and 90% intervals: that is one
order statistic, and it is the entire resolution of a ten-point sample.

**Exercise 3.** There is no single correct answer, but the acceptance criteria are checkable. The
common finding is a calibration window that sits in a quiet season, giving coverage near 1.0 in the
quiet weeks and below 0.7 in the ramp. A submission that reports only the average has not done the
exercise. A submission that finds no failure should still state where the calibration window sat in
the cycle and why that position was benign, because "no failure" without that statement is an
untested claim.

**Exercise 4.** Ask: (a) coverage by item class and by week, not by category-quarter — this
separates a genuine marginal pass from a conditional failure and would change whether any per-item
policy is safe; (b) coverage by horizon — if the decision consumes `h = 4` and the report pools
`h = 1..4`, the relevant number is missing and the answer would change the calibration design; (c)
where the calibration window sat in the seasonal cycle — if it missed the peak, the threshold is
right for a season that has passed, and the answer would trigger a rolling recalibration.

**Exercise 5.** A defensible table:

1. Rolling calibration, window 52 weeks, cost is a lag of one window and a smaller effective `n`;
   diagnostic is threshold volatility across windows.
2. ACI with `gamma` around 0.01 to 0.05, cost is the loss of any finite-sample per-step guarantee;
   diagnostic is the `alpha_t` path and its oscillation.
3. EnbPI, ensemble size as already deployed, cost is approximate rather than exact coverage;
   diagnostic is ensemble diversity and residual-pool age.
4. Horizon-specific calibration with a Mondrian split by season, cost is a much smaller `n` per cell;
   diagnostic is `n` per cell against `1/alpha - 1`.
5. Mondrian by volume class, three to five classes, cost is feasibility pressure in the smallest
   class; diagnostic is group sizes and per-group coverage.

A weak answer names variants without parameters or without the cost.

**Exercise 6.** `E = max(180 - 230, 230 - 260) = max(-50, -30) = -30`. A negative score means the
outcome fell inside the quantile band, which is possible and is why the score is a `max` rather than
an absolute value. Sorted scores: `-30, -12, 4, 9, 22, 31` for the five calibration values plus this
one; if only the five are used, sorted are `-12, 4, 9, 22, 31`, `k = ceil(6 * 0.8) = 5`, `Q = 31`.
Interval `[200 - 31, 290 + 31] = [169, 321]`, width 152 units. A symmetric absolute-residual
interval built from the same five absolute values `12, 4, 9, 22, 31` would use `k = ceil(6 * 0.9) =
6 > 5` at 90%, and at 80% `k = 5`, `q = 31`, width 62 around the point forecast. The comparison is
instructive: CQR's width is a band around a quantile model that already knows the spread, so the
absolute width is larger while the *coverage-conditional* width is better placed.

**Exercise 7.** Hypothesis one: the horizon gap. Calibration on one-step residuals applied at six
weeks. Discriminating measurement: compute coverage by horizon within the live quarter; if `h = 1`
coverage is near nominal and `h = 6` is far below, the gap is the cause. Repair: calibrate per
horizon. Hypothesis two: the calibration window was not representative, for example a quiet twelve
weeks before a seasonal ramp or a promotion. Discriminating measurement: compare the calibration and
live blocks on mean level and on residual variance; also plot the score by week across the boundary.
Repair: rolling or weighted calibration, and a longer window that spans a full cycle.

**Exercise 8.** There is no universal winner and the acceptance criteria are the deliverable. The
expected pattern on a seasonal series is: the normal interval under-covers at the peak and is too
wide off-peak; the empirical residual interval is better but still constant-width; split conformal
is comparable to the empirical interval with a guarantee; CQR has the best peak width at the same
coverage. A submission that reports coverage without width has measured half the problem.

**Exercise 9.** A strong specification has eight fields: decision and functional; score with formula
and units; three date ranges with counts; `n`, `k`, and the attainable coverage band per horizon; the
coverage target and its provenance; the diagnostic (coverage by segment, PIT, width); the reporting
segmentation; and the break condition. Three prohibitions: no pooled coverage number without the
segmented table, no threshold from an interpolated quantile, no interval transferred across a
horizon without recalibration.


## 14. Interview questions

**1. "What does conformal prediction guarantee?"** Strong: finite-sample marginal coverage at a stated
level, from a calibration set and a nonconformity score, under exchangeability, naming the order
statistic and the feasibility condition. Weak: "distribution-free prediction intervals," with no
mention of the assumption or the word marginal.

**2. "What is exchangeability and why does a time series break it?"** Strong: the joint distribution is
unchanged by permutation; a series violates it because the index carries information, and names
drift, seasonality, autocorrelation, and regime change as distinct mechanisms with distinct
consequences. Weak: "the data are not i.i.d.," which is true and explains nothing.

**3. "Ninety per cent coverage. What is wrong with that sentence?"** Strong: it is marginal, averaged
over test points and calibration draws; conditional coverage at the decision point may be far from
90%, and no distribution-free method can fix that. Weak: "nothing, if the calibration was done
properly."

**4. "How many calibration points do you need for a 95% interval?"** Strong: derives
`n >= 1/alpha - 1 = 19` and adds that 19 is the feasibility floor, not a recommendation, because its
coverage band still spans five percentage points. Weak: gives a number with no derivation.

**5. "You have one-step residuals and a four-week decision. What do you do?"** Strong: recognises the
horizon gap as a difference in random variables rather than a matter of scaling, and calibrates per
horizon on folds whose outcome fell at that horizon, accepting the smaller `n`. Weak: multiplies the
threshold by the square root of the horizon, which is not a conformal argument.

**6. "Why not just use quantile regression?"** Strong: quantile regression gives adaptive width but no
finite-sample guarantee; CQR gives both, at the cost of two quantile fits and a crossing repair, so the
choice depends on whether you need the guarantee or the shape. Weak: "conformal is distribution-free,"
which does not answer the question.

**7. "Your coverage is 90% overall and 50% in December. Is the method broken?"** Strong: no, it
delivered its marginal promise; the assumption that the calibration window represented December
failed, and the repair is seasonal calibration or a wider window plus a segmented report. Weak: "yes,
retrain the model," which treats a distributional problem as a fitting problem.

**8. "What does ACI buy you, and what does it cost?"** Strong: long-run coverage under arbitrary shift
via an online update of the working miscoverage level; it costs any finite-sample per-step guarantee,
needs a step size tuned to the shift speed, and needs enough points for the discrete threshold to
move. Weak: "it adapts," with no mechanism and no cost.

**9. "When would you not use conformal?"** Strong: when the series has an undated regime change, when
the decision needs conditional coverage and no partition is available, when the calibration sample is
too small for the target level, or when a well-specified generative model already gives the whole
distribution. Weak: "never, it always works."

**10. "A vendor says their intervals are assumption-free. What do you say?"** Strong: names
exchangeability, notes that the literature from 2005 onward states it explicitly and generalises it to
other randomness models, and asks which calibration population and horizon the claim covers. Weak:
accepts the phrase.

**11. "How would you monitor a conformal interval in production?"** Strong: coverage by week and by
segment with binomial bands, a PIT histogram, a width path, and a break register, alerting when any
segment falls outside its band for two consecutive periods. Weak: "check coverage quarterly."

**12. "Conformal or conformalised quantile regression?"** Strong: split conformal for a homoscedastic
series and an auditable artefact; CQR when width must follow the predicted scale, accepting two
quantile fits and the crossing problem. Weak: picks one without stating the data condition.

## 15. Key terms

- **Conformal prediction** — a framework producing prediction sets with finite-sample coverage from a score and an exchangeability assumption.
- **Exchangeability** — a joint distribution unchanged by reordering its arguments.
- **Nonconformity score** — a scalar measuring how badly a prediction fits an observation.
- **Calibration set** — held-out data used only to compute scores and the threshold.
- **Proper training set** — data used to fit the point predictor.
- **Split conformal** — conformal with one fixed split into training and calibration.
- **Quantile index `k`** — `ceil((n + 1)(1 - alpha))`; the order statistic used as the threshold.
- **Order statistic** — the `k`-th smallest value of a sorted sample.
- **Marginal coverage** — coverage averaged over calibration sets and test points.
- **Conditional coverage** — coverage at a fixed feature value; unattainable distribution-free.
- **Local validity** — coverage within a pre-specified partition cell.
- **Coverage band** — the achievable range `[1 - alpha, 1 - alpha + 1/(n + 1)]`.
- **Feasibility condition** — `n >= 1/alpha - 1`, below which the interval is vacuous.
- **Interval width** — upper minus lower endpoint at a stated `x`.
- **Conformalized quantile regression (CQR)** — conformalising a pair of conditional quantile estimates.
- **Pinball loss** — the loss whose minimiser is a quantile.
- **Weighted conformal prediction** — conformal with weights on calibration scores, usually recency.
- **Adaptive conformal inference (ACI)** — online update of the working miscoverage level.
- **EnbPI** — ensemble batch prediction intervals; bootstrap ensembles plus a rolling residual pool.
- **Block conformal** — conformal on blocks of consecutive observations.
- **Mondrian conformal** — conformal run separately within each group.
- **Horizon-specific calibration** — a separate threshold per forecast horizon.
- **Probability integral transform (PIT)** — the predictive CDF evaluated at the outcome; uniform when calibrated.
- **Coverage curve** — empirical coverage against nominal level or against time.
- **Interval score** — a scoring rule penalising width and misses.
- **Regime shift** — a persistent change in the data-generating process.
- **Beta-mixing** — a dependence condition under which split conformal retains near-nominal coverage.

## 16. Sources

**Peer-reviewed and academic**

- Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. DOI: 10.1007/b106715. The canonical statement; chapter 8 generalises to on-line compression models including Markov. Second edition 2022. Grade A.
- Lei, J. and Wasserman, L. (2014). "Distribution-free prediction bands for non-parametric regression." *JRSS Series B*, 76(1), 71–96. DOI: 10.1111/rssb.12021. Exact conditional validity requires infinite-width bands. Grade A.
- Barber, R. F., Candès, E. J., Ramdas, A. and Tibshirani, R. J. (2021). "The limits of distribution-free conditional predictive inference." *Information and Inference*, 10(2), 455–482. DOI: 10.1093/imaiai/iaaa017. Grade A.
- Barber, R. F., Candès, E. J., Ramdas, A. and Tibshirani, R. J. (2023). "Conformal prediction beyond exchangeability." *Annals of Statistics*, 51(2). DOI: 10.1214/23-AOS2276. Weighted quantiles; coverage gap bounded by weighted total variation. Grade A.
- Gibbs, I. and Candès, E. J. (2021). "Adaptive conformal inference under distribution shift." *NeurIPS* 34. arXiv:2106.00170. Long-run coverage under arbitrary shift. Grade A.
- Xu, C. and Xie, Y. (2021). "Conformal prediction interval for dynamic time-series." *ICML*, PMLR 139, 11559–11569. EnbPI. Grade A.
- Romano, Y., Patterson, E. and Candès, E. J. (2019). "Conformalized quantile regression." *NeurIPS* 32. arXiv:1905.03222. Grade A.
- Chernozhukov, V., Wüthrich, K. and Zhu, Y. (2021). "Distributional conformal prediction." *PNAS*. DOI: 10.1073/pnas.2107794118. Grade A.
- Oliveira, R. I., Orenstein, P., Ramos, T. and Romano, J. V. (2024). "Split conformal prediction and non-exchangeable data." *JMLR*, 25, 1–38. Grade A.

**Preprints and benchmarks**

- "Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking." arXiv:2601.18509. Roughly 3,000 monthly sales series at 90% nominal; multi-step split conformal met coverage and was best on the interval score, while EnbPI and SPCI under-covered. Grade B.
- Barber, R. F. and Pananjady, A. (2025). "Predictive inference for time series: why is split conformal effective despite temporal dependence?" arXiv:2510.02471. Grade B.
- "A Gentle Introduction to Conformal Time Series Forecasting." arXiv:2511.13608. A review classifying the remedies into weighted, ensemble, adaptive, and block families. Grade B.

**Books and teaching material**

- Tibshirani, R. J. Lecture notes on conformal prediction, Carnegie Mellon and Stanford course material. The rank-uniformity derivation and the marginal-coverage warning. Grade B.

**Standards and official**

- ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1*. Grade B.
- ISO 5725-1:1994, *Accuracy (trueness and precision) of measurement methods and results — Part 1*. Grade B.
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in measurement*. Grade B.
- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time — Representations for information interchange*. Grade B.
- Board of Governors of the Federal Reserve System (2011). *Supervisory Guidance on Model Risk Management*, SR Letter 11-7. Grade B.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 15. Grade B.

**Practitioner, vendor, and reported cases**

- Public benchmark repository comparing exponential smoothing conformal intervals, a gradient-boosted model, and a foundation model on store-item and M5 samples: 65.25% coverage against an 80% target for the conformal residual intervals. Unreviewed notebook. Grade E.
- "Stability-Aware Hierarchical Forecasting: Synergizing Conformal Prediction with Decomposition Ensembles": coverage near 0.38 to 0.40 against an 80% target on M5 with pooled centered CQR. Journal review status not established. Grade C.
- Applied write-up of a neural base model with conformal intervals on M5: 96.5% marginal coverage at 95% nominal, around 91% on smooth and erratic series. Grade D.
- Library documentation for MAPIE, Fortuna, PUNCC, and a Julia implementation with a time-series guide. Grade D for method claims; useful for what the defaults are.

> **Note:** Findings are reported as the cited works state them; the benchmarks in section 9.3 are
> graded accordingly and none is an estimate of an industry-wide rate. No citation here is invented.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Previous topic, on the distributions and calibration this builds on: [`07_probabilistic-forecasts.md`](07_probabilistic-forecasts.md)
- Next topic, on scoring the intervals this produces: [`09_scoring-rules.md`](09_scoring-rules.md)
- Later, on choosing the quantile from the decision: [`10_quantiles-decisions.md`](10_quantiles-decisions.md)
- Intermittent-demand methods that make the score awkward: [`05_intermittent-classification.md`](05_intermittent-classification.md) and [`06_compound-poisson-hurdle.md`](06_compound-poisson-hurdle.md)
- The evaluation design this topic depends on: [`../07_time_series_forecasting/08_rolling-origin-backtesting.md`](../07_time_series_forecasting/08_rolling-origin-backtesting.md)
- The metrics that score an interval rather than a point: [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Probability, sampling, and the bootstrap behind the calibration argument: [`../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md`](../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md)
- Intermittent-demand metrics and their denominators: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Demand and lead-time distributions, which set the horizon: [`../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Calendars and week numbering, which decide where the split falls: [`../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- This part's map: [`../_index.md`](../_index.md)
- Module survey file: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
