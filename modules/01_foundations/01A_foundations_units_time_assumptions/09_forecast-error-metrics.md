> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 09 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01, 01A.08

# Forecast Error Metrics

---

## 1. Why this topic exists

Two analysts evaluate the same eight weeks of the same forecast. The first reports a mean absolute
percentage error (MAPE) of 22.1%. The second reports a mean absolute scaled error (MASE) of 0.81.
Both numbers are correct. A third reports "forecast accuracy of 85.4%", which is also correct and
is a different quantity again. The meeting concludes that the forecast is acceptable, and nobody
asks which model was chosen to produce it.

That last omission is the subject of this topic. A forecast error metric is not a neutral summary
of a forecast. It is an objective function. Whatever you measure is what the model, the planner,
and the override process will optimise, whether or not anyone intended it. Change the metric and
you change the model that wins, the bias the forecast carries, the inventory the business holds,
and the customers who run short.

Four independent choices are buried inside every metric, and each has a cost:

1. **What it weights.** Mean absolute error (MAE) weights every unit equally. Root mean squared
   error (RMSE) weights large misses quadratically. MAPE weights small-actual weeks heavily. MASE
   weights each series by how hard its naive benchmark is to beat.
2. **What scale it lives on.** Units, percent, or a dimensionless ratio against a benchmark. An
   MAE in eaches cannot be averaged across a product family; a MASE can, but the average means
   something specific and slightly strange.
3. **What it does at zero and near zero.** Percentage measures divide by the actual. At zero they
   are undefined; near zero they explode. Many real item portfolios have zeros in most weeks.
4. **Where it is aggregated.** Item-location-week is where replenishment is decided;
   category-country-month is where the report is read. Errors cancel across items, so the
   aggregate nearly always looks better than the decision grain.

There is a fifth, deeper point. **A point forecast is a one-number summary of a distribution, and
the metric decides which summary you get.** Squared error is minimised by the mean, absolute error
by the median. For right-skewed demand the mean sits above the median, so a model trained on RMSE
forecasts a higher level than the same model trained on MAE. That is several days of inventory
decided by a line in an evaluation script nobody read.

Topic 01A.08 supplied the vocabulary of measurement error: bias versus random error, tolerance
versus uncertainty, resolution. Here the "true value" is a future observation, the "instrument" is
a model plus a planner, and the "tolerance" is set by an inventory policy. Everything that made
measurement hard in Topic 08 is present, plus a feedback loop: the metric you publish changes the
behaviour that produces the next forecast.

**A 4.5-point accuracy gain that lost service.** The following composite is constructed to show
the mechanism, not drawn from a published case. A distributor replaced its statistical engine.
Aggregate forecast accuracy, defined as 100% minus the volume-weighted absolute percentage error,
rose from 81.2% to 85.7%. Item-level fill rate fell by three points. The new engine had been
tuned on MAPE at item level, and MAPE is dominated by low-volume items, where a one-unit miss is
a large percentage. The engine learned to under-forecast slow movers. Aggregate accuracy, weighted
by volume, barely noticed, because slow movers are a small share of units. The customers who
ordered them noticed.

> **Note:** Nothing in this topic says one metric is correct. It says that a metric is a claim
> about consequences, and that the claim should be written down before the results are seen.
> A metric chosen after looking at the leaderboard is not a measurement. It is a selection
> procedure with the answer built in.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** the formula for MAE, MSE, RMSE, MAPE, symmetric MAPE (sMAPE), MASE, bias, and the
   tracking signal, stating the unit of each and the sign convention.
2. **Predict** which of two forecasts will win under each metric given their error vectors, and
   **explain** the mechanism that produces the ranking flip.
3. **Identify** the functional of the predictive distribution that a given loss implies — mean,
   median, or a quantile — and **state** the operational consequence of the difference on skewed
   demand.
4. **Diagnose** why a percentage-based metric is undefined or misleading on a series containing
   zeros, and **select** a scaled alternative with a stated denominator.
5. **Compute** a tracking signal, **interpret** it against a stated threshold, and **explain** why
   the threshold is a convention rather than a probability statement.
6. **Quantify** the effect of asymmetric over- and under-forecast costs on the preferred model,
   using a critical ratio, and **show** where the preference reverses.
7. **Reconcile** a metric computed at item-location-week with the same metric at category-month,
   and **explain** the direction and size of the difference.
8. **Write** a metric specification that another analyst could implement without asking you a
   question, and **audit** an existing specification against that standard.

## 3. The core mental model

Hold one picture: a vector of errors, and a set of lenses held over it. The errors do not change.
The ranking does.

```text
   SAME EIGHT WEEKS, TWO FORECASTS, FIVE METRICS

   week            1    2    3    4    5    6    7    8
   actual        120   60  200   40  160   30  180  100     (eaches)
   error A        +5   -5  +10  -15  +10  -15  +10  +60
   error B       -18  -18  +20  -20  +18  -20  +18  -18
                                                    ^
                                        A's one large miss

   metric      A        B       winner    reacts to
   --------   ------   ------   ------    ---------------------------
   MAE         16.25    18.75     A       every unit equally
   RMSE        23.45    18.77     B       the square of every unit
   MAPE       22.10%   26.36%     A       percent error, small weeks
   sMAPE      23.36%   22.44%     B       percent error, capped at 200%
   MASE        0.81     0.94      A       error relative to the naive benchmark

   A wins three, B wins two. Not one forecast value changed.
```

The reason is visible in the error row. Method A is close on seven weeks and wrong by 60 on one.
Method B is moderately wrong every week. MAE, which is linear, prefers the method with the smaller
total absolute error. RMSE, which squares, prefers the method whose errors are spread evenly. No
subtlety is required to see this; what is required is to look at the error vector before looking
at the score.

Every metric answers four questions, by design or by accident: **scale** (units, percent, or
scaled against a benchmark), **penalty** (linear or quadratic), **denominator** (the actual, the
naive error, or none), and **grain** (item-week or category-month). Section 5 works through each.
Five properties follow, and they explain the rest of the topic.

**Property 1 — The metric chooses the functional, not just the score.** A point forecast is a
single number standing in for a distribution. Which number it should be depends on the loss.
Under squared error the optimal point is the mean; under absolute error it is the median; under
a newsvendor cost it is a quantile set by the critical ratio. Gneiting (2011) makes this formal:
a scoring function is consistent for the mean exactly when it is a Bregman function, and for a
quantile exactly when it is a generalised piecewise-linear function. Gneiting and Raftery (2007)
show that the continuous ranked probability score (CRPS) reduces to absolute error when the
forecast is a point measure, which is the bridge between distributional and point evaluation.
The practical version: **if demand is right-skewed, an RMSE-trained model forecasts higher than
an MAE-trained one, and holds more inventory.** That is the single most consequential sentence in
this topic.

**Property 2 — Every metric has denominator politics.** Units-based metrics cannot be compared
across products of different size. Percentage metrics can, but divide by a random quantity that
is sometimes near zero. Scaled metrics divide by a benchmark's error, which is stable but makes
the result a ratio of ratios, with the statistical behaviour that implies. There is no free
choice; there is only a choice whose failure mode you have accepted and written down.

**Property 3 — Sign carries information that magnitude does not.** A forecast that is wrong by
20 units in both directions is not the same as a forecast that is wrong by 40 one way and right
the other. Bias and the tracking signal measure direction. MAE and RMSE are blind to it. A
forecast can have a respectable MAE and a bias that empties a shelf.

**Property 4 — Aggregation is not neutral.** Errors cancel when you add them. The MAE of the
total is at most the sum of the MAEs of the parts, and usually much less. A metric reported at
category level is therefore a statement about net error, not about the errors that individual
replenishment decisions experience. Report at the grain where the decision is made, and report
the aggregate separately with an explicit statement of what it hides.

**Property 5 — Value is defined against a benchmark.** A MASE of 0.94 means the forecast beat
the naive one-step forecast by 6%. A MAPE of 22% means nothing until someone says what a naive
forecast would have scored on the same weeks. Forecast value added (FVA) makes this explicit:
the value of a forecasting step is the improvement it delivers over the cheapest available
alternative, which is usually last period's actual or the same period last year.

The five properties combine into one working rule. **Before comparing two forecasts, plot the
errors; before choosing a metric, name the decision; before publishing a number, name the
benchmark.** Every failure mode in section 10 is a violation of one of those three.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Forecast error | The difference between an observed value and the forecast for the same period | Called "accuracy" | Same unit as the series |
| Signed error | `e_t = y_t − F_t`, positive when the actual exceeds the forecast | Sign convention left unstated | Units |
| Forecast origin | The time at which the forecast is made and beyond which no data is used | Confused with the period being forecast | Timestamp or period index |
| Forecast horizon | Number of periods between origin and target | Reported without the origin | Periods |
| Forecast vintage | The version of a forecast as it stood at a given origin, before later revision | Overwritten in the database | Origin timestamp |
| Out-of-sample | Evaluation on observations not used to fit the model | Applied to a holdout reused for tuning | Boolean |
| Rolling-origin evaluation | Repeatedly moving the origin forward, re-fitting, and scoring each horizon | Replaced by a single split | Procedure identifier |
| Forecast accuracy | Closeness of forecasts to actuals; in practice always one minus some error measure | Treated as a single defined quantity | Proportion or percent, definition-dependent |
| Forecast bias | Mean signed error over a stated window; the systematic component | Used for any error | Units, or percent of mean actual |
| Mean absolute error (MAE) | `(1/n) Σ |e_t|` | Compared across products of different scale | Units |
| Mean squared error (MSE) | `(1/n) Σ e_t²` | Quoted alongside unit-scale metrics | Units squared |
| Root mean squared error (RMSE) | Square root of MSE | Treated as "the same as MAE but better" | Units |
| Mean absolute percentage error (MAPE) | `(100/n) Σ |e_t| / y_t` | Used when actuals can be zero or negative | Percent |
| Symmetric MAPE (sMAPE) | `(100/n) Σ 2|e_t| / (|y_t| + |F_t|)` | Assumed symmetric because of the name | Percent, 0 to 200 |
| Mean absolute scaled error (MASE) | MAE divided by the in-sample MAE of the naive one-step forecast | Denominator computed on the test window | Dimensionless |
| Root mean squared scaled error (RMSSE) | RMSE divided by the in-sample RMSE of the naive one-step forecast | Confused with MASE | Dimensionless |
| Weighted absolute percentage error (WAPE) | `Σ|e_t| / Σ y_t`, equivalent to volume-weighted MAPE | Called MAPE | Proportion or percent |
| Average relative MAE (AvgRelMAE) | Geometric mean of the ratio of a method's MAE to a benchmark's, across series | Treated as identical to MASE | Dimensionless |
| Tracking signal (TS) | Cumulative signed error divided by mean absolute deviation | Reported without the window length | Dimensionless ratio |
| Running sum of forecast errors (RSFE) | Cumulative `Σ e_t` over the tracking window; the numerator of TS | Reset without disclosure | Units |
| Mean absolute deviation (MAD) | `(1/n) Σ |e_t|` over the tracking window; the denominator of TS | Confused with standard deviation | Units |
| Trip value | The threshold at which a tracking signal is judged to indicate bias | Treated as a significance level | Dimensionless |
| Naive forecast | Forecast equal to the most recent observation | Assumed to be a weak straw man | Units |
| Pinball loss | `q·e` if `e ≥ 0`, `(q−1)·e` otherwise, for quantile level q | Unknown outside specialist teams | Units |
| Quantile forecast | A forecast of the value exceeded with probability `1 − q` | Treated as a point forecast with a fudge factor | Units at level q |
| Proper scoring rule | A score whose expectation is optimised by reporting the true distribution | Applied to point forecasts without qualification | Depends on the score |
| Continuous ranked probability score (CRPS) | An integral of the quantile loss over all levels; a proper score for a predictive distribution | Used when only a point forecast exists | Units |
| Forecast value added (FVA) | Change in a performance metric attributable to one step in the forecasting process | Reported without the naive baseline | Percent improvement, or metric difference |
| Intermittent demand | Demand with periods of zero between nonzero observations | Treated as a small-numbers problem | Units per period |
| Average inter-demand interval (ADI) | Mean number of periods between nonzero demands | Computed including leading zeros inconsistently | Periods |
| Squared coefficient of variation (CV²) | Variance of nonzero demand sizes divided by their squared mean | Computed on all periods, including zeros | Dimensionless |
| Aggregation grain | The level at which a metric is computed: item, location, week, customer | Left implicit | Key definition |

Three of these are routinely conflated and repay a sentence each. **MAPE, WAPE, and "accuracy"**
answer different questions: MAPE is the unweighted mean of per-period percentages, so a week with
actual 30 and error 15 contributes 50% while representing 3% of volume; WAPE divides total absolute
error by total actual and is therefore volume-weighted; "forecast accuracy" is usually `1 − WAPE`
in consumer goods and `100 − MAPE` in older systems, so two companies can both report 85% and mean
different things. **Bias and the tracking signal** differ in kind: bias is a mean with the units of
the series and answers "by how much, on average, and in which direction"; the tracking signal is a
ratio and answers "is the cumulative error large relative to the typical miss". Neither is a
probability. **Proper scoring rules** apply to distributions; MAE and squared error take a point
forecast as input and are consistent scoring functions for particular functionals, which is why a
model can have an excellent MAE and a badly calibrated predictive distribution.

## 5. The mechanics, step by step

This section builds a metric from the error definition outward: first the anatomy of one error,
then the four design decisions, then each family, then bias and the cost asymmetry, then grain,
then the effect on model selection, and finally the specification that survives handover.

### 5.1 The anatomy of a forecast error

**Step 1. Fix the origin.** A forecast is indexed by the time it was made, `T`, and the target
period, `T + h`. Writing `F_{T+h|T}` makes both explicit. A file of forecasts with no origin
column cannot be evaluated, because there is no way to know what information was available.

**Step 2. Fix the vintage.** Forecasts are revised. The value used in the Monday planning meeting
is not the value in the system on Friday. Evaluation must use the vintage that was acted upon,
or it measures a forecast nobody used.

**Step 3. Fix the actual.** The actual is itself a measurement, subject to the errors described in
Topic 01A.08, and it may be restated later. State whether the evaluation uses the first published
actual, the latest restated actual, or a reconciled figure.

**Step 4. Fix the sign and the horizon.** Use `e = actual − forecast` throughout. Positive means
the actual came in above the forecast, which is an under-forecast. The opposite convention is
equally defensible and produces opposite-signed bias; mixing them within one report is not. A
one-step-ahead error and a twelve-week-ahead error are different measurements, and pooling them
produces a metric that describes no decision.

> **Note:** Most "forecast accuracy" disputes are not disputes about models. They are disputes
> about origin, vintage, actual, sign, or horizon. Resolve those five before computing anything.

### 5.2 The scale-dependent family: MAE, MSE, RMSE, MdAE

These live in the units of the series, or in squared units for MSE.

| Metric | Formula | Unit | Behaviour |
|---|---|---|---|
| MAE | `(1/n) Σ |e_t|` | Units | Linear penalty; robust to outliers; optimised by the median |
| MSE | `(1/n) Σ e_t²` | Units squared | Quadratic penalty; dominated by the largest error |
| RMSE | `sqrt(MSE)` | Units | Same ranking as MSE; readable unit |
| MdAE | `median(|e_t|)` | Units | Ignores the tail entirely; useful for a "typical week" statement |

Two practical consequences follow.

**RMSE is not "MAE but stricter".** It is a different loss with a different target. Because the
square weights large deviations heavily, the minimiser is the conditional mean. On a right-skewed
demand distribution the mean is above the median, so an RMSE-optimised forecast sits higher. That
is a level shift, not an accuracy improvement, and it is invisible in the RMSE number itself.
Neither measure can be averaged across a product family: a MAE of 12 eaches means something for a
product selling 40 a week and nothing for one selling 4,000.

### 5.3 The percentage family: MAPE, sMAPE, WAPE

Percentage errors normalise by the actual, which makes them comparable across scale and unstable
near zero.

| Metric | Formula | Range | Key property |
|---|---|---|---|
| MAPE | `(100/n) Σ |e_t| / y_t` | 0 to ∞ | Asymmetric; undefined at `y_t = 0`; punishes over-forecast more when actual is small |
| sMAPE | `(100/n) Σ 2|e_t| / (|y_t| + |F_t|)` | 0 to 200 | Bounded; still asymmetric; undefined when both terms are zero |
| WAPE | `100 · Σ|e_t| / Σ y_t` | 0 to ∞ | Volume weighted; equals weighted MAPE; undefined when total actual is zero |

**MAPE's asymmetry.** For a fixed absolute error, the percentage is larger when the actual is
small, so over-forecasting a low-volume item costs more in MAPE than under-forecasting it by the
same amount. A model tuned on MAPE therefore learns to keep forecasts low on small-volume items.
This follows directly from dividing by `y_t`, and it biases MAPE-optimised forecasts toward
under-forecasting exactly where intermittent items live.

**sMAPE is not symmetric.** Goodwin and Lawton (1999) showed that the measure proposed to fix
MAPE's asymmetry is itself far from symmetric, particularly for large absolute errors, and that
under some circumstances the relationship between sMAPE and absolute error is non-monotonic —
a worse forecast can score better. The worked example in section 6 produces exactly this pattern:
a 60-unit under-forecast is penalised at 85.7% by sMAPE against 60.0% by MAPE, while a 15-unit
over-forecast is penalised at 40.0% against 50.0%. The formula looks balanced because the
denominator contains both actual and forecast; the penalty is not.

> **Note:** sMAPE is bounded at 200%, often cited as a virtue. The bound is reached when the
> forecast is zero and the actual is positive, or the reverse, and a metric that saturates stops
> distinguishing a bad forecast from a catastrophic one.

**WAPE is the volume-weighted version and is frequently what people mean.** Weighting each
period's percentage error by its actual volume collapses to `Σ|e_t| / Σ y_t`, so WAPE has no
zero-division problem unless the total is zero and is dominated by high-volume periods. Its
weakness is the mirror of MAPE's strength: it can be excellent while many low-volume items are
badly wrong.

### 5.4 The scaled family: MASE and RMSSE

Scaled errors divide by the error a trivial benchmark would have made on the same series, in the
same in-sample period.

```text
   scale  = (1/(n-1)) * SUM over t=2..n of | y_t - y_(t-1) |      (naive, non-seasonal)

   MASE   = MAE_forecast / scale
   RMSSE  = RMSE_forecast / sqrt( (1/(n-1)) * SUM ( y_t - y_(t-1) )^2 )
```

Four properties matter.

**Interpretation is built in.** MASE below 1 means the forecast beats the naive one-step
forecast on the same history; above 1 means it is worse than a forecast that says "same as last
week". That is a genuinely useful zero point, and it is why MASE is preferred to MAPE in
situations where the benchmark matters.

**The denominator must come from the training period.** Computing it on the evaluation window
inflates or deflates the score depending on how volatile that window happened to be. This is the
most common implementation error.

**The denominator assumes the differenced series is stationary.** Hyndman and Koehler (2006) note
that scaling by the in-sample naive error is valid when the series has at most one unit root.
For strongly trending or seasonal series, the seasonal naive difference is the appropriate scale,
and the choice must be stated.

**Averaging MASE across series is not a simple average of skill.** Davydenko and Fildes (2016)
show that MASE equals the weighted arithmetic mean of relative MAEs across series. Arithmetic
averaging of ratios penalises a bad forecast more than it rewards a good one, so the aggregate is
pulled toward the benchmark: relative MAEs of 1/2 and 2/1 average to 1.25, implying the benchmark
beats a method that was twice as good on one series and half as good on another. They recommend a
geometric mean of relative MAEs (AvgRelMAE) with trimming.

### 5.5 Bias and the tracking signal

Bias is the mean signed error. It has units, it cancels across directions, and it is the only
common metric that distinguishes "wrong both ways" from "wrong one way".

```text
   bias  = (1/n) * SUM e_t
   RSFE  = SUM e_t                      over the tracking window
   MAD   = (1/n) * SUM |e_t|            over the tracking window
   TS    = RSFE / MAD
```

Interpretation is conventional. A tracking signal near zero indicates no persistent direction.
A large positive value indicates the actual has been consistently above the forecast
(under-forecast); a large negative value indicates over-forecast. The APICS/ASCM practitioner
literature uses a trip value of about ±4, with ±3 and ±6 also in use, and notes that for a normal
error distribution the standard deviation is roughly 1.25 times the MAD.

> **Note:** The ±4 trip value is a convention, not a significance test. If errors were
> independent and unbiased, `RSFE` would have standard deviation `sqrt(n)·σ` and the tracking
> signal would have standard deviation roughly `1.25·sqrt(n)` when MAD estimates `0.8σ`. At
> `n = 8` that is about 3.5, so a ±4 threshold is roughly a 1.1-sigma rule with a two-sided false
> alarm rate near 26%. At `n = 100` the standard deviation is about 12.5 and a ±4 threshold trips
> constantly. The threshold's meaning depends entirely on the window length, which is why the
> window must be stated alongside the number.

Three practical rules follow. Use a rolling window of fixed length (twelve periods is common) so
the statistic's variance does not grow with the history. Report bias in units and as a percentage
of mean actual. And plot the cumulative error, because a signal that crosses zero regularly is
behaving differently from one that drifts steadily, and the ratio alone cannot tell you which.

### 5.6 Asymmetric loss and the critical ratio

Symmetric metrics assume a unit over-forecast costs the same as a unit under-forecast. Almost no
supply chain decision satisfies that assumption. Over-forecasting perishable goods costs the
margin plus disposal; under-forecasting costs the margin plus goodwill plus, sometimes, a
contractual penalty. The two are not equal, and the metric should not pretend they are.

The standard formulation is the newsvendor loss:

```text
   loss(a, y) = C_u * max(y - a, 0) + C_o * max(a - y, 0)

   where a is the forecast (the quantity committed), y is the actual,
   C_u is the cost of one unit short, C_o is the cost of one unit over.

   optimal a* is the q-th quantile of the demand distribution,
   q = C_u / (C_u + C_o)
```

When `C_u = C_o`, the critical ratio is 0.5 and the optimal forecast is the median, which is what
MAE targets. When `C_u > C_o`, the optimal forecast sits above the median and the right metric is
the pinball loss at level `q`, not MAE. When `C_o > C_u`, it sits below. The pinball loss at level
`q` is the scoring function that is consistent for the `q`-quantile, and its average over all
levels is the CRPS.

The consequence for metric choice is direct. **A symmetric metric is a claim that the two errors
cost the same.** If that claim is false, the model selected by the symmetric metric is
systematically wrong in the direction the business cares about, and it will look good on the
dashboard while doing so. Section 6 quantifies this on the worked example: the model with the
better MAE has 22% higher asymmetric cost at a critical ratio of 0.75.

### 5.7 Evaluation grain and aggregation

Metrics are computed at a grain, and the grain is a decision.

| Grain | Typical use | What it reveals | What it hides |
|---|---|---|---|
| Item–location–week | Replenishment, safety stock | The errors that trigger orders | Nothing about portfolio performance |
| Item–week | Purchasing, production scheduling | Item-level model quality | Location-specific behaviour |
| Category–month | S&OP, financial forecast | Net error after cancellation | Item-level failure entirely |
| Customer–order | Service and allocation | Promise reliability | Supply-side causes |
| Total business–quarter | Board reporting | Direction and scale of the aggregate miss | Almost everything operational |

Three mechanisms produce the gap between grains.

**Cancellation.** Positive and negative errors offset when summed. `|Σ e| ≤ Σ |e|`, so the
aggregate absolute error is at most the sum of the item absolute errors and typically far smaller.
An aggregate WAPE of 4% can coexist with item-level WAPEs averaging 30%. Both are correct. The
aggregate answers "did we buy roughly the right total"; the item level answers "did we buy the
right things".

**Weighting.** Averaging per-item percentage errors gives each item equal weight, so 900 slow
movers outvote 100 fast movers. Averaging with volume or revenue weights reverses the result.
Neither is more truthful; they answer different questions. The M5 competition made this explicit
by weighting its hierarchical metric by cumulative dollar sales, so that error on high-revenue
series counted more (Makridakis et al., 2022).

**Reconciliation.** Hierarchical forecasts are usually reconciled so that item forecasts sum to
the category forecast, which changes the item-level errors and the aggregate error by
construction. A metric computed before reconciliation is not comparable with one computed after.

> **Note:** Report the metric at the grain where the decision is taken, and report the aggregate
> beside it with a sentence naming what it conceals. Whoever chooses the grain chooses the
> conclusion.

### 5.8 How the choice changes the model selected

This is the payoff of the whole topic. The metric is not a passive scorecard; it selects.

| Metric | What the optimiser learns | Model that tends to win | Operational consequence |
|---|---|---|---|
| MAE | Fit the conditional median | Robust level models, damped trends | Lower forecast level on skewed demand; leaner inventory; more shortfall risk |
| RMSE | Fit the conditional mean | Models that chase large deviations | Higher forecast level on skewed demand; more inventory; fewer shortfalls |
| MAPE | Minimise percentage error, weighted toward small actuals | Models that under-forecast slow movers | C-item service failure hidden by good aggregate accuracy |
| sMAPE | Similar, with a cap and a different asymmetry | Models that avoid extreme under-forecast | Intermediate, but the asymmetry is undocumented and easy to misread |
| MASE | Beat the naive benchmark across series | Simple, robust models; complex models are penalised when they do not help | Avoids sophistication that adds no value |
| WAPE | Minimise error weighted by volume | Models tuned to A items | Good A-item service; C items can be arbitrarily bad |
| Pinball at q | Fit the q-quantile | Quantile regression, gradient boosting with quantile loss | Matches the service target directly; requires stating q |

Two derived observations are worth carrying into practice.

First, **a change of metric is a change of model, even when nobody retrains anything.** Switching
an evaluation dashboard from MAPE to WAPE changes which planners' overrides look good, which
changes override behaviour, which changes the forecast. The metric is an incentive.

Second, **the metric must match the decision's asymmetry.** MAE is the right default when the cost
of a unit of error is roughly flat. It is the wrong default for a perishable with a short shelf
life or a spare part with a long downtime cost, where the critical ratio is far from 0.5 and the
honest metric is a quantile loss.

### 5.9 The metric specification that survives handover

A metric definition that lives in one analyst's head does not survive a reorganisation. The
specification below is the minimum that lets a successor reproduce the number without asking a
question.

| Field | What to write | Example |
|---|---|---|
| Name and version | Exact name, version, effective date | `wape_item_week v3, 2026-01-01` |
| Formula | The formula, with the summation index and the sign convention | `100 · Σ|y−F| / Σ y`, `e = y − F` |
| Unit | Units, percent, or dimensionless | Percent |
| Grain | The keys over which the metric is computed | SKU × location × ISO week |
| Horizon | Steps ahead, pooled or per-step | 1 to 4 weeks, reported separately |
| Origin and vintage | Which forecast snapshot is scored | Thursday 06:00 run, frozen |
| Actual definition | Which version of the actual, and when it is considered final | First posted actual, restated after 4 weeks |
| Population | Included and excluded series | Excludes new items with < 8 weeks history |
| Zero and null handling | Explicit rule | Series with zero total actual excluded, count reported |
| Weighting | Equal, volume, revenue, or margin | Volume, in eaches |
| Benchmark | The naive comparator and its exact definition | Seasonal naive, 52-week lag |
| Threshold | Trip value and window for bias | TS ±4, rolling 12 weeks |
| Owner | Role, not person | Demand planning analytics lead |
| Change control | Who may change it and how the change is disclosed | Versioned; restatement note in the monthly pack |

The specification is the difference between a metric and an opinion. Two competent analysts given
this document produce the same number to the last decimal; given a dashboard labelled "forecast
accuracy", they produce two numbers and an argument.

## 6. Worked example

### 6.1 Setting

A distributor forecasts weekly demand for one stock-keeping unit (SKU) at one distribution centre.
Demand is in eaches. Two candidate models are on the table.

- **Method A** is a smoothed level model. It tracks the series closely and occasionally misses a
  turning point badly.
- **Method B** is a more responsive model with a shorter memory. It moves earlier but never gets
  very close.

The evaluation window is eight weeks. The in-sample naive one-step mean absolute error, computed
on the preceding thirteen weeks of history, is 20.0 eaches per week. That is the MASE denominator,
and it is fixed before the evaluation window is seen.

### 6.2 The data

| Week | Actual (eaches) | Forecast A | Error A | Forecast B | Error B |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 115 | +5 | 138 | −18 |
| 2 | 60 | 65 | −5 | 78 | −18 |
| 3 | 200 | 190 | +10 | 180 | +20 |
| 4 | 40 | 55 | −15 | 60 | −20 |
| 5 | 160 | 150 | +10 | 142 | +18 |
| 6 | 30 | 45 | −15 | 50 | −20 |
| 7 | 180 | 170 | +10 | 162 | +18 |
| 8 | 100 | 40 | +60 | 118 | −18 |
| **Total** | **890** | | **+60** | | **−38** |

The error vectors tell most of the story before any metric is computed. Method A is within 15
eaches on seven weeks and misses week 8 by 60. Method B is never within 15 and never worse than
20. A is a sharp model with one failure; B is a blunt model with none. Which is a virtue depends
entirely on what a failure costs.

### 6.3 The metrics, computed by hand

**Mean absolute error.** Sum the absolute errors and divide by eight.

| Method | Sum of absolute errors | MAE |
|---|---:|---:|
| A | 5 + 5 + 10 + 15 + 10 + 15 + 10 + 60 = 130 | 130 / 8 = **16.25 eaches** |
| B | 18 + 18 + 20 + 20 + 18 + 20 + 18 + 18 = 150 | 150 / 8 = **18.75 eaches** |

**Mean squared error and root mean squared error.** Square each error, sum, divide, then root.

| Method | Sum of squared errors | MSE | RMSE |
|---|---:|---:|---:|
| A | 25 + 25 + 100 + 225 + 100 + 225 + 100 + 3,600 = 4,400 | 550.0 | 23.45 eaches |
| B | 324 + 324 + 400 + 400 + 324 + 400 + 324 + 324 = 2,820 | 352.5 | 18.77 eaches |

The flip is already visible. A's single 60-unit error contributes 3,600 of its 4,400 squared
units — 82% of its MSE comes from one week. Under a linear penalty that week contributes 60 of
130, or 46%. Squaring converts one bad week into a verdict.

**Mean absolute percentage error.** Divide each absolute error by its actual, average, multiply
by 100.

| Week | Actual | Error A | MAPE term A | Error B | MAPE term B |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 5 | 4.17% | 18 | 15.00% |
| 2 | 60 | 5 | 8.33% | 18 | 30.00% |
| 3 | 200 | 10 | 5.00% | 20 | 10.00% |
| 4 | 40 | 15 | 37.50% | 20 | 50.00% |
| 5 | 160 | 10 | 6.25% | 18 | 11.25% |
| 6 | 30 | 15 | 50.00% | 20 | 66.67% |
| 7 | 180 | 10 | 5.56% | 18 | 10.00% |
| 8 | 100 | 60 | 60.00% | 18 | 18.00% |
| **Sum** | | | **176.81%** | | **210.92%** |
| **Mean** | | | **22.10%** | | **26.36%** |

Note what happened to week 6: an error of 15 on an actual of 30 is 50%, the same weight as an
error of 60 on an actual of 120. A week representing 3.4% of volume carries the same weight as a
week representing 13.5%. That is the denominator problem in one line.

**Symmetric MAPE.** Divide twice the absolute error by the sum of actual and forecast.

| Week | Actual | F A | sMAPE term A | F B | sMAPE term B |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 115 | 10/235 = 4.26% | 138 | 36/258 = 13.95% |
| 2 | 60 | 65 | 10/125 = 8.00% | 78 | 36/138 = 26.09% |
| 3 | 200 | 190 | 20/390 = 5.13% | 180 | 40/380 = 10.53% |
| 4 | 40 | 55 | 30/95 = 31.58% | 60 | 40/100 = 40.00% |
| 5 | 160 | 150 | 20/310 = 6.45% | 142 | 36/302 = 11.92% |
| 6 | 30 | 45 | 30/75 = 40.00% | 50 | 40/80 = 50.00% |
| 7 | 180 | 170 | 20/350 = 5.71% | 162 | 36/342 = 10.53% |
| 8 | 100 | 40 | 120/140 = 85.71% | 118 | 36/218 = 16.51% |
| **Sum** | | | **186.84%** | | **179.53%** |
| **Mean** | | | **23.36%** | | **22.44%** |

The ranking flips against MAPE, and the mechanism is the asymmetry. In week 8, A under-forecasts
by 60 on an actual of 100: MAPE charges 60.0%, sMAPE charges 85.7%. In week 6, A over-forecasts
by 15 on an actual of 30: MAPE charges 50.0%, sMAPE charges 40.0%. sMAPE penalises the
under-forecast harder and the over-forecast more lightly. Since A's largest error is an
under-forecast, sMAPE moves against it. This is the Goodwin–Lawton result made concrete.

**MASE.** Divide each MAE by the in-sample naive scale of 20.0 eaches.

| Method | MAE | Scale | MASE | Reading |
|---|---:|---:|---:|---|
| A | 16.25 | 20.0 | **0.8125** | 19% better than naive |
| B | 18.75 | 20.0 | **0.9375** | 6% better than naive |

Both beat the naive benchmark, which is the minimum acceptable result. A model with MASE above
1 has consumed effort and made the forecast worse than "same as last week", and that fact is
worth more than any percentage improvement in MAPE.

**Bias and tracking signal.** Sum the signed errors; divide by eight for bias, divide by MAD for
the tracking signal. MAD is the mean absolute deviation, which here equals the MAE.

| Method | Sum of errors | Bias (eaches/week) | MAD | Tracking signal |
|---|---:|---:|---:|---:|
| A | +60 | +7.50 | 16.25 | +3.69 |
| B | −38 | −4.75 | 18.75 | −2.03 |

Method A under-forecasts by 7.5 eaches per week on a mean actual of 111.25, which is 6.7%. Method
B over-forecasts by 4.75, or 4.3%. A's tracking signal of 3.69 sits just inside the conventional
±4 trip value; B's is comfortably inside. Read the pair together: A has the better average miss
and the worse directional error. That combination is common and is exactly what a single
accuracy number cannot express.

**WAPE, and the volume-weighted view.**

| Method | Sum absolute errors | Sum actual | WAPE | Implied "accuracy" |
|---|---:|---:|---:|---:|
| A | 130 | 890 | 14.61% | 85.39% |
| B | 150 | 890 | 16.85% | 83.15% |

WAPE is far below MAPE for both methods — 14.6% against 22.1% for A. The difference is not an
error in either calculation. MAPE weights week 6 (actual 30) as heavily as week 3 (actual 200);
WAPE weights week 3 nearly seven times more. The 22.1% describes the typical week's percentage
miss. The 14.6% describes the total units miss. A business that reports "85% forecast accuracy"
is usually reporting the second while believing it is reporting the first.

### 6.4 The ranking flip, assembled

Five metrics, two winners: A takes MAE, MAPE, and MASE; B takes RMSE and sMAPE. A's seven good
weeks outweigh one bad one under a linear penalty, but its 60-unit miss is 82% of its MSE, so the
quadratic penalty reverses the verdict. MAPE and sMAPE split for a different reason — sMAPE
penalises A's large under-forecast harder than MAPE does. Bias and the tracking signal do not rank
the methods at all; they describe direction, and A is the one near the conventional trip value.
No forecast value changed. Any statement that "model A is more accurate" is therefore incomplete
without the metric, and any metric choice is incomplete without the decision.

### 6.5 The cost of the choice

Suppose the SKU is a seasonal item with a nine-week shelf life. A unit left over at the end of
the season costs €3 in markdown and disposal. A unit short costs €9 in lost margin and expediting.
The critical ratio is `9 / (9 + 3) = 0.75`, so the cost-minimising forecast is the 75th percentile
of demand, not the median.

Score both methods under this asymmetric loss, with `e = actual − forecast`, so positive errors
are shortfalls:

| Method | Sum of positive errors (short) | Sum of negative errors (over) | Loss per week | Rank |
|---|---:|---:|---:|---|
| A | 5 + 10 + 10 + 10 + 60 = 95 | 5 + 15 + 15 = 35 | (9·95 + 3·35)/8 = 960/8 = **€120.00** | 2 |
| B | 20 + 18 + 18 = 56 | 18 + 18 + 20 + 20 + 18 = 94 | (9·56 + 3·94)/8 = 786/8 = **€98.25** | 1 |

Method B is 18% cheaper per week despite having the worse MAE. The reason is directional: A's
errors are concentrated on the short side (95 against 35), and the short side costs three times
as much. A metric that is blind to sign cannot see this, and MAE is blind to sign.

Now reverse the economics. Suppose the item is a perishable with a two-day life, where a leftover
unit costs €9 and a shortfall costs €3. The critical ratio falls to 0.25 and the penalty reverses:

| Method | Loss per week at `C_u = 3, C_o = 9` | Rank |
|---|---:|---|
| A | (3·95 + 9·35)/8 = 600/8 = **€75.00** | 1 |
| B | (3·56 + 9·94)/8 = 1,014/8 = **€126.75** | 2 |

Same two forecasts, same errors, opposite conclusion. The selection is a property of the cost
structure, not of the forecasts. This is why the critical ratio belongs in the metric
specification and why "which model is more accurate" is an incomplete question.

> **Note:** The asymmetric loss figures above treat the forecast as the committed quantity, which
> is the newsvendor interpretation. In practice the committed quantity is the forecast rounded to
> a case pack and adjusted by safety stock policy. The worked example isolates the metric choice;
> it does not model the full inventory policy chain.

## 7. Code, query, or calculation

One implementation, deliberately short. Everything else in this topic is arithmetic that belongs
on paper; this block exists so that the definitions are unambiguous when they reach code.

```python
import numpy as np

def metrics(actual, forecast, naive_scale):
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    e = a - f                                   # positive = under-forecast
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    mape = np.mean(np.abs(e) / a) * 100          # undefined if any actual is 0
    smape = np.mean(2 * np.abs(e) / (np.abs(a) + np.abs(f))) * 100
    mase = mae / naive_scale                    # scale from the TRAINING history
    bias = np.mean(e)
    ts = np.sum(e) / mae                        # RSFE / MAD, MAD taken as MAE
    wape = np.sum(np.abs(e)) / np.sum(a) * 100
    return dict(mae=mae, rmse=rmse, mape=mape, smape=smape,
                mase=mase, bias=bias, ts=ts, wape=wape)

a = [120, 60, 200, 40, 160, 30, 180, 100]
for name, f in [("A", [115, 65, 190, 55, 150, 45, 170, 40]),
                ("B", [138, 78, 180, 60, 142, 50, 162, 118])]:
    print(name, {k: round(v, 3) for k, v in metrics(a, f, 20.0).items()})
```

Expected output: A reports `mae 16.25, rmse 23.452, mape 22.101, smape 23.355, mase 0.812,
bias 7.5, ts 3.692, wape 14.607`; B reports `mae 18.75, rmse 18.774, mape 26.365, smape 22.441,
mase 0.938, bias -4.75, ts -2.027, wape 16.854`. These are the numbers in section 6 to three
decimals, which is the check: if your code disagrees, the definition disagrees, and the
definition in the specification is the one that governs.

Three implementation cautions, each of which has produced a wrong dashboard somewhere:

- `np.mean(np.abs(e) / a)` silently returns `inf` or `nan` on a zero actual. That is preferable to
  a substituted value, but it must be handled deliberately, not by a filter added later.
- The naive scale must be computed from the training window and passed in. Computing it inside
  this function from `a` would use the evaluation period and inflate the denominator's volatility.
- `MAD` here is the mean absolute error over the same window. Some systems use a smoothed MAD or a
  rolling MAD; both change the tracking signal, and the specification must say which.

## 8. Data, standards, and regulatory context

There is no international standard that defines forecast error metrics. This is worth stating
plainly, because analysts often assume otherwise and cite a standard that does not say what they
think it says. What exists is a vocabulary of measurement, a set of time conventions, and a body
of practitioner convention. Each constrains the metric indirectly.

### 8.1 What the vocabulary standards supply

| Standard | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of error, bias, variance, estimator | Any forecast error formula | B |
| ISO 5725-1 | Trueness and precision of measurement methods | Accuracy as a combination of bias and precision | Forecast evaluation, which has no repeated reference | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | Accuracy is not a quantity and has no numerical value | Anything specific to forecasts | B |
| JCGM 100:2008 (GUM) | Expression of uncertainty | How to combine and report uncertainty | Forecast error, which is not a calibration problem | B |
| ISO 8601 | Date and time representation | Week numbering, durations, ordering | Which week a demand observation belongs to | B |

The VIM point is the one that transfers most directly. "Forecast accuracy of 85%" is a conformance
rate against an implicit tolerance, exactly as "inventory accuracy of 98%" was in Topic 01A.08.
It is a statistic about a definition and a sample, not a property of the forecast. The standard
does not forbid the phrase; it forbids pretending the phrase has a unique meaning.

### 8.2 Time, systems, and governance

Every error metric is computed over buckets, and buckets are conventions. ISO 8601 weeks start on
Monday; a 4-4-5 retail calendar does not, and a "month" can be 28, 35, or 31 days. When actuals and
forecasts are aggregated on different definitions, the error contains a definitional component no
model can remove. Topic 01A.04 treats the machinery; here the point is that the specification must
name the calendar, not just the grain.

Enterprise planning suites expose several error measures, and their defaults differ on whether
zero-actual periods are excluded, whether the denominator is actual or forecast, whether the
horizon is pooled, and whether the comparison is against a statistical baseline or an approved
plan. That documentation is marketing-adjacent: grade it D and date it. Compute the metric
independently from the raw tables and reconcile to the system's number once.

Forecast accuracy feeds inventory valuation and revenue forecasts, which in listed companies fall
under internal control requirements such as Section 404 of the Sarbanes-Oxley Act. Those controls
govern the process — who may change a forecast, how the change is documented, whether the model is
validated — not the metric. Banking model-risk guidance such as the Federal Reserve's SR 11-7 is a
useful template by analogy, not a legal requirement for demand planning.

### 8.3 What regulation does not supply

No regulator specifies whether MAPE or MASE is correct, what trip value a tracking signal should
use, or how to weight a hierarchical metric. Those are engineering decisions with commercial
consequences, which is why the written specification in section 5.9 matters.

## 9. Industry practice and case evidence

### 9.1 What the forecasting competitions show

The M competitions are the closest thing the field has to a controlled comparison of evaluation
practice, because the same forecasts are scored under several metrics by an independent organiser.

| Evidence | What it shows | Grade |
|---|---|---|
| M4 used sMAPE, MASE, and an overall weighted average across 100,000 series | No single metric was trusted; the organiser published several and a composite | A |
| Hyndman and Koehler (2006) rescored the M3 methods and found that MASE distinguished substantially more method pairs at the 5% level than sMAPE did | Metric choice changes statistical power, not only ranking | A |
| M5 used RMSSE and a revenue-weighted hierarchical version (WRMSSE) over 42,840 series and 12 aggregation levels | Intermittent series with many zeros made percentage measures unusable; the weighting was made explicit and published | A |
| The M5 weighting was by cumulative dollar sales in the last 28 training days | Aggregation weights are a design choice, and the competition treated them as one | A |

The M5 choice is the most instructive. A competition with a large, sparse, hierarchical dataset
did not ask "which metric is best". It asked "which metric answers the question we are scoring",
decided that the question was mean demand at a revenue-weighted hierarchy, and published the
weights so the answer could be audited. That is the pattern to copy.

### 9.2 Forecast value added

FVA compares each step of the forecasting process against a naive benchmark, on the same series
and the same horizon. A step with negative FVA has made the forecast worse and is a candidate for
elimination.

| Claim | Source and grade |
|---|---|
| Across eight companies and more than 300,000 forecasts, 52% failed to beat the naive model | Reported in the Foresight FVA review (2013); Grade C, practitioner survey |
| Around 75% of statistical forecasts were manually adjusted; small adjustments had negligible impact and large downward adjustments tended to help | Fildes and Goodwin (2007), as reported in the same review; Grade C for the figure as summarised |
| Overrides at one technology manufacturer were worse than the baseline 60% of the time | Reported in the same review; Grade C, single-company anecdote |
| FVA can invert rankings based on MAPE | Illustrated in the SAS FVA white paper; Grade D, vendor material |

The pattern across these sources is consistent even where the individual figures are soft: a
substantial share of forecasting activity does not beat a trivial benchmark, and the share is
invisible under a metric that has no benchmark in it. That is the argument for reporting FVA
beside accuracy, not instead of it.

### 9.3 The MAPE default

MAPE remains the most widely reported measure in commercial demand planning, for historical rather
than statistical reasons: it is scale-free, it was easy to compute in spreadsheets, and "100 minus
MAPE" reads as an accuracy percentage even when it is asymmetric and undefined at zero.

The cost is documented. Syntetos and Boylan (2005), working with 3,000 automotive spare-parts
series, excluded MAPE and median absolute percentage error because actual demand can be zero, and
rejected sMAPE on the Goodwin–Lawton asymmetry grounds. They used mean signed error, a scaled
version of it, and a relative geometric root-mean-squared error. When series are intermittent, the
percentage family is not merely imperfect; it is inapplicable. Topic 10 takes this up.

### 9.4 What the evidence supports

Three conclusions survive the mixed quality of the sources. **Metric choice changes the ranking of
models**, by enough to matter commercially; the M3 rescoring and section 6 agree. **No metric is
universally valid**: the competition organisers changed metrics when the data changed, and
published why. **A benchmark is not optional**: the FVA literature, the MASE construction, and the
M5 weighting all embed a comparison. A number without a comparator is a reading without a scale.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| MAPE on series containing zeros | `inf`, `NaN`, or a filter that drops rows | Division by a zero actual | Scores computed on a biased subset | Use MASE, WAPE, or a scaled measure; report the excluded count |
| MAPE on low-volume items | Slow movers systematically under-forecast | Percentage error is largest when the actual is small | C-item stock-outs hidden by good aggregate accuracy | Stratify by volume; use WAPE for the aggregate view |
| sMAPE described as symmetric | Large under-forecasts score worse than intuition suggests | Goodwin–Lawton asymmetry; bounded at 200% | Model selection tilted against under-forecasting | Rename it, state the asymmetry, or use MASE |
| MAE compared across products of different size | "Average forecast error of 12 units" | Units are not comparable across scales | Meaningless portfolio average | Normalise, or stratify, or use a scaled metric |
| RMSE decided by one outlier | One week contributes most of the MSE | Quadratic penalty | Model chosen for an unrepresentative week | Inspect the error vector; report MAE and RMSE together |
| MASE denominator from the evaluation window | Score changes when the window changes | Denominator computed in-sample instead of out-of-sample | Inflated or deflated skill | Compute the scale on training history only; document the period |
| MASE with the wrong seasonal structure | MASE near 1 on strongly seasonal data | Non-seasonal naive scale too large | Model looks mediocre when it is good, or the reverse | Use the seasonal naive difference and state the season length |
| Averaging MASE across heterogeneous series | Aggregate dominated by a few difficult series | MASE equals a weighted mean of relative MAEs | Benchmark appears better than any method | Use AvgRelMAE or report the distribution |
| Bias reported without magnitude | "+3.2 units" with no denominator | Units without scale | Reader cannot judge severity | Report bias in units and as a percentage of mean actual |
| Tracking signal on a growing window | Trips more often as history lengthens | Statistic variance grows with `sqrt(n)` | False alarms; alarm fatigue | Fix the window length; state it |
| Tracking signal on autocorrelated errors | Persistent runs, frequent trips | Independence assumption violated | Threshold has no stable meaning | Plot cumulative error; use a wider window or a different test |
| Metric at aggregate, decision at item | Aggregate accuracy good, service poor | Error cancellation across items | Wrong inventory in the wrong places | Report at decision grain; reconcile explicitly |
| One-step metric used to justify a long-lead decision | Evaluation horizon shorter than the commitment | Different task measured | Model optimised for a decision nobody makes | Evaluate at the horizon that matches the lead time |
| Metric changed after seeing results | The winning model changes with the metric | Selection on the evaluation set | Inflated apparent performance | Pre-register the metric in the specification |
| Naive benchmark omitted | No sense of whether the model adds value | Accuracy reported in a vacuum | Effort spent on models worse than "last week" | Report FVA or MASE beside every accuracy number |

Three of these deserve a note. **MAPE on zeros** is the most common technical failure and the
easiest to detect, except that someone has usually added a filter first; excluding zero-actual
periods removes exactly the weeks in which the forecast was most likely to be wrong. **RMSE decided
by one outlier** is the most common analytical failure, and it is invisible in the score itself:
only the error series shows it. **Metric changed after seeing results** is the most common
institutional failure, and it usually arrives by drift rather than decision — the dashboard starts
on MAPE, WAPE is added, a "blended accuracy" appears, and within two years nobody can tell whether
performance improved. Section 5.9 exists to prevent exactly that.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The actual is measured without material error | Every metric treats the actual as truth | Scan undercount, receipt lag, restatement | Compare first-posted and final actuals; quantify the gap using Topic 01A.08 |
| Forecast and actual share a unit and a bucket | A ratio of unlike quantities is meaningless | Cases versus eaches; ISO week versus 4-4-5 | Check unit and calendar metadata before computing; reconcile a sample by hand |
| The evaluation sample resembles the deployment period | A metric is an estimate of future performance | Promotions, new items, structural breaks | Compare error distributions across periods; hold out a full seasonal cycle |
| Errors are stationary enough for a mean to be meaningful | Averages assume a stable process | Trend changes, regime shifts, model retraining | Plot error over time; test for a break |
| The loss is symmetric | MAE, RMSE, MAPE, and MASE all assume it | Service targets, perishability, penalty clauses | State the critical ratio; compare under asymmetric loss |
| Errors are independent across periods | Tracking-signal thresholds assume it | Promotions, autocorrelated demand, forecast smoothing | Autocorrelation of residuals; runs test |
| The denominator is non-zero and stable | Percentage and scaled metrics divide by it | Zero demand, near-zero actuals, tiny naive errors | Count zeros; inspect the denominator distribution |
| Series are comparable in scale | Absolute metrics are averaged across products | Mixed portfolio of fast and slow movers | Stratify or normalise before averaging |
| The benchmark was available at the decision time | FVA and MASE compare against a real alternative | Look-ahead in the naive construction; revised history | Reconstruct the benchmark from the vintage in use |
| One forecast per series per period is scored | Multiple forecasts allow selection on noise | Reforecasts, overrides, "best of" reporting | Audit the number of vintages per target period |
| The horizon matches the decision | Otherwise the metric measures a different task | Lead times longer than the evaluation horizon | Compare horizon to replenishment lead time |
| Aggregation weights reflect decision value | An unweighted mean implies equal value | Revenue concentration, margin differences | Publish the weights; test sensitivity to them |

Two limits of the whole framework are worth stating.

**The metric cannot tell you the cost of being wrong.** It can tell you which model minimises a
stated loss. The loss has to come from operations, finance, or the customer, and it is usually
known only approximately. The asymmetric loss calculation in section 6 is illustrative; a real
one would require a cost model, and the cost model would be the weaker part of the analysis.

**The metric is a sample statistic.** Eight weeks is not a population. A ranking that flips on
one week is a ranking with a confidence interval wider than the difference between the models.
Formal comparison requires a paired test across series or periods, and even then the result is
conditional on the evaluation window. Report the window, and treat small differences as ties.

## 12. Exercises

### Exercise 1 — Compute seven metrics by hand (calculation, 45 minutes)

Using the six-week table below, compute MAE, RMSE, MAPE, sMAPE, bias, and the tracking signal for
each of two methods. The naive scale from training history is 14.0 units.

| Week | Actual | Forecast P | Forecast Q |
|---:|---:|---:|---:|
| 1 | 50 | 55 | 40 |
| 2 | 20 | 25 | 35 |
| 3 | 80 | 70 | 60 |
| 4 | 10 | 20 | 25 |
| 5 | 60 | 55 | 45 |
| 6 | 30 | 25 | 40 |

**Deliverable:** a table with every metric, its unit, and the sum used in the numerator.
**Acceptance criteria:** all six metrics per method; correct units; the MAPE terms shown
individually; a one-sentence statement of which method you would choose and why.
**Time:** 45 minutes.

### Exercise 2 — Build the ranking flip (analysis, 40 minutes)

Construct two error vectors of eight periods each such that Method X wins on MAE and Method Y
wins on RMSE, with a difference of at least 10% on each metric.

**Deliverable:** the two error vectors, the four metrics, and a paragraph explaining the shape
difference that produces the flip.
**Acceptance criteria:** the arithmetic is reproducible by hand; the explanation names the
distributional property, not the numbers; both vectors are plausible for a real SKU.
**Time:** 40 minutes.

### Exercise 3 — Audit an accuracy definition (critique, 50 minutes)

Find a real or constructed "forecast accuracy" definition from a dashboard, vendor document, or
internal report. Reconstruct its numerator, denominator, grain, horizon, and exclusions.

**Deliverable:** a metric specification in the format of section 5.9, with gaps marked as
unknown, plus an estimate of how much the number could move under a plausible alternative
definition.
**Acceptance criteria:** at least six fields completed or explicitly marked unknown; the
sensitivity estimate is reasoned, not asserted; one exclusion is identified that would bias the
number upward.
**Time:** 50 minutes.

### Exercise 4 — Write a specification for a decision (design, 60 minutes)

Choose a replenishment decision with a lead time of at least four weeks. Write the full metric
specification that would govern model selection for it.

**Deliverable:** the fourteen-field specification, plus a justification of the penalty shape and
the grain.
**Acceptance criteria:** the horizon matches the lead time; the critical ratio is stated or its
absence is justified; the benchmark is defined precisely enough to compute; the owner is a role.
**Time:** 60 minutes.

### Exercise 5 — Choose under asymmetric cost (calculation, 45 minutes)

Using the data in Exercise 1, compute the asymmetric loss for both methods at `C_u = 12`,
`C_o = 4`, and again at `C_u = 4`, `C_o = 12`.

**Deliverable:** two loss tables, the critical ratios, and the reversal point stated as a ratio.
**Acceptance criteria:** positive and negative error sums shown separately; loss per period, not
total; the reversal explained in one sentence about the error direction.
**Time:** 45 minutes.

### Exercise 6 — Repair a percentage metric (analysis, 40 minutes)

Take a series with at least three zero-demand weeks in twelve. Compute MAPE with zeros included
and with zeros dropped, and compute MASE and WAPE on the same data.

**Deliverable:** a table of the four results, the count of dropped periods, and a recommendation.
**Acceptance criteria:** the direction of the bias from dropping zeros is identified; the MASE
denominator and its source period are stated; the recommendation names a decision.
**Time:** 40 minutes.

### Exercise 7 — Tracking signal behaviour (calculation, 35 minutes)

Simulate or construct twenty periods of unbiased, independent errors with a standard deviation
of 10. Compute the tracking signal cumulatively and on a rolling twelve-period window.

**Deliverable:** both series plotted or tabulated, with the periods in which `|TS| > 4`.
**Acceptance criteria:** the cumulative version trips and the rolling version trips less; the
difference is explained by the variance of the statistic; the window length is stated.
**Time:** 35 minutes.

### Exercise 8 — Design an FVA report (design, 50 minutes)

Design a monthly report that shows, for a product family, the value added by: the statistical
forecast over seasonal naive, the planner override over the statistical forecast, and the final
consensus over the override.

**Deliverable:** a one-page layout with the metric, the comparison, the weighting, and the
decision each row supports.
**Acceptance criteria:** every row has a named benchmark; the sign convention is stated; a row
with negative FVA has a stated action; the weighting matches the business grain.
**Time:** 50 minutes.

## 13. Self-check answers

### 13.1 Exercise 1 arithmetic

Actuals: 50, 20, 80, 10, 60, 30. Total 250.

**Method P.** Forecasts 55, 25, 70, 20, 55, 25. Errors `e = actual − forecast`:
−5, −5, +10, −10, +5, +5.

```text
sum |e|   = 5 + 5 + 10 + 10 + 5 + 5 = 40
MAE       = 40 / 6                    = 6.667 units

sum e^2   = 25 + 25 + 100 + 100 + 25 + 25 = 300
MSE       = 300 / 6 = 50.0
RMSE      = sqrt(50.0)                = 7.071 units

MAPE terms: 5/50=10.00, 5/20=25.00, 10/80=12.50,
            10/10=100.00, 5/60=8.33, 5/30=16.67
sum       = 172.50
MAPE      = 172.50 / 6                = 28.75%

sMAPE terms: 10/105=9.52, 10/45=22.22, 20/150=13.33,
             20/30=66.67, 10/115=8.70, 10/55=18.18
sum       = 138.62
sMAPE     = 138.62 / 6                = 23.10%

bias      = (-5 -5 +10 -10 +5 +5) / 6 = 0.00 units
TS        = 0 / 6.667                 = 0.00
MASE      = 6.667 / 14.0              = 0.476
```

**Method Q.** Forecasts 40, 35, 60, 25, 45, 40. Errors: +10, −15, +20, −15, +15, −10.

```text
sum |e|   = 10 + 15 + 20 + 15 + 15 + 10 = 85
MAE       = 85 / 6                     = 14.167 units

sum e^2   = 100 + 225 + 400 + 225 + 225 + 100 = 1,275
MSE       = 1,275 / 6 = 212.5
RMSE      = sqrt(212.5)                = 14.577 units

MAPE terms: 10/50=20.00, 15/20=75.00, 20/80=25.00,
            15/10=150.00, 15/60=25.00, 10/30=33.33
sum       = 328.33
MAPE      = 328.33 / 6                 = 54.72%

sMAPE terms: 20/90=22.22, 30/55=54.55, 40/140=28.57,
             30/35=85.71, 30/105=28.57, 20/70=28.57
sum       = 248.19
sMAPE     = 248.19 / 6                 = 41.37%

bias      = (10 -15 +20 -15 +15 -10) / 6 = +0.833 units
TS        = 5 / 14.167                = +0.353
MASE      = 14.167 / 14.0             = 1.012
```

Method P wins on every metric and its MASE is 0.476, comfortably better than the naive
benchmark. Method Q's MASE of 1.012 means it is marginally worse than "same as last period",
which is the most useful single statement in the exercise. Note that P's bias is exactly zero
while its errors still include a 100% MAPE week — zero bias does not mean accurate, and accurate
on average does not mean unbiased at the item level.

A common wrong answer divides sMAPE by the actual only, producing a number close to MAPE and
missing the point. Another computes MAPE on nonzero actuals only; here that drops week 4 and
cuts Q's MAPE from 54.72% to 44.67%, a 10-point gain produced entirely by exclusion.

### 13.2 Exercise 2 model answer

The flip requires one method to have a small number of large errors and the other to have many
moderate errors, with similar total absolute error. A workable pair:

```text
Method X errors:  +2, +2, +2, +2, +2, +2, +2, +20     sum |e| = 34,  sum e^2 = 428
Method Y errors:  -5, +5, -5, +5, -5, +5, -5, +5      sum |e| = 40,  sum e^2 = 200

MAE:  X = 4.25,  Y = 5.00      -> X wins
RMSE: X = 7.31,  Y = 5.00      -> Y wins
```

Any submission with this shape is acceptable. The explanation must say that X's errors are
concentrated with a heavy tail and Y's are diffuse, and that squaring changes which property
dominates. Making one method absurdly bad in a single week, without keeping the MAEs in a
plausible range, shows the arithmetic but not the judgment.

### 13.3 Exercise 3 rubric

A strong audit reconstructs at least six of the fourteen fields and marks the rest unknown rather
than guessing. It identifies the most consequential ambiguity — usually whether the denominator
is the actual or the plan, or whether zero-demand periods are excluded — and estimates its effect
by calculation rather than by adjective. It notes the direction of the bias: exclusions usually
flatter, and a plan-based denominator usually flatters more. A weak audit lists the fields without
testing sensitivity, or accepts the dashboard's own label as the definition.

### 13.4 Exercise 4 rubric

A strong specification names a horizon equal to the replenishment lead time, states the critical
ratio or explains why a symmetric loss is defensible, and defines the benchmark precisely enough
that a stranger could compute it. It chooses a grain that matches the order decision and states
what the aggregate report will conceal. It names a role as owner and a change-control rule. A
weak specification gives formulas without a grain, or states "MAPE" without a denominator, or
omits the benchmark entirely.

### 13.5 Exercise 5 arithmetic

Using P and Q from Exercise 1, with `e = actual − forecast` so positive is short:

```text
P: positive errors 10 + 5 + 5 = 20;  negative errors 5 + 5 + 10 = 20
Q: positive errors 10 + 20 + 15 = 45; negative errors 15 + 15 + 10 = 40

At C_u = 12, C_o = 4   (critical ratio 0.75):
  P: (12*20 + 4*20) / 6 = 320 / 6  = 53.33 per period
  Q: (12*45 + 4*40) / 6 = 700 / 6  = 116.67 per period
  P wins.

At C_u = 4, C_o = 12   (critical ratio 0.25):
  P: (4*20 + 12*20) / 6 = 320 / 6  = 53.33 per period
  Q: (4*45 + 12*40) / 6 = 660 / 6  = 110.00 per period
  P wins again.
```

P wins in both cases because it dominates on magnitude as well as direction, so this dataset does
not produce a reversal. The correct answer says so. A reversal requires two methods with similar
total absolute error and opposite directional bias, as in the worked example; manufacturing one by
changing the numbers has not understood the exercise.

### 13.6 Exercise 6 rubric

A strong answer reports MAPE with zeros present as undefined rather than substituting a value,
reports MAPE on the dropped-zero subset with the count of dropped periods, and shows that the
dropped subset is the one where the forecast is most likely to be wrong. It then computes MASE
with a stated denominator and WAPE on the full series, and recommends the scaled measure for
model selection and WAPE for the aggregate report. A weak answer silently replaces zeros with a
small number, or reports only the filtered MAPE.

### 13.7 Exercise 7 model answer

With independent, unbiased, normal errors of standard deviation 10, the cumulative `RSFE` after
`n` periods has standard deviation `10·sqrt(n)`, and `MAD` estimates about `8.0`, so the tracking
signal has standard deviation about `1.25·sqrt(n)`: 3.95 at 10 periods, 4.33 at 12, and 12.5 at
100. A fixed ±4 threshold is therefore crossed by chance a substantial fraction of the time at
short windows and almost always at long ones. A rolling twelve-period window keeps the standard
deviation near 4.3, which makes ±4 roughly a one-sigma rule — still not a significance test, but a
stable one. The acceptable answer states the standard deviation, shows the trips, and concludes
that the threshold must be quoted with its window.

### 13.8 Exercise 8 rubric

A strong report has one row per process step, each with its own benchmark and its own metric, and
reports FVA as a signed change so that a harmful step is visible. It weights by the same quantity
the business plans in — units for operations, revenue for finance — and says which. It includes a
row for the naive forecast itself, scored zero by construction, so the reader can see the
baseline. A weak report shows one accuracy number per step with no comparator, which cannot
distinguish a useful override from a harmful one.

## 14. Interview questions

**1. "What is the difference between MAE and RMSE, and when would you choose each?"**

A strong answer gives both formulas, states that RMSE squares before averaging so large errors
dominate, and adds the deeper point: MAE targets the conditional median and RMSE the conditional
mean, so on skewed demand RMSE selects a higher forecast level. It then ties the choice to how
the business experiences error — as a rate or as an event. A weak answer says RMSE "penalises
large errors more" and stops, which is true but does not reach the selection consequence.

**2. "Why is MAPE problematic?"**

A strong answer gives three distinct problems: undefined at zero actuals, asymmetric because a
fixed error is a larger percentage when the actual is small, and unstable when the actual is near
zero. It notes the behavioural consequence — models tuned on MAPE under-forecast low-volume items
— and names MASE or WAPE as alternatives with the trade-off stated. A weak answer says "it
doesn't work with zeros" and stops.

**3. "What does MASE measure that MAPE does not?"**

A strong answer gives the formula, explains that the denominator is the in-sample naive error so
the measure is scaled to the series' own difficulty, and states the interpretation: below 1 beats
the naive benchmark. It mentions that the denominator must come from the training period and that
seasonal data needs a seasonal scale. A weak answer calls it "a normalised MAE" without the
benchmark interpretation.

**4. "Is sMAPE symmetric?"**

A strong answer says no, cites the Goodwin–Lawton result, and gives a concrete example where an
under-forecast is penalised more heavily than the equivalent over-forecast. It notes the 200% cap
and that a saturated metric stops discriminating. A weak answer says yes because the formula has
both actual and forecast in the denominator.

**5. "A model has zero bias. Is it a good forecast?"**

A strong answer says bias measures direction only; a forecast can have zero mean error and large
absolute errors, and errors can cancel across periods or items. It asks for MAE or WAPE alongside,
and for the error distribution. A weak answer treats zero bias as evidence of accuracy.

**6. "How would you set a tracking signal threshold?"**

A strong answer describes the statistic, notes that ±4 is a practitioner convention, and explains
that the statistic's variance grows with the window, so the threshold must be quoted with the
window length. It suggests a rolling window and a plot of cumulative error rather than a single
ratio. A weak answer quotes ±4 as if it were a significance level.

**7. "Our forecast accuracy is 85%. What questions do you ask?"**

A strong answer asks: accuracy of what metric, at what grain, at what horizon, against which
actual vintage, weighted how, and compared with what benchmark. It notes that 85% could be
`1 − WAPE` or `100 − MAPE`, which are different quantities. A weak answer asks whether 85% is
good, which cannot be answered without the comparator.

**8. "How does the choice of metric change which model you deploy?"**

A strong answer explains that the metric is the objective function, gives the mean-versus-median
example on skewed demand, and notes that a change of metric changes the selected model, the
resulting bias, and the inventory consequence even when nothing is retrained. A weak answer says
the metric only affects reporting.

**9. "How would you evaluate a forecast when over- and under-forecasting have different costs?"**

A strong answer introduces the critical ratio `C_u/(C_u + C_o)`, states that the optimal point
forecast is that quantile, and proposes pinball loss at that level rather than a symmetric metric.
It notes that the costs are usually estimates and that the sensitivity to them should be shown.
A weak answer suggests weighting the errors arbitrarily without deriving the weights.

**10. "Why might aggregate forecast accuracy look better than item-level accuracy?"**

A strong answer explains cancellation — `|Σe| ≤ Σ|e|` — and weighting, and notes that both are
legitimate answers to different questions rather than errors. It recommends reporting at the
decision grain and reconciling explicitly. A weak answer says the aggregate is "smoothed" without
explaining the mechanism.

## 15. Key terms

- **Forecast error** — actual minus forecast, in the units of the series, with the sign stated.
- **MAE** — mean absolute error; linear penalty; targets the median; units.
- **MSE** — mean squared error; quadratic penalty; units squared.
- **RMSE** — root mean squared error; targets the mean; units.
- **MAPE** — mean absolute percentage error; undefined at zero; asymmetric; percent.
- **sMAPE** — symmetric MAPE; bounded at 200%; not actually symmetric; percent.
- **WAPE** — weighted absolute percentage error; total absolute error over total actual; equals volume-weighted MAPE.
- **MASE** — mean absolute scaled error; MAE divided by the in-sample naive MAE; dimensionless.
- **RMSSE** — root mean squared scaled error; the squared-error analogue of MASE.
- **Bias** — mean signed error; direction and magnitude in units.
- **MAD** — mean absolute deviation; the denominator of the tracking signal.
- **RSFE** — running sum of forecast errors; the numerator of the tracking signal.
- **Tracking signal** — RSFE divided by MAD; a bias monitor, not a significance test.
- **Trip value** — the threshold at which a tracking signal flags bias; conventionally ±4.
- **Naive forecast** — last observation carried forward; the standard benchmark.
- **Seasonal naive** — same period last season; the benchmark for seasonal data.
- **Scale-free** — a metric whose value does not depend on the unit of the series.
- **Pinball loss** — the loss consistent for a quantile; the asymmetric alternative to MAE.
- **Critical ratio** — `C_u/(C_u + C_o)`; the quantile that minimises newsvendor cost.
- **Proper scoring rule** — a score optimised by reporting the true predictive distribution.
- **CRPS** — continuous ranked probability score; reduces to absolute error for a point forecast.
- **FVA** — forecast value added; improvement over a naive benchmark attributable to a step.
- **Intermittent demand** — demand with zero periods between nonzero observations.
- **ADI** — average inter-demand interval; the mean gap between nonzero demands.
- **CV²** — squared coefficient of variation of nonzero demand sizes.
- **Aggregation grain** — the keys over which a metric is computed; a decision, not a default.
- **Forecast vintage** — the version of a forecast at a given origin, before revision.
- **Rolling-origin evaluation** — repeated re-fitting and scoring as the origin moves forward.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Hyndman, R. J. and Koehler, A. B. (2006). "Another look at measures of forecast accuracy."
  *International Journal of Forecasting*, 22(4), 679–688. DOI: 10.1016/j.ijforecast.2006.03.001.
  Source of MASE and the M3 rescoring.
- Goodwin, P. and Lawton, R. (1999). "On the asymmetry of the symmetric MAPE." *International
  Journal of Forecasting*, 15(4), 405–408. DOI: 10.1016/S0169-2070(99)00007-2.
- Syntetos, A. A. and Boylan, J. E. (2005). "The accuracy of intermittent demand estimates."
  *International Journal of Forecasting*, 21(2), 303–314. DOI: 10.1016/j.ijforecast.2004.10.001.
- Syntetos, A. A., Boylan, J. E., and Croston, J. D. (2005). "On the categorization of demand
  patterns." *Journal of the Operational Research Society*, 56(5), 495–503. ADI and CV² cut-offs.
- Kostenko, A. V. and Hyndman, R. J. (2006). "A note on the categorization of demand patterns."
  *Journal of the Operational Research Society*. Corrects the ADI boundary to 4/3 and CV² to 0.5.
- Gneiting, T. and Raftery, A. E. (2007). "Strictly proper scoring rules, prediction, and
  estimation." *Journal of the American Statistical Association*, 102(477), 359–378.
  DOI: 10.1198/016214506000001437.
- Gneiting, T. (2011). "Making and evaluating point forecasts." *Journal of the American
  Statistical Association*. DOI: 10.1198/jasa.2011.r10138. Consistency for means and quantiles.
- Davydenko, A. and Fildes, R. (2013). "Measuring the accuracy of judgmental adjustments to
  SKU-level demand forecasts." *International Journal of Forecasting*.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2020). "The M4 Competition: 100,000
  time series and 61 forecasting methods." *International Journal of Forecasting*.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). "The M5 accuracy competition:
  Results, findings and conclusions." *International Journal of Forecasting*.

**Book chapters and professional references — Grade B**

- Davydenko, A. and Fildes, R. (2016). "Forecast error measures: Critical review and practical
  recommendations." In *Business Forecasting: Practical Problems and Solutions*. Wiley.
- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, current edition.

**Standards and official — Grade B**

- ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1*. Current edition.
- ISO 5725-1, *Accuracy (trueness and precision) of measurement methods and results — Part 1*.
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in
  measurement* (GUM).
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM).
- ISO 8601, *Date and time — Representations for information interchange*.
- Board of Governors of the Federal Reserve System (2011). *Supervisory Guidance on Model Risk
  Management*, SR 11-7. Cited by analogy only; it does not govern demand planning models.

**Practitioner and vendor — Grade C or D**

- APICS/ASCM Dictionary, current edition. MAD, tracking signal, and the ±4 trip value. Grade C.
- Gilliland, M. *Forecast Value Added Analysis: Step-by-Step*. SAS white paper. Grade D.
- "FVA: A Reality Check on Forecasting Practices." *Foresight*, Spring 2013. Grade C.
- M4 and M5 competition repositories and evaluation code, GitHub. Grade B for the method as
  published.
- Raz, T. "Managing Inventories: The Newsvendor Model." SSRN 2974971. Grade B, teaching note.
- Gogolev, D. and Ozhegov, E. (2023). "Asymmetric loss function in product-level sales
  forecasting: An empirical comparison." *Applied Econometrics*, 70, 109–121. Grade A.

**Note on citations.** The M4 and M5 papers are cited by title and year; volume and page numbers
were not confirmed in the sources consulted and are omitted. The practitioner figures in section
9.2 are reported as the reviewing article states them and are not independently verified. Where a
number here is derived rather than sourced — the tracking-signal variance approximation, the
asymmetric loss calculations — the derivation is shown so a reader can check it.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisites: [`01_unit-of-measure-hierarchy.md`](01_unit-of-measure-hierarchy.md) and
  [`08_measurement-error-bias.md`](08_measurement-error-bias.md)
- Time buckets behind every metric: [`04_calendars-weeks-time-zones.md`](04_calendars-weeks-time-zones.md)
- Related: [`05_lead-time-definitions.md`](05_lead-time-definitions.md) and
  [`07_hierarchy-design.md`](07_hierarchy-design.md)
- Next topic: topic 10, *Why MAPE Fails on Intermittent Demand* (`10_intermittent-metrics.md`,
  not yet written); forward reference: topic 11, *Demand and Lead-Time Distributions*
  (`11_demand-lead-time-distributions.md`, not yet written)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Deeper treatment of forecasting method selection:
  [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Deeper treatment of hierarchical and probabilistic forecasting:
  [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- Return to the part map: [`../_index.md`](../_index.md)
