> Module 07 — Time Series Forecasting Foundations
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topics 09–11; Module 04 topic 04; Module 05 topic 06
> Version 2026.5

# Forecastability and Benchmarks First

---

## 1. Why this topic exists

A demand planning team presents a new forecasting model. The slide says mean absolute percentage
error (MAPE) fell from 31.4% to 26.8%, a 4.6-point improvement. The room is pleased. Someone asks
the only question that matters: better than what?

Nobody knows. The comparison is against last year's plan, which was itself a forecast, overridden
by planners, revised twice by sales, and inflated to protect a service target. The model may be
excellent. It may also be worse than writing down last month's shipment and walking away, and the
slide cannot tell you which. The improvement is real as a number and uninterpretable as a claim.

This is the most common analytical failure in commercial forecasting, and it is not a modelling
failure. It is a missing comparison. Three versions recur:

1. **No benchmark at all.** Accuracy is reported as an absolute number, so nobody can tell whether
   26.8% is good. On a highly seasonal series it may be mediocre; on a noise-dominated series it
   may be better than any method could achieve.
2. **The wrong benchmark.** The model is compared against a plan that contains planner judgement,
   promotional assumptions, and a management buffer. Beating it may mean only that the buffer was
   wrong.
3. **A benchmark computed on different data.** The model is scored on a rolling-origin backtest
   over eighteen months; the benchmark on a single three-month holdout. The gap between the two
   numbers is an artefact of the evaluation design.

Underneath all three is a question that precedes modelling and is rarely asked: **is this series
forecastable at all?** Some series have a strong, stable seasonal pattern and a clean trend, and a
simple method will beat a naive one by half. Others are statistically close to noise around a
constant, and the best available forecast is the average of recent history — a number no model can
improve on, because there is nothing in the history to learn. Building a model for the second kind
is not neutral. It costs money, it adds variance to the plan, and it creates a false sense that
the forecast is under control.

The evidence is unusually direct, because the forecasting competitions measure exactly this
comparison at scale. In the fifth Makridakis competition (M5), run on 42,840 daily retail series,
only **7.5% of competing teams beat a bottom-up exponential smoothing benchmark**, while **48.4%
beat the naive forecast**. The naive forecast is a bar almost half the field can clear.
Exponential smoothing is a bar almost nobody can. The gap between those two numbers is the whole
argument for choosing your benchmark deliberately.

There is a second gap, less discussed and more consequential commercially. **Beating the naive
benchmark is not the same as beating the current process.** A company's existing forecast is
usually better than naive, because it contains promotion calendars, customer intelligence, and
planner judgement. A model that beats naive by 12% and loses to the incumbent by 4% is a
regression dressed as an improvement. The correct comparator for a deployment decision is the
incumbent; the correct comparator for a research question is the naive benchmark.

This topic therefore does three things. It defines the four benchmarks every forecasting project
must compute before it builds anything. It gives you a way to estimate, in advance, how much room
a series offers — its forecastability — so you can decide where modelling is worth the cost. And
it sets out the governance rule that makes the decision auditable: a new model ships only if it
beats the incumbent baseline on a backtest specified before the results were seen.

Three later topics depend on this one. Exponential smoothing (Topic 02) is only meaningful as a
family of candidates against these baselines. Rolling-origin backtesting (Topic 08) is the
machinery that makes the comparison fair. Forecast value added (Topic 10) is the accounting system
that attributes improvement to each step of the process. None can be applied to a project that
never established what it was trying to beat.

> **Note:** Nothing in this topic argues against complex models. It argues that complexity has to
> be paid for, in the only currency that matters: error on a backtest that nobody could tune
> against, against a comparator fixed in advance.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** the mean, naive, seasonal naive, and drift forecasts at a stated origin and horizon,
   and **state** the unit and the seasonal period used.
2. **Select** which of the four is the appropriate floor for a given series, with a one-sentence
   justification grounded in the series' memory structure.
3. **Estimate** forecastability from history using at least three measures, and **explain** why the
   coefficient of variation alone is a misleading proxy.
4. **Construct** a single backtest on which a benchmark, an incumbent, and a challenger are all
   scored, and **identify** the four design choices that make the comparison fair.
5. **Distinguish** beating the naive benchmark from beating the incumbent process, and **say**
   which comparison a given decision requires.
6. **Write** a pre-registered evaluation protocol naming the metric, the benchmark, the horizon,
   the population, and the pass rule.
7. **Interpret** a forecast value added (FVA) result and **explain** what a negative value implies.
8. **Deliver** a benchmark result to a stakeholder whose model lost, without either softening the
   finding or overstating it.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is a ladder. The second is a bar chart of variance.

```text
   ERROR ON THE SHARED BACKTEST
     ^
     |   mean forecast        "the average of history, repeated"
     |   ------------------
     |   naive forecast       "same as the last observation"
     |   ------------------
     |   drift forecast       "last observation, plus the average change"
     |   ------------------
     |   seasonal naive       "same as this period last season"
     |   ------------------
     |   candidate model      everything you actually want to try
     |   ..................
     |   noise floor          what no method can remove
     +------------------------------------------------> method complexity

   Every rung is scored on the SAME origins, the SAME horizons,
   the SAME series, and the SAME metric. A model that cannot
   clear the rung beneath it has added risk, not value.
```

The ladder is not a ranking of quality. It is a ranking of *assumptions*. The mean forecast
assumes no memory and no trend. The naive forecast assumes one unit root and no seasonality. The
drift forecast assumes a deterministic linear trend from first to last observation, which is
usually a bad assumption and occasionally the only one that works. Seasonal naive assumes the
pattern repeats exactly one cycle later. Each rung adds structure and therefore narrows the class
of series on which it is the right floor.

The second picture explains why the ladder sometimes looks flat.

```text
   variance of a series, decomposed into what a method could remove and what it cannot

   A. seasonal + trend   [##### signal #####][## noise ##]        forecastable
   B. random walk        [# signal #][###### noise ######]        benchmark is naive
   C. noise around mean  [############# noise #############]      not forecastable

   In A a model can beat the benchmark by a wide margin.
   In B the naive forecast is already close to optimal, and MASE near 1.0 is the ceiling.
   In C the mean forecast is already optimal, and no method can beat it at all.
```

Three properties follow, and they organise the rest of the topic.

**Property 1 — The benchmark is a floor.** You do not "beat" the naive forecast in the sense of
winning a contest. You establish that the structure you added — seasonality, exogenous
regressors, a learned nonlinearity — carries information the persistence assumption does not. If
it does not, the correct action is to ship the trivial method and spend the budget elsewhere.
That is a successful project outcome, and organisations that cannot say so will keep funding
models that lose to arithmetic.

**Property 2 — Forecastability is a property of the series, not of the method.** It is the share
of a series' variation that any method could in principle remove. It cannot be increased by
trying harder, by adding features, or by hiring a better team. It can be increased by *changing
the series*: aggregating across locations, moving from weekly to monthly buckets, or forecasting
a product family rather than a stock-keeping unit (SKU). That is why the first analytical move on
a noise-dominated series is not a model but a change of grain.

**Property 3 — Two comparisons, two decisions.** The comparison against the naive benchmark
answers "is there exploitable structure here?" It is a research question and its answer is a
property of the data. The comparison against the incumbent answers "should we change what we do
on Monday?" It is a deployment question and its answer is a property of the organisation. A model
can win the first and lose the second, and that is common rather than rare, because the incumbent
usually has information the benchmark does not.

The two comparisons combine into a decision table you can draw on a whiteboard.

```text
                          beats the naive benchmark?
                              no              yes
                       +----------------+----------------+
        beats the      |  The process   |  The incumbent |
        incumbent?  no |  is the        |  may be stale, |
                       |  problem, not  |  or the gain   |
                       |  the model     |  is too small  |
                       +----------------+----------------+
                    yes|  Contradiction:|  Ship it, and  |
                       |  re-check the  |  monitor the   |
                       |  backtest      |  benchmark     |
                       +----------------+----------------+
```

If a model beats the incumbent but not the naive forecast, one of the two measurements is wrong.
Usually the incumbent is scored on a different population — for example only the easy high-volume
items — or the naive benchmark has a look-ahead error. Audit before reporting.

The upper-right cell is where most real projects land and where most of the political difficulty
lives. The model is better. It is not better by enough to justify a migration, a retraining
programme, and a change of planning process. Section 5.6 gives the rule for deciding, and section
6.8 gives the conversation.

### 3.1 What "beat" means numerically

```text
   skill         = 1 - (E_c / E_b)        positive means the candidate is better
   error ratio   = E_c / E_b              below 1 means the candidate is better
   improvement   = (E_b - E_c) / E_b      same as skill, as a percentage
```

A skill of 0.20 means the candidate removed 20% of the benchmark's error. A skill of −0.05 means
it added 5%. Neither says whether the difference is distinguishable from zero, which is why
section 5.5 insists on a per-series win rate and a paired comparison rather than a pooled ratio.

> **Note:** Skill scores are not additive across steps. If a statistical model has skill 0.30 over
> naive and a planner override has skill 0.10 over the statistical model, the override's skill
> over naive is not 0.40; it is `1 - (1 - 0.30)(1 - 0.10) = 0.37`. Additive reasoning overstates
> stacked improvements and is one of the most common errors in FVA reporting.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Benchmark forecast | The cheapest defensible forecast a candidate must beat | Treated as a straw man rather than a floor | Units of the series |
| Naive forecast | Last observed value carried forward for every horizon | Called a random walk without checking the differenced series | Units |
| Seasonal naive forecast | Value from the same period one season earlier | Computed with the wrong seasonal period `m` | Units |
| Drift forecast | Last observation plus `h` times the average change per period | Confused with a fitted trend model | Units; slope in units per period |
| Mean forecast | Arithmetic mean of the history, repeated for every horizon | Dismissed as trivial; it is the null model | Units |
| Naive2 | Seasonally adjusted random walk used as the M3 benchmark | Assumed identical to seasonal naive | Units |
| Random walk | A series whose first differences are white noise | Used loosely for any volatile series | Dimensionless characterisation |
| Forecastability | Share of a series' variation that a method could in principle remove | Equated with a low coefficient of variation | Proportion, 0 to 1 |
| Signal-to-noise ratio | Predictable variation divided by unpredictable variation | Quoted without stating the variance decomposition | Ratio |
| Coefficient of variation (CV) | Standard deviation divided by mean over a stated window | Used as a forecastability score | Dimensionless |
| Strength of seasonality | One minus remainder variance over deseasonalised variance | Computed on a history shorter than two cycles | Proportion, 0 to 1 |
| Seasonal period `m` | Number of observations in one seasonal cycle | Set to 12 on weekly data by habit | Count of periods |
| Forecast horizon `h` | Periods between the origin and the target | Pooled across steps without per-step results | Periods |
| Forecast origin `T` | Last period with observed data when the forecast is made | Confused with the target period | Period index or timestamp |
| Backtest | Evaluation of a forecast procedure on data not used to fit or tune it | Called a holdout when the holdout was used for tuning | Procedure identifier |
| Rolling origin | Repeated re-fitting as the origin advances through history | Replaced by a single train/test split | Origin schedule |
| Pre-registration | Writing metric, benchmark, population, and pass rule before results | Written after the first results meeting | Document version |
| Incumbent | The process or model currently producing the operational forecast | Assumed to be the naive forecast | Version identifier |
| Challenger | A candidate model evaluated against the incumbent | Deployed before a backtest is complete | Version identifier |
| MASE | Mean absolute scaled error; MAE divided by the in-sample naive MAE | Denominator taken from the evaluation window | Dimensionless |
| RMSSE | Root mean squared scaled error; the squared-error analogue of MASE | Confused with MASE | Dimensionless |
| Relative absolute error (RAE) | Mean absolute error divided by a stated benchmark's | Benchmark left unstated | Dimensionless |
| Theil's U | Ratio of forecast RMSE to naive RMSE | Cited without the version number | Dimensionless |
| Skill score | One minus candidate error over benchmark error | Sign convention reversed between reports | Proportion |
| Forecast value added (FVA) | Change in a metric attributable to one process step | Reported without the benchmark step | Percent, or metric difference |
| Beat-the-naive rate | Share of series on which a method beats the naive forecast | Averaged without weighting by volume | Percent of series |
| Noise floor | The error no method can remove | Assumed to be zero | Units |
| Aggregation gain | Error reduction from forecasting a sum rather than its parts | Assumed to be free of disaggregation cost | Percent |
| Intermittent demand | Demand with periods of zero between nonzero observations | Treated as a small-numbers problem | Units per period |
| Average inter-demand interval (ADI) | Mean number of periods between nonzero demands | Leading zeros handled inconsistently | Periods |
| Squared coefficient of variation (CV²) | Variance of nonzero sizes over their squared mean | Computed on all periods including zeros | Dimensionless |
| Zero forecast | A forecast of exactly zero for every horizon | Scored on MAE, which rewards it on intermittent series | Units |
| Look-ahead bias | Using information unavailable at the origin | Introduced by refreshed history extracts | Boolean |
| Data leakage | Training information reaching the evaluation set | Confused with look-ahead bias | Boolean |
| Portfolio stratification | Reporting results by demand class rather than pooled | Pooled average reported alone | Class label |
| Pass rule | The written condition under which a challenger ships | Decided in the meeting after the results | Threshold plus metric |
| Evaluation population | The exact set of series included in a backtest | Changed between benchmark and candidate | Series identifier list |

Four of these repay a sentence each.

**Naive, seasonal naive, Naive2.** Naive is `y_T` for all horizons. Seasonal naive is `y_{T+h-m}`
for horizon `h`, with `m` the seasonal period. Naive2, the benchmark used in the third Makridakis
competition (M3), is a seasonally adjusted random walk — the series is deseasonalised, the last
deseasonalised value carried forward, and the seasonal factor restored. It is not the same
computation as seasonal naive, though on strongly seasonal data the two are often close.

**Forecastability and the coefficient of variation.** A series with a large seasonal swing has a
high CV and can be highly forecastable. A series that oscillates randomly within a narrow band has
a low CV and cannot be forecast at all. CV measures how much a series moves, not how much of that
movement is predictable. Section 6.6 works a comparison where the lower-CV series is the less
forecastable one.

**MASE and skill.** MASE is a scaled absolute error whose denominator is the naive benchmark's
in-sample error, so a MASE below 1 means the method beat naive. Skill is the same idea as a
proportion. They carry the same information in different units, and mixing them in one report
creates the appearance of two independent findings.

**Intermittent demand and the zero forecast.** On a series that is zero in most periods, a
forecast of zero has a low mean absolute error and a terrible service outcome. The benchmark
comparison is still valid, but the metric has become disconnected from the decision. Module 01A
topic 10 and Topic 06 treat this in depth; here it is enough that on intermittent series the
benchmark must be scored on a cost or service metric, not on MAE alone.

## 5. The mechanics, step by step

This section runs in the order a real project should run, which is the reverse of the order most
projects actually run. The frame is fixed first, the benchmarks second, forecastability third, and
only then is a model fitted. Each step states its input, its output, and the branch point where
the project can legitimately stop.

### 5.1 Fix the comparison frame

Six things must be written down before any number is computed. Every one of them has, at some
point in a real organisation, been left implicit and produced a dispute.

**Step 1. Name the decision and its grain.** A replenishment forecast is made at SKU, location, and
week; a financial plan at category, region, and month. Write the grain as a key definition:
`SKU × ship-to location × ISO week`. *Input:* the decision, owner, and cadence. *Output:* one grain,
in writing. *Branch:* two decisions need two backtests, because errors cancel differently at each
grain.

**Step 2. Fix the horizon to the lead time.** A model evaluated one step ahead and deployed twelve
steps ahead has been evaluated on a different task. Set `h = 1 … H` with `H` equal to the lead time
plus the review period, and report every step. *Branch:* if history is too short for non-overlapping
windows, use overlapping rolling origins and say so; do not shorten the horizon silently.

**Step 3. Fix the unit and the calendar.** Units, cases, or eaches; ISO weeks, 4-4-5 retail periods,
or calendar months. A benchmark on calendar months compared with a model trained on 4-4-5 periods
differs by a definitional component no method can remove.

**Step 4. Fix the origin and the vintage.** The benchmark must be computed with exactly the
information available at each origin — no revised history, no later restatements, no promotion
calendar published after the fact. *Branch:* if the warehouse stores only current-state history,
you have look-ahead bias by construction. Reconstruct point-in-time history or declare the
limitation.

**Step 5. Fix the population.** New items with eight weeks of history, discontinued items with a
partial final year, series with zero total demand — all need an explicit inclusion rule, and the
exclusions must be counted. If the candidate and the benchmark have different populations, the
comparison is void.

**Step 6. Fix the metric and its aggregation.** Choose before you look, and state the weighting.
*Branch:* if the primary and secondary metrics disagree in ranking, the disagreement is a finding.
Report both and explain the mechanism.

> **Note:** These six choices are the entire content of a fair comparison. Most disputes about
> whether a model "works" are disputes about one of them, and the fastest way to end such a dispute
> is to reconstruct which choice differed between the two numbers.

### 5.2 The four benchmarks, and when each is the right floor

```text
   with history y_1 ... y_T and horizon h = 1 ... H

   mean            F(T+h) = (1/T) * SUM(t=1..T) y_t
   naive           F(T+h) = y_T
   seasonal naive  F(T+h) = y_(T + h - k*m),   k = ceil(h / m)
   drift           F(T+h) = y_T + h * (y_T - y_1) / (T - 1)
```

Check the seasonal index by hand. For `m = 12` and `h = 1` it returns `y_(T-11)`, the same calendar
month one year before the last observation. For `h = 12` it returns `y_T`. For `h = 13` it returns
`y_(T-11)` again. An off-by-one here produces a benchmark that scores either far too well or far
too badly for no visible reason.

| Benchmark | Assumption it encodes | Right floor when | Fails when |
|---|---|---|---|
| Mean | No memory, no trend, stationary level | Series is noise around a constant; you want the null model | Any trend or seasonality |
| Naive | One unit root; best predictor of tomorrow is today | Levels wander without seasonality: prices, some commodity indices | Strong seasonality or a clear trend |
| Seasonal naive | The pattern repeats exactly one cycle later | Seasonality dominates and the year-on-year level is stable | The series trends, or the seasonal amplitude changes |
| Drift | A deterministic linear trend from first to last observation | Short, stable, near-linear growth with no seasonality | Endpoints are atypical; seasonality is present; the horizon is long |

Two rules follow. **Use the strongest benchmark the data supports**: seasonal naive on weekly data
is a much harder bar than naive, and it is the right bar for a product with an annual pattern.
**Compute all four anyway**, because the spread is diagnostic: mean and naive within 5% means no
trend; seasonal naive 40% better than naive means seasonality dominates; drift beating everything
means checking whether the trend is real or an artefact of two unusual endpoints.

### 5.3 Estimate forecastability before you model

Forecastability is not observable. What is observable are proxies, and each fails in a different
direction. Use at least three and read them together.

**Measure 1 — the mean-forecast benchmark.** Compute the MAE of the mean forecast on the evaluation
window. It is the error of assuming nothing changes. If the best of the other three benchmarks
cannot beat it by more than a few percent, the series is close to noise around a constant and no
method will help.

**Measure 2 — benchmark spread.** Compute each benchmark's skill against the mean forecast:

```text
   benchmark skill = 1 - MAE(benchmark) / MAE(mean forecast)
```

Seasonal naive with skill 0.55 says more than half of the error of "assume the average" is
removable by knowing the month. That is a structural series. Skill 0.08 says the seasonal pattern
is present but weak relative to the noise.

**Measure 3 — strength of seasonality.** Decompose into trend, seasonal, and remainder, then
compute `1 - Var(remainder) / Var(detrended series)`. Above roughly 0.6 is strong; below 0.2 is
weak. The threshold is a convention, not a test.

> **Note:** Never compute a seasonality measure on fewer than two complete cycles. With one cycle
> you cannot separate the seasonal pattern from the trend, and the decomposition attributes all
> movement to seasonality. For monthly data that means 24 observations minimum.

**Measure 4 — the coefficient of variation, used carefully.** CV is standard deviation over mean.
It measures total variation, which includes trend and seasonal swing, both of which are signal,
and it uses the same observations to compute mean and deviation, which biases it downward on short
windows. It is a description, not a skill score. A CV above roughly 0.5 combined with many zeros is
a warning that item-level modelling will be hard; a CV below 0.05 usually means the series is flat,
and flat series are often unforecastable for a different reason — there is nothing to forecast.

**Measure 5 — intermittency, where relevant.** The average inter-demand interval (ADI) and the
squared variation of nonzero demand sizes (CV²) classify a series as smooth, erratic, intermittent,
or lumpy. Lumpy series are the hardest and the ones where a model most often loses to a benchmark.

| Class | Typical signature | Best benchmark | Modelling verdict |
|---|---|---|---|
| Structural | Strong seasonality or trend, benchmark skill above 0.35 | Seasonal naive, or drift plus seasonality | Model worth building; expect double-digit gains |
| Marginal | Weak structure, benchmark skill 0.10 to 0.35 | Whichever benchmark scores best | Model may add 5 to 15%; require a clear pass rule |
| Noise-dominated | Flat, benchmark skill below 0.10 | Mean forecast | Do not model at item level; aggregate, or hold safety stock |
| Intermittent | ADI above 1.3, many zeros | Naive with a zero floor, or a Croston-family method | Model on a cost metric, not MAE |

### 5.4 One backtest for all competitors

| Design choice | Correct practice | Failure mode if wrong |
|---|---|---|
| Origin schedule | Rolling origin, re-fitting at each step | Single split: the result depends on which split was chosen |
| Horizon coverage | Score every step 1 to H, report per step | Score only `h = 1`, deploy at `h = 12` |
| History window | Fixed-length rolling or expanding, stated | Silently expanding for the model and fixed for the benchmark |
| Refit frequency | Same for every competitor | Model refitted weekly, benchmark computed once |

The refit frequency is the one people forget. A benchmark computed once and applied across three
years will look worse than a model refitted monthly, because the benchmark is stale and the model
is not. That produces a benchmark that is easy to beat for a reason unrelated to modelling. Refit
on the same schedule, or state that you deliberately did not.

### 5.5 Score, then compare honestly

A single pooled error ratio is not a result. It is a point estimate with an unknown sampling
distribution, computed on a population that was chosen after the fact often enough that you should
assume it was. Report four things.

1. **The pooled metric**, with unit, weighting, and population.
2. **The per-series win rate.** A pooled improvement of 10% can coexist with a win rate of 45%,
   meaning the candidate wins big on a few series and loses slightly on many. That is a different
   product decision from a uniform 10% gain.
3. **A paired comparison across series** — a signed-rank test on per-series error ratios, or a
   bootstrap interval on the pooled difference — to show the gap exceeds between-series variation.
4. **The result stratified by demand class**, because aggregate improvements are frequently
   produced by one class and hide losses in another.

A worked illustration of the gap: pooled MAE 4% better than seasonal naive, win rate 41%, median
per-series error ratio 1.03. Read together, the model is worse than the benchmark on the typical
series and much better on a small group. Deploying on the whole portfolio is a mistake; deploying
on the group where it wins is a legitimate, narrower project.

### 5.6 The two thresholds and the pass rule

The pass rule is written before the backtest runs. It has three parts. **Threshold 1 — the
benchmark.** The candidate must beat the best cheap benchmark, on the pre-registered metric and
population. Beating the mean forecast is not enough if seasonal naive is available. **Threshold 2
— the incumbent.** The candidate must win by a margin that pays for the change: migration cost, a
new failure mode, retraining, and the loss of institutional knowledge in the current process. A
defensible rule is the larger of five percentage points of error and twice the standard error of
the backtest difference. **Threshold 3 — the segment.** If the candidate wins on a segment rather
than the population, the pass rule should have anticipated it. A rule written after the
segmentation was discovered is not a rule; it is a description of the result.

```text
   ship if, and only if:
     (1) E_candidate < E_best_benchmark on the primary metric, and
     (2) E_candidate <= E_incumbent - max(0.05 * E_incumbent, 2 * SE_diff), and
     (3) the gain holds in at least two of the three most recent backtest folds
```

The third condition guards against a result driven by one unusual period. It is not a statistical
test; it is a stability check, and it is the one that most often catches a model that learned a
regime which has since ended.

### 5.7 Pre-register, then monitor

Write the protocol as a one-page document with a version number and a date, before the backtest
runs. It contains the six frame choices, the four benchmarks, the metric definition with its
denominator, the population rule, and the pass rule. Circulate it to whoever will approve the
deployment. The purpose is not bureaucracy; it is to make the difference between a measurement and
a selection procedure. A metric chosen after the leaderboard is visible is a way of arriving at
the answer you wanted, with arithmetic attached.

After deployment, monitor three things monthly: the candidate's error against the benchmark on the
same rolling window, the benchmark's own error, and the gap between the two. A model that was 8%
better at launch and is 2% worse a year later has not necessarily degraded; the benchmark may have
improved because the demand environment became less forecastable for everyone.

## 6. Worked example

### 6.1 Setting

Meridian Medical Supply distributes consumables to clinics. It forecasts monthly demand in eaches
at item–region level, with a three-month lead time and a monthly planning cycle, so the decision
horizon is four months. The team has built a gradient-boosted model on 38 features and wants to
deploy it across a 40-item portfolio. The evaluation uses 24 months of history and a 12-month
holdout, origin fixed at month 24, horizons 1 to 12 scored separately. All figures are constructed
to be internally consistent; they are not taken from a company.

### 6.2 One series, in full

SKU A is a seasonal consumable. Its 24 months of history, in eaches:

| Month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Year 1 | 400 | 440 | 520 | 600 | 700 | 790 | 760 | 650 | 550 | 470 | 420 | 520 |
| Year 2 | 430 | 470 | 560 | 650 | 760 | 850 | 820 | 700 | 590 | 510 | 450 | 560 |

Year 1 totals 6,820 and year 2 totals 7,350, so the series grows about 7.8% a year. The seasonal
shape peaks in June and troughs in November. The 12-month holdout totals 8,000.

| Month | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Actual | 480 | 520 | 610 | 700 | 820 | 910 | 880 | 760 | 640 | 560 | 500 | 620 |

### 6.3 The four benchmarks, computed at the origin

**Mean.** `14,170 / 24 = 590.42` eaches, applied to all twelve horizons. **Naive.** `y_24 = 560`,
applied to all twelve. **Seasonal naive.** The year-2 month for each horizon: 430, 470, 560, 650,
760, 850, 820, 700, 590, 510, 450, 560. **Drift.** Slope `(560 − 400) / 23 = 6.9565` eaches per
month, so the horizon-`h` forecast is `560 + 6.9565h`: 566.96 at `h = 1`, 643.48 at `h = 12`.

### 6.4 The errors

With `e = actual − forecast`, in eaches:

| h | Actual | Mean | Naive | Seasonal naive | Drift | Model |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 480 | −110.4 | −80 | +50 | −87.0 | +20 |
| 2 | 520 | −70.4 | −40 | +50 | −53.9 | +20 |
| 3 | 610 | +19.6 | +50 | +50 | +29.1 | +25 |
| 4 | 700 | +109.6 | +140 | +50 | +112.2 | +35 |
| 5 | 820 | +229.6 | +260 | +60 | +225.2 | +40 |
| 6 | 910 | +319.6 | +350 | +60 | +308.3 | +45 |
| 7 | 880 | +289.6 | +320 | +60 | +271.3 | +40 |
| 8 | 760 | +169.6 | +200 | +60 | +144.4 | +30 |
| 9 | 640 | +49.6 | +80 | +50 | +17.4 | +20 |
| 10 | 560 | −30.4 | 0 | +50 | −69.6 | +25 |
| 11 | 500 | −90.4 | −60 | +50 | −136.5 | +20 |
| 12 | 620 | +29.6 | +60 | +60 | −23.5 | +30 |
| **Sum abs** | | **1,518.3** | **1,640.0** | **640.0** | **1,478.3** | **350.0** |
| **MAE** | | **126.5** | **136.7** | **53.3** | **123.2** | **29.2** |

Four readings. **Seasonal naive beats naive by 61%** (`1 − 53.3/136.7`), so seasonality is the
dominant information and any candidate ignoring it will lose. **Drift barely beats the mean
forecast** (123.2 against 126.5), because the trend is small relative to the seasonal swing.
**Naive is worse than the mean forecast**, the signature of a trending series: carrying the last
value forward under-forecasts every high month, so using naive as the only floor would flatter the
model by 78%. **The model beats seasonal naive by 45%** (`1 − 29.2/53.3`), and that comparison,
not the naive one, is the justification.

### 6.5 MASE, computed two ways

```text
   non-seasonal scale = (1/23) * SUM(t=2..24) |y_t - y_(t-1)| = 1,880 / 23 = 81.74
   seasonal scale     = (1/12) * SUM(t=13..24) |y_t - y_(t-12)| =   530 / 12 = 44.17
```

| Method | MAE | MASE, non-seasonal | MASE, seasonal | Skill vs mean forecast |
|---|---:|---:|---:|---:|
| Mean | 126.5 | 1.548 | 2.865 | 0.000 |
| Naive | 136.7 | 1.672 | 3.094 | −0.080 |
| Drift | 123.2 | 1.507 | 2.789 | 0.026 |
| Seasonal naive | 53.3 | 0.652 | 1.207 | 0.578 |
| Model | 29.2 | 0.357 | 0.660 | 0.769 |

The two columns tell different stories, and the difference is the point. On the non-seasonal scale
seasonal naive scores 0.652 and looks like a modest improvement over naive. On the seasonal scale,
the correct scale for this series, it scores 1.207 — worse than the benchmark it is supposed to be.
The seasonal pattern is growing: the year-over-year difference rose from an average of 44.2 in
training to 53.3 in the holdout, and pure seasonal naive carries that growing bias.

> **Note:** The choice of MASE denominator is not cosmetic. For a seasonal series the seasonal
> scale is the honest one, and reporting only the non-seasonal version understates every method's
> difficulty. State the scale and the seasonal period in the protocol.

### 6.6 The series where the model lost

SKU B is a mature, high-volume item with 24 months of history: mean 988 eaches, standard deviation
41.2, range 905 to 1,062. There is a weak month-of-year pattern and no trend. The championed model
uses the same 38 features.

| Method | MAE (eaches) | Skill vs mean | Reading |
|---|---:|---:|---|
| Seasonal naive | 17.1 | 0.252 | Best available |
| Mean forecast | 22.8 | 0.000 | Null model |
| Naive | 25.0 | −0.095 | Worse than the mean |
| Championed model | 27.5 | −0.204 | Worse than everything except drift |
| Drift | 36.8 | −0.612 | Extrapolates a spurious trend |

The model is 61% worse than seasonal naive (`1 − 27.5/17.1 = −0.608`) and worse than the mean
forecast, which it was never compared against. The diagnosis is in the descriptors: SKU B has a CV
of 0.042, SKU A has 0.227. The series with the *lower* variation is the one that cannot be
forecast, because almost all of its variation is noise rather than structure. A model with 38
features and 24 observations fits that noise, and the noise does not repeat.

### 6.7 The portfolio

| Class | Items | Median benchmark skill | Items where a model beat the best benchmark by 5%+ |
|---|---:|---:|---:|
| Structural | 11 | 0.47 | 7 |
| Marginal | 18 | 0.14 | 2 |
| Noise-dominated | 11 | 0.03 | 0 |
| **Total** | **40** | | **9** |

Nine of forty series — 22.5% of the portfolio — justified a model, and seven of those nine came
from the eleven structural series. The championed model was proposed portfolio-wide; the evidence
supports deployment on nine items and a benchmark-only policy on the other thirty-one. This ratio
is the practical content of "the share of series that are forecastable". It is not a universal
number, and any vendor quoting one should be asked for the population. But the direction is
consistent with the competition evidence in section 9: the share of series where sophistication
pays is a minority, concentrated in the series with visible structure.

### 6.8 Telling the stakeholder

The conversation is usually handled badly in one of two directions: softened until the finding
disappears, or stated so bluntly that the model's owner defends it out of pride. Five sentences
work better than either.

1. **State the result in the stakeholder's units.** "On the 31 items outside the nine-item
   deployment, the model's average miss is 27.5 eaches a month against 17.1 for the seasonal
   benchmark. That is 61% more error, roughly €41,000 a year in extra holding and stock-out cost."
2. **Give the mechanism, not a verdict on the person.** "The model is sound; the series are not.
   Those items have almost no repeatable pattern, and 24 months is too little history for 38
   features, so it is fitting noise."
3. **Show where it does work.** "On the eleven structural items it beats the benchmark by 45% on
   average, and seven of them justify deployment. That is where I would put it."
4. **Name the decision.** "I am asking to deploy on nine items and keep the seasonal benchmark on
   the rest, with a review in two quarters."
5. **Say what would change your mind.** "Three years of history, or aggregation to family level,
   may change the picture. I will re-run the backtest then."

Three things to avoid. Do not compare the model to the naive forecast in the meeting, even though
it probably beats it, because that invites the reply that the comparison was chosen to flatter. Do
not present a pooled portfolio number without the segmentation, because the pooled number will be
positive and will hide the loss. And do not offer to "tune it a bit" as a way of ending the
conversation; that converts a measurement into an open-ended optimisation with no pass rule.

## 7. Code, query, or calculation

The arithmetic in section 6 is meant to be done by hand once. The block below exists so the four
definitions are unambiguous when they reach code, and so the seasonal index is written down in a
form you can check against your own implementation.

```python
def benchmarks(y, m, h):
    """y: history list, m: seasonal period, h: horizon. Returns h-step forecasts."""
    T = len(y)
    level = sum(y) / T
    return {
        "mean":   [level] * h,
        "naive":  [y[-1]] * h,
        "snaive": [y[T - m + (i % m)] for i in range(h)],
        "drift":  [y[-1] + (i + 1) * (y[-1] - y[0]) / (T - 1) for i in range(h)],
    }

def mase(actual, forecast, scale):
    """scale: in-sample naive (or seasonal naive) MAE from the TRAINING history only."""
    return sum(abs(a - f) for a, f in zip(actual, forecast)) / len(actual) / scale

hist = [400, 440, 520, 600, 700, 790, 760, 650, 550, 470, 420, 520,
        430, 470, 560, 650, 760, 850, 820, 700, 590, 510, 450, 560]
actual = [480, 520, 610, 700, 820, 910, 880, 760, 640, 560, 500, 620]
b = benchmarks(hist, m=12, h=12)
print(b["snaive"][:3], b["drift"][0], b["naive"][0])   # [430, 470, 560] 566.9565 560
print(round(mase(actual, b["snaive"], 44.17), 3))        # 1.207
```

Expected output: `[430, 470, 560] 566.9565 560` then `1.207`, both matching section 6. Four
cautions. `y[T - m + (i % m)]` is the seasonal index; an off-by-one puts the benchmark a period
out of phase. The MASE scale must come from training history and be passed in. Use the seasonal
scale for seasonal series; the two scales produce different rankings, as section 6.5 shows. Drift
is undefined at `T = 1`, so guard it.

## 8. Data, standards, and regulatory context

No international standard defines a forecasting benchmark, a backtest protocol, or a pass rule.
Analysts sometimes assume otherwise and cite a standard that does not say what they think it says.
What exists is a vocabulary of measurement, a set of calendar conventions, and a body of model
governance practice that applies by analogy. Knowing which is binding and which is habit is part
of the skill.

### 8.1 What the standards supply

| Standard or framework | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of error, bias, variance, estimator | Any forecast benchmark or backtest design | B |
| ISO 5725-1 | Trueness and precision of measurement methods | Accuracy as a combination of bias and precision | Forecast evaluation, which has no repeated reference | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | "Accuracy" is not a quantity and has no numerical value | A benchmark, a metric, or a pass rule | B |
| ISO 8601 | Date and time representation | Week numbering, durations, ordering | Which week a demand observation belongs to | B |
| SR 11-7 (U.S. Federal Reserve) | Model risk management for banking | Model validation, documentation, ongoing monitoring | Demand planning models, which it does not govern | B |
| Sarbanes-Oxley Section 404 | Internal control over financial reporting, U.S. listed firms | Process controls over figures feeding reported financials | The statistical design of a backtest | B |

The VIM point transfers most directly. "Forecast accuracy of 85%" is a conformance rate against an
implicit tolerance, exactly as "inventory accuracy of 98%" is. It is a statistic about a definition
and a sample, not a property of the forecast. The standard does not forbid the phrase; it forbids
pretending the phrase has a unique meaning. That is why the benchmark and the metric definition
must travel together in every report.

### 8.2 Calendars, and why they change the benchmark

Every benchmark is computed over buckets, and buckets are conventions. ISO 8601 weeks start on
Monday; a 4-4-5 retail calendar does not. A calendar month is 28 to 31 days, so a monthly series
contains a moving amount of "day" that the forecast is implicitly asked to absorb.

Three consequences. **Seasonal naive depends on bucket alignment:** with a 4-4-5 calendar and
`m = 52` the lag is exactly 364 days and weekday alignment is preserved; with ISO weeks it is 364
days plus or minus one, and the benchmark picks up a small calendar artefact. **Holidays move:** a
week containing a public holiday looks like a demand drop, and if the holiday moves between years
seasonal naive attributes the drop to the wrong week. **Month length affects monthly series:**
February is 10% shorter than January, so a model ignoring trading days shows a systematic February
error that a fixed-lag benchmark also shows — the comparison is fair but both are wrong.

Module 01A topic 04 treats the calendar machinery. The point here is narrow: name the calendar in
the protocol, because a benchmark computed on one calendar and a model trained on another is not a
comparison.

### 8.3 Model governance, by analogy

Listed companies in the United States operate under internal control requirements that reach
forecast accuracy where it feeds revenue guidance or inventory valuation. Those controls govern
the process — who may change a forecast, how the change is documented, whether the model is
validated — not the statistic.

The useful template is banking model risk management, which requires a documented purpose, a
defined scope, independent validation, and ongoing monitoring against a benchmark. SR 11-7 is not
law for demand planning, and this course does not present it as such. It is a design pattern that
answers "what would an auditor want to see": the protocol, the population, the benchmark
definition, the result, and the decision, all with dates and versions. That is the practical
content of pre-registration — not a statistical technique, but the document that lets a successor
reconstruct why the company runs the model it runs.

### 8.4 What is convention rather than requirement

A MASE of 1.0 as the naive threshold is a definition, not a standard. "Beat the benchmark by 5%"
is a convention no regulator has endorsed. `m = 12` for monthly data is arithmetic, not policy. The
0.5 CV threshold used to screen intermittent items is a convention from the
Syntetos–Boylan–Croston classification, and its boundary has been corrected in the literature. The
claim that "50% of forecasts fail to beat naive" is a finding from specific practitioner studies,
reported in section 9 with its provenance. An analyst who can separate a requirement from a
convention can challenge a convention; one who cannot will treat every inherited parameter as a
physical constant.

## 9. Industry practice and case evidence

### 9.1 The M-competition lineage

The forecasting competitions are the closest thing the field has to a controlled comparison,
because the same data and metric are applied to every entrant by an independent organiser. They
are also why "simple methods are hard to beat" is a cliché rather than a hypothesis. The lineage
matters because the answer changed.

| Competition | Year | Scale | Headline result | Grade |
|---|---|---|---|---|
| M-competition | 1982 | A small set of mostly economic series | Simple methods found to be at least as accurate as complex ones | A |
| M3 | 2000 | 3,003 series, 24 methods | "Statistically sophisticated or complex methods do not necessarily provide more accurate forecasts than simpler ones"; THETA and ForecastPro best overall; the Naive2 benchmark competitive on several subsets | A |
| M4 | 2018 | 100,000 series, 61 methods | Of the 17 most accurate methods, 12 were combinations; six pure machine learning (ML) methods performed poorly, none beat the combination benchmark, only one beat Naive2; the winner was 19.3% better than Naive2 | A |
| M5 | 2022 | 42,840 daily retail series, 5,507 teams | The winner scored a weighted root mean squared scaled error (WRMSSE) of 0.520 against 0.671 for the best exponential smoothing benchmark, 22.4% better; only 7.5% of teams beat that benchmark, while 48.4% beat the naive forecast | A |

Module 06, which would cover machine learning foundations, is deferred in this edition; the
statistical background needed to read this table is in Module 04.

Three things are worth extracting, and one is worth refusing to extract.

**Extract 1 — the benchmark moved faster than the methods.** In 1982 and again in 2000 the finding
was that simple methods were competitive. By 2018 and 2022 complex methods won, but the comparator
had also changed, from naive to a tuned exponential smoothing benchmark. The literature's "simple
methods" and a practitioner's "naive forecast" are not the same bar, and most of the apparent
contradiction dissolves once you notice that.

**Extract 2 — the spread across entrants is enormous.** Only 7.5% of M5 teams beat exponential
smoothing while 48.4% beat naive. That distribution is the empirical answer to "what share of
attempts add value". It is a distribution over *teams*, not over *series*, and should not be
quoted as the latter.

**Extract 3 — gains concentrate at aggregate levels.** The M5 organisers report that accuracy gains
shrink from roughly 53% at the top aggregation level to roughly 3% at the lowest product-store
level. The bottom level is where replenishment decisions are made. This is the most important
caveat for supply chain practice, and it is reported by the competition's own organisers.

**Refuse to extract — a universal number for "the share of forecastable series".** No competition
reports this and none is designed to. The 7.5% figure is a share of teams beating a specific
benchmark on a specific dataset. Using it as a planning assumption is a category error; use it as a
warning about base rates.

### 9.2 What the competitions do not establish

The competitions are built on retail and macroeconomic series. Industrial demand differs in ways
that matter, and the literature says so directly.

| Difference | Consequence for the benchmark | Grade |
|---|---|---|
| Competition series are long, clean, and continuous; industrial item histories are short and censored by stock-outs | Benchmarks look stronger on competition data because the history is more informative | A |
| Competition data have no promotions being actively changed by the forecaster | Seasonal naive can look strong where the seasonal pattern is undisturbed | A |
| Competition evaluation is typically single-origin and concealed; deployment is rolling | Single-origin results can be an artefact of the chosen split | A |
| Competition metrics are statistical; industrial decisions have asymmetric costs | A method that wins on sMAPE can lose on service and inventory cost | A |
| Competition data are grocery retail in one country and period | Intermittency and entropy are lower than in Kaggle retail datasets, and much lower than in industrial demand | A |

These are not reasons to ignore the competitions. Treat the result as a prior about *method
families* rather than a prediction about *your portfolio*: combinations of simple methods are
strong, and the benchmark is usually harder to beat than it looks.

> **Note:** M5's strongest benchmark was bottom-up exponential smoothing, not seasonal naive. If
> your project reports only the naive comparison, you are using a weaker bar than the competition
> used, and your improvement percentage is not comparable with the published ones.

### 9.3 Forecast value added in practice

FVA compares each step of the forecasting process — statistical model, planner override, consensus
review — against the step before it, on the same series and horizon. A step with negative FVA made
the forecast worse and is a candidate for elimination.

| Claim | Source and grade |
|---|---|
| Across eight supply chain companies and more than 300,000 forecasts, 52% failed to beat the naive forecast | Morlidge, reported in the FVA practitioner literature; Grade C, practitioner study |
| Around 40 to 50% of low-level forecasts perform worse than naive in consulting experience | Morlidge, as summarised in the same literature; Grade E, experience claim |
| A statistical forecast added about 5% FVA at Newell Rubbermaid, while judgmental overrides subtracted about 2% | Schubert and Rickard (2011), as summarised in the FVA literature; Grade C, single company |
| Judgmental adjustments improved accuracy or bias for just over half of SKUs across six studies and about 147,000 forecasts; upward adjustments were more likely to hurt | Fildes, Goodwin and De Baets (2024), *International Journal of Forecasting*; Grade A |
| A perfect method on a pure-noise series would achieve an MSE ratio of about 0.5 and an MAE ratio of about 0.7 against naive | Goodwin, as summarised in the FVA literature; Grade C for the summary, derivable in principle |

The last row deserves care. It is a statement about the theoretical floor on a series with no
signal, and it implies that even a perfect forecaster looks only about 30% better than naive on MAE
when the series is pure noise. If your model is 30% better than naive on a series you believe is
noise, the likelier explanation is leakage or a lucky evaluation window than a breakthrough.

### 9.4 Why practitioners argue about benchmarks

The argument is not really statistical. It is about what a benchmark is for. **The research view**
holds that naive is the correct null: it encodes "no structure", so beating it is evidence of
structure, and any stronger benchmark contaminates the experiment with modelling choices of its
own. **The operational view** holds that the incumbent is the correct comparator, because the
decision is whether to change what people do on Monday. **The vendor view** prefers whichever
benchmark produces the largest defensible improvement, which is usually naive. All three are
internally coherent; the failure is to answer a question from one view using a number produced
under another. The fix is procedural: report both comparisons, label them, and state which one the
decision uses.

The gap between recommendation and practice is wide: benchmark-beside-accuracy is uncommon,
benchmarking on the same backtest is rare, and pre-registered pass rules are very rare. That gap is
why this topic comes first: a team that adopts only the discipline, with no new algorithm, will
improve outcomes more than one that adopts an algorithm and keeps reporting accuracy in a vacuum.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No benchmark reported | The accuracy slide has one number and no comparator | Improvement defined against nothing | Uninterpretable claims; no basis for approval | Require a benchmark row on every accuracy slide |
| Benchmark only against naive on a seasonal series | "We beat naive by 78%" on data with a clear annual pattern | Naive is a weak floor when seasonality dominates | Improvement overstated by the size of the seasonal effect | Compute all four; report the strongest |
| Benchmark on a different window | The model has eighteen origins, the benchmark has one | Different sampling variation and regime | Comparison is an artefact of the design | One backtest, every competitor, same origins |
| Benchmark not refitted | The benchmark is applied unchanged across three years | The model is fresher than the comparator | Model wins for reasons unrelated to modelling | Refit on the same schedule, or state the difference |
| Benchmark with look-ahead | The naive forecast uses a revised history extract | Information unavailable at the origin enters the benchmark | Benchmark is wrong in an unpredictable direction | Reconstruct point-in-time history; audit one origin by hand |
| Seasonal period set wrongly | The seasonal benchmark scores far worse than naive | Wrong `m`, or an off-by-one index | Seasonality declared absent when it is present | Verify `h = 1` returns `y[T-m]` by hand |
| Forecastability never assessed | The project plan has modelling tasks and no triage task | Uniform method on heterogeneous series | Effort concentrated where it cannot pay | Triage by class before fitting; report the split |
| CV used as the forecastability score | The "easy" items are the flat ones | CV measures variation, not predictability | Models built where there is nothing to learn | Use benchmark skill and seasonality strength alongside CV |
| Pooled metric without segmentation | One portfolio number, no per-class table | Gains in one class offset losses in another | A losing model is deployed portfolio-wide | Report per class and per-series win rate |
| Metric chosen after results | The protocol document is dated later than the results deck | Selection on the evaluation set | Inflated apparent performance | Pre-register; version the protocol |
| Population differs between candidate and benchmark | The model excludes new items; the benchmark does not | Different denominators and difficulty | Comparison is void | Fix the population rule before the run |
| Beating naive confused with beating the incumbent | "We beat the baseline" with no incumbent number | The incumbent contains information the benchmark lacks | A regression is shipped as an improvement | Report both comparisons; name which governs |
| Winner's curse on a small backtest | A 2% gain on 40 series and 12 origins | Sampling variation exceeds the effect | A model that is not better gets deployed | Require the gain to exceed twice its standard error |
| Zero forecast rewarded by MAE | The best "model" on slow movers forecasts zero | MAE favours under-forecasting on intermittent series | Service failure hidden by a good score | Score intermittent series on a cost or service metric |
| Benchmark retired after launch | No benchmark in the monthly pack | Drift in either direction is invisible | A decayed model runs for years | Monitor the gap monthly, not just at launch |

The three most consequential: **benchmark only against naive on a seasonal series** is the most
common way an improvement is overstated, and usually not dishonest — the analyst computed the
benchmark their tool provides by default. **Metric chosen after results** is the most common
institutional failure, and arrives by drift rather than decision: a secondary metric is added, then
becomes primary, and within a year nobody can reconstruct the original rule. **Beating naive
confused with beating the incumbent** is the most expensive, because it is the one that produces a
deployment.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The evaluation window resembles the deployment period | A backtest is an estimate of future performance | Promotions, new items, a structural break, a pandemic | Compare error distributions across folds; hold out a full seasonal cycle |
| The history is long enough to estimate seasonality | Seasonal naive and seasonal MASE need complete cycles | Fewer than two cycles; a product launched mid-season | Count complete cycles, not calendar length |
| The actual is measured without material error | Every benchmark treats the actual as truth | Scan undercount, receipt lag, restatement | Compare first-posted and final actuals on a sample |
| Benchmark and candidate share a population | A comparison of unlike sets is meaningless | New-item exclusion applied to one side only | Diff the series identifier lists |
| The benchmark is computable at the origin | A benchmark using later information is not a benchmark | Revised history, late promotion calendar | Rebuild one origin from the point-in-time snapshot |
| The metric reflects the decision's cost | Symmetric metrics assume symmetric costs | Perishability, service penalties, expediting premiums | State the critical ratio; test under an asymmetric loss |
| Errors are stationary enough for a mean to be meaningful | A mean error assumes a stable process | Regime change, retraining, demand shift | Plot error over time; test for a break |
| Per-series results are comparable in scale | Pooling assumes commensurable units | Mixed fast and slow movers | Stratify, or scale before pooling |
| One forecast per series per target is scored | Multiple vintages allow selection on noise | Reforecasts, overrides, "best of" reporting | Audit the number of vintages per target |
| The incumbent can be reconstructed | The deployment comparison needs an incumbent number | The old forecast was never archived | State the limitation; use the benchmark as a proxy |

Two limits of the framework are worth stating plainly.

**Forecastability cannot be measured, only estimated.** Every proxy in section 5.3 is computed on
history and assumes the future's predictability resembles the past's. When it does not — a new
competitor, a regulatory change, a channel shift — the estimate is wrong in the direction that
matters, because the series usually becomes less forecastable, not more. Re-estimate on a rolling
basis rather than once at project start.

**Clearing the benchmark proves structure exists, not that your method found it.** Persistence and
operating value are separate claims, each needing its own evidence, and a project that stops at the
benchmark has done the easy half.

## 12. Exercises

### Exercise 1 — Compute four benchmarks (calculation, 45 minutes)

A quarterly series has 12 quarters of history, `m = 4`, and four held-out quarters.

| Quarter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Demand (units) | 100 | 130 | 90 | 110 | 108 | 140 | 96 | 118 | 116 | 150 | 104 | 126 |

Holdout, quarters 13 to 16: 124, 160, 110, 134.
**Deliverable:** the four benchmark forecasts, their errors, MAE, both MASE variants, and skill
against the mean forecast. **Acceptance criteria:** forecasts to one decimal; both scales computed
with their source periods stated; one sentence naming the correct floor. **Time:** 45 minutes.

### Exercise 2 — Choose the floor (analysis, 30 minutes)

Name the benchmark that should be the floor for each, with a one-sentence justification: (a) a
weekly SKU with a strong summer peak and a flat year-on-year level; (b) monthly steel prices over
six years with no seasonal pattern; (c) a new product with 14 months of history and rapid
adoption; (d) a spare part with demand in 4 of the last 24 months.
**Deliverable:** four lines naming a benchmark, the seasonal period if any, and the reason.
**Acceptance criteria:** at least one answer names a benchmark other than naive; at least one
states that no benchmark is adequate and proposes a different action. **Time:** 30 minutes.

### Exercise 3 — Estimate forecastability (analysis, 40 minutes)

Using Exercise 1, compute the CV, the benchmark skill of seasonal naive against the mean forecast,
and the proxy `1 − Var(in-sample seasonal difference) / Var(series)`. Classify the series.
**Deliverable:** three numbers, the class, and a caveat about what the CV alone would have said.
**Acceptance criteria:** all three computed from stated windows; the class follows from the
numbers; the caveat names the direction of the CV error. **Time:** 40 minutes.

### Exercise 4 — Audit a protocol (critique, 45 minutes)

A protocol states: "Model accuracy is measured as weighted absolute percentage error (WAPE) on the
2024 holdout. The model was tuned on 2022–2023 and validated on 2024. Baseline is the naive
forecast on 2024 actuals. Items launched after January 2022 are excluded. Deployment target is a
10% improvement." Identify at least five defects.
**Deliverable:** a numbered list, each with the mechanism and a corrected statement. **Acceptance
criteria:** at least five distinct defects; at least two concern the comparison rather than the
model; each correction is implementable. **Time:** 45 minutes.

### Exercise 5 — Write a pre-registration (design, 60 minutes)

Write the one-page protocol for a replenishment forecast with a six-week lead time, a weekly
planning cycle, 300 SKUs across four distribution centres, and an asymmetric cost structure where a
stock-out costs three times a holding unit.
**Deliverable:** the protocol with six frame choices, four benchmarks, metric definition,
population rule, and pass rule. **Acceptance criteria:** horizon matches the decision; the metric is
asymmetric and the critical ratio is stated; the pass rule has a numeric threshold and a stability
condition; the owner is a role. **Time:** 60 minutes.

### Exercise 6 — The two comparisons (calculation, 30 minutes)

A model has MAE 88. The naive benchmark scores 100, the incumbent 80, and the standard error of the
difference between model and incumbent is 5.0. Compute both skill values, apply the pass rule from
section 5.6, and state the decision.
**Deliverable:** the two skills, the threshold arithmetic, and a one-paragraph decision.
**Acceptance criteria:** both comparisons computed; `max(0.05 × 80, 2 × 5.0)` evaluated explicitly;
the decision stated, not implied. **Time:** 30 minutes.

### Exercise 7 — FVA across a chain (calculation, 40 minutes)

A process has four steps. MAE by step: naive 100, statistical 85, planner override 82, consensus
review 86. Compute the FVA of each step against the step before it, and the total against naive.
Explain why the step values do not sum to the total.
**Deliverable:** a table of steps, benchmarks, FVA percentages, and the reconciliation.
**Acceptance criteria:** each FVA names its own benchmark; the discrepancy is explained
mechanically; the step with negative FVA has a stated action. **Time:** 40 minutes.

### Exercise 8 — Transferability memo (critique, 50 minutes)

A vendor cites an M5 result — "the winning method beat exponential smoothing by 22.4%" — as
evidence that their product will improve a pharmaceutical distributor's forecast by a similar
margin. Write a one-page memo assessing the claim.
**Deliverable:** the memo, with the claim restated accurately, three reasons the transfer is
uncertain, and what evidence would change your assessment. **Acceptance criteria:** the 22.4% is
identified as one team's result against one benchmark on one dataset; at least one reason concerns
data structure and one concerns evaluation design; the evidence requested is gatherable.
**Time:** 50 minutes.

### Exercise 9 — The conversation (design, 40 minutes)

Using section 6.8, write the five sentences you would deliver to the model's owner in 90 seconds,
plus three questions you expect and a short answer to each.
**Deliverable:** the five sentences and the three question-and-answer pairs. **Acceptance
criteria:** a number in the stakeholder's units; a mechanism rather than a verdict; a specific ask;
an explicit statement of what would change your mind; no comparison to naive alone. **Time:** 40
minutes.

## 13. Self-check answers

**13.1 Exercise 1.** History totals 1,388 over 12 quarters, so the mean is 115.667.

```text
mean            115.667 for all four horizons
naive           y_12 = 126 for all four horizons
seasonal naive  y_9, y_10, y_11, y_12 = 116, 150, 104, 126
drift           slope = (126 - 100) / 11 = 2.3636
                forecasts 128.36, 130.73, 133.09, 135.45
```

| Method | Forecasts | Sum abs error | MAE | Skill vs mean |
|---|---|---:|---:|---:|
| Mean | 115.7, 115.7, 115.7, 115.7 | 76.67 | 19.17 | 0.000 |
| Naive | 126, 126, 126, 126 | 60.00 | 15.00 | 0.217 |
| Seasonal naive | 116, 150, 104, 126 | 32.00 | 8.00 | 0.583 |
| Drift | 128.36, 130.73, 133.09, 135.45 | 58.17 | 14.54 | 0.241 |

```text
non-seasonal scale = 294 / 11 = 26.727
seasonal scale     =  66 /  8 =  8.250
MASE (non-seasonal): mean 0.717, naive 0.561, snaive 0.299, drift 0.544
MASE (seasonal):     mean 2.323, naive 1.818, snaive 0.970, drift 1.763
```

The correct floor is seasonal naive: the pattern is strong and stable, and 0.970 on the seasonal
scale says it repeated almost exactly, so a candidate must beat 8.0 units of MAE, not 15.0. A common
wrong answer uses `y_13` as the first seasonal-naive forecast, shifting the benchmark by a quarter
and producing a score near 26 — enough to conclude wrongly that the series has no seasonality.

**13.2 Exercise 2.** **(a)** Seasonal naive with `m = 52`, because the pattern dominates and the level
is stable; if the level trends, seasonal naive plus drift is the better floor and should be stated.
**(b)** Naive, because a price series typically behaves as a random walk and a seasonal benchmark
would fit noise; drift is worth computing as a check. **(c)** Neither: 14 months cannot establish a
seasonal pattern and the adoption curve is not a random walk, so the honest benchmark is the mean
forecast with a caveat that the series is in a regime the history does not cover. **(d)** Naive with
a zero floor, or a Croston-family benchmark — but on 4 nonzero months in 24 MAE rewards a forecast
of zero, so the comparison must run on a cost or service metric.

**13.3 Exercise 3.**

```text
CV                      = SD / mean = 17.34 / 115.667 = 0.150
seasonal-naive skill    = 1 - 8.00 / 19.17 = 0.583
seasonal difference var = variance of [8, 10, 6, 8, 8, 10, 8, 8] = 1.438
series variance         = 300.56
seasonal strength proxy = 1 - 1.438 / 300.56 = 0.995
```

The series is structural: skill 0.583 and seasonal strength near 1.0 point the same way. The caveat
is that CV 0.150 alone would place it in a "low variation" bracket that some screening rules treat
as easy and others as flat, because the seasonal amplitude is moderate relative to the mean.

**13.4 Exercise 4.** Six defects, and there are more.

1. **Tuning and validation share the year.** If hyperparameters were selected on 2024, the score is
   in-sample. Tune on 2022–2023 with an inner rolling origin and touch 2024 once.
2. **The model's origin schedule is unstated.** Naive on 2024 actuals is a one-step benchmark if
   recomputed weekly and a twelve-step benchmark if computed once. Fix both.
3. **Population exclusion is one-sided.** If the model learned from the excluded items' category
   patterns, the comparison is not like-for-like. Apply the same exclusion and report the count.
4. **WAPE is symmetric and the costs are not.** No critical ratio is stated. State the asymmetry or
   justify a symmetric metric explicitly.
5. **The 10% target has no stability condition.** One holdout can produce 10% by chance; require the
   gain to hold in two of three folds.
6. **No incumbent comparator.** The target is measured against naive, not the current process, so it
   cannot support a deployment decision.

**13.5 Exercise 5 rubric.** A strong protocol fixes a horizon of seven weeks (six-week lead time plus
one review week), `m = 52`, active SKUs with at least 104 weeks of history, all four benchmarks on
weekly rolling origins over the last 52 weeks, and pinball loss at the critical ratio
`3/(3+1) = 0.75` as primary with WAPE secondary. The pass rule names a numeric margin and requires
the gain in two of three folds; the owner is a role. A weak protocol says "MAPE on a holdout, target
10% improvement" and leaves benchmark, population, and horizon implicit.

**13.6 Exercise 6.**

```text
skill vs naive     = 1 - 88/100 =  0.12
skill vs incumbent = 1 - 88/80  = -0.10
threshold          = max(0.05 * 80, 2 * 5.0) = max(4.0, 10.0) = 10.0
required MAE      <= 80 - 10.0 = 70.0 ; actual MAE = 88
```

The model fails threshold 2 by a wide margin. It beats naive, which establishes that exploitable
structure exists, and it loses to the incumbent, which is the decision-relevant comparison. Do not
ship. The correct framing is that the incumbent already captures the structure, plausibly through
promotion calendars or customer intelligence the model lacks. A common wrong answer reports "12%
improvement" and stops, answering the research question and skipping the deployment question.

**13.7 Exercise 7.**

| Step | MAE | Benchmark | FVA vs benchmark |
|---|---:|---|---:|
| Naive | 100 | — | — |
| Statistical | 85 | Naive (100) | +15.00% |
| Planner override | 82 | Statistical (85) | +3.53% |
| Consensus review | 86 | Override (82) | −4.88% |
| **Final vs naive** | **86** | **Naive (100)** | **+14.00%** |

The steps sum to 13.65%, not 14.00%. The discrepancy is the base effect: each percentage uses a
different denominator, so increments are not additive; the multiplicative reconciliation is
`0.8500 × 0.9647 × 1.0488 = 0.8600`, which is `86/100`. The consensus review destroyed 4.88% of the
error reduction; investigate that step rather than eliminating the chain, and do not add the
percentages, which makes it look less harmful than it is.

**13.8 Exercise 8 rubric.** A strong memo restates the claim: one winning team, on 42,840 Walmart
daily series, against a bottom-up exponential smoothing benchmark, on a revenue-weighted
hierarchical metric, achieved 22.4%. It then gives three reasons the transfer is uncertain: data
structure (contract-driven, censored, intermittent demand, with M5 gains shrinking to roughly 3%
at the lowest level); the benchmark (the distributor's incumbent is not exponential smoothing);
and design (M5 was single-origin and concealed). A weak memo accepts the claim or dismisses
competitions entirely.

**13.9 Exercise 9 model answer.**

1. "On the 31 items outside the nine-item deployment, the model averages 27.5 eaches of monthly
   error against 17.1 for the seasonal benchmark — 61% more, about €41,000 a year in holding and
   stock-out cost."
2. "The model is sound; the series are not. Those items have almost no repeatable pattern, and 24
   months is too little history for 38 features, so it is fitting noise."
3. "On the eleven structural items it beats the benchmark by 45% on average; seven of those are
   where I would deploy."
4. "I am asking for a nine-item deployment and a benchmark-only policy on the rest, reviewed in two
   quarters."
5. "If we get three years of history, or aggregate to family level, I will re-run the backtest."

Expected questions: "Did you compare it to naive?" (yes, and it wins, which is why the seasonal
benchmark is the relevant comparator); "Can we tune it?" (not without a pre-registered rule,
because tuning against the holdout makes the result unverifiable); "Will the vendor accept this?"
(the vendor's figure is a portfolio average, and the segmentation is the finding).

## 14. Interview questions

**1. "What is a benchmark forecast and why does every model need one?"**

A strong answer names the four benchmarks, says each encodes "no structure" or the simplest
structure, and adds that the benchmark must share the model's backtest. A weak answer calls it "a
baseline" and cannot say what it controls for.

**2. "Why is the naive forecast so hard to beat?"**

A strong answer explains that many business series are close to random walks, so the last value is
near-optimal, and that naive has no parameters and therefore no estimation error; on a seasonal
series, seasonal naive is the correct bar. A weak answer says "because demand is random".

**3. "What is forecastability and how would you measure it?"**

A strong answer defines it as the share of variation a method could in principle remove, and gives
two measures: skill of the best cheap benchmark over the mean forecast, and seasonality strength. A
weak answer equates it with a low CV or with data quality.

**4. "A model beats naive by 30% but loses to the current process. What do you conclude?"**

A strong answer separates the questions: beating naive shows structure exists, beating the incumbent
decides deployment, and here the incumbent already captures the structure. Do not ship. A weak
answer recommends shipping because 30% sounds large.

**5. "How would you set up a fair benchmark comparison?"**

A strong answer lists the design choices: same origins, horizons, population, refit schedule,
metric, and history vintage, with the benchmark refitted on the same schedule. A weak answer says
"use the same test set".

**6. "What did the M competitions establish about simple methods?"**

A strong answer gives the arc: simple methods were competitive through M3, complex and combined
methods won in M4 and M5, but the benchmark also got stronger and few entrants beat the tuned one.
A weak answer says "simple methods win" without the comparator.

**7. "What are the limits of citing competition results in a business case?"**

A strong answer names data structure (retail scanner versus censored, intermittent industrial
demand), the benchmark used, the single-origin design, and the absence of cost-based metrics; the
result is a prior about method families, not a forecast of your gain. A weak answer treats the
winner as a recommendation.

**8. "What is forecast value added, and what does a negative value tell you?"**

A strong answer defines FVA as the change in a metric attributable to one process step against the
step before it, notes that a negative value means the step hurt, and that the percentages are not
additive. A weak answer defines it as "improvement over last year".

**9. "How would you tell a stakeholder their model lost to seasonal naive?"**

A strong answer gives the number in the stakeholder's units, the mechanism, the segment where the
model does work, a specific ask, and what would change the analyst's mind. A weak answer softens
the finding until it disappears, or delivers a verdict on the model's author.

**10. "Why must a benchmark be computed over the same horizon as the model?"**

A strong answer explains that error variance grows with horizon, so a one-step benchmark and a
twelve-step model measure different tasks, and pooling horizons hides convergence toward the
benchmark. A weak answer says "to be fair" without the mechanism.

**11. "What would make you decide not to build a model at all?"**

A strong answer describes a triage: if the best cheap benchmark cannot beat the mean forecast by
more than a few percent, or the series is lumpy and short, or the aggregation gain exceeds any
model gain, then the deliverable is a policy change, a safety-stock adjustment, or better data. A
weak answer cannot imagine declining the project.

## 15. Key terms

- **Benchmark forecast** — the cheapest defensible forecast a candidate must beat.
- **Mean forecast** — the average of history repeated; the null model.
- **Naive forecast** — last observation carried forward.
- **Seasonal naive** — the value from the same period one season earlier.
- **Drift** — last observation plus the average change per period.
- **Naive2** — seasonally adjusted random walk used as the M3 benchmark.
- **Random walk** — a series whose first differences are white noise; naive is optimal.
- **Forecastability** — the share of variation a method could in principle remove.
- **Coefficient of variation** — standard deviation over mean; a description, not a skill score.
- **Strength of seasonality** — one minus remainder variance over deseasonalised variance.
- **Seasonal period `m`** — observations per cycle; 12 monthly, 52 weekly.
- **Backtest** — evaluation on data not used to fit or tune.
- **Rolling origin** — repeated re-fitting as the origin advances.
- **Forecast origin** — the moment the forecast is made.
- **Forecast horizon** — periods between origin and target.
- **MASE** — MAE divided by the in-sample naive MAE; below 1 beats the benchmark.
- **RMSSE** — the squared-error analogue of MASE.
- **Skill score** — one minus candidate error over benchmark error.
- **Forecast value added** — the improvement attributable to one process step.
- **Incumbent** — the process currently producing the operational forecast.
- **Challenger** — a candidate evaluated against the incumbent.
- **Pre-registration** — fixing the metric, population, and pass rule before results.
- **Pass rule** — the written condition under which a challenger ships.
- **Beat-the-naive rate** — share of series on which a method beats naive.
- **Noise floor** — the error no method can remove.
- **Intermittent demand** — demand with zero periods between nonzero observations.
- **ADI and CV²** — average inter-demand interval and squared variation of nonzero sizes.
- **Look-ahead bias** — using information unavailable at the origin.
- **Aggregation gain** — error reduction from forecasting a sum rather than its parts.
- **WRMSSE** — the revenue-weighted hierarchical metric used to score M5.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Makridakis, S. and Hibon, M. (2000). "The M3-Competition: results, conclusions and implications."
  *International Journal of Forecasting*, 16(4), 451–476. DOI: 10.1016/S0169-2070(00)00057-1.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2018). "The M4 Competition: Results,
  findings, conclusion and way forward." *International Journal of Forecasting*.
  DOI: 10.1016/j.ijforecast.2018.06.001.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time
  series and 61 forecasting methods." DOI: 10.1016/j.ijforecast.2019.04.014.
- Makridakis, S. and Petropoulos, F. (2019). "The M4 competition: Conclusions." *International
  Journal of Forecasting*. DOI: 10.1016/j.ijforecast.2019.05.006.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). "M5 accuracy competition: Results,
  findings, and conclusions." *International Journal of Forecasting*, 38(4), 1346–1364.
  DOI: 10.1016/j.ijforecast.2021.11.013.
- Makridakis, S., Petropoulos, F., and Spiliotis, E. (2022). "The M5 competition: Conclusions."
  *International Journal of Forecasting*. DOI: 10.1016/j.ijforecast.2022.04.006.
- Fildes, R., Goodwin, P., and De Baets, S. (2024). "Forecast value added in demand planning."
  *International Journal of Forecasting*. DOI: 10.1016/j.ijforecast.2024.07.006.
- Goodwin, P., Petropoulos, F., and Hyndman, R. J. (2017). "A note on upper bounds for
  forecast-value-added relative to naïve forecasts." *Journal of the Operational Research Society*.
  DOI: 10.1057/s41274-017-0218-3.
- Spiliotis, E., Koulakos, K., and Assimakopoulos, V. (2020). "Are forecasting competitions data
  representative of the reality?" DOI: 10.1016/j.ijforecast.2018.12.007.
- Bojer, C. S. and Meldgaard, J. P. (2020). "Kaggle forecasting competitions: An overlooked
  learning opportunity." DOI: 10.1016/j.ijforecast.2020.07.007.
- Hyndman, R. J. and Koehler, A. B. (2006). "Another look at measures of forecast accuracy."
  *International Journal of Forecasting*, 22(4), 679–688. Source of MASE; DOI not verified here.
- Syntetos, A. A., Boylan, J. E., and Croston, J. D. (2005). "On the categorization of demand
  patterns." *Journal of the Operational Research Society*, 56(5), 495–503. Source of ADI and CV².

**Preprints — Grade A for method, not for peer review**

- Theodorou, E., Wang, S., Kang, Y., and Spiliotis, E. "Exploring the representativeness of the M5
  competition data." arXiv:2103.02941.
- Makridakis, S., Fry, C., Petropoulos, F., and Spiliotis, E. "The future of forecasting
  competitions: Design attributes and principles." arXiv:2102.04879.

**Books, standards, and official — Grade B**

- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, current edition.
  Section 3.1 defines the four benchmark methods; section 12.8 covers training and test sets.
- ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1*. Current edition.
- ISO 5725-1, *Accuracy (trueness and precision) of measurement methods and results — Part 1*.
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM).
- ISO 8601, *Date and time — Representations for information interchange*.
- Board of Governors of the Federal Reserve System (2011). *Supervisory Guidance on Model Risk
  Management*, SR 11-7. Cited by analogy; it does not govern demand planning models.
- M4 and M5 competition repositories and evaluation code, GitHub. Grade B for the published
  evaluation method.

**Practitioner and vendor — Grade C, D, or E**

- Gilliland, M. "FVA: A Reality Check on Forecasting Practices." *Foresight*, Spring 2013.
  Grade C.
- Morlidge, S. "FVA and the limits of forecastability," practitioner presentations and articles,
  2013–2015. Grade C for the 52% figure; Grade E for the consulting-experience ranges.
- Schubert, K. and Rickard, K. (2011). Newell Rubbermaid forecast value added case, as summarised
  in the FVA literature. Grade C, single company.
- Manning, P. *Demand Forecasting Best Practices*. Practitioner text, forecastability chapter.
  Grade C.
- Hyndman, R. J. "Benchmarks for forecasting." Blog post, robjhyndman.com/hyndsight/benchmarks/.
  Grade C despite the author's standing.
- APICS/ASCM Dictionary, current edition. Grade C.
- SAS forecast value added white paper. Grade D, vendor material.

**Note on citations.** Every competition figure in section 9.1 is reported as the cited paper states
it, and the 52% FVA figure is a practitioner study quoted at second hand and labelled accordingly.
Every figure in section 6, the portfolio table in 6.7, and the Exercise 1 data are constructed, and
the construction is shown. No statistic here is invented, and no figure is quoted more precisely
than its source supports.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic, exponential smoothing as the first family of challengers:
  [`02_exponential-smoothing.md`](02_exponential-smoothing.md)
- Backtesting machinery that makes the comparison fair: [`08_rolling-origin-backtesting.md`](08_rolling-origin-backtesting.md)
- FVA and override measurement: [`10_overrides-fva.md`](10_overrides-fva.md)
- Intermittent demand, where MAE misleads: [`06_intermittent-demand.md`](06_intermittent-demand.md)
- Error metrics and their cost interpretation:
  [`../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- Intermittent demand classification: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Demand and lead-time distributions: [`../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Regression with time-series structure: [`../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Power and minimum detectable effect, for sizing a backtest: [`../../01_foundations/05_experimentation_pilots_evidence/06_power-mde.md`](../../01_foundations/05_experimentation_pilots_evidence/06_power-mde.md)
- Hierarchical, intermittent, and probabilistic forecasting:
  [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
- Promotions and demand sensing, where the incumbent usually wins:
  [`../09_demand_sensing_promotions_npi/_index.md`](../09_demand_sensing_promotions_npi/_index.md)
- Inventory policy downstream of the forecast:
  [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Part map for Methods: [`../_index.md`](../_index.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Hierarchical and probabilistic forecasting survey: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
