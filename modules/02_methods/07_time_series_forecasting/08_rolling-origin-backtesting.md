# Rolling-Origin Backtesting

> Module 07 — Time Series Forecasting Foundations
> Topic 08 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topic 09; Module 04 topics 02 and 05; Module 07 topics 01–07
> Version 2026.5

---

## 1. Why this topic exists

A forecast model is a claim about the future. A backtest is the argument for that claim. Most
forecasting arguments are made badly, and in a specific way: someone fits a model on the first
eighty percent of a series, scores it on the last twenty, reports one number, and the number is
treated as a property of the method.

It is not a property of the method. It is a property of the method, the weeks that happened to
fall in the holdout, and the choice of where the cut was made. Move the cut four weeks later and
the same method on the same series moves several percentage points. This topic exists because
that movement is larger than most of the improvements teams spend quarters chasing.

Three failures recur.

**One. The holdout that was one draw.** A planner compares exponential smoothing against a
seasonal-naive baseline on a twelve-week holdout containing a promotion. Exponential smoothing,
which has no promotional information, loses by nine points, and the team concludes it is
unsuitable. A regression model is built and evaluated on a holdout containing a stock-out that
censored demand, and wins. Neither conclusion follows. Each is a single draw presented as a
verdict.

**Two. The pooled number that describes the experiment.** A team reports "MAPE 18.4% at four
weeks." The figure averages four horizons, one to four weeks. Short horizons are more numerous in
that average, so it is weighted towards the easiest forecasts. It cannot be compared with another
team's 22.1% unless the weighting, the denominator convention, and the item mix match. Two numbers
that look like the same measurement are two experiments.

**Three. The selection reported as the evaluation.** Twenty configurations are scored on the same
forty origins, and the best is reported with its score. That score is the maximum of twenty
correlated random variables, so it is optimistically biased by construction. The bias grows with
the number of candidates and shrinks with the number of origins, and in the common case it is
larger than the difference between the two best candidates.

**What production actually does.** Production does not run a holdout. It runs a sequence of
one-shot forecasts, each made when some data exists and some does not, each scored against what
happened afterwards. The correct backtest imitates that sequence. It is called rolling-origin
evaluation, and its defining move is that the cut point advances through time and the procedure
is re-run at each position.

**The counter-example worth knowing.** The M4 forecasting competition, the largest in the field,
did not use rolling origins. Its protocol was a fixed-origin holdout: one train/test split per
series, frequency-dependent horizons, no splitting strategy in the evaluation catalogue. It got
away with it because it evaluated a very large cross-section of series, so the cross-section
supplied the replication the time axis did not. You do not have that cross-section. If you have one
series, or five hundred, you must manufacture replication along time. That is the whole argument
for this topic.

**What it costs.** Rolling origins buy a distribution instead of a point, a horizon profile
instead of an average, and a defence against the accident of the cut. They cost compute, data
(the evaluation window must be long), and clarity (there is a table, not a number). The rest of
this topic is about spending those costs well.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** why a single train/test split is one draw, and **quantify** how far the score moves
   when the cut point moves, on your own series.
2. **Design** a rolling-origin evaluation for a stated decision: first origin, step, origins,
   horizon, window scheme, and refit policy, each with a one-sentence justification.
3. **Choose** between expanding and sliding windows on stated evidence about regime stability, and
   **state** what would change the choice.
4. **Compute** an error matrix indexed by origin and horizon, and **derive** per-horizon accuracy,
   bias, and dispersion from it.
5. **Compute** the effective number of independent origins under overlapping windows, and
   **explain** why forty origins do not carry forty observations at a twelve-week horizon.
6. **Separate** evaluation of a fixed method from selection among methods, and **design** a
   protocol in which selection and reporting origins are disjoint.
7. **Backtest** a process containing a human override, distinguishing a recorded override from a
   reconstructed one, and **state** which claims the design supports.
8. **Specify** the minimum backtest window for a stated seasonal period, and **test** a proposed
   design against it.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three objects: a moving cut point on a timeline, a two-dimensional array of errors, and a
distinction between measuring and choosing.

### 3.1 The origin is a pretence

A forecast origin is a moment at which you pretend to stand. Everything left of it is known,
everything right of it is unknown, and the forecast uses exactly the information available at that
moment. Nothing later is allowed to inform it: not the outcome, not a revised figure, not a
promotion calendar published the following month.

The pretence is the method. When it breaks, the backtest stops measuring the forecasting problem
and starts measuring your ability to fit the past. The break is rarely deliberate. It is almost
always a join to a table built later, or a revision replacing the number that was visible at the
time.

```text
  THE ORIGIN IS A MOVING CUT POINT
  ==================================================================
  week ->    1     26     52     78    104    130    156
             |------|------|------|------|------|------|
  history   ##############################################

  EXPANDING WINDOW: start fixed at week 1, end moves

  O1  [######### fit 1..104 #########]|hold 105..108|
  O2  [########## fit 1..108 ##########]|hold 109..112|
  O3  [########### fit 1..112 ###########]|hold 113..116|
  O4  [############ fit 1..116 ############]|hold 117..120|
                                          ^
                                    horizon = 4 weeks,
                                    one block per origin

  SLIDING WINDOW: length fixed at 52, both ends move

  O1       [#### fit 53..104 ####]|hold 105..108|
  O2        [#### fit 57..108 ####]|hold 109..112|
  O3         [#### fit 61..112 ####]|hold 113..116|
  O4          [#### fit 65..116 ####]|hold 117..120|

  Each row is one fold: one forecast the organisation would
  actually have made, and one outcome it would have observed.
```

Read the diagram for its three parameters, because they are the only design choices you have. The
**first origin** sets how much history the earliest fold gets. The **step** is the distance
between origins, and it sets how many folds a fixed span yields. The **horizon** is the width of
the holdout block, and it sets which decision the backtest is about.

### 3.2 The error matrix is the object

A backtest does not produce a number. It produces a matrix with origins on one axis and horizons on
the other, and every honest summary reduces that matrix along a stated axis. Section 6.2 prints one
in full; the shape is what matters here.

```text
                h=1    h=2    h=3    h=4
  origin W100    +4     -7    +12    +19
  origin W104    -3     +6    +11    +15
  ...            ...    ...    ...    ...
  per-horizon   4.7    7.8   13.7   18.3   MAE, units
  column mean   1.7    2.8   10.7   17.0   bias, units
  column sd     5.2    8.6   10.3   11.0   dispersion, units
```

A pooled MAE of 11.1 is a weighted average of the four column means, with weights set by the
number of origins that reached each horizon — a property of the design.

Three readings answer three different questions.

- **Down a column** — "at this horizon, how did the method do across occasions?" This is what a
  decision at that horizon asks, and the spread down the column is the uncertainty on the answer.
- **Across a row** — "at this origin, how did the forecast degrade with distance?" This is the
  horizon profile, and its slope is the method's signature.
- **Over the whole matrix** — "on average, over this mix of occasions and distances, what is the
  error?" The least useful of the three, because the mix is an artefact, and the one most often
  reported.

### 3.3 Horizons do not average

Error grows with horizon. For a random walk with innovation standard deviation `sigma`, the
`h`-step forecast error has standard deviation `sigma * sqrt(h)`. The mechanism is additive: the
`h`-step error is a sum of `h` independent innovations, variances add, and the standard deviation
grows as the square root.

The consequences are blunt. A one-week and a twelve-week forecast sit on scales differing by about
3.5 under that model, so averaging their errors gives a number whose value depends on the ratio in
which they were mixed — and the ratio is set by your design, not your method. Change the maximum
horizon from twelve weeks to four and the pooled number improves although the method is unchanged,
because you removed the hardest forecasts.

The rolling design makes this worse in a way that is easy to miss. With `K` origins and maximum
horizon `H`, the number of forecasts at horizon `h` is `K - h + 1`. Horizon 1 gets every origin;
horizon `H` gets only the earliest `K - H + 1`. Pool the matrix and short horizons dominate, so
the pooled number flatters any method that is bad at distance.

### 3.4 Overlapping origins are not independent observations

Adjacent origins share almost all their training data, and their errors at long horizons share
almost all their innovations. For a random walk, the `h`-step errors from origins `t` and `t + 1`
share `h - 1` of their `h` innovations, so their correlation is `(h - 1) / h`.

At `h = 1` that correlation is zero and adjacent origins are genuinely independent. At `h = 12` it
is `11/12 = 0.917`, and forty origins carry roughly the information of two. The effective sample
size for the mean error at horizon `h`, under an AR(1) approximation with autocorrelation `rho_h`,
is approximately `K (1 - rho_h) / (1 + rho_h)`.

This is the most under-appreciated fact about backtesting, and it has three consequences. The
standard error of a long-horizon accuracy estimate is much larger than `sd / sqrt(K)` suggests.
Adding origins at a fine step adds very little information at long horizons. And the dispersion
column matters more than the point estimate, because it honestly represents how little you know.

### 3.5 Measuring is not choosing

Two activities wear the same clothes. **Evaluation** asks how well a fixed procedure would have
done. **Selection** asks which of several procedures to keep. They need different data, and doing
both on one set of origins converts the first into a biased version of the second.

The mechanism is the maximum of correlated random variables. Score twenty candidates on the same
origins and the best score is not an unbiased estimate of that candidate's skill; it estimates the
best score achievable by twenty candidates on this sample. White's Reality Check formalises the
correction, and the illustration in that paper is worth remembering: a naive p-value of 0.0036 for
the best of a search became 0.2040 once the search was accounted for.

The remedies are older than the problem. Hold origins back: select on the first two-thirds, report
on the last third, and never let reporting origins inform selection. Pre-register the candidate
set so the search space is known. Or sidestep selection: combine the candidates rather than
choosing among them, which the previous topic in this module argues for and which holds up better
than most selection procedures on real demand series.

Every backtest also runs on three clocks, and leakage is what happens when they collapse into one.
The **decision clock** is when the forecast had to be frozen and acted on. The **data clock** is
when each input became visible. The **outcome clock** is when the outcome was recorded. A design
respecting all three is reproducible; one respecting none produces a number that is internally
consistent, externally unverifiable, and usually too good.

## 4. Definitions and vocabulary

The fourth column names the unit or coding convention. A term without one will be used two ways
in the same meeting.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Forecast origin | The moment a forecast is made; all later information is excluded | Confused with the first date of the holdout | Timestamp or period index |
| Forecast horizon | Periods between the origin and the target period | Confused with the length of the holdout | Periods, at the series frequency |
| Decision lead time | The interval between using a forecast and the outcome it concerns | Assumed equal to the forecast horizon | Days or weeks |
| Fold | One origin with its training data and its holdout | Used to mean the whole backtest | Integer index `k = 1..K` |
| Expanding window | Fixed start, moving end | Called "rolling" | Start index fixed |
| Sliding window | Fixed length, both ends moving | Called "expanding" when the step is small | Length in periods |
| Step size | Distance between consecutive origins | Set to 1 without regard to cost | Periods |
| Holdout | The periods scored in one fold | Called "the test set" when there is only one | Period range |
| Rolling-origin evaluation | Re-running the procedure at advancing origins | Called cross-validation without qualification | `K` folds |
| Fixed-origin evaluation | One origin with a multi-period holdout | Reported as if it were a backtest | One fold |
| Time series cross-validation (TSCV) | Rolling origins with one-period holdouts and a growing window | Used interchangeably with `k`-fold CV | `K` folds |
| Backtest | Any re-execution of a forecasting procedure against historical outcomes | Used to mean a single holdout | Protocol plus result |
| Error matrix | Errors indexed by origin and horizon | Collapsed to a mean before inspection | `K x H`, units |
| h-step error | Actual minus forecast for the period `h` ahead of an origin | Signed and unsigned versions mixed in one column | Units |
| Pooled error | A measure over all origins and horizons | Compared across designs with different horizon mixes | Units, weighted |
| Mean absolute error (MAE) | Mean of absolute errors | Compared across series in different units | Units |
| Root mean squared error (RMSE) | Square root of mean squared error | Used without stating that it penalises large errors | Units |
| Mean absolute percentage error (MAPE) | Mean of absolute errors divided by actuals | Used when actuals can be zero or near zero | Percent, undefined at zero |
| Symmetric MAPE (sMAPE) | Percentage error symmetric in forecast and actual | Assumed bounded by 100% | Percent, bounded by 200% in common forms |
| Mean absolute scaled error (MASE) | MAE divided by an in-sample naive MAE | Denominator computed on the wrong window | Dimensionless |
| Bias | The mean signed error | Reported as accuracy | Units |
| Naive benchmark | Last observation carried forward | Omitted because it looks trivial | Units |
| Seasonal naive benchmark | Observation from one seasonal cycle earlier | Omitted when the series is short | Units |
| Seasonal cycle | Periods after which the pattern repeats | Assumed rather than estimated | Periods, e.g. 52 weeks |
| Regime change | Persistent shift in the data-generating process | Confused with a single outlier | Date of break |
| Structural break | Testable change in process parameters | Asserted from a plot | Test statistic and date |
| Look-ahead bias | Using information unavailable at the decision clock | Introduced by a join, not by intent | Boolean per input |
| Leakage | Any path by which future information reaches the model | Confused with overfitting | Boolean per input |
| Selection bias | Optimism from choosing the best of several candidates on one sample | Reported as skill | Difference in metric |
| Data snooping | Reusing one dataset for search and inference | Treated as a moral failing, not a statistical one | Number of candidates |
| Reality Check | Bootstrap test of whether the best of a search beats a benchmark | Applied without stating the candidate set | p-value |
| Effective sample size | Independent observations equivalent to a dependent sample | Reported as the raw origin count | Count |
| Origin correlation | Correlation between errors from adjacent origins at a fixed horizon | Assumed zero | Dimensionless |
| Vintage | A dataset as it existed at a stated date | Ignored when only the current snapshot exists | Snapshot date |
| Restatement | A revision changing a previously published actual | Treated as truth for all history | Revision date |
| Refit policy | Whether parameters are re-estimated at each origin | Differing between backtest and production | `refit` / `update` |
| Forecast value added (FVA) | Metric change attributable to one process step | Computed against the previous step only | Metric difference |
| Override | A manual adjustment to a statistical forecast | Treated as noise or as expertise without evidence | Units and author |
| Diebold-Mariano test | Test of equal predictive accuracy between two forecasts | Applied to nested models without correction | Test statistic |
| Model confidence set | Models not distinguishable from the best at stated confidence | Reported as a ranking | Set of models |

Four of these carry most of the practical damage.

**Horizon versus lead time.** A forecast horizon is a property of the model run. A decision lead
time is a property of the business. They coincide only when someone designed it that way. If your
replenishment lead time is six weeks and your backtest is one week ahead, you have measured a
decision nobody makes.

**Pooled versus per-horizon.** A pooled error is a weighted average whose weights come from the
design. It is not wrong; it is uninterpretable without the weights and the horizon mix. Two teams
can report the same pooled number from designs that disagree at every horizon.

**Selection versus evaluation.** The same computation on different samples with different
purposes. Doing both on one sample is not a shortcut; it is a different and biased estimator. The
bias shrinks with more origins or a held-back reporting sample, not with better models.

**Vintage versus truth.** The number in the warehouse today is not the number the planner saw at
the origin. For most demand series the difference is small; for series subject to restatement,
promotion reclassification, or item renumbering, it is the difference between a valid backtest and
a fictional one.

## 5. The mechanics, step by step

Five phases, twenty steps. Each step names what you do and what fails if you skip it.

### Phase A — Fix the evaluation question

**Step 1. Name the decision and its clock.** "The weekly replenishment order for item 4471 at DC
Leeds, placed Friday for arrival in four weeks." Write the day and the cut-off hour. Failure: a
monthly backtest used to justify a daily process.

**Step 2. Convert the decision to a horizon.** An order placed now and arriving in four weeks is
driven by a four-week forecast, so `H = 4`. Failure: `H` from the software default, so the backtest
scores a decision nobody makes.

**Step 3. Fix the unit and aggregation level.** Units per item per location per week. Every later
number inherits this, and changing it mid-project invalidates comparisons already made. Failure:
an item-level backtest for an item-location decision, hiding the failures that matter.

**Step 4. Choose the metric family before seeing results.** Typical error (MAE), large-error
avoidance (RMSE), cross-series comparability (MASE or sMAPE), or bias. Failure: the metric changes
after the results arrive, which is metric shopping and ends the evaluation as evidence.

**Step 5. Name the benchmark and make it the incumbent.** Naive, seasonal naive, or the current
process including its overrides. Failure: no benchmark, so a MAPE of 14% has no interpretation.

**Step 6. State whether this is evaluation or selection.** One pre-specified method is evaluation,
and all origins may be reported; several candidates is selection. Failure: the same origins serve
both, and the reported number is a maximum.

### Phase B — Design the split

**Step 7. Fix the first origin from data sufficiency.** The first fold needs enough training data
to fit the method: for a seasonal method with period `m`, at least two full cycles, and three is
safer. With `m = 52` the first origin cannot precede week 104, and week 156 is better. If the
series is only 140 weeks long, you cannot do this, and must either use a non-seasonal method or
exclude the early folds as unreliable. Failure: the first origin is week 30 because the plot
looked better there.

**Step 8. Fix the step size from compute cost and the resolution you need.** Step 1 gives the most
folds; step `H` gives non-overlapping holdouts and the least correlation between adjacent origins.
Step equal to the planning cycle is a defensible compromise, because it matches the occasions on
which forecasts are actually made. Failure: step 1 with 200 folds, three days of compute, and no
extra information at long horizons where the origins are nearly perfectly correlated.

**Step 9. Fix the number of origins from the outcome span you need.** The span is
`(K - 1) * step + H` periods. Two constraints apply and usually conflict: enough folds to estimate
dispersion, and a span of at least one full seasonal cycle. Failure: `K = 6`, step 4, `H = 4`,
span 24 weeks, and a conclusion about a seasonal method drawn from under half a cycle.

**Step 10. Choose expanding or sliding, and justify it in one sentence.** The justification is a
claim about how much the past resembles the future, and it belongs in writing. Section 5.2 gives
the decision table. Failure: the window scheme is the library default and nobody knows which
default it is.

**Step 11. Fix the refit policy and match it to production.** Either re-estimate parameters at
every origin or hold them fixed and update only the state. Production does one of these on a
schedule; the backtest must do the same one on the same schedule. Failure: the backtest refits
weekly, production quarterly, and the difference is unmeasured optimism.

**Step 12. Fix the horizon strategy.** A direct forecast fits one model per horizon; a recursive
forecast fits one and iterates. Direct is more robust at long horizons; recursive is cheaper and
smoother. The backtest must use what production uses, and the matrix must record which. Failure:
the backtest uses direct, production iterates, and the reported accuracy belongs to a system that
does not exist.

**Step 13. Fix the vintage policy.** Record whether the backtest uses data as it exists today or
as it existed at each origin. If no revision history exists, say so and state the direction of the
bias. Failure: a retroactive promotion reclassification makes the model look as though it
anticipated an event it never saw.

### Phase C — Run and check

**Step 14. Build the error matrix before computing any summary.** One row per origin, one column
per horizon, signed errors in the unit of the series. Inspect the table before reducing it.
Failure: the summary is computed first and the matrix never exists, so the structure that would
have explained it is discarded.

**Step 15. Check every input for look-ahead.** For each feature, ask what date it became knowable;
anything knowable after the origin is excluded. This is a manual audit, not a software flag.
Failure: a calendar table containing next year's holidays gives the model foreknowledge of
demand-suppressing weeks.

**Step 16. Check the ragged edge.** The last origin with complete outcomes defines the end of the
evaluation. Near that edge, `n` falls with horizon, and the matrix will have missing cells that
silently change the sample per column. Failure: `n` runs from 26 at horizon 1 to 23 at horizon 4,
and the per-horizon comparison uses different samples without saying so.

**Step 17. Check origin correlation and compute effective sample size.** Use `rho_h` from the error
series, or the random-walk approximation `(h - 1) / h`, and report `K (1 - rho_h) / (1 + rho_h)`.
Failure: a standard error computed as `sd / sqrt(K)` at horizon 12, understated by a factor near
five.

**Step 18. Check seasonal coverage.** Compare the outcome span with `m`. If the span is under one
cycle, restrict the claims to what it supports: a short-horizon ranking, not a certification of a
seasonal method. Failure: a twelve-week backtest used to reject a method whose advantage only
appears at the seasonal turn.

**Step 19. Check the benchmark on identical origins and horizons.** A benchmark computed on the
full history while the candidate is computed on the holdout is a comparison of two experiments.
Failure: the benchmark is not in the table and the table cannot be read.

### Phase D — Separate selection from evaluation, then report

**Step 20. Split the origins.** Use earlier origins for selection and later ones for reporting.
Two-thirds/one-third is conventional, and the reporting block must still pass Step 18. Failure:
selection and reporting on the same origins, so the reported number is a maximum.

**Step 21. Report the table, not the number.** Origins, horizon, metric, value, dispersion, and
`n`, plus the design parameters: window scheme, length, step, refit policy, vintage. Failure: one
number on a slide, and a reviewer who cannot reconstruct it.

**Step 22. Report the dispersion.** Standard deviation or interquartile range of the per-origin
errors at each horizon. This is where the honest uncertainty lives. Failure: an MAE of 18.3 units
without the 11.0-unit standard deviation that makes it indistinguishable from 22.

**Step 23. Report the limits.** State what the backtest does not cover: future regime change, a
promotion absent from the history, the human step if excluded, horizons beyond `H`. Failure: the
backtest is read as a guarantee, and the first quarter outside its window is treated as a model
failure rather than as out of scope.

**Step 24. Freeze the protocol.** Write the design down before running it, date it, file it with
the results. The next run uses it or states why it changed. Failure: protocol drift between
quarters, and a quarter-on-quarter improvement that is a protocol change.

### 5.1 How many origins

There is no published rule, and the search for one is instructive. Tashman's review gives the
counting arithmetic — a test period of length `N` yields `N` forecasts under a fixed origin and
`N(N + 1) / 2` under a rolling origin advancing one period at a time — and a sufficiency rule for
the minimum number of forecasts at the longest lead time, but no general prescription for `K`.

| Constraint | Expression | What it sets |
|---|---|---|
| Minimum forecasts at the longest horizon | Test period `N >= H + M - 1` for `M` forecasts at `H` | A floor on the span |
| Outcome span | `(K - 1) * step + H` periods | Coverage of a seasonal cycle |
| Effective sample size | `K (1 - rho_h) / (1 + rho_h)` at horizon `h` | The information available |
| Compute budget | `K` fits of the method | An upper bound |

A defensible default for a weekly series with an annual cycle is `K` between 26 and 52, step equal
to the planning cycle, span at least 104 weeks. Where that is impossible, reduce the claim rather
than the honesty.

The variance consequence deserves the arithmetic. The standard error of a mean error from `K`
origins is not `sd / sqrt(K)` when errors are correlated; under AR(1) it is inflated by
`sqrt((1 + rho) / (1 - rho))`. At `rho = 0.917`, the twelve-week random-walk case, that factor is
`sqrt(1.917 / 0.083) = 4.81`, so a standard error that looks like 1.8 units is really 8.7.
Doubling `K` divides it by 1.41, not 2, and the return falls further as the step shrinks. That is
the arithmetic behind a rule practitioners state without it: a long-horizon backtest with many
origins is not much better than one with few, and reporting dispersion is the only way to say so.

### 5.2 Expanding versus sliding

The choice is a statement about stationarity, with a clean bias-variance structure.

| Scheme | Estimation variance | Misspecification bias | Right when | Wrong when |
|---|---|---|---|---|
| Expanding | Falls as history grows | Rises if old regimes persist | Parameters stable; history informative | A break invalidated most history |
| Sliding | Fixed, higher | Falls after a break | Recent regime differs; enough post-break data | Short windows leave too few observations |
| Post-break only | Highest | Lowest | A dated break with a long post-break sample | The break date is uncertain |

Expanding is the default for demand series, for three reasons. Demand parameters are usually
stable enough that more history reduces estimation variance more than it imports bias. Sliding
adds a second tuning parameter, the window length, which then needs validating out of sample, and
choosing it on the evaluation sample is itself selection. And rare events — a promotion, a
stock-out, a competitor exit — are observed only in the distant history, so a sliding window
removes the only evidence about them.

Sliding is right when a break is dated and large. The theoretical result is that the optimal
pre-break history depends on break size, post-break sample length, model, and loss function, which
is a complicated way of saying there is no universal answer.

> **Note:** The empirical record is friendlier to expanding windows than the regime-change
> argument predicts. Work on window selection under recent breaks has found expanding windows
> usually most accurate after a break, partly because many breaks are volatility changes where
> longer estimation helps, and partly because post-break samples are too short to estimate
> anything well. Do not assume a sliding window will rescue you after a break; test it. Practical
> discipline: run both, report both, and treat a large disagreement as a finding about the series,
> because it is non-stationarity a single-window backtest would have hidden.

### 5.3 Horizon handling

With `K` origins stepped by 1 and maximum horizon `H`, horizon `h` receives `K - h + 1` forecasts.
The pooled average's weights are therefore a design artefact.

| Max horizon `H` | Forecasts at `h = 1` | Forecasts at `h = H` | Share of forecasts at `h <= 4` |
|---:|---:|---:|---:|
| 4 | 26 | 23 | 100% |
| 8 | 26 | 19 | 60% |
| 13 | 26 | 14 | 32% |
| 26 | 26 | 1 | 17% |

The horizon strategy also changes the profile. Direct forecasts fit `H` models and avoid error
compounding; recursive forecasts fit one and iterate, accumulating error through the model's own
dynamics. The two are not comparable across strategies, so the profile must record which was used.

The reporting rule: never average across horizons without publishing the per-horizon table. The
average is a weighted mean of that table with the weights above, and a reader with the table can
form any average they want.

### 5.4 Refit, update, and vintage

The backtest must reproduce the production fitting schedule. **Refit** re-estimates all parameters
at each origin; **update** holds them fixed and updates only the state. A model re-estimated weekly
beats the same model re-estimated quarterly, so a weekly-refit backtest against a quarterly-refit
planning system measures a system that does not exist.

**Vintage** is the same problem applied to data rather than parameters. Using today's revised
figures at a historical origin is a small look-ahead when revisions are small and a large one when
they are not: modest for shipments and invoices, moderate for scanner data, variable for inventory
snapshots, and large for promotion calendars, where retrospective coding changes sit exactly on the
events that matter. If the warehouse keeps no `valid_from` and `valid_to`, the vintage policy is
"today's data for all origins," and the report must say so.

### 5.5 Backtesting a process with a human override

Overrides are where a backtest most often becomes fiction, because the override was a decision
made by a person with information that is not in the dataset.

| Design | What you do | Question it answers | What it cannot answer |
|---|---|---|---|
| Replay recorded overrides | Use override values as entered at the time | How did the process, including people, perform? | Anything about a different person or policy |
| Simulate a rule | Replace the person with a stated rule on available data | How would a specified override policy perform? | Whether real planners would follow the rule |
| Exclude the override | Score the statistical forecast alone | How good is the model? | Whether the process adds value |

Replay is the only design that evaluates the process, and it is valid only if the override was
recorded before the outcome was known and the record has the value, timestamp, and author. If
overrides are reconstructed afterwards from a variance report, the replay inherits the
reconstructor's hindsight and the evaluation is worthless.

**Forecast value added** (FVA) is the measurement framework. Each step — naive, statistical,
statistical plus override, consensus, final — is scored on the same origins and horizons, and each
step's contribution is the change in the metric. Negative FVA means the step made the forecast
worse, and the standard remedy is to remove the step or retrain the people in it.

Two cautions. FVA against the previous step rewards a step that undoes the previous step's error,
which is not the same as adding information, so report FVA against the naive benchmark as well. And
the human sees information the model does not — a customer call, a competitor's stock-out — which
the backtest also cannot see. A process that looks value-destroying on recorded data may be
compensating for a data gap, and the fix is the data, not the person.

### 5.6 The seasonal cycle rule

The evaluation window must contain at least one full seasonal cycle, and two is better. The rule
is arithmetic.

```text
  outcome span  =  (K - 1) * step + H          periods

  one cycle:       (K - 1) * step + H  >=  m
  two cycles:      (K - 1) * step + H  >=  2m

  with m = 52, K = 26, step = 4, H = 4:
       span = 25 * 4 + 4 = 104 weeks  =  2 cycles        PASS

  with m = 52, K = 6, step = 4, H = 4:
       span = 5 * 4 + 4  =  24 weeks  =  0.46 cycles    FAIL
```

Why it exists. A seasonal method's advantage appears when the holdout contains the season in which
its seasonal estimate is wrong in a way a non-seasonal method cannot be. A twelve-week window in a
stable mid-season period shows almost no difference between seasonal and non-seasonal methods,
because there is no seasonal variation inside the window to get right or wrong. Rejecting the
seasonal method on that evidence rejects it for a reason unrelated to its quality.

The training side has a parallel rule: a seasonal model needs two cycles of training data to
estimate `m` indices, and three to estimate them against a trend. Combine the two and you get a
requirement that surprises most teams: `2m` of training plus `2m` of evaluation plus `H`, or 212
weeks for `m = 52`, `H = 4`. Section 6.7 works this through.

### 5.7 The protocol document

A backtest that cannot be re-run is an anecdote. The protocol is one page, written before the run,
recording ten fields: the decision and its clock; the horizons, reported separately; the
origination scheme (window type, first origin, step, `K`); the refit policy; the horizon strategy;
the vintage; the benchmark and its protocol; the metric and its denominator; the selection rule and
the origins it used; and the reporting format. Filed with the results, it converts a number into
evidence. Section 6.1 shows one.

## 6. Worked example

Meridian Home distributes replacement filter cartridges for domestic water systems. Part FH-220
is a steady seller with a pronounced annual pattern: demand rises through spring and falls in late
summer, averaging about 240 units a week. All figures below are constructed for teaching and are
internally consistent. Section 7 reproduces the arithmetic in code.

### 6.1 The question and the design actually used

The decision: place a replenishment order every four weeks, for arrival four weeks later. The
forecast that drives it is four weeks ahead. A separate expediting decision consumes a one-week
forecast, so both horizons are evaluated.

| Field | Value | Justification given at the time |
|---|---|---|
| History | 156 weeks | Everything in the warehouse |
| Horizons | 1 to 4 weeks | Four is the lead time |
| Scheme | Expanding | The default |
| First origin | Week 100 | "Enough history" |
| Step | 4 weeks | Matches the order cycle |
| Origins | 6 (weeks 100, 104, 108, 112, 116, 120) | "Six seemed enough" |
| Refit | Every origin | Correct |
| Benchmark | Seasonal naive | Correct |
| Metric | MAE in units | Matches the unit cost |

The outcome span is `(6 - 1) * 4 + 4 = 24` weeks against a 52-week cycle: 0.46 of a cycle. That is
the design's central defect, and it is invisible in every number the design produces.

### 6.2 The error matrix

Two methods were scored: exponential smoothing with additive trend and additive seasonality
(ETS(A,A,A)), and the seasonal-naive benchmark.

**Table 6.1 — Signed errors, ETS(A,A,A), units**

| Origin | `h = 1` | `h = 2` | `h = 3` | `h = 4` |
|---|---:|---:|---:|---:|
| W100 | +4 | −7 | +12 | +19 |
| W104 | −3 | +6 | +11 | +15 |
| W108 | +8 | +14 | +21 | +26 |
| W112 | −6 | −8 | −9 | −4 |
| W116 | +2 | +5 | +13 | +22 |
| W120 | +5 | +7 | +16 | +24 |
| **Mean** | **+1.7** | **+2.8** | **+10.7** | **+17.0** |

Read the matrix before reducing it. The column means rise monotonically, so the model
systematically under-forecasts and the under-forecast grows with distance. Origin W112 is the only
all-negative row: something around week 112 was absorbed into the level and then reversed, a
datable event that belongs in the investigation, not the average. Origin W108 is the worst row at
every horizon, so a summary over six origins partly describes that one.

### 6.3 The per-horizon arithmetic

```text
  MAE:  h=1  (4+3+8+6+2+5)   /6 =  28/6 =  4.667 units
        h=2  (7+6+14+8+5+7)  /6 =  47/6 =  7.833 units
        h=3  (12+11+21+9+13+16)/6 = 82/6 = 13.667 units
        h=4  (19+15+26+4+22+24)/6 =110/6 = 18.333 units
        pooled = 267/24 = 11.125 units

  Bias: h=1  10/6 =  1.667      h=3  64/6 = 10.667
        h=2  17/6 =  2.833      h=4 102/6 = 17.000

  RMSE: h=1 sqrt(154/6)  =  5.066      h=3 sqrt(1212/6) = 14.213
        h=2 sqrt(419/6)  =  8.357      h=4 sqrt(2338/6) = 19.740

  SD of signed errors, n-1 denominator:
        h=1 sqrt(137.333/5) =  5.241    h=3 sqrt(529.333/5) = 10.289
        h=2 sqrt(370.833/5) =  8.612    h=4 sqrt(604.000/5) = 10.991
```

Dispersion grows with horizon, which is expected and worth reporting in its own right. At `h = 4`
the standard deviation of 11.0 units is 60% of the MAE of 18.3. An accuracy claim of "18.3 units"
with an 11.0-unit spread is a different claim from one with a 3-unit spread, and the reader cannot
tell them apart without the second number.

### 6.4 The comparison with the benchmark

**Table 6.2 — MAE by horizon, two methods, units**

| Horizon | ETS(A,A,A) | Seasonal naive | Difference | Winner |
|---|---:|---:|---:|---|
| `h = 1` | 4.667 | 5.167 | −0.500 | ETS |
| `h = 2` | 7.833 | 8.500 | −0.667 | ETS |
| `h = 3` | 13.667 | 9.000 | +4.667 | Seasonal naive |
| `h = 4` | 18.333 | 9.833 | +8.500 | Seasonal naive |
| Pooled | 11.125 | 8.125 | +3.000 | Seasonal naive |

The ranking reverses between horizons. ETS is better at one and two weeks; the seasonal naive is
better at three and four, by a wide margin at four. The pooled number says the seasonal naive
wins, which is true for the pool and false for half the decision.

Which horizon matters? Replenishment consumes a four-week forecast, so the answer there is the
seasonal naive; expediting consumes a one-week forecast, so the answer there is ETS. A pooled
comparison supports neither conclusion. That is the practical content of "horizons must be scored
separately": here the pooled metric points to the wrong method for the decision that matters most.

### 6.5 The scale-free view

With the scale set to the in-sample seasonal-naive MAE at the first origin, **9.4 units**, ETS gives
MASE 0.50, 0.83, 1.45 and 1.95 at horizons one to four, against 0.55, 0.90, 0.96 and 1.05 for the
seasonal naive. Above 1 the forecast is worse than the benchmark's historical average, so ETS is
beaten by the benchmark's own long-run behaviour from horizon 3 — a stronger statement than "worse
than the benchmark."

> **Note:** Both the in-sample naive and the in-sample seasonal-naive denominators appear in
> software and give different values. State which you used, and never compare MASE across systems
> without checking that they agree.

### 6.6 What the design could not see

The holdout runs from week 101 to week 124. If the annual peak falls in weeks 118 to 128, the
holdout contains the rise into the peak but not the fall. A method that forecasts the rise and
misses the fall scores well; a flat method scores badly. The ranking at `h = 3` and `h = 4` is
therefore a ranking under one seasonal condition.

The second consequence is subtler. Six origins at a four-week step, with the random-walk
correlation `rho_h = (h - 1) / h`, give:

```text
  n_eff = K (1 - rho_h) / (1 + rho_h)

  h = 1:  rho = 0.000   n_eff = 6.00
  h = 2:  rho = 0.500   n_eff = 2.00
  h = 3:  rho = 0.667   n_eff = 1.20
  h = 4:  rho = 0.750   n_eff = 0.86
```

At `h = 4` the backtest carries less than one independent observation. The standard deviation of
11.0 units is not variability across independent occasions; it describes six correlated ones. The
mean of 17.0 could plausibly be 5 or 29. The design supports a claim about horizon 1 and almost
nothing about horizon 4, even though the point estimate at horizon 4 is the one that matters.

### 6.7 What a compliant design would require

```text
  requirement = 2 cycles training + 2 cycles evaluation + horizon
              = 2*52 + 2*52 + 4 = 212 weeks
  available   = 156 weeks,  shortfall 56 weeks
```

Three designs are available. As run (100 weeks training, 24 weeks evaluation) supports a
short-horizon ranking and nothing seasonal. One cycle each (52 weeks training, 52 plus the horizon
of evaluation) supports a ranking across a full cycle, with unstable seasonal estimates in the
early folds. Full compliance (104 and 104) supports a defensible accuracy claim.

The middle design is the realistic compromise for a five-year-old series: one cycle of training is
the minimum for estimating seasonal indices and is below the two-cycle standard. The honest
sentence is "52 weeks of training, one full cycle of evaluation, seasonal parameters unstable in
the first folds," not "validated on two years of data." If the series is younger than about four
years, reduce the frequency instead — four-week buckets from 39 observations of a four-weekly
series with an annual cycle of 13 restores the ratio at the cost of resolution.

### 6.8 What the analyst reports

```text
  DESIGN     Expanding, first origin W100, step 4, K=6, refit each origin,
             direct multi-horizon, seasonal naive on identical origins.

  RESULT     MAE by horizon, units, ETS(A,A,A):
               h=1  4.7 (sd 5.2, n=6)     h=3 13.7 (sd 10.3, n=6)
               h=2  7.8 (sd 8.6, n=6)     h=4 18.3 (sd 11.0, n=6)
             Pooled MAE 11.1 is not a headline.

  BENCHMARK  Seasonal naive: h=1 5.2  h=2 8.5  h=3 9.0  h=4 9.8

  FINDING    ETS better at h=1 and h=2; seasonal naive better at h=3 and
             h=4. The replenishment decision consumes h=4, so ETS is not
             the method for that decision.

  LIMIT      Span 24 weeks against a 52-week cycle; n_eff at h=4 is 0.86.
             The h=4 comparison is a direction, not an estimate. A
             compliant design needs 212 weeks.
```

## 7. Code, query, or calculation

One block, deliberately small. It reproduces the per-horizon table of section 6.3 from the error
matrix.

```python
# One reduction of the error matrix. Errors are actual - forecast, units.
import statistics as st

origins = [100, 104, 108, 112, 116, 120]          # week numbers
H = 4
err = {
    100: [None,  4, -7, 12, 19],
    104: [None, -3,  6, 11, 15],
    108: [None,  8, 14, 21, 26],
    112: [None, -6, -8, -9, -4],
    116: [None,  2,  5, 13, 22],
    120: [None,  5,  7, 16, 24],
}

for h in range(1, H + 1):
    col  = [err[o][h] for o in origins]
    mae  = sum(abs(e) for e in col) / len(col)
    bias = sum(col) / len(col)
    sd   = st.stdev(col)
    print(f"h={h}  MAE={mae:5.1f}  bias={bias:6.1f}  sd={sd:5.1f}")
```

Expected output, in units: MAE 4.7, 7.8, 13.7 and 18.3 at horizons one to four; bias 1.7, 2.8,
10.7 and 17.0; standard deviation 5.2, 8.6, 10.3 and 11.0.

Three checks before trusting any such reduction.

- **Recompute one column by hand.** The `h = 1` column sums to 28 across six origins; 28/6 is
  4.667, printed as 4.7. A disagreement means the matrix is not the one you think it is.
- **Confirm `n` per column.** All six here. In a real backtest the counts fall with horizon as
  origins run out of outcomes, and a table printing one `n` for all horizons is hiding that.
- **Check the sign convention once.** Errors are actual minus forecast throughout. Reversing it
  turns +17 into −17 and changes the recommendation from "raise the forecast" to "lower it."

> **Note:** `statistics.stdev` uses the `n - 1` denominator. Reporting a population standard
> deviation over the origins you evaluated gives a smaller number by `sqrt(5/6) = 0.913`, and the
> difference matters when dispersion is compared with a threshold. State the denominator.

The arithmetic form, for a reader checking without code: `mae_h = sum |e| / n_h`,
`bias_h = sum e / n_h`, `sd_h = sqrt(sum (e - bias_h)^2 / (n_h - 1))`, and
`pooled_mae = sum_h (n_h * mae_h) / sum_h n_h`. The last expression is the important one: the
pooled figure is a weighted average of the per-horizon figures with weights `n_h`, so it leans
towards short horizons when counts fall with horizon. The number itself does not say which.

## 8. Data, standards, and regulatory context

### 8.1 What the data must contain

| Requirement | Why it matters | What to record | If absent |
|---|---|---|---|
| Frozen outcome table | Errors must be against the outcome, not a later revision | Observation date, value, revision date | Leakage through restatement |
| Origin-addressable inputs | Each forecast must be reproducible from data dated before it | Source table, `valid_from`, load timestamp | Unreproducible forecasts |
| Stated time zone and calendar | Week boundaries and holidays change the series | ISO 8601 offsets; week-numbering rule | Off-by-one folds |
| Stable series definition | Item and location must mean the same thing at every origin | Keys with validity dates | Silent discontinuities |
| Censoring flag | Stock-outs truncate demand | Boolean per item-location-week | Understated error |
| Override log | Human adjustments must be replayable | Value, author, timestamp, reason code | Process cannot be evaluated |

The last is the one most organisations lack and most need. Overrides usually live in a planning
system's audit table, which is not the demand history, which is not the warehouse. Joining them is
a data engineering task that has to happen before the backtest, not after.

### 8.2 The censoring problem

A demand series measured as shipments is truncated at available stock. When service is poor, the
largest true demands are exactly the ones that never appear. A backtest scoring against shipments
therefore rewards a method that under-forecasts, because the actuals are themselves suppressed.

The direction is consistent: the method that forecasts low looks better than it is, and the error
it is rewarded for is the error that causes the stock-out. Three mitigations, in order of
preference: score against unconstrained demand where it can be reconstructed; score against
shipments but report the censoring rate per horizon; or restrict to periods with no stock-out, and
state the selection that creates.

### 8.3 Standards that constrain vocabulary, not method

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Meanings of accuracy, precision, bias, prediction interval | B |
| ISO 5725-1:1994 | Accuracy, trueness, precision of measurement methods | Systematic versus random error | B |
| ISO 8601-1:2019, ISO 8601-2:2019 | Date and time representation | Durations, intervals, week numbering, offsets | B |
| JCGM 100:2008 | Expression of uncertainty | The vocabulary of an uncertainty statement | B |

None mandates rolling origins, a step size, or a number of folds. They settle what the words mean.
A report saying "accuracy" without saying which of ISO 3534-1's related concepts it means will be
read differently by each reader. A backtest claim is therefore a **model-risk claim**, not a
compliance claim, and the documentation that survives review answers six questions: what decision,
what horizons, what origins, what window scheme, what vintage, what dispersion.

### 8.4 Where regulation does impose backtesting

Supply chain forecasting is lightly regulated. Two adjacent domains are not, and their practice is
worth borrowing.

| Regime | Domain | What it requires | Grade |
|---|---|---|---|
| SR 11-7, Federal Reserve model risk guidance (2011) | Banking models | Validation including outcomes analysis, ongoing monitoring, documented conceptual soundness | B |
| Regulation (EU) 2024/1689, AI Act, Article 15 | High-risk AI systems | Accuracy, robustness, cybersecurity, with declared metrics and levels | B |
| FDA guidance on predetermined change control plans | AI-enabled medical devices | A pre-specified plan for model change, with supporting evaluation | B |

The relevance is instructive rather than direct. A demand forecast scheduling production is not a
high-risk AI system under the AI Act, and a distribution centre is not a bank. But the shape of the
obligation transfers: pre-specify the metric, document the evaluation, monitor after deployment,
keep the record that lets someone else re-run the test. That is the protocol of section 5.7,
written for a different reason.

> **Note:** Model risk guidance is written for institutions whose models carry prudential capital
> consequences. Applying its language to a forecasting team is useful discipline, not a legal
> obligation. Do not describe a demand forecast as "SR 11-7 compliant"; the phrase has no meaning
> outside its domain.

### 8.5 Convention rather than requirement

- **A two-thirds/one-third split** between selection and reporting origins is convention with no
  optimality argument. It is defensible because it leaves a reporting block long enough to matter.
- **`K = 52`** is a round number inherited from weekly data with an annual cycle, not a power
  calculation; no published rule supplies one.
- **Step equal to one period** is the software default and usually wrong for multi-step horizons,
  because it generates nearly perfectly correlated folds at distance.
- **"MAPE below 20% is good"** is folklore. The threshold depends on the item, the aggregation
  level, and the cost function, and varies by an order of magnitude across categories.

An analyst who can tell a requirement from a convention can defend a design; one who cannot will
produce the default and call it a standard.

## 9. Industry practice and case evidence

### 9.1 What the competition record shows

| Item | What the record shows | Implication | Grade |
|---|---|---|---|
| M4 evaluation design | Fixed-origin holdout: one split per series, frequency-dependent horizons, no splitting strategy in the evaluation catalogue | A single split is defensible when the cross-section is large | A |
| M4 horizons | Yearly 6, quarterly 8, monthly 18, weekly 13, daily 14, hourly 48 | Horizon is set by frequency, not by decision analysis | A |
| M4 metrics | sMAPE and MASE combined into a weighted average against a seasonal-naive benchmark | Two metrics, two failure modes, one score | A |
| M4 MASE denominator | In-sample seasonal-naive error over the training period | The scaling choice is stated and fixed for all entrants | A |
| M4 use of rolling origins | Appears in some entrants' internal model combination, not in the official protocol | Rolling origins were a technique, not the benchmark | B |

The lesson is not that M4 did it wrong: a fixed-origin holdout is defensible when very many series
each contribute one observation, because the replication is cross-sectional. A company with forty
item-location series has no such replication and must manufacture it along time. Copying the
competition protocol without its cross-section is the error.

> **Note:** The M4 protocol above was checked against the competition's published evaluation code
> and a reproduction notebook that states the design explicitly. Do not assume other competitions
> used the same protocol; each publishes its own, and they differ.

### 9.2 What the methodology literature establishes

| Finding | Source | Implication | Grade |
|---|---|---|---|
| Rolling origins yield `N(N+1)/2` forecasts from a test period of length `N`, against `N` for a fixed origin | Tashman (2000) | The replication argument is arithmetic | A |
| Test period must be at least the maximum horizon; `H + M - 1` gives `M` forecasts at lead `H` | Tashman (2000) | A floor on the evaluation span | A |
| Optimal estimation window depends on break size, post-break sample, model, and loss | Pesaran and Timmermann (2007) | No universal expanding-versus-sliding rule | A |
| Expanding windows usually most accurate after recent breaks, in the series studied | Hännikäinen (2015) | The regime-change intuition is not reliably right | B |
| A verification score without a standard error is of little use | Jolliffe (2007) | Dispersion is part of the result | A |
| The best of a search has an optimistically biased score | White (2000) | Separate selection from evaluation, or correct | A |
| Selection over a large rule universe produces in-sample significance that fails out of sample | Sullivan, Timmermann and White (1999) | The bias grows with the search size | A |
| Backtest overfitting is selection bias under multiple testing; three trials can produce likely-false strategies | Bailey and López de Prado (2021) | Small searches are not safe searches | C |

### 9.3 What practice looks like

Planning functions overwhelmingly evaluate with a single holdout, often a "last twelve weeks"
window, and report one number. Where rolling evaluation exists it is usually a spreadsheet macro or
a planning system's built-in backtest, and the design parameters are defaults nobody in the room
can name.

Two vendor patterns matter because they shape belief. A planning system's accuracy report usually
fixes the horizon at the planning bucket and reports a weighted MAPE across items, so the number
describes the item mix as much as the method. A cloud forecasting service offers a rolling
evaluation with configurable context length, horizon, and step — better, but with defaults chosen
for demonstration and documentation that does not say which horizon the decision consumes. The gap
between the two is not sophistication; it is whether anyone wrote down the decision.

### 9.4 What is missing from the public record

Searches for this topic returned no peer-reviewed study of backtesting practice in commercial
supply chain planning, and no published distribution of how many organisations separate selection
from reporting origins. Claims about "most companies" are impressions.

What the record does support is narrower. The arithmetic of rolling origins is settled and old;
Tashman's review is from 2000 and the counting rules have not changed. The failure modes are
documented in adjacent fields with quantitative consequences: a naive p-value of 0.0036 becoming
0.2040 after adjustment for a search is a concrete, published example of the bias that selection on
the evaluation sample produces. What no source supplies is a rule for how many origins are enough.
Section 5.1 gives the constraints; the choice among them remains a judgement, and a team that
documents its judgement is ahead of one that inherits a default.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Single fixed-origin split | One "test set" with one date range; no other evaluation exists | The score is one draw | Conclusions that reverse under a different cut | Rolling origins, or a stated interval |
| Pooled metric as headline | One MAE or MAPE with no horizon table behind it | Weights come from the design, not the method | Method chosen for the wrong horizon | Publish per-horizon; pool only with weights |
| Tuning on the full series | Hyperparameters chosen on all data, then backtested | Parameters saw the holdout | Optimism of unknown size | Refit and retune inside every fold |
| Selection on the reporting sample | "We compared 20 models and the winner scored..." | The score is a maximum | Inflated winner; ranking may be noise | Hold back reporting origins; state the search size |
| Overlapping origins called independent | Standard error computed as `sd / sqrt(K)` at a multi-week horizon | Adjacent origins share innovations | Standard error understated two- to fivefold | Report effective sample size and dispersion |
| Backtest shorter than a season | Outcome span under `m`; the report never states the span | One seasonal condition only | A seasonal method judged on non-seasonal evidence | Apply the `2m` arithmetic; reduce claims |
| Refit policy differs from production | Backtest refits weekly; production refits annually | Measures a better system than exists | Optimism equal to the refit value | Match the schedule; document it |
| Restated data at historical origins | Revisions, remapped items, recoded promotions applied retroactively | Future information reaches the model | Optimism concentrated on what matters | Vintage policy; state when snapshots do not exist |
| Overrides replayed from memory | Override values reconstructed from a variance report afterwards | Replay inherits hindsight | The process evaluation is fiction | Use only overrides logged before the outcome |
| No benchmark in the table | MAPE alone, or a benchmark on a different window | No scale for the number | Any number can be called an improvement | Recompute the benchmark on identical origins |
| Metric chosen after results | The metric changes between drafts | Metric shopping | The evaluation stops being evidence | Fix the metric in the protocol |
| Ragged edge handled silently | `n` differs across horizons but only one `n` is printed | Late origins lack long-horizon outcomes | Comparisons use different samples | Print `n` per horizon and method |

Two rows deserve expansion. **The pooled metric** is not a lie; every number in it is correct. It
is a summary whose weights are invisible, and the reader supplies an assumption about what they
are. In the worked example the pooled figure names the seasonal naive as winner while the ETS model
wins the horizon that drives the replenishment order. Both are true; only one is about a decision.

**The overlapping-origin problem** survives because the standard error formula is taught for
independent samples and applied without modification to correlated ones. At a four-week horizon
with six origins the effective sample size is under one. A team that reports a mean without
dispersion has not estimated its own sampling error, and that error may exceed every difference
under discussion.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The evaluation period resembles the deployment period | A backtest is evidence about the future only through resemblance | A regime change after the last origin makes the estimate irrelevant | Compare the last 26 weeks with the evaluation window on level, variance, seasonality |
| Inputs are vintage-correct | The forecast must be reproducible from information available at the origin | Leakage inflates accuracy | Audit lineage; look for revision tables |
| The method refits identically at each origin | The backtest must imitate the pipeline | The measured system is not the deployed one | Run one code path; diff backtest and production outputs |
| The series definition is stable | Errors must be comparable across folds | Discontinuities read as demand events | Key-validity audit; flag master-data changes |
| The horizon matches the decision lead time | A backtest at the wrong horizon measures a decision nobody makes | Effort on irrelevant accuracy | Ask the planner which horizon they consume |
| The benchmark is meaningful for this series | A naive benchmark is a floor only if there is structure to beat | A trivially beatable benchmark flatters everything | Compare naive and seasonal naive |
| The origins span the relevant regimes | Conclusions must survive which weeks were evaluated | A ranking that holds in one season only | Split origins and compare |
| Errors are comparable across horizons | Pooling requires a common scale | A pooled number dominated by the easiest horizon | Report per horizon; use scale-free metrics |
| The override can be reproduced or excluded honestly | The process includes people, or the evaluation is of the model alone | Reported process performance is fictional | Check override timestamps precede outcomes |
| The metric maps to the decision's cost | Accuracy is a proxy for what the business cares about | Optimising the metric worsens the decision | Compare metric ranking with cost ranking |

**What no backtest covers.** A future structural break, a competitor's exit, a new channel, a change
in the supply contract. These are not estimable from history, and the dispersion does not include
them. A backtest gives the uncertainty of a method applied to a world like the one evaluated; it
says nothing about a world that has changed.

**The data floor is real.** Section 5.6 derives `2m + 2m + H` periods, or 212 weeks for weekly data
with an annual cycle. Series shorter than that cannot support the strongest claim, and no amount of
compute repairs it. The honest responses are to reduce the claim, reduce the frequency, or wait.

**The cross-sectional shortcut has a price.** A large panel of short series can substitute
cross-sectional for temporal replication, which is how M4 justified one split. But the resulting
statement is about the panel, not about any series in it. A method that wins on average across very
many series can lose on your forty, and the backtest that tells you which is the temporal one.

## 12. Exercises

### Exercise 1 — Move the cut, watch the number (calculation, 40 minutes)

Take a series of at least 100 weekly observations. Fit one simple method and score it on four
single holdouts of eight weeks, with cut points at weeks 68, 76, 84, and 92.

**Deliverable:** the four scores, their range, and the ratio of largest to smallest.
**Acceptance criteria:** same method and metric throughout; cut points stated; one sentence on what
the spread implies for a single-split claim.
**Time:** 40 minutes.

### Exercise 2 — Build the error matrix (calculation, 50 minutes)

Using Exercise 1's method, build a rolling-origin evaluation with `K = 8`, step 4, `H = 4`.

**Deliverable:** the matrix, per-horizon MAE, bias, and standard deviation, and the pooled MAE with
its weights.
**Acceptance criteria:** every cell present or marked missing; `n` per column; the pooled figure
shown as a weighted mean of the columns.
**Time:** 50 minutes.

### Exercise 3 — Effective sample size (calculation, 30 minutes)

For Exercise 2's matrix, compute `n_eff` at each horizon using `rho_h = (h - 1) / h`, and the
standard-error inflation factor at `h = 4`.

**Deliverable:** a four-row table of `rho_h`, `n_eff`, and inflation factor.
**Acceptance criteria:** formula written explicitly; inflation factor computed as a square root;
one sentence on what it does to confidence at `h = 4`.
**Time:** 30 minutes.

### Exercise 4 — Choose the window scheme (design and critique, 45 minutes)

A series has 180 weekly observations. A documented network change at week 120 shifted demand
between two DCs. Propose an expanding and a sliding design, and argue for one.

**Deliverable:** two protocol specifications and a 200-word justification.
**Acceptance criteria:** each design satisfies the span requirement; the post-break sample length
is stated; the justification names what would change your mind.
**Time:** 45 minutes.

### Exercise 5 — Apply the seasonal cycle rule (calculation, 30 minutes)

A monthly series has 60 observations and `m = 12`. You propose `K = 12`, step 3, `H = 3`. Compute the
span in months and cycles, state both verdicts, and revise the design.

**Deliverable:** the arithmetic, two verdicts, and a revised design.
**Acceptance criteria:** span computed as `(K - 1) * step + H`; both thresholds stated; the revision
satisfies at least the one-cycle test with a reason.
**Time:** 30 minutes.

### Exercise 6 — Separate selection from reporting (design, 40 minutes)

You have 40 origins and 15 candidate configurations. Design a protocol that reports an honest
accuracy estimate for the chosen method.

**Deliverable:** a one-page protocol with the origin split, selection rule, and reporting rule.
**Acceptance criteria:** selection and reporting origins disjoint; the reporting block passes the
span test; the number of candidates is disclosed; a tie-breaking rule is specified.
**Time:** 40 minutes.

### Exercise 7 — Backtest an override (design and critique, 50 minutes)

A team records overrides in an audit table with value, user, and timestamp. Design an evaluation of
whether overrides add value.

**Deliverable:** three stairstep series (naive, statistical, statistical plus override) on identical
origins, with the FVA for each step.
**Acceptance criteria:** identical origins and horizons; FVA against both the previous step and the
naive benchmark; one stated limitation about information the human had and the data does not.
**Time:** 50 minutes.

### Exercise 8 — Write the reporting standard (design, 50 minutes)

Write a one-page standard any analyst in a planning team could follow when reporting a backtest.

**Deliverable:** the standard plus one worked example table.
**Acceptance criteria:** requires origin count, step, window scheme, horizon, metric, dispersion,
`n` per horizon, benchmark, vintage, and refit policy; forbids a pooled number without the
per-horizon table; states what the backtest does not cover.
**Time:** 50 minutes.

## 13. Self-check answers

**Exercise 1.** There is no single correct number, and that is the answer. A typical result on a
stable weekly series is 9.8, 12.4, 11.1 and 15.7 units of MAE: a range of 5.9 and a ratio of 1.60.
The spread across cut points is often larger than the gap between two competing methods, so a
single split cannot rank methods whose true difference is under that gap — which is most methods.
The direction is not random: holdouts containing a promotion, a stock-out, or a holiday cluster are
harder, and which falls in a given eight-week holdout is an accident of the calendar.

**Exercise 2.** With `K = 8`, step 4, `H = 4`, the span is `7 * 4 + 4 = 32` weeks and the counts per
horizon are 8, 7, 6 and 5, because the last origins cannot reach the longer horizons within the
available outcomes.

```text
  pooled MAE = (8*m_1 + 7*m_2 + 6*m_3 + 5*m_4) / 26
  weights      0.31    0.27    0.23    0.19      -- not equal, rarely stated
```

Two acceptance points are commonly missed: printing `n` per column, because a reader assuming
`n = 8` everywhere computes a different standard error; and showing the pooled figure as a weighted
mean, without which the reader cannot reconstruct it from the table.

**Exercise 3.**

```text
  h=1  rho 0.000  n_eff = 8*(1.000/1.000) = 8.00  inflation 1.00
  h=2  rho 0.500  n_eff = 8*(0.500/1.500) = 2.67  inflation 1.73
  h=3  rho 0.667  n_eff = 8*(0.333/1.667) = 1.60  inflation 2.24
  h=4  rho 0.750  n_eff = 8*(0.250/1.750) = 1.14  inflation 2.65
```

At `h = 4` the standard error is 2.65 times the independent-sample value. A 3-unit MAE difference
with a per-origin standard deviation of 8 units has a naive standard error of `8/sqrt(5) = 3.58`,
raised to 9.5 — not separable from zero. The honest sentence: "at the four-week horizon, eight
origins carry about one independent observation, so the comparison is a direction, not an estimate."

**Exercise 4.** Both designs satisfy the span test: `19 * 4 + 4 = 80` weeks, above 52 and below 104,
so the claim is "one full cycle," stated explicitly. Sliding is argued from the dated, physically
meaningful network change; a 52-week window reaches back to week 68 at the latest origin, so it has
not fully excluded the old regime either. Expanding is argued from the 60 post-break weeks, which
is just over one cycle — too little to separate a seasonal estimate from a level shift. The
defensible resolution is to run both and report both, treating the disagreement as the finding.
What would change the answer: 104 post-break weeks would make sliding clearly preferable.

**Exercise 5.**

```text
  span = (12 - 1) * 3 + 3 = 36 months
  one cycle  (12):  36 >= 12   PASS
  two cycles (24):  36 >= 24   PASS
```

Span passes, but training fails: with 60 observations, a first origin at month 25 gives 24 months
of training and puts the last outcome at month `25 + 33 + 3 = 61`, one month beyond the data.
Revised: first origin month 24, `K = 11`, step 3, `H = 3`, span 33 months, last outcome month 60,
training 24 months — exactly two cycles, which is the minimum. State it as "two cycles of training,
2.75 cycles of evaluation."

**Exercise 6.** Origins 1–40; selection block 1–26; reporting block 27–40; choose the lowest MAE at
the decision horizon on the selection block; if the top three are within one standard deviation,
combine them equally; report the chosen method's per-horizon table on the reporting block only; and
disclose that 15 configurations were scored. The reporting block must pass the span test
independently — if 14 origins cannot cover a cycle, shorten the selection block, not the reporting
block. A strong answer also reports the selection-block score beside the reporting-block score,
because the gap between them estimates the selection bias on this data.

**Exercise 7.**

```text
  Step                    MAE (h=4)   FVA vs previous   FVA vs naive
  Naive (random walk)         24.0              --             0.0
  Statistical (ETS)           18.3            -5.7            -5.7
  Statistical + override      16.9            -1.4            -7.1
```

The override adds 1.4 units at the four-week horizon; the model adds 5.7 against the naive, so most
of the process value is the model. Repeat at every horizon, because override behaviour is
horizon-dependent: planners adjust short horizons more often and long horizons more heavily. The
limitation: the planner saw a customer call or a competitor stock-out that is not in the dataset,
so the FVA may be measuring the information rather than the judgement. The fix is to capture the
information as a feature, not to remove the planner.

**Exercise 8.** A strong standard has eleven required fields — origin count, step, window scheme
and length, horizons reported separately, metric definition, dispersion with denominator, `n` per
horizon, benchmark and its protocol, refit policy, vintage policy, selection/reporting split — and
three prohibitions: no pooled metric without the per-horizon table, no comparison against a
benchmark computed on different origins, no accuracy claim without the evaluation span stated
relative to the seasonal cycle. The worked example should be a six-row table and fit on one slide;
if it does not, the standard is unusable.

## 14. Interview questions

**1. "How would you evaluate a demand forecasting model?"**
Strong: asks which decision the forecast feeds and at what horizon, designs a rolling-origin run with stated first origin, step, window, and refit policy, reports a per-horizon table with dispersion, and names a benchmark. Weak: "split into train and test and compute MAPE," which is what this topic replaces.

**2. "Why not just use a train/test split?"**
Strong: it is one draw — the score depends on which weeks fell in the holdout, and the spread across cuts often exceeds the gap between methods; defensible only with a large cross-section of independent series. Weak: "less rigorous," with no mechanism.

**3. "Expanding or sliding window?"**
Strong: frames it as bias and variance about stationarity — expanding has lower estimation variance and higher bias after a regime change; sliding adapts but needs post-break data and adds a length to tune; the record favours expanding more than intuition suggests, and running both is the honest move. Weak: picks the default.

**4. "Forty origins, twelve-week horizon. How many independent observations?"**
Strong: fewer than forty, because adjacent origins share innovations; the random-walk correlation is `11/12` and the effective sample size is about three, which widens the standard error accordingly. Weak: says forty.

**5. "Why never average a twelve-week and a one-week forecast error?"**
Strong: error grows with horizon — `sqrt(h)` for a random walk — so the two sit on different scales and the average's value depends on a mix the design sets; adds that horizon `h` gets `K - h + 1` forecasts, so pooling weights short horizons. Weak: "they are different," with no consequence.

**6. "How many origins do you need?"**
Strong: says there is no published rule, then derives from four constraints — minimum forecasts at the longest horizon, seasonal span, effective sample size, compute — landing on 26 to 52 for weekly data with an annual cycle. Weak: gives a number with no derivation.

**7. "Twenty models compared, best scored 12% MAPE. What is wrong with reporting 12%?"**
Strong: it is a maximum over twenty correlated candidates and is optimistically biased; the bias grows with candidates and shrinks with origins; fix by selecting on one block and reporting on another, or with a Reality Check correction, and state the search size. Weak: "overfitting," and stops.

**8. "How would you backtest a process with planner overrides?"**
Strong: distinguishes replaying recorded overrides from reconstructing them afterwards, says only replay is valid, describes the FVA stairstep against a naive benchmark, and names the limit that the planner saw information the dataset does not hold. Weak: "compare with and without the override," never asking when the override was recorded.

**9. "What is the minimum length of a backtest?"**
Strong: derives it — one full seasonal cycle of outcomes, two is better, plus enough history to estimate the model, which for a seasonal model is two cycles; `2m + 2m + H`, or 212 weeks for weekly data. Weak: "as long as possible."

**10. "The pooled metric favours the benchmark but your model wins at one week. What do you report?"**
Strong: both, separated by horizon, and names which horizon the decision consumes, because that decides the recommendation; notes the pooled number is a weighted average with design weights. Weak: reports one of the two without stating the choice.

**11. "What does a backtest not tell you?"**
Strong: nothing about a future regime change, a promotion absent from the history, a channel that did not exist, horizons beyond those evaluated, or the human step if excluded; dispersion covers method uncertainty in a world like the evaluated one, not world uncertainty. Weak: "it does not guarantee future performance," which is true, vague, and unactionable.

**12. "A vendor claims 15% MAPE. What do you ask?"**
Strong: origin scheme, number of origins, horizon, window, refit policy, vintage, benchmark, item and location mix, and whether 15% is pooled or per-horizon. Weak: asks only the sample size.


## 15. Key terms

- **Forecast origin** — the moment a forecast is made; later information is excluded.
- **Forecast horizon** — periods from origin to target; distinct from decision lead time.
- **Decision lead time** — interval between using a forecast and the outcome it concerns.
- **Fold** — one origin with its training data and holdout.
- **Cut point** — the boundary between training data and holdout in one fold.
- **Training window** — the data used to fit the model in one fold.
- **Expanding window** — fixed start, moving end; lower variance, higher bias under a break.
- **Sliding window** — fixed length, both ends moving; adapts after a break, noisier.
- **Step size** — distance between consecutive origins.
- **Holdout** — the periods scored in one fold.
- **Rolling-origin evaluation** — re-running a procedure at advancing origins.
- **Fixed-origin evaluation** — one origin with a multi-period holdout; a single split.
- **Time series cross-validation (TSCV)** — rolling origins with one-period holdouts.
- **Error matrix** — errors indexed by origin and horizon; the object a backtest produces.
- **h-step error** — actual minus forecast for the period `h` ahead of an origin.
- **Pooled error** — a measure over all origins and horizons, with design-dependent weights.
- **MAE / RMSE / MAPE / sMAPE / MASE** — standard error measures, each with a failure mode.
- **Bias** — the mean signed error; a systematic component accuracy measures hide.
- **Seasonal cycle** — `m`, the period after which the pattern repeats.
- **Regime change** — a persistent shift in the data-generating process.
- **Leakage** — any path by which future information reaches the model.
- **Selection bias** — optimism from choosing the best of several candidates on one sample.
- **Reality Check** — a bootstrap test correcting for the search when reporting the best model.
- **Effective sample size** — independent observations equivalent to a correlated sample.
- **Origin correlation** — correlation between errors from adjacent origins at a fixed horizon.
- **Vintage** — a dataset as it existed at a stated date.
- **Refit policy** — whether parameters are re-estimated at each origin.
- **Forecast value added (FVA)** — the metric change attributable to one process step.
- **Override** — a manual adjustment to a statistical forecast.

## 16. Sources

**Peer-reviewed and academic**

- Tashman, L. J. (2000). "Out-of-sample tests of forecasting accuracy: an analysis and review." *International Journal of Forecasting*, 16(4), 437-450. DOI: 10.1016/S0169-2070(00)00065-0. Rolling-origin design and counting rules. Grade A.
- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*. Selection bias when comparing many forecasters on one sample. Grade A.
- Sullivan, R., Timmermann, A. and White, H. (1999). "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap." *Journal of Finance*. Grade A.
- Pesaran, M. H. and Timmermann, A. (2007). "Selection of Estimation Window in the Presence of Breaks." Optimal window depends on break size, post-break sample, model, and loss. Grade A.
- Jolliffe, I. T. (2007). "Uncertainty and Inference for Verification Measures." *Weather and Forecasting*, 22, 637-650. DOI: 10.1175/WAF989.1. A score without a standard error is of little use. Grade A.
- Bradley, A. A., Schwartz, S. S. and Hashino, T. (2008). "Sampling Uncertainty and Confidence Intervals for the Brier Score and Brier Skill Score." *Weather and Forecasting*. DOI: 10.1175/2007WAF2007049.1. Effective sample size under serial dependence. Grade A.
- Ferro, C. A. T. (2007). "Comparing Probabilistic Forecasting Systems with the Brier Score." *Weather and Forecasting*. DOI: 10.1175/WAF1034.1. Standard errors for score comparisons. Grade A.

**Competition material**

- M4 evaluation code and reproduction notebook stating the fixed-origin holdout design, the frequency-dependent horizons, and the sMAPE/MASE/OWA metrics. Consulted 2026. Grade A for the protocol description; Grade B for the notebook.
- Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020). "The M4 Competition." *International Journal of Forecasting*. Results and the OWA metric. Grade A.

**Books and textbooks**

- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition, section 5.10 on time series cross-validation. Grade B.
- Svetunkov, I. *Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)*, section 2.4 on rolling origin and the three window variants. Grade B.

**Official and regulatory**

- Board of Governors of the Federal Reserve System (2011). *Supervisory Guidance on Model Risk Management*, SR Letter 11-7. Grade B.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act), Article 15 on accuracy and robustness. Grade B.
- ISO 3534-1, *Statistics - Vocabulary and symbols - Part 1*. Grade B.
- ISO 5725-1:1994, *Accuracy (trueness and precision) of measurement methods and results*. Grade B.
- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time - Representations for information interchange*. Grade B.
- JCGM 100:2008, *Evaluation of measurement data - Guide to the expression of uncertainty in measurement*. Grade B.

**Preprints and working papers**

- Hannikainen, J. (2015). "Selection of an estimation window in the presence of data revisions and recent structural breaks." MPRA working paper. Grade B.
- Pohle, J., Zahn, M. and Lerch, S. "Uncertainty Quantification in Forecast Comparisons." Working paper. Grade B.

**Practitioner and vendor material**

- Gilliland, M. "Forecast Value Added Analysis: Step-by-Step." SAS white paper. Grade D.
- Gilliland, M. (2013). "Forecast Value Added: A Reality Check on Forecasting Practices." *Foresight*, Issue 29. Grade C.
- Bailey, D. H. and Lopez de Prado, M. (2021). "How 'Backtest Overfitting' in Finance Leads to False Discoveries." *Significance*, 18(6). Grade C.
- Microsoft Azure Machine Learning documentation on forecasting model evaluation. Grade D.

> **Note:** Findings are reported as the cited works state them. The M4 protocol claim was checked against the competition's published evaluation code and a reproduction notebook, because "the largest forecasting competition used a single split" is surprising enough to warrant checking. No citation in this file is invented.


## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Benchmarks this topic scores against: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- The models being backtested: [`02_exponential-smoothing.md`](02_exponential-smoothing.md) and [`03_arima-sarimax.md`](03_arima-sarimax.md)
- The leakage this topic audits: [`04_regression-lags-fourier.md`](04_regression-lags-fourier.md)
- The level at which to evaluate: [`05_hierarchy-aggregation.md`](05_hierarchy-aggregation.md)
- Where short backtests mislead most: [`06_intermittent-demand.md`](06_intermittent-demand.md)
- Previous topic, on combining rather than selecting: [`07_forecast-combination.md`](07_forecast-combination.md)
- Next topic, on the metric and its cost interpretation: [`09_error-metrics-cost.md`](09_error-metrics-cost.md)
- Later, on measuring planner overrides directly: [`10_overrides-fva.md`](10_overrides-fva.md)
- The uncertainty machinery behind the dispersion column: [`../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md`](../../01_foundations/04_statistics_causal_inference/02_probability-bootstrap.md)
- Autocorrelated errors and robust standard errors: [`../../01_foundations/04_statistics_causal_inference/05_autocorrelation-robust-errors.md`](../../01_foundations/04_statistics_causal_inference/05_autocorrelation-robust-errors.md)
- Time-series regression foundations: [`../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Calendars, week numbering, and time zones, which decide where a fold boundary falls: [`../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../../01_foundations/01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Error metric definitions and denominators: [`../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- Module survey file: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Return to the part map: [`../_index.md`](../_index.md)
