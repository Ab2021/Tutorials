---
collection: time-series-playbook
file_id: '04'
topic: Rolling-origin validation, leakage, and demand accuracy
level: 3
status: draft
version: '1.0'
created: '2026-11-07'
reviewed: null
review_trigger: Changes to decision cadence, data availability, labels, or source definitions
notation_style: plain-text
prerequisites: Basic averages, absolute errors, timestamps, and supervised learning
pattern_tags: [forecast-validation, point-in-time, demand]
source_status: OPEN
owner_role: forecasting curriculum maintainer
source_index: INDEX.md
---
# Rolling-Origin Validation, Time Leakage, and Demand Forecast Accuracy
## 1. File Header
**Purpose:** Design an honest historical replay
of a demand forecasting decision, calculate
decision-relevant accuracy, and diagnose
deceptively good results.
**Audience:** Analysts, forecasting engineers,
inventory planners, and interview candidates who
can calculate means and interpret timestamps.
**Outcome:** Given a forecast cadence, horizon,
data publication times, and observed outcomes,
construct legal folds, compute comparable
errors, identify leakage, and explain
operational consequences.
**Scope:** Daily and weekly demand forecasts for
replenishment; the same reasoning applies to
other scheduled forecasts when their decision
and observation clocks are explicit.
**Exclusions:** This is not a full ARIMA fitting
tutorial, a causal promotion-effect estimator,
or a complete inventory optimization model. The
parent playbook lists these adjacent topics as
planned.
**Evidence:** All datasets, schedules, costs,
and outcomes below are ILLUSTRATIVE, not
observations or measured deployments. DERIVED
indicates a consequence of the stated arithmetic
or invariant. OPEN indicates a question
requiring actual operational evidence. Draft has
not received independent review.
**Retrieval cue:** Start at Section 4 for
formulas, Section 5 for a complete replay, and
Section 8 for suspicious results.
**Takeaway:** Accuracy is meaningful only for
forecasts that could actually have been issued
at the relevant decision time.
## 2. Concept Map / Mental Model
```text
Replenishment decision and lead time
    -> origin, horizon, freeze time, unit of decision
    -> historical information available at each origin
    -> train and transform on legal history only
    -> issue forecast and freeze the prediction record
    -> wait for mature actuals and choose actuals vintage
    -> compare matched baseline and candidate by horizon
    -> quantify uncertainty, cost, slices, and guardrails
    -> decide whether to run a controlled operational trial
```
A forecast origin is a snapshot of what could be
known before a decision; it is not necessarily
midnight or the end of the most recent observed
day.
The target time is the period whose demand is
predicted.
The horizon is the distance from origin to
target, measured in the time unit specified by
the forecast contract.
A forecast issued on Monday for Thursday and a
forecast issued on Wednesday for Thursday are
distinct predictions of the same outcome.
They must not be silently merged because their
information sets and remaining lead times
differ.
A rolling-origin evaluation reconstructs many
such historical snapshots rather than performing
a single random train/test split.
It approximates a historical operational
process; it cannot by itself establish future
performance under a different process.
The evaluation must simulate the same update
cadence, data delays, feature vintages, horizon,
and action constraints as the intended use.
An honest replay can still be misleading if it
scores sales where the decision requires
unconstrained customer demand.
It can also be misleading if a pooled score
hides failures in a small but strategically
important slice.
The decision owner defines the trade-off; the
forecaster translates it into a metric and
guardrails.
**Takeaway:** Treat each origin as a historical
decision, not as just another row in a
machine-learning table.
## 3. Foundational Definitions
A **series** is an entity at the decision grain,
such as a SKU-store pair. SKU means stock
keeping unit, an item identifier.
A **period** is an agreed non-overlapping
interval, such as a local calendar day or a
business week. Boundaries, timezone, and
daylight-saving transitions belong in the
contract.
A **target** is a specified quantity, such as
fulfilled units, orders, or latent unconstrained
demand. The three are not interchangeable.
A **forecast origin** is the time at which the
model's information set closes. A decision may
happen later, but the simulation must use the
actual feature freeze.
A **horizon** is a positive integer number of
periods after the origin in this sheet. For
intraday systems, define exact elapsed-time
boundaries instead.
A **fold** is one origin plus its training
history and its selected future target periods.
An **expanding window** keeps the historical
start fixed while the origin advances. A
**sliding window** advances the start as well as
the end.
A **refit** re-estimates model parameters. A
**forecast refresh** creates new predictions,
potentially using the same fitted parameters.
A **gap** removes observations from a region
near the evaluation boundary to model delay or
prevent overlap. It is not a substitute for
checking every data availability timestamp.
A **vintage** is the value of a revisable data
item as it was available at a particular time.
Revised history is not necessarily the
historical information set.
Event time records when something occurred;
availability time records when the forecasting
system could use it. Both matter.
A **label maturity time** is when a target or
corrected target is reliable enough under the
evaluation policy. Reporting early results is
possible but must label them provisional.
A **baseline** is a simple, frozen competitor
given the same legal information, horizon,
target, and scoring population.
**Seasonal naive** repeats the most recent
available observation from the corresponding
seasonal position. With a long forecast horizon,
the rule must explicitly repeat the seasonal
cycle rather than use future actuals.
**Bias** in this sheet means mean forecast minus
actual, so positive values indicate
overforecasting. Other teams use the opposite
sign; document it.
**MAE** is mean absolute error, measured in
target units. **RMSE** is the square root of
mean squared error, also in target units.
**WAPE** is weighted absolute percentage error:
aggregate absolute error divided by aggregate
actual demand, multiplied by 100 for a percent.
**MAPE** averages individual absolute percentage
errors. It is undefined for zero actuals and
unstable near zero.
**MASE** divides each error by a training-only
in-sample naive error scale for that series and
fold, before averaging.
**Pinball loss** evaluates one forecast quantile
and gives an asymmetric penalty for under- and
over-prediction.
A **stockout** can censor demand: sales cannot
reveal all demand that would have arrived with
unlimited availability.
A **hierarchy** maps detailed series to totals,
for example SKU to category and store to region.
Coherent forecasts sum consistently across the
declared hierarchy.
An **evaluation population** is the explicit set
of series, origins, horizons, and target periods
eligible for the reported metric.
**Takeaway:** Define the target, clocks, grain,
and sign conventions before choosing a metric.
## 4. Core Formulas with Full Derivation
### 4.1 Base: error and mean absolute error
Validity: Every evaluated prediction and actual
refer to the same series and target period,
measured in the same unit. Missing outcomes are
not zero-filled without evidence.
DERIVED: Subtracting the actual from a
prediction gives signed error. Taking absolute
values discards direction; averaging over the
chosen population yields an average magnitude.
```text
FORMULA [E1]: e_j = f_j - y_j; MAE = sum_j |e_j| / n
WHERE: j indexes n forecast-target pairs; f_j forecast [units];
       y_j actual [units]; e_j [units]; n positive integer.
CONSTRAINT: matching grain, units and target definition; n > 0.
NUMERICAL EXAMPLE: actual=(10,20), forecast=(12,17);
                   errors=(2,-3), MAE=(2+3)/2=2.5 units.
INTERPRETATION: average absolute miss is 2.5 units per pair.
```
Boundary: A two-unit error for an item with
typical demand of two is different from the same
error for an item with typical demand of two
thousand. MAE alone does not make those business
impacts comparable.
The mean signed error is separately useful for
diagnosing directional over- or
under-forecasting; opposite signed misses cancel
there but not in MAE.
### 4.2 General: rolling-origin estimate
Validity: Origins and target horizons match the
intended operational decision, and all training
and feature values are legal as of each origin.
The following describes an empirical average,
not an unbiased guarantee for future demand.
DERIVED: Each origin produces a vector of
forecast errors. Assign a nonnegative weight to
each eligible series-origin-horizon cell and
divide the weighted loss sum by the sum of
weights.
```text
FORMULA [E2]: L = sum_(i,o,h) w_(i,o,h) * loss(y_(i,o+h), f_(i,o,h))
                  / sum_(i,o,h) w_(i,o,h)
WHERE: i series; o origin; h positive horizon; y actual [units];
       f forecast [units]; w nonnegative weight; loss has declared units.
CONSTRAINT: positive total weight; matched cells and as-of inputs;
            fixed eligibility, loss, and weighting policy.
NUMERICAL EXAMPLE: two eligible cells have absolute errors 2 and 6,
                   weights 3 and 1; L=(3*2+1*6)/4=3 units.
INTERPRETATION: weighting changes which decisions dominate the result.
```
Boundary: Setting weights after viewing errors
can favor a preferred model. A weight derived
from future realized demand can be a legitimate
descriptive scoring weight if declared, but
cannot be used as an as-of model feature or a
prospective decision weight without
justification.
Choose equal origin weights when each decision
cycle is equally important; choose explicit
business weights when those reflect the
objective. A series with more eligible dates
otherwise contributes more cells.
Overlapping horizons cause some target dates to
appear multiple times. This is correct for
evaluating every issued decision but not
equivalent to a report with one forecast per
realized date.
### 4.3 Variants: WAPE, RMSE, MASE, and quantile loss
Validity for WAPE: Nonnegative actual demand and
strictly positive total actual demand on the
scored population. Forecast and actual units
must match.
DERIVED: Sum the magnitudes of all unit errors
and normalize by the sum of observed demand.
This equals a demand-weighted average of
individual absolute percentage errors only for
positive individual actuals; the aggregate
expression still includes errors when some
actuals are zero.
```text
FORMULA [E3]: WAPE = 100 * sum_j |f_j-y_j| / sum_j y_j
WHERE: j indexes scored forecast-target pairs; y_j >= 0;
       f_j and y_j measured in the same units; result in percent.
CONSTRAINT: sum_j y_j > 0; define treatment of returns and missingness.
NUMERICAL EXAMPLE: actual=(10,20), forecast=(12,17);
                   WAPE=100*(2+3)/(10+20)=16.67%.
INTERPRETATION: absolute error totals one sixth of actual volume.
```
Boundary: If total actual demand is zero, report
undefined WAPE and the absolute error in units,
not a fabricated zero or infinity. WAPE can
improve merely because total demand grows while
absolute error does not.
Validity for RMSE: finite squared errors and a
meaningful reason to give larger errors greater
influence.
DERIVED: Squaring error makes positive and
negative misses contribute positively and
emphasizes large misses; the square root returns
the result to the original unit.
```text
FORMULA [E4]: RMSE = sqrt(sum_j (f_j-y_j)^2 / n)
WHERE: n > 0 matched pairs; forecast and actual in target units.
CONSTRAINT: squared errors are finite; population is fixed.
NUMERICAL EXAMPLE: errors=(2,-3); RMSE=sqrt((4+9)/2)
                   =sqrt(6.5), approximately 2.55 units.
INTERPRETATION: a large miss affects RMSE more than MAE.
```
Boundary: RMSE can prioritize an exceptional
outlier over many smaller operational misses;
investigate the outlier and the business loss
rather than automatically discarding it.
Validity for MASE: A nonzero seasonal-naive
in-sample scale computed only from legally
available training observations, with a defined
seasonal period and sufficient training history.
DERIVED: Divide the out-of-sample absolute
forecast error by the average in-sample absolute
difference from the seasonal lag. Averaging
these dimensionless ratios yields the scaled
error.
```text
FORMULA [E5]: scale_(i,o) = sum_(t=m+1 to T) |y_(i,t)-y_(i,t-m)| / (T-m)
              MASE = mean_(i,o,h) (|f_(i,o,h)-y_(i,o+h)| / scale_(i,o))
WHERE: m seasonal period [periods]; T training count at origin o;
       scale [units]; MASE dimensionless; T > m.
CONSTRAINT: scale > 0; training data available by origin; declared averaging.
NUMERICAL EXAMPLE: m=1, training=(2,4,3,6), scale=(2+1+3)/3=2;
                   future actual=5, forecast=8; MASE=3/2=1.5.
INTERPRETATION: this error is 1.5 times the training naive scale.
```
Boundary: A constant or all-zero training series
has zero scale. Mark its scaled score undefined
and report another metric; never add an
undocumented epsilon. A MASE below one compares
with an in-sample naive scale, not necessarily a
same-horizon out-of-sample baseline.
Validity for quantile loss: The target quantile
level is fixed strictly between zero and one and
the outcome is numeric.
DERIVED: The asymmetric absolute loss charges
underprediction at the quantile level and
overprediction at one minus that level. Its
population minimizer is a corresponding
conditional quantile under standard
integrability conditions.
```text
FORMULA [E6]: pinball_q(y,f) = q*(y-f) if y >= f,
                                (1-q)*(f-y) otherwise
WHERE: q in (0,1) is dimensionless; y actual and f forecast [units].
CONSTRAINT: evaluate a declared quantile on matched target pairs.
NUMERICAL EXAMPLE: q=0.8, y=10, f=7 -> loss=0.8*3=2.4 units;
                   y=7, f=10 -> loss=0.2*3=0.6 units.
INTERPRETATION: this quantile penalizes a shortfall more heavily.
```
Boundary: The 0.8 quantile need not equal the
mean or median. Its usefulness depends on the
actual shortage and surplus trade-off, not on a
generic claim that higher is safer.
### 4.4 Limits and counterexample: causal feature availability
A chronological split is necessary for this
replay but not sufficient. For each historical
training row, the feature value must be the
value known at that row's simulated origin, not
a later corrected value merely associated with
an earlier event date.
Consider an actual weather reading recorded on
Friday. An extract may label it with Friday's
date, but a Tuesday forecast for Friday could
only use a weather forecast published by
Tuesday's freeze time.
Likewise, a promotion planned on Tuesday but
amended on Thursday must be represented by
Tuesday's plan in Tuesday's forecast. The
realized Thursday promotion is a legitimate
target-time description, not a legal Tuesday
feature.
A global scaler fitted on all dates passes a
superficial chronological row split while
leaking later distribution information into
earlier training and predictions.
An embargo of one day does not fix a weekly
batch that arrives two days late. Derive
availability from the source's publication and
ingestion rules.
Preconditions: an immutable origin, an
available-at timestamp for each feature version,
and a deterministic tie-breaking rule.
State transition: filter candidate versions to
those published and ingested by the cutoff;
select the latest legal version for the entity
and target period; then record its lineage with
the prediction.
Invariant: no selected record has an
availability time after the origin's feature
freeze. Justification: the filter excludes every
later record before selection.
Trace: a plan for Friday published Monday is
selected at Tuesday's origin; a revision
published Wednesday is not; a correction
ingested after the Tuesday freeze is not
selected even if its source timestamp says
Monday.
Recovery: if no legal version exists, use the
declared missing-value handling or a safe
baseline; do not backfill from a later vintage.
Failure boundary: a source that overwrites
history without publication or ingestion
timestamps cannot support a faithful as-of
replay until historical vintages are recovered
or the missingness is explicitly acknowledged.
**Takeaway:** Derive each score from a declared
population and enforce the information boundary
on every historical feature and transformation.
## 5. Worked Numerical Example
### 5.1 Decision and data contract
ILLUSTRATIVE: A planner orders one SKU for one
store at the end of each day, for delivery
covering the next two days.
The forecast origin occurs after that day's
finalized sales file arrives. There is no
ingestion delay in this deliberately small
example.
The target is observed fulfilled units, not
inferred unconstrained demand. No stockout
adjustment is assumed.
Two origins, day 4 and day 5, are replayed. Both
models use an expanding training window starting
at day 1.
Daily actuals for days 1 through 7 are
respectively 8, 10, 12, 10, 14, 13, and 15
units.
At day 4, training ends with day 4; day 5 and
day 6 are held out. At day 5, day 5 is now
known; day 6 and day 7 are held out.
The same realized day 6 therefore appears at
horizon 2 for origin 4 and horizon 1 for origin
5. These are two different operational
forecasts.
The baseline forecasts the last observed actual
at both horizons. The candidate's predictions
are supplied below as hypothetical values; no
fitting result is claimed.
```text
Origin  Horizon  Target  Actual  Baseline  Candidate
4       1        5       14      10        12
4       2        6       13      10        12
5       1        6       13      14        13
5       2        7       15      14        14
```
The baseline uses only day 4 at the first origin
and day 5 at the second. The candidate must also
be generated using only data available at the
respective origin.
The actuals are shown together only for
explanatory scoring after maturation. Their
placement in this table does not make them legal
model inputs.
### 5.2 Error table and arithmetic
```text
Origin  Horizon  Baseline error  Candidate error  Baseline AE  Candidate AE
4       1        -4              -2               4            2
4       2        -3              -1               3            1
5       1        +1               0               1            0
5       2        -1              -1               1            1
Total            -7              -4               9            4
```
Here AE means absolute error in units. Error is
forecast minus actual, following Section 3.
Using E1, baseline MAE is 9/4 = 2.25 units per
forecast-target pair. Candidate MAE is 4/4 =
1.00 units.
The candidate's absolute improvement is 1.25
units per pair. Its relative reduction against
this baseline is 1.25/2.25, approximately
55.56%.
The denominator in that relative comparison is
the baseline MAE, not the candidate MAE. A zero
baseline MAE would make a relative reduction
undefined.
Using E3, the repeated-decision actual total is
14+13+13+15 = 55 units. Baseline WAPE is 9/55
times 100, approximately 16.36%.
Candidate WAPE is 4/55 times 100, approximately
7.27%. Both use the same repeated-decision
population.
The WAPE difference is approximately 9.09
percentage points, not 9.09 percent relative
improvement. Relative reduction is 5/9,
approximately 55.56%.
Using E4, baseline squared-error sum is 16+9+1+1
= 27, giving RMSE sqrt(27/4), approximately 2.60
units.
Candidate squared-error sum is 4+1+0+1 = 6,
giving RMSE sqrt(6/4), approximately 1.22 units.
Mean signed errors are -7/4 = -1.75 and -4/4 =
-1 unit. Both models underforecast on average
under the chosen sign convention.
For horizon 1, baseline MAE is (4+1)/2 = 2.5 and
candidate MAE is (2+0)/2 = 1.0 units.
For horizon 2, baseline MAE is (3+1)/2 = 2.0 and
candidate MAE is (1+1)/2 = 1.0 units.
The candidate is better in both displayed
horizon slices, but two origins do not establish
a reliable population improvement.
Independent arithmetic check: forecast totals
are 48 for baseline and 51 for candidate,
compared with actual total 55; their summed
signed errors are therefore -7 and -4.
The absolute error totals reconstructed from the
row errors are 9 and 4. Multiplying the
displayed WAPE values by 55/100 recovers those
totals up to rounding.
### 5.3 Lead-time aggregation changes the question
Suppose one decision requires the sum of the two
daily predictions, rather than treating each day
as a separate decision. For origin 4, actual
lead-time total is 14+13 = 27 units.
The baseline total is 10+10 = 20 and the
candidate total is 12+12 = 24; absolute errors
are 7 and 3 units.
For origin 5, actual total is 13+15 = 28;
baseline total is 14+14 = 28 and candidate total
is 13+14 = 27.
The respective absolute errors are 0 and 1.
Lead-time-total MAEs are therefore (7+0)/2 = 3.5
and (3+1)/2 = 2 units per order decision.
Aggregation allows errors in opposite directions
to cancel, as at origin 5 for the baseline.
Daily MAE and lead-time-total MAE answer
different questions and neither replaces the
other.
A forecast for cumulative demand should ideally
be evaluated against actual cumulative demand at
the exact replenishment lead time and decision
cadence.
### 5.4 Sensitivity and decision boundary
At origin 5, replace the day 7 actual of 15 by
20 units, leaving all forecasts unchanged. The
baseline's horizon-2 error becomes -6 and the
candidate's becomes -6.
Their total absolute errors become 14 and 9,
respectively. Candidate MAE becomes 2.25 and
baseline MAE becomes 3.5 units.
Candidate relative MAE reduction falls to
(3.5-2.25)/3.5, approximately 35.71%. A single
large target substantially changes the apparent
relative advantage in this tiny sample.
If candidate day 7 forecast instead equals 8
while the original actual remains 15, candidate
absolute-error total becomes 2+1+0+7 = 10, worse
than baseline's 9.
This demonstrates why model selection based on a
few origins is fragile. A decision maker needs
many relevant historical periods, uncertainty
intervals, and future operational monitoring.
A sales-based result cannot establish
improvement in latent demand or service level
when inventory availability constrains sales.
The latter requires a separate measurement
strategy.
**Takeaway:** Reconstruct the exact forecast
cells, check the arithmetic, and score both
daily and decision-level quantities where each
has a clear use.
## 6. Visual / Geometric Intuition
```text
Time:        1   2   3   4   5   6   7
Origin 4:   [------train------]  H1  H2
Origin 5:   [--------train--------]  H1  H2
Target 6:                        H2  H1
```
The diagram places day 6 in two forecasts, one
at each origin. It is not double counting if
both forecasts were actual decision
opportunities.
For a one-forecast-per-target report, define
which origin is retained before computing
metrics, such as the forecast issued exactly two
days ahead.
Visualize a second axis for availability time. A
historical row can sit to the left of the split
by event time while arriving after the origin on
the availability axis.
An as-of join moves along the availability axis
and chooses a prior version, rather than simply
selecting the latest value in the modern
warehouse.
A prediction interval can be drawn as a band
around a central forecast. Counting target
points within the band measures empirical
coverage, while band width expresses sharpness.
A very wide band may achieve high coverage
without being useful; a very narrow band may be
useful only if it still achieves adequate
coverage.
In a supply chain, actual sales can lie below
true demand because inventory was exhausted. A
low residual against sales may conceal unmet
demand.
**Takeaway:** A calendar diagram shows
chronological order, but an availability-time
diagram is needed to expose leakage.
## 7. Comparison / Selection Table
| Choice | Prefer when | Main risk | Disqualifier or fallback |
|---|---|---|---|
| Expanding window | Older history remains informative and storage is available | Old regimes can dominate | Use sliding or weighted history if validated drift makes old data harmful |
| Sliding window | Recent regime is more relevant and enough examples remain | Seasonal coverage and rare events may disappear | Expand window or pool related series when sparse |
| Fixed holdout | A final untouched period is needed | One period can be unrepresentative | Report its dates and complement with earlier rolling folds |
| Rolling origins | Decisions recur and multiple regimes are relevant | Expensive replay; dependent outcomes | Predeclare cadence and use cluster-aware uncertainty |
| Random split | Independent, exchangeable observations are justified | Future information enters training for time series | Avoid for operational forecasting unless the independence claim is defensible |
| MAE | Equal cost per unit of miss is a useful approximation | Masks scale and asymmetric cost | Add slices and cost-based metrics |
| RMSE | Large errors deserve disproportionately greater penalty | Sensitive to outliers and high-volume items | Show residual distribution and MAE |
| WAPE | Aggregate unit error relative to total actual is useful | Undefined at zero total; high-volume series dominate | Add zero-volume absolute error and macro slices |
| MASE | Scale-free comparison across series is useful | Zero scale or unstable short histories | Mark undefined; use matched out-of-sample baseline |
| Pinball | Decision depends on a stated quantile | Quantile level may not match economics | Validate costs and inspect calibration by slice |
| Sales target | Fulfilled volume itself is the decision target | Stockout censoring | Separate available periods or model lost demand explicitly |
| Latent demand | Procurement needs all customer demand | Unobserved target and identifying assumptions | Use availability data and bounds; report OPEN uncertainty |
| Daily score | Daily allocation matters | Does not measure summed lead-time exposure | Also score lead-time aggregate |
| Lead-time score | Orders cover a fixed lead-time window | Daily timing errors can cancel | Keep daily service and capacity guardrails |
The table is a selection aid, not a universal
ranking. Use a matched seasonal-naive or
last-value comparator even when the candidate
has sophisticated features.
If a proposed choice conflicts with an actual
decision contract, the decision contract wins;
update the evaluation rather than assuming a
standard split applies.
**Takeaway:** Select the window, metric, and
aggregation for the operational question, not
for convenience or a single leaderboard.
## 8. Failure Modes & Edge Cases
All cases in this section are ILLUSTRATIVE
diagnostic scenarios, not documented incidents.
Each requires evidence before attributing a root
cause.
### F1. Implausibly accurate future forecasts
Symptom: Validation error is much lower than a
production-like shadow period or a legal
baseline.
Competing causes: future feature leakage,
accidental target join, real predictive signal,
or a population mismatch.
Diagnostic: For each selected record compare
publication and ingestion time with the
historical feature freeze; inspect join keys and
target-date offsets.
Expected signal for leakage: at least one used
feature version was unavailable at its forecast
origin, or an exact future target entered a
feature.
Bounded fix: reconstruct as-of snapshots and
refit all fold-local transformations; do not
merely drop the suspicious validation rows.
Verification: replay the same origins with
audited feature lineage and compare against a
matched baseline; document any remaining missing
vintages.
Risk: replay may become impossible if sources
have overwritten revisions. Escalate rather than
silently substitute today's version.
### F2. Improvement disappears after a late data feed
Symptom: An offline model beats baseline but
operational features are frequently missing at
decision time.
Competing causes: late ingestion, wrong
timezone, upstream outage, or training-serving
transformation mismatch.
Diagnostic: Compare event, source-publication,
ingestion, feature-materialization, and decision
timestamps for a sample of missed runs.
Expected signal for late ingestion: the source
published in time but materialization completed
after the feature freeze.
Bounded fix: model historical availability
honestly, adjust the feed SLA or use a
documented fallback; do not shift the decision
cutoff without planner approval.
Verification: evaluate coverage and accuracy
under the actual fallback path over comparable
origins.
Risk: a later cutoff may miss ordering deadlines
even if it improves model accuracy.
### F3. WAPE looks excellent while low-volume items fail
Symptom: Aggregate WAPE improves but planners
report persistent misses on low-volume or
critical items.
Competing causes: volume weighting, a genuine
shift in item mix, rounding, or different
eligibility rules.
Diagnostic: Recompute shared-population unit
errors, equal-series summaries, zero-demand
errors, and critical-item slices alongside the
aggregate.
Expected signal for weighting: a few high-volume
series dominate the denominator while many small
series have large per-series misses.
Bounded fix: retain WAPE for volume exposure but
add an equal-series metric and operational
service guardrails with minimum slice sizes.
Verification: publish denominators, coverage,
and paired comparisons by slice; make sure no
undefined zero-total WAPE is displayed as zero.
Risk: macro averages can overemphasize noisy
low-volume series. Choose weights transparently
with the decision owner.
### F4. Zero or negative demand breaks percentages
Symptom: MAPE becomes infinite or WAPE is
missing or changes sign.
Competing causes: zero-demand days, returns
booked as negative units, missing values encoded
as zero, or a unit conversion defect.
Diagnostic: Separate zero, negative, null, and
positive target counts; reconcile returns and
sales to the source ledger.
Expected signal for a true zero: the store was
open, inventory was available, and the source
asserts no units sold.
Bounded fix: report absolute errors on
zero-demand slices and define a returns-adjusted
target if that is the decision quantity.
Verification: calculate metrics on a manually
inspected sample and verify denominators are
strictly positive where required.
Risk: excluding zero days biases evaluation
toward high activity. Keep their errors in
aggregate absolute-error totals where the metric
permits.
### F5. Sales are accurate but shelves are empty
Symptom: Forecasts track observed sales while
stockout rates remain high.
Competing causes: censored demand, poor
replenishment execution, incorrect lead times,
or insufficient stock allocation.
Diagnostic: Compare forecasts, on-hand
inventory, availability intervals, lost-order
signals, lead-time records, and order execution.
Expected signal for censoring: sales terminate
when available stock reaches zero while evidence
of demand continues.
Bounded fix: separate uncensored periods;
estimate or bound missing demand only with
explicit assumptions and independent validation.
Verification: evaluate forecast and operational
service on comparable availability regimes; use
a controlled trial where feasible.
Risk: treating a model-imputed lost-demand
estimate as ground truth can reinforce its own
assumptions.
### F6. A sudden metric jump after a data revision
Symptom: Historical scores change after a
restatement even though frozen forecasts did not
change.
Competing causes: revised actuals, changed
calendar, altered product hierarchy, or an
accidental prediction overwrite.
Diagnostic: Compare immutable forecast IDs,
actual vintages, scoring code version, and
mapping snapshots between reports.
Expected signal for revision: identical forecast
records joined to different corrected actual
values.
Bounded fix: publish both first-mature and
latest-restated scores with their evaluation
timestamps; do not rewrite prior reports without
trace.
Verification: rerun the same score on the same
immutable inputs and compare cell-level
differences.
Risk: a restated truth may be more accurate, but
it is not the truth available to the original
monitoring process.
### F7. Scores improve when tuning on the final period
Symptom: Repeated model iterations improve the
last holdout but not later observations.
Competing causes: selection overfitting, a real
improvement, changing data, or an untracked
eligibility change.
Diagnostic: Inspect experiment history, number
of configurations, validation folds, and when
the holdout was first consulted.
Expected signal for selection overfitting: many
decisions were made in response to the supposed
final test results.
Bounded fix: reserve a later untouched period,
or evaluate a new forward shadow period; use
earlier nested rolling folds for selection.
Verification: pre-register the selection rule
and evaluate once on the new protected period.
Risk: waiting for a new holdout delays
decisions; a cautious limited trial may be
possible with explicit governance.
### F8. Duplicate target periods distort confidence
Symptom: Very small uncertainty intervals appear
despite overlapping forecast horizons.
Competing causes: treating dependent cells as
independent, excessive weighting of repeated
targets, or a computation bug.
Diagnostic: Count unique target dates and
origins; map shared targets and shared series to
scored rows.
Expected signal for dependence: many residuals
share a target date, origin shock, or
overlapping lead-time window.
Bounded fix: use a block or cluster resampling
design reflecting the dependence structure and
compare paired model losses.
Verification: report resampling unit, block
length, effective number of blocks, and
sensitivity to plausible alternatives.
Risk: few blocks cannot support precise
intervals; report uncertainty as wide or
unquantified rather than manufacturing
precision.
### F9. Hierarchy appears inconsistent
Symptom: Category forecast differs from the sum
of item forecasts, or a change in item
membership changes historical totals.
Competing causes: independently fitted levels,
different hierarchy vintages, unit conversion,
or late item remapping.
Diagnostic: Reaggregate lower-level predictions
using the hierarchy known at each origin and
compare with reported parent predictions.
Expected signal for incoherence: a nonzero
parent-minus-child sum under identical units,
target dates, and mapping vintage.
Bounded fix: apply a declared reconciliation
procedure or select one authoritative
aggregation level; preserve both raw and
reconciled forecasts.
Verification: assert coherence for every
relevant origin and level, then reevaluate
accuracy and operational constraints.
Risk: forcing coherence may increase error for
some nodes; it is a structural property, not a
guarantee of higher accuracy.
### F10. Backtest excludes hard-to-forecast new products
Symptom: Offline accuracy is strong but live
forecasts fail on launches.
Competing causes: minimum-history filtering,
entity mapping errors, or genuine launch drift.
Diagnostic: Compare eligibility rates and
excluded volumes by age, origin, and product
segment.
Expected signal for selection: live launches
were omitted from historical folds because the
model required history unavailable at launch.
Bounded fix: evaluate an explicit cold-start
policy using only launch-time attributes and a
matched fallback.
Verification: report full-population coverage,
new-item scores, and performance of both model
and fallback together.
Risk: excluding difficult series can make a
candidate look better while reducing the number
of decisions it can serve.
**Takeaway:** Diagnose competing causes with
timestamped lineage and matched populations
before deciding that the model itself is at
fault.
## 9. Complexity & Cost Analysis
Let N be the number of series, K the number of
origins, H the number of horizons, and P the
number of features per series-origin-horizon
cell.
Materializing all feature cells can require
order N*K*H*P values; materializing just
forecasts and actuals requires order N*K*H
values, ignoring metadata and compression.
These are storage-growth descriptions, not
benchmark claims. Actual bytes depend on
encoding, sparse values, lineage, indexes,
retention, and platform overhead.
If training a model on M records with a
specified algorithm costs C(M), refitting at
every origin costs approximately the sum of
C(M_o) across origins, plus feature construction
and scoring.
For an expanding window, M_o generally grows
with origin. Reusing a model where production
would refit changes the policy being evaluated;
it is not merely an optimization.
Conversely, refitting every origin when
production refreshes monthly measures a more
expensive and potentially more informed system
than the one being proposed.
Feature lookup cost depends on the index and
join implementation. No universal database
complexity or runtime is asserted here.
ILLUSTRATIVE sizing: N=10,000 series, K=52
weekly origins, H=14 daily horizons, and P=20
numeric features.
The number of forecast cells is 10,000*52*14 =
7,280,000. The number of feature values is
145,600,000.
At eight bytes per numeric value, raw feature
payload is 1,164,800,000 bytes, approximately
1.085 GiB using 1 GiB = 1,073,741,824 bytes.
This excludes keys, validity flags, data-frame
overhead, indexes, duplicate snapshots,
metadata, compression, and intermediate copies.
It is not a capacity recommendation.
A streaming scorer can aggregate sufficient
error sums without retaining every feature cell
in memory, provided lineage and prediction
records remain auditable elsewhere.
If every origin fits an independent model,
parallel execution can reduce elapsed time but
increases simultaneous CPU, memory, and storage
pressure. Set concurrency from measured resource
limits.
For a global cross-series model, the training
population and entity lifecycle must be
reconstructed at each origin; a series
introduced later cannot be used earlier just
because its current identifier exists.
Partition outputs by origin and version; make
retries idempotent so a repeated scoring task
does not create duplicate forecast cells.
A missing fold should be visible as incomplete
coverage, not interpreted as a zero-error fold.
Compare costs and accuracy using the same
completed population.
Monitor data freshness and scoring latency
separately from forecast loss. Error is
unavailable until labels mature, but missing or
stale input signals are available sooner.
**Takeaway:** Estimate replay cost from cells
and actual refit cadence, then measure platform
overhead before sizing a production workload.
## 10. Key Theorems / Results Reference
**DERIVED — median and absolute loss:** A
conditional median minimizes expected absolute
error when the expectation exists. It need not
minimize squared loss or asymmetric shortage
cost.
**DERIVED — mean and squared loss:** The
conditional mean minimizes expected squared
error when the second moment is finite. The
result follows by expanding the square around
the conditional mean; the cross term vanishes.
**DERIVED — quantile and pinball loss:** For a
chosen level strictly between zero and one, a
conditional quantile minimizes expected pinball
loss under integrability. Discrete distributions
can have multiple minimizing quantiles.
**DERIVED — aggregation and cancellation:** The
absolute error of a sum is at most the sum of
absolute errors by the triangle inequality.
Equality need not hold when daily errors have
opposite signs.
**DERIVED — no chronological guarantee:** A
time-ordered split does not imply feature
availability. A counterexample is a revised
prior-day feature published only after the
forecast origin.
**OPEN — future generalization:** Historical
backtest scores alone do not guarantee
performance after policy, regime, assortment, or
measurement changes. Establish a monitoring and
validation plan for the actual deployment
context.
**SOURCED — training-only preprocessing:** The
scikit-learn documentation describes data
leakage from fitting preprocessing on test data
and recommends separating training and test
transformations:
https://scikit-learn.org/stable/common_pitfalls.html
. The course master records consultation on
2026-11-06; the unpinned stable page may change,
so recheck before publication. This supports the
general preprocessing principle, not any
illustrative result in this sheet.
**OPEN — empirical outcomes:** No production
forecast, service-level improvement, inventory
saving, benchmark, or source-specific deployment
has been measured or asserted here.
**Takeaway:** Mathematical loss-optimality and
past replay accuracy depend on different
assumptions; neither guarantees business impact.
## 11. Interview / Exam Questions
### Q1. What is an origin, and why does it matter?
**Answer:** It is the feature-information cutoff
for a historical decision. It determines which
data and model state were available for the
forecast.
**Reasoning:** The same target date may have
forecasts at multiple origins with different
information. Merging them hides lead-time
effects.
**Follow-up:** If the model runs at noon but
ordering occurs at 5 p.m., use the actual noon
feature freeze unless a later refresh really
occurs.
**Red flag:** Defining the origin after seeing
the target or using event time as a proxy for
publication time.
### Q2. A fold trains through day 10 and predicts days 11–13. What changes at the next daily origin?
**Answer:** If day 11 actuals are available by
the next freeze, the expanding training window
can include day 11 and the targets become days
12–14.
**Reasoning:** The horizon rolls with the
decision time; availability, not just calendar
order, governs training eligibility.
**Follow-up:** If day 11 arrives late, exclude
it or use the same fallback that production
uses.
**Red flag:** Automatically using day 11
regardless of reporting delay.
### Q3. Calculate WAPE for actuals 0, 10, 30 and forecasts 2, 8, 36.
**Answer:** The absolute errors are 2, 2, and 6;
WAPE is 100*10/40 = 25%.
**Reasoning:** E3 includes the error on the
zero-actual cell in the numerator, while the
positive aggregate actual supplies a valid
denominator.
**Follow-up:** If all actuals were zero, WAPE
would be undefined and absolute error should be
reported instead.
**Red flag:** Dropping the zero-demand cell
because individual percentage error is
undefined.
### Q4. Why can a random split be misleading?
**Answer:** It can train on observations later
than the simulated forecast, leak temporal
structure, and create an easier information set
than operations.
**Reasoning:** The valid experiment must
reproduce the intended decision's available data
and update schedule.
**Follow-up:** A random split may be defensible
only for an explicitly different exchangeable
prediction task with a justified information
boundary.
**Red flag:** Claiming that random splitting is
universally invalid for every imaginable dataset
without discussing the task.
### Q5. A candidate improves WAPE from 20% to 15%. How do you describe the change?
**Answer:** It improves by five percentage
points and reduces WAPE by 25% relative to the
20% baseline.
**Reasoning:** The relative change is 5/20,
while the absolute percentage-point change is
20-15.
**Follow-up:** First confirm that both scores
use the same forecast cells, target vintage, and
denominator.
**Red flag:** Calling a five-point difference a
five-percent relative reduction without
qualification.
### Q6. How would you detect a promotion-plan leak?
**Answer:** Check the publication and ingestion
timestamps of the exact plan version used by
each origin and compare it with the feature
freeze.
**Reasoning:** A plan finalized after the
forecast could differ from the plan actually
known at decision time.
**Follow-up:** Replay with archived plan
vintages, or state the missing-vintage
limitation if the source overwrote history.
**Red flag:** Using the realized promotion
merely because it describes the forecast target
date.
### Q7. Why might a lower sales MAE fail to improve availability?
**Answer:** Sales can be censored by stockouts
and availability also depends on replenishment,
allocation, and lead-time execution.
**Reasoning:** A model can predict constrained
sales accurately while missing the latent
customer demand that would have occurred with
stock.
**Follow-up:** Examine availability intervals,
lost-order evidence, lead-time demand, and
service outcomes under a controlled evaluation.
**Red flag:** Declaring a causal service
improvement from offline sales MAE alone.
### Q8. Calculate the MASE scale for training values 2, 4, 3, 6 with seasonal period one. What if the forecast is 8 and actual is 5?
**Answer:** The scale is (2+1+3)/3 = 2 units,
and the absolute error 3 divided by 2 gives MASE
1.5.
**Reasoning:** The denominator uses only
adjacent training observations, not future
actuals or the test fold.
**Follow-up:** Constant training values produce
zero scale, so this MASE is undefined.
**Red flag:** Adding a hidden epsilon and
presenting the result as directly comparable
without disclosing the modification.
### Q9. How do overlapping forecast horizons affect uncertainty?
**Answer:** Errors can share realized outcomes,
origin shocks, and adjacent time effects; they
should not automatically be treated as
independent observations.
**Reasoning:** A larger count of cells does not
necessarily mean the same number of independent
pieces of information.
**Follow-up:** Use an appropriate paired block
or cluster resampling strategy, report the
resampling unit, and examine sensitivity to
block length.
**Red flag:** Computing a narrow independent-row
interval while ignoring shared target dates.
### Q10. Choose a loss for costly shortages and inexpensive surplus.
**Answer:** Consider pinball loss at a quantile
derived from explicit underage and overage
costs, subject to the real inventory model and
constraints.
**Reasoning:** Higher quantiles penalize
underprediction more heavily than
overprediction; the right level depends on the
decision economics.
**Follow-up:** Check calibration, lead-time
aggregation, and operational service in addition
to average pinball loss.
**Red flag:** Selecting the 99th percentile
simply because more inventory feels safer,
without considering spoilage or capital cost.
### Q11. When does an expanding window lose to a sliding one?
**Answer:** When old data are less
representative of the current regime and the
sliding window retains enough signal to estimate
the model reliably.
**Reasoning:** A smaller window can adapt faster
but increases estimation variance and can omit
annual seasonality or rare events.
**Follow-up:** Compare window strategies on the
same historical origins and validate under the
intended refit cadence.
**Red flag:** Presenting either window as
universally superior.
### Q12. What would you do if the final holdout was used repeatedly for tuning?
**Answer:** Treat it as validation rather than
an untouched final test, freeze the selection
procedure, and obtain a later protected period
or forward evaluation.
**Reasoning:** Repeated decisions based on the
same holdout adapt the model to its noise.
**Follow-up:** Report the tuning history,
remaining uncertainty, and operational
constraints if waiting for new data is costly.
**Red flag:** Relabeling the same repeatedly
inspected data as a fresh holdout.
### Q13. Two models have equal daily MAE but different lead-time-total errors. Which should be chosen?
**Answer:** The one better aligned with the
actual replenishment loss may be preferred,
subject to daily service and capacity
guardrails.
**Reasoning:** Daily over- and under-errors can
cancel when summed; an order for a multi-day
window is exposed to the cumulative error.
**Follow-up:** Evaluate the cumulative
distribution or quantiles if shortage risk
depends on uncertainty rather than only the
point total.
**Red flag:** Assuming that equal daily MAE
implies equal inventory impact.
### Q14. What must be stored to reproduce a historical score?
**Answer:** Immutable forecast IDs and values,
model and feature versions, origin and target
timestamps, availability lineage, actual
vintage, eligibility, metric implementation, and
hierarchy mapping.
**Reasoning:** A score is a join between
prediction and outcome under a defined
population and policy; changing any component
can change the result.
**Follow-up:** Record missing data and fallback
decisions, not merely successful predictions.
**Red flag:** Retaining only a dashboard
aggregate and current warehouse tables.
**Takeaway:** Strong answers name the decision,
legal information, denominator, failure
boundary, and evidence needed for the
conclusion.
## 12. Quick Reference Card
| Item | Value |
|---|---|
| Topic | Rolling-origin evaluation and demand forecast accuracy |
| Core formula | MAE = sum of absolute errors / eligible forecast count; WAPE = 100 * total absolute error / total actual demand |
| Key assumption | Match forecast and target grain; reconstruct inputs available at each origin; WAPE requires positive total actual demand |
| Complexity | Depends on number of folds and model refits; consult Section 9 for the stated workload |
| Use when | Forecast decisions recur and their historical inputs and outcomes can be reconstructed |
| Avoid when | Future data vintages leak into predictions or changing eligibility makes comparisons invalid |
| Go deeper | [Collection index](INDEX.md); inventory optimization is outside this sheet |
| Diagnostic | Inspect horizon slices, signed bias, zero-demand periods, late feeds and censored demand |
| Evidence | Illustrative calculations; operational benefit not measured |
