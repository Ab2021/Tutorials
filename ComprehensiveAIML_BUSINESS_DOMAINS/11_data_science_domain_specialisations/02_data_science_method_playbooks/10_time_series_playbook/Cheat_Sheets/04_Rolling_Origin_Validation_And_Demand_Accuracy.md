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
**Purpose:** Design an honest historical replay of a demand forecasting decision, calculate decision-relevant accuracy, and diagnose deceptively good results.
**Audience:** Analysts, forecasting engineers, inventory planners, and interview candidates who can calculate means and interpret timestamps.
**Outcome:** Given a forecast cadence, horizon, data publication times, and observed outcomes, construct legal folds, compute comparable errors, identify leakage, and explain operational consequences.
**Scope:** Daily and weekly demand forecasts for replenishment; the same reasoning applies to other scheduled forecasts when their decision and observation clocks are explicit.
**Exclusions:** This is not a full ARIMA fitting tutorial, a causal promotion-effect estimator, or a complete inventory optimization model.
The parent playbook lists these adjacent topics as planned.
**Evidence:** All datasets, schedules, costs, and outcomes below are ILLUSTRATIVE, not observations or measured deployments.
DERIVED indicates a consequence of the stated arithmetic or invariant.
OPEN indicates a question requiring actual operational evidence.
Draft has not received independent review.
**Retrieval cue:** Start at Section 4 for formulas, Section 5 for a complete replay, and Section 8 for suspicious results.
**Takeaway:** Accuracy is meaningful only for forecasts that could actually have been issued at the relevant decision time.
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
A forecast origin is a snapshot of what could be known before a decision; it is not necessarily midnight or the end of the most recent observed day.
The target time is the period whose demand is predicted.
The horizon is the distance from origin to target, measured in the time unit specified by the forecast contract.
A forecast issued on Monday for Thursday and a forecast issued on Wednesday for Thursday are distinct predictions of the same outcome.
They must not be silently merged because their information sets and remaining lead times differ.
A rolling-origin evaluation reconstructs many such historical snapshots rather than performing a single random train/test split.
It approximates a historical operational process; it cannot by itself establish future performance under a different process.
The evaluation must simulate the same update cadence, data delays, feature vintages, horizon, and action constraints as the intended use.
An honest replay can still be misleading if it scores sales where the decision requires unconstrained customer demand.
It can also be misleading if a pooled score hides failures in a small but strategically important slice.
The decision owner defines the trade-off; the forecaster translates it into a metric and guardrails.
**Takeaway:** Treat each origin as a historical decision, not as just another row in a machine-learning table.
### 2.1 Decision-time contract: questions that change the replay
Each row identifies a different operational choice; resolve it with the owner before selecting a model.
| Decision question | Consequence for the historical experiment |
|---|---|
| Who can place the order? | Define whether the prediction serves a central planner, a store, or an automated policy. |
| When does authority end? | Use the last time that an authorized actor can actually alter the order. |
| When does the feature snapshot close? | Reconstruct information at the real cutoff, even if the order is submitted later. |
| Can the planner refresh the forecast? | Replay every refresh that can change the action, not a later theoretical refresh. |
| Is the order irreversible? | Model cancellation rights and penalties if they change the relevant horizon. |
| What is the review cadence? | Issue origins on actual review days, including holiday shifts. |
| Does the site operate every day? | Specify whether closed days contribute zero demand, missing observations, or another target. |
| What unit is ordered? | Convert forecasts to cases or pallets using the conversion known at the order cutoff. |
| What location receives stock? | Score at that location or explicitly propagate the allocation decision to downstream sites. |
| What is the product identifier? | Freeze the effective product mapping, including replacement and pack-size changes. |
| Is demand for orders or shipments? | Use the process actually driving the decision; shipping delays can separate the two. |
| Can unfilled demand be backordered? | Record when the same demand is counted and how eventual fulfillment is attributed. |
| Are returns part of demand? | Define gross demand and returns separately if net units can be negative. |
| Are lost orders observable? | State what the available evidence can and cannot identify about latent demand. |
| Does a forecast cover lead time only? | Match the scored periods to the time before an order can replenish stock. |
| Is a review period also covered? | Include the additional exposure if the actual order-up-to policy covers it. |
| Can lead time vary? | Score realized lead-time exposure separately from any fixed contractual horizon. |
| Are arrivals partial? | Describe how staggered receipts change the period each decision must cover. |
| Are there supplier calendars? | Apply cutoff, weekend, and holiday rules as they were known at the origin. |
| Are forecasts consumed directly? | Capture rounding, overrides, constraints, and any downstream aggregation. |
| What happens if data are late? | Replay the operational fallback and retain its decisions in the main population. |
| What happens if the model fails? | Define the fallback owner, deadline, and escalation before evaluating availability. |
| Are human overrides permitted? | Distinguish model, override, and final-action performance without assuming causality. |
| Is a confidence range consumed? | Evaluate quantile calibration at the actual decision horizon, not only daily point error. |
| Which cost matters? | Specify underage, overage, handling, spoilage, or contractual penalties where measurable. |
| Who bears errors? | Identify whether stockouts, excess, and allocation failures accrue to different owners. |
| Is storage capacity binding? | Separate forecast error from the downstream policy's feasible action set. |
| Is a minimum order binding? | Include its conversion effect when judging final decisions. |
| Can one site substitute for another? | Capture transfers and availability before assigning a miss to one location. |
| Is substitution between products possible? | Record the substitution rule; observed sales may move when another item stocks out. |
| Do promotions change the feasible price? | Preserve the planned price vintage rather than a later realized price. |
| Is a new product expected? | Evaluate the launch fallback even when no item-level demand history exists. |
| Can an item be discontinued? | Determine whether forecasts after its effective end date are in scope. |
| What label is considered mature? | Record the maturity lag and the restatement policy independently of model features. |
| When are results reviewed? | Keep early provisional reporting separate from mature-score comparisons. |
| What is the decision population? | Include all eligible decisions, not only successful candidate forecasts. |
| What is an acceptable abstention? | Publish the conditions and coverage of any fallback or manual review. |
| Which candidate wins? | Freeze the primary loss, minimum sample, uncertainty rule, and guardrails before evaluation. |
| What can be concluded offline? | Restrict claims to the replayed forecasting task unless a causal policy evaluation is designed. |
| What changes the contract? | Revisit the experiment when cadence, source availability, target, or action policy changes. |
## 3. Foundational Definitions
A **series** is an entity at the decision grain, such as a SKU-store pair.
SKU means stock keeping unit, an item identifier.
A **period** is an agreed non-overlapping interval, such as a local calendar day or a business week.
Boundaries, timezone, and daylight-saving transitions belong in the contract.
A **target** is a specified quantity, such as fulfilled units, orders, or latent unconstrained demand.
The three are not interchangeable.
A **forecast origin** is the time at which the model's information set closes.
A decision may happen later, but the simulation must use the actual feature freeze.
A **horizon** is a positive integer number of periods after the origin in this sheet.
For intraday systems, define exact elapsed-time boundaries instead.
A **fold** is one origin plus its training history and its selected future target periods.
An **expanding window** keeps the historical start fixed while the origin advances.
A **sliding window** advances the start as well as the end.
A **refit** re-estimates model parameters.
A **forecast refresh** creates new predictions, potentially using the same fitted parameters.
A **gap** removes observations from a region near the evaluation boundary to model delay or prevent overlap.
It is not a substitute for checking every data availability timestamp.
A **vintage** is the value of a revisable data item as it was available at a particular time.
Revised history is not necessarily the historical information set.
Event time records when something occurred; availability time records when the forecasting system could use it.
Both matter.
A **label maturity time** is when a target or corrected target is reliable enough under the evaluation policy.
Reporting early results is possible but must label them provisional.
A **baseline** is a simple, frozen competitor given the same legal information, horizon, target, and scoring population.
**Seasonal naive** repeats the most recent available observation from the corresponding seasonal position.
With a long forecast horizon, the rule must explicitly repeat the seasonal cycle rather than use future actuals.
**Bias** in this sheet means mean forecast minus actual, so positive values indicate overforecasting.
Other teams use the opposite sign; document it.
**MAE** is mean absolute error, measured in target units.
**RMSE** is the square root of mean squared error, also in target units.
**WAPE** is weighted absolute percentage error: aggregate absolute error divided by aggregate actual demand, multiplied by 100 for a percent.
**MAPE** averages individual absolute percentage errors.
It is undefined for zero actuals and unstable near zero.
**MASE** divides each error by a training-only in-sample naive error scale for that series and fold, before averaging.
**Pinball loss** evaluates one forecast quantile and gives an asymmetric penalty for under- and over-prediction.
A **stockout** can censor demand: sales cannot reveal all demand that would have arrived with unlimited availability.
A **hierarchy** maps detailed series to totals, for example SKU to category and store to region.
Coherent forecasts sum consistently across the declared hierarchy.
An **evaluation population** is the explicit set of series, origins, horizons, and target periods eligible for the reported metric.
**Takeaway:** Define the target, clocks, grain, and sign conventions before choosing a metric.
### 3.1 Point-in-time record contract
A durable replay needs a record of the forecast as issued, not just a recomputable model expression.
The following fields answer different audit questions and should not be conflated.
| Field | Meaning and audit use |
|---|---|
| Forecast identifier | Immutable identifier for one prediction version and its issuing run. |
| Series identifier | Entity at the declared target grain, with an effective mapping. |
| Origin timestamp | Information cutoff expressed with timezone and unambiguous offset. |
| Run start timestamp | Operational start of computation; this is not automatically the feature cutoff. |
| Run completion timestamp | Determines whether the forecast arrived before the action deadline. |
| Target start | Inclusive start of the predicted period under the calendar contract. |
| Target end | Exclusive end of that period, unless another interval convention is declared. |
| Horizon | Declared distance from origin to target, not merely elapsed seconds. |
| Prediction value | Issued numeric forecast in the target's agreed unit. |
| Prediction type | Mean, median, quantile, scenario, or other declared forecast functional. |
| Quantile level | Probability level for a quantile forecast, where relevant. |
| Model artifact identifier | Exact fitted artifact, including training configuration and version. |
| Training cutoff | Latest legal label availability included in fitting the artifact. |
| Feature definition version | Transform and schema used when assembling model inputs. |
| Feature source version | Selected as-of source record or reproducible immutable snapshot. |
| Source publication time | Time the upstream producer made that version available. |
| Ingestion time | Time the forecasting platform could first access that version. |
| Materialization time | Time the transformed value became available to the forecasting job. |
| Missingness flag | Whether a value was absent at the cutoff, including the reason if known. |
| Fallback flag | Whether a fallback produced the issued forecast and which fallback was used. |
| Calendar version | Calendar and timezone rules for mapping instants to target periods. |
| Hierarchy version | Effective product and location mapping used for aggregation. |
| Eligibility version | Rules defining which decisions enter the denominator. |
| Actual vintage | Outcome value and correction state used by a particular report. |
| Maturity timestamp | When the chosen scoring policy permits that outcome to be called mature. |
| Scoring version | Loss definition, rounding, exclusions, and aggregation implementation. |
| Override record | Human or policy change after model output, with time and reason where available. |
| Action record | Final order or allocation, stored separately from the forecast. |
| Failure record | Failed or late runs retained with their fallback or unserved status. |
| Report identifier | Immutable link between the score and all inputs used to compute it. |
### 3.2 Availability states and their consequences
| State | Decision-time interpretation |
|---|---|
| Event occurred, no record published | The value is not available even if the event date precedes the origin. |
| Published but not ingested | The source knew the value, but the forecasting platform could not use it. |
| Ingested but transformation unfinished | The raw value may exist while the serving feature remains unavailable. |
| Materialized exactly at cutoff | Apply the documented inclusive or exclusive boundary consistently. |
| Available after cutoff | Exclude it from that run even if it arrives before the order is placed. |
| Earlier version available | Select that legal version if the current one arrived too late. |
| No earlier version exists | Apply the recorded missingness policy rather than a later backfill. |
| Upstream value retracted | Respect the retraction only in snapshots after its availability time. |
| Timestamp corrected retroactively | Preserve both correction and original ingestion evidence for audit. |
| Multiple versions share timestamps | Use a deterministic sequence or source version tie-breaker. |
| Warehouse row overwritten | Historical reconstruction is unsupported unless another archive exists. |
| Label corrected after fitting | Keep the original training label in the historical model state. |
| Label corrected before scoring | Use it only if the declared actual-vintage policy permits it. |
| Clock skew is suspected | Compare synchronized logs and treat uncertain cutoff ordering as a limitation. |
| Local time repeats at daylight saving | Store an offset or UTC instant to distinguish the repeated times. |
| Local time is skipped at daylight saving | Define the scheduler's actual behavior instead of inventing a missing origin. |
## 4. Core Formulas with Full Derivation
### 4.1 Base: error and mean absolute error
Validity: Every evaluated prediction and actual refer to the same series and target period, measured in the same unit. Missing outcomes are not zero-filled without evidence. DERIVED: Subtracting the actual from a prediction gives signed error. Taking absolute values discards direction; averaging over the chosen population yields an average magnitude.
```text
FORMULA [E1]: e_j = f_j - y_j; MAE = sum_j |e_j| / n
WHERE: j indexes n forecast-target pairs; f_j forecast [units];
       y_j actual [units]; e_j [units]; n positive integer.
CONSTRAINT: matching grain, units and target definition; n > 0.
NUMERICAL EXAMPLE: actual=(10,20), forecast=(12,17);
                   errors=(2,-3), MAE=(2+3)/2=2.5 units.
INTERPRETATION: average absolute miss is 2.5 units per pair.
```
Boundary: A two-unit error for an item with typical demand of two is different from the same error for an item with typical demand of two thousand. MAE alone does not make those business impacts comparable. The mean signed error is separately useful for diagnosing directional over- or under-forecasting; opposite signed misses cancel there but not in MAE.
### 4.2 General: rolling-origin estimate
Validity: Origins and target horizons match the intended operational decision, and all training and feature values are legal as of each origin. The following describes an empirical average, not an unbiased guarantee for future demand. DERIVED: Each origin produces a vector of forecast errors. Assign a nonnegative weight to each eligible series-origin-horizon cell and divide the weighted loss sum by the sum of weights.
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
Boundary: Setting weights after viewing errors can favor a preferred model. A weight derived from future realized demand can be a legitimate descriptive scoring weight if declared, but cannot be used as an as-of model feature or a prospective decision weight without justification. Choose equal origin weights when each decision cycle is equally important; choose explicit business weights when those reflect the objective. A series with more eligible dates otherwise contributes more cells. Overlapping horizons cause some target dates to appear multiple times. This is correct for evaluating every issued decision but not equivalent to a report with one forecast per realized date.
### 4.3 Variants: WAPE, RMSE, MASE, and quantile loss
Validity for WAPE: Nonnegative actual demand and strictly positive total actual demand on the scored population.
Forecast and actual units must match.
DERIVED: Sum the magnitudes of all unit errors and normalize by the sum of observed demand.
This equals a demand-weighted average of individual absolute percentage errors only for positive individual actuals; the aggregate expression still includes errors when some actuals are zero.
```text
FORMULA [E3]: WAPE = 100 * sum_j |f_j-y_j| / sum_j y_j
WHERE: j indexes scored forecast-target pairs; y_j >= 0;
       f_j and y_j measured in the same units; result in percent.
CONSTRAINT: sum_j y_j > 0; define treatment of returns and missingness.
NUMERICAL EXAMPLE: actual=(10,20), forecast=(12,17);
                   WAPE=100*(2+3)/(10+20)=16.67%.
INTERPRETATION: absolute error totals one sixth of actual volume.
```
Boundary: If total actual demand is zero, report undefined WAPE and the absolute error in units, not a fabricated zero or infinity. WAPE can improve merely because total demand grows while absolute error does not. Validity for RMSE: finite squared errors and a meaningful reason to give larger errors greater influence. DERIVED: Squaring error makes positive and negative misses contribute positively and emphasizes large misses; the square root returns the result to the original unit.
```text
FORMULA [E4]: RMSE = sqrt(sum_j (f_j-y_j)^2 / n)
WHERE: n > 0 matched pairs; forecast and actual in target units.
CONSTRAINT: squared errors are finite; population is fixed.
NUMERICAL EXAMPLE: errors=(2,-3); RMSE=sqrt((4+9)/2)
                   =sqrt(6.5), approximately 2.55 units.
INTERPRETATION: a large miss affects RMSE more than MAE.
```
Boundary: RMSE can prioritize an exceptional outlier over many smaller operational misses; investigate the outlier and the business loss rather than automatically discarding it. Validity for MASE: A nonzero seasonal-naive in-sample scale computed only from legally available training observations, with a defined seasonal period and sufficient training history. DERIVED: Divide the out-of-sample absolute forecast error by the average in-sample absolute difference from the seasonal lag. Averaging these dimensionless ratios yields the scaled error.
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
Boundary: A constant or all-zero training series has zero scale. Mark its scaled score undefined and report another metric; never add an undocumented epsilon. A MASE below one compares with an in-sample naive scale, not necessarily a same-horizon out-of-sample baseline. Validity for quantile loss: The target quantile level is fixed strictly between zero and one and the outcome is numeric. DERIVED: The asymmetric absolute loss charges underprediction at the quantile level and overprediction at one minus that level. Its population minimizer is a corresponding conditional quantile under standard integrability conditions.
```text
FORMULA [E6]: pinball_q(y,f) = q*(y-f) if y >= f,
                                (1-q)*(f-y) otherwise
WHERE: q in (0,1) is dimensionless; y actual and f forecast [units].
CONSTRAINT: evaluate a declared quantile on matched target pairs.
NUMERICAL EXAMPLE: q=0.8, y=10, f=7 -> loss=0.8*3=2.4 units;
                   y=7, f=10 -> loss=0.2*3=0.6 units.
INTERPRETATION: this quantile penalizes a shortfall more heavily.
```
Boundary: The 0.8 quantile need not equal the mean or median. Its usefulness depends on the actual shortage and surplus trade-off, not on a generic claim that higher is safer.
### 4.4 Limits and counterexample: causal feature availability
A chronological split is necessary for this replay but not sufficient.
For each historical training row, the feature value must be the value known at that row's simulated origin, not a later corrected value merely associated with an earlier event date.
Consider an actual weather reading recorded on Friday.
An extract may label it with Friday's date, but a Tuesday forecast for Friday could only use a weather forecast published by Tuesday's freeze time.
Likewise, a promotion planned on Tuesday but amended on Thursday must be represented by Tuesday's plan in Tuesday's forecast.
The realized Thursday promotion is a legitimate target-time description, not a legal Tuesday feature.
A global scaler fitted on all dates passes a superficial chronological row split while leaking later distribution information into earlier training and predictions.
An embargo of one day does not fix a weekly batch that arrives two days late.
Derive availability from the source's publication and ingestion rules.
Preconditions: an immutable origin, an available-at timestamp for each feature version, and a deterministic tie-breaking rule.
State transition: filter candidate versions to those published and ingested by the cutoff; select the latest legal version for the entity and target period; then record its lineage with the prediction.
Invariant: no selected record has an availability time after the origin's feature freeze.
Justification: the filter excludes every later record before selection.
Trace: a plan for Friday published Monday is selected at Tuesday's origin; a revision published Wednesday is not; a correction ingested after the Tuesday freeze is not selected even if its source timestamp says Monday.
Recovery: if no legal version exists, use the declared missing-value handling or a safe baseline; do not backfill from a later vintage.
Failure boundary: a source that overwrites history without publication or ingestion timestamps cannot support a faithful as-of replay until historical vintages are recovered or the missingness is explicitly acknowledged.
**Takeaway:** Derive each score from a declared population and enforce the information boundary on every historical feature and transformation.
### 4.5 Additional scoring and decision calculations
The calculations below are illustrative and use the error convention forecast minus actual.
A metric is a summary of a declared population; no single value establishes operational benefit.
#### Signed bias and cancellation
Let forecasts be 12, 8, and 10 against actuals 10, 10, and 10 units.
The signed errors are +2, -2, and 0 units.
The mean signed error is (2-2+0)/3 = 0 units.
The MAE is (2+2+0)/3 = 4/3, approximately 1.33 units.
Zero signed bias therefore does not imply accurate individual forecasts.
If all three decisions are aggregated into one total, the forecast and actual both equal 30 units.
The absolute aggregate error is zero even though two individual forecasts miss by two units each.
This cancellation is appropriate only when the decision can really pool those units.
#### WAPE weighting and zero demand
Consider actuals of 100, 10, and 0 units with forecasts of 110, 0, and 5 units.
The absolute errors are 10, 10, and 5, for a total of 25 units.
Total actual demand is 110 units, so WAPE is 100*25/110, approximately 22.73%.
The zero-actual cell contributes five units to the numerator and zero to the denominator.
The individual percentage error on that cell is undefined, but aggregate WAPE remains defined.
If the first item alone is scored, its WAPE is 10%; the second item's is 100%.
A simple mean of those two percentages is 55%, and it excludes the zero-actual item's error.
That mean answers a different question from the 22.73% aggregate value.
#### Pooled and equal-series MAE
Suppose series A has absolute errors 1, 1, and 1; series B has a single absolute error of 5.
The pooled MAE is (1+1+1+5)/4 = 2 units per scored cell.
The per-series MAEs are 1 and 5 units, so the equal-series average is (1+5)/2 = 3 units.
The difference arises because the pooled calculation gives A three times as many observations as B.
Neither aggregation is automatically correct: decide whether cells or series represent the business objective.
An equal-series score still requires a minimum-history policy to avoid unstable slice rankings.
#### Weighted signed error and asymmetric loss
Suppose errors are -3 and +2 units, with nonnegative weights 2 and 1.
The weighted mean signed error is (2*(-3)+1*2)/3 = -4/3 units.
The weighted MAE is (2*3+1*2)/3 = 8/3 units.
If each unit underforecast costs 4 and each unit overforecast costs 1, weighted cost is (2*12+1*2)/3 = 26/3 cost units per weighted cell.
The cost calculation is illustrative and assumes linear, separable costs at this grain.
It must not be interpreted as an inventory profit estimate without the relevant policy and costs.
#### Quantile choice under a simple one-period loss
Assume a unit of shortage costs 3 and a unit of surplus costs 1, with linear costs and no other constraints.
The implied critical fractile is 3/(3+1) = 0.75.
A 0.75 conditional demand quantile minimizes expected cost under those stated assumptions.
For demand 12 and forecast 10, pinball loss at 0.75 is 0.75*2 = 1.5 units.
For demand 10 and forecast 12, it is 0.25*2 = 0.5 units.
The corresponding actual costs are 6 and 2, exactly four times these pinball losses.
Fixed ordering costs, capacity, substitutions, and multi-period effects can invalidate this simple decision rule.
#### Empirical interval coverage and width
For four actuals 4, 5, 8, and 10, suppose issued intervals are [3,5], [4,6], [5,7], and [8,11].
Three of four actuals are covered, giving empirical coverage 3/4 = 75%.
The widths are 2, 2, 2, and 3 units; their mean is 9/4 = 2.25 units.
These four cases cannot establish calibration, especially when targets or origins overlap.
Always state whether endpoints count as covered; this example includes both endpoints.
Report coverage by horizon and relevant slice alongside width and sample count.
A broad interval can improve coverage without being a useful operational prediction.
#### Matched comparison and missing forecasts
For three shared cells, baseline absolute errors are 3, 2, and 4; candidate errors are 2, 3, and 1.
The paired improvements, baseline minus candidate, are 1, -1, and 3 units.
Their mean is 1 unit, equal to baseline MAE 3 minus candidate MAE 2.
Suppose a fourth eligible decision has baseline error 2 but no candidate forecast.
Dropping it silently measures conditional performance on successful candidate runs, not the proposed service.
If the declared fallback has error 4, candidate-plus-fallback MAE is (2+3+1+4)/4 = 2.5 units.
The baseline on those same four decisions has MAE (3+2+4+2)/4 = 2.75 units.
The full-policy paired improvement is 0.25 units, smaller than the shared-cell estimate.
#### Seasonal naive beyond one cycle
At an origin, the last complete seven-day cycle is 2, 3, 4, 5, 6, 7, 8 units.
For horizons 1 through 7, repeat those values in that order.
For horizon 8, repeat the first value, 2, rather than using the as-yet-unobserved horizon-1 actual.
For horizon 9, repeat the second value, 3, under this same frozen-cycle rule.
If the most recent day is not available at the origin, construct the seasonal reference from the latest legal observations.
A different missing-day policy is valid only if it is specified and replayed consistently.
#### Fixed-origin versus updated forecast
Let a two-day target have actual demand 9 and 11 units.
At the first origin the candidate forecasts 8 and 10; both daily errors are -1 and the total error is -2.
At the next origin, after the first day becomes legally available, the revised forecast for the second day is 12.
That revised forecast has error +1, but it was not available to the first decision.
A first-origin lead-time score uses 8+10 = 18 against 20, not 8+12 = 20.
Using the revised value in the earlier order would leak information even though both forecasts are genuine.
#### Quantiles do not add in general
Let each of two daily demands be 0 or 10 with equal probability, and assume independence for this example.
Using the lower median convention, each daily median is 0.
The sum has probabilities 0.25 at 0, 0.50 at 10, and 0.25 at 20; its median is 10.
Adding the chosen daily medians gives 0, not the median of the sum.
Dependence between daily outcomes changes the distribution of the total and must be represented when relevant.
For a lead-time quantile, model or simulate the joint demand path and score the resulting total quantile.
#### Score maturity and actual restatement
Suppose a frozen forecast is 10 units, the first mature actual is 8, and the restated actual is 11.
The first-mature absolute error is 2 units; the restated absolute error is 1 unit.
The one-unit change is due entirely to the actual vintage, not a new forecast.
Publish the two scores with their as-of dates rather than silently overwriting the earlier report.
If the restatement was unknown at the origin, it cannot retroactively become a training input in that replay.
#### Relative change with zero baseline
Suppose baseline MAE is 0 and candidate MAE is 1 unit on the same cells.
The absolute candidate-minus-baseline difference is +1 unit.
A relative change divided by baseline MAE is undefined because the denominator is zero.
Report the absolute difference and underlying errors rather than an invented percentage.
If both scores are zero, the absolute difference is zero but the relative change remains undefined.
#### Percentage-point arithmetic
If baseline WAPE is 24% and candidate WAPE is 18%, the absolute difference is 6 percentage points.
The relative reduction is (24-18)/24 = 0.25, or 25%.
The shared actual total matters: changing the denominator between models makes a direct comparison misleading.
If total demand doubles with fixed aggregate absolute error, WAPE halves without reducing unit error.
Pair the ratio with absolute error totals and volume when interpreting changes over time.
#### Rolling lag legality
At a Monday 09:00 cutoff, suppose Sunday's actual is not materialized until Monday 10:00.
A feature named lag-1 that references Sunday is unavailable, even though Sunday precedes Monday on the calendar.
If Saturday's actual is available, it can support a declared last-available-value fallback.
The age of that value is two calendar days, so its age should accompany the feature.
A training example must reproduce that historical lag and missingness state rather than use today's corrected table.
#### Fold-local scaling
At origin A, the legal training values are 2 and 4; their mean is 3.
A value of 4 centered by that training mean becomes 1.
A future value of 10 would make the mean of all three values 16/3, approximately 5.33.
Centering with 16/3 leaks information from the held-out period into the earlier fold.
At the next origin, the 10 may enter the fitting mean only if its label is actually available by then.
#### Gap versus delayed labels
Assume labels for day t become usable at 10:00 on day t+2, and the forecast cutoff is day 10 at 09:00.
Day 8's label is not yet usable because it arrives at 10:00 on day 10.
Day 7's label is usable if its expected arrival at day 9 10:00 actually occurred.
A two-day calendar gap that includes day 8 would therefore be insufficient at this cutoff.
The replay should use recorded arrival timestamps and a defined fallback for missing records.
A fixed gap is at best a conservative approximation if it covers the worst permitted availability delay.
#### Moving averages and overlapping labels
Suppose a target is the sum of demand on days t+1 through t+3.
At origin t+1, the preceding target's last component is still in the future.
Training on that preceding target at origin t+1 would leak unrealized demand.
The label becomes eligible only after all three components have arrived and met the maturity policy.
The same rule applies to engineered rolling targets, future revenue totals, and lead-time demand labels.
#### Hierarchy coherence
At one origin, item forecasts are 4 and 7 units and their declared category forecast is 10 units.
The child sum is 11, so the category-minus-child discrepancy is -1 unit.
The comparison is meaningful only if both items and the category share target period, unit, and mapping vintage.
Replacing the category forecast with 11 enforces this particular sum but may alter its accuracy.
Keep raw and reconciled predictions separately so the reconciliation effect can be evaluated.
#### Rounding and order multiples
Suppose a point forecast is 13 units and the supplier only accepts orders in multiples of 6.
Rounding up yields 18 units, while rounding to the nearest multiple yields 12 units.
Against realized demand of 14, the raw forecast's absolute error is 1 unit.
The two rounded quantities miss the actual by 4 and 2 units respectively, before accounting for inventory on hand.
These are not complete order costs: opening inventory, lead time, and policy must also be included.
#### Coverage under fallback
Assume 100 eligible decisions, of which the candidate serves 80 on time and a fallback serves the remaining 20.
Candidate availability is 80%; policy availability is 100% if every fallback also arrives on time.
If the 80 candidate errors total 80 units and the 20 fallback errors total 60 units, policy MAE is 140/100 = 1.4 units.
The candidate-only MAE is 80/80 = 1 unit, which is not the full-policy result.
Report timing and coverage beside accuracy so a missing difficult segment does not disappear from the comparison.
#### Lead-time exposure under variable arrival
Suppose a decision forecasts daily demand 3, 4, and 5 units for the next three days.
If the receipt arrives at the start of day 3, demand before receipt is 3+4 = 7 units.
If it arrives at the end of day 3, the comparable three-day forecast total is 12 units.
Define whether receipt timestamps are inclusive and whether intra-day demand is available before determining exposure.
Using a future realized receipt time for a retrospective slice is descriptive, not a legal origin-time feature.
#### Discounted and undiscounted decisions
Suppose absolute misses of 2 and 6 units occur one and two periods after the origin.
Equal horizon weights give MAE (2+6)/2 = 4 units.
Weights 2 and 1 give weighted loss (2*2+1*6)/3 = 10/3 units.
The shorter-horizon emphasis is justified only if it reflects the decision or a predeclared priority.
Weights chosen after observing the two errors would bias the reported comparison.
## 5. Worked Numerical Example
### 5.1 Decision and data contract
ILLUSTRATIVE: A planner orders one SKU for one store at the end of each day, for delivery covering the next two days. The forecast origin occurs after that day's finalized sales file arrives. There is no ingestion delay in this deliberately small example. The target is observed fulfilled units, not inferred unconstrained demand. No stockout adjustment is assumed. Two origins, day 4 and day 5, are replayed. Both models use an expanding training window starting at day 1. Daily actuals for days 1 through 7 are respectively 8, 10, 12, 10, 14, 13, and 15 units. At day 4, training ends with day 4; day 5 and day 6 are held out. At day 5, day 5 is now known; day 6 and day 7 are held out. The same realized day 6 therefore appears at horizon 2 for origin 4 and horizon 1 for origin 5. These are two different operational forecasts. The baseline forecasts the last observed actual at both horizons. The candidate's predictions are supplied below as hypothetical values; no fitting result is claimed.
```text
Origin  Horizon  Target  Actual  Baseline  Candidate
4       1        5       14      10        12
4       2        6       13      10        12
5       1        6       13      14        13
5       2        7       15      14        14
```
The baseline uses only day 4 at the first origin and day 5 at the second.
The candidate must also be generated using only data available at the respective origin.
The actuals are shown together only for explanatory scoring after maturation.
Their placement in this table does not make them legal model inputs.
### 5.2 Error table and arithmetic
```text
Origin  Horizon  Baseline error  Candidate error  Baseline AE  Candidate AE
4       1        -4              -2               4            2
4       2        -3              -1               3            1
5       1        +1               0               1            0
5       2        -1              -1               1            1
Total            -7              -4               9            4
```
Here AE means absolute error in units.
Error is forecast minus actual, following Section 3.
Using E1, baseline MAE is 9/4 = 2.25 units per forecast-target pair.
Candidate MAE is 4/4 = 1.00 units.
The candidate's absolute improvement is 1.25 units per pair.
Its relative reduction against this baseline is 1.25/2.25, approximately 55.56%.
The denominator in that relative comparison is the baseline MAE, not the candidate MAE.
A zero baseline MAE would make a relative reduction undefined.
Using E3, the repeated-decision actual total is 14+13+13+15 = 55 units.
Baseline WAPE is 9/55 times 100, approximately 16.36%.
Candidate WAPE is 4/55 times 100, approximately 7.27%.
Both use the same repeated-decision population.
The WAPE difference is approximately 9.09 percentage points, not 9.09 percent relative improvement.
Relative reduction is 5/9, approximately 55.56%.
Using E4, baseline squared-error sum is 16+9+1+1 = 27, giving RMSE sqrt(27/4), approximately 2.60 units.
Candidate squared-error sum is 4+1+0+1 = 6, giving RMSE sqrt(6/4), approximately 1.22 units.
Mean signed errors are -7/4 = -1.75 and -4/4 = -1 unit.
Both models underforecast on average under the chosen sign convention.
For horizon 1, baseline MAE is (4+1)/2 = 2.5 and candidate MAE is (2+0)/2 = 1.0 units.
For horizon 2, baseline MAE is (3+1)/2 = 2.0 and candidate MAE is (1+1)/2 = 1.0 units.
The candidate is better in both displayed horizon slices, but two origins do not establish a reliable population improvement.
Independent arithmetic check: forecast totals are 48 for baseline and 51 for candidate, compared with actual total 55; their summed signed errors are therefore -7 and -4.
The absolute error totals reconstructed from the row errors are 9 and 4.
Multiplying the displayed WAPE values by 55/100 recovers those totals up to rounding.
### 5.3 Lead-time aggregation changes the question
Suppose one decision requires the sum of the two daily predictions, rather than treating each day as a separate decision.
For origin 4, actual lead-time total is 14+13 = 27 units.
The baseline total is 10+10 = 20 and the candidate total is 12+12 = 24; absolute errors are 7 and 3 units.
For origin 5, actual total is 13+15 = 28; baseline total is 14+14 = 28 and candidate total is 13+14 = 27.
The respective absolute errors are 0 and 1.
Lead-time-total MAEs are therefore (7+0)/2 = 3.5 and (3+1)/2 = 2 units per order decision.
Aggregation allows errors in opposite directions to cancel, as at origin 5 for the baseline.
Daily MAE and lead-time-total MAE answer different questions and neither replaces the other.
A forecast for cumulative demand should ideally be evaluated against actual cumulative demand at the exact replenishment lead time and decision cadence.
### 5.4 Sensitivity and decision boundary
At origin 5, replace the day 7 actual of 15 by 20 units, leaving all forecasts unchanged.
The baseline's horizon-2 error becomes -6 and the candidate's becomes -6.
Their total absolute errors become 14 and 9, respectively.
Candidate MAE becomes 2.25 and baseline MAE becomes 3.5 units.
Candidate relative MAE reduction falls to (3.5-2.25)/3.5, approximately 35.71%.
A single large target substantially changes the apparent relative advantage in this tiny sample.
If candidate day 7 forecast instead equals 8 while the original actual remains 15, candidate absolute-error total becomes 2+1+0+7 = 10, worse than baseline's 9.
This demonstrates why model selection based on a few origins is fragile.
A decision maker needs many relevant historical periods, uncertainty intervals, and future operational monitoring.
A sales-based result cannot establish improvement in latent demand or service level when inventory availability constrains sales.
The latter requires a separate measurement strategy.
**Takeaway:** Reconstruct the exact forecast cells, check the arithmetic, and score both daily and decision-level quantities where each has a clear use.
### 5.5 Complete delayed-feed replay
ILLUSTRATIVE: A store orders daily at 09:00 for the next two local calendar days.
The forecast freeze is 08:30, and each day's sales become usable at 09:00 the following morning.
The model refits daily using the latest labels available at the freeze.
No intra-day demand, stockouts, or corrections are assumed in this exercise.
The observed sales for days 1 through 7 are 4, 6, 5, 7, 8, 6, and 9 units.
At day 4 08:30, day 3 sales are not yet usable; legal training sales are 4 and 6.
At day 5 08:30, day 4 sales are not yet usable; legal training sales are 4, 6, and 5.
The last-available baseline forecasts the most recent legal sales value at both horizons.
The candidate's hypothetical forecasts and matured actuals are shown below.
| Origin | Horizon | Target | Last legal label | Actual | Baseline | Candidate |
|---|---:|---:|---:|---:|---:|---:|
| Day 4 | 1 | Day 5 | 6 | 8 | 6 | 7 |
| Day 4 | 2 | Day 6 | 6 | 6 | 6 | 7 |
| Day 5 | 1 | Day 6 | 5 | 6 | 5 | 6 |
| Day 5 | 2 | Day 7 | 5 | 9 | 5 | 8 |
The baseline signed errors are -2, 0, -1, and -4 units.
The candidate signed errors are -1, +1, 0, and -1 units.
The baseline absolute-error sum is 7 and its MAE is 7/4 = 1.75 units.
The candidate absolute-error sum is 3 and its MAE is 3/4 = 0.75 units.
The paired MAE difference is 1 unit, or 1/1.75, approximately 57.14% relative to baseline.
The repeated-decision actual total is 8+6+6+9 = 29 units.
Baseline WAPE is 100*7/29, approximately 24.14%; candidate WAPE is 100*3/29, approximately 10.34%.
The baseline mean signed error is -7/4 = -1.75 units; the candidate mean signed error is -1/4 = -0.25 units.
At day 4, the baseline lead-time total is 12 versus actual 14, giving absolute error 2.
At day 4, the candidate lead-time total is 14 versus actual 14, giving absolute error 0.
At day 5, the baseline lead-time total is 10 versus actual 15, giving absolute error 5.
At day 5, the candidate lead-time total is 14 versus actual 15, giving absolute error 1.
The lead-time-total MAEs are therefore 3.5 and 0.5 units per decision.
These results depend on the stated delay; pretending day 4 sales were legal at day 4 08:30 would change the baseline.
The candidate values are supplied rather than fitted, and two origins cannot establish a reliable improvement.
### 5.6 Matched slices and aggregation reversal
ILLUSTRATIVE: Compare two models on two populations with equal numbers of candidate and baseline forecasts within each population.
For the high-volume slice, the baseline makes 10 forecasts with total absolute error 20 units.
The candidate makes the same 10 forecasts with total absolute error 10 units.
The slice MAEs are 2 and 1 units, favoring the candidate by 1 unit.
For the low-volume slice, both models make 90 forecasts.
Their total absolute errors are 90 and 99 units respectively.
The slice MAEs are 1 and 1.1 units, favoring the baseline by 0.1 units.
Across all 100 cells, baseline MAE is (20+90)/100 = 1.1 units.
Candidate MAE is (10+99)/100 = 1.09 units, favoring the candidate by only 0.01 units.
An equal-slice average of slice MAEs is (2+1)/2 = 1.5 for the baseline.
The corresponding candidate average is (1+1.1)/2 = 1.05 units.
Equal-cell and equal-slice summaries use different weights and therefore answer different questions.
A critical-item guardrail could reject the candidate if the second slice contains essential products.
That rule and minimum slice sample should be set before examining the comparison.
No inference about individual item performance follows from these slice totals alone.
### 5.7 Vintage-aware promotion replay
ILLUSTRATIVE: A Friday promotion plan is published Monday at 10:00 with a 10% discount.
It reaches the forecasting system Monday at 10:15 and is materialized at 10:20.
The Tuesday forecast has a 09:00 freeze, so the 10% plan is legal for that forecast.
A revision to 20% is published Tuesday at 08:55 but reaches the system at 09:05.
Although published before the freeze, the revision is not ingested by the freeze and is illegal for that run.
The later Wednesday forecast may use the 20% revision if it has been materialized by Wednesday's cutoff.
A Thursday cancellation cannot be used in either earlier forecast merely because it describes Friday.
The actual discount executed on Friday is an outcome or retrospective diagnostic, not a Tuesday feature.
If the Monday plan was overwritten and no immutable archive exists, the Tuesday plan cannot be verified from the current table alone.
A replay without that archive must explicitly mark the missing evidence rather than silently use the 20% value.
### 5.8 Comparing quantile forecasts with an explicit loss
ILLUSTRATIVE: Actual demands are 4, 8, 10, and 12 units.
Candidate A issues 0.75-quantile forecasts of 5, 9, 9, and 10 units.
Candidate B issues 0.75-quantile forecasts of 7, 7, 11, and 11 units.
For A, the pinball losses are 0.25, 0.25, 0.75, and 1.5 units.
The total loss for A is 2.75 and the mean is 2.75/4 = 0.6875 units.
For B, the losses are 0.75, 0.75, 0.25, and 0.75 units.
The total loss for B is 2.5 and the mean is 2.5/4 = 0.625 units.
B has lower mean pinball loss on these four shared observations.
A has two outcomes at or below its forecast; B also has two, so both show 50% empirical coverage under the inclusive convention.
That small sample is not evidence that either model is calibrated or miscalibrated in the population.
Coverage alone cannot distinguish the two candidates here; the magnitudes and directions of misses differ.
If costs satisfy the simple 3-to-1 shortage-to-surplus assumptions, the corresponding average costs are four times the pinball losses.
Real inventory economics may differ, so that cost interpretation must be validated independently.
### 5.9 Block comparison without invented confidence bounds
ILLUSTRATIVE: A three-day lead-time forecast is issued daily, producing overlapping target windows.
Consider six consecutive origin-level paired loss differences: 1, -1, 2, 0, 3, and 1 units.
The mean paired difference is (1-1+2+0+3+1)/6 = 1 unit in favor of the baseline-minus-candidate convention.
Adjacent origins share target days, so the six numbers are not automatically independent.
One possible resampling design draws contiguous blocks of at least three origins, preserving both models within each block.
A longer block may be appropriate if demand shocks last longer than the lead-time overlap.
Six origins supply too little information for a stable block-bootstrap confidence interval under most plausible choices.
Report the observed paired differences, the tiny number of origins, and the absence of a defensible precision estimate.
Gather more decision periods or run a prospective evaluation rather than fabricating a confidence interval.
## 6. Visual / Geometric Intuition
```text
Time:        1   2   3   4   5   6   7
Origin 4:   [------train------]  H1  H2
Origin 5:   [--------train--------]  H1  H2
Target 6:                        H2  H1
```
The diagram places day 6 in two forecasts, one at each origin.
It is not double counting if both forecasts were actual decision opportunities.
For a one-forecast-per-target report, define which origin is retained before computing metrics, such as the forecast issued exactly two days ahead.
Visualize a second axis for availability time.
A historical row can sit to the left of the split by event time while arriving after the origin on the availability axis.
An as-of join moves along the availability axis and chooses a prior version, rather than simply selecting the latest value in the modern warehouse.
A prediction interval can be drawn as a band around a central forecast.
Counting target points within the band measures empirical coverage, while band width expresses sharpness.
A very wide band may achieve high coverage without being useful; a very narrow band may be useful only if it still achieves adequate coverage.
In a supply chain, actual sales can lie below true demand because inventory was exhausted.
A low residual against sales may conceal unmet demand.
**Takeaway:** A calendar diagram shows chronological order, but an availability-time diagram is needed to expose leakage.
### 6.1 Temporal diagrams for commonly confused clocks
| Clock pair | What can be mistaken | Correct reading |
|---|---|---|
| Event time and publication time | A prior event is assumed available immediately. | Only published versions can enter an as-of snapshot. |
| Publication and ingestion | An upstream timestamp is treated as proof of internal availability. | Check the system's ingestion and access time. |
| Ingestion and materialization | A raw record is mistaken for a ready model feature. | Use the actual serving feature availability. |
| Feature freeze and order deadline | Data arriving before order submission is assumed legal. | The frozen run cannot use data arriving after its own cutoff. |
| Forecast run and model refit | Every refresh is assumed to retrain parameters. | Reuse or refit exactly when the operating process does. |
| Target end and label arrival | Completed demand is assumed to be instantly observed. | Use the recorded label availability time. |
| Label arrival and maturity | The first reported outcome is treated as final. | Apply the declared correction and maturity policy. |
| First maturity and restatement | A later corrected score is treated as the original report. | Preserve both report vintages with lineage. |
| Origin and target date | Every forecast for the same date is pooled without horizon. | Retain origin, target, and horizon as a joint key. |
| Forecast and action | An accurate prediction is treated as the order itself. | Record inventory, constraints, overrides, and final action separately. |
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
The table is a selection aid, not a universal ranking.
Use a matched seasonal-naive or last-value comparator even when the candidate has sophisticated features.
If a proposed choice conflicts with an actual decision contract, the decision contract wins; update the evaluation rather than assuming a standard split applies.
**Takeaway:** Select the window, metric, and aggregation for the operational question, not for convenience or a single leaderboard.
### 7.1 Operational selection decisions
| If the decision is ... | Start with ... | Also check ... |
|---|---|---|
| A daily labor roster | Daily forecast loss at staffing freeze | Shift-level demand, overtime constraints, and schedule changes. |
| A weekly supplier order | Cumulative demand across the protection period | Supplier calendar, case sizes, stock on hand, and service. |
| A perishable item order | Asymmetric loss with documented waste and shortage costs | Shelf life, substitutions, and inventory age. |
| A critical spare part | Critical-item and zero-demand slices | Downtime consequence, long lead times, and sparse-history uncertainty. |
| A seasonal launch | As-of analog and cold-start fallback | Launch timing, available attributes, and launch cohort coverage. |
| A promotion replenishment | Plan-vintage-aware forecast and matched nonpromotion slices | Promotion execution differences and stockout censoring. |
| A regional allocation | Coherent location-level forecasts and allocation guardrails | Transfer constraints and local demand mix. |
| A category capacity plan | Aggregate total forecast at planning freeze | Whether item errors cancel in a way the operation can actually exploit. |
| An intermittent order | Count of positive-demand events and unit error | Zeros, size conditional on occurrence, and undefined percentage scores. |
| An uncertain supplier arrival | Demand over the actual protection period | Forecasting or scenario assumptions for lead-time uncertainty. |
| A manual planner workflow | Model and final-decision records | Override timing, selection into override, and no unsupported causal attribution. |
| A fully automated policy | On-time policy coverage and fallback performance | Data freshness, safety constraints, and rollback ownership. |
## 8. Failure Modes & Edge Cases
All cases in this section are ILLUSTRATIVE diagnostic scenarios, not documented incidents. Each requires evidence before attributing a root cause.
### F1. Implausibly accurate future forecasts
Symptom: Validation error is much lower than a production-like shadow period or a legal baseline. Competing causes: future feature leakage, accidental target join, real predictive signal, or a population mismatch. Diagnostic: For each selected record compare publication and ingestion time with the historical feature freeze; inspect join keys and target-date offsets. Expected signal for leakage: at least one used feature version was unavailable at its forecast origin, or an exact future target entered a feature. Bounded fix: reconstruct as-of snapshots and refit all fold-local transformations; do not merely drop the suspicious validation rows. Verification: replay the same origins with audited feature lineage and compare against a matched baseline; document any remaining missing vintages. Risk: replay may become impossible if sources have overwritten revisions. Escalate rather than silently substitute today's version.
### F2. Improvement disappears after a late data feed
Symptom: An offline model beats baseline but operational features are frequently missing at decision time. Competing causes: late ingestion, wrong timezone, upstream outage, or training-serving transformation mismatch. Diagnostic: Compare event, source-publication, ingestion, feature-materialization, and decision timestamps for a sample of missed runs. Expected signal for late ingestion: the source published in time but materialization completed after the feature freeze. Bounded fix: model historical availability honestly, adjust the feed SLA or use a documented fallback; do not shift the decision cutoff without planner approval. Verification: evaluate coverage and accuracy under the actual fallback path over comparable origins. Risk: a later cutoff may miss ordering deadlines even if it improves model accuracy.
### F3. WAPE looks excellent while low-volume items fail
Symptom: Aggregate WAPE improves but planners report persistent misses on low-volume or critical items. Competing causes: volume weighting, a genuine shift in item mix, rounding, or different eligibility rules. Diagnostic: Recompute shared-population unit errors, equal-series summaries, zero-demand errors, and critical-item slices alongside the aggregate. Expected signal for weighting: a few high-volume series dominate the denominator while many small series have large per-series misses. Bounded fix: retain WAPE for volume exposure but add an equal-series metric and operational service guardrails with minimum slice sizes. Verification: publish denominators, coverage, and paired comparisons by slice; make sure no undefined zero-total WAPE is displayed as zero. Risk: macro averages can overemphasize noisy low-volume series. Choose weights transparently with the decision owner.
### F4. Zero or negative demand breaks percentages
Symptom: MAPE becomes infinite or WAPE is missing or changes sign. Competing causes: zero-demand days, returns booked as negative units, missing values encoded as zero, or a unit conversion defect. Diagnostic: Separate zero, negative, null, and positive target counts; reconcile returns and sales to the source ledger. Expected signal for a true zero: the store was open, inventory was available, and the source asserts no units sold. Bounded fix: report absolute errors on zero-demand slices and define a returns-adjusted target if that is the decision quantity. Verification: calculate metrics on a manually inspected sample and verify denominators are strictly positive where required. Risk: excluding zero days biases evaluation toward high activity. Keep their errors in aggregate absolute-error totals where the metric permits.
### F5. Sales are accurate but shelves are empty
Symptom: Forecasts track observed sales while stockout rates remain high. Competing causes: censored demand, poor replenishment execution, incorrect lead times, or insufficient stock allocation. Diagnostic: Compare forecasts, on-hand inventory, availability intervals, lost-order signals, lead-time records, and order execution. Expected signal for censoring: sales terminate when available stock reaches zero while evidence of demand continues. Bounded fix: separate uncensored periods; estimate or bound missing demand only with explicit assumptions and independent validation. Verification: evaluate forecast and operational service on comparable availability regimes; use a controlled trial where feasible. Risk: treating a model-imputed lost-demand estimate as ground truth can reinforce its own assumptions.
### F6. A sudden metric jump after a data revision
Symptom: Historical scores change after a restatement even though frozen forecasts did not change. Competing causes: revised actuals, changed calendar, altered product hierarchy, or an accidental prediction overwrite. Diagnostic: Compare immutable forecast IDs, actual vintages, scoring code version, and mapping snapshots between reports. Expected signal for revision: identical forecast records joined to different corrected actual values. Bounded fix: publish both first-mature and latest-restated scores with their evaluation timestamps; do not rewrite prior reports without trace. Verification: rerun the same score on the same immutable inputs and compare cell-level differences. Risk: a restated truth may be more accurate, but it is not the truth available to the original monitoring process.
### F7. Scores improve when tuning on the final period
Symptom: Repeated model iterations improve the last holdout but not later observations. Competing causes: selection overfitting, a real improvement, changing data, or an untracked eligibility change. Diagnostic: Inspect experiment history, number of configurations, validation folds, and when the holdout was first consulted. Expected signal for selection overfitting: many decisions were made in response to the supposed final test results. Bounded fix: reserve a later untouched period, or evaluate a new forward shadow period; use earlier nested rolling folds for selection. Verification: pre-register the selection rule and evaluate once on the new protected period. Risk: waiting for a new holdout delays decisions; a cautious limited trial may be possible with explicit governance.
### F8. Duplicate target periods distort confidence
Symptom: Very small uncertainty intervals appear despite overlapping forecast horizons. Competing causes: treating dependent cells as independent, excessive weighting of repeated targets, or a computation bug. Diagnostic: Count unique target dates and origins; map shared targets and shared series to scored rows. Expected signal for dependence: many residuals share a target date, origin shock, or overlapping lead-time window. Bounded fix: use a block or cluster resampling design reflecting the dependence structure and compare paired model losses. Verification: report resampling unit, block length, effective number of blocks, and sensitivity to plausible alternatives. Risk: few blocks cannot support precise intervals; report uncertainty as wide or unquantified rather than manufacturing precision.
### F9. Hierarchy appears inconsistent
Symptom: Category forecast differs from the sum of item forecasts, or a change in item membership changes historical totals. Competing causes: independently fitted levels, different hierarchy vintages, unit conversion, or late item remapping. Diagnostic: Reaggregate lower-level predictions using the hierarchy known at each origin and compare with reported parent predictions. Expected signal for incoherence: a nonzero parent-minus-child sum under identical units, target dates, and mapping vintage. Bounded fix: apply a declared reconciliation procedure or select one authoritative aggregation level; preserve both raw and reconciled forecasts. Verification: assert coherence for every relevant origin and level, then reevaluate accuracy and operational constraints. Risk: forcing coherence may increase error for some nodes; it is a structural property, not a guarantee of higher accuracy.
### F10. Backtest excludes hard-to-forecast new products
Symptom: Offline accuracy is strong but live forecasts fail on launches. Competing causes: minimum-history filtering, entity mapping errors, or genuine launch drift. Diagnostic: Compare eligibility rates and excluded volumes by age, origin, and product segment. Expected signal for selection: live launches were omitted from historical folds because the model required history unavailable at launch. Bounded fix: evaluate an explicit cold-start policy using only launch-time attributes and a matched fallback. Verification: report full-population coverage, new-item scores, and performance of both model and fallback together. Risk: excluding difficult series can make a candidate look better while reducing the number of decisions it can serve. **Takeaway:** Diagnose competing causes with timestamped lineage and matched populations before deciding that the model itself is at fault.
### F11. A plan is published before the cutoff but arrives after it
Symptom: A replay selects a promotion revision that the serving platform never saw before the run.
Competing explanations: delayed ingestion, timezone conversion, or a mistaken source timestamp.
Diagnostic: Compare source publication, ingestion, feature materialization, and freeze for the exact revision.
Expected evidence: the selected revision has at least one required availability time after the cutoff.
Bounded response: select the latest fully available prior revision and replay all affected decisions.
Verification: inspect lineage at boundary timestamps, including an explicit equality rule.
Risk: source publication alone is not proof that the model could consume the value.
### F12. A daily label arrives after the following morning's freeze
Symptom: Offline lag-one features outperform the same feature in live runs.
Competing explanations: late labels, stale materialization, or a genuinely changed demand pattern.
Diagnostic: Trace the label from event completion through ingestion and feature publication.
Expected evidence: the label arrives after the historical cutoff in the apparently successful replay.
Bounded response: reconstruct the last legal label and its age at each origin.
Verification: compare live and replayed feature values for matched historical runs.
Risk: a fixed lag in calendar days does not guarantee a fixed data age.
### F13. A long-horizon baseline uses future actuals
Symptom: Seasonal-naive accuracy is implausibly high beyond one seasonal cycle.
Competing explanations: actual leakage, an unusually stable seasonal process, or incorrect horizon labels.
Diagnostic: Inspect every baseline dependency for horizons beyond the seasonal period.
Expected evidence: a predicted horizon references an observation that had not occurred at the origin.
Bounded response: repeat the last legal seasonal cycle under a documented missing-day rule.
Verification: perturb future actuals and confirm frozen baseline predictions do not change.
Risk: a leaky baseline can make a valid candidate look falsely uncompetitive.
### F14. The final holdout contains a transformed training statistic
Symptom: A chronological split still produces an unexpectedly favorable score.
Competing explanations: global normalization, target leakage, or a genuinely easy holdout regime.
Diagnostic: Trace the rows used to fit scalers, imputers, encoders, and feature selectors.
Expected evidence: a fitted transformation includes holdout observations or later labels.
Bounded response: fit each transformation inside the training portion of each origin.
Verification: perturb held-out data and confirm fitted training artifacts are unchanged.
Risk: correcting only the model fit leaves leakage inside preprocessing.
### F15. Retraining occurs more often offline than in operation
Symptom: The replay adapts rapidly while live results lag after a regime change.
Competing explanations: refit-cadence mismatch, late inputs, or deployment-specific defects.
Diagnostic: Compare historical artifact creation times and parameter versions with the simulated schedule.
Expected evidence: offline folds fit new parameters where the intended service reuses an older model.
Bounded response: replay the real refit schedule while allowing permitted forecast refreshes.
Verification: audit artifact identifiers and legal training cutoffs at every origin.
Risk: refitting more frequently estimates a different, potentially more expensive policy.
### F16. Refresh and refit are accidentally treated as identical
Symptom: A weekly model seems unable to use a newly available daily feature.
Competing explanations: frozen parameters, frozen input snapshot, or misunderstood service behavior.
Diagnostic: Separate parameter artifact timestamps from feature snapshot timestamps.
Expected evidence: the service refreshes inputs daily but the replay incorrectly freezes both state and inputs.
Bounded response: preserve the weekly artifact while reconstructing each legal daily feature snapshot.
Verification: compare forecast inputs and artifact IDs on several refresh days.
Risk: updating parameters without authorization still changes the evaluated system.
### F17. Sales are recorded at shipment rather than order time
Symptom: Forecasts appear late around supply disruptions and shipping backlogs.
Competing explanations: target timing mismatch, delayed shipping, or demand timing changes.
Diagnostic: Reconcile order creation, cancellation, shipment, and fulfillment timestamps.
Expected evidence: the target date is systematically shifted by the fulfillment process.
Bounded response: define the target matching the decision and preserve distinct shipment metrics.
Verification: manually trace a sample of orders across event dates and periods.
Risk: shifting all sales by an average delay can hide a variable backlog.
### F18. A zero is actually a missing source partition
Symptom: An apparent demand collapse coincides with a source or pipeline incident.
Competing explanations: true zero demand, a store closure, or missing ingestion.
Diagnostic: Compare source row counts, availability flags, store status, and ingestion logs.
Expected evidence: the partition is absent while the store was expected to report.
Bounded response: mark the observation missing and apply the predeclared incomplete-label policy.
Verification: reconcile repaired partitions against source totals and rescore with a new vintage.
Risk: replacing unknown values with zeros biases both training and WAPE.
### F19. Returns dominate a net-demand denominator
Symptom: WAPE is negative, undefined, or volatile despite modest unit misses.
Competing explanations: negative returns, unit conversion errors, or genuine low net volume.
Diagnostic: Separate gross sales, returns, cancellations, and target definition by period.
Expected evidence: aggregate net actual is nonpositive or near zero.
Bounded response: report unit error and evaluate a separately defined gross-demand target when appropriate.
Verification: reconcile both measures to the transaction ledger without discarding returns silently.
Risk: changing the target changes the question and requires a new matched baseline.
### F20. Late actual corrections alter past scores
Symptom: A dashboard changes historical performance without a new forecast run.
Competing explanations: revised actuals, mapping changes, or overwritten predictions.
Diagnostic: Compare forecast IDs, outcome versions, mapping versions, and scoring code.
Expected evidence: identical forecast values join to different actual vintages.
Bounded response: retain immutable first-mature and restated reports with timestamps.
Verification: recompute a sampled cell from each archived input version.
Risk: a newer value may improve truth quality without representing the original monitoring state.
### F21. An incomplete horizon looks better than a complete one
Symptom: Recent long-horizon scores improve while the newest targets are still immature.
Competing explanations: selective maturity, a real improvement, or a reporting-window change.
Diagnostic: Count eligible and mature cells separately for each origin and horizon.
Expected evidence: recent difficult targets are absent because their labels have not matured.
Bounded response: compare models on the same mature population and disclose incomplete origins.
Verification: wait for maturity and compare the predeclared cohort against its provisional report.
Risk: restricting to mature outcomes can still create a nonrepresentative recent cohort.
### F22. Daylight-saving fall-back duplicates an origin
Symptom: Two runs share the same local clock label but have different data availability.
Competing explanations: repeated local hour, scheduler retry, or duplicate event ingestion.
Diagnostic: Compare UTC instants, offsets, scheduler IDs, and intended business calendar.
Expected evidence: the local hour occurred twice with different offsets.
Bounded response: identify each real instant and apply the documented scheduling policy.
Verification: replay both transitions using the applicable timezone rules and archived run logs.
Risk: deduplicating only by local timestamp can discard a legitimate decision.
### F23. A spring clock transition skips a scheduled time
Symptom: A forecast run is missing on the day a local clock jumps forward.
Competing explanations: scheduler behavior, pipeline outage, or an invalid local timestamp.
Diagnostic: Inspect scheduler configuration and actual UTC execution history.
Expected evidence: the configured local time did not exist on that transition date.
Bounded response: replay the scheduler's actual skip or adjusted-run behavior.
Verification: check both the forecast deadline and the fallback used on the affected day.
Risk: inventing an on-time run produces an optimistic coverage estimate.
### F24. A product code is reused for a different item
Symptom: A new item inherits an unrelated product's historical demand.
Competing explanations: identifier reuse, hierarchy remapping, or a genuine assortment change.
Diagnostic: Inspect effective-dated product identity, packaging, and lifecycle records.
Expected evidence: one identifier maps to distinct product identities in different intervals.
Bounded response: use a stable entity key and explicit effective dates.
Verification: trace training histories across launches, replacements, and identifier reuse.
Risk: a current mapping applied retroactively can leak later product knowledge.
### F25. A pack-size change creates a step in units
Symptom: Unit forecast errors jump around a supplier packaging change.
Competing explanations: demand change, case-to-each conversion, or mixed units.
Diagnostic: Compare source units, conversion factors, and effective dates at each origin.
Expected evidence: an old case conversion is applied to a new pack or vice versa.
Bounded response: normalize with the conversion known and effective for the target decision.
Verification: reconcile sample forecasts and transactions in both cases and eaches.
Risk: correcting values with a later conversion without a vintage record may distort the replay.
### F26. An item launch is absent from the evaluation
Symptom: The model looks strong offline but requires manual handling for new items.
Competing explanations: history filters, missing launch records, or delayed entity creation.
Diagnostic: Compare live decision counts with eligible historical cohorts by product age.
Expected evidence: early lifecycle decisions are systematically omitted from the backtest.
Bounded response: include the legal cold-start fallback in policy-level scoring.
Verification: report coverage and loss for launch cohorts and the complete population.
Risk: scoring only established items estimates a different service than the one proposed.
### F27. Discontinued products continue to receive forecasts
Symptom: Forecast volume persists after an item is no longer orderable.
Competing explanations: stale lifecycle metadata, delayed discontinuation notices, or planned sell-through.
Diagnostic: Compare effective discontinuation, notice availability, and orderability at each origin.
Expected evidence: the issuing system used an obsolete lifecycle state after a legal update.
Bounded response: apply the effective as-of lifecycle policy and record final-order exceptions.
Verification: audit forecasts around the last valid order date, including late notices.
Risk: using a later discontinuation announcement in earlier folds is leakage.
### F28. A new store borrows future peer information
Symptom: Cold-start forecasts depend on a peer group defined after the launch.
Competing explanations: a current-store mapping, a legitimate prelaunch attribute, or code reuse.
Diagnostic: Identify the timestamp of each peer assignment and its input attributes.
Expected evidence: the historical forecast uses peer labels created with later performance data.
Bounded response: rebuild the peer rule using only prelaunch information.
Verification: reproduce each cohort assignment from its origin-time inputs.
Risk: a convenient current similarity table may encode future outcomes.
### F29. A calendar holiday is moved retrospectively
Symptom: Holiday features disagree between replayed predictions and archived runs.
Competing explanations: calendar revision, timezone conversion, or incorrect regional mapping.
Diagnostic: Compare calendar publication versions, locality, and effective holiday dates.
Expected evidence: the replay uses an announcement or correction issued after the origin.
Bounded response: select the calendar version known to the decision maker at the time.
Verification: inspect origin-level lineage for announced and unannounced schedule changes.
Risk: known future holidays are legal features, but later calendar revisions are not.
### F30. A training example contains a forward rolling label
Symptom: A model predicts unusually well immediately after a rolling target window starts.
Competing explanations: overlapping future labels, a join error, or stable demand.
Diagnostic: Compute each training label's latest component and actual availability timestamp.
Expected evidence: at least one label used in fitting was not complete by the fit cutoff.
Bounded response: exclude the label until every required component is available and mature.
Verification: assert label availability for every training row and every fold.
Risk: a chronological feature timestamp alone does not prove that the label is legal.
### F31. Actuals and predictions use different day boundaries
Symptom: Errors alternate signs between adjacent days but multi-day totals look plausible.
Competing explanations: timezone mismatch, delayed transactions, or true day-to-day shifts.
Diagnostic: Compare inclusive start and exclusive end instants for forecast and actual periods.
Expected evidence: the same transaction is assigned to different days by the two pipelines.
Bounded response: rebuild the matched target periods using a single declared business calendar.
Verification: trace transactions near midnight and daylight-saving boundaries.
Risk: aggregation can hide daily timing errors that matter for labor or shelf availability.
### F32. Forecast records are duplicated by retry
Symptom: Reported forecast counts and apparent sample size increase after an operational retry.
Competing explanations: duplicate inserts, legitimate refreshes, or a changed decision cadence.
Diagnostic: Compare run IDs, origin, target, model version, and action deadlines.
Expected evidence: identical predictions are inserted twice for the same idempotency key.
Bounded response: enforce an immutable decision key and distinguish refresh versions explicitly.
Verification: reconcile unique eligible decisions with forecast and fallback counts.
Risk: deduplication must not erase genuinely distinct decisions or authorized refreshes.
### F33. An overridden forecast is labeled as raw model output
Symptom: Offline model predictions cannot be reproduced from the deployed model artifact.
Competing explanations: planner override, postprocessing, or artifact mismatch.
Diagnostic: Inspect model output, override event, rounding, and final issued forecast separately.
Expected evidence: a timestamped override changes the issued value after model inference.
Bounded response: report raw model and final forecast metrics as distinct descriptive measures.
Verification: reconstruct each value from its own immutable event chain.
Risk: overridden cases are selected by planners, so differences do not prove override impact.
### F34. Forecast accuracy improves because a difficult slice vanished
Symptom: The overall metric improves while coverage decreases in a specific region.
Competing explanations: changing eligibility, a genuine quality gain, or an assortment shift.
Diagnostic: Compare eligible, predicted, fallback, and missing counts by region and origin.
Expected evidence: high-error decisions appear in baseline reporting but not candidate reporting.
Bounded response: score both policies on the shared eligible decision population.
Verification: publish a denominator bridge from all eligible decisions to scored outcomes.
Risk: a complete-case score may be useful diagnostically but is not the whole-policy score.
### F35. A model wins only after excluding stockouts
Symptom: A filtered accuracy score improves dramatically when unavailable periods are removed.
Competing explanations: censoring, missing inventory data, or selection of easier demand regimes.
Diagnostic: Compare availability, demand proxies, and excluded volume across models and periods.
Expected evidence: the excluded periods differ systematically in demand or operational conditions.
Bounded response: report the filtered estimand and a separate stockout-sensitive diagnostic.
Verification: audit availability timestamps and examine sensitivity to classification uncertainty.
Risk: uncensored sales periods are not necessarily representative of all customer demand.
### F36. An apparent service improvement follows a policy change
Symptom: Stockouts fall after a forecast release but order quantities also increase.
Competing explanations: inventory policy, seasonal demand, supplier reliability, or forecast quality.
Diagnostic: Compare order rules, lead times, inventory positions, and contemporaneous controls.
Expected evidence: multiple operational changes coincide with the forecast change.
Bounded response: avoid attributing the service change to the model without a causal design.
Verification: design a controlled trial or justified quasi-experiment where feasible.
Risk: historical forecast error alone cannot identify incremental business benefit.
### F37. Quantile coverage appears correct in aggregate
Symptom: Overall 90% interval coverage looks acceptable while peak-day coverage is poor.
Competing explanations: mixture cancellation, small slices, or a changed target distribution.
Diagnostic: Compute coverage and interval width by horizon, promotion, volume, and maturity.
Expected evidence: undercoverage in a decision-critical slice is offset by overcoverage elsewhere.
Bounded response: report slice counts and evaluate recalibration using only prior legal data.
Verification: assess a later protected period with the recalibration policy frozen.
Risk: many small slices produce noisy rates and invite post hoc selection.
### F38. Interval coverage is purchased with excessive width
Symptom: Nearly every target is covered but intervals are too broad for ordering decisions.
Competing explanations: overly conservative calibration, mixed horizons, or a scale defect.
Diagnostic: Compare interval widths, coverage, and decision loss on matched cells.
Expected evidence: widths grow substantially without a proportionate operational use.
Bounded response: evaluate sharpness subject to a predeclared calibration requirement.
Verification: compare horizon-specific coverage and width on a protected period.
Risk: optimizing width alone encourages intervals that miss too often.
### F39. Quantiles cross
Symptom: A predicted 90th percentile is lower than the predicted 50th percentile for the same cell.
Competing explanations: independent quantile fitting, incorrect level labels, or sorting defects.
Diagnostic: Compare all issued quantiles for identical series, origin, horizon, and target.
Expected evidence: forecast values decrease as quantile levels increase.
Bounded response: apply a declared monotonic correction or use a coherent quantile method.
Verification: assert monotonicity and rescore each quantile after the correction.
Risk: postprocessing can change marginal calibration and loss.
### F40. A lead-time quantile is made by adding daily quantiles
Symptom: A nominal high-quantile inventory target has unexpected total-demand coverage.
Competing explanations: dependence assumptions, quantile addition, or variable lead times.
Diagnostic: Compare the total distribution from joint paths with the sum of marginal quantiles.
Expected evidence: the calculation assumes a sum of marginal quantiles is the same quantile of the sum.
Bounded response: estimate or simulate joint demand paths and compute the total quantile directly.
Verification: score the issued total quantile against mature realized protection-period totals.
Risk: dependence estimation is uncertain, especially for sparse or changing series.
### F41. One large error dominates RMSE
Symptom: Candidate ranking changes sharply when one high-volume event enters the report.
Competing explanations: genuine rare-event failure, corrupted actual, or unit mismatch.
Diagnostic: Inspect the largest squared-error contributions and reconcile their source records.
Expected evidence: one cell contributes most of the total squared error.
Bounded response: correct documented data defects; otherwise retain the event and report its influence.
Verification: show MAE, RMSE, and sensitivity with the exceptional event clearly labeled.
Risk: removing valid extremes can hide the very risk RMSE is intended to surface.
### F42. MASE becomes undefined for constant demand
Symptom: Some series vanish from a macro MASE report or display extreme values.
Competing explanations: zero seasonal scale, missing training points, or a hidden epsilon.
Diagnostic: Inspect the training-only denominator and the count of valid seasonal differences.
Expected evidence: every legal seasonal difference is zero, producing a zero scale.
Bounded response: mark MASE undefined for those series and report absolute error and coverage.
Verification: reconcile the number of defined scores with all eligible series.
Risk: adding an arbitrary tiny denominator can dominate the aggregate without meaning.
### F43. A mean of WAPEs differs from pooled WAPE
Symptom: A dashboard's headline changes when regions are grouped differently.
Competing explanations: unweighted averaging of ratios, changed denominators, or missing regions.
Diagnostic: Reconstruct each region's absolute-error numerator and actual-volume denominator.
Expected evidence: the headline averages regional percentages instead of pooling their components.
Bounded response: publish both the explicitly weighted pooled score and any equal-region summary.
Verification: reconcile numerator and denominator totals before and after regrouping.
Risk: neither weighting scheme is inherently correct without a decision objective.
### F44. Model benefit reverses after fallback is included
Symptom: Candidate-only accuracy beats baseline but full-service accuracy does not.
Competing explanations: difficult missing cases, poor fallback, or different forecast deadlines.
Diagnostic: Join every eligible decision to candidate success, fallback, and baseline records.
Expected evidence: fallback errors on candidate failures outweigh the candidate-only gain.
Bounded response: improve or replace the fallback only under a new frozen policy and evaluation.
Verification: score the complete candidate-plus-fallback policy on the matched population.
Risk: removing failed runs makes the reported model impossible to operate as measured.
### F45. A model silently misses the ordering deadline
Symptom: Forecast accuracy looks good but planners frequently use yesterday's forecast.
Competing explanations: long computation, late feature feeds, or an incorrect deadline configuration.
Diagnostic: Compare forecast completion and availability times against each actual action cutoff.
Expected evidence: the prediction arrives too late to influence the intended order.
Bounded response: score the on-time fallback and report late-run frequency separately.
Verification: reconstruct which forecast version each decision could actually consume.
Risk: a technically accurate late forecast may have no value for that decision.
### F46. Forecasts are scored after the planner acted
Symptom: A later refresh is credited for an order already submitted.
Competing explanations: timestamp join error, post-order refresh, or undocumented order modification.
Diagnostic: Link prediction availability, submission, and any authorized amendment timestamps.
Expected evidence: the scored forecast was not available before the relevant action was locked.
Bounded response: select the last legally usable forecast for each specific action.
Verification: manually trace a sample of orders with refreshes on both sides of the deadline.
Risk: later forecasts are valid for later decisions, not retroactive evidence of earlier capability.
### F47. An origin is dropped because one horizon is missing
Symptom: Long-horizon evaluation includes fewer difficult decisions than short-horizon evaluation.
Competing explanations: immature actuals, selective filtering, or failed predictions.
Diagnostic: Tabulate eligibility and outcome maturity independently by origin and horizon.
Expected evidence: a complete-vector filter excludes otherwise scoreable early horizons.
Bounded response: report horizon-specific populations and a separate complete-vector cohort.
Verification: reconcile both cohorts with the intended downstream decision windows.
Risk: different populations make cross-horizon score comparisons descriptive rather than paired.
### F48. A cross-series model learns from a future launch
Symptom: Global-model backtests include product records that did not exist at early origins.
Competing explanations: current catalog joins, incorrect effective dates, or valid advance launch plans.
Diagnostic: Trace entity creation and source availability relative to every training origin.
Expected evidence: a training entity or its realized demand was introduced after the fit cutoff.
Bounded response: reconstruct the historical catalog and legal cross-series training population.
Verification: assert entity and label availability for all records used by each fit.
Risk: even a series not scored at an origin can leak information through global training.
### F49. Weather actuals replace historical weather forecasts
Symptom: Weather-sensitive demand is predicted unusually accurately in historical replay.
Competing explanations: realized-weather leakage, a strong legal weather forecast, or a target join error.
Diagnostic: Match each feature to weather forecast issuance time, location, and target interval.
Expected evidence: the feature is a later observed reading rather than an issued forecast vintage.
Bounded response: use archived weather forecasts available at each freeze.
Verification: inspect a sample around storms where forecast and realized weather differ materially.
Risk: missing forecast archives limit the credibility of the weather-feature comparison.
### F50. An external data revision leaks into old folds
Symptom: Historical accuracy changes after a macroeconomic series is revised.
Competing explanations: revised inputs, refitted preprocessing, or changed actuals.
Diagnostic: Compare the input release vintage and transformation artifact for each origin.
Expected evidence: the replay uses a revision published after the original cutoff.
Bounded response: join to the latest release actually available at each forecast origin.
Verification: perturb later releases and confirm earlier frozen predictions are unchanged.
Risk: event-period dates do not establish when an external series was published.
### F51. Label maturity differs by channel
Symptom: One sales channel shows systematically better recent accuracy than another.
Competing explanations: different correction delays, channel mix, or true model quality.
Diagnostic: Plot maturity age and correction rates by channel and target date.
Expected evidence: recent outcomes in one channel are provisional or selectively missing.
Bounded response: apply a channel-aware maturity policy and report provisional counts.
Verification: compare first-mature and later-restated metrics by channel.
Risk: equal calendar cutoffs need not produce equally mature outcomes.
### F52. The supplier changes delivery frequency
Symptom: Previously meaningful seven-day forecast scores no longer predict inventory exposure.
Competing explanations: a new delivery schedule, altered lead time, or demand shift.
Diagnostic: Compare historical order cadence, receipt dates, and protection periods.
Expected evidence: the period covered by each replenishment decision has changed.
Bounded response: define a new replay matching the current policy and retain the old report separately.
Verification: trace sample orders through the revised supplier calendar.
Risk: pooling incompatible policies can obscure both historical and current performance.
### F53. A lead-time slice uses future realization as a feature
Symptom: Forecast performance appears strongest when grouped by realized delivery delay.
Competing explanations: descriptive slicing, an illegal model feature, or a genuine association.
Diagnostic: Determine whether realized arrival time entered the forecast or only later analysis.
Expected evidence: the model consumes the actual arrival date before it was known.
Bounded response: remove the future value and use only origin-time lead-time information.
Verification: compare input lineage with supplier notices available at the freeze.
Risk: retrospective slices describe outcomes but do not justify prospective targeting.
### F54. Product substitution obscures demand
Symptom: An item spikes when a neighboring item is unavailable.
Competing explanations: substitution, common demand shocks, or a promotion.
Diagnostic: Align both items' availability, price, and sales at a consistent time grain.
Expected evidence: sales displacement coincides with unavailable substitutes, subject to confounding.
Bounded response: report availability regimes and avoid treating observed sales as isolated latent demand.
Verification: examine multiple comparable episodes or design a separate causal study.
Risk: correlation around stockouts alone does not identify substitution magnitude.
### F55. A planner changes the order but not the forecast
Symptom: Forecast errors are small while stock remains insufficient.
Competing explanations: constrained orders, manual reductions, supplier short shipments, or bad inventory records.
Diagnostic: Reconcile forecast, recommended order, submitted order, receipt, and inventory position.
Expected evidence: the submitted or received quantity differs from the recommendation.
Bounded response: attribute the discrepancy to the relevant stage without assuming model failure.
Verification: trace a sample from prediction through receipt and inventory update.
Risk: an accurate forecast cannot by itself overcome execution or capacity constraints.
### F56. Forecast intervals cover the wrong target
Symptom: Daily interval coverage is acceptable but lead-time service remains volatile.
Competing explanations: daily dependence, variable lead time, or policy constraints.
Diagnostic: Compare the interval target with the quantity actually used in the order decision.
Expected evidence: marginal daily intervals are substituted for a cumulative-demand distribution.
Bounded response: evaluate the cumulative target at the applicable protection horizon.
Verification: compare realized lead-time totals with the issued total-demand intervals.
Risk: daily coverage does not establish coverage for a sum of dependent outcomes.
### F57. A hierarchy mapping changes after the decision
Symptom: Historical category forecasts no longer sum to today's category totals.
Competing explanations: effective-dated reclassification, unit conversion, or missing items.
Diagnostic: Reaggregate using both the origin-time mapping and the current restated mapping.
Expected evidence: item membership differs between the two mapping vintages.
Bounded response: label as-issued and restated hierarchy reports separately.
Verification: check parent-child sums within each mapping and matched unit convention.
Risk: comparing scores across mappings without disclosure can manufacture a trend.
### F58. New model forecasts use a different target vintage
Symptom: One model appears better even though its forecast values are unchanged.
Competing explanations: actual-vintage mismatch, changed eligibility, or a scoring bug.
Diagnostic: Compare actual record IDs and scoring timestamps for every paired cell.
Expected evidence: models are joined to different actual versions for the same target.
Bounded response: rescore both on one declared vintage and publish the vintage separately.
Verification: assert identical actual IDs across the matched comparison.
Risk: restated actuals can legitimately support a second report, not an asymmetric comparison.
### F59. A ratio improves because demand increases
Symptom: WAPE falls while absolute forecast error in units stays constant.
Competing explanations: a larger denominator, true forecast improvement, or an assortment shift.
Diagnostic: Compare absolute-error numerator, actual-volume denominator, and population mix.
Expected evidence: volume rises while the numerator is unchanged or rises more slowly.
Bounded response: report unit errors and volume alongside WAPE.
Verification: recompute both periods under a common population where feasible.
Risk: an improved ratio is not necessarily fewer missed units or better service.
### F60. A model is selected using too many slice comparisons
Symptom: One segment shows a large apparent gain after many model and slice searches.
Competing explanations: chance selection, a genuine segment effect, or changed population rules.
Diagnostic: Review all tried configurations, slices, and the order in which results were inspected.
Expected evidence: the highlighted segment was chosen only after looking at many noisy outcomes.
Bounded response: treat the finding as exploratory and specify a prospective confirmation rule.
Verification: evaluate the frozen slice and model on a new protected period.
Risk: multiple comparisons can make a chance result look like a robust operational opportunity.
### Additional decision-time diagnostic exercises
These illustrative probes extend the failure analyses above; they are not reports of production incidents.
### Q16. A forecast arrives at 09:10 for an order submitted at 09:00. How should it be scored?
**Answer:** Do not credit it as the forecast used for that order; score the on-time version or fallback.
**Reasoning:** A prediction must be available before the action it is supposed to inform.
**Follow-up:** Retain the late forecast for diagnostics and report deadline coverage.
**Red flag:** Comparing a late candidate with an on-time baseline as though both were actionable.
### Q17. What if a label's event date precedes the origin but its correction arrives later?
**Answer:** Use only the label version legally available for training at that origin.
**Reasoning:** Event time does not establish knowledge of later corrections.
**Follow-up:** A separate restated evaluation may use the later actual if the policy declares that vintage.
**Red flag:** Assuming the current warehouse value was always known historically.
### Q18. How do you compare models when one fails on difficult products?
**Answer:** Score the candidate plus its operational fallback against the baseline on all eligible decisions.
**Reasoning:** Candidate-only error conditions on successful runs and can hide selection.
**Follow-up:** Report candidate coverage, fallback accuracy, and unserved decisions by product age.
**Red flag:** Dropping failures and presenting the resulting score as whole-service performance.
### Q19. How would you score a three-day order based on daily forecasts?
**Answer:** Sum the three forecasts issued at that order's origin and compare with the corresponding actual total.
**Reasoning:** The decision is exposed to cumulative demand over the protection window.
**Follow-up:** Report daily losses too when timing affects service or capacity.
**Red flag:** Replacing a day-three prediction with a later refresh that was unavailable at order time.
### Q20. Can the median of a three-day sum be obtained by adding daily medians?
**Answer:** Not in general; the joint distribution and dependence determine the sum's quantile.
**Reasoning:** Quantile operators do not distribute over addition.
**Follow-up:** Construct legal joint demand paths and evaluate the resulting total quantile.
**Red flag:** Treating daily marginal quantile addition as universally valid.
### Q21. What does a zero mean signed error tell you about MAE?
**Answer:** Nothing sufficient to determine it; positive and negative errors may cancel.
**Reasoning:** Signed error preserves direction while absolute error removes sign.
**Follow-up:** Errors +2 and -2 have mean signed error zero but MAE two units.
**Red flag:** Calling a zero-bias forecast perfect.
### Q22. Why is a MASE below one not proof of beating a test-set seasonal-naive forecast?
**Answer:** Its denominator is an in-sample training naive scale, not necessarily the actual held-out baseline error.
**Reasoning:** The scale and the matched out-of-sample loss are computed on different observations.
**Follow-up:** Also score a frozen legal seasonal-naive baseline on the same target cells.
**Red flag:** Interpreting the scaled ratio as a direct paired win rate.
### Q23. A constant training series has zero MASE scale. What should be reported?
**Answer:** Mark its MASE undefined and report unit error and the count of affected series.
**Reasoning:** Division by a zero scale has no ordinary finite interpretation.
**Follow-up:** Investigate whether constant demand is genuine or caused by data censoring.
**Red flag:** Adding an undisclosed epsilon and presenting the number as standard MASE.
### Q24. How do you choose between expanding and sliding training windows?
**Answer:** Replay both on the same origins and refit schedule, considering seasonal coverage and drift.
**Reasoning:** Sliding windows adapt faster but can discard useful rare events and increase variance.
**Follow-up:** Protect a later evaluation period from repeated window tuning.
**Red flag:** Declaring one window universally superior without a decision-time comparison.
### Q25. How does a gap differ from an availability-time join?
**Answer:** A gap removes a region of observations; an as-of join selects the version available by a specific cutoff.
**Reasoning:** Different sources can have different delays and revisions that a single gap cannot describe.
**Follow-up:** Use a conservative gap only when it covers the relevant delays and document its limits.
**Red flag:** Assuming a one-day gap fixes an unknown two-day source delay.
### Q26. How should a weekly-refitted model with daily inputs be replayed?
**Answer:** Refit on the historical weekly schedule and refresh inputs daily where the actual system does so.
**Reasoning:** Parameter updates and feature refreshes are distinct state transitions.
**Follow-up:** Record both artifact ID and feature snapshot for each daily decision.
**Red flag:** Refitting daily offline while describing the result as a weekly-refit system.
### Q27. What does it mean if WAPE drops while total absolute error is unchanged?
**Answer:** The denominator may have increased; fewer missed units have not been demonstrated.
**Reasoning:** WAPE is total absolute error divided by total actual volume.
**Follow-up:** Report numerator, denominator, and population composition for both periods.
**Red flag:** Calling the ratio change an operational saving without additional evidence.
### Q28. How do you interpret 20% WAPE versus 15% WAPE on the same population?
**Answer:** The decrease is five percentage points, or a 25% relative reduction.
**Reasoning:** Five divided by the baseline value of 20 is one quarter.
**Follow-up:** Verify that actual vintage and demand denominator match before comparison.
**Red flag:** Describing the difference as a five-percent relative reduction.
### Q29. Why might equal-series and pooled MAE disagree?
**Answer:** They assign different weights when series have different numbers of scored cells.
**Reasoning:** Pooled MAE weights observations equally; equal-series MAE first averages within series.
**Follow-up:** Show counts, uncertainty, and the intended business weighting.
**Red flag:** Calling one average wrong without specifying the estimand.
### Q30. What if the total actual demand for a slice is zero?
**Answer:** WAPE is undefined for that slice; report absolute-error units and the slice count.
**Reasoning:** Its denominator is zero even when forecast errors are positive.
**Follow-up:** Keep zero-actual cells in a larger valid aggregate's absolute-error numerator.
**Red flag:** Reporting zero WAPE or silently dropping the slice.
### Q31. How can a data correction change a historical score without changing a forecast?
**Answer:** The forecast can be joined to a revised actual vintage with a different value.
**Reasoning:** A score depends on both prediction and chosen outcome version.
**Follow-up:** Publish original and restated reports with their scoring timestamps.
**Red flag:** Attributing every score change to a new model.
### Q32. What is a defensible tie rule for a record timestamp equal to the cutoff?
**Answer:** Declare whether equality is allowed and apply the rule consistently to actual system availability.
**Reasoning:** A timestamp alone may not encode event ordering within its precision.
**Follow-up:** Investigate clock skew and use a deterministic version sequence when available.
**Red flag:** Changing the boundary rule after seeing model performance.
### Q33. Why can a model's daily intervals be calibrated but its lead-time interval fail?
**Answer:** Marginal daily calibration does not establish the joint dependence needed for a sum.
**Reasoning:** Correlated errors change the distribution of cumulative demand.
**Follow-up:** Evaluate total-demand coverage and width at the actual protection horizon.
**Red flag:** Claiming that daily 90% coverage guarantees 90% coverage for a multi-day total.
### Q34. How do you evaluate an interval that covers all observations?
**Answer:** Report its empirical coverage together with width, horizon, population, and decision utility.
**Reasoning:** An extremely wide interval can cover nearly everything while being operationally uninformative.
**Follow-up:** Compare sharpness subject to a predeclared calibration requirement.
**Red flag:** Treating coverage alone as the complete interval score.
### Q35. How does an override comparison become misleading?
**Answer:** Planners may override cases they already believe are unusually difficult.
**Reasoning:** The overridden and untouched groups can differ before the intervention.
**Follow-up:** Preserve raw forecasts, override reasons, timing, and final decisions; use a causal design for impact.
**Red flag:** Treating a raw difference in overridden cases as the causal effect of overrides.
### Q36. A single extreme observation changes the RMSE ranking. Should it be removed?
**Answer:** Investigate its provenance and retain it if valid unless a predeclared estimand excludes it.
**Reasoning:** Squared loss intentionally gives large misses greater weight.
**Follow-up:** Show influence and a clearly labeled sensitivity analysis alongside the primary score.
**Red flag:** Deleting an inconvenient valid event to improve the candidate ranking.
### Q37. What does a coherent forecast hierarchy guarantee?
**Answer:** Declared parent forecasts equal the sum of their mapped child forecasts under the same units and period.
**Reasoning:** Coherence is an arithmetic constraint, not an accuracy theorem.
**Follow-up:** Score both raw and reconciled forecasts on the appropriate mapping vintage.
**Red flag:** Assuming reconciliation improves accuracy at every node.
### Q38. Why can rolling-origin errors be dependent even when forecast rows differ?
**Answer:** Different rows may share an actual target, overlapping lead-time windows, or a common origin shock.
**Reasoning:** Row count is not necessarily the count of independent information units.
**Follow-up:** Consider paired block or cluster resampling with a justified dependence structure.
**Red flag:** Treating all forecast cells as independent by default.
### Q39. How many blocks are needed for a valid uncertainty interval?
**Answer:** No universal count guarantees validity; assess dependence, block length, and available independent periods.
**Reasoning:** A few highly overlapping blocks cannot support a precise estimate simply through repeated resampling.
**Follow-up:** Report sensitivity and state when data are insufficient for reliable uncertainty quantification.
**Red flag:** Presenting a narrow interval without the resampling unit or limitations.
### Q40. A holiday calendar is known months ahead. Is it legal as a future feature?
**Answer:** Yes, if that specific calendar version was available by the forecast freeze.
**Reasoning:** Future-dated information can be legal when genuinely known at decision time.
**Follow-up:** A later announced change must appear only in subsequent snapshots.
**Red flag:** Treating every future-dated feature as leakage or every current calendar value as historical truth.
### Q41. Why does accurate fulfilled-sales prediction not prove accurate latent-demand prediction?
**Answer:** Stockouts can prevent customer demand from appearing in observed sales.
**Reasoning:** The label is censored by inventory availability and execution.
**Follow-up:** Examine availability and lost-order evidence, and state assumptions for any demand bounds.
**Red flag:** Treating model-imputed lost demand as independently observed ground truth.
### Q42. How would you evaluate a launch with no item-level training history?
**Answer:** Replay the legal cold-start policy using attributes and peer mappings available at launch.
**Reasoning:** Filtering the item out estimates performance only for established products.
**Follow-up:** Report launch-cohort coverage and the fallback's contribution to full-policy accuracy.
**Red flag:** Using a similarity group defined later using realized launch performance.
### Q43. When is a higher forecast quantile economically appropriate?
**Answer:** When the actual shortage-versus-surplus trade-off supports it under the decision model's assumptions.
**Reasoning:** For simple linear one-period costs, the critical fractile is underage cost divided by total marginal cost.
**Follow-up:** Check capacity, fixed costs, lead-time uncertainty, and spoilage before applying that rule.
**Red flag:** Selecting an extreme percentile solely because shortage is undesirable.
### Q44. A baseline has zero MAE and the candidate has MAE one. What is the relative improvement?
**Answer:** It is undefined because the baseline denominator is zero; the candidate is one unit worse in absolute MAE.
**Reasoning:** A relative ratio cannot be computed by dividing by zero.
**Follow-up:** Give both underlying scores and their shared evaluation population.
**Red flag:** Reporting an invented infinite or zero percentage as an ordinary comparison.
### Q45. What changes when the supplier moves from weekly to daily deliveries?
**Answer:** The relevant order origins and protection windows may change, requiring a new decision contract and replay.
**Reasoning:** Accuracy at an obsolete horizon may not measure the new exposure.
**Follow-up:** Keep old-policy results labeled separately and assess the new schedule on legal data.
**Red flag:** Pooling incompatible decisions into one unqualified headline.
### Q46. What would establish that a forecast release caused a stockout reduction?
**Answer:** A suitable controlled or defensible causal evaluation of the forecasting policy, not offline error alone.
**Reasoning:** Inventory, supplier performance, promotions, and demand can change at the same time.
**Follow-up:** Prespecify assignment, outcomes, spillovers, and operational guardrails for a feasible trial.
**Red flag:** Attributing a before-and-after service change solely to the forecast release.
### Q47. Which event should trigger a revisit of the evaluation contract?
**Answer:** A material change in decision cadence, feature availability, target definition, or action policy.
**Reasoning:** Each change can alter the relevant information set, population, or loss.
**Follow-up:** Version the contract and preserve comparability limits across reports.
**Red flag:** Assuming a historic leaderboard remains valid after a process redesign.
### Q48. What can you conclude from a two-origin worked example with a 57% gain?
**Answer:** Only that the specified candidate wins by that arithmetic on the illustrative cells.
**Reasoning:** Two dependent origins are too few to establish a reliable future or business effect.
**Follow-up:** Gather more relevant periods, quantify uncertainty where defensible, and protect a later evaluation.
**Red flag:** Presenting the toy percentage as a measured deployment result.
**Takeaway:** Diagnose the decision-time contract and population before attributing a forecast score change to the model.
## 9. Complexity & Cost Analysis
Let N be the number of series, K the number of origins, H the number of horizons, and P the number of features per series-origin-horizon cell.
Materializing all feature cells can require order N*K*H*P values; materializing just forecasts and actuals requires order N*K*H values, ignoring metadata and compression.
These are storage-growth descriptions, not benchmark claims.
Actual bytes depend on encoding, sparse values, lineage, indexes, retention, and platform overhead.
If training a model on M records with a specified algorithm costs C(M), refitting at every origin costs approximately the sum of C(M_o) across origins, plus feature construction and scoring.
For an expanding window, M_o generally grows with origin.
Reusing a model where production would refit changes the policy being evaluated; it is not merely an optimization.
Conversely, refitting every origin when production refreshes monthly measures a more expensive and potentially more informed system than the one being proposed.
Feature lookup cost depends on the index and join implementation.
No universal database complexity or runtime is asserted here.
ILLUSTRATIVE sizing: N=10,000 series, K=52 weekly origins, H=14 daily horizons, and P=20 numeric features.
The number of forecast cells is 10,000*52*14 = 7,280,000.
The number of feature values is 145,600,000.
At eight bytes per numeric value, raw feature payload is 1,164,800,000 bytes, approximately 1.085 GiB using 1 GiB = 1,073,741,824 bytes.
This excludes keys, validity flags, data-frame overhead, indexes, duplicate snapshots, metadata, compression, and intermediate copies.
It is not a capacity recommendation.
A streaming scorer can aggregate sufficient error sums without retaining every feature cell in memory, provided lineage and prediction records remain auditable elsewhere.
If every origin fits an independent model, parallel execution can reduce elapsed time but increases simultaneous CPU, memory, and storage pressure.
Set concurrency from measured resource limits.
For a global cross-series model, the training population and entity lifecycle must be reconstructed at each origin; a series introduced later cannot be used earlier just because its current identifier exists.
Partition outputs by origin and version; make retries idempotent so a repeated scoring task does not create duplicate forecast cells.
A missing fold should be visible as incomplete coverage, not interpreted as a zero-error fold.
Compare costs and accuracy using the same completed population.
Monitor data freshness and scoring latency separately from forecast loss.
Error is unavailable until labels mature, but missing or stale input signals are available sooner.
**Takeaway:** Estimate replay cost from cells and actual refit cadence, then measure platform overhead before sizing a production workload.
## 10. Key Theorems / Results Reference
**DERIVED — median and absolute loss:** A conditional median minimizes expected absolute error when the expectation exists.
It need not minimize squared loss or asymmetric shortage cost.
**DERIVED — mean and squared loss:** The conditional mean minimizes expected squared error when the second moment is finite.
The result follows by expanding the square around the conditional mean; the cross term vanishes.
**DERIVED — quantile and pinball loss:** For a chosen level strictly between zero and one, a conditional quantile minimizes expected pinball loss under integrability.
Discrete distributions can have multiple minimizing quantiles.
**DERIVED — aggregation and cancellation:** The absolute error of a sum is at most the sum of absolute errors by the triangle inequality.
Equality need not hold when daily errors have opposite signs.
**DERIVED — no chronological guarantee:** A time-ordered split does not imply feature availability.
A counterexample is a revised prior-day feature published only after the forecast origin.
**OPEN — future generalization:** Historical backtest scores alone do not guarantee performance after policy, regime, assortment, or measurement changes.
Establish a monitoring and validation plan for the actual deployment context.
**SOURCED — training-only preprocessing:** The scikit-learn documentation describes data leakage from fitting preprocessing on test data and recommends separating training and test transformations: https://scikit-learn.org/stable/common_pitfalls.html .
The course master records consultation on 2026-11-06; the unpinned stable page may change, so recheck before publication.
This supports the general preprocessing principle, not any illustrative result in this sheet.
**OPEN — empirical outcomes:** No production forecast, service-level improvement, inventory saving, benchmark, or source-specific deployment has been measured or asserted here.
**Takeaway:** Mathematical loss-optimality and past replay accuracy depend on different assumptions; neither guarantees business impact.
## 11. Interview / Exam Questions
### Q1. What is an origin, and why does it matter?
**Answer:** It is the feature-information cutoff for a historical decision.
It determines which data and model state were available for the forecast.
**Reasoning:** The same target date may have forecasts at multiple origins with different information.
Merging them hides lead-time effects.
**Follow-up:** If the model runs at noon but ordering occurs at 5 p.m., use the actual noon feature freeze unless a later refresh really occurs.
**Red flag:** Defining the origin after seeing the target or using event time as a proxy for publication time.
### Q2. A fold trains through day 10 and predicts days 11–13. What changes at the next daily origin?
**Answer:** If day 11 actuals are available by the next freeze, the expanding training window can include day 11 and the targets become days 12–14.
**Reasoning:** The horizon rolls with the decision time; availability, not just calendar order, governs training eligibility.
**Follow-up:** If day 11 arrives late, exclude it or use the same fallback that production uses.
**Red flag:** Automatically using day 11 regardless of reporting delay.
### Q3. Calculate WAPE for actuals 0, 10, 30 and forecasts 2, 8, 36.
**Answer:** The absolute errors are 2, 2, and 6; WAPE is 100*10/40 = 25%.
**Reasoning:** E3 includes the error on the zero-actual cell in the numerator, while the positive aggregate actual supplies a valid denominator.
**Follow-up:** If all actuals were zero, WAPE would be undefined and absolute error should be reported instead.
**Red flag:** Dropping the zero-demand cell because individual percentage error is undefined.
### Q4. Why can a random split be misleading?
**Answer:** It can train on observations later than the simulated forecast, leak temporal structure, and create an easier information set than operations.
**Reasoning:** The valid experiment must reproduce the intended decision's available data and update schedule.
**Follow-up:** A random split may be defensible only for an explicitly different exchangeable prediction task with a justified information boundary.
**Red flag:** Claiming that random splitting is universally invalid for every imaginable dataset without discussing the task.
### Q5. A candidate improves WAPE from 20% to 15%. How do you describe the change?
**Answer:** It improves by five percentage points and reduces WAPE by 25% relative to the 20% baseline.
**Reasoning:** The relative change is 5/20, while the absolute percentage-point change is 20-15.
**Follow-up:** First confirm that both scores use the same forecast cells, target vintage, and denominator.
**Red flag:** Calling a five-point difference a five-percent relative reduction without qualification.
### Q6. How would you detect a promotion-plan leak?
**Answer:** Check the publication and ingestion timestamps of the exact plan version used by each origin and compare it with the feature freeze.
**Reasoning:** A plan finalized after the forecast could differ from the plan actually known at decision time.
**Follow-up:** Replay with archived plan vintages, or state the missing-vintage limitation if the source overwrote history.
**Red flag:** Using the realized promotion merely because it describes the forecast target date.
### Q7. Why might a lower sales MAE fail to improve availability?
**Answer:** Sales can be censored by stockouts and availability also depends on replenishment, allocation, and lead-time execution.
**Reasoning:** A model can predict constrained sales accurately while missing the latent customer demand that would have occurred with stock.
**Follow-up:** Examine availability intervals, lost-order evidence, lead-time demand, and service outcomes under a controlled evaluation.
**Red flag:** Declaring a causal service improvement from offline sales MAE alone.
### Q8. Calculate the MASE scale for training values 2, 4, 3, 6 with seasonal period one. What if the forecast is 8 and actual is 5?
**Answer:** The scale is (2+1+3)/3 = 2 units, and the absolute error 3 divided by 2 gives MASE 1.5.
**Reasoning:** The denominator uses only adjacent training observations, not future actuals or the test fold.
**Follow-up:** Constant training values produce zero scale, so this MASE is undefined.
**Red flag:** Adding a hidden epsilon and presenting the result as directly comparable without disclosing the modification.
### Q9. How do overlapping forecast horizons affect uncertainty?
**Answer:** Errors can share realized outcomes, origin shocks, and adjacent time effects; they should not automatically be treated as independent observations.
**Reasoning:** A larger count of cells does not necessarily mean the same number of independent pieces of information.
**Follow-up:** Use an appropriate paired block or cluster resampling strategy, report the resampling unit, and examine sensitivity to block length.
**Red flag:** Computing a narrow independent-row interval while ignoring shared target dates.
### Q10. Choose a loss for costly shortages and inexpensive surplus.
**Answer:** Consider pinball loss at a quantile derived from explicit underage and overage costs, subject to the real inventory model and constraints.
**Reasoning:** Higher quantiles penalize underprediction more heavily than overprediction; the right level depends on the decision economics.
**Follow-up:** Check calibration, lead-time aggregation, and operational service in addition to average pinball loss.
**Red flag:** Selecting the 99th percentile simply because more inventory feels safer, without considering spoilage or capital cost.
### Q11. When does an expanding window lose to a sliding one?
**Answer:** When old data are less representative of the current regime and the sliding window retains enough signal to estimate the model reliably.
**Reasoning:** A smaller window can adapt faster but increases estimation variance and can omit annual seasonality or rare events.
**Follow-up:** Compare window strategies on the same historical origins and validate under the intended refit cadence.
**Red flag:** Presenting either window as universally superior.
### Q12. What would you do if the final holdout was used repeatedly for tuning?
**Answer:** Treat it as validation rather than an untouched final test, freeze the selection procedure, and obtain a later protected period or forward evaluation.
**Reasoning:** Repeated decisions based on the same holdout adapt the model to its noise.
**Follow-up:** Report the tuning history, remaining uncertainty, and operational constraints if waiting for new data is costly.
**Red flag:** Relabeling the same repeatedly inspected data as a fresh holdout.
### Q13. Two models have equal daily MAE but different lead-time-total errors. Which should be chosen?
**Answer:** The one better aligned with the actual replenishment loss may be preferred, subject to daily service and capacity guardrails.
**Reasoning:** Daily over- and under-errors can cancel when summed; an order for a multi-day window is exposed to the cumulative error.
**Follow-up:** Evaluate the cumulative distribution or quantiles if shortage risk depends on uncertainty rather than only the point total.
**Red flag:** Assuming that equal daily MAE implies equal inventory impact.
### Q14. What must be stored to reproduce a historical score?
**Answer:** Immutable forecast IDs and values, model and feature versions, origin and target timestamps, availability lineage, actual vintage, eligibility, metric implementation, and hierarchy mapping.
**Reasoning:** A score is a join between prediction and outcome under a defined population and policy; changing any component can change the result.
**Follow-up:** Record missing data and fallback decisions, not merely successful predictions.
**Red flag:** Retaining only a dashboard aggregate and current warehouse tables.
### Q15. A source publishes at 08:55, arrives at 09:05, and the freeze is 09:00. Is it legal?
**Answer:** No, not for a system that requires ingestion before the freeze.
**Reasoning:** Publication and internal availability are separate clocks, and the latter is after the cutoff.
**Follow-up:** Use the latest fully available earlier version or the declared fallback.
**Red flag:** Treating source publication as proof of serving availability.
**Takeaway:** Strong answers name the decision, legal information, denominator, failure boundary, and evidence needed for the conclusion.
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
