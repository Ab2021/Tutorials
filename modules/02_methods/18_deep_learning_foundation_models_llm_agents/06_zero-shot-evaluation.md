# Evaluating Zero-Shot and Fine-Tuned Models

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: arithmetic and basic spreadsheet skills; forecasting and modeling concepts are introduced here  
> Version 2026.5

## 1. Why this topic exists

A retailer replenishes stores every week and must decide whether to retain a
simple forecasting system or buy a pretrained time-series model.
A vendor reports a strong public benchmark, but that score does not state
whether the model reduces this retailer's lost sales, inventory, planner
workload, or total operating cost.
A local model that is less fashionable may be cheaper, easier to explain, and
more accurate for an important replenishment horizon.
The decision is not which architecture wins a leaderboard; it is which
forecast-and-decision process earns the right to operate.

A time series is a sequence of observations indexed by time, such as units
sold per stock keeping unit (SKU) per store per week.
Demand is what customers would have bought if supply had been available;
observed sales can be lower when stock is unavailable.
A forecast is an estimate of a future quantity, not an instruction to order.
A plan is a feasible choice of actions; a target is a desired result; a
commitment is an approved promise to another party.
Forecast improvements matter only if they change feasible actions or reduce
the effort and risk of producing those actions.

A foundation model is pretrained on many series and reused for a new series.
Zero-shot evaluation supplies past target observations as context but makes no
task-specific parameter update before prediction.
Fine-tuning updates model weights using examples from the target environment.
The fair question is whether either option adds value relative to feasible
local baselines under the same information, horizons, and business
constraints.
A separately fitted calibration layer is local training even when the base
model is frozen.

This unit builds a decision-focused evaluation without assuming completion of
Module 06.
Its fictional example separates statistical error, decision cost,
model-compute expense, and uncertainty.
The procedure can reject a candidate, approve a controlled pilot, or leave the
decision unresolved.
Neither the existence of a pretrained model nor a single numerical improvement
establishes a causal inventory saving.

## 2. Learning outcomes

After completing this unit, you can:

1. Distinguish frozen zero-shot inference, locally fitted calibration, and weight fine-tuning.
   Exercise 1 checks the boundary.
2. Construct an as-of rolling-origin experiment and specify protected training, validation, and test periods.
   Exercise 2 checks its timeline.
3. Compute horizon-specific absolute error, weighted absolute percentage error, and bias against a seasonal-naive baseline.
   Exercise 3 checks the arithmetic.
4. Translate forecast errors into explicit overage and underage cost under a stated order rule.
   Exercise 4 checks the business calculation.
5. Price local and hosted forecasts on a comparable annual volume and distinguish one-off from recurring cost.
   Exercise 5 checks the cost comparison.
6. Diagnose a misleading aggregate result and explain model-ranking uncertainty.
   Exercise 6 asks you to critique a proposed launch.
7. Design a leakage-safe fine-tuning and controlled deployment protocol.
   Exercises 7 and 8 assess design choices.

The outcomes require you to show inputs and units, not merely name a metric.
A correct business recommendation can be to retain a baseline while collecting
more evidence.

## 3. The core mental model

The model is one component in a timed decision system.
Think of each historical forecast as a sealed envelope: what was knowable when
it was made goes in, and observations after the origin are used only for
scoring.
The same envelope is handed to each candidate subject to its declared input
interface.

```text
As-of data snapshot at origin t
        | past target, available calendar and plans
        +----------------+------------------+
        |                |                  |
local baseline     frozen model        fine-tuned model
        |                |             trained before t
        +----------------+------------------+
                         |
              forecasts for t+1 ... t+H
                         |
             common decision rule and constraints
                         |
         actual demand proxy, service, inventory, cost
                         |
        horizon/segment comparison and uncertainty
                         |
        go / controlled pilot / retain baseline
```

Each arrow has a time, ownership, and evidence boundary.
The frozen model may use recent target observations without changing its
weights.
A fine-tuned model learns target-specific parameters; its training labels must
already exist at its training cutoff.
The baseline may also be fitted, but it must have the same as-of cutoff and
comparable allowed inputs.

A horizon is the distance from the forecast origin to the predicted bucket.
One-week and four-week horizons can behave differently because intervening
events and uncertainty accumulate.
For a four-week replenishment lead time, the quantity needed may be total
demand across four weeks, not the forecast of week four alone.
Summing four point forecasts does not by itself yield the distribution of
their total: errors across weeks can be correlated.

Evaluation has three layers.
First, a forecast score describes agreement with observed outcomes.
Second, a decision simulation applies an explicit common operating policy to
the forecast.
Third, an economic comparison subtracts the incremental cost and risk of the
full solution.
An improvement in the first layer is evidence about the second, not proof of
the third.

A backtest is a replay on historical dates, not a randomized deployment.
It can be biased by missing data, selection, pretrained data exposure, and
policies that changed what was observed.
A future shadow or controlled pilot supplies additional evidence but does not
erase flaws in the original replay.

## 4. Definitions and vocabulary

| Term | Operational definition and unit | Frequent misuse |
|---|---|---|
| Series | Ordered observations for a stable item-location identifier, units/week here | Combining unlike products without explaining the aggregation |
| Origin | Last time with information available for a particular forecast run | Treating the last event date as the data availability date |
| Horizon | Number of future buckets after origin, weeks here | Mixing week-four demand with demand summed over four weeks |
| Target | Quantity predicted, such as unconstrained units demanded | Calling observed sales true demand during stockout |
| Context | Past target values and allowed covariates supplied at inference | Assuming context use necessarily updates weights |
| Covariate | Input other than target, such as a published holiday flag | Using a future promotion outcome as a known plan |
| As-of snapshot | Values and revisions actually available at a decision time | Using today's corrected history in every historical run |
| Leakage | Information unavailable at prediction time entering fitting, features, or model selection | Restricting leakage to obviously future target values |
| Pretraining | Earlier learning of shared weights across a corpus | Assuming the corpus is unrelated to the test set |
| Frozen zero-shot | No target-task parameter update before inference | Calling a fitted target-specific output head zero-shot |
| Fine-tuning | Updating pretrained weights on target-related examples | Equating inference on context with gradient updates |
| Calibration | Adjustment of outputs using observed local outcomes | Treating a fitted calibrator as a pure zero-shot system |
| Local model | Separate parameters estimated for an individual series | Assuming it always uses only one past observation |
| Global model | Shared parameters learned across multiple series | Assuming it cannot use series-specific features |
| Naive forecast | Forecast equal to the latest observed value | Calling it a seasonal benchmark |
| Seasonal naive | Repeats the most recent observed value from the matching season | Accessing future seasonal values for long horizons |
| Rolling origin | Multiple historical forecast issuance dates evaluated forward | Re-randomizing rows from adjacent weeks |
| Training set | Data used to estimate weights or local parameters | Repeatedly changing it after inspecting test outcomes |
| Validation set | Data used for selection, tuning, and policy choices | Reporting its best score as independent final evidence |
| Test set | Protected period used for final comparison after choices are fixed | Tuning the winner on it and calling it untouched |
| Absolute error | Absolute value of actual minus forecast, units | Confusing a sum of absolute errors with its mean |
| MAE | Mean absolute error, units per observation | Comparing values across differently scaled series without context |
| WAPE | Sum absolute error divided by sum actuals, dimensionless or percent | Averaging row-wise percentages and calling the result WAPE |
| Bias | Mean forecast minus actual, units per observation here | Failing to state the sign convention |
| Quantile | Value with specified modeled cumulative probability at or below it | Calling a 90th percentile a 90 percent guarantee |
| Coverage | Fraction of actuals inside an interval, percent | Reporting coverage without interval width |
| Pinball loss | Asymmetric score for a forecast quantile, target units | Evaluating a quantile only by point MAE |
| Underage cost | Marginal cost per unit of unmet demand in a chosen decision model | Assuming all unmet sales have the same lifetime value |
| Overage cost | Marginal cost per excess unit in the chosen period | Counting the full acquisition price as a loss when stock remains usable |
| Forecast request | One invocation of an inference interface | Assuming one request always contains one series or one horizon |
| Series-origin | A particular series at one issuance time | Counting correlated horizon rows as independent customers |
| TCO | Total cost of ownership, currency over a stated period | Comparing only API price with total internal staff cost |
| Shadow mode | Producing forecasts without allowing them to change live decisions | Treating offline shadow scores as realized financial savings |
| Contamination | Evaluation overlap or exposure that compromises the intended independence claim | Claiming all public data exposure proves direct memorization |

## 5. The mechanics, step by step

### Step 1: Specify the decision before selecting a model

**Input:** the replenishment process, decision timing, product scope, and business owner.
**Output:** a written decision contract that fixes the outcome, forecast usage, operating constraints, and evaluation population.

Write down who will act, what they can change, and when the action becomes
irreversible.
For example, a planner may release an order every Monday for a four-week
replenishment lead time.
A forecast for a distant week cannot repair a purchase order already committed
to the supplier.
The relevant output may be aggregate lead-time demand plus a protection
period, rather than every individual weekly prediction.

Separate the forecast from the decision policy.
The forecast describes a future distribution or central estimate; the policy
maps it and current inventory to an order quantity.
The feasible order is then adjusted for pack sizes, minimum order quantities,
capacity, supplier calendars, and service constraints.
A target service level is not itself a forecast, and a promised delivery
quantity is not established by a model output.

Define exclusions before model scores are visible.
Discontinued products, returns, new launches, stockout weeks, and exceptional
outages each need a reasoned rule.
Do not remove hard cases solely because the candidate performs poorly.
Publish counts and sales share for eligible and excluded observations.
If the intended deployment includes new products, report a cold-start cohort
rather than quietly dropping it.

### Step 2: Define truth and its limitations

**Input:** transactions, inventory, item master, calendar, and relevant corrections.
**Output:** a versioned target table with availability times and a record of target-quality limitations.

Observed sales in units per week may be a suitable target for a sales planning
task.
For replenishment, sales during unavailable periods can censor latent demand:
an empty shelf cannot reveal every customer who would have purchased.
If reliable unconstrained-demand estimates exist, explain their estimation and
uncertainty.
Otherwise score against sales as a proxy, flag stockout exposure, and avoid
claiming measured lost demand.

Identify the observation date and the date the observation became available to
the forecasting system.
A corrected transaction may describe an earlier sale but arrive after the
forecast origin.
As-of evaluation uses the value available at that origin, not a corrected
value that became known later.
Final audited outcomes may be used for scoring if their role and revision
policy are fixed in advance.

Check units, product identifiers, location changes, launch dates, and
time-zone or retail-calendar boundaries.
A pack-to-each conversion error can look like a model failure.
Returns can make net sales negative; decide whether the modeled target should
be gross sales, net sales, or another quantity.
Record missing weeks separately from verified zero-sales weeks.

### Step 3: Fix the evaluation timeline

**Input:** historical coverage, model availability evidence, and operational run cadence.
**Output:** a dated training, validation, and final test design plus a replay schedule.

For illustration, train through week 100, select settings on origins in weeks
101–120, and test on later origins in weeks 121–140.
If four future weeks are scored, ensure the last validation label is available
before final settings are frozen.
A label for origin 120 at horizon 4 arrives in week 124, not week 120.
Either delay the first test origin until those labels are available or leave a
gap that satisfies the actual data latency.
Overlapping forecast windows are allowed when they reflect real operations;
they are not independent samples.

A rolling-origin backtest repeats the forecast at multiple origins and looks
forward from each one.
At each origin, all models receive only information available by that origin.
Use the same eligible series-origin-horizon cells for the primary paired
comparison.
A model's missing or failed forecasts must be reported and handled by the
predeclared fallback policy.
Scoring only its successful outputs creates selection bias.

A pretrained model released after a historical origin creates a counterfactual
issue.
A retrospective test can still assess the model on old data, but cannot
literally claim the model was available at that origin.
Record release date, corpus cutoffs if known, data provenance, version, and
any known benchmark overlap.
When corpus disclosure is incomplete, the degree of pretraining contamination
is unknown, not automatically zero or proven.
A prospective holdout collected after a fixed model release strengthens the
evidence, although it does not solve all distribution-shift problems.

### Step 4: Establish useful local baselines

**Input:** as-of target histories and the agreed seasonal calendar.
**Output:** executable naive, seasonal-naive, and, where warranted, fitted local benchmark forecasts.

The naive forecast repeats the latest available observed value at all
horizons.
It is a strong minimal benchmark when demand levels move gradually and
seasonality is weak.
The seasonal-naive forecast repeats the most recent observed matching season.
For weekly data with a 52-week season, week t+h uses the most recent available
value corresponding to that position in the annual cycle.
If horizon exceeds a full season, repeat the observed seasonal cycle rather
than reading an unobserved future target.

A local moving average, exponential smoothing model, or regularized local
regression can be an additional comparator if feasible in production.
Fit each using only data available at the same cutoff; tune hyperparameters on
the same validation budget.
A seasonal baseline is not meaningful for a product with less than a season of
history unless a fallback is explicitly defined.
Declare that fallback and count its use.

The baseline must include the actual incumbent process if that process is
feasible and measurable.
A historical manual override may contain future knowledge if its timestamp is
unavailable; do not silently use a final revised plan as an as-of forecast.
Compare candidate-only and incumbent-with-overrides only if their information
and labor differences are made explicit.
Evaluate both the forecast engine and the complete operational workflow where
the decision requires it.

### Step 5: Specify frozen and adapted candidates

**Input:** model versions, licenses, context limits, compute requirements, and feature contracts.
**Output:** reproducible candidate configurations with a clear adaptation label.

Frozen zero-shot means the target-task weights remain unchanged before each
prediction.
The history passed in the context may differ at each origin and need not be
empty.
Inference-time normalization computed solely from available context is
preprocessing; a normalization learned from all target data is local fitting.
If a validation-trained output correction is used, label the system frozen
base plus locally fitted calibrator.

For fine-tuning, define which weights change, the training window, objective,
number of updates, checkpoint rule, and retraining schedule.
Use only examples whose inputs and labels are available by the relevant
training cutoff.
A sliding-window training example must not contain a label beyond the cutoff
even if its input window starts earlier.
Tune learning rate, training duration, and model selection on validation data,
not the final test period.

A static fine-tuned model can be trained once and evaluated at later origins
using new contexts.
A rolling retrained model updates on an agreed schedule, with all training and
label latency represented.
Both are valid, but they answer different cost and responsiveness questions.
Count retraining, failed runs, and version approvals in the operational
comparison.

If one candidate receives future promotion plans, ask whether those plans were
actually known at each origin.
Known future calendar features differ from future realized prices or realized
weather.
An input advantage may be a legitimate system capability, but then call the
comparison a system comparison rather than an architecture-only comparison.
Where interfaces permit, also compare on a matched input set.

### Step 6: Run a reproducible replay

**Input:** immutable snapshots, candidate configurations, test origins, and scoring cells.
**Output:** a forecast ledger with one row per model, series, origin, and horizon.

Store the issued timestamp, model and code version, input-data version, target
period, forecast, and any distributional output.
Store failure status and fallback output as well as successful predictions.
Do not replace a failed output with an actual value or omit the corresponding
decision.
Track training seed, hardware or service region where meaningful, batch size,
and inference settings.
For stochastic models, state whether outputs are sampled and how many samples
support a reported quantile.

For each origin, reconstruct the as-of input.
Generate baseline and candidate forecasts without allowing the scorer to
change the model.
Apply the same target definitions and nonnegativity or rounding rules when
appropriate.
If constraints differ by system, record them as part of the system and show
their influence separately.
Save predictions before joining final outcomes.

If a hosted model fails or exceeds a deadline, run the declared fallback.
A model with excellent conditional accuracy and frequent timeouts may be
inferior as a complete service.
Report both conditional model accuracy and delivered-system accuracy.
The denominator of the delivered-system score includes fallback cells.

### Step 7: Score by horizon and segment

**Input:** matched forecast and outcome ledger, coverage flags, and business segment definitions.
**Output:** a scorecard by model, horizon, segment, and decision-relevant aggregate.

For each cell, define error as forecast minus actual.
Absolute error is the magnitude of that error; mean absolute error (MAE)
averages magnitudes across cells.
Bias is the arithmetic mean of signed errors using the stated convention.
Positive bias therefore indicates average overforecasting.
Bias can cancel across products, so inspect its distribution as well as its
overall mean.

Weighted absolute percentage error (WAPE) is sum absolute error divided by sum
actuals, multiplied by 100 percent.
It is undefined when the sum of actuals is zero and can be unstable when that
sum is very small.
Do not compute the mean of individual percentage errors and call it WAPE.
WAPE implicitly emphasizes high-volume observations, not necessarily
high-margin or high-risk ones.
Report numerator and denominator in units so another analyst can reconstruct
the result.

Report MAE and WAPE at each relevant horizon.
Also score lead-time demand by summing forecast and actual weekly values
within each series-origin before taking absolute error.
The absolute error of a sum is generally not the sum of weekly absolute
errors, because weekly errors can offset.
The operational interpretation depends on whether those offsets are usable
under inventory and timing constraints.

Segment by demand volume, intermittent demand, product age, lead time,
promotion exposure, and business criticality when sample size supports it.
Define segments before examining scores; otherwise treat them as exploratory.
A high-volume aggregate improvement can conceal worse performance for critical
low-volume items.
Report cohort size and demand volume next to each score.
Do not present a tiny cohort's apparent ranking as precise evidence.

### Step 8: Evaluate probabilistic output

**Input:** forecast quantiles or sample paths and the same held-out outcomes.
**Output:** proper distributional scores, calibration diagnostics, and decision-relevant uncertainty measures.

A p-quantile q means the model assigns probability p to outcomes at or below
q, subject to discrete-value conventions.
For actual y and error u = y - q, pinball loss is p times u when u is
nonnegative and (p - 1) times u when u is negative.
For p = 0.9, underprediction is penalized nine times as much per unit as
overprediction.
A model should not be penalized for a high quantile being above the actual as
though it were a median forecast.
Compare equal quantiles at the same horizon and on the same rows.

For an 80 percent interval, report empirical coverage and average width in
units.
Very wide intervals can have high coverage but provide little decision value.
Check coverage separately by horizon and sufficiently large segment, with
uncertainty around observed coverage.
For discrete demand, exact nominal coverage need not be attainable without
randomization.

For a multiweek order, use a predictive distribution of the sum over lead time
when that is the decision target.
Do not add weekly 90th-percentile forecasts and label the result the 90th
percentile of the sum.
Joint sample paths or an explicitly modeled dependence structure can represent
cross-week correlation.
Marginal quantiles alone are insufficient to recover the joint sum
distribution in general.

### Step 9: Translate forecasts into decisions

**Input:** forecasts, inventory state, operating constraints, and an agreed cost model.
**Output:** simulated orders, service outcomes, and incremental decision cost with a sensitivity range.

A simple one-period illustration orders a quantity Q and compares it with
demand D.
Underage is max(D - Q, 0); overage is max(Q - D, 0).
Cost is underage times the underage cost per unit plus overage times the
overage cost per unit.
This is a simplified decision model, not an accounting statement or a full
multi-period inventory simulator.
Specify whether unmet demand is lost or backordered and whether excess
inventory retains value.

With known per-unit underage and overage costs, the classical single-period
critical fractile is Cu/(Cu+Co).
Choose the corresponding demand quantile only under that model's assumptions
and a credible demand distribution.
Fixed order costs, capacity, multi-period carryover, and service constraints
can change the optimal policy.
A point forecast equal to expected demand is not automatically the correct
order quantity.

When comparing systems, keep the policy and initial inventory identical unless
the explicit research question is a whole-system redesign.
Represent pack rounding, lead time, minimum orders, stock already on hand, and
incoming orders when those affect the actual operation.
Avoid double-counting the same lost margin in both the underage penalty and a
separate lost-sales benefit.
Use sensitivity cases for cost parameters whose values are disputed.

### Step 10: Measure the full cost of service

**Input:** measured request volumes, vendor pricing, infrastructure, labor, and deployment constraints.
**Output:** annual incremental total cost and cost per successful forecasted series-origin.

A request can contain one or many series and one or many horizons.
Define the denominator before comparing published prices.
Cost per request, cost per series-origin, and cost per series-origin-horizon
are not interchangeable without batch and horizon details.
Include failed calls, retries, minimum charges, storage, networking,
monitoring, access controls, and model maintenance where they are incremental.

For local inference, include compute, reserved capacity or utilization
assumptions, operations, and allocated engineering only where they change with
the decision.
For a hosted service, include service fees and integration and review costs,
not just nominal token or request prices.
For fine-tuning, include training runs, scheduled retrains, checkpoint
storage, and approval effort.
Separate one-off implementation from recurring annual cost and state the
decision horizon for amortization.
Do not mix sunk expenses into an incremental future decision.

Measure latency at the required load and calculate the proportion of runs
meeting the decision deadline.
A low average can hide a poor high-percentile response time or a rate-limit
failure.
Price the fallback and the operational effect of failure.
Use matched workload, hardware availability, and service-level assumptions
when comparing local and hosted alternatives.

### Step 11: Quantify uncertainty and choose

**Input:** paired origin-level outcomes, model costs, decision guardrails, and final protected test results.
**Output:** a decision memo with point estimates, uncertainty, risk, and an owner.

Compare models on paired outcomes: each difference uses the same series,
origin, and horizon.
Adjacent rolling origins and products in the same promotion can be correlated.
Resampling individual rows as independent observations understates uncertainty
in such settings.
Use a block bootstrap over time, or another dependence-aware approach suitable
for the data, and preserve paired model results inside each resampled block.
If cross-series common shocks matter, resample time blocks jointly across
series; explain the choice and its limitations.

Report the number of origins and effective coverage, not just a confidence
interval.
A confidence interval describes uncertainty under its sampling assumptions; it
is not a probability that a fixed historical result is true.
When many cohorts or models were explored, selection can exaggerate the
apparent winner.
Reserve a later holdout or prospective pilot for confirmation rather than
treating exploratory slices as separate proofs.

Predeclare a minimum meaningful business improvement and guardrails for
critical segments, service, latency, cost, and security.
If results are close or intervals include an economically material loss,
prefer more evidence or the simpler baseline.
A successful offline score can justify a shadow deployment, but not an
unreviewed purchase-order integration.
The business owner signs the decision with the finance and operations
assumptions visible.

### Step 12: Operate, monitor, and reconsider

**Input:** approved model, fallback, monitored forecast ledger, and review schedule.
**Output:** controlled deployment with explicit rollback and re-evaluation conditions.

Begin with shadow predictions and compare against actual outcomes as labels
mature.
Then use a controlled pilot where a suitable randomized or otherwise
defensible comparison is possible.
Assign comparable stores or series to treatment and control before observing
outcomes, and account for shared inventory and interference.
If randomization is infeasible, state why and document the limits of
before-after or matched comparisons.

Monitor data freshness, schema changes, request errors, latency, forecast
coverage, horizon-specific error, decision cost, and service by cohort.
Record overrides with timestamps and reasons to distinguish model quality from
planner behavior.
A forecast drift alarm is a prompt to investigate, not proof that the model
caused a business change.
Retraining changes the model and needs a new version, evaluation, and
approval.

Define a rollback trigger such as sustained deadline misses or deterioration
beyond a pre-agreed tolerance in a critical cohort.
Retain a tested baseline fallback and make ownership explicit.
Review model and data licenses when the vendor changes weights or hosting
region.
Periodically revisit whether the original decision policy and cost assumptions
still describe the business.

## 6. Worked example

### Business situation and boundaries

A fictional retailer orders a shelf-stable product for one store each week.
The historical replay contains four paired series-origin observations, two at
a one-week horizon and two at a four-week horizon.
Each observation is expressed in units per target week.
The table is intentionally small so that every calculation can be checked by
hand; it is not enough to establish production superiority.

The evaluation compares three systems.
The local seasonal-naive system repeats an available comparable seasonal
observation.
The frozen system is a pretrained model used with past local context and no
target-task parameter updates.
The fine-tuned system uses the same base model after training on earlier
target examples; its settings were fixed before these test rows.
All three receive only information available at each origin.

| Cell | Horizon, weeks | Actual, units | Local, units | Frozen, units | Fine-tuned, units |
|---|---:|---:|---:|---:|---:|
| A | 1 | 100 | 90 | 96 | 98 |
| B | 1 | 80 | 90 | 83 | 82 |
| C | 4 | 120 | 100 | 125 | 110 |
| D | 4 | 100 | 110 | 95 | 90 |

The actual column is a sales proxy assumed to equal demand in this
illustration.
That equality is an assumption, not a property of sales data.
The four cells do not form a complete four-week lead-time demand trajectory.
Consequently the horizon-four score below measures the fourth future week, not
the sum of four weeks of demand.

### Absolute errors and bias

| Cell | Local absolute error, units | Frozen absolute error, units | Fine-tuned absolute error, units |
|---|---:|---:|---:|
| A | 10 | 4 | 2 |
| B | 10 | 3 | 2 |
| C | 20 | 5 | 10 |
| D | 10 | 5 | 10 |
| Total | 50 | 17 | 24 |

Total actual volume is 100 + 80 + 120 + 100 = 400 units.
Local WAPE is 50 / 400 = 12.50 percent.
Frozen WAPE is 17 / 400 = 4.25 percent.
Fine-tuned WAPE is 24 / 400 = 6.00 percent.
The frozen model has the lowest aggregate WAPE on these four cells, but this
is descriptive evidence only.

Local MAE is 50 / 4 = 12.50 units per cell.
Frozen MAE is 17 / 4 = 4.25 units per cell.
Fine-tuned MAE is 24 / 4 = 6.00 units per cell.
The absolute WAPE improvement of frozen versus local is 8.25 percentage
points, not 8.25 percent relative improvement.
The relative reduction is (50 - 17) / 50 = 66.0 percent in total absolute
error.

Signed errors are forecast minus actual.
Local signed errors are -10, +10, -20, and +10 units; their sum is -10 units.
Local bias is -10 / 4 = -2.50 units per cell.
Frozen signed errors are -4, +3, +5, and -5 units; their sum is -1 unit.
Frozen bias is -0.25 units per cell.
Fine-tuned signed errors are -2, +2, -10, and -10 units; their sum is -20
units.
Fine-tuned bias is -5.00 units per cell.
Low aggregate bias can hide large offsetting errors.

### Horizon effects

At horizon one, actual volume is 180 units.
Local absolute error is 20 units, so WAPE is 20 / 180 = 11.11 percent.
Frozen absolute error is 7 units, so WAPE is 7 / 180 = 3.89 percent.
Fine-tuned absolute error is 4 units, so WAPE is 4 / 180 = 2.22 percent.
Fine-tuning has the lowest one-week WAPE on this tiny example.

At horizon four, actual volume is 220 units.
Local absolute error is 30 units, so WAPE is 30 / 220 = 13.64 percent.
Frozen absolute error is 10 units, so WAPE is 10 / 220 = 4.55 percent.
Fine-tuned absolute error is 20 units, so WAPE is 20 / 220 = 9.09 percent.
The ranking reverses between the two neural candidates.
If the operational decision depends mainly on week four, the short-horizon win
is not sufficient justification for fine-tuning.
If it depends on total demand over the next four weeks, collect the missing
full paths and evaluate that sum instead.

### Decision cost

For a deliberately simple one-period comparison, assume the forecast itself is
the order quantity.
There is no opening stock, no incoming order, no pack constraint, and no
carryover in this illustration.
Each unit ordered above demand incurs USD 2 of overage cost; each unit below
demand incurs USD 8 of underage cost.
The USD 2 is assumed to be an incremental excess-stock cost, not the full
purchase price of a reusable unit.
The USD 8 represents a stated marginal unmet-demand cost and must not also be
counted as a separate lost-margin saving.

| Cell | Local cost, USD | Frozen cost, USD | Fine-tuned cost, USD |
|---|---:|---:|---:|
| A | 10 × 8 = 80 | 4 × 8 = 32 | 2 × 8 = 16 |
| B | 10 × 2 = 20 | 3 × 2 = 6 | 2 × 2 = 4 |
| C | 20 × 8 = 160 | 5 × 2 = 10 | 10 × 8 = 80 |
| D | 10 × 2 = 20 | 5 × 8 = 40 | 10 × 8 = 80 |
| Total | 280 | 88 | 180 |

In this assumed policy, frozen reduces modeled decision cost by USD 192 versus
local across the four observations.
Fine-tuned reduces it by USD 100 versus local.
Fine-tuned has a lower one-week absolute error but performs worse in this
four-cell decision-cost calculation.
This comparison uses forecasts as order quantities only to isolate arithmetic;
a realistic policy would include stock, lead time, and constraints.
The cost-minimizing theoretical quantile under the single-period assumptions
is 8 / (8 + 2) = 0.80.
None of the three point forecasts has been established as an 80th percentile,
so the calculation does not prove any policy is optimal.

### Annual operating economics

Suppose the business expects 1,000,000 series-origins per year, each issuing a
four-week forecast.
Assume, only for illustration, that a validated larger study estimates
frozen-system decision benefit at USD 0.060 per series-origin relative to
local.
The implied gross annual benefit is USD 60,000 per year before incremental
system cost.
It is not legitimate to extrapolate USD 192 from the four-row hand example to
the entire network.

Assume incremental frozen-service charges of USD 0.012 per series-origin, or
USD 12,000 per year.
Assume incremental annual monitoring and support cost of USD 8,000 and
implementation cost of USD 30,000 in year one.
Year-one incremental net benefit is USD 60,000 - USD 12,000 - USD 8,000 - USD
30,000 = USD 10,000.
Later-year net benefit, assuming unchanged volume and costs and no further
implementation charge, is USD 40,000 per year.
This is a scenario, not an observed saving or a cash-flow valuation.

Under a downside benefit of USD 0.025 per series-origin, gross benefit becomes
USD 25,000 per year.
Year-one net is USD 25,000 - USD 12,000 - USD 8,000 - USD 30,000 = negative
USD 25,000.
Later-year net is USD 5,000 per year under the same assumptions.
A decision-maker should seek stronger evidence before committing based on the
narrow base-case margin.
The illustration excludes taxes, financing, contract minimums, and changes in
inventory working capital.

### Recommendation

Retain the local system as the production fallback.
Use the frozen candidate in a shadow test with full lead-time trajectories and
enough paired origins to estimate uncertainty.
Include the fine-tuned candidate only if its incremental training and
operation costs are measured and its critical-horizon behavior improves.
Require a predeclared service guardrail and an economic benefit that remains
acceptable in plausible downside cases.
No firm forecast, inventory plan, sales target, or supplier commitment follows
automatically from these four observations.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as a local script and run it with `python`.
It reproduces the worked example and computes WAPE, MAE, bias,
horizon-specific WAPE, and the simplified decision cost.
The rows are already aligned and are not a real rolling-origin pipeline.
The script does not train or call a foundation model and makes no claim about
model performance outside the supplied rows.

```python
from collections import defaultdict

# Weekly target units; each row represents a distinct series-origin-horizon cell.
rows = [
    {"cell": "A", "h": 1, "actual": 100, "local": 90, "frozen": 96, "tuned": 98},
    {"cell": "B", "h": 1, "actual": 80, "local": 90, "frozen": 83, "tuned": 82},
    {"cell": "C", "h": 4, "actual": 120, "local": 100, "frozen": 125, "tuned": 110},
    {"cell": "D", "h": 4, "actual": 100, "local": 110, "frozen": 95, "tuned": 90},
]
models = ("local", "frozen", "tuned")
underage_usd = 8.0
overage_usd = 2.0


def score(group, model):
    """Return metrics for matched cells; WAPE requires positive total actual."""
    if not group:
        raise ValueError("Cannot score an empty group")
    total_actual = sum(row["actual"] for row in group)
    if total_actual <= 0:
        raise ValueError("WAPE requires positive total actual")
    errors = [row[model] - row["actual"] for row in group]
    abs_sum = sum(abs(error) for error in errors)
    cost = sum(
        underage_usd * max(-error, 0) + overage_usd * max(error, 0)
        for error in errors
    )
    return {
        "wape_pct": 100.0 * abs_sum / total_actual,
        "mae_units": abs_sum / len(group),
        "bias_units": sum(errors) / len(group),
        "cost_usd": cost,
    }


by_horizon = defaultdict(list)
for row in rows:
    by_horizon[row["h"]].append(row)

for model in models:
    result = score(rows, model)
    print(
        f"{model}: WAPE={result['wape_pct']:.2f}% "
        f"MAE={result['mae_units']:.2f} units "
        f"bias={result['bias_units']:.2f} units "
        f"cost=USD {result['cost_usd']:.2f}"
    )
    for horizon in sorted(by_horizon):
        value = score(by_horizon[horizon], model)["wape_pct"]
        print(f"  horizon {horizon} weeks: WAPE={value:.2f}%")

volume = 1_000_000
benefit_per_series_origin = 0.060
service_per_series_origin = 0.012
support = 8_000
implementation = 30_000
net_year_one = (
    volume * (benefit_per_series_origin - service_per_series_origin)
    - support - implementation
)
print(f"Illustrative year-one net: USD {net_year_one:,.2f}")
```

Expected output:

```text
local: WAPE=12.50% MAE=12.50 units bias=-2.50 units cost=USD 280.00
  horizon 1 weeks: WAPE=11.11%
  horizon 4 weeks: WAPE=13.64%
frozen: WAPE=4.25% MAE=4.25 units bias=-0.25 units cost=USD 88.00
  horizon 1 weeks: WAPE=3.89%
  horizon 4 weeks: WAPE=4.55%
tuned: WAPE=6.00% MAE=6.00 units bias=-5.00 units cost=USD 180.00
  horizon 1 weeks: WAPE=2.22%
  horizon 4 weeks: WAPE=9.09%
Illustrative year-one net: USD 10,000.00
```

The positivity check is appropriate for this nonnegative-demand example.
A production evaluator should report zero-denominator cohorts rather than
silently discard them.
It must also join on explicit identifiers, preserve failure rows, reconstruct
as-of data, and account for overlapping forecast origins.
A full inventory simulator would separately model stock, lead time, policy,
service, and carryover.

## 8. Data, standards, and regulatory context

### Data contracts

The minimum ledger identifies series, location, origin timestamp, target
bucket, horizon, actual, forecast, version, and availability time.
For example, a retail item may move between locations or change package size
during the test period.
A stable identifier is not enough unless the associated unit of measure and
hierarchy are versioned.
Join failures should be counted before scoring rather than hidden by an inner
join.

A product identifier standard can improve matching across trading partners,
but it does not certify demand quality.
GS1 defines Global Trade Item Numbers (GTINs) and related identification
practices.
A GTIN identifies a trade item; it does not itself say whether a store was in
stock or whether a sale was recorded at forecast time.
Maintain the relationship among GTIN, internal SKU, package hierarchy, store,
and effective dates.

A weekly calendar must specify time zone, week start, retail fiscal calendar,
and treatment of a 53-week year.
An annual seasonal-naive reference must be mapped to the intended comparable
week.
Holiday shifts may make a simple 52-week offset a poor business match; a
calendar-adjusted baseline can be evaluated if defined before test inspection.
Record the time at which a promotion plan was published, and separately record
any later revision.

Training data and inference context need provenance and access control.
For hosted inference, the contract should state retention, permitted secondary
use, region, subprocessors, deletion, and incident obligations.
A vendor's zero-shot label does not answer whether your submitted context is
stored or used later.
Avoid sending supplier-sensitive costs or identifiable customer data unless
there is an approved purpose and lawful basis.

### Standards and governance

The NIST AI Risk Management Framework (AI RMF 1.0, January 2023) is voluntary
guidance, not a law requiring a particular forecasting model.
Its govern, map, measure, and manage functions provide a useful structure for
documenting ownership, context, performance, and monitoring.
The framework does not supply a universal WAPE threshold or guarantee a model
is safe.
Organizations should connect the evaluation ledger and rollback plan to their
own risk controls.

The European Union AI Act, Regulation (EU) 2024/1689, uses a risk-based legal
framework and has staged applicability dates.
Whether any particular supply-chain application or provider obligation falls
within a specified category depends on its facts and the applicable
provisions.
A generic demand forecast is not automatically a high-risk system solely
because it uses a neural model.
Obtain qualified legal review for actual obligations, timing, and
jurisdiction; this educational discussion is not legal advice.

Privacy laws may apply if inputs or outputs contain personal data, such as
named planner notes or customer-level transaction details.
The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679,
imposes requirements on covered processing; application depends on the facts.
Data minimization and retention decisions should be reviewed by the
responsible privacy team.
Aggregating series reduces some exposure but does not prove that all
re-identification or confidentiality risk is absent.

A model license may limit commercial use, distribution, derivatives, or
deployment environment.
Inspect the exact model version and license, not a general product page.
Document rights to the training and evaluation data separately from rights to
model weights.
An open weight file is not necessarily unrestricted data or an unconditional
right to redistribute a fine-tuned model.

### Audit artifacts

Preserve a manifest of data snapshots, model versions, parameter settings,
origins, exclusions, and scoring code.
Record the date a model version was available, the claimed pretraining cutoff,
and unknowns about public benchmark exposure.
A test set can be temporally clean locally while the pretrained corpus has
unknown overlap with it.
Make that uncertainty visible in the approval memo.

The outcome ledger should allow a reviewer to reconstruct aggregate scores
from row-level predictions and final targets.
Protect commercial data and limit access according to internal policy.
A durable audit trail enables investigation of changing scores, model
versions, and revised actuals.
It does not establish that the underlying counterfactual demand or costs are
perfectly measured.

## 9. Industry practice and case evidence

### Public evidence and its transfer limits

The M5 forecasting competition used Walmart unit-sales data and evaluated
forecasts for many retail series.
The published M5 accuracy competition paper describes a hierarchical retail
forecasting challenge and the competition design.
This is evidence that large-scale retail evaluation and aggregation choices
matter.
It is not evidence that a particular foundation model will outperform a local
method in a new retailer's operation.
The competition target and scoring rules should be read before importing any
ranking into a different setting.

The M4 competition compared a broad collection of statistical and
machine-learning forecasting methods across many series.
Its published results are useful evidence about the importance of empirical
benchmarks and diverse data.
They do not validate a modern pretrained model for any specific supply chain.
A benchmark's average rank can conceal changes in horizon, data frequency, and
decision loss.

The Chronos paper describes pretrained time-series models and reports
evaluations on benchmark datasets.
Those results support investigating a frozen pretrained candidate, not
assuming its local incremental profit.
Check the exact model and data used, whether benchmark exposure is disclosed,
and whether the task matches the intended horizon and target.
The paper's pretraining and evaluation setup should be distinguished from a
customer's private-data replay.

The TimesFM paper similarly presents a pretrained time-series foundation model
and benchmark evidence.
A prospective user must independently measure performance, compute cost, and
feasibility on their own series.
Model releases, context limits, and licenses can change; record the precise
artifact and date used.
Neither a research paper nor a vendor benchmark substitutes for an as-of
operational test.

### What to infer and what not to infer

Documented fact: the M5 dataset reflects one particular retail setting and
competition protocol.
[INFERENCE] A retailer with different promotions, stockout patterns, or lead
times may observe a different model ranking.
That inference is plausible but must be tested locally.
Do not attribute the fictional worked example's USD savings to Walmart, a
benchmark author, or a model vendor.

Documented fact: public forecasting benchmarks often supply many series and
standardized metrics.
[INFERENCE] Standardization makes repeatable comparisons easier while
potentially omitting a specific buyer's cost asymmetry and operating
constraints.
An operational evaluator should therefore reproduce a common metric and add a
separately stated decision simulation.
A lower benchmark error does not establish realized reduction in inventory or
an improvement in fill rate.

A vendor may report an average improvement across selected datasets.
Ask for the datasets, dates, baselines, excluded series, horizon weighting,
version, and uncertainty.
A baseline chosen for convenience may not represent the strongest feasible
incumbent.
If the vendor cannot disclose pretraining data, record the limitation rather
than inventing a clean independence guarantee.

Industry practice often combines automated forecasts with planner review and
operational constraints.
A historical forecast archive should distinguish model output from overrides
and approved plans.
Without timestamps, a final plan can accidentally contain information that
arrived after the original decision.
A pilot should measure the combined workflow and the effort required to
operate it.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future actual in inputs | feature timestamps exceed origin | future outcome leaks into context | inflated backtest | reconstruct as-of snapshots |
| Training-label leakage | fine-tune window ends after origin | target examples contain later labels | unrealistic adaptation | enforce label-availability cutoff |
| Validation reused as test | repeated changes after scores | selection fits evaluation noise | optimistic winner | protect later final period |
| Unknown pretraining overlap | corpus provenance absent | model may have seen test data | uncertain independence | disclose and use prospective holdout |
| Unfair weak baseline | only a constant comparator | practical incumbent omitted | overstated uplift | include feasible local and incumbent systems |
| Missing failure rows | lower model coverage | hard cells or timeouts disappear | biased accuracy | score fallback and report coverage |
| Seasonal mismatch | 53-week or holiday anomalies | wrong comparable period | misleading baseline or forecast | version calendar mapping |
| Censored demand | error improves in empty-stock weeks | sales proxy hides unmet demand | poor replenishment decisions | flag stockouts and assess proxy limits |
| Aggregate-only metric | critical items worsen | high volume dominates WAPE | service degradation | report predeclared segments |
| Horizon collapse | week-one score drives four-week decision | short and long horizons differ | wrong model chosen | score each horizon and lead-time sum |
| Quantile arithmetic | summed marginal quantiles treated as total quantile | dependence ignored | wrong safety stock | use joint paths or justified dependence model |
| Wrong cost denominator | price shown only per request | batching and horizons differ | false financial comparison | normalize per series-origin |
| Correlated test rows | thousands of rows but few origins | overlapping windows share shocks | narrow uncertainty interval | paired block resampling |
| Improper cost mapping | acquisition price treated as total excess loss | salvage or carryover ignored | distorted policy | specify marginal overage economics |
| Retraining ignored | training jobs absent from quote | fine-tuning effort omitted | understated TCO | count training, approval, and support |
| Plan mistaken for truth | final plan used as forecast | later overrides enter backtest | impossible historical accuracy | timestamp original forecast and revisions |
| Model-version drift | service silently changes weights | scores no longer describe deployment | unrecognized degradation | pin version and requalify changes |
| Timeout ignored | low mean latency, missed deadline | tail latency or rate limits | orders use stale numbers | measure deadline success and fallback |
| Zero WAPE denominator | cohort has no sales | metric divides by zero | misleading or missing score | report absolute units and no WAPE |
| Double-counted value | lost margin appears twice | cost and benefit overlap | overstated business case | reconcile each cost category |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Actual sales proxy demand | lets example score predictions | stockout or substitution | compare availability flags; qualify demand claims |
| As-of inputs are reconstructible | makes replay historically feasible | overwritten plans or late corrections | audit event and ingestion timestamps |
| Series identifiers are stable | aligns forecasts and actuals | product, pack, or store changes | version master-data mappings |
| Candidate cells match | supports paired comparison | missing outputs or different filters | reconcile row counts and use fallback |
| Season exists for baseline | makes seasonal naive meaningful | short history or calendar shift | report fallback and calendar sensitivity |
| Evaluation reflects future workload | supports transfer to deployment | new stores, promotions, or regime shifts | prospective shadow and cohort monitoring |
| Cost inputs are marginal | makes simulated net value interpretable | sunk costs or double-counted margins | finance review and sensitivity analysis |
| Order rule captures constraints | connects forecasts to business decisions | inventory, pack, or lead-time omitted | simulate realistic inventory state |
| Probability output is calibrated enough | makes quantile policy interpretable | distribution shift or sparse samples | horizon-level coverage and proper scores |
| Dependence is represented | supports uncertainty and lead-time risk | common shocks and overlapping origins | paired time-block resampling or joint model |
| Model version is stable | makes replay reproducible | silent vendor update | pin artifact and monitor version |
| Legal rights permit use | enables production processing | license, privacy, or regional restrictions | obtain legal and security review |

The worked example additionally assumes a single-period decision, no inventory
carryover, and a linear penalty per unit.
Its four cells are chosen to illustrate arithmetic, not a statistically valid
experiment.
The annual scenario introduces an independently assumed benefit and cannot be
inferred from those cells.
A real business case needs representative demand, actual decision constraints,
and uncertainty around both costs and benefits.

## 12. Exercises

### Exercise 1 — Classify adaptation

A frozen model consumes 52 weeks of local history and produces a forecast.
A second system applies a correction factor fitted on local validation
outcomes.
A third updates network weights on earlier local examples.

**Deliverable:** label each system and explain what changed.
**Acceptance criteria:** distinguish context from parameter updates and identify the fitted correction as local adaptation.
**Time:** 8 minutes.

### Exercise 2 — Protect the timeline

A model is tuned on origins through week 120 with a four-week horizon.
Weekly labels become available at the end of their week, and training finishes
overnight.
The team proposes a final test at the beginning of week 121 using all
validation labels.

**Deliverable:** draw the boundary and name the earliest point when the last validation label is available.
**Acceptance criteria:** identify week 124 as the last label week and distinguish end-of-week availability from the next usable decision time.
**Time:** 10 minutes.

### Exercise 3 — Score a horizon

For two one-week cells, actuals are 100 and 80 units.
Frozen predictions are 96 and 83 units; fine-tuned predictions are 98 and 82
units.

**Deliverable:** compute absolute errors, WAPE, and signed bias for both candidates.
**Acceptance criteria:** use sum absolute errors divided by 180 units and declare forecast-minus-actual bias.
**Time:** 12 minutes.

### Exercise 4 — Calculate operational penalty

Use the same two cells and forecasts as Exercise 3.
Assume a forecast is the order quantity, underage costs USD 8 per unit, and
overage costs USD 2 per unit.

**Deliverable:** calculate each model's total two-cell cost and identify the cheaper model under these assumptions.
**Acceptance criteria:** apply the correct cost to each signed error and do not claim a general policy optimum.
**Time:** 10 minutes.

### Exercise 5 — Compare annual costs

A hosted candidate processes 800,000 series-origins per year at USD 0.015
each.
Incremental support is USD 9,000 per year and one-off implementation is USD
24,000 in year one.
A separate validated study estimates gross decision benefit of USD 0.070 per
series-origin.

**Deliverable:** compute gross benefit, annual service expense, and year-one incremental net benefit.
**Acceptance criteria:** use the same denominator and include implementation only in year one.
**Time:** 10 minutes.

### Exercise 6 — Critique an aggregate winner

A report says WAPE improved from 10 percent to 8 percent across all horizons.
Its one-week cohort improved, its four-week critical items worsened, and 5
percent of candidate requests timed out and were excluded.
It reports a narrow interval based on independently resampling every
series-horizon row.

**Deliverable:** list four threats and propose a correction for each.
**Acceptance criteria:** address horizon, critical segments, missing predictions, and correlated rows.
**Time:** 15 minutes.

### Exercise 7 — Design a fine-tuning test

A distributor has two years of weekly data, a frozen pretrained model, and a
monthly fine-tuning budget.
Some promotion plans are revised after issuance and some products have less
than one seasonal cycle of history.

**Deliverable:** write a one-page experiment plan with baseline, snapshots, tuning, test, cost, and fallback.
**Acceptance criteria:** respect label availability, define short-history behavior, protect final test, and price retraining.
**Time:** 25 minutes.

### Exercise 8 — Choose a deployment path

A shadow replay shows a modest improvement in average error, but the estimated
net benefit range spans negative and positive values.
A safety-critical product cohort has too few observations to draw a reliable
conclusion.
The vendor will not commit to a pinned model version.

**Deliverable:** recommend launch, restricted pilot, or no deployment with decision gates and owners.
**Acceptance criteria:** discuss uncertainty, critical-cohort protection, version control, fallback, and what evidence could change the decision.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The first is frozen zero-shot for the stated task: local history is context,
not a weight update.
The second is a frozen base plus a locally fitted calibration layer; the
overall system is not purely zero-shot.
The third is fine-tuned because target-related examples changed network
weights.
All three must obey the same forecast-origin information boundary.

### Exercise 2

The label for origin 120 at horizon four is the outcome in week 124.
It is available at the end of week 124 under the exercise assumptions.
A beginning-of-week-121 test cannot use it; if fitting takes place overnight
after that label arrives, the first usable decision is no earlier than the
next scheduled run.
An alternative is to use only earlier validation origins whose labels have
matured before the test.

### Exercise 3

Frozen absolute errors are 4 and 3 units, totaling 7 units.
Its WAPE is 7 / 180 × 100 = 3.89 percent.
Its signed errors are -4 and +3 units, so bias is -0.5 units per cell.
Fine-tuned absolute errors are 2 and 2 units, totaling 4 units.
Its WAPE is 4 / 180 × 100 = 2.22 percent.
Its signed errors are -2 and +2 units, so bias is 0 units per cell.
Neither result says anything about the four-week horizon.

### Exercise 4

Frozen incurs 4 × USD 8 = USD 32 on the first cell and 3 × USD 2 = USD 6 on
the second.
Its total is USD 38.
Fine-tuned incurs 2 × USD 8 = USD 16 and 2 × USD 2 = USD 4, totaling USD 20.
Fine-tuned is cheaper for these two cells under the stated
point-forecast-as-order rule.
A full policy comparison needs inventory, lead time, and a suitably chosen
demand quantile.

### Exercise 5

Gross annual benefit is 800,000 × USD 0.070 = USD 56,000.
Service expense is 800,000 × USD 0.015 = USD 12,000 per year.
Year-one incremental net benefit is USD 56,000 - USD 12,000 - USD 9,000 - USD
24,000 = USD 11,000.
Under unchanged assumptions, later-year net benefit before other costs is USD
35,000 per year.
The figure depends on the independent study's benefit estimate and does not
prove that benefit will recur.

### Exercise 6

The aggregate can hide horizon reversal; report horizon-specific outcomes and
the relevant lead-time total.
Critical items can be harmed despite volume-weighted improvement; specify a
segment guardrail and report cohort size.
Excluding timeouts favors the candidate; include the actual fallback forecast
in delivered-system results and report failure rate.
Independent row resampling ignores common shocks and overlapping windows; use
paired blocks that preserve time and shared structure.
The proposed report is not sufficient evidence for an unrestricted launch.

### Exercise 7 rubric

A strong answer names an operational decision and target, then creates as-of
snapshots of both target and promotion-plan versions.
It includes naive and seasonal-naive or an explicit short-history fallback,
plus a feasible incumbent.
It fixes validation choices before the protected test and ensures every
training label has matured before the monthly retrain.
It distinguishes frozen, fine-tuned, and any calibrated versions.
It prices training, inference, monitoring, and fallback on the same
series-origin volume.
It reports horizon and critical-cohort scores, failure coverage, and
uncertainty.

### Exercise 8 rubric

A strong answer does not treat a small average error gain as a guaranteed
financial return.
It protects the critical cohort, requires a reproducible version or equivalent
change control, and retains an operational fallback.
It assigns an owner for finance, operations, security, and model monitoring.
A restricted shadow or pilot is defensible only if the vendor and governance
constraints allow it and the safety guardrails are met.
The decision gates should specify the evidence and thresholds needed for
expansion or rollback.

## 14. Interview questions

1. **What does zero-shot mean when a model consumes target history?**
   A strong answer separates context from target-task weight updates and
mentions locally fitted calibration.
   A weak answer says zero-shot requires no target observations at all.
2. **Why include a seasonal-naive comparator?**
   A strong answer explains a cheap operational benchmark, seasonal alignment,
and fallback for short histories.
   A weak answer compares only with a random forecast.
3. **What is an as-of snapshot?**
   A strong answer distinguishes event time, availability time, and subsequent
corrections.
   A weak answer assumes today's data warehouse reproduces the past
automatically.
4. **Why can the winning model differ by horizon?**
   A strong answer discusses information decay, seasonality, different
uncertainty, and decision timing.
   A weak answer assumes an aggregate ranking holds for every week.
5. **How is lead-time demand scored?**
   A strong answer aggregates each complete path before measuring error and
distinguishes marginal from joint uncertainty.
   A weak answer calls a week-four forecast the four-week total.
6. **When is WAPE undefined or misleading?**
   A strong answer cites a zero denominator, volume weighting, and segment
masking.
   A weak answer treats the average of individual percentage errors as WAPE.
7. **How do you price underforecasting and overforecasting?**
   A strong answer states marginal costs, retained value, unmet-demand
treatment, and sensitivity.
   A weak answer assumes every excess unit loses its whole acquisition cost.
8. **How would you compare compute costs?**
   A strong answer normalizes workload and includes retries, training,
support, and implementation.
   A weak answer compares unrelated request prices without batching.
9. **What makes a backtest confidence interval credible?**
   A strong answer describes paired comparisons, temporal dependence, and
selection effects.
   A weak answer assumes every overlapping horizon row is independent.
10. **What warrants production approval?**
    A strong answer combines protected evidence, guardrails, financial
downside, version control, and fallback.
    A weak answer deploys solely because a public benchmark rank is high.

## 15. Key terms

- [As-of snapshot](#4-definitions-and-vocabulary): information available at a particular origin.
- [Bias](#4-definitions-and-vocabulary): average signed forecast error under a declared convention.
- [Calibration](#4-definitions-and-vocabulary): fitted adjustment of forecasts or probabilities.
- [Context](#4-definitions-and-vocabulary): past values and allowed features supplied to inference.
- [Contamination](#4-definitions-and-vocabulary): exposure undermining an intended independence claim.
- [Covariate](#4-definitions-and-vocabulary): a non-target model input.
- [Fine-tuning](#4-definitions-and-vocabulary): target-related updates of pretrained weights.
- [Forecast origin](#4-definitions-and-vocabulary): issuance boundary for a prediction.
- [Frozen zero-shot](#4-definitions-and-vocabulary): no target-task parameter update before prediction.
- [Global model](#4-definitions-and-vocabulary): model sharing parameters across series.
- [Horizon](#4-definitions-and-vocabulary): future distance measured in buckets.
- [Lead-time demand](#step-7-score-by-horizon-and-segment): total demand during the replenishment interval.
- [Leakage](#4-definitions-and-vocabulary): use of information not available at prediction time.
- [Local model](#4-definitions-and-vocabulary): separately fitted model for a series.
- [MAE](#4-definitions-and-vocabulary): mean absolute error in target units.
- [Naive forecast](#4-definitions-and-vocabulary): repetition of the latest observed value.
- [Overage cost](#4-definitions-and-vocabulary): marginal penalty for a unit above demand.
- [Pinball loss](#4-definitions-and-vocabulary): asymmetric error score for a quantile.
- [Pretraining](#4-definitions-and-vocabulary): learning shared weights before the target task.
- [Quantile](#4-definitions-and-vocabulary): modeled cumulative-probability threshold.
- [Rolling origin](#4-definitions-and-vocabulary): repeated historical issuance and forward scoring.
- [Seasonal naive](#4-definitions-and-vocabulary): repetition of the last observed matching season.
- [Series-origin](#4-definitions-and-vocabulary): one series at one forecast issuance.
- [Shadow mode](#4-definitions-and-vocabulary): generating outputs without changing live decisions.
- [TCO](#4-definitions-and-vocabulary): total cost of ownership over a defined period.
- [Underage cost](#4-definitions-and-vocabulary): marginal penalty for a unit below demand.
- [Validation set](#4-definitions-and-vocabulary): period for model and policy selection.
- [WAPE](#4-definitions-and-vocabulary): total absolute error divided by total actual volume.

## 16. Sources

1. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting* 38(4), 2022. [https://doi.org/10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Retail competition design and results; not a local deployment claim.
2. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “The M4 Competition: 100,000 time series and 61 forecasting methods.” *International Journal of Forecasting* 36(1), 2020. [https://doi.org/10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014). Broad empirical forecasting comparison.
3. Ansari, A. F., et al. “Chronos: Learning the Language of Time Series.” *Transactions on Machine Learning Research*, 2024. [https://arxiv.org/abs/2403.07815](https://arxiv.org/abs/2403.07815). Pretraining and benchmark design for a time-series foundation model.
4. Das, A., et al. “A decoder-only foundation model for time-series forecasting.” *Proceedings of the 41st International Conference on Machine Learning*, 2024. [https://proceedings.mlr.press/v235/das24c.html](https://proceedings.mlr.press/v235/das24c.html). TimesFM research description and evaluation.
5. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, third edition. [Time series cross-validation](https://otexts.com/fpp3/tscv.html) and [forecast accuracy](https://otexts.com/fpp3/accuracy.html). Definitions and rolling-origin evaluation.
6. Gneiting, T., and Raftery, A. E. “Strictly Proper Scoring Rules, Prediction, and Estimation.” *Journal of the American Statistical Association* 102(477), 2007. [https://doi.org/10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437). Foundations of probabilistic scoring.
7. National Institute of Standards and Technology. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, January 2023. [https://doi.org/10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1). Voluntary risk management guidance.
8. GS1. [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Identification of trade items, not evidence of demand quality.
9. European Union. [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), Artificial Intelligence Act. Consult current applicable provisions and qualified counsel.
10. European Union. [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), General Data Protection Regulation. Applicability depends on processing and jurisdiction.

## 17. Links

- [Module 18 index](_index.md): topic sequence and assessment context.
- [Global versus local models](03_global-local-models.md): cross-series and local estimation choices.
- [Neural forecasting zoo](04_neural-forecasting-zoo.md): architecture candidates and operating tradeoffs.
- [Time-series foundation models](05_ts-foundation-models.md): pretraining, adaptation, and vendor claims.
- [Module 18 survey](../../../course/18_deep_learning_foundation_models_llm_agents.md): broader module narrative and lab context.
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md): decision simulation context.
