# Demand Forecasting at Short Horizons

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 07; Module 22, Topic 01  
> Version 2026.5

## 1. Why this topic exists

The store manager must decide how many pickers to schedule before the dinner peak.
The courier operations lead must decide where and when to recruit delivery capacity.
The customer experience lead must decide whether a delivery slot can remain open.
All three decisions depend on demand, but not on the same definition of demand or the same deadline.

A forecast of tomorrow's total orders can be accurate while a store is overwhelmed for 30 minutes.
For example, 240 orders distributed evenly over four hours require a different roster than 240 orders concentrated in one hour.
Hourly averages can conceal a 15-minute surge that makes a short promise unattainable.
A slot-level forecast therefore matters because capacity must be present when work arrives, not merely at some time during the day.

Quick commerce has unusually short feedback loops.
Rain can begin after a schedule is published.
An event may release a crowd within minutes.
A promotion can bring both more customers and larger baskets.
An app can also suppress orders when a store closes, delivery fees rise, or promised times become unattractive.
Observed completed orders are therefore not always the demand that would have arrived under normal availability.

The forecasting task is not to guess a single number and call it a staffing plan.
It is to predict a defined random quantity, at a specified time, with only information then available.
The operator combines that distribution with productivity, labor rules, queue capacity, cost and service policies.
A forecast is a probability statement; a plan is a decision; a target is an aspiration; and a commitment is a promise made to another party.
Confusing these objects can lead to understaffing or a misleading retrospective evaluation.

The cadence also matters.
A weekly schedule may use a forecast produced several days ahead.
A same-day adjustment uses newer weather and order signals but faces fewer available workers.
A 15-minute dispatch decision can respond to incoming orders, but it cannot retroactively change a roster.
Each forecast must be evaluated at the horizon and information cutoff of the decision it supports.

This unit builds a store-by-slot demand forecast and converts it into a transparent staffing decision.
It treats weather, events and promotions as information with publication times, not magical knowledge of the future.
It also explains how to backtest without data leakage and how to distinguish forecast error from execution failure.
The numbers are illustrative and are not claims about a particular operator.

## 2. Learning outcomes

After completing this unit, you can:

1. **Define** an hourly or 15-minute demand target, its geography, event time, forecast origin and operational use.
   Exercise 1 checks the target contract.
2. **Construct** an as-of feature table for weather, events and promotions without future information.
   Exercise 2 checks the timestamps and version choices.
3. **Calculate** a seasonal benchmark and a transparent adjusted forecast for successive slots.
   Exercises 3 and 4 test arithmetic and interpretation.
4. **Design** a rolling-origin backtest that matches actual planning lead times.
   Exercise 5 checks the split, label maturity and baselines.
5. **Compare** point and probabilistic forecasts using appropriate loss and calibration measures.
   Exercise 6 asks for metrics and their limitations.
6. **Translate** demand into picker capacity and a risk-aware staffing recommendation.
   Exercise 7 checks rounding, occupancy and the distinction between a plan and a promise.
7. **Diagnose** demand censoring, feature leakage and policy feedback.
   Exercise 8 asks for evidence and a corrective design.
8. **Govern** overrides and operational handoffs using named decision rights.
   Exercise 9 checks the proposed decision record and ownership.

The outcomes deliberately separate prediction from action.
You can improve forecast accuracy without improving service if scheduling constraints are ignored.
Conversely, expensive extra capacity may improve service without demonstrating a better forecast.
Assess both the information and the decision that used it.

## 3. The core mental model

```text
  Prior comparable orders       Published future information
  and availability logs         weather / event / promotion vintages
             \                      /
              \                    /
               v                  v
         AS-OF FEATURE SNAPSHOT AT ORIGIN t
                         |
                         v
       forecast distribution for store x future slot
              (mean / quantiles / scenarios)
                         |
            +------------+-------------+
            |                          |
            v                          v
   expected picker workload    expected delivery workload
            |                          |
  productivity + breaks       travel + batching + availability
            |                          |
            v                          v
    roster recommendation       courier capacity plan
            \                          /
             +---------> slot policy <-+
                              |
                              v
                 actual orders and outcomes
                              |
                              v
          label maturity, evaluation, decision review
```

Read the diagram from left to right in time.
At forecast origin `t`, the feature snapshot contains only records available by `t`.
Each forecast refers to a future target interval and is stored with its origin, model version and data vintage.
A distribution is preferable to a point when under-capacity and over-capacity have different costs.

Orders generate at least two different kinds of work.
Picker workload depends on basket composition, stock location and substitutions.
Courier workload depends on distance, traffic, stacking and handoff time.
One order is not necessarily one courier trip, and one order is not necessarily a fixed number of picker minutes.
A shared demand forecast can feed both calculations without making their productivity assumptions identical.

The world changes after a forecast is issued.
The later realized weather belongs in outcome analysis, not in a historical feature vector unless it was already observed at the origin.
Likewise, a promotional cancellation published at 16:20 cannot be used in a forecast supposedly issued at 16:00.
The backtest must reproduce what the operator could actually know, not what the database shows today.

Feedback is a loop, not an arrow of causation from staffing to unconstrained demand.
Staffing can affect acceptance, promised time and slot availability.
Those policies can affect observed orders, which become future training labels.
If a model learns only from constrained outcomes, it can recommend repeating the constraint.
Log the intervention and retain exposure data so analysts can identify the affected observations.

### Decision clocks

| Decision | Typical horizon in this example | Decision owner | What can still change |
|---|---|---|---|
| Base picker roster | Previous day | Store labor manager | Overtime and flex calls, subject to policy |
| Courier recruitment | Previous day to hours ahead | Courier operations lead | Incentives and zone allocation |
| Intraday picker adjustment | 60–120 minutes ahead | Shift manager | Approved flex staff or task reassignment |
| Slot availability | 15–60 minutes ahead | Marketplace control lead | Slot cap or promise, within customer policy |
| Individual dispatch | Minutes ahead | Dispatch system operator | Assignment and batching |

These are example horizons, not universal rules.
The important requirement is that the forecast origin precede the last moment at which a particular decision can be made.
A 10-minute-ahead forecast cannot fairly be credited for a roster locked the previous afternoon.

### One prediction, multiple decisions

Suppose a slot has a mean forecast of 18 orders and a 90th percentile of 24 orders.
The mean is appropriate for an expected-volume discussion under the model.
The 90th percentile is a possible input to a service-oriented staffing rule.
Neither number itself says how many workers should be booked.
That decision requires service time, utilization, breaks, the shape of arrivals within the slot, costs and contractual constraints.
A high percentile is not a guarantee: even a calibrated 90th percentile is exceeded about 10% of the time over comparable forecasts.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse or qualification | Unit or coding |
|---|---|---|---|
| Forecast origin | Time at which a prediction is issued | Confused with the future slot start | Timestamp with timezone |
| Target interval | Half-open time interval being predicted | Assigning a boundary order to both slots | `[start, end)` |
| Lead time | Time from origin to target start | Measured from model training time instead | Minutes or hours |
| Horizon | Future interval or set of intervals predicted | Mixing horizons in one accuracy score | Minutes or slots |
| Bucket | Aggregation interval of a series | Treating daily totals as evidence of slot fit | 15 minutes or one hour |
| Store | Physical fulfillment node responsible for work | Using a customer location as if it were a store | Stable store identifier |
| Zone | Service geography or courier operating area | Assuming zone boundaries never change | Versioned polygon or ID |
| Requested order | Order submission attempt meeting the defined eligibility rule | Assuming all attempts were accepted | Count per slot |
| Accepted order | Order the platform accepted for fulfillment | Calling it unconstrained demand | Count per slot |
| Completed order | Fulfilled order under the final status definition | Using it as the only measure of customer intent | Count per slot |
| Censored demand | Demand not observed because of availability or policy | Treating the missing quantity as known | Unobserved count |
| Exposure | Opportunity for customers to see and use an offer | Assuming every potential customer was exposed | Sessions or eligible impressions |
| Forecast vintage | Forecast as issued at a particular origin | Replacing it with the final revised value | Origin timestamp and version |
| Weather vintage | Weather forecast available at a particular origin | Using realized rainfall for an earlier decision | Issue and valid timestamps |
| Event calendar | Known planned events and their timing | Assuming attendance is certain | Event ID, location, start/end |
| Promotion | Commercial offer with scope and eligibility | Assuming a discount caused all observed uplift | ID, discount, dates, channel |
| Feature | Input to a prediction procedure | Assuming it was available in real time | Typed value with availability time |
| Target leakage | Use of information unavailable at prediction time | Limiting the concept to an obvious future column | Information-set violation |
| Label maturity | Delay before a target status is stable enough for evaluation | Training on incomplete recent outcomes | Minutes or hours after interval |
| Baseline | Simple reproducible comparator | Comparing only to a prior complex model | Orders per slot |
| Seasonal naive | Forecast based on matching earlier seasonal slots | Assuming a holiday is a normal weekday | Orders per slot |
| Hierarchy | Related aggregation levels such as city, store and slot | Summing percentiles as if they were means | Counts at defined levels |
| Reconciliation | Adjusting forecasts to meet aggregation identities | Claiming reconciliation guarantees accuracy | Same unit across levels |
| Mean | Expected value of a forecast distribution | Treating it as the most likely integer | Orders per slot |
| Quantile | Threshold with a specified cumulative probability | Treating it as a hard maximum | Orders per slot and probability |
| Prediction interval | Range intended to cover a stated share of outcomes | Confusing coverage with narrowness | Lower and upper order counts |
| Bias | Signed average forecast error under a declared sign convention | Reporting absolute error as bias | Orders per slot |
| Mean absolute error | Average absolute difference between actual and prediction | Claiming it measures timing of every peak | Orders per slot |
| Pinball loss | Quantile-specific asymmetric forecast loss | Using mean absolute error to assess all quantiles | Orders per slot |
| Calibration | Agreement between predicted probability and observed frequency | Assuming aggregate calibration implies local calibration | Proportion over comparable cases |
| Utilization | Productive demand divided by nominal available capacity | Assuming 100% utilization is a safe service level | Fraction |
| Occupancy | Fraction of available time consumed by relevant work | Ignoring breaks or non-order work | Fraction |
| Shrinkage | Paid or scheduled time unavailable for productive work | Applying it twice in the capacity formula | Fraction of paid time |
| Override | Recorded human change to a forecast or plan | Replacing the model history without a trace | Orders, workers or policy value |
| Forecast value added | Comparison of accuracy or decision loss before and after a process step | Claiming positive value from a single anecdote | Difference in error or cost |
| Drift | Change in data or performance over time | Concluding every unusual day is structural change | Metric and comparison window |
| Stockout | Required item unavailable for sale or fulfillment | Treating missing SKU sales as zero interest | Item, location and interval |
| Service level | Explicit measure of operational performance | Using the phrase without a threshold or denominator | E.g., fraction delivered on time |
| Decision loss | Cost or harm from a chosen action and realized outcome | Equating it automatically with forecast error | Currency or service impact |

Definitions are contractual, not merely glossary entries.
Before changing any definition, determine whether historical forecasts, staffing models and scorecards remain comparable.
A count of unique orders, for example, is not comparable to a count of order lines or checkout attempts.

## 5. The mechanics, step by step

### Step 1 — Specify the decision and target

**Input:** a decision request, operating calendar and fulfillment process.
**Output:** a written target contract with owner, cadence, units and deadlines.

Begin with the operational question.
For base staffing, predict orders that enter the picking queue by store and 15-minute interval.
For customer demand analysis, a submission attempt or eligible checkout may be more relevant.
For courier coverage, define the point at which an order becomes eligible for pickup.
These events can occur in different slots for the same order.

Write a contract such as: “At 16:00 local time, predict the number of accepted orders assigned to store A in each 15-minute interval from 17:00 through 18:00 local time.”
Store both local and Coordinated Universal Time (UTC) timestamps.
The target is an integer count; a forecast mean may be fractional.
Intervals are half-open, so an order at 17:15:00 belongs to 17:15–17:30, not 17:00–17:15.

Define duplicates, test orders, cancellations, reassignments and late-arriving records.
A business rule might count an order at acceptance even if it is later canceled, because it initially created work.
If cancellations are material, predict cancellation-related labor separately or add it to the service-time model.
Do not silently remove canceled orders from historical labels when the real decision incurred their workload.

**Branch:** if the requested target is latent customer intent rather than accepted orders, the data contract must include exposure and constraint records.
Observed accepted orders alone cannot identify the number of customers who would have ordered under a different policy.
Use a range or an explicit counterfactual model with assumptions; do not invent a precise lost-demand label.

### Step 2 — Map the clock and availability

**Input:** data source inventory, ingestion logs and decision deadline.
**Output:** a table of event time, publication time, ingestion time and permissible cutoff for every feature.

For each record distinguish when the event occurred, when the source published it and when the forecasting system could access it.
The safe availability time is no earlier than the latest required publication and ingestion time.
If a pipeline refreshes at 16:05, a value published at 15:55 but ingested at 16:05 is not available to a 16:00 run.

For each origin, select the latest available version whose valid period overlaps the target slot.
A weather record needs both an issue time and a valid time.
A promotion record needs planned effective dates and a version history showing when each change became known.
A scheduled event needs announcement time, location and planned timing, not only its final reported attendance.

Use a point-in-time join rather than joining current dimension tables onto historical outcomes.
Retain archived input snapshots if a provider overwrites previous forecasts.
A historical endpoint that returns the eventually observed weather is not a substitute for archived forecasts.
Where vintage data are missing, either omit that feature in the honest comparison or label the result an oracle upper-bound experiment.
Do not market the oracle score as deployable performance.

**Branch:** if publication and ingestion timestamps are unavailable, reconstruct only when defensible audit logs exist.
Otherwise mark the feature unavailable for the historical backtest and request instrumentation.

### Step 3 — Build a reliable order panel

**Input:** order event log, store calendar, timezone mapping and state transitions.
**Output:** one row per store, slot and target definition, including explicit zeroes and data-quality flags.

Generate a complete time grid for the evaluation period.
Left-join deduplicated qualifying events to that grid.
A missing row in the source is not automatically a zero: it can represent pipeline failure or a closed store.
Record whether the store was open, whether orders were accepted, and whether the feed was healthy.

When the store is fully open and the event feed is complete, zero is a legitimate observation.
When the store was closed, zero accepted orders is a valid outcome under that policy but is not evidence of zero underlying interest.
When the feed was down, the label is missing rather than zero.
A model trained to forecast accepted orders during explicitly planned closures may use closure as a known feature.
A model intended to estimate normal operations should flag or exclude closure periods according to the documented objective.

Record store boundary changes and service-zone reassignments.
A neighborhood moving from store A to store B changes both series even if customer demand is unchanged.
Either map past observations to a consistent geography where source data permit, or retain the break and treat it as a known regime change.
Never backfill with information that would not have existed at the historical forecast origin.

Handle daylight-saving changes explicitly.
A local hour can be missing in spring or repeated in autumn.
Use UTC for unique storage and local time plus offset for business seasonality.
Do not assume every local day has 96 distinct 15-minute buckets.

### Step 4 — Establish operational baselines

**Input:** mature comparable historical slots and the target contract.
**Output:** seasonal-naive and pooled baselines with documented fallbacks.

A useful first benchmark is the same store, same weekday and same local slot one week earlier.
A second is the median of several comparable weeks.
The median resists one unusual peak, but can lag a genuine trend.
For a new store, use a carefully chosen peer-store or zone baseline and record the lack of local evidence.

Compare only eligible histories that would have been known at the origin.
A week with an outage or closure may need exclusion, but apply an explicit rule rather than removing inconvenient errors by hand.
Mark holidays and large events so the evaluation can show whether a baseline fails in those conditions.
Keep the baseline available in production as a fallback when the preferred model or external feed fails.

**Branch:** if store-slot counts are sparse, pooling may be more stable than a highly granular separate model.
Estimate the hour or store total, then allocate to slots using historical shares conditional on day type.
Check shares sum to one and use a defensible fallback when the denominator is zero.
Pooling trades local detail for lower variance; measure that trade-off on the relevant horizon.

### Step 5 — Assemble calendar and recent-demand signals

**Input:** store time grid, historical order counts and known calendar attributes.
**Output:** features with explicit cutoffs and missing-value treatment.

Include local slot, weekday, public holiday indicator and time since store opening where useful.
Calendar features are usually known ahead, but a store-specific operating calendar may change; version that calendar too.
Use lagged demand only if the relevant slot has ended and its data have arrived by the forecast origin.
For example, at 16:00, the completed 15:30–15:45 count may be usable if ingestion latency permits; the final 15:45–16:00 count may not be mature.

Rolling averages must end before the origin and respect label availability.
For multi-step forecasts, do not insert actual future demand as a lag in later predicted slots.
Use recursively predicted lags, a direct model for each horizon, or only lags known at the common origin.
Compare approaches under the same information set.

A recent surge may reveal an emerging event or app change, but it can also be random noise.
A short moving average reacts quickly and is volatile; a longer one is stable and slow.
Select the window through rolling-origin evaluation rather than by inspecting the final test period.

### Step 6 — Add weather honestly

**Input:** archived weather forecast issues, store locations and a join policy.
**Output:** weather features selected as of each prediction origin.

A practical weather table contains provider, spatial grid or station, issue time, valid interval, precipitation probability, predicted precipitation amount, temperature and ingestion time.
The model may use both expected amount and probability because a low-probability heavy storm differs from a likely drizzle with the same expected amount.
Use the latest eligible issue, not the issue whose prediction later proved best.

Map each store to a relevant forecast location and preserve the mapping version.
Urban weather can vary across zones; a citywide value can miss local rain.
If the weather issue has a one-hour valid interval while demand uses 15-minute slots, document the temporal alignment rather than fabricating quarter-hour detail.

Weather effects can differ by store, time, temperature, transport mode and intensity.
Rain may increase some delivery orders while reducing others through lower mobility or service disruption.
Estimate relationships on comparable history and show uncertainty; do not assume every rainy slot increases demand.
A weather forecast can also affect courier travel time and availability, so a demand increase does not imply supply remains unchanged.

**Branch:** when the provider feed is late or missing, use a predetermined fallback such as the most recent still-valid issue or a model without weather.
Expose the missingness flag to operators and compare fallback performance separately.
Do not replace a missing forecast with realized future weather.

### Step 7 — Represent events and promotions

**Input:** versioned event calendar, campaign plans, location and customer eligibility.
**Output:** known-at-origin event and promotion features with scope.

For events, capture venue, expected start and end, relevant distance or catchment, type, announcement time and expected attendance if published.
Use post-event actual attendance only for retrospective analysis, not as a feature before it was known.
A concert ending may generate a demand spike after the listed end; learn a time window from historical data and validate it out of time.
A canceled event must only affect forecasts after the cancellation became available.

For promotions, capture campaign identifier, valid period, eligible store or zone, channels, discount and any spend threshold.
A planned campaign can differ from delivered impressions because of customer targeting or budget limits.
At origin, use planned exposure or the cumulative exposure already observed, not final impressions from the entire campaign.
Track promotion interactions with normal peak periods and capacity caps.
A large predicted uplift may be unservable if the app restricts slots, which changes observed accepted orders.

Promotions are usually targeted rather than randomized.
A high sales week under promotion does not prove the entire difference is incremental.
The campaign may have been launched because demand was already expected to rise.
Use experiments or an appropriately controlled causal design to estimate incrementality; forecasting association and causal impact are different tasks.

### Step 8 — Fit distributions and maintain aggregation identities

**Input:** as-of feature panel, training windows and business hierarchy.
**Output:** point forecasts, quantile or scenario forecasts, and reconciliation diagnostics.

Start with the baseline before adding a regression or count model.
A count model can impose nonnegative expected demand and use calendar, weather and campaign features.
A flexible machine-learning model may capture nonlinear effects, but complexity is justified only by honest improvement and operational reliability.
Sparse counts, changing assortment and short store histories make elaborate local models particularly easy to overfit.

For deterministic additive examples, the expected hourly total equals the sum of expected slot counts when all slots are included.
A top-down split can multiply an hourly mean by slot shares that sum to one.
Round only after aggregation if integer displays are required; retain decimal means internally.

Quantiles do not generally add.
The sum of four slot 90th percentiles is not the 90th percentile of the hour unless a special dependence structure makes it so.
Likewise, independently sampling slots can understate a common weather or event shock.
For hourly risk, model or sample the joint distribution and sum each simulated trajectory before computing the hourly quantile.
State how correlation is estimated and test coverage at both slot and hour levels.

A quantile model may produce crossing quantiles, such as a predicted 90th percentile below its median.
Detect crossing and apply an approved monotonic correction or use a distributional model that enforces order.
Report the correction and its effect on evaluation.

### Step 9 — Backtest at real decision horizons

**Input:** frozen historical source vintages, maturity rules and candidate methods.
**Output:** predictions generated at rolling origins, scored by horizon and operating segment.

Choose training cutoffs and test origins in chronological order.
At each origin, fit or update only using labels mature by then.
Build features with the same as-of joins and data latency that production would have used.
Forecast the exact future slots relevant to the decision, save predictions, and advance the origin.

Separate model selection from the final holdout.
If you repeatedly tune on the final month, it becomes training information even if its rows were not used for fitting.
Use a development period for feature choices and a later untouched period for final evaluation.
Include multiple weekdays, seasonal conditions and promotion regimes where historical coverage permits.

For a previous-day schedule, use previous-day origins; for an intraday staffing call, use intraday origins.
Score each separately because the second forecast has more information and a different action set.
If the staffing rule was updated during the backtest, either freeze it or simulate its historical version.
A hypothetical policy evaluated with future knowledge is not evidence of a deployable process.

**Branch:** if an event type appears only once, do not present a confident event-specific accuracy estimate.
Report the count, uncertainty and a sensitivity analysis, and use a conservative operational contingency.

### Step 10 — Evaluate accuracy and business impact

**Input:** frozen forecasts, mature actuals, benchmark predictions and decision outcomes.
**Output:** a segmented scorecard and an explanation of uncertainty.

Define error consistently, for example forecast minus actual.
Mean error then reveals systematic overforecasting when positive and underforecasting when negative.
Mean absolute error (MAE) is the average absolute error in orders per slot and is easy to interpret.
Weighted absolute percentage error divides the sum of absolute errors by the sum of actual orders; it is undefined when that denominator is zero and can obscure performance in low-volume stores.
Avoid ordinary percentage errors for individual zero-actual slots.

For a predicted quantile `q`, use pinball loss: `max(q*(y-f), (q-1)*(y-f))`, where `y` is actual and `f` is the predicted quantile.
For a 90th percentile forecast, underprediction costs nine times as much as equally sized overprediction in this scoring rule.
The weights evaluate the quantile, not the actual business cost unless the decision's cost ratio supports that choice.

Coverage is the fraction of actuals at or below a predicted upper quantile.
Because counts are discrete and forecasts may be rounded, empirical coverage can be above the nominal percentage even under a well-specified model.
Assess interval width and pinball loss as well as coverage; a huge upper bound can achieve high coverage without being useful.
Report sample size and uncertainty, especially for rare rainy-event combinations.

Segment by store, slot, horizon, weekday, promotion, weather and availability status.
Compare the distribution of errors during peak periods with the distribution over all periods.
A model can win overall by predicting quiet intervals and still fail the dinner decision.
Also evaluate decision loss using actual staffing constraints and a clearly defined cost model.
Keep the forecast score separate from service outcomes influenced by dispatch, traffic or execution.

### Step 11 — Convert workload into a constrained staffing plan

**Input:** slot-level demand distribution, service-time assumptions, paid-time availability, labor constraints and service objective.
**Output:** recommended roster, alternatives and a record of residual risk.

Translate orders to workload in minutes using basket mix and observed picker service time.
If mean service time is `s` minutes per order and the demand scenario is `d` orders, required productive work is `d*s` minutes.
In a 15-minute slot, one continuously productive picker supplies at most 15 productive minutes before breaks, non-order tasks and a utilization buffer.
If productive availability after shrinkage is `a` and the chosen maximum occupancy is `u`, the planning capacity per scheduled picker is `15*a*u` productive minutes.
Round the resulting headcount up, then apply shift-length, break, labor availability and skill constraints.

Do not subtract the same break in both `a` and `s`.
Specify whether the service-time estimate includes walking, handoff, substitutions and rework.
If basket size changes under promotion, a fixed service time may be inappropriate.
Use task-specific workload or scenario-specific service time when evidence supports it.

A simple occupancy buffer is a planning approximation, not a queueing guarantee.
Arrival clustering, variable service times, finite waiting space and carryover from earlier slots can still create lateness below the chosen occupancy.
Use queue simulation or empirical service curves before claiming a particular on-time probability.
A staffing decision should account for existing backlog at the beginning of a slot.

For courier capacity, translate orders into courier-minutes using route length, handoff, batching and travel conditions.
Courier-minutes are not interchangeable with picker-minutes.
If demand and supply both worsen in rain, stress both sides of the capacity calculation together.
Workers' safety, breaks and legal arrangements are constraints, not residuals to be optimized away.

**Branch:** if the recommended headcount exceeds available or permitted staff, present a gap and feasible options.
Options include an approved flex call, an adjusted promise, a slot cap, or a promotional change.
Each has a different owner and customer impact; the forecaster does not unilaterally choose among them.

### Step 12 — Publish, monitor and learn

**Input:** forecast package, staffing recommendation, data-quality flags and operating thresholds.
**Output:** versioned decision record, live dashboard and review actions.

Publish the origin, target, model version, mean, selected quantiles, baseline, weather vintage and major feature flags.
Record whether a forecast was unavailable or a fallback was used.
Show the recommended action separately from the forecast and from any human override.
Name an accountable owner and a time at which the recommendation becomes stale.

Monitor pipeline freshness before monitoring model error.
A perfect model on stale data is not a functioning forecasting service.
After labels mature, monitor bias, MAE, quantile loss, coverage and operational decisions by meaningful segment.
For small samples, use rolling windows and avoid interpreting random variation as drift.
Investigate changes in demand mix, store boundaries, campaigns, app exposure and logging before retraining automatically.

Maintain an override ledger with author, timestamp, reason, original value, adjusted value and evidence.
Compare model-only and overridden forecasts on the same slots.
An override that helps service but worsens forecast accuracy may actually be a capacity buffer and belongs in the decision layer.
Retain it as a policy if it has value, but do not rewrite the original prediction to make the model look more accurate.

## 6. Worked example

### Case and information set

A fictional dark store, Store A, is planning the 17:00–18:00 local-time dinner period on a Thursday.
At 16:00, the shift manager can request approved flex pickers for the hour.
The forecast covers four consecutive 15-minute slots.
Orders are counted when accepted and assigned to Store A; cancellations after acceptance remain in the count.
The store is scheduled to remain open and accept orders throughout the hour.

A previous-day base roster is already published.
The 16:00 recommendation can add flex capacity but cannot revise the guaranteed shifts already in place.
The forecast is not a customer promise and does not itself open or close a slot.
The shift manager owns the staffing request; marketplace control owns any slot cap.

The historical comparable-week medians are illustrative, measured in accepted orders per 15-minute slot.
A promotion scheduled before 16:00 covers the entire hour.
The latest weather forecast ingested by 16:00 predicts rain during the last three slots.
An event near the service area is expected to release customers around 17:30.
No causal claim is made for the adjustments: they are simple forecast components for teaching.

### Inputs and forecast calculation

| Local slot | Comparable median | Promotion adjustment | Weather adjustment | Event adjustment | Mean forecast |
|---|---:|---:|---:|---:|---:|
| 17:00–17:15 | 12.0 orders | +1.2 orders | +0.0 orders | +0.0 orders | 13.2 orders |
| 17:15–17:30 | 14.0 orders | +1.4 orders | +1.0 orders | +0.0 orders | 16.4 orders |
| 17:30–17:45 | 16.0 orders | +1.6 orders | +1.2 orders | +2.0 orders | 20.8 orders |
| 17:45–18:00 | 13.0 orders | +1.3 orders | +1.0 orders | +1.0 orders | 16.3 orders |
| Hour total | 55.0 orders | +5.5 orders | +3.2 orders | +3.0 orders | 66.7 orders |

The promotion component is 10% of the comparable median for each slot.
For 17:30–17:45, `16.0 + 1.6 + 1.2 + 2.0 = 20.8` orders.
The hour's mean is `13.2 + 16.4 + 20.8 + 16.3 = 66.7` orders.
Linearity of expectation allows this sum even if slot outcomes are correlated.
These adjustments are assumptions chosen for this worked example, not estimates taken from real operations.

Assume a separately validated probabilistic model supplies the following upper quantiles.
They are inputs to the staffing example, not computed by the additive calculation above.
The quantiles refer to the same accepted-order target and the same 16:00 origin.

| Local slot | Mean | 90th percentile | Interpretation |
|---|---:|---:|---|
| 17:00–17:15 | 13.2 orders | 17 orders | Approximately 90% of comparable outcomes are at or below the threshold if calibrated |
| 17:15–17:30 | 16.4 orders | 21 orders | A higher threshold allows for order variation |
| 17:30–17:45 | 20.8 orders | 27 orders | The largest forecast workload is in this slot |
| 17:45–18:00 | 16.3 orders | 21 orders | Demand is expected to decline after the peak |

Do not add 17, 21, 27 and 21 and call 86 orders the 90th percentile of the hour.
The joint slot distribution is needed to find that hourly percentile.
The thresholds also do not imply that 90% of entire four-slot trajectories stay below all four thresholds simultaneously.

### Picker workload and recommendation

Assume each accepted order creates 3.0 productive picker-minutes on average in this promotion mix.
This number includes walking, picking and handoff but excludes breaks and other non-order work.
Assume 90% of scheduled time is available after breaks and other shrinkage.
The operations policy also limits planned occupancy to 80% of that available time to provide a buffer.
One scheduled picker therefore supplies `15 × 0.90 × 0.80 = 10.8` planning minutes per 15-minute slot.
The two fractions apply to different reductions: availability first, then the occupancy buffer.

| Local slot | 90th-percentile orders | Workload at 3.0 min/order | Raw headcount | Rounded recommendation |
|---|---:|---:|---:|---:|
| 17:00–17:15 | 17 orders | 51.0 min | 51.0 / 10.8 = 4.72 | 5 pickers |
| 17:15–17:30 | 21 orders | 63.0 min | 63.0 / 10.8 = 5.83 | 6 pickers |
| 17:30–17:45 | 27 orders | 81.0 min | 81.0 / 10.8 = 7.50 | 8 pickers |
| 17:45–18:00 | 21 orders | 63.0 min | 63.0 / 10.8 = 5.83 | 6 pickers |

Assume six pickers are already scheduled for the full hour and two approved flex pickers can work the full hour.
The feasible recommendation is eight pickers for the hour if the peak-oriented rule is chosen.
That plan provides capacity above the slot-specific minimum in the other three slots.
The manager may assign spare time to approved replenishment or non-order tasks, provided these tasks can be interrupted for the peak.
The roster recommendation is not a guarantee of on-time service.
The 90th-percentile thresholds and occupancy buffer have not been proven to imply a 90% delivery service level.

An alternative is to retain six pickers and reduce accepted demand during the peak.
At six pickers, planning capacity is `6 × 10.8 = 64.8` productive minutes per slot, equivalent to `64.8 / 3.0 = 21.6` orders under the simplified mean-workload assumption.
That is not a hard cap on the number that can physically be handled; it is a policy planning capacity with a buffer.
The upper demand scenario of 27 orders exceeds this planning capacity by `27 - 21.6 = 5.4` orders.
A slot cap, if used, requires marketplace approval and must account for existing backlog and promised orders.

Suppose a flex picker costs an incremental USD 18 per hour and two are needed for one hour.
The incremental direct staffing cost is `2 × USD 18 = USD 36` before any additional overhead.
This is a decision cost, not a forecast error.
A cost-benefit comparison would also need credible estimates of avoided late orders, cancellations, customer harm and worker impact.
Those values are not supplied, so the example cannot establish that the flex choice is economically optimal.

### Outcome and review

Suppose the realized accepted-order counts are 15, 19, 25 and 18 orders.
The realized hourly count is `15 + 19 + 25 + 18 = 77` orders.
Using forecast minus actual, the slot errors are −1.8, −2.6, −4.2 and −1.7 orders.
Mean signed error is `(−1.8 − 2.6 − 4.2 − 1.7) / 4 = −2.575` orders per slot.
All four errors are negative, so this small example underforecast each slot.
The MAE is `(1.8 + 2.6 + 4.2 + 1.7) / 4 = 2.575` orders per slot.
The hourly error is `66.7 − 77 = −10.3` orders.

Each actual is at or below its respective 90th-percentile threshold in this one example.
That is four covered slots out of four, not proof of 90% calibration.
With only four observations the coverage estimate is too imprecise for a credible reliability conclusion.
Analyze many rolling origins, and examine comparable event and weather conditions separately when there are enough cases.

The six-person plan would have been below the simplified peak requirement of `25 × 3 / 10.8 = 6.94`, rounded to seven pickers.
The eight-person plan meets that simplified requirement, but actual service may still fail because service times vary and orders can cluster.
Do not attribute all on-time performance to the forecast without examining backlog, labor attendance, picking productivity and dispatch.

### Sensitivity and ownership

If the average productive service time rises from 3.0 to 3.6 minutes per order, the peak upper-scenario workload becomes `27 × 3.6 = 97.2` minutes.
At 10.8 planning minutes per picker, that requires `97.2 / 10.8 = 9.0`, or nine pickers.
The available eight-person plan then has a one-picker gap under this scenario.
The manager should escalate rather than present eight as sufficient under both assumptions.

The demand forecaster owns the forecast, assumptions and data-vintage record.
The labor manager owns the roster, attendance and lawful scheduling constraints.
The marketplace lead owns any customer-facing cap or promise adjustment.
The courier lead separately owns delivery capacity; the picker calculation does not certify courier coverage.
The commercial lead owns promotion changes and their customer communication.

## 7. Code, query, or calculation

The following standard-library Python program reproduces the worked forecast, scoring and staffing arithmetic.
Save it as `forecast_example.py` and run `python forecast_example.py`.
It uses no external files or packages and does not fit a model.
The upper quantiles and service assumptions are illustrative inputs, not statistically estimated by the program.
It also prints a simple as-of weather selection to demonstrate information availability.
Timestamps are ISO 8601 strings with a common UTC offset, which makes lexical comparison safe in this small example.
Production code should parse timezone-aware timestamps and handle multiple locations and valid intervals.

```python
from math import ceil

slots = [
    {
        "slot": "17:00-17:15",
        "base": 12.0,
        "weather": 0.0,
        "event": 0.0,
        "q90": 17,
        "actual": 15,
    },
    {
        "slot": "17:15-17:30",
        "base": 14.0,
        "weather": 1.0,
        "event": 0.0,
        "q90": 21,
        "actual": 19,
    },
    {
        "slot": "17:30-17:45",
        "base": 16.0,
        "weather": 1.2,
        "event": 2.0,
        "q90": 27,
        "actual": 25,
    },
    {
        "slot": "17:45-18:00",
        "base": 13.0,
        "weather": 1.0,
        "event": 1.0,
        "q90": 21,
        "actual": 18,
    },
]

promotion_rate = 0.10
service_minutes = 3.0
slot_minutes = 15.0
availability = 0.90
occupancy = 0.80
capacity = slot_minutes * availability * occupancy

forecasts = []
errors = []
for row in slots:
    forecast = (
        row["base"]
        + row["base"] * promotion_rate
        + row["weather"]
        + row["event"]
    )
    required = ceil(row["q90"] * service_minutes / capacity)
    error = forecast - row["actual"]
    forecasts.append(forecast)
    errors.append(error)
    print(
        f'{row["slot"]}: mean={forecast:.1f}, '
        f'q90={row["q90"]}, pickers={required}'
    )

print(f'Hour mean: {sum(forecasts):.1f} orders')
print(f'Hour actual: {sum(row["actual"] for row in slots)} orders')
print(f'Mean error: {sum(errors) / len(errors):.3f} orders/slot')
print(f'MAE: {sum(abs(e) for e in errors) / len(errors):.3f} orders/slot')

# Availability is when both publication and ingestion have happened.
origin = "2026-10-01T16:00:00+00:00"
weather_issues = [
    {
        "published": "2026-10-01T14:00:00+00:00",
        "ingested": "2026-10-01T14:05:00+00:00",
        "valid_start": "2026-10-01T17:00:00+00:00",
        "valid_end": "2026-10-01T18:00:00+00:00",
        "rain_probability": 0.4,
    },
    {
        "published": "2026-10-01T15:45:00+00:00",
        "ingested": "2026-10-01T15:50:00+00:00",
        "valid_start": "2026-10-01T17:00:00+00:00",
        "valid_end": "2026-10-01T18:00:00+00:00",
        "rain_probability": 0.7,
    },
    {
        "published": "2026-10-01T15:55:00+00:00",
        "ingested": "2026-10-01T16:05:00+00:00",
        "valid_start": "2026-10-01T17:00:00+00:00",
        "valid_end": "2026-10-01T18:00:00+00:00",
        "rain_probability": 0.9,
    },
]
target_start = "2026-10-01T17:00:00+00:00"
target_end = "2026-10-01T17:15:00+00:00"
eligible = [
    issue
    for issue in weather_issues
    if issue["published"] <= origin
    and issue["ingested"] <= origin
    and issue["valid_start"] <= target_start
    and issue["valid_end"] >= target_end
]
if eligible:
    selected = max(eligible, key=lambda issue: issue["published"])
    print(f'Available rain probability: {selected["rain_probability"]:.1f}')
else:
    print('Available rain probability: missing; use approved fallback')
```

Expected output:

```text
17:00-17:15: mean=13.2, q90=17, pickers=5
17:15-17:30: mean=16.4, q90=21, pickers=6
17:30-17:45: mean=20.8, q90=27, pickers=8
17:45-18:00: mean=16.3, q90=21, pickers=6
Hour mean: 66.7 orders
Hour actual: 77 orders
Mean error: -2.575 orders/slot
MAE: 2.575 orders/slot
Available rain probability: 0.7
```

The 15:55 weather issue is excluded because it was not ingested until 16:05.
The 15:45 issue is the most recently published eligible issue.
This example selects one issue covering the full target slot; a real pipeline needs a deliberate policy for partial overlaps and revised valid intervals.
It must also log missing data, provider revisions and the precise timezone of each timestamp.

## 8. Data, standards, and regulatory context

### Operational data contract

The minimum order-event record includes a stable order identifier, store assignment, event type, event timestamp, ingestion timestamp and status version.
For auditability, keep the history of reassignments and corrections rather than only the final order state.
A store-slot label can then be reconstructed under the target definition in force for that experiment.
A label's finalization timestamp must be available to the training pipeline, not inferred from the current database state.

Weather data require provider and location identifiers, issuance time, valid interval, ingestion time and units.
Precipitation probability is a probability, not a millimeter amount.
Temperature must include its scale, and rainfall amounts must include their accumulation interval.
A daily rainfall total cannot be safely assigned to a 15-minute slot without an explicit assumption.

Promotion data need campaign versions, applicable customers, stores, products, dates, channels and monetary terms.
Event records should distinguish planned schedule from subsequently observed attendance or cancellation.
Availability data should capture closures, slot caps, promised-time changes, stockouts, app outages and delivery-zone eligibility.
No single event stream necessarily records every reason a customer did not order.

### Time and identifiers

ISO 8601 provides a standard for representing dates and times; the standard does not by itself choose your business timezone or resolve ambiguous local timestamps.
Store UTC instants alongside the local zone identifier and offset used for slot assignment.
Use a versioned calendar for changes in opening hours and service boundaries.
Retain the forecast origin and the target interval as separate fields.

Global Trade Item Number (GTIN) identifiers from GS1 can help align product data across trading partners.
A GTIN identifies a trade item, not a customer, order, store workload or unique physical picking event.
Where product-level data feed a workload model, document local SKU-to-GTIN mappings and changes in pack size.
GS1 conventions do not determine the correct forecasting target or replace local data-quality controls.

### Privacy and employment

Customer addresses and precise order locations can be personal data.
The European Union General Data Protection Regulation (GDPR), where applicable, imposes requirements including purpose limitation, data minimization and security; see Articles 5 and 32 in the official text.
The regulation's applicability depends on the processing context and territorial scope.
This educational discussion is not legal advice, and it does not assert that every aggregate store-slot forecast is itself personal data.
Apply access controls, retention limits and aggregation appropriate to the local legal and operational assessment.

Employee and courier availability records may also be sensitive.
Minimize individual-level attributes in forecasting datasets and separate staffing eligibility from protected or irrelevant personal information.
Employment status, scheduling, working time and safety rules vary by jurisdiction and contract.
A forecast or mathematical utilization target does not override those rules.
Have qualified local reviewers define binding constraints before deployment.

### Audit and provenance

Keep a reproducible prediction record: origin, target, training cutoff, model artifact, input versions, forecast, uncertainty and decision recipient.
Log changes to feature definitions, locations and calendars.
Keep a separate immutable record of human overrides and their rationale.
A dataset snapshot without source publication times is insufficient to prove that a backtest was leakage-free.

Access to historical weather forecast archives may be licensed or limited.
The actual weather observation is not equivalent to the forecast that was issued earlier.
When a source cannot provide past vintages, report this limitation and omit it from the deployable historical comparison.
Retention and licensing requirements should be reviewed before copying third-party weather or event data into long-lived training archives.

### Data-quality controls

| Control | Detection | Operational response |
|---|---|---|
| Late order feed | Last ingestion timestamp exceeds freshness threshold | Flag recent lags and switch to approved baseline |
| Missing weather issue | No eligible issue covers target | Use documented fallback and show missing-data flag |
| Duplicate order event | Repeated stable event ID or impossible state transition | Deduplicate under contract; investigate source |
| Store closure mismatch | Calendar says open while app acceptance is disabled | Quarantine label for normal-operation modeling |
| Invalid slot time | Ambiguous local time lacks offset | Resolve from UTC or exclude pending correction |
| Changed service area | Geometry or assignment version changes | Mark regime break and reassess comparability |
| Late promotion revision | Version publication follows origin | Exclude revision from that historical prediction |
| Immature label | Status still within correction window | Delay scoring or mark provisional |
| Missing exposure | No impression or eligibility data during a cap | Do not infer unconstrained demand from accepted orders |
| Unmapped product | SKU changes or pack conversion absent | Flag workload estimate and repair mapping |

Data-quality controls are part of the business process.
A warning that nobody owns is not a control.
Assign an on-call team, response time and fallback action for each critical input.

## 9. Industry practice and case evidence

### Weather observations and forecast archives

The U.S. National Oceanic and Atmospheric Administration's National Weather Service publishes forecasts and documents access through its API.
Its forecasts are issued for particular locations and valid periods; applications need to preserve issue and retrieval times to reconstruct the information set.
The existence of a public forecast does not establish that a particular retailer used it or that weather improves every retailer's model.
For this unit, the practical inference is that weather must be treated as a time-stamped forecast product, not as eventual truth.

Weather affects both customer decisions and delivery operations.
A local rain forecast can be informative about order volume, but the direction and magnitude vary across settings.
It can also change travel times and the number of couriers willing or able to work.
An operator should evaluate the coupled demand and supply scenarios rather than extrapolate from an aggregate correlation.
That recommendation is an operational inference, not a claim about a named company's internal model.

### Forecasting competitions and point-in-time evaluation

The M5 forecasting competition used retail sales data and hierarchical series to assess forecasting approaches.
Its published guide describes a large-scale retail forecasting setting and the relevance of evaluation and hierarchy.
It is not a quick-commerce experiment and does not establish that its winning method is best for 15-minute delivery slots.
The transferable lesson is to compare methods on a fixed target and a fair out-of-sample evaluation, while adapting the operational horizon and granularity.

The M4 competition similarly demonstrates the value of evaluating forecasting methods across many series and using benchmarks.
Its datasets and scoring setup do not recreate app exposure, store closures or courier constraints.
An operator still needs a rolling-origin backtest using its own realistic data availability and actions.
A published competition ranking cannot substitute for that test.

### Instacart public dataset

Instacart released a public anonymized grocery-order dataset for a Kaggle competition.
The dataset is useful for exploring baskets, reorder patterns and order timing at the fields it actually provides.
It does not expose the complete historical decision-time weather vintages, staffing decisions, lost demand or delivery-system constraints needed to validate this unit's end-to-end example.
Treat it as an educational data source, not as proof of a short-horizon staffing improvement.
Any derived analysis must respect its terms and carefully check the resolution of the provided timestamps.

### Evidence and generalization

| Evidence | What it supports | What it does not establish |
|---|---|---|
| National Weather Service API documentation | Forecast products have identifiable temporal and geographic context | A universal rain uplift for delivery orders |
| M5 competition guide | Hierarchical retail forecasting and standardized comparisons are practical | A specific quick-commerce slot model's causal or business impact |
| M4 competition paper | Benchmarking across many forecasting series is valuable | A deployed store's staffing return on investment |
| Instacart public competition data | Public anonymized order data can support basket and timing exercises | Counterfactual lost demand or actual staffing adequacy |
| EU GDPR official text | Applicable legal obligations must be assessed from the text and context | A universal rule that all aggregated demand data are personal |

Named examples provide context rather than a claimed case study of a successful staffing rollout.
No numeric business outcome is attributed to any named organization here.
Local operators should run prospective or carefully designed retrospective evaluations before generalizing.

### Operating practice to adopt

Maintain a baseline and a probabilistic alternative side by side.
Give users a frozen forecast and the evidence available when it was made.
Keep demand, workload and staffing as separate layers so a productivity change is not mistaken for demand drift.
Review losses around peaks, outages and promotions, not just the all-day average.
Use operational exceptions to trigger a decision discussion rather than an automatic promise to the customer.

A disciplined rollout can begin in shadow mode.
Compare the proposed recommendation with the actual decision using historical decisions and documented assumptions.
Shadow evaluation cannot reveal the full counterfactual effect of a staffing change on acceptance, service or customer behavior.
A controlled operational pilot, if feasible and ethically approved, provides stronger evidence than a retrospective correlation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Realized weather in backtest | Weather feature matches later observations unusually well | Future weather substituted for forecast vintage | Inflated accuracy | Preserve issue and ingestion times; rerun as-of joins |
| Promotion revision leakage | Historical campaign rows show later cancellations | Current campaign table overwrites old plans | Unrealistic retrospective knowledge | Version plans and select at origin |
| Censored order labels | Accepted volume flattens exactly at slot caps | Policy limits observed acceptance | Underestimated underlying demand | Log caps and exposure; identify constrained cases |
| Data-feed gap treated as zero | Simultaneous zeroes across open stores | Missing events filled with zero | Spurious trough and biased model | Monitor feed completeness and distinguish missing from zero |
| Immature cancellations | Recent counts change after scoring | Labels revised after evaluation | Unstable performance claims | Define maturity delay and rescore only under policy |
| Random train-test split | Validation score far better than live score | Future regimes and nearby observations leak | False model selection | Use rolling chronological origins |
| Target mismatch | Staffing rises despite stable predicted orders | Orders forecasted at a different event than picker work | Capacity decision uses wrong clock | Define queue-entry target and reconcile event times |
| Misleading percentage metric | Large metric swings in quiet slots | Actual count near or equal to zero | Unstable ranking | Report MAE and aggregate denominators with zero handling |
| Quantiles added | Hourly upper bound equals sum of slot bounds | Dependence ignored | Wrong risk estimate | Use joint trajectories and evaluate hourly coverage |
| Double-counted breaks | Capacity far lower than observed output | Break time removed from both productivity and availability | Excess staffing | Define productive time and audit the formula |
| Ignored basket mix | Order forecast accurate but pick queue grows | Promotion changes items or handling time per order | Understaffing despite accurate demand | Forecast workload or apply mix-specific service times |
| Unmodeled backlog | First peak slot already has a queue | Capacity calculation assumes empty starting system | Late fulfillment | Include opening backlog and carryover |
| Geography drift | Store counts shift at zone change | Customers reassigned between stores | Apparent model drift | Version boundaries and mark regime breaks |
| Wrong daylight-saving bucket | Duplicate or missing local hour | Local timestamp used without offset | Incorrect lags and staffing | Store UTC plus local timezone and offset |
| Availability feedback loop | Repeated low forecasts for capped slots | Suppressed observations become training truth | Persistent undercapacity | Track exposure, caps and counterfactual limitations |
| Unrecorded override | Published forecast differs from model artifact | Human adjustment overwrites original | No forecast value-added audit | Retain both values, author, timestamp and reason |
| Aggregate-only calibration | Overall 90% coverage but peaks fail | Quiet slots dominate the score | Understaffing when it matters | Evaluate by peak, store and horizon with sample counts |
| Forecast treated as guarantee | Staff promised for all possible outcomes | Upper quantile interpreted as maximum | Unplanned service failures | Explain residual tail risk and contingency |
| Courier/picker conflation | Adequate pick capacity but delivery delays persist | Different bottleneck and workload unit | Customer promise missed | Model courier capacity separately |
| Extrapolated event effect | Large response to a novel event type | Sparse history and overfitted coefficients | Costly inaccurate forecast | Pool carefully, show uncertainty and use scenarios |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Comparable weeks represent normal demand | Supplies a baseline | Holidays, store change or structural trend | Compare matched weeks and mark known regime changes |
| Event timing is known by origin | Makes event feature usable | Late announcement or cancellation | Check announcement and ingestion timestamps |
| Weather issue is archived | Enables honest as-of join | Provider overwrites forecast history | Archive prospectively; omit feature from unsupported backtest |
| Promotion applies to the stated store and slot | Allows uplift to be allocated | Eligibility or budget changes | Compare versioned campaign scope with delivery logs |
| Accepted orders reflect the planning workload | Aligns target and picker decision | High cancellation, reassignments or unaccepted attempts create work | Reconcile queue events and service-time logs |
| Store remains open | Supports accepted-order forecast for normal operations | Closure, cap or app outage | Include known closures and flag unexpected constraints |
| Three picker-minutes per order in example | Converts count into workload | Large baskets, substitution or walking-time changes | Measure service time by basket mix and store |
| Availability and occupancy are distinct | Prevents double counting | Productivity estimate already excludes or includes buffer differently | Trace numerator and denominator of each rate |
| Slot upper quantile is calibrated | Supports a risk-oriented scenario | Distribution shift or sparse tail data | Backtest pinball loss and coverage by segment |
| Slot workload is independent of earlier backlog in simplified table | Permits a one-slot arithmetic illustration | Queue carries over from prior period | Add backlog dynamics or simulate the queue |
| Flex workers are actually available | Makes the proposed roster feasible | Late refusal or legal scheduling restriction | Confirm eligibility and run an alternative plan |
| Each flex worker costs USD 18 per hour | Allows an illustrative cost calculation | Premiums, overhead or minimum shift length differ | Use local incremental cost and contract rules |
| Hourly mean is the sum of slot means | Follows from consistent, exhaustive slot definitions | Overlapping slots or different target definitions | Check bucket boundaries and aggregation identities |
| Weather and promotion effects are additive in the example | Keeps calculation transparent | Interaction or nonlinear saturation | Test interactions with enough historical support |
| Observed service outcomes identify forecast value | Needed for a naive before-after comparison | Concurrent staffing, traffic and policy changes | Separate predictive scoring from controlled decision evaluation |

A model can be calibrated under one app policy and miscalibrated after that policy changes.
The target distribution is conditional on the operating environment, not an immutable property of customers.
If the model is deployed to set caps, the resulting data-generating process changes.
Plan a monitoring and experimentation strategy before drawing causal conclusions from post-deployment results.

### What the worked percentile cannot decide

The 90th-percentile input in the example is a policy choice, not a universal staffing standard.
An appropriate risk threshold depends on the relative consequences of excess and insufficient capacity.
It also depends on whether the organization can add capacity later, reduce intake, or carry backlog safely.
If those options differ by time of day, a single percentile for every slot may be inappropriate.

For a simplified single-period decision, consider the marginal underage cost of one additional unit of unmet workload and the marginal overage cost of one unused unit of capacity.
When costs are linear and all other standard single-period assumptions hold, their ratio suggests a critical fractile.
For example, if one unit of shortage costs USD 9 and one unused unit costs USD 1, the critical fractile is `9 / (9 + 1) = 0.90`.
That arithmetic provides a rationale for considering a 90th percentile; it does not validate the worked example's assumed threshold.
The costs must refer to the same workload unit and decision interval.

Real rosters violate the simplified rule in several ways.
Workers are scheduled in indivisible shifts rather than independent 15-minute increments.
Backlog carries forward, service times vary, and the next slot's demand is correlated with the current slot.
Safety and contractual restrictions may prohibit a mathematically cheapest roster.
Therefore, use the critical-fractile calculation as a diagnostic and compare feasible roster scenarios using a more complete model.

When reliable cost estimates are unavailable, present a small set of scenarios rather than inventing a cost ratio.
For example, show the roster, paid hours, simulated lateness and remaining tail risk for a median-oriented plan and a higher-capacity plan.
The decision owner should record the chosen trade-off and the evidence available at the time.
Revisit the choice when observed service, staff availability or operating policy changes.

## 12. Exercises

### Exercise 1 — Write the target contract

**Time:** 12 minutes.  
**Type:** Recall and specification.  
**Scenario:** At 14:00, a manager schedules pickers for 17:00–18:00 in four 15-minute slots.  
**Deliverable:** A five-line target contract.  
**Acceptance criteria:** State origin, location, count event, half-open intervals and treatment of cancellations; distinguish forecast from roster.

### Exercise 2 — Select available features

**Time:** 15 minutes.  
**Type:** Application and leakage detection.  
**Scenario:** At a 16:00 origin, weather issue A was published at 15:30 and ingested at 15:40; issue B was published at 15:55 and ingested at 16:05.
A campaign cancellation was published at 16:10, after the origin.  
**Deliverable:** Identify the usable weather issue and whether the cancellation is usable.  
**Acceptance criteria:** Explain both publication and ingestion cutoffs; state what to do if neither weather issue covers the target interval.

### Exercise 3 — Compute an adjusted slot forecast

**Time:** 10 minutes.  
**Type:** Calculation.  
**Scenario:** The comparable-slot baseline is 18.0 orders; promotion uplift is 10% of baseline; expected weather and event adjustments are +1.5 and −0.5 orders.  
**Deliverable:** Calculate the mean forecast and label the components.  
**Acceptance criteria:** Show arithmetic and units; explain that a negative event adjustment is possible and is not automatically causal.

### Exercise 4 — Evaluate slot errors

**Time:** 12 minutes.  
**Type:** Calculation.  
**Scenario:** Forecasts are 10, 15 and 20 orders; actuals are 12, 14 and 23 orders for three equal slots.  
**Deliverable:** Calculate signed errors using forecast minus actual, mean error and MAE.  
**Acceptance criteria:** Show all three errors and retain units; distinguish signed bias from absolute error.

### Exercise 5 — Design a rolling-origin evaluation

**Time:** 20 minutes.  
**Type:** Design.  
**Scenario:** Compare a previous-day roster forecast and a 60-minute-ahead flex forecast over 12 weeks.  
**Deliverable:** Draw the training, validation and final evaluation timeline and list the data allowed at each origin.  
**Acceptance criteria:** Include distinct horizons, label maturity, weather vintages, a seasonal baseline and an untouched final period; avoid random splitting.

### Exercise 6 — Check a quantile

**Time:** 15 minutes.  
**Type:** Application.  
**Scenario:** A model's 90th-percentile forecast is 20 orders and the actual is 24 orders for one slot.  
**Deliverable:** Calculate pinball loss and explain what a 90th percentile means.  
**Acceptance criteria:** Show the asymmetric-loss formula; do not claim one observation proves calibration; mention the discrete-count qualification.

### Exercise 7 — Convert demand to staffing

**Time:** 15 minutes.  
**Type:** Calculation and decision.  
**Scenario:** A 15-minute slot has a selected demand scenario of 24 orders, with 3.0 productive minutes per order.
Availability is 80% and maximum planning occupancy is 75%.  
**Deliverable:** Calculate the rounded picker requirement and identify one operational caveat.  
**Acceptance criteria:** Apply both fractions once, show minutes per worker, round upward and do not call the result a service guarantee.

### Exercise 8 — Critique a constrained-demand model

**Time:** 20 minutes.  
**Type:** Critique.  
**Scenario:** An app closes a slot after 20 accepted orders. The model learns 20 as the usual peak and recommends staffing for 20.  
**Deliverable:** Explain the feedback loop and propose data and an evaluation approach.  
**Acceptance criteria:** Identify censoring, request exposure and cap logs, distinguish accepted from latent demand and avoid asserting an unobserved count as fact.

### Exercise 9 — Design the decision handoff

**Time:** 20 minutes.  
**Type:** Governance and design.  
**Scenario:** The forecast rises after an event announcement, but no additional staff are available.  
**Deliverable:** Draft a decision record with at least three named roles and two feasible responses.  
**Acceptance criteria:** Separate forecast, roster and customer promise; record model version, origin, uncertainty, constraints, owner and override rationale; include worker safety.

## 13. Self-check answers

### Answer 1

A suitable contract is: at 14:00 local time, forecast accepted orders assigned to Store A in each of `[17:00,17:15)`, `[17:15,17:30)`, `[17:30,17:45)` and `[17:45,18:00)` local time.
Count each deduplicated order at acceptance, retaining orders later canceled because the initial acceptance generated work.
Record the timezone, offset and late-arrival policy.
The resulting counts and quantiles are forecasts; the manager's staffing choice is a separate roster decision.
Other count definitions can be valid if they align with the work and are explicit.

### Answer 2

Use issue A if it covers the target location and time.
It was both published and ingested before 16:00.
Do not use issue B because it was not ingested until 16:05.
Do not use the cancellation because it was not published until 16:10.
If neither issue covers the interval, use the documented missing-weather fallback and flag it; do not use the eventual observation.

### Answer 3

Promotion adjustment is `18.0 × 0.10 = 1.8` orders.
The forecast is `18.0 + 1.8 + 1.5 − 0.5 = 20.8` orders per slot.
The event component could be negative if comparable evidence suggests displacement, but its sign alone does not prove causation.
The decimal is a mean, not a fractional realized order.

### Answer 4

Errors are `10 − 12 = −2`, `15 − 14 = +1` and `20 − 23 = −3` orders.
Mean error is `(−2 + 1 − 3) / 3 = −1.33` orders per slot, approximately.
MAE is `(2 + 1 + 3) / 3 = 2.00` orders per slot.
Negative mean error means net underforecasting under the stated convention; the MAE does not indicate direction.

### Answer 5 — Rubric

A strong timeline uses historical origins in chronological order and only mature labels available before each origin.
It reconstructs separate previous-day and 60-minute information sets rather than giving both the later features.
It reserves a final period from model selection and compares both models with a same-slot seasonal benchmark.
It retains archived weather and promotion versions and reports performance by horizon and peak segment.
A weak design randomly shuffles observations, uses actual weather or tunes repeatedly on the final period.

### Answer 6

For `q = 0.9`, `y = 24` and `f = 20`, pinball loss is `max(0.9 × 4, −0.1 × 4) = 3.6` orders.
A 90th percentile is a threshold intended to have approximately 90% of comparable outcomes at or below it.
One exceedance does not establish miscalibration; use many appropriate out-of-sample observations.
With discrete integer demand, the achieved coverage of a valid quantile can exceed 90% because probabilities jump at integer counts.

### Answer 7

One worker supplies `15 × 0.80 × 0.75 = 9.0` planning minutes per slot.
Demand generates `24 × 3.0 = 72.0` productive minutes.
The requirement is `ceil(72.0 / 9.0) = 8` pickers.
This assumes an empty starting queue and constant service time; it does not guarantee any particular service level.

### Answer 8 — Rubric

A strong response explains that the cap prevents observation of accepted orders above 20 and can suppress future model predictions.
It asks for slot availability, eligible sessions or checkout attempts, promised times, store outages and cap timing.
It scores accepted-order forecasts under the actual policy separately from estimates of unconstrained interest.
It proposes controlled variation or a defensible counterfactual design where feasible, acknowledging that exposure alone may not identify all lost demand.
A weak answer assumes the missing number is exactly equal to the number of customers who viewed the app.

### Answer 9 — Rubric

A strong record names the forecaster as prediction owner, the shift manager as labor decision owner and marketplace control as promise or slot-policy owner.
It may also include the courier lead and commercial lead.
It retains origin, model version, selected distribution, event evidence, available staffing, queue state and alternative actions.
Possible responses include a compliant promise adjustment or slot cap, or rescheduling a promotion with commercial approval.
The record states customer impact, worker safety, authorization, review time and whether an override changed the forecast or only the operational action.

## 14. Interview questions

1. **How would you define a 15-minute demand target for picking?**
   A strong answer specifies store assignment, queue event, slot boundaries, cancellation treatment and availability.
   A weak answer says “sales” without identifying an event or time.
2. **Why is the latest weather table unsafe for a historical backtest?**
   A strong answer distinguishes forecast issue, valid and ingestion times and reconstructs the as-of vintage.
   A weak answer assumes the current weather value was always available.
3. **What is the difference between observed orders and unconstrained demand?**
   A strong answer discusses caps, closures, exposure and the inability to observe every lost customer.
   A weak answer treats all zeroes as absence of interest.
4. **How do you forecast the hour from four slots?**
   A strong answer sums consistent means, but derives an hourly quantile from a joint distribution.
   A weak answer sums four upper quantiles without a dependence assumption.
5. **Why use rolling origins instead of random cross-validation?**
   A strong answer reproduces the decision horizon, available features and mature labels at each origin.
   A weak answer focuses only on increasing the number of training examples.
6. **How do you evaluate a 90th-percentile forecast?**
   A strong answer uses pinball loss, empirical coverage, width or sharpness and segment sample size.
   A weak answer claims one exceedance proves the model wrong.
7. **How would you turn a demand scenario into a picker recommendation?**
   A strong answer uses workload, availability, occupancy, rounding, roster constraints and backlog.
   A weak answer equates orders with workers or ignores productive minutes.
8. **Why might an accurate order forecast still lead to late deliveries?**
   A strong answer mentions basket mix, queueing, picker attendance, courier supply and travel.
   A weak answer assumes forecast accuracy controls the entire delivery process.
9. **How do you distinguish a promotion feature from a causal promotion effect?**
   A strong answer notes targeting, confounding and the need for experiments or an appropriate causal design.
   A weak answer interprets every observed sales increase as incremental.
10. **What should happen when a model recommends more workers than are available?**
    A strong answer exposes the gap, discusses feasible owner-approved actions and protects safety and labor constraints.
    A weak answer silently increases utilization to 100% or promises the original service anyway.
11. **How should overrides be governed?**
    A strong answer preserves the original prediction, author, reason, time and downstream action, then measures value added.
    A weak answer overwrites model history and evaluates only the final number.
12. **What is your fallback when the weather feed is late?**
    A strong answer uses an approved fallback with a freshness flag and later evaluates that segment.
    A weak answer inserts realized future weather or silently fills with zero.

## 15. Key terms

- [Forecast origin](#4-definitions-and-vocabulary) — issuance time of a prediction.
- [Target interval](#4-definitions-and-vocabulary) — time window whose demand is predicted.
- [Lead time](#4-definitions-and-vocabulary) — time from issuance to target start.
- [Bucket](#4-definitions-and-vocabulary) — aggregation period for a time series.
- [Accepted order](#4-definitions-and-vocabulary) — order accepted under the stated event definition.
- [Censored demand](#4-definitions-and-vocabulary) — demand hidden by an operating constraint.
- [Exposure](#4-definitions-and-vocabulary) — opportunity to see or use an offer.
- [Forecast vintage](#4-definitions-and-vocabulary) — prediction as issued at one origin.
- [Weather vintage](#4-definitions-and-vocabulary) — weather forecast available at one origin.
- [Feature](#4-definitions-and-vocabulary) — time-eligible predictor input.
- [Target leakage](#4-definitions-and-vocabulary) — use of unavailable prediction-time information.
- [Label maturity](#4-definitions-and-vocabulary) — point at which an outcome is stable enough to use.
- [Seasonal naive](#4-definitions-and-vocabulary) — comparable past-slot baseline.
- [Hierarchy](#4-definitions-and-vocabulary) — related store, geographic or temporal aggregations.
- [Reconciliation](#4-definitions-and-vocabulary) — alignment of related forecast totals.
- [Mean](#4-definitions-and-vocabulary) — expected value of a predictive distribution.
- [Quantile](#4-definitions-and-vocabulary) — probability threshold of a predictive distribution.
- [Prediction interval](#4-definitions-and-vocabulary) — range with stated intended coverage.
- [Bias](#4-definitions-and-vocabulary) — average signed error under a defined convention.
- [Mean absolute error](#4-definitions-and-vocabulary) — average magnitude of errors.
- [Pinball loss](#4-definitions-and-vocabulary) — asymmetric score for a quantile.
- [Calibration](#4-definitions-and-vocabulary) — agreement of predicted probabilities and frequencies.
- [Utilization](#4-definitions-and-vocabulary) — workload relative to available capacity.
- [Occupancy](#4-definitions-and-vocabulary) — share of time consumed by defined work.
- [Shrinkage](#4-definitions-and-vocabulary) — scheduled time unavailable for productive work.
- [Override](#4-definitions-and-vocabulary) — documented human change to a prediction or action.
- [Forecast value added](#4-definitions-and-vocabulary) — measured effect of a forecasting process step.
- [Drift](#4-definitions-and-vocabulary) — change in data or model performance over time.
- [Stockout](#4-definitions-and-vocabulary) — unavailability of a required item.
- [Service level](#4-definitions-and-vocabulary) — explicitly defined operating performance measure.
- [Decision loss](#4-definitions-and-vocabulary) — cost or harm associated with an action and outcome.

## 16. Sources

1. Hyndman, Rob J., and George Athanasopoulos. *Forecasting: Principles and Practice*, 3rd edition.
   Chapters on forecast accuracy, time-series cross-validation and hierarchical forecasting.
   [Online textbook](https://otexts.com/fpp3/).
   General forecasting methodology; not specific evidence for a particular delivery operator.
2. Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos.
   “The M4 Competition: 100,000 time series and 61 forecasting methods.”
   *International Journal of Forecasting* 36, no. 1 (2020): 54–74.
   [DOI: 10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014).
   Evidence about large-scale forecast comparison, not a quick-commerce deployment.
3. Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos.
   “The M5 accuracy competition: Results, findings, and conclusions.”
   *International Journal of Forecasting* 38, no. 4 (2022): 1346–1364.
   [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
   Retail forecasting evidence with different timing and operational constraints from the present case.
4. Gneiting, Tilmann, and Adrian E. Raftery.
   “Strictly Proper Scoring Rules, Prediction, and Estimation.”
   *Journal of the American Statistical Association* 102, no. 477 (2007): 359–378.
   [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
   Background on probabilistic forecast scoring.
5. U.S. National Weather Service.
   [API documentation](https://www.weather.gov/documentation/services-web-api).
   Official documentation for weather products and access; consult current data and licensing terms for a deployment.
6. European Union.
   [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Official legal text; applicability and interpretation depend on the processing context.
7. GS1.
   [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin).
   Official description of trade-item identification; not a demand or staffing standard.
8. ISO.
   [ISO 8601 date and time format](https://www.iso.org/iso-8601-date-and-time-format.html).
   Official overview of date and time representation.
9. Instacart.
   [The Instacart Online Grocery Shopping Dataset 2017](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data).
   Public competition dataset; examine its documentation and terms before use.

## 17. Links

- [Module 22 index](_index.md).
- [Topic 01: Promise tiers and dark-store design](01_promise-tiers-dark-stores.md).
- Topic 03, assortment for dark stores, is listed in the module index for the assortment and workload connection.
- Topic 05, courier supply and demand, is listed in the module index for the delivery-capacity connection.
- [Module 22 survey](../../../course/22_quick_commerce_on_demand_delivery.md).
- [Forecastability and benchmarks](../../02_methods/07_time_series_forecasting/01_forecastability-benchmarks.md).
- [Rolling-origin backtesting](../../02_methods/07_time_series_forecasting/08_rolling-origin-backtesting.md).
- [Simulation and optimization labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
- [Public Instacart dataset](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data) for a separate, limited-scope basket and timing exercise.
