# ETA Prediction

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 09 of 10
> Estimated study time: 3 hours
> Prerequisites: topics 01–03
> Version 2026.5

## 1. Why this topic exists

An estimated time of arrival (ETA) influences customer communications, warehouse labor, dock allocation and exception escalation. A single optimistic prediction can cause a missed promise; a deliberately conservative prediction can create unnecessary waiting and capacity reservations. The useful answer is therefore not merely a point estimate but a distribution, its uncertainty, and an action rule matched to the cost of early and late outcomes.

Predictions are not promises. A promise is a commercial commitment that can remain unchanged while new evidence moves the ETA. Forecast updates should inform whether to intervene or communicate, not quietly rewrite the original commitment and erase evidence of a breach.

This topic treats all numerical examples as fictional 2026 teaching assumptions. They are not measured carrier performance or a service guarantee.

### The business decisions behind the clock

A distribution center receives a mixture of appointment deliveries and unscheduled carrier arrivals.
The receiving manager must decide when to call labor, which door to reserve, and whether a vehicle
should wait in the yard. A forecast made at dispatch can support a broad staffing plan; a forecast
made 30 minutes before arrival can support a door assignment. Neither decision is improved merely
because an algorithm produces more timestamp updates.

The transportation control tower has a different job. It must identify shipments likely to miss a
customer commitment while there is still time to act. An exception queue containing every shipment
with any risk will bury the few cases that can be rescued. A queue containing only very high-risk
shipments may arrive too late for a useful intervention.

The customer service team needs a communication that can be understood and honored. A customer may
prefer a reliable two-hour range to a precise timestamp that moves every few minutes. Some contracts
require notification before a deadline; a technically accurate warning after the deadline has little
commercial value. The team also needs to know whether the customer is being told a forecast or offered
a new promise.

These decisions can conflict. The staffing manager may prefer a later planning quantile to reduce
vehicle waiting, while the customer may need an earlier warning of potential lateness. A single
"best ETA" cannot encode both cost structures. The underlying forecast should be shared, but the
decision rule, lead time, and accountable owner belong to each business process.

### Where the uncertainty comes from

Travel time is only one part of an arrival. A vehicle may not have left the origin, may have remaining
stops, may face an uncertain loading delay, or may arrive at the destination entrance and then wait for
access. A road-routing service that estimates travel between two coordinates does not by itself know
which of those activities remains.

Uncertainty changes as events occur. A new departure scan removes some loading uncertainty, but a
stale GPS position or an unrecorded stop can introduce a different uncertainty. Even a perfect
record of the past does not reveal a future incident. A model that continually narrows its interval
solely because the clock advances can therefore be confidently wrong.

An enterprise may buy an external ETA, develop an internal model, or combine several signals. The
commercial decision depends on event coverage, data rights, integration latency, auditability, and
measured performance for the intended decision. Vendor accuracy claims based on a different target
or only the easiest tracked shipments cannot be substituted for a local comparison.

The lesson follows a fictitious receiving and customer-service workflow. Its examples use USD and
minutes, and assume 2026 conditions solely for teaching. They illustrate calculation, not empirical
performance, contractual terms, or legal advice.

## 2. Learning outcomes

You can:

1. Define the predicted event, prediction time and available information.
2. Build a leakage-safe training snapshot from event streams.
3. Distinguish right censoring, competing outcomes and missing labels.
4. Calculate point error, interval coverage and quantile loss.
5. Select a decision quantile using asymmetric early and late costs.
6. Diagnose conditional calibration and subgroup bias without assuming causality.
7. Design time-based backtests, monitoring and a human escalation process.

Each outcome has an exercise and answer below.

## 3. The core mental model

```text
Known at prediction time          Unknown future
shipment + route + events  ----> arrival-time distribution
                                 |          |
                          prediction     promise deadline
                                 \          /
                              risk and action
                         notify / replan / wait
                                 |
                      actual event and feedback
```

An ETA is conditional on the information available at a specific instant. At 08:00 the model might forecast arrival at 11:30; after a traffic incident at 09:00 its forecast may move to 12:10. The customer promise of 12:00 is a distinct record. The relevant intervention may be a warning, a dock reassignment or a call, not changing the promise in the historical record.

A distribution answers more than a point estimate: the probability of arriving after noon and the range containing most likely outcomes are often more actionable than a mean. Forecast uncertainty must include operational variability, not merely a model's internal parameter uncertainty. A calibrated 90% interval contains the realized event about 90% of the time for an appropriate evaluation population; that statement does not guarantee 90% coverage for every individual shipment or group.

### Four records, four questions

| Record | Question | Typical owner | Can it change? |
|---|---|---|---|
| Promise | What was agreed with the customer? | Commercial operations | Only through a recorded amendment |
| Plan | What sequence and timing does operations intend? | Dispatcher | Yes, as a versioned decision |
| Forecast | What outcomes are plausible given current information? | ETA service | Yes, when information or time changes |
| Actual | What event was observed and when? | Event system and operations | A correction may be appended with provenance |

The separation matters in an investigation. Suppose the original promise was noon, the dispatcher
planned for 11:45, and the forecast at 09:00 put the median at 11:30. After an incident, the 10:30
forecast moves to 12:20; actual gate-in is 12:15. The latest forecast is only five minutes late as a
point estimate, but the delivery still missed the noon commitment by 15 minutes. Both statements
are true and describe different performance.

The event stream is not the same thing as the physical journey. A driver may cross a geofence at
11:53, a gate sensor may record entry at 11:58, and an operator may enter dock arrival at 12:11.
The first number is not necessarily a verified gate-in. The target must name both the operational
event and the rule for selecting its evidence.

### Reading the distribution

Let `T` be the eventual arrival timestamp and `I_s` the information available at snapshot time `s`.
The forecasting object is the conditional distribution `P(T <= t | I_s)` for each relevant time `t`.
The median is a summary of this distribution. The probability of missing a noon deadline is
`P(T > noon | I_s)` if arrival exactly at noon is on time.

These statements only apply to shipments for which the target event is defined. If cancellation can
prevent gate-in, distinguish the probability of gate-in by noon from the probability of cancellation.
A conditional ETA among eventual arrivals answers a useful question, but it does not by itself
assign a probability to every shipment outcome.

For a customer-facing interval, define both the lower and upper bounds. A nominal central 90%
interval can use the fifth and 95th percentiles; a 90th percentile alone is a one-sided upper bound.
The interval is a forecast, not a contractual window. A wide interval can be honest and still
insufficient for scheduling, so coverage must be interpreted together with its width.

### Information flows forward, not backward

At 09:00 the prediction service receives the latest available route version and telematics update.
At 09:20 the truck passes a checkpoint, but that scan arrives at 09:45. An as-of 09:30 replay cannot
include the checkpoint, even though its event timestamp is earlier than 09:30. At 09:45 the
prediction may legitimately update. A backtest that ignores this delay silently gives the historical
model knowledge that the production service did not possess.

The operational loop then adds another distinction. If dispatch changes the route after an alert,
the observed arrival belongs to the alerted-and-intervened process. It is not the counterfactual
arrival that would have occurred without action. Forecast scoring and the value of an alert policy
are related, but they are not the same measurement problem.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| ETA | Forecast of a defined arrival event, timestamp | A contractual promise |
| Prediction time | Instant at which features are frozen | Event time of a later update |
| Target event | Precisely defined arrival such as gate-in | Treating gate-in and dock-in as identical |
| Horizon | Time from prediction to event, minutes | Comparing horizons without stratification |
| Residual | Actual minus predicted time, minutes | Switching sign convention silently |
| Bias | Mean signed residual under a stated convention | Treating bias as proof of discrimination |
| MAE | Mean absolute error, minutes | Assuming it captures tail risk |
| Quantile | Value at or below which a stated fraction falls | A guarantee for an individual |
| Calibration | Agreement of predicted probabilities with observed frequency | Accuracy of individual predictions |
| Coverage | Fraction of outcomes inside an interval | Average interval width |
| Right censoring | Event time exceeds observation cutoff but remains unknown | Treating cutoff as arrival |
| Competing event | An event preventing the target, such as cancellation | Treating cancellation as an ordinary late arrival |
| Leakage | Information unavailable at prediction time entering features | A harmless historical variable |
| Promise | Committed service interval or deadline | Current predicted timestamp |
| Pinball loss | Asymmetric quantile-forecast loss | Absolute error for every quantile |

### Additional operational vocabulary

| Term | Definition and unit | Why the distinction matters |
|---|---|---|
| As-of join | Select records available no later than a snapshot | Prevents a later correction from leaking backward |
| Snapshot grain | The unit of one prediction, such as shipment × target × time | Determines denominators and repeated-observation dependence |
| Ingestion latency | Availability time minus event time, minutes | An old event can still be new information |
| Freshness age | Snapshot time minus latest eligible observation time, minutes | A last known position need not be current |
| Lead time | Time between decision or warning and required action, minutes | A correct alert can still be too late |
| Prediction interval | Lower and upper arrival bounds for a stated nominal coverage | Not automatically a commercial service window |
| Sharpness | Concentration of predictive distributions, often interval width in minutes | Useful only with adequate calibration |
| Pinball loss | Asymmetric loss for a chosen forecast quantile, in minutes | Rewards the requested percentile, not necessarily the mean |
| Brier score | Mean squared error of probability forecasts for a binary event | Scores both probability and outcome, not a single classification |
| Maturity | Whether sufficient follow-up exists to evaluate an outcome | Recent cohorts can look artificially fast |
| Competing risk | An event that precludes the original event | Cannot be represented as an arbitrarily late arrival |
| Intervention | Operational action taken after a forecast | Can change the observed outcome being scored |
| Policy | Rule connecting information to an action | Needs business evaluation in addition to model evaluation |
| Override | Human decision differing from a recommendation | Needs a reason and timestamp, not automatic treatment as error |
| Drift | Change in inputs, outcomes, or relationships over time | Input drift alone does not establish forecast deterioration |
| Fallback | Defined behavior when a forecast or required feed is unavailable | Should not masquerade as a fresh model prediction |
| Prediction version | Immutable identifier for one issued forecast | Connects a later decision to what was actually known |
| Label correction | Later revision of the recorded outcome | Must retain both the initial and corrected record |

**Residual convention.** This lesson uses `actual − forecast` throughout. A positive residual means
the arrival occurred later than the point prediction. Another organization may use the opposite
sign, so reports must show the formula rather than rely on the word "bias".

**Units and clocks.** Duration errors and interval widths are expressed in minutes. Timestamps are
stored as instants with a time-zone conversion for display. When a commitment is stated in local
civil time, the zone and any daylight-saving ambiguity are part of the commitment record.

## 5. The mechanics, step by step

### 5.1 Choose the event and decision

Define a target such as first verified gate-in at the destination, not the ambiguous phrase "arrival". State whether the prediction is generated at dispatch, at regular intervals or in response to a new event. Specify the decision maker, forecast horizon, required lead time and intervention costs. A warehouse labor decision and a customer notification may require different quantiles from the same predictive distribution.

Define the evaluation population and terminal outcomes. A cancelled shipment has no ordinary arrival time for that delivery. A shipment diverted to a different facility may have a different event definition, and a missed scan may hide a true arrival. These cases need separately specified handling rather than arbitrary imputation.

### 5.2 Construct as-of snapshots

For every historical prediction instant, retain only information available then: order attributes, scheduled route, known stops, location observations, traffic data actually available at that instant and operational events received by the system. Event time alone is insufficient: a late-arriving scan may describe an earlier event but was not available to the historical prediction.

Record both event timestamp and ingestion timestamp. Join the most recent eligible feature record whose availability time does not exceed the prediction time. Version route plans and promises; a final route or corrected actual destination can leak future decisions. Preserve immutable prediction snapshots and model version for later evaluation.

Repeated snapshots from one shipment are correlated. Splitting them randomly between training and test leaks the shipment trajectory and exaggerates generalization. Split by time and keep related records together as appropriate to the deployment decision.

### 5.3 Reconcile outcomes and censoring

For an event that has not yet occurred by the data cutoff, the observation says only that its event time exceeds the cutoff if the shipment was still at risk and tracking was reliable. This is right censoring, not a recorded late arrival at the cutoff. A survival model can use censored observations under its stated censoring assumptions. Simply dropping recent in-transit shipments preferentially retains fast arrivals and biases the apparent distribution toward early arrival.

Cancellation, return or rerouting may prevent the original event entirely. Model the probability of such a competing outcome separately or use a competing-risk framework appropriate to the question. A technical missing scan is neither confirmed censoring nor proof of cancellation. Reconcile source records and report the unresolved fraction.

### 5.4 Build baselines and forecasts

Start with a schedule-based or historical conditional-median baseline available at prediction time. Add information only when it improves out-of-time performance and has an operationally reliable source. Compare point forecasts with MAE and signed error, and compare tail forecasts with pinball loss and coverage. The median minimizes expected absolute error under standard conditions; the mean minimizes expected squared error.

A point prediction cannot be turned into a credible probability of lateness without a predictive error distribution. Estimate conditional quantiles or distributions and check whether the implied quantiles are ordered. If separate quantile models cross, fix the method and re-evaluate calibration rather than treating the inverted interval as valid.

### 5.5 Evaluate calibration, sharpness and bias

For a predicted 90th percentile, approximately 90% of realized arrival times should be at or before that percentile in the evaluated population. Check empirical coverage at multiple quantiles and within horizon, lane, carrier, facility and relevant service groups. A pooled 90% can hide severe overcoverage for one group and undercoverage for another. Small groups need uncertainty intervals and suppression where identification is possible.

Calibration is not enough: an interval spanning the entire day can cover nearly every arrival and still be useless. Report interval width or quantile loss along with coverage. Define residual as actual minus predicted; a positive mean residual means actual arrival was later than the point forecast on average. Report missingness and outcome maturity by group so apparent bias is not merely selective observation.

### 5.6 Convert the forecast into an action

If an action must be set at time a, and each minute of arrival after a costs c_late while each minute of arrival before a costs c_early, the cost-minimizing quantile is c_late / (c_late + c_early), under linear costs and a correct distribution. This is a newsvendor-style decision rule, not a universal service-level rule. Fixed penalties, capacity constraints and shared resources change the optimum.

Escalate when the estimated probability of missing the commitment exceeds a business threshold justified by intervention effectiveness and cost. Track notification timing, false alerts, missed alerts and customer response. A notification does not itself make the delivery arrive earlier.

### 5.7 Backtest and monitor

Hold out later time periods, include all eligible shipments with sufficiently mature outcomes and compare the proposed model against the baseline at identical prediction times. Use rolling-origin evaluation across seasons where data permits. Examine model performance after an intervention separately: dispatch action can alter the observed arrival and make a forecast appear inaccurate precisely because it caused a successful rescue.

Monitor input freshness, missingness, calibration, coverage, error, subgroup gaps and the downstream outcome. Establish ownership and fallback: if location or traffic data is stale, show a suitably qualified baseline or suppress automated action. Do not continue displaying an old prediction as fresh.

### 5.8 Build a point-in-time feature contract

For every input, document its owner, identifier, availability timestamp, expected refresh rate and fallback.
The contract should distinguish a missing observation from an observed value of zero.
For example, zero remaining stops is a meaningful route state; unknown remaining stops is not.
Store the difference as an explicit missingness indicator when the model can use it safely.

Location data needs a maximum age.
A position observed 45 minutes ago is not the vehicle's current position, even when it is the latest record.
Pass its age to the model and define when the prediction is too stale for automated use.
Do not infer precise movement from a single old coordinate.

Route plans require versioning because a dispatcher may reorder stops after the prediction.
At each historical prediction instant, use the version received by that instant.
The final route can be useful for operational analysis but is not an admissible historical feature.
If version history is unavailable, explicitly limit the evaluation rather than reconstructing an unrealistically informed snapshot.

Traffic forecasts also need an availability timestamp.
The observed traffic at 11:00 cannot be used to simulate a prediction made at 09:00.
If historical forecasts were not retained, substitute an available historical baseline and label the limitation.
Changing the input availability rule between training and deployment invalidates the backtest.

### 5.9 Separate data quality from predictive quality

Build a reconciliation table for each prediction cohort.
Count predictions, uniquely linked shipments, observed target events, cancellations, unresolved records and records still in transit.
The categories must be mutually exclusive at the reporting cutoff and sum to the eligible cohort.
Publish unresolved and censored counts alongside accuracy metrics.

Check event provenance before treating a timestamp as ground truth.
A mobile scan, a gate sensor and a manually corrected record can disagree.
Define a source hierarchy with operations, retain corrections and sample disagreements.
Do not silently overwrite the original value when a later correction arrives.

Compare timestamp quality by carrier, facility, shift and forecast horizon.
An apparent improvement may occur if difficult shipments lose their event feed and disappear from scoring.
When missingness is substantial, present sensitivity bounds or targeted audits rather than claiming unbiased accuracy.
The cause of missingness must be investigated; it cannot be inferred from the missing value alone.

### 5.10 Select an alert rule

Define the action available after an alert and the minimum useful lead time.
A warehouse that needs 45 minutes to reassign labor cannot benefit from an alert 5 minutes before arrival.
Likewise, notifying a customer does not necessarily change the vehicle's progress.
Measure the operational response separately from forecast accuracy.

Set a candidate risk threshold using the cost of alerts, the cost of missed events and the expected effectiveness of the intervention.
If intervention capacity is limited, rank cases by expected incremental benefit rather than risk alone.
A highly late shipment that cannot be rescued may be less actionable than a moderately late shipment that can.
Document how capacity and costs influence the ranking.

Report true alerts, false alerts, missed events and alerts that arrived too late to act.
Evaluate notification volume per shift against the team's actual capacity.
Repeated alerts for the same shipment should be grouped for workload measurement.
Maintain a visible owner and an escalation path for alerts that remain unacknowledged.

### 5.11 Manage feedback and model change

Once dispatchers act on forecasts, observed arrival times reflect both the shipment and the intervention.
Comparing alerted shipments with unalerted shipments alone confounds risk with action.
Log the forecast, recommended action, actual action, action time and operational reason for overrides.
Where safe and appropriate, randomized rollout can estimate the effect of a policy rather than the accuracy of a model alone.

Use a shadow period to compare a candidate model and the existing forecast on the same eligible predictions.
Shadow performance does not establish the benefit of a live intervention.
Before release, specify rollback triggers for stale data, calibration deterioration and operational overload.
Keep the previous model and its feature contract available for a controlled rollback.

Monitor separately after major route redesigns, carrier changes and seasonal peaks.
A shift in the distribution of inputs is a warning, not proof that outcomes have degraded.
When labels mature, examine conditional error and calibration before deciding whether to retrain.
Retain the model version and decision record for each forecast.

### 5.12 Specify the operational service level

Begin with a decision inventory rather than a list of available model features.
For each decision, record the decision owner, the population, the latest useful decision time,
the actions available and the consequence of a wrong choice.
A forecast that is optimal for staffing may not be the right notification trigger.
Document the distinction before choosing a model metric.

For a receiving center, the unit might be one destination stop, predicted every 30 minutes from
dispatch until gate-in. The operational action may be to reserve a door for a 45-minute interval.
For customer service, the unit might be one customer delivery, updated after material route events.
The action may be to contact a customer only when a potential miss can still be mitigated.

Specify the denominator as carefully as the event. Are cancelled orders included in a service-risk
report? Are transfers between a carrier's terminals included? Do multiple packages on one vehicle
produce one forecast or one per delivery stop? Answers change both workload and the interpretation
of model accuracy.

**Output of this step:** an agreed target-event definition, eligibility rule, forecast schedule,
consumer, and action deadline. A service owner signs off on the ambiguity, not merely the model team.

### 5.13 Model the journey at the right level

An intuitive decomposition of destination gate-in is:

```text
gate-in = snapshot time
        + remaining origin delay
        + remaining drive time
        + remaining stop and break time
        + destination approach or queue time
```

Only include a component when it is genuinely still outstanding at the snapshot. An already completed
stop should not be predicted again. The terms are not generally independent: congestion can delay a
previous stop and the subsequent drive, and a late departure can cause arrival in a busier queue.
Adding component means is valid when expectations exist, but adding independently estimated
percentiles does not generally produce the percentile of their sum.

For an early-horizon forecast, a lane-level historical distribution may be more dependable than a
precise live route with no confirmed departure. At a short horizon, current vehicle position and
destination access conditions may dominate. Compare candidate representations within horizon bands
instead of assuming that one feature set is best for the entire journey.

When a stop sequence changes, use the new plan only after it is available to the ETA service.
If the destination changes, determine whether the original forecast is closed as diverted and
a new target created. Silently relabeling the destination makes historical forecast errors
uninterpretable.

### 5.14 Construct an auditable snapshot

Use a stable key such as `(shipment_id, destination_stop_id, prediction_time, forecast_version)`.
Keep the target definition and the promise version with the prediction. Store the features used,
their source availability timestamps, the generated quantiles, and the model and calibration versions.
The immutable record should permit an investigator to reconstruct what a dispatcher saw.

Suppose three records concern the same trip:

| Record | Event time | Available time | Usable at 09:30? |
|---|---|---|---|
| Departure | 08:45 | 08:47 | Yes |
| Checkpoint scan | 09:15 | 09:42 | No |
| Route revision | 09:25 | 09:29 | Yes, if the service could access it |

The route revision is admissible because its availability precedes the prediction. The checkpoint
scan is not admissible merely because the physical event preceded the prediction. If the route
revision was written into the warehouse at 09:29 but delivered to the online service at 09:37,
the relevant availability is 09:37 for replaying that service.

Set a deterministic rule for ties, duplicates and corrections. For instance, use the latest record
available by the snapshot, ordered by availability time and then by source sequence number.
If a correction later changes an event, retain the earlier value that was visible at prediction time
and separately retain the corrected label used for retrospective scoring.

**Branch:** if historical availability times are absent, a full point-in-time replay is not
established. Quantify affected records, use a defensible lagged baseline where possible, and label
the result as limited. Do not replace missing availability with event time and call it leakage-safe.

### 5.15 Define outcome states and the reporting cutoff

At a fixed cutoff, classify each eligible target once: verified arrival, cancellation before target,
diversion that ends the original target, still at risk and observed through cutoff, or unresolved.
The final two categories differ. A tracked in-transit truck can support a lower bound on arrival;
an absent feed may provide no trustworthy statement that the truck is still in transit.

Example cohort of 1,000 shipment targets:

| Status at cutoff | Count | Interpretation |
|---|---:|---|
| Verified gate-in | 870 | Exact arrival label, subject to correction |
| Cancelled | 30 | Original arrival no longer expected |
| Diverted | 20 | Original target ended under the chosen definition |
| Tracked and still at risk | 50 | Right-censored at cutoff |
| Unresolved | 30 | Insufficient evidence for the other states |
| Total | 1,000 | All eligible targets accounted for |

A score of 870 observed arrivals describes those arrivals, not automatically all 1,000 targets.
If the 30 unresolved shipments disproportionately come from a carrier with unreliable scans, that
selection can change the reported metric. Publish the accounting table next to the score.

A survival approach can incorporate the 50 censored observations when its assumptions and
observation process are appropriate. It does not magically repair missing scans or informative
tracking dropout. A maturity lag can reduce censoring, but do not declare a cohort complete merely
because a fixed number of days has passed; check remaining unresolved and in-transit cases.

### 5.16 Compare baseline and candidate fairly

The baseline is the forecast that would actually have been available to the decision maker,
not a final schedule reconstructed after the event. Possibilities include a scheduled arrival, the
carrier's current ETA, or a conditional historical median based on lane and horizon. Record its
coverage and availability as well as its accuracy.

Compare forecasts at the same snapshot and for the same eligible target. A model issued only for
well-tracked vehicles may look better than a baseline issued for every vehicle. Report both the
matched-population comparison and the proportion of the full eligible population each method serves.
Then evaluate the fallback policy on all eligible cases rather than omitting the difficult cases.

For example, a candidate has 24 minutes of MAE on 700 matched cases, while the baseline has
30 minutes on those same cases. The six-minute improvement is meaningful for that matched set.
It says nothing by itself about the remaining 300 cases, which may require a different forecast.
If the fallback has 45 minutes of MAE on those 300 cases, the combined policy MAE is
`(700 × 24 + 300 × 45) / 1,000 = 30.3 minutes`, assuming every case has an observed label.
Compare this 30.3-minute policy score to the baseline's full-population score, not its matched score.

### 5.17 Use out-of-time evaluation without false independence

Choose a training cutoff, fit the model using only eligible historical snapshots, and score a later
period. Repeat at several rolling origins if operational history permits. Ensure training labels
were available by each training cutoff, not merely that their events happened by then.
Include peaks and disruptions as separate slices rather than blending every period into one average.

Keep all snapshots of a shipment on the appropriate side of a split. If the intended deployment
predicts newly dispatched shipments, a later snapshot of a training shipment should not appear as
an independent test example. If an operational question deliberately concerns ongoing shipments
that existed at the cutoff, state that design and prevent later outcome information from entering
training or features.

Repeated hourly snapshots create more rows for longer trips. Snapshot-weighted MAE answers how the
service performs on its issued predictions; shipment-weighted MAE answers how the average shipment
performs under a defined aggregation. Neither is universally correct. Report both or select the
one corresponding to the workload, and estimate uncertainty by resampling shipments or another
appropriate independent unit rather than treating every snapshot as independent.

### 5.18 Measure point and probability forecasts

For `n` observed arrivals, let residual `e_i = actual_i − forecast_i` in minutes.
Mean signed error is `sum(e_i) / n`, while MAE is `sum(abs(e_i)) / n`.
The first exposes average direction; the second gives the average magnitude.
Large positive and negative errors can cancel in signed error.

For quantile level `q`, pinball loss is `q × e` when `e >= 0` and `(q − 1) × e` when `e < 0`.
At `q = 0.5` it is half absolute error. At `q = 0.9`, underpredicting by 20 minutes costs
18 loss-minutes and overpredicting by 20 minutes costs two loss-minutes.
Average the loss over a defined population and compare models at the same quantile and snapshots.

For a binary late event, the Brier score is the mean of `(p_i − y_i)^2`, with `y_i = 1` for a
miss and zero otherwise. Lower is better on the same eligible cohort. A forecast of 0.20 followed
by a miss contributes 0.64; the same probability followed by an on-time arrival contributes 0.04.
Probability scoring is not a substitute for a reliability plot or a decision-value calculation.

### 5.19 Evaluate calibration and usefulness together

Group predictions into predeclared risk bands, then compare mean predicted probability with observed
late frequency. A band containing 200 shipments with mean predicted risk 20% and 42 observed misses
has observed frequency 21%. This is close in that sample, not proof that every shipment had a 20%
chance or that the model will remain calibrated next month.

For an interval, compute the observed fraction whose target timestamps fall between the stated
bounds, with an explicit inclusion rule for endpoints. Report median or mean interval width in
minutes and pinball losses for the endpoints. A central 90% interval that covers 91% of arrivals
but is 12 hours wide may be less operationally useful than a narrower, adequately calibrated
alternative. Do not optimize width alone.

Break down results by prediction horizon, lane type, carrier, facility and time period where counts
support it. Avoid causal claims from differences in descriptive scores: carrier mix, traffic,
data quality and customer requirements can all differ. Use uncertainty intervals appropriate to
the unit of dependence; shipment-level or route-level clustering may be needed.

### 5.20 Convert risk into an implementable queue

For each candidate alert, estimate the incremental benefit of the action, not just the probability
of lateness. The relevant calculation is expected loss without action minus expected loss with action
minus action cost. Only count effects that the intervention can plausibly change.

Suppose two shipments each cost USD 40 if late, and each intervention costs USD 5. Shipment A has
50% late risk, but the action prevents only 10% of its late loss; expected net benefit is
`0.50 × 40 × 0.10 − 5 = −USD 3`. Shipment B has 30% late risk and a 60% effective action;
expected net benefit is `0.30 × 40 × 0.60 − 5 = USD 2.20`. Under these teaching assumptions,
the lower-risk case is the better use of a scarce intervention slot.

Efficacy estimates must match the intervention and population. A customer notification may reduce
the cost of a miss without changing the probability of late arrival. A carrier escalation may change
the arrival time, but its effect is uncertain. Do not substitute the model's calibration statistic
for evidence that either action works.

Implement a queue with an owner, acknowledgement deadline, suppression of duplicate alerts and a
route for escalation. Acknowledge, act, override and close are distinct states. If a forecast
changes, preserve the original alert and record whether the new forecast changes its priority.
When capacity is exhausted, document what was not handled and why.

### 5.21 Release, fallback and retire

Before launch, agree on an owner for the forecast service, one for data feeds and one for each
downstream action. Define an operating-hours support path and a change-control process.
The release record should include target and cohort definitions, backtest period, baseline,
coverage, calibration, decision rule and known exclusions.

Run a shadow comparison against the incumbent before using the new forecast to drive actions.
Shadow mode establishes whether outputs can be produced and compared; it does not establish
intervention benefit. A limited live pilot can measure workflow and outcomes, ideally with a
defensible comparison design and safeguards against denying required service.

Set operational rollback triggers before release. Examples include a location-feed freshness
breach, a large rise in unmatched targets, invalid quantile order, or sustained undercoverage
after enough labels mature. Specify who can pause automation, which baseline is served, and how
customers and operators see the fallback's age and limitations.

On retirement, retain forecast and decision records according to the applicable retention policy.
Reconcile historical reports to the version that generated each decision rather than rewriting
history with the replacement model. Remove obsolete integrations only after their consumers
have moved to an agreed replacement.

## 6. Worked example

At 09:00, a fictional shipment has an unchanged customer promise of arrival by 12:00 local time. A model provides this conditional distribution:

| Arrival time | Cumulative probability |
|---|---:|
| 11:00 | 0.20 |
| 11:30 | 0.50 |
| 12:00 | 0.80 |
| 12:30 | 0.95 |
| 13:00 | 1.00 |

For this discrete teaching distribution, the probability of arrival after 12:00 is 1 − 0.80 = 0.20, or 20%. It does not mean this shipment will be 20% late. The median is 11:30, but a median forecast is not a 90% service commitment.

Suppose a dock staffing decision incurs USD 2 per minute if the shipment arrives after the chosen staffing time, and USD 1 per minute if it arrives before. Under the linear-cost assumptions, the optimal decision quantile is 2 / (2 + 1) = 0.667. The earliest listed time whose cumulative probability is at least 0.667 is 12:00. This staffing time is not a revised delivery promise.

Check the expected costs directly. At 11:30, the outcome probabilities are 0.20, 0.30, 0.30, 0.15 and 0.05. Expected cost is 0.20 × 30 × USD 1 + 0.30 × 30 × USD 2 + 0.15 × 60 × USD 2 + 0.05 × 90 × USD 2 = USD 51. At 12:00 it is 0.20 × 60 × USD 1 + 0.30 × 30 × USD 1 + 0.15 × 30 × USD 2 + 0.05 × 60 × USD 2 = USD 36. At 12:30 it is 0.20 × 90 × USD 1 + 0.30 × 60 × USD 1 + 0.30 × 30 × USD 1 + 0.05 × 30 × USD 2 = USD 48.

Thus 12:00 minimizes expected cost among these choices. Real dock staffing involves discrete shifts and other trucks, so an integrated schedule may differ. The distribution itself is assumed; no evidence is supplied that it is calibrated.

### Reconstruct the probability table

The table above is cumulative, so individual arrival probabilities are differences between successive rows.
The mass at 11:00 is 0.20.
The mass at 11:30 is 0.50 − 0.20 = 0.30.
The mass at 12:00 is 0.80 − 0.50 = 0.30.
The mass at 12:30 is 0.95 − 0.80 = 0.15.
The mass at 13:00 is 1.00 − 0.95 = 0.05.
The five probabilities sum to 1.00.

The probability of arriving exactly at 12:00 is 30% under this discrete assumption.
Consequently, the probability of arriving at or after 12:00 is 50%, while the probability of arriving strictly after 12:00 is 20%.
The distinction depends on the business definition of a missed deadline.
For a continuous forecast distribution, an exact timestamp usually has no point mass; operational rounding can nevertheless create ties in recorded data.

The 90th percentile is 12:30 because 12:00 covers only 80% and 12:30 covers 95%.
For this discrete distribution the realized coverage at that percentile can exceed 90%.
Do not interpret that extra 5 percentage points as proof of poor calibration without considering the distribution's discrete support.

### Calculate forecast error

Assume the recorded arrival is 12:30 and the point forecast is the median, 11:30.
With residual defined as actual minus predicted, the residual is positive 60 minutes.
The absolute error is 60 minutes.
This single example cannot establish average bias or model quality.

For the 90th-percentile prediction of 12:30, the residual is zero minutes.
The quantile loss is zero for this realized event.
Had the actual arrival been 13:00, the residual would be 30 minutes and pinball loss would be 0.90 × 30 = 27 minutes.
Had the actual arrival been 12:00, the residual would be −30 minutes and pinball loss would be 0.10 × 30 = 3 minutes.
These asymmetric losses discourage underpredicting the 90th percentile more heavily than overpredicting it.

### Examine an alert decision

Suppose a customer-contact action costs USD 4 per alert and prevents an expected USD 30 loss if a late arrival occurs.
Assume the action is fully effective when late, has no other effect, and can be completed in time.
At a 20% late probability, expected avoided loss is 0.20 × USD 30 = USD 6.
Expected net benefit is USD 6 − USD 4 = USD 2 per shipment.
Under these assumptions, the action is justified because the break-even probability is USD 4 / USD 30, or about 13.3%.

If the action prevents only half the late loss on average, expected avoided loss is USD 3.
The same alert now has expected net benefit of negative USD 1.
An accurate risk forecast alone cannot settle this decision; the intervention's effect and cost matter.
The assumed efficacy must be estimated with an appropriate evaluation rather than inferred from notification delivery.

### Extend the same case through a shift

The 09:00 forecast describes one target: first verified gate-in at the destination.
The contractual deadline is 12:00 local time, and arrival at exactly 12:00 is on time.
The staffing decision is made at 09:00; it does not alter the truck's physical arrival.
An alert decision is made separately and may change the cost of a late arrival.

The receiver must choose its staffing time from the five times in the original probability table.
The USD 1 per minute early cost represents staff waiting before the truck arrives.
The USD 2 per minute late cost represents the vehicle and downstream operation waiting after the
chosen staffing time. This simplified cost applies to one truck and a continuously adjustable start.
It does not include shift minimums, multiple doors, or customer penalties.

The full calculation for 12:00 makes the direction of each term explicit:

| Gate-in | Probability | Minutes before staffing | Minutes after staffing | Scenario cost | Weighted cost |
|---|---:|---:|---:|---:|---:|
| 11:00 | 0.20 | 60 | 0 | USD 60 | USD 12.00 |
| 11:30 | 0.30 | 30 | 0 | USD 30 | USD 9.00 |
| 12:00 | 0.30 | 0 | 0 | USD 0 | USD 0.00 |
| 12:30 | 0.15 | 0 | 30 | USD 60 | USD 9.00 |
| 13:00 | 0.05 | 0 | 60 | USD 120 | USD 6.00 |
| Total | 1.00 | — | — | — | USD 36.00 |

Check the extreme candidate times as well. At 11:00, expected late cost is
`0.30 × 30 × 2 + 0.30 × 60 × 2 + 0.15 × 90 × 2 + 0.05 × 120 × 2 = USD 93`.
At 13:00, expected early cost is
`0.20 × 120 + 0.30 × 90 + 0.30 × 60 + 0.15 × 30 = USD 73.50`.
Together with the USD 51, USD 36 and USD 48 calculations above, this confirms that 12:00 is the
lowest-cost choice among all five listed times.

The result is sensitive to economics. If late and early costs were each USD 1 per minute, the
decision quantile would be 0.50 and 11:30 would be a valid median choice. If early cost were zero,
with all other assumptions unchanged, staffing at the latest possible arrival avoids waiting for the
vehicle, though a real shift constraint or other trucks could reverse that operational choice.

### Add a live update without rewriting history

At 10:00, suppose a verified incident report reaches the ETA service.
The new model distribution assigns a 45% probability to gate-in after 12:00.
This is a new conditional forecast, not evidence that the 09:00 forecast of 20% was wrong.
Both forecasts must be scored over many comparable instances using the information available at
their respective times; one observed late arrival cannot establish which probability was calibrated.

Assume an available action reduces the loss conditional on lateness by 50%.
The loss is USD 30 if late and the action costs USD 4, as in the previous alert calculation.
At 10:00, expected avoided loss is `0.45 × 30 × 0.50 = USD 6.75`.
Expected net benefit is `USD 6.75 − USD 4 = USD 2.75`.
The break-even risk under these assumptions is `4 / (30 × 0.50) = 0.2667`, or about 26.7%.
The action is justified on expected value if it can still be completed and no better use of
constrained capacity exists.

At 09:00, the same half-effective action had a negative USD 1 expected net benefit.
Thus the recommended action can change as information changes even when the promise does not.
The record should show both risk estimates, each issuance time, the decision, the action time and
the assumptions used in the decision. A customer communication must say that noon remains the
commitment unless an authorized amendment has actually been agreed.

### Score a small later cohort

The following four fictional shipments are a separate evaluation cohort.
Each row uses the last eligible forecast at a common two-hour lead-time decision point.
The late event is gate-in strictly after the original noon promise.
All four outcomes are observed; this example ignores sampling uncertainty only to show arithmetic.

| Shipment | Median ETA | Actual gate-in | Residual, minutes | Late probability | Late outcome |
|---|---|---|---:|---:|---:|
| A | 11:30 | 11:40 | +10 | 0.20 | 0 |
| B | 11:45 | 12:15 | +30 | 0.40 | 1 |
| C | 12:10 | 11:50 | −20 | 0.70 | 0 |
| D | 12:00 | 12:30 | +30 | 0.60 | 1 |

Mean signed error is `(10 + 30 − 20 + 30) / 4 = +12.5 minutes`.
MAE is `(10 + 30 + 20 + 30) / 4 = 22.5 minutes`.
The positive signed error indicates that actuals were later than the medians on average in this
small sample, not that every prediction was optimistic.

The four Brier contributions are `0.20² = 0.04`, `(0.40 − 1)² = 0.36`,
`0.70² = 0.49`, and `(0.60 − 1)² = 0.16`.
Their mean is `(0.04 + 0.36 + 0.49 + 0.16) / 4 = 0.2625`.
This number has no useful standalone business interpretation without a comparator and a larger
representative sample. It nevertheless penalizes confidently wrong probabilities more than
less-confident ones.

If the alert threshold is 0.50, the rule alerts on C and D.
It catches D, misses B, and sends one false alert for C; A is a correct no-alert.
Alert precision is `1 / 2 = 50%`, and recall of late shipments is `1 / 2 = 50%`.
These are counts of shipment-level first alerts, not counts of every forecast refresh.
They do not establish the economic value of the action or whether an alert arrived early enough.

### Evaluate the operational response

Suppose the alert for D arrives 70 minutes before the deadline and is acknowledged after 10 minutes.
If the team requires 45 minutes after acknowledgement to arrange the action, 60 minutes remain and
the intervention is actionable. If the alert for C arrives only 20 minutes before the deadline, it
may still help with customer communication but is too late for a 45-minute carrier intervention.
The same alert can therefore be useful for one workflow and unusable for another.

If D is rescued and arrives at 11:55 after intervention, the late outcome changes from one to zero.
This observed result does not prove that the alert was unnecessary or that the original risk was
incorrect. The no-action arrival is not observed. Evaluate the alert policy against a suitable
comparison group, and avoid treating intervention-caused success as a model failure.

### Decision handoff

The receiving supervisor receives the 12:00 staffing recommendation with its cost assumptions.
The customer-service owner receives a separate alert only when its own benefit and lead-time rules
are met. The data owner reconciles actual gate-in against the gate system and retains any correction.
The forecasting owner monitors probability quality over matured cohorts.
The commercial owner retains the noon commitment and any authorized change.
The workflow is complete only when these records can be joined to the issued forecast.

## 7. Code, query, or calculation

The following Python 3 code evaluates the same discrete expected costs. Times are minutes after 11:00; all costs are fictional USD.

```python
arrivals = [(0, 0.20), (30, 0.30), (60, 0.30),
            (90, 0.15), (120, 0.05)]
early_cost = 1.0
late_cost = 2.0

def expected_cost(action):
    return sum(p * (early_cost * max(action - t, 0)
                    + late_cost * max(t - action, 0))
               for t, p in arrivals)

for action in (30, 60, 90):
    print(f"{action:3d} min: ${expected_cost(action):.2f}")
print(f"Decision quantile: {late_cost / (early_cost + late_cost):.3f}")
```

Expected output:

```text
 30 min: $51.00
 60 min: $36.00
 90 min: $48.00
Decision quantile: 0.667
```

This code is a decision calculation, not an ETA model. In production validate that probabilities are nonnegative and sum to one, costs represent the relevant decision and all timestamps share an unambiguous time basis.

### A reproducible probability and point-score calculation

The next standard-library Python example scores the four-shipment cohort in section 6.
The input probabilities are risk of arrival strictly after the promise.
Residuals are actual minus median forecast in minutes, already computed from timestamps.
Every shipment has one observed outcome and one forecast at the same decision point.

```python
rows = [
    {"id": "A", "error": 10, "risk": 0.20, "late": 0},
    {"id": "B", "error": 30, "risk": 0.40, "late": 1},
    {"id": "C", "error": -20, "risk": 0.70, "late": 0},
    {"id": "D", "error": 30, "risk": 0.60, "late": 1},
]

def pinball(error, quantile):
    if error >= 0:
        return quantile * error
    return (quantile - 1) * error

n = len(rows)
bias = sum(row["error"] for row in rows) / n
mae = sum(abs(row["error"]) for row in rows) / n
brier = sum((row["risk"] - row["late"]) ** 2 for row in rows) / n
median_loss = sum(pinball(row["error"], 0.5) for row in rows) / n
threshold = 0.50
tp = sum(row["risk"] >= threshold and row["late"] == 1 for row in rows)
fp = sum(row["risk"] >= threshold and row["late"] == 0 for row in rows)
fn = sum(row["risk"] < threshold and row["late"] == 1 for row in rows)

print(f"Mean signed error: {bias:.1f} min")
print(f"MAE: {mae:.1f} min")
print(f"Median pinball loss: {median_loss:.2f} min")
print(f"Brier score: {brier:.4f}")
print(f"Alerts: {tp + fp}; true positives: {tp}; missed late: {fn}")
```

Expected output:

```text
Mean signed error: 12.5 min
MAE: 22.5 min
Median pinball loss: 11.25 min
Brier score: 0.2625
Alerts: 2; true positives: 1; missed late: 1
```

Median pinball loss equals half the MAE here because all point forecasts are medians and use the
same observations. This code intentionally contains no timestamp parsing, censoring adjustment,
calibration model, confidence interval, or decision-effect estimate. Production scoring must
define these explicitly instead of assuming that four complete records resemble the full fleet.

### Point-in-time query pattern

The following illustrative PostgreSQL query retrieves the most recent eligible location for each
snapshot. It assumes each snapshot and location has a unique identifier, and the prediction service
could read the location at `available_at`. Both timestamps are `timestamptz`.

```sql
SELECT
    s.snapshot_id,
    s.shipment_id,
    s.prediction_time,
    p.event_time,
    p.available_at,
    p.latitude,
    p.longitude,
    EXTRACT(EPOCH FROM (s.prediction_time - p.event_time)) / 60.0
        AS location_age_minutes
FROM prediction_snapshots AS s
LEFT JOIN LATERAL (
    SELECT e.event_time, e.available_at, e.latitude, e.longitude
    FROM location_events AS e
    WHERE e.shipment_id = s.shipment_id
      AND e.available_at <= s.prediction_time
      AND e.event_time <= s.prediction_time
    ORDER BY e.event_time DESC, e.available_at DESC, e.event_id DESC
    LIMIT 1
) AS p ON TRUE;
```

The `LEFT JOIN` retains snapshots without an eligible position; it does not turn missing locations
into zero coordinates. Ordering by event time finds the most recent observed position among records
already available; the later fields resolve ties deterministically. A late correction to an older
position is eligible when received but does not automatically become the most recent physical
position. A production source with correction identifiers needs a source-specific version rule.

For a snapshot at 09:30, a position observed at 09:10 and available at 09:12 is eligible.
A position observed at 09:20 but available at 09:42 is not eligible.
The returned age is 20 minutes. If the operational freshness limit is 15 minutes, the downstream
policy must mark that location stale even though the SQL found an eligible record.
Indexing, volume, clock quality, and data access are implementation concerns outside this example.

### A one-sided calibration calculation

For each matured, observed target, record whether actual gate-in was at or before the forecast
90th percentile. Add the indicators and divide by the number of eligible targets.
If 174 of 200 arrivals meet the bound, empirical coverage is 87%.
The observed difference from nominal coverage is minus three percentage points.
With only 200 outcomes and possible correlation, this result is a diagnostic, not by itself
proof of persistent undercoverage. Examine the uncertainty, cohorts, missingness, and width.

## 8. Data, standards, and regulatory context

At minimum retain shipment identifier, target facility, event definition, event time, time zone, source, ingestion time, snapshot time, route-plan version, forecast distribution, model version, commitment and final outcome. Preserve updates rather than overwriting them. Track which operational decisions were made after a prediction so the feedback loop can be studied.

GS1 [EPCIS](https://www.gs1.org/standards/epcis) defines a way to exchange supply-chain event information; it does not prescribe a forecasting method or establish scan truth. Carrier telematics and third-party traffic data require contractual permission, appropriate security and documented latency. Record whether historical traffic was actually available at the simulated prediction time rather than joining the final traffic reconstruction.

Location traces and driver identifiers may be personal data. The [EU General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj) applies in relevant circumstances; lawful basis, retention and access controls require context-specific review. Its existence does not imply that every ETA use is automatically lawful or prohibited. Apply local employment, privacy and consumer rules and obtain professional advice when needed.

UTC timestamps permit ordering across regions, while local civil time is needed for a customer's commitment and dock calendar. Store an IANA time-zone identifier where possible, and explicitly handle daylight-saving transitions and corrected timestamps.

### A practical data contract

| Field | Grain and example | Validation or operational use |
|---|---|---|
| Shipment and stop key | One destination target per row | Prevent mixing a vehicle's multiple stops |
| Target definition version | First verified gate-in, version 2 | Distinguish revised event definitions |
| Prediction timestamp | Instant at which forecast was issued | Set the as-of boundary |
| Source availability timestamp | Time the online consumer could use a record | Exclude information received later |
| Event timestamp | Time the source says the event occurred | Calculate freshness and elapsed time |
| Source identifier | Gate sensor, carrier scan, or dispatcher | Support provenance and conflict resolution |
| Route-plan version | Plan visible at prediction time | Avoid joining the eventual route |
| Commitment version | Original and approved amendments | Score the correct contractual deadline |
| Forecast version | Model, calibration and output identifiers | Reproduce the issued prediction |
| Forecast distribution | Quantiles or another defined representation | Support uncertainty and decision thresholds |
| Label status | Arrived, cancelled, diverted, censored, unresolved | Keep denominators honest |
| Intervention record | Recommendation, action and timestamps | Study operational feedback |

The schema should also specify valid units, allowed nulls, update cadence and ownership.
Latitude and longitude are not interchangeable, and a missing coordinate must not become `(0, 0)`.
A route-plan timestamp that is populated only when the database is backfilled is not a reliable
online availability timestamp. Quality rules must refer to what the consuming service actually saw.

### Clocks, corrections and event semantics

Use UTC instants for ordering, and preserve the relevant IANA time-zone identifier and local
commitment representation for display. In a fall daylight-saving transition, a local time such as
01:30 can occur twice. The local timestamp without an offset or disambiguation rule does not identify
a unique instant. In a spring transition, some local times do not exist.

If a gate sensor reports 12:04 but a later audit establishes 11:58, keep the original observation,
the correction, the correction's availability time and its reason. A report may be recomputed using
the corrected label, but a historical forecast replay must not act as though the correction was
known before it arrived. Maintain a clear policy for which label version supports contractual
reporting and which supports model evaluation.

GS1 EPCIS provides a structure for exchanging event information across organizations.
It does not guarantee that every participant sends an event promptly or that the physical event
matches the business target. Contractual service definitions, source reconciliation and freshness
checks remain necessary even when parties use a common event standard.

### Data rights and proportionality

Driver location can reveal sensitive patterns even if a report displays only an ETA.
Restrict access to raw traces, retain them only for a defined purpose and period, and separate
operational needs from employee performance monitoring. A derived ETA can itself be sensitive when
linked to an individual, vehicle or customer; removing a driver's name does not automatically make
the record anonymous.

The EU GDPR is a binding regulation where applicable, but its obligations depend on the facts,
including roles, purpose and jurisdiction. The official regulation is linked in section 16.
This lesson does not determine a lawful basis, a retention period, or whether an individual use is
permitted. Consult the organization's privacy and employment specialists before deployment.
Likewise, carrier contracts and traffic-provider terms can restrict use, retention or onward sharing.

For an external ETA provider, document whether source data may be stored, whether model outputs may
be shown to customers, and what happens when the provider is unavailable. Define deletion and
access procedures for linked operational records. Commercial permission and legal compliance are
separate questions.

## 9. Industry practice and case evidence

Google Maps Platform documents a [Routes API](https://developers.google.com/maps/documentation/routes) that can compute routes and travel-duration information. This is official product documentation, not independent evidence that its travel time alone predicts a particular shipment's gate-in: loading, remaining stops, breaks and facility queues can contribute additional uncertainty.

The US Bureau of Transportation Statistics publishes the [Airline On-Time Performance data](https://www.transtats.bts.gov/ONTIME/), including scheduled and actual flight event times. It demonstrates an official source of event-based transport outcomes, not a last-mile benchmark. Definitions and cancellation handling must be read before applying it as a teaching dataset. [INFERENCE] Comparing predicted and actual flight events could illustrate horizon and selection issues, but the operational target and inputs differ from a road shipment.

In either case, public documentation does not establish the calibration of a specific enterprise ETA model. A local backtest and operational pilot are required before drawing a deployment conclusion.

### Interpreting a route-provider output

The Google Maps Platform Routes API documentation describes route computation and travel-duration
information. A road duration can be a useful component of an ETA system, particularly for a vehicle
with a known position and destination. It does not establish the time a driver will finish a prior
stop, take a required break, pass a security gate, or wait for a dock.

An evaluator should ask whether the API response used for an old prediction was retained at the time
or reconstructed later. A current route query for a historical trip does not necessarily reproduce
the traffic information that was available to the original decision maker. The provider's product
documentation also does not report the enterprise-specific calibration required for a customer
promise or staffing rule.

**Operational implication:** treat a routing duration as an input or benchmark whose source,
timestamp, geographic coverage and fallback are documented. Validate the end-to-end target using
the enterprise's own event definitions and cohorts. Do not attribute the resulting model's quality
to the road provider without a separately supported comparison.

### Interpreting a public transport dataset

The US Bureau of Transportation Statistics publishes airline on-time performance data with
scheduled and actual flight event fields and cancellation information. This is a named public
source for studying transport event definitions and selection effects. Flight operations, airports,
flight schedules and recorded events are not interchangeable with truck gate-in or last-mile
customer delivery.

A teaching analysis can first define a specific flight event and reporting population, then account
for cancelled flights instead of forcing them into an arrival-delay calculation. It can stratify
results by reporting period and departure horizon where suitable prediction-time inputs exist.
The published actual event alone is not an ETA forecast. A valid forecast evaluation additionally
requires predictions or features that were available at the selected decision time.

**[INFERENCE]** A group exercise using this data can help learners practice event reconciliation and
missing-outcome accounting before applying the method to road shipments. It cannot establish a
road-carrier benchmark, last-mile model accuracy, or the value of a warehouse alert policy.

### Buy, build, or combine

| Decision question | Buy a provider forecast | Build internally | Combine |
|---|---|---|---|
| Coverage | Check contracted lanes and event types | Depends on internal integrations | Depends on both inputs and fallback |
| Target event | Verify the provider's arrival definition | Can match the local gate definition | Must reconcile incompatible targets |
| Historical replay | Request archived outputs and availability | Retain snapshots and feature history | Retain both sources as issued |
| Explanation | Verify available operational detail | Design explanations with operators | Distinguish input from final output |
| Dependency | Contract, feed latency and vendor continuity | Data, engineering and model ownership | Both dependencies remain |
| Evaluation | Compare against an actual baseline | Compare out of time | Evaluate the combined policy end to end |

Buying does not remove the need to check performance. Building does not guarantee that a company has
enough data for rare lanes. Combining outputs does not guarantee improvement: highly correlated
errors, missing source forecasts and mismatched events can make an ensemble worse.
Choose using matched-cohort scoring, full-population coverage, lifecycle cost and the economics of
the actual decision rather than a single headline accuracy number.

### A defensible pilot sequence

1. Agree the target, promise treatment, eligible shipments, decision owner and current baseline.
2. Audit a sample of timestamp provenance and reconcile all outcome states.
3. Produce point-in-time historical comparisons and document missing historical inputs.
4. Run the candidate in shadow alongside the current process with no hidden auto-actions.
5. Review alert volume and cases with dispatchers and customer-service staff.
6. Pilot an action policy with consent and safeguards appropriate to the operation.
7. Compare process outcomes and business value against a credible counterfactual where feasible.
8. Set ownership, fallback, monitoring and retirement conditions before wider rollout.

This sequence is course guidance, not a claim about a particular company's implementation.
Shadow accuracy alone does not show that staff can act on alerts or that interventions create value.
Conversely, a workflow can create value through communication even if it does not shorten travel.

## 10. Failure modes and red flags

| Failure | Warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Promise overwritten | Breaches disappear after updates | Forecast replaces commitment | Misleading service | Preserve immutable promise |
| Target ambiguity | Gate and dock results conflict | Different events combined | Wrong action | Define event and owner |
| Future leakage | Unrealistically good backtest | Final route in snapshot | Deployment failure | Availability-time join |
| Informative censoring | Fast arrivals dominate | In-transit records dropped | Optimistic ETA | Model censoring and mature cohorts |
| Cancellation mishandled | Extreme fabricated lateness | Competing event treated as arrival | Distorted tails | Separate outcomes |
| Wrong time zone | One-hour error spike | Daylight-saving conversion | Missed appointment | Store UTC and zone |
| Pooled calibration | Overall coverage looks good | Opposite group errors cancel | Unequal service | Stratify and quantify uncertainty |
| Wide intervals | High coverage but no utility | Uninformative range | Poor decisions | Report width and loss |
| Stale data | Constant ETA despite movement | Feed outage | False confidence | Freshness alerts and fallback |
| Intervention feedback | Rescued loads seem early | Actions change outcomes | Misread model quality | Log action and evaluate policy |
| Correlated split | Test error implausibly low | Same shipment in train and test | Inflated performance | Time and entity split |
| Unresolved scans | Coverage excludes hard cases | Missing labels | Selection bias | Reconcile and report missingness |

### Triage an apparently excellent score

Suppose the dashboard shows MAE falling from 38 to 25 minutes after a carrier feed migration.
Before celebrating, compare the number of eligible shipments, the matched-label rate, the mix of
forecast horizons and the distribution of lanes. If missing scans rose on difficult lanes, the
scored population may have become easier. If the new system issues forecasts closer to arrival,
the comparison may also be biased in favor of the new system.

Investigate the feed and reconcile the entire cohort before attributing improvement to the model.
Compare both models at the same timestamps on matched cases, and separately report the fallbacks.
If historical records cannot support this comparison, state the limitation rather than backfilling
future information into the old snapshots.

### Triage an apparently poor score after intervention

An alert predicts a high risk of missing noon. Dispatch calls the carrier, and the truck arrives
at 11:55. Labeling this a false alert assumes the truck would have arrived on time without the call.
That counterfactual is not observed. The event can be recorded as an on-time outcome, but the
business value and counterfactual risk need a separate policy evaluation.

Check whether action timestamps and reasons are recorded, whether the action can plausibly affect
arrival, and whether an appropriate comparison exists. Do not delete successful interventions from
the evaluation or assume that every intervention caused the outcome.

### Additional release red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Quantile crossing | Upper bound earlier than lower bound | Separately fitted quantiles conflict | Invalid interval | Constrain or repair and reassess calibration |
| Late alert | Lead time shorter than required action | Scoring ignores response time | Accurate but unusable queue | Measure actionable recall and lead time |
| Coverage mismatch | Candidate issued mainly for tracked lanes | Population selection | Exaggerated improvement | Matched and full-policy reporting |
| Overalerting | Acknowledgement backlog rises | Threshold ignores team capacity | Missed high-value cases | Rank by incremental benefit and monitor workload |
| Model-version loss | Decision cannot be linked to output | Updates overwrite earlier forecasts | No audit trail | Store immutable forecast identifiers |
| Wrong endpoint | Strict and inclusive deadlines are mixed | Rounding or ties ignored | Incorrect breach probability | State and implement deadline rule |
| Feed silent failure | Latest record ages but service remains green | Availability checked without freshness | Stale position treated as current | Alert on age and trigger fallback |
| Corrected-label leakage | Revised scan appears in old snapshot | History overwritten | Inflated replay accuracy | Keep correction availability and versions |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test |
|---|---|---|---|
| Target event is observable | Evaluate forecasts | Missing scans | Audit source reconciliation |
| Availability times are accurate | Prevent leakage | Backfilled timestamps | Compare ingestion logs |
| Censoring is modeled appropriately | Use partial follow-up | Informative dropout | Sensitivity analysis |
| Forecast distribution is calibrated | Probability-based action | Season or network shift | Rolling holdout |
| Linear early and late costs | Quantile decision rule | Fixed penalties | Scenario cost curve |
| Decision does not alter arrival | Simple expected cost | Dispatch rescue | Log intervention and evaluate policy |
| Group labels are reliable | Bias analysis | Missing or misleading attributes | Audit definitions |
| Sufficient group observations | Stable coverage estimate | Rare lanes | Intervals and appropriate pooling |
| Promise unchanged | Breach evaluation | Contract amendment | Version commitments |
| Historical data representative | Future forecast | New site or route | Out-of-time monitoring |

### Sensitivity before commitment

The single-truck cost rule is sensitive to the relative early and late costs.
Run at least a low, central and high scenario for costs that the business cannot estimate precisely.
If all plausible scenarios choose noon, the choice is robust to those assumptions.
If the chosen time ranges from 11:30 to 12:30, present that sensitivity to the decision owner
instead of reporting one quantile as if it were objectively correct.

An intervention's effectiveness is often less certain than its forecast risk.
For USD 30 of loss, a USD 4 action and 20% late probability, efficacy of 50% gives
`0.20 × 30 × 0.50 − 4 = −USD 1`.
Efficacy of 80% gives `0.20 × 30 × 0.80 − 4 = +USD 0.80`.
The break-even efficacy is `4 / (0.20 × 30) = 66.7%`.
This threshold is a decision aid, not evidence that efficacy actually reaches that level.

Calibration measured before a route redesign may not transfer after the redesign.
A new facility or carrier may have little labeled history, and low-volume subgroups may have
intervals too wide for confident conclusions. Use qualified baselines and an explicit review
period; avoid claiming that a pooled calibration result proves local performance.

The simple expected-cost model assumes the cost for this truck is separable from other trucks.
When two vehicles compete for one door, staffing one may delay the other. The correct decision
can require a joint schedule with capacity and shift constraints, not independent quantiles.
An ETA model supplies inputs to that schedule; it does not solve the resource allocation itself.

## 12. Exercises

1. **Event definition** (10 minutes). Define gate-in ETA, prediction time and one event that is not gate-in. Acceptance requires a measurable timestamp and source.
2. **Leakage** (10 minutes). A scan occurs at 09:00 but arrives in the system at 10:00. Can it be used in a 09:30 historical prediction? Explain. Acceptance requires availability-time reasoning.
3. **Censoring** (15 minutes). At the cutoff a shipment is still in transit and tracked. State what is known and why it is not an arrival. Acceptance requires a distinction from cancellation.
4. **Quantile** (15 minutes). Late cost is USD 9 per minute and early cost is USD 1 per minute. Compute the decision quantile under the linear model. Acceptance requires the formula.
5. **Bias** (15 minutes). Actual minus forecast errors are +10, +20 and −15 minutes. Compute mean signed error and MAE. Acceptance requires signs and units.
6. **Calibration** (15 minutes). A nominal 90% interval contains 90 of 100 observed arrivals, but coverage is 48 of 50 in one group and 42 of 50 in another. Calculate both group rates and explain the limitation. Acceptance requires the small-sample caveat.
7. **Design** (25 minutes). Propose a time-based backtest for repeated predictions and a guardrail for stale data. Acceptance requires avoiding future leakage and specifying a fallback.

8. **Alert economics** (20 minutes).
   A late shipment causes USD 80 of avoidable loss, an action costs USD 6, and the action removes
   25% of the loss when the shipment is late. The predicted late probability is 0.40.
   Assume the action has no other effect and can be completed on time.
   Deliverable: calculate expected net benefit and break-even probability.
   Acceptance: show the efficacy term and explain whether to act under these assumptions.

9. **Outcome reconciliation** (20 minutes).
   A 500-shipment cohort has 420 verified arrivals, 20 cancellations, 10 diversions,
   30 tracked shipments still at risk, and 20 unresolved records at cutoff.
   A report claims 90% interval coverage because 378 verified arrivals fall in the interval.
   Deliverable: reproduce the coverage and construct a cohort accounting table.
   Acceptance: distinguish observed-arrival coverage from full-cohort coverage and identify
   what additional information is needed before making a population claim.

10. **Release and rollback** (30 minutes).
    A candidate provider supplies a gate-arrival distribution for 75% of shipments.
    On 600 matched shipments its MAE is 22 minutes versus 28 minutes for the current baseline.
    The remaining shipments use the baseline, but the provider's feed can be delayed.
    Deliverable: write a release decision memo with an evaluation plan and operating controls.
    Acceptance: include matched versus full-population results, uncertainty, target equivalence,
    fallback freshness, alert capacity, ownership, and a measurable rollback trigger.
    Do not infer the full-population improvement from the matched comparison alone.

### Submission guidance

For exercises 1–3, supply a short event-state diagram or table rather than only a narrative.
For exercises 4–6 and 8–9, show the numerator, denominator, units and assumptions.
For exercise 7, define the prediction cadence and whether the evaluation weights shipments or
snapshots. For exercise 10, identify the person who can stop automated actions and what the
operator sees when fallback is active.

For every design answer, distinguish a forecast-quality claim from a policy-effect claim.
A good answer can recommend delaying release when essential data or a safe fallback is missing;
the explanation must identify the missing evidence and the decision affected by it.

## 13. Self-check answers

1. Example: predict first verified destination gate-in timestamp from data available at dispatch; the destination gate system supplies the event. Dock-door arrival is a distinct event.
2. No, if ingestion at 10:00 is the first time the prediction system could access it. A 09:00 event timestamp cannot make future knowledge available at 09:30.
3. Arrival has not been observed by the cutoff; if tracking establishes continued risk, the arrival time exceeds that cutoff. Cancellation prevents the original event and is a competing outcome, not merely a censored late arrival.
4. 9 / (9 + 1) = 0.90, the 90th percentile. This depends on the assumed linear costs and predictive distribution.
5. Mean signed error is (10 + 20 − 15) / 3 = +5 minutes; MAE is (10 + 20 + 15) / 3 = 15 minutes. Positive bias here means actuals are later on average.
6. Group coverage is 96% and 84% respectively. Aggregate coverage is 90%, but it masks a 12 percentage-point difference. Each group has only 50 outcomes, so uncertainty and selection should be examined before claiming a persistent gap.
7. A strong answer freezes as-of features, holds later periods out, keeps shipment snapshots together, includes mature and unresolved outcomes in a reconciliation, compares a baseline and suppresses or labels stale estimates. It logs later interventions.

### 8. Alert economics

The avoided loss conditional on a late shipment is `USD 80 × 0.25 = USD 20`.
At a late probability of 0.40, expected avoided loss is `0.40 × USD 20 = USD 8`.
Subtract the USD 6 action cost to get an expected net benefit of USD 2 per eligible shipment.
Break-even risk is `USD 6 / USD 20 = 0.30`, or 30%.
Under the stated assumptions, 40% exceeds the threshold, so acting has positive expected value.
An operational capacity constraint or different action efficacy can change that conclusion.

### 9. Outcome reconciliation

| Status | Shipments | Share of 500 |
|---|---:|---:|
| Verified arrival | 420 | 84% |
| Cancelled | 20 | 4% |
| Diverted | 10 | 2% |
| Tracked and still at risk | 30 | 6% |
| Unresolved | 20 | 4% |
| Total | 500 | 100% |

Among verified arrivals, coverage is `378 / 420 = 90%`.
The denominator is not 500; cancellation and diversion may prevent the target, and unresolved or
in-transit shipments do not yet have observed gate-in labels. A claim about eventual-arrival
coverage needs sufficient follow-up, outcome reconciliation and an assessment of selection.
Even a fully matured cohort needs uncertainty analysis and target-consistent handling of
non-arrival outcomes. It is not valid simply to divide 378 by 500 and call the result coverage.

### 10. Release and rollback rubric

A complete memo states that 22 versus 28 minutes is a six-minute MAE improvement on the
600 matched cases, not proof of a six-minute improvement for all shipments.
It asks for counts and outcomes by excluded group, a full-policy comparison, out-of-time periods,
shipment-level uncertainty and a check that both systems predict the same target at the same time.
It also reports probability calibration and width rather than evaluating only MAE.

The operating design identifies the accountable release owner, feed owner and control-tower owner.
It defines a measurable maximum age for provider data, labels the fallback and its timestamp,
and pauses automated actions when the freshness or validity rule fails.
The chosen threshold must be agreed with the operational owner; a numerical threshold not justified
by the business is not inherently safe. The plan also limits alerts to actionable capacity,
logs overrides and proposes a safe comparison for intervention value.

Award full credit to a recommendation to defer release if these prerequisites are not met.
Shadow comparison can establish production feasibility and matched forecast performance;
it cannot establish that an alert caused improved service or lower cost.

## 14. Interview questions

1. What is the predicted event? Strong answers specify gate-in or another event and source; weak answers say merely "delivery".
2. How is an ETA different from a promise? Strong answers preserve the contractual record; weak answers overwrite it.
3. What is leakage? Strong answers use availability time; weak answers rely only on event timestamp.
4. Why does dropping in-transit loads matter? Strong answers identify selection and censoring; weak answers call them missing at random without evidence.
5. What does 90% coverage mean? Strong answers describe population and uncertainty; weak answers promise every shipment 90% certainty.
6. How do you choose a quantile? Strong answers relate asymmetric costs to the decision; weak answers always choose the median.
7. Can aggregate calibration hide bias? Strong answers stratify and check missingness; weak answers cite the pooled number alone.
8. What happens after dispatch acts on an alert? Strong answers consider intervention feedback; weak answers assume outcomes are unaffected.

### Follow-up probes for an experienced candidate

9. **A vendor covers 75% of shipments and reports a lower MAE. Would you replace the incumbent?**
   A strong answer asks for common targets and snapshots, matched comparison, coverage of the
   remaining population, fallback performance, uncertainty and operational costs.
   A weak answer assumes the published score describes every eligible shipment.

10. **How would you score a 90th-percentile forecast and decide whether it is useful?**
    A strong answer calculates one-sided empirical coverage and pinball loss, then examines
    interval width or quantile sharpness, subgroups and the action's economics.
    A weak answer treats the percentile as a guarantee or selects the widest possible interval.

11. **A model's error rose after dispatchers started using its alerts. What next?**
    A strong answer separates the issued forecast from subsequent action, checks data quality and
    compares policies using a defensible counterfactual where feasible.
    A weak answer assumes a successful intervention proves the original risk was wrong.

12. **When would you pause automated ETA actions?**
    A strong answer names measurable triggers, an owner, fallback behavior, freshness labeling and
    how to communicate the change to operators. A weak answer waits for a customer complaint
    without an operational detection rule.

## 15. Key terms

- **Arrival:** defined target event; see section 5.
- **Availability time:** when a feature was accessible; see section 5.
- **Bias:** mean signed error under a defined convention; see section 4.
- **Calibration:** frequency agreement with forecasts; see section 5.
- **Censoring:** incomplete event-time observation; see section 5.
- **Competing event:** outcome precluding the target; see section 5.
- **Coverage:** share of outcomes in a forecast interval; see section 5.
- **Decision quantile:** percentile selected by asymmetric cost; see section 6.
- **ETA:** estimated time of arrival; see section 3.
- **Event time:** when an operational event occurred; see section 8.
- **Horizon:** lead time between prediction and outcome; see section 4.
- **Ingestion time:** when information entered the system; see section 5.
- **Leakage:** use of unavailable information; see section 5.
- **MAE:** mean absolute error; see section 5.
- **Median:** 50th percentile; see section 5.
- **Pinball loss:** quantile forecast scoring rule; see section 4.
- **Promise:** commercial commitment; see section 3.
- **Quantile:** percentile of a predictive distribution; see section 4.
- **Residual:** actual minus forecast here; see section 4.
- **Sharpness:** concentration of predictive intervals; see section 5.
- **Snapshot:** information frozen at prediction time; see section 5.

## 16. Sources

- GS1, [EPCIS](https://www.gs1.org/standards/epcis), supply-chain event standard.
- European Union, [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), official legal text.
- Google Maps Platform, [Routes API documentation](https://developers.google.com/maps/documentation/routes), product documentation.
- US Bureau of Transportation Statistics, [Airline On-Time Performance](https://www.transtats.bts.gov/ONTIME/), official data source.
- Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation," *Journal of the American Statistical Association*, 2007, [DOI](https://doi.org/10.1198/016214506000001437).

## 17. Links

- [Module index](_index.md)
- [Freight modes](01_freight-modes.md)
- [Rate structures](02_rate-structures.md)
- [Lane and tender analytics](03_lane-tender-analytics.md)
- [Last mile](08_last-mile.md)
- [Module survey](../../../course/12_transportation_routing_last_mile.md)
- [Simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
