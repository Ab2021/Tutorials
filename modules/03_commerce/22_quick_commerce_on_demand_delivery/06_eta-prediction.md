# ETA Prediction for On-Demand Delivery

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: probability, time-series validation, and [promise tiers](01_promise-tiers-dark-stores.md)  
> Version 2026.5

## 1. Why this topic exists

A customer deciding whether to order needs an honest answer to a practical question:
when will the order be in their hands?
A courier needs enough time to travel safely.
A store manager needs to know whether preparation or courier availability is the constraint.
A dispatcher needs to know whether adding a second order to a route will break a promise.
These are related decisions, but none is answered by a road travel time alone.

Estimated time of arrival (ETA) is a forecast of an operational event.
The event might be arrival at a building, but customers normally care about completed handoff.
A delivery estimate that stops at the curb can be accurate for navigation and misleading at checkout.
Likewise, a picker finishing a basket is not the same event as a courier collecting it.
Define the event before calculating the number.

The customer sees a range or deadline, not the underlying distribution.
The business chooses how to translate uncertain completion time into a promise.
A forecast describes what might happen; a target expresses an aspiration;
a plan assigns resources; a commitment states what the customer can rely on.
Treating these as synonyms hides the cost of overpromising.

A short quoted time may improve checkout conversion while increasing late deliveries,
refunds, support contacts, food-temperature risk, and pressure on couriers.
A long quote can protect the on-time metric while causing customers to abandon the order.
Improving the estimate and choosing the promise are therefore separate jobs.
A calibrated model does not decide the right service guarantee by itself.

The analytics owner supplies an auditable distribution and monitors its errors.
The product owner owns presentation and customer communication.
The operations owner controls capacity, dispatch, and escalation.
Safety and legal owners review constraints and claims.
No prediction should be used as a reason to exceed safe speeds or ignore rest.

This topic develops an event-level method for preparation, courier approach,
travel, and handoff, including their dependence and uncertainty.
It closes with a reproducible synthetic calculation and practical controls.
The numeric case is illustrative, not an observed company performance claim.

## 2. Learning outcomes

After this topic you can:

1. Draw an event timeline and identify overlapping work without double-counting time.
2. Compute a deterministic completion estimate from preparation, courier, travel, and handoff inputs.
3. Construct and interpret a probabilistic completion estimate and a customer promise.
4. Explain why adding marginal component percentiles does not generally produce a total percentile.
5. Design point-in-time training data, a temporal evaluation, and a simple baseline.
6. Diagnose calibration, segment performance, and operational feedback loops.
7. Specify customer communication, dispatch response, ownership, and safety guardrails.
8. Evaluate a rollout and define a rollback decision with measurable criteria.

Exercises 1–8 respectively test these outcomes.
Use minutes for durations and a named time zone for wall-clock timestamps.
Unless otherwise stated, intervals are elapsed time and include waiting.

## 3. The core mental model

```text
Checkout accepted at t0
       |                            |
       v                            v
 preparation starts           courier assignment/approach
       |                            |
       v                            v
 order ready at store          courier arrives at store
       +----------- max ------------+
                    |
                pickup complete
                    |
          route and road travel
                    |
       building access and handoff
                    |
          customer receives order
                    |
         compare with promise saved at t0
```

Preparation and courier approach can run in parallel.
Pickup cannot occur until both the order and courier are ready.
The later branch controls the earliest feasible pickup.
If a pickup scan happens after both branches finish, the additional time is pickup service.
After pickup, road travel and destination access are usually sequential.
The final event is completed handoff, not the driver's arrival near the address.

For a simplified single-order delivery, define all durations from acceptance in minutes:

```text
completion = max(preparation, courier_ready) + pickup_service
             + travel + destination_handoff
```

The formula assumes the pickup service starts only after both branches finish.
If a courier begins loading before preparation is complete, redefine the events rather than
forcing the observed process into this formula.
If pickup service is already included in travel or preparation, remove it from that component.
Component definitions must be mutually exclusive along each sequential path.

The total is random at checkout.
Rain, order volume, store congestion, and courier availability can affect several stages at once.
The uncertainty in the maximum and the sum is consequently not obtained by simply adding
independent error bars or by summing marginal 90th percentiles.
A joint sample of complete order paths can preserve dependence.
Alternatively, a model can predict the end-to-end completion distribution directly.
Both approaches require out-of-time validation.

At checkout the model uses only information available then.
Later, the estimate can be refreshed when preparation finishes, a courier is assigned,
or a route changes.
The first customer promise remains a historical record even if a later estimate moves.
Evaluate the original promise against actual completion, and evaluate each refresh
against its own prediction time and information set.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Acceptance | Platform accepts the order; timestamp in UTC plus local display zone. | Treating app-open time as acceptance. |
| Prediction time | Instant at which a forecast is issued. | Using a later feature as if known at checkout. |
| Preparation duration | Acceptance to order-ready event, elapsed minutes in this example. | Assuming all minutes are active picking. |
| Courier-ready duration | Acceptance to courier at pickup, elapsed minutes. | Counting only vehicle travel and omitting assignment. |
| Ready event | Store confirms the whole order is available for pickup. | Using first item picked as ready. |
| Pickup service | Time from both branches ready to pickup completion, minutes. | Adding it twice when the pickup scan is already the travel start. |
| Pickup completion | Courier takes custody of the order. | Confusing store arrival with custody. |
| Travel duration | Pickup completion to destination arrival, minutes. | Using straight-line distance as travel time. |
| Handoff duration | Destination arrival to completed customer receipt, minutes. | Omitting building access, parking, or customer response. |
| Completion | Successful handoff or a separately defined terminal outcome. | Mixing failed attempts with successful completion. |
| ETA | Estimated time of the specified arrival or completion event. | Assuming every ETA means the same event. |
| Remaining time | Forecast duration from the current prediction time. | Comparing it directly to an acceptance-based duration. |
| Forecast | Distribution or statistic describing uncertain outcomes. | Calling a service target a forecast. |
| Point estimate | One summary, such as median or mean, in minutes. | Treating it as a guaranteed deadline. |
| Quantile | Value at which a specified fraction of the predictive distribution is at or below it. | Assuming a 90th percentile is a worst case. |
| Prediction interval | Lower and upper values intended to contain a stated fraction of outcomes. | Confusing an interval for an individual outcome with uncertainty in a mean. |
| Calibration | Agreement between predicted probabilities and observed frequencies over suitable groups. | Claiming calibration for every individual order. |
| Sharpness | Concentration of a predictive distribution, subject to calibration. | Rewarding narrow ranges that systematically miss. |
| Promise | Customer-facing commitment with an explicit event and deadline or window. | Replacing the original promise after the fact. |
| On-time rate | Share completed by the applicable deadline among a defined cohort. | Excluding cancellations without reporting them. |
| Lateness | Positive part of completion minus deadline, minutes. | Ignoring the magnitude of late cases. |
| Bias | Average signed prediction error, with sign convention specified. | Calling all large errors bias. |
| Mean absolute error | Average absolute point-prediction error, minutes. | Using it alone to choose a promise quantile. |
| Coverage | Fraction of outcomes inside a stated prediction interval. | Evaluating only delivered orders without attrition disclosure. |
| Pinball loss | Asymmetric loss for a forecast quantile, in minutes. | Assessing a 90th-percentile forecast with mean absolute error alone. |
| Censoring | Outcome is only partly observed by the analysis cutoff. | Labeling an open order as having zero duration. |
| Leakage | Use of information unavailable at prediction time. | Treating a final route as known at checkout. |
| Batch | Two or more orders served in a combined courier route. | Treating a batch as independent direct trips. |
| Drift | Change in data, relationships, or performance over time. | Assuming all drift requires model retraining. |
| Fallback | Approved estimate or promise policy when normal scoring is unavailable. | Returning an arbitrary optimistic default. |
| Time zone | Rules for converting an instant to local wall-clock display. | Subtracting ambiguous local clock strings across daylight-saving changes. |

## 5. The mechanics, step by step

### 5.1 Choose the decision and event contract

1. Name the decision: display a checkout window, trigger an intervention,
   or select between feasible dispatch options.
2. Name the prediction timestamp, completion event, and population.
3. Specify units, clock handling, and treatment of canceled and failed orders.
4. Record who owns the event definitions and who approves changes.

For checkout, the event can be successful customer receipt after order acceptance.
For a failed handoff, record the failure event and reason rather than inventing a receipt time.
A promise performance report must include the share canceled, failed, and still open;
otherwise a seemingly improved on-time rate can be caused by excluding difficult orders.

**Input:** product journey, event schema, customer promise wording.
**Output:** versioned contract describing start, end, clock, cohorts, and exclusions.
**Branch:** if receipt cannot be observed reliably, first improve capture or explicitly
label the proxy and quantify the difference on an audited sample.

### 5.2 Map operational states

Draw the actual workflow with store and courier timestamps.
At minimum capture acceptance, order ready, courier arrival at pickup,
pickup completion, destination arrival, and completed handoff.
Add assignment, reassignment, preparation start, and route-change events where available.
Keep original event timestamps and ingestion timestamps separately.
The latter reveal late-arriving data and help reproduce what was actually known.

The parallel-branch maximum is appropriate only when readiness and courier arrival
are separately observed and pickup depends on both.
A restaurant order may have a different preparation lifecycle from a dark-store basket.
A customer pickup order may omit a courier branch entirely.
Use a different state graph when the physical process differs.

**Input:** operational workflow and raw event stream.
**Output:** state diagram and a mapping from raw event types to canonical events.
**Branch:** if scans are missing, mark the component unknown; do not infer exact times
from later status updates without a validated imputation model.

### 5.3 Construct examples without future information

Create a training row for each chosen forecast moment.
Use the latest eligible feature version that had arrived by that moment.
Examples include store backlog, basket count, historical store preparation rates,
road-network travel estimate, courier availability, and forecast weather available then.
A weather observation recorded after checkout is not a checkout feature.
A final courier identifier is not available before assignment.
A final realized route is not available before route selection.

Store features as-of the forecast time and attach a feature-availability timestamp.
Make labels from the eventual completed event, but never expose label-derived values as features.
When evaluating refresh forecasts, retain the prediction time as well as the order identifier.
Repeated snapshots from one order are correlated; group uncertainty estimates by order,
and consider time and store clustering when reporting confidence intervals.

**Input:** event log, feature histories, model decision times.
**Output:** point-in-time feature table and eventual outcomes with provenance.
**Branch:** when feature arrival times are absent, replay conservatively or exclude the feature.

### 5.4 Make the baseline explicit

Start with a simple, auditable historical estimate.
For example, estimate the median and 90th percentile of completed durations
for the same store, hour-of-week band, and broad distance band from prior weeks.
Smooth sparse bands toward a larger store or regional cohort.
Freeze the training cutoff before evaluation.
Do not calculate a cohort percentile using the very order being scored.

Also calculate a mechanistic baseline using the parallel-branch maximum.
A model should add value over both the historical baseline and the operational estimate.
The simple baseline remains useful as a fallback if a model service or feature feed fails.
A fallback needs its own calibration review; it is not automatically safe.

**Input:** historical completed orders before the cutoff and a hierarchy of cohorts.
**Output:** baseline median, promise quantile, sample size, and fallback rule.
**Branch:** if a local group is sparse, use a preapproved broader cohort and disclose it.

### 5.5 Estimate each branch and the whole path

Preparation can depend on basket composition, substitutions, queue length,
store staffing, item locations, and whether cold or regulated items need special handling.
The estimate should reflect elapsed queue and service time, not just picking speed.
A missing item may lead to substitution or cancellation; model such outcomes separately
or explicitly condition the duration model on successful completion.

Courier readiness can include matching delay, acceptance or rejection,
travel to the store, parking, and entry.
A courier's nominal distance from the store does not reveal whether they are available.
A reassignment can reset the relevant approach time.
Availability signals need timestamps and a defined staleness threshold.

Pickup service may reflect queueing at the counter and custody confirmation.
Destination travel may depend on road network, mode, traffic, weather, route,
stop sequence, and local restrictions.
Handoff may depend on building access, parking, elevator, customer instructions,
and customer response; use sensitive access data only when justified and permitted.

A direct model of completion can learn interactions without an explicit decomposition.
A component model helps operational diagnosis and can reuse new information at refresh time.
Neither is automatically superior: evaluate both end to end and by component.
Check that component outputs share event boundaries and are not double-counted.

**Input:** point-in-time features and historical outcomes.
**Output:** conditional distribution of total remaining minutes, optionally with components.
**Branch:** if a component is not identifiable due to missing events, use an end-to-end model
and avoid unsupported attribution to that component.

### 5.6 Represent uncertainty and dependence

One practical approach is to sample complete historical residual vectors
from relevant prior orders and add them to current component predictions.
Preserve the joint vector rather than independently shuffling each component.
Then apply the state graph to each sampled path and calculate total quantiles.
Another approach is a direct conditional quantile model trained on total duration.
A third is an empirically calibrated residual interval around a point model.
Whichever method is used, check calibration out of time and across meaningful groups.

Residual sampling assumes the selected history is relevant to current conditions.
If a severe event has no comparable history, samples may underestimate the tail.
Use operational scenario analysis and a conservative policy for such conditions;
do not describe an arbitrary safety margin as a statistically guaranteed percentile.
Conformal methods can provide marginal coverage under their assumptions,
but exchangeability can fail under temporal drift and selection.
A nominal coverage claim is not a universal per-order guarantee.

For a quantile level q, pinball loss for observed duration y and estimate p is
`max(q * (y - p), (q - 1) * (y - p))`.
When q is 0.90, underprediction is penalized nine times as heavily per minute
as an equally sized overprediction.
This scoring rule measures quantile quality; it does not set the business cost of delay.

**Input:** joint historical residuals or a fitted conditional distribution.
**Output:** samples or quantiles of complete duration, with method and training period.
**Branch:** if observed coverage is poor or a feature is unavailable, use the approved fallback.

### 5.7 Convert prediction to a customer promise

The model produces a forecast distribution.
Product and operations choose a policy based on missed-promise cost,
customer value, capacity, safety, and business constraints.
For instance, a 90th-percentile deadline is intended to be met for about 90% of
comparable orders if the distribution is calibrated and the operating policy remains stable.
It is not a guarantee for every order and should not be called one without the appropriate
commercial and legal basis.

If a display window uses the 20th and 90th percentiles, its nominal coverage is 70%,
not 90%, under a continuous calibrated distribution.
Round the lower end down and the upper end up to the chosen display increment.
Evaluate coverage after rounding, not just on unrounded model outputs.
If a policy uses a fixed earliest displayed time, explicitly evaluate early arrivals too.
Never hide uncertainty by changing the reference deadline after an order is placed.

A promise can also be an eligibility decision.
When predicted probability of meeting a service tier falls below a policy threshold,
offer a longer tier, scheduled delivery, or no immediate service.
Do not silently shorten a quote to protect conversion.
Customer-facing words must explain what the quoted range represents and how updates work.

**Input:** predictive distribution, policy, operating constraints, and display rules.
**Output:** quoted window or deadline plus policy version and timestamp.
**Branch:** if the quote exceeds the supported service tier, route to an approved alternative.

### 5.8 Update without rewriting history

Refresh on meaningful changes: preparation ready, assignment, pickup, route change,
or a material new disruption.
At each refresh, condition only on information actually received at that time.
Once pickup has occurred, remaining time begins at the current instant;
do not add the already elapsed preparation time a second time.
A new prediction can improve guidance, but the original checkout promise is immutable.
Retain each version, reason, customer display, and notification status.

Use a threshold or hysteresis to avoid notifying customers about every small fluctuation.
A threshold is a product decision, not a statistical property of the model.
Notify promptly when a meaningful delay is likely, with a new range and available remedies.
Avoid repeatedly moving the deadline without clearly acknowledging the earlier commitment.

**Input:** new verified events, current state, prior forecast and commitment.
**Output:** versioned remaining-time forecast and, where required, customer notification.
**Branch:** if event freshness fails, mark the estimate stale and use a state-specific fallback.

### 5.9 Act on risk without creating unsafe incentives

An operations console can rank orders by probability of missing their original deadline,
expected lateness, food-safety exposure, or customer vulnerability.
Each ranking requires a defined action: investigate store backlog, reassign within policy,
contact the customer, or offer a remedy.
An alert without an owner and action only creates noise.

A dispatcher should account for the effect of a change on all orders in a batch.
A reassignment may improve one order while worsening several others.
The optimization must respect courier availability, legal constraints, and safe travel.
A prediction is an estimate, not an instruction to speed or skip safety checks.
Escalate systematic capacity deficits to the operations owner rather than pressuring workers.

**Input:** current forecast, immutable promises, feasible interventions, and safety rules.
**Output:** prioritized action with owner, timestamp, and outcome.
**Branch:** when no safe feasible action exists, communicate delay and apply the remedy policy.

### 5.10 Validate and release

Split by time, not random rows, for the primary evaluation.
Train on an earlier period, tune on a later period, and test on a still later period.
Use a gap when labels or features mature slowly.
Include a full business cycle and relevant peaks if the available history permits.
Document promotions, weather, holidays, service-area changes, and staffing shifts.

Compare mean absolute error for point estimates, pinball loss for quantiles,
interval coverage and width, deadline on-time rate, and lateness magnitude.
Break down by store, local hour, mode, distance, basket, batch status,
and other lawful operational slices with enough support.
Report cohort size and uncertainty intervals; suppress unstable small-cell conclusions.
Compare the same orders and prediction timestamps across candidate and baseline.

Before launch, shadow score without changing customer promises.
Then consider a controlled experiment with a prespecified unit of randomization,
metrics, duration, guardrails, and analysis plan.
Marketplace interference can invalidate order-level independence because orders share
couriers and capacity; a cluster or time-block design may be more appropriate.
Keep the old model and an approved rollback procedure available.

**Input:** frozen candidate, historical holdout, shadow outputs, and release criteria.
**Output:** decision record, experiment plan, dashboard, and rollback owner.
**Branch:** fail closed to the approved fallback if the service or critical data is unhealthy.

### 5.11 Monitor and improve

Monitor event delivery latency, missingness, feature freshness, score failures,
forecast distribution, calibration, actual lateness, and user-visible quote changes.
Report completed, canceled, failed, and open order counts separately.
Delayed labels mean a recent cohort may appear better or worse than it really is;
use matured cohorts and show the observation cutoff.

The operations lead investigates spikes in real waiting and travel.
The data owner investigates missing events and clock anomalies.
The model owner investigates calibration and segment shifts.
The product owner reviews communication and remedies.
A change in a real process can be a capacity problem rather than a model problem.
Retraining on an incident without understanding the operational cause can normalize failure.

### 5.12 Specify event-level acceptance tests

A data contract is useful only if producers and consumers can test it.
Write tests against event sequences before building a model.
For an ordinary successful order, acceptance precedes order readiness.
Courier arrival and order readiness may occur in either order.
Pickup completion follows both events under the case's state definition.
Destination arrival follows pickup completion.
Handoff follows destination arrival.

The checks above are logical constraints, not instructions to rewrite raw history.
If the source system records a late scan, distinguish event time from ingestion time.
If the clock on a device was wrong, record the correction and its provenance.
If the process permits a partial collection, define a separate event for it.
An exception can reveal either invalid data or a missing business state.

Use a small set of explicit sequences as fixtures:

| Sequence | Expected interpretation | Required response |
|---|---|---|
| Ready before courier | Courier branch is binding. | Calculate maximum from observed event times. |
| Courier before ready | Store branch is binding. | Preserve courier wait as elapsed time. |
| Simultaneous readiness | Neither branch has a positive lead. | Record both timestamps at source precision. |
| Pickup before both | Invalid under this contract. | Investigate scan or change the contract. |
| No destination scan | Handoff may still be observable. | Mark component missing; do not fabricate it. |
| Duplicate pickup scan | Same event may be retried. | Deduplicate with an event identifier and source rules. |
| Order canceled before pickup | No successful handoff exists. | Record cancellation and exclude from conditional duration label. |
| Failed first handoff | A later attempt may occur. | Keep attempt events and define terminal policy. |

If a scan is optional, the training pipeline must not treat absence as zero duration.
For example, a missing destination-arrival event makes the division between travel
and handoff unknown, even if pickup and completed receipt are known.
The end-to-end label remains available if acceptance and receipt are trustworthy.
The component model may need a missing-data strategy or a narrower supported population.

### 5.13 Build a historical cohort deliberately

Start with accepted orders in a specified geography and time period.
Join each order to the original quote if one was shown.
Assign each order a final status at a fixed observation cutoff.
Keep successful, canceled, failed, and open counts in a cohort ledger.
For each count, state the fraction of accepted orders.
An order can change status after the cutoff, so freeze the extraction date.

For completed orders, calculate elapsed time from the prediction instant to receipt.
For canceled or failed orders, retain the terminal event and its elapsed time separately.
For open orders, retain elapsed time to the observation cutoff and mark them open.
These values answer different questions and should not share an undifferentiated label.

Consider 1,000 accepted orders in an illustrative matured cohort:

| Outcome | Count | Share of accepted orders |
|---|---:|---:|
| Completed | 930 orders | 93.0% |
| Canceled | 45 orders | 4.5% |
| Failed handoff | 15 orders | 1.5% |
| Still open at cutoff | 10 orders | 1.0% |

Suppose 837 of the 930 completed orders met their original deadline.
Conditional on completion, the on-time rate is `837 / 930 = 90.0%`.
The fraction of all accepted orders both completed and on time is `837 / 1,000 = 83.7%`.
These are different estimands, not competing arithmetic answers.
The second quantity does not say why the other 163 orders were not on-time completions.
Report the outcome table so an analyst can interpret both measures.

If cancellation is caused by a long delay, completed-only evaluation is selected.
For example, an updated policy might appear to improve on-time performance
because dissatisfied customers cancel before a late completion can be observed.
Measure cancellation and customer support contacts, and consider the timing of cancellation.
Avoid labeling a cancellation as necessarily caused by ETA without evidence.

### 5.14 Select features for the operational question

The same raw source can yield eligible and ineligible features.
For checkout, the number of items in the accepted basket is ordinarily known.
The final number of substituted items is not yet known.
For an in-progress refresh, the number already substituted can be eligible
if its event was ingested before the refresh time.
Record which definition is used rather than naming both columns `substitution_count`.

Store backlog must also have a precise definition.
It might mean accepted orders not yet marked ready at a particular instant.
It might instead mean active picking tasks, including partially completed orders.
Those measures can diverge during a substitution or pickup queue.
Choose the definition that matches the operational hypothesis and the reliable feed.
Never calculate historical backlog using status corrections unknown at the time.

Courier supply is similarly stateful.
An online courier may be unavailable because they are already handling a delivery,
have declined a task, or cannot travel to the store within the service area.
An aggregate count should state the radius, modes, time window, and availability rule.
If the count is stale, pass a freshness indicator and use the approved fallback policy.
The model owner must not silently interpret a missing feed as zero available couriers.

Routing inputs should distinguish the distance to pickup from the route after pickup.
Road distance, estimated travel time, and straight-line distance are different fields.
The road-network version, travel mode, and route calculation time should be recorded.
The model must know whether a value describes a proposed route or the actual route.
Proposed route features may be appropriate at dispatch but not at checkout.

Weather inputs should be the forecast that was actually available at prediction time.
Store the issue time and valid time of the forecast.
For a nowcast, record when the observation was produced and received.
Historical weather data downloaded later may have been revised.
Use the archived forecast version for a faithful offline replay.

### 5.15 Compare distributions, not only averages

Suppose two models make predictions for the same 1,000 matured completed orders.
Model A has a median absolute error of 4 minutes and 82% coverage at its claimed
90th-percentile deadline.
Model B has a median absolute error of 5 minutes and 90% deadline coverage.
These hypothetical metrics alone do not prove which model has better business value.
Model B may quote substantially longer deadlines and lose orders at checkout.
Model A may generate more late orders and support contacts.

Add interval width, average quoted deadline, pinball loss, observed lateness,
and coverage by relevant segment.
Also report accepted-order volume and cancellation rates when comparing policies.
The experiment should capture economic and service outcomes, not just model scores.
If assignment decisions change, record those policy changes.
The choice belongs to product and operations with analytics and safety input.

For a 90th-percentile deadline, compare observed on-time frequency with 90%
within groups of forecasts having comparable operating conditions.
If a model reports an explicit probability of meeting a fixed deadline,
group predictions into probability bands and compare forecast probabilities
with observed frequencies in each band.
Report the number of observations and uncertainty for each band.
Small samples can make a perfectly reasonable model look poor by chance.

Avoid evaluating an interval solely by coverage.
An interval from 0 to 180 minutes may cover most deliveries,
but it gives little useful information for a 30-minute service.
Coverage and sharpness must be considered together.
Similarly, a narrow interval with poor coverage overstates certainty.
There is no single universally correct trade-off independent of the service decision.

### 5.16 Understand changes caused by the model

An ETA is not always a passive observation.
An unusually long quoted time may lead a customer not to place an order.
An accepted-order training dataset then omits those abandoned sessions.
A faster quote may increase demand and create new congestion.
A dispatch optimizer may preferentially assign couriers to orders predicted to be late.
These actions can change the distribution the model is meant to predict.

Log the candidate quote, whether it was displayed, and whether an order was placed,
subject to the applicable privacy and retention rules.
Do not assign a hypothetical delivery duration to an abandoned session.
Conversion can be analyzed as a separate outcome in an appropriate experiment.
For operational decisions, record the intervention and time it was applied.
Without that record, a model may learn the consequences of an old policy
as though they were unchangeable properties of the order.

When randomization is acceptable, define the decision unit before launch.
An order-level experiment is simple but orders in the same area share courier capacity.
If treatment shifts capacity from one order to another, outcomes can spill across arms.
Store, geographic zone, or time-block randomization may reduce some spillovers.
Each choice has costs: fewer independent units, changing demand mix, or boundary effects.
Use the design and analysis that match the marketplace, and document remaining interference.

### 5.17 Define an operational exception ladder

The first response to an invalid feature is not necessarily retraining.
Classify the issue by scope and severity.
An isolated missing store-readiness event may need a store-level review.
A region-wide courier feed outage may require a broader fallback.
An actual road closure may require a longer promise or temporary service restriction.
A major weather event may require human operational judgment in addition to a forecast.

Example ladder, with illustrative thresholds that a local owner must set:

| Level | Signal | Decision owner | Possible response |
|---|---|---|---|
| Normal | Feeds fresh and matured coverage acceptable. | Model service owner. | Use the approved model and policy. |
| Degraded | One noncritical feature is stale. | On-call data owner. | Use documented imputation or fallback and alert. |
| Material | Critical feed absent across a store or zone. | Operations incident lead. | Apply a conservative approved quote or pause the affected tier. |
| Severe | No safe feasible service under current conditions. | Operations and safety leads. | Suspend affected offers and communicate with customers. |

The thresholds must specify a measurement window, denominator, and recovery condition.
A single delayed event should not necessarily suspend an entire region.
Conversely, an aggregate average can conceal a severely affected store.
Use both local and aggregate views, with a human escalation route.

### 5.18 Review model changes and data changes separately

A new model version can be numerically unchanged while a feature definition changes.
For example, a backlog feed may start counting orders awaiting substitution.
That is a semantic change even if the column name remains the same.
The feature owner should publish the definition and effective timestamp.
The model owner should assess whether backfills and live data are comparable.

An event producer might change when it emits the ready scan.
If a scan moves from picker completion to bag sealing, the measured preparation
duration increases even if physical work and customer service are unchanged.
Examine the event definition before concluding that store productivity worsened.
Run old and new definitions in parallel where feasible.
Do not splice incompatible series without annotation.

For every release, retain model artifact, feature schema, transformation version,
training cutoff, evaluation cohort, policy version, and approval record.
Maintain the old approved artifact for rollback subject to security and retention policy.
The record should make it possible to reconstruct a historical forecast
without claiming that today's data warehouse exactly matches what was known then.

## 6. Worked example

### 6.1 Case and assumptions

At 18:02 local time, a customer places one grocery order.
The store is open, the address is serviceable, and the order is not yet assigned.
The business wants a handoff estimate at checkout and an original deadline.
All quantities below are illustrative minutes from acceptance unless stated otherwise.

The order has 11 items, a 2.7 km road route, and one delivery stop.
The order is not batched in the initial plan.
A route estimate is based on the road network, not straight-line distance.
The assumptions do not guarantee that the order will remain unbatched.

The model's illustrative point inputs are:

| Component | Estimate | Interpretation |
|---|---:|---|
| Preparation | 12 min | Acceptance to complete order ready. |
| Courier ready | 16 min | Acceptance to courier at pickup. |
| Pickup service | 2 min | After both are ready to custody. |
| Travel | 9 min | Custody to destination arrival. |
| Handoff | 4 min | Arrival to completed customer receipt. |

The point estimate is `max(12, 16) + 2 + 9 + 4 = 31 min`.
At 18:02 this corresponds to 18:33 local time.
Preparation and courier readiness overlap, so adding 12 and 16 would overcount.
This 31-minute calculation is a scenario, not necessarily the mean or median
of the total random duration: `E[max(P, C)]` need not equal `max(E[P], E[C])`.

### 6.2 Joint scenarios

Suppose five equally weighted illustrative joint scenarios have already been selected
from relevant historical conditions, with the same component boundaries as above.
Five paths are far too few for a production percentile; they make the mechanics visible.

| Path | Preparation | Courier ready | Pickup | Travel | Handoff | Total |
|---|---:|---:|---:|---:|---:|---:|
| A | 10 min | 12 min | 2 min | 8 min | 3 min | 25 min |
| B | 12 min | 15 min | 2 min | 9 min | 4 min | 30 min |
| C | 14 min | 17 min | 2 min | 10 min | 5 min | 34 min |
| D | 18 min | 16 min | 3 min | 12 min | 6 min | 39 min |
| E | 21 min | 23 min | 4 min | 15 min | 8 min | 50 min |

For path D, preparation controls pickup readiness: `max(18,16) = 18 min`.
Total completion is `18 + 3 + 12 + 6 = 39 min`.
For path E, the total is `23 + 4 + 15 + 8 = 50 min`.
Because whole rows remain together, simultaneous congestion in path E affects
multiple components; reshuffling columns would destroy that dependence.

The sorted total durations are 25, 30, 34, 39, and 50 minutes.
Use the nearest-rank empirical quantile: sorted value at position `ceil(n*q)`,
with positions numbered from one.
The empirical median is the third value, 34 minutes.
The empirical 20th percentile is the first value, 25 minutes.
The empirical 90th percentile is the fifth value, 50 minutes.
With so few paths, both the tail estimate and its coverage are highly unstable.

A product policy might display a 25–50 minute range and a 50-minute upper deadline.
That range is a policy translation of the sample, not a guarantee.
At 18:02 it corresponds to approximately 18:27–18:52 local time.
The example does not claim that a five-path estimate is operationally acceptable.
In production, choose sample size and a calibration approach using holdout evidence.

### 6.3 Observe and update

Suppose the actual order is ready at 18:14, 12 minutes after acceptance.
The courier arrives at the pickup location at 18:18, 16 minutes after acceptance.
Pickup completes at 18:20, destination arrival is 18:31,
and the customer receives the order at 18:36.

The observed total is 34 minutes.
The observed components are 12, 16, 2, 11, and 5 minutes.
The state-graph check gives `max(12,16) + 2 + 11 + 5 = 34 min`.
The point scenario predicted 31 minutes, so its signed error
`prediction - actual` is `31 - 34 = -3 min`, an underprediction.
Absolute error is 3 minutes.
The original 50-minute deadline was met with 16 minutes to spare.
The actual completion also fell in the 25–50 minute displayed range.
One successful order cannot establish 90% calibration.

At pickup, 18 minutes have elapsed since acceptance.
A new forecast might use 11 minutes of remaining travel and 5 minutes of handoff,
yielding a 16-minute point scenario and a predicted completion at 18:36.
The original 18:52 deadline stays in the event record.
The refresh is assessed against its own 18:20 prediction timestamp,
while the original customer commitment is assessed from checkout.

### 6.4 Business interpretation

The courier branch was the binding pre-pickup branch by 4 minutes:
`16 - 12 = 4 min`.
This describes the observed path; it does not prove that assigning a courier
4 minutes earlier would reduce completion by exactly 4 minutes.
The intervention might change wait time, costs, or other orders' outcomes.
A controlled analysis is needed for causal claims.

The forecast, operating plan, customer commitment, and target differ:

| Object | Example | Owner |
|---|---|---|
| Forecast | Illustrative 34-minute empirical median. | Analytics. |
| Plan | One courier, one order, estimated 2.7 km road route. | Dispatch. |
| Commitment | Original displayed upper deadline of 18:52. | Product and commercial policy. |
| Target | Example management objective of 90% of eligible orders by deadline. | Operations and product. |
| Actual | Handoff at 18:36, 34 minutes after acceptance. | Recorded by operations systems. |

A target of 90% is a hypothetical management choice, not a universal standard.
A single-case on-time result does not establish target attainment.

### 6.5 A delayed courier scenario

Keep the same checkout and the same original 18:52 deadline.
Now consider a separate hypothetical order with preparation ready at 18:14,
but courier arrival at 18:29 because assignment took longer than expected.
Suppose pickup completes at 18:32, destination arrival is 18:44,
and handoff completes at 18:50.

The resulting elapsed durations are:

| Interval | Start | End | Duration |
|---|---|---|---:|
| Preparation | 18:02 | 18:14 | 12 min |
| Courier ready | 18:02 | 18:29 | 27 min |
| Pickup service | 18:29 | 18:32 | 3 min |
| Travel | 18:32 | 18:44 | 12 min |
| Handoff | 18:44 | 18:50 | 6 min |

The state equation gives `max(12,27) + 3 + 12 + 6 = 48 min`.
The 50-minute original deadline is met by 2 minutes.
However, the 31-minute point scenario underpredicts this order by 17 minutes.
Meeting a conservative deadline does not imply the point estimate was accurate.
It also does not imply the customer received timely updates during the wait.

Suppose at 18:24 the forecast changes materially because no courier has accepted.
The product policy says to notify the customer when the upper forecast exceeds
the original deadline by a specified material threshold.
That threshold must be chosen and tested before the incident.
If triggered, the notification should mention the delay and provide a new estimate;
the 18:52 original deadline remains recorded for service reporting.
If the order ultimately arrives by 18:52, the original promise was still met.
The notification is evaluated separately for timeliness and usefulness.

The incident record should include the number of assignment attempts,
the time of each attempt, and the state of eligible couriers.
It should also note whether the store had completed preparation.
These facts help distinguish a courier shortage from a stale availability feed.
They do not by themselves prove that individual couriers acted incorrectly.

### 6.6 An access-delay scenario

Consider another separate hypothetical order with the same initial point estimate.
Preparation takes 12 minutes, courier readiness 16 minutes,
pickup service 2 minutes, and travel 9 minutes.
The courier arrives at the building 27 minutes after acceptance:
`max(12,16) + 2 + 9 = 27 min`.
At that point the point scenario expected a further 4-minute handoff.

Suppose the building entry system is unavailable and handoff actually takes 18 minutes.
Completion takes `27 + 18 = 45 min`, at 18:47.
The customer receives the order 5 minutes before the original 18:52 deadline.
The curb-arrival timestamp of 18:29 would understate actual customer wait by 18 minutes.
If a dashboard stops its clock at destination arrival,
it will miss the very delay that matters in this scenario.

An access issue may be rare at the network level but common at a particular site.
Analyze aggregated patterns only where data use is justified and permitted.
Avoid using sensitive building information in ways unrelated to delivery.
Offer a customer a way to correct inaccurate access instructions.
The data owner should define access control and retention for those instructions.

The operations response is not automatically to blame the courier.
Review the evidence: arrival event reliability, contact attempts,
entry instructions, and the customer's reported experience.
If the access failure is systematic, address the site-level process.
If the arrival event is unreliable, improve measurement before attributing cause.

### 6.7 Promise-policy sensitivity

Return to the five synthetic path totals: 25, 30, 34, 39, and 50 minutes.
They have equal weights only for this teaching example.
Compare three possible deadlines without claiming production-level calibration.

| Deadline | Sample paths on time | Sample on-time fraction | Lateness among all five paths |
|---|---:|---:|---:|
| 34 min | 3 of 5 | 60% | `(0 + 0 + 0 + 5 + 16) / 5 = 4.2 min` |
| 39 min | 4 of 5 | 80% | `(0 + 0 + 0 + 0 + 11) / 5 = 2.2 min` |
| 50 min | 5 of 5 | 100% | `0 / 5 = 0 min` |

A longer deadline mechanically increases on-time performance for this fixed sample.
It may also make the offer less attractive, but the sample contains no conversion data.
Therefore the table cannot identify the best commercial promise.
It also cannot estimate future 90% coverage reliably from only five observations.
Business owners need a larger, time-separated evaluation and, where appropriate,
a controlled policy experiment with customer, economic, and safety guardrails.

The 39-minute deadline has 11 minutes of lateness on path E.
If a report averages lateness only over late orders, its value is 11 minutes.
If it averages lateness over all five orders, the value is 2.2 minutes.
Both calculations can be useful, but their denominators must be explicit.
For path E, the customer experiences an 11-minute miss, not a 2.2-minute miss.

### 6.8 Operations handoff for the case

At checkout, the product service records the 18:27–18:52 displayed window.
The analytics service logs the five illustrative path totals and policy version.
The dispatch service records the initial single-order plan.
The store records readiness at 18:14.
The courier system records arrival at 18:18 and pickup at 18:20.
The destination and handoff events close the order at 18:31 and 18:36.

At the end of the day, the analytics owner checks event completeness and latency.
The operations owner reviews whether delays reflect capacity or execution.
The product owner compares the customer-visible quote with the server log.
The safety owner reviews any complaint or signal of pressure to travel unsafely.
Customer support owns the remedy under the applicable customer policy.
The store or courier should have a route to correct a disputed event.

The team does not conclude from one 34-minute delivery that the model is calibrated.
It places the order in its correct matured cohort and compares it with many others.
The team keeps the original quote even if a later forecast was more precise.
These records connect the model's technical evaluation to the actual customer experience.

## 7. Code, query, or calculation

The following Python 3 script uses only the standard library.
It reproduces the five-path calculation and scores the observed order.
The arrays are illustrative joint paths, not independent component samples.
Run it as a script; no network, package installation, or data file is required.

```python
from datetime import datetime, timedelta
from math import ceil

# Each tuple contains preparation, courier ready, pickup service,
# travel, and handoff, all in elapsed minutes.
paths = [
    (10, 12, 2, 8, 3),
    (12, 15, 2, 9, 4),
    (14, 17, 2, 10, 5),
    (18, 16, 3, 12, 6),
    (21, 23, 4, 15, 8),
]


def completion_minutes(path):
    preparation, courier_ready, pickup, travel, handoff = path
    if min(path) < 0:
        raise ValueError("Durations must be nonnegative")
    return max(preparation, courier_ready) + pickup + travel + handoff


def nearest_rank(values, q):
    if not values or not 0 < q <= 1:
        raise ValueError("Need values and 0 < q <= 1")
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * q) - 1]


def pinball(actual, prediction, q):
    error = actual - prediction
    return max(q * error, (q - 1) * error)


totals = [completion_minutes(path) for path in paths]
low = nearest_rank(totals, 0.20)
median = nearest_rank(totals, 0.50)
high = nearest_rank(totals, 0.90)
point = completion_minutes((12, 16, 2, 9, 4))
actual = completion_minutes((12, 16, 2, 11, 5))
start = datetime(2026, 5, 12, 18, 2)
print("Path totals (min):", totals)
print("Empirical q20/q50/q90 (min):", low, median, high)
print("Point and actual (min):", point, actual)
print("Point error prediction - actual (min):", point - actual)
print("Absolute error (min):", abs(point - actual))
print("q90 pinball loss (min):", round(pinball(actual, high, 0.90), 1))
print("On time:", actual <= high)
print("Displayed local window:",
      (start + timedelta(minutes=low)).strftime("%H:%M"), "-",
      (start + timedelta(minutes=high)).strftime("%H:%M"))
```

Expected output:

```text
Path totals (min): [25, 30, 34, 39, 50]
Empirical q20/q50/q90 (min): 25 34 50
Point and actual (min): 31 34
Point error prediction - actual (min): -3
Absolute error (min): 3
q90 pinball loss (min): 1.6
On time: True
Displayed local window: 18:27 - 18:52
```

The pinball loss is `(1 - 0.90) * (50 - 34) = 1.6 min`.
This penalizes overprediction at the 90th percentile less than the same-sized underprediction.
The program uses a naive local datetime solely to display a same-day illustrative example.
Production systems should store instants in UTC, retain the applicable IANA time zone,
and use timezone-aware conversion for customer display and daylight-saving transitions.
Nearest-rank quantiles are deliberately specified because software libraries use
different interpolation conventions; record the convention in a production contract.

## 8. Data, standards, and regulatory context

### 8.1 Event and data contract

A usable event record contains an order identifier, event type, event timestamp,
ingestion timestamp, event source, and correction or version metadata.
For courier events, attach an appropriate pseudonymous courier identifier where needed.
Keep time zones and clock synchronization policies explicit.
A correction must not silently erase the earlier event as it was known to a prior prediction.

Typical event checks include negative durations, pickup before readiness,
completion before destination arrival, duplicate events, missing acknowledgments,
and implausibly long intervals.
A negative duration can indicate bad clocks or event semantics rather than physical impossibility.
Quarantine or flag the record; do not automatically set negative durations to zero.
When a prediction is made before a late event arrives, its feature snapshot must not
be retroactively updated with that event.

A forecast log should contain the event being predicted, prediction instant,
model and feature versions, forecast distribution or quantiles, original quote,
policy version, and whether the customer actually saw the quote.
Keep the eventual outcome separate from the prediction record.
This makes offline replay and promise auditing possible.

### 8.2 Standards and conventions

ISO 8601 defines representations for dates and times; it does not define which
operational event counts as a completed delivery.
Use unambiguous offsets or UTC instants and keep a separate time-zone identifier
when local civil-time rules matter.
A time-zone offset alone does not encode future daylight-saving rules.

GS1 EPCIS is a standard for sharing visibility event data about the what, when,
where, why, and how of objects in supply chains.
Its vocabulary can support interoperable event exchange when organizations choose it;
it does not prescribe an ETA model, guarantee scan accuracy, or require every
last-mile platform to implement EPCIS.
Internal event contracts may be simpler, provided their semantics are documented.

### 8.3 Privacy and worker data

Precise locations, addresses, access instructions, and courier trajectories can be
personal data depending on context and jurisdiction.
Collect only what is needed for the specified operational purpose,
restrict access, set retention limits, and assess re-identification risk.
Aggregate dashboards should not expose small groups or individual home locations unnecessarily.
Audit access to detailed route and handoff records.

The EU General Data Protection Regulation (GDPR), where applicable, includes
principles of purpose limitation, data minimization, accuracy, and storage limitation
in Article 5, and requires a lawful basis under Article 6.
These are legal obligations within the regulation's scope, not optional modeling conventions.
Whether a particular use is lawful depends on facts and jurisdiction.
Automated decision-making rules, worker protections, and local employment law may also apply;
seek qualified advice rather than assuming that an ETA model is exempt.
This educational discussion is not legal advice.

Separate the ETA's operational purpose from worker discipline or compensation decisions.
A prediction error can reflect traffic, building access, or missing data rather than behavior.
Do not infer misconduct from lateness alone.
Provide a route for affected workers to challenge incorrect event records.

### 8.4 Selection and missing outcomes

Only successful deliveries have a successful handoff timestamp.
A duration model trained on them estimates time conditional on successful completion,
not the probability that every accepted order will be delivered.
Report cancellation, failed handoff, and open-order rates alongside conditional ETA quality.
For orders not completed by the extract cutoff, preserve the cutoff and censoring status.
Do not treat their observed elapsed time as a completed duration.

A survival or competing-risks method may be appropriate when the question requires
explicitly modeling time to multiple outcomes, but its assumptions must be evaluated.
At minimum, conduct sensitivity analysis and report how much of the original cohort
is excluded from the main ETA metric.

## 9. Industry practice and case evidence

### 9.1 Public engineering evidence

Uber Engineering's published article on predicting food delivery times discusses
end-to-end estimation and the multiple stages between order placement and delivery.
It demonstrates that marketplace ETAs require more than a road-navigation estimate.
The article is a company engineering account, not an independently audited estimate
of performance for every city or current model version.
Its architectural details and measured performance, if any, should be read in their
original temporal context rather than assumed to apply today.

DoorDash Engineering has published work on delivery-time prediction and its
machine-learning platform.
The relevant transferable practice is to connect operational event data,
features available at the decision time, and evaluation of delivery outcomes.
A public engineering post establishes what its authors describe at publication;
it does not prove another operator will achieve the same results.

### 9.2 Evidence versus inference

The event-graph method in this topic is an educational abstraction.
It is not represented as the internal algorithm of either named company.
**Inference:** an operator with substantial store queueing and assignment delays
is likely to gain diagnostic value from component-level measurements.
The size of that gain must be demonstrated with that operator's own data.

**Inference:** improved ETA accuracy can help customers plan and support teams prioritize,
but a conversion or retention improvement is a causal business claim requiring a suitable test.
Forecasts also affect behavior: a longer promise may change which customers order,
and a dispatch policy based on ETA changes the outcomes the next model observes.
A before-and-after comparison alone cannot isolate these effects.

### 9.3 Operational adoption

A useful release begins with a shared definition of the customer event.
Store teams validate readiness and pickup timestamps.
Dispatch teams validate courier states and route-change capture.
Customer support reviews how late notifications and remedies appear.
Analytics verifies point-in-time data, calibration, and comparisons to the baseline.
Product approves the displayed range and wording.
Safety owners review whether incentives can induce unsafe conduct.

At each handoff between teams, identify the accountable owner and the artifact:
event contract, model card, quote policy, runbook, or experiment decision record.
Review not just average error but the order categories with the greatest practical harm.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Road-only ETA | Curb arrival is accurate but receipt is late. | Preparation and access are omitted. | Misleading customer expectation. | Predict the handoff event and measure each interval. |
| Summed parallel branches | Predicted duration exceeds observed time systematically. | Preparation and courier approach overlap. | Needlessly long quote. | Use the later readiness event, then add sequential stages. |
| Double-counted pickup | Total exceeds sum reconstructed from scans. | Service interval is included in two components. | Bias and confusing diagnostics. | Version mutually exclusive event boundaries. |
| Future route leakage | Offline scores greatly outperform live scores. | Realized route was not known at checkout. | Failed rollout. | Enforce as-of joins and replay feature availability. |
| Independent residual sampling | Tail coverage falls during disruption. | Shared congestion is removed by shuffling components. | Understated uncertainty. | Preserve joint paths and validate by condition. |
| Sparse tail cohort | Quantile changes sharply after a few orders. | Too little tail data. | Unstable promises. | Pool cautiously and show sample size. |
| Delivery-only selection | On-time rate rises while cancellations rise. | Difficult orders disappear from the denominator. | False service improvement. | Report all terminal states and attrition. |
| Immature labels | Recent on-time rate looks unusually high. | Slow open orders have not finished. | Premature decisions. | Use matured cohorts and disclose cutoff. |
| Stale courier availability | Assigned courier repeatedly needs reassignment. | State feed is delayed. | Late pickup and worker frustration. | Monitor freshness and switch to fallback. |
| Clock mismatch | Negative or impossible durations. | Devices disagree or time zones are misread. | Corrupt labels. | Synchronize clocks and quarantine anomalies. |
| Batch omitted | Second stops consistently run late. | Direct-trip assumption ignores stop order. | Unequal service and food exposure. | Model route sequence and batch impacts. |
| Promise overwritten | Historical quote always matches latest forecast. | Updates replace original records. | Unmeasurable broken commitments. | Store immutable quote versions and display events. |
| Range interpreted as guarantee | Support complaints despite nominal coverage. | UI wording hides residual risk. | Trust and legal exposure. | Explain range and review customer-facing claims. |
| Handoff proxy | Large destination-to-receipt gap. | GPS arrival is used as customer receipt. | Systematic underprediction. | Measure handoff and audit proxy error. |
| Safety pressure | Unusual speed or shortcut reports rise. | Staff are rewarded only for on-time rate. | Worker and public harm. | Use safety guardrails and prohibit speed incentives. |
| Retraining during incident | New model normalizes unusually long delays. | Operational failure enters training without diagnosis. | Persistent poor service. | Investigate cause and approve data windows. |
| Alert fatigue | Many alerts have no action or acknowledgment. | Thresholds ignore capacity to respond. | Critical orders missed. | Assign owners and tune action-oriented alerts. |
| Rounding bias | Displayed range coverage differs from model coverage. | UI rounds both endpoints in the same direction. | Misleading promises. | Evaluate displayed values and round outward. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Both parallel branches start at acceptance. | Makes durations comparable in the example. | Prep or matching began earlier. | Align actual event origins; redefine durations. |
| Pickup follows both readiness events. | Justifies the maximum. | Partial pickup begins before order-ready scan. | Audit workflow and revise state graph. |
| Component intervals do not overlap after pickup. | Makes addition valid. | Handoff clock starts before travel clock ends. | Reconcile event boundaries on sampled orders. |
| Historical paths resemble current conditions. | Supports empirical simulation. | New store, storm, labor disruption. | Backtest by regime and activate fallback. |
| Joint paths retain dependence. | Preserves shared congestion. | Component residuals independently resampled. | Compare joint tail coverage on holdout. |
| Outcomes are recorded accurately. | Labels define actual service. | Inferred receipt or missing scan. | Audit against independent customer and operational evidence. |
| Completed orders represent the modeled population. | Allows conditional duration interpretation. | Cancellations concentrate among delayed orders. | Report all outcomes and conduct sensitivity analysis. |
| Prediction features existed at forecast time. | Supports live reproducibility. | Late data or post-outcome joins. | Replay ingestion timestamps and reject leakage. |
| The model population is sufficiently supported. | Makes segment metrics interpretable. | New geography or rare hour. | Report counts and broaden or withhold unsupported estimates. |
| Policy remains stable during evaluation. | Supports interpretation of calibration. | Dispatch changes in response to scores. | Version policies and evaluate by exposure. |
| Time conversion is correct. | Connects duration to customer clock. | DST ambiguity or device clock drift. | Store UTC instants and verify zone conversion. |
| Promises are presented as logged. | Makes quote audit meaningful. | Client caches or display fails. | Log render events and reconcile client/server. |

A well-calibrated aggregate forecast can still be badly calibrated for a particular
store or time of day.
Conversely, a tiny segment can show apparent miscalibration by chance.
Use sample sizes and uncertainty intervals before acting on segment differences.
Forecast accuracy also does not establish a causal effect of a dispatch intervention.

## 12. Exercises

### Exercise 1 — Map the process

A customer orders at 12:00.
The store is ready at 12:13, the courier reaches the store at 12:10,
pickup completes at 12:15, destination arrival is 12:24,
and handoff completes at 12:29.

**Deliverable:** a timeline, component durations, and total completion duration.
**Acceptance:** identify the binding branch and do not add parallel waits twice.
**Time:** 10 minutes.

### Exercise 2 — Calculate the point estimate

At checkout, estimated preparation is 14 minutes and courier readiness is 19 minutes.
Pickup service is 3 minutes, travel is 8 minutes, and handoff is 6 minutes.

**Deliverable:** a completion duration and the arithmetic.
**Acceptance:** use the stated event boundaries and explain the parallel portion.
**Time:** 5 minutes.

### Exercise 3 — Translate a distribution

A calibrated continuous duration distribution has a 20th percentile of 24 minutes
and a 90th percentile of 46 minutes.
The customer-facing display rounds outward to 5-minute increments.

**Deliverable:** the displayed range and its nominal unrounded coverage.
**Acceptance:** distinguish the 90th-percentile deadline from the interval coverage.
**Time:** 10 minutes.

### Exercise 4 — Challenge a percentile claim

A colleague adds the 90th percentile of preparation to the 90th percentile of
courier readiness and calls the result the 90th percentile of pickup readiness.

**Deliverable:** a short critique and an alternative calculation approach.
**Acceptance:** address both overlap and dependence, without claiming independence solves all cases.
**Time:** 10 minutes.

### Exercise 5 — Find leakage

At checkout a candidate feature table contains final courier ID,
current basket size, actual destination arrival, and last available weather forecast.
Assume the courier is not yet assigned, basket size is final at checkout,
and the weather forecast was published and ingested before checkout.

**Deliverable:** classify each feature and justify the decision.
**Acceptance:** distinguish feature availability from usefulness and protect the outcome.
**Time:** 10 minutes.

### Exercise 6 — Score and diagnose

Four matured completed orders have actual durations of 20, 30, 40, and 50 minutes.
Their point estimates are 22, 27, 35, and 54 minutes.
Their original deadlines are 25, 35, 38, and 55 minutes.

**Deliverable:** mean absolute error, signed bias using prediction minus actual,
on-time rate, and mean lateness over all four orders.
**Acceptance:** show arithmetic, units, denominator, and one limitation of this sample.
**Time:** 15 minutes.

### Exercise 7 — Design the operations response

A store's 90th-percentile promise coverage falls for three consecutive matured days.
Assignment events are often stale, but actual store wait has also risen.

**Deliverable:** a brief incident plan with owner, diagnostic, fallback,
customer communication, and safe operational action.
**Acceptance:** separate data failure from real capacity deterioration;
retain the original promise and avoid unsafe courier pressure.
**Time:** 20 minutes.

### Exercise 8 — Design a release

A new model improves offline average absolute error by 1 minute,
but its 90th-percentile coverage is lower in long-distance orders.
The model may affect dispatch and therefore shared courier capacity.

**Deliverable:** a release recommendation, experimental unit, primary metrics,
guardrails, and rollback criteria.
**Acceptance:** address calibration, interference, sample size, and accountable decision ownership.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

Preparation is 13 minutes and courier readiness is 10 minutes.
The store is the binding branch: `max(13,10) = 13 min`.
Pickup service is 2 minutes, travel is 9 minutes, and handoff is 5 minutes.
Total is `13 + 2 + 9 + 5 = 29 min`.
The customer receives the order at 12:29.

### Exercise 2

The estimated duration is `max(14,19) + 3 + 8 + 6 = 36 min`.
The 14-minute preparation and 19-minute courier readiness are parallel.
The courier branch is binding in this point scenario.

### Exercise 3

The displayed range is 20–50 minutes after rounding outward.
The unrounded 24–46 minute interval has nominal coverage of `90% - 20% = 70%`.
Actual coverage after rounding must be measured; it is not necessarily exactly 70%.
The upper endpoint is a 90th-percentile deadline only under the stated calibration assumptions.

### Exercise 4

Pickup readiness is the maximum of the two branch durations, not their sum.
Even for sequential components, summing marginal percentiles is generally not the percentile
of their sum, because the joint distribution and dependence matter.
Sample joint paths and compute the maximum on each path, or fit the readiness distribution directly.
Validate the resulting percentile on later observations.

### Exercise 5

Final courier ID is unavailable before assignment and must not be used at checkout.
Final basket size is eligible under the stated assumption, though its usefulness is empirical.
Actual destination arrival is a future outcome and is leakage.
The weather forecast is eligible only because its publication and ingestion preceded checkout;
its validity and value still require testing.

### Exercise 6

Absolute errors are 2, 3, 5, and 4 minutes, giving `14 / 4 = 3.5 min` mean absolute error.
Signed errors are 2, -3, -5, and 4 minutes, giving `-2 / 4 = -0.5 min` bias.
Three of four orders meet their original deadline, so on-time rate is 75%.
Only the third order is late, by 2 minutes; mean lateness over all four is `2 / 4 = 0.5 min`.
The four-order sample is too small to establish stable performance or population coverage.
It also excludes any canceled or failed orders by construction.

### Exercise 7 rubric

A strong answer assigns an incident commander and separate data, model, operations,
and product owners.
It checks feed latency and event completeness, compares live and fallback forecasts,
and investigates actual preparation and courier queues.
It logs original commitments, communicates meaningful delays, and offers approved remedies.
It adds capacity or widens/withdraws the offer where feasible rather than encouraging unsafe travel.
It records resolution and follows up on both data and process causes.

### Exercise 8 rubric

A strong answer does not launch solely on average error improvement.
It requires acceptable long-distance calibration and compares uncertainty with adequate support.
It proposes shadow evaluation followed by a suitable store/time or geographic cluster experiment
if interference makes order-level randomization unreliable.
It specifies quote coverage, lateness, cancellation, conversion, safety, and operational cost
as appropriate, with prespecified thresholds and a rollback owner.
It distinguishes the analytics recommendation from product and operations approval.

## 14. Interview questions

1. **Why is road travel time not a delivery ETA?**
   Strong: distinguishes preparation, courier approach, pickup, destination access, and handoff.
   Weak: equates arrival at the curb with customer receipt.
2. **How do parallel preparation and courier approach combine?**
   Strong: uses the maximum under explicit event assumptions.
   Weak: blindly adds both durations.
3. **Why can a sum of 90th percentiles fail?**
   Strong: explains overlap, dependence, and the distribution of the total.
   Weak: assumes percentiles add like deterministic durations.
4. **What is point-in-time correctness?**
   Strong: reconstructs what was published and ingested before the prediction.
   Weak: uses a final route because it is present in the warehouse.
5. **How do you assess a 90th-percentile deadline?**
   Strong: uses coverage, pinball loss, cohort definitions, and tail slices.
   Weak: relies only on mean absolute error.
6. **What happens to the original promise after a refresh?**
   Strong: preserves it and evaluates refreshes separately.
   Weak: replaces the original deadline with the latest estimate.
7. **How do canceled orders affect ETA reporting?**
   Strong: reports all outcome states and recognizes selection or censoring.
   Weak: excludes cancellations without disclosure.
8. **What would cause a calibrated model to fail after deployment?**
   Strong: identifies drift, changed policy, stale feeds, and feedback effects.
   Weak: assumes historical calibration is permanent.
9. **How do you decide whether to notify a customer?**
   Strong: balances materiality, timing, actionability, and communication ownership.
   Weak: sends every small numerical change or hides material delays.
10. **How do you prevent unsafe incentives?**
    Strong: separates forecast from instruction and enforces safety and capacity guardrails.
    Weak: treats courier speed as the sole lever for lateness.

## 15. Key terms

- **Acceptance:** the start event in the example; see [definitions](#4-definitions-and-vocabulary).
- **As-of join:** a join restricted to data available by forecast time; see [feature construction](#53-construct-examples-without-future-information).
- **Baseline:** simple comparison or fallback model; see [baseline](#54-make-the-baseline-explicit).
- **Batch:** multiple orders on one route; see [failure modes](#10-failure-modes-and-red-flags).
- **Bias:** mean signed error under an explicit sign convention; see [Exercise 6](#exercise-6--score-and-diagnose).
- **Calibration:** agreement of predicted and observed frequencies; see [uncertainty](#56-represent-uncertainty-and-dependence).
- **Censoring:** partial observation of an outcome; see [missing outcomes](#84-selection-and-missing-outcomes).
- **Commitment:** customer-facing promise, distinct from a forecast; see [case interpretation](#64-business-interpretation).
- **Completion:** the defined successful handoff event; see [event contract](#51-choose-the-decision-and-event-contract).
- **Coverage:** fraction of outcomes inside an interval; see [Exercise 3](#exercise-3--translate-a-distribution).
- **Courier readiness:** time to courier arrival at pickup; see [component modeling](#55-estimate-each-branch-and-the-whole-path).
- **Drift:** change affecting inputs or model performance; see [monitoring](#511-monitor-and-improve).
- **ETA:** estimated time of a specified event; see [mental model](#3-the-core-mental-model).
- **Fallback:** approved alternative when normal scoring is unavailable; see [baseline](#54-make-the-baseline-explicit).
- **Handoff:** destination arrival to customer receipt; see [case](#63-observe-and-update).
- **Leakage:** information from after forecast time used as input; see [Exercise 5](#exercise-5--find-leakage).
- **Lateness:** positive difference between completion and deadline; see [Exercise 6](#exercise-6--score-and-diagnose).
- **Mean absolute error:** average magnitude of point errors; see [scoring](#exercise-6--score-and-diagnose).
- **Nearest-rank quantile:** ordered observation at `ceil(n*q)`; see [joint scenarios](#62-joint-scenarios).
- **Pinball loss:** asymmetric scoring rule for a quantile; see [uncertainty](#56-represent-uncertainty-and-dependence).
- **Prediction interval:** interval intended to contain a specified share of outcomes; see [promise policy](#57-convert-prediction-to-a-customer-promise).
- **Preparation:** acceptance to order-ready elapsed time in this case; see [definitions](#4-definitions-and-vocabulary).
- **Quantile:** a specified cumulative-probability threshold; see [joint scenarios](#62-joint-scenarios).
- **Refresh:** a new forecast made after new information arrives; see [updates](#58-update-without-rewriting-history).
- **Selection:** inclusion of only some outcomes in the analysis; see [missing outcomes](#84-selection-and-missing-outcomes).
- **Sharpness:** concentration of a forecast distribution; see [definitions](#4-definitions-and-vocabulary).

## 16. Sources

- ISO, [ISO 8601-1:2019, Date and time — Representations for information interchange](https://www.iso.org/standard/70907.html).
  The standard concerns representation, not delivery event semantics.
- GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis).
  Event visibility and interoperability context; adoption does not imply accurate scans.
- European Union, [Regulation (EU) 2016/679, GDPR, Articles 5 and 6](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
  Primary legal text; applicability requires jurisdiction-specific assessment.
- Gneiting, T. and Raftery, A. E. (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437), *Journal of the American Statistical Association*.
  Background on probabilistic forecast scoring.
- Koenker, R. and Bassett, G. (1978), [Regression Quantiles](https://doi.org/10.2307/1913643), *Econometrica*.
  Foundational quantile-regression reference.
- Angelopoulos, A. N. and Bates, S. (2023), [Conformal Prediction: A Gentle Introduction](https://arxiv.org/abs/2107.07511).
  Explains coverage assumptions and limitations; temporal data require care.
- Uber Engineering, [Meet Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/).
  Company account of delivery-time prediction; not independent validation.
- DoorDash Engineering, [Engineering blog](https://careersatdoordash.com/blog/category/engineering/).
  Public company engineering material; search the archive for current delivery-time work and publication dates.
- Python Software Foundation, [`datetime` documentation](https://docs.python.org/3/library/datetime.html).
  Defines Python date/time behavior and aware versus naive objects.

Sources support the particular contextual statements above, not the numerical example,
which is synthetic and constructed for instruction.

## 17. Links

- [Module 22 index](_index.md).
- [Module 22 survey](../../../course/22_quick_commerce_on_demand_delivery.md).
- [Promise tiers and dark-store design](01_promise-tiers-dark-stores.md).
- [Short-horizon forecasting](02_short-horizon-forecasting.md).
- [Dark-store assortment](03_dark-store-assortment.md).
- [Replenishment and waste](04_replenishment-waste.md).
- The survey's lab section provides the module's practical assessment context.
