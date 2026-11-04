# ETA Prediction

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 4 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06, and 12; Topics 01–03  
> Version 2026.5

## 1. Why this topic exists

A carrier says a truck should arrive at 14:00 local time.
The receiving manager needs to know whether to hold a 15:00 dock slot, call in overtime, or ask for a later appointment.
A customer-service agent needs to decide whether an existing delivery commitment is at risk.
The inventory planner needs to decide whether a transfer is worth buying.
None of these decisions is answered by a single timestamp alone.

An estimated time of arrival (ETA) is a forecast of an operational milestone.
It is not automatically an appointment, a service target, or a customer promise.
An arrival at a gate is also not the same as unloading, receipt into available inventory, or delivery to the customer's site.
Confusing those milestones can turn a numerically accurate model into a failed business process.

A useful estimate includes uncertainty and the time at which information was available.
An 80% prediction interval describes a range intended to contain the actual milestone in about 80% of comparable future cases under the stated evaluation conditions.
Its width matters because planners must reserve resources around uncertainty.
Its upper end is not a guaranteed latest arrival, and it is not necessarily an appropriate promise.
A separate probability of missing a decision deadline is often more useful than an interval alone.

The distribution changes as a journey unfolds.
A new departure scan can narrow a range; a missed connection can widen or shift it.
A missing scan does not establish that the vehicle has stopped.
The tower must distinguish stale evidence from a genuine disruption.

Predictions are also affected by what is not observed.
At an analysis cutoff, many active shipments have not yet arrived.
Discarding them makes completed fast shipments look typical and can make performance look better than it is.
Cancellations, returns, shipment splitting, and missing proof of delivery require explicit handling rather than being silently called arrivals.

The aim is a decision system that combines a defensible distribution, business costs, feasible actions, and the authority to communicate.
This lesson presents a reproducible baseline and a worked decision; it does not claim that the illustrative data represent a real carrier.

## 2. Learning outcomes

After this topic you can:

1. Define the forecast target, observation cutoff, and competing operational milestones; exercise 1 checks the distinction.
2. Construct a leakage-safe historical ETA baseline and explain its limitations; exercises 2 and 4 test the method.
3. Calculate a prediction interval, deadline risk, and a cost-sensitive action; exercises 2 and 3 require arithmetic.
4. Explain right censoring and distinguish it from cancellation and missing labels; exercise 4 asks for a cohort design.
5. Separate forecast, plan, target, and commitment; exercise 5 requires a communication record.
6. Choose retrospective and prospective evaluation measures, including interval coverage and decision outcomes; exercise 6 applies them.
7. Design escalation, monitoring, and safeguards for changing conditions; exercises 7 and 8 assess the operating design.

## 3. The core mental model

```text
Shipment + destination + target milestone + prediction cutoff
             |
             v
Events known by cutoff ---> quality and freshness checks ---> as-of feature snapshot
             |                                                |
             |                                                v
             |                                     predictive distribution
             |                                       /       |        \
             |                               point ETA    interval    P(miss deadline)
             |                                       \       |       /
             |                                        v      v      v
             +----------------------------------> feasible actions and cost
                                                       |
                                                       v
                                              authorized decision
                                                       |
                                    +------------------+-----------------+
                                    v                                    v
                             internal plan                       customer commitment
                                    \                                    /
                                     v                                  v
                                  actual milestone and intervention record
                                                       |
                                                       v
                                  calibration, decision review, and correction
```

The inputs are facts available at a particular cutoff, not the final journey record.
The output is a distribution for one defined milestone, conditional on that information.
A point estimate is one summary of this distribution, not the whole forecast.
An interval communicates a chosen band of outcomes.
The deadline risk asks a different question: what fraction of the predicted distribution lies after the actual operational cutoff?

The policy layer combines the forecast with the costs, constraints, and owner of each action.
For example, a planner may reserve a backup dock while keeping the existing customer commitment unchanged.
A second policy determines if and when an authorized representative changes that commitment.
A revised forecast alone does not authorize a change to a contractual promise.

The bottom loop records both outcomes and interventions.
If a team expedites the riskiest shipments, their observed arrival times are partly caused by intervention.
A model scored on those shipments must not treat the improved outcomes as proof that the initial warnings were false.
Separate predictive calibration from the causal value of acting on the prediction.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| ETA | Forecast time of a named arrival milestone, with time zone | A promise or a guaranteed latest arrival |
| Target milestone | Physical or recorded event being forecast, such as gate-in | Treating gate-in as goods available for sale |
| Prediction cutoff | Instant at which a prediction is made | Using later events to reconstruct earlier forecasts |
| Event time | When the underlying activity occurred, if known | Assuming it was already visible to the predictor |
| Processing time | When the system received or processed an event | Treating a delayed message as a new physical delay |
| Lead time | Time from a named start to a named end, in hours or days | Comparing lanes with different start and end events |
| Remaining time | Time from prediction cutoff to target milestone, in hours | Applying a total-trip distribution without subtracting elapsed time |
| Point forecast | One summary, such as a median arrival instant | An exact arrival time |
| Median | The 50th percentile of a distribution | The mean in a skewed distribution |
| Quantile | Value at which a specified fraction of the distribution lies at or below it | A guarantee for an individual load |
| Prediction interval | Range intended to contain a future outcome at stated coverage | A confidence interval for a parameter |
| Nominal coverage | Coverage level requested, for example 80% | The coverage actually measured in production |
| Empirical coverage | Fraction of eligible actual outcomes within the interval | Evidence of calibration for every subgroup |
| Calibration | Agreement between stated probabilities and observed frequencies under evaluation conditions | Precision or usefulness by itself |
| Sharpness | Concentration of a predictive distribution or interval width | Good performance if intervals miss too often |
| Deadline risk | Predicted probability that the target milestone occurs after a named deadline | The same quantity as the upper interval endpoint |
| Snapshot | Versioned feature and prediction record at one cutoff | A view assembled from the latest corrected data |
| Right censoring | Outcome occurs after the last observation time or is still unknown there | A measured duration ending at the data cutoff |
| Competing event | A different event, such as cancellation, preventing the original target as defined | An ordinary observed arrival |
| Missing label | Actual outcome is unavailable even though it may have occurred | Proof that the shipment has not arrived |
| Promise | Authorized commitment communicated to a customer, with conditions | A direct copy of the most recent model output |
| Plan | Intended internal allocation or schedule | A guarantee to the customer |
| Target | Desired service performance or date | A probability forecast |
| Commitment | A binding or operationally accepted obligation | The most likely modeled outcome |
| Intervention | Action such as expedite or rebooking that can change the outcome | A passive observation |
| As-of join | Join using only records available by a historical cutoff | A join against today's corrected master data |
| Coverage cohort | Fixed set of predictions with defined eligibility and maturity | Only shipments that arrived before extraction |
| Service-level agreement (SLA) | Agreed service obligation with defined measurement terms | A model's nominal interval coverage |

## 5. The mechanics, step by step

### Step 1 — Name the decision before choosing the model

**Input:** a business process, decision deadline, and list of possible actions.
**Output:** a written prediction and decision contract.
Ask who uses the forecast, when they act, and what happens if they are late.
For dock staffing, the target may be gate arrival before an appointment cutoff.
For inventory allocation, the target may instead be receipt into available stock after quality release.
Do not reuse the former as the latter without modeling the remaining process.
Record the physical facility, shipment identifier, time zone, event definition, and inclusion rules.
Specify the horizon: an estimate made 24 hours before the appointment addresses a different task from an estimate made 30 minutes before it.
If no feasible action changes with the prediction, the tower may need a status view rather than a predictive model.
If decisions occur at multiple moments, specify separate snapshots and evaluate each horizon separately.

### Step 2 — Define the population and the forecast unit

**Input:** shipment, leg, stop, appointment, and order relationships.
**Output:** an eligible set of prediction units and a crosswalk to business objects.
Decide whether one row represents a trip, a consignment, a stop, an order line, or an individual handling unit.
A truck with three deliveries has three possible arrival targets; one vehicle position does not identify three arrival times.
A consolidated shipment can affect several customers; avoid double-counting freight cost as if each order were an independent trip.
Define how split loads, merged loads, multi-stop routes, canceled orders, and rescheduled appointments enter the cohort.
Keep an immutable original identifier and an explicit mapping for subsequent changes.
Freeze eligibility using facts known at the cutoff rather than excluding hard cases after their outcome.
If destination or target changes, record whether the original prediction is closed, superseded, or evaluated against the original target.
A customer promise analysis may need order-line quantities as well as shipment-level arrival.

### Step 3 — Reconstruct what was knowable

**Input:** append-only event versions, receipt timestamps, reference data versions, and cutoff.
**Output:** features computed as of that cutoff, with provenance and freshness.
Use an event only if it was available to the model at or before the prediction instant.
Keep both event time and ingestion time; a scan recorded yesterday but received tomorrow was not available today.
Resolve duplicates and corrections using the version known at the time, not the final corrected history.
Store planned route and appointment versions as of the cutoff.
Persist the model version, feature values, prediction timestamp, and upstream quality flags.
Consider only known public or contractually authorized traffic, weather, operating calendar, and facility capacity data.
A high-frequency telematics update does not make the data accurate if it is old, mislocated, or assigned to the wrong vehicle.
When data arrive out of sequence, decide whether to recompute an active prediction and separately preserve its prior version.
A historical backtest should reproduce the actual data latency whenever possible.
If receipt timestamps are unavailable, state that a fully leakage-safe reconstruction cannot be established.

### Step 4 — Establish a business baseline

**Input:** completed historical journeys available before model development, segmented carefully.
**Output:** a simple ETA distribution and a comparison benchmark.
A reasonable first baseline uses historical remaining times for comparable origin-destination, mode, and journey state.
Use elapsed time consistently: historical time from departure is not the same as remaining time at the current snapshot.
When a lane is sparse, pool with a documented broader segment instead of publishing unstable local quantiles.
A carrier-provided ETA or scheduled arrival can be another benchmark if its historical versions were retained.
Use an operationally available benchmark, not the final schedule corrected after arrival.
Evaluate all candidates on the same eligible snapshots and target definition.
Document how a model behaves for a new lane, an unknown carrier, or an absent milestone.
A pooled baseline is still a forecast, not a false assurance that the new lane matches the old one.

### Step 5 — Account for incomplete journeys

**Input:** cohort entry time, observation cutoff, outcome event, and non-arrival status.
**Output:** a duration dataset with event indicator and censoring or competing-event reason.
At a fixed analysis date, a shipment still traveling contributes a lower bound on its duration.
For example, a shipment in transit for 30 hours with no arrival is known to take more than 30 hours at that cutoff; its final duration is not 30 hours.
Treating that value as an arrival biases the estimated distribution downward.
Dropping it can also bias estimates if long trips are less likely to have finished.
A survival-analysis estimator can use arrival events and right-censored durations under appropriate assumptions.
The Kaplan–Meier estimator is a nonparametric baseline for time-to-event distributions, assuming censoring is independent of event time within the analyzed population or conditioning strategy.
Independence can fail if long-delayed shipments are more likely to lose tracking.
Assess data feed loss, selection into the cohort, and administrative cutoffs separately.
If the estimated survival curve never falls below a desired probability, the corresponding quantile is not identified from that cohort; do not invent it.
A cancellation that prevents the target event is a competing event, not automatically ordinary independent censoring.
If a shipment physically arrived but its proof is missing, the issue is missing measurement; seek another source and report the missingness.
Choose an estimand: time to arrival among a defined eligible population, or the probability of arrival before a deadline in the presence of cancellation.
Different estimands need different treatment of competing events and can yield different business conclusions.

### Step 6 — Produce and interpret a distribution

**Input:** leakage-safe features, eligible historical outcomes, and an estimation method.
**Output:** a timestamp distribution or a set of quantiles plus a deadline probability.
For a two-sided 80% interval, the lower and upper bounds are commonly the 10th and 90th percentiles.
State the method because different interval constructions can have different interpretations.
A percentile is a threshold in a distribution, not the probability that the specific observed shipment will arrive at that instant.
The median minimizes expected absolute error for the predicted distribution.
The mean minimizes expected squared error, but it may be pulled later by a long tail.
If the relevant action concerns a 15:00 cutoff, calculate the probability of arrival after 15:00 from the full predictive distribution.
Do not infer that probability exactly from only the 10th, 50th, and 90th percentiles.
Express timestamps in a standard storage zone and display the local zone and daylight-saving offset.
If the predicted interval begins before the cutoff for an unfinished shipment, investigate the target and conditional distribution; remaining duration should not be negative for an event known not to have happened.
A statistical interval is not a contractual service window.

### Step 7 — Calibrate by horizon and segment

**Input:** out-of-sample predictions with mature or appropriately censored outcomes.
**Output:** a calibration report and decision on where intervals may be used.
For each nominal interval, count actual arrivals inside it and divide by eligible outcomes, subject to the cohort and censoring design.
Report sample size and uncertainty, not only a percentage.
Compare early, middle, and late prediction horizons because uncertainty should generally change as information accumulates.
Stratify by mode, lane, carrier, destination, weekday, and disruption condition when sample sizes support it.
Overall 80% coverage can hide 95% coverage for easy lanes and 50% for hard lanes.
Plot or tabulate predicted deadline risk against actual miss frequency in risk bands.
For example, among many predictions near 30% risk, approximately 30% misses would support calibration for that population and horizon.
Do not use a few observations to claim a stable subgroup rate.
Recalibration must be trained on earlier data than the period used to test it.
Coverage alone can be improved by making intervals very wide, so review interval width and decision utility too.

### Step 8 — Translate risk into a decision

**Input:** predictive deadline risk, feasible interventions, costs, and decision rights.
**Output:** an auditable action and expected-cost comparison.
A simple decision compares the expected costs of acting and not acting.
Suppose an intervention costs C and avoids L of loss if a miss occurs, with no effect otherwise.
Under that simplifying assumption, act when p times L exceeds C, where p is the probability of a miss without intervention.
The break-even probability is C divided by L when L is positive.
If intervention only reduces the chance of a miss, use C plus residual expected loss rather than assuming perfect prevention.
Include capacity, side effects, lead time, and authorization limits.
A high-risk shipment is not necessarily first in the queue if a lower-risk shipment has much larger avoidable loss or a faster-closing action window.
For constrained capacity, compare incremental expected benefit per constrained resource, then apply service and fairness constraints.
Record rejected actions and the reason: unavailable capacity is different from a low expected return.
The forecast does not itself decide whether a person is allowed to incur a charge.

### Step 9 — Govern the promise

**Input:** prediction, existing customer commitment, commercial terms, and communication policy.
**Output:** an authorized promise decision and a versioned customer message.
Record the original commitment, its source, and whether it refers to a date, time window, or named milestone.
Determine which actions can preserve it before changing it.
A customer-facing window may reflect a chosen service quantile, customer preferences, operational buffers, and contractual obligations.
No single quantile automatically satisfies all those requirements.
Use a separate policy for material forecast changes and for actual commitment changes.
A small model revision should not automatically trigger repeated customer messages.
Specify thresholds, minimum change duration where appropriate, escalation time, and override authority.
Do not wait for a smoothing rule if the risk becomes urgent or the contract requires faster notice.
Show the customer the status, the expected window, its time zone, and the next update; avoid describing a probabilistic interval as guaranteed.
Retain prior versions and explain whether the current value is a forecast or a revised commitment.

### Step 10 — Evaluate the operating system

**Input:** time-ordered prediction snapshots, outcomes, actions, costs, and a frozen evaluation plan.
**Output:** a scorecard and an action for the model and process owners.
Use rolling-origin or other chronological splits; all training labels and calibration data must be available before the test period.
Keep shipments or correlated journeys together where practical to avoid inflated precision from repeated snapshots.
Evaluate at fixed decision horizons and report cohort maturity and outcome completeness.
Mean absolute error in hours evaluates a point forecast but does not measure uncertainty or the value of a decision.
Pinball loss evaluates a specified quantile; a proper distributional score can evaluate probabilistic quality when its requirements are met.
Report interval coverage and mean interval width together.
Report deadline-risk calibration, Brier score, and decision metrics such as avoidable missed slots, expedite spend, and net cost.
Brier score is the mean squared difference between predicted event probability and the observed binary event; lower is better on the same eligible cohort.
A miss-versus-no-miss threshold creates a confusion matrix, but the threshold must reflect the decision and cost structure.
Track alert burden and lead time because a correct warning after the last feasible action has little operational value.
A pilot comparing policies needs an appropriate control or randomization and a pre-specified analysis; before-after comparisons alone can reflect seasonality or changed traffic.
Investigate actions that changed the outcome before calling a high-risk warning a false positive.

### Step 11 — Release, monitor, and recover

**Input:** validated model, operational controls, and accountable owners.
**Output:** a deployment decision and an incident response path.
Run a shadow period when operationally appropriate, storing predictions without using them to trigger decisions.
Check data freshness and coverage, invalid identifiers, missing milestones, prediction latency, and unusual interval widths.
Monitor outcome completeness and delayed labels as well as feature drift.
Drift alone does not prove model failure; investigate whether calibration and costs changed.
Set thresholds with owners and choose an explicit fallback such as a labeled historical baseline or manual review.
If no defensible prediction is possible, report unavailable rather than silently copying an obsolete timestamp.
Document who can pause alerts, restore the prior model, notify users, and authorize a customer update.
Retain model and data lineage under the organization's retention and access rules.
Schedule a review of high-cost exceptions, overrides, and affected customer groups.

### Step 12 — Choose a model appropriate to the decision

**Input:** defined target, representative historical snapshots, evaluation budget, and operating constraints.
**Output:** a justified baseline and candidate model family.
Begin with the simplest distribution that respects the target and the decision horizon.
A lookup distribution by lane and journey state can be useful when there are enough relevant observations.
It has a transparent explanation but may fail to adapt to a new road closure or facility condition.
A regression model for remaining hours can incorporate known distance, elapsed time, calendar, and latest valid milestone.
A point regression by itself does not supply a calibrated distribution.
Quantile regression can estimate selected quantiles directly, but fitted quantiles can cross unless the method or post-processing prevents it.
A survival model can represent time to arrival and account for censoring under its assumptions.
A multi-state model may be appropriate when the journey has meaningful intermediate states and transition times.
For example, port arrival, terminal release, truck pickup, and warehouse gate-in are different states.
The model must represent missing and corrected state transitions rather than assuming every event is observed perfectly.
A simulation can propagate estimated leg and handoff durations through a route.
If those durations are correlated, independently sampling every leg can understate or overstate overall tail risk.
Use the observed joint behavior or a defensible dependence scenario where the information exists.
A flexible machine-learning model can learn nonlinear interactions but remains subject to the same data and evaluation requirements.
Adding features increases operational complexity if their historical availability and production freshness are not controlled.
A model selected on overall average error may not be best for the deadline-risk decision.
Compare the candidates on a fixed, as-of cohort and on both statistical and operational measures.
A more complex model earns deployment only if its improvement is material enough to justify maintenance and risk.

### Step 13 — Set up a historical replay

**Input:** versioned event and plan records plus a schedule of prediction moments.
**Output:** a historical prediction table that resembles real production use.
First choose the eligible shipment population before looking at the outcomes.
Choose one or more fixed horizons relative to the relevant appointment, such as 24 hours and 4 hours before the slot.
An appointment can be revised, so use the appointment version available at each prediction moment.
Alternatively evaluate at named milestones such as departure; specify the milestone and the decision it supports.
Reconstruct the feature snapshot using the received versions of all events up to the cutoff.
Do not use a final route, final carrier assignment, or corrected destination if those facts were not yet known.
Store a row even if the model cannot produce a prediction, with an explicit reason.
Otherwise a model that fails on difficult shipments can appear artificially accurate on the remaining ones.
Fit the baseline on a period ending before the test period.
For a model requiring calibration, reserve an earlier calibration period separate from the final test period.
Use rolling periods when the organization needs to understand performance across seasons or changing network conditions.
Keep the training cutoff explicit for every replayed model version.
If a shipment generates several prediction rows, report both row counts and distinct shipment counts.
Repeated rows are correlated and should not be treated as independent shipment evidence.
The evaluation should account for shipments whose outcome is not yet observable at extraction.
For mature-cohort analysis, choose a maturity window before inspecting which loads were late.
Report the proportion still unresolved after that window and test sensitivity to longer windows.
A mature-only approach is not automatically unbiased if the unresolved fraction is systematically difficult.
For censoring-aware analysis, choose an estimand and method appropriate to the observation and competing-event process.
A retrospective replay measures what the system would have predicted from stored information.
It cannot by itself prove that operational actions would have been taken or would have had the assumed effect.

### Step 14 — Read a probability calibration table

**Input:** predicted deadline probabilities and comparable eligible outcomes.
**Output:** a table of risk bands with counts and observed miss rates.
Group predictions using bands selected before reviewing the outcome, or use a documented calibration plot method.
For each band, record the number of predictions, mean predicted miss probability, and observed miss fraction.
A band with mean predicted risk of 20% and 18 misses among 100 shipments has an observed rate of 18%.
The difference of two percentage points is a diagnostic, not a complete uncertainty analysis.
One band with 10 shipments and three misses has a 30% observed rate but substantial sampling uncertainty.
Consider correlated shipments, the number of lanes, and the period covered before drawing conclusions.
If the 20% band includes only low-risk lanes while the 50% band contains a different season, avoid treating a crude aggregate as a causal comparison.
Inspect calibration within business-relevant horizons and populations where counts support it.
A useful model should rank higher-risk cases above lower-risk cases, but ranking is distinct from calibration.
For example, a model can rank shipments correctly while systematically predicting twice the actual risk.
Conversely, a constant prediction equal to the overall miss rate can be calibrated in aggregate but unable to prioritize individual shipments.
The action threshold depends on calibrated probabilities and action-specific costs, not ranking alone.
For Brier score, compute the squared difference between probability and binary miss outcome for each eligible prediction.
Average those values over the defined cohort.
A miss forecast at probability 0.8 has squared error (0.8 − 1)² = 0.04.
A non-miss forecast at probability 0.8 has squared error (0.8 − 0)² = 0.64.
Compare scores against a benchmark on the same cohort; event prevalence and case mix matter.
A single score does not replace an operational review of late warnings, interventions, and affected customers.

### Step 15 — Manage the handoff between teams

**Input:** risk assessment, decision deadline, authority matrix, and communication rules.
**Output:** one owned case with traceable actions.
The system creates an exception only when its rule says a decision or communication is needed.
The exception contains the shipment and target milestone, current forecast, interval, deadline probability, and data freshness.
It also contains the business deadline for taking action and the existing commitment.
The transport planner checks whether a backup slot or alternate route is feasible.
The warehouse confirms receiving capacity and any restrictions on the product.
The inventory planner assesses the downstream orders and available substitutes.
The account manager decides whether customer communication or a commitment change is authorized and required.
A manager approves spend above the planner's limit.
The system records each response with timestamp and responsible role.
An unresolved case is escalated before the action deadline, not merely after the estimated arrival.
If the source data are unreliable, the case should identify evidence collection as the action rather than presenting a precise recommendation.
When the shipment arrives, close the case only after recording the actual milestone and any intervention.
If the label is disputed, leave the outcome status unresolved until a defined reconciliation process finishes.
Review the case later against the information available when the decision was made.
A bad outcome does not automatically imply a bad decision, and a lucky outcome does not establish a good one.
The comparison should use the expected information and feasible choices at the time, not hindsight.

### Step 16 — Define change control for a customer window

**Input:** previous communicated window, new forecast, communication preference, and notice obligation.
**Output:** a send, hold, or escalate decision with a reason.
A policy can require a material movement of the predicted window before sending a routine update.
The policy should define what constitutes movement: median, interval endpoints, or estimated arrival date.
It should define how the threshold behaves when the uncertainty interval widens but the median does not move.
The communication owner should also define a maximum interval between updates while a material exception is open.
A time-based hold can reduce routine churn when the estimate oscillates near a threshold.
It must not override urgent safety, contractual, or legally required notices.
A forecast that suddenly moves across a critical customer deadline can require immediate review.
A message should state the estimated window and the milestone to which it refers.
It should say when the estimate was last updated and when the next update is expected.
If an actual commitment is revised, identify the authorized decision, the new terms, and customer acknowledgment if required.
If a change cannot yet be authorized, communicate only what the policy allows and escalate promptly.
Do not show the previous commitment as fulfilled merely because the customer has seen a forecast.
Retain both the old and new message versions to reconstruct what the recipient knew.
Measure the number of updates per shipment, the fraction judged actionable, and complaints about conflicting messages.
Review excessive messages by cause: model volatility, new evidence, bad data, or an overly sensitive rule.
Do not solve real volatility by hiding updates that affect a customer's decision.

## 6. Worked example

### Context and decision

The following numbers are illustrative, not a reported company outcome.
A distributor has a shipment of temperature-stable packaged products traveling to a regional warehouse.
The business decision is whether to reserve a backup receiving slot before a 10:00 local-time reservation deadline.
The existing appointment is 15:00 local time on Tuesday.
A missed appointment creates expected disruption loss of USD 2,400, including incremental receiving and downstream service costs.
A backup slot costs USD 420 and, in this simplified case, eliminates that disruption loss if the original slot is missed.
The backup slot is available now, has no capacity side effects, and cannot be reserved after 10:00.
Actual contracts may have different fees, cancellation rights, and effectiveness; these assumptions make the arithmetic transparent.
The target is physical gate-in at the named warehouse, not unloaded inventory.
The prediction snapshot is Tuesday 09:00 local time.
All times below use the same local zone and refer to Tuesday unless otherwise stated.

The customer commitment is delivery by Wednesday 12:00 local time.
The internal target is to complete receiving Tuesday evening.
The operational plan is the existing 15:00 dock appointment plus any authorized backup.
The forecast is a probability distribution for gate-in, not any of those other three things.

### Distribution and uncertainty

A lane-and-state baseline uses 20 comparable historical snapshots from earlier journeys.
For this worked illustration only, assume they were all observed to arrival and are exchangeable enough with the current load to motivate an empirical forecast.
They are not enough to establish reliable conditional calibration.
The 20 predicted gate-in outcomes, ordered by time, are:

| Rank | Gate-in local time | Rank | Gate-in local time |
|---:|---|---:|---|
| 1 | 12:50 | 11 | 14:35 |
| 2 | 13:10 | 12 | 14:42 |
| 3 | 13:25 | 13 | 14:50 |
| 4 | 13:40 | 14 | 15:05 |
| 5 | 13:50 | 15 | 15:15 |
| 6 | 14:00 | 16 | 15:25 |
| 7 | 14:10 | 17 | 15:40 |
| 8 | 14:18 | 18 | 15:55 |
| 9 | 14:25 | 19 | 16:10 |
| 10 | 14:30 | 20 | 16:35 |

Use the nearest-rank empirical quantile: rank is the ceiling of probability multiplied by 20.
The 10th percentile is rank 2, or 13:10.
The 50th percentile is rank 10, or 14:30.
The 90th percentile is rank 18, or 15:55.
Thus the illustrative central 80% interval is 13:10–15:55 local time, 165 minutes wide.
Nearest-rank conventions differ from interpolated quantiles, so a production report must name its convention.
This is an empirical interval, not a distribution-free guarantee of exactly 80% future coverage.
The median 14:30 suggests arrival before the appointment, but the upper bound is 55 minutes after it.
Seven of the 20 historical outcomes are strictly later than 15:00, so the empirical miss probability is 7/20 = 35%.
An arrival exactly at 15:00 would count as on time under this example's strictly-later rule.
The definition must be agreed with the receiving operation before use.

### Cost and choice

Without a backup, expected disruption cost is 0.35 × USD 2,400 = USD 840.
With a backup, expected cost is the reservation fee of USD 420 under the assumed perfect prevention.
Expected saving from buying it is USD 840 − USD 420 = USD 420 per comparable decision.
The break-even miss probability is USD 420 / USD 2,400 = 0.175, or 17.5%.
The 35% estimate exceeds 17.5%, so the planner should reserve the backup if authorized and the assumptions hold.
The next model update is not a reason to wait past the 10:00 reservation deadline.
If the planner can only approve USD 300, the decision must go to the authorized manager before 10:00.

Now relax perfect prevention.
Suppose the backup reduces miss probability from 35% to 12%, leaving the same USD 2,400 loss if the disruption still occurs.
The expected cost with backup becomes USD 420 + 0.12 × USD 2,400 = USD 708.
The expected saving falls to USD 840 − USD 708 = USD 132.
That comparison requires a credible estimate of the risk *under the intervention*; the arrival model alone does not supply it.
If the 12% figure is unsupported, present a sensitivity case rather than reporting a proven saving.

### Promise and closeout

The dock planner books the backup under the agreed authority.
Customer service does not automatically move the Wednesday 12:00 commitment merely because the 90th-percentile gate-in is Tuesday 15:55.
Instead, it checks unloading, inventory release, downstream transport, and the customer-notification rule.
If those steps indicate material commitment risk, the authorized representative contacts the customer with a forecast window and a separately recorded commitment decision.
Suppose the actual truck reaches the gate Tuesday at 15:20.
The point forecast error is 50 minutes early: actual 15:20 minus forecast 14:30.
The actual falls within the 13:10–15:55 interval and misses the 15:00 appointment.
That one observation is compatible with a 35% miss forecast, but does not establish or disprove calibration.
Because the backup was booked, record its use, fee, actual downstream outcome, and any changed customer promise.
Do not label the alert false simply because the backup prevented the business loss.

### Censoring contrast

Assume a separate historical extraction is made at 18:00 on a Friday.
A load departed Thursday at 12:00 and has no known gate-in by the cutoff.
It contributes at least 30 hours of elapsed time and an event indicator of zero if the absence of arrival is genuinely observed.
It is not a completed 30-hour trip.
If the carrier feed stopped at Friday 08:00, its arrival status after 08:00 is unknown: the reliable observation window may be only 20 hours, not 30.
If the customer canceled and the truck was redirected, determine whether cancellation is a competing outcome under the target definition.
A survival estimator needs an appropriately defined cohort, censoring times, and a justified censoring assumption.
The 20-record calculation above deliberately avoids this issue for transparency and cannot simply include the incomplete record as a 21st completed time.

### An interval is not a deadline probability

Consider the worked example's 10th, 50th, and 90th percentile values of 13:10, 14:30, and 15:55.
The 15:00 appointment lies between the median and the upper percentile.
That fact bounds the deadline's position in a well-defined continuous cumulative distribution, but the three quantiles alone do not specify the exact mass after 15:00.
The empirical 35% estimate came from inspecting all 20 observations, not interpolating the three reported values.
Another distribution could share those quantiles and place a different proportion of its mass after 15:00.
The operations screen should therefore display the explicitly computed deadline probability when that is the decision input.
If only quantiles are stored, an assumed interpolation rule must be documented and validated rather than silently treated as fact.
The distinction matters most when the appointment falls near the action threshold and a few percentage points change the decision.

### Sensitivity to miss cost

The original case assumes the backup perfectly prevents a miss-related USD 2,400 loss and costs USD 420.
Hold the 35% miss risk fixed and consider alternative loss estimates.
At USD 800 loss, the no-backup expected loss is 0.35 × USD 800 = USD 280.
The USD 420 backup is then more expensive by USD 140 under perfect prevention.
At USD 1,200 loss, the no-backup expected loss is USD 420, so the two choices are tied before other considerations.
At USD 3,000 loss, the no-backup expected loss is USD 1,050, making the backup lower in expected cost by USD 630.
The break-even loss at 35% risk is USD 420 / 0.35 = USD 1,200.
These comparisons do not validate any of the assumed losses; they show which values would reverse the action.
If contractual obligations require a different action, the economic comparison does not waive those obligations.
If a backup is unavailable, the relevant choice may be escalation or an alternate intervention rather than inaction.

### Sensitivity to intervention effect

Hold the original 35% baseline risk and USD 2,400 loss fixed.
If backup reduces the miss probability to 25%, expected cost is USD 420 + 0.25 × USD 2,400 = USD 1,020.
That is USD 180 more than the USD 840 no-backup expected cost.
If backup reduces the probability to 10%, expected cost is USD 420 + USD 240 = USD 660.
That is USD 180 less than no backup.
The minimum required absolute risk reduction is USD 420 / USD 2,400 = 17.5 percentage points.
Starting at 35%, the post-intervention risk must therefore be below 17.5% for strictly positive expected savings.
At exactly 17.5%, expected costs are equal under these assumptions.
An intervention model should estimate the outcome under each action, not treat observational differences between acted-on and untouched shipments as automatically causal.
The most disrupted shipments are often the ones most likely to receive intervention.
That selection can make an effective action appear ineffective or a poorly targeted action appear useful.
A controlled pilot, when feasible and ethical, or a justified observational design can improve the evidence.

### Sensitivity to a shared capacity limit

Suppose only one backup slot remains for the current shipment and another load.
For the current load, the perfect-prevention expected benefit net of fee is USD 420.
For the second load, assume 20% miss risk, USD 5,000 loss, USD 500 fee, and perfect prevention.
Its expected benefit net of fee is 0.20 × USD 5,000 − USD 500 = USD 500.
Under those assumptions, allocate the single slot to the second load even though its miss probability is lower.
This example ignores differences in notice obligations, product restrictions, and the last feasible reservation time.
A real allocator must apply those constraints and consider other feasible actions for the first shipment.
If the first reservation deadline expires sooner, a sequential policy may be needed instead of a static ranking.
The point is not that risk is unimportant; risk must be combined with avoidable loss and resource use.

### Close the measurement loop

At the end of the day, the team records whether the backup slot was actually used and what it cost.
It records the physical gate-in, the time the event was received, and any later correction.
It links the shipment to affected orders and the customer communication record.
It distinguishes a missed dock appointment from a missed customer delivery commitment.
It records whether the model predicted before the booking deadline and whether the owner saw the alert in time.
A later review compares the forecast to the target milestone, not to the time the data pipeline processed the event.
The team separately reviews whether the chosen action was reasonable given information available at 09:00.
If actual gate-in was 15:20, that result alone cannot reveal the outcome that would have occurred under a different action.
Across many decisions, randomized or otherwise defensible comparisons can estimate intervention value.
Those results can update the business-cost and intervention-effect assumptions for future cases.

### A small censoring calculation

Suppose four comparable loads leave the same origin at their own cohort entry times.
Their observed times in transit are 10, 12, 14, and 16 hours.
The first arrives at 10 hours, the second arrives at 12 hours, the third is still in transit when observation ends at 14 hours, and the fourth arrives at 16 hours.
Assume the third load is independently right-censored at 14 hours and there are no tied event times.
The third load is not known to arrive at 14 hours.
At the start, all four loads are at risk of arrival.
At 10 hours, one of four arrives, leaving an estimated survival probability of 1 × (1 − 1/4) = 0.75.
Here survival means the target arrival has not yet occurred.
At 12 hours, one of the remaining three arrives, so survival becomes 0.75 × (1 − 1/3) = 0.50.
At 14 hours, the third load is censored and leaves the risk set without an arrival event.
Censoring does not itself change the estimated survival probability at that instant.
At 16 hours, the fourth load is the only load still in the observed risk set and arrives.
The estimated survival probability becomes 0.50 × (1 − 1/1) = 0.
This example demonstrates the arithmetic, not reliable tail inference from four loads.
The result is very sensitive to the independent-censoring assumption and the small risk set.
If the censored load lost tracking because of a disruption that also delayed arrival, the assumption is doubtful.
In that case, report the concern and examine alternative observation sources or sensitivity analyses.
The survival curve is a population estimate, not a claim about when the censored load actually arrived.
If the last observed record instead were a censoring at 16 hours, the curve would not necessarily fall to zero.
A requested late quantile might then be unidentifiable without additional assumptions or data.

### Absolute time and local calendars

Suppose a prediction is issued at 16:00 UTC and a warehouse appointment is stated as 12:00 local time.
The comparison requires the named location's offset on that particular date.
An offset of UTC−04:00 makes those instants equal, while UTC−05:00 makes the local appointment one hour later.
Do not apply one fixed offset to a location across dates that observe daylight saving time.
A local time repeated when clocks move backward needs an offset or another disambiguation rule.
A nonexistent local time during a clock jump must be resolved according to the source and business process.
A customer-facing date window can also depend on warehouse opening hours, holidays, and cutoff times.
Those calendar rules should be versioned and available as of the decision.
A forecast of arrival on Sunday is not equivalent to a plan for receipt on Sunday if the facility is closed.
When a missed appointment automatically moves receipt to the next operating day, model that consequence separately from physical gate-in.
The storage convention, source timestamp, display time zone, and operational calendar all need to be explicit.

### Label maturity and a fixed cohort

A Monday evaluation extracts all predictions issued in the preceding seven days.
Some loads predicted on Sunday are still moving, while most loads predicted on Monday have completed.
Scoring only completed shipments disproportionately selects short-duration recent journeys.
One alternative is to evaluate predictions from an earlier fixed calendar cohort after a predeclared maturation period.
For instance, report all predictions issued during a named week after allowing 30 days for outcomes, if that matches the business.
The 30-day window is an example, not a universally adequate maturity rule.
Report how many of that fixed cohort still have unresolved outcomes at day 30.
Separate confirmed active loads, canceled loads, feed outages, and disputed arrivals.
Examine whether unresolved cases concentrate in particular carriers, lanes, or disruption periods.
A second report can use a censoring-aware method if its observation assumptions are appropriate.
Compare results under a longer follow-up where available.
If the unresolved cases are likely to be systematically late, a complete-case score can remain optimistic even after waiting.
The decision owner should see the denominator and unresolved count alongside every headline metric.

### Confidence about a reported coverage rate

Suppose 80 of 100 mature shipment-level intervals contain their observed arrivals.
The empirical coverage is 80%, which happens to equal the nominal 80% level.
That numerical match does not establish exact future coverage.
There is uncertainty from sampling, cohort selection, label quality, and changes in operating conditions.
If many observations are repeated snapshots of the same few journeys, the effective independent evidence is smaller than 100 shipments.
A confidence interval or bootstrap must reflect the sampling design and dependence structure.
For repeated snapshots, resampling at the journey level is often more defensible than treating each snapshot as independent.
A disruption shared by many journeys can create further dependence by day, lane, or facility.
Choose a resampling or statistical design that reflects the actual decision population.
Present interval width, segment results, and the reference period next to the coverage estimate.
A broad aggregate average should not conceal a poor result for the facility that relies on the forecast.
Do not claim subgroup calibration where the sample is too small to support it.

## 7. Code, query, or calculation

The Python 3 example below uses only the standard library.
It recreates the illustrative empirical quantiles, risk, and decision, then computes coverage and point error for a small separate evaluation set.
It is a teaching baseline, not a fitted model, a survival estimator, or a production calibration test.
All inputs use minutes after local midnight on the same date; no time-zone conversion or overnight trip is represented.
All 20 historical outcomes are assumed complete and available before the current prediction.
The evaluation sample is deliberately tiny, so its results do not establish operating performance.

```python
from math import ceil

# Minutes after midnight, sorted by the quantile function.
history = [
    770, 790, 805, 820, 830, 840, 850, 858, 865, 870,
    875, 882, 890, 905, 915, 925, 940, 955, 970, 995,
]


def nearest_rank(values, probability):
    """Return the empirical nearest-rank quantile for 0 < p <= 1."""
    if not values or not 0 < probability <= 1:
        raise ValueError("nonempty values and 0 < probability <= 1 required")
    ordered = sorted(values)
    return ordered[ceil(probability * len(ordered)) - 1]


def clock(minutes):
    """Format minutes after midnight on the same day."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


lower = nearest_rank(history, 0.10)
median = nearest_rank(history, 0.50)
upper = nearest_rank(history, 0.90)
deadline = 15 * 60
miss_probability = sum(t > deadline for t in history) / len(history)
loss_if_missed = 2400.0  # USD
backup_fee = 420.0       # USD
no_backup_cost = miss_probability * loss_if_missed
backup_cost = backup_fee  # assumes perfect prevention
break_even = backup_fee / loss_if_missed

print(f"median: {clock(median)}")
print(f"80% interval: {clock(lower)}–{clock(upper)}")
print(f"late probability: {miss_probability:.1%}")
print(f"break-even probability: {break_even:.1%}")
print(f"expected cost without backup: USD {no_backup_cost:.0f}")
print(f"expected cost with backup: USD {backup_cost:.0f}")
print("decision:", "reserve backup" if backup_cost < no_backup_cost else "do not reserve")

# (lower, point, upper, actual), in minutes after midnight.
evaluation = [
    (790, 870, 955, 920),
    (800, 875, 960, 970),
    (780, 860, 945, 850),
    (810, 890, 980, 900),
]
covered = sum(lo <= actual <= hi for lo, _, hi, actual in evaluation)
mae = sum(abs(point - actual) for _, point, _, actual in evaluation) / len(evaluation)
mean_width = sum(hi - lo for lo, _, hi, _ in evaluation) / len(evaluation)
print(f"illustrative coverage: {covered / len(evaluation):.1%}")
print(f"illustrative mean interval width: {mean_width:.1f} minutes")
print(f"illustrative MAE: {mae:.1f} minutes")
```

Expected output:

```text
median: 14:30
80% interval: 13:10–15:55
late probability: 35.0%
break-even probability: 17.5%
expected cost without backup: USD 840
expected cost with backup: USD 420
decision: reserve backup
illustrative coverage: 75.0%
illustrative mean interval width: 165.0 minutes
illustrative MAE: 41.2 minutes
```

The four interval widths are 165, 160, 165, and 170 minutes.
Their sum is 660 minutes, or 165 minutes on average.
The absolute errors are 50, 95, 10, and 10 minutes; their mean is 41.25 minutes, displayed to one decimal place.
The second actual is outside its interval, giving 3/4 or 75% empirical coverage.
The expected values are arithmetic outputs of the assumptions, not observed savings.
For production, replace the empirical lookup only after defining as-of data, mature cohorts, target censoring, and out-of-sample evaluation.
The snippet intentionally omits date and daylight-saving handling; real predictions need timezone-aware timestamps and correct calendar boundaries.

## 8. Data, standards, and regulatory context

### Required operational records

An ETA product needs more than coordinates.
At minimum, retain the shipment and stop identifiers, origin and destination, transport mode, planned milestone, actual milestone, and time-zone-aware timestamps.
Store the prediction cutoff and the time each source event became available to the predictor.
Persist source, event version, correction status, and confidence or quality flags where the source supports them.
Tie each prediction to the applicable route, appointment, and customer commitment versions.
The data contract should say whether time denotes physical occurrence, scan creation, message transmission, or processing.
When a partner transmits only a date, do not manufacture minute-level precision.
A local timestamp without a zone can be ambiguous around daylight-saving transitions.
Preserve the original timestamp and offset alongside a normalized instant when feasible.

Transport identifiers may change between ocean, port, rail, and road segments.
The matching process should retain a mapping and the evidence used to associate messages.
An incorrect mapping can generate a confident prediction for the wrong physical load.
A missing association is not an instruction to join records on approximate location alone.
Compare counts, unique identifiers, and event chronology with the originating partner before relying on the feed.
A transport management system appointment can be changed while the shipment is already in transit.
A retrospective analysis must use the appointment version that was known when the prediction was issued.
An outcome record can itself be corrected; preserve both its original and revised versions.
Set a maturation period and a process for incorporating late-arriving corrections.

### Standards are not a model

GS1 Electronic Product Code Information Services (EPCIS) 2.0 is a standard for sharing visibility event data.
Its event model supports statements about what, when, where, why, and how an event occurred, subject to the fields and implementation used.
A conforming event does not by itself prove the event is correct or make every commercial ETA comparable.
EPCIS event time and record time serve different purposes; retain the actual receipt time needed for the as-of reconstruction as well.
Use the GS1 documentation and the implementation agreement to determine which fields are provided.
Do not assume a standard guarantees low latency, calibrated forecasts, or partner permission to share.

The Digital Container Shipping Association (DCSA) publishes Track & Trace interface standards for container shipping.
A shared event vocabulary and API specification can reduce integration ambiguity across participating organizations.
Implementations still need an agreement on identifiers, milestone scope, message timing, and correction handling.
A vessel arrival event is not automatically a container available-for-pickup event or final warehouse arrival.
Intermodal ETA requires modeling or estimating downstream legs and handoffs, not relabeling an ocean milestone.

Time representation should follow the agreed interface specification, including offsets and precision.
ISO 8601 is a widely used representation convention for dates and times; a format alone does not settle which physical event a timestamp denotes.
Contract terms, appointment calendars, and local holidays still determine whether a particular arrival is on time.
Define a documented hierarchy when a source supplies an impossible timestamp or conflicting milestones.
Do not silently select whichever source makes service performance look best.

### Privacy and governance

Vehicle telematics and driver-linked records can contain personal data.
In the European Union, the General Data Protection Regulation (GDPR) establishes obligations for processing personal data, including principles of data minimization and storage limitation; applicability and lawful basis depend on facts and jurisdiction.
The regulation does not prescribe a particular ETA algorithm or prediction interval.
Obtain legal and privacy review for the intended processing, sharing, retention, and access controls.
A business need for shipment visibility does not automatically justify exposing a driver's identity or precise location to every customer.
Limit partner views to the data needed for the agreed decision and retain an access audit.
Aggregate performance reports where practicable, but recognize that aggregation is not necessarily anonymization.
Define incident response for unauthorized access and contracts for processors and data recipients.

Prediction accuracy is not a substitute for transport safety or a work-hours compliance check.
Do not instruct a driver to exceed legal driving limits to satisfy a modeled arrival.
Safety and local labor requirements must be reviewed separately for the relevant operation.
Service agreements can define binding notice or delivery obligations, but the precise consequence depends on the agreement and applicable law.
This educational discussion is not legal or regulatory advice.

### Data-quality controls at ingestion

| Control | Test | Decision if it fails |
|---|---|---|
| Identity | Identifier maps to the intended shipment and stop | Quarantine or route to reconciliation |
| Availability | Source receipt precedes the prediction cutoff | Exclude from that historical snapshot |
| Sequence | Departure and arrival events are chronologically plausible | Investigate corrections and source semantics |
| Freshness | Last usable update is within policy | Widen uncertainty only if modeled; otherwise label stale or unavailable |
| Location | Reported site or coordinates match a valid route context | Investigate wrong asset or geofence |
| Completeness | Required target and timestamp are present | Do not claim a measured outcome |
| Timeliness | Event latency is measured from occurrence to receipt | Monitor the tail, not only the mean |
| Versioning | Corrections and superseded appointments are retained | Block leakage-sensitive retrospective comparison if unrecoverable |
| Units | Time zone, date, and time precision are explicit | Reject or mark ambiguous timestamps |
| Access | Recipient is entitled to the fields and purpose | Restrict distribution and escalate |

## 9. Industry practice and case evidence

Public examples show why transport milestones and estimated times are useful, but they rarely reveal an entire customer's decision economics.
Treat company-authored evidence as evidence of the stated product or workflow, not independent proof of a universal benefit.
The examples below do not establish a specific numerical improvement from the worked case.

### Container shipping: standardizing events

DCSA's Track & Trace initiative defines common interfaces and event definitions for container-shipping visibility.
The public standard addresses how participating systems exchange shipment, transport, and equipment event information.
It is evidence of industry coordination on event semantics, not evidence that all carriers have identical coverage or predictive accuracy.
A shipper can use normalized events to align a carrier's container status with a purchase order and a downstream truck appointment.
That last operational application is an inference from the standard's purpose, not a claimed DCSA case outcome.
The decision remains local: determine which event corresponds to customs release, terminal availability, pickup, and final receipt.
For a warehouse ETA, the ocean event stream is only one part of the journey.
Missing container release information should not be disguised as certainty about when the truck will depart the terminal.

### Parcel delivery: customer-facing estimate

UPS publicly describes its My Choice service, including delivery alerts and estimated delivery windows where available.
This shows an operator presenting delivery information to recipients and enabling some delivery-management choices.
The public service description does not disclose the model's full training set, the calibration of every window, or the economics of each notification.
It also does not establish that every shipment receives the same type of estimate or options in every market.
A business lesson is to separate a recipient-facing window from internal control signals and to state available choices.
[INFERENCE] Notification fatigue is a reason to set a material-change policy rather than send every internal model revision.
That inference is a design consideration, not a measured outcome attributed to UPS.

### Ocean schedule reliability: baseline and external conditions

The U.S. Federal Maritime Commission's Bureau of Trade Analysis publishes information on ocean shipping and supply-chain conditions.
Public aggregate information can help an analyst understand the broader context in which a carrier's historical lane data were generated.
An aggregate report does not identify the arrival distribution of a specific container at a specific warehouse.
A period of port congestion or service changes can make old historical quantiles unsuitable without further analysis.
Compare external indicators with internal event and carrier data; document differences in scope and reporting lag.
Use external information as a feature or scenario only if it was available at the prediction cutoff.

### Operating pattern for a multi-party tower

A shipper may receive a scheduled appointment from a warehouse, a carrier ETA, and GPS observations from a telematics provider.
Those sources can disagree because they describe different milestones or use different update times.
The tower should display source and freshness alongside any synthesized forecast.
It should retain the carrier's original estimate when producing its own estimate so performance can be compared fairly.
An exception owner may call the carrier when predicted risk crosses a decision threshold and an action is still feasible.
The owner should record the intervention and any customer communication in a shared case record.
The existence of the pattern is not proof that a particular provider implements it or achieves a stated saving.

### Evidence grading for a deployment

Peer-reviewed survival-analysis and probabilistic-evaluation literature supports the statistical mechanics.
GS1 and DCSA documents support claims about the relevant standards and published event semantics.
Carrier service pages support narrow claims about described customer functionality, subject to their date and scope.
Internal operational logs are needed to establish coverage, decision benefit, costs, and behavior for a particular deployment.
A reference architecture, vendor demonstration, or aggregate traffic report cannot substitute for those local measurements.

## 10. Failure modes and red flags

The table is an operational diagnostic, not an exhaustive list.
A red flag should lead to an owner and an evidence-gathering action, not an automatic assumption of model failure.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong milestone | Gate-in errors look good but stock remains unavailable | Target ends before unloading and release | A customer is promised unusable inventory | Name the milestone and model downstream stages |
| Event leakage | Backtest errors are implausibly low | Future-received scans appear in past snapshots | Production performance collapses | Reconstruct features as of receipt time |
| Dropped active loads | Recent cohort is mostly short journeys | Long durations are not mature at extraction | ETA and risk are optimistic | Track censoring and cohort maturity |
| False completion | Many durations end exactly at extraction time | Cutoff is treated as arrival | Distribution is artificially compressed | Store event indicator and censor time |
| Missing label mistaken for in-transit | Carrier feed has stopped | No scan is interpreted as no arrival | Duration and late risk are distorted | Separate feed availability from physical status |
| Cancellation treated as arrival | Redirected loads appear as early deliveries | Competing outcomes are recoded as completion | Misleading service and model scores | Define event types and estimand |
| Too-narrow interval | Coverage below target on difficult lanes | Tail risk or disruption is underrepresented | Planners underreserve capacity | Segment, recalibrate, and widen only with evidence |
| Too-wide interval | Coverage is high but all shipments trigger alerts | Width is optimized without usefulness | Expensive noise and inaction | Report width and decision costs with coverage |
| Misread deadline risk | Users treat the 90th percentile as a 90% guarantee | Quantiles and SLA commitments are conflated | Broken promises and mistrust | Display definitions and policy separately |
| Stale evidence | Last update grows old while ETA keeps changing | Predictor assumes location is current | False precision | Surface freshness and define fallback |
| Bad identifier mapping | One vehicle appears on incompatible routes | Asset or shipment crosswalk is wrong | Forecast belongs to another load | Reconcile mappings and quarantine conflicts |
| Time-zone error | Errors cluster around midnight or clock changes | Local time is stored without offset | Wrong appointment and customer message | Store timezone-aware instants and original values |
| Intervention blindness | High-risk alerts have few observed failures | Team expedited them | Valuable alerts appear inaccurate | Log actions and distinguish prediction from causal impact |
| Alert flooding | Operators dismiss or delay most warnings | Threshold ignores cost and capacity | Actionable exceptions are missed | Prioritize avoidable loss and action deadline |
| Promise churn | Customers receive multiple contradictory windows | Every model update becomes a message | Confusion and loss of trust | Use material-change and urgent-escalation rules |
| Shift in mix | Overall accuracy remains stable while one lane worsens | Easy volume masks a hard subgroup | Unequal service and hidden cost | Monitor segments and sample sizes |
| Training-serving mismatch | Live features differ from historical values | Different transformations or latency | Calibration degrades after release | Version feature definitions and compare distributions |
| Circular evaluation | Only predictions that triggered action are scored | Selection depends on the model | Biased estimated quality | Preserve an eligible all-prediction cohort |
| Unsupported fallback | An old carrier timestamp is shown as current | Missing predictions are silently replaced | Apparent confidence is false | Label fallback, age, and source or show unavailable |
| Perverse incentives | Staff suppress exceptions near reporting cutoff | Performance targets reward appearance | Bad data and hidden service failures | Audit overrides and align incentives with outcomes |

A few additional checks make the table actionable.
For any observed coverage drop, first inspect label maturity and timestamp corrections.
Then compare prediction horizons and subgroup composition with the previous evaluation period.
Only after those checks should the owner attribute the change to model behavior.
For any surge in alerts, compare both predicted risk and the number of economically feasible interventions.
For any promise complaint, compare the original commitment, forecast versions, and actual communication record.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Target is consistently defined | Outcome and prediction must refer to the same event | Carrier reports gate-in while customer expects unloading | Audit source definitions and sample event traces |
| Historical data were available as of cutoff | Prevents hindsight leakage | Late scan or later schedule revision enters features | Compare receipt and cutoff timestamps; replay versions |
| Comparables are informative for current load | Empirical frequencies are used as probabilities | New route, weather regime, or facility process | Compare covariates and recent outcomes; use conservative fallback |
| Censoring is appropriately handled | Completed-only data can understate long durations | Delayed or disrupted loads lose visibility more often | Analyze observation process and use sensitivity analysis |
| Outcome status is observable | Calibration needs a trustworthy label | Proof of delivery is missing or disputed | Reconcile sources and report unresolved label rate |
| Cost of a miss is estimated | Expected-cost threshold needs an economic consequence | Lost sales, penalties, or recovery costs vary widely | Use ranges, update from case records, and test sensitivity |
| Intervention effectiveness is known | Action changes future outcomes | Backup slot fails to prevent downstream delay | Estimate action-specific risk with suitable evidence |
| Action remains feasible | A useful warning must precede its deadline | Slot is full or approval arrives too late | Store decision deadline and capacity state |
| Costs are comparable | Choice minimizes a common objective | Service obligations or safety constraints override cost | Apply hard constraints before expected-cost comparison |
| Sufficient independent evidence exists | Coverage estimates need adequate effective sample size | Many correlated snapshots from the same trip | Group by journey and report counts and uncertainty |
| Production matches evaluation | Backtest estimates future behavior only under relevant conditions | Route, partner, or label process changes | Monitor by cohort and run prospective evaluation |
| Promise policy has authority | Forecast does not change customer terms itself | No one owns communication or approval | Assign named role, delegation, and escalation path |
| Time representation is unambiguous | Cutoffs and intervals need comparable instants | Missing offsets or daylight-saving ambiguity | Preserve zone and source value; resolve exceptions |
| Uncertainty is decision-relevant | Intervals must relate to actionable thresholds | Only a median is available or deadlines are unspecified | Request a distribution and an explicit decision cutoff |

Even a well-calibrated model cannot predict every individual event correctly.
Coverage describes a frequency over a population and period, not a warranty for each shipment.
A high-quality forecast can coexist with poor decisions if the business costs are wrong or the action is unavailable.
An apparently profitable intervention can be harmful if its estimated effectiveness comes from a biased comparison.
A forecast developed for one milestone, horizon, and customer population should not silently be reused for another.

## 12. Exercises

Work independently before reading the answer section.
Use the definitions and time conventions stated in each exercise.
Unless specified otherwise, currency is USD and costs are per shipment.

### Exercise 1 — Target and information cutoff

A carrier predicts a truck's gate-in at 14:00 on Wednesday.
The warehouse normally unloads within two hours but sometimes needs a quality hold.
A customer has a commitment for available inventory by 17:00.
At 09:00 the tower sees a scan timestamped 08:30 that arrived at 08:35 and a scan timestamped 08:45 that arrived at 09:10.
**Deliverable:** Define the customer-relevant target and state which scan can be used at 09:00.
**Acceptance criteria:** Distinguish gate-in from released inventory, identify the available scan, and name an additional downstream uncertainty.
**Time:** 10 minutes.

### Exercise 2 — Quantiles and a deadline

Eight comparable completed arrival times are 10:00, 10:20, 10:40, 11:00, 11:20, 11:40, 12:00, and 12:20 local time.
Use nearest-rank quantiles and define late as strictly later than 11:00.
**Deliverable:** Calculate the median, the 25th–75th percentile interval, and empirical late probability.
**Acceptance criteria:** Show ranks, endpoints, denominator, and the strict inequality.
**Time:** 12 minutes.

### Exercise 3 — An imperfect intervention

A missed slot costs USD 1,500.
Without action its probability is 40%; a backup costs USD 200 and is estimated to reduce the probability to 20%.
Assume the cost of a miss is the same in both states and there are no other effects.
**Deliverable:** Calculate the expected cost of each action and choose one.
**Acceptance criteria:** Include residual loss after the backup and report the expected saving.
**Time:** 10 minutes.

### Exercise 4 — Censoring and competing events

A load departs Monday at 06:00.
At Tuesday 12:00, the analysis cutoff, it has not arrived and a working carrier feed confirms it remains in transit.
A second load was canceled and redirected before its target arrival.
A third may have arrived but its tracking feed failed Monday at 18:00.
**Deliverable:** Classify each record and give the observed duration or observation boundary.
**Acceptance criteria:** Do not turn the cutoff into an arrival; distinguish competing event from feed-related missingness; state a censoring assumption.
**Time:** 15 minutes.

### Exercise 5 — Promise communication

The current customer commitment is Thursday by 17:00.
At Wednesday 12:00 a distribution for the relevant customer milestone has a median of Thursday 15:00 and a 90th percentile of Thursday 19:00.
The account manager may revise commitments, while the transport planner may reserve backup capacity up to USD 500.
**Deliverable:** Draft an internal case note and a short customer update appropriate to the information given.
**Acceptance criteria:** Separate forecast from commitment, identify owners and the missing action/cost information, and do not promise a change without authorization.
**Time:** 15 minutes.

### Exercise 6 — Coverage and operational scorecard

Ten mature, eligible predictions have nominal 80% intervals; seven actual arrivals fall inside.
Their total interval width is 1,600 minutes.
For the same cohort, three interventions were taken; one shipment missed its slot despite intervention.
**Deliverable:** Calculate empirical coverage and mean width, then propose three additional measures.
**Acceptance criteria:** Report numerator and denominator, calculate minutes per interval, and explain why intervention outcomes alone do not prove avoided misses.
**Time:** 12 minutes.

### Exercise 7 — As-of evaluation design

A carrier sends position every 15 minutes but occasionally uploads a day of buffered events after reconnecting.
An analyst has only a table containing event timestamps and the latest corrected appointment, not receipt times or appointment history.
**Deliverable:** Design a retrospective evaluation and identify what cannot be established from the current table.
**Acceptance criteria:** Specify cohort, fixed horizon, temporal split, snapshot and label policy, censoring treatment, and missing provenance remedy.
**Time:** 25 minutes.

### Exercise 8 — Capacity-constrained queue

Five shipments have predicted deadline risk, missed-slot cost, and feasible action deadlines.
Only two backup slots remain, and some actions have different effectiveness and fees.
A major customer has a contractual notice clause.
**Deliverable:** Design a prioritization and escalation rule and identify the necessary input fields.
**Acceptance criteria:** Use incremental avoidable expected cost rather than probability alone, account for capacity and timing, and separate notice from optimization.
**Time:** 25 minutes.

## 13. Self-check answers

### Answer 1

The decision target is the time inventory is available after unloading and any necessary quality release, at the named site.
Gate-in is an upstream milestone and may be useful input but does not establish the target.
The 08:30 event arrived at 08:35 and is eligible for the 09:00 snapshot, subject to validity checks.
The 08:45 event arrived at 09:10 and must not be used in a 09:00 prediction even though its event time is earlier.
The unloading duration and possibility of a quality hold are separate sources of remaining uncertainty.
The Wednesday 17:00 customer commitment is not itself a forecast.

### Answer 2

The median rank is ceiling(0.50 × 8) = 4, giving 11:00.
The 25th percentile rank is ceiling(0.25 × 8) = 2, giving 10:20.
The 75th percentile rank is ceiling(0.75 × 8) = 6, giving 11:40.
The stated empirical interval is therefore 10:20–11:40 local time.
Four times are strictly later than 11:00, giving 4/8 = 50%.
The observation exactly at 11:00 is on time under the exercise rule.
This small empirical sample does not establish future 50% interval coverage.

### Answer 3

Without action, expected cost is 0.40 × USD 1,500 = USD 600.
With backup, expected cost is USD 200 + 0.20 × USD 1,500 = USD 500.
The expected saving is USD 600 − USD 500 = USD 100 per comparable shipment.
Choose the backup under the stated assumptions.
The result depends on the estimated 20% risk under intervention, not just the original ETA distribution.

### Answer 4

The first load is right-censored at 30 hours after departure, with no observed arrival by the cutoff.
The second load has a cancellation or redirection that may prevent the defined arrival and should be treated as a competing event if that is the business definition.
For the third load the last reliable observation boundary is Monday 18:00, or 12 hours after departure, assuming the feed was valid up to then.
Its actual arrival after that time is unknown; feed failure does not show that the truck remained in transit.
A simple survival estimate needs an assumption about whether censoring is independent of the unobserved event time, possibly after conditioning on relevant variables.
Feed failures associated with disruption can violate that assumption.

### Answer 5 — Rubric

A strong internal note records the Thursday 17:00 commitment separately from the Thursday 15:00 median and Thursday 19:00 90th percentile.
It identifies the account manager as the commitment owner and the transport planner's USD 500 backup authority.
It requests the probability of missing 17:00, intervention effectiveness, feasible actions, and notice obligations.
The two supplied quantiles alone do not identify the exact miss probability at 17:00.
A strong customer update says there is a risk to the existing commitment, gives the forecast as an estimate when appropriate, and names the next update time.
It does not state a new commitment unless the authorized manager has approved and communicated one.
A weak answer silently substitutes the 90th percentile as a guaranteed delivery time.

### Answer 6

Empirical coverage is 7/10 = 70% for this tiny mature cohort.
Mean interval width is 1,600/10 = 160 minutes.
Additional measures could include point mean absolute error, deadline-probability calibration, alert lead time, and cost per eligible shipment.
A rate based on ten predictions is highly uncertain and does not alone establish persistent undercoverage.
The intervention count and one observed miss do not reveal how many misses would have occurred without intervention.
A comparison policy, appropriate control, or justified causal analysis is needed for an avoided-loss claim.

### Answer 7 — Rubric

A strong design freezes a representative cohort and produces predictions at a fixed decision horizon using only historically available inputs.
It trains and calibrates on periods earlier than the evaluation period and keeps correlated snapshots appropriately grouped.
It distinguishes completed arrivals, right-censored trips, competing cancellations, and missing tracking.
It preserves label maturation and correction rules, and reports the unresolved share.
The supplied table cannot prove a leakage-safe as-of reconstruction because event receipt times and historical appointment versions are missing.
The team should recover archived messages and version histories, or disclose the limitation and avoid claiming a faithful retrospective simulation.
A prospective shadow collection can establish the required provenance for a future evaluation.
A weak design uses all event times before 09:00 regardless of their receipt time.

### Answer 8 — Rubric

A strong rule calculates expected loss without action minus expected loss with each feasible intervention, including fees and residual risk.
It applies capacity, availability, approval, safety, and timing constraints before ranking.
With heterogeneous use of resources, it considers the constrained allocation rather than simply taking the two highest probabilities.
It stores risk, loss estimate, intervention effect, resource use, decision deadline, owner, and override reason.
The contractual notice obligation is evaluated independently and escalated even if an intervention is not economically selected.
A weak rule uses probability alone or waits for a more certain forecast after the slot booking deadline.

## 14. Interview questions

### 1. What exactly is an ETA?

A strong answer defines the named milestone, location, population, prediction cutoff, and time zone.
It separates the forecast from an appointment and a customer commitment.
A weak answer treats any carrier timestamp as the final delivery promise.

### 2. What does an 80% prediction interval mean?

A strong answer discusses intended coverage for comparable future observations and checks empirical coverage and width.
It recognizes that individual outcomes are not guaranteed to lie inside the interval.
A weak answer calls the upper endpoint a contractual latest arrival.

### 3. How do you avoid hindsight in a backtest?

A strong answer uses event receipt and reference-data version times to recreate the as-of snapshot.
It uses time-ordered training, calibration, and evaluation with mature outcomes.
A weak answer uses event time alone and joins today's corrected schedule.

### 4. How do you use an active shipment in a duration analysis?

A strong answer identifies a right-censored lower bound and states the observation and independence assumptions.
It distinguishes lost tracking and competing cancellation from a verified active trip.
A weak answer calls the time since departure a completed trip duration.

### 5. Why might a median ETA be insufficient for a dock decision?

A strong answer uses deadline risk, loss, intervention effect, and the last feasible action time.
It notes that the upper tail can make a backup worthwhile even when the median is on time.
A weak answer chooses the fastest-looking shipment without considering cost or capacity.

### 6. How would you set an alert threshold?

A strong answer compares incremental expected costs under available actions and checks operational constraints.
It states the intervention and loss assumptions and separates contractual notice rules.
A weak answer adopts a universal probability threshold without a decision context.

### 7. How do you evaluate uncertainty?

A strong answer reports coverage and width by horizon and material segment, with cohort size and maturity.
It also evaluates probability calibration and proper scores where applicable.
A weak answer reports only average absolute point error.

### 8. Why can a successful alert look wrong in the outcome data?

A strong answer explains that intervention can prevent the event being warned about and must be logged.
It distinguishes predictive evaluation from the causal effect of the alert policy.
A weak answer calls every prevented late arrival a false positive.

### 9. How would you handle a stale GPS feed?

A strong answer exposes freshness, investigates feed and identity quality, and applies a documented fallback or unavailable state.
It does not claim a precise location or fabricate an interval.
A weak answer continues to display the last point as live without its age.

### 10. When should a customer promise change?

A strong answer distinguishes the forecast from the commitment and checks authorized policy, contractual notice, and feasible alternatives.
It preserves versions and avoids unnecessary notification churn without suppressing urgent warnings.
A weak answer automatically rewrites the promise after every small model update.

## 15. Key terms

- [ETA](#4-definitions-and-vocabulary) — forecast time of a named arrival milestone.
- [Target milestone](#4-definitions-and-vocabulary) — event whose time is predicted.
- [Prediction cutoff](#4-definitions-and-vocabulary) — instant defining available information.
- [Event time](#4-definitions-and-vocabulary) — time the source says an event occurred.
- [Processing time](#4-definitions-and-vocabulary) — time a system received or processed an event.
- [As-of join](#4-definitions-and-vocabulary) — reconstruction restricted to historically available records.
- [Snapshot](#4-definitions-and-vocabulary) — versioned state and prediction at a cutoff.
- [Lead time](#4-definitions-and-vocabulary) — elapsed time between named milestones.
- [Remaining time](#4-definitions-and-vocabulary) — time from cutoff to target event.
- [Point forecast](#4-definitions-and-vocabulary) — single summary of a predictive distribution.
- [Median](#4-definitions-and-vocabulary) — 50th percentile of a distribution.
- [Quantile](#4-definitions-and-vocabulary) — threshold for a stated cumulative probability.
- [Nearest rank](#distribution-and-uncertainty) — empirical quantile convention using a ceiling rank.
- [Prediction interval](#4-definitions-and-vocabulary) — range with a stated intended future coverage.
- [Nominal coverage](#4-definitions-and-vocabulary) — requested coverage level.
- [Empirical coverage](#4-definitions-and-vocabulary) — observed fraction within the interval.
- [Calibration](#4-definitions-and-vocabulary) — agreement of predicted probabilities with observed frequencies.
- [Sharpness](#4-definitions-and-vocabulary) — concentration of a predictive distribution.
- [Deadline risk](#4-definitions-and-vocabulary) — probability that the target occurs after the cutoff.
- [Right censoring](#4-definitions-and-vocabulary) — an outcome is not yet known beyond an observation boundary.
- [Competing event](#4-definitions-and-vocabulary) — alternative outcome preventing the target as defined.
- [Missing label](#4-definitions-and-vocabulary) — unavailable outcome measurement.
- [Cohort maturity](#step-10--evaluate-the-operating-system) — extent to which eligible outcomes are observable.
- [Kaplan–Meier estimator](#step-5--account-for-incomplete-journeys) — survival baseline for censored time-to-event data under assumptions.
- [Promise](#4-definitions-and-vocabulary) — authorized customer commitment.
- [Plan](#4-definitions-and-vocabulary) — intended internal schedule or allocation.
- [Target](#4-definitions-and-vocabulary) — desired business performance.
- [Intervention](#4-definitions-and-vocabulary) — action that can change the outcome.
- [Break-even probability](#cost-and-choice) — risk at which expected costs of alternatives are equal under stated assumptions.
- [Brier score](#step-10--evaluate-the-operating-system) — mean squared error of binary-event probabilities.
- [EPCIS](#standards-are-not-a-model) — GS1 event-sharing standard.
- [SLA](#4-definitions-and-vocabulary) — agreement specifying service obligations.

## 16. Sources

The following sources are public and checkable; their scope is stated rather than extrapolated to the illustrative case.
Access and exact interface implementations can change; consult the current version before deployment.

1. GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis), including EPCIS 2.0 documentation.
   Primary source for visibility-event standards; it is not proof of data quality for an individual partner.
2. Digital Container Shipping Association, [Track & Trace standards](https://dcsa.org/standards/track-and-trace).
   Primary source for published container-shipping event interface standards and implementation resources.
3. Kaplan, E. L., and Meier, P. (1958), “Nonparametric Estimation from Incomplete Observations,” *Journal of the American Statistical Association*, 53(282), 457–481, [DOI: 10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452).
   Foundational source for the Kaplan–Meier estimator; its censoring assumptions matter in application.
4. Gneiting, T., and Raftery, A. E. (2007), “Strictly Proper Scoring Rules, Prediction, and Estimation,” *Journal of the American Statistical Association*, 102(477), 359–378, [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
   Supports use of proper probabilistic scores; does not choose the operational cost function.
5. Hyndman, R. J., and Fan, Y. (1996), “Sample Quantiles in Statistical Packages,” *The American Statistician*, 50(4), 361–365, [DOI: 10.1080/00031305.1996.10473566](https://doi.org/10.1080/00031305.1996.10473566).
   Explains why empirical quantile conventions differ.
6. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Article 5.
   Primary legal text; applicability and lawful processing require fact-specific professional review.
7. ISO, [ISO 8601 date and time format](https://www.iso.org/iso-8601-date-and-time-format.html).
   Overview of the representation standard; the relevant interface specification determines implementation details.
8. UPS, [UPS My Choice](https://www.ups.com/us/en/track/ups-my-choice).
   Company description of recipient-facing delivery-management features; availability varies.
9. U.S. Federal Maritime Commission, [Bureau of Trade Analysis](https://www.fmc.gov/about-the-fmc/bureaus/bureau-of-trade-analysis/).
   Official context on ocean transportation analysis; aggregate information does not establish individual shipment ETA accuracy.
10. NIST/SEMATECH, [e-Handbook of Statistical Methods: censoring](https://www.itl.nist.gov/div898/handbook/apr/section1/apr131.htm).
    Introductory explanation of censored observations and their implications for reliability data.

## 17. Links

- [Module 19 index](_index.md) — topic sequence and status.
- [The Five Visibility Layers](01_visibility-layers.md) — connect shipment, order, inventory, supplier, and network views.
- [Event Streams, IoT, Telematics, and EPCIS 2.0](02_event-streams-epcis.md) — event evidence and data quality.
- [Event Time versus Processing Time](03_event-time-processing-time.md) — as-of availability and late events.
- [Delay Propagation](05_delay-propagation.md) — downstream consequences of a missed milestone.
- [Exception Management and Queue Design](07_exception-queue.md) — operational ownership and queue state.
- [Module 19 survey and lab sections](../../../course/19_control_towers_visibility_events.md) — module overview and applied exercises.

Sibling links may point to topics still listed as planned in the module index; consult the index for availability.
