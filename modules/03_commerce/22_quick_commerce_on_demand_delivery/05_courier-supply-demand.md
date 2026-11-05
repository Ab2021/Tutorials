# Courier Supply and Demand

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics  
> Topic 5 of 10  
> Estimated study time: 4 hours  
> Prerequisites: [Promise tiers and dark stores](01_promise-tiers-dark-stores.md), [Short-horizon forecasting](02_short-horizon-forecasting.md)  
> Version 2026.5

## 1. Why this topic exists

A delivery promise is a claim on a worker's future time, not merely an estimate on a screen.
A store may have the order picked and packed while no suitable courier is available to collect it.
The resulting delay can spoil food, disappoint a customer, and waste a worker's unpaid travel to the pickup.
A courier logged in on the other side of a river does not automatically solve that delay.
The river crossing, vehicle restrictions, other assignments, and the courier's choice to accept all matter.

A marketplace operations lead must decide how much supply to attract into each zone and time bucket.
The lead must make that decision before demand is known, while respecting pay commitments and worker constraints.
A dispatch team then decides which available courier to offer each order.
A store manager controls picking readiness, which changes how long the courier must wait.
Finance evaluates contribution after courier pay, incentive expense, refunds, and other variable costs.
Trust and safety teams set operational safety constraints, and legal specialists assess the governing rules.
No single team can repair a broken promise by optimizing its own metric alone.

The wrong response to a shortage is often to count every logged-in courier as usable capacity.
Some are already occupied, on break, outside the pickup catchment, or unable to carry the order.
Some will decline an offer with a long pickup, unclear pay, or a destination far from future work.
A target that assumes acceptance without compensating for these frictions overstates service capacity.

The wrong response to a surplus is often to reward high utilization without regard to time or risk.
A courier at 100% engaged utilization has no margin for traffic, breaks, or a delayed store.
Long periods of idle time can also undermine earnings, even when individual jobs appear profitable.
An apparently cheap marketplace can therefore lose supply over the next shift.

This topic connects demand forecasts, supply forecasts, incentives, worker decisions, and service outcomes.
It treats workers as people who can choose, not interchangeable capacity units.
It separates evidence about a policy from stories told after the fact.
The goal is not to maximize orders at any cost; it is to choose a feasible, economically defensible and safe operating plan.

## 2. Learning outcomes

After working through this unit you should be able to:

1. **Define** online, available, engaged and paid time, and calculate utilization with an explicit denominator.
   Exercise 1 checks the definitions and arithmetic.
2. **Convert** a slot-level order forecast into required courier-hours while accounting for acceptance and lost capacity.
   Exercises 2 and 3 test this conversion.
3. **Diagnose** a marketplace imbalance using spatial, temporal and worker-eligibility cuts rather than one city average.
   Exercise 4 asks for a diagnosis.
4. **Compare** an incentive with a promise or capacity adjustment using incremental contribution and risk.
   Exercise 5 tests this decision.
5. **Design** a credible incentive experiment with an appropriate counterfactual and spillover controls.
   Exercise 6 assesses the design.
6. **Specify** privacy, pay, safety and worker-choice guardrails for a proposed intervention.
   Exercise 7 requires a written guardrail plan.
7. **Communicate** a forecast, an operating plan, a target and a commitment as different objects.
   Exercise 8 requires a decision memo.

A correct calculation is necessary but not sufficient.
A strong answer also names the time bucket, geographic boundary, population, available evidence and decision owner.

## 3. The core mental model

The marketplace clears only when an eligible worker accepts an offer and can complete it within the relevant constraints.
A forecast is an estimate of possible future load, not proof that the market will clear.

```text
Customer demand forecast -----> expected orders by zone and slot
          |                                  |
          v                                  v
Promise / admission policy ---> required courier task-hours
                                             |
Worker choice <--- offer economics <--- incentive and dispatch policy
     |                                       |
     v                                       v
Online -> available -> eligible -> accepts -> engaged -> completes
     ^                    |                     |
     |                    v                     v
Outside options     distance / vehicle      travel / wait / handoff
     |                    |                     |
     +---- future participation <--- earnings, experience, safety
                                             |
                                             v
                           customer service + costs + worker outcomes
```

The chain is a set of conditional states, not a funnel that can be multiplied blindly.
Availability varies during the bucket and the same courier may complete multiple jobs.
A courier who accepts a job becomes engaged and cannot simultaneously serve an incompatible order.
Once that job ends, the worker may become available in another zone.
A forecast based on distinct courier counts ignores this recycling of capacity and can double-count workers crossing zones.

Separate the **stock** of workers at an instant from the **flow** of completed jobs over an interval.
Workers online at 18:00 are a stock measured in couriers.
Online time from 18:00 to 19:00 is a flow measured in courier-hours.
Orders received between 18:00 and 19:00 are a flow measured in orders per hour.
The staffing calculation compares task-hours required with courier-hours supplied over the same boundary.

Demand has a distribution, not one guaranteed value.
Supply also varies because a worker can log off, reject an offer, face traffic, or finish a prior job late.
Rain can raise both order arrivals and travel time while reducing safe cycling speed.
A plan based on independent, average inputs can fail precisely when the joint tail matters most.

Utilization is a useful diagnostic but not an objective by itself.
A low value can mean oversupply, a store bottleneck, or an overbroad boundary.
A high value can mean healthy throughput, or a queue that is growing too quickly to clear.
At very high utilization, variation produces disproportionately longer waits in common queueing models.
That qualitative result does not determine the correct buffer for every dispatch network.
Estimate the buffer using the network's own history and stress scenarios.

The marketplace has feedback loops.
A poor offer can be rejected repeatedly, increasing customer wait and further reducing the attractiveness of the remaining trip.
A temporary bonus may attract couriers, but it may also pull them away from an adjacent zone.
A higher customer fee can lower demand, alter customer composition, and reduce future retention.
The feedback is part of the intervention, not noise to ignore.

A practical control loop runs at two speeds.
Before a shift, operations chooses staffing, coverage and published incentives from forecasts.
During a shift, operations observes actual orders, ready times, eligible capacity, and queues, then applies preapproved controls.
After the shift, analysts reconcile pay and outcomes and measure whether the intervention caused improvement.

## 4. Definitions and vocabulary

The definitions below are analytical conventions for this unit, not legal definitions of compensable work.
Local law, contracts and pay guarantees may require different time boundaries.

| Term | Definition and unit | Common misuse |
|---|---|---|
| Zone | Geographic planning or reporting area, with versioned boundaries. | Treating an administrative polygon as a frictionless travel area. |
| Slot | Fixed interval, for example 18:00–19:00 local time. | Mixing orders placed in a slot with all deliveries completed in that slot. |
| Demand | Customer requests in a stated zone and slot, in orders. | Counting only completed orders and calling that unconstrained demand. |
| Forecast | Distribution or estimate of future demand or supply. | Treating the mean as a promise. |
| Plan | Chosen action based on forecasts and constraints. | Calling a desired supply level a forecast. |
| Target | Desired outcome used to manage performance. | Treating a target as a contractual commitment. |
| Commitment | An obligation or promise made to a customer or worker. | Assuming a modeled target has been communicated or accepted. |
| Online time | Duration a courier is connected and considered online, in courier-hours. | Assuming all online time is usable for every offer. |
| Available time | Online duration not engaged and eligible under the defined state model. | Including breaks, offline gaps or unavailable vehicle classes. |
| Engaged time | Time from the chosen engagement start to completion or cancellation, in courier-hours. | Comparing a pickup-to-drop metric with an offer-to-drop metric. |
| Paid time | Time covered by the applicable pay rule, in hours. | Equating paid time with engaged time in every jurisdiction. |
| Courier-hour | One courier providing one hour in the stated state. | Confusing headcount with available hours. |
| Utilization | Engaged courier-hours divided by online courier-hours, for the same population and interval. | Comparing ratios with different denominators or state definitions. |
| Offer | A proposed job or bundle sent to a courier. | Counting an offer as an accepted job. |
| Acceptance rate | Accepted eligible offers divided by eligible offers under a stated attribution rule. | Treating repeated offers for one order as independent orders. |
| Assignment | A job allocated to a courier after applicable acceptance or assignment rules. | Assuming assignment guarantees pickup or completion. |
| Completion rate | Completed orders divided by the specified cohort of orders or assignments. | Switching denominators when cancellations rise. |
| Task time | Courier time consumed by an order or bundle, including defined pickup, wait, travel and handoff components. | Using customer elapsed time as courier time without adjustment. |
| Pickup deadhead | Travel to pickup without the customer's order, in minutes or kilometers. | Omitting it from capacity and cost. |
| Idle time | Online, unengaged time under a stated state definition. | Calling all idle time avoidable or unpaid. |
| Marketplace clearing | Matching feasible demand to willing and eligible capacity at acceptable service and economics. | Equating clearing with a single price or 100% completion. |
| Incentive | Incremental payment or other offer condition intended to influence participation or acceptance. | Assuming the entire payout creates incremental supply. |
| Surge | A time- or place-dependent increase in pay or price under a stated rule. | Assuming a customer fee increase is identical to worker pay. |
| Reservation value | A worker's minimum perceived value for accepting an offer, including alternatives and costs. | Treating it as observable or constant across workers. |
| Elasticity | Percentage response in an outcome per percentage change in an input, locally and under an identification strategy. | Using a correlation as a causal elasticity. |
| Spillover | An intervention's effect outside the treated unit, such as supply moving to a neighboring zone. | Counting relocated supply as newly created citywide capacity. |
| Contribution | Revenue less specified variable costs, in currency per order or slot. | Calling it profit without fixed costs and a defined cost scope. |
| Counterfactual | Outcome that would have occurred without the intervention. | Using the treated outcome before intervention without checking common shocks. |
| Service level | A defined customer outcome, such as on-time deliveries divided by eligible deliveries. | Reporting only delivered orders when failed orders are excluded. |
| Exposure | Eligibility to receive or see an intervention, defined before outcomes. | Assigning treatment by who actually takes the bonus. |
| Guardrail | A metric or hard constraint preventing an intervention from causing unacceptable harm. | Treating safety as a soft metric that pay can offset. |
| Fairness | A stated comparison of opportunity or outcomes across relevant groups or areas. | Claiming fairness from a citywide average alone. |
| Handoff | Transfer between store, courier and customer, including waiting and verification where required. | Assuming handoff takes zero minutes. |

## 5. The mechanics, step by step

### Step 1 — Frame the operating decision

**Input:** a proposed service area, decision horizon, customer promise and intervention authority.
**Output:** a decision statement, a named owner and a list of constraints.

Write a decision such as: “By 16:00, select the 18:00–19:00 incentive in North and East, or widen the promise.”
State whether the decision is made once before the shift or revised during it.
An hourly staffing plan cannot replace a dispatch decision made every few seconds.
Name the person who may authorize extra spend, the person responsible for customer promises, and the person who can pause an unsafe operation.

List actions that are actually available.
Possible actions include an advance shift guarantee, an offer-specific bonus, a temporary order cap, or a longer delivery window.
Do not assume the platform can compel independent workers to log in or accept offers.
Where workers are employees, scheduling and compensation constraints may be different.
Document the employment model instead of treating one model as universal.

Separate four statements in the decision record:

- Forecast: predicted order and eligible-supply distributions for the slot.
- Plan: the incentive and service controls selected before the slot.
- Target: the desired on-time rate or coverage threshold.
- Commitment: the actual customer promise and the worker-facing pay terms.

A target is not permission to break a commitment.
If predicted capacity cannot meet the published promise, the owner must choose a feasible intervention or change the offer before accepting more orders.

### Step 2 — Define boundaries and the event clock

**Input:** versioned zone polygons, timestamps, order and courier state events.
**Output:** a reproducible slot cohort and state-time accounting rule.

Use local time with an unambiguous time zone and daylight-saving transition policy.
Store the underlying event timestamps in a common time standard.
Define whether an order belongs to its placement zone, pickup zone, or delivery zone.
A pickup-capacity plan usually attributes load to pickup location, but cross-boundary travel must still be charged to the correct interval.

Split courier intervals at slot boundaries rather than counting a worker in full in every slot touched.
A courier online from 17:45 to 18:15 contributes 0.25 online hour to each adjacent hourly bucket.
If the same worker crosses zones, allocate time by the chosen location rule and ensure the allocated pieces sum to the original interval.
Log changes in geofence versions so historical comparisons are not silently reclassified.

Define state precedence where events overlap.
For example, treat an active assignment as engaged even if a location event says the courier is available.
Handle late or duplicated events with an explicit reconciliation rule.
Unknown state should remain unknown; do not turn missing data into idle time by default.

### Step 3 — Estimate demand without hiding unmet demand

**Input:** historical order attempts, accepted orders, cancellations, traffic and short-horizon drivers.
**Output:** a forecast distribution for order arrivals by pickup zone and slot.

Use attempted requests when available, including requests that could not be served because the platform paused orders or showed an unattractive promise.
Completed orders alone are censored by past capacity and operating policy.
Record when the application was unavailable, fees changed, coverage was restricted, or customers abandoned checkout.
Those observations may require a separate demand model rather than simple historical averaging.

Forecast at the decision resolution, not automatically at the finest available grid.
Sparse five-minute cells are noisy; pooling across similar periods or areas may be necessary.
Avoid smoothing across a barrier that makes neighboring areas operationally distinct.
Use weather, promotions and events only when their values would have been available at decision time.
Evaluate forecasts on rolling historical decisions to prevent leakage.

Create at least a central and a stress scenario.
The stress scenario should combine higher demand with a plausible increase in task time or reduction in supply when those events occur together.
Do not call an independently assembled worst case a calibrated percentile unless the joint probability is actually estimated.
For a real commitment, examine service risk across the joint distribution rather than just the average workload.

### Step 4 — Convert orders into task-hours

**Input:** demand distribution and task-time distribution segmented by geography, vehicle, time and batch policy.
**Output:** expected and stressed courier task-hours per slot.

For a simple unbatched workload, multiply expected orders by expected task minutes per order and divide by 60.
Task time should include travel to pickup, pickup waiting, delivery travel, handoff, and any other engaged interval in the state definition.
Include time spent on cancelled assignments when it consumes capacity.
If cancellations occur before assignment, do not automatically charge them a full completed-order task time.

Averages need consistent weights.
If long trips are disproportionately rejected, accepted-trip times may understate time required to clear all requests.
If the order mix changes during rain, use a rain-specific or scenario-specific time estimate rather than a dry-day mean.
A change in batching policy alters task-hours per order and must be estimated jointly with lateness and eligibility.

For batches, count actual courier minutes once.
Two orders carried during a shared 35-minute tour consume 35 courier-minutes, not 70 courier-minutes.
Allocate costs to the individual orders for reporting only after calculating the physical resource constraint.
Respect food compatibility, carrying capacity and promise windows; a mathematical batch is not necessarily feasible.

### Step 5 — Forecast effective supply

**Input:** planned shifts or historical login behavior, courier availability, location and eligibility.
**Output:** distribution of eligible online and usable courier-hours by zone and slot.

For scheduled workers, start with rostered hours and adjust for absence, breaks, training, and vehicle restrictions.
For a voluntary marketplace, estimate actual login and duration conditional on known offers, weather, events and competing opportunities.
Do not use future logins as if known when backtesting a pre-shift decision.

A logged-in courier may be occupied on a prior job at the slot boundary.
Model that carry-in workload or use a longer rolling horizon with a warm-up period.
Similarly, work assigned near the end of the slot can consume capacity after the slot.
End-of-slot truncation must not make a long trip appear cheap.

The usable fraction represents a planning allowance for breaks, matching friction and variability when those factors are not already included elsewhere.
Write down which effects it includes.
If task time already includes store wait, do not subtract the same store wait a second time in the usable fraction.
The ratio is an operational assumption, not a claim that workers should be engaged for every remaining minute.

Availability is conditional on location and capability.
A car courier may not access a pedestrian-only street; a bicycle courier may not be able to carry a bulky order.
A courier near a zone border may be closer to a pickup in the next zone than a courier inside the zone.
Use travel-time catchments or a network model where the decision warrants it.

### Step 6 — Calculate the gap and its uncertainty

**Input:** scenario task-hours, eligible online hours and a usable fraction.
**Output:** capacity gap in courier-hours, plus its probability or scenario range.

For one bucket and one independent capacity pool:

`usable hours = eligible online hours × usable fraction`

`gap = required task-hours − usable hours`

A positive gap indicates insufficient planned capacity under the stated scenario.
A negative gap indicates modeled headroom, not proof that each order will find an eligible worker.
For multiple zones, a single total can hide simultaneous shortages and surpluses.
Use a feasible flow model or conservative transfer assumptions before treating one zone's surplus as another zone's supply.

A planning coverage ratio is usable hours divided by required task-hours.
It is undefined when required task-hours are zero; report the underlying hours instead.
The ratio is not the same as observed on-time service.
Capacity can be adequate in aggregate while orders arrive in an unmanageable burst.

Estimate uncertainty from held-out forecast errors, simulation or scenario analysis.
Preserve correlation between demand, supply, travel time and store readiness.
Report the range of gap and the service-risk measure that the decision owner actually needs.
Avoid presenting a single modeled decimal place as operational certainty.

### Step 7 — Choose a portfolio of controls

**Input:** gap scenarios, feasible interventions, response estimates, economics and hard constraints.
**Output:** an intervention plan with triggers, owners and expiry times.

A pre-shift guaranteed payment may attract workers earlier and reduce startup uncertainty.
An offer-specific bonus may improve acceptance for a difficult trip without paying the same bonus on every easy trip.
A pickup-wait payment may compensate workers for delays but does not by itself repair the store bottleneck.
The appropriate instrument depends on whether the actual constraint is participation, location, acceptance, or task duration.

A customer-side lever can also change demand.
A wider promise, scheduled slot, temporary delivery fee or pause can reduce immediate load.
Each lever has customer and fairness consequences and may affect future demand.
Do not describe a customer fee as courier pay unless the accounting and communication actually make it so.

A store-side lever can reduce courier task time.
For example, accurate ready notifications can prevent couriers from arriving before picking is complete.
The benefit depends on whether the order would otherwise wait for a courier later.
Improving pickup readiness can shift waiting from the worker to the packed order; food safety and customer timing still matter.

Define the trigger before the shift where possible.
For example, a sustained shortage in eligible courier-hours and increasing unassigned age may trigger a bounded bonus.
Specify how long the bonus lasts, how a courier qualifies, whether a worker sees it before accepting, and what happens if an order cancels.
Avoid retroactive or ambiguous conditions that make pay unpredictable.

### Step 8 — Evaluate incremental economics

**Input:** a baseline and intervention scenario with the same accounting boundary.
**Output:** incremental contribution, cash cost, worker outcomes and uncertainty.

Estimate contribution as incremental completed-order revenue less incremental food and fulfillment costs, base courier pay, incentive pay, refunds and other variable costs included in the decision.
Do not subtract a cost twice when it is already embedded in a per-order margin.
Keep taxes, fixed overhead and customer lifetime effects separate unless they are estimated and explicitly included.

The incentive cost applies to all eligible paid activity, not only newly attracted activity.
A per-order bonus paid on 100 orders at $2 per order costs $200 even if only 10 orders are incremental.
The incremental cost per additional completion is then $20 before any other costs.
That number is not a complete business case: account for contribution from additional completions and avoided failure costs.

If a bonus simply moves couriers from an adjacent zone, local gains may be offset elsewhere.
Evaluate the entire affected market or a credible network of treatment and comparison clusters.
Calculate both the short-term decision and the possibility of future worker participation changes, but do not assign a fictional monetary value to the latter.

### Step 9 — Respect worker and safety constraints

**Input:** applicable law and contract, worker-facing terms, route and vehicle restrictions, safety policy.
**Output:** a feasible policy and escalation path.

Legal and worker-relations owners determine the applicable pay, working-time and classification requirements.
Analysts must not infer that an hour is unpaid merely because a model labels it idle.
Where tips exist, identify whether and how they enter a pay guarantee; rules vary.
Reconcile the modeled payment with the actual pay ledger.

Workers may need breaks, accessible facilities, safe parking and a way to report unsafe pickups or routes.
A capacity plan that can only work by eliminating breaks is not feasible.
Do not make a bonus depend on unsafe speed, handling a phone while moving, or accepting work beyond vehicle capability.
A declining worker should not automatically be classified as unavailable for all future work.

Separate safety guardrails from business trade-offs.
A positive contribution estimate does not justify overriding a safety stop or a legal pay obligation.
Use aggregate reporting and minimize access to sensitive worker information.
Review any individual-level automated decision for the applicable human-review and transparency requirements.

### Step 10 — Execute and monitor

**Input:** authorized plan, live event stream and operational escalation rules.
**Output:** timestamped actions and a shift-level outcome record.

Monitor order age, ready-to-pickup wait, eligible available supply, acceptance, cancellations and actual courier pay.
A low acceptance rate can be caused by poor pay, long pickup distances, unsuitable vehicles or inaccurate offer information.
Inspect the distribution of offer attributes rather than assuming one cause.

Distinguish offered, accepted, picked up and completed orders in dashboards.
A rising count of offers may simply indicate repeated unsuccessful offers for the same order.
An incentive trigger based on offer count alone can therefore amplify a rejection loop.
Monitor unique orders awaiting a feasible assignment and the age of their oldest unassigned order.

Set a maximum intervention duration and an owner for changing or stopping it.
If live data are stale, fall back to a conservative, preapproved operating rule.
The fallback may be a wider promise or a temporary order cap rather than an unbounded payment increase.
Record policy changes and exposure times so later evaluation can reconstruct the counterfactual.

### Step 11 — Learn without confusing correlation and causation

**Input:** intervention logs, eligibility assignments, outcomes, comparison design and pre-intervention history.
**Output:** an incremental effect estimate with uncertainty and limitations.

Bonuses are often activated when demand is unusually high.
A naive comparison of bonus periods with ordinary periods can make bonuses look harmful even when they help.
Comparing only workers who accepted the bonus with those who did not also mixes different worker preferences and opportunities.

Where feasible, randomize policy eligibility at a level that limits spillovers, such as comparable zone-time clusters.
Analyze units as assigned, not merely those who take the offer, to preserve the randomized comparison.
Predefine primary outcomes, the exposure window, exclusions and stopping rules.
Measure adjacent areas and later periods because supply can move in space and time.
If randomization is unavailable, document the assumptions of the chosen observational comparison and the remaining confounding.

A credible result includes precision and practical significance.
A measured increase in online hours may not translate into feasible pickups or better worker earnings.
An improved local outcome may be offset by lost service next door.
Carry findings into the next forecast only when the intervention and context are comparable.

### Step 12 — Translate acceptance into offer capacity

**Input:** an offer-attempt cohort, response timestamps and a breakdown of offer attributes.
**Output:** an estimate of matching friction, with repeated offers preserved.

An accepted job is not equivalent to an offer.
When one order is declined, dispatch may offer it to another eligible worker.
The resulting sequence uses time even if the completed task itself is short.
A long sequence can exhaust the customer's promise before pickup.

Consider 100 unique orders that each need one courier.
Suppose every offer independently has a 0.8 acceptance probability, and at most two offers are possible.
The first attempt produces an expected 80 acceptances.
The expected 20 remaining orders receive a second attempt, producing another 16 acceptances.
The expected number accepted is therefore 96, not 100.
The expected number of offers is 120, not 100.
The accepted-offer fraction is 96 divided by 120, or 80%.
The order acceptance fraction within two attempts is 96 divided by 100, or 96%.
Those are different metrics with different denominators.

This calculation makes strong assumptions.
In practice the second offer is not identical to the first.
The remaining orders may have longer pickups or worse pay relative to their time.
The set of workers offered the job also changes after each decline.
An independence assumption can therefore understate repeated failure.

If the first offer has a 20-second response window and the second has a 30-second response window, an order declined once may lose nearly a minute before assignment.
Actual elapsed time depends on whether an offer is declined early or expires.
Sending offers in parallel can shorten waiting, but it must avoid allocating the same order twice.
Parallel offers also change the worker experience and the denominator of the acceptance metric.
Record those policy details before comparing acceptance across experiments.

Inspect acceptance by pickup distance, offered pay, estimated total time, destination and worker vehicle.
For an hourly pre-shift decision, only use the attributes known at that decision time.
For an offer-level dispatch decision, offer-specific travel estimates may be available.
Do not compare a pre-shift forecast with a model that was allowed to see the realized offer sequence.

A low acceptance probability does not automatically mean that worker preferences are irrational.
The displayed pay may not cover the courier's time, expenses or subsequent repositioning.
The destination may have little subsequent demand.
The order may be unsafe, too bulky, or outside the worker's feasible travel area.
An analyst should test plausible mechanisms rather than label a decline as a failure by the courier.

### Step 13 — Evaluate incentive design before launch

**Input:** candidate payment rules, worker-visible terms, eligibility and a baseline forecast.
**Output:** a rule card that pay operations, legal, finance and worker operations can review.

Record whether the incentive is paid per completed order, per accepted order, per online hour, or as a minimum earnings guarantee.
These rules respond to different constraints.
A completion bonus rewards completed volume but does not compensate all waiting time.
A guaranteed amount can reduce earnings uncertainty, but its effective incremental cost depends on the guarantee's offset and eligibility rules.
An online-hour payment can encourage availability, but its value depends on how the platform defines online time.
Do not assume any of these payment definitions overrides applicable law.

For each proposed rule, document:

1. The precise start and end timestamps, including time zone.
2. The geographic boundary and whether pickup, delivery or acceptance location determines eligibility.
3. The eligible workers, vehicle types and order categories.
4. The moment when the worker sees the amount and conditions.
5. The treatment of multi-order bundles and reassigned orders.
6. The treatment of cancellations, store closures and customer no-shows.
7. Whether the payment stacks with base pay, tips or other incentives.
8. Whether any guarantee is reduced by other earnings and which earnings count.
9. The ledger line, payment date, adjustment process and inquiry channel.
10. The person authorized to pause, correct or withdraw the rule.

Suppose a $4 guarantee is defined as a top-up to base earnings during a qualifying interval.
If a worker earns $3 in eligible base earnings, the illustrative top-up is $1.
If that worker earns $5, the top-up is $0.
The guarantee is therefore not necessarily a $4 incremental payment.
Whether tips or other amounts may be included is a policy and legal question, not an arithmetic assumption.
Show workers the exact applicable rule rather than only the headline amount.

A threshold bonus can have discontinuous effects.
For example, a $12 bonus after four completed jobs has no payout after three jobs.
It may motivate a worker near the threshold, but also expose that worker to uncertainty when jobs are scarce.
It may encourage accepting an unsuitable final job to avoid losing the entire bonus.
Evaluate the distribution of outcomes around the threshold rather than only the average payout.
Include the risk of uncompleted work and show the qualifying progress accurately.

A bonus with a hard expiry can create a rush near the end of its window.
Inspect whether offer acceptance, speed, cancellations or incident reports change at that boundary.
If a worker cannot reasonably predict whether an order will qualify, the advertised amount may not function as the intended signal.
Clarify whether qualification depends on offer time, acceptance time or completion time.
If an operational failure prevents completion, specify the adjustment mechanism in advance.

### Step 14 — Construct a practical experiment

**Input:** a policy that has passed worker, legal and safety review, plus candidate comparable clusters.
**Output:** a predeclared comparison and a plan for interpreting its uncertainty.

Define the estimand before selecting the comparison.
For example: the average change in completed orders per requested order across eligible zone-hours when a preapproved bonus policy is assigned.
That differs from the effect on an individual courier who happens to receive a payment.
It also differs from the effect of paying a fixed amount to every worker.

Randomize assignment at the zone-by-shift level if individual randomization would cause workers to share information or move between treatment arms.
Pair comparable zones or periods on pre-intervention demand, travel time and worker availability where possible.
Choose assignment before observing outcomes for the tested shift.
Keep a durable assignment log, including any exceptions and policy overrides.

The analysis population should include all randomized eligible clusters.
This intention-to-treat comparison preserves the original random assignment even when the operational team cannot fully deliver the policy.
Report take-up and protocol deviations as secondary implementation measures.
Do not remove low-take-up clusters after seeing that their performance was poor.

Use requested orders as one denominator, and separately report completion among admitted orders.
Measure courier-hours, worker earnings distributions, total bonus cost and safety indicators.
The primary result should show the treatment-minus-control difference and an uncertainty interval.
An apparently positive point estimate with a wide interval may still be consistent with material harm.
The number of independent clusters, not merely the number of offers, constrains statistical precision.

Supply mobility complicates the design.
Workers assigned to a treated zone may have worked in a neighboring control zone without the policy.
This interference means the comparison can reflect both treatment benefit and control harm.
Potential responses include larger geographic clusters, buffer zones, or explicitly measuring the whole affected network.
Larger clusters reduce the number of independent units, so they are not a free statistical fix.

Temporal interference is also possible.
A worker may shift a planned evening session into a bonus hour and log off earlier afterward.
Measure outcomes before and after treatment, including the next comparable slot.
Avoid claiming increased total labor supply when only within-day redistribution is observed.

Plan a safe operational override for extreme weather, data failures or a safety incident.
Log every override and the reason for it.
An override may reduce the precision or generalizability of the experiment, but worker safety is not optional.
Avoid repeatedly inspecting results and stopping for apparent success without a predeclared statistical approach.

If a randomized experiment cannot be run, compare similar historical conditions with transparent adjustments.
Check pre-intervention trends and document how policy activation was chosen.
Such a comparison may still be biased by unobserved weather severity, competitor actions or store disruption.
Report it as observational evidence, not as if randomization had occurred.

### Step 15 — Hand over a decision-ready control sheet

**Input:** the forecast, policy comparison, worker-facing terms and monitoring plan.
**Output:** a concise record that an operator can execute and an auditor can reconstruct.

The control sheet should contain the planning date, local slot, zone version and forecast issue time.
It should show order requests, task-hours, eligible courier-hours and the scenario gap.
It should name any feature or data source that was unavailable at decision time.
It should also state the range or scenarios rather than silently substituting the central estimate for certainty.

The selected plan must name the intervention and its alternatives.
For an incentive, include the exact rule version and budget authority.
For an admission cap, include the maximum admitted requests and the customer communication.
For a longer promise, identify when the revised window is displayed and how already accepted orders are handled.
An internal target cannot retroactively alter a customer's existing commitment.

Add a trigger table with an observable metric, a threshold, a minimum persistence period and an owner.
A threshold on a single noisy minute can cause the bonus to oscillate.
A persistence period trades slower response against fewer unnecessary changes.
Choose both from operating evidence and document them as policy parameters.
If the data feed is stale, the control sheet must state who is alerted and which fallback applies.

Example control sheet fields are shown below; values are illustrative and are not a proposed policy for the worked case.

| Field | Example | Owner |
|---|---|---|
| Decision slot | Friday, 18:00–19:00 local time | Marketplace operations |
| Forecast issued | Friday, 16:00 local time | Forecasting |
| Customer promise | Window displayed at checkout | Customer operations |
| Incentive terms | Versioned rule shown before acceptance | Worker operations |
| Spend authority | Approved slot-level budget | Finance |
| Service trigger | Sustained unassigned-order age | Dispatch operations |
| Worker guardrail | Earnings and safety indicators | Worker operations and safety |
| Data fallback | Preapproved conservative admission rule | Incident commander |
| Pay reconciliation | Ledger matched to exposure | Payroll or payments |
| Post-shift review | Outcomes and policy exceptions | Analytics lead |

Each owner should acknowledge the fields they control.
The analytics lead should retain the forecast as issued, not overwrite it with the observed result.
Finance should retain both estimated and actual incentive cost.
Worker operations should record disputes and corrections without exposing individual records to unnecessary audiences.

After the shift, compare actual requests, task time and eligible hours with their respective forecasts.
Decompose the gap into demand error, task-time error and supply error where possible.
Because these factors interact, the decomposition depends on the chosen calculation order.
Label the decomposition method and do not present one ordering as a unique causal explanation.
For a policy effect, use the separate causal comparison rather than the forecast-error decomposition.

Record whether the customer target was met, whether commitments were kept, and whether any safety or pay guardrail was breached.
If the target improved only because requests were rejected, say so explicitly.
If a worker payment was corrected after the shift, keep both the original and correction in the ledger.
The review should produce a concrete owner and due date for changes to the next plan.
It should not convert an uncertain one-shift observation into a permanent incentive coefficient.

## 6. Worked example

### Case and decision

A fictional city marketplace serves two pickup zones, North and East, from 18:00 to 19:00 on a Friday.
All monetary values are US dollars for this example.
The decision is made at 16:00, before actual slot demand and courier attendance are known.
The operations lead can authorize a North incentive or a longer customer promise; legal and safety constraints remain binding.
The arithmetic is illustrative, not an estimate for a named company.

The baseline forecast and planned supply are:

| Input | North | East | Unit |
|---|---:|---:|---|
| Expected order arrivals | 108 | 72 | orders in slot |
| Mean courier task time | 24 | 25 | minutes per order |
| Eligible online supply | 60 | 45 | courier-hours in slot |
| Usable fraction | 0.75 | 0.75 | usable task-hours per online hour |

The task-time estimate includes pickup travel, store wait, delivery and handoff.
It excludes a separate post-delivery repositioning component; the usable fraction is assumed to cover that and other matching slack.
The example is unbatched and assumes one order consumes one task-time observation.
It assumes all listed online hours are eligible for the zone's mix of orders.
In production, verify this by vehicle and geography rather than relying on the simplifying assumption.

The **forecast** is 108 North orders and 72 East orders with the stated task times and online hours.
The **plan** initially provides 60 North and 45 East online courier-hours without an extra incentive.
The **target** is at least 95% of accepted customer orders delivered within the published window, including a separately reported failure rate.
The **commitment** is the window shown to each customer and the pay terms shown to each courier; the target does not replace either commitment.
The capacity arithmetic alone cannot predict whether the 95% target will be met.

### Baseline capacity

North required task-hours are:

`108 orders × 24 minutes/order ÷ 60 minutes/hour = 43.2 courier-hours`.

North usable hours are:

`60 online courier-hours × 0.75 = 45.0 usable courier-hours`.

North headroom is `45.0 − 43.2 = 1.8 courier-hours`.
Its coverage ratio is `45.0 ÷ 43.2 = 1.042`, or about 104.2%.
This is a small aggregate buffer, not evidence that every pickup is covered.

East required task-hours are:

`72 orders × 25 minutes/order ÷ 60 minutes/hour = 30.0 courier-hours`.

East usable hours are `45 × 0.75 = 33.75 courier-hours`.
Its headroom is `33.75 − 30.0 = 3.75 courier-hours`.
Its coverage ratio is `33.75 ÷ 30.0 = 1.125`, or 112.5%.

The two zones together have 105 online courier-hours and 78.75 usable courier-hours.
Their combined workload is 73.2 courier-hours, leaving 5.55 modeled hours of headroom.
That total must not be treated as freely transferable: a bridge separates parts of North and East.
The aggregate can look adequate while North's local pickup queue grows.

### Joint stress scenario

A weather alert creates a plausible joint stress scenario, not a claimed percentile.
North demand rises to 126 orders, mean task time rises to 27 minutes, and eligible online supply falls to 56 courier-hours.
East stays at the baseline values for this simplified scenario.
These changes are assumed together; they are not independent draws.

North workload becomes `126 × 27 ÷ 60 = 56.7 courier-hours`.
North usable hours become `56 × 0.75 = 42.0 courier-hours`.
The North gap is `56.7 − 42.0 = 14.7 courier-hours`.
Coverage is `42.0 ÷ 56.7 ≈ 74.1%`.
The 3.75 East headroom cannot fully cover this gap even under a fictional costless transfer.
Actual transfers take time and may expose East to an unmodeled shortage.

### Intervention options

The team considers a North per-completion bonus and a promise-and-admission adjustment.
Historical estimates are uncertain and are shown here only as scenario assumptions.

| Scenario in North | No intervention | Bonus | Wider promise and cap |
|---|---:|---:|---:|
| Orders requested | 126 | 126 | 126 |
| Orders admitted | 126 | 126 | 105 |
| Completed orders | 93 | 108 | 101 |
| Incremental bonus | $0 | $2 per completed order | $0 |
| Refund or failure cost | $3 per admitted, uncompleted order | $3 per admitted, uncompleted order | $3 per admitted, uncompleted order |

The completion figures are scenario inputs, not outputs mechanically derived from the aggregate coverage ratio.
They reflect assumed matching, timing and cancellation behavior that would require empirical calibration.
All admitted orders are assumed to be either completed or uncompleted by the measurement cutoff.
The 21 orders not admitted under the cap are not counted as failures in the admitted-order denominator, but must be reported as unmet requests.

Suppose each completed order contributes $8 after all ordinary variable costs, including base courier pay, but before the incremental bonus and failure cost.
The baseline contribution is `93 × $8 − (126 − 93) × $3 = $645`.
The bonus contribution is `108 × $8 − 108 × $2 − (126 − 108) × $3 = $594`.
The bonus yields 15 extra completions but reduces modeled slot contribution by `$594 − $645 = −$51`.
The platform spends `$216` in bonus on all 108 completed orders, not just the incremental 15.
That is `$216 ÷ 15 = $14.40` bonus spend per incremental completion, before considering extra margin and avoided failure costs.

The wider-promise-and-cap contribution is `101 × $8 − (105 − 101) × $3 = $796`.
It improves modeled immediate contribution by `$796 − $645 = $151` and completes eight more orders than baseline.
However, 21 requests are not admitted, and customer retention or lost future demand is not priced in this short-run calculation.
The wider promise may also change demand; the table assumes 126 requests before the policy and does not model behavioral response.
These caveats prevent interpreting the result as a universal recommendation.

The operations lead should choose the capped plan only if the customer promise can be changed before commitment and the excluded demand is acceptable under policy.
The lead should consider a more targeted incentive if evidence shows it can recover difficult orders for less spend.
Finance owns the cost boundary; customer operations owns the promise and admission rule.
Worker operations owns clear pay communication and reconciliation; safety can stop the operation if conditions are unsafe.

### Worker and service interpretation

The bonus scenario's higher completion count is not enough to establish improved worker welfare.
It might raise earnings, or it might increase time spent in traffic and unpaid repositioning.
Measure gross and net earnings where reliable, online and engaged hours, pickup wait, and distributions across couriers.
Do not infer net earnings from gross platform payments without vehicle and other expense data.

The capped scenario can improve service among admitted orders while denying access to some customers.
Report completed divided by requested, completed divided by admitted, and on-time among completed alongside cancellations.
For the capped scenario, completed/requested is `101 ÷ 126 ≈ 80.2%`.
Completed/admitted is `101 ÷ 105 ≈ 96.2%`.
Neither ratio alone is an on-time rate; the example does not provide delivery timestamps.

A follow-up experiment should compare policies across comparable zone-time clusters, including neighboring-zone outcomes.
The worker-facing bonus terms must specify eligibility and cancellation treatment before the offer is accepted.
The numerical example demonstrates a decision framework, not a causal claim about actual courier behavior.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the workload and scenario contribution calculations above.
The program is deterministic; it does not fit a forecast or simulate dispatch.
The online-hours calculation is a planning approximation and does not establish customer service probability.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    name: str
    orders: int
    task_minutes: float
    online_hours: float
    usable_fraction: float

    def required_hours(self) -> float:
        return self.orders * self.task_minutes / 60.0

    def usable_hours(self) -> float:
        return self.online_hours * self.usable_fraction

    def gap_hours(self) -> float:
        return self.required_hours() - self.usable_hours()

    def coverage(self) -> float | None:
        required = self.required_hours()
        return self.usable_hours() / required if required else None


@dataclass(frozen=True)
class Scenario:
    name: str
    requested: int
    admitted: int
    completed: int
    bonus_per_completion: float
    margin_before_bonus: float = 8.0
    failure_cost: float = 3.0

    def contribution(self) -> float:
        failures = self.admitted - self.completed
        return (
            self.completed * self.margin_before_bonus
            - self.completed * self.bonus_per_completion
            - failures * self.failure_cost
        )

    def unmet_requests(self) -> int:
        return self.requested - self.admitted


zones = [
    Zone("North baseline", 108, 24, 60, 0.75),
    Zone("East baseline", 72, 25, 45, 0.75),
    Zone("North stress", 126, 27, 56, 0.75),
]

for zone in zones:
    coverage = zone.coverage()
    coverage_text = "n/a" if coverage is None else f"{coverage:.1%}"
    print(
        f"{zone.name}: required={zone.required_hours():.2f} h, "
        f"usable={zone.usable_hours():.2f} h, "
        f"gap={zone.gap_hours():+.2f} h, "
        f"coverage={coverage_text}"
    )

scenarios = [
    Scenario("Baseline", 126, 126, 93, 0),
    Scenario("Bonus", 126, 126, 108, 2),
    Scenario("Wider promise and cap", 126, 105, 101, 0),
]

base = scenarios[0].contribution()
for scenario in scenarios:
    print(
        f"{scenario.name}: contribution=${scenario.contribution():.2f}, "
        f"delta=${scenario.contribution() - base:+.2f}, "
        f"unadmitted={scenario.unmet_requests()}"
    )
```

Use Python 3.10 or later because the type annotation uses the `|` union syntax.
Save the block to a `.py` file and run it with `python filename.py`.
The expected output is:

```text
North baseline: required=43.20 h, usable=45.00 h, gap=-1.80 h, coverage=104.2%
East baseline: required=30.00 h, usable=33.75 h, gap=-3.75 h, coverage=112.5%
North stress: required=56.70 h, usable=42.00 h, gap=+14.70 h, coverage=74.1%
Baseline: contribution=$645.00, delta=$+0.00, unadmitted=0
Bonus: contribution=$594.00, delta=$-51.00, unadmitted=0
Wider promise and cap: contribution=$796.00, delta=$+151.00, unadmitted=21
```

A negative gap means modeled headroom, not negative work.
The monetary output uses the contribution boundary stated in the worked example.
The program deliberately does not infer completions from hours; those are separate scenario assumptions.
For production use, add validated inputs, timestamped event accounting, uncertainty and reconciliation to the pay ledger.
Do not deploy this teaching calculation as an automated pay or admission decision.

## 8. Data, standards, and regulatory context

### Minimum data contract

A usable event model needs stable identifiers for orders, offers, couriers, stores and policy versions.
Limit access to identifiers to authorized users and retain them only for a justified period.
Use an event timestamp and an ingestion timestamp so late data are visible.
Do not assume the first received record is the first event that occurred.

| Dataset | Required fields | Quality check |
|---|---|---|
| Order | order ID, request time, pickup location, promise, status | Request cohort reconciles to accepted and rejected requests. |
| Offer | order ID, courier ID, offered time, pay shown, response and expiry | Repeated offers are distinguishable from distinct orders. |
| Courier state | courier ID, online/offline and engagement timestamps | Intervals do not overlap under the declared precedence rule. |
| Location | timestamp, coarse location or authorized precise location, quality | Stale locations are not treated as current positions. |
| Task | pickup, arrival, handoff, cancellation and completion events | Durations are nonnegative and missing events are flagged. |
| Pay ledger | base pay, incentive, adjustment, tips where applicable | Paid amounts reconcile to worker statements and policy terms. |
| Policy | version, eligible population, effective times, trigger | Each exposure can be joined to the rule in force. |
| Store readiness | order-ready and courier-arrival events | Late or premature ready signals are measured. |
| Context | weather, road closures, events, zone version | Features reflect information available at decision time. |

A courier can receive multiple offers for one order and a single offer can cover a bundle.
Specify the grain of each table before joining them.
Joining an order table to offers without deduplication can multiply revenue and order counts.
Joining location pings without interval logic can multiply courier-hours.

A historical warehouse extract must retain both the original promise and later changes.
Otherwise a delay may appear on time after the promise is silently revised.
Similarly, retain offer terms as displayed, not merely the final pay calculation.
Corrections should be recorded as new ledger events rather than overwriting the original record.

### Measurement and privacy

Record the data dictionary, unit, time zone, location precision, missingness and retention owner for every measure.
A precise GPS trace can reveal sensitive worker behavior beyond the operational need.
Aggregate or coarsen location where the analytical question permits it.
Access restrictions, retention schedules and worker notices should be reviewed by the responsible privacy team.

A dashboard should show coverage of missing state or pay records next to the metric.
A sudden rise in “idle” time may be an event-ingestion failure.
A sudden drop in acceptance may be a changed offer denominator rather than a worker response.
Compare counts from operational events with the finance ledger and customer order records.

### Standards and regulatory boundaries

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, sets requirements for processing personal data where applicable.
Its principles include purpose limitation, data minimization and storage limitation; a location trail is not exempt merely because it is operationally useful.
Legal bases, notices, data subject rights and automated-decision rules depend on the processing and jurisdiction.
Consult the regulation and counsel rather than assuming that aggregation after unrestricted collection resolves the initial collection question.

EU Directive (EU) 2024/2831 addresses platform work, including employment status and algorithmic management.
Member-state transposition and national law determine the applicable implementation and timing.
The directive is not a universal worker-classification rule for all jurisdictions.
A policy owner should obtain current local legal review before using automated decisions about access to work, pay or account restrictions.

In the United States, the Fair Labor Standards Act (FLSA) and applicable state or local law can affect minimum wage, overtime and compensable time when an employment relationship exists.
Classification is fact-specific, and platform terms alone do not determine the answer.
The U.S. Department of Labor's guidance is an entry point, not a substitute for current case-specific advice.
Rules can change, and local requirements may impose additional obligations.

The International Labour Organization (ILO) has published research on digital labor platforms and working conditions.
That research is evidence for examining earnings, hours and risks; it is not itself a binding pay standard for every platform.
The ISO 8601 date-and-time convention can help exchange timestamps consistently, but choosing a timestamp format does not resolve time-zone or interval-allocation logic.
A location boundary is an internal analytical convention unless a contract or local rule gives it legal significance.

Regulatory discussion in this unit is educational, dated to November 2026, and not legal advice.
Check current primary law and local implementation before operational deployment.
No analytics model can authorize a policy that violates a worker's applicable rights.

## 9. Industry practice and case evidence

### Platform evidence: dynamic supply and demand

Uber describes surge pricing as a response to a temporary imbalance between rider demand and available drivers.
Its public explanation describes a customer-facing price mechanism for rides, not proof of a specific food-delivery courier elasticity.
The operational analogy is that time and location affect both sides of a marketplace.
Do not transfer a rides result to grocery delivery without measuring the different trips, pay rules and worker choices.

**Inference:** a localized delivery bonus could shift some workers toward a short zone.
That does not establish how many additional courier-hours appear across the whole city.
A credible analysis would measure neighboring zones and later hours, not just treated-zone logins.
It would also measure whether the additional workers can serve the actual order mix.

### Public evidence: platform work

The ILO's *World Employment and Social Outlook 2021* examines digital labor platforms, including location-based work.
The report documents wide variation in working conditions and institutional arrangements across countries and platforms.
It motivates reporting hours, remuneration and worker constraints in addition to throughput.
It does not validate a particular platform's utilization threshold or bonus rule.

**Inference:** an operator that reports only pay per completed order can miss long online waiting periods.
A practical response is to report the distribution of gross platform earnings per online hour alongside per-engaged-hour measures.
Such measures must state whether tips, expenses and guarantees are included.
They do not decide legal entitlement or replace individual pay records.

### Public evidence: pay rules and measurement

New York City's Department of Consumer and Worker Protection publishes information on its minimum pay rate for app-based restaurant delivery workers.
Its local rule is a concrete example of why platform compensation and time definitions must be reviewed in the applicable jurisdiction.
The agency's current published materials should be checked for current rates and requirements.
This example does not establish the pay rule for all New York workers, all delivery types, or other cities.

**Inference:** a platform operating across jurisdictions should version its pay-rule logic by location and effective date.
The analytics team should reconcile observed online and trip time with the ledger without asserting that its analytical state model is the legal definition.
An engineering deployment also needs an owner for corrections and worker inquiries.

### Research evidence: market design

Castillo, Knoepfle and Weyl study surge pricing and the risk of market failure in ride-hailing.
Their theoretical setting explains why prices can affect both matching and participation.
It is a model of a particular market, not a measured universal delivery-pay response.
Use it to formulate a hypothesis and design a test rather than to insert a borrowed coefficient into a grocery business case.

Across these examples, separate documented facts, model implications and local causal estimates.
Named company descriptions may explain policy intent but need not reveal the counterfactual.
Academic models clarify mechanisms but still require local calibration.
Regulatory sources describe obligations only within their scope and current effective version.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Citywide pooling | City coverage healthy, one zone's queue rising. | Nearby capacity is not reachable in time. | Local lateness and cancellations. | Segment by catchment and travel time; measure transfers. |
| Counting logins as availability | High online count, few eligible offers. | Workers are engaged, on break or unsuitable. | Inflated capacity forecast. | Reconstruct state intervals and eligibility. |
| Censored demand | Forecast falls after repeated order caps. | Rejected requests disappear from training data. | Chronic underplanning. | Retain attempts, availability and policy exposure. |
| Ignoring carry-in | Morning slot starts with unusually low availability. | Prior assignments cross the bucket boundary. | Capacity overstated at shift start. | Split intervals and model carry-in and carry-out. |
| Double-counted batch time | Per-order hours exceed observed engaged hours. | Shared courier time assigned in full to each order. | Biased staffing and economics. | Count physical tour time once before allocation. |
| Repeated-offer inflation | Offers rise faster than unique waiting orders. | One difficult order is repeatedly broadcast. | False demand and trigger signals. | Report unique order cohort and offer attempts separately. |
| Bonus paid to baseline supply | Large spend, small incremental online hours. | Most recipients would have worked anyway. | Negative incremental contribution. | Estimate counterfactual and target the actual constraint. |
| Spatial displacement | Treated zone improves, neighbor degrades. | Workers relocate rather than enter the market. | Citywide service does not improve. | Measure a wider network and randomize suitable clusters. |
| Temporal displacement | Pre-bonus hours improve, later hours collapse. | Workers shift their working time. | The shortage moves to another slot. | Include pre- and post-exposure windows. |
| Unsafe incentive | Speed rises and incident reports increase. | Pay rewards rushing or unsafe route choices. | Harm to workers and the public. | Set safety guardrails and remove speed-linked terms. |
| Misstated pay denominator | Pay per trip rises while online earnings fall. | More waiting or unpaid travel is omitted. | Misleading worker outcome. | Report state-specific hours and reconcile the ledger. |
| Store wait hidden | Courier arrival precedes ready signal. | Dispatch sends workers before picking is complete. | Capacity waste and poor worker experience. | Improve readiness prediction and compensate as required. |
| Policy leakage | Control zones receive workers attracted by treatment. | Mobile supply crosses experiment boundaries. | Biased estimated effect. | Use cluster design and measure spillovers. |
| Promise denominator gaming | On-time percentage rises after rejecting more requests. | Failed or unadmitted orders leave the denominator. | Apparent service improvement masks lost access. | Report requested, admitted, completed and on-time cohorts. |
| Stale location | Offers frequently expire or require long pickups. | Last ping no longer reflects position. | Rejections and delay. | Flag stale pings and use uncertainty-aware matching. |
| Unreconciled pay | Adjustments and complaints rise after rollout. | Policy version or cancellation rule mismatches ledger. | Underpayment risk and loss of trust. | Reconcile by exposure and provide correction workflow. |
| False precision | A single coverage ratio is reported as a service probability. | Workload averages hide timing and matching variation. | Underestimated tail risk. | Use joint scenarios and validate observed service. |
| Broken event pipeline | Unknown state grows while idle time spikes. | Missing updates default to idle. | Wrong intervention and inaccurate worker measures. | Surface data completeness and use a safe fallback. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Task-time mean is representative. | Converts orders into hours. | Weather, geography or order mix shifts. | Compare held-out segment distributions and stress tails. |
| Usable fraction is stable. | Provides a simple capacity allowance. | Breaks, supply density or dispatch policy changes. | Backtest by zone and slot; recalibrate. |
| Task time and usable fraction do not duplicate losses. | Prevents double counting. | Both include store wait or repositioning. | Trace each time component to exactly one term. |
| Workers listed in supply can serve the mix. | Makes hours operationally usable. | Vehicle, accessibility or geography differs. | Join eligibility and travel-time catchments. |
| Slot hours capture relevant work. | Aligns workload and supply. | Assignments cross boundaries. | Split intervals and include carry-in/out. |
| Demand forecast is not censored. | Estimates true request load. | Pauses or long promises suppress attempts. | Retain rejected requests and policy exposure. |
| Completion scenarios are plausible. | Supports comparison of interventions. | Offer response or store behavior changes. | Estimate with a controlled pilot or robust historical comparison. |
| A cost boundary is consistent. | Makes contribution comparable. | Bonus or base pay is omitted or double-counted. | Reconcile each line with finance records. |
| No unmeasured spillovers dominate. | Gives local comparison meaning. | Workers or customers move across zones. | Measure neighboring zones and longer windows. |
| Worker pay terms are executable. | Makes the plan implementable. | Ambiguous eligibility or ledger mismatch. | Review terms and test reconciliation before rollout. |
| Aggregated coverage proxies feasibility. | Provides a fast planning screen. | Bursty arrivals or incompatible matches. | Validate against assignment and service distributions. |
| Historical response generalizes. | Transfers estimated policy effects. | New competitors, rules or weather change behavior. | Monitor drift and update with fresh experiments. |

The simple calculations are not a queueing simulation or an optimization model.
They do not prove a target service level, quantify individual welfare, or establish regulatory compliance.
An estimated positive margin is not evidence of positive long-term value if retention, fixed costs or future supply are omitted.

## 12. Exercises

### Exercise 1 — State accounting

A courier is online for 3.0 hours in a shift, including 1.8 engaged hours and a 0.3-hour break.
The remaining online time is unengaged and not on break; use the definitions in this unit.
**Deliverable:** Calculate engaged utilization and identify each component of online time.
**Acceptance:** Show the denominator, state the units, and explain why the result does not determine legally paid time.
**Time:** 8 minutes.

### Exercise 2 — Workload conversion

A zone expects 96 orders in one hour, each requiring 22.5 courier-minutes on average.
The forecast is unbatched and includes all engaged components.
**Deliverable:** Calculate required courier-hours and usable hours if 52 eligible online hours have an assumed usable fraction of 0.72.
**Acceptance:** State the gap and whether aggregate capacity is sufficient under these assumptions.
**Time:** 10 minutes.

### Exercise 3 — Stress and supply

The same zone faces 110 orders, 25.2 task-minutes per order, and 49 online courier-hours.
Retain the 0.72 usable fraction and assume the joint scenario is plausible but not a calibrated percentile.
**Deliverable:** Calculate stressed workload, usable hours and gap.
**Acceptance:** Compare the stress gap with Exercise 2 and identify one omitted source of uncertainty.
**Time:** 10 minutes.

### Exercise 4 — Diagnose the market

West has 20 usable courier-hours and 25 required task-hours; Central has 12 usable and 8 required hours.
The fastest West–Central crossing takes 25 minutes and customers have short promises.
**Deliverable:** Write a four-sentence diagnosis and name two additional data cuts.
**Acceptance:** Do not treat the four Central hours as costlessly available; distinguish aggregate and local gap.
**Time:** 12 minutes.

### Exercise 5 — Incremental business case

Under a baseline, 80 orders complete and 20 admitted orders fail.
Under a bonus, 90 orders complete and 10 admitted orders fail.
Each completion contributes $7 before bonus and failure cost; the bonus is $1.50 on every completion and each failure costs $2.
**Deliverable:** Calculate contribution for both policies and the bonus cost per additional completion.
**Acceptance:** Include bonus paid to all 90 completions and state whether the incremental result includes retention or fixed costs.
**Time:** 12 minutes.

### Exercise 6 — Experiment design

A city proposes a 30-minute pickup bonus when forecast rain is heavy.
Couriers may move between adjacent zones and choose when to log in.
**Deliverable:** Propose a randomized design, primary outcome, two guardrails, spillover measurement and an analysis population.
**Acceptance:** Explain why comparing only bonus takers with non-takers is biased and specify a fallback if randomization is unavailable.
**Time:** 20 minutes.

### Exercise 7 — Worker constraints

A product manager proposes a bonus for deliveries completed within 12 minutes of acceptance.
Some trips include a 10-minute drive and an unpredictable store wait.
**Deliverable:** Write a review recommending whether to ship the rule, naming the decision owners and a safer alternative.
**Acceptance:** Address speed pressure, pay transparency, applicable legal review, breaks, and a mechanism for reporting unsafe conditions.
**Time:** 15 minutes.

### Exercise 8 — Decision memo

Use the worked example's North stress scenario and the three business-case options.
**Deliverable:** Write a one-page decision memo with forecast, plan, target and commitment in separate labeled lines.
**Acceptance:** Show the gap and incremental contribution, disclose the 21 unadmitted requests, identify decision owners, and name one reason the decision could change.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

Engaged utilization is `1.8 ÷ 3.0 = 0.60`, or 60% of online hours.
Online time consists of 1.8 engaged hours, 0.3 break hours and 0.9 other unengaged hours.
These categories sum to 3.0 hours under the exercise's assumptions.
The legal definition of paid time may differ from these analytic categories.

### Exercise 2

Required hours are `96 × 22.5 ÷ 60 = 36.0 courier-hours`.
Usable hours are `52 × 0.72 = 37.44 courier-hours`.
The gap is `36.0 − 37.44 = −1.44 courier-hours`, or 1.44 hours of modeled headroom.
This aggregate result does not establish on-time service for individual orders.

### Exercise 3

Required hours are `110 × 25.2 ÷ 60 = 46.2 courier-hours`.
Usable hours are `49 × 0.72 = 35.28 courier-hours`.
The stress gap is `46.2 − 35.28 = 10.92 courier-hours`.
Compared with the baseline's 1.44 hours of headroom, the net position worsens by 12.36 hours.
An omitted uncertainty is whether eligible supply and task time are even worse during the peak five minutes.

### Exercise 4

West has a five-hour local shortage and Central has four hours of local headroom.
The combined arithmetic shows a one-hour shortage, even before any transfer loss.
A 25-minute crossing consumes time and may make some transfers incompatible with the promise.
Inspect pickup travel-time distributions, vehicle eligibility, order arrival timing and cross-zone assignments.
A strong diagnosis also checks whether Central's headroom is needed for its own uncertain demand.

### Exercise 5

Baseline contribution is `80 × $7 − 20 × $2 = $520`.
Bonus contribution is `90 × $7 − 90 × $1.50 − 10 × $2 = $475`.
The bonus reduces immediate contribution by `$45` despite ten additional completions.
Bonus spend per additional completion is `$135 ÷ 10 = $13.50`.
Neither figure includes retention, fixed costs or unmodeled changes in worker expenses.

### Exercise 6 rubric

A strong design randomizes comparable zone-time clusters or another unit justified by the network.
It defines eligibility before outcomes and analyzes the assigned population.
A suitable primary outcome might be completed orders per requested order across the affected network.
Guardrails might include worker earnings per online hour and safety incident indicators, with privacy protections.
Measure neighboring zones and subsequent slots for displacement.
Takers differ in location, availability and preferences, so a taker comparison is not a random counterfactual.
If randomization is infeasible, use a predeclared observational comparison and explicitly discuss confounding and spillovers.

### Exercise 7 rubric

A strong answer does not ship a blunt 12-minute deadline that could reward unsafe speed.
It asks safety and legal owners to review the rule and worker operations to review clear pay terms and breaks.
A safer alternative compensates difficult pickup or waiting conditions without rewarding speed or requiring unsafe conduct.
The review includes a worker reporting channel and a stop mechanism for hazardous conditions.
No proposed alternative is assumed compliant until reviewed in the relevant jurisdiction.

### Exercise 8 rubric

A strong memo states the 14.7 courier-hour stress gap and separates scenario assumptions from observed facts.
It reports baseline $645, bonus $594, and capped plan $796 under the stated contribution boundary.
It reports 21 unadmitted requests under the cap and does not call the 96.2% completion/admission ratio an on-time rate.
It names the customer-promise, finance, worker-operations and safety owners.
It distinguishes the published promise and pay terms from the internal target.
It identifies missing retention effects or uncertain completion estimates as possible reasons to change the decision.

## 14. Interview questions

1. **How do you define courier utilization?**  
   Strong: specifies engaged and online state boundaries and checks legal pay separately.  
   Weak: divides deliveries by headcount without a time unit.
2. **Why can a market have adequate citywide supply and still miss promises?**  
   Strong: explains geography, timing, eligibility and travel friction.  
   Weak: assumes every logged-in courier can serve every order.
3. **How would you turn orders into hours?**  
   Strong: includes pickup, wait, delivery, handoff, cancellations and batching without double counting.  
   Weak: uses customer ETA as courier task time without checking overlap.
4. **Why is a high utilization target risky?**  
   Strong: discusses variability, queueing, breaks and worker earnings.  
   Weak: treats 100% engagement as universally optimal.
5. **How do you price an incentive?**  
   Strong: compares incremental outcomes and all eligible payouts against a counterfactual.  
   Weak: divides bonus only by orders that appear incremental without measuring them.
6. **What is the causal problem with bonus-period comparisons?**  
   Strong: identifies activation during high demand, selection and spillovers.  
   Weak: interprets a raw correlation as elasticity.
7. **How would you measure displacement?**  
   Strong: includes adjacent zones and pre/post time windows, with appropriate experimental units.  
   Weak: evaluates only the treated zone during the bonus.
8. **How do you handle rejected customer requests in service reporting?**  
   Strong: reports request, admission, completion and on-time cohorts separately.  
   Weak: drops rejected requests and calls the remaining rate universal service.
9. **What data would you require to audit pay?**  
   Strong: joins displayed terms, versioned eligibility, state times, cancellations and ledger adjustments.  
   Weak: uses only completed-order count and an advertised bonus.
10. **What would make you stop a policy?**  
    Strong: names safety, pay, privacy and service triggers with owners and fallback.  
    Weak: stops only when contribution is negative.

## 15. Key terms

Each link points to the definition and misuse in [Definitions and vocabulary](#4-definitions-and-vocabulary).

- [Acceptance rate](#4-definitions-and-vocabulary) — accepted eligible offers divided by eligible offers.
- [Assignment](#4-definitions-and-vocabulary) — allocation after applicable offer rules.
- [Available time](#4-definitions-and-vocabulary) — defined unengaged and eligible online time.
- [Commitment](#4-definitions-and-vocabulary) — communicated obligation or promise.
- [Completion rate](#4-definitions-and-vocabulary) — completed orders over a specified cohort.
- [Contribution](#4-definitions-and-vocabulary) — revenue less a stated variable-cost boundary.
- [Counterfactual](#4-definitions-and-vocabulary) — outcome without the intervention.
- [Courier-hour](#4-definitions-and-vocabulary) — one courier's hour in a named state.
- [Demand](#4-definitions-and-vocabulary) — customer requests for a zone and slot.
- [Elasticity](#4-definitions-and-vocabulary) — local percentage response to a percentage change.
- [Engaged time](#4-definitions-and-vocabulary) — time in a defined assignment state.
- [Exposure](#4-definitions-and-vocabulary) — pre-outcome eligibility for an intervention.
- [Fairness](#4-definitions-and-vocabulary) — a specified comparison of opportunities or outcomes.
- [Forecast](#4-definitions-and-vocabulary) — estimate or distribution of future conditions.
- [Guardrail](#4-definitions-and-vocabulary) — constraint or metric limiting harm.
- [Handoff](#4-definitions-and-vocabulary) — transfer involving store, courier or customer.
- [Idle time](#4-definitions-and-vocabulary) — online unengaged time under stated rules.
- [Incentive](#4-definitions-and-vocabulary) — incremental payment or offer condition.
- [Marketplace clearing](#4-definitions-and-vocabulary) — feasible matching of demand and willing supply.
- [Offer](#4-definitions-and-vocabulary) — proposed job presented to a courier.
- [Online time](#4-definitions-and-vocabulary) — time connected under the state definition.
- [Paid time](#4-definitions-and-vocabulary) — time covered by the applicable pay rule.
- [Pickup deadhead](#4-definitions-and-vocabulary) — travel to pickup without the order.
- [Plan](#4-definitions-and-vocabulary) — chosen action based on available information.
- [Reservation value](#4-definitions-and-vocabulary) — worker's perceived minimum acceptable offer value.
- [Service level](#4-definitions-and-vocabulary) — defined customer-outcome ratio.
- [Slot](#4-definitions-and-vocabulary) — fixed decision interval.
- [Spillover](#4-definitions-and-vocabulary) — effect outside the treated unit.
- [Surge](#4-definitions-and-vocabulary) — time- or place-dependent price or pay increase.
- [Target](#4-definitions-and-vocabulary) — desired internal performance outcome.
- [Task time](#4-definitions-and-vocabulary) — defined courier time consumed by work.
- [Utilization](#4-definitions-and-vocabulary) — engaged hours divided by online hours.
- [Zone](#4-definitions-and-vocabulary) — versioned geographic planning area.

## 16. Sources

Sources are public and checkable; access and applicable law should be reviewed for the decision date.
They support the specific claims indicated, not the fictional numerical case.

1. International Labour Organization, [*World Employment and Social Outlook 2021: The role of digital labour platforms in transforming the world of work*](https://www.ilo.org/publications/flagship-reports/world-employment-and-social-outlook-2021-role-digital-labour).
   Primary institutional research on platform work and conditions; cross-country evidence is not a local causal estimate.
2. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary legal text for the privacy principles discussed; scope and application require legal review.
3. European Union, [Directive (EU) 2024/2831 on improving working conditions in platform work](https://eur-lex.europa.eu/eli/dir/2024/2831/oj).
   Primary directive; national implementation and current effective obligations must be checked.
4. U.S. Department of Labor, [Fact Sheet #22: Hours Worked Under the Fair Labor Standards Act](https://www.dol.gov/agencies/whd/fact-sheets/22-flsa-hours-worked).
   Official guidance on hours worked for covered employees; not a determination of any courier's classification.
5. New York City Department of Consumer and Worker Protection, [Delivery Workers](https://www.nyc.gov/site/dca/workers/workersrights/Delivery-Workers.page).
   Official local information on app-based restaurant delivery worker protections and pay.
6. Uber, [Surge pricing](https://www.uber.com/us/en/drive/driver-app/how-surge-works/).
   Company explanation of a rides marketplace mechanism; not independent evidence of grocery courier response.
7. Juan Camilo Castillo, Dan Knoepfle and Glen Weyl, [“Surge Pricing Solves the Wild Goose Chase”](https://doi.org/10.1145/3033274.3085098), *Proceedings of the 2017 ACM Conference on Economics and Computation*.
   Theoretical market-design work; its assumptions and setting limit direct transfer to grocery delivery.
8. ISO, [ISO 8601 — Date and time format](https://www.iso.org/iso-8601-date-and-time-format.html).
   Date-time representation context; does not determine local analytical time allocation.

## 17. Links

- [Module 22 index](_index.md): topic sequence and scope.
- [Module 22 survey](../../../course/22_quick_commerce_on_demand_delivery.md): integrated module overview and lab context.
- [Promise tiers and dark stores](01_promise-tiers-dark-stores.md): customer promises and physical coverage.
- [Short-horizon forecasting](02_short-horizon-forecasting.md): slot-level demand forecasts and uncertainty.
- [Dark-store assortment](03_dark-store-assortment.md): store availability and order mix.
