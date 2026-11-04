# Delay Propagation

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 5 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06, and 12; topics 1–4 are recommended  
> Version 2026.5

## 1. Why this topic exists

A carrier reports that a vehicle will reach a distribution center 47 minutes after its planned arrival.
The transport planner can see the delay, but the commercial consequence is not yet known.
If the receiving dock is open, the delay may be absorbed by slack.
If a fixed appointment is missed, the vehicle may wait until the next day.
If the shipment contains a component needed for a production run, the wait may postpone that run.
If the finished product misses a store's delivery window, a promotion may begin without stock.
Those outcomes are not equivalent to adding 47 minutes to every downstream milestone.

A control tower connects movement evidence to operational dependencies and business decisions.
It asks which downstream activities need the delayed quantity, when they need it, and whether their constraints can be changed.
It distinguishes a delay to a physical event from a delay to a customer commitment.
A delay can be harmless for one order and critical for another on the same vehicle.
The same late arrival can be absorbed in one warehouse and trigger a day-long wait in another.

The decisions are time-sensitive.
The planner might hold an outbound vehicle, book a replacement slot, transfer inventory, split a shipment, or tell a customer that a commitment is at risk.
Each response consumes scarce resources or changes someone else's plan.
An expensive expedite may be unnecessary if inventory already covers demand.
A cheap-looking diversion may simply move the bottleneck to a full receiving dock.
Waiting for certainty can close the last practical intervention window.

The analysis is therefore a comparison of feasible actions under stated uncertainty, not a ranking of late vehicles.
The accountable person needs the expected service consequence, an estimate of avoidable loss, the cost of action, and the latest time at which the action can still work.
Where probabilities or financial inputs are weak, a bounded scenario and an explicit escalation are better than a spurious exact answer.

This lesson uses a deterministic dependency model first, then introduces scenarios and expected loss.
Its numbers are illustrative; they are not benchmarks for transit, contractual liability, or customer behavior.
The model does not replace carrier confirmation, a site appointment system, inventory records, or a customer's actual agreement.

## 2. Learning outcomes

After studying this topic you can:

1. **Map** a late leg to downstream activities and name the relevant dependency at each edge; see exercise 1.
2. **Calculate** earliest feasible milestones with durations, release times, and fixed departures; see exercises 2 and 3.
3. **Distinguish** event evidence, forecasts, operating plans, and customer commitments; see exercise 4.
4. **Translate** a delayed receipt into order-level service and financial consequences without double counting; see exercises 5 and 6.
5. **Compare** feasible mitigation actions using incremental cost and scenario-weighted residual loss; see exercises 6 and 7.
6. **Critique** data, dependence, and uncertainty assumptions in an exception assessment; see exercise 8.
7. **Design** an owner, escalation, and closure protocol for an actionable delay; see exercise 9.

A sound answer explains why the proposed action remains feasible at the time of decision.
It gives units, timestamps, quantity, and the decision owner rather than just an alert color.

## 3. The core mental model

A dependency graph represents the chain of activities that must occur before a service outcome is possible.
Nodes are milestones or activities; directed edges mean that one activity cannot proceed until a predecessor condition is satisfied.
The graph is a simplified decision model, not a complete replica of the physical network.

```text
late inbound leg --> receiving appointment --> unloading / quality release
                                                   |
other stock -----------------------> allocation ---+--> outbound cutoff
                                                        |
                                             scheduled departure
                                                        |
                                              customer receipt
                                                        |
                                      availability / service / loss

observations --> updated ETA --> scenarios --> impact and feasible actions
                                              |                  |
                                              +---- owner -------+--> outcome
```

The propagation rule depends on the activity.
A continuous operation may start as soon as both its predecessor and its resource are available.
A departure or an appointment may occur only in a discrete window.
At a window boundary, a small additional delay can produce a much larger downstream delay.
A parallel predecessor can dominate the critical path even when another predecessor is late.
A quantity requirement can make only part of an order late.

For a simple continuous activity, let `ready` be the latest completion among its required predecessors.
Let `release` be the earliest permitted start and `duration` the processing time.
Then `finish = max(ready, release) + duration`.
If a slot must also be booked, the start must be an available feasible slot at or after `ready`, not merely `ready` itself.
If no such slot is known within the planning horizon, the result is unknown rather than automatically tomorrow.

The baseline is the same network evaluated with the agreed operating plan and original constraints.
The scenario is the network evaluated with a changed event, forecast, or action.
Downstream delay is scenario milestone minus baseline milestone, expressed in minutes or hours.
That delay is not itself a financial loss.
Financial loss requires an explicit link from time and quantity to service outcomes and then to cash consequences.

A **forecast** describes a possible future, often with a distribution.
A **plan** is the current intended operational schedule.
A **target** is a performance aspiration such as 98% on-time delivery.
A **commitment** is a promise made to a party under defined terms.
A forecast that arrives after a target is not automatically a breached commitment.
An updated plan is not a customer acceptance of a revised promise.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Leg | Movement between specified locations, with start and end timestamps | Treating a multileg trip as one homogeneous duration |
| Event | Recorded occurrence at a location and time | Assuming a message timestamp proves physical occurrence |
| Event time | When the occurrence happened, with time zone | Substituting the time a message was received |
| Processing time | When the tower ingested or processed a message | Treating a late message as a late truck |
| Estimated time of arrival (ETA) | Forecast arrival timestamp, ideally with uncertainty | Calling it a confirmed appointment or commitment |
| Actual time of arrival (ATA) | Observed arrival timestamp at a defined geofence or gate | Assuming a geofence crossing means unloading is complete |
| Planned arrival | Intended arrival in a particular plan version | Replacing the original baseline whenever the plan changes |
| Commitment | An accepted promise with a defined condition and deadline | Equating an internal plan or target with a legal obligation |
| Milestone | Named point in a process, measured in time | Mixing gate-in, dock-in, and inventory availability |
| Predecessor | Activity or condition that must be satisfied first | Treating every correlated event as a dependency |
| Successor | Activity that depends on a preceding condition | Assuming it always starts immediately |
| Directed acyclic graph (DAG) | Dependency network with no directed cycles | Ignoring a rework loop instead of modeling it explicitly |
| Lead time | Elapsed time between defined start and end milestones | Combining calendar time and working time without conversion |
| Processing duration | Time required to perform an activity | Assuming waiting time is included when it is not |
| Release time | Earliest time an activity may start | Confusing material availability with resource availability |
| Slack | Time an activity can move without changing a specified downstream milestone | Assuming the same slack applies to every order |
| Critical path | Longest binding dependency path for a specified completion under stated assumptions | Assuming it remains unchanged after an intervention |
| Join | Condition that waits for required inputs from multiple predecessors | Adding all predecessor times rather than taking the latest |
| Split | Division of quantity or flow into multiple paths | Assuming split quantities are automatically fungible |
| Slot | Reserved or available resource window, with start and end | Treating an advertised window as an accepted booking |
| Cutoff | Latest qualifying time for an action or handoff | Assuming a cutoff is inclusive without agreement |
| Connection | Scheduled transfer from one activity or service to another | Assuming the next service departs immediately |
| Missed connection | Failure to meet a defined handoff requirement | Treating every late arrival as a missed connection |
| Reslot | A newly authorized appointment | Assuming the next theoretical opening is free |
| Dwell | Elapsed time spent waiting or being processed at a location | Labeling all dwell as carrier delay |
| Capacity | Maximum usable quantity or work over a defined interval | Treating a daily total as available at every minute |
| Buffer | Deliberate time, inventory, or capacity held against variability | Counting the same buffer as available to multiple orders |
| Allocation | Assignment of specific available quantity to demand | Assuming all on-hand stock is unreserved and saleable |
| Stockout | Demand exceeds available saleable stock at a place and time | Treating an inbound delay as proof of a lost sale |
| Lost sale | Demand that does not subsequently convert because service failed | Treating every backorder as permanently lost |
| Backorder | Demand retained for later fulfillment | Counting it as both a full lost sale and full eventual revenue |
| Contribution margin | Relevant revenue less relevant variable costs, in currency per unit | Using revenue or accounting gross margin as the avoidable loss automatically |
| Incremental cost | Change in cost caused by choosing an action rather than baseline | Adding the same unavoidable cost to one option only |
| Expected loss | Probability-weighted loss across mutually exclusive, exhaustive scenarios | Multiplying unrelated marginal probabilities as if independent |
| Counterfactual | What would occur under an alternative action in the same scenario | Comparing an action's best case with inaction's worst case |
| Action deadline | Latest time at which a response can still be executed effectively | Equating it with the eventual customer delivery deadline |
| Decision owner | Person or role authorized to select an action within a limit | Assuming alert recipient has spending authority |
| Residual risk | Remaining exposure after the selected action | Reporting gross exposure as if fully avoided |
| Value of information | Improvement in expected decision value from information, net of its cost when assessed | Valuing information after the decision window has closed |
| Scenario | Coherent set of assumptions about uncertain events and resulting outcomes | Calling one point forecast a probability distribution |
| Horizon | Time interval over which the model evaluates outcomes | Treating unmodeled later consequences as zero |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the time of decision

**Input:** an alert or business question, the current clock time, and the decision maker's authority.
**Output:** one decision statement, a deadline, and a short list of feasible action types.

Write the decision in operational terms.
For example: “By 10:20 local time, should the transport controller reserve the 14:20 recovery departure for order O-17?”
Specify whether the decision is to reserve, purchase, execute, cancel, or notify.
Those verbs can have different costs and approval requirements.
The actual time of the assessment matters because an action that worked at 09:00 may be impossible at 11:00.

Record who owns the action and who must approve exceptional spend.
Distinguish a decision deadline from the later customer delivery commitment.
If the option requires carrier acceptance, make that a dependency rather than claiming the option exists.
If an action is merely requested, mark it requested until the counterparty confirms it.

A notification is sometimes the immediate decision.
A retailer may need time to revise labor or promotions even if the physical flow cannot be improved.
Do not classify communication as physical recovery unless it actually changes the flow.
If no action remains feasible, the appropriate output may be an impact estimate and an escalation, not a fabricated mitigation.

### Step 2 — Fix the baseline and service boundary

**Input:** the original plan version, relevant order commitments, locations, and milestone definitions.
**Output:** a baseline timeline and an explicit definition of success.

Choose a baseline that existed before the disruption.
Preserve its identifier and issue time.
Use the same milestone definition in baseline and scenario: gate arrival is not equivalent to inventory available for allocation.
Define the customer commitment in local time and specify whether it is an arrival, unload, or delivery-complete deadline.
If the contractual rule is unknown, report exposure against the operational deadline and ask the owner to verify the contract.

Define the planning horizon long enough to see whether demand is served late or lost.
A one-day horizon may exaggerate lost sales if customers normally accept next-day delivery.
A month-long horizon may hide a missed promotional window.
The relevant boundary is a business choice, not a model default.

Separate the baseline plan from performance targets.
A target of 98% on-time shipments does not establish an individual appointment.
Conversely, an individual promised deadline may be violated even if the aggregate target is achieved.
Record the current plan separately from the original baseline so that rescheduling does not erase the disruption.

### Step 3 — Establish the affected objects and quantities

**Input:** shipment, order, item, lot, inventory, and facility identifiers, with allocation records.
**Output:** a reconciled list of impacted quantities and alternative sources.

Join a transport leg to handling units and then to the orders they can serve.
An order may use multiple shipments, and one shipment may contain multiple orders.
Check the join keys and effective times before multiplying quantities across links.
A shipment total cannot simply be assigned in full to every associated order.

Identify the saleable quantity expected to become available, not just gross units in transit.
Account for damage, quality holds, temperature excursions, expiry, and customer-specific restrictions when evidence exists.
Do not assume an unverified hold or release; mark the disposition as unknown if it is not known.

List stock already available at each relevant location.
Deduct quantities already committed elsewhere, unless the owner can legitimately reallocate them.
Count a unit only once in a shared inventory pool.
Record substitution rules and whether a substitute requires customer consent.
A nearby unit is not a feasible alternative if transfer time, cold-chain requirements, or reservation rules prevent its use.

### Step 4 — Draw the dependency graph

**Input:** the operating process, resource calendars, and predecessor conditions.
**Output:** a directed graph with durations, constraints, and quantities on relevant nodes and edges.

Use milestones that correspond to decisions or constraints: arrival, check-in, unload, quality release, allocation, load, departure, and receipt.
For each edge, write why the successor cannot proceed first.
A receiving task may require both a truck and an available dock.
A production run may require all critical components, trained labor, and a line.
An outbound order may require a complete order, or it may permit partial shipment.
Those alternatives produce different graphs.

At a join requiring all predecessors, use the latest predecessor completion.
At an optional alternative, use the earliest feasible alternative subject to capacity and approval.
Do not add times for independent parallel activities.
For a split, propagate the relevant quantity along each branch and keep an allocation ledger.
A branch that carries zero allocated units should not generate a duplicate order loss.

Start with an acyclic graph for a bounded planning horizon.
If inspection can trigger rework, either represent distinct attempts within the horizon or evaluate a separate rework scenario.
Do not silently ignore a loop because a DAG algorithm cannot represent it.
Make shared resources visible: two trucks may each have a feasible slot individually but not the same single-capacity slot together.

### Step 5 — Calculate a continuous-time baseline

**Input:** predecessor completion times, release times, and durations.
**Output:** earliest feasible start and finish for each activity, before discrete slot constraints.

Process the graph in an order in which predecessors are evaluated before successors.
For each activity, compute the maximum of predecessor completions and resource release time.
Add its duration using the correct calendar.
If work pauses outside operating hours, calculate in working time or split the activity at the closure.
Never add elapsed hours to a working-hours duration without checking the calendar.

Suppose material is ready at 10:17 and the dock opens at 10:30.
A 25-minute unload finishes at 10:55, not 10:42.
If the dock had opened at 09:00, the unload would finish at 10:42.
The difference is waiting caused by a release constraint, not a longer unload.

When multiple components are needed, the latest available component controls the join.
A 30-minute delay to an early component may have no effect if another component remains 45 minutes later.
If the first component becomes the last, it can become critical.
Recalculate after each major intervention; criticality is not a permanent property of a shipment.

### Step 6 — Apply appointment and departure rules

**Input:** confirmed slot windows, check-in rules, duration requirements, and capacity.
**Output:** the earliest feasible booked start or departure, or an explicit unavailable result.

A fixed receiving appointment is not a continuous resource release time.
If an appointment accepts arrival through 09:15 inclusive, arrival at 09:15 qualifies and arrival at 09:16 does not.
If the rule requires unloading to finish inside a window, test finish time as well as start time.
A window that only controls check-in has different semantics.
State which rule applies rather than using a generic “slot missed” flag.

If a slot is missed, query the actual rescheduling process.
A published calendar is not proof of spare capacity.
The next confirmed slot may be later than the next nominal opening.
If no booking is confirmed, show candidate slots separately from the committed one.
The model should not treat a request as a successful booking.

For scheduled departures, compare load-ready time with the latest handoff cutoff.
The cutoff may precede departure by a fixed loading or documentation interval.
At the boundary, use the actual inclusive or exclusive rule.
If the connection is missed, the next available departure must also have sufficient capacity.
The resulting delay is the difference between the chosen departure outcomes, not the inbound lateness alone.

For example, a 12:00 departure with an 11:40 load cutoff and a later 16:00 departure creates a discontinuity.
A load ready at 11:39 catches the first service.
A load ready at 11:41 misses it and may leave at 16:00 if that service is available.
A two-minute change in readiness can therefore change departure by four hours.
That difference depends on the timetable and capacity, not a universal multiplier.

### Step 7 — Propagate the disruption

**Input:** an observed or forecast change to one or more predecessor milestones.
**Output:** scenario-specific downstream times, lateness, and affected quantities.

Replace the affected baseline input with the scenario input and recalculate all reachable successors.
Retain the unaffected milestones and calendars unless the scenario explicitly changes them.
Do not update only the next node and assume that later milestones shift by the same amount.
At each node ask whether slack absorbs the delay, whether a different predecessor becomes binding, and whether a window is missed.

Measure each resulting delay against the same baseline milestone.
A later gate arrival can lead to an unchanged departure if there is sufficient slack.
A small delay at the gate can result in a much later receipt if it misses a slot and a connection.
Recovery can make a later milestone earlier than its baseline if it uses a faster or earlier authorized alternative.
Explain such negative delay rather than clipping it to zero.

Propagate quantities as well as times.
If only 40 of 100 units fit on the recovery vehicle, 60 units remain on the ordinary path.
Do not claim the entire order recovered.
When a partial delivery is not accepted, distinguish physical arrival of 40 units from fulfillment of the order.

### Step 8 — Translate times into service outcomes

**Input:** scenario receipt times and quantities, order commitments, stock, and demand assumptions.
**Output:** orders and quantities on time, late, backordered, or potentially lost.

Test the right condition for each order.
A customer may measure on-time delivery by complete order, line item, or unit.
Those measures can disagree for a partially fulfilled order.
Report the denominator and the unit of measure.
A delivery can be on time in shipment count while still leaving important units unfilled.

Subtract usable stock only once and apply the documented allocation priority.
Model the time sequence: stock used at 09:00 cannot also cover a different order at 11:00 unless replenished.
If demand is uncertain, show a demand scenario or distribution rather than labeling every delayed unit a lost sale.
If customers can backorder, estimate the portion retained and the incremental cost of serving it later.
If a promotion ends before stock arrives, explain the assumed conversion or lost demand and its evidence.

Identify the causal chain carefully.
A stockout at a store does not prove this truck caused every lost sale.
Another supplier, existing stock, substitution, and demand timing can change the counterfactual.
Attribute only the incremental outcome relative to the same baseline.
If the counterfactual cannot be estimated credibly, report units at risk and leave monetary impact as a range or unknown.

### Step 9 — Build feasible mitigation alternatives

**Input:** available actions, capacity, constraints, lead times, owner authority, and decision deadline.
**Output:** a short set of executable alternatives, including doing nothing.

Typical options include holding a connection, reslotting the dock, expediting transport, splitting an order, reallocating inventory, substituting a product, and changing the customer appointment.
Each changes different nodes or constraints.
A faster truck does not help if the receiving site is closed when it arrives.
An extra dock does not help if the outbound carrier has no available capacity.

For each alternative, identify who can approve it and who must execute it.
Record reservation and cancellation deadlines, available capacity, and whether acceptance is confirmed.
Check safety, quality, temperature control, labor, and customer constraints that could make an apparently fast action invalid.
Avoid treating an unconfirmed capacity quote as a secured route.

Include interactions with other orders.
Holding an outbound vehicle may protect one order but delay several others.
Diverting inventory may shift rather than eliminate lost sales.
Include those incremental impacts in the same evaluation boundary.
Where a resource is shared, model competing actions together or reserve capacity before claiming savings.

### Step 10 — Compare incremental economics

**Input:** action costs, scenario outcomes, probabilities if defensible, and the same baseline for every option.
**Output:** expected incremental total loss or a decision-ready range for each action.

For scenario `s` and action `a`, define `loss(a,s)` as incremental service and operating loss relative to the no-disruption baseline.
Add action cost once, in the scenario in which it is incurred.
If scenario probabilities `p(s)` are calibrated and sum to one, calculate `expected loss(a) = sum(p(s) * loss(a,s))`.
Choose the feasible action with the lowest expected loss, subject to mandatory service and risk constraints.
This is not necessarily the action with the lowest freight cost or the earliest forecast arrival.

Use incremental contribution margin for truly lost sales when it represents avoidable economics.
Include incremental handling, penalty, and transport costs when they would actually occur.
Do not count full sales revenue as lost profit if variable costs are also avoided.
Do not count a contractual penalty unless the contract and circumstances make it applicable.
Do not charge both the full lost-sale margin and an eventual fulfilled sale as if both describe the same unit.

Include the cost of the option even if the disruption does not materialize when a nonrefundable reservation must be made in advance.
If an action is canceled without charge when the good scenario occurs, model the cancellation rule explicitly.
If a mitigation has a success probability, represent success and failure as conditional outcomes or joint scenarios.
Do not assume it succeeds just because its timetable works on paper.

Where probabilities are not credible, compare a scenario table and thresholds instead of inventing confidence.
A break-even calculation can say what probability or margin would justify a spend.
The decision maker can then decide whether available evidence supports that threshold.
For nonlinear or severe consequences, report tail exposure or a worst-case bound alongside expected loss.

### Step 11 — Decide, communicate, and execute

**Input:** comparison, uncertainty, policy constraints, and current resource confirmations.
**Output:** authorized decision, execution request, customer message if needed, and next review time.

Make the decision explicit: action, quantity, cost ceiling, owner, and timestamp.
Record what was known and which forecast version was used.
If a person with sufficient authority cannot be reached before the deadline, apply the documented escalation rule.
Do not let a dashboard color silently commit money or change a customer's promise.

Request execution from the carrier or site and track acceptance.
The requested state, accepted state, and completed state are distinct.
Tell customer service whether an update is an estimate or an accepted revised commitment.
A revised customer commitment requires the appropriate customer's agreement, not merely an internal plan change.
If the action fails or capacity is withdrawn, return to the feasible alternatives while any deadline remains open.

Close only after the relevant result is observed and reconciled.
A vehicle departure is not proof of customer receipt.
A delivered quantity is not proof of a fully served order if units are damaged or rejected.
Capture actual costs and outcomes for later learning.

### Step 12 — Learn from outcome and counterfactual

**Input:** decision log, actual events, actual costs, and an appropriately reconstructed baseline.
**Output:** a reviewed outcome, identified data gaps, and a prioritized improvement.

Compare the forecast at decision time with actual event times; do not use hindsight to rewrite the original forecast.
Separate forecast error from an execution failure and from a change in customer demand.
Record whether the action was available, accepted, and completed.
A successful arrival after an expedite does not prove the expedite was necessary.
Estimate the no-action counterfactual with a defensible method or state that avoided loss is unknown.

Review repeated patterns across lanes, sites, suppliers, and time periods.
A recurring missed appointment may justify changing the plan or buffer rather than buying repeated expedites.
Check whether improvements shift work or risk to another facility or partner.
Assign an owner and measure the effect of a change on service, total cost, and capacity use.

## 6. Worked example

### Situation and decision

A distributor serves a retail customer's promotion with item P.
All timestamps below are on a Tuesday in the same local time zone; there is no daylight-saving change in the example.
At 09:30, a carrier forecasts an inbound gate arrival after a receiving appointment's grace period.
The decision is whether to reserve a recovery outbound vehicle before 10:00.
The transport controller may approve up to USD 400 of incremental transport expense.
The carrier has confirmed the recovery vehicle and its capacity through 10:00; the reservation is nonrefundable once made.

The original plan is a 09:00 inbound arrival, a 09:15 receiving check-in cutoff, and unloading from 09:00 to 09:45.
A normal outbound departure at 11:00 requires the order to be load-ready by 10:30 inclusive.
The normal departure reaches the customer at 14:00.
The customer's accepted commitment is receipt of the complete order by 16:00 Tuesday.
A separate internal target is 98% of orders delivered on time during the month; it is not this order's deadline.

The shipment contains 120 saleable units allocated to the customer's 120-unit order.
The retailer has 20 additional unreserved units of the same item already at its store.
Those 20 units are separate from the 120-unit order and can serve promotion demand before the replenishment arrives.
The store expects 80 units of demand between 14:00 Tuesday and 10:00 Wednesday.
For this illustration, demand is fixed and no substitution is available.
The assumed contribution margin on a truly lost sale is USD 12 per unit.
Demand not served immediately is assumed lost, not backordered.
These commercial assumptions are illustrative and require validation in practice.

### Receiving and outbound rules

The 09:00 receiving appointment accepts gate check-in through 09:15 inclusive.
An arrival at 09:16 or later misses it.
If missed, the site has confirmed a 12:00 reslot and 45 minutes of unloading.
The ordinary outbound service after the 11:00 departure leaves at 16:00, with a 15:30 load-ready cutoff.
It reaches the customer at 19:00 Tuesday, after the 16:00 commitment.
For this illustrative case, the retailer cannot receive the ordinary late delivery until 10:00 Wednesday because its dock is closed after 16:00.
The vehicle and delivery arrangement are assumed to support that next-day receipt without extra cost beyond amounts explicitly listed below.
The customer accepts that revised receipt operationally, but the original on-time commitment is still missed.

The recovery vehicle leaves at 14:20 and can accept 120 units ready by 13:00 inclusive.
It arrives at the customer at 15:50 Tuesday and the customer has confirmed a receiving slot.
Its nonrefundable incremental cost is USD 360, payable whether or not the appointment is ultimately missed.
The extra vehicle is available for the full 120 units and no other order is displaced.
There is no additional handling cost or contract penalty in the stated case.

### Forecast scenarios

The decision-time forecast has two mutually exclusive and exhaustive scenarios:

| Scenario | Probability | Gate arrival | Appointment outcome |
|---|---:|---|---|
| On time | 0.40 | 09:10 Tuesday | Check-in by 09:15; original appointment usable |
| Late | 0.60 | 09:32 Tuesday | Check-in after 09:15; 12:00 reslot required |

The probabilities are assumed decision-time inputs, not observed frequencies established by this example.
Because the decision occurs at 09:30, the 09:10 outcome would normally already be knowable from reliable gate events.
For internal consistency, interpret the scenario arrival as a projected gate check-in for a vehicle whose gate event feed is unavailable at 09:30, and treat 09:10 as an unverified possible past check-in.
A reliable gate confirmation before 10:00 would change the decision and should be sought if it can arrive in time.
The case deliberately illustrates how missing event evidence can leave uncertainty about an event that may already have happened.

### Baseline and late propagation

Under the baseline plan, gate check-in occurs at 09:00.
Unload finishes at 09:45, leaving 45 minutes until the 10:30 outbound cutoff.
The 11:00 departure arrives at 14:00.
The 120-unit order is received before the 16:00 commitment.
The store has enough inventory to serve the assumed 80 units of demand, so there is no lost-sale margin in this baseline.

In the on-time scenario, check-in at 09:10 meets the 09:15 cutoff.
The original slot is available from 09:00 and unloading starts at 09:10.
Unloading finishes at 09:55, still before the 10:30 outbound cutoff.
The normal service arrives at 14:00; the gate is 10 minutes later than plan but customer receipt is unchanged.
The scenario consumes 10 minutes of the original 45-minute outbound slack.
The remaining slack is 35 minutes for the modeled unload-to-cutoff sequence.

In the late scenario, check-in at 09:32 misses the cutoff by 17 minutes.
The next confirmed receiving slot begins at 12:00.
Unloading finishes at 12:45, 180 minutes later than the 09:45 baseline unload finish.
The 10:30 cutoff is missed, so the normal next departure is 16:00.
Although that vehicle reaches the customer at 19:00, the closed dock defers receipt to 10:00 Wednesday.
Relative to the 14:00 Tuesday baseline, receipt is 20 hours later.
Relative to the 16:00 Tuesday commitment, receipt is 18 hours late.
The 32-minute gate delay has not been multiplied mechanically; the missed slot, missed connection, and customer calendar create the larger outcome.

### Service and cost without recovery

The retailer starts the demand period with 20 unreserved units.
Demand before the next-day receipt is 80 units.
Thus `max(0, 80 - 20) = 60` units of demand are unserved and assumed permanently lost in the late scenario.
The 120-unit replenishment still arrives Wednesday; it is not itself 120 lost sales.
The original 120-unit order is late as a complete order, while 60 units of consumer demand are assumed lost.

Incremental lost contribution in the late scenario is `60 units × USD 12/unit = USD 720`.
There is no loss in the on-time scenario under the stated assumptions.
Expected incremental loss without recovery is `0.40 × USD 0 + 0.60 × USD 720 = USD 432`.
This is an expected financial consequence, not a guaranteed USD 432 invoice.

### Recovery and decision

If recovery is reserved and the on-time scenario occurs, the ordinary departure can still carry the order at 11:00.
The recovery reservation nevertheless costs USD 360 because it is nonrefundable.
If the late scenario occurs, unloading at 12:45 meets the recovery cutoff of 13:00 by 15 minutes.
The recovery vehicle arrives at 15:50, 10 minutes before the customer commitment.
It carries all 120 units, and the confirmed customer slot permits receipt.
The initial 20 store units cover the assumed demand before recovery receipt in this example; assume at most 20 of the 80 demand units occur before 15:50.
The recovery receipt then covers the remaining demand, so the modeled lost-sale contribution is USD 0.
This demand-timing condition is essential; a total-period demand of 80 alone would not prove no early stockout.

| Action | On-time scenario loss | Late scenario loss | Expected loss |
|---|---:|---:|---:|
| No recovery | USD 0 | USD 720 | USD 432 |
| Reserve recovery | USD 360 | USD 360 | USD 360 |

The expected advantage of reserving is `USD 432 - USD 360 = USD 72`.
The break-even late-scenario probability is `USD 360 / USD 720 = 0.50`, or 50%, assuming full recovery and unchanged loss estimates.
The assumed 60% late probability exceeds that threshold, so reserve within the controller's USD 400 authority.
The decision is sensitive: at a 40% late probability, expected no-recovery loss is USD 288 and reservation would be more expensive by USD 72.
If only half of the lost-sale margin were avoidable, reservation would not be economical at the stated probability.
Mandatory service rules or reputational risks could still change the authorized decision, but they are not monetized in this case.

The controller records the 09:30 evidence, reserves before 10:00, and requests gate confirmation.
If the gate event confirms the on-time scenario after a nonrefundable reservation, the USD 360 remains spent.
If confirmation arrives before any irreversible commitment, the controller should use that information rather than blindly reserving.
The customer message should identify the current receipt estimate and avoid claiming a revised promise until accepted.
The action remains open until the carrier confirms execution and the customer confirms receipt.

### Second worked case: split recovery and displaced demand

The first case uses one order and a vehicle that can carry all 120 units.
The following separate case shows why a partial recovery must be allocated before its benefit can be counted.
The currency is USD and all times are local on the same Wednesday.
The numbers are illustrative rather than a description of an actual retailer.

A distributor has two customer orders awaiting the same inbound shipment.
Customer A ordered 70 units and requires receipt by 16:00.
Customer B ordered 50 units and requires receipt by 16:00.
Both customers accept a partial shipment, and each line is measured on units received by the deadline.
The original plan delivers both orders on time.
After a missed inbound appointment, the entire shipment is forecast to miss the ordinary connection.
Without intervention both orders arrive Thursday morning.

The retailer-specific consumption consequences are different.
Customer A expects 45 units of demand before Thursday receipt and has 15 unreserved units on hand.
Customer B expects 35 units of demand before Thursday receipt and has 10 unreserved units on hand.
For this calculation, all unmet demand is permanently lost and no substitute is available.
The relevant contribution is USD 14 per lost unit for A and USD 9 per lost unit for B.
Assume no other penalties or delivery costs.

At 11:00 the controller can book a recovery vehicle for USD 240.
It has capacity for only 40 units.
It arrives at both customers by 15:30 and the customers' receiving appointments are confirmed.
The recovery vehicle must leave at 13:30, and all 120 units will be ready by 13:00 in this scenario.
The residual units follow the ordinary Thursday route.
There is no cost to split the load other than the stated USD 240.

Suppose A has 5 demand units before 15:30 and B has 4 demand units before 15:30.
Each customer's existing stock therefore covers demand up to the recovery arrival.
Recovery quantities can cover the remaining pre-Thursday demand.
This timing assumption is necessary because a recovery at 15:30 cannot repair sales already lost at 14:00.
All quantities are integer units in this example.

The no-action incremental lost demand at A is `max(0, 45 - 15) = 30` units.
At B it is `max(0, 35 - 10) = 25` units.
No-action incremental lost contribution is therefore `30 × USD 14 + 25 × USD 9`.
That is `USD 420 + USD 225 = USD 645`.
The order delay affects 120 units, but only 55 units of consumer demand are lost under these assumptions.

Each recovered unit can prevent at most one lost sale.
The first 30 recovery units assigned to A each avoid USD 14 of loss.
Further units assigned to A do not avoid modeled loss because A's stockout is already covered.
The first 25 units assigned to B each avoid USD 9 of loss.
Further units assigned to B likewise have no modeled lost-sale benefit.
This is a marginal-benefit calculation, not a claim that the customers have equal contractual priority.

Allocate 30 of the 40 recovery units to A and the remaining 10 to B.
A then has `15 + 30 = 45` units available across the demand period.
B has `10 + 10 = 20` units against 35 units of demand.
A loses zero modeled sales; B loses 15 modeled sales.
The residual lost contribution is `15 × USD 9 = USD 135`.
Total incremental loss with recovery is `USD 240 + USD 135 = USD 375`.
The modeled net improvement is `USD 645 - USD 375 = USD 270`.

The 30 recovery units assigned to A are not also available to B.
The 80 units not on the recovery vehicle still arrive Thursday.
Their later delivery satisfies remaining orders, but it does not reverse demand assumed permanently lost.
If the customers backorder instead, the lost-sale model and its financial conclusion must change.
On-time order-line quantities are 30 of 70 for A and 10 of 50 for B.
Neither order is completely on time, even though 40 of 120 units arrive by the deadline.
The customer service measure must specify whether it counts complete orders, lines, or units.

Consider the alternative allocation of all 40 recovery units to B.
B needs only 25 additional units to avoid its modeled stockout.
The other 15 recovery units provide no additional lost-sale benefit in the stated horizon.
A still loses 30 sales, worth USD 420 of contribution.
Total modeled loss is `USD 420 + USD 240 = USD 660`.
This is USD 15 worse than taking no action, despite delivering 40 units early.
The difference demonstrates why a partial recovery needs an allocation decision rather than a simple “vehicle booked” flag.

The allocation is economically preferred only under the stated objective and constraints.
A contract could require an equal split, or a customer may refuse a partial order.
A may have a strategic priority not captured in contribution margin.
If a constraint requires at least 20 recovery units for B, allocate 20 to each customer.
A then loses 10 sales, and B loses 5.
Lost contribution becomes `10 × USD 14 + 5 × USD 9 = USD 185`.
Adding USD 240 for the vehicle gives USD 425.
The constraint costs `USD 425 - USD 375 = USD 50` relative to the unconstrained modeled allocation.
That USD 50 is an opportunity cost under these assumptions, not evidence that the constraint should be ignored.

Before committing, the controller confirms that A's 30 units and B's 10 units can be physically separated.
The warehouse checks labels, lot restrictions, and the available handling time.
Customer service verifies both receiving appointments and partial-delivery acceptance.
The controller records the residual Thursday quantities: 40 units for A and 40 for B.
The decision log records the commercial assumptions and any customer-priority rule.
On completion, proof of delivery and actual saleable quantities replace the forecast in the outcome record.

### Sensitivity: recovery vehicle is not certain to succeed

Return to the first worked case, not the split case.
Assume the recovery vehicle still costs USD 360 regardless of the scenario.
Now suppose that, conditional on the late scenario, it succeeds with probability 0.80.
If it fails, the full USD 720 late-scenario lost contribution remains.
The on-time scenario continues to have no service loss, even if recovery is not used.
This scenario definition captures only the recovery failure relevant when the inbound appointment is missed.

The probability of both late inbound and failed recovery is `0.60 × 0.20 = 0.12`.
The probability of both late inbound and successful recovery is `0.60 × 0.80 = 0.48`.
The on-time inbound scenario has probability 0.40.
These three mutually exclusive outcomes sum to `0.12 + 0.48 + 0.40 = 1.00`.
The joint probabilities use the stated conditional success rate; they do not assume independent events.

Expected residual service loss is `0.12 × USD 720 = USD 86.40`.
Expected total loss with recovery is `USD 360 + USD 86.40 = USD 446.40`.
No recovery still has expected loss of USD 432.
Recovery is now USD 14.40 more expensive in expectation.
In this version, the controller should not reserve solely on the stated expected-cost objective.
The earlier decision changed because success was no longer certain, not because the arithmetic was inconsistent.

The break-even recovery success probability, conditional on the late scenario, is:

```text
USD 360 = 0.60 × success_probability × USD 720
success_probability = USD 360 / USD 432 = 0.8333, or 83.33%
```

An 80% conditional success probability falls below that threshold.
If the USD 360 charge applied only when a late inbound actually occurred, the threshold would be different.
For example, with a conditional USD 360 charge and certain success, the expected action cost would be `0.60 × USD 360 = USD 216`.
The reservation terms are therefore a causal input to the economic comparison.
An actual decision should use the applicable quote rather than choosing the cheaper assumption.

### Sensitivity: value of a timely gate confirmation

Again use the first case with certain recovery success and a nonrefundable USD 360 reservation.
Suppose a trusted gate confirmation can be obtained before 10:00 at a cost of USD 20.
Assume it reveals the scenario perfectly and the recovery vehicle remains available until the confirmation arrives.
If the inbound is on time, the controller does not reserve.
If it is late, the controller reserves and incurs USD 360.
The expected reservation cost with information is `0.40 × USD 0 + 0.60 × USD 360 = USD 216`.
Adding the USD 20 information cost gives expected total loss of USD 236.
Relative to immediate reservation at USD 360, the net expected benefit is USD 124.
The gross expected value of perfect information is USD 144.

That value depends on the information arriving before the irreversible booking deadline.
If the gate confirmation arrives at 10:05 and the vehicle must be reserved by 10:00, it cannot guide that reservation.
If the information is imperfect, evaluate outcomes conditional on each possible report rather than treating it as perfect.
If waiting for the report risks losing the vehicle, model the probability and cost of that lost option.
The value of an event feed is therefore decision- and time-specific, not a fixed percentage of freight spend.

### Sensitivity: a shared dock changes the counterfactual

Suppose the original first-case recovery consumes a customer dock for 20 minutes.
Another customer's previously planned delivery would then be deferred and incur USD 100 of incremental loss.
Assume the extra loss occurs only in the late scenario when the recovery is actually used.
The expected displacement cost is `0.60 × USD 100 = USD 60`.
Add it to the original USD 360 expected recovery loss to obtain USD 420.
Recovery remains USD 12 better than no action at USD 432, under these assumptions.

If displacement instead costs USD 150 in the late scenario, its expected cost is USD 90.
The recovery total becomes USD 450, making no action USD 18 better in expectation.
This comparison requires evidence that the dock truly is shared and that the other loss is incremental.
It also requires checking that the customer's own commitment permits the proposed displacement.
The example does not authorize a planner to override that commitment.

### Decision handoff checklist

The controller's handoff should give the original customer deadline and baseline receipt.
It should identify the forecast issue time and the event source used for gate status.
It should show the confirmed receiving reslot and the latest outbound handoff.
It should state how much quantity can travel on each path.
It should show the cost, cancellation terms, and conditional success of each option.
It should name the individual authorized to spend and the person executing the booking.
It should identify any orders or dock slots displaced by the intervention.
It should state which customer communication is an estimate and which promise has been accepted.
Finally, it should specify the next observation and the time at which the decision must be revisited.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It evaluates the worked example with minute offsets from Tuesday 00:00.
The code models the two scenarios and checks the appointment, receiving, departure, customer calendar, and recovery cutoff.
It assumes all capacities and demand-timing conditions stated in section 6 are confirmed.
It is not a general scheduling optimizer, event processor, or replacement for a time-zone-aware production system.

```python
from dataclasses import dataclass


def minute(day: int, hour: int, minute_of_hour: int) -> int:
    """Tuesday is day 0; Wednesday is day 1."""
    return day * 24 * 60 + hour * 60 + minute_of_hour


def label(value: int) -> str:
    day, rest = divmod(value, 24 * 60)
    hour, mins = divmod(rest, 60)
    return f"{'Tue' if day == 0 else 'Wed'} {hour:02d}:{mins:02d}"


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    gate: int


SCENARIOS = (
    Scenario("on_time", 0.40, minute(0, 9, 10)),
    Scenario("late", 0.60, minute(0, 9, 32)),
)
APPOINTMENT_START = minute(0, 9, 0)
CHECKIN_CUTOFF = minute(0, 9, 15)  # inclusive
RESLOT = minute(0, 12, 0)
UNLOAD_MIN = 45
NORMAL_CUTOFF = minute(0, 10, 30)  # inclusive
RECOVERY_CUTOFF = minute(0, 13, 0)  # inclusive
CUSTOMER_DEADLINE = minute(0, 16, 0)  # inclusive
FIRST_RECEIPT = minute(0, 14, 0)
RECOVERY_RECEIPT = minute(0, 15, 50)
NEXT_DAY_RECEIPT = minute(1, 10, 0)
DEMAND = 80
STORE_STOCK = 20
MARGIN_USD = 12
RECOVERY_COST_USD = 360


def unload_finish(gate: int) -> int:
    start = max(gate, APPOINTMENT_START) if gate <= CHECKIN_CUTOFF else RESLOT
    return start + UNLOAD_MIN


def receipt(gate: int, reserve: bool) -> int:
    ready = unload_finish(gate)
    if ready <= NORMAL_CUTOFF:
        return FIRST_RECEIPT
    if reserve and ready <= RECOVERY_CUTOFF:
        return RECOVERY_RECEIPT
    return NEXT_DAY_RECEIPT


def incremental_loss(gate: int, reserve: bool) -> int:
    arrival = receipt(gate, reserve)
    # The stated demand-timing assumption prevents a pre-15:50 stockout.
    lost_units = max(0, DEMAND - STORE_STOCK) if arrival > CUSTOMER_DEADLINE else 0
    return lost_units * MARGIN_USD + (RECOVERY_COST_USD if reserve else 0)


assert abs(sum(s.probability for s in SCENARIOS) - 1.0) < 1e-12
for scenario in SCENARIOS:
    print(
        f"{scenario.name}: unload={label(unload_finish(scenario.gate))}; "
        f"no_recovery={label(receipt(scenario.gate, False))}; "
        f"recovery={label(receipt(scenario.gate, True))}"
    )
for reserve in (False, True):
    expected = sum(
        s.probability * incremental_loss(s.gate, reserve) for s in SCENARIOS
    )
    print(f"{'reserve' if reserve else 'no recovery'}: USD {expected:.2f}")
print(f"break-even probability: {RECOVERY_COST_USD / (60 * MARGIN_USD):.2f}")
```

Expected output:

```text
on_time: unload=Tue 09:55; no_recovery=Tue 14:00; recovery=Tue 14:00
late: unload=Tue 12:45; no_recovery=Wed 10:00; recovery=Tue 15:50
no recovery: USD 432.00
reserve: USD 360.00
break-even probability: 0.50
```

The program uses integer minutes and an inclusive check-in cutoff, as specified.
It treats the ordinary late delivery as received on Wednesday because the Tuesday customer dock is closed.
The loss function is deliberately specific to the case: it is not a general demand-by-time inventory model.
In particular, its zero loss before the commitment depends on the stated assumption about demand before 15:50.
The simplified function also assumes a late receipt loses exactly 60 units of demand and creates no other cost.
Production implementations should use time-indexed demand, allocations, actual appointment capacity, explicit time zones, and audited scenario inputs.
They should account for action success and shared-resource conflicts rather than assuming both away.

## 8. Data, standards, and regulatory context

A usable propagation assessment joins several types of evidence.
The carrier may supply gate events and an estimated arrival; a site system may own appointments and unloading completion.
An order management system may own customer lines and commitments, while a warehouse system owns inventory and reservations.
A finance or commercial owner should supply relevant margins, costs, and penalty rules.
None of these systems automatically establishes the truth of every other system.

At minimum retain identifiers for shipment, leg, handling unit, item, order line, location, and partner where relevant.
Keep quantity, unit of measure, event type, event time, time zone, processing time, source, and data quality status.
Preserve the original planned milestones and their version or effective time.
Record forecast issue time and uncertainty separately from the forecasted event time.
Store the decision, its inputs, the requested action, acceptance, and observed result with timestamps.

The GS1 Electronic Product Code Information Services (EPCIS) 2.0 standard provides a vocabulary for sharing supply-chain event data about what, when, where, why, and how.
Its event types and identifiers can help organizations exchange evidence about movements and transformations.
EPCIS does not itself guarantee that an event is timely, complete, accurate, or commercially authorized.
It does not infer that a late container caused a lost sale or determine which mitigation should be purchased.
See the GS1 standard in section 16 and agree a partner-specific profile before operational use.

The GS1 Global Location Number (GLN) can identify a party or location in a shared coding scheme.
A GLN still requires a maintained mapping to the facility, gate, or operational resource that the decision model uses.
A facility-level location may not identify the dock needed for a slot calculation.
Similarly, a shipment identifier must be reconciled to order lines and quantities; an identifier alone does not establish a complete allocation.

EDI, or electronic data interchange, may carry shipment notices, status messages, and appointment-related information under a trading-partner agreement.
Message semantics, version, code lists, and acknowledgment behavior depend on the applicable implementation guide.
A sent message is not the same as an accepted reslot or an observed physical event.
Validate which fields represent original plans, revised plans, forecasts, or actual events.

Use an agreed definition of on-time service.
A trading partner may require arrival at gate, delivery at dock, proof of delivery, complete order, or a defined percentage of units.
The tower's metric must not override the contract.
Contractual penalties and force-majeure terms depend on actual agreements and governing law; no general penalty is assumed here.
This material is educational, not legal advice.

Location and driver data can be personal data when they identify or relate to a person.
Where the EU General Data Protection Regulation (GDPR) applies, processing must have an appropriate legal basis and comply with applicable principles including data minimization and storage limitation.
The regulation's scope, lawful basis, and retention requirements depend on the particular processing context; obtain legal and privacy advice rather than assuming a universal retention period.
Commercially sensitive order allocations and carrier positions also require access controls and appropriate sharing agreements.
A control tower should share the minimum information needed for the recipient's decision.

Operational timestamps require particular care.
Use a common representation internally, retain the original zone and offset, and display local time where decisions are made.
Define behavior at daylight-saving transitions and for ambiguous or nonexistent local times.
An event received late may describe an on-time physical arrival; do not convert message latency into transport delay.
Deduplicate repeated messages without discarding legitimate corrections or updated estimates.

A completeness check should establish whether all relevant legs and orders are represented.
A freshness check should distinguish the age of the latest position from the age of the forecast.
An accuracy check should compare milestones with independently verified outcomes when available.
A lineage check should show which source and transformation produced a downstream risk score.
Unknown, stale, or conflicting inputs should surface as uncertainty, not silently become zero delay.

## 9. Industry practice and case evidence

### What operating teams commonly combine

A practical control tower combines transport visibility with site calendars, inventory availability, and order priorities.
The transport event answers where a shipment is or may be.
The appointment system answers whether the site can process it at a specific time.
The inventory and order records answer which demand can still be served.
The economic assessment adds which feasible response is justified.
These are distinct questions, often owned by different teams and partners.

A mature operational process typically preserves both the original and revised schedule.
Without the original baseline, a rescheduled shipment can appear on time even when the customer's original need was missed.
Without the revised plan, the site cannot coordinate current resources.
The two views serve different purposes and should not overwrite each other.
A useful exception record links them to the evidence and the person who made the change.

### Toyota and the 2011 earthquake

Toyota reported on April 8, 2011, that after the Great East Japan Earthquake it planned production at all Japanese vehicle-production facilities from April 18 to April 27 at approximately 50% of normal volume.
It said that the decision on production after the spring holiday would follow an assessment of the parts supply situation.
The documented fact is that parts supply affected the production planning response after the earthquake.
The announcement does not validate the numerical model in this lesson or establish the loss from any specific delayed shipment.

[INFERENCE] A tower that only tracks the first-tier shipment may miss a production risk when a lower-tier site supplies a unique component.
To act, the planner would still need part-to-product mappings, inventory coverage, recovery time, and alternative-source feasibility.
Mapping a dependency does not guarantee that an alternative source is qualified or has capacity.
No quantified avoided loss is asserted here.

### Maersk and port congestion

Maersk's public customer advisories discuss port congestion, changing vessel schedules, and downstream disruption to inland transport and delivery plans.
Those advisories illustrate the industry practice of issuing revised operational information when a network constraint changes.
A revised schedule is an input to a customer's planning process; it does not establish the customer's actual inventory or financial loss.
The specific appointment, probability, and margin figures in section 6 are invented for teaching and are not Maersk case results.

[INFERENCE] An importer could use an updated vessel arrival together with its own drayage appointment and warehouse capacity to assess whether an inland slot is at risk.
A credible analysis would need the specific vessel, terminal, container, customer appointment, and time-dependent capacity.
The port delay alone would not establish which customer order is late.
Nor would it establish that paying for a faster inland leg can recover a missed ship arrival.

### What evidence can and cannot establish

A carrier status report is evidence about an operational event or estimate, depending on its type and source.
An appointment confirmation establishes a booking under stated conditions, not necessarily that the dock will be ready when the truck arrives.
Proof of delivery supports a receipt milestone, but may not establish that the goods are saleable or the final consumer's demand was fulfilled.
A financial estimate requires additional assumptions about demand, substitution, margin, and contract terms.

For after-action evaluation, compare decisions against what was knowable at the time.
A before-and-after reduction in delay may reflect a network change, seasonality, or lower demand rather than the intervention alone.
Where possible, compare similar exposed flows or use a documented counterfactual.
Do not label the full value of a saved order as savings without deducting costs and considering what would have happened anyway.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating processing delay as transport delay | Old event arrives with a recent ingestion timestamp | Event and processing times are conflated | Unnecessary escalation or expedite | Preserve and compare both timestamps and source quality |
| Missing a slot boundary | ETA moves past a cutoff but risk remains unchanged | Model adds a constant transit delay | Order impact is understated | Evaluate actual appointment and grace-period rules |
| Inventing a reslot | Dashboard shows a next-day slot with no booking identifier | Nominal opening is treated as available capacity | False recovery commitment | Separate candidate, requested, and confirmed slots |
| Ignoring customer hours | Carrier ETA falls after receiving closes | Transport arrival is treated as receipt | Service recovery is overstated | Apply destination calendar and receipt definition |
| Mixing local times | Apparent negative travel time across zones | Offset or daylight-saving rule is missing | Wrong connections and missed deadlines | Store offsets and validate time-zone conversions |
| Duplicate events | Same milestone appears repeatedly | Retries are interpreted as new occurrences | Multiple alerts or double-counted movement | Deduplicate by source identity while retaining corrections |
| Double-counting inventory | Two orders claim the same available units | Allocations are not reconciled | Both orders appear protected | Use a time-stamped reservation ledger |
| Double-counting financial loss | Revenue loss and full lost margin both appear | Overlapping economic measures are added | Excessive mitigation spending | Use incremental contribution and reconcile categories |
| Treating backorders as lost sales | All delayed units use full lost-sale margin | Retained demand is ignored | Exposure is overstated | Separate permanently lost demand from delayed fulfillment |
| Ignoring partial quantities | Full order marked recovered when one pallet moves | Split capacity is not propagated | Hidden customer shortfall | Track quantity by branch and acceptance rule |
| Overlooking a parallel predecessor | A late part is marked critical while another part is later | Join is modeled as a sum or single path | Resources spent on a nonbinding delay | Use the maximum of required predecessor times |
| Assuming independent delays | Several legs all worsen during a weather event | Shared disruption is omitted | Tail risk is understated | Use joint scenarios or a defensible dependence model |
| Assuming an action succeeds | Booked expedite has no confirmed capacity | Requested execution is treated as certain | Residual exposure disappears on paper | Model acceptance and conditional failure |
| Forecast overwrite | Only the latest ETA remains | Prior forecast versions are discarded | Cannot audit decision quality | Retain issue time, version, and input snapshot |
| Stale commitments | Order deadline differs between systems | Revised plan overwrites original promise | Breach is hidden or falsely declared | Preserve accepted promise and change history |
| Ignoring displacement | Recovery vehicle protects one order but delays others | Shared capacity is modeled independently | Net loss increases | Include displaced flows and capacity constraints |
| Unbounded queue | Every small late leg creates an urgent alert | No decision deadline or impact threshold | Operators miss high-value actions | Rank by feasible decision window and incremental impact |
| Missing demand timing | Total-day stock appears sufficient | Demand precedes recovery receipt | Early lost sales are missed | Model inventory and demand by time bucket |
| False precision | Probabilities reported to two decimals without evidence | A point estimate is treated as calibrated | Weak decisions appear certain | Show ranges, sensitivity, and data provenance |
| Unauthorized promise | Internal revised ETA is sent as an accepted commitment | Communication rights are undefined | Customer relationship and contract risk | Separate forecast, proposal, and accepted promise |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or relax it |
|---|---|---|---|
| Predecessor relationships are known | Determines what can propagate | Hidden supplier or material dependency | Reconcile bills of material and operating process with owners |
| Activities follow the stated calendar | Converts readiness into completion | Shift change, closure, or overtime | Compare calendar with site confirmation and actual events |
| Slot capacity is confirmed | Establishes a feasible path | Another booking consumes the slot | Retain booking ID and recheck before execution |
| Arrival event has correct semantics | Locates the actual milestone | Geofence crossing differs from gate check-in | Compare source definitions and sample gate records |
| Forecast scenarios cover relevant outcomes | Supports expected-loss comparison | Extreme delay or disruption omitted | Add a tail scenario and review historical misses |
| Probabilities are suitable at decision time | Gives meaningful expected loss | Stale or uncalibrated forecast | Backtest by lane and horizon; show sensitivity |
| Demand timing is known or modeled | Determines actual stockout exposure | Demand peaks before recovery receipt | Use time-bucket demand and scenario ranges |
| Inventory is saleable and unallocated | Prevents false coverage | Hold, damage, expiry, or reservation | Reconcile disposition and reservation snapshots |
| Full order can fit on recovery vehicle | Makes recovery feasible | Vehicle or handling capacity is lower | Confirm capacity and model quantity splits |
| Action cost and cancellation terms are known | Makes alternatives comparable | Quote excludes fees or reservation is refundable | Obtain a dated quote and model conditional charges |
| Margin represents avoided loss | Connects lost demand to incremental value | Substitution or variable cost structure differs | Validate with finance and use a range |
| Other orders are unaffected | Bounds the worked case | Shared dock or vehicle is displaced | Add impacted orders and opportunity costs |
| Customer accepts modeled receipt | Distinguishes physical arrival from service | Dock rejects unscheduled or partial delivery | Confirm appointment and acceptance criteria |
| The network is acyclic within horizon | Enables simple forward propagation | Rework or return loops occur | Unroll bounded attempts or use separate scenarios |
| No recovery failure occurs | Simplifies worked arithmetic | Vehicle breaks down or slot is withdrawn | Add conditional failure outcomes and residual loss |
| Financial horizon captures relevant effects | Avoids omitting downstream costs | Repeat purchase, goodwill, or penalties arise later | Extend horizon or report unquantified effects separately |

The worked case is intentionally small enough to calculate by hand.
Real networks can have many orders, shared resources, uncertain durations, and changing action availability.
The simple forward pass does not solve resource-constrained scheduling or optimize a network of competing interventions.
When those features matter, use an appropriate scheduling, simulation, or optimization approach with validated inputs and human ownership.
More model complexity does not compensate for missing appointment confirmations or wrong order mappings.

## 12. Exercises

### Exercise 1 — Draw the causal chain

**Time:** 15 minutes.
A component truck is forecast 30 minutes late.
A factory unloads it, performs a quality release, runs a production batch, and ships finished goods to a customer.
A second component must also be available before production starts.

**Deliverable:** Draw a graph from both components to customer receipt and label each join.
**Acceptance criteria:** Show the two required predecessors, the release and production milestones, and at least one resource or calendar constraint.
Identify one fact that is observed and one that must be forecast.
Do not claim the customer's order is late without comparing the result with its commitment.

### Exercise 2 — Compute slack at a join

**Time:** 15 minutes.
Component A is ready at 09:20 and component B at 10:05.
Production can start when both are ready and the line is available at 09:45.
Production takes 50 minutes of elapsed time.

**Deliverable:** Calculate the earliest production finish and the effect of delaying A by 30 minutes.
**Acceptance criteria:** Show the maximum operation, the finish timestamp, and whether B remains binding.
State all timestamps in the same local time zone.

### Exercise 3 — Evaluate a missed connection

**Time:** 20 minutes.
The baseline load is ready at 11:35 for a 12:00 departure with an 11:40 inclusive cutoff.
A disruption moves readiness to 11:41.
The next confirmed departure is 16:00 and both departures have the same 90-minute transit time.

**Deliverable:** Compute baseline and disrupted receipt times and their difference.
**Acceptance criteria:** Apply the inclusive cutoff and show why the six-minute readiness change creates a larger receipt delay.
Do not assume any customer commitment not stated in the question.

### Exercise 4 — Classify operational statements

**Time:** 15 minutes.
A carrier says “arrival most likely 15:30”; a site books a 16:00 appointment; a manager sets a 98% monthly service goal; a customer accepts delivery by 17:00.

**Deliverable:** Classify each statement as forecast, plan or booking, target, or commitment.
**Acceptance criteria:** Explain why the forecast is not itself a customer promise and why the target is not an individual deadline.
Identify one additional event needed to establish actual receipt.

### Exercise 5 — Quantify service impact

**Time:** 20 minutes.
A store has 18 unreserved saleable units.
Before the next receipt it expects 67 units of demand, of which 70% of unserved demand is assumed permanently lost and the rest backordered.
Relevant contribution margin is USD 8 per lost unit.
Assume fractional expected units are acceptable for planning and no other costs.

**Deliverable:** Calculate unserved demand, expected lost units, expected backordered units, and expected lost contribution.
**Acceptance criteria:** Show the arithmetic and do not treat backordered units as permanently lost.
Explain why this expected-value calculation is not an observed count of transactions.

### Exercise 6 — Compare a mitigation

**Time:** 20 minutes.
Without action, a late scenario has probability 0.40 and costs USD 900; the other scenario costs USD 0.
A nonrefundable USD 250 action eliminates the late loss if successful.
Assume certain success and no other consequences.

**Deliverable:** Compute expected cost for both options, choose one, and find the break-even probability.
**Acceptance criteria:** Include the action cost in both scenarios, show units, and identify the assumptions under which the answer changes.

### Exercise 7 — Add residual risk

**Time:** 25 minutes.
Use exercise 6, but the action succeeds with probability 0.80 conditional on the late scenario.
When it fails, the full USD 900 loss remains; the USD 250 action cost is always incurred.

**Deliverable:** Calculate expected cost with action and compare it with no action.
**Acceptance criteria:** Use the conditional failure probability correctly and identify the residual risk.
Explain whether the action remains attractive under the given assumptions.

### Exercise 8 — Critique the evidence

**Time:** 20 minutes.
A dashboard reports a truck late because its last location message was ingested at 10:10, although the message says the gate event happened at 09:00.
It assigns all 200 shipment units to each of two orders and labels USD 10,000 of revenue as lost profit.
The receiving slot has not been confirmed.

**Deliverable:** List at least four distinct defects and propose a correction for each.
**Acceptance criteria:** Address timestamps, quantity allocation, economics, and slot status.
State what remains unknown rather than assuming the shipment is on time or late.

### Exercise 9 — Design an operational decision record

**Time:** 25 minutes.
A high-priority order may miss an outbound cutoff in 45 minutes.
Two carriers offer recovery capacity, but neither has confirmed a booking.
The planner can authorize USD 300; a larger cost requires a manager.

**Deliverable:** Draft a decision record and escalation flow.
**Acceptance criteria:** Include the decision deadline, evidence time, alternatives and capacity status, owner and spending authority, customer communication, and closure condition.
Distinguish requested from accepted actions and explain the fallback if neither carrier confirms.

## 13. Self-check answers

### Exercise 1

A valid graph has component A and component B flowing into an all-required join.
The join then flows through quality release, production, outbound shipment, and customer receipt.
Depending on the actual process, quality release may apply to each incoming component before the join rather than after it.
A line-availability calendar can delay production even when both components are present.
A confirmed gate event is an observation; a future completion or receipt is a forecast.
The late truck alone does not establish a late customer receipt because slack and other constraints have not been evaluated.

### Exercise 2

The start is `max(09:20, 10:05, 09:45) = 10:05`.
Adding 50 minutes gives a finish of 10:55.
A 30-minute delay moves component A to 09:50.
The revised start remains `max(09:50, 10:05, 09:45) = 10:05`.
The finish remains 10:55, and component B remains binding.
The first 45 minutes of A's lateness are absorbed with respect to this start, assuming all other inputs remain fixed.

### Exercise 3

Baseline readiness at 11:35 meets the inclusive 11:40 cutoff.
The 12:00 departure arrives at 13:30 after 90 minutes.
Disrupted readiness at 11:41 misses the cutoff by one minute.
The 16:00 departure arrives at 17:30.
Receipt is therefore four hours later even though readiness changed by six minutes.
No conclusion about customer commitment compliance is possible without the promised receipt deadline.

### Exercise 4

“Most likely 15:30” is a forecast, not an observed arrival or accepted promise.
A 16:00 appointment is a booked operational plan, subject to its terms and capacity.
The 98% monthly goal is a target measured over an aggregate population.
Delivery by 17:00 accepted by the customer is a commitment in the stated example.
A verified proof-of-delivery or other agreed receipt event is needed to establish actual receipt.
The operational appointment and customer commitment can differ without contradiction.

### Exercise 5

Unserved demand is `67 - 18 = 49` units.
Expected permanently lost demand is `49 × 0.70 = 34.3` units.
Expected backordered demand is `49 × 0.30 = 14.7` units.
Expected lost contribution is `34.3 × USD 8 = USD 274.40`.
Fractional units describe an expectation across possible outcomes, not a literal fraction of an observed transaction.
The answer excludes any incremental backorder fulfillment cost because none was supplied.

### Exercise 6

No-action expected cost is `0.40 × USD 900 = USD 360`.
The action costs USD 250 regardless of the scenario and eliminates the modeled late loss.
Expected cost with action is USD 250; the expected advantage is USD 110.
The break-even probability is `USD 250 / USD 900 ≈ 0.278`, or 27.8%.
The result assumes full success, no displacement, and no additional losses or policy constraints.

### Exercise 7

Conditional failure in the late scenario is `1 - 0.80 = 0.20`.
The joint probability of late and action failure is `0.40 × 0.20 = 0.08`.
Expected residual loss is `0.08 × USD 900 = USD 72`.
Adding the nonrefundable action cost gives `USD 250 + USD 72 = USD 322`.
The action remains USD 38 better than the USD 360 no-action expectation.
The residual risk is that the late scenario occurs and the action fails; it is not zero just because recovery was purchased.

### Exercise 8

The message's 09:00 event time and 10:10 ingestion time describe different facts.
Verify event semantics and source reliability before deciding whether the truck was late.
Reconcile the 200 units to the two orders instead of attributing all 200 units to both.
Use incremental contribution or another justified cost measure rather than relabeling revenue as profit.
Treat the receiving slot as unconfirmed and seek an actual booking.
The order-level impact, true available stock, customer commitment, and demand consequence remain unknown.

### Exercise 9 rubric

A strong record names the order, decision time, cutoff, and latest time to book each alternative.
It records each carrier's quoted cost, quantity capacity, and unconfirmed status.
It gives the planner's USD 300 authority and the manager escalation for a higher spend.
It identifies who will request and confirm execution and when to reassess.
It provides a fallback impact and customer notification if neither carrier accepts in time.
It closes only after receipt or another agreed outcome is confirmed and costs are recorded.
A weak record merely marks the alert red or assumes a carrier quote is an accepted booking.

## 14. Interview questions

1. **How can a ten-minute transport delay create a four-hour delivery delay?**
   A strong answer explains a missed discrete cutoff, the next feasible service, and capacity.
   A weak answer multiplies the ten minutes by an arbitrary factor.
2. **When does a late component have no effect on production?**
   A strong answer compares all required predecessors and available slack at the join.
   A weak answer assumes every component delay shifts production equally.
3. **How would you distinguish a late message from a late truck?**
   A strong answer separates event time, ingestion time, source semantics, and evidence quality.
   A weak answer treats the most recent timestamp as the physical arrival.
4. **How do you translate a missed slot into lost sales?**
   A strong answer traces receipt, time-indexed stock and demand, substitution, and lost versus delayed demand.
   A weak answer equates all delayed shipment units with lost sales.
5. **How do you compare a nonrefundable expedite with inaction?**
   A strong answer compares incremental scenario losses, action cost in every applicable scenario, and residual risk.
   A weak answer compares freight cost alone with total order revenue.
6. **What changes if two recovery actions need the same dock?**
   A strong answer models shared capacity, displaced orders, and reservation status.
   A weak answer evaluates both as independently feasible.
7. **What does an ETA confidence interval not tell you?**
   A strong answer notes that it is not a slot booking, customer promise, or automatic loss distribution.
   A weak answer treats the interval upper bound as an accepted commitment.
8. **How would you evaluate whether an expedite actually saved money?**
   A strong answer requests actual cost and a credible no-action counterfactual, acknowledging uncertainty.
   A weak answer calls the entire delivered revenue a saving.
9. **What should happen when required data are missing near an action deadline?**
   A strong answer distinguishes unknown from zero, uses bounded scenarios, and escalates to an authorized owner.
   A weak answer silently assumes the most favorable value.
10. **What is the difference between a forecast, target, plan, and commitment?**
    A strong answer defines each and preserves their versions and decision rights.
    A weak answer treats any revised ETA as a customer-approved promise.

## 15. Key terms

- **Action deadline:** latest time a recovery can still be committed; see [mechanics](#step-1--define-the-decision-and-the-time-of-decision).
- **Allocation:** assignment of available quantity to demand; see [affected quantities](#step-3--establish-the-affected-objects-and-quantities).
- **Backorder:** retained demand fulfilled later; see [service outcomes](#step-8--translate-times-into-service-outcomes).
- **Baseline:** pre-disruption reference plan; see [baseline](#step-2--fix-the-baseline-and-service-boundary).
- **Booking:** accepted reservation for a slot or capacity; see [appointment rules](#step-6--apply-appointment-and-departure-rules).
- **Commitment:** accepted promise with defined conditions; see [mental model](#3-the-core-mental-model).
- **Contribution margin:** relevant revenue less relevant variable cost; see [economics](#step-10--compare-incremental-economics).
- **Counterfactual:** outcome under another action in the same scenario; see [learning](#step-12--learn-from-outcome-and-counterfactual).
- **Critical path:** binding longest path under current assumptions; see [continuous baseline](#step-5--calculate-a-continuous-time-baseline).
- **Cutoff:** latest qualifying handoff time; see [missed connection](#exercise-3--evaluate-a-missed-connection).
- **Decision owner:** person authorized to choose a response; see [execution](#step-11--decide-communicate-and-execute).
- **Dependency:** condition that must hold before a successor can proceed; see [graph](#step-4--draw-the-dependency-graph).
- **Dwell:** elapsed time at a location; see [vocabulary](#4-definitions-and-vocabulary).
- **ETA:** estimated time of arrival, a forecast; see [data context](#8-data-standards-and-regulatory-context).
- **Event time:** when an event occurred; see [data context](#8-data-standards-and-regulatory-context).
- **Expected loss:** probability-weighted modeled loss; see [economics](#step-10--compare-incremental-economics).
- **Forecast:** estimate of a future or uncertain milestone; see [forecast scenarios](#forecast-scenarios).
- **Incremental cost:** change attributable to an alternative; see [economics](#step-10--compare-incremental-economics).
- **Join:** point requiring multiple predecessor conditions; see [exercise 2](#exercise-2--compute-slack-at-a-join).
- **Lost sale:** demand not ultimately converted because service failed; see [service impact](#service-and-cost-without-recovery).
- **Milestone:** defined point in an operating process; see [vocabulary](#4-definitions-and-vocabulary).
- **Plan:** intended schedule or operating arrangement; see [baseline](#step-2--fix-the-baseline-and-service-boundary).
- **Processing time:** time an event is ingested or processed; see [failure modes](#10-failure-modes-and-red-flags).
- **Recovery:** feasible intervention that changes a path or service outcome; see [alternatives](#step-9--build-feasible-mitigation-alternatives).
- **Residual risk:** exposure remaining after action; see [exercise 7](#exercise-7--add-residual-risk).
- **Reslot:** newly accepted appointment after a missed slot; see [appointment rules](#step-6--apply-appointment-and-departure-rules).
- **Scenario:** coherent combination of uncertain assumptions; see [forecast scenarios](#forecast-scenarios).
- **Slack:** delay absorbable without changing a specified downstream milestone; see [worked propagation](#baseline-and-late-propagation).
- **Slot:** bounded appointment or capacity window; see [appointment rules](#step-6--apply-appointment-and-departure-rules).
- **Stockout:** demand exceeding usable stock at a time and place; see [service outcomes](#step-8--translate-times-into-service-outcomes).
- **Target:** desired aggregate or individual performance level; see [mental model](#3-the-core-mental-model).

## 16. Sources

1. GS1, [EPCIS Standard](https://www.gs1.org/standards/epcis), including EPCIS 2.0 resources. Standards-body source for event-sharing semantics; it does not specify a lost-sale model.
2. GS1, [Global Location Number](https://www.gs1.org/standards/id-keys/gln). Standards-body description of location and party identification.
3. European Union, [Regulation (EU) 2016/679 (General Data Protection Regulation)](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5 and 6. Binding where applicable; assess scope and obligations for the particular processing context.
4. Toyota Motor Corporation, [Japan Earthquake and Production](https://global.toyota/en/detail/246346), April 8, 2011. Company announcement; use for documented production context, not independent proof of savings.
5. Maersk, [Customer advisories](https://www.maersk.com/news/articles). Company operational publications; individual advisory applicability depends on date, geography, and service.
6. Hau L. Lee, V. Padmanabhan, and Seungjin Whang, “Information Distortion in a Supply Chain: The Bullwhip Effect,” *Management Science* 43(4), 1997, [doi:10.1287/mnsc.43.4.546](https://doi.org/10.1287/mnsc.43.4.546). Peer-reviewed context on information distortion, not evidence for the specific worked case.
7. David Simchi-Levi, William Schmidt, and Yehua Wei, “From Superstorms to Factory Fires: Managing Unpredictable Supply-Chain Disruptions,” *Harvard Business Review*, 2014, [article](https://hbr.org/2014/01/from-superstorms-to-factory-fires-managing-unpredictable-supply-chain-disruptions). Practitioner discussion of supply-chain exposure and recovery.

Accessed references are provided for independent review; company publications describe their own operations and are not independent impact evaluations.
The numerical example, probabilities, costs, calendars, and demand assumptions in this lesson are illustrative and are not attributed to those sources.

## 17. Links

- [Module 19 index](./_index.md).
- [The Five Visibility Layers](./01_visibility-layers.md).
- [Event Streams, IoT, Telematics, and EPCIS 2.0](./02_event-streams-epcis.md).
- [Module 19 survey and its lab sections](../../../course/19_control_towers_visibility_events.md).
