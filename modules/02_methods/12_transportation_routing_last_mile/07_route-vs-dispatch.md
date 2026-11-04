# Route Optimisation versus Dispatch

> Module 12 — Transportation, Routing, and Last-Mile Analytics  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: vehicle-routing and freight-cost basics  
> Version 2026.1

## 1. Why this topic exists

A route plan assigns work before the day unfolds; dispatch controls work while the day is unfolding.
The distinction matters when a vehicle breaks down, a customer refuses a delivery, or a driver reports an inaccessible entrance.
A mathematically attractive plan can be operationally unsafe or commercially damaging if no one owns the decisions after release.
Conversely, continuous local improvisation can destroy a good plan and conceal recurring data problems.

The transport manager needs a repeatable operating process that joins optimization, driver communication, customer promises, and exception recovery.
The decision is not whether humans or software should win.
It is which decisions can be automated within approved guardrails and which require a named accountable person.
This topic follows a fictional delivery day from planning through closeout, with measurable service and cost consequences.

## 2. Learning outcomes

You should be able to:

1. Distinguish a route plan, dispatch instruction, estimated arrival, and customer commitment.
2. Describe the information and authority transfers from planning to dispatch and back.
3. Classify exceptions by severity and select a safe response.
4. Recalculate a live recovery decision using incremental costs and service consequences.
5. Define operational KPIs with explicit event timestamps and denominators.
6. Design controls for reoptimization, overrides, and customer communication.
7. Diagnose misleading performance claims caused by missing or biased event data.

Exercises 1–7 map to these outcomes.

## 3. The core mental model

```text
Forecast orders -> create plan -> review and release -> execute stops
                        ^                 |                  |
                        |                 v                  v
                 data corrections    driver accepts     events + exceptions
                        ^                                    |
                        |                                    v
                   closeout <- compare actuals <- dispatch decisions
                                               |
                                               v
                                  notify customers and update ETA
```

A plan is a versioned allocation and sequence based on a snapshot.
A dispatch instruction is the authorized work issued to a particular driver or carrier.
An estimated time of arrival (ETA) is a forecast that changes with observations.
A commitment is a communicated obligation under the applicable customer agreement.
These objects may have different owners, timestamps, and legal consequences.
An ETA update does not automatically amend a delivery commitment.

A dispatcher observes the current state, checks available choices, selects and records an action, and confirms execution.
The optimizer can propose a revised plan, but a proposal is not an instruction until the designated release control approves it.
Driver safety, road restrictions, and working-time requirements are hard gates, not costs to trade away for a lower route score.
Feedback closes the loop: recurring gate delays should change future service-time assumptions, not merely generate daily overrides.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Route plan | Versioned vehicle assignments, sequence, and schedule | Treating a draft as released work |
| Dispatch | Operational control of assigned work and exceptions | Calling a solver result dispatch |
| Release | Authorization of a particular plan version | Assuming driver receipt equals acceptance |
| Stop | One planned service location or task | Assuming one stop always equals one order |
| Attempt | A visit or delivery attempt under a defined event rule | Counting a planned stop as an attempt |
| Proof of delivery | Evidence that an agreed delivery event occurred | Treating a GPS ping as proof of receipt |
| ETA | Forecast arrival time with a timestamp | Treating it as a contractual promise |
| Commitment | Agreed service window or deadline | Replacing it silently with a forecast |
| Exception | Material divergence requiring an operational response | Logging a known planned delay as unexpected |
| Reoptimization | Recalculation using current state and restrictions | Moving completed stops or unbreakable commitments |
| Override | Authorized departure from a recommended plan | Unrecorded driver improvisation |
| Frozen prefix | Work already completed or underway that cannot be freely reassigned | Replanning from the original depot |
| Dwell | Time spent at a location, minutes | Equating all dwell to customer delay |
| Incremental cost | Additional avoidable cost relative to a feasible alternative, currency | Recounting sunk or common cost |
| On-time rate | Eligible on-time outcomes divided by eligible outcomes | Excluding failed deliveries without disclosure |
| Version | Immutable identifier for a plan or event revision | Overwriting historical plans |

## 5. The mechanics, step by step

### 5.1 Define the decision rights

The planner owns the proposed route and its planning assumptions.
The dispatch lead owns the release decision and exceptions during execution.
The driver owns safe vehicle operation and must be able to refuse an unsafe instruction.
Customer service owns external notification under an agreed escalation matrix.
The transport manager owns policy, commercial trade-offs, and recurring-performance review.
These are example roles; actual organizations should assign named backups and document their authority.

Decision rights include thresholds: when may a dispatcher add paid overtime, hire rescue capacity, resequence stops, or change a customer window?
A recommended route cannot grant authority to violate a safety rule, contract, or operating permit.
Predefine who can approve a new commitment and whether the customer must consent.

### 5.2 Build a coherent planning snapshot

Freeze order, fleet, driver, location, service-time, and traffic inputs at a recorded cutoff.
Distinguish confirmed orders from forecasts and identify late orders separately.
Check customer windows, vehicle capacity, driver qualifications, site access, and depot readiness.
Identify unassigned orders explicitly; do not hide them in a cost total.
Record data source, timestamp, units, and owner for each critical field.
A plan made from stale inputs can be feasible in the model but impossible in the operation.

### 5.3 Review the proposed plan

Check that all mandatory orders appear exactly once unless split delivery is permitted.
Check capacity, route duration, time windows, vehicle compatibility, and required breaks.
Compare predicted service to customer commitments, not merely to updated ETAs.
Use an independent schedule or constraint audit to catch model and interface errors.
Identify fragile routes: little window slack, high traffic sensitivity, or a customer with an uncertain receiving time.
Record the model version, input snapshot, objective, run status, and proposed plan identifier.
A time-limited optimizer returning a plan does not establish global optimality.

### 5.4 Release and acknowledge

Assign the approved version to a vehicle and driver or carrier.
Send stops, sequence, access details, special handling requirements, contact information, and escalation instructions.
Confirm the driver or carrier received and accepted the instruction.
If the driver rejects the work or the vehicle is unavailable, return it to dispatch rather than marking it in execution.
Maintain an immutable release timestamp and correlate subsequent events with that version.
Protect driver and customer contact information according to organizational policy and applicable law.

### 5.5 Observe events, not just positions

Capture departure, arrival, service start, service completion, attempted delivery, proof of delivery, and return.
Record event occurrence time separately from receipt time; mobile devices may upload late.
Record source and confidence: a driver-entered timestamp, a geofence inference, and a customer signature support different claims.
Position alone does not establish that goods were delivered or accepted.
Use idempotent event identifiers so repeated uploads do not count as multiple deliveries.
Preserve corrections and their reason instead of silently replacing historical facts.

### 5.6 Triage exceptions

First protect life and safety: stop work and use emergency procedures where appropriate.
Then identify the affected order, vehicle, driver, current location, time, and remaining commitments.
Classify the exception as vehicle, driver, traffic, site, customer, order, data, or external disruption.
Estimate the time at which action becomes irreversible, such as a receiving dock closing.
Select an owner and escalation deadline; a queue without an owner is not control.

Examples of different responses:

- A short queue may be absorbed within existing schedule slack while monitoring the next customer.
- A road closure may justify a safe detour and ETA update without changing customer commitments.
- A failed vehicle may require rescue capacity, transfer controls, and customer notifications.
- A customer refusal may require documented disposition and an authorized reattempt rather than an automatic second visit.
- A safety or legal restriction stops the affected instruction regardless of its economic score.

### 5.7 Evaluate recovery options

Start from the vehicle's actual state, not the original planned depot departure.
Freeze completed stops, active service, cargo already onboard, and instructions that cannot safely be withdrawn.
For each option, calculate incremental avoidable cost from the decision time onward.
Include external carrier charges, additional paid time, reattempts, transfer labor, and customer credits where contractually applicable.
Separate certain costs from uncertain forecasts and state the scenario behind each estimate.
Check feasibility and service impact before ranking options by cost.
If no option meets every commitment, escalate the trade-off and obtain authority to notify or renegotiate.

### 5.8 Replan and communicate

A reoptimization run needs live positions, completed and remaining orders, onboard cargo, vehicle availability, and remaining working time.
Represent drivers already committed to a stop and customers whose sequence cannot change.
Compare the proposed revision with the released plan, including which driver and customer instructions change.
Issue a new version only after its owner approves it.
Acknowledge receipt and avoid sending conflicting instructions to the same driver.
Update ETA forecasts with their prediction time and uncertainty; do not silently alter the customer commitment.
Escalate when the forecast misses a commitment rather than waiting until the miss occurs.

### 5.9 Close the day and learn

Reconcile planned orders against delivered, failed, canceled, and unresolved outcomes.
Match carrier invoices, accessorials, driver paid time, reattempts, and customer credits to routes where possible.
Record exceptions, root causes, and override decisions with sufficient detail for review.
Compare the initial plan, every approved revision, and actual events.
Trend systemic issues such as inaccurate service durations or recurring access failures separately from one-off events.
Feed verified corrections back to planning master data with an owner and effective date.

### 5.10 Establish a state machine for each order

At release, an order is assigned but not yet attempted.
A driver acknowledgment confirms an instruction, not a delivery.
An arrival event can move the order into an at-site state if the event passes the operation's location and timestamp rules.
Service can then result in delivery, refusal, partial delivery, or another defined exception.
Do not mark an order delivered merely because the vehicle departed the site.
Where a partial delivery is allowed, retain the quantity delivered and the quantity outstanding.
An unsuccessful attempt needs a disposition: reattempt, return, customer cancellation, or investigation.
The disposition must not overwrite the evidence of the first attempt.
Each transition should identify the actor, occurrence time, source, and any correction.
If an event arrives out of order, preserve its occurrence time and receipt time and reconcile the state.
An automated state transition should never erase a more authoritative correction.

### 5.11 Freeze what cannot be changed

Completed work is historical and cannot be assigned again.
A stop being served is normally frozen until the driver confirms whether service can safely be interrupted.
Cargo already loaded is not freely available to another route without a physical transfer.
A vehicle's present position and remaining legal duty replace its original depot and full-shift assumptions.
Customer appointments, security access, and equipment constraints may further restrict the remaining sequence.
Mark each restriction with a reason and an owner rather than using one unexplained locked flag.
If a restriction is negotiable, record the person authorized to release it and the evidence of that decision.
An optimizer may propose a swap that is mathematically feasible but requires a loading-dock transfer not represented in its data.
Dispatch must either validate that transfer or reject the swap.

### 5.12 Set a decision deadline before comparing options

For the noon customer in the example, work backward from the latest acceptable delivery event.
If a compatible rescue vehicle needs 25 minutes to reach the broken truck, allow that travel time.
If a safe transfer requires 15 minutes and the drive to the customer requires 30 minutes, allow another 45 minutes.
If service must start by 12:00 and these estimates exclude service, a departure toward the broken truck after 10:50 risks missing the start deadline.
The arithmetic is 12:00 minus 25, 15, and 30 minutes, or 10:50.
This is a deterministic latest-departure estimate, not a guarantee.
Any carrier acceptance delay, congestion, or additional transfer requirement moves the practical decision deadline earlier.
If the commitment is completion by noon, subtract service duration too.
The dispatcher should state both the assumptions and the contingency buffer when requesting approval.
When the deadline passes, update the feasible option set rather than continuing to display an obsolete recommendation.

### 5.13 Distinguish notification, consent, and evidence

A message saying that a shipment may be late is a notification.
A customer's reply agreeing to a revised window may be consent if the person is authorized and the applicable agreement allows it.
Neither message alone proves the goods were delivered.
Keep the original window visible alongside any approved amendment.
Record who proposed the amendment, who accepted it, the effective time, and the specific orders affected.
If no acceptance is received, a revised ETA remains a forecast rather than a revised commitment.
If the receiver says the dock will stay open but cannot amend the commercial agreement, escalate the distinction.
When a delivery does occur, retain the appropriate proof under the contract and data policy.
Avoid sending a revised ETA as though it were a guaranteed appointment when the recovery vehicle is not yet accepted.

### 5.14 Compare operational choices on a common boundary

Define the decision time and list costs already incurred before it.
Those sunk costs usually do not distinguish the options.
List future costs common to every feasible option separately so that they do not distort the comparison.
For each remaining option, estimate additional carrier charges, handling, paid time, and contractually applicable credits.
Check whether a carrier quote already includes the driver, fuel, or handling before adding those items again.
For a proposed internal rescue, include the work displaced from that resource and its downstream consequences.
Separate a certain carrier quote from an uncertain overtime estimate and a conditional customer credit.
Record the conditions under which the credit applies rather than automatically adding it to every late delivery.
Reject options that fail non-negotiable safety or service gates before ranking the remaining options by cost.
If all options fail a gate, escalate the conflict and document the authorized service recovery decision.

### 5.15 Govern human overrides

Give the dispatcher the proposed action, the inputs that drove it, the affected orders, and its predicted consequences.
Offer structured reasons such as inaccurate location, inaccessible entrance, vehicle incompatibility, customer request, and safety concern.
Allow a free-text explanation where the taxonomy is insufficient, subject to privacy policy.
Record the plan version, actor, decision time, reason, and resulting instruction.
An override is a source of evidence, not automatically a mistake by either the dispatcher or the optimizer.
Review repeated overrides by site and reason after checking whether the underlying data were accurate.
For example, repeated access failures at one customer may justify updating the approved entrance and service duration.
Do not train a future model on overrides without reviewing whether they were feasible and whether their outcomes were observed.
Monitor whether the interface discourages necessary safety overrides or encourages unrecorded workarounds.

### 5.16 Close the information loop

At end of day, reconcile each released order to a final outcome or an explicitly owned unresolved state.
Match route versions to the instructions actually acknowledged by drivers.
Compare predicted and observed service starts using the forecast that was available before the event.
Group deviations by causes that can lead to action: data, capacity, traffic, site access, customer, and equipment.
Validate a proposed master-data correction with the data owner and record its effective date.
If the evidence is insufficient, retain the issue for investigation instead of changing a parameter from one observation.
Review costs with the same scope as the plan, including recovery invoices and any applicable credits.
Assign a deadline and owner to material open issues.
This closes the operational learning loop without pretending that one disrupted day proves a fleet-wide causal effect.

### 5.17 Operate from a shared decision clock

The relevant clock begins when a material event is detected.
Detection can occur after the disruption itself.
A driver may lose connectivity before reporting a breakdown.
Record the reported occurrence time, receipt time, and dispatch detection time.
If any time is uncertain, keep its source and uncertainty visible.
Do not backdate the dispatcher's decision to the occurrence time.
The operational response interval starts from a defined, observable event.

For each exception, record the next irreversible deadline.
A receiving closure is one example.
A temperature excursion, a driver duty limit, or a carrier quote expiry may be earlier.
The earliest relevant deadline controls the escalation clock.
Work backward through travel, handling, approvals, and communication.
Include a contingency buffer where the underlying estimates are uncertain.
The buffer is a policy choice and not a guarantee.
Update the deadline when new information changes the feasible options.

Use distinct statuses for detected, acknowledged, decision pending, and resolved.
Assign one accountable owner at each stage.
If the owner changes shifts, explicitly transfer both the case and its deadline.
An escalation should carry the affected orders, current state, and decision requested.
It should not require the next person to reconstruct the case from a chat thread.
Record the time an escalation was received and accepted.
If the designated person cannot respond, follow the documented backup path.

### 5.18 Establish a live operational picture

The display should distinguish facts, forecasts, and unconfirmed reports.
The last reliable vehicle position is a fact about a past instant.
The present position inferred from it is an estimate.
An ETA uses that estimate and a travel-time forecast.
A driver report can override a stale telemetry view without erasing the old event.
Keep the observation time alongside the value.
Show how old the last known position is at the decision time.

For each affected vehicle, show the driver, vehicle, current assignment, and duty information.
For each affected order, show its custody, remaining quantity, and current commitment.
For each route, show completed stops, active work, and remaining sequence.
Also show access restrictions, special handling, and known vehicle compatibility.
The view should expose unresolved discrepancies instead of silently choosing a source.
If a package count differs from the manifest, the route cannot assume the disputed item is onboard.
Send the discrepancy to the designated inventory or custody owner.
Refresh the decision picture before issuing a revised instruction.

### 5.19 Classify exceptions by consequence and urgency

The category of an exception identifies the likely response owner.
Severity identifies what may be harmed.
Urgency identifies how quickly the available choices will disappear.
Do not treat these three labels as interchangeable.
A minor access question can become urgent when a dock is about to close.
A serious cargo issue may require a stop even when delivery is not due for hours.

| Example | First gate | Owner to involve | Decision time consideration |
|---|---|---|---|
| Driver injury | Emergency and safety response | Dispatch lead and emergency services as applicable | Immediate |
| Vehicle failure | Safe location and cargo condition | Fleet and dispatch | Before rescue or commitment deadline |
| Closed gate | Verify site access and driver safety | Site contact and dispatch | Before receiving closes |
| Missing handling unit | Preserve custody evidence | Inventory or claims owner | Before transfer or departure |
| Traffic delay | Validate route and updated ETA | Dispatcher | Before a commitment becomes unrecoverable |
| Customer refusal | Record reason and protect goods | Customer service and dispatch | Before disposition or reattempt |
| Duplicate delivery event | Preserve event history | Data operations | Before KPI cutoff and invoicing |
| Temperature alarm | Protect product under approved policy | Quality and dispatch | According to product-specific limits |

Severity criteria must be agreed locally by operational and safety owners.
This table is a teaching taxonomy, not an emergency procedure.
A dispatcher can escalate more than one consequence from the same incident.
For example, a breakdown may create safety, product-quality, and customer-service cases.
Link them to one incident ID while preserving separate decisions and owners.

### 5.20 Form the feasible recovery set

Start with the orders and physical goods that remain at the decision time.
Identify which goods are on each vehicle and whether transfer is permitted.
Check the location and availability of potential rescue resources.
Confirm the vehicle type, equipment, driver qualifications, and remaining duty.
Check safe handling, custody, access, and all affected customer commitments.
An option is not feasible merely because its estimated cost is low.

For a proposed internal rescue, identify work displaced from that vehicle.
Recalculate its own remaining stops as well as the rescued orders.
If it must visit a transfer point, include that travel and handling.
Do not assume that cargo can teleport from one vehicle to another.
Check the combined capacity throughout the planned sequence.
The order of unloading and loading may matter physically.
If the transfer requires special equipment, verify that it is present.
If the receiving customer must consent to a new window, mark that option conditional.

For an external carrier, verify acceptance rather than relying on a directory listing.
Confirm the quote's scope, expiry, vehicle, arrival estimate, and responsibility for transfer.
Check whether the carrier is approved for the cargo and location.
Do not represent a quote as a binding acceptance unless the carrier actually accepted.
If an option depends on an unconfirmed input, show it as conditional.
Keep at least one fallback visible when the lead option can fail before acceptance.
Escalate when no option satisfies all hard requirements.

### 5.21 Rank feasible alternatives consistently

Once safety and mandatory rules are checked, compare the remaining alternatives.
State the decision time and the common service scope.
Separate costs already incurred from avoidable costs after the decision.
Include costs displaced to a later day when an order is deferred.
Exclude shared costs only if they are genuinely the same across alternatives.
Identify taxes, credits, and charges according to the chosen accounting boundary.
The lowest cost option is not automatically the best option if service differs.

List uncertain inputs separately from confirmed amounts.
A rescue quote may be fixed while additional handling time is estimated.
A customer credit may be conditional on a contract clause and a documented failure.
Avoid multiplying a speculative probability by a cost and presenting it as an invoice.
If expected values are used, disclose the probability source and alternatives.
When probabilities are unavailable, provide named scenarios and consequences.
For a high-consequence hard requirement, keep the gate separate from the monetary ranking.

Consider the value of waiting for more information.
An updated repair estimate may arrive in 15 minutes.
Waiting can avoid a rescue charge if repair is fast enough.
But waiting can also cause the rescue option to miss its latest departure.
Calculate the latest useful time for the information before postponing the decision.
If the next update comes after the deadline, it cannot preserve the original option.
Record why the dispatcher acted or waited with the information then available.

### 5.22 Govern a route revision as a transaction

A revision begins as a proposal against a named currently released version.
It should identify exactly which stops, drivers, and times would change.
Validate that its starting state is sufficiently current.
If a driver completes another stop during computation, the proposal may be stale.
Reconcile the change before issuing the instruction.
Do not overwrite a completed stop to make the revised sequence look consistent.

The approver checks hard gates, delegated spend authority, and affected commitments.
Approval identifies the proposal, approver, time, and scope.
Issuance transmits an immutable instruction version to the affected parties.
Each driver acknowledges the version relevant to their work.
Until acknowledgment, dispatch must manage the possibility that the old version is being followed.
If a driver cannot be reached, use the defined fallback communication channel.
Avoid sending two active conflicting sequences for the same vehicle.

An acknowledgment is not execution.
After acknowledgment, observe whether transfer and subsequent service actually occur.
If the driver rejects the revision, record the reason and return the work to dispatch.
If a later revision supersedes it, retain both versions and their effective times.
Link each operational event to the instruction believed to be active at that moment.
Allow reconciliation when an offline device uploads events after a newer release.
The historical record should explain what each participant knew and when.

### 5.23 Reconcile physical custody before reassignment

An order identifier represents a commercial obligation.
A handling-unit identifier represents a physical item or package.
One order may contain several handling units.
One stop may contain several orders.
Changing the route assignment does not itself transfer physical custody.
The receiving driver needs the goods and the information needed to handle them.

Create a transfer manifest with order IDs, handling-unit IDs, and quantities.
Record the sending and receiving parties, location, and transfer time.
Count or scan each unit according to the operation's policy.
Record visible damage and relevant product-condition evidence.
Investigate a discrepancy before marking the transfer complete.
If a partial transfer is authorized, identify exactly what moved and what remains.
Ensure the revised route reflects the actual cargo, not the intended cargo.

For temperature-controlled goods, consult the product-specific procedure.
The allowable exposure and evidence requirements depend on the product and agreement.
A route optimizer cannot infer product fitness from a transfer timestamp alone.
If quality has not authorized release, the delivery decision may need to wait.
An operational decision to proceed does not erase a custody discrepancy.
Keep customer proof of delivery separate from inter-driver transfer evidence.

### 5.24 Manage customer communication as a controlled workflow

Identify the audience and purpose before sending a message.
The driver needs an instruction and an escalation contact.
The receiver needs a realistic ETA and information necessary for site access.
The commercial owner may need to approve a proposed amendment.
The customer service representative needs the affected orders and message history.
Sending the same generic text to all parties can conceal critical differences.

Use the current forecast and its update time.
State material uncertainty if the rescue resource has not accepted.
Do not promise a time that has not been authorized.
If a window change is proposed, identify the existing commitment and requested change.
Record the customer contact, response, authority, and scope of any acceptance.
An unanswered message is not agreement.
An ETA update is not an amendment.
An amendment is not proof that delivery occurred.

If the forecast changes materially, notify the responsible customer contact again.
Choose a communication channel and escalation appropriate to the contract and urgency.
Avoid exposing unnecessary driver, customer, or cargo information.
Retain the message and delivery or response evidence under the organization's policy.
If the customer declines a proposed revision, escalate the remaining alternatives.
Do not silently rewrite the original commitment to make the performance report look better.

### 5.25 Define event quality before calculating metrics

An event has an identifier, subject, type, occurrence time, and source.
It also has a receipt time and, where appropriate, an instruction version.
Some events are directly observed by a person.
Others are inferred from a device, geofence, or integration.
A GPS position near a building is not equivalent to a signed delivery receipt.
Keep source confidence visible to downstream users.

Require a stable event ID for safe retries.
An exact duplicate upload should not create a second business event.
An update or correction must reference the event it revises.
Retain the original and the correction rather than silently replacing history.
For a late upload, recalculate the current state using occurrence time and applicable ordering rules.
Do not assume receipt order equals physical event order.
Conflicting evidence requires a resolution owner.
Record the resolution reason and who made it.

Check for impossible event sequences and clock errors.
Examples include completion before start or a future event from a misconfigured device.
An apparent negative dwell may also reflect a time-zone conversion problem.
Quarantine suspect records for review instead of discarding inconvenient failures.
Report missing and corrected event counts alongside published metrics.
Choose a late-data cutoff and a revision policy for dashboards.
The cutoff trades timeliness against completeness and must be disclosed.

### 5.26 Separate service measures by population

Start with all orders eligible under the published reporting rule.
Use the original commitment for the original-window on-time measure.
Include undelivered eligible orders in that denominator.
Publish cancellations and exclusions separately with their reasons.
If a commitment is amended, publish an additional amended-window measure.
Do not replace the original measure with the amended one.

First-attempt success uses orders with an eligible first attempt as its denominator.
It therefore says nothing about orders that never received an attempt.
Pair it with attempt coverage.
Same-day completion includes late deliveries made on the scheduled date.
It does not establish that delivery met a specific window.
For ETA accuracy, match predictions to actual events at a specified prediction horizon.
An ETA issued one minute before arrival is not comparable to one issued two hours earlier.
Report prediction coverage and the count of unmatched orders.

Segment metrics by relevant customer, geography, service level, and period.
Interpret differences cautiously when populations or order mixes differ.
For small samples, a single order can materially change the percentage.
Show numerator and denominator rather than only a rounded rate.
Compare each metric with a defined decision and action owner.
A metric without a response process cannot by itself improve dispatch.

### 5.27 Build a cost ledger that can be reconciled

Define the grain of each expense before allocating it.
A carrier rescue invoice may relate to an incident, route, or several orders.
Driver paid time may be recorded by shift rather than by stop.
Transfer labor may be recorded against a depot or a cost center.
A customer credit may be booked weeks after the delivery.
Keep source references and allocation rules with the calculation.

Separate forecast, authorized amount, and actual posted cost.
The accepted quote is not necessarily the final invoice.
The invoice is not necessarily a cash payment or a full cost-to-serve.
Reconcile differences and identify whether they are timing, scope, or price variances.
Do not allocate the same invoice to both the rescue route and the original route.
If an allocation is approximate, disclose the method and sensitivity.

Cost per delivered order can increase when fewer orders are delivered.
That increase can reflect a denominator change rather than higher route spend.
Pair it with eligible-order count, unserved count, and total cost.
Cost per scheduled order answers a different question.
The appropriate denominator depends on the decision, not on which result looks better.
Exclude revenue and margin claims unless those components are actually available.

### 5.28 Review overrides without discouraging safe action

An override reason should capture the operational fact driving the change.
Examples include an inaccessible entrance, damaged goods, or an unavailable vehicle.
A generic reason such as “other” should permit a concise explanation.
The reason field must not require unnecessary personal or medical details.
The decision record should identify the recommendation and its input version.
It should also identify the instruction actually released.

Review repeated overrides in context.
If an entrance is consistently inaccessible, the location master may be wrong.
If routes repeatedly need extra service time, estimate the distribution by relevant conditions.
If drivers reject a safe-looking instruction, investigate what the system cannot see.
Do not penalize a safety refusal simply because it reduces adherence.
Do not assume every override is beneficial without examining outcomes.

Use learning reviews to assign a corrective action and owner.
Examples include a verified site update, matrix refresh, or policy clarification.
Keep the evidence and effective date for each change.
Measure whether the change reduces the recurring issue on comparable work.
Separate training needs from defects in data or system design.
Escalate systemic hazards through the organization's safety process.

### 5.29 Prepare continuity when systems fail

Decide in advance how dispatch operates during a mobile, network, or planning outage.
Keep an approved way to contact drivers and customers.
Protect access to the latest released instructions and essential handling details.
Use a controlled manual log for decisions, times, and acknowledgments.
The log should distinguish a temporary instruction from a draft.
Assign a single owner for reconciling offline work after recovery.

When connectivity returns, ingest late events idempotently.
Compare the manual log with system versions before sending automated updates.
Resolve conflicting instructions with the drivers rather than guessing.
Preserve the original event occurrence time and the later upload time.
Reconcile every affected order to a current custody and service state.
Record any information that could not be recovered.
Review the outage separately from ordinary route-performance results.

### 5.30 Evaluate changes to dispatch practice

Define the intervention before measuring the outcome.
It might be a new release gate, ETA model, or exception queue.
Specify which sites and shifts use the intervention and when.
Keep customer commitments and reporting rules stable where possible.
Measure both service and economic outcomes.
Include workload, exception handling time, and unserved orders.

Where feasible, allocate comparable units randomly.
Otherwise compare similar areas and disclose differences.
Seasonality, traffic, order density, and staffing can confound comparisons.
A reduction in route mileage can coexist with increased rescue cost.
An apparent improvement in on-time rate can result from excluding unserved orders.
Report missing data and adherence to the intended process.
Do not generalize a single site's result to every operation without evidence.
Use a safety stop rule and a named rollback owner.

## 6. Worked example

### 6.1 Opening plan and commitments

A fictional local distributor releases two routes at 08:00.
It has 20 eligible deliveries, with ten assigned to each vehicle.
Each delivery has a customer-agreed window; one high-priority stop closes receiving at 12:00.
The plan predicts both vehicles will return by 15:00.
Budgeted paid route time is 14 driver-hours total, at USD 24 per paid hour.
Budgeted vehicle and fuel cost is USD 150 for the day across the two vehicles.
The baseline planned operating cost is therefore 14 × USD 24 + USD 150 = USD 486.
This baseline is a plan, not an actual cost or a forecast of customer credits.

### 6.2 Exception at 10:00

Vehicle A breaks down after completing four of its ten deliveries.
Vehicle B has completed five of its ten deliveries and still carries its remaining goods.
There are eleven undelivered orders: six on A and five on B.
The broken vehicle's goods require a controlled transfer before another vehicle can carry them.
A dispatcher confirms that everyone is safe and obtains an estimated repair time beyond the closing time of the priority customer.
The dispatcher must choose a recovery plan; simply recomputing the original routes from the depot would ignore goods and work already in progress.

The following alternatives are hypothetical estimates from 10:00 onward.
Their incremental costs exclude original costs common to all alternatives and are not total daily cost.
Service outcomes are forecasts, not guarantees.

| Option | Operational action | Incremental cost estimate | Forecast service outcome |
|---|---|---:|---|
| R: rescue | Hire a compatible rescue vehicle and transfer A's goods | USD 180 carrier + USD 30 transfer labor = USD 210 | Priority delivery before 12:00; all 11 remaining delivered today |
| B: use vehicle B | Add a transfer and resequence B after its existing work | USD 72 overtime + USD 30 transfer labor = USD 102 | Priority delivery forecast after close; all 11 delivered today only if receiver stays open |
| D: defer | Finish B's five and reattempt A's six tomorrow | USD 96 next-day reattempt + USD 60 estimated customer credit = USD 156 | Six orders deferred, including priority delivery |

The rescue carrier confirms vehicle compatibility and an available driver.
The site confirms that the goods can be transferred safely within the relevant time.
The USD 60 credit is an assumed contractual estimate; it is not an actual incurred charge.
If a customer credit is already included in the rescue quote or would occur under every option, remove it from the incremental comparison.

### 6.3 Choose and communicate

Option B is the cheapest forecast at USD 102 but relies on an unconfirmed receiver staying open.
If the noon commitment is mandatory and no extension is agreed, B is not an acceptable plan merely because it is cheaper.
Option D is USD 54 less than rescue but misses six commitments under the stated assumptions.
Choose R if the dispatcher is authorized to spend USD 210 and the transfer and driver checks pass.
Otherwise escalate the approval immediately; the cost threshold is a governance rule, not a solver output.
Notify the priority customer of a revised ETA and explicitly identify any remaining risk.
Record the rescue decision, expected cost, approval, carrier acceptance, and plan version.

### 6.4 Outcome and performance calculation

Suppose, separately from the forecast, that the rescue succeeds: all 20 eligible orders are delivered on the same day, 19 within their original windows and one late.
The on-time delivery rate against original commitments is 19 / 20 = 95%.
The same-day completion rate is 20 / 20 = 100%.
These are distinct metrics; counting only successful same-day deliveries would not reveal the late order.
Assume actual total driver paid time is 15 hours, original vehicle and fuel cost is USD 150, and rescue plus transfer actually costs USD 210.
Actual modeled operating cost is 15 × USD 24 + USD 150 + USD 210 = USD 720.
That is USD 234 above the USD 486 baseline, consisting of USD 24 additional driver labor plus USD 210 recovery cost.
The actual-cost comparison assumes the rescue charge does not already include the distributor's 15 paid hours or the original vehicle and fuel cost.
It excludes any unreported costs, credits, and revenue effects; it is not a full profit calculation.

### 6.5 Counterfactual discipline

The USD 102 and USD 156 alternatives were not executed; their costs and outcomes remain forecasts.
You cannot claim that rescue caused a specific amount of savings against them without a credible counterfactual.
A useful after-action review asks whether the transfer took the assumed time, whether the carrier could meet vehicle requirements,
and whether the one late order was caused by the breakdown or another event.
A repeated breakdown pattern may justify fleet maintenance; a single day does not establish a fleet-wide rate.

### 6.6 Transfer a recovery plan into instructions

The following hypothetical sequence makes the USD 210 rescue decision executable.
At 10:05 the dispatcher confirms the driver is safe and the vehicle cannot continue before the priority customer's closing time.
At 10:10 the dispatcher checks the six remaining order IDs against the cargo physically on vehicle A.
At 10:15 the carrier confirms a compatible rescue vehicle, driver, and quoted USD 180 charge.
At 10:20 an authorized manager approves the carrier and the USD 30 transfer labor estimate.
At 10:25 dispatch releases version 2, identifying the transfer location and the six orders to be moved.
The rescue driver acknowledges version 2; the original instruction to vehicle A is withdrawn where safe to do so.
The handover records package counts and visible damage, with any discrepancy escalated before departure.
Customer service communicates a forecast to the priority receiver and records the message time.
These times are illustrative events, not additional evidence that a particular transfer can meet the deadline.
The actual transfer, travel, and service times must be checked against the commitment.

### 6.7 Reconcile the handover

Suppose the six orders contain 14 labeled handling units.
The broken vehicle's manifest lists all 14 units, but the rescue driver initially counts 13.
The transfer is not complete simply because both drivers are at the same location.
Stop and identify the missing unit using order and handling-unit identifiers.
If the fourteenth unit is located and the count is confirmed, both parties record the corrected handover.
If it cannot be found, record which order and quantity are affected and alert the decision owner.
The remaining units may proceed only under the applicable cargo and partial-delivery policy.
A generic note saying “transferred” is insufficient to establish custody for all 14 units.
The subsequent customer delivery evidence is a separate event from the driver-to-driver handover.

### 6.8 Calculate the effect of an amended commitment

Suppose the one late delivery occurred at 12:20 against an original latest time of 12:00.
Before delivery, an authorized customer representative explicitly agreed to extend that order's window to 12:30.
Against the original commitment, the outcome remains one late order and 19 / 20 = 95% on time.
Against the separately reported amended commitment, all 20 orders are on time and the rate is 20 / 20 = 100%.
This comparison assumes every other order met both its original and any amended commitment.
Report both rates with their definitions; do not rewrite the original window in the history.
If the extension was communicated but not accepted, it is not an amended commitment in this example.
Whether an amendment affects contractual performance depends on the applicable agreement, not on the dashboard label.

### 6.9 Evaluate sensitivity without inventing a counterfactual

Rescue costs USD 210 under the original estimate.
If its carrier charge increases by USD 45 while transfer labor is unchanged, the estimate becomes USD 255.
The increase relative to the original rescue estimate is USD 45, or approximately 21.4%.
The estimated difference from option B becomes USD 255 − USD 102 = USD 153.
That difference is not a realized saving because B was not chosen and its service outcome is uncertain.
If the priority commitment is mandatory and B remains infeasible, the higher rescue cost alone does not make B acceptable.
If the customer agrees to an extension, recheck every other constraint before treating B as feasible.
The dispatch lead should capture the range of plausible costs and the time at which the quote expires.

### 6.10 Reconstruct the first decision interval

The following timeline extends the fictional case with explicit assumptions.
The breakdown occurs at 09:58 and the driver reports it at 10:00.
Dispatch receives the report immediately at 10:00.
The driver confirms a safe stopping location at 10:05.
The vehicle remains unavailable for the remainder of the relevant window.
The forecast and actual times in this illustration are local to the operation.

| Time | Event or decision | Information available at that time |
|---|---|---|
| 09:58 | Breakdown occurs | Driver observes vehicle failure |
| 10:00 | Dispatch receives report | Vehicle A has stopped; cargo status not yet confirmed |
| 10:05 | Safety confirmation | Driver safe; repair forecast exceeds priority window |
| 10:10 | Manifest review | Six orders remain on A; physical count still required |
| 10:15 | Carrier quote | Compatible vehicle offered at USD 180 |
| 10:20 | Approval | USD 180 carrier plus USD 30 transfer labor authorized |
| 10:25 | Revised instruction | Version 2 issued for the transfer and remaining orders |

The two-minute difference from occurrence to report is not dispatch decision time.
The elapsed time from detection at 10:00 to approval at 10:20 is 20 minutes.
The elapsed time from detection to instruction issuance at 10:25 is 25 minutes.
Neither number proves the driver acknowledged or executed the instruction.
Those events need their own timestamps.
If the driver accepts at 10:27, issuance-to-acceptance is 2 minutes.
If the carrier arrives at 10:50, acceptance-to-arrival is 23 minutes.
These two additional times are illustrative extensions, not observed case evidence.

The dispatcher can compute the latest rescue departure using section 5.12.
With 25 minutes to reach the broken vehicle, 15 minutes for transfer,
and 30 minutes from transfer to the priority customer, the total is 70 minutes.
For a service-start commitment of 12:00, the unbuffered latest departure is 10:50.
Issuing an instruction at 10:25 leaves 25 minutes until that deadline.
This margin assumes the carrier is ready to depart at the instruction time.
If carrier dispatch preparation takes 10 minutes, the effective margin falls to 15 minutes.
If service must instead finish by 12:00 and lasts 10 minutes,
the unbuffered latest departure becomes 10:40.
These calculations exclude traffic and transfer variability.
The actual decision deadline should allow the agreed operational buffer.

### 6.11 Account for option B's displaced work

The original USD 102 estimate for B is conditional on its stated scope.
Suppose a later review identifies an additional USD 24 of paid time
for vehicle B's existing customers that was not included in that estimate.
Assume this is genuinely avoidable, incremental, and not already in the USD 72 overtime.
Then the revised estimate is USD 72 + USD 30 + USD 24 = USD 126.
The original rescue estimate remains USD 210 under the original assumptions.
The forecast difference is USD 210 − USD 126 = USD 84.

That arithmetic does not resolve the priority customer's closing time.
If the original noon commitment remains binding and B is forecast late,
B still fails the stated service gate.
If an authorized extension is obtained, recompute all remaining commitments.
An extension for one customer does not automatically extend the other ten orders.
Check B's capacity, legal duty, transfer feasibility, and customer access.
If all those conditions pass, compare B and rescue on the same cost boundary.
The USD 24 is a hypothetical estimate, not an additional actual charge in section 6.4.

### 6.12 Reconcile actual cost with the planning baseline

The original daily budget includes 14 driver-hours at USD 24 per hour.
That component is USD 336.
The original vehicle and fuel allowance is USD 150.
The planned total is USD 486.
The actual scenario uses 15 distributor driver-hours, or USD 360.
It retains the USD 150 vehicle and fuel figure as an assumption.
The carrier charge is USD 180 and transfer labor is USD 30.
The actual modeled total is USD 360 + USD 150 + USD 180 + USD 30 = USD 720.

The labor variance is USD 360 − USD 336 = USD 24.
The vehicle and fuel variance is USD 150 − USD 150 = USD 0.
The incremental rescue and transfer cost is USD 210.
The total variance is USD 24 + USD 0 + USD 210 = USD 234.
It equals USD 720 − USD 486.
This bridge assumes the 15 hours exclude the labor included in the carrier charge.
It also assumes USD 30 transfer labor is not included in those 15 hours.
Verify both assumptions against payroll and the invoice before using actual ledgers.
If transfer labor is already in the 15 hours, remove the duplicate USD 30.

With 20 delivered orders, the modeled actual cost is USD 36 per delivered order.
The plan's USD 486 divided by 20 planned orders is USD 24.30 per planned order.
The USD 11.70 difference per order is a descriptive difference for this scenario.
It is not the causal cost of the breakdown without a credible counterfactual.
If fewer than 20 orders had been delivered, the denominator would also change.
Report total cost and service outcomes alongside per-order figures.

### 6.13 Explain a late upload without changing the event

Suppose an order is delivered at 11:55 but the device uploads its proof at 12:18.
The original commitment requires delivery by 12:00.
The relevant event is delivery at 11:55, assuming the evidence is valid.
The upload delay is 23 minutes.
Using receipt time as delivery time would incorrectly label the order 18 minutes late.
That conclusion depends on accepting the occurrence timestamp and its source.
If the device clock was incorrect, the evidence needs investigation.

At 12:05 the operational dashboard may still show the order as unresolved.
At 12:18 it can incorporate the late evidence and revise the status.
Preserve both the earlier dashboard state and the later correction.
Publish the data cutoff if reporting was produced before the upload.
If the report is revised, record the revision time and reason.
Do not fabricate an earlier notification merely because delivery was on time.

### 6.14 Compare ETA accuracy at a common horizon

Suppose three orders each had a forecast issued approximately one hour before arrival.
Their predicted arrival times were 10:00, 10:30, and 11:00.
Their actual arrival times were 10:10, 10:25, and 11:20.
The signed errors, actual minus predicted, are +10, −5, and +20 minutes.
The absolute errors are 10, 5, and 20 minutes.
Mean absolute error is (10 + 5 + 20) / 3 = 11.67 minutes.
The mean signed error is (10 − 5 + 20) / 3 = 8.33 minutes.
Positive signed error here means arrival later than forecast.

This three-order example is not evidence of stable system performance.
If another scheduled order lacks a matched prediction, report coverage as 3 / 4 = 75%.
Do not quietly calculate the error for only three matches and imply full coverage.
A one-minute-ahead ETA should not be mixed with these one-hour-ahead forecasts.
Arrival accuracy is also not equivalent to delivery-window compliance.
The relevant commitment might apply to service start or completion instead.

## 7. Code, query, or calculation

The following standalone Python calculation compares explicit recovery scenarios and computes two service KPIs.
It is a decision aid, not an optimizer or a safety authorization.
Every scenario's feasibility and cost inputs must be checked outside the script.

```python
# Illustrative USD and order counts for one delivery day.
options = {
    "rescue": {"carrier": 180, "transfer": 30, "overtime": 0,
               "reattempt": 0, "credit": 0, "meets_commitments": True},
    "use_B": {"carrier": 0, "transfer": 30, "overtime": 72,
              "reattempt": 0, "credit": 0, "meets_commitments": False},
    "defer": {"carrier": 0, "transfer": 0, "overtime": 0,
              "reattempt": 96, "credit": 60, "meets_commitments": False},
}
cost_fields = ("carrier", "transfer", "overtime", "reattempt", "credit")
for name, item in options.items():
    cost = sum(item[field] for field in cost_fields)
    print(f"{name}: incremental USD {cost}; commitment-feasible={item['meets_commitments']}")

eligible_orders = 20
on_time = 19
same_day_delivered = 20
print(f"on-time: {on_time / eligible_orders:.1%}")
print(f"same-day completion: {same_day_delivered / eligible_orders:.1%}")

planned_cost = 14 * 24 + 150
actual_cost = 15 * 24 + 150 + 180 + 30
print(f"planned USD {planned_cost}; actual USD {actual_cost}; variance USD {actual_cost - planned_cost}")
```

Run with `python dispatch_cost.py` after saving the block to that file.
Expected costs are rescue USD 210, use_B USD 102, and defer USD 156.
Expected rates are 95.0% on time and 100.0% same-day completion.
Expected planned, actual, and variance values are USD 486, USD 720, and USD 234 respectively.
The `meets_commitments` flag reflects the stated scenario, not an independently checked fact.
A real decision would also display uncertainty, legal feasibility, approval limits, and customer preferences.

### 7.1 KPI definitions

| KPI | Numerator | Denominator | Interpretation |
|---|---|---|---|
| On-time delivery | Eligible orders delivered within original agreed window | All eligible committed orders | Counts failed and undelivered eligible orders as not on time |
| Same-day completion | Eligible orders delivered on scheduled date | All eligible orders scheduled that date | Does not establish window compliance |
| First-attempt success | Orders delivered on first eligible attempt | Orders with a first eligible attempt | Excludes unattempted orders; report them separately |
| Attempt rate | Orders with an eligible attempt | Orders scheduled | Reveals omissions hidden by first-attempt success |
| ETA error | Absolute difference between predicted and actual arrival, minutes | Matched predictions | Report prediction horizon and coverage |
| Override rate | Released route changes with a documented override | Released routes or stops, as specified | Stable denominator and reason taxonomy required |
| Cost per delivered order | Allocated route operating cost, USD | Delivered orders | Cannot replace service and undelivered-order reporting |
| Exception response time | Decision timestamp minus detection timestamp, minutes | Exceptions with both events | Exclude invalid timestamps with disclosed count |

Define cancellation eligibility and rescheduled commitments before measurement.
Freeze original commitments and record approved amendments separately.
Report late-arriving events and missing timestamps rather than silently excluding them.
An on-time rate based only on completed stops can improve while unserved orders increase.

### 7.2 Calculate service with an explicit eligible population

This standalone Python example uses four fictional orders.
Times are integer minutes from the same local planning origin.
The delivery event, rather than arrival, is the contractual event in this example.
One eligible order has no delivery event and stays in the denominator.
One canceled order is excluded under this example's predetermined eligibility rule.
The code does not determine whether a cancellation is contractually excludable.
Save the block as `service_kpis.py` and run it with Python 3.

```python
orders = [
    {"id": "A", "eligible": True, "start": 60, "end": 120,
     "delivered": 115, "first_attempt_success": True},
    {"id": "B", "eligible": True, "start": 60, "end": 120,
     "delivered": 125, "first_attempt_success": False},
    {"id": "C", "eligible": True, "start": 60, "end": 120,
     "delivered": None, "first_attempt_success": None},
    {"id": "D", "eligible": False, "start": 60, "end": 120,
     "delivered": None, "first_attempt_success": None},
]

eligible = [order for order in orders if order["eligible"]]
delivered = [order for order in eligible if order["delivered"] is not None]
on_time = [
    order for order in delivered
    if order["start"] <= order["delivered"] <= order["end"]
]
attempted = [
    order for order in eligible
    if order["first_attempt_success"] is not None
]
first_success = [
    order for order in attempted if order["first_attempt_success"]
]

def show(label, numerator, denominator):
    if denominator == 0:
        print(f"{label}: undefined (0 eligible)")
    else:
        print(f"{label}: {numerator}/{denominator} = "
              f"{numerator / denominator:.1%}")

show("on time", len(on_time), len(eligible))
show("completion", len(delivered), len(eligible))
show("attempt coverage", len(attempted), len(eligible))
show("first-attempt success", len(first_success), len(attempted))
print("excluded", len(orders) - len(eligible))
```

Expected output:

```text
on time: 1/3 = 33.3%
completion: 2/3 = 66.7%
attempt coverage: 2/3 = 66.7%
first-attempt success: 1/2 = 50.0%
excluded 1
```

Order B is delivered but outside the original window.
Order C is neither delivered nor attempted.
Order D is outside the eligible population under the stated rule.
Order B's failed first attempt is an assumption in this synthetic data.
Its eventual delivery could have occurred on a later attempt.
The code does not infer an attempt merely from the existence of a delivery.
A real ledger needs validated attempt events and correction rules.
The first-attempt percentage is conditional on an attempt having occurred.
It should be read with the attempt-coverage percentage.

### 7.3 Build a decision record from the calculation

The calculation is one component of a decision record, not the record itself.
For the breakdown, retain the original route and the approved revision.
Record the detection time, owner, and latest useful decision time.
List each considered option and the facts used to evaluate it.
Record conditional assumptions, especially vehicle availability and customer consent.
Record the approved spend and the approver's authority.
Link the driver's acknowledgment and the transfer evidence.
Retain the eventual invoice and any subsequent adjustment.

For service reporting, record the eligible population and reporting cutoff.
List the number of exclusions and their categories.
Identify whether delivery, arrival, or service completion is the measured event.
Link the event's occurrence time and source rather than only the ingestion time.
Publish a late-data correction when the underlying state changes.
A reproducible record allows another analyst to obtain the same result.
It does not prove that an unchosen option would have had its forecast outcome.

### 7.4 Interpret an undefined rate

A percentage with a zero denominator is undefined.
For example, if no eligible orders were attempted, first-attempt success has no denominator.
Reporting zero percent would imply that eligible attempts occurred and all failed.
Reporting 100 percent would imply that all attempted orders succeeded.
Neither statement follows from an empty attempt population.
Show the zero count and explain that the rate is unavailable.
Attempt coverage may still be zero if eligible orders exist.
This distinction helps identify an operational omission rather than a failed visit.

## 8. Data, standards, and regulatory context

An event ledger should contain order ID, stop ID, route version, vehicle, event type, occurrence time, receipt time, source, and correction history.
Order and stop are separate entities: multiple orders may share a stop, and a failed order may require another attempt.
Store UTC instants and the relevant local time zone so that customer-window calculations are reproducible.
Preserve the original window, negotiated changes, and the time at which each change became effective.
A late-arriving proof-of-delivery event should update operational status without erasing its delayed receipt.

GS1 provides [identification keys](https://www.gs1.org/standards/id-keys) for trading-partner identification where adopted.
Standard identifiers improve joins but do not prove that a driver attended the correct entrance.
Electronic proof of delivery must be assessed against the contract, evidence policy, and applicable law.
US motor-carrier hours-of-service requirements vary by operation; consult the current
[FMCSA summary](https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations).
A dispatcher should not infer legal compliance from a route duration alone; prior duty and applicable exceptions may matter.
This lesson is educational, not legal or regulatory advice.

Data retention and access to location history, signatures, and contact information require organizational privacy controls.
Limit access by role and record corrections and overrides for accountability.
Do not use estimated timestamps as though they were signed customer evidence.
A reported KPI needs a defined period, source, eligibility rule, and late-data cutoff.

## 9. Industry practice and case evidence

UPS describes [ORION](https://about.ups.com/us/en/our-company/innovation-driven/orion.html) as a route optimization capability.
This is a company source, not an independent measurement of a particular site's dispatch outcomes.
The OR-Tools documentation describes [vehicle routing](https://developers.google.com/optimization/routing) models and constraints.
It does not prescribe a company's exception-approval process or prove that a modeled route is executable.
Inference: operational deployment needs a process for reconciling solver output with changing driver, customer, and vehicle conditions.
The fictional breakdown and economics above are teaching assumptions, not a reported UPS case.

A credible field evaluation would define a baseline before launch and track both planned and actual service.
Compare similar service areas, weekdays, order volumes, customer windows, and vehicle constraints.
Measure paid hours and external recovery charges alongside customer outcomes.
Report unserved orders, not just completed-stop statistics.
Capture dispatcher overrides and examine whether they improve service, reveal bad inputs, or bypass policy.
Where feasible, use a randomized or otherwise defensible comparison and disclose remaining confounding.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Draft sent as instruction | Driver sees unapproved route | No release state | Conflicting work | Versioned approval and acknowledgment |
| ETA mistaken for promise | Customer sees changing deadline | Forecast overwrites commitment | Dispute and service failure | Store both independently |
| Replan ignores cargo | Suggested vehicle lacks goods | State starts at old depot | Impossible recovery | Freeze onboard inventory and transfer |
| Safety exception traded for cost | Pressure to skip break | Objective lacks hard rule | Injury or violation | Safety gate and escalation |
| Duplicate events | Multiple identical completions | Mobile retries without IDs | Inflated delivery count | Idempotent event ingestion |
| Missing failed orders | KPI only uses completed orders | Selection bias | False service improvement | Report eligible denominator and failures |
| Stale position | Replan starts at old location | Delayed telemetry | Infeasible arrival | Timestamp and verify position |
| Double-counted cost | Carrier charge plus included labor | Overlapping cost categories | Wrong decision | Reconcile cost definitions |
| No owner | Exception queue ages | Responsibility ambiguous | Missed escalation window | Assign owner and deadline |
| Uncommunicated revision | Driver follows old sequence | No acknowledgment | Missed stop | Confirm receipt of new version |
| Incentive to hide overrides | No reason codes recorded | Punitive metric | Lost learning | Review context and support safe overrides |
| Timestamp ambiguity | Negative dwell or ETA error | Mixed local and UTC times | Misleading KPI | Store instants and zone metadata |

## 11. Assumptions and limits

| Assumption | Why used | What breaks it | How to test or respond |
|---|---|---|---|
| Rescue capacity available | Makes R actionable | Carrier declines | Confirm acceptance before release |
| Transfer is safe | Goods move from A | Product or site restriction | Verify equipment, chain of custody, and handling |
| Costs are incremental | Enables option comparison | Common or sunk costs included | Reconcile components at decision time |
| Noon is a firm commitment | Makes B unacceptable without consent | Receiver agrees to extension | Record authorized amended window |
| Forecast repair misses noon | Motivates recovery | Repair is faster | Refresh repair estimate and decision deadline |
| Actual event times are reliable | Enables KPI calculation | Clock drift or delayed entry | Retain source, receipt time, and corrections |
| Twenty orders are eligible | Sets denominator | Valid exclusions apply | Publish eligibility and exclusion counts |
| Rescue and distributor labor do not overlap | Prevents double count | Quote includes same labor | Inspect invoice and costing policy |
| One-day case generalizes cautiously | Illustrates process | Different network or contract | Run a comparable multi-day evaluation |
| Driver and vehicle are legal | Feasibility prerequisite | Duty or equipment limits | Check live compliance before assignment |

## 12. Exercises

1. **Distinguish objects**, 10 minutes. Deliver definitions of route plan, dispatch instruction, ETA, and customer commitment. Acceptance: name owner or trigger and explain why ETA is not a promise.
2. **Handoff map**, 20 minutes. Draw a planning-to-closeout process for the fictional distributor. Acceptance: include release, acknowledgment, event capture, exception ownership, and feedback.
3. **Triage**, 15 minutes. Classify a vehicle breakdown carrying temperature-sensitive goods. Acceptance: prioritize safety, specify cargo and customer risks, and assign a decision owner.
4. **Recovery economics**, 20 minutes. Recalculate rescue when the carrier costs USD 195 and transfer labor costs USD 36. Compare it with the original B and D estimates. Acceptance: show arithmetic and explain why price alone does not settle the choice.
5. **KPIs**, 15 minutes. Of 24 eligible orders, 21 are delivered within the original window, 2 late, and 1 not delivered. Calculate on-time and completion rates. Acceptance: state both denominators and distinguish the measures.
6. **Reoptimization control**, 25 minutes. Design a release checklist for a revised route while a driver is serving a customer. Acceptance: freeze active work, identify onboard cargo, check working time, and require driver acknowledgment.
7. **Measurement critique**, 15 minutes. A manager reports 21 on-time deliveries out of 23 completed orders as the day's on-time rate. Diagnose the issue using exercise 5. Acceptance: calculate both rates and identify the omitted order.
8. **Design trade-off**, 25 minutes. The customer offers to extend the noon window by one hour. Specify what you need before selecting B. Acceptance: include authorized confirmation, revised feasibility, customer impact, and cost uncertainty.

## 13. Self-check answers

1. A route plan is a versioned proposal or schedule. A dispatch instruction is authorized work communicated to a resource. An ETA is a time-stamped forecast; a commitment is the agreed window or deadline. Only an authorized agreement changes the commitment.
2. A strong map shows a snapshot, proposal, feasibility review, approved release, acknowledgment, events and exception triage, decision and communication, then reconciliation and feedback. Every handoff has a named owner.
3. First protect the driver and other road users and invoke emergency procedures as appropriate. Preserve temperature and chain-of-custody information, assess rescue equipment, notify the accountable dispatcher, and communicate customer risk. Do not instruct an unsafe transfer to save cost.
4. Rescue costs USD 195 + USD 36 = USD 231. B remains USD 102 and D USD 156 under their original estimates. Rescue costs USD 129 more than B and USD 75 more than D, but the alternatives have different commitment outcomes and uncertain feasibility.
5. On-time is 21 / 24 = 87.5%. Completion is 23 / 24 ≈ 95.8%. The undelivered order is included in both denominators and is neither on time nor completed.
6. The checklist should lock completed and active work, locate the driver and cargo, validate vehicle compatibility and remaining legal duty, inspect changed commitments, approve a version, and confirm receipt. A proposal that fails a hard gate is rejected or escalated.
7. The completed-only rate is 21 / 23 ≈ 91.3%, while the eligible-order rate is 21 / 24 = 87.5%. The former excludes the undelivered order and should not be presented as performance against all scheduled commitments.
8. Obtain authorized customer agreement and record its effective time. Recompute actual transfer, routing, capacity, duty, and remaining windows. Compare incremental costs including any extension charge, and communicate residual risk; do not assume that an extension alone makes B feasible.

### Answer detail: objects and handoffs

For exercise 1, the planner produces a candidate route using a dated snapshot.
The release owner approves a particular version for operational use.
The dispatch instruction tells a named driver or carrier what work to perform.
The ETA is a forecast with an issue time and a target event.
The commitment is the agreed window or deadline for a defined event.
An updated forecast does not itself change the agreement.
An authorized amendment needs evidence and should not erase the original.

For exercise 2, a strong handoff map identifies both sender and receiver.
The planner sends a candidate and its input versions to the release owner.
The release owner checks constraints and sends an approved instruction.
The driver acknowledges or rejects that instruction.
Events flow back to dispatch during execution.
An exception has an owner and a deadline for action.
Customer service sends appropriate updates and records any response.
Closeout reconciles service, physical custody, cost, and unresolved cases.
The data owner evaluates verified corrections for future planning.

For exercise 3, first confirm immediate safety using established procedures.
The dispatcher establishes where the vehicle and goods are located.
The product-quality owner determines how to assess temperature exposure.
The fleet or carrier owner confirms a compatible rescue resource.
The customer-service owner communicates risk without claiming a guaranteed outcome.
The transfer record must identify goods and any condition evidence.
If handling conditions are uncertain, escalate rather than guessing product fitness.

### Answer detail: economics and metrics

For exercise 4, add only the changed rescue components.
The carrier is USD 195 and transfer is USD 36.
Their sum is USD 231.
Against the unchanged USD 102 B estimate, the difference is USD 129.
Against the unchanged USD 156 D estimate, the difference is USD 75.
Those are forecast differences, not realized savings.
The USD 60 credit in D is conditional under the case assumptions.
The service gate and the uncertain receiver extension still matter.
Check that the carrier's quote does not already include the transfer labor.

For exercise 5, start with 24 eligible orders.
Twenty-one on time plus two late plus one undelivered equals 24.
On-time performance is 21 ÷ 24 = 0.875, or 87.5%.
Completion is (21 + 2) ÷ 24 = 23 ÷ 24, or approximately 95.8%.
These rates measure different outcomes against the same eligible population.
If the late orders were delivered on another day, label completion accordingly;
the calculation alone does not establish same-day completion.

For exercise 7, 21 ÷ 23 is approximately 91.3%.
It is an on-time rate conditional on completion, not the eligible-order rate.
The missing order makes the conditional rate 3.8 percentage points higher,
after rounding both results to one decimal place.
The manager should show the eligible-order rate and the undelivered count.
The completed-only figure can be useful if its denominator is explicitly labeled.
It cannot replace service performance against all eligible commitments.

### Answer detail: control and authorization

For exercise 6, the current stop remains frozen until safe interruption is confirmed.
The checklist identifies all completed and remaining work.
It records cargo location and any required transfer before moving an assignment.
It checks the actual position, duty state, equipment, and remaining windows.
It compares the proposed revision with the released instruction.
An authorized person approves the new version before it is transmitted.
The driver acknowledges that version or dispatch follows the no-response process.
The revised plan remains subject to actual events after release.

For exercise 8, document which person offered the extension and their authority.
Record the affected order, new deadline, acceptance, and effective time.
Keep the original deadline for historical reporting.
Recalculate B's entire remaining route from its current location and cargo state.
Include transfer duration, capacity, duty, and other customer commitments.
Update incremental labor and any customer or carrier charges.
If any required input is unconfirmed, mark the option conditional.
If a hard rule fails, do not select B solely because the extension was offered.

## 14. Interview questions

1. **What is dispatch?** Strong: operational control and exception handling after release. Weak: merely running an optimizer.
2. **What distinguishes an ETA from a commitment?** Strong: forecast versus agreed obligation, separately versioned. Weak: a revised ETA automatically changes the promise.
3. **How do you replan after a breakdown?** Strong: freeze completed work, locate cargo, check transfer and remaining duty, then authorize and communicate. Weak: solve again from the original depot.
4. **How do you compare recovery choices?** Strong: feasible options, incremental avoidable costs, service consequences, and uncertainty. Weak: choose the lowest displayed route score.
5. **How would you define on-time delivery?** Strong: specify eligible population, original window, event, exclusions, and late-data policy. Weak: count only successful deliveries.
6. **Why retain old plan versions?** Strong: explain decisions, audit communications, and compare forecast with actual. Weak: overwrite them to simplify reporting.
7. **What does driver acknowledgment add?** Strong: confirmation that the new instruction was received and accepted, not proof it was executed. Weak: assumes sending equals delivery.
8. **What if a route violates a safety rule?** Strong: stop or reject the instruction and escalate. Weak: accept if penalty cost is low enough.
9. **How do you use overrides?** Strong: record reason and context, then investigate recurring data or policy issues. Weak: treat every override as driver error.
10. **How do you evaluate a pilot?** Strong: comparable baseline, eligible-order service, full costs, and explicit confounders. Weak: claim savings from one optimized route.

## 15. Key terms

- **Acknowledgment:** confirmation of instruction receipt or acceptance; see section 5.4.
- **Actual:** observed outcome, subject to event quality; see section 5.5.
- **Attempt:** a defined visit or delivery attempt; see section 4.
- **Commitment:** customer-agreed service obligation; see section 3.
- **Completion rate:** deliveries completed divided by eligible scheduled orders; see section 7.1.
- **Counterfactual:** outcome under an unchosen alternative; see section 6.5.
- **Decision deadline:** latest time for a useful response; see section 5.6.
- **Dispatch:** control of execution and exceptions; see section 3.
- **Dwell:** time at a location; see section 4.
- **Eligibility:** rule determining inclusion in a KPI; see section 7.1.
- **ETA:** forecast arrival time; see section 3.
- **Event time:** time an operational event occurred; see section 5.5.
- **Exception:** divergence requiring attention; see section 5.6.
- **First-attempt success:** success among attempted orders; see section 7.1.
- **Frozen prefix:** work not freely changeable in replanning; see section 5.7.
- **Incremental cost:** avoidable difference from an alternative; see section 5.7.
- **On-time rate:** eligible orders delivered within commitment divided by eligible orders; see section 7.1.
- **Override:** authorized deviation from a recommendation; see section 4.
- **Plan version:** identifiable historical route plan; see section 5.4.
- **Proof of delivery:** evidence of a delivery event; see section 4.
- **Reoptimization:** revised planning from current operational state; see section 5.8.
- **Rescue:** alternative resource used to recover disrupted work; see section 6.2.
- **Route plan:** versioned assignment and sequence; see section 3.
- **Service window:** agreed period for a defined delivery event; see section 8.

## 16. Sources

- Google OR-Tools, [vehicle routing overview](https://developers.google.com/optimization/routing).
- UPS, [ORION](https://about.ups.com/us/en/our-company/innovation-driven/orion.html), company description; claims require independent evaluation in another operation.
- FMCSA, [summary of hours-of-service regulations](https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations); confirm current applicability.
- GS1, [identification keys](https://www.gs1.org/standards/id-keys).

## 17. Links

- [Module index](_index.md)
- [Runnable CVRPTW in OR-Tools](06_cvrptw-ortools.md)
- [Freight modes](01_freight-modes.md)
- [Rate structures](02_rate-structures.md)
- [Lane and tender analytics](03_lane-tender-analytics.md)
- [Module survey and lab context](../../../course/12_transportation_routing_last_mile.md)
