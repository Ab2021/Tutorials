# Twin Architecture and When Not to Build

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 11, and 15; Topics 01, 04, and 05  
> Version 2026.5

## 1. Why this topic exists

A distribution manager must decide which inbound trailer to unload next.
The decision recurs throughout a shift, and an incorrect choice can delay store replenishment.
A vendor offers a live digital twin that combines appointments, yard events, labor, and inventory.
The proposal promises faster decisions, but the underlying operating problem is not a software category.
The organization must decide whether better, more current state changes enough decisions to justify a maintained system.

The manager could use the existing appointment board.
An analyst could build a weekly simulation of alternative staffing policies.
A connected shadow could display estimated conditions without sending commands.
A decision-support twin could recommend the next trailer while a supervisor retains authority.
An automatic controller could execute a dispatch decision, but that requires a separate safety and authority case.
These alternatives have different costs, risks, and evidence requirements.
A three-dimensional visualization is not evidence that any of them improves throughput.

Integration is often the hidden commitment.
A source system can report a trailer arrival quickly but report a door change much later.
A warehouse management system can correct an inventory transaction after the initial event.
A model that ignores these differences may simulate a coherent but false present.
Re-running that model more frequently does not repair its inputs.

The cost continues after deployment.
Interfaces fail, site processes change, identifiers are reused, and operational teams need support.
Someone must approve model changes, investigate degraded forecasts, and decide when recommendations stop.
A business case that counts only a software subscription and a one-time implementation is incomplete.

This topic describes a decision-first architecture for a maintained operational model.
It covers data contracts, synchronization, safe recommendation and control boundaries, governance, maintenance, and total cost.
It also shows how to compare a twin with less elaborate alternatives.
The worked example is fictional; its numbers are assumptions for learning, not results from a named company.

## 2. Learning outcomes

After this topic you can:

1. Define the decision, horizon, owner, and baseline before selecting a system architecture.
   Exercise 1 tests the boundary and decision rights.
2. Specify a minimum state and a source contract, including identity, time, corrections, and quality.
   Exercise 2 tests the contract.
3. Calculate a freshness budget and choose a synchronization cadence appropriate to a decision.
   Exercise 3 tests the calculation.
4. Describe replay, reconciliation, and recovery after delayed or incorrect events.
   Exercise 4 tests recovery design.
5. Assign ownership for data, model, operation, security, and economic outcomes.
   Exercise 5 tests governance.
6. Compute incremental discounted value and compare a twin with a cheaper alternative.
   Exercise 6 tests the business case.
7. Design validation, maintenance, and retirement gates.
   Exercise 7 tests lifecycle planning.
8. Recommend not building when benefits, data, rights, or evidence are insufficient.
   Exercise 8 tests the decision.

Each outcome concerns an observable deliverable rather than a promise of operational competence.
Real deployment requires site-specific evidence, technical review, and accountable approval.

## 3. The core mental model

A twin is a maintained decision system, not a picture of a facility.
Use the following working convention: an offline simulation is run from a specified state or dataset;
a shadow receives operational data but does not directly change the physical system;
a decision-support twin couples a maintained representation to recurring decisions;
and closed-loop control additionally sends authorized actions to the operation.
Terminology differs across organizations, so describe the actual flows rather than relying on the label.

```text
Business decision, decision owner, alternatives, constraints
                         |
Physical operation --> source events --> ingestion and data contracts
       ^                                 |
       |                         identity, time, quality
       |                                 |
       |                       reconciled state + uncertainty
       |                                 |
       |                     model and scenario experiments
       |                                 |
       |                      recommendation + evidence
       |                                 |
       +-- authorized action <-- human or separately governed controller
                                     |
                       actual outcome, audit, review, learning
```

The forward path begins with observations, not perfect knowledge of physical reality.
The state estimate is assembled from sources with different delays and error modes.
The model projects consequences conditional on that state and on explicit assumptions.
A recommendation is a proposed action, not an instruction to machinery.
A human or separately authorized controller decides whether an action is permitted.
The outcome must be compared with an appropriate baseline and fed into review.

Three clocks matter.
The physical clock is when an event occurred.
The data clock is when it became usable in the state estimate.
The decision clock is when a person or controller needs an answer.
A data feed can be fast on average and still be unusable if its tail delay exceeds the decision window.
Conversely, a daily planning decision rarely needs every sensor event streamed in real time.

There are also three different forms of uncertainty.
Measurement uncertainty concerns what is currently happening.
Model uncertainty concerns how the operation responds to an alternative.
Decision uncertainty concerns which feasible action is best given imperfect information.
A live feed primarily addresses the first; it does not automatically solve the other two.

Draw the system boundary around the decision and its material constraints.
For dock sequencing, trailers, appointments, unloading resources, and urgent orders may be inside.
Detailed truck engine telemetry is outside unless it changes the decision.
A supplier's capacity can be represented as an assumption when direct observation is unavailable.
Make that boundary visible to users so that a detailed interface does not imply complete coverage.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse or unit |
|---|---|---|
| Decision | Choice among feasible actions by an accountable owner. | Treating a dashboard metric as a decision. |
| Baseline | Actual or credible alternative practice used for comparison. | Comparing with no operation instead of the best feasible alternative. |
| Physical system | The real network, site, assets, people, and policies in scope. | Assuming every physical event is observed. |
| Model boundary | Explicit inclusion and exclusion of processes and constraints. | Equating a data interface with model coverage. |
| Simulation | Executable representation used to explore system behavior. | Assuming it is always connected to current data. |
| Shadow | Representation fed by observations without direct physical actuation in this convention. | Assuming it necessarily validates future scenarios. |
| Decision-support twin | Maintained, synchronized representation used for recurring decisions in this convention. | Treating the term as a universal standard definition. |
| Closed loop | Authorized actions can flow back to the physical process. | Treating recommendations as safe automatic control. |
| State | Variables describing the modeled system at a defined time. | Mixing values from incompatible timestamps. |
| State estimate | Best available reconstructed state with provenance and uncertainty. | Calling it ground truth. |
| Entity | Individually identified object, such as a trailer or door. | Assuming IDs are globally unique without a namespace. |
| Event | Report that something occurred, with identity and time. | Treating an arrival at the data platform as the physical event time. |
| Event time | Time the source says the physical or business event occurred. | Confusing it with ingestion time; use a time zone. |
| Processing time | Time the platform processed the event. | Assuming processing order is event order. |
| Watermark | Policy estimate of event-time progress used to handle lateness. | Treating it as a guarantee no later event will arrive. |
| Latency | Elapsed time between specified stages. | Reporting only an average; specify endpoints and percentile. |
| Freshness | Age of usable information at a defined decision time. | Assuming ingestion latency alone measures freshness. |
| Cadence | Frequency of an observation, refresh, model run, or decision. | Assuming these frequencies must be equal. |
| Snapshot | State captured as of a defined time and version. | Assuming a multi-source snapshot is automatically consistent. |
| Replay | Reprocessing retained events to reconstruct state. | Repeating external actions when replaying data. |
| Idempotency | Repeating an operation produces no additional effect. | Assuming all upstream event identifiers are unique. |
| Reconciliation | Comparison and correction against an authoritative record or physical count. | Silently overwriting unexplained discrepancies. |
| Data contract | Agreed schema, meaning, delivery, quality, and ownership expectations. | A schema alone without semantics or an escalation owner. |
| Provenance | Record of sources and transformations behind an output. | Assuming a model version alone identifies the data. |
| Model drift | Change in the relationship between modeled and observed behavior. | Treating every random deviation as drift. |
| Validation | Evidence that a model is adequate for a specified use. | A claim of universal correctness. |
| Verification | Evidence that implementation follows its specification. | Confusing working code with a valid model. |
| Override | Authorized departure from a recommendation, with reason. | Treating every override as user error. |
| Fail-safe | Defined response when a critical condition fails. | Assuming an automatic stop is always safe without operational analysis. |
| Service-level objective | Internal target for an observable service measure. | Confusing it with a contractual guarantee. |
| Total cost of ownership | Full lifecycle expenditure for owning and operating the capability. | Counting only licenses. |
| Incremental benefit | Benefit beyond the chosen alternative, net of overlap. | Counting all improvement since launch as causal impact. |
| Net present value | Discounted incremental benefits less discounted costs. | Mixing nominal and real rates or double-counting costs. |

## 5. The mechanics, step by step

### Step 1 — Establish the decision and its counterfactual

**Input:** an operational problem, its owner, and a proposed intervention.
**Output:** a decision record and a baseline suitable for evaluation.

Write the decision in one sentence: who chooses what, how often, for which horizon, under which constraints?
For example: a shift supervisor chooses the next eligible inbound trailer for each available door every 15 minutes.
The immediate horizon is the next 90 minutes; safety and labor rules remain binding.
Record who will actually use the recommendation and who owns the consequence.
Specify whether the output is a forecast, a suggested plan, a target, or a commitment.
A forecast estimates what may happen; it does not authorize a change.
A plan is a proposed sequence; a target is a desired outcome; a commitment is an accepted obligation.

List feasible alternatives before comparing technology.
The baseline might be the existing supervisor procedure, not first-in-first-out if supervisors already prioritize urgent loads.
A stronger comparison might be a simple daily priority list generated from existing data.
Include implementation and recurring costs for each alternative on a consistent basis.
If an offline study can answer a one-time investment decision, compare it explicitly before paying for continuous synchronization.

Document value mechanisms rather than broad promises.
Fewer urgent-order delays might reduce expedited freight, but only if those shipments would otherwise have been expedited.
Faster unloading may release capacity without creating savings if labor and door costs are fixed and unused capacity has no value.
Prevented double counting is a design requirement: the same avoided late shipment cannot be claimed as both full freight savings and full service-penalty savings unless both costs truly disappear.

**Branch:** If there is no recurring decision, no action authority, or no credible incremental value mechanism, stop or use a one-off study.
Do not create a live integration merely because data are available.

### Step 2 — Define the minimum viable state

**Input:** decision record, process map, and interviews with operators.
**Output:** state dictionary, boundary diagram, and list of non-observable quantities.

Trace the decisions backward to variables that can change the ranking of feasible actions.
For each variable specify unit, grain, time, source, confidence, owner, and permissible use.
A dock example requires trailer identity, eligibility, location, arrival time, order priority, unloading workload, door status, and labor availability.
It may require estimates of unloading time by load type rather than a single deterministic duration.
Avoid acquiring detailed data simply because a source can provide it.
Data with no decision sensitivity create maintenance and privacy costs without demonstrated benefit.

Separate state from parameters.
A trailer's current door is state; an estimated unloading-time distribution is a model parameter.
The state may update every few minutes; a parameter may need weekly review or change when packaging changes.
A historical distribution is not a direct observation of the next trailer's duration.
Treat unobserved physical conditions as unknown, not as zeros.

Map identifiers across systems.
A trailer identifier may be reused across visits, so use a site identifier and visit identifier as well.
Define whether a shipment, trailer visit, appointment, and purchase order are distinct entities.
Maintain explicit mappings and effective dates for doors, facilities, products, and organizational changes.
A successful join is not proof that the joined records represent the same physical event.

### Step 3 — Contract for data and semantics

**Input:** state dictionary and source-system inventory.
**Output:** signed source contracts and testable acceptance criteria.

For every source, record the producer, business owner, technical contact, and expected availability.
Define required fields, permitted values, units, time zone, identifier namespace, and versioning policy.
Distinguish mandatory fields from optional fields and define the treatment of missing values.
Include event time and ingestion time; a correction must identify the original fact or define the replacement rule.
Specify whether a cancellation is a new event, a deletion, or a change of status.
Define duplicate detection and what happens when identifiers are absent or conflicting.

Specify quality indicators that users can act on.
Examples include percentage of eligible trailer visits with a valid arrival timestamp, percentage of records with an unmatched visit ID, and age of the oldest unresolved event.
A completeness measure needs a denominator: expected records, observed records, or sampled physical visits produce different conclusions.
Record thresholds, measurement windows, and escalation owners.
Do not interpret an internal service-level objective as proof of contractual or regulatory compliance.

Negotiate change notification.
A producer adding an optional field may be harmless; changing a field's unit or reusing an enum value may silently corrupt the model.
Specify compatibility tests and a notice period appropriate to the organization.
For a breaking change, retain a rollback route and prevent the consumer from treating incompatible data as valid.

### Step 4 — Choose cadence from the decision window

**Input:** decision interval, state volatility, source delay distribution, and consequence of staleness.
**Output:** refresh, reconciliation, decision, and fallback policies.

Begin with the maximum tolerated age of each decision-critical variable.
It need not be the same for appointments, door status, and order priority.
Estimate end-to-end age from physical observation to availability at decision time, not merely message transport time.
For periodic sampling, a change can wait almost one full sampling interval before it is observed.
Add ingestion, transformation, model computation, and presentation delays.
If a user may wait before acting, include that delay in the operational freshness budget.

Use distributional evidence, not just means.
If the 95th-percentile delay is acceptable but the 99th-percentile delay creates a safety or service problem, the design needs an appropriate tail threshold and fallback.
Do not add component percentiles and claim the result is the exact percentile of total delay.
A conservative sum of individual bounds is a budget, not a measured end-to-end percentile.
Measure total elapsed time on traced events after implementation.

A fast feed does not oblige the model to run after every event.
Batching may be sufficient when decisions occur every 15 minutes and stable state changes do not affect the ranking.
Conversely, trigger an exceptional refresh for a material event such as a door outage, if the source and decision process can support it.
A periodic full reconciliation can coexist with event-driven updates.
Specify what happens when events arrive late, out of order, duplicated, or not at all.

**Branch:** If critical state is older than the permitted age, show a stale-state warning and withhold the affected recommendation.
The approved fallback is the existing supervisor procedure, not fabricated current state.
If the operational decision is sensitive to conditions faster than they can be observed or safely acted upon, do not claim the architecture supports that decision.

### Step 5 — Build a reproducible state pipeline

**Input:** contracted events and a state reconstruction policy.
**Output:** a versioned state snapshot, quality status, and provenance.

Retain source events according to security, retention, and contractual constraints.
Separate immutable event evidence from derived current-state tables where feasible.
Key events by producer, source event ID, and version rather than assuming a single global identifier.
Use idempotent updates so that duplicate delivery does not create a second trailer arrival or a second action.
Where the source lacks a stable identifier, define and test a defensible deduplication rule; do not silently merge uncertain records.

Order by business event time only when that is semantically valid.
A later correction may supersede an earlier record even when it describes an earlier occurrence.
Track processing time and source sequence or version where available.
A watermark can determine when a preliminary window closes, but late events still need a correction path.
Record the affected decision snapshots when a late event changes historical state.
Do not rewrite an old recommendation as though its later correction had been known at the time.

Reconcile state to authoritative systems and physical counts.
Compare the set of open trailer visits, occupied doors, and completed unloads, not just row counts.
An equal number of trailers on each side can conceal different identities.
Investigate discrepancies by age and materiality, with named ownership and a resolution deadline.
For a major mismatch, suspend recommendations for the affected area while retaining the last trustworthy audit record.

### Step 6 — Couple the model without overstating what it knows

**Input:** quality-qualified state, model version, assumptions, and action constraints.
**Output:** reproducible scenarios and a recommendation with uncertainty and limitations.

Pass a frozen state snapshot into a model run.
Record snapshot ID, event-time cutoff, source status, model version, scenario settings, random seed where applicable, and run timestamp.
The run must not quietly combine a corrected new order list with old door occupancy unless the inconsistency is disclosed and acceptable.
When exact cross-system consistency is unavailable, report each source's as-of time and use a conservative rule for conflicting state.

Check model structure with operators before interpreting simulation output.
A model that omits required inspection or labor breaks can optimize an infeasible schedule.
Verify implementation separately from validation against historical and prospective observations.
Compare distributions and operationally relevant tails, not only average throughput.
Validate for the intended decision range; good historical fit does not establish that a new policy's causal effect is correct.
Use paired experiments and uncertainty intervals where appropriate; see Topics 04 and 05.

Publish a recommendation only if its inputs and constraints pass the gate.
Include eligible alternatives, the reason for the ranking, the state age, uncertainty, and a clear expiry time.
Expose material missing information rather than presenting spurious precision.
A small predicted difference may be too uncertain to justify changing the supervisor's existing plan.

### Step 7 — Set action rights and safety boundaries

**Input:** process ownership, hazard analysis, and the proposed action interface.
**Output:** an approval matrix and a documented degraded-mode procedure.

Separate read access, model administration, recommendation viewing, approval, and execution.
A developer who edits model parameters should not automatically have authority to dispatch equipment.
Start with human approval when the proposal affects operational decisions and the evidence is immature.
Define which actions are prohibited, which require supervisor approval, and which may be automated only after independent authorization.
The model cannot waive a safety rule, labor agreement, carrier commitment, or other binding constraint.

An approval record should identify the proposed action, actual action, person or service, time, data and model version, and override reason where supplied.
Avoid treating an override as proof the person was wrong.
Overrides may expose an omitted safety constraint, an unavailable worker, or a source-data defect.
Operators need a clear way to reject a recommendation without workarounds.

For any closed-loop extension, require a separate hazard and security review.
Define authorized commands, limits, acknowledgments, timeouts, duplicate-command handling, interlocks, and emergency procedures with the physical-system owner.
This educational architecture is not an industrial control design or a substitute for site safety engineering.
A safe state depends on the actual process; automatically stopping a process can itself be hazardous.

### Step 8 — Govern change and performance

**Input:** operational ownership, service objectives, validation evidence, and monitoring design.
**Output:** an approved release and review process.

Name a business owner for the decision and benefit measurement.
Name data owners for source semantics, a model owner for assumptions and validation, and a service owner for availability and incident response.
Security and privacy owners review access, retention, and exposure.
Site supervisors own acceptance of operational procedures.
One person may hold several roles in a small deployment, but each responsibility needs an explicit owner and deputy.

Track three distinct categories of measures.
Data measures include freshness, completeness, reconciliation differences, and contract failures.
Model measures include calibration, constraint violations, prediction errors, and uncertainty coverage where measurable.
Decision measures include adoption, overrides, service outcomes, and incremental value versus baseline.
A high adoption rate is not evidence of a correct model; users can comply with poor recommendations.
A low error rate is not proof of business value if it does not change a decision.

Specify review triggers before deployment.
A new door layout, major product mix change, labor rule change, or source-system migration may invalidate assumptions.
A sustained deterioration against a predefined threshold should open an investigation, not automatically retrain or replace the model.
Test changes in a representative environment, document their effects, and obtain approvals before promotion.
Keep a rollback to the prior validated release and its compatible data contract.

### Step 9 — Estimate full lifecycle costs and benefits

**Input:** alternative architectures, delivery plan, run model, and decision-value estimates.
**Output:** comparable cash flows, sensitivity ranges, and an investment gate.

Include discovery, process mapping, data cleanup, integration, model development, testing, validation, security, training, change management, and deployment.
Include internal labor at a stated economic cost, not only vendor invoices.
Include cloud or infrastructure, software licenses, data transfer, monitoring, support, incident response, model review, periodic revalidation, source changes, and business-user time.
Include expected replacement, decommissioning, and data-retention costs where material.
Do not count sunk past expenditure as an incremental cost of a new decision.
Do include shared costs that will actually change because of the project.

State whether prices and discount rates are nominal or real, the currency, the time period, and tax treatment.
Avoid presenting a point estimate as certainty.
Use conservative, base, and optimistic benefit assumptions tied to adoption, causal impact, and attainable operational capacity.
For capacity release, distinguish an accounting saving from additional capacity that may or may not be monetized.
Include the baseline's cost and benefit, and compare incremental cash flows.

A pilot can reduce uncertainty before full deployment.
Set a fixed budget, an evaluation design, and a stop rule before observing results.
If the value depends on an untested assumption, make that assumption a gate rather than treating it as a guaranteed saving.

### Step 10 — Operate, improve, and retire

**Input:** service runbook, business case, validation and incident history.
**Output:** continued authorization, remediation, scope change, or retirement.

At each review compare actual data quality, system costs, model performance, and decision outcomes with the approved thresholds.
Investigate a data-quality failure before attributing an error to model drift.
Revalidate after process redesign, parameter change, source migration, or a material performance shift.
Do not deploy an automatic recalibration merely because new data exist.
Check that the new data are representative and that the new parameters improve the intended use.

Define incident severity and escalation.
A missing optional visualization may have a different impact from an incorrect door-availability feed.
Record detection time, operational impact, fallback, recovery, and root cause.
Exercise recovery from retained events and snapshots without replaying physical actions.
Test restoration of service accounts, secrets, and configuration under organizational procedures.

Retire when the decision disappears, the source becomes unavailable, performance cannot be repaired economically, or the baseline is better.
Revoke access, stop scheduled jobs, archive evidence according to policy, and transfer responsibility for retained data.
Update runbooks and notify users of the replacement procedure.
Retirement is a normal lifecycle decision, not a failure to preserve sunk costs.

### Operational design: classify sources by consequence

Not all missing observations have the same effect.
Classify each variable by the decision it supports and the harm from an incorrect value.
This classification determines whether a missing value blocks a recommendation, narrows its scope, or merely lowers confidence.
It also determines who must be notified.

For a safety hold, an unknown value is not equivalent to permission to proceed.
The decision should be withheld until an authorized source confirms the hold status.
The physical operation still follows its own safety procedures.
The digital system does not become the final authority on whether equipment is safe.

For predicted unloading duration, a missing value may be handled with a conservative, validated distribution for a similar load type.
That substitution is acceptable only if the business owner agrees it does not mask a material difference.
Record the substituted parameter and identify the recommendation as based on an estimate.
If no defensible distribution exists, withhold the affected comparison.

For optional explanatory information, a missing field might not prevent a recommendation.
For example, a descriptive carrier name can be absent while a stable visit identifier remains available.
The interface should not invent the missing description.
The owner should still investigate a persistent data-quality defect.

Create a decision-specific matrix rather than a universal list of critical fields.
An order priority may be critical for selecting a trailer but unnecessary for counting occupied doors.
An estimated unload duration may matter for a 90-minute plan but not for confirming a current safety hold.
A model can be fit for one decision and unsuitable for another.

The input contract should specify the permitted degradation for each state component.
It should also specify the condition that returns the service to normal operation.
Recovery should require evidence that the source has caught up and reconciliation is complete.
Simply receiving one new event after an outage is not evidence that all missed events arrived.

### Operational design: snapshots and consistency

A snapshot has a decision time, a construction time, and one or more source cutoffs.
These are distinct quantities.
If the order system is current at 10:00 and the yard system at 09:55, the combined state is not necessarily the physical state at either time.
Label both source cutoffs and any known completeness gaps.

Suppose a trailer is assigned to a door at 09:57.
The order feed reflects the assignment at 10:00, but the yard feed still shows the trailer in the yard as of 09:55.
Both records can be internally correct for their own times.
Combining them without acknowledging the different cutoffs can create a false contradiction or a duplicate opportunity.

One policy is to wait for both sources to reach a common cutoff.
That improves temporal consistency but can make the resulting state too old for the decision.
Another policy is to use the latest acceptable value from each source with an explicit inconsistency check.
That can improve timeliness but requires careful rules for conflicting records.
Neither policy dominates in every operation.

Choose the policy by estimating the costs of delay and of inconsistent state.
Record the chosen cutoff and source ages in the decision audit.
When a source is late, do not silently replace its as-of time with the time the snapshot was assembled.
When a source is unavailable, do not label its last known value as current.

For material cross-source conflicts, identify an authoritative source or require human confirmation.
The authority can differ by field.
The yard system may be authoritative for gate arrival while the warehouse system is authoritative for a door assignment.
No single system necessarily owns every fact about a trailer.
Escalate unresolved conflicts rather than forcing an arbitrary winner solely to keep the model running.

### Operational design: event corrections and recovery

Consider an event describing a trailer arrival.
The original event contains the visit ID, reported time, producer, and source event ID.
The platform receives it, checks the contract, and updates its derived state.
If the same event arrives again, the state must not acquire a second arrival.
The audit can record the duplicate delivery without applying a second state transition.

If the producer later corrects the arrival time, the correction must be distinguishable from an accidental duplicate.
Use a version, correction identifier, or other agreed source mechanism.
Retain the original event and the correction when policy and law permit.
The current state should reflect the correction under the contract's rules.
The decision archive should retain the state actually available when a prior recommendation was made.

Replaying the event stream creates a new derived state; it does not recreate history.
A past supervisor could not have used a correction that arrived later.
Evaluate a historical decision using the information available at that time.
Evaluate data quality using both the as-known and corrected views.
This distinction prevents hindsight from making the model appear better informed than it was.

Recovery begins by containing action risk.
Pause affected recommendation publication or route users to the approved fallback.
Determine the last trusted snapshot, the affected sources, and the interval of possible loss.
Check whether the source supports a bounded replay, an authoritative snapshot, or both.
Restore data and state without emitting operational commands.

After replay, reconcile identities and statuses to the source systems.
Measure backlog age and confirm that all required source partitions have caught up.
Investigate mismatched records rather than treating a low average lag as proof of completeness.
Obtain the responsible owner's approval before resuming recommendations.
Document any period that remains uncertain.

Recovery exercises should include a duplicate, an out-of-order update, and a correction that changes a prior recommendation.
They should also include a source that cannot replay its missing events.
In that case, a physical count or authoritative current snapshot may be required.
Some historical decisions may remain unreconstructable.
The correct audit statement is that the historical state is unknown, not that it was recovered.

### Operational design: model release and validation gates

Separate changes to data mapping, model logic, parameters, and user interface.
Each can affect a decision even when the other components remain unchanged.
A source mapping change may turn a door marked unavailable into an apparently available door.
A parameter change may alter the preferred trailer without changing the state.
A user-interface change may hide a warning that previously prompted a supervisor to check a hold.

For every release, describe the change and its expected operational effect.
Record the source data version and the affected decisions.
State whether the change alters a constraint, assumption, model mechanism, or presentation.
Identify an independent reviewer appropriate to the risk and organizational size.
Do not let release documentation substitute for actual review.

Use historical cases for regression checks.
Include ordinary periods, peak periods, late events, holds, and known exceptions.
Test whether all recommended actions satisfy declared constraints.
For stochastic simulations, fix or record random seeds for reproducibility and assess variation across replications separately.
A fixed seed helps compare implementations but does not establish that the model represents uncertainty correctly.

Use prospective observation to assess whether outputs are timely and credible in the intended environment.
Initially run in a mode that does not alter the operation when appropriate.
Compare predictions with subsequent outcomes and review discrepancies with operators.
This is useful evidence about prediction, but it does not automatically identify the effect of an alternative policy.
Use an appropriate comparison design to evaluate the intervention itself.

Promotion requires an explicit decision by the accountable owner.
The release record includes the approved version, decision scope, unresolved limitations, and rollback trigger.
Maintain a compatible previous model, configuration, and data mapping where feasible.
A rollback that restores old code but cannot interpret the current source schema is not a usable rollback.
Test the full path in the designated preproduction environment.

### Operational design: benefits measurement

Map each claimed benefit to a decision and an observable transaction.
For expedited freight, link an urgent order to the shipment, expedite charge, and controllable operational cause.
Exclude the charge if the order would have been expedited regardless of the sequencing decision.
Exclude the charge if it was already avoided by the comparator.
Include only the part of a charge that the organization would actually avoid.

Some benefits are service improvements rather than immediate cash savings.
A shorter delay may improve customer experience without reducing an invoice in the current year.
Report the service measure separately if there is no defensible monetary conversion.
If a contract imposes a penalty that is actually avoided, document the relevant terms and likelihood.
Do not assign both a full theoretical customer-value estimate and the same avoided contractual penalty without checking overlap.

Additional throughput can have value if it supports real demand or avoids an otherwise necessary investment.
It has less immediate financial value when the facility has unused capacity and no avoidable cost.
An extra available labor hour becomes a cash saving only if staffing or paid overtime actually changes.
Otherwise report it as capacity released, with a separate utilization assumption.
Finance and operations should agree on the distinction before the pilot.

Use a comparison period or group that is as similar as practical.
Adjust for known differences in volume, product mix, staffing, and disruption exposure.
Where assignment is feasible, a randomized or otherwise well-designed comparison can improve causal interpretation.
Where it is not feasible, disclose the remaining confounding and avoid claiming precision the design cannot support.
An increase in model adoption is not a counterfactual.

Document the unit of analysis before collecting outcomes.
It might be a trailer visit, an order, a shift, or a site-week.
Avoid counting a single avoided shipment once for each order line it contains.
Check whether outcomes within a shift or site are correlated before treating them as independent observations.
Choose an analysis appropriate to the study design and consult statistical expertise where needed.

### Operational design: ownership and service agreement

The business owner accepts or rejects the decision scope.
The owner also decides whether a degraded service is still useful to operators.
The owner cannot delegate responsibility merely by purchasing a vendor platform.
The vendor may operate infrastructure under contract, but the business retains responsibility for its own operational decisions.

The source owner defines what a field means and when it changes.
The source owner also notifies consumers of proposed changes.
The platform owner monitors delivery and maintains the ingestion and state services.
These are different responsibilities even when a single team performs both.

The model owner maintains the mathematical and operational assumptions.
That owner documents validation evidence and proposes changes when the process changes.
An independent reviewer challenges whether the evidence supports the intended use.
The operations owner approves changes to local procedures and supervises fallback use.

Finance agrees how incremental costs and realized benefits will be measured.
Security reviews identities, privileges, network boundaries, and supplier access.
Privacy or legal specialists review personal data and applicable obligations.
The incident commander coordinates recovery when several owners are involved.
Every role needs a named primary and a workable escalation path.

A service agreement should identify operating hours and coverage.
If a night shift relies on the recommendation, weekday-only support may be inadequate.
Set measurable objectives for availability, freshness, and recovery that reflect business need.
Define the denominator and exclusions for each measure.
An uptime figure that excludes planned outages may be misleading if those outages occur during peak operations.

Specify the boundary of vendor support.
For example, the vendor may own the platform while the customer owns source data quality and local training.
Document who pays for source-system upgrades, new interfaces, and revalidation.
Clarify whether the customer can export model configuration and decision records at contract termination.
A low initial price can be expensive if essential maintenance and exit rights are excluded.

### Operational design: stop and retirement gates

A stop rule is more credible when agreed before a pilot produces attractive demonstrations.
Set it around the decision, not around visual completeness.
An example is failure to obtain a critical hold feed or equivalent authorized confirmation.
Another is inability to reconcile enough visits to estimate whether the model changes outcomes.
The threshold must be selected for the local risk and decision, not copied blindly from an example.

Pause deployment if operators cannot understand the recommendation or use the fallback.
Pause if model changes bypass approval or if a source change invalidates a critical constraint.
Pause if the output suggests unsafe behavior, even when the computed economic benefit is high.
Investigate whether the issue is data, model structure, interface, or process ownership.
Resume only after the responsible owners accept the corrective evidence.

An economic stop can be appropriate even when the software functions correctly.
If the improved manual baseline produces nearly the same result for much less cost, choose that baseline.
If the decision occurs too infrequently to recover ongoing integration cost, use an offline study.
If benefits depend on actions that the organization cannot authorize, revise the decision scope rather than counting unreachable savings.
If a critical supplier will not share necessary data, reassess whether a less connected approach answers the question.

Before retirement, inventory the data, interfaces, users, and downstream consumers.
Provide a replacement procedure and a cutover date.
Disable access and automated jobs under the organization's change process.
Retain or delete records according to approved retention and legal requirements.
Assign an owner for archived evidence and unresolved incidents.

### Operational design: communicating uncertainty

The interface should distinguish observed facts from estimates and recommendations.
For each material estimate, identify its time horizon and relevant uncertainty.
Show when a value is missing, stale, corrected, or manually confirmed.
Use clear operational language rather than a single unexplained confidence score.
Make the basis of an alternative recommendation accessible to the supervisor.

Suppose two trailer sequences have projected completion times of 88 and 89 minutes.
If plausible variation is several minutes, the one-minute difference may not justify disruption.
The system can present both as practically similar rather than asserting that the first is certain to be better.
That conclusion depends on a properly estimated comparison, not a visual rule of thumb.
An operator may select the second because of a constraint not represented in the model.

Record the override reason as a useful category when practical.
Examples include an unreported safety hold, labor unavailability, carrier commitment, or local process knowledge.
Allow free text only under an appropriate privacy and retention policy.
Review patterns with users to identify omitted constraints and confusing presentation.
Do not treat override count as a standalone performance target.

When presenting business value, show the baseline and the range of plausible effects.
Separate measured outcomes from projections and targets.
State whether estimates reflect full adoption or observed adoption.
Identify which costs are incremental and which are shared with other systems.
That presentation makes the decision contestable and correctable.

## 6. Worked example

### Business setting

Fictional North Quay Distribution operates 18 inbound doors at one regional distribution center.
Supervisors select an eligible trailer for each free door approximately every 15 minutes.
The next 90 minutes of unloading affect the availability of urgent store orders.
The facility currently uses an appointment board, a warehouse management system, and radio confirmation.
A supervisor can change the sequence but cannot waive safety checks or carrier agreements.

Management proposes a decision-support twin, not an automatic controller.
The recommendation is a suggested plan; it is not a commitment to a carrier or a command to equipment.
The demand forecast estimates expected store needs; it is not a target.
The target is a management goal for fewer urgent-order delays, not a guaranteed saving.
The actual commitment is made only through the existing authorized operational process.

The baseline is an improved daily priority list using existing systems.
Both the baseline and twin require supervisors to verify trailer eligibility.
The incremental question is whether the connected model's more current state and scenario ranking are worth more than the daily list.
All monetary figures below are illustrative US dollars (USD), nominal, before tax, and incremental to that baseline.

### State and source design

| Decision variable | Source | Update treatment | Main risk |
|---|---|---|---|
| Trailer visit and arrival | Yard system | Event feed, periodic reconciliation | Reused trailer ID. |
| Door occupancy | Warehouse management system | Five-minute extraction and event updates where available | Door release reported late. |
| Urgent order priority | Order system | Five-minute extraction | Changed priority not propagated. |
| Expected unloading duration | Historical unload records | Weekly parameter review | New load mix invalidates fit. |
| Available trained labor | Shift roster and supervisor confirmation | At shift start and on material change | Roster does not capture absence. |
| Inspection or safety hold | Site hold register | Before recommendation and at action | Missing hold creates infeasible plan. |

The compound visit key is site ID plus visit ID.
Trailer plate or equipment number is an attribute, not the sole primary key.
Each event carries its source, source event ID, event time with time zone, ingestion time, and applicable version.
A supervisor confirms labor and holds because those states are not reliably observed by the feed.

### Freshness budget

The team chooses an operational maximum of 10 minutes from a door-status change to a supervisor's use of the recommendation.
This is a design assumption for the pilot, not a universal standard.
The budget is: up to 5 minutes for periodic source sampling, 1 minute for ingestion and transformation, 2 minutes for computation and presentation, and 2 minutes for supervisor response.
The sum is 5 + 1 + 2 + 2 = 10 minutes.
These are design bounds, not observed percentile measurements.
A traced end-to-end latency distribution must be measured in the pilot.

If a status change occurs just after a scheduled extraction, it can wait almost 5 minutes for sampling.
An 8-minute-old source snapshot does not automatically fit a 10-minute use-time budget when computation and response still remain.
If the remaining budget is 4 minutes, the source age must be no more than 6 minutes at run time.
If the data exceed that limit or a hold source is unavailable, the affected recommendation is withheld.
The supervisor uses the existing priority list and local confirmation instead.
A fresh source timestamp alone is not proof that its contents are complete.

### Architecture and governance

Events enter an ingestion layer with duplicate detection and source-contract checks.
The state builder reconciles trailer visits and door status and retains source-specific as-of times.
The model evaluates only eligible trailers and available resources against the next 90 minutes.
The interface shows recommendation, alternatives, constraints, input ages, and expiration.
The supervisor accepts or overrides; no command goes directly to a door or vehicle.

The warehouse operations manager owns the decision and fallback.
The yard-system owner owns event semantics and escalation for missing visits.
The model owner maintains duration assumptions and validation records.
The platform owner maintains access, recovery, monitoring, and incident response.
Finance owns the economic measurement method jointly with operations.
A change in appointment rules triggers review of both source mapping and model constraints.

### Benefits and costs

Suppose the improved daily list would avoid 40 urgent-order expedites per year relative to the old practice.
The twin is estimated to avoid another 90 expedites per year relative to the daily list.
The expected avoidable cost is USD 1,000 per expedite, giving USD 90,000 per year incremental gross benefit.
This count is an assumed incremental causal effect, not total post-launch expedites and not a measured result.
The estimate already reflects expected adoption and feasible operational response.
Do not multiply by adoption again unless the estimate is redefined as a full-adoption effect.

Assume implementation costs USD 130,000 at time zero.
This includes USD 30,000 discovery and data cleanup, USD 45,000 integration, USD 25,000 model and validation, USD 15,000 security and training, and USD 15,000 deployment and contingency.
The arithmetic is 30,000 + 45,000 + 25,000 + 15,000 + 15,000 = USD 130,000.
The daily-list baseline's own implementation is excluded only because these figures are explicitly incremental to it.

Assume recurring incremental cost is USD 45,000 per year for three years.
It comprises USD 12,000 cloud and software, USD 15,000 support and incident response, USD 10,000 data and model maintenance, and USD 8,000 operational review and training.
The sum is USD 45,000 per year.
A USD 10,000 incremental retirement and retention cost is due at the end of year 3.
If a recurring cost already includes a service fee, do not add the same fee again elsewhere.

The annual incremental net benefit before retirement is USD 90,000 − USD 45,000 = USD 45,000.
At a nominal 10% annual discount rate, year-end discount factors are approximately 0.9091, 0.8264, and 0.7513.
The net present value (NPV) is:

```text
NPV = -130,000 + 45,000/1.10 + 45,000/1.10² + 35,000/1.10³
    = -130,000 + 40,909.09 + 37,190.08 + 26,296.02
    = -25,604.81 USD, approximately
```

The year-3 net amount is USD 35,000 because it includes the USD 10,000 retirement cost.
Undiscounted operating payback would be 130,000 / 45,000 = 2.89 years if annual benefits accrued evenly and retirement were ignored.
That simplified payback does not incorporate the terminal cost, timing uncertainty, or discounting.
Under the stated three-year discounted case, the twin does not pay for itself.

The break-even gross annual benefit is approximately USD 100,296.
It solves `-130,000 + (B − 45,000) × (1/1.10 + 1/1.10² + 1/1.10³) − 10,000/1.10³ = 0`.
At USD 1,000 per avoidable expedite, that is about 101 incremental expedites per year when rounded up to whole expedites.
The base assumption of 90 incremental expedites per year falls below that threshold.
The appropriate decision is not to approve the full twin on this evidence.
A bounded pilot may be justified only if its cost is separately approved and the value of learning can change the decision.

### Sensitivity and decision gate

If incremental benefits fall to USD 60,000 per year, annual net benefit is USD 15,000 and the investment is much less attractive.
If they rise to USD 120,000 per year, annual net benefit is USD 75,000 and NPV becomes approximately USD 49,001.
A high-benefit scenario requires evidence of at least 120 genuinely avoidable expedites per year at the assumed unit cost, not merely 120 recommendations.
The benefit estimate should exclude delays caused by upstream shortages that sequencing cannot prevent.
A site-wide reduction after launch is not necessarily caused by the twin; seasonality and other improvement programs can explain it.

A pilot gate can require a representative set of shifts, a comparison with the daily-list baseline, and independent review of missing-data rates.
Agree sample size or precision criteria with an analyst before running the evaluation.
Predefine how disrupted shifts, overrides, and source outages are treated.
If benefits cannot be identified credibly or critical feeds miss their freshness requirements, stop rather than extrapolating a favorable demonstration.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It compares the same three-year cash flows, computes the break-even annual gross benefit, and checks an illustrative freshness budget.
It is an economic and architecture worksheet, not a validated production implementation or a statistical evaluation.
Inputs are incremental to the daily-list baseline; all amounts are nominal USD.
Yearly net benefits and the retirement cost occur at year end, and implementation occurs at time zero.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BusinessCase:
    implementation: float
    annual_gross_benefit: float
    annual_operating_cost: float
    retirement_cost: float
    discount_rate: float
    years: int

    def npv(self) -> float:
        value = -self.implementation
        for year in range(1, self.years + 1):
            cash = self.annual_gross_benefit - self.annual_operating_cost
            if year == self.years:
                cash -= self.retirement_cost
            value += cash / (1 + self.discount_rate) ** year
        return value

    def break_even_gross_benefit(self) -> float:
        annuity = sum(
            1 / (1 + self.discount_rate) ** year
            for year in range(1, self.years + 1)
        )
        retirement_pv = self.retirement_cost / (1 + self.discount_rate) ** self.years
        return self.annual_operating_cost + (self.implementation + retirement_pv) / annuity


def state_usable(
    source_age_minutes: float,
    compute_minutes: float,
    response_minutes: float,
    maximum_age_minutes: float,
    hold_source_available: bool,
) -> bool:
    """Check a simple deterministic budget; not a percentile estimate."""
    return (
        hold_source_available
        and source_age_minutes >= 0
        and source_age_minutes + compute_minutes + response_minutes
        <= maximum_age_minutes
    )


case = BusinessCase(
    implementation=130_000,
    annual_gross_benefit=90_000,
    annual_operating_cost=45_000,
    retirement_cost=10_000,
    discount_rate=0.10,
    years=3,
)
print(f"Base NPV: USD {case.npv():,.2f}")
print(f"Break-even gross benefit: USD {case.break_even_gross_benefit():,.2f}/year")
for source_age in (5, 8):
    usable = state_usable(source_age, 2, 2, 10, True)
    print(f"Source age {source_age} min: {'use' if usable else 'fallback'}")
```

Expected output, allowing for rounding:

```text
Base NPV: USD -25,604.81
Break-even gross benefit: USD 100,296.07/year
Source age 5 min: use
Source age 8 min: fallback
```

The second age check fails because 8 + 2 + 2 = 12 minutes exceeds the 10-minute maximum.
A true result checks only the supplied arithmetic and hold-source flag.
Production systems also need completeness, identity, authorization, model validity, timestamp integrity, and source-specific quality checks.
The code does not infer whether the benefit is causal; it consumes an assumption.
Use sensitivity analysis and an evaluation design to establish whether that assumption is defensible.

## 8. Data, standards, and regulatory context

A source contract should distinguish syntax, semantics, operational expectations, and rights.
Syntax specifies field names, data types, and permitted formats.
Semantics specifies whether `arrival_time` means gate arrival, appointment time, or unloading start.
Operational expectations cover availability, change notice, latency measurement, and incident contacts.
Rights cover authorized access, permitted purposes, retention, transfer, and deletion.
These layers are related but none substitutes for the others.

For identity, GS1 provides standards for identifying parties, locations, products, and logistics units.
Its Electronic Product Code Information Services (EPCIS) standard provides a common way to capture and share visibility events.
Use the appropriate GS1 identifiers and event semantics only when the business process and trading partners support them.
Adopting EPCIS does not guarantee a complete or correct physical state, nor does it require a digital twin.
Internal trailer-visit identifiers may still be needed to distinguish repeated visits.
See the GS1 source in Section 16 for the current specification.

For enterprise and control integration, the ISA-95 series describes models and terminology for enterprise-control system integration.
It can help clarify the boundary between business planning, manufacturing operations, and control functions.
It is not a universal requirement to automate a warehouse and does not itself authorize a control command.
The relevant edition and contractual or sector requirements must be checked for the deployment.
Similarly, the ISO 23247 series addresses digital twin frameworks for manufacturing; it is a useful architecture reference, not proof that a particular implementation is valid.

For security, the National Institute of Standards and Technology (NIST) Cybersecurity Framework 2.0 organizes cybersecurity outcomes around governance and risk management functions.
NIST Special Publication 800-82 Rev. 3 discusses operational technology security and the distinct safety and availability needs of such environments.
Neither document automatically makes an implementation safe or legally compliant.
Segmentation, least-privilege access, logging, supplier access, recovery, and patching need local engineering and governance.
A recommendation system connected only to enterprise data still creates risk if its outputs are trusted for physical actions.

Personal data can appear in worker rosters, driver identities, user activity, and override logs.
The EU General Data Protection Regulation (GDPR), where applicable, imposes obligations including data minimization and security; applicability depends on facts and jurisdiction.
As of November 2026, use the official consolidated legal text and current local guidance rather than assuming every workforce dataset is exempt.
Do not infer that a technical standard provides a legal basis to process personal data.
Define purpose, access, retention, and worker-notice requirements with privacy and legal specialists.
Do not use the twin as a covert worker-ranking tool merely because the event trail permits it.

Record timestamps with an explicit time zone and daylight-saving handling.
Store enough information to reproduce the time interpretation; UTC timestamps and source-local context are often useful together.
Late corrections and retention duties can conflict if historical states are overwritten without an audit trail.
Retention is not a blanket instruction to keep all raw data forever.
Determine what is necessary, contractually permitted, and legally required before designing the event archive.

Contracts and standards do not remove the need for local acceptance tests.
Test a reused trailer ID, a duplicate event, a late cancellation, a site time-zone transition, a missing labor roster, and a source schema change.
Check not only that messages parse, but that the resulting state and decision remain meaningful.

## 9. Industry practice and case evidence

### Evidence hierarchy

Public reports frequently describe a platform, an intended use, or a business initiative without providing a controlled estimate of incremental value.
Treat a vendor demonstration or company announcement as evidence of what was claimed or deployed, not as independent proof of a causal benefit.
Look for a defined baseline, period, scope, method, exclusions, and independent evaluation before transferring a reported percentage to another facility.
A named technology supplier has an incentive to emphasize successes.
A missing cost or failed pilot is not evidence that either was zero.

### DHL and Tetra Pak warehouse case

DHL and Tetra Pak publicly described a digital twin implementation in a Singapore warehouse in a 2019 DHL announcement.
The announcement describes an operational digital representation and integration of warehouse data for monitoring and planning.
This is a company-reported case, not an independent randomized evaluation of the value of synchronization.
It demonstrates that warehouse digital-twin architectures have been publicly reported in practice.
It does not establish that a particular cadence, business benefit, or model design will transfer to North Quay.

**Inference:** a site considering a comparable system must inventory the systems behind its representation and budget for their ongoing maintenance.
That inference follows from the architecture described in this topic; it is not a claim that the announcement disclosed a complete lifecycle cost.
The appropriate next step is to request the actual decision, baseline, integration scope, and operating-cost evidence, subject to confidentiality.

### Port of Rotterdam digital twin initiative

The Port of Rotterdam Authority has publicly described work on a digital twin of the port and use of digital data for port operations.
A port-scale representation involves different geography, stakeholders, and information rights from a single distribution center.
The public initiative is evidence of an organizational use case, not evidence that a small warehouse needs a port-scale platform.
Do not infer a universal return on investment from a public strategic announcement.
Ask which specific navigation, planning, or asset decisions are supported and what data are observable to each participant.

**Inference:** a multi-organization twin requires especially explicit data ownership and access agreements because the organizations do not share a single operational authority.
The cited public sources should be consulted for their actual scope and date before making any procurement claim.

### What to request from a supplier

Request a source-by-source inventory and an end-to-end latency measurement method.
Ask for an example of a stale-data incident and the documented fallback, with sensitive details removed.
Request a demonstration of replay that does not reissue physical commands.
Ask who approves model changes and how a previous release is restored.
Ask for references that distinguish one-time integration fees from annual support and customer labor.
Request evidence of outcomes against an agreed baseline rather than a visualization or an uptime figure alone.
Verify independently whether the cited deployment resembles your decision, site, and data environment.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No decision owner | Dashboard has users but no accountable approver. | Outputs are not tied to an action. | Integration cost without realized value. | Name owner and decision before build. |
| Wrong baseline | Business case compares with no process. | Existing manual improvements are ignored. | Inflated incremental return. | Compare with best feasible alternative. |
| Stale door status | Source-age tail grows during busy periods. | Delayed status is treated as current. | Recommendation sends a trailer to an occupied door. | Gate by age and use fallback. |
| Missing holds | Safety register feed is unavailable. | Feasibility constraints are incomplete. | Unsafe or prohibited recommendation. | Require hold confirmation; withhold on failure. |
| Duplicate events | Arrival counts exceed physical visits. | At-least-once delivery is applied twice. | Incorrect queues and workload. | Stable event keys and idempotent processing. |
| Late corrections | Old states change without explanation. | Reconstructed history overwrites what was known. | Audit and evaluation are misleading. | Preserve as-known and corrected views. |
| ID collisions | One trailer appears at two sites. | Equipment ID is used as visit key. | Wrong entity receives an action. | Namespace and visit identity. |
| Semantic change | Values remain valid but behavior shifts after source release. | Field meaning changes without schema change. | Silent model corruption. | Contract tests and change notification. |
| Unvalidated intervention | Historical averages fit but new-policy outcomes diverge. | Calibration is mistaken for causal validation. | Wrong investment or sequencing decision. | Prospective evaluation and decision-scope limits. |
| Over-automation | Operator cannot reject a recommendation. | Decision rights are embedded in software. | Unsafe or unaccountable action. | Explicit approvals and independent control review. |
| Override suppression | Override rate falls while complaints rise. | Users avoid recording disagreement. | Hidden model defects and lost trust. | Low-friction override and review. |
| Unfunded support | No one owns a source outage on a weekend. | Project ends at deployment. | Long outages and unmanaged fallback. | Fund service ownership and escalation. |
| Duplicate benefit | Same shipment counted in freight and service categories without proof. | Benefit accounting overlaps. | Inflated NPV. | Transaction-level benefit reconciliation. |
| False precision | Rankings differ by seconds despite broad uncertainty. | Single-point estimates hide variability. | Unnecessary policy changes. | Show uncertainty and meaningful thresholds. |
| Insecure integration | Shared service accounts have broad permissions. | Data and control privileges are mixed. | Unauthorized access or action. | Least privilege and separate identities. |
| Unrehearsed recovery | No successful restore has been demonstrated. | Event log and snapshot dependencies are unknown. | Prolonged outage or accidental replay. | Exercise restore without reissuing actions. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Supervisor can change order. | Recommendation must affect an actionable decision. | Carrier or safety rule fixes the sequence. | Observe shifts and inspect constraints. |
| Eligible-load state is observable. | Model must compare feasible trailers. | Holds or arrivals are missing. | Reconcile with physical checks. |
| Ten-minute age is acceptable. | Freshness gate needs a decision-specific limit. | Door status changes faster than response. | Study state changes and decision outcomes by age. |
| Source timestamps are meaningful. | Age calculation uses event times. | Clock drift or backdated entry. | Compare source, platform, and physical observations. |
| Five-minute extraction is achievable. | Budget assumes an upper sampling interval. | Source load or scheduled maintenance delays jobs. | Trace tail delay under peak load. |
| Labor availability is known. | Capacity affects feasible sequences. | Absence or reassignment is not reported. | Compare roster with supervisor confirmation. |
| Duration history is relevant. | Model uses historical unload times. | New product mix or process redesign. | Segment and validate against recent observations. |
| Effect is incremental to daily list. | NPV compares alternatives. | Benefit estimate uses the old process as baseline. | Review evaluation and transaction attribution. |
| USD 1,000 per expedite is avoidable. | Gross benefit converts counts to cash. | Costs are fixed or borne by another party. | Review invoices and contracting terms. |
| Three-year horizon is suitable. | Discounted comparison needs a common horizon. | System replacement or benefit lasts longer. | Compare scenarios and document terminal costs. |
| Costs include internal time. | Economic comparison requires full incremental costs. | Support or user time is unrecorded. | Sample time and reconcile with budgets. |
| Model outputs are understood. | Supervisors need informed discretion. | Interface hides staleness and limitations. | Observe use and interview operators. |

The financial example excludes taxes and any residual value by assumption.
It does not estimate risk-adjusted cost of capital for a particular firm.
The calculated NPV is not a confidence interval and should not be treated as a measured outcome.
The tutorial does not design a safety-critical controller or determine legal obligations for a specific operation.

## 12. Exercises

### Exercise 1 — Decision and boundary

**Time:** 15 minutes.
A retailer wants a live three-dimensional replica to study whether to close one of five distribution centers next year.
The proposed replica has no owner for the closure decision and uses daily inventory snapshots.
**Deliverable:** a one-paragraph decision statement and an alternative to the replica.
**Acceptance:** name owner, alternatives, horizon, relevant constraints, baseline, and a reason a continuously synchronized model may not be necessary.

### Exercise 2 — Source contract

**Time:** 20 minutes.
A yard feed reports `trailer_id`, `status`, and a timestamp without a time zone.
Duplicate messages occur, and the same trailer can visit twice in a day.
**Deliverable:** a contract table for at least six fields or operational obligations.
**Acceptance:** cover visit identity, event identity, event time and zone, ingestion time, status semantics, correction and duplicate rules, and an owner.

### Exercise 3 — Freshness arithmetic

**Time:** 15 minutes.
A decision must use door state no more than 12 minutes old.
At model start the source snapshot is 7 minutes old; computation takes 2 minutes and user response takes 4 minutes.
**Deliverable:** calculate age at action and recommend use or fallback.
**Acceptance:** state the 13-minute total, compare it with 12 minutes, and specify what happens if the gate fails.

### Exercise 4 — Delayed correction

**Time:** 25 minutes.
At 10:00 a supervisor receives a recommendation based on a trailer's reported arrival at 09:50.
At 10:15 a correction says the actual arrival was 10:05.
**Deliverable:** a sequence of state, audit, and reconciliation actions.
**Acceptance:** preserve what was known at 10:00, apply the correction to corrected history, identify affected recommendations, and avoid reissuing physical actions during replay.

### Exercise 5 — Governance

**Time:** 20 minutes.
A software vendor proposes that its administrator modify model parameters and send dispatch commands during an outage.
**Deliverable:** an approval matrix with at least four roles and a degraded-mode procedure.
**Acceptance:** separate model modification from execution authority; name business, data, model, and service owners; define a fallback and audit record.

### Exercise 6 — Investment calculation

**Time:** 25 minutes.
Use the worked case, but assume USD 120,000 annual incremental gross benefits and unchanged costs and timing.
**Deliverable:** calculate annual net benefit and three-year NPV at 10%.
**Acceptance:** include the year-3 retirement cost and state whether the NPV is positive; show the equation.

### Exercise 7 — Validation and maintenance

**Time:** 30 minutes.
A warehouse changes from pallet unloading to mixed-case unloading, and a new management system changes door-release semantics.
**Deliverable:** a revalidation and release plan.
**Acceptance:** include source-contract checks, affected model parameters and constraints, representative comparison data, approval, rollback, monitoring, and a condition for suspending recommendations.

### Exercise 8 — Build or not build

**Time:** 25 minutes.
A planner makes one annual network decision; inputs are monthly, changes require executive approval, and a spreadsheet sensitivity study costs one-tenth as much as the proposed connected twin.
**Deliverable:** a recommendation and evidence gate.
**Acceptance:** compare alternatives and full lifecycle cost, identify what additional recurring decision could justify synchronization, and explain what evidence would change the conclusion.

## 13. Self-check answers

### Exercise 1

A strong statement identifies an executive or network-planning owner who chooses whether to close a center for the next planning horizon.
Alternatives include retaining all five centers or closing each feasible candidate, subject to service, lease, labor, and capacity constraints.
A baseline is the current network or another feasible network plan, depending on the actual decision.
An offline scenario model using validated demand and cost assumptions can compare alternatives without a continuous feed.
A live representation might become worthwhile only if another recurring decision depends on current state and justifies the additional costs.
A diagram alone does not answer the closure decision.

### Exercise 2

A satisfactory contract contains a site-scoped visit ID, not just a reusable equipment ID.
It also contains a producer-scoped event ID and version so repeated delivery and corrections can be distinguished.
The source must define event time, time zone or offset, and the meaning of each status transition.
Ingestion time makes transport delay observable; it does not replace the source event time.
The contract states how cancellations and corrected events refer to prior records.
It names a producer owner, a consumer owner, quality thresholds, and escalation arrangements.
An answer that adds only a timestamp format but ignores identity and corrections is incomplete.

### Exercise 3

The age at action is 7 + 2 + 4 = 13 minutes.
That exceeds the 12-minute maximum by 1 minute.
Withhold the affected recommendation and use the approved manual procedure or obtain a sufficiently current state.
Do not claim that a 7-minute-old snapshot is safe without accounting for the remaining decision path.
This calculation is a deterministic example; actual tail delay must be measured.

### Exercise 4

Retain the 10:00 snapshot, its source cutoff, and the recommendation actually made using information then available.
Apply the later correction to a corrected event history with an explicit correction version and arrival time.
Identify analyses and decisions affected by the changed arrival; notify the owner if operational action remains relevant.
Reconstructing state is not permission to resend a historical dispatch command.
A valid audit can answer both what was believed at 10:00 and what is now believed about that time.

### Exercise 5

A strong matrix assigns the business owner to approve scope and fallback, the data owner to define source meaning, the model owner to approve model assumptions, and the service owner to operate the platform.
An authorized site supervisor or separately governed control owner retains dispatch authority.
The vendor administrator's permissions are limited to approved technical tasks and do not silently include operational command rights.
During an outage, the supervisor uses the documented manual procedure and records any material action under existing rules.
The audit records actor, time, version, decision, and outcome where available.
Assess the specific hazard before authorizing any automated control.

### Exercise 6

Annual net benefit before retirement is USD 120,000 − USD 45,000 = USD 75,000.
Year 3 net cash flow is USD 65,000 after the USD 10,000 retirement cost.
NPV = −130,000 + 75,000/1.10 + 75,000/1.10² + 65,000/1.10³.
The result is approximately positive USD 49,001.
A positive point estimate does not prove that the assumed incremental benefit is achievable.

### Exercise 7

Award credit for isolating both the data-contract change and the physical-process change.
The team should test release-event semantics, identity, timing, and reconciliation in a representative environment.
It should re-estimate or review unload durations, validate against representative mixed-case observations, and review constraints with operators.
Release requires named approval, a compatible rollback, and monitoring of state quality and decision outcomes.
Suspend affected recommendations when a critical source is incompatible or validation fails; use the approved fallback.
A plan that simply retrains on all available data misses source semantics, approval, and safe operation.

### Exercise 8

A strong answer initially favors the spreadsheet or offline model because the decision is annual and the inputs are monthly.
Compare full incremental costs, decision quality, and constraints rather than assuming the lower purchase price is the only consideration.
A recurring allocation decision with fast-changing, observable state and material avoidable losses could create a case for synchronization.
Require evidence that timely state changes the decision and yields incremental benefits beyond the offline alternative.
Retain a stop rule if the proposed data cannot be obtained or the expected value remains below lifecycle cost.

## 14. Interview questions

1. **When is a live twin unnecessary?**
   A strong answer starts with decision cadence, feasible alternatives, and incremental value.
   A weak answer says a live twin is always better because it has more data.
2. **How would you select refresh cadence?**
   A strong answer budgets source, processing, computation, and response delays against decision-specific staleness.
   A weak answer chooses a one-second refresh without considering data latency or actionability.
3. **What is the difference between event time and ingestion time?**
   A strong answer explains late and out-of-order events and how both timestamps support audit.
   A weak answer assumes messages arrive in the order physical events occurred.
4. **How do you recover from a late correction?**
   A strong answer preserves the as-known snapshot, updates corrected history, and avoids duplicate physical actions.
   A weak answer silently overwrites the past and replays all commands.
5. **What makes an event pipeline idempotent?**
   A strong answer uses stable event identity and ensures repeat processing does not repeat effects.
   A weak answer assumes duplicates never happen.
6. **How would you validate a recommendation?**
   A strong answer separates data quality, implementation verification, model validation, feasibility, and prospective business evaluation.
   A weak answer relies only on a visually convincing animation.
7. **What should happen when a critical source is stale?**
   A strong answer names a decision-specific gate, operator warning, and approved fallback.
   A weak answer replaces missing values with zeros and proceeds.
8. **What belongs in a lifecycle business case?**
   A strong answer includes internal labor, integration, validation, ongoing support, changes, retirement, and incremental benefits versus a baseline.
   A weak answer counts only subscription fees and gross savings.
9. **How do you interpret overrides?**
   A strong answer treats them as evidence requiring contextual review, not automatic errors.
   A weak answer penalizes all overrides to improve adoption.
10. **What changes when the system sends commands?**
    A strong answer requires explicit control authority, hazard analysis, limits, acknowledgments, security, and safe degraded operation.
    A weak answer treats a recommendation API as sufficient authorization.

## 15. Key terms

- [Baseline](#4-definitions-and-vocabulary): feasible comparator for incremental value.
- [Cadence](#4-definitions-and-vocabulary): frequency of observation, refresh, run, or decision.
- [Closed loop](#4-definitions-and-vocabulary): authorized action flow back to operations.
- [Data contract](#4-definitions-and-vocabulary): agreement on meaning, delivery, quality, and ownership.
- [Decision](#4-definitions-and-vocabulary): choice among feasible actions.
- [Decision-support twin](#4-definitions-and-vocabulary): maintained representation for recurring decisions under this topic's convention.
- [Entity](#4-definitions-and-vocabulary): identified object in the modeled system.
- [Event](#4-definitions-and-vocabulary): report of a business or physical occurrence.
- [Event time](#4-definitions-and-vocabulary): time the occurrence is said to have happened.
- [Fail-safe](#4-definitions-and-vocabulary): specified response to a critical failure.
- [Freshness](#4-definitions-and-vocabulary): information age at a specified point.
- [Idempotency](#4-definitions-and-vocabulary): repeat processing without extra effect.
- [Incremental benefit](#4-definitions-and-vocabulary): gain beyond the selected alternative.
- [Latency](#4-definitions-and-vocabulary): elapsed time between specified stages.
- [Model boundary](#4-definitions-and-vocabulary): scope of represented processes.
- [Model drift](#4-definitions-and-vocabulary): changed relationship between model and operation.
- [Net present value](#4-definitions-and-vocabulary): discounted incremental benefits less costs.
- [Override](#4-definitions-and-vocabulary): authorized departure from a recommendation.
- [Physical system](#4-definitions-and-vocabulary): real operation represented by the model.
- [Processing time](#4-definitions-and-vocabulary): time a platform handles an event.
- [Provenance](#4-definitions-and-vocabulary): record of sources and transformations.
- [Reconciliation](#4-definitions-and-vocabulary): comparison and correction against an authoritative reference.
- [Replay](#4-definitions-and-vocabulary): reprocessing retained events to reconstruct state.
- [Service-level objective](#4-definitions-and-vocabulary): internal measurable service target.
- [Shadow](#4-definitions-and-vocabulary): observation-fed representation without direct actuation in this convention.
- [Simulation](#4-definitions-and-vocabulary): executable representation for exploring behavior.
- [Snapshot](#4-definitions-and-vocabulary): versioned state at a defined time.
- [State](#4-definitions-and-vocabulary): variables describing the modeled system at a time.
- [State estimate](#4-definitions-and-vocabulary): reconstructed, uncertain view of state.
- [Total cost of ownership](#4-definitions-and-vocabulary): complete lifecycle expenditure.
- [Validation](#4-definitions-and-vocabulary): evidence of adequacy for a specified use.
- [Verification](#4-definitions-and-vocabulary): evidence that implementation follows specification.
- [Watermark](#4-definitions-and-vocabulary): policy estimate of event-time progress.

## 16. Sources

1. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis), specification and implementation resources. Consult the applicable version for event and identifier semantics.
2. International Society of Automation, [ISA-95 enterprise-control system integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard). Describes the series and its integration scope.
3. International Organization for Standardization, [ISO 23247-1:2021, Automation systems and integration — Digital twin framework for manufacturing — Part 1: Overview and general principles](https://www.iso.org/standard/75066.html). Manufacturing-oriented framework, not a certification of a given model.
4. NIST, [Cybersecurity Framework 2.0](https://www.nist.gov/cyberframework), 2024. Cybersecurity risk management framework, not a substitute for site-specific controls.
5. NIST, [SP 800-82 Rev. 3, Guide to Operational Technology (OT) Security](https://csrc.nist.gov/pubs/sp/800/82/r3/final), 2023. Operational technology security guidance.
6. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text; applicability depends on the facts and current law.
7. DHL, [DHL Supply Chain partners Tetra Pak to implement its first digital twin warehouse in Asia Pacific](https://group.dhl.com/en/media-relations/press-releases/2019/dhl-supply-chain-partners-tetra-pak-implement-first-digital-twin-warehouse-asia-pacific.html), 2019. Company-reported deployment; not independent evidence of causal return.
8. Port of Rotterdam, [Annual report 2019 highlights](https://www.portofrotterdam.com/sites/default/files/2021-06/annual-report-highlights-port-of-rotterdam-2019_0.pdf). Port authority account of its digital transformation initiative.
9. Sargent, R. G., “Verification and validation of simulation models,” *Journal of Simulation*, 7, 12–24, 2013. [DOI: 10.1057/jos.2012.20](https://doi.org/10.1057/jos.2012.20). Distinguishes verification and validation of simulation models.
10. NIST, [Framework for Improving Critical Infrastructure Cybersecurity](https://doi.org/10.6028/NIST.CSWP.29), version 2.0, 2024. Primary framework publication.

Sources establish definitions or report specific initiatives; they do not validate the fictional worked example.
Regulatory information is educational, not legal advice.

## 17. Links

- [Module 20 index](_index.md).
- [Twin versus shadow versus simulation](01_twin-shadow-simulation.md).
- [Discrete-event simulation](02_discrete-event-simulation.md).
- [Validation](04_validation.md).
- [Experiment design](05_experiment-design.md).
- [Module 20 survey](../../../course/20_digital_twins_simulation_optimization_uncertainty.md).
- [Simulation and optimization production labs survey](../../../course/30_simulation_optimization_production_labs.md).
