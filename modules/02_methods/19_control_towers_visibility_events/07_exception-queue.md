# Exception Management and Queue Design

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: 01 visibility layers, 03 event time, 04 ETA prediction, and 05 delay propagation  
> Version 2026.5

## 1. Why this topic exists

A control tower earns its place when it helps people make better decisions, not when it displays more events.
A carrier can send a late-arrival prediction, a warehouse can report a missed appointment, and a customer can call about the same shipment.
If each message becomes an independent urgent task, three people may contact the carrier while no one arranges a replacement delivery slot.
If no message has a named owner, everyone may assume that someone else is acting.

The queue is the operational bridge between detection and intervention.
It holds a bounded set of unresolved business situations, records who is accountable, exposes the next action and deadline, and prevents an issue from disappearing in a dashboard refresh.
It also gives a manager a way to compare incoming workload with the team's actual capacity.

Consider a supplier shipment forecast to arrive after a production changeover.
The delayed shipment matters only if the material is needed, no acceptable substitute is available, and action can still change the outcome.
The owner must determine whether to expedite, reschedule production, use alternate stock, or accept the delay.
A queue record should make that choice visible without treating the predicted arrival time as a customer commitment.

Queue design is a business-methodology problem as much as a software problem.
A score cannot grant an analyst authority to buy premium freight.
A workflow state cannot prove that a customer received a revised promise.
An escalation email cannot create carrier capacity.
The operating model must specify decision rights, coverage, service levels, and evidence for closure.

A poorly designed queue has costs in both directions.
Too many weak alerts consume attention and bury consequential cases.
Too few alerts leave important risks invisible until there is no practical remedy.
The team should measure both the work it performs and the outcomes it misses.
This lesson focuses on the work item, its ownership and lifecycle, prioritization, capacity, escalation, and measurement.
It does not present a universal algorithm for all networks.

## 2. Learning outcomes

After this lesson you can:

1. **Differentiate** an event, alert, exception, and action using a shipment scenario.
   Exercise 1 checks the distinctions.
2. **Design** a state machine with permitted transitions, ownership, and auditable closure.
   Exercise 2 tests the transition rules.
3. **Calculate** and challenge a risk-based queue order using explicit probabilities, units, and deadlines.
   Exercises 3 and 4 test the arithmetic and judgment.
4. **Assign** an accountable owner, backup, and decision authority across organizational boundaries.
   Exercise 5 tests the handoff.
5. **Estimate** workload and assess whether staffing is adequate for the arrival rate and service-time distribution.
   Exercises 6 and 10 test the capacity calculation and the allocation of scarce skills.
6. **Specify** escalation triggers, response clocks, and fallback paths that can be exercised on a shift.
   Exercise 7 tests the escalation design.
7. **Evaluate** queue performance using cohort-based metrics, missed outcomes, and data-quality controls.
   Exercise 8 tests the measurement plan.
8. **Critique** a closed-case claim when outcomes are censored, delayed, or affected by selection bias.
   Exercise 9 tests the evidence standard.

The outcomes are operational: a successful learner can explain the next decision and who can make it, not merely label a row high priority.

## 3. The core mental model

```text
Physical/business reality
       |
       v
Events and observations --> data quality checks --> detection rule or analyst report
                                                        |
                                                        v
                                      correlate to an existing case or create one
                                                        |
                                                        v
                         qualify --> assign owner --> prioritize against other work
                                      |                  |
                                      v                  v
                               authority check      deadline/capacity check
                                      |                  |
                                      +-------> decide and act
                                                   |
                                   +---------------+----------------+
                                   |                                |
                              await evidence                   escalate/transfer
                                   |                                |
                                   +--------------+-----------------+
                                                  |
                                       verify result or accept risk
                                                  |
                                               close
                                                  |
                                  outcome follow-up, measurement, learning
```

An event is evidence that something was observed or reported.
A detection rule proposes that the observation may require attention.
An exception is an actionable deviation or credible threat to a business objective.
A case groups the evidence and actions associated with that situation.
A queue is an ordered view of open cases, not a chronological copy of the event stream.

The distinction matters because more than one event can describe the same problem.
A late departure, a revised estimated time of arrival (ETA), and a missed appointment may all belong to one shipment-delay case.
Correlation must preserve the original evidence while avoiding duplicate work.
Conversely, two shipments to the same customer may need separate cases when their recovery options and owners differ.

The business decision determines the useful unit of work.
A production planner might manage a component shortage across several inbound orders as one case.
A transportation coordinator might manage each carrier booking separately under that parent case.
The case hierarchy should make dependencies visible without multiplying ownership ambiguously.

A priority is a scheduling recommendation, not a guarantee of outcome.
Expected loss is one useful input, but the next feasible intervention deadline, safety, customer commitments, uncertainty, and the cost and duration of action may change the order.
An owner can override the calculated order when the reason is recorded and reviewed.

Time has multiple meanings.
The event timestamp describes when a business event reportedly happened.
The received timestamp describes when the system learned of it.
The case-created timestamp begins some internal service clocks.
The intervention deadline is the last time a useful option can be exercised.
The customer commitment deadline is the promised delivery time, which may be later than the intervention deadline.
Confusing these clocks makes the team look fast while it acts too late.

The state machine is a contract between workers and managers.
It says what a state means, which transitions are allowed, what evidence is required, and which timers continue to run.
It should not pretend to model every email or phone call as a separate state.
Keep action history separate from the current state so that multiple actions can occur during one investigation.

Ownership is not the same as participation.
One person or staffed role is accountable for ensuring the next action occurs.
Other participants may provide evidence, approve expenditure, execute a shipment change, or receive updates.
A handoff is complete only after the receiving owner accepts it or a defined backup assumes responsibility.

Closure is an operational assertion, not proof that the underlying physical risk never existed.
A case may close as resolved, accepted, duplicate, invalid, or superseded.
The closure reason and evidence determine which metrics can legitimately count it as a successful intervention.
An outcome can be verified later, after the operational queue item has closed.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Event | Time-stamped observation or business fact about an object; one event record. | Treating a message as proof that the physical event occurred exactly as reported. |
| Event time | Reported time of the underlying occurrence; timestamp with zone. | Replacing it with system receipt time. |
| Processing time | Time the receiving application processes the record; timestamp with zone. | Assuming it is the operational occurrence time. |
| Alert | Notification generated by a rule, model, or person; one alert. | Calling every alert a confirmed exception. |
| Exception | Deviation or credible future threat requiring a decision against an objective; one business situation. | Defining importance solely by detector output. |
| Case | Managed record grouping an exception, its evidence, decisions, and history; one case identifier. | Creating a separate case for every update. |
| Correlation | Linking evidence or cases likely to concern the same situation; identifier and rule version. | Merging solely because two messages mention the same customer. |
| Deduplication | Suppressing or linking repeated representations of the same event or case; count and key. | Deleting original source evidence. |
| Parent case | Record for an aggregate impact such as a shortage across shipments; one identifier. | Assuming a parent automatically owns every child action. |
| Child case | Distinct actionable component linked to a parent; one identifier. | Treating linked work as independent total business loss. |
| Queue | Ordered view of eligible open work; cases and snapshot time. | Confusing the list with total downstream exposure. |
| Backlog | Eligible unresolved cases at a stated point in time; cases. | Comparing counts without checking scope and age. |
| Arrival rate | New eligible cases entering a queue per unit time; cases/hour. | Using raw event rate as case arrival rate. |
| Service time | Active staff time required for a case; minutes/case. | Equating active work with elapsed waiting time. |
| Lead time | Elapsed time from a specified start to a specified endpoint; hours or minutes. | Calling a lead time labor time. |
| Wait time | Elapsed time before a defined action starts; minutes. | Hiding an off-shift interval without stating the clock. |
| Acknowledgment | Recorded acceptance of responsibility for the next action. | Treating an unread notification as acknowledgment. |
| Owner | Person or staffed role accountable for moving the case forward. | Naming an unstaffed team mailbox as the sole owner. |
| Backup | Named alternate who takes responsibility under a stated trigger. | Listing a manager who has not accepted coverage. |
| Assignee | Person asked to perform a task, who may differ from the case owner. | Assuming task assignment transfers accountability. |
| Decision right | Authority to approve a particular action within stated constraints; for example USD per case. | Inferring approval authority from a queue priority. |
| Priority | Ordering or service class assigned to work under an explicit policy. | Treating a high score as evidence of an effective remedy. |
| Severity | Magnitude or type of potential consequence if a condition materializes; currency, units, or class. | Confusing severity with probability. |
| Probability | Estimated chance of a defined event in a specified horizon; fraction from 0 to 1. | Presenting an uncalibrated model score as a probability. |
| Exposure | Defined loss if a specified adverse outcome occurs; e.g., USD/case. | Adding revenue, margin, penalties, and disruption losses when they overlap. |
| Expected loss | Probability multiplied by the defined conditional loss; e.g., USD/case. | Claiming it is a certain or realized loss. |
| Avoidable loss | Expected loss reduction attributable to an available intervention; USD/case. | Equating total exposure with preventable loss. |
| Intervention deadline | Latest time at which a named action remains feasible; timestamp and time zone. | Using the delivery commitment as the deadline for all actions. |
| Commitment | An authorized promise to a customer or partner; quantity and date/time. | Calling an ETA forecast a promise. |
| Forecast | Estimated future outcome with an uncertainty basis and horizon. | Treating a forecast as an approved plan. |
| Plan | Chosen intended course of action, subject to execution and change. | Treating a proposed plan as completed work. |
| Target | Desired internal performance level, such as acknowledgment within 15 minutes. | Treating the target as a contractual service commitment. |
| Service level | Defined response or resolution objective, including clock and population. | Reporting an SLA percentage without its denominator. |
| Escalation | Transfer or addition of authority, attention, or support under a trigger. | Assuming every escalation transfers ownership. |
| Handoff | Explicit transfer of accountability with acceptance and history. | Declaring a case transferred at message send time. |
| Waiting | State in which a named external or internal dependency prevents the next action. | Letting waiting stop all clocks indefinitely. |
| Blocker | Specific missing input, approval, or external action preventing progress. | Recording only “pending” without an accountable party or due time. |
| Reopen | Return a previously closed case to active work for a linked unresolved issue. | Erasing the prior closure and history. |
| False positive | Alert judged not to represent its defined target condition. | Calling a prevented incident a false positive because harm did not occur. |
| False negative | Relevant target condition not detected under the stated detection process. | Assuming a closed-case sample reveals all misses. |
| Censoring | Outcome not yet observable by the measurement cutoff. | Treating immature cases as on time. |
| Aging | Elapsed time since a specified case milestone; minutes or hours. | Ranking all old cases ahead of urgent new cases without context. |
| Work in progress | Cases being actively handled within a defined queue; cases. | Counting waiting cases as active staff work without distinction. |
| Workload | Required active work over a period; staff-hours/period. | Using case count without service-time estimates. |
| Utilization | Busy staff capacity divided by available staff capacity; fraction. | Planning a variable queue to operate safely at 100%. |
| Audit trail | Append-only or otherwise controlled history of changes, actors, times, and evidence references. | Treating the latest editable status as a full history. |

## 5. The mechanics, step by step

### Step 1 — Define the decision and service boundary

**Input:** business objective, commitments, physical network, and available interventions.
**Output:** written exception type with a decision owner and eligible population.

Start with a decision such as “protect production from an inbound component shortage.”
State the affected object, the risk horizon, and the action that an owner can take.
A shipment event alone is not the decision boundary.
A shortage affecting a plant may depend on several orders and available inventory.

Write a concise entry rule.
For example: “Create a case when projected usable stock falls below required consumption before the next confirmed replenishment and an action is feasible.”
The rule must identify which forecast version, inventory snapshot, and replenishment assumption were used.

Specify exclusions and an alternate path.
For example, a safety-critical incident may bypass a commercial risk queue and go directly to the safety response process.
A data-quality defect with no reliable operational conclusion may enter a data investigation queue, not disappear.

Name what a case is not.
It is not an event log, a customer order, or a vehicle unless that object is also the chosen decision unit.
It should not require a human to act when no useful action or required notification exists.

### Step 2 — Establish case identity and correlation

**Input:** source event identifier, event type, affected object identifiers, timestamps, and existing open cases.
**Output:** a new case or a documented link to an existing case.

Create a stable case identifier independent of the source event identifier.
Store the event identifier, source, event time, receipt time, and original payload reference.
The event record and the case record have different lifecycles.

Define correlation keys by decision type.
A shipment delay might use shipment identifier, operational leg, and disruption episode.
A plant shortage might use site, material, and a bounded time window.
A common customer identifier alone is usually too broad.

Use a deterministic match where possible.
Where identity is uncertain, route a possible match for review rather than silently discarding either report.
A late update can change the apparent episode boundary; preserve both the original and revised linkage.

Separate event deduplication from case correlation.
Receiving the same message twice is not the same as receiving two different messages about one disruption.
Record both delivery attempts when operational auditability requires it, while avoiding duplicate active work.

When a parent case groups children, define which loss is counted at each level.
The parent can own the plant shortage and children can own bookings.
Do not add a parent loss estimate to the identical losses already included in its children.

### Step 3 — Qualify the exception

**Input:** correlated evidence, baseline plan or commitment, and detection rationale.
**Output:** confirmed actionable case, invalid alert, duplicate, or data-quality investigation.

Check whether the information is current enough for the decision.
A carrier ETA received an hour ago may be stale after a port release.
Record source reliability, observation time, and any unresolved contradiction.

Compare the predicted or observed outcome against the relevant baseline.
A forecast arrival after a warehouse appointment differs from a forecast arrival after the customer's committed delivery.
Both can matter, but they may require different owners and decisions.

Ask whether an action remains available.
If the booking cutoff has passed, expediting the original route may no longer be feasible.
A customer notification, revised production plan, or mitigation at destination may still be feasible.
Do not suppress a risk merely because one specific remedy is unavailable.

Separate “not enough evidence” from “no risk.”
A missing location update might reflect a sensor outage, a carrier integration failure, or a real delay.
Assign an information-gathering action and deadline when uncertainty itself affects the decision.

### Step 4 — Assign ownership and coverage

**Input:** case type, location, affected business process, shifts, and authority matrix.
**Output:** one accountable owner, an accepted backup, and any supporting assignees.

Route first by business decision, not just by the data source.
A carrier may provide the data while a plant planner owns the production decision.
A customer service representative may own the communication without having authority to change freight mode.

Assign an individual when the process requires a named person.
A staffed role is acceptable only when the roster identifies who currently fulfills it and who covers absence.
A shared inbox can deliver a notification but cannot by itself establish accountability.

Record the acknowledgment time and the deadline for acknowledgment.
If the owner does not accept in time, notify the backup and retain visible responsibility until acceptance.
If the backup also fails, escalate to a staffed duty manager or an equivalent operational role.

Define shift transfer explicitly.
Before the shift ends, the outgoing owner documents current state, evidence, next action, deadline, and unresolved approval.
The incoming owner accepts the handoff.
If acceptance does not occur, the duty manager owns finding a covered recipient.

### Step 5 — Compute the economic view

**Input:** adverse-event probability, defined conditional consequence, intervention options, and costs.
**Output:** comparable expected loss and, where possible, incremental expected benefit.

For a simplified case, expected loss without intervention is:

```text
expected loss = probability of adverse outcome × loss if adverse outcome occurs
```

This formula requires the probability and consequence to refer to the same outcome and time horizon.
If the adverse outcome is a production stop before Tuesday 18:00, do not combine it with a probability of late delivery by Wednesday noon.
Use one currency and a stated valuation basis.

For an action that changes the probability or severity, calculate:

```text
incremental expected benefit = expected loss without action
                             - expected loss with action
                             - incremental action cost
```

Include the action's remaining failure risk.
If a premium carrier has only an estimated chance of meeting the new slot, do not value the move as a guaranteed rescue.
Include only costs and benefits attributable to the choice, and avoid counting the same lost margin twice.

An estimate can be approximate and still help the discussion.
Label its assumptions and sensitivity.
If the probability is a model score without calibration, use an ordinal class or expert range rather than multiplying it as if it were a validated probability.

### Step 6 — Add time, feasibility, and protected classes

**Input:** economic view, current time, intervention deadlines, safety rules, and available capacity.
**Output:** an ordered queue with documented overrides and review times.

Expected loss alone does not encode the value of acting now.
A moderate-loss case with a cutoff in ten minutes may outrank a larger-loss case with a usable option tomorrow.
Compare the time until the next useful decision with estimated handling time and the queue ahead.

Identify hard constraints before numerical ranking.
A credible safety issue, a legal notification obligation, or an explicit contractual response deadline can require a separate response class.
The owner should verify whether the particular obligation applies; do not assume that every late shipment is legally reportable.

A practical policy can use bands rather than a single opaque score:

1. Route applicable safety and mandatory obligations to their dedicated process.
2. Identify cases whose next action deadline is inside the review window.
3. Within the same deadline band, order by estimated avoidable loss, adjusted for feasible actions and known dependencies.
4. Apply an aging review so low-ranked cases do not wait indefinitely.
5. Record any manual override, its rationale, and the next review time.

This is an example policy, not an optimal scheduling theorem.
A case requiring a five-minute phone call may be worth starting before a slightly higher-valued case requiring a two-hour investigation, but preemption and context-switching have costs.
The manager must choose a policy that matches the operating environment.

Do not manipulate expected loss simply to create urgency.
Preserve the economic estimate and record deadline, safety class, and override separately.
This allows later review of both the estimate and the priority decision.

### Step 7 — Manage the lifecycle

**Input:** qualified case, owner, evidence, and selected next action.
**Output:** auditable state changes with timers and valid closure.

Use the following minimal state vocabulary:

| State | Meaning | Required current information |
|---|---|---|
| New | Detected or reported, not yet qualified. | Source, creation time, and triage deadline. |
| Qualified | Decision-relevant risk confirmed; accountable owner not yet acknowledged. | Exception type, affected object, and proposed owner. |
| Active | Owner has accepted and is investigating or acting. | Owner, next action, and action deadline. |
| Waiting | Progress depends on a named response or event. | Blocker, responsible party, chase time, and next review. |
| Escalated | Additional authority or capacity requested; the case remains owned. | Trigger, recipient, requested decision, and response deadline. |
| Resolved pending verification | Chosen mitigation executed or risk accepted, but closure evidence is not yet complete. | Decision, execution evidence, and verification owner/time. |
| Closed | Closure reason recorded with required evidence. | Reason, closer, close time, and outcome follow-up where applicable. |

**Escalated** is represented here as a state for a small implementation.
An alternative is an escalation flag orthogonal to the base state, which supports simultaneous waiting and escalation.
Choose one representation consistently; do not hide an escalation just because an external dependency is also pending.
The transition rules below explicitly allow escalation from active or waiting and return to either.

| From | To | Guard and effect |
|---|---|---|
| New | Qualified | Evidence supports a decision-relevant exception; record type and proposed owner. |
| New | Closed | Invalid or duplicate with a reason, reviewer, and link to the surviving case if duplicate. |
| Qualified | Active | Authorized owner accepts; record acknowledgment and next action. |
| Qualified | Escalated | Assignment deadline missed; retain current routing responsibility and name recipient. |
| Active | Waiting | Named blocker, responsible party, chase time, and next review are present. |
| Active | Escalated | Authority, deadline, safety, or capacity trigger met; owner remains accountable. |
| Active | Resolved pending verification | Action completed or risk accepted by an authorized decision maker; evidence linked. |
| Waiting | Active | Dependency resolved or next action can proceed; record response. |
| Waiting | Escalated | Chase or intervention deadline threatens; document requested decision. |
| Escalated | Active | Owner or accepted new owner can proceed; log escalation response. |
| Escalated | Waiting | Named unresolved dependency remains and a chase time is set. |
| Escalated | Resolved pending verification | Authorized decision executed or accepted; preserve escalation record. |
| Resolved pending verification | Closed | Evidence meets the chosen closure reason's criteria. |
| Resolved pending verification | Active | Verification fails or new evidence requires action. |
| Closed | Active | New relevant evidence or failed outcome justifies reopening; retain prior closure. |

A qualified case that is canceled before acknowledgment may need a controlled closure path.
Require a triage supervisor to record invalidation, duplicate linkage, or supersession; do not silently delete it.
An implementation may add that transition if its role and audit controls are explicit.
The table above is the core path, not a claim that every operational variant has the same permissions.

A state change records actor, old state, new state, timestamp, reason, and supporting evidence reference.
Use authoritative server receipt time for the audit event and preserve reported business time separately.
Reject an invalid transition or request an authorized correction; do not overwrite history.

### Step 8 — Specify clocks and escalation

**Input:** case creation time, response objective, action deadline, coverage calendar, and authority matrix.
**Output:** clock definitions, trigger conditions, and named escalation recipients.

Define at least an acknowledgment clock and an intervention clock.
The acknowledgment clock measures whether the organization accepted responsibility.
The intervention clock measures time remaining to an action that could still affect the outcome.
Meeting the first does not imply meeting the second.

State whether each clock uses elapsed time or covered business time.
If a clock pauses outside staffed hours, disclose that choice and route an urgent case to an on-call process when the business risk continues overnight.
Safety and customer commitment clocks generally do not stop merely because an internal team is off shift.

Waiting for a carrier can pause active handling time, but should not automatically pause the intervention deadline.
Define a chase interval and an escalation deadline.
If the response is late, the case returns to an active decision: wait, use an alternative, or accept exposure with authority.

A good escalation request contains the decision needed, available options, deadline, estimated consequences, and the recommended choice.
It names the recipient and backup, and asks for acknowledgment.
Escalation is not simply sending an informational copy to a larger distribution list.

### Step 9 — Control work in progress and staffing

**Input:** case arrival history, active service-time distribution, staffing calendar, and backlog age.
**Output:** capacity plan, work-in-progress limits, and overload response.

Estimate required active staff-hours as cases per period multiplied by active hours per case.
Compare with effective capacity after breaks, meetings, training, and other assigned work.
Do not equate scheduled headcount with uninterrupted productive capacity.

Variability matters.
Arrivals cluster around vessel departures, weather disruptions, system outages, and shift handoffs.
Some cases need minutes while a small fraction need hours or repeated partner calls.
A team planned at 100% average utilization will generally have little buffer for that variability.

Segment by work type before interpreting averages.
A simple missing scan may not require the same expertise as a cross-border release problem.
A single average can conceal a shortage of the specific decision makers needed for high-impact work.

Limit the number of cases each worker actively handles if excessive switching slows completion.
A work-in-progress limit should not stop new critical cases from being surfaced.
It should trigger triage, reallocation, capacity escalation, or explicit deferral rather than hiding a case.

During overload, retain a protected path for safety and mandatory obligations.
Make deferral visible with owner, reason, next review, and residual exposure.
Managers should not mark cases closed merely to meet a queue-length target.

### Step 10 — Close and follow through

**Input:** executed action or authorized risk acceptance, evidence, and outcome follow-up plan.
**Output:** closed case with a reason and a reproducible history.

Use distinct closure categories.
“Resolved” means the required corrective action is verified against the agreed criterion.
“Risk accepted” means an authorized decision maker accepted the remaining exposure; it is not a prevented loss.
“Duplicate” requires a link to an active or historically relevant surviving case.
“Invalid” requires a reason such as erroneous identity or faulty source evidence.
“Superseded” requires a link to the replacement decision or case.

A booking confirmation is evidence of a booking, not proof of delivery.
Close an operational booking task when its defined criterion is met, but retain a separate outcome check for actual delivery if that is the business outcome being measured.
A queue can close earlier than the final outcome as long as the distinction is recorded.

Reopen when materially new evidence invalidates the closure or the expected outcome fails.
A reopened case keeps its identifier and prior history; a genuinely new episode may need a linked new case instead.
Define this distinction before reporting reopen rates.

### Step 11 — Measure and learn

**Input:** case history, detection population, commitments, outcome records, and staffing data.
**Output:** a small measurement set with denominators, cohort rules, and ownership.

Report queue health and business outcomes together.
A short acknowledgment time is useful only if cases receive effective decisions before options expire.
A low backlog can reflect better execution, lower arrivals, aggressive closure, or missed detection.

Define the reporting population before computing a metric.
For example, calculate acknowledgment compliance for cases created in the week whose acknowledgment deadline has passed by the extraction time.
Exclude cases with immature deadlines from that denominator and report their count separately.

Track arrivals, closures, net backlog change, backlog age distribution, acknowledgment compliance, deadline breaches, active handling minutes, and escalation response.
Segment by case type, site, shift, customer group where authorized, and source quality.
Avoid publishing case-level partner data to audiences without a need to know.

Track outcome measures such as verified on-time delivery, production interruption hours, and incremental recovery cost.
The outcome metric needs a defined cohort and sufficient maturity.
An intervention success estimate needs a defensible comparison or explicit counterfactual assumptions; it is not simply the sum of gross estimated exposures for cases acted on.

Review missed incidents outside the queue.
Compare independent downstream service failures, customer claims, and production interruptions with detected cases after matching on identifiers and time.
Investigate misses and linkage gaps before calling the detection process complete.

### Step 12 — Operate an overload protocol

**Input:** open cases, deadlines, available skills, expected arrivals, and escalation coverage.
**Output:** a time-bounded allocation of staff and a visible list of deferred work.

An overload protocol starts before every analyst is fully occupied.
Set a trigger using a measurable condition, such as cases approaching their intervention cutoff faster than qualified staff can process them.
Backlog count alone is not enough: ten cases awaiting tomorrow's information and ten cases expiring in the next hour imply different responses.
Compare required active minutes within each deadline window with the staff minutes available in that same window.
Deduct already committed work and coverage obligations before treating minutes as available.

At the start of a surge, the duty manager first confirms the scope of the disruption.
An integration outage that produces repeated missing-scan alerts may require one incident owner and many linked affected shipments.
A severe weather disruption may instead produce genuinely separate decisions for several destinations.
Do not merge merely to make the queue appear manageable.
Do not split merely to distribute a single loss across more owners.

The duty manager then assigns available capacity to protected obligations and imminent cutoffs.
For each remaining case, record the next useful action, estimated staff time, decision authority, and latest execution time.
If two cases need the same approver, count that approver as a constrained resource rather than assuming all staff minutes are interchangeable.
If the time estimate is uncertain, reserve a buffer or choose a conservative estimate and document the basis.
An average handling time is not a promise that the next case will finish in that time.

For cases that cannot all be completed, compare feasible alternatives rather than silently letting the clock expire.
The alternatives can include requesting surge coverage, reducing nonessential investigation, using a preauthorized contingency, or accepting a defined residual exposure.
The duty manager must check that abbreviated handling does not bypass mandatory controls or quality checks.
If an action is deferred, record who made the decision, what may be lost, when to review it, and who must be notified.
A deferred case remains visible until resolved, accepted by an authorized person, or legitimately superseded.

The trigger and response should be specific enough to exercise.
For example, a team could initiate a 15-minute capacity review when the sum of estimated work due in the next hour exceeds 80% of the effective minutes available from staff with the required skills.
The 80% figure is an illustrative management trigger, not a universal staffing standard.
It is a leading warning, not proof that a deadline will be missed.
The appropriate threshold depends on variability, potential harm, and the speed of obtaining additional capacity.

During the review, check both the count of cases and the amount of action-specific work.
One case may require an initial call now and a second approval later.
Do not assign both activities to the first deadline if their actual constraints differ.
Conversely, do not omit the second activity simply because the first call can be finished on time.
Where several cases share an outcome, identify the dependency before adding their claimed benefits.

Use a short decision log with the following fields:

| Field | Question answered |
|---|---|
| Snapshot time and zone | What was known when capacity was assigned? |
| Coverage and skills | Who could actually perform each action before its cutoff? |
| Case and next action | What specific work was competing for attention? |
| Required minutes and deadline | How much labor was estimated, and when did the option expire? |
| Protected obligation | Did a separate safety, legal, or contractual process constrain the order? |
| Decision and authority | Who approved the allocation, deferral, or acceptance? |
| Next review | When must the decision be revisited or escalated? |
| Subsequent evidence | What happened after the decision, including changed estimates? |

When the immediate pressure subsides, reconcile the decision log against actual case transitions.
Look for expired options, unidentified dependencies, and cases that were not surfaced during the surge.
Compare estimated active minutes with recorded active minutes by case type and role.
Review whether staff invoked the protocol early enough for the requested help to arrive.
Update thresholds only after considering changes in case mix, staffing, and data quality.

### Overload example — two scarce resources

At 13:00 local time, one coordinator and one supervisor are available for the next 30 minutes.
The coordinator has 30 effective minutes for queue work.
The supervisor has 10 effective minutes after a previously scheduled customer call.
Both staff members can work concurrently, but the supervisor's approvals are sequential.
Assume each task can start only after the prerequisites in the table are complete.

| Case | Coordinator preparation | Supervisor approval | Latest completion | Estimated net benefit |
|---|---:|---:|---|---:|
| D | 8 minutes | 4 minutes | 13:15 | USD 3,100 |
| E | 12 minutes | 6 minutes | 13:30 | USD 5,400 |
| F | 10 minutes | None | 13:30 | USD 1,200 |

The estimates refer to different, non-overlapping outcomes and already deduct incremental intervention cost.
No safety or mandatory-notification obligation applies in this illustration.
Completion means the preparation and any approval are finished; execution is assumed to occur instantly on approval for this scheduling exercise.
In a real operation, booking or allocation execution time must also be included before the cutoff.

One feasible plan prepares D from 13:00 to 13:08.
The supervisor approves D from 13:08 to 13:12.
The coordinator prepares E from 13:08 to 13:20.
The supervisor approves E from 13:20 to 13:26.
The coordinator prepares F from 13:20 to 13:30.
All three modeled tasks meet their cutoffs, using all 30 coordinator minutes and all 10 supervisor minutes.
The total modeled net benefit is USD 9,700 if all three actions execute as assumed.
There is no capacity buffer for an interruption, a longer approval, or execution work.

Now suppose D's preparation actually takes 13 minutes rather than 8 minutes.
Its approval cannot begin before 13:13 and finishes at 13:17, after its 13:15 cutoff.
The supervisor has not failed to respond slowly: the prerequisite was late.
The team should escalate as soon as the revised estimate makes the deadline infeasible and look for a different option.
For example, an authorized backup may be able to start some preparation in parallel if the work can genuinely be divided.
The original USD 3,100 benefit must not be counted if the underlying option is no longer available.

This example distinguishes resource utilization from service reliability.
The initial schedule appears feasible in deterministic arithmetic but is brittle.
The coordinator should not simply book every minute and report the plan as a service guarantee.
The manager must decide whether surge cover, earlier preparation, or a deliberate deferral is justified by the risk and the operating constraints.

## 6. Worked example

### Business context and assumptions

At 09:00 local time on a Tuesday, a plant control tower has one trained coordinator available for the next hour.
A supervisor can authorize premium freight and a planner can change the production sequence.
The coordinator is the accountable queue owner for all three cases until an accepted handoff.
The values and probabilities below are illustrative, not claims about a real company.

The production plan, customer commitment, and arrival forecasts are different objects:

- The plan schedules the plant to use material A at 15:30 Tuesday.
- A customer commitment calls for finished goods by 12:00 Wednesday.
- A carrier forecasts material A arrival at 16:10 Tuesday, with uncertainty.
- An internal target asks for an acknowledgment within 15 minutes of case creation.

A forecast is not an approved change to the customer commitment.
The coordinator may not independently authorize the expedited freight or alter the customer promise.

Three cases are eligible at 09:00:

| Case | Situation | Probability of adverse outcome | Loss if outcome occurs | Next action deadline | Active coordinator time |
|---|---|---:|---:|---|---:|
| A | Component shortage before planned production start. | 0.65 | USD 18,400 | 09:20 Tuesday, premium-truck booking | 12 minutes |
| B | Customer shipment may miss a Wednesday appointment. | 0.40 | USD 28,000 | 14:00 Tuesday, appointment change | 18 minutes |
| C | Store replenishment may arrive after promotion starts. | 0.30 | USD 9,000 | 10:00 Tuesday, alternate allocation | 10 minutes |

Each probability refers to the defined outcome in the stated horizon and is assumed calibrated for this exercise.
Each loss is the incremental business cost if that outcome occurs, excluding double-counted revenue and margin.
The model does not claim that these estimates will hold in another network.

The no-action expected losses are:

```text
A = 0.65 × USD 18,400 = USD 11,960
B = 0.40 × USD 28,000 = USD 11,200
C = 0.30 × USD  9,000 = USD  2,700
```

Raw expected loss would order A, B, C.
The coordinator also considers feasible intervention deadlines and the expected value of each available action.
Assume there are no safety or mandatory-notification overrides in these illustrative cases.

### Available interventions

For A, a premium truck costs USD 2,600 incrementally.
If booked by 09:20, it reduces the probability of the defined production loss from 0.65 to 0.15.
The estimated incremental expected benefit is:

```text
A: (0.65 - 0.15) × USD 18,400 - USD 2,600
 = USD 9,200 - USD 2,600
 = USD 6,600
```

For B, a warehouse appointment change costs USD 500 incrementally.
If agreed by 14:00, it reduces the probability of the defined customer loss from 0.40 to 0.10.

```text
B: (0.40 - 0.10) × USD 28,000 - USD 500
 = USD 8,400 - USD 500
 = USD 7,900
```

For C, reallocating local stock costs USD 350 incrementally and reduces the adverse-outcome probability from 0.30 to 0.10 if approved by 10:00.

```text
C: (0.30 - 0.10) × USD 9,000 - USD 350
 = USD 1,800 - USD 350
 = USD 1,450
```

The highest estimated intervention benefit belongs to B, but its decision deadline is much later.
A must be initiated first to preserve the 09:20 booking option.
C's allocation deadline follows at 10:00.
The suggested order is A, C, B, subject to new evidence and the assumption that the coordinator can execute the next steps sequentially.

### Timeline and ownership

At 09:00 the coordinator acknowledges A, C, and B in the system.
The acknowledgment does not imply that their interventions have been completed.
It records acceptance of the next action and stops only the acknowledgment clock.

From 09:00 to 09:12 the coordinator checks A's inventory, obtains the latest carrier information, and requests the USD 2,600 freight approval.
At 09:12 the supervisor approves, and the coordinator submits the booking.
The case enters waiting with carrier confirmation due at 09:17 and a chase at 09:16.
The 09:20 booking deadline continues to run.

From 09:12 to 09:22 the coordinator works C.
The stock owner approves the allocation at 09:22, leaving time before 10:00 for execution.
A's carrier confirms at 09:15; an automatic notification reaches the coordinator immediately, so the 09:16 chase is unnecessary.
For this example assume that the 09:15 booking confirmation satisfies the 09:20 cutoff and that the notification takes negligible handling time.
The coordinator records the confirmation at 09:22, retaining both timestamps.
If the notification had not arrived by 09:16, the procedure would require an immediate chase or reassignment rather than uninterrupted work on C.

At 09:22 the coordinator starts B and finishes the appointment request at 09:40.
The warehouse confirms the change at 10:15.
B can move to resolved pending verification when the authorized appointment change is documented, then close after its stated operational evidence criterion is met.
The actual customer delivery remains a later outcome check.

The active coordinator time is 12 + 10 + 18 = 40 minutes, excluding interruptions and later follow-up.
The elapsed time from 09:00 to B's 10:15 confirmation is 75 minutes.
Those are different measures; waiting for a warehouse response is not necessarily 75 minutes of staff labor.

### Interpreting the result

The sum of estimated incremental benefits, assuming the three losses and actions are independent and non-overlapping, is:

```text
USD 6,600 + USD 7,900 + USD 1,450 = USD 15,950
```

This is a prospective expected benefit under the given assumptions, not observed savings.
It must not be booked as realized value without a measurement design and an appropriate counterfactual.
If A and B are different manifestations of the same lost customer order, their impacts may overlap and the sum overstates benefit.

Suppose A's true no-action adverse probability is only 0.25 rather than 0.65, while the intervention probability remains 0.15.
The estimated net benefit becomes:

```text
(0.25 - 0.15) × USD 18,400 - USD 2,600 = -USD 760
```

That sensitivity changes the freight recommendation.
The owner should obtain better evidence when it can arrive before the booking cutoff, or present the uncertainty to the supervisor.
A negative expected benefit is not an automatic rejection if a separate protected obligation applies; no such obligation is assumed here.

### Second business case: missed coverage

A temperature-controlled shipment reports a sensor gap at 18:40, after the day team signs off.
The shipment contains perishable product, but a missing reading alone does not establish temperature excursion.
The on-call quality role accepts the case, requests the logger history, and holds the release decision until the quality procedure permits it.
The transport coordinator remains responsible for carrier contact; the quality role owns product disposition.
If no logger response arrives by 19:00, the quality lead is escalated with the remaining hold time and alternate evidence options.
The case closes only with a documented disposition and supporting evidence, not merely because a tracking feed resumes.
This is an illustrative operating design; actual product, customer, and regulatory requirements must be checked for the relevant jurisdiction.

## 7. Code, query, or calculation

The following standard-library Python example reproduces the economic calculations and a transparent deadline-band queue order.
Save it as `queue_example.py` and run `python queue_example.py` with Python 3.9 or later.
It uses integer minutes relative to 09:00 and decimal currency arithmetic.
There is no external data, network access, or package dependency.

The ranking policy is intentionally simple: deadlines within 30 minutes precede those within 60 minutes, then later deadlines; within a band, greater estimated incremental benefit comes first.
This is not a complete scheduling optimizer and does not model staff interruption or protected classes.
A real deployment would also check eligibility, authority, dependencies, calibration, and capacity.

```python
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Case:
    name: str
    deadline_minutes: int
    probability_without: Decimal
    probability_with: Decimal
    conditional_loss_usd: Decimal
    action_cost_usd: Decimal
    handling_minutes: int

    def expected_loss(self) -> Decimal:
        return self.probability_without * self.conditional_loss_usd

    def incremental_benefit(self) -> Decimal:
        reduction = self.probability_without - self.probability_with
        return reduction * self.conditional_loss_usd - self.action_cost_usd

    def deadline_band(self) -> int:
        if self.deadline_minutes <= 30:
            return 0
        if self.deadline_minutes <= 60:
            return 1
        return 2


cases = [
    Case("A", 20, Decimal("0.65"), Decimal("0.15"),
         Decimal("18400"), Decimal("2600"), 12),
    Case("B", 300, Decimal("0.40"), Decimal("0.10"),
         Decimal("28000"), Decimal("500"), 18),
    Case("C", 60, Decimal("0.30"), Decimal("0.10"),
         Decimal("9000"), Decimal("350"), 10),
]

ranked = sorted(
    cases,
    key=lambda case: (
        case.deadline_band(),
        -case.incremental_benefit(),
        case.deadline_minutes,
        case.name,
    ),
)

for case in ranked:
    print(
        f"{case.name}: expected loss USD {case.expected_loss():.2f}; "
        f"incremental benefit USD {case.incremental_benefit():.2f}; "
        f"deadline in {case.deadline_minutes} min"
    )

print(f"Active handling: {sum(c.handling_minutes for c in ranked)} min")
print(
    "Total estimated incremental benefit: USD "
    f"{sum((c.incremental_benefit() for c in ranked), Decimal('0')):.2f}"
)
```

Expected output:

```text
A: expected loss USD 11960.00; incremental benefit USD 6600.00; deadline in 20 min
C: expected loss USD 2700.00; incremental benefit USD 1450.00; deadline in 60 min
B: expected loss USD 11200.00; incremental benefit USD 7900.00; deadline in 300 min
Active handling: 40 min
Total estimated incremental benefit: USD 15950.00
```

The tie-breakers are explicit: within the same deadline band, higher benefit is listed first, then earlier deadline, then case name.
A negative-benefit case remains visible in this demonstration; a real policy must decide whether to defer, accept the risk, or use another route.
The sum assumes no overlap among the three case benefits, as stated in the worked example.
The code does not assert that an intervention was authorized, executed, or successful.
It is a decision-support calculation, not a control that makes a carrier booking.

### Capacity calculation

Suppose a different team receives 36 qualified cases per eight-hour shift.
Average active handling is 20 minutes per case, including documented follow-up.
Required work is:

```text
36 cases/shift × 20 minutes/case = 720 minutes/shift
720 minutes/shift ÷ 60 minutes/hour = 12 staff-hours/shift
```

Two people each provide six effective handling hours after breaks, meetings, and other duties.
Their combined effective capacity is 12 staff-hours per shift.
Average utilization is therefore 12 / 12 = 1.00, or 100%.
This has no average capacity buffer for arrival bursts, long cases, absence, or backlog recovery.
A third person providing six effective hours would give 18 staff-hours, making the arithmetic average utilization 12 / 18 = 0.667, or about 66.7%.
This calculation alone does not prove a particular service level; arrival variability, skill constraints, and shift coverage still matter.

## 8. Data, standards, and regulatory context

### Minimum data contract

A useful case record has stable identity, business context, and enough operational history to support the next decision.
The following fields are illustrative; the owner should define permitted values and retention for the specific process.

| Field | Purpose | Quality check |
|---|---|---|
| Case identifier | Stable reference across changes. | Unique and immutable. |
| Case type and version | Identifies entry and handling rule. | Controlled vocabulary; historical version retained. |
| Shipment, order, site, or material identifier | Links the case to affected business objects. | Resolve against the relevant master data. |
| Source and source event identifier | Supports traceability and deduplication. | Source-scoped uniqueness and provenance. |
| Event time and time zone | Records reported occurrence. | Plausible range; preserve original time and offset. |
| Receipt and case creation times | Supports latency and response measurement. | System-generated timestamps with known clock basis. |
| Baseline commitment and forecast | Explains the gap and impact. | Separate fields with version and authority. |
| Probability, consequence, and horizon | Makes economic prioritization inspectable. | Units, calibration basis, and valid range. |
| Intervention and deadline | Preserves the practical option. | Deadline has zone, source, and update history. |
| Owner and backup | Defines accountability. | Map roles to staffed roster and acceptance. |
| Next action and review time | Prevents indefinite waiting. | Required for active, waiting, and escalated work. |
| State and transition history | Provides process control. | Enforce allowed transitions and record actor. |
| Closure reason and evidence | Makes closure meaningful. | Validate reason-specific evidence and links. |
| Outcome and observation cutoff | Supports later measurement. | Distinguish missing, immature, and observed outcomes. |

Identifiers can be reused or corrected by source systems.
A case should retain its original identifier and link corrected identifiers rather than rewriting the history without explanation.
When a partner transmits a local time without a time-zone offset, mark the uncertainty and obtain the operational time basis before comparing it with an intervention cutoff.

### Event standards and interfaces

GS1 Electronic Product Code Information Services (EPCIS) 2.0 defines a standard for sharing visibility event data, including what, when, where, why, and how dimensions.
It can supply interoperable event evidence to an exception workflow.
It does not define this lesson's queue score, staffing plan, ownership matrix, or commercial approval limits.
See the GS1 standard in section 16 for the authoritative specification.

Business partners may exchange shipment and order information through electronic data interchange (EDI), application programming interfaces (APIs), files, or portals.
A standard message does not guarantee timely, complete, or accurate information.
Define partner-specific service expectations, identifier mapping, correction procedures, and acknowledgment behavior in the data contract.
A missing carrier event should not be automatically interpreted as physical nonmovement.

Use an idempotency key appropriate to each interface to avoid creating repeated work from retries.
A correction or cancellation is not necessarily a duplicate; preserve the relationship to the prior event.
Out-of-order arrival must not overwrite newer business evidence merely because it was processed later.
A case can be reprioritized when late evidence arrives, with the reason and prior decision retained.

### Privacy, security, and jurisdiction

Case records may contain names, phone numbers, location traces, customer details, or commercially sensitive rates.
Apply role-based access, minimum necessary sharing, defined retention, and secure access to evidence references.
A partner may need the shipment status without needing the buyer's full customer list or the internal loss estimate.

The European Union General Data Protection Regulation (GDPR) applies only where its territorial and material scope conditions are met.
It contains principles including data minimization, storage limitation, and integrity and confidentiality in Article 5.
A queue design does not by itself establish a lawful basis, retention period, or compliance.
Obtain qualified privacy advice for the relevant personal data, roles, and jurisdiction.

Specific goods, modes, contracts, and jurisdictions may impose additional safety, traceability, or reporting requirements.
A queue escalation target is an internal operating rule unless a contract or applicable law makes it an obligation.
Do not infer a mandatory notification deadline from an internal service-level target.
The case owner needs an accessible escalation path to the compliance or quality function when applicability is uncertain.

### Retention and correction

Distinguish the mutable current view from the history of what was known at the time.
Historical analysis requires the prediction version, priority, owner, and decision available at each decision point.
Otherwise a later corrected ETA can make an earlier reasonable action look irrational or vice versa.

Retention must balance audit needs, contractual duties, and privacy obligations.
Restrict access to sensitive fields and document deletion or de-identification policies.
Keep a reference to evidence where permitted, but do not assume every original payload can be retained forever.

## 9. Industry practice and case evidence

### Public evidence and what it supports

Control towers are used to combine operational information and coordinate responses across organizational boundaries.
Public descriptions often emphasize visibility, but a visibility platform does not by itself demonstrate the quality of ownership, prioritization, or intervention.
When reading a case study, ask which decisions changed, who had authority, and how outcomes were measured.

**IBM and Lenovo.** IBM's published Lenovo supply-chain transformation case describes the use of AI and a control-tower approach in Lenovo's operations.
The source is a company case study and should be read as a company/vendor account rather than independent causal evidence.
It supports the observation that a named company publicly describes control-tower use; it does not validate the hypothetical scoring policy or the numerical outcomes in this lesson.
**Inference:** a distributed manufacturer using such visibility still needs named owners and approval rules to turn information into action.
The inference is an operating implication, not a reported Lenovo internal procedure.

**Maersk and TradeLens.** Maersk announced in November 2022 that the TradeLens platform would be discontinued, citing a lack of the commercial viability needed for a fully independent platform.
This is evidence about a named multi-party platform's adoption and viability, not proof that all supply-chain information sharing fails.
**Inference:** a shared exception queue needs aligned incentives, data rights, partner participation, and a sustainable operating model in addition to technical connectivity.
The lesson's case state machine is not claimed to have been used by TradeLens.

### Practices worth examining in your own operation

A shipper may group updates by shipment and leg, while the customer service team groups impact by customer order.
Neither view is universally correct.
A shared case can link both views, but the organizations must agree which team owns each decision and which information can be shared.

A plant can maintain a parent shortage case with several inbound child cases.
The parent owner selects the production response; transport owners handle carrier options.
A parent should not be closed simply because one carrier confirms a booking if another component remains critical.

A retailer can route store-level exceptions to regional teams while protecting centrally managed allocation decisions.
An unresolved promotion shortage may require both allocation and customer communication.
The queue should make their dependencies visible without giving two teams contradictory ownership of the same decision.

A logistics provider may offer a customer a portal with case status.
The customer still needs to know whether “resolved” means carrier contacted, booking confirmed, or delivery completed.
A shared status dictionary and evidence rules reduce disagreement about service performance.

### Evidence discipline

A before-and-after reduction in backlog is not automatically caused by a new queue.
Arrival volume, seasonal demand, business mix, staffing, and case-entry thresholds may also have changed.
A comparison should document these changes and, where feasible, compare similar cases or sites over a common period.

An apparent reduction in late delivery may result from longer customer promises rather than improved transport execution.
Report commitment changes separately from physical transit performance.
Likewise, a lower reported financial exposure may arise from revised loss assumptions, not better recovery.

Use evidence grades appropriate to source type.
An official standard supports claims about the standard's content.
A company case supports the fact that the company made a particular claim, not necessarily an independently verified causal effect.
An illustrative classroom calculation supports only its stated assumptions.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Alert flood | Alerts per qualified case rise sharply. | A broad rule creates work faster than it can be triaged. | Critical cases wait behind noise. | Tune entry rules using labeled outcomes and protected controls. |
| Missed detection | Customer complaints lack linked cases. | Monitoring covers only a subset of lanes or sources. | Risk appears after the remedy expires. | Reconcile independent outcome and claim feeds. |
| Duplicate cases | Several owners contact one carrier about one episode. | Retry messages or correlated alerts create new records. | Wasted capacity and conflicting instructions. | Stable source keys and reviewable case correlation. |
| Over-merged cases | Unrelated shipments share one case and owner. | Broad customer-only matching. | One resolution hides unresolved child exposure. | Use decision-level keys and allow controlled splits. |
| Unowned case | A team mailbox is the only assignee. | No individual or staffed role accepts the work. | No one executes the next action. | Named roster, acknowledgment deadline, and backup. |
| Broken handoff | Outgoing owner marks transfer before acceptance. | Responsibility falls between shifts or teams. | Action deadline is missed. | Require recipient acceptance and duty-manager fallback. |
| Score used as authority | Analyst books freight above their limit. | Priority is mistaken for approval. | Unauthorized expense or contractual breach. | Separate ranking from decision-right checks. |
| Stale evidence | Case shows an old ETA after a newer arrival event. | Receipt order overwrites event semantics. | Unnecessary expedite or missed intervention. | Preserve timestamps, versions, and correction logic. |
| Deadline hidden | Case is low priority until the delivery commitment approaches. | Booking cutoff is not recorded. | Useful recovery option expires. | Store action-specific deadlines and monitor slack. |
| Waiting forever | Cases remain pending without a chase date. | External dependency pauses all attention. | Backlog ages and outcomes deteriorate. | Named blocker, chase, review, and escalation. |
| Escalation theater | Messages sent but no recipient acknowledgment. | Notification is confused with decision transfer. | No timely approval. | Require explicit request, recipient, response deadline, and fallback. |
| Invalid closure | Many cases close immediately before weekly reporting. | Team is rewarded for low backlog without evidence. | Work disappears while risk persists. | Reason-specific evidence, audit sampling, and outcome follow-up. |
| Double-counted value | Parent and child cases both claim the full avoided loss. | Hierarchical exposure is summed without allocation. | Benefits are overstated. | Define economic ownership and non-overlap rules. |
| Probability misuse | Model scores are multiplied directly by dollar losses. | Uncalibrated rankings are treated as probabilities. | False precision and wrong priority. | Calibrate by horizon or use ordinal bands. |
| Capacity overload | Oldest case age rises despite fast acknowledgments. | Arrivals exceed effective service capacity. | Deadline breaches and staff burnout. | Measure active work, add coverage, and make deferrals explicit. |
| Censored outcomes | Recent closed cases are reported as successful before delivery. | Missing outcomes are counted as success. | Inflated intervention effectiveness. | Mature cohorts and report unresolved outcomes separately. |
| Inadequate access control | Partner sees unrelated customer rates or personal data. | One shared view exposes all case fields. | Confidentiality and privacy harm. | Role-based views, minimization, and access review. |
| Unstable baseline | Priority changes after a promise is edited without history. | Forecast, plan, and commitment are overwritten. | Audit and performance analysis become unreliable. | Version baselines and record authorized changes. |

The warning signs are diagnostic prompts, not automatic proof of the listed mechanism.
Investigate a sample of cases and the underlying data before changing a policy.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test it |
|---|---|---|---|
| Probability is calibrated for the defined horizon. | Expected-loss arithmetic uses a probability, not merely an ordinal score. | A new lane or changed carrier performance shifts outcomes. | Compare predicted bands with mature observed outcomes by cohort. |
| Conditional loss is incremental and in one currency. | Economic comparisons need a common basis. | Revenue and margin are counted together or currency conversions differ. | Reconcile loss components with finance and document rates. |
| An intervention changes the stated probability. | Benefit requires a credible causal effect. | The premium route has the same constraint as the original route. | Compare similar eligible cases and review operational feasibility. |
| Cases have a workable decision boundary. | Owners can act without conflicting instructions. | A shortage spans several independent plants and approvals. | Trace cases through real decisions and split or link as needed. |
| Data are sufficiently timely. | A priority based on an old event can be wrong. | Partner feed outages or long batch intervals. | Measure event-to-receipt and receipt-to-case latency. |
| The owner or backup is staffed. | A named role must be able to act. | Overnight, leave, holidays, or network failure. | Exercise the roster and escalation path across shifts. |
| Intervention deadline is correct. | The queue must preserve feasible options. | Carrier cutoff changes or local time zone is wrong. | Confirm with the decision maker and track revisions. |
| Service time estimates represent actual work. | Capacity planning depends on labor required. | Rare complex cases and follow-up are excluded. | Sample active handling by case type and percentile. |
| Independent cases have non-overlapping benefits. | Summing expected benefit otherwise double-counts. | Parent and child cases reflect one customer loss. | Map outcomes and allocate loss to a single economic unit. |
| Closure evidence matches the stated outcome. | Closure counts need comparable meaning. | A booking confirmation is treated as delivery. | Audit a sample from closure to physical outcome. |
| An operational record can be linked to outcomes. | Misses and effectiveness need a denominator and follow-up. | Missing identifiers or partner access restrictions. | Measure linkage completeness and investigate unmatched records. |
| Constraints and obligations are known. | A financial score cannot capture every mandatory action. | Product-specific or jurisdiction-specific rules change. | Maintain a reviewed obligation register and escalation contact. |

No single queue order is optimal under every combination of deadlines, uncertain service times, shared resources, and action dependencies.
The example uses a transparent heuristic because it can be explained and challenged.
When a network needs formal scheduling or stochastic optimization, validate the model against actual operational constraints and keep an override path.

## 12. Exercises

### Exercise 1 — Classify the information

A carrier sends the same departure message twice.
A later ETA forecast suggests that the shipment will miss a booked warehouse slot.
An analyst contacts the warehouse to request a new slot.

**Deliverable:** Label each item as event, alert or forecast, exception/case, or action, and say how many active cases you would initially create.
**Acceptance criteria:** Distinguish duplicate message delivery from a second operational event; explain why the forecast is not a commitment; state a reasonable case boundary.
**Time:** 10 minutes.

### Exercise 2 — Test the state machine

A qualified case is assigned to an owner who has not acknowledged it.
A system integration tries to move it directly to closed as resolved because a booking message appeared.
The message has not been verified and no authorized person accepted the case.

**Deliverable:** Write the valid next steps and identify the rejected transition.
**Acceptance criteria:** Preserve accountability, require evidence and authorization, and distinguish a booking from the business outcome.
**Time:** 15 minutes.

### Exercise 3 — Compute benefit

A missed customer appointment has probability 0.50 without action and 0.20 with action.
The defined conditional loss is USD 12,000 and the action costs USD 1,100.
The action remains feasible until 11:00 local time.

**Deliverable:** Calculate expected loss without action and incremental expected benefit.
**Acceptance criteria:** Show arithmetic, units, and at least one assumption about the loss or intervention.
**Time:** 10 minutes.

### Exercise 4 — Decide the order

At 10:00, case X has an estimated benefit of USD 4,000, a 10:15 action cutoff, and needs 10 minutes of staff work.
Case Y has an estimated benefit of USD 7,000, an 11:30 cutoff, and needs 20 minutes.
One worker is free and there are no protected obligations or dependencies.

**Deliverable:** Propose an order and the finish time of each action.
**Acceptance criteria:** Preserve both options under the stated assumptions and explain why benefit alone does not determine the first action.
**Time:** 10 minutes.

### Exercise 5 — Allocate ownership

A component shipment may miss a plant production start.
A carrier can confirm an alternative truck, a buyer owns the supplier relationship, a plant planner owns the sequence, and a supervisor can approve premium freight.

**Deliverable:** Create a compact responsibility table naming the case owner, supporting roles, approver, backup, and handoff condition.
**Acceptance criteria:** Exactly one accountable case owner at a time; separate action execution from spending approval; specify acceptance and escalation.
**Time:** 20 minutes.

### Exercise 6 — Estimate workload

A queue receives 45 qualified cases per day.
Average active handling is 24 minutes per case.
Each analyst has six effective handling hours per day.
There are four analysts with the necessary skills.

**Deliverable:** Calculate daily work, available capacity, and average utilization; recommend a management response.
**Acceptance criteria:** Show staff-hours and percentage; note at least two limitations of the average; avoid treating the calculation as an SLA guarantee.
**Time:** 15 minutes.

### Exercise 7 — Design escalation

A waiting case has a carrier response due at 16:10 and an alternative booking cutoff at 16:30.
The first owner ends the shift at 16:00.

**Deliverable:** Specify a chase time, a transfer mechanism, an escalation recipient, an acknowledgment deadline, and a fallback.
**Acceptance criteria:** Preserve ownership during handoff; keep the booking clock running; include enough time to make and execute a decision.
**Time:** 20 minutes.

### Exercise 8 — Define measurement

The manager wants a weekly dashboard reporting queue responsiveness and business impact.
Some cases are still open and some shipments have not yet reached their committed delivery time.

**Deliverable:** Define four metrics with numerator, denominator, cohort, and cutoff; include an independent missed-case check.
**Acceptance criteria:** Separate process from outcome; handle immature outcomes; avoid claiming gross risk as realized savings; identify a segmentation dimension.
**Time:** 25 minutes.

### Exercise 9 — Critique a success claim

A vendor says 90% of closed cases were successful because 90% did not have a recorded service failure at the reporting cutoff.
The underlying extract includes cases closed yesterday and only queue-linked shipment outcomes.

**Deliverable:** Write a short critique and propose a stronger evaluation.
**Acceptance criteria:** Identify censoring, missing linkage, selection or detection bias, and counterfactual uncertainty; propose measurable remedies.
**Time:** 20 minutes.

### Exercise 10 — Allocate surge capacity

At 14:00, one coordinator has 30 effective minutes and one supervisor has 8 effective minutes available before 14:30.
Case G needs 10 coordinator minutes followed by 3 supervisor minutes, with both steps completed by 14:15.
Case H needs 12 coordinator minutes followed by 5 supervisor minutes, with both steps completed by 14:30.
Case J needs 8 coordinator minutes and no supervisor time, with completion by 14:30.
The supervisor may approve one case at a time, and the coordinator may prepare one case at a time.
The two roles can work simultaneously; assume action execution requires no additional time in this exercise.

**Deliverable:** Produce a minute-by-minute schedule and identify its capacity buffer.
**Acceptance criteria:** Meet all three deadlines, observe each approval prerequisite, and explain one disruption that would make the schedule infeasible.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The two identical departure messages are duplicate deliveries of one reported event, assuming their source identifier and payload confirm the match.
Keep the evidence and delivery history as required while avoiding duplicate active work.
The ETA is a forecast and can generate an alert; it is not the warehouse's accepted appointment or a customer promise.
The expected missed slot is an exception if a useful decision is required.
The warehouse contact is an action.
One shipment-slot case is reasonable initially, with the departure and ETA linked as evidence.
A separate case can be justified if another independent decision or owner is needed, with the relationship recorded.

### Exercise 2

Reject the direct qualified-to-closed resolved transition in the core state machine.
Route the case to an accountable owner or escalate the missed acknowledgment to the staffed backup.
Verify that the booking message refers to the correct shipment, is authentic enough for the process, and represents an authorized action.
The owner can then record the decision and move through resolved pending verification to closed when the operational criterion is satisfied.
A booking confirmation does not prove actual delivery; schedule the separate delivery outcome check if required.
If the case was genuinely invalid or duplicate, a controlled supervisor closure may be possible under an explicitly defined exception path.

### Exercise 3

```text
Expected loss without action = 0.50 × USD 12,000 = USD 6,000
Expected loss with action    = 0.20 × USD 12,000 = USD 2,400
Incremental expected benefit = USD 6,000 - USD 2,400 - USD 1,100
                             = USD 2,500
```

The answer assumes the action probabilities refer to the same appointment-loss outcome and horizon, and that USD 1,100 is incremental and not already included in the conditional loss.
It is an estimate, not realized savings.

### Exercise 4

Start X at 10:00 and finish at 10:10, before its 10:15 cutoff.
Then start Y at 10:10 and finish at 10:30, before its 11:30 cutoff.
Both estimated benefits remain attainable under the simplified assumptions.
Starting Y first would finish at 10:20 and miss X's cutoff, despite Y's larger estimated benefit.
If Y can be interrupted or the times are uncertain, reassess using those facts rather than assuming the deterministic schedule is guaranteed.

### Exercise 5

A strong answer names the plant planner, buyer, or other staffed role as the single case owner and explains why that role can coordinate the production decision.
The carrier executes or confirms the transport option; the buyer supplies supplier information; the supervisor approves premium spend within the applicable limit.
A named backup accepts if the owner is absent or the acknowledgment deadline expires.
A transfer requires the recipient's acceptance, current evidence, next action, and cutoff.
The supervisor or duty manager resolves an unaccepted handoff before the option expires.
Different ownership choices can be valid when they match the local authority matrix.

### Exercise 6

```text
45 cases/day × 24 minutes/case = 1,080 minutes/day
1,080 minutes/day ÷ 60 = 18 staff-hours/day
4 analysts × 6 effective hours/day = 24 staff-hours/day
18 ÷ 24 = 0.75 = 75% average utilization
```

The arithmetic leaves six effective staff-hours per day on average, before allowing for any omitted work.
It does not describe peak arrival periods, high-percentile service time, skill-specific bottlenecks, absences, or the starting backlog.
Segment the data, examine deadline breaches and distribution tails, and provide surge coverage if necessary.
Do not claim 75% utilization proves that every case will meet its deadline.

### Exercise 7

One defensible design transfers the case before 16:00, with acceptance by 15:55 and a duty-manager fallback if acceptance is missing.
The incoming owner chases the carrier at 16:10 if no response has arrived and escalates by a pre-agreed time such as 16:15 to a supervisor who can approve the alternative.
The supervisor is asked to acknowledge promptly, for example by 16:18, with a named backup if unavailable.
The exact times require knowledge of the booking's execution duration; the key is leaving enough time before 16:30.
Waiting does not stop the booking cutoff.
The outgoing owner retains responsibility until acceptance or an explicit manager takeover.

### Exercise 8

A strong dashboard could use the following definitions:

1. **Acknowledgment compliance:** cases acknowledged by their defined acknowledgment deadline divided by cases whose deadline fell within the reporting week; report unacknowledged and clock-exempt cases separately.
2. **Deadline preservation:** cases where a selected feasible action was decided and executed before its action cutoff divided by cases with a documented applicable cutoff that matured in the week; separate cases with no feasible action.
3. **Mature on-time outcome:** cases in a specified creation cohort with delivery commitments already due and verified on time divided by cases in that cohort with a mature, observable delivery outcome; report missing and immature outcomes as separate counts.
4. **Backlog age:** open eligible cases at the weekly snapshot grouped by elapsed age from creation; state whether elapsed or business time is used and report both count and age percentiles.

Reconcile an independent feed of late deliveries or customer complaints to the case population and investigate unmatched records.
Segment by case type, shift, and lane where privacy and sample size permit.
Do not sum gross expected exposures and call them verified savings.
For effectiveness, define an eligible population, comparison strategy, outcome horizon, intervention costs, and uncertainty.

### Exercise 9

The claim treats outcomes not yet observable as successes, so it is censored.
Queue-linked records omit unlinked failures and may exclude incidents never detected by the queue.
Closed cases may differ systematically from open cases in severity, resolution ease, or age.
Even a completely observed on-time outcome does not establish that the action caused it; it might have happened without intervention.

Use a mature cohort defined before outcome observation, link to an independent outcome source, and report missing linkage and open cases.
Compare intervention and non-intervention outcomes only with a design that addresses eligibility and selection, such as an ethically appropriate randomized rollout or a carefully adjusted comparison.
Report uncertainty, costs, changes in baseline commitments, and the limits of causal interpretation.

### Exercise 10

Prepare G from 14:00 to 14:10 and approve it from 14:10 to 14:13.
Prepare H from 14:10 to 14:22 and approve it from 14:22 to 14:27.
Prepare J from 14:22 to 14:30.
The coordinator uses 10 + 12 + 8 = 30 minutes and the supervisor uses 3 + 5 = 8 minutes.
Both available capacity buffers are zero, although the timing of each role's work differs.
For example, if G takes 13 coordinator minutes, its approval finishes no earlier than 14:16 and misses the 14:15 cutoff.
Escalate before the deadline if the revised estimate makes the option infeasible; seek a genuine parallel resource or an alternative action rather than counting G's projected benefit automatically.
Other schedules are acceptable if they meet the prerequisites and all three deadlines.

## 14. Interview questions

1. **What is the difference between an event and an exception?**  
   A strong answer separates evidence from a decision-relevant deviation and gives a correlation example.  
   A weak answer makes every event a case.
2. **How do you prevent a queue from losing ownership at a shift change?**  
   A strong answer specifies an accepting recipient, current action and deadline, backup, and fallback.  
   A weak answer treats an email to a team inbox as a completed handoff.
3. **When would you prioritize a lower expected-loss case first?**  
   A strong answer discusses action cutoffs, protected obligations, feasibility, and processing time.  
   A weak answer assumes the largest dollar number always wins.
4. **How would you calculate the value of a proposed expedite?**  
   A strong answer compares outcome distributions with and without the action and subtracts incremental cost.  
   A weak answer calls the entire shipment value saved.
5. **What fields are required to put a case into waiting?**  
   A strong answer names the blocker, responsible party, chase time, next review, and continuing deadline.  
   A weak answer records only “pending.”
6. **What is the difference between escalation and transfer?**  
   A strong answer retains the owner during a request for authority unless a recipient explicitly accepts transfer.  
   A weak answer assumes copying a manager transfers the work.
7. **How do you size a queue team?**  
   A strong answer uses qualified arrivals, active service time, effective capacity, variability, skills, and coverage.  
   A weak answer divides raw event count by scheduled headcount.
8. **How would you verify that a case can be closed?**  
   A strong answer uses a reason-specific criterion, authority, evidence, and a later outcome check where needed.  
   A weak answer treats any status update as proof of resolution.
9. **How can a queue look healthy while service deteriorates?**  
   A strong answer mentions missed detection, low-quality closures, changing entry thresholds, and immature outcomes.  
   A weak answer relies only on backlog count.
10. **How do you handle late or corrected event data?**  
    A strong answer preserves event and processing times, versions evidence, and re-evaluates priority without erasing history.  
    A weak answer overwrites the case with the last message received.
11. **What does a common event standard solve?**  
    A strong answer distinguishes interoperable data structures from ownership, data quality, incentives, and authority.  
    A weak answer claims standards automatically produce a correct queue.
12. **How would you show that an intervention caused a benefit?**  
    A strong answer defines a counterfactual, eligible cohort, outcome maturity, costs, and bias controls.  
    A weak answer sums gross exposures for all acted-on cases.

## 15. Key terms

- [Event](#4-definitions-and-vocabulary) — an observation with time and provenance.
- [Alert](#4-definitions-and-vocabulary) — notification of a possible condition.
- [Exception](#3-the-core-mental-model) — a decision-relevant deviation or threat.
- [Case](#3-the-core-mental-model) — managed record of a business situation.
- [Correlation](#step-2--establish-case-identity-and-correlation) — linking evidence to the right situation.
- [Deduplication](#step-2--establish-case-identity-and-correlation) — preventing repeated representations from multiplying work.
- [Parent case](#step-2--establish-case-identity-and-correlation) — aggregate record for a shared impact.
- [Backlog](#step-9--control-work-in-progress-and-staffing) — eligible unresolved work at a point in time.
- [Arrival rate](#step-9--control-work-in-progress-and-staffing) — cases entering a queue per unit time.
- [Service time](#7-code-query-or-calculation) — active labor per case.
- [Lead time](#4-definitions-and-vocabulary) — elapsed time between defined milestones.
- [Acknowledgment](#step-4--assign-ownership-and-coverage) — acceptance of responsibility.
- [Owner](#step-4--assign-ownership-and-coverage) — accountable person or staffed role.
- [Backup](#step-4--assign-ownership-and-coverage) — alternate who accepts under a defined trigger.
- [Decision right](#step-4--assign-ownership-and-coverage) — authority to approve a specified action.
- [Expected loss](#step-5--compute-the-economic-view) — probability times conditional loss.
- [Incremental expected benefit](#step-5--compute-the-economic-view) — expected loss reduction less action cost.
- [Intervention deadline](#step-6--add-time-feasibility-and-protected-classes) — last practical time for a named action.
- [Commitment](#6-worked-example) — an authorized promise, not a forecast.
- [Forecast](#6-worked-example) — an estimate of a future outcome.
- [Plan](#6-worked-example) — the chosen intended course of action.
- [Target](#6-worked-example) — an internal desired performance level.
- [Waiting](#step-7--manage-the-lifecycle) — a state with a named dependency and review.
- [Escalation](#step-8--specify-clocks-and-escalation) — request for additional authority or support.
- [Handoff](#step-4--assign-ownership-and-coverage) — accepted transfer of accountability.
- [Work in progress](#step-9--control-work-in-progress-and-staffing) — work actively handled under a defined scope.
- [Utilization](#7-code-query-or-calculation) — required work divided by effective capacity.
- [Closure reason](#step-10--close-and-follow-through) — recorded explanation of why active work ended.
- [Censoring](#step-11--measure-and-learn) — an outcome not yet observable at cutoff.
- [Audit trail](#8-data-standards-and-regulatory-context) — controlled history of decisions and changes.

## 16. Sources

1. GS1, [EPCIS and CBV implementation resources](https://www.gs1.org/standards/epcis). Official standards entry point for EPCIS visibility events; consult the applicable version for normative requirements.
2. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 3 and 5. Official legal text; applicability depends on the facts and jurisdiction.
3. John D. C. Little, “A Proof for the Queuing Formula: L = λW,” *Operations Research*, 1961, [DOI: 10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). Foundational queueing result; its assumptions and population definitions matter when applying it.
4. Mor Harchol-Balter, [*Performance Modeling and Design of Computer Systems*](https://www.cs.cmu.edu/~harchol/PerformanceModeling/), Cambridge University Press, 2013. Queueing and variability background, not a supply-chain-specific staffing rule.
5. IBM, [Lenovo case study](https://www.ibm.com/case-studies/lenovo). Company case material; claims are not treated here as independent causal estimates.
6. A.P. Moller–Maersk, [“A.P. Moller - Maersk and IBM to discontinue TradeLens, a blockchain-enabled global trade platform”](https://www.maersk.com/news/articles/2022/11/29/maersk-and-ibm-to-discontinue-tradelens), 29 November 2022. Company announcement about platform discontinuation.
7. National Institute of Standards and Technology, [NIST Privacy Framework](https://www.nist.gov/privacy-framework). Voluntary risk-management resource; it does not replace applicable law.

Sources distinguish official standards and legal text from company accounts and general analytical literature.
The hypothetical case values and queue policy are teaching assumptions, not sourced industry benchmarks.
Regulatory context is educational and is not legal advice.

## 17. Links

- [Module 19 index](_index.md)
- [Module 19 survey](../../../course/19_control_towers_visibility_events.md)
- [Visibility layers](01_visibility-layers.md)
- [Event streams and EPCIS](02_event-streams-epcis.md)
- [Event time versus processing time](03_event-time-processing-time.md)
- [ETA prediction](04_eta-prediction.md)
- [Delay propagation](05_delay-propagation.md)
- [Labs and electives index](../../05_labs_electives/_index.md)
