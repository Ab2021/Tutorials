# Event Time versus Processing Time

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06 and 12; Module 19 topics 01 and 02  
> Version 2026.5

## 1. Why this topic exists

A control-tower operator may see an exception after the opportunity to act has already passed.
The shipment may have arrived on time, but its arrival scan was uploaded after a connectivity outage.
Conversely, a recently received message can describe a delay that started hours earlier.
The time at which information appears on a dashboard is not necessarily the time at which the underlying business event occurred.

Consider a retailer with a delivery appointment ending at 10:00 UTC.
A truck reaches the gate at 09:52 UTC, while the carrier's handheld device uploads the scan at 10:18 UTC.
At 10:05 UTC, the operator has no arrival evidence and may reasonably start a check.
At 10:18 UTC, the correct historical arrival is 09:52 UTC, not 10:18 UTC.
A completed delivery history can therefore change after an operational decision was made.
That change does not imply that the operator acted irrationally with the information available at 10:05 UTC.

The reverse is also possible.
A gate-out event with an occurrence time of 08:05 UTC may arrive after a gate-in event recorded at 09:52 UTC.
If the application treats arrival order as physical order, a delivered truck can appear to return to transit.
A retried arrival message can create two alerts, two delivery counts, or two customer notifications unless the application recognizes the retry.
A later authorized correction can legitimately change the event without being an ordinary retry.

These distinctions affect dispatchers, customer-service teams, warehouse schedulers, inventory planners, and analysts.
They also affect the carrier whose performance is judged from a timestamp.
A control tower must choose when to act provisionally, when to revise, and when to close reporting periods.
It must make those choices visible rather than bury them in a streaming configuration.

This lesson uses a fictional grocery shipment.
All money amounts, event times, and business rules are illustrative, not observed industry benchmarks.
The examples use Coordinated Universal Time (UTC), unless explicitly stated otherwise.
The decision is whether to release a backup delivery slot; it is not a forecast of arrival, a contractual commitment, or a legal determination.

## 2. Learning outcomes

After this lesson you can:

1. **Distinguish** occurrence, source recording, tower receipt, and processing times; Exercise 1 tests the distinction.
2. **Reconstruct** an as-known-at-the-time view and a corrected event-time view; Exercises 2 and 3 test both.
3. **Calculate** lateness, an event-time window, and the effect of a late correction; Exercises 3 and 4 test the arithmetic.
4. **Choose** a watermark and allowed-lateness policy based on an action deadline and a stated loss trade-off; Exercise 5 asks for a policy.
5. **Differentiate** a retry, a distinct physical event, and a source-authorized correction; Exercise 6 tests identity rules.
6. **Design** an auditable exception lifecycle with idempotent actions and reversible or compensating steps; Exercise 7 assesses the design.
7. **Evaluate** timeliness and correction controls without confusing absence of evidence with evidence of absence; Exercise 8 tests monitoring.
8. **Explain** the limits of standards, processing guarantees, and public case evidence; the interview questions test this outcome.

A good answer identifies the decision owner, deadline, evidence available at that deadline, and treatment of later evidence.
It states which results are provisional and which historical versions remain accessible.

## 3. The core mental model

```text
PHYSICAL WORLD     depart 08:05 -------- gate arrival 09:52 ---- actual delivery
                       |                         |
SOURCE RECORD          |                    scan recorded 09:53
                       |                         |
TRANSPORT          buffered / retried      buffered until 10:18
                       |                         |
TOWER ARRIVAL      received 10:23          received 10:18
                       |                         |
PROCESS             validate -> identity -> event-time order -> state / windows
                                                     |
OPERATE             at 10:05: arrival UNKNOWN -> investigate / reserve backup
                                                     |
REVISE              at 10:18: evidence of 09:52 arrival -> revise current state
                                                     |
AUDIT               retain 10:05 knowledge, actions, 10:18 correction and reason
```

The physical world and the tower's knowledge are separate timelines.
A recorded event is an assertion by a source, not a direct guarantee of physical truth.
Event time is the source's claimed time of occurrence.
Processing time is when the application performs a computation.
Receipt time is when the tower first receives the message.
Source recording time may differ from all three.

A valid historical view asks: what events are now believed to have happened, and in what event-time order?
An as-known view asks: what could the operator see at a particular wall-clock time?
These are different questions and should not share a single mutable timestamp.

A watermark is a processing system's estimate of progress through event time.
It helps decide when to emit a provisional result or apply a late-data policy.
A watermark does not prove that every earlier event has arrived.
Watermark behavior varies with the engine and configuration; the lesson's worked policy is an explicit business convention, not a universal rule.

Late evidence can change an aggregate, the present shipment state, or both.
It does not reverse a physical action already taken.
A backup truck already dispatched may require a cancellation or a compensating action.
A corrected report must retain the prior result and explain why it changed.

Three boundaries deserve separate ownership:

- The **decision deadline** is when an operator must act or deliberately accept the risk of waiting.
- The **event-time window boundary** defines which physical occurrences belong in a metric.
- The **revision cutoff** determines when routine automatic updates stop; it does not make subsequent evidence false.

A single setting cannot safely represent all three.
The workflow below binds them to a decision and a documented correction path.

## 4. Definitions and vocabulary

| Term | Definition and units | Common misuse |
|---|---|---|
| Physical occurrence | A real-world activity at an actual place and time | Treating an assertion as independent proof |
| Event | A recorded assertion about an occurrence or condition | Treating every message as a separate occurrence |
| Event time | Source-asserted occurrence instant, stored with offset or UTC | Equating it with a dashboard refresh time |
| Capture time | Time a source first records the observation | Assuming capture was instantaneous |
| Receipt time | Time the tower first durably receives the message | Calling it the physical arrival time |
| Processing time | Wall-clock time at which a computation executes | Assuming it is stable across a replay |
| Publication time | Time the producer sends or publishes an event | Assuming it is when the source observed the event |
| Ingestion lag | Receipt time minus event time, in minutes or seconds | Treating negative lag as impossible rather than a clock warning |
| Processing lag | Processing time minus receipt time, in minutes or seconds | Counting transport delay as internal compute delay |
| Out of order | Receipt or processing order differs from the relevant event order | Assuming it necessarily means bad data |
| Late event | Event arriving after a defined deadline, watermark, or result emission | Using late without naming the reference |
| Watermark | Estimate of event-time progress used by a processing policy | Treating it as proof of completeness |
| Allowed lateness | Extra event-time or processing-policy tolerance for updates, engine-specific | Assuming the same semantics in every engine |
| Window | Set of events defined by a time interval and grouping keys | Omitting the boundary convention |
| Half-open interval | Interval including its start and excluding its end, written [start, end) | Including a boundary event in two windows |
| Trigger | Rule that emits a result or update | Treating an early result as final |
| Revision cutoff | Business rule after which routine automatic revisions stop | Declaring later evidence impossible |
| Provisional result | Result explicitly subject to change under the policy | Presenting it as settled performance |
| Deduplication | Recognition of repeated delivery of the same logical event | Removing distinct events with similar content |
| Idempotency | Repeating an operation has the same intended effect as performing it once | Claiming it guarantees the physical world was changed once |
| Event identity | Stable, source-scoped identifier for a logical event | Using shipment ID alone as an event ID |
| Revision | Authorized change or replacement of an earlier assertion | Treating changed payload as an ordinary retry |
| Supersession | Explicit relation identifying the assertion replaced | Deleting the old assertion without lineage |
| Retraction | A statement that an earlier assertion should no longer be used as valid | Silently overwriting the prior value |
| Replay | Reprocessing retained messages under a specified version and scope | Re-sending irreversible actions automatically |
| As-known view | Result using only information received by a chosen cutoff | Reconstructing it from only the latest record |
| Corrected view | Latest accepted interpretation for a specified event-time period | Pretending it was available at the earlier decision |
| Decision deadline | Latest time for a named intervention | Confusing it with a reporting cutoff |
| Source clock skew | Difference between a source clock and trusted reference, in seconds | Interpreting all negative lag as time travel |
| Dead-letter queue | Holding area for records that cannot safely be processed | Treating a holding area as a completed resolution |
| Event-time completeness | Estimated coverage of expected occurrences for a specified population and period | Measuring only arrivals and calling the denominator complete |
| Exactly-once effect | An effect whose result is applied once under stated boundaries | Assuming a broker guarantee covers calls to external partners |
| Backfill | Ingestion or processing of historical records after the original period | Mixing it with live actions without a replay label |
| State transition | Rule-based change in the operational interpretation of a shipment | Using the last received message as the latest physical state |
| Exception | A condition requiring attention under a business rule | Assuming every missing message proves a physical delay |

The term **late** always needs a reference.
An event can arrive after an operator's decision deadline yet before a reporting window closes.
It can be on time for a daily metric but too late for a missed dock appointment.

## 5. The mechanics, step by step

### Step 1 — Identify the decision and the irreversible boundary

**Input:** the operational decision, its owner, cost, and deadline.

**Action:** write the decision as a conditional question, not as a dashboard requirement.
For example: by 10:05 UTC, should the dispatcher release a backup dock slot for a shipment due by 10:00 UTC?
Identify who can reserve the slot, who can cancel it, and when cancellation stops being free.

**Output:** a decision record containing shipment ID, owner, decision deadline, options, costs, and evidence requirements.
A reservation is a plan, not proof that the shipment is late.
A customer promise is a commitment, not an event observation.
An estimated arrival is a forecast and must remain labeled as such.

**Branch:** if the decision can be reversed cheaply, an early provisional action may be sensible.
If it is irreversible or costly, require stronger evidence or an explicit risk authorization.
Neither branch justifies fabricating a physical status.

### Step 2 — Define the expected event population

**Input:** shipment milestones, carrier obligations, shipment identifiers, and planned routes.

**Action:** enumerate which events are expected, from which source, and within which time range.
A planned arrival is not itself an actual arrival.
Associate each expected event with its object, shipment, location, business meaning, and source.

**Output:** an expectation table with a denominator for missing-event and coverage metrics.
Include exceptions for legitimately canceled shipments and changed routes.
Version the expectation when a plan changes, with an effective time and an approval trail.

**Branch:** if a source does not provide a particular milestone, do not count its absence as a failed transmission without an agreed obligation.
If the event is expected but not yet received, show **unknown**, not **did not occur**.

### Step 3 — Preserve the distinct clocks

**Input:** source payload, source metadata, and tower receipt metadata.

**Action:** retain the source's original timestamp and offset, source recording time if available, producer publication time if available, tower receipt time, and processing time.
Normalize unambiguous instants to UTC for comparisons while keeping the original representation.
Record clock quality and timezone assumptions rather than silently inventing an offset.

**Output:** an immutable raw envelope and a normalized event record with named clocks.
Timestamp precision is part of the evidence: a minute-level timestamp does not establish ordering within that minute.

**Branch:** quarantine an unparseable or ambiguous local time for resolution, or apply a documented source-specific rule if authorized.
Flag event times beyond a plausible future-skew tolerance.
A negative ingestion lag can reveal an incorrect source clock or an incorrectly assigned timezone.
It should not be zeroed to make a quality metric look better.

### Step 4 — Establish identity before deduplicating

**Input:** authenticated source, producer event ID, shipment ID, payload, and source version.

**Action:** prefer a stable key such as `(source_id, event_id)` when the producer contract guarantees its uniqueness and stability across retries.
Record the transport message ID separately: it may change on every delivery attempt.
Do not use shipment ID alone, because one shipment legitimately produces many milestones.

**Output:** a logical-event identity and a record of each received delivery attempt.
An exact retry under the same stable ID and unchanged semantic payload must not add another physical milestone.
A same-ID, different-payload message must enter a correction or conflict path rather than being silently discarded.

**Branch:** if no stable ID exists, negotiate one.
A content fingerprint can be a provisional aid, but timestamp rounding or identical repeated scans may collapse distinct occurrences.
Make that limitation visible and route uncertain matches to a reconciliation process.

### Step 5 — Classify the incoming message

**Input:** incoming message, existing logical event, source authorization, and correction relation.

**Action:** classify it as a new assertion, exact retry, authorized revision or retraction, conflicting assertion, or invalid record.
A revision should identify the event it supersedes and carry a revision identity or version.
Its author must have the authority required by the data contract.
Preserve the original and the revision with receipt times.

**Output:** a classified record and a reason code.
Distinct event IDs do not automatically mean distinct physical activity: a source may violate its identity contract.
Conversely, similar payloads do not prove duplication.

**Branch:** an unrecognized conflicting revision is held for investigation.
An authenticated retry is acknowledged without applying a second operational effect.
A valid correction starts a state and metric reassessment.

### Step 6 — Decide the temporal ordering rule

**Input:** accepted events for the same object and the business milestone model.

**Action:** order by event time where occurrence order is the question.
Use an explicit tie policy, such as source sequence when reliable, and otherwise mark the order as uncertain.
The chosen tie-breaker should make computations deterministic without claiming physical certainty.

**Output:** an ordered, provenance-bearing set of accepted assertions.
Compare the result to allowed business transitions.
A late departure preceding an arrival should normally enrich the history rather than move a delivered shipment back into transit.

**Branch:** when event times conflict with an impossible route or a trusted source, flag a conflict.
Do not automatically select the latest received message as the truth.
When two distinct sources disagree, apply the agreed authority hierarchy or seek human resolution.

### Step 7 — Design windows and emissions

**Input:** metric definition, event-time field, grouping key, and business reporting frequency.

**Action:** define intervals explicitly, for example hourly UTC windows `[09:00, 10:00)`.
An event at exactly 10:00 belongs in the next window.
Specify whether a metric counts shipments, events, distinct shipments with an arrival, or something else.

**Output:** a reproducible metric contract, including event selection, exclusion, and correction rules.
A metric's window end is not necessarily the time at which the metric is published.
A provisional count can be published while earlier events are still in transit.

**Branch:** if the business requires a quick operational count, emit early and label it provisional.
If financial settlement requires a stronger close, choose a longer cutoff and an explicit post-close adjustment process.
A longer wait may improve completeness but can miss the intervention deadline.

### Step 8 — Choose watermark and lateness policy

**Input:** measured lag distribution by source, source outage history, decision loss, and revision cost.

**Action:** choose an initial watermark policy and compare it with service requirements.
One simplified policy is `watermark = maximum observed event time - 15 minutes` for a partition.
In a real engine, partition idleness, input aggregation, timestamp extraction, trigger semantics, and state retention must be configured and checked.
A maximum-observed-time rule is vulnerable to a future-dated bad timestamp, so validate timestamps before advancing progress.

**Output:** a named policy with parameters, owners, version, expected latency, and handling after cutoff.
For this lesson's daily reporting illustration, a window can be updated when late data arrives until the watermark passes its end plus 30 minutes.
The exact inequality and engine behavior must be tested in the chosen implementation.

**Branch:** if the source is idle, a watermark may stall and delay results; an idle-source rule trades possible missed late data for progress.
If records arrive beyond the revision horizon, retain them and route them to controlled reconciliation rather than discarding them as false.
A watermark is an estimate; it is not a contractual proof that no event remains.

### Step 9 — Act on knowledge available at the deadline

**Input:** the as-known state, current time, ETA forecast if available, action cost, and escalation policy.

**Action:** determine what is known and unknown at the decision deadline.
Record the actual evidence version used, the operator, and the choice.
An exception can say “arrival evidence missing at 10:05 UTC”; it should not say “arrival occurred late” without evidence.

**Output:** a decision and an action record linked to the evidence available at the time.
A notification should include a stable action ID and a reason.
An operational state can be provisional even while an action is appropriate.

**Branch:** if a low-cost check is available, contact the carrier before authorizing an expensive backup.
If the deadline has passed, escalate or choose a fallback according to delegated authority.
Do not wait for a reporting watermark when a live business decision must be made sooner.

### Step 10 — Handle duplicates and action side effects

**Input:** logical event identity, exception identity, action request, and prior action ledger.

**Action:** derive a stable idempotency key for each intended business action.
For example, `(shipment_id, exception_type, decision_version)` can identify one backup-slot reservation if the business contract permits it.
Use a durable ledger or transactional outbox to connect accepted state changes to outbound requests.
External partners must also support idempotent requests, or the caller needs reconciliation and compensation.

**Output:** one recorded intended action and its delivery attempts and acknowledgments.
The transport can retry; the receiving business operation should not create a second reservation for the same key.
Keep a trace of retries even when their business effect is suppressed.

**Branch:** if an acknowledgment is lost, query by idempotency key before sending a non-idempotent request again.
If the external system cannot query or deduplicate, route uncertainty to an operator rather than asserting exactly-once execution.
A messaging system's delivery guarantee is not proof of exactly-once physical dispatch.

### Step 11 — Revise state and metrics when evidence changes

**Input:** late accepted assertion or authorized correction, prior state, metric versions, and action history.

**Action:** recompute the affected object and event-time windows using the accepted facts and the stated rules.
For a correction, remove the old assertion's contribution and add the new contribution if both fall in scope.
A revised timestamp can move an event across two windows, requiring updates to both.

**Output:** a new version with old and new values, cause, source, effective time, and publication time.
Preserve the prior as-known view and link the update to the affected exception and decision.
Tell downstream consumers whether the message is an upsert, a retraction, or a new revision.

**Branch:** before a routine cutoff, apply the automatic revision policy.
After cutoff, send the case to an authorized reconciliation process and issue an adjustment if approved.
If a late event invalidates an open alert, resolve it with a reason; if it invalidates a completed action, record the cost and any compensation.

### Step 12 — Replay safely

**Input:** retained raw messages, code and policy version, replay scope, and action configuration.

**Action:** distinguish analytical reconstruction from live operation.
Run historical replay with external actions disabled or in a designated sandbox.
If a repair must emit live corrections, explicitly authorize the scope, deduplicate output, and reconcile affected consumers.

**Output:** a replay manifest, reproducible results, and a list of deliberate corrections.
Include the input range, source partitions, software version, and policy version.
Make it possible to distinguish original decisions from later reconstructions.

**Branch:** if historical identities or timestamp semantics are unavailable, disclose that the reconstruction is approximate.
If a replay would issue an irreversible command, stop and obtain the decision owner's authorization.

### Step 13 — Monitor and improve by source and decision

**Input:** expected event population, receipt logs, event-time values, incident records, and action outcomes.

**Action:** monitor ingestion lag percentiles, negative lag, missing coverage, duplicates, correction frequency, stalled watermarks, and events beyond revision cutoff.
Use eligible expected events as the denominator for coverage; exclude only documented ineligible cases.
Track metrics by source, lane, milestone, and time period where sample size permits.

**Output:** a dashboard and a queue of owners with thresholds, escalation, and follow-up dates.
Measure decision usefulness separately from technical latency: an event can meet a daily service level yet miss an operational decision.
Compare an action's timestamp with the arrival of decision-relevant information.

**Branch:** if lag increases, investigate source buffering, connectivity, clock quality, and internal processing independently.
If apparent completeness improves while expected-event coverage falls, inspect the denominator and late records.
A metric without an owner and a response is only an observation.

### Operational design review: one late message, several decisions

The same event can be useful for one decision and useless for another.
Suppose a facility uses the gate-arrival scan for four purposes.
The first purpose is to decide whether to hold a dock.
The second is to tell customer service whether the shipment reached the site.
The third is to calculate the carrier's daily arrival performance.
The fourth is to investigate a disputed accessorial invoice.

These purposes need different timing and confidence.
The dock decision may expire in minutes.
Customer service may still benefit from a later update, provided the original observation time is visible.
The daily performance measure can be revised after the operating day.
An invoice dispute may require source documents and approval well after the routine reporting cutoff.

Do not force the four purposes into a single binary field called `late`.
Instead, define separate questions.
Was the event received before the dock decision?
Was the asserted arrival before the appointment end?
Was the first result included in the daily publication?
Was the evidence accepted for the invoice dispute?
Each answer has a different reference and owner.

The event's relevance also changes as the workflow progresses.
Before a decision, it can prevent an unnecessary backup reservation.
After the reservation, it can support cancellation.
After cancellation, it can explain the fee.
After reporting close, it can support an adjustment rather than an automatic rewrite.
The event's event time does not change merely because its business use changes.

### Compare alternative decision rules

Suppose a dispatcher has three choices when no arrival event is visible.
The first is to reserve backup capacity immediately.
The second is to contact the carrier and wait briefly.
The third is to do nothing.
The preferred choice depends on the time remaining, contact success rate, and losses of an unnecessary or missed reservation.

Assume an immediate reservation costs USD 85 regardless of use.
Assume that doing nothing incurs USD 420 only when the appointment is missed.
If the probability of a miss is 10%, the expected cost of doing nothing is USD 42.
Under these simplified assumptions, that is lower than USD 85.
If the probability is 35%, expected cost is USD 147.
An immediate reservation then has the lower modeled cost.

Now suppose a carrier call costs USD 8 in staff time.
Suppose it gives a definitive answer with probability 60% before the reservation deadline.
If it gives no answer, the dispatcher still has time to reserve.
If a definitive answer permits the dispatcher to reserve only when the appointment will actually be missed, the expected modeled cost at a 35% miss probability is USD 8 plus 35% of USD 85, or USD 37.75.
That figure assumes the call result is perfectly accurate and the reservation remains available.
Those are strong assumptions, not established facts.
If the call can consume the remaining deadline or return inaccurate information, the expected cost increases.

The point is not that calling is always optimal.
The point is that information has value only if it arrives before the relevant action can still be taken.
A late but accurate message can improve reporting while providing no opportunity to avoid a fee.
A fast but unreliable message can lead to a worse decision.
Choose the policy from measured outcomes, not a latency target alone.

### Record the information set

A defensible decision record needs more than the chosen option.
It should identify the decision's timestamp and the time standard used.
It should identify the shipment and the commitment being protected.
It should identify the rule version or the reason for discretionary override.
It should show the evidence and forecasts visible to the dispatcher.
It should name the actor or automated service that made the choice.

The record should include the estimate's provenance where practical.
For an ETA model, retain the model version and prediction timestamp.
For a carrier phone call, retain the call time, source, and substance of the answer.
For a missing scan, retain the query time and the system checked.
Do not record only the latest status, because the latest status can incorporate evidence unavailable to the original decision.

The record also needs the action's lifecycle.
An issued reservation request is not the same as a confirmed reservation.
A cancellation request is not the same as a confirmed cancellation.
An external acknowledgment may arrive later or never arrive.
Represent those facts independently rather than advancing directly from request to success.

When evidence changes, append a link to the revised evidence and any resulting action.
Do not rewrite the earlier information set to include the later event.
The result should support two legitimate questions:
“What is our best present understanding of the shipment?” and
“Why did the dispatcher reserve a backup at 10:05 UTC?”

### Reason about timestamp quality before ordering

Timestamp fields are not equally reliable merely because they have the same format.
A site scanner connected to a managed time service may have a different error profile from a driver-entered timestamp.
A location-based estimate can be generated automatically but may represent proximity rather than a gate crossing.
A source can also apply a timestamp when it uploads a stored observation instead of when the observation occurred.

For each producer, request a field-level semantic definition.
Ask whether occurrence time is captured by the device or entered by a person.
Ask whether offline events retain the original device clock.
Ask whether a timestamp can be edited later and how edits are represented.
Ask what happens if the device has no trusted clock.

Keep timestamp precision alongside the value.
An event recorded at `09:52` with minute precision may have occurred at any instant in that minute under a truncation convention.
Another event displayed as `09:52` may have been rounded from a different instant.
Sorting the two values by event ID yields reproducible output, but does not establish the actual physical sequence.
If the sequence changes an intervention or dispute, mark it uncertain and seek additional evidence.

Apply source-specific plausibility checks carefully.
A future timestamp can be a device error, a timezone error, or a legitimately scheduled future activity.
An ETA prediction for tomorrow is not a future physical-arrival assertion.
Validate the semantic field and event type before using a generic future-time threshold.
Do not reject legitimate forecasts simply because their predicted value is later than receipt.

### Source outages and recovery

When an entire carrier feed stops, a missing-arrival alert for every shipment can flood the queue.
The data-quality incident and the individual business exposures are related but not identical.
Group the technical symptom at the source level.
Keep shipment-level risk visible for shipments with imminent decisions.
One source incident can explain many unknown statuses without proving that any truck is delayed.

During an outage, freeze or label freshness indicators appropriately.
Show the last successful receipt and the expected update cadence.
Do not keep displaying a stale “in transit” status as though it were a current observation.
If a manual carrier call provides evidence, record its source and time separately from the automated feed.

When connectivity returns, buffered events may arrive in a burst.
The burst can contain old events, retries, and corrections.
Apply the same identity and authorization rules as for live traffic.
Do not prioritize arrival order over physical order merely to drain the backlog faster.
Protect downstream systems against a flood of notifications for events that no longer require action.

Backlog recovery should have an operator-visible completion criterion.
For example, reconcile the expected source sequence or agreed batch count if the source provides one.
If no such reference exists, report uncertainty about completeness.
Review expired decisions separately from still-actionable exceptions.
Issue current notifications only when their consequences remain relevant.

### Partition and watermark behavior

A stream processor may read several carrier partitions.
Each partition can advance through event time at a different rate.
An engine may combine progress across inputs using rules that make the slowest active input important.
The precise rule depends on the system and configuration.
Document the actual behavior rather than assuming that the maximum timestamp in the entire stream defines global progress.

Consider two inputs under an illustrative minimum-of-input-watermarks rule.
Carrier A has watermark 10:20 UTC.
Carrier B has watermark 09:45 UTC.
The combined watermark is 09:45 UTC while both are active.
An hourly window ending at 10:00 UTC has not yet passed that combined watermark.
Carrier A's timely data alone does not close the shared window.

If Carrier B is declared idle and excluded by policy, the combined watermark may advance to 10:20 UTC.
That improves reporting latency.
If B later sends a 09:55 UTC event, the record is late relative to that progress.
The application must know whether its state is still retained and whether its policy permits a revision.
An idle-input policy is therefore a business trade-off, not a harmless performance switch.

Source time validation matters before watermark advancement.
Suppose an invalid device timestamp says tomorrow at 12:00 UTC.
If a simplistic maximum-based watermark accepts it, normal current-day records may appear extremely late.
A plausibility check can quarantine the questionable assertion.
Investigate and correct the source clock; do not silently alter the original timestamp.

### Separate reporting revisions from operational action

An arrival count is a derived quantity.
A revised arrival timestamp may change the count, its window, its on-time status, or none of these.
The system should identify which derived products are affected.
It should not notify every consumer simply because an unrelated field changed.

For a changed timestamp, determine the old event-time window and the new event-time window.
If they are the same, a distinct-shipment count may remain unchanged.
An on-time classification can still change if the appointment boundary lies inside the window.
For example, a correction from 09:52 to 09:56 UTC stays in the same hourly window but could matter for an appointment ending at 09:54 UTC.
The metric contract must identify the relevant commitment version.

If a correction crosses windows, make the change atomic from the consumer's perspective where practical.
If atomic publication is impossible, identify the revision batch and allow consumers to reconcile partial updates.
A temporarily inconsistent two-window total should not be mistaken for two shipments.
Consumers that cannot process retractions need either a complete replacement snapshot or an agreed reconciliation mechanism.

Operational actions need a different response.
If an alert is still open, new evidence may resolve it.
If a request is in flight, the system may need to cancel after confirmation rather than assume no action occurred.
If the truck has already been dispatched, a cancellation message may be ineffective.
Record the real outcome and any compensating action, including who authorized it.

### Build useful lag metrics

Define first-receipt lag as the first durable tower receipt minus the accepted event-time assertion.
The original assertion and a corrected event time produce different values.
Retain both versions so that a metric change can be explained.
Do not overwrite the original lag without preserving its basis.

Choose a measurement period and eligibility rules.
An event with a missing or ambiguous event time cannot contribute a trustworthy numeric lag.
Count it in a separate quality category instead of silently excluding it.
An event with a future timestamp produces negative lag and should be tracked as a clock-quality signal.
Duplicates should generally contribute to delivery-attempt metrics, not multiple observations of logical-event timeliness.

Percentiles require a denominator.
For example, a 95th-percentile lag among 950 received events says nothing about 50 expected events that have not arrived.
Report received-event lag alongside expected-event coverage.
Also report the number of records behind each percentile.
A p95 derived from a handful of events is unstable and should not drive a major contractual conclusion without context.

Measure deadline usefulness directly.
For each eligible decision, compare the time at which relevant evidence was first received with that decision's deadline.
Count evidence received at or before the deadline separately from evidence received afterward.
Specify how equality is handled.
This metric may be more actionable than an overall lag percentile because it reflects the intervention opportunity.

Investigate distributions rather than only averages.
A source may provide most events in one minute but buffer a small fraction for several hours.
Those tail events can disproportionately affect urgent decisions.
Break down the tail by source, site, event type, connectivity state, and shift where samples support interpretation.
Avoid publishing small or sensitive groups without the necessary privacy review.

### Negotiate source obligations

A useful partner agreement distinguishes a source's responsibilities from the tower's.
The source should identify event semantics, time capture, retry identity, and authorized correction procedure.
The tower should identify receipt acknowledgment, validation, conflict handling, and the channels through which partners can investigate rejects.
Both should agree on contact ownership and incident escalation.

Define a timeliness objective only after agreeing on the start and end clocks.
“Within ten minutes” is ambiguous unless it says ten minutes from what to what.
If the event-time clock is unreliable, a purportedly precise service level may primarily measure clock error.
If receipt is recorded after a validation queue, the measure combines partner transport and tower processing.
Use separate measurements where accountability requires them.

Include a method for disputed records.
The source may supply a scanner log, signed document, location evidence, or an authorized correction.
The tower should retain both the original and the decision on the dispute.
The contract should say who can approve a post-close adjustment.
It should also say how downstream reports and invoices are notified.

Do not use a service-level penalty as a substitute for fixing an unreliable feed.
An incentive that rewards only received-event lag can encourage omission of difficult events.
Pair timeliness with coverage and correction transparency.
Review exclusions and denominator changes with both operational and data owners.

### Test the policy with controlled scenarios

Before deployment, the decision owner and data owner should agree on representative scenarios.
Use a known set of records with specified event times and receipt times.
Record the expected state, metric revision, and permitted operational action for each scenario.
The expected result is a business acceptance criterion, not merely successful message ingestion.

Include a message received before the decision deadline and one received one minute afterward.
Confirm that the first can affect the decision and the second changes only subsequent decisions and reports.
Include an exact retry before and after the first result is emitted.
Neither retry should create a second milestone or reservation.

Include an authorized correction that stays inside its original window.
Confirm that the distinct count stays constant even if an appointment classification changes.
Include another correction that crosses a window boundary.
Confirm that the old contribution is removed and the new one appears in the correct period.

Include an event that arrives after the automated revision cutoff.
Confirm that the original evidence is retained and a named owner receives a reconciliation case.
Include a replay of the same input range.
Confirm that it can reconstruct the report without sending a new live booking.
Finally, simulate an external request that times out after acceptance.
The expected state is uncertain until the partner confirms the outcome, not automatically failed.

These scenarios do not replace source audits, security assessment, or real operating measurements.
They make the intended temporal and business behavior concrete enough for partners to review.
Repeat them when changing the source contract, engine version, watermark policy, or action integration.

## 6. Worked example

### Business context and commitments

A grocery distributor has shipment S-417 on a carrier route to a regional distribution center.
The customer commitment is arrival by 10:00 UTC on 14 May 2026.
The operating plan assigns the truck to dock 6 and keeps a backup dock slot available until 10:05 UTC.
The carrier's 09:30 UTC estimated time of arrival is 10:07 UTC; this is a forecast, not a commitment or observation.
The target is to use the primary dock without customer disruption, but the target does not override the committed appointment.

At 10:05 UTC, the dispatcher must decide whether to reserve the backup slot.
The reservation costs USD 85 if later canceled; an unused backup is released at 10:30 UTC.
For illustration, an unprotected missed appointment has an estimated incremental cost of USD 420.
These are decision inputs, not verified company costs.
Assume the dispatcher reserves if the estimated chance of a missed appointment exceeds the break-even probability and no stronger rule applies.

The simplified break-even calculation compares USD 85 with probability `p` times USD 420.
The threshold is `85 / 420 = 0.20238`, or approximately 20.2%.
This assumes the backup completely avoids the USD 420 loss and the reservation fee is incurred in either outcome.
If the backup only reduces the loss, its effectiveness must enter the denominator.
The operator's probability estimate is an input, not something derivable solely from the arrival messages below.

### Event and receipt ledger

All times below are UTC on the same date.
The source event ID is stable across a retry, and revisions are explicit in this example.

| Receipt time | Source event ID | Assertion | Event time | Interpretation at receipt |
|---|---|---|---|---|
| 08:10 | C-11 | Depart origin | 08:05 | Accepted departure |
| 09:31 | C-12 | ETA forecast 10:07 | 09:30 | Forecast; not arrival proof |
| 10:18 | C-13 | Arrive gate | 09:52 | Accepted late arrival evidence |
| 10:20 | C-13 | Arrive gate | 09:52 | Exact retry; no second arrival |
| 10:24 | C-14 | Gate scan correction for C-13 | 09:56 | Authorized revision; replaces 09:52 |
| 10:26 | C-10 | Load complete | 07:54 | Accepted out of order |

C-14 is a correction message with its own identity and a reference to C-13.
It changes the asserted gate arrival from 09:52 to 09:56 UTC.
The two timestamps are not two gate arrivals.
The original assertion and correction remain available for audit.
The later received load event precedes both departure and arrival in event time.

### Reconstruct the live decision

At 10:05 UTC the accepted records available to the dispatcher are C-11 and C-12.
There is no received arrival scan.
The event-time fact “arrived at 09:52” is not yet known to the dispatcher.
The operational state is **arrival unknown**, not **confirmed late**.

Suppose the dispatcher estimates a 35% chance of a missed appointment using the 10:07 UTC forecast and other available information.
Under the simplified loss assumptions, 35% exceeds the 20.2% break-even threshold.
The dispatcher reserves the backup slot at 10:05 UTC, with action ID `S-417/backup/v1`.
This is a defensible provisional decision under the assumed rule, not evidence that a miss occurred.

At 10:18 UTC C-13 reports arrival at 09:52 UTC.
Its ingestion lag is 26 minutes: 10:18 minus 09:52.
The slot is canceled at 10:19 UTC and incurs USD 85 under the case assumption.
The exception resolves as “late evidence of on-time arrival,” not “physical delay confirmed.”
The cancellation cannot erase the decision or the fee.

At 10:20 UTC the retry of C-13 has a receipt-to-event difference of 28 minutes, but it is not a second physical event.
Count it as a delivery attempt for transport monitoring, not a second accepted arrival or another customer notification.
The first receipt remains 10:18 UTC for the event's first-receipt lag.

At 10:24 UTC C-14 replaces 09:52 with 09:56 UTC.
The corrected physical assertion remains four minutes before the 10:00 UTC commitment.
For a corrected event-time view, the arrival-to-first-receipt difference becomes 22 minutes, with the correction itself received 28 minutes after the corrected event time.
These are different measures; the correction's publication delay must not be mistaken for the original message's first receipt.
The alert history retains both the originally accepted value and the later correction.

At 10:26 UTC the 07:54 load-complete event arrives.
Its ingestion lag is 152 minutes.
Its receipt after arrival does not mean the shipment was loaded after reaching the destination.
The corrected event-time sequence is load complete at 07:54, depart at 08:05, and gate arrival at 09:56.

### Reporting windows and revisions

Define hourly arrival windows as half-open UTC intervals.
The 09:52 arrival initially belongs to `[09:00, 10:00)`.
The correction to 09:56 remains in that same window, so its distinct-shipment arrival count remains one.
The retry adds zero to that count.
The 07:54 load event belongs to a different milestone and must not increase the arrival count.

If the authorized correction instead changed arrival to 10:03 UTC, the corrected count in `[09:00, 10:00)` would decrease by one and the count in `[10:00, 11:00)` would increase by one.
The corrected on-time classification would change from on time to late by three minutes.
That hypothetical correction is separate from the actual 09:56 case.
Both affected windows and the performance record need a versioned adjustment.

Suppose the tower published an hourly count at 10:05 UTC before C-13 was received.
For this one-shipment cohort, the provisional count was zero, and the expected count was one.
At 10:18 UTC the revised count is one.
At 10:24 UTC the count remains one, while the arrival timestamp is revised.
A report labeled only “final count 1” cannot reconstruct why the dispatcher saw zero at 10:05 UTC.

### Watermark illustration

Assume a simplified watermark of the largest valid observed event time minus 15 minutes.
At some later processing instant, suppose that maximum is 10:20 UTC; the watermark is 10:05 UTC.
The watermark has passed the 10:00 UTC window end, so the configured policy may emit an on-time result.
A further 30-minute allowed-lateness horizon would end when that watermark passes 10:30 UTC, using the explicit rule in this lesson.
An accepted arrival at 09:56 UTC can revise the window until that horizon is passed.
These illustrative maxima are not claims about what the six-row shipment ledger alone would produce.
A real multi-partition engine may use a different aggregate watermark and trigger behavior.

### Decision review

The physical outcome under the latest accepted correction is on-time arrival.
The decision outcome is an unnecessary USD 85 reservation fee in hindsight.
The service outcome is a canceled backup, not an observed customer delay.
The process finding is that an arrival scan useful by 10:05 UTC was first received at 10:18 UTC.
A retrospective must assess the decision using the information and policy available at 10:05 UTC, not only the corrected 10:24 UTC history.
It should also investigate why the carrier upload was delayed before attributing the fee to any party.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It demonstrates stable-ID deduplication, an explicitly authorized correction, as-known snapshots, and event-time sorting.
It is a deterministic batch illustration, not a streaming engine, watermark implementation, production identity service, or external-action dispatcher.
It assumes UTC timestamps, a single shipment, unique correction IDs, and trusted correction authorization supplied in the records.

```python
from datetime import datetime, timezone


def instant(hm):
    return datetime.strptime(
        "2026-05-14 " + hm, "%Y-%m-%d %H:%M"
    ).replace(tzinfo=timezone.utc)


messages = [
    {"received": "08:10", "id": "C-11", "kind": "event",
     "step": "depart", "occurred": "08:05"},
    {"received": "09:31", "id": "C-12", "kind": "event",
     "step": "forecast", "occurred": "09:30"},
    {"received": "10:18", "id": "C-13", "kind": "event",
     "step": "arrival", "occurred": "09:52"},
    {"received": "10:20", "id": "C-13", "kind": "event",
     "step": "arrival", "occurred": "09:52"},
    {"received": "10:24", "id": "C-14", "kind": "correction",
     "target": "C-13", "occurred": "09:56", "authorized": True},
    {"received": "10:26", "id": "C-10", "kind": "event",
     "step": "load", "occurred": "07:54"},
]


def snapshot(cutoff):
    """Apply only messages received by cutoff; preserve first event receipt."""
    events = {}
    correction_ids = set()
    duplicates = 0
    for message in messages:
        if instant(message["received"]) > instant(cutoff):
            continue
        if message["kind"] == "event":
            event_id = message["id"]
            if event_id in events:
                old = events[event_id]
                if (old["step"], old["original_time"]) != (
                    message["step"], message["occurred"]
                ):
                    raise ValueError("conflicting event identity")
                duplicates += 1
                continue
            events[event_id] = {
                "step": message["step"],
                "original_time": message["occurred"],
                "current_time": message["occurred"],
                "first_received": message["received"],
            }
        else:
            if message["id"] in correction_ids:
                raise ValueError("duplicate correction ID")
            if not message["authorized"] or message["target"] not in events:
                raise ValueError("untrusted or missing correction target")
            correction_ids.add(message["id"])
            events[message["target"]]["current_time"] = message["occurred"]
    return events, duplicates


def arrival_summary(cutoff):
    events, duplicates = snapshot(cutoff)
    arrivals = [event for event in events.values()
                if event["step"] == "arrival"]
    if not arrivals:
        return "unknown", duplicates
    return arrivals[0]["current_time"], duplicates


for cutoff in ("10:05", "10:18", "10:20", "10:24"):
    arrival, duplicates = arrival_summary(cutoff)
    print(f"{cutoff}: arrival={arrival}; retries={duplicates}")

final, _ = snapshot("10:26")
physical = [(event["current_time"], event["step"])
            for event in final.values() if event["step"] != "forecast"]
for event_time, step in sorted(physical):
    print(f"{event_time} {step}")

arrival = final["C-13"]
lag = instant(arrival["first_received"]) - instant(arrival["current_time"])
print(f"corrected arrival to first receipt: {int(lag.total_seconds() / 60)} min")
```

Expected output:

```text
10:05: arrival=unknown; retries=0
10:18: arrival=09:52; retries=0
10:20: arrival=09:52; retries=1
10:24: arrival=09:56; retries=1
07:54 load
08:05 depart
09:56 arrival
corrected arrival to first receipt: 22 min
```

The example assumes `messages` are in receipt order and that the initial event precedes its correction in that order.
A production implementation must handle concurrent arrivals, persistence, source-scoped keys, correction chains, authorization, invalid records, and potentially multiple arrivals.
It must also retain each original message and action ledger; this compact snapshot function does not implement durable audit storage.
The code deliberately does not dispatch a backup or calculate a watermark.
Those require business permissions and engine-specific behavior rather than a hidden side effect in a teaching example.

## 8. Data, standards, and regulatory context

### A practical data contract

The parties should document the meaning and expected availability of each timestamp.
For each event type, specify its producer, event ID scope, timestamp source, timezone, precision, and expected delivery channel.
Define how to identify a retry and how an authorized correction references the earlier event.
Specify whether receipt time is recorded before or after validation and whether it is durable.
A contract should name the owner who resolves conflicting milestone claims.

The contract also needs service objectives.
For example, “95% of eligible gate-arrival events received within 10 minutes of occurrence” requires a definition of eligible events, measurement period, first receipt, and exclusions.
A percentile based only on events received can hide missing events.
Pair it with coverage of the expected event population and a separate measure of events that arrive after the decision deadline.
State whether corrections are measured separately from original event deliveries.

A correction policy must identify valid issuers, reason codes, version order, and retention of superseded assertions.
A data contract can establish these requirements between parties, but it cannot prove the source clock or physical observation was accurate.
Clock synchronization, site procedures, sample audits, and dispute resolution remain necessary.

### EPCIS and event semantics

GS1 Electronic Product Code Information Services (EPCIS) 2.0 provides a standard model and interfaces for sharing visibility event data.
Its event model distinguishes event time, event-time timezone offset, and record time.
Consult the normative specification for precise field definitions and repository behavior.
The standard provides error declaration and corrective-event mechanisms; an implementation must follow the applicable specification rather than assume that an arbitrary overwrite is a compliant correction.

An EPCIS event can improve interoperability when parties agree on identifiers, business steps, locations, and timing semantics.
It does not guarantee that a scanner was online, that a source clock was correct, or that the event was available to a dispatcher before a deadline.
Nor does adopting EPCIS automatically choose a watermark, retention horizon, or financial-close policy.
Those are application and governance decisions.

### Time representations

Use a timestamp with an explicit offset, or an unambiguous UTC instant, for inter-system comparison.
Retain the source's original text and timezone information where needed to investigate interpretation errors.
A local wall-clock time without an offset can be ambiguous during a daylight-saving transition.
A date without a time is insufficient to infer a minute-level arrival sequence.

Internet Engineering Task Force RFC 3339 defines a profile for timestamps representing instants in Internet protocols.
It is a technical specification, not a guarantee of device-clock accuracy or a universal supply-chain law.
A contract should also state precision and any accepted clock-skew tolerance.

### Privacy, access, and retention

Location and driver-linked data may be personal data depending on the facts and applicable law.
For example, the European Union General Data Protection Regulation (GDPR) contains requirements concerning lawful processing, data minimization, security, and data-subject rights when it applies.
Applicability depends on jurisdiction, controller or processor role, and actual processing; this lesson is not legal advice.
Legal retention obligations, deletion duties, litigation holds, and contractual audit rights can conflict and require counsel's assessment.

Protect raw and corrected events with role-based access and a documented retention schedule.
Keep audit lineage without granting every operational user access to sensitive identity or location detail.
Separate a data-quality correction from a request to exercise a legal right; both need controlled workflows.
A technically append-only design still needs an authorized approach to applicable deletion or restriction requirements.

### Regulatory boundary

A carrier service-level objective or internal watermark is not itself a statutory rule.
Product-specific traceability, transport, safety, and recordkeeping obligations can impose additional requirements on particular goods and jurisdictions.
Consult the applicable current rule and counsel rather than deriving a retention period or legal deadline from this example.
As of the course version, no universal regulatory allowed-lateness interval is asserted here.

## 9. Industry practice and case evidence

### Streaming systems make the trade-off explicit

The Apache Beam programming guide describes event time, watermarks, windows, triggers, and late data.
Its model is useful because it separates when an event occurred from when a runner processes it.
The guide also illustrates that watermarks are estimates and that triggers can produce early or late results.
This is official project documentation about a programming model, not an independent measurement of a grocery carrier's performance.
Actual semantics must be checked for the chosen runner and version.

Google Cloud Dataflow documentation discusses streaming watermarks and late data in its managed implementation.
It provides evidence that production streaming systems expose these concepts to operators.
It does not establish the right lateness threshold for any given supply chain.
Differences in input sources, idle partitions, and triggers can change the observed behavior.

### Visibility standards record multiple times

The GS1 EPCIS 2.0 standard defines visibility-event information and distinguishes times associated with an event and its recording.
This is evidence of an interoperable data model, not evidence that a particular carrier's milestone is accurate.
The distinction matters to a tower that must reconstruct what happened separately from when a repository learned it.
A correction mechanism supports traceability only when partners actually retain and exchange the relevant records.

### Public company illustration

Walmart and IBM have publicly described blockchain-based food traceability work using shared product information and a mango-origin tracing demonstration.
The reported reduction in tracing time is a company and vendor demonstration claim, not evidence that the event-time or watermark policy in this lesson was used.
The case illustrates the business value of retrieving lineage quickly; it does not prove that all submitted events are timely, correct, or deduplicated.
[INFERENCE] A tower applying this lesson to such a traceability network would still need to distinguish physical event time from message availability and to govern corrections.
That inference is an architectural lesson, not a claim about Walmart's internal implementation.

### What a practitioner should measure

A carrier can meet an average transmission-lag target while a critical lane experiences late arrivals near its decision deadline.
Measure the distribution by milestone and decision window, not only the overall mean.
Check whether a partner outage creates a cluster of delayed uploads that changes multiple historical reports at once.
Review whether a correction rate reflects honest discovery of errors, a faulty source clock, or an identity mismatch.

A technology vendor's demonstration, a standards document, and a company's performance claim answer different questions.
None substitutes for an operator's measured decision lead time, eligible event population, and realized intervention costs.
When comparing solutions, ask for timestamp definitions, replay behavior, external side-effect boundaries, and an audit of revision handling.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Receipt treated as occurrence | Arrival times cluster at upload batches | Transport time replaces source event time | False late-delivery claims | Preserve and label separate clocks |
| Latest message wins | Delivered shipment returns to transit | Late departure overwrites later physical state | Wrong intervention | Recompute by event time and transition rules |
| Duplicate delivery counted | Same event ID appears repeatedly | Retry applied as new milestone | Inflated throughput and alerts | Stable source-scoped identity and idempotent effects |
| Correction discarded as retry | Same event ID has changed payload | Identity check ignores revision semantics | Known error persists | Explicit correction path and conflict queue |
| Distinct scans merged | Same location and minute for two objects | Overbroad fingerprint | Missing traceability evidence | Include object and producer identity; resolve uncertain matches |
| Future clock advances watermark | Sudden jump in event time | Bad source clock drives progress | Valid older events treated as too late | Validate skew before watermark input |
| Idle partition stalls progress | No window closes for a quiet lane | Watermark depends on silent input | Delayed reporting | Explicit idle policy and late-data recovery |
| Late record silently dropped | Corrections vanish beyond retention | Technical cutoff mistaken for truth | Unreconciled performance | Preserve raw records and route post-cutoff adjustments |
| Dashboard hides revision | Count changes without explanation | Mutable result lacks lineage | Disputes and lost trust | Version results and expose revision reason |
| Replay sends actions again | Old messages produce new bookings | Side effects not separated from recomputation | Duplicate cost or dispatch | Disable live actions during replay; use action ledger |
| Missing counted as physical delay | Alert says late with no arrival evidence | Unknown mapped to failure | Unnecessary escalation | Separate evidence status from physical status |
| Local time interpreted as UTC | Events shift by hours | Missing offset or daylight-saving ambiguity | Wrong window and SLA | Preserve offset; quarantine ambiguity |
| Lag metric hides missing events | Fast p95 with falling coverage | Percentile uses received events only | False service confidence | Pair lag with expected-event coverage |
| Correction crosses a window | One window changes, another does not | Old contribution not retracted | Double count or undercount | Recompute both affected windows |
| Lost acknowledgment duplicates booking | Retry made after uncertain response | External operation lacks idempotency | Two backup slots reserved | Query by action key and reconcile |
| Unauthorized correction accepted | Unexpected partner changes timestamps | Weak source authorization | Manipulated performance history | Authenticate, authorize, and audit corrections |
| Ambiguous tie treated as fact | Same-minute milestones given precise order | Precision ignored | False causal explanation | Preserve uncertainty and source sequence |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Event time approximates occurrence | Supports physical ordering | Device clock error or manual backdating | Compare clock telemetry and independent evidence |
| Source event ID is stable | Enables retry recognition | Producer generates a new ID per retry | Audit retry traces with producer |
| IDs are unique within source | Prevents unrelated events merging | ID reuse or counter reset | Monitor same-ID payload conflicts |
| Correction is authorized | Prevents arbitrary rewriting | Compromised or mistaken source | Verify identity, permissions, and reason |
| Shipment association is correct | Attributes event to decision | Mis-scanned label or reused ID | Reconcile shipment, object, location, and route |
| Expected population is known | Gives coverage a denominator | Unreported cancellation or new route | Reconcile with current plan and shipment master |
| Decision loss estimates are usable | Supports threshold calculation | Costs or backup effectiveness differ | Validate against actual invoices and outcomes |
| Backup is available | Makes reservation a real option | Capacity already allocated | Confirm reservation acknowledgment |
| UTC conversion is unambiguous | Enables comparisons | Missing offset or repeated local hour | Preserve original value and resolve timezone |
| Revision lineage is retained | Supports as-known reconstruction | Overwrite or retention deletion | Audit sample records and retention policy |
| External partner honors action key | Prevents repeated business effect | Partner ignores or expires key | Contract test and reconcile acknowledgments |
| Watermark configuration matches purpose | Makes output timing interpretable | Idle or future-dated input | Inspect watermark traces and engine docs |
| Reported event represents the right milestone | Makes appointment comparison valid | Gate arrival differs from unload completion | Agree on milestone definition in contract |
| Historical replay is isolated | Prevents new operational actions | Live endpoints remain enabled | Review replay controls and action ledger |

No lateness policy can make a missing event available before it was sent.
No timestamp alone establishes that the observed object was the correct pallet or that a claimed activity occurred.
A high-quality event stream reduces uncertainty but does not eliminate business judgment.

## 12. Exercises

### Exercise 1 — Separate the clocks

A driver reaches a gate at 08:42 UTC, the scanner records it at 08:44 UTC, the source publishes at 09:02 UTC, and the tower receives it at 09:05 UTC.
The application evaluates the message at 09:06 UTC.

- Deliverable: a four-column timeline identifying occurrence, capture, receipt, and processing, plus publication.
- Acceptance: identify each timestamp correctly and calculate first-receipt lag in minutes.
- Time: 8 minutes.

### Exercise 2 — Reconstruct knowledge

Use the S-417 ledger in Section 6.
At 10:05 UTC the dispatcher sees the departure and ETA forecast; the arrival message is first received at 10:18 UTC.

- Deliverable: one sentence describing arrival status at 10:05 UTC and another describing the latest corrected arrival at 10:24 UTC.
- Acceptance: distinguish unknown at the first time from the 09:56 UTC corrected assertion at the second; do not call the forecast an arrival.
- Time: 8 minutes.

### Exercise 3 — Calculate lag and order

Use the S-417 ledger.
Compare the first arrival assertion and the load-complete event.

- Deliverable: calculate their first-receipt lags and state the corrected physical milestone order.
- Acceptance: show the arithmetic in minutes, identify the 09:56 UTC correction, and distinguish receipt order from event-time order.
- Time: 12 minutes.

### Exercise 4 — Move a contribution between windows

A distinct-shipment arrival initially asserted at 09:58 UTC is corrected to 10:03 UTC.
The two hourly arrival windows are `[09:00, 10:00)` and `[10:00, 11:00)` UTC.
Before correction, their counts are 12 and 7 shipments respectively, including this shipment once.

- Deliverable: give the revised counts and the net change in the combined two-window total.
- Acceptance: retract from the first window and add to the second; preserve one total shipment.
- Time: 10 minutes.

### Exercise 5 — Choose a policy

A carrier's gate scan arrives within 5 minutes for 80% of eligible events and within 25 minutes for 98%.
A backup decision is due 10 minutes after the scheduled appointment.
An audit report is issued the next day.
Assume the two percentages are measured over an adequately defined eligible population and no missing events are hidden.

- Deliverable: propose a live decision policy and a separate report revision policy, each with an owner and cutoff.
- Acceptance: identify the speed/completeness trade-off, name a post-cutoff correction path, and avoid claiming that the 98% figure guarantees every event arrives.
- Time: 18 minutes.

### Exercise 6 — Classify four messages

A source sends event `(carrier-A, E7)` asserting gate arrival at 11:02 UTC.
Later it sends the identical assertion with the same ID, another message with the same ID but arrival 11:09 UTC and no revision reference, and a separate authorized correction `E8` referring to E7 and changing it to 11:09 UTC.
Finally it sends `(carrier-B, E7)` for a different shipment.

- Deliverable: classify the later four messages and state the action for each.
- Acceptance: distinguish retry, unresolved conflict, authorized correction, and separately scoped event; preserve lineage.
- Time: 12 minutes.

### Exercise 7 — Design the exception lifecycle

A tower reserves backup trucks through a partner API that can time out after accepting a request.
A late arrival scan can invalidate the underlying exception.
The partner cannot currently guarantee idempotency.

- Deliverable: draw or list states, evidence, action IDs, timeout handling, cancellation, and audit fields.
- Acceptance: prevent blind retries after an ambiguous response, preserve as-known decisions, distinguish reversible and irreversible effects, and give an owner for reconciliation.
- Time: 20 minutes.

### Exercise 8 — Audit the metric

A report says 99% of arrival messages arrived within 10 minutes.
It was calculated from 990 received messages; the plan listed 1,000 eligible expected arrival events.
There are no documented ineligible cases.

- Deliverable: calculate observed coverage and explain what the 99% does and does not measure.
- Acceptance: give 99% coverage, distinguish coverage from conditional timeliness, and propose at least two additional controls.
- Time: 12 minutes.

## 13. Self-check answers

### Exercise 1

The occurrence is 08:42 UTC, capture is 08:44 UTC, publication is 09:02 UTC, receipt is 09:05 UTC, and processing is 09:06 UTC.
First-receipt lag is 23 minutes, calculated as 09:05 minus 08:42.
The two-minute occurrence-to-capture gap and three-minute publication-to-receipt gap describe different segments.
The one-minute receipt-to-processing gap is internal processing lag, not total ingestion lag.

### Exercise 2

At 10:05 UTC, arrival is unknown because the tower has not received an arrival assertion.
At 10:24 UTC, the latest authorized assertion is gate arrival at 09:56 UTC, replacing the earlier 09:52 UTC assertion.
The 10:07 UTC ETA is a forecast, not an actual arrival.
The earlier decision remains auditable against the evidence available when it was made.

### Exercise 3

The initial arrival's first-receipt lag is 26 minutes: 10:18 minus 09:52.
The load-complete lag is 152 minutes: 10:26 minus 07:54.
After correction, the physical sequence asserted by accepted events is load at 07:54, departure at 08:05, and arrival at 09:56 UTC.
The corrected arrival to the original first receipt is 22 minutes; the correction itself was received at 10:24 UTC.
The load message's late receipt does not place loading after arrival in physical time.

### Exercise 4

The corrected counts are 11 shipments in `[09:00, 10:00)` and 8 shipments in `[10:00, 11:00)`.
The combined count remains 19 shipments; its net change is zero.
The movement requires a retraction and an addition, not a second arrival.
The report should version both windows and identify the correction.

### Exercise 5 rubric

A strong answer acts by the 10-minute decision deadline using current evidence and a documented cost-based escalation rule.
It does not wait 25 minutes merely to improve a reporting percentile.
It publishes the next-day report with a named revision cutoff and a controlled post-close adjustment process.
It assigns operations to a dispatcher and report integrity to a data or finance owner.
It notes that the reported percentiles do not guarantee complete or correct data.
The exact cutoff is a justified business choice, not a uniquely determined numerical answer.

### Exercise 6

The identical same-source same-ID assertion is an exact retry and must not create another arrival.
The same-ID changed payload without a correction relation is a conflict requiring investigation, not an automatic overwrite or discarded duplicate.
The authorized E8 correction supersedes E7 according to the source contract while retaining both versions.
Carrier B's E7 is a separately scoped event because the source is part of the key; it still needs normal validation and shipment association.

### Exercise 7 rubric

A strong design includes at least open, action requested, outcome uncertain, confirmed reserved, cancel requested, resolved, and reconciliation states.
It records a stable action key, evidence version, requester, request and response times, partner reference, and cancellation result.
On timeout it queries or reconciles before a repeat because acceptance is unknown.
It treats a late scan as a reason to reassess and possibly cancel, not as proof that the original decision was negligent.
If a physical dispatch cannot be reversed, it records a compensating action or cost and assigns an accountable owner.

### Exercise 8

Observed coverage is `990 / 1,000 = 99%` of eligible expected events.
The stated 99% timeliness is conditional on the 990 received messages, not on all 1,000 expected events.
Without knowing the ten missing events' eventual arrival times, the all-expected-event timeliness cannot be asserted as 99%.
Useful controls include missing-event aging, first-receipt lag by source and milestone, clock-skew checks, and correction rates.
Also investigate whether the 990 messages represent 990 distinct logical events rather than retries.

## 14. Interview questions

1. **Why can a dashboard show a late arrival even if the truck arrived on time?**
   A strong answer distinguishes occurrence time from receipt and explains delayed evidence.
   A weak answer assumes a recent message proves a recent physical event.
2. **What does a watermark tell you?**
   A strong answer calls it an estimate of event-time progress under a particular engine and policy.
   A weak answer calls it proof that all earlier events exist in the system.
3. **How do you set an allowed-lateness horizon?**
   A strong answer connects lag distribution, decision deadlines, revision cost, and post-cutoff reconciliation.
   A weak answer copies a vendor default without a business consequence.
4. **What is the difference between a duplicate and a correction?**
   A strong answer uses source-scoped identity, unchanged retry payload, explicit supersession, and authorization.
   A weak answer drops every repeated event ID without inspecting a changed assertion.
5. **How do you preserve the fairness of a retrospective?**
   A strong answer reconstructs the information available at decision time alongside the corrected physical history.
   A weak answer judges a past operator using evidence that arrived later.
6. **What happens when a corrected timestamp crosses a window boundary?**
   A strong answer retracts the old contribution, adds the new one, and versions both results.
   A weak answer only increments the new window.
7. **How do you handle an API timeout after requesting a backup truck?**
   A strong answer treats the outcome as uncertain, queries by action key, and reconciles before a blind retry.
   A weak answer repeatedly sends the request and assumes the first one failed.
8. **Can exactly-once messaging guarantee one physical dispatch?**
   A strong answer scopes guarantees and explains idempotency, external acknowledgments, and compensation.
   A weak answer treats a broker configuration as a guarantee about a carrier's real-world action.
9. **Why is a 99th-percentile lag insufficient by itself?**
   A strong answer asks about missing-event coverage, sample denominator, milestone, source, and decision deadline.
   A weak answer assumes unreceived events are included in the distribution.
10. **What should happen to an event received after reporting close?**
    A strong answer retains evidence and routes it to an authorized adjustment process.
    A weak answer silently deletes it because the window is closed.

## 15. Key terms

- [Allowed lateness](#4-definitions-and-vocabulary) — configured tolerance for late revisions under a specified engine policy.
- [As-known view](#3-the-core-mental-model) — information available at a particular decision time.
- [Capture time](#4-definitions-and-vocabulary) — source recording time.
- [Clock skew](#5-the-mechanics-step-by-step) — difference between a source clock and a reference.
- [Corrected view](#6-worked-example) — latest accepted interpretation of earlier occurrences.
- [Correction](#5-the-mechanics-step-by-step) — authorized replacement or adjustment of an assertion.
- [Decision deadline](#5-the-mechanics-step-by-step) — latest time to choose an intervention.
- [Deduplication](#5-the-mechanics-step-by-step) — recognizing repeated delivery of one logical event.
- [Event identity](#4-definitions-and-vocabulary) — stable source-scoped logical event key.
- [Event time](#3-the-core-mental-model) — asserted occurrence time.
- [Exception](#5-the-mechanics-step-by-step) — condition that requires attention under a business rule.
- [Forecast](#6-worked-example) — estimate of a future outcome, not a commitment.
- [Half-open window](#5-the-mechanics-step-by-step) — interval including its start and excluding its end.
- [Idempotency](#5-the-mechanics-step-by-step) — repeatable operation with the same intended effect.
- [Ingestion lag](#4-definitions-and-vocabulary) — first receipt minus occurrence time.
- [Late event](#4-definitions-and-vocabulary) — event arriving after a named temporal reference.
- [Out-of-order event](#6-worked-example) — event received in a different order from occurrence.
- [Processing time](#3-the-core-mental-model) — time at which the application computes.
- [Provisional result](#5-the-mechanics-step-by-step) — result subject to an explicit revision policy.
- [Receipt time](#4-definitions-and-vocabulary) — time the tower durably receives a message.
- [Replay](#5-the-mechanics-step-by-step) — reprocessing retained historical messages.
- [Retraction](#5-the-mechanics-step-by-step) — removal of a prior contribution or assertion.
- [Revision cutoff](#3-the-core-mental-model) — boundary for routine automatic updates.
- [Supersession](#5-the-mechanics-step-by-step) — explicit relationship to a replaced assertion.
- [Trigger](#4-definitions-and-vocabulary) — rule that emits a window result or revision.
- [Watermark](#5-the-mechanics-step-by-step) — estimate of progress through event time.
- [Window](#6-worked-example) — time-bounded grouping with explicit endpoint rules.

## 16. Sources

1. GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis), including EPCIS 2.0 specification and event model. Standard and supporting documentation; consult the applicable version for normative fields and correction semantics.
2. Apache Beam, [Programming Guide](https://beam.apache.org/documentation/programming-guide/), especially watermarks, late data, triggers, and windowing. Official project documentation; runner-specific behavior can differ.
3. Google Cloud, [Streaming pipelines](https://cloud.google.com/dataflow/docs/concepts/streaming-pipelines). Official Dataflow documentation on event time, watermarks, and late data; product-specific rather than a universal standard.
4. Klyne and Newman, [RFC 3339: Date and Time on the Internet: Timestamps](https://www.rfc-editor.org/rfc/rfc3339), Internet Engineering Task Force, 2002. Technical timestamp profile, not a clock-accuracy guarantee.
5. European Parliament and Council, [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), General Data Protection Regulation. Legal text; applicability and current obligations require jurisdiction-specific assessment.
6. Walmart, [A New Era of Food Transparency Powered by Blockchain](https://corporate.walmart.com/news/2018/09/24/a-new-era-of-food-transparency-powered-by-blockchain), 24 September 2018. Company account of food traceability; not independent evidence of the implementation in this lesson.
7. Tyler Akidau et al., [The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf), *Proceedings of the VLDB Endowment*, 2015. Research discussion of the correctness, latency, and cost trade-off.

## 17. Links

- [Module 19 index](_index.md)
- [Topic 01: Visibility layers](01_visibility-layers.md)
- [Topic 02: Event streams and EPCIS](02_event-streams-epcis.md)
- [Module 19 survey and its lab sections](../../../course/19_control_towers_visibility_events.md)

The survey's lab sections provide the relevant control-tower exercise context.
Sibling topics listed as planned in the module index may not yet be available.
