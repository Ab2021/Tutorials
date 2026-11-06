# Streaming Events: Exceptions, Event Time, and Replay

> Module 30 — Simulation, Optimization, and Real-Time Production Labs
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Python 3.9 or later, basic SQL transactions and timestamps, and the [environment and repository lab](01_environment-repo.md).
> Version 2026.5

## 1. Why this topic exists

A temperature-controlled distribution network cannot wait for a monthly report to discover that a shipment was exposed to heat.
A transport control tower needs to know which shipment needs investigation, when the reading was taken, and whether the signal is still actionable.
The analyst must also know whether a repeated notification is a second excursion or a repeated delivery of the same reading.

The operational decision is deliberately narrow: should an operator investigate a shipment temperature reading above a configured threshold?
A sensor observation is not itself proof that product is unsafe.
The carrier, quality team, and product owner determine whether a reading is valid and what disposition is appropriate.
The lab's threshold of 8 °C is a synthetic rule, not a universal cold-chain standard or a legal limit.

A producer creates a durable record when a sensor reading enters the platform.
A consumer validates the record, distinguishes first delivery from retry, and records an exception candidate.
The consumer records its progress with the state change so a restart does not silently skip work.
A separate alert delivery process would contact an operator; the local lab stops at a durable alert intent.
This separation matters because writing a row and sending a message are different side effects.

The clock at the sensor and the clock at the consumer answer different questions.
A reading taken at 10:01 but delivered at 10:23 describes conditions at 10:01.
A current alert based on that observation needs its age and a review policy.
Late data must not be quietly treated as an on-time current condition.
Nor should it vanish merely because the operational window has moved on.

The dispatcher owns the response workflow.
The quality team owns interpretation of exposure and product disposition.
The event platform team owns transport, state, observability, and replay safety.
The producer team owns event identity and the meaning of the sensor payload.
The customer service team may need to communicate an impact once authorized.
A runbook must name real on-call roles and contacts before a production rollout.

This lab distinguishes four business artifacts.
A forecast estimates future conditions or volumes; this lab does not produce a forecast.
A plan specifies how operators will respond to an alert; the lab proposes no live dispatch plan.
A target expresses desired processing speed or service; example thresholds here are illustrative, not measured service levels.
A commitment is an agreed obligation to a customer or regulator; a synthetic alert creates no such commitment.

The exercise uses fabricated readings, a single-process Python consumer, and a local SQLite database.
The producer fixture is an ordered in-memory list, not a broker, live sensor, or fault-tolerant message system.
No cloud account, credentials, container, external package, or network access is required.
The result demonstrates rules and transaction boundaries, not production throughput or reliability.
A later section explains how a real broker, persistent source, and delivery service would change the design.

## 2. Learning outcomes

After completing the lab, you can:

1. Distinguish event time, ingestion time, processing time, and a consumer's progress position.
   Exercise 1 asks you to classify each clock and explain a retry.
2. Compute a bounded-out-of-orderness watermark and classify an observation against the previous watermark.
   Exercise 2 asks for each intermediate value, including the equality boundary.
3. Identify a logical duplicate without confusing event identity with shipment identity.
   Exercise 3 asks you to compare a retry with two distinct readings for the same shipment.
4. Explain the atomic relationship between input progress, state, and an alert intent.
   Exercise 4 asks you to place a crash at each transaction boundary.
5. Run a producer fixture and restartable consumer, then inspect the late-data and exception outputs.
   Exercise 5 asks for the exact local counts and reasons.
6. Design an operational replay that avoids repeat customer-facing effects.
   Exercise 6 asks for a shadow replay, reconciliation, and promotion decision.
7. Evaluate lag, freshness, ownership, and event quality without claiming delivery guarantees the system does not provide.
   Exercises 7 and 8 ask for operational metrics and a design critique.
8. Apply data minimization and identify standards and jurisdictional questions that need review.
   Exercise 9 asks you to construct a data and controls checklist.

Success means explaining the result as well as running the artifact.
A successful local run is evidence that the stated fixture satisfies its assertions on that machine.
It is not evidence that a broker, sensor network, notification channel, or real carrier meets a service level.

## 3. The core mental model

```text
 sensor measurement
       |
       v
 producer: assign stable event ID; retain event time and source identity
       |
       v
 ordered source log / partition (fixture: ordered Python list)
       |
       v
 consumer reads next source position
       |
       +--> schema or identity conflict --> stop / quarantine under policy
       |
       v
 transaction: check event ID and previous event-time watermark
       |
       +--> exact duplicate --> count occurrence, advance source progress
       |
       +--> too late --> retain as late for review, advance source progress
       |
       +--> accepted --> update state; create alert intent if rule matches
       |
       v
 commit progress and state together
       |
       +--> alert outbox --> separate controlled delivery --> operator acknowledgement
       |
       +--> late-data review / historical correction
       |
       +--> isolated replay and reconciliation, not automatic re-notification
```

The source log and the consumer state are separate responsibilities.
A real source must retain records independently of the consumer database long enough for the agreed recovery period.
The fixture is deterministic and ordered, but it is not durable outside the Python process.
The database records the processed position and the consumer's derived state together.
The two are never committed separately in the example.

A source position says how far a particular consumer has progressed through a particular source ordering.
An event ID says which business observation a record represents.
These are not interchangeable.
Two source positions can contain the same event ID after a producer retry.
Two readings for one shipment can have different event IDs and both be legitimate.
The lab assumes the producer never reuses an event ID for different content.
If this invariant is violated, it stops rather than silently choosing a version.

The consumer tracks the greatest accepted event time so far.
Its watermark is that maximum minus the allowed disorder interval.
The previous watermark classifies the next unseen observation.
An observation strictly earlier than that watermark is late; an observation equal to it is accepted.
The fixture uses ten minutes of allowed disorder and expresses event time in integer minutes from a declared origin.
The watermark is a deterministic policy signal, not a claim that no older reading can arrive.

On-time observations update the maximum event time.
A duplicate does not update it again.
A late observation is retained with a late status, but this lab does not raise its operational exception automatically.
The separate late review is important: a high late temperature may still matter for quality even when a live alert is stale.
A real policy could choose to notify a quality queue instead of suppressing all operational notification.

The consumer's alert intent is keyed by the event ID and rule version.
A unique key prevents this consumer database from creating the same intent twice.
It does not prove an external notification was sent once.
An outbox publisher can retry after a crash, so the receiver needs its own idempotency key or reconciliation process.
A human acknowledgement, if required, is another recorded event with its own identity and lifecycle.

A local restart resumes at the stored source position.
A replay creates a fresh isolated database and applies the same source records and rule version.
Its results are compared with the original database without sending any messages.
Replaying into the live database is not the same operation as rebuilding in isolation.
For external effects, replay needs explicit authorization and a deduplicated delivery plan.

## 4. Definitions and vocabulary

The following definitions apply to this lab; vendor products may use narrower or different terminology.

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Event | A recorded fact that a reading occurred; one logical observation | Treating a retry as a new observation |
| Producer | Component that creates or publishes a record | Assuming publishing implies an operator saw it |
| Consumer | Component that reads and processes records | Assuming a read alone is a committed result |
| Record | One delivery in source order | Equating every record with a distinct event |
| Event ID | Stable unique identity of one logical reading | Using shipment ID, which groups many readings |
| Shipment ID | Identifier of a shipment, not of an individual reading | Deduplicating all readings for a shipment |
| Event time | Time the sensor says the observation occurred; minutes here | Assuming it equals arrival time |
| Ingestion time | Time a source accepts a record; UTC instant | Treating it as the physical measurement time |
| Processing time | Time the consumer evaluates a record; UTC instant | Using it as a substitute for event time |
| Source position | Position of a record in a specific source ordering | Treating a position as a global business ID |
| Partition | Ordered subset of a broker stream | Assuming order between independent partitions |
| Offset | Position or progress marker in one partition | Assuming offset numbers are globally unique |
| Checkpoint | Durable record of progress and associated state | Advancing it before durable output exists |
| Event-time maximum | Highest accepted event timestamp seen by this consumer | Treating it as the wall clock |
| Watermark | Policy estimate of event-time completeness | Claiming it prevents future late arrivals |
| Allowed disorder | Lag between the maximum observed event time and watermark; minutes | Confusing it with source retention duration |
| Late event | Previously unseen event older than the previous watermark | Labeling every out-of-order event late |
| Boundary | Comparison at the watermark; this lab accepts equality | Leaving `<` versus `<=` unspecified |
| Duplicate | Same event ID and same event content delivered again | Treating two distinct readings as duplicates |
| Identity conflict | Same event ID paired with different content | Silently overwriting an earlier observation |
| Idempotency | Repeating an operation has the same intended effect | Assuming an idempotent database write makes all external actions exactly once |
| At-least-once | Delivery may repeat but is retried under stated conditions | Believing duplicates are impossible |
| At-most-once | Delivery can be lost rather than repeated | Calling it reliable without considering loss |
| Exactly-once effect | A defined side effect occurs once within a stated boundary | Assuming an end-to-end guarantee across unrelated systems |
| Exception | An observation that matches a business rule | A confirmed product safety incident |
| Alert intent | Durable record that a notification should be considered | Evidence that a person received a message |
| Outbox | Durable table of messages or intents committed with state | A complete notification delivery system |
| Rule version | Identifier for the logic that generated an alert | Reusing a version after changing rule semantics |
| Replay | Reprocessing retained source records under a declared policy | Blindly re-sending every historical alert |
| Backfill | Processing historical data to fill or revise a dataset | Assuming a backfill is an urgent live signal |
| Lag | Difference between a source head and a consumer position, often records or time | Assuming record lag is the same as event freshness |
| Freshness | Age of the latest relevant event or result, usually seconds or minutes | Measuring only consumer process uptime |
| Retention | How long a source or state store preserves data | Assuming archived records are always immediately replayable |
| Dead-letter queue | A location for records needing separate investigation | A place to discard records without ownership |
| Quarantine | Isolating suspect records pending review | A permanent resolution with no disposition |
| Consumer group | Coordinated consumers sharing partition assignments | Assuming a group guarantees ordered effects across partitions |
| Alert acknowledgement | Evidence a responsible person or system accepted responsibility | Proof the underlying quality incident is resolved |

## 5. The mechanics, step by step

### 5.1 Define the decision and the effect

1. Start with the decision: an operator should investigate a current temperature excursion.
2. Define the input grain as one physical sensor reading associated with one shipment.
3. Define the exception rule as `temperature_c > 8.0` for this synthetic scenario.
4. State that exactly 8.0 °C does not match the rule.
5. Define an alert intent as one row for one event ID and one rule version.
6. Define the person or queue that would own the intent in a real deployment.
7. Define a separate quality review for observations classified as late.
8. Document that this lab does not implement that queue or deliver a notification.
9. Identify the desired time-to-awareness from a real operational service agreement, not from the fixture.
10. Determine which actions are advisory, which require approval, and who records the final decision.

Input: an operating scenario and ownership information.
Output: a rule, a named effect, and an explicit response boundary.
Branch: if an alert is not actionable, redesign the rule or route it to an analytical report instead.

### 5.2 Specify the producer contract

1. Assign an event ID at the source and preserve it across retries.
2. Associate the reading with a shipment ID and a known sensor source.
3. Attach an event time in UTC or a precisely specified offset-aware representation.
4. Record the temperature with its unit and calibration context.
5. State whether a correction replaces an event, supersedes it, or creates a new event.
6. Include a schema version when the payload can evolve.
7. Agree on required fields, type limits, and validation rules with the consumer owner.
8. Define acceptable clock skew and what happens if a sensor clock is wrong.
9. Record source-system identity so independently generated IDs do not accidentally collide.
10. Assign a partition key that preserves order for the entity whose transitions need it.
11. Determine whether producer acknowledgements indicate local buffering, broker receipt, or durable replication.
12. Keep retries bounded and observable; do not generate a new identity for every retry.
13. Do not place unnecessary personal information into the event body.
14. Decide how long the original data can be retained and who can access it.

Input: a physical reading and the producer's identity context.
Output: one immutable logical event and possibly multiple identical delivery attempts.
Branch: a producer that cannot preserve identity needs a documented source-specific deduplication strategy.
A hash of payload alone can collapse two real identical readings, so it is not automatically a safe substitute.

### 5.3 Define ordering and source durability

1. Pick a source with an ordering contract appropriate for the workflow.
2. Record whether that order is global, per partition, or only per producer session.
3. Store source coordinates with enough namespace to locate a record later.
4. Determine how long the source retains events in normal and incident conditions.
5. Verify whether retention is time-based, size-based, compacted, or affected by administration.
6. Set the recovery objective so the oldest needed position remains available during a realistic outage.
7. Define how consumers detect a position that has expired from the source.
8. Preserve a replayable archive if broker retention is insufficient.
9. Ensure that restoring an archive preserves identity, event time, and necessary ordering information.
10. Document partition-count changes and their consequences for key placement and state.
11. Avoid assuming ordering across partitions unless an explicit merge policy defines it.
12. Treat a local list as a fixture only; it provides none of the durability above.

Input: a transport and an operational retention requirement.
Output: a source contract and a recovery horizon.
Branch: if retention expires before an incident can be recovered, restore from a verified archive or declare a gap.
Never silently restart at the latest record while claiming a complete replay.

### 5.4 Validate before changing business state

1. Parse the record according to the declared schema version.
2. Require a nonempty event ID and a valid shipment ID.
3. Require an event time that can be interpreted consistently.
4. Require a numeric temperature within a documented sensor-validity range if the contract defines one.
5. Reject nonfinite numeric values before comparing thresholds.
6. Check source authorization separately from payload shape.
7. Determine whether a malformed record should block a partition or enter a durable quarantine.
8. If quarantining, preserve the source coordinate and error category without exposing unnecessary sensitive data.
9. Advance progress only after the quarantine record is durable under the chosen transaction design.
10. Assign an owner, age limit, and disposition process to quarantined records.
11. Do not retry a deterministic schema error indefinitely without escalation.
12. Treat a conflicting event identity as a data integrity incident.

The fixture in section 7 assumes valid, trusted input and explicitly detects only conflicting identity reuse.
It does not implement generic schema validation, authorization, or a quarantine transport.
Those omissions are limitations, not implied production controls.

### 5.5 Distinguish duplicate, conflict, and new event

1. Look up the event ID in durable state.
2. If it exists and the business payload matches, classify the source record as a duplicate.
3. Advance the source position atomically with recording the duplicate occurrence.
4. Do not emit another alert intent for that duplicate.
5. If the event ID exists but any defined immutable field differs, stop and surface a conflict.
6. Preserve both source coordinates for investigation in a real system.
7. Do not overwrite the existing event to make the conflict disappear.
8. If the ID does not exist, classify the event against the previous watermark.
9. Record its final classification and payload before advancing the checkpoint.
10. Keep deduplication state for at least as long as duplicates and replays may legitimately arrive.
11. If state is expired, document the changed guarantee and alternate reconciliation mechanism.
12. Include rule version in alert identity so a deliberate rule change can be evaluated separately.

A retry of event `e2` remains a duplicate even if it arrives after the watermark advances.
It is checked before late classification because it does not represent a newly discovered observation.
A new ID for the same shipment and the same temperature is not necessarily a duplicate.

### 5.6 Apply event-time policy

1. Start with no maximum event time and no watermark.
2. For a new event, compute the previous watermark as `maximum - allowed_disorder`.
3. If there is no previous maximum, accept the first valid event.
4. Mark the event late only if its time is strictly less than the previous watermark.
5. If accepted, set the maximum to the greater of the previous maximum and the event time.
6. If late, retain the event without advancing the maximum.
7. Compare accepted event temperatures with the rule threshold.
8. For a matching accepted event, insert a durable alert intent.
9. For a late matching event, record it for late review rather than a live alert in this lab.
10. Record enough context to explain why an event was classified as it was.
11. Measure event-time age against an independently measured ingestion time in a real deployment.
12. Investigate future-dated sensor clocks because one bad high timestamp can move a simple watermark too far.
13. Define idleness and partition-level watermark merging before scaling beyond one source ordering.
14. Define revisions if historical reports must incorporate late observations.

A watermark is not a scheduler and it is not a proof of data completeness.
It represents a trade-off between waiting longer for out-of-order data and taking action sooner.
This fixture's fixed ten-minute interval is illustrative, not calibrated to a production network.

### 5.7 Commit state and progress together

1. Begin a database transaction before examining and changing event state.
2. Verify that the record being processed is the next expected source position.
3. Look up identity and classify the event inside the transaction.
4. Insert the event, occurrence classification, and any alert intent.
5. Update the maximum event time when appropriate.
6. Update the source progress position in the same transaction.
7. Commit all these writes together.
8. If an exception occurs before commit, roll back all of them.
9. On restart, read the last committed position and start with the next source record.
10. If the source redelivers a previously processed logical event at a later position, deduplicate by ID.
11. If the source redelivers the same already-committed position, seek from the persisted checkpoint instead of processing it again.
12. Monitor transaction failures and repeated restarts as operational incidents.

The lab has one writer and one source ordering.
Concurrent consumers need partition ownership, locking, fencing, and a transaction strategy suited to the selected systems.
A database transaction cannot atomically commit an independent broker offset without an appropriate integration design.
A common design stores the source coordinate in the same database transaction as the effects, then seeks from that coordinate.
A broker-managed transactional processing system has its own scoped guarantees and restrictions.
Do not combine guarantees from unrelated systems into an unqualified exactly-once claim.

### 5.8 Deliver an exception safely

1. Store the intent with the stable event ID and rule version.
2. Have a separate publisher read pending outbox rows.
3. Include a stable idempotency key in the downstream request where supported.
4. Mark delivery only after the receiver acknowledges the request.
5. Expect a crash after receiver acceptance but before the publisher marks the row delivered.
6. On retry, rely on receiver idempotency or perform a documented reconciliation.
7. Preserve delivery attempts, timestamps, and bounded error details.
8. Escalate persistent failures to a named operator.
9. Distinguish delivery acknowledgement from human acknowledgement.
10. Record ownership changes, investigation, and resolution separately.
11. Avoid exposing credentials or sensitive shipment details in broad notifications.
12. Rate-limit notification storms without dropping the audit trail.

Section 7 creates only the intent table.
It implements no publisher, receiver, message transport, retry scheduler, or human acknowledgement.
Therefore the lab cannot demonstrate notification delivery or exactly-once external effects.

### 5.9 Restart, replay, and reconcile

1. Record the source name, partition, start position, end position, and retention availability.
2. Stop or fence the affected consumer before changing a live checkpoint.
3. Preserve a snapshot of current state and the rule and schema versions used.
4. Prefer an isolated replay database with external effects disabled.
5. Reproduce the original order when validating an order-sensitive watermark policy.
6. Reuse stable event IDs and preserve duplicates when testing deduplication.
7. Compare accepted events, late events, duplicate occurrences, and alert intents.
8. Explain expected differences when changing a rule version or a lateness policy.
9. Inspect conflicts, missing positions, and retention gaps before declaring reconciliation complete.
10. Obtain authorization before promoting any corrected state or sending a new notification.
11. For approved notifications, use stable downstream idempotency keys and explicit time relevance.
12. Record the replay operator, inputs, software version, result, and approvals.
13. Resume live processing from an agreed cutover position and monitor the transition.
14. Keep the original audit trail; do not rewrite it to hide the incident.

A replay under a new rule answers a different question from reproduction of a previous result.
A replay in a different order can produce different late classifications even with identical distinct events.
A corrected sensor reading is not a retry unless the source contract defines it as the same immutable content.
The lab tests an isolated replay with the same order and same rule, not a live state promotion.

### 5.10 Monitor the whole path

1. Monitor producer publication errors and oldest buffered event age.
2. Monitor source availability, retained duration, and per-partition record lag.
3. Monitor processing failures and time since the last successful checkpoint.
4. Monitor observation freshness using both event and ingestion clocks.
5. Monitor counts of accepted, duplicate, late, invalid, and conflicting records.
6. Monitor age and count of pending alert intents.
7. Monitor delivery failure, delivery latency, and unacknowledged alerts where delivery exists.
8. Segment metrics by source or partition where that helps locate failure.
9. Avoid event IDs and shipment IDs as unrestricted metric labels because of cardinality and privacy.
10. Define thresholds using baseline data and incident response capacity.
11. Page for a breach that requires action, not for every individual duplicate.
12. Test alerts and escalation routes through controlled drills.

Input: source, consumer, outbox, and operational response telemetry.
Output: a diagnosis and an assigned action, not merely a dashboard.
Branch: if event age rises while record lag stays low, investigate the producer or source clock rather than only scaling consumers.

## 6. Worked example

### 6.1 Scenario and contract

A synthetic shipment moves through a temperature-controlled network.
Each source record has an event ID, a shipment ID, an event time, and a temperature in °C.
The threshold is strictly greater than 8.0 °C.
The allowable out-of-order interval is ten minutes.
The input ordering below is the order records appear in the local producer fixture.
The integer event-time origin is arbitrary, so minute 100 is not presented as a real timestamp.
All events refer to shipment `S-17`.

The business forecast is absent.
The operational plan is to investigate timely matching readings if an authorized delivery system exists.
The target of ten minutes is a disorder policy in the example, not a measured latency objective.
There is no customer commitment or actual shipment disposition in the fixture.

| Position | ID | Event minute | Temperature °C | Interpretation before processing |
|---:|---|---:|---:|---|
| 1 | e1 | 100 | 5.0 | First reading |
| 2 | e2 | 107 | 12.0 | New excursion |
| 3 | e3 | 101 | 9.0 | Out of order but possibly acceptable |
| 4 | e4 | 123 | 7.0 | Advances event-time maximum |
| 5 | e5 | 110 | 15.0 | New high reading delivered after a larger event time |
| 6 | e2 | 107 | 12.0 | Exact retry of position 2 |
| 7 | e6 | 124 | 10.0 | New excursion |

### 6.2 Step-by-step classification

At position 1, there is no earlier maximum and hence no previous watermark.
`e1` is accepted.
The new maximum is 100 minutes and the resulting watermark is 90 minutes.
Its temperature, 5.0 °C, does not exceed 8.0 °C.
There is no alert intent.

At position 2, the previous watermark is `100 - 10 = 90` minutes.
The event time 107 is not less than 90, so `e2` is accepted.
The maximum becomes 107 and the resulting watermark is 97 minutes.
`12.0 > 8.0`, so the consumer records an alert intent for `(e2, temp-v1)`.

At position 3, the previous watermark is `107 - 10 = 97` minutes.
The event time 101 is earlier than 107 but is not earlier than 97.
It is therefore out of order but accepted.
The maximum stays 107 and the watermark stays 97 minutes.
`9.0 > 8.0`, so `(e3, temp-v1)` is a second alert intent.

At position 4, the previous watermark is still 97 minutes.
Event time 123 is accepted and advances the maximum to 123 minutes.
The resulting watermark is `123 - 10 = 113` minutes.
`7.0` is not greater than 8.0 °C, so there is no alert intent.

At position 5, the previous watermark is 113 minutes.
Event time 110 is strictly less than 113, so the previously unseen `e5` is late.
The event and its 15.0 °C payload are retained with a late classification.
The lab creates no live alert intent for it, even though it crosses the temperature threshold.
A real quality workflow must decide how to investigate this late high reading.
The maximum remains 123 minutes and the watermark remains 113 minutes.

At position 6, event ID `e2` is already present with exactly the same payload.
The record is a duplicate, not a new late event, even though event minute 107 is below the current watermark.
The consumer records the duplicate occurrence and advances the position.
It does not insert a second `(e2, temp-v1)` intent.
The maximum and watermark do not change.

At position 7, the previous watermark is 113 minutes.
Event time 124 is accepted and advances the maximum to 124 minutes.
The resulting watermark is 114 minutes.
`10.0 > 8.0`, so `(e6, temp-v1)` is the third alert intent.

### 6.3 Reconciliation table

| Position | Previous watermark, min | Classification | Maximum after, min | Alert intent |
|---:|---:|---|---:|---|
| 1 | none | accepted | 100 | no |
| 2 | 90 | accepted | 107 | e2 |
| 3 | 97 | accepted | 107 | e3 |
| 4 | 97 | accepted | 123 | no |
| 5 | 113 | late | 123 | no; review required |
| 6 | 113 | duplicate | 123 | no; e2 already exists |
| 7 | 113 | accepted | 124 | e6 |

Seven source records contain six distinct event IDs.
Five distinct events are accepted and one distinct event is late.
One source record is an exact duplicate.
Three alert intents are created, for `e2`, `e3`, and `e6`.
The final source position is seven and the final watermark is 114 minutes.
There are no human notifications or confirmed quality incidents in these results.

The equality boundary is also important.
If a new event arrived at minute 113 immediately after position 4, the previous watermark would be 113 minutes.
It would be accepted because `113 < 113` is false.
If it arrived at minute 112, it would be late under this policy.
Neither hypothetical event is part of the runnable fixture.

### 6.4 Crash and recovery example

Suppose the consumer has committed through position 3.
It starts the transaction for position 4, then the process fails before commit.
The event insertion and position update both roll back.
After restart, the stored position is still 3, so the consumer processes position 4 again.
No committed state for position 4 is lost or duplicated in this database.

Suppose instead the commit of position 4 succeeds but the process exits before the next read.
The stored position is 4 and the event is present.
The restarted consumer reads position 5 next.
This works only because the source still contains the required records and its ordering matches the stored position.
The fixture is present again because it is defined in the program; it does not simulate a durable broker.

If a real outbox publisher sends a notification and crashes before storing the acknowledgement, the transaction above does not solve that external ambiguity.
The receiver may see a retry.
Use an idempotency key and delivery audit, or reconcile before re-sending.
The database's unique alert intent is not a universal exactly-once guarantee.

### 6.5 What the example does and does not establish

The arithmetic establishes how this policy handles this particular order of seven synthetic records.
The local replay establishes deterministic agreement for the same fixture and implementation when its assertions pass.
Neither establishes a suitable late threshold for a real sensor network.
Neither measures throughput, availability, carrier compliance, temperature exposure duration, or product quality.
The late high reading is not safe merely because no live alert was generated.
It is visible for a separate review, and a real service must make that review operational.

## 7. Code, query, or calculation

### 7.1 Prerequisites and execution

Use Python 3.9 or later with its standard-library `sqlite3` module.
No package installation or network access is required.
Copy the following complete block into a new file named `stream_lab.py` in a scratch directory.
Run `python stream_lab.py` from that directory; on some Windows installations use `py stream_lab.py`.
The program creates its databases in a temporary directory and removes them when it exits.
It does not connect to a broker, send alerts, or modify production records.
The fixture acts as the producer: enumeration assigns source positions to records in list order.
The consumer persists position and derived rows in one SQLite transaction for each record.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3

# Times are integer minutes from an arbitrary origin; temperatures are °C.
# Repeating e2 represents a producer retry, not a second observation.
SOURCE = [
    ("e1", "S-17", 100, 5.0),
    ("e2", "S-17", 107, 12.0),
    ("e3", "S-17", 101, 9.0),
    ("e4", "S-17", 123, 7.0),
    ("e5", "S-17", 110, 15.0),
    ("e2", "S-17", 107, 12.0),
    ("e6", "S-17", 124, 10.0),
]
DISORDER_MINUTES = 10
THRESHOLD_C = 8.0
RULE = "temp-v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    position INTEGER NOT NULL,
    max_event_minute INTEGER
);
INSERT OR IGNORE INTO state VALUES (1, 0, NULL);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    shipment_id TEXT NOT NULL,
    event_minute INTEGER NOT NULL,
    temperature_c REAL NOT NULL,
    classification TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS occurrences (
    position INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL,
    classification TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    event_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    PRIMARY KEY (event_id, rule_version)
);
"""


def open_db(path):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def consume_one(db, position, record):
    event_id, shipment_id, minute, temperature = record
    # The context manager commits on success and rolls back on exceptions.
    with db:
        previous_position, maximum = db.execute(
            "SELECT position, max_event_minute FROM state WHERE id = 1"
        ).fetchone()
        if position != previous_position + 1:
            raise ValueError("source position is not the next position")
        existing = db.execute(
            "SELECT shipment_id, event_minute, temperature_c "
            "FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing is not None:
            if existing != (shipment_id, minute, temperature):
                raise ValueError("event ID reused with different content")
            classification = "duplicate"
        else:
            watermark = None if maximum is None else maximum - DISORDER_MINUTES
            late = watermark is not None and minute < watermark
            classification = "late" if late else "accepted"
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (event_id, shipment_id, minute, temperature, classification),
            )
            if not late:
                maximum = minute if maximum is None else max(maximum, minute)
                if temperature > THRESHOLD_C:
                    db.execute(
                        "INSERT INTO alerts VALUES (?, ?)", (event_id, RULE)
                    )
        db.execute(
            "INSERT INTO occurrences VALUES (?, ?, ?)",
            (position, event_id, classification),
        )
        db.execute(
            "UPDATE state SET position = ?, max_event_minute = ? WHERE id = 1",
            (position, maximum),
        )


def consume_remaining(db, source, stop_after=None):
    position = db.execute("SELECT position FROM state WHERE id = 1").fetchone()[0]
    end = len(source) if stop_after is None else min(stop_after, len(source))
    for index in range(position, end):
        consume_one(db, index + 1, source[index])


def snapshot(db):
    return {
        "state": db.execute(
            "SELECT position, max_event_minute FROM state WHERE id = 1"
        ).fetchone(),
        "events": db.execute(
            "SELECT * FROM events ORDER BY event_id"
        ).fetchall(),
        "occurrences": db.execute(
            "SELECT * FROM occurrences ORDER BY position"
        ).fetchall(),
        "alerts": db.execute(
            "SELECT * FROM alerts ORDER BY event_id, rule_version"
        ).fetchall(),
    }


def main():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "consumer.sqlite"
        db = open_db(path)
        consume_remaining(db, SOURCE, stop_after=3)
        assert snapshot(db)["state"] == (3, 107)
        db.close()  # Simulate a clean stop; reopen persistent state.

        db = open_db(path)
        consume_remaining(db, SOURCE)
        result = snapshot(db)
        assert result["state"] == (7, 124)
        assert [row[2] for row in result["occurrences"]] == [
            "accepted", "accepted", "accepted", "accepted",
            "late", "duplicate", "accepted",
        ]
        assert result["alerts"] == [
            ("e2", RULE), ("e3", RULE), ("e6", RULE)
        ]
        assert len(result["events"]) == 6
        assert [row[0] for row in result["events"] if row[4] == "late"] == ["e5"]

        # A conflicting retry must roll back, including source progress.
        try:
            consume_one(db, 8, ("e2", "S-17", 107, 99.0))
        except ValueError as exc:
            assert "different content" in str(exc)
        else:
            raise AssertionError("identity conflict was not rejected")
        assert snapshot(db) == result

        replay = open_db(Path(directory) / "replay.sqlite")
        consume_remaining(replay, SOURCE)
        assert snapshot(replay) == result
        replay.close()
        db.close()

        print("positions=7 distinct=6 accepted=5 late=1 duplicates=1")
        print("alerts=e2,e3,e6 watermark=114")
        print("restart=ok conflict_rollback=ok isolated_replay=ok")


if __name__ == "__main__":
    main()
```

### 7.2 Expected output and interpretation

If all embedded assertions pass, the program prints:

```text
positions=7 distinct=6 accepted=5 late=1 duplicates=1
alerts=e2,e3,e6 watermark=114
restart=ok conflict_rollback=ok isolated_replay=ok
```

The count `accepted=5` counts distinct accepted event IDs, not source records.
The duplicate is counted as one occurrence and does not create another event row.
The watermark of 114 is the maximum 124 minus ten minutes.
The clean stop after position 3 tests persistence across closing and reopening the database.
The intentionally conflicting `e2` tests that the transaction does not advance position 8.
The separate replay database tests the same source order, code, rule, and policy against the original result.

These checks are part of the runnable example; they have not been represented as a measured production test.
A clean close is not a forced process crash or a power-loss simulation.
SQLite's context manager provides the transaction behavior used here; filesystem and durability settings matter in a real deployment.
The list can be reconstructed on rerun but has no independent retention, acknowledgement, or concurrent partition behavior.
The late event is stored in `events`; there is no implemented late-review queue.
The alert rows are intents only; there is no implemented notifier.

### 7.3 Suggested inspection and extensions

Set a breakpoint after the first `consume_remaining` call to inspect position 3 and maximum 107.
Inspect `occurrences` to distinguish the event-level `late` status from the source-record-level `duplicate` status.
The `events` primary key enforces one row per event ID.
The `occurrences` primary key enforces one processed entry per fixture position.
The `alerts` composite primary key scopes uniqueness to event ID and rule version.
The code intentionally uses a new temporary location every execution, so it leaves no persistent audit artifact.

For a boundary experiment, insert a new event with minute 113 immediately after `e4` and before `e5`.
Update the expected occurrence and count assertions before running the modified program.
Because the comparison is strict, that new event is accepted against watermark 113.
Do not interpret its insertion as proof that a real sensor clock is accurate.
For a rule experiment, change the rule and version together, then run a separate replay and compare outputs.
Never use this local exercise to send historical notifications to real operators.

## 8. Data, standards, and regulatory context

### 8.1 Data contract and provenance

An event contract should identify the producer, schema version, event ID, shipment identifier, event time, value, and unit.
It should also define the source and semantics of the timestamp.
A sensor timestamp might represent measurement time, transmission time, or gateway receipt time.
A contract that calls all three simply `timestamp` creates ambiguity in late-data handling.

The lab uses an arbitrary integer minute axis so the arithmetic is easy to audit.
A production system should use unambiguous instants, typically UTC with an explicit encoding and precision.
Keep the original local time and zone separately if required for interpretation or audit.
Specify how clock synchronization, clock resets, daylight-saving transitions, and future timestamps are handled.
Record ingestion and processing times independently; they are not substitutes for event time.

Temperature units must be explicit.
A value of 8 means different things in °C and °F.
Also define measurement resolution, calibration, sensor placement, validity range, and missing-value semantics.
A single point measurement does not establish duration above a limit.
A product-specific exposure decision may require time integration, device calibration, location, and quality expertise.
The fixture has none of these additional facts.

Identity must survive a retry.
It must not silently change when a transport is retried or a record is copied to an archive.
Conversely, correcting an observation requires explicit versioning or a superseding event.
A same-ID different-content conflict in this lab is rejected rather than resolved automatically.
The source contract must decide how an authorized correction enters the system.

### 8.2 Relevant standards and representations

GS1 Electronic Product Code Information Services (EPCIS) is a standard for sharing visibility event data.
Its event model includes structured event time and business context; it is not a requirement to use EPCIS for this lab.
The official GS1 EPCIS resources linked in section 16 describe the standard and its current specifications.
A deployment must select the applicable version and profile rather than assuming every field in this example is an EPCIS field.
The simple Python tuple in this lab is not an EPCIS document.

The Internet Engineering Task Force's RFC 3339 specifies a profile for representing timestamps in Internet protocols.
Its offset and date-time syntax can reduce ambiguity in exchanged timestamps.
It does not guarantee that a sensor clock is correct or that a consumer's watermark is appropriate.
The lab's integer minute fixture is a teaching representation, not an RFC 3339 timestamp.

The CloudEvents specification defines an interoperable event envelope, including event identity and source context.
It can support consistent integration across systems when the participating systems agree on a profile.
It does not by itself provide durable transport, deduplication storage, a broker checkpoint, or an exactly-once business effect.
This lab does not claim its tuples are CloudEvents-compliant.

Apache Kafka documentation describes ordering and delivery semantics within Kafka's documented scope.
Its transactional and idempotent features have specific boundaries and configuration requirements.
They cannot automatically make an unrelated email or human action exactly once.
Consult the documentation for the deployed version before designing the guarantee.
The local lab uses no Kafka library or server.

### 8.3 Privacy, access, and regulation

A shipment event can indirectly identify a driver, customer, delivery location, or other person when joined with other data.
Collect the minimum fields needed for the decision and define access by role.
Do not assume a pseudonymous shipment ID is anonymous in every context.
Protect event archives, late-data queues, logs, and replay environments, not only the live database.
A synthetic fixture is preferable for teaching and local development.

The EU General Data Protection Regulation (GDPR) may apply to processing of personal data depending on context and jurisdiction.
Article 5 includes principles such as data minimization and storage limitation.
Whether a particular event is personal data and which legal obligations apply require a case-specific assessment.
The text of the regulation, not this lab, controls its interpretation.
Other jurisdictions and sector-specific rules may impose different obligations.

If a regulated product is involved, consult the applicable product, distribution, food, drug, or medical-device requirements.
Neither an 8.0 °C threshold nor a ten-minute lateness policy is established here as a regulatory requirement.
An alert is a prompt for qualified review, not a regulatory disposition.
Document approved retention, correction, access, audit, and incident processes with appropriate counsel and quality owners.
A replay archive can itself be regulated data and should follow approved retention and deletion policy.

### 8.4 Time, retention, and deletion

Retention for event replay and retention for personal data must be designed together.
Keeping every payload forever can conflict with storage limitation and organizational policy.
Deleting deduplication keys too early can make an old retry look new.
A system may need a minimal tombstone or another approved identity record after payload deletion.
Whether that is lawful and sufficient depends on the use case and jurisdiction.

A retention policy should distinguish the original source, derived state, alert audit, backup, and replay archive.
Deletion must account for each copy and documented restoration procedures.
Restoring an old backup can reintroduce data that was deleted later unless deletion state is reconciled.
That issue is not solved by the temporary SQLite databases in the lab.

## 9. Industry practice and case evidence

### 9.1 What the public sources establish

The Apache Kafka documentation describes an event streaming platform in which producers publish records and consumers read them.
It describes topics, partitions, ordering within a partition, offsets, and consumer groups.
The same documentation distinguishes delivery semantics and describes transactional processing capabilities.
These are platform capabilities and design concepts, not a claim that this lab runs Kafka.
The applicable guarantees depend on configuration, version, and the scope of the transaction.

The Google Cloud Dataflow documentation explains watermarks, event time, and late data in a stream-processing context.
It presents a watermark as an estimate of event-time completeness, not as a guarantee that late data is impossible.
Different frameworks have different watermark generation, idleness, trigger, and window semantics.
The simple maximum-minus-ten-minutes policy in this lab is not a claim to reproduce Dataflow internals.
The comparison is useful because both require an explicit choice about progress in event time.

The GS1 EPCIS standard is intended for visibility event sharing among applications and trading partners.
Its data model addresses what happened, when, where, and why in an interoperable form.
It provides a common vocabulary; it does not decide a specific organization's alert routing or temperature policy.
Using a common standard does not eliminate the need for validation, permissions, and business ownership.

These are public technical sources, not independent evidence of savings in the synthetic scenario.
They support the concepts of partitioned streams, event-time progress, and standardized visibility data.
They do not validate any numerical service-level claim in this lesson.

### 9.2 Named public case: UPS ORION

UPS publicly describes its On-Road Integrated Optimization and Navigation (ORION) system as a routing and navigation capability.
It is a named logistics example of operational decisions depending on data and execution context.
The linked UPS corporate material is a company description, not an independent causal evaluation of this lesson's alert policy.
It does not establish that UPS uses the lab's SQLite implementation, watermark, sensor fixture, or temperature threshold.

[INFERENCE] A routing operation can face a similar distinction between an observation's occurrence time and the time a decision system receives it.
That inference is a teaching analogy, not a documented statement about ORION's specific event-time architecture.
Do not transfer this fixture's counts or reliability behavior to UPS operations.

### 9.3 Public technical case: Kafka's log and consumer model

Kafka's official documentation describes a distributed log organized into topic partitions.
It describes each partition as an ordered sequence and consumer offsets as positions in that log.
A record's position is meaningful within its partition, not as a universal identifier for a physical business event.
Consumers can resume from stored positions when retained data and the consumer configuration permit.
Those documented concepts motivate preserving both source coordinates and stable business event IDs.

[INFERENCE] In a multi-carrier supply chain, partitioning by shipment may help preserve order for a shipment if producers consistently use the same key and the chosen partitioning scheme remains stable.
This is a design hypothesis, not a guarantee of order across partitions or through every upstream system.
A real design must investigate key selection, partition changes, retries, and producer behavior.

### 9.4 Practical design choices

Some organizations prefer to alert immediately on every individual threshold crossing.
Others require a sustained excursion, several consecutive measurements, a product-specific tolerance, or a quality review.
Those are business and product decisions, not consequences of selecting a streaming platform.
An individual-point rule is used here because its arithmetic is transparent.
It can produce noisy alerts when a sensor oscillates near the threshold.

A live operational view often differs from an authoritative historical view.
The operational view emphasizes prompt action with a stated lateness policy.
A historical view may revise earlier aggregates when late evidence arrives.
Both views should retain provenance so users can understand why numbers differ.
The lab retains late observations but does not implement a historical correction pipeline.

Operators need an escalation path, not merely an alert table.
A meaningful service definition includes the upstream observation, the processing path, the notification route, and the response owner.
A healthy consumer process does not establish that sensors are publishing or that staff received messages.
A low source-record lag does not establish that the sensor clock is current.
The operational metrics in section 5 therefore cover the whole path.

### 9.5 Evidence quality

The Kafka, Dataflow, GS1, RFC, and SQLite references are primary technical documentation or standards.
They support definitions and documented platform behavior, subject to the cited version and configuration.
The UPS material is a company source and should be read as a company claim.
None is an independent study of the lab's threshold or a measured result from the fixture.
Test local assumptions with historical data and controlled drills before proposing production service levels.
Do not interpret the educational result as a business case.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Producer generates a new ID on retry | Similar readings proliferate after network errors | Retry is indistinguishable from new event | Repeated alert intents | Assign stable identity before first publish |
| Two true readings share an ID | Same ID has differing payload | Source identity is not unique | One observation is rejected or overwritten | Stop on conflict; fix source identity and reconcile |
| Shipment ID used as event ID | Many readings for one shipment disappear | Dedupe grain is too coarse | Missed excursions | Use one ID per logical reading |
| Future-dated sensor | Watermark jumps far ahead | Maximum event time is poisoned | Later valid readings appear late | Validate clock skew and quarantine anomalies |
| Late high event dropped | Late count rises but no review exists | Only on-time path is monitored | Historical exposure is missed | Retain late payload and assign a quality review owner |
| Duplicate handled after lateness | Retry counts appear as late events | Watermark check precedes identity check | Misleading quality and alert metrics | Check known IDs before classifying new events |
| Progress committed before state | Checkpoint moves but output is missing | Crash separates independent writes | Permanent lost effect | Commit source position and state atomically |
| External notification retried | Same operator receives identical pages | Send succeeded but acknowledgement was not recorded | Alert fatigue | Stable idempotency key and receiver reconciliation |
| Retention too short | Consumer lag nears source retention | Outage exceeds replay horizon | Irrecoverable source gap | Alert on retention headroom; maintain verified archive |
| Cross-partition ordering assumed | State transitions appear in impossible order | Independent partitions interleave | Incorrect derived state | Key related events consistently and define merge policy |
| Source position used as business ID | Same logical event appears at multiple offsets | Retry creates a new record position | Duplicate effects | Persist logical identity separately from source coordinates |
| One global watermark for unrelated streams | Quiet source records become late | Active source advances shared maximum | Biased classification | Define per-partition progress and idle-source policy |
| Schema changes without contract | Parsing errors increase after producer release | Field type or meaning changes | Stalled consumer or invalid alerts | Version schema and test compatibility |
| Dead-letter queue has no owner | Quarantine age increases | Exceptions are stored but not resolved | Silent data loss by neglect | Assign owner, alert on age, record disposition |
| Celsius/Fahrenheit confusion | Extreme values cluster after integration | Unit not carried or converted | False or missed alerts | Require explicit unit and validate conversion |
| Rule threshold silently changes | Alert counts shift with no deployment record | Version not tied to semantics | Unreproducible replay | Version rules and retain deployment metadata |
| Replay sends live messages | Historical period causes alert storm | Outbox or notifier is enabled in replay | Duplicate operational actions | Isolate replay and disable external effects by default |
| Lag alone used as freshness | Record lag is low but readings are old | Upstream is stalled or buffered | False health signal | Monitor event age and producer heartbeat |
| Alert threshold is overly sensitive | Many alerts close without action | Noise around a point threshold | Operator fatigue and slow response | Validate policy with quality owner and historical review |
| Secrets in replay artifacts | Logs or archives contain credentials | Raw payloads copied broadly | Security incident | Minimize data, restrict access, rotate exposed secrets |
| Dedupe state expires too soon | Old retries become new events | Identity retention shorter than replay window | Duplicate intent | Align retention horizons and test old retries |
| Clean shutdown mistaken for crash test | Only orderly restarts have been exercised | No abrupt failure at commit boundary | False confidence in recovery | Inject failures around transactions in a safe test system |

A red flag is an observation requiring investigation, not proof of a specific root cause.
For example, rising late counts could indicate clock drift, network buffering, or a changed producer schedule.
Check source, event time, ingestion time, partition, and deployment changes before changing the watermark.
Do not increase allowed disorder solely to make a dashboard look healthy.
A wider interval changes action delay and may mask an upstream failure.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or address it |
|---|---|---|---|
| One source ordering | Makes a single position meaningful | Multiple independent partitions | Run partition-specific tests and define merge behavior |
| Stable immutable event IDs | Makes exact duplicate detection possible | Producer reuses IDs or changes content | Contract tests and conflict-rate monitoring |
| Fixture remains available on restart | Lets the checkpoint locate remaining records | Source deletion or retention expiry | Restore from a durable source or verified archive |
| One consumer writer | Avoids coordination and fencing complexity | Concurrent instances write the same state | Test ownership, isolation, and fencing in the real design |
| Event times are comparable | Supports a shared maximum and watermark | Unsynchronized or differently encoded clocks | Compare to trusted ingestion clocks and validate skew |
| Ten minutes is an illustrative disorder bound | Gives a calculable policy | Distribution differs by source or changes over time | Measure lateness by source and agree on a policy |
| Late observations are retained | Preserves evidence for review | Storage failure or deletion | Verify durable quarantine and owner workflow |
| A single point can trigger an intent | Keeps business rule transparent | Product requires duration or cumulative exposure | Obtain validated quality rules and relevant sensor history |
| 8.0 °C is a scenario threshold | Enables a deterministic example | Different product and transport requirements | Have qualified quality owners set and approve limits |
| Alert intent is enough for this lab | Avoids uncontrolled external effects | Operator needs actual notification | Build and test publisher, receiver, and acknowledgement |
| SQLite transaction covers local state | Makes checkpoint and state commit together | State or offset is stored in another system | Design transactional integration or reconciliation |
| Same order yields same classification | Supports isolated replay comparison | Replay changes ordering or watermark settings | Preserve source coordinates and compare policy versions |
| Input records have valid types | Keeps example focused on streaming semantics | Malformed, unauthorized, or nonfinite input | Add a validated schema and durable rejection policy |
| Rule version identifies semantics | Makes alert identity auditable | Logic changes without version change | Tie version to reviewed release and test fixtures |
| Local temporary database is sufficient | Enables an isolated educational run | Need for retention, sharing, or disaster recovery | Use an approved durable environment with backup controls |
| Sensor values describe useful facts | Permits interpretation as observations | Bad calibration, placement, or tampering | Calibrate, validate provenance, and investigate anomalies |

The code does not claim to solve concurrent ordering, state migration, schema evolution, or distributed transactions.
It does not create broker topics, consume a real partition, or send an operational page.
It cannot establish a latency percentile or an availability objective without a timed production-like experiment.
It also cannot determine whether a particular product is fit for use.
Those limitations do not invalidate the arithmetic; they bound what can be inferred from it.

## 12. Exercises

### Exercise 1 — Separate the clocks

Time: 10 minutes.
A sensor measures a shipment at 09:04 UTC, a gateway accepts it at 09:08 UTC, and a consumer evaluates it at 09:11 UTC.
The producer retries the same logical event at 09:12 UTC.
Deliverable: label event, ingestion, and processing times, then explain what the retry changes.
Acceptance: preserve the original event identity and event time; distinguish the second delivery from a second physical measurement.

### Exercise 2 — Calculate watermark classifications

Time: 20 minutes.
Use accepted event minutes 40 and 48, an allowed disorder of six minutes, and then a previously unseen event at minute 42.
Next consider a previously unseen event at minute 41.
Deliverable: show the previous watermark before each event and classify both later arrivals.
Acceptance: apply the strict comparison, show the maximum after each event, and explain the equality boundary.

### Exercise 3 — Choose identity grain

Time: 15 minutes.
Shipment `S-9` has event `a` at minute 20 and event `b` at minute 21, both 9.0 °C.
A retry of `a` arrives at a third source position with the same payload.
Deliverable: list event rows, occurrence classifications, and potential intent keys.
Acceptance: preserve two distinct readings and identify only the repeated `a` as a duplicate.

### Exercise 4 — Reason about crash points

Time: 20 minutes.
Consider a consumer transaction that writes a matching event, an alert intent, and source progress together.
Place a crash immediately before commit, immediately after commit, and after an external receiver accepts a notification but before its acknowledgement is stored.
Deliverable: describe the durable state and safe next action at each point.
Acceptance: distinguish local transaction atomicity from the external notification ambiguity.

### Exercise 5 — Run and inspect the fixture

Time: 30 minutes.
Copy and run the section 7 program in a scratch directory.
Deliverable: record its three output lines and explain why `e5` is late and the repeated `e2` is not an additional alert.
Acceptance: report seven positions, six distinct events, one late event, one duplicate occurrence, and alert IDs `e2`, `e3`, and `e6`.
Explain that the alert table is not a notification service.

### Exercise 6 — Design a safe replay

Time: 30 minutes.
A consumer was unavailable for two hours and the source retains three days of records.
An operator wants to rebuild derived state and send all historical alerts again.
Deliverable: write a replay plan with source boundaries, isolation, verification, promotion, notification policy, and authorization.
Acceptance: preserve original identities and order, disable external effects by default, reconcile outputs, and distinguish current action from historical review.
Do not assume the source has every required partition without checking.

### Exercise 7 — Diagnose misleading health

Time: 20 minutes.
Consumer record lag is zero, but the latest event time is 45 minutes old.
The normal sensor reporting interval is five minutes.
Deliverable: name three plausible explanations, two additional measurements, and an owner for the first investigation.
Acceptance: do not conclude that the consumer alone is healthy or unhealthy; distinguish sensor, transport, and clock issues.

### Exercise 8 — Critique a delivery guarantee

Time: 20 minutes.
A vendor proposal says: `Our broker has transactions, so every customer email is exactly once.`
Deliverable: diagram the broker, state store, outbox publisher, email provider, and recipient boundary.
Acceptance: identify the send-versus-acknowledgement crash window and specify an idempotency or reconciliation control.
State which boundary is actually covered by any documented guarantee.

### Exercise 9 — Apply data and quality controls

Time: 25 minutes.
A real deployment would link sensor readings to driver identities and customer delivery addresses.
Deliverable: prepare a one-page data contract and review checklist.
Acceptance: include event identity, time semantics, unit, source provenance, minimum necessary data, access, retention, late-data ownership, and qualified regulatory review.
Do not assert that this exercise determines the legal basis or product safety requirement.

## 13. Self-check answers

### Answer 1 — Clocks and retry

09:04 UTC is event time because it is when the physical observation occurred.
09:08 UTC is the first gateway ingestion time in the scenario.
09:11 UTC is the first consumer processing time.
A retry at 09:12 UTC is a new delivery attempt, not a new physical observation.
It should retain the original stable event ID and 09:04 UTC event time.
The retry may have its own ingestion and processing timestamps for operational monitoring.

### Answer 2 — Watermark and boundary

After minute 40, the maximum is 40 and the watermark is `40 - 6 = 34` minutes.
Minute 48 is accepted because 48 is not less than 34.
The maximum becomes 48 and the watermark becomes `48 - 6 = 42` minutes.
The new event at minute 42 is accepted because the policy says late only when time is strictly less than 42.
The maximum remains 48 and the watermark remains 42.
The new event at minute 41 is late because `41 < 42`.
The maximum remains 48 and the watermark remains 42.
A different implementation using `<=` would have a different boundary and must document it explicitly.

### Answer 3 — Grain and deduplication

The event table has two event IDs, `a` and `b`.
The occurrence table has three source positions: accepted `a`, accepted `b`, and duplicate `a`.
If both accepted temperatures exceed the configured threshold, the intent keys are `(a, rule)` and `(b, rule)`.
The third occurrence adds no new intent for `a`.
The same temperature and shipment do not prove that `a` and `b` are the same physical observation.

### Answer 4 — Crash boundaries

Before commit, the event, intent, and progress writes roll back together.
Restart from the previous committed source position and retry the record.
After commit, the event, intent, and new position are durable together within the database's configured durability boundary.
Restart from the next source position, assuming the source remains available and ordered as expected.
After a receiver accepts a notification but before the sender stores acknowledgement, the sender cannot infer whether the receiver acted.
A retry can create a duplicate unless the receiver honors an idempotency key or a reconciliation process resolves the uncertainty.
The local database transaction does not cover the external receiver.

### Answer 5 — Fixture

The expected first line is `positions=7 distinct=6 accepted=5 late=1 duplicates=1`.
The expected second line is `alerts=e2,e3,e6 watermark=114`.
The expected third line is `restart=ok conflict_rollback=ok isolated_replay=ok`.
At position 5, the previous maximum is 123 and the watermark is 113 minutes.
`e5` is at minute 110, so it is late under the strict comparison.
At position 6, `e2` is already known with the same payload, so the occurrence is a duplicate.
Its original intent remains one row; no new message is delivered by the lab.

### Answer 6 — Replay rubric

A strong plan identifies each affected partition, the inclusive start and end boundaries, and actual retention availability.
It fences the live consumer if checkpoint or live state might be changed.
It records software, schema, rule version, source coordinates, and the reason for replay.
It runs in an isolated state store with notification effects disabled.
It preserves event IDs, source order, and duplicate records where reproducibility requires them.
It compares counts and identities for accepted, late, duplicate, conflict, and alert results.
It investigates gaps and unexplained differences before approval.
It sends only explicitly authorized, relevant notifications through a deduplicated mechanism.
It documents who approved promotion and monitors the live cutover.
A weak answer simply resets offsets and enables the ordinary notification channel.

### Answer 7 — Health diagnosis rubric

Plausible causes include a sensor outage, gateway buffering, producer failure, an upstream transport delay, or a wrong sensor clock.
A zero consumer lag means the consumer is caught up with the source as measured, not that the source is current.
Useful measurements include the last producer heartbeat, the ingestion timestamp of the last record, and the event-to-ingestion delay distribution.
Compare the sensor clock with a trusted time source and inspect gateway backlog.
A named sensor or producer operations owner should investigate first, with platform support as needed.
Do not assert a single root cause without evidence.

### Answer 8 — Guarantee rubric

A strong answer identifies the documented transaction boundary and any assumptions on producers and consumers.
It draws a distinct outbox-to-email-provider boundary.
The crash after provider acceptance but before sender acknowledgement can cause a retry.
The provider or downstream system needs a stable idempotency key, or the organization needs explicit reconciliation.
Even a successful provider acknowledgement is not the same as a human reading or acting on the email.
A weak answer treats the broker transaction as a guarantee over every external system and human action.

### Answer 9 — Data and quality rubric

A strong contract defines one reading's event ID, producer identity, shipment link, event-time meaning, unit, and version.
It separates event, ingestion, and processing clocks.
It defines validation, correction, late-data review, and operational ownership.
It minimizes driver and customer data and specifies access, retention, audit, and deletion handling.
It requests qualified review of applicable law and product-specific quality requirements.
It does not present the synthetic 8.0 °C rule as a universal legal threshold.

## 14. Interview questions

### 1. Why is event time different from processing time?

A strong answer distinguishes occurrence at the source from evaluation at the consumer and discusses delay or buffering.
A weak answer assumes the two times are always equal.

### 2. What does a watermark mean?

A strong answer explains an event-time completeness estimate and the implications for late records.
A weak answer claims that no event older than the watermark can ever arrive.

### 3. How do you select an event ID?

A strong answer chooses the grain of one logical observation and preserves identity across retries.
A weak answer uses a shipment ID for all readings or generates a new ID on every retry.

### 4. How do you detect conflicting retries?

A strong answer compares the immutable payload for a known ID and routes mismatches to investigation.
A weak answer overwrites earlier content without preserving evidence.

### 5. What belongs in the same transaction as a checkpoint?

A strong answer includes durable derived state and the relevant alert intent, while naming the transaction boundary.
A weak answer advances progress first and assumes the output will always be written later.

### 6. Does an outbox guarantee a human sees a message once?

A strong answer distinguishes intent, publisher retries, receiver idempotency, and human acknowledgement.
A weak answer treats one database row as proof of one delivered and acted-upon notification.

### 7. Why can replay change the answer?

A strong answer names changed ordering, rule versions, schema versions, policy settings, and retained source boundaries.
A weak answer assumes identical records always imply identical order-sensitive classifications.

### 8. How do you handle a late high-temperature reading?

A strong answer preserves it, explains the live policy, and assigns a quality review based on business requirements.
A weak answer deletes it because the live watermark passed.

### 9. What does zero consumer lag tell you?

A strong answer says the consumer is caught up with the available source position, subject to measurement scope.
It separately checks source freshness and producer health.
A weak answer concludes the physical sensors and downstream notifications are healthy.

### 10. How do you prevent a replay alert storm?

A strong answer uses isolated replay, disabled effects, reconciliation, explicit approval, and stable downstream keys.
A weak answer replays into the live notifier and relies on operators to ignore duplicates.

## 15. Key terms

- **Accepted event:** A new observation that meets this lab's event-time policy; see [mechanics](#5-the-mechanics-step-by-step).
- **Alert intent:** A durable request for possible notification, not proof of delivery; see [delivery](#58-deliver-an-exception-safely).
- **Allowed disorder:** The event-time interval subtracted from the maximum; see [worked example](#6-worked-example).
- **At-least-once delivery:** A delivery model that permits retries; see [vocabulary](#4-definitions-and-vocabulary).
- **Checkpoint:** Durable consumer progress coupled to state; see [transactions](#57-commit-state-and-progress-together).
- **Conflict:** Reuse of an event ID with a different payload; see [identity](#55-distinguish-duplicate-conflict-and-new-event).
- **Consumer:** A component that reads and processes source records; see [mental model](#3-the-core-mental-model).
- **Deduplication:** Identification of repeated deliveries of one logical event; see [identity](#55-distinguish-duplicate-conflict-and-new-event).
- **Disorder:** Difference between source arrival order and event-time order; see [event-time policy](#56-apply-event-time-policy).
- **Duplicate:** An exact repeat of an already known event ID and payload; see [worked example](#6-worked-example).
- **Event:** A recorded observation with a defined grain; see [producer contract](#52-specify-the-producer-contract).
- **Event ID:** Stable identity of one logical observation; see [vocabulary](#4-definitions-and-vocabulary).
- **Event time:** Time the observation occurred according to its source; see [data context](#8-data-standards-and-regulatory-context).
- **Exception:** A record matching a stated business rule; see [decision](#51-define-the-decision-and-the-effect).
- **Freshness:** Age of a relevant event or output; see [monitoring](#510-monitor-the-whole-path).
- **Idempotency:** Repetition without an additional intended effect; see [delivery](#58-deliver-an-exception-safely).
- **Ingestion time:** Time a source accepts a record; see [clocks](#answer-1--clocks-and-retry).
- **Lag:** Difference between source progress and consumer progress; see [monitoring](#510-monitor-the-whole-path).
- **Late event:** A new observation earlier than the previous watermark; see [event-time policy](#56-apply-event-time-policy).
- **Offset:** A source position within a partition; see [vocabulary](#4-definitions-and-vocabulary).
- **Outbox:** A durable table of intents for later delivery; see [delivery](#58-deliver-an-exception-safely).
- **Partition:** An independently ordered subset of a source; see [source durability](#53-define-ordering-and-source-durability).
- **Processing time:** Time a consumer evaluates a record; see [clocks](#answer-1--clocks-and-retry).
- **Producer:** A component that creates or publishes records; see [producer contract](#52-specify-the-producer-contract).
- **Quarantine:** Isolation of records that require investigation; see [validation](#54-validate-before-changing-business-state).
- **Replay:** Reprocessing retained records under a declared policy; see [replay](#59-restart-replay-and-reconcile).
- **Retention:** Duration for which records or state remain available; see [source durability](#53-define-ordering-and-source-durability).
- **Rule version:** Identifier for the semantics used to derive an alert; see [fixture code](#7-code-query-or-calculation).
- **Source position:** Location in a particular ordered source; see [mental model](#3-the-core-mental-model).
- **Watermark:** A policy estimate of event-time completeness; see [event-time policy](#56-apply-event-time-policy).

## 16. Sources

The following sources are publicly checkable primary technical documentation, standards, or a named company source.
Their inclusion does not imply that the lab implements or is certified against a standard.

1. Apache Kafka, [Documentation](https://kafka.apache.org/documentation/).
   Consult the deployed version for partition ordering, consumer offsets, producer configuration, and delivery semantics.
2. Apache Kafka, [Design](https://kafka.apache.org/documentation/#design).
   Describes log design and delivery semantics within the platform's documented scope.
3. Google Cloud, [Watermarks and late data in Dataflow](https://cloud.google.com/dataflow/docs/concepts/streaming-pipelines#watermarks-and-late-data).
   Explains event-time progress and late data in that service; this lab does not implement Dataflow.
4. GS1, [EPCIS and CBV](https://www.gs1.org/standards/epcis).
   Official entry point for the visibility-event standard and its versions.
5. IETF, [RFC 3339: Date and Time on the Internet: Timestamps](https://www.rfc-editor.org/rfc/rfc3339).
   Defines an Internet date-time profile, not sensor-clock correctness.
6. Cloud Native Computing Foundation, [CloudEvents specification](https://github.com/cloudevents/spec).
   Defines event interchange conventions, not an end-to-end delivery guarantee.
7. SQLite, [Atomic Commit in SQLite](https://www.sqlite.org/atomiccommit.html).
   Describes SQLite atomic-commit behavior and underlying assumptions.
8. Python Software Foundation, [`sqlite3` — DB-API 2.0 interface for SQLite databases](https://docs.python.org/3/library/sqlite3.html).
   Documents connection context managers and database operations used in the code.
9. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary legal text; applicability and interpretation require case-specific advice.
10. UPS, [ORION: On-Road Integrated Optimization and Navigation](https://about.ups.com/us/en/our-company/innovation-driven/orion.html).
    Company source for the named routing system; it is not independent evidence about this lab's performance.

## 17. Links

- [Module 30 topic index](_index.md): position of this lab in the course sequence.
- [Module 30 survey](../../../course/30_simulation_optimization_production_labs.md): broader simulation, optimization, and production context.
- [Environment and repository lab](01_environment-repo.md): Python environment and reproducible lab setup.
- [Demand forecast pipeline](02_demand-forecast-pipeline.md): batch pipeline validation and publishing context.
- [Vehicle routing and dispatch replay](05_vrp-dispatch.md): dispatch decisions and replay without future information.
- [Module expansion specification](../../../course/MODULE_EXPANSION_SPEC.md): authoring requirements and evidence conventions.
