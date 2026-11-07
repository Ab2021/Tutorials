---
collection: streaming-event-systems
file_id: "02"
topic: CDC ordering, snapshot handoff, and duplicate-safe materialization
level: 3
status: draft
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: Source database, Debezium connector, broker, schema, or sink semantics change
notation_style: plain-text
prerequisites: ["01_Kafka_Consumer_Partitions_and_Lag.md"]
pattern_tags: [cdc, snapshots, ordering, materialization, recovery]
source_status: OPEN
owner_role: streaming-platform maintainer
source_index: INDEX.md
---
# CDC Ordering, Snapshot Handoff, and Duplicate-Safe Materialization

## 1. File Header

**Outcome:** Design and diagnose a source-log-to-materialized-state pipeline without inventing global ordering or exactly-once guarantees.
**Audience:** Engineers who can read a relational transaction, reason about partitioned logs, and operate durable sinks.
**Prerequisite:** [Kafka consumer partitions and lag](01_Kafka_Consumer_Partitions_and_Lag.md).
**Scope:** Snapshot consistency, source positions, handoff, per-key ordering, deletes, transactions, replay, recovery, and sizing.
**Exclusions:** Vendor-specific configuration recipes, cross-database distributed transactions, and a complete security implementation.
**Evidence:** ILLUSTRATIVE and DERIVED; no production measurement or deployment is claimed.
**Editorial status:** Draft and unreviewed; validate every operational instruction for the deployed versions.
**Provenance:** The directory inventory names `Case_Studies_Streaming_Kafka_Flink_Redpanda/Kafka_Architecture/02_Real_Time_CDC_Data_Replication_Debezium.md`, but that case file was not available in this workspace.
Its title is a topic lead, not evidence of its contents or of a real incident.
The [source coverage brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#kafka) maps this subject to the streaming module.
The conceptual protocol below is a design contract, not a claim that every Debezium connector implements identical behavior.
For implementation, consult the deployed connector's [Debezium documentation](https://debezium.io/documentation/), the database's snapshot and log documentation, and the matching [Apache Kafka documentation](https://kafka.apache.org/documentation/).
Specific product behavior, versions, schemas, and permissions remain OPEN until checked against that deployment.

**Takeaway:** State exactly which source history the snapshot represents before designing replay or claiming convergence.

## 2. Concept Map / Mental Model

```text
source transactions ------> committed database log ------------> connector
        |                          |                                  |
        v                          v                                  v
consistent snapshot ------> boundary / overlap protocol ------> keyed topic
                                                                      |
                         per-partition replay and durable effects <---+
                                      |
                                      v
                         versioned materialized view + tombstones
                                      |
                                      v
                         reconciliation, retention, recovery
```

A consistent snapshot is a database view, not simply a table scan.
The log position identifies progress in a particular source history, not in the Kafka topic.
A connector offset records connector progress and can lag emitted events.
A broker offset identifies a record's position within one Kafka partition.
A sink checkpoint records completed input, not necessarily completed effects unless its commit protocol makes that true.
The same logical change may appear more than once after a failure.
A correct handoff has neither an uncovered interval nor a stale snapshot row that overwrites a newer stream update.
A correct materializer tolerates duplicate delivery without permitting older distinct changes to reverse state.
Transactions may span keys, tables, and partitions, so per-key convergence is weaker than atomic cross-key visibility.

**Takeaway:** The snapshot boundary, source log, topic offsets, and sink state are separate coordinate systems.

## 3. Foundational Definitions

- **Change data capture (CDC):** Representation of database changes as a stream derived from a database log or another capture mechanism.
- **Source history:** A particular database's lineage of committed changes, including any epoch or failover identity required to interpret positions.
- **Consistent snapshot:** A set of rows read according to a database-supported consistency point or transaction snapshot.
- **Snapshot watermark:** The logical boundary separating the state represented by the snapshot from subsequent committed changes, when the protocol provides such a boundary.
- **Log coordinate:** Database-specific location or token used to resume capture; its representation and comparison rules depend on the source.
- **Kafka offset:** Position in one Kafka topic partition; it cannot be compared numerically with a database log coordinate.
- **Source event identity:** An identifier stable across re-emission of the same logical change, including enough source and within-position information to avoid collisions.
- **Entity key:** Stable identity used to select the destination row and, if applicable, partition events for that entity.
- **Event version:** Comparable per-key ordering token whose semantics are explicitly defined for the source and all relevant changes.
- **Transaction identifier:** Token grouping changes in a source transaction; it is not automatically a sortable commit sequence.
- **Commit order:** Order in which transactions become durable according to source semantics; transaction start order can differ.
- **Snapshot read event:** A representation of a row observed by a snapshot; it is not automatically a later business update.
- **Create/update/delete:** Logical mutations, which may be encoded with before/after images and connector-specific operation codes.
- **Delete envelope:** A change record describing a row deletion; its representation and prior image depend on connector configuration.
- **Kafka tombstone:** A keyed record with null value, conventionally relevant to compaction; it is not necessarily the only representation of a database delete.
- **Logical tombstone:** Sink metadata recording a deletion and its version, preventing older records from resurrecting the entity.
- **Deduplication:** Recognizing repeated delivery of the same logical event; it is distinct from rejecting an older, nonduplicate event.
- **Materialization:** Maintaining a queryable derived state from the stream.
- **Idempotent effect:** An operation whose repetition with the same logical input does not change the outcome beyond the first application.
- **Contiguous durable frontier:** Next input offset after the longest partition prefix whose required effects are durably complete.
- **Retention horizon:** Period or position range for which source or broker history remains recoverable under actual policies.
- **Reconciliation:** Comparing source and destination at compatible boundaries to find missing, extra, or different state.
- **Epoch:** A namespace distinguishing histories whose positions are not safely comparable, such as an unsupported failover lineage change.

A scalar source position is not necessarily sufficient to distinguish multiple row events at the same location.
A transaction ID alone may be reused, unordered, or insufficiently scoped.
A timestamp is not necessarily unique, monotonic, or comparable across hosts.
A snapshot watermark is a boundary contract, not a wall-clock estimate.
A durable delete marker must survive at least the allowed stale replay horizon if it is the only anti-resurrection guard.

**Takeaway:** Identity, order, progress, and time are different concepts even if a product exposes them in one record.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Consider one source, one table, one stable primary key, a consistent snapshot at boundary W, and an ordered source history after W.
Assume every committed row mutation has a strictly ordered per-key version and that the capture protocol covers every change after W.
The snapshot row is the value at W; it is not a fresh update at the later time when it is delivered.
For each key, the destination stores a payload or delete marker together with its last applied version.
A newer version replaces the stored version and payload atomically.
An equal version is a retry only if event identity and semantics also match; a conflict is quarantined, not silently accepted.
An older version has no effect on the current state.
The database's conditional update or a fenced single owner must enforce the comparison under concurrent delivery.

**Mechanism contract:** Input is `(key, version, event_id, operation, payload)`.
Output is the newest accepted value or delete marker for that key.
Preconditions are complete capture, deterministic decoding, and a version comparator valid within that source lineage.
The state transition compares the incoming version to the stored version in the same atomic destination transaction.
The invariant is that the stored version never decreases.
The proof is induction: initial state has no version; a transition either preserves it or installs a strictly greater version.
This establishes ordering of accepted state, not completeness of captured history.
A lost newer event still leaves the sink stale even though the invariant holds.

**Worked trace:** A snapshot yields key A at W with value 10.
A later source update has version 12 and value 20.
If update 12 arrives first and the snapshot is then presented as version W, the snapshot is rejected as older.
If the snapshot arrives first, update 12 replaces it.
Both paths end with 20, provided W compares correctly with 12 and no required later event is missing.
A connector whose snapshot records do not expose a comparable boundary needs a different documented overlap protocol.
Do not invent a comparison between a row-read timestamp and a database log position.

### 4.2 General Case

In a multi-table pipeline, define source identity, lineage, table identity, row key, and the source-specific commit coordinate for every event.
If one commit coordinate contains several mutations to the same key, include a documented within-transaction or within-log order.
If source positions are partially ordered, a scalar maximum is not valid without a proven source-specific reduction.
The sink's atomic comparison key must be scoped to the entity's actual ordering domain.
A unique event identifier supports duplicate recognition; a separate version supports ordering distinct events.
The two can coincide only if the connector guarantees both uniqueness and ordering at the required grain.
Store the resulting entity state and the progress needed to recover it atomically when the sink supports such a transaction.
Otherwise, write effects first and commit the input frontier afterward, allowing duplicates on crash and requiring idempotent effects.
A Kafka offset can be persisted with a destination mutation in a destination transaction when the application deliberately implements that protocol.
This does not automatically make effects in another service atomic.
External side effects require their own idempotency key, transactional outbox, or explicit compensating design.

**Materializer pseudocode:**

```text
begin destination transaction
  prior = lock_or_compare_and_swap(entity_key)
  if event_id is known with a different immutable payload:
      abort and quarantine collision
  else if incoming_version > prior.version:
      apply after-image or logical delete marker
      store incoming_version and event identity
  else if incoming_version == prior.version and identity/payload agree:
      no-op duplicate
  else if incoming_version < prior.version:
      no-op stale event, increment observable counter
  else:
      abort and quarantine incomparable or conflicting event
  advance only the contiguous durable input frontier
commit destination transaction
```

A conditional write must include creation races, delete races, and concurrent workers in its atomicity scope.
Do not advance the checkpoint in a transaction that can commit while its required effect rolls back.
If processing records in parallel, completion of offset 19 does not authorize committing 20 while offset 18 is unresolved.
If partition ownership changes, fence the old owner before it can advance durable state under stale ownership.
A source transaction's row events may be individually applied under this algorithm; the output is not necessarily transaction-atomic.
Cross-key visibility requires an additional transaction-aware staging and commit design described below.

### 4.3 Key Special Cases / Variants

| Variant | What changes | Preconditions | Use when | Avoid when |
|---|---|---|---|---|
| Quiesced initial load | Freeze writes around copy and boundary | An authorized maintenance window; full write freeze | Small systems can accept downtime | Writes cannot pause or freeze is incomplete |
| Consistent snapshot plus log | Copy at W, capture the required suffix | Database and connector support a documented boundary protocol | Continuous writes and current state are required | Log history cannot be retained until handoff |
| Incremental snapshot | Snapshot chunks overlap with streaming | Connector-specific collision and watermark rules | Full long snapshot is operationally risky | Connector/version semantics are unknown |
| Versioned upsert | Retain latest per-key version and payload | Comparable stable versions and atomic conditional write | Independent key-level state is sufficient | Cross-key atomic reads are required |
| Event identity ledger | Persist identifiers for a bounded replay horizon | Stable collision-free identity and retention | Effects are not naturally idempotent | Ledger cannot cover permissible replay |
| Transaction staging | Buffer transaction events before visibility | Reliable transaction boundaries, completeness, durable staging | Readers need multi-row atomic visibility | Transaction metadata is unavailable or unbounded buffering is unacceptable |
| Full destination rebuild | Materialize into an empty new generation | Complete snapshot/history and controlled cutover | Correctness cannot be proved after a gap | No usable history or consistency boundary exists |

In a snapshot protocol, the exact boundary inclusion rule may be inclusive or exclusive.
Record the rule alongside W, not merely its numeric representation.
With inclusive replay, overlapping boundary events require explicit duplicate and snapshot collision handling.
With exclusive replay, the database and connector must guarantee that all changes not represented by the snapshot are strictly after W.
A primary-key change may appear as delete-old-key plus create-new-key, but this representation is connector-specific.
A key move across partitions is not a single ordered per-key operation for the old and new identities.
If readers require atomic identity moves, stage the relevant transaction or impose a higher-level indirection.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

| Assumption | Counterexample | Consequence | Correct response |
|---|---|---|---|
| Same key routes to one ordered partition | Partition count or partitioner changes | Updates may arrive through different partitions out of source order | Establish version comparison or a controlled repartition migration |
| Version uniquely orders changes | Two updates share a transaction commit coordinate | Equal-position conflict or silent loss | Add a documented event sequence within the transaction |
| Snapshot state precedes stream changes | Last snapshot row arrives after an update | Blind upsert restores stale data | Apply the connector's snapshot collision protocol |
| Checkpoint follows effect | Checkpoint is committed first and process crashes | An effect may be skipped permanently | Commit effects first with idempotency, or use an atomic boundary |
| Latest payload alone is enough | Delete marker is physically removed, then old update is replayed | Deleted key is resurrected | Retain a versioned tombstone or rebuild under a new generation |
| All rows in a transaction are visible together | Events traverse two partitions with independent consumers | Readers observe partial source transaction | Stage by transaction and publish after completeness |
| History is indefinitely replayable | Log expires before restart | No safe continuation from the old position | Stop, preserve evidence, and rebootstrap using a verified boundary |

A strictly increasing version is not a substitute for source-event completeness.
If position B has been skipped, accepting position C proves only that C is newer than the stored position.
If arrival and processing rates are equal, a positive backlog does not drain in a constant-rate model.
If replay can begin arbitrarily far back, no finite tombstone or event ledger horizon alone guarantees protection forever.
Restrict replay by policy, retain sufficient durable metadata, or rebuild a new isolated generation.

**Takeaway:** Correctness needs a complete handoff, a valid ordering relation, and an atomic sink transition; no one of them implies the others.

## 5. Worked Numerical Example

### 5.1 Boundary and state trace

This is an ILLUSTRATIVE trace, not a report of measured connector behavior.
Assume a source offers a consistent snapshot representing all committed changes through W=100 and streaming starts strictly after W.
Assume the numeric versions below are valid, unique, per-key comparable source positions; real source coordinate formats vary.
Assume the source and connector preserve every required post-boundary change through recovery.
The snapshot contains A=10 and B=5 at W.
After W, a committed change sets A=12 at position 101.
A second committed change deletes B at position 102.
A third committed change sets A=15 at position 103.
The connector can replay a previously emitted change after an offset checkpoint failure.
The destination retains `(key, version, value-or-delete)` and compares atomically.

| Arrival | Input | Stored A | Stored B | Decision |
|---:|---|---|---|---|
| 1 | A update 101 = 12 | 101:12 | absent | Apply newer update |
| 2 | A snapshot W = 10 | 101:12 | absent | Reject stale snapshot |
| 3 | B snapshot W = 5 | 101:12 | 100:5 | Insert snapshot |
| 4 | B delete 102 | 101:12 | 102:deleted | Retain logical delete marker |
| 5 | A update 103 = 15 | 103:15 | 102:deleted | Apply newer update |
| 6 | A update 101 = 12 again | 103:15 | 102:deleted | Reject older replay |
| 7 | B snapshot W = 5 again | 103:15 | 102:deleted | Reject stale resurrection |

An independent source-order replay gives A: 10 → 12 → 15 and B: 5 → deleted.
The same final state is A=15 and B absent from ordinary queries, with B's delete version retained internally.
The trace verifies this chosen sequence only; it does not prove a real connector's boundary semantics.
If the initial snapshot cannot be assigned a valid W version, this version comparison is not justified.
Use the source connector's documented algorithm instead, including any chunk collision handling.

### 5.2 Throughput, backlog, and recovery

All figures are ILLUSTRATIVE inputs, not measured workload or a service-level promise.
Let the source produce 20,000 events/s at an average 500 payload bytes/event.
Assume a single logical stream's event count matches the events being sized and ignore compression, headers, indexes, and protocol overhead.
Assume constant rates, no loss, no additional retries, an initially empty queue, and sufficient retained storage.
For the first 600 s, the destination durably processes 15,000 events/s.
Afterward its durable processing rate rises to 25,000 events/s while arrivals remain 20,000 events/s.

**Validity and derivation:** Arrival count equals rate times interval under constant-rate assumptions.
Payload bytes equal event count times mean payload bytes per event, with no overhead included.
Backlog increase equals arrival count minus completed count when the queue remains nonempty after starting at zero.
Drain time equals backlog divided by the strictly positive excess of recovery service over arrivals.

```text
FORMULA [CDC-1]: payload_rate = arrival_rate * bytes_per_event
WHERE: payload_rate [bytes/s]; arrival_rate [events/s]; bytes_per_event [bytes/event].
CONSTRAINT: mean payload and event rate are representative; excludes wire/storage overhead.
NUMERICAL EXAMPLE: 20,000 * 500 = 10,000,000 bytes/s = 10 MB/s (decimal).
INTERPRETATION: payload ingress alone is 10 MB/s before operational headroom.

FORMULA [CDC-2]: backlog = (arrival_rate - service_rate) * duration
WHERE: backlog [events]; rates [events/s]; duration [s].
CONSTRAINT: initially empty, constant rates, arrival_rate > service_rate, no loss or retry.
NUMERICAL EXAMPLE: (20,000 - 15,000) * 600 = 3,000,000 events.
INTERPRETATION: the ten-minute shortfall creates three million outstanding events.

FORMULA [CDC-3]: payload_backlog = backlog * bytes_per_event
WHERE: payload_backlog [bytes]; backlog [events]; bytes_per_event [bytes/event].
CONSTRAINT: same representative event size and payload-only accounting.
NUMERICAL EXAMPLE: 3,000,000 * 500 = 1,500,000,000 bytes = 1.5 GB.
INTERPRETATION: the buffered logical payload is 1.5 GB before overhead or replicas.

FORMULA [CDC-4]: drain_time = backlog / (recovery_service_rate - arrival_rate)
WHERE: drain_time [s]; backlog [events]; rates [events/s].
CONSTRAINT: constant recovery rates, service strictly greater than arrivals, no retry amplification.
NUMERICAL EXAMPLE: 3,000,000 / (25,000 - 20,000) = 600 s = 10 min.
INTERPRETATION: catching up requires ten additional minutes at the assumed surplus.
```

Independent check: 20,000 × 600 = 12,000,000 arrivals during overload.
Independent check: 15,000 × 600 = 9,000,000 durable completions in that interval.
Their difference is 3,000,000 events, agreeing with CDC-2.
During recovery, 25,000 × 600 = 15,000,000 completions and 20,000 × 600 = 12,000,000 new arrivals.
The difference again removes exactly 3,000,000 events.
A second unit check gives 10 MB/s × 600 s = 6,000 MB incoming payload during the overload.
The excess of 5,000 events/s × 500 bytes/event = 2.5 MB/s yields 1,500 MB in 600 s.
These checks use decimal MB and GB, not MiB or GiB.
At recovery service of 20,000 events/s, surplus is zero and the positive backlog never drains in this model.
At 22,000 events/s, the surplus is 2,000 events/s and drain time is 1,500 s, or 25 min.
At 19,000 events/s, backlog grows by 1,000 events/s rather than draining.
A hot partition can fail to drain despite a positive aggregate surplus across all partitions.
Capacity, queue age, broker overhead, replication, compression, retries, and connector snapshot traffic require separate measurement.

### 5.3 Daily payload and replication budget

Assume the same 20,000 events/s and 500 payload bytes/event for 86,400 s.
Using CDC-1, logical ingress is 10,000,000 bytes/s.
Multiplication by the number of seconds gives 864,000,000,000 bytes = 864 GB of logical payload per day.
For three complete local copies, a simplified payload-only footprint is 3 × 864 GB = 2,592 GB = 2.592 TB.
The independent check is 20,000 × 86,400 = 1,728,000,000 events/day; × 500 bytes gives the same 864 GB.
This assumes every copy contains the same uncompressed payload for a full day.
It excludes record metadata, segment/index overhead, filesystem slack, compression, compaction, transient re-replication, and tiered copies.
A replication factor of three is not a measured physical write or network multiplier for a specific cluster.
Actual producer ingress, inter-broker replication, consumer egress, and recovery traffic use different network paths.
At 250 bytes/event under the same count rate, payload ingress halves to 5 MB/s and daily logical payload to 432 GB.
For delete-heavy compaction, retained physical bytes cannot be derived from this daily payload number alone.

**Takeaway:** Size overload and recovery in the same units and separately budget source retention, broker storage, and network paths.

## 6. Visual / Geometric Intuition

### 6.1 Snapshot boundary versus delivery order

```text
source commit order:    [snapshot state at W] -- update 101 -- delete 102 -->
delivery possibility:  update 101 -- snapshot W -- delete 102
sink comparison:       accept 101 -- reject W --- accept 102
```

The upper line represents logical source history.
The lower line represents possible delivery to a materializer, not necessarily connector emission order.
An asynchronous consumer, retries, or multiple inputs can change observation order.
The boundary labels matter more than the wall-clock arrival order.
A stream event cannot safely be ordered against a snapshot row unless the comparison is defined by the protocol.

### 6.2 Snapshot and log coverage

```text
represented by snapshot:  <================ W
required stream suffix:                       W+ ================ >
                              exact inclusion is source-specific
```

There must be no hole between the state represented by the snapshot and the captured suffix.
An overlap is acceptable only when the protocol resolves duplicate or stale records correctly.
A gap is not repaired by idempotency: an event never delivered cannot be deduplicated into existence.
A stale snapshot is not repaired by capture completeness: the sink must resolve ordering too.

### 6.3 Partition progress versus transaction completion

```text
partition 0:  [T row A] [later record]
partition 1:  [T row B] [transaction-related control, if provided]
reader:       may see A before B unless publication is coordinated
```

Partition order says nothing by itself about when another partition's row becomes visible.
A transaction-aware materializer must know which records belong to T and when the complete set has arrived.
Metadata must be enabled and interpreted according to the connector/version's documented contract.
A timeout alone is not proof that the transaction is complete.

**Takeaway:** Draw source order, delivery order, and destination visibility separately when diagnosing a mismatch.

## 7. Comparison / Selection Table

| Choice | Benefit | Cost or risk | Use when | Avoid when | Evidence needed |
|---|---|---|---|---|---|
| Periodic full copy | Simple bounded batch | Stale data, heavy scans, cutover complexity | Batch freshness is adequate and a consistent copy is available | Low latency or continuous high write volume is mandatory | Snapshot duration and source impact |
| Quiesced copy then stream | Simple handoff reasoning | Write outage and operational coordination | A controlled freeze is permissible | Writes cannot be reliably stopped | Freeze enforcement and restart boundary |
| Continuous log CDC | Captures committed changes without polling every row | Connector, retention, schema and recovery complexity | Freshness justifies operational complexity | Source log cannot be retained or accessed safely | Connector and database version contracts |
| Unversioned blind upsert | Small implementation | Stale replay and delete resurrection | Only when a stronger upstream ordering and replay contract is proved | Multiple paths or historical replay can reorder | End-to-end ordering proof |
| Per-key versioned upsert | Duplicate/stale-safe current state | Metadata and comparator design | Source versions are valid and per-key consistency suffices | True cross-key atomicity is mandatory | Comparator, atomic sink write, retention tests |
| Transaction staging | Atomic group visibility can be implemented | Buffering, transaction completeness and timeout operations | Readers require source-transaction grouping | Boundaries or event counts are not available | Transaction metadata and sink publish semantics |
| Event ledger plus effect | Exact logical event recognition within retention | Ledger growth and atomicity requirements | A repeated event could trigger a non-idempotent effect | Ledger and effect cannot share a correctness protocol | Event identity and failure injection |
| New-generation rebuild | Clean isolation from suspect state | Extra storage and controlled cutover | Existing state cannot be repaired confidently | No consistent snapshot or durable change suffix exists | Reconciliation and reader cutover checks |

Resolve conflicting requirements by correctness first, then latency and cost.
If transaction-wide atomicity is required, key-level versioning alone is disqualified even if it is cheaper.
If no reliable boundary exists, operational convenience does not justify calling the result a consistent replica.

**Takeaway:** Choose the cheapest design that meets the actual visibility and recovery contract, not the strongest-sounding label.

## 8. Failure Modes & Edge Cases

Each entry is a diagnostic hypothesis, not a claim about a real incident.
Before intervention, preserve source identity, offsets, configurations, logs, timestamps, and the affected key set.
Do not reset offsets, delete a replication slot, or truncate sink state until the recovery path and data loss consequences are understood.

### F01 — A row reverts after bootstrap

**Symptom:** A destination row contains an older value after an initial copy.
**Competing causes:** Late snapshot record, stale update replay, wrong version comparator, or source rollback misunderstanding.
**Discriminator:** Trace this key's source commit sequence, snapshot boundary, event identities, destination versions, and delivery order.
**Expected signal:** A lower-version record was applied after a higher-version record if stale overwrite is the cause.
**Bounded mitigation:** Pause the affected sink path; preserve evidence and restore the key only from an authoritative consistent boundary.
**Risk:** An ad hoc manual update without fencing can race the live stream and be overwritten again.
**Prevention:** Enforce atomic version comparison and a documented snapshot overlap algorithm.
**Verification:** Replay update-before-snapshot and snapshot-before-update traces and reconcile the affected key at a common boundary.

### F02 — Deleted entity returns

**Symptom:** A source-deleted key reappears in the materialized view.
**Competing causes:** Lost delete event, tombstone misinterpretation, expired destination delete metadata, or old replay.
**Discriminator:** Compare the source delete position with the sink's retained version and subsequent records for the key.
**Expected signal:** A lower version is accepted after the logical delete if tombstone expiry caused resurrection.
**Bounded mitigation:** Stop stale replay, restore a versioned delete marker from trusted history, and repair the key.
**Risk:** Physical deletion of the sink row removes the only evidence needed to reject old records.
**Prevention:** Keep delete versions for the complete allowed replay horizon or rebuild in an isolated generation.
**Verification:** Deliver a pre-delete update after the delete and confirm the key remains logically absent.

### F03 — Missing changes after connector restart

**Symptom:** Some source changes have no corresponding destination state.
**Competing causes:** Expired source log, connector filtering, an offset committed too early, or lag not yet drained.
**Discriminator:** Compare source log availability, connector resume position, emitted record range, sink checkpoints, and filter settings.
**Expected signal:** A required coordinate older than the retained source log establishes an unrecoverable suffix from that position.
**Bounded mitigation:** Halt misleading downstream publication and preserve the existing offsets and retention evidence.
**Risk:** Forcing the connector to a newer position can convert uncertainty into a permanent silent gap.
**Prevention:** Alert on retention headroom and rehearse rebootstrap before the history is needed.
**Verification:** Bootstrap a new generation at a validated boundary and reconcile state before cutover.

### F04 — Duplicate side effects

**Symptom:** A single source mutation triggers multiple emails, charges, or counter increments.
**Competing causes:** Connector replay, sink retry after timeout, nonunique identity, or a genuinely repeated source transaction.
**Discriminator:** Compare stable event identities and source transaction context with external request identifiers.
**Expected signal:** Identical source event identity maps to multiple external effects if retry deduplication failed.
**Bounded mitigation:** Disable the unsafe side-effect path and assess compensation with its business owner.
**Risk:** Retrying an ambiguous payment without provider-side idempotency can create another charge.
**Prevention:** Atomically record event identity with the effect or use an external provider's supported idempotency contract.
**Verification:** Inject failure after the effect but before checkpoint and confirm only one business effect survives retry.

### F05 — Partial transaction visible

**Symptom:** Related rows disagree briefly even though the source updated them in one transaction.
**Competing causes:** Cross-partition publication, different destination commit times, or inconsistent reader snapshots.
**Discriminator:** Identify source transaction membership, all emitted events, destination commit times, and reader isolation.
**Expected signal:** One transaction member is visible while another remains in flight or uncommitted.
**Bounded mitigation:** Route critical readers away from the partial view or publish only a previously reconciled generation.
**Risk:** Buffering indefinitely for an incomplete transaction can exhaust memory or violate freshness targets.
**Prevention:** Stage durable transaction members and release them only when documented completeness conditions hold.
**Verification:** Delay one partition deliberately and confirm readers cannot observe the partial transaction.

### F06 — Connector cannot resume

**Symptom:** Capture stops after downtime with an unavailable source position.
**Competing causes:** Retention exhaustion, source lineage change, permission loss, or a corrupt connector checkpoint.
**Discriminator:** Check log/slot status, source epoch, saved offsets, and authentication errors independently.
**Expected signal:** The saved position lies outside the source's retained history if retention exhaustion is responsible.
**Bounded mitigation:** Stop attempts that discard offsets and protect remaining history while assessing a new snapshot.
**Risk:** Deleting a slot or resetting to the current position may destroy the evidence or omit changes.
**Prevention:** Capacity-plan retention for outage plus detection, restart, and catch-up with measured headroom.
**Verification:** Perform a documented rebootstrap and reconcile the new generation at a common source boundary.

### F07 — Backlog grows while fleet CPU is low

**Symptom:** Aggregate consumer CPU is low while age and lag rise.
**Competing causes:** One hot partition, destination throttling, connector pause, or slow serialization.
**Discriminator:** Compare per-partition arrival and durable completion rates, sink latency, and connector state.
**Expected signal:** A single partition's arrivals exceed its durable service even if fleet averages appear healthy.
**Bounded mitigation:** Reduce upstream load or repair the identified bottleneck without violating key ordering.
**Risk:** Adding consumers does not split a single partition and can trigger disruptive rebalances.
**Prevention:** Model key skew, partition capacity, and recovery surplus before production.
**Verification:** Demonstrate declining oldest-work age and positive per-partition recovery surplus over a representative interval.

### F08 — Events cannot be decoded after DDL

**Symptom:** Deserialization or destination writes fail after a schema change.
**Competing causes:** Incompatible schema evolution, stale schema history, permission changes, or invalid data conversion.
**Discriminator:** Match the failing source position, schema version, payload, destination contract, and connector logs.
**Expected signal:** Old records decode under their historical schema but fail under an incorrectly assumed latest schema.
**Bounded mitigation:** Pause unsafe progress and retain original bytes plus schema identifiers for recovery.
**Risk:** Skipping bad records can produce permanent divergence while dashboards show progressing offsets.
**Prevention:** Test compatible producer/consumer changes, preserve schema history, and define a controlled DDL rollout.
**Verification:** Replay both pre-change and post-change records into a disposable destination under the intended decoder.

### F09 — Position comparisons fail after failover

**Symptom:** Apparently smaller positions arrive after source promotion or connector migration.
**Competing causes:** Legitimate new lineage, position format change, incorrect parser, or wrong source identity.
**Discriminator:** Identify source history and vendor-supported failover continuity before comparing old and new coordinates.
**Expected signal:** Positions belong to incomparable histories if the promoted source lacks a documented mapping.
**Bounded mitigation:** Fence old producers and pause materialization at the uncertain lineage transition.
**Risk:** Treating a new epoch's numeric position as an old epoch's smaller value can drop valid changes.
**Prevention:** Include source lineage in the ordering contract and rehearse failover with the actual source configuration.
**Verification:** Reconcile across the switchover using a supported source-specific continuity or rebootstrap procedure.

### F10 — Checkpoint progresses but state is missing

**Symptom:** Consumer lag falls while destination rows are absent.
**Competing causes:** Offset committed before effect, asynchronous work not drained, wrong destination, or filtering.
**Discriminator:** Correlate contiguous completed offsets with durable sink commits and filter decisions.
**Expected signal:** A committed frontier beyond a failed effect demonstrates unsafe checkpointing.
**Bounded mitigation:** Pause commits and recover from a known safe input position if history remains available.
**Risk:** Blind rewind can repeat non-idempotent external effects.
**Prevention:** Couple the checkpoint with durable state or commit only after effects and implement duplicate safety.
**Verification:** Crash between effect and checkpoint, and between checkpoint and attempted effect, in a disposable environment.

### F11 — Snapshot never completes

**Symptom:** Bootstrap runtime exceeds the planned retention window.
**Competing causes:** Slow scan, database contention, continuously restarting chunks, sink backpressure, or snapshot lock waits.
**Discriminator:** Separate scanned rows/s, source read latency, connector queue pressure, and downstream service rate.
**Expected signal:** Snapshot scan stalls with source lock waits if database contention is the bottleneck.
**Bounded mitigation:** Reduce competing load or enlarge retention without invalidating the active consistent snapshot.
**Risk:** Restarting repeatedly can extend retention exposure and intensify source pressure.
**Prevention:** Measure snapshot throughput and log generation under representative load, then size for worst credible interruption.
**Verification:** Complete a representative bootstrap and prove its captured suffix remains available through catch-up.

### F12 — Reconciliation reports inconsistent counts

**Symptom:** Source and destination counts differ during active writes.
**Competing causes:** Different observation boundaries, filtering, real missing rows, or soft-delete treatment.
**Discriminator:** Compare both datasets at a common defined boundary with identical filters and delete semantics.
**Expected signal:** Counts match at a common consistent boundary if the earlier discrepancy was purely timing.
**Bounded mitigation:** Treat the count as an investigation signal, not proof of data loss; sample mismatched keys.
**Risk:** Repairing against a moving source snapshot can overwrite newer destination state.
**Prevention:** Define boundary-aware counts, partitioned hashes, and targeted key comparisons.
**Verification:** Independently recompute selected key values and hashes at the same source position or isolated generation.

### F13 — Primary key changes

**Symptom:** Both the old and new identities appear, or neither does temporarily.
**Competing causes:** Delete/create handling, routing across partitions, missing transaction grouping, or an incorrect key serializer.
**Discriminator:** Inspect the actual connector's events for the key change and compare old/new partition and transaction metadata.
**Expected signal:** A create-new event arrives before delete-old if the two identities are independently processed.
**Bounded mitigation:** Protect consumers requiring uniqueness and repair the identities at a consistent boundary.
**Risk:** Assuming a primary-key change is an in-place update can strand the old row.
**Prevention:** Treat actual connector output as authoritative and explicitly define atomic identity-move visibility.
**Verification:** Inject retries and reordering around an actual key change in a version-matched test environment.

### F14 — Log retention fills source disk

**Symptom:** Source log or replication-slot storage grows during connector outage.
**Competing causes:** Unconsumed retained log, another slow subscriber, unusually high source write volume, or database housekeeping.
**Discriminator:** Inspect the retained position per consumer and the source's disk and log generation metrics.
**Expected signal:** A stationary consumer position combined with advancing source head increases retained history.
**Bounded mitigation:** Protect database availability, notify the source owner, and recover or safely rebootstrap the consumer.
**Risk:** Removing retention can restore disk while making historical recovery impossible.
**Prevention:** Budget outage duration, log generation bursts, disk headroom, and escalation thresholds.
**Verification:** Confirm source disk headroom increases while a correct consumer advances through the required suffix.

### F15 — Repeated events have conflicting payloads

**Symptom:** Identical event identifiers decode to different logical changes.
**Competing causes:** Incomplete identity, source epoch collision, payload corruption, or nondeterministic transformation.
**Discriminator:** Compare original bytes, source identity, coordinate components, schema version, and transformation version.
**Expected signal:** Two different records sharing an identity after failover indicate insufficient lineage scoping.
**Bounded mitigation:** Quarantine both conflicting versions and stop affected-key checkpoint progress.
**Risk:** Silently choosing one makes later audit and deterministic replay impossible.
**Prevention:** Specify identity construction and immutability, including within-position sequence and epoch.
**Verification:** Replay the same source records through deterministic decoding and confirm unique stable identities.

### Operational decision drills

The drills below are illustrative procedures for design reviews and controlled tests.
They are not evidence that any named connector has been exercised.
The source database, connector, broker, and destination owners must agree on actual commands and permission boundaries.
Each drill distinguishes a useful observation from a conclusion it cannot establish by itself.

#### D01 — Establish snapshot scope
Identify the exact database and server lineage.
List every table required by the materialized view.
Record whether the snapshot covers those tables under one source transaction.
Document any excluded table or row predicate.
Record the snapshot isolation contract from the database documentation.
Identify schema changes allowed while the scan runs.
Determine whether reads share one consistency point or use separate chunk boundaries.
Treat an unknown cross-table guarantee as a blocker to claiming cross-table consistency.
Capture the boundary representation in the bootstrap manifest.
Have the source owner approve the operational impact of the scan.

#### D02 — Define source lineage
Name the database cluster and source instance independently of connector name.
Document how failover changes the source identifier.
Document whether position continuity is supported across promotion.
Identify which lineage component travels with each emitted event.
Define a policy for records from an unexpected former primary.
Fence the former writer before accepting the promoted writer's history.
Avoid comparing positions from two histories without a source-provided mapping.
Store the lineage in checkpoints and repair manifests.
Exercise the actual failover semantics before promising automatic continuity.
Escalate an incomparable position instead of inventing a numeric ordering.

#### D03 — Record the handoff contract
Identify how the snapshot's boundary is acquired.
Record whether events at that boundary are included in the snapshot.
Record whether replay starts at or after the boundary.
Explain how the design resolves an intentional overlap.
Explain why no interval is omitted.
Document any lock or transaction required to bind snapshot and log.
Record the maximum expected time before the stream catches up.
Keep the source log long enough for that interval and realistic interruptions.
Write the contract in a manifest that survives connector restarts.
Revalidate the contract after changing connector mode or database version.

#### D04 — Inspect snapshot event metadata
Collect a representative snapshot event in a safe test environment.
Inspect its operation type and snapshot-specific fields.
Compare its source metadata with a streamed update for the same key.
Check whether the snapshot position is meaningful at row granularity.
Determine whether the connector exposes an actual comparable watermark.
If not, do not fabricate a row version from event arrival time.
Consult the documented overlap and collision-resolution mechanism.
Trace a row modified while the snapshot is in progress.
Record the expected final state and the observed result separately.
Repeat after changes to snapshot mode or transformation configuration.

#### D05 — Trace an update during scanning
Choose a key that exists before the snapshot begins.
Read its pre-update value from an authoritative source transaction.
Update that key after the snapshot boundary under a controlled transaction.
Record the source transaction's commit position if available.
Allow snapshot and stream records to reach the destination.
Observe whether the snapshot record arrives before or after the update.
Apply the declared overlap protocol rather than an assumed ordering.
Compare the final value with the source at the same boundary.
Repeat with a second update after the first one.
An apparently correct final value does not prove that intermediate visibility was atomic.

#### D06 — Trace an insert during scanning
Choose a primary key absent from the snapshot's consistent view.
Insert the row after the source snapshot boundary.
Record the commit position and the complete after-image.
Check that the post-boundary stream contains the insertion.
Check whether any snapshot chunk also emits the row under the selected mode.
Resolve any overlap with the connector's documented chunk protocol.
Materialize the event using its correct source identity.
Verify one visible row in the destination at a comparable boundary.
Do not infer missing capture from a source read at a later time alone.
Repeat with the insert near the end of a long scan.

#### D07 — Trace a delete during scanning
Choose a key present in the initial consistent view.
Delete it after the snapshot boundary.
Capture the source commit coordinate and emitted deletion representation.
Allow a late snapshot record to reach the destination after the delete.
Require the destination to retain sufficient deletion ordering metadata.
Verify that the late row cannot restore the deleted entity.
Check both the delete envelope and any null-value tombstone handling.
Treat a physically absent sink row and an absent deletion guard as different facts.
Repeat the test after restarting the sink before offset commit.
Keep the test's replay horizon within the declared metadata retention policy.

#### D08 — Handle an initially empty table
Capture a consistent boundary even if the table has no rows.
Do not infer that an empty snapshot means no log suffix is required.
Insert a row immediately after the boundary.
Confirm that the row is represented by the change stream.
Record the stream starting condition for a zero-row scan.
Confirm that completion is represented in bootstrap bookkeeping.
Ensure empty chunks do not cause skipped stream subscription.
Check that a table added later has an explicit onboarding procedure.
Distinguish an empty source table from a failed source query.
Require an observable successful scan state before publication.

#### D09 — Handle an update to a nullable field
Determine whether the event carries a full after-image or a partial update.
Record how the source distinguishes an unchanged field from an explicit null.
Ensure the serializer preserves that distinction.
Apply an explicit null without deleting the whole entity.
Preserve an omitted field only when the connector contract says it is unchanged.
Do not treat a null field as a Kafka null-value tombstone.
Inspect schema evolution when nullability changes.
Check the sink's own default-value behavior on insert and update.
Replay the same event after a restart and compare the resulting value.
Quarantine a payload whose missing-field semantics cannot be determined.

#### D10 — Validate before-image assumptions
List every consumer that requires a previous value.
Determine whether the source log actually records that value.
Check database and connector settings for their effect on before images.
Treat unavailable before-image columns as unknown, not null by default.
Avoid computing inverse changes from a partial previous image.
For a current-state sink, prefer a valid after-image when available.
For an audit or aggregate sink, define how missing previous values are recovered.
Inspect update and delete events separately because their images can differ.
Test with large or unchanged columns under the actual source configuration.
Record any data minimization implications of capturing historical values.

#### D11 — Compare two database positions
Identify the precise position type and source lineage.
Read the documented ordering rule rather than sorting position strings casually.
Check whether the value denotes a transaction, log record, or emitted row.
Determine whether multiple events can share the same coordinate.
Identify a within-position sequence where the source provides one.
Reject comparisons across unsupported epochs.
Verify that the comparison handles the source's numeric or structured format.
Avoid using a wall-clock timestamp as an undocumented replacement.
Persist the original coordinate alongside any normalized representation.
Test comparator boundaries using examples from the source's documentation.

#### D12 — Design an event identity
Start with the source identity and lineage needed to avoid cross-history collisions.
Include table or collection identity when positions can repeat across tables.
Include the source coordinate at its documented granularity.
Include a row or event ordinal when several events share that coordinate.
Distinguish snapshot identities from streamed mutations when their semantics differ.
Define behavior for changes that affect more than one primary key.
Ensure serialization of the identity is deterministic.
Do not include volatile connector process identifiers unless they are part of source identity.
Persist enough original metadata to investigate a collision.
Prove uniqueness for the deployed connector contract before using the identity for money-moving effects.

#### D13 — Validate entity-key serialization
Specify the logical source primary key columns and types.
Define canonical encoding, including null and composite-key handling.
Check whether the sink treats equivalent strings or numeric encodings as equal.
Avoid locale-sensitive transformations for identifiers.
Distinguish integer 1 from string 1 if the schema requires that distinction.
Verify that all events for the same entity use the same key serialization.
Account for connector schema evolution that changes key types.
Confirm whether changing key encoding changes partition placement.
Plan an explicit migration rather than silently changing hash inputs.
Test collisions and round trips with boundary values.

#### D14 — Investigate partition migration
Record the old partition count and the producer's partitioning function.
Record the new partition count and any changed key serializer.
Identify whether a key can map to different partitions after the change.
Assume no cross-partition broker ordering unless a stronger protocol is established.
Check whether old and new consumers can process the same key concurrently.
Fence prior workers or apply a valid source version at the sink.
Drain or explicitly coordinate the old routing epoch.
Preserve a rollback path that does not reintroduce stale events.
Trace one key across the migration boundary.
Validate both current-state convergence and any stronger visibility requirement.

#### D15 — Separate replay identity from order
Select an event and record its source event identity.
Deliver the same event twice under the same immutable identity.
Verify that the second delivery does not repeat the logical effect.
Now deliver a different event with an older source version.
Verify that the older event does not replace the newer state.
These are two different tests even if both end in a no-op.
A dedup ledger may recognize the duplicate without ordering the older event.
A version guard may reject the older event without proving event identity uniqueness.
Audit collisions when an equal version has a different payload.
Keep both contracts explicit in the sink design.

#### D16 — Define source transaction membership
Identify how a transaction begins and commits in the source.
Determine what transaction identifier appears on data events.
Find whether a connector exposes transaction begin and end metadata.
Check whether such metadata is enabled for the deployed connector.
Determine whether an event count or equivalent completeness signal is provided.
Account for filters that alter the subset reaching the destination.
Identify tables that are part of a transaction but absent from capture.
Do not assume event arrival at an end marker implies all partitions have drained.
Specify the exact reader-visible atomicity requirement.
If the completeness contract is insufficient, document the weaker guarantee.

#### D17 — Stage a cross-key transaction
Assign each staged record to a source transaction identity.
Persist staged records before acknowledging their input offsets.
Track the expected set or count only if the connector provides a trustworthy contract.
Handle records arriving before their transaction metadata.
Handle completion metadata arriving before some data partitions catch up.
Deduplicate repeated members before counting transaction completion.
Prevent readers from seeing staged data before publication.
Commit the complete state change using an actual destination atomic boundary.
Preserve staged state across worker failures and ownership changes.
Quarantine transactions whose completeness cannot be established.

#### D18 — Handle a large transaction
Measure or estimate the largest transaction's event count and payload size.
Include staged metadata, indexes, and copies in the storage budget.
Avoid assuming that an average transaction represents the maximum.
Apply a documented limit before a transaction can exhaust the staging store.
Choose whether exceeding the limit pauses, spills to durable storage, or escalates.
Do not silently split visibility if readers require source transaction atomicity.
Measure time spent waiting for the final member and publication.
Alert on the oldest unfinished transaction, not only aggregate input lag.
Ensure abort and cleanup cannot remove a still-live transaction's data.
Test restart while a large transaction is partially staged.

#### D19 — Handle a rolled-back source transaction
Confirm whether the connector emits only committed changes under the chosen source mode.
If uncommitted data can appear, identify the abort representation and buffering rule.
Never publish a staged transaction merely because its first row arrived.
Record transaction termination semantics for rollback and connector restart.
Ensure rollback cleanup is durable and idempotent.
Verify that a transaction identifier is scoped against reuse.
Test an insert followed by rollback in a disposable source database.
Confirm the destination has no committed visible row afterward.
Inspect whether diagnostic logs distinguish abort from missing completion.
Do not generalize one connector's rollback behavior to every source.

#### D20 — Coordinate input checkpointing
For each partition, define the next input offset eligible for commit.
Track completion of all earlier records, including async work.
Treat the earliest unresolved record as a frontier barrier.
Persist state effects before advancing a separate checkpoint.
If effect and checkpoint share one transaction, document the actual atomicity domain.
Do not equate a client acknowledgement with durable destination commit.
Ensure offset commit errors are observable and do not discard in-memory work.
Revoke ownership only after the old worker is fenced from further writes.
After restart, expect replay from the last durable safe frontier.
Confirm duplicate handling covers all effects between that frontier and the crash.

#### D21 — Crash before destination commit
Begin with a source event whose destination effect has not committed.
Terminate the worker before its destination transaction commits.
Ensure the transaction rolls back or is otherwise not externally visible.
Restart from the last durable input checkpoint.
Replay the event and apply its effect once.
Check that no partial dedup marker survived without the effect.
Inspect how the sink resolves an interrupted connection.
Use the actual destination isolation and durability settings in the test.
Distinguish a rollback from a commit whose acknowledgement was lost.
Record uncertainty until the destination transaction outcome is known.

#### D22 — Crash after destination commit
Commit the destination mutation but stop the worker before checkpoint acknowledgement.
Restart from the earlier committed input frontier.
Expect the same source event to be delivered again.
Confirm the event identity or version guard makes the replay harmless.
Check that external effects use their own durable idempotency contract.
Verify that the checkpoint advances after the replay is safely resolved.
Confirm the destination payload and version are unchanged by the duplicate.
Inspect metrics for replay and deduplication rather than treating them as errors automatically.
Test a lost commit response as well as an intentional crash.
Never assume an ambiguous client timeout implies destination rollback.

#### D23 — Handle asynchronous gaps
Receive offsets 10, 11, and 12 from the same partition.
Suppose work for 10 and 12 finishes while 11 remains unresolved.
The durable contiguous frontier can advance past 10 but not past 11.
Completion of 12 alone does not justify committing the next offset 13.
Retain enough bookkeeping to resume safely after a crash.
Bound the number of outstanding records so the completion map cannot grow without limit.
Pause that partition when an unresolved failure blocks too much work.
Preserve ordering for effects whose semantics require it.
Escalate a permanently failing event rather than silently skipping it.
Resume frontier advancement only after the gap is durably resolved.

#### D24 — Fence a rebalance
Identify the old owner and the new owner of the partition.
Stop the old owner from initiating additional external effects.
Wait for or cancel in-flight operations according to the destination contract.
Use a generation token or equivalent fencing when the destination supports it.
Reject writes from an owner whose generation is no longer current.
Commit only a safe contiguous frontier from completed effects.
Allow the new owner to replay from the last durable frontier.
Deduplicate any work committed before the ownership transition.
Monitor rebalance duration and abandoned in-flight work.
Test a delayed old-owner write after the new owner has started.

#### D25 — Size a deletion guard
Define the earliest replay position permitted into the live materialization.
Estimate how long source, broker, and backup workflows can retain old events.
Include manual repair and disaster recovery in the replay policy.
Keep the delete version at least as long as the oldest permissible replay can arrive.
If replay is unbounded, a finite time-to-live alone is not sufficient.
Use a separate durable high-watermark or a clean generation where appropriate.
Ensure compaction of a broker topic is not mistaken for sink deletion protection.
Verify that restore procedures also restore the ordering metadata.
Test stale pre-delete updates just before and after the policy boundary.
Document who authorizes replay beyond the protected interval.

#### D26 — Recover an expired delete guard
Identify the earliest event position that may still be replayed.
Identify which deleted keys have lost their destination version metadata.
Stop replay into the live generation if stale records remain possible.
Do not reconstruct deletion order from absent rows alone.
Recover deletion metadata from an authoritative retained history if available.
Otherwise consider a new consistent snapshot and isolated destination generation.
Avoid exposing the new generation until its suffix has caught up.
Reconcile key presence and value at a compatible source boundary.
Record any unrecoverable interval as an explicit evidence gap.
Resume normal replay only after its horizon is covered by the new policy.

#### D27 — Diagnose compacted-topic rebuild limits
List the broker retention and compaction policies for each required topic.
Determine whether the topic contains complete current-key state or partial change records.
Check whether delete tombstones are retained long enough for the planned rebuild.
Identify keys whose latest value might no longer be reconstructible from retained records.
Account for schema history needed to decode older retained values.
Do not assume compaction preserves every historical mutation.
Do not assume an empty topic read proves an empty source table.
Compare the intended rebuild duration with the applicable retention windows.
Use a source snapshot if the topic cannot prove complete current state.
Validate a full rebuild in an isolated environment before relying on it for recovery.

#### D28 — Protect source log retention
Measure source log generation in bytes per unit time.
Estimate the outage plus detection, response, and catch-up interval.
Account for write bursts rather than relying only on a daily average.
Inspect the minimum retained position across all relevant consumers.
Distinguish configured retention from actual free disk available to enforce it.
Alert before the required resume position approaches the oldest retained position.
Coordinate retention changes with the source database owner.
Do not delete a retention mechanism just to silence a disk alert.
Prepare a rebootstrap plan for a history gap.
Recheck both source health and recovery completeness after mitigation.

#### D29 — Protect broker replay retention
Identify the earliest Kafka offset needed by each consumer group.
Compare it with each partition's current log start offset.
Track retention in both time and bytes when both policies apply.
Account for changes in partition traffic and record size.
Do not rely on total topic size to prove every partition's history remains present.
Check whether compaction changes the records available for the intended replay.
Distinguish broker retention from database log retention.
Monitor the oldest outstanding work and the actual available history.
Preserve offset evidence before adjusting a consumer group.
Rebuild from a valid source boundary when a required broker interval is gone.

#### D30 — Reconcile current-state keys
Define the source and destination key domains and filters.
Choose a source boundary that the destination has fully processed.
Read the source using a consistency mechanism valid for that boundary.
Exclude logical deletion markers from normal visible-row comparisons.
Compare partitioned counts to localize large discrepancies.
Compare key sets when counts match but identities may differ.
Compare canonicalized values for keys present in both systems.
Treat a hash match as evidence subject to the hash and coverage assumptions.
Inspect actual rows for selected mismatches.
Repair only after accounting for changes newer than the comparison boundary.

#### D31 — Reconcile transaction visibility
Choose a transaction that affects at least two independently routed keys.
Record its source commit and complete captured member set.
Delay one member in a controlled downstream test.
Observe the reader-facing state while the other member is ready.
If atomic visibility is required, no partial transaction should be exposed.
Release the delayed member and allow publication.
Confirm that all intended members become visible under one destination boundary.
Restart during staging and repeat the observation.
Check that duplicated members do not trigger early completion.
Record actual reader isolation as part of the result.

#### D32 — Plan a clean generation rebuild
Allocate a new isolated destination namespace or generation.
Capture a source-consistent snapshot and its log handoff contract.
Materialize all rows and subsequent changes into the new generation.
Keep the existing generation available only under its known limitations.
Track both snapshot progress and the stream catch-up frontier.
Compare the new state against a source view at a compatible boundary.
Validate deletion and transaction semantics for the new generation.
Switch readers using a controlled atomic pointer or equivalent routing change.
Retain a rollback strategy consistent with subsequent writes.
Retire the old generation only after retention and audit policies permit it.

#### D33 — Diagnose a stalled snapshot chunk
Identify the chunk key range and the last successfully completed chunk.
Check source query latency and any lock or transaction waits.
Check connector memory, queue capacity, and output backpressure.
Determine whether retrying the chunk preserves the same snapshot semantics.
Avoid restarting a snapshot blindly if doing so changes the handoff boundary.
Check whether concurrent changes in that key range are buffered or reconciled.
Measure remaining source log retention while the chunk is stalled.
Escalate source impact before increasing scan parallelism.
Confirm chunk completion and the resulting change coverage.
Record which source-specific collision protocol was actually used.

#### D34 — Handle schema addition
Record the source DDL and its effective source position.
Determine whether the new column has a default, null value, or backfill.
Check whether old and new records carry schema identifiers.
Ensure downstream decoders can still interpret retained old records.
Choose destination behavior for records emitted before the field existed.
Avoid silently assigning a business value to an unknown historical field.
Deploy compatible readers before enabling incompatible writers where applicable.
Test snapshot and streamed records across the schema boundary.
Verify that a connector restart can recover required schema history.
Reconcile representative rows after backfill has reached the chosen boundary.

#### D35 — Handle destructive schema change
Identify the consumers that still read the column or type being changed.
Retain old schema information for all records that can be replayed.
Plan a staged migration rather than removing the field without a compatibility window.
Test deserialization of historical and new data independently.
Determine how destination indexes and constraints are migrated.
Pause or quarantine unsupported records rather than skipping them invisibly.
Record a rollback plan for both source schema and downstream application.
Check whether snapshot schema and log-event schema can differ during the transition.
Reconcile values before dropping obsolete destination state.
Declare the replay boundary after which historical decoding support may be retired.

#### D36 — Handle a source failover
Record the source role and current lineage before promotion.
Fence the old primary to prevent simultaneous accepted histories.
Follow the database's supported promotion and log continuity procedure.
Check whether the connector's saved position exists on the new source.
Verify whether source transaction and ordering metadata remain comparable.
Pause the sink if the relation between old and new histories is unknown.
Preserve source and connector checkpoints before any reset.
Resume only from a supported continuity mapping or a new bootstrap.
Compare representative keys and transaction outcomes across the boundary.
Document any accepted recovery point objective and resulting data loss.

#### D37 — Diagnose a corrupt destination value
Capture the key, source position, destination version, and serialized record.
Read the source value at a compatible boundary if possible.
Compare raw event bytes with the decoded application object.
Inspect schema identifier and transformation version.
Check destination type coercion, rounding, timezone, and null behavior.
Determine whether the fault affects a key, a schema version, or an entire partition.
Do not overwrite with a moving source value without guarding against newer events.
Correct the transformation and replay or rebuild within the protected horizon.
Verify corrected examples on both sides of the schema boundary.
Document whether historical derived outputs also require repair.

#### D38 — Handle invalid or poison events
Retain the original event bytes, source identity, and decode error.
Determine whether failure is transient, schema-related, or a genuine data constraint violation.
Keep the unresolved event visible in operational state.
Block the contiguous partition frontier when skipping would violate completeness.
Use quarantine only with an explicit policy for later repair and ordering.
Prevent a later update from hiding an unresolved required transaction member.
Bound retries to avoid overloading the source or destination.
Fix the schema or transform before reprocessing the retained event.
Verify that the corrected event and its successors are durably applied.
Escalate any intentional skip as a data-quality exception, not a success.

#### D39 — Measure duplicate rate responsibly
Define the numerator as repeated deliveries of an already observed logical event.
Define the denominator as all deliveries within the same window and population.
Specify whether snapshot overlap events are counted as duplicates.
Do not use the number of stale distinct events as a duplicate count.
Distinguish connector re-emission from consumer retry where metadata permits.
Break down by source, partition, deployment, and restart interval.
Watch for identity collisions that would inflate the apparent duplicate rate.
Correlate with checkpoint failures and destination timeouts.
A nonzero rate can be expected under at-least-once recovery.
Investigate a sudden change against a documented baseline rather than a universal threshold.

#### D40 — Measure freshness responsibly
Specify the entity population and which source timestamp is used.
Distinguish source commit time from event creation time and sink processing time.
Check clock synchronization before subtracting timestamps across machines.
Report percentile and worst observed age as well as a mean when appropriate.
Include stalled keys or partitions, not only events that completed successfully.
Separate source capture lag, broker backlog, and destination commit lag.
Record the measurement interval and sampling limitations.
Avoid calling a Kafka offset distance a duration without a justified conversion.
Compare freshness to an explicit service objective and guardrails.
Use reconciliation to detect missing changes that a freshness chart cannot see.

#### D41 — Estimate outage budget
Specify the usable source-log capacity in bytes after safety reserves.
Measure or conservatively estimate source-log generation bytes per second.
State that a constant-rate division is only an approximate planning model.
Derive the time budget by dividing usable bytes by generation rate.
Subtract detection, escalation, restart, and catch-up time from the available window.
Include burst growth and other consumers' retention requirements.
Do not use Kafka payload bytes as a proxy for source log bytes without measurement.
Check disk constraints and source retention configuration independently.
Alert well before the calculated boundary is reached.
Recalculate after workload growth or changes to captured tables.

#### D42 — Estimate snapshot completion risk
Measure the amount of source data in the snapshot's actual projection.
Measure sustainable scan throughput under an acceptable source load.
Account for concurrent changes that accumulate while scanning.
Estimate the scan interval under stated throughput assumptions.
Add handoff and catch-up time before assessing log retention.
Include restart and failure scenarios rather than only the best run.
Track source read latency and production database guardrails.
Confirm the chosen snapshot method preserves a valid consistency view for the full interval.
If the estimate exceeds retention, change the plan before starting.
Recheck after changes in source size, throughput, or write rate.

#### D43 — Plan a replay
State the exact source or Kafka position range to replay.
Identify all affected partitions and source lineages.
Confirm the necessary records and schemas remain retained.
Check that destination versions and delete guards cover the replay horizon.
Inventory downstream non-idempotent effects before allowing duplicates.
Decide whether to target the live generation or an isolated rebuild.
Limit replay rate so recovery does not overwhelm current traffic.
Record the operator, reason, intended end boundary, and rollback procedure.
Monitor both replay progress and current-work freshness.
Reconcile the affected keys after reaching the declared boundary.

#### D44 — Restrict access to captured data
Inventory source columns and their sensitivity before enabling capture.
Grant the connector only the required source and metadata privileges.
Restrict broker topic access to authorized principals.
Encrypt in transit and at rest according to the applicable system policy.
Account for before images and historical events in privacy reviews.
Avoid placing sensitive full payloads in operational error logs.
Define retention and deletion rules for source, topic, sink, and backups separately.
Review how derived materializations propagate source deletions.
Audit access to replay tooling and destination repair commands.
Verify actual security settings; this sheet does not certify compliance.

#### D45 — Assess clock disagreement
Identify which component assigned each timestamp.
Check whether the timestamp reflects event creation, commit, capture, or processing.
Compare clock synchronization health on the relevant hosts.
Prefer source-supported ordering coordinates over wall-clock time for correctness.
Use timestamps for latency only when their semantics and clock uncertainty are known.
Do not infer a causal inversion solely from a negative measured latency.
Record time zones and precision when comparing logs.
Account for timestamp truncation that gives multiple changes the same value.
Use monotonic timers for elapsed durations within one process when available.
Escalate persistent clock anomalies separately from event-order violations.

#### D46 — Control multi-writer ambiguity
Identify whether multiple independent databases can mutate the same logical entity.
Define which source is authoritative for each key or field.
Do not compare unrelated database log positions as if they shared a total order.
If conflict resolution is required, specify its business semantics explicitly.
Record how writer identity and source lineage enter the event identity.
Determine whether last-writer-wins timestamps are acceptable under clock uncertainty.
Consider routing each entity to a single authority instead of merging concurrent writes.
Keep conflict decisions auditable and deterministic.
Test concurrent conflicting updates and delayed delivery.
Treat this sheet's single-source per-key invariant as insufficient without an added merge contract.

#### D47 — Interpret a healthy checkpoint
Record the connector checkpoint and the source position it represents.
Record the broker partition offsets and the consumer group's committed offsets.
Record the destination's durable frontier for each input partition.
Check whether the connector has emitted all records covered by its progress claim.
Check whether sink effects actually committed before their input frontier advanced.
Inspect quarantined records and any intentional filters.
Compare source head with capture position to measure source-side lag.
Compare broker and sink boundaries to measure downstream progress.
Use a compatible-boundary reconciliation for state correctness.
A green offset dashboard alone is not an equality proof.

#### D48 — Decide whether to publish
Require a documented snapshot consistency and suffix coverage contract.
Require a valid ordering and event identity definition.
Require a tested duplicate and stale-event strategy.
Require delete handling that survives the permitted replay horizon.
Require an explicit answer on cross-key transaction visibility.
Require source and broker retention sufficient for the recovery plan.
Require observable checkpoints, backlog, and quarantine state.
Require a boundary-aware reconciliation plan and accountable owners.
Record unresolved version-specific semantics as blockers to a production guarantee.
An unreviewed teaching draft does not substitute for these deployment decisions.

**Takeaway:** Diagnose the precise boundary and failure interval before changing offsets or state.

## 9. Complexity & Cost Analysis

Let N be the number of destination keys, E the number of change events, and T the number of events held for an unfinished transaction.
With a hash-indexed sink, an expected constant-time key lookup and update per event is possible under the sink's implementation assumptions.
Across E events this is expected O(E) application work, excluding storage-engine, network, index, and contention costs.
A balanced-tree indexed destination typically has O(log N) lookup/update work under its usual model.
Actual storage latency and throughput are NOT MEASURED here; neither complexity bound is a benchmark.
A current-state destination stores O(N) entity records plus retained delete and version metadata.
An event deduplication ledger grows with distinct event identities retained, not merely the number of keys.
A transaction staging buffer needs O(T) space for the buffered events, plus metadata and durability overhead.
An unbounded transaction or missing completion signal therefore needs a declared resource and incident policy.
Partitioned transaction assembly may add network shuffles and skew around large transactions.
A full bootstrap must read the chosen source data and emit or transfer its materialized representation.
Source log retention during bootstrap depends on elapsed snapshot plus handoff and catch-up time, not snapshot duration alone.
Provision for the maximum credible source byte generation during the entire recoverability window, with separately measured overhead and headroom.
The source database and Kafka may retain different histories and can fail independently.
A replay may increase both read load and destination write load while new arrivals continue.
The CDC-4 drain model applies only when the actual bottleneck has a positive sustained surplus.
If one partition's service rate is below its arrival rate, excess capacity elsewhere does not automatically drain it.
A replicated Kafka topic's logical payload footprint is not equal to total infrastructure cost.
Budget broker replication traffic, source log disk, Kafka disk, destination indexes, snapshot reads, and cross-zone egress separately.

**Takeaway:** Model cost at every durability boundary and evaluate recovery at the bottleneck, not just the aggregate.

## 10. Key Theorems / Results Reference

### R1 — Monotone-version invariant

If each incoming version is comparable within a key's lineage and the destination atomically installs only a strictly larger version, stored versions never decrease.
The inductive justification is in Section 4.1.
This is a design result under the given assumptions, not a database vendor guarantee.
It does not imply that every source event was captured or that an event's payload was interpreted correctly.

### R2 — Duplicate effect safety has a transaction boundary

An effect repeated after a crash is safe only if the effect is idempotent or the event identity and effect are coordinated by a correct durable protocol.
Recording a deduplication marker before an independent effect may lose the effect on crash.
Recording the marker afterward may duplicate the effect on crash.
A single transactional boundary or an external idempotency contract resolves the corresponding ambiguity; neither should be assumed without evidence.

### R3 — Partition order is not transaction-wide visibility

Per-partition ordering cannot by itself establish simultaneous visibility of records placed in different partitions.
Transaction metadata, complete membership, durable staging, and an atomic publication boundary are additional requirements when that visibility is needed.
If downstream readers accept eventual per-key convergence, those costs may not be necessary.

### R4 — Progress does not prove equality

A connector checkpoint or Kafka consumer offset demonstrates a progress claim in its own coordinate system.
It does not prove that all source rows were captured, correctly decoded, and durably reflected in a sink.
Reconciliation at compatible boundaries supplies independent evidence of state agreement, subject to its own coverage and collision limitations.

### Source and review ledger

| Source | Claim scope | Version/date | Limitation and review trigger |
|---|---|---|---|
| [Debezium documentation](https://debezium.io/documentation/) | Connector-specific snapshot, event, and transaction behavior | Deployment version not supplied | OPEN: confirm exact connector, database, settings, and documentation version before use |
| [Apache Kafka documentation](https://kafka.apache.org/documentation/) | Broker/client partition, retention, and transactional features | Deployment version not supplied | OPEN: verify precise version, client settings, and isolation semantics |
| [Course source brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#kafka) | Topic mapping only | Course planning inventory | Source case content unavailable; not an empirical architecture record |
| This sheet | ILLUSTRATIVE traces and DERIVED arithmetic | Draft 2026-11-07 | No runtime experiment, benchmark, or substantive review performed |

**Takeaway:** Treat correctness arguments as conditional proofs and version-specific implementation behavior as evidence to obtain.

## 11. Interview / Exam Questions

The following fictional questions are training exercises; no company attribution is implied.
All answers assume the contracts stated locally and are not evidence of an exercised deployment.
Scoring convention: 0 incorrect; 1 partial; 2 correct with assumptions; 3 defended with a boundary or tradeoff.

### Level 1 — Definition / Recall

**Q1 — What is a consistent snapshot?**
**Answer:** A database-supported view of rows at a defined consistency point or transaction snapshot, rather than an arbitrary scan across changing states.
A table scan that sees one row before a transaction and another row after it can violate this property.
**Alternative:** A quiesced copy can be consistent if writes truly remain frozen during its defined boundary.
**Follow-up:** Can two independently captured tables be treated as one consistent database snapshot? Only if the source's snapshot mechanism guarantees the shared view.
**Trap:** Equating scan start time with a database consistency point.
**Source:** Sections 3 and 4.

**Q2 — How does a source position differ from a Kafka offset?**
**Answer:** A source position identifies progress in a database history; a Kafka offset identifies a location within one topic partition.
Neither is generally numerically comparable to the other, and a Kafka offset does not by itself identify the source transaction.
**Alternative:** A record may carry both coordinates, but that is an explicit mapping, not identity.
**Follow-up:** What must a failover position include? The source-specific lineage or continuity information needed for valid comparison.
**Trap:** Selecting the largest numeric offset across different partitions as a global watermark.
**Source:** Sections 2–3.

**Q3 — Distinguish a delete event from a Kafka tombstone.**
**Answer:** A delete event describes a source deletion; a keyed null-value Kafka tombstone is a distinct record that can participate in log compaction.
A destination may require its own retained logical delete version even after physically hiding the entity.
**Alternative:** Actual connector emission and transformation settings determine whether either representation is observed downstream.
**Follow-up:** Why keep sink delete metadata? To prevent a permissible older replay from resurrecting the key.
**Trap:** Treating a null-value record as necessarily the sole delete representation in all connectors.
**Source:** Sections 3 and 8 F02.

### Level 2 — Derivation / Apply

**Q4 — Trace a late snapshot row.**
**Given:** Snapshot at W=100 has A=10; stream version 101 sets A=12 and arrives first.
**Answer:** Install 101:12; reject the subsequent W:10 because W is older under the stipulated comparator; final A=12.
The reverse delivery order gives 100:10 followed by 101:12 and the same state.
**Alternative:** A connector-specific collision protocol can solve this without assigning a synthetic comparable snapshot version.
**Follow-up:** What if 100 and 101 are not comparable? Do not run this algorithm until the documented handoff defines a safe relation.
**Trap:** Blindly overwriting by arrival order.
**Source:** Sections 4.1 and 5.1.

**Q5 — Calculate recovery duration.**
**Given:** Three million pending events, 20,000 arrivals/s, and 25,000 durable completions/s.
**Answer:** Surplus is 5,000 events/s; 3,000,000 / 5,000 = 600 s = 10 min under constant rates and no retry amplification.
The independent check is 5,000 × 600 = 3,000,000 events.
**Alternative:** Use a time-varying measured model if the constant-rate assumptions do not hold.
**Follow-up:** What happens at service 20,000/s? There is no surplus, so the positive backlog does not drain.
**Trap:** Dividing by total service rather than service minus continuing arrivals.
**Source:** CDC-4 and Section 5.2.

**Q6 — Explain why an idempotent upsert is not sufficient.**
**Answer:** Repeating the same upsert may be harmless, but a different older update can still overwrite newer state.
For example, applying A=12 then an old snapshot A=10 twice produces A=10 under blind arrival-order writes.
Compare valid source versions atomically or use a documented ordering protocol.
**Alternative:** An upstream ordering guarantee can suffice only if it holds through every replay, migration, and concurrency path.
**Follow-up:** What if two different events have the same version? Require a documented tie-breaker or quarantine the conflict.
**Trap:** Confusing duplicate identity with temporal order.
**Source:** Section 4.

**Q7 — Prove the monotone-version property.**
**Answer:** The empty initial state trivially has no decreasing transition.
Each atomic transition either retains its version or installs a strictly greater comparable version; induction proves nondecrease.
This does not prove the maximum is the true source latest version, because events can be missing.
**Alternative:** A single fenced owner can implement the atomic comparison without a database compare-and-swap primitive.
**Follow-up:** What changes if versions are incomparable across failover? The proof's comparison premise fails and the pipeline must resolve lineage.
**Trap:** Calling monotonicity proof of complete replication.
**Source:** Sections 4.1 and 10 R1.

### Level 3 — Judgement / Design / Debug

**Q8 — Select a bootstrap approach under a strict write-freeze allowance.**
**Given:** A small database permits a fully enforced maintenance window and batch freshness during the copy.
**Answer:** A quiesced consistent copy with a documented log start boundary can be simpler than an online snapshot protocol.
The write freeze must cover every writer and the handoff must be tested before reopening writes.
**Alternative:** Online consistent snapshot plus retained log may reduce downtime if its additional complexity is justified.
**Follow-up:** The freeze is withdrawn: choose an online connector-supported boundary and size log retention for scan plus catch-up.
**Trap:** Copying tables while writes continue and calling the result consistent without a source snapshot guarantee.
**Source:** Section 7.

**Q9 — Diagnose rising lag with low aggregate CPU.**
**Answer:** Compare per-partition input and durable completion rates, sink latency, connector status, and key distribution before scaling consumers.
A single hot partition can grow while other partitions and CPUs are idle.
**Alternative:** Destination throttling or connector pause can produce the same symptom; the discriminating metrics differ.
**Follow-up:** What if a hot key cannot be split? Reduce its work, add capacity within its ordering domain, or revise the business ordering contract.
**Trap:** Assuming another consumer automatically divides one Kafka partition.
**Source:** Section 8 F07 and the prerequisite sheet.

**Q10 — Design for atomic cross-table visibility.**
**Answer:** Establish source transaction membership and a documented completeness condition; durably stage all members and publish them in one destination visibility boundary.
This is in addition to idempotency, per-key version checks, and checkpoint safety.
**Alternative:** If the business accepts temporary partial state, use per-key materialization and explicitly document weaker visibility.
**Follow-up:** Metadata is unavailable: obtain a supported source contract or decline to promise atomic cross-table visibility.
**Trap:** Claiming Kafka's ordering within a partition makes multiple partitions visible simultaneously.
**Source:** Sections 4.2, 6.3, and 8 F05.

**Q11 — Recover after required source history expires.**
**Answer:** Stop publishing unverified state, preserve offsets and evidence, and build a new generation from a fresh consistent snapshot plus a valid change suffix.
Reconcile at a compatible boundary before moving readers.
**Alternative:** A verified independent complete archive may supply the missing history if its identity and ordering contracts are sufficient.
**Follow-up:** Can the connector simply start at the current position? Not without accepting and explicitly accounting for missing changes.
**Trap:** Resetting to latest and declaring the preexisting materialization correct.
**Source:** Section 8 F03 and F06.

**Q12 — Resolve a suspected duplicate external charge.**
**Answer:** Compare source event identity, transaction context, request idempotency key, and provider responses; distinguish retry from a genuinely repeated business action.
Suspend unsafe retries and coordinate remediation with the owner rather than issuing a new charge.
**Alternative:** A provider's durable idempotency API can resolve retry ambiguity if its scope and retention cover the retry horizon.
**Follow-up:** The provider lacks idempotency: require a different workflow or explicit reconciliation and compensation; do not promise exactly-once effect.
**Trap:** Believing a committed Kafka offset proves an external charge happened exactly once.
**Source:** Section 8 F04 and Section 10 R2.

**Q13 — Assess a green dashboard with mismatched row counts.**
**Answer:** Progress and state equality are different claims; first align source and sink boundary, filters, key scope, and delete semantics.
Then compare partitioned counts or hashes and inspect representative mismatched keys.
**Alternative:** If an exact shared boundary is unavailable, report uncertainty and use an isolated generation or a documented bounded comparison.
**Follow-up:** Equal counts but wrong rows? Counts can cancel; compare key sets and values, not totals alone.
**Trap:** Treating zero reported lag as proof of correct materialized state.
**Source:** Section 8 F12 and Section 10 R4.

**Q14 — Change the replay horizon.**
**Given:** Destination delete metadata expires after seven days; an operator proposes replay from a month ago into the live generation.
**Answer:** Reject that replay unless another retained ordering guard covers the earlier history; an old update can resurrect a deleted key.
Build an isolated generation or extend and validate retained metadata before changing the policy.
**Alternative:** A proven complete source history and clean rebuild may remove the need to apply old records to the live state.
**Follow-up:** Does a stable event ID alone solve the problem? Only if a complete, still-retained deduplication record also covers the old event and effect.
**Trap:** Assuming current-state rows encode deletion history after physical removal.
**Source:** Sections 4.4 and 8 F02.

**Takeaway:** The strongest answers name the scope of each guarantee, the evidence required, and a failure that invalidates it.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Correct snapshot-to-stream CDC handoff and duplicate-safe current-state materialization |
| Core formula | Drain time = backlog / (durable service − arrivals), for a positive sustained surplus; also cover the snapshot boundary without gaps and atomically reject older per-key versions |
| Key assumption | Source history, event identity, version order, and boundary inclusion are documented and retained |
| Complexity | Expected O(E) key operations for E events with a suitable hash-indexed sink; actual performance unmeasured |
| Use when | Continuous source changes and replay-safe per-key state are required |
| Avoid when | Source history is unavailable or incomparable, or cross-key atomic visibility is promised without transaction staging |
| Diagnostic | Compare source position, snapshot boundary, event identity, partition offset, and destination version for an affected key |
| Capacity | Drain time = backlog / (durable service − arrivals), only for positive sustained surplus and stated constant-rate assumptions |
| Go deeper | [Kafka partitions, offsets, and lag](01_Kafka_Consumer_Partitions_and_Lag.md); connector-specific documentation for deployed versions |
| Evidence | Illustrative draft; original case and deployment behavior remain unverified |
