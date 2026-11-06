---
collection: streaming-event-systems
file_id: "01"
topic: Kafka consumer partitions and lag
level: 3
status: draft
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: Kafka client, broker, group protocol, workload or sink contract changes
notation_style: plain-text
prerequisites: ["../INDEX.md"]
pattern_tags: [partitioning, consumer-groups, recovery]
source_status: OPEN
owner_role: streaming-platform maintainer
source_index: INDEX.md
---
# Kafka Consumer Partitions and Lag

## 1. File Header

> **Purpose:** Diagnose a lagging consumer group without sacrificing ordering or replay correctness.
> **Prerequisites:** Logs, distributed failures, rates and units, and basic database transactions; see [module](../INDEX.md).
> **Outcome:** Given partition rates, offsets, assignment and sink constraints, estimate recovery and select a bounded intervention.
> **Exclusions:** Detailed CDC snapshot semantics, Flink checkpoint implementation, and tiered-storage sizing are separate topics.
> **Level:** Advanced production; **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** ILLUSTRATIVE and DERIVED; no production measurements are claimed.
> **Retrieval cue:** Rising lag → Sections 5 and 8; offset safety → Section 4.

This is an unreviewed draft, not an incident runbook approved for a live cluster.
The named source tree appears in the supplied directory inventory, but its files were unavailable in this workspace.
Consequently no claimed incident detail or empirical finding is attributed to those files.
The [source brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#kafka) assigns the topic to this module.
The [incident brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#stream-incidents) describes a related planned incident.
A related [draft incident case](../../../10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/Case_Studies/01_hot_partition_consumer_lag_incident.md) offers a separate case perspective.
Kafka-specific behavior must be checked against the deployed broker and client versions.
The [Kafka documentation](https://kafka.apache.org/documentation/) is the primary reference for those version-specific semantics.
No production workload, measured capacity, benchmark, or successful recovery is asserted here.

**Takeaway:** First establish a correct processing boundary, then diagnose the slowest partition.

## 2. Concept Map / Mental Model

```text
producer key + partitioner -> topic partition -> ordered partition log
                                      |               |
                                      v               v
                            group assignment    retained offsets
                                      |               |
                                      v               v
                               fetch/poll -> application work -> durable sink effect
                                      |               |                 |
                                      v               v                 v
                               current position    in-flight      safe commit frontier
                                      |                                 |
                                      +----------> diagnostic <---------+
                                                       |
                                             mitigation / remeasurement
```

Arrows show data or control flow, not proof of causal attribution.
Each partition belongs to at most one active group member at a time under group assignment.
A member can own multiple partitions; an unassigned member adds no processing capacity.
Group membership determines partition ownership, while an independent group has its own offsets.
The commit frontier must not advance beyond work whose required effects are durable.
A growing offset distance is a signal; its cause may be production, processing, assignment, or measurement.

**Takeaway:** Partition ownership, durable effects, and offset progress are distinct state transitions.

## 3. Foundational Definitions

The following definitions apply to a single topic, group, and observation time unless stated otherwise.
- **Topic:** named log collection; it is not itself a globally ordered sequence.
- **Partition:** one ordered append log; distinct partitions have no Kafka-provided total order.
- **Record key:** producer-supplied bytes used by some partitioners; null-key behavior depends on client and settings.
- **Record offset:** position within one partition, not an event timestamp or globally unique ID.
- **Log start offset:** earliest currently addressable position, which can advance with retention.
- **End offset E:** a chosen end boundary for measurement, expressed as the next offset after that boundary.
- **Committed offset C:** group checkpoint for the next record to read after restart, not the last completed record.
- **Consumer position P:** next offset a particular consumer will fetch or return, not necessarily durably processed.
- **Lag L:** offset distance between an explicitly defined end boundary and C, in offset units.
- **Arrival rate a:** effective incoming work per second over a declared interval and partition.
- **Service rate s:** sustained durable completion rate in work units per second under the same conditions.
- **Backpressure:** a capacity constraint propagating upstream through bounded buffers, blocking, or paused input.
- **Replay:** reading records again from an earlier available position for recovery or recomputation.
- **Idempotency key:** stable identifier that permits a sink to recognize the same logical operation across retries.
- **Commit frontier:** next offset after the longest contiguous prefix with completed required durable effects.
- **Rebalance:** transition that changes group ownership; implementation and protocol determine disruption.
- **Retention:** policy limiting what history is available; compaction can remove records without removing all offset gaps.
- **Event time:** time represented by the event; it is distinct from broker append and processing times.
- **Oldest-record age:** elapsed wall time since a defined timestamp on the oldest outstanding record.

Offsets can contain gaps, including due to compaction, transactions, or control records.
Therefore offset distance is not always the number of readable business records.
End offset may mean log end, high watermark, or a transaction-aware boundary depending on the tool.
Record the tool's exact meaning and isolation mode before comparing its result to another dashboard.
A committed offset is a checkpoint, not a proof that an external effect happened.

**Takeaway:** A number labeled lag is meaningful only with its boundary, units, group, and timestamp.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Validity: one partition, steady rates, no loss, no gaps, and each record represents equal work.
Treat the queue as fluid while backlog is positive; rates refer to completed durable work.
Derivation: backlog change equals arrivals minus completions, multiplied by elapsed time.

```text
[1] FORMULA: B(t) = max(0, B(0) + (a - s) * t)
[2] WHERE: B is records, a and s are records/s, t is seconds.
[3] CONSTRAINT: constant rates, no loss and service capacity available whenever backlog exists.
[4] NUMERICAL EXAMPLE: B(0)=0, a=120/s, s=100/s, t=60 s => B=1,200 records.
[5] INTERPRETATION: backlog grows by 20 records/s until rates or workload change.
```

Independent check: 120/s produces 7,200 records; 100/s completes 6,000 in one minute.
Boundary: the floor at zero prevents a negative queue when capacity exceeds arrivals.
A measured offset delta is only an approximation to B if offset units map to business records.

### 4.2 General Case

Validity: per-partition independent queues; the following is an instantaneous approximation across a defined interval.
For partition i, arrivals and completions must be recorded in matching work units.
Sum the nonnegative per-partition backlogs, not the signed unused capacities across partitions.

```text
[1] FORMULA: B_total(t) = sum_i max(0, B_i(0) + (a_i - s_i) * t)
[2] WHERE: i identifies a partition; B_i is records; a_i and s_i are records/s.
[3] CONSTRAINT: constant rates and no cross-partition work stealing inside a single partition.
[4] NUMERICAL EXAMPLE: two empty queues: (a,s)=(120,100),(20,100), t=60 s => 1,200 records total.
[5] INTERPRETATION: spare capacity on the cold partition does not automatically serve the hot partition.
```

Reduction: for one partition this is the base-case expression.
Counterexample to aggregate averaging: total arrival 140/s is below total nominal service 200/s, yet the first queue grows.
Shared workers, CPU, sink pools, or broker throughput can make rates interdependent; measure them under concurrency.

### 4.3 Key Special Cases / Variants

| Variant | State change | Preconditions | Use | Avoid |
|---|---|---|---|---|
| Sequential partition handler | One record reaches sink at a time | Sink respects key order | Strict simple ordering | Throughput exceeds single-handler capacity |
| Parallel work with frontier | Many records in flight; commit contiguous completed prefix | Completion tracking and bounded in-flight work | Latency-bound independent work | Effects require sequential order without serialization |
| More group members | Reassign existing partitions | Unused assignable partitions and independent resources | Cold partitions share overloaded members | One partition alone is overloaded |
| More partitions | Future key mapping may change | Migration and ordering contract | Increase long-term parallelism | Immediate safe fix assumed without migration |
| Separate replay group | Independent offsets and sink isolation | Retained source and replay-safe output | Bounded historical rebuild | Shared irreversible sink has no deduplication |

Invariant trace, ILLUSTRATIVE:
1. A partition returns offsets 10, 11, and 12 in order.
2. Effects for 10 and 12 complete, but 11 is still in flight.
3. The safe committed next offset is 11, because 11 is the first unfinished record.
4. Committing 13 now risks losing 11 after a crash.
5. Once 11 completes durably, the contiguous prefix includes 10–12 and the next offset is 13.
6. If ownership is revoked, stop new work, resolve or abandon in-flight work, and commit only an owned safe frontier.
7. Exact callback and commit guarantees depend on the deployed group protocol and client.

Input contract: ordered offsets with per-record durable completion signals.
Output contract: a next offset never beyond the first unfinished item in the observed sequence.
A tracker records completed items and advances while its head is complete.
It terminates when the next observed item is incomplete or no tracked item remains.
Each item is inserted once and removed once; tracking costs linear operations in observed items and memory proportional to in-flight items.
A bounded in-flight limit avoids unbounded memory when one slow item blocks the frontier.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

When s equals a while backlog is positive, backlog does not drain in the constant-rate model.
When s is less than a, no finite recovery time exists without a workload or capacity change.
When s exceeds a, backlog can drain, subject to the bottleneck and retention window.
More members than partitions cannot increase the number of concurrently owned partitions within one group.
More partitions do not split already-written records or guarantee that one hot key is split.
Salting a hot key can distribute load, but sacrifices simple same-key partition ordering unless the application restores it.
Changing partition count can remap keys, so ordering across the change requires an explicit migration protocol.
A lag of zero can coexist with missing external effects if offsets were committed too early.
A low record count can coexist with unacceptable age when one old item is blocked.

**Takeaway:** Capacity must exceed arrivals at the bottleneck, and safe commits follow durable contiguous completion.

## 5. Worked Numerical Example

### 5.1 Throughput and skew

ILLUSTRATIVE: four partitions serve an order event stream with no gaps and equal record cost.
Assume a constant 300 records/s arrival rate per partition except partition 0 at 900 records/s.
Assume each partition can durably complete 600 records/s while backlog exists.
The other three partitions are not allowed to process records owned by partition 0.
Total arrival is 900 + 3 × 300 = 1,800 records/s.
Nominal total service is 4 × 600 = 2,400 records/s.
A naive aggregate calculation predicts no growth because 2,400 exceeds 1,800.
The hot partition actually grows at 900 − 600 = 300 records/s.
The cold partitions have unused capacity, not transferable service for that single partition.
After 20 minutes, elapsed time is 20 × 60 = 1,200 seconds.
The backlog gained by the hot partition is 300 × 1,200 = 360,000 records.
Independent check: arrivals are 900 × 1,200 = 1,080,000 records.
Completions are 600 × 1,200 = 720,000 records.
Their difference is 360,000 records, matching the rate calculation.
The correct decision is to investigate the hot key and its per-record cost, not blindly add consumers.

### 5.2 Recovery while traffic continues

Validity: fixed backlog B, constant incoming rate a, constant durable service rate s > a.
Derivation: net drain rate is s − a; dividing backlog by that rate gives drain time.

```text
[1] FORMULA: T = B / (s - a)
[2] WHERE: T is seconds, B is records, a and s are records/s.
[3] CONSTRAINT: B >= 0, s > a; arrivals continue and rates remain constant.
[4] NUMERICAL EXAMPLE: B=360,000, a=900/s, s=1,200/s => T=1,200 s = 20 min.
[5] INTERPRETATION: a doubled partition service rate drains this backlog in 20 minutes under the model.
```

Independent check: over 1,200 seconds, 1,440,000 completions minus 1,080,000 arrivals equals 360,000.
Wrong path: dividing by 1,200/s yields 300 seconds and ignores ongoing arrivals.
If service instead reaches only 1,000/s, net drain is 100/s and time is 3,600 seconds, or one hour.
If service stays at 900/s, net drain is zero and the model gives no finite recovery time.
Recovery capacity must exist on the hot partition itself, not merely on idle group members.
The estimate ignores bursts, retries, variable record cost, rebalances, retention and changing downstream limits.
Measure sustained rates and oldest-record age after any intervention; do not present this estimate as an SLA.

### 5.3 Retention and payload context

ILLUSTRATIVE: 20,000 events/s, 500 bytes/event, 86,400 s/day, three full replicas.
Daily logical payload is 20,000 × 500 × 86,400 = 864,000,000,000 bytes.
That is 864 GB using decimal GB, or approximately 804.66 GiB using 2^30 bytes/GiB.
Three full replicas contain 2,592,000,000,000 payload bytes, or 2,592 GB.
Independent check: 10 MB/s × 86,400 s = 864,000 MB = 864 GB.
These figures exclude compression, record framing, indexes, internal topics, disk reserve and remote storage.
Retention configuration is not evidence that the oldest required record is still readable.
A replay plan needs both the log start offset and the actual records required by the downstream computation.

**Takeaway:** Model each bottleneck separately and subtract ongoing arrivals when estimating drain time.

## 6. Visual / Geometric Intuition

```text
time ->             t0                 t1                  t2
partition log:     [10][11][12]------->[13][14]----------->[15]
completion:          10   pending 12     done
safe next offset:             11 --------------------------> 13 after 11 completes
```

Caption: schematic ordering and completion; offsets identify positions, not durations.
Offset 12 may finish before 11, but the safe frontier stays at 11 until 11 completes.
The next offset is 13 only after the required effects for the full prefix are durable.

```text
backlog (records)
  ^              /  arrival > service
  |             /
  |            * intervention
  |             \  service > arrival
  |              \_______ 0
  +--------------------------------> time (seconds)
```

Caption: schematic fluid queue; slopes are net rate, not absolute service rate.
At the intervention, the slope changes only if effective service exceeds continuing arrivals.

**Takeaway:** A completion frontier can stall behind one unfinished record even when later work completes.

## 7. Comparison / Selection Table

| Choice | Preconditions | Guarantee / risk | Cost | Use | Disqualifier | Evidence |
|---|---|---|---|---|---|---|
| Sequential baseline | One handler per partition | Simple in-partition effect order | Limited latency-bound throughput | Strong order needed | Sustainable rate below input | DERIVED |
| Async work and frontier | Independent effects or explicit key serialization | Prevents commit past incomplete prefix | Memory and bookkeeping | Sink latency dominates | Unordered effects violate contract | DERIVED |
| Add members | Spare partitions and shared resource headroom | Spreads partition ownership | More instances and rebalance | Many hot partitions on one member | Single hot partition | DERIVED |
| Optimize handler | Measurable slow operation | Depends on preserved semantics | Engineering and regression risk | CPU or I/O bottleneck | Unverified behavior change | HEURISTIC |
| Repartition / salt | Migratable ordering and state | More parallelism; key order may change | Producer and consumer migration | Hot-key load can be split safely | Strict unsplittable ordered key | DERIVED |
| Pause producers | Upstream can buffer or reject safely | Slows future growth | Availability and upstream backlog | Emergency overload | Upstream data loss or unsafe operation | HEURISTIC |

Priority: preserve irreversible-effect correctness and retention before optimizing throughput.
If a single key needs strict ordering, keep its serialized lane or redesign its business contract.
If many partitions are busy and members have spare resources, rebalance or scale members first.
If downstream saturation dominates, relieve the sink rather than raising concurrency into it.
If evidence conflicts, bound load, capture per-partition and sink metrics, then compare controlled changes.
No benchmark is claimed; compare options on the same workload, versions, hardware, sink, and correctness checks.

**Takeaway:** Select the intervention matching the limiting resource and ordering contract, not the aggregate graph.

## 8. Failure Modes & Edge Cases

All cases below are HYPOTHETICAL diagnostic exercises, not observed incidents.
Each distinguishes at least two plausible causes and identifies an actionable verification.
Never reset production offsets, delete a group, or change partition counts as a first diagnostic step.
Capture current offsets, assignments, deployment version, and incident time before intervention.

### 8.1 Hot partition

- Symptom: One partition rises while peers stay flat.
- Candidate A: A dominant key or unequal record cost.
- Candidate B: An assigned consumer is throttled or stalled.
- Discriminator: Compare keyed byte and record rates with CPU and processing latency for the same partition.
- Immediate bounded action: Bound or isolate the hot workload; preserve key order.
- Prevention: Revisit the key design and publish a skew budget.
- Verification: Hot partition net drain becomes positive without sink errors.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.2 Too few active members

- Symptom: Many partitions rise on a few busy members.
- Candidate A: Insufficient consumer instances.
- Candidate B: Members are alive but blocked on a common sink.
- Discriminator: Compare assignments, CPU, poll cadence and sink latency.
- Immediate bounded action: Add members only if partitions and sink capacity permit.
- Prevention: Capacity-test member loss and rebalances.
- Verification: Partition ownership spreads and durable throughput rises.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.3 Idle extra members

- Symptom: Replicas are idle despite lag.
- Candidate A: Partition count limits group parallelism.
- Candidate B: Subscription or authorization prevents assignment.
- Discriminator: Compare assigned partition count with member count and subscription errors.
- Immediate bounded action: Remove unnecessary replicas or plan a safe repartition.
- Prevention: Size partitions against throughput and ordering needs.
- Verification: Assignments and throughput match the revised design.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.4 Slow sink

- Symptom: Completion latency and lag rise together.
- Candidate A: Database or API saturation.
- Candidate B: Consumer CPU or network delay.
- Discriminator: Correlate sink queue and latency with client timings.
- Immediate bounded action: Limit concurrency and restore sink capacity.
- Prevention: Load-test the whole pipeline with sink limits.
- Verification: Completion latency falls and backlog drains.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.5 Repeated retries

- Symptom: Same event is processed repeatedly.
- Candidate A: Transient sink failure.
- Candidate B: Permanent invalid data or non-idempotent retry.
- Discriminator: Inspect classified errors and retry counts by stable event ID.
- Immediate bounded action: Bound retries and route permanent failures under an explicit policy.
- Prevention: Separate retryable from permanent errors.
- Verification: One durable effect per ID and no retry storm.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.6 Poison record

- Symptom: One partition stops at one offset.
- Candidate A: A malformed or unsupported payload.
- Candidate B: Temporary schema-registry or downstream outage.
- Discriminator: Inspect error class and payload schema against dependency health.
- Immediate bounded action: Quarantine only under an approved no-loss policy.
- Prevention: Validate producer schemas and retain original bytes.
- Verification: Next valid record progresses and quarantine can be replayed.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.7 Premature commit

- Symptom: Lag is low but sink records are missing.
- Candidate A: Offsets were committed before effects were durable.
- Candidate B: Sink query or reconciliation uses the wrong cohort.
- Discriminator: Compare committed frontier with durable effect ledger by event ID.
- Immediate bounded action: Stop unsafe advancement and recover from retained source.
- Prevention: Tie commits to completed contiguous work.
- Verification: Reconcile source IDs against sink effects after replay.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.8 Crash after sink write

- Symptom: Duplicate sink effects appear after restart.
- Candidate A: Effect committed before offset commit.
- Candidate B: Producer emitted duplicate logical events.
- Discriminator: Compare source coordinates and stable business event IDs.
- Immediate bounded action: Use transactional deduplication or idempotent sink operation.
- Prevention: Exercise crash points between effect and commit.
- Verification: Repeated replay yields one logical effect.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.9 Out-of-order async completion

- Symptom: Later offsets finish before earlier offsets.
- Candidate A: Concurrent request latency varies.
- Candidate B: Source records already encode different business ordering.
- Discriminator: Trace source offsets, entity IDs, and sink completion order.
- Immediate bounded action: Hold commit frontier and serialize effects where needed.
- Prevention: Test delayed early requests and per-key sequencing.
- Verification: No incorrect effect order and frontier waits for gaps.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.10 Rebalance churn

- Symptom: Ownership moves repeatedly with pauses.
- Candidate A: Consumer processing violates poll timing.
- Candidate B: Coordinator or network instability.
- Discriminator: Correlate poll intervals and processing duration with coordinator errors.
- Immediate bounded action: Reduce per-poll work or repair connectivity.
- Prevention: Set polling and workload budgets using deployed-client semantics.
- Verification: Membership stabilizes and throughput recovers.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.11 Long garbage collection

- Symptom: Processing pauses coincide with heap pressure.
- Candidate A: Oversized batches or unbounded in-flight tasks.
- Candidate B: Memory leak unrelated to workload.
- Discriminator: Compare heap, allocation, GC pauses and in-flight counts.
- Immediate bounded action: Reduce bounded concurrency or batch size.
- Prevention: Set memory budgets and test sustained peak traffic.
- Verification: Pause time and in-flight memory remain bounded.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.12 Broker throttling

- Symptom: Fetch throughput plateaus with broker throttle signals.
- Candidate A: Quota or broker resource saturation.
- Candidate B: Consumer application is not polling.
- Discriminator: Compare broker quota metrics, fetch latency and poll cadence.
- Immediate bounded action: Coordinate quota or capacity changes with tenant owners.
- Prevention: Budget tenant throughput and alert on throttling.
- Verification: Fetch rate rises without harming other tenants.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.13 Network degradation

- Symptom: Fetch times increase and throughput falls.
- Candidate A: Packet loss or bandwidth contention.
- Candidate B: Slow broker disk or leader overload.
- Discriminator: Compare client network telemetry with broker disk and leader metrics.
- Immediate bounded action: Reroute or repair the affected network path.
- Prevention: Monitor link saturation and failure-domain diversity.
- Verification: Fetch latency and network errors return to baseline.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.14 Partition leader change

- Symptom: Lag spikes during broker failover.
- Candidate A: Leadership movement interrupts fetch.
- Candidate B: Client metadata is stale or connection retries are excessive.
- Discriminator: Correlate leader events and client metadata refresh errors.
- Immediate bounded action: Restore healthy replicas and client connectivity.
- Prevention: Test broker loss and monitor replica health.
- Verification: Leader stabilizes and clients resume fetching.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.15 Offset out of range

- Symptom: Consumer cannot resume at committed position.
- Candidate A: Retention advanced past the checkpoint.
- Candidate B: A reset chose an invalid future offset.
- Discriminator: Compare committed offset with log start and end offsets.
- Immediate bounded action: Escalate data-loss decision; recover from an approved archive if possible.
- Prevention: Alert on retention headroom and practice recovery.
- Verification: Reconciled replay covers the required interval.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.16 Retention headroom exhausted

- Symptom: Oldest needed data approaches deletion.
- Candidate A: Slow recovery relative to retention.
- Candidate B: Retention policy is shorter than expected.
- Discriminator: Compare oldest required offset and actual log start progression.
- Immediate bounded action: Increase safe recovery capacity and coordinate retention if feasible.
- Prevention: Set retention for outage plus catch-up and margin.
- Verification: Required range remains available through replay.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.17 Compacted source gap

- Symptom: Replay does not recreate all intermediate actions.
- Candidate A: Compaction removed historical versions.
- Candidate B: Application expected only the latest state.
- Discriminator: Inspect topic cleanup policy and required event history.
- Immediate bounded action: Use an uncompacted audit source when full history is required.
- Prevention: Document snapshot versus event-history semantics.
- Verification: Rebuild reconciles to the stated output contract.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.18 Transaction visibility

- Symptom: End offset moves while read-committed consumer waits.
- Candidate A: Open transaction delays visible records.
- Candidate B: Consumer is paused or not polling.
- Discriminator: Compare transaction-aware readable boundary with group activity.
- Immediate bounded action: Resolve the stuck producer transaction safely.
- Prevention: Monitor transaction duration and producer health.
- Verification: Readable boundary advances and consumption resumes.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.19 Timestamp confusion

- Symptom: Oldest age disagrees with offset lag.
- Candidate A: Producer event timestamps are delayed or skewed.
- Candidate B: Monitoring uses broker timestamp or scrape time.
- Discriminator: Compare producer event time, broker time and processing time.
- Immediate bounded action: Use the timestamp matching the actual freshness objective.
- Prevention: Specify clocks and timestamp semantics in dashboards.
- Verification: Age metric aligns with independently traced records.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.20 Offset metric mismatch

- Symptom: Dashboards disagree on lag.
- Candidate A: Different end boundaries or isolation modes.
- Candidate B: Metrics were scraped at different times.
- Discriminator: Inspect tool definitions, group, partition and collection timestamps.
- Immediate bounded action: Choose one documented operational metric.
- Prevention: Version metric definitions with dashboards.
- Verification: Same snapshot and boundary produce consistent values.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.21 Coarse aggregate alert

- Symptom: Total lag looks healthy while one customer is stale.
- Candidate A: A hot partition is hidden by aggregation.
- Candidate B: A downstream consumer has a separate backlog.
- Discriminator: Inspect maximum partition age and downstream stage age.
- Immediate bounded action: Alert on worst affected partition and business freshness.
- Prevention: Define per-stage SLOs and skew-sensitive alerts.
- Verification: Alert fires on a targeted skew test.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.22 Large record imbalance

- Symptom: Record rate is low but byte traffic is high.
- Candidate A: Payload size distribution shifted.
- Candidate B: Compression or serialization changed.
- Discriminator: Compare per-partition byte rate, size percentiles and compression.
- Immediate bounded action: Bound payloads or provision for bytes.
- Prevention: Set producer payload contracts and size monitoring.
- Verification: Byte throughput and end-to-end latency stabilize.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.23 Schema incompatibility

- Symptom: Deserializer failures begin after deploy.
- Candidate A: Producer schema changed incompatibly.
- Candidate B: Consumer configuration points to the wrong registry.
- Discriminator: Compare schema IDs, compatibility settings and deployment timeline.
- Immediate bounded action: Roll back or deploy a compatible reader.
- Prevention: Test old and new schemas across rollout order.
- Verification: Replay representative old and new records successfully.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.24 Authorization regression

- Symptom: A subset of consumers cannot fetch.
- Candidate A: ACL or credential rotation broke access.
- Candidate B: Network connectivity to brokers failed.
- Discriminator: Distinguish authentication, authorization and connection error codes.
- Immediate bounded action: Restore least-privilege access via approved procedure.
- Prevention: Test rotation and monitor credential expiry.
- Verification: Affected clients fetch only authorized resources.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.25 Downstream connection pool exhaustion

- Symptom: Workers wait while CPU remains low.
- Candidate A: Too many concurrent sink calls.
- Candidate B: A connection leak or unusually slow transactions.
- Discriminator: Compare active, idle and waiting connections with transaction duration.
- Immediate bounded action: Reduce concurrency or fix leaked connections.
- Prevention: Bound pool use and set request deadlines.
- Verification: Wait time falls without increased sink errors.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.26 Consumer pause forgotten

- Symptom: Assigned partition stops fetching after a recovery action.
- Candidate A: Application did not resume a paused partition.
- Candidate B: A legitimate backpressure gate is still active.
- Discriminator: Inspect pause state, gate condition and downstream health.
- Immediate bounded action: Resume only after the documented safety condition clears.
- Prevention: Log pause reason and expiry ownership.
- Verification: Fetch resumes and buffers remain bounded.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.27 Producer surge

- Symptom: All partition ingress rates rise together.
- Candidate A: A legitimate traffic burst.
- Candidate B: Accidental producer retry amplification.
- Discriminator: Compare unique event IDs, producer retries and business demand.
- Immediate bounded action: Apply admission control or coordinate upstream throttling.
- Prevention: Plan burst capacity and producer retry budgets.
- Verification: Unique arrival rate and backlog trend become sustainable.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.28 Partition count change

- Symptom: One entity appears in multiple partition histories.
- Candidate A: Key remapping after partition expansion.
- Candidate B: Producer changed key serialization.
- Discriminator: Compare partition count and serialized key bytes across the transition.
- Immediate bounded action: Use a controlled migration with an ordering barrier.
- Prevention: Version partitioning contracts and test migrations.
- Verification: Per-entity sequence reconciles across the cutover.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.29 Replay contaminates live output

- Symptom: Historical events trigger live external actions.
- Candidate A: Replay writes to the production sink.
- Candidate B: Event IDs differ between replay and live ingestion.
- Discriminator: Compare group, sink destination and event identity.
- Immediate bounded action: Stop replay side effects and isolate its output.
- Prevention: Require replay mode and sink authorization boundaries.
- Verification: Replay produces no unapproved external action.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.30 Deduplication expiry

- Symptom: Old replay creates duplicate effects.
- Candidate A: Deduplication TTL is shorter than replay horizon.
- Candidate B: Stable IDs were not propagated.
- Discriminator: Compare dedup retention with oldest replayable event and IDs.
- Immediate bounded action: Extend or rebuild dedup state before replay.
- Prevention: Align dedup horizon with retention and recovery policy.
- Verification: Replaying the oldest supported event creates no duplicate effect.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.31 Incorrect reset target

- Symptom: Reset resumes at an unexpected position.
- Candidate A: Timestamp lookup or timezone assumption was wrong.
- Candidate B: Offsets were advanced before the reset finished.
- Discriminator: Inspect explicit per-partition target offsets and group state.
- Immediate bounded action: Stop the group and restore recorded safe offsets if available.
- Prevention: Dry-run reset plans with approvals and captured offsets.
- Verification: Read-back positions match the approved target.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.32 Lag grows after scaling

- Symptom: More consumers increase sink errors and lag.
- Candidate A: Shared downstream resource is saturated.
- Candidate B: Rebalance interrupted work temporarily.
- Discriminator: Compare sink concurrency and error rates before and after stable assignment.
- Immediate bounded action: Roll back concurrency to a safe limit.
- Prevention: Load-test scaling against downstream quotas.
- Verification: Error rate falls and net completion rate rises.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.33 Batch size too large

- Symptom: Poll interval breaches occur under load.
- Candidate A: A large batch takes too long to process.
- Candidate B: One operation hangs without a deadline.
- Discriminator: Compare batch duration distribution with per-operation duration.
- Immediate bounded action: Reduce batch work or bound each call.
- Prevention: Budget worst-case poll intervals and timeouts.
- Verification: Poll cadence remains within configured limits.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.34 Excessive commit frequency

- Symptom: Commit requests consume coordinator capacity.
- Candidate A: Offsets are committed on every record.
- Candidate B: Coordinator is already overloaded by group churn.
- Discriminator: Compare commit request volume and rebalance rate.
- Immediate bounded action: Batch safe commits while accepting replay window.
- Prevention: Set explicit duplicate and recovery budgets.
- Verification: Coordinator latency improves without missed effects.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.35 Sparse commit frequency

- Symptom: Restart replays a large amount of completed work.
- Candidate A: Commit interval is too long.
- Candidate B: Commits failed silently during normal processing.
- Discriminator: Compare last successful commit time with configured cadence and errors.
- Immediate bounded action: Restore successful safe commits.
- Prevention: Alert on commit failures and frontier staleness.
- Verification: Restart replay remains within the declared budget.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.36 Offset gaps mistaken for loss

- Symptom: Offset distance exceeds business record count.
- Candidate A: Compaction or transaction/control records produce gaps.
- Candidate B: Records were actually dropped downstream.
- Discriminator: Compare readable records and sink IDs rather than raw offset subtraction.
- Immediate bounded action: Reconcile with an appropriate source-of-truth ledger.
- Prevention: Document offset units and cleanup policy.
- Verification: Business-event reconciliation has no unexplained missing IDs.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.37 Clock skew

- Symptom: Latency is negative or implausibly high.
- Candidate A: Producer or consumer clocks are unsynchronized.
- Candidate B: Event timestamp represents a different business time.
- Discriminator: Compare synchronized host clocks and timestamp definitions.
- Immediate bounded action: Repair time synchronization or change the metric.
- Prevention: Monitor clock offset and preserve multiple timestamps.
- Verification: Known test events produce plausible stage latencies.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.38 Disk pressure

- Symptom: Broker fetch or produce latency rises with disk utilization.
- Candidate A: Retention or replication consumed capacity.
- Candidate B: A replica or disk has failed.
- Discriminator: Compare per-broker storage, replica health and disk errors.
- Immediate bounded action: Restore safe capacity without deleting required history.
- Prevention: Model replicated bytes and retain operational reserve.
- Verification: Disk headroom and replica health recover.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.39 Unbounded retry queue

- Symptom: Memory grows while throughput falls.
- Candidate A: Retries are enqueued without capacity limits.
- Candidate B: The main processing queue is genuinely growing.
- Discriminator: Compare retry queue size and age with main in-flight work.
- Immediate bounded action: Bound retry queues and apply backpressure.
- Prevention: Set retry budgets, jitter and dead-letter governance.
- Verification: Memory stays bounded and valid events progress.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.40 Cross-partition order assumption

- Symptom: A customer sees an older state after a newer state.
- Candidate A: Related events use different partitions.
- Candidate B: Sink applies parallel updates without version checks.
- Discriminator: Compare entity key, partition, event version and sink write order.
- Immediate bounded action: Use a common key or conditional versioned updates.
- Prevention: Define entity ordering and migration contracts.
- Verification: Adversarial ordering tests preserve correct state.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.41 Group identity mistake

- Symptom: A deployment unexpectedly reprocesses or shares work.
- Candidate A: A new group ID has no prior checkpoint.
- Candidate B: Two applications accidentally share a group ID.
- Discriminator: Compare configured group IDs and expected ownership.
- Immediate bounded action: Restore the intended group only after checking offset consequences.
- Prevention: Treat group IDs as versioned deployment configuration.
- Verification: Expected members and offsets are visible in the intended group.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.42 Bad autoscaling signal

- Symptom: Replica count rises without improved throughput.
- Candidate A: Scaling uses total lag without partition limits.
- Candidate B: Sink capacity has already saturated.
- Discriminator: Compare per-partition rates, assignment and sink capacity.
- Immediate bounded action: Cap scaling and address the bottleneck.
- Prevention: Combine lag age, net drain and resource saturation.
- Verification: Scale tests improve throughput without increasing errors.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.43 State rebuild exceeds capacity

- Symptom: A restarted stateful consumer never catches up.
- Candidate A: Replay competes with live work for limited resources.
- Candidate B: Required history is absent or state checkpoint is invalid.
- Discriminator: Compare replay rate, incoming rate, checkpoint age and log start.
- Immediate bounded action: Isolate rebuild resources or restore a valid checkpoint.
- Prevention: Exercise cold-start rebuild before release.
- Verification: Rebuild finishes before retention or freshness deadline.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.44 Unexpected side effects on retry

- Symptom: An email or payment occurs twice.
- Candidate A: External API lacks idempotency support.
- Candidate B: Application generates a new key per attempt.
- Discriminator: Trace logical operation IDs and downstream request IDs.
- Immediate bounded action: Stop automatic retries for irreversible effects until safeguarded.
- Prevention: Use stable provider keys or an outbox with reconciliation.
- Verification: Crash and retry tests cause one intended effect.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.45 CPU-heavy deserialization

- Symptom: Fetch is fast but durable completion is slow.
- Candidate A: Decoding or validation dominates CPU.
- Candidate B: Garbage collection or downstream calls dominate.
- Discriminator: Profile CPU alongside allocation and external call latency.
- Immediate bounded action: Optimize the measured hot path without changing semantics.
- Prevention: Benchmark realistic schemas and payload sizes.
- Verification: CPU per event falls and outputs remain equivalent.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.46 Broker imbalance

- Symptom: Partitions on one broker show correlated fetch delays.
- Candidate A: Leader placement concentrates hot partitions.
- Candidate B: A single network path or storage device is degraded.
- Discriminator: Map leaders to broker resources and compare affected clients.
- Immediate bounded action: Rebalance leaders or repair the constrained broker.
- Prevention: Capacity-plan leader placement and failover.
- Verification: Per-broker latency and partition drain converge.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.47 DLQ silently loses data

- Symptom: Lag falls but records vanish from business output.
- Candidate A: Failed records were sent to an unmonitored dead-letter queue.
- Candidate B: A commit occurred despite failed DLQ publication.
- Discriminator: Trace failure ID through DLQ durable write and source commit.
- Immediate bounded action: Restore durable DLQ publication and alert owners.
- Prevention: Define DLQ retention, access, replay and reconciliation.
- Verification: Every rejected ID is accounted for and replayable.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.48 Partial database transaction

- Symptom: One event updates only some related rows.
- Candidate A: Sink writes are not atomic.
- Candidate B: A later compensating operation changed a row.
- Discriminator: Inspect transaction boundaries and audit timestamps.
- Immediate bounded action: Use atomic transaction or explicit compensating workflow.
- Prevention: Test failure between every related mutation.
- Verification: Reconciliation finds no partial logical operation.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.49 Rebalance during in-flight work

- Symptom: A new owner and old owner both perform an effect.
- Candidate A: Old work continued after revocation.
- Candidate B: Producer sent two distinct records for the same operation.
- Discriminator: Compare assignment epochs, event IDs and sink timestamps.
- Immediate bounded action: Fence stale work and use sink idempotency.
- Prevention: Test revocation under slow sink calls.
- Verification: Ownership changes do not duplicate logical effects.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.

### 8.50 Recovery estimate misses arrivals

- Symptom: Observed drain is far slower than forecast.
- Candidate A: Estimate divided backlog by gross service.
- Candidate B: Traffic or service rate changed during recovery.
- Discriminator: Compare time-series of arrivals, completions and backlog.
- Immediate bounded action: Recalculate with net service and update incident forecast.
- Prevention: Publish assumptions and sensitivity with estimates.
- Verification: Predicted and observed net drain agree within stated uncertainty.
- Provenance: HYPOTHETICAL; collect timestamps and per-partition evidence before choosing a cause.


### 8.51 Controlled diagnostic experiments

These experiments are proposed tests, not results or instructions to disrupt production.
Use a representative isolated environment, synthetic data, an approved rollback, and no real customer side effects.
Record broker/client versions, partition assignment, rate window, sink behavior and test timestamp.
Do not infer causation from the first correlated metric; compare the counter-hypothesis.

**Experiment 1 — Hot partition.**
- Reproduce the condition represented by: one partition rises while peers stay flat.
- Hypothesis to challenge: a dominant key or unequal record cost.
- Alternative explanation to retain: an assigned consumer is throttled or stalled.
- Collect the discriminating evidence: compare keyed byte and record rates with cpu and processing latency for the same partition.
- Exercise the proposed control: bound or isolate the hot workload; preserve key order.
- Acceptance observation: hot partition net drain becomes positive without sink errors.
- Before rollout, document the prevention control: revisit the key design and publish a skew budget.

**Experiment 2 — Too few active members.**
- Reproduce the condition represented by: many partitions rise on a few busy members.
- Hypothesis to challenge: insufficient consumer instances.
- Alternative explanation to retain: members are alive but blocked on a common sink.
- Collect the discriminating evidence: compare assignments, cpu, poll cadence and sink latency.
- Exercise the proposed control: add members only if partitions and sink capacity permit.
- Acceptance observation: partition ownership spreads and durable throughput rises.
- Before rollout, document the prevention control: capacity-test member loss and rebalances.

**Experiment 3 — Idle extra members.**
- Reproduce the condition represented by: replicas are idle despite lag.
- Hypothesis to challenge: partition count limits group parallelism.
- Alternative explanation to retain: subscription or authorization prevents assignment.
- Collect the discriminating evidence: compare assigned partition count with member count and subscription errors.
- Exercise the proposed control: remove unnecessary replicas or plan a safe repartition.
- Acceptance observation: assignments and throughput match the revised design.
- Before rollout, document the prevention control: size partitions against throughput and ordering needs.

**Experiment 4 — Slow sink.**
- Reproduce the condition represented by: completion latency and lag rise together.
- Hypothesis to challenge: database or api saturation.
- Alternative explanation to retain: consumer cpu or network delay.
- Collect the discriminating evidence: correlate sink queue and latency with client timings.
- Exercise the proposed control: limit concurrency and restore sink capacity.
- Acceptance observation: completion latency falls and backlog drains.
- Before rollout, document the prevention control: load-test the whole pipeline with sink limits.

**Experiment 5 — Repeated retries.**
- Reproduce the condition represented by: same event is processed repeatedly.
- Hypothesis to challenge: transient sink failure.
- Alternative explanation to retain: permanent invalid data or non-idempotent retry.
- Collect the discriminating evidence: inspect classified errors and retry counts by stable event id.
- Exercise the proposed control: bound retries and route permanent failures under an explicit policy.
- Acceptance observation: one durable effect per id and no retry storm.
- Before rollout, document the prevention control: separate retryable from permanent errors.

**Experiment 6 — Poison record.**
- Reproduce the condition represented by: one partition stops at one offset.
- Hypothesis to challenge: a malformed or unsupported payload.
- Alternative explanation to retain: temporary schema-registry or downstream outage.
- Collect the discriminating evidence: inspect error class and payload schema against dependency health.
- Exercise the proposed control: quarantine only under an approved no-loss policy.
- Acceptance observation: next valid record progresses and quarantine can be replayed.
- Before rollout, document the prevention control: validate producer schemas and retain original bytes.

**Experiment 7 — Premature commit.**
- Reproduce the condition represented by: lag is low but sink records are missing.
- Hypothesis to challenge: offsets were committed before effects were durable.
- Alternative explanation to retain: sink query or reconciliation uses the wrong cohort.
- Collect the discriminating evidence: compare committed frontier with durable effect ledger by event id.
- Exercise the proposed control: stop unsafe advancement and recover from retained source.
- Acceptance observation: reconcile source ids against sink effects after replay.
- Before rollout, document the prevention control: tie commits to completed contiguous work.

**Experiment 8 — Crash after sink write.**
- Reproduce the condition represented by: duplicate sink effects appear after restart.
- Hypothesis to challenge: effect committed before offset commit.
- Alternative explanation to retain: producer emitted duplicate logical events.
- Collect the discriminating evidence: compare source coordinates and stable business event ids.
- Exercise the proposed control: use transactional deduplication or idempotent sink operation.
- Acceptance observation: repeated replay yields one logical effect.
- Before rollout, document the prevention control: exercise crash points between effect and commit.

**Experiment 9 — Out-of-order async completion.**
- Reproduce the condition represented by: later offsets finish before earlier offsets.
- Hypothesis to challenge: concurrent request latency varies.
- Alternative explanation to retain: source records already encode different business ordering.
- Collect the discriminating evidence: trace source offsets, entity ids, and sink completion order.
- Exercise the proposed control: hold commit frontier and serialize effects where needed.
- Acceptance observation: no incorrect effect order and frontier waits for gaps.
- Before rollout, document the prevention control: test delayed early requests and per-key sequencing.

**Experiment 10 — Rebalance churn.**
- Reproduce the condition represented by: ownership moves repeatedly with pauses.
- Hypothesis to challenge: consumer processing violates poll timing.
- Alternative explanation to retain: coordinator or network instability.
- Collect the discriminating evidence: correlate poll intervals and processing duration with coordinator errors.
- Exercise the proposed control: reduce per-poll work or repair connectivity.
- Acceptance observation: membership stabilizes and throughput recovers.
- Before rollout, document the prevention control: set polling and workload budgets using deployed-client semantics.

**Experiment 11 — Long garbage collection.**
- Reproduce the condition represented by: processing pauses coincide with heap pressure.
- Hypothesis to challenge: oversized batches or unbounded in-flight tasks.
- Alternative explanation to retain: memory leak unrelated to workload.
- Collect the discriminating evidence: compare heap, allocation, gc pauses and in-flight counts.
- Exercise the proposed control: reduce bounded concurrency or batch size.
- Acceptance observation: pause time and in-flight memory remain bounded.
- Before rollout, document the prevention control: set memory budgets and test sustained peak traffic.

**Experiment 12 — Broker throttling.**
- Reproduce the condition represented by: fetch throughput plateaus with broker throttle signals.
- Hypothesis to challenge: quota or broker resource saturation.
- Alternative explanation to retain: consumer application is not polling.
- Collect the discriminating evidence: compare broker quota metrics, fetch latency and poll cadence.
- Exercise the proposed control: coordinate quota or capacity changes with tenant owners.
- Acceptance observation: fetch rate rises without harming other tenants.
- Before rollout, document the prevention control: budget tenant throughput and alert on throttling.

**Experiment 13 — Network degradation.**
- Reproduce the condition represented by: fetch times increase and throughput falls.
- Hypothesis to challenge: packet loss or bandwidth contention.
- Alternative explanation to retain: slow broker disk or leader overload.
- Collect the discriminating evidence: compare client network telemetry with broker disk and leader metrics.
- Exercise the proposed control: reroute or repair the affected network path.
- Acceptance observation: fetch latency and network errors return to baseline.
- Before rollout, document the prevention control: monitor link saturation and failure-domain diversity.

**Experiment 14 — Partition leader change.**
- Reproduce the condition represented by: lag spikes during broker failover.
- Hypothesis to challenge: leadership movement interrupts fetch.
- Alternative explanation to retain: client metadata is stale or connection retries are excessive.
- Collect the discriminating evidence: correlate leader events and client metadata refresh errors.
- Exercise the proposed control: restore healthy replicas and client connectivity.
- Acceptance observation: leader stabilizes and clients resume fetching.
- Before rollout, document the prevention control: test broker loss and monitor replica health.

**Experiment 15 — Offset out of range.**
- Reproduce the condition represented by: consumer cannot resume at committed position.
- Hypothesis to challenge: retention advanced past the checkpoint.
- Alternative explanation to retain: a reset chose an invalid future offset.
- Collect the discriminating evidence: compare committed offset with log start and end offsets.
- Exercise the proposed control: escalate data-loss decision; recover from an approved archive if possible.
- Acceptance observation: reconciled replay covers the required interval.
- Before rollout, document the prevention control: alert on retention headroom and practice recovery.

**Experiment 16 — Retention headroom exhausted.**
- Reproduce the condition represented by: oldest needed data approaches deletion.
- Hypothesis to challenge: slow recovery relative to retention.
- Alternative explanation to retain: retention policy is shorter than expected.
- Collect the discriminating evidence: compare oldest required offset and actual log start progression.
- Exercise the proposed control: increase safe recovery capacity and coordinate retention if feasible.
- Acceptance observation: required range remains available through replay.
- Before rollout, document the prevention control: set retention for outage plus catch-up and margin.

**Experiment 17 — Compacted source gap.**
- Reproduce the condition represented by: replay does not recreate all intermediate actions.
- Hypothesis to challenge: compaction removed historical versions.
- Alternative explanation to retain: application expected only the latest state.
- Collect the discriminating evidence: inspect topic cleanup policy and required event history.
- Exercise the proposed control: use an uncompacted audit source when full history is required.
- Acceptance observation: rebuild reconciles to the stated output contract.
- Before rollout, document the prevention control: document snapshot versus event-history semantics.

**Experiment 18 — Transaction visibility.**
- Reproduce the condition represented by: end offset moves while read-committed consumer waits.
- Hypothesis to challenge: open transaction delays visible records.
- Alternative explanation to retain: consumer is paused or not polling.
- Collect the discriminating evidence: compare transaction-aware readable boundary with group activity.
- Exercise the proposed control: resolve the stuck producer transaction safely.
- Acceptance observation: readable boundary advances and consumption resumes.
- Before rollout, document the prevention control: monitor transaction duration and producer health.

**Experiment 19 — Timestamp confusion.**
- Reproduce the condition represented by: oldest age disagrees with offset lag.
- Hypothesis to challenge: producer event timestamps are delayed or skewed.
- Alternative explanation to retain: monitoring uses broker timestamp or scrape time.
- Collect the discriminating evidence: compare producer event time, broker time and processing time.
- Exercise the proposed control: use the timestamp matching the actual freshness objective.
- Acceptance observation: age metric aligns with independently traced records.
- Before rollout, document the prevention control: specify clocks and timestamp semantics in dashboards.

**Experiment 20 — Offset metric mismatch.**
- Reproduce the condition represented by: dashboards disagree on lag.
- Hypothesis to challenge: different end boundaries or isolation modes.
- Alternative explanation to retain: metrics were scraped at different times.
- Collect the discriminating evidence: inspect tool definitions, group, partition and collection timestamps.
- Exercise the proposed control: choose one documented operational metric.
- Acceptance observation: same snapshot and boundary produce consistent values.
- Before rollout, document the prevention control: version metric definitions with dashboards.

**Experiment 21 — Coarse aggregate alert.**
- Reproduce the condition represented by: total lag looks healthy while one customer is stale.
- Hypothesis to challenge: a hot partition is hidden by aggregation.
- Alternative explanation to retain: a downstream consumer has a separate backlog.
- Collect the discriminating evidence: inspect maximum partition age and downstream stage age.
- Exercise the proposed control: alert on worst affected partition and business freshness.
- Acceptance observation: alert fires on a targeted skew test.
- Before rollout, document the prevention control: define per-stage slos and skew-sensitive alerts.

**Experiment 22 — Large record imbalance.**
- Reproduce the condition represented by: record rate is low but byte traffic is high.
- Hypothesis to challenge: payload size distribution shifted.
- Alternative explanation to retain: compression or serialization changed.
- Collect the discriminating evidence: compare per-partition byte rate, size percentiles and compression.
- Exercise the proposed control: bound payloads or provision for bytes.
- Acceptance observation: byte throughput and end-to-end latency stabilize.
- Before rollout, document the prevention control: set producer payload contracts and size monitoring.

**Experiment 23 — Schema incompatibility.**
- Reproduce the condition represented by: deserializer failures begin after deploy.
- Hypothesis to challenge: producer schema changed incompatibly.
- Alternative explanation to retain: consumer configuration points to the wrong registry.
- Collect the discriminating evidence: compare schema ids, compatibility settings and deployment timeline.
- Exercise the proposed control: roll back or deploy a compatible reader.
- Acceptance observation: replay representative old and new records successfully.
- Before rollout, document the prevention control: test old and new schemas across rollout order.

**Experiment 24 — Authorization regression.**
- Reproduce the condition represented by: a subset of consumers cannot fetch.
- Hypothesis to challenge: acl or credential rotation broke access.
- Alternative explanation to retain: network connectivity to brokers failed.
- Collect the discriminating evidence: distinguish authentication, authorization and connection error codes.
- Exercise the proposed control: restore least-privilege access via approved procedure.
- Acceptance observation: affected clients fetch only authorized resources.
- Before rollout, document the prevention control: test rotation and monitor credential expiry.

**Experiment 25 — Downstream connection pool exhaustion.**
- Reproduce the condition represented by: workers wait while cpu remains low.
- Hypothesis to challenge: too many concurrent sink calls.
- Alternative explanation to retain: a connection leak or unusually slow transactions.
- Collect the discriminating evidence: compare active, idle and waiting connections with transaction duration.
- Exercise the proposed control: reduce concurrency or fix leaked connections.
- Acceptance observation: wait time falls without increased sink errors.
- Before rollout, document the prevention control: bound pool use and set request deadlines.

**Experiment 26 — Consumer pause forgotten.**
- Reproduce the condition represented by: assigned partition stops fetching after a recovery action.
- Hypothesis to challenge: application did not resume a paused partition.
- Alternative explanation to retain: a legitimate backpressure gate is still active.
- Collect the discriminating evidence: inspect pause state, gate condition and downstream health.
- Exercise the proposed control: resume only after the documented safety condition clears.
- Acceptance observation: fetch resumes and buffers remain bounded.
- Before rollout, document the prevention control: log pause reason and expiry ownership.

**Experiment 27 — Producer surge.**
- Reproduce the condition represented by: all partition ingress rates rise together.
- Hypothesis to challenge: a legitimate traffic burst.
- Alternative explanation to retain: accidental producer retry amplification.
- Collect the discriminating evidence: compare unique event ids, producer retries and business demand.
- Exercise the proposed control: apply admission control or coordinate upstream throttling.
- Acceptance observation: unique arrival rate and backlog trend become sustainable.
- Before rollout, document the prevention control: plan burst capacity and producer retry budgets.

**Experiment 28 — Partition count change.**
- Reproduce the condition represented by: one entity appears in multiple partition histories.
- Hypothesis to challenge: key remapping after partition expansion.
- Alternative explanation to retain: producer changed key serialization.
- Collect the discriminating evidence: compare partition count and serialized key bytes across the transition.
- Exercise the proposed control: use a controlled migration with an ordering barrier.
- Acceptance observation: per-entity sequence reconciles across the cutover.
- Before rollout, document the prevention control: version partitioning contracts and test migrations.

**Experiment 29 — Replay contaminates live output.**
- Reproduce the condition represented by: historical events trigger live external actions.
- Hypothesis to challenge: replay writes to the production sink.
- Alternative explanation to retain: event ids differ between replay and live ingestion.
- Collect the discriminating evidence: compare group, sink destination and event identity.
- Exercise the proposed control: stop replay side effects and isolate its output.
- Acceptance observation: replay produces no unapproved external action.
- Before rollout, document the prevention control: require replay mode and sink authorization boundaries.

**Experiment 30 — Deduplication expiry.**
- Reproduce the condition represented by: old replay creates duplicate effects.
- Hypothesis to challenge: deduplication ttl is shorter than replay horizon.
- Alternative explanation to retain: stable ids were not propagated.
- Collect the discriminating evidence: compare dedup retention with oldest replayable event and ids.
- Exercise the proposed control: extend or rebuild dedup state before replay.
- Acceptance observation: replaying the oldest supported event creates no duplicate effect.
- Before rollout, document the prevention control: align dedup horizon with retention and recovery policy.

**Experiment 31 — Incorrect reset target.**
- Reproduce the condition represented by: reset resumes at an unexpected position.
- Hypothesis to challenge: timestamp lookup or timezone assumption was wrong.
- Alternative explanation to retain: offsets were advanced before the reset finished.
- Collect the discriminating evidence: inspect explicit per-partition target offsets and group state.
- Exercise the proposed control: stop the group and restore recorded safe offsets if available.
- Acceptance observation: read-back positions match the approved target.
- Before rollout, document the prevention control: dry-run reset plans with approvals and captured offsets.

**Experiment 32 — Lag grows after scaling.**
- Reproduce the condition represented by: more consumers increase sink errors and lag.
- Hypothesis to challenge: shared downstream resource is saturated.
- Alternative explanation to retain: rebalance interrupted work temporarily.
- Collect the discriminating evidence: compare sink concurrency and error rates before and after stable assignment.
- Exercise the proposed control: roll back concurrency to a safe limit.
- Acceptance observation: error rate falls and net completion rate rises.
- Before rollout, document the prevention control: load-test scaling against downstream quotas.

**Experiment 33 — Batch size too large.**
- Reproduce the condition represented by: poll interval breaches occur under load.
- Hypothesis to challenge: a large batch takes too long to process.
- Alternative explanation to retain: one operation hangs without a deadline.
- Collect the discriminating evidence: compare batch duration distribution with per-operation duration.
- Exercise the proposed control: reduce batch work or bound each call.
- Acceptance observation: poll cadence remains within configured limits.
- Before rollout, document the prevention control: budget worst-case poll intervals and timeouts.

**Experiment 34 — Excessive commit frequency.**
- Reproduce the condition represented by: commit requests consume coordinator capacity.
- Hypothesis to challenge: offsets are committed on every record.
- Alternative explanation to retain: coordinator is already overloaded by group churn.
- Collect the discriminating evidence: compare commit request volume and rebalance rate.
- Exercise the proposed control: batch safe commits while accepting replay window.
- Acceptance observation: coordinator latency improves without missed effects.
- Before rollout, document the prevention control: set explicit duplicate and recovery budgets.

**Experiment 35 — Sparse commit frequency.**
- Reproduce the condition represented by: restart replays a large amount of completed work.
- Hypothesis to challenge: commit interval is too long.
- Alternative explanation to retain: commits failed silently during normal processing.
- Collect the discriminating evidence: compare last successful commit time with configured cadence and errors.
- Exercise the proposed control: restore successful safe commits.
- Acceptance observation: restart replay remains within the declared budget.
- Before rollout, document the prevention control: alert on commit failures and frontier staleness.

**Experiment 36 — Offset gaps mistaken for loss.**
- Reproduce the condition represented by: offset distance exceeds business record count.
- Hypothesis to challenge: compaction or transaction/control records produce gaps.
- Alternative explanation to retain: records were actually dropped downstream.
- Collect the discriminating evidence: compare readable records and sink ids rather than raw offset subtraction.
- Exercise the proposed control: reconcile with an appropriate source-of-truth ledger.
- Acceptance observation: business-event reconciliation has no unexplained missing ids.
- Before rollout, document the prevention control: document offset units and cleanup policy.

**Experiment 37 — Clock skew.**
- Reproduce the condition represented by: latency is negative or implausibly high.
- Hypothesis to challenge: producer or consumer clocks are unsynchronized.
- Alternative explanation to retain: event timestamp represents a different business time.
- Collect the discriminating evidence: compare synchronized host clocks and timestamp definitions.
- Exercise the proposed control: repair time synchronization or change the metric.
- Acceptance observation: known test events produce plausible stage latencies.
- Before rollout, document the prevention control: monitor clock offset and preserve multiple timestamps.

**Experiment 38 — Disk pressure.**
- Reproduce the condition represented by: broker fetch or produce latency rises with disk utilization.
- Hypothesis to challenge: retention or replication consumed capacity.
- Alternative explanation to retain: a replica or disk has failed.
- Collect the discriminating evidence: compare per-broker storage, replica health and disk errors.
- Exercise the proposed control: restore safe capacity without deleting required history.
- Acceptance observation: disk headroom and replica health recover.
- Before rollout, document the prevention control: model replicated bytes and retain operational reserve.

**Experiment 39 — Unbounded retry queue.**
- Reproduce the condition represented by: memory grows while throughput falls.
- Hypothesis to challenge: retries are enqueued without capacity limits.
- Alternative explanation to retain: the main processing queue is genuinely growing.
- Collect the discriminating evidence: compare retry queue size and age with main in-flight work.
- Exercise the proposed control: bound retry queues and apply backpressure.
- Acceptance observation: memory stays bounded and valid events progress.
- Before rollout, document the prevention control: set retry budgets, jitter and dead-letter governance.

**Experiment 40 — Cross-partition order assumption.**
- Reproduce the condition represented by: a customer sees an older state after a newer state.
- Hypothesis to challenge: related events use different partitions.
- Alternative explanation to retain: sink applies parallel updates without version checks.
- Collect the discriminating evidence: compare entity key, partition, event version and sink write order.
- Exercise the proposed control: use a common key or conditional versioned updates.
- Acceptance observation: adversarial ordering tests preserve correct state.
- Before rollout, document the prevention control: define entity ordering and migration contracts.

**Experiment 41 — Group identity mistake.**
- Reproduce the condition represented by: a deployment unexpectedly reprocesses or shares work.
- Hypothesis to challenge: a new group id has no prior checkpoint.
- Alternative explanation to retain: two applications accidentally share a group id.
- Collect the discriminating evidence: compare configured group ids and expected ownership.
- Exercise the proposed control: restore the intended group only after checking offset consequences.
- Acceptance observation: expected members and offsets are visible in the intended group.
- Before rollout, document the prevention control: treat group ids as versioned deployment configuration.

**Experiment 42 — Bad autoscaling signal.**
- Reproduce the condition represented by: replica count rises without improved throughput.
- Hypothesis to challenge: scaling uses total lag without partition limits.
- Alternative explanation to retain: sink capacity has already saturated.
- Collect the discriminating evidence: compare per-partition rates, assignment and sink capacity.
- Exercise the proposed control: cap scaling and address the bottleneck.
- Acceptance observation: scale tests improve throughput without increasing errors.
- Before rollout, document the prevention control: combine lag age, net drain and resource saturation.

**Experiment 43 — State rebuild exceeds capacity.**
- Reproduce the condition represented by: a restarted stateful consumer never catches up.
- Hypothesis to challenge: replay competes with live work for limited resources.
- Alternative explanation to retain: required history is absent or state checkpoint is invalid.
- Collect the discriminating evidence: compare replay rate, incoming rate, checkpoint age and log start.
- Exercise the proposed control: isolate rebuild resources or restore a valid checkpoint.
- Acceptance observation: rebuild finishes before retention or freshness deadline.
- Before rollout, document the prevention control: exercise cold-start rebuild before release.

**Experiment 44 — Unexpected side effects on retry.**
- Reproduce the condition represented by: an email or payment occurs twice.
- Hypothesis to challenge: external api lacks idempotency support.
- Alternative explanation to retain: application generates a new key per attempt.
- Collect the discriminating evidence: trace logical operation ids and downstream request ids.
- Exercise the proposed control: stop automatic retries for irreversible effects until safeguarded.
- Acceptance observation: crash and retry tests cause one intended effect.
- Before rollout, document the prevention control: use stable provider keys or an outbox with reconciliation.

**Experiment 45 — CPU-heavy deserialization.**
- Reproduce the condition represented by: fetch is fast but durable completion is slow.
- Hypothesis to challenge: decoding or validation dominates cpu.
- Alternative explanation to retain: garbage collection or downstream calls dominate.
- Collect the discriminating evidence: profile cpu alongside allocation and external call latency.
- Exercise the proposed control: optimize the measured hot path without changing semantics.
- Acceptance observation: cpu per event falls and outputs remain equivalent.
- Before rollout, document the prevention control: benchmark realistic schemas and payload sizes.

**Experiment 46 — Broker imbalance.**
- Reproduce the condition represented by: partitions on one broker show correlated fetch delays.
- Hypothesis to challenge: leader placement concentrates hot partitions.
- Alternative explanation to retain: a single network path or storage device is degraded.
- Collect the discriminating evidence: map leaders to broker resources and compare affected clients.
- Exercise the proposed control: rebalance leaders or repair the constrained broker.
- Acceptance observation: per-broker latency and partition drain converge.
- Before rollout, document the prevention control: capacity-plan leader placement and failover.

**Experiment 47 — DLQ silently loses data.**
- Reproduce the condition represented by: lag falls but records vanish from business output.
- Hypothesis to challenge: failed records were sent to an unmonitored dead-letter queue.
- Alternative explanation to retain: a commit occurred despite failed dlq publication.
- Collect the discriminating evidence: trace failure id through dlq durable write and source commit.
- Exercise the proposed control: restore durable dlq publication and alert owners.
- Acceptance observation: every rejected id is accounted for and replayable.
- Before rollout, document the prevention control: define dlq retention, access, replay and reconciliation.

**Experiment 48 — Partial database transaction.**
- Reproduce the condition represented by: one event updates only some related rows.
- Hypothesis to challenge: sink writes are not atomic.
- Alternative explanation to retain: a later compensating operation changed a row.
- Collect the discriminating evidence: inspect transaction boundaries and audit timestamps.
- Exercise the proposed control: use atomic transaction or explicit compensating workflow.
- Acceptance observation: reconciliation finds no partial logical operation.
- Before rollout, document the prevention control: test failure between every related mutation.

**Experiment 49 — Rebalance during in-flight work.**
- Reproduce the condition represented by: a new owner and old owner both perform an effect.
- Hypothesis to challenge: old work continued after revocation.
- Alternative explanation to retain: producer sent two distinct records for the same operation.
- Collect the discriminating evidence: compare assignment epochs, event ids and sink timestamps.
- Exercise the proposed control: fence stale work and use sink idempotency.
- Acceptance observation: ownership changes do not duplicate logical effects.
- Before rollout, document the prevention control: test revocation under slow sink calls.

**Experiment 50 — Recovery estimate misses arrivals.**
- Reproduce the condition represented by: observed drain is far slower than forecast.
- Hypothesis to challenge: estimate divided backlog by gross service.
- Alternative explanation to retain: traffic or service rate changed during recovery.
- Collect the discriminating evidence: compare time-series of arrivals, completions and backlog.
- Exercise the proposed control: recalculate with net service and update incident forecast.
- Acceptance observation: predicted and observed net drain agree within stated uncertainty.
- Before rollout, document the prevention control: publish assumptions and sensitivity with estimates.

### 8.52 Recovery exercise: checkpoint and ownership timeline

Use an isolated topic with one partition and deterministic logical event identifiers.
Produce five records with IDs A through E and retain the producer acknowledgments.
Record the broker offsets assigned to all five records rather than assuming their values.
Start a group with automatic offset advancement disabled in the test configuration.
Arrange for A and B to complete their database mutations atomically with deduplication IDs.
Hold C at the sink while allowing D to complete; keep E unprocessed.
Observe that D's completion does not make it safe to commit beyond C.
Record the safe next offset as the actual offset of C in this test.
Terminate the consumer after D is durable but before C completes.
Restart the same group with the same isolated database and deduplication table.
Confirm the replay reads C and may also reread D depending on the checkpoint.
Verify the durable database contains exactly one effect for each completed logical ID.
Release C, complete E and advance the frontier only through the completed sequence.
Compare the final IDs and effect amounts against the original five-record input ledger.
Repeat with ownership revoked rather than a process crash and inspect stale in-flight work.
This exercise tests a proposed correctness invariant; it is not a reported test execution.

### 8.53 Recovery exercise: resource constraints

Create four partitions and drive a known synthetic rate into each partition.
Measure source ingress, durable sink completions and offset positions over aligned intervals.
Use distinct keys to produce a deliberately skewed partition without changing record sizes.
Measure the hot partition separately from the sum over all partitions.
Add a consumer while holding the number of partitions and sink capacity fixed.
Observe whether ownership changes and whether hot-partition completion changes.
Then restore the original membership and introduce a controlled downstream latency increase.
Measure queue growth, in-flight work, completion latency and the rate of retries.
Reduce the load before memory exhaustion; never use production data for destructive injection.
Compare measured net drain after release against the constant-rate equation in Section 5.
Explain discrepancies using measured rate changes, batching, gaps and shared-resource contention.
A numerical forecast is only credible over the interval where its assumptions hold.

**Takeaway:** Correlate a discriminating signal with a bounded change and a correctness check.

## 9. Complexity & Cost Analysis

Let P be partitions, N group members, K in-flight records per member, and R records processed.
A member can own multiple partitions, but a partition has one active owner in a group.
The number of concurrently owned partitions is at most min(P, N) under normal assignment.
This is a concurrency bound, not a throughput prediction or guarantee of balanced load.
A sequential handler's effective throughput depends on record cost, batching, CPU, and sink latency.
An asynchronous handler can overlap latency, but K increases memory and replay exposure.
A completion frontier with a queue and completion flags uses O(K) memory.
Over R records it performs O(R) insertions and removals under an appropriate queue implementation.
Retries, external I/O and coordination are outside that local operation count.
Large fetches may improve amortization but increase memory and poll-cycle processing time.
Frequent commits reduce potential replay work but add coordinator traffic.
Infrequent commits increase potential duplicate reprocessing after failure.
Payload-only storage is not provisioned disk: include replica count, indexes, compression and reserve.
Capacity estimates need failure-state headroom, not only normal steady-state utilization.
A sink quota shared across consumers can turn added parallelism into a retry storm.
Repartitioning adds migration, operational coordination and possible downstream state costs.
For a replay group, account for broker read bandwidth and possible competition with live readers.
Use per-partition age, bytes, completion rates and business reconciliation alongside offset distance.

**Takeaway:** Concurrency, memory, replay exposure and downstream capacity must be sized together.

## 10. Key Theorems / Results Reference

### 10.1 Single-owner bound

Under group partition assignment, a partition has at most one active owner in that group at a time.
This bounds parallel ownership by the smaller of group membership and partition count.
It does not prohibit application-level concurrent processing inside the owning consumer.
It does not guarantee that concurrent side effects preserve partition order.
Ownership transitions can overlap with stale application work unless that work is controlled or fenced.

### 10.2 Safe frontier invariant

If only a contiguous completed prefix is committed, a committed offset does not skip unfinished earlier work.
This depends on completion meaning that every required effect is durable.
It does not alone prevent duplicate effects after crash and replay.
Idempotency requires stable identity and an atomic check-and-effect or equivalent sink guarantee.
A Kafka transaction can coordinate Kafka records and offsets within its documented scope.
It does not automatically make arbitrary external database or API effects exactly once.

### 10.3 Recovery feasibility

A positive net drain rate is necessary for a finite catch-up under continuing constant arrivals.
It is not sufficient if required source records disappear or the sink cannot sustain the rate.
Even zero offset lag does not establish end-to-end business correctness or freshness.
These are logical consequences of the stated model, not measured Kafka benchmarks.

**Takeaway:** Ownership, safe checkpoints, and replay-safe effects are separate necessary design concerns.

## 11. Interview / Exam Questions

### 11.1 What does ordering mean in Kafka?

Ordering is defined within a partition's append sequence, not across the entire topic.
To preserve a business entity's order, related records must follow a stable partitioning contract.
Concurrent consumers or external sinks must also avoid reordering effects for that entity.
Changing partition count or key serialization can break the continuity of that contract.
A test must include producer retry behavior, migration, consumer concurrency, and sink writes.

### 11.2 Why do extra consumers fail to solve one hot partition?

One active member owns a partition in a group, so more members cannot split its log ownership.
A hot partition may still benefit from safe internal concurrency when its effects permit it.
Alternatively, redesigning the key or work can distribute future records across partitions.
Neither remedy is safe if it violates per-key ordering or relies on a saturated common sink.

### 11.3 What is a committed offset?

It is the group's stored next position for a partition, not the last processed record.
If offset 41 is the last completed item in a contiguous prefix, commit next offset 42.
A commit does not prove an external write succeeded; the application must enforce that relationship.
After restart, replay can include already completed effects that were not yet checkpointed.

### 11.4 How do you commit after out-of-order completions?

Track the completion state of every in-flight record in observed partition order.
Advance the frontier only past the longest prefix whose required effects are durable.
If 10 and 12 complete while 11 does not, the safe next offset remains 11.
Once 11 completes, the frontier may move to 13; cap in-flight work to bound memory.

### 11.5 What is consumer lag, and why is it imperfect?

It is an offset distance between a specified end boundary and a group checkpoint.
Different metrics may use different end boundaries and isolation modes.
Offset gaps mean it need not equal a count of readable business events.
It also does not reveal event age, sink correctness, or a hot partition hidden in totals.

### 11.6 How do you calculate recovery time?

With 360,000 records backlogged, 900 arriving per second, and 1,200 completed per second,
net drain is 300 records/s and recovery takes 1,200 seconds, or 20 minutes.
This assumes fixed rates, a durable completion definition and retained source data.
If service is at most arrivals, the constant-rate model has no finite catch-up time.

### 11.7 Can a consumer provide exactly-once effects in a database?

Not by committing a Kafka offset alone, because a crash can occur between two systems' writes.
Use an atomic database transaction that deduplicates a stable event ID with the business mutation.
After replay, the database recognizes the ID and avoids applying the operation again.
The deduplication lifetime must cover the entire supported replay horizon.
External APIs need their own idempotency contract or a compensating/reconciliation strategy.

### 11.8 What happens if a sink succeeds and the process crashes before commit?

The group can replay the same source record after restart.
An idempotent sink should recognize the original logical operation and not duplicate its effect.
This is normally preferable to committing first and silently losing an unfinished effect.
Exercise this exact crash point with a stable ID and inspect the final sink state.

### 11.9 When would you increase partition count?

Consider it when independent parallelism is the bottleneck and the ordering contract can migrate.
First examine hot-key concentration, sink capacity, assignment, broker resources and client limits.
More partitions do not redistribute old log entries or necessarily split a single key.
Plan key remapping, state migration, cutover ordering, client compatibility and rollback explicitly.

### 11.10 How do you distinguish a hot key from a slow consumer?

Compare per-partition ingress and byte rates, key concentration, ownership and completion latency.
A concentrated ingress skew with normal host behavior supports a hot-key hypothesis.
Several partitions on one member slowing together suggests a member or shared-resource issue.
Correlate these signals over the same window, then validate with a bounded intervention.

### 11.11 Why can zero lag coexist with missing records?

The application may have committed offsets before its external side effects became durable.
A dead-letter path may also accept failures without an adequate reconciliation process.
Compare source IDs and durable sink effects; include rejected and quarantined IDs.
Never treat a zero checkpoint distance as an end-to-end correctness certificate.

### 11.12 How do retention and replay interact?

A replay can only read history still available in an appropriate source.
Log start offsets and compaction policy determine whether the required records remain addressable.
Estimate outage plus catch-up duration with margin, then verify actual retained history.
If history is missing, escalate recovery from an approved archive or a documented business reconciliation.

### 11.13 What should an incident responder do first?

Record group, topic, partition, offsets, assignment, timestamps, rates and current deployments.
Protect retention and irreversible sink correctness before changing offsets or partitioning.
Separate producer surge, per-partition skew, consumer resource limits and downstream saturation.
Choose a reversible bounded mitigation and verify both net drain and business-effect reconciliation.

### 11.14 How does backpressure affect a lagging pipeline?

A slow sink can fill bounded application queues and slow polling or pause fetches.
Lag may rise at Kafka while the actual bottleneck is farther downstream.
Unbounded queues merely move the backlog into memory and can cause process failure.
Measure each stage's queue, throughput and latency before increasing concurrency.

**Takeaway:** Strong answers connect offset mechanics to durable effects, per-partition rates and recovery constraints.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Kafka partition ownership, durable offset advancement and lag recovery |
| Core formula | Drain time = backlog / (durable service rate − arrival rate), when rates are constant and service exceeds arrivals |
| Key assumption | Backlog and rates describe the same partition, interval, unit and durable completion boundary |
| Complexity | For R records and at most K in-flight records per member, contiguous-completion tracking takes O(R) operations and O(K) state under the stated tracker |
| Use when | Diagnosing partition-level skew, processing lag or replay safety |
| Avoid when | Offset distance is treated as an exact record count without accounting for gaps or visibility |
| Go deeper | [Collection index](INDEX.md); verify client and broker semantics for the deployed version |
| Diagnostic | Compare per-partition arrivals, durable completions, oldest age, assignment and downstream saturation |
| Evidence | Examples are illustrative; incident claims remain unverified |
