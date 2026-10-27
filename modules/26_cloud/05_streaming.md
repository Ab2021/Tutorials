# Streaming

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 05 of 09
> Estimated study time: 4 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A streaming platform is the part of a clinical data system that keeps working while everything
else is changing. Events arrive continuously, consumers restart, partitions are reassigned, schemas
evolve, and the platform is expected to deliver each event once, in order per patient, without
loss. Most of that expectation is not delivered by the broker. It is delivered by the producer
configuration, the consumer's commit discipline, the deduplication store, and the schema policy
working together.

The reference implementation in `labs/cloud/kafka/` is deliberately small. It has a Pydantic schema
in `schemas.py`, an idempotent producer in `producer.py`, and a consumer with manual commits,
SQLite deduplication, and a dead-letter topic in `consumer.py`. It runs against a single Redpanda
broker defined in `docker-compose.yml`. That is enough to demonstrate the mechanics and not enough
to be production.

### 1.1 The gap between "it works" and "it is correct"

A producer that sends events and a consumer that reads them will pass a demo. The failures appear
under conditions the demo does not create:

| Condition | Demo behaviour | Production behaviour |
|---|---|---|
| Broker restarts | Not exercised | Producer retries, possible duplicates |
| Consumer restarts mid-batch | Not exercised | Offsets committed after processing or not at all |
| Two consumers join | Not exercised | Rebalance, partition reassignment, possible reprocessing |
| A record is malformed | Not exercised | Consumer crashes or dead-letters |
| Schema adds a field | Not exercised | Old consumers reject or ignore |
| Disk fills | Not exercised | Retention deletes segments, consumers fall behind |
| A patient generates 40 percent of events | Not exercised | One partition hot, others idle |

Each row maps to a design decision in Section 5.

### 1.2 Why the biomedical case is stricter than the general case

Clinical events are state transitions, not independent messages. An admission followed by a
transfer followed by a discharge is a sequence that must be applied in order for a patient's
location to be correct. Keying by `patient_id` gives per-patient ordering within a partition, which
is the strongest guarantee a partitioned log can offer without a single partition. A single
partition would serialise the entire stream and cap throughput at one consumer.

The second difference is audit. A pipeline that processes an event and discards it cannot answer
"what did we know at 14:32 on 3 October". A retained log can. That makes retention a clinical and
regulatory parameter, not a capacity parameter.

### 1.3 What the lab does and does not prove

`labs/cloud/docker-compose.yml` runs one Redpanda broker with `--smp 1` and `--memory 1G`. The
producer creates the topic with three partitions and replication factor one. That configuration
survives no broker failure and provides no redundancy. A production configuration uses at least
three brokers across three availability zones, replication factor three, `min.insync.replicas` of
two, TLS, authentication, and a schema registry. Section 5 states each change.

> **Note:** The lab stack is a teaching scaffold. It is not a compliant deployment and the sample
> events contain synthetic identifiers only.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** how a topic, partition, offset, and consumer group combine to give ordering and
   parallelism, and state the limits of each.
2. **Configure** a producer for durable, idempotent delivery and explain what each setting
   guarantees and does not.
3. **Design** a consumer commit and deduplication strategy that is safe under replay and under
   rebalance.
4. **Size** partitions, retention, and broker storage for a stated event rate, event size, and
   replay requirement.
5. **Diagnose** consumer lag from the difference between throughput limits and processing limits.
6. **Choose** a schema compatibility mode and an upgrade order for producers and consumers.
7. **Distinguish** at-most-once, at-least-once, and effectively-once, and state which the lab
   provides.

Outcomes 1 and 4 are applied in Exercises 12.2 and 12.3. Outcomes 2 and 3 in 12.4 and 12.5.
Outcomes 5 and 6 in 12.6 and 12.7. Outcome 7 in 12.8.

## 3. The core mental model

A topic is a set of partitions. Each partition is an append-only log with a monotonically increasing
offset. A record is assigned to a partition by the producer, usually by hashing its key. A consumer
group divides partitions among its members, and each partition is read by exactly one member of the
group at a time.

```text
                    topic: adt.events (3 partitions)
      ┌─────────────────────────────────────────────────────┐
      │ partition 0:  [0][1][2][3][4][5] ...                 │
      │ partition 1:  [0][1][2][3] ...                       │
      │ partition 2:  [0][1][2][3][4][5][6] ...              │
      └─────────────────────────────────────────────────────┘
             ▲                    │
             │ key hash           │ consumer group "biomed-adt-consumer"
             │                    ▼
      producer(key=patient_id)   ┌───────────┐  ┌───────────┐
                                 │ consumer A│  │ consumer B│
                                 │ p0, p1    │  │ p2        │
                                 └───────────┘  └───────────┘
```

Ordering is guaranteed within a partition, not across a topic. Two events for the same patient go to
the same partition and therefore stay ordered. Two events for different patients may be processed in
any order.

### 3.1 Offsets are the consumer's responsibility

The broker does not track what a consumer has processed. It tracks what the consumer has committed.
The gap between the last committed offset and the log end offset is the lag.

```text
partition 1:
  committed offset ──▶ 42
  log end offset   ──▶ 58
  lag              ──▶ 16 records
```

If a consumer processes records 43 to 58 and crashes before committing, another consumer starts at
43 and reprocesses. That is at-least-once. If the consumer commits 58 before processing, and then
crashes, records are lost. That is at-most-once. The lab commits after processing, which is the
correct choice for clinical events.

### 3.2 The three delivery guarantees

| Guarantee | Mechanism | Duplicates | Loss | Typical implementation |
|---|---|---|---|---|
| At-most-once | Commit before processing | None | Possible | Auto-commit before handling |
| At-least-once | Commit after processing | Possible | None | Manual commit after the write |
| Effectively-once | At-least-once plus idempotent writes | None observable | None | Deduplication key, transactional sink |

"Exactly-once" is often used for the third row. The distinction matters because the broker can only
provide exactly-once within its own boundary, using transactions. Once the consumer writes to an
external database, the guarantee is a property of the write, not of the broker.

### 3.3 The rebalance is a pause, not a failure

When a consumer joins or leaves, the group rebalances. In the eager protocol, every consumer revokes
all partitions, and processing stops until the new assignment is complete. In the cooperative
protocol, consumers keep partitions they still own and revoke only what moved.

```text
eager:        revoke all ──▶ reassign ──▶ resume        (stop-the-world)
cooperative:  revoke moved ──▶ assign ──▶ continue      (incremental)
```

A rebalance during a large batch means the batch is reprocessed. That is why `max.poll.interval.ms`
exists: a consumer that takes longer than the interval between polls is considered dead, and the
group rebalances without it.

### 3.4 Retention is a contract with the future

Retention determines how far back a replay can go. It is expressed as time, size, or both, and it
applies per segment. A segment is closed when it reaches a size or age threshold, and only closed
segments are eligible for deletion. A topic with a one-hour retention and a one-day segment age has
an effective minimum retention of about one day.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Topic | Named stream of records | A queue | Name |
| Partition | Ordered, append-only slice of a topic | A replica | Integer |
| Offset | Position of a record in a partition | A timestamp | 64-bit integer |
| Log end offset | Offset of the next record to be written | Last record offset | Integer |
| Committed offset | Offset up to which a group has acknowledged | Processed offset | Integer |
| Consumer group | Consumers dividing partitions among themselves | A subscription | Group identifier |
| Rebalance | Reassignment of partitions to consumers | A failure | Event |
| Assignor | Strategy that maps partitions to consumers | A broker setting | Class name |
| Producer | Client that writes records | A broker | Client |
| Consumer | Client that reads records | A subscriber | Client |
| Idempotent producer | Producer that suppresses duplicates from retries within a session | Exactly-once | Boolean |
| Transactional producer | Producer that writes atomically across partitions | Idempotent producer | `transactional.id` |
| Acknowledgment | Broker confirmation of a write | Delivery to the consumer | `acks` setting |
| In-sync replica | Replica that has caught up with the leader | Any replica | Count |
| Minimum in-sync replicas | Minimum replicas that must acknowledge a write | Replication factor | Count |
| Replication factor | Number of copies of each partition | Durability guarantee | Count |
| Retention | How long records remain available | Backup | Time or bytes |
| Segment | File containing a range of offsets | A partition | Bytes |
| Compaction | Retention by key, keeping the latest value | Deletion | Policy |
| Schema registry | Service storing and validating schemas | A validation library | Service |
| Compatibility mode | Which schema changes are accepted | A version number | Enum |
| Dead-letter queue | Destination for unprocessable records | An error log | Topic |
| Poison record | Record that always fails processing | A slow record | Record |
| Lag | Difference between log end and committed offset | Latency | Records |
| Watermark | Bound on acceptable event lateness | A timeout | Event time |
| Event time | Time the event occurred | Ingest time | Timestamp |
| At-least-once | Delivery with possible duplicates | Exactly-once | Guarantee |
| Effectively-once | At-least-once plus idempotent writes | Broker transaction | System property |
| `acks=all` | Leader waits for all in-sync replicas | All replicas | Producer setting |
| `enable.idempotence` | Producer suppresses retry duplicates | Transaction | Producer setting |
| `isolation.level` | Whether consumers see uncommitted transactional records | Consistency level | Enum |
| Tiered storage | Offloading old segments to object storage | Archiving | Feature |

Three distinctions carry the operational weight.

**Lag is not latency.** A consumer with zero lag can still take ten seconds to process the next
record. A consumer with high lag may still meet its latency SLO if the backlog is drained in a burst.

**Idempotence is per producer session.** A producer with `enable.idempotence=true` suppresses
duplicates caused by its own retries. A new producer instance gets a new producer identifier and
cannot deduplicate against the old one unless it uses a transactional identifier.

**Compaction is not retention.** A compacted topic keeps the last record per key indefinitely. A
delete-policy topic removes records after the retention period. Using compaction for an event stream
loses history.

## 5. The mechanics, step by step

The procedure builds a stream that is ordered per patient, durable across a broker failure,
idempotent at the consumer, and replayable. It follows the reference implementation and states where
production differs.

### Step 1 — Model the stream before configuring it

**Input.** The source system, the event types, and the questions the stream must answer.

**Output.** A list of event types, the entity each belongs to, the key, and the ordering requirement.

**Rationale.** The key determines ordering and distribution. Choosing it after the topic exists is
expensive. For admission, discharge, and transfer events, the entity is the patient, the key is the
patient identifier, and the ordering requirement is per patient. For device telemetry, the entity is
the device and the key is the device identifier.

**Branch point.** If the ordering requirement is global, a single partition is the only way to
satisfy it, and throughput is capped at one consumer. Most clinical streams do not need global
ordering.

**Edge case.** A patient identifier that changes across a merger or a correction breaks ordering.
The stream needs a stable surrogate identifier, assigned before the event is produced.

### Step 2 — Choose partitions and replication factor

**Input.** Peak event rate, per-partition capacity, consumer parallelism, and availability
requirement.

**Output.** A partition count and a replication factor.

**Rationale.** Partitions serve two purposes: they shard storage and throughput, and they cap
consumer parallelism. A topic with three partitions can use at most three consumers in a group.
Throughput sizing usually produces a small number; parallelism and headroom usually produce a larger
one.

**Decision table.** More than two outcomes, so the choice is tabulated.

| Requirement | Partitions | Replication | Minimum in-sync | Rationale |
|---|---|---|---|---|
| Local demo | 3 | 1 | 1 | No redundancy, fast |
| Single-zone production | 12 | 2 | 1 | Tolerates one broker, no zone loss |
| Multi-zone production | 12 | 3 | 2 | Tolerates one broker or zone |
| High consumer parallelism | 48 | 3 | 2 | Allows 48 consumers |
| Strict global ordering | 1 | 3 | 2 | Throughput capped |

The lab uses three partitions and replication factor one in
`ensure_topic(... NewTopic(topic, num_partitions=3, replication_factor=1))`. That is the first row.

**Edge case.** Increasing partitions later changes the key-to-partition mapping. Events for the same
patient produced before and after the change can land in different partitions and be processed out
of order relative to each other. Plan a pause or a dual-write window.

### Step 3 — Define the schema and the compatibility policy

**Input.** The event fields, their types, and the expected rate of change.

**Output.** A schema with an explicit version, and a compatibility mode enforced by a registry.

**Rationale.** `labs/cloud/kafka/schemas.py` defines `AdtEvent` with `event_id`, `patient_id`,
`event_type` as a literal of `admit`, `discharge`, or `transfer`, `event_time`, `site_id`, an
optional `ward`, and a `schema_version` defaulting to `"1.0"`. Pydantic validates types at the
producer and at the consumer, which catches malformed records before they reach the processing
logic.

A header or field carrying the version is necessary but not sufficient. Compatibility is enforced
by a registry that rejects an incompatible schema at registration time.

**Branch point.** The schema registry's default compatibility mode is `BACKWARD`, meaning new
consumers can read data written with the previous schema. That supports upgrading consumers first.
`FORWARD` supports upgrading producers first. `FULL` requires both, and the `_TRANSITIVE` variants
check all previous versions rather than only the latest.

**Edge case.** A new required field without a default is a breaking change under every mode. Make it
optional with a default, or release it in two steps: add optional, deploy consumers, then require.

### Step 4 — Configure the producer for durability

**Input.** The durability requirement and the latency budget.

**Output.** Producer settings: `enable.idempotence`, `acks`, `retries`,
`max.in.flight.requests.per.connection`.

**Rationale.** `labs/cloud/kafka/producer.py` sets:

```python
Producer({
    "bootstrap.servers": args.bootstrap,
    "enable.idempotence": True,
    "acks": "all",
    "retries": 5,
    "max.in.flight.requests.per.connection": 5,
})
```

`enable.idempotence` assigns a producer identifier and a sequence number per partition, and the
broker rejects a duplicate sequence. `acks=all` waits for all in-sync replicas. The in-flight limit of
five is the maximum that preserves ordering with idempotence enabled.

Since Kafka 3.0, `enable.idempotence` defaults to true and `acks` defaults to `all`. The lab sets
them explicitly, which is correct for a teaching example because the reader needs to see the
contract.

**Branch point.** If latency matters more than durability, `acks=1` reduces the wait at the cost of
losing records when the leader fails before replication. For clinical state transitions that is
usually unacceptable.

**Edge case.** `retries=5` is lower than the idempotent producer's default of effectively unlimited.
A transient broker outage longer than five retries produces a delivery failure that the callback
reports. The application must decide whether to retry the batch, and that decision needs an
idempotency key.

### Step 5 — Key by the entity, and check the distribution

**Input.** The key chosen in Step 1 and a sample of real keys.

**Output.** A key expression and an expected partition distribution.

**Rationale.** Keying by `patient_id` preserves per-patient order. A hash of the key selects the
partition. A skewed key concentrates load: a long-stay patient with many transfers, or a device that
emits at a higher rate, can make one partition the bottleneck while others are idle.

**Branch point.** If skew is severe, salt the key with a bounded suffix and accept that ordering is
preserved only within a salt bucket. Reconstruct order downstream using event time and a sequence
number.

**Edge case.** A null key sends records round-robin across partitions, which gives distribution and
no ordering. Some producers allow null keys by default; a clinical stream should reject them.

### Step 6 — Define the consumer's processing contract

**Input.** The output system and the processing logic.

**Output.** A written statement of what "processed" means, and what the consumer does on failure.

**Rationale.** The commit must happen after the effect is durable. In `consumer.py`, the effect is
the insertion into the SQLite `processed` table, followed by `consumer.commit(message=msg,
asynchronous=False)`. The synchronous commit ensures the offset is acknowledged before the next
record, which is slower and safer.

**Branch point.** If the output is a database, the deduplication insert and the output write should
be in the same transaction. The lab uses one SQLite database, so they can be. A production design
with a separate output database needs either a distributed transaction, which is usually a bad
trade, or an idempotent output write keyed by `event_id`.

### Step 7 — Choose the commit strategy

**Input.** The processing contract and the acceptable duplicate rate.

**Output.** Auto-commit disabled, manual commit after processing, and a maximum batch size.

**Rationale.** `consumer.py` sets `enable.auto.commit: False` and commits explicitly. Auto-commit
commits offsets on a timer regardless of whether processing succeeded, which produces at-most-once
behaviour under failure.

**Edge case.** Committing per record is slow at high throughput. Batch commits are acceptable if the
processing is idempotent, because a crash between the last processed record and the commit causes
reprocessing, not loss.

### Step 8 — Deduplicate with a stable key

**Input.** The event identifier and the output store.

**Output.** A uniqueness constraint or a deduplication table.

**Rationale.** `consumer.py` creates `processed(event_id TEXT PRIMARY KEY, processed_at TEXT)` and
treats `sqlite3.IntegrityError` as evidence of a duplicate. A duplicate is committed and skipped.
This makes a replay safe.

**Branch point.** A local SQLite file works for one consumer. With multiple consumers, each has its
own file and cannot see the others' duplicates. A production design uses a shared store with a
unique constraint, or a transactional output that is naturally idempotent, such as an upsert on a
natural key.

**Edge case.** Deduplication by content hash fails when two genuine events are byte-identical, which
happens with periodic device readings. Use a source-assigned event identifier or a device sequence
number.

### Step 9 — Route poison records to a dead-letter topic

**Input.** The validation rules and the dead-letter topic.

**Output.** A dead-letter producer and a documented triage procedure.

**Rationale.** In `consumer.py`, a `ValidationError` produces the raw record to
`adt.events.dlq` with an `error` header, and then commits the original offset. Committing is
essential: without it, the consumer retries the poison record forever and the partition stalls.

**Branch point.** If the dead-letter write fails, the consumer must not commit. Otherwise the record
is lost. The lab calls `dlq.flush(5)` before committing, which is the right order.

**Edge case.** A dead-letter topic with no retention limit grows without bound. Set a retention and
alert on depth and on the age of the oldest record.

### Step 10 — Handle rebalance without duplication storms

**Input.** The consumer configuration and the expected processing time per record.

**Output.** `max.poll.interval.ms`, `session.timeout.ms`, `heartbeat.interval.ms`, and an assignor
choice.

**Rationale.** Kafka's defaults for the classic consumer protocol are `session.timeout.ms` 45,000,
`heartbeat.interval.ms` 3,000, and `max.poll.interval.ms` 300,000. A consumer that processes a batch
for longer than the poll interval is removed from the group, its partitions are reassigned, and the
batch is reprocessed elsewhere. If the processing time is consistently near the limit, the group
rebalances repeatedly and lag grows even though every consumer is busy.

**Branch point.** `CooperativeStickyAssignor` reduces the scope of each rebalance. It requires a
rolling restart to change the assignment strategy, because the group must agree on the protocol.

**Edge case.** A very long single-record processing time, such as a model inference call, should be
moved off the poll thread. Polling must continue even while work is in progress.

### Step 11 — Size retention and broker storage

**Input.** Daily volume, retention requirement, replication factor, and disk per broker.

**Output.** Retention settings and broker disk size.

**Rationale.** Retained bytes equal daily bytes times retention days times replication factor. Disk
must hold that plus headroom for segment rolls, consumer lag, and compaction. A broker that fills its
disk goes offline, and an offline broker in a three-replica partition reduces the in-sync set.

Apache Kafka's `log.retention.hours` defaults to 168, which is seven days, and `log.segment.bytes`
defaults to 1 GiB. Redpanda's topic-level `retention.ms` defaults to 604,800,000 ms, also seven days,
inherited from the cluster's `log_retention_ms`. The two systems agree on the default period and
differ in how segment settings interact with it.

**Edge case.** Retention applies to closed segments. A low-volume topic with a long segment age can
retain data far beyond the nominal period. If storage must be tightly bounded, set segment age as
well as retention.

### Step 12 — Instrument lag and define the SLO

**Input.** The consumer group and the processing SLO.

**Output.** Lag metrics per partition, an alert threshold, and a latency indicator.

**Rationale.** Lag is the queue length the consumer has not yet drained. Alert on lag in records and
on the age of the oldest unprocessed record. The second is closer to the clinical impact: 10,000
records of lag is harmless if the consumer processes 50,000 per second and serious if it processes
100.

**Branch point.** If the SLO is event-to-visible latency, lag alone is insufficient. Instrument the
event timestamp and the serving timestamp and compute the difference.

### Step 13 — Plan replay and schema migration

**Input.** The retention window and the deployment process.

**Output.** A replay procedure and an upgrade order that matches the compatibility mode.

**Rationale.** Replay uses a separate consumer group reading from an earlier offset, writing to a
shadow output. It does not reset the production group's offsets, because that would reprocess into
the live output. The upgrade order depends on the compatibility mode: `BACKWARD` allows consumers
first, `FORWARD` allows producers first, `FULL` allows either.

**Edge case.** A replay at full speed can saturate the output database. Rate-limit it and monitor the
output's own SLO during the replay.

### 5.1 Edge cases that change the answer

**Partition rebalance during a long batch.** The consumer loses its partitions, another consumer
reprocesses the batch, and the deduplication store absorbs the duplicates. Without deduplication,
the output has duplicate rows and no way to tell them apart.

**Consumer lag with idle CPU.** The consumer is not CPU-bound; it is waiting, rebalancing, or blocked
on a downstream call. Adding consumers does not help if the partition count caps parallelism, and
does not help if the cause is a rebalance loop.

**Duplicate delivery with distinct identifiers.** The producer generates a new `event_id` on retry.
No downstream deduplication can work, because the two records genuinely look different. The fix is
at the producer: derive the identifier from the source event, not from the attempt.

**Schema evolution with an old consumer.** A new field is added and an old consumer ignores it,
which is backward compatible. A field is removed and an old consumer fails, which is not. The
registry's compatibility check is the gate; without it, the failure appears in production.

**Credential expiry during a replay.** A long replay outlives a short-lived token. The consumer
fails with an authentication error, which must not be treated as a poison record. Distinguish
authorization errors from validation errors in the error handling.

**Quota exhaustion.** A broker disk quota, a topic byte quota, or a client connection quota is
reached. The producer receives a quota-violation error and retries, which looks like latency.
Check broker quota metrics before scaling.

**Zone failure.** With replication factor three across three zones and
`min.insync.replicas=2`, the loss of one zone leaves two replicas and writes continue. With
replication factor two in two zones and a minimum of two, writes stop. The availability outcome is
set by the combination, not by the replication factor alone.

### 5.2 A decision table for consumer failure handling

| Failure | Retryable | Action | Commit offset | Risk if mishandled |
|---|---|---|---|---|
| Deserialization error | No | Dead-letter | Yes | Infinite retry, partition stall |
| Schema validation error | No | Dead-letter | Yes | Same |
| Downstream timeout | Yes | Retry with backoff | No | Data loss if committed |
| Authentication failure | Yes, after refresh | Pause and refresh | No | Poison-lettering an outage |
| Output constraint violation | Depends | Deduplicate or dead-letter | Yes if duplicate | Duplicate rows or stall |
| Out of memory | No | Restart, reduce batch | No | Rebalance loop |

## 6. Worked example

A hospital wants to stream device telemetry from 500 monitored beds. Each bed emits one observation
per second at 200 bytes. The clinical requirement is per-device ordering, a replay window of seven
days, and survival of a single availability zone loss.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Devices | 500 | Bed count |
| Sampling rate | 1 Hz per device | Device configuration |
| Event size | 200 B | Measured JSON |
| Mean event rate | 500 events/s | 500 × 1 |
| Peak factor | 3 | Assumed shift-change burst |
| Retention | 7 days | Incident review policy |
| Replication factor | 3 | Three zones |
| Broker type | `m7g.large`, 2 vCPU, 8 GiB | Candidate |
| Broker price | USD 0.0816/hour | AWS Price List, us-east-2, October 2026 |
| Disk per broker | 500 GB gp3 | Candidate |
| gp3 price | USD 0.08/GB-month | AWS EBS published example rate |
| Consumer service rate | 400 events/s each | Measured, 2.5 ms per event |
| Consumers | 4 | Candidate |
| Per-partition throughput | 2 MB/s at p99 | Conservative planning value |

All costs are USD, us-east-2, for 730 hours.

### 6.2 Arithmetic: throughput and retention

Mean throughput:

```text
500 events/s × 200 B = 100,000 B/s = 0.1 MB/s
```

Peak throughput:

```text
0.1 MB/s × 3 = 0.3 MB/s
```

Daily volume:

```text
0.1 MB/s × 86,400 s = 8,640 MB/day = 8.64 GB/day
```

Seven-day retained volume with replication factor three:

```text
8.64 GB/day × 7 days × 3 = 181.44 GB
```

Thirty-day retained volume:

```text
8.64 × 30 × 3 = 777.6 GB
```

### 6.3 Arithmetic: partitions

Partitions required by throughput at 2 MB/s per partition and 70 percent target utilisation:

```text
ceil(0.3 ÷ (2.0 × 0.7)) = ceil(0.214) = 1 partition
```

Partitions required by consumer parallelism, to run four consumers:

```text
4 consumers → at least 4 partitions
```

The chosen count is 12, which allows growth to twelve consumers and gives headroom for a hot key.
The throughput requirement is not the constraint; the parallelism and ordering requirements are.
That is a common result and worth stating explicitly in a design review, because a reviewer who
computes only throughput will conclude that one partition is enough.

### 6.4 Arithmetic: broker storage and cost

Three brokers with 500 GB each:

```text
total disk = 3 × 500 GB = 1,500 GB
seven-day retention = 181.44 GB = 12.1% of disk
thirty-day retention = 777.6 GB = 51.8% of disk
```

Broker compute and storage:

```text
compute = 3 × USD 0.0816/hour × 730 hours = USD 178.70
storage = 1,500 GB × USD 0.08/GB-month   = USD 120.00
total                                        USD 298.70 per month
```

The disk is sized for thirty days of retention and a comfortable headroom, not for seven days. That
is deliberate: the marginal cost of the extra capacity is USD 120 per month for the whole cluster,
and the alternative is a migration during an incident.

### 6.5 Arithmetic: consumer capacity and lag

Four consumers at 400 events per second each give a service rate of 1,600 events per second. At the
peak arrival rate of 1,500 events per second, utilisation is:

```text
1,500 ÷ 1,600 = 0.9375
```

That is too close to one for comfort. Losing one consumer drops the service rate to 1,200 events
per second, below the peak, and the backlog grows at:

```text
1,500 − 1,200 = 300 events/s
```

After 60 seconds the backlog is 18,000 events. At 200 bytes per event that is 3.6 MB, which is
trivial in storage but represents 60 seconds of unprocessed telemetry at peak. The operational fix
is either a fifth consumer, which requires at least five partitions, or a lower per-event processing
cost.

### 6.6 Arithmetic: availability budget

A 99.9 percent monthly SLO over 30 days:

```text
30 × 24 × 60 = 43,200 minutes
43,200 × 0.001 = 43.2 minutes
```

A single broker loss with replication factor three and `min.insync.replicas=2` should cause no
unavailability. A single zone loss removes one broker and leaves two, which is exactly the minimum.
A second failure during the first is an outage. The SLO is achievable but has no margin for a
second concurrent failure, which is the normal meaning of "three nines".

### 6.7 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: lag rises while consumer CPU is at 15 percent.**

```bash
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group biomed-adt-consumer
```

The output shows `CONSUMER-ID`, `HOST`, `CLIENT-ID`, `#PARTITIONS`, and `LAG`. If the consumer count
equals the partition count and CPU is low, the bottleneck is not compute. Check the group state for
repeated `PreparingRebalance` entries and the consumer log for
`Member ... has left group ... due to consumer poll timeout`. That message means a batch took longer
than `max.poll.interval.ms`. The fix is to lower `max.poll.records` or move the slow work off the
poll loop, not to add consumers.

**Diagnostic 2: the output table has duplicate rows with different offsets.**

Query the serving table for duplicate natural keys:

```sql
SELECT patient_id, event_time, COUNT(*) AS copies
FROM admissions
GROUP BY patient_id, event_time
HAVING COUNT(*) > 1;
```

If the duplicates share an `event_id`, the consumer's deduplication is not running or is scoped to a
local file that another consumer cannot see. If they have different `event_id` values, the producer
is generating a new identifier on retry, and the fix is upstream. The distinction determines which
team owns the incident.

### 6.8 Vary one assumption

Raise retention from seven days to thirty. Retained volume rises from 181.44 GB to 777.6 GB, which
still fits in 1,500 GB. Now add waveform data: 500 devices at 250 samples per second and 2 bytes per
sample:

```text
500 × 250 × 2 = 250,000 B/s = 0.25 MB/s = 21.6 GB/day
```

At thirty days and replication factor three:

```text
21.6 × 30 × 3 = 1,944 GB
```

Combined with telemetry:

```text
777.6 + 1,944 = 2,721.6 GB
```

That exceeds the 1,500 GB of broker disk by 1,221.6 GB. The options are more disk, which costs
`1,221.6 × 0.08 = USD 97.73` per month at gp3 rates and requires a volume resize, or tiered storage,
which moves old segments to object storage at a lower per-gigabyte price. The second option changes
the replay latency for old data and must be stated in the recovery procedure: replaying 30-day-old
data now reads from object storage, which is slower.

## 7. Code, query, or calculation

The following standard-library Python model computes partition assignment, key distribution,
retention, and lag. It reproduces the arithmetic in Section 6. The partitioner uses SHA-256 as a
stand-in for Kafka's murmur2; the distribution properties are similar and the exact assignment
differs.

```python
"""Partition assignment, retention, and lag model for a clinical event stream. Stdlib only."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Stream:
    name: str
    events_per_second: float
    bytes_per_event: int
    peak_factor: float
    retention_days: int
    replication: int
    partitions: int


def partition_for(key: str, partitions: int) -> int:
    """Stand-in for Kafka's murmur2 keyed partitioner.

    Kafka hashes the key bytes with murmur2 and takes the positive modulo of the
    partition count. This uses SHA-256 for portability; it produces a different
    assignment but the same distribution properties.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % partitions


def range_assign(partitions: int, consumers: int) -> dict[int, list[int]]:
    """Kafka RangeAssignor: contiguous ranges, first consumers get the extra."""
    base, extra = divmod(partitions, consumers)
    assignment: dict[int, list[int]] = {}
    start = 0
    for consumer in range(consumers):
        size = base + (1 if consumer < extra else 0)
        assignment[consumer] = list(range(start, start + size))
        start += size
    return assignment


def daily_gb(stream: Stream) -> float:
    return stream.events_per_second * 86_400 * stream.bytes_per_event / 1e9


def retained_gb(stream: Stream) -> float:
    return daily_gb(stream) * stream.retention_days * stream.replication


def peak_mb_per_s(stream: Stream) -> float:
    return stream.events_per_second * stream.peak_factor * stream.bytes_per_event / 1e6


def partitions_needed(stream: Stream, per_partition_mb: float, target: float) -> int:
    return max(1, -(-int(peak_mb_per_s(stream) * 1000) // int(per_partition_mb * target * 1000)))


def lag_projection(arrival_eps: float, service_eps: float, seconds: int) -> float:
    """Backlog growth when arrival exceeds service. Returns events after `seconds`."""
    return max(0.0, (arrival_eps - service_eps) * seconds)


def main() -> None:
    stream = Stream("device.telemetry", 500.0, 200, 3.0, 7, 3, 12)

    print(f"mean rate        {stream.events_per_second:8.2f} events/s")
    print(f"peak rate        {stream.events_per_second * stream.peak_factor:8.2f} events/s")
    print(f"peak throughput  {peak_mb_per_s(stream):8.3f} MB/s")
    print(f"daily volume     {daily_gb(stream):8.3f} GB/day")
    print(f"retained RF=3    {retained_gb(stream):8.3f} GB for {stream.retention_days} days")
    print(f"retained RF=3    "
          f"{retained_gb(Stream('x', stream.events_per_second, stream.bytes_per_event, 3.0, 30, 3, 12)):8.3f} GB for 30 days")
    print(f"partitions for 2 MB/s at 70%: {partitions_needed(stream, 2.0, 0.7)}")

    print()
    print("assignment of 12 partitions to consumers (range)")
    for consumers, assignment in range_assign(12, 4).items():
        print(f"  consumer {consumers}: {assignment}")

    print()
    print("assignment with uneven partition count (7 partitions, 3 consumers)")
    for consumers, assignment in range_assign(7, 3).items():
        print(f"  consumer {consumers}: {assignment}")

    print()
    print("key distribution across 12 partitions, 500 synthetic patients")
    counts = [0] * 12
    for i in range(500):
        counts[partition_for(f"synth-{1000 + i}", 12)] += 1
    print(f"  min={min(counts)} max={max(counts)} mean={sum(counts)/12:.1f}")
    print(f"  counts={counts}")

    print()
    print("lag growth with four consumers at 400 events/s each (service 1,600/s)")
    for arrival in (1_200, 1_500, 1_600, 1_800, 2_400):
        backlog = lag_projection(arrival, 1_600, 60)
        print(f"  arrival={arrival:5d}/s  backlog after 60 s = {backlog:9.0f} events")
    print("lag growth after losing one consumer (service 1,200/s)")
    for arrival in (1_200, 1_500, 1_800):
        backlog = lag_projection(arrival, 1_200, 60)
        print(f"  arrival={arrival:5d}/s  backlog after 60 s = {backlog:9.0f} events")


if __name__ == "__main__":
    main()
```

Actual output:

```text
mean rate          500.00 events/s
peak rate         1500.00 events/s
peak throughput     0.300 MB/s
daily volume        8.640 GB/day
retained RF=3     181.440 GB for 7 days
retained RF=3     777.600 GB for 30 days
partitions for 2 MB/s at 70%: 1

assignment of 12 partitions to consumers (range)
  consumer 0: [0, 1, 2]
  consumer 1: [3, 4, 5]
  consumer 2: [6, 7, 8]
  consumer 3: [9, 10, 11]

assignment with uneven partition count (7 partitions, 3 consumers)
  consumer 0: [0, 1, 2]
  consumer 1: [3, 4]
  consumer 2: [5, 6]

key distribution across 12 partitions, 500 synthetic patients
  min=33 max=50 mean=41.7
  counts=[50, 37, 41, 42, 43, 42, 33, 36, 48, 43, 43, 42]

lag growth with four consumers at 400 events/s each (service 1,600/s)
  arrival= 1200/s  backlog after 60 s =         0 events
  arrival= 1500/s  backlog after 60 s =         0 events
  arrival= 1600/s  backlog after 60 s =         0 events
  arrival= 1800/s  backlog after 60 s =     12000 events
  arrival= 2400/s  backlog after 60 s =     48000 events
lag growth after losing one consumer (service 1,200/s)
  arrival= 1200/s  backlog after 60 s =         0 events
  arrival= 1500/s  backlog after 60 s =     18000 events
  arrival= 1800/s  backlog after 60 s =     36000 events
```

Two observations. The key distribution across 12 partitions is uneven but not pathological: the
busiest partition has 50 of 500 keys, 20 percent above the mean. With a larger patient population
the spread narrows. The lag model is linear and ignores the arrival burstiness; a real backlog grows
with the variance of arrivals and is worse than this estimate at the same mean.

The following is the production change to the reference consumer's deduplication. It is Python, uses
only the standard library, and demonstrates the schema for a shared deduplication table rather than
a local SQLite file. It is a design sketch, not a runnable service.

```python
import sqlite3
from contextlib import contextmanager


@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def process_once(conn: sqlite3.Connection, event_id: str, apply) -> bool:
    """Apply `apply` at most once for `event_id`. Returns True if applied.

    The insert and the output write share one transaction, so a crash between
    them leaves neither. A duplicate event_id raises IntegrityError, which is
    the signal to skip.
    """
    with transaction(conn) as tx:
        try:
            tx.execute("INSERT INTO processed(event_id) VALUES (?)", (event_id,))
        except sqlite3.IntegrityError:
            return False
        apply(tx)
        return True
```

For a multi-consumer deployment, replace SQLite with the output database itself and use a unique
constraint on `event_id`, so the deduplication and the effect are in the same transaction by
construction.

## 8. Data, standards, and regulatory context

### 8.1 HL7 version 2 ADT semantics

The lab's `AdtEvent` models three trigger events: admit, discharge, and transfer. In HL7 version 2
these are A01, A03, and A02 respectively, carried in pipe-delimited messages. The mapping from the
wire format to the event schema is where most data quality defects enter the stream: a missing
`PV1` segment, a ward that does not match the site's vocabulary, or a transfer that is recorded as a
discharge followed by an admit.

The stream design cannot fix those defects. It can preserve them faithfully and make them visible.
A validation layer at the producer rejects records that do not satisfy the schema, and a
dead-letter topic preserves the rest for review.

### 8.2 FHIR R4 Subscription

FHIR R4 defines a `Subscription` resource with a `criteria` search string and a `channel`. For a
`rest-hook` channel, the server either sends an empty POST and the subscriber re-queries, or
forwards the resource when a payload is requested. The criteria are evaluated against the new
resource value, so an update that no longer matches produces no notification, and the base resource
does not notify on delete.

A pipeline that assumes FHIR subscriptions give exactly-once delivery, or that deletions are
notified, is making two assumptions the specification does not support. Both must be handled at the
subscriber.

### 8.3 HIPAA and PHI in motion

A Kafka topic carrying patient identifiers is electronic PHI. The Security Rule's transmission
security standard at 45 CFR 164.312(e)(1) requires protection against unauthorized access during
transmission, and the access control standard at 164.312(a)(1) requires limiting access. In practice:

| Control | Kafka mechanism | Lab status |
|---|---|---|
| Encryption in transit | TLS listeners, TLS for inter-broker | Not configured |
| Encryption at rest | Broker volume encryption, KMS | Not configured |
| Authentication | SASL, mTLS | Not configured |
| Authorization | ACLs per topic and group | Not configured |
| Audit | Broker audit log, client principal logging | Not configured |

The broker software is not "HIPAA certified". The deployment is assessed, and the controls above are
what an assessment examines.

### 8.4 Retention and the audit trail

Retention interacts with the audit obligation. If a topic is the only record of an event, deleting
it after seven days may conflict with a retention policy that requires six years. Most designs
resolve this by landing raw events in object storage, which is cheap and durable, and treating the
topic as a transport with a short retention. The landing object becomes the audit record; the topic
becomes the delivery mechanism. That separation is an architecture decision, not a broker setting.

### 8.5 Schema evolution as a change-control object

In a regulated environment, a schema is a configuration item. A change to `AdtEvent` requires review,
a compatibility check, and a record of which producer and consumer versions were deployed when. A
schema registry with a compatibility mode provides the technical gate; the change-control process
provides the record.

## 9. Industry practice and case evidence

### 9.1 Documented behaviour

**Producer defaults changed in Kafka 3.0.** KIP-679 made `enable.idempotence` default to true, which
in turn forces `acks=all` and `retries` to a very high value. The lab sets the values explicitly,
which is both correct and pedagogically useful.

**Consumer group defaults are documented.** `session.timeout.ms` defaults to 45,000 ms,
`heartbeat.interval.ms` to 3,000 ms, `max.poll.interval.ms` to 300,000 ms, and
`max.poll.records` to 500. The default assignor list begins with `RangeAssignor`; cooperative
sticky assignment must be selected explicitly and requires a rolling restart.

**KIP-848 changes the protocol.** Kafka 4.0 made the new consumer protocol generally available,
with broker-controlled heartbeats and assignors. The client-side `session.timeout.ms`,
`heartbeat.interval.ms`, and `partition.assignment.strategy` settings do not apply under
`group.protocol=consumer`. A team that upgrades without reading this will find its tuning
ineffective.

**Schema Registry defaults to `BACKWARD`.** Confluent's documentation states that the default
compatibility mode is `BACKWARD` (non-transitive), chosen so that consumers can be rewound to the
beginning of a topic. The seven modes are `BACKWARD`, `BACKWARD_TRANSITIVE`, `FORWARD`,
`FORWARD_TRANSITIVE`, `FULL`, `FULL_TRANSITIVE`, and `NONE`.

**Redpanda's retention default matches Kafka's period.** The topic property `retention.ms` defaults
to 604,800,000 ms, one week, inherited from `log_retention_ms`. Redpanda's documentation warns that
retention applies only to closed segments, so a long `segment.ms` extends effective retention.

### 9.2 Case evidence

1. **KIP-98, Exactly Once Delivery and Transactional Messaging.** The design proposal, adopted in
   Kafka 0.11.0.0, introduced the producer identifier, sequence numbers, the transaction
   coordinator, and the `read_committed` isolation level. It is the primary source for what
   transactions actually guarantee.
2. **The Kafka 4.0 consumer protocol (KIP-848).** A public, documented protocol change that
   invalidates a class of client-side tuning. Teams that pinned old clients and upgraded brokers
   encountered it in 2025 and 2026.
3. **The 2021 Log4Shell incident.** Not a Kafka incident, but a supply-chain one that forced teams to
   identify every running image containing a vulnerable library. Streaming platforms were affected
   through their connector and client dependencies.

### 9.3 Inference

The following are inferences, labelled as such:

- Most duplicate records in clinical streams originate at the producer, not the consumer. The
  identifier is regenerated on retry, or the producer is not idempotent across restarts.
- Teams that set retention by capacity rather than by replay requirement end up with either too
  little history or too much disk. The replay procedure is the better starting point.
- The most common cause of a stalled partition is a poison record that is retried forever because
  the consumer commits only after successful processing and has no dead-letter path.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Producer not idempotent | Duplicate records after broker failover | Retry writes twice | Duplicate clinical events | `enable.idempotence=true` |
| New event ID per retry | Duplicates with different IDs | Identifier generated at send time | Undetectable duplicates | Derive ID from source event |
| Auto-commit enabled | Records lost on crash | Offsets committed before processing | Silent data loss | Manual commit after write |
| No deduplication store | Replay duplicates output | At-least-once plus non-idempotent write | Duplicate rows | Unique constraint on event ID |
| Poison record without DLQ | Partition lag grows, consumer logs repeat | Infinite retry | Partition stalls | Dead-letter and commit |
| DLQ without retention | DLQ grows unbounded | No cleanup policy | Disk exhaustion | Retention and depth alert |
| Rebalance loop | Repeated `PreparingRebalance` | `max.poll.interval.ms` exceeded | Lag grows with busy consumers | Reduce batch, offload work |
| Too few partitions | Consumers idle at the limit | Parallelism capped | Cannot scale | Size for peak consumers |
| Repartition without a pause | Out-of-order patient events | Key mapping changes | Wrong patient state | Planned pause or dual write |
| Long segment age | Disk full despite short retention | Retention applies to closed segments | Broker offline | Set segment age with retention |
| Replication factor one | Broker loss loses data | No copies | Data loss | RF 3, min in-sync 2 |
| `acks=1` on clinical events | Records lost on leader failure | No replica acknowledgment | Silent loss | `acks=all` |
| Schema break | Consumers fail after producer deploy | Removed or retyped field | Pipeline halt | Registry with compatibility mode |
| Credential expiry treated as poison | Valid records dead-lettered | Auth error routed to DLQ | Data loss during outage | Classify errors |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Events are keyed by a stable patient identifier | Per-patient ordering | Identifier changes on merge | Compare IDs across a known merge |
| Peak factor is three | Capacity sizing | Shift change, backlog replay | Measure hourly counts for 30 days |
| Consumer service rate is 400 events/s | Lag arithmetic | Larger payloads, downstream latency | Load test with production records |
| Retention applies as configured | Storage sizing | Open segments, compaction | Inspect segment timestamps |
| Broker disk is the only storage | Cost arithmetic | Tiered storage enabled | Check topic storage class |
| Three zones are available | Replication arithmetic | Two-zone region | Describe availability zones |
| The producer survives restarts with its identity | Idempotence | New producer ID per restart | Force a restart and inspect duplicates |
| Schema changes are reviewed | Compatibility claim | Direct registry writes | Audit registry changes |
| The dead-letter topic is monitored | Loss detection | No alert on depth | Inject a poison record |
| The consumer can reprocess within the SLO | Replay planning | Downstream saturation | Time a one-day replay |

## 12. Exercises

### 12.1 Trace an event

**Deliverable.** For the six events in `labs/cloud/kafka/events.jsonl`, compute the partition each
would land in for a three-partition topic using the lab's key, given a stated hash function, and
state which events are ordered relative to each other.

**Acceptance criteria.** Correctly identifies that events for `synth-1001` share a partition.
States that events for different patients have no relative order guarantee.

**Time.** 25 minutes.

### 12.2 Size partitions

**Deliverable.** For 2,000 events per second at 400 bytes, a peak factor of 4, a per-partition
capacity of 2 MB/s, and a target utilisation of 0.7, compute the partitions required by throughput
and by a desire to run eight consumers.

**Acceptance criteria.** Both calculations shown. The recommendation takes the larger value and
explains why.

**Time.** 30 minutes.

### 12.3 Size retention

**Deliverable.** Retained gigabytes for 30 days at 2,000 events per second, 400 bytes, replication
factor three, and a broker disk of 2 TB per broker across three brokers. State whether it fits.

**Acceptance criteria.** Correct units. States the percentage of total disk used and the headroom.

**Time.** 25 minutes.

### 12.4 Configure a producer

**Deliverable.** A producer configuration for clinical events that must not be lost and must not be
duplicated by retries, with a one-paragraph justification per setting.

**Acceptance criteria.** Includes idempotence, acknowledgments, retries, in-flight limit, and
compression. States what the configuration does not guarantee.

**Time.** 35 minutes.

### 12.5 Design deduplication

**Deliverable.** A schema and transaction boundary for a consumer that writes to PostgreSQL and
must be safe under replay and rebalance.

**Acceptance criteria.** Deduplication and output write share a transaction. A unique constraint is
named. The design works with more than one consumer.

**Time.** 45 minutes.

### 12.6 Diagnose lag

**Deliverable.** A decision tree that distinguishes partition saturation, poll-interval rebalance,
downstream latency, and poison-record stall, with the command or metric for each.

**Acceptance criteria.** Each branch ends in a different fix. Does not recommend adding consumers as
a universal remedy.

**Time.** 40 minutes.

### 12.7 Plan a schema change

**Deliverable.** A two-release plan to add a required `encounter_id` field to `AdtEvent`, with the
compatibility mode and the deployment order.

**Acceptance criteria.** Uses an optional field first. States which mode permits the order. Names
the rollback.

**Time.** 35 minutes.

### 12.8 Compare delivery guarantees

**Deliverable.** A table mapping at-most-once, at-least-once, and effectively-once to the lab's
producer and consumer settings, and a statement of which one the lab actually provides.

**Acceptance criteria.** Correctly concludes at-least-once with best-effort deduplication in a local
file. Identifies the multi-consumer gap.

**Time.** 30 minutes.

## 13. Self-check answers

### 13.1 Partition tracing

With three partitions and a hash of `patient_id`, `synth-1001` maps to one partition, so `evt-0001`,
`evt-0003`, and `evt-0006` are ordered relative to each other. `synth-1002` maps to a partition that
may be the same or different, and its events are ordered among themselves. There is no ordering
between the two patients. The exact partition number depends on the hash function; the lab does not
pin one, which is itself a finding: the assignment is stable for a given client library and can
change with a library upgrade.

### 13.2 Partition sizing

Throughput: 2,000 × 400 = 800,000 B/s = 0.8 MB/s. Peak: 3.2 MB/s. At 2 MB/s per partition and 0.7
target: `ceil(3.2 ÷ 1.4) = ceil(2.29) = 3` partitions. For eight consumers: eight partitions.
Choose eight, because the consumer requirement is larger and partitions cannot be reduced without
losing ordering.

### 13.3 Retention

Daily volume: 2,000 × 400 × 86,400 = 69.12 GB/day. Thirty days at replication factor three:
69.12 × 30 × 3 = 6,220.8 GB. Total disk is 3 × 2,000 = 6,000 GB. It does not fit; the shortfall is
220.8 GB, or about 3.7 percent. Reduce retention to 28 days or add disk.

### 13.4 Producer configuration

`enable.idempotence=true`, `acks=all`, `retries` high or default, `max.in.flight.requests.per.connection`
at most 5, and compression enabled. This prevents duplicates from the producer's own retries within
a session and prevents loss when a leader fails. It does not prevent duplicates across producer
restarts, and it does not make an external write idempotent.

### 13.5 Deduplication design

Create a table `processed(event_id TEXT PRIMARY KEY)` in the same PostgreSQL database as the output.
In one transaction, insert the event identifier and apply the output change. A duplicate raises a
unique violation, and the transaction rolls back without applying the change. This works with any
number of consumers because the constraint is in the shared database.

### 13.6 Lag decision tree

If consumer count equals partition count and CPU is low, check for rebalance messages; if present,
the poll interval is too short for the batch. If no rebalance and the downstream latency is high, the
consumer is waiting, not computing. If one partition has lag and others do not, the key is skewed. If
lag grows on one partition and the consumer logs the same record repeatedly, it is a poison record
and needs a dead-letter path.

### 13.7 Schema change plan

Release one adds `encounter_id` as optional with a default of null. Consumers are deployed and
ignore it. Release two makes it required after all producers populate it. `BACKWARD` compatibility
permits consumers first, which matches this order. `FULL` would also permit it. The rollback is to
revert the producer and keep the field optional; reverting the consumer is not necessary because it
tolerates the field's absence.

### 13.8 Delivery guarantees

The lab provides at-least-once delivery: the producer is idempotent within a session, the consumer
commits after processing, and duplicates are suppressed by a local SQLite primary key. With one
consumer, the observable effect is effectively-once for the records that consumer sees. With more
than one consumer, each has its own SQLite file, so a duplicate processed by a different consumer is
not detected. That is the gap.

## 14. Interview questions

**1. What does exactly-once require beyond a broker setting?**
A strong answer describes idempotent writes, a stable key, a transactional sink where available, and
replay-safe logic. A weak answer names `enable.idempotence`.

**2. Why key by patient rather than by event?**
A strong answer explains per-entity ordering and the cost of skew. A weak answer says "for
partitioning".

**3. What happens when a consumer takes longer than `max.poll.interval.ms`?**
A strong answer says the group removes it, rebalances, and reprocesses its batch elsewhere, causing
lag and duplicates. A weak answer says it times out.

**4. How do you choose a partition count?**
A strong answer takes the maximum of the throughput requirement and the consumer parallelism
requirement, then adds headroom, and states the repartition cost. A weak answer picks a round number.

**5. What does a dead-letter topic need besides existing?**
A strong answer says retention, monitoring, a triage owner, and a commit on the original offset. A
weak answer says it stores bad records.

**6. When is `BACKWARD` compatibility the right default?**
A strong answer explains that it lets consumers upgrade first and supports rewinding to the start of
a topic. A weak answer says it is the default.

**7. How do you diagnose rising lag with idle consumers?**
A strong answer separates partition saturation, rebalance loops, and downstream waits, and names the
metric for each. A weak answer says "add consumers".

**8. What does replication factor three with `min.insync.replicas=2` survive?**
A strong answer says one broker or one zone failure, and that a second concurrent failure stops
writes. A weak answer says "three failures".

**9. Why is retention a product decision?**
A strong answer ties it to replay and audit requirements, and notes that object storage is usually the
audit record. A weak answer says it controls disk.

**10. How would you replay a week of events without double-counting?**
A strong answer uses a separate consumer group, a shadow output, rate limiting, and a comparison
before promotion. A weak answer resets the production offsets.

## 15. Key terms

- **`acks`** — Producer acknowledgment setting.
- **At-least-once** — Delivery with possible duplicates.
- **At-most-once** — Delivery with possible loss.
- **Compaction** — Retention by key.
- **Consumer group** — Consumers dividing partitions.
- **Dead-letter queue** — Destination for unprocessable records.
- **Effectively-once** — At-least-once plus idempotent writes.
- **Event time** — Time an event occurred.
- **Idempotent producer** — Producer that suppresses retry duplicates in a session.
- **In-sync replica** — Replica caught up with the leader.
- **Isolation level** — Whether uncommitted transactional records are visible.
- **KIP** — Kafka improvement proposal.
- **Lag** — Log end offset minus committed offset.
- **Log end offset** — Next offset to be written.
- **`max.poll.interval.ms`** — Maximum time between consumer polls.
- **Offset** — Position in a partition.
- **Partition** — Ordered slice of a topic.
- **Poison record** — Record that always fails processing.
- **Rebalance** — Reassignment of partitions.
- **Replication factor** — Number of partition copies.
- **Retention** — How long records remain.
- **Schema registry** — Service storing and validating schemas.
- **Segment** — File holding a range of offsets.
- **Tiered storage** — Offloading old segments to object storage.
- **Topic** — Named stream of records.
- **Transactional producer** — Producer writing atomically across partitions.
- **Watermark** — Bound on acceptable lateness.

## 16. Sources

All sources accessed 26 October 2026.

1. Apache Kafka. *Producer Configs*.
   https://kafka.apache.org/documentation/#producerconfigs
2. Apache Kafka. *Consumer Configs*.
   https://kafka.apache.org/documentation/#consumerconfigs
3. Apache Kafka. *Broker Configs*.
   https://kafka.apache.org/documentation/#brokerconfigs
4. Apache Kafka. *Consumer Rebalance Protocol*.
   https://kafka.apache.org/43/operations/consumer-rebalance-protocol/
5. Apache Kafka. *KIP-98: Exactly Once Delivery and Transactional Messaging*.
   https://cwiki.apache.org/confluence/display/KAFKA/KIP-98+-+Exactly+Once+Delivery+and+Transactional+Messaging
6. Confluent. *Kafka Message Delivery Guarantees*.
   https://docs.confluent.io/kafka/design/delivery-semantics.html
7. Confluent. *Schema Evolution and Compatibility*.
   https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html
8. Redpanda. *Topic Configuration Properties*.
   https://docs.redpanda.com/current/reference/properties/topic-properties/
9. Redpanda. *Manage Disk Space*.
   https://docs.redpanda.com/current/manage/cluster-maintenance/disk-utilization/
10. HL7. *FHIR R4 Subscription*. https://hl7.org/fhir/R4/subscription.html
11. HHS. *HIPAA Security Rule*.
    https://www.hhs.gov/hipaa/for-professionals/security/index.html
12. AWS. *EC2 On-Demand pricing*, US East (Ohio). https://aws.amazon.com/ec2/pricing/on-demand/
13. AWS. *Amazon EBS pricing*. https://aws.amazon.com/ebs/pricing/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_infrastructure-as-code.md`](04_infrastructure-as-code.md)
- Next topic: [`06_observability.md`](06_observability.md)
- Related topic: [`01_architecture-patterns.md`](01_architecture-patterns.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Producer: [`../../labs/cloud/kafka/producer.py`](../../labs/cloud/kafka/producer.py)
- Consumer: [`../../labs/cloud/kafka/consumer.py`](../../labs/cloud/kafka/consumer.py)
- Schema: [`../../labs/cloud/kafka/schemas.py`](../../labs/cloud/kafka/schemas.py)
- Local broker: [`../../labs/cloud/docker-compose.yml`](../../labs/cloud/docker-compose.yml)
- Sample events: [`../../labs/cloud/kafka/events.jsonl`](../../labs/cloud/kafka/events.jsonl)
