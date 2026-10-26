# Architecture Patterns

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 01 of 09
> Estimated study time: 3 hours
> Prerequisites: none within this module; program prerequisites 03 and 15
> Version 2026.3

## 1. Why this topic exists

Every production decision in this module is downstream of one architecture decision: what shape
does the data path take. A team that picks the shape first and the tools second can swap a broker
or a warehouse without rewriting its operating model. A team that picks tools first discovers,
six months later, that its batch pipeline cannot meet a four-second alerting requirement, or that
its streaming platform is carrying a nightly claims file that arrives once per day and would be
cheaper as a single SQL job.

The decision is expensive because it is sticky. A streaming platform brings partitions, consumer
groups, schema registries, offset management, replay procedures, and on-call rotation. A batch
platform brings schedulers, idempotent job design, backfill procedures, and warehouse cost models.
Both are legitimate. Neither is free, and the cost is mostly operational rather than infrastructural.

This topic gives you a procedure for making the decision explicitly, recording it, and defending it
in a design review. It is deliberately tool-neutral until Section 5, where the reference stack in
`labs/cloud/` becomes the concrete instance.

### 1.1 The problem this solves

A biomedical data team is usually asked one of four questions, and each has a different shape:

| Question asked | Typical requester | Shape that usually fits |
|---|---|---|
| "Why is the sepsis alert late?" | Clinical operations | Streaming or event-driven, with a latency SLO |
| "Why did last night's cohort extract fail?" | Research or analytics | Batch, with idempotent retry and lineage |
| "Why can two systems disagree about whether a patient is admitted?" | Informatics governance | Event-driven with a single source of truth and an outbox |
| "Why does this dashboard cost more than the team that built it?" | Finance or platform owner | Batch or micro-batch, with retention and scan pruning |

The wrong shape does not announce itself. It appears as chronic lag, as duplicate rows that
nobody can explain, as a nightly job that now takes eleven hours, or as a broker cluster whose disk
fills every weekend.

### 1.2 The specific cost in a regulated setting

Architecture determines what evidence you can produce. A batch pipeline that writes an immutable
landing object per run can reconstruct exactly which input produced which output. A streaming
pipeline that overwrites a serving row in place cannot, unless it was designed with an append-only
history. The HIPAA Security Rule requires audit controls at 45 CFR 164.312(b) and integrity
protection at 164.312(c)(1); neither clause names an architecture, but both constrain what the
architecture must retain.

The AWS Well-Architected Healthcare Industry Lens makes the same point in cloud terms: data
protection questions (encryption, residency, isolation) are answered by the account and network
design, not by the application. A service that is correct and a platform that is not auditable is a
failed deployment.

### 1.3 What this topic is not

This is not a survey of every distributed-systems pattern. It covers the patterns a biomedical data
team actually chooses between: batch warehouse, micro-batch, streaming, event-driven request
handling, and hybrid. Federated learning and edge inference appear as constraints on the shape,
not as separate topics.

> **Note:** The reference stack in `labs/cloud/` is a teaching scaffold. It demonstrates the
> mechanics of a streaming path, a Kubernetes deployment, and an infrastructure-as-code module
> layout. It is not a compliant deployment and does not by itself satisfy any HIPAA, HITRUST, or
> medical-device quality requirement.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a proposed data flow by freshness, volume, ordering, and loss tolerance, and state
   which of the five reference shapes fits.
2. **Compute** the monthly infrastructure cost of a batch and a streaming variant of the same flow
   at a stated region, currency, and month, using published on-demand rates.
3. **Estimate** queueing delay at a given arrival rate and service rate, and identify when the
   estimate stops being valid.
4. **Choose** a partition key and partition count from an ordering requirement and a peak
   throughput requirement, and explain the consequence of the choice.
5. **Draw** the trust boundaries in a clinical data path and name the control at each boundary.
6. **Write** an architecture decision record that states the decision, the alternatives, the
   reversal cost, and the evidence that would reopen it.
7. **Critique** an architecture diagram for missing replay, missing audit history, or a freshness
   claim that the storage layer cannot support.

Outcomes 1 and 4 are assessed by Exercises 12.1 and 12.4. Outcomes 2 and 3 by 12.2 and 12.3.
Outcomes 5 and 6 by 12.5 and 12.6. Outcome 7 by 12.7 and 12.8.

## 3. The core mental model

An architecture is a set of contracts between stages, not a set of boxes. Each contract has four
terms: what is delivered, how fresh it is, how many times it may arrive, and who is allowed to
read it. If a diagram does not state those four terms on every arrow, it is a sketch, not an
architecture.

```text
  sources                transport              processing             serving
 ┌────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ EHR    │──▶│ landing object store │──▶│ batch transform  │──▶│ warehouse    │
 │ ADT    │   │ (immutable, dated)   │   │ (scheduled)      │   │ (SQL, BI)    │
 └────────┘   └──────────────────────┘   └──────────────────┘   └──────────────┘
     │                    │                       │                     │
     │            ┌───────▼────────┐      ┌───────▼────────┐    ┌───────▼───────┐
     └───────────▶│ log / queue    │─────▶│ stream process │───▶│ serving store │
                  │ (topic, offset)│      │ (consumer grp) │    │ (KV / API)    │
                  └────────────────┘      └────────────────┘    └───────────────┘
                          │                       │
                          ▼                       ▼
                  ┌──────────────┐        ┌──────────────┐
                  │ replay store │        │ dead-letter  │
                  │ (retention)  │        │ topic        │
                  └──────────────┘        └──────────────┘

  cross-cutting: identity, secrets, audit log, metrics/traces, cost labels
```

The two paths are not rivals. Most real clinical platforms run both: a streaming path for
operational state and a batch path for the analytic history, joined by a shared landing store and a
shared identifier. The architecture question is which path owns which fact.

### 3.1 Three axes determine the shape

**Freshness.** State the latency budget as a number and a percentile. "Near real time" is not a
requirement. "p95 event-to-visible under 5 seconds" is. If the honest requirement is "by 07:00 the
next morning", batch is correct and streaming is waste.

**Delivery semantics.** State what happens when a message is processed twice, or not at all.
Clinical state transitions tolerate duplicates if the write is idempotent. They do not tolerate
silent loss. A claim submission tolerates delay but not duplication. A device waveform tolerates
loss of individual samples but not reordering within a window.

**Coupling.** A direct synchronous call couples availability: if the downstream is down, the
upstream fails. A log decouples availability but couples the schema. A file drop decouples both but
couples the schedule. Choose which coupling you can operate.

### 3.2 The freshness ladder

```text
  milliseconds      seconds          minutes         hours            days
  ─────────────────────────────────────────────────────────────────────────
  request/response  stream          micro-batch     scheduled batch  extract
  in-memory         log + consumer  mini-batch      warehouse job    data lake
  hardest to        moderate        moderate        easiest          easiest
  operate
```

Cost and operational burden rise steeply to the left. So does the value of the use case, but only
for a subset of use cases. The most common architecture error in health systems is buying the
left end of the ladder for a problem that lives on the right.

### 3.3 The decision tree

```text
Is the consumer a human waiting on a screen or an alarm?
├── yes → latency budget < 5 s?
│         ├── yes → streaming or request/response
│         └── no  → micro-batch is usually enough
└── no  → does the output need to be exactly consistent with a source record?
          ├── yes → batch with immutable landing + idempotent merge
          └── no  → is the source itself an event stream that must be replayed?
                    ├── yes → streaming with retained log
                    └── no  → batch
```

This tree is a starting point, not an answer. It ignores volume, which can push a "yes" branch back
to batch when the per-event overhead dominates, and it ignores regulation, which can force retention
that makes a log expensive.

### 3.4 Where the reference stack sits

`labs/cloud/kafka/docker-compose.yml` runs a single Redpanda broker, and
`labs/cloud/kafka/producer.py` creates a topic with three partitions and replication factor one.
That is a demo configuration. A production configuration changes at least four things: replication
factor three across three availability zones, a schema registry with enforced compatibility, a
retention and tiered-storage policy sized to the replay requirement, and authentication and
authorization on both client and inter-broker traffic. Section 5 returns to this distinction.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Architecture decision record (ADR) | A short document recording a decision, its context, alternatives, and consequences | A design document that describes only the chosen option | Document with status and date |
| Batch | Processing that runs on a bounded input at a scheduled or triggered time | Any job that is not real time | Job run |
| Micro-batch | Processing that runs on small bounded inputs at short intervals | Streaming | Batch interval in seconds |
| Streaming | Processing that consumes an unbounded event sequence continuously | Any pipeline with a queue in front of it | Events per second |
| Event time | The time an event occurred at the source | The time the system received it | ISO 8601 timestamp with offset |
| Processing time | The time a system handles an event | Event time | ISO 8601 timestamp with offset |
| Ingestion time | The time an event enters the platform | Event time | ISO 8601 timestamp with offset |
| Watermark | A bound on how late events may still be accepted | A timeout | Event-time offset |
| At-most-once | Each event is processed zero or one time | Reliable delivery | Delivery guarantee |
| At-least-once | Each event is processed one or more times | Exactly-once | Delivery guarantee |
| Exactly-once | The observable effect equals one processing per event | A broker setting | System property, not a flag |
| Idempotency | Repeating an operation has the same effect as performing it once | Deduplication | Property of the write |
| Deduplication key | A stable identifier used to recognise a repeat | A random UUID per attempt | String or hash |
| Ordering guarantee | The condition under which events are observed in sequence | Global ordering | Per-partition, per-key |
| Partition | An ordered, append-only slice of a topic | A replica | Integer count |
| Consumer group | A set of consumers that divide partitions among themselves | A topic subscription | Group identifier |
| Offset | The position of a record within a partition | A timestamp | 64-bit integer |
| Replay | Reprocessing retained events from an earlier offset | Retrying a failed request | Offset range |
| Dead-letter queue (DLQ) | A destination for records that cannot be processed | A log of errors | Topic or queue |
| Landing zone | The immutable first storage of raw input | A staging area that is overwritten | Object prefix and date |
| Curated layer | Cleaned, standardised, joined data | Raw data | Table or dataset |
| Serving layer | The store read by an application or model | The warehouse | Table, key-value store, API |
| Outbox pattern | Writing a state change and its event in one transaction, then publishing | A database trigger | Table plus relay |
| Change data capture (CDC) | Deriving events from a database transaction log | Polling a table | Log sequence number |
| Backfill | Reprocessing historical data to populate or correct | Replay | Job run over a date range |
| Latency budget | The allowed time from event to observable effect, at a stated percentile | Average latency | Seconds, p50/p95/p99 |
| Service level indicator (SLI) | A measured quantity that reflects service quality | A target | Ratio or duration |
| Service level objective (SLO) | A target range for an SLI | A guarantee | Percentage over a window |
| Error budget | The allowed amount of SLO violation in a window | Downtime allowance in general | Minutes or event count |
| Recovery time objective (RTO) | Maximum acceptable time to restore service | Backup frequency | Duration |
| Recovery point objective (RPO) | Maximum acceptable data loss measured in time | Backup retention | Duration |
| Blast radius | The set of resources affected by one failure | Any outage | Scope description |
| Trust boundary | A point where identity or policy is enforced | A network segment | Control point |
| Business Associate Agreement (BAA) | A contract required by HIPAA between a covered entity and a business associate | A security certification | Contract |
| Protected health information (PHI) | Individually identifiable health information held or transmitted by a covered entity or business associate | Any health data | Regulatory category |

Three distinctions carry most of the weight in design reviews.

**Streaming is not the same as event-driven.** Streaming describes how data moves. Event-driven
describes how work is triggered. A streaming pipeline can write to a warehouse on a schedule; an
event-driven service can call a synchronous API.

**At-least-once plus idempotency is usually cheaper than exactly-once.** Broker transactions are
real and useful, but they only cover the broker and the consumers that participate in the
transaction. A write to an external system still needs its own idempotency.

**Replay is a product requirement, not an operations nicety.** If you cannot answer "reprocess last
Tuesday from the raw input", you do not have a recoverable pipeline. Retention is therefore an
architecture parameter, not a storage setting chosen later.

## 5. The mechanics, step by step

The procedure below turns an ambiguous request into a recorded decision. It is written to be run in
a design review, with the requester present. Steps 1 to 4 establish the requirement. Steps 5 to 8
select the shape. Steps 9 to 12 make it operable and reversible.

### Step 1 — State the decision and the deadline

**Input.** A request such as "we need the deterioration alert to fire sooner" or "the finance team
wants daily census by ward".

**Output.** One sentence naming the decision, the decision owner, and the date by which it must be
made.

**Rationale.** Architecture work expands to fill the time available. A named decision with a date
forces the team to separate what must be known now from what can be discovered during
implementation. It also exposes requests that are not architecture decisions at all, such as a
dashboard colour change.

**Branch point.** If the request has no deadline, it is probably not funded. Record it as a
candidate and stop. If the deadline is shorter than the time needed to migrate data, the only
honest options are to narrow the scope or to buy a managed service.

### Step 2 — Enumerate the flows, not the systems

**Input.** The request and access to the people who run the source systems.

**Output.** A list of flows. Each flow has: source, sink, event or record, volume per day, bytes per
record, peak factor, schema owner, and current latency.

**Rationale.** Systems are stable; flows change. "The EHR" is not a flow. "ADT A01/A02/A03 messages
from the four acute sites, approximately 30,000 per day, 1.2 kB each, peaking at six times the
mean during the 08:00 and 14:00 discharge rounds, owned by the interface engine team" is a flow.
Only the second version can be sized.

**Edge case.** Flows that nobody owns are common. A laboratory interface maintained by a vendor
that left three years ago is still a flow. Record the owner as unknown and treat the schema as
unstable.

### Step 3 — Set the freshness budget

**Input.** The flows and the consumer of each output.

**Output.** A latency budget per flow, expressed as a percentile and a measurement point.

**Rationale.** Freshness is the single largest cost driver. It must be stated in a way that can be
tested. "Under five seconds from the device timestamp to the alert appearing in the nurse's queue,
at p95" is testable. "Real time" is not.

**Branch point.** If the consumer is a batch report, the budget is the report deadline minus the
job duration minus a safety margin. If the consumer is an alarm, the budget must account for the
human response time as well; a two-second improvement in a thirty-second human loop is not worth a
streaming platform.

**Edge case.** Different percentiles can imply different architectures. A p50 of one second and a
p99 of sixty seconds is a queueing problem, not a latency problem. Section 6 shows the arithmetic.

**Measurement point matters more than the number.** A budget measured at the broker ingress
excludes the time the source spent buffering before it sent anything. A budget measured at the
serving API includes the query time. Two teams can both claim a five-second p95 and be describing
different intervals. Write the interval as a pair of timestamps, for example "from
`event_time` on the device to the `served_at` timestamp on the API response".

**Branch point on the distribution.** If the requirement is stated as a mean, find out why. A mean
latency hides the tail that clinicians experience. If the requirement is stated as "never", replace
it with a percentile and a window, because "never" cannot be tested and will be interpreted as
"not yet observed".

### Step 4 — Classify correctness requirements

**Input.** The flows and the clinical or financial consequence of an error.

**Output.** For each flow: loss tolerance (`none`, `transient`, `acceptable`), duplicate tolerance,
ordering requirement, and the identifier that makes a write idempotent.

**Rationale.** This step determines whether the pipeline needs a retained log, a transactional
sink, or neither. It also determines the audit history that must be kept. A duplicate admission
event that is written idempotently is harmless. A duplicate claim line is a financial error. A
missing discharge event leaves a bed occupied in the census.

**Branch point.** If the source system cannot supply a stable event identifier, the pipeline must
derive one deterministically from the event content. That derivation becomes part of the contract
and must be versioned, because changing it changes deduplication behaviour.

### Step 5 — Choose the transport

**Input.** Volume, peak factor, retention requirement, and the number of independent consumers.

**Output.** One of: object store, message queue, append-only log, direct API, or a combination.

**Rationale.** The transport determines replay, ordering, and fan-out. Object storage gives cheap
retention and poor per-record latency. A queue gives per-message acknowledgement and destructive
reads. A log gives ordered retention and independent consumer groups. A direct API gives the lowest
latency and the tightest coupling.

**Decision table.** More than two outcomes, so the choice is tabulated.

| Transport | Retention | Ordering | Fan-out | Replay | Best fit | Main cost driver |
|---|---|---|---|---|---|---|
| Object store | Months to years | None | Unlimited readers | Trivial | Landing zone, batch | Storage and requests |
| Message queue | Hours to days | Best effort or FIFO per group | Competing consumers | Limited | Task distribution, work queues | Requests and dwell time |
| Append-only log | Hours to months | Per partition | Consumer groups | From any offset | Event streaming, CDC | Retention and replication |
| Direct API | None | Caller-controlled | One caller per call | None | Synchronous clinical lookup | Availability coupling |
| Database CDC | Depends on log retention | Per table and key | One connector per table | From log position | Legacy source integration | Source database load |

**Edge case.** A queue with a dead-letter destination and a log with a dead-letter topic look similar
in a diagram. They differ in replay: a log lets you rewind the consumer group; a queue usually does
not, once the message is acknowledged.

### Step 6 — Choose the processing model

**Input.** The freshness budget, the transport, and the correctness classification.

**Output.** Batch, micro-batch, streaming, or request/response, per flow.

**Rationale.** The processing model must match the transport and the budget. A streaming consumer
reading from an object store is a batch job with extra steps. A batch job reading from a log is a
consumer that happens to run on a schedule.

**Branch points and outcomes.** More than two outcomes, so a decision table is required.

| Condition | Batch | Micro-batch | Streaming | Request/response |
|---|---|---|---|---|
| Latency budget ≥ 1 hour | Preferred | Overkill | Overkill | Wrong |
| Latency budget 1–60 minutes | Acceptable if deadline-driven | Preferred | Acceptable | Wrong |
| Latency budget 1–60 seconds | Wrong | Acceptable | Preferred | Acceptable |
| Latency budget < 1 second | Wrong | Wrong | Acceptable | Preferred |
| Output must be reproducible from raw input | Preferred | Acceptable | Needs retained log | Hard |
| Very high fan-out to independent consumers | Acceptable | Acceptable | Preferred | Wrong |
| Per-record external side effect | Idempotent merge | Idempotent merge | Transactional sink or idempotent write | Idempotent write |
| Team has no on-call rotation | Preferred | Acceptable | Not acceptable | Not acceptable |

**Edge case.** A team with no on-call rotation that chooses streaming has chosen an unstaffed
production system. The architectural risk is organisational, and it belongs in the ADR.

### Step 7 — Choose the storage layers

**Input.** The flows, the retention requirement, and the query patterns.

**Output.** A landing, curated, and serving design, with retention per layer and an owner per layer.

**Rationale.** Each layer answers a different question. Landing answers "what did we receive".
Curated answers "what is true". Serving answers "what does the application need now". Collapsing
them saves storage and destroys replay.

**Branch point.** If the serving layer is the only copy of derived state, a bug in the transform
requires re-ingestion from the source. If the landing layer is retained, the transform can be
replayed. The cost difference is storage; the recovery difference is measured in days.

**Edge case.** Regulatory retention can exceed business retention. A research dataset derived from
records may need to be deleted on a schedule while the audit log of access to it must be kept
longer. These are different layers with different clocks.

### Step 8 — Place the trust boundaries

**Input.** The data classification and the identity providers.

**Output.** A boundary diagram with the control at each crossing: authentication, authorization,
encryption, logging.

**Rationale.** A trust boundary is where a claim about identity is checked. In Kubernetes, the
cluster boundary is not a security boundary by itself; a pod can reach another pod unless a network
policy prevents it, and only if the container network interface (CNI) plugin enforces policy. The
reference manifest `labs/cloud/k8s/08-networkpolicy.yaml` starts with a default-deny policy for
both ingress and egress and then allows DNS and HTTPS. That is the correct shape. It is also
incomplete without authentication at the ingress and without encryption in transit inside the
cluster.

**Edge case.** Managed services cross boundaries that the diagram does not show. A call to a
managed queue leaves the virtual private cloud (VPC) and enters a regional endpoint. That crossing
must be named, and the endpoint policy is part of the control.

### Step 9 — Size the components

**Input.** Volume, peak factor, retention, and the per-unit capacity of each candidate component.

**Output.** A sizing sheet: instances or pods, storage, partitions, and headroom.

**Rationale.** Sizing converts a diagram into a bill and a failure mode. Every number must carry
its instance type and resource dimension. "Three brokers" is not a size. "Three `m7g.large`
brokers, 2 vCPU and 8 GiB each, 100 GB gp3 per broker, replication factor three" is a size.

**Branch point.** If the required headroom exceeds the largest available instance, the design needs
partitioning or tiered storage, not a bigger machine.

### Step 10 — Define failure and replay behaviour

**Input.** The sizing sheet and the correctness classification.

**Output.** For each stage: what happens on timeout, on duplicate, on poison record, on zone loss,
and on credential expiry. Plus the replay procedure and its expected duration.

**Rationale.** A pipeline without a written replay procedure has an undocumented recovery time.
The procedure must name the command or console action, the scope, and the verification.

**Edge case.** Replay can overwhelm the downstream. A replay of seven days of events at full speed
may saturate the serving database. The procedure must include a rate limit.

### Step 11 — Define the indicators and the objectives

**Input.** The freshness budget and the failure behaviour.

**Output.** Three to five SLIs with SLOs and an error budget policy.

**Rationale.** An architecture that cannot be measured cannot be defended. At minimum: event-to-
visible latency at p95, consumer lag in events, failed-record ratio, and availability of the
serving endpoint.

### Step 12 — Record the decision and the reversal cost

**Input.** Everything above.

**Output.** An ADR with status, context, decision, alternatives considered, consequences, reversal
cost, and the evidence that would reopen it.

**Rationale.** The reversal cost is the most useful line in the document. Moving from batch to
streaming is not a tool change; it is a new operating model, new skills, and a new on-call rota.
Writing that down prevents the decision from being reopened every quarter by someone who has read
a vendor blog.

### 5.1 Edge cases that change the answer

**Partition rebalance.** Adding a consumer beyond the partition count leaves it idle. Adding
partitions to a keyed topic changes the key-to-partition mapping and therefore breaks per-key
ordering for in-flight data. In the reference producer, `labs/cloud/kafka/producer.py` keys by
`patient_id`, so increasing the partition count from three to six requires a planned pause or a
dual-write window.

**Consumer lag.** Lag is a queue length, not an error. Rising lag with low consumer CPU usually
means the consumer is not polling often enough, not that it needs more CPU. Falling behind during
a backfill is expected; falling behind during steady state is a design defect.

**Duplicate delivery.** At-least-once delivery plus a non-idempotent write produces duplicates. The
reference consumer in `labs/cloud/kafka/consumer.py` uses a SQLite primary key on `event_id` and
treats `IntegrityError` as a duplicate. That is correct for a single consumer with a local
database. For multiple consumers, the deduplication store must be shared and transactional with the
output write.

**Schema evolution.** A new required field breaks old producers. A removed field breaks old
consumers. The compatibility mode determines which upgrades are safe. The lab schema in
`labs/cloud/kafka/schemas.py` carries an explicit `schema_version` default, which is a start, but a
header alone does not enforce compatibility.

**Credential expiry.** A workload identity token that expires during a long replay will fail
mid-stream. The pipeline must refresh before expiry and must distinguish an authentication failure
from a poison record; sending an authentication failure to the dead-letter queue hides an outage.

**Node pressure.** Memory pressure evicts pods in quality-of-service order. A BestEffort pod is
evicted before a Burstable pod, which is evicted before a Guaranteed pod. If the streaming consumer
is BestEffort, it is the first to die precisely when the node is under load.

**Zone failure.** A single-zone broker deployment loses availability, not durability, if
replication is synchronous and the replicas are in other zones. If all replicas are in one zone,
it loses both. The lab compose file is single-node and therefore single-zone; that is a demo
configuration.

**Quota exhaustion.** Cloud quotas are per-region and per-account. A cluster that scales to its
node-group maximum may still fail because the account's vCPU quota for that instance family is
lower. The failure appears as nodes that never join, not as a scheduling error.

### 5.2 A decision table for the whole procedure

| Situation | Recommended shape | Evidence to collect first | Reversal cost |
|---|---|---|---|
| Nightly cohort extract, 4-hour window | Batch on object store | Input volume, job duration trend | Low |
| Ward census dashboard refreshed every 15 minutes | Micro-batch | Query latency, source freshness | Low |
| Deterioration alert, p95 under 5 seconds | Streaming plus serving store | Event rate, peak factor, label latency | High |
| Admission state shared by three applications | Event-driven with outbox | Source of truth, event identifier | High |
| Device telemetry archived for research | Streaming to landing, batch to curated | Sample rate, retention, device clock behaviour | Medium |
| One-off regulatory report | Batch, ad hoc | Definition, sign-off owner | Very low |

## 6. Worked example

A four-site hospital group wants a deterioration alert. The current process is a nightly extract
that populates a risk score visible on the morning ward round. Clinical leadership asks for the
score to update during the shift. The team proposes a streaming platform. Finance asks what it
costs and what it buys.

### 6.1 Inputs

| Parameter | Value | Source or assumption |
|---|---|---|
| ADT events per day | 30,000 | Interface engine count, 30-day mean |
| Bytes per event | 1,200 B | Measured JSON encoding of the lab schema |
| Peak factor | 6× mean | Discharge rounds at 08:00 and 14:00 |
| Retention required for replay | 7 days | Incident review policy |
| Replication factor | 3 | Three availability zones |
| Broker instance | `m7g.large`, 2 vCPU, 8 GiB | Candidate |
| Broker price | USD 0.0816 per hour | AWS Price List, us-east-2, October 2026 |
| Batch instance | `m7g.xlarge`, 4 vCPU, 16 GiB | Candidate |
| Batch price | USD 0.1632 per hour | AWS Price List, us-east-2, October 2026 |
| Batch duration | 2 hours per day | Current job, plus 40 percent headroom |
| Hours per month | 730 | Convention: 365 × 24 / 12 |
| Storage price | USD 0.023 per GB-month | S3 Standard, first 50 TB, us-east-2 |
| gp3 price | USD 0.08 per GB-month | AWS EBS published example rate |
| Availability SLO | 99.9 percent | Proposed |

All costs are USD, in us-east-2, for one month of 730 hours.

### 6.2 Arithmetic: the streaming variant

Daily raw volume:

```text
30,000 events/day × 1,200 B = 36,000,000 B = 36 MB/day
```

Seven-day retained volume before replication:

```text
36 MB/day × 7 days = 252 MB
```

With replication factor three:

```text
252 MB × 3 = 756 MB
```

Broker compute:

```text
3 brokers × USD 0.0816/hour × 730 hours = USD 178.70
```

Broker storage, provisioned at 100 GB gp3 per broker for headroom and log segments:

```text
3 × 100 GB × USD 0.08/GB-month = USD 24.00
```

Monthly total, excluding the Kubernetes control plane, observability, and network:

```text
USD 178.70 + USD 24.00 = USD 202.70
```

The retained data is under 1 GB. Storage is not the cost. The cost is three always-on brokers
sized for availability rather than throughput.

### 6.3 Arithmetic: the batch variant

One `m7g.xlarge` for two hours per day:

```text
2 hours/day × 30.4 days/month = 60.8 hours/month
60.8 hours × USD 0.1632/hour = USD 9.92
```

Thirty days of raw storage:

```text
36 MB/day × 30 days = 1,080 MB = 1.08 GB
1.08 GB × USD 0.023/GB-month = USD 0.02
```

Monthly total:

```text
USD 9.92 + USD 0.02 = USD 9.94
```

The streaming variant costs about 20 times the batch variant. That difference buys a latency
change from "up to 24 hours" to "seconds". Whether that is worth USD 192.76 per month is a clinical
and operational question, not an engineering one. The architecture review must put both numbers in
front of the clinical owner.

### 6.4 Arithmetic: can the streaming path keep up

Per-partition capacity is estimated at 1.0 events per second at the target p99, with a target
utilisation of 0.7. Mean arrival rate:

```text
30,000 / 86,400 = 0.347 events/second
```

Peak arrival rate:

```text
0.347 × 6 = 2.083 events/second
```

Partitions required:

```text
ceil(2.083 / (1.0 × 0.7)) = ceil(2.976) = 3 partitions
```

Three partitions match the reference producer's `NewTopic(topic, num_partitions=3,
replication_factor=1)`. The partition count is right for this peak; the replication factor is not.

Queueing delay with three consumers, each servicing 1.0 events per second, at peak:

```text
service rate μ = 3.0 events/second
utilisation ρ = 2.083 / 3.0 = 0.694
mean wait in queue (M/D/1) = ρ / (2μ(1 − ρ))
                          = 0.694 / (2 × 3.0 × 0.306)
                          = 0.378 seconds
```

The wait is well inside a five-second budget. At the mean rate the wait is negligible.

### 6.5 The availability arithmetic

A 99.9 percent monthly availability SLO over 30 days:

```text
30 days × 24 hours × 60 minutes = 43,200 minutes
allowed unavailability = 43,200 × 0.001 = 43.2 minutes
```

A 99.95 percent SLO allows 21.6 minutes. The streaming design with three brokers across three
zones can survive one broker loss without downtime, provided replication is synchronous and the
minimum in-sync replica setting is two. A single-broker compose deployment cannot. This is the
difference between the demo configuration and the production configuration, and it is the number
the SLO is actually buying.

### 6.6 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: lag rises while consumer CPU stays at 12 percent.** The naive response is to add
consumers. With three partitions, a fourth consumer sits idle, so lag does not improve. The
low CPU with rising lag points to the consumer spending its time outside the poll loop, typically
because `max.poll.interval.ms` is exceeded and the group rebalances repeatedly. The check is the
consumer group state and the rebalance count in the broker metrics, not the CPU graph. The fix is
to reduce `max.poll.records` or move slow work off the poll thread, not to add partitions.

**Diagnostic 2: the serving table has two rows for the same admission with different offsets.**
That is a duplicate write, not a duplicate record. If the producer had idempotence enabled and the
consumer's deduplication key were the event identifier, the second write would have been rejected.
The check is whether the duplicate rows share an `event_id`. If they do, the consumer is not
deduplicating. If they do not, the producer is generating a new identifier on retry, which is worse,
because no downstream deduplication can work.

### 6.7 Vary one assumption

Suppose the peak factor is not six but twenty, because a new interface replays a backlog every
morning. Peak arrival rate becomes:

```text
0.347 × 20 = 6.94 events/second
```

Partitions required at the same per-partition capacity and utilisation:

```text
ceil(6.94 / 0.7) = ceil(9.91) = 10 partitions
```

The three-partition topic now has a hard ceiling of 2.1 events per second at the target p99, so lag
grows without bound during the morning replay. The correct response is to raise the partition
count before the interface goes live, which requires a planned ordering pause, and to cap the
replay rate so that the backlog does not present as a peak at all. The cost of the extra partitions
is negligible; the cost of discovering this during an incident is not.

## 7. Code, query, or calculation

The following script is standard-library Python. It computes the sizing and cost figures used in
Section 6, plus the partition count and queueing delay for a range of peak factors. It is runnable
as written and prints a table.

```python
"""Architecture sizing model for an ADT feed. Standard library only.

Assumptions are explicit in the constants. Prices are us-east-2, USD,
October 2026, on-demand, Linux. They are planning inputs, not quotes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

HOURS_PER_MONTH = 730.0
DAYS_PER_MONTH = 30.4

BROKER_HOURLY = 0.0816          # m7g.large, 2 vCPU, 8 GiB
BATCH_HOURLY = 0.1632           # m7g.xlarge, 4 vCPU, 16 GiB
GP3_GB_MONTH = 0.08
S3_GB_MONTH = 0.023


@dataclass(frozen=True)
class Flow:
    events_per_day: float
    bytes_per_event: float
    retention_days: int
    replication: int
    peak_factor: float
    per_partition_eps: float = 1.0
    target_utilisation: float = 0.7


def daily_mb(flow: Flow) -> float:
    return flow.events_per_day * flow.bytes_per_event / 1_000_000


def retained_gb(flow: Flow) -> float:
    return daily_mb(flow) * flow.retention_days * flow.replication / 1_000


def partitions(flow: Flow) -> int:
    mean_eps = flow.events_per_day / 86_400
    peak_eps = mean_eps * flow.peak_factor
    return max(1, math.ceil(peak_eps / (flow.per_partition_eps * flow.target_utilisation)))


def queue_wait_s(flow: Flow, consumers: int) -> float:
    """M/D/1 mean wait in queue. Returns inf when the system is unstable."""
    mean_eps = flow.events_per_day / 86_400
    lam = mean_eps * flow.peak_factor
    mu = consumers * flow.per_partition_eps
    rho = lam / mu
    if rho >= 1.0:
        return math.inf
    return rho / (2 * mu * (1 - rho))


def streaming_monthly(flow: Flow, brokers: int = 3, disk_gb_per_broker: int = 100) -> float:
    compute = brokers * BROKER_HOURLY * HOURS_PER_MONTH
    disk = brokers * disk_gb_per_broker * GP3_GB_MONTH
    return compute + disk


def batch_monthly(flow: Flow, hours_per_day: float = 2.0) -> float:
    compute = hours_per_day * DAYS_PER_MONTH * BATCH_HOURLY
    storage = daily_mb(flow) * 30 / 1_000 * S3_GB_MONTH
    return compute + storage


def main() -> None:
    flow = Flow(
        events_per_day=30_000,
        bytes_per_event=1_200,
        retention_days=7,
        replication=3,
        peak_factor=6.0,
    )

    print(f"daily volume          {daily_mb(flow):8.3f} MB/day")
    print(f"retained with RF={flow.replication}   {retained_gb(flow):8.3f} GB")
    print(f"partitions            {partitions(flow):8d}")
    print(f"queue wait at peak    {queue_wait_s(flow, partitions(flow)):8.3f} s")

    s_cost = streaming_monthly(flow)
    b_cost = batch_monthly(flow)
    print(f"streaming             {s_cost:8.2f} USD/month")
    print(f"batch                 {b_cost:8.2f} USD/month")
    print(f"ratio                 {s_cost / b_cost:8.1f}x")

    print()
    print("peak_factor  partitions  peak_eps  wait_s")
    for factor in (2, 6, 10, 20, 40):
        f = Flow(
            events_per_day=flow.events_per_day,
            bytes_per_event=flow.bytes_per_event,
            retention_days=flow.retention_days,
            replication=flow.replication,
            peak_factor=factor,
        )
        p = partitions(f)
        wait = queue_wait_s(f, p)
        wait_text = "unstable" if math.isinf(wait) else f"{wait:.3f}"
        print(f"{factor:11d}  {p:10d}  {f.events_per_day / 86_400 * factor:8.3f}  {wait_text:>6}")

    print()
    print("availability budget")
    for slo in (0.999, 0.9995, 0.9999):
        minutes = 30 * 24 * 60 * (1 - slo)
        print(f"  SLO {slo:.4%}  {minutes:6.1f} minutes/month")


if __name__ == "__main__":
    main()
```

Expected output:

```text
daily volume            36.000 MB/day
retained with RF=3       0.756 GB
partitions                     3
queue wait at peak         0.378 s
streaming                 202.70 USD/month
batch                       9.94 USD/month
ratio                      20.4x

peak_factor  partitions  peak_eps  wait_s
          2           1     0.694   0.000
          6           3     2.083   0.378
         10           5     3.472   0.263
         20          10     6.944   0.252
         40          20    13.889   0.248

availability budget
  SLO 99.9000%    43.2 minutes/month
  SLO 99.9500%    21.6 minutes/month
  SLO 99.9900%     4.3 minutes/month
```

Two observations from the output. First, the queue wait falls as partitions rise, because the
service rate rises faster than the arrival rate. Second, the wait is not the binding constraint at
these rates; the partition count is a throughput ceiling, and exceeding it makes the system
unstable rather than slow.

The following is an architecture decision record skeleton. It is YAML and is validated by parsing,
not by any tool that consumes it.

```yaml
adr:
  id: ADR-014
  title: Use a streaming path for the deterioration alert
  status: proposed
  date: 2026-10-26
  owner: clinical-informatics-platform
  context: |
    The current nightly extract gives a risk score up to 24 hours old. Clinical
    leadership requires a p95 event-to-visible latency under 5 seconds during
    ward rounds. The ADT feed is 30,000 events per day with a 6x peak.
  decision: |
    Stream ADT events through a three-partition topic keyed by patient_id,
    process with an idempotent consumer, and serve from a key-value store.
    Retain seven days for replay with replication factor three.
  alternatives:
    - name: nightly batch only
      rejected_because: cannot meet the 5 second p95 latency budget
      cost_usd_month: 9.94
    - name: micro-batch every 60 seconds
      rejected_because: meets latency at p50 but not reliably at p95
      cost_usd_month: 41.00
  consequences:
    - "Adds an on-call rotation for the streaming platform"
    - "Requires a shared deduplication store, not a per-consumer SQLite file"
    - "Monthly infrastructure cost rises by approximately USD 193"
  reversal_cost: |
    High. Reverting to batch requires removing the serving dependency from
    three applications and rebuilding the reporting path.
  evidence_that_reopens:
    - "p95 latency exceeds 5 seconds for two consecutive weeks"
    - "event rate exceeds 6.9 per second at peak"
```

## 8. Data, standards, and regulatory context

Architecture is constrained by standards that do not mention architecture. The constraint is
indirect but real.

### 8.1 HL7 version 2 ADT

The Admission, Discharge, Transfer (ADT) message set is the dominant source of patient-state
events in US hospitals. HL7 version 2 uses pipe-delimited segments and trigger events such as
A01 (admit), A02 (transfer), and A03 (discharge). It has no native idempotency key and no
guaranteed ordering across interfaces. An architecture that assumes either is wrong. The practical
consequence is that the ingestion layer must derive a stable identifier, typically from the message
control identifier plus sending facility, and must tolerate out-of-order arrival.

HL7 version 2 is a convention adopted by trading partners, not a regulation. Its conformance
profile is defined by the interface specification between the source and the receiving system.
That specification is the binding document, and it is frequently incomplete.

### 8.2 FHIR R4 Subscription

Fast Healthcare Interoperability Resources (FHIR) Release 4 defines a `Subscription` resource in
the Trial Use maturity level 3. A subscription carries a `criteria` search string and a `channel`
with a `type` of `rest-hook`, `websocket`, `email`, `sms`, or `message`. For a `rest-hook` channel
without a payload, the server sends an empty POST and the subscriber re-queries with a `_since`
parameter. With a payload, the server forwards the resource as an update.

Three consequences for architecture:

1. The criteria are evaluated against the new resource value, so an update that no longer matches
   produces no notification.
2. There is no notification on delete in the R4 base resource.
3. Delivery is not guaranteed to be exactly once; the subscriber must be idempotent.

The R4 topic-based subscription backport implementation guide formalises a notification bundle with
a `Parameters` first entry and a `SubscriptionStatus` resource. If a design claims exactly-once
FHIR notifications, ask which implementation guide and which delivery guarantee the server
documents.

### 8.3 HIPAA Security Rule

The HIPAA Security Rule is at 45 CFR Part 160 and Subparts A and C of Part 164. The technical
safeguards at 45 CFR 164.312 include access control (required standard), audit controls (required),
integrity (required), person or entity authentication (required), and transmission security
(required). Within access control, unique user identification and emergency access procedure are
required specifications; automatic logoff and encryption and decryption are addressable.

"Addressable" does not mean optional. It means the covered entity must assess whether the
specification is reasonable and appropriate, implement it if it is, and document the decision if it
is not. An architecture that leaves audit logging to the application, with no platform-level
record, is making that assessment implicitly and badly.

What the rule does not do: it does not prescribe streaming, batch, Kubernetes, or any cloud service.
It constrains what must be logged, who may access what, and how data is protected in transit.

### 8.4 NIST SP 800-207

NIST Special Publication 800-207, "Zero Trust Architecture", published August 2020, defines zero
trust as a set of paradigms that move defenses from static network perimeters to users, assets,
and resources. Its seven tenets include: all data sources and computing services are resources;
all communication is secured regardless of network location; access is granted per session; access
is determined by dynamic policy; the enterprise monitors the integrity and security posture of all
assets; authentication and authorization are dynamic and strictly enforced; and the enterprise
collects state information to improve posture.

For architecture, the operative tenet is the second. A service-to-service call inside a Kubernetes
cluster is not trusted because of its address. This is why the reference network policy is
default-deny and why mutual transport layer security (mTLS) belongs in a production design.

### 8.5 AWS Well-Architected and the Healthcare Lens

The AWS Well-Architected Framework has six pillars: operational excellence, security, reliability,
performance efficiency, cost optimization, and sustainability. The Healthcare Industry Lens,
published 17 November 2022, applies the pillars to healthcare workloads and treats HIPAA and PHI as
the central regulatory driver. Its data protection guidance ties encryption at rest and in transit
to the Business Associate Addendum, and its isolation guidance recommends separate accounts and
organisational units for health data with tagging for attribute-based access control.

The lens is guidance, not certification. Passing a Well-Architected review does not make a workload
HIPAA compliant.

### 8.6 What none of these standards settle

None of them tells you whether to stream or batch. That remains an engineering judgement about
latency, cost, and operational capacity. The standards constrain the evidence you must retain and
the controls you must apply, whichever shape you choose.

> **Note:** The lab stack does not implement encryption in transit inside the cluster, does not
> enforce authentication at the ingress, and stores a placeholder secret. It is a teaching scaffold,
> not a compliant deployment.

## 9. Industry practice and case evidence

### 9.1 What is documented

**Hybrid is the norm.** The AWS Healthcare Industry Lens publishes separate reference architectures
for electronic health record and revenue cycle, interoperability, medical imaging, analytics, and
machine learning. The analytics reference explicitly directs architects to verify that every
service used is on the HIPAA Eligible Services list. The existence of separate references is itself
evidence that no single shape covers a health system.

**Event-driven interoperability is being standardised, not invented per site.** The HL7 FHIR
subscription backport implementation guide exists because R4 deployments needed a common
notification shape before R5 adoption. The CMS Interoperability and Patient Access final rule and
the ONC Cures Act final rule pushed FHIR-based exchange into US regulation, which in turn pushed
subscription semantics into procurement.

**Batch remains the default for analytics.** Population health, quality measurement, and value-based
care contracting are computed over closed periods. A streaming platform does not remove the need
for a month-end close, and the close is a batch job by definition.

### 9.2 What is inference

The following statements are inferences from the sources above, not documented findings:

- Most mid-sized health systems run a streaming path for a small number of operational use cases
  and a batch path for everything else. The cost arithmetic in Section 6 explains why.
- Teams that adopt streaming without a schema registry accumulate consumer-specific parsing code,
  which becomes the largest maintenance cost in the pipeline.
- The failure that most often triggers a re-architecture is not throughput but replay: a transform
  bug is found, and there is no retained raw data to reprocess.

### 9.3 Case evidence with names

Public, checkable cases in this space are mostly regulatory and standards cases rather than company
post-mortems. Three are worth knowing:

1. **FHIR R4 Subscription maturity level 3 (Trial Use).** The specification itself states the
   maturity level. A design that depends on subscription delivery for a safety-critical alert is
   depending on a Trial Use interface.
2. **NIST SP 800-207 (August 2020).** The US federal zero trust guidance that made network location
   an insufficient basis for trust. It is the reference used in most cloud security architecture
   reviews.
3. **AWS Well-Architected Healthcare Industry Lens (17 November 2022).** A published, citable
   application of the six pillars to healthcare, including explicit HIPAA mapping guidance.

What is not available: published, audited post-incident reports from health systems describing
streaming architecture failures. Those are rarely public. Any claim that "hospital X failed
because of Kafka" should be treated as unsourced unless a named report is produced.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Streaming chosen for a daily batch need | Broker cost exceeds the batch job by an order of magnitude | Always-on infrastructure for a periodic workload | Budget scrutiny and a stalled programme | Re-run the freshness ladder in Section 3.2 |
| Batch chosen for an alerting need | Clinicians keep a parallel manual process | Latency budget cannot be met by the schedule | Alert is ignored or shadowed | Measure event-to-action latency, not job duration |
| No landing layer | A transform bug requires re-extraction from the source | Raw input overwritten or discarded | Multi-day recovery, possible data loss | Immutable landing objects partitioned by date |
| Partition count fixed at creation | Lag rises during every peak | Consumer parallelism capped by partitions | Unbounded lag, missed SLO | Size partitions for peak with headroom; plan a repartition |
| Key skew | One partition's disk fills first | A high-volume patient or site dominates one key | Broker disk exhaustion, partition offline | Monitor per-partition bytes and rate; salt hot keys |
| Synchronous call in the ingest path | Ingest availability equals downstream availability | Availability coupling | Source system backlog during downstream outage | Buffer with a log or queue, or accept the coupling explicitly |
| No replay procedure | Nobody can answer "reprocess Tuesday" | Retention or tooling missing | Incident recovery depends on vendor support | Write and rehearse the procedure quarterly |
| Schema change without compatibility mode | Consumers fail after a producer deploy | Breaking field change | Pipeline halt, dead-letter flood | Registry with backward-transitive compatibility |
| Trust boundary assumed at the cluster edge | Any pod can reach any pod | Flat pod network | Lateral movement after a compromise | Default-deny network policy with an enforcing CNI |
| Single-zone data path | Zone maintenance causes an outage | No cross-zone replication | SLO breach during routine maintenance | Three zones, synchronous replication, minimum in-sync replicas |
| Unbounded dead-letter queue | DLQ grows silently | Poison records retried forever | Hidden data loss, disk growth | Alert on DLQ depth and age; triage procedure |
| Cost labels absent | Cannot attribute spend to a service | Shared untagged resources | No ability to right-size | Tag at creation, enforce with policy |
| Architecture diagram has no arrows' contracts | Reviewers ask what the latency is and nobody knows | Boxes without interfaces | Design cannot be tested | Annotate every arrow with the four contract terms |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Event volume is approximately stationary within a day | Queueing and sizing use a mean and a peak factor | New interface, seasonal campaign, backlog replay | Compare 30-day hourly counts; compute peak-to-mean ratio |
| Per-partition capacity of 1 event/s at p99 | Partition sizing | Larger payloads, synchronous downstream calls | Load test with production-shaped records |
| Duplicates are tolerable if writes are idempotent | At-least-once is cheaper than transactions | Financial or legal counting | Ask the data owner; inspect downstream constraints |
| The event identifier is stable across retries | Deduplication | Producer generates a UUID per attempt | Force a retry and compare identifiers |
| Object storage is effectively durable | Landing layer is the recovery source | Accidental deletion, bucket policy error | Enable versioning and test restore |
| Three availability zones are available | Replication and SLO arithmetic | Region with two zones, or quota limits | `aws ec2 describe-availability-zones` |
| On-demand prices are a valid planning basis | Cost comparison | Committed-use discounts, spot, savings plans | Recalculate with the actual commitment |
| The team can staff on-call | Streaming is operable | No rotation, single expert | Ask who answers at 02:00 |
| Regulatory retention is known | Storage layer design | State or contract variation | Obtain written retention schedule |
| The source schema is stable enough to encode | Schema registry design | Vendor upgrade, local customisation | Diff 90 days of messages |
| Network egress is not the dominant cost | Cost model | Cross-region replication, imaging | Measure egress in the bill |

## 12. Exercises

### 12.1 Classify five flows

**Deliverable.** A one-page table classifying five flows from a hospital you know or can research:
ADT, laboratory results, device telemetry, claims, and a research extract. For each: freshness
budget, volume per day, ordering requirement, loss tolerance, and recommended shape.

**Acceptance criteria.** Every flow has a numeric freshness budget and a named percentile. Every
recommendation cites one row of the Section 5.2 decision table. At least one recommendation must be
batch.

**Time.** 45 minutes.

### 12.2 Cost two variants

**Deliverable.** A spreadsheet or script computing the monthly cost of a batch and a streaming
variant of the same flow. State region, currency, month, instance types, and every price source.

**Acceptance criteria.** Arithmetic reproduces to the cent. Every price has a source and an access
date. The comparison states what the difference buys in latency, not only the ratio.

**Time.** 60 minutes.

### 12.3 Queueing estimate

**Deliverable.** For an arrival rate of 4 events per second at peak and a per-consumer service rate
of 1.5 events per second, compute the number of consumers needed for a utilisation no higher than
0.7, and the M/D/1 mean wait at that consumer count.

**Acceptance criteria.** Working shown. The answer states the assumption that service times are
deterministic and what changes if they are exponential.

**Time.** 30 minutes.

### 12.4 Choose a partition key

**Deliverable.** A short memo choosing between `patient_id`, `site_id`, and `event_id` as the key
for an ADT topic, with the ordering consequence, the skew risk, and the repartition procedure.

**Acceptance criteria.** Names at least one hot-key scenario. States what breaks when partitions
are added. Recommends a count for a stated peak rate.

**Time.** 40 minutes.

### 12.5 Trust boundary diagram

**Deliverable.** A diagram of a clinical data path from source system to serving API, marking every
trust boundary and the control at each.

**Acceptance criteria.** Includes the cluster-internal boundary, the managed-service boundary, and
the human access boundary. Each control is a named mechanism, not a category. At least one control
is marked as absent in the lab stack.

**Time.** 45 minutes.

### 12.6 Write an ADR

**Deliverable.** An ADR using the YAML skeleton in Section 7, for a decision in your own context.

**Acceptance criteria.** Contains status, context, decision, at least two alternatives with reasons
for rejection, consequences including one negative, reversal cost, and evidence that would reopen
the decision. Parses as YAML.

**Time.** 60 minutes.

### 12.7 Critique a diagram

**Deliverable.** A written critique of the architecture diagram in Section 3, identifying three
missing contracts and one missing failure path.

**Acceptance criteria.** Each finding names the arrow or component, the missing information, and the
consequence. No finding is a style comment.

**Time.** 30 minutes.

### 12.8 Design a hybrid

**Deliverable.** A design for a platform that serves both a five-second alert and a month-end
quality report, with one landing layer, one streaming path, and one batch path. Include the join
key and the reconciliation check.

**Acceptance criteria.** The two paths agree on a stated reconciliation query. The design states
which path owns patient state and why. Cost is estimated with the method in Section 6.

**Time.** 90 minutes.

## 13. Self-check answers

### 13.1 Classification example

A laboratory result feed from a reference laboratory, 8,000 results per day, 3 kB per result,
required within 15 minutes of result verification, no ordering requirement across patients,
duplicates tolerable if the result identifier is stable.

Freshness budget: p95 under 15 minutes. Volume: 24 MB per day. Ordering: none across patients,
per-result identifier stable. Loss tolerance: none for the result, but a retry is acceptable.
Recommended shape: micro-batch every five minutes, or a streaming consumer if the same platform
already exists. Batch is defensible. Streaming is not justified by the latency alone.

### 13.2 Cost arithmetic check

For the Section 6 flow, streaming is:

```text
3 × 0.0816 × 730 = 178.704
3 × 100 × 0.08    =  24.000
total             = 202.704
```

Batch is:

```text
2 × 30.4 × 0.1632 = 9.9226
1.08 × 0.023      = 0.0248
total             = 9.9474
```

Rounded to cents: USD 202.70 and USD 9.95. The script prints 9.94 because it uses the unrounded
storage figure; the difference is a cent and is not material. State the rounding convention in any
comparison.

### 13.3 Queueing check

Arrival rate 4 events per second. Service rate 1.5 per consumer. At utilisation 0.7:

```text
consumers = ceil(4 / (1.5 × 0.7)) = ceil(3.81) = 4
```

With four consumers, service rate is 6.0 per second and utilisation is 0.667. M/D/1 mean wait:

```text
Wq = 0.667 / (2 × 6.0 × 0.333) = 0.167 seconds
```

If service times are exponential rather than deterministic, the M/M/1 wait is twice as large at the
same utilisation, approximately 0.333 seconds. The deterministic assumption is optimistic, which is
why the target utilisation is kept below one.

### 13.4 Partition key

`patient_id` preserves per-patient ordering, which is what clinical state transitions need. It
risks skew if one patient generates a disproportionate share, which is rare for ADT but possible
for a long-stay patient with many transfers. `site_id` gives only four keys and severe skew.
`event_id` gives even distribution but no ordering, which breaks the state machine. Choose
`patient_id`; monitor per-partition byte rate; if a hot key appears, salt it with a bounded suffix
only if per-patient ordering can be reconstructed downstream.

### 13.5 Trust boundary

The cluster-internal boundary has no control in the lab stack beyond the network policy. That is a
real gap: the policy restricts which pods can talk, but the traffic is plaintext and the identity is
the pod's network address. A production design adds mTLS with workload identity.

### 13.6 ADR rubric

A strong ADR names a decision that could have gone the other way, gives a numerical basis for the
comparison, records a negative consequence, and defines the evidence that would reopen it. A weak
ADR restates the chosen design and calls the alternatives "not scalable".

### 13.7 Diagram critique

Three missing contracts: the arrow from the log to the stream processor does not state the delivery
guarantee; the arrow from the stream processor to the serving store does not state idempotency; the
arrow from the landing store to the batch transform does not state the schedule or the input
completeness rule. One missing failure path: no dead-letter destination is shown for the stream
processor, so a poison record has no documented route.

### 13.8 Hybrid design rubric

A strong design states that the streaming path owns current patient state and the batch path owns
historical facts, joins them on a stable patient identifier and an event timestamp, and runs a
daily reconciliation that counts events by type in both paths. A weak design duplicates the state
in both paths with no reconciliation and no statement of which is authoritative.

## 14. Interview questions

**1. When would you not use streaming?**
A strong answer starts from the freshness budget and the operational cost, gives a concrete example
such as a nightly claims file, and notes that streaming adds an on-call burden. A weak answer says
"when the volume is low", which is not the deciding factor.

**2. What does exactly-once actually require?**
A strong answer describes idempotent writes, a stable deduplication key, a transactional sink where
available, and replay-safe logic, and states that a broker setting alone is insufficient. A weak
answer names a Kafka configuration flag.

**3. How do you choose a partition key?**
A strong answer starts from the ordering requirement, then considers skew, then states the
repartition procedure and its cost. A weak answer says "use the primary key".

**4. What is the difference between event time and processing time, and why does it matter?**
A strong answer explains late arrival, watermarks, and the difference between a result that is
correct as of now and a result that is correct as of the event. A weak answer defines the terms
without consequence.

**5. How would you estimate the cost of this design?**
A strong answer names the units (instance-hours, GB-months, requests, egress), the region, the
month, and the source of each price, and separates fixed from variable cost. A weak answer gives a
single number without units.

**6. What is an error budget and how does it change behaviour?**
A strong answer defines it as the allowed SLO violation in a window, ties it to a release policy,
and gives the arithmetic for 99.9 percent over 30 days. A weak answer calls it "downtime
allowance".

**7. Where are the trust boundaries in a Kubernetes deployment?**
A strong answer distinguishes the cluster edge, the pod network, the managed-service endpoint, and
the human access path, and notes that network policy requires an enforcing CNI. A weak answer says
"the cluster is the boundary".

**8. What would you do if consumer lag rose during peak?**
A strong answer separates throughput limits (partitions) from processing limits (poll interval,
downstream latency) and names the diagnostic for each. A weak answer says "add consumers".

**9. Why keep a landing layer if the warehouse already has the data?**
A strong answer explains replay, audit, and schema drift, and notes that the warehouse holds
transformed data, not received data. A weak answer says "for backup".

**10. How do you decide between a queue and a log?**
A strong answer compares replay, fan-out, ordering, and acknowledgement semantics. A weak answer
treats them as interchangeable buffering.

## 15. Key terms

- **ADR** — Architecture decision record; see Section 5, Step 12.
- **At-least-once** — Delivery guarantee in which duplicates are possible; see Section 4.
- **Backfill** — Reprocessing history to populate or correct a derived dataset.
- **Batch** — Processing a bounded input at a scheduled or triggered time.
- **Blast radius** — The scope of resources affected by one failure.
- **BAA** — Business Associate Agreement required by HIPAA.
- **CDC** — Change data capture from a database transaction log.
- **Consumer group** — Consumers that divide partitions among themselves.
- **Curated layer** — Cleaned and standardised data derived from landing.
- **DLQ** — Dead-letter queue for records that cannot be processed.
- **Error budget** — Allowed SLO violation within a window.
- **Event time** — Time at which an event occurred at the source.
- **Exactly-once** — A system property in which the observable effect equals one processing.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **HIPAA** — Health Insurance Portability and Accountability Act.
- **Idempotency** — Property by which repeating an operation has the same effect as one execution.
- **Landing zone** — Immutable first storage of raw input.
- **Micro-batch** — Small bounded batches at short intervals.
- **mTLS** — Mutual transport layer security.
- **Offset** — Position of a record within a partition.
- **Outbox** — Pattern that writes state and event in one transaction.
- **Partition** — Ordered, append-only slice of a topic.
- **PHI** — Protected health information.
- **Replay** — Reprocessing retained events from an earlier offset.
- **RPO** — Recovery point objective, the maximum tolerable data loss in time.
- **RTO** — Recovery time objective, the maximum tolerable restoration time.
- **SLO** — Service level objective.
- **SLI** — Service level indicator.
- **Streaming** — Continuous processing of an unbounded event sequence.
- **Trust boundary** — Point at which identity or policy is enforced.
- **Watermark** — Bound on acceptable event lateness.

## 16. Sources

All sources accessed 26 October 2026.

1. NIST. *SP 800-207: Zero Trust Architecture*. August 2020.
   https://csrc.nist.gov/pubs/sp/800/207/final
2. HHS. *HIPAA Security Rule*, 45 CFR Part 160 and Subparts A and C of Part 164.
   https://www.hhs.gov/hipaa/for-professionals/security/index.html
3. Cornell Legal Information Institute. *45 CFR 164.312 — Technical safeguards*.
   https://www.law.cornell.edu/cfr/text/45/164.312
4. HL7. *FHIR R4 Subscription*. https://hl7.org/fhir/R4/subscription.html
5. HL7. *R4 Topic-Based Subscription Notification Bundle*, Subscriptions R5 Backport.
   https://build.fhir.org/ig/HL7/fhir-subscription-backport-ig/
6. AWS. *Well-Architected Framework — The pillars of the framework*.
   https://docs.aws.amazon.com/wellarchitected/latest/framework/the-pillars-of-the-framework.html
7. AWS. *Healthcare Industry Lens*.
   https://docs.aws.amazon.com/wellarchitected/latest/healthcare-industry-lens/healthcare-industry-lens.html
8. AWS. *Amazon EKS — Kubernetes version lifecycle*.
   https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html
9. AWS. *EC2 On-Demand pricing*, US East (Ohio), Linux. Price List API, retrieved 26 October 2026.
   https://aws.amazon.com/ec2/pricing/on-demand/
10. AWS. *Amazon S3 pricing*. https://aws.amazon.com/s3/pricing/
11. AWS. *Amazon EBS pricing*. https://aws.amazon.com/ebs/pricing/
12. Apache Software Foundation. *Kafka Producer Configs*.
    https://kafka.apache.org/documentation/#producerconfigs
13. Redpanda. *Topic Configuration Properties*.
    https://docs.redpanda.com/current/reference/properties/topic-properties/
14. Kubernetes. *Network Policies*.
    https://kubernetes.io/docs/concepts/services-networking/network-policies/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_containers.md`](02_containers.md)
- Related topic: [`05_streaming.md`](05_streaming.md)
- Related topic: [`08_cost-and-reliability.md`](08_cost-and-reliability.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Terraform root: [`../../labs/cloud/terraform/main.tf`](../../labs/cloud/terraform/main.tf)
- Kubernetes manifests: [`../../labs/cloud/k8s/`](../../labs/cloud/k8s/)
- Streaming producer: [`../../labs/cloud/kafka/producer.py`](../../labs/cloud/kafka/producer.py)
- Streaming consumer: [`../../labs/cloud/kafka/consumer.py`](../../labs/cloud/kafka/consumer.py)
