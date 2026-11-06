---
case_id: "CASE-STREAM-001"
case_title: "Recovering a hot partition without losing customer-event ordering"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "streaming-systems"
domain: "Real-time analytics and event processing"
primary_pattern: "Incident diagnosis and recovery"
secondary_patterns: ["Partition-aware capacity planning"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Streaming platform lead"
source_index: "INDEX.md"
review_trigger: "Changed ordering requirement, workload distribution, or consumer capacity"
---

# Recovering a hot partition without losing customer-event ordering

## 1. The One-Line Framing

Given customer events keyed by account, diagnose and contain a growing partition backlog for an operations team while preserving per-account ordering and avoiding duplicate downstream effects, instead of merely adding consumers to the existing group.
The operational decision is continuous; the immediate incident decision is whether to throttle, provision, or change the partitioning contract.
The primary success condition is timely processing of eligible events without missing or duplicating business effects.
The primary pattern is incident diagnosis and recovery; the secondary pattern is partition-aware capacity planning.

**Source declaration:** This entire scenario, its timestamps, architecture, traffic rates, capacities, targets and proposed actions are hypothetical teaching constructs, not an account of an actual incident.
Calculations are estimates derived from those illustrative inputs, not measurements or a simulation.
No intervention has been implemented or evaluated.
The important tradeoff is that one partition is assigned to at most one active consumer in the same consumer group; adding group members alone cannot parallelize that partition.
Averages hide this bottleneck.

## 2. Context & Background

An illustrative online marketplace uses account events to maintain a near-real-time operations dashboard.
A producer emits one event for each order-state transition.
A keyed log preserves the intended order of events within an account, and a consumer updates a materialized order view.
Operations staff use that view to find orders requiring attention; it is not the source of truth for order execution.
The source order service remains authoritative and can be queried when the dashboard is stale.
A stream-platform engineer owns the broker and assignment telemetry; an application engineer owns consumers and sink correctness.
The incident commander coordinates communications, the operations lead approves degraded workflows, and the security team owns access policy.
The assumed deployment is one region, one topic, one consumer group and 12 partitions.
All scenario times below use UTC and are hypothetical, not historical facts.
The current process hashes account ID to a partition, consumes each assigned partition serially, and commits offsets after durable writes.
This retains a simple per-account ordering contract but concentrates all events for a very large account on one partition.
There are no documented earlier attempts, actual customers, real deployment records, or measured outcomes in this teaching case.
No specific jurisdiction or legal obligation is asserted; retention and privacy requirements require local review before deployment.

## 3. The Business Question

**Illustrative stakeholder question:** How can we restore the dashboard's freshness without creating missing, duplicated, or incorrectly ordered order states?

| Element | Decision contract |
|---|---|
| Owner | Incident commander with application and platform leads |
| Trigger | Hypothetical per-partition lag grows during a burst |
| Deadline | Begin containment immediately; no unverified production cutover during the incident |
| Actions | Retain current design, throttle the source, tune processing, split the ordering key, or redesign aggregation |
| Objective | Maximize the fraction of eligible order events reflected in the dashboard within the freshness window |
| Hard constraints | No silent data loss, no duplicate state transition, and per-order ordering |
| Scope | One illustrative regional event pipeline and its dashboard |
| Exclusions | Changing payment processing, customer contracts, and the source-of-truth order service |

Lag is a diagnostic proxy, not the business objective.
A low committed-offset lag could coexist with missing effects if commits occur too early.
A temporarily larger lag can be preferable to a dashboard that is fast but wrong.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Check and owner | If wrong | Fallback / trigger |
|---|---|---|---|---|
| A-01 | Events of different orders may be processed concurrently | Product and data owners inspect state transitions | Splitting by order can violate a hidden account-wide invariant | Keep account key and provision or throttle; revisit on invariant change |
| A-02 | Event ID is globally unique and stable across retries | Producer owner audits source contract | Deduplication may suppress valid effects or repeat effects | Use source-generated composite identity and migration plan |
| A-03 | 1,000 events/s is sustainable for a hot consumer | Performance owner runs representative sink-inclusive load test | Drain estimate fails | Re-estimate and reduce ingress or improve actual bottleneck |
| A-04 | Source can throttle without losing accepted events | Producer owner exercises backpressure | Throttle may silently drop data | Buffer durably upstream or use operational fallback |
| A-05 | The sink supports an atomic deduplication and state update | Application owner performs crash-injection tests | Replay duplicates effects | Implement transactional inbox/outbox or reconcile side effects |
| C-01 | Preserve order within an order | Illustrative business constraint, pending approval | Parallel split becomes invalid | Retain account key until contract is clarified |
| C-02 | Never discard accepted events to improve freshness | Illustrative safety constraint, pending approval | Apparent recovery hides loss | Pause and reconcile durable source records |
| C-03 | Only approved operators may change offsets | Illustrative security constraint; security owner reviews | Accidental or unauthorized replay | Restrict and audit offset alteration |

The most consequential unknowns are the true ordering scope, sink transaction semantics, and sustainable partition service rate.
No assumption is represented as verified.
Costs and supplier prices are intentionally unpriced because no actual deployment or dated commercial quote is provided.

## 5. Data, Tools & Method

### 5a. Data Specification

One event represents one order transition, not one customer or one aggregate.
Proposed fields are event_id, account_id, order_id, order_sequence, event_time, produced_at, event_type, schema_version and payload.
The key in the baseline is account_id; the proposed new key is order_id only after approval of A-01.
Timestamps are UTC instants, and event_time may precede arrival because of retries or network delay.
The sink records the source topic, partition, offset, event_id, order_sequence and processing timestamp for reconciliation.
A valid comparison uses the same accepted event population and explicitly counts quarantined records.
No training set or model label exists; the correctness oracle is a source-to-sink reconciliation plus ordered transition checks.
A future replay sample must include normal accounts, the hot account, late arrivals, duplicates and malformed messages.
No workload trace, source snapshot or actual data is available.

### 5b. Tools & Stack

The proposed stack is a Kafka-compatible partitioned log, a consumer service, a transactional database for the view, and an independently controlled monitoring system.
Kafka semantics and version-specific configuration must be confirmed against the actual deployed version before use.
A broker exposes topic partitions and consumer-group offsets; the service owns application-level idempotency.
The database provides the hypothetical atomic transaction joining the inbox deduplication record to the view update.
No language version, cluster size or actual platform version is established.

### 5c. Method Summary

Inspect per-partition input rate and lag slope, compare them with consumer throughput, then discriminate hot-key skew from slow processing, assignment churn and broker trouble.
For immediate containment, preserve the existing key and lower the arrival rate or improve the constrained consumer's effective service rate.
For durable mitigation, consider routing by order_id only if the business ordering invariant permits it.

| Candidate | Evidence | Strength | Limitation | Decision |
|---|---|---|---|---|
| No change | Constructed inputs only | No migration risk | Backlog grows if arrival exceeds service | Baseline; not tested |
| Add same-group consumers | Partition assignment semantics, SRC-01 | Helps if multiple busy partitions share consumers | Cannot divide one partition across active group members | Not selected for the single hot partition |
| Throttle upstream | Hypothetical durable upstream | Can reverse lag slope without changing order | Transfers backlog and may affect customers | Proposed containment, subject to source checks |
| Key by order | Assumed order-level independence | Distributes different orders for a large account | Migration and ordering risks | Proposed only after invariant validation |
| Optimize sink | Hypothetical service bottleneck | Preserves key and ordering | May not meet burst rate | Benchmark before deciding |

### 5d. Data provenance, quality, exclusions and leakage controls

All numbers in this case are constructed inputs.
There is no private data extraction, observed production event, experiment, or external workload trace.
If implemented, access to event payloads should be limited to authorized service identities; diagnostics should use redacted identifiers and aggregate metrics.
Monitor missing event IDs, duplicate IDs, sequence gaps, clock skew, malformed schemas, and source-to-sink count discrepancies separately.
Do not compare produced event timestamps with processing timestamps without addressing clock synchronization and late arrivals.
Data retention, replay access, encryption and deletion requests require a real policy and review; none are assumed to be legally sufficient here.

### 5e. Reproducibility and method-selection ledger

| ID | Location and date | Supports | Limits |
|---|---|---|---|
| SRC-01 | [Apache Kafka documentation: introduction](https://kafka.apache.org/documentation/#intro_consumers), version unspecified, accessed for this draft as a method reference | Partitioned consumption and consumer groups | Does not establish this scenario or its performance |
| SRC-02 | [Kafka delivery semantics](https://kafka.apache.org/documentation/#semantics), version unspecified, method reference | Delivery and transactional semantics require explicit boundaries | Does not make an external database update exactly once |
| SRC-03 | [Google SRE Workbook: implementing SLOs](https://sre.google/workbook/implementing-slos/), method reference | User-centered service-level objectives | Does not validate this case's target |
| CALC-01 | Section 6 of this document | Recomputable analytical queue estimates | Constant rates and no failure or burst variation |
| RUN-01 | No run exists | Nothing | No implementation, benchmark, trace or measured outcome |

Source pages are public; deployed software versions and actual retrieval records require confirmation before operational reliance.
The hypothetical calculations require no random seed; no code revision, container image, data snapshot or run artifact exists.

**Claim ledger.** These identifiers distinguish teaching inputs, derived estimates and unevaluated operational claims.

| ID | Claim and scope | Evidence and source | Supports | Does not establish | Recheck trigger |
|---|---|---|---|---|---|
| CLM-01 | One group member at a time consumes a partition in the same group | Published method reference, SRC-01 | Consumer-count limitation in Sections 5c and 6 | Actual deployed assignment or service rate | Kafka version or group protocol change |
| CLM-02 | 1,400/s arrival and 1,000/s service for 900 seconds produce 360,000 queued events from zero | Illustrative inputs and estimated arithmetic, CALC-01 | Section 7 conditional recommendation | An observed backlog | Changed workload or initial backlog |
| CLM-03 | At 600/s arrival and 1,000/s service, 360,000 events drain in 900 seconds | Estimated, CALC-01 | Section 7 sensitivity and Section 9 test design | A measured recovery or user-facing outcome | Changed service rate or buffer policy |
| CLM-04 | User freshness and correctness improved | Not measured; no run artifact | Nothing yet | Any outcome in Section 11 | Approved rollout and independent measurement |
| CLM-05 | Atomic inbox and view writes can prevent duplicate database effects under the stated crash sequence | Proposed design; transaction semantics require test | Section 8 failure handling | Exactly-once delivery to arbitrary external services | Sink technology or side-effect boundary changes |

## 6. Analysis Walkthrough

The decisive arithmetic is that one hypothetical partition receiving 1,400 events/s while processing 1,000 events/s accumulates 400 events/s, even if other partitions are idle.
This is a conditional estimate, not a finding from an incident.

| Step | Input and operation | Intermediate result | Interpretation and limitation |
|---|---|---|---|
| 1 | 12 partitions; one hot partition receives 1,400/s | Hot rate 1,400/s | Assignment and input skew must be inspected per partition |
| 2 | Hypothetical service rate 1,000/s | Net +400/s | Consumer capacity is below ingress under assumed steady rates |
| 3 | Initial lag 0; overload lasts 15 minutes | 400 × 900 = 360,000 events | Excludes initial backlog, bursts, failures and retention effects |
| 4 | Throttle hot ingress to 600/s; capacity remains 1,000/s | Net drain 400/s | Requires durable, safe upstream backpressure |
| 5 | 360,000 / 400 | 900 seconds, or 15 minutes | Lower-bound-style steady-rate projection, not measured recovery time |
| 6 | Recover under 800/s ingress | 360,000 / 200 = 1,800 seconds | A modest ingress change doubles estimated drain time |
| 7 | Recover under 1,000/s ingress | Zero net drain | Backlog does not drain at equal rates |

The queue model is B(t) = max(0, B(0) + (arrival rate − service rate) × t) for constant rates and available backlog.
Its units are events, events per second, and seconds.
It does not predict tail latency, source buffering, rebalances, failure retries or transaction overhead.
Partition end offset minus committed offset measures an offset-distance approximation, not necessarily a precise count of user-visible outstanding business events; transactional markers, compaction and gaps require care.
Inspect the age of the oldest unprocessed event and compare source, log and sink timestamps to identify user impact.
A competing hypothesis is slow sink writes: a hot partition might have normal ingress but a slow consumer.
Compare partition ingress with processing throughput, sink latency, error rates, CPU and assignment churn before choosing a mitigation.
No approach was actually tried first; all alternatives remain untested.

## 7. The Finding

For the hypothetical workload, preserve the existing ordering contract during containment and reduce hot-partition ingress below demonstrated service capacity; consider a new order-level key only after the ordering invariant and migration are proven.
At the illustrative constant rates, a 360,000-event backlog would take 15 minutes to drain at a net 400 events/s, versus no drain when arrival equals service.
Neither the backlog nor the recovery is a measured outcome, and service rates may vary substantially in reality.
The next justified action is to validate the bottleneck, ordering boundary, sink atomicity and upstream buffering before any production change.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

There is no model training path.
A proposed batch reconciliation reads immutable source events and a versioned sink snapshot for the same event-time and ingestion-time bounds.
It compares event IDs and order sequences, records missing and duplicate effects, and produces a reviewed repair manifest.
Snapshot watermarks and delayed arrivals must be recorded so late data is not misclassified as lost.
Repairs are applied through the same idempotency rules as live traffic, with separate rate limits and audit records.

```text
Source snapshot + bounds -> validate and deduplicate -> compare sink snapshot
                                    |                       |
                                    v                       v
                              quarantine report      repair manifest -> human approval
                                                                   -> controlled replay
```

### 8b. Online / Serving / Real-Time Path

```text
Authenticated producer -> schema validation -> keyed topic partition -> consumer group
        |                         |                                    |
        v                         v                                    v
  durable upstream          quarantine + alert                inbox + view transaction
                                                                      |
                                                                      v
                                                           commit next source offset
                                                                      |
                                                                      v
                                              dashboard + freshness indicator -> operator
```

Producer identity and topic access are authorized before publish; credentials are not included in event payloads.
A valid event has a stable event ID and per-order sequence.
The consumer reads a record, validates its schema and expected sequence, and inserts the event ID in an inbox table within the same transaction as the view update.
A duplicate inbox key means the effect is already durable and must not be applied again.
Only after the transaction commits may the consumer commit the next offset.
If it crashes between database commit and offset commit, replay repeats the record but the inbox prevents a second effect.
If it commits the offset first and crashes before the database transaction, the effect can be lost; that ordering is prohibited.
A downstream notification outside the transaction requires its own idempotency or transactional outbox; a database transaction cannot atomically cover an arbitrary remote API.
A malformed record is durably quarantined with source position and reason; only an explicitly approved policy may advance past it.
A sequence gap holds the affected order for reconciliation; it must not silently apply a later transition as if no predecessor existed.
At cold start, restore view and deduplication state from an aligned backup or rebuild from retained source events.
At peak load, throttle accepted ingress or serve a visibly stale dashboard with source-of-truth escalation; never drop accepted events to hide lag.
If the database is unavailable, pause or back off with jitter and bounded in-flight records; avoid committing unfinished work.
Proposed timeouts, retry limits and freshness thresholds require workload measurement and owner approval.

### 8c. Key Design Decisions

| ID | Choice | Alternative and tradeoff | Failure boundary / revisit |
|---|---|---|---|
| DEC-01 | Maintain account key for containment | Repartition immediately might increase parallelism but risks reordered transitions | Revisit after explicit order-level invariant approval |
| DEC-02 | Deduplicate and update view atomically | Offset commits alone are simpler but cannot protect external effects | Revisit if sink cannot transact both writes |
| DEC-03 | Separate user freshness from internal lag | Lag alone is cheap but can mask data loss | Revisit when source or dashboard semantics change |
| DEC-04 | Cut over at an explicit source boundary | Uncoordinated dual writing can reorder or duplicate | Revisit if producers cannot fence old writes |

### 8d. Architecture Artifacts and Operational Readiness

**Interfaces.** Schema version and compatibility policy travel with each event; identity, sequence, UTC timestamps and type are mandatory.
Invalid schemas are quarantined rather than interpreted using a guessed version.
The input contract distinguishes source acceptance from broker acknowledgement and sink completion.
The output contract includes order ID, latest applied sequence, freshness timestamp and a stale indicator.

**State and retention.** The application owns inbox and materialized view; the platform owns topic configuration and access control.
Inbox retention must cover the maximum replay and restore horizon, or old events need a stronger durable deduplication strategy.
Backups must capture a consistent database state and corresponding source positions; otherwise a restore must reconcile before serving.
Source retention must exceed the longest outage plus detection, repair and replay time, with headroom for uncertainty.

**Sizing.** At the hypothetical peak, 1,400 events/s on one partition exceeds its 1,000 events/s service rate by 40% of service capacity.
With an assumed average serialized payload of 2 KiB, that partition receives 2,800 KiB/s, approximately 2.73 MiB/s before replication and protocol overhead.
A 360,000-event backlog at 2 KiB/event occupies approximately 703 MiB of payload before metadata, indexes, replicas and compression.
With three replicas, a naive payload-only copy estimate is approximately 2.06 GiB; it is not a storage quote or a durability guarantee.
If each processed event requires one database transaction, the partition demands up to 1,000 transactions/s at the assumed service rate; batching changes both transaction rate and failure granularity.
Memory depends on in-flight messages, decompression, client buffers and sink state; for 500 in-flight events at 2 KiB, raw payload alone is about 1 MiB, not a process memory budget.
Network and disk limits, replica traffic, retries, CPU, database contention and event-size distributions require benchmarks.
No actual concurrency, instance count, hardware, spend or utilization is measured.

**Recovery and migration.** Fence or pause producers at a documented cutover watermark, drain the old topic through that watermark, and verify per-order state and counts.
Start new-key writes only after the boundary is safe, and persist a routing epoch so the same order cannot alternate between old and new paths unknowingly.
If producers cannot pause, a sequencing and buffering protocol is required; simply dual-writing and switching readers is not sufficient to preserve cross-topic order.
Rollback after new-key processing cannot blindly resume the old topic because new events and state may already exist; reconcile and establish another fenced boundary first.
Restore tests must include crash before sink commit, after sink commit before offset commit, and after offset commit.

**Observability and ownership.** The platform lead monitors partition ingress, end and committed offsets, consumer assignments, rebalance frequency and broker health.
The application lead monitors processing duration, transaction failure, dedup hits, sequence gaps, quarantine and oldest unprocessed event age.
Alert on sustained positive lag slope and oldest-event age, with source and sink diagnostics linked in the runbook.
The operations lead owns user-facing stale indicators and manual source-of-truth fallback.
Audited, least-privilege identities control offset resets, replay, topic changes and quarantine access.

**Release and economics.** Shadow processing must not emit side effects, and its outputs are compared with the authoritative view before a canary.
Canary and rollback gates include missing effects, ordering violations, freshness and cost; a failed correctness gate stops rollout.
Compute, storage replicas, network, database writes, retained inbox entries, on-call time and replay labor are cost drivers, but no price is asserted.
A real budget requires dated regional prices and a measured useful-event denominator; cost per useful outcome cannot be calculated here.

## 9. Evaluation & Validation

The proposed primary service-level indicator is the number of eligible accepted events visible in the dashboard within five minutes divided by all eligible accepted events in a rolling 30-day window.
The denominator is defined at durable source acceptance, including events later quarantined; otherwise dropping or quarantining records could artificially improve the metric.
The proposed target is 99.9%, subject to stakeholder approval, and is not a measured service-level objective.
Dashboard visibility must be joined back to stable event identity, with clock skew and late-arriving source records assessed.
The baseline is the existing account-key design under matched input traces and the same correctness oracle.

| Evaluation item | Proposed protocol |
|---|---|
| Correctness gate | Zero unexplained missing, duplicate, or out-of-order applied order transitions in the finite test corpus |
| Freshness gate | At least 99.9% within five minutes in the agreed window, with no omission from denominator |
| Load | Replay a versioned trace and synthetic hot-account bursts, including peak and sustained overload |
| Recovery | Inject consumer and database crashes at the three commit boundaries; reconcile source, inbox, view and committed offsets |
| Repartition | Exercise fence, watermark, late old-key arrival, rollback and replay against order-sequence oracle |
| Guardrails | Quarantine volume, oldest event age, upstream backlog, cost, sink errors and business workflow availability |
| Sample and uncertainty | Include independent workload periods where possible; report per-period results and a block-bootstrap interval for correlated events, not an independent-event binomial interval |
| Validity threat | Synthetic burst, unrepresentative event size, warm caches or omitted sink effects can overstate performance |
| Artifacts | No trace, configuration, run ID, query or test artifact exists yet |
| Decision | Not executed; no pass or failure can be claimed |

Predefine the workload mix, observation window, thresholds and stop rules before running the comparison.
A finite test showing zero errors does not prove the true error probability is zero.
The team should calculate sensitivity and required duration from the risk tolerance and dependency structure before treating a clean run as evidence of a rare-error rate.
A shadow test checks compatibility but cannot establish user impact unless the actual user path is included.

## 10. Action Taken

No action taken as of 2026-11-07; this is an illustrative exercise, not an incident record.
The recommendation is to validate assumptions, prepare a throttling runbook, and only then evaluate an order-key migration.
Approval is proposed from the incident commander, application owner, platform owner and operations owner; no approval record exists.
Implementation, ownership handoff, scope, canary dates and effective deployment remain unassigned.
A production rollout would stop on any unexplained missing effect, duplicate business effect, order violation or unacceptable upstream backlog.
Rollback requires a fenced boundary and reconciliation rather than an unqualified offset reset.
Changing the source-of-truth order workflow is excluded.

## 11. Outcome & Results

**Source declaration:** The scenario's figures are constructed teaching values, not measurements of any named organization or evidence of realized impact.

| Metric | Baseline | After | Change | Evidence / limitation |
|---|---|---|---|---|
| Five-minute freshness SLI | Not measured | Not measured | Not calculable | No intervention or observation |
| Missing or duplicate effects | Not measured | Not measured | Not calculable | Requires reconciliation across accepted population |
| Backlog drain time | Not measured | Not measured | Not calculable | CALC-01 is only a conditional estimate, not a result |
| Operational cost | Not measured | Not measured | Not calculable | No workload or priced resource record |

Outcome not yet measured.
The proposed measurement owner is the application lead; the proposed observation window is 30 days after any approved rollout.
A failed correctness gate or sustained freshness breach would trigger review and a stop decision.
No adoption, realized saving or causal improvement can be asserted.

## 12. Failure Modes & What Went Wrong

No actual incident occurred; the following timeline is entirely hypothetical and is included only to practice diagnosis.

| UTC | Hypothetical event | Teaching implication |
|---|---|---|
| 10:00 | Hot-key arrival rises above assumed service capacity | This is a constructed onset, not a measured event |
| 10:05 | Oldest-event age and per-partition slope trigger an alert | Detection delay would be five minutes under this invented sequence |
| 10:08 | Incident commander checks assignment, sink latency and ingress | Distinguish skew from downstream slowness and rebalances |
| 10:12 | Operator proposes upstream backpressure | Confirm durability and business approval before applying it |
| 10:15 | Hypothetical containment starts at 600 events/s | Backlog at this point would be 360,000 under Section 6 assumptions |
| 10:30 | Hypothetical backlog reaches zero under constant rates | This is arithmetic, not actual restoration evidence |
| 10:35 | Reconciliation is scheduled | Zero lag alone does not establish correct state |
| Next day | Owners review retention and partition-key alternatives | Follow-up has not occurred |

Under these invented times, detection time is 10:05 − 10:00 = 5 minutes; containment time is 10:15 − 10:00 = 15 minutes; backlog-drain time is 10:30 − 10:15 = 15 minutes.
Full recovery would require correctness and freshness confirmation, not just an empty queue, so its endpoint is not specified by this timeline.

| ID | Hypothetical symptom / cause | Discriminating check | Containment and preventive owner |
|---|---|---|---|
| F-01 | Single partition grows; one account dominates its key | Compare partition ingress and key distribution with other partitions | Rate-limit safely; platform owner tests skew alert |
| F-02 | All partitions grow because database writes slow | Compare transaction latency, error rate and per-partition throughput | Restore sink capacity or buffer; application owner tests dependency failure |
| F-03 | Repeated group rebalances stop progress | Inspect assignment and rebalance events alongside processing pauses | Stabilize consumers; platform owner tests slow-poll behavior |
| F-04 | Offset advances but effects are missing | Reconcile accepted source IDs against durable sink IDs | Stop unsafe commits; application owner tests crash boundaries |
| F-05 | Replay repeats a notification | Trace event ID through external side effect | Pause notification and reconcile; application owner introduces outbox/idempotency |
| F-06 | New partition key reorders old and new records | Compare per-order sequence across cutover watermark | Fence writes; migration owner tests late arrival and rollback |
| F-07 | Quarantine hides a widespread schema break | Alert on rejected fraction and producer version | Stop incompatible rollout; schema owner validates compatibility |
| F-08 | Source throttling shifts overload to an unbounded buffer | Measure upstream queue age, size and acceptance failures | Bound intake and invoke manual workflow; producer owner tests backpressure |
| F-09 | Replay horizon exceeds inbox retention | Compare oldest replay offset with dedup retention | Halt replay; data owner reconciles and redesigns retention |
| F-10 | Dashboard appears fresh while accepted records are omitted | Join source acceptance to visible event identities | Mark view unreliable; operations owner uses authoritative source |

Fault tree: freshness failure can arise from insufficient partition service, downstream outage, consumer reassignment, or missing records that appear falsely complete.
Hot-key skew is a hypothesis until per-key and per-partition evidence separates it from these alternatives.
The hypothetical alert design would fail if it only measured aggregate topic lag or omitted event age; no real alarm failure is claimed.
Adding consumers is a plausible but untested mitigation that does not split a single partition in one group.
Offset reset is not a safe universal rollback: it may replay external effects or skip uncommitted work.


### Diagnostic field guide: hypothetical incident variants

Each distinct exercise below is hypothetical; no symptom, mitigation or outcome is an observation from a real system.
The diagnostic distinguishes a competing mechanism from hot-key overload and names an accountable response owner.

**Diagnostic 001: Key concentration.**
- Symptom: One account supplies most records on one partition.
- Discriminating evidence: Compare sampled key frequencies to partition ingress.
- Proposed response: Apply approved account-level admission control.
- Decision risk: Throttling can delay legitimate orders.
- Responsible role: Producer owner.

**Diagnostic 002: Hash collision.**
- Symptom: Several medium accounts share the overloaded partition.
- Discriminating evidence: Map privacy-safe key hashes to partition assignments.
- Proposed response: Balance permissible keys in a new routing epoch.
- Decision risk: Rehashing changes ordering boundaries.
- Responsible role: Platform owner.

**Diagnostic 003: Single huge order.**
- Symptom: One order dominates traffic after order-key repartition.
- Discriminating evidence: Plot rate per order rather than per account.
- Proposed response: Reduce redundant source updates while preserving transitions.
- Decision risk: Splitting an order can violate sequence.
- Responsible role: Application owner.

**Diagnostic 004: Uneven partition count.**
- Symptom: The topic has fewer partitions than concurrently busy keys.
- Discriminating evidence: Measure utilization and backlog per partition.
- Proposed response: Plan a fenced repartition with a larger count.
- Decision risk: More partitions increase broker and operational overhead.
- Responsible role: Platform owner.

**Diagnostic 005: Partition expansion.**
- Symptom: The producer's hash mapping changes after adding partitions.
- Discriminating evidence: Compare old and new key-to-partition maps.
- Proposed response: Treat expansion as a migration with a controlled fence.
- Decision risk: The same key can be observed on different partitions.
- Responsible role: Migration owner.

**Diagnostic 006: Producer batching.**
- Symptom: Large producer batches create apparent periodic lag spikes.
- Discriminating evidence: Overlay batch flush timestamps with ingress and age.
- Proposed response: Tune batching against end-to-end latency requirements.
- Decision risk: Smaller batches can increase request overhead.
- Responsible role: Producer owner.

**Diagnostic 007: Producer retry storm.**
- Symptom: Repeated sends inflate ingress after acknowledgement timeouts.
- Discriminating evidence: Compare producer attempts to unique event IDs.
- Proposed response: Bound retries with jitter and preserve stable IDs.
- Decision risk: Dropping retries may lose unacknowledged accepted work.
- Responsible role: Producer owner.

**Diagnostic 008: Duplicate source event.**
- Symptom: The producer republishes the same transition with a new transport offset.
- Discriminating evidence: Compare event IDs and sequence values across records.
- Proposed response: Apply transactional deduplication at the sink.
- Decision risk: Unstable event IDs defeat the deduplication rule.
- Responsible role: Application owner.

**Diagnostic 009: Reused event identity.**
- Symptom: Distinct valid transitions carry the same event ID.
- Discriminating evidence: Compare payload and order sequence for duplicate IDs.
- Proposed response: Quarantine identity collisions for source-owner adjudication.
- Decision risk: Blind deduplication can suppress legitimate transitions.
- Responsible role: Data owner.

**Diagnostic 010: Sequence gap.**
- Symptom: Sequence 18 arrives when the view expects 17.
- Discriminating evidence: Compare source history with the consumer's per-order state.
- Proposed response: Hold the affected order and request the missing event.
- Decision risk: Blocking the whole partition unnecessarily expands impact.
- Responsible role: Application owner.

**Diagnostic 011: Sequence regression.**
- Symptom: An older transition arrives after a newer one was applied.
- Discriminating evidence: Inspect stored order sequence and producer epoch.
- Proposed response: Reject or reconcile the stale update according to the contract.
- Decision risk: Ignoring sequence can regress visible order state.
- Responsible role: Application owner.

**Diagnostic 012: Concurrent source writers.**
- Symptom: Two producers emit conflicting sequences for one order.
- Discriminating evidence: Trace source transaction IDs and writer identities.
- Proposed response: Restore a single sequencing authority for the order.
- Decision risk: A consumer cannot infer intended order from timestamps alone.
- Responsible role: Source owner.

**Diagnostic 013: Clock skew.**
- Symptom: An event appears to have arrived before it was produced.
- Discriminating evidence: Compare producer and broker clocks with a trusted time source.
- Proposed response: Flag age calculations that cross unreliable clocks.
- Decision risk: Negative apparent age can hide stale processing.
- Responsible role: Platform owner.

**Diagnostic 014: Late event.**
- Symptom: An event arrives after the reconciliation cutoff.
- Discriminating evidence: Compare ingestion time with event time and the declared watermark.
- Proposed response: Reopen or amend the reconciliation window.
- Decision risk: A premature missing-event declaration causes false repair.
- Responsible role: Data owner.

**Diagnostic 015: Timestamp unit mismatch.**
- Symptom: Seconds are interpreted as milliseconds.
- Discriminating evidence: Inspect schema version and timestamp magnitude.
- Proposed response: Quarantine invalid records and repair the producing version.
- Decision risk: Automatic coercion can make plausible but wrong timestamps.
- Responsible role: Schema owner.

**Diagnostic 016: Payload growth.**
- Symptom: The same event rate yields unexpectedly high network throughput.
- Discriminating evidence: Compare serialized size percentiles and recent schema changes.
- Proposed response: Limit payload size and move large objects to controlled storage.
- Decision risk: Average event size conceals large outliers.
- Responsible role: Producer owner.

**Diagnostic 017: Compression change.**
- Symptom: CPU rises while logical event rate is unchanged.
- Discriminating evidence: Compare compression ratio and CPU by producer version.
- Proposed response: Revert or tune compression after benchmarking.
- Decision risk: Network savings may be offset by consumer CPU.
- Responsible role: Platform owner.

**Diagnostic 018: Decompression bomb.**
- Symptom: A small compressed message expands beyond memory limits.
- Discriminating evidence: Inspect expansion ratio in an isolated validation path.
- Proposed response: Enforce decompressed-size limits and quarantine the record.
- Decision risk: Rejecting a legitimate oversized event requires reconciliation.
- Responsible role: Security owner.

**Diagnostic 019: Schema incompatibility.**
- Symptom: New producer records fail deserialization.
- Discriminating evidence: Compare writer schema with deployed consumer schema.
- Proposed response: Pause the incompatible producer version and quarantine failures.
- Decision risk: Skipping poison records can hide missing business effects.
- Responsible role: Schema owner.

**Diagnostic 020: Unknown event type.**
- Symptom: The schema is valid but the transition type is unsupported.
- Discriminating evidence: Compare event type with the consumer's supported contract.
- Proposed response: Route the event to a durable exception workflow.
- Decision risk: Treating unknown types as harmless can corrupt state.
- Responsible role: Application owner.

**Diagnostic 021: Missing required identifier.**
- Symptom: Records arrive without a stable event ID.
- Discriminating evidence: Measure missing-ID rate by producer version.
- Proposed response: Reject before acceptance or durably quarantine after acceptance.
- Decision risk: Synthetic IDs may not survive a producer retry.
- Responsible role: Producer owner.

**Diagnostic 022: Invalid tenant identity.**
- Symptom: The message tenant disagrees with authorized producer identity.
- Discriminating evidence: Compare authentication context with signed tenant metadata.
- Proposed response: Deny cross-tenant writes and preserve audit evidence.
- Decision risk: Trusting payload tenant alone enables boundary crossing.
- Responsible role: Security owner.

**Diagnostic 023: Oversized individual record.**
- Symptom: One record consistently exceeds fetch or processing limits.
- Discriminating evidence: Inspect rejection reason and serialized record size.
- Proposed response: Quarantine and repair through an approved large-payload path.
- Decision risk: Repeated unbounded retries can block partition progress.
- Responsible role: Platform owner.

**Diagnostic 024: Poison message.**
- Symptom: One deterministic record repeatedly crashes the consumer.
- Discriminating evidence: Replay the single record in an isolated test environment.
- Proposed response: Quarantine with source position and explicit approval to advance.
- Decision risk: Advancing without preserving the record creates silent loss.
- Responsible role: Application owner.

**Diagnostic 025: Transient parse dependency.**
- Symptom: A schema registry timeout resembles a bad message.
- Discriminating evidence: Compare registry availability with record-specific failures.
- Proposed response: Retry with bounded backoff and restore the dependency.
- Decision risk: Quarantining all records during outage misclassifies healthy data.
- Responsible role: Platform owner.

**Diagnostic 026: Registry authorization.**
- Symptom: Consumers receive permission errors fetching schemas.
- Discriminating evidence: Check service identity and registry policy changes.
- Proposed response: Restore least-privilege access through the approved workflow.
- Decision risk: Broad emergency grants can outlive the incident.
- Responsible role: Security owner.

**Diagnostic 027: Schema cache expiry.**
- Symptom: A previously healthy consumer fails after cache eviction.
- Discriminating evidence: Correlate failures with schema-cache misses.
- Proposed response: Restore registry reachability and verify cache behavior.
- Decision risk: A warm-cache test misses cold-start risk.
- Responsible role: Application owner.

**Diagnostic 028: Deserializer deployment.**
- Symptom: Only one application version fails on a valid schema.
- Discriminating evidence: Compare version-specific error traces on the same record.
- Proposed response: Roll back the consumer binary if backward compatible.
- Decision risk: Rollback may be unsafe after state migration.
- Responsible role: Application owner.

**Diagnostic 029: Garbage collection pause.**
- Symptom: Processing pauses while memory pressure and collection time rise.
- Discriminating evidence: Compare pause durations with poll intervals and throughput.
- Proposed response: Reduce allocation pressure or tune the runtime after measurement.
- Decision risk: Excessive tuning without heap evidence can worsen pauses.
- Responsible role: Application owner.

**Diagnostic 030: Memory leak.**
- Symptom: Resident memory grows across otherwise stable workload intervals.
- Discriminating evidence: Compare retained-object profiles across comparable snapshots.
- Proposed response: Restart safely and fix the retained reference.
- Decision risk: Repeated restarts can trigger assignment churn.
- Responsible role: Application owner.

**Diagnostic 031: CPU saturation.**
- Symptom: Consumer CPU remains near its effective quota while ingress grows.
- Discriminating evidence: Profile time by deserialization, validation and sink operations.
- Proposed response: Remove the measured bottleneck or adjust resource allocation.
- Decision risk: More CPU cannot fix a serialized downstream lock.
- Responsible role: Application owner.

**Diagnostic 032: Container throttling.**
- Symptom: Host CPU is available but the container hits its quota.
- Discriminating evidence: Inspect cgroup throttling metrics and resource limits.
- Proposed response: Adjust limits after checking cluster capacity and fairness.
- Decision risk: Raising limits can starve neighboring workloads.
- Responsible role: Platform owner.

**Diagnostic 033: Slow sink transaction.**
- Symptom: Database commit latency increases with consumer lag.
- Discriminating evidence: Compare transaction latency distribution and database wait events.
- Proposed response: Reduce contention or restore database capacity.
- Decision risk: Increasing consumer concurrency may deepen database overload.
- Responsible role: Database owner.

**Diagnostic 034: Connection pool exhaustion.**
- Symptom: Consumers wait for database connections despite low compute utilization.
- Discriminating evidence: Inspect pool occupancy, wait time and leaked connections.
- Proposed response: Restore a bounded pool and fix long-held connections.
- Decision risk: Increasing pool size can exceed database capacity.
- Responsible role: Application owner.

**Diagnostic 035: Database lock contention.**
- Symptom: Transactions wait on shared rows even for different orders.
- Discriminating evidence: Inspect lock wait graph and affected row keys.
- Proposed response: Reduce lock scope while preserving atomicity.
- Decision risk: Removing locks without an invariant can corrupt balances.
- Responsible role: Database owner.

**Diagnostic 036: Hot aggregate row.**
- Symptom: Many orders update one account summary row.
- Discriminating evidence: Measure lock waits and update rates for summary keys.
- Proposed response: Separate order-level processing from safely aggregated summaries.
- Decision risk: Asynchronous summaries can be temporarily inconsistent.
- Responsible role: Data owner.

**Diagnostic 037: Database deadlock.**
- Symptom: Concurrent transactions abort with deadlock errors.
- Discriminating evidence: Inspect deadlock graphs and transaction ordering.
- Proposed response: Retry aborted transactions with bounded backoff.
- Decision risk: Unbounded retries can amplify load and mask a structural conflict.
- Responsible role: Database owner.

**Diagnostic 038: Database failover.**
- Symptom: Writes fail briefly during a primary election.
- Discriminating evidence: Correlate write errors with database topology events.
- Proposed response: Pause consumption and retry within a bounded budget.
- Decision risk: Committing offsets during write failures loses effects.
- Responsible role: Database owner.

**Diagnostic 039: Read replica staleness.**
- Symptom: The dashboard reads an outdated replica after the sink commits.
- Discriminating evidence: Compare primary and replica positions and dashboard freshness.
- Proposed response: Expose staleness and route critical checks to the authoritative source.
- Decision risk: Fresh consumer offsets do not prove fresh reads.
- Responsible role: Database owner.

**Diagnostic 040: Disk exhaustion.**
- Symptom: Broker or sink writes fail as local storage fills.
- Discriminating evidence: Compare disk utilization, retention and write error timestamps.
- Proposed response: Free capacity through approved retention or capacity expansion.
- Decision risk: Deleting retained events can make replay impossible.
- Responsible role: Platform owner.

**Diagnostic 041: Disk latency.**
- Symptom: Broker append latency rises before disk fills.
- Discriminating evidence: Inspect disk service time and broker request queues.
- Proposed response: Rebalance storage or reduce contention after identifying the device.
- Decision risk: Free space alone does not establish healthy throughput.
- Responsible role: Platform owner.

**Diagnostic 042: Broker leader change.**
- Symptom: Ingress or fetch latency spikes during partition leadership changes.
- Discriminating evidence: Correlate partition metadata changes and client errors.
- Proposed response: Restore stable leadership and inspect underlying broker health.
- Decision risk: Repeated manual leader moves can increase disruption.
- Responsible role: Platform owner.

**Diagnostic 043: Under-replicated partition.**
- Symptom: Replication falls behind for the hot partition.
- Discriminating evidence: Inspect replica lag and broker network and disk metrics.
- Proposed response: Reduce load or restore replica capacity under durability policy.
- Decision risk: Reducing replication guarantees to improve latency risks loss.
- Responsible role: Platform owner.

**Diagnostic 044: Network saturation.**
- Symptom: Broker or consumer links approach their measured capacity.
- Discriminating evidence: Compare byte throughput and packet loss on the relevant paths.
- Proposed response: Relieve the constrained link or reduce unnecessary payload.
- Decision risk: Application-level retry storms can hide the original network bottleneck.
- Responsible role: Network owner.

**Diagnostic 045: Cross-zone transfer.**
- Symptom: Traffic placement creates unexpected network cost and latency.
- Discriminating evidence: Trace broker leader, consumer and database placement.
- Proposed response: Consider locality without sacrificing fault isolation.
- Decision risk: Co-locating every replica defeats zone resilience.
- Responsible role: Platform owner.

**Diagnostic 046: DNS failure.**
- Symptom: Consumers cannot locate brokers after restart.
- Discriminating evidence: Compare DNS resolution errors with broker availability.
- Proposed response: Restore name resolution and use bounded retry.
- Decision risk: A healthy existing connection may conceal cold-start failure.
- Responsible role: Network owner.

**Diagnostic 047: TLS certificate expiry.**
- Symptom: New connections fail while existing sessions continue.
- Discriminating evidence: Inspect certificate expiry and handshake errors.
- Proposed response: Rotate certificates through the approved identity process.
- Decision risk: Disabling certificate verification exposes interception risk.
- Responsible role: Security owner.

**Diagnostic 048: Credential rotation.**
- Symptom: A consumer starts failing after its secret changes.
- Discriminating evidence: Compare credential version and authorization logs.
- Proposed response: Refresh credentials and validate least privilege.
- Decision risk: Copying credentials into incident logs exposes secrets.
- Responsible role: Security owner.

**Diagnostic 049: Authorization revocation.**
- Symptom: Only one topic or consumer group becomes inaccessible.
- Discriminating evidence: Inspect denied operations and policy changes.
- Proposed response: Restore the minimum required permission after approval.
- Decision risk: Granting wildcard access broadens the attack surface.
- Responsible role: Security owner.

**Diagnostic 050: Consumer crash.**
- Symptom: An assigned partition repeatedly loses its active processor.
- Discriminating evidence: Correlate process exits with assignment changes.
- Proposed response: Stabilize the process and inspect the root exception.
- Decision risk: Restart loops can postpone useful processing indefinitely.
- Responsible role: Application owner.

**Diagnostic 051: Slow polling.**
- Symptom: Long processing prevents timely group participation.
- Discriminating evidence: Compare processing duration with configured poll and session limits.
- Proposed response: Bound work per poll and manage in-flight processing safely.
- Decision risk: Increasing timeouts delays detection of a genuinely dead consumer.
- Responsible role: Application owner.

**Diagnostic 052: Rebalance cascade.**
- Symptom: Membership changes repeatedly interrupt useful processing.
- Discriminating evidence: Inspect join and leave events with deployment and restart timelines.
- Proposed response: Stabilize membership before adjusting group settings.
- Decision risk: A larger timeout cannot fix a persistent crash loop.
- Responsible role: Platform owner.

**Diagnostic 053: Uneven assignment.**
- Symptom: One consumer owns several busy partitions while peers are underused.
- Discriminating evidence: Compare assignment and per-partition service demand.
- Proposed response: Reassign or add consumers when busy partitions can be distributed.
- Decision risk: One hot partition remains indivisible within the group.
- Responsible role: Platform owner.

**Diagnostic 054: Idle consumers.**
- Symptom: More consumers exist than assigned partitions.
- Discriminating evidence: Inspect group membership and partition ownership.
- Proposed response: Scale consumers only when assignment and demand justify it.
- Decision risk: Extra instances incur cost without increasing partition concurrency.
- Responsible role: Platform owner.

**Diagnostic 055: Wrong consumer group.**
- Symptom: An application accidentally joins a different group.
- Discriminating evidence: Compare configured group ID with the intended deployment manifest.
- Proposed response: Restore the correct identity after assessing offset position.
- Decision risk: A new group may start at an unsafe default offset.
- Responsible role: Application owner.

**Diagnostic 056: Accidental offset reset.**
- Symptom: Committed offsets jump backward or forward unexpectedly.
- Discriminating evidence: Audit operator actions and compare offsets with sink state.
- Proposed response: Stop consumption and construct a reviewed replay plan.
- Decision risk: Forward reset can skip records; backward reset can repeat effects.
- Responsible role: Platform owner.

**Diagnostic 057: Offset commit ahead.**
- Symptom: Committed position exceeds the last durable sink effect.
- Discriminating evidence: Compare commit logs against transaction completion markers.
- Proposed response: Stop unsafe commits and reconcile the affected range.
- Decision risk: Lag can appear healthy despite missing business effects.
- Responsible role: Application owner.

**Diagnostic 058: Offset commit delayed.**
- Symptom: Effects are durable but offsets remain behind.
- Discriminating evidence: Compare inbox entries with the committed offset and retry logs.
- Proposed response: Restore offset commits while retaining deduplication.
- Decision risk: Replay is expected and must not duplicate effects.
- Responsible role: Application owner.

**Diagnostic 059: Commit during rebalance.**
- Symptom: An old owner attempts a late commit after reassignment.
- Discriminating evidence: Inspect generation errors and revoke callbacks.
- Proposed response: Fence revoked work and commit only completed contiguous offsets.
- Decision risk: Committing around unfinished records can skip work.
- Responsible role: Application owner.

**Diagnostic 060: Out-of-order asynchronous completion.**
- Symptom: A later record finishes before an earlier record in the same partition.
- Discriminating evidence: Compare completion positions with the contiguous commit frontier.
- Proposed response: Commit only the highest contiguous completed offset.
- Decision risk: Committing the maximum completed offset can lose earlier work.
- Responsible role: Application owner.

**Diagnostic 061: In-flight queue growth.**
- Symptom: Consumer memory grows while the sink slows.
- Discriminating evidence: Measure in-flight count, queue age and sink completion rate.
- Proposed response: Apply bounded queues and pause fetching when capacity is exhausted.
- Decision risk: An unbounded buffer relocates lag into process memory.
- Responsible role: Application owner.

**Diagnostic 062: Retry amplification.**
- Symptom: A sink outage generates more retry traffic than original work.
- Discriminating evidence: Measure attempts per event and downstream error saturation.
- Proposed response: Apply exponential backoff, jitter and a retry budget.
- Decision risk: Aggressive retries can prevent the dependency from recovering.
- Responsible role: Application owner.

**Diagnostic 063: Circuit breaker misconfiguration.**
- Symptom: A breaker repeatedly opens and closes during recovery.
- Discriminating evidence: Compare failure windows, probe rate and dependency capacity.
- Proposed response: Tune bounded half-open probes against measured recovery.
- Decision risk: A permanently open breaker prevents useful recovery.
- Responsible role: Application owner.

**Diagnostic 064: Timeout mismatch.**
- Symptom: Client timeouts expire before the database normally commits.
- Discriminating evidence: Compare timeout settings with the actual latency distribution.
- Proposed response: Set a justified deadline and handle ambiguous completion idempotently.
- Decision risk: A timeout does not prove the remote operation failed.
- Responsible role: Application owner.

**Diagnostic 065: Ambiguous transaction outcome.**
- Symptom: A client loses its connection during database commit.
- Discriminating evidence: Query the inbox and view by event ID before retrying.
- Proposed response: Replay safely through the idempotency boundary.
- Decision risk: Assuming failure can duplicate an already committed effect.
- Responsible role: Application owner.

**Diagnostic 066: External notification duplicate.**
- Symptom: An email is sent twice after a process crash.
- Discriminating evidence: Trace notification identity and delivery attempts.
- Proposed response: Use a transactional outbox and receiver idempotency where supported.
- Decision risk: An outbox alone cannot force an external receiver to deduplicate.
- Responsible role: Application owner.

**Diagnostic 067: Outbox backlog.**
- Symptom: Database updates succeed but notifications are delayed.
- Discriminating evidence: Compare unsent outbox age and dispatcher throughput.
- Proposed response: Restore dispatcher service with bounded retries.
- Decision risk: Reporting only consumer lag misses delayed downstream actions.
- Responsible role: Application owner.

**Diagnostic 068: Quarantine overload.**
- Symptom: Rejected records accumulate faster than operators can inspect them.
- Discriminating evidence: Measure quarantine rate, age and producer-version distribution.
- Proposed response: Pause the faulty source or scale a controlled review workflow.
- Decision risk: A quarantine queue without retention and ownership becomes data loss.
- Responsible role: Operations owner.

**Diagnostic 069: Quarantine access leak.**
- Symptom: Exception records expose sensitive payloads to broad audiences.
- Discriminating evidence: Audit access policy and sample redacted records.
- Proposed response: Restrict access and minimize retained payload fields.
- Decision risk: Logging complete malformed messages can expose credentials.
- Responsible role: Security owner.

**Diagnostic 070: Quarantine replay loop.**
- Symptom: A repaired record repeatedly returns to quarantine.
- Discriminating evidence: Trace event ID, repair revision and validation failure reason.
- Proposed response: Stop the loop and correct the underlying data or parser.
- Decision risk: Unbounded replay wastes capacity and conceals unresolved errors.
- Responsible role: Data owner.

**Diagnostic 071: Dedup retention expiry.**
- Symptom: A replay reaches records whose inbox IDs were already deleted.
- Discriminating evidence: Compare replay start with the oldest retained deduplication key.
- Proposed response: Stop replay or rebuild into an isolated reconciled target.
- Decision risk: Expired identities make old duplicates indistinguishable.
- Responsible role: Data owner.

**Diagnostic 072: Dedup table growth.**
- Symptom: The inbox consumes increasing storage and slows lookups.
- Discriminating evidence: Measure table size, index growth and retention horizon.
- Proposed response: Partition or archive dedup state without weakening replay guarantees.
- Decision risk: Deleting IDs solely by age can invalidate disaster recovery.
- Responsible role: Database owner.

**Diagnostic 073: Backup inconsistency.**
- Symptom: A restored view and source offsets refer to different moments.
- Discriminating evidence: Compare backup transaction watermark with stored source positions.
- Proposed response: Reconcile from a known consistent boundary before serving.
- Decision risk: A successful backup job does not prove a recoverable pair.
- Responsible role: Recovery owner.

**Diagnostic 074: Corrupt backup.**
- Symptom: Restore completes but state fails integrity checks.
- Discriminating evidence: Run a checksum and sampled source-to-view reconciliation.
- Proposed response: Restore another verified snapshot or rebuild from retained events.
- Decision risk: Backup completion is not equivalent to restore success.
- Responsible role: Recovery owner.

**Diagnostic 075: Retention too short.**
- Symptom: The oldest needed source records have already expired.
- Discriminating evidence: Compare outage and repair horizon to retained offsets.
- Proposed response: Escalate for source-of-truth backfill and disclose the gap.
- Decision risk: Offset reset cannot recreate deleted records.
- Responsible role: Platform owner.

**Diagnostic 076: Compacted source log.**
- Symptom: Older transitions disappear before a full historical rebuild.
- Discriminating evidence: Inspect topic cleanup policy and actual retained history.
- Proposed response: Use an immutable archive or a validated snapshot-plus-log scheme.
- Decision risk: A latest-value log cannot necessarily reproduce every transition.
- Responsible role: Data owner.

**Diagnostic 077: Replay competes with live traffic.**
- Symptom: Repair workers reduce capacity for current orders.
- Discriminating evidence: Compare live freshness with replay throughput and shared resource use.
- Proposed response: Rate-limit replay and reserve live-processing capacity.
- Decision risk: Aggressive catch-up can make current users less well served.
- Responsible role: Recovery owner.

**Diagnostic 078: Replay wrong interval.**
- Symptom: A repair includes events outside the approved range.
- Discriminating evidence: Compare manifest offsets and timestamps with the incident boundary.
- Proposed response: Stop replay and validate the manifest before resuming.
- Decision risk: Time-based bounds alone can omit late records.
- Responsible role: Recovery owner.

**Diagnostic 079: Replay wrong environment.**
- Symptom: A test worker connects to production instead of staging.
- Discriminating evidence: Inspect environment identity and destination authorization.
- Proposed response: Revoke credentials and contain any unintended writes.
- Decision risk: Similar endpoint names can bypass human expectations.
- Responsible role: Security owner.

**Diagnostic 080: Cross-topic ordering.**
- Symptom: A transition is distributed across two topics without a shared order.
- Discriminating evidence: Compare producer sequence numbers across both paths.
- Proposed response: Introduce explicit sequencing or consolidate the ordering boundary.
- Decision risk: Each topic's internal order does not imply cross-topic order.
- Responsible role: Architecture owner.

**Diagnostic 081: Dual-write gap.**
- Symptom: The source writes the old topic but fails before writing the new topic.
- Discriminating evidence: Reconcile source transaction IDs across both topics.
- Proposed response: Use an atomic outbox or fenced single-write migration.
- Decision risk: Blind dual writing permits divergence.
- Responsible role: Migration owner.

**Diagnostic 082: Dual-write duplicate.**
- Symptom: The same event arrives through old and new topics.
- Discriminating evidence: Compare stable event IDs and routing epochs.
- Proposed response: Deduplicate before applying the business effect.
- Decision risk: Different IDs for the same logical transition defeat deduplication.
- Responsible role: Migration owner.

**Diagnostic 083: Late cutover record.**
- Symptom: An old-route event appears after the new route starts.
- Discriminating evidence: Inspect routing epoch, source sequence and watermark.
- Proposed response: Hold the affected order and reconcile before applying later events.
- Decision risk: Assuming the old route is empty from one lag sample is unsafe.
- Responsible role: Migration owner.

**Diagnostic 084: Rollback divergence.**
- Symptom: The new route has already applied effects absent from the old route.
- Discriminating evidence: Compare source sequences and sink state at the proposed rollback fence.
- Proposed response: Establish a new boundary and replay the missing interval safely.
- Decision risk: Simply switching the consumer back can regress or duplicate state.
- Responsible role: Migration owner.

**Diagnostic 085: Canary contamination.**
- Symptom: A shadow consumer accidentally writes to the production sink.
- Discriminating evidence: Inspect destination permissions and side-effect logs.
- Proposed response: Revoke shadow write access and reconcile affected effects.
- Decision risk: A configuration flag alone is weaker than a separate identity boundary.
- Responsible role: Release owner.

**Diagnostic 086: Canary selection bias.**
- Symptom: The canary excludes the hot account that motivated the change.
- Discriminating evidence: Compare canary key and event-size distributions with target traffic.
- Proposed response: Include representative stress cases before expanding rollout.
- Decision risk: A healthy low-load canary says little about hot-key capacity.
- Responsible role: Release owner.

**Diagnostic 087: Canary correctness failure.**
- Symptom: One order transition is duplicated during a small rollout.
- Discriminating evidence: Reconcile event IDs and order sequence against the baseline.
- Proposed response: Stop rollout and investigate before expanding scope.
- Decision risk: A low error percentage may still violate the zero-loss contract.
- Responsible role: Release owner.

**Diagnostic 088: Feature flag inconsistency.**
- Symptom: Producers disagree about the active routing epoch.
- Discriminating evidence: Compare signed configuration versions across producer instances.
- Proposed response: Fence stale producers until they receive the approved configuration.
- Decision risk: A partially propagated flag can split one key across routes.
- Responsible role: Release owner.

**Diagnostic 089: Dashboard cache staleness.**
- Symptom: The sink is current but users still see old values.
- Discriminating evidence: Compare sink commit time with cache insertion and expiry.
- Proposed response: Invalidate or label the stale cache while investigating.
- Decision risk: Consumer lag does not measure presentation-layer delay.
- Responsible role: Dashboard owner.

**Diagnostic 090: Dashboard clock error.**
- Symptom: Freshness indicators disagree across clients.
- Discriminating evidence: Compare server-derived age and client clock readings.
- Proposed response: Compute the indicator using a trusted server clock.
- Decision risk: Client clock drift can create false freshness assurance.
- Responsible role: Dashboard owner.

**Diagnostic 091: Missing SLI denominator.**
- Symptom: Quarantined source-accepted events are excluded from the metric.
- Discriminating evidence: Join accepted IDs against all final processing states.
- Proposed response: Correct the metric and mark historical comparisons unreliable.
- Decision risk: A high SLI with missing inputs is not user success.
- Responsible role: Metrics owner.

**Diagnostic 092: Lag offset illusion.**
- Symptom: Offset distance falls while data is missing in the sink.
- Discriminating evidence: Reconcile source event IDs with applied business effects.
- Proposed response: Treat the dashboard as unreliable until missing effects are repaired.
- Decision risk: A committed offset is not proof of a successful external effect.
- Responsible role: Metrics owner.

**Diagnostic 093: Lag average masking.**
- Symptom: Average lag is small while one partition is severely delayed.
- Discriminating evidence: Inspect maximum and distribution of partition age and lag slope.
- Proposed response: Alert on affected partition and user-facing age.
- Decision risk: Averages can hide concentrated customer harm.
- Responsible role: Platform owner.

**Diagnostic 094: Noisy lag alert.**
- Symptom: A brief batch creates spikes that resolve without user impact.
- Discriminating evidence: Compare spike duration with oldest event age and SLO window.
- Proposed response: Tune alert duration without removing per-partition visibility.
- Decision risk: Excessive smoothing can delay detection of sustained overload.
- Responsible role: On-call owner.

**Diagnostic 095: Alert routing failure.**
- Symptom: The correct alert fires but reaches no responder.
- Discriminating evidence: Inspect delivery logs, on-call schedule and acknowledgement time.
- Proposed response: Restore the escalation route and test paging end to end.
- Decision risk: A configured threshold alone does not create an effective response.
- Responsible role: On-call owner.

**Diagnostic 096: Runbook authority gap.**
- Symptom: Operators see the issue but cannot approve safe throttling.
- Discriminating evidence: Review the incident command and product approval matrix.
- Proposed response: Name a delegated approver and an emergency escalation path.
- Decision risk: Unauthorized mitigation can harm customers or breach commitments.
- Responsible role: Incident commander.

**Diagnostic 097: Conflicting incident commands.**
- Symptom: Two operators concurrently change offsets and ingress limits.
- Discriminating evidence: Compare change audit entries and command timestamps.
- Proposed response: Use a single incident commander and record each action.
- Decision risk: Simultaneous changes make causal diagnosis difficult.
- Responsible role: Incident commander.

**Diagnostic 098: Overbroad access.**
- Symptom: Every responder can reset production offsets.
- Discriminating evidence: Review actual role grants and audit logs.
- Proposed response: Limit mutation rights to approved operators.
- Decision risk: Broad emergency access increases accidental and malicious change risk.
- Responsible role: Security owner.

**Diagnostic 099: Customer communication delay.**
- Symptom: Operations staff keep trusting a stale dashboard.
- Discriminating evidence: Compare first known freshness breach with stakeholder notification.
- Proposed response: Expose a stale banner and notify the operations lead.
- Decision risk: Technical recovery alone does not undo decisions made with stale data.
- Responsible role: Operations owner.

**Diagnostic 100: Business fallback saturation.**
- Symptom: Manual source-of-truth checks exceed staff capacity.
- Discriminating evidence: Measure manual queue age and staffing during a rehearsal.
- Proposed response: Prioritize critical orders and define an escalation threshold.
- Decision risk: A fallback that cannot handle peak load is not a complete plan.
- Responsible role: Operations owner.

**Diagnostic 101: SLO target mismatch.**
- Symptom: A five-minute freshness target is inadequate for a critical workflow.
- Discriminating evidence: Interview workflow owners and examine decision deadlines.
- Proposed response: Set separate indicators or priority classes where justified.
- Decision risk: A single aggregate target can conceal harm to urgent orders.
- Responsible role: Product owner.

**Diagnostic 102: Clock-window contamination.**
- Symptom: Replay records are counted as newly accepted events in the SLI.
- Discriminating evidence: Inspect source acceptance timestamps and replay metadata.
- Proposed response: Preserve original event identity and acceptance time in measurement.
- Decision risk: Replays can artificially change both numerator and denominator.
- Responsible role: Metrics owner.

**Diagnostic 103: Synthetic-load blind spot.**
- Symptom: A benchmark uses uniform keys and misses the hot account.
- Discriminating evidence: Compare test key-frequency distribution with representative observations.
- Proposed response: Include skewed and bursty traces before approving capacity.
- Decision risk: Uniform load may falsely validate a partitioning scheme.
- Responsible role: Performance owner.

**Diagnostic 104: Warm-cache benchmark.**
- Symptom: A benchmark passes only after all schemas and database pages are cached.
- Discriminating evidence: Measure a cold-start run alongside steady state.
- Proposed response: Include restarts and dependency cache misses in the test plan.
- Decision risk: Steady-state throughput does not establish recovery behavior.
- Responsible role: Performance owner.

**Diagnostic 105: Short benchmark.**
- Symptom: A test ends before sustained backlog begins to grow.
- Discriminating evidence: Graph arrival and service over a sufficiently long constant-load period.
- Proposed response: Extend the run and include burst and drain phases.
- Decision risk: A brief peak throughput number is not a sustainable rate.
- Responsible role: Performance owner.

**Diagnostic 106: Unrepresentative payload.**
- Symptom: A load test uses small messages but production includes large payloads.
- Discriminating evidence: Compare event-size and compression distributions.
- Proposed response: Replay a representative size mix and large outliers.
- Decision risk: Event rate alone does not determine byte or CPU demand.
- Responsible role: Performance owner.

**Diagnostic 107: Correlated test failures.**
- Symptom: A million events all exercise the same code path.
- Discriminating evidence: Stratify by key, schema, failure boundary and workload period.
- Proposed response: Report coverage and use uncertainty methods reflecting dependence.
- Decision risk: Independent-trial confidence can greatly overstate evidence.
- Responsible role: Validation owner.

**Diagnostic 108: Incomplete crash injection.**
- Symptom: Tests cover crashes before commit but not after sink commit.
- Discriminating evidence: Map every durable boundary and inject faults on both sides.
- Proposed response: Add the missing crash point and reconcile effects.
- Decision risk: A passing happy path cannot establish idempotent replay.
- Responsible role: Validation owner.

**Diagnostic 109: Unpriced capacity.**
- Symptom: The design recommends more resources without a cost basis.
- Discriminating evidence: List peak resource quantities and obtain dated regional prices.
- Proposed response: Estimate cost per useful event including retries and operations.
- Decision risk: Payload-only storage is not a complete cost model.
- Responsible role: FinOps owner.

**Diagnostic 110: Unreviewed privacy retention.**
- Symptom: Inbox IDs are kept indefinitely without a data policy.
- Discriminating evidence: Classify identifiers and review retention and deletion obligations.
- Proposed response: Choose an approved retention and replay architecture together.
- Decision risk: Deleting early can break replay; retaining forever can increase exposure.
- Responsible role: Privacy owner.

**Diagnostic 111: False source acceptance.**
- Symptom: The producer reports acceptance before durable persistence.
- Discriminating evidence: Compare client acknowledgements with source commit and broker records.
- Proposed response: Change the acceptance boundary before relying on the SLI denominator.
- Decision risk: A crash can erase records the customer was told were accepted.
- Responsible role: Producer owner.

**Diagnostic 112: Missing audit correlation.**
- Symptom: Incident changes cannot be linked to an operator or ticket.
- Discriminating evidence: Compare control-plane audit entries with incident timestamps.
- Proposed response: Require change identifiers and authenticated operator identity.
- Decision risk: Without attribution a harmful offset reset may be missed.
- Responsible role: Security owner.

**Diagnostic 113: Clock step during incident.**
- Symptom: A host clock jumps backwards during processing.
- Discriminating evidence: Compare monotonic durations with wall-clock timestamps.
- Proposed response: Use monotonic clocks for durations and UTC for event chronology.
- Decision risk: Elapsed-time calculations can become negative or misleading.
- Responsible role: Platform owner.

**Diagnostic 114: Producer partitioner drift.**
- Symptom: Two producer versions compute different partitions for the same key.
- Discriminating evidence: Run contract vectors through both partitioner implementations.
- Proposed response: Pin and migrate the partitioner version with explicit fencing.
- Decision risk: Per-key ordering fails even without a topic partition-count change.
- Responsible role: Producer owner.

**Diagnostic 115: Serialization key drift.**
- Symptom: Whitespace or Unicode normalization changes the bytes used as a partition key.
- Discriminating evidence: Compare canonicalized key bytes across producer versions.
- Proposed response: Define canonical encoding and test representative identifiers.
- Decision risk: Equivalent business keys can be routed to different partitions.
- Responsible role: Schema owner.

**Diagnostic 116: Null partition key.**
- Symptom: A producer unexpectedly omits the account key.
- Discriminating evidence: Measure null-key fraction by producer and schema version.
- Proposed response: Reject or quarantine according to the acceptance contract.
- Decision risk: Round-robin routing breaks the intended account ordering.
- Responsible role: Producer owner.

**Diagnostic 117: Partition affinity loss.**
- Symptom: A worker hands records to a generic concurrent executor.
- Discriminating evidence: Trace start and finish order for records of the same order.
- Proposed response: Enforce a per-order sequencing gate and contiguous commit frontier.
- Decision risk: Parallel execution can reorder effects despite ordered fetching.
- Responsible role: Application owner.

**Diagnostic 118: Stuck low-volume partition.**
- Symptom: A partition has little backlog but a very old unprocessed record.
- Discriminating evidence: Compare oldest-record age with offset distance per partition.
- Proposed response: Investigate a blocked record rather than suppressing the age alert.
- Decision risk: An event-count threshold alone misses a stranded customer.
- Responsible role: On-call owner.

**Diagnostic 119: Lost source-to-sink lineage.**
- Symptom: The view contains state but no originating event identity.
- Discriminating evidence: Inspect schema and trace fields for representative transitions.
- Proposed response: Persist event and source-position lineage with the state update.
- Decision risk: Reconciliation cannot reliably distinguish loss from transformation.
- Responsible role: Data owner.

**Diagnostic 120: Misleading end-to-end percentile.**
- Symptom: Teams add component p99 values and call the sum a measured system p99.
- Discriminating evidence: Inspect the trace-level latency distribution and aggregation method.
- Proposed response: Measure end-to-end latency directly from accepted IDs.
- Decision risk: Marginal percentiles do not determine a joint percentile.
- Responsible role: Metrics owner.

**Diagnostic 121: Retry after permanent rejection.**
- Symptom: An invalid order transition retries indefinitely.
- Discriminating evidence: Classify error codes into transient and permanent categories.
- Proposed response: Quarantine permanent failures with an operator repair path.
- Decision risk: Retrying deterministic errors consumes capacity without progress.
- Responsible role: Application owner.

**Diagnostic 122: Retry classification drift.**
- Symptom: A database authorization error is treated as temporary forever.
- Discriminating evidence: Inspect retry age and failure class distribution.
- Proposed response: Escalate persistent authorization failures after a bounded budget.
- Decision risk: An unavailable operator workflow leaves accepted events unresolved.
- Responsible role: Application owner.

**Diagnostic 123: Watermark misuse.**
- Symptom: A reconciliation treats event-time watermark as proof every source event arrived.
- Discriminating evidence: Compare source acceptance and ingestion timestamps around the boundary.
- Proposed response: Include a late-data allowance and explicit completeness check.
- Decision risk: A watermark is an operational assumption rather than proof of completeness.
- Responsible role: Data owner.

**Diagnostic 124: Replica recovery surge.**
- Symptom: Restoring a broker triggers heavy replication during backlog drain.
- Discriminating evidence: Correlate replica catch-up bytes with client latency.
- Proposed response: Coordinate replication and replay rates while preserving durability.
- Decision risk: Competing recovery traffic can extend the customer-facing outage.
- Responsible role: Platform owner.

**Diagnostic 125: Partial region outage.**
- Symptom: The log remains reachable but the dashboard dependency fails.
- Discriminating evidence: Compare user-path availability with individual component health.
- Proposed response: Expose an explicit degraded mode and authoritative fallback.
- Decision risk: Healthy broker metrics do not establish a functioning user workflow.
- Responsible role: Incident commander.

**Diagnostic 126: Retention policy change.**
- Symptom: An emergency cleanup shortens replay history during a prolonged incident.
- Discriminating evidence: Audit retention configuration changes against recovery requirements.
- Proposed response: Obtain recovery-owner approval before reducing retention.
- Decision risk: Short-term disk relief can destroy the only replay source.
- Responsible role: Platform owner.

**Diagnostic 127: Overloaded reconciliation.**
- Symptom: The correctness audit scans a primary database during peak recovery.
- Discriminating evidence: Measure audit query impact on transaction latency.
- Proposed response: Use an isolated snapshot and bounded query resources.
- Decision risk: An aggressive audit can worsen the incident it is validating.
- Responsible role: Database owner.

**Diagnostic 128: Unfenced former owner.**
- Symptom: A revoked consumer continues a side effect after reassignment.
- Discriminating evidence: Trace consumer generation and sink write times.
- Proposed response: Fence writes using ownership epochs or an equivalent verified mechanism.
- Decision risk: Partition assignment alone may not cancel already running work.
- Responsible role: Application owner.

**Diagnostic 129: Incorrect dedup scope.**
- Symptom: Two tenants legitimately use the same event ID namespace.
- Discriminating evidence: Inspect the uniqueness contract and inbox composite key.
- Proposed response: Include tenant identity in the deduplication key if required.
- Decision risk: Global deduplication can erase a different tenant's valid transition.
- Responsible role: Data owner.

**Diagnostic 130: Reconciliation collision.**
- Symptom: A hashed diagnostic identifier collides for distinct source events.
- Discriminating evidence: Inspect collision handling and compare authorized original identities.
- Proposed response: Use a collision-resistant keyed identifier with documented scope.
- Decision risk: Assuming hashes prove identity can misclassify missing records.
- Responsible role: Security owner.

**Diagnostic 131: Restore without producer fence.**
- Symptom: Replay begins while producers emit unordered replacement transitions.
- Discriminating evidence: Compare source epochs and restore start times.
- Proposed response: Establish the authorized write epoch before serving restored state.
- Decision risk: Concurrent unfenced writes can invalidate recovery reconciliation.
- Responsible role: Recovery owner.

**Diagnostic 132: Operator fatigue.**
- Symptom: Repeated low-value alerts make responders slow to acknowledge a severe breach.
- Discriminating evidence: Review alert frequency, acknowledgement and actionable responses.
- Proposed response: Remove non-actionable pages and route diagnostic signals appropriately.
- Decision risk: Suppressing alerts without alternate coverage can delay real detection.
- Responsible role: On-call owner.

**Diagnostic 133: Insufficient upstream visibility.**
- Symptom: Downstream lag falls but source requests accumulate elsewhere.
- Discriminating evidence: Compare accepted, rejected and queued requests across the boundary.
- Proposed response: Expose upstream queue age and customer-visible admission failures.
- Decision risk: Local recovery can simply transfer the outage to another component.
- Responsible role: Producer owner.

**Diagnostic 134: Silent repair mutation.**
- Symptom: A repair worker directly edits a view without a recorded manifest.
- Discriminating evidence: Compare database audit entries with approved repair IDs.
- Proposed response: Stop direct mutations and require reviewed, replayable repair records.
- Decision risk: Untraceable changes defeat later source-to-sink reconciliation.
- Responsible role: Recovery owner.

**Diagnostic 135: Unbounded incident scope.**
- Symptom: Operators cannot identify which customer orders are affected.
- Discriminating evidence: Join affected offsets and time windows to authorized order identifiers.
- Proposed response: Communicate a bounded impact set with uncertainty to operations.
- Decision risk: Assuming all customers or none were affected misdirects response.
- Responsible role: Incident commander.

## 13. Transferable Lesson

The initial hypothesis is that key skew can make a single partition the capacity boundary even when average cluster utilization is low.
The worked arithmetic supports that conditional mechanism, but no experiment has validated it for an actual deployment.
The necessary preconditions are partition-affine processing, a rate imbalance, and sufficient telemetry to distinguish arrival from service.
Changing the key helps only when the business permits independent ordering across the new keys and a safe migration can be demonstrated.
Avoid repartitioning when an account-wide invariant needs total order or when no reliable boundary can fence old and new routes.
For an IoT system, replace order ID with device ID only if each device's actions are independent; test the invariant and peak device distribution first.
Structural constraints are ordering and per-partition ownership; the particular rate, threshold, database and target are local illustrative choices.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Target | Shared mechanism / boundary |
|---|---|---|
| Uses method | [Kafka consumer groups](https://kafka.apache.org/documentation/#intro_consumers) | Partition ownership constrains same-group parallelism; deployed version must be checked |
| Uses method | [SLO workbook](https://sre.google/workbook/implementing-slos/) | User-centered service indicators; numerical target here is hypothetical |
| Collection | [Collection index](INDEX.md) | Catalog and publication status; no comparative outcome established |

A comparable completed incident case has not been established; a future comparison remains a backlog item, not a link to an invented case.

### 14b. What this case leaves open

| Question | Why open | Resolution / owner |
|---|---|---|
| Is order-level independence valid? | No domain contracts or source transitions are available | Product and application owners audit invariants and edge cases |
| What is real hot-partition service capacity? | No representative workload or deployed sink exists | Performance owner benchmarks with sink and replay overhead |
| Can upstream apply durable backpressure? | Source acceptance and buffer semantics are unspecified | Producer owner injects sustained overload and reconciles accepted IDs |
| Can replay and rollback preserve effects? | No implementation or failure-injection record exists | Application owner runs crash and fenced-cutover tests |

### 14c. Intense Interview Pack

Use the shared rubric for every question: 0 incorrect, 1 partial, 2 correct with assumptions, 3 correct with a defended tradeoff.
Prompts are candidate-facing; answers, alternatives, red flags and follow-ups are interviewer notes.

**Q01 — framing.** Prompt: What decision is being made, and for whom?
Answer: Restore timely dashboard updates for operations without losing or duplicating accepted effects or violating per-order ordering; the baseline is the account-key consumer.
Alternative: Defer any design change while using the authoritative source if correctness cannot be established.
Follow-up: What if the dashboard is advisory rather than operationally binding? Adjust the acceptable stale-window and fallback, not the correctness definition.
Red flag: Optimizing aggregate throughput without a user outcome. Support: Sections 1–3.

**Q02 — ordering.** Prompt: Why not always key by order?
Answer: The change parallelizes independent orders only if there is no account-wide invariant requiring a total order and the cutover preserves sequence.
Alternative: Keep account ordering and provision or throttle when the invariant is broader.
Follow-up: An account has a shared credit limit. Investigate serialization of that shared state before splitting.
Red flag: Assuming independent order IDs prove independent business effects. Support: Sections 4 and 8.

**Q03 — diagnosis.** Prompt: One partition's lag rises; what would you inspect first?
Answer: Compare partition ingress, service throughput, assignment history, sink latency and errors; a lag graph alone does not prove hot-key skew.
Alternative: Inspect broker health first if multiple unrelated groups show correlated failures.
Follow-up: All partitions grow and sink latency spikes. Prioritize the sink as a competing cause.
Red flag: Immediately resetting offsets. Support: Sections 6 and 12.

**Q04 — capacity.** Prompt: Arrival is 1,400/s, service 1,000/s, for 15 minutes. How much backlog accumulates from zero?
Answer: (1,400 − 1,000) × 900 = 360,000 events, assuming constant rates and no initial backlog.
Alternative: Integrate measured time-varying arrival minus departure rates if available.
Follow-up: If initial backlog is 20,000, the resulting estimate is 380,000 events.
Red flag: Multiplying total arrival by time without subtracting completed work. Support: CALC-01.

**Q05 — recovery.** Prompt: With 360,000 queued, arrival 600/s and service 1,000/s, estimate drain time.
Answer: Net drain is 400/s, so 360,000 / 400 = 900 seconds, or 15 minutes, under steady rates.
Alternative: Use a time series model if rates vary or service degrades with backlog.
Follow-up: Arrival rises to 800/s: drain becomes 1,800 seconds.
Red flag: Divide by 1,000/s and ignore continuing arrivals. Support: Section 6.

**Q06 — partition scaling.** Prompt: Would 20 consumers fix one hot partition in a 12-partition group?
Answer: No; only one active group member owns that partition at a time, although reassigning other partitions may free its owner to focus on it.
Alternative: More consumers help when several busy partitions are sharing consumer resources.
Follow-up: If the hot consumer also owns another busy partition, reassignment may improve its effective capacity; measure it.
Red flag: Treating consumer count as parallelism within a partition. Support: SRC-01 and Section 5c.

**Q07 — idempotency.** Prompt: The database commits but offset commit fails. What happens?
Answer: The record is replayed; an event-ID inbox written atomically with the view update suppresses the second database effect.
Alternative: An intrinsically idempotent state update can work if sequence handling is also correct.
Follow-up: A remote email was sent before the crash. Database deduplication cannot undo it; use an outbox and an idempotent receiver where available.
Red flag: Claiming broker transactions automatically cover arbitrary external systems. Support: Section 8b and SRC-02.

**Q08 — migration.** Prompt: How do you move from account to order partitioning safely?
Answer: Verify the business invariant, fence producers, drain to a documented watermark, reconcile state, then enable the new routing epoch.
Alternative: A validated dual-path sequencer with bounded buffering may be necessary when a full pause is impossible.
Follow-up: An old-key event arrives after cutover. Hold and reconcile it; do not apply without sequence checks.
Red flag: Assuming dual writes preserve order across topics. Support: Section 8d.

**Q09 — measurement.** Prompt: Define the primary service indicator and identify its denominator.
Answer: Eligible source-accepted events visible within five minutes divided by all eligible source-accepted events over 30 days; include quarantined events in the denominator.
Alternative: Measure per-order freshness if the product decision is explicitly order-based and the population is stable.
Follow-up: Why not exclude malformed records? Exclusion could reward silently dropping inputs unless eligibility was specified at acceptance.
Red flag: Using only consumed events as the denominator. Support: Section 9.

**Q10 — evaluation.** Prompt: A clean replay shows zero errors. Does it prove correctness?
Answer: No; finite test coverage and correlated failures limit inference. Report workload coverage and a suitable uncertainty analysis, then inject failure boundaries.
Alternative: Formal verification may support a narrower invariant, but still depends on implementation and environment assumptions.
Follow-up: One million nearly identical events have the same failure trigger. Independent-event confidence calculations can be overoptimistic.
Red flag: Treating no observed errors as zero true risk. Support: Section 9.

**Q11 — retention.** Prompt: Why retain inbox IDs and source events?
Answer: Source retention enables replay; inbox retention prevents replay from reapplying already committed effects, and both horizons must support recovery.
Alternative: Rebuild into an isolated sink with a complete deterministic source replay if identity retention is inadequate.
Follow-up: The oldest available inbox ID is newer than the required replay start. Stop and reconcile before proceeding.
Red flag: Assuming offset retention alone protects downstream effects. Support: Section 8d.

**Q12 — overload.** Prompt: Is throttling always safe?
Answer: No; it is safe only if upstream buffering, acceptance semantics and business consequences are understood and bounded.
Alternative: Provision the sink or apply admission control before accepting requests, depending on the workflow.
Follow-up: Upstream has no durable queue. Do not silently discard accepted events; escalate to a controlled business fallback.
Red flag: Declaring recovery solely because downstream lag falls. Support: Sections 3 and 12.

**Q13 — sizing.** Prompt: Estimate payload storage for 360,000 events averaging 2 KiB.
Answer: 720,000 KiB / 1,024 ≈ 703 MiB for one payload copy; three copies are about 2.06 GiB before overhead or compression.
Alternative: Use a measured event-size distribution and storage benchmark for capacity planning.
Follow-up: Payloads are heavy-tailed. Average size alone is inadequate for memory and burst sizing.
Red flag: Calling payload-only bytes the full provisioned storage requirement. Support: Section 8d.

**Q14 — transfer.** Prompt: Would device-keyed telemetry benefit from the same design?
Answer: Possibly, if devices are independently ordered and hot devices can be safely partitioned at the proposed grain; inspect cross-device control invariants and bursts.
Alternative: Retain device ordering and isolate a large device on dedicated capacity when its sequence cannot be split.
Follow-up: A fleet-wide safety limit is shared. Add a separately coordinated decision boundary rather than assuming devices are independent.
Red flag: Treating the illustrative recommendation as a universal rule. Support: Section 13.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Illustrative marketplace event processing |
| Problem type | Hot-partition diagnosis, containment and safe recovery |
| Method | Compare per-partition ingress and service, throttle safely, then validate a possible key migration against the account-key baseline |
| Primary result | Not measured; the 15-minute drain is a conditional illustrative estimate |
| Key lesson | One overloaded partition can dominate freshness; repartition only when invariants and migration permit it |
| Tags | Incident diagnosis and recovery; partition-aware capacity planning |
| Case ID / version / as-of | CASE-STREAM-001 / 1.0 / 2026-11-07 |
| Evidence level | Illustrative assumptions and estimated arithmetic; no observed outcome |
| Implementation stage | Proposed |
| Baseline / primary metric | Account-key processing / fraction of accepted eligible events visible within five minutes over 30 days |
| Key limitation | Ordering contract, real workload and sink semantics are unverified |
| Review trigger / owner | Change in ordering requirement, workload or capacity / streaming platform lead |
