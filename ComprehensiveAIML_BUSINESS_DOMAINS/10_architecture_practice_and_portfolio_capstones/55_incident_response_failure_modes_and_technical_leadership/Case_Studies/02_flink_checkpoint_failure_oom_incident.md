---
case_id: CASE-STREAM-002
title: Flink checkpoint failure and memory pressure
case_type: incident
domain: stateful streaming operations
evidence: illustrative
stage: proposed
editorial_status: draft-unreviewed
last_reviewed: null
source_index: INDEX.md
review_trigger: Flink version, state backend, connector or recovery contract changes
---
# Flink checkpoint failure and memory pressure

## 1. The One-Line Framing
Given a stateful event-processing job and ambiguous checkpoint failures, the incident commander must choose safe containment and recovery for downstream analysts while preserving replay correctness, compared with repeatedly restarting unchanged workers.
The operating decision is incident diagnosis, not model training.
The primary success condition is a restored, reconciled stream without unsafe external effects.
The primary pattern is Incident diagnosis and recovery; the secondary pattern is Partition-aware capacity planning.
**Evidence declaration:** This is an entirely hypothetical teaching scenario, including the timestamps, architecture, measurements and numbers.
The source directory supplies a topic title, not an accessible incident narrative.
No system described here, observed failure, completed intervention, or outcome is claimed to exist.
The key tradeoff is that larger resources can relieve memory pressure yet leave the actual bottleneck and replay risk intact.
The symptoms alone cannot identify whether a large state, a slow sink, or checkpoint storage caused the failure.

## 2. Context & Background
The illustrative product produces near-real-time account activity aggregates for operational dashboards.
Each event has an immutable event ID, account key, event timestamp and partition offset.
A Kafka-compatible log retains input for replay within a proposed retention window.
A Flink job validates events, groups them by account, maintains keyed state, and sends updates to a sink.
The hypothetical dashboard uses the latest committed aggregate; stale output can mislead operators.
The decision owner is the incident commander, assisted by the stream platform, storage and application owners.
The business owner decides whether degraded or stale results can remain visible.
The baseline response is to restart the failed job unchanged and watch it restart again.
That baseline is a scenario assumption, not an observed prior response.
No company, jurisdiction, customer population or vendor deployment is implied.
The incident time is a constructed relative clock, not a historical date.
A single regional job and one sink are assumed; replication and disaster recovery are outside scope.
This exercise assumes operational access to metrics, logs and a known checkpoint location.
An actual team must obtain authorization and redact customer identifiers before accessing evidence.
The architecture could instead have multiple sinks or an asynchronous enrichment dependency; then additional recovery boundaries apply.

## 3. The Business Question
The illustrative stakeholder question is: should the team restart, reduce load, change resources, or restore a prior checkpoint while investigating failure?
The incident commander decides under an immediate availability deadline.
The owner can pause an unsafe sink, rate-limit admission, preserve evidence, restore a compatible checkpoint, or defer recovery.
Doing nothing leaves output stale and can exhaust input retention.
An unsafe restore may instead duplicate externally visible actions.
The objective is to recover useful, correct updates as soon as practical.
The hard scenario constraints are no silent loss, no uncontrolled duplicate external effects, and no deletion of the only viable recovery point.
The scope is one stateful job and its source, checkpoint store and sink.
Changing the business aggregation, replacing the broker, and promising a regulatory conclusion are excluded.
Freshness is a proxy for usefulness; a rapidly updating but incorrect dashboard is not success.
The application owner approves the correctness contract before traffic resumes.

## 4. Assumptions & Constraints
| ID | Assumption or constraint | Status and test | Consequence / fallback | Owner and trigger |
|---|---|---|---|---|
| A1 | Input retained through recovery | Unverified; compare earliest retained offsets with restored offsets | Missing data requires source backfill or explicit loss decision | Source owner before restore |
| A2 | Checkpoint is complete and readable | Unverified; inspect metadata and rehearse restore | Select earlier compatible point if unavailable | Platform owner before deployment |
| A3 | Sink writes are idempotent or transactionally coordinated | Unverified; inspect connector and destination contract | Fence or reconcile external effects before replay | Sink owner before replay |
| A4 | Event IDs are stable across retries | Unverified; compare producer retry samples | Use a different durable business key | Producer owner before deduplication |
| A5 | Restore is compatible with job and serializer versions | Unverified; isolated restore using exact artifacts | Restore matching version or execute tested migration | Platform owner on version change |
| C1 | No silent data loss | Scenario constraint, not a verified external regulation | Stop and escalate if retained offsets are insufficient | Business owner at recovery gate |
| C2 | Controlled duplicates | Scenario constraint; destination semantics must be established | Pause effects and reconcile if uncertain | Sink owner at recovery gate |
| C3 | Protect incident evidence | Scenario constraint; save configuration and logs | Snapshot metadata before changing workers | Incident commander at triage |
The most decision-sensitive assumptions are A1, A2 and A3.
If any fails, ordinary checkpoint restore is not enough to guarantee correctness.
Memory availability, budget and permissible throttling remain locally negotiated; none is a universal limit.

## 5. Data, Tools & Method
### 5a. Data Specification
The unit of analysis is an event, identified by event ID and additionally located by source partition and offset.
The diagnostic unit is a checkpoint attempt, associated with one job execution and checkpoint ID.
Metrics must carry timestamps, task identity and execution-attempt identity.
Record checkpoint trigger, completion or failure, duration, state size and per-task statistics.
Collect arrival and output rates per partition, sink latency, backpressure, busy time, garbage collection and container memory.
Record both source watermark and processing time where available; neither alone establishes correctness.
Preserve checkpoint metadata and state-backend configuration, avoiding secret-bearing configuration exports.
Compare data over a pre-incident baseline, incident period and recovery period using consistent labels.
The case has no training labels or train/test split; replay correctness uses a known input and reference output.
Excluded or late events must be identified, not silently omitted from the denominator.
Event time, processing time, checkpoint time and log offset are distinct quantities.
The source offset is not interchangeable with an event count if messages can be compacted or filtered.
A restore oracle compares logical records and side effects, not just aggregate counts.
### 5b. Tools & Stack
The proposed system uses a Kafka-compatible event log, Flink stateful job, checkpoint object store and external sink.
The selected Flink version, connector versions, state backend, deployment platform and exact memory configuration are not specified.
Inspect the deployed version documentation before using specific metric names or configuration keys.
The deployment platform supplies container termination reason and resource accounting.
The object-store team supplies request latency, throttling and access-denial evidence.
The source owner supplies retention, offsets and rate; the sink owner supplies transaction and idempotency semantics.
No commands, binaries or workload runs were executed for this case.
### 5c. Method Summary
Trace events from source through keyed state and sink, and trace checkpoint barriers and snapshot completion separately.
Preserve evidence first, then rank competing hypotheses against time-aligned metrics.
| Option | Benefit | Boundary | Status |
|---|---|---|---|
| Restart unchanged | Minimal immediate change | Repeats deterministic failure and risks repeated side effects | Baseline; not tested |
| Relieve sink or limit admission | May reduce queue pressure | Requires safe backlog and retention | Proposed conditional action |
| Increase memory | Can add headroom | Does not fix unbounded state, storage or sink failure | Considered; not tested |
| Restore known checkpoint | Recovers logical state and source position | Requires compatibility, retention and sink guarantees | Proposed after verification |
| Reset state | Simple to describe | Loses state or corrupts derived outputs without full replay | Rejected absent explicit rebuild plan |
The invariant is that restored state and restored source positions represent the same completed checkpoint.
A replayed input may reach an external sink more than once unless its connector and destination jointly prevent duplicate effects.
### 5d. Data provenance, quality, exclusions and leakage controls
Every number in this case is constructed for teaching, not sampled from production.
No real checkpoint, log, job graph, input event or output table was available.
Proposed evidence must include collection time, timezone, query, metric labels and retention policy.
Do not conflate an absent metric with a zero value.
Apply least privilege to state snapshots because state may contain customer data.
Record sampling, missing intervals, aggregation and excluded task attempts before comparing traces.
Redact account identifiers while preserving stable correlations where approved.
Predictive-label leakage is not applicable; retrospective selection bias remains relevant to incident analysis.
### 5e. Reproducibility and method-selection ledger
| ID | Artifact | Scope | Availability and limitation |
|---|---|---|---|
| SRC-1 | [Flink stateful stream processing](https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/stateful-stream-processing/) | Checkpoints and state concepts | Public conceptual reference; stable documentation changes over time |
| SRC-2 | [Flink checkpointing operations](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/state/checkpoints/) | Operational checkpoint behavior | Public; verify exact deployed release and settings |
| SRC-3 | [Flink backpressure monitoring](https://nightlies.apache.org/flink/flink-docs-stable/docs/ops/monitoring/back_pressure/) | Backpressure interpretation | Public; metrics and UI vary by release |
| SRC-4 | [Flink checkpoint alignment](https://nightlies.apache.org/flink/flink-docs-stable/docs/concepts/stateful-stream-processing/#checkpointing) | Barrier and checkpoint concepts | Public; not evidence of this hypothetical incident |
| RUN | No run | None | No reproduction, logs, code revision, or measured results exist |
Sources are pointers for review, not an assertion that a particular version was tested.
The source-topic crosswalk maps the visible incident title to this module; the original incident contents were unavailable.
A real reproduction requires immutable code revision, image digest, Flink and connector versions, configuration, input trace, resource limits and sink contract.
Capture checkpoint ID, operator UIDs, serializer snapshots and source offsets in the restore record.

## 6. Analysis Walkthrough
The key conclusion is conditional: a failed checkpoint and an out-of-memory termination require separate causal tests before a mitigation is selected.
The timeline and input values below are deliberately hypothetical and form no empirical result.
Assume an input rate of 1,200 events/s and sustained processing capacity of 1,000 events/s for 600 seconds.
Assume no preexisting backlog and a constant rate during this window.
A fluid queue balance yields accumulated backlog as arrivals minus departures.
```text
FORMULA: B = max(0, (lambda - mu) * t)
WHERE: B is events; lambda and mu are events/s; t is seconds.
CONSTRAINT: rates are constant, no initial backlog, and service is available when work exists.
NUMERICAL EXAMPLE: B = (1200 - 1000) * 600 = 120000 events.
INTERPRETATION: buffering absorbs the burst but cannot stabilize sustained overload.
```
Independent arithmetic: 720,000 arriving events minus 600,000 processed events equals 120,000 queued events.
With 1,024 payload bytes/event, payload-only backlog is 122,880,000 bytes, or 117.1875 MiB.
That is not a heap estimate: metadata, serialization, replication and queue placement are excluded.
Suppose restored throughput is 1,000 events/s while arrivals are throttled to 800 events/s.
```text
FORMULA: T = B / (mu_recovery - lambda_recovery)
WHERE: T is seconds; B is events; rates are events/s.
CONSTRAINT: recovery capacity exceeds arrivals and both rates remain constant.
NUMERICAL EXAMPLE: T = 120000 / (1000 - 800) = 600 s = 10 min.
INTERPRETATION: retention must cover the outage, restart, drain and additional safety margin.
```
The independent check is 200 events/s surplus times 600 s equals 120,000 events.
If arrival rises to 900 events/s, the surplus is 100 events/s and drain takes 1,200 seconds.
If arrival is 1,000 events/s or more, there is no finite drain time in this model.
A quiet average can hide an overloaded partition, uneven state distribution, or a slower downstream sink.
For sizing, suppose a state snapshot requires 60 GiB written at an effective 100 MiB/s without overlap.
The idealized transfer lower bound is 60 * 1024 / 100 = 614.4 s.
That estimate excludes serialization, storage latency, contention and checkpoint coordination.
Incremental snapshots may transfer a different amount; total state size is not automatically checkpoint bytes.
If the configured timeout is 600 s, this idealized 614.4 s alone exceeds it under those assumptions.
A faster connection, incremental transfer, parallel upload or changed state size would change the conclusion.
The simplified bound must not be treated as an actual checkpoint duration.
No alternative was tried; each option is an unexecuted hypothesis.

## 7. The Finding
Recommend preserving evidence and isolating the limiting component before choosing a conditional containment path.
Neither a cause nor an improvement has been measured.
The arithmetic shows that finite recovery needs service capacity above arrivals and sufficient retained input; it does not establish the actual bottleneck.
Verify checkpoint compatibility, input retention and sink effects before restoring or resuming traffic.
A storage failure, heap leak or sink bottleneck would require different corrective action.

## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
There is no model-training path in this case.
A proposed offline rehearsal copies an approved, redacted trace and compatible checkpoint to an isolated environment.
A deterministic reference computes expected logical aggregate results for the selected trace.
Record the versioned job image, configuration, state schema, source positions and expected outputs.
Inject failure at input, snapshot and sink boundaries and reconcile the output before authorizing production recovery.
No rehearsal has been executed.
### 8b. Online / Serving / Real-Time Path
```text
Producer -> durable partitioned log -> source reader -> validate/key -> keyed state
                                                                    |        |
                                                                    |        v
                                                                    |   checkpoint store
                                                                    v
                                                    sink -> dashboard / audit ledger
                                     failures -> incident commander -> isolate / restore
```
On success, a validated input updates keyed state and is emitted according to the application's output semantics.
Periodic checkpoints coordinate a recoverable state and input position; completion must be verified, not inferred from initiation.
On failure, pause unsafe effects, retain the last completed checkpoint, restart a compatible job and replay retained input.
Reconcile the output by event or business key and examine gaps, duplicates and ordering before lifting the pause.
A checkpoint alone does not make an arbitrary external API exactly once.
Backpressure can propagate upstream from a slow sink, delaying barrier progress and extending checkpoint duration.
An unaligned checkpoint may reduce alignment delay under pressure but can capture in-flight data and increase snapshot I/O; verify version and configuration.
Source authentication, state and checkpoint store permissions, sink credentials and audit access have distinct owners.
### 8c. Key Design Decisions
| Decision | Proposed choice | Alternative / boundary | Evidence status |
|---|---|---|---|
| Restore point | Latest completed compatible checkpoint with available inputs | Earlier point if validation fails; more replay | Not tested |
| Admission | Reduce only where upstream buffering and customer contract permit | Spill to durable log; reject only under explicit policy | Not tested |
| Output safety | Transactional connector or idempotent writes with reconciliation | Manually fenced side effects if no suitable connector | Contract unverified |
| Memory | Inspect heap, native and container use before adjusting | Scale or tune after identifying pressure | No allocation measurements |
| Checkpoint mode | Keep configured mode until bottleneck understood | Evaluate unaligned only with state and I/O budget | Not tested |
### 8d. Architecture Artifacts and Operational Readiness
The application owner controls output correctness and visible degraded-mode communication.
The platform owner controls restore artifacts, capacity, checkpoint configuration and deployment.
The source owner verifies partition offsets and retention before restore.
The sink owner verifies deduplication, transaction expiry and any external effects.
The security owner approves access to state, logs and any replay copy.
Use versioned operator identities and verify state serializer compatibility before changing job graph or parallelism.
Restoring from an older checkpoint increases the amount of replay and possibly the reconciliation window.
Output fencing must prevent old and new attempts from writing concurrently if the destination requires a single writer.
Avoid inferring exactly-once end-to-end from the presence of checkpoints alone.
Monitor backlog by partition, checkpoint age, completed checkpoint count, memory, sink latency and end-to-end freshness.
Proposed alerts must name owner, evaluation window, threshold and response; thresholds below are exercise choices, not vendor guidance.
If the newest completed checkpoint approaches input retention, escalate before recoverability is lost.
A canary must use isolated output or an explicitly safe shadow path; an external side effect cannot be rolled back by deleting internal state.
Capture storage and network costs, scaling constraints and operational effort before choosing extra capacity.

## 9. Evaluation & Validation
Primary metric: fraction of eligible input events whose intended logical effect is present exactly once in the reconciled output.
Numerator: eligible event IDs with one correct, committed logical effect and no conflicting effect.
Denominator: all eligible event IDs in the selected replay window, including late and failed records.
Population: one declared set of partitions and offsets; window: checkpoint position through the selected recovery cutoff.
A proposed pass requires 100% on a finite, known replay set, zero unexplained missing IDs and zero duplicate external effects.
The application owner must approve this threshold before running the exercise.
Record freshness as the elapsed time between event timestamp and committed output, with clock and late-event caveats.
Guardrails include checkpoint completion, stable memory, input retention margin, sink errors and cost.
Compare unchanged and proposed configurations on the same trace, source positions and output oracle.
Run both normal-load and overload scenarios; inject store latency, task kill, sink timeout and worker memory pressure separately.
Inspect per-partition results rather than relying only on aggregate counts.
A finite test cannot prove absence of all rare failures; repeated trials and realistic long-running workloads broaden coverage.
The proposed test has not been executed; there is no baseline, treatment result, confidence interval or production artifact.
A changed workload or mismatched sink implementation would invalidate a simple causal comparison.

## 10. Action Taken
No operational action was taken; this is a proposed teaching exercise with no real incident.
The incident commander would authorize a containment plan after the source and sink owners check safety.
The platform team would preserve logs and configuration, establish a recovery point and run the approved restore procedure.
The sink owner would confirm transaction status and reconciliation before effects resume.
Rollback would restore the last compatible binary and checkpoint, subject to serializer and sink compatibility.
If compatibility or retention is not established, defer restore and escalate to the business owner.
No canary, production rollout, resource change or configuration change occurred.

## 11. Outcome & Results
**Outcome not measured.** Figures elsewhere are constructed teaching inputs, not observations or evidence of realized impact.
| Metric | Baseline | After | Difference | Evidence and limitation |
|---|---|---|---|---|
| Correct logical effects | Unavailable | Not measured | Undefined | No replay or output record |
| Checkpoint completion | Unavailable | Not measured | Undefined | No actual attempts |
| Input freshness | Unavailable | Not measured | Undefined | No source or sink timestamps |
| Memory and cost | Unavailable | Not measured | Undefined | No deployment |
A future owner would collect these measures over the same declared replay window before deciding whether the proposal works.
No numerical improvement, incident duration or reliability gain is claimed.

## 12. Failure Modes & What Went Wrong
All scenarios below are **HYPOTHETICAL**; none is confirmed, reproduced or observed.
The timestamped sequence below is an instructional simulation, not a reconstructed incident.
| Relative time | Hypothetical event | Interpretation or proposed response |
|---|---|---|
| 00:00 | Sink response time begins increasing | Potential downstream bottleneck; not proof |
| 00:05 | Backpressure spreads toward sources | Check per-task busy and backpressured time |
| 00:08 | A checkpoint times out | Inspect per-subtask phases and store errors |
| 00:12 | A worker exceeds its container limit | Distinguish heap, native and cgroup causes |
| 00:15 | Alert reaches incident commander | Preserve artifacts before restarts |
| 00:20 | Team considers admission control | Verify upstream durability and retention first |
| 00:30 | Team identifies a candidate checkpoint | Check completeness, compatibility and source offsets |
| 00:40 | Restore would start | Fence unsafe effects and replay in an approved environment |
| 00:55 | Reconciliation would begin | Compare input IDs and output effects |
| 01:10 | Recovery gate would be assessed | Resume only if correctness and health criteria pass |
| Later | Follow-up would assign prevention tests | No prevention action is claimed complete |
Detection delay means alert time minus onset; in the constructed timeline it is 15 minutes.
Recovery time means authorized recovery gate minus onset; the hypothetical interval is 70 minutes, not a measured service outcome.
The timeline specifies a possible sequence, not proof that backpressure caused the OOM.
The causal graph to test is sink slowdown -> queue pressure -> barrier delay and memory pressure; each arrow needs independent evidence.
Alternate branches include state growth -> memory pressure, store throttling -> timeout, and version incompatibility -> restore failure.
An alert can fail because its thresholds are poorly chosen, labels are missing or it monitors only successful checkpoints; no particular alert defect is asserted.
The following distinct diagnostic cards are hypotheses to discriminate, not completed postmortem findings.

### F-01 — Sink saturation
- Provenance: hypothetical; not an observed incident.
- Initial signal: High sink latency and busy sink.
- First discriminating check: Sink request histograms and destination capacity.
- Candidate mechanism: Queueing or retries inside sink.
- Immediate containment: Throttle only after confirming source retention.
- Regression experiment: Capacity test with controlled destination delay.
- Accountable owner: Destination team.
- Counterevidence: Healthy destination with similar latency disproves a simple saturation story.

### F-02 — Checkpoint store throttling
- Provenance: hypothetical; not an observed incident.
- Initial signal: Upload errors or slow snapshot phase.
- First discriminating check: Object-store request status and per-task upload duration.
- Candidate mechanism: Store throttling or bandwidth contention.
- Immediate containment: Reduce competing I/O without deleting checkpoints.
- Regression experiment: Inject rate limits into isolated object store.
- Accountable owner: Storage team.
- Counterevidence: Slow alignment with fast uploads points upstream.

### F-03 — Checkpoint permission loss
- Provenance: hypothetical; not an observed incident.
- Initial signal: Checkpoint writes fail with authorization errors.
- First discriminating check: Audit principal, permission and expiry timestamps.
- Candidate mechanism: Revoked identity or changed bucket policy.
- Immediate containment: Restore authorized access under change control.
- Regression experiment: Revoke permission in a nonproduction environment.
- Accountable owner: Security team.
- Counterevidence: Successful writes by the same identity weaken this hypothesis.

### F-04 — Checkpoint metadata corruption
- Provenance: hypothetical; not an observed incident.
- Initial signal: Completed-looking checkpoint fails metadata read.
- First discriminating check: Read checkpoint metadata and validate all referenced files.
- Candidate mechanism: Missing or corrupt metadata or state object.
- Immediate containment: Protect checkpoint and choose earlier verified point.
- Regression experiment: Remove a referenced test object and exercise restore.
- Accountable owner: Platform team.
- Counterevidence: Matching checksums and successful restore reject this branch.

### F-05 — Long barrier alignment
- Provenance: hypothetical; not an observed incident.
- Initial signal: Alignment phase dominates checkpoint duration.
- First discriminating check: Compare per-input barrier arrival times and backpressure.
- Candidate mechanism: One input channel is delayed.
- Immediate containment: Relieve bottleneck; consider mode only after testing.
- Regression experiment: Stall one channel in replay.
- Accountable owner: Streaming team.
- Counterevidence: Fast alignment with slow upload needs a different diagnosis.

### F-06 — Slow asynchronous snapshot
- Provenance: hypothetical; not an observed incident.
- Initial signal: Alignment is quick but snapshot phase is long.
- First discriminating check: Inspect per-task sync and async durations separately.
- Candidate mechanism: State upload or backend snapshot bottleneck.
- Immediate containment: Reduce concurrent store contention.
- Regression experiment: Constrain storage throughput in a controlled run.
- Accountable owner: Platform team.
- Counterevidence: A short async phase contradicts this specific explanation.

### F-07 — State growth
- Provenance: hypothetical; not an observed incident.
- Initial signal: Checkpoint size and keyed state grow over time.
- First discriminating check: State size and key cardinality by operator and subtask.
- Candidate mechanism: Retention or key cardinality grows beyond capacity.
- Immediate containment: Stop unsafe growth if business policy permits.
- Regression experiment: Run long-lived keys and measure state growth.
- Accountable owner: Application team.
- Counterevidence: Stable state size suggests another memory component.

### F-08 — Hot key
- Provenance: hypothetical; not an observed incident.
- Initial signal: One subtask holds disproportionate state.
- First discriminating check: Compare key distribution and state size per subtask.
- Candidate mechanism: Skew concentrates work and memory.
- Immediate containment: Route or split only if aggregate semantics permit.
- Regression experiment: Replay skewed keys and verify output equivalence.
- Accountable owner: Application team.
- Counterevidence: Uniform state and CPU weaken the hot-key hypothesis.

### F-09 — Missing state expiration
- Provenance: hypothetical; not an observed incident.
- Initial signal: Old inactive keys persist beyond intended retention.
- First discriminating check: Inspect state TTL configuration and logical expiry contract.
- Candidate mechanism: Expiration absent or semantics misunderstood.
- Immediate containment: Apply tested cleanup only with business approval.
- Regression experiment: Advance time and inspect state and output correctness.
- Accountable owner: Application team.
- Counterevidence: A required indefinite history makes expiration inappropriate.

### F-10 — Heap exhaustion
- Provenance: hypothetical; not an observed incident.
- Initial signal: JVM reports Java heap space exhaustion.
- First discriminating check: Heap usage, GC logs and heap dump where safe.
- Candidate mechanism: Heap-resident objects exceed available heap.
- Immediate containment: Reduce load or add bounded heap after analysis.
- Regression experiment: Replay allocation profile with representative state.
- Accountable owner: Runtime team.
- Counterevidence: Container OOM without heap growth points elsewhere.

### F-11 — Native memory exhaustion
- Provenance: hypothetical; not an observed incident.
- Initial signal: Container dies while JVM heap remains below limit.
- First discriminating check: Compare process RSS, native memory and managed state.
- Candidate mechanism: Native allocations or backend cache exceed container budget.
- Immediate containment: Adjust managed and container allocation after diagnosis.
- Regression experiment: Stress native backend and observe RSS.
- Accountable owner: Runtime team.
- Counterevidence: RSS below enforced limit disproves container-memory exhaustion.

### F-12 — Direct buffer pressure
- Provenance: hypothetical; not an observed incident.
- Initial signal: Network-heavy tasks use growing off-heap memory.
- First discriminating check: Inspect direct buffer metrics and network configuration.
- Candidate mechanism: Direct buffers consume memory outside the Java heap.
- Immediate containment: Reduce buffer pressure within supported configuration.
- Regression experiment: Vary network buffering in a representative replay.
- Accountable owner: Runtime team.
- Counterevidence: Flat direct-memory use rules out this contributor.

### F-13 — GC pause
- Provenance: hypothetical; not an observed incident.
- Initial signal: Worker heartbeat or checkpoint times out during long pauses.
- First discriminating check: Correlate GC pause timestamps with missing acknowledgements.
- Candidate mechanism: Long garbage collections stall task progress.
- Immediate containment: Reduce allocation pressure or provision headroom.
- Regression experiment: Run allocation-heavy workload and record pauses.
- Accountable owner: Runtime team.
- Counterevidence: No correlated pauses require another cause.

### F-14 — Container eviction
- Provenance: hypothetical; not an observed incident.
- Initial signal: Task disappears without a JVM OOM exception.
- First discriminating check: Read orchestrator termination reason and node pressure.
- Candidate mechanism: Node eviction rather than application heap exhaustion.
- Immediate containment: Move or reserve capacity under platform policy.
- Regression experiment: Apply controlled node pressure in a test cluster.
- Accountable owner: Infrastructure team.
- Counterevidence: Explicit JVM heap error supports a different failure path.

### F-15 — Network bottleneck
- Provenance: hypothetical; not an observed incident.
- Initial signal: High network utilization and delayed transfers.
- First discriminating check: Measure task and checkpoint-store network throughput.
- Candidate mechanism: Shared network saturation.
- Immediate containment: Relieve competing transfers or increase tested capacity.
- Regression experiment: Limit network in a test and compare checkpoint phases.
- Accountable owner: Network team.
- Counterevidence: Low utilization does not prove health if packet loss is high.

### F-16 — Disk saturation
- Provenance: hypothetical; not an observed incident.
- Initial signal: Local state access and snapshot preparation slow.
- First discriminating check: Inspect disk latency, throughput and available space.
- Candidate mechanism: Backend compaction or disk bottleneck.
- Immediate containment: Protect disk space and reduce competing local I/O.
- Regression experiment: Stress local disk while retaining an output oracle.
- Accountable owner: Storage team.
- Counterevidence: Low disk latency during failure weakens this hypothesis.

### F-17 — Compaction backlog
- Provenance: hypothetical; not an observed incident.
- Initial signal: State backend local disk and I/O rise.
- First discriminating check: Inspect backend-specific compaction and write metrics.
- Candidate mechanism: Compaction cannot keep pace with state updates.
- Immediate containment: Reduce write load or provision I/O after validation.
- Regression experiment: Generate sustained state churn with backend metrics.
- Accountable owner: State platform team.
- Counterevidence: Backend without compaction requires another explanation.

### F-18 — Oversized records
- Provenance: hypothetical; not an observed incident.
- Initial signal: Occasional inputs spike memory and serialization time.
- First discriminating check: Inspect bounded record-size distribution and error samples.
- Candidate mechanism: Exceptional payload expands in memory.
- Immediate containment: Quarantine according to explicit data contract.
- Regression experiment: Test maximum accepted payload and rejection path.
- Accountable owner: Producer team.
- Counterevidence: Normal payload sizes throughout the failure weaken this branch.

### F-19 — Serialization amplification
- Provenance: hypothetical; not an observed incident.
- Initial signal: Memory rises despite stable input byte volume.
- First discriminating check: Compare serialized bytes with live object allocation.
- Candidate mechanism: Deserialized object graph and copies consume more space.
- Immediate containment: Avoid avoidable copies after profiling.
- Regression experiment: Profile representative records and serializer versions.
- Accountable owner: Application team.
- Counterevidence: Low allocation rate suggests a different memory source.

### F-20 — Unbounded retry buffer
- Provenance: hypothetical; not an observed incident.
- Initial signal: Memory grows with failed sink requests.
- First discriminating check: Track pending retries, queue depth and sink errors.
- Candidate mechanism: Retries accumulate faster than they complete.
- Immediate containment: Pause admission and bound retries without losing records.
- Regression experiment: Inject persistent sink errors and verify bounded queues.
- Accountable owner: Sink team.
- Counterevidence: A bounded stable retry queue rejects this mechanism.

### F-21 — Excessive concurrency
- Provenance: hypothetical; not an observed incident.
- Initial signal: More in-flight requests correlate with memory spikes.
- First discriminating check: Count pending requests per task and response times.
- Candidate mechanism: Concurrency limit is too high for available memory.
- Immediate containment: Lower concurrency with durable upstream buffering.
- Regression experiment: Sweep concurrency and verify throughput and correctness.
- Accountable owner: Application team.
- Counterevidence: Flat in-flight count argues against this explanation.

### F-22 — Slow external enrichment
- Provenance: hypothetical; not an observed incident.
- Initial signal: Async dependency stalls input processing.
- First discriminating check: Examine dependency latency and outstanding calls.
- Candidate mechanism: Slow enrichment fills in-flight request capacity.
- Immediate containment: Apply approved fallback or durable backpressure.
- Regression experiment: Inject latency and verify timeout and replay behavior.
- Accountable owner: Dependency team.
- Counterevidence: Healthy dependency latency during stalls weakens this branch.

### F-23 — Checkpoint overlap
- Provenance: hypothetical; not an observed incident.
- Initial signal: Multiple snapshots compete for shared resources.
- First discriminating check: Check configured concurrency and checkpoint attempt overlap.
- Candidate mechanism: Concurrent checkpoints increase resource pressure.
- Immediate containment: Reduce overlap if compatible with recovery objectives.
- Regression experiment: Compare one and multiple concurrent checkpoints.
- Accountable owner: Platform team.
- Counterevidence: No overlap rules out concurrent-snapshot contention.

### F-24 — Checkpoint interval mismatch
- Provenance: hypothetical; not an observed incident.
- Initial signal: New checkpoints start before previous ones can finish.
- First discriminating check: Inspect trigger schedule and duration distribution.
- Candidate mechanism: Interval too short for actual sustained checkpoint work.
- Immediate containment: Adjust schedule only after validating recovery objectives.
- Regression experiment: Replay with measured duration and alternative intervals.
- Accountable owner: Platform team.
- Counterevidence: Long intervals with one active checkpoint indicate another issue.

### F-25 — Checkpoint timeout too short
- Provenance: hypothetical; not an observed incident.
- Initial signal: Otherwise healthy snapshots approach a fixed timeout.
- First discriminating check: Compare checkpoint phase durations and timeout.
- Candidate mechanism: Timeout lower than legitimate completion time.
- Immediate containment: Increase timeout only with enough recovery margin.
- Regression experiment: Replay with controlled slow storage and recovery assertions.
- Accountable owner: Platform team.
- Counterevidence: A worker death before timeout is not solved by timeout alone.

### F-26 — Task restart loop
- Provenance: hypothetical; not an observed incident.
- Initial signal: Repeated failure occurs shortly after each restore.
- First discriminating check: Compare exception signatures and restore progress.
- Candidate mechanism: Persistent fault or poison input is replayed.
- Immediate containment: Stop restart storm and preserve evidence.
- Regression experiment: Inject deterministic error and exercise bounded restart policy.
- Accountable owner: On-call team.
- Counterevidence: Distinct failures at random times require broader investigation.

### F-27 — Poison event
- Provenance: hypothetical; not an observed incident.
- Initial signal: Same source offset precedes repeated task failure.
- First discriminating check: Correlate exception with event ID and offset.
- Candidate mechanism: Malformed or unsupported payload.
- Immediate containment: Quarantine only through an approved auditable policy.
- Regression experiment: Replay exact redacted payload and inspect contract.
- Accountable owner: Producer team.
- Counterevidence: Different failing offsets weaken a deterministic poison-event explanation.

### F-28 — Schema incompatibility
- Provenance: hypothetical; not an observed incident.
- Initial signal: Restore fails after a job or serializer change.
- First discriminating check: Compare operator IDs and serializer compatibility artifacts.
- Candidate mechanism: New job cannot interpret saved state.
- Immediate containment: Restore compatible binary or use tested migration.
- Regression experiment: Exercise old checkpoint against new binary in isolation.
- Accountable owner: Release team.
- Counterevidence: A successful compatibility restore directs attention elsewhere.

### F-29 — Missing operator identity
- Provenance: hypothetical; not an observed incident.
- Initial signal: State cannot map cleanly after graph changes.
- First discriminating check: Compare stable operator UIDs and job graph versions.
- Candidate mechanism: Changed identity breaks mapping to prior state.
- Immediate containment: Roll back graph change or run approved migration.
- Regression experiment: Restore across a graph change using fixed operator identities.
- Accountable owner: Release team.
- Counterevidence: Identical mappings do not establish serializer compatibility.

### F-30 — Source retention gap
- Provenance: hypothetical; not an observed incident.
- Initial signal: Restored source offsets precede earliest retained offsets.
- First discriminating check: Compare checkpoint offsets with broker retention boundaries.
- Candidate mechanism: Input needed for replay has expired.
- Immediate containment: Pause recovery and seek authoritative backfill.
- Regression experiment: Test restore after deliberately expiring test records.
- Accountable owner: Source team.
- Counterevidence: Available offsets across all partitions remove this specific blocker.

### F-31 — Sink duplicate effects
- Provenance: hypothetical; not an observed incident.
- Initial signal: Replay repeats a previously committed effect.
- First discriminating check: Compare idempotency keys and sink transaction records.
- Candidate mechanism: External sink is not coordinated with checkpoint.
- Immediate containment: Fence writers and reconcile before replay.
- Regression experiment: Kill job after sink commit but before checkpoint completes.
- Accountable owner: Sink team.
- Counterevidence: An auditable atomic transaction may prevent this specific duplicate.

### F-32 — Sink transaction expiry
- Provenance: hypothetical; not an observed incident.
- Initial signal: Recovery cannot commit an in-flight transaction.
- First discriminating check: Inspect connector transaction timeouts and recovery duration.
- Candidate mechanism: Transaction expired before recovery completed.
- Immediate containment: Use supported recovery or reconcile affected writes.
- Regression experiment: Delay restart beyond transaction timeout in isolation.
- Accountable owner: Sink team.
- Counterevidence: A nontransactional sink requires a different correctness mechanism.

### F-33 — Misleading aggregate lag
- Provenance: hypothetical; not an observed incident.
- Initial signal: Total lag appears acceptable while one partition grows.
- First discriminating check: Compare per-partition offsets and rates.
- Candidate mechanism: Skew masks a local bottleneck.
- Immediate containment: Identify overloaded partition and investigate its consumers.
- Regression experiment: Replay uneven partition workload.
- Accountable owner: Source team.
- Counterevidence: Even per-partition rates weaken the skew explanation.

### F-34 — Watermark stall
- Provenance: hypothetical; not an observed incident.
- Initial signal: Event-time output stops while processing continues.
- First discriminating check: Inspect watermarks and idle source partitions.
- Candidate mechanism: Late or idle input prevents event-time advancement.
- Immediate containment: Apply tested idleness handling if semantics allow.
- Regression experiment: Pause one input and verify window outputs and late events.
- Accountable owner: Application team.
- Counterevidence: Healthy watermark progression suggests sink or computation issues.

### F-35 — Late-event surge
- Provenance: hypothetical; not an observed incident.
- Initial signal: State and timer count grow during delayed input.
- First discriminating check: Measure event-time lateness and timer cardinality.
- Candidate mechanism: Late arrivals extend state lifecycle or firing work.
- Immediate containment: Control source delay or revise approved lateness policy.
- Regression experiment: Replay realistic lateness distribution.
- Accountable owner: Application team.
- Counterevidence: A processing-time-only job may not exhibit this mechanism.

### F-36 — Checkpoint storage quota
- Provenance: hypothetical; not an observed incident.
- Initial signal: Snapshots fail as storage approaches quota.
- First discriminating check: Inspect allocated bytes and quota errors.
- Candidate mechanism: Checkpoint store is full or quota exhausted.
- Immediate containment: Secure capacity while preserving recovery points.
- Regression experiment: Fill isolated test quota and verify error escalation.
- Accountable owner: Storage team.
- Counterevidence: Abundant quota and successful writes weaken this branch.

### F-37 — Clock mismatch
- Provenance: hypothetical; not an observed incident.
- Initial signal: Cross-system logs appear causally reversed.
- First discriminating check: Compare time synchronization and monotonic durations.
- Candidate mechanism: Hosts have inconsistent clocks or timezone conversion.
- Immediate containment: Correct clocking and retain raw timestamps.
- Regression experiment: Inject clock skew into a test timeline.
- Accountable owner: Infrastructure team.
- Counterevidence: Consistent monotonic timestamps help rule out clock skew.

### F-38 — Metrics aggregation error
- Provenance: hypothetical; not an observed incident.
- Initial signal: Dashboard suggests all workers are healthy.
- First discriminating check: Inspect labels, missing series and task attempt churn.
- Candidate mechanism: Aggregation hides failed or absent subtasks.
- Immediate containment: Use per-attempt views and alert on missing series.
- Regression experiment: Remove one metric source and validate alerts.
- Accountable owner: Observability team.
- Counterevidence: Complete raw series with correct labels reject this explanation.

### F-39 — Concurrent writers
- Provenance: hypothetical; not an observed incident.
- Initial signal: Old and restored executions both write output.
- First discriminating check: Inspect writer identity, leases and destination commits.
- Candidate mechanism: Fencing absent or incorrectly scoped.
- Immediate containment: Fence old attempts before resuming writes.
- Regression experiment: Run overlapping attempts in an isolated sink.
- Accountable owner: Sink team.
- Counterevidence: Strong destination fencing prevents this particular overlap.

### F-40 — Misleading reset
- Provenance: hypothetical; not an observed incident.
- Initial signal: Empty state makes job appear healthy.
- First discriminating check: Compare restored state and output to known prior checkpoint.
- Candidate mechanism: State reset silently loses accumulated history.
- Immediate containment: Stop publishing and restore verified state or rebuild.
- Regression experiment: Exercise full rebuild against authoritative input.
- Accountable owner: Application team.
- Counterevidence: Explicitly stateless jobs have a different recovery contract.
