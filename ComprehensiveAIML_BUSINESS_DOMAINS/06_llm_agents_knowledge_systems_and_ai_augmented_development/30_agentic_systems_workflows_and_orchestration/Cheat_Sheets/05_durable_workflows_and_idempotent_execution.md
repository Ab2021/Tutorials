---
id: agent-workflow-05
title: Durable workflows and idempotent execution
level: 3
status: draft
version: "1.0"
created: "2026-11-07"
reviewed: null
evidence: "DERIVED and ILLUSTRATIVE; source contents unavailable"
---
# 1. File Header
> **Purpose:** Design and diagnose recovery of queued workflows without silently duplicating business effects.
> **Prerequisites:** Transaction boundaries, queues, basic rate arithmetic, and [sessions and lifecycle](04_agent_sessions_state_isolation_and_lifecycle.md).
> **Outcome:** Given a crash window and downstream contract, identify a safe next action and estimate a bounded backlog.
> **Exclusions:** Vendor-specific replay APIs, a complete payment implementation, and exhaustive workflow-engine selection.
> **Evidence:** Worked systems and values are illustrative, not observed benchmarks.
> **Review:** Draft; no substantive independent review; no implementation or operational verification claimed.
> **Retrieval cue:** For a timeout or duplicate effect start at Sections 6 and 8.
The requested source is catalogued as `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/02_durable_execution.md` in `Train/Templates/directory.txt`.
The course [source coverage record](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-runtimes) maps the topic here.
The source file itself was not present in the accessible workspace; its actual contents and any vendor-specific claims are not verified.
This independent teaching treatment does not claim source-faithful coverage beyond the visible source title.
The reader must validate actual broker, database, runtime, and provider semantics before deployment.
**Takeaway:** Durable orchestration is a recovery decision, not proof that an external effect happened once.
# 2. Concept Map / Mental Model
```text
request -> stable business operation identity -> durable workflow state
                 |                                  |
                 v                                  v
          payload binding                  checkpoint / replay
                 |                                  |
                 v                                  v
        outbox or activity intent -------> delivery attempt
                                                    |
                                                    v
                                      provider idempotency boundary
                                                    |
                                    +---------------+-------------+
                                    |                             |
                             confirmed result                 unknown result
                                    |                             |
                                    v                             v
                            persist completion          query/reconcile, then retry
                                    |                             |
                                    +--------------+--------------+
                                                   v
                                         business ledger / audit
```
Arrows indicate information or control flow, not a distributed atomic transaction.
The business operation identity stays stable across attempts and workflow restarts.
A checkpoint captures workflow progress, not the state of an independent remote system.
An outbox makes local intent atomic with a local transaction, not remote delivery.
A provider's deduplication contract must cover the actual side effect and its replay window.
If remote outcome is uncertain, uncertainty remains until authoritative evidence resolves it.
A safe fallback can be manual review rather than a speculative second call.
**Takeaway:** Align workflow history, business identity, and the downstream effect boundary.
# 3. Foundational Definitions
## 3.1 Workflow and run
A workflow is a logical multi-step business process with persisted progress.
A run is an execution instance identified separately from the underlying business operation.
A run may be retried, resumed, replaced, or migrated while the business intent remains the same.
A run ID alone is therefore a poor payment idempotency key if reruns receive new IDs.
The logical workflow can span process failures, deployments, and long waits.
A durable runtime offers recovery only within its own documented persistence and retention contract.
The workflow output is not automatically proof of the provider's ledger state.
## 3.2 Checkpoint and replay
A checkpoint is a durable record of state or progress from which work can resume.
A replay reconstructs or restores execution using persisted history or state, according to the engine.
Replay does not imply that all engines deterministically re-execute code in the same manner.
Nondeterministic reads, clock values, randomness, and remote calls must follow engine-specific rules.
Checkpoint storage failure leaves progress uncommitted even if in-memory code has advanced.
An old checkpoint can cause already-attempted activities to be invoked again.
A checkpoint's recovery point objective depends on where and when it is committed.
## 3.3 Activity, attempt, and effect
An activity is a separately executed unit of workflow work.
An attempt is one invocation of that unit, including retries after an ambiguous outcome.
A business effect is a consequential external state change, such as a charge or shipment.
The same logical activity can have multiple attempts but must represent one authorized intent.
An HTTP success response and a settled financial effect are distinct provider states.
An attempt count is not a count of confirmed business effects.
A transport acknowledgment is not necessarily a business commitment.
## 3.4 Idempotency
Idempotency means repeated application of the same intended operation has the same intended effect as one application.
A key is a stable identifier for a logical operation in the namespace required by the receiver.
The receiver must atomically associate the key, payload identity, state, and effect or a recoverable protocol.
For a repeated key and equivalent payload, return the recorded outcome or an explicit pending state.
For the same key and different payload, reject the conflict rather than silently reuse a result.
Different keys do not guarantee different business operations: callers can accidentally mint duplicates.
A local cache or header alone is not an idempotency guarantee at the business-side effect boundary.
## 3.5 Timeout, retry, and ambiguity
A timeout is a local decision to stop waiting after a configured interval.
It does not prove the remote operation was canceled or failed.
A retry is a new attempt, potentially concurrent with the original attempt.
Backoff spaces retry attempts; jitter spreads synchronized clients.
An end-to-end deadline is the overall time budget, not a per-attempt timeout.
An ambiguous outcome is one for which the caller cannot establish whether the effect committed.
Reconciliation compares durable local intent with authoritative external evidence.
## 3.6 Queue and capacity notation
`B` is pending queue backlog in jobs, a nonnegative count.
`lambda` is the rate of new logical jobs in jobs per second.
`mu` is the successful completion capacity in jobs per second under the stated model.
`t` is elapsed time in seconds; `T` is backlog drain time in seconds.
`a` is attempts per logical job, dimensionless, when explicitly modeled.
Attempt capacity and logical-job completion capacity are not interchangeable.
`K` is a stable business-operation key; `H` is the digest of canonical request parameters.
Both `K` and `H` are identifiers, not secret authorization credentials.
**Takeaway:** Distinguish a logical operation, its attempts, its recorded progress, and its actual effect.
# 4. Core Formulas with Full Derivation
## 4.1 Simplest / Base Case
**Validity:** Initially empty, lossless fluid queue; constant arrivals and service capacity; backlog grows during an interval with arrivals above capacity.
Conservation says backlog change equals arrivals admitted minus completed work.
For duration `t`, admitted work is `lambda*t` and completed work is `mu*t`.
Under this overload condition, subtracting the two gives the positive backlog.
```text
FORMULA [Q1]: B = (lambda - mu) * t
WHERE: B [jobs]; lambda, mu [jobs/s]; t [s].
CONSTRAINT: B starts at zero; lambda > mu; constant rates; no loss, retries or blocking.
NUMERICAL EXAMPLE: (12 - 10) jobs/s * 30 s = 60 jobs.
INTERPRETATION: 60 jobs must fit in the backlog after this interval.
```
Units cancel from jobs/s times seconds to jobs.
The result is a fluid average, not a per-request wait-time prediction.
The simple calculation does not guarantee enough buffer for a burst within the interval.
## 4.2 General Case
For an arbitrary positive initial backlog and constant rates, conservation applies until the queue empties.
An ideal work-conserving fluid server cannot process nonexistent jobs, so backlog is floored at zero.
```text
FORMULA [Q2]: B_end = max(0, B_start + (lambda - mu)*t)
WHERE: B_start and B_end [jobs]; lambda and mu [jobs/s]; t [s].
CONSTRAINT: nonnegative initial backlog; constant rates; lossless fluid server; no retries.
NUMERICAL EXAMPLE: max(0, 60 + (8 - 10)*20) = 20 jobs.
INTERPRETATION: 20 jobs remain after 20 seconds of recovery.
```
Setting `B_start` to zero and using `lambda > mu` recovers Q1.
For varying rates, integrate the net rate while reflecting at zero; a single final clamp on total net arrivals is not generally sufficient.
The simple formula excludes queue caps, producer blocking, retry amplification, and priority scheduling.
When arrivals remain below service while backlog is positive, the net drain rate is service minus arrivals.
```text
FORMULA [Q3]: T = B_start / (mu - lambda)
WHERE: T [s]; B_start [jobs]; mu and lambda [jobs/s].
CONSTRAINT: B_start > 0; mu > lambda; constant rates; no loss or retries.
NUMERICAL EXAMPLE: 60 jobs / (10 - 8) jobs/s = 30 s.
INTERPRETATION: 30 seconds are needed to clear the initial backlog in this model.
```
A zero or negative denominator does not give a finite drain time.
Service capacity must be sustainable while the backlog is present.
The average does not specify percentiles, starvation, or ordering.
## 4.3 Key Special Cases / Variants
| Variant | State change | Required contract | Use | Boundary |
|---|---|---|---|---|
| Database-only effect | Commit key and mutation together | One transactional database | Local ledger change | Cannot include independent API |
| Transactional outbox | Commit business intent and message in one transaction | Reliable relay and downstream dedupe | Cross-process delivery | Relay can publish twice |
| Provider key | Reuse business key on each request | Provider atomically deduplicates same payload | Remote effects | Retention and namespace may be limited |
| Query then retry | Resolve uncertain result before retry | Query is authoritative and stable enough | Timeouts with lookup | Negative lookup may be stale |
| Compensation | Submit a separate reversing operation | Reversal is valid and authorized | Recoverable business actions | A refund is not erasure of a charge |
| Manual review | Pause unsafe automation | Operators have evidence and authority | Unresolvable uncertainty | Slower and incurs operational cost |
A key-value store holding a pre-call marker is not equivalent to a committed remote effect.
A queue's at-least-once delivery needs consumer-side business deduplication.
A provider key may provide at-most-once effects only within its precise contract and retention period.
## 4.4 Asymptotic / Limiting Behaviour and Counterexample
If `mu = lambda` and backlog is positive, Q2 predicts no drain.
If `mu < lambda`, backlog grows while those constant rates persist.
If the queue is finite, eventually admission blocks or drops instead of following the infinite-buffer model.
If retries occupy capacity, use attempt load and measured service rather than assuming retries are free.
If a checkpoint says “call not complete” after the remote service charged, blindly retrying without a valid key can charge twice.
If the provider's key has expired, a retry with the same text key can still create a second effect.
If the provider returns a cached error for the key, retry policy must follow its actual documented contract.
If lookup is eventually consistent, a missing record at one instant is not proof of absence.
**Takeaway:** Conservation bounds queue capacity; effect safety follows atomicity and evidence, not checkpoint frequency.
# 5. Worked Numerical Example
## 5.1 Bounded workload and decision
This scenario is ILLUSTRATIVE; no workload, provider, or performance value was measured.
A durable workflow submits one shipping-label purchase per approved order.
The operational decision is whether a queue can absorb a 30-second overload and recover thereafter.
One job is one distinct approved label operation, not one network attempt.
At the start, the pending queue is empty.
During overload, new arrivals are 12 jobs/s.
The worker fleet completes 10 jobs/s.
The overload lasts 30 s.
Assume constant rates, no dropped jobs, retries, capacity fluctuation, or admission blocking.
Assume completed jobs leave the queue and there is sufficient queue space.
The scenario ignores queue metadata, storage replication, and variable payload sizes.
By Q1, backlog is `(12 - 10) * 30 = 60 jobs`.
Independently, arrivals are `12 * 30 = 360 jobs`.
Completions are `10 * 30 = 300 jobs`.
The difference `360 - 300` is 60 jobs.
The result describes the endpoint backlog, not the maximum latency of any particular job.
## 5.2 Recovery and boundary
After overload, arrivals fall to 8 jobs/s.
Workers continue at 10 completed jobs/s while the queue is nonempty.
By Q3, spare capacity is 2 jobs/s.
The estimated drain time is `60 / 2 = 30 s`.
Independently, recovery arrivals in 30 s total 240 jobs.
Recovery completions in 30 s total 300 jobs.
The difference removes the original 60 jobs.
At recovery arrival rate 9 jobs/s, net drain is 1 job/s and drain takes 60 s.
At 10 jobs/s there is no spare capacity, so the backlog does not drain under this model.
At 11 jobs/s it grows by 1 job/s until the model's assumptions fail.
Any real deployment needs measured utilization, variance, retry traffic, broker limits, and headroom.
The decision is to provision at least 60 jobs of theoretical backlog capacity plus separately justified overhead and burst headroom.
No exact production buffer setting can be inferred from these inputs alone.
## 5.3 Payload-only sizing
Suppose each pending job carries exactly 1,024 payload bytes.
That is an additional illustrative assumption, not a broker guarantee.
The 60-job backlog holds `60 * 1,024 = 61,440 bytes` of payload.
Since 1 KiB is 1,024 bytes, the payload is 60 KiB.
Indices, metadata, replicas, allocator overhead, and safety margin are excluded.
When queue entries store references rather than full payloads, size the references and external objects separately.
## 5.4 One job's crash-window trace
Order `O-17` has a single approved label purchase with operation key `tenant-A/O-17/label/1`.
The amount, currency, destination, and operation version form a canonical payload bound to `H`.
The workflow durably stores intent and `K` before dispatch.
Attempt 1 sends `K` and the payload to a provider that contractually deduplicates this operation.
The provider commits label `L-17` and records `K`, `H`, and the result atomically.
The provider's response is lost before the worker records completion.
The activity reaches its timeout; the outcome is unknown, not failed.
The workflow resumes from the last committed state and makes a new attempt with exactly the same `K` and `H`.
Within the provider's stated retention window, its dedupe record returns `L-17` rather than creating a second label.
The worker persists the result and advances the workflow checkpoint.
A duplicate message later returns the stored local result without issuing a new purchase.
This safe trace relies on the stated provider contract; it is not a universal property of API keys.
If retention expired or the provider cannot establish the original outcome, halt automatic purchase and reconcile.
## 5.5 Attempt-load sensitivity
Suppose 100 logical jobs each require one attempt except 20 that each need one retry.
The fleet processes `80*1 + 20*2 = 120` attempts for 100 jobs.
Average attempts per logical job are `120/100 = 1.2`.
At the same arrival rate and equal per-attempt cost, this simplified case demands 20% more attempt work.
Actual load depends on timing, errors, concurrency, payloads, and retries that overlap.
Do not reuse Q1's 10 completed jobs/s if that measured capacity excludes the extra retry work.
Backoff can spread the 20 retries but cannot make their total work disappear.
**Takeaway:** Size backlog from job conservation, then separately model retry work and side-effect safety.
# 6. Visual / Geometric Intuition
## 6.1 Time and ambiguity
```text
workflow          worker                     provider
  |                 |                            |
  | persist intent  |                            |
  |---------------->| request K,H                |
  |                 |--------------------------->|
  |                 |                    commit effect + K
  |                 |<------ response lost -------|
  |           timeout; result UNKNOWN            |
  | replay          |                            |
  |---------------->| retry same K,H             |
  |                 |--------------------------->|
  |                 |<------ original result -----|
  | checkpoint      |                            |
```
The dotted conceptual gap between provider commit and local checkpoint is the dangerous window.
Moving the checkpoint closer reduces the window but cannot make independent systems atomic.
A timeout may occur before dispatch, during transport, during execution, or after commit.
Only a reliable provider contract distinguishes these possibilities after communication loss.
## 6.2 State sketch
```text
NEW -> INTENT_RECORDED -> SUBMITTING -> CONFIRMED -> COMPLETED
                              |             |
                              v             v
                           UNKNOWN      reconciliation
                              |
                    query / safe same-key retry
                              |
                  CONFIRMED or MANUAL_REVIEW
```
The state sketch is a design example, not a vendor runtime's documented state machine.
A confirmed result must match the expected business identity and canonical payload.
A terminal local failure cannot assert the absence of an external effect when outcome remains unknown.
A compensating action is a new effect with its own identity and audit trail.
## 6.3 Checkpoint cut
```text
[local intent committed] | [send] | [remote effect] | [local result committed]
                         ^ crash cuts are possible at every separator
```
Before local intent commits, replay can safely recreate it only if identity is stable.
After send but before remote outcome is established, treat the result as unknown.
After remote commit but before local result commits, duplicate-safe retry or reconciliation is required.
After local result commits, repeated delivery should return or recover that result.
## 6.4 Timeline versus lease
A worker lease expiring is not proof that its process or network call stopped.
A second worker can start while the first is paused or partitioned.
A fencing token works only if the protected resource rejects stale tokens.
Remote providers that do not understand the fence require a separate idempotency guarantee.
Do not infer mutual exclusion at the provider from a queue lease alone.
**Takeaway:** Draw the remote commit and local checkpoint as separate moments.
# 7. Comparison / Selection Table
| Choice | Preconditions | Guarantee / limitation | Cost | Best use | Avoid or qualify |
|---|---|---|---|---|---|
| In-memory task | Restart may lose progress | No durable recovery | Low | Disposable computation | Irreversible effects |
| Database job row | Transactional persistence and polling | Durable local status; duplicate delivery possible | Polling and state storage | Simple bounded tasks | Multi-system atomicity assumptions |
| Durable workflow engine | Compatible persistence and engine semantics | Resumes workflow under engine contract | History, operations, migration | Long-lived multi-step work | Treating checkpoint as remote commit |
| Outbox plus consumer inbox | Same local database transaction and receiver dedupe | Durable intent and duplicate-tolerant messaging | Tables, relay, cleanup | Services with durable local effects | Claiming exactly-once network delivery |
| Provider idempotency key | Provider enforces key/payload and retention | Conditional suppression of repeated provider effects | Provider state, reconciliation | Remote business effect | Unsupported operations or expired keys |
| Distributed transaction | All participants support the same coordinator protocol | Atomic commit within its actual participant set | Locking and availability costs | Narrow compatible systems | Assuming generic HTTP participates |
| Reconciliation and review | Authoritative records, owner, alerting | Resolves or contains ambiguity | Operator time and latency | Missing/uncertain provider outcomes | Unowned permanent backlog |
## Selection rules
Start with a database transaction if the full effect is inside one database.
Use a durable workflow when process lifetime, timeouts, waits, and recovery justify the operational burden.
Use an outbox if a committed local decision must reliably lead to a message.
Choose a provider key only after checking namespace, mismatch behavior, retention, and atomic scope.
If a remote provider has neither deduplication nor authoritative lookup, do not promise at-most-one effect automatically.
Prefer a manual uncertainty queue to a second potentially harmful operation in that case.
A local unique constraint is valuable but does not close the remote commit gap by itself.
For multi-step business changes, consider explicit compensations with separate business approval.
A compensation can fail independently and must itself be idempotent or reconciled.
A simple database queue remains a credible baseline for bounded workloads.
The choice should consider recovery complexity, owner availability, and effect severity.
**Takeaway:** Choose guarantees at every boundary rather than assigning “exactly once” to the whole diagram.
# 8. Failure Modes & Edge Cases
All entries below are HYPOTHETICAL diagnostic exercises, not observed incidents.
The observed signal alone is never enough to establish the cause.
Each entry names an alternative explanation, discriminating evidence, bounded response, and verification.
## 8.1 Response lost after commit
**Signal:** Caller times out; provider has a completed effect.
**Competing cause:** Worker may have crashed before recording the response.
**Discriminating evidence:** Provider operation lookup by K shows the committed result.
**Immediate response:** Persist the matching result, not a new key.
**Prevention:** Protect later retries with the same bound K.
**Verification:** Confirm one provider record and a completed local record.
## 8.2 Failure before dispatch
**Signal:** The workflow timed out without a provider record.
**Competing cause:** A request may instead be queued in transit or lookup may lag.
**Discriminating evidence:** Correlate gateway send records with provider request logs and lookup consistency.
**Immediate response:** If absence is authoritative, retry with the original K.
**Prevention:** Persist intent before any send.
**Verification:** Check the retry yields one matching provider effect.
## 8.3 Crash before result checkpoint
**Signal:** Replay starts the effect activity again.
**Competing cause:** The provider may not have committed the first attempt.
**Discriminating evidence:** Compare provider status for K with the last durable checkpoint.
**Immediate response:** Reuse K or reconcile; never infer absence from the checkpoint.
**Prevention:** Test both sides of the remote-commit window.
**Verification:** Confirm history converges on the single provider result.
## 8.4 Checkpoint storage unavailable
**Signal:** Workflow progress repeatedly returns to an earlier step.
**Competing cause:** The worker could also be failing before checkpoint submission.
**Discriminating evidence:** Inspect checkpoint write acknowledgments and storage health.
**Immediate response:** Restore storage and pause unsafe effect dispatch.
**Prevention:** Alert on checkpoint commit failures rather than only worker exits.
**Verification:** Resume a bounded run and inspect its persisted progress.
## 8.5 Checkpoint restore from stale backup
**Signal:** Previously completed steps reappear after disaster recovery.
**Competing cause:** The provider may also have rolled back independently.
**Discriminating evidence:** Compare backup recovery point with external effect timestamps.
**Immediate response:** Hold dispatch and reconcile operations after the recovery point.
**Prevention:** Document cross-system recovery points and reconciliation ownership.
**Verification:** Match every affected intent to a final provider outcome.
## 8.6 Two workers hold apparent ownership
**Signal:** Two activity attempts overlap for one operation.
**Competing cause:** One worker may merely have an expired local lease.
**Discriminating evidence:** Inspect lease epochs and actual request timing.
**Immediate response:** Fence stale writers where the resource supports fencing; retain K downstream.
**Prevention:** Use monotonic epochs and enforce them at the protected store.
**Verification:** Inject a paused worker and confirm stale writes are rejected.
## 8.7 Lease expires during slow call
**Signal:** A second worker starts while the first is still running.
**Competing cause:** Slow provider responses may be caused by queueing rather than a stuck worker.
**Discriminating evidence:** Compare lease expiration, request start, and provider completion times.
**Immediate response:** Treat both calls as concurrent and require provider deduplication.
**Prevention:** Set leases and heartbeats for detection, not effect safety.
**Verification:** Verify concurrent same-key calls cause no second effect under contract.
## 8.8 Key changes on retry
**Signal:** The same business order produces two provider keys.
**Competing cause:** The order may actually contain two separately authorized purchases.
**Discriminating evidence:** Compare business operation versions and authorization records.
**Immediate response:** Stop new submissions and reconcile both provider effects.
**Prevention:** Derive K from a stable operation identity rather than attempt ID.
**Verification:** Replay the same logical order and inspect the generated K.
## 8.9 Key collision across tenants
**Signal:** One tenant receives a result belonging to another.
**Competing cause:** A response-cache isolation fault can produce the same symptom.
**Discriminating evidence:** Compare tenant namespace in K, database constraints, and provider scope.
**Immediate response:** Block cross-tenant replay and contain affected records.
**Prevention:** Include an unambiguous tenant namespace and authorize lookup.
**Verification:** Exercise identical order IDs for two isolated tenants.
## 8.10 Same key different amount
**Signal:** Provider reports a payload mismatch or silently returns an old result.
**Competing cause:** A legitimate amendment might need a new authorized operation.
**Discriminating evidence:** Compare canonical amount, currency, recipient, and payload digest.
**Immediate response:** Reject the conflicting request; investigate whether amendment is authorized.
**Prevention:** Bind K to canonical intent and require explicit versioning for changes.
**Verification:** Submit same K with a changed amount and verify rejection.
## 8.11 Noncanonical request encoding
**Signal:** Equivalent requests produce different payload digests.
**Competing cause:** The payload may contain an actual semantic difference.
**Discriminating evidence:** Compare normalized fields, decimal precision, and absent versus null values.
**Immediate response:** Apply a documented canonicalization version before binding.
**Prevention:** Pin normalization rules and retain their version.
**Verification:** Hash equivalent representations and verify identical semantic binding.
## 8.12 Key retention expired
**Signal:** A late retry creates a second remote effect.
**Competing cause:** The provider may not support deduplication for that endpoint at all.
**Discriminating evidence:** Inspect first request age and provider retention contract.
**Immediate response:** Stop automatic retries; reconcile or obtain human approval.
**Prevention:** Set local retry horizon below the documented provider retention.
**Verification:** Simulate expiry with a safe provider sandbox or controlled fake.
## 8.13 Provider lookup is stale
**Signal:** A lookup reports absent just after a successful submission.
**Competing cause:** The initial submission may really have failed before arrival.
**Discriminating evidence:** Compare visibility delay and authoritative status semantics.
**Immediate response:** Wait or route to reconciliation rather than treating absence as proof.
**Prevention:** Document consistency and safe retry conditions for each provider.
**Verification:** Inject delayed visibility and verify no unsafe fresh-key submission.
## 8.14 Concurrent same-key requests
**Signal:** Two simultaneous calls appear for one K.
**Competing cause:** They may have distinct payloads and represent a conflict.
**Discriminating evidence:** Inspect provider in-progress semantics and payload digests.
**Immediate response:** Wait for the recorded outcome or follow documented same-key behavior.
**Prevention:** Ensure provider enforces atomic in-flight deduplication.
**Verification:** Race identical requests and inspect the resulting effects.
## 8.15 Outbox publish repeated
**Signal:** The broker receives the same event twice.
**Competing cause:** Two separately committed outbox rows may be duplicates at source.
**Discriminating evidence:** Compare stable event IDs and local transaction records.
**Immediate response:** Let the consumer deduplicate by event or business operation ID.
**Prevention:** Make publication retries expected and consumer updates atomic.
**Verification:** Replay the event and confirm a single consumer-side mutation.
## 8.16 Outbox row never published
**Signal:** Business intent exists but no downstream event appears.
**Competing cause:** The consumer may have received the event but failed to acknowledge it.
**Discriminating evidence:** Compare outbox state, broker receipts, and consumer inbox.
**Immediate response:** Restore relay operation and replay the pending record.
**Prevention:** Monitor oldest unpublished outbox age.
**Verification:** Confirm the record reaches the consumer and is marked delivered.
## 8.17 Consumer acknowledges before commit
**Signal:** The broker shows delivery but the business row is absent.
**Competing cause:** A transaction rollback may have occurred for a validation error.
**Discriminating evidence:** Inspect acknowledgment time relative to database commit and errors.
**Immediate response:** Move acknowledgment after durable transaction commit.
**Prevention:** Treat redelivery as normal and use an inbox uniqueness constraint.
**Verification:** Crash between commit and acknowledgment and confirm safe redelivery.
## 8.18 Consumer commits then crashes
**Signal:** The broker redelivers an already applied event.
**Competing cause:** The first transaction might actually have rolled back.
**Discriminating evidence:** Read inbox record and business row in the same transactional database.
**Immediate response:** Return or acknowledge the recorded outcome on duplicate.
**Prevention:** Commit inbox marker and mutation atomically.
**Verification:** Verify replay does not increment the business value twice.
## 8.19 Dedupe marker written before effect
**Signal:** A record says processed but the business effect is missing.
**Competing cause:** The effect could be present in a delayed replica.
**Discriminating evidence:** Read the primary transaction log and authoritative business row.
**Immediate response:** Repair using a transaction that binds marker and local mutation.
**Prevention:** Do not commit a success marker before a non-atomic remote call.
**Verification:** Crash at the marker/effect boundary and inspect consistency.
## 8.20 Effect written before dedupe marker
**Signal:** Duplicate delivery applies the same local mutation twice.
**Competing cause:** Two legitimate operations may have been incorrectly grouped.
**Discriminating evidence:** Inspect transaction IDs and the business operation identity.
**Immediate response:** Reconcile duplicates and make mutation plus inbox insert atomic.
**Prevention:** Use a unique constraint on the actual operation identity.
**Verification:** Force redelivery after commit and confirm one mutation.
## 8.21 Infinite retry loop
**Signal:** An activity consumes workers indefinitely.
**Competing cause:** A slow but valid long-running operation can resemble repeated failure.
**Discriminating evidence:** Inspect attempt count, retry reason, elapsed budget, and provider status.
**Immediate response:** Apply a bounded attempt and deadline policy with escalation.
**Prevention:** Separate retryable transport faults from permanent errors.
**Verification:** Trigger a permanent error and confirm termination or review.
## 8.22 Retry storm after outage
**Signal:** Attempts spike when a dependency returns.
**Competing cause:** New user demand may have increased independently.
**Discriminating evidence:** Break down traffic by original versus retry attempt and scheduling time.
**Immediate response:** Throttle recovery and apply randomized backoff within deadline.
**Prevention:** Reserve capacity for first attempts and critical operations.
**Verification:** Observe bounded attempt rate during a controlled recovery drill.
## 8.23 Synchronized backoff
**Signal:** Many clients retry at identical intervals.
**Competing cause:** A shared batch scheduler may also cause periodic load.
**Discriminating evidence:** Plot attempt timestamps grouped by retry generation.
**Immediate response:** Introduce jitter and a bounded concurrency budget.
**Prevention:** Seed or use a safe random strategy appropriate to the runtime.
**Verification:** Check that retries spread across the intended interval.
## 8.24 Timeout shorter than normal latency
**Signal:** Healthy calls repeatedly time out and retry.
**Competing cause:** The provider may actually be degraded or the network path unstable.
**Discriminating evidence:** Compare downstream latency distribution, queue time, and timeout settings.
**Immediate response:** Adjust timeout only with a measured end-to-end budget.
**Prevention:** Monitor timeout rate alongside provider completion after timeout.
**Verification:** Confirm retries decrease without hiding genuine hangs.
## 8.25 Timeout longer than business deadline
**Signal:** A stale operation completes after the user-facing decision expired.
**Competing cause:** An explicit cancellation may also have failed to propagate.
**Discriminating evidence:** Compare deadline, provider commit time, and cancellation acknowledgment.
**Immediate response:** Reconcile the late result and apply authorized compensation if needed.
**Prevention:** Carry deadline and policy through each activity boundary.
**Verification:** Simulate a late response and confirm correct business disposition.
## 8.26 Cancellation assumed to undo effect
**Signal:** A canceled run still results in a charge.
**Competing cause:** The provider might have processed cancellation after committing.
**Discriminating evidence:** Check provider commit and cancellation timestamps and statuses.
**Immediate response:** Record the charge and follow the separate refund policy.
**Prevention:** Represent cancel-requested separately from cancel-confirmed.
**Verification:** Race cancel and commit and verify both possible outcomes.
## 8.27 Retry on permanent validation error
**Signal:** Every attempt returns the same rejected payload.
**Competing cause:** An intermittent schema deployment mismatch can mimic a permanent error.
**Discriminating evidence:** Inspect stable error code, schema version, and sample payload.
**Immediate response:** Stop automatic retries and correct the underlying input or version.
**Prevention:** Classify documented errors without assuming every 4xx or 5xx is uniform.
**Verification:** Verify the same invalid input does not loop.
## 8.28 Authorization expires during retry
**Signal:** The remote service rejects later attempts.
**Competing cause:** The principal may lack permission even with a fresh credential.
**Discriminating evidence:** Compare token expiry, audience, and authorization decision.
**Immediate response:** Refresh credentials only under the original valid authority.
**Prevention:** Keep authentication refresh separate from business approval validity.
**Verification:** Test a retry across credential expiry without changing K.
## 8.29 Approval no longer matches intent
**Signal:** A resumed workflow sends an amended purchase under old approval.
**Competing cause:** The change may have been a nonmaterial display-only transformation.
**Discriminating evidence:** Compare signed approval scope with canonical operation version and digest.
**Immediate response:** Pause and obtain a new approval for material changes.
**Prevention:** Bind approval to exact intent, actor, scope, and expiry.
**Verification:** Change amount after approval and verify execution is blocked.
## 8.30 Approval expires while waiting
**Signal:** A workflow resumes after its authorization window.
**Competing cause:** The displayed expiry may be in a different timezone or clock domain.
**Discriminating evidence:** Check stored UTC timestamps and authoritative authorization policy.
**Immediate response:** Request fresh approval rather than silently extending it.
**Prevention:** Use a trusted clock and explicit expiration semantics.
**Verification:** Resume just after expiry and verify a new approval is required.
## 8.31 Poison message blocks a partition
**Signal:** One event is repeatedly redelivered and later work is delayed.
**Competing cause:** The consumer could also have a broad database outage.
**Discriminating evidence:** Inspect per-message error, neighboring messages, and dependency health.
**Immediate response:** Quarantine the identified message with context and an owner.
**Prevention:** Define ordering and replay policy before using a dead-letter queue.
**Verification:** Repair the message or handler and verify controlled reprocessing.
## 8.32 Dead-letter treated as completion
**Signal:** Queue depth falls while business operations remain unfinished.
**Competing cause:** The jobs may have been intentionally rejected by policy.
**Discriminating evidence:** Compare dead-letter records with terminal business dispositions.
**Immediate response:** Assign review and reconciliation rather than reporting success.
**Prevention:** Track age and count of unresolved dead-letter items.
**Verification:** Confirm every removed item has an explicit terminal outcome.
## 8.33 Queue overflow
**Signal:** Arrivals disappear or producers receive backpressure.
**Competing cause:** A metrics gap may hide accepted records.
**Discriminating evidence:** Inspect admission acknowledgments, capacity, and broker drop policy.
**Immediate response:** Apply documented backpressure or shed work by policy.
**Prevention:** Set alerts on oldest age, depth, and rejected admissions.
**Verification:** Load a controlled queue to its limit and inspect producer outcomes.
## 8.34 Retry work consumes all capacity
**Signal:** Fresh requests wait while repeated attempts fill workers.
**Competing cause:** The service fleet may have lost workers or CPU capacity.
**Discriminating evidence:** Compare attempt generations, worker utilization, and logical completions.
**Immediate response:** Bound retry concurrency and protect capacity for new work.
**Prevention:** Measure attempts per operation and allocate retry budgets.
**Verification:** Exercise an outage and check first-attempt service remains bounded.
## 8.35 Checkpoint schema incompatible
**Signal:** Restored workflows fail after a deployment.
**Competing cause:** The stored state may instead be corrupt or missing.
**Discriminating evidence:** Compare state schema version, migration logs, and decoder error.
**Immediate response:** Pause affected runs and restore a compatible reader or migrate safely.
**Prevention:** Version persisted state and rehearse upgrades with representative histories.
**Verification:** Resume a copied old checkpoint under the new release.
## 8.36 Replay changes nondeterministically
**Signal:** The same history chooses a different branch after deployment.
**Competing cause:** An external event may legitimately have changed the history.
**Discriminating evidence:** Compare recorded inputs, code version, and runtime replay violations.
**Immediate response:** Use engine-supported versioning and recorded nondeterministic values.
**Prevention:** Do not call unrecorded external systems inside replay-sensitive code.
**Verification:** Replay a fixed history in the documented engine test environment.
## 8.37 Duplicate client submission
**Signal:** A user clicks twice and creates two workflow runs.
**Competing cause:** The user may genuinely intend two independent operations.
**Discriminating evidence:** Compare client intent identifier, timestamps, and business authorization.
**Immediate response:** Coalesce only when the operations represent the same intent.
**Prevention:** Accept a stable client operation identifier with payload binding.
**Verification:** Submit the same intent twice and verify one business operation.
## 8.38 Distinct operations share one key
**Signal:** The second legitimate purchase returns the first result.
**Competing cause:** The first result may belong to a real duplicate request.
**Discriminating evidence:** Compare authorized business operation IDs and operation versions.
**Immediate response:** Assign separate keys to genuinely distinct approved actions.
**Prevention:** Document the exact granularity of an idempotency key.
**Verification:** Submit two authorized purchases and verify distinct outcomes.
## 8.39 Provider caches transient error
**Signal:** Repeating K returns the same error indefinitely.
**Competing cause:** The provider may have no record of the initial request.
**Discriminating evidence:** Read provider documentation and inspect the key-specific response history.
**Immediate response:** Follow documented recovery or escalation; do not invent a fresh key blindly.
**Prevention:** Record provider-specific error caching behavior in the runbook.
**Verification:** Exercise the documented error path using a controlled integration.
## 8.40 Provider effect is asynchronous
**Signal:** API returns accepted but no final business outcome is known.
**Competing cause:** The provider could have already completed asynchronously.
**Discriminating evidence:** Query the provider state or validate the authenticated callback.
**Immediate response:** Track accepted separately from completed and reconcile overdue work.
**Prevention:** Define terminal states and expected callback/lookup windows.
**Verification:** Drop a callback and confirm polling or reconciliation closes the case.
## 8.41 Duplicate or forged callback
**Signal:** An operation advances twice or a malicious status is accepted.
**Competing cause:** A legitimate provider may retry delivery after a lost acknowledgment.
**Discriminating evidence:** Verify signature, event identity, operation association, and provider status.
**Immediate response:** Deduplicate authenticated callbacks and reject invalid signatures.
**Prevention:** Persist callback processing atomically with the local state change.
**Verification:** Replay a valid callback and attempt an invalid signature.
## 8.42 Out-of-order callbacks
**Signal:** A completed operation appears to return to pending.
**Competing cause:** The later event may describe an independent reversal.
**Discriminating evidence:** Compare provider sequence/version and event type rather than arrival time.
**Immediate response:** Apply valid state transitions and reconcile contradictions.
**Prevention:** Persist monotonic versions or query authoritative current state.
**Verification:** Deliver callbacks out of order and check state does not regress.
## 8.43 Unknown result marked failed
**Signal:** Operators retry a supposedly failed purchase with a new key.
**Competing cause:** An authoritative provider rejection may actually prove no effect.
**Discriminating evidence:** Inspect whether the error proves noncommit or is merely a timeout.
**Immediate response:** Restore UNKNOWN status and reconcile before new authorization.
**Prevention:** Model unknown as a first-class state.
**Verification:** Lose a response after commit and confirm no terminal failure assertion.
## 8.44 Compensation fails
**Signal:** An unintended effect remains after an attempted reversal.
**Competing cause:** The reversal may be accepted but not yet finalized.
**Discriminating evidence:** Inspect separate original and compensation operation states.
**Immediate response:** Escalate the unresolved compensation and keep both audit records.
**Prevention:** Give the compensation its own stable key and retry policy.
**Verification:** Replay the compensation and verify one reversal, not multiple.
## 8.45 Local dedupe record deleted early
**Signal:** An old duplicate is processed as a new operation.
**Competing cause:** The duplicate may be a legitimate later business action.
**Discriminating evidence:** Compare original key age, redelivery horizon, and retention policy.
**Immediate response:** Restore or reconcile history before allowing reprocessing.
**Prevention:** Align dedupe retention with maximum replay and audit horizons.
**Verification:** Replay near the retention boundary and verify the expected disposition.
## 8.46 Business state and event disagree
**Signal:** An event claims completion while the local ledger remains pending.
**Competing cause:** The event could describe a different operation version.
**Discriminating evidence:** Compare operation key, payload hash, transaction and event IDs.
**Immediate response:** Reconcile using the authoritative source and record correction.
**Prevention:** Write local state and outbox event in the same transaction.
**Verification:** Crash around commit and confirm no impossible partial local state.
## 8.47 Incorrect cross-region failover
**Signal:** Both regions submit the same effect during a partition.
**Competing cause:** Two different operations might have similar display identifiers.
**Discriminating evidence:** Compare global operation identity, replication lag, and provider requests.
**Immediate response:** Disable one submission path and reconcile all uncertain effects.
**Prevention:** Use globally enforced ownership or a shared downstream idempotency boundary.
**Verification:** Partition regions in a controlled drill and inspect provider effect count.
## 8.48 Stale fencing token accepted
**Signal:** A resumed old worker overwrites newer workflow state.
**Competing cause:** The newer worker may not have committed its update.
**Discriminating evidence:** Inspect monotonic token values in writes and storage enforcement.
**Immediate response:** Reject writes with stale tokens at the actual state store.
**Prevention:** Ensure token comparison is atomic with each protected mutation.
**Verification:** Pause an old worker and verify its subsequent write is rejected.
## 8.49 Audit history omits correlation
**Signal:** A provider effect cannot be matched to a workflow.
**Competing cause:** Logging may exist in a separate system or be redacted.
**Discriminating evidence:** Search secure records using provider reference and operation identity.
**Immediate response:** Quarantine unresolved effects for manual matching.
**Prevention:** Log stable identifiers, transitions, and versions without secrets.
**Verification:** Trace a sample operation end to end across boundaries.
## 8.50 Sensitive key exposed in logs
**Signal:** An operation identifier reveals customer or payment data.
**Competing cause:** The displayed value may be an opaque non-sensitive token.
**Discriminating evidence:** Inspect key construction and data-classification policy.
**Immediate response:** Restrict access and rotate or replace exposed identifiers as appropriate.
**Prevention:** Use opaque stable keys; never treat key possession as authorization.
**Verification:** Review representative logs for sensitive values and access control.
## 8.51 Provider refund changes outcome
**Signal:** The local workflow still displays the original purchase as final.
**Competing cause:** A refund request may be pending rather than completed.
**Discriminating evidence:** Compare settlement and reversal statuses from the provider ledger.
**Immediate response:** Record the reversal as a separate linked effect.
**Prevention:** Reconcile post-completion financial events on a defined schedule.
**Verification:** Inject a later reversal and verify ledger and user state agree.
## 8.52 Queue drain estimate misses bursts
**Signal:** The actual queue exceeds the calculated fluid backlog.
**Competing cause:** The average arrival rate or service capacity may also be wrong.
**Discriminating evidence:** Inspect short-interval rates, tail service times, and buffer rejection.
**Immediate response:** Add measured burst allowance and admission controls.
**Prevention:** Use the fluid result as a lower-detail planning model, not an SLO.
**Verification:** Compare predicted and observed backlog over a controlled burst.
## 8.53 Retry follows provider retention horizon
**Signal:** A delayed worker retries beyond the guaranteed dedupe window.
**Competing cause:** The provider may retain records longer but without a guarantee.
**Discriminating evidence:** Compare current time against first submission and contractual horizon.
**Immediate response:** Stop automatic dispatch and obtain authoritative reconciliation.
**Prevention:** Enforce a retry deadline shorter than the guaranteed window.
**Verification:** Advance a controlled clock past the horizon and check review routing.
## 8.54 Partial batch success
**Signal:** A batch call times out after only some items commit.
**Competing cause:** The provider may have applied the whole batch atomically.
**Discriminating evidence:** Inspect item-level status and documented batch atomicity.
**Immediate response:** Retry only unresolved items with stable item-level identities when supported.
**Prevention:** Design per-item reconciliation and persist item outcomes.
**Verification:** Fail a batch midstream and confirm no successful item is duplicated.
## 8.55 Non-idempotent notification
**Signal:** A retry sends the same customer email multiple times.
**Competing cause:** Two different notifications may have identical visible text.
**Discriminating evidence:** Compare notification intent IDs, recipient, and provider receipts.
**Immediate response:** Coalesce duplicate intent where provider capabilities allow it.
**Prevention:** Assign one operation key per distinct intended communication.
**Verification:** Redeliver the notification event and check recipient-side effect count.
## 8.56 Workflow history grows without bound
**Signal:** Replay and storage costs rise for long-lived executions.
**Competing cause:** A separate storage outage can cause similar restore latency.
**Discriminating evidence:** Measure history length, checkpoint size, and restore time by run age.
**Immediate response:** Use the engine-supported history rollover or checkpoint strategy.
**Prevention:** Set retention and archival policies consistent with audit needs.
**Verification:** Resume representative long histories after the change.
## 8.57 Version rollback loses decisions
**Signal:** An older binary cannot interpret new persisted state.
**Competing cause:** The state might have been independently corrupted.
**Discriminating evidence:** Compare writer version, reader compatibility, and state checksum.
**Immediate response:** Use a compatible reader or explicit reverse migration when safe.
**Prevention:** Plan both forward upgrade and rollback compatibility.
**Verification:** Rehearse rollback using a copy of representative new-version state.
## 8.58 Provider key scope misunderstood
**Signal:** The same key is accepted independently on different endpoints.
**Competing cause:** One call might use a different provider account.
**Discriminating evidence:** Compare endpoint, account, environment, and documented namespace.
**Immediate response:** Stop cross-boundary retry and reconcile each scope separately.
**Prevention:** Include endpoint and account in local operation mapping.
**Verification:** Exercise intended namespace boundaries in a safe integration.
## 8.59 False completion from transport ack
**Signal:** Broker acknowledgment is mistaken for a purchased label.
**Competing cause:** The consumer may already have completed the purchase.
**Discriminating evidence:** Compare broker receipt with consumer and provider terminal statuses.
**Immediate response:** Keep transport-delivered separate from effect-confirmed.
**Prevention:** Expose distinct state transitions and service-level metrics.
**Verification:** Hold the consumer and verify transport ack does not complete the workflow.
## 8.60 Non-atomic provider deduplication
**Signal:** Concurrent same-key calls create two effects despite matching keys.
**Competing cause:** The provider may scope keys per account or payload.
**Discriminating evidence:** Compare provider contract, concurrent request traces, and effect ledger.
**Immediate response:** Stop automated retries and escalate or change integration.
**Prevention:** Require explicit atomic in-flight semantics before relying on provider keys.
**Verification:** Use controlled concurrency tests where permitted and reconcile results.
**Takeaway:** Diagnose from independent workflow, transport, and provider evidence before retrying or closing an incident.
# 9. Complexity & Cost Analysis
## 9.1 Define the workload
A sizing calculation must state whether its unit is logical operations, delivery attempts, bytes, or worker-seconds.
A retry can add network work without adding an authorized logical operation.
A completed delivery can still leave an unresolved asynchronous provider effect.
Queue depth counts pending entries, while oldest age indicates whether a particular entry is starved.
A stable average throughput can coexist with long tails and deadline failures.
Capacity must cover ordinary arrivals, expected retries, replay traffic, and reconciliation requests.
An outage may increase all of these categories at once.
A priority queue changes who waits, not the total work demanded.
## 9.2 Attempt work
Consider the constructed 100-job, 120-attempt example in Section 5.
The arithmetic average of 1.2 attempts per job is a realized example, not an estimated population parameter.
If each attempt used exactly 0.5 worker-seconds, total attempt service would be 60 worker-seconds.
Without retries, 100 attempts at the same cost would use 50 worker-seconds.
The 10 worker-second difference assumes all attempts cost the same, including failures.
In practice a timeout can consume more worker time than a quick successful request.
Concurrency, queue wait, and provider throttling can change the effective service cost.
The 60 worker-seconds can be spread across multiple workers; it is not automatically 60 seconds of elapsed time.
If ten workers each sustain one worker-second of work per second, the ideal lower bound is six seconds of service.
That lower bound excludes coordination, concurrency limits, network waiting, and imperfect utilization.
For comparison, idealized 100-attempt service at this worker rate takes five seconds.
No tail-latency, throughput, or production fleet-size claim follows from this idealization.
## 9.3 Retention and storage
The job payload calculation in Section 5 gives 60 KiB for the illustrated endpoint backlog.
A dedupe record can outlive the queue item because retries and audit lookups may occur later.
Suppose an illustrative service retains 1,000,000 operation records at 512 bytes of logical payload each.
The payload-only total is 512,000,000 bytes, approximately 488.28 MiB.
This conversion divides by 1,048,576 bytes per MiB; it does not assume compression.
Indexes, row headers, write-ahead logs, replication, and backups are excluded.
Doubling retention volume doubles this payload-only total under fixed record size.
Deleting records earlier may break safety if old operations can still be replayed.
Permanent retention is not automatically appropriate because privacy, legal, and security obligations also matter.
Retention policy must reconcile provider guarantees, queue retention, workflow replay, legal holds, and data minimization.
When the safe replay horizon is unknown, do not present an arbitrary TTL as safe.
## 9.4 Algorithmic work and actual resources
A workflow step can add a checkpoint write and potentially a durable history entry.
The cost of restoring history depends on the specific engine and persistence implementation.
A point lookup into an indexed idempotency table depends on index, database, and load; do not assume constant time.
An append-only event history grows with recorded events unless archival or rollover changes the representation.
An outbox relay requires scanning or indexed selection, broker sends, and state updates.
A consumer inbox requires uniqueness checking and durable records for the retention window.
Reconciliation has request volume, provider-rate-limit, and operator-time costs.
A provider may charge for attempts, successful effects, polling, or combinations; contract terms are required.
Avoid assuming that a retry is free just because the provider suppresses a duplicate effect.
Monitoring also consumes storage and must avoid high-cardinality metrics keyed by every operation.
Use operation IDs in controlled logs or traces, not as unbounded metrics labels.
## 9.5 Operational signals
Record accepted logical operations separately from outbound attempt counts.
Record successful business effects separately from transport acknowledgments.
Track the population, window, numerator, and denominator for each rate.
A useful timeout rate is timed-out attempts divided by started attempts in the same window.
It is not the fraction of business operations that ultimately failed.
Track `UNKNOWN` age and count, not only explicit errors.
Track retry budget exhaustion and reason categories.
Track oldest unpublished outbox record and unprocessed inbox lag.
Track checkpoint write failure and restore failure independently.
Track provider lookup availability and reconciliation backlog.
Track observed provider-key conflicts without logging sensitive payloads.
Alert thresholds require measured baselines, impact, and an accountable response owner.
**Takeaway:** Capacity and cost must count attempt work, retained safety state, and the operational burden of uncertainty.
# 10. Key Theorems / Results Reference
## 10.1 Local transactional invariant
**Claim:** In the illustrative local database design, a business mutation and its inbox marker are both committed or neither is committed.
**Preconditions:** Both writes are in the same ACID transaction and the database guarantees atomic commit.
**Argument:** A committed transaction contains both writes; a rolled-back transaction contains neither.
A unique operation constraint prevents two successful insertions for the same operation identity.
A duplicate transaction must inspect and return the existing result rather than apply another mutation.
The effect of the mutation must itself be in that transaction for this argument to hold.
An external HTTP request is not included merely because it was made between transaction statements.
A database backup restored without its corresponding event history can invalidate an operational interpretation of the invariant.
## 10.2 Conditional at-most-one provider effect
**Claim:** At most one provider effect follows for one K under the explicit provider contract in Section 5.
**Preconditions:** Same provider namespace, same K and canonical payload, unexpired retention, and atomic concurrent deduplication.
The provider must atomically bind effect and dedupe outcome, or provide an equivalent durable guarantee.
**Argument:** The first accepted operation claims K; all later matching attempts resolve to its recorded state or result.
The result does not establish that an effect will eventually occur.
It also does not cover a second key, a second provider account, or an expired key.
The claim is a conditional engineering invariant, not a universal theorem about HTTP.
## 10.3 Recovery liveness
A retry policy can improve the chance of progress after a transient fault.
Liveness additionally needs eventual dependency recovery, scheduling, available capacity, and a valid retry horizon.
A bounded retry budget deliberately stops attempting after a limit.
Stopping can preserve safety but leaves an explicit unresolved case for reconciliation.
A provider that never answers cannot be made to prove the outcome by infinite local retries.
Exactly-once business outcome is a composite claim requiring both at-most-once safety and eventual completion.
The latter can fail even when the former holds perfectly.
## 10.4 Conservation result
Q1–Q3 follow from a lossless, constant-rate fluid model and the nonnegative queue floor.
They are not queueing-theory latency distributions and do not assume Poisson arrivals.
They do not account for variable-rate intervals by simply applying one average in every intermediate period.
An empty queue cannot accumulate negative backlog to offset a later burst.
## 10.5 Outbox safety and delivery boundary
An outbox transaction can ensure that a committed local decision has a corresponding locally committed event.
The transaction must encompass both the decision and the outbox insert in the same database.
If the transaction aborts, neither record is committed under the assumed atomicity guarantee.
The relay can publish an event and crash before marking it delivered.
After restart, the relay may publish the same event again.
Thus the outbox supports durable delivery intent, not exactly-once transport.
The receiver must use a stable event identity and an atomic inbox/business mutation to tolerate duplicates.
If the receiving effect is external, the receiver still needs the provider-side contract or reconciliation.
## 10.6 Authentication is not deduplication
An authorized caller can make the same request twice after a lost response.
An idempotency key distinguishes operations but does not prove the caller may perform them.
The service must authenticate and authorize both initial requests and result lookups.
An opaque key reduces unnecessary disclosure but is not an access token.
A stored result should be released only to an authorized principal in its tenant scope.
A retry after an approval expires requires the applicable business policy, not just a valid credential.
## 10.7 Retention is part of the guarantee
A provider's minimum guaranteed key retention bounds how long duplicate suppression can be assumed.
A locally retained key can still be treated as new by the provider after its retention expires.
The local operation record must remain available for the longest legitimate replay and reconciliation horizon.
Provider lookup and deduplication windows can differ and must be documented separately.
If those windows are shorter than the business recovery horizon, an accountable manual process is necessary.
Never infer a safe retention interval solely from a single successful test.
**Takeaway:** State each guarantee with its participant set, assumptions, and safety-versus-liveness boundary.
# 11. Interview / Exam Questions
All cases and numbers below are constructed exercises, not claims about named employers.
Each answer is conditional on the stated contracts; alternative defensible designs are noted.
## Level 1 — Recall / Concept
### 11.1 What does a durable checkpoint establish?
**Prompt:** What does a durable checkpoint establish?
**Answer:** A persisted recovery point under the runtime contract, not a guarantee about a remote effect.
**Reasoning:** The provider may commit after the checkpoint but before the worker records a result.
**Mechanism:** The next run must inspect durable history and reconcile external state.
**Boundary:** It can safely skip a completed local pure computation when the engine records that completion.
**Operational implication:** It cannot infer that a timed-out purchase never happened.
**Additional check:** Checkpoint more often if recomputation cost matters, but retain downstream idempotency.
**Acceptable alternative:** Alternative: a transactional local job table can be sufficient for simpler work.
**Red flag:** Red flag: claiming every resumed activity is executed exactly once.
**Follow-up:** Follow-up: identify the commit boundary for each independent participant.
### 11.2 Define idempotency for a business operation.
**Prompt:** Define idempotency for a business operation.
**Answer:** Repeating the same authorized intent yields the same intended effect as one application.
**Reasoning:** A stable key identifies that intent across attempts.
**Mechanism:** The receiver must bind the key to the same canonical payload and effect.
**Boundary:** A repeated key with different material payload must be rejected or handled by an explicit contract.
**Operational implication:** A duplicated network request can still happen even when the effect is deduplicated.
**Additional check:** A separate authorized purchase needs its own operation identity.
**Acceptable alternative:** Alternative: a single database transaction and unique business constraint may provide the property locally.
**Red flag:** Red flag: asserting that adding a request header is sufficient by itself.
**Follow-up:** Follow-up: explain key namespace and retention.
### 11.3 Why is a timeout not a failure proof?
**Prompt:** Why is a timeout not a failure proof?
**Answer:** A timeout only proves that the caller stopped waiting within its local deadline.
**Reasoning:** The remote endpoint can still be executing or may have committed before the response was lost.
**Mechanism:** Mark the effect UNKNOWN until authoritative information resolves it.
**Boundary:** A retry must use the same valid business key if the provider contract permits.
**Operational implication:** Cancellation must be separately confirmed; it is not implied by timeout.
**Additional check:** A documented synchronous rejection can provide stronger evidence than a transport timeout.
**Acceptable alternative:** Alternative: route directly to manual reconciliation when safe automated lookup is unavailable.
**Red flag:** Red flag: minting a new key because the first request timed out.
**Follow-up:** Follow-up: distinguish attempt timeout from the overall business deadline.
## Level 2 — Application / Derivation
### 11.4 Calculate backlog for a bounded overload.
**Prompt:** Calculate backlog for a bounded overload.
**Answer:** With 12 arrivals/s, 10 completions/s, and 30 seconds, the backlog is 60 jobs.
**Reasoning:** Start empty; assume constant rates, sufficient buffer, no retries, and no loss.
**Mechanism:** Arrivals total 360 and completions total 300.
**Boundary:** Their difference is 60 jobs, the same result as Q1.
**Operational implication:** The units of jobs/s times seconds are jobs.
**Additional check:** The number is a fluid endpoint backlog, not a latency percentile.
**Acceptable alternative:** Alternative: use time-stepped conservation when rates vary materially.
**Red flag:** Red flag: calling 60 jobs the drain time.
**Follow-up:** Follow-up: explain why a finite queue might reject work instead.
### 11.5 Calculate recovery time with continuing arrivals.
**Prompt:** Calculate recovery time with continuing arrivals.
**Answer:** A 60-job backlog drains in 30 seconds when capacity is 10 jobs/s and arrivals are 8 jobs/s.
**Reasoning:** The drain rate is the 2 jobs/s surplus, not the full 10 jobs/s.
**Mechanism:** Divide 60 jobs by 2 jobs/s to get 30 s.
**Boundary:** Check: 300 completions minus 240 arrivals remove 60 jobs.
**Operational implication:** If arrivals rise to 9 jobs/s, the same model gives 60 seconds.
**Additional check:** If arrivals equal capacity, no finite drain time follows.
**Acceptable alternative:** Alternative: use measured interval rates or simulation when capacity is variable.
**Red flag:** Red flag: calculating 60/10 while ignoring new arrivals.
**Follow-up:** Follow-up: discuss retry traffic that consumes worker capacity.
### 11.6 Trace a crash after remote commit.
**Prompt:** Trace a crash after remote commit.
**Answer:** Persist intent with K, send K, lose the response, and mark the outcome unknown.
**Reasoning:** On recovery, use provider lookup or a same-K retry supported by the contract.
**Mechanism:** If the matching result is returned, durably record it and advance the workflow.
**Boundary:** The provider must bind K to the correct payload and preserve the binding during the retry.
**Operational implication:** A unique local marker alone cannot prove what happened remotely.
**Additional check:** A checkpoint before the request does not prove the request was never committed.
**Acceptable alternative:** Alternative: place the case in a manual reconciliation queue if no safe query or retry exists.
**Red flag:** Red flag: use a new key to force a successful response.
**Follow-up:** Follow-up: explain what changes after provider dedupe retention expires.
### 11.7 Show why a local inbox and mutation must commit together.
**Prompt:** Show why a local inbox and mutation must commit together.
**Answer:** Atomic commit prevents a durable inbox marker without the corresponding local mutation.
**Reasoning:** A unique operation constraint ensures only one transaction can establish that marker.
**Mechanism:** If the transaction aborts, both writes roll back.
**Boundary:** If it commits, a redelivery can discover the stored result.
**Operational implication:** An HTTP request made during the transaction remains outside this atomicity boundary.
**Additional check:** A marker committed before the effect can falsely suppress needed work.
**Acceptable alternative:** Alternative: use a naturally idempotent database operation when its contract is explicit.
**Red flag:** Red flag: assuming a queue acknowledgment is equivalent to a business commit.
**Follow-up:** Follow-up: describe a crash between database commit and broker acknowledgment.
### 11.8 Choose a key for a retried purchase.
**Prompt:** Choose a key for a retried purchase.
**Answer:** Use a stable opaque identifier for one tenant-scoped authorized business operation.
**Reasoning:** Persist it before dispatch and reuse it for every attempt of that operation.
**Mechanism:** Bind it to canonical amount, currency, recipient, operation type, and relevant version.
**Boundary:** The run ID or attempt number should not create a new identity on replay.
**Operational implication:** A separate intentional purchase requires a distinct authorized operation identity.
**Additional check:** A material amendment requires explicit policy, often a new approval and operation version.
**Acceptable alternative:** Alternative: use a randomly generated persisted operation UUID with an enforced unique business mapping.
**Red flag:** Red flag: deriving K only from a mutable timestamp or retry count.
**Follow-up:** Follow-up: explain how the provider scopes and expires the key.
## Level 3 — Judgment / Design / Debug
### 11.9 Choose between an outbox and provider deduplication.
**Prompt:** Choose between an outbox and provider deduplication.
**Answer:** Use an outbox for atomic local intent and message creation, and provider deduplication for the independent external effect.
**Reasoning:** The two controls solve different transaction-boundary problems and may both be necessary.
**Mechanism:** A relay can publish a message more than once after an acknowledgment loss.
**Boundary:** The consumer must deduplicate the resulting local business mutation.
**Operational implication:** A downstream provider must independently protect or reconcile its side effect.
**Additional check:** Neither control makes a generic remote HTTP call part of the database transaction.
**Acceptable alternative:** Alternative: use only a database transaction if the complete effect is local to that database.
**Red flag:** Red flag: claiming the outbox alone guarantees exactly one provider charge.
**Follow-up:** Follow-up: identify where the outbox row and business state commit.
### 11.10 Diagnose two apparent active workers for one operation.
**Prompt:** Diagnose two apparent active workers for one operation.
**Answer:** Check leases, epochs, start times, provider K, and actual downstream effects.
**Reasoning:** An expired lease does not terminate the original worker or its in-flight call.
**Mechanism:** A fenced store rejects stale epochs only if every protected write checks them atomically.
**Boundary:** The provider may not understand those epochs and still needs its own dedupe contract.
**Operational implication:** Contain new dispatch while reconciling the operation if duplicate safety is uncertain.
**Additional check:** A slow call and a dead worker require different operational fixes.
**Acceptable alternative:** Alternative: use a single globally enforced owner plus downstream idempotency where feasible.
**Red flag:** Red flag: assume a lease alone prevents two external submissions.
**Follow-up:** Follow-up: propose a paused-worker fault injection.
### 11.11 Handle a retry beyond the provider key-retention window.
**Prompt:** Handle a retry beyond the provider key-retention window.
**Answer:** Stop automatic resubmission unless an authoritative contract still guarantees deduplication.
**Reasoning:** The same textual key may be treated as new after expiry.
**Mechanism:** Query the authoritative ledger and reconcile with local intent and any provider reference.
**Boundary:** A stale negative lookup is insufficient if the provider is eventually consistent.
**Operational implication:** If the outcome cannot be resolved, send the case to an accountable manual process.
**Additional check:** A new purchase needs explicit authorization, not a hidden key change.
**Acceptable alternative:** Alternative: use a longer contractual retention or independently durable provider operation reference.
**Red flag:** Red flag: assume key validity is permanent.
**Follow-up:** Follow-up: choose a retry deadline relative to the guaranteed horizon.
### 11.12 Respond to an outage that triggers a retry storm.
**Prompt:** Respond to an outage that triggers a retry storm.
**Answer:** Separate new logical arrivals from attempts and identify the retry generation.
**Reasoning:** Apply bounded retries with jitter, concurrency controls, and end-to-end deadlines.
**Mechanism:** Reserve appropriate capacity for new or critical work while maintaining fairness.
**Boundary:** Do not drop unknown outcomes merely to lower the retry queue depth.
**Operational implication:** Use provider health and per-error classification to avoid retrying permanent errors.
**Additional check:** Check that reconciliation capacity and provider rate limits can absorb recovery traffic.
**Acceptable alternative:** Alternative: use a circuit breaker with explicit probing and durable deferred work.
**Red flag:** Red flag: remove all retry limits or assume jitter reduces total attempt work.
**Follow-up:** Follow-up: calculate the 1.2 attempts/job example and its workload implications.
### 11.13 Design recovery after a restored stale database backup.
**Prompt:** Design recovery after a restored stale database backup.
**Answer:** Freeze risky dispatch and identify the backup recovery point and affected operations.
**Reasoning:** Compare local intents and results with authoritative provider effects after that point.
**Mechanism:** Preserve original operation identities even when workflow runs are recreated.
**Boundary:** Rebuild local state from trusted records only under documented access and consistency assumptions.
**Operational implication:** If provider key retention has elapsed, do not automatically replay unresolved operations.
**Additional check:** Route mismatches to reconciliation, and document which system is authoritative for each field.
**Acceptable alternative:** Alternative: restore a coordinated consistent snapshot if participants and recovery tooling actually support it.
**Red flag:** Red flag: replay every uncompleted local row as a fresh provider operation.
**Follow-up:** Follow-up: state how you will verify closure and report unresolved uncertainty.
**Takeaway:** Strong answers identify the exact boundary, show the trace or arithmetic, and preserve uncertainty when evidence is incomplete.
# 12. Quick Reference Card
| Item | Value |
|---|---|
| Topic | Durable workflows, retry safety and externally visible effects |
| Core formula | Drain time = backlog / (completion rate − arrival rate), when rates are constant and spare capacity is positive |
| Key assumption | One operation key and canonical payload map to at most one effect only when the executor enforces that contract atomically within its retention period |
| Complexity | Retry work and retained state depend on attempt count, queue size, payload, history and retention; Section 9 gives the workload assumptions |
| Use when | Recovering long-running work after failures without losing operation identity |
| Avoid when | A timed-out side effect is blindly replayed without deduplication or authoritative reconciliation |
| Go deeper | [Workflow collection index](INDEX.md); verify the specific provider's versioned semantics |
| Diagnostic | Compare operation identity, checkpoint, request trace, provider ledger and deduplication retention |
| Evidence | Examples are illustrative; source contents unavailable; draft and unreviewed |
