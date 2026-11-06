---
collection: "30_agentic_systems_workflows_and_orchestration"
file_id: "04"
topic: "Agent sessions, state ownership, isolation and run lifecycle"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Change in identity, storage consistency, tool contracts or retention policy"
notation_style: "plain-text"
prerequisites: []
pattern_tags: [agent, control-plane, session, state, lifecycle]
source_status: "OPEN"
owner_role: "Agent platform engineering"
source_index: "INDEX.md"
---
# Agent Sessions, State Ownership, Isolation and Run Lifecycle

## 1. File Header
> **Purpose:** Design and troubleshoot durable agent runs without confusing model context with authoritative state.
> **Prerequisites:** Basic HTTP authentication, database transactions, distributed failure and access control; see [PostgreSQL concurrency control](https://www.postgresql.org/docs/current/mvcc.html).
> **Key outcome:** Given tenant, session, run and tool requirements, trace ownership and recovery, choose isolation and estimate retained state.
> **This file does NOT cover:** Model training, a specific agent framework's API, or legal advice on retention.
> **Level:** 3 — Advanced / Production.
> **Version / reviewed:** 1.0 / not reviewed; draft.
> **Evidence:** DERIVED invariants and illustrative examples; external normative behavior is sourced individually. Production performance NOT MEASURED.
> **Retrieval cue:** For duplicate side effects use Section 8; for sizing use Sections 4, 5 and 9.

The decision is whether to resume, retry, stop or reconcile a run without exposing another principal's state or repeating an external action.
The control plane owns identity, admission and policy decisions; an execution worker performs assigned work under that authority.
A session can outlive a run, while an external payment or ticket can outlive both.
A model response is not an authorization decision or a durable source of truth.
**Takeaway:** Define the ownership and recovery boundary before choosing storage or an agent framework.

## 2. Concept Map / Mental Model
```text
Authenticated caller
    | request + tenant identity
    v
Control plane --authorize/admit--> session record (tenant, owner, policy)
    |                                |
    | creates run                    | versioned references
    v                                v
Run state machine <----checkpoint--- durable state store
    | dispatch + scoped capability
    v
Worker --> model context (derived, untrusted content included)
    |          |
    |          +--> model proposal, not authority
    v
Policy check --> tool adapter --> external system of record
    |                 |                |
    |                 +-- effect key --+
    +-- audit/receipt --> durable run history
```
Arrows denote request or data flow, not causal guarantees; durable commit boundaries matter.
The authorization boundary is checked on each read and action, not just when the session is created.
A run is a finite attempt to advance a session, and an approval may suspend that attempt.
A restart reconstructs working context from committed records rather than trusting worker memory.
The central invariant is that every visible state transition is attributable to the right tenant and run version.
The external effect boundary is separate: a database commit cannot by itself atomically commit an unrelated external API.
**Takeaway:** A model can suggest the next step, but the control plane decides what can be committed or executed.

## 3. Foundational Definitions

### Control plane
- **Definition:** Authority that admits, schedules and governs runs.
- **Type / units:** logical service; identifiers are opaque, not secret proofs.
- **Example:** A scheduler's authenticated admission decision.
- **Nonexample:** A model response is not an authority.
- **Operational check:** Inspect policy version and admission record.

### Data plane
- **Definition:** Execution path that invokes models or tools.
- **Type / units:** worker process and network calls; identifiers are opaque, not secret proofs.
- **Example:** A worker receives a scoped run assignment.
- **Nonexample:** A worker is not the tenant owner.
- **Operational check:** Inspect dispatch and runtime identity.

### Tenant
- **Definition:** Administrative isolation domain.
- **Type / units:** opaque tenant identifier; identifiers are opaque, not secret proofs.
- **Example:** Organization A and organization B.
- **Nonexample:** A user can belong to more than one tenant.
- **Operational check:** Verify authenticated tenant binding.

### Principal
- **Definition:** Authenticated human or service identity.
- **Type / units:** subject identifier; identifiers are opaque, not secret proofs.
- **Example:** A support agent authenticated through SSO.
- **Nonexample:** A session identifier is not an identity credential.
- **Operational check:** Inspect identity issuer and subject.

### Session
- **Definition:** Durable interaction container with ownership and policy.
- **Type / units:** tenant-scoped session identifier; identifiers are opaque, not secret proofs.
- **Example:** A support conversation spanning several requests.
- **Nonexample:** An individual worker process is not the session.
- **Operational check:** Read the session owner and version.

### Run
- **Definition:** Bounded execution advancing a session toward a requested outcome.
- **Type / units:** run identifier and state; identifiers are opaque, not secret proofs.
- **Example:** One response generation and its tool calls.
- **Nonexample:** A run is not the whole conversation.
- **Operational check:** Inspect terminal state and attempt lineage.

### Attempt
- **Definition:** One execution of a run or step, potentially retried.
- **Type / units:** attempt identifier and ordinal; identifiers are opaque, not secret proofs.
- **Example:** Worker restart after a crash.
- **Nonexample:** A new attempt is not automatically a new intended action.
- **Operational check:** Compare attempt and effect identifiers.

### State owner
- **Definition:** Component authoritative for a particular state field.
- **Type / units:** owner and write contract; identifiers are opaque, not secret proofs.
- **Example:** Ticket system owns ticket status.
- **Nonexample:** A cached copy is not necessarily authoritative.
- **Operational check:** Read the ownership map and conflict rule.

### Transcript
- **Definition:** Ordered recorded messages and tool events.
- **Type / units:** sequence of timestamped records; identifiers are opaque, not secret proofs.
- **Example:** User request followed by tool result.
- **Nonexample:** The model prompt may include only a subset.
- **Operational check:** Inspect persisted event sequence.

### Context
- **Definition:** Material assembled for a model invocation.
- **Type / units:** ordered tokens with provenance; identifiers are opaque, not secret proofs.
- **Example:** A selected history, policy and retrieved excerpts.
- **Nonexample:** Context is not necessarily complete durable history.
- **Operational check:** Record assembly version and source IDs.

### Checkpoint
- **Definition:** Committed recovery position and state snapshot or references.
- **Type / units:** versioned durable record; identifiers are opaque, not secret proofs.
- **Example:** Run paused before approval.
- **Nonexample:** A log line alone may not be a recoverable checkpoint.
- **Operational check:** Restore in a controlled exercise.

### Lease
- **Definition:** Time-bounded claim on work.
- **Type / units:** holder, expiry and generation; identifiers are opaque, not secret proofs.
- **Example:** Worker A owns a run until expiry.
- **Nonexample:** Lease expiry alone does not stop a partitioned worker.
- **Operational check:** Enforce fencing at the write boundary.

### Fencing token
- **Definition:** Monotonically ordered ownership generation.
- **Type / units:** integer per protected resource; identifiers are opaque, not secret proofs.
- **Example:** Generation 8 rejects a late generation 7 write.
- **Nonexample:** A random request ID is not necessarily ordered.
- **Operational check:** Check receiver rejects older generations.

### Idempotency key
- **Definition:** Stable identifier for one intended effect.
- **Type / units:** scoped opaque key; identifiers are opaque, not secret proofs.
- **Example:** One order submission across retries.
- **Nonexample:** A fresh key per retry defeats deduplication.
- **Operational check:** Query receiver's key scope and retention.

### Approval
- **Definition:** A principal's scoped authorization of a specific proposed action.
- **Type / units:** decision plus payload hash, actor and expiry; identifiers are opaque, not secret proofs.
- **Example:** Approve one exact refund request.
- **Nonexample:** General agreement to help is not approval for arbitrary effects.
- **Operational check:** Compare approved hash with executed payload.

### Retention
- **Definition:** Policy for how long data remains available before deletion.
- **Type / units:** time duration and hold rules; identifiers are opaque, not secret proofs.
- **Example:** Transcript expires after a configured period.
- **Nonexample:** Expiration of a cache is not deletion everywhere.
- **Operational check:** Trace copies, backups and legal holds.

### Reconciliation
- **Definition:** Process comparing local intent with external outcome.
- **Type / units:** bounded query or operator workflow; identifiers are opaque, not secret proofs.
- **Example:** Find whether a timed-out submission created an order.
- **Nonexample:** Blind retry is not reconciliation.
- **Operational check:** Match effect key to external receipt.

### Isolation
- **Definition:** Prevention of unauthorized cross-scope access or interference.
- **Type / units:** policy and enforcement boundary; identifiers are opaque, not secret proofs.
- **Example:** Tenant-scoped read and write checks.
- **Nonexample:** A unique identifier alone is not access control.
- **Operational check:** Attempt cross-tenant operations under test identities.

**Takeaway:** Each identifier, record and boundary has a distinct role; none substitutes for authorization.

## 4. Core Formulas with Full Derivation
### 4.1 Simplest / Base Case
**Mechanism — compare-and-set transition.**
Preconditions: authoritative storage atomically compares the current version and writes the new state; one tenant-scoped run key is selected.
Input: tenant T, run R, expected version v, allowed transition s to s-prime, and authenticated actor A.
Algorithm: authorize A for T and R; read state s and v; validate transition; atomically update only if T, R, s and v still match.
On success persist s-prime and v+1 in the same transaction with the audit event.
On a zero-row update return a conflict and reload; never silently overwrite the newer value.
Output: a committed transition or an explicit conflict; validation and retry loops must have a bounded deadline.
Invariant: at most one concurrent compare-and-set can change the same version when the database provides atomic conditional updates.
Justification: the first successful update changes v; the predicate for another update expecting v is then false.
Trace: workers A and B both see version 4; A commits version 5; B's predicate version=4 matches zero rows.
Boundary: this protects this database record, not an external side effect already sent by B.
A storage backend must actually implement the required atomicity; do not assume it from a generic key-value interface.

**Sizing equation S1 — retained payload only.**
Validity: constant number of closed sessions, constant mean raw bytes per session, same retention for all, no deletion lag or replicas.
Derivation: sum the per-session bytes across N sessions; constant average B yields N times B.
[1] FORMULA: P = N * B.
[2] WHERE: P is payload bytes, N is retained session count, B is mean payload bytes/session.
[3] CONSTRAINT: N and B are nonnegative; B excludes metadata, indexes and replication unless explicitly included.
[4] NUMERICAL EXAMPLE: N=100 sessions and B=2,000 bytes/session gives P=200,000 bytes.
[5] INTERPRETATION: This is the lower-bound logical payload, not physical disk provisioned.
Boundary: a heavy-tailed size distribution can make mean-based planning insufficient for per-session limits.

### 4.2 General Case
**Sizing equation S2 — cohort retention and physical multiplier.**
Validity: independent daily cohorts with retention D days and stationary daily arrivals A; each retained session has mean B bytes; multiplier M represents measured physical overhead and replica copies.
Derivation: D daily cohorts each contribute A sessions, so N=A*D; substitute into S1 and multiply physical overhead by M.
[1] FORMULA: C = A * D * B * M.
[2] WHERE: C is physical bytes; A is sessions/day; D is days; B is bytes/session; M is dimensionless.
[3] CONSTRAINT: consistent cohort granularity, stationary rates and an explicit definition of M; backups and holds must be separately included if omitted from M.
[4] NUMERICAL EXAMPLE: A=1,000/day, D=30 days, B=10,000 bytes/session, M=3 gives C=900,000,000 bytes.
[5] INTERPRETATION: Approximately 900 MB in decimal units is a planning estimate, not a measured disk footprint.
Boundary: event growth during an active session, churn, legal holds or backups require cohort-specific sums.
More generally sum arrival cohort times surviving fraction times mean cohort bytes, then apply the relevant storage multiplier.
For a single retained cohort with M=1, S2 reduces to S1 with N=A*D.

### 4.3 Key Special Cases / Variants
| Variant | Difference | Preconditions | Use when | Avoid when |
|---|---|---|---|---|
| Stateless request | No persistent interaction state | Request contains all needed inputs | Simple read-only task | Continuation or audit is required |
| Single-writer session | One authorized sequencer | Correct lease and fencing | Ordering matters | Receiver cannot enforce fencing |
| Optimistic versions | Concurrent contenders compare versions | Atomic conditional update | Conflicts are uncommon | Side effects happen before commit without safeguards |
| Serialized queue | One ordered partition per key | Broker and consumer preserve required order | Per-session ordering is essential | One hot session must scale without a plan |
| Append-only events | Append events and reconstruct projections | Event schema and replay semantics are maintained | Audit and recovery matter | Deletion and schema migration are not designed |
| Snapshot plus log | Replay suffix after snapshot | Snapshot and offset form a consistent boundary | Long histories make full replay costly | Snapshot can get ahead of durable events |

### 4.4 Asymptotic / Limiting Behaviour and Counterexample
If A or D is zero, S2 yields zero retained cohort payload; standing indexes or minimum provisioned capacity may still exist.
If D doubles while all other assumptions stay fixed, C doubles; this is not a prediction under nonstationary workloads.
If the external service executed an operation but the worker crashed before recording its receipt, a local version check cannot determine the external outcome.
Reconcile against the external system, using a stable effect key when supported; otherwise hold for manual resolution rather than assert exactly-once delivery.
A lease with no fencing lets an expired worker continue writing after a replacement starts.
A timeout is evidence that a response was not observed, not proof that the action did not happen.
**Takeaway:** Local atomicity is useful but does not cross an independent side-effect boundary.

## 5. Worked Numerical Example
**ILLUSTRATIVE / DERIVED:** A support-assistant platform needs a 30-day storage budget and safe order-submission recovery.
All numbers below are constructed inputs, not measured traffic or results.
Daily new sessions A: 1,000 sessions/day.
Retention D: 30 days, with no holds or delayed purge.
Mean recorded session payload B: 10,000 bytes/session.
Physical multiplier M: 3, explicitly assumed to include all replicas and index overhead but not backups.
Applying S2 yields 1,000 * 30 * 10,000 * 3 = 900,000,000 bytes.
In decimal units this is 900 MB; in binary units it is approximately 858.31 MiB.
Independent check: 30,000 sessions * 30,000 physical bytes/session = 900,000,000 bytes.
An unreplicated calculation of 300 MB would omit the assumed physical multiplier.
A 20% headroom policy applied to this estimate yields 1,080,000,000 bytes, not a guarantee of sufficient capacity.
If the mean grows to 15,000 bytes and all else is unchanged, the S2 result grows to 1,350,000,000 bytes before headroom.
Real sizing also needs active-session growth, percentile distributions, compression, backup copies, restore staging and deletion lag.
**ILLUSTRATIVE lifecycle trace:** Tenant T owns session S; principal P has permission to propose a purchase.
1. Admission binds T, P and S after checking membership; the client-supplied session ID is only a lookup key.
2. The control plane creates run R in `queued` with version 0 and policy version 12.
3. A worker claims R under lease generation 7; the claim and expiry are persisted.
4. R moves to `running` at version 1 through conditional update.
5. The worker assembles selected context, recording source references and version; retrieved text is untrusted.
6. The model proposes a purchase of item X for 40 currency units; this is a proposal, not an authorized order.
7. The policy requires approval; a canonical payload and hash H are persisted with R in `waiting_approval` at version 2.
8. An authorized approver approves H before expiry; identity, decision and expiration are durable.
9. Resume checks current membership, policy, hash, expiry and run version; an altered price requires a new approval.
10. The worker derives effect key K for this single intended order and durably records intent before sending.
11. The order API receives K and creates order O; its reply is lost before the worker records the receipt.
12. Worker lease 7 expires; a successor receives generation 8 and reconstructs the run from durable state.
13. The successor asks the external system for K; if a trustworthy lookup returns O, it records the receipt rather than submitting again.
14. If lookup is unavailable and the receiver offers no applicable deduplication contract, R enters `reconciliation_required` for human resolution.
15. With a confirmed receipt, R commits `completed` using the current version and generation; late generation 7 writes are rejected where fencing is supported.
The trace does not prove external exactly-once behavior; that depends on the actual order API contract and key lifetime.
A cancellation requested after step 11 cannot guarantee that the already-sent order was not created.
An independent operational check compares the local intent and external order ledger before closing any ambiguous case.
**Takeaway:** Sizing and lifecycle correctness depend on explicit scope, assumptions and the independent external system.

## 6. Visual / Geometric Intuition
```text
Time  ------------------------------------------------------------->
Worker A  claim(g=7) -> intent(K) -> send(K) -> timeout --- late write(g=7)
Database  g=7       -> intent saved             -> g=8 -> reject old generation
Order API                           -> order O created -> lookup(K) returns O
Worker B                                      claim(g=8) -> reconcile -> complete
```
Caption: schematic crash-window timeline, not to scale; g is a fencing generation and K an effect key.
The lost reply does not undo order O, and worker B must not equate missing local receipt with absent external effect.
The old worker can still contact a system that does not enforce fencing; a database fence is not a network kill switch.
```text
queued --> running --> waiting_approval --> running --> completed
   |          |                 |              |
   +----------+-----------------+--------------+--> cancel_requested
              |                                |          |
              +--> failed                      +--> reconciliation_required
```
Caption: illustrative application state machine; transition rules are design choices, not framework defaults.
Cancellation requires separate handling for a not-yet-sent effect and an effect with an unknown outcome.
**Takeaway:** Draw time and trust boundaries explicitly; a clean state diagram cannot erase a partial external failure.

## 7. Comparison / Selection Table
| Approach | Preconditions | Guarantee / boundary | Cost and use | Avoid when | Evidence |
|---|---|---|---|---|---|
| Stateless baseline | Inputs reproducible per request | No durable continuation | Small state burden; simple reads | Approval or cross-request history needed | DERIVED |
| Shared DB, tenant predicates | Enforced on every path | Logical separation depends on policy | Operationally simple | Untrusted admin access bypasses safeguards | DERIVED |
| Database row security | Correct database roles and policy | Database enforces applicable row policies | Central policy; test privileged roles | Bypass roles or missing tenant context | SOURCED [S2] |
| Separate tenant databases | Provisioning and routing per tenant | Stronger operational separation, not automatic network isolation | Higher management cost | Tenant count makes operations unaffordable | HEURISTIC |
| Isolated worker sandbox | Boundaries for process, filesystem and network | Contains only permissions actually enforced | Useful for untrusted code | Credentials or network remain broadly shared | DERIVED |
| Serialized session processing | Partitioned queue, bounded worker lease | Limits concurrent mutation per session | Predictable ordering; hot-key risk | Very high single-session write concurrency | DERIVED |

Selection priority:
1. Reject any option unable to enforce required tenant isolation, data residency or external-effect authorization.
2. Choose stateless processing for truly independent read-only requests; do not introduce a session for convenience alone.
3. Use tenant-scoped shared storage when its enforcement and operational access meet the threat model.
4. Escalate to separate databases or stronger runtime isolation when contractual risk or blast-radius requirements demand it.
5. Add serialized processing or atomic versioning when concurrent runs can mutate shared state.
6. Choose the simplest mechanism that satisfies correctness; benchmark realistic skew before scaling it.
7. If options conflict, security and irreversible-effect safety take priority over throughput; record owner and decision.
Comparable production performance is **NOT MEASURED**; benchmark the same workload, versions, hardware, concurrency, payload distribution and fault injection across options.
Measure unauthorized access attempts blocked, successful run rate, conflict rate, tail latency, cost per useful completion and recovery time; preserve sample counts and uncertainty.
**Takeaway:** Isolation is a threat-model decision first and an infrastructure-cost decision second.

## 8. Failure Modes & Edge Cases
The following cases are HYPOTHETICAL teaching diagnostics, not reported incidents.
Each includes a discriminating signal, bounded response and a verification target.
A negative signal rules out only the specific hypothesis tested; preserve evidence and escalate unresolved ambiguity.
The collection's final Common Issues artifact is planned, not yet written.

### F-01 — Cross-tenant session lookup
- **Trigger / symptom:** A user sees a different organization's conversation.
- **Candidate mechanism:** A lookup trusts session ID without tenant predicate.
- **Competing hypothesis:** A routing error can also return the wrong storage shard.
- **Discriminating diagnostic:** Replay the read with the same session ID under two isolated tenant identities.
- **Expected signal:** The unauthorized identity must get no record even if the identifier is known.
- **Immediate mitigation:** Block affected route and rotate exposed secrets if disclosure is confirmed.
- **Correction / prevention:** Enforce tenant binding in the query and authorization layer.
- **Verification:** Test reads writes exports and cache hits under both identities.
- **Boundary / escalation:** A passing application test does not validate privileged database roles; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-02 — Forged tenant header
- **Trigger / symptom:** Caller sets a tenant header for an organization they do not belong to.
- **Candidate mechanism:** Server treats an untrusted header as identity.
- **Competing hypothesis:** Misconfigured gateway identity propagation is another possibility.
- **Discriminating diagnostic:** Compare signed principal claims with the resolved tenant at admission.
- **Expected signal:** The requested tenant must be a verified membership or explicitly delegated scope.
- **Immediate mitigation:** Reject the request and preserve the gateway trace.
- **Correction / prevention:** Derive tenant context from trusted authentication and validate membership.
- **Verification:** Exercise direct backend access as well as gateway access.
- **Boundary / escalation:** A signed token can still have stale membership; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-03 — Shared cache key collision
- **Trigger / symptom:** A cached answer appears in the wrong tenant.
- **Candidate mechanism:** Cache key omits tenant or authorization scope.
- **Competing hypothesis:** Incorrect cache invalidation can mimic this symptom.
- **Discriminating diagnostic:** Inspect exact key construction and source tenant of the cached value.
- **Expected signal:** Different tenant scopes must never resolve to the same private cache entry.
- **Immediate mitigation:** Disable or partition the affected cache.
- **Correction / prevention:** Include tenant authorization and relevant resource version in the cache key.
- **Verification:** Prime as one tenant then fetch as another under test identities.
- **Boundary / escalation:** Public immutable content can have different sharing rules; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-04 — Object storage path traversal
- **Trigger / symptom:** An export request reads another tenant's object.
- **Candidate mechanism:** Client-controlled path escapes its permitted prefix.
- **Competing hypothesis:** A stale signed URL may also grant access.
- **Discriminating diagnostic:** Resolve the canonical object key and compare its authorized resource binding.
- **Expected signal:** A rejected traversal should never reach the object read.
- **Immediate mitigation:** Disable the vulnerable export route and revoke affected URLs.
- **Correction / prevention:** Construct object paths server-side and enforce object-store permissions.
- **Verification:** Test encoded separators and cross-tenant object IDs.
- **Boundary / escalation:** A prefix string check alone may not handle normalization; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-05 — Unauthorized resume
- **Trigger / symptom:** A previously authorized user resumes after access was revoked.
- **Candidate mechanism:** Resume trusts the original session authorization indefinitely.
- **Competing hypothesis:** A delayed identity synchronization can produce similar behavior.
- **Discriminating diagnostic:** Compare current membership and approval policy with the original admission.
- **Expected signal:** The resumed action must fail if current authorization is absent.
- **Immediate mitigation:** Pause affected runs and revoke outstanding capabilities.
- **Correction / prevention:** Reauthorize on resume and before privileged tool use.
- **Verification:** Revoke membership between pause and resume in a controlled test.
- **Boundary / escalation:** Historical records remain subject to their own access policy; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-06 — Approval payload substitution
- **Trigger / symptom:** The user approves one payment but the worker submits another.
- **Candidate mechanism:** Approval is attached to a run rather than an immutable payload.
- **Competing hypothesis:** A serialization difference can produce false hash mismatches.
- **Discriminating diagnostic:** Compare canonical approved fields with the actual outgoing request.
- **Expected signal:** Any material change must invalidate the approval.
- **Immediate mitigation:** Stop the send and request approval for the changed payload.
- **Correction / prevention:** Bind approval to canonical action tenant destination amount and expiration.
- **Verification:** Change one protected field and confirm resume is rejected.
- **Boundary / escalation:** Canonicalization must include all semantically relevant fields; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-07 — Expired approval
- **Trigger / symptom:** A delayed run executes after its approval window.
- **Candidate mechanism:** Resume checks decision but not expiry.
- **Competing hypothesis:** Clock skew can create ambiguous boundary cases.
- **Discriminating diagnostic:** Compare trusted server time with the stored expiry and clock health.
- **Expected signal:** An expired approval cannot authorize a fresh effect.
- **Immediate mitigation:** Pause the action and obtain a new decision.
- **Correction / prevention:** Validate expiry at the effect boundary using trusted time.
- **Verification:** Resume just before and after expiration under controlled clock conditions.
- **Boundary / escalation:** Time validity does not replace current principal authorization; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-08 — Duplicate purchase after timeout
- **Trigger / symptom:** Two orders appear after an API timeout.
- **Candidate mechanism:** Retry creates a fresh effect key or receiver does not deduplicate.
- **Competing hypothesis:** A user may have intentionally submitted two separate requests.
- **Discriminating diagnostic:** Compare intent IDs effect keys external receipts and payloads.
- **Expected signal:** One intended effect should map to one durable effect key across attempts.
- **Immediate mitigation:** Stop automatic retries and reconcile all matching external records.
- **Correction / prevention:** Persist the key before send and use the provider's documented deduplication contract.
- **Verification:** Inject lost responses and compare order ledger to intent ledger.
- **Boundary / escalation:** A key outside the receiver's retention window may no longer deduplicate; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-09 — Idempotency key collision
- **Trigger / symptom:** An unrelated request receives an old result.
- **Candidate mechanism:** The key scope is too broad or reused for distinct intent.
- **Competing hypothesis:** An external API may scope keys differently than the client assumes.
- **Discriminating diagnostic:** Compare tenant operation destination key and payload hash.
- **Expected signal:** Distinct intended actions must not share a key within receiver scope.
- **Immediate mitigation:** Suspend affected key generation and inspect returned receipts.
- **Correction / prevention:** Use high-entropy keys with explicit tenant and operation scope.
- **Verification:** Send two different payloads with deliberately repeated key in a sandbox.
- **Boundary / escalation:** Never assume a provider rejects mismatched payloads without documentation; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-10 — Unknown send outcome
- **Trigger / symptom:** Run is stuck after the remote API times out.
- **Candidate mechanism:** The remote action may have completed despite missing response.
- **Competing hypothesis:** The request may also never have left the worker.
- **Discriminating diagnostic:** Inspect client trace and independently query the external ledger by effect key.
- **Expected signal:** A timeout alone must not be interpreted as a failed effect.
- **Immediate mitigation:** Hold the run in reconciliation-required state.
- **Correction / prevention:** Build a lookup or operator review path before enabling the action.
- **Verification:** Simulate lost response after remote commit and observe resolution.
- **Boundary / escalation:** A missing lookup result may itself be eventually consistent; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-11 — Concurrent session updates
- **Trigger / symptom:** One user's latest message disappears.
- **Candidate mechanism:** Two runs write full session snapshots from the same version.
- **Competing hypothesis:** Incorrect projection rebuild can look similar.
- **Discriminating diagnostic:** Inspect expected and committed versions for competing updates.
- **Expected signal:** The second stale update must conflict instead of overwriting.
- **Immediate mitigation:** Stop unsafe writers and reconstruct from durable records where possible.
- **Correction / prevention:** Use atomic compare-and-set or per-session serialization.
- **Verification:** Race two writes from the same version and confirm one conflict.
- **Boundary / escalation:** A correct local update does not serialize an external system; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-12 — Stale lease holder
- **Trigger / symptom:** A former worker changes run state after reassignment.
- **Candidate mechanism:** Lease expiry was treated as process termination.
- **Competing hypothesis:** A misconfigured clock can trigger early reassignment.
- **Discriminating diagnostic:** Compare writer generation with the current resource generation.
- **Expected signal:** The receiver must reject an older generation even if worker A is alive.
- **Immediate mitigation:** Disable the stale worker credentials and contain writes.
- **Correction / prevention:** Use monotonically increasing fencing tokens at each protected write boundary.
- **Verification:** Partition worker A then reassign and attempt its old write.
- **Boundary / escalation:** A third-party tool that ignores the token remains unfenced; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-13 — Split-brain scheduler
- **Trigger / symptom:** Two workers both believe they own one run.
- **Candidate mechanism:** Leadership or ownership changes lack a consistent authority.
- **Competing hypothesis:** Duplicate queue delivery alone is also plausible.
- **Discriminating diagnostic:** Compare claim transaction IDs and ownership generations.
- **Expected signal:** Only one generation should be accepted for a protected transition.
- **Immediate mitigation:** Pause dispatch and preserve lease records.
- **Correction / prevention:** Make ownership assignment atomic and fence all subsequent writes.
- **Verification:** Exercise failover during an active run.
- **Boundary / escalation:** A lease reduces concurrent work but cannot guarantee one execution; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-14 — Checkpoint before durable event
- **Trigger / symptom:** Resume skips a tool call whose intent was never committed.
- **Candidate mechanism:** Snapshot offset advances before its referenced events are durable.
- **Competing hypothesis:** A projection bug may make an existing event invisible.
- **Discriminating diagnostic:** Read snapshot offset and confirm all referenced events exist durably.
- **Expected signal:** Every event at or before the snapshot boundary must be recoverable.
- **Immediate mitigation:** Stop advancing snapshots and recover from the last consistent point.
- **Correction / prevention:** Commit snapshot and event position atomically or use a safe protocol.
- **Verification:** Crash between each persistence step during a restore exercise.
- **Boundary / escalation:** A snapshot is not automatically consistent merely because it is recent; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-15 — Replay repeats an effect
- **Trigger / symptom:** A restored run submits a tool request again.
- **Candidate mechanism:** Replay executes side effects instead of consuming recorded receipts.
- **Competing hypothesis:** An earlier intent may genuinely have no external outcome.
- **Discriminating diagnostic:** Compare replay mode and the stored intent receipt pair.
- **Expected signal:** A confirmed receipt must suppress re-execution for the same intent.
- **Immediate mitigation:** Disable effectful replay and reconcile the external system.
- **Correction / prevention:** Separate deterministic reconstruction from live tool execution.
- **Verification:** Restore a run after an effect and check that no second call occurs.
- **Boundary / escalation:** Unconfirmed effects require reconciliation rather than silent suppression; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-16 — Poison event blocks recovery
- **Trigger / symptom:** Every resume fails at the same history position.
- **Candidate mechanism:** The event schema is malformed or no longer supported.
- **Competing hypothesis:** Storage corruption can produce a similar parse error.
- **Discriminating diagnostic:** Inspect event version bytes checksum and decoder error.
- **Expected signal:** A supported historical event must decode under the declared migration policy.
- **Immediate mitigation:** Quarantine the run without discarding history.
- **Correction / prevention:** Version event schemas and provide tested upcasters or migrations.
- **Verification:** Replay representative historical fixtures with the new reader.
- **Boundary / escalation:** Do not mutate immutable audit evidence just to make a parser pass; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-17 — Out-of-order tool results
- **Trigger / symptom:** The transcript attaches a result to the wrong request.
- **Candidate mechanism:** Responses are matched by arrival order instead of call ID.
- **Competing hypothesis:** The tool itself may emit an incorrect correlation ID.
- **Discriminating diagnostic:** Compare request and response correlation IDs and timestamps.
- **Expected signal:** Each result must bind to the original invocation ID.
- **Immediate mitigation:** Pause downstream decisions and recover the correct mapping.
- **Correction / prevention:** Persist unique call IDs and validate response correlation.
- **Verification:** Delay and reorder responses in a sandbox.
- **Boundary / escalation:** Timestamp ordering alone cannot establish a causal match; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-18 — Duplicated queue delivery
- **Trigger / symptom:** A run starts twice after broker redelivery.
- **Candidate mechanism:** Consumer treats message delivery as unique execution.
- **Competing hypothesis:** Two genuinely separate submissions can have different request IDs.
- **Discriminating diagnostic:** Compare run ID message ID and attempt lineage.
- **Expected signal:** Duplicate delivery should not create a second run for the same admission key.
- **Immediate mitigation:** Stop duplicate consumers if effects are at risk.
- **Correction / prevention:** Make run admission and step transitions idempotent.
- **Verification:** Redeliver the identical message and check persisted state.
- **Boundary / escalation:** At-least-once delivery does not imply one execution; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-19 — Unbounded retry storm
- **Trigger / symptom:** Queue depth and upstream errors rise together.
- **Candidate mechanism:** Every failure triggers immediate synchronized retries.
- **Competing hypothesis:** A real arrival spike can produce similar queue growth.
- **Discriminating diagnostic:** Measure retry attempt counts error classes and backoff distribution.
- **Expected signal:** Retry traffic should be separately visible from original requests.
- **Immediate mitigation:** Apply admission limits and stop unsafe retries.
- **Correction / prevention:** Use capped exponential backoff with jitter and a deadline.
- **Verification:** Inject an outage and verify attempts remain within configured budgets.
- **Boundary / escalation:** Backoff is not a remedy for permanent authorization failures; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-20 — Permanent error retried
- **Trigger / symptom:** A run repeatedly calls a tool with invalid credentials.
- **Candidate mechanism:** All errors are classified as transient.
- **Competing hypothesis:** A temporary credential service outage may look similar.
- **Discriminating diagnostic:** Inspect error code provider contract and credential validity.
- **Expected signal:** A permanent denial should terminate or request intervention.
- **Immediate mitigation:** Pause the run and repair credentials or permission.
- **Correction / prevention:** Classify retryable errors by documented tool semantics.
- **Verification:** Inject a permission denial and verify no blind repeated calls.
- **Boundary / escalation:** Error classifications can change with provider behavior; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-21 — Cancellation race
- **Trigger / symptom:** A cancelled run still causes an external effect.
- **Candidate mechanism:** Cancellation arrived after dispatch or was not checked at the send boundary.
- **Competing hypothesis:** The external service may have already accepted the action.
- **Discriminating diagnostic:** Compare cancellation timestamp intent send and remote commit.
- **Expected signal:** A pre-send cancellation should prevent a new dispatch under the designed protocol.
- **Immediate mitigation:** Stop future steps and reconcile any in-flight action.
- **Correction / prevention:** Define cancellation checkpoints and explicit post-send semantics.
- **Verification:** Cancel before dispatch during dispatch and after remote commit.
- **Boundary / escalation:** Cancellation cannot retroactively guarantee reversal; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-22 — Cancellation marked complete too early
- **Trigger / symptom:** UI reports cancelled while a payment is still processing.
- **Candidate mechanism:** Terminal status ignores unresolved in-flight actions.
- **Competing hypothesis:** A delayed status projection can also mislead.
- **Discriminating diagnostic:** Inspect in-flight effect intents and cancellation state.
- **Expected signal:** Terminal cancellation requires a defined disposition for each outstanding effect.
- **Immediate mitigation:** Expose pending reconciliation instead of a false terminal result.
- **Correction / prevention:** Separate cancel-requested from cancelled and reconciliation-required.
- **Verification:** Inject a slow external response while cancelling.
- **Boundary / escalation:** Some external systems provide no reliable cancellation mechanism; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-23 — Infinite tool loop
- **Trigger / symptom:** Run consumes budget without useful progress.
- **Candidate mechanism:** Model repeatedly proposes the same unsuccessful action.
- **Competing hypothesis:** A long but legitimate task may have many distinct steps.
- **Discriminating diagnostic:** Inspect repeated normalized calls and remaining budget.
- **Expected signal:** Repeated identical failures without state change indicate no progress.
- **Immediate mitigation:** Stop the run and surface a bounded failure.
- **Correction / prevention:** Set step token time and cost budgets plus loop detection.
- **Verification:** Feed a repeating tool response and confirm termination.
- **Boundary / escalation:** A repetition detector may need exceptions for legitimate polling; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-24 — Context truncation loses approval
- **Trigger / symptom:** The model claims approval exists but the control plane has none.
- **Candidate mechanism:** Important history is missing or model inferred authorization.
- **Competing hypothesis:** A genuine approval may exist in an independent record.
- **Discriminating diagnostic:** Query the approval store by run and payload hash.
- **Expected signal:** Only a valid durable approval can authorize the action.
- **Immediate mitigation:** Block the tool call and request verification.
- **Correction / prevention:** Keep approval state outside model context and check it at the tool gate.
- **Verification:** Remove the approval message from context and confirm enforcement remains correct.
- **Boundary / escalation:** Adding more prompt tokens does not establish authority; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-25 — Prompt injection via retrieval
- **Trigger / symptom:** Retrieved document tells the agent to disclose another tenant's data.
- **Candidate mechanism:** Untrusted source text is promoted into instructions or tool authority.
- **Competing hypothesis:** An ordinary quoted instruction may be harmless if not acted upon.
- **Discriminating diagnostic:** Trace source provenance model proposal and enforcement decision.
- **Expected signal:** A tool action outside authorized scope must be rejected regardless of text.
- **Immediate mitigation:** Disable the affected tool path and assess exposure.
- **Correction / prevention:** Treat retrieved text as data and enforce permissions outside the model.
- **Verification:** Place a malicious instruction in a test document and verify denial.
- **Boundary / escalation:** Prompt wording alone is not a complete security boundary; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-26 — Tool output injection
- **Trigger / symptom:** A tool response instructs the agent to send secrets elsewhere.
- **Candidate mechanism:** External tool text is treated as a trusted system directive.
- **Competing hypothesis:** A compromised tool endpoint may also fabricate structured fields.
- **Discriminating diagnostic:** Inspect raw tool response provenance and subsequent tool calls.
- **Expected signal:** The downstream authorization layer must deny unapproved exfiltration.
- **Immediate mitigation:** Revoke exposed credentials and quarantine the tool response.
- **Correction / prevention:** Constrain outbound destinations and validate structured tool outputs.
- **Verification:** Return hostile content from a sandbox tool and observe denied action.
- **Boundary / escalation:** Output validation cannot prove arbitrary natural language is safe; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-27 — Overbroad service identity
- **Trigger / symptom:** Every run can access all tenant records.
- **Candidate mechanism:** Worker uses a shared administrator credential.
- **Competing hypothesis:** An access proxy may nevertheless enforce per-request authorization.
- **Discriminating diagnostic:** Inspect effective credentials and resource-level authorization logs.
- **Expected signal:** A tenant-scoped run should lack unrelated tenant privileges.
- **Immediate mitigation:** Revoke the broad credential and pause sensitive actions.
- **Correction / prevention:** Issue short-lived scoped capabilities and apply least privilege.
- **Verification:** Attempt cross-tenant access from a worker identity.
- **Boundary / escalation:** A scoped token still requires secure issuance and revocation; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-28 — Secret in transcript
- **Trigger / symptom:** An API key appears in persisted conversation history.
- **Candidate mechanism:** Raw tool arguments or output are logged without redaction.
- **Competing hypothesis:** A user may have intentionally pasted the secret.
- **Discriminating diagnostic:** Trace secret origin through ingress logs tool events and exports.
- **Expected signal:** The original secret should be considered compromised once exposed.
- **Immediate mitigation:** Revoke the credential and restrict access to affected records.
- **Correction / prevention:** Use secret references and minimize capture before persistence.
- **Verification:** Inject a test credential and search authorized storage surfaces.
- **Boundary / escalation:** Redaction does not erase immutable backups already containing a secret; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-29 — Stale policy at resume
- **Trigger / symptom:** A run uses an old permission after policy was tightened.
- **Candidate mechanism:** Policy was captured at creation and never revalidated.
- **Competing hypothesis:** A policy cache propagation delay may also cause it.
- **Discriminating diagnostic:** Compare effective policy versions at admission resume and effect dispatch.
- **Expected signal:** The defined current-policy rule must be enforced at the action boundary.
- **Immediate mitigation:** Pause privileged runs until policy propagation is confirmed.
- **Correction / prevention:** Record policy version and explicitly reauthorize on resume.
- **Verification:** Change policy while a run waits then attempt its action.
- **Boundary / escalation:** Policy migration semantics must be decided for already-approved work; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-30 — Lost audit event
- **Trigger / symptom:** A state change is visible without a corresponding audit record.
- **Candidate mechanism:** State commit and audit publication are independent writes.
- **Competing hypothesis:** A delayed consumer might make a valid event temporarily invisible.
- **Discriminating diagnostic:** Compare database transaction records with audit stream offsets.
- **Expected signal:** Committed transitions should have a durable corresponding audit entry.
- **Immediate mitigation:** Stop high-risk transitions if audit is mandatory.
- **Correction / prevention:** Persist audit event transactionally or use an outbox in the same transaction.
- **Verification:** Crash after the state write and verify eventual audit publication.
- **Boundary / escalation:** An outbox gives reliable publication only with correct consumer handling; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-31 — Audit log tampering
- **Trigger / symptom:** An operator can silently alter recorded approvals.
- **Candidate mechanism:** The same broad identity can modify operational and audit records.
- **Competing hypothesis:** A legitimate retention job may also change availability.
- **Discriminating diagnostic:** Compare access logs immutable storage controls and approved deletion records.
- **Expected signal:** Unauthorized alteration should be prevented or detectable.
- **Immediate mitigation:** Preserve evidence and restrict administrative access.
- **Correction / prevention:** Separate duties and use tamper-evident retention controls.
- **Verification:** Attempt unauthorized modification with a test operator role.
- **Boundary / escalation:** Tamper evidence does not prove the original event was truthful; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-32 — Tenant deletion incomplete
- **Trigger / symptom:** Deleted user data remains searchable.
- **Candidate mechanism:** A deletion flow misses derived indexes caches exports or backups.
- **Competing hypothesis:** A valid legal hold may explain retention.
- **Discriminating diagnostic:** Trace the data inventory and deletion receipts by resource ID.
- **Expected signal:** All in-scope copies should follow the documented deletion or hold policy.
- **Immediate mitigation:** Disable access and begin tracked remediation.
- **Correction / prevention:** Maintain a deletion graph including derived stores and backup expiry.
- **Verification:** Create a marked test record delete it and inspect all documented copies.
- **Boundary / escalation:** Backup restoration must not reintroduce deleted data without controls; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-33 — Cross-region state leakage
- **Trigger / symptom:** Session data appears outside its required region.
- **Candidate mechanism:** A global cache or failover path ignores residency routing.
- **Competing hypothesis:** Replicated control metadata may be mistaken for content.
- **Discriminating diagnostic:** Classify the actual data copied and inspect region routing.
- **Expected signal:** Restricted content must stay in approved locations under the applicable policy.
- **Immediate mitigation:** Disable the affected route and assess contractual impact.
- **Correction / prevention:** Enforce region-aware storage cache logging and failover rules.
- **Verification:** Run region-failure exercises with tagged synthetic records.
- **Boundary / escalation:** Metadata may also be regulated depending on context; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-34 — Hot session partition
- **Trigger / symptom:** One session's runs queue while the rest remain healthy.
- **Candidate mechanism:** All its events serialize on one hot key.
- **Competing hypothesis:** A single slow downstream tool can cause the same shape.
- **Discriminating diagnostic:** Compare per-key queue depth processing time and tool latency.
- **Expected signal:** A concentrated queue with normal downstream latency supports hot-key contention.
- **Immediate mitigation:** Throttle the session and preserve fairness for other tenants.
- **Correction / prevention:** Bound per-session concurrency and redesign independent work where safe.
- **Verification:** Load-test skewed keys with the actual serialization rule.
- **Boundary / escalation:** Splitting a key can violate required ordering; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-35 — Unbounded transcript growth
- **Trigger / symptom:** Context assembly and storage latency climb throughout a long session.
- **Candidate mechanism:** Every turn rereads the entire history.
- **Competing hypothesis:** A slow database or retrieval service may be responsible.
- **Discriminating diagnostic:** Plot bytes read per invocation against session event count.
- **Expected signal:** Linear growth in bytes per call supports full-history reread.
- **Immediate mitigation:** Cap request size and pause oversized sessions safely.
- **Correction / prevention:** Use bounded context selection and consistent snapshots with explicit provenance.
- **Verification:** Replay a growing synthetic history and measure read bytes and latency.
- **Boundary / escalation:** Summaries can omit facts; retain authoritative records separately; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-36 — Stale summary
- **Trigger / symptom:** The agent acts on a summary that omits a recent correction.
- **Candidate mechanism:** Summary is not invalidated after a new event.
- **Competing hypothesis:** A bad summarization model can create the same symptom.
- **Discriminating diagnostic:** Compare summary coverage offset with the latest relevant event.
- **Expected signal:** The summary must not claim coverage past its actual source offset.
- **Immediate mitigation:** Rebuild the context from authoritative events for the affected run.
- **Correction / prevention:** Version summaries and record their source event ranges.
- **Verification:** Append a correction after summarization and confirm it is visible.
- **Boundary / escalation:** A fresh summary can still be semantically inaccurate; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-37 — Checkpoint contains secrets
- **Trigger / symptom:** A recovery snapshot exposes tool credentials to broad readers.
- **Candidate mechanism:** Runtime environment or token values were serialized wholesale.
- **Competing hypothesis:** A secret may instead originate from a user message.
- **Discriminating diagnostic:** Inspect snapshot schema and the credential access path.
- **Expected signal:** Recovery state should hold references rather than reusable secrets.
- **Immediate mitigation:** Revoke exposed credentials and restrict snapshot access.
- **Correction / prevention:** Whitelist checkpoint fields and encrypt sensitive state.
- **Verification:** Create a synthetic secret and inspect restored snapshot access.
- **Boundary / escalation:** Encryption at rest does not restrict authorized overly broad readers; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-38 — Misrouted tool response
- **Trigger / symptom:** One run receives another run's tool result.
- **Candidate mechanism:** Correlation keys omit tenant or run identity.
- **Competing hypothesis:** A shared tool adapter may return a wrong request identifier.
- **Discriminating diagnostic:** Trace tenant run call ID and adapter response mapping.
- **Expected signal:** A result must match its originating authorized invocation.
- **Immediate mitigation:** Quarantine both runs and stop affected adapter traffic.
- **Correction / prevention:** Bind responses to tenant run and call ID with authenticated transport.
- **Verification:** Interleave responses from two tenants in a sandbox.
- **Boundary / escalation:** A globally unique ID reduces collision risk but does not authorize access; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-39 — Orphaned waiting approval
- **Trigger / symptom:** A run remains paused long after the decision window.
- **Candidate mechanism:** No expiry worker or resume trigger handles the waiting state.
- **Competing hypothesis:** A human reviewer may simply be unavailable.
- **Discriminating diagnostic:** Compare approval expiry queue status and trigger delivery.
- **Expected signal:** Expired work should follow a defined timeout transition.
- **Immediate mitigation:** Escalate the pending item and stop any unsafe dispatch.
- **Correction / prevention:** Persist deadlines and run a durable expiry mechanism.
- **Verification:** Disable and restart the scheduler across the deadline.
- **Boundary / escalation:** Expiry handling must be idempotent under duplicate triggers; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-40 — Duplicate approval webhook
- **Trigger / symptom:** Two identical approvals produce two resumed attempts.
- **Candidate mechanism:** Webhook delivery is treated as a unique event every time.
- **Competing hypothesis:** Two separate approval decisions may be legitimately distinct.
- **Discriminating diagnostic:** Compare event ID approval ID payload hash and run version.
- **Expected signal:** Only one valid transition should consume a decision for one run version.
- **Immediate mitigation:** Pause effectful steps and reconcile attempts.
- **Correction / prevention:** Deduplicate delivery and use an atomic state transition.
- **Verification:** Deliver the same signed webhook twice and inspect transitions.
- **Boundary / escalation:** Signature validation alone does not prevent replay; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-41 — Approval from wrong actor
- **Trigger / symptom:** A valid-looking approval comes from an unauthorized principal.
- **Candidate mechanism:** The system verifies webhook origin but not approver entitlement.
- **Competing hypothesis:** A stale group membership cache may cause the same failure.
- **Discriminating diagnostic:** Check actor identity membership at decision time and delegation policy.
- **Expected signal:** The actor must possess authority over this action and tenant.
- **Immediate mitigation:** Suspend the approval and seek an authorized decision.
- **Correction / prevention:** Authorize the approver independently of the requester.
- **Verification:** Submit the same decision from authorized and unauthorized test identities.
- **Boundary / escalation:** Separation of duties may additionally prohibit self-approval; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-42 — Stale tool schema
- **Trigger / symptom:** A resumed run sends parameters incompatible with the current adapter.
- **Candidate mechanism:** A stored plan refers to an older tool version.
- **Competing hypothesis:** A malformed model response can also fail validation.
- **Discriminating diagnostic:** Compare stored tool schema version with adapter version and validation error.
- **Expected signal:** Unsupported stored requests should not be silently reinterpreted.
- **Immediate mitigation:** Pause and require migration or a fresh proposal.
- **Correction / prevention:** Version tool contracts and define compatibility on resume.
- **Verification:** Resume a historical checkpoint against the new adapter in staging.
- **Boundary / escalation:** Passing JSON validation does not guarantee equivalent side effects; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-43 — Credential expiry mid-run
- **Trigger / symptom:** A tool call fails after a long approval wait.
- **Candidate mechanism:** A short-lived credential was persisted instead of refreshed.
- **Competing hypothesis:** Access may have been intentionally revoked.
- **Discriminating diagnostic:** Inspect token expiry and current authorization independently.
- **Expected signal:** A refreshed token must still satisfy present policy.
- **Immediate mitigation:** Pause the call and obtain a current scoped credential.
- **Correction / prevention:** Keep credential references out of checkpoints and reissue on resume.
- **Verification:** Wait past token expiry in a controlled scenario.
- **Boundary / escalation:** Refreshing credentials must not restore revoked privileges; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-44 — Partial multi-tool workflow
- **Trigger / symptom:** One system updates while the second update fails.
- **Candidate mechanism:** Independent systems do not share a transaction.
- **Competing hypothesis:** A retry may also have created duplicate first-system effects.
- **Discriminating diagnostic:** Compare receipts and current state in both systems.
- **Expected signal:** The run must expose partial completion rather than claim atomic success.
- **Immediate mitigation:** Reconcile and apply a preapproved compensation where safe.
- **Correction / prevention:** Design explicit saga steps compensation and manual escalation.
- **Verification:** Fail the second action after the first commits.
- **Boundary / escalation:** Compensation may itself fail or have irreversible consequences; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-45 — Incorrect compensation
- **Trigger / symptom:** A rollback creates a second business error.
- **Candidate mechanism:** Compensating action assumes no intervening external changes.
- **Competing hypothesis:** The first action might not have committed at all.
- **Discriminating diagnostic:** Read current external state and version before compensation.
- **Expected signal:** Compensation requires a confirmed effect and valid preconditions.
- **Immediate mitigation:** Pause automated reversal and involve the business owner.
- **Correction / prevention:** Model compensation as a separately authorized idempotent action.
- **Verification:** Change the external state before triggering compensation.
- **Boundary / escalation:** A compensating action is not a database rollback; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-46 — Model output treated as final state
- **Trigger / symptom:** A dashboard marks work complete based only on model text.
- **Candidate mechanism:** Narrative output is mistaken for a committed tool receipt.
- **Competing hypothesis:** The tool may have completed but receipt capture failed.
- **Discriminating diagnostic:** Compare the claimed outcome against authoritative system state.
- **Expected signal:** Completion requires the specified durable evidence.
- **Immediate mitigation:** Mark the result unverified and reconcile.
- **Correction / prevention:** Derive completion from receipts and state transitions not prose.
- **Verification:** Return a fabricated success sentence without a tool call.
- **Boundary / escalation:** A valid receipt can still require later business settlement; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-47 — Inconsistent run terminal state
- **Trigger / symptom:** A completed run still has unresolved effects.
- **Candidate mechanism:** Terminal transition ignores pending intent records.
- **Competing hypothesis:** A slow projection may display older data.
- **Discriminating diagnostic:** Query the authoritative run state and all intent dispositions.
- **Expected signal:** No unresolved effect should be hidden by a success status.
- **Immediate mitigation:** Reopen or flag the case for reconciliation according to policy.
- **Correction / prevention:** Gate terminal completion on defined effect disposition.
- **Verification:** Inject an unresolved intent before terminal transition.
- **Boundary / escalation:** Some workflows may intentionally complete with explicit partial results; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-48 — Backpressure failure
- **Trigger / symptom:** Admission continues while workers cannot drain the queue.
- **Candidate mechanism:** Queue capacity and downstream saturation are not tied to admission.
- **Competing hypothesis:** A monitoring delay may overstate backlog.
- **Discriminating diagnostic:** Compare arrival rate service rate queue age and admission decisions.
- **Expected signal:** Persistent arrivals above service capacity must grow backlog absent shedding.
- **Immediate mitigation:** Throttle new work and prioritize safe recovery.
- **Correction / prevention:** Use quotas bounded queues and overload responses.
- **Verification:** Sustain an illustrative overload then verify admission control.
- **Boundary / escalation:** Storage capacity alone cannot fix a sustained throughput deficit; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-49 — Duplicate user submission
- **Trigger / symptom:** A network retry creates two runs for the same request.
- **Candidate mechanism:** Admission has no client request idempotency contract.
- **Competing hypothesis:** The user may intend two distinct requests.
- **Discriminating diagnostic:** Compare request keys principal payload hashes and timestamps.
- **Expected signal:** Same intended admission should resolve to one run within the declared scope.
- **Immediate mitigation:** Stop duplicates before side effects and ask for clarification if ambiguous.
- **Correction / prevention:** Persist a scoped admission key with payload binding.
- **Verification:** Retry an identical admission after losing the first response.
- **Boundary / escalation:** Never merge distinct requests solely because their text matches; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-50 — Long-lived signed URL
- **Trigger / symptom:** A revoked user retains access to an exported transcript.
- **Candidate mechanism:** A bearer URL outlives user membership or intended access.
- **Competing hypothesis:** The export may have been copied before revocation.
- **Discriminating diagnostic:** Check URL expiry audience restrictions and object access logs.
- **Expected signal:** Revocation expectations must match the actual URL capabilities.
- **Immediate mitigation:** Revoke or delete the object where feasible.
- **Correction / prevention:** Use short expiry and authenticated download for sensitive exports.
- **Verification:** Revoke a user and test previously issued download links.
- **Boundary / escalation:** A downloaded copy cannot be recalled by URL revocation; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-51 — Unexpected tool destination
- **Trigger / symptom:** A model proposes sending private data to an attacker-controlled endpoint.
- **Candidate mechanism:** Tool allows arbitrary network destinations.
- **Competing hypothesis:** A legitimate integration may have a newly changed endpoint.
- **Discriminating diagnostic:** Compare destination against an independently managed allowlist and action approval.
- **Expected signal:** Unapproved egress must be denied even when the model requests it.
- **Immediate mitigation:** Block egress and investigate attempted disclosure.
- **Correction / prevention:** Use destination allowlists scoped credentials and network policy.
- **Verification:** Attempt a forbidden destination from the execution sandbox.
- **Boundary / escalation:** Allowlisted services can themselves be compromised; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-52 — Data source ownership conflict
- **Trigger / symptom:** Two services disagree about the current order status.
- **Candidate mechanism:** A session cache is mistaken for the order system of record.
- **Competing hypothesis:** The external system may be eventually consistent.
- **Discriminating diagnostic:** Compare source version freshness and authoritative ownership contract.
- **Expected signal:** The designated authority wins subject to documented consistency.
- **Immediate mitigation:** Avoid irreversible action until the discrepancy is resolved.
- **Correction / prevention:** Document ownership freshness and conflict handling per field.
- **Verification:** Change the source record and inspect the session projection.
- **Boundary / escalation:** An authoritative source can still contain erroneous business data; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-53 — Metrics hide tenant skew
- **Trigger / symptom:** Aggregate success rate looks healthy while one tenant fails.
- **Candidate mechanism:** Aggregated counts conceal a small affected cohort.
- **Competing hypothesis:** A tenant may have a legitimately different workload.
- **Discriminating diagnostic:** Break down outcomes by tenant cohort and error class with access controls.
- **Expected signal:** Compare like workloads and sample sizes before concluding a regression.
- **Immediate mitigation:** Support the affected tenant and preserve scoped traces.
- **Correction / prevention:** Track fairness and isolation-safe cohort diagnostics.
- **Verification:** Inject a failure for one synthetic tenant and check the alert.
- **Boundary / escalation:** Per-tenant labels may create high cardinality or privacy exposure; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-54 — Unsafe retry after key expiry
- **Trigger / symptom:** A delayed run repeats a previously successful external action.
- **Candidate mechanism:** The receiver has forgotten the old idempotency key.
- **Competing hypothesis:** A new intended transaction may legitimately reuse the same payload.
- **Discriminating diagnostic:** Compare original send time provider key retention and external ledger.
- **Expected signal:** Expired deduplication protection cannot establish absence of the old effect.
- **Immediate mitigation:** Hold for reconciliation instead of automatically retrying.
- **Correction / prevention:** Set retry deadline below the documented deduplication horizon.
- **Verification:** Resume beyond the horizon in a sandbox and verify safe handling.
- **Boundary / escalation:** Provider key retention and lookup guarantees must be verified; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-55 — Version counter reset
- **Trigger / symptom:** An old worker's generation becomes valid after resource recreation.
- **Candidate mechanism:** A new resource reuses the same identifier and starts generation from zero.
- **Competing hypothesis:** A stale cache can also report an old generation.
- **Discriminating diagnostic:** Compare resource incarnation and generation in all writes.
- **Expected signal:** A generation is meaningful only within a unique resource incarnation.
- **Immediate mitigation:** Disable stale workers and isolate the recreated resource.
- **Correction / prevention:** Include an immutable incarnation ID in fencing checks.
- **Verification:** Delete and recreate a test run then replay an old write.
- **Boundary / escalation:** Monotonic counters alone do not protect across reset without incarnation scope; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-56 — Failure hidden by retry
- **Trigger / symptom:** A dashboard shows eventual success while many attempts fail.
- **Candidate mechanism:** Metrics count only terminal outcomes.
- **Competing hypothesis:** The model or tool may be degrading without total failure.
- **Discriminating diagnostic:** Compare per-attempt errors and latency against per-run outcomes.
- **Expected signal:** Rising attempts per success can indicate masked degradation.
- **Immediate mitigation:** Limit retries and investigate the failing dependency.
- **Correction / prevention:** Instrument attempt-level and run-level outcomes separately.
- **Verification:** Inject transient failures and inspect both metric families.
- **Boundary / escalation:** Retry success can be acceptable within an explicitly budgeted policy; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-57 — Tool result not persisted
- **Trigger / symptom:** A model uses a result that is lost during worker crash.
- **Candidate mechanism:** Tool output was kept only in process memory.
- **Competing hypothesis:** The tool may not be reproducible on a later invocation.
- **Discriminating diagnostic:** Compare invocation receipt persistence time with model continuation time.
- **Expected signal:** Required inputs must be durably recorded before dependent continuation.
- **Immediate mitigation:** Pause and reconstruct from the authoritative tool if possible.
- **Correction / prevention:** Commit tool result or a durable reference before advancing the run.
- **Verification:** Crash after tool response and before model continuation.
- **Boundary / escalation:** Sensitive tool outputs still need minimization and retention controls; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-58 — Overshared model context
- **Trigger / symptom:** A run includes another tenant's retrieved documents.
- **Candidate mechanism:** Retrieval filter or index namespace is missing.
- **Competing hypothesis:** A document may itself contain incorrectly labeled content.
- **Discriminating diagnostic:** Trace document IDs tenant metadata and retrieval authorization.
- **Expected signal:** Every included private document must pass the current tenant scope.
- **Immediate mitigation:** Stop affected retrieval and investigate exposure.
- **Correction / prevention:** Apply authorization at retrieval and verify returned document ownership.
- **Verification:** Seed two tenant corpora with distinctive markers and attempt cross-retrieval.
- **Boundary / escalation:** A relevant result is not necessarily an authorized result; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-59 — Unbounded approval wait
- **Trigger / symptom:** A run holds scarce worker resources for days.
- **Candidate mechanism:** Waiting is implemented as a blocking worker task.
- **Competing hypothesis:** A normal brief approval delay is not necessarily a defect.
- **Discriminating diagnostic:** Inspect worker occupancy while runs are in waiting state.
- **Expected signal:** Durable waiting should release compute resources when possible.
- **Immediate mitigation:** Drain waiting tasks into durable state.
- **Correction / prevention:** Persist continuation and schedule resume on decision or timeout.
- **Verification:** Pause many test approvals and observe worker slot usage.
- **Boundary / escalation:** Durable waits still consume storage and operational attention; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-60 — Incompatible snapshot deployment
- **Trigger / symptom:** New workers fail to restore old checkpoints.
- **Candidate mechanism:** A schema or serializer change breaks backward compatibility.
- **Competing hypothesis:** A corrupted checkpoint can produce similar errors.
- **Discriminating diagnostic:** Compare checkpoint version and supported decoder versions.
- **Expected signal:** Each retained checkpoint version needs a supported migration or explicit retirement.
- **Immediate mitigation:** Route affected runs to a compatible worker or pause safely.
- **Correction / prevention:** Test rolling deployment with historical snapshots and migrations.
- **Verification:** Restore old snapshots on the candidate release.
- **Boundary / escalation:** Rollback also needs compatibility with state written by the newer release; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

### F-61 — Hidden manual override
- **Trigger / symptom:** An operator fixes a run without an attributable record.
- **Candidate mechanism:** Administrative intervention bypasses normal transition logging.
- **Competing hypothesis:** An audit exporter outage can make a legitimate record temporarily invisible.
- **Discriminating diagnostic:** Compare database changes operator identity and audit stream.
- **Expected signal:** Every privileged override must be attributable and reviewable.
- **Immediate mitigation:** Freeze further overrides and preserve forensic evidence.
- **Correction / prevention:** Provide a controlled break-glass path with reason and audit.
- **Verification:** Exercise the override under a test role and inspect the durable trail.
- **Boundary / escalation:** An authenticated operator can still make an incorrect change; escalate if the outcome remains ambiguous.
- **Provenance:** HYPOTHETICAL; no incident or measured result is asserted.

**Takeaway:** Diagnose with correlated evidence, contain the unsafe boundary, and verify a specific invariant rather than merely retrying.

## 9. Complexity & Cost Analysis
### 9a. Algorithmic Complexity
Let E be the number of persisted events in a run, K the number replayed after the latest snapshot, and B the bytes of selected context.
These are logical operation counts, not observed latency; database indexes, contention and network round trips change elapsed time.
| Operation | Input size | Time | Auxiliary memory | I/O and assumptions |
|---|---|---|---|---|
| Admit and lookup | One indexed key | Expected indexed lookup, backend-dependent | O(1) excluding response | Authentication and authorization calls may dominate |
| Replay full log | E events | O(E) event applications | O(state size), excluding stored log | Reads all relevant event bytes; assumes constant-time application per event |
| Replay from snapshot | K suffix events | O(K) plus snapshot decode | O(snapshot size) | Correct only if snapshot covers the stated log prefix |
| Assemble context | B selected bytes | O(B) to read/copy selected material | O(B) if materialized | Retrieval and tokenizer cost may add work |
| Conditional state update | One indexed run key | Backend-dependent lookup and atomic write | O(1) client state | Conflicts cause additional attempts; no fixed worst-case latency claimed |
| Reconcile effects | J unresolved intents | O(J) external lookups in a simple sequential scan | O(1) working state if streamed | Remote latency, rate limits and pagination dominate |

### 9b. Back-of-Envelope Sizing
**ILLUSTRATIVE:** Extend Section 5's 900 MB physical storage estimate, with the same stated assumptions.
Daily admission is 1,000 sessions per 86,400 seconds, an average of approximately 0.01157 sessions/second.
That average does not characterize bursts, concurrency or the maximum queue length.
If each session adds an average of 10 events, the 30-day cohort contains 300,000 events.
If the average event requires two indexed writes, this simplified accounting predicts 600,000 logical writes across the retained cohort.
The writes include neither retry attempts nor updates, compaction, deletion, audit export or database internal amplification.
**DERIVED queue calculation Q1.** Validity: initially empty queue, constant arrival rate a and service rate s, no losses, a greater than s, and sufficient capacity during the interval.
Starting from backlog change per second a-s and integrating a constant rate for t seconds gives Q=(a-s)*t.
[1] FORMULA: Q = (a - s) * t.
[2] WHERE: Q is added queued tasks; a and s are tasks/second; t is seconds.
[3] CONSTRAINT: a > s and rates stay constant; this is a fluid approximation, not a tail-latency model.
[4] NUMERICAL EXAMPLE: a=120 tasks/s, s=100 tasks/s, t=300 s gives Q=20*300=6,000 tasks.
[5] INTERPRETATION: The system needs overload control or increased service capacity; a larger queue only postpones saturation.
Boundary: when rates change, use the integral of net rate while accounting for the zero-backlog floor; dropped or retried tasks require separate accounting.
If the 6,000 tasks each occupy 2,000 bytes of payload, the payload is 12,000,000 bytes before queue metadata and replicas.
That multiplication is an illustrative payload estimate, not a provisioned queue capacity or latency claim.
Environment for measurement remains OPEN: actual database, broker, hardware, region, software versions and autoscaling configuration are not specified.
Budget remains OPEN: define end-to-end deadline, per-tenant fairness, recovery objective, storage budget and external API quota before production selection.
Benchmark plan: replay realistic size and key-skew distributions; record admitted and completed runs, queue age, retries, p50/p95/p99 latency and error counts.
Inject a lost external response, a worker crash and a tenant authorization failure under the same workload.
Report repetitions and uncertainty, resource usage, region, price date and all software versions.
Economic measure: total attributable currency cost divided by successfully completed and accepted tasks; include model, storage, egress, labor and external tool charges where available.
This ratio is NOT MEASURED and should not be inferred from raw run count or token count alone.
**Takeaway:** Count physical storage, retries, queue growth and recovery work separately; neither big-O nor averages establish production latency.

## 10. Key Theorems / Results Reference
| Result | Statement and hypotheses | Practical use | Does not imply | Source |
|---|---|---|---|---|
| Atomic compare-and-set invariant | Given one atomic conditional update on a version, only one contender can change that specific version | Detect conflicting transitions | External effects are atomic with the database | DERIVED in Section 4.1; verify backend semantics |
| Little's law | For a stable system with well-defined long-run averages, average number in system equals arrival rate times average time in system | Cross-check measured queue and processing populations | A bound on tail latency or validity during growing backlog | [S3](https://doi.org/10.1287/opre.9.3.383) |
| Row security policy | PostgreSQL applies row security policies under its documented role and bypass rules when row security is enabled | Implement an additional database enforcement layer | Table owners or bypass roles are always subject to the same policy | [S2](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) |
| Idempotent HTTP methods | HTTP defines idempotency in terms of intended effect of repeated identical requests | Reason about transport retry semantics when the endpoint follows the specification | Any POST or arbitrary business action is automatically safe to repeat | [S4](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2) |

### Source and evidence registry
- **S1 — Scope crosswalk:** [Agents Stack Infra Module 03](../../../../Templates/directory.txt), lines naming control planes, ownership, state, isolation and lifecycle; directory inventory, not implementation evidence.
- **S2 — PostgreSQL documentation:** [Row security policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html), current documentation; supports the qualified database-policy statement only; version and deployment role behavior must be pinned for use.
- **S3 — Little (1961):** [A Proof for the Queuing Formula L = λW](https://doi.org/10.1287/opre.9.3.383), Operations Research 9(3); supports the stable long-run relationship, not the illustrative workload.
- **S4 — HTTP Semantics (2022):** [RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2); supports the definition of idempotent method semantics, not any particular vendor's deduplication behavior.
- **S5 — Security guidance:** [OWASP LLM Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/); relevant to untrusted input and constrained privileges, not a claim that one mitigation eliminates injection.
- **S6 — Authorization design:** [NIST SP 800-207 Zero Trust Architecture](https://csrc.nist.gov/pubs/sp/800/207/final), 2020; informs resource-focused authorization, not a certification of this design.
- Source versions marked current are mutable; confirm the deployed version and role configuration before implementation.
- Review date: not reviewed; no external performance claim or benchmark has been validated for this workload.
- OPEN: external provider's idempotency scope, retention window, lookup semantics and cancellation behavior require its primary API contract.
- OPEN: organization-specific data classification, deletion, backup hold, approval and residency requirements require owner confirmation.
**Takeaway:** A theorem or standard is useful only within its hypotheses and does not certify a particular deployment.

## 11. Interview / Exam Questions

**Interviewer scoring for each question:** 0 = incorrect or unsafe; 1 = partially correct but omits a critical boundary; 2 = correct with assumptions; 3 = correct and defends a tradeoff and verification. Candidate-facing prompts precede each answer. Unless a specific source is named, answers are supported by Sections 3–10.

**Alternative designs:** Accept a different mechanism when it enforces the stated invariant and the candidate identifies its failure boundary. For example, a serialized session queue can replace optimistic concurrency for ordinary updates only if single-writer ownership and failover are actually enforced; it does not by itself make an external effect atomic.

### Q01. What is the difference between a session and a run?
**Answer:** A session is an owned durable interaction container; a run is one bounded execution that advances it. Several runs can share a session, and a run can fail without deleting that session.
**Boundary / follow-up:** A process or model context is neither an ownership record nor necessarily durable.

### Q02. How would you prevent a known session ID from granting access?
**Answer:** Bind the authenticated principal to the tenant and check authorization for the session on every relevant read and write. Use the identifier only to locate a resource.
**Boundary / follow-up:** Test both direct APIs and cache/export paths with an unauthorized tenant.

### Q03. Two workers read version 4; how can they avoid a lost update?
**Answer:** Each conditionally updates only when the version is still 4. The first success changes it to 5, making the second update match zero rows; the loser reloads or reports a conflict.
**Boundary / follow-up:** This protects only the store enforcing the atomic predicate.

### Q04. What does a lease fail to guarantee?
**Answer:** Lease expiry does not stop a partitioned worker from executing. A receiver must enforce a monotonically ordered fencing generation on protected writes.
**Boundary / follow-up:** A third-party service that ignores fencing needs a separate safe-effect strategy.

### Q05. An external call times out after send; should you retry?
**Answer:** Not blindly. The remote action may have committed; query by stable effect key or use the provider's documented deduplication contract and retention scope. If no reliable resolution exists, hold for reconciliation.
**Boundary / follow-up:** A timeout is an observation about the response, not proof of remote failure.

### Q06. What should an approval bind to?
**Answer:** Bind the authenticated authorized actor, tenant, run, canonical action payload including destination and amount, policy context and expiry. Revalidate these at dispatch.
**Boundary / follow-up:** Changing a material field invalidates the approval; a model's assertion is not approval.

### Q07. Can you guarantee exactly-once effects with a local database transaction?
**Answer:** No, not across an independent service without a suitable shared protocol or receiver contract. Persist intent, use stable keys, reconcile receipts and explicitly handle unknown outcomes.
**Boundary / follow-up:** Deduplication is bounded by the receiver's scope and key retention.

### Q08. What is the difference between transcript and model context?
**Answer:** The transcript is the recorded sequence of interaction events; context is the selected material sent to a particular model call. It may omit history or contain retrieved untrusted material.
**Boundary / follow-up:** Authorization and durable state must not depend solely on whether a prompt contains a message.

### Q09. How do you safely cancel an in-flight run?
**Answer:** Persist cancel-requested, stop future dispatches, check at defined safe points and determine whether any already-sent effects committed. Mark a terminal state only under the documented disposition rules.
**Boundary / follow-up:** A cancellation request cannot undo an irreversible external effect.

### Q10. How would you size thirty days of state for 1,000 sessions daily at 10,000 bytes and multiplier 3?
**Answer:** Use S2: 1,000 * 30 * 10,000 * 3 = 900,000,000 physical bytes, assuming stationary cohorts and the stated multiplier.
**Boundary / follow-up:** This excludes backups and other omitted overhead and is not measured usage.

### Q11. What changes when policy is tightened while a run waits for approval?
**Answer:** Reauthorize the principal and action under the defined current policy on resume and again at the privileged boundary. If policy or payload no longer qualifies, pause or request a new approval.
**Boundary / follow-up:** Record policy versions; do not assume an earlier authorization remains valid indefinitely.

### Q12. How would you distinguish hot-key contention from a slow external tool?
**Answer:** Compare per-session queue depth and worker execution time with external tool latency and error rate. Concentration on one key with normal downstream latency supports a serialization bottleneck.
**Boundary / follow-up:** Correlation alone is not proof; reproduce with controlled skew and account for rate limits.

### Q13. Why is an append-only log not enough for safe recovery?
**Answer:** The log needs durable ordering, version-compatible readers, a consistent checkpoint boundary and explicit rules for external effects. Replay should reconstruct state rather than blindly execute side effects.
**Boundary / follow-up:** Logs can record intent without proving what the remote system committed.

### Q14. How would you test tenant isolation?
**Answer:** Create two synthetic tenants with distinct identities and markers, then exercise reads, writes, retrieval, caches, exports, resume and tool calls across boundaries. Verify both denials and legitimate access.
**Boundary / follow-up:** Tests must include privileged roles, background jobs and administrative routes in scope.

### Q15. What should be measured before declaring the design production-ready?
**Answer:** Measure task completion, authorization denials, unresolved effects, per-attempt retries, queue age, tail latency, resource consumption and cost per accepted outcome under realistic skew and faults.
**Boundary / follow-up:** Report workload, versions, sample size and uncertainty; the examples here are not benchmarks.

**Takeaway:** Strong answers state the authority, atomic boundary, failure window and evidence required to resolve uncertainty.

## 12. Quick Reference Card
| Item | Value |
|---|---|
| Topic | Agent session ownership, isolation and run recovery |
| Core formula / invariant | `C = A * D * B * M` for storage; authorize each state access and effect |
| Key assumption | Storage estimate assumes stationary cohorts and a stated multiplier; external effects need their own contract |
| Complexity | Per transition: one conditional state update plus applicable authorization and audit; actual latency is not measured |
| Use when | Designing durable, tenant-scoped runs with explicit approval and recovery |
| Avoid when | Provider effect semantics are unknown and the design assumes a timeout proves no action occurred |
| Go deeper | [Collection index](INDEX.md); specific framework contracts remain outside this sheet |
| Recovery guardrail | Fence stale writers where receivers enforce fencing; reconcile ambiguous external effects |
| Approval guardrail | Bind actor, tenant, payload, policy and expiry; recheck at dispatch |
| Evidence | Examples are illustrative; no production benchmark or outcome is established |
