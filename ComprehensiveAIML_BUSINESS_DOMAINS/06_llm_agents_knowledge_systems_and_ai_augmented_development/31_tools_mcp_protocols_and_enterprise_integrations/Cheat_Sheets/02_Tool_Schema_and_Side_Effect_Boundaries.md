---
collection: "tools-mcp-protocols-and-enterprise-integrations"
file_id: "02"
topic: "Tool schema and side-effect boundaries"
level: 2
status: "draft; unreviewed"
version: "1.1"
created: "2026-11-09"
reviewed: null
review_trigger: "protocol, executor, authorization, or downstream-contract change"
notation_style: "plain-text"
prerequisites: ["01_model_context_protocol.md"]
source_status: "OPEN — source bodies unavailable; visible titles are scope only"
owner_role: "curriculum maintainer"
source_index: "INDEX.md"
---

# Tool Schema and Side-Effect Boundaries

## 1. File Header
Scope: a tool is a bounded capability contract, not a model instruction or a business guarantee.
Outcome: given a trace, select a safe contract, calculate recovery capacity, and reject unsafe retry.
Evidence: workload values are ILLUSTRATIVE and arithmetic is VERIFIED; invariants are DERIVED; no deployed integration is asserted.
Prerequisites: typed records, authentication, authorization, queues, and failure semantics.
Source limitation: original source bodies were unavailable; filenames establish scope only.
Takeaway: a plausible call must still pass semantic, authority, and receipt checks.

## 2. Concept Map / Mental Model
```text
model proposal -> typed intent -> policy -> key/digest ledger -> executor -> receipt or UNKNOWN
                         |             |                    |              |
                     untrusted      deny/hold            audit/version    reconcile
```
Discovery describes an interface; it does not grant a principal permission.
A request identifier correlates messages; a durable operation key correlates one intended business effect.
A client response can be absent even after an external executor commits.
The actual executor must enforce scope independently of host/model context.
Takeaway: distinguish proposal, admission, dispatch, and outcome evidence.

## 3. Foundational Definitions
| Term | Meaning | Units/type | Boundary |
|---|---|---|---|
| Schema | accepted input/output shape | versioned record | not authorization |
| Canonical payload | normalized effect-relevant fields | deterministic object | not display prose |
| Idempotency key | stable intended operation correlation | opaque string | not request UUID |
| Receipt | authoritative outcome | terminal state | not UI toast |
| UNKNOWN | outcome not knowable yet | durable enum | not failure |
| lambda | arrival rate | operations/s | not a count |
| mu | completion rate | operations/s | not latency |
Takeaway: every symbol names one type and one evidence boundary.

## 4. Core Formulas with Full Derivation
### 4.1 Admission invariant
Validity: use only when the ledger atomically binds key/digest and the executor rechecks policy. Derivation: an effect is admissible only when all independent predicates hold.
```text
FORMULA [T-01]: accept=auth(p,r,t) AND schema(x) AND semantic(x,s) AND ledger(k,h(x))=NEW
WHERE: p=principal; r=resource; t=authority time [s]; x=typed input; s=business state; k=operation key; h=canonical digest.
CONSTRAINT: comparison-and-insert is atomic; every effect-relevant field enters h; authorization is enforced at executor.
NUMERICAL EXAMPLE: auth=1, schema=1, semantic=1, ledger=0 because K7 is COMMITTED; accept=0.
INTERPRETATION: return or reconcile K7; do not send a new write.
```
Counterexample: client cache state cannot prove a sink outcome after a crash.
### 4.2 Backlog and recovery
Validity: constant rates, no retries/bursts; a sizing estimate, not a latency guarantee. Derivation: conservation gives arrival minus completion, integrated over time.
```text
FORMULA [T-02]: B=max(0,lambda-mu)*t; T_drain=B/(mu_r-lambda_r)
WHERE: B [operations]; lambda,mu,mu_r,lambda_r [operations/s]; t,T_drain [s].
CONSTRAINT: drain needs mu_r>lambda_r and a measured initial backlog.
NUMERICAL EXAMPLE: lambda=18, mu=15, t=120 => B=360; mu_r=21, lambda_r=15 => T_drain=60 s.
INTERPRETATION: recovery works only with positive surplus.
```
Check: 3*120=360 and 360/(21-15)=60. At mu_r=lambda_r no finite drain exists.
### 4.3 Semantic collision
```text
FORMULA [T-03]: conflict=(k_old=k_new) AND (h_old != h_new)
WHERE: k is one intended operation and h covers amount, target, action, and approval-relevant fields.
CONSTRAINT: omitted field means this detector cannot see its semantic change.
NUMERICAL EXAMPLE: K9 changes amount 40 USD to 45 USD; digest differs, so reject as conflict.
INTERPRETATION: create a newly approved operation; never silently reuse the key.
```
Limit: no retry cache alone establishes exactly-once across an opaque external system.
Takeaway: load math bounds queues; enforced contracts bound duplicate effects.

## 5. Worked Numerical Example
ILLUSTRATIVE: a procurement connector admits 18 operations/s for 120 s while execution completes 15/s.
Backlog is (18-15)*120=360 operations.
A recovery lane completes 21/s while live work remains 15/s, so surplus is 6/s.
Drain is 360/6=60 s; arithmetic independently agrees in both directions.
A response for K-204 is lost: query the authoritative ledger/sink with K-204, never mint K-205.
Sensitivity: at arrivals 22/s, the same interval queues 840 operations; with live arrivals 22/s and recovery 21/s, backlog cannot drain.
Decision: provide a bounded reconciliation lane and hold UNKNOWN writes rather than retrying blindly.
Takeaway: the material boundary is recovery surplus, not the nominal average.

## 6. Visual / Geometric Intuition
Picture a gate before a conveyor: the model labels a box, but the gate attendant verifies identity, destination, rule, and duplicate label.
The receipt exists beyond the gate; a broken radio does not prove the box did not pass.
State is a one-way graph: PROPOSED -> VALIDATED -> RESERVED -> SUBMITTED -> COMMITTED | UNKNOWN | REJECTED.
Recovery never jumps from UNKNOWN to NOT_SENT without new authoritative evidence.
Takeaway: draw the evidence boundary at the executor, not the chat interface.

## 7. Comparison / Selection Table
| Option | Use when | Benefit | Avoid when | Fallback |
|---|---|---|---|---|
| Narrow read | scoped fresh lookup | small blast radius | result is sensitive without policy | redacted result |
| Deterministic flow | route/actions known | auditable recovery | adaptive discovery necessary | bounded planner |
| Idempotent write | key and receipt exist | safe duplicate recovery | opaque irreversible sink | manual review |
| Atomic batch | one semantic operation | efficiency | mixed independent intents | per-item keys |
| Protocol discovery | manifest is pinned | interoperability | it is treated as authority | executor policy |
Takeaway: choose the smallest contract whose failure boundary matches the consequence.

## 8. Failure Modes & Edge Cases
All issues are HYPOTHETICAL teaching scenarios, not observed incidents.
### ISSUE-001 — Wildcard target [VIOLATED_ASSUMPTION]
**Trigger/environment.** A syntactically valid request expands beyond the reviewed resource.
**Competing causes.** schema overbreadth; resolver expansion; stale preview.
**Diagnostic / expected signal.** compare canonical resolved targets with the approval.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** hold and require an explicit bounded target list.
**Mitigation risk.** confirmation fatigue.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-002 — Lost response [NUMERICAL_EDGE]
**Trigger/environment.** The caller times out after dispatch.
**Competing causes.** network loss; slow executor; downstream commit.
**Diagnostic / expected signal.** keyed receipt is COMMITTED, NOT_FOUND, or UNKNOWN.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** freeze retry and reconcile the same key.
**Mitigation risk.** long wait consumes workers.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-003 — Key collision [SCALE_FAILURE]
**Trigger/environment.** A retry changes an amount or destination.
**Competing causes.** UI mutation; canonicalization error; replay.
**Diagnostic / expected signal.** stored payload digest differs.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** reject and require a new approved operation.
**Mitigation risk.** automatic key rollover hides duplicates.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-004 — Schema-valid denial [CONCEPT_CONFUSION]
**Trigger/environment.** Shape passes but business policy denies.
**Competing causes.** missing entitlement; expired delegation; state change.
**Diagnostic / expected signal.** policy trace names a failed predicate.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** return a typed denial before execution.
**Mitigation risk.** detail can aid probing.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-005 — Secret leakage [VIOLATED_ASSUMPTION]
**Trigger/environment.** Tool output reaches logs or model context.
**Competing causes.** overbroad output; debugging; error serialization.
**Diagnostic / expected signal.** redaction scan finds secret-shaped fields.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** stop propagation and minimize output.
**Mitigation risk.** over-redaction harms repair.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-006 — Retry storm [NUMERICAL_EDGE]
**Trigger/environment.** Failures multiply request volume.
**Competing causes.** missing backoff; shared outage; slow key store.
**Diagnostic / expected signal.** attempt/key rises while receipts fall.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** bound retries and prioritize reconciliation.
**Mitigation risk.** hard shedding strands work.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-007 — Version drift [SCALE_FAILURE]
**Trigger/environment.** Client and server interpret a field differently.
**Competing causes.** stale manifest; silent default; deployment mismatch.
**Diagnostic / expected signal.** pinned schema versions mismatch.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** route compatible version or hold migration.
**Mitigation risk.** permissive conversion changes meaning.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-008 — Cross-tenant read [CONCEPT_CONFUSION]
**Trigger/environment.** A query returns another tenant’s handle.
**Competing causes.** missing filter; confused deputy; cache key omission.
**Diagnostic / expected signal.** same query under two tenants returns forbidden data.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** deny, invalidate, enforce at server.
**Mitigation risk.** client filters are bypassable.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-009 — Detached approval [VIOLATED_ASSUMPTION]
**Trigger/environment.** Review accepted intent but not exact payload.
**Competing causes.** free text; omitted field; stale digest.
**Diagnostic / expected signal.** approval digest/version differs from action.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** re-present canonical action.
**Mitigation risk.** extra review delays safe work.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-010 — Expired receipt [NUMERICAL_EDGE]
**Trigger/environment.** A delayed retry cannot find its record.
**Competing causes.** short TTL; clock error; partition.
**Diagnostic / expected signal.** lookup expires before recovery horizon.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** size retention to recovery window.
**Mitigation risk.** longer retention enlarges metadata surface.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-011 — Burst blindness [SCALE_FAILURE]
**Trigger/environment.** Average rate says capacity is sufficient.
**Competing causes.** bursts; tail latency; serial bottleneck.
**Diagnostic / expected signal.** queue percentile exceeds fluid forecast.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** add admission guard and measure tail.
**Mitigation risk.** overprovisioning wastes capacity.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-012 — False rollback [CONCEPT_CONFUSION]
**Trigger/environment.** A compensation is described as undo.
**Competing causes.** irreversible effect; downstream fanout; partial commit.
**Diagnostic / expected signal.** audit shows a second distinct effect.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** use explicit compensating process.
**Mitigation risk.** compensation may worsen state.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-013 — Injected description [VIOLATED_ASSUMPTION]
**Trigger/environment.** Tool text changes routing.
**Competing causes.** untrusted description; retrieved text; model confusion.
**Diagnostic / expected signal.** policy-independent text changes action proposal.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** treat text as data and allowlist actions.
**Mitigation risk.** blocking all text harms usability.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-014 — Missing principal [NUMERICAL_EDGE]
**Trigger/environment.** Only a service account is audited.
**Competing causes.** lost delegation; async propagation gap; shared credential.
**Diagnostic / expected signal.** trace lacks initiating principal.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** fail closed for writes until bound.
**Mitigation risk.** strictness interrupts work.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-015 — Unknown closed as failed [SCALE_FAILURE]
**Trigger/environment.** An ambiguous effect is silently terminal.
**Competing causes.** no runbook; on-call pressure; opaque sink.
**Diagnostic / expected signal.** no receipt evidence in closure.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** assign owner and reconcile.
**Mitigation risk.** manual queue needs access control.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-016 — Stale policy [CONCEPT_CONFUSION]
**Trigger/environment.** Permission changes during a long run.
**Competing causes.** revocation; role change; resource move.
**Diagnostic / expected signal.** execution-time policy differs from admission.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** re-authorize at effect boundary.
**Mitigation risk.** rechecks add dependency load.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-017 — Path escape [VIOLATED_ASSUMPTION]
**Trigger/environment.** Allowed path resolves outside scope.
**Competing causes.** symlink; TOCTOU; normalization defect.
**Diagnostic / expected signal.** canonical target differs from approved target.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** resolve then authorize target.
**Mitigation risk.** rejecting links can break workspaces.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-018 — Opaque sink [NUMERICAL_EDGE]
**Trigger/environment.** Downstream cannot deduplicate or query.
**Competing causes.** legacy API; header dropped; vendor limit.
**Diagnostic / expected signal.** no correlation at executor.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** do not automate irreversible write.
**Mitigation risk.** manual work is slower.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-019 — Prose error [SCALE_FAILURE]
**Trigger/environment.** Caller cannot classify recovery.
**Competing causes.** exception mapping; transport error; missing taxonomy.
**Diagnostic / expected signal.** error lacks outcome/retry class.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** emit typed error states.
**Mitigation risk.** details may leak internals.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-020 — Cached grant [CONCEPT_CONFUSION]
**Trigger/environment.** Revoked user continues acting.
**Competing causes.** long TTL; bad cache key; clock skew.
**Diagnostic / expected signal.** live policy denies while cache permits.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** bound cache and revalidate.
**Mitigation risk.** cache removal adds load.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-021 — Mixed batch [VIOLATED_ASSUMPTION]
**Trigger/environment.** One approval covers many independent actions.
**Competing causes.** array ambiguity; aggregation bug; UI summary.
**Diagnostic / expected signal.** per-item digest/count absent.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** split keys or define atomic batch.
**Mitigation risk.** many prompts cause fatigue.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-022 — Blind dead-letter replay [NUMERICAL_EDGE]
**Trigger/environment.** Old event applies to new state.
**Competing causes.** stale payload; duplicate delivery; version mismatch.
**Diagnostic / expected signal.** expected version mismatch.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** revalidate current state before replay.
**Mitigation risk.** dropping loses remediation.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-023 — Stale discovery [SCALE_FAILURE]
**Trigger/environment.** Model chooses unsupported tool.
**Competing causes.** server restart; cached manifest; routing bug.
**Diagnostic / expected signal.** capability/version error.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** refresh manifest and use safe fallback.
**Mitigation risk.** fallback semantics differ.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-024 — Budget omission [CONCEPT_CONFUSION]
**Trigger/environment.** Retries exceed nominal cost cap.
**Competing causes.** hidden calls; retry loop; token drift.
**Diagnostic / expected signal.** actual attempts exceed admitted cap.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** charge budget at every descendant.
**Mitigation risk.** hard cap leaves partial task.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

### ISSUE-025 — Raw audit [VIOLATED_ASSUMPTION]
**Trigger/environment.** Diagnostic event stores sensitive arguments.
**Competing causes.** debug mode; exception capture; broad retention.
**Diagnostic / expected signal.** sample has raw sensitive field.
**Provenance.** HYPOTHETICAL; inferred from the stated contract, not a production claim.
**Immediate mitigation.** redact at source and restrict access.
**Mitigation risk.** hash-only evidence can slow forensics.
**Correction.** Repair authoritative policy/schema/state and replay only from a versioned record with explicit outcome.
**Prevention.** Add a contract test covering this trigger and the nearest competing cause.
**Verification.** Inject it in isolation; inspect policy, ledger, executor receipt, and redacted audit terminal state.
**Related sheet.** Sections 4–5; a UI success message is not the oracle.

Takeaway: a symptom is not a cause; require a signal that differs under the closest alternative.

## 9. Complexity & Cost Analysis
For n attempts, append-only transition storage is O(n). Snapshots every k transitions bound recent replay to O(k) while adding O(n/k) snapshot writes.
A bounded attribute policy lookup is O(a) for a relevant attribute count a; unbounded graph expansion or free-text execution invalidates that bound.
Illustrative storage: 360 operations * 1.5 KiB redacted metadata = 540 KiB payload-only. Indexes, encryption, replication, tool payloads, and retention overhead are excluded.
This is sizing, not a price or throughput benchmark.
Takeaway: name record units and exclusions before capacity claims.

## 10. Key Theorems / Results Reference
DERIVED atomic admission: an atomic key/digest insertion prevents two concurrent attempts both receiving NEW from that ledger; it says nothing about an uncontrolled sink.
DERIVED uncertainty: absence of a caller response is insufficient evidence of absence of an executor effect.
SOURCED method reference: OWASP LLM01:2025 supports treating untrusted content as threat input; it does not validate this design.
OPEN: framework/provider behavior and original source bodies were not reviewed; pin and test versions before implementation.
Takeaway: preserve unknowns beside the conclusion.

## 11. Interview / Exam Questions
### Q01. Why is schema validation insufficient?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q02. What does a timeout prove?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q03. Compute drain for 360 operations and 6 operations/s surplus.
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q04. When may an idempotency key be reused?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q05. Why authorize at the executor?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q06. What does UNKNOWN mean?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q07. When does a deterministic workflow win?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q08. How do you test a duplicate window?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q09. What is a confused deputy?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q10. What is the cost of longer key retention?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q11. Why can compensation be unsafe?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q12. How should typed errors guide callers?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q13. How does versioning limit semantic drift?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q14. What audit fields are minimally useful?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

### Q15. What changes invalidate this design?
**Answer.** Start with the authoritative effect boundary and separate accepted syntax, permission, dispatch, and committed outcome.
**Reasoning.** The correct choice depends on the invariant and on evidence that distinguishes the nearest competing cause.
**Follow-up.** Change the downstream receipt or authorization assumption and state whether automation remains safe.
**Acceptable alternative.** A simpler flow is valid if it narrows actions and retains equivalent evidence/recovery controls.
**Red flag.** Treating a model proposal, client timeout, or UI message as proof of business outcome.
**Scoring.** 0 assertion; 1 control named; 2 invariant plus diagnostic; 3 boundary and verification oracle.
**Answer support.** Sections 4, 5, and 8.

Takeaway: a strong answer says what is known, what is unknown, and what to observe next.

## 12. Quick Reference Card
| Situation | Rule | Critical assumption | Avoid | Evidence |
|---|---|---|---|---|
| Tool proposal | treat as untrusted intent | executor enforces policy | model authority | DERIVED |
| Timeout | reconcile same key | receipt/dedup exists | fresh-key retry | DERIVED |
| Payload changes | reject same-key different digest | canonical fields complete | silent reuse | DERIVED |
| High load | calculate surplus | rates representative | capacity guess | ILLUSTRATIVE |
| Source claim | pin version/scope | primary source available | filename-as-evidence | OPEN |
| Diagnosis | test competing cause | trace retained/redacted | symptom-only fix | HYPOTHETICAL |
**Sources.** MCP specification dated 2025-06-18 (version-specific; recheck) and OWASP LLM01:2025; original catalog source bodies remain OPEN.
**Status.** Draft; examples have local arithmetic/logic checks only. No independent technical, source, or deployment review is claimed.
