---
collection: "30_agentic_systems_workflows_and_orchestration"
file_id: "01"
topic: "Agent boundaries and request ownership"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in identity, tool, state-store, or execution contracts"
notation_style: "plain-text"
prerequisites: ["Basic HTTP authentication and transactions"]
pattern_tags: [agent, boundary, request-path, ownership]
source_status: "OPEN"
owner_role: "Agent platform engineering"
source_index: "INDEX.md"
---
# Agent Boundaries and Request Ownership

> **Source inventory:** [directory entry](../../../../Templates/directory.txt#L14) names `Agents_Stack_Infra/Module_01_Agent_Stack_Overview` and visible titles “why agent is overloaded,” “ten layer stack model,” “the request path,” “critical boundary distinctions,” “what breaks when layers collapse,” and “builder checklist.” Their bodies were unavailable. This sheet teaches the visible boundary topics without attributing unseen claims; see the [focused coverage brief](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-stack-model).

## 1. File Header
> **Purpose:** Separate model generation, control-plane authority, runtime state, and external execution so that a request can be audited and safely retried.
> **Prerequisites:** HTTP identity propagation, authorization, transactions, and queue/retry basics.
> **Key outcome:** Given a request path and a retry, assign every decision and state transition to one owner and identify which effects may occur twice.
> **This file does NOT cover:** A vendor-specific agent API, model training, or legal access-policy interpretation.
> **Level:** 1 — Foundations.
> **Version / reviewed:** 1.0 / not reviewed; draft.
> **Evidence:** DERIVED invariants and VERIFIED arithmetic on ILLUSTRATIVE traces. Source filenames are an incomplete syllabus, not source-body evidence.
> **Retrieval cue:** For a suspected cross-tenant or duplicate-action bug, start at Section 8.

An agent is an overloaded word. It can mean a model loop, a product workflow, a runtime, or a person with delegated authority. Treating those as one component makes it impossible to say who authorizes an action or who owns a retry. This sheet uses *agent* only for the bounded workflow that proposes and advances work; it does not confer identity or permission.

**Takeaway:** Give each request, decision, durable record, and effect a named owner before discussing autonomy.

## 2. Concept Map / Mental Model
```text
caller --authenticated request--> control plane --admission--> run record
                                      |                         |
                                policy + quota                  | checkpoint
                                      v                         v
model context <- assembler <- permitted state              runtime worker
     | proposal                                           | scoped command
     v                                                    v
tool request --validate/authorize--> adapter --effect key--> system of record
                                      |                         |
                                      +------receipt/audit-------+
```
The model sees derived context, including potentially untrusted text. The control plane is the decision owner for admission and policy. The runtime owns progression through a run state machine; a worker is replaceable execution capacity, not the source of truth. The adapter owns translation to a specific external contract. The system of record owns the business effect.

A boundary is useful only when it has an input contract, an output contract, an identity, a state owner, and a failure/recovery rule. A network hop is not automatically a boundary, and a single service can implement several logical boundaries if their contracts remain separately enforceable.

**Takeaway:** The request path is a chain of contracts, not a chain of prompts.

## 3. Foundational Definitions
| Term | Meaning | Type / unit | Boundary test | Nonexample |
|---|---|---|---|---|
| Principal | Authenticated actor on whose behalf access is evaluated | identity claim | Can policy evaluate it at the action? | A session ID alone |
| Session | Tenant-scoped interaction container | durable record | Can it outlive one run? | Authorization proof |
| Run | One finite attempt to advance a request | state machine instance | Has a version and terminal state | An external effect |
| Control plane | Admission, policy, scheduling, and audit authority | logical service | Can it reject before work starts? | Model output |
| Runtime | Component that executes transitions and restores checkpoints | process/service | Can another worker resume it? | External system of record |
| Tool adapter | Validates, authorizes, and translates a requested capability | protocol boundary | Can it deny schema-valid input? | A free-form shell prompt |
| Effect key | Stable identifier for one intended business effect | opaque string | Is it recorded before/with submission? | A request timeout |
| Receipt | Authoritative external outcome or queryable status | record | Can reconciliation use it? | An LLM assertion |

A session may contain many runs. A run may attempt one effect several times at the transport layer, but it must name one intended business operation. Context is neither durable authority nor necessarily complete history. A tool capability describes what could be called; authorization decides whether this principal may call it now with these arguments.

**Takeaway:** IDs locate records; authorization and receipts justify access and outcome.

## 4. Core Formulas with Full Derivation
### 4.1 Effect-attempt bound
Validity before use: this is an upper bound for a policy that caps branches and calls; it does not predict model behavior, latency, or successful effects.

Derivation: if a run permits `b` branches and at most `c` calls per branch, multiplication counts the largest call attempts. Let `e` be calls that cross an effecting adapter; `e <= b*c`.

```text
FORMULA [AB-01]: A_max = b * c
WHERE: A_max [call attempts/run]; b [branches/run]; c [calls/branch].
CONSTRAINT: fixed independent caps; no recursive spawning outside the budget.
NUMERICAL EXAMPLE: b=4, c=3; A_max=4*3=12 attempts/run.
INTERPRETATION: twelve attempts is a budget alarm threshold, not twelve permitted business effects.
```
Dimension check: branches/run × calls/branch = calls/run. If `b=2`, `A_max=6`; halving a branch cap halves this bound. Counterexample: an uncapped retry loop violates the premise, so no finite bound follows.

### 4.2 Duplicate-risk accounting
Validity before use: count effect *attempts* when a response can be lost after submission. It does not claim a duplicate business outcome when the sink enforces idempotency.

Derivation: an effect can be attempted once before a lost response and once after an unsafe retry. For `r` ambiguous operations with one retry each, the maximum attempts are `2r`.

```text
FORMULA [AB-02]: A_effect,max = r * (1 + q)
WHERE: r [ambiguous intended operations]; q [retries/operation]; A_effect,max [attempts].
CONSTRAINT: one initial submission and bounded retries; each retry reaches the adapter.
NUMERICAL EXAMPLE: r=3, q=1; A_effect,max=3*(1+1)=6 attempts.
INTERPRETATION: three operation IDs, not six, must be reconciled before declaring outcome.
```
Independent check: enumerate `op-1a, op-1b, op-2a, op-2b, op-3a, op-3b`: six attempts. With an adapter that rejects a repeated effect key, accepted effects are at most three; with an opaque non-idempotent sink, acceptance is unknown until queried. 

### 4.3 Ownership invariant
Validity before use: an atomic compare-and-swap or serialized owner is available for run transitions.

```text
INVARIANT [AB-03]: transition(run_id, v -> v+1) is accepted only when
                   tenant_id, principal scope, expected version, and policy version match.
PRECONDITIONS: immutable run_id; versioned durable record; authorization at transition time.
STATE TRANSITION: READY(v) -> RUNNING(v+1) -> WAITING|SUCCEEDED|FAILED.
JUSTIFICATION: two writers proposing v+1 cannot both satisfy one expected-version update.
```
Worked trace: W1 and W2 read version 7. W1 writes `RUNNING, version=8`; W2's compare-and-swap expects 7 and fails. W2 reloads 8, re-authorizes, and chooses a legal next transition. The check is one accepted write plus one rejected write; accepting both is a broken invariant. Counterexample: last-write-wins can hide the conflict and overwrite a cancellation.

### 4.4 Lease and fencing separation
Validity before use: leases are used to decide whether a worker may *try* to own work, and the state store or sink can reject a stale owner with a monotonically increasing fencing token. A lease alone does not stop a paused worker from resuming after expiry.

Derivation: a lease end is `t_exp = t_grant + d`. The control plane issues token `f+1` for a successor. Every protected write carries its token; the sink accepts only a token greater than the last accepted token for that protected resource.

```text
FORMULA [AB-04]: t_exp = t_grant + d
WHERE: t_exp [time]; t_grant [time]; d [seconds, positive lease duration].
CONSTRAINT: all comparisons use one trusted time source or a stated skew bound.
NUMERICAL EXAMPLE: t_grant=100 s, d=30 s; t_exp=130 s.
INTERPRETATION: after 130 s a new owner may be elected, but a delayed old owner remains dangerous without fencing.
```

```text
INVARIANT [AB-05]: accept_write(resource, f_request) only if f_request > f_last_accepted.
PRECONDITIONS: tokens are monotonically issued per resource and the receiver durably remembers f_last_accepted.
TRACE: W1 has f=41 and pauses; lease expires; W2 gets f=42 and writes; W1 resumes with f=41 and is rejected.
FAILURE BOUNDARY: a receiver that ignores f makes lease expiry advisory only.
```

The clock calculation describes eligibility, not correctness. With a possible 4-second clock skew and a 30-second lease, renew before the useful remaining interval is consumed; do not allow a worker to decide safety from its local clock alone. A long lease reduces renewal traffic but lengthens failover; a short lease improves failure detection but increases false takeovers during pauses. These are workload and scheduler measurements to obtain, not universal defaults.

### 4.5 Cancellation as a state transition
Cancellation is a request to stop future work, not a retroactive deletion of completed effects. It needs a versioned state transition and propagation boundary.

```text
INVARIANT [AB-06]: once CANCEL_REQUESTED(v) is committed, no later transition may initiate
                   a new effect unless an explicit authorized resume creates a new version.
PRECONDITIONS: workers check cancellation before each effect boundary; adapters reject stale run versions.
TRACE: planner completes; cancel commits at v=12; worker sees v=12 before tool call and records CANCELLED(v=13).
RECOVERY: if an effect was already SUBMITTING, reconcile it; do not label it cancelled without a receipt.
```

The common wrong implementation merely cancels an in-memory task. That loses cancellation on restart and cannot stop a different worker. The common overcorrection assumes cancellation reverses a ticket or payment; reversal requires a separately authorized compensating action with its own effect key.

### 4.6 Canonical-action binding
An approval or authorization decision must bind to canonical action data, not to a conversational paraphrase. Define `H` as a collision-resistant digest of a canonical representation containing operation type, resource, normalized arguments, tenant, principal, policy version, expiry, and effect key.

```text
FORMULA [AB-07]: approval_valid = (H(action_now)=H(action_reviewed)) AND (now < expiry)
WHERE: H [digest]; action_now/action_reviewed [canonical byte sequences]; expiry [time].
CONSTRAINT: canonicalization is deterministic and includes all effect-relevant fields.
NUMERICAL EXAMPLE: changing quantity from 10 to 11 changes the canonical bytes; equality check fails.
INTERPRETATION: the system must request a new approval rather than treating the old intent as permission.
```

The formula is a mechanism statement, not a cryptographic proof of semantic equivalence. Omitting currency, resource scope, or policy version creates a hole: the hash can match while the effective action changed. A hash also does not authorize anything unless an independent approver identity and server-side policy check are present.

### 4.7 Request-path trace matrix
| Step | Owner | Input | Durable evidence before next step | Permitted recovery |
|---|---|---|---|---|
| Authenticate | gateway/control plane | credentials and request | principal, tenant, auth method | reject or re-authenticate |
| Admit | control plane | principal, quota, policy | run ID, policy version, budget | deny without run |
| Assemble context | context service | permitted records | source IDs and access decisions | rebuild from sources |
| Propose | model/runtime | bounded context | prompt/model revision and output reference | regenerate only as a new proposal |
| Validate | adapter | structured arguments | schema result and canonical action | return validation error |
| Authorize | policy service | principal/action/current policy | decision and expiry | deny or re-evaluate |
| Reserve effect | runtime/store | effect key/action digest | submitting state/version | reconcile same key |
| Execute | adapter/sink | canonical request/key | receipt or ambiguity record | query/compensate/manual review |
| Finalize | runtime | receipt/status | terminal run transition | resume reconciliation |

**Takeaway:** Leases elect an owner, fencing rejects stale owners, cancellation blocks future effects, and canonical binding prevents approval drift.

**Takeaway:** Bound calls separately from effects, and serialize ownership before retrying state transitions.

## 5. Worked Numerical Example
**Evidence:** ILLUSTRATIVE workload; arithmetic checked, not a production measurement.

A support assistant receives one request in tenant `T9`. Its planner may create `b=3` branches; each has `c=2` tool calls. Only one branch may submit a ticket. A network response is lost after the adapter submits the ticket with effect key `T9:R81:ticket:1`.

1. Call budget: `A_max=3*2=6` calls. This prevents an accidental exponential planner from silently consuming an unbounded budget.
2. The runtime commits `EFFECT_SUBMITTING(key)` before calling the adapter.
3. The adapter sends the key. The client times out; this is ambiguous, not failure proof.
4. Recovery queries by key. If it receives a receipt, it writes `EFFECT_CONFIRMED(receipt)` and does not submit again.
5. If querying is temporarily unavailable, recovery writes `RECONCILIATION_PENDING`; it does not invent a new key.

Sensitivity: with `b=6`, the call cap becomes 12, doubling available tool attempts and audit volume. That can be acceptable only if quotas, approvals, and observation capacity also cover it. Replacing the adapter with one that cannot query or deduplicate removes the evidence needed for a safe automatic retry; the design must require an approval/reconciliation path.

### Extended trace: cancellation, failover, and an already-submitted effect
This is a hand-derived trace, not an executed fault-injection result. Times are illustrative seconds on the control-plane clock; worker observations are never substituted for durable evidence.

| Time | Actor | Durable state / event | Required conclusion |
|---:|---|---|---|
| 0 | gateway | authenticated `T9:P7` request | identity is established for this request |
| 1 | control plane | `RUN_CREATED(R81,v=1,policy=44,budget=6)` | no external action exists |
| 2 | W1 | lease token `f=41`, expiry 32 | W1 may attempt transitions |
| 4 | runtime | `ACTION_PROPOSED(v=2,digest=D1)` | proposal is not approval |
| 5 | policy | `ALLOW(D1, expiry=25, policy=44)` | only digest D1 is permitted |
| 6 | runtime | `EFFECT_SUBMITTING(v=3,key=K1,digest=D1)` | K1 is the sole intended ticket effect |
| 7 | adapter | sends K1; response path fails | sink outcome is unknown |
| 8 | caller | cancellation request arrives | durable cancellation is required |
| 9 | control plane | `CANCEL_REQUESTED(v=4)` | no new effect may start |
| 33 | scheduler | W2 receives `f=42` after W1 expiry | W1 is stale |
| 34 | W1 | delayed write with `f=41` | receiver must reject it |
| 35 | W2 | query K1 returns `ticket=812` | one effect occurred |
| 36 | runtime | `CANCELLED_AFTER_EFFECT(v=5,receipt=812)` | cancellation stopped follow-on work, not history |

`CANCELLED` without the receipt would be misleading, while `SUCCEEDED` conceals that follow-on work did not run. If the receipt query is temporarily unavailable, the only honest terminal decision is a non-final `RECONCILIATION_PENDING` state. It must preserve K1; a replacement key creates a second business operation.

#### Design-oracle inputs
The following inputs specify a later executable test. They are not execution evidence.

```text
initial_version=1; policy_version=44; approved_digest=D1; effect_key=K1
W1_token=41; W2_token=42; cancellation_after_effect_reservation=true
receipt_lookup(K1)=ticket-812
EXPECT accepted_effect_keys=[K1]
EXPECT rejected_stale_tokens=[41]
EXPECT terminal_state=CANCELLED_AFTER_EFFECT
EXPECT terminal_receipt=ticket-812
EXPECT new_effects_after_cancel=0
```

Changed constraint: set `approved_digest=D2` while the proposed action hashes to D1. The adapter must deny or ask for new approval. It must not borrow the authority for D2. Unavailable receipt lookup must yield `RECONCILIATION_PENDING`, not a claimed failure or success.

### Identity propagation decision record
| Option | What the adapter proves | Use when | Disqualifier |
|---|---|---|---|
| Forward caller credential | caller identity and scoped access | audience and lifetime are tightly constrained | token is broad or cannot safely reach worker |
| Token exchange | delegated caller authority for adapter audience | downstream service needs caller context | exchange policy is not auditable |
| Service identity plus signed delegation | service execution and original-principal evidence | sink cannot accept user token | delegation is not independently checked |
| Service identity only | which service acted | shared read-only work | per-user entitlement affects outcome |

Record original principal, effective service identity, tenant, delegation method, policy decision ID, and action digest. Never log raw credentials. Service identity alone is insufficient for an effect that depends on user entitlement.

### Durable state versus durable effects
| Artifact | Authoritative owner | Recovery question | Invalid inference |
|---|---|---|---|
| Run phase/version | state store | what transition is legal next? | worker memory is current |
| Approval | policy/audit store | does it bind current digest and time? | old approval always applies |
| External ticket | external system | did K1 commit? | local transaction proves remote effect |
| Receipt reference | run audit | which outcome was observed? | receipt means permanent validity |

**Source scope ledger:** NIST's least-privilege glossary supports the narrow minimum-privileges definition only. OWASP LLM01:2025 (modified 2025-04-17) supports treating indirect injection in external content as a threat, and recommends enforced least privilege, high-risk approval, untrusted-content segregation, and adversarial testing. Neither source validates this architecture, its safety, or an immunity claim.

**Takeaway:** A timeout changes knowledge, not the external world; reconcile the same operation identity.

## 6. Visual / Geometric Intuition
Think of authority as nested regions. A tenant boundary encloses its sessions; a session encloses its runs; a run contains proposed actions. A model-generated string lies inside none of these authority regions until an adapter validates it. Moving text from context to a tool argument crosses into a smaller, more controlled region and must therefore add validation rather than trust.

```text
[tenant T9]
  [session S4]
    [run R81]
      proposal: "create ticket"  -- untrusted intent
      approved canonical action   -- bound arguments + policy
      effect key                  -- one business operation
```

**Takeaway:** Narrower execution regions require stronger binding of identity, arguments, and policy.

## 7. Comparison / Selection Table
| Design | Use when | Benefit | Disqualifier | Fallback |
|---|---|---|---|---|
| Deterministic workflow | Steps and routing are known | Small audit surface | Open-ended decomposition is necessary | Add bounded planner at one node |
| Bounded planner + tools | Alternatives need evaluation | Flexible proposal generation | Tools have irreversible broad effects | Approval gate and narrow tools |
| Supervisor + specialists | Work partitions with clear contracts | Domain isolation | Shared mutable state is unclear | One workflow with explicit phases |
| Free-form multi-agent chat | Exploration only | Rapid ideation | Any production effect or audit requirement | Convert outputs to reviewed artifacts |
| Direct external call from model | Read-only disposable sandbox | Low setup | Identity, money, records, or safety involved | Adapter with schema/policy/receipt |

**Takeaway:** Add an agentic layer only where it changes a bounded decision more safely or effectively than a workflow.

## 8. Failure Modes & Edge Cases
### F-01 Collapsed trust boundary
**Symptom:** A retrieved document causes a tool request outside the user’s intended scope. **Competing causes:** prompt injection, overbroad tool schema, or missing downstream authorization. **Diagnostic:** log canonical arguments, principal, policy decision, and document provenance; an allowed action despite absent scope identifies enforcement failure. **Bounded fix:** require adapter-side authorization over structured arguments. **Risk:** a blanket deny may block legitimate automation. **Prevention/verification:** adversarial document test must produce a denied adapter decision without an effect.

### F-02 Double-owned state
**Symptom:** Two workers report conflicting terminal states. **Competing causes:** missing compare-and-swap, lease expiry without fencing, or replaying stale events. **Diagnostic:** inspect expected/current versions and owner fencing tokens. **Fix:** single writer or versioned transition. **Verification:** concurrent version-7 trace accepts exactly one transition.

### F-03 Ambiguous retry creates duplicate ticket
**Symptom:** Two tickets share business content after a timeout. **Competing causes:** regenerated effect key, sink ignores key, or reconciliation query races visibility. **Diagnostic:** query receipt by the original key and compare adapter audit IDs. **Fix:** persist key before submit and reconcile. **Verification:** injected lost-response test produces one key and either one receipt or a pending state, never a new key.

### F-04 Cross-session leakage
**Symptom:** Context from `S4` appears during `S5`. **Competing causes:** cache key omits tenant/session, worker memory reused, or authorization only happened at creation. **Diagnostic:** trace cache key and read authorization at every retrieval. **Fix:** tenant-scoped keys and per-read policy checks. **Verification:** same session identifier in two tenants must return deny or isolated records.

### F-05 Budget escape through nested spawning
**Symptom:** Tool costs exceed the stated branch cap. **Competing causes:** child calls excluded from accounting, retries uncapped, or alternate path uses a different executor. **Diagnostic:** aggregate parent and descendant run IDs. **Fix:** propagate one budget ledger. **Verification:** nested request is denied once aggregate six-call budget is exhausted.

**Takeaway:** Each symptom needs a signal that distinguishes plausible causes before an irreversible mitigation.


### Expanded diagnostic investigations
These are design investigations and expected assertions; none reports an observed incident or executed test.

#### F-06 cache-key omission
**Trigger/environment:** A shared retrieval cache omits tenant in its key.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Compare cache key components with authorization decision.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Include tenant, principal scope, source revision, and policy epoch.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Two tenants requesting the same document label must receive isolated handles.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-07 stale policy cache
**Trigger/environment:** A worker uses an allow decision after entitlement revocation.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Compare policy decision time and revocation epoch.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Bound policy cache TTL and require online check for effects.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Revocation before action must yield deny.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-08 schema coercion
**Trigger/environment:** A parser converts a string amount into an unintended numeric amount.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Log raw input, canonical parsed value, and validation result.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Use strict typed parsing and range checks.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Malformed or ambiguous amount must be rejected.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-09 confused deputy
**Trigger/environment:** A privileged adapter uses its own authority without checking caller scope.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Compare effective service identity with delegated principal.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Require action-level authorization on delegated principal.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Unentitled caller cannot cause privileged effect.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-10 approval drift
**Trigger/environment:** An approved action is edited after review.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Recompute canonical digest against approval digest.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Bind approval to digest, expiry, and policy version.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Changed resource or quantity requires new approval.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-11 retry storm
**Trigger/environment:** Independent clients retry the same failed run.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Count attempts per effect key and caller.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Centralize retry budget and apply backoff.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Aggregate attempts cannot exceed run budget.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-12 orphan worker
**Trigger/environment:** A worker continues after the run is terminal.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Check terminal version before every effect boundary.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Fence workers and cancel descendants.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Terminal run admits no later effect.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-13 replay nondeterminism
**Trigger/environment:** Replay consults live mutable context and chooses a different action.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Compare recorded inputs/model revision with replay inputs.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Persist decision inputs or require reapproval.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Replay changes cannot create an effect without fresh authorization.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-14 receipt mismatch
**Trigger/environment:** A receipt belongs to another tenant or operation.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Check receipt correlation key, tenant, and canonical action.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Validate receipt binding before finalization.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Wrong-key receipt never completes the run.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-15 audit gap
**Trigger/environment:** The adapter executes but records no decision ID.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Join sink request ID to adapter audit trail.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Make audit append part of adapter admission.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Every accepted effect has traceable decision evidence.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-16 budget aliasing
**Trigger/environment:** Child agents receive fresh instead of inherited budgets.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Aggregate lineage usage by root run ID.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Propagate one immutable budget ledger.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Child calls count against root cap.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-17 unsafe compensation
**Trigger/environment:** A cancellation handler blindly reverses a completed effect.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Classify effect reversibility and receipt status.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Model compensation as a new authorized operation.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Unknown original outcome forbids automatic compensation.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-18 secret leakage
**Trigger/environment:** Prompt logs retain credentials from tool output.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Inspect redaction coverage and access logs.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Separate secret references from content and redact before persistence.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Known secret sentinel never appears in retained trace.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-19 routing loop
**Trigger/environment:** Supervisor delegates in a cycle.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Track visited routing states and remaining hop budget.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Set termination predicate and cycle detection.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Repeated state consumes budget then fails closed.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-20 partial authorization
**Trigger/environment:** Batch action checks only first record.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Enumerate per-record authorization outcomes.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Authorize each resource or use a verified set policy.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Unauthorized member prevents or excludes only itself by contract.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-21 clock ambiguity
**Trigger/environment:** Local clock determines expiry after suspension.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Compare authoritative time and observed skew.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Use server-side expiry evaluation.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Expired approval is rejected despite worker clock.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-22 model-version drift
**Trigger/environment:** A new model emits a changed action format.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Compare model/version output conformance tests.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Pin parser contract and stage rollout.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Unknown fields cannot broaden an action.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-23 context poisoning
**Trigger/environment:** Untrusted retrieved text is treated as instruction.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Label content provenance and inspect tool intent derivation.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Segregate untrusted text and enforce tools outside model text.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Injected text cannot alter adapter authorization.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-24 state migration
**Trigger/environment:** A checkpoint schema changes while runs are waiting.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Attempt migration on representative old versions.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Version state schema and provide migration or quarantine.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** Unknown schema resumes only through review.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.

#### F-25 queue starvation
**Trigger/environment:** One tenant consumes worker capacity.
**Observable symptom:** The run can appear successful while violating an ownership or authorization boundary.
**Competing causes:** implementation defect, stale configuration, or incomplete telemetry; do not attribute causality from one log line.
**Discriminating diagnostic:** Measure per-tenant queue age and admission counts.
**Expected signal:** A trace must expose tenant, run version, action digest, effect key, policy decision, and adapter outcome where relevant.
**Immediate mitigation:** Stop new effecting transitions for the affected run and preserve correlation identifiers.
**Correction:** Use quotas/fair scheduling with bounded priority.
**Prevention:** Add a contract test at the named boundary and review changes to identity, schema, policy, or state versions.
**Verification oracle:** No tenant exceeds configured concurrent-run share.
**Mitigation risk:** A broad stop can delay legitimate work; release only after the discriminating check passes.
**Provenance:** HYPOTHETICAL teaching scenario; source bodies unavailable.


### Boundary review drills
Each drill is a HYPOTHETICAL review protocol, not an incident report or executed result.

#### F-26 serialization failure
**Boundary at risk:** serialization failure can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-27 policy-version rollback
**Boundary at risk:** policy-version rollback can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-28 resource-name confusion
**Boundary at risk:** resource-name confusion can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-29 delegation token replay
**Boundary at risk:** delegation token replay can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-30 adapter timeout classification
**Boundary at risk:** adapter timeout classification can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-31 duplicate webhook delivery
**Boundary at risk:** duplicate webhook delivery can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-32 lost checkpoint write
**Boundary at risk:** lost checkpoint write can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-33 approval expiry race
**Boundary at risk:** approval expiry race can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-34 cross-region replica lag
**Boundary at risk:** cross-region replica lag can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-35 tool-result truncation
**Boundary at risk:** tool-result truncation can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-36 unsafe default arguments
**Boundary at risk:** unsafe default arguments can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-37 tenant deletion race
**Boundary at risk:** tenant deletion race can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-38 retention-policy mismatch
**Boundary at risk:** retention-policy mismatch can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-39 unbounded fan-out
**Boundary at risk:** unbounded fan-out can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-40 manual override without audit
**Boundary at risk:** manual override without audit can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-41 privilege escalation by path traversal
**Boundary at risk:** privilege escalation by path traversal can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-42 incomplete error normalization
**Boundary at risk:** incomplete error normalization can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-43 model-generated identifier collision
**Boundary at risk:** model-generated identifier collision can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-44 stale feature flag
**Boundary at risk:** stale feature flag can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-45 batch cancellation race
**Boundary at risk:** batch cancellation race can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-46 queue poison message
**Boundary at risk:** queue poison message can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-47 dead-letter replay without scope
**Boundary at risk:** dead-letter replay without scope can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-48 approval reviewer identity change
**Boundary at risk:** approval reviewer identity change can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-49 irreversible sink endpoint
**Boundary at risk:** irreversible sink endpoint can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-50 opaque third-party status
**Boundary at risk:** opaque third-party status can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-51 unexpected tool output encoding
**Boundary at risk:** unexpected tool output encoding can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-52 recursive reflection loop
**Boundary at risk:** recursive reflection loop can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-53 trace sampling gap
**Boundary at risk:** trace sampling gap can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-54 configuration split-brain
**Boundary at risk:** configuration split-brain can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

#### F-55 emergency bypass persistence
**Boundary at risk:** emergency bypass persistence can make a local success signal disagree with the authoritative policy, state, or external-effect record.
**Question:** Which component owns the fact that permits the next irreversible transition?
**Evidence to collect:** Correlate tenant, principal, run ID, expected version, policy version, action digest, effect key, adapter request ID, and receipt state.
**Competing explanation A:** A client/network observation is stale or incomplete.
**Competing explanation B:** A valid contract was evaluated against the wrong scope or version.
**Discriminating check:** Reconstruct the transition from durable records; do not infer it from model text or a worker log alone.
**Safe containment:** Freeze only new effecting transitions for the affected operation while allowing receipt reconciliation.
**Correction rule:** Make the authority check and durable record occur before the boundary, with an explicit failure state when the authority cannot be established.
**Prevention rule:** Add a negative contract test that changes one identity, version, argument, or timing condition and expects denial or a pending state.
**Expected oracle result:** No new effect key is created; stale versions/tokens are rejected; uncertainty remains visible until receipt evidence exists.
**Mitigation risk:** Overbroad blocking can create backlog and operator work, so release criteria must name the evidence that resolves the ambiguity.
**Source status:** OPEN; visible source titles establish syllabus scope only.

## 9. Complexity & Cost Analysis
A request trace stores at least one record per transition and tool attempt. If `n` is the number of attempts, append-only audit storage is `O(n)` records and reconstruction is `O(n)` without snapshots; a snapshot every `k` transitions bounds replay of recent state to `O(k)` while adding snapshot writes. This is an algorithmic statement, not a throughput benchmark.

For the illustrative six-call budget, six attempt events plus one admission, one checkpoint, and one terminal record give nine minimum audit events before receipts or model payloads. Retention cost depends on bytes/event, encryption, replication, and policy; no price is asserted. Do not retain raw prompts by default merely because a run is auditable.

**Takeaway:** Size audit and recovery from transition volume and retention policy, not from model-token intuition.

## 10. Key Theorems / Results Reference
- **Safety invariant (DERIVED):** If every accepted transition checks tenant, expected version, and current authorization, stale writers cannot overwrite the same versioned record.
- **At-least-once delivery consequence (DERIVED):** A lost response cannot distinguish “not received” from “received but response lost”; safe automatic retry requires an idempotency/reconciliation contract.
- **Least privilege (SOURCED):** NIST defines least privilege as granting only the minimum access needed. See [NIST glossary](https://csrc.nist.gov/glossary/term/least_privilege), accessed 2026-11-09; it does not validate this illustrative architecture.
- **Prompt injection boundary (SOURCED):** Untrusted content can manipulate model instructions; use it as a threat-model input, not authority. See [OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/), accessed 2026-11-09.

**Takeaway:** These results constrain a design; none establishes a vendor’s exact behavior.

## 11. Interview / Exam Questions
1. **Why is a session ID not authorization?** It locates state but can be guessed, replayed, or stale; authorize the authenticated principal on every protected read/action. *Follow-up:* cache policy only with a bounded invalidation contract.
2. **What does a timeout prove?** Only that the caller lacks a response; it does not prove the sink did not commit. Reconcile by operation key.
3. **Calculate the call cap for five branches and four calls each.** `5*4=20` attempts/run, under fixed independent caps. This is not an effect count.
4. **When does a deterministic workflow beat a planner?** When routes and actions are known and auditability/minimal surface matter more than adaptive decomposition.
5. **Why use a compare-and-swap version?** It turns concurrent stale writes into an explicit conflict. A lock lease without fencing can still allow an expired owner to write.
6. **A tool schema validates. Why can execution still be denied?** Schema validity says arguments have shape; authorization decides whether this principal, resource, and time are permitted.
7. **What state must precede an effect?** A durable intent/effect key and enough canonical arguments to reconcile; otherwise recovery cannot know which operation was intended.
8. **How would you test cross-tenant isolation?** Create identical session labels in two tenants, attempt cross reads and tool calls, and expect deny/isolation in traces.
9. **What changes if the sink has no receipt query?** Automatic retry is unsafe after ambiguity; choose manual reconciliation, a compensating process, or a different integration.
10. **What is a plausible wrong fix for duplicate effects?** Retrying with a new request ID; it removes correlation and can create a second business operation.
11. **Why is a lease not sufficient for exclusive execution?** A paused owner can resume after expiry. A receiver-enforced fencing token rejects the stale owner. *Follow-up:* if the external sink cannot accept fencing, isolate the effect behind one serialized adapter or do not claim exclusivity.
12. **The clock may skew by four seconds. What is the safe decision?** Do not use local expiry as a write permission. Obtain/renew ownership through the authority and require the receiver to reject older fencing tokens. *Red flag:* “make the lease five seconds longer” without considering delayed workers.
13. **What does cancellation mean after `EFFECT_SUBMITTING` is persisted?** It blocks future effects but does not establish that the current one failed. Reconcile K1 and record a terminal state that preserves both cancellation and receipt outcome.
14. **How should an approval change when a ticket’s priority changes?** Canonicalize all effect-relevant fields, compute a new digest, and require a new authorization if it differs. Reusing approval for a paraphrased action is a binding failure.
15. **A trace shows only service-account identity. What can you conclude?** Which service called the sink, not which user or delegated policy justified the action. Inspect the delegation record, tenant, principal, and policy decision before accepting the audit trail.

**Takeaway:** Strong answers name the invariant, the uncertainty, and the observable test.

## 12. Quick Reference Card
| Situation | Rule | Critical assumption | Avoid | Evidence |
|---|---|---|---|---|
| Model proposes action | Treat as untrusted intent | Adapter validates independently | Model-as-authority | DERIVED |
| Retry after timeout | Query/reconcile same effect key | Sink exposes receipt or deduplication | New key retry | DERIVED |
| Concurrent run update | CAS expected version | Atomic versioned store | Last-write-wins | DERIVED |
| Tenant state read | Authorize at read/action | Identity propagation is intact | Session-ID-only checks | DERIVED |
| Planner expansion | Enforce aggregate call/token budget | Descendants report usage | Per-agent uncapped counters | ILLUSTRATIVE |
| Tool action | Bind canonical arguments to policy | Adapter controls execution | Free-form broad capability | SOURCED/DERIVED |
| Source boundary | Visible filenames are syllabus only | Source bodies unavailable | Attributing unseen claims | OPEN |
