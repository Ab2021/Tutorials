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

### 7.1 Ten-layer stack review
The visible source title “ten layer stack model” is a syllabus commitment, not evidence for a vendor diagram. This deliberately logical stack separates responsibilities that may be co-located in deployment but must remain distinguishable in review.

#### Layer 1 — Caller and interaction contract
**Owns:** the user-visible request, input limits, response channel, and correlation handle.
**Does not own:** a durable run, authorization decision, or business effect.
**Input contract:** request bytes, client authentication material, requested operation, and client-generated correlation ID.
**Output contract:** acceptance/rejection or a run reference; never an invented claim that an asynchronous effect completed.
**Boundary check:** disconnecting the client must not erase an admitted run, and reconnecting must not grant access by correlation ID alone.
**Review question:** can a different authenticated principal observe or resume this interaction?

#### Layer 2 — Identity and tenancy
**Owns:** authenticated principal, tenant binding, delegation chain, and credential audience validation.
**Does not own:** the model's proposal or the sink's business receipt.
**Input contract:** verified credential plus transport context; unverified identity claims are data, not identity.
**Output contract:** a narrow principal/tenant/delegation assertion with expiry and provenance.
**Boundary check:** a session identifier is rejected where a principal assertion is required.
**Review question:** is the effective service identity recorded separately from the original user identity?

#### Layer 3 — Admission and policy control plane
**Owns:** quota, policy version, run creation, budget reservation, and audit decision.
**Does not own:** worker process liveness or external-system truth.
**Input contract:** principal, tenant, requested workflow, current policy, and declared limits.
**Output contract:** accepted run ID with policy/budget version, or a denial with a bounded reason.
**Boundary check:** no runtime work begins merely because a model or caller requested it.
**Review question:** can the control plane reject a request before any tool-capable context is assembled?

#### Layer 4 — Session and durable state
**Owns:** session scope, run version, checkpoint, lifecycle state, and parent/child lineage.
**Does not own:** authorization validity after its recorded decision expires.
**Input contract:** versioned transition request from an authorized runtime owner.
**Output contract:** one accepted next version or a conflict; never silent last-write-wins.
**Boundary check:** a restarted worker can reconstruct legal next work from this record without trusting RAM.
**Review question:** is every mutable fact assigned one durable writer or a declared conflict rule?

#### Layer 5 — Context assembly
**Owns:** allowed source selection, ordering, budget allocation, provenance, and truncation declaration.
**Does not own:** durable memory truth, user authority, or tool permission.
**Input contract:** scoped source handles, access decisions, token budget, and requested task.
**Output contract:** bounded context with source IDs, revisions, trust labels, and omitted-material record.
**Boundary check:** retrieval content cannot create a new principal scope or tool capability.
**Review question:** can a reviewer tell what was omitted when the context budget is exhausted?

#### Layer 6 — Model and inference service
**Owns:** transforming provided context into a proposal under a pinned model/configuration identity.
**Does not own:** policy interpretation, durable run transition, or external effect.
**Input contract:** bounded prompt/context and constrained output grammar where applicable.
**Output contract:** proposed text or structured action candidate plus model/template revision.
**Boundary check:** model output remains untrusted even when syntactically valid.
**Review question:** does a model change alter a separately tested parser or action contract?

#### Layer 7 — Planner and workflow runtime
**Owns:** legal node progression, branch budget, checkpoint timing, cancellation observation, and retry classification.
**Does not own:** final authorization or receipt from an external sink.
**Input contract:** current run version, proposal, permitted capabilities, and remaining budget.
**Output contract:** next state transition, bounded child work, or an action candidate for validation.
**Boundary check:** child work inherits root budget, tenant, and cancellation scope rather than minting them.
**Review question:** after a crash, which node can rerun and which must reconcile before rerun?

#### Layer 8 — Tool schema and validation adapter
**Owns:** typed parsing, defaults policy, canonicalization, range checks, and structured error mapping.
**Does not own:** a broad implicit right to perform every schema-valid action.
**Input contract:** candidate action and declared tool version.
**Output contract:** canonical action or a machine-readable validation failure.
**Boundary check:** unknown fields, duplicate keys, unsafe coercions, and incomplete results are explicit outcomes.
**Review question:** are review and execution using the same canonical representation?

#### Layer 9 — Authorization and effect gateway
**Owns:** action-level policy evaluation, approval binding, effect-key reservation, delegated identity enforcement, and audit append.
**Does not own:** the third party's eventual persistence after the network breaks.
**Input contract:** canonical action, current principal/delegation, policy revision, and effect key.
**Output contract:** deny, authorized submission record, or a reconciliation-required ambiguity record.
**Boundary check:** a schema-valid model proposal cannot bypass current policy or a required approval.
**Review question:** which evidence identifies the exact action that the approver authorized?

#### Layer 10 — External system of record
**Owns:** the authoritative business effect and its queryable receipt or status semantics.
**Does not own:** the agent run's narrative of what happened.
**Input contract:** adapter-specific canonical request, identity, and idempotency/correlation key.
**Output contract:** receipt, rejection, or an ambiguous transport outcome whose state must be queried.
**Boundary check:** a local transaction cannot prove remote commit.
**Review question:** what is the recovery path when this system cannot deduplicate or query by key?

### 7.2 Request-path proof obligations
For one effecting request, collect evidence in this order: authenticate the caller; bind tenant; admit a run; record policy and budget; assemble permission-filtered context; pin proposal provenance; validate a typed action; authorize its canonical form; reserve one effect key; submit; reconcile a receipt; then finalize the run. Each clause answers a different question. Combining them into “the agent did it” destroys both recovery and auditability.

| Transition | Required durable evidence | Unsafe substitute | Recovery decision |
|---|---|---|---|
| caller to identity | principal, tenant, credential expiry | session ID | reauthenticate |
| identity to admission | policy decision and budget reservation | model output | deny or create a new run |
| admission to context | source handles and access checks | transcript copy | rebuild from permitted sources |
| context to proposal | model/template revision and prompt reference | chat message | regenerate as a new proposal |
| proposal to action | canonical typed arguments | free-form prose | return validation error |
| action to authority | current decision, approval digest, expiry | old approval | deny or reapprove |
| authority to submission | effect key and submitting state | HTTP request log | reconcile same key |
| submission to finality | sink receipt/status | timeout outcome | pending/manual resolution |

### 7.3 Builder checklist with executable oracles
1. **Name the tenant boundary.** Oracle: equal session labels in two tenants never return each other’s source handles.
2. **Name the principal at every effect.** Oracle: an adapter audit row contains original principal and effective service identity.
3. **Version every run transition.** Oracle: two writers expecting version 7 produce one accepted write and one conflict.
4. **Fence stale workers.** Oracle: token 41 is rejected after token 42 has been accepted for the same protected resource.
5. **Make cancellation durable.** Oracle: restart after `CANCEL_REQUESTED` permits no new effect reservation.
6. **Classify timeouts as ambiguity.** Oracle: lost response produces receipt lookup or `RECONCILIATION_PENDING`, never an invented failure.
7. **Bind approval to canonical arguments.** Oracle: modifying quantity, resource, or policy version invalidates prior approval.
8. **Reserve the effect key before sending.** Oracle: crash after reservation leaves one recoverable operation identity.
9. **Constrain child work.** Oracle: descendants cannot exceed the parent’s aggregate call and spend cap.
10. **Label context trust.** Oracle: retrieved instructions cannot grant a capability absent from policy.
11. **Pin execution-critical manifests.** Oracle: a changed tool schema cannot resume an incompatible checkpoint unchecked.
12. **Separate trace from audit.** Oracle: a sampled-out trace still has a minimal durable effect-journal record.

**Takeaway:** The ten layers are not ten microservices; they are ten independently reviewable ownership contracts along one request path.

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

### Boundary investigation playbook
The five short entries above identify the first symptoms to triage. The following investigations are deliberately different failure mechanisms rather than a numbered catalogue of variations on one generic procedure. All are HYPOTHETICAL design exercises. They specify what an executable fault-injection harness should assert; they do not report incidents or test runs.

#### F-06 — Policy-cache revocation race
**Trigger:** A user loses entitlement after a worker cached an allow decision.
**Competing causes:** an expiry bug, a revocation-feed delay, or the action being authorized against service rather than delegated identity.
**Discriminating signal:** compare the authorization decision's policy epoch and expiry with the revocation epoch at adapter admission. A valid old cache entry is not a valid action decision.
**Containment:** hold only actions requiring the revoked scope; reads already permitted by a separately scoped policy are not evidence of action permission.
**Correction:** make the adapter evaluate a current policy decision for effecting calls, and bind its decision ID and epoch to the action record.
**Prevention:** exercise “allow at epoch 20, revoke at epoch 21, submit at epoch 21”; the expected outcome is deny, not a cache hit.
**Risk:** forcing every harmless read through a remote decision point can create an availability dependency; classify effecting and non-effecting capabilities separately.

#### F-07 — Canonicalization ambiguity
**Trigger:** The review UI displays `1,000` while the adapter parses a locale-specific quantity or silently applies a default currency.
**Competing causes:** parser coercion, omitted fields in the digest, or UI/server serialization mismatch.
**Discriminating signal:** retain raw arguments, parsed typed arguments, canonical bytes, and the digest presented to the reviewer. The same visible form must yield one typed action.
**Containment:** reject ambiguous numbers, missing currency, duplicate JSON fields, and unknown effecting fields rather than guessing.
**Correction:** use one versioned canonicalizer at review and execution; include all effect-relevant fields in its schema and digest.
**Prevention:** oracle inputs `amount="01"` and `amount=1` may be equivalent only if the declared schema says so; `"1,000"` without locale must fail.
**Risk:** strict rejection can increase human rework, so return a bounded validation error identifying the field instead of silently normalizing it.

#### F-08 — Confused-deputy adapter
**Trigger:** A highly privileged integration uses its service account to perform an action requested by a caller lacking that scope.
**Competing causes:** absent delegated-principal check, an overbroad service role, or an incorrectly mapped resource scope.
**Discriminating signal:** inspect the sink request alongside the authorization record. It must identify both the service that executed and the principal whose scope was evaluated.
**Containment:** disable the affected effecting method, not all tool reads, while the delegation path is repaired.
**Correction:** authorize `(principal, tenant, action, resource)` at the adapter; service identity supplies execution identity, never caller entitlement by itself.
**Prevention:** a caller authorized for project A must be denied when it asks the same adapter to modify project B even though the service can access both.
**Risk:** token exchange may reduce availability during identity-provider failures; define its failure mode before relying on it.

#### F-09 — Approval-digest drift
**Trigger:** An action is approved, then its resource, quantity, priority, policy version, or expiry changes before execution.
**Competing causes:** a UI update failed to create a new proposal, a digest omitted a field, or a worker resumed an old checkpoint.
**Discriminating signal:** recompute the adapter's canonical digest and compare it with the stored approval digest and policy version.
**Containment:** place the run in `AWAITING_REAPPROVAL`; do not execute the edited request under a superficially similar approval.
**Correction:** bind approval to canonical action, principal, tenant, policy version, effect key, and server-evaluated expiry.
**Prevention:** changing `priority=normal` to `priority=urgent` must generate a different digest and a new reviewer decision.
**Risk:** overly broad binding can force reviews for presentation-only changes; keep display metadata out of the canonical effect contract.

#### F-10 — Parent-budget aliasing
**Trigger:** Each delegated worker receives a fresh token, tool-call, or spend counter.
**Competing causes:** omitted lineage propagation, retries counted outside the budget, or a fallback executor not reporting usage.
**Discriminating signal:** aggregate attempts and reserved cost by immutable root-run ID, including descendants and reconciliation calls.
**Containment:** stop new delegates when the root ledger is exhausted; allow receipt lookup required to resolve already reserved effects.
**Correction:** create one durable root budget with atomic reservations; child limits are partitions, not independent allowances.
**Prevention:** a root cap of six and two children each requesting four calls must admit at most six aggregate reservations.
**Risk:** a single shared cap can starve a high-value recovery step; reserve a separately bounded recovery allowance in the admission policy.

#### F-11 — Replay against mutable context
**Trigger:** A crash replay reassembles context from a newer document, model, or feature flag and selects a different action.
**Competing causes:** missing input snapshot, intentionally live retrieval, or non-deterministic model output mistaken for state recovery.
**Discriminating signal:** compare recorded source revisions, prompt/template revision, model identifier, and retrieved handles with the replay inputs.
**Containment:** prevent effect execution from a semantically changed replay until it produces a new proposal and authorization.
**Correction:** checkpoint decision inputs or explicitly classify the step as recomputable and require revalidation after recomputation.
**Prevention:** update a retrieved policy document between attempt one and replay; expected result is re-review or a visible changed-input state.
**Risk:** immutable snapshots can retain data longer than desired; store references and access checks where policy permits rather than raw content by default.

#### F-12 — Receipt correlation mismatch
**Trigger:** A delayed receipt is attached to the wrong tenant, effect key, or canonical action.
**Competing causes:** correlation-key collision, adapter routing defect, or a sink receipt API that returns insufficient identity fields.
**Discriminating signal:** validate tenant, effect key, action digest or sink request ID, and expected operation type before finalizing the run.
**Containment:** move the run to `RECONCILIATION_PENDING` and preserve both candidate receipts for operator resolution.
**Correction:** require the adapter to persist a unique correlation contract before submission and to reject receipts that do not bind back to it.
**Prevention:** provide receipt `K2` to a run reserved with `K1`; expected result is no terminal success and an auditable mismatch.
**Risk:** a strict match may expose a deficient third-party API; then automatic terminal success is unavailable, not a reason to weaken matching.

#### F-13 — Cross-tenant cache collision
**Trigger:** Retrieval, tool-result, or session cache keys omit tenant or effective permission scope.
**Competing causes:** incomplete key construction, background worker reuse, or authorization performed only when a session was created.
**Discriminating signal:** log a privacy-safe cache-key decomposition and the access decision used for the cached object; test identical user-visible labels in two tenants.
**Containment:** invalidate affected shared entries and switch to uncached authorized reads until the key and invalidation rule are corrected.
**Correction:** scope cache identity by tenant, resource revision, principal or permission class where necessary, and policy epoch.
**Prevention:** tenant T1 writes `guide`; T2 requests `guide`; the oracle is T2's own permitted result or deny, never T1's bytes.
**Risk:** finer keys reduce cache hit rate; measure that cost instead of relaxing isolation by assertion.

#### F-14 — Cancellation/effect race
**Trigger:** Cancellation commits after the runtime reserves an external effect but before a receipt returns.
**Competing causes:** worker ignores new run version, adapter accepts stale run version, or cancellation is implemented only in process memory.
**Discriminating signal:** reconstruct ordered durable transitions and inspect whether the adapter checked the current cancellation/version fence before submission.
**Containment:** block follow-on effects and reconcile the reserved effect key; do not claim that cancellation undid an already submitted operation.
**Correction:** make cancellation a versioned durable transition and make adapters reject stale run versions for new effects.
**Prevention:** use the Section 5 oracle: cancellation after `EFFECT_SUBMITTING(K1)` plus receipt `ticket-812` ends `CANCELLED_AFTER_EFFECT`.
**Risk:** aggressively aborting adapters can obscure an in-flight commit; ambiguity must remain represented until a receipt or bounded escalation resolves it.

#### F-15 — Queue fairness collapse
**Trigger:** A noisy tenant fills a shared queue, delaying another tenant's approval or reconciliation work.
**Competing causes:** no per-tenant admission cap, priority inversion, or workers pinned on slow tools rather than queue policy itself.
**Discriminating signal:** compare queue age, queued count, admitted concurrency, and effecting versus reconciliation lane by tenant.
**Containment:** enforce tenant concurrency and reserve capacity for reconciliation; never drop an effect reservation merely to make a dashboard look healthy.
**Correction:** use explicit fair-queue or weighted scheduling semantics, with a documented bounded-priority rule for safety or recovery work.
**Prevention:** inject one tenant with ten long runs and a second with one receipt lookup; the lookup's maximum wait must be evaluated against its declared lane policy.
**Risk:** priority lanes can be abused; admission must classify by server-side contract rather than model-provided labels.

#### F-16 — Checkpoint schema incompatibility
**Trigger:** A waiting run is resumed after its state or tool schema changes.
**Competing causes:** unversioned checkpoint, lossy migration, or a changed tool contract accepted as if compatible.
**Discriminating signal:** compare checkpoint schema/version, transition name, action digest, and tool schema revision against the deployed runtime manifest.
**Containment:** quarantine incompatible runs with an explicit recovery owner; do not deserialize unknown fields into defaults for an effecting transition.
**Correction:** version checkpoint schemas, test forward/backward migrations, and define a manual migration or expiration policy.
**Prevention:** resume a version-1 checkpoint under a version-2 runtime that requires a new field; expected result is migration or review, never a default external call.
**Risk:** retaining old runtime readers costs operational effort; deleting them without a recovery policy strands legitimate work.

#### F-17 — Secret-bearing trace retention
**Trigger:** A tool response or prompt trace includes an access token, password, or sensitive value and is copied to broad telemetry.
**Competing causes:** missing redaction, a new response field, or a debugging bypass that wrote pre-redaction data.
**Discriminating signal:** insert a controlled secret sentinel into a proposed harness and scan every persistence path, export, and support view for it.
**Containment:** restrict access, rotate exposed credentials according to the credential owner’s process, and preserve only the minimum forensic metadata.
**Correction:** keep secret references outside model context when possible; redact before persistence and enforce field-level logging contracts.
**Prevention:** the sentinel must be absent from traces while a correlation ID and redacted field marker remain available for diagnosis.
**Risk:** indiscriminate redaction can remove evidence needed to distinguish users; retain non-secret stable identifiers with access controls.

#### F-18 — Untrusted-content instruction crossover
**Trigger:** Retrieved text tells the model to override policy or call a tool, and that text is treated as executable authority.
**Competing causes:** prompt-injection susceptibility, tool schema overbreadth, or absent adapter authorization.
**Discriminating signal:** label provenance and trust class in assembled context, then trace whether the proposed action survives independent schema and policy checks.
**Containment:** prevent effecting transition; preserve the source handle for review without repeating its instructions into operator tooling.
**Correction:** segregate untrusted content, narrow tools, and enforce authorization outside the model's text channel.
**Prevention:** an injected document requesting a cross-project export must yield an adapter deny even if the model repeats the request.
**Risk:** treating all retrieved content as unusable defeats retrieval; the boundary is that content may inform a proposal but cannot grant authority.

#### F-19 — Non-idempotent compensation
**Trigger:** An operator or runtime attempts to “undo” an effect whose original outcome is unknown or whose reversal has different business semantics.
**Competing causes:** timeout misclassified as failure, assumed reversibility, or a compensation endpoint using the original effect key.
**Discriminating signal:** classify receipt state, reversibility, and compensating action contract before any reversal request.
**Containment:** leave the original operation pending reconciliation; route any compensating action through a separate authorized effect key.
**Correction:** model compensation as a new business operation with its own approval, audit record, and receipt.
**Prevention:** an unknown ticket creation may not automatically create a closing ticket; first query K1, then decide whether a new authorized action is warranted.
**Risk:** delayed compensation can inconvenience users; unsafe automatic compensation can create a second harmful effect.

#### F-20 — Tool-output truncation
**Trigger:** A tool result is truncated before the model or validator sees its decisive status field.
**Competing causes:** output-size limit, serialization error, or a tool returning a partial page without continuation metadata.
**Discriminating signal:** record declared result length, retained bytes, truncation flag, schema validation result, and continuation token presence.
**Containment:** prohibit effecting decisions that depend on an incomplete result; fetch a bounded continuation or ask for review.
**Correction:** design structured result contracts with completeness/status fields separate from large payloads and validate them at the adapter.
**Prevention:** a 10,000-byte result truncated before `status=DENY` must be classified incomplete, not interpreted as successful absence of an error.
**Risk:** increasing limits can inflate context cost and leakage exposure; select a status-first contract rather than indiscriminate expansion.

#### F-21 — Manual override without provenance
**Trigger:** An operator bypasses a failed workflow but the override lacks actor, scope, reason, expiry, and link to the blocked run.
**Competing causes:** emergency process gap, unavailable audit store, or UI-only override path.
**Discriminating signal:** audit query must join override ID to authenticated actor, canonical action, policy exception, and resulting receipt.
**Containment:** time-bound the override and require a second review for irreversible scope; do not silently retry the original run.
**Correction:** treat override as a first-class authorized transition with its own effect key and expiration.
**Prevention:** an override record missing a reason or target action cannot release a blocked external effect.
**Risk:** extra controls can slow genuine emergencies; define a narrow break-glass path with post-action review rather than an unaudited bypass.

#### F-22 — Path-normalization escape
**Trigger:** A file or repository tool authorizes a displayed path but resolves a different path through traversal, symlink, encoding, or case rules.
**Competing causes:** inconsistent canonicalization, filesystem-specific semantics, or authorization before resolution.
**Discriminating signal:** compare requested path, normalized path, resolved target identity, and authorized root under the actual execution environment.
**Containment:** disable write methods for the affected adapter and restrict reads to handles issued after resolution.
**Correction:** resolve against an allowlisted root before authorization and reject ambiguous encodings, traversal, and escaping symlinks.
**Prevention:** `docs/../secrets.txt` and a symlink leaving the workspace must be denied even when the initial string appears under `docs`.
**Risk:** strict normalization can reject legitimate platform-specific names; make supported filesystem semantics explicit in the contract.

#### F-23 — Stale feature-flag authority
**Trigger:** A runtime receives a flag snapshot enabling a capability after the server has disabled it.
**Competing causes:** cache propagation lag, client-side flag enforcement, or an old deployment still accepting the route.
**Discriminating signal:** attach evaluated flag revision and server decision to adapter admission; compare against the control-plane revision.
**Containment:** deny new effecting calls for the disabled capability while allowing status and reconciliation operations.
**Correction:** evaluate execution-critical flags at the enforcing boundary and bind rollout revision to the run record.
**Prevention:** disable `write_ticket` between proposal and adapter call; expected result is deny despite an earlier model-visible tool list.
**Risk:** strongly consistent flag reads can increase latency; only execution-critical flags require that path.

#### F-24 — Dead-letter replay loses authority
**Trigger:** A failed message is replayed days later using its old payload and a service identity, after user scope or policy changed.
**Competing causes:** replay treats transport durability as permission, original principal is missing, or expiry was ignored.
**Discriminating signal:** inspect message metadata for principal, tenant, authorization expiry, canonical action, and replay reason; then re-evaluate current policy.
**Containment:** quarantine expired or incompletely attributed messages rather than mass-replaying them.
**Correction:** define replay as a new admission decision that references, but does not inherit, prior authorization; preserve the original effect key for reconciliation only.
**Prevention:** replay a message after its approval expiry; expected result is reauthorization or review, not execution.
**Risk:** reauthorization can block time-sensitive recovery; use a narrowly authorized recovery role with explicit scope where policy permits.

#### F-25 — Trace sampling hides an effect
**Trigger:** Telemetry sampling drops the single span containing adapter admission or sink receipt.
**Competing causes:** head sampling, propagation break, or the adapter not emitting an auditable event at all.
**Discriminating signal:** compare durable effect journal counts to trace counts by effect key; audit completeness cannot depend on sampled observability.
**Containment:** use the effect journal and sink query for reconciliation; do not infer absence of an effect from a missing span.
**Correction:** keep minimal durable, access-controlled audit records for effect admission and receipts; sample bulky diagnostic payloads separately.
**Prevention:** a sampled-out trace with an accepted journal record must still allow reconstruction of principal, canonical action reference, decision ID, and receipt state.
**Risk:** full payload logging is expensive and unsafe; complete metadata is not a mandate to retain raw prompts or secrets.

#### F-26 — Split-brain configuration
**Trigger:** Two runtime pools use incompatible tool schemas or policy manifests during a partial rollout.
**Competing causes:** staged deployment overlap, missing manifest compatibility gate, or worker resumption on an old image.
**Discriminating signal:** compare run-pinned manifest ID with worker manifest, adapter schema version, and policy revision at every effect boundary.
**Containment:** stop cross-version effecting transitions and route compatible runs to their pinned worker pool or review queue.
**Correction:** publish immutable compatible manifests and define explicit migration/rollback rules for checkpoints and tool contracts.
**Prevention:** a run created under manifest M1 cannot execute an M2-only tool merely because a scheduler moved it to an M2 worker.
**Risk:** pinning reduces scheduling flexibility; a compatibility matrix and migration path are preferable to silent best-effort execution.

#### F-27 — Model-generated identifier collision
**Trigger:** The model invents an effect identifier instead of the runtime allocating one.
**Causes / signal:** weak ID generation or retry regeneration; two distinct canonical actions share one key in the effect journal.
**Correction:** allocate opaque, tenant-scoped keys at the runtime and bind each to one action digest.
**Oracle:** submit two actions with the same model text; their reserved keys differ, while retries of one action reuse its key.
**Risk:** a key without an action binding merely moves the collision problem.

#### F-28 — Partial batch authorization
**Trigger:** A batch tool checks entitlement only for its first resource.
**Causes / signal:** shortcut authorization or wildcard scope confusion; per-item decision evidence is absent.
**Correction:** authorize each member or use a formally defined all-or-nothing set policy.
**Oracle:** a batch containing one forbidden resource is denied or returns a declared partial result; it never silently effects that member.
**Risk:** per-item checks add latency and require a clear atomicity contract.

#### F-29 — Result pagination omission
**Trigger:** A planner decides from page one while a later page contains a conflict or deny status.
**Causes / signal:** missing continuation handling or a deceptive count; response metadata says `has_more=true`.
**Correction:** expose completeness and continuation in the tool schema; forbid a complete-set claim without traversing required pages.
**Oracle:** a relevant record on page two makes the tool result incomplete until page two is fetched.
**Risk:** exhaustive paging can exceed budget; choose and state a bounded sampling decision instead.

#### F-30 — Invalid capability discovery
**Trigger:** A stale capability list advertises a method removed from the gateway.
**Causes / signal:** cached discovery or rollout skew; schema revision differs at discovery and invocation.
**Correction:** version discovery and invocation contracts, and treat a mismatch as a retryable compatibility decision, not an alternate broad call.
**Oracle:** removed method discovery followed by invocation yields a structured unsupported-version response without effect.
**Risk:** refreshing discovery every call is costly; define a safe cache lifetime.

#### F-31 — Webhook signature scope gap
**Trigger:** A webhook is accepted because its body signature verifies but tenant, event type, or replay window is unchecked.
**Causes / signal:** incomplete signed fields or missing nonce; identical event ID is processed twice.
**Correction:** validate sender, timestamp, event identity, tenant binding, and replay protection before transition.
**Oracle:** a valid old signed payload outside the replay window is recorded as rejected.
**Risk:** narrow clock windows can reject delayed legitimate delivery; reconcile from source rather than widening indefinitely.

#### F-32 — Search-index revision drift
**Trigger:** Context retrieval cites a document handle whose indexed text no longer matches its source revision.
**Causes / signal:** asynchronous indexing or deleted source; source checksum and index revision disagree.
**Correction:** retain source revision in retrieval evidence and reauthorize/rebuild context when it changes.
**Oracle:** modifying a policy after indexing prevents old excerpts from supporting a new effecting action.
**Risk:** frequent refresh raises cost; critical sources need a tighter freshness rule than background knowledge.

#### F-33 — Summary overwrites transcript fact
**Trigger:** A lossy session summary replaces a qualified user statement with a stronger assertion.
**Causes / signal:** compression hallucination or unversioned overwrite; summary lacks pointers to supporting turns.
**Correction:** store summaries as derived, revisable records with provenance rather than as authoritative history.
**Oracle:** removing the source turn invalidates the summary claim and blocks its use as a permission fact.
**Risk:** retaining every turn increases privacy exposure; apply retention separately from truth labeling.

#### F-34 — Tool-result type confusion
**Trigger:** A string resembling a success receipt is treated as a typed receipt.
**Causes / signal:** permissive parser or model-mediated interpretation; required receipt fields are missing.
**Correction:** validate result schema at the adapter and distinguish receipt, pending, and error union variants.
**Oracle:** text `done` cannot finalize an effect without matching key and sink request identifier.
**Risk:** strict schemas need compatible evolution rules for real providers.

#### F-35 — Retry after permanent denial
**Trigger:** Runtime retries an authorization denial as if it were a transport timeout.
**Causes / signal:** flattened error taxonomy; repeated attempts share identical deny code and policy revision.
**Correction:** classify denial, validation, ambiguity, and transient transport states before scheduling retry.
**Oracle:** a deny at policy epoch 44 consumes no retry budget and requires changed input or policy.
**Risk:** some denials are temporary quota limits; their structured code must say so.

#### F-36 — Human review queue impersonation
**Trigger:** A reviewer sees an action card detached from its requester identity or tenant.
**Causes / signal:** UI join defect or shared queue; approval record lacks reviewer-visible binding fields.
**Correction:** display and persist canonical action, tenant, requester, scope, expiry, and policy basis together.
**Oracle:** changing tenant after review invalidates approval even when all natural-language text is identical.
**Risk:** excessive detail can expose sensitive data; use authorized redaction without omitting binding facts.

#### F-37 — Silent tool default
**Trigger:** An omitted argument becomes an effecting default such as “all resources.”
**Causes / signal:** server default evolution or schema mismatch; canonical action contains a value absent from proposal.
**Correction:** require explicit values for effecting scope and show server-applied defaults before approval.
**Oracle:** missing resource selector fails validation rather than expanding to a global scope.
**Risk:** required fields can burden harmless calls; classify read-only defaults separately.

#### F-38 — Async callback crosses run
**Trigger:** A delayed callback updates a newer run that reused a human-readable correlation label.
**Causes / signal:** non-unique correlation or missing version fence; callback run version differs from active record.
**Correction:** bind callbacks to immutable run and effect IDs and reject callbacks for terminal or mismatched versions.
**Oracle:** an old callback after a new run begins is quarantined and cannot alter the new run.
**Risk:** strict rejection requires a reconciliation channel for legitimately delayed callbacks.

#### F-39 — Orphaned child delegation
**Trigger:** Parent cancellation or failure leaves a child worker able to reserve new effects.
**Causes / signal:** missing lineage check or independent child lifecycle; child parent-version is stale.
**Correction:** make child admission check live parent state and propagate cancellation through a durable lineage record.
**Oracle:** canceling parent R1 denies a child R1.3 new effect reservation while allowing receipt lookup.
**Risk:** cascading cancellation can stop useful diagnostic work; exempt only explicitly non-effecting tasks.

#### F-40 — Context-window accounting error
**Trigger:** Token estimates omit tool schemas or reserved output, causing truncation of a safety instruction.
**Causes / signal:** tokenizer mismatch or hidden overhead; assembled token count exceeds declared capacity.
**Correction:** count serialized components with the serving tokenizer and reserve output before retrieval selection.
**Oracle:** adding one 800-token chunk beyond budget produces a deterministic omit/shorten decision with provenance.
**Risk:** conservative reserves reduce answer capacity; calibrate with measured serving behavior before changing them.

#### F-41 — Read capability becomes write path
**Trigger:** A supposedly read-only tool follows a URL, expands an archive, or triggers an external mutation.
**Causes / signal:** hidden side effect or misclassified tool; adapter cannot state effect classification.
**Correction:** classify execution surfaces by actual side effects and isolate ambiguous tools behind approval.
**Oracle:** tool contract review identifies whether a request can modify remote state before it enters a model tool list.
**Risk:** overclassification impairs useful automation; documented no-side-effect guarantees are the criterion.

#### F-42 — Undeclared cross-region replica
**Trigger:** Authorization read uses a stale replica while effect gateway writes to a primary.
**Causes / signal:** replica lag or inconsistent routing; decision timestamp predates revocation commit.
**Correction:** route execution-critical checks to an authority with declared consistency, or fence using policy epoch.
**Oracle:** revoke then submit through a lagging read path; gateway must deny or detect stale epoch.
**Risk:** stronger consistency increases latency and availability coupling.

#### F-43 — Review decision expiry race
**Trigger:** Approval is valid when displayed but expires while the adapter is submitting.
**Causes /signal:** client-side expiry check or clock skew; adapter time is after recorded expiry.
**Correction:** evaluate expiry server-side at the enforcement point and make the submit/authorization ordering explicit.
**Oracle:** an action at expiry plus one millisecond is denied or re-reviewed, never accepted by a worker clock.
**Risk:** tight expiries create avoidable review churn; choose duration from workflow evidence.

#### F-44 — Change-set scope expansion
**Trigger:** A coding agent’s reviewed diff differs from the files actually written.
**Causes / signal:** workspace mutation or unpinned base revision; execution manifest lists extra paths.
**Correction:** bind approval to repository revision, normalized patch, target paths, and test command contract.
**Oracle:** adding an unreviewed file after approval forces a new review before write or merge.
**Risk:** strict manifests complicate legitimate generated files; include a declared generation rule if needed.

#### F-45 — Sandbox escape assumption
**Trigger:** Runtime treats a container or process boundary as proof that secrets and network effects are impossible.
**Causes / signal:** permissive mounts, credentials, or egress; sandbox manifest permits an undeclared capability.
**Correction:** define the sandbox threat model, mount policy, network egress, identity, and resource limits explicitly.
**Oracle:** a test process cannot read a secret path or reach a forbidden host under the proposed profile.
**Risk:** a restrictive sandbox may prevent required integrations; grant narrow capabilities with separate audit.

#### F-46 — Prompt-template provenance loss
**Trigger:** A run records model ID but not the system/template revision that shaped its proposal.
**Causes / signal:** unversioned prompts or dynamic server injection; replay cannot reconstruct input contract.
**Correction:** version prompt assets and record their digest/reference with the proposal.
**Oracle:** two templates with equal model ID yield distinguishable proposal provenance records.
**Risk:** logging template text may leak policy content; retain an access-controlled digest and immutable reference.

#### F-47 — Error message instruction injection
**Trigger:** A tool error includes untrusted text that the model treats as a command to broaden its next call.
**Causes / signal:** error channel is merged with trusted orchestration instructions; provenance label is absent.
**Correction:** represent errors as typed data and preserve trust class through retries and reflection.
**Oracle:** an error saying “retry with admin mode” cannot alter adapter authorization or tool scope.
**Risk:** hiding errors impairs repair; expose bounded fields, not executable instructions.

#### F-48 — Cost reservation release bug
**Trigger:** A failed reservation is never released, or a completed charge is released twice.
**Causes / signal:** non-idempotent accounting transition; ledger balance fails reconciliation with attempt states.
**Correction:** use versioned reservation, capture, and release states keyed to the run/effect operation.
**Oracle:** repeating release for one failed attempt leaves available budget unchanged after the first release.
**Risk:** delayed settlement means “available” is a policy estimate; state that uncertainty.

#### F-49 — Evaluation leakage into live context
**Trigger:** Grading labels, hidden answers, or test fixtures appear in a production context retrieval path.
**Causes / signal:** shared index namespace or source-tag omission; retrieved handle is marked evaluation-only.
**Correction:** isolate evaluation corpora and enforce source-class filters at context assembly.
**Oracle:** a live run query cannot retrieve a hidden benchmark answer even with matching lexical text.
**Risk:** isolation can reduce realistic test fidelity; use controlled copies with explicit access gates.

#### F-50 — Regional data residency bypass
**Trigger:** A tool adapter sends content to a region outside the run’s declared boundary.
**Causes / signal:** failover route or provider default; execution region differs from policy-approved region.
**Correction:** attach residency constraint to action admission and enforce it at egress selection.
**Oracle:** simulated preferred-region outage yields deny or approved alternative policy, never silent reroute.
**Risk:** strict residency reduces availability; the trade-off belongs in admission policy, not worker improvisation.

#### F-51 — Retrieval permission TOCTOU
**Trigger:** A source is authorized during search but revoked before its chunks are assembled.
**Causes / signal:** separate authorization phases or stale handles; access epoch changed between search and read.
**Correction:** authorize at content read/assembly and record the decision epoch with each included source.
**Oracle:** revocation between retrieval and assembly excludes the chunk from final context.
**Risk:** repeated checks cost latency; batch only within an explicit, short consistency boundary.

#### F-52 — Duplicate incident escalation
**Trigger:** Several agents open parallel escalations for one failing effect.
**Causes / signal:** no incident key or child agents cannot see existing escalation; multiple tickets share effect key.
**Correction:** reserve a deterministic escalation key derived from the affected operation and failure class.
**Oracle:** three concurrent ambiguity detectors produce one escalation record with three observations.
**Risk:** overdeduplication can merge distinct failures; include tenant, operation, and bounded time window.

#### F-53 — Unbounded reflection loop
**Trigger:** Critique/replan cycles continue after evidence stops improving.
**Causes / signal:** missing termination predicate or quality metric; iteration count exceeds configured cap.
**Correction:** set loop budget, stopping evidence, and a fail-closed handoff state before enabling reflection.
**Oracle:** unchanged critique output at the cap ends in review or failure, never another tool reservation.
**Risk:** low caps may truncate useful repair; tune from measured tasks, not a universal number.

#### F-54 — Source deletion after memory write
**Trigger:** A durable memory remains retrievable after its underlying source is deleted or access is revoked.
**Causes / signal:** detached embedding or missing deletion propagation; memory provenance points to unavailable source.
**Correction:** retain source linkage, apply deletion/revocation propagation, and recheck permission at retrieval.
**Oracle:** deleting a source makes related memory unavailable or visibly stale according to retention policy.
**Risk:** eager deletion may remove independently valid facts; define derivation and ownership rules.

#### F-55 — Incomplete terminal-state taxonomy
**Trigger:** Runtime collapses denied, failed, cancelled, ambiguous, and succeeded outcomes into one “done” state.
**Causes / signal:** UI simplification or event mapping loss; terminal record lacks a receipt/decision classification.
**Correction:** model distinct terminal and nonterminal reconciliation states with permitted next actions.
**Oracle:** lost response ends `RECONCILIATION_PENDING`; it cannot be rendered as success or failure without a receipt.
**Risk:** more states complicate operations; a simpler label may be presented only if it links to the authoritative state.

**Takeaway:** Diagnosis starts from the specific authority, state, or evidence contract that failed; it does not start by applying the same “stop everything and retry” script to every symptom.


### Boundary trace exercises
Each exercise is ILLUSTRATIVE and hand-derived. It is an executable-oracle specification, not evidence that a provider or production system was tested.

#### Trace A — Admission before context
**Precondition:** principal `P1` requests tenant `T1`; policy denies the requested workflow.
**Input:** `request_id=R1`, `tenant=T1`, `principal=P1`, `workflow=export`.
**Transition:** identity verification succeeds; admission evaluates current policy before any retrieval.
**Invariant:** a denied request has no run checkpoint, context-source list, tool reservation, or effect key.
**Expected output:** `DENIED(policy_reason)` with correlation R1.
**Negative control:** assembling context before this decision makes unauthorized source access possible.
**Recovery:** user changes scope or an administrator changes policy; the next attempt is a new admission.
**Decision:** do not treat model availability as a reason to bypass an admission denial.

#### Trace B — Compare-and-swap collision
**Precondition:** run R2 is `READY` at version 7.
**Input:** W1 and W2 each submit `READY(7) -> RUNNING(8)`.
**Transition:** W1 commits first; W2 receives a version conflict.
**Invariant:** exactly one successor version exists for every `(run_id, version)` pair.
**Expected output:** W1 accepted at version 8; W2 reloads instead of overwriting.
**Negative control:** last-write-wins can erase W1’s cancellation or checkpoint.
**Recovery:** W2 reads version 8 and only proposes a legal transition from that state.
**Decision:** conflict is useful evidence, not an exceptional success path to hide.

#### Trace C — Receipt-less timeout
**Precondition:** R3 reserved effect key K3 and the sink accepts no status query.
**Input:** adapter sends K3; client transport times out.
**Transition:** runtime cannot observe a receipt and cannot prove non-commit.
**Invariant:** timeout changes knowledge only; it does not establish sink outcome.
**Expected output:** `RECONCILIATION_REQUIRED(K3)` and no replacement key.
**Negative control:** creating K3b produces a second intended business operation.
**Recovery:** manual sink investigation or a separately approved compensating process.
**Decision:** an integration without deduplication/query semantics limits safe automation.

#### Trace D — Approval expiry
**Precondition:** canonical action D4 has approval expiring at server time 25.
**Input:** worker reaches adapter at server time 26 with an otherwise matching digest.
**Transition:** adapter evaluates expiry using its authority clock.
**Invariant:** an approval is valid only for matching action and unexpired authority interval.
**Expected output:** `DENY_EXPIRED_APPROVAL`, no sink submission.
**Negative control:** accepting a worker-local time of 24 allows a stale actor to decide validity.
**Recovery:** regenerate or re-present the same canonical action for current approval.
**Decision:** clocks are part of the authorization contract, not merely telemetry.

#### Trace E — Parent cancellation
**Precondition:** root R5 owns child R5.2, which has not reserved an effect.
**Input:** cancellation commits on R5 before child admission.
**Transition:** child checks parent lifecycle and cancellation version at its effect boundary.
**Invariant:** a cancelled ancestor prevents descendants from initiating new effects.
**Expected output:** child records `CANCELLED_BY_ANCESTOR`; no effect key is reserved.
**Negative control:** cancelling only an in-memory worker leaves queued child messages live.
**Recovery:** non-effecting cleanup may run under an explicit recovery contract.
**Decision:** lineage is a state contract, not a label for observability alone.

#### Trace F — Retrieval revocation
**Precondition:** a search result points to source S6 at access epoch 10.
**Input:** access to S6 is revoked at epoch 11 before context assembly.
**Transition:** assembler rechecks permission when resolving the source handle.
**Invariant:** search discovery is not a durable grant to read source content.
**Expected output:** S6 excluded with a provenance record showing revocation.
**Negative control:** trusting the epoch-10 search cache leaks newly restricted text.
**Recovery:** rebuild context from sources still permitted at epoch 11.
**Decision:** permission filtering belongs at the content boundary as well as search.

#### Trace G — Manifest mismatch
**Precondition:** checkpoint R7 pins tool schema M1; assigned worker advertises M2.
**Input:** M2 changes an effecting field from optional to required.
**Transition:** runtime compares pinned and available compatibility records before resume.
**Invariant:** an incompatible contract cannot be repaired by defaulting missing effect fields.
**Expected output:** migration, compatible worker routing, or review hold.
**Negative control:** resume under M2 with a default field silently changes the action.
**Recovery:** use tested migration or explicitly re-propose/re-authorize under M2.
**Decision:** deployment rollback and workflow recovery need one shared manifest vocabulary.

#### Trace H — Fair reconciliation lane
**Precondition:** T8 has ten slow planning runs; T9 has one ambiguous submitted effect.
**Input:** both tenants enter a bounded worker pool.
**Transition:** scheduler applies its documented tenant and reconciliation lane policy.
**Invariant:** planned work cannot starve receipt reconciliation indefinitely.
**Expected output:** R9 receives service according to the stated maximum-wait policy.
**Negative control:** FIFO without tenant accounting can make safety recovery wait behind speculative work.
**Recovery:** increase measured capacity or alter documented weights; do not silently bypass policy.
**Decision:** queue policy is a correctness and customer-impact choice, not only utilization tuning.

**Takeaway:** A trace oracle states preconditions, a transition, an invariant, a negative control, and a recovery decision; a prose claim alone cannot test a boundary.


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
