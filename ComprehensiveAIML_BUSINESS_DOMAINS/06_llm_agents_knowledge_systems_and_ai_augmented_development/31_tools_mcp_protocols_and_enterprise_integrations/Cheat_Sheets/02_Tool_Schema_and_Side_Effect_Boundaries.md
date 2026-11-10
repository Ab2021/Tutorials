---
title: "Tool Schema and Side-Effect Boundaries"
topic: "Tool schema design, admission checks, canonical payloads, and side-effect recovery boundaries"
collection: "06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
file_id: "02"
prerequisites: ["01_model_context_protocol.md"]
owner_role: "curriculum maintainer"
source_index: "INDEX.md"
review_trigger: "protocol revision, executor change, authorization model change, or downstream receipt contract change"
---

# Tool Schema and Side-Effect Boundaries

A tool is a capability contract with a blast radius. The schema says which bytes are acceptable; the
boundary says which business effects are admissible, by whom, once, and with what evidence afterwards.
Those are four different questions and they are answered by four different components. Most agent
integration failures come from collapsing them: treating a syntactically valid call as permission,
treating a timeout as failure, treating a UI toast as a receipt, or treating a model's retry as a new
intention rather than the same one.

This sheet is about the seam between a model proposal and an irreversible external effect. It assumes
you already know how a tool is discovered and described at the protocol layer (see sheet 01) and asks
the harder operational question: when exactly does a proposal become an effect, and what evidence do
you hold afterwards? All workload figures below are ILLUSTRATIVE constructions used to make the
arithmetic inspectable; every formula is derived and every numeric substitution is recomputed by hand
in section 5. No deployed integration, vendor outage, or company result is asserted anywhere in this
file. The evidence grade is DERIVED for the invariants and ILLUSTRATIVE for the parameters.

## 1. File Header

**Scope.** The contract surface of a single tool: input schema, output schema, error taxonomy, effect
classification, canonicalization rules, idempotency key semantics, authorization placement, receipt
semantics, retry and reconciliation policy, audit fields, and versioning. The unit of analysis is one
tool invocation plus its recovery path, not a whole agent.

**Audience.** Engineers who own an executor, a connector, or a platform team that hosts tools for
agent use; reviewers who must sign off that a given write action is safe to automate; and analysts who
must explain after the fact why a duplicate appeared, why a call vanished, or why a retry storm took
out a downstream ERP.

**Outcome.** After this sheet you should be able to (a) classify any tool into an effect class,
(b) state the admission predicate that must hold before dispatch, (c) size the recovery lane and the
deduplication window from arrival and completion rates, (d) choose between hold, shed, and replay
under load, and (e) reject a proposed design whose schema is sound but whose authority or receipt
model is absent.

**Non-goals.** Transport framing, session negotiation, capability discovery payloads, model-side
planning quality, prompt injection as a general topic, and the internal consistency guarantees of any
particular vendor product. Those are adjacent topics; this sheet touches them only where they change
the side-effect boundary.

**Evidence status.** Invariants and derivations are DERIVED from the definitions and are marked with
FORMULA blocks. Numeric parameters (arrival rates, storage sizes, cost per review, error rates) are
ILLUSTRATIVE values chosen to make the arithmetic checkable; they are not measurements. No source body
was available when this sheet was written, so claims about a specific protocol specification are
deliberately stated as version-pinned pointers to recheck rather than as quoted requirements.

**Notation.** Rates are per second and written λ, μ; durations are seconds written t, T; counts are
plain integers. Money is in minor currency units with the currency named. Digests are abstract
functions written h(·); the concrete algorithm is a deployment choice and never part of the invariant.
Every symbol is defined at first use; section 7 of sheet 01 carries the cross-file symbol glossary.

**How to use it.** Read sections 2 and 3 once for vocabulary. Use section 4 when you are specifying a
new tool or arguing about a failure. Use section 5 as a template for the sizing memo that goes with a
launch review. Use section 8 as a checklist against a real trace, section 11 to prepare for review
questions, and section 12 as the two-minute reminder on an incident bridge.

## 2. Concept Map / Mental Model

The mental model is a four-gate corridor with an evidence ladder behind it. A model can walk to the
first gate on its own; each later gate must be opened by a component that does not trust the model.

```text
 model output (text)                     <- untrusted, may be attacker-influenced
        |
        v
 [G1] syntactic gate: input schema, types, bounds, enum membership
        |
        v
 [G2] semantic gate: business state, budget, period lock, duplicate business key
        |
        v
 [G3] authority gate: principal, tenant, delegation, resource scope, at-effect-time check
        |
        v
 [G4] effect gate: key/digest ledger, dispatch, downstream commit
        |
        v
 evidence: receipt state in {COMMITTED, REJECTED, UNKNOWN, NOT_FOUND} + audit record
```

The four gates are independent predicates, not a sequence of increasingly strict versions of the same
check. A payload can pass G1 and fail G3 because the shape was always legal but this principal may not
spend from that cost centre. A payload can pass G1, G2, and G3 and still be refused at G4 because the
same intended operation was already committed under the same key with a different digest. Confusing
these gates produces the classic wrong fix: loosening the schema to stop an authorization error, or
widening a permission to stop a duplicate error.

The second half of the model is the evidence ladder. There are four distinct facts a system can hold,
and they are not interchangeable:

- **Proposed.** The model emitted a call. This is an intention and carries no external truth.
- **Admitted.** Local gates passed and the operation was durably recorded under a key. Still no effect.
- **Dispatched.** Bytes left the process. The effect may or may not exist downstream.
- **Committed.** An authoritative system reports the effect in its own state, or a receipt says so.

Recovery decisions live on the boundary between *dispatched* and *committed*, which is exactly the
interval where the caller's own memory is least trustworthy. Everything in section 4 that looks like
formalism exists to make that interval explicit instead of assumed.

A third useful mental device is the **capability gradient**: at each gate you should be able to name
the principal whose credentials are used. If the same broad service credential is used at G2, G3, and
G4 for every tenant, then the gates are decorative and the tenant isolation is a naming convention. The
gradient should narrow: the planner sees a description, the admission service sees a scoped intent, the
executor sees a freshly minted, audience-restricted credential bound to one resource and one operation.

A fourth device is **distance to effect**. Rank tools by how far a single call moves the world and how
hard it is to walk back. A read of a price list is distance zero. Creating a draft invoice is one step
and reversible. Posting a journal entry may be reversible only inside an accounting period. Triggering
a payment is irreversible once the rail settles. Dispatching physical goods is irreversible once the
carrier scans. Controls should become denser as distance grows: at distance zero, caching and retries
are free; at distance three, every retry must be keyed, every dispatch must be preceded by a durable
intent record, and every unknown outcome must page a human rather than loop.

A fifth device is **the two clocks**. Event time is when the business fact happened; processing time is
when your system learned about it. A receipt with only processing time cannot answer "was this
operation already effective before the period closed?" and will silently allow a duplicate after a
reconciliation gap. Store both, and store the authority time used for the authorization decision, so a
reviewer can distinguish "policy denied" from "policy was checked too early".

## 3. Foundational Definitions

| Term | Meaning | Units / type | Boundary that matters |
|---|---|---|---|
| Tool | Named capability with an input schema, an output schema, and an effect class | versioned contract | a description, not a permission |
| Input schema | Accepted argument shape, types, bounds, enums | versioned JSON-schema-like record | validation only; says nothing about business legality |
| Output schema | Declared shape of success and typed-failure results | versioned record | without it callers guess outcome from HTTP status |
| Error taxonomy | Closed set of machine-readable rejection reasons | enum | free-text errors force callers to parse prose |
| Effect class | E0 read, E1 natural upsert, E2 keyed write, E3 irreversible, E4 fanout | enum | determines which controls are mandatory |
| Canonical payload | Deterministic projection of effect-relevant fields | byte string | must exclude display text, formatting, and floats |
| Digest h(x) | Hash of the canonical payload | fixed-width bytes | only as strong as field coverage, not hash width |
| Idempotency key k | Stable identifier of one intended business effect | opaque string, client-minted | not a request id, not a hash of the payload |
| Receipt | Authoritative record of what the sink decided for key k | durable state | not a client acknowledgement, not a log line |
| Admission | Decision to accept an operation for dispatch | boolean plus reason | precedes dispatch; revocation requires compensation |
| UNKNOWN | Outcome genuinely not yet knowable | durable enum member | not an exception, not a failure, not a retry trigger |
| Reconciliation | Procedure that resolves UNKNOWN by querying the sink | process | the only safe exit from UNKNOWN |
| Compensation | New, separately approved effect that offsets a committed one | operation | not a rollback; it can fail and can be illegal |
| Approval binding | Cryptographic or structural tie between a human decision and one exact payload | digest plus approver plus expiry | free-text approval binds nothing |
| Blast radius | Set of resources reachable by one credential or one call | count or scope set | decided by schema breadth and authority scope |
| λ (lambda) | Arrival rate of operations at the executor | operations/s | measured per tool, not per tenant average |
| μ (mu) | Steady-state completion rate of the executor | operations/s | completion, not acceptance |
| ρ (rho) | Utilisation λ/μ | dimensionless, 0≤ρ<1 | queueing explodes as ρ→1 |
| B | Backlog accumulated during an overload interval | operations | only grows while λ>μ |
| Dedup window W | Time during which a repeated key returns the stored receipt | seconds | must exceed the retry horizon plus human delay |
| TOCTOU gap | Interval between an authority check and the effect it authorises | seconds | nonzero whenever check and effect are separated |
| Retry horizon R | Maximum number of attempts a caller will make for one key | integer | bounds amplification, not correctness |

Six of these distinctions carry most of the practical weight, and each is worth stating in the form an
incident reviewer will ask for it.

**Schema is not policy.** A schema can say `amount` is a positive integer below 10,000,000 minor units.
It cannot say the amount must be within the requesting cost centre's remaining budget at execution
time, because that fact lives in mutable authoritative state that the schema author does not control.
Any design that tries to encode policy into schema bounds produces a schema that is either too tight
(rejecting legal edge cases) or a false sense of safety (accepting illegal ones). Policy evaluation
belongs to a component that reads current state and returns a typed reason.

**An idempotency key is not a request id.** A request id identifies one transport attempt; a retry gets
a new one. An idempotency key identifies one intended effect; a retry must reuse the old one. If the key
is derived from the request id, deduplication never fires. If the key is derived by hashing the payload,
then any payload edit silently becomes a new operation and the ledger cannot detect the collision the
design was supposed to catch. The correct construction is a random or monotonic identifier minted at
the moment the intent is formed, persisted write-ahead, and reused for every attempt of that intent.

**A receipt is not an acknowledgement.** An HTTP 200 from a gateway proves the gateway received bytes. A
receipt proves the sink decided. The gap between them is populated by proxies, retries, async workers,
and batch interfaces that accept work and process it minutes later. Recovery must therefore query a
store that the sink itself writes, or a store written in the same transaction as the sink's decision.
Querying your own outbox to learn whether the sink committed is a category error.

**UNKNOWN is a state, not an error.** Most integrations map every timeout to `failed` and every
`failed` to a retry. That mapping is the single largest source of duplicate effects in agent systems.
Timeouts are ambiguous by construction: the work may have committed and the response may have been lost.
A system that can represent UNKNOWN durably, and that has a reconciliation procedure with a known cost,
can be correct. A system that cannot represent it is forced to guess, and the guess is wrong in exactly
the cases that matter.

**Compensation is not rollback.** A rollback assumes the effect never became visible. Compensation
assumes it did and issues a second, opposite effect. The second effect has its own failure probability,
its own authorization requirements, and its own side effects on downstream systems that already consumed
the first. A refund after shipment does not unship goods; a reversing journal entry does not un-audit
the original posting. Designs that call compensation "undo" in a runbook reliably produce accounting
differences that nobody can explain later.

**Approval binds to a payload, not to an intention.** "Approve the vendor payment" approved in a chat
window binds nothing. A reviewer must approve a specific digest of the specific canonical payload, with
an expiry, and the executor must compare the digest of what it is about to send against the approved
digest. Where the comparison is skipped, the approval becomes a rubber stamp whose scope grows silently
with every field the model can still choose.

## 4. Core Formulas with Full Derivation

Section 4 is organised as one base case and then general cases that each remove a simplifying
assumption. Every block uses the same anatomy: the formula, the symbols with units, the constraint that
makes it valid, a substituted numerical example, and the interpretation a practitioner should take from
it. The section ends with the limits, because each of these formulas has a specific way of being wrong.

### 4.1 Base case

The base case asks the narrowest useful question: given an already-parsed call, when may the executor
dispatch it? Treat it as a mechanism block rather than a scalar formula, because the interesting
content is in the ordering and atomicity of the predicates.

```text
MECHANISM [T-01]: admission of one operation
PRECONDITIONS: (i) input parsed and schema-valid; (ii) principal resolved to a subject and a tenant;
  (iii) authoritative business state readable; (iv) a durable key/digest store exists with
  compare-and-insert semantics; (v) the executor holds its own credential, not the caller's.
STATE TRANSITION: PROPOSED -> ADMITTED(k, d) | REJECTED(reason)
  where admission performs exactly one atomic operation:
  INSERT (k, d, policy_version, authority_time) IF ABSENT, ELSE compare stored d' with d.
INVARIANT: for every dispatched effect there exists exactly one durable record keyed by k whose
  digest equals the digest of the bytes actually sent. Justification: the insert happens before
  dispatch, and dispatch bytes are derived from the same canonical payload that produced d, so any
  effect observed downstream has a matching record; and because the insert is compare-and-insert,
  two different payloads cannot both occupy k.
WORKED TRACE: amount 4,250.00 EUR to vendor V-88; d = h(...) = 9f2c...; insert(k=K-77, d) -> NEW;
  dispatch; sink commits; receipt COMMITTED. A retry of K-77 arrives with d unchanged -> insert
  returns DUPLICATE-SAME -> executor returns the stored receipt without dispatch. A retry arrives
  with amount 5,000.00 EUR -> d' != d -> CONFLICT -> reject, no dispatch.
FAILURE BOUNDARY: the invariant holds only for fields inside d, only while the store is available,
  and only while the sink honours k. If the sink is opaque and ignores k, the invariant degrades to
  "at most one dispatch per key", which is not "at most one effect".
```

Four properties of this block deserve separate comment because each is a common design error.

*Ordering.* The durable insert must precede dispatch. If the record is written after the response, a
crash between dispatch and record leaves an effect with no key, and the retry after restart is
indistinguishable from a fresh intent. If the record is written before dispatch but not fsynced, the
same hole reappears at the storage layer. The invariant is only as strong as the durability of the
insert, which is why compare-and-insert belongs to a store with a stated durability contract.

*Atomicity of compare-and-insert.* A read-then-write sequence in application code is not atomic. Two
concurrent retries of K-77 can both read absent and both insert, or insert one after the other with
different digests depending on interleaving. The fix is a single operation whose semantics are "insert
unless present, and tell me which happened", or a conditional write with a version precondition.

*Digest derivation from sent bytes.* If the executor canonicalises once for the digest and then
serialises separately for transport, the two can diverge: a rounding step, a default applied by a
mapper, a locale-dependent decimal comma. The invariant should be restated operationally as "the bytes
hashed are the bytes sent", enforced by hashing the final serialised form.

*Authority freshness.* `policy_version` and `authority_time` are recorded so that a later reviewer can
distinguish a wrong decision from a stale one. Without them, incident analysis collapses into opinion
about which service was at fault. The atomic insert is the cheapest place to capture both.

The base case also fixes the meaning of the four rejection reasons, which the error taxonomy must be
able to express distinctly: `SCHEMA_INVALID`, `SEMANTIC_DENIED`, `AUTHORITY_DENIED`, and
`KEY_CONFLICT`. Collapsing these four into one generic failure is what forces callers to retry
operations that can never succeed, and retrying a permanent denial is how a broken client turns a
policy problem into a load problem.

### 4.2 General case

The general case removes the assumption that the caller knows whether the operation happened. It also
removes the assumption that exactly one attempt will be made. The result is a statement about the
probability of a duplicate effect given a deduplication mechanism with imperfect coverage, and it is the
formula a reviewer will use to decide whether a given duplicate rate is acceptable.

```text
FORMULA [T-02]: expected duplicate effects under a keyed dedup with partial coverage
D = N * q * (1 - c) * (1 - p_sink)
WHERE: N = operations per period [count]; q = fraction of operations retried at least once
  [dimensionless]; c = dedup coverage, the probability that a repeated key is recognised as a
  repeat [dimensionless, 0..1]; p_sink = probability the sink itself deduplicates on the key
  [dimensionless]. D = expected duplicate effects per period [count].
CONSTRAINT: retries are assumed independent of the failure that caused them, and coverage is assumed
  uniform across operations. Both assumptions fail in correlated outages (section 4.4).
NUMERICAL EXAMPLE: N = 10,000 writes/day; q = 0.40; c = 0.97; p_sink = 0.
  Retried operations = 10,000 * 0.40 = 4,000.
  Unrecognised repeats = 4,000 * (1 - 0.97) = 4,000 * 0.03 = 120.
  D = 120 duplicates/day.
INTERPRETATION: coverage is the only lever with proportional effect. Raising c from 0.97 to 0.997
  cuts duplicates by a factor of ten (120 -> 12); halving q cuts them in half. Since q is set by
  upstream timeout policy you do not fully control, budget for coverage.
```

The multiplication by (1 - p_sink) is worth dwelling on because it decides an architecture argument.
When the sink deduplicates on your key, duplicates cost you nothing and the ledger's job is latency
optimisation and audit. When the sink is opaque, the ledger is the only defence and coverage must be
designed to be near-complete, which in practice means three things: the key store is consulted on
*every* attempt including the first, the key store is in the same failure domain as dispatch, and the
key survives process restarts because it is minted before the first attempt and persisted.

Coverage is not a property of the hash. It is a property of the paths that reach the sink. A common
defect is a "shadow path": a human operator, a scheduled reconciliation job, or a legacy batch process
that can create the same effect without presenting the key. Coverage then measures only the automated
path, and q is not the right parameter. The honest model in that case is to add the shadow path's
volume as a separate term with coverage zero, which is why the ledger needs a rule that no component
may write to a controlled resource outside the executor.

The same formula, read backwards, prices a design decision. Suppose duplicates cost C_d each in manual
rework, support contact, and finance correction. Then the annual cost of duplicates is D * C_d * 250
for a working year, and the value of improving coverage from c to c' is N * q * (c' - c) * C_d * 250.
With the numbers above and C_d = 45 EUR (an ILLUSTRATIVE blended rework cost):

```text
COST CHECK [T-02b]:
  D(0.97) = 10,000 * 0.40 * 0.03 = 120/day
  D(0.997) = 10,000 * 0.40 * 0.003 = 12/day
  annual saving = (120 - 12) * 45 * 250 = 108 * 45 * 250
  108 * 45 = 4,860 EUR/day; 4,860 * 250 = 1,215,000 EUR/year
INTERPRETATION: this is an upper bound that assumes all duplicates are detected and reworked at the
  blended cost. Undetected duplicates usually cost more per event but occur at unknown frequency,
  which is an argument for detection, not for a smaller budget.
```

### 4.3 Variants

The base case and the general case are the two you will use most, but six variants cover the questions
that actually arrive in a design review: capacity under overload, tail latency at high utilisation,
retry amplification, staleness of a separated authorization check, approval binding coverage, and the
cost of durability.

**Variant A — backlog and drain under a sustained overload.** This is the sizing calculation for the
recovery lane. It answers "how long until we are caught up" and, more importantly, "are we ever caught
up".

```text
FORMULA [T-03]: backlog accumulated during overload, then time to drain
B = max(0, λ - μ) * t
T_drain = B / (μ_r - λ_r)
WHERE: λ = arrival rate during the overload [ops/s]; μ = completion rate of the normal lane [ops/s];
  t = duration of the overload [s]; B = backlog at the end of the overload [ops]; μ_r = completion
  rate of the recovery lane [ops/s]; λ_r = arrivals that continue during drain [ops/s];
  T_drain = drain time [s].
CONSTRAINT: rates are treated as constant over each interval; T_drain is finite only when μ_r > λ_r.
  This is a conservation statement, not a latency model: it says nothing about how long an individual
  operation waits.
NUMERICAL EXAMPLE: peak λ = 24 ops/s for t = 90 s; μ = 20 ops/s.
  B = (24 - 20) * 90 = 4 * 90 = 360 operations.
  Recovery lane μ_r = 26 ops/s while live arrivals continue at λ_r = 21 ops/s.
  Surplus = 26 - 21 = 5 ops/s; T_drain = 360 / 5 = 72 s.
  Independent check: in 72 s the recovery lane completes 72 * 26 = 1,872 operations, of which
  72 * 21 = 1,512 are new arrivals, leaving 360 for the backlog. Consistent.
INTERPRETATION: the number that matters is the surplus, not the recovery lane's nominal speed. At
  μ_r = 21 and λ_r = 21 the surplus is zero and the drain time is infinite even though the lane is
  completing work at 21 ops/s.
```

The boundary case is the whole point. A team that adds a "fast retry lane" sized at the same rate as
the live traffic has added capacity without adding recovery. The arithmetic says the backlog is
conserved: it drains only at the difference. Two corollaries follow. First, during an incident the
cheapest way to create surplus is usually to reduce λ_r by shedding low-value traffic, not to increase
μ_r. Second, if the overload is not bounded in time, B grows without limit and no drain plan exists;
the only correct action is admission control that caps λ before the queue grows.

**Variant B — tail latency at high utilisation.** Backlog math answers "how much"; queueing answers
"how long". For a single-server queue with Poisson arrivals and exponential service, the probability
that a request waits longer than t seconds at utilisation ρ is a closed form, and it is the reason
utilisation targets for interactive tools sit well below the naive "keep the CPU busy" instinct.

```text
FORMULA [T-04]: M/M/1 tail probability of waiting more than t
P(W > t) = ρ * exp(-μ * (1 - ρ) * t), ρ = λ / μ
WHERE: λ [ops/s] arrivals; μ [ops/s] service rate; ρ [dimensionless, 0 ≤ ρ < 1]; t [s] the wait
  threshold; W [s] the random waiting time in queue.
CONSTRAINT: Poisson arrivals, exponential service, one server, FIFO, infinite queue. Real executors
  with downstream sync calls violate exponential service, so treat the result as a lower bound on
  the true tail for heavy-tailed service.
NUMERICAL EXAMPLE: μ = 20 ops/s.
  ρ = 0.80: P(W > 1 s) = 0.80 * exp(-20 * 0.20 * 1) = 0.80 * exp(-4) = 0.80 * 0.0183 = 0.0147.
  ρ = 0.90: P(W > 1 s) = 0.90 * exp(-20 * 0.10 * 1) = 0.90 * exp(-2) = 0.90 * 0.1353 = 0.1218.
  ρ = 0.95: P(W > 1 s) = 0.95 * exp(-1) = 0.95 * 0.3679 = 0.3495.
INTERPRETATION: moving from 80% to 90% utilisation multiplies the one-second tail by 8.3 while
  adding 12.5% throughput. If the tool's contract promises sub-second admission for interactive use,
  the capacity plan must target the tail, not the mean.
```

**Variant C — retry amplification.** A retry policy converts a transient failure into additional load.
The expected number of attempts per operation is bounded and computable, and its boundedness is what a
retry budget exploits.

```text
FORMULA [T-05]: expected attempts with a per-attempt success probability and a cap
E[attempts] = (1 - s^R) / (1 - s) for s < 1, and E[attempts] = R for s = 1
WHERE: s = probability that one attempt succeeds [dimensionless]; R = maximum attempts per
  operation [integer ≥ 1]; E[attempts] [count].
CONSTRAINT: attempts are independent and identically distributed; the cap R is enforced by the
  caller, not by hope. Correlated failures (a downstream outage) make s ≈ 0 for all attempts, and
  then E[attempts] = R, which is the worst case rather than the model.
NUMERICAL EXAMPLE: s = 0.70, R = 4: E = (1 - 0.7^4) / 0.30 = (1 - 0.2401) / 0.30 = 0.7599 / 0.30
  = 2.533. With no cap, 1 / 0.30 = 3.333. So a cap of four saves 0.80 attempts per operation.
  With s = 0.50, R = 3: E = (1 - 0.125) / 0.50 = 0.875 / 0.50 = 1.75.
INTERPRETATION: the load multiplier is E[attempts] / 1 and must be budgeted as capacity. At 24
  ops/s of intents and E = 1.05, the executor sees 25.2 ops/s; at E = 3.33 it sees 80 ops/s.
```

The design consequence is that retry policy belongs with capacity planning, not only with resilience.
A retry budget expressed as a fraction of successful traffic, for example "retries may not exceed 10% of
the previous minute's successes", bounds amplification by construction and degrades gracefully: when the
sink fails completely the budget is consumed in seconds and further retries are refused rather than
queued. Without a budget, the natural client behaviour under a partially degraded sink is to multiply
load exactly when the sink is least able to absorb it.

**Variant D — staleness of a separated authorization decision.** When the authority check happens in a
different component from the effect, a window opens in which the world can change. If state changes
arrive as a Poisson process with rate ν, the probability that at least one relevant change occurred in a
window of length g is elementary.

```text
FORMULA [T-06]: probability that authority state changed inside the TOCTOU gap
P(stale) = 1 - exp(-ν * g)
WHERE: ν = rate of authority-relevant state changes [changes/s]; g = gap between the authority check
  and the effect [s]; P(stale) [dimensionless].
CONSTRAINT: changes arrive independently at constant rate; "authority-relevant" means changes that
  would have altered the decision, which is a subset of all changes and must be counted separately.
NUMERICAL EXAMPLE: an approval token has a 180 s validity window; authority-relevant changes occur at
  ν = 0.40 per minute = 0.00667 per second. ν * g = 0.00667 * 180 = 1.20.
  P(stale) = 1 - exp(-1.20) = 1 - 0.3012 = 0.6988, so about 70%.
INTERPRETATION: a three-minute approval window is not a small gap when the underlying state changes
  every 2.5 minutes. Either shorten the token to a few seconds and re-check at the executor, or move
  the check to the executor at effect time and treat the token as an intent record.
```

This formula is the quantitative argument against "check once, execute later". It also explains why
downscoped, short-lived credentials are more than hygiene: they bound g by construction. A token valid
for 30 seconds with ν = 0.00667/s gives ν*g = 0.20 and P(stale) = 0.181, which is a different risk
class from 70%. Neither number is safe by itself; the point is that g is a design parameter with a
computable consequence, and it should be chosen rather than inherited from a default.

**Variant E — approval binding coverage.** A digest that omits effect-relevant fields gives an approval
a false scope. Coverage is a ratio of field counts weighted by whether each field changes the effect.

```text
FORMULA [T-07]: unbound-field fraction of an approval digest
U = |F_effect \ F_digest| / |F_effect|
WHERE: F_effect = set of fields that can change the effect or its legality [set]; F_digest = set of
  fields included in the approved digest [set]; U [dimensionless, 0..1] = fraction of effect-relevant
  fields a reviewer did not actually approve.
CONSTRAINT: counts fields, not risk; a single omitted `amount` field outweighs ten omitted display
  fields. Use the count as a screening test and then hand-audit the largest omissions.
NUMERICAL EXAMPLE: F_effect = {principal, tenant, verb, target_id, quantity, amount, currency,
  effective_date, cost_centre} = 9 fields; F_digest omits effective_date and cost_centre, so
  |...| = 7 and U = (9 - 7) / 9 = 0.2222.
INTERPRETATION: 22% of the decision surface was outside the approval. Moving `effective_date` into
  the digest changes the semantics of what a reviewer is approving, which is why digest coverage is
  a product decision as well as an engineering one.
```

**Variant F — durability and audit cost.** Deduplication windows are short and audit retention is long;
they are different stores with different cost curves. Sizing both prevents the classic mistake of
retaining the dedup ledger for years "because audit needs it".

```text
FORMULA [T-08]: dedup ledger size and cost over the retention window
S = λ_avg * 86400 * d * b / 1e9
WHERE: λ_avg = steady arrival rate [ops/s]; d = retention in days [days]; b = bytes per record
  including key, digest, state, timestamps, and index overhead [bytes]; S = size [GB].
CONSTRAINT: 86,400 s/day; index overhead is included in b, which is why b is much larger than the
  key length; no compression, so this is an upper bound for a row store.
NUMERICAL EXAMPLE: λ_avg = 24 ops/s, d = 7, b = 768 bytes.
  Records = 24 * 86,400 * 7 = 2,073,600 * 7 = 14,515,200.
  Bytes = 14,515,200 * 768 = 11,147,673,600. S = 11.1 GB.
  At d = 30 the same ledger needs 11.1 * 30 / 7 = 47.7 GB.
INTERPRETATION: a week of dedup state is single-digit gigabytes at this scale, which is cheap; the
  cost that surprises teams is the write rate and the index churn, not the bytes. Keep the audit
  store separate, append-only, and free of the uniqueness constraint the dedup store needs.
```

**Variant G — collision probability of the key or digest space.** Keys minted by clients and digests
truncated by implementers both consume a finite space. The birthday bound says how many operations you
can run before a collision is likely.

```text
FORMULA [T-09]: birthday-bound collision probability
P(collision) ≈ n^2 / (2 * M)
WHERE: n = number of distinct values stored [count]; M = size of the space [count]; P [dimensionless].
CONSTRAINT: n << sqrt(M); the approximation degrades above roughly n = sqrt(M), which is exactly where
  it stops mattering because the probability is already near one.
NUMERICAL EXAMPLE: 64-bit keys, M = 2^64 = 1.845e19, n = 2e6 values.
  n^2 = 4e12. 2 * M = 3.689e19. P ≈ 4e12 / 3.689e19 = 1.084e-7, about 1 in 9.2 million.
  128-bit keys: M = 3.4e38, P ≈ 4e12 / 6.8e38 = 5.9e-27.
INTERPRETATION: 64-bit client keys are adequate at millions of operations and become uncomfortable
  only near 1e9 operations per key space; 128-bit or UUIDv4-class keys remove the question entirely
  at negligible cost. Truncating a digest to 8 bytes for storage convenience re-imports the risk.
```

**Variant H — concurrency and in-flight work.** Little's law connects the three quantities every
operator reasons about incorrectly: rate, latency, and how many operations are "in progress".

```text
FORMULA [T-10]: in-flight operations
L = λ * W
WHERE: λ = throughput [ops/s]; W = mean time in system per operation [s]; L = mean in-flight
  operations [count].
CONSTRAINT: applies to a stable system in steady state; during a ramp the instantaneous relation does
  not hold, and L will trail the product.
NUMERICAL EXAMPLE: λ = 24 ops/s, W = 0.35 s (admission 45 ms + dispatch 90 ms + sink 215 ms).
  L = 24 * 0.35 = 8.4 operations in flight. At a 99th-percentile W of 2.5 s the tail carries
  24 * 2.5 = 60 operations, which is the number that sizes worker pools and connection limits.
INTERPRETATION: if the worker pool is 16 and the mean is 8.4, there is no headroom for the tail; the
  pool will be saturated by the slowest decile, which then becomes the new mean.
```

### 4.4 Limits and counterexamples

Every formula above has a boundary, and the boundary is usually the interesting part of the review.

**T-01 fails when the sink does not honour the key.** If the downstream system accepts your key field
but treats it as metadata, the invariant degrades from "one effect" to "one dispatch". Reconciliation
must then compare sink state (a business key such as invoice number plus amount plus date) rather than
look up your key. Test this explicitly: send the same key twice and observe whether the sink holds one
record or two. A sink that returns success for both is telling you the truth about itself.

**T-01 also fails under key reuse across tenants.** If the key namespace is global and the key is
minted from a tenant-local counter, two tenants can collide. Namespace keys by tenant in the store, and
include the tenant in the digest, so a cross-tenant attempt is a CONFLICT rather than a silent duplicate
suppression. This is a security property, not only a correctness one: suppressing tenant A's write
because tenant B used the same key is a data-integrity incident.

**T-02 fails under correlated retries.** The independence assumption means the duplicate estimate is
too optimistic exactly when load is high. The correct move is to add a floor: assume any operation that
was retried during an outage had coverage equal to the fraction of the interval in which the key store
was available. In a ten-minute outage where the ledger was unavailable for 90 seconds, coverage for that
cohort is at best 0.85 regardless of the steady-state number.

**T-03 fails when arrivals are not constant.** A ten-minute burst at 24 ops/s is not the same as a
two-minute burst at 120 ops/s even though both move 14,400 operations, because the second exceeds the
recovery lane's surplus for a period during which the backlog keeps growing. Integrate piecewise and
check each interval's surplus sign; the sum of positive-surplus intervals bounds the drain.

**T-04 fails under heavy-tailed service.** Downstream calls that occasionally take 30 seconds do not
have exponential service, and the M/M/1 tail understates the true tail. Use it as a lower bound and add
a timeout that converts unbounded tail into a bounded UNKNOWN, which is a state you can reconcile. The
timeout is not a latency optimisation; it is a mechanism that makes the outcome space enumerable.

**T-05 fails when retries are synchronised.** If all clients retry at 1, 2, 4, 8 seconds without jitter,
attempts arrive in correlated waves and the effective instantaneous arrival rate exceeds the average by
the wave height. Full jitter (uniform in [0, backoff]) is the minimum; decorrelated jitter spreads the
waves further. Without jitter, the load multiplier from T-05 understates peak load by roughly the number
of clients retrying in phase.

**T-06 fails when ν is not constant.** Authority state does not change uniformly: it changes in bursts
around period close, month end, and batch runs. A three-minute gap is far more dangerous at 23:58 on a
period-end date than at 11:00 on a Tuesday, which argues for gap limits that tighten near period
boundaries rather than a single static value.

**T-07 fails as a risk measure.** Equal weighting of fields makes an omitted currency look like an
omitted display name. The counterexample to fix in your head: a digest covering eight fields including
`amount` but omitting `currency` scores better than a digest covering seven fields including both. Rank
omissions by consequence and treat any omission of amount, currency, target, or verb as a defect
regardless of U.

**T-09 fails when the key is not random.** A monotonic counter has zero collision probability within a
process and total collision probability across processes that restart from zero. A hash of a payload is
not random because payloads repeat. The birthday bound applies to the *distribution*, so the bound is
only meaningful for keys that are actually random or namespaced by a unique instance identifier.

**T-10 fails as a capacity argument at the tail.** The mean L says nothing about the queue length at
p99, and it is the p99 queue that determines whether the system behaves linearly under the next burst.
Pair it with T-04 rather than using it alone.
## 5. Worked Numerical Example

The example is a single procurement connector that turns an approved purchase request into a posted
supplier invoice. Every parameter is ILLUSTRATIVE and chosen so the arithmetic is checkable by hand;
the structure is what transfers to your system, not the values.

**Setup.** One tool, `create_supplier_invoice`, effect class E2 (keyed write against a sink that honours
the key). Population: 500,000 invoice-creation operations per business day, concentrated in an
eight-hour window of 28,800 seconds. Within that window there is a one-hour close-out peak. The sink is
an ERP that is idempotent on our key for seven days and then forgets it. Manual review exists for
payments above a threshold and costs 45 EUR in blended reviewer time when a duplicate must be unwound.

**Step 1 — mean arrival rate.** Conservation of counts: 500,000 operations spread over 28,800 seconds.

```text
λ_mean = 500,000 / 28,800 = 17.361 operations/s
Check: 17.361 * 28,800 = 499,997 ≈ 500,000 (rounding of λ). Consistent.
```

**Step 2 — peak and off-peak rates.** The peak hour carries 45,000 operations; the remaining 455,000
spread over the other 27,000 seconds.

```text
λ_peak = 45,000 / 1,800 = 25.000 operations/s
λ_off  = 455,000 / 27,000 = 16.852 operations/s
Check: 25.000 * 1,800 = 45,000; 16.852 * 27,000 = 455,004; total 500,004 ≈ 500,000. Consistent.
```

**Step 3 — backlog during the peak.** The executor completes 22 operations/s in steady state, so the
peak runs a deficit and a backlog accumulates. This is T-03 applied to one interval.

```text
B = (λ_peak - μ) * t_peak = (25.000 - 22.000) * 1,800 = 3.000 * 1,800 = 5,400 operations
Check: 25 * 1,800 = 45,000 arrived; 22 * 1,800 = 39,600 completed; difference 5,400. Consistent.
```

**Step 4 — drain during the off-peak.** A recovery lane runs at 28 operations/s while live arrivals
continue at 16.852 operations/s. Surplus is what drains the backlog.

```text
surplus = μ_r - λ_off = 28.000 - 16.852 = 11.148 operations/s
T_drain = 5,400 / 11.148 = 484.4 s ≈ 8.1 minutes
Check: in 484.4 s the lane completes 484.4 * 28 = 13,563 operations and live arrivals add
484.4 * 16.852 = 8,163, so the net reduction is 13,563 - 8,163 = 5,400. Consistent.
```

**Step 5 — the boundary of that drain plan.** The same peak lasting 45 minutes instead of 30 minutes
gives a different answer, and the off-peak surplus then has a fixed budget.

```text
B(45 min) = 3.000 * 2,700 = 8,100 operations
T_drain = 8,100 / 11.148 = 726.6 s ≈ 12.1 minutes
Remaining off-peak time = 27,000 - 2,700 = 24,300 s; drain consumes 726.6 s, leaving 23,573 s of
uninterrupted off-peak operation. Safe.
But if the peak rate were 28.000 operations/s (equal to μ_r), surplus during the peak is zero and
B = (28.000 - 22.000) * 1,800 = 10,800 with no lane fast enough to catch up during the same hour.
```

The second number is the one to remember: a recovery lane slower than the peak arrival rate cannot
recover *during* the peak no matter how long it runs, because the backlog is not merely conserved, it is
growing. Capacity decisions made against the daily mean hide this entirely.

**Step 6 — expected duplicates.** Retried fraction q = 0.004 (4 in 1,000 operations experience at least
one timeout), ledger coverage c = 0.985, sink self-dedup p_sink = 0 for the first 60 seconds because the
sink's dedup is asynchronous. Applying T-02 for the full day:

```text
D = N * q * (1 - c) * (1 - p_sink) = 500,000 * 0.004 * 0.015 * 1
  = 2,000 * 0.015 = 30 duplicate effects/day
Check: 500,000 * 0.004 = 2,000 retried; 2,000 * 0.015 = 30. Consistent both ways.
```

**Step 7 — price the fix.** Improving coverage from 0.985 to 0.9999 reduces the uncovered fraction from
0.015 to 0.0001, a factor of 150.

```text
D(0.9999) = 500,000 * 0.004 * 0.0001 = 0.2 duplicates/day
Annual cost at 45 EUR per rework and 250 working days:
  D(0.985)  -> 30.0 * 45 * 250 = 337,500 EUR/year
  D(0.9999) ->  0.2 * 45 * 250 =   2,250 EUR/year
  difference = 337,500 - 2,250 = 335,250 EUR/year
```

That saving is an upper bound, because it assumes every duplicate is detected and every detection costs
exactly 45 EUR. In practice an undetected duplicate invoice costs far more per event (a payment run, an
apology, a credit note) but occurs at an unknown rate. The bound is still useful: it shows the coverage
work is worth more than the engineering it costs at almost any plausible rate.

**Step 8 — authority staleness.** Approvals carry a 45-second execution token; authority-relevant
changes (budget revisions, delegation revocations, period locks) arrive at ν = 0.0025 per second, about
one every 6.7 minutes. Applying T-06:

```text
ν * g = 0.0025 * 45 = 0.1125
P(stale) = 1 - exp(-0.1125) = 1 - 0.8936 = 0.1064, about 10.6% of approved operations
```

Ten percent is not a defect rate you can wave through: it means one in ten dispatches uses an authority
decision that the world has already invalidated. The two corrections both cost less than the exposure:
re-check at the executor (g becomes the duration of the executor's own check, effectively milliseconds)
or shorten the token to 5 seconds, which gives ν*g = 0.0125 and P = 0.0124.

**Step 9 — dedup ledger sizing.** Steady 25 operations/s is the sizing rate for the seven-day window
(the peak hour dominates storage), at 768 bytes per record including index overhead.

```text
Records = 25 * 86,400 * 7 = 2,160,000 * 7 = 15,120,000
Bytes   = 15,120,000 * 768 = 11,612,160,000
S = 11.61 GB for seven days
Check: 25 * 86,400 = 2,160,000/day; * 7 = 15,120,000; * 768 = 1.1612e10 B. Consistent.
```

**Step 10 — in-flight and queueing.** Mean time in system W = 0.42 s (admission 45 ms, dispatch 90 ms,
sink 285 ms). Utilisation at the daily mean against the normal lane is 17.361/22.000 = 0.789.

```text
L = λ_mean * W = 17.361 * 0.42 = 7.29 in-flight operations
P(W_queue > 1 s) at ρ = 0.789, μ = 22: 0.789 * exp(-22 * 0.211 * 1)
  = 0.789 * exp(-4.642) = 0.789 * 0.00964 = 0.0076, about 0.8%
At ρ = 0.90: 0.90 * exp(-2.2) = 0.90 * 0.1108 = 0.0997, about 10.0%
```

The pair is the argument for a utilisation target. At the daily mean the tail is under 1%; push the same
system to 90% by adding traffic and the one-second tail multiplies by thirteen.

**Step 11 — retry amplification.** Per-attempt success s = 0.75, cap R = 4, with jitter.

```text
E[attempts] = (1 - 0.75^4) / (1 - 0.75) = (1 - 0.3164) / 0.25 = 0.6836 / 0.25 = 2.734
Attempt load = 17.361 * 2.734 = 47.5 operations/s at the sink
Check: 0.75^4 = 0.316406; 1 - 0.316406 = 0.683594; / 0.25 = 2.734. Consistent.
```

The sink must be provisioned for 47.5 ops/s, not 17.4, whenever the first-attempt success rate is 0.75.
This is the number that turns "we have headroom" into an outage: the headroom was computed against
first attempts only.

**Step 12 — key space.** 2.5 million distinct keys per year, 64-bit space.

```text
P ≈ n^2 / (2M) = (2.5e6)^2 / (2 * 1.8447e19) = 6.25e12 / 3.689e19 = 1.694e-7, about 1 in 5.9 million
```

Acceptable, and 128-bit keys would remove the question. The point of computing it is that the number is
tiny *only because the key is random*; a tenant-local counter restarts at zero and collides immediately.

**Sensitivity.** Four parameters dominate the answers. (i) Doubling q from 0.004 to 0.008 doubles
duplicates to 60/day and doubles the rework cost. (ii) Raising c from 0.985 to 0.9999 reduces duplicates
by 150x, which is why coverage beats every other lever. (iii) Raising the peak from 25 to 28 ops/s makes
the backlog 10,800 and removes any same-hour recovery. (iv) Raising W from 0.42 s to 0.60 s at the same
arrival rate pushes L from 7.29 to 10.4 and ρ from 0.789 to a level where the pool saturates. Only
(ii) is a design decision under your control in a single sprint; the other three are properties of the
traffic and the sink you must measure.

## 6. Visual / Geometric Intuition

**Picture 1 — the conveyor with four gates.** A conveyor belt carries a labelled box. The label is the
model's proposal: a description of what it thinks should happen. Gate 1 checks the box's shape against a
template; a box that is too wide is rejected without judgement. Gate 2 checks whether this box is
allowed to exist at all today: is the period open, is the budget unspent, does the business key already
exist in a posted document. Gate 3 checks the bearer's badge against the destination zone: this principal
may move boxes into warehouse 7 but not warehouse 9. Gate 4 is the loading dock: a clerk stamps a
manifest with the box's identity and only then lets it leave, and the manifest is what the receiving
warehouse confirms.

The picture makes one thing obvious that prose obscures: the gates are physically ordered and each has
its own failure mode. A box rejected at Gate 3 does not need Gate 4 to be consulted; a box admitted at
Gate 4 has a manifest regardless of whether the radio to the receiving warehouse works.

**Picture 2 — the evidence ladder as a ladder, not a staircase.** Draw four rungs bottom to top:
proposed, admitted, dispatched, committed. The model lives on rung 1. Your admission service climbs to
rung 2 and stops. The network climbs to rung 3 and stops. Only the sink reaches rung 4. Every recovery
question is really the question "which rung am I standing on, and which rung did I assume I was on?" A
timeout looks like rung 1 from the client's memory and rung 4 from the sink's memory; that discrepancy
*is* the UNKNOWN state, and it exists because rungs are climbed by different parties.

**Picture 3 — the state graph is a partial order, not a cycle.** Model the lifecycle as directed edges
and forbid the ones that do not exist.

```text
PROPOSED ──schema ok──> VALIDATED ──policy ok──> ADMITTED ──dispatch──> SUBMITTED
    │                       │                        │                    │
    └─reject──> REJECTED    └─reject──> REJECTED     └─conflict──> CONFLICT
                                                                          │
                                                              ┌───────────┼───────────┐
                                                              v           v           v
                                                         COMMITTED    REJECTED     UNKNOWN
                                                                          ^           │
                                                                          │           │
                                                                    reconcile ──────┘
```

Three edges are forbidden and each one is a real defect if drawn. `UNKNOWN -> NOT_SENT` is forbidden
without new authoritative evidence, because the only thing that changed is your patience. `COMMITTED ->
PROPOSED` is forbidden, because a committed effect cannot become an intention again; a correction is a
new operation with its own key. `REJECTED -> SUBMITTED` is forbidden for permanent reasons, because
retrying a permanent denial is the mechanism that converts a policy error into a load spike.

**Picture 4 — blast radius as concentric circles.** Centre: one resource (one invoice, one record).
Ring 1: the document family (all invoices for one vendor). Ring 2: the tenant. Ring 3: the region. Ring
4: everything the service credential can reach. Schema breadth and credential scope decide which ring
one call can touch. A wildcard target field or a `*` in a resource path moves a single call outward by a
ring, which is why wildcard-shaped parameters deserve review even when the schema is perfectly typed.

**Picture 5 — the two clocks as a slanted plane.** Draw event time on the horizontal axis and processing
time on the vertical. A healthy system is a narrow diagonal band. A reconciliation gap is a widening of
the band; a period close is a vertical cliff in processing time where many events process at one instant.
An operation whose receipt records only processing time sits at the cliff and cannot be compared against
a period boundary. This is why the audit record needs both stamps: the diagram shows exactly where the
missing one hurts.

**Picture 6 — the digest as a shadow of the payload.** The canonical payload is a solid object; the
digest is its shadow. Two objects with the same shadow are the same for approval purposes. The failure of
digest coverage is a light source that casts no shadow for certain parts of the object: those parts are
invisible to the approval, not absent from the effect. When you draw it this way, "we include the amount"
immediately invites the question "what else casts no shadow", and the answer is usually currency, date,
or cost centre.

## 7. Comparison / Selection Table

**Effect classes and their mandatory controls.** Classify first, then choose controls. The class is a
property of the sink, not of your intentions.

| Class | Example | Reversible? | Key required | Receipt source | Human gate |
|---|---|---|---|---|---|
| E0 read | price lookup, record fetch | n/a (no effect) | no | response body | none |
| E1 natural upsert | write a config value, set a field | yes, last write wins | no, but record digest | read-back | none |
| E2 keyed write | create invoice, create ticket | usually compensate | yes, durable | sink receipt on key | above threshold |
| E3 irreversible | payment release, delete record, send email | no | yes, plus intent record | sink transaction id | always |
| E4 fanout | broadcast to N systems | partially | yes, plus per-item keys | per-item receipts | above N items |

The table's most consequential row is E2, because it is where most business tools actually live and where
teams wrongly apply E1 thinking ("we can just retry, it will overwrite"). Creating a second invoice does
not overwrite the first; it duplicates it. If the sink cannot hold a uniqueness constraint on your key,
the tool is E3 for practical purposes and needs an intent record plus reconciliation.

**Recovery strategies compared.** Four options are usually on the table after a lost response. They
differ in what evidence they require and what they cost.

| Strategy | Requires | Duplicate risk | Recovery latency | Use when | Refuse when |
|---|---|---|---|---|---|
| Blind retry, new key | nothing | unbounded, grows with outage length | seconds | E0 reads only | any write |
| Blind retry, same key | key persisted pre-dispatch | low, bounded by coverage | seconds | E2 with reliable ledger | key was not persisted |
| Reconcile then retry | queryable sink or outbox in sink transaction | near zero | seconds to hours | E3, or opaque sink | no authoritative query exists |
| Manual review queue | reviewer capacity, redacted context | zero by construction | hours to days | high value, low volume | volume exceeds reviewer capacity |

The last row has a hard arithmetic limit that teams routinely ignore. If an outage strands 3,000
operations and a reviewer can process 40 per hour with two reviewers, the queue takes 3,000 / 80 = 37.5
hours to clear, during which new operations keep arriving. A manual fallback is a recovery plan only when
its throughput exceeds the arrival rate of the thing it is recovering.

**Hold, shed, or replay.** Under overload, three levers exist and they are not interchangeable.

| Lever | Mechanism | Effect on correctness | Effect on user | Prefer when |
|---|---|---|---|---|
| Hold (queue) | bounded queue with backpressure | preserves every intent; risks stale intents | latency grows, eventually timeouts | overload is short and capacity returns |
| Shed | reject with retry-after and a typed reason | loses no committed work; drops intents | explicit, actionable failure | overload is sustained or the tool is low value |
| Replay | reconcile and re-drive from a durable record | duplicates only where coverage is incomplete | delayed completion | after recovery, from recorded intents only |

Hold is the instinct and the worst choice for long overloads, because a queue of stale intents is a queue
of operations whose authority has expired by the time they run. Shed is honest. Replay is only legitimate
from a durable intent record with an explicit outcome, never by scanning logs.

**Authorization placement.** Where the check happens determines what the check is worth.

| Placement | Latency added | Freshness | Weakness | Mitigation |
|---|---|---|---|---|
| Prompt time (model context) | none | none | attacker-influenced text, no principal | never count as authorization |
| Admission service | ~5–20 ms | good | TOCTOU gap to effect | short scope, recheck at executor |
| Executor, at effect time | ~1–5 ms (local policy) | best | needs fresh state or a policy cache | cache with version stamp |
| Downstream sink | 0 for you | authoritative | error surfaces as sink failure | map sink denials to typed reasons |

The second row is where most systems stop and where T-06 then bites. The fourth row is often forgotten:
the sink has its own authorization and will deny operations your admission service allowed. That denial
must be mapped back to a typed error, or callers will treat a permanent denial as a transient failure and
retry it.

**Schema strictness compared.** The schema is the first gate and its strictness is a trade with real
consequences on both sides.

| Choice | Rejects | Accepts | Failure mode | When it wins |
|---|---|---|---|---|
| Strict, no defaults, all fields required | malformed and under-specified calls | exactly one interpretation | brittle to optional business fields | E3 tools, regulated flows |
| Strict with typed optionals | malformed | several legal shapes | ambiguity about which shape ran | E2 with optional metadata |
| Permissive, extra fields allowed | almost nothing | future fields silently | forward-compatible but effect-opaque | E0 reads only |
| Open free-form object | nothing | anything | schema becomes documentation | never for a write |

The permissive row deserves a warning: allowing unknown fields means a new field added by a client
deployment can enter the canonical payload and change the digest without any reviewer seeing a schema
change. If the tool must tolerate extra fields, exclude them from the digest explicitly and record them
for inspection, so that "unknown field present" is a visible audit fact rather than an invisible one.

**Receipt queries compared.** How you learn the outcome determines how fast and how safely you recover.

| Source | Authority | Latency | Failure when unavailable | Cost |
|---|---|---|---|---|
| Your own outbox row | weak: written before or after the effect | ms | outage loses the trail | low |
| Outbox written in sink transaction | strong | ms to s | only if the sink's DB is down | medium |
| Sink read API on your key | strong | 10 ms to s | outages block all recovery | low |
| Sink business-key search | medium: may match unrelated records | s to minutes | fuzzy matches mislead | medium |
| Reviewer inspecting the sink UI | strong but slow | minutes | human bottleneck | high |

The first row is the trap. An outbox you write yourself before dispatch tells you what you *intended*
and nothing about the effect. It is a legitimate record of intent and an illegitimate receipt. Name it
accordingly in code, because a field called `status: COMMITTED` in a pre-dispatch table will eventually
be read as proof by someone under pressure at 03:00.
## 8. Failure Modes & Edge Cases

Each entry below is a distinct mechanism, not a variation on "monitor the model". They are ordered by
where in the corridor they bite. For every entry the useful question is the same: which observation
separates this cause from its nearest rival, and what would you look at first? All scenarios are
constructed teaching cases (HYPOTHETICAL) built from the stated contract; none asserts an observed
incident at any organisation.

### F-01 — Wildcard resolution widens the target set after approval [VIOLATED_ASSUMPTION]

**Mechanism.** The request legitimately contains a selector (`vendor_group: "SUP-*"`, `cost_centres:
["CC-1", "CC-2", ...]`, or a filter expression). The approval covers the selector as written. The
resolver, the sink, or a later state change expands that selector into a larger set than the approver
mentally enumerated, so one approved operation produces N effects, some outside the reviewed scope.

**Competing causes.** (a) The resolver expanded a legitimate wildcard. (b) The selector was legal but
matched unintended records because a naming convention changed. (c) The approval UI displayed the
selector rather than the resolved list, so nothing expanded — the approver simply never saw it. (d)
Pagination: the resolved list was longer than one page and only the first page was reviewed.

**Discriminating check.** Log both the declared selector and the canonical resolved target list at
admission, and compare cardinality with what the approval record contains. If the approval record holds
a selector string rather than an item list, cause (c) is confirmed regardless of what happened
downstream. If the approval holds a list, diff it against the dispatched list and the first divergence
identifies (a) versus (d).

**Correction.** Replace selector-shaped targets with explicit bounded lists at approval time, or make
expansion a two-phase operation: resolve, re-present the concrete list with a count, then approve. Cap
the permitted expansion cardinality per operation (for example, 50 items) and require a separate
approval above the cap.

**Prevention.** Treat any parameter whose cardinality is not 1 as a schema-level hazard. The contract test
is: submit a selector that resolves to the cap plus one item and assert the tool refuses before dispatch.

**Why it recurs.** Wildcards are convenient and every demo uses them. The convenience is real at E0 and
false at E2 and above, because the effect multiplies while the approval does not.

### F-02 — Response lost after a successful commit [NUMERICAL_EDGE]

**Mechanism.** The sink committed, the response was lost to a timeout, a proxy reset, a load balancer
idle-close, or a process death between `recv` and the durable write. The caller holds no outcome. The
natural next action — retry — is correct only if the key is reused and the ledger is consulted.

**Competing causes.** (a) The sink never received the request. (b) The sink received and is still
processing (async queue). (c) The sink committed and the response was lost. (d) The sink rejected and
the rejection was lost. All four look identical from the caller: a timeout.

**Discriminating check.** Query the sink on the key. `COMMITTED` selects (c); `NOT_FOUND` after the
sink's own processing window selects (a); `PENDING` or a queued job id selects (b); a stored rejection
with a reason selects (d). If the sink cannot be queried, the honest observation is "UNKNOWN", and the
only separating evidence is a later business-level check (does the invoice appear in the vendor's
statement next cycle).

**Correction.** Freeze the retry, promote the operation to UNKNOWN durably, and drive reconciliation.
Do not mint a new key: a new key converts an ambiguous outcome into a probable duplicate.

**Prevention.** Timeouts shorter than the sink's observed p99.9 commit time are *designed* to produce
UNKNOWN. Measure the sink's commit latency distribution and set the caller timeout above the point where
the ambiguity rate is acceptable, then make the ambiguity visible in metrics as a first-class outcome.

### F-03 — Same key, different payload [NUMERICAL_EDGE]

**Mechanism.** A retry or a replay presents key K with a payload whose canonical digest differs from the
stored one. This is not a duplicate to be suppressed and not a new operation to be created; it is
evidence that two different intentions claimed one identity.

**Competing causes.** (a) A UI edit changed the amount between attempts. (b) Canonicalisation is
non-deterministic: float formatting, key ordering, locale decimal separators, timestamp precision,
Unicode normalisation of a vendor name. (c) A default was applied by a mapper on one path and not the
other. (d) A genuine bug minted one key for two operations.

**Discriminating check.** Recompute the digest from the stored canonical payload bytes on read. If it
differs from the stored digest, the canonicalisation is broken, not the client (b). If it matches but the
incoming payload's digest differs, print both canonical payloads and diff them: a one-character diff in a
decimal field points at (b); a whole-field diff points at (a) or (c); two unrelated documents point at
(d).

**Correction.** Reject with `KEY_CONFLICT` and require a newly approved operation. Never "update" the
stored digest to match the new payload: that converts the ledger into a last-writer-wins cache and
removes the only duplicate detector you have.

**Prevention.** Canonicalise with a single serialiser used for both hashing and transport, and include a
canonicalisation version in the stored record so a format change is a migration rather than a silent
digest change. Freeze decimals as scaled integers, sort keys, and normalise strings before hashing.

### F-04 — Syntactically valid, semantically denied [CONCEPT_CONFUSION]

**Mechanism.** The call passes the schema and fails a business rule: period closed, budget exhausted,
duplicate business document, approval reference unknown, quantity below the order minimum. Callers that
see only a generic failure retry, which cannot help, and the retry load lands on the policy layer.

**Competing causes.** (a) A real policy denial. (b) A stale policy cache (the denial reflects state from
minutes ago). (c) The wrong tenant or cost centre was resolved because of an identifier confusion. (d)
A period boundary was crossed between admission and effect.

**Discriminating check.** Return and log a typed reason plus the `policy_version` and the authority time
used. Evaluate the same predicate again against current state: if the second evaluation allows it, (b) or
(d); if it denies with the same reason, (a); if it denies with a different reason, (c) is likely because
the resolved subject differs from the intended one.

**Correction.** Surface the typed reason to the caller and mark the operation non-retryable. Retry only
the class of denials that are explicitly time-based (period lock opening, budget top-up), and treat those
as scheduled re-submissions with fresh approval, not as backoff loops.

**Prevention.** Make `retryable` an explicit boolean field on the error type rather than something each
caller infers from prose or from a status code. A taxonomy where `SCHEMA_INVALID`, `AUTHORITY_DENIED`,
and `KEY_CONFLICT` are all `retryable: false` removes an entire family of client bugs.

### F-05 — Credential or data leakage through tool output [VIOLATED_ASSUMPTION]

**Mechanism.** A read tool returns a record containing a bearer token, a session cookie, a bank account
number, or a full personal record. That output enters model context, is summarised into a transcript, is
written to a debug log, and is later retrievable by an unrelated principal.

**Competing causes.** (a) The output projection includes secret-bearing fields. (b) An error path
serialises the entire request including headers and credentials. (c) A debug flag is enabled in
production. (d) A retrieved document (not tool output) contains a secret and is conflated with it.

**Discriminating check.** Scan the transcript and logs with secret-shaped patterns and compare against
the tool's declared output schema. Fields present in the payload but absent from the schema identify the
projection as the source. If the secret appears only in error records, (b) is the cause. If it appears
in a document body, (d).

**Correction.** Stop propagation: revoke anything exposed, purge the affected transcripts and logs by
policy, and narrow the projection to an allowlist of fields per tool rather than a denylist of known
secret names.

**Prevention.** Deny-by-default output projection is cheap to implement and hard to regress: a field not
listed in the output schema is not serialised at all. Pair it with a scan in CI over serialised fixtures
so that a mistakenly added field fails a test rather than a review.

### F-06 — Retry storm against a partially degraded sink [SCALE_FAILURE]

**Mechanism.** A sink slows to 60% of capacity. Clients see more timeouts, retry more, and the executor's
arrival rate rises while its service rate falls. The queue grows at the sum of both effects, and the
system enters a state where throughput is below the pre-incident level and cannot recover while retries
continue.

**Competing causes.** (a) A client retry policy without a budget. (b) Retries without jitter arriving in
phase. (c) A timeout set below the degraded service time, guaranteeing a timeout even for work that will
succeed. (d) Synchronous downstream calls inside the tool that multiply per-operation occupancy.

**Discriminating check.** Plot attempts per intent alongside successful receipts. Rising attempts with
flat receipts is (a) or (b). If attempts per intent is near 1 but queue depth still grows, the timeout
is too tight for current service times (c). If occupancy per operation rose without an attempt increase,
look for added synchronous calls in the tool implementation (d).

**Correction.** Engage a retry budget (for example, retries may not exceed 10% of prior-minute
successes), shed rather than queue beyond the budget, and raise the caller timeout for the duration of
the incident with a compensating reduction in attempts.

**Prevention.** Budget retries in the executor, not in each client SDK. A shared budget is enforceable;
five independent SDK policies are not. Add jitter by default and treat "no jitter" as a defect in review.

### F-07 — Version drift between client and server interpretations [SCALE_FAILURE]

**Mechanism.** A field changes meaning, units, or optionality. The client still sends the old shape; the
server accepts it under a permissive rule and applies a default. Both sides are internally consistent and
the effect is wrong: amounts in minor units read as major units, or an omitted `currency` defaults to the
server's home currency.

**Competing causes.** (a) A schema version mismatch during a rolling deploy. (b) A silent default applied
by the server. (c) A cached tool manifest describing a previous version. (d) Units changed without a
version bump.

**Discriminating check.** Record the negotiated schema version and the canonicalisation version on every
admission. A cluster of bad effects all reporting version v1 from a client that believes it is on v2
confirms (c). If versions agree but a field is absent from the request and present in the effect, (b).
If the magnitude is wrong by exactly 100, (d).

**Correction.** Reject unknown-version calls rather than applying defaults; replay the affected window
from the intent record after reconciliation, with a human check on the corrected amounts.

**Prevention.** Version the tool contract, not only the API. Make defaults explicit in the schema as
`required` with no fallback, and treat "server silently filled a field" as an audit event rather than a
convenience. Include the version in the canonical digest so a version change cannot reuse an in-flight
key.

### F-08 — Cross-tenant read or write through a shared cache or handle [CONCEPT_CONFUSION]

**Mechanism.** A cache key omits the tenant, a resource handle is guessable, or a "find by name" lookup
returns another tenant's record. The tool is functioning as designed; the design omitted tenancy from
the identity of the thing.

**Competing causes.** (a) Cache key missing tenant. (b) Server-side filter absent, with a client-side
filter as the only defence. (c) A shared service credential acting as a confused deputy. (d) A handle
that is globally sequential, so identifiers leak across tenants by construction.

**Discriminating check.** Issue the same logical query under two tenants and compare results and cache
hit keys. Identical objects returned for different tenants confirms (a) or (b). If the tool's own audit
shows the same principal subject for both tenants, (c). If the leak is only of existence, not content,
(d).

**Correction.** Deny and invalidate cached entries; enforce tenancy in the data access layer, not in the
tool handler, so every path inherits it.

**Prevention.** Make the tenant part of the resource identity in the schema (not an optional header) and
part of every cache key. Test with a matrix of two tenants by two tools asserting zero overlap; this is
one of the few tests that catches tenancy regressions before they reach production.

### F-09 — Approval detached from the payload that executes [VIOLATED_ASSUMPTION]

**Mechanism.** A reviewer approves an intention ("pay vendor V-88 for the Q3 order") in prose or via a
summary, and the executor runs a payload the reviewer never saw, because the model re-derived the
amount, the date, or the target between approval and dispatch.

**Competing causes.** (a) Approval stored as free text with no digest. (b) Approval bound to a digest of
an earlier payload version. (c) Re-derivation between approval and dispatch changed a field. (d) A
partial approval UI that displayed only summary fields.

**Discriminating check.** Compare the approved digest, the digest at dispatch, and the digest of the
canonical payload the sink received. Three-way mismatch isolates the stage: approval-to-dispatch
divergence is (c); approval carrying no digest at all is (a); dispatch-to-sink divergence means the
serialiser is not the canonical one.

**Correction.** Re-present the canonical payload for approval whenever the digest changes; treat a
changed digest as a new approval requirement, not an update.

**Prevention.** Bind approval to `(digest, approver, expiry, policy_version)` and verify it inside the
executor immediately before dispatch. A digest check that happens anywhere upstream of the last possible
mutation is theatre.

### F-10 — Deduplication window shorter than the retry and human horizon [NUMERICAL_EDGE]

**Mechanism.** The ledger retains keys for 24 hours. A reconciliation gap, a weekend, a support ticket,
or a delayed replay arrives on day three with the same key. The ledger says NOT_FOUND, the operation is
admitted as new, and the sink creates a second effect because its own dedup also expired.

**Competing causes.** (a) TTL shorter than the retry horizon. (b) Two different ledgers (per-region) each
retaining their own keys. (c) Clock skew making a record look expired. (d) A restored backup losing the
recent window.

**Discriminating check.** Compute the actual maximum age of any replayed key from the audit log; if it
exceeds the TTL, (a) is proven by arithmetic alone. If duplicates come from different regions, (b). If
expiry timestamps are inconsistent across replicas, (c).

**Correction.** Extend retention to the measured maximum replay age plus a margin, and for operations
whose sink retention is shorter than the human horizon, add a business-key check (vendor + amount + date
+ reference) as a second line.

**Prevention.** Derive the TTL from a stated horizon rather than a default: `W = max(retry horizon,
reconciliation lag p99, support resolution time) + margin`. Write that sentence into the tool's contract
so a future change to the retry horizon forces a TTL review.

### F-11 — Burst blindness: the average says fine, the queue says otherwise [SCALE_FAILURE]

**Mechanism.** Capacity is planned against a daily mean of 17 operations/s while the close-out hour
arrives at 25 operations/s and the recovery lane must also absorb the backlog. The plan is arithmetically
correct about the mean and structurally wrong about the shape.

**Competing causes.** (a) Under-provisioned peak capacity. (b) Correct capacity with no recovery surplus.
(c) A serialised bottleneck (one database row, one lock, one vendor API quota) capping effective μ below
the measured mean. (d) A retry amplification layered on top of the peak.

**Discriminating check.** Compare per-minute arrival and completion curves rather than daily totals. If
completion flat-tops below the provisioned rate, the bottleneck is (c) and adding workers will not help.
If completions track arrivals but the queue persists into the next hour, (b). If attempts per intent rises
during the peak, (d).

**Correction.** Create surplus during the drain (shed low-value arrivals first), and fix the serialised
bottleneck before adding workers.

**Prevention.** Plan with a load shape, not a mean: peak rate, peak duration, and drain surplus are three
numbers that belong in the launch review, and the drain formula should be recomputed whenever any of them
changes.

### F-12 — Compensation described as undo, executed as a second effect [CONCEPT_CONFUSION]

**Mechanism.** A runbook says "roll back the invoice". The tool issues a credit note. The credit note is
a new committed effect with its own approval, its own downstream consequences (the vendor reconciles it,
the tax report changes), and its own failure probability. Six sequential compensations with a 92% success
rate each leave a 0.92^6 = 0.606 chance that all succeeded, so there is a 39% chance of a partial state.

**Competing causes.** (a) The original effect was already consumed downstream, so compensation cannot
restore the prior state. (b) The compensating operation was not authorised, so it failed. (c) The
original and the compensation both succeeded but the downstream consumer applied them out of order. (d)
A compensation of a compensation was issued after a timeout, creating net drift.

**Discriminating check.** Compare the sink's transaction ledger, ordered by event time, against the
expected sequence. A missing compensation with a rejection reason is (b). Both present but the consumer's
derived balance is wrong is (c). Extra compensating entries are (d).

**Correction.** Treat compensation as a first-class operation with its own key, approval, and typed
outcomes; make partial compensation an explicit state with an owner, not an error to retry.

**Prevention.** Rename the concept in code and runbooks. A type called `Reversal` with a docstring saying
"this is a new effect and may fail" prevents the mental substitution that "Undo" invites. Model
compensation chains explicitly and alert on chains longer than two.

### F-13 — Injected tool description or retrieved text changing routing [VIOLATED_ASSUMPTION]

**Mechanism.** A tool's description, a retrieved document, or a resource annotation contains text that
changes which tool the model selects or which arguments it fills. Because descriptions are discovered at
runtime, an attacker who can influence a description or a shared resource influences routing.

**Competing causes.** (a) A malicious or compromised description in the manifest. (b) Injected text in a
retrieved document that the model treats as instruction. (c) Legitimate ambiguity in a description that
the model resolves differently across runs. (d) A stale manifest serving an older, permissive description.

**Discriminating check.** Pin the manifest digest and diff on change; an action change correlated with a
manifest digest change is (a) or (d). Replay the same request with retrieval disabled: if the action
stabilises, (b). Non-determinism with no text change points at (c) and should be measured as proposal
variance, not as an attack.

**Correction.** Allowlist the set of tools and arguments available in a given context, and ignore
description text as an authority signal. Rotate anything the description could have reached.

**Prevention.** Treat all discovered text as data with a provenance label, enforce the allowed action set
outside the model, and pin manifest versions with an explicit review on change. A description is
documentation for a human reviewer; it is not a control.

### F-14 — Typed failure collapsed into a generic error at a boundary [CONCEPT_CONFUSION]

**Mechanism.** The tool returns a precise reason, but an intermediate layer (a gateway, an adapter, an
SDK) maps everything to "tool error". Callers lose the `retryable` flag and apply a default policy, which
is almost always retry. A permanent denial becomes a retry loop, and a KEY_CONFLICT becomes a duplicate.

**Competing causes.** (a) An adapter that rethrows a generic exception. (b) An HTTP status mapping that
collapses 4xx into one code. (c) A client SDK that treats any non-200 as transient. (d) An orchestrator
that swallows the error and synthesises a new call.

**Discriminating check.** Capture the error object at each hop and compare type and fields. The first hop
where the reason becomes generic is the culprit. If the reason survives to the client but the client still
retries, the defect is in the client policy, not the boundary (c).

**Correction.** Make the error taxonomy structural: a closed enum with `retryable`, `remediation`, and
`owner` fields, carried unchanged through every hop. Enforce with a contract test that asserts the reason
code at the outermost boundary.

**Prevention.** Test the error path as carefully as the happy path. Most integration test suites exercise
success and one timeout, which is precisely why taxonomy collapse survives to production.

### F-15 — Audit record written before the effect, read as a receipt [NUMERICAL_EDGE]

**Mechanism.** The outbox row is written pre-dispatch for durability. Someone later reads
`outbox.status` and reports it as outcome evidence, or a dashboard counts outbox rows as successes. The
numbers are plausible and systematically optimistic by exactly the failure rate.

**Competing causes.** (a) Pre-dispatch and post-commit states stored in one column. (b) A dashboard
aggregating intents rather than receipts. (c) A reconciliation job that marks rows complete on timeout
"to keep the queue moving". (d) Duplicate counting because both the outbox row and the receipt are summed.

**Discriminating check.** Compare the count of `COMMITTED` rows against the sink's own count for the same
window. A constant ratio below 1 indicates (a) or (b); a ratio above 1 indicates (d). Rows with
`COMMITTED` but no sink transaction id indicate (c).

**Correction.** Separate intent state from outcome state in the schema (`dispatch_state` and
`outcome_state`), and never let a single enum span both. Recompute dashboards from outcome state only.

**Prevention.** Name fields for the rung they represent. A field called `status` invites exactly this
confusion; `dispatch_state` and `outcome_state` do not. This is a naming decision with measurable
consequences during incident review.

## 9. Complexity & Cost Analysis

Costs here are of four kinds: per-operation time added by controls, storage for dedup and audit,
cognitive cost of taxonomy and review, and the expected cost of the failures the controls prevent. The
last one is the only one that justifies the first three, and it is the one most often left unstated.

**Per-operation latency budget.** The controls are cheap individually and additive. For the E2 tool in
section 5, an ILLUSTRATIVE breakdown:

```text
COST MODEL [C-01]: added latency per operation
schema validation            2 ms
canonicalise + digest        1 ms
policy evaluation (cached)   4 ms
key/digest compare-and-insert 6 ms   (single-region durable write)
authority recheck at executor 3 ms
dispatch (network + sink)   285 ms
receipt persist             5 ms
TOTAL                       306 ms, of which 21 ms is control overhead (6.9%)
WHERE: all values are illustrative single-region measurements; cross-region durable writes multiply the
  insert cost by the round-trip, typically 30-120 ms.
INTERPRETATION: control overhead is small until the durable insert crosses a region boundary. That is the
  one line item worth engineering: co-locate the key store with the executor.
```

The design consequence is that the deduplication store should live in the same failure domain and region
as the executor, even if the audit store does not. Teams frequently share one database for both and then
pay cross-region writes on the hot path while paying for long retention they do not need on that path.

**Storage economics.** Two stores with different requirements, which is why they should not be one.

```text
COST MODEL [C-02]: dedup ledger vs audit log, one year at 17.361 ops/s mean
DEDUP: 7-day retention, 768 B/record, uniqueness index required
  records = 17.361 * 86,400 * 7 = 1,499,990 * 7 ≈ 10,500,000 (rounding to 17.361 exactly)
  bytes   = 10,500,000 * 768 = 8.06 GB
  write rate = 17.361/s = 1.5M writes/day against a unique index
AUDIT: 400-day retention, 1,200 B/record, append-only, no uniqueness
  records = 17.361 * 86,400 * 400 = 1,499,990 * 400 ≈ 600,000,000
  bytes   = 600,000,000 * 1,200 = 720 GB
INTERPRETATION: audit dominates bytes by ~90x while dedup dominates write amplification and index churn.
  Putting them in one store forces the audit volume through a unique index, which is the expensive part.
```

The operational reading is that the dedup ledger is a small, hot, indexed component and the audit log is
a large, cold, append-only one. Merging them is a common simplification that inverts the cost curve:
you pay uniqueness maintenance on 600 million rows instead of 10.5 million, and every audit insert
contends with the hot path.

**Review cost, and the throughput ceiling of a human gate.** A human gate has a service rate, and a
service rate below the arrival rate is not a control, it is a queue.

```text
COST MODEL [C-03]: human review lane
arrival needing review: 8% of 17.361 ops/s = 1.389 ops/s = 5,000 ops/hour
reviewer throughput: 4 minutes/op = 15 ops/hour/reviewer
reviewers required at parity: 5,000 / 15 = 333 reviewers
At 10 reviewers: capacity 150 ops/hour vs 5,000 arriving -> the queue grows 4,850/hour
INTERPRETATION: a "human reviews all exceptions" policy is only viable if the exception rate is far
  below 0.3% or the reviewer pool is implausible. Otherwise the gate must be selective by consequence.
```

This arithmetic is why approval thresholds exist and why they must be set by money or blast radius rather
than by "anything that feels risky". A gate at 8% of volume with four-minute reviews needs 333 reviewers;
a gate at 0.1% of volume needs about 4. The control that actually scales is the keyed executor plus
reconciliation, with humans reserved for the E3 tail.

**Expected cost of the failure the controls prevent.** Compare the cost of a control against the expected
loss it removes, and be explicit that the loss estimate is uncertain.

```text
COST MODEL [C-04]: duplicate-write exposure versus control cost
duplicates/day at c=0.985: 30 (section 5, step 6)
rework cost/duplicate: 45 EUR (illustrative blended)
annual rework: 30 * 45 * 250 = 337,500 EUR
undetected share: assume 25% of duplicates are found only at period close, at 900 EUR each
  undetected: 30 * 0.25 = 7.5/day -> 7.5 * 900 * 250 = 1,687,500 EUR/year
control cost: two engineer-weeks to raise coverage to 0.9999 + 11.6 GB storage + index write load
INTERPRETATION: the dominant term is the undetected share, and it is the term with the widest error bar.
  Even at a tenth of the assumed rate the control pays for itself in the first year.
```

The honest caveat is that the 25% undetected share and the 900 EUR figure are constructed, not measured.
The method transfers: separate detected from undetected duplicates, because detected ones have a known
unit cost and undetected ones have an unknown but larger one, and it is the undetected ones that justify
coverage investment.

**Cognitive cost of taxonomy.** A closed error taxonomy costs design time once and saves per-call
decision time forever. An open vocabulary of prose errors makes every client decide retryability by
string matching, which is exactly the kind of implicit coupling that fails silently when a message is
edited. Four reason codes with a `retryable` flag cost a few hours to agree on and remove an entire class
of duplicate-producing bugs.

**Where the complexity actually lives.** Three of the ten formulas in section 4 are single-line
arithmetic; the complexity is never in the math. It is in the placement of checks (executor versus
admission service), the durability of the record relative to dispatch, and the honesty of the outcome
states. A design that gets those three right and the arithmetic roughly right will be safe; a design that
gets the arithmetic exactly right and places the dedup store in the wrong failure domain will duplicate
under precisely the conditions it was built for.
## 10. Key Theorems / Results Reference

The results below are the ones worth committing to memory. Each is stated with its hypothesis, because a
result quoted without its condition is the most common way an experienced engineer produces a wrong
answer confidently.

**R-01 — Admission is only as strong as its weakest conjunct.** If admission is defined as the conjunction
of syntax, semantics, authority, and ledger novelty, then the probability of a wrong dispatch is at least
the probability that the weakest conjunct is evaluated against stale or absent evidence. *Hypothesis:*
the conjuncts are evaluated at the same point in time against the same version of state. *Consequence:*
when the conjuncts are spread across services with different caches, the useful strength of admission is
that of the stalest conjunct. This is the formal reason T-06 matters: an authority predicate evaluated
180 seconds before the effect contributes almost nothing when state changes every 2.5 minutes.

**R-02 — At-most-once dispatch requires durable pre-dispatch state; at-most-once effect requires sink
cooperation.** The ledger guarantees one dispatch per key. It cannot guarantee one effect per key when the
sink ignores the key, because between dispatch and sink commit there is a window the ledger does not
control. *Hypothesis:* crash-consistent durable write of the key before dispatch. *Consequence:* every
claim of "exactly once" must name which of the two guarantees it means. The honest phrasing for an opaque
sink is "at most one dispatch, at most one effect only if reconciliation confirms".

**R-03 — Duplicate effects are proportional to uncovered retries, not to retry count.** Restating T-02:
D = N q (1-c)(1-p_sink). *Hypothesis:* independent retries and uniform coverage. *Consequence:* improving
coverage has proportional effect at any retry volume; reducing retries is multiplicative on top. A design
review that discusses retry policy without discussing coverage is discussing the smaller lever.

**R-04 — A backlog drains only at the surplus, and never at the nominal rate.** From T-03, T_drain = B /
(μ_r - λ_r), which is finite only for positive surplus. *Hypothesis:* constant rates per interval.
*Consequence:* "we added capacity equal to the peak" produces zero surplus and an unbounded queue. The
correct move under sustained overload is to reduce λ_r, because increasing μ_r is bounded by the sink.

**R-05 — Tail latency is superlinear in utilisation.** P(W>t) = ρ exp(-μ(1-ρ)t) for M/M/1. Between ρ=0.8
and ρ=0.9 with μ=20 and t=1 s, the probability rises from 0.0147 to 0.1218, a factor of 8.3 for a 12.5%
throughput gain. *Hypothesis:* Poisson arrivals and exponential service. *Consequence:* utilisation
targets for interactive tools belong near 70–80%, and a system at 95% has a one-second tail of 0.35 —
which is not a tail at all but the normal case.

**R-06 — Retry amplification is bounded by the attempt cap and multiplied by client count.** E[attempts]
≤ R, and synchronised retries make the instantaneous arrival rate up to C times the average, where C is
the number of clients retrying in phase. *Hypothesis:* independent attempts with jitter. *Consequence:*
jitter is not a refinement, it is what keeps the cap meaningful. Without jitter, E[attempts] ≤ R bounds
the total work but not the peak rate.

**R-07 — Approval is a predicate over a payload, and its scope is exactly the digest's coverage.** Two
payloads sharing a digest are indistinguishable to the approval. *Hypothesis:* the digest covers every
effect-relevant field. *Consequence:* the approval's legal scope equals the set of payloads that collide
under that digest. Any effect-relevant field outside the digest is outside the approval, regardless of
what the UI displayed.

**R-08 — Key collision probability follows the birthday bound, which applies to random keys only.** For n
values in a space M, P ≈ n²/(2M). At n=2.5e6 and M=2^64, P ≈ 1.7e-7. *Hypothesis:* values are uniformly
distributed. *Consequence:* a sequential or time-based key has a collision probability that is a property
of the numbering scheme, not of the space size. Namespacing by instance id is what converts a counter
into a random-ish key.

**R-09 — A human gate is a queue with a service rate.** The gate is a control only while its capacity
exceeds the arrival rate of what requires review. *Hypothesis:* review time is roughly stationary.
*Consequence:* at 8% of 17.4 ops/s with four-minute reviews, the gate needs 333 reviewers; the control
that scales is keyed automation with humans reserved for irreversible operations.

**R-10 — Two clocks are the minimum for period-boundary correctness.** A receipt with only processing
time cannot answer whether an effect was effective before a period closed, which is the question that
matters for accruals, cut-offs, and reversals. *Hypothesis:* event time is knowable at the source.
*Consequence:* store event time, processing time, and the authority time used for the decision; the
absence of any one of the three makes a class of audit question unanswerable.

**R-11 — Compensation is a new effect with its own failure probability.** A chain of k compensations each
succeeding with probability s leaves the all-succeeded probability s^k. *Hypothesis:* independent failures.
*Consequence:* at s=0.92 and k=6, the chance that all succeeded is 0.606, so 39% of compensation chains
end in a partial state that needs a human owner. Chains longer than two deserve an explicit state machine,
not a loop.

**R-12 — Error taxonomy is an interface, and collapsing it is an interface break.** If retryability is
inferred from prose or from a generic status, then any change to a message or a status mapping is a
behavioural change for every client. *Hypothesis:* clients rely on the taxonomy for control flow, which
they always do. *Consequence:* treating the reason enum as a versioned interface with a contract test is
the only way to change it safely.

**R-13 — Retention is derived, not defaulted.** W = max(retry horizon, reconciliation lag p99, support
resolution time) + margin. *Hypothesis:* the maximum replay age is measurable from the audit log.
*Consequence:* the default 24-hour TTL in most infrastructure is shorter than a weekend, so any
Monday-morning replay of Friday's work is admitted as new. Deriving W from the horizon converts a silent
duplicate source into an explicit number.

**R-14 — Schema permissiveness transfers control of the digest to the client.** If unknown fields are
accepted and included in the digest, a client deployment can change what is hashed without a schema
review. *Hypothesis:* the digest serialises what it is given. *Consequence:* allow extra fields for
forward compatibility, exclude them from the digest, and record their presence as an audit event.

## 11. Interview / Exam Questions

Fifteen questions follow, mixing recall, calculation, and diagnosis. Each has its own answer, its own
reasoning, and its own red flag. Answers are intentionally specific: the point of the exercise is to say
which observation would change your mind, not to recite that safety matters.

### Q1. A tool call validates against its schema and the executor dispatches it. Name the three predicates the schema did not establish, and say what component must establish each.

**Answer.** The schema established syntax only. It did not establish (1) semantic legality — that the
business state permits this operation now (period open, budget sufficient, no existing document for the
business key); (2) authority — that this principal, in this tenant, may effect this resource, evaluated at
or near effect time; (3) novelty — that this intended operation has not already been admitted or committed
under its key. Semantic legality is established by a policy component that reads authoritative business
state and returns a typed reason; authority by the executor holding its own scoped credential and
rechecking; novelty by a durable compare-and-insert on the key/digest store before dispatch.

**Reasoning.** Each predicate reads different state at a different time. Collapsing them into the schema
forces the schema to encode mutable business state, which it cannot do without becoming a policy engine
with a stale cache.

**Follow-up.** Which of the three can you evaluate locally at the executor without a remote call, and what
does that cost you in freshness?

**Red flag.** "The schema is strict, so invalid calls cannot happen." Strictness bounds shape, not
legality.

### Q2. Compute expected duplicates per day: 40,000 writes/day, 3% retried at least once, ledger coverage 0.97, sink dedup absent. Then compute the same with coverage 0.999.

**Answer.** D = N q (1-c)(1-p_sink). First: 40,000 × 0.03 = 1,200 retried; 1,200 × (1-0.97) = 1,200 ×
0.03 = 36 duplicates/day. Second: 1,200 × (1-0.999) = 1,200 × 0.001 = 1.2 duplicates/day. The ratio is
30×, matching the ratio of uncovered fractions (0.03/0.001 = 30).

**Reasoning.** Only retried operations can duplicate, and only the uncovered fraction of those becomes a
second effect. Coverage improvement is proportional; retry reduction is proportional too, but you control
coverage more directly.

**Follow-up.** If the sink deduplicates on your key for 60 seconds and your retries occur within 5
seconds, what is p_sink and how does D change?

**Red flag.** Reporting 40,000 × 0.03 = 1,200 as the duplicate count, which confuses retried operations
with duplicated effects.

### Q3. Why must an idempotency key be minted before the first dispatch and persisted, rather than derived from the request?

**Answer.** If the key is derived from a request identifier, every retry produces a new request id, hence
a new key, hence no deduplication. If the key is derived from the payload by hashing, then any payload
edit produces a new key, so the ledger cannot distinguish "the user changed the amount" from "a new
operation", and the KEY_CONFLICT detector that catches exactly that case never fires. The key must
identify the intention, which exists before the bytes and survives every attempt. That means: mint at
intent formation, persist durably before the first attempt, reuse on every attempt, and keep the digest as
a separate field so a changed payload under the same key is a detectable conflict.

**Reasoning.** Keys answer "is this the same intention"; digests answer "is this the same payload". The
pair is what makes CONFLICT expressible. Derive either from the transport and you lose the distinction.

**Follow-up.** What must be true for a key to survive a process restart mid-dispatch?

**Red flag.** "We use a UUID per call." That is a request id with a nicer name.

### Q4. A backlog of 5,400 operations exists. The recovery lane completes 28 ops/s and live arrivals continue at 16.9 ops/s. Drain time? What if live arrivals rise to 28 ops/s during the drain?

**Answer.** Surplus = 28 - 16.9 = 11.1 ops/s; T = 5,400 / 11.1 = 486 s ≈ 8.1 minutes. If live arrivals
rise to 28, the surplus is 28 - 28 = 0 and the drain time is infinite: the lane is fully occupied by new
work and the backlog is conserved indefinitely. In that case no amount of waiting helps; the only fix is
to reduce live arrivals (shed low-value work) or raise μ_r above 28.

**Reasoning.** Conservation. The backlog changes only by the difference between completion and arrival,
so a lane at parity adds capacity without adding recovery.

**Follow-up.** Which lever can you pull within one minute during an incident, and what does it cost in
user-visible failure?

**Red flag.** "We will add more workers" without checking whether the surplus is positive after the new
arrivals are counted.

### Q5. At μ = 20 ops/s, what is the probability a request waits more than one second at ρ = 0.8, and at ρ = 0.9? What do you conclude about utilisation targets?

**Answer.** P = ρ exp(-μ(1-ρ)t). At ρ=0.8: 0.8 × exp(-20 × 0.2 × 1) = 0.8 × exp(-4) = 0.8 × 0.0183 =
0.0147. At ρ=0.9: 0.9 × exp(-20 × 0.1 × 1) = 0.9 × exp(-2) = 0.9 × 0.1353 = 0.1218. A 12.5% throughput
increase multiplies the one-second tail by 8.3. Conclusion: for interactive tools, target 70–80%, and
treat 90% as an incident condition rather than an efficiency goal.

**Reasoning.** The exponential term dominates because (1-ρ) appears in the exponent. The intuition "we
have 10% spare, so we are fine" reads the wrong variable.

**Follow-up.** What does the model assume about service time, and how does a heavy-tailed downstream call
change the true tail?

**Red flag.** Computing ρ and stopping there, without the tail.

### Q6. Distinguish UNKNOWN from REJECTED and from NOT_FOUND, and say which one justifies a new key.

**Answer.** UNKNOWN means the outcome is not yet knowable: the request was dispatched and the response
was lost, or the sink is still processing. REJECTED means the sink or a gate decided not to effect the
operation, with a reason. NOT_FOUND means an authoritative query found no record for the key, which is
evidence of non-effect only if the query is authoritative and the record would be durable. None of the
three justifies a new key for a retried intention. A new key is justified only when the intention itself
changes — a different amount, a different target, a different effective date — and then it is a new
operation requiring its own approval.

**Reasoning.** A new key restarts the duplicate-detection story. Minting one to escape an ambiguous
outcome is the canonical way to convert one unknown into two effects.

**Follow-up.** What must be true about the query for NOT_FOUND to mean "not effected"?

**Red flag.** Treating a timeout as REJECTED, which makes it retryable-by-default.

### Q7. Design the rejection reasons for a write tool. How many, and what fields does each carry?

**Answer.** Four to six reason codes are enough for almost every write tool: `SCHEMA_INVALID`,
`SEMANTIC_DENIED`, `AUTHORITY_DENIED`, `KEY_CONFLICT`, `SINK_REJECTED`, `RATE_LIMITED`. Each carries
`retryable` (boolean), `remediation` (what the caller should do instead), `owner` (caller, operator, or
reviewer), and a correlation key plus `policy_version` where a policy was involved. `SCHEMA_INVALID`,
`SEMANTIC_DENIED`, `AUTHORITY_DENIED`, and `KEY_CONFLICT` are `retryable: false`; `RATE_LIMITED` is
retryable with backoff; `SINK_REJECTED` is retryable only if the sink's own reason says so.

**Reasoning.** The taxonomy exists so clients stop inferring control flow from status codes and prose.
Every field is a decision the client would otherwise have to guess.

**Follow-up.** Where must the reason be asserted in tests, and why at the outermost boundary?

**Red flag.** A single `ERROR` code with a message string, which forces every client to parse English.

### Q8. A digest over seven of nine effect-relevant fields is used to bind an approval. Compute the unbound fraction, and say why the number alone is not the risk measure.

**Answer.** U = (9-7)/9 = 0.222, so about 22% of the effect-relevant field set is outside the approval.
The number alone misleads because fields are not equal: omitting `currency` changes the effect's meaning
by orders of magnitude, while omitting a display name changes nothing. Rank omissions by consequence and
treat any missing amount, currency, target, or verb as a defect regardless of U.

**Reasoning.** Coverage is a count; risk is a weighted sum with heavy tails. The count is a screening
test that catches gross negligence, not a risk score.

**Follow-up.** Which omitted field would you approve shipping for one sprint, and what compensating check
would you add?

**Red flag.** "U is 0.22, which is below our 0.25 threshold, so it ships."

### Q9. Explain why a ledger with a 24-hour TTL produced duplicates on a Monday morning.

**Answer.** The maximum replay age exceeded the TTL. Friday's operations that timed out were reconciled on
Monday, 60+ hours later; the ledger had expired their keys, so the reconciliation presented them as new
intentions and the sink created second effects. The failure is arithmetic, not subtle: TTL < maximum
replay age. The fix is to derive W from max(retry horizon, reconciliation lag p99, support resolution
time) plus a margin, and — when the sink's own dedup window is shorter than that horizon — to add a
business-key check as a second line of defence.

**Reasoning.** Retention is a claim about the future: no replay will arrive older than W. When the claim
is violated, the ledger stops being a detector and becomes a cache.

**Follow-up.** How would you measure the maximum replay age from existing data?

**Red flag.** "We extended it to 48 hours", which is still shorter than a weekend.

### Q10. A client retries with attempt cap 4 and per-attempt success probability 0.75. Compute expected attempts and the load multiplier at 17.4 intents/s. What changes if the sink degrades to s = 0.25?

**Answer.** E = (1 - 0.75^4)/(1 - 0.75) = (1 - 0.3164)/0.25 = 0.6836/0.25 = 2.734 attempts. Load =
17.4 × 2.734 = 47.6 ops/s at the sink. At s = 0.25 with the same cap: 0.25^4 = 0.003906; (1 - 0.003906)
/0.75 = 0.996094/0.75 = 1.328 attempts, so load is 17.4 × 1.328 = 23.1 ops/s — but successful completions
fall to a quarter, so the *useful* throughput collapses while attempt load rises above intent rate. That
asymmetry is the storm: the sink sees more work and delivers less.

**Reasoning.** The cap bounds work per intent but not the aggregate when many intents retry
simultaneously. A retry budget expressed against recent successes bounds the aggregate instead.

**Follow-up.** At what per-attempt success rate does a retry budget of 10% get exhausted, and what
happens then?

**Red flag.** Reporting the expected-attempts number as throughput without netting successes.

### Q11. An outbox row is written before dispatch and marked `COMMITTED` after the sink responds. Under what condition is it a valid receipt, and how do you make the condition true?

**Answer.** Valid only if the `COMMITTED` transition happens in a transaction that also persists the
sink's own transaction identifier, and only if a row that never receives that identifier is distinguishable
from one that did. Otherwise the row is evidence of dispatch, not of effect. Make it true by separating
two fields: `dispatch_state` (written pre-dispatch) and `outcome_state` with `outcome_evidence` (written
only from an authoritative response, carrying the sink's id). A timeout leaves `dispatch_state =
DISPATCHED` and `outcome_state = UNKNOWN`, which is the truth.

**Reasoning.** One enum spanning two rungs of the evidence ladder guarantees that some reader will
interpret the top rung as reached. Two fields make the ambiguity unrepresentable.

**Follow-up.** What query resolves UNKNOWN, and what makes that query authoritative?

**Red flag.** "The row says COMMITTED, so it is committed."

### Q12. A tool's description is fetched at runtime from a shared registry. What can change routing without any schema change, and what do you pin?

**Answer.** The description text, the parameter descriptions, and any examples are model-visible and
influence tool selection and argument filling. An attacker who can edit a description or a shared resource
that the description references can change which tool runs or which arguments are used, with no schema
diff. Pin the manifest by digest, review changes explicitly, keep the allowed tool-and-argument set
enforced outside the model, and treat description text as data with provenance rather than authority.

**Reasoning.** Discovery is a data path into the model's context, and the model's output is the proposal.
Anything that can write to a data path can influence the proposal; only components outside the model can
constrain the effect.

**Follow-up.** What is the smallest allowlist that still permits useful adaptation, and who owns it?

**Red flag.** "The descriptions come from our own internal registry, so they are trusted."

### Q13. Price a compensation plan: six sequential compensation steps, each succeeding with probability 0.92. What is the probability of a fully compensated outcome, and how do you handle the rest?

**Answer.** 0.92^6 = 0.606, so there is a 39.4% chance that at least one step failed and the entity is in a
partial state. Handle it by modelling compensation as a first-class operation with its own key, approval,
and typed outcomes, and by making "partially compensated" an explicit state with a named owner and a
follow-up task, rather than an error to retry. Alert on chains longer than two and on any entity that
remains partially compensated past its SLA.

**Reasoning.** Compensation is a sequence of independent effects, so its reliability multiplies down.
Retrying a failed compensation without a new key risks a double compensation, which is a worse state
than the one being fixed.

**Follow-up.** Which of the six steps is irreversible, and why is it in the chain at all?

**Red flag.** "We re-run the compensation job until it shows green."

### Q14. What single audit field, if missing, makes period-boundary disputes unanswerable, and why?

**Answer.** Event time — the time the business fact took effect, as distinct from the time your system
processed the record. Without it, a receipt cannot be placed relative to a period close, so a late-posted
invoice cannot be attributed to the correct period, and a reversal cannot be distinguished from a
different-period correction. The complete minimum is three stamps: event time, processing time, and the
authority time used for the decision.

**Reasoning.** Period boundaries are cut in event time, while most systems store processing time because
that is what their logging framework provides. The mismatch is invisible until an auditor asks which
period the effect belongs to.

**Follow-up.** Where does event time come from when the source system only sends processing time?

**Red flag.** "We log the timestamp", where the timestamp is when the row was inserted.

### Q15. Your tool passes all contract tests and duplicates appear only during downstream incidents. What is the most likely structural cause, and which measurement confirms it?

**Answer.** The dedup store and the executor are not in the same failure domain — typically the key store
is remote or shared, and during a downstream incident the key store is slow or unavailable while dispatch
still proceeds (or dispatch is retried from a path that skips the store). Coverage effectively drops to
zero for the affected interval while retries continue. Confirmation: measure key-store availability and
latency during the incident window and compare with the duplicate cohort's timestamps; if duplicates
cluster exactly where key-store latency exceeded the executor's timeout, the structural cause is
confirmed. The correction is to fail closed — refuse dispatch when the key store is unavailable — and to
co-locate the store with the executor.

**Reasoning.** A detector that is unavailable during the failure it exists to detect is not a detector.
Fail-closed converts an availability problem into an explicit outage instead of a silent data problem.

**Follow-up.** What is the business cost of failing closed for ten minutes versus duplicating for ten
minutes, and who decides?

**Red flag.** "The contract tests pass, so the ledger works."

## 12. Quick Reference Card

| Situation | Rule | Critical assumption | Avoid | Consequence if wrong |
|---|---|---|---|---|
| Designing a write tool | classify effect class first (E0–E4) | sink reversibility is known | treating a create as an upsert | duplicate documents that look like edits |
| Before dispatch | durable compare-and-insert on key+digest | store is crash-consistent and local | writing the record after the response | effects with no ledger entry |
| Timeout received | promote to UNKNOWN, reconcile on the key | an authoritative query exists | retrying with a new key | two effects, one intention |
| Payload differs under same key | reject as KEY_CONFLICT | digest covers all effect fields | updating the stored digest | ledger becomes last-writer-wins |
| Approval flow | bind digest + approver + expiry + policy version | executor verifies at dispatch | free-text approval | approval scope silently grows |
| Capacity plan | compute peak, backlog, and drain surplus separately | rates constant per interval | planning against the daily mean | queue grows during the peak hour |
| Utilisation target | 70–80% for interactive tools | service times roughly exponential | running at 90%+ for efficiency | one-second tail ~0.12 at ρ=0.9 |
| Dedup retention | W = max(retry, reconciliation p99, support) + margin | maximum replay age is measurable | a default 24-hour TTL | weekend replay duplicates |
| Sink ignores your key | treat as E3 and reconcile | sink state is queryable | claiming exactly-once | unprovable duplicate suppression |
| Incident with growth | check whether surplus is positive before adding lanes | arrival rate known | adding capacity at parity | no drain, ever |

**Three sentences to keep.** A schema grants shape, never authority. A receipt comes from the sink, never
from your own optimism. A key identifies an intention, so it outlives the request that carried it.

**Status.** Draft. All invariants and derivations in sections 4 and 10 are DERIVED from the stated
definitions; all parameter values in sections 5, 9, and 11 are ILLUSTRATIVE constructions whose
arithmetic was recomputed by hand. No deployed integration, vendor behaviour, measured incident, or
company result is asserted. Protocol details are version-specific and must be rechecked against the
current specification before being relied on; the original catalog source bodies were not available when
this sheet was written.
