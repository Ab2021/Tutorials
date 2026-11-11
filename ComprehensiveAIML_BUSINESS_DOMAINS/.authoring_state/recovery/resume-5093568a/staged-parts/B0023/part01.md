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
VALIDITY: T-02 and its cost form hold only while coverage is uniform across the population and retries
  are independent of the failures that cause them. Both conditions are routinely violated during a
  correlated outage, when coverage falls and retries rise together; in that case the estimate is a lower
  bound, and the design question becomes whether fail-closed is cheaper than duplicating.
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
  At d = 30 the same ledger needs 11.148 * 30 / 7 = 47.8 GB.
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

**Validity conditions for the variants.** Stated once, compactly, because each one is the sentence a
reviewer should ask you to defend:

- T-03 holds only while rates are constant within an interval and the recovery lane is dedicated. If the
  lane shares workers with the live lane, μ_r is not a constant and the piecewise integration must use
  the shared-resource model instead.
- T-04 holds only for Poisson arrivals, exponential service, and a single FIFO server. With a bounded
  pool and a queue cap, the interesting quantity becomes blocking probability, not waiting time.
- T-05 holds only for independent attempts under a hard cap. Under correlated failure the expectation
  collapses to R, and the peak rate — not the mean — is what the sink experiences.
- T-06 holds only when ν is measured for authority-relevant changes, not all changes. Using total change
  rate inflates P(stale) and pushes teams toward needlessly short tokens.
- T-07 holds only as a screening count. It says nothing about consequence and must be paired with a
  named-field review of amount, currency, target, and verb.
- T-08 holds only while record size is stable. Adding a per-record payload snapshot for audit doubles b,
  which is why the snapshot belongs in the audit store rather than the dedup ledger.
- T-09 holds only for uniformly distributed keys in a single namespace. Multi-region key spaces that are
  merged later break the uniformity the bound assumes.
- T-10 holds only in steady state. During a ramp, in-flight count lags the product, and capacity sized
  from L alone will be short exactly when the ramp is steepest.


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
