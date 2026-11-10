---
title: "Agent Boundaries and Request Ownership"
topic: "Agent boundaries and request ownership"
collection: "30_agentic_systems_workflows_and_orchestration"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Agent Boundaries and Request Ownership

## 1. File Header

> **Purpose:** decide which component owns each decision, each durable record and each external effect on an agent request path, so that a retry cannot silently duplicate authority or work.
> **Prerequisites:** HTTP identity propagation, authorization basics, transaction isolation, at-least-once delivery.
> **Key outcome:** given any agent request, name the owner of every state fragment, count the maximum effect attempts, and find the layer that will duplicate work first.
> **Not covered here:** one vendor's agent SDK, model training, or legal interpretation of access policy.
> **Level:** foundation; the arithmetic is deliberately small so it can be re-done on paper.
> **Evidence:** invariants DERIVED from stated preconditions; every numeric example is ILLUSTRATIVE and computed in place; nothing here is a measured deployment result.
> **Retrieval cues:** duplicate side effects → §8.1 and §8.6; runaway retries → AB-01 in §4.1 and §9.2; unclear ownership → §7.4 and §8.7.

The word "agent" carries at least four meanings in practice: a model loop that proposes actions, a product workflow that users invoke, a runtime that executes steps, and a delegated authority acting for a person. Merging them into one box is the root defect this sheet exists to remove. Once merged, no one can say who authorized a refund, who owns the retry, or which record wins when two workers disagree.

The sheet therefore works with a narrow definition. An agent is the bounded workflow that proposes and advances work. It has a budget, a state machine and a name in the audit log. It has no identity of its own and no standing permission; those come from the principal on whose behalf it runs and from the policy evaluated at the moment of action.

Ownership, in this sheet, is a write permission enforced by a credential rather than a convention. If two components can write a fragment, the fragment has two owners and no authoritative value. Declaring an owner in a design document is not ownership; revoking the other component's write permission is.

**Takeaway:** name the owner of every decision, record and effect before discussing autonomy, and enforce ownership with credentials rather than documentation.

Reading paths. For an incident, jump to §8 and use its discriminating checks. For a design review, use §7.1 for the layer table and §7.4 for the owner matrix. For a cost argument about retries, use §4.3 and §9.2, which quantify amplification rather than asserting it.

## 2. Concept Map / Mental Model

```text
[1] caller (browser, CLI, another service)
      authenticated identity: tenant, subject, roles, policy version
        |
[2] edge gateway        strips untrusted headers, resolves tenant from token
        |
[3] control plane       admit / reject / quota / schedule  <-- policy bundle (versioned)
        |
[4] run state store     versioned state machine: the ONLY durable owner of progress
        |
[5] context assembler   builds a per-call prompt from transcript + retrieval
        |
[6] model               proposes an action; holds no authority and no durable state
        |
[7] tool adapter        validates schema, re-checks policy, attaches effect key
        |
[8] external system     owns the business effect and the receipt
        |
[9] audit trail         append-only, attribution-complete, separate credentials
```

Nine boxes, seven contracts. Each arrow is a contract with an input schema, an output schema, a credential, a failure mode and a recovery rule. A hop that lacks any of those five is not a boundary; it is an undefined interface that will be crossed by accident.

Two arrows deserve attention because they are the ones teams collapse under delivery pressure. The arrow from model to adapter is where proposals gain authority; if the adapter trusts the proposal because a model produced it, authorization has moved into generated text. The arrow from control plane to state store is where versions are enforced; if the runtime writes state directly, the state machine has no single writer.

Authority flows rightward and downward. Evidence flows leftward and upward. Nothing skips a version check in either direction, and no evidence path is allowed to depend on the model's own narration of what happened.

**Takeaway:** the request path is a chain of contracts, and the two contracts most often collapsed are model-to-adapter and control-plane-to-store.

## 3. Foundational Definitions

| Term | Meaning in this sheet | Unit / type | Boundary test that distinguishes it |
|---|---|---|---|
| Principal | authenticated actor whose permissions are evaluated | identity tuple | can policy evaluate it at the exact moment of action |
| Tenant | administrative isolation domain | opaque identifier | does every storage read carry it as a predicate |
| Session | tenant-scoped interaction container | durable record | can it outlive the run that created it |
| Run | one bounded attempt to advance a request | state machine instance | does it have a version and one terminal state |
| Attempt | one dispatch of a run to a worker | integer counter | is a timed-out dispatch still counted |
| Control plane | admission, quota, policy and scheduling authority | logical service | can it reject work before any model call |
| Runtime worker | replaceable execution capacity | process | can a different worker resume the same run |
| Context assembler | builds the per-call prompt | pure function of state | can you reconstruct the exact context of a call |
| Model | proposes next action as text or a tool call | external dependency | would its output change without authorization |
| Tool adapter | validates, authorizes, translates, deduplicates | protocol boundary | can it refuse schema-valid input |
| Effect | irreversible change in an external system | business fact | does it have a receipt or a queryable status |
| Effect key | stable identifier for one intended effect | opaque string | does it survive retries and process restarts |
| Receipt | external system's own record of the effect | record | can reconciliation read it without the worker |
| Lease | time-bounded claim to act on a version | duration | does expiry stop a paused holder from writing |
| Fence token | monotonic generation attached to protected writes | integer | does the receiver compare it or only log it |
| Capability | granted scope for one action family | set of scopes | is it narrower than the principal's full rights |
| Budget | cap on branches, calls, tokens or wall clock | counter | does exhaustion produce a terminal state |
| Owner | component permitted to write a fragment | credential binding | can another component write it without an error |
| Request id | identifies one transport attempt | opaque string | does it change when the same work is retried |
| Replay | re-deriving state from recorded events | mode of a reader | does the reader hold adapter write credentials |
| Reconciliation | resolving an unknown external outcome | durable queue | is there an owner and an age metric per item |
| Quota | admission-time limit on work per tenant | counter per window | is it checked before any model call is billed |
| Delegation depth | number of links between principal and executor | integer | is it capped, and does the cap decrement on spawn |
| Terminal state | state with no outgoing transition | enum member | can any writer leave it without a new run |

Definitions alone do not settle arguments, so each row carries a test that produces an observable difference. Two components that both pass every test in the table are the same owner under two names, which is the ambiguity to remove.

Three pairs are commonly confused and worth separating explicitly:

- **Session versus run.** A session is the container; runs are attempts to advance it. A retry creates a new attempt, sometimes a new run, never a new session.
- **Capability versus permission.** A capability is what a component could do with the credential it holds; permission is what policy allows now. A tool with a broad capability and narrow permission is safe only while the permission check sits at the effect boundary.
- **Effect key versus request id.** A request id identifies one transport attempt; an effect key identifies one intended business outcome across all attempts.

**Takeaway:** every definition in this sheet earns its place by producing a test whose two outcomes look different in logs.

## 4. Core Formulas with Full Derivation

Every block uses the anatomy FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION. All numbers are ILLUSTRATIVE, chosen so the arithmetic is checkable by hand.

### 4.1 Base case

```text
FORMULA [AB-01] effect attempts under independent layer retries
  A = (1 + r1) * (1 + r2) * ... * (1 + rL)
  A = R^L when every layer uses the same retry count r
  R = 1 + r

WHERE
  r_i = retries configured at layer i        (integer, >= 0)
  L   = number of layers that retry the same call independently (>= 1)
  A   = maximum attempts reaching the external system (attempts/call)
  R   = per-layer amplification factor (dimensionless)

CONSTRAINT
  valid only if retries at each layer are independent and each retry actually
  re-invokes the downstream call. A layer that retries only on connection-refused
  before send contributes less than R; a layer that retries on timeout after send
  contributes the full factor and is the dangerous one.

NUMERICAL EXAMPLE 1 (single layer)
  client only, r = 2: R = 3, L = 1, A = 3^1 = 3 attempts
NUMERICAL EXAMPLE 2 (three layers)
  edge r=2, runtime r=2, adapter r=2: R = 3, L = 3, A = 3^3 = 27 attempts
NUMERICAL EXAMPLE 3 (four layers, client added)
  L = 4: A = 3^4 = 81 attempts for one intended business action

INTERPRETATION
  The growth is multiplicative in the exponent, not in L: going from 3 layers to 4
  multiplies attempts by 3, and each of those attempts carries a full model or API
  charge. A 2 s downstream call with 3 attempts per layer also multiplies worst-case
  latency: 2 * 3^4 = 162 s, which exceeds any client deadline that started at 30 s.
```

```text
FORMULA [AB-02] boundary completeness
  C = D / (5 * B)
  missing_classes = 5 * B - D

WHERE
  B = number of logical boundaries on the request path
  D = number of declared contract elements present
      (input schema, output schema, credential, failure mode, recovery rule)
  5 * B = required declarations

CONSTRAINT
  "declared" means recorded in the interface definition and checkable by a test,
  not described in a design document. Each boundary needs all five; a boundary
  missing its recovery rule will be crossed during an incident with no plan.

NUMERICAL EXAMPLE
  B = 6 boundaries, required = 30 declarations
  observed: 24 present -> C = 24/30 = 0.80, missing classes = 6
  if the 6 missing are spread over 6 different boundaries, every boundary is
  individually incomplete; if they sit on one boundary, one arrow is undefined
  both numbers give C = 0.8 and describe very different risk

INTERPRETATION
  The scalar hides the arrangement, so report both the ratio and the shape.
  Recovery rules are the element most often absent and the one an incident needs
  first; a boundary that lists only schemas has documented the easy half.
```

```text
MECHANISM BLOCK [AB-03] single-writer ownership of a state fragment
  preconditions:
    the fragment has exactly one credential able to issue writes
    writes are conditional on an expected version
  state transition:
    attempt(fragment, expected_v, new_state) -> accepted(v+1) | rejected(v_changed)
  invariant:
    for any fragment and any point in time, at most one writer credential is valid,
    and at most one accepted write exists per version
  justification:
    with two valid writers, "last write wins" is the de facto rule, so a cancelled
    run can be revived by a slower writer and the audit log shows two legal writers
  worked trace:
    W1 holds credential for run RUN-31 v=11; operator revokes it at 10:00:00Z
    W2 obtains credential at 10:00:04Z and writes v=11 -> accepted, now v=12
    W1 writes at 10:00:09Z with expected v=11 -> rejected (v_changed)
    one accepted write, one rejection: ownership was enforced, not assumed
  failure boundary:
    if the revocation takes 10 s to propagate and W1 writes at 10:00:08Z, both
    writes can be accepted; the fence token, not the revocation, is what closes
    that window
```

### 4.2 General case

The general case replaces "one request, one call" with a budget-limited plan whose branches can fail, retry and spawn sub-work.

```text
FORMULA [AB-04] total call budget under nested branching
  N_total = c * (1 + s1 * (1 + s2 * (1 + ... )))
  flat case: N_total = c * (1 + s) when each branch spawns s children of equal depth

WHERE
  c = calls made directly by one agent instance (calls/instance)
  s = sub-agents spawned per instance (instances/instance)
  depth d assumed 1 in the flat case; total instances = 1 + s

CONSTRAINT
  each spawned instance inherits a cap equal to the parent's, and the parent's cap
  is not decremented by children. If the budget is enforced per instance rather
  than per tree, the tree total is what the numbers below show, not the headline.

NUMERICAL EXAMPLE
  c = 6 calls, s = 5 sub-agents, per-instance token cap 120,000
  total instances = 1 + 5 = 6
  total calls = 6 * 6 = 36
  total token ceiling = 6 * 120,000 = 720,000
  headline cap 120,000 understates the ceiling by 720,000 / 120,000 = 6x
  if each of the 5 children spawns 5 more: instances = 1 + 5 + 25 = 31
  calls = 6 * 31 = 186; token ceiling = 31 * 120,000 = 3,720,000 = 31x headline

INTERPRETATION
  A per-instance cap is a per-node cap, and the tree is not a node. The 31x figure
  is not a model of misbehaviour; it is what the configuration permits. Either
  decrement a shared tree budget or cap depth explicitly, and record which one.
```

```text
FORMULA [AB-05] effective capability along a delegation chain
  effective_scope = intersection of scope_1 .. scope_k
  |effective_scope| <= min over i of |scope_i|

WHERE
  scope_i = capability set granted at link i of the chain (user -> run -> tool)
  k = chain length (links)
  |.| = cardinality of the scope set

CONSTRAINT
  intersection is only sound when each link's grant is a subset of its parent's.
  A link that can grant anything it likes (a "deputy" that mints tokens) breaks the
  bound and makes the chain a privilege escalator.

NUMERICAL EXAMPLE
  user scopes: {orders.read, tickets.read}                    |2|
  run scopes : {orders.read, tickets.read, tickets.write}     |3|
  tool needs : {tickets.write}
  intersection = {orders.read, tickets.read}  -> tickets.write absent -> DENY
  with a minting deputy, the run could add tickets.write legitimately; the
  intersection then equals the deputy's own scope, not the user's 2-scope set
  numeric gap: 3 granted scopes against 2 delegable scopes = 1 escalatable scope

INTERPRETATION
  The narrowest link binds, which is the property that makes delegation auditable.
  Escalation appears exactly where a link is allowed to widen instead of narrow.
```

```text
FORMULA [AB-06] attribution completeness of an effect record
  fields_required = tenant, principal, run, attempt, boundary, policy_version  (6)
  attribution = present / 6
  evidence_gap = 1 - attribution

WHERE
  present = count of the six fields actually written with the effect record
  a record missing any field cannot be joined back to an authorization decision

CONSTRAINT
  the six are a minimum, not a design. Adding the effect key and the receipt id
  makes reconciliation possible; the six only make attribution possible.

NUMERICAL EXAMPLE
  5 of 6 fields present: attribution = 5/6 = 0.8333, gap = 0.1667
  at 8,000 effects/day, effects with an attribution gap = 8,000 * 0.1667 = 1,334/day
  each such effect cannot be traced to a policy version, so a policy review of
  those effects is impossible rather than merely slow

INTERPRETATION
  A missing field is not a formatting defect; it removes a join. The join that
  matters most in practice is run-to-policy_version, which is the one teams drop
  because it is not needed until an audit.
```

### 4.3 Variants

```text
VARIANT A [AB-07] timeout budget across hops
  T_max = t_model + H * t_hop
  deadline_ok iff T_max * tries <= client_deadline

WHERE
  t_model = one model call wall clock (s), H = number of synchronous hops
  t_hop = per-hop overhead including serialisation (s)
  tries = attempts the client makes

NUMERICAL EXAMPLE
  t_model = 900 ms, H = 6 hops, t_hop = 40 ms -> T_max = 900 + 240 = 1,140 ms
  6 model calls per attempt -> 6 * 1,140 = 6,840 ms = 6.84 s per attempt
  client deadline 30 s -> floor(30,000 / 6,840) = 4 attempts fit
  with 3 attempts configured: 3 * 6,840 = 20,520 ms, leaving 9,480 ms headroom
  if model p99 triples to 2,700 ms: 6 * (2,700 + 240) = 17,640 ms per attempt
  2 attempts = 35,280 ms > 30,000 ms -> the deadline fails midway through attempt 2

INTERPRETATION
  Hop overhead is 240/1,140 = 21.1% of a call's latency at these values, so it is
  not noise; every added synchronous boundary costs roughly a fifth of a model call.
  The p99 case is the design case, because a retry policy sized on p50 is a policy
  that abandons work at p99.
```

```text
VARIANT B [AB-08] duplicate exposure without an effect key
  P(at least one duplicate) = 1 - (1 - p_lost)^A
  expected duplicates = n_ops * (1 - (1 - p_lost)^A)

WHERE
  p_lost = probability a submitted call's response is lost but the effect commits
  A = attempts per intended operation (from AB-01)
  n_ops = intended operations per day

NUMERICAL EXAMPLE
  p_lost = 0.002 per attempt, A = 3
  P(duplicate per op) = 1 - 0.998^3 = 1 - 0.994012 = 0.005988
  n_ops = 4,000/day -> expected duplicates = 4,000 * 0.005988008 = 23.95 per day
  with A = 27: P = 1 - 0.998^27 = 1 - 0.947381 = 0.052619 -> 210.48/day
  adding an adapter that rejects repeats: duplicates = 0 by construction, while
  suppressed attempts remain 27 - 1 = 26 per intended operation

INTERPRETATION
  The exponent is the amplifier: the same 0.2% loss probability yields 23.95 or
  210.48 duplicates per day depending only on how many layers retry. The effect key
  does not reduce attempts; it removes the duplicates those attempts would create.
```

```text
VARIANT C [AB-09] collapse penalty
  exposed_decisions = D_defect * n_runs_per_day * days_undetected
  blast_ratio = exposed_decisions / total_decisions

WHERE
  D_defect = defects introduced per day in the collapsed component
  n_runs_per_day = runs whose decisions flow through the collapsed component

NUMERICAL EXAMPLE
  control plane and worker share one process; a policy-version read omission is
  introduced at 1 defect per week and detected after 3 days
  n_runs_per_day = 2,400
  exposed = (1/7) * 2,400 * 3 = 1,028.6 decisions -> about 1,029
  with the layers separated, the same defect affects only runs scheduled by the
  affected replica: 5 of 20 replicas -> 2,400 * 5/20 * (3/7) = 257 decisions
  ratio = 1,029 / 257 = 4.0 exactly (20 replicas / 5 affected)

INTERPRETATION
  Separation does not reduce the defect rate; it reduces the population exposed.
  The ratio here is the replica count divided by the affected replica count, which
  is why collapse penalties scale with deployment width rather than with code size.
```

```text
VARIANT D [AB-10] ownership ambiguity cost
  ambiguous_fragments = fragments with more than one write-capable credential
  ambiguity_rate = ambiguous_fragments / total_fragments
  review_cost_min = ambiguous_fragments * minutes_per_disagreement

WHERE
  fragments = independently written state items (session status, budget counter,
  retry counter, approval record, cancellation flag, effect table, audit cursor)

NUMERICAL EXAMPLE
  total fragments = 7
  two have two writers (budget counter shared by worker and scheduler; retry
  counter shared by adapter and runtime)
  ambiguity_rate = 2/7 = 0.2857
  disagreements per ambiguous fragment per week = 1.5 -> 3/week
  review cost = 3 * 35 min = 105 min/week = 1.75 h/week
  annualised = 1.75 * 52 = 91 h/year, against roughly 4 h to revoke a credential

INTERPRETATION
  Ambiguity has a maintenance price that is paid in a currency engineers recognise.
  The comparison is deliberately unfair in the right direction: the fix is a
  credential change, the cost is recurring review.
```

```text
MECHANISM BLOCK [AB-11] retry safety as a classification of failure points
  preconditions:
    every outbound call is made through an adapter that can report whether bytes
    were sent, and the receiver can be queried by a stable identifier
  state transition:
    classify(error) -> before_send | after_send_known_failed | after_send_unknown
    retry allowed:  before_send (any attempt budget)
                    after_send_known_failed (only with new stable key refused by
                    design, i.e. the intent is genuinely new)
                    after_send_unknown: NEVER without a lookup
  invariant:
    a retry of an after_send_unknown call is a new business effect attempt, so it
    must be treated as a duplicate risk rather than as a recovery
  justification:
    TCP and TLS failures before the request line are indistinguishable from an
    accepted request whose response was lost, unless the receiver can be queried;
    therefore any layer that maps all transport errors to "retryable" converts a
    recovery mechanism into a duplication mechanism
  worked trace:
    30 s read timeout on a 4,200-unit refund:
      t=0 submit, t=1.2 receiver commits, t=30 client times out
      classification: after_send_unknown
      permitted: query by key, then retry only if the query says absent
      forbidden: retry with a new key, which creates a second refund of 4,200
  failure boundary:
    a receiver with no lookup endpoint makes unknown permanently unknown; the only
    safe policy is to park and page a human, and the cost of that policy must be
    sized (see 4.3 variant A for the attempt arithmetic it avoids)
```

```text
VARIANT E [AB-12] human-in-the-loop cost of the unknown queue
  reviewer_minutes_per_day = n_unknown * t_each / 60
  fte_equivalent = reviewer_minutes_per_day / (60 * 7.5)

WHERE
  n_unknown = unresolved effects per day requiring a decision
  t_each = mean minutes to resolve one item from the effect key and receipt
  7.5 = productive hours in a reviewer day (dimensionless, an assumption)

NUMERICAL EXAMPLE
  1 in 200 effects become unknown at 4,000 effects/day -> n_unknown = 20
  t_each = 4 min -> reviewer_minutes_per_day = 80, i.e. 1 h 20 min
  fte_equivalent = 80 / 450 = 0.178 -> about 18% of one reviewer
  halving the unknown rate via a lookup-by-key API saves 40 min/day
  a lookup API that costs 2 engineering days to integrate pays back at 20 items/day
  within about 24 working days (2 * 8 h = 16 h; 16 h / (40/60 h per day) = 24 days)

INTERPRETATION
  The unknown queue is a recurring operating cost, not an incident. The break-even
  above is deliberately crude - it ignores on-call load and error cost - but it
  shows the shape: integration effort is compared against minutes per day, so a
  small daily rate still justifies the work within a month.
```

### 4.4 Limits and counterexamples

Each formula above fails somewhere specific, and the failure is more instructive than the formula.

AB-01 fails when retries are not independent: an adapter that retries internally and a client that retries on the same timeout can produce the product without either team knowing the other retries. The observable test is a single request with a forced 30 s downstream hang and a count of arrivals at the external system; if the count exceeds the configured r + 1 at one layer, another layer is retrying.
AB-01 also fails to bound cost when each attempt is not equal: attempt 3 of a run re-sends a longer context than attempt 1, so multiplying attempts by a single per-attempt price understates spend. The correction is to price attempts at their own context size, which is the whole point of §9.3 in the companion lifecycle sheet.
AB-02 fails to capture severity. Six missing declarations on six boundaries and six on one boundary are equally incomplete by the ratio and not equally dangerous, because one undefined arrow is a single incident surface while six undefined arrows include the effect boundary. Report the ratio and name the arrows.
AB-04's flat form fails as soon as depth exceeds one, and depth is exactly what agents do by default when a sub-task looks separable. The correction is to compute the tree total (1 + s + s² ...), which at s = 5 and depth 3 gives 1 + 5 + 25 + 125 = 156 instances, not 31.
AB-05 fails under a minting deputy, and the failure is silent because the intersection still produces a plausible set. The test is to request a scope the user does not hold and observe whether any link grants it.
AB-06 fails when fields are present but wrong: a run id written from a stale in-memory value passes the count and breaks the join. Presence is necessary and not sufficient; the check is a join against the run table, not a schema check.
AB-08 assumes loss events are independent across attempts. They are not when a partition causes all attempts in a window to lose responses, so the expected value understates clustering even though it correctly bounds the long-run rate.
AB-09 assumes the defect lands uniformly across days; a defect introduced just before a low-traffic weekend produces a smaller exposed count and a larger per-day severity when traffic returns.
AB-10 assumes disagreements are detected. Undetected ambiguity is invisible in review counts, which means the measured 1.75 h/week is a floor rather than an estimate.
AB-11 fails whenever the adapter cannot tell whether bytes were sent, which is the normal case for pooled connections where the failure surfaces on the next use. The correction is not a better classifier but a receiver-side lookup keyed by intent, and the classification then becomes a hint rather than a decision.
AB-12 fails when the unknown rate is bursty: 20/day averaged over a month can be 0/day for three weeks and 400 on the day a provider degrades. Averaging hides the staffing requirement, which is set by the burst, not the mean.

## 5. Worked Numerical Example

**ILLUSTRATIVE trace.** One support request, one tenant, six boundaries, one external ticket creation that gets retried at three layers. All values are constructed; the arithmetic is recomputed in the checks.

Setup:

- Tenant T7, principal `noor`, roles {support, billing.read}, policy version 22.
- Request RQ-8802 arrives at 10:00:00Z; run RUN-31 admitted at 10:00:00.180Z.
- Plan: 4 branches, up to 3 calls each, per-run token cap 120,000, wall-clock cap 90 s.
- One effecting step: create a ticket in an external tracker, base latency 2.0 s.
- Retries configured at three layers: edge 2, runtime 2, adapter 2. Client adds 2 more attempts.
- Six synchronous hops at 40 ms; one model call at 900 ms; 6 model calls per attempt.

Step 1, admission. The control plane evaluates four predicates in one statement: tenant binding, role sufficiency, quota remaining, run-state legality. It admits at 10:00:00.180Z, 180 ms after arrival, of which about 40 ms is the hop and the rest is policy evaluation.

Step 2, attempt bound. The declared caps give A_max = b × c = 4 × 3 = 12 call attempts inside the plan. That is a budget alarm threshold, not twelve business effects, because only calls that cross an effecting adapter can change the world; here exactly one step does.

Step 3, effect amplification. The intended operation is one ticket. With R = 3 at four layers, A = 3^4 = 81 attempts. The adapter holds an effect key computed from tenant, run, step and canonical payload: `T7/RUN-31/create_ticket/v1`. Under the key, 81 attempts collapse to at most one accepted effect and 80 suppressed attempts.

Step 4, latency. One attempt costs 6 × (900 + 240) = 6,840 ms. The 90 s wall-clock cap admits floor(90,000 / 6,840) = 13 attempts of model work. The client deadline of 30 s admits floor(30,000 / 6,840) = 4. The tighter of the two is the client's, so the run's effective attempt ceiling is 4, not the configured 12.

Step 5, failure. At 10:00:14Z the tracker hangs. The adapter times out at 10:00:16Z after its 2.0 s budget, retries twice (10:00:16Z, 10:00:18Z), then returns a transport error. The runtime retries the step twice, each carrying the same effect key, and the edge retries the whole request twice because it sees a 5xx.

Step 6, arrivals. Attempts reaching the tracker: 3 (adapter) × 3 (runtime) × 3 (edge) × 3 (client) = 81, consistent with A = 3^4. Wall-clock cost of those attempts at 2.0 s each is 162 s of downstream time, spread over a client deadline that ended at 10:00:30Z. Most attempts therefore occur after the client already gave up, which is the signature of amplification: work continues to be produced for a request nobody is waiting for.

Step 7, token spend. Six model calls per attempt at 9,500 mean context tokens gives 57,000 tokens per attempt. Four attempts fit inside the deadline, so 4 × 57,000 = 228,000 tokens, against a per-run cap of 120,000 — the cap is exceeded before the attempt ceiling is reached, so the budget is the binding constraint and the run ends in `budget_exhausted`, not `failed`. That distinction matters: `failed` invites a retry, `budget_exhausted` should not.

Step 8, attribution. The accepted effect record carries tenant T7, principal noor, run RUN-31, attempt 4, boundary `adapter.createticket`, policy version 22, effect key and receipt id: 8 fields, attribution 6/6 = 1.0 on the required set.

**Independent arithmetic re-check.**

```text
check 1 (attempt bound inside the plan)
  b = 4, c = 3 -> 4 * 3 = 12
  cross-check: branches enumerated 4 times, 3 calls each, 4 + 4 + 4 = 12  -> agrees

check 2 (layer amplification)
  3^2 = 9; 3^3 = 27; 3^4 = 81
  cross-check by division: 81 / 27 = 3 (one layer adds exactly one factor)
  cross-check by multiplication: 3 * 3 = 9, 9 * 3 = 27, 27 * 3 = 81

check 3 (attempt latency)
  900 + 240 = 1,140 ms per model call
  6 * 1,140 = 6,840 ms
  cross-check: 6 * 900 = 5,400 and 6 * 240 = 1,440; 5,400 + 1,440 = 6,840  -> agrees
  hop share: 1,440 / 6,840 = 0.2105 -> 21.05%

check 4 (deadline fit)
  floor(30,000 / 6,840): 6,840 * 4 = 27,360 <= 30,000; 6,840 * 5 = 34,200 > 30,000
  -> 4 attempts, headroom 30,000 - 27,360 = 2,640 ms
  wall-clock cap fit: 6,840 * 13 = 88,920 <= 90,000; * 14 = 95,760 > 90,000 -> 13

check 5 (token spend)
  6 * 9,500 = 57,000
  4 * 57,000 = 228,000
  228,000 / 120,000 = 1.9 -> cap exceeded by 90%
  the cap is crossed during attempt 3: 3 * 57,000 = 171,000 > 120,000
  so the run should terminate inside attempt 3, not attempt 4; recompute the
  attempts actually completed = 2 full attempts (114,000 tokens) plus a partial
  third, and the token arithmetic above is the ceiling, not the spend

check 6 (downstream time)
  81 attempts * 2.0 s = 162 s
  client deadline 30 s -> 132 s of downstream work after the client stopped waiting
  132 / 162 = 0.8148 -> 81.5% of the produced work was unobserved
```

Check 5 records a correction rather than hiding it: the first pass assumed four completed attempts, and the recomputation shows the budget is crossed during the third. The corrected reading is that both limits bind at different times, and only measurement of the actual attempt log can say which one fired.

**Sensitivity.** Four parameters decide the outcome, and three of them are configuration rather than code.

- Retry depth: dropping the edge layer from 2 retries to 0 takes A from 81 to 27, a 3× reduction, and cuts unobserved downstream time from 132 s to about 38 s.
- Hop count: removing one synchronous boundary takes per-call latency from 1,140 ms to 1,100 ms (3.5%) but reduces the deadline-relevant total by 240 ms per call, which is 6 × 240 = 1,440 ms across an attempt — the same 1,440 ms that is 21% of one attempt.
- Model latency: at p99 = 2,700 ms, per-attempt cost becomes 17,640 ms and only one attempt fits in 30 s, so the configured retry policy becomes dead code that fires after the client has left.
- Token cap: at 120,000 the cap fires mid-attempt-3; doubling it lets all four attempts run and raises spend to 228,000, which is why the cap should be expressed per accepted outcome rather than per attempt.

## 6. Visual / Geometric Intuition

Picture the request path as a corridor whose width is authority and whose length is latency. Ownership is a door on the corridor that only one key opens; duplication is what happens when the corridor forks and the doors are not locked.

```text
amplification as a branching diagram (R = 3 per layer)

client attempt ----+---- edge attempt 1 ---+---- runtime 1 ---+-- adapter 1 -> effect
                   |                       |                  +-- adapter 2 -> suppressed
                   |                       |                  +-- adapter 3 -> suppressed
                   |                       +---- runtime 2 ---+-- adapter 1 -> suppressed
                   |                       |                  +-- adapter 2 -> suppressed
                   |                       |                  +-- adapter 3 -> suppressed
                   |                       +---- runtime 3 ---+-- adapter 1 -> suppressed
                   |                                          +-- adapter 2 -> suppressed
                   |                                          +-- adapter 3 -> suppressed
                   +---- edge attempt 2 --- ... 9 leaves
                   +---- edge attempt 3 --- ... 9 leaves

leaves = 27 at three layers; adding the client layer gives 81
with a stable effect key: accepted effects = 1, suppressed attempts = 80
without it and with p_lost = 0.002 per attempt: expected spurious effects per
  operation = 1 - 0.998^81 = 0.1497, which at 4,000 operations/day is 599/day
note the linear shortcut 81 * 0.002 = 0.162 overstates it, because two attempts
  cannot both be the duplicate of the same accepted effect independently
```

The diagram makes one thing obvious that prose hides: the leaves are not evenly spaced in time. The adapter retries happen within 4 s, the runtime retries within about 20 s, and the edge retries at 30 s or later, by which point the client has already returned an error. So the late attempts are pure cost and pure risk: no user is waiting, and the only thing they can produce is a duplicate.

A second geometric reading concerns boundary count. Each synchronous boundary is a place where latency accumulates additively and where a failure must be classified as before-send or after-send. With H = 6 there are 6 classification points, and one misclassified as "safe to retry" is enough to convert a 1× operation into a 27× one.

A third reading concerns capability width. Draw the principal's scopes as a wide band and each delegated scope as a narrower band inside it. If any link can widen its band, delegation has become a privilege escalator, and the drawing shows it immediately: the child band is not inside the parent band.

## 7. Comparison / Selection Table

### 7.1 The ten-layer stack, with owner and collapse symptom

| Layer | What it decides | Credentialed owner | If collapsed into a neighbour |
|---|---|---|---|
| 1 identity | who the caller is | identity provider | tenant comes from a header and becomes forgeable |
| 2 edge | which headers survive | gateway | untrusted headers reach the control plane |
| 3 admission | whether work may start | control plane | model calls bill before quota is checked |
| 4 policy | what this action may do now | policy service | authorization is cached for the run's lifetime |
| 5 run state | what has happened | state store | worker memory becomes truth and dies with it |
| 6 context | what the model sees | assembler | retrieval text silently becomes instructions |
| 7 model | proposal quality | model provider | output is treated as a decision |
| 8 adapter | schema, policy recheck, dedup | adapter | retries duplicate effects with no key |
| 9 external | the business fact | system of record | local state disagrees with the real world |
| 10 audit | what evidence exists | audit sink | attribution is reconstructed from logs after the fact |

The useful reading is column three: each collapse has a specific, nameable consequence, and none of them is "reduced modularity". If a review cannot name the consequence of a proposed merge, the review is not finished.

### 7.2 Critical boundary distinctions

| Distinction | Left side owns | Right side owns | Confusing them causes |
|---|---|---|---|
| proposal versus decision | model | policy service | model text becomes an approval |
| capability versus permission | credential issuer | policy evaluator | a broad token acts as blanket authority |
| attempt versus operation | runtime | adapter, via effect key | retries create duplicate business facts |
| lease versus lock | scheduler | receiver, via fence token | a paused worker writes after takeover |
| transcript versus context | session store | assembler | summarised-away facts look like they never existed |
| cancellation versus rollback | control plane | a compensating run | cancelled runs reverse nothing |
| budget per node versus per tree | scheduler | tree accounting | nested spawning multiplies the headline cap |
| effect versus receipt | adapter submission | external system | a timeout is recorded as a failure |

### 7.3 Retry placement selection

| Placement | Protects against | Amplifies when | Acceptable condition | Forbidden |
|---|---|---|---|---|
| Client only | transient client network faults | server already retries | one layer retries, others fail fast | client retries after a 4xx |
| Edge only | connection resets before send | body already consumed | idempotent method or effect key | retry after partial upload |
| Runtime step retry | model and tool flakiness | tool also retries | step carries a stable effect key | retry inside a non-transactional sequence |
| Adapter retry | single downstream blip | caller retries too | bounded count and jitter | retry on validation errors |
| Queue redelivery | crashed consumer | handler is not idempotent | handler collapses duplicate ids | redelivery of effects without a key |

### 7.4 Ownership matrix for one run

| Fragment | Single owner | Second writer (must not exist) | Enforcement |
|---|---|---|---|
| run state and version | state store, via transition service | worker process | credential scope plus version predicate |
| attempt counter | runtime worker | adapter | reserved field, adapter writes effects only |
| token budget | scheduler | model caller | shared counter with conditional decrement |
| approval record | approval service | worker | separate signing key |
| cancellation flag | control plane | operator console | one API, audited actor |
| effect table | adapter | reconciliation job | both write via the same idempotent upsert |
| audit cursor | audit sink | none | append-only credential, no updates |
| transcript | session store | assembler | assembler is read-only on this fragment |

Two rows deserve commentary. The effect table is the one place where two writers are legitimate, provided they write through one idempotent upsert keyed by effect key; anything else duplicates effects on reconcile. The audit cursor has no legitimate second writer, and any process needing to "fix" it is a sign that the sink rejected a record rather than that the cursor was wrong.

### 7.5 Identity propagation options

| Option | What the callee learns | Forgery resistance | Audit quality | Cost |
|---|---|---|---|---|
| Shared service account | nothing about the user | none | attribution is inferred | lowest |
| Forwarded signed token | user and tenant | high if verified | direct | low |
| Token exchange per hop | narrowed audience | high | direct, plus chain | medium |
| Impersonation with actor header | user plus actor | depends on a second check | best available | highest |

## 8. Failure Modes & Edge Cases

Each entry gives what an operator sees, the mechanism, one competing cause, the check that separates them, and the numbers that make the risk concrete. All are HYPOTHETICAL teaching cases; none is a reported incident.

### 8.1 Duplicate ticket from three retrying layers
- **Symptom:** two tracker tickets with identical text, created 6 s apart, one abandoned.
- **Mechanism:** adapter retries on timeout after send, and the runtime retries the step, with no effect key at either layer.
- **Competing cause:** the tracker's own webhook retried a delivery, which also yields two objects but with distinct creation paths in its log.
- **Check:** compare the `Idempotency-Key` header absence and the tracker's request log timestamps; duplicate paths with the same key are a receiver bug, without a key it is ours.
- **Numbers:** A = 3² = 9 attempts at two layers; with p_lost = 0.002, expected duplicates per operation = 1 − 0.998⁹ = 0.017857, so 4,000 operations/day produce 71.43 duplicates/day.
- **Containment:** pause the ticket action family, merge duplicates, notify requesters per policy.
- **Prevention:** one adapter-owned key from canonical payload; runtime treats a missing key as a hard failure.
- **Verification:** fault injection that drops the response after commit, asserting exactly one tracker object and two recorded attempts.
- **Provenance:** HYPOTHETICAL; constructed for teaching.

### 8.2 Control plane collapsed into the worker
- **Symptom:** a policy change requires a full redeploy of the execution tier, and during the rollout some runs use the old version.
- **Mechanism:** admission and execution share a process, so policy version travels with the deploy rather than with the request.
- **Competing cause:** a stale policy cache in a separate service, which produces mixed versions without a rollout.
- **Check:** read the policy version recorded on each effect; a version that correlates with pod start time is a collapse, a version that correlates with cache age is staleness.
- **Numbers:** 2,400 runs/day is 100 runs/hour or 1.667 runs/min, so a 10 min rolling replacement puts 1.667 × 10 = 16.7 runs on mixed versions; at 20 replicas replaced one at a time the affected share is 1/20 of that per replica step.
- **Containment:** pin the policy version at admission and re-check at the effect boundary for the mixed window.
- **Prevention:** policy as a versioned input to the request, not a property of the binary.
- **Verification:** deploy with a policy diff and assert every effect record carries one of two explicitly allowed versions, never a third.
- **Provenance:** HYPOTHETICAL; illustrative scenario.

### 8.3 Tenant derived from a forwarded header
- **Symptom:** an internal caller sets a tenant header and reads another organisation's run.
- **Mechanism:** the gateway forwards all headers and the service prefers the header over the token claim for convenience.
- **Competing cause:** a legitimate multi-tenant support tool whose operator really is a member of both tenants.
- **Check:** compare the token claim with the header value and with membership records; claim mismatch is a defect, membership match is legitimate.
- **Numbers:** with 40 tenants, one trusted header exposes up to 39 foreign tenants per compromised client; stripping it at the edge removes all 39.
- **Containment:** strip and log the header, audit reads in the last retention window.
- **Prevention:** ignore the header entirely; resolve tenant from verified claims and load membership server-side.
- **Verification:** send a valid token with a foreign header and require a 403 at the service, not merely at the gateway.
- **Provenance:** HYPOTHETICAL; no incident attributed.

### 8.4 Budget escape through nested spawning
- **Symptom:** one request consumes 31× the advertised per-run token cap.
- **Mechanism:** each spawned sub-agent receives a fresh cap rather than decrementing a tree budget.
- **Competing cause:** a retry storm that also multiplies spend, but with a different shape: repeated identical contexts rather than distinct subtasks.
- **Check:** group spend by run id and inspect distinct prompt hashes; identical hashes mean retries, distinct hashes mean spawning.
- **Numbers:** 1 + 5 + 25 = 31 instances at depth 2 with s = 5; 31 × 120,000 = 3,720,000 tokens against a 120,000 headline.
- **Containment:** freeze spawning, terminate depth-2 instances, reconcile partial effects.
- **Prevention:** a tree-scoped counter decremented at admission of every child, plus an explicit depth cap.
- **Verification:** a load test asserting total tokens for a run never exceed the tree cap, with 5 children per node.
- **Provenance:** HYPOTHETICAL; arithmetic illustrative.

### 8.5 Capability widening by a deputy service
- **Symptom:** a tool executes with a scope the initiating user does not hold.
- **Mechanism:** a token-minting deputy grants from its own scope instead of intersecting with the caller's.
- **Competing cause:** a stale membership cache that still shows a revoked role, which yields the same outcome through a different path.
- **Check:** compare the scope set on the effect's credential with the principal's current scope set; the deputy shows a superset, the cache shows an outdated but smaller set.
- **Numbers:** user holds 2 scopes, deputy holds 6, tool requests 3 → intersection with user grants 2, deputy minting grants 3; the extra scope is 1 of 3 = 33.3% of the request.
- **Containment:** revoke minted tokens, audit effects under the extra scope for the token lifetime.
- **Prevention:** deputies intersect and never widen; encode that as a test over the scope lattice.
- **Verification:** request a scope the user lacks and assert denial at the deputy, not only at the adapter.
- **Provenance:** HYPOTHETICAL; constructed example.

### 8.6 Timeout after send classified as failure
- **Symptom:** the run records `failed`, the external system records success, and support sees a mismatch.
- **Mechanism:** the adapter maps every transport error to failure, losing the distinction between before-send and after-send.
- **Competing cause:** a genuinely failed request whose retry succeeded, which produces a similar mismatch in the opposite direction.
- **Check:** compare the effect key's presence at the receiver with the local attempt record; present at receiver means after-send.
- **Numbers:** 6 boundaries, one of them misclassifying, converts a 1× operation into a 27× attempt path as shown in §6; the mismatch rate then follows AB-08.
- **Containment:** reclassify affected records to unknown, run reconciliation by key.
- **Prevention:** three dispositions only — committed, not committed, unknown — and no mapping from timeout to not-committed.
- **Verification:** a fault proxy that drops responses after forwarding, asserting the record is `unknown` and never `failed`.
- **Provenance:** HYPOTHETICAL; teaching case, unmeasured.

### 8.7 Two writers for the approval record
- **Symptom:** an approval is shown as granted in one screen and pending in another.
- **Mechanism:** the worker and the approval service both write the record, and each caches its own view.
- **Competing cause:** replica lag on a read path, which converges within seconds rather than persisting.
- **Check:** read the record from the primary and compare `updated_at` and writer identity; persistent divergence with two writer identities is the defect.
- **Numbers:** 7 fragments, 2 ambiguous → ambiguity rate 0.286; at 1.5 disagreements/week the review load is 3/week ≈ 1.75 h/week.
- **Containment:** freeze approval writes, reconcile open approvals against the decision log.
- **Prevention:** one signing key, one writer; the worker reads and never writes.
- **Verification:** attempt a worker write and assert a permission error rather than a successful update.
- **Provenance:** HYPOTHETICAL; illustrative only.

### 8.8 Model output used as authorization
- **Symptom:** a tool call proceeds because the model asserted the user is an administrator.
- **Mechanism:** the adapter trusts a `confirmed: true` field produced from model text.
- **Competing cause:** a legitimate delegation recorded earlier and forgotten by the reviewer.
- **Check:** replay the adapter authorization with the model's text removed; denial means the text mattered, success means real delegation exists.
- **Numbers:** one prompt-injection success in a 4,000-request day is 0.025% of traffic and 100% of the affected account's exposure; percentage framing hides this asymmetry.
- **Containment:** revoke the capability used, audit all effects in the window, notify per policy.
- **Prevention:** adapters accept structured intents only, never prose; authority comes from policy evaluation.
- **Verification:** an injection corpus in CI asserting zero authority changes, with the corpus version recorded.
- **Provenance:** HYPOTHETICAL; no measured exploitation.

### 8.9 Hop count growth during an incident
- **Symptom:** p99 latency rises from 6.84 s to 13.08 s with no code deploy.
- **Mechanism:** a debugging proxy added two synchronous hops, each with a 120 ms overhead and one extra retry layer.
- **Competing cause:** model provider latency degradation, which raises `t_model` and leaves hop count unchanged.
- **Check:** decompose measured latency into model time and hop time; hop share rising from 21.05% to 58.7% with flat model time indicates added hops rather than a slower provider.
- **Numbers:** 6 hops × 40 ms = 240 ms per call; 8 hops × 160 ms = 1,280 ms per call; per attempt 6 × (900 + 240) = 6,840 ms becomes 6 × (900 + 1,280) = 13,080 ms, and hop share moves from 1,440/6,840 = 21.05% to 7,680/13,080 = 58.72%.
- **Containment:** remove the proxy from the request path; use sampled out-of-band tracing.
- **Prevention:** a latency budget per boundary, enforced in review; any new synchronous hop must state its milliseconds.
- **Verification:** a synthetic request that asserts total hop overhead below a threshold, failing the build on regression.
- **Provenance:** HYPOTHETICAL; numbers illustrative.

### 8.10 Audit record missing the policy version
- **Symptom:** an auditor asks which policy allowed an action and the record cannot answer.
- **Mechanism:** the adapter writes tenant, principal and run but not the policy version evaluated at dispatch.
- **Competing cause:** the field exists but was populated from a default, which looks present and answers incorrectly.
- **Check:** join each effect to a policy bundle version and count rows with no match; unmatched rows are genuine gaps, matched-but-wrong rows surface as a constant version across releases.
- **Numbers:** 5 of 6 fields present gives attribution 0.833; at 8,000 effects/day that is 1,334 unattributable effects/day, or 40,020 over a 30-day retention window.
- **Containment:** backfill from the control plane's decision log where timestamps allow, and mark the rest unrecoverable.
- **Prevention:** make the six attribution fields non-nullable at the schema level, so omission fails the write.
- **Verification:** a schema test that rejects an effect record with any of the six fields absent or empty.
- **Provenance:** HYPOTHETICAL; constructed for the exercise.

### 8.11 Quota checked after the first model call
- **Symptom:** a tenant over its daily quota still generates spend, and the overage scales with how expensive the first call was.
- **Mechanism:** the quota counter is decremented in the runtime after the model returns, so admission never sees the exhausted state.
- **Competing cause:** a quota counter that is checked at admission but shared across tenants, which produces the opposite symptom: one tenant is blocked while others are under limit.
- **Check:** compare the timestamp of the first billed call with the timestamp of the admission record for over-quota runs; a billed call before any admission record means the check is absent, not merely late.
- **Numbers:** a 500-request/day quota exceeded by 4% is 20 requests/day; at 57,000 tokens each that is 1,140,000 tokens/day of unbilled work, and at a 30-day window 34,200,000 tokens.
- **Containment:** hard-stop the tenant, credit or bill per the contract, and record the overage explicitly rather than silently absorbing it.
- **Prevention:** reserve quota at admission with a conditional decrement, then reconcile the reservation against actual usage.
- **Verification:** drive one synthetic tenant to exactly the quota and assert the (quota+1)-th request is rejected before any model call is dispatched.
- **Provenance:** HYPOTHETICAL; illustrative arithmetic only.

### 8.12 Replay service holds adapter credentials
- **Symptom:** rebuilding a day of history creates external objects.
- **Mechanism:** the replay tool runs the same code path with the same credential as live execution, so replayed intents are indistinguishable from new ones.
- **Competing cause:** a duplicate broker delivery to a live consumer, which also produces external objects but keeps the original event ids.
- **Check:** compare event ids and creation times; distinct ids in one burst indicate replay, repeated ids indicate redelivery.
- **Numbers:** 40,000 recorded events replayed with a working adapter produce 40,000 dispatches; with the adapter credential removed, the same replay produces 40,000 state transitions and 0 effects.
- **Containment:** revoke the replay credential, enumerate and reverse objects created inside the replay window.
- **Prevention:** separate replay into a state-only mode with no effecting credentials, and make that separation a deployment property rather than a code flag.
- **Verification:** run replay against a sandbox receiver and assert zero objects created while the state digest still matches the recorded state.
- **Provenance:** HYPOTHETICAL; teaching scenario.

### 8.13 Cancellation treated as rollback
- **Symptom:** a user cancels, the UI confirms cancellation, and the effect still appears in the external system an hour later.
- **Mechanism:** cancellation transitions the run to a terminal state immediately, while the adapter's in-flight call completes afterwards.
- **Competing cause:** the external system's own delayed posting, which produces the same visible outcome with no in-flight call at all.
- **Check:** compare the external object's creation time with the adapter's submit time; a creation after submit indicates the call was in flight, a creation long after submit indicates the receiver's queue.
- **Numbers:** 4 in-flight effects per cancelled run at a 30 s timeout and a 10 s poll interval means up to 40 s of undetermined work, which is 4 poll cycles per run.
- **Containment:** re-open the affected runs as compensating runs, notify the requester with the actual state.
- **Prevention:** model `cancelling` as a distinct non-terminal state whose exit requires a disposition for every in-flight effect.
- **Verification:** cancel during a deliberately slow call and assert the state sequence contains `cancelling` before any terminal state, with exactly one terminal write.
- **Provenance:** HYPOTHETICAL; constructed example.

## 9. Complexity & Cost Analysis

### 9.1 Asymptotic behaviour of the request path

- Admission: one indexed read plus policy evaluation, effectively O(1) per request in the tenant and quota dimensions, O(p) in the number of applicable policy rules p.
- Versioned transition: O(1) with a primary-key lookup; the cost is dominated by write durability, not by search.
- Context assembly: O(c) in context tokens c, because every token is serialised and transmitted; this dominates the per-call cost by roughly an order of magnitude over all other per-call work.
- Adapter validation: O(f) in payload fields f, trivially small relative to a model call.
- Amplification: O(R^L) attempts, which is the only term in this list that grows exponentially in a design parameter.
- Attribution join: O(1) per record with an index on run id, O(n log n) for a batch reconciliation of n records.

The practical consequence: every complexity term except amplification is benign at realistic scale, and amplification is exponential in the number of retrying layers. Design review should therefore spend its time on retry placement, not on micro-optimising admission.

### 9.2 Back-of-envelope sizing

**ILLUSTRATIVE.** Same parameters as §5, varied one at a time.

```text
latency budget (6 hops at 40 ms, 6 model calls at 900 ms)
  per model call     = 900 + 240 = 1,140 ms; each call crosses every hop, so the
                       hop cost is paid per call and not amortised across calls
  per attempt        = 6 * 1,140 = 6,840 ms
  hop share          = 1,440 / 6,840 = 0.2105 -> 21.05%
  attempts in 30 s   = 4 (27,360 ms) with 2,640 ms headroom

amplification (R = 3)
  L = 2 ->   9 attempts -> 18.0 s of downstream time
  L = 3 ->  27 attempts -> 54.0 s
  L = 4 ->  81 attempts -> 162.0 s
  ratio 81/9 = 9 = 3^2, not 2 -> amplification is not linear in layers

duplicate exposure (p_lost = 0.002, 4,000 operations/day)
  0.998^3  = 0.994011992 -> 1 - that = 0.005988008 ->  23.95/day
  0.998^27 = 0.947381    -> 1 - that = 0.052619    -> 210.48/day
  0.998^81 = 0.850303    -> 1 - that = 0.149697    -> 598.79/day
  ratio 598.79 / 23.95 = 25.0, versus 81/3 = 27 attempts ratio: the duplicate
  rate grows more slowly than attempts because the exponent saturates

budget tree (c = 6 calls, s = 5, cap 120,000 per instance)
  depth 1: instances 6,     calls  36, ceiling   720,000
  depth 2: instances 31,    calls 186, ceiling 3,720,000
  depth 3: instances 156,   calls 936, ceiling 18,720,000
```

The budget block is the one that ends discussions. At depth 3 the ceiling is 18.72 million tokens, which is not a runaway model; it is arithmetic on a configuration that permits five children per node. Nothing in the model's behaviour has to be unusual for it to happen.

The duplicate block carries a subtler lesson. Going from 3 to 81 attempts multiplies attempts by 27 but duplicates by only 25, because (1 − 0.998ⁿ) saturates as n grows. That means an argument of the form "three more retries barely changed the duplicate rate" can be true and still be the wrong conclusion: the cost is in the attempts and the latency, and the duplicates are merely the part that becomes visible.

### 9.3 Cost per accepted outcome

The only defensible cost unit is currency per accepted business outcome, because attempts and calls are intermediate quantities that a retry policy can inflate without bound.

```text
ILLUSTRATIVE unit economics for the §5 shape
  model tokens per attempt        = 6 * 9,500 = 57,000
  attempts per accepted outcome   = 1.6 (client retries plus internal retries)
  tokens per accepted outcome     = 57,000 * 1.6 = 91,200
  downstream calls per outcome    = 1.6 * 3 = 4.8 (adapter retries included)
  suppressed duplicates per outcome = 4.8 - 1 = 3.8
  attribution records per outcome = 1 effect row + 4.8 attempt rows = 5.8
```

Read the third line against the fifth: for every accepted ticket, the system records 5.8 rows and performs 4.8 downstream calls. The rows are cheap; the calls are not, and 79.2% of them (3.8/4.8) exist only to be suppressed. That ratio, not the retry count, is the honest measure of whether deduplication is masking an amplification problem.

Environment for any real measurement is OPEN here: database engine and version, broker, model provider, region and hardware class are all unspecified. Any latency or cost figure in this sheet is therefore a design estimate, not a benchmark.

A benchmark that would settle the retry question: replay 4,000 operations with the tracker's p99 held at 2 s, inject a 30 s hang on 1 in 200 calls and a post-commit response drop on 1 in 500, then report arrivals per intended operation, duplicates accepted, unobserved downstream seconds after the client deadline, and tokens per accepted outcome. Report sample size, dates, versions and price date alongside.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Practical use | What it does not imply | Origin |
|---|---|---|---|---|
| Multiplicative retry amplification | independent retries at L layers multiply attempts by R^L | place retries at exactly one layer | anything about latency distributions | DERIVED §4.1 |
| End-to-end idempotence needs a key | a receiver can collapse repeats only if repeats are distinguishable | design the effect key before the retry policy | that every receiver implements dedup | DERIVED §4.3 variant B |
| Single-writer invariant | one valid writer credential per fragment gives one accepted write per version | enforce ownership by credential | that the writer is correct | DERIVED §4.1 |
| Fencing beats leases alone | a monotonic token lets a receiver reject a stale holder | safe takeover on shared storage | that a third-party API honours tokens | DERIVED §4.3 companion sheet |
| Little's law | L = λW for stable long-run averages | cross-check concurrency against latency | tail behaviour under growing backlog | Little (1961) |
| Intersection of delegated scopes | effective scope is the intersection when each link narrows | audit delegation chains | the same under a minting deputy | DERIVED §4.2 |
| Idempotent method semantics | HTTP defines idempotence in terms of intended effect | reason about transport retries | that any POST is safe to repeat | RFC 9110 §9.2.2 |
| Least privilege at the resource | authorization is evaluated per request and resource | place enforcement below business logic | that a given product implements it | NIST SP 800-207 |

### Source and evidence registry

- **S1 — RFC 9110 §9.2.2, idempotent methods:** <https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2>; supports the semantics of idempotent methods as a property of intended effect, and nothing about a vendor's deduplication.
- **S2 — Little, J. D. C. (1961), Operations Research 9(3):** the queueing identity L = λW for stable averages; used only as a cross-check, never as a latency model.
- **S3 — NIST SP 800-207 (2020), Zero Trust Architecture:** per-request, per-resource authorization as a design stance; not a certification of any implementation.
- **S4 — OWASP LLM Top 10 (prompt injection):** <https://genai.owasp.org/llmrisk/llm01-prompt-injection/>; supports treating model input as untrusted, and does not claim any single mitigation is sufficient.
- **S5 — PostgreSQL row security documentation:** <https://www.postgresql.org/docs/current/ddl-rowsecurity.html>; supports the qualified claim that policies apply subject to role and bypass rules.
- **S6 — Unit topic index:** the `INDEX.md` topic list for this collection; inventory evidence only, and it does not describe any implementation.

Every number in §§4, 5 and 9 was derived here from the stated parameters and is ILLUSTRATIVE. No production latency, duplicate rate, cost or incident outcome is claimed.
Provider behaviour — deduplication window, key retention, lookup-by-key support — is OPEN and requires the provider's own contract.
Organisational rules — data classification, approval authority, retention and residency — are OPEN and require owner confirmation before any of this is implemented.

## 11. Interview / Exam Questions

Scoring: 0 unsafe or wrong; 1 partially right with a critical gap; 2 correct with stated assumptions; 3 correct, defends a trade-off and names a verification. The questions mix recall, calculation and diagnosis; answers differ deliberately.

### Q01. Why is "agent" a bad name for a component boundary?
**Answer:** it names four different things at once — proposal loop, product workflow, runtime and delegated authority — so ownership cannot be assigned while the word is undivided.
**Reasoning:** a boundary is defined by a contract and a credential, and the merged word provides neither; any team can claim to own "the agent".
**Worked check:** nine layers in §7.1 with ten distinct owners; merging three into one drops the count to seven and leaves three collapse symptoms unowned.
**Boundary:** the naming problem is not cosmetic — it is the reason retry ownership is usually unassigned, which is what §4.1 then punishes.

### Q02. Compute the attempt count for retries at the edge, the runtime and the adapter, two retries each.
**Answer:** R = 3, L = 3, so 27 attempts; adding a client that also retries twice gives 81.
**Reasoning:** the factors multiply because each layer re-invokes the one below it; the exponent is the number of independent retrying layers.
**Worked check:** 3³ = 27 and 27 × 3 = 81; the ratio 81/27 = 3 confirms one layer adds exactly one factor.
**Boundary:** the product is a ceiling for configured retries, and it understates reality when a layer retries internally without declaring it.

### Q03. A tracker hangs for 30 s. What does the client observe, and what does the tracker observe?
**Answer:** the client sees a timeout at its own deadline; the tracker may see up to R^L attempts arriving, most of them after the client abandoned the request.
**Reasoning:** each layer's timeout is shorter than the one above it, so retries cascade downward while the caller has already returned.
**Worked check:** at 2.0 s per attempt and 81 attempts, 162 s of downstream time against a 30 s client deadline leaves 132 s unobserved, which is 81.5% of the work performed.
**Boundary:** if the adapter retries a non-idempotent call, those unobserved attempts create real duplicates; if a key exists, they are suppressed but still billed.

### Q04. Where must an effect key be created, and why not at the transport layer?
**Answer:** in the adapter, from canonical business content plus the run and step identity, and persisted before the first send.
**Reasoning:** the transport layer sees one attempt, so any key it generates changes per attempt and cannot collapse them; business content is what makes two attempts the same intent.
**Worked check:** with p_lost = 0.002 and A = 27, expected duplicates per operation are 1 − 0.998²⁷ = 0.052619, which at 4,000 operations/day is 210.48/day; a stable key drives that to zero by construction.
**Boundary:** the key does nothing if the receiver ignores it; verify by sending the same key twice and counting objects.

### Q05. What is the difference between a lease and a lock, numerically?
**Answer:** a lease grants action until an expiry; a lock denies action to others. Neither stops a paused holder, so fencing is what actually prevents the stale write.
**Reasoning:** expiry is a permission change at the scheduler, and a partitioned process cannot be told that its permission changed.
**Worked check:** 30 s lease, 45 s pause → 15 s overlap; at 3 writes/s that is 45 writes at risk, all rejected when the receiver compares a monotonic token.
**Boundary:** a receiver that logs tokens without comparing them leaves 45 writes undetected, converting prevention into slow detection.

### Q06. A sub-agent gets its own token cap. What is the tree ceiling at depth 2 with five children?
**Answer:** 31 instances (1 + 5 + 25), so 31 × 120,000 = 3,720,000 tokens against a headline of 120,000, a 31× overstatement.
**Reasoning:** caps are per node and the tree is not a node, so the ceiling is the sum over instances.
**Worked check:** 1 + 5 = 6 at depth 1 giving 720,000; adding 25 gives 31 and 3,720,000; the increment 25 × 120,000 = 3,000,000 matches the difference.
**Boundary:** depth 3 with the same fan-out gives 156 instances and 18,720,000 tokens, which is why a depth cap is a budget control.

### Q07. Delegation chain: user holds 2 scopes, run holds 3, tool needs 1. What happens?
**Answer:** the effective scope is the intersection, so the tool's need is granted only if it lies inside the user's 2; otherwise the call is denied.
**Reasoning:** each link must narrow; intersection is the operation that makes the narrowest link binding and keeps the audit chain meaningful.
**Worked check:** user {orders.read, tickets.read}, run adds tickets.write, tool needs tickets.write → intersection lacks it → deny; adding one minting link that widens permits 3 of 3, a 33.3% expansion of the requested set.
**Boundary:** intersection is only sound when no link can mint; test by requesting a scope the user never held.

### Q08. Which of six attribution fields would you add first if you could add only one?
**Answer:** the policy version, because it is the only field that connects an effect to the decision that allowed it.
**Reasoning:** tenant and principal are usually present because they are needed for filtering; policy version is the field nobody needs until an audit, so it is the one that goes missing.
**Worked check:** 5 of 6 fields gives attribution 0.833 and 1,334 unattributable effects/day at 8,000/day; over 30 days that is 40,020 records with no join to a decision.
**Boundary:** a present-but-defaulted policy version is worse than an absent one, because it survives a schema check and misleads the auditor.

### Q09. Six hops at 40 ms and six model calls at 900 ms: what fraction of attempt latency is boundary overhead?
**Answer:** 1,440 ms of 6,840 ms, which is 21.05%.
**Reasoning:** the per-call cost is model plus hop overhead, and hops are additive per call while the model time is per call as well.
**Worked check:** per call 900 + 240 = 1,140 ms; 6 calls give 6,840 ms; hop time is 6 × 240 = 1,440 ms; 1,440/6,840 = 0.21053.
**Boundary:** with 8 hops at 160 ms the hop time becomes 6 × 1,280 = 7,680 ms of a 13,080 ms attempt, or 58.72%, so hop overhead can dominate after an incident-time proxy is added.

### Q10. Diagnose: p99 latency tripled, no deploy, model provider metrics flat.
**Answer:** a synchronous hop was added somewhere in the path, most likely by an observability or debug proxy, since model time is unchanged.
**Reasoning:** hop overhead is additive and multiplicative across calls, so a per-hop increase shows up as a step change unrelated to deploy events.
**Worked check:** at flat model time of 900 ms, 17,640 ms per attempt over 6 calls is 2,940 ms per call, so the implied hop overhead is 2,940 − 900 = 2,040 ms against 240 ms before, an 8.5× growth that network variance cannot produce at six hops.
**Boundary:** a retry layer added at the same time confounds the reading; count downstream arrivals to separate them.

### Q11. Why is "last write wins" acceptable for a counter and unacceptable for a cancellation flag?
**Answer:** a counter's value is a quantity where a lost increment costs accuracy, while a flag encodes an authority decision where a lost write reverses intent.
**Reasoning:** the two fragments have different invariants: monotone counting versus monotone intent, and only the second requires that a later writer cannot revive an earlier state.
**Worked check:** 7 fragments, 2 with two writers → 0.286 ambiguity; at 1.5 disagreements/week each the review load is about 1.75 h/week, or 91 h/year, against hours to revoke a credential.
**Boundary:** even a counter needs a conditional decrement when a budget is enforced, otherwise concurrent decrements can overspend.

### Q12. What does a boundary need beyond an input and an output schema?
**Answer:** a credential, a declared failure mode and a recovery rule — the three elements that determine what happens when the hop is crossed under stress.
**Reasoning:** schemas describe the happy path; recovery rules are what an incident uses, and their absence is why two teams each assume the other retries.
**Worked check:** 6 boundaries × 5 elements = 30 declarations; at 24 present, completeness is 0.80 with 6 missing classes, and their arrangement matters more than the ratio.
**Boundary:** a recovery rule that says "retry" without stating whether the failure was before or after send is not a recovery rule.

### Q13. How would you prove that exactly one layer retries?
**Answer:** force a downstream hang longer than every configured timeout, then count arrivals at a controlled receiver: the count equals the retrying layers' product, and 1 means one layer.
**Reasoning:** configuration files lie about inheritance; a controlled receiver counts what actually happened.
**Worked check:** expected 27 arrivals with three retrying layers at two retries; observing 9 means one layer's retries were disabled by a timeout shorter than the hang.
**Boundary:** an async queue in the path decouples the count from the client deadline, so the receiver must record arrival timestamps too.

### Q14. What would you measure before claiming the request path is production-ready?
**Answer:** arrivals per intended operation, duplicates accepted, unobserved downstream seconds, tokens per accepted outcome, and attribution completeness across all effect records.
**Reasoning:** each maps to one invariant in §4, and an invariant without a metric is an intention that nobody can falsify.
**Worked check:** at 4,000 operations/day and 1.6 attempts per outcome, 6,400 attempt records/day give enough volume for a weekly duplicate rate but not for a daily p99.
**Boundary:** any figure quoted without workload, versions and dates is a design estimate, which is all this sheet contains.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Core rule | name one credentialed owner per decision, record and effect before discussing autonomy |
| Amplification | A = R^L; 3 retries at 3 layers is 27 attempts, at 4 layers 81 |
| Effect safety | stable key from canonical content + run + step, persisted before first send |
| Budget | cap the tree, not the node; depth 2 with five children is 31× the headline |
| Delegation | effective scope is the intersection; any minting link breaks the argument |
| Latency | hops are additive per call; 6 hops at 40 ms was 21% of a 1,140 ms call |
| Attribution | six required fields; missing one removes a join, not a column |
| Ownership test | try the write from the wrong component; a success means two owners |
| Failure classification | committed, not committed, unknown — never timeout to failed |
| Evidence | all figures ILLUSTRATIVE; provider and organisational rules OPEN |
