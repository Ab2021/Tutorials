---
case_id: "CASE-03-001"
case_title: "Design a Rate Limiter"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing / admission control"
primary_pattern: "resource-admission"
secondary_patterns: ["correctness-contract", "capacity-safety"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "platform reliability lead"
source_index: "03_dsa_and_computational_thinking/Case_Studies/INDEX.md"
review_trigger: "Change to arrival profile, backend service capacity, tenant mix, clock discipline, or the accepted overshoot bound"
---

# Design a Rate Limiter

**Evidence declaration.** This is an ILLUSTRATIVE teaching case. The gateway, tenant roster, arrival profile, backend capacity, prices and results are constructed to make one decision auditable. Nothing here is a company deployment, a measured incident, or a vendor benchmark. **Implementation stage:** proposed. **Editorial status:** draft; primary-source verification, capacity measurement and publication review remain OPEN.

All arithmetic in this manuscript is *derived* from stated inputs and was recomputed independently as part of the authoring check. Derived arithmetic is still not evidence about a real system: the arrival model is a fluid approximation, and no queue, clock or network behaviour was observed.

## 1. The One-Line Framing

Given **one inbound API request with its tenant identifier and arrival timestamp**, decide **whether the gateway admits or rejects it**, for the **platform reliability lead**, so that **the backend stays inside its measured service capacity while paying tenants keep the share of throughput they were promised**, subject to **a hard global budget that must not be exceeded even during a coordinator failure**, compared with **unbounded admission plus reactive overload handling at the origin**.

That is an admission-control decision, taken thousands of times per second, with a per-request deadline measured in single-digit milliseconds. It is not a modelling decision about traffic; the traffic model only parameterises a data structure that must make a decision in constant time.

The unit of analysis is the *request*, not the tenant, not the second, and not the incident. A tenant-level average hides the burst that causes the outage; a per-second aggregate hides the 200-millisecond amplification that a fixed-window counter permits at a boundary. The decision must be expressed per request because that is the only granularity at which harm can be prevented.

Three reusable patterns carry the case:

| Pattern | What it fixes | What it cannot fix |
|---|---|---|
| Resource admission | Whether work enters the system at all | The cost of work already admitted |
| Correctness contract | What each decision records and who owns it | Whether the recorded inputs were truthful |
| Capacity safety | The margin between admitted rate and service rate | Structural inefficiency inside the service |

**Where the obvious framing fails.** The intuitive design is "count requests per second, reject the overflow". That design has four defects that this case exists to expose. First, a fixed-window counter allows roughly twice the nominal limit across a window boundary, so the *worst* second is far worse than the *average* second. Second, counting globally lets one tenant consume the whole budget, which converts a fairness problem into a churn problem. Third, a distributed counter that is consulted on every request either adds a network round trip to every decision or loses atomicity. Fourth, and most importantly, a limiter set exactly equal to backend capacity does not protect the backend at all: a queue fed at its service rate is critically loaded, and its wait time diverges as utilisation approaches one.

**Decisive tradeoff.** The case resolves to a choice between a *token bucket with coordinated leases* and *per-node partitioned buckets*, and the resolution is not "which is faster". The two designs fail in opposite directions: leases conserve the global budget and therefore under-admit (wasting purchased capacity), while partitioned buckets never contact a coordinator and therefore over-admit by an amount bounded by clock skew and partition imbalance. The selection criterion is the accepted overshoot bound, which is a business statement about how much a backend may be overrun, not a performance preference.

**Interview cue.** A reader should be able to state in thirty seconds: "admit requests into a token bucket with capacity C and refill r strictly below backend service rate μ, enforce the global budget through deducted leases so that overshoot is impossible rather than improbable, and report rejections as a first-class outcome rather than hiding them behind admitted-request latency."

**Acceptance for this opening.** Input, output, beneficiary, binding constraint, baseline and evidence basis are all identifiable above, and the three pattern names come from the collection index rather than being invented here.

## 2. Context & Background

The system under discussion is an HTTP gateway that fronts a write-heavy backend service. Requests arrive from customer applications, each carrying a tenant identifier. The backend performs a fixed amount of work per request: it validates a payload, writes one row, and emits one event. Its sustainable service rate was *not* measured for this manuscript; the value μ = 100 requests per second is a declared planning parameter, and every conclusion that depends on it is conditional on that declaration.

| Context field | Illustrative content |
|---|---|
| Product | Metered ingestion API for a business-analytics product |
| Workflow before decision | Client library signs a request, resolves the gateway, opens or reuses a connection, sends headers and body |
| Workflow after decision | Admitted: gateway proxies to backend, backend writes and acknowledges. Rejected: gateway returns 429 with a retry hint and does no backend work |
| Decision owner | Platform reliability lead, who owns the admission policy and its budget |
| Other stakeholders | Backend service owner (capacity), finance partner (purchased capacity and egress), support lead (rejection complaints), security (abuse), tenant success (fairness commitments) |
| Baseline | Unbounded admission; overload is handled reactively by backend timeouts and client-side retries |
| Scale (illustrative) | 120 requests/s peak arrival, 5,000 tenants, 8 gateway nodes, one region |
| Timing | A 6-minute peak window (360 s) is the modelling unit; the business cycle is a month-end reporting spike |
| Boundaries | Single region, single currency, no cross-region replication in scope |
| Prior attempt | An in-process counter per gateway node was tried informally and produced uneven rejection rates across nodes |
| Applicable requirements | None cited here; contractual rate commitments would require primary documents and are outside this manuscript |

**Why the baseline persists.** Unbounded admission is genuinely simpler and, most of the time, correct: at 60 requests/s against 100 requests/s of capacity the system is comfortable, and any limiter is pure overhead. The baseline fails only in the tail, but the tail is where the business consequence lives. A backend at critical load does not degrade gracefully: its queue grows, its timeouts fire, clients retry, and the retries become additional arrivals that were never counted by anyone. The failure is self-reinforcing, which is why "we will add capacity when it hurts" is not a sufficient answer.

**The retry amplification mechanism.** Suppose the backend's own request timeout is 2 s. When the queue wait exceeds 2 s, the client gives up and retries, but the original request is still queued and will still be executed. Arrivals have now increased while service capacity has not. In the worst case each original request generates two or three executions, so an effective arrival rate of 120 can produce 240 to 360 units of backend work. A limiter applied only to *first attempts* therefore does not bound backend work; the admission contract must be defined on executions, or the timeout must be set below the point where retries become cheaper than waiting.

**Who experiences the failure.** Three different populations, and they must not be conflated:

| Population | What they experience | What they need from the design |
|---|---|---|
| Compliant tenants below their quota | Nothing, ideally; latency rises if the backend is overloaded | Guaranteed non-rejection |
| Bursting tenants inside their quota | Short rejections if the bucket is too small | Burst capacity, not just average rate |
| Abusive or runaway clients | Rejections | A clear signal, a retry hint, and no back-door bypass |

A design that is excellent for the second population (large C) is dangerous for the backend (large instantaneous burst); a design that is excellent for the backend (small C) causes spurious rejections for legitimate batch jobs that are momentarily above their average. This tension, not the choice of counter arithmetic, is the actual content of the decision.

**Prior attempt, examined.** The informal per-node counter used a limit of 100 per second per node. With eight nodes, the aggregate ceiling was 800 per second against a 100 per-second target, so it never rejected anything and created a false sense of protection. The lesson generalises: a distributed limit expressed per node must be derived by dividing the global budget, and the division must account for the fact that nodes do not receive equal traffic.

**Takeaway.** The context establishes three things that a tool choice cannot: the decision granularity is the request, the protected resource is backend execution capacity, and the failure of interest is a positive feedback loop between queueing and retries.

## 3. The Business Question

| Field | Decision contract |
|---|---|
| Question | For an arrival profile peaking at 120 requests/s against a declared 100 requests/s of backend capacity, which admission mechanism and which global budget should the gateway enforce, and what overshoot bound does the organisation accept? |
| Decision owner | Platform reliability lead; backend owner supplies the capacity parameter; finance supplies cost inputs; the lead signs the policy |
| Trigger and deadline | Peak-season readiness review; the policy must be decided before the traffic increase, not during it |
| Available actions | Token bucket with coordinated leases; token bucket with partitioned per-node buckets; fixed-window counter; exact sliding log; concurrency limiter at the backend; retain unbounded admission |
| Value | Preserved backend availability and predictable latency for compliant tenants; purchased capacity is not wasted |
| Downside | Legitimate rejections, retry storms, unfairness between tenants, added latency per decision, operational complexity of a coordinator |
| Primary objective | Keep admitted execution rate strictly below declared service capacity while rejecting as few compliant requests as possible |
| Hard constraints | Global budget must never be exceeded by design (not merely on average); rejection must be cheap; decision must be bounded-work and constant-time; a coordinator outage must not remove the limit |
| Population | Illustrative arrivals in one 6-minute peak window; 5,000 tenants; one region |
| Out of scope | Autoscaling policy, payload validation, billing reconciliation, security abuse detection, multi-region admission |
| Proxy that must not substitute | "Requests per second" as a single number. The objective is a *distribution* — admitted rate below capacity, bounded burst, bounded overshoot — and a mean rate cannot express it |

The question is deliberately narrower than "design a scalable API gateway". It is: *which admission rule, at which budget, with which failure mode*. A reader who answers only "use Redis with INCR and EXPIRE" has answered a mechanism question and not the decision question, because that mechanism has a boundary-doubling defect and a per-request network cost that this case will quantify.

**Why the baseline is a real competitor.** Unbounded admission with reactive handling is not a straw man. It has zero added latency, zero coordination, zero configuration surface, and it works for all traffic below capacity. Any limiter must justify its own cost — a network hop or a state store, plus the operational burden of a budget that must be calibrated. The case therefore computes that cost explicitly rather than asserting that limiting is obviously correct.

**The proxy/objective conflict.** If the success measure is "median latency of admitted requests", a limiter looks excellent: it removes the requests that would have queued longest, so the surviving median improves. But the organisation has not gained anything if the removed requests were legitimate. This is the coordinated-omission problem in miniature, and Section 9 makes the counter-metric mandatory: rejection rate and rejected-request latency must be reported alongside admitted latency, or the measurement rewards shedding work.

## 4. Assumptions & Constraints

Each assumption is stated with a verification test, the consequence of violation, and the safe fallback. The shared invariants that apply to every row are stated once here and are not repeated later: (i) every request carries a stable tenant identifier and an arrival timestamp from the gateway's clock; (ii) no request may execute at the backend without an admission record; (iii) every rejection must carry a machine-readable reason code; (iv) the policy version is recorded with each decision; (v) the owner, not the software, authorises any change to the global budget.

| ID | Assumption or constraint | Type | Verification | Status | If violated | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Arrival profile peaks near 120 req/s over 360 s | workload | compare with gateway request log histogram | unverified | budget and burst sizing are wrong | recompute from observed histogram | reliability lead |
| A-02 | Backend sustains ~100 req/s per instance | capacity | load test at sustained rate with fixed payload | unverified | limiter admits past capacity | lower r and re-derive | backend owner |
| A-03 | Service time is roughly constant (write + one event) | queue model | measure service-time coefficient of variation | unverified | M/D/1 wait formula understates wait | use measured distribution | backend owner |
| A-04 | Retries are bounded by client contract to at most 1 per request | workload | inspect client-side retry configuration | unverified | effective arrivals exceed the modelled value | count executions, not attempts | client success |
| A-05 | Gateway nodes share a disciplined clock within 250 ms | operations | monitor NTP offset per node | unverified | skew inflates local refill | use monotonic local time plus coordinator counters | platform |
| A-06 | Coordinator (state store) p99 round trip is under 1 ms in-region | platform | measure state-store latency percentiles | unverified | per-request check cost rises | switch to leases | platform |
| A-07 | Rejection cost is dominated by the 429 response, not by backend work | design | confirm gateway short-circuits before proxying | proposed | rejects still consume backend capacity | enforce short-circuit ordering | gateway owner |
| A-08 | Tenant quota commitments exist and are enforceable | commercial | inspect current rate-card documents | OPEN | fairness rule has no authority | apply global-only limit and escalate | commercial |
| C-01 | No request executes without an admission record | authority | audit backend writes against admission ledger | required | unaccounted work | block the bypass path | reliability lead |
| C-02 | Global budget may never be exceeded by design | safety | derive worst case analytically per mechanism | required | backend can be overrun during a race | reject the mechanism | reliability lead |
| C-03 | Rejections are a reported outcome, not a hidden error | measurement | confirm dashboards carry reject counts | required | measurement rewards shedding | add rejection counters | observability |
| C-04 | A coordinator outage must not remove the limit | resilience | inject coordinator failure in a replay | required | unbounded admission during outage | local emergency bucket | platform |
| C-05 | Burst capacity C is chosen with the backend queue in mind | capacity | compare C with queue headroom | proposed | a full bucket spikes the queue | reduce C or smooth | capacity owner |

**Which violation reverses the decision.** Two assumptions are decision-critical, not merely numeric. A-02 (capacity) and C-02 (no over-admission by design) each individually reverse the recommendation. If the backend capacity is materially higher than declared, the correct action is to raise the budget and reduce rejections rather than to keep a conservative limiter; if no mechanism can guarantee the overshoot bound, the correct action is to keep unbounded admission and invest in backend headroom, because a limiter that over-admits under contention provides false assurance.

**The constraint people forget.** C-04 is not a nice-to-have. The most common real-world outcome of a centralised limiter is that the limiter itself becomes the outage: the coordinator is unavailable, the gateway cannot decide, and the safe-looking behaviour is either to reject everything (a self-inflicted denial of service) or to admit everything (a self-inflicted overload). Both are policy failures. A design must state explicitly which one it chooses and bound the consequence. This case chooses a *bounded emergency bucket held locally*, sized so that even total coordinator loss cannot exceed a declared multiple of the budget for a declared duration.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | One request attempt, keyed by `request_id`; the tenant dimension is `tenant_id`; the time dimension is the gateway receive time in UTC |
| Population | Illustrative inbound requests in the 360 s peak window; retries after a rejection are separate attempts and must not be merged into the original |
| Decision-time inputs | tenant identifier, arrival timestamp, policy version, current bucket state (tokens and last-refill time, or TAT), lease state if used |
| Output | admit or reject, with reason code (`OK`, `LIMIT_GLOBAL`, `LIMIT_TENANT`, `COORD_UNAVAILABLE_ADMITTED`, `COORD_UNAVAILABLE_REJECTED`) |
| Backend outcome | execution status, service duration, and whether the execution had a matching admission record |
| Join semantics | One admission record joins to at most one backend execution by `request_id`; a duplicate execution with the same `request_id` is an idempotency breach and is counted, not deduplicated away |
| Time basis | Gateway monotonic clock for refill arithmetic; UTC wall clock for reporting; the two are recorded separately because they can disagree |
| Labels | None. There is no supervised target here; the evaluation compares an admission policy's observable consequences, not predictions against truth |
| Exclusions | Requests rejected at the edge before reaching this gateway (TLS failure, authentication failure) are excluded and counted separately |

The absence of labels is worth stating plainly: this is not a machine-learning case, and no amount of data will produce a "correct" admission rate. What the data can do is bound consequences — how many requests were rejected, how much capacity was consumed, and how far the observed admitted rate deviated from the budget.

### 5b. Tools & Stack

| Component | Role | Why it is needed | What it must not do |
|---|---|---|---|
| Gateway (8 nodes) | Terminates the request, evaluates admission, proxies or short-circuits | Decision point closest to arrival | Must not buffer the body before deciding |
| In-process bucket state | Holds tokens/TAT for the fast path | Constant-time decision, no network hop | Must not be the sole authority for the global budget |
| Coordinator state store | Holds the global counter and issues leases | Atomic global deduction | Must not be on the path of every request if latency budget is tight |
| Policy registry | Versioned C, r, per-tenant shares | Reproducible decisions and audit | Must not be edited in place without a version bump |
| Admission ledger | Append-only decision records | Post-hoc reconciliation and fairness audit | Must not be sampled below the reconciliation requirement |
| Metrics pipeline | Counters for admit/reject by reason, tenant, node | Detects imbalance and coordinated omission | Must not report only admitted-request latency |

### 5c. Method Summary

The method is a fixed pipeline with a single decision point and an explicit failure branch. It is written as a state machine because the interesting questions are about transitions, not about the happy path.

```text
INPUT: (tenant_id, request_id, t_arrive) 
STEP 1  short-circuit check: is the request within the tenant's declared burst?
STEP 2  local token check: does the local bucket (or TAT) permit one unit at t_arrive?
STEP 3  global budget check: if the local grant requires a lease, is a lease available?
STEP 4  decide: admit | reject(reason)
STEP 5  record: append admission record with policy version and state snapshot
STEP 6  act: admit -> proxy; reject -> 429 with retry hint and no backend call
```

The invariant that makes this pipeline auditable is stated once: **for every admitted request there is exactly one admission record whose recorded policy version was in force at `t_arrive`, and that record's admitted count never exceeds the budget of that policy version at any point in time.** The invariant is checkable after the fact by replaying the ledger, which is why the ledger is append-only and why the policy version is per-record rather than per-day.

| Candidate mechanism | Decision cost | Overshoot bound | Burst shape | State per tenant | Verdict in this case |
|---|---|---|---|---|---|
| Fixed-window counter | 1 state read/write | up to 2× limit across a boundary | step | 1 counter | rejected: worst second is 2× nominal |
| Sliding log | O(log n) append/trim | exact | exact | O(r·W) entries | rejected: memory grows with rate |
| Sliding-window counter | 2 state reads | approximate, error bounded by previous window | smoothed | 2 counters | considered: acceptable only if overshoot tolerance is loose |
| Token bucket | O(1) arithmetic | C + r·W in any window W | controlled by C | 2 scalars | selected with leases |
| GCRA (leaky-bucket equivalent) | O(1) arithmetic | same as token bucket | same, one state variable | 1 timestamp | selected as implementation of the same policy |
| Concurrency limiter | O(1) counter | none on rate; bounds in-flight only | adaptive | 1 counter | complementary, not a substitute |
| Unbounded admission | 0 | unbounded | backend-defined | 0 | baseline only |

**Why two of these are mathematically the same policy.** Token bucket and GCRA differ in representation, not in admitted set. A token bucket with capacity C and refill r admits a request at time t if the number of tokens available is at least one; GCRA stores a single theoretical arrival time and admits if `t ≥ TAT − τ` with `τ = (C−1)·T` and `T = 1/r`. Because they describe the same constraint, choosing between them is an engineering question about state size and integer arithmetic, not a policy question. Stating this prevents the common error of treating them as two different fairness guarantees.

### 5d. Data provenance, quality, exclusions and leakage controls

- The arrival histogram must be taken from gateway logs, not from backend logs, because backend logs are already filtered by admission and therefore cannot show the requests the policy removed.
- The backend capacity parameter must come from a load test at the *declared payload distribution*, not from a synthetic benchmark with a smaller body. A capacity number obtained at a different payload is a different quantity with the same units.
- Rejected requests must be retained in the ledger with their timestamp even though no backend work occurred; dropping them makes the rejection rate uncomputable and makes a policy that rejects everything look idle.
- Retry attempts must be recorded as separate attempts with a link to the original `request_id`; merging them hides amplification, which is the single largest source of model error in this case.
- Clock offset per node must be logged alongside decisions. Without it, a skew-induced overshoot is indistinguishable from a policy bug.
- No decision-time input may be derived from the backend outcome. In particular, "was this request eventually served" is a post-hoc label and must never be read by the admission path, because that would be leakage in the strict sense — the decision would use information created after the decision.
- Excluded traffic (pre-authentication rejections) must be reported as a separate denominator so that the admitted fraction is never computed against an inflated base.

**Leakage, stated concretely.** The temptation is to build a "smart" limiter that learns which tenants are likely to be abusive. Any feature derived from a tenant's *future* behaviour — for example "this tenant was throttled three times later in the window" — is unavailable at decision time. A model trained with such a feature will look excellent offline and will fail in production. This case avoids the problem by not making the admission decision statistical at all; the note is retained because the same failure appears whenever a case study drifts from mechanism to prediction.

### 5e. Reproducibility and method-selection ledger

| ID | Artifact | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| ILL-01 | Arrival profile (120 req/s for 360 s) | constructed | parameterise the budget | illustrative | not a measured histogram |
| ILL-02 | Capacity parameter μ = 100 req/s | constructed | set the admitted budget | illustrative | not a load-test result |
| CALC-01 | Fluid backlog and drain derivation | derived | show the cost of no limiter | reproducible | fluid model, not a queue simulation |
| CALC-02 | Token-bucket envelope C + r·W | derived | bound burst in any window | reproducible | assumes full bucket at window start |
| CALC-03 | Lease overshoot bound | derived | compare mechanisms | reproducible | assumes worst-case lease interleaving |
| CALC-04 | M/D/1 wait at utilisation ρ | derived | show why r must be below μ | reproducible | assumes Poisson arrivals and deterministic service |
| CFG-01 | Policy configuration (C, r, shares, leases) | proposed | repeatable decisions | proposed | requires change control |
| ORC-01 | Independent envelope oracle | proposed | check bucket arithmetic | proposed | checks only the stated formula |

**What the oracle can and cannot do.** ORC-01 is a small reference implementation of the token-bucket envelope: given a sequence of arrival timestamps and a (C, r) pair, it returns the admitted subsequence by direct simulation. It validates the arithmetic of the chosen mechanism on short traces. It cannot validate that the trace resembles production traffic, and it cannot validate the choice of C and r — those are judgement calls about cost and fairness that no oracle resolves.

**Selection rule used in this case.** A mechanism is admissible only if it satisfies C-02 (no over-admission by design), then ranked by (a) rejection rate of compliant traffic, (b) decision cost, (c) operational surface. C-02 is a filter, not a weight; mechanisms that merely over-admit *rarely* are excluded, because "rarely" is a description of an untested race, not a bound.

## 6. Analysis Walkthrough

Everything in this section is ILLUSTRATIVE input with DERIVED arithmetic. Each result states its model, and each model's boundary is given before the number is used. The section is ordered so that the reader sees the cost of doing nothing, then the cost of the obvious fix, then the cost of the correct fix — which is the order in which the decision actually has to be defended.

### 6.1 The cost of doing nothing

The baseline admits everything. Arrivals exceed the declared service rate during the peak, so a backlog accumulates. The fluid model is the smallest model that answers "how much work is stranded", and it is chosen deliberately because it needs no distributional assumption.

```text
FORMULA [C-01]: B = max(0, λ − μ) · t
WHERE: B backlog [requests]; λ arrival rate [requests/s]; μ service rate [requests/s];
       t overload duration [s].
CONSTRAINT: initially empty queue; constant rates; no losses, no retries, infinite buffer.
NUMERICAL EXAMPLE: B = (120 − 100) · 360 = 7,200 requests.
INTERPRETATION: the interval strands 7,200 units of work in the queue; buffering
                absorbs the interval but does not make sustained overload safe.
```

Two objections must be answered before this number is used. First, real arrivals are not constant; a fluid model understates the peak because it averages across the window. Second, real buffers are not infinite; when the queue is full, work is dropped, and the drop is invisible to the fluid model. Both objections push in the direction of the real system being *worse* than 7,200, which is sufficient for the decision: the number is a lower bound on stranded work under a declared model, not an estimate of user-visible latency.

Drain time under recovery gives the second half of the baseline's cost:

```text
FORMULA [C-02]: T = B / (μ_r − λ_r)
WHERE: T drain time [s]; μ_r recovery service rate [requests/s];
       λ_r arrival rate during drain [requests/s]; B [requests].
CONSTRAINT: μ_r > λ_r; constant rates; no new saturation, no retries.
NUMERICAL EXAMPLE: μ_r = 140, λ_r = 80 → T = 7,200 / 60 = 120 s.
INDEPENDENT CHECK: 60 requests/s surplus × 120 s = 7,200 requests. ✓
BOUNDARY: if μ_r = λ_r the surplus is zero and T is undefined: a positive backlog
          never drains under this model, regardless of buffer size.
```

The boundary is the important part. The natural operational instinct — "add buffer" — cannot fix a zero-surplus situation, because buffer capacity changes how long the queue takes to fill, not whether it fills. This is the reason the decision cannot be "buy more queues".

### 6.2 The cost of the obvious fix: fixed window

A fixed-window counter allows L requests per aligned window W. Its mean rate is exactly L/W, which is why it appears in so many implementations. Its worst-case rate is not L/W.

```text
FORMULA [C-03]: A_max = 2 · L  (over any interval of length 2ε straddling a boundary)
WHERE: A_max maximum admissions [requests]; L per-window limit [requests];
       ε → 0 is the offset from the window boundary [s].
CONSTRAINT: the client is allowed to exhaust one window and immediately begin the next.
NUMERICAL EXAMPLE: L = 100 per second → A_max = 200 requests in a vanishingly small interval.
INTERPRETATION: the worst second admits twice the nominal limit; the backend sees the
                average only if arrivals are smooth, which they are not.
```

This is the first genuine defect of the intuitive design, and it is not a rounding error: it is a factor of two on exactly the quantity the limiter exists to bound. A token bucket with the same mean, by contrast, bounds admissions in any window of length W by `C + r·W`.

```text
FORMULA [C-04]: A_max(W) = C + r · W
WHERE: A_max maximum admissions in a window of length W [requests];
       C bucket capacity [requests]; r refill rate [requests/s]; W window [s].
CONSTRAINT: bucket full at the start of the window; perfect clock; no lease effects.
NUMERICAL EXAMPLE (C = 240, r = 100 req/s):
  W = 1 s   → 240 + 100 = 340
  W = 10 s  → 240 + 1,000 = 1,240
  W = 60 s  → 240 + 6,000 = 6,240
  W = 360 s → 240 + 36,000 = 36,240
INTERPRETATION: C is the entire burst allowance and it is paid once, not per second.
```

Note the shape: the envelope is linear in W with slope r and intercept C. A fixed window has zero slope within the window and a jump at the boundary, which is the discontinuity the counter cannot express. Choosing C is therefore choosing how much instantaneous burst the backend must absorb — a capacity decision, not a policy preference.

### 6.3 GCRA representation and the state-size argument

The same policy has a one-variable representation. Let `T = 1/r` be the emission interval and `τ = (C − 1)·T` the burst tolerance. The decision is: admit if `t ≥ TAT − τ`, then set `TAT = max(TAT, t) + T`.

```text
FORMULA [C-05]: T = 1/r ; τ = (C − 1) · T
WHERE: T emission interval [s/request]; r refill [request/s]; τ tolerance [s];
       C capacity [requests].
CONSTRAINT: C ≥ 1 integer; r > 0; single variable TAT held per tenant.
NUMERICAL EXAMPLE: r = 100 → T = 0.01 s = 10 ms; C = 240 → τ = 239 · 0.01 = 2.39 s.
INTERPRETATION: the bucket's 240-request burst is exactly a 2.39 s tolerance on TAT.
```

The practical value is the state size. A per-tenant token bucket stores two scalars (tokens, timestamp); GCRA stores one (TAT). For 5,000 tenants, that is tens of kilobytes either way — both are negligible, and the argument for GCRA is therefore *arithmetic cleanliness* (one comparison, integer nanoseconds, no division per request) rather than memory. Division is the expensive operation in a token bucket: `tokens = min(C, tokens + r·Δt)` needs a multiply and a min, and a naive float implementation accumulates drift. GCRA needs one addition and one comparison.

The honest caveat: GCRA's single variable makes per-tenant *burst smoothing* harder to compose with a global lease, because the lease must translate into an additional tolerance rather than an additional token count. In this case the lease is applied to the token accounting, so the token-bucket representation is retained for the global layer and GCRA is noted as an equivalent local optimiser.

### 6.4 The utilisation question: why r must be below μ

The most consequential number in the case is not the burst; it is the admitted rate. Setting r = μ looks correct — the limiter admits exactly what the backend can serve — and it is wrong, because a queue fed at its service rate is critically loaded.

```text
FORMULA [C-06]: W_q = ρ / (2 · μ · (1 − ρ)) ,  ρ = λ_a / μ
WHERE: W_q mean queueing wait [s]; λ_a admitted rate [requests/s];
       μ service rate [requests/s]; ρ utilisation [dimensionless, 0 ≤ ρ < 1].
CONSTRAINT: M/D/1 queue — Poisson arrivals, deterministic service, single server,
            infinite buffer, first-come-first-served. Not valid at ρ = 1.
NUMERICAL EXAMPLE (μ = 100): 
  λ_a = 60 → ρ = 0.60 → W_q = 0.60/(2·100·0.40) = 0.0075 s = 7.5 ms
  λ_a = 80 → ρ = 0.80 → W_q = 0.80/(2·100·0.20) = 0.0200 s = 20.0 ms
  λ_a = 85 → ρ = 0.85 → W_q = 0.85/(2·100·0.15) = 0.0283 s = 28.3 ms
  λ_a = 90 → ρ = 0.90 → W_q = 0.90/(2·100·0.10) = 0.0450 s = 45.0 ms
  λ_a = 95 → ρ = 0.95 → W_q = 0.95/(2·100·0.05) = 0.0950 s = 95.0 ms
  λ_a = 99 → ρ = 0.99 → W_q = 0.99/(2·100·0.01) = 0.4950 s = 495 ms
INTERPRETATION: the wait is not linear in admitted rate; the last 5% of capacity
                costs five times the wait of the first 80%.
```

Little's law closes the loop and provides an independent check:

```text
FORMULA [C-07]: L = λ · W ,  W = W_q + 1/μ
WHERE: L mean items in system [requests]; λ arrival rate [requests/s];
       W mean sojourn time [s]; 1/μ mean service time [s].
CONSTRAINT: stable system (ρ < 1); long-run averages.
NUMERICAL EXAMPLE (λ = 85, μ = 100): W = 0.0283 + 0.0100 = 0.0383 s;
             L = 85 · 0.0383 = 3.26 requests.
CROSS-CHECK: L = L_q + ρ with L_q = ρ²/(2(1−ρ)) = 0.7225/0.30 = 2.408;
             2.408 + 0.850 = 3.258 requests. ✓ (rounding)
INTERPRETATION: at 85% utilisation the backend holds about three requests, not a queue.
```

The decision consequence is direct: **the global budget must be set at ρ_target below 1, and the choice of ρ_target is a latency decision.** This case uses r = 85 against μ = 100, giving ρ = 0.85 and a modelled wait near 28 ms, because the alternative (r = 100) leaves no margin for the burst that the bucket itself permits or for the service-time variability that the M/D/1 assumption excludes.

The second consequence is a rejection cost that must be stated honestly. At r = 85, admissions over 360 s are bounded by `C + r·t = 240 + 30,600 = 30,840`, so rejections are `43,200 − 30,840 = 12,360`, or 28.6% of arrivals. At r = 100 they would be 6,960, or 16.1%. **Protecting the backend by 15 percentage points of utilisation costs 5,400 additional rejections in the window.** That is the actual trade the owner signs, and it is invisible in a mean-rate discussion.

| Budget choice | Admitted cap in 360 s | Rejections | Rejection share | Modelled W_q | Modelled system held |
|---|---:|---:|---:|---:|---:|
| r = 100 (equals μ) | 36,240 | 6,960 | 16.1% | undefined (ρ = 1) | unbounded |
| r = 95 | 34,440 | 8,760 | 20.3% | 95.0 ms | 9.98 |
| r = 90 | 32,640 | 10,560 | 24.4% | 45.0 ms | 4.95 |
| r = 85 (selected) | 30,840 | 12,360 | 28.6% | 28.3 ms | 3.26 |
| r = 80 | 29,040 | 14,160 | 32.8% | 20.0 ms | 2.40 |

The final column is Little's law applied to the same model: `L = λ(W_q + 1/μ)`, so at r = 95 the backend holds about ten requests and at r = 80 it holds about two and a half. This is the quantity that decides whether a burst can be absorbed at all: a full bucket of C = 240 arriving at a backend already holding ten requests adds 240, not 10.

The table is the case's central artefact. It shows that every millisecond of tail latency is purchased with rejections, and it shows why "r = μ" is the one row that cannot be defended: the model does not produce a wait at ρ = 1, because the wait is infinite.

### 6.5 Sizing the burst: C is a backend decision

C is the number of requests the backend may receive back-to-back after an idle period. Two independent constraints bound it: correctness (the bucket must not admit more than the backend can absorb) and fairness (a tenant must be able to run a legitimate batch).

```text
FORMULA [C-08]: C ≤ κ · H ,  H = μ · T_absorb
WHERE: C capacity [requests]; κ headroom fraction [dimensionless, 0 < κ ≤ 1];
       H absorbable burst [requests]; μ service rate [requests/s];
       T_absorb tolerated absorption time [s].
CONSTRAINT: backend buffer can hold H; service continues during absorption.
NUMERICAL EXAMPLE: μ = 100, T_absorb = 24 s → H = 2,400 requests;
                   κ = 0.10 → C ≤ 240 requests.
INTERPRETATION: a full bucket is a 2.4 s burst at service rate, i.e. 10% of the
                absorption budget; emptying it at a surplus of 40 req/s takes 6 s.
```

The drain check is what makes C defensible:

```text
FORMULA [C-09]: T_drain = C / (μ − r)
WHERE: T_drain time to empty a full bucket at admitted rate r [s]; C [requests];
       μ [requests/s]; r [requests/s].
CONSTRAINT: μ > r; no new burst during drain.
NUMERICAL EXAMPLE: 240 / (100 − 85) = 240 / 15 = 16.0 s.
INTERPRETATION: after one maximal burst the backend needs 16 s of quiet to return
                to its pre-burst queue state; a tenant that bursts every 10 s
                would keep the queue permanently elevated.
```

This produces a fairness constraint that is easy to miss: a bucket of 240 with refill 85 can be refilled from empty in `C/r = 2.82 s`, which is *faster* than the backend can drain the corresponding burst (16 s). Therefore a single tenant bursting every 3 s would keep the backend loaded even though its long-run rate is at budget. The resolution is per-tenant buckets (so one tenant cannot do this) plus a global bucket whose C is smaller than any single tenant's C. The case therefore uses a two-level structure: per-tenant C_t = 60, r_t = 20 for a standard tenant; global C_g = 240, r_g = 85.

### 6.6 Distributed accounting: leases versus partitions

Eight gateway nodes must enforce one global budget. Two designs are compared.

**Design A — deducted leases.** The coordinator holds a global token count. A node requests a lease of L tokens; the coordinator atomically subtracts L and returns it. The node spends locally. Unspent lease tokens are *lost* until the next request (or are returned on a timer).

Invariant: the sum of tokens issued never exceeds the budget, so **overshoot is impossible by construction**. The failure mode is the opposite: utilisation loss.

```text
FORMULA [C-10]: U_loss ≤ (N − 1) · L / (r · P)
WHERE: U_loss wasted fraction of budget [dimensionless];
       N nodes; L lease size [tokens]; r budget [tokens/s];
       P lease period [s].
CONSTRAINT: worst case — N−1 nodes hold a full unused lease for a whole period.
NUMERICAL EXAMPLE: N = 8, L = 13, r = 85, P = 1 s →
                   U_loss ≤ 7 · 13 / (85 · 1) = 91/85 = 1.07 → bounded at 100%.
INTERPRETATION: with a one-second period the worst case can strand the entire
                budget, so the period must be sized against L, not chosen by taste.
```

The 100% figure is not a modelling error; it is the correct answer for that configuration and it explains why leases must be *returned* or must be short relative to the traffic. With P = 0.25 s and L = 13, the bound becomes `91/(85·0.25) = 4.28`, still above 1, so the only workable regime is where `(N−1)·L < r·P`. That inequality is the design rule:

```text
FORMULA [C-11]: L < r · P / (N − 1)
WHERE: symbols as above; strict inequality required for the bound to be < 100%.
NUMERICAL EXAMPLE: r = 85, P = 1, N = 8 → L < 85/7 = 12.14, so the largest integer
                   lease satisfying the inequality is L = 12.
CHECK: U_loss ≤ 7·12/(85·1) = 84/85 = 98.8%, i.e. the inequality holds but leaves
       almost the entire budget vulnerable to being stranded.
```

The check exposes the real constraint: a one-second lease period is far too long for eight nodes at 85 tokens/s, because each node's fair share is only ~10.6 tokens/s. With P = 0.1 s, `L < 85·0.1/7 = 1.21`, i.e. a lease of one token — which collapses into a per-request coordinator call and defeats the purpose. **This is the honest finding of the lease analysis: at a budget of 85 tokens/s across eight nodes, leases of useful size require a coordinator call rate close to the request rate, so the latency argument for leases does not hold at this scale.** At larger scale the arithmetic reverses, because the coordinator call rate scales with `r/L` while the request rate scales with `r`.

**Design B — partitioned local buckets.** Each node runs its own token bucket with `C/N` and `r/N`. No coordinator on the request path.

Overshoot sources: (i) clock skew inflating local refill; (ii) traffic imbalance, where one node receives more than its share but can only spend its own bucket; (iii) bucket refill at different times producing a combined burst.

```text
FORMULA [C-12]: O_skew = r · δ
WHERE: O_skew excess admissions per skew event [requests]; r global budget [req/s];
       δ clock skew magnitude [s].
NUMERICAL EXAMPLE: r = 85, δ = 0.25 → O_skew = 21.25 requests.
INTERPRETATION: a 250 ms advance on every node's clock releases 21 extra tokens,
                i.e. 25% of one second's budget, once per skew event.
```

```text
FORMULA [C-13]: O_part = Σ_i (C/N + (r/N)·W) = C + r·W   [ideal partition]
WHERE: O_part aggregate admissions in window W under equal partitioning.
NUMERICAL EXAMPLE (C = 240, r = 85, W = 1): 240 + 85 = 325.
COMPARISON: the single global bucket admits 325 in the same window.
INTERPRETATION: equal partitioning recreates the global envelope exactly; the
                deviation comes from unequal traffic, not from partitioning itself.
```

The consequence is a selection rule with a number attached: partitioning is *budget-equivalent* under equal traffic and over-admits under imbalance plus skew. Whether that is acceptable is exactly the C-02 question. Since C-02 forbids over-admission by design, Design B is admissible only with a skew bound enforced by monitoring (δ ≤ 250 ms alerts) and a compensating reduction of r by the skew allowance: `r_eff = r · (1 − δ/P)`. With δ = 0.25 s and P = 1 s, `r_eff = 85 · 0.75 = 63.75`, i.e. a 25% haircut on purchased capacity.

| Design | Overshoot by construction | Capacity wasted | Coordinator QPS | Failure-mode shape |
|---|---|---:|---:|---|
| A: deducted leases | none | up to `(N−1)L` per period | `r/L` | under-admits during coordinator loss |
| B: partitioned local | `r·δ` per skew event | 0 (but over-admits) | 0 | over-admits during imbalance |
| C: per-request central check | none | 0 | `r` | limiter becomes the bottleneck |
| D: hybrid (lease + local emergency) | bounded by emergency size | bounded | `r/L` | bounded both ways |

**Selected: D.** The hybrid holds deducted leases for the normal path, so the budget is never exceeded, and a local emergency bucket of `C/N` tokens that refills at `r_eff/N` only while the coordinator is unreachable, capped at a declared duration `T_emergency = 30 s`. The worst case during an outage is therefore `C/N + (r_eff/N)·30` per node, and the outage is declared by alert at 5 s. This is the only design in the table whose failure is bounded in both directions, and it is selected for that reason rather than for speed.

### 6.7 Cost of the decision path

The limiter's cost must be counted in the same units as the capacity it protects. Three components dominate: coordinator operations, rejection egress, and added latency.

```text
FORMULA [C-14]: Ops_day = q · 86,400
WHERE: Ops_day coordinator operations [ops/day]; q coordinator query rate [ops/s].
NUMERICAL EXAMPLE (per-request central check at r = 85): 85 · 86,400 = 7,344,000 ops/day.
      (lease path, L = 12: one call per 12 tokens → q = 85/12 = 7.08 calls/s
       → 7.08 · 86,400 = 611,712 ops/day; a 12× reduction, not an elimination.)
INTERPRETATION: a per-request central check costs one operation per admitted request,
                which is the largest possible multiplier; a lease divides it by L.
```

With an illustrative price of $0.20 per million operations (a constructed figure, not a quote), the per-request path costs `7.344 · 0.20 = $1.47/day`, and the lease path with one call per 12 tokens costs `($1.47/12) = $0.12/day` at the same budget. Both are immaterial at this scale; the cost argument for leases is therefore *latency*, not price, and Section 6.6 has already shown that the latency argument does not survive at 85 tokens/s. The conclusion is that the case should not claim an economics benefit it cannot demonstrate.

Rejection egress is more interesting because it scales with the rejection rate that the budget choice creates:

```text
FORMULA [C-15]: E = n_rej · b
WHERE: E egress [bytes]; n_rej rejected responses [count]; b bytes per 429 response [B].
NUMERICAL EXAMPLE: n_rej = 12,360, b = 200 B → E = 2,472,000 B = 2.472 MB per window.
INTERPRETATION: rejections are cheap in bytes; the cost of a rejection is client-side
                retry behaviour and support load, not network egress.
```

The negative result is useful: egress is not a reason to prefer one budget over another, so the budget decision must be made on latency and fairness alone. Case studies that justify a limiter on egress savings are usually confusing the limiter with a cache.

### 6.8 A worked decision trace

The envelope formulas say what is *possible*; a trace says what the implementation *does*. The following seven arrivals exercise the states that matter, including the boundary the formulas cannot express. Times are seconds from window start; the bucket starts full at C = 4, r = 2 for readability (a 4-token bucket refilling twice per second), which preserves every decision rule while fitting on the page.

| # | t (s) | tokens before | Δt since last refill (s) | refill | tokens available | decision | tokens after |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.00 | 4.000 | 0.00 | 0.000 | 4.000 | admit | 3.000 |
| 2 | 0.10 | 3.000 | 0.10 | 0.200 | 3.200 | admit | 2.200 |
| 3 | 0.20 | 2.200 | 0.10 | 0.200 | 2.400 | admit | 1.400 |
| 4 | 0.30 | 1.400 | 0.10 | 0.200 | 1.600 | admit | 0.600 |
| 5 | 0.40 | 0.600 | 0.10 | 0.200 | 0.800 | **reject** | 0.600 (unchanged) |
| 6 | 1.90 | 0.600 | 1.50 | 3.000, capped at 3.600 | 3.600 | admit | 2.600 |
| 7 | 1.95 | 2.600 | 0.05 | 0.100 | 2.700 | admit | 1.700 |

The trace is worth reading closely for three reasons.

First, row 5 shows the correct failure: with 0.800 tokens available the request is rejected and the bucket is *not* debited below zero. A naive implementation that subtracts unconditionally drives the counter negative and then silently owes tokens, which produces an under-admission that appears as unexplained rejections later.

Second, row 6 shows the cap doing real work. Between t = 0.40 and t = 1.90 the bucket would accrue 1.50 s · 2.0 = 3.000 tokens. Adding those to the 0.600 held gives 3.600, which is below C = 4.000, so the cap does *not* bind here — the bucket refills to 3.600 and only then admits, leaving 2.600. A longer idle period (beyond 1.70 s of accumulated refill) would hit the cap at 4.000 and discard the excess. Whether an implementation caps on write or caps on read changes nothing in this trace but does change the stored value between decisions, which matters for any operator reading raw state.

Third, rows 6 and 7 show the burst the envelope predicted. Two admits 50 ms apart, immediately after the bucket recovered, is exactly the `C + r·W` behaviour: the bucket pays for the recovery burst out of C, not out of r. The pair also shows why C is a backend decision: those two admits arrive at the backend together regardless of what the long-run rate permits.

The corresponding GCRA trace for the same arrivals is a single variable. With `T = 1/r = 0.5 s` and `τ = (C − 1)T = 1.5 s`, the decision at time `t` is `t ≥ TAT − τ`, followed by `TAT = max(TAT, t) + T`. Starting with `TAT = 0`:

| # | t (s) | TAT before | test `t ≥ TAT − 1.5` | decision | TAT after |
|---:|---:|---:|---|---:|
| 1 | 0.00 | 0.000 | 0 ≥ −1.5 true | admit | 0.500 |
| 2 | 0.10 | 0.500 | 0.10 ≥ −1.0 true | admit | 0.600 |
| 3 | 0.20 | 0.600 | 0.20 ≥ −0.9 true | admit | 0.700 |
| 4 | 0.30 | 0.700 | 0.30 ≥ −0.8 true | admit | 0.800 |
| 5 | 0.40 | 0.800 | 0.40 ≥ −0.7 true | **admit** | 0.900 |
| 6 | 1.90 | 0.900 | 1.90 ≥ 0.9 true | admit | 2.400 |
| 7 | 1.95 | 2.400 | 1.95 ≥ 0.9 true | admit | 2.450 |

The two traces disagree at row 5. The token bucket rejected because it had only 0.800 tokens; GCRA admitted because `0.40 ≥ 0.80 − 1.5`. This is not an error in either: the equivalence between the two representations holds for the *admitted set*, and the disagreement here comes from the token bucket starting with 4 full tokens while GCRA's initial `TAT = 0` encodes a bucket that has been idle long enough to be full. Aligning the initial conditions (`TAT = (C − 1)T = 1.5` at t = 0) does not fix it either, because the token bucket's refill is capped at C while GCRA's TAT can run arbitrarily far ahead of the current time.

The honest conclusion is the one that matters operationally: **the two representations are equivalent as envelopes but not as implementations**, and a migration between them will change individual decisions at the boundary even though the long-run rate is unchanged. That is a reason to pick one and version it, not to assume interchangeability.

### 6.9 Per-tenant share arithmetic

A global budget does not say how it is divided. Two rules are defensible and they fail differently.

**Proportional to declared quota.** If tenant `i` has declared rate `q_i` and the sum of declared rates is `Q = Σq_i`, the tenant's share is `s_i = q_i/Q`. This preserves contractual proportions but does not reserve capacity: if the sum of declared quotas exceeds the global budget (which is normal, because quotas are sold on peak expectations), then `Σs_i·r < r` only if every tenant uses its full quota simultaneously, which never happens — and when it does, everyone is throttled proportionally.

**Equal share per active tenant.** `s_i = 1/|A|` where `A` is the set of tenants with traffic in the window. This is simple and adapts to the live population, but it makes a tenant's throughput depend on how many other tenants happen to be active, which is unacceptable for a metered product.

```text
FORMULA [C-16]: Σ_i s_i · r ≤ r  with  s_i = q_i / Q
WHERE: s_i share fraction [dimensionless]; q_i declared rate [req/s]; Q = Σq_i [req/s];
       r global budget [req/s].
CONSTRAINT: declared quotas may legitimately sum above r (oversubscription).
NUMERICAL EXAMPLE: q_1 = 30, q_2 = 20, and 4,998 tenants at 0.01 each → Q ≈ 99.98;
                   s_1 = 0.300, s_2 = 0.200 → at r = 85 tenant 1 receives 25.5 req/s.
INTERPRETATION: oversubscription is safe only while simultaneous full-quota use is rare;
                the limiter must therefore expose a global rejection reason separately
                from a tenant rejection reason, or the two failures cannot be told apart.
```

The oversubscription ratio is the number that decides whether the product is honest:

```text
FORMULA [C-17]: OS = Σ q_i / r
WHERE: OS oversubscription ratio [dimensionless]; Σq_i declared rates [req/s]; r [req/s].
NUMERICAL EXAMPLE: Σq_i = 99.98, r = 85 → OS = 1.176.
INTERPRETATION: if every tenant used its declared peak at once, the system would reject
                15% of it; the commercial claim must therefore be a burst allowance,
                not a guaranteed simultaneous rate.
```

An OS of 1.176 is mild. At OS = 4 (a common shape when quotas are sold against a much smaller steady-state budget) the product promises something it cannot deliver simultaneously, and the rejection reason code becomes a legal artefact rather than a diagnostic one. This case does not adjudicate any contract; it records that the ratio exists, that it is computable, and that it must be disclosed to whoever owns the commercial commitment.

**Sensitivity and the boundaries that change the answer.**

| Parameter changed | From → To | Effect | Decision consequence |
|---|---|---|---|
| Overload duration | 360 s → 720 s | backlog 7,200 → 14,400 | buffer sizing is not the fix; budget is |
| Recovery surplus | 60 → 20 req/s | drain 120 s → 360 s | recovery plan must include rate reduction |
| Recovery surplus | 60 → 0 | T undefined | no drain without a surplus, ever |
| Refill r | 85 → 95 | rejects 12,360 → 8,760; W_q 28.3 → 95 ms | 5× tail wait for 3,600 fewer rejects |
| Capacity C | 240 → 480 | max 1 s burst 325 → 565 | full bucket exceeds the absorption budget |
| Clock skew δ | 0.25 s → 1.0 s | over-admission 21 → 85 req | partition design becomes indefensible |
| Tenants | 5,000 → 50 | state memory irrelevant; fairness still needed | per-tenant buckets remain |

The two rows that reverse the design are the recovery-surplus row (zero surplus invalidates the entire queueing model and forces a rate reduction, not a limiter) and the skew row (a one-second skew makes partitioned local buckets over-admit a full second of budget per event, which no compensating haircut can cover without destroying utilisation).

**What this walkthrough does not establish.** No latency, rejection rate, or overshoot value here was observed. The M/D/1 model assumes Poisson arrivals, but a token bucket *shapes* arrivals into bursts, which are worse than Poisson for a downstream queue; the modelled 28.3 ms is therefore optimistic for a bursty admitted stream and must be treated as a lower bound on real wait. The correct statement is: "under a fluid arrival model and an M/D/1 service model, the declared budget produces a wait near 28 ms; the true wait for bursty admitted traffic is higher and unmeasured." A reader who repeats "28 ms" as a system property has misread the case.

## 7. The Finding

The supported conclusion is conditional and has four parts.

1. **The budget must be below capacity.** A global admitted rate equal to service capacity is not protective; the wait diverges as utilisation approaches one, so the policy must declare a target utilisation (85% here) and accept the rejection cost that follows. For the illustrative window that cost is 12,360 rejections instead of 6,960 — 28.6% versus 16.1% of arrivals.
2. **Burst must be bounded by a named absorption budget, not by a round number.** With μ = 100 and a 24 s absorption budget, C ≤ 240 requests; a full bucket drains in 16 s at the selected refill, which is the real constraint on how often a tenant may burst.
3. **Global enforcement must conserve the budget.** Deducted leases make overshoot impossible and therefore satisfy the hard constraint; partitioned local buckets over-admit by `r·δ` per skew event and are admissible only with a measured skew bound and a compensating haircut that can cost a quarter of purchased capacity.
4. **At this scale the coordinator cost argument fails.** With 85 tokens/s across eight nodes, a lease small enough to keep stranded capacity tolerable is small enough that the coordinator call rate approaches the request rate; the case therefore selects a hybrid design for its *bounded failure*, not for its latency.

**What is not claimed.** No rejection rate, latency, overshoot, or cost saving was measured. The recommendation is a design recommendation for a proposed policy, and its acceptance criteria are in Section 9. The arithmetic is checked; the system is not built.

**What would falsify the finding.** If a load test at the declared payload showed capacity near 140 requests/s rather than 100, the correct budget would rise, the rejection cost would fall, and the entire trade table in Section 6.4 would shift downward. If the measured service-time coefficient of variation were near 1 rather than near 0, the M/D/1 wait would be an underestimate by roughly a factor of two and the target utilisation would have to fall. Both are cheap tests and both are prerequisites for using this case's numbers in a real decision.

## 8. System/Solution Design

### 8.1 Request path

```text
client
  |
  v
[gateway node i]
  |-- 1. authenticate, extract tenant_id (no body buffering)
  |-- 2. per-tenant bucket (local state)  --reject--> 429 LIMIT_TENANT
  |-- 3. global lease check               --reject--> 429 LIMIT_GLOBAL
  |        |-- coordinator reachable? --no--> local emergency bucket
  |        |                                    |--exhausted--> 429 COORD_UNAVAILABLE_REJECTED
  |        |                                    '--available--> admit COORD_UNAVAILABLE_ADMITTED
  |        '--yes--> deduct from lease (atomic, in-memory)
  |-- 4. append admission record (policy version, node, tenant, decision, state snapshot)
  |-- 5. admit --> proxy to backend --> backend executes and acknowledges
  '-- 6. reject --> 429 with Retry-After and reason code; no backend call
```

The order is not arbitrary. Authentication precedes admission so that an unauthenticated flood cannot consume the tenant bucket of a real tenant. The per-tenant check precedes the global check so that a single abusive tenant cannot drain the global lease before being stopped. The ledger write precedes the proxy call so that an admitted request cannot execute without a record; the cost is one local append, which is why the ledger is node-local and batched rather than a synchronous remote write.

### 8.2 State ownership

| State | Owner | Durability | Loss consequence | Recovery |
|---|---|---|---|---|
| Per-tenant bucket (tokens, TAT) | gateway node memory | volatile | a restart grants a full bucket | refill from coordinator snapshot; accept one burst |
| Lease remainder | gateway node memory | volatile | unspent tokens stranded | return on shutdown; else bounded waste |
| Global token count | coordinator | replicated | budget unknown | re-derive from ledger over the last period |
| Policy version | policy registry | versioned | decisions not reproducible | block traffic until version resolves |
| Admission ledger | append-only store | durable | audit impossible | stop admitting; this is a hard dependency |

The ledger being a hard dependency is a deliberate choice with a stated cost: if the ledger cannot be written, the gateway must stop admitting rather than admit unrecorded work, because C-01 makes an execution without a record a policy breach. The alternative — admit and buffer records locally — is what the design actually does for short outages, with a bounded buffer whose overflow triggers rejection. That is the compromise, and it is written here rather than discovered during an incident.

### 8.3 Failure transitions

| Transition | Trigger | Behaviour | Bound |
|---|---|---|---|
| Normal → degraded | coordinator errors above threshold | switch to emergency bucket | `C/N` per node, refill `r_eff/N` |
| Degraded → normal | coordinator healthy for 3 consecutive probes | return to lease path | leases re-deducted, no double spend |
| Normal → rejecting | global budget exhausted | 429 with `Retry-After` | per-tenant share preserved |
| Node restart | process loss | full bucket on restart | one extra burst per restart, bounded by C |
| Policy change | new version published | new version applies to new decisions | old decisions remain reproducible |
| Clock step | NTP correction | monotonic clock unaffected; wall clock jumps | refill uses monotonic only |

The clock row deserves emphasis: refill arithmetic must use a monotonic clock, because a wall-clock step backwards would make `Δt` negative and either freeze the bucket or, with careless code, refund tokens. Reporting uses wall clock. Mixing the two is the origin of most "the limiter randomly allows a burst" reports.

### 8.4 Operational readiness

The design is not ready to operate until four artefacts exist: a dashboard showing admit/reject by reason and node, an alert on per-node skew above 250 ms, a replay tool that reconstructs decisions from the ledger for a chosen window, and a runbook that states the emergency-bucket duration and the escalation path when it is exhausted. None of these exists in this manuscript; they are listed as acceptance prerequisites so that "the limiter is deployed" cannot be mistaken for "the limiter is operable".

## 9. Evaluation & Validation

Validation is organised as a numbered protocol so that each item produces an artefact rather than an opinion. Every item is currently **not executed**; the table states what would count as a pass.

| ID | Check | Method | Pass criterion | Status |
|---|---|---|---|---|
| V01 | Envelope correctness | compare bucket against ORC-01 on 10,000 synthetic arrivals | zero disagreements | not executed |
| V02 | Window-boundary defect reproduced | run fixed window at a boundary, then token bucket | fixed window shows ~2×; bucket shows C + r·W | not executed |
| V03 | No over-admission | replay worst-case lease interleaving | admitted ≤ budget at every prefix | not executed |
| V04 | Rejection accounting | sum admits + rejects = arrivals | exact equality, no dropped records | not executed |
| V05 | Ledger completeness | join admissions to backend executions | every execution has exactly one admission | not executed |
| V06 | Coordinator outage | inject loss for 60 s | admits bounded by emergency budget | not executed |
| V07 | Clock skew | offset one node by 250 ms | measured over-admission ≤ r·δ | not executed |
| V08 | Fairness | one tenant sends 60% of traffic | that tenant rejected, others unaffected | not executed |
| V09 | Burst absorption | one tenant sends C at once | backend queue returns to baseline within T_drain | not executed |
| V10 | Latency cost | measure decision latency p50/p99 | p99 within the declared budget | not executed |
| V11 | Retry amplification | clients retry per Retry-After | executions per attempt ≤ declared bound | not executed |
| V12 | Coordinated omission | report admitted and rejected latency separately | both present, neither aggregated alone | not executed |
| V13 | Policy reproducibility | replay a window with the stored version | identical decisions | not executed |
| V14 | Restart burst | restart a node mid-window | extra burst ≤ C, detected by alert | not executed |
| V15 | Recovery | restore normal path after outage | no double-spend, no stranded lease | not executed |

**Why V12 is a pass criterion and not a reporting detail.** If the only latency metric is computed over admitted requests, a policy improves its own score by rejecting the slow ones. V12 forces both populations into the same report, which makes the trade in Section 6.4 visible to whoever owns the budget. A dashboard that cannot show rejected-request latency cannot support the decision.

**Statistical honesty.** There is no confidence interval in this case because there is no sample: all values are derived from declared inputs. Once V01–V15 produce data, the reported rejection rate must carry a denominator (arrivals in the window), a population (authenticated requests), a window (start and end in UTC), and a comparator (the previous policy version). A rejection rate without a denominator is the single most common way this kind of work is misreported.

**Boundary tests that must be included.** Zero-traffic window (denominators become zero; the correct output is "undefined", not "0%"); C = 1 (no burst at all; GCRA tolerance τ = 0); r = 0 (nothing admitted; must be a configuration error, not a silent block); a tenant with a single request (fair-share arithmetic must not divide by zero); and a clock step backwards (must not refund tokens).

## 10. Action Taken

No production action was taken. The recorded decision is a *proposal* with a defined gate:

1. Publish the policy as a versioned configuration (C = 240, r = 85, C_t = 60, r_t = 20, lease L = 12, emergency window 30 s) into a non-serving registry so that its contents are reviewable before it can affect traffic.
2. Execute V01–V03 offline against synthetic traces; these require no production traffic and can falsify the arithmetic.
3. Obtain the backend capacity parameter from a load test at the declared payload before any budget is treated as calibrated.
4. Run the limiter in shadow mode — decide, record, do not enforce — for one full business cycle, then compare shadow decisions with actual outcomes.
5. Enforce only after shadow data shows that the rejection share for compliant tenants is within the owner's stated tolerance.

The shadow-mode step is the part most often skipped. It is also the only step that can reveal that the arrivals modelled at 120 requests/s are, in reality, 40 requests/s with a 3-second burst of 300 — a shape that invalidates the fluid model while leaving its arithmetic intact.

**Rollback.** Disabling the limiter is a configuration change that restores unbounded admission, and it is safe only if the backend has headroom. The rollback therefore must be paired with a rate reduction at the client or an autoscale step; otherwise the rollback converts a rejection problem into an overload problem. This pairing is part of the action record, not an operational afterthought.

## 11. Outcome & Results

**Result status: not measured.** The only observed results in this manuscript are arithmetic derived from declared inputs and recomputed independently:

| Quantity | Value | Derivation | Evidence |
|---|---:|---|---|
| Backlog without a limiter | 7,200 requests | (120 − 100) · 360 | derived |
| Drain time at 60 req/s surplus | 120 s | 7,200 / 60 | derived |
| Token-bucket 1 s envelope | 325 requests | 240 + 85 · 1 | derived |
| Rejections at r = 85 | 12,360 | 43,200 − 30,840 | derived |
| Rejections at r = 100 | 6,960 | 43,200 − 36,240 | derived |
| Modelled wait at ρ = 0.85 | 28.3 ms | 0.85/(2·100·0.15) | derived |
| Skew over-admission at 250 ms | 21.25 requests | 85 · 0.25 | derived |
| Leases stranded, worst case | 91 tokens | (8 − 1) · 13 | derived |

Nothing in the table is a measurement. A future evaluation must report, at minimum: admitted count, rejected count and their sum; the population and window; the policy version; the observed admitted-rate maximum over any 1 s window compared with `C + r·W`; the observed per-node skew distribution; and the admitted-versus-rejected latency distributions reported separately. Until those exist, the honest status is "designed and derived, not validated".

**What would count as a positive result.** Not "the backend stayed up" — that is a statement about one window and could be luck. A positive result is: the observed maximum admissions in any 1 s window did not exceed the envelope; no admitted request lacked a ledger record; the rejection share for tenants below their declared quota stayed under the owner's tolerance; and the modelled wait was not exceeded at the 99th percentile for two consecutive business cycles. Each of those is checkable and each can fail independently.

## 12. Failure Modes & What Went Wrong

Each entry separates the mechanism from the symptom, because the same symptom has several mechanisms and choosing the wrong one produces a fix that moves the failure. These are hypothetical teaching failures unless a row says otherwise.

### F-01 — Boundary doubling mistaken for a burst

**Symptom.** The backend sees a spike of roughly twice the configured limit, always at a round timestamp boundary.
**Mechanism.** A fixed-window counter resets at the boundary, so a client with 100 units of patience spends 100 in the last moment of one window and 100 in the first moment of the next.
**Competing causes.** A genuine client burst; a clock-aligned batch job; a retry wave after a brief error.
**Discriminating check.** Plot admissions in 50 ms buckets against the window boundary. If the spike is phase-locked to the boundary rather than to a client identifier, it is the counter.
**Immediate recovery.** Nothing to recover; the backend absorbed it or did not. Reject at the client for one window if it did not.
**Repair.** Replace the counter with a token bucket or sliding-window estimate; then verify with V02.
**Verification.** The synthetic boundary trace in V02 must show the 2× spike for the old mechanism and `C + r·W` for the new one.

### F-02 — Budget set equal to measured capacity

**Symptom.** Tail latency is five to ten times the design value whenever the limiter is active, even though admission is "working".
**Mechanism.** ρ = 1. The queue is critical; the wait model has no finite value and the real system grows a backlog with any variance at all.
**Competing causes.** A slow downstream dependency; a garbage-collection pause; a hot partition on the backend's storage.
**Discriminating check.** Compare admitted rate against measured capacity. If admitted ≈ capacity within a few percent and waits exceed 100 ms, the budget is the cause, not the dependency.
**Immediate recovery.** Lower r by 10–15% and observe whether tail wait falls; if it does not, the cause is elsewhere.
**Repair.** Set the budget from a target utilisation (`r = ρ_target · μ`), and record ρ_target as the decision variable it is.
**Verification.** Re-run the load test at the new budget and confirm the wait curve matches the intended ρ.

### F-03 — Coordinator outage turns into either a blackout or an overload

**Symptom.** During a coordinator restart, either all traffic is rejected or none is limited.
**Mechanism.** The failure policy was never written, so the code does whatever the exception path happens to do.
**Competing causes.** An authentication dependency failure producing the same all-reject symptom; a load-balancer drain producing the same all-admit symptom.
**Discriminating check.** Correlate the onset with coordinator availability probes. A one-second offset that matches the coordinator restart is diagnostic.
**Immediate recovery.** Enable the local emergency bucket if it exists; otherwise reduce traffic at the client.
**Repair.** Implement the hybrid: bounded local emergency bucket plus an alert at 5 s of coordinator unavailability.
**Verification.** V06 must show admits bounded by `C/N + (r_eff/N)·T_emergency` per node.

### F-04 — One tenant consumes the global budget

**Symptom.** Many tenants receive 429 while one tenant's traffic is served normally.
**Mechanism.** Only a global bucket exists; the loudest client wins by definition.
**Competing causes.** A routing bug sending all traffic through one node whose local bucket is large; a shared tenant identifier due to a client misconfiguration.
**Discriminating check.** Compute per-tenant admitted counts against per-tenant quotas. A single tenant above 50% of the global budget with others near zero is conclusive.
**Immediate recovery.** Apply a temporary per-tenant cap in the gateway configuration.
**Repair.** Two-level buckets with declared shares; the global bucket remains the outer bound.
**Verification.** V08 sends 60% of traffic from one tenant and asserts the others are unaffected.

### F-05 — Retries double the effective arrival rate

**Symptom.** Backend executions exceed admission records by a ratio above one, and this ratio grows during incidents.
**Mechanism.** The limiter counts attempts; the backend executes attempts plus timed-out originals plus client retries.
**Competing causes.** A duplicate-delivery bug in the proxy; a ledger write that fails and is retried, inflating records instead of executions.
**Discriminating check.** Join by `request_id`: executions without a matching record indicate an unrecorded path; multiple executions sharing a record indicate retry duplication.
**Immediate recovery.** Set `Retry-After` and enforce server-side deduplication on `request_id` for a bounded window.
**Repair.** Define admission on executions; make the backend idempotent on `request_id`; cap client retries by contract.
**Verification.** V11 asserts executions per attempt at or below the declared bound.

### F-06 — Clock skew inflates local refill

**Symptom.** Aggregate admissions exceed the budget by about a quarter of one second's worth, intermittently, with no configuration change.
**Mechanism.** Partitioned local buckets refill from wall clocks. A node advanced by δ gains `r/N · δ` tokens that were never deducted anywhere.
**Competing causes.** A policy version mismatch across nodes; a node running an older binary with a larger C.
**Discriminating check.** Correlate excess admissions with per-node NTP offset. If the excess tracks offset, skew is the cause.
**Immediate recovery.** Reduce global r temporarily by the skew allowance.
**Repair.** Use monotonic clocks for refill and deducted leases for the global budget; alert on skew above 250 ms.
**Verification.** V07 offsets one node deliberately and measures the excess against `r·δ`.

### F-07 — A "safety" limiter that never limits

**Symptom.** Rejection counters are flat at zero for months, and an overload incident still occurs.
**Mechanism.** A per-node limit was set equal to the global budget, so the aggregate ceiling is N times the intent. This is the prior-attempt failure in Section 2.
**Competing causes.** Traffic genuinely below budget (then the incident has another cause); a metrics bug where the counter is not exported.
**Discriminating check.** Multiply the configured per-node limit by the node count and compare with capacity.
**Immediate recovery.** None needed; this is a latent defect, not an active failure.
**Repair.** Derive per-node values from the global budget and add a configuration assertion that `N · limit_node ≤ budget · (1 + tolerance)`.
**Verification.** A configuration test fails the build when the assertion is violated; plus V03.

### F-08 — Rejected requests still cost backend work

**Symptom.** Backend load does not fall when the rejection rate rises.
**Mechanism.** The gateway buffers and parses the body before deciding, or the rejection happens after the proxy call was already issued.
**Competing causes.** A retry loop that converts rejections into new attempts (load falls nowhere because total attempts rise); a cache-miss path that performs the expensive fetch before the check.
**Discriminating check.** Compare backend request count with the admission ledger's admit count. Equality means the short-circuit works; a gap means rejected work reached the backend.
**Immediate recovery.** Move the check ahead of body parsing behind a feature flag.
**Repair.** Enforce the ordering as an invariant with a test that asserts zero backend calls for a rejected request.
**Verification.** V05's join shows one execution per admission and none for rejections.

### F-09 — Burst capacity larger than the absorption budget

**Symptom.** Latency spikes immediately after an idle period, exactly when the system looked healthy.
**Mechanism.** C was chosen for fairness (a batch job must fit) without checking C against `μ · T_absorb`. A full bucket is a burst the backend must absorb, and the queue needs `C/(μ − r)` seconds to return.
**Competing causes.** A cold cache after idle producing slow first requests; a connection-pool warm-up effect.
**Discriminating check.** Plot inter-arrival gaps against subsequent latency. Spikes that follow idle gaps are bucket bursts, not cache effects; a cache effect would appear on the first request only, not across C requests.
**Immediate recovery.** Reduce C; accept that batch jobs will now spread.
**Repair.** Size C from the absorption budget (C-08) and state `T_drain` in the policy.
**Verification.** V09 sends exactly C at once and asserts the queue returns to baseline within `T_drain`.

### F-10 — The limiter's own latency becomes the tail

**Symptom.** Gateway p99 rises after the limiter is enabled, and rises further under load.
**Mechanism.** A synchronous coordinator call on every request adds a network round trip whose tail is much worse than its median, and the coordinator itself becomes loaded by the limiter's traffic.
**Competing causes.** Logging the full request at decision time; a lock held across the proxy call; TLS renegotiation.
**Discriminating check.** Compare decision latency with and without the coordinator, and correlate coordinator CPU with gateway p99. Co-movement indicates the limiter is the source.
**Immediate recovery.** Increase lease size to reduce call rate, accepting more stranded capacity.
**Repair.** Move to the hybrid lease design and measure the coordinator's own queue.
**Verification.** V10 asserts decision p99 within budget at the design request rate and at 2× it.

### F-11 — Fairness arithmetic divides by zero

**Symptom.** A tenant with no recorded traffic receives NaN or a full-budget share; a new tenant is rejected immediately.
**Mechanism.** Share computation over an empty population, or a new tenant initialised with zero tokens and a refill that requires a prior timestamp.
**Competing causes.** A missing tenant record treated as "unlimited"; a default that grants the global bucket.
**Discriminating check.** Enumerate tenants with no prior state and inspect the first decision. A first decision of "reject" or a share of "NaN" is diagnostic.
**Immediate recovery.** Seed new tenants with `min(C_t, 1)` token and a current timestamp.
**Repair.** Define the empty-population policy explicitly: a new tenant receives its declared burst, not a share of an undefined denominator.
**Verification.** Boundary test with one tenant and one request.

### F-12 — Coordinated omission inflates the apparent result

**Symptom.** The latency dashboard improves after the limiter is enabled, while support tickets about rejections rise.
**Mechanism.** Latency is aggregated over admitted requests only, so the policy improves its own metric by discarding the slowest work.
**Competing causes.** A genuine improvement from reduced backend load (then rejection tickets should not rise); a sampling change in the metrics pipeline.
**Discriminating check.** Recompute latency including rejected requests, using time-to-429 as their latency. If the combined distribution is worse, the "improvement" is an artefact.
**Immediate recovery.** Add the rejection series to the same dashboard.
**Repair.** Make the paired report mandatory in the acceptance criteria (V12).
**Verification.** The dashboard must fail review if either series is absent.

### F-13 — Policy drift without a version

**Symptom.** Two decisions with identical inputs produce different outcomes, and nobody can say why.
**Mechanism.** Configuration was edited in place; the change was not versioned, so the ledger cannot distinguish the runs.
**Competing causes.** Non-deterministic tie-breaking (for example, iteration order over a hash map); a node running a stale binary.
**Discriminating check.** Compare ledgers from two nodes for the same window; differing policy hashes with identical inputs is conclusive.
**Immediate recovery.** Pin all nodes to one version and replay.
**Repair.** Version the configuration, record the hash per decision, and require a change record for any budget edit.
**Verification.** V13 replays a window and asserts identical decisions.

### F-14 — Recovery after outage double-spends

**Symptom.** Just after the coordinator returns, admissions exceed the budget for a few seconds.
**Mechanism.** Emergency-bucket tokens were spent locally *and* the lease was re-deducted on recovery without accounting for what the emergency path already granted.
**Competing causes.** A retry storm from clients that were rejected during the outage (a different mechanism with the same timing).
**Discriminating check.** Compare emergency-path admissions with the first post-recovery lease deductions. Overlap means double spend.
**Immediate recovery.** Reduce r for one period equal to the overlap.
**Repair.** Record emergency grants in the ledger with their reason code and subtract them from the first recovered lease.
**Verification.** V15 asserts no double spend across an injected 60 s outage.

### F-15 — The limiter is treated as a security control

**Symptom.** A credential-stuffing attempt is slowed but not stopped, and the incident review asks why the limiter did not prevent it.
**Mechanism.** Role confusion: admission control protects capacity, not authorisation. A distributed attacker with many tenant identifiers has many buckets.
**Competing causes.** A genuine authorisation bypass; a shared credential.
**Discriminating check.** Count distinct tenant identifiers per source. A large number with small per-tenant volume indicates the limiter is being bypassed by identity rotation.
**Immediate recovery.** Apply a source-based limit as a separate control.
**Repair.** State in the policy that admission is a capacity control; add the security control as a distinct mechanism with its own owner.
**Verification.** A red-team replay with rotating identifiers must show the capacity limiter behaving as specified and the security control catching the pattern.

### F-16 — Zero-surplus recovery never drains

**Symptom.** The backlog persists indefinitely after the peak, even though the peak is over.
**Mechanism.** During "recovery", the arrival rate is still at or above service capacity. `T = B/(μ_r − λ_r)` has no finite value.
**Competing causes.** A leak in the queue (items never acknowledged, so they are re-delivered); a background job competing for the same service capacity.
**Discriminating check.** Measure λ and μ separately during the supposed recovery. If `μ_r − λ_r ≤ 0`, the model says no drain and the measurement will agree.
**Immediate recovery.** Reduce arrival rate (this is the only lever that works) or add capacity.
**Repair.** Make the recovery plan include a rate reduction step, not only a capacity step.
**Verification.** Replay with `λ_r = μ_r` and confirm the tool reports "no drain" rather than a large number.

## 13. Transferable Lesson

The transferable content is a checklist of six decisions, in order, each of which can invalidate the ones after it.

1. **Name the protected resource and its capacity in the same units as the limiter.** Here both are requests per second at a declared payload. A capacity measured at a different payload is a different quantity.
2. **Choose the target utilisation before the mechanism.** ρ_target converts a capacity into a budget, and it is the only place where the latency-versus-rejection trade is decided. A mechanism chosen first will be retrofitted with a budget that cannot be defended.
3. **Bound the burst with an absorption budget, not with a round number.** C is how much the protected resource can absorb in a stated time; the drain time `C/(μ − r)` then constrains how often a burst may occur.
4. **Decide whether over-admission is forbidden or merely unlikely, and say which.** This single question selects between conserving mechanisms (leases, central counters) and local mechanisms (partitions). Treating it as a performance question produces a design whose worst case is unknown.
5. **Write the failure policy as a bound.** "Fail open" and "fail closed" are not policies; "admit at most `C/N + (r_eff/N)·30 s` while the coordinator is unreachable" is.
6. **Make rejections a first-class reported outcome.** Otherwise the measurement rewards shedding work and the organisation learns the wrong lesson from its own dashboard.

**Where this lesson does not transfer.** If the protected resource has *no* meaningful service rate — an event stream with elastic backpressure, a batch job with a deadline rather than a rate — then the utilisation argument in Section 6.4 does not apply, and the budget should be derived from the deadline rather than from capacity. If arrivals are inherently smooth and low-volume, a limiter is pure overhead and the baseline is correct. The lesson transfers as a *procedure*, never as a set of numbers.

**The counterexample worth remembering.** A correctly implemented token bucket at r = μ is a correct implementation of the wrong decision. Every gate in this case can pass while the backend still melts, because the budget was set at the critical load. This is why the case puts the utilisation table before the mechanism comparison: the mechanism is the easy part.

## 14. Connections & Cross-References

**Related methods in this subtree.** The DSA patterns sheet supplies the sliding-window and prefix-sum machinery used for exact accounting and explains why an exact sliding log costs `O(r·W)` state. The system-design flow sheet supplies the capacity-estimation procedure that produces μ and the peak factor. The top-K streaming case shares the same "bounded state, approximate answer" tension that distinguishes a sliding-window counter from a token bucket.

**Related cases.** The feature-serving case shares the "two-level budget" pattern (per-entity plus global). The notification-platform case shares the idempotency-key discipline that makes retry accounting possible.

**Open questions.** (i) What is the measured service-time distribution at the declared payload? Without it, the M/D/1 wait is a lower bound. (ii) What is the real arrival histogram, including sub-second burstiness? The fluid model cannot express the shape that matters most. (iii) What per-tenant quota commitments exist in current commercial documents? Their absence keeps fairness at "illustrative". (iv) What is the coordinator's own p99 under limiter load, rather than under an idle-benchmark load?

### Interview pack

Each answer states the mechanism first and the operational consequence second, because an answer that begins with a tool name has not yet engaged the question.

**Q1. Why is a token bucket preferred to a fixed-window counter here?**
**Answer.** A fixed window admits up to `2L` across a boundary, so its worst second is twice its nominal limit — precisely the quantity the limiter exists to bound. A token bucket bounds any window of length W by `C + r·W`, which is expressible as a single envelope the backend owner can check.
**Reasoning.** The defect is structural, not statistical: it arises from the discontinuity at the reset, so no amount of tuning L removes it; only a mechanism without the discontinuity does.
**Follow-up.** What if the backend can tolerate a 2× second?
**Acceptable alternative.** If the tolerance is real and documented, a fixed window is legitimate and cheaper to operate.
**Red flag.** Answering "token bucket is more accurate" without naming the boundary mechanism.
**Scoring.** Full credit requires the `2L` statement, the reset mechanism, and a consequence for backend capacity.
**Support.** Section 6.2, FORMULA C-03 and C-04.

**Q2. Set the budget when capacity is 100 requests/s and the tail target is tight.**
**Answer.** Choose ρ_target first. At 95% utilisation the M/D/1 wait is 95 ms; at 85% it is 28.3 ms. Since the model excludes burstiness, the honest budget for a tight tail is 85, accepting 12,360 rejections in the illustrative window rather than 6,960.
**Reasoning.** The wait is `ρ/(2μ(1−ρ))`, so the marginal cost of admitted rate grows as `1/(1−ρ)²`; the last 5% of capacity costs more than the first 80%.
**Follow-up.** What if the business refuses the extra 5,400 rejections?
**Acceptable alternative.** Raise capacity instead of raising r; that converts a rejection problem into a spending decision, which is the correct place for it to live.
**Red flag.** Setting r = μ and calling it "maximum throughput".
**Scoring.** Full credit requires the utilisation argument plus the rejection consequence in the same answer.
**Support.** Section 6.4, table of budgets.

**Q3. How do you make a distributed limit exact?**
**Answer.** Deduct at a single authoritative point and hand out leases. The coordinator subtracts L tokens atomically and returns them; the node spends locally. The sum of issued tokens never exceeds the budget, so no interleaving can over-admit.
**Reasoning.** The property is conservation, not synchronisation: over-admission requires tokens to be created somewhere, and deduction-only leases create none.
**Follow-up.** What does that cost you?
**Acceptable alternative.** A central counter per request, which is exact but puts the coordinator's tail on every decision.
**Red flag.** Claiming exactness from "atomic increment with a short TTL" without stating what the TTL window does at the boundary.
**Scoring.** Full credit names the conservation invariant and the stranded-capacity cost.
**Support.** Section 6.6, FORMULA C-10 and C-11.

**Q4. Why did the lease argument fail at this scale?**
**Answer.** With 85 tokens/s across eight nodes, the stranded-capacity bound `(N−1)L/(rP) < 1` forces `L < rP/(N−1)`. At P = 0.1 s that gives L < 1.21, so a useful lease size requires a coordinator call rate near the request rate — which is the design leases were meant to avoid.
**Reasoning.** The coordinator call rate is `r/L` while the request rate is `r`; the ratio is `1/L`. Only when L is large relative to per-node share does leasing pay, and per-node share here is ~10.6 tokens/s.
**Follow-up.** At what request rate would leases pay off?
**Acceptable alternative.** Keep leases anyway for the bounded-failure property, and justify them on that basis rather than on latency.
**Red flag.** Asserting "leases reduce coordinator load" without the ratio.
**Scoring.** Full credit derives the inequality and states the resulting call rate.
**Support.** Section 6.6.

**Q5. How large should the burst be?**
**Answer.** `C ≤ κ·μ·T_absorb`. With μ = 100, a 24 s absorption budget and κ = 0.10, C ≤ 240. Then check the drain: `C/(μ − r) = 240/15 = 16 s` of quiet to return to the pre-burst queue state.
**Reasoning.** C is a promise that the backend can absorb the whole burst; the drain time is a promise about how often that promise can be called on. Sizing only the first is the common error.
**Follow-up.** A tenant needs to send 500 at once. What now?
**Acceptable alternative.** Give that tenant a separate queue or a scheduled window rather than a larger shared bucket.
**Red flag.** Choosing C as a round number such as 1,000.
**Scoring.** Full credit includes both the absorption bound and the drain check.
**Support.** Section 6.5, FORMULA C-08 and C-09.

**Q6. What happens during a coordinator outage?**
**Answer.** The design switches to a local emergency bucket of `C/N` tokens refilling at `r_eff/N`, for at most 30 s, with an alert at 5 s. Worst case per node is `C/N + (r_eff/N)·30`, which is a stated bound; "fail open" and "fail closed" are not bounds.
**Reasoning.** Both naive policies are self-inflicted failures — one removes the limit, the other denies service — so the only defensible choice is a bounded third behaviour.
**Follow-up.** How do you prevent double spend on recovery?
**Acceptable alternative.** Reject everything during the outage if the backend cannot absorb the emergency rate.
**Red flag.** Answering "fail open, it is safer" without a number.
**Scoring.** Full credit states a numeric bound and the recovery accounting.
**Support.** Sections 8.3 and F-03, F-14.

**Q7. Prove the limiter does not over-admit.**
**Answer.** For the lease path: admissions ≤ tokens issued from the coordinator, and the coordinator never issues more than the budget in any interval, so admissions ≤ budget. For the partitioned path: admissions ≤ `Σ(C/N + (r/N)W) = C + rW` under equal traffic, with deviation `r·δ` from skew.
**Reasoning.** The two paths have different proof obligations; a single sentence cannot cover both, and the partitioned path's bound is conditional on a traffic assumption that must be stated.
**Follow-up.** What invalidates the partitioned bound?
**Acceptable alternative.** Restrict the claim to "bounded when skew is monitored and traffic is balanced".
**Red flag.** Claiming exactness for a design that has no deduction point.
**Scoring.** Full credit separates the two proofs and names the conditional.
**Support.** Sections 6.6, C-12, C-13.

**Q8. How do you test the admission arithmetic without production traffic?**
**Answer.** ORC-01 is a direct simulation of the envelope: feed it arrival timestamps and (C, r), and it returns the admitted subsequence. Compare it with the gateway's decision stream on the same synthetic arrivals; zero disagreements is the pass. Then add the boundary cases: C = 1, r = 0, an empty window, and a clock step.
**Reasoning.** Simulation validates arithmetic and ordering, not realism; it cannot tell you whether the synthetic arrivals resemble customers, so it must be paired with shadow mode.
**Follow-up.** What does the oracle fail to catch?
**Acceptable alternative.** A property test asserting that any window's admissions ≤ `C + rW` for all W.
**Red flag.** Treating a passing oracle as evidence of production readiness.
**Scoring.** Full credit names the limitation explicitly.
**Support.** Section 5e, ORC-01; V01.

**Q9. A tenant sends 60% of traffic. What do you do?**
**Answer.** Two-level buckets with declared shares: the tenant bucket `C_t = 60, r_t = 20` stops that tenant without touching others; the global bucket remains the outer bound. Verify with V08, which asserts others are unaffected.
**Reasoning.** A global-only limiter makes the loudest client the winner by construction, so fairness cannot be added afterwards as a policy; it must exist as separate state.
**Follow-up.** What if the tenant's quota document does not exist?
**Acceptable alternative.** Global-only limit plus an escalation to commercial, and record the fairness gap as OPEN.
**Red flag.** "Just increase the global limit."
**Scoring.** Full credit states the two levels and the missing-authority condition.
**Support.** Sections 6.5, F-04; assumption A-08.

**Q10. Why is the modelled 28.3 ms not a system property?**
**Answer.** It is an M/D/1 result: Poisson arrivals, deterministic service, single server. A token bucket shapes arrivals into bursts, which are worse than Poisson for the downstream queue, and service times are not deterministic. The number is a lower bound under a declared model.
**Reasoning.** Model outputs inherit the model's assumptions; the assumption that fails here is arrival smoothness, and it fails in the direction that makes the real wait larger.
**Follow-up.** Which measurement would replace it?
**Acceptable alternative.** Report the measured p99 with its window and population and drop the model claim entirely.
**Red flag.** Quoting a queue formula as a guarantee.
**Scoring.** Full credit names the violated assumption and the direction of the error.
**Support.** Section 6.4 and the closing paragraph of Section 6.8.

**Q11. How can a limiter make latency look better while making the service worse?**
**Answer.** By measuring latency only over admitted requests. The policy removes the requests that would have queued longest, so the surviving distribution improves while the organisation loses work — coordinated omission.
**Reasoning.** The metric and the policy share a degree of freedom, so the policy can optimise the metric by changing the population rather than the behaviour.
**Follow-up.** What exactly must be on the dashboard?
**Acceptable alternative.** A single end-to-end metric that counts a rejection as a failure with its own latency.
**Red flag.** "We only care about served requests."
**Scoring.** Full credit names the population change and prescribes the paired report (V12).
**Support.** Section 9, V12; F-12.

**Q12. Which change to the inputs would reverse the recommendation?**
**Answer.** Capacity materially above 100 requests/s, or a service-time distribution far from deterministic. Either raises the defensible budget, which lowers the rejection cost and can make unbounded admission with headroom the better choice.
**Reasoning.** The recommendation is downstream of ρ_target and of the wait model; a change to either moves the trade table rather than merely rescaling it.
**Follow-up.** What is the cheapest test that would settle it?
**Acceptable alternative.** Argue from a measured p99 rather than from a model, which is stronger but requires the measurement first.
**Red flag.** Treating the illustrative inputs as facts about the system.
**Scoring.** Full credit names a specific input and the specific consequence.
**Support.** Sections 4 and 7.

**Q13. Is a concurrency limiter a substitute?**
**Answer.** No. A concurrency limiter bounds in-flight work, which bounds queue depth and therefore wait, but it does not bound arrival rate: excess requests wait for a slot rather than being rejected early. It is complementary, and it is the better choice when the backend has a hard in-flight constraint (a small connection pool) rather than a rate constraint.
**Reasoning.** The two mechanisms control different quantities — rate versus occupancy — and Little's law relates them only through the wait, which is the thing both are trying to control.
**Follow-up.** Which do you put first in the pipeline?
**Acceptable alternative.** Rate limit at the edge and concurrency-limit at the backend, which is the usual pairing.
**Red flag.** Conflating "requests per second" with "requests in flight".
**Scoring.** Full credit distinguishes the controlled quantity.
**Support.** Section 5c comparison table; C-07.

**Q14. How would you detect a limiter that is silently not working?**
**Answer.** Multiply the configured per-node limit by the node count and compare with the global budget; assert the product against capacity in a configuration test. Then compare the observed maximum admissions in any 1 s window with `C + rW` — a value above it means either the configuration or the accounting is wrong.
**Reasoning.** A limiter's failure mode is silence: it produces no errors, only the absence of rejections, so detection must come from an invariant check rather than from an alert on errors.
**Follow-up.** What else has the same signature?
**Acceptable alternative.** Alert on "rejections == 0 while admitted > 0.9·capacity", which catches the same defect from the other side.
**Red flag.** Relying on a dashboard that only shows rejections.
**Scoring.** Full credit proposes a checkable invariant.
**Support.** F-07; Section 9 V03.

**Q15. State the invariant you would put in the runbook.**
**Answer.** "For every admitted request there is exactly one admission record whose policy version was in force at arrival, and the count of admissions in any prefix of the ledger never exceeds that version's budget." Everything else — bucket shape, lease size, node count — is an implementation detail of that sentence.
**Reasoning.** The invariant is checkable after the fact by replaying the ledger, which is the only mechanism that survives a disagreement between the dashboard and the customer.
**Follow-up.** What breaks the invariant first?
**Acceptable alternative.** A weaker invariant that excludes the policy-version clause, if versions are immutable by construction.
**Red flag.** An invariant that cannot be tested by replay.
**Scoring.** Full credit states a replay-testable property.
**Support.** Section 5c invariant; V05, V13.

## 15. Quick Reference Card

| Field | Decision card |
|---|---|
| Unit | one request attempt, keyed by `request_id`, with tenant and arrival time |
| Baseline | unbounded admission plus reactive backend timeouts and client retries |
| Rule | per-tenant bucket → deducted global lease → admit or 429; record every decision |
| Invariant | one admission record per admitted request; admissions never exceed the versioned budget in any prefix |
| Envelope | `A_max(W) = C + r·W`; fixed window is `2L` across a boundary and is rejected |
| Budget | `r = ρ_target·μ`; at ρ = 0.85 the modelled wait is 28.3 ms and the illustrative rejection share is 28.6% |
| Burst | `C ≤ κ·μ·T_absorb`; drain `C/(μ − r) = 16 s` at C = 240, r = 85, μ = 100 |
| Distributed | leases conserve the budget (no overshoot, stranded capacity); partitions over-admit by `r·δ` |
| Failure bound | coordinator loss admits at most `C/N + (r_eff/N)·30 s` per node, alert at 5 s |
| Evidence | illustrative inputs, derived arithmetic, no measurement; V01–V15 not executed |
| Escalate | capacity unmeasured, quota authority absent, skew above 250 ms, or rejection share above the owner's tolerance |
