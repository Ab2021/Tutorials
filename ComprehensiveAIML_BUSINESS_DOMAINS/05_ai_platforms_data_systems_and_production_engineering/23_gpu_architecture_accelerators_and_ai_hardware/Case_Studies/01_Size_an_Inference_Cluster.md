---
case_id: "CASE-2026-023-001"
case_title: "Size an inference cluster under an interactive latency and memory budget"
case_version: "1.0-draft"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "GPU hardware and interactive model serving"
primary_pattern: "resource-admission"
secondary_patterns: ["cost-latency-tradeoff", "correctness-contract"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "platform capacity owner"
source_index: "INDEX.md"
review_trigger: "model, dtype, context/output bounds, runtime, device budget, arrival mix, SLO, or retry policy change"
---

# Size an inference cluster under an interactive latency and memory budget

## 1. The One-Line Framing

Given an illustrative stream of tenant-authorized interactive generation requests, the platform capacity owner must choose device count and an admission policy that preserves a 2-second admitted-queue budget and a cache safety margin, rather than accept work until allocation fails.

The input grain is one attempted request: tenant, model version, input tokens, declared output cap, cancellation/idempotency key, and arrival time.
The output is one of accepted-with-lease, rejected-with-reason, duplicate-of-existing-attempt, or terminal outcome.
The beneficiary is the interactive application user, whose useful result is a completed response before abandonment, not a high aggregate token number.
The baseline is accept-all-until-runtime-allocation-fails with one shared queue and no reservation.
The binding scenario constraint is that admitted work waits no more than two seconds in queue at the stated peak policy boundary.
Evidence declaration: every workload rate, device budget, service rate, and cost in this case is constructed for teaching.
No constructed number is a vendor benchmark, a cloud price, a production trace, or a result from a named organization.
The surprising but ordinary tradeoff is that one device can fit the snapshot in memory while still being unable to honor a queue deadline.
Resource admission is the primary reusable pattern because it makes the decision, resource owner, and rejection reason explicit before runtime state exists.
Cost-latency tradeoff is secondary because added devices buy headroom and change the modeled device-hour cost without proving lower tails.
Correctness contract is secondary because lease state, terminal events, retries, and billing must not confuse an attempt with a completed task.
A raw-memory division is insufficient because weights, runtime workspace, KV cache, fragmentation, prefill, decode, batching, cancellation, and worker recovery occupy different boundaries.

## 2. Context & Background

An internal platform exposes one text-generation model version to several interactive applications.
A caller submits a request with an input context and a maximum output-token limit.
The gateway authenticates the caller before any capacity state is created.
An eligibility check verifies tenant, model, region, and declared bounds.
The admission controller either creates a bounded resource lease or returns a reasoned rejection.
The scheduler sends only a live lease to the runtime.
The runtime performs prefill, then repeated decode steps, and streams output tokens.
A terminal outcome meter records completion, cancellation, failure, or duplicate semantics separately from the lease release.
The baseline has a single shared queue and lets requests enter while a runtime allocation happens to succeed.
That baseline has an average-latency dashboard but no reservation ledger, queue deadline, or per-request reason code.
A long context can therefore occupy cache needed by many short interactions after those short interactions have already been promised service.
A caller timeout can create a retry while the original request is still running, multiplying both queue pressure and billable work.
The platform capacity owner owns the capacity decision and proposed admission policy.
The application owner owns prompt templates and the declared input/output caps.
The serving engineer owns runtime configuration and allocation instrumentation.
The SRE owns availability, loss recovery, and the evidence needed to select headroom.
The security owner owns tenant authorization and the isolation boundary in cache and leases.
Finance reviews a constructed planning cost, but price negotiation is out of scope.
The case concerns one model version, one peak interval, and a proposed pre-launch decision.
It does not select a model, prescribe a GPU vendor, assert a runtime implementation, or claim a deployment.

### Operating vocabulary
- **Weights:** Read-only model parameters loaded for a replica; their bytes are paid once per replica, not once per sequence.
- **Runtime reserve:** Non-KV device memory for executable code, kernels, communication buffers, temporary activations, allocator metadata, and safety slack; it is not safely available to requests.
- **KV cache:** Per-token attention state retained while a sequence remains resident; this case uses a constructed MiB-per-resident-token accounting rate.
- **Token:** One tokenizer output unit; a token count is not a sequence count, a byte count, or a request count.
- **Sequence:** One request’s resident context plus generated tokens until terminal release; it can contain many tokens.
- **Prefill:** The initial processing of input tokens, often a burst of prompt work that may consume different compute and scheduling resources from decode.
- **Decode:** The repeated generation of usually one next token per active sequence; its cadence drives streaming responsiveness.
- **Tensor parallelism (TP):** A single model replica split across ranks that jointly serve the same sequence; TP changes per-rank memory and adds communication, but is not independent replication.
- **Replication:** Multiple independently schedulable model replicas; replicas add capacity for separate sequences but duplicate weights and runtime reserve.
- **TTFT:** Time to first token, measured from arrival or admission only when the denominator is named.
- **Inter-token latency:** Time between streamed tokens after TTFT; it does not measure queue wait before dispatch.
- **Admission:** A promise to reserve bounded resources under a policy, not a promise that a response completes or is correct.

## 3. The Business Question

| Decision contract | Illustrative teaching definition |
|---|---|
| Question | How many constructed devices and what admission rule should be proposed for the declared peak? |
| Decision owner | Platform capacity owner, with SRE, security, application, and finance review. |
| Trigger and deadline | A new interactive workload needs a capacity proposal before a closed-loop load test. |
| Available actions | Retain accept-all; cap only; use bounded admission; add replicas; defer launch. |
| Value and downside | Bounded waiting and failure visibility versus rejected work, idle headroom, and modeled device cost. |
| Objective | Maximize completed valid requests within the stated SLO, not output tokens alone. |
| Hard constraints | Tenant isolation, bounded leases, no policy-budget overcommit, and terminally fenced release. |
| Scope | One model version and one illustrative mixed peak interval. |
| Out of scope | Model quality, production procurement, cloud pricing, and vendor performance comparison. |

A completed valid request is one authorized logical request with exactly one terminal outcome and no cross-tenant state use.
The proxy “tokens per second” can disagree with that objective when it is achieved by letting short requests wait behind large prefills.
The queue guardrail prevents a policy from calling indefinitely delayed work successful merely because the runtime eventually emits tokens.
The decision is bounded: select the policy and minimum modeled replica count for this scenario, then require measured evidence before launch.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Maximum resident tokens per request are 2,048. | workload | Validate input plus output reservation bound at gateway. | illustrative | Reservation can understate cache. | Lower cap or reserve actual declared bound. | application owner |
| A-02 | KV footprint is 0.75 MiB per resident token. | model/runtime | Measure target runtime allocation by length and concurrency. | illustrative | All memory concurrency math changes. | Use measured worst class footprint. | serving engineer |
| A-03 | A device has 72 GiB cache budget after weights and runtime reserve. | hardware | Inspect configured process memory and allocator. | illustrative | Raw device capacity is misused. | Reduce cache budget. | platform owner |
| A-04 | One device provides 120 decode output tokens/s for the declared mix. | performance | Closed-loop trace with synchronized timestamps. | illustrative | Queue/device count conclusion changes. | Lower accepted rate or add replicas. | performance engineer |
| A-05 | One model replica uses TP=2 ranks. | architecture | Measure rank-local weights, cache, and collective behavior. | illustrative | Rank budget or communication behavior differs. | Recompute per-rank policy. | serving engineer |
| A-06 | Peak offered load is 36 valid attempts/s for 20 seconds. | workload | Capture event-time arrivals and retry chains. | illustrative | Queue calculation is not representative. | Use observed peak envelope. | application owner |
| C-01 | No request starts without tenant-authorized reservation. | security | Negative authorization and key-audit tests. | required | Isolation violation. | Fail closed. | security owner |
| C-02 | Admitted queue wait budget is two seconds. | product | Written SLO decision. | scenario constraint | Policy target changes. | Revise SLO explicitly. | product owner |
| C-03 | At least 20% cache headroom remains unreserved. | reliability | Compare ledger with allocator during adverse trace. | scenario constraint | Fragmentation and recovery risk rise. | Increase replicas or headroom. | SRE |
| C-04 | A duplicate logical request cannot create a second live lease. | correctness | Idempotency replay oracle. | required | Over-admission and double charge. | Return original result. | platform owner |
| C-05 | Cancellation does not release until runtime terminal confirmation or reconciled expiry. | correctness | Fault replay with disconnects. | required | Live work can be overcommitted. | Fence and reconcile. | serving engineer |

- A-02, A-03, and A-04 are the assumptions most likely to reverse the selected device count.
- Weights and runtime reserve must be established before treating the residual as cache budget.
- The stated cache budget is per device rank only if the runtime allocation boundary is per rank; otherwise it is a replica-level figure that must be divided consistently.
- A token cap bounds a sequence only when input and reserved output are both included; a request count cap alone does not.
- A two-second queue budget is a policy target, not a percentile derivable from a mean rate without an arrival and service distribution.

## 5. Data, Tools & Method

### 5a. Data Specification

- **Unit of analysis:** One attempted logical request keyed by tenant, idempotency key, model version, and policy version.
- **Window:** Constructed 20-second burst plus drain interval; event time and monotonic service time are separate.
- **Inputs:** Arrival time, input tokens, requested output cap, tenant authorization, and cancellation event.
- **Derived reservation:** Reserved resident tokens and bytes, device/replica choice, deadline, and lease expiry.
- **Outcomes:** Rejected, duplicate, completed, cancelled-after-confirmation, runtime failure, or expired-and-reconciled.
- **Exclusions:** Prompt content, proprietary traces, real identifiers, actual price, and any claimed benchmark.
- **Join semantics:** One logical request may have retry transport attempts; all map to one idempotency key and one terminal outcome.
- **Population slices:** Short, long, cancelled, retried, tenant, replica, and admission reason classes.
- **Correctness oracle:** Sum live lease bytes by policy boundary and verify exactly-once terminal release.

The constructed mix has 36 short requests with 256 resident tokens and 12 long requests with 2,048 resident tokens.
This mix is a concurrency/memory thought experiment, not proof that all 48 arrive or remain resident at the same instant.
Arrival rate is requests per second; decode rate is output tokens per second; neither may be substituted for the other.
A sequence’s resident tokens equal input tokens already prefetched plus generated tokens still retained under the policy, subject to the chosen runtime accounting convention.

### 5b. Tools & Stack

- Authenticated API gateway: validates identity, schema, idempotency key, and declared caps before a lease.
- Eligibility service: evaluates model, region, tenant, and policy-version eligibility.
- Admission controller: atomically records a lease and budget counter at the selected policy boundary.
- Scheduler: chooses an eligible replica/device only from live leases; it owns dispatch, not caller identity.
- Model runtime: loads weights, manages runtime workspace, allocates cache, runs prefill/decode, and emits terminal callbacks.
- Device metrics: collect allocation, fragmentation signals, rank status, and timing without reading prompt content.
- Durable event log: records state transitions sufficient to reconcile a lease after worker loss.
- Outcome meter: records attempted, accepted, completed, cancelled, and failed logical requests separately.
- Test oracle: computes ledger invariants; it intentionally does not predict GPU throughput.

### 5c. Method Summary

Input -> validate declared bounds -> calculate worst-case reservation -> atomically admit or reject -> deadline-aware queue -> runtime allocation -> prefill/decode -> fenced terminal accounting.

| Candidate | Evidence considered | Strength | Limitation | Status | Revisit if |
|---|---|---|---|---|---|
| Accept-all baseline | Conceptual failure analysis | No explicit rejection | OOM and queues are unbounded | rejected as design | No interactive deadline exists |
| Token cap only | Considered | Bounds one request | Does not bound concurrent residency | not tested | Single-flight serving only |
| Bounded admission | Derived arithmetic and state contract | Named capacity/release boundary | May reject useful work | proposed | Measured footprint and queue differ |
| Add replicas without admission | Considered | More service headroom | Still hides overload and retry duplicates | not chosen alone | Admission is already correct |
| Static tenant partition | Considered | Simple isolation narrative | Can strand capacity under skew | not tested | Hard reservations dominate pooling |

### 5d. Provenance, Quality, Exclusions and Leakage

- There is no production trace; constructed values must never be relabeled observed.
- A real trace must redact content and restrict pseudonymous tenant identifiers.
- Admission eligibility may use only data available before reservation; completion and actual output length are unavailable at that moment.
- Using completion time to decide admission leaks future runtime behavior into the policy.
- A retry attempt must not be counted as another user request when it carries the same idempotency key.
- A cancelled request remains attempted and may remain accepted even when it is not completed.
- The source bodies needed for vendor- or runtime-specific claims are unavailable in this case; they remain OPEN.

### 5e. Reproducibility and Method-Selection Ledger

| Artifact ID | Location | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | NVIDIA CUDA Programming Guide | version to pin at implementation | Terminology only | public | Does not establish the constructed rates |
| SRC-02 | Roofline paper, DOI 10.1145/1498765.1498785 | 2009 | Bound-model vocabulary | public | Not an inference benchmark |
| RUN-01 | Proposed synthetic trace | not executed | Admission state oracle | not created | No runtime measurement |
| CFG-01 | Case sections 4 and 6 | 2026-11-10 | Illustrative constants | public manuscript | Not a deployment configuration |

- A future reproduction bundle must pin model revision, tokenizer, dtype, runtime, compiler, driver, TP degree, device partitioning, allocator configuration, trace hash, policy configuration, and time source.
- A random seed does not reproduce scheduling, allocation, networking, or clock behavior.
- The reservation method is selected because it supplies an auditable invariant before a runtime allocation race; it is not selected because it has been benchmarked here.

## 6. Analysis Walkthrough

### 6.1 Establish the memory boundary

A device’s installed memory is not the cache budget.
Weights are replicated per model replica, although TP shards them across its ranks.
Runtime reserve is kept separate because kernels, workspaces, communication, allocator metadata, and safety margin are not request cache.
KV cache grows with resident tokens across sequences.
The constructed per-device cache budget is 72 GiB only after those non-cache categories have been excluded.
Binary conversion is used throughout: 1 GiB = 1,024 MiB.

```text
FORMULA [M-01]: M_cache_budget = M_device - M_weights_rank - M_runtime_rank - M_noncache_margin
WHERE: all M terms are GiB at one explicitly named device/rank boundary.
CONSTRAINT: do not add TP ranks as if they were independent replicas; each sequence spans the required ranks.
NUMERICAL EXAMPLE: this case starts after that accounting with M_cache_budget=72 GiB per policy boundary.
INTERPRETATION: 72 GiB is a constructed residual, not a claim about installed hardware.
```

```text
FORMULA [M-02]: M_request = T_resident * m_token
WHERE: M_request [MiB/sequence]; T_resident [tokens/sequence]; m_token [MiB/token].
CONSTRAINT: fixed-footprint approximation; non-cache bytes already excluded; actual runtime allocation must be measured.
NUMERICAL EXAMPLE: 2,048 tokens/sequence * 0.75 MiB/token = 1,536 MiB/sequence = 1.5 GiB/sequence.
INTERPRETATION: a long sequence reserves 1.5 GiB before it enters the service queue.
```

The dimensions are tokens/sequence times MiB/token, yielding MiB/sequence.
A 256-token short sequence reserves 192 MiB, or 0.1875 GiB.
A sequence is therefore not “one token”: one long sequence holds eight times the constructed cache of one short sequence.
If the footprint rises to 1 MiB/token, the long reservation becomes 2 GiB; that change is material.

### 6.2 Apply headroom and integer admission

The policy does not allocate the entire cache budget.
It reserves 20% as unreserved headroom for fragmentation, drift, recovery, and uncertainty pending measurement.

```text
FORMULA [M-03]: B_policy = H * M_cache_budget
WHERE: B_policy [GiB]; H allocatable fraction [0,1].
CONSTRAINT: H=0.80 leaves 20% unreserved; it is a scenario policy, not a measured allocator guarantee.
NUMERICAL EXAMPLE: B_policy=0.80*72 GiB=57.6 GiB.
INTERPRETATION: active lease totals may not exceed 57.6 GiB at the chosen boundary.
```

```text
FORMULA [M-04]: N_long = floor(B_policy / M_long)
WHERE: N_long [sequences]; M_long [GiB/sequence].
CONSTRAINT: homogeneous long requests; use a mixed-request sum for real admission.
NUMERICAL EXAMPLE: floor(57.6 / 1.5) = floor(38.4) = 38 long sequences.
INTERPRETATION: the policy admits 38, not 39, worst-case long sequences per boundary.
```

Thirty-nine long sequences reserve 58.5 GiB.
That remains below the 72-GiB constructed raw cache residual.
It nevertheless exceeds the 57.6-GiB policy budget, so it is rejected by policy.
The distinction is intentional: raw fit is not permission to consume failure headroom.

### 6.3 Mixed sequences, not average requests

The 48-request illustrative snapshot contains 36 short and 12 long sequences.
Its token sum is 36*256 + 12*2,048 = 9,216 + 24,576 = 33,792 resident tokens.
Its cache sum is 33,792*0.75 MiB = 25,344 MiB = 24.75 GiB.
This snapshot fits the 57.6-GiB policy budget in memory.
It does not show that all 48 can start or finish before any queue deadline.
The average reservation is 24.75/48 = 0.515625 GiB/request, but that average is not a safe per-request admission bound.
A burst composed entirely of long sequences would invalidate an average-based policy.

### 6.4 Prefill and decode have different work shapes

- Prefill processes the input context and often has work proportional to prompt length; it can create a latency spike even when decode capacity looks adequate.
- Decode repeatedly advances active sequences and consumes a share of each scheduling step; increasing concurrency can improve aggregate work while worsening time between tokens.
- A prefill-heavy burst and a decode-heavy steady state can have equal resident-token totals but different TTFT and queue behavior.
- This case therefore keeps a constructed decode rate separate from memory fit and marks prefill performance not measured.
- A policy may reserve output growth at admission even though those output tokens are not yet present; that is output reservation, not a statement that cache is already allocated.
- Chunking, paging, and preemption may change these relationships but are not asserted or measured in this case.

### 6.5 Service capacity is distinct from memory capacity

Assume illustratively that one replica supplies 120 output tokens/s during the declared decode mix.
Assume a long request may produce 512 output tokens.

```text
FORMULA [S-01]: t_decode = O / r_decode
WHERE: t_decode [s/request]; O [output tokens/request]; r_decode [output tokens/s].
CONSTRAINT: fluid single-class estimate; excludes prefill, batching variation, queueing, and token-rate tails.
NUMERICAL EXAMPLE: 512 / 120 = 4.266666... s/request.
INTERPRETATION: all-long serial work completes at about 0.234375 requests/s under this narrow estimate.
```

The denominator is output tokens per second, not requests per second.
The result cannot prove p99 queue wait because no interarrival distribution, service-time distribution, scheduling discipline, or prefill cost is supplied.
It is useful only to reject the false implication that 38 memory-resident long sequences imply 38 prompt completions within two seconds.

### 6.6 Queue boundary and admission rate

For a deliberately simple burst boundary, suppose valid arrivals are 36 requests/s for 20 seconds.
Suppose the scheduler admits at most 30 service units/s during that same period.
The word “unit” is deliberately abstract here; it is not equated to output tokens.

```text
FORMULA [Q-01]: backlog = max(0, lambda - mu) * t
WHERE: backlog [requests]; lambda arrivals [requests/s]; mu accepted service units [requests/s]; t [s].
CONSTRAINT: initially empty fluid queue, constant rates, lossless requests, and no retries; not a percentile model.
NUMERICAL EXAMPLE: (36-30)*20 = 120 requests.
INTERPRETATION: allowing the excess to queue creates 120 pending requests under this approximation.
```

If arrivals then fall to 24 requests/s while service stays 30 requests/s, the spare rate is 6 requests/s.
The fluid drain time is 120/6 = 20 seconds.
A two-second admitted-queue target is already contradicted by this 20-second drain behavior.
If recovery arrivals equal service, the spare-rate denominator is zero and a positive backlog never drains in the model.
If recovery arrivals exceed service, backlog grows; adding a larger queue is not a remedy.
Queue percentiles cannot be deduced from these means because arrivals and service variability are intentionally unspecified.
The appropriate conclusion is a bound on the modeled imbalance, not an invented p95 or p99.

### 6.7 Compare the three decisions

| Option | Admission boundary | Modeled memory result | Queue/deadline result | Cost boundary | Decision |
|---|---|---|---|---|---|
| Accept all | Runtime allocation race | Can consume raw residual until failure | No bounded queue; 120-unit fluid backlog under the scenario | Unpriced failed/retried work | reject |
| Bounded admission, one replica | 57.6-GiB policy budget and deadline gate | 38 long-equivalent limit; mixed snapshot is 24.75 GiB | Reject or defer excess rather than promise unbounded wait | 1 constructed replica-hour rate | propose for measured test |
| Bounded admission, two replicas | Two independent 57.6-GiB budgets | 76 long-equivalent total if traffic is balanced | More modeled service headroom; p99 remains unmeasured | 2 constructed replica-hour rates | consider when test shows one is insufficient |
| Bounded admission, three replicas | Three independent budgets plus failover rule | 114 long-equivalent total before failure reservation | Allows one planned replica to be unavailable only if remaining two meet cap | 3 constructed replica-hour rates | use only after resilience calculation |

Replication adds independently schedulable capacity only when each replica has its own weights, runtime reserve, and admission budget.
TP=2 does not turn two ranks into two replicas; a sequence still needs both TP ranks together.
For failure headroom, N active replicas are not N resilient replicas unless the remaining N-1 capacity can meet the declared admitted load.
A two-replica plan that runs both at the limit has zero device-failure capacity.
A three-replica plan may tolerate one replica loss only after rechecking arrival rate, memory distribution, queue deadline, and rebalance time.

### 6.8 Sensitivity and decision consequences

| Stress | Original | Alternative | Checked change | Consequence |
|---|---|---|---|---|
| KV footprint | 0.75 MiB/token | 0.90 MiB/token | Long reservation=1.8 GiB; floor(57.6/1.8)=32 | One replica loses six long-sequence slots |
| Context/output resident bound | 2,048 tokens | 4,096 tokens | Reservation=3 GiB; floor(57.6/3)=19 | Long-context class needs separate policy or more replicas |
| Decode rate | 120 output tokens/s | 90 output tokens/s | 512/90=5.689 s fluid long decode | Memory addition alone does not repair service deficit |
| Headroom fraction | 20% reserved | 30% reserved | 0.70*72/1.5=33.6 -> 33 | Five fewer long-sequence slots buy larger uncertainty margin |
| One device loss | two active replicas | one remaining replica | remaining long-equivalent capacity=38 | No claim of peak continuity without load shedding |
| Retry factor | one logical request | 1.2 transport attempts/logical request | offered attempt rate becomes 43.2/s from 36/s | Admission and meter must separate attempts from logical work |

The context sensitivity is particularly important because a 4,096-token bound doubles reservation under the fixed-footprint approximation.
The retry sensitivity is not free demand: it can be prevented by idempotency and bounded retry policy, unlike a genuine new request.
The service sensitivity is deliberately incomplete: it gives no p99 estimate and must not be used to publish one.

## 7. The Finding

For the constructed inputs, propose bounded admission with a 1.5-GiB reservation for a 2,048-resident-token long sequence, a 57.6-GiB per-policy-boundary budget, and a 38-long-sequence integer limit.
Reject the accept-all baseline because it cannot identify promised work, limit queue growth, or reconcile retry and cancellation state.
A second replica doubles modeled independent admission capacity only if it is a true replica rather than a second TP rank, but neither one nor two devices has a measured latency claim in this case.
The 36-versus-30 fluid burst demonstrates a 120-request modeled backlog and therefore rules out silently queuing excess work under a two-second objective; it does not calculate queue percentiles.
The decision is proposed, illustrative, and contingent on a closed-loop test that measures target runtime footprint, prefill/decode behavior, retries, cancellations, fragmentation, and failure recovery.

## 8. System/Solution Design

```text
caller -> authenticated gateway -> eligibility -> admission ledger -> deadline queue -> scheduler
                                                                     -> model replica/TP ranks -> stream
runtime terminal event -> fenced release -> outcome meter -> durable trace / SLO monitor
worker loss -> reconciliation of heartbeat + terminal event + epoch fence -> reclaim or hold
```

### 8.1 Boundary and ownership

- The gateway rejects unauthorized or malformed requests before capacity accounting.
- The request validator computes a declared maximum resident-token bound from input tokens plus allowed output growth under the policy.
- The admission controller atomically checks the relevant device or replica budget and writes one lease.
- The lease contains tenant ID, request ID, idempotency key digest, model version, policy version, reservation bytes, chosen boundary, expiry, and worker epoch when dispatched.
- The scheduler may only dispatch a live lease and must preserve the deadline attached at admission.
- The runtime owns actual allocation and emits a heartbeat plus a terminal event; it cannot silently convert a rejected lease into work.
- The outcome meter owns business outcome classification and must not count acceptance as completion.
- The reconciler owns expired-lease recovery and must check fenced evidence before it decrements capacity.

### 8.2 Admission pseudocode and invariant

```text
INPUT: authorized request r with declared bound b, deadline d, idempotency key k
OUTPUT: existing result, rejection reason, or one live lease l
PRECONDITION: b is derived from validated policy caps; r carries tenant/model/policy scope.
1. if terminal_or_live_record(k, tenant, model, policy) exists: return its recorded result.
2. choose an eligible policy boundary with remaining budget >= b and deadline feasibility policy.
3. if none exists: persist rejection(reason=budget_or_deadline) and return it.
4. atomically create l and increment exactly one boundary counter by b.
5. enqueue l with d; scheduler dispatches only while l is live.
INVARIANT: sum(bytes of live leases at boundary) <= boundary policy budget.
TERMINATION: terminal event atomically fences l, decrements exactly once, and retains idempotency outcome.
RECOVERY: expiry is reclaimed only after runtime/worker reconciliation proves no current-epoch execution.
```

### 8.3 Output reservation and queue deadline

- Admission reserves the worst allowed output growth because output tokens may become resident after admission.
- If the product permits a lower output cap, the reservation may be lower only when the cap is immutable for that request.
- A request that exceeds a cap must be terminated or re-admitted under an explicit new contract; it must not silently borrow headroom.
- The queue deadline is checked before dispatch and at scheduling; a request whose remaining budget is impossible to honor is rejected or cancelled with an explicit reason.
- The policy must distinguish “not admitted” from “admitted then cancelled,” because their user and cost denominators differ.
- Batch formation is bounded by live reservations and deadline priority; a full batch is not permission to violate an earlier request’s deadline.
- The scheduler may use separate prefill and decode queues, but that is a proposed variant requiring its own fairness and isolation tests.

### 8.4 Cancellation, retry, and billing accounting

| Event | Capacity state | Logical outcome state | Billing/accounting rule |
|---|---|---|---|
| Gateway rejection | No lease | Rejected | Count attempted valid request; no completed-work credit |
| Duplicate retry before terminal | Unchanged | Existing live or terminal result | Do not create lease or count a new logical request |
| Client disconnect | Lease remains live pending runtime confirmation | Cancellation requested | Do not call completed; record disconnect separately |
| Runtime confirms cancellation | Fenced release once | Cancelled terminal | Record actual emitted work separately from completion |
| Runtime completion | Fenced release once | Completed terminal | One completed logical request only |
| Worker loss | Hold or reconcile lease | Unknown until reconciled | No automatic completion or duplicate retry credit |
| Expiry with proof of no execution | Release once | Expired/retryable if contract allows | Retry uses same logical key and bounded attempt budget |

### 8.5 Failure headroom and modeled cost

Let c be a constructed device-hour planning rate in currency/device-hour; c is deliberately unpriced in this manuscript.
One replica costs c per hour in the planning model, two cost 2c, and three cost 3c before storage, networking, labor, and failed work.
Cost per completed valid task would be (device cost + attributable shared cost + failed-work cost) / completed valid tasks for a named window.
No such numerator or denominator is observed here, so no cost-per-request number is claimed.
If a two-replica cluster must tolerate one loss while peak requires more than one replica’s bounded capacity, it needs load shedding, more replicas, or a revised availability objective.
Headroom is therefore a deliberate cost for uncertainty and recovery, not accidental unused memory.

## 9. Evaluation & Validation

| Metric | Numerator / denominator / units | Population/window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|
| Admission rejection rate | valid rejected logical requests / valid attempted logical requests [fraction] | overall and tenant/class slices per trace | accept-all | reason completeness and authorization correctness | proposed |
| Queue wait p99 | 99th percentile(dispatch - admission) [s] | admitted requests, named trace window | 2-s policy | reported separately from rejected requests | not measured |
| TTFT p99 | 99th percentile(first token - arrival) [s] | streaming completions and cancellations separately | baseline | clock source and cancellation slice | not measured |
| Inter-token latency | percentile(next token - prior token) [s] | streamed sequences | prior runtime configuration | separate from prefill | not measured |
| Cache headroom | 1 - live reserved bytes / policy bytes [fraction] | per boundary over time | 20% policy | allocator/reservation discrepancy | proposed |
| Lease-release discrepancy | live ledger bytes - reconciled actual ownership bytes [bytes] | fault scenarios | zero unexplained discrepancy | no negative/double release | proposed |
| Completed-valid rate | completed valid logical requests / valid attempted logical requests [fraction] | same offered trace | baseline and alternatives | do not hide rejections | not measured |
| Successful-work cost | attributable cost / completed valid logical requests [currency/task] | declared interval | replica options | failed-work share | not measured |

### Validation matrix

| Test | Controlled condition | Required observation | Pass/fail rule | Limitation |
|---|---|---|---|---|
| V-01 memory boundary | Increase long concurrency one lease at a time | Reservations, allocator bytes, fragmentation indicator | Stop on unexplained delta | Does not establish latency |
| V-02 mixed load | Replay event-time short/long trace closed loop | offered, accepted, rejected, queue/TTFT distributions | No silent deadline breach | No real trace exists yet |
| V-03 prefill burst | Hold total tokens but concentrate long inputs | TTFT and decode interference | Choose class policy only from measured result | Constructed workload |
| V-04 cancellation storm | Cancel 10% at declared stream positions | release latency and exactly-once terminal state | No leaked or premature release | Not executed |
| V-05 retry replay | Send duplicate idempotency key before and after terminal | lease count and logical outcomes | One logical result only | Not executed |
| V-06 worker loss | Terminate worker after dispatch | epoch fence, reconciliation, duplicate dispatch | No over-admission or duplicate work | Not executed |
| V-07 tenant negative | Alternate identical prefixes across tenants | key and lookup audit | No cross-tenant state access | Not executed |
| V-08 replica loss | Remove one replica under peak offer | remaining capacity and shedding reasons | Meet stated degraded policy or reject explicitly | Not executed |

Closed-loop means the client schedules declared arrivals and waits for terminal responses; an open-loop flood is not a latency evaluation.
Offered load, accepted load, and completed work must be reported together because a controller can appear fast by rejecting almost everything.
No queue percentile is reported before a trace and timing distribution exist.

## 10. Action Taken

No production deployment action was taken.
The proposed action is to implement the admission ledger behind a disabled feature flag.
The first implementation milestone is a deterministic state oracle for reserve, duplicate, cancel, completion, expiry, and worker-loss transitions.
The second milestone is target-runtime measurement of weights, runtime reserve, and KV allocation under the chosen TP and replica boundary.
The third milestone is a closed-loop trace test that reports the metrics in Section 9, including adverse slices.
Promotion is blocked until the selected policy has explicit rejection semantics, release reconciliation, and measured headroom under fragmentation and failure tests.
A favorable average output-token rate alone is not a promotion criterion.

## 11. Outcome & Results

Outcome not measured.
The only derived result is the illustrative arithmetic: 2,048*0.75 MiB = 1.5 GiB per long reservation; 0.80*72 GiB = 57.6 GiB policy budget; floor(57.6/1.5) = 38 long reservations.
The constructed mixed snapshot reserves 24.75 GiB, which is below the policy budget but says nothing about timing.
The constructed fluid burst produces 120 pending request units when 36 arrivals/s exceed 30 service units/s for 20 seconds.
Those calculations establish arithmetic and a decision boundary under stated assumptions.
They do not establish device throughput, queue percentiles, price, production availability, allocation behavior, isolation, or completed-task cost.
A future result record must name trace hash, runtime/device configuration, distribution window, attempted/accepted/completed denominators, failed runs, and uncertainty.

## 12. Failure Modes & What Went Wrong

| ID | Symptom | Competing causes | Discriminating check | Bounded recovery | Verification |
|---|---|---|---|---|---|
| F-01 | OOM below policy | understated KV; fragmentation; leaked lease | compare ledger, allocator, and terminal events | lower H and fix release | no unexplained delta in replay |
| F-02 | p99 rises with free cache | prefill interference; decode deficit; unfair batch | slice queue, TTFT, ITL by class | deadline queue or class cap | same trace meets guardrail |
| F-03 | long requests crowd out short | average admission; no class limit; output cap drift | inspect reservation distribution | reserve worst bound or class quota | short slice reported |
| F-04 | retry overload | timeout short; duplicate keys not deduped; delayed release | correlate logical key and attempts | bounded retry and idempotency | attempt/logical ratio bounded |
| F-05 | premature capacity release | disconnect mistaken for termination; stale epoch | compare runtime terminal and lease state | fenced terminal acknowledgement | no dispatch on live work |
| F-06 | leaked capacity | terminal event lost; reconciliation gap | aged lease and heartbeat audit | reconcile before expiry reclaim | zero stale leases after recovery |
| F-07 | cross-tenant cache use | tenant omitted from key; authorization bypass | negative tenant trace | bind tenant in every lookup | no cross-read in audit |
| F-08 | TP rank mismatch | per-rank versus replica budget confused | rank-local allocation inventory | name one accounting boundary | budget invariant at all ranks |
| F-09 | device loss causes cascade | no N-1 capacity; synchronized retries | remove replica under offered load | shed load and stagger retries | degraded-policy result explicit |
| F-10 | cost looks artificially low | rejections/failures excluded from denominator | reconcile attempts and terminal states | publish numerator/denominator | cost metric reproducible |
| F-11 | queue deadline is claimed but not met | deadline checked only at admission | trace remaining slack at dispatch | recheck before dispatch | no expired queued work starts |
| F-12 | headroom is consumed silently | policy counter ignores runtime reserve drift | monitor non-cache bytes | reduce policy budget and alert | headroom threshold preserved |

These are considered failure scenarios, not observed incidents.
Adding devices can conceal a duplicate-release or tenant-boundary defect, so recovery and authorization checks precede scale-out conclusions.

## 13. Transferable Lesson

- Capacity planning starts by naming the resource boundary, owner, reservation lifetime, and rejection behavior.
- Weights are per replica, runtime reserve is non-request state, and KV cache is per resident token across sequences; collapsing them creates false capacity.
- TP distributes one replica across ranks and does not create independent request capacity; replication does.
- Memory fit, throughput, per-request latency, and queue percentiles are separate propositions with separate evidence.
- Output reservation is necessary when generated tokens can grow resident state after admission.
- A cancellation signal is not proof that work stopped; capacity release needs a fenced terminal or reconciled expiry.
- Retries are transport attempts until idempotency establishes whether they represent new logical work.
- Failure headroom is a deliberately purchased ability to absorb uncertainty or loss, not a mistake to optimize away.
- An illustrative calculation can define a testable policy boundary but cannot be promoted into a benchmark or deployment claim.

## 14. Connections & Cross-References

Related material: [roofline diagnosis](../Cheat_Sheets/01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md), [H100 resource limits](../Cheat_Sheets/01_H100_Architecture_and_Resource_Limits.md), and the [Module 25 serving index](../../25_model_serving_inference_optimisation_and_cost/INDEX.md).
The source-tree commitments concerning continuous batching, chunked prefill, paged KV storage, prefix authorization, and speculative decoding remain planned and OPEN; this illustrative case does not claim those mechanisms are implemented.

### Interview pack

1. **Why is memory fit not a latency guarantee?**
   **Answer:** Memory says a reservation may coexist; it says nothing about queue delay, prefill, decode service distribution, scheduler fairness, or retries.
   **Reasoning:** Memory says a reservation may coexist; it says nothing about queue delay, prefill, decode service distribution, scheduler fairness, or retries.
   **Follow-up:** Which timing components would you measure?
   **Follow-up answer:** Queue wait, TTFT, prefill duration, and inter-token latency by request class.
   **Acceptable alternative:** A validated queueing model with explicit distributions.
   **Red flag:** “More memory always lowers p99.”
   **Scoring:** 2: separates memory from time; 1: mentions queue.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

2. **Derive the long-request reservation.**
   **Answer:** 2,048 resident tokens times 0.75 MiB/token equals 1,536 MiB, which is 1.5 GiB using 1,024 MiB/GiB.
   **Reasoning:** 2,048 resident tokens times 0.75 MiB/token equals 1,536 MiB, which is 1.5 GiB using 1,024 MiB/GiB.
   **Follow-up:** What is it at 0.90 MiB/token?
   **Follow-up answer:** 1,843.2 MiB, or 1.8 GiB.
   **Acceptable alternative:** Equivalent binary-unit derivation.
   **Red flag:** Treating MB and MiB as identical.
   **Scoring:** 2: arithmetic and units; 1: correct product only.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

3. **Why are tokens and sequences different denominators?**
   **Answer:** A sequence is one request lifecycle containing many resident tokens; cache grows with tokens while concurrency is counted in sequences.
   **Reasoning:** A sequence is one request lifecycle containing many resident tokens; cache grows with tokens while concurrency is counted in sequences.
   **Follow-up:** Can 48 sequences have equal cache?
   **Follow-up answer:** No; their token lengths determine the sum.
   **Acceptable alternative:** Explicit per-sequence token vector.
   **Red flag:** Using request count as cache bytes.
   **Scoring:** 2: names both denominators; 1: says lengths vary.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

4. **What is the TP versus replication distinction?**
   **Answer:** TP ranks jointly serve one model replica and one sequence; replicas independently schedule separate sequences and duplicate non-sharded state.
   **Reasoning:** TP ranks jointly serve one model replica and one sequence; replicas independently schedule separate sequences and duplicate non-sharded state.
   **Follow-up:** Does TP=2 double admission slots?
   **Follow-up answer:** No; it changes rank-local allocation and communication.
   **Acceptable alternative:** A precisely defined alternative parallelism scheme.
   **Red flag:** Counting ranks as replicas.
   **Scoring:** 2: shared-sequence consequence; 1: says they differ.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

5. **Why reserve output before it exists?**
   **Answer:** Future generated tokens can grow resident state, so the policy reserves the declared worst permitted growth before promising queue service.
   **Reasoning:** Future generated tokens can grow resident state, so the policy reserves the declared worst permitted growth before promising queue service.
   **Follow-up:** When can the reservation shrink?
   **Follow-up answer:** Only under an immutable lower output contract or fenced re-admission.
   **Acceptable alternative:** Measured runtime policy with safe dynamic resizing.
   **Red flag:** Borrowing headroom silently.
   **Scoring:** 2: future growth plus contract; 1: mentions cap.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

6. **Show the headroom calculation.**
   **Answer:** 0.80 times 72 GiB is 57.6 GiB; floor(57.6/1.5) is 38, while 39 would reserve 58.5 GiB.
   **Reasoning:** 0.80 times 72 GiB is 57.6 GiB; floor(57.6/1.5) is 38, while 39 would reserve 58.5 GiB.
   **Follow-up:** Why not use raw 72 GiB?
   **Follow-up answer:** The 20% policy margin is reserved for uncertainty, fragmentation, and recovery.
   **Acceptable alternative:** Different measured headroom with evidence.
   **Red flag:** Calling 39 permissible because raw bytes fit.
   **Scoring:** 2: floor and policy distinction; 1: states 38.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

7. **What does the 36 versus 30 burst establish?**
   **Answer:** Under a constant fluid approximation, excess is 6 requests/s for 20 seconds, so backlog is 120 requests.
   **Reasoning:** Under a constant fluid approximation, excess is 6 requests/s for 20 seconds, so backlog is 120 requests.
   **Follow-up:** What p99 follows?
   **Follow-up answer:** None; distributions and discipline are unspecified.
   **Acceptable alternative:** A validated stochastic model.
   **Red flag:** Inventing a percentile from means.
   **Scoring:** 2: states bound and limit; 1: arithmetic only.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

8. **How do prefill and decode change scheduling?**
   **Answer:** Prefill concentrates input work and affects TTFT; decode repeatedly services active sequences and affects stream cadence, so equal token totals can have different latency.
   **Reasoning:** Prefill concentrates input work and affects TTFT; decode repeatedly services active sequences and affects stream cadence, so equal token totals can have different latency.
   **Follow-up:** What experiment isolates this?
   **Follow-up answer:** Hold total token volume and vary prompt versus output composition.
   **Acceptable alternative:** Trace-based critical-path analysis.
   **Red flag:** One aggregate rate explains both.
   **Scoring:** 2: distinct mechanisms; 1: names both.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

9. **What is the correct cancellation release rule?**
   **Answer:** Keep the lease until a fenced runtime terminal confirmation or reconciled expiry proves current-epoch work stopped; then release exactly once.
   **Reasoning:** Keep the lease until a fenced runtime terminal confirmation or reconciled expiry proves current-epoch work stopped; then release exactly once.
   **Follow-up:** Why not release on disconnect?
   **Follow-up answer:** The runtime may still execute or flush.
   **Acceptable alternative:** A bounded cancellation handshake.
   **Red flag:** Disconnect equals completion.
   **Scoring:** 2: fence and exactly-once; 1: delayed release.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

10. **How should retries be counted?**
   **Answer:** Transport attempts are grouped by idempotency key into one logical request; a duplicate returns existing state and creates no second lease.
   **Reasoning:** Transport attempts are grouped by idempotency key into one logical request; a duplicate returns existing state and creates no second lease.
   **Follow-up:** What if expiry permits retry?
   **Follow-up answer:** Reuse the logical key with a bounded attempt budget and explicit terminal/retry state.
   **Acceptable alternative:** Durable request ledger.
   **Red flag:** Counting every retry as completed work.
   **Scoring:** 2: logical/attempt separation; 1: idempotency mention.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

11. **How do you test tenant isolation?**
   **Answer:** Submit structurally identical cacheable prefixes under distinct tenants and audit every lease/cache lookup key and authorization decision.
   **Reasoning:** Submit structurally identical cacheable prefixes under distinct tenants and audit every lease/cache lookup key and authorization decision.
   **Follow-up:** Is a UI namespace sufficient?
   **Follow-up answer:** No; the boundary must bind every stateful lookup and dispatch path.
   **Acceptable alternative:** Cryptographic partition plus audit.
   **Red flag:** Relying on application display identity.
   **Scoring:** 2: end-to-end key audit; 1: negative test.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

12. **What fails under one replica loss?**
   **Answer:** Two replicas at their normal limit may leave only one 38-long-equivalent boundary after loss, so peak continuity is not implied.
   **Reasoning:** Two replicas at their normal limit may leave only one 38-long-equivalent boundary after loss, so peak continuity is not implied.
   **Follow-up:** What is the recovery choice?
   **Follow-up answer:** Load shed, provision N+1 capacity, or revise the degraded SLO.
   **Acceptable alternative:** Warm standby with measured failover.
   **Red flag:** “Two devices are automatically highly available.”
   **Scoring:** 2: remaining-capacity calculation; 1: names load shedding.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

13. **Why can successful-work cost be misleading?**
   **Answer:** A denominator of completed requests hides rejected, cancelled, retried, and failed work unless the metric publishes its population and shared-cost attribution.
   **Reasoning:** A denominator of completed requests hides rejected, cancelled, retried, and failed work unless the metric publishes its population and shared-cost attribution.
   **Follow-up:** What is required?
   **Follow-up answer:** Named numerator, denominator, window, and treatment of failed work.
   **Acceptable alternative:** Separate unit economics by outcome class.
   **Red flag:** Divide device-hours by only happy-path tokens.
   **Scoring:** 2: denominator boundary; 1: says include failures.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

14. **A new 4,096-token contract arrives. What changes?**
   **Answer:** At fixed 0.75 MiB/token, the reservation is 3 GiB and floor(57.6/3) is 19; remeasure because the footprint model may change.
   **Reasoning:** At fixed 0.75 MiB/token, the reservation is 3 GiB and floor(57.6/3) is 19; remeasure because the footprint model may change.
   **Follow-up:** Can output caps offset it?
   **Follow-up answer:** Only if the resident bound and product contract genuinely change.
   **Acceptable alternative:** Route a separately measured long-context class.
   **Red flag:** Change the cap without changing admission.
   **Scoring:** 2: recalculation and evidence boundary; 1: says capacity falls.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

15. **What result reverses the recommendation?**
   **Answer:** Measured target runtime evidence showing a different cache footprint, non-cache reserve, latency distribution, isolation requirement, or recovery behavior can invalidate the proposed policy.
   **Reasoning:** Measured target runtime evidence showing a different cache footprint, non-cache reserve, latency distribution, isolation requirement, or recovery behavior can invalidate the proposed policy.
   **Follow-up:** Is one favorable mean benchmark enough?
   **Follow-up answer:** No; test mixed bursts, cancellation, fragmentation, and failure recovery.
   **Acceptable alternative:** Staged enablement with halt criteria.
   **Red flag:** Reinterpreting illustrative inputs as proof.
   **Scoring:** 2: names invalidator and test scope; 1: says measure first.
   **Answer support:** Sections 4, 6, 8, 9, and 12 define the relevant boundary and limitation.

### Request-lifecycle trace and oracle

| Step | Event | Ledger state after event | Expected oracle result |
|---:|---|---|---|
| 1 | A/red authorizes and reserves 1.5 GiB on boundary 0 | A live; b0=1.5 GiB | accepted |
| 2 | B/blue reserves 0.1875 GiB | A+B live; b0=1.6875 GiB | accepted |
| 3 | A retries same idempotency key | unchanged | return A existing result |
| 4 | A dispatches and runtime heartbeats | unchanged | one live lease |
| 5 | B cancels before dispatch | A live; b0=1.5 GiB | one terminal release |
| 6 | D/blue reserves 1.5 GiB | A+D live; b0=3.0 GiB | accepted if budget permits |
| 7 | A client disconnects | unchanged | cancellation requested, no release |
| 8 | A runtime confirms terminal cancellation | D live; b0=1.5 GiB | fenced release once |
| 9 | E/red reserves 1.5 GiB | D+E live; b0=3.0 GiB | uses released capacity |
| 10 | stale A worker emits duplicate release | unchanged | reject duplicate terminal |

Oracle invariant one: at every step, live lease bytes for boundary 0 are no greater than 57.6 GiB.
Oracle invariant two: a logical idempotency key has at most one live lease.
Oracle invariant three: terminal transition decrements the matching live lease exactly once.
Oracle invariant four: a tenant-scoped key cannot resolve a cache or lease owned by a different tenant.

### Capacity decision drills
Each drill below is a bounded review scenario, not an additional interview question or observed incident.
It connects a state boundary to the evidence required before a capacity conclusion is safe.

#### D-01 input cap drift
- **Trigger:** A gateway accepts a 1,900-token input and later permits a 512-token output under a 2,048 resident cap.
- **Decision boundary:** The declared maximum is 2,412 tokens, so admission must reject or alter the contract before lease creation.
- **Required evidence:** Validated input count, output cap, and policy version.
- **Do not infer:** A later output clamp is not evidence that the original reservation was valid.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-02 tokenizer mismatch
- **Trigger:** The client’s token estimate is lower than the runtime tokenizer count.
- **Decision boundary:** The runtime-authoritative tokenizer count must drive the reservation before dispatch.
- **Required evidence:** Model revision, tokenizer revision, and counted token total.
- **Do not infer:** Character count is not a safe substitute for token count.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-03 output-cap mutation
- **Trigger:** A caller attempts to raise output cap after it receives a lease.
- **Decision boundary:** Reject the mutation or fenced re-admit with a larger reservation; never mutate budget in place.
- **Required evidence:** Lease version, original cap, requested cap, and counter delta.
- **Do not infer:** A mutable cap makes the original admission guarantee false.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-04 short-to-long burst
- **Trigger:** Thirty-six short reservations are followed by a burst of long reservations.
- **Decision boundary:** The admission ledger sums each actual reservation; it does not rely on the earlier average.
- **Required evidence:** Token-class histogram and live-byte total before each decision.
- **Do not infer:** An average-request policy can accept a dangerous long tail.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-05 all-long burst
- **Trigger:** Only 2,048-token requests arrive until the policy boundary is full.
- **Decision boundary:** Admission stops at 38 long reservations per 57.6-GiB boundary under the constructed constants.
- **Required evidence:** Integer lease count and live bytes.
- **Do not infer:** This checks memory policy only, not whether 38 streams meet latency.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-06 prefill convoy
- **Trigger:** Several long prompts enter ahead of short decode-sensitive requests.
- **Decision boundary:** Measure prefill occupancy and queue delay by class before introducing class prioritization.
- **Required evidence:** Prefill start/end times, TTFT, and queue order.
- **Do not infer:** A decode-rate average cannot diagnose a prefill convoy.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-07 decode crowding
- **Trigger:** Many active sequences each receive small decode slices and inter-token gaps rise.
- **Decision boundary:** Bound active decode concurrency or use a measured scheduler policy; do not call memory headroom a cure.
- **Required evidence:** Active sequence count and inter-token latency distribution.
- **Do not infer:** Aggregate output rate can rise while an individual stream degrades.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-08 batching deadline
- **Trigger:** Waiting for a fuller batch consumes an admitted request’s remaining two-second budget.
- **Decision boundary:** Dispatch or reject based on remaining slack rather than batch fullness alone.
- **Required evidence:** Admission time, dispatch time, deadline, and batch decision.
- **Do not infer:** Batch efficiency is not an override for a stated SLO.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-09 queue-before-reserve
- **Trigger:** A design queues requests before creating a lease.
- **Decision boundary:** It must either reserve at queue entry or label queued work unadmitted; otherwise it promises unconstrained future capacity.
- **Required evidence:** Queue state, lease creation timestamp, and rejection reason.
- **Do not infer:** Queue length is not a capacity reservation.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-10 reservation-before-authorize
- **Trigger:** A design increments a counter before checking tenant authorization.
- **Decision boundary:** Authorization must precede reservation so rejected callers cannot consume or probe capacity.
- **Required evidence:** Authorization decision timestamp and lease audit record.
- **Do not infer:** Later release does not repair a capacity side channel.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-11 duplicate-before-dispatch
- **Trigger:** A transport retry arrives while the original lease is queued.
- **Decision boundary:** Return the original admission result and preserve one lease under the logical idempotency key.
- **Required evidence:** Logical key, lease ID, and counter history.
- **Do not infer:** Two queued attempts are not two user requests.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-12 duplicate-after-completion
- **Trigger:** A retry arrives after a durable completed terminal record.
- **Decision boundary:** Return the retained terminal result according to contract; do not regenerate work.
- **Required evidence:** Terminal record retention and idempotency lookup.
- **Do not infer:** Retention duration is a policy choice that must be explicit.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-13 disconnect-during-prefill
- **Trigger:** The client disconnects while prefill is still executing.
- **Decision boundary:** Record cancellation requested, signal runtime, and keep bytes reserved until terminal confirmation.
- **Required evidence:** Disconnect, runtime cancellation acknowledgment, and terminal epoch.
- **Do not infer:** A socket close is not proof device work ended.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-14 disconnect-after-output
- **Trigger:** The client disconnects after some tokens were emitted but before terminal completion.
- **Decision boundary:** Classify partial output and capacity release separately from completed-valid outcome.
- **Required evidence:** Emitted-token count, terminal reason, and billing rule.
- **Do not infer:** Partial work cannot silently become a full completion.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-15 lost-terminal-event
- **Trigger:** A worker completed but its terminal callback was not persisted.
- **Decision boundary:** Reconcile durable runtime evidence before expiry reclaim, then create one terminal state.
- **Required evidence:** Worker epoch, heartbeat, output log, and lease state.
- **Do not infer:** A timeout alone cannot distinguish slow from lost.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-16 stale-worker-release
- **Trigger:** A restarted worker emits a release for an old lease epoch.
- **Decision boundary:** Reject the stale epoch so it cannot decrement current capacity.
- **Required evidence:** Lease ID, worker epoch, and fence value.
- **Do not infer:** Lease ID without epoch is insufficient after restart.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-17 allocator-fragmentation
- **Trigger:** Allocator bytes approach failure while summed reservations remain below policy.
- **Decision boundary:** Lower allocatable fraction only after correlating fragmentation with allocation patterns and runtime state.
- **Required evidence:** Reserved bytes, allocated bytes, free-block distribution, and request lengths.
- **Do not infer:** Do not claim a fragmentation percentage without measurement.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-18 runtime-reserve-growth
- **Trigger:** A compiler/runtime change raises non-cache workspace use.
- **Decision boundary:** Reduce cache budget and rerun the full capacity arithmetic before increasing admission.
- **Required evidence:** Versioned non-cache memory inventory.
- **Do not infer:** KV measurements alone miss runtime-reserve drift.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-19 TP-rank-asymmetry
- **Trigger:** One TP rank reports more allocation than another.
- **Decision boundary:** The policy boundary must use the constraining rank or explicitly model rank-local budgets.
- **Required evidence:** Per-rank weights, workspace, cache, and communication buffers.
- **Do not infer:** A replica-wide average can overrun one rank.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-20 replica-imbalance
- **Trigger:** One replica has free budget while another rejects.
- **Decision boundary:** Use an explicit routing policy and report skew; do not aggregate bytes and assume immediate portability.
- **Required evidence:** Per-replica live bytes, queue age, and route decisions.
- **Do not infer:** State locality and TP grouping may prevent instant migration.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-21 device-loss
- **Trigger:** A replica becomes unavailable after dispatch.
- **Decision boundary:** Fence the old worker, reconcile in-flight leases, and load shed or reroute only under a new valid reservation.
- **Required evidence:** Health epoch, lease ownership, and remaining capacity.
- **Do not infer:** Failover cannot create N-1 capacity from a saturated plan.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-22 correlated-loss
- **Trigger:** Two replicas share a failure domain.
- **Decision boundary:** Capacity planning must name the failure domain rather than call device count availability.
- **Required evidence:** Rack/host/power/network placement and outage assumptions.
- **Do not infer:** Independent replica arithmetic does not prove independent failures.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-23 retry-storm
- **Trigger:** Timeouts cause 1.2 attempts per logical request.
- **Decision boundary:** The meter reports 43.2 attempts/s from 36 logical requests/s and admission deduplicates known keys.
- **Required evidence:** Attempt-to-logical ratio and retry reason chain.
- **Do not infer:** New idempotency keys for retries defeat deduplication.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-24 deadline-expiry
- **Trigger:** A queued lease reaches its deadline before dispatch.
- **Decision boundary:** Terminally reject or cancel it and release once; never start known-expired interactive work without a new contract.
- **Required evidence:** Deadline, dispatch timestamp, terminal reason, and release event.
- **Do not infer:** Completion after expiry cannot retroactively meet the SLO.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-25 policy-version-rollover
- **Trigger:** A policy changes headroom from 20% to 30% while old leases exist.
- **Decision boundary:** Keep each lease bound to its original policy version and apply new admission against a clearly reconciled budget.
- **Required evidence:** Policy version on lease and budget transition audit.
- **Do not infer:** Reinterpreting old leases under new math can double-count or undercount.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-26 model-version-rollover
- **Trigger:** A model update changes weights or KV footprint.
- **Decision boundary:** Treat it as a separate capacity pool until measured; do not reuse the old 72-GiB cache residual.
- **Required evidence:** Model revision, runtime revision, and allocation inventory.
- **Do not infer:** Same model name is not evidence of same memory shape.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-27 tenant-quota
- **Trigger:** One tenant consumes all pooled leases with valid traffic.
- **Decision boundary:** Apply an explicit tenant quota or weighted borrowing policy and report denied capacity separately.
- **Required evidence:** Tenant live bytes, quota, borrow state, and rejection reason.
- **Do not infer:** Global fairness is not implied by first-come-first-served.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-28 prefix-cache-boundary
- **Trigger:** A cacheable prefix is reused across requests.
- **Decision boundary:** Every reuse key must include tenant/policy/model authorization boundary or be disabled across tenants.
- **Required evidence:** Cache key fields, authorization result, and lookup audit.
- **Do not infer:** Identical text does not authorize shared state.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-29 cost-denominator
- **Trigger:** Finance asks for cost per completed request after a rejection-heavy test.
- **Decision boundary:** Publish attempts, accepted, completed, cancelled, failed, device-hours, and allocation rule before computing a ratio.
- **Required evidence:** Named numerator, denominator, window, and outcome classes.
- **Do not infer:** A happy-path denominator rewards indiscriminate rejection.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-30 scale-out-delay
- **Trigger:** An autoscaler adds a replica after a burst begins.
- **Decision boundary:** Admission still protects the current pool; future capacity cannot justify current overcommit.
- **Required evidence:** Replica-ready time, queue age, and pre-scale decisions.
- **Do not infer:** Autoscaling delay is part of the latency path.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-31 warmup
- **Trigger:** A newly loaded replica has different first-request latency and reserve.
- **Decision boundary:** Exclude it from steady-state capacity until warmup behavior is measured and tagged.
- **Required evidence:** Load timestamp, first-request timings, and memory inventory.
- **Do not infer:** A warmed benchmark does not prove cold-start readiness.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-32 clock-skew
- **Trigger:** Gateway and runtime clocks disagree.
- **Decision boundary:** Use a monotonic correlation method or declare which components cannot be compared.
- **Required evidence:** Clock source, offsets, and event ordering.
- **Do not infer:** Negative queue time is a telemetry defect, not low latency.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-33 metric-censoring
- **Trigger:** Cancelled requests are removed from a TTFT chart.
- **Decision boundary:** Report them as a distinct population and explain whether first token occurred before cancellation.
- **Required evidence:** Cancellation timestamps, first-token timestamps, and inclusion rule.
- **Do not infer:** Dropping hard cases can improve a percentile without improving users.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

#### D-34 source-boundary
- **Trigger:** A reader asks whether the constructed 120-token/s value matches a product.
- **Decision boundary:** State that it is illustrative and require a versioned closed-loop target measurement.
- **Required evidence:** Evidence label, runtime configuration, and source availability.
- **Do not infer:** No external benchmark is supported by this case.
- **Owner:** The platform capacity owner owns admission policy; the serving engineer supplies runtime evidence; the SRE validates recovery evidence.
- **Accounting:** Preserve attempted, admitted, terminal, and released states as separate fields for this scenario.
- **Safety check:** The live-byte counter must remain at or below its versioned policy budget throughout the trace.
- **Latency check:** Report queue wait, TTFT, and inter-token latency only for their named population and window.
- **Recovery check:** Any terminal transition or reclaim must be idempotent and fenced against stale worker state.

## 15. Quick Reference Card

| Field | Reference |
|---|---|
| Decision | Propose authorized bounded admission; reject accept-all allocation races. |
| Core calculation | M_request=T_resident*m_token; B_policy=H*M_cache_budget; N=floor(B_policy/M_request). |
| Worked values | 2,048*0.75 MiB=1.5 GiB; 0.80*72=57.6 GiB; floor(57.6/1.5)=38. |
| Token/sequence rule | Cache is summed over resident tokens across sequences; request count is not a byte budget. |
| TP/replica rule | TP ranks jointly serve one replica; independent replicas add schedulable capacity. |
| Latency rule | Memory fit and mean rates do not establish per-request latency or queue percentiles. |
| Queue rule | At 36 versus 30 units/s for 20 s, fluid backlog is 120; no p99 follows. |
| Release rule | Release only a fenced terminal or expired-and-reconciled lease; dedupe retries by logical key. |
| Failure headroom | Leave the policy margin unreserved and prove N-1 behavior separately. |
| Evidence/stage/review | illustrative; proposed; draft; not reviewed; runtime/source evidence OPEN. |
