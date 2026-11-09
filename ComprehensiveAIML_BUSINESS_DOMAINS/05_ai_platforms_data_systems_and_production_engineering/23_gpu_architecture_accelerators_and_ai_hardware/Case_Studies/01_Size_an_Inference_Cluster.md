---
case_id: "CASE-2026-023-001"
case_title: "Size an inference cluster under an interactive latency and memory budget"
case_version: "0.1-draft"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "GPU hardware and model serving"
primary_pattern: "resource-admission"
secondary_patterns: ["cost-latency-tradeoff", "correctness-contract"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "platform capacity owner"
source_index: "INDEX.md"
review_trigger: "model, dtype, context length, arrival mix, device, runtime, or SLO change"
---

# Size an inference cluster under an interactive latency and memory budget

## 1. The One-Line Framing

Given an illustrative arrival stream of interactive generation requests, choose a device count and admission policy for a platform capacity owner so that accepted requests have a bounded queueing policy and memory reservation, compared with admitting every request until an out-of-memory failure occurs.

**Evidence declaration:** All workload values, device capacities, and rates in this case are constructed teaching inputs. They are not a measurement of a vendor device, a provider, or an organization. The recommendation is a proposed validation plan, not a deployment result.

The reusable pattern is resource admission: the service must reserve the resources that a request can consume before it promises service. A simple device-count calculation is insufficient because context length, concurrent decode steps, cache fragmentation, cancellation, and failure recovery change the usable capacity.

## 2. Context & Background

An internal platform team operates a text-generation endpoint used by interactive applications. A caller submits a tenant-scoped request with an input context and a maximum output-token limit. The serving plane authenticates the caller, selects an eligible model version, reserves runtime capacity, streams tokens, records billable outcome fields, and releases state on completion or cancellation.

The existing baseline is a single shared queue that accepts every request while device-memory allocation succeeds. It has no per-request cache reservation and exposes only an average latency dashboard. During an arrival burst, a long-context request can consume state needed by many short requests; a retry after a timeout may amplify the queue. The baseline therefore cannot state which requests were knowingly rejected, which were accepted but abandoned, or whether a latency promise was feasible.

The decision owner is the platform capacity owner. Application teams own prompt and output caps; the SRE owns availability telemetry; the security owner owns tenant-scoped authorization. The case deliberately does not assume a particular GPU, inference engine, autoscaler, cloud, or proprietary model.

## 3. The Business Question

| Decision contract | Explicit teaching case definition |
|---|---|
| Question | How many illustrative devices and what admission rule are required to serve a declared peak while preserving a memory and queue safety margin? |
| Decision owner | Platform capacity owner with SRE and finance review |
| Trigger and deadline | A new interactive workload needs an initial capacity proposal before a load test |
| Available actions | Retain accept-all baseline; cap requests only; reserve cache plus queue budget; add devices; defer launch |
| Value and downside | Better completion and predictable waiting versus rejected work, idle headroom, and cost |
| Objective | Minimize rejected useful work subject to an explicit deadline and memory safety constraint |
| Hard constraints | Tenant isolation; output correctness; no cache allocation beyond the admitted budget; cancellation must release reservation |
| Scope | One model version and one illustrative peak interval |
| Out of scope | Pricing negotiation, model quality selection, and production hardware benchmarking |

The primary business outcome is completed useful requests, not tokens per second alone. A policy that raises aggregate tokens per second by allowing long waits or cross-tenant cache reuse can fail the user outcome and the security constraint.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Owner |
|---|---|---|---|---|---|---|---|
| A-01 | Each admitted request has at most 2,048 total resident tokens | workload | capture request traces and enforce cap | unverified | memory reservation is too small | lower cap or reserve by actual bound | application owner |
| A-02 | Cache footprint is 0.75 MiB per resident token | model/runtime | inspect runtime allocation on target build | illustrative | all memory arithmetic changes | measure worst observed footprint | serving engineer |
| A-03 | A device provides 72 GiB usable cache budget after non-cache reservation | hardware | query configured process and allocator | illustrative | device count is wrong | reserve measured headroom | platform owner |
| A-04 | Each device sustains 120 output tokens/s for this mix | performance | closed-loop load test with synchronized timing | illustrative | latency/capacity conclusion reverses | reduce admission or add devices | performance engineer |
| C-01 | No request starts without a tenant-authorized reservation | security | negative authorization tests and trace audit | required | cross-tenant state exposure | fail closed | security owner |
| C-02 | Queue wait budget is 2 seconds after admission | product | written SLO decision | scenario constraint | completion objective changes | revise SLO explicitly | product owner |
| C-03 | At least 20% cache headroom remains unreserved | reliability | admission telemetry | scenario constraint | fragmentation/recovery risk rises | increase devices or lower cap | SRE |

A-02 and A-04 are the assumptions most likely to reverse the device count. They cannot be inferred from nominal hardware specifications because runtime, dtype, model shape, kernel choice, fragmentation, and request mix matter.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is an attempted request identified by an opaque request ID and tenant ID. The required trace fields are arrival timestamp, authorization decision, input-token count, requested output cap, actual output tokens, cancellation timestamp, reservation bytes, queue-enter and first-token timestamps, completion timestamp, error code, model/runtime version, and device/rank identifier.

For the worked trace, use 48 attempted requests in a 10-second interval: 36 short requests with 256 resident tokens and 12 long requests with 2,048 resident tokens. This is illustrative, not sampled data. The trace must preserve event time and monotonic service time separately; substituting processing time for arrival time hides bursts.

### 5b. Tools & Stack

The proposed stack has an authenticated API gateway, an eligibility service, a tenant-scoped admission controller, a scheduler, a model runtime, device metrics, a durable event log, and an outcome meter. The scheduler owns only a lease, not the caller's identity. The runtime owns allocation/release callbacks. The meter records outcome after the terminal event rather than charging an accepted request as completed work.

### 5c. Method Summary

The baseline accepts until allocation fails. Candidate A caps input and output tokens but has no shared reservation. Candidate B reserves worst-case cache and a bounded queue budget before dispatch; it is the proposed method. Candidate C statically partitions all capacity by tenant; it may improve isolation but is not tested and can strand capacity.

| Candidate | Evidence considered | Strength | Limitation | Status |
|---|---|---|---|---|
| Accept-all baseline | conceptual failure analysis | no explicit rejection | OOM and queue amplification | rejected as design |
| Token cap only | considered | bounds one request | does not bound concurrent residency | not tested |
| Reservation admission | derived workload arithmetic | explicit capacity and release contract | can reject useful work | proposed |
| Static tenant partition | considered | simple isolation story | poor pooling under skew | not tested |

### 5d. Provenance, Quality, Exclusions and Leakage

There is no production trace in this illustrative case. The constructed values must never be reported as observed traffic. A real trace must redact content, limit tenant identifiers to access-controlled pseudonyms, and verify that output caps and cancellation signals were available before admission. Do not use completion-time fields to estimate admission eligibility.

### 5e. Reproducibility and Method-Selection Ledger

| Artifact ID | Location | Version/date | Purpose | Accessibility | Limitation |
|---|---|---|---|---|---|
| SRC-01 | NVIDIA CUDA Programming Guide, public documentation | version to pin at implementation | terminology only | public | does not support throughput claim |
| SRC-02 | Roofline paper, 2009, DOI 10.1145/1498765.1498785 | 2009 | bound-model vocabulary | public | not an inference benchmark |
| RUN-01 | proposed synthetic trace | not executed | admission oracle | not created | values are illustrative |

The eventual reproduction bundle must pin model, tokenizer, runtime, compiler, driver, device partitioning, allocator settings, request trace hash, and admission configuration. A seed does not reproduce runtime scheduling.

## 6. Analysis Walkthrough

The decisive calculation is cache capacity, followed by service capacity. Use binary units throughout: 1 GiB = 1,024 MiB. The values are hand-derived and independently recomputable; no runtime test was executed.

**Validity before use:** assume each resident token consumes a constant 0.75 MiB for the chosen model/runtime path, the 72 GiB budget is genuinely available to cache, and the reservation uses a worst-case total token cap. This is a capacity estimate, not a latency guarantee.

```text
FORMULA [C-01]: M_request = T_resident * m_token
WHERE: M_request is cache reservation [MiB]; T_resident is resident tokens [tokens];
       m_token is cache footprint [MiB/token].
CONSTRAINT: fixed footprint model; excludes allocator fragmentation and non-cache state already reserved.
NUMERICAL EXAMPLE: T_resident=2,048 tokens; m_token=0.75 MiB/token;
                   M_request=2,048*0.75=1,536 MiB=1.5 GiB.
INTERPRETATION: one capped long request reserves 1.5 GiB before it starts.
```

The dimensional check is tokens times MiB/token equals MiB. At 256 tokens the same model gives 192 MiB = 0.1875 GiB. If footprint rises to 1 MiB/token, the long request instead reserves 2 GiB; using the lower estimate would over-admit.

```text
FORMULA [C-02]: N_memory = floor(H * M_device / M_request)
WHERE: N_memory [requests] is concurrent capacity; H is allocatable fraction [0,1];
       M_device [GiB] is cache budget; M_request [GiB/request].
CONSTRAINT: homogeneous requests at the selected cap; H accounts for required headroom.
NUMERICAL EXAMPLE: H=0.80; M_device=72 GiB; M_request=1.5 GiB;
                   N_memory=floor(0.80*72/1.5)=floor(38.4)=38 requests.
INTERPRETATION: memory policy admits at most 38 worst-case long requests/device.
```

The 20% headroom leaves 14.4 GiB unreserved. The integer floor is material: admitting 39 requests would reserve 58.5 GiB, still below raw 72 GiB but exceeds the 57.6 GiB policy budget. The policy reserves resilience; it does not claim the remaining bytes are unusable.

For service, suppose a device delivers 120 output tokens/s and a long request may produce 512 output tokens. A fluid upper estimate is 120/512 = 0.234375 long-request completions/s, or one completion every 4.267 seconds if all service is devoted to that class. This cannot be combined with the memory result as a guarantee because prefill, short requests, scheduling, and tail variation were omitted.

A 48-request trace contains 36*256 + 12*2,048 = 33,792 resident-token requests if they all overlap. At 0.75 MiB/token, its instantaneous worst-case reservation is 25,344 MiB = 24.75 GiB. One device has the illustrative memory budget for this snapshot, but service demand may still violate the queue budget. This distinction is why capacity must be checked at both memory and time boundaries.

**Sensitivity:** If `m_token` changes from 0.75 to 0.90 MiB/token, long-request reservation becomes 1,843.2 MiB = 1.8 GiB and `floor(57.6/1.8)=32` requests/device. The decision changes from 38 to 32 worst-case long requests, a 15.8% reduction. If sustained performance falls from 120 to 90 output tokens/s, the 512-token fluid service time increases from 4.267 to 5.689 seconds; adding memory capacity does not correct that service deficit.

## 7. The Finding

For the illustrative inputs, reserve a maximum of 1.5 GiB per 2,048-token request and admit no more than 38 such worst-case reservations per device while retaining 20% cache headroom. This is a proposed safety policy, not an achieved performance result. Device count must then be selected from an observed closed-loop trace that includes queue wait, first-token latency, output rate, cancellation, and fragmentation; the hand calculation only shows that raw memory fit cannot substitute for a latency test. The accept-all baseline is rejected because it cannot explain or bound the failure mode.

## 8. System/Solution Design

```text
caller -> authenticated gateway -> eligibility check -> tenant-scoped admission lease
       -> scheduler -> model runtime -> streamed tokens -> terminal outcome meter
                                      |                     |
                                      v                     v
                               allocation/release      trace + SLO monitor

failed allocation/recovery: runtime failure -> release or expiry fence -> retry decision
```

The gateway validates identity and request schema before any capacity decision. Eligibility checks policy, region, model version, output cap, and tenant authorization. The admission controller atomically records a lease with the resource amount, expiry, tenant, request ID, and model version. The scheduler may dispatch only a live lease. The runtime sends heartbeats and a terminal release; expiry is a recovery backstop, not normal completion.

A successful request path is: authorize → compute reservation → atomically create lease → enqueue → allocate → stream → record terminal status → release lease. A failed allocation path is: runtime reports failure before streaming → scheduler marks dispatch failed → controller releases exactly that lease → caller receives a retryable error only if idempotency policy permits. A lost worker path is: lease expires only after a conservative timeout and reconciliation checks for a terminal runtime event; blindly releasing a live lease can over-admit.

## 9. Evaluation & Validation

| Metric | Definition | Population/window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|
| admission rejection rate | rejected valid requests / attempted valid requests | per tenant and overall, peak trace | baseline | reject reason completeness | proposed |
| queue wait p99 | p99(first dispatch minus admission) [s] | admitted requests | 2-second policy | cancellation rate | proposed |
| TTFT p99 | p99(first token minus arrival) [s] | streaming completions and cancellations separately | baseline | authorization errors | proposed |
| cache headroom | 1 - reserved/cache budget | per device | >=20% target | allocator OOM | proposed |
| successful-work cost | attributable cost / completed valid task | declared interval | alternative policy | failed-work share | not measured |

Validation requires a synthetic trace with fixed seeds and a replay of a redacted real trace if access permits. Run one-device saturation, two-device skew, long-context burst, cancellation storm, allocator fragmentation, worker-loss, and authorization-negative scenarios. Report offered versus accepted load; measuring only completed requests hides a rejecting policy's behavior.

## 10. Action Taken

No deployment action was taken. The proposed next action is to implement the admission ledger behind a disabled feature flag, construct an oracle that sums active reservations by device and tenant, and run closed-loop tests before enabling any caller traffic. Promotion requires meeting the declared guardrails on the selected trace and passing failure/recovery tests; a favorable average throughput figure is insufficient.

## 11. Outcome & Results

**Not measured.** The only result is hand-derived illustrative arithmetic: 1.5 GiB per capped long request and a policy capacity of 38 such requests per 72-GiB device with 20% headroom. It does not establish a device's throughput, latency, cost, allocation behavior, or production reliability. The recorded result should be replaced by a dated test artifact with trace hash, runtime/device versions, metric definitions, uncertainty, and failed-run record.

## 12. Failure Modes & What Went Wrong

| Failure ID | Symptom | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|---|
| F-01 | OOM below reservation cap | cache footprint underestimated; allocator fragmentation; leaked lease | compare active runtime allocations, reservation ledger, and terminal events | lower allocatable fraction; repair release path | no unexplained delta over fault replay |
| F-02 | p99 grows while memory headroom is high | service-rate deficit; scheduler unfairness; prefill/decode interference | slice queue wait and service time by request class | cap long requests or separate schedule | p99 meets target on same offered load |
| F-03 | tenant sees another tenant's cached state | key omission; stale lease; authorization bypass | tenant-negative trace with cache-key audit | bind tenant/policy/model to cache and lease | zero cross-tenant lookup in test |
| F-04 | retries amplify overload | timeout too short; non-idempotent retry; release delayed | correlate retry chain, cancellation, and lease lifetime | bounded retry budget and idempotency key | offered load remains bounded under fault |
| F-05 | capacity appears larger after worker loss | expiry releases live work; metrics lag | inject worker partition and reconcile runtime heartbeat | fence worker epoch before release | no double dispatch in replay |

These are considered failure scenarios, not observed incidents. The distinguishing checks are more valuable than a generic “scale out” response because adding devices may conceal an ownership or retry bug.

## 13. Transferable Lesson

Capacity design begins by naming the resource boundary, owner, reservation lifetime, and rejection behavior. Memory fit, service rate, and end-to-end user latency are separate propositions. A platform can make a defensible initial proposal with illustrative arithmetic, but it must label that proposal honestly and replace it with workload-specific evidence before treating it as a service commitment.

## 14. Connections & Cross-References

Related material: [roofline diagnosis](../Cheat_Sheets/01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md), [H100 resource limits](../Cheat_Sheets/01_H100_Architecture_and_Resource_Limits.md), and the [Module 25 serving index](../../25_model_serving_inference_optimisation_and_cost/INDEX.md). The current source-tree commitment for continuous batching, chunked prefill, paged KV storage, prefix authorization, and speculative decoding remains planned; this case does not claim to cover those implementation-specific mechanisms.

### Interview pack

1. **Why is device memory capacity not a latency guarantee?**
   **Answer:** A request can fit while service rate, queue discipline, prefill/decode interference, or retries still make its wait exceed the deadline. Measure queue and service distributions separately.
   **Follow-up:** What changes if queue wait is unbounded? **Answer:** Capacity becomes a throughput-only calculation; that is unsuitable for an interactive SLO.
   **Acceptable alternative:** A queueing model with validated arrival/service assumptions.
   **Red flag:** “More memory always lowers p99.”
   **Score:** 2 for separate memory and time boundaries; 1 for mentioning queues.

2. **Derive the 1.5-GiB reservation.**
   **Answer:** 2,048 tokens × 0.75 MiB/token = 1,536 MiB = 1.5 GiB.
   **Follow-up:** What if footprint is 0.9 MiB/token? **Answer:** 1,843.2 MiB = 1.8 GiB.
   **Acceptable alternative:** State binary conversion explicitly.
   **Red flag:** Treating MiB and MB as interchangeable.
   **Score:** 2 for arithmetic and units.

3. **Why reserve before dispatch?**
   **Answer:** Reservation turns a shared, failure-prone allocation race into an observable admission decision with an owner and expiry.
   **Follow-up:** How can reservation itself fail? **Answer:** stale or double-released leases; use request IDs, fencing, and reconciliation.
   **Acceptable alternative:** Transactional capacity ledger.
   **Red flag:** “The scheduler will notice eventually.”
   **Score:** 2 for ownership and recovery.

4. **What is a plausible wrong method?**
   **Answer:** Divide total memory by average request size. A burst of long requests can exceed that average even though the arithmetic looks favorable.
   **Follow-up:** When is an average useful? **Answer:** for a separately validated probabilistic policy with explicit tail risk.
   **Acceptable alternative:** percentile-based class reservations.
   **Red flag:** Calling an average a hard limit.
   **Score:** 2 for boundary recognition.

5. **A device has 72 GiB cache budget and 20% headroom. Why not admit 48 1.5-GiB requests?**
   **Answer:** 48×1.5=72 GiB consumes raw budget; the policy allocates only 57.6 GiB, so it permits floor(57.6/1.5)=38.
   **Follow-up:** What does the remainder protect? **Answer:** fragmentation and recovery margin, subject to measurement.
   **Acceptable alternative:** a smaller cap with documented rationale.
   **Red flag:** Ignoring the policy fraction.
   **Score:** 2 for checked calculation.

6. **How would you diagnose rising p99 with free memory?**
   **Answer:** Compare queue wait, prefill time, decode rate, request mix, scheduling fairness, and retry rate by class before adding memory.
   **Follow-up:** Which metric could mislead? **Answer:** average throughput can improve while p99 deteriorates.
   **Acceptable alternative:** trace-level critical path analysis.
   **Red flag:** “It must be an OOM.”
   **Score:** 2 for competing causes.

7. **What changes if long contexts are allowed to 4,096 tokens?**
   **Answer:** Worst-case reservation doubles under the fixed-footprint model, reducing memory concurrency; remeasure footprint and service behavior because the model may no longer hold.
   **Follow-up:** Can output caps offset it? **Answer:** only if residency accounting includes the changed bound and the product contract permits it.
   **Acceptable alternative:** route long contexts to a separate pool.
   **Red flag:** Changing cap without updating admission.
   **Score:** 2 for changed-constraint reasoning.

8. **How should cancellation interact with billing and capacity?**
   **Answer:** cancellation releases runtime state after a terminal/fenced acknowledgment; billing records the actual outcome separately from capacity release.
   **Follow-up:** Why not release immediately on client disconnect? **Answer:** the runtime may still execute or flush; premature release can over-admit.
   **Acceptable alternative:** bounded cancellation confirmation protocol.
   **Red flag:** Equating disconnected client with stopped work.
   **Score:** 2 for state distinction.

9. **What test demonstrates tenant isolation?**
   **Answer:** a negative test submits identical cacheable prefixes under different tenants and verifies distinct authorization-bound keys, traces, and no cross-read.
   **Follow-up:** Is a namespace enough? **Answer:** no; every state lookup and lease must carry the boundary.
   **Acceptable alternative:** cryptographic partition plus audit.
   **Red flag:** relying only on UI identity.
   **Score:** 2 for end-to-end scope.

10. **What should a capacity dashboard show?**
    **Answer:** offered/accepted/rejected requests, reservation bytes, actual allocation, headroom, queue wait, TTFT, output rate, cancellation, retry chains, and release discrepancies, sliced by class and tenant where permitted.
    **Follow-up:** Why show accepted load? **Answer:** completions alone hide admission behavior.
    **Acceptable alternative:** equivalent trace-derived metrics.
    **Red flag:** only GPU utilization.
    **Score:** 2 for complete decision signals.

11. **When does static partitioning win?**
    **Answer:** when isolation or predictable reserved capacity outweighs pooling efficiency and the workload skew is known and acceptable.
    **Follow-up:** What must be measured? **Answer:** stranded capacity, per-tenant SLO, and denial behavior.
    **Acceptable alternative:** weighted quotas with borrowing.
    **Red flag:** claiming universal fairness.
    **Score:** 2 for tradeoff.

12. **What outcome would reverse the recommendation?**
    **Answer:** a measured allocator/runtime footprint, service distribution, or authorization requirement that invalidates worst-case reservation or makes a shared pool unsafe. The correct response is to revise the admission model, not reinterpret the illustrative figures.
    **Follow-up:** Is one favorable load test sufficient? **Answer:** no; include bursts, failure/recovery, and adverse slices.
    **Acceptable alternative:** staged deployment with a halt criterion.
    **Red flag:** treating a single mean benchmark as proof.
    **Score:** 2 for evidence boundary.

### Request-lifecycle trace and oracle

The admission test harness needs an oracle that is intentionally simpler than the runtime. It must not predict GPU throughput. It verifies only the state contract: no device’s active reservation total exceeds its policy budget; a request has at most one live lease; a terminal event releases exactly the matching live lease; and a tenant cannot read or reuse another tenant’s reservation key.

| Step | Event time | Request | Tenant | Operation | Reservation ledger after operation | Expected result |
|---:|---:|---|---|---|---|---|
| 1 | 00:00.000 | A | red | authorize and reserve 1.5 GiB on device 0 | A: live, d0=1.5 GiB | accepted |
| 2 | 00:00.020 | B | blue | authorize and reserve 0.1875 GiB on device 0 | A+B, d0=1.6875 GiB | accepted |
| 3 | 00:00.040 | A | red | retry with same idempotency key | unchanged | return A’s existing admission result |
| 4 | 00:00.100 | C | red | allocate A and begin stream | unchanged | runtime heartbeat associated with A lease |
| 5 | 00:00.500 | B | blue | cancel before dispatch | A only, d0=1.5 GiB | B terminal cancellation and one release |
| 6 | 00:01.000 | D | blue | request a 2,048-token reservation | A+D, d0=3.0 GiB | accepted if policy budget permits |
| 7 | 00:01.200 | A | red | client disconnects | unchanged | begin cancellation; do not immediately release |
| 8 | 00:01.350 | A | red | runtime confirms terminal cancellation | D only, d0=1.5 GiB | fenced release succeeds once |
| 9 | 00:01.360 | E | red | submit long request | D+E, d0=3.0 GiB | may use released capacity |
| 10 | 00:01.400 | stale A worker | red | emits duplicate release | unchanged | rejected as already terminal |

At Step 3, treating a retry as a new request would reserve an additional 1.5 GiB for the same logical operation. That is a state-accounting bug, even if the allocator has spare memory. At Step 7, treating a disconnect as completed work could release capacity while kernels or cleanup still run. The terminal confirmation is therefore both an accounting and a safety boundary.

The minimal ledger key is `(tenant_id, request_id, model_version, policy_version)`. `device_id` is assigned after reservation only if the architecture uses a global pool; when the reservation is device-local, it belongs in the key or immutable lease payload. A key that omits tenant ID makes a cross-tenant collision possible. A key that omits model or policy version may replay a result across incompatible cache semantics.

```text
STATE: PENDING -> RESERVED -> DISPATCHED -> STREAMING -> TERMINAL
PRECONDITION reserve: caller authorized; requested bound validated; no live key exists.
TRANSITION reserve: atomically create one lease and increment one budget counter.
INVARIANT: sum(live lease bytes for device) <= device policy budget.
TRANSITION terminal: atomically mark terminal, decrement the same counter once, retain
                     enough idempotency record to reject duplicate completion/retry.
RECOVERY: on lease expiry, reconcile runtime heartbeat and terminal event before reclaiming.
FAILURE BOUNDARY: no lease protocol proves GPU output correctness; use model-output tests separately.
```

### Capacity experiment design

Run a closed-loop experiment, not an open-loop benchmark that simply sends requests as fast as possible. The client should schedule arrivals from a declared trace and continue waiting for terminal responses. Record the offered rate separately from accepted rate; otherwise an aggressively rejecting controller can appear fast.

| Experiment | Controlled change | Constant inputs | Required observation | Decision rule |
|---|---|---|---|---|
| E1 memory boundary | increase long-context concurrency one request at a time | model, dtype, device, cache policy | reservation versus allocator bytes | stop before unexplained reservation/allocator delta |
| E2 queue boundary | raise offered mixed traffic | same request mix and output caps | queue-wait p50/p95/p99, rejection reasons | reject/scale when p99 violates declared budget |
| E3 cancellation | cancel 10% of requests at fixed stream positions | same concurrency | release latency, duplicate releases, billed terminal status | fail if any live-reservation leak remains |
| E4 tenant isolation | alternate identical prefix requests by tenant | same prompt shape | cache/lease key audit and output access path | fail on any cross-tenant key or lookup |
| E5 worker loss | terminate a worker after dispatch | same ledger TTL and trace | fencing, reclaim delay, duplicate dispatch | fail if a request runs twice without explicit idempotent contract |
| E6 fragmentation | vary request lengths at equal total tokens | same aggregate budget | allocator bytes, admission rejections, OOM | choose headroom from observed worst slice |

E1 cannot establish end-to-end performance because it only measures capacity at one allocation boundary. E2 cannot prove safe recovery because it does not lose a worker. E5 cannot establish model quality because a successful state transition says nothing about generated content. Keeping these claims separate prevents a single convenient benchmark from being misused as proof of unrelated properties.

For an explicit queue example, suppose the scheduler admits at most 30 service units/s during a 20-second burst while valid arrivals occur at 36 requests/s. Under a fluid, lossless approximation, the queue grows by `(36 - 30) * 20 = 120` requests. If the arrival rate subsequently drops to 24 requests/s and service remains 30 requests/s, the same 6 requests/s surplus drains the queue in `120 / 6 = 20` seconds. These values are illustrative; a request is not a fluid unit, and service time varies by context/output length. The calculation says the system cannot meet a two-second queue target merely by waiting for the burst to pass.

The independent arithmetic check is directional: overload adds 6 requests/s for 20 seconds, producing 120; recovery removes 6 requests/s for 20 seconds, removing 120. If recovery arrivals equal service, the denominator is zero and a positive backlog never drains under the model. This provides a crisp admission decision: either temporarily reduce accepted load, add verified service capacity, or acknowledge an SLO violation rather than silently queuing.

## 15. Quick Reference Card

| Field | Reference |
|---|---|
| Decision | admit only requests with an authorized, bounded resource lease |
| Core calculation | `M_request = T_resident * m_token`; `N = floor(H*M_device/M_request)` |
| Critical assumptions | footprint, usable cache budget, headroom, and request caps match runtime behavior |
| Primary metrics | offered/accepted load, queue wait p99, TTFT p99, reservation/actual delta, successful-work cost |
| Do not infer | measured latency, throughput, price, or isolation from the illustrative arithmetic |
| Recovery rule | release only a fenced terminal or expired-and-reconciled lease |
| Evidence/stage/review | illustrative; proposed; draft; not reviewed |
