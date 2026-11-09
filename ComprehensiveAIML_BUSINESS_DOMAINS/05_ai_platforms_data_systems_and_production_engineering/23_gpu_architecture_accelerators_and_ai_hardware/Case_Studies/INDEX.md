# Module 23 Case Studies — GPU Architecture, Accelerators and AI Hardware

**Collection status:** planning catalog; no `CASE-*` artifact is authored.  
**Contract:** [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). A future case needs all 15 sections, a checked calculation, 12–18 answered interview questions, and a final Section 15 card.

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | GPU architecture, accelerators, and AI hardware |
| Canonical case types | `streaming-systems`, `incident`, or genuinely integrated `hybrid` |
| Evidence boundary | Illustrative/proposed unless a case cites an accessible primary source; no vendor-workload result is implied |
| Source limits | The visible source tree names topics, but original lesson bodies are unavailable |
| Editorial state | planned; review when device, compiler, workload, or measured constraints change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-023-001 | [Size an inference cluster](01_Size_an_Inference_Cluster.md) | hardware | hybrid | select capacity and admission policy | resource-admission | cost-latency-tradeoff | illustrative | proposed | — | draft; workload trace and device measurements remain required |
| PLAN-23-02 | Profile memory-bound training | hardware | incident | choose the bottleneck-removal change | resource-accounting | correctness-contract | not-measured | proposed | — | needs profiler and numerical-parity evidence |
| PLAN-2026-IR007 | Balance interactive inference latency and admitted throughput | hardware/serving | hybrid | scheduling and KV-memory policy | resource-admission | cost-latency-tradeoff; authorization-scope | not-measured | proposed | 2026-11-06 (planning) | reference only; geometry and speedups unmeasured |

## 3. Canonical reusable patterns

- **Resource admission:** accept work only when a defined memory, compute, and deadline budget remains; test rejection and recovery, not just average throughput.
- **Resource accounting:** account for a named boundary (registers, shared memory, HBM, host link, or network) with units and ownership.
- **Cost–latency tradeoff:** compare successful-request cost, tail latency, and capacity headroom under the same workload.
- **Correctness contract:** require numerical or target-equivalence checks before adopting a performance change.

## 4. Similarity / contrast map

| Case | Shares | Contrasts with | Transfer boundary |
|---|---|---|---|
| PLAN-23-01 | admission and capacity reasoning with serving cases | training has long-lived state rather than interactive deadlines | device-count arithmetic is not achieved latency |
| PLAN-23-02 | profiling and parity validation with distributed-training cases | hardware counters do not explain data/label quality | optimize only the measured limiting boundary |
| PLAN-2026-IR007 | batching and KV accounting with Module 25 | authorization scope is a serving concern | speculative decoding can be slower |

## 5. Questions and future-case backlog

1. What device, dtype, shapes, and traffic boundary make a roofline calculation relevant?
2. Which trace captures peak concurrency, arrivals, cancellation, and KV residency?
3. What numerical invariant detects an invalid fused or reduced-precision kernel?
4. How is a failed request recovered without double-counting or cross-tenant cache exposure?
5. Which primary vendor or operator source supports a non-illustrative architecture claim?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | Collection catalog | retained as planned | no primary workload evidence or authored cases |

## 7. Pre-publish checks

- [ ] Assign stable `CASE-*` IDs only when the linked files exist.
- [ ] Reconcile the 11 catalog columns with each file's metadata and evidence label.
- [ ] Verify source date/version, workload units, arithmetic, architecture failure/recovery path, and all links.
- [ ] Keep proposed validation distinct from executed validation and preserve this catalog's planned rows.
