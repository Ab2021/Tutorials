# Module 23 Case Studies — GPU Architecture, Accelerators and AI Hardware

**Collection status:** one authored illustrative case is a proposed intervention and draft document; remaining rows are planned.  
**Contract:** [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). Evidence level, implementation stage, and editorial state are separate.

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | GPU architecture, accelerators, and AI hardware |
| Canonical case types | `streaming-systems`, `incident`, or genuinely integrated `hybrid` |
| Evidence boundary | Illustrative/proposed unless a case cites an accessible primary source; no vendor-workload result is implied |
| Source limits | Visible source-tree topic names remain OPEN where original lesson bodies are unavailable |
| Editorial state | CASE-2026-023-001 is draft; planned rows are not authored cases |
| Review trigger | Device/runtime/model/dtype/context, workload, failure-domain, or measured constraint change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-023-001 | [Size an inference cluster under an interactive latency and memory budget](01_Size_an_Inference_Cluster.md) | hardware/interactive serving | hybrid | select capacity, replica count, and admission policy | resource-admission | cost-latency-tradeoff; correctness-contract | illustrative | proposed | — | draft; arithmetic and state boundary are derived; workload/runtime evidence remains required |
| PLAN-23-02 | Profile memory-bound training | hardware | incident | choose the bottleneck-removal change | resource-accounting | correctness-contract | not-measured | proposed | — | planned; needs profiler and numerical-parity evidence |
| PLAN-2026-IR007 | Balance interactive inference latency and admitted throughput | hardware/serving | hybrid | scheduling and KV-memory policy | resource-admission | cost-latency-tradeoff; authorization-scope | not-measured | proposed | 2026-11-06 (planning) | planned reference addition; geometry and speedups unmeasured |

## 3. Canonical reusable patterns

- **Resource admission:** accept work only when a defined memory, compute, and deadline budget remains; test rejection and recovery, not just average throughput.
- **Resource accounting:** account for a named boundary—weights, runtime reserve, KV cache, rank, replica, host link, or network—with units and ownership.
- **Cost–latency tradeoff:** compare successful-request cost, tail latency, and capacity headroom under the same offered workload and outcome denominator.
- **Correctness contract:** require lease, retry, terminal-release, and numerical/target-equivalence checks before adopting a performance change.

## 4. Similarity / contrast map

| Case | Shares | Contrasts with | Transfer boundary |
|---|---|---|---|
| CASE-2026-023-001 | admission, KV accounting, cancellation, and capacity reasoning | illustrative target-runtime values rather than a measured benchmark | memory fit is not an interactive latency result |
| PLAN-23-02 | profiling and parity validation with distributed-training cases | training has long-lived state rather than interactive deadlines | optimize only the measured limiting boundary |
| PLAN-2026-IR007 | batching and KV accounting with Module 25 | authorization scope and lease fencing are serving concerns | speculative decoding or paged KV needs its own evidence |

## 5. Questions and future-case backlog

1. What device, dtype, shapes, and traffic boundary make a roofline calculation relevant?
2. Which trace captures peak concurrency, arrivals, cancellation, retry chains, and KV residency?
3. What numerical invariant detects an invalid fused or reduced-precision kernel?
4. How is a failed request recovered without double-counting or cross-tenant cache exposure?
5. Which primary vendor or operator source supports a non-illustrative architecture claim?
6. Under what measured prefill/decode mix does a class-aware queue improve the stated SLO?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | Collection catalog | retained planned rows | no primary workload evidence or authored case then |
| 2026-11-10 | CASE-2026-023-001 | catalog reconciled as authored draft / proposed intervention | constructed arithmetic and state contract are complete; runtime evidence remains OPEN |

## 7. Pre-publish checks

- [x] Stable CASE ID, linked manuscript, 15 sections, and final card are present for CASE-2026-023-001.
- [x] Catalog has 11 columns and keeps evidence, stage, and editorial status distinct in its limitation field and metadata.
- [ ] Verify target-runtime source version, workload trace, device/rank memory inventory, and closed-loop measurements before review/publication.
- [ ] Verify all links and rerun the arithmetic, failure/recovery trace, and N-1 capacity boundary when the configuration changes.
- [x] Retain proposed cases and source/reference additions without treating them as completed.
