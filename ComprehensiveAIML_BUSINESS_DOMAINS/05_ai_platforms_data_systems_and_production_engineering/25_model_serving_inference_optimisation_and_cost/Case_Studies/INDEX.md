# Module 25 Case Studies — Model Serving, Inference Optimisation and Cost

**Status:** planned catalog; no authored `CASE-*` files. Use the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | online and batch model serving |
| Evidence boundary | proposed illustrative designs; provider capability, price, and performance require dated primary evidence |
| Review trigger | model/provider, routing policy, workload, SLO, tenant boundary, or cost change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-25-01 | Multi-model inference gateway | serving | llm-agentic | select eligible route | eligibility-before-ranking | cost-latency-tradeoff | not-measured | proposed | — | needs fault replay and current provider evidence |
| PLAN-25-02 | Reduce p99 LLM latency | serving | hybrid | select intervention | resource-admission | correctness-contract | not-measured | proposed | — | needs workload-slice measurements |
| PLAN-2026-IR002 | Select model routes within hard customer constraints | serving | llm-agentic | choose eligible provider/fallback | eligibility-before-ranking | cost-latency-tradeoff | not-measured | proposed | 2026-11-06 (planning) | source tables are not current guarantees |
| PLAN-2026-IR004 | Customize tenant assistants without crossing isolation boundaries | serving | llm-agentic | adapter lifecycle and rollback | authorization-scope | atomic-release-bundle; rollout-regression | not-measured | proposed | 2026-11-06 (planning) | adapters do not establish isolation |
| PLAN-2026-IR007 | Balance interactive inference latency and admitted throughput | serving | hybrid | scheduling/KV policy | resource-admission | cost-latency-tradeoff; authorization-scope | not-measured | proposed | 2026-11-06 (planning) | speculative decoding can be slower |
| PLAN-2026-IR014 | Separate extraction correctness from routing economics | serving | llm-agentic | accept contracts/routes | evaluation-mismatch | idempotent-side-effects; cost-latency-tradeoff | not-measured | proposed | 2026-11-06 (planning) | valid JSON is not business success |

## 3. Canonical reusable patterns

- **Eligibility before ranking:** reject routes that violate residency, capability, authorization, or deadline before optimizing score or price.
- **Resource admission:** reserve bounded KV, queue, and compute capacity before accepting work.
- **Authorization scope:** bind cache, adapter, prompt, and tool state to the tenant and policy boundary.

## 4. Similarity / contrast map

| Cluster | Shared reasoning | Critical distinction |
|---|---|---|
| routing and gateway | eligibility and fallback correctness | health is not authorization or capability evidence |
| latency and batching | queue/KV/capacity accounting | lower mean latency can worsen tail isolation |
| extraction and adapters | versioning and rollback | syntax validity is not domain correctness |

## 5. Questions and future-case backlog

1. Which constraint makes a route ineligible before ranking?
2. How are retry budgets, cancellation, and idempotency carried across a fallback?
3. What target-equivalence test rejects a speedup that changes output policy?
4. Which tenant-negative tests prove cache and adapter isolation?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | retained as planned | no executed workload, provider verification, or authored case |

## 7. Pre-publish checks

- [ ] Assign `CASE-*` only to a linked 15-section case.
- [ ] Check offered/accepted load, TTFT, inter-token latency, successful-task cost, and isolation failures on the same workload.
- [ ] Preserve planned rows and reconcile source, evidence, stage, and review status.
