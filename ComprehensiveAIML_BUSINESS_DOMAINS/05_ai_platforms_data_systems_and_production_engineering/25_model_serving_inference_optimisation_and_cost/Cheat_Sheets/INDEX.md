# Module 25 Cheat Sheets â€” Model Serving, Inference Optimisation and Cost

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | inference optimisation | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | SLO worksheet | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | capacity planning formulas | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR002 | Eligibility before scoring; capability adapters; health-aware routing; bounded retry budgets | Constraint violations, successful-task cost, deadline misses and fallback correctness on fault replays. Boundary: Revalidate provider capabilities and economics when authoring; source tables are not current service guarantees. | [REF-IR002](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md) | planned; not written |
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix. Boundary: Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements. | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | planned; not written |
| PLAN-2026-IR007 | Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness | TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests. Boundary: GPU geometry, concurrency limits and speedups require workload measurements; speculative decoding can be slower. | [REF-IR007](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md) | planned; not written |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing. Boundary: Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes. | [REF-IR014](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
