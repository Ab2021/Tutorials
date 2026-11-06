# Module 49 Cheat Sheets â€” AI Security, Privacy, Safety and Red Teaming

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | AI threat matrix | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | red-team plan | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | secure-agent checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix. Boundary: Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements. | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | planned; not written |
| PLAN-2026-IR008 | Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring | Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests. Boundary: A judge score is not a deployment guarantee; workload volume and evaluation deadlines remain unspecified. | [REF-IR008](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
