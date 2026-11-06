# Module 23 Cheat Sheets â€” GPU Architecture, Accelerators and AI Hardware

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a draft catalog, not a published collection; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | [Roofline arithmetic intensity and kernel diagnosis](01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md) | Boundary-specific traffic, compute and bandwidth ceilings, numerical examples, and production diagnosis. | draft; review and publication checks open |
| 2 | precision guide | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | hardware sizing worksheet | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR007 | Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness | TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests. Boundary: GPU geometry, concurrency limits and speedups require workload measurements; speculative decoding can be slower. | [REF-IR007](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
