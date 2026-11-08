# Module 20 Cheat Sheets â€” RAG, Context Engineering and Enterprise Knowledge

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This collection has one unreviewed course-topic draft; remaining sheets and the final Common Issues artifact (at least 25 complete issues) are planned. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | [Permission-aware retrieval](01_Permission_Aware_Retrieval.md) | Eligibility, provenance, access evaluation and failure containment. | Draft; unreviewed. |
| 2 | chunking guide | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | retrieval metric card | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR003 | Authorized retrieval; effective policy versions; claim-level citations; embedding migrations | Unsupported-claim rate, eligible-span recall, abstention quality and policy-version errors against expert labels. Boundary: Citations and entailment checks require validation; never transfer the example refund window into a new business. | [REF-IR003](../../../../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md) | planned; not written |
| PLAN-2026-IR012 | Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation | All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency. Boundary: Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements. | [REF-IR012](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | planned; not written |
| PLAN-2026-IR013 | Effective intervals; recorded-time history; jurisdiction rules; hybrid retrieval; reranking and claim verification | Boundary-date and amendment replays, eligible recall, wrong-version citations and expert-reviewed abstentions. Boundary: This is an architecture exercise, not current legal guidance; eligibility depends on maintained authoritative metadata. | [REF-IR013](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
