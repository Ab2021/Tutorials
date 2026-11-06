# Module 06 Cheat Sheets â€” Data Modelling, Databases, Lakehouse and Governance

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | data model notation | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | warehouse query patterns | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | data contract checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR012 | Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation | All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency. Boundary: Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements. | [REF-IR012](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | planned; not written |
| PLAN-2026-IR021 | WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones | Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity. Boundary: Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks. | [REF-IR021](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | planned; not written |
| PLAN-2026-IR031 | N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads | Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior. Boundary: One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound. | [REF-IR031](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
