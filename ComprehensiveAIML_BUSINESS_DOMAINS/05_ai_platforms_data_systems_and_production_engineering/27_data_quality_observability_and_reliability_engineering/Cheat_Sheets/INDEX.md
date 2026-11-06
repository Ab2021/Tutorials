# Module 27 Cheat Sheets â€” Data Quality, Observability and Reliability Engineering

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | data observability checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | ML SLOs | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | incident postmortem template | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR001 | Dataset lineage; synthetic-data feedback; deduplication limits; language and domain coverage | Held-out long-tail factual accuracy, diversity and per-slice retention; compare filtering with provenance controls. Boundary: A synthetic-text classifier cannot establish authorship; source incident names and costs are unverified. | [REF-IR001](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md) | planned; not written |
| PLAN-2026-IR033 | Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones | Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races. Boundary: Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share. | [REF-IR033](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
