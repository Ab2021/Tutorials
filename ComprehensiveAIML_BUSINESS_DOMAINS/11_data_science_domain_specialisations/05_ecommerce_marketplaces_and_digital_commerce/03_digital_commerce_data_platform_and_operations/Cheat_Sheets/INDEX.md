# Digital Commerce Data Platform and Operations - Cheat Sheets

> **Mandatory authoring contract:** Every authored sheet must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Purpose

These planned reference cards support repeatable decisions in Digital Commerce Data Platform and Operations. They complement case studies; they do not replace the assumptions, evidence, and evaluation required in a full case.

## Planned references

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | Commerce event schema | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 2 | Experiment platform checklist | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 3 | Order-state machine | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |

## Required card fields

Every card must state: intended decision and reader; preconditions and inputs; default recommendation; metrics/formulas or architecture pattern; trade-offs; failure signals; security/privacy/governance constraints; and links to deeper material. Review tool-, market-, and regulation-sensitive cards quarterly.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR033 | Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones | Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races. Boundary: Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share. | [REF-IR033](../../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
