# Module 26 Cheat Sheets â€” MLOps, Feature Platforms and Model Lifecycle

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This collection has one unreviewed course-topic draft; remaining sheets and the final Common Issues artifact (at least 25 complete issues) are planned. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | [ML release and rollback checklist](01_ML_Release_and_Rollback_Checklist.md) | Release bundle, gates, staged rollout, rollback and operating controls. | Draft; unreviewed. |
| 2 | release checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | model-card template | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR005 | Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment | Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery. Boundary: A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults. | [REF-IR005](../../../../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
