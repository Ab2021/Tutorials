# Evaluation, Reliability and Incidents - Cheat Sheets

> **Mandatory authoring contract:** Every authored sheet must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Purpose

These planned reference cards support repeatable decisions in Evaluation, Reliability and Incidents. They complement case studies; they do not replace the assumptions, evidence, and evaluation required in a full case.

## Planned references

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | Evaluation matrix | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 2 | SLO and error-budget guide | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 3 | Incident postmortem | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |

## Required card fields

Every card must state: intended decision and reader; preconditions and inputs; default recommendation; metrics/formulas or architecture pattern; trade-offs; failure signals; security/privacy/governance constraints; and links to deeper material. Review tool-, market-, and regulation-sensitive cards quarterly.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR008 | Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring | Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests. Boundary: A judge score is not a deployment guarantee; workload volume and evaluation deadlines remain unspecified. | [REF-IR008](../../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md) | planned; not written |
| PLAN-2026-IR030 | Hash flooding; BFS OOM; regex backtracking; recursive DFS depth; ORM N+1; overflow; unstable ties; floating search termination | Reproduce each mechanism separately with bounded adversarial traces; track latency, memory, database calls and deterministic outputs. Boundary: Ten reconstructed incident units include recurring mechanisms; reported timings are not measured outcomes of this library. | [REF-IR030](../../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
