# Module 13 Cheat Sheets â€” Optimisation, Operations Research and Simulation

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Authored and planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | optimisation formulation | Variables, objective, constraints, duality and certificates. | planned |
| 2 | solver selection | Exact, heuristic and simulation selection under runtime/certificate constraints. | planned |
| 3 | simulation experiment plan | Event semantics, warm-up, replications and uncertainty. | planned |
| 4 | network and scheduling models | Flow, cuts, precedence and schedule feasibility. | planned |
| 5 | integer programming | Binaries, relaxations, branching and infeasibility. | planned |
| 6 | metaheuristics | Neighborhoods, acceptance and reproducible comparisons. | planned |
| 7 | [Interval greedy and weighted scheduling](07_interval_greedy_and_weighted_scheduling.md) | Half-open intervals, deterministic ties, exchange proof, weighted-DP boundary and subset oracle. | Draft; unreviewed. |
| 8 | 25 Common Issues | Collection-wide diagnostics. | planned |

The authored sheet uses the visible inventory and [REF-IR025](../../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir025-choose-scheduling-algorithms-from-objective-and-constraints) as scope, not as evidence that the original lesson body or a deployed scheduler is available.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR025 | Half-open intervals; deterministic ties; exchange proof; weighted interval DP boundary | Exhaustive small-instance optimum comparisons and counterexamples for weighted or buffered variants. Boundary: The reference proves equal-value single-room cardinality only; warehouse jobs and staffing need their own constraints. | [REF-IR025](../../../../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
