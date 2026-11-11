# Module 13 â€” Optimisation, Operations Research and Simulation — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-2026-OR001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-OR001 | [Last-mile routing optimisation](./01_last_mile_routing_optimisation.md) | Module 13 â€” Optimisation, Operations Research and Simulation | analytical | decision record in the linked case | constrained vehicle routing with time windows | constructive, savings, heuristic, independent, feasibility,  | illustrative | proposed | null | authored |
| CASE-2026-OR002 | [Workforce scheduling under uncertainty](./02_workforce_scheduling_under_uncertainty.md) | Module 13 â€” Optimisation, Operations Research and Simulation | analytical | decision record in the linked case | two-stage stochastic staffing with scenario-weighted service penalty | shift-cover, matrix, absence, overlay, via, binomial, availa | illustrative | proposed | null | authored |
| CASE-2026-OR003 | [Scheduling objective and constraints](./03_scheduling_objective_and_constraints.md) | Module 13 â€” Optimisation, Operations Research and Simulation | analytical | decision record in the linked case | objective-choice failure between cardinality and weighted interval selection | half-open, interval, convention, audit, buffer-gap, sensitiv | illustrative | proposed | null | authored |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 3 case file(s), 3 passing structural lint, and 0 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

No unfulfilled case commitment is recorded for this collection.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Last-mile routing optimisation](./01_last_mile_routing_optimisation.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Workforce scheduling under uncertainty](./02_workforce_scheduling_under_uncertainty.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Scheduling objective and constraints](./03_scheduling_objective_and_constraints.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
