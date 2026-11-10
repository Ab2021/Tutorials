# Module 13 Cheat Sheets — Optimisation, Operations Research and Simulation

## 1. Overview

This collection teaches how to state a constrained operational decision before selecting an exact solver, constructive rule, heuristic, or simulation. It serves analysts who must show what is feasible, which objective is being optimized, and how a result changes when a capacity, interval, or stochastic assumption changes. Readiness means the learner can formulate variables, units and constraints; calculate or trace the claimed method; reject an invalid formulation; and return an explicit infeasibility or uncertainty boundary. Individual teaching artifacts are governed by the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [cheat-sheet template](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md).

## 2. Topic Ladder

| Level | File | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| 1 | `01_optimisation_formulation.md` | express a decision as variables, objective and constraints; interpret a dual/bound | algebra; inequalities | registered |
| 1 | `02_solver_selection.md` | select an exact, heuristic, or simulation approach from required guarantees | formulation | registered |
| 1 | `03_simulation_experiment_plan.md` | specify events, warm-up, replications and uncertainty | probability; queues | registered |
| 2 | `04_network_and_scheduling_models.md` | trace conservation, cuts and precedence feasibility | formulation; graphs | registered |
| 2 | `05_integer_programming.md` | formulate binaries and interpret relaxation/branch certificates | formulation | registered |
| 2 | `06_metaheuristics.md` | compare neighborhood search and multi-start evidence without calling it optimal | formulation; probability | registered |
| 2 | [07 Interval greedy and weighted scheduling](07_interval_greedy_and_weighted_scheduling.md) | select cardinality greedy or weighted DP under declared interval semantics | sorting; induction; DP | draft; unreviewed |
| Final | `08_25_Common_Issues.md` | diagnose 25 distinct formulation, numerical, scale and concept failures | all prior files | registered |

## 3. Quick Topic Lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| What is a valid optimization model? | `01`, §3–§5 | decision variables and units are known | a verbal objective is not a constraint |
| Must a result be proven optimal? | `02`, §7 | guarantee requirements are explicit | a good heuristic trace is not a certificate |
| Is queue performance stable? | `03`, §4–§5 | event and warm-up rules are fixed | one run cannot quantify stochastic uncertainty |
| Are touching appointments compatible? | [07, §3–§5](07_interval_greedy_and_weighted_scheduling.md#3-foundational-definitions) | endpoint convention is contractual | a buffer changes predecessor links |
| Why did a schedule change across runs? | [07, §3.5](07_interval_greedy_and_weighted_scheduling.md#35-determinism-ties-and-the-audit-record) | ties exist | unspecified sort/reconstruction breaks reproducibility |

## 4. Reading Paths — Three Options

- **Beginner:** formulation → solver selection → simulation plan. Objective: formulate a bounded decision and state evidence needed to choose a method.
- **Practitioner:** Beginner path → network/scheduling → integer programming → interval scheduling. Objective: diagnose feasibility and select a defendable solver/algorithm.
- **Advanced:** all files in ladder order → metaheuristics → Common Issues → module cases. Objective: defend certificates, stochastic assumptions and changed-constraint behavior.

## 5. Prerequisite Skills

| Skill | Why needed | Handoff |
|---|---|---|
| Algebra, inequalities and units | constraints and dual interpretation | Module 05 mathematical foundations |
| Graph traversal and sorting | flow/scheduling construction | Part 01 algorithms |
| Probability and confidence intervals | simulation uncertainty | Module 05 statistics |
| Event time and business calendars | interval and queue contracts | Module 06 data modelling |

## 6. What This Collection Does NOT Cover

This collection does not claim that an illustrative model is an executed dispatch, labor, routing, or solver deployment. It does not replace safety analysis, legal labor constraints, contract-specific calendars, nonlinear global optimization, or a measured workload benchmark. Such facts require their own evidence and decision owner.

## 7. Master Symbol Glossary

| Symbol/term | Meaning | Units/type | Valid range | First use |
|---|---|---|---|---|
| `x` | decision-variable vector | scalar/vector | stated domain | `01` |
| `c` | objective coefficient vector | value per variable unit | finite stated values | `01` |
| `A x <= b` | linear resource constraints | compatible matrix/vector | dimensions agree | `01` |
| `z` | objective value | value units | objective-specific | `01` |
| `[s_i,f_i)` | job occupancy interval | timestamps | `s_i < f_i` | [07 §3](07_interval_greedy_and_weighted_scheduling.md#3-foundational-definitions) |
| `w_i` | additive job value | value units | explicitly stated | [07 §3](07_interval_greedy_and_weighted_scheduling.md#3-foundational-definitions) |
| `p(i)` | latest compatible predecessor index | integer | `0 <= p(i) < i` | [07 §4](07_interval_greedy_and_weighted_scheduling.md#4-core-formulas-with-full-derivation) |
| `OPT(i)` | best value through finish-sorted job `i` | value units | nondecreasing if values nonnegative | [07 §4](07_interval_greedy_and_weighted_scheduling.md#4-core-formulas-with-full-derivation) |

## 8. Quick Reference Card Directory

| File | Card | Core rule | Critical assumption | Avoid when |
|---|---|---|---|---|
| `07` | Section 12 card not yet authored in the interrupted draft | earliest finish maximizes cardinality; recurrence maximizes additive value | one resource and declared compatibility | treating the existing derivation as a completed final card |
| `01`–`06` | planned | no card available | artifact not authored | treating a title as a reference card |

## 9. Common Issues Locator

`08_25_Common_Issues.md` is registered but not authored. It must contain 25 distinct, complete issues across `VIOLATED_ASSUMPTION`, `NUMERICAL_EDGE`, `SCALE_FAILURE`, and `CONCEPT_CONFUSION`, with diagnostics, mitigations, risks and verification. The five localized failure rows in [07 §8](07_interval_greedy_and_weighted_scheduling.md#8-failure-modes--edge-cases) are not the final collection artifact.

## 10. Empirical Sources, Versions and Collection Maintenance

| Artifact | Source/evidence state | Review trigger |
|---|---|---|
| [07](07_interval_greedy_and_weighted_scheduling.md) | DERIVED recurrence/exchange reasoning; ILLUSTRATIVE jobs; locally documented hand trace; original source body OPEN | endpoint semantics, value objective, buffer rule, or tie policy changes |
| `01`–`06`, `08` | registered destinations; no authored evidence | artifact is authored or primary sources are obtained |
| IR025 | [REF-IR025 plan](../../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir025-choose-scheduling-algorithms-from-objective-and-constraints) is scope only | source access or plan changes |

No artifact is independently reviewed or published on the basis of this index.

## 11. Collection Acceptance Checklist

- [x] Original three reference families and five module outcomes have stable destinations.
- [x] IR025 mechanisms are mapped to the existing interval-scheduling draft without claiming source access.
- [x] Draft card and localized failure anchors are navigable for file 07.
- [ ] Files 01–06 each have a substantive 12-section teaching artifact.
- [ ] File 07 reaches the course substantive-depth gate and passes independent technical review.
- [ ] File 08 contains 25 complete, distinct diagnostics and a final Section 12 card.
- [ ] Collection arithmetic, links, sources and editorial states have independent review.
