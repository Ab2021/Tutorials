# Module 13 â€” Optimisation, Operations Research and Simulation

> **Part:** [Part 03 â€” Machine Learning, Optimisation and Intelligent Retrieval](../INDEX.md)  
> **Prerequisites:** 05,10  
> **Purpose:** Model constrained decisions with linear programming, integer optimisation, heuristics, and simulation.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | formulation and duality | [authored](Cheat_Sheets/01_optimisation_formulation.md) |
| 2 | network and scheduling models | [authored](Cheat_Sheets/04_network_and_scheduling_models.md) |
| 3 | integer programming | [authored](Cheat_Sheets/05_integer_programming.md) |
| 4 | metaheuristics | [authored](Cheat_Sheets/06_metaheuristics.md) |
| 5 | discrete-event simulation | planned |
## Exact local coverage ledger

Registered before individual authoring on 2026-11-08. Original commitments are retained verbatim.
Editorial state is distinct from implementation stage and evidence; all designs remain proposed.

| Commitment | Stable teaching ID | Individual destination | Bounded outcome | Numerical oracle | Initial editorial state |
|---|---|---|---|---|---|
| formulation and duality; original optimisation formulation sheet | OR-S01 | Cheat_Sheets/01_optimisation_formulation.md | Derive primal/dual certificates and interpret local resource values | Enumerate feasible integer grid and compare dual bound | registered |
| original solver selection sheet | OR-S02 | Cheat_Sheets/02_solver_selection.md | Choose a method from objective, integrality and certificate requirements | Classify explicit models; compare valid bounds | registered |
| discrete-event simulation; original simulation experiment plan sheet | OR-S03 | Cheat_Sheets/03_simulation_experiment_plan.md | Trace events, integrate queue area and design independent replications | FCFS recurrence and conservation | registered |
| network and scheduling models | OR-S04 | Cheat_Sheets/04_network_and_scheduling_models.md | Derive flow conservation, cuts and precedence timing | Enumerate flows and topological schedules | registered |
| integer programming | OR-S05 | Cheat_Sheets/05_integer_programming.md | Formulate binaries and explain branch-and-bound certificates | Exhaustive binary feasible-set enumeration | registered |
| metaheuristics | OR-S06 | Cheat_Sheets/06_metaheuristics.md | Compare neighborhoods, acceptance and multi-start evidence | Exact landscape and transition probabilities | registered |
| PLAN-2026-IR025 source addition | OR-S07 / CASE-2026-OR003 | [Cheat_Sheets/07_interval_greedy_and_weighted_scheduling.md](Cheat_Sheets/07_interval_greedy_and_weighted_scheduling.md); Case_Studies/03_scheduling_objective_and_constraints.md | Prove cardinality greedy and derive weighted DP with ties and buffers | Exhaustive subsets, predecessor and reconstruction trace | draft; unreviewed; case registered |
| original Last-mile routing optimisation case | CASE-2026-OR001 | Case_Studies/01_last_mile_routing_optimisation.md | Select feasible routes with capacity, windows and auditable fallbacks | All small route permutations and constraint checks | registered |
| original workforce scheduling under uncertainty case | CASE-2026-OR002 | Case_Studies/02_workforce_scheduling_under_uncertainty.md | Choose staffing against scenario shortages and protected constraints | Enumerate staffing choices and scenario cost | registered |
| final troubleshooting collection requirement | OR-S08 | Cheat_Sheets/08_25_Common_Issues.md | Diagnose 25 distinct complete failures across all five topics | Conservation, feasibility and certificate checks | registered |

The [source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) does not assign a dedicated visible
directory-tree lesson group to this module; the five original rows remain syllabus commitments.
The [actual inventory](../../../Templates/directory.txt) is a reconstructed visible tree, not access
to the hidden original source lessons. The available interval interview reference is supporting
material, not evidence of deployed scheduling outcomes. Its focused brief is
[REF-IR025](../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir025-choose-scheduling-algorithms-from-objective-and-constraints).
Collection catalogs retain the three original sheet titles and both original cases.

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR025 | Half-open intervals; deterministic ties; exchange proof; weighted interval DP boundary | When is earliest-finish greedy optimal, and when must scheduling become a weighted optimization problem? | Exhaustive small-instance optimum comparisons and counterexamples for weighted or buffered variants |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
