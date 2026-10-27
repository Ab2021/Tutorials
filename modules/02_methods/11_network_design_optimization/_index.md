# Module 11 — Network Design and Optimization

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)
> **Folder:** `modules/02_methods/11_network_design_optimization/`
> **Version:** 2026.5

---

## 1. Module purpose

This module formulates the network decision: which facilities, which flows, which capacity, at what cost, under which scenarios, and with carbon in the objective.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01, 10 |
| Unlocks | 13, 14A |
| Assessment | Network design study (Week 13) |
| Survey file | `11_network_design_optimization.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_network-decision-ladder.md`](01_network-decision-ladder.md) | The Network Decision Ladder | Strategic, tactical, and operational network choices and their time scales. | 1,200–1,800 | planned |
| 2 | [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md) | The Network as a Graph; Cost-to-Serve | Nodes, arcs, flows, and the decomposition of delivered cost. | 1,200–1,800 | planned |
| 3 | [`03_milp-anatomy.md`](03_milp-anatomy.md) | MILP Anatomy | Variables, constraints, objective, and the modelling choices that decide solvability. | 1,200–1,800 | planned |
| 4 | [`04_facility-location.md`](04_facility-location.md) | The Baseline Capacitated Facility-Location Model | Fixed charge, capacity, and demand allocation. | 1,200–1,800 | planned |
| 5 | [`05_multi-commodity-period.md`](05_multi-commodity-period.md) | Multi-Commodity and Multi-Period Models | Product families, inventories, and time-expanded networks. | 1,200–1,800 | planned |
| 6 | [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md) | Greenfield and Brownfield Design | Building from scratch versus adapting an existing footprint. | 1,200–1,800 | planned |
| 7 | [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md) | Scenario and Sensitivity Analysis | Which assumptions change the answer, and by how much. | 1,200–1,800 | planned |
| 8 | [`08_robust-stochastic.md`](08_robust-stochastic.md) | Robust and Stochastic Optimization | Hedging against demand, cost, and disruption uncertainty. | 1,200–1,800 | planned |
| 9 | [`09_carbon-objective.md`](09_carbon-objective.md) | Carbon in the Objective | Emissions factors, scope boundaries, and multi-objective trade-offs. | 1,200–1,800 | planned |
| 10 | [`10_solvers-heuristics.md`](10_solvers-heuristics.md) | Solvers, Heuristics, and Practical Limits | MIP gaps, decomposition, and when a heuristic is the professional answer. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Network Decision Ladder ->
02 The Network as a Graph; Cost-to-Serve ->
03 MILP Anatomy ->
04 The Baseline Capacitated Facility-Location Model ->
05 Multi-Commodity and Multi-Period Models ->
06 Greenfield and Brownfield Design ->
07 Scenario and Sensitivity Analysis ->
08 Robust and Stochastic Optimization ->
09 Carbon in the Objective ->
10 Solvers, Heuristics, and Practical Limits
```

## 5. What this module does not cover

- It does not solve routing or scheduling; Modules 12 and 15 cover those decision layers.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`11_network_design_optimization.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
