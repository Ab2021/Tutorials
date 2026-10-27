# Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/20_digital_twins_simulation_optimization_uncertainty.md`](../../../course/20_digital_twins_simulation_optimization_uncertainty.md)
> **Folder:** `modules/02_methods/20_digital_twins_simulation_optimization_uncertainty/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches when a twin is worth building, how to validate a simulation, and how to optimise under uncertainty with scenarios, stochastic programming, robust methods, and MDPs.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06, 11, 15 |
| Unlocks | 22, 25, 30 |
| Assessment | Simulation study (Week 23) |
| Survey file | `20_digital_twins_simulation_optimization_uncertainty.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_twin-shadow-simulation.md`](01_twin-shadow-simulation.md) | Twin versus Shadow versus Simulation, and Model-Family Choice | The three artefacts, the decision each supports, and DES versus system dynamics versus agent-based models. | 1,200–1,800 | planned |
| 2 | [`02_discrete-event-simulation.md`](02_discrete-event-simulation.md) | Discrete-Event Simulation | Entities, events, resources, queues, and the clock. | 1,200–1,800 | planned |
| 3 | [`03_input-modeling.md`](03_input-modeling.md) | Input Modelling | Fitting demand, service, and failure distributions from real data. | 1,200–1,800 | planned |
| 4 | [`04_validation.md`](04_validation.md) | Validation | Face, structural, and output validation against a known period. | 1,200–1,800 | planned |
| 5 | [`05_experiment-design.md`](05_experiment-design.md) | Experiment Design | Replications, common random numbers, and confidence intervals. | 1,200–1,800 | planned |
| 6 | [`06_stochastic-optimization.md`](06_stochastic-optimization.md) | Stochastic Optimization | Scenarios, sample average approximation, and the value of the stochastic solution. | 1,200–1,800 | planned |
| 7 | [`07_robust-dro.md`](07_robust-dro.md) | Robust and Distributionally Robust Optimization | Hedging against a misspecified distribution. | 1,200–1,800 | planned |
| 8 | [`08_mdp-rl.md`](08_mdp-rl.md) | Markov Decision Processes and Reinforcement Learning | State, action, reward, and when RL is the wrong tool. | 1,200–1,800 | planned |
| 9 | [`09_simulation-optimization.md`](09_simulation-optimization.md) | Simulation-Optimization | Searching a policy space when the objective is a simulation. | 1,200–1,800 | planned |
| 10 | [`10_twin-architecture.md`](10_twin-architecture.md) | Twin Architecture and When Not to Build | Data, cadence, maintenance, and the honest cost of a twin. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Twin versus Shadow versus Simulation ->
02 Discrete-Event Simulation ->
03 Input Modelling ->
04 Validation ->
05 Experiment Design ->
06 Stochastic Optimization ->
07 Robust and Distributionally Robust Optimization ->
08 Markov Decision Processes and Reinforcement Learning ->
09 Simulation-Optimization ->
10 Twin Architecture and When Not to Build
```

## 5. What this module does not cover

- It does not cover real-time control-system engineering or plant automation integration.
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

See section 14 of the survey file (`20_digital_twins_simulation_optimization_uncertainty.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
