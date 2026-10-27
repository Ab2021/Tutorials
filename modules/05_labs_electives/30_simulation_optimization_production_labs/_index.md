# Module 30 — Simulation, Optimization, and Real-Time Production Labs

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/30_simulation_optimization_production_labs.md`](../../../course/30_simulation_optimization_production_labs.md)
> **Folder:** `modules/05_labs_electives/30_simulation_optimization_production_labs/`
> **Version:** 2026.5

---

## 1. Module purpose

This module is the lab bench: a reproducible repository, forecasting and inventory pipelines, MILP and VRP, warehouse simulation, streaming, containerized serving, Kubernetes, and disaster recovery.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 20, 26, 29 |
| Unlocks | None |
| Assessment | Lab portfolio and disaster-recovery report (Week 36) |
| Survey file | `30_simulation_optimization_production_labs.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_environment-repo.md`](01_environment-repo.md) | Environment and Repository Layout | Python, DuckDB, Docker, fixtures, seeds, and a test-first lab contract. | 1,200–1,800 | planned |
| 2 | [`02_demand-forecast-pipeline.md`](02_demand-forecast-pipeline.md) | Demand Forecast Pipeline | Ingest, validate, backtest, select, and publish a forecast. | 1,200–1,800 | planned |
| 3 | [`03_inventory-policy-simulation.md`](03_inventory-policy-simulation.md) | Inventory Policy Simulation | Policy sweep against a simulated demand stream and a cost model. | 1,200–1,800 | planned |
| 4 | [`04_network-milp.md`](04_network-milp.md) | Network MILP | Build, solve, inspect, and stress-test a facility-location model. | 1,200–1,800 | planned |
| 5 | [`05_vrp-dispatch.md`](05_vrp-dispatch.md) | VRP and Dispatch | CVRPTW, batching, and a dispatch replay. | 1,200–1,800 | planned |
| 6 | [`06_warehouse-des.md`](06_warehouse-des.md) | Warehouse Discrete-Event Simulation | Model a pick line, find the bottleneck, and validate against data. | 1,200–1,800 | planned |
| 7 | [`07_streaming-events.md`](07_streaming-events.md) | Streaming Events | Produce, consume, handle late data, and alert on an exception. | 1,200–1,800 | planned |
| 8 | [`08_model-service-container.md`](08_model-service-container.md) | Model Service and Container | Package a model behind an API with health checks and logging. | 1,200–1,800 | planned |
| 9 | [`09_kubernetes-terraform.md`](09_kubernetes-terraform.md) | Kubernetes and Terraform | Deploy, scale, and reproduce an environment declaratively. | 1,200–1,800 | planned |
| 10 | [`10_disaster-recovery.md`](10_disaster-recovery.md) | Observability and Disaster Recovery | Metrics, logs, backups, restore drill, and an incident runbook. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Environment and Repository Layout ->
02 Demand Forecast Pipeline ->
03 Inventory Policy Simulation ->
04 Network MILP ->
05 VRP and Dispatch ->
06 Warehouse Discrete-Event Simulation ->
07 Streaming Events ->
08 Model Service and Container ->
09 Kubernetes and Terraform ->
10 Observability and Disaster Recovery
```

## 5. What this module does not cover

- It does not teach cloud procurement or provide production credentials; use an approved sandbox.
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

See section 14 of the survey file (`30_simulation_optimization_production_labs.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
