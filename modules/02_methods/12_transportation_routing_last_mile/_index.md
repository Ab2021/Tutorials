# Module 12 — Transportation, Routing, and Last-Mile Analytics

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
> **Folder:** `modules/02_methods/12_transportation_routing_last_mile/`
> **Version:** 2026.5

---

## 1. Module purpose

This module covers freight economics, rate and tender analytics, vehicle routing, dispatch, last-mile density, and ETA prediction as one service-cost system.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01, 10 |
| Unlocks | 13, 19, 21, 22 |
| Assessment | Routing and tender analysis (Week 14) |
| Survey file | `12_transportation_routing_last_mile.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_freight-modes.md`](01_freight-modes.md) | Freight Modes and Their Economics | Truckload, LTL, parcel, rail, air, and ocean cost structures. | 1,200–1,800 | planned |
| 2 | [`02_rate-structures.md`](02_rate-structures.md) | Rate Structures and Accessorials | Linehaul, fuel, minimums, demurrage, detention, and accessorial creep. | 1,200–1,800 | planned |
| 3 | [`03_lane-tender-analytics.md`](03_lane-tender-analytics.md) | Lane and Tender Analytics | Lane segmentation, acceptance rates, and carrier mix. | 1,200–1,800 | planned |
| 4 | [`04_consolidation-cube.md`](04_consolidation-cube.md) | Consolidation and Cube Utilisation | Weight, cube, and the cost of running half-empty. | 1,200–1,800 | planned |
| 5 | [`05_vrp-formulation.md`](05_vrp-formulation.md) | The Vehicle Routing Problem | Capacity, time windows, and the formulation choices that matter. | 1,200–1,800 | planned |
| 6 | [`06_cvrptw-ortools.md`](06_cvrptw-ortools.md) | A Runnable CVRPTW in OR-Tools | Model, solve, inspect, and explain a route plan. | 1,200–1,800 | planned |
| 7 | [`07_route-vs-dispatch.md`](07_route-vs-dispatch.md) | Route Optimisation versus Dispatch | Planning the route and running the day are different systems. | 1,200–1,800 | planned |
| 8 | [`08_last-mile.md`](08_last-mile.md) | Last Mile: Density, Windows, and Failure | Drop density, failed delivery, and the economics of time windows. | 1,200–1,800 | planned |
| 9 | [`09_eta-prediction.md`](09_eta-prediction.md) | ETA Prediction | Features, censoring, and the difference between a promise and a prediction. | 1,200–1,800 | planned |
| 10 | [`10_dock-yard.md`](10_dock-yard.md) | Dock Scheduling and Yard Management | Appointment slots, dwell, and the handoff between transport and warehouse. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Freight Modes and Their Economics ->
02 Rate Structures and Accessorials ->
03 Lane and Tender Analytics ->
04 Consolidation and Cube Utilisation ->
05 The Vehicle Routing Problem ->
06 A Runnable CVRPTW in OR-Tools ->
07 Route Optimisation versus Dispatch ->
08 Last Mile: Density, Windows, and Failure ->
09 ETA Prediction ->
10 Dock Scheduling and Yard Management
```

## 5. What this module does not cover

- It does not cover warehouse labour or inventory policy; Modules 13 and 10 own those decisions.
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

See section 14 of the survey file (`12_transportation_routing_last_mile.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
