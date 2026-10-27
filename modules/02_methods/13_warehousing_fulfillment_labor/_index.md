# Module 13 — Warehousing, Fulfillment, and Labor Analytics

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/13_warehousing_fulfillment_labor.md`](../../../course/13_warehousing_fulfillment_labor.md)
> **Folder:** `modules/02_methods/13_warehousing_fulfillment_labor/`
> **Version:** 2026.5

---

## 1. Module purpose

This module models the warehouse as a flow system: capacity, labour, slotting, picking, travel, automation, and the fulfillment network behind the promise.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01, 12 |
| Unlocks | 19, 22, 23 |
| Assessment | Fulfillment capacity model (Week 15) |
| Survey file | `13_warehousing_fulfillment_labor.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_warehouse-flow-system.md`](01_warehouse-flow-system.md) | The Warehouse as a Flow System | Receiving, put-away, storage, pick, pack, ship, and the buffers between. | 1,200–1,800 | planned |
| 2 | [`02_wms-data-model.md`](02_wms-data-model.md) | Process Flows and the WMS Data Model | Tasks, locations, inventory states, and the timestamps that reveal bottlenecks. | 1,200–1,800 | planned |
| 3 | [`03_productivity-metrics.md`](03_productivity-metrics.md) | Productivity Metrics | Units, lines, orders per hour and the denominators that make them comparable. | 1,200–1,800 | planned |
| 4 | [`04_littles-law-queueing.md`](04_littles-law-queueing.md) | Little's Law and Queueing | Flow time, work in process, and why utilisation is not free. | 1,200–1,800 | planned |
| 5 | [`05_labor-standards.md`](05_labor-standards.md) | Labour Standards and Engineered Standards | Time and motion, allowances, and the politics of the standard. | 1,200–1,800 | planned |
| 6 | [`06_slotting-abc.md`](06_slotting-abc.md) | Slotting and ABC | Velocity, affinity, and the travel cost of a bad slot. | 1,200–1,800 | planned |
| 7 | [`07_picking-strategies.md`](07_picking-strategies.md) | Picking Strategies | Discrete, batch, zone, wave, and cluster picking trade-offs. | 1,200–1,800 | planned |
| 8 | [`08_pick-path-travel.md`](08_pick-path-travel.md) | Pick Path and Travel Time | Routing inside the four walls and the value of a better sequence. | 1,200–1,800 | planned |
| 9 | [`09_capacity-des.md`](09_capacity-des.md) | Capacity and Discrete-Event Simulation | Validating a capacity plan before peak. | 1,200–1,800 | planned |
| 10 | [`10_automation-micro-fulfillment.md`](10_automation-micro-fulfillment.md) | Automation and Micro-Fulfillment | AS/RS, AMRs, goods-to-person, and when the payback is real. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Warehouse as a Flow System ->
02 Process Flows and the WMS Data Model ->
03 Productivity Metrics ->
04 Little's Law and Queueing ->
05 Labour Standards and Engineered Standards ->
06 Slotting and ABC ->
07 Picking Strategies ->
08 Pick Path and Travel Time ->
09 Capacity and Discrete-Event Simulation ->
10 Automation and Micro-Fulfillment
```

## 5. What this module does not cover

- It does not cover transport routing or inventory policy; Modules 12 and 10 own those decisions.
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

See section 14 of the survey file (`13_warehousing_fulfillment_labor.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
