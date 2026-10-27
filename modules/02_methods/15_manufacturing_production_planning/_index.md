# Module 15 — Manufacturing and Production Planning

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/15_manufacturing_production_planning.md`](../../../course/15_manufacturing_production_planning.md)
> **Folder:** `modules/02_methods/15_manufacturing_production_planning/`
> **Version:** 2026.5

---

## 1. Module purpose

This module covers the planning cascade from S&OP to the schedule: MPS, MRP, capacity, finite scheduling, lot sizing, changeovers, OEE, bottlenecks, lean, and SPC.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01, 10 |
| Unlocks | 20, 30 |
| Assessment | Production planning model (Week 18) |
| Survey file | `15_manufacturing_production_planning.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_planning-cascade.md`](01_planning-cascade.md) | The Planning Cascade and S&OP | S&OP, IBP, S&OE, MPS, MRP, scheduling, and execution with the decisions at each level. | 1,200–1,800 | planned |
| 2 | [`02_manufacturing-master-data.md`](02_manufacturing-master-data.md) | The Manufacturing Master Data Model | Items, BOMs, routings, work centres, calendars, and their data defects. | 1,200–1,800 | planned |
| 3 | [`03_master-production-scheduling.md`](03_master-production-scheduling.md) | Master Production Scheduling | MPS as the buffer between demand plan and material plan. | 1,200–1,800 | planned |
| 4 | [`04_mrp-explosion.md`](04_mrp-explosion.md) | MRP Logic and a Worked Explosion | Gross to net requirements, lead-time offsetting, and nervousness. | 1,200–1,800 | planned |
| 5 | [`05_capacity-requirements.md`](05_capacity-requirements.md) | Capacity Requirements Planning | Load, capacity, and the gap that must be resolved by a decision. | 1,200–1,800 | planned |
| 6 | [`06_finite-scheduling.md`](06_finite-scheduling.md) | Finite Scheduling | Single machine, flow shop, job shop, and dispatching rules. | 1,200–1,800 | planned |
| 7 | [`07_lot-sizing.md`](07_lot-sizing.md) | Lot Sizing: Four Methods Compared | Lot-for-lot, EOQ, POQ, and Wagner-Whitin with setup and holding costs. | 1,200–1,800 | planned |
| 8 | [`08_changeovers-smed.md`](08_changeovers-smed.md) | Sequence-Dependent Changeovers and SMED | Setup time as a scheduling constraint and an improvement target. | 1,200–1,800 | planned |
| 9 | [`09_oee-loss-tree.md`](09_oee-loss-tree.md) | OEE and the Loss Tree | Availability, performance, quality, and the losses hidden by the average. | 1,200–1,800 | planned |
| 10 | [`10_bottlenecks-lean-toc.md`](10_bottlenecks-lean-toc.md) | Bottlenecks, Little's Law, Lean, JIT, Kanban, CONWIP, TOC | The constraint logic behind flow and inventory. | 1,200–1,800 | planned |
| 11 | [`11_spc-cpk.md`](11_spc-cpk.md) | SPC, Cp/Cpk, and Defect Analytics | Control charts, capability, and the difference between common and special cause. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Planning Cascade ->
02 The Manufacturing Master Data Model ->
03 Master Production Scheduling ->
04 MRP Logic and a Worked Explosion ->
05 Capacity Requirements Planning ->
06 Finite Scheduling ->
07 Lot Sizing: Four Methods Compared ->
08 Sequence-Dependent Changeovers and SMED ->
09 OEE and the Loss Tree ->
10 Bottlenecks, Little's Law, Lean, JIT, Kanban, CONWIP, TOC ->
11 SPC, Cp/Cpk, and Defect Analytics
```

## 5. What this module does not cover

- It does not cover plant layout design or detailed maintenance engineering; those are specialised engineering decisions.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 11 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`15_manufacturing_production_planning.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
