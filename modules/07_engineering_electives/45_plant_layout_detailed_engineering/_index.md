# Module 45 — Plant Layout, Facility Design, and Detailed Engineering Interfaces

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/45_plant_layout_detailed_engineering.md`](../../../course/45_plant_layout_detailed_engineering.md)
> **Folder:** `modules/07_engineering_electives/45_plant_layout_detailed_engineering/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches the translation from flow and volume to physical facility: layout planning, storage and dock capacity, material handling, simulation validation, engineering documentation, revision control, and safety.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 10, 15, 18, 20 |
| Unlocks | — (capstone domain elective) |
| Assessment | Facility design and engineering interface pack (Week 44) |
| Survey file | `45_plant_layout_detailed_engineering.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_facility-types-flow.md`](01_facility-types-flow.md) | Facility Types and Flow | DC, cross-dock, cold store, plant, and micro-fulfilment. | 1,200–1,800 | planned |
| 2 | [`02_systematic-layout-planning.md`](02_systematic-layout-planning.md) | Systematic Layout Planning | From–to charts, relationship charts, and block layouts. | 1,200–1,800 | planned |
| 3 | [`03_capacity-space-calculation.md`](03_capacity-space-calculation.md) | Capacity and Space Calculation | Positions, levels, aisle factors, and area. | 1,200–1,800 | planned |
| 4 | [`04_dock-staging.md`](04_dock-staging.md) | Dock and Staging Design | Door-hours, utilisation, and peak arrivals. | 1,200–1,800 | planned |
| 5 | [`05_handling-equipment-automation.md`](05_handling-equipment-automation.md) | Handling Equipment and Automation | Aisle geometry, AS/RS, and payback. | 1,200–1,800 | planned |
| 6 | [`06_engineering-documents-pid.md`](06_engineering-documents-pid.md) | Engineering Documents and P&ID | P&IDs, isometrics, line lists, and tag numbering. | 1,200–1,800 | planned |
| 7 | [`07_mto-revision-control.md`](07_mto-revision-control.md) | MTO and Revision Control | Material take-off, reconciliation, and change propagation. | 1,200–1,800 | planned |
| 8 | [`08_safety-fire-ergonomics.md`](08_safety-fire-ergonomics.md) | Safety, Fire, and Ergonomics | Egress, sprinklers, segregation, and traffic. | 1,200–1,800 | planned |
| 9 | [`09_layout-simulation.md`](09_layout-simulation.md) | Layout Simulation | Discrete-event validation and bottleneck diagnosis. | 1,200–1,800 | planned |
| 10 | [`10_design-for-supply-chain.md`](10_design-for-supply-chain.md) | Design for Supply Chain | Standardisation, modularity, transport, and maintenance. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Facility Types and Flow ->
02 Systematic Layout Planning ->
03 Capacity and Space Calculation ->
04 Dock and Staging Design ->
05 Handling Equipment and Automation ->
06 Engineering Documents and P&ID ->
07 MTO and Revision Control ->
08 Safety, Fire, and Ergonomics ->
09 Layout Simulation ->
10 Design for Supply Chain
```

## 5. What this module does not cover

- It does not cover structural, civil, mechanical, or electrical detailed design.
- It does not cover construction management, commissioning, or contractor procurement.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: building codes, fire safety, and process safety require professional review.

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

See section 14 of the survey file (`45_plant_layout_detailed_engineering.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
