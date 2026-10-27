# Module 41 — Maritime, Rail, and Air Cargo Operations Engineering

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/41_maritime_rail_air_operations.md`](../../../course/41_maritime_rail_air_operations.md)
> **Folder:** `modules/07_engineering_electives/41_maritime_rail_air_operations/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches the physical operations beneath global freight: berth and quay crane capacity, terminal yards and gates, rail paths and shunting, air cargo build-up and ULDs, hinterland evacuation, and disruption recovery.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 11, 12, 20, 36 |
| Unlocks | — (capstone domain elective) |
| Assessment | Terminal or corridor operations plan (Week 41) |
| Survey file | `41_maritime_rail_air_operations.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_berth-quay-crane.md`](01_berth-quay-crane.md) | Berth and Quay Crane Operations | Crane productivity, berth allocation, and vessel turnaround. | 1,200–1,800 | planned |
| 2 | [`02_terminal-yard-gate.md`](02_terminal-yard-gate.md) | Terminal Yard and Gate | Stack allocation, dwell, rehandles, and gate flow. | 1,200–1,800 | planned |
| 3 | [`03_rail-network-paths.md`](03_rail-network-paths.md) | Rail Network and Path Capacity | Timetables, headway, signalling, and mixed traffic. | 1,200–1,800 | planned |
| 4 | [`04_rail-yards-shunting.md`](04_rail-yards-shunting.md) | Rail Yards and Shunting | Marshalling, block trains, and wagon cycles. | 1,200–1,800 | planned |
| 5 | [`05_air-cargo-uld.md`](05_air-cargo-uld.md) | Air Cargo and ULD | Build-up, breakdown, cutoffs, and belly capacity. | 1,200–1,800 | planned |
| 6 | [`06_intermodal-hinterland.md`](06_intermodal-hinterland.md) | Intermodal and Hinterland | Rail, road, barge evacuation and port connectivity. | 1,200–1,800 | planned |
| 7 | [`07_disruption-recovery.md`](07_disruption-recovery.md) | Disruption and Recovery | Queue sequencing, priorities, and recovery policies. | 1,200–1,800 | planned |
| 8 | [`08_terminal-simulation.md`](08_terminal-simulation.md) | Terminal Simulation | Discrete-event modelling of quay, yard, and gate. | 1,200–1,800 | planned |
| 9 | [`09_dg-security-operations.md`](09_dg-security-operations.md) | Dangerous Goods and Security Operations | Segregation, screening, ISPS, and access control. | 1,200–1,800 | planned |
| 10 | [`10_automation-capital.md`](10_automation-capital.md) | Automation and Capital Decisions | Terminal automation, utilisation, and payback. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Berth and Quay Crane ->
02 Terminal Yard and Gate ->
03 Rail Network and Path Capacity ->
04 Rail Yards and Shunting ->
05 Air Cargo and ULD ->
06 Intermodal and Hinterland ->
07 Disruption and Recovery ->
08 Terminal Simulation ->
09 Dangerous Goods and Security ->
10 Automation and Capital Decisions
```

## 5. What this module does not cover

- It does not cover vessel naval architecture, aircraft engineering, or rail vehicle design.
- It does not cover shipping commercial terms or Incoterms: see Module 36.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: safety, dangerous goods, and security content requires professional review.

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

See section 14 of the survey file (`41_maritime_rail_air_operations.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
