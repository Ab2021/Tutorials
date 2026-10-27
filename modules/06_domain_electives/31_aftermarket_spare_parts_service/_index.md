# Module 31 — Aftermarket, Spare Parts, and Service Supply Chains

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
> **Folder:** `modules/06_domain_electives/31_aftermarket_spare_parts_service/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches how to forecast failure demand, set spares policy for installed assets, design a repair loop, and run field service under contract — the supply chain that starts after the sale.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06, 08, 10, 15 |
| Unlocks | 33, 34, 37 |
| Assessment | Service spares and availability study (Week 34) |
| Survey file | `31_aftermarket_spare_parts_service.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_installed-base-failure-forecast.md`](01_installed-base-failure-forecast.md) | Installed Base and Failure Demand | Asset register, exposure, failure rate, and why service demand is not retail demand. | 1,200–1,800 | planned |
| 2 | [`02_criticality-spares-classification.md`](02_criticality-spares-classification.md) | Criticality and Spares Classification | Consequence, FMEA, VED, and the difference between expensive and critical. | 1,200–1,800 | planned |
| 3 | [`03_multi-echelon-sparing.md`](03_multi-echelon-sparing.md) | Multi-Echelon Sparing and METRIC | Where to hold spares across depots, repair shops, and field locations. | 1,200–1,800 | planned |
| 4 | [`04_repair-loop-turnaround.md`](04_repair-loop-turnaround.md) | Repair Loops and Turnaround | Repairable parts, pipeline stock, and the economics of repair capacity. | 1,200–1,800 | planned |
| 5 | [`05_field-service-scheduling.md`](05_field-service-scheduling.md) | Field Service Scheduling | Dispatch, routing, skills, first-time fix, and the service window. | 1,200–1,800 | planned |
| 6 | [`06_service-contracts-sla.md`](06_service-contracts-sla.md) | Service Contracts and SLAs | Availability, response time, power-by-the-hour, and penalty design. | 1,200–1,800 | planned |
| 7 | [`07_no-fault-found-returns.md`](07_no-fault-found-returns.md) | No-Fault-Found and Service Returns | Test, disposition, and the hidden cost of unverified failures. | 1,200–1,800 | planned |
| 8 | [`08_connected-assets-predictive-service.md`](08_connected-assets-predictive-service.md) | Connected Assets and Predictive Service | Telemetry, condition monitoring, and when prediction beats stocking. | 1,200–1,800 | planned |
| 9 | [`09_service-network-design.md`](09_service-network-design.md) | Service Network Design | Depot location, coverage, and the cost of response time. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Installed Base and Failure Demand ->
02 Criticality and Spares Classification ->
03 Multi-Echelon Sparing and METRIC ->
04 Repair Loops and Turnaround ->
05 Field Service Scheduling ->
06 Service Contracts and SLAs ->
07 No-Fault-Found and Service Returns ->
08 Connected Assets and Predictive Service ->
09 Service Network Design
```

## 5. What this module does not cover

- It does not cover consumer warranty administration or call-centre operations.
- Detailed reliability engineering and Weibull estimation are covered in Module 06.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: contract content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 9 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`31_aftermarket_spare_parts_service.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
