# Module 10 — Inventory Optimization and Multi-Echelon Planning

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
> **Folder:** `modules/02_methods/10_inventory_optimization_multi_echelon/`
> **Version:** 2026.5

---

## 1. Module purpose

This module translates a forecast into an executable policy: EOQ, newsvendor, safety stock, service metrics, multi-echelon optimization, pooling, postponement, and segmentation.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01A, 07, 08 |
| Unlocks | 11, 21, 22, 23 |
| Assessment | Inventory policy model (Week 12) |
| Survey file | `10_inventory_optimization_multi_echelon.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_cost-service-frontier.md`](01_cost-service-frontier.md) | The Cost-Service Frontier | The trade-off curve and the marginal cost of the next point of service. | 1,200–1,800 | planned |
| 2 | [`02_eoq-limits.md`](02_eoq-limits.md) | EOQ and Its Limits | Setup, holding, and the assumptions that rarely hold in practice. | 1,200–1,800 | planned |
| 3 | [`03_newsvendor.md`](03_newsvendor.md) | Newsvendor and the Critical Fractile | Optimal order quantity under a single selling season. | 1,200–1,800 | planned |
| 4 | [`04_safety-stock.md`](04_safety-stock.md) | Safety Stock under Demand and Lead-Time Uncertainty | Combining variability, lead time, and review period. | 1,200–1,800 | planned |
| 5 | [`05_service-metrics.md`](05_service-metrics.md) | Service Metrics: CSL, Fill Rate, OTIF | Definitions that are not interchangeable and the policies they imply. | 1,200–1,800 | planned |
| 6 | [`06_policy-families.md`](06_policy-families.md) | Policy Families | (s,Q), (R,S), min-max, and periodic review in operational terms. | 1,200–1,800 | planned |
| 7 | [`07_lead-time-demand.md`](07_lead-time-demand.md) | Lead-Time Demand Distributions | Compounding demand and lead-time variability correctly. | 1,200–1,800 | planned |
| 8 | [`08_multi-echelon.md`](08_multi-echelon.md) | Multi-Echelon Inventory, MEIO, and DRP | Echelon stock, lead-time demand, DRP deployment, shortage allocation, and where to hold the buffer. | 1,200–1,800 | planned |
| 9 | [`09_pooling-postponement.md`](09_pooling-postponement.md) | Pooling and Postponement | Risk pooling, component commonality, and delayed differentiation. | 1,200–1,800 | planned |
| 10 | [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md) | Segmentation, DDMRP, and Shelf Life | ABC/XYZ, decoupling points, and perishable inventory constraints. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Cost-Service Frontier ->
02 EOQ and Its Limits ->
03 Newsvendor and the Critical Fractile ->
04 Safety Stock under Demand and Lead-Time Uncertainty ->
05 Service Metrics: CSL, Fill Rate, OTIF ->
06 Policy Families ->
07 Lead-Time Demand Distributions ->
08 Multi-Echelon Inventory and MEIO ->
09 Pooling and Postponement ->
10 Segmentation, DDMRP, and Shelf Life
```

## 5. What this module does not cover

- It does not cover warehouse layout or transport routing; Modules 13 and 12 own those constraints.
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

See section 14 of the survey file (`10_inventory_optimization_multi_echelon.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
