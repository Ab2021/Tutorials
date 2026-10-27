# Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/22_quick_commerce_on_demand_delivery.md`](../../../course/22_quick_commerce_on_demand_delivery.md)
> **Folder:** `modules/03_commerce/22_quick_commerce_on_demand_delivery/`
> **Version:** 2026.5

---

## 1. Module purpose

This module covers the fastest promise in supply chain: dark stores, short-horizon forecasting, perishable replenishment, courier supply and demand, dispatch, ETA, and unit economics.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 10, 12, 21 |
| Unlocks | 23, 25 |
| Assessment | Quick-commerce operating model (Week 28) |
| Survey file | `22_quick_commerce_on_demand_delivery.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_promise-tiers-dark-stores.md`](01_promise-tiers-dark-stores.md) | Promise Tiers and Dark-Store Design | 10-minute, 30-minute, and scheduled promises and the footprint each needs. | 1,200–1,800 | planned |
| 2 | [`02_short-horizon-forecasting.md`](02_short-horizon-forecasting.md) | Demand Forecasting at Short Horizons | Hourly and slot-level demand with weather, events, and promotions. | 1,200–1,800 | planned |
| 3 | [`03_dark-store-assortment.md`](03_dark-store-assortment.md) | Assortment for Dark Stores | Breadth versus depth under a hard space constraint. | 1,200–1,800 | planned |
| 4 | [`04_replenishment-waste.md`](04_replenishment-waste.md) | Replenishment, Shelf Life, and Waste | FEFO, expiry risk, and the cost of a stockout versus a write-off. | 1,200–1,800 | planned |
| 5 | [`05_courier-supply-demand.md`](05_courier-supply-demand.md) | Courier Supply and Demand | Incentives, utilization, and the marketplace that clears the market. | 1,200–1,800 | planned |
| 6 | [`06_eta-prediction.md`](06_eta-prediction.md) | ETA Prediction | Preparation, travel, handoff, and the uncertainty a customer actually sees. | 1,200–1,800 | planned |
| 7 | [`07_batching-routing.md`](07_batching-routing.md) | Batching and Routing | Multi-order batches, walking time, and the trade-off against lateness. | 1,200–1,800 | planned |
| 8 | [`08_dispatch-matching.md`](08_dispatch-matching.md) | Dispatch and Matching | Assignment policies, fairness, and the cost of a bad match. | 1,200–1,800 | planned |
| 9 | [`09_unit-economics.md`](09_unit-economics.md) | Unit Economics | Basket size, delivery cost, courier pay, and contribution per order. | 1,200–1,800 | planned |
| 10 | [`10_food-safety-welfare.md`](10_food-safety-welfare.md) | Food Safety, Cold Chain, and Courier Welfare | HACCP, temperature control, and the human side of the promise. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Promise Tiers and Dark-Store Design ->
02 Demand Forecasting at Short Horizons ->
03 Assortment for Dark Stores ->
04 Replenishment, Shelf Life, and Waste ->
05 Courier Supply and Demand ->
06 ETA Prediction ->
07 Batching and Routing ->
08 Dispatch and Matching ->
09 Unit Economics ->
10 Food Safety, Cold Chain, and Courier Welfare
```

## 5. What this module does not cover

- It does not cover restaurant operations, app product design, or employment-law advice.
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

See section 14 of the survey file (`22_quick_commerce_on_demand_delivery.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
