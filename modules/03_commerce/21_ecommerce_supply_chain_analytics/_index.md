# Module 21 — E-commerce Supply Chain Analytics

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/21_ecommerce_supply_chain_analytics.md`](../../../course/21_ecommerce_supply_chain_analytics.md)
> **Folder:** `modules/03_commerce/21_ecommerce_supply_chain_analytics/`
> **Version:** 2026.5

---

## 1. Module purpose

This module covers the e-commerce fulfillment system: node-level forecasting, inventory placement, the delivery promise, carrier economics, returns, peak, and unit economics per order.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 07, 10, 12 |
| Unlocks | 22, 23, 25 |
| Assessment | E-commerce fulfillment design (Week 27) |
| Survey file | `21_ecommerce_supply_chain_analytics.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_ecommerce-fulfillment-models.md`](01_ecommerce-fulfillment-models.md) | Fulfillment Models | 1P, 3P, dropship, marketplace, and their inventory and service implications. | 1,200–1,800 | planned |
| 2 | [`02_order-grain-forecasting.md`](02_order-grain-forecasting.md) | Demand Forecasting at Order Grain | Day, slot, and node-level forecasting, not just monthly SKU totals. | 1,200–1,800 | planned |
| 3 | [`03_long-tail-forecasting.md`](03_long-tail-forecasting.md) | Long-Tail Forecasting | Millions of slow-moving SKUs and the hierarchy that makes them forecastable. | 1,200–1,800 | planned |
| 4 | [`04_inventory-placement.md`](04_inventory-placement.md) | Inventory Placement | Which node holds which SKU and the split between speed and holding cost. | 1,200–1,800 | planned |
| 5 | [`05_delivery-promise.md`](05_delivery-promise.md) | The Delivery Promise | Promise as a product decision and a constraint on the network. | 1,200–1,800 | planned |
| 6 | [`06_carrier-selection.md`](06_carrier-selection.md) | Shipping Cost and Carrier Selection | Rate cards, zone skipping, dimensional weight, and service mix. | 1,200–1,800 | planned |
| 7 | [`07_returns.md`](07_returns.md) | Returns | Return reasons, rates, and the cost of a reverse logistics loop. | 1,200–1,800 | planned |
| 8 | [`08_peak-planning.md`](08_peak-planning.md) | Peak Planning | Capacity, labour, inventory, and the failure modes unique to peak. | 1,200–1,800 | planned |
| 9 | [`09_unit-economics.md`](09_unit-economics.md) | Unit Economics per Order | Contribution margin after fulfillment, returns, and payment costs. | 1,200–1,800 | planned |
| 10 | [`10_marketplace-cross-border-cx.md`](10_marketplace-cross-border-cx.md) | Marketplace, Cross-Border, and Customer Experience | Customs, duty, delivery experience, and the NPS that follows. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Fulfillment Models ->
02 Demand Forecasting at Order Grain ->
03 Long-Tail Forecasting ->
04 Inventory Placement ->
05 The Delivery Promise ->
06 Shipping Cost and Carrier Selection ->
07 Returns ->
08 Peak Planning ->
09 Unit Economics per Order ->
10 Marketplace, Cross-Border, and Customer Experience
```

## 5. What this module does not cover

- It does not cover marketplace platform engineering, checkout design, or paid-media optimisation.
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

See section 14 of the survey file (`21_ecommerce_supply_chain_analytics.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
