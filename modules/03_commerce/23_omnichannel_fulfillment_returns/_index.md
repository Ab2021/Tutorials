# Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/23_omnichannel_fulfillment_returns.md`](../../../course/23_omnichannel_fulfillment_returns.md)
> **Folder:** `modules/03_commerce/23_omnichannel_fulfillment_returns/`
> **Version:** 2026.5

---

## 1. Module purpose

This module connects stores, DCs, and digital channels: inventory visibility, ATP, order routing, ship-from-store, BOPIS, promise accuracy, returns, and channel P&L.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 13, 21 |
| Unlocks | 25, 26 |
| Assessment | Omnichannel routing policy (Week 29) |
| Survey file | `23_omnichannel_fulfillment_returns.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_inventory-visibility.md`](01_inventory-visibility.md) | Inventory Visibility | Perpetual inventory, phantom stock, and the data quality behind a promise. | 1,200–1,800 | planned |
| 2 | [`02_available-to-promise.md`](02_available-to-promise.md) | Available-to-Promise | ATP, ATS, and the difference between what exists and what can be sold. | 1,200–1,800 | planned |
| 3 | [`03_order-routing-dom.md`](03_order-routing-dom.md) | Order Routing and DOM | Rules, costs, and the optimization behind a routing decision. | 1,200–1,800 | planned |
| 4 | [`04_ship-from-store.md`](04_ship-from-store.md) | Ship-from-Store Economics | Store labour, packaging, inventory risk, and the cost of a marginal order. | 1,200–1,800 | planned |
| 5 | [`05_bopis-curbside.md`](05_bopis-curbside.md) | BOPIS and Curbside | Pick accuracy, wait time, substitution, and the labour model. | 1,200–1,800 | planned |
| 6 | [`06_promise-accuracy.md`](06_promise-accuracy.md) | Promise Accuracy | Measuring the promise a customer saw against what happened. | 1,200–1,800 | planned |
| 7 | [`07_returns-networks.md`](07_returns-networks.md) | Returns Networks | Return-to-store, return-to-DC, and the reverse flow design. | 1,200–1,800 | planned |
| 8 | [`08_disposition-economics.md`](08_disposition-economics.md) | Disposition Economics | Resell, refurbish, liquidate, donate, or destroy, with the margin of each. | 1,200–1,800 | planned |
| 9 | [`09_return-fraud.md`](09_return-fraud.md) | Return Fraud | Wardrobing, empty-box, and the detection limits of policy and data. | 1,200–1,800 | planned |
| 10 | [`10_channel-pnl.md`](10_channel-pnl.md) | Channel P&L | Allocating cost to channel without double counting shared capacity. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Inventory Visibility ->
02 Available-to-Promise ->
03 Order Routing and DOM ->
04 Ship-from-Store Economics ->
05 BOPIS and Curbside ->
06 Promise Accuracy ->
07 Returns Networks ->
08 Disposition Economics ->
09 Return Fraud ->
10 Channel P&L
```

## 5. What this module does not cover

- It does not cover store operations management or point-of-sale engineering.
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

See section 14 of the survey file (`23_omnichannel_fulfillment_returns.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
