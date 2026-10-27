# Module 14 — Procurement, Supplier Risk, and Spend Analytics

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/14_procurement_supplier_risk_spend.md`](../../../course/14_procurement_supplier_risk_spend.md)
> **Folder:** `modules/02_methods/14_procurement_supplier_risk_spend/`
> **Version:** 2026.5

---

## 1. Module purpose

This module turns procurement data into sourcing decisions: spend cubes, category strategy, should-cost, TCO, sourcing optimization, scorecards, and multi-tier risk.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01 |
| Unlocks | 14A |
| Assessment | Supplier risk scorecard (Week 16) |
| Survey file | `14_procurement_supplier_risk_spend.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_source-to-pay.md`](01_source-to-pay.md) | Source-to-Pay | Requisition, sourcing, contract, purchase order, receipt, invoice, and payment. | 1,200–1,800 | planned |
| 2 | [`02_spend-cube.md`](02_spend-cube.md) | Spend Cube Construction | Supplier, category, business unit, and the reconciliation to the general ledger. | 1,200–1,800 | planned |
| 3 | [`03_classification-taxonomies.md`](03_classification-taxonomies.md) | Classification: UNSPSC, GS1 GPC, and Internal Taxonomies | Why the same spend lands in different buckets. | 1,200–1,800 | planned |
| 4 | [`04_kraljic-segmentation.md`](04_kraljic-segmentation.md) | Kraljic Segmentation and Sourcing Strategy | Leverage, strategic, bottleneck, and non-critical categories. | 1,200–1,800 | planned |
| 5 | [`05_should-cost-tco.md`](05_should-cost-tco.md) | Should-Cost and Total Cost of Ownership | Price versus landed, quality, risk, and switching cost. | 1,200–1,800 | planned |
| 6 | [`06_sourcing-optimization.md`](06_sourcing-optimization.md) | Sourcing Optimization | Award allocation under capacity, minimum volume, and risk constraints. | 1,200–1,800 | planned |
| 7 | [`07_auctions-negotiations.md`](07_auctions-negotiations.md) | Auctions and Negotiations | When competitive bidding helps and when it destroys value. | 1,200–1,800 | planned |
| 8 | [`08_supplier-scorecards.md`](08_supplier-scorecards.md) | Supplier Scorecards | Quality, delivery, cost, responsiveness, and the gaming they invite. | 1,200–1,800 | planned |
| 9 | [`09_supplier-risk.md`](09_supplier-risk.md) | Supplier Risk: Financial and Operational | Concentration, solvency, geography, and single-source exposure. | 1,200–1,800 | planned |
| 10 | [`10_multi-tier-contracts.md`](10_multi-tier-contracts.md) | Multi-Tier Mapping and Contract Data | Tier-n visibility, clauses, and obligations as data. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Source-to-Pay ->
02 Spend Cube Construction ->
03 Classification: UNSPSC, GS1 GPC, and Internal Taxonomies ->
04 Kraljic Segmentation and Sourcing Strategy ->
05 Should-Cost and Total Cost of Ownership ->
06 Sourcing Optimization ->
07 Auctions and Negotiations ->
08 Supplier Scorecards ->
09 Supplier Risk: Financial and Operational ->
10 Multi-Tier Mapping and Contract Data
```

## 5. What this module does not cover

- It does not cover contract law or detailed trade compliance; legal review is required for both.
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

See section 14 of the survey file (`14_procurement_supplier_risk_spend.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
