# Module 32 — Food, Agriculture, and Perishable Supply Chains

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/32_food_agriculture_perishables.md`](../../../course/32_food_agriculture_perishables.md)
> **Folder:** `modules/06_domain_electives/32_food_agriculture_perishables/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches how to plan a supply chain where the product decays: harvest and yield, perishable forecasting, shelf-life-aware inventory, cold chain, traceability, and food-waste economics.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 07, 08, 10, 21 |
| Unlocks | 33, 37, 38 |
| Assessment | Perishable inventory and waste study (Week 35) |
| Survey file | `32_food_agriculture_perishables.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_agri-production-harvest.md`](01_agri-production-harvest.md) | Agricultural Production and Harvest Planning | Yield, seasonality, contracts, and the farm-to-supply link. | 1,200–1,800 | planned |
| 2 | [`02_perishable-forecasting.md`](02_perishable-forecasting.md) | Forecasting Perishable Demand | Short horizons, weather, freshness, and promotion effects. | 1,200–1,800 | planned |
| 3 | [`03_shelf-life-fefo.md`](03_shelf-life-fefo.md) | Shelf Life, FEFO, and Age-Based Inventory | Remaining shelf life as a state variable and FEFO policy. | 1,200–1,800 | planned |
| 4 | [`04_newsvendor-salvage.md`](04_newsvendor-salvage.md) | Newsvendor with Salvage and Freshness Pricing | Order quantity when unsold stock has salvage value and markdown. | 1,200–1,800 | planned |
| 5 | [`05_cold-chain-quality.md`](05_cold-chain-quality.md) | Cold Chain and Quality | Temperature, MKT, excursions, and quality degradation. | 1,200–1,800 | planned |
| 6 | [`06_traceability-food-safety.md`](06_traceability-food-safety.md) | Traceability and Food Safety | HACCP, FSMA 204, EPCIS, and lot-level trace. | 1,200–1,800 | planned |
| 7 | [`07_food-waste-donation.md`](07_food-waste-donation.md) | Food Waste and Donation | Waste hierarchy, measurement, and donation economics. | 1,200–1,800 | planned |
| 8 | [`08_food-service-quick-commerce.md`](08_food-service-quick-commerce.md) | Food Service and Quick Commerce | Restaurant supply, dark stores, and meal delivery. | 1,200–1,800 | planned |
| 9 | [`09_sourcing-contracts-yield-risk.md`](09_sourcing-contracts-yield-risk.md) | Sourcing Contracts and Yield Risk | Grower contracts, price mechanisms, and supply risk. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Agricultural Production and Harvest Planning ->
02 Forecasting Perishable Demand ->
03 Shelf Life, FEFO, and Age-Based Inventory ->
04 Newsvendor with Salvage and Freshness Pricing ->
05 Cold Chain and Quality ->
06 Traceability and Food Safety ->
07 Food Waste and Donation ->
08 Food Service and Quick Commerce ->
09 Sourcing Contracts and Yield Risk
```

## 5. What this module does not cover

- It does not cover farm agronomy, crop science, or animal husbandry.
- Food chemistry and microbiological risk assessment require specialist expertise.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: food-safety regulation is educational and requires professional review.

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

See section 14 of the survey file (`32_food_agriculture_perishables.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
