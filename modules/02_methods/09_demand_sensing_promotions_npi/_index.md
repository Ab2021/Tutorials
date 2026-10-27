# Module 09 — Demand Sensing, Promotions, and New Product Forecasting

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
> **Folder:** `modules/02_methods/09_demand_sensing_promotions_npi/`
> **Version:** 2026.5

---

## 1. Module purpose

This module separates signal from noise in short-horizon demand, measures promotion incrementality rather than promoted sales, and forecasts products that have no history.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 05, 07, 08 |
| Unlocks | 16 |
| Assessment | Demand-sensing evaluation (Week 11) |
| Survey file | `09_demand_sensing_promotions_npi.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md) | Demand Sensing versus Forecasting | Which signals have lead time, which are lagging, and which are noise. | 1,200–1,800 | planned |
| 2 | [`02_sales-decomposition.md`](02_sales-decomposition.md) | Sales Decomposition | Baseline, seasonality, promotion, distribution, and price separated explicitly. | 1,200–1,800 | planned |
| 3 | [`03_promotion-mechanics.md`](03_promotion-mechanics.md) | Promotion Mechanics and Calendar Data | Feature, display, TPR, coupon, and the data each generates. | 1,200–1,800 | planned |
| 4 | [`04_baseline-incrementality.md`](04_baseline-incrementality.md) | Baseline and Incrementality | Counterfactual baselines and why promoted sales are not lift. | 1,200–1,800 | planned |
| 5 | [`05_cannibalization-halo.md`](05_cannibalization-halo.md) | Cannibalization, Halo, and Pull-Forward | The three adjustments that turn lift into profit. | 1,200–1,800 | planned |
| 6 | [`06_price-display-effects.md`](06_price-display-effects.md) | Price and Display Effects | Elasticity, threshold effects, and visual attention as confounders. | 1,200–1,800 | planned |
| 7 | [`07_causal-ml-uplift.md`](07_causal-ml-uplift.md) | Causal Machine Learning and Uplift | Heterogeneous treatment effects and who responds to a promotion. | 1,200–1,800 | planned |
| 8 | [`08_npi-forecasting.md`](08_npi-forecasting.md) | New Product Forecasting | Analogy, attribute, diffusion, and distribution-led methods before launch. | 1,200–1,800 | planned |
| 9 | [`09_distribution-assortment.md`](09_distribution-assortment.md) | Assortment and Distribution Effects | Availability and facings as drivers that dominate early demand. | 1,200–1,800 | planned |
| 10 | [`10_post-launch-learning.md`](10_post-launch-learning.md) | Post-Launch Learning and ROI | Updating the forecast with early sales and settling the ROI question. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Demand Sensing versus Forecasting ->
02 Sales Decomposition ->
03 Promotion Mechanics and Calendar Data ->
04 Baseline and Incrementality ->
05 Cannibalization, Halo, and Pull-Forward ->
06 Price and Display Effects ->
07 Causal Machine Learning and Uplift ->
08 New Product Forecasting ->
09 Assortment and Distribution Effects ->
10 Post-Launch Learning and ROI
```

## 5. What this module does not cover

- It does not design the promotional calendar or set price; Modules 16 and 17 own those decisions.
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

See section 14 of the survey file (`09_demand_sensing_promotions_npi.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
