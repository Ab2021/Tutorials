# Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
> **Folder:** `modules/02_methods/08_hierarchical_intermittent_probabilistic_forecasting/`
> **Version:** 2026.5

---

## 1. Module purpose

This module makes forecasts coherent and probabilistic: reconciliation across product, location, and time, intermittent-demand methods, predictive distributions, conformal prediction, and proper scoring rules.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 07 |
| Unlocks | 09, 10, 21 |
| Assessment | Probabilistic forecast dossier (Week 10) |
| Survey file | `08_hierarchical_intermittent_probabilistic_forecasting.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_hierarchy-structures.md`](01_hierarchy-structures.md) | Hierarchy Structures | Product, location, customer, and temporal aggregation and the paths between them. | 1,200–1,800 | planned |
| 2 | [`02_aggregation-consistency.md`](02_aggregation-consistency.md) | Aggregation Consistency and the Four Classical Methods | Bottom-up, top-down, middle-out, and optimal reconciliation. | 1,200–1,800 | planned |
| 3 | [`03_mint-reconciliation.md`](03_mint-reconciliation.md) | MinT in Detail | Covariance shrinkage, constraints, and a reference implementation. | 1,200–1,800 | planned |
| 4 | [`04_temporal-cross-temporal.md`](04_temporal-cross-temporal.md) | Temporal and Cross-Temporal Hierarchies | Reconciling across time buckets as well as entities. | 1,200–1,800 | planned |
| 5 | [`05_intermittent-classification.md`](05_intermittent-classification.md) | Intermittent Demand: Classification and Methods | Syntetos-Boylan classification and the methods that fit each class. | 1,200–1,800 | planned |
| 6 | [`06_compound-poisson-hurdle.md`](06_compound-poisson-hurdle.md) | Compound Poisson and Hurdle Models | Separating occurrence from size in slow-moving demand. | 1,200–1,800 | planned |
| 7 | [`07_probabilistic-forecasts.md`](07_probabilistic-forecasts.md) | Probabilistic Forecasts: What a Distribution Is For | Quantiles, intervals, and the decisions they enable. | 1,200–1,800 | planned |
| 8 | [`08_conformal-prediction.md`](08_conformal-prediction.md) | Conformal Prediction for Time Series | Distribution-free coverage under dependence and its limits. | 1,200–1,800 | planned |
| 9 | [`09_scoring-rules.md`](09_scoring-rules.md) | Scoring Probabilistic Forecasts | CRPS, pinball loss, log score, and calibration diagnostics. | 1,200–1,800 | planned |
| 10 | [`10_quantiles-decisions.md`](10_quantiles-decisions.md) | Decision Use of Quantiles | Choosing a quantile from service level, cost, and risk appetite. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Hierarchy Structures ->
02 Aggregation Consistency and the Four Classical Methods ->
03 MinT in Detail ->
04 Temporal and Cross-Temporal Hierarchies ->
05 Intermittent Demand: Classification and Methods ->
06 Compound Poisson and Hurdle Models ->
07 Probabilistic Forecasts: What a Distribution Is For ->
08 Conformal Prediction for Time Series ->
09 Scoring Probabilistic Forecasts ->
10 Decision Use of Quantiles
```

## 5. What this module does not cover

- It does not cover inventory policy design; Module 10 uses the distributions produced here.
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

See section 14 of the survey file (`08_hierarchical_intermittent_probabilistic_forecasting.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
