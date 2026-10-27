# Module 04 — Statistics and Causal Inference

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
> **Folder:** `modules/01_foundations/04_statistics_causal_inference/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches the difference between prediction and causation and gives the designs an analyst needs to estimate what a price change, promotion, or process change actually caused.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01A, 02, 03 |
| Unlocks | 05, 16, 26 |
| Assessment | Causal demand protocol (Week 6) |
| Survey file | `04_statistics_causal_inference.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_estimands-question-types.md`](01_estimands-question-types.md) | Estimands and Question Types | Descriptive, predictive, and causal questions and the estimand each implies. | 1,200–1,800 | planned |
| 2 | [`02_probability-bootstrap.md`](02_probability-bootstrap.md) | Probability, Sampling, Bootstrap, and Intervals | Uncertainty that survives non-normal, dependent supply chain data. | 1,200–1,800 | planned |
| 3 | [`03_hypothesis-tests.md`](03_hypothesis-tests.md) | Hypothesis Tests and Multiple Comparisons | p-values, effect sizes, and the false discovery problem in metric dashboards. | 1,200–1,800 | planned |
| 4 | [`04_regression-time-series.md`](04_regression-time-series.md) | Regression with Time Series, Seasonality, and Holidays | Fourier terms, holiday effects, and distributed lags. | 1,200–1,800 | planned |
| 5 | [`05_autocorrelation-robust-errors.md`](05_autocorrelation-robust-errors.md) | Autocorrelation, Heteroskedasticity, and Robust Errors | Why naive standard errors overstate confidence. | 1,200–1,800 | planned |
| 6 | [`06_dags.md`](06_dags.md) | DAGs: Confounding, Colliders, Mediators, Bad Controls | Drawing the causal story before choosing the regression. | 1,200–1,800 | planned |
| 7 | [`07_experiments-quasi.md`](07_experiments-quasi.md) | Experiments versus Quasi-Experiments | When randomisation is possible and what to do when it is not. | 1,200–1,800 | planned |
| 8 | [`08_difference-in-differences.md`](08_difference-in-differences.md) | Worked Example: Difference-in-Differences | Parallel trends, event time, and a credible counterfactual. | 1,200–1,800 | planned |
| 9 | [`09_elasticity-endogeneity.md`](09_elasticity-endogeneity.md) | Worked Example: Price Elasticity and Endogeneity | Simultaneity, omitted variables, and instrument or panel strategies. | 1,200–1,800 | planned |
| 10 | [`10_sensitivity-omitted-bias.md`](10_sensitivity-omitted-bias.md) | Sensitivity and Omitted-Variable Bias | How fragile is the estimate, and what would overturn it? | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Estimands and Question Types ->
02 Probability, Sampling, Bootstrap, and Intervals ->
03 Hypothesis Tests and Multiple Comparisons ->
04 Regression with Time Series, Seasonality, and Holidays ->
05 Autocorrelation, Heteroskedasticity, and Robust Errors ->
06 DAGs: Confounding, Colliders, Mediators, Bad Controls ->
07 Experiments versus Quasi-Experiments ->
08 Worked Example: Difference-in-Differences ->
09 Worked Example: Price Elasticity and Endogeneity ->
10 Sensitivity and Omitted-Variable Bias
```

## 5. What this module does not cover

- It does not run a field experiment; Module 05 covers design, power, and operational execution.
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

See section 14 of the survey file (`04_statistics_causal_inference.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
