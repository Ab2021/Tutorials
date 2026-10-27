# Module 07 — Time Series Forecasting Foundations

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
> **Folder:** `modules/02_methods/07_time_series_forecasting/`
> **Version:** 2026.5

---

## 1. Module purpose

This module builds the forecasting baseline: benchmarks first, then exponential smoothing, ARIMA, regression with calendar effects, combination, rolling-origin backtesting, and honest error metrics.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01, 01A, 06 |
| Unlocks | 08, 09, 10, 21, 22, 24 |
| Assessment | Forecast benchmark report (Week 9) |
| Survey file | `07_time_series_forecasting.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md) | Forecastability and Benchmarks First | Naive and seasonal-naive baselines and the share of series that are forecastable. | 1,200–1,800 | planned |
| 2 | [`02_exponential-smoothing.md`](02_exponential-smoothing.md) | Exponential Smoothing | Level, trend, seasonality, damping, and the state-space view. | 1,200–1,800 | planned |
| 3 | [`03_arima-sarimax.md`](03_arima-sarimax.md) | ARIMA, SARIMA, and SARIMAX | Differencing, orders, seasonality, and exogenous regressors. | 1,200–1,800 | planned |
| 4 | [`04_regression-lags-fourier.md`](04_regression-lags-fourier.md) | Regression with Lags, Fourier Terms, and Exogenous Variables | Calendar and causal features without leaking the future. | 1,200–1,800 | planned |
| 5 | [`05_hierarchy-aggregation.md`](05_hierarchy-aggregation.md) | Hierarchy and Aggregation | Where to forecast and how aggregation changes error. | 1,200–1,800 | planned |
| 6 | [`06_intermittent-demand.md`](06_intermittent-demand.md) | Intermittent Demand | Croston, SBA, TSB, and classification before method choice. | 1,200–1,800 | planned |
| 7 | [`07_forecast-combination.md`](07_forecast-combination.md) | Forecast Combination | Why averaging beats model selection more often than expected. | 1,200–1,800 | planned |
| 8 | [`08_rolling-origin-backtesting.md`](08_rolling-origin-backtesting.md) | Rolling-Origin Backtesting | Expanding and sliding origins, horizon effects, and test discipline. | 1,200–1,800 | planned |
| 9 | [`09_error-metrics-cost.md`](09_error-metrics-cost.md) | Error Metrics and Their Cost Interpretation | Choosing a metric that maps to the decision's cost. | 1,200–1,800 | planned |
| 10 | [`10_overrides-fva.md`](10_overrides-fva.md) | Overrides and Forecast Value Added | Measuring whether a planner's adjustment improves the forecast. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Forecastability and Benchmarks First ->
02 Exponential Smoothing ->
03 ARIMA, SARIMA, and SARIMAX ->
04 Regression with Lags, Fourier Terms, and Exogenous Variables ->
05 Hierarchy and Aggregation ->
06 Intermittent Demand ->
07 Forecast Combination ->
08 Rolling-Origin Backtesting ->
09 Error Metrics and Their Cost Interpretation ->
10 Overrides and Forecast Value Added
```

## 5. What this module does not cover

- It does not cover hierarchy reconciliation, intermittent demand at depth, or probabilistic scoring; those are Module 08.
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

See section 14 of the survey file (`07_time_series_forecasting.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
