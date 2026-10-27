# Module 01A — Foundations: Units, Time, Measurement, and Assumptions

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
> **Folder:** `modules/01_foundations/01A_foundations_units_time_assumptions/`
> **Version:** 2026.5

---

## 1. Module purpose

This module is the quantitative anchor: it makes units, calendars, lead times, error metrics, and assumptions explicit so later models cannot silently mix eaches with cases or weeks with buckets.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 00, 01 |
| Unlocks | 04, 06, 08, 10 |
| Assessment | Assumptions ledger (Week 3) |
| Survey file | `01A_foundations_units_time_assumptions.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_unit-of-measure-hierarchy.md`](01_unit-of-measure-hierarchy.md) | The Unit-of-Measure Hierarchy | Each, inner pack, case, layer, pallet, and the conversions that lose information. | 1,200–1,800 | planned |
| 2 | [`02_container-loading.md`](02_container-loading.md) | Container Loading: Which Constraint Binds? | Weight, cube, floor space, and the minimum of the three. | 1,200–1,800 | planned |
| 3 | [`03_case-pack-moq-rounding.md`](03_case-pack-moq-rounding.md) | Case Pack, MOQ, and Integer Rounding | Why a fractional plan is not executable and how rounding compounds. | 1,200–1,800 | planned |
| 4 | [`04_calendars-weeks-time-zones.md`](04_calendars-weeks-time-zones.md) | Calendars, Weeks, Time Zones, and Buckets | Fiscal calendars, 4-4-5, ISO weeks, cutoffs, and planning buckets. | 1,200–1,800 | planned |
| 5 | [`05_lead-time-definitions.md`](05_lead-time-definitions.md) | Lead-Time Definitions | Nominal, actual, promised, replenishment, and the distribution behind the average. | 1,200–1,800 | planned |
| 6 | [`06_currency-and-fx.md`](06_currency-and-fx.md) | Currency, FX, and Constant Currency | Transaction versus reporting currency and the effect of rate changes on comparisons. | 1,200–1,800 | planned |
| 7 | [`07_hierarchy-design.md`](07_hierarchy-design.md) | Hierarchy Design | Product, location, customer, and time hierarchies that reconcile without double counting. | 1,200–1,800 | planned |
| 8 | [`08_measurement-error-bias.md`](08_measurement-error-bias.md) | Measurement Error, Accuracy, Precision, and Bias | The four words that are not synonyms and the errors they hide. | 1,200–1,800 | planned |
| 9 | [`09_forecast-error-metrics.md`](09_forecast-error-metrics.md) | Forecast Error Metrics | MAE, RMSE, MAPE, sMAPE, MASE, bias, and the cost of each choice. | 1,200–1,800 | planned |
| 10 | [`10_intermittent-metrics.md`](10_intermittent-metrics.md) | Why MAPE Fails on Intermittent Demand | Division by small denominators and the metrics that survive zero demand. | 1,200–1,800 | planned |
| 11 | [`11_demand-lead-time-distributions.md`](11_demand-lead-time-distributions.md) | Demand and Lead-Time Distributions | Normal, gamma, lognormal, Poisson, and compound distributions in planning. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Unit-of-Measure Hierarchy ->
02 Container Loading: Which Constraint Binds? ->
03 Case Pack, MOQ, and Integer Rounding ->
04 Calendars, Weeks, Time Zones, and Buckets ->
05 Lead-Time Definitions ->
06 Currency, FX, and Constant Currency ->
07 Hierarchy Design ->
08 Measurement Error, Accuracy, Precision, and Bias ->
09 Forecast Error Metrics ->
10 Why MAPE Fails on Intermittent Demand ->
11 Demand and Lead-Time Distributions
```

## 5. What this module does not cover

- It does not teach statistical inference or forecasting methods; it supplies the measurement discipline they depend on.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 11 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`01A_foundations_units_time_assumptions.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
