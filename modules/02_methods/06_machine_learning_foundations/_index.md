# Module 06 — Machine Learning Foundations

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/06_machine_learning_foundations.md`](../../../course/06_machine_learning_foundations.md)
> **Folder:** `modules/02_methods/06_machine_learning_foundations/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches prediction as a contract: a decision, a time zero, a label, features that exist at that time, a cost, and a calibrated output that changes an action.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01A, 03, 04 |
| Unlocks | 07, 08, 18, 20 |
| Assessment | Calibrated prediction model (Week 8) |
| Survey file | `06_machine_learning_foundations.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_prediction-contract.md`](01_prediction-contract.md) | The Prediction Contract | Decision, time zero, label, features, action, and cost written before modelling. | 1,200–1,800 | planned |
| 2 | [`02_labels-time-zero.md`](02_labels-time-zero.md) | Labels and Time Zero | Constructing outcomes without look-ahead and defining the moment of prediction. | 1,200–1,800 | planned |
| 3 | [`03_leakage-taxonomy.md`](03_leakage-taxonomy.md) | Leakage Taxonomy | Target, temporal, entity, and preprocessing leakage with supply chain examples. | 1,200–1,800 | planned |
| 4 | [`04_splits-validation.md`](04_splits-validation.md) | Splits and Validation | Random, temporal, grouped, and rolling-origin splits and when each is honest. | 1,200–1,800 | planned |
| 5 | [`05_preprocessing-fit-scope.md`](05_preprocessing-fit-scope.md) | Preprocessing Pipelines and Fit Scope | Fitting transforms inside the training fold only. | 1,200–1,800 | planned |
| 6 | [`06_model-families.md`](06_model-families.md) | Model Families | Linear, tree, gradient-boosted, and nearest-neighbour models as decision tools. | 1,200–1,800 | planned |
| 7 | [`07_class-imbalance.md`](07_class-imbalance.md) | Class Imbalance and Cost-Sensitive Learning | Why resampling is often the wrong first move. | 1,200–1,800 | planned |
| 8 | [`08_metrics-calibration.md`](08_metrics-calibration.md) | Metrics and Calibration | ROC, PR, Brier, reliability curves, and the cost of miscalibration. | 1,200–1,800 | planned |
| 9 | [`09_decision-curves.md`](09_decision-curves.md) | Decision Curves and Threshold Selection | Choosing a threshold from asymmetric over- and under-prediction costs. | 1,200–1,800 | planned |
| 10 | [`10_fairness-diagnostics.md`](10_fairness-diagnostics.md) | Fairness and Model Diagnostics | Subgroup error, proxy variables, and the limits of a single fairness metric. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Prediction Contract ->
02 Labels and Time Zero ->
03 Leakage Taxonomy ->
04 Splits and Validation ->
05 Preprocessing Pipelines and Fit Scope ->
06 Model Families ->
07 Class Imbalance and Cost-Sensitive Learning ->
08 Metrics and Calibration ->
09 Decision Curves and Threshold Selection ->
10 Fairness and Model Diagnostics
```

## 5. What this module does not cover

- It does not cover deep learning, forecasting-specific validation, or deployment engineering; see Modules 18, 07, and 30.
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

See section 14 of the survey file (`06_machine_learning_foundations.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
