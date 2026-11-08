---
collection: "supervised-ml-and-feature-engineering"
file_id: "01"
topic: "Model validation and error analysis"
level: 2
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible directory inventory consulted"
---

# Model Validation and Error Analysis

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as an inventory. This course-topic draft does not reconstruct unavailable source bodies.

## 1. Validation answers a deployment question

Define the decision, prediction time, available information, target horizon, unit of analysis, loss and operational constraints before selecting a split. A high offline score with unavailable-at-prediction features is leakage, not performance.

## 2. Split by dependency

| Dependency at deployment | Split rule | Risk if ignored |
|---|---|---|
| Independent, stationary units | stratified random split where labels need balance | accidental class imbalance |
| Same entity repeats | group split by entity | entity memorization |
| Time evolves | forward/rolling split | future information leaks backward |
| Correlated locations | geography/group split | local correlation overstates transfer |

## 3. Metrics and threshold

For binary classification, `precision=TP/(TP+FP)` and `recall=TP/(TP+FN)`. With `TP=36`, `FP=12`, `FN=24`, precision is `0.75` and recall is `0.60`. Select a threshold from explicit error costs, capacity and calibration—not default `0.5`. Ranking metrics do not prove probability calibration.

## 4. Leakage audit

For every feature, trace source, event time, materialization time, availability at prediction, transformations and joins. Fit imputers, encoders, scalers, feature selection and resampling only on each training fold; apply the fitted transform to its validation fold. Label-derived aggregates require out-of-fold or time-safe construction.

## 5. Error slices

Build slices from use context and credible harms: time, geography, cohort, class, source, missingness, score band and decision boundary. Report support and uncertainty with every slice. A small slice can diagnose a failure but cannot justify a broad claim without enough observations.

## 6. Decision procedure

1. Freeze the prediction contract and baseline.
2. Select split matching dependency and rollout horizon.
3. Train preprocessing and model within folds.
4. Evaluate aggregate, calibration and error slices.
5. Inspect examples and root causes, then revise data/contract/model deliberately.
6. Confirm on an untouched final holdout or next time window.

## 7. Failure modes

| Failure | Signal | Mitigation |
|---|---|---|
| Future leakage | implausibly large offline lift | time/availability audit and rebuild split |
| Group leakage | held-out entity score collapses | group-aware split |
| Metric mismatch | good AUC, unacceptable queue load | optimize cost/capacity threshold |
| Class imbalance hidden | accuracy high, recall negligible | precision/recall and prevalence baseline |
| Slice regression | aggregate stable, cohort harm | minimum support and slice gates |

## 8. Interview checks

1. **Why random split can fail?** It can place correlated future/entity information in both sets.
2. **What is leakage?** Use of information unavailable at the decision time.
3. **Why calibration?** A score used as probability must correspond to observed frequency.
4. **What is a useful slice?** A supported subgroup tied to deployment mechanism or harm.
5. **Why final holdout?** Iteration has otherwise adapted to validation noise.

## 9. Quick Reference Card

Record prediction time, label horizon, unit, source availability, split dependency, baseline, metric/cost, threshold, calibration, slice gates, rollout and monitoring. Revalidate when data, policy, timing or use context changes.

**Limitations:** Example counts are illustrative; no source model, dataset or measured result is claimed.
