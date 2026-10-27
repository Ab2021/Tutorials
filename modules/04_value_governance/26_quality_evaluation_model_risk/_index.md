# Module 26 — Quality, Evaluation, and Model Risk

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/26_quality_evaluation_model_risk.md`](../../../course/26_quality_evaluation_model_risk.md)
> **Folder:** `modules/04_value_governance/26_quality_evaluation_model_risk/`
> **Version:** 2026.5

---

## 1. Module purpose

This module governs the model after it exists: inventory and tiering, conceptual soundness, validation independence, model cards, monitoring, revalidation, and vendor model review.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06, 07, 25 |
| Unlocks | 28, 30 |
| Assessment | Model card and validation dossier (Week 32) |
| Survey file | `26_quality_evaluation_model_risk.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_model-inventory-tiering.md`](01_model-inventory-tiering.md) | Model Inventory and Tiering | What counts as a model, and how materiality sets the control burden. | 1,200–1,800 | planned |
| 2 | [`02_conceptual-soundness.md`](02_conceptual-soundness.md) | Conceptual Soundness | Does the method fit the decision, the data, and the failure cost? | 1,200–1,800 | planned |
| 3 | [`03_data-representativeness.md`](03_data-representativeness.md) | Data Quality and Representativeness | Coverage, drift, and the population the model was never tested on. | 1,200–1,800 | planned |
| 4 | [`04_outcomes-backtesting.md`](04_outcomes-backtesting.md) | Outcomes Analysis and Backtesting | Temporal validation, benchmark comparison, and challenger models. | 1,200–1,800 | planned |
| 5 | [`05_planning-model-metrics.md`](05_planning-model-metrics.md) | Metrics That Matter for Planning Models | Accuracy, bias, calibration, stability, and decision cost. | 1,200–1,800 | planned |
| 6 | [`06_model-documentation.md`](06_model-documentation.md) | Model Documentation and the Model Card | Purpose, limitations, assumptions, owners, and approved use. | 1,200–1,800 | planned |
| 7 | [`07_reproducibility-audit.md`](07_reproducibility-audit.md) | Reproducibility and Audit Trail | Code, data, parameters, and the ability to recreate a past result. | 1,200–1,800 | planned |
| 8 | [`08_validation-independence.md`](08_validation-independence.md) | Independence and Validation | Who may validate, and the conflicts that disqualify a reviewer. | 1,200–1,800 | planned |
| 9 | [`09_monitoring-revalidation.md`](09_monitoring-revalidation.md) | Ongoing Monitoring and Revalidation Triggers | Drift, incident, market change, and scheduled review. | 1,200–1,800 | planned |
| 10 | [`10_vendor-model-evaluation.md`](10_vendor-model-evaluation.md) | Vendor Model Evaluation and SR 11-7 | Evidence to demand from a vendor and the limits of a black-box review. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Model Inventory and Tiering ->
02 Conceptual Soundness ->
03 Data Quality and Representativeness ->
04 Outcomes Analysis and Backtesting ->
05 Metrics That Matter for Planning Models ->
06 Model Documentation and the Model Card ->
07 Reproducibility and Audit Trail ->
08 Independence and Validation ->
09 Ongoing Monitoring and Revalidation Triggers ->
10 Vendor Model Evaluation and SR 11-7
```

## 5. What this module does not cover

- It does not provide a legal opinion or certify compliance with SR 11-7 or any regulator.
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

See section 14 of the survey file (`26_quality_evaluation_model_risk.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
