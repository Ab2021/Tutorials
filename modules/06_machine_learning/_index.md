# Module 06 — Machine Learning Foundations

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../06_machine_learning_foundations.md`](../../06_machine_learning_foundations.md)
> **Folder:** `modules/06_machine_learning/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 04 |
| Unlocks | 07, 10, 11, 14B |
| Assessment | Model development report (Week 12) |
| Survey file | `06_machine_learning_foundations.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_problem-framing.md`](01_problem-framing.md) | Problem Framing | Prediction versus decision; choosing a target that changes an action. | 1,200-1,800 | planned |
| 2 | [`02_labels-and-outcomes.md`](02_labels-and-outcomes.md) | Labels and Outcomes | Label sources, ascertainment bias, proxy labels, and label noise. | 1,200-1,800 | planned |
| 3 | [`03_splitting-and-validation.md`](03_splitting-and-validation.md) | Splitting and Validation | Patient-level, temporal, and site-level splits; nested cross-validation. | 1,200-1,800 | planned |
| 4 | [`04_preprocessing.md`](04_preprocessing.md) | Preprocessing | Imputation, scaling, encoding, and fitting inside the fold. | 1,200-1,800 | planned |
| 5 | [`05_model-families.md`](05_model-families.md) | Model Families | Linear, tree, gradient-boosted, kernel, and neural approaches with trade-offs. | 1,200-1,800 | planned |
| 6 | [`06_class-imbalance.md`](06_class-imbalance.md) | Class Imbalance | Resampling, class weights, and why AUROC hides the problem. | 1,200-1,800 | planned |
| 7 | [`07_metrics.md`](07_metrics.md) | Metrics | Discrimination, accuracy, sensitivity, PPV, F1, and their dependence on prevalence. | 1,200-1,800 | planned |
| 8 | [`08_calibration.md`](08_calibration.md) | Calibration | Reliability diagrams, slope and intercept, Brier decomposition, and recalibration. | 1,200-1,800 | planned |
| 9 | [`09_thresholds-and-decision-curves.md`](09_thresholds-and-decision-curves.md) | Thresholds and Decision Curves | Net benefit, harm ratios, capacity constraints, and choosing an operating point. | 1,200-1,800 | planned |
| 10 | [`10_interpretability.md`](10_interpretability.md) | Interpretability | Coefficients, partial dependence, SHAP, and the limits of post-hoc explanation. | 1,200-1,800 | planned |
| 11 | [`11_uncertainty.md`](11_uncertainty.md) | Uncertainty | Confidence intervals, conformal prediction, and abstention. | 1,200-1,800 | planned |
| 12 | [`12_fairness-and-subgroups.md`](12_fairness-and-subgroups.md) | Fairness and Subgroups | Subgroup performance, fairness criteria, and the impossibility results. | 1,200-1,800 | planned |
| 13 | [`13_leakage.md`](13_leakage.md) | Leakage | A catalogue of leaks in clinical ML and how to test for each. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Problem Framing ->
02 Labels and Outcomes ->
03 Splitting and Validation ->
04 Preprocessing ->
05 Model Families ->
06 Class Imbalance ->
07 Metrics ->
08 Calibration ->
09 Thresholds and Decision Curves ->
10 Interpretability ->
11 Uncertainty ->
12 Fairness and Subgroups ->
13 Leakage
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 13 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

