# Module 04 — Statistics and Causal Inference

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/04_statistics_causal_inference.md`](../../course/04_statistics_causal_inference.md)
> **Folder:** `modules/04_statistics/`
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
| Prerequisites | 01A |
| Unlocks | 05, 06, 13, 18 |
| Assessment | Causal analysis protocol (Week 9) |
| Survey file | `04_statistics_causal_inference.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_estimation-and-inference.md`](01_estimation-and-inference.md) | Estimation and Inference | Estimands, standard errors, confidence intervals, and what a p-value does not say. | 1,200-1,800 | complete |
| 2 | [`02_regression-models.md`](02_regression-models.md) | Regression Models | Linear, logistic, Poisson, and Cox models as tools for adjustment and prediction. | 1,200-1,800 | complete |
| 3 | [`03_survival-analysis.md`](03_survival-analysis.md) | Survival Analysis | Censoring, hazards, Kaplan-Meier, competing risks, and time-varying confounding. | 1,200-1,800 | complete |
| 4 | [`04_missing-data.md`](04_missing-data.md) | Missing Data | MCAR, MAR, MNAR, multiple imputation, and sensible sensitivity analysis. | 1,200-1,800 | complete |
| 5 | [`05_causal-inference-and-dags.md`](05_causal-inference-and-dags.md) | Causal Inference and DAGs | Confounding, colliders, mediators, and drawing the graph before the model. | 1,200-1,800 | complete |
| 6 | [`06_propensity-scores-and-weighting.md`](06_propensity-scores-and-weighting.md) | Propensity Scores and Weighting | Matching, IPTW, trimming, balance diagnostics, and effective sample size. | 1,200-1,800 | complete |
| 7 | [`07_target-trial-emulation.md`](07_target-trial-emulation.md) | Target Trial Emulation | Specifying eligibility, treatment strategies, assignment, time zero, and outcomes. | 1,200-1,800 | complete |
| 8 | [`08_sensitivity-analysis.md`](08_sensitivity-analysis.md) | Sensitivity Analysis | Negative controls, E-values, quantitative bias analysis, and tipping points. | 1,200-1,800 | complete |
| 9 | [`09_subgroups-and-heterogeneity.md`](09_subgroups-and-heterogeneity.md) | Subgroups and Heterogeneity | Pre-specification, interaction tests, multiplicity, and credible subgroup claims. | 1,200-1,800 | complete |
| 10 | [`10_reporting-statistical-results.md`](10_reporting-statistical-results.md) | Reporting Statistical Results | Effect sizes with intervals, absolute risk, number needed to treat, and honest uncertainty. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Estimation and Inference ->
02 Regression Models ->
03 Survival Analysis ->
04 Missing Data ->
05 Causal Inference and DAGs ->
06 Propensity Scores and Weighting ->
07 Target Trial Emulation ->
08 Sensitivity Analysis ->
09 Subgroups and Heterogeneity ->
10 Reporting Statistical Results
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 0 |
| draft | 0 |
| complete | 10 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

