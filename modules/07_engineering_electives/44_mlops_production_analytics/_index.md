# Module 44 — MLOps and Production Analytics Engineering

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/44_mlops_production_analytics.md`](../../../course/44_mlops_production_analytics.md)
> **Folder:** `modules/07_engineering_electives/44_mlops_production_analytics/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches the engineering discipline that keeps analytics running: versioning, experiment tracking, point-in-time features, orchestration, deployment, monitoring, retraining, and model governance.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06, 07, 08, 19 |
| Unlocks | — (capstone domain elective) |
| Assessment | Production analytics system (Week 43) |
| Survey file | `44_mlops_production_analytics.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_lifecycle-framing.md`](01_lifecycle-framing.md) | Lifecycle and Problem Framing | Decision link, success metric, and stage gates. | 1,200–1,800 | planned |
| 2 | [`02_versioning-reproducibility.md`](02_versioning-reproducibility.md) | Versioning and Reproducibility | Code, data, config, and environment. | 1,200–1,800 | planned |
| 3 | [`03_experiment-tracking.md`](03_experiment-tracking.md) | Experiment Tracking | Runs, parameters, metrics, and fair comparison. | 1,200–1,800 | planned |
| 4 | [`04_point-in-time-features.md`](04_point-in-time-features.md) | Point-in-Time Features | Feature stores, effective dating, and leakage. | 1,200–1,800 | planned |
| 5 | [`05_orchestration-pipelines.md`](05_orchestration-pipelines.md) | Orchestration and Pipelines | DAGs, retries, backfills, and lineage. | 1,200–1,800 | planned |
| 6 | [`06_serving-deployment.md`](06_serving-deployment.md) | Serving and Deployment | Batch, micro-batch, API, shadow, and canary. | 1,200–1,800 | planned |
| 7 | [`07_cicd-ct.md`](07_cicd-ct.md) | CI/CD and Continuous Training | Test gates, packaging, and automated retraining. | 1,200–1,800 | planned |
| 8 | [`08_monitoring-drift.md`](08_monitoring-drift.md) | Monitoring and Drift | PSI, data quality, proxies, and delayed labels. | 1,200–1,800 | planned |
| 9 | [`09_retraining-champion-challenger.md`](09_retraining-champion-challenger.md) | Retraining and Champion–Challenger | Triggers, shadow evaluation, and rollback. | 1,200–1,800 | planned |
| 10 | [`10_model-governance-platform.md`](10_model-governance-platform.md) | Model Governance and Platform Economics | SR 11-7, model cards, and build versus buy. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Lifecycle and Problem Framing ->
02 Versioning and Reproducibility ->
03 Experiment Tracking ->
04 Point-in-Time Features ->
05 Orchestration and Pipelines ->
06 Serving and Deployment ->
07 CI/CD and Continuous Training ->
08 Monitoring and Drift ->
09 Retraining and Champion-Challenger ->
10 Model Governance and Platform Economics
```

## 5. What this module does not cover

- It does not cover model algorithms or statistical learning theory: see Modules 06–08.
- It does not cover cloud infrastructure engineering, networking, or database administration.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: AI regulation and model risk content requires professional review.

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

See section 14 of the survey file (`44_mlops_production_analytics.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
