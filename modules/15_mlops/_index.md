# Module 15 — Deployment and MLOps

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
> **Folder:** `modules/15_mlops/`
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
| Prerequisites | 03, 06 |
| Unlocks | 26 |
| Assessment | Deployment and monitoring plan (Week 26) |
| Survey file | `15_deployment_mlops.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | `01_the-production-loop.md` | The Production Loop | Train, validate, deploy, monitor, retrain, retire; where each stage fails. | 1,200-1,800 | planned |
| 2 | `02_deployment-patterns.md` | Deployment Patterns | Batch, real-time, streaming, embedded, and shadow deployment. | 1,200-1,800 | planned |
| 3 | `03_packaging-and-reproducibility.md` | Packaging and Reproducibility | Containers, lockfiles, model artifacts, and environment drift. | 1,200-1,800 | planned |
| 4 | `04_ci-cd-ct.md` | CI/CD/CT | Continuous integration, delivery, and training with clinical gates. | 1,200-1,800 | planned |
| 5 | `05_monitoring.md` | Monitoring | Service health, data quality, model performance, and outcome monitoring. | 1,200-1,800 | planned |
| 6 | `06_drift.md` | Drift | Covariate, label, concept, and population drift; detection and response. | 1,200-1,800 | planned |
| 7 | `07_incident-response.md` | Incident Response | Severity, triage, rollback, communication, and post-incident review. | 1,200-1,800 | planned |
| 8 | `08_rollout-and-rollback.md` | Rollout and Rollback | Canary, shadow, blue-green, feature flags, and safe reversal. | 1,200-1,800 | planned |
| 9 | `09_model-governance.md` | Model Governance | Inventory, ownership, approvals, change control, and audit evidence. | 1,200-1,800 | planned |
| 10 | `10_cost-and-capacity.md` | Cost and Capacity | Serving economics, autoscaling, quotas, and capacity planning. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Production Loop ->
02 Deployment Patterns ->
03 Packaging and Reproducibility ->
04 CI/CD/CT ->
05 Monitoring ->
06 Drift ->
07 Incident Response ->
08 Rollout and Rollback ->
09 Model Governance ->
10 Cost and Capacity
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

