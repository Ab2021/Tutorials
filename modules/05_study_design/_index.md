# Module 05 — Study Design and Evidence Generation

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/05_study_design_evidence.md`](../../course/05_study_design_evidence.md)
> **Folder:** `modules/05_study_design/`
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
| Unlocks | 13, 18, 25 |
| Assessment | Evidence generation plan (Week 10) |
| Survey file | `05_study_design_evidence.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_from-claim-to-evidence.md`](01_from-claim-to-evidence.md) | From Claim to Evidence | Translating a marketing or clinical claim into a testable evidence plan. | 1,200-1,800 | complete |
| 2 | [`02_trial-designs.md`](02_trial-designs.md) | Trial Designs | Parallel, crossover, adaptive, cluster, platform, and decentralised designs. | 1,200-1,800 | complete |
| 3 | [`03_real-world-evidence.md`](03_real-world-evidence.md) | Real-World Evidence | When RWE is fit for a regulatory or payer decision, and when it is not. | 1,200-1,800 | complete |
| 4 | [`04_validation-designs.md`](04_validation-designs.md) | Validation Designs | Internal, temporal, external, geographic, and prospective validation. | 1,200-1,800 | complete |
| 5 | [`05_bias-taxonomy.md`](05_bias-taxonomy.md) | Bias Taxonomy | Selection, information, confounding, immortal time, attrition, and measurement bias. | 1,200-1,800 | complete |
| 6 | [`06_sample-size-and-power.md`](06_sample-size-and-power.md) | Sample Size and Power | Power for differences, for precision, and for subgroup claims. | 1,200-1,800 | complete |
| 7 | [`07_estimands.md`](07_estimands.md) | Estimands | ICH E9(R1) five attributes and why the estimand comes before the method. | 1,200-1,800 | complete |
| 8 | [`08_external-validation.md`](08_external-validation.md) | External Validation | Site selection, transportability, recalibration, and reporting standards. | 1,200-1,800 | complete |
| 9 | [`09_evidence-synthesis.md`](09_evidence-synthesis.md) | Evidence Synthesis | Systematic review, meta-analysis, indirect comparison, and network meta-analysis. | 1,200-1,800 | complete |
| 10 | [`10_protocol-writing.md`](10_protocol-writing.md) | Protocol Writing | A protocol skeleton that a regulator, IRB, and statistician can all read. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 From Claim to Evidence ->
02 Trial Designs ->
03 Real-World Evidence ->
04 Validation Designs ->
05 Bias Taxonomy ->
06 Sample Size and Power ->
07 Estimands ->
08 External Validation ->
09 Evidence Synthesis ->
10 Protocol Writing
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

