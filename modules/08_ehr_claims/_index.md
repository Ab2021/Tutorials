# Module 08 — EHR, Claims, and Health System Operations

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/08_ehr_claims_operations.md`](../../course/08_ehr_claims_operations.md)
> **Folder:** `modules/08_ehr_claims/`
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
| Prerequisites | 02 |
| Unlocks | 17, 19 |
| Assessment | Operational decision product (Week 15) |
| Survey file | `08_ehr_claims_operations.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_health-system-operating-model.md`](01_health-system-operating-model.md) | Health System Operating Model | How money, clinicians, and patients move through a hospital and clinic network. | 1,200-1,800 | complete |
| 2 | [`02_ehr-cohorts.md`](02_ehr-cohorts.md) | EHR Cohorts | Defining populations from encounter and order data without fooling yourself. | 1,200-1,800 | complete |
| 3 | [`03_claims-analytics.md`](03_claims-analytics.md) | Claims Analytics | Episode construction, attribution, risk adjustment, and cost analysis. | 1,200-1,800 | complete |
| 4 | [`04_operational-models.md`](04_operational-models.md) | Operational Models | No-show, length of stay, readmission, staffing, and OR scheduling. | 1,200-1,800 | complete |
| 5 | [`05_capacity-and-patient-flow.md`](05_capacity-and-patient-flow.md) | Capacity and Patient Flow | Queues, bottlenecks, boarding, and simulation of flow interventions. | 1,200-1,800 | complete |
| 6 | [`06_quality-measures.md`](06_quality-measures.md) | Quality Measures | HEDIS, CMS measures, STAR ratings, and measure specification logic. | 1,200-1,800 | complete |
| 7 | [`07_revenue-cycle.md`](07_revenue-cycle.md) | Revenue Cycle | Coding, denials, prior authorization, and where analytics changes cash flow. | 1,200-1,800 | complete |
| 8 | [`08_evaluating-operational-interventions.md`](08_evaluating-operational-interventions.md) | Evaluating Operational Interventions | Before/after, interrupted time series, difference-in-differences, and step-wedge. | 1,200-1,800 | complete |
| 9 | [`09_case-studies.md`](09_case-studies.md) | Case Studies | Documented operational analytics programmes and what actually moved. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Health System Operating Model ->
02 EHR Cohorts ->
03 Claims Analytics ->
04 Operational Models ->
05 Capacity and Patient Flow ->
06 Quality Measures ->
07 Revenue Cycle ->
08 Evaluating Operational Interventions ->
09 Case Studies
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
| complete | 9 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

