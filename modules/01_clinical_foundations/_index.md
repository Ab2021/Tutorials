# Module 01 — Clinical Foundations for Data Scientists

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
> **Folder:** `modules/01_clinical_foundations/`
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
| Prerequisites | 00 |
| Unlocks | 01A, 02, 08, 09, 23 |
| Assessment | Clinical plausibility memo (Week 3) |
| Survey file | `01_clinical_foundations.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_clinical-reasoning-and-workflow.md`](01_clinical-reasoning-and-workflow.md) | Clinical Reasoning and Workflow | How clinicians form and test hypotheses; where data enters and leaves the encounter. | 1,200-1,800 | complete |
| 2 | [`02_anatomy-physiology-and-vital-signs.md`](02_anatomy-physiology-and-vital-signs.md) | Anatomy, Physiology, and Vital Signs | Organ systems, normal ranges, compensation, and what derangement looks like in data. | 1,200-1,800 | complete |
| 3 | [`03_laboratory-medicine.md`](03_laboratory-medicine.md) | Laboratory Medicine | Specimens, assays, reference intervals, sensitivity/specificity, and pre-analytical error. | 1,200-1,800 | complete |
| 4 | [`04_pharmacology-basics.md`](04_pharmacology-basics.md) | Pharmacology Basics | Absorption, distribution, metabolism, excretion, interactions, and why dosing data is messy. | 1,200-1,800 | complete |
| 5 | [`05_disease-taxonomy-and-coding.md`](05_disease-taxonomy-and-coding.md) | Disease Taxonomy and Coding | ICD-10/11, SNOMED CT, MeSH, Orphanet, and the difference between a code and a diagnosis. | 1,200-1,800 | complete |
| 6 | [`06_radiology-and-imaging-basics.md`](06_radiology-and-imaging-basics.md) | Radiology and Imaging Basics | Modalities, acquisition, reporting workflow, and what a radiologist does with an image. | 1,200-1,800 | complete |
| 7 | [`07_cardiology-and-ecg-basics.md`](07_cardiology-and-ecg-basics.md) | Cardiology and ECG Basics | The cardiac cycle, leads, intervals, common rhythms, and artifact that mimics disease. | 1,200-1,800 | complete |
| 8 | [`08_documentation-and-clinical-narratives.md`](08_documentation-and-clinical-narratives.md) | Documentation and Clinical Narratives | Note types, copy-forward, templates, coded vs narrative content, and documentation bias. | 1,200-1,800 | complete |
| 9 | [`09_special-populations.md`](09_special-populations.md) | Special Populations | Pediatrics, pregnancy, older adults, renal/hepatic impairment, and device-dependent patients. | 1,200-1,800 | complete |
| 10 | [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md) | Clinical Plausibility Checks | A checklist for catching impossible values, impossible trajectories, and impossible cohorts. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Clinical Reasoning and Workflow ->
02 Anatomy, Physiology, and Vital Signs ->
03 Laboratory Medicine ->
04 Pharmacology Basics ->
05 Disease Taxonomy and Coding ->
06 Radiology and Imaging Basics ->
07 Cardiology and ECG Basics ->
08 Documentation and Clinical Narratives ->
09 Special Populations ->
10 Clinical Plausibility Checks
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

