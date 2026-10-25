# Module 10 — Medical Imaging and Pathology AI

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
> **Folder:** `modules/10_imaging/`
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
| Prerequisites | 07 |
| Unlocks | 18, 23 |
| Assessment | Imaging evidence package (Week 18) |
| Survey file | `10_medical_imaging.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_modalities.md`](01_modalities.md) | Modalities | CT, MRI, X-ray, ultrasound, PET, digital pathology, and their data characteristics. | 1,200-1,800 | complete |
| 2 | [`02_dicom-and-metadata.md`](02_dicom-and-metadata.md) | DICOM and Metadata | Tags, series, spacing, windowing, and the metadata that leaks labels. | 1,200-1,800 | complete |
| 3 | [`03_preprocessing.md`](03_preprocessing.md) | Preprocessing | Resampling, normalization, augmentation, and preserving diagnostic content. | 1,200-1,800 | complete |
| 4 | [`04_labels-and-annotation.md`](04_labels-and-annotation.md) | Labels and Annotation | Radiologist reads, segmentation masks, adjudication, and label noise. | 1,200-1,800 | complete |
| 5 | [`05_architectures.md`](05_architectures.md) | Architectures | Classification, detection, segmentation, and self-supervised approaches. | 1,200-1,800 | complete |
| 6 | [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md) | Leakage and Shortcuts | Site signatures, scanner effects, laterality markers, and hidden text. | 1,200-1,800 | complete |
| 7 | [`07_evaluation.md`](07_evaluation.md) | Evaluation | Lesion-level versus image-level metrics, bootstrap intervals, and reader studies. | 1,200-1,800 | complete |
| 8 | [`08_workflow-integration.md`](08_workflow-integration.md) | Workflow Integration | PACS, worklists, reading room latency, and radiologist trust. | 1,200-1,800 | complete |
| 9 | [`09_regulatory-path.md`](09_regulatory-path.md) | Regulatory Path | FDA clearance routes for imaging AI and the evidence each requires. | 1,200-1,800 | complete |
| 10 | [`10_foundation-models.md`](10_foundation-models.md) | Foundation Models | Vision encoders, multimodal reporting, and their evaluation gaps. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Modalities ->
02 DICOM and Metadata ->
03 Preprocessing ->
04 Labels and Annotation ->
05 Architectures ->
06 Leakage and Shortcuts ->
07 Evaluation ->
08 Workflow Integration ->
09 Regulatory Path ->
10 Foundation Models
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

