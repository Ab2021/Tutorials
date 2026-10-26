# Module 02 — The Biomedical Data Landscape

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/02_biomedical_data_landscape.md`](../../course/02_biomedical_data_landscape.md)
> **Folder:** `modules/02_data_landscape/`
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
| Prerequisites | 01, 01A |
| Unlocks | 03, 08, 21 |
| Assessment | Data-source decision memo (Week 5) |
| Survey file | `02_biomedical_data_landscape.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_ehr-data.md`](01_ehr-data.md) | EHR Data | Encounter, order, result, medication, note, and flow-sheet data; what each can and cannot answer. | 1,200-1,800 | complete |
| 2 | [`02_claims-and-billing.md`](02_claims-and-billing.md) | Claims and Billing Data | Claim lifecycle, coding, adjudication, and the economics embedded in the data. | 1,200-1,800 | complete |
| 3 | [`03_registries-and-public-health.md`](03_registries-and-public-health.md) | Registries and Public Health Data | Disease registries, surveillance, vital statistics, and reporting incentives. | 1,200-1,800 | complete |
| 4 | [`04_imaging-and-dicom.md`](04_imaging-and-dicom.md) | Imaging and DICOM | Pixel data, metadata, series structure, and the imaging archive as a data source. | 1,200-1,800 | complete |
| 5 | [`05_signals-and-wearables.md`](05_signals-and-wearables.md) | Signals and Wearables | ECG, PPG, accelerometry, CGM, and the provenance problems of consumer devices. | 1,200-1,800 | complete |
| 6 | [`06_omics-data.md`](06_omics-data.md) | Omics Data | Genomic, transcriptomic, proteomic, and single-cell data at a survey level. | 1,200-1,800 | complete |
| 7 | [`07_trial-data.md`](07_trial-data.md) | Trial Data | EDC, CDISC, randomization, protocol deviations, and the difference from routine care data. | 1,200-1,800 | complete |
| 8 | [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md) | Standards and Terminologies | HL7 v2, FHIR, OMOP, LOINC, SNOMED CT, ICD, RxNorm, and how they relate. | 1,200-1,800 | complete |
| 9 | [`09_licensing-and-access.md`](09_licensing-and-access.md) | Licensing and Access | DUAs, BAAs, data-use tiers, credentialed access, and the cost of getting data wrong. | 1,200-1,800 | complete |
| 10 | [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) | Data Quality Dimensions | Completeness, accuracy, timeliness, consistency, validity, and fitness for purpose. | 1,200-1,800 | complete |
| 11 | [`11_global-health-and-lmic-data.md`](11_global-health-and-lmic-data.md) | Global Health and LMIC Data Systems | Over **85% of the world's population** resides in Low- and Middle-Income Countries (LMICs) across sub-Saharan Africa,... | 1,200-1,800 | draft (262 lines) |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 EHR Data ->
02 Claims and Billing Data ->
03 Registries and Public Health Data ->
04 Imaging and DICOM ->
05 Signals and Wearables ->
06 Omics Data ->
07 Trial Data ->
08 Standards and Terminologies ->
09 Licensing and Access ->
10 Data Quality Dimensions ->
11 Global Health and LMIC Data Systems
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 0 |
| draft | 1 |
| complete | 10 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

