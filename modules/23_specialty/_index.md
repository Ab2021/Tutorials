# Module 23 — Specialty Electives

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
> **Folder:** `modules/23_specialty/`
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
| Prerequisites | 01, 08 |
| Unlocks | 19 |
| Assessment | Elective primer assessment |
| Survey file | `23_specialty_electives.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | `01_oncology.md` | Oncology | Tumour biology, staging, RECIST, endpoints, and oncology data. | 1,200-1,800 | planned |
| 2 | `02_cardiology.md` | Cardiology | Heart failure, ACS, arrhythmia, devices, and cardiology registries. | 1,200-1,800 | planned |
| 3 | `03_pathology.md` | Pathology | Histology, whole-slide imaging, molecular pathology, and reporting. | 1,200-1,800 | planned |
| 4 | `04_critical-care.md` | Critical Care | ICU physiology, ventilation, sepsis, and high-frequency data. | 1,200-1,800 | planned |
| 5 | `05_pediatrics-and-obstetrics.md` | Pediatrics and Obstetrics | Growth, dosing, maternal-fetal data, and special consent. | 1,200-1,800 | planned |
| 6 | `06_public-health.md` | Public Health | Population surveillance, equity, and policy analytics. | 1,200-1,800 | planned |
| 7 | `07_cross-specialty-comparison.md` | Cross-Specialty Comparison | What transfers between specialties and what does not. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Oncology ->
02 Cardiology ->
03 Pathology ->
04 Critical Care ->
05 Pediatrics and Obstetrics ->
06 Public Health ->
07 Cross-Specialty Comparison
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 7 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

