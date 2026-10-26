# Module 18 — Quality, Evaluation, and Reporting

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
> **Folder:** `modules/18_quality/`
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
| Prerequisites | 05, 06 |
| Unlocks | 19 |
| Assessment | Evidence dossier (Week 29) |
| Survey file | `18_quality_evaluation_reporting.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_the-evidence-chain.md`](01_the-evidence-chain.md) | The Evidence Chain | Claim to endpoint to dataset to analysis to report; where chains break. | 1,200-1,800 | complete |
| 2 | [`02_reporting-guidelines.md`](02_reporting-guidelines.md) | Reporting Guidelines | TRIPOD+AI, CONSORT-AI, STARD-AI, SPIRIT-AI, DECIDE-AI, and PROBAST. | 1,200-1,800 | complete |
| 3 | `03_appraising-clinical-ai-evidence.md` | Appraising Clinical AI Evidence | A structured appraisal worksheet with worked examples. | 1,200-1,800 | planned |
| 4 | `04_model-cards-and-datasheets.md` | Model Cards and Datasheets | What to document and what readers actually need. | 1,200-1,800 | planned |
| 5 | `05_reproducibility.md` | Reproducibility | Code, data, environment, and the difference between repeatable and reproducible. | 1,200-1,800 | planned |
| 6 | `06_quality-management.md` | Quality Management | QMS basics for analytics teams, SOPs, and CAPA. | 1,200-1,800 | planned |
| 7 | `07_post-market-evaluation.md` | Post-Market Evaluation | Performance monitoring, drift, and real-world evidence generation. | 1,200-1,800 | planned |
| 8 | `08_audit.md` | Audit | Preparing for an audit and what auditors ask for. | 1,200-1,800 | planned |
| 9 | `09_documentation.md` | Documentation | Writing documents that survive staff turnover and inspections. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Evidence Chain ->
02 Reporting Guidelines ->
03 Appraising Clinical AI Evidence ->
04 Model Cards and Datasheets ->
05 Reproducibility ->
06 Quality Management ->
07 Post-Market Evaluation ->
08 Audit ->
09 Documentation
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
| complete | 2 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

