# Module 13 — Biopharma and Clinical Development

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
> **Folder:** `modules/13_biopharma/`
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
| Prerequisites | 05, 12 |
| Unlocks | 25 |
| Assessment | Translational data plan (Week 21) |
| Survey file | `13_biopharma_clinical_development.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_development-funnel.md`](01_development-funnel.md) | The Development Funnel | Discovery to Phase IV, decision gates, cost, and attrition. | 1,200-1,800 | complete |
| 2 | [`02_data-in-pharma.md`](02_data-in-pharma.md) | Data in Pharma | Preclinical, clinical, safety, and commercial data domains. | 1,200-1,800 | complete |
| 3 | [`03_biomarker-strategy.md`](03_biomarker-strategy.md) | Biomarker Strategy | Prognostic, predictive, pharmacodynamic, and surrogate endpoints. | 1,200-1,800 | complete |
| 4 | [`04_feasibility-and-recruitment.md`](04_feasibility-and-recruitment.md) | Feasibility and Recruitment | Site selection, enrollment modelling, and diversity planning. | 1,200-1,800 | complete |
| 5 | [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md) | CDISC and Trial Standards | CDASH, SDTM, ADaM, define.xml, and submission-ready data. | 1,200-1,800 | complete |
| 6 | [`06_safety-and-pharmacovigilance.md`](06_safety-and-pharmacovigilance.md) | Safety and Pharmacovigilance | AE coding, signal detection, FAERS/AEMS, and aggregate reporting. | 1,200-1,800 | complete |
| 7 | [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md) | RWE in Pharma | External controls, label expansion, and payer evidence. | 1,200-1,800 | complete |
| 8 | [`08_ai-in-discovery-and-development.md`](08_ai-in-discovery-and-development.md) | AI in Discovery and Development | Where models are used, validated, and rejected. | 1,200-1,800 | complete |
| 9 | [`09_decentralized-trials.md`](09_decentralized-trials.md) | Decentralised Trials | Remote consent, ePRO, wearables, and the data integrity questions. | 1,200-1,800 | complete |
| 10 | [`10_submission-data.md`](10_submission-data.md) | Submission Data | eCTD structure, datasets, and what reviewers actually check. | 1,200-1,800 | complete |
| 11 | [`11_digital-twins-in-clinical-trials.md`](11_digital-twins-in-clinical-trials.md) | Digital Twins in Clinical Trials | Patient-level generative emulators, PROCOVA method, ANCOVA variance reduction, and trial downsizing. | 1,200-1,800 | complete |
| 12 | [`12_statistical-pharmacovigilance.md`](12_statistical-pharmacovigilance.md) | Statistical Pharmacovigilance | Disproportionality metrics (PRR, ROR), Empirical Bayes (MGPS/EBGM), FAERS, and the Weber effect. | 1,200-1,800 | complete |
| 13 | [`13_pass-and-active-safety-surveillance.md`](13_pass-and-active-safety-surveillance.md) | PASS and Active Safety Surveillance | Post-Authorization Safety Studies, Self-Controlled Case Series (SCCS), FDA Sentinel, and TreeScan. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Development Funnel ->
02 Data in Pharma ->
03 Biomarker Strategy ->
04 Feasibility and Recruitment ->
05 CDISC and Trial Standards ->
06 Safety and Pharmacovigilance ->
07 RWE in Pharma ->
08 AI in Discovery and Development ->
09 Decentralised Trials ->
10 Submission Data
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

