# Module 01A — Foundations, Assumptions, and Units

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md)
> **Folder:** `modules/01A_foundations/`
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
| Unlocks | every quantitative module |
| Assessment | Assumptions ledger (Week 4) |
| Survey file | `01A_foundations_assumptions_and_units.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_mathematical-basics.md`](01_mathematical-basics.md) | Mathematical Basics | Functions, logs, exponents, rates, ratios, and the algebra used in every later module. | 1,200-1,800 | complete |
| 2 | [`02_probability-and-statistics-basics.md`](02_probability-and-statistics-basics.md) | Probability and Statistics Basics | Distributions, expectation, variance, sampling, estimation, and hypothesis testing intuition. | 1,200-1,800 | complete |
| 3 | [`03_units-and-metrology.md`](03_units-and-metrology.md) | Units and Metrology | SI units, clinical units, conversion factors, significant figures, and measurement uncertainty. | 1,200-1,800 | complete |
| 4 | [`04_dimensional-analysis.md`](04_dimensional-analysis.md) | Dimensional Analysis | Checking equations by units; common conversion errors in laboratory and device data. | 1,200-1,800 | complete |
| 5 | [`05_biological-basics.md`](05_biological-basics.md) | Biological Basics | Cells, genes, proteins, metabolism, homeostasis, and the timescales that matter for data. | 1,200-1,800 | complete |
| 6 | [`06_pharmacokinetics.md`](06_pharmacokinetics.md) | Pharmacokinetics | One- and two-compartment models, half-life, clearance, volume of distribution, dosing. | 1,200-1,800 | complete |
| 7 | [`07_epidemiology-basics.md`](07_epidemiology-basics.md) | Epidemiology Basics | Incidence, prevalence, risk, rate, standardization, and measures of association. | 1,200-1,800 | complete |
| 8 | [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md) | Engineering and Device Basics | Signals, sampling, noise, resolution, calibration, and the device as a measurement instrument. | 1,200-1,800 | complete |
| 9 | [`09_computing-basics.md`](09_computing-basics.md) | Computing Basics | Floating point, missing values, dates and time zones, identifiers, and file formats. | 1,200-1,800 | complete |
| 10 | [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md) | The Assumptions Ledger | A reusable artifact for recording, testing, and communicating every assumption behind a claim. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Mathematical Basics ->
02 Probability and Statistics Basics ->
03 Units and Metrology ->
04 Dimensional Analysis ->
05 Biological Basics ->
06 Pharmacokinetics ->
07 Epidemiology Basics ->
08 Engineering and Device Basics ->
09 Computing Basics ->
10 The Assumptions Ledger
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

