# Module 14 — Medical Devices, SaMD, and Regulation

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
> **Folder:** `modules/14_devices_regulation/`
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
| Prerequisites | 00, 01 |
| Unlocks | 14A, 14B, 18, 25 |
| Assessment | Regulatory evidence package (Week 22) |
| Survey file | `14_medical_devices_samd_regulation.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_definitions-and-classification.md`](01_definitions-and-classification.md) | Definitions and Classification | Device, SaMD, SiMD, IVD, accessory; risk classes in the US and EU. | 1,200-1,800 | complete |
| 2 | [`02_is-it-a-device.md`](02_is-it-a-device.md) | Is It a Device? | The decision tree, clinical decision support exclusions, and why vendors get this wrong. | 1,200-1,800 | complete |
| 3 | [`03_fda-pathways.md`](03_fda-pathways.md) | FDA Pathways | 510(k), De Novo, PMA, and the evidence each demands. | 1,200-1,800 | complete |
| 4 | [`04_gmlp-and-pccp.md`](04_gmlp-and-pccp.md) | GMLP and PCCP | Good machine learning practice and predetermined change control plans. | 1,200-1,800 | complete |
| 5 | [`05_qms-and-design-controls.md`](05_qms-and-design-controls.md) | QMS and Design Controls | ISO 13485, QMSR, design history file, and document control. | 1,200-1,800 | complete |
| 6 | [`06_iso-14971-risk-management.md`](06_iso-14971-risk-management.md) | ISO 14971 Risk Management | Hazard analysis, risk estimation, controls, and residual risk. | 1,200-1,800 | complete |
| 7 | [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md) | IEC 62304 Software Lifecycle | Safety classes, software requirements, architecture, and verification. | 1,200-1,800 | complete |
| 8 | [`08_eu-mdr-and-ivdr.md`](08_eu-mdr-and-ivdr.md) | EU MDR and IVDR | Classification, conformity assessment, notified bodies, and post-market obligations. | 1,200-1,800 | complete |
| 9 | [`09_the-evidence-package.md`](09_the-evidence-package.md) | The Evidence Package | What goes in a submission and how the pieces support the claim. | 1,200-1,800 | complete |
| 10 | [`10_post-market-surveillance.md`](10_post-market-surveillance.md) | Post-Market Surveillance | Complaints, MDR reporting, PSUR, and real-world performance monitoring. | 1,200-1,800 | complete |
| 11 | [`11_in-silico-trials-and-simulation.md`](11_in-silico-trials-and-simulation.md) | In-Silico Trials and Simulation | ASME V&V 40 credibility, ISO/TR 23809, statistical shape modeling, and FDA CM&S dossier. | 1,200-1,800 | complete |
| 12 | [`12_udi-and-pms-analytics.md`](12_udi-and-pms-analytics.md) | UDI and Post-Market Surveillance Analytics | Unique Device Identification (DI/PI), FDA GUDID, MAUDE text mining, and CUSUM quality control. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Definitions and Classification ->
02 Is It a Device? ->
03 FDA Pathways ->
04 GMLP and PCCP ->
05 QMS and Design Controls ->
06 ISO 14971 Risk Management ->
07 IEC 62304 Software Lifecycle ->
08 EU MDR and IVDR ->
09 The Evidence Package ->
10 Post-Market Surveillance
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

