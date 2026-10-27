# Module 33 — Healthcare Provider and Life Sciences Operations

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/33_healthcare_provider_life_sciences.md`](../../../course/33_healthcare_provider_life_sciences.md)
> **Folder:** `modules/06_domain_electives/33_healthcare_provider_life_sciences/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches the supply chain of a hospital and of a life-sciences company: item identity, par replenishment, surgical case carts, traceability, shortages and recalls, and clinical trial supply.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 07, 10, 13, 29 |
| Unlocks | 34, 37 |
| Assessment | Provider supply chain design (Week 36) |
| Survey file | `33_healthcare_provider_life_sciences.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_provider-network-item-master.md`](01_provider-network-item-master.md) | The Provider Network and Item Master | Manufacturer to patient, identity, UDI, and contract linkage. | 1,200–1,800 | planned |
| 2 | [`02_par-levels-clinical-replenishment.md`](02_par-levels-clinical-replenishment.md) | Par Levels and Clinical Replenishment | Demand, review period, lead time, criticality, and cabinet data. | 1,200–1,800 | planned |
| 3 | [`03_or-case-carts-preference-cards.md`](03_or-case-carts-preference-cards.md) | Operating Rooms, Case Carts, and Preference Cards | Kit design, pick accuracy, and first-case delays. | 1,200–1,800 | planned |
| 4 | [`04_consignment-vmi.md`](04_consignment-vmi.md) | Consignment and Vendor-Managed Inventory | Ownership, counting, expiry, and contract terms. | 1,200–1,800 | planned |
| 5 | [`05_udi-traceability.md`](05_udi-traceability.md) | UDI and Traceability | Receipt-to-patient trace, point-of-use capture, and recall readiness. | 1,200–1,800 | planned |
| 6 | [`06_shortage-recall-management.md`](06_shortage-recall-management.md) | Shortage and Recall Management | Allocation rules, substitution, and recall execution. | 1,200–1,800 | planned |
| 7 | [`07_clinical-trial-supply.md`](07_clinical-trial-supply.md) | Clinical Trial Supply | Randomization, blinding, kits, resupply, and expiry. | 1,200–1,800 | planned |
| 8 | [`08_provider-metrics-governance.md`](08_provider-metrics-governance.md) | Provider Metrics and Governance | Fill rate, turns, case-cart accuracy, and clinical value analysis. | 1,200–1,800 | planned |
| 9 | [`09_privacy-clinical-risk.md`](09_privacy-clinical-risk.md) | Privacy and Clinical Risk | HIPAA, GDPR, de-identification, and decision-support boundaries. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Provider Network and Item Master ->
02 Par Levels and Clinical Replenishment ->
03 Operating Rooms, Case Carts, and Preference Cards ->
04 Consignment and Vendor-Managed Inventory ->
05 UDI and Traceability ->
06 Shortage and Recall Management ->
07 Clinical Trial Supply ->
08 Provider Metrics and Governance ->
09 Privacy and Clinical Risk
```

## 5. What this module does not cover

- It does not cover clinical medicine, diagnosis, or treatment decisions.
- Pharmaceutical manufacturing and GMP production are covered only at supply-chain level.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 9 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`33_healthcare_provider_life_sciences.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
