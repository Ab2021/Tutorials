# Module 43 — Security Engineering for Supply Chains

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/43_security_engineering_supply_chain.md`](../../../course/43_security_engineering_supply_chain.md)
> **Folder:** `modules/07_engineering_electives/43_security_engineering_supply_chain/`
> **Version:** 2026.5

---

## 1. Module purpose

This module teaches supply chain security as an engineering discipline: threat modelling, attack trees, OT and IT segmentation, third-party risk, software supply chain provenance, cargo crime, payment fraud, incident response, and risk quantification.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 05, 16, 20, 24 |
| Unlocks | — (capstone domain elective) |
| Assessment | Security architecture and risk case (Week 42) |
| Survey file | `43_security_engineering_supply_chain.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_threat-modelling-stride.md`](01_threat-modelling-stride.md) | Threat Modelling and STRIDE | Data-flow diagrams and systematic threat enumeration. | 1,200–1,800 | planned |
| 2 | [`02_attack-trees-risk.md`](02_attack-trees-risk.md) | Attack Trees and Risk Quantification | Path probability, ALE, and control comparison. | 1,200–1,800 | planned |
| 3 | [`03_ot-security-purdue.md`](03_ot-security-purdue.md) | OT Security and the Purdue Model | PLC, SCADA, WCS, zones, and conduits. | 1,200–1,800 | planned |
| 4 | [`04_zero-trust-segmentation.md`](04_zero-trust-segmentation.md) | Zero Trust and Segmentation | Least privilege, micro-segmentation, and third-party access. | 1,200–1,800 | planned |
| 5 | [`05_third-party-risk.md`](05_third-party-risk.md) | Third-Party and Supplier Risk | Assessment, evidence, contracts, and fourth parties. | 1,200–1,800 | planned |
| 6 | [`06_software-supply-chain-sbom.md`](06_software-supply-chain-sbom.md) | Software Supply Chain and SBOM | Dependencies, provenance, signing, and SLSA. | 1,200–1,800 | planned |
| 7 | [`07_physical-cargo-security.md`](07_physical-cargo-security.md) | Physical and Cargo Security | Strategic theft, seals, yard access, and C-TPAT/AEO. | 1,200–1,800 | planned |
| 8 | [`08_fraud-payment-controls.md`](08_fraud-payment-controls.md) | Fraud and Payment Controls | BEC, invoice fraud, and vendor master integrity. | 1,200–1,800 | planned |
| 9 | [`09_incident-response-continuity.md`](09_incident-response-continuity.md) | Incident Response and Continuity | Playbooks, containment, recovery, and exercises. | 1,200–1,800 | planned |
| 10 | [`10_standards-governance.md`](10_standards-governance.md) | Standards and Governance | ISO 27001, NIST CSF, IEC 62443, and assurance. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Threat Modelling and STRIDE ->
02 Attack Trees and Risk Quantification ->
03 OT Security and the Purdue Model ->
04 Zero Trust and Segmentation ->
05 Third-Party and Supplier Risk ->
06 Software Supply Chain and SBOM ->
07 Physical and Cargo Security ->
08 Fraud and Payment Controls ->
09 Incident Response and Continuity ->
10 Standards and Governance
```

## 5. What this module does not cover

- It does not cover offensive penetration testing techniques or exploit development.
- It does not cover national security, intelligence, or classified defence programmes.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: security testing and incident reporting require professional and legal review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 10 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`43_security_engineering_supply_chain.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
