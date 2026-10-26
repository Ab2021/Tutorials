# Module 16 — Privacy, Security, and Ethics

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
> **Folder:** `modules/16_privacy_security/`
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
| Prerequisites | 02, 03 |
| Unlocks | 17, 18 |
| Assessment | Privacy and ethics impact assessment (Week 27) |
| Survey file | `16_privacy_security_ethics.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_privacy-principles.md`](01_privacy-principles.md) | Privacy Principles | Purpose limitation, minimization, consent, and the clinical exceptions. | 1,200-1,800 | complete |
| 2 | [`02_deidentification-and-anonymization.md`](02_deidentification-and-anonymization.md) | De-identification and Anonymization | HIPAA Safe Harbor, expert determination, GDPR pseudonymization, and residual risk. | 1,200-1,800 | complete |
| 3 | [`03_privacy-enhancing-technologies.md`](03_privacy-enhancing-technologies.md) | Privacy-Enhancing Technologies | Differential privacy, federated learning, secure computation, and synthetic data. | 1,200-1,800 | complete |
| 4 | [`04_threat-modeling.md`](04_threat-modeling.md) | Threat Modeling | STRIDE, attack trees, insider risk, and the clinical threat surface. | 1,200-1,800 | complete |
| 5 | [`05_access-control-and-cryptography.md`](05_access-control-and-cryptography.md) | Access Control and Cryptography | RBAC, ABAC, key management, and encryption in practice. | 1,200-1,800 | complete |
| 6 | [`06_fairness.md`](06_fairness.md) | Fairness | Definitions, measurement, mitigation, and the limits of each. | 1,200-1,800 | complete |
| 7 | [`07_ethics.md`](07_ethics.md) | Ethics | Consent, equity, automation bias, and the ethics of deployment. | 1,200-1,800 | complete |
| 8 | [`08_governance.md`](08_governance.md) | Governance | Committees, data trusts, model review boards, and accountability. | 1,200-1,800 | complete |
| 9 | [`09_breach-and-incident.md`](09_breach-and-incident.md) | Breach and Incident | Detection, notification, containment, and lessons. | 1,200-1,800 | complete |
| 10 | [`10_regulatory-map.md`](10_regulatory-map.md) | Regulatory Map | HIPAA, GDPR, EU AI Act, state law, and sector-specific rules. | 1,200-1,800 | complete |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Privacy Principles ->
02 De-identification and Anonymization ->
03 Privacy-Enhancing Technologies ->
04 Threat Modeling ->
05 Access Control and Cryptography ->
06 Fairness ->
07 Ethics ->
08 Governance ->
09 Breach and Incident ->
10 Regulatory Map
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

