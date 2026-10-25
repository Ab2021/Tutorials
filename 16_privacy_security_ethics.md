# Module 16 — Privacy, Security, and Ethics

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 24 hours (9 lecture, 5 case clinic, 10 lab)
> **Track fit:** Common core
> **Prerequisites:** Modules 02, 03, 14
> **Tools:** Threat modeling, data-flow diagrams, privacy-enhancing technologies, fairness libraries
> **Industry question:** "Who could be harmed by this data or model, and what controls make that harm acceptably unlikely?"

---

## 1. Why industry cares

Biomedical data are among the most sensitive data people generate. A privacy breach can expose diagnoses, reproductive history, genetic risk, mental health, location, and identity. A biased model can deny care. A insecure device can be used to harm patients.

Trust is the product. A health system, payer, or regulator will not adopt a system it cannot govern.

This module teaches the privacy, security, fairness, and ethics controls that must be designed in, not bolted on.

## 2. Learning outcomes

By the end of this module, you can:

- Threat-model a biomedical data system.
- Apply data minimization, purpose limitation, and access control.
- Distinguish de-identification, anonymization, pseudonymization, and synthetic data.
- Evaluate privacy-enhancing technologies and their limits.
- Identify common adversarial ML attacks and defenses.
- Measure fairness and explain trade-offs among definitions.
- Design a consent and community-engagement approach.
- Build a governance structure for data and model access.
- Assess regulatory obligations under HIPAA, GDPR, and the EU AI Act.
- Write a privacy and ethics impact assessment.

## 3. Mental model: harms, controls, evidence

```
Data / model / workflow
        │
        ├── Who is affected?
        ├── What harm could occur?
        ├── How likely and severe?
        ├── What controls reduce risk?
        └── How will we know controls work?
```

**Rule:** privacy and ethics are risk-management problems, not compliance checkboxes.

## 4. Privacy principles

| Principle | Meaning | Practical control |
|---|---|---|
| Lawfulness | Valid legal basis | Consent, contract, public interest |
| Purpose limitation | Use only for stated purpose | Data use agreement, access review |
| Data minimization | Collect only what is needed | Variable-level justification |
| Accuracy | Data are correct | Quality controls, correction process |
| Storage limitation | Retain only as long as needed | Retention schedule, deletion |
| Integrity and confidentiality | Protect against misuse | Security controls |
| Accountability | Demonstrate compliance | Audit, DPIA, governance |

## 5. De-identification and anonymization

### 5.1 HIPAA

- **Safe Harbor:** remove 18 identifier categories.
- **Expert determination:** qualified expert determines re-identification risk is very small.

### 5.2 GDPR

- **Pseudonymization:** replace identifiers but retain linkage key.
- **Anonymization:** irreversible; high bar.
- **Personal data:** broad definition; includes online identifiers.

### 5.3 Residual risk

Even de-identified data can be re-identified through:

- Dates and geography.
- Rare diagnoses.
- Free-text.
- Imaging metadata.
- Genetic data.
- Linkage with public datasets.

**Rule:** de-identification reduces risk; it does not eliminate it.

## 6. Privacy-enhancing technologies

| Technology | What it does | Limits |
|---|---|---|
| Differential privacy | Adds calibrated noise | Utility loss; privacy budget |
| Federated learning | Trains without centralizing data | Leakage, heterogeneity, coordination |
| Secure multi-party computation | Computes on encrypted data | Compute overhead |
| Homomorphic encryption | Computes on ciphertext | Performance |
| Synthetic data | Generates artificial records | Fidelity, privacy leakage |
| Trusted execution environments | Isolates computation | Supply-chain trust |
| Tokenization | Replaces identifiers | Linkage risk |

**Rule:** no single technology solves privacy. Combine governance, architecture, and measurement.

## 7. Security

### 7.1 CIA triad

- **Confidentiality:** only authorized access.
- **Integrity:** data and models are accurate and unaltered.
- **Availability:** systems work when needed.

### 7.2 Controls

- Identity and access management.
- Least privilege.
- Multi-factor authentication.
- Encryption at rest and in transit.
- Network segmentation.
- Endpoint protection.
- Secrets management.
- Vulnerability management.
- Supply-chain security.
- Audit logging.
- Backup and recovery.
- Incident response.

### 7.3 Threat modeling

Use STRIDE:

- **Spoofing:** impersonation.
- **Tampering:** unauthorized modification.
- **Repudiation:** denying actions.
- **Information disclosure:** data leak.
- **Denial of service:** availability.
- **Elevation of privilege:** unauthorized access.

### 7.4 Adversarial ML

| Attack | Description | Defense |
|---|---|---|
| Evasion | Inputs crafted to fool model | Adversarial training, input validation |
| Poisoning | Corrupt training data | Data provenance, outlier detection |
| Membership inference | Determine if a record was in training | Differential privacy, regularization |
| Model inversion | Reconstruct training data | Privacy-preserving training |
| Model extraction | Steal model behavior | Rate limits, watermarking |
| Prompt injection | Hijack LLM behavior | Input sanitization, tool allowlists |

## 8. Fairness

### 8.1 Definitions

- **Demographic parity:** equal positive rates.
- **Equal opportunity:** equal true positive rates.
- **Equalized odds:** equal TPR and FPR.
- **Predictive parity:** equal PPV.
- **Calibration within groups:** predicted risk matches observed.

Definitions can conflict. The right choice depends on the decision and the harm.

### 8.2 Sources of bias

- Label bias.
- Measurement bias.
- Selection bias.
- Access bias.
- Historical bias.
- Representation bias.
- Deployment bias.

### 8.3 Mitigation

- Pre-processing: reweighting, resampling.
- In-processing: fairness constraints.
- Post-processing: threshold adjustment.
- Governance: inclusive data, community review, monitoring.

**Rule:** fairness is not a one-time metric. It is an ongoing property of the system.

## 9. Ethics

### 9.1 Core principles

- Respect for persons.
- Beneficence and non-maleficence.
- Justice.
- Transparency.
- Accountability.

### 9.2 Consent

- Informed, voluntary, specific enough.
- Broad consent for research may be acceptable with governance.
- Commercial use may require separate consideration.
- Dynamic consent is promising but operationally complex.

### 9.3 Equity and accessibility

- Include disabled and neurodivergent users.
- Provide multiple formats and interaction modes.
- Avoid time pressure when not clinically necessary.
- Test with affected communities.
- Measure whether the tool widens disparities.

### 9.4 Returning results

- Incidental findings.
- Actionability.
- Counseling.
- Privacy of family members.
- Legal obligations.

## 10. Governance

| Body | Role |
|---|---|
| IRB / ethics committee | Human-subjects research |
| Privacy office | HIPAA/GDPR compliance |
| Data access committee | Data sharing decisions |
| Security team | Threat model and controls |
| Model review board | Clinical and ethical review |
| Quality team | Design controls and audits |
| Community advisory board | Patient and public input |

## 11. Industry case studies

### 11.1 Obermeyer et al. and algorithmic bias

A cost-prediction algorithm used as a proxy for need under-referred Black patients because spending was lower at equal need.

**Lesson:** label choice and proxy variables are ethical decisions.

Source: *Science*, 2019. https://doi.org/10.1126/science.aax2342

### 11.2 Pulse oximetry bias

Pulse oximeters can overestimate oxygenation in patients with darker skin, affecting care and model inputs.

**Lesson:** measurement bias is a safety and equity issue.

Source: Sjoding, M. W. et al., *New England Journal of Medicine*, 2020. https://doi.org/10.1056/NEJMc2029240

### 11.3 Genetic privacy

Genetic data can reveal information about relatives and cannot be changed like a password. Re-identification risks are significant.

**Lesson:** genomic privacy requires special governance.

Source: NIH genomic data sharing policy. https://sharing.nih.gov/

### 11.4 Wearable surveillance

Continuous physiological and location data can reveal behavior, health, and relationships.

**Lesson:** consumer consent may not reflect the sensitivity of inferred data.

Source: FTC health breach notification and privacy guidance. https://www.ftc.gov/

### 11.5 EU AI Act

High-risk AI systems require risk management, data governance, transparency, human oversight, and post-market monitoring.

**Lesson:** AI governance is now law in the EU.

Source: Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

## 12. Labs

### Lab 16.1 — Threat model

**Objective:** Threat-model a connected clinical AI system.

**Steps:**

1. Draw data flows.
2. Identify assets and trust boundaries.
3. Apply STRIDE.
4. Rate risks.
5. Define controls.
6. Plan verification.

**Deliverable:** threat model.

**Acceptance criteria:** coverage of device, network, cloud, and update paths.

### Lab 16.2 — De-identification audit

**Objective:** Evaluate a de-identified dataset for residual risk.

**Steps:**

1. Inspect direct identifiers.
2. Inspect quasi-identifiers.
3. Check free text and metadata.
4. Attempt linkage with public data in a safe environment.
5. Document residual risk.

**Deliverable:** audit report.

**Acceptance criteria:** no real PHI; specific risks; remediation.

### Lab 16.3 — Fairness audit

**Objective:** Measure subgroup performance and trade-offs.

**Steps:**

1. Define groups and outcomes.
2. Compute TPR, FPR, PPV, calibration.
3. Apply a mitigation.
4. Measure trade-offs.
5. Recommend a policy.

**Deliverable:** fairness report.

**Acceptance criteria:** multiple definitions; trade-offs; stakeholder input.

### Lab 16.4 — Membership inference

**Objective:** Test whether a model leaks training membership.

**Steps:**

1. Train a model on a small dataset.
2. Implement a simple membership inference attack.
3. Measure advantage.
4. Apply regularization or DP.
5. Re-test.

**Deliverable:** privacy attack report.

**Acceptance criteria:** ethical use of synthetic data; measured leakage.

### Lab 16.5 — Ethics impact assessment

**Objective:** Complete an ethics impact assessment for a product.

**Steps:**

1. Identify stakeholders.
2. Map benefits and harms.
3. Assess consent and power.
4. Assess equity and accessibility.
5. Define mitigations and review.

**Deliverable:** assessment.

**Acceptance criteria:** affected communities included; harms specific.

## 13. Assignment: privacy and ethics impact assessment

Write a 2,000-word assessment:

1. System description.
2. Data flows and assets.
3. Legal basis and consent.
4. Privacy risks and controls.
5. Security threats and controls.
6. Fairness analysis.
7. Accessibility and neurodiversity.
8. Governance and accountability.
9. Residual risk.
10. Monitoring and review.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Privacy | 20% | Generic | Data minimization and de-identification | PETs and residual risk |
| Security | 20% | Ignores | Threat model and controls | Adversarial testing |
| Fairness | 20% | One metric | Multiple metrics and trade-offs | Mitigation and monitoring |
| Ethics | 20% | Generic | Stakeholders and harms | Community engagement |
| Governance | 10% | No owner | Roles defined | Auditable accountability |
| Communication | 10% | Technical | Clear | Board-ready |

## 15. Regulatory, ethical, and commercial lens

- **HIPAA:** US privacy and security.
- **GDPR:** EU data protection.
- **EU AI Act:** high-risk obligations.
- **FTC:** consumer health data and deceptive claims.
- **Sectoral laws:** state privacy laws, genetic privacy, mental health.
- **Commercial:** trust and security are procurement requirements.

## 16. Failure modes and red flags

- Treating de-identification as anonymization.
- Sending PHI to unapproved tools.
- No threat model.
- No access reviews.
- Ignoring subgroup harm.
- No consent for secondary use.
- No incident response.
- No model documentation.
- Assuming a vendor's security is sufficient.
- Ignoring accessibility.

## 17. Interview and discussion questions

1. **What is the difference between de-identification and anonymization?**
   Hint: residual re-identification risk.
2. **How do you threat-model a clinical AI system?**
   Hint: data flows, assets, trust boundaries, STRIDE.
3. **What is differential privacy?**
   Hint: calibrated noise with a privacy budget.
4. **What are the limits of federated learning?**
   Hint: leakage, heterogeneity, coordination, inference.
5. **How do you choose a fairness definition?**
   Hint: decision, harm, stakeholder, legal context.
6. **What is label bias?**
   Hint: target reflects access or documentation, not need.
7. **How do you handle incidental findings?**
   Hint: consent, counseling, actionability, policy.
8. **What is membership inference?**
   Hint: determine whether a record was in training.
9. **How do you govern model access?**
   Hint: review board, documentation, monitoring, audit.
10. **What does the EU AI Act require?**
    Hint: risk management, data governance, transparency, human oversight, monitoring.
11. **How do you involve patients?**
    Hint: advisory boards, participatory design, compensation.
12. **What would make you stop a project on ethical grounds?**
    Hint: unacceptable harm, no consent, no mitigation, inequitable access.

## 18. Key terms

| Term | Definition |
|---|---|
| PHI | Protected health information |
| PII | Personally identifiable information |
| De-identification | Removing or transforming identifiers |
| Pseudonymization | Replacing identifiers with a key |
| Differential privacy | Noise-based privacy guarantee |
| Federated learning | Distributed training without central data |
| Synthetic data | Artificially generated records |
| Threat model | Structured analysis of security risks |
| STRIDE | Threat categorization framework |
| Membership inference | Privacy attack on training data |
| Demographic parity | Equal positive prediction rates |
| Equal opportunity | Equal true positive rates |
| Calibration within groups | Predicted risk matches observed by group |
| DPIA | Data protection impact assessment |

## 19. Sources and further reading

### Law and guidance
- HHS HIPAA: https://www.hhs.gov/hipaa/
- GDPR: https://gdpr.eu/
- EU AI Act: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- FTC Health Privacy: https://www.ftc.gov/business-guidance/privacy-security/health-privacy
- NIST Privacy Framework: https://www.nist.gov/privacy-framework
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- WHO, *Ethics and governance of AI for health*. https://www.who.int/publications/i/item/9789240029200

### Methods and evidence
- Obermeyer, Z. et al., *Science*, 2019. https://doi.org/10.1126/science.aax2342
- Sjoding, M. W. et al., *New England Journal of Medicine*, 2020. https://doi.org/10.1056/NEJMc2029240
- Barocas, S., Hardt, M., Narayanan, A., *Fairness and Machine Learning*. https://fairmlbook.org/
- Fairlearn: https://fairlearn.org/
- OpenMined: https://www.openmined.org/

> **Next:** [Module 17](17_product_economics_leadership.md) connects trust to product, economics, and adoption.
