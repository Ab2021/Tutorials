# Module 14 — Medical Devices, SaMD, and Regulation

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** MedTech & Imaging core; required for product roles
> **Prerequisites:** Modules 05, 06, 10, 11
> **Tools:** Regulatory guidance, risk register, requirements traceability matrix, model card, cybersecurity threat model
> **Industry question:** "Is this software a medical device, what claim are we making, and what evidence and controls must exist before a patient is exposed?"

---

## 1. Why industry cares

A model in a notebook is a research artifact. A model that informs diagnosis or treatment is potentially a regulated medical device. The difference is not the algorithm. It is the **intended use, the claim, the risk, and the lifecycle controls**.

Regulatory mistakes are expensive:

- Building a product that requires a PMA when a 510(k) was assumed.
- Making a claim that the evidence does not support.
- Changing a model after clearance without a change-control plan.
- Missing cybersecurity requirements.
- Failing a quality-system audit.
- Launching in the EU without MDR clinical evaluation.

This module gives data scientists the regulatory literacy to design products that can actually reach patients. For the hardware side — sensors, analog front ends, semiconductors, firmware, electrical safety, and robotics — see Module 14A.

## 2. Learning outcomes

By the end of this module, you can:

- Determine whether a software function is likely a medical device, SaMD, SiMD, or CDS.
- Write an intended-use statement and indications for use.
- Identify the likely FDA pathway and predicate strategy.
- Explain GMLP, PCCP, and total product lifecycle requirements.
- Map ISO 13485, IEC 62304, ISO 14971, and IEC 62366 to engineering artifacts.
- Design a risk management file and requirements traceability matrix.
- Plan analytical, clinical, usability, and cybersecurity evidence.
- Navigate EU MDR/IVDR and the EU AI Act at a working level.
- Design post-market surveillance and model monitoring.
- Build a regulatory evidence package for a data-driven product.

## 3. Mental model: claim → risk → controls → evidence

```
Intended use / claim
        │
        ▼
Risk classification
        │
        ├── FDA pathway (510(k), De Novo, PMA)
        ├── EU classification (MDR/IVDR, AI Act)
        └── Other markets
        │
        ▼
Quality management system
        ├── Design controls
        ├── Risk management (ISO 14971)
        ├── Software lifecycle (IEC 62304)
        ├── Usability (IEC 62366)
        ├── Cybersecurity
        └── Clinical evaluation
        │
        ▼
Evidence package → Submission → Clearance/CE → Launch → Post-market surveillance
```

## 4. Definitions

| Term | Definition |
|---|---|
| Medical device | Article intended for diagnosis, prevention, monitoring, treatment, or alleviation of disease |
| SaMD | Software as a Medical Device; software intended for a medical purpose, not part of a hardware device |
| SiMD | Software in a Medical Device; embedded software |
| CDS | Clinical decision support; software that supports clinical decisions |
| Intended use | What the product does, for whom, and in what setting |
| Indications for use | Specific clinical conditions and populations |
| Predicate | Legally marketed device used to support substantial equivalence |
| Substantial equivalence | Same intended use and similar technological characteristics, or different with comparable safety/effectiveness |
| GMLP | Good Machine Learning Practice |
| PCCP | Predetermined Change Control Plan |
| TPLC | Total Product Life Cycle |
| QMS | Quality Management System |
| Design controls | Structured process for design inputs, outputs, review, verification, validation, transfer, changes |
| Risk management | Systematic process for identifying, evaluating, controlling, and monitoring risks |
| Post-market surveillance | Monitoring after launch |

## 5. Is it a device?

Ask:

1. Does the software make or inform a medical decision?
2. Is the output intended for diagnosis, treatment, prevention, or monitoring of disease?
3. Does it provide patient-specific output?
4. Is it intended for use by a clinician or patient?
5. Does it merely support administrative or operational tasks?
6. Does it meet an FDA CDS exclusion?

### 5.1 FDA CDS criteria

The FDA's CDS guidance describes criteria for software functions that may not be devices. The analysis is fact-specific. Key considerations include:

- Whether the software displays, analyzes, or interprets patient-specific information.
- Whether the user can independently review the basis for the recommendation.
- Whether the software is intended to support, not replace, clinical judgment.
- Whether the recommendation is specific and actionable.

**Rule:** do not self-declare "CDS-exempt." Involve regulatory counsel and document the analysis.

### 5.2 Examples

| Function | Likely status | Why |
|---|---|---|
| Claims denial dashboard | Not a device | Administrative |
| Hospital bed capacity forecast | Not a device | Operational |
| Sepsis risk score in EHR | Likely device/CDS | Patient-specific diagnosis/treatment support |
| Arrhythmia detection on wearable | Device | Diagnosis/monitoring |
| Autonomous retinopathy screening | Device | Diagnosis without clinician review |
| Clinical documentation summarizer | Depends | If it summarizes for a clinician, lower risk; if it diagnoses, device |
| Patient triage chatbot | Depends | If it directs care, likely device |

## 6. FDA pathways

| Pathway | Use | Evidence | Timeline (typical) |
|---|---|---|---|
| 510(k) | Substantial equivalence to predicate | Performance testing, often clinical | 90-day review goal |
| De Novo | Novel low/moderate-risk device, no predicate | Benefit-risk, clinical evidence | 90-day review goal |
| PMA | High-risk device | Extensive clinical evidence, manufacturing | 180-day review goal |
| Exempt | Low-risk devices | None or limited | Immediate |
| HDE | Humanitarian use | Limited evidence | Varies |

### 6.1 Predicate strategy

- Identify legally marketed predicate.
- Compare intended use and technological characteristics.
- Address differences with performance data.
- Consider multiple predicates.
- Check FDA's 510(k) database and product codes.

### 6.2 AI/ML-specific

- GMLP principles.
- PCCP for planned changes.
- Transparency and labeling.
- Post-market monitoring.
- Cybersecurity.
- Bias and subgroup performance.

## 7. GMLP and PCCP

### 7.1 GMLP

The FDA, MHRA, and Health Canada published guiding principles:

1. Leverage multidisciplinary expertise.
2. Implement good software engineering and security practices.
3. Ensure representative data and data quality.
4. Use independent test sets.
5. Use reference standards.
6. Test in clinically relevant conditions.
7. Address performance of human-AI teams.
8. Provide clear information.
9. Monitor deployed models.
10. Manage risk and change.

### 7.2 PCCP

A PCCP describes planned modifications and the associated verification and validation.

It includes:

- Description of modifications.
- Impact assessment.
- Verification and validation protocols.
- Acceptance criteria.
- Monitoring and update procedures.

**Rule:** a PCCP covers planned changes, not arbitrary continuous learning.

## 8. Quality management system

### 8.1 ISO 13485

Requirements for a quality management system for medical devices:

- Document control.
- Management responsibility.
- Resource management.
- Product realization.
- Measurement, analysis, improvement.
- Corrective and preventive action (CAPA).
- Design controls.

### 8.2 IEC 62304

Software lifecycle processes:

- Software development planning.
- Requirements analysis.
- Architectural design.
- Detailed design.
- Unit implementation and verification.
- Integration and integration testing.
- System testing.
- Release.
- Maintenance.
- Risk management integration.
- Configuration management.
- Problem resolution.

Software safety classes:

- Class A: no injury possible.
- Class B: non-serious injury possible.
- Class C: death or serious injury possible.

### 8.3 ISO 14971

Risk management process:

1. Risk management plan.
2. Risk analysis.
3. Risk evaluation.
4. Risk control.
5. Residual risk evaluation.
6. Risk management report.
7. Production/post-production monitoring.

### 8.4 IEC 62366

Usability engineering:

- Use specification.
- Hazard-related use scenarios.
- User interface design.
- Formative evaluation.
- Summative evaluation.

### 8.5 Cybersecurity

- Threat modeling.
- Security risk assessment.
- Secure design.
- Security testing.
- Vulnerability monitoring.
- Patch management.
- Coordinated disclosure.

## 9. EU regulation

### 9.1 MDR / IVDR

- Classification rules (Class I, IIa, IIb, III).
- Conformity assessment with notified body.
- Clinical evaluation and post-market clinical follow-up.
- Technical documentation.
- Unique Device Identification (UDI).
- Economic operator obligations.

### 9.2 EU AI Act

The AI Act introduces risk-based obligations:

- Prohibited practices.
- High-risk AI systems: risk management, data governance, technical documentation, transparency, human oversight, accuracy, robustness, cybersecurity, post-market monitoring.
- Limited-risk transparency obligations.
- General-purpose AI obligations.

**Interaction:** a medical AI product may need to satisfy both MDR/IVDR and the AI Act. Do not treat them as alternatives.

### 9.3 2026 regulatory watchlist

These items are current as of 24 October 2026. Verify before relying on them.

| Item | Status | Practical implication |
|---|---|---|
| FDA PCCP final guidance | Issued December 2024; reissued August 2025 | Plan model changes through a pre-specified modification protocol and impact assessment |
| FDA QMSR | Effective 2 February 2026; incorporates ISO 13485:2016 | Align quality system documentation with QMSR, not only legacy 21 CFR 820 |
| FDA computer software assurance | Final guidance September 2025; revised February 2026 | Use risk-based assurance proportionate to intended use and record integrity |
| FDA generative AI device discussion paper | August 2026; comments closed October 2026 | No final rule; pre-submission dialogue and local validation remain essential |
| FDA LDT rule | Vacated March 2025; FDA rescinded September 2025 | Many lab-developed tests remain under CLIA/CMS, not FDA device review |
| EU AI Act high-risk obligations for medical devices | Regulation (EU) 2026/1744 (Digital Omnibus) defers Annex I products to 2 August 2028; Annex III systems to 2 December 2027 | MDR/IVDR still apply; dual compliance planning continues |
| EHDS | Regulation in force 2025; secondary-use rules phase in 2029–2031 | Plan consent, access, and secure-processing requirements for EU data |
| HTI-1 decision support interventions | USCDI v3 baseline January 2026; DSI transparency criteria | Model documentation may be required in EHR certification workflows |
| CMS prior authorization APIs | Phased January 2026–January 2027 | Payer-facing automation must support electronic prior authorization |
| HIPAA Security Rule NPRM | Proposed January 2025; final rule not published at the source cutoff | Expect stronger encryption, MFA, asset inventory, and contingency requirements if finalized |
| HTI-2 / TEFCA | HTI-2 finalized December 2024; QHIN governance | Health-information exchange and certification obligations |
| Information blocking | ONC rule in effect | Data access and exchange may be a product requirement |
| UDI / EUDAMED | US UDI and EU device database obligations | Traceability, recalls, and post-market data |
| NIST AI RMF / ISO/IEC 42001 | Voluntary frameworks, increasingly referenced in procurement | AI governance and management-system evidence |
| FDA AEMS (formerly FAERS) | FAERS renamed/merged in March 2026 | Adverse-event analytics and post-market surveillance |

**Rule:** regulatory dates are not product deadlines. Build the evidence and quality system before the date, not after.

## 10. Evidence package

| Evidence | Purpose | Artifacts |
|---|---|---|
| Intended use | Define claim and population | Labeling, indications |
| Analytical validation | Does the algorithm measure what it claims? | Test datasets, metrics |
| Clinical validation | Does it improve diagnosis/outcome? | Reader study, prospective trial |
| Usability | Can users use it safely? | Formative/summative reports |
| Cybersecurity | Is it secure? | Threat model, penetration test |
| Software verification | Does it meet requirements? | Test plan, traceability |
| Risk management | Are risks controlled? | Risk file |
| Post-market plan | How will we monitor? | PMS/PMCF plan |

## 11. Industry case studies

### 11.1 IDx-DR

IDx-DR received De Novo authorization for autonomous diabetic retinopathy screening. The evidence included a prospective clinical trial and a clear intended use.

**Lesson:** narrow claims, prospective evidence, and workflow integration can support autonomy.

Source: FDA De Novo DEN180001. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### 11.2 Apple ECG

Apple's ECG app received De Novo authorization for irregular rhythm notification and ECG recording. Labeling and intended use were tightly defined.

**Lesson:** consumer devices can be regulated devices; claims and labeling matter.

Source: FDA De Novo DEN180044. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf

### 11.3 Viz.ai LVO

Viz.ai received clearance for CT angiography triage of large-vessel occlusion, emphasizing workflow and time to treatment.

**Lesson:** the product is a workflow intervention, not only a detector.

Source: FDA 510(k) database; company information.

### 11.4 Epic Sepsis Model

The Epic sepsis model was widely deployed but external validation showed lower performance than claimed. It illustrates the gap between clearance/marketing and local evidence.

**Lesson:** regulatory status does not replace local validation and monitoring.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### 11.5 EU AI Act and medical AI

Medical AI systems can be high-risk under the EU AI Act, requiring risk management, data governance, transparency, human oversight, and post-market monitoring.

**Lesson:** regulatory strategy must cover both device law and AI law.

Source: Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

## 12. Labs

### Lab 14.1 — Regulatory triage

**Objective:** Classify five product concepts.

**Steps:**

1. Write intended use for each.
2. Determine device/SaMD/CDS status.
3. Identify FDA pathway.
4. Identify EU classification.
5. List evidence needs.
6. Document assumptions.

**Deliverable:** triage memo.

**Acceptance criteria:** fact-specific; counsel review flagged; no self-declaration.

### Lab 14.2 — Intended-use statement

**Objective:** Write and critique intended-use statements.

**Steps:**

1. Draft for a sepsis model.
2. Draft for a wearable AF detector.
3. Draft for a documentation summarizer.
4. Compare with cleared labeling.
5. Identify overclaims.

**Deliverable:** statements and critique.

**Acceptance criteria:** population, setting, user, output, and limitation explicit.

### Lab 14.3 — Risk management file

**Objective:** Build a risk register for an AI device.

**Steps:**

1. Identify hazards.
2. Estimate severity and probability.
3. Identify controls.
4. Evaluate residual risk.
5. Link to requirements and tests.

**Deliverable:** risk register.

**Acceptance criteria:** traceability; residual risk; monitoring.

### Lab 14.4 — Software requirements and traceability

**Objective:** Write requirements and a traceability matrix.

**Steps:**

1. Write functional and performance requirements.
2. Write safety requirements.
3. Map to design and tests.
4. Identify verification evidence.
5. Check completeness.

**Deliverable:** requirements and matrix.

**Acceptance criteria:** testable; traceable; no orphan requirements.

### Lab 14.5 — PCCP draft

**Objective:** Draft a predetermined change control plan.

**Steps:**

1. Define planned modifications.
2. Assess impact.
3. Specify verification and validation.
4. Define acceptance criteria.
5. Define monitoring.

**Deliverable:** PCCP draft.

**Acceptance criteria:** changes bounded; evidence plan; no open-ended learning.

### Lab 14.6 — Cybersecurity threat model

**Objective:** Threat-model a connected device.

**Steps:**

1. Diagram data flows.
2. Identify assets and trust boundaries.
3. Enumerate threats.
4. Rate risk.
5. Define controls.
6. Plan testing.

**Deliverable:** threat model.

**Acceptance criteria:** coverage of network, device, cloud, and update paths.

## 13. Assignment: regulatory evidence package

Write a 2,500-word package:

1. Product description.
2. Intended use and indications.
3. Device classification and pathway.
4. Predicate analysis.
5. Risk analysis.
6. Software lifecycle and requirements.
7. Analytical and clinical validation.
8. Usability.
9. Cybersecurity.
10. Labeling and post-market plan.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Intended use | 20% | Vague | Specific population and claim | Aligned with evidence and labeling |
| Classification | 20% | Guesses | Pathway with rationale | Predicate and jurisdiction strategy |
| Risk | 20% | Generic list | ISO 14971 process | Traceable controls and residual risk |
| Evidence | 20% | Accuracy only | Analytical and clinical plan | Prospective and workflow evidence |
| Lifecycle | 10% | Ignores | QMS and monitoring | PCCP and post-market surveillance |
| Communication | 10% | Technical | Clear | Submission-ready |

## 15. Regulatory, ethical, and commercial lens

- **FDA:** pre-submission is your friend. Engage early.
- **EU:** notified body capacity and MDR timelines are real constraints.
- **AI Act:** high-risk obligations are substantial; plan for them.
- **Reimbursement:** FDA clearance does not guarantee payment.
- **Liability:** labeling and intended use define the boundary.
- **Equity:** subgroup performance and access are regulatory and ethical issues.

## 16. Failure modes and red flags

- Building before defining intended use.
- Assuming "CDS" without analysis.
- No predicate or pathway strategy.
- Ignoring cybersecurity.
- No usability testing.
- No post-market plan.
- Open-ended continuous learning without PCCP.
- Training data not representative.
- No traceability.
- Treating a cleared product as locally validated.

## 17. Interview and discussion questions

1. **What makes software a medical device?**
   Hint: intended medical purpose, patient-specific output, claim.
2. **What is SaMD?**
   Hint: software as a medical device, not part of hardware.
3. **How do you choose a FDA pathway?**
   Hint: risk, predicate, novelty, evidence.
4. **What is a PCCP?**
   Hint: pre-authorized planned model changes.
5. **What is IEC 62304?**
   Hint: medical device software lifecycle.
6. **What is ISO 14971?**
   Hint: risk management.
7. **What is the EU AI Act's relevance?**
   Hint: high-risk obligations layered on MDR/IVDR.
8. **How do you validate a model for a device?**
   Hint: analytical, clinical, usability, cybersecurity, post-market.
9. **What is a risk control?**
   Hint: design, protective, information for safety.
10. **Why is local validation still needed after clearance?**
    Hint: population, workflow, data drift.
11. **What is a traceability matrix?**
    Hint: requirements → design → tests → evidence.
12. **How do you handle a model update?**
    Hint: change control, PCCP, revalidation, monitoring.

## 18. Key terms

| Term | Definition |
|---|---|
| SaMD | Software as a Medical Device |
| SiMD | Software in a Medical Device |
| CDS | Clinical decision support |
| 510(k) | Premarket notification via substantial equivalence |
| De Novo | Classification for novel low/moderate-risk device |
| PMA | Premarket approval for high-risk device |
| Predicate | Legally marketed comparator device |
| GMLP | Good Machine Learning Practice |
| PCCP | Predetermined Change Control Plan |
| QMS | Quality Management System |
| CAPA | Corrective and preventive action |
| IEC 62304 | Medical device software lifecycle standard |
| ISO 14971 | Risk management standard |
| ISO 13485 | Medical device QMS standard |
| IEC 62366 | Usability engineering standard |
| MDR / IVDR | EU medical device / IVD regulations |
| PMS / PMCF | Post-market surveillance / clinical follow-up |
| UDI | Unique Device Identification |

## 19. Sources and further reading

### FDA
- FDA, *Artificial Intelligence-Enabled Medical Devices* (device list and program page). https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- FDA, *Clinical Decision Support Software*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- FDA, *Good Machine Learning Practice*. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- FDA, *PCCP guidance*. https://www.fda.gov/media/166704/download
- FDA, *Cybersecurity*. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- FDA 510(k) database: https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm
- FDA De Novo DEN180001: https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
- FDA De Novo DEN180044: https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
- FDA, *Quality Management System Regulation (QMSR)*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
- FDA, *Computer Software Assurance*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-management-system-software
- FDA, *Generative AI-Enabled Medical Devices: Discussion Paper*, 2026. https://www.fda.gov/medical-devices/digital-health-center-excellence/considerations-regulation-generative-ai-enabled-medical-devices-discussion-paper-and-request
- FDA, *Laboratory Developed Tests*. https://www.fda.gov/medical-devices/in-vitro-diagnostics/laboratory-developed-tests
- ONC, *HTI-1 Final Rule*. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
- CMS, *Interoperability and Prior Authorization*. https://www.cms.gov/priorities/electronic-prior-authorization/overview

### International
- EU MDR: https://eur-lex.europa.eu/eli/reg/2017/745/oj
- EU IVDR: https://eur-lex.europa.eu/eli/reg/2017/746/oj
- EU AI Act: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6, *AI Act and medical devices*. https://health.ec.europa.eu/system/files/2025-06/mdcg_2025-6_en.pdf
- EHDS Regulation 2025/327. https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en
- IMDRF: https://www.imdrf.org/
- ISO 13485: https://www.iso.org/standard/59752.html
- IEC 62304: https://www.iec.ch/
- ISO 14971: https://www.iso.org/standard/72704.html
- IEC 62366: https://www.iec.ch/

### Methods and evidence
- Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626
- FDA, *GMLP*. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles

> **Next:** [Module 15](15_deployment_mlops.md) moves from approval to production and monitoring.
