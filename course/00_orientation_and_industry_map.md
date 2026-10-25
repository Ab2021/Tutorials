# Module 00 — Orientation and the 2026 Industry Map

> **Program:** Biomedical Data Science for Industry (2026 Edition)
> **Time:** 16 hours (6 lecture, 4 workshop, 6 project)
> **Track fit:** Common core
> **Prerequisites:** Basic programming and statistics; no clinical background required
> **Tools:** Python, Jupyter, Git, a modern LLM assistant, a structured note-taking system
> **Industry question:** "Where does data science actually create value in biomedicine — and what evidence, workflow, and regulatory burden must accompany that value?"

---

## 1. Why this module exists

Most biomedical data science courses teach model building against a tidy benchmark. Industry does not buy models. It buys **decisions that are better, faster, cheaper, safer, or more equitable** — and it must be able to defend those decisions to clinicians, regulators, payers, and patients.

The failure pattern is remarkably consistent:

1. A team obtains an interesting dataset.
2. It trains a model with strong holdout performance.
3. Nobody can say which decision the model changes.
4. The data pipeline cannot run in the real workflow.
5. The validation does not match the deployment population.
6. The regulatory pathway was never scoped.
7. A clinician does not trust or cannot use the output.
8. The project dies in pilot purgatory.

This module installs the operating system for the rest of the program: **start with the decision, design the evidence, engineer the data, build the simplest adequate model, validate in context, and plan for production and governance from day one.**

## 2. Learning outcomes

By the end of this module, you can:

- Explain the value chain of biomedical data science across providers, payers, pharma, diagnostics, devices, sensors, and public health.
- Distinguish research, quality-improvement, clinical-decision-support, and regulated-device use cases, and identify what changes when a model becomes a product.
- Map a proposed project to a decision, user, workflow, outcome, and economic buyer.
- Identify the data modality, access pathway, and minimum evidence package for a project.
- Recognize the major regulatory regimes that can apply to a data product and when to involve regulatory, quality, legal, and clinical stakeholders.
- Choose an appropriate track and capstone direction.
- Write a one-page project charter that survives scrutiny from a clinical champion, an engineering lead, and a compliance officer.
- Evaluate an AI product claim critically: population, comparator, endpoint, validation design, and deployment context.

## 3. The 2026 industry context

### 3.1 From model-centric to decision-centric

The competitive advantage is no longer access to a standard architecture. Open-weight models, cloud APIs, and mature libraries make model building cheap. The scarce assets are:

- **Trusted, longitudinal, well-characterized data.**
- **Clinical and operational workflow access.**
- **Regulatory and quality infrastructure.**
- **Evaluation evidence in the intended population.**
- **Distribution and reimbursement.**
- **The ability to monitor and maintain performance after deployment.**

A data scientist who understands only the model is replaceable. A data scientist who can connect model behavior to a decision, a workflow, a contract, and a regulatory submission is not.

### 3.2 Five structural shifts

| Shift | What changed | Consequence for your work |
|---|---|---|
| Foundation models and LLMs | General-purpose models can be adapted with prompting, retrieval, or fine-tuning | Evaluation, grounding, privacy, and change control matter more than architecture novelty |
| Multimodal data | Imaging, text, waveforms, labs, genomics, and wearables increasingly converge | Data engineering and missing-modality robustness become core skills |
| Real-world evidence | Regulators and payers increasingly accept, and demand, data from routine care | Causal inference and data provenance are not optional |
| Regulated AI | AI-enabled devices, clinical software, and the EU AI Act create explicit lifecycle obligations | Quality management, risk management, and post-market monitoring must be designed in |
| Operational pressure | Staffing shortages, drug pricing, device recalls, and reimbursement reform | Solutions must show workflow value and total cost of ownership, not just AUC |

### 3.3 The biomedical data value chain

```
Biology / patient / operation
        ↓
Sensor, EHR, claims, lab, imaging, omics, trial, registry
        ↓
Ingestion, linkage, de-identification, standardization
        ↓
Cohort construction, phenotyping, feature engineering
        ↓
Analysis, prediction, causal inference, generation
        ↓
Clinician / patient / scientist / operator decision
        ↓
Outcome, workflow, cost, equity, safety
        ↓
Monitoring, audit, revalidation, model retirement
```

Every arrow is a place where projects fail. The course spends time on the arrows, not only the box labeled "model."

### 3.4 Industry segments and their core problems

| Segment | Typical data | Core decisions | Common products | Primary constraints |
|---|---|---|---|---|
| Health systems / providers | EHR, imaging, labs, scheduling, claims | Who needs outreach? Which patient deteriorates? How to reduce length of stay? | Risk scores, triage, documentation, capacity forecasting | HIPAA, clinical safety, workflow, liability |
| Payers / insurers | Claims, eligibility, pharmacy, utilization | Who is high-risk? What intervention saves cost? Is care appropriate? | Population health, utilization management, fraud detection | Fairness, regulation, member experience |
| Pharma / biotech | Omics, assay, preclinical, trial, RWE | Which target? Which patient? Will the trial succeed? | Target discovery, biomarker, trial optimization, pharmacovigilance | GxP, privacy, evidence standards, long timelines |
| Medical devices | Sensor, imaging, device telemetry | Does this patient have the condition? Is the device safe? | SaMD, monitoring, diagnostics, closed-loop control | FDA/CE, IEC 62304, ISO 14971, cybersecurity |
| Diagnostics | Imaging, pathology, molecular | Who has disease? Which therapy? | Triage, detection, companion diagnostics | CLIA/CAP, IVDR, clinical validation |
| Wearables / digital health | PPG, accelerometry, ECG, sleep, GPS | Is this event real? Will the user adhere? | Screening, monitoring, digital biomarkers | Validation, battery, equity, engagement |
| Public health | Surveillance, registries, administrative | Where is risk rising? Where to allocate? | Dashboards, outbreak models, policy simulation | Privacy, timeliness, communication |

### 3.5 What "good" looks like in 2026

A credible industry project has:

- A named **decision owner** and a described **workflow change**.
- A **target population** and an explicit **intended use** statement.
- A **comparator**: current practice, not a strawman.
- A **primary metric** tied to the decision, plus safety and equity metrics.
- **Temporal and external validation**, not only random splits.
- **Calibration** and **uncertainty**, not only discrimination.
- A **data provenance** and **versioning** story.
- A **monitoring** and **change-control** plan.
- A **regulatory and ethical** triage.
- A **health-economic** or operational case.
- A **dissemination** artifact that a non-modeler can act on.

## 4. The program architecture

### 4.1 Three terms, 40 weeks

| Term | Weeks | Modules | Theme | Major deliverable |
|---|---:|---|---|---|
| 1 | 1–12 | 00–06 incl. 01A | Clinical, data, statistical, and ML foundations | Reproducible risk-model report on a real cohort |
| 2 | 13–26 | 07–14B | Modalities, hardware, efficient AI, and industry domains | Modality deep dive, hardware/efficiency plan, domain playbook |
| 3 | 27–40 | 15–22 plus 23–26 electives | Production, governance, product, capstone, advanced depth | Deployable prototype, evidence dossier, demo |

### 4.2 Module dependency map

```
00 Orientation
 ├── 01 Clinical foundations
 │    ├── 08 EHR / claims / operations
 │    ├── 10 Imaging
 │    └── 11 Signals and wearables
 ├── 01A Foundations, units, measurement, assumptions
 │    ├── 04 Statistics and causal inference
 │    ├── 06 ML foundations
 │    └── 14A Hardware, sensors, chips, robotics
 ├── 02 Data landscape
 │    └── 03 Data engineering and interoperability
 │         ├── 08 EHR / claims
 │         ├── 09 Clinical NLP
 │         └── 15 MLOps
 ├── 04 Statistics and causal inference
 │    ├── 05 Study design and evidence
 │    ├── 13 Biopharma clinical development
 │    └── 18 Quality and reporting
 ├── 06 ML foundations
 │    ├── 07 Deep learning, multimodal, LLMs
 │    ├── 09 Clinical NLP
 │    ├── 10 Imaging
 │    └── 11 Signals
 ├── 12 Omics and precision medicine
 ├── 14 Devices, SaMD, regulation
 │    └── 14A Hardware, sensors, chips, robotics (also requires 11)
 │         └── 14B Efficient AI: distillation, quantization, on-device inference
 ├── 16 Privacy, security, ethics
 ├── 17 Product, economics, leadership
 └── 19 Capstone and hackathons
```

Modules are not strictly sequential. The recommended order is 00 → 01 → 01A → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 14A → 14B → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23–26 (optional electives). Track electives can be taken in parallel after the core.

### 4.3 Weekly rhythm

| Component | Time | Purpose |
|---|---:|---|
| Lecture / seminar | 3 h | Concepts, industry context, worked examples |
| Lab | 3 h | Reproducible implementation on realistic data |
| Case clinic | 2 h | Critique a real product, paper, or regulatory decision |
| Project studio | 4 h | Team work with weekly deliverables and critique |
| Independent reading | 2–4 h | Primary sources, documentation, book chapters |
| Assessment / portfolio | 2–3 h | Rubric work, reflection, portfolio updates |

A 17–19 hour week for 40 weeks is roughly **680–760 hours**. The core modules total about **690–730 hours**; Module 14B adds 28 hours; advanced electives 23–26 add up to 98 more when labs, case clinics, project studio, and the capstone are included; use the module numbers for planning, not accounting. Cohorts can compress or extend the schedule.

### 4.4 Tracks

| Track | Required modules | Recommended electives | Capstone archetypes |
|---|---|---|---|
| Health Systems Analytics | Core (00, 01, 01A, 02–06); 08, 09, 14, 15, 16, 17, 18, 19 | 07, 10, 11, 14A, 14B, 23, 25, 26, 20, 21 | Sepsis early warning, readmission reduction, capacity forecasting, documentation burden |
| Biopharma & Translational | Core (00, 01, 01A, 02–06); 07, 12, 13, 14, 15, 16, 17, 18, 19 | 08, 09, 10, 11, 14A, 14B, 23, 25, 20, 21 | Biomarker discovery, trial recruitment, external control arm, pharmacovigilance signal detection |
| MedTech, Sensors & Imaging | Core (00, 01, 01A, 02–06); 07, 10, 11, 14, 14A, 15, 16, 17, 18, 19 | 08, 09, 12, 13, 14B, 23, 24, 26, 20, 21 | Wearable arrhythmia detection, imaging triage, device safety, SaMD evidence package |

### 4.5 Assessment

| Assessment | Weight | Individual/team | What it measures | Brief/rubric |
|---|---:|---|---|---|
| Weekly labs | 20% | Individual | Reproducible execution and judgment | Module labs |
| Case critiques | 15% | Individual | Critical appraisal and industry reasoning | 19 §11A.3 |
| Midterm evidence dossier | 20% | Team | Study design, validation, regulatory triage | 19 §11A.1 |
| Track playbook | 15% | Team | Domain depth and translation | 19 §11A.2 |
| Capstone | 30% | Team + individual defense | End-to-end delivery under constraints | 19 §12 |

Passing requires at least 50% on the individual capstone defense, regardless of team score. This prevents social loafing and protects the credential.

## 5. The industry map: who you will work with

### 5.1 Roles

| Role | Primary output | What they need from you | What you need from them |
|---|---|---|---|
| Clinical champion | Adoption, workflow design | Trustworthy output, minimal disruption | Access, clinical context, validation population |
| Product manager | Scope, prioritization, business case | Trade-offs, estimates, risks | User access, success metrics, roadmap |
| Regulatory affairs | Submission strategy, claims | Evidence of intended use, validation, change control | Predicate analysis, guidance interpretation |
| Quality engineer | Design controls, traceability | Requirements, risk analysis, verification | QMS process, audit readiness |
| Data engineer | Pipelines, availability, lineage | Schemas, SLAs, quality rules | Reliable data, documentation |
| Hardware/robotics engineer | Sensor, firmware, robot, and safety design | Data constraints, calibration, failure modes | HIL tests, logs, device state |
| Biostatistician | Estimands, analysis plans | Pre-specification, sensitivity analyses | Rigor, multiplicity control |
| Health economist | Value dossier, budget impact | Cost and outcome inputs | Payer evidence requirements |
| Privacy / legal | Data use agreements, consent | Data flow, minimum necessary use | Contract terms, jurisdiction |
| Security | Threat model, controls | Architecture, data flows | Pen testing, incident response |
| Clinician end user | Care decisions | Actionable, calibrated, explainable output | Feedback, workflow constraints |

### 5.2 The three languages

You will translate among:

- **Clinical language:** symptoms, diagnoses, pathways, guidelines, harm.
- **Engineering language:** schemas, latency, availability, versioning, failure modes.
- **Commercial language:** value, cost, reimbursement, adoption, risk.

A project charter that only speaks one language will fail. Module 17 develops the translation skill explicitly.

## 6. The project charter

Every project in this program begins with a one-page charter. The charter is not bureaucracy; it is a forcing function.

### 6.1 Charter template

```markdown
# Project charter

## Decision
What decision will change? Who owns it? How often is it made?

## User and workflow
Who uses the output? At what point? What do they do differently?

## Intended use and population
Who is eligible? Who is excluded? In what setting? With what data available at decision time?

## Comparator
What happens today? What is the current performance, cost, and workflow?

## Outcome
Primary clinical/operational outcome. Secondary outcomes. Safety. Equity.

## Data
Sources, years, linkage, access, quality, missingness, labels.

## Evidence plan
Design, splits, validation sites, metrics, calibration, uncertainty, subgroup analyses.

## Regulatory and ethical triage
Is this research, QI, CDS, or a device? Which regimes? What approvals?

## Production
Integration point, latency, availability, monitoring, ownership, fallback.

## Business case
Who pays? What is the value? What is the cost of a false positive/negative?

## Risks
Top five, with mitigations and owners.

## Stop criteria
What evidence would cause us to stop?
```

### 6.2 Worked example: sepsis early warning

| Charter element | Weak version | Strong version |
|---|---|---|
| Decision | "Predict sepsis" | "Trigger a nurse-led review within 30 minutes for patients on general wards with suspected infection" |
| User | "Doctors" | "Charge nurse on the ward, using the EHR banner" |
| Population | "All patients" | "Adults admitted to general wards, excluding ICU transfers and comfort-care-only patients" |
| Comparator | "No model" | "Current SIRS-based screen plus nurse judgment" |
| Outcome | "AUC" | "Time to antibiotics, mortality, alert burden per 1,000 patient-days" |
| Validation | Random split | Temporal holdout plus external site; calibration by site |
| Regulatory | "Not a device" | "Likely CDS; analyze FDA CDS criteria and state-law implications with counsel" |
| Production | "Deploy in EHR" | "Epic Best Practice Advisory, 15-minute refresh, silent mode for 4 weeks" |
| Business | "Saves lives" | "Reduced ICU transfers and bed-days; nurse time cost; alert fatigue risk" |

The strong version is not automatically correct. It is *specific enough to be wrong*, which is the precondition for learning.

## 7. Evidence hierarchy for industry claims

When you read a vendor claim or a paper, ask:

1. **Intended use:** What exactly is the product allowed to do, for whom, in what setting?
2. **Population:** Does the validation population match the deployment population? Consider age, sex, race/ethnicity, language, comorbidities, care setting, device vendor.
3. **Comparator:** Current practice or an artificially weak baseline?
4. **Endpoint:** Surrogate (AUC, Dice) or patient-relevant (mortality, time-to-treatment, cost)?
5. **Design:** Retrospective, prospective, randomized, silent-mode, pragmatic?
6. **Temporal validity:** Was the model validated on data after development? Did practice change?
7. **Calibration:** Are predicted probabilities meaningful at the decision threshold?
8. **Subgroups:** Are performance and harm reported for clinically relevant groups?
9. **Workflow:** Did the study measure use, alert burden, and downstream actions?
10. **Independence:** Who funded and conducted the evaluation? Was the analysis pre-specified?
11. **Reproducibility:** Is the model, code, or sufficient detail available?
12. **Regulatory status:** Cleared/approved, CE-marked, research-use-only, or marketing claim?

A high AUC with no calibration, no external validation, and no workflow evaluation is a research artifact, not an industry solution.

## 8. The 2026 regulatory triage map

This is an orientation, not legal advice. Module 14 goes deeper.

| Question | If yes | Regime / action |
|---|---|---|
| Does the output inform diagnosis, treatment, or prevention for a specific patient? | Possibly a medical device / SaMD | FDA 510(k), De Novo, or PMA; EU MDR/IVDR; QMS |
| Is it clinical decision support? | Check FDA CDS criteria and local law | May be excluded from device definition, but not automatically |
| Does it learn or change after deployment? | Predetermined change control plan may be needed | FDA PCCP; EU post-market performance follow-up |
| Is it used in a clinical trial? | Computerized system validation | 21 CFR Part 11, ICH E6 GCP |
| Does it process protected health information? | HIPAA Security/Privacy Rules (US) or GDPR (EU) | BAA, minimum necessary, de-identification, DPIA |
| Is it an AI system in the EU? | EU AI Act risk classification | Prohibited/high-risk/limited-risk obligations, timing |
| Is it a laboratory test? | CLIA/CAP (US), IVDR (EU) | Analytical and clinical validation |
| Is it sold to a hospital? | Procurement, security, interoperability | SOC 2, HITRUST, FHIR, accessibility, liability |
| Does it make claims about cost savings? | Payer/HTA evidence | Budget impact, cost-effectiveness, real-world evidence |

**Rule:** Regulatory triage happens in week one, not after the model is built.

## 9. Data ethics and professional conduct

### 9.1 Non-negotiables

- Do not attempt to re-identify de-identified data.
- Do not use protected health information in external tools without an approved data use agreement and security review.
- Do not present synthetic or simulated results as real-world evidence.
- Do not hide subgroup harm or negative results.
- Do not use patient data for marketing without consent and legal review.
- Do not deploy a model into clinical use without appropriate validation, approvals, and monitoring.
- Do not strip attribution from data or code licenses.

### 9.2 Neurodiversity and accessibility

The supplied book *The Power of Neurodiversity* is a reminder that "normal" is not a universal baseline. In product design:

- Offer multiple representations of information (text, speech, visual).
- Avoid forced time pressure when it is not clinically necessary.
- Support cognitive load and attention differences.
- Test with disabled and neurodivergent users, not only clinicians.
- Measure whether the tool widens or narrows existing disparities.

### 9.3 AI assistance policy

Use LLMs for drafting, code explanation, and critique. You remain accountable for:

- Every factual claim.
- Every line of code you submit.
- Data privacy and license compliance.
- The reproducibility of your analysis.
- The clinical and regulatory implications of your recommendations.

Disclose material AI assistance in project appendices. Never paste identifiable data into a public model.

## 10. How to use this package

| If you are... | Start here | Then |
|---|---|---|
| A data scientist new to health | 00, 01, 02 | 03–06, then your track |
| A clinician moving into analytics | 00, 02, 03 | 04–07, then 08 or 13 |
| An ML engineer moving into medtech | 00, 01, 14 | 10–11, 15–18 |
| A pharma scientist | 00, 01, 05 | 12–13, 18 |
| A manager / product lead | 00, 17, 18 | 14, 16, 19 |
| An instructor | README, 00, 19 | Module guides and rubrics |

Each module is designed to be usable as:

- A self-study guide with labs and sources.
- A university course unit.
- A corporate upskilling workshop.
- A reference playbook during a real project.

## 11. Cohort variants

### 11.1 12-week intensive

- Weeks 1–4: modules 00–04, compressed.
- Weeks 5–8: modules 05–08 plus one modality.
- Weeks 9–12: modules 14–18 plus a scoped capstone.
- Trade-off: less modality breadth, more execution pressure.

### 11.2 24-week part-time

- Weeks 1–12: modules 00–06.
- Weeks 13–20: modules 07–14, track electives.
- Weeks 21–24: modules 15–19, capstone sprint.
- Trade-off: limited time for deep omics or imaging labs.

### 11.3 Corporate cohort

- Pre-work: 00, 02, 03.
- Live workshops: domain-specific modules plus the cohort's real use case.
- Capstone: an internal project with executive sponsor, clinical champion, and regulatory reviewer.
- Success metric: a decision-ready artifact, not a certificate.

## 12. What this course is not

- It is not a medical degree. It teaches enough clinical vocabulary to collaborate safely, not to practice medicine.
- It is not a guarantee of regulatory clearance. It teaches the evidence and quality mindset regulators expect.
- It is not a tour of every model architecture. It prioritizes the models and methods that change decisions.
- It is not a substitute for local legal, privacy, and regulatory counsel.
- It is not a collection of copy-paste notebooks. Labs require judgment and documentation.

## 13. Success metrics for the program

| Metric | Target | How measured |
|---|---|---|
| Reproducibility | 90% of labs rerun from a clean environment | Automated check |
| Evidence quality | 80% of capstones include temporal/external validation | Rubric |
| Regulatory literacy | 80% pass a regulatory triage quiz | Quiz |
| Communication | 80% can explain a model to a clinician in 5 minutes | Oral assessment |
| Equity | 100% of capstones report at least one subgroup analysis | Rubric |
| Placement | Tracked at 6 and 12 months | Alumni survey |
| Industry review | ≥ 4/5 from external reviewers on capstone relevance | Review panel |

## 14. Portfolio artifacts you will leave with

1. A project charter for a real or realistic decision.
2. An assumptions ledger covering biology, measurement, statistics, causal claims, ML, workflow, devices, economics, and ethics.
3. A reproducible cohort-definition and data-quality report.
4. A risk-prediction model with calibration and decision-curve analysis.
5. A causal-inference protocol using target trial emulation.
6. A clinical NLP evaluation with grounding and hallucination analysis.
7. An imaging or signal model with site-held-out validation.
8. A regulatory triage memo and draft intended-use statement.
9. A model card, datasheet, and monitoring plan.
10. A health-economic or operational value case.
11. A capstone evidence dossier and 10-minute executive pitch.
12. A public-safe portfolio write-up with no proprietary data.
13. A 90-day post-course learning plan.
14. A device data and evidence plan covering sensors, chips, firmware, robotics, and safety.
15. An efficient-model dossier covering distillation, quantization, target-hardware latency/energy, and regulatory change control.

## 15. Key terms

| Term | Working definition |
|---|---|
| SaMD | Software as a Medical Device; software intended for a medical purpose without being part of a hardware device |
| CDS | Clinical decision support; software that helps clinicians make decisions |
| GMLP | Good Machine Learning Practice; FDA/MHRA/Health Canada guiding principles for medical-device ML |
| PCCP | Predetermined Change Control Plan; pre-authorized model changes |
| RWE | Real-world evidence derived from routine care or other non-trial data |
| FHIR | Fast Healthcare Interoperability Resources; modern HL7 data-exchange standard |
| OMOP CDM | Common Data Model for standardized observational health data |
| DICOM | Standard for medical imaging data and metadata |
| LOINC | Standard for laboratory and clinical observations |
| SNOMED CT | Clinical terminology for concepts, findings, and procedures |
| ICD-10/11 | International Classification of Diseases coding systems |
| CPT | Current Procedural Terminology; US procedure coding |
| EHR | Electronic health record |
| PHI | Protected health information |
| HIPAA | US health privacy and security law |
| GDPR | EU data protection regulation |
| MDR / IVDR | EU medical device / in-vitro diagnostic regulations |
| IEC 62304 | Medical device software lifecycle standard |
| ISO 14971 | Medical device risk management standard |
| ISO 13485 | Medical device quality management system standard |
| GxP | Good practice regulations for labs, clinical trials, manufacturing |
| CLIA | US Clinical Laboratory Improvement Amendments |
| HTI-1 | US health IT certification rule with AI transparency provisions |
| HTA | Health technology assessment |
| QALY | Quality-adjusted life year |
| Drift | Change in data or target relationship after deployment |
| Calibration | Agreement between predicted probabilities and observed rates |
| Decision curve | Net benefit across decision thresholds |
| Intended use | Regulated statement of what a product does, for whom, and in what setting |

## 16. Discussion and interview questions

1. **A hospital wants a readmission model. What is the first question you ask?**
   Hint: who acts on it, and what intervention follows? A prediction without an intervention is entertainment.
2. **Why is a random train/test split often misleading in clinical ML?**
   Hint: temporal drift, patient leakage, site leakage, repeated measures.
3. **What is the difference between a research model and a medical device?**
   Hint: intended use, claims, risk, quality system, lifecycle controls.
4. **When is a model "good enough" for a pilot?**
   Hint: pre-specified performance, calibration, safety, workflow, monitoring, and a stop rule.
5. **What does a payer care about that a clinician may not?**
   Hint: total cost of care, utilization, budget impact, coverage, member experience.
6. **Why can a model with excellent AUC be clinically useless?**
   Hint: poor calibration, low prevalence, no actionable threshold, alert fatigue, no outcome improvement.
7. **What is the minimum evidence package for an external hospital to trust your model?**
   Hint: intended use, external validation, calibration, subgroup performance, data provenance, monitoring, contract terms.
8. **How would you explain a 0.85 AUC to a hospital CEO?**
   Hint: avoid jargon; translate into detection rate, false alarms, workload, and expected outcomes.
9. **What is the most common cause of pilot purgatory?**
   Hint: workflow and ownership, not model performance.
10. **Name three ways a model can be biased without using race as a feature.**
    Hint: label bias, access bias, measurement bias, site effects, proxy variables.
11. **What would make you stop a project?**
    Hint: no decision owner, no usable data at decision time, unacceptable harm, unaffordable monitoring, regulatory path unavailable.
12. **How does an LLM change the regulatory question?**
    Hint: probabilistic output, change over time, grounding, hallucination, versioning, transparency.

## 17. Sources and further reading

### Regulation and policy
- FDA, *Artificial Intelligence and Machine Learning (AI/ML) Software as a Medical Device Action Plan*, 2021. https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- FDA, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- FDA, *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions*, final guidance, 2024; reissued 2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents
- FDA, *Clinical Decision Support Software*, final guidance, 2022. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- HHS, *HIPAA Privacy Rule*, https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
- ONC, *HTI-1 Final Rule*, 2024. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
- European Commission, *EU Artificial Intelligence Act*, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Commission, *Medical Device Regulation (EU) 2017/745* and *IVDR (EU) 2017/746*. https://eur-lex.europa.eu

### Industry and evidence
- Topol, E. J., "High-performance medicine: the convergence of human and artificial intelligence," *Nature Medicine*, 2019. https://doi.org/10.1038/s41591-018-0300-7
- Ghassemi, M., Oakden-Rayner, L., Beam, A. L., "The false hope of current approaches to explainable artificial intelligence in health care," *Lancet Digital Health*, 2021. https://doi.org/10.1016/S2589-7500(21)00208-9
- Wiens, J. et al., "Do no harm: a roadmap for responsible machine learning for health care," *Nature Medicine*, 2019. https://doi.org/10.1038/s41591-019-0548-6
- Singhal, K. et al., "Large language models encode clinical knowledge," *Nature*, 2023. https://doi.org/10.1038/s41586-023-06291-2
- Obermeyer, Z. et al., "Dissecting racial bias in an algorithm used to manage the health of populations," *Science*, 2019. https://doi.org/10.1126/science.aax2342

### Books
- Bickley, L. S., *Bates' Guide to Physical Examination and History Taking*, 13th ed.
- Goldberger, A. L. et al., *Goldberger's Clinical Electrocardiography: A Simplified Approach*, 9th ed.
- Thaler, M. S., *The Only EKG Book You'll Ever Need*, 9th ed.
- Herring, W., *Learning Radiology: Recognizing the Basics*, 5th ed.
- Lilly, L. S. et al., *Pathophysiology of Heart Disease*, 7th ed.
- West, J. B. et al., *West's Respiratory Physiology*, 11th ed.
- Blumenfeld, H., *Neuroanatomy through Clinical Cases*, 2nd ed.
- Jankovic, J. et al., *Bradley and Daroff's Neurology in Clinical Practice*, 8th ed.
- Armstrong, T., *The Power of Neurodiversity*.
- Schwinger, R. H. G., "Pathophysiology of heart failure," *Cardiovasc Diagn Ther*, 2021;11(1):263–276. https://doi.org/10.21037/cdt-20-302

### Tools
- Python, pandas, scikit-learn, PyTorch, SQL, Git, Docker, MLflow, Great Expectations, Evidently, Fairlearn, SHAP.
- FHIR: https://hl7.org/fhir/
- OMOP CDM: https://ohdsi.github.io/CommonDataModel/
- DICOM: https://www.dicomstandard.org/
- PhysioNet: https://physionet.org/
- MIMIC: https://mimic.mit.edu/
- Open Targets: https://www.opentargets.org/
- Hugging Face medical benchmarks: https://huggingface.co/

> **Next:** [Module 01](01_clinical_foundations.md) builds the clinical vocabulary and reasoning needed to ask better data questions.
