# Competency Map — Biomedical Data Science for Industry (2026)

> **Purpose:** Map every program learning outcome to modules, labs, assessments, and evidence. This is a course crosswalk, not a professional certification or an accreditation claim.
>
> **Version:** 2026.2.1
> **Source cutoff:** 24 October 2026
> **Status:** Desk-built; external review pending (see `GOVERNANCE_AND_EXTERNAL_REVIEW.md`).

---

## 1. Framework alignment

The map is aligned with, but not certified against:

| Framework | Use in this course |
|---|---|
| AMIA Clinical Informatics Core Competencies | Clinical workflow, decision support, data governance, leadership |
| ACM/IEEE computing and data-science curricula | Programming, data management, modeling, ethics |
| FDA Good Machine Learning Practice (GMLP) | Representative data, independent test sets, clinical relevance, monitoring |
| ISO 13485 / IEC 62304 / ISO 14971 | Design controls, software lifecycle, risk management |
| ICH E6 GCP and ICH E9(R1) | Trial conduct, estimands, evidence |
| WHO ethics and governance guidance for AI in health | Equity, transparency, accountability |
| NIST AI Risk Management Framework | Govern, map, measure, manage |
| Health technology assessment (HTA) good practice | Cost-effectiveness, budget impact, uncertainty |

**Important:** alignment means the course teaches concepts and artifacts used in these frameworks. It does not make a learner certified, accredited, or legally qualified.

## 2. Program learning outcomes

By graduation, a learner can:

| ID | Outcome |
|---|---|
| PLO-1 | Frame a biomedical problem as a decision with a user, workflow, outcome, comparator, and economic buyer. |
| PLO-2 | Acquire, standardize, link, and govern EHR, claims, imaging, signal, wearable, omics, and trial data. |
| PLO-3 | Design observational and experimental evidence plans, including estimands, target trial emulation, external validation, and sensitivity analysis. |
| PLO-4 | Build transparent, calibrated, reproducible models appropriate to the data and decision. |
| PLO-5 | Evaluate discrimination, calibration, clinical utility, subgroup performance, robustness, and workflow impact. |
| PLO-6 | Interpret clinical NLP and LLM outputs with grounding, hallucination, and safety controls. |
| PLO-7 | Navigate FDA, EU MDR/IVDR, the EU AI Act, HIPAA/GDPR, GxP, CLIA, and health-IT certification at a working level. |
| PLO-8 | Engineer production pipelines with versioning, monitoring, drift detection, incident response, and change control. |
| PLO-9 | Quantify value using cost-effectiveness, budget impact, operational KPIs, and adoption analysis. |
| PLO-10 | Communicate to clinicians, engineers, executives, regulators, and patients without losing truth. |
| PLO-11 | Govern data and models with attention to consent, equity, accessibility, neurodiversity, and security. |
| PLO-12 | Deliver a decision-ready capstone under realistic time, data, and regulatory constraints. |
| PLO-13 | Explain how sensors, analog front ends, chips, firmware, and robots shape data quality, latency, safety, and regulation. |
| PLO-14 | State, document, and test the biological, statistical, causal, device, and economic assumptions behind a claim. |
| PLO-15 | Compress and deploy models under explicit accuracy, calibration, latency, energy, memory, and safety budgets. |

## 3. Competency domains

| Domain | Competencies | Primary modules |
|---|---|---|
| C1. Clinical and biological foundations | Clinical vocabulary, physiology, measurement, disease mechanisms, special populations | 01, 01A, 23 |
| C2. Data and interoperability | Data sources, standards, linkage, quality, lineage, de-identification | 02, 03, 21 |
| C3. Statistics and causal inference | Estimands, regression, survival, missing data, DAGs, target trial emulation | 01A, 04, 05 |
| C4. Evidence and study design | Trials, RWE, validation, bias, reporting | 05, 13, 18 |
| C5. Machine learning and AI | Model development, calibration, leakage, deep learning, LLMs, agents | 06, 07, 09 |
| C6. Domain analytics | EHR/claims/operations, imaging, signals, omics, pharma | 08, 10, 11, 12, 13, 23 |
| C7. Devices, hardware, and robotics | Sensors, AFEs, chips, firmware, electrical safety, kinematics, control | 14, 14A, 24 |
| C8. Efficient AI and edge inference | Distillation, quantization, pruning, runtimes, target validation | 14B |
| C9. Production and MLOps | Deployment, monitoring, drift, incident response, rollback | 15, 26 |
| C10. Privacy, security, and ethics | Threat modeling, privacy, fairness, consent, accessibility | 16 |
| C11. Product, economics, and leadership | Value, reimbursement, adoption, procurement, communication | 17, 25 |
| C12. Quality and regulation | QMS, design controls, PCCP, reporting, audit | 14, 18, 25 |
| C13. Translation and capstone | Scoping, evidence dossier, demo, defense | 19, 20 |

## 4. Outcome-to-module map

| PLO | Modules | Labs / artifacts | Assessment |
|---|---|---|---|
| PLO-1 | 00, 01A, 05, 17, 19 | Charter; stakeholder map; value case | Midterm dossier; capstone |
| PLO-2 | 02, 03, 08, 10, 11, 12, 21 | FHIR/OMOP/DICOM labs; cohort pipeline; data audit | Midterm dossier; track playbook |
| PLO-3 | 01A, 04, 05, 13, 18 | DAG; target trial; RWE protocol; appraisal | Midterm dossier; case critiques |
| PLO-4 | 06, 07, 14B | Calibration; pipeline; compression | Midterm dossier; capstone |
| PLO-5 | 05, 06, 10, 11, 18 | Calibration, decision curve, subgroup audit | Midterm dossier; capstone |
| PLO-6 | 07, 09, 14B | RAG; grounding; red team; LLM quantization | Track playbook; capstone |
| PLO-7 | 14, 14A, 16, 18, 25 | Regulatory triage; risk file; verification log | Regulatory memo; capstone |
| PLO-8 | 03, 15, 26 | CI/CD; monitoring; incident; streaming | Deployment plan; capstone |
| PLO-9 | 17, 25 | ROI; budget impact; HTA | Track playbook; value case |
| PLO-10 | 00, 17, 19, 20 | Pitch; critique; defense | Capstone defense; case critiques |
| PLO-11 | 01, 16, 23 | Fairness audit; de-identification; ethics assessment | Capstone; case critiques |
| PLO-12 | 19, 20 | Capstone dossier | Capstone |
| PLO-13 | 01A, 11, 14A, 24 | Signal chain; HIL; robot kinematics | Device plan; capstone |
| PLO-14 | 01A, 04, 05, 06, 14A | Assumptions ledger; sensitivity analysis | Midterm dossier; capstone |
| PLO-15 | 14B | Profiling; PTQ/QAT; distillation; HIL; Pareto | Efficient-model dossier; capstone |

## 5. Module competency matrix

Legend: **P** primary, **S** supporting, blank not a focus.

| Module | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | S | | | S | | S | | | | S | S | S | P |
| 01 | P | S | | | | S | | | | S | | | |
| 01A | P | S | P | S | S | | S | | | S | S | | |
| 02 | S | P | | | | S | | | | S | | S | |
| 03 | | P | | | | S | | | P | S | | S | |
| 04 | | | P | P | S | | | | | | | | |
| 05 | S | | P | P | | | | | | S | S | P | |
| 06 | | S | S | S | P | | | | | S | | | |
| 07 | | | | | P | S | | S | S | S | | | |
| 08 | S | P | S | S | S | P | | | S | S | S | | |
| 09 | S | S | | | P | S | | | | S | | | |
| 10 | S | S | | | P | P | S | | | S | | S | |
| 11 | S | S | | | P | P | S | S | | S | | S | |
| 12 | S | S | S | S | S | P | | | | S | | S | |
| 13 | S | S | S | P | S | P | | | | S | S | P | |
| 14 | | | | S | | | S | | | S | S | P | |
| 14A | S | | | | | | P | S | S | S | | P | |
| 14B | | | | | S | | P | P | S | S | S | P | |
| 15 | | S | | | | | S | S | P | S | S | P | |
| 16 | | S | | | | | | | | P | S | S | |
| 17 | | | | | | | | | | S | P | S | S |
| 18 | | | S | P | S | | | | S | S | | P | |
| 19 | | S | S | S | S | S | S | S | S | S | S | S | P |
| 20 | | | | | | | | | | | S | | P |
| 21 | | P | | S | S | S | S | S | S | S | | S | |
| 22 | S | S | S | S | S | S | S | S | S | S | S | S | S |
| 23 | P | S | S | S | S | P | | | | S | S | S | |
| 24 | S | | | | | | P | P | S | S | | P | |
| 25 | | | S | P | | S | | | | S | P | P | |
| 26 | | S | | | | | | S | P | S | S | P | |


## 6. Assessment blueprint

| Assessment | Weight | PLOs measured | Evidence |
|---|---:|---|---|
| Weekly labs | 20% | 2, 4, 5, 8, 13, 14, 15 | Runnable artifacts and acceptance criteria |
| Case critiques | 15% | 1, 3, 5, 7, 11 | 1-page structured critiques |
| Midterm evidence dossier | 20% | 1, 2, 3, 4, 5, 7, 10, 14 | 2,500-word dossier plus appendix |
| Track playbook | 15% | 1, 6, 7, 9, 10, 11 | 12–15 page playbook plus brief |
| Capstone | 30% | All | Dossier, prototype, demo, defense |

**Pass rule:** at least 50% on the individual capstone defense, regardless of team score.

## 7. Lab coverage map

| Lab family | Modules | Competency |
|---|---|---|
| Data audit, FHIR/OMOP, cohort pipeline | 02, 03, 21 | C2 |
| DAG, target trial, survival, missing data | 04, 05 | C3, C4 |
| ML pipeline, calibration, fairness, leakage | 06 | C5 |
| LLM/RAG/red team | 07, 09 | C5 |
| Imaging and signal pipelines | 10, 11 | C6, C7 |
| Omics and single-cell | 12 | C6 |
| Trial, safety, RWE | 13 | C4, C6 |
| Regulatory, risk, QMS | 14, 18 | C12 |
| Sensor chain, HIL, robot kinematics | 14A, 24 | C7 |
| Distillation, quantization, target inference | 14B | C8 |
| MLOps, drift, incident, cloud | 15, 26 | C9 |
| Privacy, security, fairness | 16 | C10 |
| Product, HTA, adoption | 17, 25 | C11 |
| Capstone | 19 | C13 |

## 8. Evidence standard for “competent”

A learner is competent when they can:

- State the decision and comparator.
- Define the cohort, label, time zero, and outcome.
- Choose a validation design that matches the claim.
- Report calibration and a clinically meaningful threshold.
- Identify the regulatory pathway and the evidence gap.
- Explain the main assumptions and one sensitivity analysis.
- Communicate the result to a clinician and an executive.
- Reproduce the analysis from a clean environment.

“Expert” requires independent critique, external validation or a credible plan, and a decision-ready recommendation.

## 9. External review and maintenance

- The map is desk-built. External clinician, regulatory, and hardware reviewers must sign off before the course is described as validated.
- Review cycle: every 6 months for regulatory changes; annually for the full map.
- Every new module must add: PLO mapping, lab, assessment, and source list.
- Deprecated modules remain in the change log with a replacement path.

## 10. Sources for framework alignment

- AMIA, *Clinical Informatics Core Competencies*. https://amia.org/education-career/clinical-informatics-core-competencies
- ACM/IEEE, *Computing Competencies for Undergraduate Data Science Curricula*. https://dl.acm.org/doi/10.1145/3467968
- FDA, *Good Machine Learning Practice*. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- ISO 13485. https://www.iso.org/standard/59752.html
- IEC 62304. https://www.iec.ch/
- ISO 14971. https://www.iso.org/standard/72704.html
- ICH, *E6 GCP* and *E9(R1)*. https://www.ich.org/
- WHO, *Ethics and governance of artificial intelligence for health*. https://www.who.int/publications/i/item/9789240029200
- NIST, *AI Risk Management Framework*. https://www.nist.gov/itl/ai-risk-management-framework
- EURETTA/HTA good practice. https://www.eureta.org/

> **Status:** The crosswalk is complete as a design artifact. It is not evidence that external reviewers have validated the course.
