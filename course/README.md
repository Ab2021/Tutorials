# Biomedical Data Science for Industry — 2026 Edition

A comprehensive, industry-oriented curriculum for people who want to turn biomedical data into **safe, useful, regulated, and commercially viable products**.

This package is designed for self-study, university teaching, and corporate upskilling. It assumes competence in Python and basic statistics, but not in medicine, biology, regulation, or healthcare operations. It is deliberately broader and more operational than a typical machine-learning syllabus.

---

## 1. What this program is

A 40-week, three-term program covering:

- Clinical foundations and clinical reasoning.
- Units, measurement, uncertainty, and an explicit assumptions ledger.
- Medical hardware: sensors, analog front ends, semiconductors, firmware, electrical safety, and robotics.
- Efficient AI: distillation, quantization, pruning, runtimes, on-device inference, and target-hardware validation.
- Biomedical data sources, standards, and interoperability.
- Statistics, causal inference, and evidence generation.
- Machine learning, deep learning, multimodal models, LLMs, and agents.
- EHR, claims, operational, imaging, signal, wearable, and omics data.
- Biopharma R&D and clinical development.
- Medical devices, SaMD, and global regulation.
- MLOps, monitoring, cybersecurity, privacy, and governance.
- Product management, health economics, adoption, and leadership.
- A capstone with realistic industry constraints and hackathon-style sprints.

The organizing question is not "Which model should I use?" It is:

> **What decision are we improving, for whom, with what evidence, under what constraints — and how will we know after deployment that it still works?**

### 1.1 Coverage map

| Domain | Modules |
|---|---|
| Units, measurement, uncertainty, assumptions | 01A |
| Clinical foundations, physiology, ECG, imaging, neurology | 01, 01A |
| Data sources, standards, interoperability, governance | 02, 03 |
| Statistics, causal inference, study design, evidence | 04, 05 |
| Machine learning, deep learning, multimodal, LLMs, agents | 06, 07 |
| EHR, claims, operations, population health | 08 |
| Clinical NLP and documentation | 09 |
| Medical imaging and pathology | 10, 21 |
| Physiological signals, wearables, digital biomarkers | 11 |
| Omics, single-cell, spatial, precision medicine | 12 |
| Biopharma, trials, safety, RWE, digital twins | 13, 05 |
| Medical devices, SaMD, FDA/EU regulation, QMS | 14 |
| Sensors, AFEs, semiconductors, firmware, electrical safety, robotics | 14A |
| Distillation, quantization, pruning, edge inference | 14B |
| Deployment, MLOps, monitoring, incident response | 15 |
| Privacy, security, fairness, ethics | 16 |
| Product, health economics, adoption, leadership | 17 |
| Quality, reporting, evidence dossiers | 18 |
| Capstone, hackathons, assessments | 19 |
| Career, portfolio, interviews | 20 |
| Datasets, benchmarks, tools | 21 |
| Sources and further reading | 22 |

## 2. Who it is for

| Audience | Best entry point | Likely outcome |
|---|---|---|
| Data scientists entering health | Modules 00–06 incl. 01A | Clinical data scientist / health AI scientist |
| ML engineers entering medtech | Modules 00, 01, 01A, 14, 14A, 14B, 15, 16 | ML engineer, SaMD, or device data scientist |
| Clinicians entering informatics | Modules 00, 01A, 02–06 | Clinical informatics / research data scientist |
| Biologists entering pharma data science | Modules 00, 01A, 04–06, 12–13 | Translational or biomarker data scientist |
| Product managers and consultants | Modules 00, 01A, 14, 14A, 14B, 17–19 | Health AI product lead / strategy |
| Researchers | Modules 01A, 04–07, 18 | Translational AI researcher with regulatory literacy |
| Instructors | README, Module 00, Module 19 | Ready-to-adapt course design |

## 3. Program at a glance

| Item | Detail |
|---|---|
| Duration | 40 weeks; 17–19 hours/week (full-time intensive; part-time variant 48–52 weeks) |
| Format | Lecture + lab + case clinic + project studio |
| Total guided time | 690–730 core hours; 14B adds 28; advanced electives add up to 98 more if all are taken; compressible or extendable |
| Tracks | Health Systems; Biopharma & Translational; MedTech, Sensors & Imaging |
| Capstone | 10–12 weeks, team-based, externally reviewed |
| Prerequisites | Python, SQL basics, probability/statistics, linear algebra fundamentals |
| Clinical prerequisite | None; Module 01 provides the minimum viable vocabulary and Module 01A supplies units, measurement, and assumptions |
| Assessment | Labs, case critiques, evidence dossier, track playbook, capstone defense |
| Source cutoff | 24 October 2026 |
| Language | English |
| License | Course text: CC BY-NC-SA 4.0 unless otherwise noted; code samples: MIT |
| Package size | 30 modules (00–22 plus 01A, 14A, 14B, and advanced electives 23–26), 2 research source packs, 20,000+ lines |

### 3.1 2026 industry signals that shape the curriculum

These are teaching anchors, not forecasts. Verify before citing; details and URLs are in `_research/industry_brief.md`.

| Signal | Why it is in the course |
|---|---|
| Provider margins remain thin and denial rates are rising | Revenue-cycle, capacity, and operations analytics are hard-dollar use cases (Module 08, 17) |
| Ambient documentation is the most adopted clinical AI, but scaling and training lag | Workflow, human factors, and adoption matter as much as model quality (Module 09, 15, 17) |
| Pharma has 75+ AI-designed molecules in trials but no AI-discovered approval | Target validation and clinical translation remain the bottleneck (Module 12, 13) |
| FDA has authorized 1,600+ AI-enabled devices, mostly imaging | Regulatory literacy and imaging evidence are core, not optional (Module 10, 14) |
| AI device recalls cluster early and are often software-related | Monitoring and post-market surveillance are essential (Module 15, 18) |
| Consumer PPG claims are narrow and performance varies by skin tone and motion | Signal validation and equity are product requirements (Module 11, 16) |
| Payer APIs, HTI-1 transparency, and prior-authorization rules are changing | Interoperability and documentation are regulatory capabilities (Module 02, 03, 14) |
| EU AI Act high-risk obligations for medical devices are phased to 2028 | Plan dual MDR/IVDR + AI Act compliance now (Module 14) |
| LLM benchmarks and agent evaluations are improving but remain fragile | Grounding, red-teaming, and human review are mandatory (Module 07, 09) |
| Devices now combine sensors, edge AI, and robotics; firmware and cyber failures are safety events | Hardware, electrical safety, and HIL validation are core MedTech skills (Module 14A) |
| Employers reward evidence, deployment, and domain judgment over generic model-building | The portfolio and capstone emphasize decision-ready artifacts (Module 19, 20) |

## 4. Learning outcomes

Graduates can:

1. **Frame** a biomedical problem as a decision with a user, workflow, outcome, and economic buyer.
2. **Acquire and standardize** EHR, claims, imaging, signal, wearable, omics, and trial data while respecting privacy and licensing.
3. **Design** observational and experimental evidence plans, including target trial emulation, external validation, and sensitivity analysis.
4. **Build** transparent, calibrated, reproducible models appropriate to the data and decision.
5. **Evaluate** models for discrimination, calibration, clinical utility, subgroup performance, robustness, and workflow impact.
6. **Interpret** clinical NLP and LLM outputs with grounding, hallucination, and safety controls.
7. **Navigate** FDA, EU MDR/IVDR, the EU AI Act, HIPAA/GDPR, GxP, CLIA, and health-IT certification at a working level.
8. **Engineer** production pipelines with versioning, monitoring, drift detection, incident response, and change control.
9. **Quantify** value using cost-effectiveness, budget impact, operational KPIs, and adoption analysis.
10. **Communicate** to clinicians, engineers, executives, regulators, and patients without losing truth.
11. **Govern** data and models with attention to consent, equity, accessibility, neurodiversity, and security.
12. **Deliver** a decision-ready capstone under realistic time, data, and regulatory constraints.
13. **Explain** how sensors, analog front ends, chips, firmware, and robots shape data quality, latency, safety, and regulation.
14. **State, document, and test** the biological, statistical, causal, device, and economic assumptions behind a claim.
15. **Compress and deploy** models under explicit accuracy, calibration, latency, energy, memory, and safety budgets.

## 5. Curriculum map

### Term 1 — Foundations (Weeks 1–12)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 00 | Orientation and the 2026 industry map | 16 | Project charter |
| 01 | Clinical foundations for data scientists | 32 | Clinical vocabulary and data-plausibility memo |
| 01A | Foundations, units, measurement, and assumptions | 20 | Assumptions ledger |
| 02 | The biomedical data landscape | 20 | Data-source map |
| 03 | Data engineering and interoperability | 28 | Reproducible cohort pipeline |
| 04 | Statistics and causal inference | 32 | Causal protocol |
| 05 | Study design and evidence generation | 24 | Evidence plan |
| 06 | Machine learning foundations | 32 | Calibrated risk model |

### Term 2 — Modalities, hardware, and efficient AI (Weeks 13–26)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 07 | Deep learning, multimodal models, LLMs, and agents | 32 | Multimodal or LLM evaluation |
| 08 | EHR, claims, and operations analytics | 28 | Operational decision-support analysis |
| 09 | Clinical NLP | 24 | Grounded extraction evaluation |
| 10 | Medical imaging | 32 | Imaging model with site-held-out validation |
| 11 | Physiological signals and wearables | 32 | Signal pipeline and digital biomarker |
| 12 | Omics and precision medicine | 32 | Biomarker discovery report |
| 13 | Biopharma and clinical development | 28 | Trial or RWE strategy |
| 14 | Medical devices, SaMD, and regulation | 32 | Regulatory evidence package |
| 14A | Medical hardware, sensors, semiconductors, and robotics | 36 | Device data and evidence plan |
| 14B | Efficient AI: distillation, quantization, and on-device inference (advanced elective) | 28 | Efficient-model dossier |

### Term 3 — Production, product, and capstone (Weeks 27–40)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 15 | Deployment and MLOps | 28 | Deployment and monitoring plan |
| 16 | Privacy, security, and ethics | 24 | Threat model and governance review |
| 17 | Product, economics, and leadership | 24 | Value case and adoption plan |
| 18 | Quality, evaluation, and reporting | 24 | Model card and evidence dossier |
| 19 | Capstone and hackathons | 80–120 | Capstone and demo |
| 20 | Career, portfolio, and interviews | 12 | Portfolio and interview plan |
| 21 | Datasets, tools, and benchmarks | 16 | Personal resource library |
| 22 | Sources and further reading | self-paced | Annotated bibliography |

### Advanced electives (post-core; optional)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 23 | Specialty electives: oncology, cardiology, pathology, critical care, pediatrics/OB, public health | 18–24 | Specialty primer and lab |
| 24 | Hardware, firmware, EMC, HIL, and robotics labs | 28 | Bench/robotics evidence package |
| 25 | Trial operations, HTA/market access, and global regulation | 28 | Trial/HTA/regulatory roadmap |
| 26 | Cloud, Kubernetes, and streaming production labs | 24 | Production deployment and DR report |

These are optional depth modules. Taking all four extends the program beyond 40 weeks; taking one or two can replace a track elective or follow the core.

## 6. Tracks

### 6.1 Health Systems Analytics

Focus: operational and clinical decisions inside provider organizations.

Recommended emphasis:

- Module 08: cohorting, risk stratification, readmission, length of stay, capacity.
- Module 09: clinical documentation, coding, and chart review.
- Module 17: value to the health system, payer contracts, and adoption.
- Capstone examples: sepsis alert, avoidable ED use, OR scheduling, discharge planning, documentation burden.

### 6.2 Biopharma and Translational Data Science

Focus: discovery, development, and evidence across the drug lifecycle.

Recommended emphasis:

- Module 12: genomics, single-cell, spatial, proteomics, target discovery.
- Module 13: trial design, biomarkers, pharmacovigilance, RWE.
- Module 05: estimands, external controls, and regulatory evidence.
- Capstone examples: biomarker signature, trial recruitment, safety signal detection, external control arm.

### 6.3 MedTech, Sensors, and Imaging

Focus: regulated products and algorithms embedded in devices or care pathways.

Recommended emphasis:

- Module 10: imaging modalities, segmentation, detection, foundation models.
- Module 11: ECG, PPG, EEG, respiratory signals, wearables.
- Module 14: design controls, risk management, FDA/CE, PCCP, post-market surveillance.
- Module 14A: sensors, analog front ends, semiconductors, firmware, electrical safety, and robotics.
- Module 14B: distillation, quantization, pruning, runtimes, and target-hardware inference.
- Optional depth: Module 23 (specialty electives), Module 24 (hardware/robotics labs), Module 26 (cloud/streaming).
- Capstone examples: arrhythmia detection, imaging triage, drift monitoring, SaMD evidence package, sensor-chain validation.

## 7. The capstone

The capstone is the center of gravity. It is not a Kaggle submission. It is an **evidence-backed product proposal** with a working prototype.

Required components:

1. Project charter and intended-use statement.
2. Data provenance, access, and quality report.
3. Reproducible analysis with versioned environment.
4. Model or analytic method with baseline comparison.
5. Temporal and external validation where possible.
6. Calibration, decision-curve, and subgroup analysis.
7. Error analysis and failure-mode register.
8. Regulatory and ethical triage memo.
9. Model card and datasheet.
10. Deployment architecture and monitoring plan.
11. Health-economic or operational value case.
12. 10-minute executive pitch and 20-minute technical defense.

See `19_capstone_hackathons.md` for briefs, milestones, rubrics, and hackathon formats.

## 8. Assessment philosophy

Assessments reward **judgment**, not just accuracy:

- Can you define the right cohort?
- Can you defend the split?
- Can you explain why a false negative costs more than a false positive, or vice versa?
- Can you identify the regulatory path?
- Can you tell a clinician what to do differently?
- Can you state what would make you withdraw the model?

Every rubric includes a "competent" level that is achievable without perfection, and an "expert" level that requires evidence of independent critical thought.

## 9. Tools and environment

### Core

- Python 3.11+, pandas, NumPy, scikit-learn, statsmodels, PyTorch.
- SQL (PostgreSQL or DuckDB), Git, Docker.
- For hardware/robotics: LTspice or Falstad, ROS 2, Gazebo or MuJoCo, Pinocchio, TensorFlow Lite Micro or ONNX Runtime, pydicom, WFDB.
- Jupyter or VS Code, a reference manager, and a diagramming tool.

### Data and interoperability

- FHIR, OMOP CDM, DICOM, LOINC, SNOMED CT, ICD-10/11, RxNorm.
- Synthea for synthetic EHR data; MIMIC-IV and eICU for credentialed access.
- PhysioNet, Open Neuro, TCGA, GEO, Open Targets, ChEMBL.

### MLOps and governance

- MLflow or Weights & Biases, DVC, Great Expectations, Evidently.
- Fairlearn, SHAP, Captum, OpenAI Evals or equivalent.
- Cloud: AWS, Azure, or GCP; preferably one with health-data services.

### Safety

- No identifiable data in public LLMs.
- No production deployment without approvals.
- No use of clinical data outside the terms of its data use agreement.

## 10. Suggested weekly cadence

| Day | Activity | Output |
|---|---|---|
| Monday | Lecture and reading | Concept map and three questions |
| Tuesday | Lab | Reproducible notebook or script |
| Wednesday | Case clinic | One-page critique |
| Thursday | Project studio | Increment and risk log |
| Friday | Office hours / review | Feedback and next-week plan |

## 11. Package contents

| File | Purpose |
|---|---|
| `README.md` | This guide |
| `COURSE_OUTLINE.md` | Master table of contents and module-by-module topic index |
| `CHANGELOG.md` | Version history and known limitations |
| `VALIDATION_STATUS.md` | What is complete, drafted, or pending; claims boundary |
| `COMPETENCY_MAP.md` | Outcome-to-module/lab/assessment crosswalk |
| `GOVERNANCE_AND_EXTERNAL_REVIEW.md` | Reviewer terms, rubric, COI, sign-off |
| `REGULATORY_VERIFICATION_LOG.md` | Tiered claim register and review status |
| `INSTRUCTOR_KIT.md` | Teaching notes, anchor answer outlines, lab index |
| `GRADING_CALIBRATION.md` | Grader training and anchor descriptors |
| `PILOT_PROTOCOL.md` | 5–10 learner feasibility pilot |
| `labs/` | Starter notebooks, cloud Terraform/K8s/Kafka stack, MLOps service, and sample solutions |
| `review/` | Private review workspace template |
| `00_orientation_and_industry_map.md` | Program orientation, industry map, charter template |
| `01_clinical_foundations.md` | Clinical vocabulary, physiology, exam, ECG, imaging, neuro |
| `01A_foundations_assumptions_and_units.md` | Math, biology, units, measurement, and assumptions ledger |
| `02_biomedical_data_landscape.md` | Data sources, standards, access, quality |
| `03_data_engineering_interoperability.md` | Pipelines, FHIR, OMOP, DICOM, de-identification |
| `04_statistics_causal_inference.md` | Inference, regression, survival, causal methods |
| `05_study_design_evidence.md` | Trials, RWE, estimands, validation |
| `06_machine_learning_foundations.md` | Supervised learning, calibration, leakage, evaluation |
| `07_deep_learning_multimodal_llms.md` | Neural networks, transformers, multimodal, LLM evaluation |
| `08_ehr_claims_operations.md` | Cohorting, risk models, operations, claims |
| `09_clinical_nlp.md` | Extraction, coding, summarization, grounding |
| `10_medical_imaging.md` | DICOM, segmentation, detection, imaging foundation models |
| `11_physiological_signals_wearables.md` | ECG, PPG, EEG, sleep, digital biomarkers |
| `12_omics_precision_medicine.md` | Genomics, single-cell, spatial, multi-omics |
| `13_biopharma_clinical_development.md` | Target discovery, trials, pharmacovigilance, RWE |
| `14_medical_devices_samd_regulation.md` | Design controls, FDA/CE, risk, PCCP, post-market |
| `14A_medical_hardware_sensors_chips_robotics.md` | Sensors, AFEs, chips, firmware, electrical safety, robotics |
| `14B_efficient_ai_distillation_quantization_on_device_inference.md` | Distillation, quantization, pruning, runtimes, target-hardware validation |
| `15_deployment_mlops.md` | Serving, monitoring, drift, incident response |
| `16_privacy_security_ethics.md` | Threat modeling, privacy, fairness, governance |
| `17_product_economics_leadership.md` | Value, reimbursement, adoption, leadership |
| `18_quality_evaluation_reporting.md` | Reporting standards, model cards, audit |
| `19_capstone_hackathons.md` | Briefs, milestones, rubrics, hackathon playbook |
| `20_career_portfolio_interviews.md` | Portfolio, interviews, negotiation |
| `21_datasets_tools_benchmarks.md` | Curated datasets and tooling |
| `22_sources_and_further_reading.md` | Annotated sources |
| `23_specialty_electives.md` | Oncology, cardiology, pathology, critical care, pediatrics/OB, public health |
| `24_hardware_robotics_labs.md` | Signal chain, firmware, EMC, HIL, ROS 2, robot kinematics |
| `25_trial_operations_hta_global_regulatory.md` | Trial operations, HTA, market access, global regulation |
| `26_cloud_streaming_production_labs.md` | Kubernetes, Terraform, streaming, observability, DR |
| `modules/` | Expanded edition: one folder per module, 300 planned topic files of 1,200–1,800 lines each |
| `MODULE_EXPANSION_SPEC.md` | Authoring contract for topic files: template, length floor, quality bar |
| `tools/scaffold_modules.py` | Regenerates module indexes and `modules/PROGRESS.md` |
| `_course_spec.md` | Production specification for instructors |
| `_research/industry_brief.md` | 2025–2026 industry, workforce, regulatory, and technical source pack |
| `_research/hackathons_brief.md` | 2023–2026 competitions, hackathons, benchmarks, and project briefs |
| `_research/link_verification_summary.md` | Link-check method, results, replacements, and residual manual checks |

## 12. How to use the package

### Self-study

1. Read Module 00 and complete the charter.
2. Work through Modules 01 → 01A → 02–06 in order.
3. Choose a track and complete the required modules.
4. Do at least one full capstone brief.
5. Assemble the portfolio artifacts listed in Module 00.

### Teaching

1. Read `_course_spec.md`.
2. Map modules to your academic calendar.
3. Replace datasets with local, ethically approved data where possible.
4. Invite clinical, regulatory, and product guest reviewers.
5. Use the capstone rubric as the final assessment.

### Corporate

1. Run Module 00 as an executive alignment session.
2. Select modules based on the business unit.
3. Use a real internal project as the capstone.
4. Require a regulatory and security review at week 4.
5. Measure success by a decision-ready artifact, not attendance.

## 13. Prerequisites and bridge material

If you are missing background:

| Gap | Bridge |
|---|---|
| Python | Complete a data-analysis course; build 3 pandas projects |
| SQL | Practice joins, window functions, CTEs on a public dataset |
| Statistics | Review probability, sampling, hypothesis tests, regression |
| Linear algebra | Vectors, matrices, eigenvalues, gradients |
| Biology | Module 01 plus a cell-biology primer |
| Clinical workflow | Shadow a clinic or interview 3 clinicians |
| Cloud | Deploy one containerized API |
| Git | Contribute to an open-source repository |

## 14. Data ethics charter

Participants agree to:

- Use data only for approved purposes.
- Minimize data collection and retention.
- Protect confidentiality and security.
- Report limitations and harms.
- Avoid stigmatizing language.
- Include affected communities where feasible.
- Respect intellectual property and data licenses.
- Escalate concerns about safety or legality.

## 15. Versioning and maintenance

- **Edition:** 2026.2.1
- **Source cutoff:** 24 October 2026.
- **Review cadence:** every 6 months for regulatory changes; annually for the full curriculum.
- **Change log:** record substantive changes in `CHANGELOG.md`.
- **Deprecation:** mark outdated modules and provide a replacement.

### 15.1 Research and verification method

The 2026 edition was built from:

1. The ten supplied clinical books, used for vocabulary, physiology, and plausibility rather than data-science methods.
2. Two structured web-research passes: one on the 2025–2026 industry, workforce, regulatory, and technical landscape; one on 2023–2026 hackathons, competitions, benchmarks, and project briefs.
3. Primary regulatory and standards sources: FDA, HHS, ONC, CMS, EUR-Lex, ICH, ISO/IEC, HL7, OHDSI, DICOM, and CDISC.
4. Peer-reviewed literature and official dataset/benchmark pages.
5. Explicit uncertainty markers for time-sensitive or single-source claims.

Research notes are preserved in `_research/industry_brief.md` and `_research/hackathons_brief.md`. Facts were checked against organizer pages, regulatory pages, or primary papers where possible. Market sizes, salaries, clearance counts, and event statuses are dated snapshots, not forecasts. The package labels announced-only, closed, concluded, and unverified items rather than presenting them as settled. The final audit checked more than 500 external URLs and 63 DOIs; residual bot-blocked or paywalled sources are documented in `_research/link_verification_summary.md`.

### 15.2 Validation status

| P0 item | Status |
|---|---|
| Competency map | Drafted (`COMPETENCY_MAP.md`); external validation pending |
| External reviewers | Not yet convened (`GOVERNANCE_AND_EXTERNAL_REVIEW.md`) |
| Regulatory claims professionally verified | Pending (`REGULATORY_VERIFICATION_LOG.md`) |
| Instructor kit, starter notebooks, grading calibration | Drafted (`INSTRUCTOR_KIT.md`, `GRADING_CALIBRATION.md`, `labs/`) |
| Learner pilot | Protocol only (`PILOT_PROTOCOL.md`) |

The course is **not** clinically validated, regulatory-approved, accredited, or pilot-tested. See `VALIDATION_STATUS.md` for the claims boundary and exit criteria.

## 16. Known limitations

- Regulation varies by jurisdiction and changes frequently. Verify before acting.
- Some datasets require institutional agreements and cannot be redistributed.
- Clinical content is educational and not medical advice.
- Vendor and market claims are dated and should be rechecked.
- The course cannot substitute for supervised clinical, regulatory, or legal expertise.

## 17. Quick-start checklist

- [ ] Read Module 00.
- [ ] Complete the project charter template.
- [ ] Set up Python, Git, Docker, and a database.
- [ ] Register for PhysioNet and review credentialed-data requirements.
- [ ] Choose a track.
- [ ] Join or form a capstone team.
- [ ] Schedule a clinical interview.
- [ ] Create a learning log.
- [ ] Identify one real decision you want to improve.

### 17.1 Quality checks performed

- [x] All 30 module files present.
- [x] README, outline, and footer next-module links resolve; body cross-references remain plain text.
- [x] Code fences balanced.
- [x] Required module sections checked.
- [x] Source packs preserved in `_research/`.
- [x] Regulatory dates reviewed; uncertain K/recall identifiers marked `[VERIFY]` in source notes.
- [x] Competition statuses labeled concluded, closed, announced, or unverified.
- [x] Uncertain dataset sizes and licenses marked "verify".
- [x] No placeholder text or fabricated winners.
- [x] Line count exceeds the 10,000-line target.
- [x] Track and capstone mappings cross-checked.
- [x] Known limitations documented.

## 18. Contact and contribution

This is an educational package. For institutional use, adapt the assessment and data-governance sections to local policy. Contributions should include sources, a rationale, and a test for reproducibility.

> **Start here:** [`00_orientation_and_industry_map.md`](00_orientation_and_industry_map.md)
