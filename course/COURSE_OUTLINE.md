# Course Outline — Biomedical Data Science for Industry (2026)

> **Master table of contents.** Use this file to navigate the 30 modules. Each module contains industry context, learning outcomes, concepts, case studies, labs, assignments, rubrics, regulatory considerations, failure modes, interview questions, key terms, and sources.

**Version:** 2026.2.1
**Source cutoff:** 24 October 2026
**Format:** 40 weeks, three terms, three tracks, 30 modules (00–22 plus 01A, 14A, 14B, and advanced electives 23–26)
**Capstone:** 10–12 weeks, evidence dossier and defense

---

## 1. Program map

| Term | Weeks | Modules | Theme | Milestone |
|---|---:|---|---|---|
| 1 | 1–12 | 00–06 incl. 01A | Clinical, data, statistical, and ML foundations | Reproducible risk-model report |
| 2 | 13–26 | 07–14B | Modalities, hardware, efficient AI, and industry domains | Modality deep dive, hardware/efficiency plan, and domain playbook |
| 3 | 27–40 | 15–22 | Production, governance, product, capstone | Evidence dossier, demo, defense |

### 1.1 Weekly rhythm

| Component | Time | Output |
|---|---:|---|
| Lecture / seminar | 3 h | Concept map and questions |
| Lab | 3 h | Reproducible notebook or pipeline |
| Case clinic | 2 h | One-page critique |
| Project studio | 4 h | Increment and risk log |
| Independent reading | 2–4 h | Annotated notes |
| Assessment / portfolio | 2–3 h | Rubric work, reflection, portfolio updates |

### 1.2 Tracks

| Track | Required | Recommended | Capstone archetypes |
|---|---|---|---|
| Health Systems Analytics | Core (00, 01, 01A, 02–06); 08, 09, 14, 15, 16, 17, 18, 19 | 07, 10, 11, 14A, 14B, 20, 21 | Sepsis, readmission, capacity, documentation |
| Biopharma & Translational | Core (00, 01, 01A, 02–06); 07, 12, 13, 14, 15, 16, 17, 18, 19 | 08, 09, 10, 11, 14A, 14B, 20, 21 | Biomarker, trial recruitment, RWE, safety |
| MedTech, Sensors & Imaging | Core (00, 01, 01A, 02–06); 07, 10, 11, 14, 14A, 15, 16, 17, 18, 19 | 08, 09, 12, 13, 14B, 20, 21 | Arrhythmia, imaging triage, device safety, SaMD |

---

## 2. Term 1 — Foundations

### Module 00 — Orientation and the 2026 Industry Map

**File:** `00_orientation_and_industry_map.md`
**Time:** 16 hours
**Core artifact:** Project charter

Key topics:

- The biomedical data value chain.
- Provider, payer, pharma, device, diagnostic, wearable, and public-health segments.
- Decision-centric versus model-centric work.
- Prediction, causal inference, and evidence.
- Industry roles and cross-functional teams.
- Regulatory triage.
- Project charter.
- Evidence hierarchy for industry claims.
- Data ethics, neurodiversity, and AI assistance.
- Portfolio artifacts.

Assessment: charter, industry map, evidence critique.

### Module 01 — Clinical Foundations for Data Scientists

**File:** `01_clinical_foundations.md`
**Time:** 32 hours
**Core artifact:** Clinical data-plausibility memo

Key topics:

- History, physical exam, and documentation.
- Vital signs and early-warning scores.
- Laboratory values and units.
- Cardiovascular physiology and heart failure.
- ECG interpretation.
- Respiratory physiology, spirometry, and acid-base.
- Neurology, localization, and neurodiversity.
- Imaging modalities.
- Coding, notes, and copy-forward.
- Pretest probability and test characteristics.
- Cognitive bias and automation bias.
- Clinical plausibility checks.

Assessment: data-plausibility memo, ECG/spirometry labs, phenotype comparison.

### Module 01A — Foundations, Units, Measurement, and Assumptions

**File:** `01A_foundations_assumptions_and_units.md`
**Time:** 20 hours
**Core artifact:** Assumptions ledger

Key topics:

- Numbers, scales, dimensions, and units.
- Clinical unit conversions and formulas.
- Accuracy, precision, bias, calibration, uncertainty, and signal-to-noise.
- Mathematical basics: algebra, functions, linear algebra, calculus, probability, statistics, graphs.
- Biological basics: cells, organ systems, homeostasis, pharmacokinetics, pharmacodynamics, genetics, immunology.
- Clinical and epidemiological basics: risk, incidence, prevalence, study designs, bias.
- Engineering basics: signal chain, sampling, filters, electrical safety, materials, control, human factors.
- Statistical, causal, ML, clinical, device, economic, and ethical assumptions.
- Assumptions ledger template and tests.
- Unit and assumption error gallery.

Assessment: conversion module, measurement-error simulation, assumption audit, pharmacokinetics, NNT/cost-effectiveness.

### Module 02 — The Biomedical Data Landscape

**File:** `02_biomedical_data_landscape.md`
**Time:** 20 hours
**Core artifact:** Data-source map

Key topics:

- EHR, claims, registries, trials, imaging, signals, omics, wearables, public health.
- FHIR, OMOP, DICOM, LOINC, SNOMED, ICD, CPT, RxNorm, UCUM.
- Access pathways: public, registered, licensed, institutional, prospective, synthetic.
- HIPAA, GDPR, DUAs, BAAs, IRB, DPIA.
- De-identification and linkage.
- Data quality dimensions.
- Source-selection framework.
- Dataset case studies.

Assessment: source-selection memo, FHIR/OMOP/DICOM labs.

### Module 03 — Data Engineering and Interoperability

**File:** `03_data_engineering_interoperability.md`
**Time:** 28 hours
**Core artifact:** Reproducible cohort pipeline

Key topics:

- Layered architecture and medallion model.
- Batch, streaming, lakehouse, federated, trusted research environments.
- SQL cohort construction, temporal joins, deduplication, gaps and islands.
- Python project structure, configuration, typed data, tests.
- dbt, Airflow, Dagster, Prefect.
- FHIR ingestion and FHIR-to-OMOP mapping.
- DICOM pipelines and de-identification.
- Data versioning and lineage.
- Security controls.
- Pipeline incident response.

Assessment: cohort pipeline, dbt tests, quality monitor, incident report.

### Module 04 — Statistics and Causal Inference

**File:** `04_statistics_causal_inference.md`
**Time:** 32 hours
**Core artifact:** Causal protocol

Key topics:

- Estimands and the estimand ladder.
- Prediction versus causation.
- Probability, uncertainty, bootstrap, p-values, multiple comparisons.
- Linear, logistic, Poisson, mixed-effects, and spline models.
- Survival analysis, censoring, competing risks, immortal time.
- Missing data mechanisms and multiple imputation.
- DAGs, confounding, colliders, mediators.
- Propensity scores, IPTW, doubly robust estimation.
- Instrumental variables, regression discontinuity, difference-in-differences.
- Target trial emulation.
- Sensitivity analysis.

Assessment: DAG, IPTW lab, target trial protocol, survival analysis.

### Module 05 — Study Design and Evidence Generation

**File:** `05_study_design_evidence.md`
**Time:** 24 hours
**Core artifact:** Evidence plan

Key topics:

- Claim → design → decision.
- Evidence hierarchy and fit-for-purpose design.
- Trial phases, randomization, blinding, controls, estimands, power.
- Explanatory versus pragmatic trials.
- Real-world evidence and external controls.
- Validation levels: internal, temporal, external, prospective, impact.
- Diagnostic accuracy and prediction-model reporting.
- Bias taxonomy.
- Regulatory and payer evidence.
- Evidence roadmap.

Assessment: protocol, sample-size lab, AI study appraisal, RWE design, roadmap.

### Module 06 — Machine Learning Foundations

**File:** `06_machine_learning_foundations.md`
**Time:** 32 hours
**Core artifact:** Calibrated risk model

Key topics:

- Decision, time zero, label, features, action.
- Label quality and leakage.
- Splitting: random, grouped, temporal, site-held-out, nested CV.
- Preprocessing pipelines.
- Logistic regression, trees, forests, boosting, SVM.
- Class imbalance.
- AUROC, AUPRC, sensitivity/specificity, PPV/NPV.
- Calibration, Brier score, decision curves.
- Threshold selection.
- Interpretability.
- Uncertainty and conformal prediction.
- Fairness and subgroup performance.
- Model selection and documentation.

Assessment: leakage-free pipeline, leakage audit, calibration report, fairness audit, model card.

---

## 3. Term 2 — Modalities, Hardware, and Efficient AI

### Module 07 — Deep Learning, Multimodal Models, LLMs, and Agents

**File:** `07_deep_learning_multimodal_llms.md`
**Time:** 32 hours
**Core artifact:** Multimodal or LLM evaluation

Key topics:

- Neural network fundamentals.
- CNNs, RNNs, transformers, ViTs.
- Self-supervised and contrastive learning.
- Multimodal fusion.
- Medical foundation models: imaging, text, omics, signals.
- LLM prompting, RAG, fine-tuning, LoRA.
- Hallucination, grounding, and evaluation.
- Agents, tools, prompt injection, human approval.
- Compute, latency, and cost.
- Regulatory and security issues.

Assessment: imaging model, LLM evaluation, RAG prototype, red-team report.

### Module 08 — EHR, Claims, and Operations Analytics

**File:** `08_ehr_claims_operations.md`
**Time:** 28 hours
**Core artifact:** Operational decision-support analysis

Key topics:

- Operational decision taxonomy.
- Cohort construction and temporal features.
- Comorbidity measures.
- Claims structure, episodes, cost, risk adjustment.
- Readmission, no-show, length of stay, deterioration, ED demand, OR scheduling.
- Care gaps.
- Alert burden.
- Operational evaluation designs.
- Equity and access.
- Value and adoption.

Assessment: readmission model, claims analysis, capacity forecast, no-show evaluation.

### Module 09 — Clinical NLP

**File:** `09_clinical_nlp.md`
**Time:** 24 hours
**Core artifact:** Grounded extraction evaluation

Key topics:

- Clinical text sources and documentation behavior.
- Negation, uncertainty, temporality, attribution.
- NER, assertion, relation extraction, phenotyping, coding, summarization.
- Rule-based, classical ML, transformers, LLMs.
- Annotation guidelines and inter-annotator agreement.
- Evaluation at span, concept, patient, and downstream levels.
- De-identification.
- Grounded summarization and abstention.
- Copy-forward and bias.
- Monitoring template changes.

Assessment: de-identification pipeline, NER, phenotype validation, summarization audit.

### Module 10 — Medical Imaging

**File:** `10_medical_imaging.md`
**Time:** 32 hours
**Core artifact:** Imaging model with site-held-out validation

Key topics:

- Modalities and physics.
- DICOM metadata and pixel data.
- Preprocessing, resampling, normalization, augmentation.
- Labels from reports and expert annotation.
- Classification, detection, segmentation.
- Foundation models.
- Shortcuts and leakage.
- Metrics: AUROC, Dice, Hausdorff, sensitivity at fixed FP.
- Site, scanner, and subgroup validation.
- Workflow roles: triage, concurrent, second read, autonomous.
- FDA/CE pathways.

Assessment: DICOM volume, chest X-ray classifier, segmentation, shortcut audit, site-held-out validation.

### Module 11 — Physiological Signals and Wearables

**File:** `11_physiological_signals_wearables.md`
**Time:** 32 hours
**Core artifact:** Signal pipeline and digital biomarker

Key topics:

- Sampling, aliasing, filtering, time-frequency analysis.
- ECG, PPG, EEG, respiratory, accelerometry.
- Signal quality and artifact.
- Feature engineering and deep learning.
- Digital biomarker validation: analytical, technical, clinical, utility, usability.
- Reference standards.
- Free-living validation.
- Skin tone, body composition, activity, adherence.
- Edge versus cloud.
- Regulatory claims.

Assessment: ECG quality, AF classification, PPG validation, sleep staging, edge benchmark.

### Module 12 — Omics and Precision Medicine

**File:** `12_omics_precision_medicine.md`
**Time:** 32 hours
**Core artifact:** Biomarker discovery report

Key topics:

- Central dogma and data types.
- Sequencing, variant calling, annotation.
- ACMG classification and VUS.
- GWAS, PRS, ancestry portability.
- Bulk RNA-seq, normalization, differential expression.
- Single-cell QC, integration, clustering, annotation.
- Spatial transcriptomics.
- Proteomics and metabolomics.
- Multi-omics integration.
- Target discovery and validation.
- Biomarker evidence and clinical utility.

Assessment: variant report, RNA-seq DE, single-cell clustering, spatial analysis, PRS, multi-omics.

### Module 13 — Biopharma and Clinical Development

**File:** `13_biopharma_clinical_development.md`
**Time:** 28 hours
**Core artifact:** Trial or RWE strategy

Key topics:

- Drug-development value chain.
- Biomarker strategy and companion diagnostics.
- Trial feasibility and recruitment.
- CDISC SDTM/ADaM, GxP, Part 11.
- Safety signal detection and pharmacovigilance.
- Real-world evidence and external controls.
- AI in discovery and development.
- Regulatory and payer evidence.
- Portfolio decisions.

Assessment: feasibility report, SDTM mapping, safety signal, RWE protocol, recruitment forecast.

### Module 14 — Medical Devices, SaMD, and Regulation

**File:** `14_medical_devices_samd_regulation.md`
**Time:** 32 hours
**Core artifact:** Regulatory evidence package

Key topics:

- Device, SaMD, SiMD, CDS definitions.
- Intended use and indications.
- FDA 510(k), De Novo, PMA.
- GMLP and PCCP.
- ISO 13485, IEC 62304, ISO 14971, IEC 62366.
- Cybersecurity.
- EU MDR/IVDR and EU AI Act.
- 2026 regulatory watchlist.
- Evidence package: analytical, clinical, usability, software, risk.
- Post-market surveillance.

Assessment: regulatory triage, intended-use statement, risk register, requirements matrix, PCCP, threat model.

---

### Module 14A — Medical Hardware, Sensors, Semiconductors, and Robotics

**File:** `14A_medical_hardware_sensors_chips_robotics.md`
**Time:** 36 hours
**Core artifact:** Device data and evidence plan

Key topics:

- Medical device signal chain: sensor, analog front end, ADC, processor, firmware, algorithm, output.
- Biopotential, optical, pressure, flow, temperature, motion, chemical, ultrasound, and imaging sensors.
- Sensor physics, calibration, artifacts, skin-tone and perfusion bias.
- Semiconductors: AFE, ADC, MCU, DSP, FPGA, SoC, GPU/NPU, ASIC, PMIC, wireless, isolation, memory.
- Edge AI, TinyML, quantization, firmware, secure boot, SBOM, supply chain.
- Electrical safety, EMC, sterilization, biocompatibility, battery safety.
- Medical equipment classes: monitors, ventilators, pumps, dialysis, ECMO, pacemakers, insulin pumps, imaging, linacs, endoscopes, lab analyzers.
- Medical robotics: surgical, rehabilitation, prosthetic, capsule, imaging, lab, pharmacy, logistics.
- Kinematics, Jacobian, singularities, PID/impedance/admittance control, registration, haptics, teleoperation, autonomy.
- IEC 60601-1, 80601-2-77, 80601-2-78, ISO/TS 15066, ISO 14971, IEC 62304, cybersecurity.
- Hardware-in-the-loop validation and device evidence.

Assessment: signal-chain simulation, PPG bias, TinyML, robot kinematics, safety monitor, device teardown, HIL plan.

### Module 14B — Efficient AI: Distillation, Quantization, and On-Device Inference (advanced elective)

**File:** `14B_efficient_ai_distillation_quantization_on_device_inference.md`
**Time:** 28 hours
**Core artifact:** Efficient-model dossier

Key topics:

- Deployment Pareto frontier: accuracy, calibration, latency, energy, memory, cost, safety.
- Target hardware: MCU, DSP, FPGA, SoC/NPU, GPU, ASIC, cloud.
- Memory hierarchy, roofline, arithmetic intensity, p99 latency, joules/inference.
- Compression family: distillation, quantization, pruning, low-rank, NAS.
- Knowledge distillation math, variants, teacher bias, calibration, subgroup risk.
- Quantization formats, affine/symmetric, per-tensor/channel, PTQ/QAT, calibration, STE, operator fallback.
- Pruning and hardware sparsity; why pruning may not reduce latency.
- Inference optimization: fusion, layout, batching, KV cache, speculative decoding, runtimes.
- On-device runtime: DMA, double buffering, RTOS, watchdog, fallback, secure OTA.
- Validation: target accuracy/calibration/subgroup, p99, energy, thermal, HIL.
- IEC 62304, PCCP, GMLP, ISO 14971, cybersecurity, post-market monitoring.

Assessment: profiling, PTQ, QAT, distillation, MCU/edge deployment, LLM quantization, HIL, subgroup audit, Pareto decision.

---

## 4. Term 3 — Production, Product, and Capstone

### Module 15 — Deployment and MLOps

**File:** `15_deployment_mlops.md`
**Time:** 28 hours
**Core artifact:** Deployment and monitoring plan

Key topics:

- Batch, real-time, streaming, edge, embedded, shadow, human-in-loop.
- Packaging, containers, versioning, model registry.
- CI/CD/CT.
- Monitoring: availability, data quality, drift, calibration, fairness, operations, outcomes, cost.
- Incident response and rollback.
- Shadow and phased rollout.
- Change control.
- Total cost of ownership.

Assessment: containerized API, CI pipeline, monitoring dashboard, shadow evaluation, incident simulation.

### Module 16 — Privacy, Security, and Ethics

**File:** `16_privacy_security_ethics.md`
**Time:** 24 hours
**Core artifact:** Privacy and ethics impact assessment

Key topics:

- Privacy principles and legal bases.
- De-identification, anonymization, pseudonymization.
- Differential privacy, federated learning, secure computation, synthetic data.
- CIA triad, threat modeling, STRIDE.
- Adversarial ML.
- Fairness definitions and trade-offs.
- Consent, equity, accessibility, neurodiversity.
- Governance bodies.
- HIPAA, GDPR, EU AI Act, FTC.

Assessment: threat model, de-identification audit, fairness audit, membership inference, ethics assessment.

### Module 17 — Product, Economics, and Leadership

**File:** `17_product_economics_leadership.md`
**Time:** 24 hours
**Core artifact:** Value case and adoption plan

Key topics:

- Stakeholders: user, buyer, beneficiary, blocker.
- Value proposition.
- ROI, cost avoidance, capacity, quality.
- Cost-effectiveness, QALY, ICER, budget impact.
- Reimbursement and procurement.
- Adoption and change management.
- Business models.
- Cross-functional leadership.
- Portfolio prioritization.
- Executive communication.

Assessment: stakeholder map, ROI model, cost-effectiveness, adoption plan, value case.

### Module 18 — Quality, Evaluation, and Reporting

**File:** `18_quality_evaluation_reporting.md`
**Time:** 24 hours
**Core artifact:** Model card and evidence dossier

Key topics:

- Reporting guidelines: TRIPOD+AI, CONSORT-AI, SPIRIT-AI, STARD-AI, DECIDE-AI, CLAIM, PROBAST.
- Appraising clinical AI evidence.
- Model cards and datasheets.
- Reproducibility levels.
- Quality management.
- Post-market evaluation and recalls.
- Audit and CAPA.
- Evidence dossier.

Assessment: study appraisal, model card, reproducibility audit, monitoring plan, recall analysis.

### Module 19 — Capstone and Hackathons

**File:** `19_capstone_hackathons.md`
**Time:** 80–120 hours
**Core artifact:** Capstone evidence dossier and demo

Key topics:

- Capstone architecture and milestones.
- Team roles and RACI.
- Eight archetypes.
- 44 project briefs across imaging, signals, EHR, NLP, omics, public health, hardware, sensors, robotics, and efficient on-device AI.
- 24-hour, 48-hour, one-week, and 12-week formats.
- Competition landscape 2025–2026.
- Scoping, validation, and leakage playbook.
- Evidence dossier.
- Judging rubric.
- Demo day and executive pitch.
- Data access and mentorship.
- Templates.

Assessment: charter, weekly milestones, dossier, demo, defense.

### Module 20 — Career, Portfolio, and Interviews

**File:** `20_career_portfolio_interviews.md`
**Time:** 12 hours
**Core artifact:** Portfolio and interview plan

Key topics:

- Role map.
- Three portfolio artifacts.
- Case-study structure.
- Resume and LinkedIn.
- Interview types.
- Technical, clinical, regulatory, product, behavioral.
- Take-home strategy.
- System design.
- Negotiation.
- 90-day onboarding.

Assessment: portfolio, mock interviews, case study, negotiation plan.

### Module 21 — Datasets, Tools, and Benchmarks

**File:** `21_datasets_tools_benchmarks.md`
**Time:** 16 hours
**Core artifact:** Personal resource library

Key topics:

- Dataset catalog by modality.
- Access, licensing, credentialing.
- Benchmark leaderboards.
- Surgical video, robot kinematics, and device telemetry datasets.
- Dataset-selection framework.
- Tool stacks.
- Benchmark pitfalls.
- Teaching cautions.
- Labs and audit exercises.

Assessment: dataset audit, license audit, benchmark critique, resource library.

### Module 22 — Sources and Further Reading

**File:** `22_sources_and_further_reading.md`
**Time:** Self-paced
**Core artifact:** Annotated bibliography

Key topics:

- Primary regulation and policy.
- Methods and reporting.
- Clinical background.
- Datasets and benchmarks.
- Industry and market.
- Professional communities.
- Source-evaluation checklist.
- Keeping current.

Assessment: annotated bibliography and source log.

---

## 4A. Advanced electives (post-core)

These are optional depth modules. They extend the program if all are taken; one or two can replace a track elective.

### Module 23 — Specialty Electives

**File:** `23_specialty_electives.md`
**Time:** 18–24 hours
**Core artifact:** Specialty primer and lab

Six primers: oncology, cardiology, pathology, critical care, pediatrics/OB, and public health. Each covers the decision and buyer, data and standards, methods and endpoints, real use cases, regulation/equity, a lab, and interview questions.

### Module 24 — Hardware, Firmware, EMC, HIL, and Robotics Labs

**File:** `24_hardware_robotics_labs.md`
**Time:** 28 hours
**Core artifact:** Bench/robotics evidence package

Ten labs: signal chain/ADC, calibration/Gauge R&R, firmware/RTOS/watchdog/secure boot, EMC pre-compliance, HIL fault injection, ROS 2/Gazebo/MuJoCo, robot FK/IK/Jacobian, teleoperation safety, TinyML, and integrated capstone.

### Module 25 — Trial Operations, HTA/Market Access, and Global Regulation

**File:** `25_trial_operations_hta_global_regulatory.md`
**Time:** 28 hours
**Core artifact:** Trial/HTA/regulatory roadmap

Trial execution, risk-based monitoring, decentralized trials, HTA, ICER/budget impact, payer dossiers, and regulatory pathways across US, EU, UK, Canada, Australia, Japan, China, Brazil, and India.

### Module 26 — Cloud, Kubernetes, and Streaming Production Labs

**File:** `26_cloud_streaming_production_labs.md`
**Time:** 24 hours
**Core artifact:** Production deployment and DR report

Containers, Kubernetes, Terraform, Kafka/streaming, FHIR subscriptions, feature stores, observability, security, cost, and disaster recovery.

## 5. Assessment summary

| Assessment | Weight | Individual/team | Module |
|---|---:|---|---|
| Weekly labs | 20% | Individual | 01–18 incl. 01A, 14A |
| Case critiques | 15% | Individual | 01–18; brief 19 §11A.3 |
| Midterm evidence dossier | 20% | Team | 00–06 incl. 01A; brief 19 §11A.1 |
| Track playbook | 15% | Team | 07–14A; brief 19 §11A.2 |
| Capstone | 30% | Team + individual defense | 19 |

Passing requires at least 50% on the individual capstone defense. Full briefs and rubrics are in Module 19, §11A.

## 6. Capstone evidence dossier

Required artifacts:

1. Project charter and intended-use statement.
2. Data provenance, access, and quality report.
3. Reproducible analysis with environment lockfile.
4. Model or analytic method with baseline.
5. Temporal and external validation where possible.
6. Calibration, decision-curve, and subgroup analysis.
7. Error analysis and failure-mode register.
8. Regulatory and ethical triage memo.
9. Model card and datasheet.
10. Deployment architecture and monitoring plan.
11. Health-economic or operational value case.
12. 10-minute executive pitch.
13. 20-minute technical defense.

## 7. Core tools

- Python, SQL, Git, Docker.
- pandas, scikit-learn, PyTorch, statsmodels.
- dbt, Airflow or Dagster, MLflow, DVC.
- FHIR, OMOP, DICOM, LOINC, SNOMED, ICD.
- MONAI, WFDB, NeuroKit2, scanpy, Seurat.
- Hugging Face, LangChain or LlamaIndex, vector database.
- Fairlearn, SHAP, Evidently, Great Expectations.
- Cloud: AWS, Azure, or GCP.

## 8. Core datasets

- MIMIC-IV, eICU, MIMIC-CXR.
- PhysioNet, PTB-XL, Sleep-EDF, TUH EEG.
- UK Biobank, All of Us, TCGA, CELLxGENE.
- Open Targets, ChEMBL, PubChem.
- CheXpert, BraTS, RSNA challenge data.
- HealthBench, MedAgentBench, EHRSHOT, FoMoH.
- Synthea, Eunomia, CMS SynPUF.
- OpenFDA/AEMS, CMS Open Payments, HCUP, NHANES.

## 9. Program outcomes

Graduates can frame a decision, acquire and govern data, design evidence, build calibrated models, evaluate clinical utility, navigate regulation, deploy and monitor systems, quantify value, communicate across functions, and deliver a decision-ready capstone.

---

> **Start:** `README.md`
> **Orientation:** `00_orientation_and_industry_map.md`
> **Capstone:** `19_capstone_hackathons.md`
> **Datasets:** `21_datasets_tools_benchmarks.md`

---

## 10. Governance and instructor artifacts

| File | Purpose |
|---|---|
| `VALIDATION_STATUS.md` | What is complete, drafted, or pending |
| `COMPETENCY_MAP.md` | Outcome-to-module/lab/assessment crosswalk |
| `GOVERNANCE_AND_EXTERNAL_REVIEW.md` | Reviewer terms, rubric, COI, sign-off |
| `REGULATORY_VERIFICATION_LOG.md` | Tiered claim register and review status |
| `INSTRUCTOR_KIT.md` | Teaching notes, anchor solutions, lab index |
| `GRADING_CALIBRATION.md` | Grader training and anchor descriptors |
| `PILOT_PROTOCOL.md` | 5–10 learner feasibility pilot |
| `labs/` | Starter notebooks, requirements, sample solutions |
