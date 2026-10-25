#!/usr/bin/env python3
"""Generate the per-module folder scaffold for the expanded course edition.

Creates modules/<folder>/_index.md for every module listed in MANIFEST.
Idempotent: re-running rewrites indexes but never deletes topic files.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(ROOT, "modules")

# (id, folder, title, source_file, prereq, unlocks, assessment, topics)
# topic = (slug, title, scope)
MANIFEST = [
    ("00", "00_orientation", "Orientation and Industry Map",
     "00_orientation_and_industry_map.md",
     "none", "every module", "Capstone charter (Week 4)",
     [
         ("01_why-this-course-exists", "Why This Course Exists",
          "The gap between academic data science and regulated biomedical work; what employers actually hire for."),
         ("02_the-2026-industry-landscape", "The 2026 Industry Landscape",
          "Health systems, payers, pharma, medtech, diagnostics, consumer health, and the AI vendor layer."),
         ("03_stakeholder-and-buyer-map", "Stakeholder and Buyer Map",
          "Who uses, who pays, who blocks, who bears harm; mapping a deal from clinician to procurement."),
         ("04_career-tracks-and-role-definitions", "Career Tracks and Role Definitions",
          "Titles, scope, and skill mixes across clinical data science, regulatory science, ML engineering, and informatics."),
         ("05_the-project-charter", "The Project Charter",
          "Turning a vague ask into a decision, user, comparator, outcome, data plan, and stop criteria."),
         ("06_evidence-hierarchy-for-industry-claims", "Evidence Hierarchy for Industry Claims",
          "Why the pyramid misleads in product work; matching evidence strength to claim strength."),
         ("07_regulatory-triage-map", "Regulatory Triage Map",
          "A first-pass map of FDA, EU MDR/IVDR, EU AI Act, HIPAA, GDPR, and CLIA touchpoints."),
         ("08_ethics-and-professional-conduct", "Ethics and Professional Conduct",
          "Conflicts, data use, authorship, escalation, and what to do when a result is inconvenient."),
         ("09_how-to-use-this-package", "How to Use This Package",
          "Reading order, tracks, labs, assessments, and how the survey files relate to topic folders."),
         ("10_success-metrics-and-portfolio", "Success Metrics and Portfolio",
          "Program metrics, personal portfolio artifacts, and what evidence of competence looks like."),
     ]),
    ("01", "01_clinical_foundations", "Clinical Foundations for Data Scientists",
     "01_clinical_foundations.md",
     "00", "01A, 02, 08, 09, 23",
     "Clinical plausibility memo (Week 3)",
     [
         ("01_clinical-reasoning-and-workflow", "Clinical Reasoning and Workflow",
          "How clinicians form and test hypotheses; where data enters and leaves the encounter."),
         ("02_anatomy-physiology-and-vital-signs", "Anatomy, Physiology, and Vital Signs",
          "Organ systems, normal ranges, compensation, and what derangement looks like in data."),
         ("03_laboratory-medicine", "Laboratory Medicine",
          "Specimens, assays, reference intervals, sensitivity/specificity, and pre-analytical error."),
         ("04_pharmacology-basics", "Pharmacology Basics",
          "Absorption, distribution, metabolism, excretion, interactions, and why dosing data is messy."),
         ("05_disease-taxonomy-and-coding", "Disease Taxonomy and Coding",
          "ICD-10/11, SNOMED CT, MeSH, Orphanet, and the difference between a code and a diagnosis."),
         ("06_radiology-and-imaging-basics", "Radiology and Imaging Basics",
          "Modalities, acquisition, reporting workflow, and what a radiologist does with an image."),
         ("07_cardiology-and-ecg-basics", "Cardiology and ECG Basics",
          "The cardiac cycle, leads, intervals, common rhythms, and artifact that mimics disease."),
         ("08_documentation-and-clinical-narratives", "Documentation and Clinical Narratives",
          "Note types, copy-forward, templates, coded vs narrative content, and documentation bias."),
         ("09_special-populations", "Special Populations",
          "Pediatrics, pregnancy, older adults, renal/hepatic impairment, and device-dependent patients."),
         ("10_clinical-plausibility-checks", "Clinical Plausibility Checks",
          "A checklist for catching impossible values, impossible trajectories, and impossible cohorts."),
     ]),
    ("01A", "01A_foundations", "Foundations, Assumptions, and Units",
     "01A_foundations_assumptions_and_units.md",
     "00", "every quantitative module", "Assumptions ledger (Week 4)",
     [
         ("01_mathematical-basics", "Mathematical Basics",
          "Functions, logs, exponents, rates, ratios, and the algebra used in every later module."),
         ("02_probability-and-statistics-basics", "Probability and Statistics Basics",
          "Distributions, expectation, variance, sampling, estimation, and hypothesis testing intuition."),
         ("03_units-and-metrology", "Units and Metrology",
          "SI units, clinical units, conversion factors, significant figures, and measurement uncertainty."),
         ("04_dimensional-analysis", "Dimensional Analysis",
          "Checking equations by units; common conversion errors in laboratory and device data."),
         ("05_biological-basics", "Biological Basics",
          "Cells, genes, proteins, metabolism, homeostasis, and the timescales that matter for data."),
         ("06_pharmacokinetics", "Pharmacokinetics",
          "One- and two-compartment models, half-life, clearance, volume of distribution, dosing."),
         ("07_epidemiology-basics", "Epidemiology Basics",
          "Incidence, prevalence, risk, rate, standardization, and measures of association."),
         ("08_engineering-and-device-basics", "Engineering and Device Basics",
          "Signals, sampling, noise, resolution, calibration, and the device as a measurement instrument."),
         ("09_computing-basics", "Computing Basics",
          "Floating point, missing values, dates and time zones, identifiers, and file formats."),
         ("10_the-assumptions-ledger", "The Assumptions Ledger",
          "A reusable artifact for recording, testing, and communicating every assumption behind a claim."),
     ]),
    ("02", "02_data_landscape", "The Biomedical Data Landscape",
     "02_biomedical_data_landscape.md",
     "01, 01A", "03, 08, 21", "Data-source decision memo (Week 5)",
     [
         ("01_ehr-data", "EHR Data",
          "Encounter, order, result, medication, note, and flow-sheet data; what each can and cannot answer."),
         ("02_claims-and-billing", "Claims and Billing Data",
          "Claim lifecycle, coding, adjudication, and the economics embedded in the data."),
         ("03_registries-and-public-health", "Registries and Public Health Data",
          "Disease registries, surveillance, vital statistics, and reporting incentives."),
         ("04_imaging-and-dicom", "Imaging and DICOM",
          "Pixel data, metadata, series structure, and the imaging archive as a data source."),
         ("05_signals-and-wearables", "Signals and Wearables",
          "ECG, PPG, accelerometry, CGM, and the provenance problems of consumer devices."),
         ("06_omics-data", "Omics Data",
          "Genomic, transcriptomic, proteomic, and single-cell data at a survey level."),
         ("07_trial-data", "Trial Data",
          "EDC, CDISC, randomization, protocol deviations, and the difference from routine care data."),
         ("08_standards-and-terminologies", "Standards and Terminologies",
          "HL7 v2, FHIR, OMOP, LOINC, SNOMED CT, ICD, RxNorm, and how they relate."),
         ("09_licensing-and-access", "Licensing and Access",
          "DUAs, BAAs, data-use tiers, credentialed access, and the cost of getting data wrong."),
         ("10_data-quality-dimensions", "Data Quality Dimensions",
          "Completeness, accuracy, timeliness, consistency, validity, and fitness for purpose."),
     ]),
    ("03", "03_data_engineering", "Data Engineering and Interoperability",
     "03_data_engineering_interoperability.md",
     "02", "06, 15, 26", "Reproducible cohort pipeline (Week 7)",
     [
         ("01_architecture-patterns", "Architecture Patterns",
          "Lakehouse, medallion layers, event-driven ingestion, and clinical constraints on each."),
         ("02_sql-for-clinical-data", "SQL for Clinical Data",
          "Window functions, temporal joins, deduplication, and the queries that define cohorts."),
         ("03_python-engineering", "Python Engineering",
          "Packaging, typing, testing, configuration, and reproducible environments."),
         ("04_orchestration", "Orchestration",
          "DAGs, idempotency, backfills, retries, and scheduling clinical pipelines."),
         ("05_fhir-ingestion", "FHIR Ingestion",
          "Resources, references, bundles, search parameters, and flattening to analytics tables."),
         ("06_dicom-pipelines", "DICOM Pipelines",
          "Series assembly, de-identification, pixel handling, and imaging cohort construction."),
         ("07_data-versioning-and-lineage", "Data Versioning and Lineage",
          "Snapshots, hashes, provenance, and answering 'where did this number come from'."),
         ("08_deidentification-engineering", "De-identification Engineering",
          "Safe Harbor, expert determination, dates, free text, and re-identification risk."),
         ("09_security-engineering", "Security Engineering",
          "Least privilege, encryption, audit logs, secrets, and pipeline threat models."),
         ("10_testing-and-observability", "Testing and Observability",
          "Data tests, freshness checks, schema contracts, and alerting on pipeline failure."),
     ]),
    ("04", "04_statistics", "Statistics and Causal Inference",
     "04_statistics_causal_inference.md",
     "01A", "05, 06, 13, 18", "Causal analysis protocol (Week 9)",
     [
         ("01_estimation-and-inference", "Estimation and Inference",
          "Estimands, standard errors, confidence intervals, and what a p-value does not say."),
         ("02_regression-models", "Regression Models",
          "Linear, logistic, Poisson, and Cox models as tools for adjustment and prediction."),
         ("03_survival-analysis", "Survival Analysis",
          "Censoring, hazards, Kaplan-Meier, competing risks, and time-varying confounding."),
         ("04_missing-data", "Missing Data",
          "MCAR, MAR, MNAR, multiple imputation, and sensible sensitivity analysis."),
         ("05_causal-inference-and-dags", "Causal Inference and DAGs",
          "Confounding, colliders, mediators, and drawing the graph before the model."),
         ("06_propensity-scores-and-weighting", "Propensity Scores and Weighting",
          "Matching, IPTW, trimming, balance diagnostics, and effective sample size."),
         ("07_target-trial-emulation", "Target Trial Emulation",
          "Specifying eligibility, treatment strategies, assignment, time zero, and outcomes."),
         ("08_sensitivity-analysis", "Sensitivity Analysis",
          "Negative controls, E-values, quantitative bias analysis, and tipping points."),
         ("09_subgroups-and-heterogeneity", "Subgroups and Heterogeneity",
          "Pre-specification, interaction tests, multiplicity, and credible subgroup claims."),
         ("10_reporting-statistical-results", "Reporting Statistical Results",
          "Effect sizes with intervals, absolute risk, number needed to treat, and honest uncertainty."),
     ]),
    ("05", "05_study_design", "Study Design and Evidence Generation",
     "05_study_design_evidence.md",
     "04", "13, 18, 25", "Evidence generation plan (Week 10)",
     [
         ("01_from-claim-to-evidence", "From Claim to Evidence",
          "Translating a marketing or clinical claim into a testable evidence plan."),
         ("02_trial-designs", "Trial Designs",
          "Parallel, crossover, adaptive, cluster, platform, and decentralised designs."),
         ("03_real-world-evidence", "Real-World Evidence",
          "When RWE is fit for a regulatory or payer decision, and when it is not."),
         ("04_validation-designs", "Validation Designs",
          "Internal, temporal, external, geographic, and prospective validation."),
         ("05_bias-taxonomy", "Bias Taxonomy",
          "Selection, information, confounding, immortal time, attrition, and measurement bias."),
         ("06_sample-size-and-power", "Sample Size and Power",
          "Power for differences, for precision, and for subgroup claims."),
         ("07_estimands", "Estimands",
          "ICH E9(R1) five attributes and why the estimand comes before the method."),
         ("08_external-validation", "External Validation",
          "Site selection, transportability, recalibration, and reporting standards."),
         ("09_evidence-synthesis", "Evidence Synthesis",
          "Systematic review, meta-analysis, indirect comparison, and network meta-analysis."),
         ("10_protocol-writing", "Protocol Writing",
          "A protocol skeleton that a regulator, IRB, and statistician can all read."),
     ]),
    ("06", "06_machine_learning", "Machine Learning Foundations",
     "06_machine_learning_foundations.md",
     "04", "07, 10, 11, 14B", "Model development report (Week 12)",
     [
         ("01_problem-framing", "Problem Framing",
          "Prediction versus decision; choosing a target that changes an action."),
         ("02_labels-and-outcomes", "Labels and Outcomes",
          "Label sources, ascertainment bias, proxy labels, and label noise."),
         ("03_splitting-and-validation", "Splitting and Validation",
          "Patient-level, temporal, and site-level splits; nested cross-validation."),
         ("04_preprocessing", "Preprocessing",
          "Imputation, scaling, encoding, and fitting inside the fold."),
         ("05_model-families", "Model Families",
          "Linear, tree, gradient-boosted, kernel, and neural approaches with trade-offs."),
         ("06_class-imbalance", "Class Imbalance",
          "Resampling, class weights, and why AUROC hides the problem."),
         ("07_metrics", "Metrics",
          "Discrimination, accuracy, sensitivity, PPV, F1, and their dependence on prevalence."),
         ("08_calibration", "Calibration",
          "Reliability diagrams, slope and intercept, Brier decomposition, and recalibration."),
         ("09_thresholds-and-decision-curves", "Thresholds and Decision Curves",
          "Net benefit, harm ratios, capacity constraints, and choosing an operating point."),
         ("10_interpretability", "Interpretability",
          "Coefficients, partial dependence, SHAP, and the limits of post-hoc explanation."),
         ("11_uncertainty", "Uncertainty",
          "Confidence intervals, conformal prediction, and abstention."),
         ("12_fairness-and-subgroups", "Fairness and Subgroups",
          "Subgroup performance, fairness criteria, and the impossibility results."),
         ("13_leakage", "Leakage",
          "A catalogue of leaks in clinical ML and how to test for each."),
     ]),
    ("07", "07_deep_learning", "Deep Learning, Multimodal Models, and LLMs",
     "07_deep_learning_multimodal_llms.md",
     "06", "09, 10, 11, 14B", "Foundation-model evaluation dossier (Week 14)",
     [
         ("01_neural-network-fundamentals", "Neural Network Fundamentals",
          "Layers, activations, loss functions, backpropagation, and optimization."),
         ("02_training-dynamics", "Training Dynamics",
          "Initialization, learning rates, regularization, overfitting, and debugging."),
         ("03_cnns-and-medical-vision", "CNNs and Medical Vision",
          "Convolutions, receptive fields, transfer learning, and medical imaging specifics."),
         ("04_sequence-models", "Sequence Models",
          "RNNs, temporal convolution, and attention for signals and text."),
         ("05_transformers", "Transformers",
          "Self-attention, positional encoding, scaling, and compute cost."),
         ("06_medical-foundation-models", "Medical Foundation Models",
          "Pretraining corpora, adaptation, and the evidence needed before clinical use."),
         ("07_llm-basics", "LLM Basics",
          "Tokenization, next-token prediction, instruction tuning, and decoding."),
         ("08_prompting-and-rag", "Prompting and Retrieval-Augmented Generation",
          "Prompt design, retrieval, grounding, and citation of sources."),
         ("09_agents-and-tools", "Agents and Tools",
          "Tool use, planning, memory, and the failure modes of autonomous clinical agents."),
         ("10_evaluating-generative-models", "Evaluating Generative Models",
          "Task metrics, human review, hallucination rates, and red teaming."),
         ("11_compute-latency-and-cost", "Compute, Latency, and Cost",
          "GPU economics, context length, batching, and serving trade-offs."),
     ]),
    ("08", "08_ehr_claims", "EHR, Claims, and Health System Operations",
     "08_ehr_claims_operations.md",
     "02", "17, 19", "Operational decision product (Week 15)",
     [
         ("01_health-system-operating-model", "Health System Operating Model",
          "How money, clinicians, and patients move through a hospital and clinic network."),
         ("02_ehr-cohorts", "EHR Cohorts",
          "Defining populations from encounter and order data without fooling yourself."),
         ("03_claims-analytics", "Claims Analytics",
          "Episode construction, attribution, risk adjustment, and cost analysis."),
         ("04_operational-models", "Operational Models",
          "No-show, length of stay, readmission, staffing, and OR scheduling."),
         ("05_capacity-and-patient-flow", "Capacity and Patient Flow",
          "Queues, bottlenecks, boarding, and simulation of flow interventions."),
         ("06_quality-measures", "Quality Measures",
          "HEDIS, CMS measures, STAR ratings, and measure specification logic."),
         ("07_revenue-cycle", "Revenue Cycle",
          "Coding, denials, prior authorization, and where analytics changes cash flow."),
         ("08_evaluating-operational-interventions", "Evaluating Operational Interventions",
          "Before/after, interrupted time series, difference-in-differences, and step-wedge."),
         ("09_case-studies", "Case Studies",
          "Documented operational analytics programmes and what actually moved."),
     ]),
    ("09", "09_clinical_nlp", "Clinical NLP",
     "09_clinical_nlp.md",
     "07", "10, 18", "Clinical NLP validation report (Week 16)",
     [
         ("01_clinical-text-sources", "Clinical Text Sources",
          "Notes, reports, messages, and forms; structure and provenance."),
         ("02_documentation-behaviour", "Documentation Behaviour",
          "Copy-forward, templates, late entries, and what they do to text data."),
         ("03_task-taxonomy", "Task Taxonomy",
          "Extraction, assertion, classification, linking, summarization, and question answering."),
         ("04_classical-methods", "Classical Methods",
          "Rules, dictionaries, regex, and statistical models that still win in production."),
         ("05_transformers-for-clinical-text", "Transformers for Clinical Text",
          "Domain pretraining, long documents, and efficient inference."),
         ("06_annotation", "Annotation",
          "Guidelines, adjudication, inter-annotator agreement, and annotation cost."),
         ("07_evaluation", "Evaluation",
          "Span-level metrics, assertion accuracy, and clinician review designs."),
         ("08_deidentification", "De-identification",
          "PHI categories, automated de-identification, and residual risk measurement."),
         ("09_summarization-and-grounding", "Summarization and Grounding",
          "Abstractive summarization, faithful citation, and hallucination detection."),
         ("10_llm-safety-in-clinical-text", "LLM Safety in Clinical Text",
          "Prompt injection, leakage, unsafe advice, and human-in-the-loop design."),
     ]),
    ("10", "10_imaging", "Medical Imaging and Pathology AI",
     "10_medical_imaging.md",
     "07", "18, 23", "Imaging evidence package (Week 18)",
     [
         ("01_modalities", "Modalities",
          "CT, MRI, X-ray, ultrasound, PET, digital pathology, and their data characteristics."),
         ("02_dicom-and-metadata", "DICOM and Metadata",
          "Tags, series, spacing, windowing, and the metadata that leaks labels."),
         ("03_preprocessing", "Preprocessing",
          "Resampling, normalization, augmentation, and preserving diagnostic content."),
         ("04_labels-and-annotation", "Labels and Annotation",
          "Radiologist reads, segmentation masks, adjudication, and label noise."),
         ("05_architectures", "Architectures",
          "Classification, detection, segmentation, and self-supervised approaches."),
         ("06_leakage-and-shortcuts", "Leakage and Shortcuts",
          "Site signatures, scanner effects, laterality markers, and hidden text."),
         ("07_evaluation", "Evaluation",
          "Lesion-level versus image-level metrics, bootstrap intervals, and reader studies."),
         ("08_workflow-integration", "Workflow Integration",
          "PACS, worklists, reading room latency, and radiologist trust."),
         ("09_regulatory-path", "Regulatory Path",
          "FDA clearance routes for imaging AI and the evidence each requires."),
         ("10_foundation-models", "Foundation Models",
          "Vision encoders, multimodal reporting, and their evaluation gaps."),
     ]),
    ("11", "11_signals", "Physiological Signals and Wearables",
     "11_physiological_signals_wearables.md",
     "01A, 06", "14A, 14B, 18", "Digital biomarker validation plan (Week 19)",
     [
         ("01_signal-fundamentals", "Signal Fundamentals",
          "Sampling, aliasing, filtering, noise, and the signal chain."),
         ("02_ecg", "ECG",
          "Leads, intervals, rhythms, quality indices, and automated interpretation."),
         ("03_eeg", "EEG",
          "Montages, frequency bands, artifacts, seizures, and sleep staging."),
         ("04_ppg-and-wearables", "PPG and Wearables",
          "Optical sensing, SpO2, heart rate, and motion artifact."),
         ("05_respiration-and-other-signals", "Respiration and Other Signals",
          "Respiratory effort, airflow, EMG, EDA, and continuous glucose."),
         ("06_feature-engineering", "Feature Engineering",
          "Time-domain, frequency-domain, and morphology features with clinical meaning."),
         ("07_deep-learning-for-signals", "Deep Learning for Signals",
          "1D CNNs, recurrent models, transformers, and augmentation."),
         ("08_digital-biomarkers", "Digital Biomarkers",
          "Definition, qualification, and the evidence a payer or regulator accepts."),
         ("09_validation", "Validation",
          "Reference standards, co-primary endpoints, and confounding by activity."),
         ("10_deployment-on-device", "Deployment on Device",
          "Latency, power, memory, and firmware constraints for signal models."),
     ]),
    ("12", "12_omics", "Omics and Precision Medicine",
     "12_omics_precision_medicine.md",
     "01A, 04", "13, 23", "Biomarker evidence plan (Week 20)",
     [
         ("01_data-types", "Data Types",
          "Genomics, transcriptomics, epigenomics, proteomics, metabolomics, and microbiomics."),
         ("02_genomics-and-variants", "Genomics and Variants",
          "Sequencing, alignment, calling, annotation, and clinical interpretation."),
         ("03_transcriptomics", "Transcriptomics",
          "Bulk RNA-seq, normalization, differential expression, and batch effects."),
         ("04_single-cell", "Single-Cell",
          "Cell capture, QC, clustering, annotation, and trajectory analysis."),
         ("05_spatial-transcriptomics", "Spatial Transcriptomics",
          "Spatial assays, deconvolution, and tissue architecture."),
         ("06_proteomics-and-metabolomics", "Proteomics and Metabolomics",
          "Mass spectrometry, quantification, and identification confidence."),
         ("07_multi-omics-integration", "Multi-Omics Integration",
          "Early, intermediate, and late fusion; when integration helps."),
         ("08_biomarker-discovery", "Biomarker Discovery",
          "Discovery, qualification, validation, and the replication crisis."),
         ("09_drug-discovery", "Drug Discovery",
          "Target identification, screening, ADMET, and AI's actual role."),
         ("10_clinical-genomics", "Clinical Genomics",
          "ACMG classification, reporting, and genetic counseling interfaces."),
     ]),
    ("13", "13_biopharma", "Biopharma and Clinical Development",
     "13_biopharma_clinical_development.md",
     "05, 12", "25", "Translational data plan (Week 21)",
     [
         ("01_development-funnel", "The Development Funnel",
          "Discovery to Phase IV, decision gates, cost, and attrition."),
         ("02_data-in-pharma", "Data in Pharma",
          "Preclinical, clinical, safety, and commercial data domains."),
         ("03_biomarker-strategy", "Biomarker Strategy",
          "Prognostic, predictive, pharmacodynamic, and surrogate endpoints."),
         ("04_feasibility-and-recruitment", "Feasibility and Recruitment",
          "Site selection, enrollment modelling, and diversity planning."),
         ("05_cdisc-and-trial-standards", "CDISC and Trial Standards",
          "CDASH, SDTM, ADaM, define.xml, and submission-ready data."),
         ("06_safety-and-pharmacovigilance", "Safety and Pharmacovigilance",
          "AE coding, signal detection, FAERS/AEMS, and aggregate reporting."),
         ("07_rwe-in-pharma", "RWE in Pharma",
          "External controls, label expansion, and payer evidence."),
         ("08_ai-in-discovery-and-development", "AI in Discovery and Development",
          "Where models are used, validated, and rejected."),
         ("09_decentralized-trials", "Decentralised Trials",
          "Remote consent, ePRO, wearables, and the data integrity questions."),
         ("10_submission-data", "Submission Data",
          "eCTD structure, datasets, and what reviewers actually check."),
     ]),
    ("14", "14_devices_regulation", "Medical Devices, SaMD, and Regulation",
     "14_medical_devices_samd_regulation.md",
     "00, 01", "14A, 14B, 18, 25", "Regulatory evidence package (Week 22)",
     [
         ("01_definitions-and-classification", "Definitions and Classification",
          "Device, SaMD, SiMD, IVD, accessory; risk classes in the US and EU."),
         ("02_is-it-a-device", "Is It a Device?",
          "The decision tree, clinical decision support exclusions, and why vendors get this wrong."),
         ("03_fda-pathways", "FDA Pathways",
          "510(k), De Novo, PMA, and the evidence each demands."),
         ("04_gmlp-and-pccp", "GMLP and PCCP",
          "Good machine learning practice and predetermined change control plans."),
         ("05_qms-and-design-controls", "QMS and Design Controls",
          "ISO 13485, QMSR, design history file, and document control."),
         ("06_iso-14971-risk-management", "ISO 14971 Risk Management",
          "Hazard analysis, risk estimation, controls, and residual risk."),
         ("07_iec-62304-software-lifecycle", "IEC 62304 Software Lifecycle",
          "Safety classes, software requirements, architecture, and verification."),
         ("08_eu-mdr-and-ivdr", "EU MDR and IVDR",
          "Classification, conformity assessment, notified bodies, and post-market obligations."),
         ("09_the-evidence-package", "The Evidence Package",
          "What goes in a submission and how the pieces support the claim."),
         ("10_post-market-surveillance", "Post-Market Surveillance",
          "Complaints, MDR reporting, PSUR, and real-world performance monitoring."),
     ]),
    ("14A", "14A_hardware", "Medical Hardware, Sensors, Chips, and Robotics",
     "14A_medical_hardware_sensors_chips_robotics.md",
     "01A", "14B, 24", "Device data and evidence plan (Week 24)",
     [
         ("01_the-signal-chain", "The Signal Chain",
          "Sensor to ADC to processor to display; where information is lost."),
         ("02_sensors", "Sensors",
          "Electrodes, optical, pressure, flow, chemical, and MEMS sensing principles."),
         ("03_analog-front-ends", "Analog Front Ends",
          "Amplification, filtering, common-mode rejection, and noise budgets."),
         ("04_adcs-and-signal-conditioning", "ADCs and Signal Conditioning",
          "Resolution, sample rate, aliasing, and quantization error."),
         ("05_mcus-dsps-and-fpgas", "MCUs, DSPs, and FPGAs",
          "Architecture choices, determinism, and real-time constraints."),
         ("06_socs-npus-and-gpus", "SoCs, NPUs, and GPUs",
          "Heterogeneous compute for on-device inference."),
         ("07_power-and-thermal", "Power and Thermal",
          "Battery budget, leakage, thermal throttling, and patient safety."),
         ("08_firmware-and-rtos", "Firmware and RTOS",
          "Scheduling, watchdogs, secure boot, and update mechanisms."),
         ("09_medical-equipment-classes", "Medical Equipment Classes",
          "Monitoring, imaging, therapeutic, laboratory, and implantable systems."),
         ("10_robotics-kinematics", "Robotics Kinematics",
          "Frames, forward and inverse kinematics, Jacobians, and singularities."),
         ("11_robotics-control", "Robotics Control",
          "PID, impedance, admittance, force control, and safety envelopes."),
         ("12_surgical-and-rehab-robotics", "Surgical and Rehabilitation Robotics",
          "Teleoperation, haptics, registration, and assist-as-needed control."),
         ("13_standards-and-emc", "Standards and EMC",
          "IEC 60601-1, collateral standards, 60601-1-2 EMC, and 80601-2-77/-2-78."),
         ("14_reliability-and-sterilization", "Reliability and Sterilization",
          "MTBF, accelerated life testing, biocompatibility, and reprocessing."),
     ]),
    ("14B", "14B_efficient_ai", "Efficient AI and On-Device Inference",
     "14B_efficient_ai_distillation_quantization_on_device_inference.md",
     "07, 14A", "15, 24", "Efficient-model dossier (Week 25)",
     [
         ("01_the-deployment-pareto-frontier", "The Deployment Pareto Frontier",
          "Accuracy, calibration, latency, memory, energy, and cost as joint constraints."),
         ("02_target-hardware-and-performance", "Target Hardware and Performance",
          "Profiling, roofline reasoning, and what each accelerator can actually do."),
         ("03_the-compression-family", "The Compression Family",
          "Distillation, quantization, pruning, low-rank, and when to use each."),
         ("04_knowledge-distillation", "Knowledge Distillation",
          "Soft targets, temperature, feature matching, and distillation failures."),
         ("05_quantization", "Quantization",
          "Affine and symmetric schemes, PTQ, QAT, calibration, and STE."),
         ("06_pruning-and-sparsity", "Pruning and Sparsity",
          "Structured versus unstructured, lottery tickets, and hardware support."),
         ("07_inference-optimization", "Inference Optimization",
          "Graph fusion, operator fallback, KV cache, and batching."),
         ("08_on-device-architecture", "On-Device Architecture",
          "Runtimes, model formats, and the deployment pipeline."),
         ("09_medical-and-regulatory-constraints", "Medical and Regulatory Constraints",
          "Change control, PCCP, and what compression does to a cleared claim."),
         ("10_validation-on-target-hardware", "Validation on Target Hardware",
          "p99 latency, energy per inference, thermal throttling, and HIL testing."),
         ("11_economics-of-compression", "Economics of Compression",
          "Cloud cost, device BOM, battery life, and the break-even calculation."),
     ]),
    ("15", "15_mlops", "Deployment and MLOps",
     "15_deployment_mlops.md",
     "03, 06", "26", "Deployment and monitoring plan (Week 26)",
     [
         ("01_the-production-loop", "The Production Loop",
          "Train, validate, deploy, monitor, retrain, retire; where each stage fails."),
         ("02_deployment-patterns", "Deployment Patterns",
          "Batch, real-time, streaming, embedded, and shadow deployment."),
         ("03_packaging-and-reproducibility", "Packaging and Reproducibility",
          "Containers, lockfiles, model artifacts, and environment drift."),
         ("04_ci-cd-ct", "CI/CD/CT",
          "Continuous integration, delivery, and training with clinical gates."),
         ("05_monitoring", "Monitoring",
          "Service health, data quality, model performance, and outcome monitoring."),
         ("06_drift", "Drift",
          "Covariate, label, concept, and population drift; detection and response."),
         ("07_incident-response", "Incident Response",
          "Severity, triage, rollback, communication, and post-incident review."),
         ("08_rollout-and-rollback", "Rollout and Rollback",
          "Canary, shadow, blue-green, feature flags, and safe reversal."),
         ("09_model-governance", "Model Governance",
          "Inventory, ownership, approvals, change control, and audit evidence."),
         ("10_cost-and-capacity", "Cost and Capacity",
          "Serving economics, autoscaling, quotas, and capacity planning."),
     ]),
    ("16", "16_privacy_security", "Privacy, Security, and Ethics",
     "16_privacy_security_ethics.md",
     "02, 03", "17, 18", "Privacy and ethics impact assessment (Week 27)",
     [
         ("01_privacy-principles", "Privacy Principles",
          "Purpose limitation, minimization, consent, and the clinical exceptions."),
         ("02_deidentification-and-anonymization", "De-identification and Anonymization",
          "HIPAA Safe Harbor, expert determination, GDPR pseudonymization, and residual risk."),
         ("03_privacy-enhancing-technologies", "Privacy-Enhancing Technologies",
          "Differential privacy, federated learning, secure computation, and synthetic data."),
         ("04_threat-modeling", "Threat Modeling",
          "STRIDE, attack trees, insider risk, and the clinical threat surface."),
         ("05_access-control-and-cryptography", "Access Control and Cryptography",
          "RBAC, ABAC, key management, and encryption in practice."),
         ("06_fairness", "Fairness",
          "Definitions, measurement, mitigation, and the limits of each."),
         ("07_ethics", "Ethics",
          "Consent, equity, automation bias, and the ethics of deployment."),
         ("08_governance", "Governance",
          "Committees, data trusts, model review boards, and accountability."),
         ("09_breach-and-incident", "Breach and Incident",
          "Detection, notification, containment, and lessons."),
         ("10_regulatory-map", "Regulatory Map",
          "HIPAA, GDPR, EU AI Act, state law, and sector-specific rules."),
     ]),
    ("17", "17_product_economics", "Product, Economics, and Leadership",
     "17_product_economics_leadership.md",
     "08", "19, 25", "Product value case (Week 28)",
     [
         ("01_product-discovery", "Product Discovery",
          "Problem interviews, workflow observation, and finding a decision worth changing."),
         ("02_value-quantification", "Value Quantification",
          "Outcome value, cost avoided, revenue, and the assumptions behind each."),
         ("03_health-economics", "Health Economics",
          "Cost-effectiveness, ICER, QALYs, budget impact, and discounting."),
         ("04_reimbursement", "Reimbursement",
          "CPT, HCPCS, DRG, coverage decisions, and payment pathways for software."),
         ("05_procurement", "Procurement",
          "RFP, security review, contracting, and how health systems actually buy."),
         ("06_adoption", "Adoption",
          "Workflow fit, trust, training, and the predictors of sustained use."),
         ("07_business-models", "Business Models",
          "SaaS, per-seat, per-test, risk-sharing, and device-plus-service."),
         ("08_leadership", "Leadership",
          "Influence without authority, decision memos, and running a review."),
         ("09_roadmap-and-prioritisation", "Roadmap and Prioritisation",
          "Sequencing evidence, engineering, and commercial work under uncertainty."),
     ]),
    ("18", "18_quality", "Quality, Evaluation, and Reporting",
     "18_quality_evaluation_reporting.md",
     "05, 06", "19", "Evidence dossier (Week 29)",
     [
         ("01_the-evidence-chain", "The Evidence Chain",
          "Claim to endpoint to dataset to analysis to report; where chains break."),
         ("02_reporting-guidelines", "Reporting Guidelines",
          "TRIPOD+AI, CONSORT-AI, STARD-AI, SPIRIT-AI, DECIDE-AI, and PROBAST."),
         ("03_appraising-clinical-ai-evidence", "Appraising Clinical AI Evidence",
          "A structured appraisal worksheet with worked examples."),
         ("04_model-cards-and-datasheets", "Model Cards and Datasheets",
          "What to document and what readers actually need."),
         ("05_reproducibility", "Reproducibility",
          "Code, data, environment, and the difference between repeatable and reproducible."),
         ("06_quality-management", "Quality Management",
          "QMS basics for analytics teams, SOPs, and CAPA."),
         ("07_post-market-evaluation", "Post-Market Evaluation",
          "Performance monitoring, drift, and real-world evidence generation."),
         ("08_audit", "Audit",
          "Preparing for an audit and what auditors ask for."),
         ("09_documentation", "Documentation",
          "Writing documents that survive staff turnover and inspections."),
     ]),
    ("19", "19_capstone", "Capstone, Hackathons, and Competitions",
     "19_capstone_hackathons.md",
     "all core modules", "20", "Capstone (Week 40)",
     [
         ("01_capstone-architecture", "Capstone Architecture",
          "Scope, team, data, evidence, and the twelve-week shape."),
         ("02_milestone-plan", "Milestone Plan",
          "Week-by-week gates, kill criteria, and deliverable definitions."),
         ("03_team-roles", "Team Roles",
          "Clinical, data, engineering, regulatory, and communication roles."),
         ("04_capstone-archetypes", "Capstone Archetypes",
          "Eight project shapes with different evidence burdens."),
         ("05_brief-catalogue", "Brief Catalogue",
          "The full set of capstone briefs with constraints and data options."),
         ("06_hackathon-formats", "Hackathon Formats",
          "How to run and compete in a 24-72 hour event without producing nonsense."),
         ("07_competition-landscape", "Competition Landscape 2025-2026",
          "Public challenges, their datasets, and what winning solutions share."),
         ("08_scoping-playbook", "Scoping Playbook",
          "Turning an open problem into a defensible scope in two days."),
         ("09_deliverables", "Deliverables",
          "Dossier, prototype, demo, and defense requirements."),
         ("10_judging", "Judging",
          "Rubric, calibration, and the failure modes of judges."),
         ("11_demo-day", "Demo Day",
          "Executive pitch structure and the questions you will be asked."),
     ]),
    ("20", "20_career", "Career, Portfolio, and Interviews",
     "20_career_portfolio_interviews.md",
     "19", "none", "Portfolio review",
     [
         ("01_role-map", "Role Map",
          "Titles, levels, and what each role is actually accountable for."),
         ("02_portfolio", "Portfolio",
          "Five-project portfolio structure and the evidence each project needs."),
         ("03_resume-and-linkedin", "Resume and LinkedIn",
          "Translating regulated work into credible, specific claims."),
         ("04_interview-types", "Interview Types",
          "Screen, technical, case, system design, and panel formats."),
         ("05_technical-interviews", "Technical Interviews",
          "SQL, statistics, ML, and coding under time pressure."),
         ("06_clinical-case-interviews", "Clinical Case Interviews",
          "Structuring a clinical analytics case out loud."),
         ("07_system-design", "System Design",
          "Designing a clinical data product in 45 minutes."),
         ("08_behavioural-interviews", "Behavioural Interviews",
          "STAR structures for safety, conflict, and failure questions."),
         ("09_negotiation", "Negotiation",
          "Level, scope, equity, and the questions to ask before accepting."),
         ("10_onboarding", "Onboarding",
          "A 90-day plan for a new biomedical data role."),
     ]),
    ("21", "21_datasets", "Datasets, Tools, and Benchmarks",
     "21_datasets_tools_benchmarks.md",
     "02", "19", "Dataset and benchmark dossier (Week 31)",
     [
         ("01_five-gate-decision", "The Five-Gate Dataset Decision",
          "Access, licence, fitness, bias, and reproducibility gates."),
         ("02_access-and-licensing", "Access and Licensing",
          "Credentialing, DUAs, BAAs, and commercial restrictions."),
         ("03_ehr-claims-and-icu", "EHR, Claims, and ICU Data",
          "MIMIC, eICU, All of Us, NHANES, Marketscan-like sources, and their limits."),
         ("04_imaging-datasets", "Imaging Datasets",
          "Public and credentialed imaging collections and their label quality."),
         ("05_signal-datasets", "Signal Datasets",
          "PhysioNet collections, wearable corpora, and annotation depth."),
         ("06_omics-datasets", "Omics Datasets",
          "TCGA, GTEx, single-cell atlases, and access terms."),
         ("07_pathology-datasets", "Pathology Datasets",
          "Whole-slide collections and the annotation bottleneck."),
         ("08_text-and-llm-benchmarks", "Text and LLM Benchmarks",
          "Clinical QA sets, summarization sets, and contamination risk."),
         ("09_public-health-datasets", "Public Health Datasets",
          "Surveillance, vital statistics, and geographic data."),
         ("10_pharma-and-safety-datasets", "Pharma and Safety Datasets",
          "Trial data sharing, FAERS/AEMS, and pharmacovigilance corpora."),
         ("11_tool-stack", "Tool Stack",
          "What to use for each function and what to avoid."),
         ("12_benchmark-pitfalls", "Benchmark Pitfalls",
          "Leaderboard chasing, contamination, and non-comparable splits."),
     ]),
    ("22", "22_sources", "Sources and Further Reading",
     "22_sources_and_further_reading.md",
     "none", "none", "Annotated bibliography",
     [
         ("01_how-to-read-this-list", "How to Read This List",
          "Tiering, recency, and how to judge a source."),
         ("02_supplied-clinical-library", "Supplied Clinical Library",
          "The books in this package and what each is good for."),
         ("03_regulation-and-policy", "Regulation and Policy",
          "Primary regulatory sources and how to track changes."),
         ("04_methods", "Methods",
          "Statistics, causal inference, ML, and evaluation references."),
         ("05_clinical-background", "Clinical Background",
          "Texts and references for clinical vocabulary."),
         ("06_datasets-and-benchmarks", "Datasets and Benchmarks",
          "Where to find data and how to cite it."),
         ("07_industry-and-market", "Industry and Market",
          "Market research, company filings, and credible industry reporting."),
         ("08_communities-and-continuing-education", "Communities and Continuing Education",
          "Societies, conferences, and ongoing learning."),
         ("09_keeping-current", "Keeping Current",
          "A maintenance routine for a fast-moving field."),
     ]),
    ("23", "23_specialty", "Specialty Electives",
     "23_specialty_electives.md",
     "01, 08", "19", "Elective primer assessment",
     [
         ("01_oncology", "Oncology",
          "Tumour biology, staging, RECIST, endpoints, and oncology data."),
         ("02_cardiology", "Cardiology",
          "Heart failure, ACS, arrhythmia, devices, and cardiology registries."),
         ("03_pathology", "Pathology",
          "Histology, whole-slide imaging, molecular pathology, and reporting."),
         ("04_critical-care", "Critical Care",
          "ICU physiology, ventilation, sepsis, and high-frequency data."),
         ("05_pediatrics-and-obstetrics", "Pediatrics and Obstetrics",
          "Growth, dosing, maternal-fetal data, and special consent."),
         ("06_public-health", "Public Health",
          "Population surveillance, equity, and policy analytics."),
         ("07_cross-specialty-comparison", "Cross-Specialty Comparison",
          "What transfers between specialties and what does not."),
     ]),
    ("24", "24_hardware_labs", "Hardware and Robotics Labs",
     "24_hardware_robotics_labs.md",
     "14A, 14B", "none", "Graded evidence dossier",
     [
         ("01_safety-and-standards", "Safety and Standards",
          "Lab safety, electrical safety, and the compliance boundary."),
         ("02_bench-and-simulation-stack", "Bench and Simulation Stack",
          "Instruments, dev boards, simulators, and software."),
         ("03_lab-conventions", "Lab Conventions",
          "Reproducibility, logging, calibration, and reporting."),
         ("04_sensor-and-signal-labs", "Sensor and Signal Labs",
          "ECG front end, PPG, temperature, and noise measurement."),
         ("05_firmware-labs", "Firmware Labs",
          "RTOS scheduling, watchdogs, secure boot, and update."),
         ("06_emc-labs", "EMC Labs",
          "Emissions, immunity, and pre-compliance testing."),
         ("07_hil-labs", "Hardware-in-the-Loop Labs",
          "Closed-loop testing, fault injection, and latency measurement."),
         ("08_robotics-labs", "Robotics Labs",
          "Kinematics, control, registration, and safety envelopes."),
         ("09_evidence-dossier", "Evidence Dossier",
          "Turning lab results into a defensible evidence package."),
     ]),
    ("25", "25_trial_ops", "Trial Operations, HTA, and Global Regulation",
     "25_trial_operations_hta_global_regulatory.md",
     "13, 14", "19", "Market access assessment",
     [
         ("01_trial-operations", "Trial Operations",
          "Start-up, site activation, enrollment, and close-out."),
         ("02_site-and-monitoring", "Site and Monitoring",
          "Risk-based monitoring, source data review, and query management."),
         ("03_clinical-data-management", "Clinical Data Management",
          "CRF design, edit checks, medical coding, and database lock."),
         ("04_health-technology-assessment", "Health Technology Assessment",
          "Evidence requirements, economic models, and submission dossiers."),
         ("05_market-access", "Market Access",
          "Pricing, reimbursement, and payer negotiation."),
         ("06_global-regulation", "Global Regulation",
          "EMA, MHRA, PMDA, NMPA, Health Canada, and TGA differences."),
         ("07_pricing-and-contracting", "Pricing and Contracting",
          "Value-based agreements and risk-sharing arrangements."),
     ]),
    ("26", "26_cloud", "Cloud, Kubernetes, and Streaming Production Labs",
     "26_cloud_streaming_production_labs.md",
     "03, 15", "none", "Production readiness review",
     [
         ("01_architecture-patterns", "Architecture Patterns",
          "Reference architectures for clinical analytics and streaming."),
         ("02_containers", "Containers",
          "Images, layers, security scanning, and reproducible builds."),
         ("03_kubernetes", "Kubernetes",
          "Deployments, services, probes, autoscaling, and network policy."),
         ("04_infrastructure-as-code", "Infrastructure as Code",
          "Terraform modules, state, and environment promotion."),
         ("05_streaming", "Streaming",
          "Kafka semantics, schemas, idempotence, deduplication, and DLQs."),
         ("06_observability", "Observability",
          "Metrics, logs, traces, and clinical service SLOs."),
         ("07_security", "Security",
          "Secrets, identity, network segmentation, and supply chain."),
         ("08_cost-and-reliability", "Cost and Reliability",
          "SLOs, error budgets, multi-AZ, and cost control."),
         ("09_reference-stack-walkthrough", "Reference Stack Walkthrough",
          "End-to-end deployment of the lab stack with acceptance criteria."),
     ]),
]


def topic_status(folder: str, slug: str) -> tuple[str, int]:
    path = os.path.join(MODULES_DIR, folder, slug + ".md")
    if not os.path.exists(path):
        return "planned", 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.readlines()
    n = len(content)
    sections = sum(1 for line in content if line.startswith("## "))
    if n >= 1000 and sections >= 17:
        return "complete", n
    return f"draft ({n} lines)", n


def index_text(module) -> str:
    mid, folder, title, source, prereq, unlocks, assessment, topics = module
    lines = []
    a = lines.append
    a(f"# Module {mid} — {title}")
    a("")
    a("> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.")
    a(f"> **Survey file:** [`../../{source}`](../../{source})")
    a(f"> **Folder:** `modules/{folder}/`")
    a("> **Version:** 2026.3")
    a("")
    a("---")
    a("")
    a("## 1. Module purpose")
    a("")
    a(f"The survey file gives the module's narrative arc in one pass. This folder breaks that arc into")
    a("teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked")
    a("example, code or calculation, failure modes, assumptions, and exercises.")
    a("")
    a("Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,")
    a("not the whole program.")
    a("")
    a("## 2. Position in the program")
    a("")
    a("| Item | Value |")
    a("|---|---|")
    a(f"| Prerequisites | {prereq} |")
    a(f"| Unlocks | {unlocks} |")
    a(f"| Assessment | {assessment} |")
    a(f"| Survey file | `{source}` |")
    a("")
    a("## 3. Topics")
    a("")
    a("| # | File | Topic | Scope | Target | Status |")
    a("|---:|---|---|---|---:|---|")
    counts = {"planned": 0, "draft": 0, "complete": 0}
    for i, (slug, ttitle, scope) in enumerate(topics, start=1):
        st, n = topic_status(folder, slug)
        key = "planned" if st == "planned" else ("complete" if st == "complete" else "draft")
        counts[key] += 1
        label = st if n == 0 else f"{st}"
        a(f"| {i} | [`{slug}.md`]({slug}.md) | {ttitle} | {scope} | 1,200-1,800 | {label} |")
    a("")
    a("## 4. How the topics fit together")
    a("")
    a("Each topic is written to stand alone, but they are ordered so that later topics can assume the")
    a("vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus")
    a("any topic that names your current decision problem.")
    a("")
    a("```text")
    for i, (slug, ttitle, _) in enumerate(topics, start=1):
        arrow = "" if i == len(topics) else " ->"
        a(f"{i:02d} {ttitle}{arrow}")
    a("```")
    a("")
    a("## 5. What this module does not cover")
    a("")
    a("- Hands-on lab execution: see `labs/` and the lab sections of the survey file.")
    a("- Legal advice: regulatory content is educational and requires professional review.")
    a("- Clinical validation: no topic file here validates a product or a model.")
    a("")
    a("## 6. Status")
    a("")
    a("| Status | Count |")
    a("|---|---:|")
    a(f"| planned | {counts['planned']} |")
    a(f"| draft | {counts['draft']} |")
    a(f"| complete | {counts['complete']} |")
    a("")
    a("Regenerate this table with `python tools/scaffold_modules.py --status`.")
    a("")
    a("## 7. Sources for this module")
    a("")
    a(f"See section 17 of the survey file and the per-topic source sections.")
    a("")
    return "\n".join(lines) + "\n"


def status_text() -> str:
    rows = []
    total_files = 0
    total_lines = 0
    for mid, folder, title, source, prereq, unlocks, assessment, topics in MANIFEST:
        folder_path = os.path.join(MODULES_DIR, folder)
        present = 0
        lines_total = 0
        complete = 0
        for slug, _, _ in topics:
            st, n = topic_status(folder, slug)
            if n > 0:
                present += 1
                lines_total += n
            if st == "complete":
                complete += 1
        total_files += present
        total_lines += lines_total
        rows.append((mid, title, len(topics), present, complete, lines_total))

    out = []
    a = out.append
    a("# Module Expansion Progress")
    a("")
    a("> Generated from folder contents. Do not edit by hand.")
    a("")
    a("| Module | Title | Topics planned | Files present | Files >= 1,000 lines | Lines |")
    a("|---|---|---:|---:|---:|---:|")
    for mid, title, planned, present, complete, lines_total in rows:
        a(f"| {mid} | {title} | {planned} | {present} | {complete} | {lines_total:,} |")
    a("")
    a(f"**Total topic files:** {total_files}")
    a("")
    a(f"**Total topic lines:** {total_lines:,}")
    a("")
    return "\n".join(out) + "\n"


def main() -> int:
    os.makedirs(MODULES_DIR, exist_ok=True)
    written = 0
    for module in MANIFEST:
        folder = module[1]
        path = os.path.join(MODULES_DIR, folder)
        os.makedirs(path, exist_ok=True)
        target = os.path.join(path, "_index.md")
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(index_text(module))
        written += 1

    with open(os.path.join(MODULES_DIR, "PROGRESS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(status_text())

    manifest_path = os.path.join(ROOT, "modules", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            [
                {
                    "id": m[0],
                    "folder": m[1],
                    "title": m[2],
                    "source": m[3],
                    "topics": [{"order": i + 1, "slug": s, "title": t, "scope": sc}
                               for i, (s, t, sc) in enumerate(m[7])],
                }
                for m in MANIFEST
            ],
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(f"wrote {written} module indexes")
    print(f"total planned topics: {sum(len(m[7]) for m in MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
