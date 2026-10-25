# Research Brief — 2025–2026 Biomedical Data Science Industry Landscape

**Cutoff:** 24 October 2026. **[F]** sourced fact; **[I]** implication; **[?]** single-source or unverified. Re-verify before publication.

## 1. Providers and health systems

- Hospital margins are thin: Kaufman Hall reported a CYTD adjusted operating margin near 1.3% for 2025, with expense growth outpacing revenue (drugs +13.6%, supplies +9.9%). [F]
- Revenue-cycle leakage is a hard-dollar AI target: median final denial rates rose from 2.5% to 2.7% (2024→2025); net revenue leakage grew from $38.6B to $48.4B; AHA estimates hospitals spent ~$18B overturning denials in 2025. [F]
- Ambient documentation is the most adopted clinical AI (~79% of organizations in KLAS 2025), but adoption is broad and shallow; agentic AI is rare. [F]
- KLAS 2026: AI users report better EHR efficiency, but gains plateau after ~4 tools; fewer than 25% received adequate training. [F]
- Epic Cosmos covers ~2,358 hospitals and ~310M de-identified patients. [F]
- KPIs: ALOS, 30-day readmissions, ED boarding/LOS, OR first-case on-time, denial rate, clean-claim rate, days in A/R, case-mix index, sepsis bundle compliance, cost per adjusted discharge, HCAHPS.
- Vendors/products: Epic Cosmos, Oracle Health, Abridge, Nuance DAX Copilot, Nabla, Ambience, Suki, LeanTaaS, Qventus, Aidoc, Viz.ai.

## 2. Pharma and biotech

- Phase 3 success ~50–53%; overall probability of success from Phase 1 ~5–14%; median Phase III study cost ~$48M. [F]
- By mid-2025, >75 AI-designed molecules had entered human trials (none in 2020). [F]
- Insilico Medicine rentosertib (TNIK inhibitor, IPF) Phase IIa in *Nature Medicine*: 71 patients; FVC +98.4 mL at 60 mg QD vs −20.3 mL placebo; safety primary endpoint met. No AI-discovered drug approved yet. [F]
- FDA CDER/CBER received >1,000 submissions with AI elements since 2016; 248 INDs in 2024, concentrated in oncology. [F]
- KPIs: cycle time, probability of success, enrollment, screen-failure rate, query rate, time to database lock, cost per patient, submission-cycle time.

## 3. Clinical trials and CROs

- Medidata 2025: 56% organizational AI adoption; 73% of users say AI met/exceeded expectations; 93% use for data capture/quality; 83% for cohort identification; auto-coding saves up to 69 hours per 1,000 verbatims. [F]
- WCG 2025: implementation at 52% of sponsors, 57% of CROs, 39% of sites; barriers: ethics 59%, infrastructure 56%, funding 52%. [F]
- Vendors: IQVIA, ICON, Parexel, Medpace, Medidata, Veeva, Unlearn.

## 4. Medical devices

- FDA authorized >1,600 AI-enabled devices as of September 2026; ~75% radiology. [F]
- 1,356 through September 2025. [F]
- Aidoc CARE1 rib-fracture triage (Feb 2025) described as first FDA-cleared foundation-model-based device. [F/?]
- HeartFlow IPO Aug 2025 at ~$2.27B valuation; $125.8M 2024 revenue from FFR-CT. [F]
- Among 950 cleared AI devices, 6.3% had recalls; 43.4% of recalls within 12 months of clearance. Diagnostic/measurement errors were the most common cause (109 of 182 recalls). [F]
- Data: sensor streams, imaging, device logs, MAUDE, EHR. KPIs: sensitivity/specificity/PPV, alert burden, time-to-diagnosis, recall rate, cybersecurity.

## 5. Wearables, sensors, digital health

- Apple hypertension notification cleared 11 Sep 2025 (K250507): passive PPG over 30-day windows; no BP reading; sensitivity 41.2% overall (53.7% Stage 2), specificity 92.3%, n=2,229. [F]
- Sleep-apnea notification cleared 2024. [F]
- US digital health funding $14.2B in 2025 (+35%); AI-enabled companies 54% of dollars; ambient scribes ~$600M market. [F]
- Data: PPG, ECG, accelerometry, CGM, sleep, SpO2, GPS, PROMs. KPIs: engagement/retention, MAE, false-alert rate, RPM reimbursement, escalation.
- Failure mode: performance varies with skin tone, BMI, motion, perfusion.

## 6. Diagnostics

- Tempus FY2025 revenue $1.27B (+83%): diagnostics $955M, data/applications $316M; oncology reimbursement ~$1,600/test. [F]
- GRAIL NHS-Galleri (~140,000 participants) missed primary endpoint (stage III/IV incidence IRR 1.03, 95% CI 0.92–1.14); stage IV fell 14%, stage I/II rose 16%; NHS England did not proceed to pilot. [F; publication venue verify]
- Vendors: Tempus, Guardant, GRAIL, Paige, PathAI, Exact Sciences.

## 7. Payers

- Medicare Advantage covered 34.1M people (54% of eligible) in 2025; UnitedHealthcare 29%, Humana 17%. [F]
- CMS-0057-F requires payer APIs for prior authorization by January 2026 and full compliance by January 2027; 72-hour urgent and 7-day standard decisions; public reporting from March 2026. [F/?]
- *Lokken v. UnitedHealth*: contract claims survived dismissal Feb 2025; bifurcation of discovery denied Sep 2025. [F]
- Data: claims/encounters, pharmacy, labs, HEDIS, RAF. KPIs: medical loss ratio, Star rating, RAF, denial/overturn rate, PA turnaround, readmission.

## 8. Public health

- CDC FY2026–2030 AI strategy: adoption, governance/trust, enterprise data platforms, workforce. [F]
- 1,500+ wastewater sites; 86% of EDs reporting; 60,000+ facilities sending electronic case reports; TowerScout Legionella detection 4 hours → 5 minutes. [F]
- Constraints: fragmented state/local governance and data-use agreements.

## 9. Roles and skills

- Titles: Clinical/Health Data Scientist; Biomedical Data Scientist; RWE Scientist; HEOR/Outcomes Scientist; Bioinformatics/Computational Biology Scientist; ML Engineer, Clinical AI; Imaging Scientist; Biostatistician; Clinical Data Manager; Pharmacovigilance Data Scientist; Regulatory Data/AI Specialist; Health Informatics Analyst; AI Product Manager.
- BLS data scientists median $120,230 (May 2025), p10 $67,240, p90 $199,130; 35% projected growth 2025–2035. [F]
- 2026 health-sector postings median $108,400; p75 $154,600; p90 $205,000. [F/?]
- Representative: Bayer biomarker data scientist $104,480–$156,720; AbbVie trial surveillance $109,500–$208,500; N-Power Medicine $132,000–$226,000; Viz.ai $129,000–$145,000. [F]
- Core: SQL and Python dominate; R third; Git; reproducible pipelines; causal inference, survival, mixed models, Bayesian, high-dimensional data; gradient boosting, deep learning, transformers, calibration, evaluation; Spark, dbt, Airflow, Databricks/Snowflake, Docker/Kubernetes, MLflow, cloud.
- Domain: clinical workflows, ICD-10/CPT/LOINC/RxNorm/SNOMED, FHIR/HL7 v2, OMOP, DICOM, CDISC SDTM/ADaM, GxP/Part 11, HIPAA/GDPR, study design, regulatory, HEOR.
- Tools: Medidata Rave, REDCap, Veeva, SAS, Spotfire, Tableau/Power BI, PyTorch, Hugging Face, MONAI, SHAP.
- Soft: clinician communication, stakeholder management, cross-functional coordination, regulatory writing, documentation, ambiguity, ethics.
- LinkedIn Jobs on the Rise 2026: AI engineer #1, AI consultant #2, data annotator #4, AI/ML researcher #5, healthcare reimbursement specialist #6. [F]
- Skills on the Rise 2026: RAG, vector databases, prompt engineering, LangChain, fine-tuning, XGBoost, data governance, responsible AI. [F]

## 10. Regulation — United States

- FDA regulates AI-enabled devices as software functions; >1,600 authorized as of Sep 2026; ~75% radiology. [F]
- GMLP: 10 guiding principles, Oct 2021, FDA/Health Canada/MHRA. [F]
- PCCP final guidance: 4 Dec 2024, reissued 18 Aug 2025; Description of Modifications, Modification Protocol, Impact Assessment; authorized modifications need no new submission. [F]
- AI device lifecycle draft: 7 Jan 2025. [F]
- AI in drug development draft: Jan 2025; risk-based seven-step credibility assessment; Jan 2026 ten guiding principles. [F]
- Generative AI devices: discussion paper 18 Aug 2026; two-axis risk framework (directive function × severity); competency assessment; risk-proportionate postmarket monitoring; comments closed 19 Oct 2026, docket FDA-2026-N-7874. [F]
- QMSR effective 2 Feb 2026; incorporates ISO 13485:2016. [F]
- Computer Software Assurance final guidance Sep 2025, revised Feb 2026. [F]
- LDT rule vacated 31 Mar 2025 (*ACLA v. FDA*); FDA rescinded 19 Sep 2025; LDTs under CLIA/CMS. [F]
- CDS final guidance dated 29 Jan 2026 listed; text not independently verified. [?]

## 11. Regulation — European Union

- AI Act in force Aug 2024; GPAI obligations 2 Aug 2025; transparency 2 Aug 2026. Digital Omnibus on AI (Regulation (EU) 2026/1744) postponed stand-alone Annex III high-risk to 2 Dec 2027 and Annex I products, including medical devices, to 2 Aug 2028. [F]
- MDR/IVDR apply in parallel. MDCG 2025-6 (June 2025): AI Act high-risk status follows MDR/IVDR class and notified-body involvement; documentation can be integrated into technical file. [F]
- EHDS Regulation (EU) 2025/327 in force 26 Mar 2025; governance 26 Mar 2027; secondary use 26 Mar 2029; omics and trial data 26 Mar 2031. [F]

## 12. US privacy, interoperability, payment

- HIPAA Security Rule NPRM 6 Jan 2025 would eliminate required/addressable, mandate encryption, MFA, asset inventories, patch timelines, annual pen testing, 72-hour critical-system restoration; final rule projected 2026 but not published as of sources reviewed. [F/?]
- HTI-1: USCDI v3 baseline 1 Jan 2026; DSI criterion requires source attributes, intended use, validation, fairness information for predictive models. [F]
- HTI-2 (Dec 2024) codified TEFCA QHIN governance. [F]
- CMS-0057-F prior authorization APIs and shorter decision windows, phased Jan 2026–Jan 2027. [F/?]
- CY2026 Physician Fee Schedule did not create a general AI payment code; solicited comments on SaaS/AI payment; −2.5% efficiency adjustment to work RVUs for non-time-based services. [F]

## 13. Technical state and failure modes

- EHR foundation models beat baselines on discrimination, especially with limited labels, but are poorly calibrated with little data and fail cross-institutional transport (FoMoH: 14 tasks, >6M patients). [F]
- Epic/Microsoft "Curiosity": scaling on 118M patients/151B tokens; relies on ICD phenotypes; lacks subpopulation calibration. [F]
- Imaging FDA evidence: 903 devices through Aug 2024; 55.9% reported clinical performance studies; 38.2% retrospective; 8.1% prospective; 2.4% randomized; 16.2% reported AUC. [F]
- Foundation models (Aidoc CARE1, MedVersa) arrive, but zero-shot generalization trails in-domain nnU-Net under shift. [F]
- PPG heart-rate error grows with dark skin tone, high BMI, motion, high heart rate; multi-wavelength study eliminated gaps only after uncertainty-guided rejection. [F]
- SpO2 bias greater in dark pigmentation; wide limits of agreement for pulse rate. [F]
- Zero-shot single-cell foundation models (scGPT, Geneformer) underperform scVI/Harmony on clustering and integration; larger pretraining did not help. [F]
- Spatial models (Nicheformer, OmiCLIP/Loki) improve niche/multimodal tasks but lack standardized benchmarks. [F]
- AI drug discovery: clinical proof-of-concept but no approval; bottleneck is target validation and translation. [F/I]
- Trials: AI adoption uneven; sites are the bottleneck. [F]
- RWE: RCT-DUPLICATE 32 emulations correlated r=0.93 with RCT results; BenchExCal proposes benchmark-then-calibrate. [F]
- HealthBench: 262 physicians, 5,000 conversations, 48,562 rubric criteria; best models still only 32% on HealthBench Hard. [F]
- Ambient scribe RCTs: Nabla −9.5% time-in-note; DAX not significant; quality gains, no hard outcome improvement. [F]
- Agentic risk: models chose adversarially altered guidelines 40.6% of the time; single injected passage succeeded in up to 61% of multi-agent cases. [F]
- Federated: 47-site ASCIN AUROC 0.76 vs 0.68 centralized for readmission; 99.1% uptime; membership-inference AUC at chance; ~$47,850 upfront and $50,400/year per site. Central DP cut AUROC ~5% and AUPRC ~35%; local DP nearly destroyed performance. [F]
- Synthetic data: process-driven PBPK/popPK/QSP accepted; data-driven synthetic controls have no approval precedent as comparator arms. [F]
- MLOps: FDA 2025 request for comment notes static benchmarks do not predict deployment; asks about drift, postmarket data, human-AI interaction. [F]
- Cybersecurity: Change Healthcare breach affected ~192.7M individuals; disrupted claims for 189,000 providers. [F]
- Fairness: 2025 review found bias in 22/24 medical LLM studies (91.7%); safety-net mitigation reduced asthma/readmission equalized-odds gaps below 5 points with threshold adjustment; high AUC in small minority subgroups can reflect access bias. [F]
- Uncertainty/human factors: only 1.4% of 705 imaging adaptation studies reported calibration/uncertainty; <25% of AI adopters report adequate training; alert fatigue and automation bias dominate. [F]

## 14. Datasets and benchmarks

| Resource | What | Access |
|---|---|---|
| MIMIC-IV v3.1 | 364,627 patients; 546,028 hospitalizations; 94,458 ICU stays; 2008–2022 | PhysioNet credentialing, CITI, DUA 1.5.0 |
| eICU-CRD | Multi-hospital ICU ~200k stays | Credentialed PhysioNet |
| PhysioNet | Signals, waveforms, clinical | Mixed |
| UK Biobank | 500k participants, genomes, imaging, EHR | Application; Tier 3 ~£9,000/3 years; UKB-RAP |
| All of Us | CDR v8: 447,278 genotyped, 414,830 WGS, 97,061 SV, 2,800 long-read | Controlled Tier; DURA; $300 credits |
| TCGA | Multi-omic cancer atlas | Open clinical/expression; controlled germline |
| Open Targets | Target–disease evidence | Open |
| ChEMBL | Bioactivity | Open CC BY-SA |
| PubChem | Chemical structures/bioassays | Open |
| Epic Cosmos | ~310M de-identified patients | Contributing systems only |
| HealthBench | 5,000 conversations, 48,562 rubrics | Open |
| EHRSHOT | Few-shot EHR benchmark | Credentialed STARR derivative |
| FoMoH | 14-task EHR foundation benchmark | Code/data links |
| MIMIC-CXR | Chest radiographs + reports | Credentialed |
| CheXpert | Chest X-ray labels | Open registration |
| BraTS | Brain tumor MRI | Open challenge |
| CELLxGENE | Single-cell atlas | Open |
| OpenFDA/AEMS | Adverse events | Open |
| CMS Open Payments | Industry payments | Open |
| HCUP | Discharge/ED data | Purchase/training |
| NHANES | Population health | Open |

Cautions: MIMIC/eICU are limited-site samples with coding/documentation artifacts; UK Biobank is healthier/older; All of Us oversamples underrepresented groups but has recruitment/consent complexity; Cosmos is contributor-only; benchmark contamination is a first-order LLM threat.

## 15. Open problems for projects

1. Prospective outcome evidence for AI devices.
2. Postmarket monitoring and drift taxonomy.
3. Evaluation science for LLMs and agents.
4. Label quality and phenotyping.
5. Equity measurement and mitigation.
6. Privacy-utility engineering.
7. Reimbursement and business models.
8. Interoperability and provenance.
9. Human factors and training.
10. Small-data and rare disease.
11. Security of agentic systems.
12. Regulatory science for continual learning.

## 16. Bottom line

The durable skill set is framing a clinical or financial decision, assembling defensible data, quantifying uncertainty and bias, producing evidence that survives regulatory and reimbursement scrutiny, and monitoring after deployment. Generic model-building is commoditizing.
