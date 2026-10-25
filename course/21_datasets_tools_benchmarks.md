# Module 21 — Datasets, Tools, and Benchmarks

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 16 hours (6 lecture, 2 reference clinic, 8 lab)
> **Track fit:** Common core; required before the capstone (Module 19)
> **Prerequisites:** Modules 02, 03, 06; basic Python and SQL
> **Tools:** Python, R, SQL, DuckDB, Hugging Face Datasets, pydicom, WFDB, Scanpy, Fairlearn, DVC
> **Industry question:** "Is this dataset allowed, adequate, and representative enough to support the claim I am about to make — and does the benchmark measure the thing I care about?"

---

## 1. Why industry cares

Every biomedical claim rests on a dataset, a license, and a benchmark. In industry these are never independent:

- A dataset that cannot be used legally in the deployment jurisdiction is worthless, however good the model.
- A benchmark that leaked into pretraining, or scores a proxy rather than the clinical outcome, produces a number that survives review but not deployment.
- A tool stack that cannot pin a version or reproduce a split turns a result into an anecdote.

The expensive failures are rarely modeling failures. They are: choosing data because it is downloadable rather than decision-relevant; signing a non-commercial license before building a product; reporting a leaderboard delta that vanishes under a patient-level split; training on a benchmark the model already saw; claiming national generalization from one ICU; and comparing numbers computed with different labels, splits, and preprocessing.

**How to use this module.** Sections 3–4 give the selection and access frameworks. Sections 5–13 are the catalog by modality. Sections 14–17 cover hubs, anti-patterns, tools, and benchmark validity. Sections 18–20 are the assessed work. Sizes, licenses, and access conditions were checked against the cited sources as of **24 October 2026**; rapidly changing resources are marked **verify** with the item to check. Never copy a size or license from here into a grant, contract, or regulatory submission without checking the primary source.

## 2. Learning outcomes

By the end of this module, you can:

- Apply a five-gate selection framework and defend the choice in writing.
- Distinguish open, registered, credentialed, controlled, licensed, contributor-only, and purchased access, and state the obligations each creates.
- Map a question to at least two candidate datasets and one fallback.
- Audit provenance, label definition, split strategy, and known biases before modeling.
- Identify license terms governing commercial use, redistribution, model training, and derivative works.
- Critique a leaderboard result for contamination, leakage, metric mismatch, and access bias.
- Reproduce a published baseline within a stated tolerance, or explain why it cannot be reproduced.
- Select a defensible stack for tabular, imaging, signal, text, omics, causal, and deployment work.
- Write a data card and a benchmark validity memo suitable for internal review.
- Recognize when no existing dataset can support the intended claim.

## 3. Mental model: the five-gate dataset decision

A dataset is admissible only if it passes all five gates. A failure at any gate is disqualifying, not a trade-off to be averaged away.

```
Gate 1  Decision fit    Does it contain the outcome, exposure, and confounders,
                        measured at the time the decision is made?
Gate 2  Population fit  Are the population, site, era, and setting close enough to deployment?
Gate 3  Access fit      Can we obtain it, at the needed granularity, before the deadline?
Gate 4  License fit     Does the license permit our use, users, locations, and outputs?
Gate 5  Benchmark fit   Does the task and metric correspond to the decision, on a clean test set?
```

| Gate | Pass evidence | Typical disqualifier |
|---|---|---|
| Decision fit | Temporal audit of every feature | Discharge diagnoses used to predict admission risk |
| Population fit | Subgroup coverage table | Single academic center, no external site |
| Access fit | Signed DUA or documented path | "We will apply later" with a nine-month review |
| License fit | Clause-by-clause mapping | Non-commercial clause in a product roadmap |
| Benchmark fit | Metric-to-cost analysis | AUROC for a fixed-capacity screening program |

**Scoring.** Score 0–2 per gate, record evidence rather than the number, and treat any zero as a veto.

| Candidate | Decision | Population | Access | License | Benchmark | Veto? |
|---|---:|---:|---:|---:|---:|---|
| MIMIC-IV | 2 | 0 | 2 | 2 | 1 | Population |
| eICU-CRD | 2 | 1 | 1 | 2 | 1 | No |
| Epic Cosmos | 2 | 2 | 0 | 1 | 1 | Access |
| National claims | 2 | 2 | 1 | 1 | 1 | No |
| Prospective site data | 2 | 2 | 0 | 2 | 2 | Access |

**Decision rules.** If the outcome is a billing concept, use claims; if it is a clinical state, use EHR or registry. If the claim is about a population, the dataset must contain that population, not a model of it. If the product is commercial, the license must permit commercial training, not only inference. If the evaluation is a public leaderboard, assume contamination until tested. If no dataset passes Gate 1, the deliverable is a data-collection plan, not a model. Prefer two imperfect independent datasets to one perfect-looking one.

## 4. Access and licensing taxonomy

| Tier | Meaning | Examples | Obligations | Latency |
|---|---|---|---|---|
| Open | Direct download, no identity | NHANES, OpenFDA, CMS Open Payments, Synthea | Attribution or none; still check terms | Minutes |
| Registered | Account and click-through | PhysioNet open projects, Kaggle, gated Hugging Face models | Accept terms; no redistribution | Hours |
| Credentialed | Identity, training, signed DUA | MIMIC-IV, eICU, MIMIC-CXR, EHRSHOT | CITI, DUA, no re-identification or redistribution | Days–weeks |
| Controlled | IRB or institutional sign-off, often fee | UK Biobank, All of Us Controlled, dbGaP, EGA | Approved analysis plan, output review | Weeks–months |
| Licensed | Commercial contract | MarketScan, Optum, TriNetX, VigiBase, COSMIC | Fee, field-of-use limits, audit rights | Weeks–months |
| Contributor-only | Access tied to membership | Epic Cosmos | Contributing organization; query-level export | Organizational |
| Purchased | Per-dataset or annual sale | HCUP NIS/SID/SEDD | Fee, training, no re-release, small-cell suppression | Days–weeks |

| License clause | What it controls | Ask before signing |
|---|---|---|
| Commercial use | Whether outputs can support a product | Does "research" exclude internal product development? |
| Redistribution | Sharing raw or derived data | Can we share weights? Features? Synthetic samples? |
| Model training | Use for training or fine-tuning | Are model weights a derivative work? |
| Publication | Preprints, figures, aggregate tables | Is pre-review required? Are small-cell counts banned? |
| Re-identification | Attempts and linkage | Does linking to another dataset count? |
| Retention | How long data may be kept | What happens to derived features at project end? |
| Jurisdiction | Where processing may occur | Does GDPR or residency law forbid the cloud region? |

**Credentialing mechanics.** PhysioNet: account, CITI "Data or Specimens Only Research", project-specific training, individual DUA; access is personal, and sharing credentials violates the DUA. dbGaP/DUOS: institutional signing official, research use statement, project-specific approval, annual renewal. UK Biobank: application, institutional approval, fee, analysis inside UKB-RAP with an approved return process. All of Us: Data User Code of Conduct, institutional login, compute in the Researcher Workbench.

**Licensing is not ethics.** A license states what the owner permits. It does not establish that the cohort consented to commercial research, that the data support the intended claim, that publishing subgroup results is safe, or that the model cannot harm the people in it. IRB review, community engagement, and a DPIA sit alongside the license.

## 5. EHR, claims, and ICU data

| Dataset | Contents | Scale | Access and license | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **MIMIC-IV v3.1** | De-identified EHR from Beth Israel Deaconess: admissions, ICU, labs, vitals, medications, procedures, diagnoses | 364,627 patients; 546,028 hospitalizations; 94,458 ICU stays; 2008–2022 | PhysioNet credentialed; CITI; DUA 1.5.0 (verify current version) | Mortality, readmission, sepsis, AKI, phenotyping, treatment effect | Single center; ICU selection; shifted dates; post-discharge mortality incomplete; one institution's coding |
| **MIMIC-IV demo** | 100-patient subset | 100 patients | Open on PhysioNet | Teaching, pipeline tests, schema exploration | Too small for scientific claims; do not tune and report on it |
| **MIMIC-IV-ED** | ED visits, triage, vitals, medications, dispositions | Hundreds of thousands of visits (verify release count) | PhysioNet credentialed | Boarding, triage, disposition, operations | ED-only; post-ED outcomes require linkage |
| **MIMIC-IV-Note** | Discharge summaries and radiology reports | Linked to MIMIC-IV | PhysioNet credentialed | Summarization, NER, factuality | De-identification removes clinically relevant tokens; copy-forward |
| **eICU-CRD** | Multi-hospital ICU physiology, diagnoses, treatments | About 200,000 stays across roughly 200 US hospitals (verify) | PhysioNet credentialed | Cross-site modeling, transportability, deterioration | Uneven variable coverage; no notes or imaging; practice variation |
| **HiRID** | High-resolution ICU physiology, Bern | Tens of thousands of ICU stays (verify) | Credentialed, Swiss governance | High-frequency deterioration, hypotension | Single center; European practice; complex ethics |
| **AmsterdamUMCdb** | Dutch academic ICU database | About 23,000 admissions (verify) | Credentialed | Phenotyping, benchmark transport | Single center; different coding and units |
| **UK Biobank** | Questionnaires, physical measures, genomics, imaging, linked hospital and death records | 500,000 participants | Application; fee; UKB-RAP; Tier 3 on the order of £9,000 per three years for academia (verify schedule) | Risk prediction, GWAS, imaging genetics, multimorbidity | Healthy-volunteer and ancestry bias; older than the UK; imaging subset selected |
| **All of Us CDR v9** | Surveys, EHR, measurements, genomics from a diverse US cohort | 747,000+ participants; 535,000+ short-read WGS; 553,000 arrays; ~96,000 structural variants; ~14,000 long-read; first clinical notes (v9, June 2026; verify release details) | Registered and Controlled tiers; DURA; Workbench; about $300 initial cloud credits (verify) | Disparities, PRS portability, PheWAS, cohort discovery | Volunteer cohort; EHR linkage varies; output review; small subgroups |
| **Epic Cosmos** | De-identified EHR across Epic customers | About 2,358 hospitals and 310 million patients (vendor figure, 2026) | Contributing organizations only | Population health, rare disease counts, cohort exploration | Contributor bias; no row-level access; vendor-mediated phenotypes |
| **EHRSHOT** | Few-shot EHR benchmark on Stanford data | 6,739 patients; 15 binary tasks (verify task list) | PhysioNet credentialing plus Stanford agreement | Few-shot and foundation-model evaluation | Small; single institution; code-derived labels |
| **FoMoH** | EHR foundation-model benchmark | 14 tasks; more than 6 million patients | Code and data links vary by constituent dataset | Cross-institutional evaluation | Fragmented access; harmonization drives results |
| **Synthea** | Open-source synthetic patient generator | Configurable; millions of synthetic patients | Apache 2.0 | Pipeline, FHIR/OMOP testing, demos | Not real distributions; never validate clinical performance |
| **CMS SynPUF** | Synthetic Medicare-like claims | About 2.3 million beneficiaries, 2008–2010 (verify) | Open from CMS | Claims pipelines, cost analytics, SQL teaching | Synthetic, dated, not current coding |
| **OMOP CDM / Eunomia** | Standardized observational model; small synthetic OMOP dataset | Eunomia is a small sample | Open tooling; verify data license | Cohort definitions, OHDSI methods, SQL portability | Synthetic; ETL mapping loss; vocabulary versions |

**Field notes.** MIMIC-IV is an instrument for one hospital, not a sample of US care: strong for methods and phenotyping, weak for national incidence, equity, or health-system performance. eICU trades depth for breadth; report coverage by hospital and variable before modeling. UK Biobank offers deep phenotyping in a healthy volunteer cohort; All of Us deliberately oversamples underrepresented groups but has heterogeneous EHR linkage and a different consent model. Neither is a probability sample. Epic Cosmos is a query product, not a row-level research database. EHRSHOT and FoMoH evaluate models, not products: few-shot gains do not imply calibration, fairness, or workflow safety.

## 6. Imaging

| Dataset | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **MIMIC-CXR-JPG v2.0.0** | Chest radiographs with reports and CheXpert-style labels | 377,110 images; 227,835 studies; 65,379 patients | PhysioNet credentialed | Multi-label classification, report generation, grounding | NLP-derived labels, not adjudicated truth; one institution; support devices confound |
| **CheXpert** | Chest radiographs with expert and rule-based labels, including uncertainty | 224,316 images; 65,240 patients | Free for non-commercial research under Stanford terms (verify) | Classification, uncertainty, calibration | Inconsistent uncertainty handling; reused validation split; no external test |
| **NIH ChestX-ray14** | Frontal radiographs with NLP-mined labels | 112,120 images; 30,805 patients | Open | Classification, weak localization | Noisy labels; patient overlap in early splits |
| **RSNA challenge data** | 2024 lumbar spine, 2025 intracranial aneurysm, 2026 knee abnormalities | Hundreds to thousands of studies per edition (verify) | Kaggle rules; research use | Detection, classification, localization | Rules restrict external data and pretrained models; hidden test shift |
| **BraTS** | Multi-modal brain MRI with tumor subregions | Roughly 1,000–1,500 cases per recent edition (verify) | Synapse/CodaLab registration; non-commercial | Glioma segmentation, survival, federated learning | Editions change labels; skull-stripping choices; Dice misses small residuals |
| **CAMELYON16/17** | Lymph-node metastasis whole-slide images | Hundreds of slides | Open via Grand Challenge | Metastasis detection, WSI classification | Slide-level leakage; scanner and stain variation |
| **PANDA** | Prostate biopsies with ISUP grade | 10,616 development biopsies | Kaggle (CC BY-SA-NC 4.0) and Grand Challenge; verify current access | Gleason grading | Inter-pathologist disagreement; imbalance |
| **VinDr-CXR / PadChest** | Chest radiographs with multi-reader or report labels | Tens of thousands each (verify) | Registration; research terms | Detection, localization, labeling | Different ontologies; language and population shift |

**Field notes.** Labels are the binding constraint. Most chest radiograph labels are extracted from reports, inheriting reporting style, negation handling, and institutional vocabulary; a model can learn phrase co-occurrence rather than the finding. Patient-level splitting is mandatory: splitting by image leaks anatomy and acquisition. Preprocessing is part of the model, so a number without published code and weights is a range, not a point. External validation across hospital, scanner, or era is the decisive test.

## 7. Signals and wearables

| Dataset | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **PTB-XL** | Clinical 12-lead ECGs, 10 s, 500 Hz, with statements | 21,837 records; 18,885 patients | Open on PhysioNet, ODC Attribution (verify) | Arrhythmia and morphology classification, pretraining | Statement-derived labels; multi-label records; German population |
| **PhysioNet Challenge 2025** | 12-lead ECGs for Chagas triage | CODE-15% about 300,000; SaMi-Trop 1,631 positives; PTB-XL negatives | Open challenge data plus credentialed sets | Capacity-constrained screening | Weak labels; prevalence shift; metric is precision at top 5% |
| **PhysioNet Challenge 2026** | Polysomnography for later cognitive impairment | Five US institutions; EDF; about 214 GiB small, 1.2 TiB large | Challenge registration; verify license | Sleep modeling, dementia prediction | Age confounding; diagnosis 1–6 years later; site shift |
| **Sleep-EDF** | Whole-night PSG: EEG, EOG, EMG, temperature, respiration | 197 recordings | Open, ODC Attribution | Sleep staging, signal quality | Small; healthy volunteers; old equipment; scoring-rule version |
| **TUH EEG** | Clinical EEG with montages and annotations | Tens of thousands of sessions (verify version) | Registration and DUA on PhysioNet | Seizure detection, artifact classification | Montage and device heterogeneity; label provenance |
| **MIMIC-IV waveform** | ICU waveforms matched to MIMIC-IV | Large subset of ICU stays (verify) | PhysioNet credentialed | Deterioration, hypotension, arrhythmia | Single center; device-specific; alignment errors |
| **Icentia11K** | Single-lead ambulatory ECG | About 11,000 patients (verify) | Open | Arrhythmia detection, pretraining | Single device; limited labels |
| **WESAD** | Chest and wrist signals for stress | 15 subjects | Open for research | Stress detection, sensor fusion | Tiny; laboratory protocol; not a clinical endpoint |
| **NSRR** | Sleep and cardiometabolic cohorts, including MESA sleep | Multiple cohorts | Application and DUA | Sleep-disordered breathing, CVD risk | Heterogeneous protocols |

**Field notes.** Signal datasets are device datasets: sampling rate, filters, electrode placement, and firmware define the distribution. A model trained on one wrist device rarely transfers without recalibration, and skin tone, perfusion, body composition, and motion change the optical signal itself. Window-level splits leak; split by patient and, where possible, by device and site. Challenge metrics encode operational constraints — precision at top 5% for scarce serology capacity, age-conditioned AUROC to remove a trivial shortcut — so read the metric before the leaderboard.

### 7.3 Surgical, robot, and device telemetry

| Dataset | Contents | Access | Tasks | Pitfalls |
|---|---|---|---|---|
| **Cholec80** | Laparoscopic cholecystectomy video with phase and tool annotations | Request from CAMMA/Strasbourg | Phase recognition, tool detection, workflow | Frames leak; single procedure type; privacy |
| **CholecT50** | Cholecystectomy video with triplet instrument–verb–target annotations | Request | Action recognition, surgical data science | Sparse annotations; class imbalance |
| **JIGSAWS** | Suturing, knot-tying, needle-passing with robot kinematics and video | Request from JHU | Skill assessment, gesture recognition | Small; expert novices; task-specific |
| **MISAW / HeiChole** | Surgical workflow and phase benchmarks | Challenge registration | Phase recognition, multi-center evaluation | Annotation conventions; site shift |
| **dVRK** | da Vinci Research Kit kinematics, video, and control interface | Open hardware/software | Control, imitation learning, teleoperation | Hardware access; safety; calibration |
| **SurRoL / Surgical Gym** | Surgical robot simulation environments | Open | Reinforcement/imitation learning, sim-to-real | Simulator fidelity; contact modeling |
| **OhioT1DM / DiaTrend** | CGM, insulin, meals, activity | Request/open subsets | Glucose prediction, artifact detection | Small; type 1 diabetes; sensor lag |
| **WESAD / Opportunity / UCI HAR** | Wearable physiology and activity | Open | Stress, activity, sensor fusion | Lab protocols; placement; small samples |
| **FDA MAUDE / recall database** | Device adverse events and recalls | Open | Safety signal detection, post-market surveillance | Under-reporting; narratives; no denominator |

**Semiconductor manufacturing data.** There is no broadly accessible, clinically representative public fab dataset. Semiconductor process control, yield, and failure-analysis data are proprietary. For teaching, use synthetic statistical-process-control or fault-detection logs and state clearly that they are simulated; do not present them as evidence about a real device supply chain.

**Field notes.** Surgical video and robot logs are high-risk data: faces, tattoos, operating-room audio, and rare procedures can identify patients and staff. De-identification is not enough; governance, consent, and secure compute are required. Robot logs also depend on firmware, calibration, and coordinate-frame conventions. A kinematics model trained on one dVRK setup may not transfer to another.

## 8. Omics, single-cell, and spatial

| Dataset | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **TCGA** | Clinical, expression, mutation, copy number, methylation, pathology | About 11,000 patients, 33 cancer types | Open via GDC; germline controlled via dbGaP | Survival, subtypes, multi-omic integration | Bulk mixtures; treatment and stage confounding; batch by center |
| **GTEx** | Normal-tissue expression and eQTLs | About 17,000 samples from about 950 donors (v8; verify) | Open via GTEx Portal and AnVIL | eQTL, tissue-specific expression | Post-mortem tissue; small per-tissue n; ancestry skew |
| **gnomAD** | Aggregate allele frequencies | Hundreds of thousands of individuals (verify release) | Open, registration for some files | Variant interpretation, constraint | Ancestry composition changes across releases; frequency is not effect |
| **1000 Genomes** | Whole-genome sequences, 26 populations | 2,504 individuals | Open | Population genetics, imputation reference | Small by modern standards |
| **CELLxGENE Discover** | Single-cell atlas with standardized metadata | Tens of millions of cells, growing (verify) | Open | Cell-type annotation, integration, label projection | Incomplete metadata harmonization; donor and tissue imbalance |
| **HuBMAP / spatial collections** | Spatial transcriptomics with histology and coordinates | Hundreds of datasets and sections (verify) | Open per consortium or study terms | Niche detection, spatial domains, cell–cell communication | Section alignment; resolution varies; no settled benchmark |
| **Human Cell Atlas** | Community atlas across tissues | Thousands of datasets | Open per dataset | Atlas building, cell ontology | Heterogeneous protocols; access varies |
| **GEO / ArrayExpress** | Functional genomics studies | Hundreds of thousands of samples | Open; per-series terms | Differential expression, reanalysis | Metadata quality; batch confounding |
| **PRIDE / MetaboLights** | Proteomics and metabolomics | Tens of thousands of datasets | Open | Biomarker discovery, pathway analysis | Identification confidence; missing values |
| **DepMap** | Cancer cell line dependency and omics | About 1,000–1,200 lines (verify) | Open with registration | Target discovery, synthetic lethality | Cell lines are not tumors; in vitro artifacts |
| **LINCS L1000 / CMap** | Perturbational expression profiles | Over a million profiles (verify) | Open | Drug repurposing, signature matching | Cell-line specific; dose and time heterogeneity |
| **Open Targets** | Target–disease evidence: genetics, expression, pathways, safety | Quarterly-ish releases; counts change (verify) | Open with attribution | Target prioritization, evidence integration | Evidence types are not independent; scores are not probabilities |
| **ChEMBL** | Bioactivity, compounds, assays, targets | Millions of activity records (verify) | Open, CC BY-SA 4.0 | Drug discovery, QSAR | Assay heterogeneity; publication bias; share-alike |
| **PubChem** | Structures, bioassays, properties | Over 100 million compounds | Open; check depositor terms | Cheminformatics, toxicity screening | Variable assay quality; identifier changes |
| **CAGI7** | Variant effects, PRS, DMS, splicing, regulatory challenges | 14 challenges (2025) | Open challenge data | Variant effect prediction, PRS portability | Small truth sets; metric choice shapes winners |
| **Open Problems** | Continuous single-cell method benchmarks | 14 problems; about 140 methods; about 127 datasets | Open | Integration, label projection, communication | Benchmark tasks may not match biological questions |

**Field notes.** Bulk and single-cell are different measurement regimes: a deconvolved proportion is a model output, not an observed count. Batch can be confounded with biology, and aggressive integration can erase disease signal — evaluate on held-out donors and report corrected and uncorrected results. Single-cell foundation models are not automatically better: recent benchmarks found zero-shot scGPT and Geneformer underperform scVI and Harmony on clustering and integration, and larger pretraining did not reliably help. Always include a PCA or logistic-regression baseline. Spatial transcriptomics still lacks settled benchmarks; treat headline numbers as provisional.

## 9. Pathology

| Resource | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **TCGA WSI** | Diagnostic slides matched to TCGA | Thousands of slides | Open via GDC | Cancer classification, survival, molecular correlation | Frozen vs FFPE; slide quality; clinical labels |
| **CAMELYON16/17** | Lymph-node metastasis WSI | Hundreds of slides | Open | Detection, WSI classification | Small; single task; slide leakage |
| **PANDA** | Prostate biopsies with ISUP grade | 10,616 development biopsies | Kaggle (CC BY-SA-NC 4.0) and Grand Challenge; verify access | Grading | Reader disagreement; imbalance |
| **MIDOG** | Mitosis detection across scanners | Multiple domains | Challenge registration | Domain generalization | Scanner shift is the point; few domains |
| **SLC-PFM** | Self-supervised pathology foundation-model benchmark | 300 million+ images; 39 cancers; 23 tasks | NeurIPS 2025 competition terms | Foundation-model probing | Enormous compute; new benchmark; label provenance |

**Field notes.** Whole-slide images are gigapixel, so pipelines tile, and tile sampling determines what the model can see. Tile-level splits leak. Stain normalization helps within a scanner and can hurt across scanners. Report by scanner, stain protocol, and slide quality, not only cancer type. Foundation-model benchmarks are expensive and easy to overfit; a linear probe on a well-chosen feature set remains a strong baseline.

## 10. Clinical text and LLM benchmarks

| Benchmark | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **HealthBench** | Multi-turn clinician-written conversations with rubrics | 5,000 conversations; 48,562 criteria; 262 physicians | Open release | Advice quality, safety, completeness | Model- or physician-mediated grading; best models about 32% on Hard in 2025; rubric coverage is not outcome |
| **MedAgentBench** | Agent tasks on a FHIR server with synthetic patients | Hundreds of clinician-authored tasks (verify) | Open code and data; verify license | Tool use, order entry, retrieval, write safety | Simulated EHR; pass@1 hides unsafe near-misses |
| **MedQA / MedMCQA / PubMedQA** | Exam and literature QA | Thousands to tens of thousands of items | Open | Knowledge QA | Pretraining contamination; exam format is not reasoning |
| **MMLU medical subsets** | General knowledge with medical items | Hundreds of items | Open | Model comparison | Saturation; leakage |
| **MedHELM** | Clinician-designed task suite | Dozens of benchmarks (verify) | Open | Holistic evaluation | Task weighting; rater reproducibility |
| **CURE-Bench** | LLM reasoning for precision therapeutics | Competition dataset | NeurIPS 2025 terms | Therapeutic reasoning, agentic search | Clinician review cost; hidden test |
| **TrialGPT-style sets** | Patient-to-trial matching | Derived from registry and notes | Code open; data varies | Eligibility matching, ranking | False exclusion harms; criteria parsing |
| **i2b2 / n2c2** | NER, relations, de-identification, social determinants | Task-specific corpora | DUA via Harvard DBMI; some open | Clinical NLP | Small; scheme-specific; PHI restrictions |
| **MIMIC-IV-Note** | Discharge summaries and radiology reports | Linked to MIMIC-IV | Credentialed | Summarization, factuality | Reference summaries are not gold for every fact |

**Field notes.** Contamination is the first-order threat: QA items, guidelines, and rubric text appear in pretraining corpora. Check publication dates against model cutoffs, run canary and paraphrase probes, and prefer fresh or private items. LLM-as-judge is a measurement instrument that needs validation: report judge model, prompt, temperature, and agreement with physician grades. Agent benchmarks reward correct tool calls, not clinical judgment, so pair pass rates with unsafe-action rates. When top models cluster within noise, the benchmark has saturated; move to harder tasks and say so.

## 11. Public health

| Dataset | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **NHANES** | Interview, examination, laboratory, dietary | About 5,000 per two-year cycle | Open | Prevalence, risk factors, disparities | Complex survey design; weights and strata required; cycle harmonization |
| **NHIS / BRFSS** | National and state health surveys | Tens to hundreds of thousands per year | Open | Access, utilization, chronic disease | Self-report; redesigns break series; state variation |
| **CDC WONDER** | Mortality and natality aggregates | National and subnational | Open | Mortality trends | Small-count suppression; ICD revisions; crude vs age-adjusted |
| **NWSS** | Wastewater surveillance | More than 1,500 sites (2026 brief) | Open | Nowcasting, outbreak detection | Sewer-shed denominators; sampling frequency |
| **ESSENCE / NSSP** | ED syndromic surveillance | National coverage | Access varies | Respiratory, overdose, heat | Syndrome definitions; reporting lag |
| **SEER** | Population-based cancer registry | Multiple registries | Registration and DUA | Incidence, survival, trends | Catchment; staging changes; limited treatment data |
| **GLOBOCAN / IARC** | Global cancer estimates | Country-level | Open aggregate | Global burden | Modeled estimates, not counts |
| **CMS Care Compare** | Hospital, nursing home, dialysis quality | National | Open | Benchmarking, readmission, stars | Measure changes; risk adjustment; lag |
| **County Health Rankings / ACS** | Social, economic, environmental indicators | County and tract | Open | Social determinants | Ecological fallacy; margins of error; vintage mismatch |

**Field notes.** Public health data are usually aggregated; joining them to clinical data at fine geography creates linkage error and disclosure risk. Survey analysis must be design-aware, or point estimates look precise and are wrong. When a policy claim rests on a modeled estimate, say so.

## 12. Pharma, trials, and safety

| Resource | Contents | Scale | Access | Tasks | Pitfalls |
|---|---|---|---|---|---|
| **ClinicalTrials.gov** | Registrations, protocols, status, results | Hundreds of thousands of studies | Open; API | Landscape, eligibility, enrollment | Registration quality; incomplete results |
| **AACT** | Relational mirror of the registry | Full registry | Open | SQL analysis, benchmarking | Refresh lag; schema changes |
| **EU CTIS** | EU trial information | Growing | Open | European landscape | Transition from EudraCT; incomplete history |
| **OpenFDA / AEMS (formerly FAERS)** | Adverse event reports and labels | Millions of reports | Open API and extracts | Signal detection, label mining | Spontaneous reports; no denominator; duplicates; not causality |
| **VAERS** | Vaccine adverse event reports | Hundreds of thousands | Open | Vaccine safety signals | Same reporting limits; media effects |
| **EudraVigilance** | EU adverse reaction reports | Millions | Public web; bulk export registration | Signal detection | Duplicates; access tiers |
| **VigiBase** | WHO global case safety reports | Tens of millions | Restricted to member countries and centers | Global pharmacovigilance | Licensed; disproportionality is not risk |
| **MAUDE** | Device adverse event reports | Millions | Open | Device safety, recalls | Under-reporting; manufacturer narratives |
| **Sentinel** | FDA distributed claims and EHR network | Large national coverage | Not open; FDA program | Active surveillance | Access via FDA or collaborators |
| **Vivli / YODA / CSDR / Project Data Sphere** | Individual participant data from trials | Study-dependent | Request and review | Reanalysis, external controls | Approval latency; use restrictions |
| **TriNetX** | Federated EHR and claims network | Large global network | Licensed | Feasibility, cohort counts, RWE | Site heterogeneity; query-only |
| **CMS Open Payments** | Industry payments to clinicians | Millions of records | Open | Conflict of interest, influence | Attribution ambiguity; coarse categories |
| **HCUP** | US discharge and ED databases: NIS, SID, SEDD | National and state | Purchase; training; DUA | Utilization, cost, readmission | Not all states; discharge-level units; coding changes |

**Field notes.** AEMS (formerly FAERS) and VAERS are reporting systems, not incidence registries. A report count cannot become a risk without a denominator and a reporting-rate model; disproportionality (PRR, ROR) generates hypotheses, not causation. HCUP is the best public-adjacent inpatient utilization source, but state coverage varies and the unit is often a discharge rather than a patient — read whether verified patient identifiers exist in the file you bought.

## 13. Operations and payments

| Resource | Contents | Access | Use | Caution |
|---|---|---|---|---|
| **CMS Open Payments** | Industry payments to clinicians | Open | COI, influence, network analysis | Attribution; specialty coding |
| **CMS Provider Data Catalog** | Quality, utilization, payment measures | Open | Benchmarking, market analysis | Risk adjustment; lag |
| **HCRIS** | Hospital cost reports | Open | Margin, cost structure, uncompensated care | Filing errors; fiscal-year variation |
| **HCUP NIS/SID/SEDD** | Discharge and ED utilization | Purchase | Cost, readmission, equity | State coverage; discharge-level |
| **CMS SynPUF** | Synthetic claims | Open | Pipeline development | Synthetic, dated |
| **MarketScan / Optum** | Commercial claims | Licensed | Cost, utilization, treatment patterns | Contract limits; coding incentives; no severity |
| **FAIR Health** | Claims-derived benchmarks | Licensed | Price transparency, geographic variation | Proprietary methodology |
| **MIMIC-IV-ED** | ED operations | Credentialed | Boarding, triage, disposition | Single site |

**Field notes.** Operations claims describe flow and payment, not clinical need: a high-cost cohort may be a well-billed one. Verify every feature is available before the operational decision, and that the label is not a downstream consequence of the intervention being guided. There is no canonical public operating-room log; do not cite one unless you can name it.

## 14. Model hubs and data catalogs

| Hub | Hosts | License reality | Teaching use |
|---|---|---|---|
| **Hugging Face Hub** | Transformers, datasets, biomedical models | Mixed; each model card governs; gated models require acceptance | Fast prototyping; inspect training data |
| **MONAI Model Zoo** | Medical imaging models with bundles | Apache 2.0 tooling; weights per bundle | Segmentation baselines |
| **BioImage Model Zoo** | Bioimage analysis models | Per-model | Microscopy segmentation |
| **PhysioNet project pages** | Challenge models and code | Per project | Signal and ICU baselines |
| **NVIDIA NGC** | Containers and pretrained models | NVIDIA terms; some research-only | GPU inference |
| **Google Health AI Developer Foundations** | MedGemma, MedSigLIP, HeAR, MedASR | Foundation terms; verify commercial use | Multimodal, low-resource prototypes |

| Catalog | Coverage | Note |
|---|---|---|
| **PhysioNet** | Signals, ICU, challenges | Credentialing hub for the MIMIC family |
| **NCI GDC** | TCGA and cancer genomics | Open and controlled tiers |
| **dbGaP / EGA** | Controlled genomics | Institutional, project-specific |
| **GEO / ArrayExpress / BioStudies** | Functional genomics | Open, heterogeneous |
| **CELLxGENE / HCA** | Single-cell atlases | Standardized or per-project schema |
| **Synapse / Kaggle** | Challenges and consortia | Governance and rules per project |
| **HealthData.gov / data.cms.gov** | US public health and payment | Open |
| **ClinicalTrials.gov / AACT / CTIS** | Trials | Registration quality varies |
| **OpenFDA** | Drugs, devices, recalls, AEMS | API rate limits |
| **Open Targets / ChEMBL / PubChem** | Targets and compounds | Version the release |
| **Zenodo / OSF** | Code, supplements, small data | Check license and provenance |
| **OHDSI / Eunomia** | OMOP tooling and synthetic data | Synthetic |

**Rule:** a model card is a claim, not evidence. Check training data, evaluation population, license, and whether the reported metric is internal or external.

## 15. "Do not use this dataset for that claim"

Each row is a claim experienced reviewers reject on sight.

| Do not use | To claim | Why it fails | Use instead |
|---|---|---|---|
| MIMIC-IV | National incidence or prevalence | Single-center, ICU-selected | NHANES, SEER, national claims, probability sample |
| MIMIC-IV | US racial or socioeconomic disparities | One catchment; race coding and access differ | All of Us, multi-site data, weighted surveys |
| eICU-CRD | Hospital quality ranking | Voluntary participation; uneven coverage | CMS Care Compare with risk adjustment |
| Epic Cosmos | Individual-level prediction research | Contributor-only, aggregate query product | Credentialed row-level data |
| UK Biobank | General-population prevalence | Healthy-volunteer and ancestry bias | NHANES, national registry |
| UK Biobank | Causal effect of a lifestyle exposure | Observational confounding; one cohort | Target trial emulation with negative controls, or an RCT |
| All of Us | Unbiased national estimates | Volunteer cohort; heterogeneous linkage | Weighted national survey; state the estimand |
| TCGA | Pan-cancer survival in the population | Tertiary-center cases; treatment selection | SEER, national registries |
| TCGA | Clinical validity of a biomarker | Retrospective; heterogeneous treatment | Prospective cohort or trial with prespecified assay |
| ChEMBL / PubChem | Clinical efficacy or safety | In vitro assays; publication bias | Trials, AEMS with denominators, regulatory reviews |
| Open Targets | Causal target validation | Evidence integration, not experiment | Perturbation experiments; human genetics triangulation |
| CELLxGENE | Tissue-wide cell-type proportions | Dataset composition, not a sampling frame | Designed single-cell study with replicates |
| OpenFDA / AEMS | Incidence or relative risk | Spontaneous reports; no denominator | Sentinel, EHR cohort, denominator-based study |
| CMS Open Payments | Proof of improper influence | Payments are legal; categories coarse | Investigative design with context |
| HCUP | Patient-level longitudinal outcomes without verified IDs | Discharge-level files; state variation | Linked data with verified identifiers |
| Synthea | Clinical performance or fairness | Synthetic, not real distributions | Real credentialed data; synthetic only for pipelines |
| A public leaderboard | Deployment performance | Contamination, split differences, overfitting | Prospective or external validation |
| A single-site model | Generalization to new hospitals | Site, scanner, coding, and workflow shift | Multi-site external test |
| A benchmark average | Clinical utility | Metric may not encode decision costs | Decision-curve or net-benefit analysis |

## 16. Tool stack by function

Tools are listed because they are widely used and inspectable, not endorsed. Pin versions, record licenses, and prefer the simplest tool that passes acceptance.

| Function | Tools | Caution |
|---|---|---|
| **Languages and data** | Python, R, SQL, DuckDB, PostgreSQL, dbt, Polars/pandas, Parquet | Environment drift; SQL dialect differences; pandas memory limits |
| **Imaging** | pydicom, Highdicom, SimpleITK, MONAI, nnU-Net, TotalSegmentator, OpenSlide, CLAM, TIAToolbox, 3D Slicer, OHIF | Private tags and burned-in text; orientation conventions; patch sampling dominates WSI results |
| **Signals** | WFDB, MNE-Python, NeuroKit2, BioSPPy, HeartPy, SciPy, Braindecode | Unit and header conventions; filter and reference choices; non-causal filters in real-time use |
| **NLP and LLM** | Transformers, spaCy/scispaCy/medspaCy, MedCAT, cTAKES, negspaCy, vLLM, LangChain, LlamaIndex, DSPy, Inspect, HELM, Presidio | Tokenizer and prompt drift; UMLS licensing; hidden agent state; PII tools are not clinical de-identification |
| **Omics** | Scanpy, AnnData, Seurat, scVI-tools, Harmony, Bioconductor, DESeq2, edgeR, Nextflow, Snakemake, nf-core, GATK, DeepVariant, bcftools, VEP, PLINK, SAIGE, regenie, squidpy, SpatialData | Normalization choices; over-correction; reference and annotation versions; container pinning |
| **Causal and statistics** | statsmodels, lifelines, scikit-survival, R survival/cmprsk, DoWhy, EconML, CausalML, MatchIt, WeightIt, cobalt, tmle, SuperLearner, brms, INLA | A graph is an assumption; positivity; balance is not unconfoundedness; priors are claims |
| **MLOps and reproducibility** | Git/Git LFS, DVC, MLflow, Weights & Biases, Docker, Apptainer, Kubernetes, Terraform, Airflow, Prefect, Dagster, Great Expectations, Evidently, BentoML, Triton, Feast, Zenodo, OSF, Code Ocean | Never log PHI; retry and idempotency semantics; drift is not harm; point-in-time correctness |
| **Fairness, privacy, visualization** | Fairlearn, AIF360, Aequitas, SHAP, Captum, LIME, DiCE, OpenDP, diffprivlib, Opacus, NVIDIA FLARE, Substra, MedPerf, ARX, sdcMicro, SDV, synthcity, matplotlib, seaborn, plotly, Altair, ggplot2, Tableau, Power BI, Vitessce, QuPath | Fairness definitions conflict; explanations are not causes; privacy budget accounting; colorblind-safe palettes |
| **Efficient inference** | ONNX Runtime, TensorRT, OpenVINO, TFLite/TFLite Micro, CMSIS-NN, STM32Cube.AI, Edge Impulse, ExecuTorch, TVM, llama.cpp, MLC-LLM | FLOPs are not latency; operator fallback; calibration and subgroup shifts after quantization; target measurement required |

## 17. Benchmark pitfalls

| Pitfall | Mechanism | Detection | Mitigation |
|---|---|---|---|
| **Pretraining contamination** | Test items in training data | Date audit; canary strings; paraphrase tests | Fresh or private test set; report probes |
| **Patient leakage** | Same patient in train and test | Check IDs and visit dates | Split by patient |
| **Study or image leakage** | Images of one study split | Group by study UID | Group-aware splitting |
| **Temporal leakage** | Future information predicts the past | Audit feature timestamps | Point-in-time construction |
| **Site leakage** | Same hospital or scanner on both sides | Metadata audit | Leave-site-out validation |
| **Label provenance** | Labels from reports, codes, or LLMs | Trace source; re-adjudicate a sample | Report source and agreement; sensitivity analysis |
| **Label noise** | Upcoding, rule-out diagnoses, ambiguity | Inter-rater study; outlier review | Probabilistic labels; noise-robust loss; report the ceiling |
| **Leaderboard overfitting** | Repeated submissions to a small test | Public vs private score gap; submission count | One probe per week; trust local validation |
| **Metric mismatch** | AUROC for a cost-sensitive or fixed-capacity decision | Map metric to decision | Precision at capacity, net benefit, calibration |
| **Access bias** | Cohort over-represents people with access | Compare to target population | Reweight, stratify, or collect |
| **Single-site** | One institution's practice and equipment | Site metadata; external test | Multi-site validation |
| **Benchmark saturation** | Top models cluster within noise | Confidence intervals; seed variance | Report CIs; move to harder tasks |

**Contamination probes that work.** Hold out items published after the model cutoff. Insert canary tokens or perturb option order. Embed test items and search public corpora for near duplicates. Keep a private holdout. Vary prompts and measure sensitivity — large swings mean the benchmark tests prompt-following as much as knowledge.

**A credible benchmark report contains:** dataset name, version, release and access dates; label definition and provenance with inter-rater agreement; split strategy and group counts; preprocessing fit scope; model checkpoint and training cutoff; metric with confidence intervals and seed variance; subgroup results by site, sex, age, race/ethnicity, and device; calibration; contamination checks; compute and cost; code or a stated reason it cannot be shared.

**Reading a leaderboard, in order:** What is the test set and has it been reused? Are splits patient-level? Are external data and pretrained models allowed? Is the metric the decision metric? What are the confidence intervals? Who is missing? What changed between editions? A leaderboard is a marketing surface and a research tool at once; treat the ranking as a hypothesis.

### Post-mortems worth teaching

**RSNA 2026 knee MRI challenge (closed 22 October 2026).** About 6,425 teams, 12 findings, macro ROC AUC. The edition surfaced disputes over LLM-derived labels and possible leakage between training and test. Lesson: when labels are generated at scale, the label pipeline is part of the benchmark and must be published with it. Verify the final organizer report before citing outcomes.

**GRAIL NHS-Galleri (primary endpoint reported 2026).** About 140,000 participants; the primary endpoint of stage III/IV cancer incidence gave an incidence rate ratio of 1.03 (95% CI 0.92–1.14), with stage IV down 14% and stage I/II up 16%. Results were presented at ASCO on 30 May 2026 and published in peer-reviewed journals on 22–23 September 2026 (verify the exact venues). NHS England had already decided in 2024 not to proceed to a pilot based on first-year data; that decision was not a consequence of the 2026 primary result. Lesson: a screening test can shift stage distribution without reducing advanced-stage incidence. The endpoint, not the AUC, decides adoption.

**Apple hypertension notification (FDA clearance 11 September 2025, K250507).** Passive PPG over 30-day windows, no blood-pressure reading; sensitivity 41.2% overall and 53.7% for Stage 2, specificity 92.3%, n=2,229. Lesson: a high-specificity, low-sensitivity consumer notification can be a legitimate public-health product and a poor screening test. Match the claim to the intended use, not to the modality's prestige.

## 18. Labs and exercises

Seven labs, 60–120 minutes each. Labs 21.1–21.3 are prerequisite for the assignment.

### Lab 21.1 — Dataset audit

**Objective:** Audit one dataset against the five gates and produce a one-page verdict.
**Data:** MIMIC-IV demo, an NHANES cycle, TCGA-BRCA clinical and expression, PTB-XL, or a CellxGene subset.
**Steps:** Record name, version, release and access dates, and owner. State what is measured and at what unit (patient, encounter, sample, cell). List outcome, exposure, and confounders, marking whether each is available at decision time. Tabulate subgroup coverage by age, sex, race/ethnicity, site, and year. Identify three validity threats and one claim the data cannot support. Score the five gates.
**Deliverable:** one-page audit with coverage table and a "cannot support" statement.
**Acceptance:** version and access date present; at least one disqualifying limitation; no claim beyond the data; all five gates addressed.
**Stretch:** Repeat on a second dataset and state which claim both can support.

### Lab 21.2 — License and access audit

**Objective:** Decide whether a planned commercial product can legally use a candidate dataset.
**Data:** Instructor supplies three terms of use (for example MIMIC DUA, CheXpert terms, UK Biobank material transfer agreement).
**Steps:** Extract clauses on commercial use, redistribution, model training, publication, and retention. Map each to the product plan: training, evaluation, demo, model release, SaaS. Identify the clause most likely to block the plan. Draft questions for the access committee. Propose an alternative dataset or a design change.
**Deliverable:** clause-by-clause matrix plus a 300-word recommendation.
**Acceptance:** quotes accurate and located; no legal advice presented as certainty; recommendation actionable.
**Stretch:** Compare a US and an EU license for cross-border processing.

### Lab 21.3 — Benchmark validity audit

**Objective:** Critique a published benchmark result.
**Data:** One recent paper or leaderboard report in imaging, LLMs, or single-cell.
**Steps:** Identify dataset version, split strategy, and label source. Check patient, site, and temporal leakage. Compare the metric to the stated decision. Look for contamination evidence. Recompute confidence intervals from reported n where possible. Write three questions for the authors and one claim you would not accept.
**Deliverable:** two-page validity memo.
**Acceptance:** specific evidence cited by section or table; at least one issue quantified; no vague "bias" claims.
**Stretch:** Reproduce the headline number on a public subset, or explain precisely why it is not reproducible.

### Lab 21.4 — Baseline reproducibility

**Objective:** Reproduce a simple baseline on a public dataset and document every choice.
**Data:** PTB-XL, a CheXpert subset, or a CellxGene subset.
**Steps:** Fix a patient-level split and write it to disk with a hash. Fit preprocessing on train only. Train a logistic-regression or gradient-boosted baseline and a small neural baseline. Report AUROC or macro-F1 with bootstrap confidence intervals over at least five seeds. Run one subgroup analysis. Package code, environment, and split.
**Deliverable:** repository plus a one-page results table.
**Acceptance:** another student reruns within 0.01 of the reported metric; seeds and versions pinned; no test-set tuning.
**Stretch:** Add a calibration curve and a decision-curve analysis.

### Lab 21.5 — Data card and datasheet

**Objective:** Produce a data card suitable for internal review or a regulatory file.
**Data:** The dataset audited in Lab 21.1.
**Steps:** Populate motivation, composition, collection, preprocessing and labeling, uses, distribution, maintenance, and ethical and legal considerations, following Datasheets for Datasets.
**Deliverable:** two-page data card.
**Acceptance:** every field populated or explicitly marked unknown with a reason; no marketing language; specific limitations.
**Stretch:** Write the matching model card and cross-reference the two.

### Lab 21.6 — Leaderboard critique

**Objective:** Decide whether a competition leaderboard supports a deployment claim.
**Data:** A live or recent Kaggle, RSNA, PhysioNet, or NeurIPS health competition.
**Steps:** Read rules on external data, pretrained models, and submission limits. Determine public/private split and team count. Estimate score variance at the top. Summarize the top three approaches and leakage risk. Compare the winning metric to a clinical decision metric. Recommend deploy, pilot, or do not use.
**Deliverable:** one-page decision memo with a leaderboard table.
**Acceptance:** distinguishes competition from clinical performance; cites rules; names at least one unmeasured harm.
**Stretch:** Re-score a winning solution under a patient-level split or a fairness constraint.

### Lab 21.7 — Tool stack bake-off

**Objective:** Choose a stack for a defined task and justify it on reproducibility, cost, and risk.
**Data:** A small task from Labs 21.1–21.4.
**Steps:** Define modality, scale, latency, team skills, budget, and governance. Compare two stacks (DuckDB + Python vs PostgreSQL + dbt; MONAI vs nnU-Net; Scanpy vs Seurat). Implement the same small analysis in both. Measure setup time, runtime, memory, and lines of code. Record license and supply-chain risk. Recommend one and state the switching cost.
**Deliverable:** comparison table and a 400-word recommendation.
**Acceptance:** both stacks run; metrics measured, not estimated; recommendation names when the other stack would win.
**Stretch:** Containerize both and compare image size and cold-start time.

## 19. Assignment: dataset and benchmark dossier

**Scenario.** You are data lead on a new product team with a proposed claim, target population, deployment setting, and 12-month timeline. Decide whether available data can support the claim, and design the evaluation that would convince an internal review board and an external regulator or payer.

**Deliverable:** 2,500–3,000 word dossier plus a repository.

1. **Decision and claim.** State the decision, population, setting, exact claim, and the counterfactual — what happens today without the model.
2. **Dataset selection.** At least three candidates scored against the five gates; one rejected with evidence.
3. **Access plan.** Credentialing steps, DUA or contract, latency, cost, responsible person, and a fallback if access is denied.
4. **License analysis.** Clause-by-clause mapping to training, evaluation, publication, model release, and deployment.
5. **Data card.** For the selected dataset, in Datasheets for Datasets structure.
6. **Benchmark design.** Label definition and provenance, split strategy, metric tied to decision costs, subgroup and calibration plans, contamination checks.
7. **Baseline and ceiling.** Simple baseline, strong baseline, and a human or label-noise ceiling with justification.
8. **Validity threats.** At least five, each with detection method and mitigation.
9. **What the data cannot support.** Explicit prohibited claims.
10. **Monitoring and retirement.** Drift metrics, re-validation triggers, and retirement conditions.
11. **Reproducibility package.** Code, environment, split hashes, retention plan.

**Constraints:** no claim beyond what the data can measure; every size, license, and access statement cites a primary source with an access date; uncertain facts marked "verify" with the item to check; include one negative result or failed baseline.

## 20. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision and claim framing | 15% | Vague goal; no counterfactual | Clear claim and population | Claim bounded by decision, costs, and evidence standard |
| Dataset selection | 20% | Picks by convenience | Compares across gates | Quantifies trade-offs; rejection justified with evidence |
| Access and license | 20% | Ignores constraints | Identifies DUA, fee, latency | Clause-level mapping to product plan; fallback designed |
| Benchmark validity | 20% | Reports a metric without provenance | States splits and labels | Metric tied to decision; leakage and contamination tested |
| Reproducibility | 10% | Cannot be rerun | Code and environment shared | Hashes, seeds, CIs, independent rerun demonstrated |
| Bias and equity | 10% | Generic mention | Subgroup metrics | Links gaps to access, labels, and deployment harm |
| Communication | 5% | Disorganized | Clear and complete | Executive-ready; candid, specific limitations |

**Pass:** all sections present; no fabricated facts; one rejected dataset with evidence; one quantified validity threat. **Distinction:** would survive internal model-risk review without major revision, and the benchmark design is executable by another team.

## 21. Regulatory, ethical, and commercial lens

| Regime or instrument | Where it binds | Engineering artifact |
|---|---|---|
| **HIPAA Privacy and Security Rules** (US) | Acquisition, de-identification, cloud processing | DUA, BAA, risk analysis, access logs |
| **GDPR** (EU) | Lawful basis, minimization, transfer, rights | DPIA, processing record, retention schedule |
| **Common Rule / IRB** (US) | Human-subjects research, secondary use | Protocol, consent, approval |
| **FDA device regulation** (US) | If output informs diagnosis or treatment | Predetermined change control plan, validation evidence, labeling |
| **EU AI Act and MDR/IVDR** | High-risk AI and devices in the EU | Risk management file, technical documentation, postmarket monitoring |
| **HTI-1 / USCDI** (US) | Certified health IT with predictive decision support | Source attributes, intended use, validation, fairness information |
| **CMS-0057-F** (US) | Payer prior authorization APIs and timelines | API conformance, turnaround measurement |
| **ICH E6 / GCP and 21 CFR Part 11** | Trial data and computerized systems | Audit trails, validation, e-signatures |

**Equity.** Check who is in the dataset and who is missing; report subgroup performance with confidence intervals; beware high AUC in a small subgroup that reflects access bias; audit labels for proxy discrimination through cost, utilization, or adherence; include language, disability, and neurodiversity where the product touches communication or access.

**Commercial.** Access latency is a cost; a six-month credentialing delay can exceed the value of a marginal accuracy gain. License restrictions can foreclose a product line. Benchmark performance does not predict procurement: buyers ask for workflow fit, integration, liability, and support. Total cost includes storage, compute, re-identification review, audit, and re-validation after drift. A dataset that cannot be refreshed becomes a liability, so plan versioning from day one.

## 22. Failure modes and red flags

1. Dataset-first thinking: the project starts with "we have MIMIC," not with a decision.
2. License blindness: non-commercial terms discovered after the model is built.
3. Benchmark worship: a leaderboard delta presented as clinical utility.
4. Split contamination: random splits on patient-correlated data.
5. Label laundering: report-derived or LLM-derived labels treated as truth.
6. Single-site generalization: one hospital used to claim national performance.
7. Hidden test reuse: the same test set tuned against for months.
8. Metric substitution: AUROC reported when the decision needs precision at capacity.
9. No calibration: discrimination without reliability or decision curves.
10. Subgroup silence: no site, sex, age, race, or device breakdown.
11. Synthetic validation: synthetic data used to claim clinical performance.
12. Credential sprawl: shared accounts or data copied outside the governed environment.
13. Unpinned environments: results that cannot be reproduced after a dependency update.
14. PHI in logs: prompts, features, or errors written to an ungoverned store.
15. No retirement plan: data or model kept after its population or label definition changed.

## 23. Interview and discussion questions

1. **Walk me through choosing a dataset for a readmission model.** Hint: define decision and prediction time; compare EHR, claims, and health-system data; check post-discharge capture and label definition; then access and license.
2. **What can claims data never tell you?** Hint: severity, labs, vitals, adherence, out-of-network care, reason for discontinuation.
3. **MIMIC-IV or eICU for a sepsis early-warning model?** Hint: MIMIC has depth but one site; eICU has breadth but sparse variables. Choose by whether the claim is methods or transportability, and validate externally either way.
4. **How would you detect benchmark contamination in an LLM evaluation?** Hint: date-based holdout, canary strings, paraphrase sensitivity, nearest-neighbor search, private holdout.
5. **A vendor reports 0.95 AUROC on a public dataset. What do you ask?** Hint: split strategy, label source, external validation, calibration, subgroups, test reuse, preprocessing fit scope.
6. **What does a non-commercial license prevent in a product company?** Hint: training, evaluation, internal demos, model release, and derived weights may each be restricted; read the clause.
7. **How do you choose between UK Biobank and All of Us?** Hint: phenotype depth and linkage versus diversity and genomic scale; estimand, consent, latency, return of results.
8. **Why is a random split dangerous in imaging?** Hint: same patient, study, scanner, and acquisition on both sides; split by patient and site.
9. **What is the difference between a data card and a datasheet?** Hint: overlapping documentation; datasheets emphasize motivation, composition, collection, preprocessing, uses, distribution, maintenance.
10. **When is synthetic data acceptable?** Hint: pipeline development, schema testing, privacy-safe demos, augmentation with validation; never clinical performance or fairness claims.
11. **How would you evaluate an EHR foundation model?** Hint: few-shot and linear probe, calibration with limited labels, cross-institution transport, subgroups, and a simple baseline; EHRSHOT and FoMoH are starting points.
12. **What would make you tell a team not to proceed?** Hint: no dataset contains the outcome at decision time; the license forbids the product; access latency exceeds the window; only evidence is a contaminated leaderboard.
13. **How do you report a benchmark result to a regulator?** Hint: versioned data, prespecified analysis, patient-level splits, confidence intervals, subgroup and calibration results, contamination checks, reproducible artifacts.
14. **What is access bias, and how does it show up in metrics?** Hint: people with more access are over-represented; a subgroup can show high AUC because the label reflects detection, not disease.
15. **Which tool would you keep, and which would you drop?** Hint: keep Git plus a pinned environment; drop anything that cannot be versioned or audited. The answer should reveal priorities, not brand loyalty.

## 24. Key terms

| Term | Definition |
|---|---|
| **Benchmark** | A fixed task, dataset, and metric used to compare methods. |
| **Benchmark contamination** | Test data or near-duplicates present in model training data. |
| **Calibration** | Agreement between predicted probabilities and observed frequencies. |
| **CITI training** | Research ethics and data protection training required by many repositories. |
| **Data card** | Structured documentation of a dataset's composition, provenance, and intended use. |
| **Data use agreement** | Contract governing access, use, retention, and redistribution. |
| **dbGaP** | US controlled-access archive for genomic and health data. |
| **Decision curve analysis** | Net benefit across threshold probabilities, incorporating harms of false positives and negatives. |
| **Differential privacy** | Formal guarantee bounding the influence of any individual record. |
| **External validation** | Evaluation on data from a different site, time, or population. |
| **Federated learning** | Training across sites without centralizing raw data. |
| **Five-gate framework** | Decision, population, access, license, and benchmark fit. |
| **Label provenance** | The origin and process by which a target label was created. |
| **Leakage** | Information in training unavailable at prediction time, or overlap between splits. |
| **Licensed access** | Contract-based access with field-of-use and redistribution limits. |
| **OMOP CDM** | Common data model for standardized observational health data. |
| **PhysioNet credentialing** | Identity, training, and DUA process for sensitive clinical datasets. |
| **Precision at capacity** | Fraction of true positives among the top-k ranked when capacity is fixed. |
| **Registered access** | Account-based access with click-through terms. |
| **Reproducibility package** | Code, environment, data pointers, and split definitions sufficient to rerun an analysis. |
| **Site shift** | Distribution change across hospitals, scanners, or workflows. |
| **Subgroup analysis** | Performance reported by demographic, clinical, or site strata. |
| **Synthetic data** | Artificially generated records; useful for pipelines, not clinical validation. |
| **Temporal leakage** | Use of future information to predict an earlier event. |
| **Test set reuse** | Repeated evaluation on the same held-out data, inflating performance. |

## 25. Sources and further reading

All URLs accessed 24 October 2026. Sizes and licenses change; verify at the primary source before use in a contract or submission.

**EHR, claims, ICU.**
- MIMIC-IV: https://mimic.mit.edu/ and https://physionet.org/content/mimiciv/ | Demo: https://physionet.org/content/mimic-iv-demo/ | ED: https://physionet.org/content/mimic-iv-ed/ | Note: https://physionet.org/content/mimic-iv-note/
- eICU: https://eicu-crd.mit.edu/ and https://physionet.org/content/eicu-crd/ | HiRID: https://physionet.org/content/hirid/ | AmsterdamUMCdb: https://amsterdammedicaldatascience.nl/
- UK Biobank: https://www.ukbiobank.ac.uk/ and https://www.ukbiobank.ac.uk/enable-your-research/costs | All of Us: https://allofus.nih.gov/ and https://www.researchallofus.org/ | Epic Cosmos: https://cosmos.epic.com/ (vendor figures; verify)
- EHRSHOT: https://ehrshot.stanford.edu/ | FoMoH: https://arxiv.org/abs/2505.16941 | Synthea: https://synthetichealth.github.io/synthea/ | SynPUF: https://data.cms.gov//medicare-claims-synthetic-public-use-files | OMOP: https://ohdsi.github.io/CommonDataModel/ | Eunomia: https://github.com/OHDSI/Eunomia

**Imaging and pathology.**
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/ | MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/ | CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/ | ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC and https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest | RSNA: https://www.rsna.org/artificial-intelligence
- BraTS: https://www.med.upenn.edu/cbica/brats/ and https://www.synapse.org/brats | CAMELYON: https://camelyon17.grand-challenge.org/ | PANDA: https://panda.grand-challenge.org/data/ and https://doi.org/10.1038/s41591-021-01620-2 | MIDOG: https://midog2025.deepmicroscopy.org/ | TCIA: https://www.cancerimagingarchive.net/
- MONAI: https://monai.io/ and https://github.com/Project-MONAI/model-zoo | nnU-Net: https://github.com/MIC-DKFZ/nnUNet | OpenSlide: https://openslide.org/

**Signals and wearables.**
- PhysioNet: https://physionet.org/ | PTB-XL: https://physionet.org/content/ptb-xl/ | Challenge 2025: https://moody-challenge.physionet.org/2025/ | Challenge 2026: https://moody-challenge.physionet.org/2026/ | Sleep-EDF: https://physionet.org/content/sleep-edfx/
- Cholec80: https://camma.unistra.fr/datasets/ | JIGSAWS: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/ | dVRK: https://dvrk.readthedocs.io/ | SurRoL: https://github.com/med-air/SurRoL | OhioT1DM: https://webpages.charlotte.edu/rbunescu/ohiot1dm.html
- TUH EEG: https://www.isip.piconepress.com/projects/tuh_eeg/ | MIMIC-IV waveform: https://physionet.org/content/mimic4wdb/ | Icentia11K: https://physionet.org/content/icentia11k-continuous-ecg/ | WESAD: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection | NSRR: https://sleepdata.org/
- WFDB: https://physionet.org/content/wfdb-python/ | MNE: https://mne.tools/ | NeuroKit2: https://neuropsychology.github.io/NeuroKit/

**Omics and targets.**
- TCGA: https://www.cancer.gov/ccg/research/genome-sequencing/tcga and https://portal.gdc.cancer.gov/ | GTEx: https://gtexportal.org/ | gnomAD: https://gnomad.broadinstitute.org/ | 1000 Genomes: https://www.internationalgenome.org/
- CELLxGENE: https://cellxgene.cziscience.com/ | HCA: https://www.humancellatlas.org/ | GEO: https://www.ncbi.nlm.nih.gov/geo/ | BioStudies: https://www.ebi.ac.uk/biostudies/ | PRIDE: https://www.ebi.ac.uk/pride/ | MetaboLights: https://www.ebi.ac.uk/metabolights/
- DepMap: https://depmap.org/ | LINCS: https://clue.io/lincs | Open Targets: https://www.opentargets.org/ | ChEMBL: https://www.ebi.ac.uk/chembl/ | PubChem: https://pubchem.ncbi.nlm.nih.gov/ | CAGI7: https://genomeinterpretation.org/cagi7-challenge.html | Open Problems: https://openproblems.bio/
- Scanpy: https://scanpy.readthedocs.io/ | Seurat: https://satijalab.org/seurat/ | scVI-tools: https://scvi-tools.org/

**Clinical text and LLM benchmarks.**
- HealthBench: https://openai.com/index/healthbench/ (verify release and license) | MedAgentBench: https://stanfordmlgroup.github.io/projects/medagentbench/ | MedQA: https://github.com/jind11/MedQA | MedMCQA: https://medmcqa.github.io/ | PubMedQA: https://pubmedqa.github.io/
- MedHELM: https://medhelm.org/ | CURE-Bench: https://curebench.ai/ (verify) | i2b2/n2c2: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/ | Hugging Face: https://huggingface.co/ | Inspect: https://inspect.aisi.org.uk/ | HELM: https://crfm.stanford.edu/helm/

**Public health, pharma, operations.**
- NHANES: https://www.cdc.gov/nchs/nhanes/ | NHIS: https://www.cdc.gov/nchs/nhis/ | BRFSS: https://www.cdc.gov/brfss/ | CDC WONDER: https://wonder.cdc.gov/ | NWSS: https://www.cdc.gov/wastewater/ | SEER: https://seer.cancer.gov/ | GLOBOCAN: https://gco.iarc.who.int/
- Care Compare: https://www.medicare.gov/care-compare/ | Open Payments: https://openpaymentsdata.cms.gov/ | HCUP: https://hcup-us.ahrq.gov/ | HCRIS: https://data.cms.gov//cost-reports
- ClinicalTrials.gov: https://clinicaltrials.gov/ | AACT: https://aact.ctti-clinicaltrials.org/ | CTIS: https://euclinicaltrials.eu/ | OpenFDA: https://open.fda.gov/ | AEMS: https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems
- VAERS: https://vaers.hhs.gov/ | EudraVigilance: https://www.ema.europa.eu/en/human-regulatory-overview/research-development/pharmacovigilance-research-development/eudravigilance | VigiBase: https://www.who-umc.org/vigibase/ | MAUDE: https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfMAUDE/search.CFM
- Sentinel: https://www.sentinelinitiative.org/ | Vivli: https://vivli.org/ | YODA: https://yoda.yale.edu/ | TriNetX: https://trinetx.com/

**Methods and documentation.**
- Datasheets for Datasets: https://arxiv.org/abs/1803.09010 | Model Cards: https://arxiv.org/abs/1810.03993 | TRIPOD+AI: https://www.bmj.com/content/385/bmj-2023-078378 | PROBAST: https://www.probast.org/ | EQUATOR: https://www.equator-network.org/
- Fairlearn: https://fairlearn.org/ | AIF360: https://github.com/Trusted-AI/AIF360 | DoWhy: https://py-why.github.io/dowhy/ | EconML: https://econml.azurewebsites.net/ | MLflow: https://mlflow.org/ | DVC: https://dvc.org/ | OpenDP: https://opendp.org/ | NVFlare: https://nvidia.github.io/NVFlare/ | MedPerf: https://medperf.org/
- Johnson et al., MIMIC-IV, *Scientific Data* 10, 1 (2023): https://doi.org/10.1038/s41597-022-01899-x

> **Next:** [Module 19](19_capstone_hackathons.md) uses these resources to scope capstone and hackathon work. [Module 22](22_sources_and_further_reading.md) collects the full source list. Return to Section 15 before writing any claim that names a dataset.
