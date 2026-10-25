# Module 23 — Specialty Electives: Oncology, Cardiology, Pathology, Critical Care, Pediatrics/OB, and Public Health

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 18–24 hours total: six 3–4 hour primers (2.5 h seminar plus 1 h lab clinic each), plus one assessed lab track
> **Track fit:** Advanced elective across Health Systems, Biopharma and Translational, and MedTech, Sensors and Imaging
> **Prerequisites:** Modules 01, 01A, 06, 08. Add 10 for Pathology, 12 for Oncology, 11 for Cardiology, and 04–05 for every primer
> **Tools:** Python, SQL, pandas, scikit-learn or PyTorch, lifelines or scikit-survival, WFDB, OpenSlide, pydicom, statsmodels, DuckDB
> **Industry question:** "Which specialty decision does this work change, who pays for it, and which specialty-specific bias will invalidate the evidence first?"

---

## 1. Why this module exists

General competence gets a data scientist into the room. Specialty fluency gets the work taken seriously. Each specialty has its own definition of an event, its own time zero, its own coding conventions, its own regulatory path, and its own way of producing convincing but wrong results.

The six primers are not a tour of body parts. Each is a compressed industry apprenticeship:

| Primer | Decision class | Signature dataset | Signature pitfall | Typical buyer |
|---|---|---|---|---|
| Oncology | Who gets which therapy, when to escalate, whom to screen | TCGA/GDC, SEER, GENIE, NCDB | Immortal time and surrogate endpoints | Pharma, payer, cancer center |
| Cardiology | Who needs monitoring, anticoagulation, or device therapy | PTB-XL, MIMIC-IV-ECG, EchoNet | Prevalence-dependent PPV and device-specific signals | Device maker, health system, payer |
| Pathology | Is this cancer, what grade, which biomarker | CAMELYON, PANDA, TCGA WSI | Slide and scanner leakage | Lab, diagnostics company, pharma |
| Critical care | Who is deteriorating, who needs escalation | MIMIC-IV, eICU, PhysioNet 2019 | Label leakage and alarm burden | Hospital, EHR vendor |
| Pediatrics and OB | What dose, what risk, when to deliver | CTU-UHB, NHANES, PHIS, vital records | Weight-based dosing and rare events | Medicaid, children's hospital, device maker |
| Public health | Where is the outbreak, whom to protect | NWSS, NHSN, NNDSS, BRFSS | Denominators and surveillance lag | Government, insurer, NGO |

**Selection principle.** Choose an elective by the decision you want to own, not by the model you want to train. The decision determines the buyer, the data access, the endpoint, and the evidence standard. A beautiful model aimed at a decision nobody budgets for is a portfolio exercise, not a product.

## 2. How each primer is built

Every primer follows the same eight-part structure so that comparisons remain honest:

1. The decision and who pays.
2. Data sources and standards.
3. Core methods and endpoints.
4. Three to five real use cases with sources.
5. Regulatory, ethical, and equity issues.
6. One lab with acceptance criteria.
7. Eight to ten interview questions with hints.
8. Sources with URLs.

**Timing for a 3–4 hour primer.** 60 minutes on decisions and payment; 45 minutes on data and standards; 45 minutes on endpoints and methods; 45 minutes on cases; 30–45 minutes on the lab brief and critique. Participants complete the lab outside the session. A cohort that completes three primers has enough depth for a specialty capstone; one primer is enough for informed conversation.

**Assessment model.** Each primer can be taken for a pass/fail lab credit or accumulated into the elective artifact described in Section 13. The lab is the assessable unit; the interview questions are formative.

## 3. Learning outcomes

By the end of this module, you can:

- Map a specialty decision to its decision-maker, economic buyer, and harm of error.
- Select a specialty dataset on decision fit, population fit, access, license, and benchmark fit, not on download convenience.
- Define time zero, eligibility, endpoint, and censoring for a specialty cohort and defend each choice.
- Identify the signature bias of each specialty and design a test that would reveal it.
- Interpret specialty endpoints (OS, PFS, MRD, MACE, QWK, SOFA, acidemia, Rt) without collapsing them into generic accuracy.
- Build and calibrate a model appropriate to the specialty's prevalence and decision threshold.
- Explain the regulatory path for a specialty product: device, diagnostic, lab-developed test, drug label, or public-health authority.
- Recognize equity and workflow failures specific to each specialty, including access, language, pregnancy, disability, and rural care.
- Critique a published specialty AI study for leakage, transportability, and endpoint validity.
- Choose an elective that matches a realistic career and data-access strategy.

## 4. The common architecture

Every specialty primer can be reduced to the same chain. The differences are in the nouns.

```
Eligible population
   → time zero (diagnosis, admission, test order, birth, symptom onset, sample collection)
   → features available at decision time
   → predicted state or effect
   → action (treat, image, biopsy, alert, isolate, allocate)
   → endpoint (survival, response, detection, deterioration, birth outcome, cases)
   → payment and regulation
```

**Five questions before any specialty model.**

| Question | Why it decides the project |
|---|---|
| What is the exact action, and who takes it? | A model without an action has no user or budget |
| What is time zero, and what is knowable then? | Most leakage is a time-zero error, not a modeling error |
| What endpoint does the buyer already trust? | Replacing a trusted endpoint requires more evidence, not less |
| What does a false positive and a false negative cost? | Sets the threshold and the acceptable PPV |
| Which specialty-specific bias is most likely? | Determines the audit you must run before claiming success |

**Cross-cutting rule.** A specialty model is a claim about a workflow. If the workflow is not described, the model cannot be validated, priced, or regulated.

---

## 5. Oncology

### 5.1 The decision and who pays

Oncology is a sequence of high-stakes, expensive, and time-sensitive decisions. Data science enters at every step, but the buyer differs by step.

| Decision | Decision-maker | Economic buyer | Cost of a false positive | Cost of a false negative |
|---|---|---|---|---|
| Screening eligibility and interval | Primary care, screening program | Payer, health system | Biopsy, anxiety, overdiagnosis | Delayed diagnosis, advanced stage |
| Diagnosis and grade | Pathologist, radiologist | Hospital, lab | Unnecessary surgery | Missed cancer |
| Staging and risk | Multidisciplinary tumor board | Cancer center | Overtreatment | Undertreatment |
| Therapy selection | Oncologist | Payer, pharma | Toxicity, cost, ineffective drug | Lost survival benefit |
| Trial matching | Research coordinator | Pharma, academic center | Screen failure cost | Missed trial access |
| Surveillance | Oncologist | Payer | Unnecessary imaging | Delayed recurrence detection |
| Prior authorization | Payer | Payer | Denied appropriate care | Paid for low-value care |

Payment mechanics matter. In the United States, Medicare Part B pays for physician-administered drugs at average sales price plus a margin, while hospital outpatient drugs are paid through the outpatient prospective payment system. Prior authorization, site-of-care rules, and 340B pricing all change whether a test or drug is adopted. A biomarker model that improves outcomes but adds an unreimbursed assay may still fail commercially.

**Industry interpretation.** Oncology data science is rarely paid for directly. It is paid for through drug margin, diagnostic reimbursement, clinical trial efficiency, or avoided escalation. State which one your project improves.

### 5.2 Data sources and standards

| Source | Contents | Access | Oncology-specific caveat |
|---|---|---|---|
| **TCGA via GDC** | About 11,000 patients across 33 cancer types; clinical, mutation, expression, copy number, methylation, pathology images | Open clinical and somatic data; germline controlled through dbGaP | Bulk tumor mixtures; treatment and stage confounding; center batch effects; survival follow-up varies |
| **SEER** | Population-based cancer registries covering roughly half the US population (verify current coverage); incidence, stage, survival, mortality | Public and SEER*Stat registration | Registry stage and treatment fields are coarse; no drug-level therapy; survival depends on linkage |
| **SEER-Medicare** | SEER linked to Medicare claims | Restricted-access DUA | Claims identify covered beneficiaries, not all care; older population; coding drift |
| **NCDB** | Hospital-based registry from Commission on Cancer programs | Application and agreement | Hospital selection; abstracted treatment fields; not population-representative |
| **AACR Project GENIE** | Clinico-genomic data from participating centers; over 100,000 samples (verify current release) | Registered access | Targeted panels differ by center; no uniform follow-up; ascertainment by testing |
| **Flatiron Health** | De-identified longitudinal oncology EHR data | Commercial license | Vendor-defined variables; site mix; informative missingness |
| **TCIA** | Cancer imaging collections linked to clinical data | Open or registered per collection | Annotation quality varies; scanner and protocol shift |
| **cBioPortal** | Curated genomic and clinical datasets | Open | Reuse of published cohorts; overlapping patients across studies |

**Standards.** AJCC TNM staging (8th edition is widely used; confirm the 9th-edition rollout in your jurisdiction), ICD-O-3 morphology, RECIST 1.1 and iRECIST for response, WHO classification of tumors, mCODE (HL7 FHIR minimal Common Oncology Data Elements), the OMOP oncology extension, CDISC SDTM and ADaM oncology domains, NAACCR registry fields, LOINC, SNOMED CT, and DICOM for imaging and pathology.

**Data-quality checks.** Verify stage edition and date; distinguish clinical from pathologic stage; confirm whether "tumor size" is invasive or in situ; reconcile vital status across registry, EHR, and claims; check that treatment start dates follow diagnosis; and inspect whether biomarker results are missing at random or missing because testing was never ordered.

### 5.3 Core methods and endpoints

| Endpoint | Definition | Window | Ascertainment risk |
|---|---|---|---|
| Overall survival (OS) | Death from any cause | Diagnosis or randomization to death | Complete only with death registry linkage; otherwise censored |
| Progression-free survival (PFS) | Progression or death | Randomization to event | Imaging schedule and assessment drift |
| Disease-free survival (DFS) | Recurrence or death after curative treatment | Surgery or randomization | Surveillance intensity affects detection |
| Event-free survival (EFS) | Progression, recurrence, death, or specified event | Trial-defined | Composite definition varies by trial |
| Objective response rate (ORR) | Complete or partial response | Usually 6–12 weeks | Measurable disease requirement; reader variability |
| Duration of response (DoR) | Response to progression or death | Responders only | Conditional on response; not a population effect |
| Pathologic complete response (pCR) | No invasive tumor at surgery | Neoadjuvant setting | Surrogate validity differs by tumor type |
| Minimal residual disease (MRD) | ctDNA or cell-based residual signal | Post-treatment | Assay sensitivity and threshold differ |
| Real-world PFS (rwPFS) | Progression or death from EHR | Start of therapy | Lines of therapy and scan intervals are inferred |

**Immortal time bias.** The classic oncology error is to classify patients as "treated" at diagnosis and compare them with untreated patients. Treated patients must survive long enough to receive treatment, so the exposure group is guaranteed a period of survival. The result is a spuriously protective hazard ratio.

Three defensible fixes:

- **Landmark analysis:** set a landmark (for example, 90 days after diagnosis) and classify exposure only among patients alive at the landmark.
- **Time-dependent covariate:** treatment becomes a time-varying exposure in a Cox model.
- **Clone-censor-weight:** clone each patient into treated and untreated strategies, censor at deviation, and weight by inverse probability of censoring.

| Approach | What it estimates | Assumption | Common failure |
|---|---|---|---|
| Naive grouped comparison | Association with a misclassified exposure | None credible | Guaranteed immortal time |
| Landmark | Survival conditional on surviving to landmark | Landmark is clinically meaningful; no exposure effect before landmark | Drops early deaths; landmark chosen post hoc |
| Time-dependent Cox | Effect of treatment while on treatment | No unmeasured time-varying confounding | Treatment switching and informative censoring |
| Clone-censor-weight | Effect of a treatment strategy | Correct censoring model; positivity | Complex weighting; unstable with small groups |

**Other oncology-specific biases.** Lead-time bias makes earlier detection look like longer survival even when the date of death is unchanged. Length-time bias favors screening for slow-growing tumors. Overdiagnosis inflates survival and prevalence without improving mortality. Stage migration (the Will Rogers phenomenon) improves stage-specific survival when imaging becomes more sensitive, without any patient living longer.

**Methods that matter.** Kaplan-Meier estimation with risk tables; Cox models with proportional-hazards checks; Fine-Gray competing risks when other-cause death is substantial; landmark and time-dependent analyses; target trial emulation for real-world treatment comparisons; external control arms with explicit positivity and comparability checks; multiple imputation for missing stage and biomarker data; and calibration of survival predictions at clinically used horizons (1, 3, 5 years).

### 5.4 Use cases

**1. Tissue-agnostic approval of pembrolizumab (FDA, 2017).** The FDA granted accelerated approval for pembrolizumab in adult and pediatric patients with unresectable or metastatic MSI-H or dMMR solid tumors, regardless of tumor site. The evidence came from a small basket trial across multiple tumor types, with ORR and duration of response as endpoints. What transferred: biomarker-defined eligibility can cross histology. What did not: the single-arm design and surrogate endpoint required confirmatory evidence, and not every MSI-H tumor responds.

**2. TMB-high approval with a companion diagnostic (FDA, 2020).** Pembrolizumab was approved for TMB-high solid tumors using FoundationOne CDx as the companion diagnostic. The case shows how a biomarker threshold, an assay, and a drug label become a single commercial package. The data-science issue is not only the classifier but the assay's reproducibility, the cut point's validation, and the fact that TMB is not comparable across panels.

**3. ctDNA-guided adjuvant therapy in stage II colon cancer (DYNAMIC, Tie et al., NEJM 2022).** Patients were randomized to ctDNA-guided management or standard clinicopathology-guided management. The ctDNA arm received less adjuvant chemotherapy without a significant reduction in recurrence-free survival. What transferred: a biomarker can safely de-escalate treatment when the endpoint is a non-inferiority question. What did not: the result is specific to the assay, population, and follow-up; it does not establish benefit for escalation in ctDNA-positive patients.

**4. AI-supported mammography screening (MASAI, Lång et al., Lancet Oncology 2023).** In a Swedish randomized screening trial of about 80,000 women, AI-supported reading increased cancer detection without increasing recall compared with standard double reading. The trial measured screening accuracy, not mortality. What transferred: AI can be evaluated as a reader-support tool within an existing program. What did not: detection is not mortality benefit, and the program, equipment, and double-reading baseline differ elsewhere.

**5. Real-world genomic cohort building (AACR Project GENIE).** GENIE aggregates clinico-genomic data across cancer centers to support research on resistance, rarity, and outcomes. Its value is scale and breadth; its limitation is that each center tests different patients with different panels, so incidence and treatment-effect estimates are not population estimates. Use it for discovery and hypothesis generation, not for prevalence claims.

### 5.5 Regulatory, ethical, and equity issues

- **Regulatory path.** A diagnostic or prediction model is usually a medical device (510(k), De Novo, or PMA) or, in the US, may be offered as a laboratory-developed test under CLIA. The status of the FDA's 2024 LDT final rule was vacated in litigation in 2025; confirm the current enforcement position before building a business model around it. In the EU, an in-vitro diagnostic falls under IVDR; an AI component may also engage the EU AI Act.
- **Drug-label evidence.** A model that guides therapy may be regulated as a companion diagnostic, requiring analytical validation, clinical validation, and a link to the drug's approved indication.
- **Consent and tissue use.** Secondary research use of tissue and genomic data is governed by consent, institutional policy, and data-use agreements. Broad consent is not unlimited consent.
- **Equity.** Screening access, biomarker testing rates, trial enrollment, and genomic reference panels skew toward insured, urban, and European-ancestry populations. A model trained on TCGA may not transport to a safety-net cohort. Report testing rates by race, ethnicity, language, insurance, and rurality; a model cannot fix a test that is never ordered.
- **Genetic information.** GINA limits some uses of genetic information in employment and health insurance in the US but does not cover life, disability, or long-term-care insurance. Incidental findings require a return-of-results policy.
- **Financial toxicity.** A more sensitive test can increase spending without improving outcomes. Include cost per additional case detected and per quality-adjusted life-year in the value case.

### 5.6 Lab: Endpoint audit and survival analysis in lung cancer

**Objective.** Build a defensible survival analysis, demonstrate immortal time bias, and compare naive and landmark estimates.

**Data.** SEER*Stat public incidence data (lung and bronchus, 2000–2022) and GDC TCGA-LUAD open clinical and mutation files. If treatment fields are too sparse for a credible comparison, use a clearly labeled teaching simulation to demonstrate the bias mechanism; do not present simulated treatment effects as evidence.

**Steps.**

1. Extract SEER cases with age, sex, stage, histology, diagnosis year, survival months, and vital status.
2. Compute 5-year relative survival by stage for two diagnosis eras. Interpret the change as a mix of real progress, stage migration, and lead time, not automatically as treatment effect.
3. Download TCGA-LUAD clinical data; define time zero as diagnosis date; exclude missing or negative follow-up.
4. Fit a Cox model for OS with age, sex, and stage; test proportional hazards; report hazard ratios with 95% confidence intervals.
5. Fit a Fine-Gray model for cancer-specific death with other-cause death as a competing event.
6. Build a naive treated-versus-untreated comparison, then repeat with a 90-day landmark and a time-dependent exposure. Document how the estimate changes and why.
7. Calibrate predicted 1-, 3-, and 5-year survival; report observed versus expected by decile.

**Deliverable.** A four-page endpoint audit plus a reproducible notebook and a one-page model card.

**Acceptance criteria.**

- Time zero is explicit and identical for every patient.
- No feature is measured after time zero; a timestamp audit is included.
- Landmark and time-dependent results are both reported.
- Competing-risk analysis is reported when other-cause mortality exceeds 10%.
- All estimates include 95% confidence intervals.
- The notebook runs end to end from a pinned environment and a fixed seed.
- The write-up distinguishes association from causation and names at least three residual biases.

**Stretch.** Repeat the analysis in a second cancer type and show whether the direction of the naive bias is the same.

### 5.7 Interview and discussion questions

| Question | Hint for the instructor |
|---|---|
| Why is a treated-versus-untreated comparison at diagnosis usually biased? | Look for immortal time: treatment cannot start before diagnosis, and patients must survive to be treated. |
| When is a landmark analysis the wrong fix? | When treatment can affect early mortality, when the landmark is chosen after looking at the data, or when the exposure is time-varying. |
| PFS improved but OS did not. What are three explanations? | Surrogate failure, crossover or subsequent therapy, and improved post-progression survival. |
| How would you validate a TMB cutoff? | Analytical reproducibility across panels, clinical validation against response, and independent cohorts; TMB is assay-specific. |
| What is stage migration and why does it inflate survival? | More sensitive imaging reassigns patients to higher stage; stage-specific survival improves without anyone living longer. |
| A vendor reports 95% sensitivity for cancer detection. What do you ask? | Prevalence, spectrum, reference standard, whether the unit is lesion or patient, and whether the test was used as a second read. |
| How do you handle missing biomarker status? | Model the testing process, use multiple imputation with sensitivity analysis, and never assume missing equals negative. |
| What is an external control arm, and when is it credible? | A non-randomized comparator; credible only with a target trial framework, comparable eligibility, and measured confounding, and even then it is not randomization. |
| Which patients are most likely to be harmed by an oncology triage model? | Those with less testing, atypical presentation, or limited access; check subgroup sensitivity and alert rates. |
| How would you price an oncology decision tool? | Link it to drug margin, diagnostic reimbursement, trial efficiency, or avoided escalation, and state who captures the value. |

### 5.8 Sources

- NCI Surveillance, Epidemiology, and End Results Program. https://seer.cancer.gov/ (accessed 24 Oct 2026).
- NIH Genomic Data Commons. https://portal.gdc.cancer.gov/ (accessed 24 Oct 2026).
- AACR Project GENIE. https://aacrprojectgenie.org/ (accessed 24 Oct 2026).
- American College of Surgeons National Cancer Database. https://www.facs.org/quality-programs/cancer-programs/national-cancer-database/ (accessed 24 Oct 2026).
- HL7 FHIR mCODE Implementation Guide. https://hl7.org/fhir/us/mcode/ (accessed 24 Oct 2026).
- RECIST guidelines. https://recist.eortc.org/ (accessed 24 Oct 2026).
- FDA. FDA approves first cancer treatment for any solid tumor with a specific genetic feature (2017). https://www.fda.gov/news-events/press-announcements/fda-approves-first-cancer-treatment-any-solid-tumor-specific-genetic-feature (accessed 24 Oct 2026).
- FDA. Pembrolizumab for TMB-H solid tumors (2020). https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-pembrolizumab-adults-and-children-tmb-h-solid-tumors (accessed 24 Oct 2026).
- Tie J, et al. Circulating tumor DNA analysis guiding adjuvant therapy in stage II colon cancer. N Engl J Med. 2022. https://doi.org/10.1056/NEJMoa2200075 (accessed 24 Oct 2026).
- Lång K, et al. Artificial intelligence-supported screen reading versus standard double reading in the MASAI trial. Lancet Oncol. 2023. https://doi.org/10.1016/S1470-2045(23)00298-X (accessed 24 Oct 2026).
- American Joint Committee on Cancer. https://www.facs.org/quality-programs/cancer-programs/american-joint-committee-on-cancer/ (accessed 24 Oct 2026).

---

## 6. Cardiology

### 6.1 The decision and who pays

Cardiology is a high-volume mix of screening, acute triage, chronic disease management, and device therapy. The economics are unusual because a single implant or procedure can dominate the cost of a patient's year.

| Decision | Decision-maker | Economic buyer | Cost of a false positive | Cost of a false negative |
|---|---|---|---|---|
| AF detection in asymptomatic adults | Primary care, cardiology | Payer, consumer device maker | Anticoagulation risk, anxiety, follow-up | Stroke |
| Chest pain rule-out | Emergency physician | Hospital, payer | Unnecessary admission or angiography | Missed MI |
| Heart failure decompensation | Cardiologist, care manager | Health system, payer | Avoidable admission | Emergency presentation |
| ICD or CRT implantation | Electrophysiologist | Payer, device maker | Procedure risk, cost | Sudden death |
| Remote device monitoring | Device clinic | Device maker, payer | Alert fatigue, unnecessary visits | Missed lead failure |
| Statin or anticoagulant initiation | Primary care | Payer | Bleeding, cost | Preventable event |

Payment mechanisms that shape adoption: Medicare remote physiologic monitoring codes (CPT 99453, 99454, 99457, 99458), remote therapeutic monitoring, transitional care management, cardiac rehabilitation, and device-based DRG payments. A wearable that detects AF creates value only if the confirmation pathway and anticoagulation decision are paid for.

**Industry interpretation.** In cardiology, the model is often cheaper than the confirmation test. The business case must include the ECG patch, the clinic visit, and the downstream procedure, not only the algorithm.

### 6.2 Data sources and standards

| Source | Contents | Access | Cardiology-specific caveat |
|---|---|---|---|
| **PTB-XL** | 21,837 clinical 12-lead ECGs, 10 s, 500 Hz, from 18,885 patients, with statement labels | Open on PhysioNet | Statement-derived labels; multi-label records; German population; resting ECGs |
| **MIMIC-IV-ECG** | Large set of 12-lead ECGs linked to MIMIC-IV | PhysioNet credentialed | Single center; machine and technician effects; labels from reports |
| **CODE-15%** | About 300,000 12-lead ECGs from a Brazilian cohort (verify release) | Open with registration | Population and device shift; label quality varies |
| **EchoNet-Dynamic** | 10,030 apical four-chamber echocardiogram videos with ejection fraction and survival | Open with registration | Single academic center; view and device dependence |
| **MIMIC-IV waveform** | ICU waveforms including ECG, arterial pressure, and respiration | PhysioNet credentialed | Device-specific; alignment errors; selected ICU population |
| **UK Biobank** | Resting ECG, cardiac MRI, and linked hospital data in a subset | Application and fee | Healthy-volunteer bias; small imaging subset |
| **CMS claims** | Diagnosis, procedure, device, and readmission data | Licensed or CMS research files | Coding is billing, not physiology; no ECG waveforms |

**Standards.** SCP-ECG for exchange, DICOM waveform for clinical signals, HL7 FHIR Observation and DiagnosticReport, LOINC codes for ECG and echo measures, SNOMED CT for findings, ICD-10-CM for diagnoses and procedures, and AHA/ACC/HRS guideline definitions for clinical endpoints. Device data adds manufacturer-specific formats and firmware versions.

**Signal-quality checks.** Sampling rate and filter settings; lead placement and reversal; baseline wander; muscle artifact; missing leads; calibration pulse; and whether the record is resting, ambulatory, or device-derived. Two ECGs with the same label can be different measurements.

### 6.3 Core methods and endpoints

| Endpoint | Definition | Window | Ascertainment risk |
|---|---|---|---|
| AF detection | First episode confirmed by ECG or patch | Monitoring period | Only detected if monitored; episode duration threshold varies |
| MACE | Composite of death, MI, stroke, or revascularization | 30 days to 5 years | Composite weighting hides component differences |
| Cardiovascular death | Death attributed to cardiac cause | Follow-up | Attribution unreliable without adjudication |
| HF hospitalization | Admission for heart failure | 30 days to 1 year | Coding and observation-status changes |
| Ejection fraction | Ventricular systolic function | Index echo | Measurement variability and vendor dependence |
| QRS duration / morphology | Conduction abnormality | Index ECG | Lead placement and machine interpretation |
| Six-minute walk distance | Functional capacity | 6 minutes | Effort and protocol |
| KCCQ | Patient-reported heart failure symptoms | 4–12 weeks | Responsiveness and missingness |

**Prevalence and PPV.** A screening test with 95% sensitivity and 95% specificity has a positive predictive value near 16% when prevalence is 1%, and near 68% when prevalence is 10%. In a consumer population with 0.5% AF prevalence, most notifications are false positives even with a good model. Always report PPV at the deployment prevalence, not only AUROC.

| Prevalence | Sensitivity | Specificity | PPV | NPV |
|---:|---:|---:|---:|---:|
| 0.5% | 0.95 | 0.95 | 8.7% | 99.97% |
| 1% | 0.95 | 0.95 | 16.1% | 99.95% |
| 5% | 0.95 | 0.95 | 50.0% | 99.72% |
| 20% | 0.95 | 0.95 | 82.6% | 98.7% |

**Verification bias.** If only symptomatic patients receive the reference ECG, the model is evaluated on a selected population and its sensitivity in asymptomatic patients is unknown. Prospective screening studies must define how positives are confirmed and how a sample of negatives is verified.

**Methods.** Signal preprocessing (bandpass filtering, QRS detection, RR interval extraction); feature-based models using heart rate variability and morphology; 1D and 2D convolutional networks; transformer models for multi-lead ECG; video models for echo; calibration with isotonic or Platt scaling; decision-curve analysis for anticoagulation; and competing-risk survival models for cardiac events.

### 6.4 Use cases

**1. Apple Heart Study (Perez et al., NEJM 2019).** About 419,000 participants used a smartwatch irregular-rhythm notification. Roughly 0.5% received a notification; among those who returned an ECG patch, the positive predictive value for AF was about 0.84. What transferred: a consumer device can generate a clinically confirmable signal at scale. What did not: participants were self-selected, confirmation was incomplete, and the study did not show stroke reduction.

**2. Fitbit Heart Study (Lubitz et al., Circulation 2022).** A large consumer cohort used a photoplethysmography-based irregular-rhythm algorithm with patch confirmation. Reported positive predictive value was high, but the population, confirmation protocol, and device firmware limit generalization; verify the current published figure before citing it. What transferred: the confirmatory pathway is the product. What did not: notification accuracy is not a diagnosis.

**3. AI-ECG for low ejection fraction (Attia et al., Nature Medicine 2019).** A convolutional network detected left ventricular systolic dysfunction from a standard 12-lead ECG in a Mayo Clinic cohort. The model creates a screening pathway for echocardiography, not a replacement for imaging. What transferred: a cheap test can triage an expensive one. What did not: single-center performance, prevalent disease, and the need for echo confirmation limit deployment claims.

**4. EchoNet-Dynamic (Ouyang et al., Nature 2020).** A video model estimated ejection fraction and assessed cardiomyopathy from 10,030 echocardiogram videos, with performance comparable to expert review on some tasks. What transferred: video-native modeling avoids hand-crafted measurements. What did not: single-center data, view dependence, and vendor-specific acquisition restrict transportability.

**5. Remote monitoring of implanted devices (TRUST trial, Varma et al., Circulation 2010).** Automatic remote monitoring of implantable cardioverter-defibrillators reduced in-person follow-up visits without a significant increase in adverse events. What transferred: remote data can substitute for some scheduled visits. What did not: the trial evaluated device follow-up, not AI, and it required a structured alert and clinic workflow.

### 6.5 Regulatory, ethical, and equity issues

- **Device regulation.** ECG analysis software and arrhythmia detection are typically regulated as medical devices. Consumer products may qualify for the FDA general wellness policy only if they make no diagnostic claim; a notification that leads to anticoagulation is not wellness.
- **Clinical decision support.** A tool that recommends anticoagulation may be regulated as device software unless it meets CDS criteria. The distinction depends on the intended user, the basis of the recommendation, and whether the clinician can independently review the evidence.
- **Reimbursement.** Remote monitoring requires a prescribing clinician, a defined episode, and documentation. A direct-to-consumer alert without a payment pathway creates cost without revenue.
- **Equity.** PPG accuracy varies with skin pigmentation, perfusion, motion, wrist size, and body composition. ECG screening studies underrepresent rural, low-income, and non-English-speaking populations. A device that requires a smartphone and broadband excludes the patients at highest stroke risk.
- **Privacy.** Continuous cardiac data reveals activity, sleep, stress, pregnancy, and possibly arrhythmia. Consumer terms may permit uses that patients do not expect.
- **Human factors.** Alert fatigue is a safety issue. A cardiology alert that fires several times per patient-day will be ignored, and the resulting false negatives are caused by the workflow, not only the model.

### 6.6 Lab: AF detection with prevalence-adjusted evaluation

**Objective.** Build an AF classifier on 12-lead ECG, split by patient, calibrate it, and report performance at realistic prevalences.

**Data.** PTB-XL (open, ODC Attribution). Optionally repeat on MIMIC-IV-ECG with credentialing.

**Steps.**

1. Load records with WFDB; retain sampling rate, leads, and SCP statement codes.
2. Map statements to a binary AF label and a multi-class rhythm label; document ambiguous and multi-label records.
3. Split by patient ID, not by record. Report the patient and record counts in each split.
4. Preprocess consistently: resample, filter, normalize per lead, and pad or crop to a fixed length.
5. Train a baseline (logistic regression on RR and morphology features) and a 1D CNN; use class weights or a suitable loss.
6. Calibrate on a held-out calibration set; report calibration slope, intercept, and Brier score.
7. Compute sensitivity and specificity at clinically plausible thresholds; compute PPV at 0.5%, 1%, 5%, and 20% prevalence.
8. Report subgroup performance by age and sex; inspect errors for lead reversal, artifact, and label ambiguity.

**Deliverable.** A model card, a calibration plot, and a PPV table at four prevalences.

**Acceptance criteria.**

- Zero patient overlap across train, calibration, and test sets.
- AUROC and AUPRC reported with 95% confidence intervals.
- Calibration slope between 0.8 and 1.2, or a documented reason it cannot be achieved.
- PPV reported at deployment prevalence, not only at the sample prevalence.
- At least one artifact-driven error gallery.
- Preprocessing and model versions pinned; random seed fixed.
- A written statement of the intended use and the confirmation pathway.

### 6.7 Interview and discussion questions

| Question | Hint for the instructor |
|---|---|
| Why can a 95% sensitive AF model still be wrong most of the time? | Low prevalence; PPV depends on prevalence. Compute it explicitly. |
| What is verification bias in a screening study? | Only a selected subset receives the reference standard, so sensitivity is estimated on a biased sample. |
| How would you design a study to show stroke reduction? | Randomized trial with stroke as the endpoint, adequate follow-up, and anticoagulation protocol; detection alone is not enough. |
| A wearable vendor claims "clinically validated." What do you ask? | Population, reference standard, confirmation rate, PPV, regulatory status, and whether the endpoint was detection or outcomes. |
| What changes when the ECG comes from a single lead? | Less morphology information, more motion artifact, different label distribution, and device-specific preprocessing. |
| How do you handle multi-label ECGs? | Multi-label loss, report per-class performance, and do not force a single rhythm label. |
| Why is a composite MACE endpoint dangerous? | Components have different mechanisms and patient importance; a large effect on one can hide harm in another. |
| How do you evaluate an alert that fires 10 times per patient-day? | Alert burden, PPV, time to action, override rate, and the number needed to alert for one true event. |
| What is the reimbursement path for remote cardiac monitoring? | Prescribing clinician, covered code, documented episode, and a workflow that acts on the data. |
| Which patients might a PPG algorithm miss? | Darker skin, low perfusion, motion, small wrists, and older patients with frail skin; demand subgroup validation. |

### 6.8 Sources

- PhysioNet PTB-XL. https://physionet.org/content/ptb-xl/ (accessed 24 Oct 2026).
- PhysioNet MIMIC-IV-ECG. https://physionet.org/content/mimic-iv-ecg/ (accessed 24 Oct 2026).
- EchoNet-Dynamic. https://echonet.github.io/dynamic/ (accessed 24 Oct 2026).
- Perez MV, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. N Engl J Med. 2019. https://doi.org/10.1056/NEJMoa1901183 (accessed 24 Oct 2026).
- Lubitz SA, et al. Detection of atrial fibrillation in a large population using wearable devices: the Fitbit Heart Study. Circulation. 2022. https://doi.org/10.1161/CIRCULATIONAHA.122.060291 (accessed 24 Oct 2026; verify reported PPV).
- Attia ZI, et al. Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram. Nat Med. 2019. https://doi.org/10.1038/s41591-018-0240-2 (accessed 24 Oct 2026).
- Ouyang D, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020. https://doi.org/10.1038/s41586-020-2145-8 (accessed 24 Oct 2026).
- Varma N, et al. Efficacy and safety of automatic remote monitoring for implantable cardioverter-defibrillator follow-up: the TRUST trial. Circulation. 2010. https://doi.org/10.1161/CIRCULATIONAHA.110.937409 (accessed 24 Oct 2026).
- FDA. General wellness policy for low-risk devices. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices (accessed 24 Oct 2026).

---

## 7. Pathology

### 7.1 The decision and who pays

Pathology converts tissue into a diagnosis, a grade, a stage contribution, and increasingly a biomarker that selects therapy. It is the rate-limiting step for most cancer treatment.

| Decision | Decision-maker | Economic buyer | Cost of a false positive | Cost of a false negative |
|---|---|---|---|---|
| Cancer present or absent | Pathologist | Hospital lab, payer | Unnecessary surgery or therapy | Missed cancer |
| Tumor grade | Pathologist | Hospital, pharma | Overtreatment | Undertreatment |
| Margin status | Surgeon and pathologist | Hospital | Re-excision | Local recurrence |
| Biomarker score (HER2, ER, PD-L1) | Pathologist, oncologist | Pharma, lab, payer | Wrong therapy, toxicity | Missed eligible therapy |
| Molecular triage | Pathologist, molecular lab | Lab, payer | Costly reflex testing | Missed actionable alteration |
| Quality control and workload triage | Lab manager | Lab | False reassurance | Delayed diagnosis |

Payment is mainly through professional and technical pathology billing, with hospital global budgets and Medicare Clinical Laboratory Fee Schedule rates. A digital pathology scanner is a capital purchase; an AI module is a recurring software cost. The buyer is often the laboratory, not the oncologist, and the budget cycle is slow.

**Industry interpretation.** In pathology, the unit of adoption is the slide and the scanner, not the patient. A model that works only on one scanner model has a small addressable market.

### 7.2 Data sources and standards

| Source | Contents | Access | Pathology-specific caveat |
|---|---|---|---|
| **CAMELYON16** | About 400 whole-slide images of sentinel lymph nodes with metastasis annotations | Open via Grand Challenge | Two centers; slide-level labels; limited scanner diversity |
| **CAMELYON17** | Multi-center lymph-node slides with patient and stage labels | Open via Grand Challenge | Class imbalance; center effects |
| **PANDA** | 10,616 prostate biopsies with ISUP grade | Kaggle, CC BY-SA-NC 4.0, and Grand Challenge | Inter-pathologist disagreement; grade imbalance |
| **PatchCamelyon (PCam)** | 327,680 small patches derived from Camelyon slides (verify count) | Open | Patch-level labels; tile leakage if split naively |
| **TCGA WSI** | Diagnostic slides for many TCGA cases | Open via GDC | Frozen-section and fixation variation; one slide per case often |
| **TCIA pathology collections** | Annotated WSI and clinical data | Open or registered | Annotation and scanner heterogeneity |
| **MIDOG** | Mitosis detection across scanners | Challenge registration | Domain shift is the point of the benchmark |

**Standards.** DICOM Supplement 145 for whole-slide imaging; vendor formats such as Aperio SVS, Hamamatsu NDPI, and generic pyramidal TIFF; OpenSlide for reading; ICC color profiles; CAP cancer protocols and synoptic reporting; ASCO/CAP HER2 testing guidelines; SNOMED CT and LOINC for observations; and CLIA/CAP accreditation requirements for the laboratory.

**Data-quality checks.** Slide level and magnification; scanner make, model, and software; stain batch and protocol; fixation time; tissue thickness; presence of pen marks, bubbles, folds, and foreign material; and whether the case has one slide or many. A slide identifier can encode the hospital, scanner, or year and become a shortcut.

### 7.3 Core methods and endpoints

| Endpoint | Definition | Unit | Failure mode |
|---|---|---|---|
| Slide-level AUROC | Discrimination of tumor vs normal or grade | Slide | Dominated by easy slides; ignores lesion size |
| Sensitivity at fixed specificity | Detection operating point | Slide or lesion | Prevalence and lesion-size dependent |
| Dice / IoU | Segmentation overlap | Pixel or region | Sensitive to small structures and annotation boundaries |
| Quadratic weighted kappa | Agreement on ordinal grade | Case | Class imbalance and ordinal distance assumptions |
| Mitosis F1 | Detection of mitotic figures | Cell | Scanner and stain shift; annotation ambiguity |
| Biomarker concordance | Agreement with reference assay | Case | Threshold proximity and assay differences |
| Turnaround time | Order to report | Case | Workflow, not model quality |

**The central method: multiple-instance learning.** A whole-slide image is a bag of tiles; the slide label is known but tile labels usually are not. Attention-based MIL learns which tiles matter. CLAM adds clustering and instance-level supervision. TransMIL models tile relationships. Patch-level CNNs with aggregation are simpler baselines and often competitive.

**Stain and scanner shift.** H&E color varies with stain concentration, protocol, scanner, and compression. Stain normalization (Macenko, Vahadane, Reinhard) reduces but does not remove shift. Color augmentation (HED jitter) improves robustness. The decisive test is external validation on slides from a different scanner and laboratory, not a random split of the same scanner's slides.

**Leakage.** Splitting by tile leaks because tiles from the same slide appear in train and test. Splitting by slide leaks if the same patient has multiple slides. Split by patient, then by slide, and report both. Check for near-duplicate tiles and for slide-level metadata that encodes site.

### 7.4 Use cases

**1. CAMELYON16 metastasis detection (Ehteshami Bejnordi et al., JAMA 2017).** Deep learning systems detected lymph-node metastases on whole-slide images with performance comparable to pathologists in a controlled reader study. What transferred: WSI analysis can meet a clinical reference standard. What did not: the task was metastasis detection in a curated challenge set, not full diagnostic workflow, and scanner diversity was limited.

**2. PANDA Gleason grading (Bulten et al., Nature Medicine 2022).** The challenge used 10,616 prostate biopsies from multiple countries. Top AI systems achieved agreement with expert uropathologists comparable to inter-pathologist agreement, and outperformed general pathologists on grade assignment. What transferred: large, multi-center data and a clinically meaningful ordinal endpoint. What did not: grade disagreement is partly irreducible, and the challenge did not test treatment decisions.

**3. MSI prediction from H&E (Kather et al., Nature Medicine 2019).** A deep model predicted microsatellite instability from routine histology in gastrointestinal cancer. What transferred: morphology carries molecular information. What did not: the signal is a surrogate, performance varies by tumor type, and clinical use requires prospective validation against the molecular assay.

**4. Digital pathology clearance (Philips IntelliSite, FDA 2017).** The first whole-slide imaging system was authorized for primary diagnostic use, enabling digital readout. What transferred: the scanner, not the AI, was the regulatory unlock. What did not: digital readout alone does not validate any algorithm built on top of it.

**5. Paige Prostate (FDA De Novo DEN200080, 2021).** An AI tool was authorized to help identify prostate cancer in biopsy images, with a workflow role as a detection aid rather than an autonomous diagnosis. What transferred: a narrow, well-specified detection task can reach market. What did not: the authorization does not establish improvement in patient outcomes, and deployment requires pathologist review.

### 7.5 Regulatory, ethical, and equity issues

- **Device or diagnostic.** Pathology AI is usually a medical device; if it reports a biomarker used to select therapy, it may be an in-vitro diagnostic under IVDR and a companion diagnostic in the US.
- **Laboratory regulation.** CLIA and CAP accreditation govern the laboratory, including analytic validation, proficiency testing, and document control. An AI tool used in a clinical lab must fit the lab's quality system.
- **Intended use.** "Detection aid," "triage," "second read," and "autonomous diagnosis" are different claims with different evidence and different liability.
- **Equity.** Scanner access, stain protocols, and pathologist shortages vary by resourced setting. A model validated on high-resource academic slides may fail in a low-resource lab. Skin and tissue color, fixation, and slide preparation differ across populations and geographies.
- **Label noise.** Pathologist disagreement is not only model error; it is a property of the reference standard. Report inter-pathologist agreement and avoid claiming superhuman accuracy on an ordinal task with irreducible disagreement.
- **Automation bias.** A pathologist may accept an AI highlight without independent review, especially under workload pressure. Design for verification, not deference.

### 7.6 Lab: Slide-level metastasis detection and leakage audit

**Objective.** Build a slide-level classifier, demonstrate tile leakage, and test robustness to stain and scanner shift.

**Data.** CAMELYON16 whole-slide images, or PCam patches if compute is limited. Use the official train/test split where available and verify patient-level separation.

**Steps.**

1. Read slides with OpenSlide; record level, magnification, and microns per pixel.
2. Segment tissue and extract tiles at a fixed magnification; discard background.
3. Split by patient or slide, never by tile. Report the split counts.
4. Apply stain normalization and HED color augmentation to a training copy.
5. Train a patch CNN and aggregate with attention-based MIL; compare with mean pooling.
6. Evaluate slide-level AUROC and sensitivity at a fixed specificity, with confidence intervals.
7. Deliberately run a tile-level split and report the inflated result; quantify the leakage gap.
8. Stress-test with color jitter and, if available, slides from a different scanner; report the performance drop.
9. Review false positives and false negatives for pen marks, folds, and annotation errors.

**Deliverable.** A leakage audit, a slide-level evaluation report, and an error gallery.

**Acceptance criteria.**

- Patient or slide separation verified with zero overlap.
- Slide-level AUROC with 95% confidence interval.
- Explicit comparison of tile-split versus slide-split performance.
- Normalized versus unnormalized results reported.
- Robustness results under at least one stain or scanner perturbation.
- Error gallery includes at least five cases with an explanation.
- Intended use, workflow role, and regulatory category stated.

### 7.7 Interview and discussion questions

| Question | Hint for the instructor |
|---|---|
| Why is a tile-level split invalid? | Tiles from one slide share stain, scanner, and tissue; the test set is not independent. |
| How would you detect scanner leakage? | Slide metadata, image fingerprints, near-duplicate detection, and external validation on a second scanner. |
| What is multiple-instance learning and why use it? | Slide labels are available, tile labels are not; MIL aggregates instance evidence. |
| Grade agreement is only moderate among pathologists. What does that imply? | The label has irreducible noise; report human agreement and avoid claiming perfect accuracy. |
| When is stain normalization harmful? | It can erase biologically meaningful color or introduce artifacts; validate rather than assume. |
| What evidence supports an autonomous diagnosis claim? | Prospective, multi-site, reader-study evidence with adjudicated reference standards and a defined workflow. |
| How do you handle a slide with multiple specimens? | Define the unit of analysis, aggregate explicitly, and avoid mixing label semantics. |
| What is the difference between detection aid and triage? | Detection aid supports the pathologist's read; triage changes order or urgency and carries different harm. |
| Which patients are underrepresented in WSI datasets? | Low-resource settings, certain tissue types, and populations with different staining or fixation. |
| How would you validate a HER2 AI scorer? | Against a validated reference assay, at clinically relevant thresholds, with concordance and discordant-case review. |

### 7.8 Sources

- CAMELYON16. https://camelyon16.grand-challenge.org/ (accessed 24 Oct 2026).
- CAMELYON17. https://camelyon17.grand-challenge.org/ (accessed 24 Oct 2026).
- PANDA prostate cancer grade assessment. https://www.kaggle.com/c/prostate-cancer-grade-assessment (accessed 24 Oct 2026).
- PatchCamelyon. https://patchcamelyon.grand-challenge.org/ (accessed 24 Oct 2026).
- DICOM Standard. https://www.dicomstandard.org/ (accessed 24 Oct 2026).
- FDA De Novo DEN200080 (Paige Prostate). https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/denovo.cfm?ID=DEN200080 (accessed 24 Oct 2026).
- Ehteshami Bejnordi B, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA. 2017. https://doi.org/10.1001/jama.2017.14585 (accessed 24 Oct 2026).
- Bulten W, et al. Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge. Nat Med. 2022. https://doi.org/10.1038/s41591-021-01620-2 (accessed 24 Oct 2026).
- Kather JN, et al. Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer. Nat Med. 2019. https://doi.org/10.1038/s41591-019-0462-y (accessed 24 Oct 2026).
- College of American Pathologists. https://www.cap.org/ (accessed 24 Oct 2026).

---

## 8. Critical Care

### 8.1 The decision and who pays

Critical care is a dense stream of decisions made under uncertainty, fatigue, and time pressure. The data are continuous, the patients are unstable, and the cost of a missed deterioration is measured in minutes.

| Decision | Decision-maker | Economic buyer | Cost of a false positive | Cost of a false negative |
|---|---|---|---|---|
| Sepsis alert | Bedside nurse, rapid response | Hospital | Alarm fatigue, unnecessary antibiotics | Delayed antibiotics, shock |
| Deterioration and ICU transfer | Ward team | Hospital | Unnecessary transfer, bed use | Code event, unplanned ICU stay |
| Ventilator weaning | Intensivist, respiratory therapist | Hospital | Failed extubation | Prolonged ventilation |
| Fluid and vasopressor titration | Intensivist | Hospital | Pulmonary edema, arrhythmia | Hypoperfusion |
| AKI detection | Nephrologist, ICU team | Hospital, payer | Fluid restriction, avoidable dialysis | Drug toxicity, chronic kidney disease |
| Discharge or step-down | ICU team | Hospital, payer | Readmission to ICU | Prolonged ICU stay |

Payment is through diagnosis-related groups, value-based purchasing, and capitated arrangements, so the hospital usually bears both the cost of the alert and the cost of the adverse event. Sepsis bundle compliance is publicly reported through CMS. A vendor sells software to the hospital, but the clinician bears the workflow cost, which is why adoption fails even when discrimination is good.

**Industry interpretation.** In critical care, the buyer is the hospital and the user is a nurse or intensivist working at the bedside. A model that adds a click without removing one will be ignored.

### 8.2 Data sources and standards

| Source | Contents | Access | Critical-care-specific caveat |
|---|---|---|---|
| **MIMIC-IV v3.1** | 364,627 patients, 546,028 hospitalizations, 94,458 ICU stays from one US academic center, 2008–2022 | PhysioNet credentialed | Single center; ICU selection; de-identified dates; post-discharge outcomes incomplete |
| **eICU-CRD** | About 200,000 ICU stays across roughly 200 US hospitals (verify) | PhysioNet credentialed | Uneven variable coverage; no notes or images; practice variation |
| **PhysioNet Challenge 2019** | ICU time series for sepsis prediction from three hospitals | Open training data; test labels withheld | Label definition and cohort construction are part of the task |
| **HiRID** | High-resolution ICU physiology from Bern | Credentialed, Swiss governance | Single center; European practice; complex ethics |
| **AmsterdamUMCdb** | Dutch ICU database, about 23,000 admissions (verify) | Credentialed | Different units and coding; single center |
| **MIMIC-IV waveform** | High-frequency waveforms and numerics | PhysioNet credentialed | Device-specific; alignment and calibration issues |
| **MIMIC-CXR-JPG** | 377,110 chest radiographs with reports for 65,379 patients | PhysioNet credentialed | Report-derived labels; support devices; single center |
| **CMS claims and HCRIS** | Cost, utilization, quality measures | Licensed or public | No physiology; coding and payment artifacts |

**Standards.** Sepsis-3 definitions and SOFA scores, qSOFA as a screening prompt, APACHE IV and SAPS II for severity adjustment, OASIS for mortality prediction, ICD-10-CM/PCS, LOINC for labs, SNOMED CT, HL7 v2 and FHIR for messaging, and CDISC for trial data. Vasopressor doses must be normalized to norepinephrine equivalents before comparison.

**Data-quality checks.** Unit conversions (mg/dL vs mmol/L, µg/kg/min vs µg/min); charted versus derived values; time alignment across devices; missingness that is informative (a lab is missing because it was not ordered); duplicate charting; and whether the ICU stay is the first or a readmission. Compute cohort flow and report excluded stays with reasons.

### 8.3 Core methods and endpoints

| Endpoint | Definition | Window | Ascertainment risk |
|---|---|---|---|
| In-hospital mortality | Death before discharge | Admission to discharge | Discharge to hospice or facility may be coded as survival |
| 28- or 30-day mortality | Death within fixed window | Fixed | Requires post-discharge linkage |
| Sepsis onset | Suspected infection plus organ dysfunction | ICU stay | Definition-dependent; retrospective SOFA |
| Time to antibiotics | Order to administration | Hours | Charting delay; missing administration time |
| Ventilator-free days | Days alive and free of ventilation | 28 days | Competing risk of death |
| AKI | KDIGO creatinine or urine criteria | ICU stay | Baseline creatinine often unknown |
| ICU length of stay | Admission to discharge | ICU stay | Discharge is a competing event, not a failure |
| Deterioration | Escalation, code, or unplanned transfer | 6–24 hours | Ascertainment depends on monitoring intensity |

**Competing risks.** In ICU data, discharge and death compete. A model that predicts "time to discharge" without accounting for death will overestimate discharge for the sickest patients. Use cause-specific hazards or Fine-Gray models, and state which question each answers.

**Label leakage in sepsis.** Sepsis-3 defines suspected infection by culture and antibiotic timing and organ dysfunction by SOFA change. If features include labs drawn after the suspicion time, or the suspicion time itself is derived from later data, the model sees the future. Fix time zero as the earlier of culture and antibiotic, restrict features to a pre-specified window, and audit every feature timestamp.

**Alarm burden.** Report alerts per patient-day, positive predictive value, number needed to alert for one true event, override rate, and time from alert to action. A model with excellent AUROC and a 2% PPV at the chosen threshold may be worse than no alert.

**Methods.** Discrete-time hazard models with time-varying covariates; gradient boosting on aggregate features; recurrent and transformer models on irregular time series; conformal prediction for uncertainty; target trial emulation for fluids, vasopressors, and transfusion; and off-policy evaluation for reinforcement-learning treatment policies, with strong skepticism about unmeasured confounding.

### 8.4 Use cases

**1. External validation of the Epic Sepsis Model (Wong et al., JAMA Internal Medicine 2021).** At Michigan Medicine, the widely deployed proprietary model achieved an AUROC of about 0.63, well below the vendor's reported performance, with poor calibration and a high alert volume. It missed a substantial fraction of sepsis cases. What transferred: independent external validation is mandatory. What did not: the result is site-specific, but it destroyed the assumption that widespread use implies validity.

**2. TREWS targeted early warning (Henry et al., Nature Medicine 2022; Adams et al., Nature Medicine 2022).** A health-system model was designed to reduce alert burden by targeting high-risk patients and integrating into workflow. Prospective evaluation reported improved outcomes when alerts were confirmed and acted on within a defined window. What transferred: alert burden and workflow are part of the intervention. What did not: single health system, evolving model, and residual confounding limit generalization.

**3. AI Clinician for sepsis treatment (Komorowski et al., Nature Medicine 2018).** A reinforcement-learning policy was learned from MIMIC-III to recommend fluid and vasopressor doses, with survival benefit in off-policy evaluation. What transferred: RL can generate hypotheses about treatment strategies. What did not: off-policy evaluation depends on untestable assumptions; Gottesman et al. (2019) documented the fragility of RL evaluation in healthcare. Do not treat a learned policy as a causal effect.

**4. PhysioNet Challenge 2019 sepsis prediction.** Teams predicted sepsis from ICU time series, with the label and utility function defined by the organizers. What transferred: a common benchmark enabled comparison and surfaced the importance of early prediction and alert utility. What did not: the winning models did not necessarily transport to other hospitals, and the utility metric encoded specific assumptions.

**5. CMS SEP-1 sepsis bundle.** The Severe Sepsis and Septic Shock Early Management Bundle ties measurement of lactate, cultures, antibiotics, and fluids to public reporting. What transferred: measurement changes behavior. What did not: bundle compliance is not the same as patient benefit, and the evidence for the bundle remains debated; do not use SEP-1 compliance as a surrogate for model success.

### 8.5 Regulatory, ethical, and equity issues

- **Device status.** A sepsis or deterioration model can be regulated as software as a medical device, or may qualify as clinical decision support if the clinician can independently review the basis. The line depends on intended use and transparency, not on the algorithm.
- **Alarm safety.** The Joint Commission and ECRI have repeatedly identified alarm fatigue as a hazard. A new alert must be evaluated as an intervention, including its effect on other alarms.
- **Consent and monitoring.** ICU patients are rarely able to consent. Secondary use of de-identified data is governed by HIPAA and institutional review, but the original consent for care does not cover all research uses.
- **Equity.** Pulse oximetry overestimates oxygen saturation in patients with darker skin, which can delay recognition of hypoxemia. Creatinine-based AKI definitions depend on muscle mass. Sepsis presentation and treatment differ by sex, age, language, and insurance. Report subgroup performance and alert rates, not only overall AUROC.
- **Automation and desensitization.** A false alert is not neutral; it trains clinicians to ignore the next one. Measure the human response, not only the model output.
- **Resource allocation.** A deterioration model can redistribute attention toward monitored patients and away from unmonitored ones. State the capacity constraint and the queuing rule.

### 8.6 Lab: Sepsis early warning with a leakage audit

**Objective.** Build a 6-hour sepsis prediction model, prove that no post-time-zero data entered, and quantify alert burden.

**Data.** MIMIC-IV (credentialed) or the PhysioNet 2019 open training set. If using MIMIC-IV, use the official subject IDs and a temporal split.

**Steps.**

1. Define the cohort and the suspicion time as the earlier of culture and antibiotic; document the Sepsis-3 implementation.
2. Set time zero at suspicion; restrict features to the preceding 6–24 hours.
3. Build features: vitals, labs, medications, urine output, ventilation, and comorbidity.
4. Train a discrete-time logistic model and a gradient-boosted baseline; report calibration.
5. Evaluate AUROC and AUPRC with confidence intervals; report sensitivity at clinically plausible alert thresholds.
6. Compute alerts per patient-day and the number needed to alert at each threshold.
7. Run a leakage audit: list every feature, its timestamp rule, and the code that enforces it.
8. Repeat with an alternative sepsis label definition and report the change.
9. Report subgroup performance by age, sex, and race, with attention to missingness.

**Deliverable.** A model card, a calibration plot, an alert-burden table, and a leakage audit.

**Acceptance criteria.**

- Time zero is defined before any feature extraction.
- No feature timestamp is after time zero; the audit is reproducible.
- AUROC and AUPRC reported with 95% confidence intervals.
- Calibration slope and intercept reported.
- Alerts per patient-day and number needed to alert reported at two thresholds.
- Competing risk of discharge is addressed in any time-to-event analysis.
- Subgroup table includes counts, event rates, and performance.
- The notebook runs from a pinned environment with a fixed seed.

### 8.7 Interview and discussion questions

| Question | Hint for the instructor |
|---|---|
| Why is sepsis label leakage so easy to create? | Sepsis-3 uses labs and SOFA components that may be measured after suspicion; any post-time-zero feature leaks. |
| AUROC 0.85 and PPV 4%. Is the model good? | Depends on prevalence and capacity; 4% PPV may be unacceptable for a high-risk intervention. |
| How do you measure alert burden? | Alerts per patient-day, number needed to alert, override rate, and time to action. |
| What is a competing risk in ICU data? | Discharge and death preclude the other outcome; standard survival analysis misattributes risk. |
| How would you validate a sepsis model across hospitals? | External validation with site-specific calibration, case-mix adjustment, and reporting by site. |
| What is the treatment paradox? | Sicker patients receive more treatment, so treatment looks harmful if severity is not modeled. |
| Why did the Epic Sepsis Model fail external validation? | Different population, case mix, label definition, and calibration; widespread use is not evidence. |
| Can reinforcement learning tell us the best treatment policy? | Only under strong assumptions; off-policy evaluation is fragile and needs clinical trials. |
| How do you handle informative missingness? | Model the measurement process, use missingness indicators cautiously, and run sensitivity analyses. |
| Which patients might a deterioration model systematically miss? | Those with less monitoring, atypical presentation, or language barriers; check alert rates and delays. |

### 8.8 Sources

- PhysioNet MIMIC-IV. https://physionet.org/content/mimiciv/ (accessed 24 Oct 2026).
- PhysioNet eICU-CRD. https://physionet.org/content/eicu-crd/ (accessed 24 Oct 2026).
- PhysioNet Challenge 2019. https://physionet.org/content/challenge-2019/ (accessed 24 Oct 2026).
- PhysioNet MIMIC-CXR-JPG. https://physionet.org/content/mimic-cxr-jpg/ (accessed 24 Oct 2026).
- Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016. https://doi.org/10.1001/jama.2016.0287 (accessed 24 Oct 2026).
- Wong A, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Intern Med. 2021. https://doi.org/10.1001/jamainternmed.2021.2626 (accessed 24 Oct 2026).
- Adams R, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine learning-based early warning system for sepsis. Nat Med. 2022. https://doi.org/10.1038/s41591-022-01894-0 (accessed 24 Oct 2026; verify study details).
- Komorowski M, et al. The artificial intelligence clinician learns optimal treatment strategies for sepsis in intensive care. Nat Med. 2018. https://doi.org/10.1038/s41591-018-0213-5 (accessed 24 Oct 2026).
- Gottesman O, et al. Guidelines for reinforcement learning in healthcare. Nat Med. 2019. https://doi.org/10.1038/s41591-018-0310-5 (accessed 24 Oct 2026).
- CMS. Severe Sepsis and Septic Shock Early Management Bundle (SEP-1). https://www.cms.gov/medicare/quality/measures (accessed 24 Oct 2026).

---

## 9. Pediatrics and Obstetrics

### 9.1 The decision and who pays

Pediatrics and obstetrics share a defining feature: the patient is not a small adult, and the outcome often affects two people at once. Both domains are underdigitized relative to adult medicine, and both are dominated by public payers.

| Decision | Decision-maker | Economic buyer | Cost of a false positive | Cost of a false negative |
|---|---|---|---|---|
| Pediatric dose and route | Pediatrician, pharmacist | Hospital, Medicaid | Toxicity, overdose | Underdosing, treatment failure |
| Growth or developmental concern | Pediatrician | Medicaid, insurer | Unnecessary referral, anxiety | Delayed intervention |
| Pediatric sepsis escalation | Emergency physician, intensivist | Children's hospital | Antibiotic overuse | Shock, death |
| ROP screening interval | Ophthalmologist | Medicaid, hospital | Unnecessary exams | Blindness |
| Prenatal aneuploidy screening | Obstetrician, genetic counselor | Insurer, Medicaid | Invasive test, pregnancy loss | Missed diagnosis |
| Preterm preeclampsia prophylaxis | Obstetrician | Insurer, Medicaid | Unnecessary aspirin, monitoring | Preterm birth, maternal morbidity |
| Intrapartum fetal monitoring | Obstetrician, midwife | Hospital, payer | Operative delivery | Hypoxic injury |
| Postpartum discharge and follow-up | Obstetric team | Medicaid, insurer | Readmission | Maternal death, severe morbidity |

Medicaid finances about 40% of US births (verify the current figure) and a large share of pediatric care. Medicaid also pays for most NICU and preterm care. Commercial insurers cover the rest, often under a global obstetric package that bundles prenatal, delivery, and postpartum care, which can mute incentives for separate digital tools. The Children's Health Insurance Program, Title V maternal and child health block grants, and state perinatal quality collaboratives are additional buyers.

**Industry interpretation.** In pediatrics and obstetrics, the purchaser is often a public program with a fixed budget and a strong equity mandate. A product that only works for commercially insured patients has a small market and a larger equity problem.

### 9.2 Data sources and standards

| Source | Contents | Access | Pediatrics/OB-specific caveat |
|---|---|---|---|
| **PHIS** | Inpatient and emergency data from dozens of US children's hospitals | Participating hospitals; licensed | Not population-based; participating-hospital case mix; verify current coverage |
| **HCUP Kids' Inpatient Database** | Pediatric discharges, national sample, every few years | Purchase and training | Discharge-level; no longitudinal follow-up; coding artifacts |
| **NHANES** | Cross-sectional nutrition, growth, and biomarker data | Open | Small annual sample; not clinical; self-report components |
| **National Survey of Children's Health** | Parent-reported child health and access | Open | Parent report; survey weighting required |
| **PIC database** | Pediatric ICU data from a Chinese children's hospital (verify size and access) | PhysioNet credentialed | Single center; different practice and units |
| **CTU-UHB CTG database** | 552 intrapartum cardiotocography recordings with umbilical artery outcomes | Open on PhysioNet | Single center; 30-minute recordings; selection and recording quality |
| **CDC natality files** | Birth certificates, about 3.6 million US births per year (verify current) | Open or restricted | Certificate reporting artifacts; gestational age and race fields vary by state |
| **PRAMS** | State-based maternal attitudes and experiences | Application | Self-report; state participation varies |
| **CDC WONDER** | Mortality and natality query system | Open | Suppression of small cells; cause-of-death coding |
| **All of Us** | Diverse cohort with surveys, EHR, and genomics | Registered and controlled tiers | Pediatric representation is limited; verify current enrollment |

**Standards.** WHO child growth standards for ages 0–5 and WHO growth reference for 5–19; CDC growth charts for US clinical use; LOINC for growth and lab observations; SNOMED CT; ICD-10-CM and ICD-10-PCS; ICD-11 for international use; HL7 FHIR pediatric profiles; CDISC pediatric domains; ICH E11(R1) for pediatric drug development; AAP Bright Futures periodicity schedule; FIGO intrapartum fetal monitoring terminology; and ACOG practice guidance.

**Data-quality checks.** Units: kilograms versus pounds, milligrams versus micrograms, and mg/kg versus mg/m². Age: gestational age, postmenstrual age, and chronological age are different variables. Growth: measurement position, equipment, and who measured. Pregnancy: gravidity, parity, and whether the denominator is pregnancies, deliveries, or live births. Fetal monitoring: recording quality, calibration, and whether the tracing is complete.

### 9.3 Core methods and endpoints

| Endpoint | Definition | Window | Ascertainment risk |
|---|---|---|---|
| Adverse drug event | Harm from a medication | Hospital stay or 30 days | Underreporting; causality attribution |
| Growth z-score | Standard deviations from reference | Age-specific | Reference choice and measurement error |
| Pediatric sepsis | Phoenix criteria: suspected infection plus organ dysfunction | ICU stay | Definition change from prior criteria |
| ROP stage or plus disease | Retinopathy of prematurity severity | Screening exam | Examiner variability; image quality |
| Preterm birth | Birth before 37 weeks | Delivery | Gestational age dating method |
| Preeclampsia | Hypertension plus proteinuria or organ dysfunction | After 20 weeks | Definition updates; late onset |
| Neonatal acidemia | Umbilical artery pH below threshold or base deficit | Delivery | Cord gas sampling and thresholds |
| Severe maternal morbidity | Transfusion, hysterectomy, ICU, organ failure | Delivery to 42 days | Coding and hospital variation |
| Maternal mortality | Death during pregnancy or within 42 days | Pregnancy to 42 days | Underreporting and misclassification |

**Weight-based dosing.** Pediatric doses are usually mg/kg or mg/m², with caps at adult doses. The Mosteller formula estimates body surface area as the square root of height in cm times weight in kg divided by 3,600. Clearance matures with age and organ function, so a neonate is not a scaled-down child. A tenfold decimal error is the classic harm; the model or rule engine must include hard stops, maximum-dose checks, and weight-unit confirmation.

**Growth and development.** Use LMS-based z-scores, not raw height or weight. A child at the 5th percentile may be normal; a child who crosses two major percentile lines is concerning. Age must be handled as a nonlinear, time-varying variable, and preterm infants are often corrected for gestational age until age 2.

**Rare events.** Pediatric and obstetric outcomes are often rare, so AUROC is unstable and sensitivity is more useful than overall accuracy. Use penalized regression (Firth), exact methods, Bayesian hierarchical models, and confidence intervals that reflect small counts. Report the number of events per variable.

**Pregnancy physiology.** Plasma volume rises by roughly 40–50%, glomerular filtration by about 50%, and cardiac output by 30–50%. Blood pressure falls in mid-pregnancy. Hemoglobin is diluted. These changes alter reference ranges and drug clearance; adult thresholds applied to pregnant patients can be wrong in both directions.

**Methods.** Pharmacokinetic modeling and dose simulation; allometric scaling; mixed-effects models for repeated growth measures; time-to-event analysis with gestational age as the time scale; test-negative designs for vaccine effectiveness; interrupted time series for policy; and hierarchical models for hospital or state variation.

### 9.4 Use cases

**1. Phoenix pediatric sepsis criteria (Schlapbach et al., JAMA 2024).** An international consensus replaced prior pediatric sepsis definitions with criteria based on suspected infection plus organ dysfunction, validated across multiple datasets. What transferred: adult Sepsis-3 criteria do not transfer to children, and a data-driven consensus can produce a more specific definition. What did not: the criteria still depend on data availability and did not resolve treatment questions.

**2. Deep learning for retinopathy of prematurity (Brown et al., JAMA Ophthalmology 2018).** A convolutional network diagnosed plus disease from retinal images with performance comparable to expert graders in a research setting. What transferred: image-based screening can support a severe shortage of ROP experts. What did not: image quality, camera type, and infant characteristics vary, and the model was not a substitute for the full screening examination.

**3. Computer-aided fetal monitoring (INFANT trial, Brocklehurst et al., Lancet 2017).** A large UK randomized trial of computer-aided cardiotocography interpretation during labor found no improvement in poor neonatal outcomes compared with standard interpretation. What transferred: a technically plausible decision aid can fail when the clinical pathway and the reference standard are weak. What did not: the trial does not prove that all fetal monitoring AI is useless, but it sets a high evidentiary bar.

**4. Aspirin for preterm preeclampsia prevention (ASPRE, Rolnik et al., NEJM 2017).** In women identified as high risk by a first-trimester screening algorithm, aspirin reduced preterm preeclampsia compared with placebo. What transferred: a multivariable risk model plus a cheap intervention can change outcomes. What did not: the screening algorithm, population, and adherence matter, and the result is specific to preterm preeclampsia.

**5. Placental growth factor testing for preeclampsia (NICE DG49).** The sFlt-1/PlGF ratio was recommended to help rule out preeclampsia in women presenting with suspected disease. What transferred: a biomarker can shorten the diagnostic pathway. What did not: the test is an aid, not a standalone diagnosis, and thresholds and assay platforms differ.

### 9.5 Regulatory, ethical, and equity issues

- **Research protections.** In the US, research with children is governed by 45 CFR 46 Subpart D and 21 CFR 50 Subpart D, including parental permission and child assent. Research with pregnant women, fetuses, and neonates is governed by 45 CFR 46 Subpart B. These are not administrative details; they shape what data can exist.
- **Pediatric drug development.** The Pediatric Research Equity Act and the Best Pharmaceuticals for Children Act create incentives and requirements for pediatric studies. The EU Pediatric Regulation provides a parallel pathway. Off-label use remains common, and dosing information is often extrapolated.
- **Devices.** Pediatric devices are a small market with high regulatory and liability costs. The FDA has specific programs and guidance for pediatric device development; verify current incentives.
- **Consent and privacy.** A pregnant patient's data and a fetus's data are not the same record. Adolescents may have confidentiality rights for certain services. State laws vary.
- **Equity.** Medicaid coverage, rural maternity care deserts, language access, and immigration status affect who is monitored and who is believed. Maternal mortality is markedly higher among Black and Indigenous women in the US; a model trained on majority populations can encode access barriers as biology.
- **Denominators.** Maternal mortality per 100,000 live births is not the same as per 100,000 pregnancies. State reporting, review committees, and cause-of-death coding change the numerator. Always state the denominator and the surveillance system.
- **Fetal monitoring.** Inter-observer agreement on cardiotocography is modest. An algorithm that claims to read a tracing better than clinicians must address the reference standard, the legal context, and the risk of defensive medicine.

### 9.6 Lab: Rare-event prediction from intrapartum CTG

**Objective.** Build a rare-event model on fetal heart rate recordings, calibrate it, and evaluate it at realistic prevalence and subgroup levels.

**Data.** CTU-UHB intrapartum CTG database (552 recordings, about 30 minutes, 4 Hz, with umbilical artery pH and base deficit).

**Steps.**

1. Load recordings and outcomes; define acidemia as umbilical artery pH below 7.05 or base deficit above a stated threshold; document the choice.
2. Perform signal quality control: remove segments with loss, artifact, or maternal heart rate confusion.
3. Extract features: baseline rate, variability, accelerations, decelerations, and their timing relative to delivery.
4. Split by recording, and verify no patient appears in more than one split.
5. Train a logistic model with penalization and a gradient-boosted baseline; use class weights.
6. Calibrate and report AUROC, AUPRC, sensitivity, and PPV at 1% and 10% prevalence.
7. Compare with a simple FIGO-style rule-based classifier.
8. Report subgroup performance by gestational age, parity, and recording quality.
9. Discuss inter-observer variability and the clinical consequence of a false positive.

**Deliverable.** A model card, a prevalence-adjusted performance table, and a one-page clinical workflow note.

**Acceptance criteria.**

- Outcome definition and threshold are explicit.
- Recording-level split with zero patient overlap.
- AUPRC reported because the event is rare.
- Calibration plot and Brier score reported.
- PPV at two prevalences, not only at the sample prevalence.
- Subgroup table with counts and confidence intervals.
- Comparison with a rule-based baseline.
- A written statement of the clinical action and its harm.

**Pediatric stretch.** Repeat the rare-event workflow on a pediatric dataset available to the institution, such as PHIS or PIC with credentialing, and add a weight-based dose-check module with hard stops for tenfold errors.

### 9.7 Interview and discussion questions

| Question | Hint for the instructor |
|---|---|
| Why is weight-based dosing a data problem, not only a clinical one? | Units, decimal errors, weight entry, and maximum-dose caps; the rule engine must fail safe. |
| What is the difference between gestational, postmenstrual, and chronological age? | Age since conception, age since birth plus gestational age, and age since birth; each changes the reference range. |
| Why do adult sepsis criteria fail in children? | Different physiology, definitions, and outcome rates; pediatric criteria were developed separately. |
| How do you evaluate a model for a 2% outcome? | AUPRC, sensitivity at fixed alert volume, calibration, and confidence intervals; AUROC alone is unstable. |
| What changes in pregnancy physiology that affects lab interpretation? | Plasma volume expansion, dilutional anemia, increased GFR, and lower blood pressure. |
| Why did the INFANT trial fail to show benefit? | The decision aid did not improve the pathway; interpretation and response, not just detection, determine outcomes. |
| How should maternal mortality be denominated? | Per live births for standard reporting, but state the system and recognize pregnancy-related deaths outside the 42-day window. |
| What is the risk of using adult reference ranges for children? | Misclassification of normal variation and missed or false diagnoses; use age- and sex-specific references. |
| How do you handle a pregnant patient in a trial? | Subpart B protections, risk-benefit assessment, and inclusion rather than blanket exclusion when appropriate. |
| Which equity issues are specific to obstetrics? | Access to maternity care, Medicaid coverage, language, racism, and rural closures; these are confounders, not biology. |

### 9.8 Sources

- PhysioNet CTU-UHB CTG database. https://physionet.org/content/ctu-uhb-ctgdb/ (accessed 24 Oct 2026).
- WHO child growth standards. https://www.who.int/tools/child-growth-standards (accessed 24 Oct 2026).
- CDC growth charts. https://www.cdc.gov/growthcharts/ (accessed 24 Oct 2026).
- CDC PRAMS. https://www.cdc.gov/prams/ (accessed 24 Oct 2026).
- CDC maternal mortality. https://www.cdc.gov/maternal-mortality/ (accessed 24 Oct 2026).
- 45 CFR 46 Subpart B. https://www.hhs.gov/ohrp/regulations-and-policy/regulations/45-cfr-46/common-rule-subpart-b/index.html (accessed 24 Oct 2026).
- 21 CFR 50 Subpart D. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-50/subpart-D (accessed 24 Oct 2026).
- FDA Pediatric Research Equity Act. https://www.fda.gov/drugs/development-resources/pediatric-research-equity-act-prea (accessed 24 Oct 2026).
- Schlapbach LJ, et al. International consensus criteria for pediatric sepsis and septic shock. JAMA. 2024. https://doi.org/10.1001/jama.2024.0179 (accessed 24 Oct 2026).
- Brocklehurst P, et al. Computerised interpretation of fetal heart rate during labour (INFANT): a randomised controlled trial. Lancet. 2017. https://doi.org/10.1016/S0140-6736(17)30568-8 (accessed 24 Oct 2026).
- Rolnik DL, et al. Aspirin versus placebo in pregnancies at high risk for preterm preeclampsia. N Engl J Med. 2017. https://doi.org/10.1056/NEJMoa1704559 (accessed 24 Oct 2026).
- Brown JM, et al. Automated diagnosis of plus disease in retinopathy of prematurity using deep convolutional neural networks. JAMA Ophthalmol. 2018. https://doi.org/10.1001/jamaophthalmol.2018.1934 (accessed 24 Oct 2026).

---

## 10. Public Health

### 10.1 The decision and who pays

Public health decisions are made for populations rather than individual patients, often with incomplete and delayed data. The buyer is usually government, and the benefit is often invisible because the outbreak that did not happen cannot be counted.

| Decision | Decision-maker | Economic buyer | Cost of a false positive | Cost of a false negative |
|---|---|---|---|---|
| Outbreak detection | Local health department | Government | unnecessary investigation, alarm | Uncontrolled transmission |
| Vaccination allocation | State or national program | Government, insurer | Wasted doses, inequity | preventable outbreaks |
| Screening program | Public health agency | Government, payer | Overdiagnosis, cost | Late-stage disease |
| Resource allocation | Emergency operations center | Government | Misallocated staff and beds | Preventable deaths |
| Policy (masks, closures, mandates) | Elected officials, agency | Government | Economic and social harm | Transmission, morbidity |
| Environmental or exposure advisory | Health and environment agencies | Government | Unnecessary avoidance | Continued exposure |

Funding comes from CDC cooperative agreements, state and local appropriations, CMS, insurers, and international donors. Public health data systems are chronically underfunded, which is why surveillance lag and manual reporting are persistent rather than temporary problems.

**Industry interpretation.** Public health is the specialty where the denominator is the product. A model that reports cases without a defensible population at risk is not a public health model.

### 10.2 Data sources and standards

| Source | Contents | Access | Public-health-specific caveat |
|---|---|---|---|
| **CDC NWSS** | Wastewater concentrations for SARS-CoV-2 and other pathogens | Open dashboards and data downloads | Sampling site coverage, normalization method, and sewershed population vary |
| **CDC NHSN** | Healthcare-associated infections, hospital capacity, respiratory data | Open or registered | Facility participation and reporting completeness vary |
| **CDC NNDSS** | National notifiable disease case counts | Open weekly tables | Reporting delay, case-definition changes, state variation |
| **CDC WONDER** | Mortality, natality, and population data | Open | Suppression of small counts; cause-of-death coding |
| **CDC BRFSS** | State-level behavioral risk and chronic disease | Open | Self-report; landline and cell sampling; state weighting |
| **CDC NHANES** | Examination-based biomarkers and nutrition | Open | Small sample; complex survey design |
| **NVSS** | Birth, death, fetal death, and marriage records | Open or restricted | Reporting artifacts; state variation |
| **ACS** | Population, income, housing, and insurance | Open | Estimates with margins of error; small-area instability |
| **CDC/ATSDR SVI** | Social vulnerability by census tract | Open | Composite index; not a measure of individual risk |
| **ESSENCE** | Syndromic surveillance from emergency departments | State and federal access | Chief-complaint coding; not a diagnosis |
| **WHO and ECDC** | Global case, death, and vaccination data | Open | Country reporting differences; definitions vary |

**Standards.** ICD-10-CM and ICD-11 for diagnoses and causes of death; LOINC for laboratory tests; SNOMED CT for clinical findings; PHIN VADS for public health vocabularies; HL7 v2 and FHIR for messaging; CSTE position statements for case definitions; ISO 8601 for dates; and GIS standards for geography. Census geography changes over time; tract and ZIP code boundaries are not stable.

**Data-quality checks.** Case definition version and date; reporting source; specimen collection date versus report date; testing policy and access; duplicate records; residence versus reporting location; population denominator and vintage; and whether counts are preliminary, revised, or final. A time series without revision metadata is not a time series.

### 10.3 Core methods and endpoints

| Endpoint | Definition | Window | Ascertainment risk |
|---|---|---|---|
| Incidence rate | New cases divided by person-time at risk | Weekly or annual | Denominator and case definition |
| Prevalence | Existing cases divided by population | Point or period | Screening and survival |
| Case fatality ratio | Deaths divided by diagnosed cases | Episode | Depends on testing and diagnosis |
| Infection fatality ratio | Deaths divided by all infections | Episode | Requires serology or modeling |
| Effective reproduction number (Rt) | Average secondary infections per case | Sliding window | Model-dependent; reporting delay |
| Excess mortality | Observed minus expected deaths | Period | Baseline choice; age structure |
| Test positivity | Positive tests divided by tests | Period | Testing policy and access |
| Vaccination coverage | Vaccinated divided by eligible population | Period | Denominator and dose definition |
| Wastewater concentration | Pathogen signal per unit | Sample date | Normalization, flow, and site coverage |

**Surveillance lag and nowcasting.** Cases are reported after diagnosis, and reports are revised. The most recent weeks are incomplete. A naive plot of the last point creates a false decline. Nowcasting estimates the true current count by modeling the reporting delay distribution and the expected revisions.

**Denominators.** The population at risk is not always the census population. For hospital-onset infections, the denominator is patient-days. For sexually transmitted infections, it is the tested population. For maternal mortality, it is live births or pregnancies, depending on the definition. For wastewater, it is the contributing population of the sewershed. State the denominator before comparing places or times.

**Age standardization.** Direct standardization applies a standard age structure to age-specific rates; indirect standardization compares observed with expected using a reference population. Crude rates differ when age structures differ, which is why national comparisons require standardization.

**Methods.** Aberration detection (EARS, CUSUM, Farrington); Bayesian hierarchical nowcasting; ensemble forecasting; Rt estimation; interrupted time series and difference-in-differences for policy; test-negative designs for vaccine effectiveness; capture-recapture for undercount; small-area estimation; and spatial cluster detection.

### 10.4 Use cases

**1. COVID-19 Forecast Hub (Cramer et al., PNAS 2022).** An ensemble of independent models consistently outperformed most individual models for US COVID-19 mortality forecasts. What transferred: ensembles are a robust operational strategy under model uncertainty. What did not: ensemble performance depends on the diversity and quality of contributors, and forecasts are not causal claims.

**2. CDC National Wastewater Surveillance System.** NWSS was launched in 2020 to detect SARS-CoV-2 and later other pathogens in wastewater. It provided an early signal that was independent of clinical testing, but interpretation required normalization, site coverage, and knowledge of the contributing population. What transferred: pooled environmental sampling can fill gaps when clinical testing declines. What did not: wastewater does not identify individuals, and the signal-to-population mapping is uncertain.

**3. UK ONS COVID-19 Infection Survey.** A large household survey estimated infection prevalence with repeated sampling, providing a denominator that case counts could not. What transferred: probability sampling produces credible prevalence estimates. What did not: the survey was expensive, had response bias, and lagged real-time decisions.

**4. Israel nationwide vaccination study (Dagan et al., NEJM 2021).** Linked national data compared vaccinated and unvaccinated members during a mass campaign, showing substantial protection against infection and severe disease. What transferred: a national linked dataset can produce rapid, high-impact evidence. What did not: observational confounding, healthy-vaccinee effects, and a specific context limit generalization.

**5. Poliovirus detection in wastewater (London and New York State, 2022).** Environmental surveillance detected vaccine-derived poliovirus in sewage, triggering booster campaigns and investigation. What transferred: wastewater can detect a pathogen before clinical cases appear. What did not: detection does not prove ongoing transmission, and the response depends on vaccination coverage and public trust.

### 10.5 Regulatory, ethical, and equity issues

- **Legal authority.** Public health powers are largely state-based in the US. Mandatory reporting, isolation, quarantine, and data collection authority vary by jurisdiction and must be grounded in statute or regulation.
- **HIPAA public health exception.** HIPAA permits disclosures to public health authorities for specified purposes under 45 CFR 164.512(b), but the exception is narrower than many assume and does not cover every research or commercial use.
- **Privacy.** Wastewater and small-area data can identify neighborhoods or facilities. Cell suppression, aggregation, and differential privacy are necessary, not optional.
- **Stigma.** Data about sexually transmitted infections, drug use, immigration status, or disability can harm the people it describes. Publish at a level that does not enable re-identification or targeting.
- **Equity.** Surveillance systems undercount people without access to testing or care. Using case counts as if they were infections blames communities with less testing. Report testing rates and access alongside outcomes.
- **Misinformation and trust.** A model that changes policy must be explainable to the public, not only to experts. Transparency about uncertainty is a core requirement.
- **Emergency use.** During emergencies, evidence standards may be relaxed under legal authorities. Document the basis and the plan for post-authorization evidence.

### 10.6 Lab: Nowcasting with reporting delay

**Objective.** Build a nowcast for a weekly notifiable disease series, quantify revision and backfill, and compare with a persistence baseline.

**Data.** CDC NNDSS weekly tables or a public NWSS series with sample dates and reporting dates. Use at least three years of data if available.

**Steps.**

1. Build a reporting triangle: event date, report date, and count.
2. Estimate the reporting-delay distribution, allowing for day-of-week and holiday effects.
3. Fit a nowcast model (Bayesian hierarchical or simple delay-adjustment) for the most recent weeks.
4. Backtest over at least 12 weeks using only data available at each vintage.
5. Compare with a persistence baseline (last observed value carried forward).
6. Report the weighted interval score (WIS), 80% and 95% interval coverage, and bias.
7. Quantify revisions: how much does the estimate for week t change when observed at t+1, t+2, and t+4?
8. Produce an age-standardized rate and state the denominator and population vintage.

**Deliverable.** A nowcast report with a backtest, a revision table, and a one-page methods note.

**Acceptance criteria.**

- Only information available at each vintage is used; no future leakage.
- Delay distribution and model are documented.
- WIS improves on the persistence baseline, or the failure is explained.
- Interval coverage is reported and close to nominal.
- Revisions and backfill are quantified.
- Denominator, case definition, and population vintage are explicit.
- Code and data vintage are pinned.

### 10.7 Interview and discussion questions

| Question | Hint for the instructor |
|---|---|
| Why do the last two weeks of a case curve often decline? | Reporting delay and backfill; the most recent counts are incomplete. |
| What is the denominator for a hospital-onset infection rate? | Patient-days, not admissions or population. |
| Cases rose but testing also rose. What can you conclude? | Little without test positivity and testing access; incidence may be flat. |
| What is the ecologic fallacy? | Inferring individual risk from group-level associations. |
| How do you compare mortality across states with different age structures? | Age-standardize and state the standard population. |
| What does a wastewater signal measure? | Pathogen shed in a sewershed, not diagnosed infections; normalization and coverage matter. |
| How would you evaluate a vaccine's real-world effectiveness? | Test-negative design or cohort with confounding control; report by variant, age, and prior infection. |
| Why is Rt model-dependent? | It depends on the generation interval, reporting delay, and assumptions about immunity. |
| When is a forecast useful even if it is wrong? | When it quantifies uncertainty and supports a decision with asymmetric costs. |
| What are the privacy risks of small-area public health data? | Re-identification of neighborhoods or facilities; use suppression and aggregation. |

### 10.8 Sources

- CDC National Wastewater Surveillance System. https://www.cdc.gov/wastewater/ (accessed 24 Oct 2026).
- CDC National Healthcare Safety Network. https://www.cdc.gov/nhsn/ (accessed 24 Oct 2026).
- CDC National Notifiable Diseases Surveillance System. https://www.cdc.gov/nndss/ (accessed 24 Oct 2026).
- CDC WONDER. https://wonder.cdc.gov/ (accessed 24 Oct 2026).
- CDC FluSight forecasting. https://www.cdc.gov/flu-forecasting/ (accessed 24 Oct 2026).
- CDC COVID-19 Forecast Hub. https://github.com/CDCgov/covid19-forecast-hub (accessed 24 Oct 2026).
- UK Office for National Statistics COVID-19 Infection Survey. https://www.ons.gov.uk/surveys/informationforhouseholdsandindividuals/householdandindividualsurveys/covid19infectionsurvey (accessed 24 Oct 2026).
- CDC/ATSDR Social Vulnerability Index. https://www.atsdr.cdc.gov/place-health/php/svi/ (accessed 24 Oct 2026).
- HHS HIPAA public health exception. https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/public-health/index.html (accessed 24 Oct 2026).
- Cramer EY, et al. Evaluation of individual and ensemble probabilistic forecasts of COVID-19 mortality in the United States. PNAS. 2022. https://doi.org/10.1073/pnas.2112532119 (accessed 24 Oct 2026).
- Dagan N, et al. BNT162b2 mRNA Covid-19 vaccine in a nationwide mass vaccination setting. N Engl J Med. 2021. https://doi.org/10.1056/NEJMoa2101765 (accessed 24 Oct 2026).

---

## 11. Cross-specialty comparison

The table is the fastest way to see why a method that works in one specialty fails in another. Read across the row, not down the column.

| Specialty | Core decision | Buyer | Signature data | Primary endpoint | Signature method | Signature pitfall | Regulatory path |
|---|---|---|---|---|---|---|---|
| Oncology | Which therapy, when to escalate, whom to screen | Pharma, payer, cancer center | TCGA/GDC, SEER, GENIE, NCDB | OS, PFS, ORR, MRD | Survival analysis, target trial emulation | Immortal time, surrogate endpoints, lead-time bias | Device, companion diagnostic, drug label |
| Cardiology | Monitor, anticoagulate, implant, admit | Device maker, hospital, payer | PTB-XL, MIMIC-IV-ECG, EchoNet | AF detection, MACE, HF admission | Signal and video models, calibration at prevalence | Prevalence-dependent PPV, device-specific signal | Device, CDS, general wellness boundary |
| Pathology | Cancer, grade, margin, biomarker | Lab, diagnostics company, pharma | CAMELYON, PANDA, TCGA WSI | Slide-level AUROC, QWK, concordance | Multiple-instance learning, stain normalization | Slide and scanner leakage, label noise | Device, IVD, CLIA/CAP laboratory |
| Critical care | Deterioration, sepsis, ventilation, transfer | Hospital, EHR vendor | MIMIC-IV, eICU, PhysioNet 2019 | Mortality, time to antibiotics, deterioration | Time-series hazard models, alert-burden analysis | Label leakage, competing risks, alarm fatigue | Device, CDS, quality program |
| Pediatrics/OB | Dose, growth, delivery, screening | Medicaid, children's hospital, device maker | CTU-UHB, NHANES, PHIS, vital records | Acidemia, preterm birth, adverse drug event | Weight-based rules, rare-event models, growth z-scores | Weight-based dosing, rare events, pregnancy physiology | Pediatric device, drug labeling, research protections |
| Public health | Detect, allocate, advise, mandate | Government, insurer, NGO | NWSS, NHSN, NNDSS, BRFSS | Incidence, Rt, coverage, excess mortality | Nowcasting, forecasting, standardization | Denominators, surveillance lag, case definitions | Public health authority, privacy law, emergency use |

**What the table implies.**

- The same AUROC means different things at 0.5% and 20% prevalence.
- The same survival model is causal in none of these settings without a design.
- The same leakage audit differs: time zero in oncology, patient identity in pathology, measurement timing in critical care, reporting vintage in public health.
- The same regulatory label "AI" maps to four different regimes: device, diagnostic, lab test, and public-health tool.

## 12. Elective selection guide

Use this page to choose one primer for a short engagement or three for a specialty track. Score each criterion from 1 (weak) to 4 (strong). A score of 1 on data access is a veto, not a trade-off.

| Criterion | Question to ask | Evidence to require |
|---|---|---|
| Buyer clarity | Is there a named budget holder whose decision changes? | A person, a budget line, and a decision cadence |
| Data access | Can I obtain decision-relevant data before the deadline? | Signed access, credentialing path, or a named sponsor |
| Method fit | Does my strongest method match the specialty's endpoint? | A prior project or a reproducible baseline |
| Regulatory path | Can I describe the likely pathway in one paragraph? | Intended use, class, and precedent product |
| Career market | Are there employers hiring for this in my region or remotely? | Five real job postings or a referral network |
| Equity exposure | Does the specialty let me work on a population I can serve credibly? | Community or clinical partner |

**Persona recommendations.**

| Background | First choice | Second choice | Reason |
|---|---|---|---|
| Health-system data scientist | Critical care | Public health | Operational buyer, existing EHR data, measurable workflow |
| Pharma or translational scientist | Oncology | Pathology | Endpoint literacy, biomarker strategy, trial and RWE links |
| MedTech or imaging engineer | Cardiology | Pathology | Signal and image pipelines, device regulation, reimbursement |
| Public-sector or policy analyst | Public health | Pediatrics/OB | Population denominators, equity, government buyer |
| Clinical informaticist | Critical care | Pediatrics/OB | Workflow, alerting, order sets, safety |
| Career switcher without clinical data access | Public health | Cardiology (public ECG) | Open datasets and a clear public decision |

**Thirty-day plan for a chosen elective.**

1. Week 1: read the primer, write a one-page decision memo naming the buyer and the action.
2. Week 2: complete the lab on the public dataset; produce the acceptance-criteria table.
3. Week 3: interview two practitioners in the specialty; revise the intended use and workflow.
4. Week 4: write the evidence gap analysis: what the lab does not prove, and what study would.

**What not to do.** Do not choose a specialty because a dataset is easy to download. Do not choose oncology because the survival models look impressive; choose it if you can name the therapy decision and the payer. Do not choose public health if you cannot defend a denominator.

## 13. Assessment rubric

The elective artifact is a specialty evidence brief plus one lab. Weights sum to 100.

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision and buyer | 10 | Names a disease, not a decision | Names the action, user, and buyer | Quantifies the harm of each error and the budget owner |
| Data provenance and access | 15 | Uses a dataset because it is available | Documents source, license, and limitations | Maps decision fit, population fit, access, license, and benchmark fit, with a fallback |
| Endpoint and time zero | 20 | Endpoint is a column in the data | Defines endpoint, window, and censoring | Audits ascertainment and demonstrates the effect of an alternative definition |
| Specialty pitfall audit | 20 | Mentions a generic bias | Tests the named specialty pitfall | Designs a falsification test and shows the result under the fix |
| Evaluation and uncertainty | 15 | Reports accuracy or AUROC only | Reports AUROC, calibration, and confidence intervals | Adds prevalence-adjusted utility, subgroup analysis, and decision-curve or net-benefit analysis |
| Regulatory and equity | 10 | Says "HIPAA" or "FDA" | Names the pathway and one equity risk | Maps intended use to pathway and shows subgroup access or performance evidence |
| Reproducibility and communication | 10 | Notebook runs once | Pinned environment, seed, and a clear memo | Independent rerun, artifact retention, and an executive-ready brief |

**Pass standard.** At least 60 overall, with no zero on endpoint/time zero or specialty pitfall. A brief that reports a strong AUROC but cannot state time zero is not competent, regardless of model quality.

## 14. Key terms

| Term | One-line definition |
|---|---|
| Immortal time bias | Survival time during which the exposure could not yet have occurred is wrongly credited to the exposure. |
| Landmark analysis | Exposure and follow-up are redefined from a fixed time after baseline, excluding earlier events. |
| Lead-time bias | Earlier detection appears to prolong survival even when the date of death is unchanged. |
| Length-time bias | Screening preferentially detects slow-growing disease, making screened patients look better. |
| Overdiagnosis | Detection of a condition that would never have caused symptoms or death. |
| Stage migration | More sensitive staging reassigns patients and improves stage-specific survival without changing true prognosis. |
| RECIST | Standard criteria for tumor response based on imaging measurement. |
| MRD | Minimal residual disease detectable below conventional imaging or pathology thresholds. |
| ctDNA | Circulating tumor DNA measured in blood, used for residual disease and resistance monitoring. |
| Competing risk | An event that prevents the event of interest, such as death preventing discharge. |
| Fine-Gray model | A subdistribution hazard model for competing-risk data. |
| Target trial emulation | Designing an observational analysis to mimic a hypothetical randomized trial. |
| PPV | Positive predictive value: the probability that a positive result is a true positive at a given prevalence. |
| Verification bias | The reference standard is obtained only in a selected subset, distorting sensitivity and specificity. |
| General wellness policy | FDA policy for low-risk products that make no diagnostic or treatment claim. |
| Alert burden | The number of alerts per unit of clinical activity, and their effect on response. |
| Number needed to alert | Alerts required to produce one true event or one clinically meaningful action. |
| Sepsis-3 | Consensus definition using suspected infection plus organ dysfunction. |
| SOFA | Sequential Organ Failure Assessment score used to quantify organ dysfunction. |
| Phoenix criteria | Pediatric sepsis consensus criteria based on suspected infection and organ dysfunction. |
| CTG | Cardiotocography, the simultaneous recording of fetal heart rate and uterine contractions. |
| FIGO classification | Standard terminology for intrapartum fetal heart rate patterns. |
| LMS method | Method for expressing growth as a z-score using skewness, median, and coefficient of variation. |
| Allometric scaling | Size-based scaling of clearance or dose, often using weight to a fractional exponent. |
| Multiple-instance learning | Learning from bags of instances when only the bag label is known. |
| Stain normalization | Transforming color appearance to reduce batch and scanner variation in histology. |
| QWK | Quadratic weighted kappa, an agreement measure for ordinal grades. |
| Nowcasting | Estimating the current state of an epidemic after correcting for reporting delay. |
| Reporting triangle | A table of counts by event date and report date used to model delays. |
| WIS | Weighted interval score, a proper scoring rule for probabilistic forecasts. |
| Age standardization | Adjusting rates to a common age structure for fair comparison. |
| Ecologic fallacy | Inferring individual-level risk from group-level associations. |
| Case definition | The criteria that determine whether a person counts as a case. |
| Capture-recapture | Estimating an unobserved population size from overlap between incomplete sources. |

## 15. Cross-cutting sources and verification notes

**Regulation and standards.**

- FDA. Artificial intelligence and machine learning in software as a medical device. https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices (accessed 24 Oct 2026).
- FDA. Clinical decision support software guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software (accessed 24 Oct 2026).
- EU Artificial Intelligence Act, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj (accessed 24 Oct 2026).
- EU In Vitro Diagnostic Regulation, Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 24 Oct 2026).
- CMS Clinical Laboratory Improvement Amendments. https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments (accessed 24 Oct 2026).
- ICH E9(R1) addendum on estimands. https://www.ich.org/page/efficacy-guidelines (accessed 24 Oct 2026).
- TRIPOD+AI reporting guideline. https://www.tripod-statement.org/ (accessed 24 Oct 2026).
- PROBAST risk-of-bias tool. https://www.probast.org/ (accessed 24 Oct 2026).

**Data and infrastructure.**

- All of Us Research Program. https://www.researchallofus.org/ (accessed 24 Oct 2026).
- PhysioNet. https://physionet.org/ (accessed 24 Oct 2026).
- NCI Cancer Imaging Archive. https://www.cancerimagingarchive.net/ (accessed 24 Oct 2026).
- OHDSI OMOP Common Data Model. https://ohdsi.github.io/CommonDataModel/ (accessed 24 Oct 2026).
- HL7 FHIR. https://hl7.org/fhir/ (accessed 24 Oct 2026).

**Verification notes for this module.**

- SEER coverage, GENIE release size, eICU hospital count, PHIS hospital count, NWSS site count, and Medicaid's share of births change over time. Each is marked verify in the text; confirm against the primary source before use in a grant or contract.
- The FDA laboratory-developed test rule was vacated in litigation in 2025. Confirm the current enforcement position before relying on it.
- The EU AI Act and IVDR transition dates are phased. Confirm the application date for the specific device class and jurisdiction.
- AJCC staging edition, HER2 scoring guidelines, and pediatric sepsis criteria are updated periodically. Cite the version used in the analysis.
- Fitbit Heart Study positive predictive value, TREWS prospective effect size, and a small number of dataset counts are marked verify; the underlying studies are real, but the exact figure should be checked against the published article before quotation.

---

> **Previous:** [`22_sources_and_further_reading.md`](22_sources_and_further_reading.md)
> **Next:** [`24_hardware_robotics_labs.md`](24_hardware_robotics_labs.md)
> **Related:** [`10_medical_imaging.md`](10_medical_imaging.md), [`11_physiological_signals_wearables.md`](11_physiological_signals_wearables.md), [`12_omics_precision_medicine.md`](12_omics_precision_medicine.md), [`14_medical_devices_samd_regulation.md`](14_medical_devices_samd_regulation.md)

