# Module 19 — Capstone and Hackathons

> **Program:** Biomedical Data Science for Industry (2026 Edition)
> **Time:** 80–120 hours (8 lecture/workshop, 12 studio, 60–100 team project)
> **Track fit:** Common core; required for all three tracks
> **Prerequisites:** Modules 00–06; at least two track modules from 08–14B; Modules 15–18 before final defense
> **Tools:** Python, Git, an environment lockfile, experiment tracking, cloud or institutional compute, a signed data-use agreement or IRB determination, and a demo environment
> **Industry question:** "Can you take an ambiguous biomedical problem, reduce it to a real decision, produce defensible evidence, and defend the result to a clinician, an engineer, and a compliance officer in the same room?"

> **Status note (as of 24 October 2026).** Competition facts here come from organizer pages and announcements available at the source cutoff. Events are labeled **concluded**, **closed with results pending**, **announced only**, **recurring**, or **status unverified**. No winner names, prize amounts, or scores are asserted unless the source brief states them. Re-check eligibility, data terms, and deadlines before assigning any competition to a team.

---

## 1. Why the capstone matters

The capstone is the only part of the program where every skill is exercised at once under employment-like constraints. Labs isolate a technique; the capstone forces integration: choosing a decision, negotiating data access, building something that runs, proving it works, and explaining why anyone should believe it.

It differs from a Kaggle-style submission in ways that hiring managers notice.

| Dimension | Competition submission | Capstone |
|---|---|---|
| Objective | Maximize a fixed leaderboard metric | Support a named decision with a defensible metric and harm analysis |
| Data | Supplied and schema-normalized | Acquired, governed, documented, often incomplete |
| Success | Rank | Adoption, safety, equity, credible evidence path |
| Lifetime | Ends at the deadline | Needs monitoring and revalidation |
| Failure | A lower score | A patient harmed, a clinician misled, a regulator unconvinced |

A strong capstone answers five questions in order. **What decision changes, and who owns it?** If no one owns it, the project is a demo. **What is the comparator?** Current practice, not a strawman model. **What evidence would make a skeptical clinician act?** Usually temporal or external validation, calibration, and a workflow argument. **What happens when it is wrong?** False positive and false negative costs, alert burden, mitigation. **What stops this reaching a patient or customer?** Regulation, data rights, integration, reimbursement, maintenance.

The capstone is also the portfolio centerpiece for Module 20. A candidate who can walk an interviewer through a failed validation, a data-access constraint, and a revised scope is more credible than one who reports a single impressive number.

## 2. Learning outcomes

By the end of this module, you can:

- **Frame** an ambiguous problem as a decision with an owner, workflow, comparator, and measurable outcome.
- **Scope** a 12-week project so the minimum viable evidence package survives a three-week data delay.
- **Acquire and govern** data under the correct access tier, completing training, agreements, and de-identification review before analysis.
- **Design** validation that mirrors the deployment population, including temporal, site-held-out, and subgroup analyses.
- **Build** a reproducible baseline and candidate model with versioned data, pinned environments, and recorded seeds.
- **Evaluate** discrimination, calibration, decision utility, alert burden, and subgroup performance together.
- **Triage** the regulatory pathway and name the artifact each regime requires.
- **Lead** a small cross-functional team, run a weekly rhythm, and escalate risks with options rather than complaints.
- **Communicate** the work in a 10-minute executive pitch, a 20-minute technical defense, and a one-page readout.
- **Critique** another team's capstone on evidence, governance, and equity, not only on model choice.

## 3. Capstone architecture

A capstone is not a model with a report attached. It has four layers, and weakness in any one caps the score.

```
┌──────────────────────────────────────────────────────────────┐
│ 4. GOVERNANCE   data rights · regulatory path · risk · equity │
├──────────────────────────────────────────────────────────────┤
│ 3. SYSTEM       pipeline · serving · latency · monitoring     │
├──────────────────────────────────────────────────────────────┤
│ 2. EVIDENCE     splits · baseline · calibration · subgroups   │
├──────────────────────────────────────────────────────────────┤
│ 1. DECISION     owner · workflow · comparator · cost of error │
└──────────────────────────────────────────────────────────────┘
```

Weak capstones invert the pyramid: eleven weeks on layer 2, one week retrofitting layers 1, 3, and 4. The milestone plan forces layer 1 in week 1 and layer 4 in week 2.

Every capstone must contain this minimum viable evidence package. Stretch goals are optional; these are not.

| Component | Minimum acceptable | Reviewer |
|---|---|---|
| Decision statement | One decision, one owner, one frequency | Clinical or operational mentor |
| Intended use | Population, setting, inputs, output, user, contraindications | Regulatory mentor |
| Comparator | Current practice quantified on the same cohort | Technical mentor |
| Data provenance | Source, version, access date, license, de-identification | Data steward |
| Split strategy | Held-out set that mimics deployment, with justification | Technical mentor |
| Baseline | Simple, interpretable, honestly tuned | Technical mentor |
| Primary metric | Tied to the decision, with a pre-specified threshold | Clinical mentor |
| Calibration | Reliability plot plus slope or intercept | Technical mentor |
| Subgroups | At least three pre-specified groups with intervals | Equity reviewer |
| Error analysis | 20–30 reviewed errors with a taxonomy | Clinical mentor |
| Regulatory triage | One-page memo naming the regime and binding artifact | Regulatory mentor |
| Monitoring | Drift, performance, alert volume, retraining triggers | Engineering mentor |

Scope in three tiers. **Core** is the evidence package, working demo, and dossier; if it is not core, it cannot delay the defense. **Stretch** is an external site, shadow mode, or cost-effectiveness model, attempted only after freeze. **Out of scope** is named explicitly: new data collection, regulatory submission, clinical trial, production deployment. Naming exclusions signals maturity.

Artifacts are reused, not rebuilt: the pipeline from Module 03, estimand and validation design from 05, model card from 07 and 18, intended use and risk analysis from 14, serving and drift plan from 15, threat model from 16, and value case from 17.

## 4. Twelve-week milestone plan

The third term runs weeks 27–36. Teams starting in week 25, or continuing after week 36, use all twelve weeks. A ten-week compression merges weeks 3–4 and 9–10; never compress weeks 1–2 or week 11.

| Week | Focus | Deliverable at gate | Gate | Reviewer |
|---:|---|---|---|---|
| 1 | Decision and scope | Charter: decision, owner, population, comparator, metric, harms | G0 | Clinical + program lead |
| 2 | Governance and data | Access plan, DUA/IRB status, de-identification, risk register | G1 | Data steward + privacy |
| 3 | Data audit | Provenance, missingness, label provenance, leakage screen | G1 | Technical mentor |
| 4 | Baseline | Reproducible baseline, locked split, confidence intervals | G2 | Technical mentor |
| 5 | Validation design | Analysis plan: splits, metrics, subgroups, stopping rules | G2 | Biostatistics mentor |
| 6 | Candidate | First candidate, error analysis, decision-curve comparison | — | Technical mentor |
| 7 | Midpoint | Go / re-scope / stop recommendation | G3 | Full panel |
| 8 | Robustness | Site-held-out or temporal validation, calibration, subgroups | G3 | Clinical mentor |
| 9 | Regulation and ethics | Triage memo, intended use, model card, datasheet | G4 | Regulatory + equity |
| 10 | System and value | Serving diagram, monitoring plan, value case | G4 | Engineering + commercial |
| 11 | Freeze and rehearse | Frozen artifacts, dossier, two full rehearsals | G5 | Program lead |
| 12 | Defense | Pitch, technical defense, live demo, peer critique | G5 | External panel |

Gate rules. **G0:** an owner is named; otherwise choose another brief or reframe as a methods project. **G1:** data are legally accessible or a signed fallback exists; no gate, no modeling. **G2:** split and metric are written down before candidate results; later changes are logged. **G3:** the panel makes an explicit recommendation. **G4:** regulatory path and equity analysis are complete. **G5:** nothing changes after freeze except a factual correction, announced on stage.

Weekly rhythm: 30-minute Monday stand-up for blockers and the decision log; 120-minute Wednesday studio for mentor critique; 45-minute Friday demo of working software only; a written status using the template in Section 17.

Data access is the most common cause of failure. File the primary credentialed request in week 1; pre-approve **fallback A**, a public or demo equivalent with the same schema; pre-approve **fallback B**, a synthetic generator calibrated to published summary statistics and labeled as synthetic everywhere. The scientific question must survive the fallback. If it cannot, say so at G0 and choose a different brief.

## 5. Team roles

Teams of four are the default. Teams of three merge the evidence lead into the lead role; teams of five split engineering from modeling.

| Role | Owns | Primary artifacts | Failure if absent |
|---|---|---|---|
| Lead / product owner | Scope, schedule, stakeholders, decision log | Charter, status, readout | Scope creep, missed gates |
| Data engineer | Acquisition, pipeline, quality, lineage | Provenance, pipeline, lockfile | Unreproducible results, silent leakage |
| Modeler / analyst | Baseline, candidate, validation, errors | Analysis plan, model card | Metric chasing, no baseline |
| Evidence and governance lead | Study design, regulation, equity | Dossier, risk register, memo | Last-minute compliance scramble |
| Clinical or domain liaison | Workflow, label meaning, acceptance | Workflow map, error review | Valid solution to the wrong problem |

| Deliverable | Lead | Data eng. | Modeler | Evidence | Clinical |
|---|---|---|---|---|---|
| Charter | A | C | C | C | C |
| Data-access pack | C | A | I | C | I |
| Analysis plan | C | C | C | A | C |
| Baseline and validation | I | C | A | C | C |
| Regulatory memo | C | I | I | A | C |
| Value case | A | I | C | C | C |
| Defense | A | C | C | C | C |

A = accountable, C = consulted, I = informed.

Team grades hide free riders and inflate strong presenters. Use a weekly **contribution log**; require an **individual defense** covering any part of the work, not only the member's own; and apply a confidential **peer adjustment** of at most ±15%, with written justification reviewed by the program lead.

## 6. Eight capstone archetypes

Choose the archetype that matches the decision, not the data you happen to have. A team with imaging data but no clinical owner should choose A5, not A1.

| # | Archetype | Core decision | Typical data | Evidence bar | Characteristic failure |
|---|---|---|---|---|---|
| A1 | Regulated detection or triage | Which case is read, escalated, or recalled | Imaging, ECG, pathology | Site-held-out validation, calibration, reader-study logic, SaMD path | Treating a research AUC as a clinical claim |
| A2 | Prognostic risk stratification | Who receives an intervention or outreach | EHR, registry, claims | Temporal validation, decision curves, subgroup equity | Unactionable score with no intervention capacity |
| A3 | Operational forecasting | How to staff, schedule, or allocate | Scheduling, flow, claims | Backtested forecast, calibration, cost of error | Optimizing a proxy operations does not control |
| A4 | Documentation or agentic workflow | Whether a draft is accepted, edited, or rejected | Notes, FHIR, guidelines | Task success, factuality, omission, human review | Fluent output with no provenance or write audit |
| A5 | Evaluation and benchmark science | Which method or model to adopt | Public benchmarks, model outputs | Reproducible harness, statistical comparison, grader validity | Leaderboard overfitting and contamination |
| A6 | Translational biomarker or omics | Which target, variant, or group advances | Omics, variants, single-cell | External cohort, multiplicity control, assay context | Discovery on one cohort with no replication |
| A7 | Trial and regulatory evidence | Whether a trial, endpoint, or strategy is viable | Trial data, RWE, synthetic SDTM | Estimand clarity, bias analysis, precedent | Causal claim from a convenience sample |
| A8 | Population and screening program | Where to allocate screening or surveillance | Surveillance, registries, SDOH | Equity-weighted evaluation, cost per case found | Aggregate accuracy hiding subgroup harm |

## 7. Project briefs

Forty-four briefs follow, condensed from the program research brief. Each states the decision changed, data, primary metric, regulatory or ethical angle, and difficulty from 1 (course-lab scale) to 5 (multi-site, high-risk, publication-grade). Confirm licenses in Module 21 before use.

### 7.1 Imaging

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 1 | Aneurysm triage on head CTA | Which scan the on-call radiologist reads next | RSNA 2025 aneurysm data; open CTA | Macro AUC; sensitivity at 2 FP/scan | High-risk SaMD; triage claim needs reader-study and workflow evidence | 5 |
| 2 | Knee MRI multi-finding classifier | Whether a knee MRI is prioritized or deferred | RSNA 2026 knee MRI | Macro ROC AUC | LLM-derived labels; leakage disputes; 12 findings with different harms | 5 |
| 3 | Lumbar spine severity grading | Surgical versus conservative referral | RSNA 2024 lumbar spine | Weighted multi-class log loss | Class imbalance; inter-reader severity variability | 4 |
| 4 | Pneumothorax triage under site shift | Which chest radiograph is escalated after hours | CheXpert; MIMIC-CXR | AUROC plus subgroup gap | SaMD; bias documentation; scanner shift | 3 |
| 5 | Tuberculosis screening on device | Whether to refer for confirmatory testing in a low-resource clinic | CODA-TB; HeAR embeddings | Sensitivity at fixed specificity; model size | Low-resource deployment; consent; connectivity | 4 |
| 6 | BI-RADS extraction from reports | Which screening exams need double reading | Institutional reports; MIMIC-style corpus | Macro-F1; calibration | PHI; de-identification; ambiguous language | 2 |
| 7 | Federated glioma segmentation | Whether a consortium can train without pooling images | BraTS; FeTS | Dice; HD95; communication cost | Privacy governance; site heterogeneity; ownership | 5 |
| 8 | Diabetic retinopathy grading across cameras | Which patients are recalled; which camera to deploy | APTOS; DeepDR | Quadratic weighted kappa | Color and ethnicity bias; camera shift | 3 |

### 7.2 ECG, sleep, and physiological signals

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 9 | Chagas triage at 5% capacity | Who receives the scarce serology slot | CODE-15%; SaMi-Trop; PTB-XL | Precision at top 5% | Screening ethics; weak labels; endemic access | 4 |
| 10 | Age-conditioned dementia risk from PSG | Who receives cognitive follow-up after a sleep study | Human Sleep Project | Age-conditioned AUROC within ±2 years | Incidental findings; consent; age shortcut | 5 |
| 11 | Atrial fibrillation from a wearable | Whether a user is invited to a confirmatory ECG | PhysioNet AF/PPG-style | F1; false alarms per hour | Consumer versus medical claims; overdiagnosis | 3 |
| 12 | Sepsis early warning | Who gets rapid-response review in six hours | MIMIC-IV; eICU | AUROC; utility; lead time | Alarm burden; automation bias; liability | 4 |
| 13 | EEG seizure classification | Whether a segment is queued for review | TUH EEG | Weighted kappa | Montage and device shift; pediatric subgroups | 4 |
| 14 | Sleep staging from a wearable | Whether a consumer sleep report is clinically usable | Sleep-EDF | Cohen's kappa | Consumer versus clinical claim; label noise | 3 |
| 15 | ICU deterioration with fairness audit | Whether a score can deploy without widening disparities | MIMIC-IV | AUROC; equalized-odds gap | Redlining risk; proxy variables | 3 |

### 7.3 EHR, claims, and operations

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 16 | Emergency department boarding | Where to open surge beds tonight | MIMIC-IV-ED | MAE; calibration | Operational claim; staffing equity | 2 |
| 17 | Readmission with social determinants | Which discharged patients get a transition bundle | MIMIC-IV plus area deprivation index | AUROC; net benefit | Proxy discrimination; intervention capacity | 3 |
| 18 | Post-transplant survival with censoring | Which candidates are counseled toward which regimen | CIBMTR-style registry | Stratified concordance | Competing risks; registry governance; small subgroups | 4 |
| 19 | Operating-room scheduling | How to sequence elective cases next week | Public OR logs | Utilization; overtime hours | Staff equity; clinical priority; schedule stability | 3 |
| 20 | Claims cost anomaly detection | Which claims are audited | CMS SynPUF | Precision at k | Payer audit fairness; false accusation; appeals | 2 |
| 21 | FHIR medication reconciliation agent | Whether a reconciled list is accepted into the chart | MedAgentBench-style FHIR environment | Pass@1 | Hallucinated writes; audit trail; sign-off | 4 |

### 7.4 Clinical NLP, LLMs, and agents

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 22 | Discharge summary drafting | Whether a resident edits or rewrites the draft | MIMIC-IV-Note | Factuality; omission rate | Hallucination liability; attribution; sign-off | 3 |
| 23 | Prior-authorization letter drafting | Whether staff send the letter with edits | Synthetic policies and notes | Rubric score; human preference | No automated denial; payer-rule transparency | 2 |
| 24 | Trial eligibility matching | Which patients are screened for a trial | TrialGPT-style; ClinicalTrials.gov | nDCG; precision | False exclusion; equity in screening; override | 3 |
| 25 | HealthBench-style evaluator | Whether a model response is safe to release | Public evaluation subset | Correlation with physician grades | Grader bias; rubric validity; inter-rater agreement | 3 |
| 26 | Multilingual triage | Whether a patient is routed to urgent care | Ilera-Bench-style | Accuracy; safety recall | Language equity; interpreter workflow; dialect | 3 |
| 27 | Radiology report to structured findings | Whether structured data can populate a registry | MIMIC-CXR | F1 per finding | PHI in prompts; negation and uncertainty | 2 |

### 7.5 Genomics, pathology, and public health

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 28 | Missense variant effect prediction | Whether a variant is reported as likely pathogenic | CAGI7 deep mutational scanning | Spearman; ROC | ACMG framing; actionability; assay context | 4 |
| 29 | Polygenic risk score portability | Whether a PRS is offered to a non-European patient | CAGI7 PRS challenges | AUC across ancestries | Ancestry bias; consent; uncertain utility | 4 |
| 30 | Pathology foundation-model probe | Whether a foundation model is worth licensing | SLC-PFM; TCGA | Balanced accuracy | Compute and data terms; clinical validity; lock-in | 5 |
| 31 | Single-cell label projection | Which annotation method the lab adopts | Open Problems in Single-Cell Analysis | Macro-F1 versus logistic baseline | Reproducibility; benchmark overfitting | 3 |
| 32 | Suicide-narrative abstraction | Which death investigations get prevention review | NVDRS; synthetic narratives | F1 | Mandated-reporter ethics; stigmatizing language | 3 |
| 33 | Wastewater and syndromic nowcasting | When to issue a public advisory | CDC NWSS plus ED visits | MAE; weighted interval score | Public communication; small-area privacy | 3 |
| 34 | FHIR quality-measure gap | Which clinics need reporting support | Synthea; US Core; DEQM | Compliance rate | CMS reporting; synthetic-data limits | 2 |

### 7.6 Hardware, sensors, and robotics

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 35 | ECG front-end hardware-in-the-loop | Whether a detector is robust enough for a real monitor | Synthetic ECG + noise; public ECG | Sensitivity at fixed false alarms; latency | IEC 60601/62304 evidence; saturation and lead-off | 3 |
| 36 | PPG skin-tone and motion audit | Whether a wearable heart-rate feature can ship | Public or approved PPG; device metadata | Stratified MAE; bias; coverage | Equity; FDA/CE claims; sensor fit | 3 |
| 37 | TinyML arrhythmia on an MCU | Whether the model fits the device power/latency budget | PTB-XL or MIT-BIH; target board | Accuracy after quantization; latency; RAM/flash | Model update control; PCCP; subgroup performance | 4 |
| 38 | Surgical phase recognition | Whether a robot can provide phase-aware assistance | Cholec80; public surgical video | Macro-F1; segmental edit distance | Patient privacy; consent; automation bias | 4 |
| 39 | Robot kinematics and safety monitor | Whether a simulated arm stays inside force/workspace limits | 2-link or dVRK simulation; synthetic forces | Safety violations; stop latency; false stops | IEC 80601-2-77; ISO 14971; fail-safe behavior | 3 |
| 40 | CGM compression-low detection | Whether a closed-loop system should suspend insulin | OhioT1DM; public CGM | Detection sensitivity; false alarms per day | Hypoglycemia safety; alarm fatigue; clinical review | 4 |
| 41 | Exoskeleton gait intent detection | Whether assistance timing is safe and useful | Public IMU/EMG; open gait data | Intent accuracy; assistance latency; stumble rate | IEC 80601-2-78; user training; fall risk | 4 |

**Hardware brief rule:** the deliverable must include a signal-chain diagram, device/firmware version, calibration assumptions, a hardware-in-the-loop or simulation test, and a failure-mode register. A notebook score alone is not a device claim.

### 7.7 Efficient AI and on-device inference

| # | Brief | Decision changed | Data | Metric | Regulatory / ethical angle | Diff. |
|---:|---|---|---|---|---|---:|
| 42 | Quantized ECG on a wearable MCU | Whether the device can run arrhythmia detection locally | PTB-XL or MIT-BIH; target board | Sensitivity at fixed false alarms; p99 latency; joules/inference | PCCP; calibration shift; subgroup performance | 4 |
| 43 | On-device LLM triage | Whether a phone can draft a safe triage response offline | HealthBench subset; synthetic notes | Factuality; safety recall; latency; memory | Hallucination; privacy; human review | 5 |
| 44 | Surgical perception on edge hardware | Whether tool/phase recognition meets the robot control loop | Cholec80; dVRK simulation | p99 latency; frame-drop rate; macro-F1 | IEC 62304/80601-2-77; fail-safe fallback | 4 |

**Efficient-AI brief rule:** report the uncompressed baseline, compression method, calibration data, target hardware, p50/p95/p99 latency, energy, peak memory, subgroup performance, and regulatory change path. A desktop benchmark is not evidence.

### 7.8 Expanded example: Brief 9, Chagas triage at 5% capacity

*Context.* Serological confirmation is scarce in endemic regions. Ranking ECGs by Chagas risk can direct testing, but only at the operating point the program can staff. *Minimum viable experiment:* train on weakly labeled CODE-15%, validate on SaMi-Trop confirmed positives with PTB-XL as an assumed-negative stress test, and report precision in the top 5% with a bootstrap interval against a logistic model on standard ECG intervals and a foundation-model embedding. *Stretch:* model label noise and prevalence shift, then simulate the number needed to test per case found. *Kill criterion:* if the top-5% precision interval overlaps the simple model's interval on confirmed positives, added complexity is not justified.

The same structure applies to Briefs 21 and 29: define the unsafe event before modeling (a hallucinated medication write; a PRS offered without calibration), and make the kill criterion a patient-safety threshold rather than a metric difference.

## 8. Hackathon formats

Hackathons and capstones train different muscles. A hackathon rewards framing speed, teamwork, and a convincing demo. A capstone rewards evidence, governance, and endurance. Run both.

| Format | Duration | Team | Scope | Deliverable | Judging emphasis | Main risk |
|---|---|---|---|---|---|---|
| Sprint hackathon | 24 h | 4–6 | One workflow, one metric, public or mock data | Demo plus 3-minute pitch | Framing, feasibility, demo | No validation; shallow clinical input |
| Grand hack | 48 h | 5–8 | Clinical unmet need to prototype | Prototype, business model, pitch | Need validation, team, pathway | Solution seeking a problem |
| Datathon | 5–7 days | 3–5 | One dataset, several questions | Analysis, notebook, short paper | Rigor, reproducibility, insight | Metric chasing; access shortcuts |
| Capstone | 12 weeks | 3–5 | Decision to evidence dossier | Prototype, dossier, defense | Evidence, governance, adoption | Scope creep; data-access failure |

**24-hour sprint.** Hours 0–2 choose one decision and write the one-page charter. 2–4 inspect data, find labels and leakage traps. 4–9 build the simplest end-to-end baseline. 9–15 improve one thing only. 15–18 validate on a held-out slice and compute a subgroup table. 18–21 build the demo and three-minute story. 21–23 rehearse and record a fallback. 23–24 buffer. No claim without a baseline; state the split; show one failure case; never demo on training data.

**48-hour grand hack.** Spend the first six hours on need, workflow, and existing solutions. Interview at least three clinicians or patients, or use a documented need assessment. Then build a narrow prototype demonstrating one step of the workflow. Venture-style judging rewards a clear user, a credible pathway, and understanding of the incumbent alternative. Enthusiasm is not evidence.

**One-week datathon.** Compress a research project. Assign a data steward, freeze the schema on day 1, and require a written analysis plan before day 3. Use the capstone evidence package reduced to one primary and two secondary analyses. Publish the notebook and lockfile. The classic failure is fitting the public leaderboard on day 6 and discovering on day 7 that the hidden split rewards a different operating point.

**Twelve-week capstone.** The only format where regulatory triage, monitoring, and a value case are mandatory, and the only one that produces an artifact an employer accepts as evidence of judgment rather than enthusiasm.

| Question | Typical competition | Course rule |
|---|---|---|
| External data | Often prohibited or restricted | Allowed if licensed and disclosed |
| Pretrained weights | Often allowed with disclosure | Allowed; record source, license, training data |
| Leaderboard probing | Limited by daily quota | One probe per week; log every submission |
| Team merging | Sometimes allowed | Not allowed after G2 |
| Code release | Sometimes required | Required in the dossier |
| PHI in prompts | Rarely policed | Prohibited; use de-identified or synthetic text |

## 9. Competition landscape, 2025–2026

Use competitions as calibration, not as a curriculum. They show what the frontier rewards; they rarely test governance or maintenance.

**Legend:** *Concluded* means the event took place and results were reported. *Announced only* means announced but not completed at the cutoff. *Closed, results pending* means the submission window closed but outcomes were not published at the cutoff. *Recurring* means a cadence exists but the specific edition must be checked. *Status unverified* means the brief could not confirm details.

| Program | Edition / task | Status | What to learn |
|---|---|---|---|
| PhysioNet / CinC Challenge | 2025: Chagas triage from 12-lead ECG | Concluded | A capacity-constrained metric (precision in top 5%) changes model selection more than AUROC |
| PhysioNet / CinC Challenge | 2026: cognitive impairment from polysomnography (PSG) | Task and data released; results not asserted | Age-conditioned AUROC removes a trivial confounder; hidden sites punish overfitting |
| RSNA AI Challenge | 2024: lumbar spine degenerative classification | Concluded; organizer-reported 1,874 teams | Localization then ROI classification; label-noise reduction |
| RSNA AI Challenge | 2025: intracranial aneurysm detection | Concluded; organizer-reported 1,147 teams | Vessel-first pipelines; location-aware models; macro AUC |
| RSNA AI Challenge | 2026: knee abnormality, 12 findings | Closed 22 October 2026; results pending | Label provenance and leakage disputes at scale; multi-finding harm weighting |
| MICCAI challenges | Annual: BraTS, FLARE, HECKTOR, and others | Recurring; per-task 2026 status varies | Held-out site evaluation and task-specific rules |
| NeurIPS competition track | 2025: CURE-Bench, SLC-PFM, EEG Foundation, PNPL | Concluded | Clinician-graded LLM evaluation; pathology at 300M+ images; cross-subject EEG |
| DREAM Challenges | 2025: SEA-AD, COVID causal diagram, olfactory, Target 35 | Concluded | Causal diagram elicitation; omics benchmarks with held-out truth |
| Open Problems in Single-Cell Analysis | 14 problems, roughly 140 methods, 127 datasets | Recurring | Simple logistic baselines remain competitive; benchmark design matters |
| CAGI | CAGI7, 14 challenges: variant effect, PRS, splicing | Concluded | Clinical interpretation framing; ancestry portability |
| HL7 FHIR Connectathon | Three to four per year with payer and provider partners | Recurring | Interoperability is tested against real servers, not specifications |
| MIT Hacking Medicine Grand Hack | Annual, Boston | 2026 dates unverified | Need validation before prototyping; venture-style judging |
| MIT Critical Data datathons | Global, 39+ countries | Recurring | Local data and questions with shared methods |
| MedGemma Impact Challenge | Prototypes on MedGemma, MedSigLIP, HeAR, MedASR | Concluded; winners reported March 2026 | Low-resource and on-device constraints; open-weight adaptation |
| J&J Innovation Data and AI QuickFire | Foundation models, multi-omics, agentic R&D | Closed February 2026 | Industry statements reward translation, not benchmark scores |
| Roche Ophtha Innovation Challenge | 2026: diabetic macular edema access; RWD | Submissions October–December 2026; not concluded | Access and real-world data, not only accuracy |
| Microsoft and UNC Health hackathon | OMOP semantic models, KPI agents | Announced for 9 November 2026; not held | Standards-based analytics; agents on real schemas |
| XPRIZE Healthspan | Multi-year healthy aging, $101M | Finalists 2026; not concluded | Long-horizon evidence and trial design |
| CDISC AI Innovation Challenge | Synthetic SDTM/ADaM; agentic TFL; traceability | Annual; current cycle status unverified | Regulatory-grade standards and auditability |

**Concluded versus announced, explicitly (24 October 2026).** Concluded: PhysioNet 2025 (Chagas), RSNA 2024 and 2025, the NeurIPS 2025 competition track, DREAM 2025, CAGI7, and the MedGemma Impact Challenge, whose winners were reported in March 2026. Closed but with unpublished outcomes: RSNA 2026 (submission closed 22 October 2026). Announced or open, not concluded: the Microsoft and UNC Health hackathon scheduled for 9 November 2026, the Roche Ophtha submission window running October to December 2026, and XPRIZE Healthspan, which remains a multi-year program with finalists named in 2026. The J&J Data and AI QuickFire closed in February 2026; closure is not the same as published results.

**Not asserted here, and why.** Results, winners, and prize allocations for PhysioNet 2026, RSNA 2026, and the 2026 MICCAI tasks are not stated because the source brief does not confirm them. MIT Hacking Medicine Grand Hack 2026 dates, the current CDISC challenge cycle, and PhysioNet 2027 are unverified. These gaps are deliberate: a course that invents a winner teaches students to invent evidence.

**How to use this table honestly.** Do not cite team counts as difficulty; they reflect marketing, prizes, and Kaggle defaults. Do not present an announced event as completed: "announced for November 2026" is not "held." Do not repeat a winner's method as a validated clinical result; a winning pipeline is benchmark evidence under a specific metric. When a source says "winning approaches included," read it as a reported trend, not a controlled comparison. Re-check every URL before assigning a brief.

What transfers to industry: metric-first thinking, hidden-split discipline, localization-then-classification for 3D imaging, label-noise handling, simple baselines as a sanity check, and pretrained backbones as a starting point. What does not: leaderboard rank as clinical value, submission-only code, ignoring calibration because the metric was AUC, and licensing weights without checking terms.

## 10. Scoping and validation playbook

### 10.1 Seven questions before any code

1. **Decision:** what action changes, and who takes it?
2. **Trigger:** when is the output available, and how long does the user have?
3. **Population:** who is eligible, who is excluded, who is missing from the data?
4. **Comparator:** what happens today, measured on the same cohort?
5. **Error economics:** what does a false positive cost, and a false negative?
6. **Evidence threshold:** what would make the owner change practice?
7. **Stop condition:** what result would make you recommend not deploying?

If an answer is "we will figure it out later," the project is not scoped.

### 10.2 Validation ladder

| Level | Design | What it proves | Use |
|---|---|---|---|
| L0 | Random split | Nothing about deployment | Debugging only |
| L1 | Patient-level held-out | Generalization to new patients, same sites and era | Minimum for a course project |
| L2 | Temporal split | Generalization forward in time | Required for most capstones |
| L3 | Site- or scanner-held-out | Generalization to a new environment | Required for regulated or multi-site claims |
| L4 | Prospective or shadow mode | Performance and workflow impact in use | Rare; describe the design |

A capstone must reach L2. A1 and A8 archetypes should reach L3 or justify why they cannot.

### 10.3 Choosing the primary metric

| Decision shape | Prefer | Avoid | Reason |
|---|---|---|---|
| Fixed review capacity | Precision at k, recall at k | AUROC | Capacity, not ranking, is the constraint |
| Rule-out with a safe threshold | Sensitivity at fixed specificity; NPV | Accuracy | The cost of a miss dominates |
| Risk-based outreach | Calibration; net benefit | AUROC alone | Contacted patients depend on absolute risk |
| Ranking a queue | AUPRC; lift at k | Accuracy | Prevalence is low |
| Time-to-event | Concordance; time-dependent AUC | Static AUC | Censoring changes the estimand |
| Forecasting | MAE; weighted interval score | R-squared | Decisions depend on error magnitude |
| LLM task | Task success; factuality; omission; unsafe writes | BLEU or ROUGE alone | Fluency is not correctness |
| Segmentation | Dice; HD95; lesion-wise detection | Pixel accuracy | Imbalance and boundary error |

### 10.4 Pre-specified analysis plan

Before candidate results, write the primary metric and threshold; split definition and the date separating train from test; at least three subgroups justified by harm or access; missing-data handling, including whether imputation is fit inside the training fold; confidence interval method and resamples; multiplicity plan; sensitivity analyses for label noise and prevalence shift; and the stopping rule for iteration. Post-hoc changes are allowed but logged with a timestamp and reason. Undisclosed post-hoc selection is the most common quiet p-hacking in capstones.

### 10.5 Leakage screen

| Leakage | Test | Fix |
|---|---|---|
| Patient | Same patient in train and test | Group split by patient identifier |
| Study | Multiple images from one study | Split by accession or study |
| Temporal | Future information in features | Rebuild features as of prediction time |
| Pretraining | Backbone trained on the test set | Check model card; use an external set |
| Label | Post-outcome fields in features | Drop fields unavailable at decision time |
| Duplicate | Near-duplicate images or notes | Hash and perceptual-hash deduplication |
| Site | Scanner or site encoded in metadata | Site-held-out validation; metadata audit |

Report confidence intervals for every primary and subgroup metric. For a proportion near 0.8 with 200 positive cases, the 95% interval is roughly ±0.055; a 0.02 improvement is not distinguishable. State the minimum detectable difference and do not interpret noise as progress.

## 11. Required deliverables and evidence dossier

The dossier is one versioned folder submitted at G5: the artifact an employer or regulator would request.

| # | Artifact | Contents | Weight |
|---:|---|---|---:|
| 1 | Charter | Decision, owner, population, comparator, metric, harms, scope | 5% |
| 2 | Data provenance | Source, version, access date, license, de-identification, quality | 8% |
| 3 | Analysis plan | Pre-specified splits, metrics, subgroups, stopping rules | 8% |
| 4 | Reproducible code | Repository, lockfile, seeds, one-command run | 12% |
| 5 | Results | Baseline and candidate point estimates with intervals | 10% |
| 6 | Validation report | L2 or L3 results, calibration, decision curve, subgroups | 15% |
| 7 | Error analysis | 20–30 reviewed errors, taxonomy, examples | 7% |
| 8 | Model card and datasheet | Intended use, limits, training data, subgroup metrics | 8% |
| 9 | Regulatory memo | Regime, classification, binding artifact, open questions | 7% |
| 10 | Risk register | Clinical, technical, privacy, equity, commercial | 5% |
| 11 | Deployment and monitoring | Architecture, latency, drift, alert volume, retraining | 8% |
| 12 | Value case | Comparator cost, budget impact, adoption assumptions | 5% |
| 13 | Readout and pitch | One-pager, slides, recording | 2% |

Repository layout: `README.md`, `charter.md`, `data/` (provenance, dictionary, immutable `splits.json`), `src/` (pipeline, features, train, evaluate, serve), `configs/`, `results/` (metrics JSON, subgroup CSV, calibration and decision-curve figures), `governance/` (analysis plan, regulatory memo, risk register, model card, datasheet), and `docs/` (readout, pitch).

Reproducibility checklist: data version and license recorded; environment pinned and base image named; seeds fixed and listed; preprocessing fit inside the training fold; one command reproduces the primary result; split file committed and immutable; metrics computed by script, not typed into slides; hardware and runtime recorded; artifacts retained for the program period.

**Definition of done.** A competent stranger can clone the repository, read the charter, run one command, reproduce the primary metric within a stated tolerance, and explain the regulatory path without asking the team a question. Anything less is a draft.

## 11A. Term assessments: midterm dossier, track playbook, and case critique

The program weights three assessments outside the capstone. Each needs a brief, a template, and a rubric.

### 11A.1 Midterm evidence dossier (20%, end of Week 12)

**Team:** 3–5. **Length:** 2,500 words plus appendix. **Assessor:** instructor plus one clinical/industry reviewer.

**Task:** Take a real or realistic decision from Module 00 and produce a decision-ready evidence dossier. It is not a model report; it is the argument a clinical or operational leader would use to fund, reject, or pilot the work.

Required sections:

1. Decision, user, workflow, and comparator.
2. Intended use and population.
3. Data provenance, access, quality, and missingness.
4. Cohort, label, and time-zero definition.
5. Study design and estimand.
6. Analysis and validation plan.
7. Calibration, decision-curve, and subgroup plan.
8. Regulatory and ethical triage.
9. Limitations and stop criteria.
10. Reproducibility appendix: code, environment, data dictionary, assumptions ledger.

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision and workflow | 20% | Model-first | Clear decision and user | Comparator, capacity, and value |
| Data and label | 20% | Convenience sample | Explicit cohort and label | Provenance, missingness, and bias quantified |
| Design and estimand | 25% | No protocol | Clear estimand and validation | External/impact plan and sensitivity |
| Analysis and validation | 20% | AUROC only | Calibration and utility | Subgroup and decision analysis |
| Communication and reproducibility | 15% | Technical | Clear memo | Executable appendix and audit trail |

### 11A.2 Track playbook (15%, end of Week 26)

**Team:** track cohort. **Length:** 12–15 pages plus a 10-minute brief. **Assessor:** track lead plus an external domain reviewer.

**Task:** Produce a practical playbook for one industry track. It must be usable by a new data scientist joining a company in that sector.

Required sections:

1. Sector map: users, buyers, workflows, payment, regulation.
2. Data landscape: sources, standards, access, quality.
3. Three use cases, each with decision, data, model/analysis, metric, and failure modes.
4. Evidence and regulatory pathway for one use case.
5. Economics: value, cost, reimbursement, adoption.
6. Implementation roadmap: 12 months, owners, milestones.
7. Risks: technical, clinical, regulatory, commercial, equity.
8. What not to build, and why.
9. Annotated source list.

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Sector depth | 20% | Generic | Accurate workflows and incentives | Insight beyond public summaries |
| Use-case selection | 20% | Technology-first | Decision-first | Portfolio logic and sequencing |
| Evidence and regulation | 20% | Names rules | Maps evidence to pathway | Anticipates reviewer objections |
| Economics and adoption | 20% | No numbers | ROI and adoption plan | Sensitivity, budget impact, procurement |
| Roadmap and communication | 20% | Vague | Milestones and owners | Credible 12-month plan with stop criteria |

### 11A.3 Case critique (15%, six per term)

**Individual.** **Length:** one page, 500–650 words. **Assessor:** instructor or peer reviewer with a rubric.

Choose a paper, product, recall, or regulatory decision. Answer:

1. What decision does it claim to improve?
2. What is the evidence, and what is missing?
3. What assumptions or biases could change the conclusion?
4. What would you do differently?
5. What would make you stop?

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision framing | 20% | Summary only | Clear decision and user | Comparator and incentives |
| Evidence appraisal | 30% | Accepts claims | Identifies design and validation issues | Quantifies uncertainty and bias |
| Bias and limitations | 20% | Generic caveats | Specific mechanisms | Links to harm and equity |
| Alternatives and judgment | 15% | No alternative | Plausible alternative | Decision-ready recommendation |
| Communication | 15% | Jargon | Clear one-pager | Persuasive to a mixed audience |

### 11A.4 AI-assistance rule for assessed work

- Disclose material use of LLMs or coding assistants in an appendix.
- You remain accountable for every claim, line of code, and citation.
- Do not paste identifiable data into external tools.
- Do not use an LLM to fabricate sources, data, or results.
- Oral defenses may include questions on any part of the submission.

## 12. Judging rubric

The rubric rewards judgment over leaderboard position. A modest metric with rigorous evidence outranks a stronger metric with no governance.

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision and scope | 15% | Topic, no decision or owner | One decision and owner; comparator described | Decision, workflow, error economics, and exclusions mutually consistent |
| Data provenance and quality | 10% | Source unnamed, no license | Source, version, missingness documented | Full lineage, de-identification method, label-provenance analysis |
| Validation design | 15% | Random split only | Temporal or patient split with justification | Site-held-out or prospective; leakage screen; prevalence-shift sensitivity |
| Modeling rigor | 15% | No baseline; unreproducible | Honest baseline; tuned candidate; intervals | Ablation and uncertainty; complexity justified |
| Calibration, error, equity | 10% | Overall metric only | Calibration and one subgroup | Decision curve, error taxonomy, intersectional harms and mitigation |
| Regulatory and ethical governance | 10% | Not addressed | Regime named; intended use stated | Binding artifact identified; claim matches evidence; residual risks owned |
| System and monitoring | 10% | Notebook only | Serving diagram and drift plan | Tested interface, latency budget, alert-volume estimate, rollback triggers |
| Value and communication | 10% | Feature list | Baseline cost and adoption path | Budget impact with assumptions, procurement obstacles, first customer |
| Reproducibility and dossier | 5% | Missing artifacts | Complete with minor gaps | Stranger-reproducible, versioned, audit-ready |

**Individual defense.** Each member answers one question on decision and evidence, one on the pipeline, and one on governance or failure. The individual score multiplies the team score; 80 with a 0.6 multiplier yields 48. This is deliberately harsh and protects the credential.

| Finding | Deduction |
|---|---:|
| Fabricated or altered result | Zero; integrity process |
| Identifiable PHI in any artifact or demo | 20 points |
| No baseline, or a baseline tuned less carefully than the candidate | 10 points |
| Test set used for selection without disclosure | 10 points |
| Claim exceeds evidence ("diagnoses" from a triage study) | 10 points |
| Missing data-access documentation | 10 points |
| No subgroup analysis for a clinical claim | 5 points |

Before judging, the panel scores the same recorded capstone from a prior year and discusses disagreements. Scores are normalized per judge; a criterion more than two standard deviations from the panel mean is discussed and, if unresolved, averaged rather than dropped.

## 13. Demo day and executive pitch

| Minutes | Content | Question answered |
|---:|---|---|
| 0–1 | The decision and its owner | Why should I care? |
| 1–2 | Current practice and its cost | What is broken today? |
| 2–4 | What you built and the evidence | Does it work? |
| 4–6 | Live demo on a held-out case | Can I see it? |
| 6–7 | Failure mode and mitigation | What happens when it is wrong? |
| 7–8 | Regulatory path in one paragraph | Can this legally reach a patient? |
| 8–9 | Value and adoption | Who pays, and why now? |
| 9–10 | The ask | What do you want next? |

Slides follow the same order: decision title, baseline versus candidate with intervals, one calibration or decision-curve visual, one subgroup table with the harm highlighted, demo frame, failure and mitigation table, pathway diagram with the binding artifact, value case with three assumptions, risks and next 90 days, ask.

Demo discipline: use a held-out case, never a training example; record a backup because live demos fail on conference networks; show one failure case and the safe behavior; keep the demo under 90 seconds; never display real patient data.

| Question | Weak answer | Strong answer |
|---|---|---|
| "What is the accuracy?" | A single number | "At the deployment threshold, sensitivity is X (95% CI Y); the false-positive rate implies Z extra reviews per day." |
| "Why not a simpler model?" | "Deep learning is better" | "The baseline reaches A, the candidate B; the difference is or is not outside the interval, and complexity costs this much to maintain." |
| "Who pays?" | "The hospital" | "The budget holder is the ED medical director; we avoid N boarder-hours monthly at loaded cost M, with payback in Q months under these assumptions." |
| "Is this regulated?" | "I think it is a device" | "Under our intended use it is triage support on a SaMD pathway; the binding artifact is a 510(k) with clinical evidence, and we would request a pre-submission meeting first." |
| "What about bias?" | "We found no bias" | "We pre-specified three subgroups; the gap is G points with this interval; here is the mitigation and monitoring trigger." |
| "What would make you stop?" | "Nothing yet" | "If site-held-out sensitivity falls below X, or alert volume exceeds Y per shift, we would not deploy." |

The 20-minute technical defense covers the analysis plan, split logic, preprocessing fit scope, metric choice, calibration, subgroups, error taxonomy, reproducibility, and the three most likely failure modes. Expect at least one question beginning, "Show me the code that proves that." Keep the repository open.

## 14. Data access and governance

| Tier | Examples | Requirements | Permitted use |
|---|---|---|---|
| Open | Public challenge sets, Synthea, open EHR demos | License review; attribution | Analysis and public demo |
| Credentialed | MIMIC-IV, eICU, restricted PhysioNet sets | CITI training, DUA, institutional account, no redistribution | Approved research and coursework on secure infrastructure |
| Restricted | Institutional EHR, registry, trial data | IRB approval or determination, DUA, minimum necessary, access controls | Named protocol only |
| Synthetic | Generated cohorts, sandbox FHIR | Generator validation; disclosure | Development and demos; never clinical claims |

PhysioNet-credentialed data cannot be redistributed to students. Use demo sets, open challenge splits, or institution-specific access. Students should complete CITI training and apply for credentials before the capstone, not in week 8.

Governance checklist: DUA signed and filed before access; IRB determination documented; minimum-necessary pull with no unused identifiers; de-identification method recorded and re-identification risk assessed; approved storage and compute; access list and audit log; no PHI in prompts sent to external model providers; no identifiable data in slides, demos, or the public repository; retention and destruction schedule; named incident-response contact.

Record the license for every weight, embedding model, and dataset. "Open weights" does not mean unrestricted commercial use; check whether the license permits clinical use, requires attribution, or imposes use restrictions. A capstone depending on a model whose license forbids clinical deployment must say so in the regulatory memo.

Equity is also a data-governance issue: who is in the data, who consented, who the access model excludes, and who bears the cost of a false positive. State subgroup representation in training and test, and the likely direction of bias when representation is inadequate. Do not report a subgroup metric from fewer than 30 events without a clear warning.

## 15. Mentorship

| Mentor | Contribution | Cadence | Boundary |
|---|---|---|---|
| Clinical | Workflow, label meaning, error review, adoption | Weekly 30 minutes | Does not write code or set scope |
| Technical | Validation, modeling, engineering | Weekly 60 minutes | Does not own the deliverable |
| Regulatory / quality | Intended use, pathway, risk analysis | Weeks 2, 9, 11 | Does not give legal advice |
| Commercial | Value case, buyer, procurement | Weeks 10, 11 | Does not set the clinical claim |
| Peer | Code review, rehearsal, critique | Weekly | No grade authority |

Make mentorship work: send a one-page brief 24 hours ahead; ask for a decision, not validation ("Should we drop the external site because access slipped?" is answerable; "How are we doing?" is not); record decisions with mentor name and date; never ask a mentor to co-author the dossier.

```
Team: [name]  Week: [n]  Decision: [one line]
Since last session: [three bullets with artifact links]
Evidence status: [validated / not validated]
Blocker: [one thing]   Question: [decision with two or three options]
Needed by: [date]
```

## 16. Common failure modes

| Failure | Early warning | Consequence | Intervention |
|---|---|---|---|
| Solution seeking a decision | Models discussed before users | No adoption story | Return to G0; interview an owner |
| Data access filed late | No DUA in week 2 | Three-week delay | Activate fallback; re-scope at G1 |
| Metric proxy drift | AUROC for a capacity-constrained decision | Unusable operating point | Re-derive the metric from the decision |
| Leakage | Suspiciously high validation | Invalid evidence | Run the leakage screen; rebuild splits |
| Leaderboard overfitting | Daily probing; score chasing | Hidden-split collapse | One probe per week; trust local validation |
| Uncritical LLM labels | No provenance or review | Label noise; circular evaluation | Sample, adjudicate, report agreement |
| No calibration | Only discrimination reported | Thresholds cannot be set | Add reliability plot and decision curve |
| Missing subgroups | "Not enough data" with no counts | Undisclosed harm | Report counts and intervals; state uncertainty |
| Governance deferred | Memo started in week 11 | Claim exceeds evidence | Draft intended use in week 2 |
| Demo dependency | Live network and real data | Public failure | Backup recording; synthetic inputs |
| Free riding | Two members commit most code | Individual defense fails | Contribution log; reassign at midpoint |
| Scope creep | New questions after week 6 | Core unfinished | Enforce tiers; park ideas in a backlog |

## 17. Templates

### 17.1 Project charter

```markdown
# Capstone charter
## Decision
Decision: [action that changes]   Owner: [role]   Frequency: [per shift/week/case]
Workflow point: [where the output arrives]
## Population and intended use
Eligible: [inclusion]   Excluded: [exclusion]
Setting: [unit, site, era]   Inputs at decision time: [list]
Output: [score/rank/draft/alert]   User: [role]
## Comparator
Current practice: [description]
Measured performance or cost on the same cohort: [number and source]
## Outcomes and harms
Primary metric: [metric, threshold, justification]
False positive cost: [clinical, operational, financial]
False negative cost: [clinical, operational, financial]
Equity: [pre-specified subgroups and why]
## Scope
Core: [three to five items]   Stretch: [after freeze]   Out: [exclusions]
## Evidence and governance
Validation level: [L2/L3/L4]   Regulatory triage: [regime and pathway]
Data tier: [open/credentialed/restricted/synthetic]   Stop condition: [result]
Signed: [team], [clinical mentor], [program lead]   Date: [YYYY-MM-DD]
```

### 17.2 Weekly status

```markdown
# Week [n] — [team]
Decision this week: [one line]
Done: [artifact] [link] · [artifact] [link]
Evidence: primary metric [value, CI] on [split]; calibration [state]
Risks: [R-ids and movement]   Blocked: [one item, owner, date]
Asks: [decision or access from a named person]
Next week: [three bullets]
```

### 17.3 Risk register

| ID | Risk | Category | L | I | Score | Owner | Mitigation | Trigger | Status |
|---|---|---|---:|---:|---:|---|---|---|---|
| R-01 | Credentialed access delayed past week 3 | Data | 4 | 5 | 20 | Data lead | Demo extract approved at G0 | No account by week 2 | Open |
| R-02 | Site-held-out performance below threshold | Evidence | 3 | 5 | 15 | Modeler | Pre-specify L2 claim; domain adaptation | L3 AUC drop > 0.05 | Open |
| R-03 | Subgroup too small to estimate | Equity | 3 | 4 | 12 | Evidence lead | Pool sites; report intervals; narrow claim | < 30 events per group | Open |
| R-04 | Demo depends on live EHR | Delivery | 2 | 4 | 8 | Lead | Recorded backup; synthetic case | Failed rehearsal | Mitigated |
| R-05 | Claim exceeds evidence | Governance | 2 | 5 | 10 | Evidence lead | Intended-use review at G4 | Claim says "diagnose" | Open |

Score = likelihood × impact. Review weekly; escalate 15 or above to the program lead.

### 17.4 Final readout, one page

```markdown
# [Project] — decision readout
## Recommendation
[Deploy/pilot/iterate/stop], for [population] in [setting], because [one-sentence evidence].
## The decision
[Decision, owner, frequency, current practice.]
## What we built
[Method in two sentences. Data source and version. Validation level.]
## Evidence
Primary metric: [value] (95% CI [low, high]) on [split].
Comparator: [value] (95% CI [low, high]).   Calibration: [summary].
Subgroups: [largest gap and direction].   Decision curve: [net benefit at threshold].
## What could go wrong
[Top failure], mitigated by [control], monitored by [signal].
[Equity risk], mitigated by [action].
## Regulatory path
[Regime], [classification], [binding artifact], [next step].
## Value
Budget impact: [number] under [three assumptions]. Payback: [months]. First customer: [role].
## The ask
[Specific decision or resource, by date.]
## Artifacts
Repository: [link] · Dossier: [link] · Demo: [link]
```

## 18. Sources and further reading

All URLs accessed 24 October 2026. Competition status is as reported at that date and must be re-verified.

### Competitions and benchmarks
- PhysioNet/CinC Challenge 2025, Chagas disease: https://moody-challenge.physionet.org/2025/
- PhysioNet/CinC Challenge 2026, cognitive impairment from polysomnography: https://moody-challenge.physionet.org/2026/
- RSNA AI Challenges: https://www.rsna.org/
- MICCAI challenges: https://www.miccai.org/
- NeurIPS 2025 competitions: https://blog.neurips.cc/2025/06/27/neurips-2025-competitions-announced/
- DREAM Challenges: https://dreamchallenges.org/
- Open Problems in Single-Cell Analysis: https://openproblems.bio/
- CAGI7: https://genomeinterpretation.org/cagi7-challenge.html
- MIT Hacking Medicine: https://hackingmedicine.mit.edu/
- MIT Critical Data: https://criticaldata.mit.edu/
- MedGemma and Google Health: https://health.google/
- HL7 FHIR Connectathons: https://www.hl7.org/
- XPRIZE Healthspan: https://www.xprize.org/
- CDISC: https://www.cdisc.org/

### Regulation and standards
- FDA, Software as a Medical Device: https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- FDA, Clinical Decision Support Software guidance: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- FDA, AI/ML-enabled medical devices and PCCP: https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- FDA, Good Machine Learning Practice: https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- EU AI Act, Regulation (EU) 2024/1689: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EU MDR, Regulation (EU) 2017/745: https://eur-lex.europa.eu/eli/reg/2017/745/oj
- IMDRF, ML-enabled medical devices: https://www.imdrf.org/
- ISO 14971, risk management: https://www.iso.org/standard/72704.html
- IEC 62304, medical device software: https://www.iec.ch/
- ICH E9(R1), estimands: https://www.ich.org/
- 21 CFR Part 11: https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
- HIPAA Privacy Rule: https://www.hhs.gov/hipaa/
- GDPR: https://eur-lex.europa.eu/eli/reg/2016/679/oj
- CLIA, CMS: https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments

### Reporting and evaluation
- TRIPOD+AI: https://www.tripod-statement.org/
- CONSORT-AI and SPIRIT-AI: https://www.consort-spirit.org/
- STARD-AI, EQUATOR: https://www.equator-network.org/
- PROBAST: https://www.probast.org/
- DECIDE-AI: https://doi.org/10.1136/bmj-2022-070904

### Data and governance
- MIMIC-IV and demo: https://physionet.org/content/mimiciv/
- eICU-CRD: https://physionet.org/content/eicu-crd/
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- BraTS via Synapse: https://www.synapse.org/
- TCGA, Genomic Data Commons: https://www.cancer.gov/tcga
- Synthea: https://synthetichealth.github.io/synthea/
- CMS public use files: https://data.cms.gov/
- CDC NWSS: https://www.cdc.gov/nwss/
- CITI Program: https://about.citiprogram.org/
- US Core and DEQM: https://hl7.org/fhir/us/core/

### Method and practice
- Steyerberg, *Clinical Prediction Models*, 2nd ed., Springer, 2019.
- Vickers and Elkin, "Decision Curve Analysis," *Medical Decision Making*, 2006: https://doi.org/10.1177/0272989X06295361
- Collins et al., "TRIPOD+AI Statement," *BMJ*, 2024: https://doi.org/10.1136/bmj-2023-078378
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015: https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
- Obermeyer et al., "Dissecting Racial Bias in an Algorithm," *Science*, 2019: https://doi.org/10.1126/science.aax2342
- Wong et al., "External Validation of a Proprietary Sepsis Model," *JAMA Internal Medicine*, 2021: https://doi.org/10.1001/jamainternmed.2021.2626

> **Next:** [Module 20](20_career_portfolio_interviews.md) turns the capstone into a portfolio, a narrative, and interview answers that survive technical and behavioral scrutiny.
