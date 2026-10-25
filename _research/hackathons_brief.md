# Research Brief — Biomedical Hackathons, Competitions, and Benchmarks (2023–2026)

**Source cutoff:** 24 October 2026. Verify all dates and eligibility before use. Some 2026 events are concluded; some are announced only. Kaggle pages are rate-limited, so some details come from organizer announcements and winner write-ups.

## 1. Recurring programs

| Program | Cadence | Typical problem | Data / metric | Notes |
|---|---|---|---|---|
| PhysioNet / Computing in Cardiology Challenge | Annual since 2000 | ECG, PSG, ICU, sepsis | Open or credentialed; metric varies | Open-source code required; strong benchmark literature |
| MIT Hacking Medicine Grand Hack | Annual, Boston | Clinical unmet needs | Venture-style judging | Tracks, mentors, accelerator |
| MIT Critical Data datathons | Global | ICU/EHR analytics | MIMIC/eICU | 39+ countries; local editions |
| RSNA AI Challenges | Annual, Kaggle | Imaging | Large imaging datasets; AUC/log loss | 2024 spine, 2025 aneurysm, 2026 knee |
| MICCAI challenges | Annual | Segmentation, detection, registration | Synapse/CodaLab | BraTS, FLARE, HECKTOR, etc. |
| NeurIPS competition track | Annual | ML methods, health | Varies | CURE-Bench, SLC-PFM, EEG Foundation |
| DREAM Challenges | Periodic | Omics, drug discovery, causal | Sage Bionetworks | SEA-AD, COVID causal diagram |
| Open Problems in Single-Cell Analysis | Continuous | Single-cell methods | Open benchmarks | Re-run every two months |
| CAGI | Periodic | Genome interpretation | Variant effect, PRS | CAGI7 in 2025 |
| HL7 FHIR Connectathon | 3–4/year | Interoperability | FHIR servers | Free; real payer/provider test partners |
| XPRIZE Healthspan | Multi-year | Healthy aging | Clinical trials | $101M; finalists 2026 |
| BARDA / ARPA-H challenges | Periodic | Biosecurity, antivirals, health security | Varies | Funding/prize hybrids |
| Wellcome Leap programs | Periodic | Antibiotics, women's health | Milestone-funded | Not open leaderboards |
| MedGemma Impact Challenge | 2026 | Health AI prototypes on MedGemma | Kaggle | Low-resource/on-device focus |
| CDISC AI Innovation Challenge | Annual | Clinical data standards | Synthetic SDTM/ADaM | Pharma standards |
| Mayo Clinic Datathon | Annual | Clinical analytics | Institutional data | CME |
| Stanford Health Hacks | Annual | Clinical AI, wearables, robotics | Hackathon | 2026 |
| JHU Flourish-A-Thon | Annual | AI for good | Hackathon | 2026 |

### 1.1 PhysioNet 2025 — Chagas disease

- Task: triage 12-lead ECGs for Chagas disease to prioritize scarce serological testing.
- Data: CODE-15% (~300k Brazilian ECGs, weak labels), SaMi-Trop (1,631 confirmed positives), PTB-XL (assumed negatives), private endemic sets.
- Metric: fraction of true cases ranked in the top 5% — a capacity-constrained metric.
- Winning approaches: ViT ECG foundation-model embeddings; Transformer/xLSTM ensembles; label-uncertainty handling.
- Lesson: the metric encodes the clinical capacity constraint. Optimize for the decision, not generic AUROC.
- URL: https://moody-challenge.physionet.org/2025/

### 1.2 PhysioNet 2026 — cognitive impairment from polysomnography

- Task: predict MCI/Alzheimer's/dementia diagnosed 1–6 years after a sleep study.
- Data: Human Sleep Project, 5 US institutions, EDF, 214 GiB small / 1.2 TiB large.
- Metric: age-conditioned AUROC (within ±2 years) to remove the age shortcut.
- Lesson: metric design can remove a trivial confounder. Design validation to match hidden sites.
- URL: https://moody-challenge.physionet.org/2026/

### 1.3 RSNA 2024/2025/2026

- 2024 lumbar spine degenerative classification: 1,874 teams; weighted multi-class log loss; two-stage localization → ROI classification; label-noise reduction.
- 2025 intracranial aneurysm detection: 1,147 teams; macro AUC; vessel-first nnU-Net pipelines; ROI classification; location-aware transformer.
- 2026 knee abnormality detection: 12 findings from knee MRI; macro ROC AUC; ~6,425 teams; closed 22 Oct 2026; LLM-derived labels and leakage disputes.
- URL: https://www.rsna.org/ and Kaggle competition pages.

### 1.4 NeurIPS 2025 health competitions

- CURE-Bench: LLM reasoning for precision therapeutics; agentic and internal-knowledge tracks; clinician review.
- SLC-PFM: self-supervised cancer pathology foundation models; 300M+ images, 39 cancer types, 23 tasks.
- EEG Foundation Challenge: cross-task/cross-subject decoding, including psychopathology.
- PNPL: LibriBrain MEG speech decoding.
- URL: https://blog.neurips.cc/2025/06/27/neurips-2025-competitions-announced/

### 1.5 DREAM, Open Problems, CAGI

- DREAM 2025: SEA-AD Alzheimer's pathology from snRNA-seq; COVID causal diagram; olfactory mixture prediction; Target 35.
- Open Problems: 14 single-cell problems, ~140 methods, ~127 datasets; logistic regression often beats complex methods for label projection; expression-magnitude cell-cell communication can beat specificity-based methods.
- CAGI7: 14 challenges including rare-genome VCF/CRAM, PRS, deep mutational scanning, MPRA, splicing.
- URLs: https://dreamchallenges.org/, https://openproblems.bio/, https://genomeinterpretation.org/cagi7-challenge.html

### 1.6 Industry-sponsored

| Sponsor | Program | Problem | Status |
|---|---|---|---|
| Google Health | MedGemma Impact Challenge | Prototypes on MedGemma/MedSigLIP/HeAR/MedASR; low-resource/on-device | Winners Mar 2026 |
| J&J Innovation | Data & AI QuickFire | Foundation models, multi-omics, virtual disease models, agentic R&D | Closed Feb 2026 |
| J&J | CAT-PD QuickFire | De novo antibody design | Finals Jan 2026 |
| J&J MedTech + NVIDIA + AWS | Polyphonic AI Fund for Surgery | Surgical computer vision | Round 2 open Oct 2025 |
| Roche | Ophtha Innovation Challenge 2026 | Diabetic macular edema access; RWD ecosystems | Submissions Oct–Dec 2026 |
| CDISC + pharma | AI Innovation Challenge | Synthetic SDTM/ADaM; agentic TFL; traceability | Annual |
| Microsoft + UNC Health | Healthcare AI Hackathon | OMOP semantic models, KPI agents | 9 Nov 2026 |
| AWS | Health AI Hackathon | Medical NLP, patient journey | Archived/regional |
| Medtronic | Internal/BMES/InnoMatch | aDBS, RAG over manuals, digital biomarkers | 2025–26 |

## 2. Task-design and winning-approach trends

1. **Metric first.** Top teams read the metric and hidden split before modeling.
2. **Localization → ROI classification** is canonical in 3D imaging.
3. **Pretrained backbones** (nnU-Net, ConvNeXt, ViT, ECG foundation models) dominate.
4. **Label noise handling** (out-of-fold residual filtering, uncertainty labels) produces large gains.
5. **Calibration and alert burden** matter more than headline AUC in clinical settings.
6. **Simple baselines remain competitive** in tabular and single-cell tasks.
7. **LLM-generated labels** are appearing, with provenance and leakage concerns.
8. **Human-in-the-loop grading** is becoming standard for LLM tasks.
9. **Federated and privacy-preserving** challenges (FeTS/MedPerf) are maturing.
10. **Equity-weighted evaluation** is increasingly explicit.

## 3. Course project briefs

Format: problem — data — metric — regulatory/ethical angle — difficulty (1–5).

### Imaging
1. Aneurysm triage on head CTA — RSNA 2025/open CT — macro AUC + sensitivity at 2 FP/scan — high-risk device; reader study — 5.
2. Knee MRI multi-finding classifier — RSNA 2026 — macro ROC AUC — label provenance — 5.
3. Lumbar spine severity grading — RSNA 2024 — weighted log loss — class imbalance — 4.
4. CXR pneumothorax triage under shift — CheXpert/MIMIC-CXR — AUROC + subgroup gap — SaMD; bias documentation — 3.
5. TB screening on-device — CODA-TB/HeAR — sensitivity at fixed specificity; model size — low-resource; consent — 4.
6. Mammography BI-RADS from reports — institutional/MIMIC-style — macro-F1 + calibration — PHI; de-identification — 2.
7. Federated glioma segmentation — BraTS/FeTS — Dice + HD95 + communication cost — privacy governance — 5.
8. Retinal DR grading across cameras — APTOS/DeepDR — quadratic weighted kappa — color/ethnicity bias — 3.

### ECG / physiology
9. Chagas triage at 5% capacity — PhysioNet 2025 — precision@top-5% — screening ethics — 4.
10. Age-conditioned dementia from PSG — Human Sleep Project — age-conditioned AUROC — incidental findings — 5.
11. AF detection from wearable — PhysioNet AF/PPG — F1 + false alarms/hour — consumer claims — 3.
12. Sepsis early warning — MIMIC-IV/eICU — AUROC + utility + lead time — alarm burden — 4.
13. EEG seizure classification — TUH EEG — weighted kappa — montage/device shift — 4.
14. Sleep staging from wearable — Sleep-EDF — Cohen's kappa — consumer vs clinical — 3.

### EHR / operations
15. ICU deterioration + fairness — MIMIC-IV — AUROC + equalized-odds gap — redlining risk — 3.
16. ED boarding prediction — MIMIC-IV-ED — MAE + calibration — operational claim — 2.
17. Readmission with social determinants — MIMIC-IV + ADI — AUROC + net benefit — proxy discrimination — 3.
18. Post-HCT survival with censoring — CIBMTR — stratified concordance — competing risks — 4.
19. OR scheduling — public OR logs — utilization + overtime — staffing equity — 3.
20. Claims cost anomaly — CMS SynPUF — precision@k — payer audit fairness — 2.

### NLP / LLM agents
21. FHIR medication reconciliation agent — MedAgentBench — pass@1 — hallucinated writes — 4.
22. Discharge summary with factuality — MIMIC-IV-Note — factuality + omission — hallucination liability — 3.
23. Prior-auth letter drafting — synthetic policies + notes — rubric + human preference — no auto-denial — 2.
24. Trial eligibility matching — TrialGPT/ClinicalTrials.gov — nDCG/precision — false exclusion — 3.
25. HealthBench-style evaluator — public subset — correlation with physician grades — grader bias — 3.
26. Multilingual triage — Ilera-Bench-style — accuracy + safety recall — language equity — 3.
27. Radiology report to structured findings — MIMIC-CXR — F1 — PHI in prompts — 2.

### Genomics / pathology / public health
28. Missense variant effect — CAGI7 DMS — Spearman/ROC — ACMG framing — 4.
29. PRS portability — CAGI7 — AUC across ancestries — ancestry bias — 4.
30. Pathology foundation-model probe — SLC-PFM/TCGA — balanced accuracy — compute/data terms — 5.
31. Single-cell label projection — Open Problems — macro-F1 vs logistic baseline — reproducibility — 3.
32. Suicide-narrative abstraction — NVDRS/synthetic — F1 — mandated-reporter ethics — 3.
33. Wastewater/syndromic nowcasting — CDC NWSS + ED — MAE/WIS — public communication — 3.
34. FHIR quality-measure gap — Synthea + US Core/DEQM — compliance rate — CMS reporting — 2.

## 4. Winning-team playbook

**Scope**
- Read the metric and split first.
- Infer the hidden test distribution (new sites, scanners, years).
- Build a local validation split that mimics it.
- Do data archaeology: metadata, missingness, label provenance, duplicates, leakage.
- Keep a boring baseline alive.

**Validate**
- Small trusted CV that mirrors the hidden split.
- One public-leaderboard probe per week, not per day.
- Ignore gains under ~0.003 unless robust.
- Report per-subgroup metrics.
- For LLM tasks, use physician-graded subsets and inter-rater agreement.
- For imaging, report per-finding and per-site results.

**Present**
- Name the clinical workflow.
- Quantify the baseline and the delta.
- Show a working demo on real inputs.
- State the failure mode.
- State the regulatory path in one sentence.
- Disclose data and label provenance.
- Report compute and cost.

**Common failures**
1. Optimizing a proxy metric that diverges from the clinical objective.
2. Leakage: patient, study, pretrained-on-test, public data.
3. Ignoring calibration and alarm burden.
4. No subgroup reporting.
5. Overfitting the public leaderboard.
6. Uncritical LLM-generated labels.
7. Pitch decks with no baseline or failure analysis.
8. Missing governance: DUA, CITI, de-identification.

## 5. Access notes for teaching

- PhysioNet-credentialed data cannot be redistributed to students.
- Use MIMIC-IV demo, open PhysioNet sets, or public challenge training splits.
- Have students complete CITI and apply for credentials for capstone access.
- Kaggle competition rules may restrict external data and pretrained models.
- Check licenses for pretrained weights and foundation models.
- Do not use identifiable data in public demos.

## 6. Uncertain / watch list

- MIT Grand Hack 2026 dates.
- PhysioNet 2027 task.
- CAGI8.
- FeTS 2025/26.
- Kaggle early sepsis 2026 details.
- BraTS 2025 per-task winners.
- RSNA 2026 prize allocation.

All items above must be re-verified before being presented as fact.
