# Module 13 — Biopharma and Clinical Development

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (10 lecture, 4 case clinic, 14 lab)
> **Track fit:** Biopharma & Translational core
> **Prerequisites:** Modules 04, 05, 12
> **Tools:** Python, R, CDISC SDTM/ADaM concepts, ClinicalTrials.gov API, AEMS/openFDA, Open Targets
> **Industry question:** "Which target, patient, site, or safety signal should we bet on — and what evidence will make that bet defensible?"

---

## 1. Why industry cares

Drug development is long, expensive, and failure-prone. Data science can improve decisions at every stage, but it operates inside a heavily regulated evidence system.

Industry uses data science for:

- Target identification and validation.
- Biomarker discovery and patient selection.
- Trial design and feasibility.
- Site and investigator selection.
- Recruitment and retention.
- Safety signal detection.
- Real-world evidence.
- Regulatory submissions.
- Commercial strategy.

The common failure is treating a pharma problem like a Kaggle problem. The data are small, the endpoints are regulated, the timelines are long, and the cost of a wrong claim is enormous.

## 2. Learning outcomes

By the end of this module, you can:

- Map the drug-development value chain and the data available at each stage.
- Design a biomarker strategy for a clinical program.
- Evaluate trial feasibility and recruitment using real-world data.
- Explain CDISC SDTM/ADaM and GxP data requirements at a working level.
- Detect and interpret safety signals responsibly.
- Design a real-world evidence study for regulatory or payer use.
- Assess AI's role in discovery and development without overclaiming.
- Build a translational data plan from target to launch.
- Communicate with clinical, regulatory, and commercial stakeholders.

## 3. Mental model: the development funnel

```
Target discovery → Validation → Lead optimization → Preclinical → Phase I → Phase II → Phase III
     → Regulatory review → Launch → Phase IV / lifecycle
```

Each stage has a different decision, data type, and evidence standard.

| Stage | Core question | Typical data | Main failure |
|---|---|---|---|
| Discovery | Is this target causal? | Omics, genetics, literature | Wrong target |
| Preclinical | Is it safe and druggable? | Assays, animal models | Toxicity, efficacy gap |
| Phase I | Is it tolerable? | Human PK/PD, safety | Safety |
| Phase II | Does it work? | Biomarkers, endpoints | No efficacy |
| Phase III | Does it work at scale? | Randomized outcomes | Effect size, population |
| Regulatory | Is the evidence sufficient? | Submission package | Missing evidence |
| Launch | Who should get it? | RWE, claims, EHR | Access, adoption |
| Lifecycle | What else can it do? | RWE, trials | Label expansion failure |

## 4. Data in pharma

| Data type | Source | Use | Limitation |
|---|---|---|---|
| Omics | Discovery, trials | Target, biomarker | Batch, sample size |
| Assay | Preclinical | Potency, ADMET | Translation gap |
| Clinical trial | Sponsors/CROs | Efficacy, safety | Selected population |
| EHR | Health systems | Feasibility, RWE | Fragmented |
| Claims | Payers | Utilization, cost | No severity |
| Registry | Disease/device | Natural history | Selection |
| Safety | AEMS (formerly FAERS), EudraVigilance | Signal detection | Underreporting |
| Literature | PubMed, conferences | Evidence | Publication bias |
| Commercial | Sales, prescriptions | Market | Lag, aggregation |

## 5. Biomarker strategy

### 5.1 Types

- **Prognostic:** predicts outcome independent of treatment.
- **Predictive:** identifies likely response to treatment.
- **Pharmacodynamic:** shows biological effect.
- **Safety:** predicts adverse events.
- **Monitoring:** tracks disease or response.

### 5.2 Development

1. Discovery in relevant tissue.
2. Analytical validation of assay.
3. Clinical validation in a defined population.
4. Clinical utility in a trial.
5. Regulatory qualification or companion diagnostic.

### 5.3 Companion diagnostics

A companion diagnostic is required for some therapies. It must be analytically and clinically validated, and the drug and test are often developed together.

**Industry lesson:** a biomarker that is not validated with the assay and population used in the trial is not a companion diagnostic.

## 6. Trial feasibility and recruitment

### 6.1 Feasibility questions

- How many eligible patients exist?
- Where are they treated?
- How many are already in trials?
- What is the standard of care?
- How long does recruitment take?
- What is the screen failure rate?

### 6.2 Data sources

- EHR and claims.
- Registries.
- Tumor boards.
- Site questionnaires.
- Historical trial performance.
- Public trial registries.

### 6.3 Pitfalls

- Coding-based eligibility overestimates.
- Trial participation varies by site.
- Rare subgroups are invisible in claims.
- Site activation delays dominate.
- Recruitment models ignore patient willingness.

## 7. Clinical trial data standards

### 7.1 CDISC

- **SDTM:** Study Data Tabulation Model; submission-ready tabulations.
- **ADaM:** Analysis Data Model; analysis-ready datasets.
- **Define-XML:** metadata.
- **Controlled terminology.**

### 7.2 GxP and Part 11

- Computerized systems used in clinical trials must be validated.
- Electronic records and signatures require controls.
- Audit trails, access controls, and data integrity are essential.

### 7.3 Trial documents

- Protocol.
- Investigator's brochure.
- Statistical analysis plan.
- Case report forms.
- Data management plan.
- Safety reporting plan.

## 8. Safety and pharmacovigilance

### 8.1 Signal detection

- Spontaneous reporting systems: AEMS (formerly FAERS), VAERS, EudraVigilance.
- Disproportionality metrics: PRR, ROR, BCPNN.
- Case series and clinical review.
- Temporal and geographic clustering.
- Label changes and risk management.

### 8.2 Limits

- Underreporting.
- Confounding by indication.
- Stimulated reporting.
- Duplicates.
- Not incidence rates.

**Rule:** a disproportionality signal is a hypothesis, not proof.

## 9. Real-world evidence

Use cases:

- Label expansion.
- Post-authorization safety.
- Effectiveness.
- Health economics.
- Trial feasibility.
- External control arms.

Requirements:

- Reliable data.
- Fit-for-purpose design.
- Pre-specification.
- Confounder control.
- Sensitivity analysis.
- Regulatory engagement.

## 10. AI in discovery and development

| Application | Promise | Evidence gap |
|---|---|---|
| Target discovery | Integrate omics and literature | Causal validation |
| Molecular design | Generate candidates | Synthesis and assay |
| ADMET prediction | Reduce failures | Prospective validation |
| Trial recruitment | Find patients | Site and consent |
| Digital endpoints | Remote measurement | Regulatory qualification |
| Safety prediction | Early signal | Real-world validation |
| RWE | Faster evidence | Data quality and confounding |

**Industry lesson:** AI can compress discovery timelines, but the bottleneck often moves to validation, manufacturing, and clinical evidence.

## 11. Industry case studies

### 11.1 RECOVERY and pragmatic trials

RECOVERY demonstrated that large, simple, pragmatic trials can answer urgent questions quickly.

**Lesson:** design simplicity and routine data can beat complexity.

Source: RECOVERY Collaborative Group, *New England Journal of Medicine*, 2021. https://doi.org/10.1056/NEJMoa2021436

### 11.2 AlphaFold and structure-based discovery

AlphaFold accelerated structural biology but did not replace experimental validation.

**Lesson:** prediction is an input to discovery, not a product.

Source: Jumper, J. et al., *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2

### 11.3 CDISC and submission data

CDISC standards enable consistent regulatory submissions and re-use.

**Lesson:** data standards are a commercial and regulatory capability, not just IT.

Source: CDISC. https://www.cdisc.org/

### 11.4 FDA RWE and external controls

FDA has accepted RWE for specific regulatory decisions, often with extensive pre-submission discussion.

**Lesson:** RWE is possible, but the evidentiary bar is high and context-specific.

Source: FDA, *Real-World Evidence*. https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence

### 11.5 Safety signal detection

Post-market safety surveillance has identified rare adverse events not seen in trials.

**Lesson:** spontaneous reporting is biased but essential; combine with active surveillance and clinical review.

Source: FDA AEMS. https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems

### 11.6 AI-designed molecules enter clinical trials

A 2025 review estimated that more than 75 AI-designed molecules had entered human trials by mid-2025, up from none in 2020; this is a review estimate, not a registry count. Insilico Medicine reported Phase IIa results for rentosertib in idiopathic pulmonary fibrosis: 71 patients, FVC change +98.4 mL at 60 mg once daily versus −20.3 mL for placebo, with safety as the primary endpoint met. No AI-discovered drug had been approved as of the source cutoff.

**Lesson:** AI can accelerate molecule generation, but the binding constraint has shifted to target validation, clinical translation, and regulatory evidence. A Phase IIa signal is not approval.

Sources: *Pharmacological Reviews*, 2025. https://pharmrev.aspetjournals.org/article/S0031-6997(25)07512-X/abstract; Insilico Medicine announcement. https://insilico.com/news/tnrecuxsc1-insilico-announces-nature-medicine-publi

## 12. Labs

### Lab 13.1 — Trial feasibility

**Objective:** Estimate eligible patients for a trial protocol.

**Steps:**

1. Translate eligibility into computable criteria.
2. Query claims/EHR data.
3. Estimate screen failure.
4. Map sites.
5. Compare with ClinicalTrials.gov.
6. Write a feasibility memo.

**Deliverable:** feasibility report.

**Acceptance criteria:** criteria transparent; assumptions stated; site variation.

### Lab 13.2 — SDTM mapping

**Objective:** Map a small dataset to SDTM domains.

**Steps:**

1. Identify domains (DM, AE, LB, VS, EX).
2. Map variables.
3. Apply controlled terminology.
4. Validate referential integrity.
5. Document mapping.

**Deliverable:** mapping specification.

**Acceptance criteria:** traceability; terminology correct; no invented variables.

### Lab 13.3 — Safety signal detection

**Objective:** Compute disproportionality for a drug-event pair.

**Steps:**

1. Load AEMS (formerly FAERS) or a synthetic dataset.
2. Compute PRR/ROR.
3. Compare with background.
4. Review case narratives.
5. Assess confounding and reporting bias.

**Deliverable:** signal assessment.

**Acceptance criteria:** metrics correct; limitations; no causal overclaim.

### Lab 13.4 — RWE study

**Objective:** Design an external control or comparative-effectiveness study.

**Steps:**

1. Define the target trial.
2. Select data.
3. Match or weight.
4. Analyze outcomes.
5. Sensitivity analysis.
6. Regulatory question list.

**Deliverable:** protocol and analysis.

**Acceptance criteria:** time zero; confounding; sensitivity.

### Lab 13.5 — Recruitment forecast

**Objective:** Forecast enrollment by site.

**Steps:**

1. Use historical site performance.
2. Model activation and screen failure.
3. Simulate scenarios.
4. Identify bottlenecks.
5. Recommend site mix.

**Deliverable:** forecast and memo.

**Acceptance criteria:** assumptions; uncertainty; actionable.

## 13. Assignment: translational data plan

Write a 2,500-word plan for a drug program:

1. Target and biological rationale.
2. Biomarker strategy.
3. Trial design and endpoints.
4. Feasibility and recruitment.
5. Safety monitoring.
6. RWE plan.
7. Data standards and submission.
8. AI use and validation.
9. Ethical and commercial issues.
10. Milestones and go/no-go criteria.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Biology and target | 20% | Generic | Causal rationale | Validated evidence |
| Biomarker | 20% | Unvalidated | Analytical and clinical plan | Companion diagnostic strategy |
| Trial design | 20% | Generic | Feasible and powered | Adaptive and pragmatic options |
| Safety | 15% | Ignores | Signal detection | Benefit-risk and surveillance |
| RWE | 15% | Naive | Target trial | Regulatory-ready |
| Communication | 10% | Technical | Clear | Cross-functional |

## 15. Regulatory, ethical, and commercial lens

- **FDA/EMA:** pre-submission meetings, scientific advice, and qualification pathways.
- **GCP:** ethical and scientific quality standard for trials.
- **21 CFR Part 11:** electronic records and signatures.
- **CDISC:** submission data standards.
- **HTA:** payer evidence needs differ from regulatory needs.
- **Ethics:** trial equity, informed consent, access after trial.
- **Commercial:** pricing, reimbursement, and market access shape which programs are viable.

## 16. Failure modes and red flags

- Biomarker without assay validation.
- Eligibility criteria that cannot be operationalized.
- Recruitment forecast without site variability.
- Safety signal treated as causal.
- RWE without pre-specification.
- Ignoring standard of care changes.
- AI claims without prospective validation.
- Submission data that cannot be traced.
- No payer evidence plan.

## 17. Interview and discussion questions

1. **How would you prioritize a drug target?**
   Hint: genetic, expression, pathway, safety, tractability.
2. **What is a companion diagnostic?**
   Hint: test required for safe/effective use of a therapy.
3. **How do you estimate trial feasibility?**
   Hint: computable eligibility, claims/EHR, screen failure, sites.
4. **What is SDTM?**
   Hint: submission tabulation standard.
5. **How do you detect a safety signal?**
   Hint: disproportionality, case review, temporal/geographic.
6. **What are the limits of AEMS (formerly FAERS)?**
   Hint: underreporting, confounding, duplicates, no denominator.
7. **When is RWE acceptable to regulators?**
   Hint: reliable data, fit-for-purpose design, pre-specification.
8. **How does AI help drug discovery?**
   Hint: target ID, design, ADMET; validation bottleneck.
9. **What is a digital endpoint?**
   Hint: sensor-derived measure of clinical benefit; needs qualification.
10. **How do you handle missing trial data?**
    Hint: estimands, multiple imputation, sensitivity.
11. **What is the difference between regulatory and HTA evidence?**
    Hint: efficacy/safety vs cost-effectiveness and budget impact.
12. **How do you communicate uncertainty to a portfolio committee?**
    Hint: probability of success, value, timelines, downside.

## 18. Key terms

| Term | Definition |
|---|---|
| Target | Molecule or pathway intended to be modulated |
| Biomarker | Measurable indicator of biology or response |
| Companion diagnostic | Test required for therapy use |
| SDTM / ADaM | CDISC submission and analysis data models |
| GCP | Good Clinical Practice |
| GxP | Good practice regulations |
| Pharmacovigilance | Safety monitoring |
| AEMS | FDA Adverse Event Monitoring System (formerly FAERS) |
| PRR / ROR | Disproportionality metrics |
| RWE | Real-world evidence |
| External control | Comparator outside the trial |
| Digital endpoint | Sensor-derived clinical measure |
| HTA | Health technology assessment |

## 19. Sources and further reading

### Standards and regulation
- CDISC: https://www.cdisc.org/
- FDA, *Real-World Evidence*. https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
- FDA, *Part 11*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application
- ICH GCP: https://www.ich.org/page/efficacy-guidelines
- EMA: https://www.ema.europa.eu/
- ClinicalTrials.gov: https://clinicaltrials.gov/
- FDA AEMS: https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems

### Methods and industry
- RECOVERY Collaborative Group, *New England Journal of Medicine*, 2021. https://doi.org/10.1056/NEJMoa2021436
- Jumper, J. et al., *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2
- Hernán, M. A., Robins, J. M., *Causal Inference: What If*.
- Open Targets: https://www.opentargets.org/

> **Next:** [Module 14](14_medical_devices_samd_regulation.md) turns these models and data into regulated medical devices and software.
