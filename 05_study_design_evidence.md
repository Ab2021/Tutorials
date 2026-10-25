# Module 05 — Study Design and Evidence Generation

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 24 hours (10 lecture, 6 case clinic, 8 lab)
> **Track fit:** Common core; required for Biopharma
> **Prerequisites:** Module 04
> **Tools:** Protocol templates, sample-size calculators, CONSORT/STARD/TRIPOD checklists
> **Industry question:** "What is the smallest, most credible study that can support this claim?"

---

## 1. Why industry cares

Evidence is the currency of biomedical products. A model, drug, device, or service does not succeed because it is clever; it succeeds because a regulator, payer, clinician, or investor believes the claim.

The evidence strategy determines:

- How long development takes.
- How much it costs.
- What claims can be made.
- Whether reimbursement is possible.
- How much liability the organization carries.

Data scientists often inherit a study after it is designed. That is too late. The design determines what can be estimated, which metrics are valid, and whether the result will be believed.

This module teaches you to choose and critique study designs across the product lifecycle.

## 2. Learning outcomes

By the end of this module, you can:

- Match a claim to an appropriate evidence design.
- Distinguish explanatory from pragmatic trials, and efficacy from effectiveness.
- Write a protocol with a clear estimand, eligibility, intervention, comparator, outcomes, and analysis plan.
- Design internal, temporal, external, and prospective validation for a model or device.
- Evaluate real-world evidence for regulatory and payer use.
- Recognize and mitigate selection, information, confounding, attrition, and reporting biases.
- Interpret diagnostic accuracy and prediction-model performance in context.
- Explain sample-size and power drivers.
- Critically appraise a clinical trial, observational study, and AI validation study.
- Build an evidence roadmap for a product.

## 3. Mental model: claim → evidence → decision

```
Claim
  │
  ├── What exactly is claimed? (intended use, population, effect)
  ├── Who needs to believe it? (regulator, payer, clinician, patient, investor)
  ├── What decision follows? (approve, cover, adopt, prescribe, buy)
  └── What evidence would be sufficient? (design, endpoint, duration, size)
        │
        ▼
   Study protocol → execution → analysis → reporting → decision
```

**Rule:** start with the claim and the decision, not the dataset.

## 4. Evidence hierarchy is not a ladder

The "hierarchy of evidence" is a useful heuristic, not a universal ranking. A well-conducted observational study can be more credible than a small, biased trial. A randomized trial can be useless if the population, comparator, or endpoint is wrong.

| Design | Strength | Key limitation | Best for |
|---|---|---|---|
| Systematic review / meta-analysis | Synthesizes evidence | Heterogeneity, publication bias | Baseline evidence |
| Randomized controlled trial | Causal inference | Cost, generalizability, ethics | Efficacy and safety |
| Pragmatic trial | Real-world effectiveness | Heterogeneity, contamination | Implementation and policy |
| Prospective cohort | Temporal ordering, multiple outcomes | Confounding, attrition | Prognosis, biomarkers |
| Retrospective cohort | Feasible, large | Confounding, data quality | Signal generation, RWE |
| Case-control | Rare outcomes | Selection, recall bias | Rare diseases |
| Cross-sectional | Prevalence | No temporality | Burden, screening |
| Case series | Feasibility | No comparison | Early safety |
| Expert opinion | Context | Bias | Guideline development |

## 5. Trial designs

### 5.1 Phases

| Phase | Purpose | Typical size | Key question |
|---|---|---:|---|
| Preclinical | Mechanism, toxicology | In vitro/animal | Is it plausible and safe enough? |
| Phase I | Safety, dose | 20–100 | What dose is tolerable? |
| Phase II | Efficacy signal, dose-ranging | 100–300 | Does it work in this population? |
| Phase III | Confirmatory efficacy and safety | 300–3,000+ | Does it work, and is the benefit worth the risk? |
| Phase IV | Post-approval safety and effectiveness | Large | What happens in real-world use? |

### 5.2 Design features

- **Randomization:** allocation by chance; reduces confounding.
- **Blinding:** masking participants, clinicians, assessors, analysts.
- **Control:** placebo, active comparator, usual care, historical control.
- **Parallel vs crossover:** independent groups vs within-person comparisons.
- **Superiority vs non-inferiority vs equivalence:** different hypotheses and margins.
- **Fixed vs adaptive:** pre-specified adaptations vs fixed design.
- **Interim analysis:** early stopping for efficacy, futility, or harm.
- **Enrichment:** select patients likely to respond.
- **Master protocols:** basket, umbrella, platform trials.

### 5.3 Explanatory vs pragmatic

| Dimension | Explanatory | Pragmatic |
|---|---|---|
| Question | Can it work? | Does it work in practice? |
| Population | Highly selected | Routine care |
| Intervention | Protocolized | Flexible |
| Comparator | Placebo | Usual care |
| Outcomes | Surrogate | Patient-relevant |
| Setting | Academic centers | Diverse settings |
| Adherence | Monitored | Variable |

**Industry lesson:** a drug that works in an explanatory trial may fail in practice because of adherence, access, or workflow. Design evidence for the decision you need.

### 5.4 Estimands

ICH E9(R1) provides a framework for aligning the trial objective, estimand, and analysis.

An estimand has five attributes:

1. **Population:** who is the target?
2. **Treatment condition:** which intervention and comparator?
3. **Variable:** what outcome?
4. **Intercurrent events:** how are they handled (treatment policy, hypothetical, composite, while-on-treatment, principal stratum)?
5. **Population-level summary:** difference in means, risk ratio, hazard ratio, etc.

**Example:** a trial of a heart-failure drug with discontinuation due to adverse events.

- Treatment-policy estimand: effect of assignment, regardless of discontinuation.
- Hypothetical estimand: effect if everyone tolerated treatment.
- Principal-stratum estimand: effect in those who would tolerate treatment.

The choice changes the analysis and the claim.

### 5.5 Sample size and power

Power depends on:

- Effect size.
- Variance.
- Significance level.
- Power target.
- Allocation ratio.
- Attrition.
- Cluster correlation.
- Multiple comparisons.

**Rule:** a study powered for a surrogate may be underpowered for the clinical outcome you actually care about.

## 6. Real-world evidence

### 6.1 Sources

- EHR data.
- Claims.
- Registries.
- Patient-generated data.
- Wearables.
- Product registries.
- Death records.
- Pharmacy data.

### 6.2 FDA RWE framework

The FDA evaluates RWE based on:

- **Data reliability:** accuracy, completeness, provenance, quality.
- **Study design and execution:** ability to support causal inference.
- **Regulatory decision:** fit for the specific question.

RWE can support label changes, new indications, and post-market requirements, but not automatically. Early engagement with regulators is essential.

### 6.3 External control arms

An external control arm uses data from outside the trial to compare with a single-arm trial.

**Requirements:**

- Comparable population and eligibility.
- Comparable standard of care and era.
- Consistent endpoint definitions.
- Confounder control.
- Sensitivity analyses.
- Regulatory agreement.

**Risk:** unmeasured confounding and drift. External controls are not a free substitute for randomization.

### 6.3a Digital twins and virtual controls

A digital twin is a patient-specific computational model used to simulate disease progression or treatment response. In drug development it may support trial design, dose selection, enrichment, or a virtual control arm.

**What it can do:** explore scenarios, reduce sample size, and sharpen endpoint choice.

**What it cannot do yet:** replace a randomized control arm as definitive evidence. Data-driven synthetic controls have no general approval precedent; process-driven models (PBPK, population PK/PD, quantitative systems pharmacology) have a longer regulatory history. EMA has qualified prognostic covariate adjustment (e.g., PROCOVA) to improve precision, not to replace randomization.

**Evidence needed:** biological plausibility, calibration, external validation, sensitivity analysis, and pre-specification with regulators.

### 6.4 Target trial emulation for RWE

Use the target trial framework (Module 04) to make design choices explicit. Pre-specify the protocol and analysis. Report the emulation's limitations.

## 7. Validation of models and devices

### 7.1 Levels of validation

| Level | Description | Strength |
|---|---|---|
| Apparent | Performance on development data | Optimistic, not credible |
| Internal | Cross-validation or bootstrap | Checks overfitting |
| Temporal | Later data from same site | Checks drift |
| External | Different site or population | Checks generalizability |
| Prospective | Silent mode or live use | Checks workflow and real-time performance |
| Impact | Randomized or stepped-wedge evaluation | Checks outcomes and cost |

### 7.2 Diagnostic accuracy

- **Sensitivity / specificity:** intrinsic test characteristics.
- **ROC AUC:** discrimination across thresholds.
- **Precision-recall:** useful for rare outcomes.
- **Calibration:** agreement between predicted and observed risk.
- **Decision curve analysis:** net benefit across threshold probabilities.
- **Likelihood ratios:** update from pretest to post-test probability.

### 7.3 Prediction-model reporting

TRIPOD+AI extends TRIPOD for models using artificial intelligence. Report:

- Intended use and population.
- Data sources and dates.
- Predictors and outcome definitions.
- Sample size and missing data.
- Model development and hyperparameters.
- Performance with confidence intervals.
- Calibration.
- Subgroup performance.
- Fairness.
- Reproducibility.
- Limitations.

### 7.4 Clinical trial AI reporting

CONSORT-AI and SPIRIT-AI extend trial reporting for AI interventions. DECIDE-AI addresses early clinical evaluation. STARD-AI addresses diagnostic accuracy.

## 8. Bias taxonomy

| Bias | Definition | Example | Mitigation |
|---|---|---|---|
| Selection | Non-representative sample | Healthy volunteers | Consecutive recruitment, weighting |
| Confounding | Common cause | Sicker patients treated | Randomization, adjustment |
| Information | Measurement error | Differential coding | Validated measures, blinded assessment |
| Attrition | Differential dropout | Sick patients leave | Intention-to-treat, imputation |
| Immortal time | Follow-up before exposure | Treatment initiation delay | Time-varying exposure, landmark |
| Lead-time | Earlier detection | Screening | Lead-time-adjusted analysis |
| Length-time | Slower-progressing cases | Screening | Randomized design |
| Recall | Memory differences | Case-control | Prospective measurement |
| Publication | Negative results unpublished | Literature | Registration, funnel plots |
| Reporting | Selective outcomes | Multiple endpoints | Pre-registration |
| Automation | Over-trust in AI | Clinician accepts error | Human factors, audit |
| Spectrum | Different case mix | Validation in tertiary center | External validation |

## 9. Industry case studies

### 9.1 RECOVERY trial

The RECOVERY trial used a large, pragmatic, randomized platform design to evaluate COVID-19 treatments in the UK. It identified dexamethasone as effective in severely ill patients and hydroxychloroquine as ineffective.

**What transferred:** simple design, broad eligibility, routine data, rapid recruitment, platform adaptivity.
**What did not:** the results were specific to the studied population and standard of care.

Source: RECOVERY Collaborative Group, *New England Journal of Medicine*, 2021. https://doi.org/10.1056/NEJMoa2021436

### 9.2 Apple Heart Study

The Apple Heart Study used a smartwatch-based irregular rhythm notification and a pragmatic, virtual design. It demonstrated feasibility but also highlighted the need for confirmation and the challenges of false positives.

**Industry lesson:** consumer-scale screening generates downstream clinical workload; validation must include the care pathway.

Source: Perez, M. V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation," *New England Journal of Medicine*, 2019. https://doi.org/10.1056/NEJMoa1901183

### 9.3 IDx-DR

IDx-DR was an autonomous AI system for diabetic retinopathy screening, authorized by the FDA through De Novo in 2018. Its evidence package included a prospective clinical trial and clear intended use.

**Industry lesson:** a narrow intended use, a well-defined workflow, and prospective evidence can support autonomy.

Source: FDA De Novo DEN180001. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### 9.4 Epic Sepsis Model external validation

The external validation found lower discrimination and high alert burden than vendor claims.

**Industry lesson:** independent validation in the deployment setting is part of evidence generation, not an optional extra.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### 9.5 CMS coverage and evidence

CMS may cover a technology with evidence development (CED) when evidence is promising but insufficient. This creates a pathway for real-world evidence generation.

**Industry lesson:** reimbursement is an evidence question. Design studies to satisfy payer questions, not only regulators.

Source: CMS Coverage with Evidence Development. https://www.cms.gov/medicare/coverage/coverage-with-evidence-development

## 10. Labs

### Lab 5.1 — Write a protocol

**Objective:** Draft a 2-page protocol for a model or device evaluation.

**Steps:**

1. State the claim and decision.
2. Define the estimand.
3. Specify eligibility, intervention, comparator, outcomes, follow-up.
4. Choose the design.
5. Pre-specify the analysis and sensitivity analyses.
6. Identify biases and mitigations.

**Deliverable:** protocol.

**Acceptance criteria:** estimand aligned; design fit for claim; biases addressed.

### Lab 5.2 — Sample size and power

**Objective:** Calculate sample size for three scenarios.

**Steps:**

1. Continuous outcome: mean difference.
2. Binary outcome: risk difference.
3. Time-to-event: hazard ratio.
4. Adjust for 20% attrition.
5. Discuss how clustering changes the calculation.

**Deliverable:** calculations and assumptions.

**Acceptance criteria:** formulas correct; assumptions explicit; sensitivity to effect size shown.

### Lab 5.3 — Appraise an AI validation study

**Objective:** Critically appraise a published clinical AI study.

**Steps:**

1. Identify intended use and population.
2. Assess dataset and split.
3. Check calibration and subgroup performance.
4. Check for leakage.
5. Assess workflow and impact claims.
6. Score with TRIPOD+AI or PROBAST.

**Deliverable:** structured appraisal.

**Acceptance criteria:** specific, evidence-based critique; no generic statements.

### Lab 5.4 — Design an RWE study

**Objective:** Design a target trial emulation using claims or EHR data.

**Steps:**

1. Define the decision and estimand.
2. Specify the target trial.
3. Choose data and cohort.
4. Address confounding and time zero.
5. Plan sensitivity analyses.
6. Write a regulatory pre-submission question.

**Deliverable:** protocol and question list.

**Acceptance criteria:** feasibility addressed; assumptions explicit; regulatory relevance clear.

### Lab 5.5 — Evidence roadmap

**Objective:** Build a 3-year evidence roadmap for a product.

**Steps:**

1. List claims needed for approval, coverage, and adoption.
2. Map each claim to a study design.
3. Sequence studies by risk and cost.
4. Identify go/no-go decision points.
5. Estimate budget and timeline.

**Deliverable:** roadmap table and Gantt chart.

**Acceptance criteria:** claims and studies linked; risks prioritized; budget realistic.

## 11. Assignment: evidence generation plan

For a product idea, write a 2,500-word evidence plan:

1. Claim and intended use.
2. Regulatory pathway.
3. Payer evidence needs.
4. Clinical adoption evidence.
5. Study designs and endpoints.
6. Sample sizes and timelines.
7. Bias mitigation.
8. Go/no-go criteria.
9. Budget estimate.
10. Risks and mitigations.

## 12. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Claim-design alignment | 25% | Generic study | Design fits claim | Design anticipates regulator and payer |
| Estimand | 15% | Vague | Clear attributes | Handles intercurrent events |
| Bias | 20% | Lists biases | Mitigates key biases | Quantifies residual risk |
| Validation | 20% | Internal only | Temporal/external | Prospective and impact |
| Feasibility | 10% | Ignores cost | Realistic timeline | Risk-adjusted roadmap |
| Communication | 10% | Technical | Clear | Executive-ready |

## 13. Regulatory, ethical, and commercial lens

- **FDA:** pre-submission (Q-Sub) meetings can clarify evidence expectations.
- **EMA:** scientific advice and qualification of novel methodologies.
- **HTA bodies:** NICE, ICER, and others have distinct evidence requirements.
- **CMS:** coverage decisions may require CED or specific endpoints.
- **Ethics:** randomized designs must be equitable; avoid exploiting vulnerable populations.
- **Commercial:** evidence generation is an investment; prioritize claims that unlock reimbursement and adoption.

## 14. Failure modes and red flags

- Designing a study without a claim.
- Using a surrogate endpoint without justification.
- Ignoring intercurrent events.
- Powering for the wrong outcome.
- Using historical controls without accounting for drift.
- Claiming effectiveness from a single-arm study.
- Validating only on a random split.
- Ignoring calibration.
- Reporting only overall performance.
- No pre-specification.
- No plan for negative results.

## 15. Interview and discussion questions

1. **What is an estimand, and why does it matter?**
   Hint: aligns question, design, and analysis; handles intercurrent events.
2. **When is a pragmatic trial preferable to an explanatory trial?**
   Hint: when the decision is about real-world effectiveness and implementation.
3. **What makes RWE credible to a regulator?**
   Hint: reliable data, appropriate design, pre-specification, fit for question.
4. **What is an external control arm, and when is it risky?**
   Hint: single-arm trials; confounding and drift.
5. **How do you validate a model for a new hospital?**
   Hint: external validation, calibration, subgroup, workflow, monitoring.
6. **What is the difference between efficacy and effectiveness?**
   Hint: ideal conditions vs routine practice.
7. **Why can a statistically significant trial fail commercially?**
   Hint: small effect, wrong population, cost, access, adherence, reimbursement.
8. **What is a surrogate endpoint? Give an example.**
   Hint: biomarker or imaging marker; not necessarily patient-relevant.
9. **How would you design a study for a screening AI?**
   Hint: prospective, intended-use population, false-positive burden, follow-up, outcomes.
10. **What is coverage with evidence development?**
    Hint: payer coverage conditional on generating more evidence.
11. **How do you handle multiple endpoints?**
    Hint: pre-specify hierarchy, alpha control, false discovery rate.
12. **What evidence would a hospital procurement team ask for?**
    Hint: clinical validation, security, integration, liability, cost, references.

## 16. Key terms

| Term | Definition |
|---|---|
| Efficacy | Benefit under ideal conditions |
| Effectiveness | Benefit in routine practice |
| Estimand | Precise definition of the treatment effect |
| Intercurrent event | Event after treatment initiation that affects interpretation |
| Pragmatic trial | Trial designed to reflect routine practice |
| Explanatory trial | Trial designed to test efficacy under controlled conditions |
| Non-inferiority | Not worse than comparator by a pre-specified margin |
| Adaptive design | Pre-specified design modification based on interim data |
| External control arm | Comparator from outside the trial |
| RWE | Real-world evidence |
| CED | Coverage with evidence development |
| Surrogate endpoint | Indirect marker of clinical benefit |
| Calibration | Agreement between predicted and observed risk |
| Decision curve | Net benefit across thresholds |
| TRIPOD+AI | Reporting guideline for AI prediction models |
| CONSORT-AI | Reporting guideline for AI trials |
| SPIRIT-AI | Protocol guideline for AI trials |
| STARD-AI | Reporting guideline for AI diagnostic studies |
| PROBAST | Risk-of-bias tool for prediction models |

## 17. Sources and further reading

### Guidance and reporting
- FDA, *Real-World Evidence*. https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
- FDA, *Clinical Decision Support Software*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- ICH, *E9(R1) Addendum on Estimands and Sensitivity Analysis*. https://www.ich.org/
- CONSORT-AI: https://www.consort-spirit.org/
- SPIRIT-AI: https://www.consort-spirit.org/
- STARD-AI: https://www.equator-network.org/
- TRIPOD+AI: https://www.tripod-statement.org/
- PROBAST: https://www.probast.org/
- CMS Coverage with Evidence Development: https://www.cms.gov/medicare/coverage/coverage-with-evidence-development

### Trials and RWE
- RECOVERY Collaborative Group, "Dexamethasone in Hospitalized Patients with Covid-19," *New England Journal of Medicine*, 2021. https://doi.org/10.1056/NEJMoa2021436
- Perez, M. V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation," *New England Journal of Medicine*, 2019. https://doi.org/10.1056/NEJMoa1901183
- Hernán, M. A., Robins, J. M., "Using Big Data to Emulate a Target Trial," *American Journal of Epidemiology*, 2016. https://doi.org/10.1093/aje/kwv254
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model," *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### Books
- Friedman, L. M., Furberg, C. D., DeMets, D. L., *Fundamentals of Clinical Trials*.
- Rothman, K. J., Greenland, S., Lash, T. L., *Modern Epidemiology*.
- Steyerberg, E. W., *Clinical Prediction Models*.

> **Next:** [Module 06](06_machine_learning_foundations.md) builds the machine-learning foundations needed to execute these designs responsibly.
