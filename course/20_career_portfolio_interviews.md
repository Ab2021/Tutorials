# Module 20 — Career, Portfolio, and Interviews

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 12 hours (6 lecture/workshop, 6 practice)
> **Track fit:** Common core
> **Prerequisites:** Modules 00–18
> **Tools:** Portfolio site, GitHub, resume, interview log
> **Industry question:** "Can a hiring manager see, in 10 minutes, that you can turn biomedical data into a trustworthy decision?"

---

## 1. Why this module exists

Biomedical data science hiring is not a single test. It is a series of inferences:

- Can this person write reliable code?
- Do they understand clinical context?
- Can they design a valid study?
- Can they communicate with clinicians and executives?
- Do they know when regulation applies?
- Will they be safe in a high-stakes environment?

Your portfolio and interviews must provide evidence for each inference.

## 2. Learning outcomes

By the end of this module, you can:

- Map your experience to industry roles and levels.
- Build a portfolio of three decision-ready artifacts.
- Write a case study that shows judgment, not just results.
- Tailor a resume and LinkedIn profile to a role.
- Prepare for technical, clinical, regulatory, product, and behavioral interviews.
- Complete a take-home without over-engineering.
- Design a system for a clinical AI use case.
- Negotiate an offer with market context.
- Build a 90-day onboarding plan.

## 3. Role map

| Role | Core work | Evidence employers want |
|---|---|---|
| Clinical/Health Data Scientist | Cohorts, models, evaluation | EHR/claims project, validation, clinical communication |
| Biomedical Data Scientist | Multimodal research translation | Imaging/signal/omics project, reproducibility |
| RWE Scientist | Causal inference, claims/EHR | Target trial emulation, regulatory awareness |
| HEOR/Outcomes Scientist | Economic models, PROs | Cost-effectiveness, budget impact |
| Bioinformatics Scientist | Omics pipelines, discovery | Genomics/single-cell project, biology |
| ML Engineer, Clinical AI | Production ML, MLOps | Deployed service, monitoring, tests |
| Imaging Scientist | Segmentation/detection | DICOM pipeline, external validation |
| Biostatistician | Trial design, analysis | Protocol, estimands, SAP |
| Clinical Data Manager | Trial data, standards | CDISC, data quality |
| Pharmacovigilance Data Scientist | Safety signals | AEMS (formerly FAERS), disproportionality, causality |
| Regulatory Data/AI Specialist | Evidence, submissions | Regulatory package, QMS literacy |
| AI Product Manager, Health | Discovery, value, adoption | Product case, stakeholder map |

### 3.1 Market context

- BLS median data scientist salary: $120,230 (May 2025); 90th percentile $199,130.
- Health-sector postings often range from ~$105k to ~$200k+ depending on role, city, and seniority.
- Titles and pay vary widely. Use levels.fyi, BLS, and recent postings, not a single aggregator.

## 4. Portfolio

### 4.1 Three artifacts

1. **Prediction/decision artifact:** a model with temporal validation, calibration, decision curve, subgroup analysis, and a model card.
2. **Causal/evidence artifact:** a target trial emulation or study protocol with a DAG, estimand, and sensitivity analysis.
3. **Systems artifact:** a reproducible pipeline or deployed service with tests, monitoring, and a runbook.

Optional fourth: a regulatory or product dossier.

### 4.2 Case study structure

```markdown
# Project title

## Decision
Who acts, when, and what changes?

## Data
Source, population, access, limitations.

## Method
Cohort, features, split, model, evaluation.

## Results
Primary metric with confidence interval; calibration; subgroups.

## What I would do differently
Specific, honest.

## Regulatory and ethical considerations
Intended use, risks, mitigations.

## Artifacts
Code, model card, memo, demo.
```

### 4.3 What not to do

- Do not lead with accuracy.
- Do not hide negative results.
- Do not use a dataset you cannot describe.
- Do not claim clinical validity from a public benchmark.
- Do not include PHI or proprietary data.
- Do not paste notebook output as a portfolio.

## 5. Resume and LinkedIn

### 5.1 Resume bullets

Use **action + method + metric + context + decision**.

Weak: "Built a machine learning model for sepsis."

Strong: "Built a temporal-validation sepsis risk model on 42k admissions; improved AUPRC from 0.18 to 0.31 over the existing screen and reduced simulated alerts per 1,000 patient-days by 37%; documented calibration and subgroup performance for clinical review."

### 5.2 Keywords

Use the language of the role:

- Clinical data, EHR, claims, OMOP, FHIR.
- Causal inference, target trial emulation, survival analysis.
- Calibration, decision curve, external validation.
- Regulatory, SaMD, GMLP, PCCP.
- MLOps, drift, monitoring.
- RWE, HEOR, budget impact.

## 6. Interview types

| Type | What they test | Preparation |
|---|---|---|
| Coding | Python/SQL fluency | Practice data manipulation, not LeetCode only |
| ML fundamentals | Bias-variance, calibration, leakage | Explain trade-offs with examples |
| Clinical case | Framing, harm, workflow | Practice with a clinician |
| Study design | Estimand, confounding, validation | Write protocols |
| Regulatory | Device/CDS classification, evidence | Read guidance summaries |
| Product | Value, adoption, prioritization | Build value cases |
| Behavioral | Collaboration, conflict, failure | STAR stories |
| Take-home | Judgment under time pressure | Scope, baseline, document |
| System design | Architecture, latency, monitoring | Draw diagrams |

## 7. Technical interview topics

- Probability and statistics.
- Regression and regularization.
- Trees and boosting.
- Neural networks and transformers.
- Calibration and decision curves.
- Missing data.
- Causal inference.
- Survival analysis.
- Leakage and validation.
- Fairness.
- SQL window functions.
- Python data structures.
- Distributed computing basics.
- MLOps and monitoring.

## 8. Clinical case interview

A common prompt:

> "A hospital wants to predict readmissions. How would you approach it?"

A strong answer:

1. Clarify the decision and user.
2. Ask about the intervention and capacity.
3. Define the cohort, index event, and outcome.
4. Discuss label limitations.
5. Choose a temporal split.
6. Choose metrics: calibration, decision curve, subgroup.
7. Discuss workflow and alert burden.
8. Discuss equity.
9. Discuss monitoring.
10. State what would make you stop.

A weak answer starts with "I would use XGBoost."

## 9. Take-home strategy

- Read the prompt twice.
- Ask clarifying questions.
- Define scope and assumptions.
- Build a simple baseline first.
- Use a temporal or grouped split.
- Report calibration and uncertainty.
- Write a short memo with limitations.
- Stop when the decision is supported; do not over-engineer.
- Include a README and reproduction steps.

## 10. System design

For a clinical AI system, cover:

- Users and workflow.
- Data sources and ingestion.
- Feature computation and time zero.
- Model serving and latency.
- EHR integration.
- Human factors.
- Monitoring and drift.
- Security and privacy.
- Failure modes and fallback.
- Regulatory change control.
- Cost.

## 11. Behavioral interviews

Prepare STAR stories for:

- A project that failed.
- A disagreement with a clinician.
- A time you simplified a solution.
- A time you found leakage or bias.
- A time you influenced a decision.
- A time you managed ambiguity.
- A time you handled sensitive data.
- A time you had to say no.

## 12. Negotiation

- Know your market range.
- Negotiate total compensation, not base only.
- Ask about scope, title, remote policy, and learning.
- Get the offer in writing.
- Be transparent but not adversarial.
- Do not negotiate with another offer you do not have.

## 13. 90-day onboarding plan

| Days | Focus | Output |
|---|---|---|
| 1–30 | Learn the data, workflow, and stakeholders | Data map and stakeholder map |
| 31–60 | Reproduce a key analysis and identify gaps | Validation memo |
| 61–90 | Deliver a small, useful improvement | Measured impact and next steps |

## 14. Interview question bank

1. **Tell me about a model you built. What decision did it change?**
   Hint: decision, user, evidence, outcome.
2. **How do you validate a clinical model?**
   Hint: patient-level, temporal, external, calibration, subgroup.
3. **What is data leakage?**
   Hint: temporal, patient, preprocessing, label.
4. **How do you choose a threshold?**
   Hint: harms, capacity, prevalence, equity.
5. **How do you handle missing data?**
   Hint: mechanism, imputation, sensitivity.
6. **When is a model a medical device?**
   Hint: intended use, patient-specific output, claim.
7. **How do you explain a model to a clinician?**
   Hint: patient impact, uncertainty, action, limitations.
8. **How do you measure fairness?**
   Hint: multiple definitions, subgroups, harms.
9. **How do you monitor a deployed model?**
   Hint: drift, calibration, outcomes, alert burden, subgroups.
10. **Tell me about a project that failed.**
    Hint: ownership, learning, specific change.
11. **How do you prioritize?**
    Hint: impact, probability, time, cost, strategy.
12. **How do you handle a stakeholder who wants an unsafe shortcut?**
    Hint: evidence, escalation, documentation, patient safety.
13. **What is an estimand?**
    Hint: population, treatment, outcome, time, summary.
14. **How do you evaluate an LLM?**
    Hint: expert rubric, grounding, safety, bias, prospective.
15. **Why do you want this role?**
    Hint: specific mission, skills, evidence.

## 15. Portfolio rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 20% | Model-first | Decision-first | Workflow and value |
| Evidence | 25% | Accuracy only | Validation and calibration | External and subgroup |
| Reproducibility | 20% | Notebook | README and env | Independent rerun |
| Communication | 20% | Technical | Clear memo | Executive and clinical |
| Judgment | 15% | No limitations | Honest limitations | Clear stop criteria |

## 16. Sources and further reading

- BLS, *Data Scientists*. https://www.bls.gov/ooh/math/data-scientists.htm
- LinkedIn, *Jobs on the Rise 2026*. https://www.linkedin.com/pulse/linkedin-jobs-rise-2026-25-fastest-growing-roles-us-linkedin-news-dlb1c
- Levels.fyi: https://www.levels.fyi/
- 80,000 Hours health AI career guide: https://80000hours.org/
- Fast.ai, *Practical Deep Learning*. https://course.fast.ai/
- Chip Huyen, *Designing Machine Learning Systems*.
- *Cracking the Coding Interview* for coding practice.

> **Next:** [Module 21](21_datasets_tools_benchmarks.md) provides the datasets, tools, and benchmarks to build your portfolio.
