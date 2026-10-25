# Module 18 — Quality, Evaluation, and Reporting

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 24 hours (9 lecture, 5 case clinic, 10 lab)
> **Track fit:** Common core
> **Prerequisites:** Modules 05, 06, 14
> **Tools:** Reporting checklists, model cards, datasheets, risk-of-bias tools, audit templates
> **Industry question:** "Would an independent reviewer reach the same conclusion from our evidence — and can they reproduce it?"

---

## 1. Why industry cares

Biomedical AI has an evidence-quality problem.

- Many FDA-cleared imaging devices were validated retrospectively and on limited populations.
- Many clinical AI studies omit calibration, subgroup performance, and workflow evaluation.
- Many models are reported with a single AUROC and no confidence interval.
- Many deployed systems are never monitored.
- Many recalls occur soon after clearance, often for software issues.

Quality and reporting are not academic formalities. They determine whether a regulator, payer, clinician, or hospital will trust the product.

## 2. Learning outcomes

By the end of this module, you can:

- Select the right reporting guideline for a study.
- Appraise a clinical AI study for bias and applicability.
- Write a model card and datasheet.
- Build a reproducibility package.
- Design an audit for a deployed model.
- Evaluate post-market performance and recalls.
- Distinguish statistical performance from clinical utility.
- Report uncertainty, calibration, and subgroup performance.
- Communicate limitations without undermining the work.
- Prepare evidence for regulatory, payer, and procurement review.

## 3. Mental model: evidence is a chain

```
Question → Design → Data → Analysis → Reporting → Decision → Monitoring
   │          │        │        │          │           │
   └──────────┴────────┴────────┴──────────┴───────────┘
                    each link can break
```

A strong model with a weak design is weak evidence. A strong design with poor reporting is not usable.

## 4. Reporting guidelines

| Guideline | Scope | Key items |
|---|---|---|
| TRIPOD+AI | Prediction models using AI | Intended use, data, predictors, performance, calibration, fairness, reproducibility |
| CONSORT-AI | Randomized trials of AI interventions | Intervention description, input/output, integration, human-AI interaction, errors |
| SPIRIT-AI | Protocols for AI trials | Trial design, intervention, monitoring, harms |
| STARD-AI | Diagnostic accuracy studies | Population, index test, reference standard, flow, bias |
| DECIDE-AI | Early clinical evaluation | Human factors, workflow, safety, reporting |
| CLAIM | Imaging AI | Data, model, training, evaluation, performance |
| PROBAST | Risk of bias in prediction models | Participants, predictors, outcome, analysis |
| QUADAS-2 | Diagnostic accuracy bias | Patient selection, index test, reference standard, flow |
| CONSORT | Randomized trials | Standard trial reporting |
| STROBE | Observational studies | Cohort, case-control, cross-sectional |
| PRISMA | Systematic reviews | Search, selection, bias, synthesis |

**Rule:** choose the guideline before the analysis, not after.

## 5. Appraising clinical AI evidence

Ask:

1. **Intended use:** Is it clearly stated?
2. **Population:** Does it match the deployment population?
3. **Data source:** Is it representative, and are labels valid?
4. **Split:** Is it patient-level, temporal, or external?
5. **Comparator:** Current practice or a weak baseline?
6. **Metrics:** Discrimination, calibration, utility, safety, equity?
7. **Sample size:** Adequate for subgroups?
8. **Missing data:** Handled and reported?
9. **Workflow:** Was human-AI interaction evaluated?
10. **Harms:** False positives, false negatives, alert fatigue, automation bias?
11. **Reproducibility:** Code, model, data access?
12. **Funding and conflicts:** Disclosed?
13. **Regulatory status:** Cleared, approved, research-only?
14. **Post-market:** Monitoring plan?

### 5.1 Evidence-quality red flags

- Single-center retrospective study.
- Random split with patient overlap.
- AUROC only.
- No calibration.
- No confidence intervals.
- No subgroup analysis.
- No workflow evaluation.
- No limitations section.
- Vendor-funded only.
- "Clinically validated" without a clinical study.

## 6. Model cards and datasheets

### 6.1 Model card

```markdown
# Model card

## Model details
- Name, version, date, owners
- Type, architecture, training data
- License and intended use

## Intended use
- Primary intended users
- Primary intended uses
- Out-of-scope uses

## Factors
- Relevant groups, environments, instruments

## Metrics
- Performance measures
- Decision thresholds
- Uncertainty

## Evaluation data
- Datasets, splits, preprocessing
- Subgroup performance
- Calibration

## Training data
- Sources, size, labels, preprocessing
- Known biases

## Quantitative analyses
- Disaggregated results
- Fairness metrics

## Ethical considerations
- Risks, harms, mitigations

## Caveats and recommendations
- Limitations
- Monitoring
- Contact
```

### 6.2 Datasheet

A datasheet documents:

- Motivation.
- Composition.
- Collection process.
- Preprocessing.
- Uses.
- Distribution.
- Maintenance.

**Rule:** a model card without evaluation data and limitations is marketing.

## 7. Reproducibility

### 7.1 Checklist

- [ ] Code repository with license.
- [ ] Environment lockfile or container.
- [ ] Data version and access instructions.
- [ ] Preprocessing code.
- [ ] Random seeds.
- [ ] Splits and cohort definition.
- [ ] Hyperparameters.
- [ ] Evaluation scripts.
- [ ] Model weights or access.
- [ ] Expected outputs.
- [ ] Hardware and runtime.
- [ ] Known nondeterminism.

### 7.2 Levels

| Level | Description |
|---|---|
| Repeatable | Same team, same setup |
| Reproducible | Different team, same data/code |
| Replicable | Different data, same methods |
| Generalizable | Different population/setting |

## 8. Quality management

- Design controls.
- Requirements traceability.
- Verification and validation.
- Risk management.
- Change control.
- CAPA.
- Audit.
- Post-market surveillance.

**Regulated context:** quality is a system, not a final report.

## 9. Post-market evaluation

### 9.1 What to monitor

- Performance drift.
- Calibration.
- Subgroup performance.
- Alert burden.
- Override rate.
- Safety events.
- Complaints.
- Recalls.
- Software vulnerabilities.

### 9.2 Recall lessons

Among 950 cleared AI devices, 6.3% had recalls; 43.4% of recalls occurred within 12 months. Diagnostic/measurement errors were the most common cause (109 of 182 recalls).

**Implication:** early monitoring is critical. A model can pass clearance and fail in deployment.

## 10. Industry case studies

### 10.1 FDA imaging device evidence

A review of 903 FDA-authorized AI-enabled devices through August 2024 (76.6% radiology) found that only 55.9% reported clinical performance studies; 38.2% were retrospective, 8.1% prospective, 2.4% randomized, and 16.2% reported AUC.

**Lesson:** clearance is not proof of clinical utility. Ask what evidence was actually submitted.

Source: *JAMA Network Open*, 2025. https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2833324

### 10.2 AI device recalls

Among 950 cleared AI devices, 6.3% had recalls; 43.4% of recalls occurred within 12 months. Diagnostic/measurement errors were the most common cause (109 of 182 recalls).

**Lesson:** post-market surveillance is a core quality function.

Source: *JAMA Health Forum*, 2025. https://jamanetwork.com/journals/jama-health-forum/fullarticle/2837802

### 10.3 Epic Sepsis Model

External validation found lower discrimination and high alert burden than vendor claims.

**Lesson:** independent evaluation and local monitoring are essential.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### 10.4 HealthBench

HealthBench used 262 physicians and 48,562 rubric criteria to evaluate LLM responses; even strong models scored only 32% on the Hard subset.

**Lesson:** evaluation must be clinically grounded, not benchmark-only.

Source: OpenAI, *HealthBench*. https://openai.com/index/healthbench/

### 10.5 Ambient scribe trials

RCTs show modest time savings and quality gains, but no hard outcome improvement.

**Lesson:** measure the outcome that matters, not just user satisfaction.

Source: *NEJM AI*, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12768499/

## 11. Labs

### Lab 18.1 — Appraise a study

**Objective:** Appraise a clinical AI paper with PROBAST/TRIPOD+AI.

**Steps:**

1. Identify intended use.
2. Assess participants and data.
3. Assess predictors and outcome.
4. Assess analysis and validation.
5. Score risk of bias.
6. Write a 1-page decision memo.

**Deliverable:** appraisal.

**Acceptance criteria:** specific evidence; no generic criticism.

### Lab 18.2 — Write a model card

**Objective:** Document a model from Module 06 or 10.

**Steps:**

1. Describe intended use.
2. Report evaluation data and metrics.
3. Add subgroup and calibration results.
4. State limitations.
5. Define monitoring.

**Deliverable:** model card.

**Acceptance criteria:** complete; honest; usable by an external reviewer.

### Lab 18.3 — Reproducibility audit

**Objective:** Reproduce another team's analysis.

**Steps:**

1. Clone the repository.
2. Build the environment.
3. Run the pipeline.
4. Compare outputs.
5. Document discrepancies.
6. Propose fixes.

**Deliverable:** audit report.

**Acceptance criteria:** exact commands; discrepancy quantified.

### Lab 18.4 — Post-market monitoring plan

**Objective:** Design monitoring for a deployed model.

**Steps:**

1. Define metrics.
2. Set thresholds.
3. Identify data sources.
4. Define review cadence.
5. Define escalation.
6. Link to change control.

**Deliverable:** plan.

**Acceptance criteria:** actionable; owned; tied to safety.

### Lab 18.5 — Recall analysis

**Objective:** Analyze a public device recall.

**Steps:**

1. Identify the device and recall reason.
2. Map to risk controls.
3. Assess whether monitoring could have detected it.
4. Propose preventive action.

**Deliverable:** case analysis.

**Acceptance criteria:** specific; sourced; no speculation.

## 12. Assignment: evidence dossier

Write a 2,500-word dossier:

1. Question and intended use.
2. Study design.
3. Data and labels.
4. Model and validation.
5. Performance and calibration.
6. Subgroup and fairness.
7. Workflow and human factors.
8. Limitations.
9. Reproducibility.
10. Monitoring and change control.

## 13. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Design | 20% | Weak split | Appropriate validation | External and prospective |
| Metrics | 20% | AUROC only | Calibration and utility | Subgroup and decision impact |
| Reporting | 20% | Incomplete | Guideline-aligned | Audit-ready |
| Reproducibility | 20% | Partial | Code and environment | Independent replication |
| Critical appraisal | 20% | Generic | Specific biases | Quantified residual risk |

## 14. Regulatory, ethical, and commercial lens

- **FDA:** QMSR, PCCP, post-market surveillance.
- **EU:** MDR/IVDR clinical evaluation and PMCF.
- **Payer:** evidence of clinical utility and cost.
- **Procurement:** security and references.
- **Ethics:** transparent reporting of harms and subgroup effects.

## 15. Failure modes and red flags

- Reporting after analysis.
- Cherry-picked metrics.
- No calibration.
- No confidence intervals.
- No subgroup analysis.
- No limitations.
- Irreproducible pipeline.
- No monitoring.
- Model card as marketing.
- Ignoring post-market evidence.

## 16. Interview and discussion questions

1. **Which reporting guideline applies to a prediction model?**
   Hint: TRIPOD+AI.
2. **What is PROBAST?**
   Hint: risk-of-bias tool for prediction models.
3. **Why is calibration essential?**
   Hint: probabilities drive decisions.
4. **What should a model card contain?**
   Hint: intended use, metrics, subgroups, limitations, monitoring.
5. **How do you audit reproducibility?**
   Hint: clean environment, code, data, seeds, expected outputs.
6. **What is the difference between reproducible and replicable?**
   Hint: same data/code vs new data.
7. **Why do AI device recalls matter?**
   Hint: early failures, software issues, monitoring gaps.
8. **How do you report subgroup performance?**
   Hint: sample sizes, confidence intervals, calibration, harms.
9. **What is clinical utility?**
   Hint: improved decision or outcome, not just accuracy.
10. **How do you report limitations?**
    Hint: specific, quantified, tied to decisions.
11. **What is a datasheet?**
    Hint: documentation of dataset motivation, composition, collection, uses.
12. **How do you evaluate an LLM clinically?**
    Hint: expert rubric, grounding, safety, bias, prospective.

## 17. Key terms

| Term | Definition |
|---|---|
| TRIPOD+AI | Reporting guideline for AI prediction models |
| CONSORT-AI | Reporting guideline for AI trials |
| STARD-AI | Reporting guideline for AI diagnostic accuracy |
| PROBAST | Risk-of-bias tool for prediction models |
| Model card | Structured model documentation |
| Datasheet | Structured dataset documentation |
| Calibration | Predicted vs observed risk |
| Clinical utility | Improvement in decision or outcome |
| Post-market surveillance | Monitoring after launch |
| CAPA | Corrective and preventive action |
| Reproducibility | Same data/code yields same result |
| Replicability | New data yields consistent result |

## 18. Sources and further reading

- TRIPOD+AI: https://www.tripod-statement.org/
- CONSORT-AI / SPIRIT-AI: https://www.consort-spirit.org/
- STARD-AI: https://www.equator-network.org/
- PROBAST: https://www.probast.org/
- CLAIM: https://www.equator-network.org/
- FDA, *QMSR*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
- FDA, *PCCP*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents
- *JAMA Network Open* imaging device evidence, 2025. https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2833324
- *JAMA Health Forum* AI device recalls, 2025. https://jamanetwork.com/journals/jama-health-forum/fullarticle/2837802
- HealthBench: https://openai.com/index/healthbench/
- Model Cards: https://arxiv.org/abs/1810.03993
- Datasheets for Datasets: https://arxiv.org/abs/1803.09010

> **Next:** [Module 19](19_capstone_hackathons.md) turns the curriculum into a capstone and hackathon experience.
