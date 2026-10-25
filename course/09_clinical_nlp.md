# Module 09 — Clinical NLP

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 24 hours (9 lecture, 3 case clinic, 12 lab)
> **Track fit:** Health Systems; common elective
> **Prerequisites:** Modules 02, 06, 07
> **Tools:** spaCy, scispaCy, Hugging Face Transformers, regex, a de-identification tool, an annotation platform
> **Industry question:** "Can we turn clinical text into reliable, auditable facts without inventing what is not there?"

---

## 1. Why industry cares

Most clinical detail lives in text: notes, discharge summaries, radiology reports, pathology reports, operative notes, messages, and care plans. Structured codes capture billing and some diagnoses; they miss symptoms, severity, social context, treatment response, and uncertainty.

Clinical NLP supports:

- Phenotyping and cohort discovery.
- Risk adjustment and quality measurement.
- Coding and revenue-cycle support.
- Documentation improvement.
- Summarization and handoffs.
- Patient messaging and triage.
- Safety event detection.
- Research abstraction.
- Prior authorization.

The hard part is not extracting words. It is extracting **meaning, assertion, temporality, and attribution** reliably enough for the decision.

## 2. Learning outcomes

By the end of this module, you can:

- Map clinical note structure and documentation behavior.
- Design an annotation guideline with high inter-annotator agreement.
- Build rule-based, dictionary-based, and transformer-based extraction pipelines.
- Detect negation, uncertainty, family history, and temporality.
- Evaluate extraction at the span, concept, and patient level.
- De-identify clinical text and evaluate residual risk.
- Build a grounded summarization workflow with citation and abstention.
- Detect copy-forward and leakage in text-derived features.
- Assess bias and equity in NLP performance.
- Write a clinical NLP validation report suitable for a product or regulatory review.

## 3. Mental model: text is a clinical assertion, not a bag of words

```
Note → Section → Sentence → Concept → Assertion → Temporality → Attribution → Patient-level fact
```

A mention of "sepsis" is not necessarily a sepsis diagnosis. It could be:

- "No evidence of sepsis." (negated)
- "Rule out sepsis." (uncertain)
- "History of sepsis." (historical)
- "Mother had sepsis." (family history)
- "Sepsis protocol initiated." (process)
- "Concern for sepsis" (clinical concern, not confirmed)

## 4. Clinical text sources

| Source | Characteristics | Common tasks |
|---|---|---|
| Discharge summary | Synthesized, structured, copy-forward | Phenotyping, readmission |
| Progress note | Daily, narrative, variable | Deterioration, treatment response |
| Admission note | Detailed history and plan | Diagnosis, risk factors |
| Radiology report | Structured sections, findings/impression | Finding extraction, triage |
| Pathology report | Specimen, diagnosis, staging | Cancer phenotyping |
| Operative note | Procedure details | Complications, outcomes |
| Nursing note | Frequent, observational | Falls, skin, mobility |
| ED note | Time-sensitive, variable | Triage, disposition |
| Patient message | Patient-generated, informal | Symptom triage, access |
| Care plan | Goals, social needs | Care management |

## 5. Documentation behavior

### 5.1 Structure

Notes often have sections:

- Chief complaint.
- History of present illness.
- Past medical history.
- Medications.
- Allergies.
- Social history.
- Review of systems.
- Physical exam.
- Assessment and plan.
- Labs and imaging.

Templates vary by specialty, site, and clinician.

### 5.2 Copy-forward

Copy-forward reuses prior text. It can:

- Preserve stale diagnoses.
- Duplicate future plans into current notes.
- Inflate note length.
- Create false temporal signals.
- Leak outcomes.

**Detection:** compare note similarity over time, flag copied sections, and examine timestamp provenance.

### 5.3 Negation and uncertainty

Common cues:

- Negation: "no", "without", "denies", "negative for".
- Uncertainty: "possible", "suspected", "rule out", "cannot exclude".
- Hypothetical: "if", "would consider".
- Family history: "mother", "father", "family history".
- Historical: "history of", "status post", "resolved".

**Rule-based tools:** NegEx, ConText, and extensions.

### 5.4 Attribution

Who is the subject?

- Patient.
- Family member.
- Clinician.
- Device.

Attribution errors are common in templated notes and messages.

## 6. NLP tasks

| Task | Input | Output | Example |
|---|---|---|---|
| NER | Text | Entity spans | "metformin" → medication |
| Assertion | Entity + context | Negated/uncertain/historical | "no pneumonia" |
| Relation extraction | Text | Relations | drug–dose, disease–site |
| Phenotyping | Text | Patient-level label | rheumatoid arthritis |
| Coding | Text | ICD/CPT codes | billing support |
| Summarization | Note(s) | Summary | handoff |
| Question answering | Question + corpus | Answer + evidence | guideline query |
| De-identification | Text | Redacted text | research dataset |
| Triage | Message | Urgency | patient portal |
| Quality measurement | Text | Measure numerator/denominator | documentation |

## 7. Methods

### 7.1 Rule-based and dictionary

Strengths:

- Transparent.
- Fast.
- Easy to audit.
- Good for well-defined patterns.

Weaknesses:

- Brittle to paraphrase.
- High maintenance.
- Poor recall on varied language.

### 7.2 Classical machine learning

- Bag-of-words + logistic regression.
- CRF for sequence labeling.
- SVM for classification.

Useful baselines, especially with limited data.

### 7.3 Transformers

- ClinicalBERT, PubMedBERT, BioBERT.
- Fine-tuned token classification for NER.
- Sequence classification for phenotyping.
- Generative models for summarization and QA.

**Caution:** domain shift across sites, note types, and coding practices.

### 7.4 LLMs

LLMs can:

- Extract structured data with prompts.
- Summarize notes.
- Answer questions.
- Generate code mappings.

They can also:

- Hallucinate facts.
- Omit critical information.
- Be inconsistent across runs.
- Leak PHI if used improperly.
- Change behavior across versions.

**Use with:**

- Retrieval or source text in context.
- Structured output schemas.
- Citation requirements.
- Human review.
- Deterministic settings where possible.
- Version pinning.

## 8. Annotation

### 8.1 Guideline design

A good annotation guideline includes:

- Task definition and scope.
- Entity categories and exclusions.
- Assertion categories.
- Temporality rules.
- Border cases.
- Examples.
- Adjudication process.

### 8.2 Inter-annotator agreement

- **Cohen's kappa:** two annotators, categorical.
- **Fleiss' kappa:** multiple annotators.
- **Krippendorff's alpha:** flexible.
- **Span-level F1:** for extraction.

Report agreement before modeling. Low agreement means the task or guideline is ambiguous.

### 8.3 Sample size

Annotate enough examples to estimate performance with acceptable confidence intervals. Stratify by note type, site, and subgroup.

## 9. Evaluation

### 9.1 Levels

- **Span-level:** exact or partial match.
- **Concept-level:** normalized concept correct.
- **Document-level:** patient has the condition.
- **Patient-level:** cohort classification.
- **Downstream:** model or decision impact.

### 9.2 Metrics

- Precision, recall, F1.
- Sensitivity/specificity at patient level.
- Agreement with chart review.
- Calibration for probabilistic outputs.
- Error severity, not just count.

### 9.3 Error analysis

Categorize errors:

- Negation.
- Temporality.
- Attribution.
- Section confusion.
- Abbreviation.
- Copy-forward.
- Annotation error.
- Model hallucination.

Report examples and rates.

## 10. De-identification

### 10.1 Identifiers

Names, dates, locations, phone numbers, MRNs, URLs, emails, ages over 89, rare diagnoses, device identifiers, and free-text identifiers.

### 10.2 Methods

- Regex for structured patterns.
- NER for names and locations.
- Dictionary for local providers and facilities.
- Human review.
- Redaction or surrogate replacement.

### 10.3 Evaluation

- Precision/recall by identifier type.
- Residual risk assessment.
- False negatives (leaked identifiers).
- Utility loss (over-redaction).

**Rule:** de-identification is a safety-critical process. Measure it.

## 11. Summarization and grounding

### 11.1 Clinical summarization

Tasks:

- Discharge summary generation.
- Handoff summary.
- Chart review.
- Prior authorization.
- Patient-friendly explanation.

### 11.2 Risks

- Omission of critical findings.
- Hallucinated facts.
- Wrong attribution.
- Outdated information.
- Losing uncertainty.

### 11.3 Grounding

Require each statement to cite a source span. Use extractive-first approaches. Abstain when evidence is insufficient. Human review for high-stakes summaries.

## 12. Industry case studies

### 12.1 n2c2 and i2b2 challenges

The n2c2/i2b2 shared tasks advanced clinical NLP for de-identification, concept extraction, relation extraction, and cohort selection. They established benchmark datasets and evaluation methods.

**Lesson:** shared tasks create comparable benchmarks, but leaderboard performance may not generalize to local data.

Source: n2c2. https://n2c2.dbmi.hms.harvard.edu/

### 12.2 MedCAT and concept annotation

MedCAT is an open-source tool for clinical concept extraction and linking to SNOMED CT, enabling phenotyping at scale.

**Lesson:** concept normalization and local vocabulary tuning are essential.

Source: Kraljevic, Z. et al., "Multi-domain clinical natural language processing with MedCAT," *Artificial Intelligence in Medicine*, 2021. https://doi.org/10.1016/j.artmed.2021.102083

### 12.3 Radiology report labeling

CheXpert used NLP to extract labels from radiology reports, including uncertainty. The approach enabled large-scale supervision but inherited labeling errors.

**Lesson:** NLP-derived labels are labels, with error. Validate them.

Source: Irvin, J. et al., *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590

### 12.4 Clinical documentation and burnout

Ambient scribes and documentation assistants aim to reduce clerical burden. Evidence varies; accuracy, note quality, and workflow matter.

**Lesson:** documentation tools must be evaluated on clinician time, note quality, and patient safety, not just word error rate.

Source: AAFP / AMA documentation burden resources; peer-reviewed evaluations.

### 12.5 Patient portal triage

NLP can triage patient messages by urgency. False reassurance is a safety risk; false escalation creates workload.

**Lesson:** design for a human safety net and measure both error directions.

Source: Health system implementations; peer-reviewed triage studies.

## 13. Labs

### Lab 9.1 — De-identification pipeline

**Objective:** Build and evaluate a de-identification pipeline.

**Data:** Public synthetic or de-identified notes.

**Steps:**

1. Implement regex for structured identifiers.
2. Add NER for names and locations.
3. Evaluate by identifier type.
4. Inspect false negatives.
5. Measure utility loss.
6. Document residual risk.

**Deliverable:** pipeline and evaluation.

**Acceptance criteria:** no real PHI; per-type metrics; residual risk stated.

### Lab 9.2 — NER and assertion

**Objective:** Extract medications and determine assertion status.

**Steps:**

1. Annotate 50 sentences with a guideline.
2. Build a dictionary baseline and a transformer model.
3. Add negation and uncertainty rules.
4. Evaluate span and assertion F1.
5. Analyze errors by category.

**Deliverable:** model and error analysis.

**Acceptance criteria:** guideline; agreement; per-category errors.

### Lab 9.3 — Phenotyping from notes

**Objective:** Identify patients with a condition from discharge summaries.

**Steps:**

1. Define the phenotype.
2. Build keyword, rule-based, and transformer baselines.
3. Validate against chart review.
4. Measure patient-level sensitivity and PPV.
5. Assess subgroup performance.

**Deliverable:** phenotype validation report.

**Acceptance criteria:** clinician-reviewed sample; confidence intervals; subgroup analysis.

### Lab 9.4 — Grounded summarization

**Objective:** Summarize a note with citations.

**Steps:**

1. Build a prompt with source text and citation requirements.
2. Generate summaries.
3. Check every claim against the source.
4. Measure omission and hallucination.
5. Add abstention for unsupported claims.

**Deliverable:** summarization evaluation.

**Acceptance criteria:** claim-level audit; hallucination rate; abstention behavior.

### Lab 9.5 — Copy-forward detection

**Objective:** Detect duplicated text across notes.

**Steps:**

1. Compute similarity between consecutive notes.
2. Flag copied sections.
3. Inspect temporal leakage.
4. Quantify impact on a text-derived feature.
5. Propose mitigation.

**Deliverable:** copy-forward report.

**Acceptance criteria:** method reproducible; impact quantified.

### Lab 9.6 — Bias audit

**Objective:** Compare NLP performance across demographic and language groups.

**Steps:**

1. Define groups.
2. Measure extraction and phenotype performance.
3. Inspect note length and documentation differences.
4. Identify mechanisms.
5. Propose mitigation and monitoring.

**Deliverable:** bias audit.

**Acceptance criteria:** sample sizes reported; mechanisms discussed.

## 14. Assignment: clinical NLP validation report

Write a 2,000-word report:

1. Task and intended use.
2. Corpus and note types.
3. Annotation guideline and agreement.
4. Method and baseline.
5. Evaluation at span, patient, and downstream levels.
6. Error taxonomy.
7. De-identification and privacy.
8. Bias and equity.
9. Limitations and monitoring.
10. Recommendation.

## 15. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Task definition | 15% | Vague | Clear and scoped | Tied to decision |
| Annotation | 20% | No guideline | Guideline and agreement | Adjudication and edge cases |
| Method | 20% | Single model | Baseline comparison | Error-driven design |
| Evaluation | 25% | Accuracy only | Span and patient metrics | Downstream and subgroup |
| Privacy | 10% | Ignores PHI | De-identification | Residual-risk assessment |
| Communication | 10% | Technical | Clear | Clinical and regulatory ready |

## 16. Regulatory, ethical, and commercial lens

- **SaMD:** NLP that informs diagnosis or treatment may be a device.
- **Transparency:** clinicians should know when text is machine-generated or machine-extracted.
- **Privacy:** text is the highest-risk PHI source. Use approved environments.
- **Bias:** documentation differences across groups can propagate into NLP.
- **Liability:** a missed finding in a summary can cause harm.
- **Commercial:** documentation and coding tools have clear ROI but require workflow validation.

## 17. Failure modes and red flags

- Treating a mention as a diagnosis.
- Ignoring negation.
- Ignoring temporality.
- Training on copy-forward text.
- Evaluating only on the same site.
- Reporting token accuracy instead of clinical utility.
- No human review for summaries.
- Sending PHI to unapproved APIs.
- Over-redacting and destroying utility.
- No plan for note-template changes.

## 18. Interview and discussion questions

1. **Why is clinical NLP harder than general NLP?**
   Hint: negation, temporality, attribution, abbreviations, domain shift.
2. **How do you detect negation?**
   Hint: rule-based cues, context windows, transformer models, annotation.
3. **What is the difference between a mention and a diagnosis?**
   Hint: assertion and patient-level interpretation.
4. **How do you evaluate de-identification?**
   Hint: per-identifier precision/recall, residual risk, utility loss.
5. **What is copy-forward, and why does it matter?**
   Hint: stale and future information; leakage.
6. **How would you build a phenotype?**
   Hint: guideline, dictionary, rules, model, chart review, subgroup.
7. **When would you use an LLM for extraction?**
   Hint: flexible schemas, few-shot, but verify and ground.
8. **How do you prevent hallucination in summarization?**
   Hint: citations, extractive-first, abstention, human review.
9. **What is inter-annotator agreement, and why does it matter?**
   Hint: task ambiguity and label quality.
10. **How do you handle abbreviations?**
    Hint: local dictionaries, context, disambiguation.
11. **What are the risks of NLP-derived labels?**
    Hint: error propagation, bias, leakage.
12. **How do you monitor an NLP model after deployment?**
    Hint: note template changes, drift, error rates, subgroup performance.

## 19. Key terms

| Term | Definition |
|---|---|
| NER | Named entity recognition |
| Assertion | Negated, uncertain, historical, or present status |
| Temporality | When an event occurred relative to the note |
| Attribution | Who the subject of a statement is |
| Phenotype | Computable patient-level condition definition |
| Copy-forward | Reuse of prior note text |
| De-identification | Removal or transformation of identifiers |
| Grounding | Linking output to source evidence |
| Abstention | Declining to answer when evidence is insufficient |
| Inter-annotator agreement | Consistency among human annotators |
| Concept normalization | Mapping mentions to standard codes |
| Section segmentation | Splitting notes into clinical sections |

## 20. Sources and further reading

### Methods and datasets
- n2c2: https://n2c2.dbmi.hms.harvard.edu/
- i2b2: https://www.i2b2.org/
- MIMIC-IV Note: https://mimic.mit.edu/
- Kraljevic, Z. et al., "Multi-domain clinical natural language processing with MedCAT," *Artificial Intelligence in Medicine*, 2021. https://doi.org/10.1016/j.artmed.2021.102083
- Irvin, J. et al., "CheXpert," *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- Savova, G. K. et al., "Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES)," *JAMIA*, 2010. https://doi.org/10.1136/jamia.2009.001560
- Chapman, W. W. et al., "A simple algorithm for identifying negated findings and diseases in discharge summaries," *Journal of Biomedical Informatics*, 2001. https://doi.org/10.1006/jbin.2001.1029

### Guidance and tools
- HIPAA de-identification: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- scispaCy: https://allenai.github.io/scispacy/
- spaCy: https://spacy.io/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- MedCAT: https://github.com/CogStack/MedCAT

> **Next:** [Module 10](10_medical_imaging.md) moves from text to pixels and volumes.
