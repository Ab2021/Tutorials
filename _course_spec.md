# Course Production Specification

**Program:** Biomedical Data Science for Industry — 2026 Edition
**Package root:** `D:\Biomedical\biomedical-data-science-2026`
**Source cutoff:** 24 October 2026. Prefer 2024–2026 sources. Never present a rumored, proposed, or future-dated item as settled fact.
**Audience:** Working data scientists, ML engineers, biomedical engineers, clinical informaticists, and advanced graduate students moving into health, pharma, medical-device, sensor, diagnostics, or health-insurance roles.

## Non-negotiable design principles

1. **Industry-first, not textbook-first.** Every module starts from a real workflow, decision, product, or regulated deliverable. Theory is introduced because it changes a decision.
2. **Evidence and harm.** Every model is framed by the clinical or commercial decision it supports, the cost of false positives and false negatives, and the evidence needed to trust it.
3. **Regulatory literacy.** Do not treat regulation as an appendix. Name the applicable regime, the stage at which it binds, and the engineering artifact it requires.
4. **Data realism.** Include missingness, leakage, site effects, coding drift, label noise, selection bias, workflow mismatch, and access constraints. Do not teach only clean benchmark datasets.
5. **Reproducibility.** Labs must specify data version, split strategy, preprocessing fit scope, random seeds, metrics with confidence intervals, and artifact retention.
6. **Equity and human factors.** Include subgroup performance, accessibility, neurodiversity, language, consent, and clinician/patient workflow.
7. **Commercial literacy.** Connect technical work to adoption, reimbursement, procurement, time-to-value, and total cost of ownership.
8. **No fabricated citations.** Every named regulation, product, dataset, company, competition, or paper must be real and traceable. If a fact is uncertain, label it `[VERIFY]` and explain what must be checked. Use URLs in the sources section.
9. **No padding.** Line count is a proxy for depth: use substantive tables, decision rules, worked examples, lab steps, rubrics, and interview prompts, not repeated boilerplate.
10. **Book grounding.** Map relevant chapters/concepts from the supplied clinical books into the clinical modules. Do not reproduce copyrighted text; synthesize and cite chapter-level concepts.

## Required files and ownership

| File | Module | Target lines | Owner |
|---|---|---:|---|
| `README.md` | Program guide, manifest, outcomes, how to use | 350–500 | lead |
| `00_orientation_and_industry_map.md` | 0 | 420–520 | lead |
| `01_clinical_foundations.md` | 1 | 480–580 | content agent |
| `01A_foundations_assumptions_and_units.md` | 01A | 700–900 | lead |
| `02_biomedical_data_landscape.md` | 2 | 430–530 | content agent |
| `03_data_engineering_interoperability.md` | 3 | 480–580 | content agent |
| `04_statistics_causal_inference.md` | 4 | 480–580 | content agent |
| `05_study_design_evidence.md` | 5 | 450–550 | content agent |
| `06_machine_learning_foundations.md` | 6 | 480–580 | content agent |
| `07_deep_learning_multimodal_llms.md` | 7 | 480–580 | content agent |
| `08_ehr_claims_operations.md` | 8 | 450–550 | content agent |
| `09_clinical_nlp.md` | 9 | 450–550 | content agent |
| `10_medical_imaging.md` | 10 | 480–580 | content agent |
| `11_physiological_signals_wearables.md` | 11 | 480–580 | content agent |
| `12_omics_precision_medicine.md` | 12 | 480–580 | content agent |
| `13_biopharma_clinical_development.md` | 13 | 480–580 | content agent |
| `14_medical_devices_samd_regulation.md` | 14 | 500–600 | content agent |
| `14A_medical_hardware_sensors_chips_robotics.md` | 14A | 800–950 | lead |
| `14B_efficient_ai_distillation_quantization_on_device_inference.md` | 14B | 700–900 | lead |
| `15_deployment_mlops.md` | 15 | 450–550 | content agent |
| `16_privacy_security_ethics.md` | 16 | 450–550 | content agent |
| `17_product_economics_leadership.md` | 17 | 450–550 | content agent |
| `18_quality_evaluation_reporting.md` | 18 | 430–530 | content agent |
| `19_capstone_hackathons.md` | 19 | 550–700 | content agent |
| `20_career_portfolio_interviews.md` | 20 | 400–500 | content agent |
| `21_datasets_tools_benchmarks.md` | 21 | 450–600 | content agent |
| `22_sources_and_further_reading.md` | 22 | 400–600 | lead + agents |
| `23_specialty_electives.md` | 23 | 900–1,100 | content agent |
| `24_hardware_robotics_labs.md` | 24 | 650–850 | content agent |
| `25_trial_operations_hta_global_regulatory.md` | 25 | 500–700 | content agent |
| `26_cloud_streaming_production_labs.md` | 26 | 450–650 | content agent |

Target package total: **≥ 20,000 lines** excluding `_course_spec.md` and research notes.

## Module template (adapt headings where useful, but keep this order)

```markdown
# Module NN — Title

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** X hours (Y lecture, Z lab, W project)
> **Track fit:** Common core / Health systems / Biopharma / MedTech & Sensors
> **Prerequisites:** ...
> **Tools:** ...
> **Industry question:** "..."

## 1. Why industry cares
Short business/clinical framing; who pays; what decision changes; what failure costs.

## 2. Learning outcomes
6–10 measurable outcomes using verbs (design, implement, critique, defend).

## 3. Mental model
A compact conceptual model, diagram in text, or decision table.

## 4. Core concepts
Subsections with definitions, equations, worked examples, and "industry interpretation" callouts.
Use tables for comparisons and decision rules.

## 5. Data, standards, and workflow
Data sources, schemas, coding systems, access constraints, privacy, data quality checks.

## 6. Industry case studies
3–5 real cases. For each: context, intervention/model, evidence, outcome, what transferred, what did not. Cite sources.

## 7. Labs
3–6 labs. Each: objective, data, steps, deliverable, acceptance criteria, stretch goal.

## 8. Assignment / project
A realistic deliverable (memo, model, validation report, regulatory artifact).

## 9. Assessment rubric
Table with criteria, weights, novice/competent/expert descriptors.

## 10. Regulatory, ethical, and commercial lens
Applicable rules; equity; human factors; reimbursement/procurement implications.

## 11. Failure modes and red flags
Specific ways projects fail in industry.

## 12. Interview and discussion questions
8–15 questions, with answer hints for the instructor.

## 13. Key terms
15–30 terms with one-line definitions.

## 14. Sources and further reading
Grouped: regulation/standards; peer-reviewed; industry; datasets/tools. URLs and access dates.
```

## Evidence rules

- Use **primary sources** for regulation: FDA guidance pages, EUR-Lex, HHS, ONC, CMS, NIST, ISO/IEC summaries, ICH.
- Use **peer-reviewed papers or official benchmark pages** for model performance. Never quote a marketing accuracy number as if it were a validated clinical result.
- For company case studies, prefer peer-reviewed evaluations, FDA decision summaries, 510(k) databases, clinicaltrials.gov, company technical reports, and reputable reporting. Label vendor claims.
- Dates: state "as of October 2026" for rapidly changing rules. Distinguish enacted law, final guidance, draft guidance, and enforcement discretion.
- Competition details: name year, task, data, metric, and URL. Mark 2026 events that are announced but not completed.

## Style

- Markdown, ATX headings, tables, fenced code only where it adds value.
- Use SI/clinical units correctly; never mix mg/dL and mmol/L without conversion.
- Prefer short paragraphs and decision tables over long prose.
- Define acronyms on first use.
- Use "clinician" and "patient" generically; use they/them for unnamed people.
- No emojis. No fake quotes. No invented statistics.
- Include `[VERIFY]` only for genuinely uncertain, time-sensitive claims; each must have a verification note.

## Book map (supplied library)

- **Bates' Guide to Physical Examination and History Taking** — history, interview, physical exam, vital signs, documentation, clinical reasoning.
- **Goldberger's Clinical Electrocardiography** — ECG morphology, rhythm, ischemia, chamber enlargement, clinical correlation.
- **The Only EKG Book You'll Ever Need** — ECG fundamentals and pattern recognition.
- **Learning Radiology** — imaging modalities, chest/abdominal/musculoskeletal/neuro imaging, interpretation workflow.
- **Pathophysiology of Heart Disease** — cardiovascular physiology and disease mechanisms.
- **West's Respiratory Physiology** — ventilation, gas exchange, acid-base, respiratory failure.
- **Neuroanatomy through Clinical Cases** — neuroanatomy, localization, neurologic exam.
- **Bradley and Daroff's Neurology in Clinical Practice** — neurologic disease, diagnostics, management.
- **The Power of Neurodiversity** — neurodiversity, accessibility, patient-centered language and design.
- **cdt-11-01-263.pdf** — review on heart failure pathophysiology and therapy (Schwinger, Cardiovasc Diagn Ther 2021).

Use these for clinical vocabulary and plausibility checks, not as data-science method sources.

## Quality gate before delivery

- [ ] File exists and has target line count ±10%.
- [ ] All template sections present or intentionally substituted.
- [ ] At least 3 real case studies with dates and sources.
- [ ] At least 3 labs with acceptance criteria.
- [ ] At least 8 interview questions with answer hints.
- [ ] Regulatory claims sourced and dated.
- [ ] No placeholder text such as "TBD", "lorem", or generic "etc." used to avoid detail.
- [ ] Cross-links to neighboring modules.
- [ ] Sources include URL and access date where web-based.
