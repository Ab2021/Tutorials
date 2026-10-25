# Documentation and Clinical Narratives

> Module 01 — Clinical Foundations for Data Scientists
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 05
> Version 2026.3

---

## 1. Why this topic exists

The clinical note is the richest and least structured record of what happened to a patient. It
contains the reasoning, the uncertainty, the social context, and the plan that no coded field
captures. It is also the most abused data source in clinical informatics, because it was never
written to be a dataset.

Four problems recur in industry.

**Problem 1: the note is written for several audiences at once.** It communicates with other
clinicians, documents for legal defence, supports billing, and satisfies quality measures. These
purposes pull in different directions. A note that is complete for billing may be vague for
research; a note that is candid for care may be hazardous for litigation.

**Problem 2: copy-forward turns the note into a palimpsest.** Text from prior notes is reused,
sometimes with edits, sometimes without. A diagnosis that was active three admissions ago can
appear in the current note as if it were current. A plan that was proposed and abandoned can
persist. The note's timestamp is the time of writing, not the time of the events it describes.

**Problem 3: templates create false completeness.** A template prompts the author to fill every
field. When a field is not applicable, the author may leave a default, write "normal," or copy
the prior value. The resulting note looks complete and is partly fictional.

**Problem 4: absence is ambiguous.** A note that does not mention chest pain may mean the patient
denied it, the clinician did not ask, or the finding was documented elsewhere. Natural language
processing (NLP) can recover what is written; it cannot recover what was never asked.

This topic covers note types, documentation purposes, copy-forward, templates, the structure of
narrative data, assertion and temporality, and the validation needed before a note-derived
phenotype is trusted.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** the major note types and the purpose each serves.
2. **Map** a note's sections to the clinical concepts they contain and the reliability of each.
3. **Detect** copy-forward and template boilerplate with reproducible methods.
4. **Classify** assertions as present, absent, uncertain, historical, or family-related.
5. **Build** a note-derived phenotype with explicit inclusion, exclusion, and assertion rules.
6. **Quantify** the effect of copy-forward on phenotype positive predictive value (PPV).
7. **Design** a documentation-aware data pipeline that respects time zero and privacy.

## 3. The core mental model

A note is a communication artifact with a timestamp, an author, a purpose, and a structure. The
clinical content is embedded in language that is shaped by all four.

```text
   CLINICAL EVENT
        |
        v
   AUTHOR (role, workload, template, habit)
        |
        v
   NOTE (type, section, text, timestamp, addendum)
        |
        +--> CARE (read by the team)
        +--> LEGAL (read in a dispute)
        +--> BILLING (read by a coder)
        +--> RESEARCH (read by an NLP pipeline)
```

The same sentence can mean different things to each audience. "Patient denies chest pain" is
reassuring to a clinician, a documented negative for a lawyer, a review-of-systems element for a
coder, and a negated concept for an NLP model.

### 3.1 The note as a layered record

```text
   Layer 1: structured fields (vitals, labs, problem list)
   Layer 2: templated sections (HPI, ROS, physical exam)
   Layer 3: free narrative (assessment, plan)
   Layer 4: metadata (author, time, type, addenda)
```

Most high-value reasoning is in layer 3. Most high-volume data are in layer 1. Layer 2 is where
templates create false completeness. Layer 4 is what makes the note interpretable.

### 3.2 Copy-forward as a temporal distortion

```text
   Admission 1: "HFrEF, EF 30%"        (true at the time)
   Admission 2: "History of HFrEF"      (historical)
   Admission 3: "HFrEF, EF 30%"         (copied forward, may be stale)
   Current echo: EF 58%                 (the physiology changed)
```

The note text is stable; the patient is not. A phenotype that reads the text without the
timestamp and the current evidence will classify the patient by the oldest statement in the
record.

### 3.3 The assertion model

A clinical concept in text has an assertion status:

| Status | Example | Analytic meaning |
|---|---|---|
| Present | "Patient has pneumonia" | Positive |
| Absent | "No evidence of pneumonia" | Negative |
| Uncertain | "Cannot exclude pneumonia" | Unknown or probabilistic |
| Historical | "History of pneumonia" | Not necessarily active |
| Family | "Mother had pneumonia" | Not the patient |
| Hypothetical | "If pneumonia develops, treat with..." | Not present |
| Conditional | "Pneumonia would explain the fever" | Reasoning, not finding |

A binary keyword search collapses all seven into "positive." That is the single most common
source of note-derived label noise.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Standard or unit |
|---|---|---|---|
| Clinical note | A narrative record of an encounter | Treated as a research annotation | FHIR DocumentReference |
| History and physical (H&P) | Initial comprehensive note | Treated as complete | Note type |
| Progress note | Daily or episodic update | Treated as a full history | Note type |
| Consultation note | Specialist opinion | Treated as a diagnosis | Note type |
| Operative note | Record of a procedure | Treated as a pathology report | Note type |
| Discharge summary | Summary at discharge | Treated as contemporaneous | Note type |
| Nursing note | Nursing documentation | Ignored in physician-focused NLP | Note type |
| Emergency department note | ED documentation | Treated as an admission note | Note type |
| Radiology report | Imaging interpretation | Treated as a diagnosis | Note type |
| Pathology report | Tissue diagnosis | Treated as a clinical note | Note type |
| SOAP | Subjective, objective, assessment, plan | Treated as a universal structure | Note structure |
| APSO | Assessment, plan, subjective, objective | Treated as equivalent to SOAP | Note structure |
| Problem-oriented record | Documentation organized by problem | Assumed to be used everywhere | Record structure |
| Copy-forward | Reuse of prior note text | Treated as efficiency only | Similarity measure |
| Copy-paste | Duplication within or across notes | Confused with copy-forward | Similarity measure |
| Template | Prefilled note structure | Treated as neutral | Note artifact |
| Smart phrase | Shortcut that expands to text | Treated as authored content | Note artifact |
| Boilerplate | Generic text repeated across notes | Treated as patient-specific | Note artifact |
| Note bloat | Excessive note length and redundancy | Treated as a quality problem only | Length, redundancy |
| Addendum | Later addition to a signed note | Treated as part of the original | Note metadata |
| Amendment | Revision of a signed note | Confused with addendum | Note metadata |
| Authorship | Who wrote the note | Assumed to be the attending | Note metadata |
| Attribution | Which clinician is responsible | Confused with authorship | Note metadata |
| Assertion | The status of a concept in text | Ignored in keyword search | NLP label |
| Negation | Explicit denial of a concept | Missed by keyword search | NLP label |
| Uncertainty | Possibility without confirmation | Treated as positive | NLP label |
| Temporality | Time reference of a concept | Ignored in phenotyping | NLP label |
| Family history | Condition in a relative | Attributed to the patient | NLP label |
| Coreference | Reference to a previously mentioned entity | Causes duplicate counting | NLP task |
| Section segmentation | Dividing a note into sections | Assumed trivial | NLP task |
| Named entity recognition | Identifying clinical entities | Confused with assertion classification | NLP task |
| Concept normalization | Mapping text to a code | Confused with extraction | NLP task |
| NegEx | A rule-based negation algorithm | Treated as a complete solution | Algorithm |
| ConText | An algorithm for negation, temporality, and experiencer | Treated as a complete solution | Algorithm |
| cTAKES | Clinical Text Analysis and Knowledge Extraction System | Treated as a universal pipeline | Tool |
| UMLS | Unified Medical Language System | Treated as a single vocabulary | Terminology |
| SNOMED CT | Clinical terminology | Treated as a note format | Terminology |
| Note-derived phenotype | A cohort definition using note text | Treated as validated | Algorithm |
| Information blocking | Interference with access to health information | Confused with privacy protection | Regulation |
| Open notes | Patient access to clinical notes | Treated as a research dataset | Policy |

Three definitions need expansion.

**Copy-forward is not the same as copy-paste.** Copy-paste duplicates text within a note or from
an external source. Copy-forward reuses text from a prior note by the same or a different author.
Both create staleness, but copy-forward is the more important source of temporal distortion
because it carries old clinical statements into a new encounter.

**A note's timestamp is not the event time.** A discharge summary written on day 5 describes
events from day 1. An addendum written on day 7 may describe a result from day 2. For time-zero
features, the note is available only when it was signed, and its content may refer to earlier or
later events.

**Open notes changed the audience.** Since the 21st Century Cures Act and the information
blocking rules, patients in the United States generally have access to their clinical notes.
That changes what clinicians write and how they write it. A note written for a patient audience
may be less candid about uncertainty or social context than a note written only for colleagues.

## 5. The mechanics, step by step

**Step 1. Identify the note type and its purpose.**

- Input: note metadata.
- Operation: classify as H&P, progress, consult, operative, discharge, nursing, ED, or other.
- Output: a note type with a purpose.
- Reasoning: each type has a different structure and reliability. A discharge summary is a
  synthesis written after the fact; a progress note is contemporaneous; an operative note
  describes a procedure in detail.
- Branch point: some notes are templates with minimal editing; the type does not guarantee
  content.
- Analytic artifact: `note_type`, `note_purpose`.

**Step 2. Capture authorship and timing.**

- Input: note metadata.
- Operation: record author role, signing time, entry time, and addenda.
- Output: a provenance record.
- Reasoning: an attending's note and an intern's note have different reliability and different
  language. A note signed at 23:00 may describe events from the morning. An addendum may contain
  the most important finding.
- Branch point: the author of a templated note may be the attending even if a trainee drafted it.
- Analytic artifact: `author_role`, `signed_ts`, `entered_ts`, `addendum_flag`.

**Step 3. Segment the note into sections.**

- Input: note text.
- Operation: identify headings (HPI, ROS, assessment, plan, medications, allergies).
- Output: section-labeled text.
- Reasoning: the same phrase means different things in different sections. "Chest pain" in the
  ROS may be a negated review; in the assessment it may be the working diagnosis.
- Branch point: headings vary by template and institution; a rule-based segmenter may fail on
  unfamiliar formats.
- Analytic artifact: `section`, `section_text`.

**Step 4. Separate structured and narrative content.**

- Input: the note and linked structured fields.
- Operation: identify which concepts are also available as coded data.
- Output: a crosswalk.
- Reasoning: a note may repeat a laboratory value that is already in the structured data. Using
  both can double-count. Using the note when the structured value exists adds noise.
- Branch point: some concepts exist only in the note (social history, functional status).
- Analytic artifact: `concept`, `structured_source`, `narrative_source`.

**Step 5. Detect copy-forward.**

- Input: current and prior notes.
- Operation: compute similarity; find long identical spans; compare with the prior note.
- Output: a copy-forward flag with the source note.
- Reasoning: a long unchanged span is evidence of reuse. The threshold is a trade-off: a low
  threshold flags legitimate standard language; a high threshold misses edited copies.
- Branch point: an edited copy may preserve the stale diagnosis while changing the surrounding
  text.
- Analytic artifact: `similarity_score`, `longest_match_chars`, `source_note_id`.

**Step 6. Detect template and boilerplate text.**

- Input: a corpus of notes.
- Operation: find sentences that appear across many notes by many authors.
- Output: a boilerplate dictionary.
- Reasoning: "Patient denies fever, chills, or night sweats" may appear in thousands of notes
  because it is a template, not because it was asked. A phenotype that treats it as a negative
  finding will misclassify.
- Branch point: boilerplate can be clinically appropriate when it reflects a real review; the
  data cannot distinguish.
- Analytic artifact: `boilerplate_flag`, `corpus_frequency`.

**Step 7. Detect negation.**

- Input: sentence with a clinical concept.
- Operation: apply a negation algorithm or a language model; check the scope.
- Output: a negated or affirmed assertion.
- Reasoning: negation scope is tricky. "No chest pain, but shortness of breath" negates chest
  pain and affirms dyspnea. "No evidence of pneumonia or effusion" negates both.
- Branch point: double negation ("cannot say there is no pneumonia") is rare but possible.
- Analytic artifact: `assertion`, `negation_scope`.

**Step 8. Detect uncertainty.**

- Input: sentence.
- Operation: identify hedges ("possible," "cannot exclude," "suspicious for," "likely").
- Output: an uncertain assertion with a direction.
- Reasoning: "likely pneumonia" and "cannot exclude pneumonia" have different probabilities. A
  binary label loses the difference. A three-level label (present, uncertain, absent) is better.
- Branch point: "rule out pneumonia" is a plan, not an assertion; it means the clinician is
  testing for it.
- Analytic artifact: `uncertainty`, `hedge_term`.

**Step 9. Detect temporality.**

- Input: sentence.
- Operation: identify past, present, future, and hypothetical references.
- Output: a temporal label.
- Reasoning: "had pneumonia last year" is historical; "has pneumonia" is current; "will treat
  pneumonia" is a plan. The tense and the time expression both matter.
- Branch point: notes often mix tenses within a paragraph.
- Analytic artifact: `temporality`, `time_expression`.

**Step 10. Detect family history and experiencer.**

- Input: sentence.
- Operation: determine whether the concept applies to the patient, a family member, or someone
  else.
- Output: an experiencer label.
- Reasoning: "mother had breast cancer" is a family history, not a patient diagnosis. A keyword
  search for breast cancer will misclassify it.
- Branch point: "family history of breast cancer" may be documented in a structured field as
  well; reconcile.
- Analytic artifact: `experiencer`.

**Step 11. Extract concepts.**

- Input: section-labeled text.
- Operation: named entity recognition for diseases, medications, symptoms, procedures, and
  measurements.
- Output: concept mentions with spans.
- Reasoning: extraction is not the same as classification. A mention is a string; an assertion
  is a status; a normalized concept is a code.
- Branch point: abbreviations are ambiguous ("MS" can be multiple sclerosis or mitral stenosis).
- Analytic artifact: `mention_text`, `start`, `end`, `concept_type`.

**Step 12. Normalize concepts to a terminology.**

- Input: mentions.
- Operation: map to SNOMED CT, RxNorm, LOINC, or UMLS.
- Output: coded concepts with confidence.
- Reasoning: normalization enables aggregation and comparison. It also introduces errors when
  the mapping is ambiguous.
- Branch point: a mention may map to multiple concepts; choose based on context or keep all.
- Analytic artifact: `concept_code`, `code_system`, `mapping_confidence`.

**Step 13. Assign assertion status to each mention.**

- Input: mention and context.
- Operation: combine negation, uncertainty, temporality, and experiencer.
- Output: a final assertion.
- Reasoning: the same mention can be present, absent, uncertain, historical, or family. The
  assertion is what a phenotype needs.
- Branch point: a mention in the assessment section is more likely to be the working diagnosis
  than a mention in the ROS.
- Analytic artifact: `final_assertion`.

**Step 14. Build the phenotype.**

- Input: assertions and structured data.
- Operation: define inclusion (present assertions), exclusion (absent or historical), and
  confirmation (structured evidence).
- Output: a cohort definition.
- Reasoning: a note-only phenotype is sensitive but noisy; a note-plus-structured phenotype is
  more specific. The choice depends on the decision.
- Branch point: requiring a structured confirmation may exclude patients whose diagnosis is
  documented only in the note.
- Analytic artifact: a phenotype specification.

**Step 15. Validate against chart review.**

- Input: a random sample.
- Operation: two clinicians adjudicate the phenotype; compute PPV, sensitivity, and agreement.
- Output: a validation report.
- Reasoning: note-derived phenotypes require validation more than code-based ones because the
  extraction error is larger and less well characterized.
- Branch point: if the adjudicators disagree, the phenotype definition is ambiguous.
- Analytic artifact: `ppv`, `sensitivity`, `kappa`, `error_categories`.

**Step 16. Handle missing notes.**

- Input: the note corpus.
- Operation: identify encounters with no note or with only a template.
- Output: a missingness flag.
- Reasoning: missing notes are not random. They are more common in short encounters, urgent
  situations, and certain services. A phenotype that requires a note will systematically exclude
  those encounters.
- Branch point: a note may exist but be inaccessible due to interface or archive issues.
- Analytic artifact: `note_missing`, `missing_reason`.

**Step 17. Respect time zero.**

- Input: note timestamps.
- Operation: use only notes signed at or before time zero; do not use content that refers to
  later events.
- Output: a time-zero-safe feature set.
- Reasoning: a discharge summary signed at discharge is not available at admission. A progress
  note signed on day 2 may describe a plan for day 3.
- Branch point: an addendum signed later may contain information that was known earlier; the
  signature time governs availability.
- Analytic artifact: `note_available_ts`.

**Step 18. Handle amendments and addenda.**

- Input: note metadata.
- Operation: link addenda to the original; use the latest signed version.
- Output: a versioned note record.
- Reasoning: an addendum may correct an error or add a critical result. Using the original
  without the addendum loses information; using both may duplicate.
- Branch point: some systems store addenda as separate documents with no link.
- Analytic artifact: `note_version`, `addendum_of`.

**Step 19. Protect privacy.**

- Input: note text.
- Operation: de-identify direct identifiers; consider the risk of indirect identification.
- Output: a de-identified corpus.
- Reasoning: notes contain names, dates, locations, and rare events. Removing 18 identifiers is
  not sufficient for all re-identification risks, especially with free text.
- Branch point: dates are often shifted; the shift must be consistent within a patient.
- Analytic artifact: `deid_method`, `date_shift_days`.

**Step 20. Document provenance.**

- Input: the pipeline.
- Operation: record the note source, extraction model, terminology version, and validation.
- Output: a reproducible phenotype package.
- Reasoning: note-derived phenotypes are model-dependent. Without provenance, they cannot be
  reproduced or audited.
- Analytic artifact: a phenotype package with code, model version, and validation.

### 5.1 Note types and reliability

| Note type | Best for | Weak for | Typical timestamp issue |
|---|---|---|---|
| H&P | Initial history, differential | Daily changes | Signed after admission |
| Progress note | Daily course, plan changes | Comprehensive history | Batch signing |
| Consult note | Specialist assessment | Longitudinal course | Signed after review |
| Operative note | Procedure details | Post-operative course | Dictated later |
| Discharge summary | Synthesis, follow-up plan | Real-time decisions | Signed at discharge |
| Nursing note | Observations, patient response | Diagnosis | Shift-based entries |
| ED note | Triage, initial workup | Inpatient course | Signed at transition |
| Radiology report | Imaging findings | Clinical context | Signed after read |

### 5.2 Assertion cues in clinical text

| Cue | Typical assertion | Caveat |
|---|---|---|
| "No evidence of" | Absent | Scope may extend across a list |
| "Denies" | Absent | May reflect the question asked, not the disease |
| "Cannot exclude" | Uncertain | Direction is toward disease |
| "Possible" | Uncertain | Probability is not specified |
| "History of" | Historical | May still be active |
| "Status post" | Historical | Procedure, not disease activity |
| "Mother had" | Family | Not the patient |
| "If" | Hypothetical | Not a finding |
| "Rule out" | Plan | Not a diagnosis |

The cues are not deterministic. "No evidence of" usually negates, but "no evidence of" can appear
inside a quoted prior report, where the negation may no longer apply. "History of" usually marks
a historical statement, but a clinician may use it loosely for an active chronic condition. A
rule-based system will make mistakes on both patterns, which is why validation against annotated
text is required.

The section also matters. A concept in the assessment is more likely to be the working diagnosis;
a concept in the review of systems is more likely to be a reported symptom; a concept in the plan
is more likely to be a future action. Section-aware extraction is more accurate than extraction
from the whole note.

## 6. Worked example

A 78-year-old woman is admitted with three weeks of progressive dyspnea and leg swelling. She has
a history of hypertension and chronic kidney disease stage 3. Two years ago, an echocardiogram
showed a left ventricular ejection fraction (LVEF) of 35%. A current echocardiogram shows an LVEF
of 58%, left ventricular hypertrophy, and grade II diastolic dysfunction.

### 6.1 The documentation

The discharge summary, signed on day 4, contains the following assessment:

> "78-year-old woman with a history of heart failure with reduced ejection fraction, ejection
> fraction 35 percent on echocardiogram two years ago. She is admitted with progressive dyspnea.
> Current echocardiogram shows ejection fraction 58 percent with diastolic dysfunction. Assessment:
> heart failure with preserved ejection fraction, likely contributing to dyspnea. Plan: diuresis,
> guideline-directed therapy for HFpEF, and outpatient follow-up."

The problem list contains only "Heart failure." The medication list contains furosemide,
metoprolol, lisinopril, and atorvastatin.

### 6.2 Step 1: what the note says versus what the patient has

The note contains two statements about heart failure:

| Statement | Status | Evidence |
|---|---|---|
| History of HFrEF, EF 35% two years ago | Historical | Prior echocardiogram |
| Current HFpEF, EF 58% | Present | Current echocardiogram |

The current physiology is HFpEF. The historical statement is true as history and false as a
description of the current state. A phenotype that extracts the phrase "heart failure with
reduced ejection fraction" without the temporal qualifier will label her as HFrEF.

### 6.3 Step 2: the extraction error

Assume an NLP pipeline extracts the first matching phrase and assigns the label HFrEF. The
pipeline is not lying; it is reading the sentence that appears first. The error is in the
assertion model, not in the string matching.

A chart review of 100 patients labelled HFrEF by this pipeline finds:

| Category | Count | Mechanism |
|---|---:|---|
| Confirmed HFrEF | 82 | Current EF 40% or less |
| Mislabeled, historical statement | 12 | Copied prior EF, current EF above 50% |
| Mislabeled, uncertain statement | 4 | "Cannot exclude HFrEF" read as positive |
| Mislabeled, other | 2 | Coding or extraction error |
| **Total** | **100** |  |

The PPV is 82 / 100 = 0.82, or 82%.

### 6.4 Step 3: the cohort impact

The health system's HFrEF registry contains 2,400 patients labelled by the pipeline. Assuming the
validation PPV applies:

```text
True HFrEF      = 2,400 x 0.82 = 1,968
Mislabeled      = 2,400 x 0.18 = 432
```

The mislabeled group is not random. It is enriched for older patients with prior cardiac
history, because those are the patients whose notes contain historical statements. That matters
if the registry is used to compare treatments or outcomes.

### 6.5 Step 4: the effect on a treatment comparison

Suppose the registry is used to compare beta-blocker prescribing between HFrEF and a control
group with HFpEF. Among true HFrEF patients, 85% receive a beta-blocker. Among the mislabeled
patients (who actually have HFpEF), 55% receive one, because beta-blockers are less strongly
indicated in HFpEF.

```text
Observed beta-blocker rate in the labelled HFrEF group
= (0.85 x 1,968 + 0.55 x 432) / 2,400
= (1,672.8 + 237.6) / 2,400
= 1,910.4 / 2,400
= 0.796, or 79.6%
```

If the control group's rate is 50%, the observed difference is 29.6 percentage points, while the
true difference among true HFrEF patients is 35.0 points. The misclassification attenuates the
association by 5.4 points, or about 15% of the true difference.

The attenuation is not catastrophic in this example. It becomes larger when the mislabeled
fraction is higher, when the treatment difference between true and mislabeled patients is larger,
or when the outcome is more strongly associated with the mislabeled group's true disease.

### 6.6 Step 5: the copy-forward evidence

The discharge summary's first sentence is nearly identical to a sentence in the prior admission's
summary. A character-level similarity check finds a longest common substring of 197 characters
and an overall similarity ratio of 0.821 between the old and new assessment paragraphs.

A copy-forward detector using a threshold of more than 850 unchanged sequential characters was
reported in a 2024 audit to detect erroneous copying with 0.88 sensitivity and 0.98 specificity.
Apply that performance to a note corpus with an 8% prevalence of erroneous copying:

```text
n = 1,000 notes
Diseased (erroneous copying) = 80
True positives = 80 x 0.88 = 70.4
False negatives = 9.6
Not diseased = 920
True negatives = 920 x 0.98 = 901.6
False positives = 18.4

PPV = 70.4 / (70.4 + 18.4) = 70.4 / 88.8 = 79.3%
NPV = 901.6 / (901.6 + 9.6) = 901.6 / 911.2 = 98.9%
```

The detector is useful for triage: a positive flag is right about 79% of the time. It is not a
substitute for review. The threshold also misses edited copies that preserve the stale diagnosis
while changing the surrounding text.

### 6.7 Step 6: the documentation quality review

A reviewer examining the discharge summary would note:

| Issue | Finding | Consequence |
|---|---|---|
| Historical statement first | HFrEF appears before HFpEF | Extraction bias toward HFrEF |
| Current evidence later | EF 58% in the same paragraph | Requires temporal reasoning |
| Problem list vague | "Heart failure" only | Cannot distinguish HFrEF from HFpEF |
| Medication list static | Metoprolol and lisinopril continued | May not reflect current plan |
| Copy-forward | 197-character match with prior note | Stale diagnosis persists |
| Template language | "Likely contributing to dyspnea" | Uncertainty not quantified |

None of these is a documentation error in the sense of a false statement. The note is clinically
defensible. The problem is that it was not written to be a database.

### 6.8 Step 7: the corrected phenotype

A corrected phenotype would:

1. Extract both statements.
2. Assign temporality to each ("two years ago" = historical; "current" = present).
3. Prefer the current echocardiogram value when available.
4. Require EF 40% or less for HFrEF, using the structured echo report if possible.
5. Exclude patients whose only HFrEF mention is historical.
6. Report the PPV and the fraction of patients whose classification depends on note text alone.

The corrected cohort would be smaller and more specific. That is the right trade-off for a
treatment comparison, where misclassification biases the estimate. For a screening tool, a more
sensitive definition may be preferable, with confirmation at the next step.

### 6.9 Step 8: the note-length context

The 2025 ACEP review of note bloat reported that notes grew 60% in length and 11% in redundancy
between 2009 and 2018, and that only 18% of note text was manually entered, 46% was copied, and
36% was imported. Those figures are from a review, not a universal audit. They describe a
direction: notes contain more text, more duplication, and less manual authorship than they did
before electronic records. A pipeline that treats note length as a proxy for clinical complexity
will confuse documentation behavior with disease.

## 7. Code, query, or calculation

The block computes similarity, assertion status, PPV, and the attenuation example.

```python
"""Note similarity, assertion classification, and misclassification impact."""

import difflib


def longest_common_substring_length(a: str, b: str) -> int:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return match.size


def similarity_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def assertion(text: str) -> str:
    """A deliberately simple rule set. Real systems need scope and context."""
    lowered = text.lower()
    if any(p in lowered for p in ("no evidence of", "denies", "without evidence")):
        return "absent"
    if any(p in lowered for p in ("cannot exclude", "possible", "suspicious for", "likely")):
        return "uncertain"
    if "history of" in lowered:
        return "historical"
    if "mother" in lowered or "father" in lowered or "family history" in lowered:
        return "family"
    return "present"


old = ("Assessment: 78-year-old woman with a history of heart failure with reduced "
       "ejection fraction, ejection fraction 35 percent on echocardiogram two years ago. "
       "She is admitted with progressive dyspnea.")
new = old + (" Current echocardiogram shows ejection fraction 58 percent with diastolic "
             "dysfunction.")

print(f"similarity ratio: {similarity_ratio(old, new):.3f}")
print(f"longest common substring: {longest_common_substring_length(old, new)} chars")

examples = [
    "No evidence of pneumonia.",
    "Cannot exclude pneumonia.",
    "History of pneumonia.",
    "Mother had pneumonia.",
    "Patient has pneumonia.",
]
print()
for sentence in examples:
    print(f"{repr(sentence)} -> {assertion(sentence)}")

# Copy-forward detector performance
prevalence = 0.08
sensitivity = 0.88
specificity = 0.98
n = 1000
tp = prevalence * n * sensitivity
fn = prevalence * n - tp
tn = (1 - prevalence) * n * specificity
fp = (1 - prevalence) * n - tn
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
print()
print(f"copy-forward PPV: {ppv * 100:.1f}%")
print(f"copy-forward NPV: {npv * 100:.1f}%")

# Misclassification attenuation
true_hfref = 1968
mislabeled = 432
rate_true = 0.85
rate_mislabeled = 0.55
observed = (rate_true * true_hfref + rate_mislabeled * mislabeled) / (true_hfref + mislabeled)
print()
print(f"observed beta-blocker rate: {observed * 100:.1f}%")
print(f"attenuation versus true rate: {(rate_true - observed) * 100:.1f} percentage points")

assert round(similarity_ratio(old, new), 3) == 0.821
assert longest_common_substring_length(old, new) == 197
assert round(ppv * 100, 1) == 79.3
assert round(observed * 100, 1) == 79.6
```

Expected output:

```text
similarity ratio: 0.821
longest common substring: 197 chars

'No evidence of pneumonia.' -> absent
'Cannot exclude pneumonia.' -> uncertain
'History of pneumonia.' -> historical
'Mother had pneumonia.' -> family
'Patient has pneumonia.' -> present

copy-forward PPV: 79.3%
copy-forward NPV: 98.9%

observed beta-blocker rate: 79.6%
attenuation versus true rate: 5.4 percentage points
```

The assertion function is intentionally crude. It fails on "no evidence of pneumonia or
effusion" if the negation scope is not handled, and it fails on "not without pneumonia." A
production system needs a scope-aware algorithm or a language model, plus validation against
annotated text. The point of the example is the architecture: extraction, assertion, temporality,
and experiencer are separate steps, and each can fail independently.

A SQL query for note availability at time zero:

```sql
-- Notes signed before the prediction time, excluding addenda after it.
SELECT n.encounter_id,
       n.note_id,
       n.note_type,
       n.signed_ts,
       n.author_role
FROM notes n
JOIN encounters e
  ON e.encounter_id = n.encounter_id
WHERE n.signed_ts <= e.prediction_ts
  AND NOT EXISTS (
      SELECT 1
      FROM notes a
      WHERE a.addendum_of = n.note_id
        AND a.signed_ts > e.prediction_ts
  );
```

The `NOT EXISTS` clause is a conservative choice: if a later addendum changes the note, the
original is not used at the earlier time. That is the correct default for a time-zero-safe
feature, even though it discards information that was physically available.

## 8. Data, standards, and regulatory context

**Document structure.** HL7 FHIR R4 represents documents with Composition (a structured
document with sections) and DocumentReference (a reference to a document). The Consolidated
Clinical Document Architecture (C-CDA) is the XML standard used for many clinical documents in
the United States. US Core profiles constrain FHIR resources for US interoperability. These
standards define structure, not content quality.

**Certification and interoperability.** The Office of the National Coordinator for Health
Information Technology (ONC) HTI-1 final rule, published 9 January 2024 and effective 8 February
2024, adopted United States Core Data for Interoperability (USCDI) version 3 as the baseline and
replaced the clinical decision support criterion with a Decision Support Interventions criterion
for predictive tools supplied by a developer. It also revised information blocking definitions
and added a TEFCA Manner Exception. These are binding requirements for certified health
information technology and for actors subject to the information blocking rule.

**Information blocking and open notes.** The 21st Century Cures Act of 2016 required the
Secretary of Health and Human Services to define information blocking and prohibited practices
that interfere with access to electronic health information. The ONC rules implementing the Act
support patient access to their notes. The practical effect is that notes are now written with
the awareness that patients may read them. That can improve clarity and can also reduce candor
about uncertainty, social factors, and differential diagnoses.

**Billing documentation.** In the United States, evaluation and management (E/M) coding was
revised in 2021 for office and outpatient visits, with changes to the history and examination
requirements and a greater emphasis on medical decision making or time. The documentation that
supports a code is not the same as the documentation that supports clinical care. A note written
to satisfy a code may emphasize elements that are irrelevant to the research question.

**Privacy.** The HIPAA Privacy Rule governs the use and disclosure of protected health
information. De-identification can use the Safe Harbor method, which removes 18 categories of
identifiers, or the expert determination method. Free-text notes are difficult to de-identify
completely because they contain indirect identifiers, rare events, and narrative details that
can re-identify a patient. Date shifting must be consistent within a patient to preserve
intervals.

**Clinical documentation integrity.** Clinical documentation improvement programs review records
and query clinicians for clarification. The queries are intended to improve accuracy and
completeness, and they also change the data. A record that was queried may contain more specific
language than a record that was not. The query process is a source of non-random variation across
sites.

**NLP tools.** NegEx and ConText are widely used rule-based algorithms for negation, temporality,
and experiencer. cTAKES is an open-source clinical NLP pipeline. These tools are aids, not
validated medical devices. A note-derived phenotype that influences clinical care may require
regulatory review depending on its intended use.

> **Note:** This section is educational and is not medical advice or legal advice. Information
> blocking, privacy, and documentation requirements depend on the entity, the data, and the
> jurisdiction.

## 9. Industry practice and case evidence

**Copy-forward is common and recognized as a risk.** A 2025 survey of 913 providers at a large
academic medical center reported that 82% used copy-paste and 52% used copy-forward; use was
highest on inpatient primary services at 91% and 68%, respectively. A majority believed the
practices improved efficiency, and 59-76% believed they worsened documentation errors. The
documented fact is the prevalence and the perception. The inference is that copy-forward is a
systemic workflow behavior, not an individual failing, and that interventions must change the
workflow rather than admonish users.

**Erroneous copying is measurable.** A 2024 audit of 128 patients' ward-round notes at an
Australian tertiary hospital found erroneous copying in 8.3% of notes, concentrated in the
issues list and mostly involving outdated antimicrobial plans. The study reported that a
threshold of more than 850 unchanged sequential characters predicted erroneous copying with 0.88
sensitivity and 0.98 specificity. These are single-center results. The method is transferable;
the threshold is not.

**Note bloat is a documented trend.** A 2025 review in *JACEP Open* reported that notes grew 60%
in length and 11% in redundancy between 2009 and 2018, and that only 18% of note text was
manually entered, 46% copied, and 36% imported. The review is a synthesis, not a primary
measurement at every site. Its value is the direction: more text does not mean more information.

**Ambient documentation changes the economics and not necessarily the accuracy.** A 2025
multicentre study of 263 ambulatory clinicians using an ambient AI scribe reported burnout
falling from 51.9% to 38.8% after 30 days, with reduced after-hours documentation. A separate
2025 evaluation of six AI scribes found that none produced error-free notes; common errors were
omissions, inappropriate generalization, and misplacement of content between SOAP sections. The
documented fact is that documentation burden can fall. The inference is that review remains
necessary and that note quality must be measured separately from time saved.

**Open notes changed documentation behavior.** The Cures Act and the information blocking rules
made notes more accessible to patients. Studies of open notes report both benefits (patient
engagement, error detection) and concerns (clinician candor, patient anxiety). The evidence is
mixed and setting-dependent. For data science, the relevant point is that the note's audience
changed, and with it the language.

**What is documented versus inferred.** The prevalence figures, the audit results, and the
burnout study are documented. The claim that a specific NLP pipeline is accurate for a specific
phenotype is not; it requires local validation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Keyword search without negation | "No pneumonia" labeled positive | Assertion ignored | False positives | Negation and scope handling |
| Historical statement read as current | Prevalence rises with prior visits | Copy-forward and history language | Overcount | Temporality classification |
| Family history attributed to patient | Family diagnoses in patient cohort | Experiencer ignored | Misclassification | Experiencer classification |
| Template boilerplate read as finding | Same sentence across thousands of notes | Template text | False negatives and positives | Boilerplate detection |
| Copy-forward contamination | Long identical spans | Reused text | Stale labels | Similarity detection |
| Discharge summary used at admission | Time-zero violation | Note signed later | Leakage | Timestamp filter |
| Note-only phenotype unvalidated | No chart review | Extraction error | Unknown PPV | Validation sample |
| Section context ignored | Assessment and ROS conflated | No segmentation | Wrong assertion | Section-aware extraction |
| Addendum ignored | Critical result missing | Version handling | Information loss | Link addenda |
| Duplicate counting | Same concept multiple mentions | Coreference | Inflated features | Deduplicate by concept |
| Abbreviation ambiguity | "MS" mapped to one concept | No context | Wrong code | Context-aware normalization |
| De-identification incomplete | Rare events in text | Indirect identifiers | Privacy breach | Expert determination; review |
| Note length as severity proxy | Long notes labeled complex | Documentation behavior | Confounded feature | Do not use length as severity |
| Open-note language shift | Uncertainty language changes over time | Audience change | Temporal drift | Monitor language drift |
| Model version undocumented | Cannot reproduce cohort | No provenance | Non-reproducible | Versioned pipeline |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The note describes the encounter | Phenotyping depends on it | Copy-forward, templates | Similarity audit |
| The timestamp reflects availability | Time-zero features depend on it | Late signing, addenda | Compare signed and event times |
| The author is the responsible clinician | Attribution depends on it | Trainee drafting, templates | Author-role review |
| The section structure is stable | Segmentation depends on it | Template changes | Section-header audit |
| Negation is explicit | Rule-based NLP depends on it | Implicit negation, double negation | Annotated validation |
| The terminology mapping is valid | Concept aggregation depends on it | Ambiguous abbreviations | Mapping review |
| The note corpus is complete | Cohort completeness | Missing notes, archive gaps | Note-missingness report |
| Documentation practice is stable | Longitudinal analysis | Template changes, open notes | Language drift analysis |
| The validation sample is representative | PPV estimate depends on it | Convenience sampling | Random or stratified sample |
| De-identification is sufficient | Privacy depends on it | Indirect identifiers | Re-identification risk assessment |

## 12. Exercises

**Exercise 1 — Note-type mapping (recall and application, 40 minutes).**
For five note types — H&P, progress note, consult, discharge summary, nursing note — list the
concepts each is most reliable for and the concepts each is least reliable for. Deliverable: a
five-row table. Acceptance criteria: at least two reliability claims per note type; the rationale
names the workflow. Time: 40 minutes.

**Exercise 2 — Assertion classification (application, 45 minutes).**
Classify ten sentences as present, absent, uncertain, historical, family, or hypothetical. Include
at least one double negative and one sentence with two concepts of different status. Deliverable:
a table with the classification and the cue. Acceptance criteria: all ten correct; scope
explained for the complex sentences. Time: 45 minutes.

**Exercise 3 — Copy-forward detection (calculation, 45 minutes).**
Given two note excerpts, compute the longest common substring and the similarity ratio. Then
apply the 850-character threshold and state whether the note would be flagged. Deliverable: a
calculation and a one-paragraph interpretation. Acceptance criteria: values correct; threshold
applied correctly; limitation stated. Time: 45 minutes.

**Exercise 4 — PPV and cohort impact (calculation, 50 minutes).**
A note-derived phenotype has 80% PPV and labels 5,000 patients. Compute the number of true and
false positives. If the false positives have a 40% outcome rate and the true positives have a
60% rate, compute the observed outcome rate and the bias relative to the true rate. Deliverable:
a calculation sheet. Acceptance criteria: arithmetic correct; bias direction explained. Time: 50
minutes.

**Exercise 5 — Template audit (application, 60 minutes).**
Using a corpus of 200 notes, identify the ten most frequent sentences and the number of distinct
authors. Propose a boilerplate list and explain how it would change a phenotype. Deliverable: a
table and a 300-word analysis. Acceptance criteria: frequency and authorship reported; at least
one phenotype impact stated. Time: 60 minutes.

**Exercise 6 — Time-zero audit (application, 45 minutes).**
For a sepsis alert that fires at hour 6, list the note types and timestamps that are available.
Identify three note-derived features that would leak. Deliverable: a table with availability and
leak verdict. Acceptance criteria: timestamps explicit; at least three leaks; mitigation stated.
Time: 45 minutes.

**Exercise 7 — Phenotype specification (design, 75 minutes).**
Write a specification for a note-derived HFpEF phenotype. Include sections used, assertion rules,
temporality rules, structured confirmation, and validation. Deliverable: a two-page
specification. Acceptance criteria: assertion and temporality explicit; validation plan includes
PPV; failure modes listed. Time: 75 minutes.

**Exercise 8 — Privacy review (design, 45 minutes).**
Review a de-identified note corpus for indirect identifiers. Propose three additional safeguards
beyond the 18 Safe Harbor identifiers. Deliverable: a risk memo. Acceptance criteria: risks are
specific to free text; safeguards are implementable; no claim of zero risk. Time: 45 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** Examples: "No evidence of pneumonia" = absent. "Cannot exclude
pneumonia" = uncertain. "History of pneumonia" = historical. "Mother had pneumonia" = family.
"Patient has pneumonia" = present. "If pneumonia develops, start antibiotics" = hypothetical.
"Not without pneumonia" is a double negative that implies pneumonia is present in context; a
simple rule will misclassify it. "No chest pain, but dyspnea" contains two concepts with
different status: chest pain absent, dyspnea present.

**Answer to Exercise 3.** The exact values depend on the excerpts. The method is: find the
longest common substring, divide by the length of the shorter text for a ratio, and compare the
longest match with the threshold. A match above 850 characters would be flagged under the cited
rule. The rule does not detect edited copies.

**Answer to Exercise 4.** True positives = 5,000 x 0.80 = 4,000. False positives = 1,000.
Observed outcome rate = (4,000 x 0.60 + 1,000 x 0.40) / 5,000 = (2,400 + 400) / 5,000 = 2,800 /
5,000 = 0.56, or 56%. The true rate among true cases is 60%, so the observed rate is biased
downward by 4 percentage points. The direction depends on the outcome rate in the false
positives; if it were higher than 60%, the bias would be upward.

**Answer to self-check on copy-forward.** Copy-forward is the reuse of text from a prior note.
It creates staleness because the text may describe a state that no longer exists. It is
detectable by similarity but not fully correctable, because the stale statement may be
indistinguishable from a current one without external evidence.

**Answer to self-check on note timestamps.** A note's signed timestamp is the time it became
available. The events it describes may be earlier. For time-zero features, use the signed time,
not the event time, and do not use a note that was signed after the prediction moment.

**Answer to self-check on open notes.** Open notes give patients access to their clinical notes.
They can improve engagement and error detection, and they can change what clinicians write. The
net effect on documentation quality is setting-dependent and not settled.

**Rubric for Exercise 7.** Full credit: sections specified; assertion and temporality rules
explicit; structured confirmation defined; validation with PPV; failure modes listed; versioning.
Half credit: phenotype logic present but validation generic. No credit: keyword list only.

**Rubric for Exercise 8.** Full credit: free-text-specific risks; three implementable safeguards;
no claim of zero risk; review process. Half credit: generic privacy statement. No credit: "we
removed the 18 identifiers, so it is safe."

## 14. Interview questions

**1. Why is a clinical note not a research annotation?**
A strong answer explains the multiple purposes, the audience, and the documentation workflow. A
weak answer says notes are unstructured.

**2. What is copy-forward, and how does it affect a phenotype?**
A strong answer explains reuse, staleness, and label noise, and proposes detection. A weak answer
says it is copying text.

**3. How do you handle negation in clinical NLP?**
A strong answer describes assertion classification with scope, and names NegEx or ConText as
examples. A weak answer says "look for 'no'."

**4. What is the difference between a mention and an assertion?**
A strong answer distinguishes the string from its status (present, absent, uncertain,
historical). A weak answer treats them as the same.

**5. Why is the discharge summary dangerous as a feature?**
A strong answer explains that it is signed after the encounter and contains synthesized
information, so it leaks future knowledge. A weak answer says it is long.

**6. How would you validate a note-derived phenotype?**
A strong answer proposes chart review, PPV and sensitivity, adjudication, and subgroup analysis.
A weak answer reports accuracy on a held-out set with the same labels.

**7. What is note bloat, and why does it matter?**
A strong answer explains length and redundancy growth and the confusion between documentation
and disease. A weak answer says notes are too long.

**8. How do templates create false completeness?**
A strong answer explains defaults, copied values, and the appearance of a full review. A weak
answer says templates are efficient.

**9. How do open notes affect documentation?**
A strong answer explains the audience change and the possible effects on candor and clarity. A
weak answer says patients can read notes.

**10. How would you de-identify free text?**
A strong answer goes beyond the 18 identifiers, discusses indirect identifiers and date shifting,
and proposes expert review. A weak answer says remove names and dates.

**11. What is the difference between authorship and attribution?**
A strong answer distinguishes who wrote the text from who is responsible for it. A weak answer
treats them as the same.

**12. How would you measure the effect of copy-forward on a model?**
A strong answer proposes a similarity flag as a feature or stratifier, and compares model
performance and calibration by copy-forward status. A weak answer says remove copied notes.

## 15. Key terms

- **Addendum** — later addition to a signed note.
- **Amendment** — revision of a signed note.
- **APSO** — Assessment, Plan, Subjective, Objective.
- **Assertion** — status of a concept in text.
- **Attribution** — responsibility for documentation.
- **Authorship** — who wrote the note.
- **Boilerplate** — generic repeated text.
- **C-CDA** — Consolidated Clinical Document Architecture.
- **CDI** — clinical documentation improvement.
- **Coreference** — reference to a previously mentioned entity.
- **Copy-forward** — reuse of prior note text.
- **Copy-paste** — duplication within or across notes.
- **cTAKES** — clinical Text Analysis and Knowledge Extraction System.
- **Discharge summary** — summary at discharge.
- **DocumentReference** — FHIR resource for a document.
- **Experiencer** — who the concept applies to.
- **Family history** — condition in a relative.
- **H&P** — history and physical.
- **Information blocking** — interference with access to health information.
- **NegEx** — rule-based negation algorithm.
- **Negation** — explicit denial.
- **Note bloat** — excessive length and redundancy.
- **Note-derived phenotype** — cohort definition using note text.
- **Open notes** — patient access to notes.
- **Problem-oriented record** — documentation organized by problem.
- **Progress note** — episodic update.
- **Section segmentation** — dividing a note into sections.
- **Smart phrase** — shortcut that expands to text.
- **SOAP** — Subjective, Objective, Assessment, Plan.
- **Template** — prefilled note structure.
- **Temporality** — time reference of a concept.
- **Uncertainty** — possibility without confirmation.
- **USCDI** — United States Core Data for Interoperability.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Regulation and standards**

- Office of the National Coordinator for Health Information Technology. HTI-1 Final Rule,
  published 9 January 2024. https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
- Health Level Seven International. FHIR R4 Composition and DocumentReference.
  https://hl7.org/fhir/R4/composition.html
- Health Level Seven International. C-CDA.
  https://www.hl7.org/implement/standards/product_brief.cfm?product_id=492
- Centers for Medicare and Medicaid Services. Evaluation and Management Services.
  https://www.cms.gov/medicare/payment/fee-schedules/physician/evaluation-management-services

**Evidence**

- Hudkins ML, et al. Exploring provider perceptions and attitudes toward copy-paste and
  copy-forward in clinical documentation. *Applied Clinical Informatics*, 2025.
  https://doi.org/10.1055/a-2574-1348
- Tsimiklis P, et al. Copying in medical documentation: developing an evidence-based approach.
  *Internal Medicine Journal*, 2024. https://doi.org/10.1111/imj.16590
- Genes N, et al. Addressing note bloat: solutions for effective clinical documentation.
  *JACEP Open*, 2025. https://doi.org/10.1016/j.acepjo.2024.100031
- Olson J, et al. Use of ambient AI scribes to reduce administrative burden and professional
  burnout. *JAMA Network Open*, 2025. https://doi.org/10.1001/jamanetworkopen.2025.34976
- Ha S, et al. Evaluating the usability, technical performance, and accuracy of artificial
  intelligence scribes for primary care. *JMIR Human Factors*, 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12309782/
- Waken RJ, et al. Longitudinal effects of ambient artificial intelligence scribe use on
  documentation burden and financial productivity in primary care. *Journal of the American
  Medical Informatics Association*, 2025. https://doi.org/10.1093/jamia/ocag151

**Background reading**

- Bickley LS. *Bates' Guide to Physical Examination and History Taking*, 13th edition.
- Chapman WW, et al. ConText: an algorithm for determining negation, experiencer, and temporal
  status from clinical reports. *Journal of Biomedical Informatics*, 2009.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_cardiology-and-ecg-basics.md`](07_cardiology-and-ecg-basics.md)
- Next topic: [`09_special-populations.md`](09_special-populations.md)
- Related topic: [`05_disease-taxonomy-and-coding.md`](05_disease-taxonomy-and-coding.md)
- Related topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Survey file: [`../../01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

