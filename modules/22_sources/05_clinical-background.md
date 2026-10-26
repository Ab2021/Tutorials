# Clinical Background

> Module 22 — Sources and Further Reading
> Topic 05 of 09
> Estimated study time: 3 hours
> Prerequisites: 01_how-to-read-this-list, 02_supplied-clinical-library
> Version 2026.3

## 1. Why this topic exists

Clinical background sources answer a question that data work asks constantly and rarely states: what does this word mean in this dataset. A feature called "sepsis" is not a clinical fact that arrives with the data. It is a label produced by a definition, applied by a coder or an algorithm, at a time, in a population. If the definition changed during the cohort period, the same column can contain two different measurements.

The failures are quiet and they survive validation:

- A cohort spanning 2013 to 2018 is labelled with a sepsis definition adopted in 2016, and the pre-2016 records are coded by a rule that no longer applies.
- A model uses "heart failure with preserved ejection fraction" as a category while the ejection fraction threshold in the guideline changed, so the category is not comparable across eras.
- A feature is built from a textbook definition that the local laboratory never used, so the values are plausible and wrong.
- A comorbidity score is computed from ICD-9 codes in one period and ICD-10 codes in another, with a crosswalk that is not one-to-one.
- A clinical term is imported from a guideline into a data dictionary without the version, and two teams implement different criteria under the same name.

The people affected are the analyst who cannot explain why the cohort changed size, the clinician who cannot reconcile the model with practice, the regulatory writer who cannot demonstrate that the intended use population matches the evidence, and the product manager who cannot explain why performance dropped after an upgrade.

Clinical background sources are also the category most often used without citation. A team will cite a paper for a model but not for the definition of the outcome, even though the definition determines the label and the label determines everything. The definition deserves the same citation discipline as the model.

This topic teaches how to treat a clinical term as a versioned object: find the definition, find its version and date, check how it was operationalised in the data, and record the mapping. It does not teach clinical medicine. It teaches the source work that makes clinical vocabulary usable in a dataset.

The distinction matters because clinical background is the one source category every data scientist uses and almost none cites. A model card will cite the training data and the architecture and will describe the outcome as "sepsis" without a definition. The definition is the part that determines what the model is.

A reader who masters this topic will notice definitional claims everywhere: in papers that compare cohorts across eras, in product descriptions that promise to detect a condition, and in regulatory submissions that describe an intended use population. The vocabulary is the same in all three, and so is the failure.

The failure is not usually dramatic. A cohort is slightly larger than it should be, a prevalence is a few points off, a model performs well in development and less well in a new year. None of those symptoms points at a definition, which is why the definition is the last thing anyone checks.

The procedure in section 5 is designed to make the definition the first thing checked, because it is the cheapest explanation to rule out and the most expensive to discover late.

> **Note:** A definition is not a fact about the patient. It is a rule for classifying patients, agreed by a body, at a time. Two definitions can both be clinically reasonable and produce different cohorts from the same records.

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a clinical term in a data dictionary to a definition, a version, and an issuing body. Assessed by Exercise 1 and self-check item 1.
2. **Distinguish** a consensus definition from a guideline recommendation, a coding rule, and a textbook description, and state which one a dataset uses. Assessed by Exercise 2.
3. **Detect** definition drift across a cohort period and quantify the overlap between eras. Assessed by Exercise 3 and section 7.
4. **Map** a clinical concept to a terminology or coding system and identify where the mapping is lossy. Assessed by Exercise 4.
5. **Compare** a textbook definition with the current consensus definition and explain any divergence. Assessed by Exercise 5.
6. **Evaluate** a claim that a cohort represents a clinical population, using the definition, the coding, and the era. Assessed by Exercise 6 and the worked example.
7. **Write** a data dictionary entry that names the definition, the version, the operational rule, and the limitations. Assessed by Exercise 7.
8. **Decide** when a clinical question needs a specialist rather than a source, and say so. Assessed by Exercise 8.

The outcomes are ordered from tracing to deciding. A learner who can trace a term but cannot decide when to escalate has the vocabulary without the judgement, and the judgement is what a review meeting tests.

Each outcome maps to an exercise or a self-check item, and each is observable in a document that a colleague could mark.

## 3. The core mental model

A clinical term is a pointer to a definition, and a definition is an object with a version, an issuer, and an operational rule. The dataset contains the operational rule's output, not the term.

```text
                    CLINICAL TERM
                "sepsis", "HFpEF", "AKI"
                              |
                              v
                    DEFINITION SOURCE
        consensus | guideline | classification | ontology | textbook
                              |
                              v
                    VERSION AND DATE
        which definition, issued when, effective when
                              |
                              v
                    OPERATIONAL RULE
        criteria | thresholds | codes | algorithm | abstraction
                              |
                              v
                    DATASET LABEL
        what the column actually contains
                              |
                              v
                    ERA AND POPULATION CHECK
        does the cohort span a change, and does the
        population match the intended use
                              |
                              v
                    DATA DICTIONARY ENTRY
        term | definition | version | rule | limits
```

Read the diagram downwards. The label is three steps removed from the term, and each step can introduce a difference: a different definition, a different version, or a different operational rule.

**Definitions are consensus objects, not natural kinds.** Sepsis, heart failure, and acute kidney injury are categories created by committees to support treatment and research. They change when evidence or practice changes, and the change is recorded in a publication with a date. The term stays the same, which is why the change is easy to miss.

**Operationalisation is where definitions meet data.** A definition says "suspected infection plus organ dysfunction"; a dataset says "blood culture ordered and SOFA increase of 2". The operational rule is a translation, and translations lose information. The loss is not a defect if it is documented; it is a defect when the column is named after the definition.

**Era is a property of the cohort.** A cohort that spans a definition change is two cohorts. The right response is to analyse era explicitly, restrict the cohort, or map the old records to the new definition with a documented crosswalk.

**Terminology is not definition.** ICD, SNOMED CT, and LOINC are coding systems. They carry concepts and sometimes definitions, and a code is not a clinical criterion. A code can change while the definition stays the same, and a definition can change while the code stays the same.

| Layer | Example | Who controls it | How it changes |
|---|---|---|---|
| Term | "Sepsis" | Clinical community | Slowly, by usage |
| Definition | Sepsis-3 criteria | Consensus body | By publication |
| Operational rule | Suspected infection plus SOFA rise | Analyst or registry | By local decision |
| Code | ICD-10-CM R65.20 | Coding authority | By annual update |
| Dataset column | `sepsis_flag` | Data team | By pipeline change |

The five layers are not a hierarchy of authority. A code can be more precise than a definition for retrieval, and a definition can be more meaningful than a code for a cohort. The point is that they are different objects, and a claim about one is not a claim about another.

A useful test is to ask what would have to change for the column to change. If the answer is "a new consensus statement", the definition layer is in play. If the answer is "an annual code release", the representation layer is in play. The two require different monitoring.

The model also explains why a clinical term can be correct in a note and wrong in a column. The note records what the clinician meant; the column records what the rule extracted. When the two diverge, the column is the one that enters the model.

## 4. Definitions and vocabulary

### 4.1 Sources of clinical definitions

| Source type | What it supplies | Authority | Typical use |
|---|---|---|---|
| Consensus statement | An agreed definition, often with criteria | High for the definition itself | Cohort definition |
| Clinical guideline | Recommendations plus definitions | High for practice | Outcome definition |
| Classification system | Categories for staging or grouping | High within its scope | Severity strata |
| Ontology | Concepts and relationships | High for structure | Concept mapping |
| Coding system | Codes and inclusion notes | High for billing and retrieval | Cohort extraction |
| Textbook | Explanation and description | Moderate | Vocabulary and mechanism |
| Registry definition | An operational definition used in a registry | Moderate to high | Benchmarks and comparisons |
| Local protocol | The institution's operational rule | Local | Data documentation |

### 4.2 Definition mechanics

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Criterion | A condition that must be met for classification. | Confused with a risk factor. | Boolean |
| Threshold | A numeric cut-point in a criterion. | Assumed stable across versions. | Clinical unit |
| Operational definition | The rule used to implement a criterion in data. | Replaced by the conceptual definition in documentation. | Algorithm |
| Case definition | The full rule for classifying a patient as a case. | Used interchangeably with "definition". | Structured rule |
| Adjudication | Expert review to resolve classification. | Omitted from the data dictionary. | Process |
| Definition drift | A change in the meaning of a term over time. | Treated as a coding issue. | Version comparison |
| Era | The period during which one definition applies. | Ignored in cohort construction. | Date range |
| Crosswalk | A mapping between coding systems or versions. | Assumed one-to-one. | Mapping table |
| Granularity | The level at which a concept is recorded. | Assumed uniform across sites. | Concept level |

### 4.3 Clinical concepts that change

| Concept | What changed | Why it matters for data |
|---|---|---|
| Sepsis | From systemic inflammatory response plus infection to organ dysfunction plus infection | Cohorts before and after are not comparable |
| Heart failure with preserved ejection fraction | Ejection fraction threshold and diagnostic criteria refined | Category membership shifts |
| Chronic obstructive pulmonary disease groups | Grouping changed from ABCD to ABE, then to a single exacerbation criterion | Severity strata are not comparable |
| Acute kidney injury | KDIGO staging replaced earlier RIFLE and AKIN criteria | Stage definitions differ |
| Delirium | Recognition and screening tools changed | Prevalence depends on ascertainment |
| Autism and neurodevelopmental categories | Diagnostic criteria revised across DSM editions | Case counts and language change |
| Obesity classes | Body mass index cut-points vary by guideline and population | Threshold-based features shift |
| Acute myocardial infarction | The universal definition has been revised several times | Troponin assay and criteria change together |
| Chronic kidney disease stages | Stage thresholds and the addition of albuminuria categories | Stage-based features shift |
| Hypertension | Guideline thresholds differ across societies and years | Prevalence depends on the threshold used |
| Autism spectrum disorder | DSM-5 consolidated categories in 2013 | Case counts and vocabulary changed |
| Long COVID | Definitions still evolving across bodies | Cohort definitions are unstable |

### 4.4 Coding and terminology systems

| System | Scope | Granularity | Common pitfall |
|---|---|---|---|
| ICD-10-CM | Diagnoses and procedures, US clinical modification | Code-level | Annual code changes |
| ICD-11 | International classification | Code-level, more granular | Limited adoption in some systems |
| SNOMED CT | Clinical concepts and relationships | Concept-level | Post-coordination complexity |
| LOINC | Laboratory and observation codes | Observation-level | Unit and method variants |
| RxNorm | Medications | Ingredient and product | Brand and generic mapping |
| HPO | Phenotypes | Term-level | Depth and specificity |
| UCUM | Units | Unit-level | Conversion errors |
| OMOP standard concepts | Harmonised concepts for analytics | Concept-level | Source-to-standard mapping loss |
| ICD-9-CM | Historical US diagnosis and procedure codes | Code-level | Crosswalk to ICD-10 is lossy |
| CPT | US procedure codes | Code-level | Annual updates |
| NDC | US drug product codes | Product-level | Repackaging and labeler changes |
| MedDRA | Adverse event terminology | Term-level | Version updates change counts |
| SNOMED CT to ICD-10 map | Reference mapping | Concept-to-code | Not a substitute for clinical review |

## 5. The mechanics, step by step

The procedure has 24 steps in five phases. It treats a clinical term as a versioned object and traces it from the data dictionary back to the definition and forward to the label.

The procedure is used whenever a clinical concept enters a dataset, a model, or a document. It is not necessary for every word in a note; it is necessary for every word that becomes a column, a cohort criterion, or a claim about a population.

### Phase A — Establish the term and the claim

**Step 1. Write the term and the sentence it supports.**
Input: the data dictionary entry, the model feature, or the draft sentence.
Output: the term, the sentence, and the population the sentence refers to.
Rationale: the term alone is ambiguous. "Sepsis" in a cohort definition, a feature, and a claim about mortality are three different uses.
Branch: if the term appears in a compound phrase, split it into its components.
Edge case: a term can be used correctly in one sentence and incorrectly in the next, so the check is per sentence.
Check: does the sentence name the population and the time period.
Record: the term, the sentence, and the population.

**Step 2. Classify the use of the term.**
Input: the sentence.
Output: one of five uses: definition, feature, outcome label, cohort criterion, or background explanation.
Rationale: the use determines how precise the definition needs to be and what evidence is required.
Branch: a background explanation can use a textbook; an outcome label needs the operational definition used to create it.
Edge case: a feature can be a proxy for a concept without being the concept, and the documentation should say so.
Check: is the use stated in the data dictionary.
Record: the use class.

**Step 3. Identify the definition source.**
Input: the term and the clinical domain.
Output: the consensus statement, guideline, classification, or coding rule that defines it.
Rationale: the definition source is the authority for the term, and it is usually not the paper that uses the term.
Branch: if a guideline and a consensus statement differ, use the one that governs the clinical action the model supports.
Edge case: a term can have a research definition and a billing definition, and they can differ.
Check: is the source cited in the data dictionary.
Record: the source and its type.

**Step 4. Fix the definition version and date.**
Input: the definition source.
Output: the version, the publication date, and the effective date if different.
Rationale: definitions change, and the version is what makes the term comparable across studies.
Branch: if the definition has been revised, record both the version used and the current version.
Edge case: a definition can be published in one year and adopted in practice in another.
Check: does the data dictionary name a version.
Record: version and dates.

### Phase B — Trace the operational rule

**Step 5. Identify the operational rule used to create the label.**
Input: the pipeline code, the registry definition, or the abstraction protocol.
Output: a written rule with criteria, thresholds, data sources, and time windows.
Rationale: the label is the output of a rule, and the rule is what the dataset actually contains.
Branch: if no rule is documented, reconstruct it from the code and record the reconstruction as an inference.
Edge case: a rule can differ by site within one dataset.
Check: can another analyst reproduce the label from the rule.
Record: the rule.

**Step 6. Map each criterion to a data field.**
Input: the rule and the available fields.
Output: a table of criterion, field, code, and coverage.
Rationale: a criterion that cannot be measured is a criterion that is silently dropped, which changes the effective definition.
Branch: if a criterion is approximated by a proxy, record the proxy and its likely direction of error.
Edge case: a criterion can be measured differently in different care settings.
Check: is every criterion represented in the data.
Record: the mapping table.

**Step 7. Check the coding system and its version.**
Input: the codes used in the rule.
Output: the coding system, version, and the date range of validity.
Rationale: codes change annually, and a code can be added, retired, or redefined.
Branch: if the cohort spans a coding change, record the crosswalk and its lossiness.
Edge case: a code can exist in one country's modification and not another's.
Check: is the code set versioned in the pipeline.
Record: the coding system and version.

**Step 8. Check the time windows.**
Input: the rule's windows for onset, measurement, and ascertainment.
Output: a statement of the windows and their effect on incidence.
Rationale: a definition with a 24-hour window and one with a 72-hour window produce different cohorts from the same records.
Branch: if the window is longer than the data capture, the label is incomplete.
Edge case: a window that crosses a shift or a weekend can change ascertainment.
Check: are the windows stated in the dictionary.
Record: the windows.

### Phase C — Check the era and the population

**Step 9. Identify the definition change dates that fall inside the cohort.**
Input: the cohort start and end, and the definition history.
Output: a list of change dates and the proportion of records on each side.
Rationale: a cohort that spans a change is two cohorts, and the label means different things on each side.
Branch: if the post-change share is between 20 and 80 percent, the cohort is mixed and needs an era variable or restriction.
Edge case: a change can be adopted at different times in different institutions.
Check: is the change date recorded in the cohort documentation.
Record: the change dates and the overlap.

**Step 10. Check the population against the definition's scope.**
Input: the definition's inclusion and exclusion criteria, and the cohort's.
Output: a comparison table.
Rationale: a definition developed in adults may not apply to children, and a definition developed in intensive care may not apply on a ward.
Branch: if the cohort includes a group outside the definition's scope, the label is an extrapolation.
Edge case: a definition can be applied to a broader population in practice than in the study that created it.
Check: is the cohort's eligibility documented against the definition.
Record: the comparison.

**Step 11. Check the setting and the ascertainment process.**
Input: the care setting and the data capture process.
Output: a statement of how the label is ascertained and what is missed.
Rationale: a definition applied in a setting with different testing or monitoring will find different cases.
Branch: if ascertainment changed during the cohort, the incidence trend is partly an artefact.
Edge case: a new assay or a new screening policy can change the label without any change in the definition.
Check: is the ascertainment process described.
Record: the process and its changes.

**Step 12. Check the unit of analysis and the timing.**
Input: the label's unit (patient, admission, day) and the index time.
Output: a statement of the unit and the timing rules.
Rationale: a patient-level label and an admission-level label answer different questions, and mixing them double-counts.
Branch: if the model predicts at a time before the label is ascertainable, the label leaks.
Edge case: recurrent events need an explicit rule for counting.
Check: is the unit stated.
Record: the unit and the index time.

### Phase D — Check the terminology mapping

**Step 13. Map the term to the coding system used in the data.**
Input: the term and the coding system.
Output: a concept or code set with inclusion and exclusion notes.
Rationale: the mapping is the bridge between the clinical term and the data, and it is where precision is lost.
Branch: if the mapping is many-to-one, record which distinctions are lost.
Edge case: a concept can be represented by a combination of codes rather than a single code.
Check: is the code set reviewed by a clinician or a coder.
Record: the code set and its notes.

**Step 14. Check the mapping's sensitivity and specificity.**
Input: a validation study or a manual review sample.
Output: an estimate of false positives and false negatives for the code set.
Rationale: a code set is a measurement instrument, and its error rates determine what the label means.
Branch: if no validation exists, state that the mapping is unvalidated.
Edge case: validation in one institution may not transfer to another.
Check: is there a validation estimate.
Record: the estimate or its absence.

**Step 15. Check for terminology version changes.**
Input: the terminology release history.
Output: a list of relevant changes and their dates.
Rationale: code sets are updated, and an update can change cohort size without a change in practice.
Branch: if the update is material, pin the terminology version in the pipeline.
Edge case: a code can be retired and its meaning redistributed across successors.
Check: is the terminology version pinned.
Record: the version.

**Step 16. Check the ontology relationships where relevant.**
Input: the ontology and the term.
Output: parent, child, and sibling concepts, and any mapping to standard concepts.
Rationale: ontology structure determines whether a query for a parent concept includes the children the clinician means.
Branch: if the query uses descendants, record the depth and the exclusions.
Edge case: a concept can have multiple parents in a polyhierarchy.
Check: is the query logic documented.
Record: the ontology path.

### Phase E — Record and review

**Step 17. Compare the textbook description with the current definition.**
Input: the textbook passage and the consensus definition.
Output: agreement, divergence, or a version gap.
Rationale: textbooks explain, guidelines define, and the two can differ in emphasis or in criteria.
Branch: if the textbook predates the definition, use it for mechanism only.
Edge case: a textbook can describe a definition that is still correct in its own context and not the one used in the dataset.
Check: is the definition source named in the dictionary.
Record: the comparison.

**Step 18. Check how local practice uses the term.**
Input: the local protocol, the order set, or a clinician.
Output: a statement of local usage and any divergence from the published definition.
Rationale: local practice is what the data reflect, and it can differ from both textbook and guideline.
Branch: if local practice diverges, the label reflects local practice and should be documented as such.
Edge case: a local protocol can be outdated relative to the guideline.
Check: is the local definition recorded.
Record: the local usage.

**Step 19. Write the data dictionary entry.**
Input: the outputs of the previous steps.
Output: a structured entry with term, definition, version, source, operational rule, coding, unit, era, and limitations.
Rationale: the entry is the artefact that makes the label reusable and auditable.
Branch: if any field is unknown, write "not established" rather than leaving it blank.
Edge case: a dictionary entry can be longer than the pipeline code that implements it, and that is appropriate.
Check: can a new analyst reproduce the label from the entry.
Record: the entry.

**Step 20. State the limitations in the model documentation.**
Input: the dictionary entry.
Output: a limitations paragraph naming the definition version, the coding, the era, and the validation status.
Rationale: a model inherits the limitations of its labels, and the documentation should say so.
Branch: if the label is unvalidated, the performance metrics are optimistic in an unknown direction.
Edge case: a limitation that is not material for one use can be material for another.
Check: does the model card name the label version.
Record: the limitations.

**Step 21. Check the cohort against the intended use population.**
Input: the model's intended use and the cohort's definition.
Output: a comparison and a statement of the gap.
Rationale: a model trained on a definition-specific cohort may not apply to a population defined differently.
Branch: if the gap is material, narrow the intended use or collect data under the intended definition.
Edge case: the intended use population can be defined by the same term with a different operational rule.
Check: is the intended use written in the same vocabulary as the cohort.
Record: the comparison.

**Step 22. Check for definition drift in the labels over time.**
Input: the label distribution by year.
Output: a trend with an assessment of whether it is clinical or artefactual.
Rationale: a sudden change in prevalence often marks a coding or definition change rather than a change in disease.
Branch: if the change coincides with a coding update, investigate before modelling.
Edge case: a genuine epidemic or policy change can also move prevalence.
Check: is there a year-by-year count.
Record: the trend and its interpretation.

**Step 23. Set the review trigger for the definition.**
Input: the definition source and the coding system.
Output: a review date or an event trigger.
Rationale: definitions and code sets change on their own schedules, and a model built on one version needs a trigger when the version changes.
Branch: the trigger should include a new guideline version, a coding update, and a change in local practice.
Edge case: a terminology update can change the cohort without changing the definition.
Check: is the trigger recorded with the model.
Record: the trigger.

**Step 24. Record the residual uncertainty.**
Input: the full trace.
Output: a sentence naming what is not known about the label.
Rationale: a label is a measurement with error, and the error should be stated rather than assumed away.
Branch: if the uncertainty is material to the claim, downgrade the claim rather than the label.
Edge case: an unvalidated mapping is not the same as a wrong mapping, and the difference should be stated.
Check: is the uncertainty visible in the model documentation.
Record: the uncertainty.

The procedure produces a dictionary entry that can be defended and a label that can be reused. It does not produce a clinically correct definition; that is the job of the body that issues definitions, and the procedure's job is to make sure the dataset uses one.

### Where definitions enter a model

| Entry point | What it affects | Failure if undocumented |
|---|---|---|
| Cohort inclusion | Who is in the training data | Population mismatch with intended use |
| Outcome label | What the model predicts | Optimistic or meaningless performance |
| Feature definition | What the model sees | Features that mean different things across eras |
| Subgroup definition | Fairness and subgroup reporting | Groups that are not comparable |
| Excluded population | Who the model was not trained on | Silent scope limit |
| Evaluation cohort | What the performance estimate means | Benchmark that does not transfer |

A definition problem at the outcome label is the most consequential, because it changes both the target and the measured performance. A definition problem at a feature is usually visible in the data dictionary; a definition problem at the outcome is often invisible until someone reads the label code.

### Worked branch: same term, two cohorts

A team builds a sepsis model on a cohort from 2013 to 2018. The label rule was written in 2019 and applies the current definition to the whole period. The pre-2016 records were created under an older definition and coded accordingly.

Two branches are possible, and they have different consequences.

If the team restricts the cohort to the post-change period, the label is consistent and the cohort is smaller. The model is then a model of the current definition, and the intended use must match it.

If the team keeps the full period, the label means two things, and the model can learn era-specific coding patterns rather than clinical signal. The fix is either an era covariate with an interaction, or a crosswalk that maps the old records to the new definition with documented loss, or exclusion of the ambiguous transition period.

The wrong branch is the one most commonly taken: keep the full period, apply one rule, and report a single performance number. The number is real, and it describes a mixture that does not correspond to any population the model will meet.

### Definition sources compared

| Source | Defines | Versioned | Operational detail | Use for a label |
|---|---|---|---|---|
| Consensus statement | The concept and criteria | Yes | Moderate | Yes, with adaptation |
| Guideline | Practice and often definition | Yes | Moderate | Yes, if it is the governing guideline |
| Classification manual | Categories and codes | Yes | Low | For staging, not for case finding |
| Registry definition | An operational rule | Yes | High | For comparability with the registry |
| Textbook | Explanation | By edition | Low | For vocabulary only |
| Local protocol | Local operational rule | Sometimes | High | For describing local data |

### What to record in the dictionary

- The term and the sentence it supports.
- The definition source, version, and date.
- The operational rule, with criteria and windows.
- The coding system and version.
- The era and the change dates inside the cohort.
- The validation status of the mapping.
- The local usage, if it differs.
- The review trigger.

### Where a definition can hide in a pipeline

| Pipeline stage | How a definition enters | How to check |
|---|---|---|
| Extraction | Code list or query | Review the code set |
| Transformation | Derived flags and thresholds | Read the transformation logic |
| Imputation | Default values for missing criteria | Document the default |
| Linkage | Combining records across systems | Check the join key and time window |
| Aggregation | Rolling windows or episode construction | State the window |
| Labelling | Final case definition | Validate against a reference |

A definition can enter at any stage, and the final label reflects all of them. Reviewing only the final query misses the thresholds introduced upstream.

### Signals that a label needs re-examination
- The prevalence changes sharply in one year.
- The cohort size changes after a software upgrade.
- A new coding system version is deployed.
- A guideline updates the definition.
- A new assay or screening policy changes ascertainment.
- A clinician says the cohort does not look like their patients.

The last signal is the most valuable and the least documented. A clinician's recognition that the cohort is wrong is often the first evidence that a definition or coding change has entered the data.

## 6. Worked example

A data dictionary entry reads: "Sepsis: suspected infection plus two or more systemic inflammatory response syndrome (SIRS) criteria." The cohort runs from 2013 to 2018 and contains 12,467 admissions. The model predicts 30-day mortality, and the team wants to describe it as a sepsis mortality model.

### Framing the claim

The dictionary entry contains a definition, and the model description contains a claim about what that definition measures. The two are not the same:

1. **Definition claim.** The column identifies patients with sepsis.
2. **Population claim.** The cohort is representative of patients with sepsis in the intended use setting.
3. **Comparability claim.** The cohort is comparable across the study period.

The first claim depends on which definition of sepsis is being used. The second depends on the setting and the eligibility criteria. The third depends on whether the definition changed between 2013 and 2018.

| Sub-claim | Type | What could support it |
|---|---|---|
| The column identifies sepsis | Definitional | The current consensus definition |
| The cohort is representative | Population | Comparison with the intended use population |
| The cohort is comparable over time | Era | A single definition applied throughout |

### Candidate sources

| # | Candidate | Type | Date | Role |
|---|---|---|---|---|
| C1 | Textbook chapter describing SIRS-based sepsis | Textbook | Older edition | The likely origin of the rule |
| C2 | Sepsis-2 consensus definitions | Consensus | 2003 | The definition the rule implements |
| C3 | Sepsis-3 consensus definitions | Consensus | 2016 | The current adult definition |
| C4 | Surviving Sepsis Campaign guideline | Guideline | 2021 | Current practice, including screening advice |
| C5 | Local coding rule based on ICD-10-CM | Local rule | Current | What the data actually contain |
| C6 | Paediatric sepsis criteria | Consensus | 2024 | Applies only if the cohort includes children |

### Scoring the candidates

**C1, the textbook.**
- Authority: moderate; a synthesis, not the defining body.
- Currency: depends on the edition, and older editions predate Sepsis-3.
- Applicability: explains the physiology well and does not define the current syndrome.
- Accessibility: library copy.
- Provenance: authored chapter.
- Verdict: background and mechanism only.

**C2, Sepsis-2.**
- Authority: high for the definition it states.
- Currency: superseded for adults by Sepsis-3, though it remains a historical reference.
- Applicability: the population in the original consensus.
- Accessibility: published.
- Provenance: consensus process described.
- Verdict: cite only as the historical definition, with the date.

**C3, Sepsis-3.**
- Authority: high; the current adult consensus definition.
- Currency: current at the review date, though later updates should be checked.
- Applicability: adults with suspected infection; the operationalisation uses SOFA change.
- Accessibility: open access.
- Provenance: derivation and validation described.
- Verdict: primary support for the definition claim.

**C4, the guideline.**
- Authority: high for practice.
- Currency: current, with a defined publication date.
- Applicability: adult sepsis management.
- Accessibility: open access.
- Provenance: GRADE-based process described.
- Verdict: primary support for screening and management, and for the advice against using qSOFA alone as a screening tool.

**C5, the local coding rule.**
- Authority: high for what the data contain, low for what the clinical term means.
- Currency: depends on the coding year.
- Applicability: the local population.
- Accessibility: internal.
- Provenance: the rule is documented but not validated against a clinical reference.
- Verdict: evidence about the data, not about the definition.

**C6, paediatric criteria.**
- Authority: high for children.
- Currency: current.
- Applicability: only if the cohort includes children.
- Accessibility: published.
- Provenance: consensus process described.
- Verdict: applicable only to a paediatric subgroup; not a substitute for the adult definition.

### The arithmetic of the overlap

The cohort runs from 2013 to 2018, and the adult definition changed in 2016. Using monthly admission counts:

- Records before the change: 5,790.
- Records after the change: 6,677.
- Total: 12,467.
- Post-change share: 53.6%.

Under the rule used in section 7, a post-change share between 20 and 80 percent is a mixed cohort. This cohort is almost exactly half and half, which is the worst case: neither era dominates, so an era effect cannot be dismissed as a small correction.

The 2016 year is itself split, with 522 pre-change and 1,648 post-change admissions, so the transition is visible within a single calendar year.

### Verdict

The dictionary entry describes Sepsis-2, and the model description claims sepsis. The two are different objects.

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| The column identifies sepsis | Reject as stated | "The column implements the 2003 SIRS-based definition, not the 2016 definition." |
| The cohort is comparable over time | Reject | "The cohort spans a definition change, with 53.6% of records after the change." |
| The model predicts mortality in SIRS-positive infection | Primary support | "The model predicts 30-day mortality among patients meeting SIRS plus suspected infection." |
| The model is a sepsis mortality model | Reject | Requires relabelling under the current definition |

The honest description is narrower and more useful: a model of mortality in SIRS-positive suspected infection, built on a mixed-era cohort, with the era effect unquantified. That description tells a reader exactly what the evidence supports.

### What changed between the definitions

| Element | Sepsis-2 (2003) | Sepsis-3 (2016) |
|---|---|---|
| Core criterion | SIRS plus suspected infection | Suspected infection plus organ dysfunction |
| Organ dysfunction | Severe sepsis as a separate category | SOFA increase of 2 or more |
| Screening tool | SIRS criteria | qSOFA as a prompt, not a definition |
| Categories | Sepsis, severe sepsis, septic shock | Sepsis and septic shock |
| Implication for cohorts | SIRS-positive patients are cases | SIRS-positive patients may not be cases |

The table is a summary of the published definitions, not a substitute for reading them. The exact wording of the criteria and the operational details in the consensus papers are what a cohort should cite.

The change is not a refinement of the same measurement. It replaces the criterion, so a SIRS-positive patient without organ dysfunction is a case under one definition and not under the other.

A further nuance matters for screening claims. Sepsis-3 introduced qSOFA as a bedside prompt, and the 2021 Surviving Sepsis Campaign guideline recommends against using qSOFA alone as a single screening tool. A model that claims to "implement Sepsis-3" by using qSOFA alone is implementing a screening prompt, not the definition.

### What changes in a mixed cohort

A cohort that loses records to a stricter definition is not a smaller version of the same cohort. It is a cohort of sicker patients, with a different case mix and a different baseline risk.

| Quantity | Effect of the definition change | Direction |
|---|---|---|
| Cohort size | Fewer patients meet the organ dysfunction criterion | Smaller |
| Prevalence | Depends on the population and the criterion | Usually lower |
| Mortality | Cases are sicker under the new definition | Usually higher |
| Feature distributions | Organ dysfunction variables become more important | Shifted |
| Coding | New codes and documentation patterns | Discontinuous |

The direction column is a hypothesis, not a guarantee. In a specific dataset, the observed effect depends on coding practice and case mix, which is why the overlap should be quantified rather than assumed.

- Cohort size usually falls under a stricter definition.
- Measured severity usually rises, because the milder cases drop out.
- Baseline mortality usually rises for the same reason.
- Feature importance can shift toward the organ dysfunction variables that define the new label.

A model's apparent improvement after a definition change can therefore be a case-mix effect rather than a modelling gain, and the two are indistinguishable without an era-stratified analysis.

### Questions for the clinician

- Which definition does your institution use in practice?
- When did the change reach your coding and documentation?
- Does the cohort look like the patients you see?
- Are there groups the definition handles poorly?
- Is the screening tool used in practice the same as the definition?
- Would a different definition change the clinical action?

A clinician can answer these in a short conversation, and the answers often reveal a local practice that is neither the textbook nor the guideline.

- A "yes, we use Sepsis-3" can mean the definition, the order set, or the coding rule.
- A "no, we still use SIRS" is a finding about the data, not a failure of the clinician.

The second answer is the one that changes a model, because it means the label reflects local practice rather than the consensus definition, and the documentation should say so.

## 7. Code, query, or calculation

The script quantifies era overlap. It assigns each admission to the pre- or post-change era and reports the share on each side, then applies a rule of thumb: a post-change share between 20 and 80 percent is a mixed cohort that needs an era term or a restriction.

```python
from datetime import date

CHANGE = date(2016, 4, 1)      # illustrative adoption date for a new definition
START = date(2013, 1, 1)
END = date(2018, 12, 31)

# Deterministic monthly admission counts (no randomness, reproducible).
MONTHLY = [120, 118, 131, 127, 140, 138, 129, 126, 134, 141, 137, 130,
           133, 136, 142, 145, 139, 148, 151, 147, 144, 150, 153, 149,
           155, 158, 161, 157, 163, 166, 159, 162, 168, 171, 167, 173,
           170, 174, 178, 181, 176, 183, 186, 180, 177, 184, 189, 192,
           188, 191, 195, 198, 193, 200, 204, 197, 201, 207, 210, 205,
           208, 212, 216, 219, 214, 221, 225, 218, 222, 228, 231, 226]


def month_iter(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def classify(y, m):
    return "post" if date(y, m, 15) >= CHANGE else "pre"


rows = [(y, classify(y, m), n) for (y, m), n in zip(month_iter(START, END), MONTHLY)]
years = sorted({r[0] for r in rows})
print(f"change date {CHANGE}   cohort {START} to {END}")
print("-" * 58)
print(f"{'year':>5s} {'pre':>6s} {'post':>6s} {'total':>6s} {'post share':>11s}")
print("-" * 58)
tot_pre = tot_post = 0
for y in years:
    pre = sum(n for yy, era, n in rows if yy == y and era == "pre")
    post = sum(n for yy, era, n in rows if yy == y and era == "post")
    tot_pre += pre
    tot_post += post
    share = post / (pre + post) if pre + post else 0
    print(f"{y:5d} {pre:6d} {post:6d} {pre + post:6d} {share:11.1%}")
print("-" * 58)
total = tot_pre + tot_post
print(f"pre-change {tot_pre:,}   post-change {tot_post:,}   total {total:,}")
print(f"post-change share {tot_post / total:.1%}")

if 0.2 <= tot_post / total <= 0.8:
    verdict = "MIXED: analyse era as a covariate or restrict the cohort"
elif tot_post / total < 0.2:
    verdict = "MOSTLY PRE: label the definition used, do not claim current practice"
else:
    verdict = "MOSTLY POST: check that early records are not mislabelled"
print("verdict:", verdict)
```

Expected output:

```text
change date 2016-04-01   cohort 2013-01-01 to 2018-12-31
----------------------------------------------------------
 year    pre   post  total  post share
----------------------------------------------------------
 2013   1571      0   1571        0.0%
 2014   1737      0   1737        0.0%
 2015   1960      0   1960        0.0%
 2016    522   1648   2170       75.9%
 2017      0   2389   2389      100.0%
 2018      0   2640   2640      100.0%
----------------------------------------------------------
pre-change 5,790   post-change 6,677   total 12,467
post-change share 53.6%
verdict: MIXED: analyse era as a covariate or restrict the cohort
```

Three readings matter. The 2016 row shows the transition inside a calendar year, which is why a year fixed effect is not sufficient; the split is within the year. The overall share of 53.6 percent is close to half, which is the case where neither era can be treated as the dominant one. And the verdict is a rule of thumb, not a statistical test; it tells you to look, not what to conclude.

To adapt the script, replace `CHANGE` with the adoption date in your institution, and replace the monthly counts with your own admission counts. If the change date is uncertain, run the script at the plausible earliest and latest dates and report the range; the overlap estimate is more honest as a range than as a point.

The script does not decide whether the cohort should be restricted. That decision depends on the intended use, the sample size after restriction, and whether the old definition is clinically meaningful for the question.

## 8. Data, standards, and regulatory context

Clinical definitions sit inside terminology and coding standards that have their own release cycles. The definition is the clinical concept; the code is the representation; the standard is the rule for the representation.

### 8.1 Terminology and coding standards

| Standard | Scope | Maintainer | Release cycle |
|---|---|---|---|
| ICD-10-CM | US diagnosis and procedure codes | Centers for Disease Control and Prevention and CMS | Annual, 1 October |
| ICD-11 | International classification | World Health Organization | Periodic revisions |
| SNOMED CT | Clinical concepts and relationships | SNOMED International | Biannual releases |
| LOINC | Laboratory and clinical observations | Regenstrief Institute | Frequent releases |
| RxNorm | Medications | US National Library of Medicine | Monthly |
| UCUM | Units of measure | Regenstrief and HL7 | Periodic |
| HPO | Phenotypes | Human Phenotype Ontology consortium | Periodic |
| CDISC | Clinical trial data standards | CDISC | Versioned |
| OMOP CDM | Observational data model | OHDSI | Versioned |
| USCDI | US core data for interoperability | ONC | Annual versions |

### 8.2 What is binding

In the United States, the Health Insurance Portability and Accountability Act (HIPAA) administrative simplification rules adopt standard code sets for electronic transactions, which makes some code systems mandatory for billing. In the European Union, the European Health Data Space regulation creates a framework for secondary use and interoperability, with implementing acts that will specify formats and standards over time.

Neither regime makes a clinical definition binding. A definition is a professional consensus, and a code set is a regulatory or contractual requirement. The two are connected by mapping, and the mapping is where most data problems live.

| Layer | Binding? | Who enforces | Typical failure |
|---|---|---|---|
| Clinical definition | No | Professional community | Version drift |
| Billing code set | Yes, for covered transactions | Payer and regulator | Code changes |
| Terminology standard | Contractual or regulatory | Contract and regulator | Mapping loss |
| Local data model | Contractual | Institution | Undocumented customisation |
| Registry definition | Contractual | Registry | Divergence from clinical definition |
| Terminology release | Contractual through the vendor | Vendor and contract | Silent code changes |
| Data use agreement | Contractual | Data owner | Restricted fields for a phenotype |
| Local coding policy | Institutional | Coding department | Local variation in case finding |
| Quality measure specification | Payer or regulator | Payer | Measure-specific definitions |
| Clinical trial protocol | Study sponsor | Sponsor and regulator | Definition fixed at protocol date |

### 8.3 What the standards do not constrain

- They do not decide which clinical definition is correct.
- They do not require a validation study for a code-based phenotype.
- They do not prevent a code from being used differently across institutions.
- They do not make a mapping one-to-one.
- They do not keep a cohort comparable across a release boundary.

The last point is the one that matters most for models. A terminology release can change the cohort without any change in clinical practice, and the change is invisible unless the release is recorded.

### 8.4 Documentation expectations

In a regulated context, the intended use population must be described precisely enough that a reviewer can tell who is covered. That description usually names the clinical condition in words, and the words inherit a definition. A submission that describes the population as "patients with sepsis" without naming the definition leaves the reviewer to guess which cohort the evidence supports.

### 8.5 From concept to code: a worked mapping

Consider the concept "acute kidney injury". The clinical definition is a rise in serum creatinine or a fall in urine output over a defined window. The data representation requires three decisions:

1. Which creatinine values are used, and from which time window.
2. Whether urine output is available and how it is measured.
3. Which codes are used to identify the diagnosis in claims.

Each decision changes the cohort. A creatinine-only definition will miss patients diagnosed by urine output. A claims-based definition will miss patients whose AKI was not coded. A definition that uses the first creatinine after admission will differ from one that uses a baseline.

| Mapping decision | What it determines | Failure if undocumented |
|---|---|---|
| Baseline definition | Which change is measured | Different incidence |
| Time window | Which events count | Under- or over-counting |
| Urine output inclusion | Whether oliguric AKI is captured | Systematic under-detection |
| Code set | Which coded cases are found | Billing-dependent cohort |
| Setting | Whether the cohort is ICU or hospital-wide | Population mismatch |

The mapping table is the artefact that makes the cohort reproducible, and it is more useful than a paragraph describing the concept.

## 9. Industry practice and case evidence

**Sepsis definitions and coding (documented).** Sepsis-3 was published in *JAMA* in 2016 and replaced the SIRS-based definition for adults. In the United States, ICD-10-CM codes for severe sepsis (R65.20 without shock, R65.21 with shock) were introduced with the ICD-10 transition in October 2015, before the consensus definition changed. The coding system and the clinical definition therefore moved on different schedules, which is exactly the kind of mismatch that produces a mixed cohort. The documentation and coding advice for these codes also changed over subsequent years.

**Chronic obstructive pulmonary disease grouping (documented).** The GOLD strategy reports changed the grouping of patients from the older ABCD scheme to ABE in the 2023 revision, and the 2026 report further adjusted the criteria so that a single moderate exacerbation places a patient in Group E. A dataset that records "GOLD group" without the year is a dataset with at least three possible meanings.

**Acute kidney injury staging (documented).** The KDIGO 2012 guideline defines AKI and its stages using serum creatinine and urine output criteria. Earlier RIFLE and AKIN criteria used different thresholds and windows. A cohort that spans the adoption of KDIGO contains at least two staging rules unless the data were re-derived.

**Diagnostic classification revisions (documented).** The Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition (DSM-5) was published in 2013 and revised as DSM-5-TR in 2022. Autism spectrum disorder replaced earlier separate diagnoses in DSM-5, which changed case definitions and counts. A cohort that spans the revision contains different categories under the same broad label.

**COVID-19 coding (documented).** The ICD-10-CM code U07.1 for COVID-19 was introduced in 2020. A model trained across the introduction of a new code learns a discontinuity that reflects coding, not epidemiology.

### Documented facts and inferences

| Statement | Status |
|---|---|
| Sepsis-3 published 2016 and replaced SIRS-based adult criteria | Documented in the consensus statement |
| ICD-10-CM severe sepsis codes introduced October 2015 | Documented in the code set |
| GOLD grouping changed to ABE and was revised in 2026 | Documented in the GOLD reports |
| KDIGO 2012 defines AKI staging | Documented in the guideline |
| DSM-5 changed autism categories in 2013 | Documented in the manual |
| Mixed-era cohorts are common in practice | Inference from data documentation practice |

### What mature teams do (inference)

- They record the definition version in the data dictionary, not only the code set.
- They plot cohort size and prevalence by year before modelling.
- They treat a coding change as a candidate explanation for any discontinuity.
- They ask a clinician to review the cohort definition.
- They pin terminology versions in the pipeline.

The practices are inexpensive and they prevent the most common form of silent cohort drift.

### A coding discontinuity in practice

A team plots cohort size by month and sees a step change in October of a year. The clinical leads see no change in practice. The data team finds that a new ICD-10-CM code set was deployed that month, and the phenotype's code list had not been updated.

Two responses are possible. The team can extend the code list to include the new codes and re-extract the cohort, which restores continuity. Or the team can restrict the cohort to the period after the code change, which produces a smaller but consistent cohort.

The wrong response is to keep the step and describe the trend as a change in disease. The step is a measurement artefact, and a model trained across it learns the artefact.

### How to tell a clinical change from a coding change

| Signal | Suggests a coding change | Suggests a clinical change |
|---|---|---|
| Timing | Coincides with a release date | Coincides with an outbreak or policy |
| Breadth | Affects one condition | Affects related conditions |
| Coding detail | New or retired codes | Same codes, different counts |
| Clinical data | Vitals and labs unchanged | Vitals or labs also change |
| Documentation | Note templates changed | Notes reflect sicker patients |

The decisive evidence is usually the combination of a coding release date and a stable clinical signal. When both are present, the coding explanation is more likely, and the cohort should be re-extracted or restricted.

### What to do when the cause is ambiguous

- Plot the monthly count and mark every release date.
- Compare the change with the clinical signal, such as vitals or laboratory values.
- Ask the coding department whether the code set changed.
- Check whether the phenotype's code list includes the new codes.
- Re-extract with an updated list and see whether the step disappears.
- If it does not, treat the change as clinical and investigate further.

The table is a heuristic. The decisive evidence is usually the combination of a coding release date and a stable clinical signal.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Superseded definition | Dictionary names a definition without a date | Definition revised | Label means something else | Record version and date |
| Mixed-era cohort | Prevalence shift in one year | Definition or coding change | Model learns era artefacts | Quantify overlap and adjust |
| Code-as-definition | Column named after the syndrome, coded by billing | Coding rule substituted for criteria | Cases do not meet the definition | Document the operational rule |
| Crosswalk assumed exact | ICD-9 to ICD-10 mapping undocumented | Codes are not one-to-one | Misclassification | Validate the crosswalk |
| Textbooks as definition | Chapter cited for a criterion | Textbook predates consensus | Outdated criterion | Cite the consensus statement |
| Terminology release change | Cohort size changes after an upgrade | Code set updated | Cohort discontinuity | Pin the terminology version |
| Unit mismatch | Lab values implausible | Conventional versus SI units | Wrong thresholds | Convert with UCUM |
| Proxy treated as concept | Feature named for the concept | Proxy measure documented loosely | Interpretive error | Rename the feature |
| Ascertainment change | Incidence rises without practice change | New test or screening policy | Trend is artefactual | Document the change |
| Paediatric criteria misapplied | Children in an adult cohort | Definition scope ignored | Wrong classification | Apply the paediatric definition |
| Local practice assumed | Cohort does not match clinician experience | Local rule differs | Population mismatch | Ask the clinician |
| Stage definition change | Severity mix shifts | Staging system revised | Strata not comparable | Record the staging version |
| Unvalidated phenotype | No positive predictive value | Code set never checked | Unknown error rate | Validate on a sample |
| Name asserts definition | Column called `sepsis` | Naming convention not reviewed | Every analyst inherits the error | Rename to the operational concept |
| Unit of analysis mixed | Counts differ between reports | Patient and admission rows combined | Double counting | State the unit |
| Coding release boundary | Step change in one month | Annual code update | Cohort discontinuity | Extend or restrict the code set |
| Phenotype applied outside scope | Children in an adult cohort | Definition scope ignored | Misclassification | Apply the correct definition |
| Validation from another site | Error rates assumed to transfer | Different coding practice | Unknown local error | Re-validate locally |
| Documentation lag | Dictionary describes an older rule | Pipeline changed without update | Analysts use the wrong rule | Version the dictionary |
| Code list not updated | New codes absent from the phenotype | Annual release | Under-counting after the release | Audit the code list |
| Site-specific coding | Same definition, different codes | Local coding practice | Pooled cohort is heterogeneous | Stratify or harmonise |
| Definition from a trial applied broadly | Eligibility criteria ignored | Trial population narrower | Cohort is not representative | Compare eligibility |
| Outcome and feature definitions conflict | Feature uses a different definition than the outcome | Separate teams wrote the logic | Leakage or mismatch | Review the dictionary together |

### A ten-minute label audit

- Does the data dictionary name the definition and its version?
- Does it name the operational rule?
- Does it name the coding system and version?
- Does the cohort span a known definition change?
- Is the post-change share documented?
- Has the phenotype been validated, and with what error rates?
- Does the local clinician recognise the cohort?
- Does the column name assert more than the rule implements?

A label that fails three or more of these checks should not be described with the clinical term in a model card or a publication.

- A label with no definition version is a label with no comparability.
- A label with no operational rule cannot be reproduced.
- A label with no coding version cannot survive an upgrade.
- A label with no validation has an unknown error rate.
- A label with no clinician review has no check against practice.

The audit is short because the questions are simple. The hard part is having the documents to answer them, which is why the dictionary entry is the real deliverable.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The definition source is the governing one | Comparability depends on it | A newer consensus exists | Check the current version |
| The operational rule implements the definition | The label is valid | Criteria approximated by proxies | Map each criterion |
| The coding system is stable over the cohort | Trends are interpretable | Annual code updates | Pin the version |
| The mapping is one-to-one | Crosswalks preserve meaning | Many-to-one mappings | Validate on a sample |
| The population matches the definition scope | The label applies | Children or another group included | Compare eligibility |
| Ascertainment is constant | Incidence is comparable | New test or policy | Document changes |
| Local practice follows the definition | The data reflect the concept | Local protocol differs | Ask the clinician |
| The validation transfers | Error rates apply | Different site or era | Re-validate locally |
| The term is used consistently across documents | The claim is coherent | Different definitions in different places | Audit the vocabulary |
| The clinician's judgement is available | Definition questions need expertise | No clinical input | Identify a reviewer |
| The dictionary matches the pipeline | Documentation is used | Pipeline changed first | Compare code and dictionary |
| The code set is complete for the period | Cohort is comparable | Codes added or retired | Audit the code list annually |
| The local coding practice is stable | Trends are interpretable | Coder training or policy change | Review coding audits |
| The validation sample is representative | Error rates transfer | Sampling from one unit | Stratify the sample |
| The code list is current for the period | The cohort is complete | New codes not added | Audit against the release |
| The local definition matches the published one | The label is comparable | Local adaptation | Ask the coding department |
| The outcome and feature definitions agree | The model is coherent | Different teams, different rules | Joint review |

Two limits are structural. First, a definition is a consensus object and can change again; the procedure manages versions but cannot make a definition permanent. Second, a code-based phenotype is a measurement with error, and the error is usually unknown unless it has been validated.

A third limit is scope. This topic covers definitional source work, not clinical reasoning. A reader who needs to know whether a cohort is clinically sensible needs a clinician, and the procedure is designed to make that conversation efficient.

A fourth limit is organisational. A data dictionary is only useful if it is maintained, and maintenance requires an owner. In organisations without an owner, the dictionary describes the pipeline as it was at the last rewrite, and analysts learn to distrust it.

## 12. Exercises

**Exercise 1 — Trace a term (recall, 20 minutes).**
Choose three clinical terms in a dataset you use and trace each to a definition, a version, and an issuing body.
Deliverable: a three-row table with term, definition source, version, and date.
Acceptance criteria: each term has a source and a version; at least one term has no documented version, which is recorded as a gap.
Time: 20 minutes.

**Exercise 2 — Classify the source (application, 25 minutes).**
For the same three terms, classify the source as consensus, guideline, classification, ontology, coding rule, or textbook, and state what each can support.
Deliverable: a table with source type and permitted use.
Acceptance criteria: the classification is correct for each source; the permitted use is stated in terms of claims, not genres.
Time: 25 minutes.

**Exercise 3 — Era overlap (calculation, 30 minutes).**
Using your cohort's admission counts, compute the share before and after a known definition change.
Deliverable: a year-by-year table and a verdict.
Acceptance criteria: the change date is sourced; the arithmetic is reproducible; the verdict states whether the cohort is mixed.
Time: 30 minutes.

**Exercise 4 — Mapping audit (application, 35 minutes).**
Take a code set used for a phenotype and identify where the mapping from concept to code is lossy.
Deliverable: a mapping table with concept, code, inclusion notes, and lost distinctions.
Acceptance criteria: at least one lossy mapping is identified; the direction of the likely misclassification is stated.
Time: 35 minutes.

**Exercise 5 — Textbook versus consensus (critique, 30 minutes).**
Compare a textbook passage with the current consensus definition and explain any divergence.
Deliverable: a two-column comparison with dates and a revised sentence.
Acceptance criteria: both sources are dated; the divergence is explained by publication timing or genuine disagreement; the revised sentence uses the current definition.
Time: 30 minutes.

**Exercise 6 — Cohort plausibility (critique, 35 minutes).**
Ask a clinician to review a cohort definition and record where their practice differs.
Deliverable: a short memo with the differences and their implications for the label.
Acceptance criteria: at least one difference is documented; the implication for the model is stated; the clinician's role is recorded.
Time: 35 minutes.

**Exercise 7 — Dictionary entry (design, 30 minutes).**
Write a data dictionary entry for one label using the fields in section 5.
Deliverable: a structured entry with term, definition, version, rule, coding, unit, era, validation, and limitations.
Acceptance criteria: every field is present or marked "not established"; a second analyst could reproduce the label; the limitations are specific.
Time: 30 minutes.

**Exercise 8 — Escalation decision (design, 25 minutes).**
For a definitional question in your project, decide whether it needs a source, a clinician, or a consensus body, and write the decision.
Deliverable: a one-page memo with the question, the source class, the reason, and the next step.
Acceptance criteria: the memo distinguishes a source question from a judgement question; it names the specific source or specialist; it states the consequence of getting it wrong.
Time: 25 minutes.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Trace | Source and version for each | "From the EHR" |
| 2 Classify | Permitted use stated | Genre listed only |
| 3 Overlap | Change date sourced | Arbitrary split |
| 4 Mapping | Lossy mapping identified | Assuming one-to-one |
| 5 Comparison | Both dates shown | Textbook treated as current |
| 6 Plausibility | Clinician difference recorded | No clinical input |
| 7 Dictionary | Reproducible entry | Rule in prose only |
| 8 Escalation | Names the specialist | "Ask someone" |

Two notes on marking. Exercise 6 requires a real clinician, and a submission without one is incomplete rather than failed; the point is to practise the conversation. Exercise 3 is arithmetic, and a submission that cannot be reproduced from the table has not met the criterion regardless of the conclusion.

The exercises are designed to be done on a live dataset. A learner who uses a public dataset will still find definitional gaps, because public datasets document less than their users assume.

## 13. Self-check answers

**1. What makes a clinical term a versioned object?**
It points to a definition issued by a body at a date, and the definition can be revised. The term stays the same while the definition changes, so the version is what makes two datasets comparable.

**2. What is the difference between a definition and an operational rule?**
A definition states the clinical concept and its criteria. An operational rule translates the criteria into data fields, codes, thresholds, and windows. The dataset contains the output of the rule, not the definition.

**3. Why is a mixed-era cohort a problem even if the definition change was an improvement?**
Because the label means different things on each side of the change. A model trained on the mixture can learn era-specific coding or practice patterns, and its performance describes a population that does not correspond to either era alone.

**4. What does a many-to-one code mapping lose?**
It loses the distinctions between the source concepts that map to the same target. Those distinctions may matter clinically, and the loss should be documented with the direction of likely misclassification.

**5. Why is a textbook not a definition source for a cohort?**
Because a textbook explains and summarises, often years before its publication, and it may describe a definition that has since been revised. The consensus statement or guideline is the definition source.

**6. What does it mean if a phenotype has no validation study?**
It means the error rate of the code set is unknown. The phenotype is not necessarily wrong, and the documentation should say that it is unvalidated rather than implying accuracy.

**7. Why can a new laboratory assay change a cohort without a definition change?**
Because the definition's criteria are operationalised through measurements. A more sensitive assay can classify more patients as cases, which changes prevalence and cohort composition even though the rule is unchanged.

**8. What should a model card say about the outcome definition?**
The definition source, version, operational rule, coding system, era, validation status, and the main limitation. A model card that names only the clinical term leaves the reader unable to tell what the model predicts.

**Rubric for Exercises 3 and 7.** A strong overlap analysis sources the change date, shows the arithmetic, and interprets the share. A weak analysis picks an arbitrary split. A strong dictionary entry is reproducible by another analyst and states its limitations. A weak entry describes the concept without the rule.

**9. Why is a clinical term not a natural kind?**
Because it is a category created by a committee to support treatment and research, and it changes when evidence or practice changes. The term is stable and the definition is not, which is why the version has to be recorded.

**10. What is the risk of using a billing code as a cohort definition?**
The cohort reflects billing practice as much as clinical practice, and it inherits the code set's changes and the institution's coding behaviour. It can be a valid operational definition and it should be labelled as a code-based phenotype rather than as the clinical concept.

**11. Why does a code-based phenotype need a validation study?**
Because the mapping from concept to code has unknown sensitivity and specificity, and the error rate determines what the cohort means. Without validation, the phenotype is a measurement with an unstated error.

**12. What should a model do when the definition changes mid-cohort?**
Either restrict the cohort to one era, include era as a covariate with an interaction, or crosswalk the old definition to the new one with documented loss. The choice should be stated and justified.

**13. Why can a new screening policy change incidence without changing the definition?**
Because the definition's criteria are applied to patients who are tested. More testing finds more cases, so the measured incidence rises even though the rule is unchanged.

**14. What is the difference between a concept and a code?**
A concept is a unit of meaning in a terminology; a code is an identifier in a classification. One concept can map to several codes, and one code can map to a concept with a narrower or broader meaning.

**15. Why is "we used the EHR definition" not a definition?**
Because the EHR contains many possible rules, and "the EHR definition" does not say which fields, codes, windows, or thresholds were used. The phrase describes a data source, not a rule.

**16. What would make you restrict a cohort to a single era?**
A definition change that cannot be crosswalked without material loss, a sample size that remains adequate after restriction, and an intended use that matches the later era. The restriction should be stated with its cost in sample size.

**17. Why can two hospitals with the same definition produce different cohorts?**
Because the operational rule, the coding practice, the testing policy, and the case mix differ. The definition is shared and the measurement is local.

**18. What is the first question to ask about a clinical feature?**
Whether it is the concept or a proxy for the concept, and if it is a proxy, what it measures instead.

**19. Why is a cohort definition a model artefact rather than a data artefact?**
Because it determines the target and the population, and both are part of what the model claims to do. Changing the definition changes the model, even if the architecture and features are identical.

**20. What would you do if two sources disagree about the definition and both are current?**
Record both, state which one the data implement, and explain the scope difference. The disagreement is often specialty-specific, and a model should not silently choose one.

## 14. Interview questions

**1. How do you know what a column called "sepsis" contains?**
Strong answer: traces it to the operational rule, the definition version, the coding, and the era, and checks whether it has been validated. Weak answer: assumes the name.

**2. What do you do if a cohort spans a definition change?**
Strong answer: quantifies the overlap, then restricts, stratifies, or crosswalks with documented loss, and describes the choice. Weak answer: ignores it or drops the earlier data without saying so.

**3. Why is a code not a definition?**
Strong answer: a code is a billing or classification artefact with its own rules and updates; a definition is a clinical consensus with criteria. A code set can approximate a definition and cannot replace it.

**4. How would you validate a code-based phenotype?**
Strong answer: samples records, compares against a clinical reference standard, estimates positive and negative predictive value, and reports the sampling frame. Weak answer: says to check a few charts.

**5. What is definition drift and why does it matter for a model?**
Strong answer: a change in a term's meaning over time, which makes labels non-comparable and can make a model learn era artefacts. Weak answer: calls it a coding issue.

**6. When would you use a textbook for a clinical concept?**
Strong answer: for mechanism, vocabulary, and examination detail, with the edition cited, and never as the definition for a cohort. Weak answer: whenever it is convenient.

**7. How do you handle a term that different specialties define differently?**
Strong answer: names the specialty and the definition used, and states the scope limit; does not average the definitions. Weak answer: picks the most familiar one.

**8. What should a data dictionary contain for a derived label?**
Strong answer: definition, version, rule, codes, windows, unit, era, validation, and limitations. Weak answer: a description and a source table.

**9. Why does the unit of analysis matter for a clinical label?**
Strong answer: patient-level, admission-level, and day-level labels answer different questions and produce different counts; mixing them double-counts. Weak answer: says the unit is the row.

**10. How do you keep a definition current?**
Strong answer: records the version, sets triggers for guideline and coding updates, and rechecks at each model version. Weak answer: reviews annually without a source.

**11. What is the difference between a definition and a diagnostic criterion?**
Strong answer: a definition states the concept and its boundaries; criteria are the operational conditions used to classify a patient. A definition can have several criteria sets, and a criterion can change while the definition stays.

**12. How would you explain definition drift to a product manager?**
Strong answer: the column name stayed the same while the rule changed, so the model was trained on two different measurements, and the fix is to version the label. Weak answer: describes it as a data quality issue.

**13. Why might a clinician disagree with a validated phenotype?**
Strong answer: the phenotype may capture a coding pattern that differs from clinical judgement, especially for conditions that are under-coded or over-coded; the disagreement is evidence about the phenotype. Weak answer: assumes the clinician is unfamiliar with the data.

**14. What should be recorded when a definition is unknown?**
Strong answer: "definition not established", the evidence used to reconstruct the rule, and the residual uncertainty; never a guess presented as documentation. Weak answer: leaves the field blank.

**15. How does a terminology release differ from a definition revision?**
Strong answer: a release updates the representation, while a revision changes the concept; either can change a cohort, and they have different owners and schedules. Weak answer: treats them as the same event.

**16. What is the risk of a phenotype validated at one institution?**
Strong answer: the error rates reflect that institution's coding and testing practice, and they may not transfer; local validation is needed. Weak answer: assumes the code set behaves the same everywhere.

**17. How would you present a cohort with an unavoidable era mix?**
Strong answer: reports the era distribution, includes era in the model or as a stratified analysis, and states the limitation in the results. Weak answer: reports a single pooled number.

**18. What does a clinician add that a data dictionary cannot?**
Strong answer: judgement about whether the operational rule captures the clinical concept, and knowledge of local practice that is not written down. Weak answer: says clinical review is a formality.

**19. Why is the intended use population a definitional question?**
Strong answer: the intended use names a condition, and the condition carries a definition; if the evidence cohort uses a different definition, the intended use is broader than the evidence. Weak answer: treats intended use as marketing language.

**20. What should trigger a re-validation of a phenotype?**
Strong answer: a coding release that changes the code set, a definition revision, a change in testing policy, or a new site. Weak answer: a fixed annual review.

**21. How would you document a definition that the team could not establish?**
Strong answer: records "not established", describes the reconstruction from code, states the uncertainty, and sets a task to confirm with the source owner. Weak answer: writes a plausible definition.

**22. Why might a model trained on a stricter definition generalise differently?**
Strong answer: the stricter cohort is sicker and more selected, so the case mix differs from the deployment population, and the model's baseline risk and feature importance may not transfer. Weak answer: assumes a better definition always improves the model.

## 15. Key terms

- **Consensus definition** — an agreed definition issued by a body.
- **Operational definition** — the rule that implements a concept in data.
- **Case definition** — the full rule for classifying a case.
- **Definition drift** — a change in meaning over time.
- **Era** — the period in which one definition applies.
- **Crosswalk** — a mapping between coding systems or versions.
- **Phenotype** — an operational definition of a clinical concept in data.
- **Positive predictive value** — the proportion of flagged cases that are true cases.
- **Adjudication** — expert review to resolve classification.
- **Ascertainment** — the process by which cases are detected.
- **SIRS** — systemic inflammatory response syndrome.
- **SOFA** — sequential organ failure assessment.
- **qSOFA** — quick SOFA, a bedside screening prompt.
- **Sepsis-2** — the 2003 SIRS-based consensus definition.
- **Sepsis-3** — the 2016 organ-dysfunction-based consensus definition.
- **HFpEF** — heart failure with preserved ejection fraction.
- **KDIGO** — the kidney disease guideline body and its AKI criteria.
- **GOLD** — the chronic obstructive pulmonary disease strategy report.
- **ICD-10-CM** — the US clinical modification of ICD-10.
- **SNOMED CT** — a clinical terminology with concepts and relationships.
- **LOINC** — a terminology for laboratory and clinical observations.
- **RxNorm** — a medication terminology.
- **UCUM** — a unit of measure standard.
- **HPO** — the Human Phenotype Ontology.
- **OMOP CDM** — a common data model for observational data.
- **USCDI** — US core data for interoperability.
- **Intended use population** — the population a device or model is intended for.
- **Spectrum bias** — performance varying with case mix.
- **Verification bias** — outcome ascertainment depending on the test result.
- **Proxy measure** — a field used in place of the concept.
- **Data dictionary** — the documentation of fields and their definitions.
- **Terminology release** — a versioned update of a coding system.
- **Concept** — a unit of meaning in a terminology.
- **Code** — an identifier in a classification.
- **Post-coordination** — combining concepts to express a compound meaning.
- **Pre-coordination** — a single concept for a compound meaning.
- **Value set** — a defined list of codes for a purpose.
- **Concept set** — a list of concepts used in an analysis.
- **Coding audit** — a review of coding accuracy against records.
- **Chart review** — manual abstraction used as a reference standard.
- **Reference standard** — the best available ascertainment of the true state.
- **Misclassification** — assigning the wrong category.
- **Differential misclassification** — error rates that differ between groups.
- **Non-differential misclassification** — error rates that do not differ between groups.
- **Positive predictive value of a phenotype** — the proportion of flagged records that meet the clinical reference.
- **Sensitivity of a phenotype** — the proportion of true cases that are flagged.
- **Specificity of a phenotype** — the proportion of non-cases that are not flagged.
- **Case mix** — the distribution of patient characteristics.
- **Era effect** — a difference attributable to the period rather than the patients.
- **Coding release** — a scheduled update to a code set.
- **Documentation template** — the structured form that shapes what is recorded.

## 16. Sources

All URLs were checked on 26 October 2026.

**Singer, M., Deutschman, C. S., Seymour, C. W., et al.** "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)." *JAMA* 2016;315(8):801-810. https://doi.org/10.1001/jama.2016.0287
Good for: the current adult definition, the SOFA criterion, and the rationale for replacing SIRS. Not for: paediatric definitions or local coding practice.

**Evans, L., Rhodes, A., Alhazzani, W., et al.** "Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021." *Critical Care Medicine* 2021;49(11):e1063-e1143. https://doi.org/10.1097/CCM.0000000000005337
Good for: current management recommendations and the advice against qSOFA alone as a screening tool. Not for: the definition itself, which is Sepsis-3.

**Schlapbach, L. J., et al.** "International Consensus Criteria for Pediatric Sepsis and Septic Shock." *JAMA* 2024. https://doi.org/10.1001/jama.2024.0179
Good for: the paediatric definition and its difference from the adult criteria. Not for: adult cohorts.

**Levy, M. M., Fink, M. P., Marshall, J. C., et al.** "2001 SCCM/ESICM/ACCP/ATS/SIS International Sepsis Definitions Conference." *Intensive Care Medicine* 2003;29:530-538.
Good for: the historical SIRS-based definition and the origin of the criteria still found in older code. Not for: current practice.

**KDIGO.** *Clinical Practice Guideline for Acute Kidney Injury*. Kidney International Supplements, 2012.
Good for: the AKI definition and staging criteria. Not for: local assay calibration.

**Global Initiative for Chronic Obstructive Lung Disease.** GOLD 2026 Report. https://goldcopd.org/
Good for: the current COPD grouping and the 2026 revision. Not for: historical groups without the year.

**Royal College of Physicians.** *National Early Warning Score (NEWS) 2*, 2017. https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/
Good for: the NEWS2 scoring system and its thresholds. Not for: sepsis definition; it is a deterioration score.

**World Health Organization.** ICD-11. https://icd.who.int/
Good for: the international classification and its concept structure. Not for: US billing codes, which are ICD-10-CM.

**SNOMED International.** SNOMED CT. https://www.snomed.org/
Good for: concept structure, relationships, and mapping. Not for: a cohort definition without a validation study.

**Regenstrief Institute.** LOINC. https://loinc.org/
Good for: observation codes and their attributes. Not for: clinical decision thresholds.

**U.S. National Library of Medicine.** RxNorm. https://www.nlm.nih.gov/research/umls/rxnorm/
Good for: medication concept normalisation. Not for: dosing recommendations.

**Observational Health Data Sciences and Informatics.** OMOP Common Data Model. https://ohdsi.github.io/CommonDataModel/
Good for: the standard structure for observational data and concept mapping. Not for: the clinical definitions themselves.

**U.S. Centers for Disease Control and Prevention and Centers for Medicare and Medicaid Services.** ICD-10-CM code set and guidelines. https://www.cdc.gov/nchs/icd/icd10cm.htm
Good for: the codes, their inclusion notes, and the annual update cycle. Not for: the clinical definition of a condition.

**U.S. National Library of Medicine.** Unified Medical Language System (UMLS). https://www.nlm.nih.gov/research/umls/
Good for: cross-terminology mapping and concept identifiers. Not for: a validated phenotype; the mapping is a starting point.

**World Health Organization.** International Classification of Diseases, 11th Revision. https://icd.who.int/
Good for: the international classification and its structure. Not for: national modifications or billing.

**Global Initiative for Chronic Obstructive Lung Disease.** 2026 report and key changes summary. https://goldcopd.org/
Good for: the current grouping and the changes from earlier reports. Not for: a data dictionary; the report must be read alongside local coding.

**KDIGO.** Acute kidney injury guideline. https://kdigo.org/guidelines/acute-kidney-injury/
Good for: the AKI definition and staging. Not for: local laboratory calibration.

**U.S. Food and Drug Administration.** Clinical decision support software guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
Good for: how the intended use of a clinical software function is described and assessed. Not for: the clinical definition of the condition it addresses.

**U.S. Centers for Disease Control and Prevention.** ICD-10-CM official guidelines for coding and reporting. https://www.cdc.gov/nchs/icd/icd10cm.htm
Good for: the coding conventions that determine how a diagnosis is recorded. Not for: the clinical criteria that define the diagnosis. The distinction is the core of this topic.

**NLM UMLS Metathesaurus.** https://www.nlm.nih.gov/research/umls/
Good for: cross-vocabulary concept identifiers and synonyms. Not for: a validated phenotype; the mapping is a starting point.

**Health Level Seven International (HL7).** Fast Healthcare Interoperability Resources (FHIR). https://hl7.org/fhir/
Good for: how clinical concepts are represented and exchanged. Not for: the definitions themselves; FHIR carries codes, not consensus criteria.

**Observational Health Data Sciences and Informatics.** Phenotype evaluation resources. https://ohdsi.org/
Good for: methods for validating phenotypes in observational data. Not for: a specific cohort definition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_methods.md`](04_methods.md)
- Next topic: [`06_datasets-and-benchmarks.md`](06_datasets-and-benchmarks.md)
- Supplied library: [`02_supplied-clinical-library.md`](02_supplied-clinical-library.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 01, clinical foundations: [`../01_clinical_foundations/`](../01_clinical_foundations/)
- Module 02, data landscape: [`../02_data_landscape/`](../02_data_landscape/)
- Module 08, EHR and claims: [`../08_ehr_claims/`](../08_ehr_claims/)
- Module 03, data engineering: [`../03_data_engineering/`](../03_data_engineering/)
- Module 12, omics definitions: [`../12_omics/`](../12_omics/)
- Module 18, quality: [`../18_quality/`](../18_quality/)
- Module 21, dataset selection: [`../21_datasets/01_five-gate-decision.md`](../21_datasets/01_five-gate-decision.md)
- Survey evaluation checklist: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md) section 10

The next topic moves from definitions to datasets, where the versioning problem is explicit and the citation has a DOI.

A reader who wants a single habit from this file should take this one: when a clinical term appears in a dataset, ask which definition the column implements, and write the answer down.

The habit is small, and it changes the questions a team asks about every cohort it builds.

A team that adopts it will stop describing a column by its name and start describing it by its rule, which is the difference between a data dictionary and a list of column names.

### Two diagnostics that would change the verdict

**Diagnostic 1: the cohort is paediatric.** The 2024 paediatric sepsis criteria differ from the adult criteria, and a paediatric cohort labelled with adult SIRS criteria would be mislabelled in a different way. If the cohort includes children, the relevant definition is the paediatric one, and the verdict changes from "wrong definition" to "wrong definition for a subset, with a separate paediatric question."

**Diagnostic 2: the research question is about SIRS-positive infection, not sepsis.** If the intended claim is about the operational phenotype rather than the clinical syndrome, the definition is appropriate and the label is correct. The verdict would change from "reject" to "accept with a renamed outcome", and the model description would need to match. This is not a loophole; it is a different study with a different meaning.

A third diagnostic, weaker, is a validated crosswalk that maps the pre-2016 records to the current definition with a measured error rate. It would allow a mixed cohort to be analysed with a stated correction, and it would change the verdict to "conditional".

### What the team should record

The dictionary entry is rewritten with the definition source (Sepsis-2), the version, the operational rule, the coding, the era, and the validation status. The model documentation is rewritten to describe the outcome as SIRS-positive infection, with the era limitation stated.

The review trigger is the next guideline update and any change in the local coding rule. The team also records that the column name `sepsis_flag` is misleading and should be renamed `sirs_infection_flag` at the next pipeline release, because a name that asserts a definition the data do not implement is a source error that will be rediscovered by every new analyst.

The renaming is not cosmetic. Column names are the documentation that analysts actually read, and a name that overstates the content propagates the error into every downstream analysis that trusts it.
