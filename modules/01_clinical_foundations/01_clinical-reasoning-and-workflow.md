# Clinical Reasoning and Workflow

> Module 01 — Clinical Foundations for Data Scientists
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: none
> Version 2026.3

---

## 1. Why this topic exists

Every clinical dataset is a residue of decisions. A laboratory result exists because someone
ordered it. An admission exists because someone decided the patient could not go home. A
diagnosis code exists because someone documented it and someone else coded it for payment. If
you model the residue without understanding the decision that produced it, you model the
decision-making process and call it biology.

This is not an abstract concern. It changes model performance, label validity, and whether a
product is used at all. Three concrete failures recur in industry:

1. **A model predicts the test, not the disease.** A sepsis alert trained on blood-culture
   orders learns which patients had a doctor worried about sepsis. It performs well in
   retrospective data because the order is in the record, then fails at the moment of
   prediction, when the order has not yet been placed.
2. **A feature encodes the future.** Discharge disposition, a note written at 18:00, or a
   consultant's final impression can leak into a feature vector timestamped at admission. The
   model looks excellent and is unusable.
3. **A cohort is assembled at the wrong time zero.** A readmission model that indexes at
   discharge includes treatments and diagnoses established during the admission, some of which
   are consequences of the outcome being predicted.

Clinical reasoning is the discipline that tells you which decision produced which data, at
which moment, with which uncertainty. You do not need to practice medicine to use it. You need
to know what the clinician was trying to decide, what information was available, and what
happened next.

The industry cost is concrete. A model rejected at clinical review costs the build plus the
review plus the delay. A model deployed with a leaked feature can cost a recall, a regulatory
conversation, and trust that is difficult to rebuild. A model that predicts a decision nobody
owns has no route to action regardless of its area under the receiver operating characteristic
curve (AUC).

This topic is the entry point for the module because every later topic — vital signs,
laboratory medicine, pharmacology, coding, imaging, electrocardiography, documentation,
special populations, plausibility — is a deeper look at one part of the loop described here.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the clinical encounter as an information flow with at least six data-producing
   steps and label where each common analytic variable enters.
2. **Distinguish** symptom, sign, test result, diagnosis, and outcome, and **select** the
   correct one for a stated decision.
3. **Reconstruct** a time zero for a stated prediction task and **list** the fields that would
   not be available at that moment.
4. **Compute** post-test probability from pretest probability, sensitivity, and specificity,
   and **explain** why the same test yields different predictive values in two settings.
5. **Critique** a proposed feature for workflow bias, documentation bias, or leakage.
6. **Map** a clinical narrative to electronic health record (EHR) fields and **classify** each
   missing field as unknown, not assessed, not applicable, or not documented.
7. **Write** a one-page decision memo that names the decision, the owner, the time zero, the
   label, and the failure mode that would make the model unsafe.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

The encounter is a loop, and the record is a shadow of the loop. The clinician moves from
uncertainty toward action; the record captures selected fragments of that movement, often after
the fact and often for a different purpose.

```text
                        PATIENT STATE
        biology | behavior | environment | care process
                             |
                             v
   +-----------------------------------------------------+
   |  ENCOUNTER                                          |
   |                                                     |
   |  1 History  ->  2 Exam  ->  3 Differential          |
   |       ^                          |                  |
   |       |                          v                  |
   |  8 Reassess  <-  7 Act  <-  6 Update <- 5 Test      |
   |       |                     ^          ^            |
   |       |                     |          |            |
   |       |              4 Pretest prob.   |            |
   |       |                                |            |
   |       +-------- new information --------+            |
   +-----------------------------------------------------+
                             |
                             v
   +-----------------------------------------------------+
   |  RECORD                                             |
   |  order -> specimen -> result -> note -> code -> bill |
   |  each step has its own timestamp and its own purpose |
   +-----------------------------------------------------+
                             |
                             v
   +-----------------------------------------------------+
   |  ANALYTIC DATASET                                   |
   |  cohort | index date | features | label | split      |
   +-----------------------------------------------------+
```

Three properties of this diagram matter more than the boxes.

**Property 1: The loop is sequential, but the record is not.** A note written at discharge
summarizes events from admission. A problem list may contain a diagnosis that was considered
and excluded. A medication list may contain a drug stopped three days ago. When you flatten the
record into a table, you lose the order in which belief changed.

**Property 2: Every data point has a purpose that is not your purpose.** The blood culture was
ordered to guide antibiotics, not to label sepsis. The code was assigned to support payment, not
to define a phenotype. The note was written to communicate and to defend. Purpose shapes
content, and content shapes your label.

**Property 3: The decision point is not the data point.** The clinician decides at 03:40 with
the information available at 03:40. Your model must be evaluated at that same boundary. If it
uses the 05:00 result, it is not solving the clinician's problem.

### 3.1 Dual-process reasoning and why it matters to data

Clinicians reason in two modes. System 1 is fast, pattern-based, and driven by illness scripts:
"middle-aged smoker, crushing chest pressure, diaphoresis — this is acute coronary syndrome."
System 2 is slow, analytic, and deliberate: "the troponin is borderline; what is the pretest
probability and what does a negative result exclude?"

Most routine decisions use System 1. System 2 engages when the pattern is incomplete, the
stakes are high, or the patient does not fit the script. This matters analytically because

- System 1 generates the orders that become your features, and it is sensitive to context
  (time of day, bed availability, clinician experience).
- System 2 generates the documentation that explains the reasoning, and it is sensitive to
  workload (it is skipped when the department is busy).
- Neither mode records its own uncertainty in a structured field.

A model that predicts a System 1 decision will learn clinician habit. A model that supports a
System 2 decision must supply information at the moment of deliberation, not a verdict after
the fact.

### 3.2 The probability-updating loop

The quantitative core is simple and worth memorizing:

```text
pretest odds      = pretest probability / (1 - pretest probability)
post-test odds    = pretest odds x likelihood ratio
post-test prob.   = post-test odds / (1 + post-test odds)
```

The likelihood ratio (LR) for a positive result is sensitivity / (1 - specificity). The LR for
a negative result is (1 - sensitivity) / specificity. The test does not create certainty; it
moves odds by a fixed factor. The same factor applied to a low pretest probability produces a
low post-test probability, which is why a positive screening test in a healthy population can
be wrong more often than right.

### 3.3 Where the data scientist enters

| Entry point | Typical task | Correct time zero | Common error |
|---|---|---|---|
| Triage | Deterioration, disposition | Arrival | Using admission labs |
| Admission | Length of stay, mortality | Admission order | Using discharge codes |
| During stay | Alerting, treatment response | Last observation | Using future observations |
| Discharge | Readmission, transition risk | Discharge order | Using post-discharge claims |
| Outpatient | Diagnosis, adherence | Visit start | Using results returned later |
| Population | Risk stratification, outreach | Enrollment | Using future utilization |

The table is the single most useful artifact in this topic. If you cannot fill in the third
column for your task, you are not ready to build.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Symptom | What the patient reports, e.g., chest pressure | Treated as an objective finding | None; free text or structured questionnaire |
| Sign | What an observer detects, e.g., diaphoresis | Confused with a symptom | None |
| Vital sign | A measured physiological variable, e.g., systolic blood pressure | Treated as error-free | mmHg, beats/min, breaths/min, °C, % |
| Laboratory result | A measurement on a specimen, with assay and units | Treated as diagnosis | Assay-specific units |
| Differential diagnosis | Ranked list of plausible explanations | Treated as a final diagnosis | Free text |
| Diagnosis | A named condition accepted after assessment | Equated with a code | ICD-10-CM, SNOMED CT |
| Pretest probability | Probability of disease before a test | Ignored; test treated as definitive | Proportion 0-1 |
| Sensitivity | TP / (TP + FN) | Confused with PPV | Proportion 0-1 |
| Specificity | TN / (TN + FP) | Confused with NPV | Proportion 0-1 |
| Positive predictive value (PPV) | TP / (TP + FP); depends on prevalence | Reported as if test-intrinsic | Proportion 0-1 |
| Negative predictive value (NPV) | TN / (TN + FN); depends on prevalence | Reported as if test-intrinsic | Proportion 0-1 |
| Likelihood ratio (LR) | How much a result multiplies the odds | Treated as a probability | Dimensionless |
| Time zero | The moment at which prediction is made and features must be available | Set to a convenient table date | Timestamp |
| Index date | The date used to anchor cohort entry | Confused with time zero | Date |
| Encounter | A single interaction with the health system | Equated with an admission | FHIR Encounter |
| Admission | Entry to inpatient care | Equated with an encounter | Timestamp |
| Discharge | Exit from inpatient care | Treated as recovery | Timestamp and disposition |
| Order | A clinician's request for a test or treatment | Treated as delivery | FHIR ServiceRequest / MedicationRequest |
| Result | The output of a test | Treated as an order | FHIR Observation / DiagnosticReport |
| Documentation | The clinical note | Treated as ground truth | FHIR DocumentReference / Composition |
| Coding | Assignment of a classification code for billing or statistics | Treated as clinical truth | ICD-10-CM, CPT, MS-DRG |
| Phenotype | A computable definition of a clinical condition | Treated as a diagnosis | Algorithm plus code set |
| Label | The target variable in a model | Treated as biology | Any |
| Label leakage | Information about the label present in features but unavailable at time zero | Missed because AUC is high | None |
| Confounding by indication | Treatment is chosen because of prognosis | Interpreted as treatment effect | None |
| Illness script | A clinician's learned pattern for a presentation | Treated as bias-free | None |
| System 1 / System 2 | Fast pattern matching / slow analytic reasoning | Treated as a personality trait | None |
| Automation bias | Over-reliance on an automated output | Treated as a user training problem | None |
| Alert fatigue | Desensitization after repeated low-value alerts | Treated as a compliance problem | None |
| Base-rate neglect | Ignoring prevalence when interpreting a result | Produces inflated PPV claims | None |
| Anchor | An initial diagnosis that resists revision | Produces diagnostic momentum | None |
| Diagnostic momentum | Continued acceptance of a prior diagnosis | Produces stale labels | None |
| Copy-forward | Reusing prior note text in a new note | Treated as documentation efficiency only | Character-level similarity |
| Problem list | Structured list of active and historical conditions | Treated as complete and current | SNOMED CT or ICD-10-CM |
| Chief complaint | The patient's reason for presenting, in their words | Overwritten by triage shorthand | Free text |
| History of present illness (HPI) | Narrative of onset, course, and context | Compressed to a template | Free text |
| Review of systems (ROS) | Structured symptom inventory | Copied forward without re-asking | Structured or free text |
| Comorbidity | A coexisting condition | Counted without severity | ICD codes, Elixhauser, Charlson |
| Care pathway | A standardized sequence of care for a condition | Assumed to be followed | None |
| Throughput | Patient flow through a care setting | Treated as unrelated to data quality | Minutes or hours |
| Turnaround time (TAT) | Time from order or collection to result | Measured from the wrong start point | Minutes |
| Data provenance | Origin and processing history of a data element | Omitted from feature dictionaries | Metadata |

Two definitions deserve expansion because they are misused constantly.

**Time zero is not the same as the index date.** The index date is the calendar anchor for
cohort construction, often the admission date. Time zero is the precise moment at which a
prediction would be made in live care. For an admission mortality model, the index date may be
admission day and time zero may be 24 hours after admission. Features must be computed from
data with a timestamp at or before time zero. A model that uses the index date but pulls all
admission data has a time-zero violation even though the cohort looks correct.

**A code is not a diagnosis.** A diagnosis is a clinical judgment. A code is an administrative
artifact assigned under a classification system, subject to documentation, coding rules,
query practices, and payment incentives. The two correlate, sometimes strongly. They are not
interchangeable, and the gap is measurable. Topic 05 treats this in detail.

## 5. The mechanics, step by step

This section walks the encounter as a data-generating process. Each step states the input, the
operation, the output, the reasoning, and the analytic artifact it leaves behind. The sequence
is not a rigid protocol; clinicians iterate. The purpose is to locate data production, not to
teach medicine.

### Phase A — Before the data exist

**Step 1. Identify the decision and its owner.**

- Input: a business or clinical problem statement, e.g., "reduce unplanned readmissions."
- Operation: name the decision, the person who makes it, the moment it is made, and the action
  available. "Which patients should receive a transitional-care call within 48 hours of
  discharge?" has an owner (the transitional-care nurse), a moment (before discharge), and an
  action (schedule a call). "Predict readmission" has none of these.
- Output: a decision statement in one sentence, with owner and action.
- Reasoning: a model without an action is a report. Reports are judged on description; models
  are judged on decisions. The evaluation metric, threshold, and error costs all follow from
  the decision, not from the data.
- Analytic artifact: a decision specification document with time zero, population, action, and
  cost of each error type.
- Branch point: if no owner can be named, stop. Do not build.

**Step 2. Fix time zero and the information boundary.**

- Input: the decision statement.
- Operation: write the exact moment of decision, then list the data that could physically be
  available. For discharge planning, time zero might be 12 hours before the planned discharge.
  For triage, it is arrival.
- Output: a timestamp rule and an availability rule.
- Reasoning: every feature must pass the test "was this known at time zero?" Features that fail
  are leaks. The most common leaks are discharge summaries, final diagnoses, later laboratory
  values, and any field updated after the encounter closes.
- Analytic artifact: a feature availability table with a source timestamp column for each
  feature.
- Branch point: if a feature has no reliable timestamp, either exclude it or assign it the
  latest possible time and justify the choice. Do not assume the row timestamp is the event
  time.

**Step 3. Define the encounter and the care setting.**

- Input: the decision and time zero.
- Operation: identify the setting — emergency department (ED), inpatient ward, intensive care
  unit (ICU), outpatient clinic, telehealth, skilled nursing facility — and the encounter
  boundaries.
- Output: an encounter key and a setting flag.
- Reasoning: measurement density, documentation style, and missingness differ by setting. An
  ICU has hourly vitals and daily labs; an outpatient visit may have one blood pressure and no
  labs. Pooling settings without a flag lets the model learn setting rather than physiology.
- Analytic artifact: `encounter_id`, `setting`, `admit_ts`, `discharge_ts`.
- Branch point: transfers between settings create multiple encounters. Decide whether to link
  them into one episode or treat them separately, and state the choice.

### Phase B — The clinical loop

**Step 4. Elicit the history and map it to fields.**

- Input: the patient interview and any prior records.
- Operation: capture chief complaint, history of present illness (HPI), past medical history,
  medications, allergies, family history, social history, and review of systems (ROS).
- Output: narrative plus structured elements.
- Reasoning: history is the highest-yield diagnostic information and the least standardized.
  Onset, duration, character, radiation, and associated symptoms are usually free text. Smoking
  status may be structured; occupation usually is not.
- Analytic artifact: a mapping table from history component to EHR field, with a missingness
  category.
- Branch point: if a field is absent, distinguish "not asked" from "asked and denied" from
  "not applicable." These have different meanings and different imputation consequences.

**Step 5. Perform the examination and record signs.**

- Input: the physical examination.
- Operation: record vital signs and system-specific findings.
- Output: measurements with method and site.
- Reasoning: a blood pressure is not one number. It is a cuff measurement, an arterial line
  measurement, or a patient-reported value; each has a different error structure. Temperature
  depends on site (oral, rectal, tympanic, temporal). Oxygen saturation depends on sensor,
  perfusion, and skin pigmentation.
- Analytic artifact: value, unit, method, site, device, and timestamp.
- Branch point: if method is missing, treat the value as lower quality and record the
  assumption. Do not silently pool methods.

**Step 6. Generate the differential diagnosis.**

- Input: history, examination, and prior probability.
- Operation: list plausible explanations, ordered by likelihood and by danger.
- Output: a differential with an implicit or explicit probability for each.
- Reasoning: the differential determines which tests are ordered, and therefore which data
  exist. A patient in whom heart failure is never considered will not get a B-type natriuretic
  peptide (BNP) test. Absence of a test is not evidence of absence of disease.
- Analytic artifact: none directly, but the differential explains order patterns and therefore
  missingness.
- Branch point: danger-first reasoning can produce tests for rare conditions. This is
  clinically appropriate and analytically distorting: the tested population is enriched for
  the condition.

**Step 7. Estimate pretest probability.**

- Input: the differential and the patient's characteristics.
- Operation: assign a probability, explicitly or implicitly, using a score, a validated model,
  or clinical judgment.
- Output: a number between 0 and 1, or a risk category.
- Reasoning: pretest probability is the anchor for all test interpretation. Without it, a
  sensitivity and specificity pair cannot be converted into a decision. In data terms, the
  prevalence in your evaluation cohort is the pretest probability, and it may differ sharply
  from the trial that produced the test characteristics.
- Analytic artifact: a prevalence estimate for the target population, with source and year.
- Branch point: if pretest probability is not recorded, reconstruct it from the cohort's
  observed prevalence, and state that this is an assumption.

**Step 8. Select and order the test.**

- Input: pretest probability, differential, and available tests.
- Operation: choose a test and a threshold; place the order.
- Output: an order with a timestamp, priority, and indication.
- Reasoning: the order is a decision, and it is observable. Ordering behaviour is influenced by
  guidelines, local availability, time of day, clinician experience, and medicolegal caution.
  Two hospitals with the same patients can have very different order rates.
- Analytic artifact: `order_id`, `test_code` (LOINC), `ordered_ts`, `priority`, `ordering_provider`.
- Branch point: "stat" versus routine priority changes turnaround time and therefore
  availability at time zero.

**Step 9. Collect the specimen or acquire the image.**

- Input: the order.
- Operation: phlebotomy, imaging acquisition, or physiological recording.
- Output: a specimen or a digital object.
- Reasoning: pre-analytical error is the largest source of laboratory error. Hemolysis,
  delayed processing, wrong tube, and insufficient volume all change the result before the
  assay runs. The collection timestamp is often missing or wrong.
- Analytic artifact: `collected_ts`, `specimen_type`, `container`, `hemolysis_index`.
- Branch point: if collection time is missing, the result time is not a substitute. Using it
  inflates apparent timeliness and can leak future information.

**Step 10. Analyze and result.**

- Input: specimen or image.
- Operation: assay, interpretation, validation, and release.
- Output: a result with units, reference interval, and flags.
- Reasoning: the result is a measurement plus an interpretation. Reference intervals are
  population- and assay-specific. Critical values trigger a call. The result timestamp is when
  it became available, not when the specimen was taken.
- Analytic artifact: `result_ts`, `value`, `unit`, `ref_low`, `ref_high`, `flag`, `assay`.
- Branch point: point-of-care results may enter the EHR through a different interface with
  different units and no reference interval.

**Step 11. Update the probability and act.**

- Input: the result and the clinical picture.
- Operation: revise the differential; treat, admit, discharge, or observe.
- Output: an action and a new probability.
- Reasoning: this is where the model's prediction would be consumed. The action is also a
  future label source: admission, antibiotic, surgery. Using the action as a feature is
  circular if the action is the outcome.
- Analytic artifact: orders, medication administrations, disposition.
- Branch point: the clinician may act before the result returns. The record will show the
  action first and the result later, which inverts the apparent causal order.

**Step 12. Document the encounter.**

- Input: the events and the clinician's reasoning.
- Operation: write or template a note; update the problem list.
- Output: narrative and structured documentation.
- Reasoning: notes serve care, communication, legal defence, and billing. They are written
  after the fact, often by template, and frequently copied forward. They are not research
  annotations and should not be treated as such.
- Analytic artifact: `note_ts`, `note_type`, `author_role`, text.
- Branch point: late notes may contain information that was not available at time zero even
  though the note timestamp is after it. The note timestamp is not an availability guarantee
  for its contents.

### Phase C — From record to dataset

**Step 13. Assign codes and bills.**

- Input: documentation and the encounter.
- Operation: clinical documentation improvement review, coding, claim submission.
- Output: ICD-10-CM diagnosis codes, ICD-10-PCS procedure codes, current procedural terminology
  (CPT) codes, and a Medicare severity diagnosis-related group (MS-DRG).
- Reasoning: coding is a translation of documentation under rules and incentives. Present-on-
  admission (POA) flags distinguish conditions present at admission from hospital-acquired
  conditions. A code without a POA flag is ambiguous for cohort work.
- Analytic artifact: `code`, `code_system`, `code_ts`, `poa_flag`, `principal_diagnosis`.
- Branch point: coding occurs after discharge, sometimes days later. A feature built from codes
  is not available at admission.

**Step 14. Extract, transform, and load.**

- Input: source systems.
- Operation: extract to a warehouse or research database; map to a common data model.
- Output: tables with timestamps, units, and identifiers.
- Reasoning: extraction introduces its own errors: timezone shifts, unit conversions, duplicate
  rows, truncated text, dropped negative signs, and de-identification shifts. A date shift
  applied for privacy can destroy interval calculations if applied inconsistently.
- Analytic artifact: an extract specification with source-to-target mappings and a data quality
  report.
- Branch point: if the extract is refreshed incrementally, late-arriving records can change
  historical cohorts. Version the extract.

**Step 15. Build the cohort and the label.**

- Input: the analytic tables.
- Operation: apply eligibility criteria, set the index date, define the outcome window, and
  compute the label.
- Output: one row per patient or encounter with a label and a time zero.
- Reasoning: cohort definition is a clinical argument, not a filter. Each criterion needs a
  rationale and an expected direction of bias. The label must be observable within the outcome
  window and must not be used to define eligibility.
- Analytic artifact: a cohort definition with inclusion, exclusion, index date, and label
  logic, versioned.
- Branch point: if the label is defined from the same data that defines eligibility, check for
  circularity. Example: excluding patients without a cardiology consult, then predicting
  cardiology consult.

**Step 16. Validate against the workflow.**

- Input: the cohort and the intended deployment.
- Operation: review a sample with clinicians; check timing, units, and definitions against the
  live workflow; test on a held-out site or period.
- Output: a validation report with disagreements and their causes.
- Reasoning: statistical validation answers "does it predict?" Workflow validation answers
  "will it be used, and will it be right in the place it is used?" Both are required.
- Analytic artifact: a sign-off record and a list of accepted limitations.
- Branch point: if clinicians cannot agree on the label in 20 chart reviews, the label is not
  ready. Fix the definition before modelling.

### 5.1 Branch points that change the analysis

The procedure above is mostly linear. Five branch points change the analytic design and deserve
separate treatment.

**Branch A: retrospective versus prospective time zero.** In retrospective data, you can see
everything. The temptation is to use it. The discipline is to simulate the information boundary
by filtering on timestamps. Write the filter as code, not as a habit.

**Branch B: order-based versus result-based features.** An order exists earlier than a result.
If the decision is "should I order this test?", the order cannot be a feature. If the decision
is "how should I interpret this result?", the result can be. Name which decision you are
supporting.

**Branch C: encounter-level versus patient-level splitting.** Multiple encounters per patient
create leakage across train and test. Split by patient, not by row, unless the deployment
genuinely treats each encounter as independent. Even then, report both.

**Branch D: code-based versus measurement-based phenotypes.** Codes are available late and are
incentive-sensitive; measurements are available early and are assay-sensitive. Choose based on
time zero and on whether the condition is chronic or acute.

**Branch E: documented versus observed.** A negative finding may be documented ("no chest
pain") or simply absent. Absence is not negation. Natural language processing (NLP) can
recover negation from text, but only if the text exists.

### 5.2 The timing table

For any proposed feature, fill in this table before modelling. If a row cannot be completed,
the feature is not ready.

| Feature | Source system | Event time | Record time | Available at time zero? | Leak risk |
|---|---|---|---|---|---|
| Triage systolic BP | ED flowsheet | 02:14 | 02:14 | Yes | Low |
| Admission diagnosis | Problem list | 02:40 | 02:55 | Yes | Medium (may change) |
| First troponin | Laboratory | 02:38 | 03:22 | Depends on time zero | High if time zero is 02:30 |
| Discharge summary | Documentation | 14:00 next day | 15:00 next day | No | Certain |
| MS-DRG | Billing | Day 5 | Day 12 | No | Certain |
| Prior-year claims | Claims warehouse | Prior year | Monthly lag | Yes, with lag | Low |

The table is deliberately mundane. Most leakage is mundane.

### 5.3 Reasoning about missingness

Missing data in clinical records has at least four mechanisms, and they are not interchangeable.

| Mechanism | Meaning | Example | Analytic consequence |
|---|---|---|---|
| Not applicable | The test does not apply | Pregnancy test in a male patient | Exclude from denominator |
| Not assessed | The clinician did not ask or examine | Sexual history not taken | Informative missingness |
| Not documented | Assessed but not recorded | Reflex documented in a note only | Measurement error |
| Not resulted | Ordered but cancelled or lost | Lab order cancelled | Workflow missingness |
| Structurally missing | The field does not exist at this site | No social work module | Site-level bias |
| Unavailable at time zero | Resulted later | 05:00 lab for a 03:00 decision | Leakage if imputed forward |

Treating all missing values as "normal" is the single most damaging imputation habit in clinical
data science. A missing respiratory rate is not 16 breaths/min. It is a missing observation, and
the reason it is missing is often that the patient was stable enough not to warrant measurement
or unstable enough that the monitor was off.

### 5.4 The label is a decision, not a fact

Every label answers a question that someone else already answered. For readmission, the label
is "admitted again within 30 days," which includes planned procedures and excludes out-of-network
admissions. For sepsis, the label may be a billable code, a culture order, or a research
definition. For mortality, the label may be in-hospital death, 30-day death, or death recorded
in a registry with a lag.

A useful test: ask three clinicians to define the label independently. If they disagree, the
label has a construct problem, and no modelling choice will fix it.

### 5.5 What the model can and cannot see

The model sees the record, not the patient. Three consequences follow.

1. **Symptoms that were not asked about do not exist.** A model cannot learn from a symptom
   that the documentation workflow never captures.
2. **Severity that was not measured is proxied by utilization.** When physiological severity is
   unmeasured, the model learns that more tests mean sicker patients. This is confounding by
   indication in its most general form.
3. **Access is encoded as disease.** A patient who cannot attend follow-up accumulates
   missed appointments, and a model may learn that missed appointments predict bad outcomes
   without representing the access barrier that causes them.

These are not statistical nuisances. They are the mechanism by which a model can be accurate
and inequitable at the same time.

## 6. Worked example

A 58-year-old man presents to the emergency department at 02:10 with 45 minutes of substernal
pressure that began at 01:25 while he was watching television. The pressure radiates to the left
arm and is accompanied by nausea. He took an antacid without relief. He has hypertension and
hyperlipidemia, takes lisinopril 10 mg daily and atorvastatin 40 mg nightly, smoked 30 pack-years
and quit two years ago, and his father had a myocardial infarction at age 62. He has no prior
cardiac history.

The case is realistic rather than exotic: the diagnostic question is common, the data are
ordinary, and the reasoning errors are the ones that appear in real charts.

### 6.1 The measurements

| Item | Value | Unit | Method or context |
|---|---:|---|---|
| Heart rate | 96 | beats/min | Monitor, regular |
| Blood pressure | 148/92 | mmHg | Cuff, right arm, seated |
| Respiratory rate | 18 | breaths/min | Manual count, 30 s x 2 |
| Oxygen saturation | 97 | % | Pulse oximetry, room air |
| Temperature | 36.8 | °C | Oral |
| Weight | 82 | kg | Standing scale |
| Height | 173 | cm | Stadiometer |
| Body mass index (BMI) | 27.4 | kg/m² | Calculated: 82 / 1.73² |

The BMI calculation: 1.73 m squared is 2.9929 m². 82 / 2.9929 = 27.40 kg/m². A BMI of 25.0 to
29.9 kg/m² is classified as overweight in adults by the World Health Organization (WHO)
classification; the value is a context variable, not a diagnosis.

### 6.2 The electrocardiogram

An electrocardiogram (ECG) recorded at 02:22 shows sinus rhythm at 96 beats/min, a PR interval
of 160 ms, a QRS duration of 92 ms, and no ST-segment elevation or depression in contiguous
leads. There are no pathological Q waves. The ECG is not diagnostic of ischemia, which does not
exclude it: a normal ECG is common in early acute coronary syndrome.

### 6.3 The pretest probability

The treating clinician estimates a 15% probability of acute coronary syndrome before the
troponin result. The estimate is supported by the History, ECG, Age, Risk factors, and Troponin
(HEART) score, a validated disposition aid:

| Component | Finding | Points |
|---|---|---:|
| History | Moderately suspicious | 1 |
| ECG | Normal | 0 |
| Age | 58 years (45-64 band) | 1 |
| Risk factors | Hypertension, hyperlipidemia, smoking, family history (4 of the listed factors) | 2 |
| Troponin | Below the assay upper reference limit at presentation | 0 |
| **Total** |  | **4** |

A HEART score of 4 is intermediate risk. The score is a communication and disposition tool, not
a diagnostic test, and it does not replace serial troponin measurement. Its value here is that
it forces the components of pretest probability into the open.

### 6.4 The test characteristics

The local laboratory uses a high-sensitivity cardiac troponin I (hs-cTnI) assay with an upper
reference limit (URL, the 99th percentile of a healthy population) of 26 ng/L. The hospital's
validation of its 0-hour/1-hour pathway, using local adjudication, reports 92% sensitivity and
88% specificity for index myocardial infarction at 30 days.

> **Note:** These performance values are local validation estimates, not universal constants.
> High-sensitivity troponin assays differ in their limits of detection, 99th-percentile URLs,
> and delta criteria. A published sensitivity from one assay and pathway cannot be transferred
> to another without verification.

The patient's results:

| Time | hs-cTnI (ng/L) | URL (ng/L) | Delta from previous (ng/L) |
|---|---:|---:|---:|
| 02:38 (0 h) | 8 | 26 | — |
| 03:52 (1 h) | 11 | 26 | 3 |

Both values are below the URL. The delta is small. Under the local pathway the sequence is
negative.

### 6.5 The arithmetic, first test

Take a hypothetical cohort of 10,000 patients with the same pretest probability of 15%.

- Diseased: 10,000 x 0.15 = 1,500.
- Not diseased: 10,000 x 0.85 = 8,500.
- True positives: 1,500 x 0.92 = 1,380.
- False negatives: 1,500 - 1,380 = 120.
- True negatives: 8,500 x 0.88 = 7,480.
- False positives: 8,500 - 7,480 = 1,020.

From these:

- PPV = 1,380 / (1,380 + 1,020) = 1,380 / 2,400 = 0.575, or 57.5%.
- NPV = 7,480 / (7,480 + 120) = 7,480 / 7,600 = 0.9842, or 98.4%.
- LR+ = 0.92 / (1 - 0.88) = 0.92 / 0.12 = 7.67.
- LR- = (1 - 0.92) / 0.88 = 0.08 / 0.88 = 0.0909.

The same test in a population with 1% prevalence gives a PPV of 7.2% and an NPV of 99.9%. In a
population with 30% prevalence it gives a PPV of 76.7% and an NPV of 96.3%. Sensitivity and
specificity are unchanged. The predictive values move by a factor of ten. This is the arithmetic
behind the claim that a test is "good" in one setting and "useless" in another.

### 6.6 The arithmetic, serial testing

The pretest odds are 0.15 / 0.85 = 0.17647.

After the first negative result: 0.17647 x 0.0909 = 0.01604. Convert to probability:
0.01604 / 1.01604 = 0.01579, or 1.58%.

The second test is not independent of the first: both are drawn from the same patient at
roughly the same time, and a patient with a small infarct may have two small values. Assume a
conditional LR- of 0.20 for the second negative result given a negative first result. This is an
assumption, and it should be replaced with local data when available.

After the second negative result: 0.01604 x 0.20 = 0.00320. Convert to probability:
0.00320 / 1.00321 = 0.00320, or 0.32%.

The post-test probability is now below 1%. For a patient with a normal ECG, a non-diagnostic
history, and two negative high-sensitivity troponins, outpatient evaluation is defensible under
many pathways. The HEART score of 4 still prompts a conversation about observation, and local
practice may admit for a provocation test. The arithmetic does not make the decision; it makes
the decision explicit.

### 6.7 What the data trail looks like

| Event | Timestamp | What it proves | What it does not prove |
|---|---|---|---|
| Symptom onset | 01:25 | Biology started | That the patient presented promptly |
| Arrival | 02:10 | Encounter began | That triage was immediate |
| Triage vitals | 02:14 | First measurement | That the cuff was correctly sized |
| ECG | 02:22 | Electrical activity at that moment | That ischemia is absent |
| Troponin order | 02:31 | Clinician suspected ACS | That the test was indicated |
| Specimen collected | 02:38 | Blood was drawn | That it was drawn correctly |
| Specimen received | 02:55 | Laboratory received it | That it was processed immediately |
| Result released | 03:22 | Result available | That the clinician saw it |
| Decision | 03:40 | Action taken | That the action was correct |
| Second result | 04:30 | Serial change available | That the first decision awaited it |

Collection-to-result turnaround time (TAT) is 44 minutes. Order-to-result TAT is 51 minutes.
Both are defensible. Neither is the number a model should use unless it matches the decision
boundary.

If the model's time zero is 03:40, the second troponin at 04:30 is a future value. If the model
uses it, the model knows the answer before the clinician did. The AUC will rise, the deployment
will fail, and the failure will be difficult to diagnose because the retrospective evaluation
will look excellent.

### 6.8 Where the label comes from

Suppose the outcome is "30-day major adverse cardiac event (MACE)," defined as death,
myocardial infarction, or urgent revascularization within 30 days. The label is assembled from
three sources: the state death registry, the hospital's troponin and procedure data, and
claims from affiliated providers. Each source has a lag. The registry may lag by weeks. Claims
may lag by a month. A patient who dies at an unaffiliated hospital may be counted as event-free.

The model may still be useful, but the label is a composite of imperfect linkages, and the
evaluation should report the linkage coverage. A 30-day MACE rate of 4.2% in a cohort with 92%
registry coverage is not the same quantity as a 4.2% rate with 99% coverage.

### 6.9 The decision memo

A one-page memo for this case would read:

- Decision: whether to discharge the patient with outpatient provocation testing or admit for
  observation.
- Owner: the emergency physician, with cardiology consultation available.
- Time zero: 03:40, after the first troponin result and before the second.
- Information available: history, examination, ECG, first troponin, local pathway performance.
- Label for a retrospective model: 30-day MACE, with linkage coverage reported.
- Failure mode: using the second troponin or the discharge diagnosis as a feature.

That memo is the deliverable of this topic. It is short because the thinking is done.

## 7. Code, query, or calculation

The following Python block is self-contained, uses only the standard library, and reproduces
every number in section 6. It states its assumptions in comments.

```python
"""Bayesian test interpretation for a chest-pain pathway.

Assumptions:
- sensitivity and specificity are those of the local 0/1h hs-cTnI pathway
- the second negative test has a conditional LR- of 0.20 given a negative first test
- prevalence is the pretest probability of the cohort, not of the general population
"""

from math import isclose


def two_by_two(prevalence: float, sensitivity: float, specificity: float,
               n: int = 10_000) -> dict:
    """Return counts and operating characteristics for a binary test."""
    diseased = prevalence * n
    not_diseased = n - diseased

    tp = sensitivity * diseased
    fn = diseased - tp
    tn = specificity * not_diseased
    fp = not_diseased - tn

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    lr_pos = sensitivity / (1 - specificity)
    lr_neg = (1 - sensitivity) / specificity
    return {
        "prevalence": prevalence,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "ppv": ppv, "npv": npv,
        "lr_pos": lr_pos, "lr_neg": lr_neg,
    }


def odds_to_probability(odds: float) -> float:
    return odds / (1 + odds)


def probability_to_odds(p: float) -> float:
    return p / (1 - p)


SENS = 0.92
SPEC = 0.88

print(f"{'prevalence':>10} {'PPV':>8} {'NPV':>8} {'LR+':>8} {'LR-':>8}")
for prev in (0.01, 0.15, 0.30):
    r = two_by_two(prev, SENS, SPEC)
    print(f"{prev:>10.2f} {r['ppv']:>8.4f} {r['npv']:>8.4f} "
          f"{r['lr_pos']:>8.4f} {r['lr_neg']:>8.4f}")

pretest = 0.15
odds_0 = probability_to_odds(pretest)
r = two_by_two(pretest, SENS, SPEC)
odds_1 = odds_0 * r["lr_neg"]
p_1 = odds_to_probability(odds_1)
odds_2 = odds_1 * 0.20          # conditional LR- for the second negative
p_2 = odds_to_probability(odds_2)

print()
print(f"pretest probability {pretest:.2f} -> odds {odds_0:.5f}")
print(f"after first negative: probability {p_1:.5f}")
print(f"after second negative (conditional LR- 0.20): probability {p_2:.5f}")


def heart_score(history: int, ecg: int, age: int, risk_factors: int,
                troponin: int) -> tuple[int, str]:
    """HEART score. Each component is scored 0, 1, or 2 by the treating clinician."""
    for name, value in (("history", history), ("ecg", ecg), ("age", age),
                        ("risk_factors", risk_factors), ("troponin", troponin)):
        if value not in (0, 1, 2):
            raise ValueError(f"{name} must be 0, 1, or 2")
    total = history + ecg + age + risk_factors + troponin
    if total <= 3:
        band = "low"
    elif total <= 6:
        band = "intermediate"
    else:
        band = "high"
    return total, band


score, band = heart_score(history=1, ecg=0, age=1, risk_factors=2, troponin=0)
print()
print(f"HEART score: {score} ({band})")

assert isclose(p_1, 0.01579, abs_tol=1e-5)
assert isclose(p_2, 0.00320, abs_tol=1e-5)
assert score == 4
```

Expected output:

```text
prevalence      PPV      NPV      LR+      LR-
      0.01   0.0719   0.9991   7.6667   0.0909
      0.15   0.5750   0.9842   7.6667   0.0909
      0.30   0.7667   0.9625   7.6667   0.0909

pretest probability 0.15 -> odds 0.17647
after first negative: probability 0.01579
after second negative (conditional LR- 0.20): probability 0.00320

HEART score: 4 (intermediate)
```

The assertions fail loudly if the arithmetic drifts. That is the point: a notebook that
silently prints a number is not a reproducible calculation.

A SQL version of the same cohort filter, written for a generic warehouse, shows the time-zero
discipline:

```sql
-- Cohort: ED encounters with a chest-pain chief complaint.
-- Time zero: first troponin result time, not arrival.
WITH ed AS (
    SELECT encounter_id,
           patient_id,
           admit_ts,
           discharge_ts
    FROM encounters
    WHERE setting = 'ED'
),
first_trop AS (
    SELECT o.encounter_id,
           MIN(r.result_ts) AS trop_ts
    FROM orders o
    JOIN results r
      ON r.order_id = o.order_id
    WHERE o.loinc_code = '10839-9'   -- troponin I, assay-specific mapping required
      AND o.ordered_ts >= (SELECT admit_ts FROM ed e WHERE e.encounter_id = o.encounter_id)
    GROUP BY o.encounter_id
)
SELECT e.encounter_id,
       e.admit_ts,
       t.trop_ts AS time_zero
FROM ed e
JOIN first_trop t USING (encounter_id)
WHERE t.trop_ts <= e.discharge_ts;
```

Two warnings about the query. First, the LOINC code shown is a placeholder for the assay in use;
troponin codes are assay-specific and must be mapped against the laboratory dictionary. Second,
the query uses only data available at or before the first troponin result. Any feature table
joined later must carry the same constraint.

## 8. Data, standards, and regulatory context

Clinical reasoning does not have its own standard. It is constrained by the standards that carry
the record. Four families matter here.

**HL7 version 2 (v2).** The dominant hospital messaging standard. An admission generates an
ADT^A01 message (admit, discharge, transfer); a laboratory result generates an ORU^R01 message.
HL7 v2 is pipe-delimited, event-driven, and locally customized. The same field can mean
different things at two sites. When you receive an extract, ask for the interface specification,
not just the table.

**HL7 Fast Healthcare Interoperability Resources (FHIR).** A resource-oriented standard, now at
Release 4 (R4) with R5 published. The resources relevant to this topic are Encounter,
ServiceRequest (an order), Observation (a result), DiagnosticReport (a report), Composition and
DocumentReference (documents), and Provenance (who did what, when). FHIR makes the time-zero
problem explicit because each resource carries its own timestamp and status.

**Logical Observation Identifiers Names and Codes (LOINC).** The vocabulary for tests and
measurements. Version 2.80 was released in February 2025 as a hotfix with 88 new survey terms
and a total of 104,672 terms; later releases continue the numbering. LOINC distinguishes the
component, property, timing, system, scale, and method, which is exactly the information needed
to avoid pooling incompatible measurements.

**Systematized Nomenclature of Medicine Clinical Terms (SNOMED CT).** The clinical terminology
used for problem lists and findings in many countries. SNOMED CT is compositional: a concept can
be refined with attributes. It is not a billing classification and does not map one-to-one to
ICD-10-CM.

**ICD-10-CM.** The United States diagnosis classification maintained by the Centers for Disease
Control and Prevention (CDC) and the Centers for Medicare and Medicaid Services (CMS). The
fiscal year 2026 release is effective 1 October 2025 through 30 September 2026. Codes are
assigned after documentation review and are not available at the point of care.

**United States Core Data for Interoperability (USCDI).** A required dataset for certified health
information technology. USCDI version 3 became the baseline under the Health Data, Technology,
and Interoperability: Certification Program Updates, Algorithm Transparency, and Information
Sharing (HTI-1) final rule, published 9 January 2024 and effective 8 February 2024. USCDI v1
expires on 1 January 2026. HTI-1 also replaced the clinical decision support criterion with a
Decision Support Interventions criterion for predictive tools supplied by a developer, effective
1 January 2025. The rule requires transparency about source attributes, user feedback, and risk
management, and it is a binding certification requirement for developers of certified health
information technology, not a general clinical guideline.

**Privacy.** In the United States, the Health Insurance Portability and Accountability Act
(HIPAA) Privacy Rule governs use and disclosure of protected health information. De-identification
can follow the Safe Harbor method, which removes 18 specified identifiers, or the expert
determination method. Date shifting is common in research extracts and can break interval
calculations if the shift is not consistent within a patient.

**Decision support regulation.** The Food and Drug Administration (FDA) guidance *Clinical
Decision Support Software* (September 2022) sets out when software that supports clinical
decisions is a device. The analysis depends on the function, the intended user, and whether the
user can independently review the basis of the recommendation. A "research use only" label does
not automatically remove other obligations.

> **Note:** The regulatory content in this section is educational. It is not medical advice or legal
> advice, and it does not establish that any product is compliant. Regulatory classification
> requires review by qualified counsel and by the relevant regulator.

What the standards do not do: they do not enforce clinical validity. A perfectly conformant
FHIR Observation can carry a wrong value in the wrong unit at the wrong time. Interoperability
is a necessary condition for good data, not a sufficient one.

## 9. Industry practice and case evidence

The gap between a model's retrospective performance and its clinical value is documented in
several public cases. Three are directly relevant to clinical reasoning and workflow.

**Case 1: the Epic Sepsis Model.** A proprietary sepsis prediction model was widely deployed in
US hospitals. An external validation at a large academic health system, published in *JAMA
Internal Medicine* in 2021, found discrimination substantially below the vendor's reported
performance and a high burden of false alerts. The study is a validation study, not a trial of
patient outcomes. The documented lesson is that local prevalence, label definition, and alert
workflow change performance, and that vendor validation does not substitute for local
evaluation. The inference — that early-warning systems can still help when paired with a
response team — is plausible but was not established by that study.

**Case 2: the commercial risk-stratification algorithm studied by Obermeyer and colleagues.** A
population health algorithm used health-care cost as a proxy for need. Because less is spent on
Black patients at the same level of illness, the algorithm systematically underestimated their
need. The study, published in *Science* in 2019, is the canonical demonstration that a label
choice embeds a social process. The algorithm was not predicting biology; it was predicting
cost, and cost reflected access. The documented finding is a large racial disparity in
identifying patients for extra care. The inference for practice is that label selection is an
ethical decision, not a technical detail.

**Case 3: copy-forward in documentation.** A 2025 survey of 913 providers at a large academic
medical center reported that 82% used copy-paste and 52% used copy-forward; use was highest on
inpatient primary services at 91% and 68%, respectively. A 2024 audit of 128 patients' ward-round
notes found erroneous copying in 8.3% of notes, concentrated in the issues list, and reported
that a threshold of more than 850 unchanged sequential characters detected erroneous copying
with 0.88 sensitivity and 0.98 specificity. These are documentation studies, not model studies.
Their relevance is direct: if a note contains a stale diagnosis, any phenotype that reads the
note inherits the staleness.

**Case 4: ambient documentation tools.** A 2025 multicentre study of 263 ambulatory clinicians
reported burnout falling from 51.9% to 38.8% after 30 days of ambient scribe use, with reduced
after-hours documentation. A separate 2025 evaluation of six AI scribes found that none produced
error-free notes, with omissions and inappropriate generalization of clinical detail as common
failure modes. The documented fact is that documentation burden can fall. The inference is that
note quality does not automatically rise, and that clinician review remains necessary.

**What good practice looks like.** The common elements in organizations that avoid the failures
above are unglamorous:

1. A written decision specification with an owner and a time zero.
2. A feature availability table reviewed by someone who knows the source workflow.
3. A label adjudication exercise with at least 20 chart reviews before modelling.
4. Site-level and period-level validation, not only random splits.
5. A monitoring plan that includes alert volume, override rate, and subgroup performance.

**What inference is not.** None of the cases above proves that a given model will fail at your
site. They establish that the failure modes are real and recurrent. Your local validation is the
only evidence about your local deployment.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future information in features | AUC far above published benchmarks | Discharge codes, late labs, or notes included | Unusable at deployment | Feature availability table; timestamp filters in code |
| Label defined from treatment | Outcome rate tracks order rates | Sepsis label from culture or antibiotics | Model predicts clinician behaviour | Adjudicate a chart sample; compare definitions |
| Cohort defined after the outcome | Eligibility depends on a post-outcome event | Excluding patients who died before a test | Collider bias | Fix eligibility at time zero |
| Confounding by indication | Treatment looks harmful | Sicker patients receive more therapy | Wrong causal claim | Restrict, stratify, or use a design that handles indication |
| Missing treated as normal | Imputed vitals cluster at defaults | Default charting, "within normal limits" templates | Underestimated severity | Missingness indicators; mechanism-specific imputation |
| Unit mixing | Values span implausible ranges | mg/dL and mmol/L pooled; arterial and venous | Wrong thresholds and labels | Unit column; assay dictionary; range checks |
| Setting mixing | Model learns site or unit | ICU, ward, and ED pooled without flags | Poor transportability | Setting flag; per-setting evaluation |
| Encounter-level leakage | Same patient in train and test | Multiple encounters per patient | Inflated test performance | Split by patient; report both splits |
| Base-rate neglect | High PPV claimed from a trial | Prevalence differs from deployment | Alert burden | Recompute PPV at local prevalence |
| Alert fatigue | Override rate rises over weeks | Low PPV, duplicate alerts | Clinicians ignore the model | Measure override rate; cap alert volume |
| Automation bias | Clinicians accept wrong suggestions | No time or information to verify | Patient harm | Show basis and uncertainty; audit overrides |
| Copy-forward contamination | Note phenotypes include historical diagnoses | Templates and copy-paste | Label noise and leakage | Note provenance; similarity detection |
| Stale problem list | Active list includes resolved conditions | Lists are not curated | False-positive cohort | Require evidence of active management |
| Unclear decision owner | No one can name the action | Model built for a report | No deployment path | Stop until owner and action are named |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The recorded timestamp reflects the event time | Time-zero filters depend on it | Interface delays, back-entry, timezone shifts | Compare order, collect, and result times on a sample |
| The label captures the clinical outcome | Supervision requires a target | Linkage gaps, code ambiguity, out-of-network events | Chart adjudication; linkage coverage report |
| Prevalence in the evaluation cohort matches deployment | PPV and NPV depend on it | Case-mix change, referral filter, seasonality | Recompute at deployment prevalence |
| Sensitivity and specificity are stable across subgroups | Test interpretation assumes transportability | Assay differences, spectrum bias, comorbidity | Subgroup re-estimation |
| Orders reflect clinical suspicion | Order-based features are meaningful | Protocol orders, standing orders, order sets | Review order indications |
| Documentation reflects what happened | Notes are used for phenotyping | Copy-forward, templates, late entries | Note similarity audit; source triangulation |
| One row represents one care episode | Cohort construction depends on it | Transfers, merged encounters, duplicates | Duplicate detection; encounter linkage review |
| The model's action is available | Value requires an intervention | No capacity, no workflow, no reimbursement | Process map with the owner |
| Clinicians will review the output | Safety assumes human oversight | Workload, automation bias, alert volume | Override and response-time monitoring |
| Retrospective performance predicts prospective performance | Investment decisions depend on it | Temporal drift, workflow change, data pipeline change | Prospective silent-mode trial |

## 12. Exercises

**Exercise 1 — Decision specification (recall and application, 45 minutes).**
Choose a clinical prediction task from a public dataset. Write a one-page specification with the
decision, owner, action, time zero, population, label, and the cost of a false positive and a
false negative. Deliverable: one page, maximum 500 words. Acceptance criteria: the time zero is a
timestamp rule, not a date; the owner is a role, not "the clinician"; both error costs are stated
in units other than "bad." Time: 45 minutes.

**Exercise 2 — Time-zero audit (application, 60 minutes).**
Take a published model or a dataset description and list ten candidate features. For each, record
the source, event time, record time, and whether it is available at the stated time zero.
Deliverable: a table with ten rows and a leak verdict per row. Acceptance criteria: at least two
features are correctly identified as leaks; the verdict cites a timestamp or a workflow reason.
Time: 60 minutes.

**Exercise 3 — Sensitivity, specificity, and prevalence (calculation, 40 minutes).**
A test has 90% sensitivity and 95% specificity. Compute PPV and NPV at prevalence 0.5%, 5%, and
25%. Then compute the number needed to test to find one true case at each prevalence. Deliverable:
a table and three sentences of interpretation. Acceptance criteria: arithmetic correct to three
decimal places; the interpretation names base-rate neglect. Time: 40 minutes.

**Exercise 4 — Label critique (critique, 60 minutes).**
Take a sepsis, heart failure, or readmission phenotype from a public paper. Identify what the
label actually measures, what it misses, and one direction of bias. Deliverable: a 400-word
critique with a revised label definition. Acceptance criteria: the critique distinguishes
construct from measurement; the revision is computable. Time: 60 minutes.

**Exercise 5 — Missingness classification (application, 45 minutes).**
For a de-identified extract, classify 15 missing fields into not applicable, not assessed, not
documented, not resulted, structurally missing, or unavailable at time zero. Deliverable: a
table with the classification and the evidence for it. Acceptance criteria: at least three
distinct mechanisms are represented; no field is classified as "unknown" without a reason. Time:
45 minutes.

**Exercise 6 — Workflow map (design, 90 minutes).**
Map the path of a laboratory result from order to clinician action at a real or simulated site.
Identify every system, interface, and manual step. Mark where delay and error can enter.
Deliverable: a one-page diagram with a delay estimate per step. Acceptance criteria: at least
six steps; at least three failure points with a detection method. Time: 90 minutes.

**Exercise 7 — Bayesian memo (application and communication, 60 minutes).**
Using the chest-pain case, write a 300-word memo to an emergency physician explaining why the
same troponin result means different things in a 45-year-old without risk factors and a
75-year-old with prior coronary disease. Deliverable: 300 words, no equations in the body, one
appendix with the arithmetic. Acceptance criteria: the mechanism is explained without jargon;
the numbers are correct; the recommendation is conditional on pretest probability. Time: 60
minutes.

**Exercise 8 — Failure-mode red team (critique, 75 minutes).**
Take a model you have built or read about and complete section 10's table for it with at least
eight rows. For each row, state the earliest observable warning sign. Deliverable: a completed
table and a 200-word priority ranking. Acceptance criteria: each mitigation is testable; the
ranking names the two failures most likely to occur first. Time: 75 minutes.

## 13. Self-check answers

**Answer to Exercise 3.** At prevalence 0.5%: diseased 5 per 1,000; TP 4.5; FN 0.5; non-diseased
995; TN 945.25; FP 49.75. PPV = 4.5 / 54.25 = 0.0829, or 8.3%. NPV = 945.25 / 945.75 = 0.9995.
At 5%: TP 45; FP 47.5; PPV 45 / 92.5 = 0.4865; NPV = 902.5 / 907.5 = 0.9945. At 25%: TP 225;
FP 37.5; PPV 225 / 262.5 = 0.8571; NPV = 712.5 / 737.5 = 0.9661. Number needed to test to find
one true case is 1 / (prevalence x sensitivity): 222 at 0.5%, 22.2 at 5%, 4.4 at 25%. The
interpretation is that a highly specific test can still have a low PPV when the condition is
rare, and that the number needed to test is the operational consequence.

**Answer to self-check on time zero.** For a decision at 03:40, the first troponin result at
03:22 is available. The second troponin at 04:30 is not. The discharge diagnosis, the MS-DRG, and
the discharge summary are not. The admission note may exist, but if it was written at 06:00 it is
not available at 03:40 even though it describes events before 03:40. The rule is record time, not
event time, for availability.

**Answer to self-check on PPV.** A test with 99% sensitivity can have a low PPV when prevalence
is low because false positives from the large healthy population outnumber true positives from
the small diseased population. At prevalence 1%, sensitivity 99%, specificity 95%: TP 99, FP 495,
PPV = 99 / 594 = 16.7%. The sensitivity is excellent; the positive result is still usually wrong.

**Answer to self-check on missing respiratory rate.** A missing respiratory rate is not normal.
It may be missing because the patient was stable and the value was never charted, because the
monitor did not capture it, or because the value was charted in free text. Imputing 16
breaths/min assumes the first mechanism and ignores the other two. The correct treatment is a
missingness indicator plus a sensitivity analysis.

**Answer to self-check on codes.** A code is assigned after documentation review, often after
discharge, under rules that reward specificity and comorbidity capture. It is not available at
admission, it may reflect "history of" rather than active disease, and it may be present because
a condition was ruled out and documented as such. A code-based cohort should state the code set,
the position (principal versus secondary), the POA flag, and the date window.

**Rubric for Exercise 1.** Full credit requires: a named role as owner; a timestamp-level time
zero; an action that exists in the workflow; a label with a named source; both error costs in
concrete units; and one stated failure mode. Half credit if the time zero is a date or the owner
is generic. No credit if the deliverable is a restatement of the prediction task.

**Rubric for Exercise 6.** Full credit requires: at least six steps; at least three failure
points; a detection method for each; and a delay estimate with a unit. Half credit if the map is
conceptual without timestamps. No credit if the map omits the laboratory interface.

**Rubric for Exercise 8.** Full credit requires: eight rows with distinct mechanisms; warning
signs observable before harm; mitigations that can be implemented and tested; and a priority
ranking justified by likelihood and severity. Half credit if rows duplicate a mechanism. No
credit if mitigations are restatements of the failure.

## 14. Interview questions

**1. What is the difference between a symptom, a sign, and a diagnosis?**
A strong answer distinguishes patient report, observer finding, and synthesized clinical
judgment, and gives an example of each from the same case. A weak answer treats them as
synonyms or says a diagnosis is "what the code says."

**2. Why is respiratory rate often unreliable in EHR data?**
A strong answer names manual charting, default values, imputation, and monitor artifacts, and
explains that missingness is informative. A weak answer says "it is noisy" without a mechanism.

**3. A model predicts sepsis with AUC 0.92. What else do you need?**
A strong answer asks for prevalence, calibration, alert burden, time to detection, subgroup
performance, label definition, and the action the alert triggers. A weak answer asks only for a
test set or a confidence interval.

**4. How would you define heart failure for a readmission model?**
A strong answer combines codes with evidence of active disease (natriuretic peptide, ejection
fraction, diuretic use), distinguishes history from active disease, and sets the time zero before
discharge. A weak answer uses a single ICD code.

**5. What is confounding by indication, and why does it matter?**
A strong answer explains that treatment is chosen because of prognosis, gives a clinical example,
and describes a design response. A weak answer calls it "correlation not causation" without a
mechanism.

**6. Why can a test with 99% sensitivity have a low PPV?**
A strong answer computes the arithmetic at low prevalence. A weak answer says "because of false
positives" without quantifying.

**7. What is automation bias, and how do you design against it?**
A strong answer describes over-reliance, names the conditions that increase it, and proposes
uncertainty display, forced review, override tracking, and audit. A weak answer says "train the
users."

**8. How does pulse oximetry bias affect a model?**
A strong answer explains that measurement error correlated with skin pigmentation produces
differential misclassification, and that the model inherits the bias. A weak answer says the
model is biased because the data are biased, without a mechanism.

**9. What is copy-forward, and why is it dangerous?**
A strong answer describes reuse of prior text, stale diagnoses, and future plans entering the
current note, and links it to label leakage. A weak answer says it is a documentation
efficiency tool.

**10. What does "clinically plausible" mean for a feature?**
A strong answer covers range, temporal sequence, physiological relationships, measurement
context, and cross-variable consistency. A weak answer says "within normal limits."

**11. When would you prefer a simple score like NEWS2 to a complex model?**
A strong answer weighs transparency, validation, implementation cost, trust, and regulatory
burden, and notes that a simple score can be safer when the data are thin. A weak answer assumes
complex models are always better.

**12. How would you evaluate a neurologic deterioration model?**
A strong answer defines deterioration, sets the time window, handles competing risks, measures
alert burden, and reports subgroup performance. A weak answer reports AUC only.

## 15. Key terms

- **AUC** — area under the receiver operating characteristic curve; a discrimination measure.
- **Automation bias** — over-reliance on an automated output.
- **Base rate** — prevalence of the condition in the population.
- **Chief complaint** — the patient's reason for presenting, in their words.
- **Clinical decision support (CDS)** — software that supports clinical decisions.
- **Confounding by indication** — treatment choice driven by prognosis.
- **Copy-forward** — reuse of prior note text in a new note.
- **Differential diagnosis** — ranked list of plausible explanations.
- **Encounter** — a single interaction with the health system.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **HEART score** — History, ECG, Age, Risk factors, Troponin disposition aid.
- **HL7 v2** — the dominant hospital messaging standard.
- **ICD-10-CM** — the US diagnosis classification.
- **Index date** — the calendar anchor for cohort entry.
- **Label** — the target variable.
- **Label leakage** — label information present in features.
- **Likelihood ratio** — factor by which a result multiplies the odds.
- **LOINC** — Logical Observation Identifiers Names and Codes.
- **MACE** — major adverse cardiac event.
- **Missingness mechanism** — the reason a value is absent.
- **NPV** — negative predictive value.
- **Phenotype** — a computable definition of a condition.
- **PPV** — positive predictive value.
- **Pretest probability** — probability of disease before a test.
- **Sensitivity** — TP / (TP + FN).
- **SNOMED CT** — Systematized Nomenclature of Medicine Clinical Terms.
- **Specificity** — TN / (TN + FP).
- **System 1 / System 2** — fast pattern matching versus slow analytic reasoning.
- **Time zero** — the moment at which a prediction is made.
- **Turnaround time (TAT)** — elapsed time from order or collection to result.
- **USCDI** — United States Core Data for Interoperability.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Guidelines and standards**

- American College of Cardiology and American Heart Association. 2025
  ACC/AHA/ACEP/NAEMSP/SCAI Guideline for the Management of Patients With Acute Coronary
  Syndromes. *Journal of the American College of Cardiology*, 2025.
  https://www.jacc.org/doi/pdf/10.1016/j.jacc.2024.11.009
- Byrne RA, Rossello X, Coughlan JJ, et al. 2023 ESC Guidelines for the management of acute
  coronary syndromes. *European Heart Journal*, 2023;44:3720-3826.
  https://doi.org/10.1093/eurheartj/ehad191
- Health Level Seven International. FHIR Release 4 specification.
  https://hl7.org/fhir/R4/
- Logical Observation Identifiers Names and Codes. Version 2.80 release notes, February 2025.
  https://loinc.org/kb/release-notes-archive/2.80
- Centers for Disease Control and Prevention. ICD-10-CM FY2026 release and official guidelines,
  effective 1 October 2025. https://www.cdc.gov/nchs/icd/icd-10-cm/files.html
- Office of the National Coordinator for Health Information Technology. HTI-1 Final Rule,
  published 9 January 2024. https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
- US Food and Drug Administration. *Clinical Decision Support Software: Guidance for Industry
  and Food and Drug Administration Staff*, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software

**Evidence**

- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*, 2021.
  https://doi.org/10.1001/jamainternmed.2021.2626
- Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used
  to manage the health of populations. *Science*, 2019. https://doi.org/10.1126/science.aax2342
- Hudkins ML, et al. Exploring provider perceptions and attitudes toward copy-paste and
  copy-forward in clinical documentation. *Applied Clinical Informatics*, 2025.
  https://doi.org/10.1055/a-2574-1348
- Tsimiklis P, et al. Copying in medical documentation: developing an evidence-based approach.
  *Internal Medicine Journal*, 2024. https://doi.org/10.1111/imj.16590
- Olson J, et al. Use of ambient AI scribes to reduce administrative burden and professional
  burnout. *JAMA Network Open*, 2025. https://doi.org/10.1001/jamanetworkopen.2025.34976
- Ha S, et al. Evaluating the usability, technical performance, and accuracy of artificial
  intelligence scribes for primary care. *JMIR Human Factors*, 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12309782/
- Rautaharju PM, Surawicz B, Gettes LS, et al. AHA/ACCF/HRS recommendations for the
  standardization and interpretation of the electrocardiogram: Part IV. *Journal of the
  American College of Cardiology*, 2009;53:982-991.
  https://www.jacc.org/doi/10.1016/j.jacc.2008.12.014

**Background reading**

- Bickley LS. *Bates' Guide to Physical Examination and History Taking*, 13th edition.
- Goldberger AL, et al. *Goldberger's Clinical Electrocardiography*, 9th edition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_anatomy-physiology-and-vital-signs.md`](02_anatomy-physiology-and-vital-signs.md)
- Related topic: [`03_laboratory-medicine.md`](03_laboratory-medicine.md)
- Related topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Survey file: [`../../01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

