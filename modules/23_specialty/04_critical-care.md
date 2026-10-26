# Critical Care

> Module 23 — Specialty Electives
> Topic 04 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a
> resuscitation protocol, and not a substitute for the current Surviving Sepsis
> Campaign guideline, the KDIGO acute kidney injury guideline, or institutional
> policy. Every threshold below names its source and version. Where a guideline
> is under revision, the status at the access date is stated.

## 1. Why this topic exists

Critical care is the specialty where the time between a data event and a clinical
consequence is shortest. A lactate result, a urine output total, a vasopressor rate,
and a ventilator setting are not abstractions; each one drives an action within
minutes. That compression is why critical-care data science attracts attention, and
why it fails in a characteristic way: the model is evaluated on a retrospective
cohort whose definition of onset would have been impossible to compute in real time.

The ordinary path of a septic patient illustrates the problem. A patient arrives in
the emergency department at 14:20 with fever and confusion. A blood culture is drawn
at 18:05. The first antibiotic is administered at 19:10. A lactate is resulted at
18:20. The patient is admitted to the intensive care unit (ICU) at 21:40. Over the
next 48 hours, platelets fall, bilirubin rises, creatinine climbs, and norepinephrine
is started. A retrospective extract joins all of these events on the patient
identifier. A model trained on that extract sees the whole sequence at once and learns
to predict sepsis from data that, at the moment of the alert, did not yet exist.

The signature failures of critical-care data science are measurement failures:

- **Onset-time drift.** Sepsis onset is anchored at ICU admission in one study, at
  the first antibiotic in another, and at the first culture in a third. These are not
  the same instant, and the difference is often six to twelve hours, which is exactly
  the window in which an early-warning model is supposed to act.
- **Window contamination.** The Sepsis-3 framework measures organ dysfunction over a
  window that extends 24 hours after suspicion. A model that uses the full window as
  input has seen the future by construction.
- **Score recomputation drift.** The Sequential Organ Failure Assessment (SOFA) score
  is computed from the worst value in a window. Pipelines that use the worst value of
  the entire stay produce a different score, a different severity distribution, and a
  different label.
- **Vasopressor dose non-comparability.** Norepinephrine, epinephrine, vasopressin,
  phenylephrine, and dopamine are recorded in different units and potencies. A cohort
  that compares "vasopressor dose" without normalisation is comparing notation.
- **Unit errors that survive every type check.** Creatinine in micromoles per litre
  read as milligrams per decilitre is a factor of 88.4. A pipeline that stores
  numbers without units will accept it, and the resulting SOFA renal component is
  wrong by two points.
- **Discharge treated as a non-event.** In ICU data, discharge alive and death
  compete. A time-to-discharge model that ignores death will report that the sickest
  patients are discharged soonest, because they died.

None of these failures produces an exception. The pipeline runs, the model trains,
the dashboard renders, and the cohort is quietly wrong.

### 1.1 Who is blocked when the measurement is wrong

The cost of a bad critical-care measurement is distributed across people who cannot
see the measurement.

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | An alert that fired too late, or a therapy titrated against a misread dose | Cannot see the cohort definition or the unit conversion |
| Bedside nurse | Trust in an alert that fires on patients who are already treated | Cannot audit the model or the feature window |
| Intensivist | A severity score that disagrees with the bedside impression | Cannot reconstruct the SOFA window |
| Hospital | Alert fatigue, avoidable escalation, and unreimbursed software cost | Buys a product, not a measurement specification |
| EHR vendor | A deployed model that fails external validation | Inherited label definitions were never documented |
| Payer | Sepsis bundle performance that reflects documentation, not care | Sees claims and measure flags, not physiology |
| Regulator | A device claim that cannot be tied to a reproducible endpoint | Receives a validation report, not the cohort |
| Data scientist | A model that validates internally and collapses at the next site | Inherited definitions were never written down |

The last row is the professional risk. A critical-care model that cannot state its
time zero, its SOFA window, its vasopressor normalisation, and its handling of
discharge is not a weak model; it is an unfalsifiable one. A clinical reviewer who
knows the domain will establish that in the first meeting, usually by asking what
happens to the label when the antibiotic is given before the culture.

### 1.2 What this topic covers and what it does not

This topic covers the measurement layer. It explains how a critical-care episode
becomes structured data, how suspicion time and organ dysfunction are defined, how
severity scores are computed, how vasopressor doses are normalised, how fluid balance
and ventilation variables are derived, how discharge competes with death, and how
unit errors enter. It does not teach resuscitation, ventilator management, or drug
pharmacology. It assumes familiarity with survival analysis and does not re-derive
the Kaplan-Meier estimator.

The emphasis is deliberate. Most critical-care machine-learning courses spend their
time on sequence models and almost none on the fact that the same patient can have
three different sepsis onset times in three different tables. The second problem
destroys more deployments.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Fix a Sepsis-3 suspicion time from culture and antibiotic timestamps, including both pairing directions | Exercise 1; self-check 1 |
| 2 | Compute a six-component SOFA score inside a stated window and defend each component's source value | Exercise 2; self-check 2 |
| 3 | Stage acute kidney injury (AKI) by the Kidney Disease: Improving Global Outcomes (KDIGO) 2012 criteria, including an unknown baseline | Exercise 3; self-check 3 |
| 4 | Normalise a multi-agent vasopressor regimen to norepinephrine equivalents and state the formula version | Exercise 4; self-check 4 |
| 5 | Derive fluid balance and ventilation variables (tidal volume per predicted body weight, driving pressure, PaO2/FiO2) with units intact | Exercise 5; interview question 5 |
| 6 | Model discharge as a competing risk and explain what a naive time-to-discharge model reports | Exercise 6; self-check 6 |
| 7 | Detect and quantify unit conversion errors in a critical-care extract | Exercise 7; interview question 7 |
| 8 | Auditing a deployed sepsis model, distinguish discrimination, calibration, alert burden, and workflow effect | Exercise 8; interview question 9 |

Outcomes 1 and 2 are the measurement core. They are separated because a learner can
often compute a SOFA score from a table and still be unable to say which values were
admissible at the decision moment. The second skill is the one that survives a
deployment review.

Outcomes 3 and 4 are the quantitative comparisons that most often break a cohort
study. Both require arithmetic and both have more than one defensible convention,
which is why the convention must be named.

Outcomes 5 and 6 are about derived variables and time. A fluid balance without a
denominator and a length-of-stay estimate without competing risks are both numbers
that look complete and are not.

Outcomes 7 and 8 are about context. They connect the measurement layer to the
validation and workflow questions that decide whether a product can be used.

### 2.1 How the outcomes map to the module

Outcomes 1 and 2 are prerequisites for the cross-specialty comparison in topic 07,
where time zero and severity adjustment reappear under different names. Outcome 3
connects to module 08 (real-world evidence), because renal endpoints are common in
drug-safety work. Outcome 6 is shared with oncology, where competing risks appear as
other-cause mortality. Outcome 8 connects to module 14 (device regulation), because a
sepsis alert is usually regulated as software.

If you retain only one outcome, retain outcome 1. Every other error is visible in a
table of values; a time-zero error is invisible because it produces a plausible
number and a plausible area under the receiver operating characteristic curve.

## 3. The core mental model

A critical-care analysis is a chain of timestamped measurements, and every join is a place where the time or the unit can be wrong.

```text
  Emergency department / ward
        |
        |  culture at 18:05        antibiotic at 19:10
        v
  [1] SUSPICION TIME  = earliest of the paired culture and antibiotic
        |                 pairing rule: culture->antibiotic <=72 h
        |                               antibiotic->culture <=24 h
        v
  [2] ORGAN DYSFUNCTION WINDOW  = -48 h to +24 h around suspicion
        |                 respiration, coagulation, liver,
        |                 cardiovascular, CNS, renal -> SOFA
        v
  [3] SUPPORT .......... vasopressors, ventilation, renal replacement
        |                 dose normalised to norepinephrine equivalents
        v
  [4] TRAJECTORY ....... fluid balance, lactate clearance, urine output
        |
        v
  [5] OUTCOME .......... death | discharge alive | transfer
                          these compete; they are not independent
```

Read the diagram as a set of joins with timestamps, not a set of stages. The analytic
table is built by joining [1] to [2] to [3] to [5]. Every join has a key and a time,
and the most common defect is that the join is done on the patient identifier without
regard to the time, so a creatinine measured on day 4 is attached to a baseline
feature defined at hour 0.

### 3.1 Time zero at suspicion

Sepsis-3, published by Singer and colleagues in the Journal of the American Medical
Association in 2016, defines sepsis as life-threatening organ dysfunction caused by a
dysregulated host response to infection, operationalised as an acute increase in
total SOFA score of at least 2 points. The operational definition of suspected
infection is a body-fluid culture and an antibiotic administered close together in
time. The pairing rule is asymmetric: if the culture comes first, the antibiotic must
follow within 72 hours; if the antibiotic comes first, the culture must follow within
24 hours. The onset, or time zero, is the earlier of the two events.

Three consequences follow. First, time zero can precede the ICU admission by many hours, and in some cohorts it
precedes the hospital admission, so a cohort restricted to ICU stays truncates the early window for patients who
deteriorated on the ward.

Second, the pairing rule creates a dependency between two tables that are usually
owned by different systems: microbiology and pharmacy. A missing administration
timestamp does not mean no antibiotic; it may mean the drug was given on the ward and
charted in a different system.

Third, the window over which organ dysfunction is assessed extends 24 hours after time zero. That is correct for
defining the label retrospectively and fatal for prediction if the same window builds features. The label may look
forward; the features may not.

```text
  culture ------------> antibiotic
     t_c                   t_a
     |<------ <= 72 h ------>|        valid pair, onset = t_c

  antibiotic ---------> culture
     t_a                   t_c
     |<------ <= 24 h ------>|        valid pair, onset = t_a

  culture .... 80 h .... antibiotic   invalid pair, not a suspicion event
```

### 3.2 The ICU data cascade

Critical-care data is not recorded; it is cascaded. Each layer loses information and adds latency, so a model that
ignores the cascade treats a charted value as a measurement.

| Layer | What happens | Latency | Typical defect |
|---|---|---|---|
| Device | Monitor, ventilator, infusion pump, laboratory analyser | Seconds | Calibration drift, artefact, wrong unit at source |
| Chart | Nurse validates or edits the value into the record | Minutes to hours | Editing, carry-forward, default values |
| Observation table | Value mapped to a concept identifier and unit | Minutes | Unit conversion, duplicate rows, null flavours |
| Hourly aggregate | Values summarised into bins | Up to an hour | Mean of a maximum, minimum of a mean |
| Model feature | Aggregate transformed for the model | Batch | Window leakage, imputation that encodes the future |

The clinical consequence is that two numbers that look identical are not. A systolic pressure of 88 mmHg charted by
a nurse after review and one read automatically from an arterial line are different measurements with different error
properties, and a pipeline that pools them has created a variable no clinician uses.

### 3.3 Competing risks

In ICU data, death and discharge alive compete: a patient who dies cannot later be discharged, and a patient
discharged alive cannot die in the ICU. Standard time-to-event analysis treats the competing event as censoring, which
assumes a patient censored at discharge would have had the same subsequent hazard as one still in the ICU. That
assumption is false, and in a direction that matters: the patients discharged earliest are usually the least sick.

The practical rule is to state the question. If the question is "what is the
probability of discharge alive by day 10", the estimand is a cumulative incidence
function, and death is a competing event. If the question is "among patients still in
the ICU, what is the instantaneous hazard of discharge", the estimand is a
cause-specific hazard, and death is a censoring event. Both are legitimate; they
answer different questions and produce different numbers. What is not legitimate is to
report one and describe it as the other.

### 3.4 What the model is allowed to see

The final mental model is a wall at the decision time. Features to the left are
admissible; features to the right are leakage.

```text
  admissible ....................... | prediction | ......... inadmissible
  demographics, comorbidities,       |   time     |  later SOFA components,
  vitals and labs before t0,         |            |  culture result, lactate
  antibiotics already given,         |            |  after t0, discharge, death
  support status at t0               |            |
```

The wall is not the same as the suspicion window. The window defines the label; the
wall defines the features. A model trained with features drawn from the full label
window will appear excellent and will be useless, because at the moment of the alert
those features do not exist.

## 4. Definitions and vocabulary

Every term below is used later. The common-misuse column is the part worth
memorising, because the misuse is what appears in real analytic tables.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Sepsis | Life-threatening organ dysfunction from a dysregulated host response to infection | Used as a synonym for bacteraemia | Clinical syndrome; no single code |
| Septic shock | Sepsis with vasopressor requirement to maintain mean arterial pressure at least 65 mmHg and lactate above 2 mmol/L after adequate fluids | Applied before fluids are given | Clinical definition |
| Suspected infection | Culture plus antibiotic within the pairing window | Treated as a diagnosis of infection | Operational rule |
| Suspicion time | Earlier of the paired culture and antibiotic | Set to ICU admission or to the first alert | Timestamp |
| SOFA | Sequential Organ Failure Assessment, six organ systems scored 0–4 | Computed over the whole stay | Ordinal 0–24 |
| Baseline SOFA | Organ dysfunction present before the acute episode | Assumed zero in every patient | Ordinal 0–24 |
| Delta SOFA | Current SOFA minus baseline SOFA | Omitted, so chronic dysfunction is counted as acute | Ordinal points |
| qSOFA | Quick SOFA: respiratory rate at least 22, altered mentation, systolic pressure at most 100 | Used as a diagnostic criterion | Count 0–3 |
| APACHE IV | Acute Physiology and Chronic Health Evaluation, version IV | Used as a real-time predictor | Probability; 0–299 points |
| SAPS II | Simplified Acute Physiology Score, version II | Compared across versions | Points; probability |
| KDIGO AKI | Kidney Disease: Improving Global Outcomes acute kidney injury definition | Applied with an arbitrary baseline creatinine | Stage 1–3 |
| Baseline creatinine | Reference creatinine before the acute episode | Taken as the first value on admission | mg/dL or micromol/L |
| Urine output criterion | Less than 0.5 mL/kg/h for 6 hours or more | Computed without a weight or with the wrong weight | mL/kg/h |
| Norepinephrine equivalent | Composite vasopressor dose normalised to norepinephrine | Compared across formulas without saying which | Micrograms per kg per minute |
| Vasopressin | Non-catecholamine vasopressor, usually fixed dose | Recorded in the same units as norepinephrine | Units per minute |
| Phenylephrine | Pure alpha agonist | Normalised with a factor that varies by source | Micrograms per kg per minute |
| Dopamine | Catecholamine with dose-dependent effects | Assumed equipotent across dose ranges | Micrograms per kg per minute |
| Fluid balance | Cumulative intake minus output | Reported without a time window or a denominator | mL; percent body weight |
| Cumulative fluid balance | Running total from admission | Reset when the patient changes unit | mL |
| Fluid overload | Cumulative balance as a percentage of body weight | Used interchangeably with positive balance | Percent |
| Predicted body weight | Weight derived from height and sex for ventilation | Replaced by actual weight | kg |
| Tidal volume | Volume delivered per breath | Reported per actual rather than predicted weight | mL; mL/kg |
| Driving pressure | Plateau pressure minus positive end-expiratory pressure | Confused with peak pressure | cm H2O |
| PEEP | Positive end-expiratory pressure | Ignored when computing PaO2/FiO2 | cm H2O |
| PaO2/FiO2 | Arterial oxygen partial pressure divided by fraction of inspired oxygen | Computed without stating the PEEP or support | Ratio; mm Hg |
| ARDS | Acute respiratory distress syndrome, Berlin definition | Diagnosed without bilateral opacities or a PEEP requirement | Clinical definition |
| Competing risk | An event that prevents the event of interest | Treated as ordinary censoring | Statistical concept |
| Cause-specific hazard | Instantaneous hazard of the event among those still at risk | Reported as a cumulative probability | Hazard |
| Cumulative incidence | Probability of the event by time t in the presence of competing risks | Computed as one minus Kaplan-Meier | Probability |
| Immortal time | Person-time before an exposure could occur | Credited to the exposed group | Person-time |
| Treatment paradox | Sicker patients receive more treatment, so treatment looks harmful | Adjusted away with severity scores alone | Confounding structure |
| Informative missingness | Whether a value is measured depends on the patient's state | Treated as missing at random | Measurement process |
| Alarm burden | Alerts per patient-day and the number needed to alert | Reported as sensitivity alone | Count per patient-day |
| CMS SEP-1 | Centers for Medicare and Medicaid Services Severe Sepsis and Septic Shock Early Management Bundle | Used as a surrogate for model quality | Quality measure |
| MIMIC-IV | Medical Information Mart for Intensive Care, version IV | Assumed to be population-representative | Single-centre database |
| eICU-CRD | Electronic ICU collaborative research database | Assumed to have uniform variable coverage | Multi-centre database |

### 4.1 Units that change the answer

Four unit families account for most of the silent errors in critical-care extracts.

**Creatinine.** Milligrams per decilitre and micromoles per litre differ by a factor
of 88.4. A value of 2.4 mg/dL is 212 micromol/L. The KDIGO thresholds are stated in
both, but a pipeline that reads the number without the unit will classify a normal
value as dialysis-level.

**Lactate.** Millimoles per litre and milligrams per decilitre differ by a factor of
9.01. The septic shock threshold of 2 mmol/L is 18 mg/dL, and the SEP-1 shock trigger
of 4 mmol/L is 36 mg/dL. A value of 3.2 mmol/L is 28.8 mg/dL.

**Bilirubin.** Milligrams per decilitre and micromoles per litre differ by 17.1. The
SOFA liver thresholds are conventionally stated in mg/dL.

**Vasopressor dose.** Micrograms per kilogram per minute, micrograms per minute,
milligrams per hour, and units per minute for vasopressin are all in use. A rate of
0.08 micrograms per kg per minute in a 70 kg patient is 5.6 micrograms per minute, or
0.336 mg per hour. Reading the hourly figure as a per-kilogram figure multiplies the
apparent dose by 4.2.

### 4.2 What counts as an event

The single most consequential coding decision in a critical-care analysis is what
counts as an event and when it occurred. The table is the minimum to record for each
endpoint.

| Endpoint | Event | Censoring or competing event | Ascertainment source |
|---|---|---|---|
| In-hospital mortality | Death before discharge | Discharge alive | EHR, vital records |
| 28-day mortality | Death within 28 days | Alive at day 28 | Linkage required |
| Sepsis onset | Suspicion plus delta SOFA at least 2 | None | Derived |
| AKI | KDIGO creatinine or urine criterion | Discharge, death | Labs, urine chart |
| Ventilator-free days | Alive and free of ventilation | Death is a competing event scored as zero | Ventilator records |
| ICU length of stay | Discharge from ICU | Death in ICU | Unit records |
| Deterioration | Escalation, code, or unplanned transfer | Discharge | Workflow records |

> **Note:** "Ventilator-free days" is not a duration. It is a composite that assigns
> zero to patients who die, which means it embeds a value judgement about death and
> cannot be analysed as a simple time-to-event outcome.

## 5. The mechanics, step by step

This section is the procedure for turning raw critical-care records into an analysable cohort. It is written as ten
steps. Each step states its input, its output, the reason it exists, its branch points, and its edge cases. A team
that can execute these ten steps in order, and document the choices, can defend the resulting dataset to a clinical
reviewer, a regulator, and a replication attempt.

The procedure assumes the question is prognostic or comparative in a defined critically ill population. Screening or
resource-allocation questions add requirements (capacity constraints, queueing rules) that are noted but not
developed.

### Step 1 — Fix suspicion time

**Input.** Microbiology culture collection timestamps, antimicrobial administration timestamps, and the encounter
identifiers that let them be joined.

**Output.** One suspicion event per patient per infectious episode, with time zero, the pair that established it, and
the pairing rule applied.

**Rationale.** Sepsis-3 defines onset as the earlier of a qualifying culture and antibiotic pair. Everything
downstream inherits that timestamp: the SOFA window, the feature wall, the antibiotic clock, and the comparison
between cohorts. A pipeline that anchors onset at ICU admission or at the first alert has changed the question
without saying so.

**Procedure.**

1. Extract every culture collection event with its specimen type and collection time, not its result time. A culture
   that grows an organism 48 hours later was collected at the bedside at a known moment.
2. Extract every antimicrobial administration event with the actual administration time. If only the order time
   exists, record that administration time is missing rather than substituting the order.
3. Join cultures to antibiotics within the same encounter. Apply the asymmetric rule: culture first allows an
   antibiotic up to 72 hours later; antibiotic first allows a culture up to 24 hours later.
4. Take the earlier event of each valid pair as time zero. Where several pairs qualify, take the earliest qualifying
   time zero for the episode and record how many pairs contributed.
5. Record the source system and the timestamp precision. A midnight default is not a measurement.

**Decision table for the pairing rule.** More than two outputs exist, so the branch must be explicit.

| Evidence pattern | Classification | Time zero | Consequence if mishandled |
|---|---|---|---|
| Culture then antibiotic within 72 h | Valid pair | Culture time | Onset too late if antibiotic time is used |
| Antibiotic then culture within 24 h | Valid pair | Antibiotic time | Onset too late if culture time is used |
| Culture then antibiotic at 80 h | Not a pair | None | A delayed culture is mistaken for suspicion |
| Culture with no antibiotic within 72 h | Not a pair; possible surveillance | None | Surveillance cultures inflate incidence |
| Antibiotic with no culture within 24 h | Not a pair; empirical treatment | None | Empiric treatment without cultures is invisible |
| Two cultures and one antibiotic | One pair, earliest culture | Earliest qualifying culture | Duplicate episodes inflate the denominator |
| Antibiotic charted in another system | Valid if timestamps reconcile | Earlier event | Missing time looks like no antibiotic |

**Branch point.** If the analysis is a prediction task rather than a retrospective label, time zero must be the
moment the model would run, which is earlier than the retrospective suspicion time. Two timestamps are then needed:
the evaluation time zero for the label and the prediction time for the features, and conflating them is the most
common leakage mechanism in sepsis modelling.

**Edge case.** Prophylactic antibiotics. A surgical patient receives cefazolin before incision and has a surveillance
culture later. The pair can satisfy the rule and create a sepsis episode that never existed, so exclude prophylaxis
by route, timing, and drug list, and treat cultures collected after death or transfer as invalid.

### Step 2 — Compute SOFA inside the stated window

**Input.** Six components, each with a timestamp, value, and unit: arterial oxygen partial pressure (PaO2) and
fraction of inspired oxygen (FiO2) with support status, platelets, bilirubin, mean arterial pressure (MAP),
vasopressor dose, Glasgow Coma Scale (GCS), creatinine, and urine output.

**Output.** Six component scores, a total score, the source value per component, and the scoring window.

**Rationale.** SOFA is the operational definition of organ dysfunction in Sepsis-3 and the severity adjustment in
most critical-care research. It is not a continuous measurement; it is a rule applied to the worst value in a window.
The rule and the window are part of the score.

**Procedure.**

1. Define the window relative to time zero. Sepsis-3 assessed variables from 48 hours before to 24 hours after the
   onset of suspected infection; record the window you use.
2. Respiration: compute PaO2 divided by FiO2. Confirm FiO2 is a fraction between 0.21 and 1.00 and that PaO2 is
   arterial. Score 0 at a ratio of at least 400, 1 below 400, 2 below 300, 3 below 200 with respiratory support, 4
   below 100 with respiratory support.
3. Coagulation: platelets in units of 10^9/L. Score 0 at 150 or more, 1 below 150, 2 below 100, 3 below 50, 4 below
   20.
4. Liver: bilirubin in mg/dL. Score 0 below 1.2, 1 at 1.2–1.9, 2 at 2.0–5.9, 3 at 6.0–11.9, 4 at 12.0 or more.
5. Cardiovascular: score 0 at a mean arterial pressure of at least 70 mmHg, 1 below 70, 2 for dopamine at most 5 or
   any dobutamine, 3 for dopamine above 5 or epinephrine or norepinephrine at most 0.1, 4 for dopamine above 15 or
   epinephrine or norepinephrine above 0.1, all in micrograms per kg per minute.
6. Central nervous system: GCS 15 scores 0, 13–14 scores 1, 10–12 scores 2, 6–9 scores 3, below 6 scores 4.
7. Renal: creatinine in mg/dL. Score 0 below 1.2, 1 at 1.2–1.9, 2 at 2.0–3.4, 3 at 3.5–4.9 or urine output below 500
   mL per day, 4 at 5.0 or more or urine output below 200 mL per day. Take the worse of creatinine and urine output.
8. Sum the six components and store the components, not only the sum.

**Branch point.** If a component is missing, do not impute zero. Zero means "no dysfunction measured", a clinical
statement rather than an absence of data. Either carry the component as missing under a documented partial-score
rule, or exclude the patient from score-based eligibility and report the exclusion.

**Edge case.** Sedation. GCS in a sedated patient measures the sedative, not the brain. Many studies substitute the
pre-sedation GCS or a Richmond Agitation-Sedation Scale estimate; the choice changes the neurological component by up
to four points, so record the convention. The baseline is the second edge case: Sepsis-3 assumes zero unless
pre-existing dysfunction is known, and for a patient with cirrhosis or chronic kidney disease zero is false. The
delta defines sepsis, so a baseline of 12 with an acute score of 13 is a delta of 1 and does not meet the definition.

### Step 3 — Normalise units before any comparison

**Input.** Laboratory values and vital signs with their source units and the interface that delivered them.

**Output.** A value in one canonical unit per concept, with the original value, original unit, conversion factor, and
conversion timestamp retained.

**Rationale.** Every threshold in Steps 2 and 7 is unit-bound. A pipeline that stores numbers without units cannot
detect a conversion error, and the error is silent because the number remains plausible.

**Procedure.**

1. Maintain a conversion table keyed by concept and unit, not by column name. A column called `creatinine` can arrive
   in either unit depending on the sending laboratory.
2. Convert creatinine between mg/dL and micromol/L with a factor of 88.4; lactate between mmol/L and mg/dL with 9.01;
   bilirubin between mg/dL and micromol/L with 17.1; glucose between mg/dL and mmol/L with 18.0.
3. Convert weight between pounds and kilograms with 2.20462, and convert vasopressor rates to micrograms per kg per
   minute, or units per minute for vasopressin, recording the weight and the scale unit used for each.
4. Apply a physiological range check after conversion. A creatinine above 20 mg/dL, a lactate above 30 mmol/L, or a
   bilirubin above 60 mg/dL is flagged for review, and the original value and unit are stored beside the converted one.

**Decision table for a value that fails a range check.**

| Pattern | Likely cause | Action | Output |
|---|---|---|---|
| Value is 88.4 times the expected magnitude | Micromol/L read as mg/dL | Convert, flag | Corrected value plus flag |
| Value is 0.011 times the expected magnitude | mg/dL read as micromol/L | Convert, flag | Corrected value plus flag |
| Value is exactly 10 or 100 times expected | Decimal entry error | Quarantine | Missing with reason |
| Value is plausible but the unit field is blank | Unknown unit | Quarantine | Missing with reason |
| Value is plausible and the unit matches the concept | None | Accept | Value |

**Edge case.** Point-of-care and central laboratory values for the same analyte can differ systematically, because
blood gas analysers use a different calibration. Do not pool them without recording the source device.

### Step 4 — Normalise vasopressor dose to norepinephrine equivalents

**Input.** Every vasoactive infusion with drug, rate, concentration, weight basis, start time, and stop time.

**Output.** A time-varying norepinephrine equivalent (NEE) dose in micrograms per kg per minute, plus the formula version and component doses.

**Rationale.** A cohort that compares "vasopressor dose" across patients on different agents is comparing notation.
Norepinephrine equivalents put agents on one scale so severity can be compared and dose-response can be estimated.
The conversion is a pharmacological approximation, not a measurement.

**Procedure.**

1. Convert each infusion to micrograms per kg per minute using the weight recorded at the time of the infusion, not a
   later weight.
2. Convert vasopressin to units per minute and keep it separate until the formula is applied.
3. Apply a named formula. The scoping review by Goradia and colleagues in the Journal of Critical Care in 2021
   proposes NEE equal to norepinephrine plus epinephrine plus phenylephrine divided by 10 plus dopamine divided by 100
   plus metaraminol divided by 8 plus vasopressin multiplied by 2.5 plus angiotensin II multiplied by 10.
4. Record that a later update by Kotani and colleagues in Critical Care in 2023 uses a phenylephrine factor of 0.06
   rather than one tenth and a different angiotensin II scaling. The two formulas disagree by 40 percent on a
   phenylephrine-only regimen.
5. Compute NEE as a time-varying variable if the question is about dose trajectory, and as a maximum if the question
   is about severity. Store the component doses, because a single NEE number cannot be decomposed later.

**Branch point.** If the analysis compares centres and the centres use different formulas, the comparison is partly a
comparison of formulas. Harmonise to one formula and report the sensitivity of the result to the other.

**Edge case.** Dobutamine is an inotrope, not a vasopressor, and is not in most NEE formulas. Adding it as if it were
a pressor overstates vasopressor requirement. A second edge case is peripheral vasopressors: the Surviving Sepsis
Campaign 2021 guideline recommends, as a new recommendation, starting vasopressors peripherally rather than delaying
for central access. Peripheral administration is often charted in a different system, so a cohort built from
central-line records will miss the early, low-dose period.

### Step 5 — Derive fluid balance and its denominator

**Input.** Intake events (intravenous fluids, drugs and diluents, enteral feed, flushes) and output events (urine,
ultrafiltrate, stool, drains, estimated insensible losses), plus a weight and a time window.

**Output.** Cumulative intake, cumulative output, cumulative balance in millilitres, and balance as a percentage of a
stated body weight.

**Rationale.** Fluid balance is one of the few ICU variables that is both a treatment and a harm. A positive balance
of 10 percent of body weight is associated with worse outcomes in observational studies, but the association depends
entirely on the denominator and the window, so a balance without either is not interpretable.

**Procedure.**

1. Define the window from ICU admission, not from hospital admission, unless the question is about the whole hospital
   stay.
2. Include every intake category. Drug diluents and flushes are commonly omitted and can account for several hundred
   millilitres per day.
3. Include every output category that is charted. Insensible losses are usually estimated or omitted; say which.
4. Compute the running balance and store it per calendar day and cumulatively.
5. Divide by the weight recorded at the same time as the balance window. Record whether the weight is measured,
   estimated, or from a bed scale.
6. Report the percentage with its threshold source; the 10 percent figure is a convention from the fluid-overload literature, not a regulatory cut point.
7. Handle renal replacement therapy: ultrafiltrate is output, but net removal is set by the machine and may already be reflected in the fluid balance, so double counting is common.

**Branch point.** If the patient changes unit or ward, the balance may reset in the source system. Either reconstruct
from event-level data or record the reset and do not sum across it.

**Edge case.** Weight change during the stay is mostly fluid. Using the admission weight as the denominator for a
day-5 balance understates the percentage; using the current weight understates the numerator. State the choice.

### Step 6 — Derive ventilation variables

**Input.** Ventilator settings (tidal volume, positive end-expiratory pressure, fraction of inspired oxygen, plateau
pressure), arterial blood gases, height, and sex.

**Output.** Tidal volume per kilogram of predicted body weight, driving pressure, PaO2/FiO2 ratio, and the support
status at the time of the gas.

**Rationale.** Lung-protective ventilation is defined per predicted body weight, not per actual weight, because lung
size scales with height and sex rather than with body fat. A tidal volume of 500 mL is protective in a tall man and
injurious in a short woman, and the number alone cannot tell you which.

**Procedure.**

1. Compute predicted body weight from height and sex. The ARDS Network formula is 50 plus 0.91 times height in
   centimetres minus 152.4 for men, and 45.5 plus 0.91 times the same difference for women.
2. Divide the set tidal volume by predicted body weight. The lung-protective target is 6 mL/kg, with a range of 4 to
   8 mL/kg in common protocols.
3. Compute driving pressure as plateau pressure minus positive end-expiratory pressure. A value above 15 cm H2O is
   associated with harm in observational analyses.
4. Compute PaO2/FiO2 only with a simultaneous arterial gas and a recorded FiO2. State the positive end-expiratory
   pressure and the support mode.
5. Record whether the patient is invasively ventilated, on non-invasive support, on high-flow nasal oxygen, or
   breathing spontaneously. The SOFA respiratory component depends on support status, and a venous gas is not a PaO2.

**Branch point.** If height is missing, predicted body weight cannot be computed. Substituting actual weight defeats
the purpose. Either obtain the height from another source or mark the ventilation variable missing.

**Edge case.** Height recorded in inches. A height of 65 read as centimetres gives a predicted body weight of 45.5 kg
for a woman who is actually 165 cm; the tidal volume target falls by 20 percent and the patient is underventilated.

### Step 7 — Stage AKI with an explicit baseline

**Input.** Serum creatinine values with timestamps, urine output with a weight, and any pre-admission creatinine.

**Output.** AKI status, KDIGO stage, the criterion that established it, and the baseline used.

**Rationale.** The KDIGO 2012 definition has three arms: a rise in serum creatinine of at least 0.3 mg/dL within 48
hours, a rise to at least 1.5 times baseline known or presumed within the prior 7 days, or urine output below 0.5
mL/kg/h for 6 hours. The baseline is the hard part, and the choice of baseline changes the stage.

**Procedure.**

1. Define baseline creatinine by a pre-specified hierarchy: a measured outpatient value within 365 days, then the
   lowest value during the admission if the patient is volume-resuscitated, then a back-calculated value assuming an
   estimated glomerular filtration rate (eGFR) of 75 mL/min/1.73 m2. Record which rung was used.
2. Apply the 48-hour delta criterion, which needs two values within 48 hours, and the 7-day ratio criterion, which
   needs a baseline within 7 days.
3. Apply the urine criterion. Stage 1 is below 0.5 mL/kg/h for 6–12 hours, stage 2 is below 0.5 for at least 12 hours,
   stage 3 is below 0.3 for at least 24 hours or anuria for at least 12 hours.
4. Take the worst stage across creatinine and urine criteria, and record the criterion that produced it. A stage 3 by
   urine output and a stage 3 by creatinine are different phenotypes with different risk.
5. Do not apply the creatinine criterion to a patient on renal replacement therapy without recording that the
   therapy, not the creatinine, defines stage 3.

**Decision table for an unknown baseline.**

| Situation | Baseline rule | Stage consequence | Risk |
|---|---|---|---|
| Outpatient creatinine within 365 days | Use the measured value | Most defensible | May reflect a different clinical state |
| No outpatient value, admission creatinine low | Use the lowest inpatient value | May understage established injury | Resuscitation dilutes creatinine |
| No baseline at all | Back-calculate from eGFR 75 | May overstage chronic kidney disease | The assumed eGFR is a population average |
| Known chronic kidney disease | Use the most recent stable value | Avoids labelling chronic disease as acute | Requires documentation |
| On renal replacement therapy | Stage 3 by definition | Uniform | Loses acute versus chronic dialysis |

**Edge case.** Creatinine is a late marker, and severe tubular injury can coexist with a normal value because of low
muscle mass, dilution, or assay delay. A creatinine-only definition undercounts AKI in frail patients.

### Step 8 — Treat discharge as a competing risk

**Input.** ICU admission time, discharge time and destination, death time, and transfer time.

**Output.** A time-to-event dataset with an event indicator that distinguishes death from discharge alive, and an
analysis that states which estimand it reports.

**Rationale.** Discharge and death compete. A naive Kaplan-Meier estimate of time-to-discharge treats death as
censoring, which assumes a dead patient could later be discharged, so the curve overestimates discharge for the
sickest patients and can make a unit look efficient because its patients died.

**Procedure.**

1. Define the event of interest and the competing event before looking at the data. "Discharge alive" with death as
   competing is the usual choice for flow questions; "death" with discharge as competing is the usual choice for
   mortality questions.
2. Create a three-level status variable: event, competing event, censored.
3. For a cumulative incidence question, use a cumulative incidence function (Fine-Gray subdistribution) and report
   the competing event's incidence alongside.
4. For an aetiological question, use cause-specific hazards and state that the estimate is a hazard among those still
   in the unit.
5. Never compute one minus Kaplan-Meier for a competing-risk question, because the complement estimates the probability in a hypothetical world without the competing event. Report the number in each terminal state and the median follow-up.

**Branch point.** If discharge destination is missing, the patient may have died after transfer. Link to vital
records or treat the transfer as censoring and report the proportion transferred.

**Edge case.** Readmission. A patient discharged alive and readmitted later contributes two ICU episodes. If the unit
of analysis is the patient, the second episode is not independent, and the discharge event is not terminal.

### Step 9 — Audit the assembled dataset

**Input.** The analytic table and the code that built it.

**Output.** An audit report with counts, cross-tabs, recomputation checks, and a list of resolved and unresolved
discrepancies.

**Rationale.** Every step above can fail silently, and the audit is the only defence and is cheap relative to the cost of a retracted model.

**Procedure.**

1. Recompute suspicion time from raw culture and antibiotic tables and compare with the stored value. Report the
   disagreement count.
2. Recompute SOFA from components and compare with the stored total. Report the disagreement rate and the components
   that disagree.
3. Recompute AKI stage under two baseline rules and report the stage shift.
4. Recompute NEE under both published formulas and report the difference distribution.
5. Check that no feature timestamp is after the prediction time. Sample rows and inspect.
6. Cross-tabulate the unit of every analyte against the expected unit. Any unexpected unit is a conversion risk.
7. Count missingness by component and subgroup, distinguishing not measured from measured and missing, and produce a cohort flow diagram with every exclusion and its count.

**Branch point.** If the SOFA recomputation disagreement exceeds a pre-set threshold, stop and fix the derivation. A
model built on a score that disagrees with its own components is measuring the disagreement.

**Edge case.** Duplicate ICU episodes from a unit transfer. Two rows for one patient in one admission can double the
cohort. Deduplicate on encounter and time, and record which row won.

### Step 10 — Freeze, version, and document

**Input.** The audited dataset and the code that built it.

**Output.** A versioned extract with a data dictionary, a changelog, and a reproducibility statement.

**Rationale.** Critical-care definitions change. Sepsis-3 replaced an earlier consensus; KDIGO is under revision; the
Surviving Sepsis Campaign updates on a multi-year cycle. A dataset without a version cannot be compared with the next
extract.

**Procedure.**

1. Pin the source extract date, the Sepsis-3 implementation, the SOFA window, the KDIGO version, the NEE formula,
   and the code commit.
2. Write a data dictionary with every derived field, its rule, and its source fields.
3. Record every exclusion with a count and a reason, store the audit outputs with the dataset, and state the intended use and limits in a model or dataset card.

**Edge case.** A guideline update arrives mid-project. Freeze the analysis on the version in force at the extract date
and run a sensitivity analysis under the new version. Do not silently re-stage the cohort.

## 6. Worked example

A 71-year-old woman, 70 kg and 165 cm, presents to the emergency department at 14:20 with cough, fever, and confusion.
Her blood culture is drawn at 18:05. The antibiotic is ordered at 18:40 and administered at 19:10. A lactate at 18:20
is 3.2 mmol/L. She is intubated at 20:15 for hypoxaemia and admitted to the ICU at 21:40. A creatinine of 0.9 mg/dL
is on file from eight months earlier, and her previous records carry no known chronic organ dysfunction.

### 6.1 Fixing time zero

The culture precedes the antibiotic by 65 minutes, inside the 72-hour culture-first pairing window. The pair is valid
and time zero is the culture time, 18:05. If the pipeline used the antibiotic time, onset would move 65 minutes later.
If it used ICU admission, onset would move 3 hours 35 minutes later, and every value drawn before 21:40 would fall
outside a window anchored at the ICU. The 65-minute difference sounds small. It is not, because the antibiotic clock
and the early-warning window are both measured from this point.

The SEP-1 bundle uses a different clock. The Centers for Medicare and Medicaid Services Severe Sepsis and Septic Shock
Early Management Bundle measures its 3-hour and 6-hour elements from the presentation of severe sepsis, which is a
documented clinical construct rather than the suspicion pair. In this case the lactate at 18:20 and the antibiotic at
19:10 are both inside three hours of 18:05, so the case would meet the 3-hour elements if the documentation supported
the presentation time. A pipeline that substitutes the suspicion time for the SEP-1 presentation time will report
bundle compliance that the measure itself does not compute. Two clocks, two numerators, one patient.

### 6.2 Computing SOFA

The window runs from 48 hours before to 24 hours after 18:05, and the worst admissible value is selected for each
component.

| Component | Source value | Score | Why |
|---|---|---:|---|
| Respiration | PaO2 78 mmHg, FiO2 0.50, intubated | 3 | Ratio 156, below 200 with support |
| Coagulation | Platelets 88 ×10^9/L | 2 | Below 100 |
| Liver | Bilirubin 1.6 mg/dL | 1 | 1.2–1.9 |
| Cardiovascular | MAP 58 on norepinephrine 0.08 | 3 | Norepinephrine at most 0.1 |
| Central nervous system | GCS 13 | 1 | 13–14 |
| Renal | Creatinine 2.4 mg/dL; urine 400 mL/24h | 3 | Urine output below 500 mL/day |
| **Total** | | **13** | |

The respiratory ratio is 78 divided by 0.50, which is 156. The renal component takes the worse of creatinine and urine
output: creatinine 2.4 scores 2, and urine output below 500 mL per day scores 3. The total is 3 + 2 + 1 + 3 + 1 + 3 =
13. With an assumed baseline SOFA of zero, the delta is 13, which meets the Sepsis-3 criterion of at least 2. With a
known baseline of 4 from chronic liver disease, the delta is 9, which still meets it. The case is sepsis either way,
but the severity differs, and a model trained on absolute scores learns the chronic disease.

### 6.3 Staging AKI with an unknown baseline

The creatinine series is short and consequential.

| Time | Creatinine (mg/dL) | Setting |
|---|---|---:|
| 8 months earlier | 0.9 | Outpatient, stable |
| 14:50 on arrival | 1.1 | Emergency department |
| 06:00 next day | 2.4 | ICU, 16 hours after arrival |

The creatinine is 2.4 mg/dL and the eight-month value is 0.9 mg/dL. The ratio is 2.4 / 0.9 = 2.67, in the KDIGO stage
2 band of 2.0–2.9 times baseline. The 48-hour criterion is met separately: the rise from 1.1 to 2.4 is 1.3 mg/dL over
15 hours 10 minutes, and any rise of at least 0.3 mg/dL within 48 hours qualifies. The case therefore meets two arms
of the definition, and the measured outpatient value takes priority over the admission value as the baseline.

The urine criterion is worse. Urine output is 400 mL over 24 hours in a 70 kg patient:

```text
400 mL / 70 kg / 24 h = 0.238 mL/kg/h
```

That is below 0.3 mL/kg/h for at least 24 hours, which is KDIGO stage 3. Taking the worst criterion, the patient is
stage 3.

If the eight-month value did not exist, the common back-calculation assumes an estimated glomerular filtration rate of
75 mL/min/1.73 m2. For a 71-year-old woman, the four-variable Modification of Diet in Renal Disease equation gives a
baseline of about 0.76 mg/dL, and the ratio becomes 2.4 / 0.76 = 3.16, which is stage 3 by creatinine alone. The stage
moves from 2 to 3 on the choice of baseline. That is the size of the assumption.

### 6.4 Vasopressor dose, fluid balance, and ventilation

At hour 6, vasopressin 0.03 units per minute is added to norepinephrine 0.08 micrograms per kg per minute. Using the
Goradia 2021 formula:

```text
NEE = 0.08 + 2.5 * 0.03 = 0.08 + 0.075 = 0.155 ug/kg/min
```

Suppose a comparison cohort used phenylephrine 1.0 microgram per kg per minute alone. The Goradia formula divides
phenylephrine by 10 and returns 0.10. The Kotani 2023 formula multiplies by 0.06 and returns 0.06. The same regimen is
0.06 or 0.10 depending on the formula version, a 40 percent difference. A multicentre comparison that does not
harmonise the formula is partly comparing formulas.

Fluid balance over three days:

| Day | Intake (mL) | Output (mL) | Daily balance (mL) | Cumulative (mL) |
|---|---:|---:|---:|---:|
| 1 | 4,850 | 1,390 | +3,460 | +3,460 |
| 2 | 3,980 | 1,610 | +2,370 | +5,830 |
| 3 | 3,240 | 1,120 | +2,120 | +7,950 |

The cumulative balance is 7,950 mL. As a fraction of 70 kg, that is 7,950 / 70,000 = 11.4 percent, above the 10
percent convention for fluid overload. The denominator is doing as much work as the numerator.

Ventilation adds a third derived variable. Predicted body weight for a woman of 165 cm is 45.5 + 0.91 × (165 − 152.4) =
45.5 + 11.47 = 56.97 kg. A set tidal volume of 500 mL is therefore 500 / 56.97 = 8.8 mL/kg, above the 6 mL/kg
lung-protective target, which would be 342 mL. Plateau pressure is 26 cm H2O and positive end-expiratory pressure is 8
cm H2O, so driving pressure is 18 cm H2O, above the 15 cm H2O threshold associated with harm. None of these three
numbers appears in the record; all three are derived, and each depends on a height and a unit.

### 6.5 Two diagnostics that reveal a broken analysis

The first diagnostic is a unit check on creatinine. The laboratory interface sends micromoles per litre. A pipeline
that expects milligrams per decilitre stores 2.4 × 88.4 = 212.2 for this patient. The SOFA renal component then reads
212.2 mg/dL, scores 4 instead of 2, and the total becomes 15 instead of 13. The KDIGO ratio also becomes absurd if the
baseline is in the other unit. The error adds 2 SOFA points and, more importantly, produces a value that no clinician
would accept. A range check at 20 mg/dL catches it immediately; without a range check, nothing in the pipeline
objects.

The second diagnostic is the SOFA window. Suppose the pipeline uses the worst value over the entire ICU stay rather
than the 48-hour-before to 24-hour-after window. Over the stay, platelets fall to 52, bilirubin rises to 7.4,
norepinephrine reaches 0.22, and the GCS falls to 9. The recomputed score is:

| Component | Window score | Whole-stay score | Change |
|---|---:|---:|---:|
| Respiration | 3 | 3 | 0 |
| Coagulation | 2 | 3 | +1 |
| Liver | 1 | 3 | +2 |
| Cardiovascular | 3 | 4 | +1 |
| Central nervous system | 1 | 3 | +2 |
| Renal | 3 | 3 | 0 |
| **Total** | **13** | **19** | **+6** |

The score rises by 6 points, or 46 percent, and three of six components change. In a cohort, the same window error
means the label and the features are computed from the same measurements, so any model that predicts the label from
the features is predicting backward from a shared source. The reported discrimination is inflated by construction, and
the model will not reproduce at a site that computes the score prospectively.

### 6.6 Vary one assumption

The weight is recorded as 70 kg. Suppose the scale reported pounds and the pipeline stored 154 as kilograms. The fluid
balance percentage becomes 7,950 / 154,000 = 5.2 percent, below the 10 percent threshold, so the fluid-overload
conclusion flips. The urine rate becomes 400 / 154 / 24 = 0.108 mL/kg/h, still below 0.3, so the AKI stage does not
change. One conclusion moves and one does not, which is the point: a unit error must be traced through each downstream
calculation rather than assumed to invalidate all of them.

The vasopressor dose is a subtler case. The recorded NEE stays at 0.08, because the numerator and denominator use the
same wrong weight. The absolute dose does not: 0.08 micrograms per kg per minute times 154 kg is 12.32 micrograms per
minute, against 5.6 micrograms per minute at 70 kg. The normalised number hides a 2.2-fold overdose. This is why the
weight used for a rate must be stored with the rate.

### 6.7 What the case does not establish

The case is a measurement worked example, not a treatment claim. It does not show that any antibiotic, fluid strategy,
or vasopressor was effective. It does not establish that the patient met SEP-1, because the bundle has documentation
requirements that are not in this narrative. It does not generalise to a different SOFA window, a different NEE
formula, or a different baseline rule.

What it does establish is narrower and more useful: every number in the case depends on a choice that is not in the
data. The table separates the choices.

| Choice | Value used here | Alternative that changes the answer | How to disclose it |
|---|---|---|---|
| Suspicion time | Culture 18:05 | Antibiotic 19:10 or ICU 21:40 | Store the pair and the rule |
| SOFA window | −48 h to +24 h | Whole stay | Window field per score |
| Baseline SOFA | 0 | 4 from chronic disease | Baseline field and source |
| AKI baseline | Measured 0.9 mg/dL | Back-calculated 0.76 mg/dL | Baseline rule field |
| NEE formula | Goradia 2021 | Kotani 2023 | Formula version field |
| Fluid denominator | 70 kg | 154 if pounds misread | Weight unit field |
| Discharge handling | Not analysed here | Competing risk | Estimand statement |

An analysis that discloses these seven choices can be reproduced and criticised. One that discloses none produces a
number that cannot be checked, which is the definition of an unfalsifiable claim.

## 7. Code, query, or calculation

Both blocks are standard-library Python and were executed as written with `python -I`. The first implements the six
SOFA components with a unit guard. The second implements KDIGO 2012 staging, baseline back-calculation, and two
norepinephrine-equivalent formulas. Assumptions: the inputs have already been selected as the worst values inside
the stated window; FiO2 is a fraction; vasopressin is in units per minute; all other vasopressors are in micrograms
per kg per minute.

### 7.1 SOFA components with a unit guard

```python
"""SOFA scoring with a window check and a unit sanity check. Standard library only."""

def resp(pao2, fio2, supported):
    if not 0.21 <= fio2 <= 1.0:
        raise ValueError(f"FiO2 must be a fraction, got {fio2}")
    r = pao2 / fio2
    return (4 if r < 100 and supported else 3 if r < 200 and supported
            else 2 if r < 300 else 1 if r < 400 else 0), r

def coag(p):
    return 4 if p < 20 else 3 if p < 50 else 2 if p < 100 else 1 if p < 150 else 0

def liver(b):
    return 4 if b >= 12 else 3 if b >= 6 else 2 if b >= 2 else 1 if b >= 1.2 else 0

def cardio(map_, ne):
    return 4 if ne > 0.1 else 3 if ne > 0 else 1 if map_ < 70 else 0

def cns(g):
    return 4 if g < 6 else 3 if g <= 9 else 2 if g <= 12 else 1 if g <= 14 else 0

def renal(cr, urine):
    if cr > 20.0:
        raise ValueError(f"creatinine {cr} mg/dL above range; suspect umol/L")
    return (4 if cr >= 5 or urine < 200 else 3 if cr >= 3.5 or urine < 500
            else 2 if cr >= 2 else 1 if cr >= 1.2 else 0)

resp_score, pf = resp(78.0, 0.50, True)
coag_score, liver_score = coag(88), liver(1.6)
cardio_score, cns_score = cardio(58, 0.08), cns(13)
renal_score = renal(2.4, 400)
total = resp_score + coag_score + liver_score + cardio_score + cns_score + renal_score

print(f"respiratory: P/F {pf:.1f} -> {resp_score}")
print(f"coagulation: 88 -> {coag_score}")
print(f"liver: 1.6 mg/dL -> {liver_score}")
print(f"cardiovascular: MAP 58, NE 0.08 -> {cardio_score}")
print(f"cns: GCS 13 -> {cns_score}")
print(f"renal: creatinine 2.4, urine 400 -> {renal_score}")
print(f"SOFA total: {total}")

try:
    renal(212.2, 400)
except ValueError as exc:
    print(f"unit check: creatinine 212.2 rejected ({exc})")
```

Expected output:

```text
respiratory: P/F 156.0 -> 3
coagulation: 88 -> 2
liver: 1.6 mg/dL -> 1
cardiovascular: MAP 58, NE 0.08 -> 3
cns: GCS 13 -> 1
renal: creatinine 2.4, urine 400 -> 3
SOFA total: 13
unit check: creatinine 212.2 rejected (creatinine 212.2 mg/dL above range; suspect umol/L)
```

### 7.2 KDIGO staging, baseline back-calculation, and norepinephrine equivalents

```python
"""KDIGO 2012 staging and norepinephrine equivalents. Standard library only."""

def kdigo_ratio(baseline, current):
    r = current / baseline
    return (3 if r >= 3.0 or current >= 4.0 else 2 if r >= 2.0
            else 1 if r >= 1.5 else 0), r

def kdigo_delta(baseline, current, hours):
    return 1 if current - baseline >= 0.3 and hours <= 48 else 0

def kdigo_urine(ml, kg, hours):
    r = ml / kg / hours
    return (3 if r < 0.3 and hours >= 24 else 2 if r < 0.5 and hours >= 12
            else 1 if r < 0.5 and hours >= 6 else 0), r

def mdrd_baseline(age, female, egfr=75.0):
    f = 175.0 * age ** -0.203 * (0.742 if female else 1.0)
    return (egfr / f) ** (-1.0 / 1.154)

def nee(ne, epi=0.0, phe=0.0, dopa=0.0, vaso=0.0, phe_factor=0.1):
    return ne + epi + phe_factor * phe + dopa / 100 + 2.5 * vaso

ratio_stage, ratio = kdigo_ratio(0.9, 2.4)
delta_stage = kdigo_delta(1.1, 2.4, 15.2)
urine_stage, urine_rate = kdigo_urine(400, 70, 24)

print(f"creatinine ratio 2.4/0.9 = {ratio:.2f} -> stage {ratio_stage}")
print(f"48h delta 2.4-1.1 = 1.3 mg/dL in 15.2 h -> stage {delta_stage}")
print(f"urine 400 mL/70 kg/24 h = {urine_rate:.3f} mL/kg/h -> stage {urine_stage}")
print(f"KDIGO stage (worst criterion): {max(ratio_stage, delta_stage, urine_stage)}")

baseline = mdrd_baseline(71, True)
back_stage, back_ratio = kdigo_ratio(baseline, 2.4)
print(f"back-calculated baseline: {baseline:.3f} mg/dL -> ratio {back_ratio:.2f} -> stage {back_stage}")
print(f"NEE (Goradia 2021): {nee(0.08, vaso=0.03):.3f} ug/kg/min")
print(f"NEE (Kotani 2023): {nee(0.08, vaso=0.03, phe_factor=0.06):.3f} ug/kg/min")
g, k = nee(0.0, phe=1.0), nee(0.0, phe=1.0, phe_factor=0.06)
print(f"phenylephrine 1.0: Goradia {g:.3f}, Kotani {k:.3f}, difference {100 * (g - k) / g:.1f}%")
```

Expected output:

```text
creatinine ratio 2.4/0.9 = 2.67 -> stage 2
48h delta 2.4-1.1 = 1.3 mg/dL in 15.2 h -> stage 1
urine 400 mL/70 kg/24 h = 0.238 mL/kg/h -> stage 3
KDIGO stage (worst criterion): 3
back-calculated baseline: 0.760 mg/dL -> ratio 3.16 -> stage 3
NEE (Goradia 2021): 0.155 ug/kg/min
NEE (Kotani 2023): 0.155 ug/kg/min
phenylephrine 1.0: Goradia 0.100, Kotani 0.060, difference 40.0%
```

### 7.3 What the code does not do

The SOFA function does not select the window, validate the source timestamp, or handle missing components. It does
not implement the dopamine, dobutamine, or epinephrine branches of the cardiovascular component, and it assumes a
single norepinephrine rate. The renal function treats urine output as a 24-hour total, which is the SOFA convention
but is not the same as the KDIGO urine criterion.

The KDIGO function applies the ratio criterion to any baseline, including the eight-month value, and does not enforce
the seven-day presumption in the guideline. It does not stage patients on renal replacement therapy, for whom stage 3
is defined by the therapy rather than by a number. The back-calculation assumes an eGFR of 75 and will overstage a
patient with chronic kidney disease.

The NEE functions implement two published formulas for the agents they name. They do not cover metaraminol,
angiotensin II, or terlipressin, and they do not capture inotropes. A production implementation must name its formula
version in the output, not only in a comment.

## 8. Data, standards, and regulatory context

Critical care is governed by a mixture of professional consensus, quality measurement, and device regulation. The
distinction matters because a team often treats a consensus definition as if it were a legal requirement, or a
quality measure as if it were a clinical standard.

### 8.1 Sepsis definitions and scores

Sepsis-3 (Singer et al., Journal of the American Medical Association, 2016;315:801–810) defines sepsis as suspected
infection plus an acute increase in SOFA of at least 2 points, and septic shock as vasopressor requirement for a mean
arterial pressure of at least 65 mmHg plus lactate above 2 mmol/L after adequate fluids. The SOFA score itself derives
from the 1996 description by Vincent and colleagues in Intensive Care Medicine. qSOFA (respiratory rate at least 22,
altered mentation, systolic pressure at most 100) was proposed as a bedside prompt for patients outside the ICU. It
was not proposed as a diagnostic criterion, and the Surviving Sepsis Campaign 2021 guideline suggests against using
qSOFA alone as the single screening tool, compared with SIRS, NEWS, or MEWS.

What these constrain: the meaning of the word sepsis in a publication, a label, or a regulatory claim. What they do
not constrain: the implementation window, the baseline assumption, or the cohort eligibility. Those remain study
decisions that must be documented.

### 8.2 Severity scores

APACHE IV (Zimmerman et al., Critical Care Medicine, 2006) and SAPS II (Le Gall et al., JAMA, 1993) are the standard
severity-adjustment instruments, with SAPS 3 (Moreno et al., Intensive Care Medicine, 2005) as a later alternative.
Each is calibrated to a particular population and era. A score developed on one case mix does not transport without
recalibration, and the version must be named because the coefficients differ.

### 8.3 Acute kidney injury

The KDIGO 2012 Clinical Practice Guideline for Acute Kidney Injury remains the published standard. Its definition and
staging are reproduced in Section 5, Step 7. A revision, the KDIGO 2026 Clinical Practice Guideline for Acute Kidney
Injury and Acute Kidney Disease, was made available for public review and was being prepared for publication at the
access date; no revised thresholds were final in the sources checked. A team that cites "KDIGO" without a year is
citing a moving target.

### 8.4 Treatment and quality standards

The Surviving Sepsis Campaign 2021 guideline (Evans et al., Critical Care Medicine, 2021;49:e1063–e1143) makes a
strong recommendation for an initial mean arterial pressure target of 65 mmHg over higher targets, a strong
recommendation for norepinephrine as the first-line vasopressor, a weak recommendation to add vasopressin rather than
escalate norepinephrine when the pressure remains inadequate, and a weak recommendation for intravenous hydrocortisone
at 200 mg per day when norepinephrine or epinephrine is at least 0.25 micrograms per kg per minute for at least four
hours. The guideline does not contain a 0.05 microgram per kg per minute threshold; that figure appears in local
protocols, not in the 2021 text.

The ARDS Berlin definition (ARDS Definition Task Force, JAMA, 2012;307:2526–2533) requires onset within one week,
bilateral opacities not fully explained by effusion, collapse, or nodule, and a PaO2/FiO2 at most 300 with a positive
end-expiratory pressure of at least 5 cm H2O, graded mild, moderate, and severe.

CMS SEP-1 is a quality measure in the Hospital Inpatient Quality Reporting Program. Its 3-hour elements include an
initial lactate, blood cultures before antibiotics, broad-spectrum antibiotics, and 30 mL/kg of crystalloid for
hypotension or a lactate of at least 4 mmol/L. Its 6-hour elements include a repeat lactate when the initial value
exceeds 2 mmol/L, vasopressors for persistent hypotension after fluids, and a repeat volume status assessment. The
measure specification in force in 2024 was version 5.16. SEP-1 is binding for reporting hospitals; it is not a clinical
practice guideline, and compliance is not a patient outcome.

### 8.5 Data standards

HL7 Fast Healthcare Interoperability Resources (FHIR) Observation and Encounter resources carry the values; Logical
Observation Identifiers Names and Codes (LOINC) carry the analyte identity; Systematized Nomenclature of Medicine
Clinical Terms (SNOMED CT) carry clinical findings; International Classification of Diseases, Tenth Revision, Clinical
Modification (ICD-10-CM) and Procedure Coding System (ICD-10-PCS) carry diagnoses and procedures for claims; Clinical
Data Interchange Standards Consortium (CDISC) SDTM and ADaM carry trial data. None of these encodes the SOFA window,
the sepsis suspicion rule, or the vasopressor normalisation. The standard makes the data exchangeable, not correct.

### 8.6 Binding requirement versus convention

| Instrument | Status | What it binds | What it does not |
|---|---|---|---|
| Sepsis-3 | Professional consensus | The meaning of sepsis in a claim or paper | The implementation window or cohort |
| SOFA | Clinical convention | The component thresholds if cited | The baseline or the data source |
| qSOFA | Screening prompt | Nothing on its own | Diagnosis, exclusion, or disposition |
| KDIGO 2012 | Clinical guideline | Practice if adopted by protocol or payer | Research cohort definitions |
| Surviving Sepsis 2021 | Clinical guideline | Practice through protocol and audit | Reimbursement or device clearance |
| CMS SEP-1 | Quality measure | Public reporting for participating hospitals | Individual clinical decisions |
| Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule | Federal regulation | Use and disclosure of protected health information | Clinical definition or model validity |
| Common Rule, 45 CFR 46 | Federal regulation | Federally funded human-subjects research | Commercial quality improvement |
| Food and Drug Administration (FDA) device pathway | Federal regulation | Marketing of a device or software function | Internal quality work |
| FHIR, LOINC, SNOMED CT | Interoperability convention | Exchange only when contract or rule requires | Semantics of a derived score |

> **Note:** Regulatory status changes. The FDA clinical decision support guidance was finalized in September 2022,
> and the boundary between device software and decision support depends on the intended user and the transparency of
> the basis. Confirm the current position with counsel before a marketing claim.

## 9. Industry practice and case evidence

The cases below are documented public events. Each separates what the source reports from what a data team may infer.

### 9.1 External validation of a widely deployed sepsis model

Wong and colleagues (JAMA Internal Medicine, 2021) evaluated the Epic Sepsis Model at Michigan Medicine across 38,455
hospitalizations with sepsis in 2,552 (7 percent). The model achieved an area under the receiver operating
characteristic curve (AUROC) of 0.63 (95 percent confidence interval 0.62–0.64), against 0.76–0.83 claimed by the
vendor. An alert threshold of 6 or more fired in 18 percent of hospitalizations, and the model missed 1,709 of 2,552
sepsis cases (67 percent). Sensitivity was 33 percent and positive predictive value 12 percent at that threshold.

Documented fact: a widely deployed proprietary model performed substantially worse on external validation than its
vendor reported, with poor calibration and high alert volume. Inference: the result is site-specific and does not
prove the model fails everywhere, but it does establish that widespread use is not evidence of validity. The
transferable lesson is procedural: demand an external validation with site-specific calibration before adoption.

### 9.2 A prospective early-warning system with workflow integration

Adams and colleagues (Nature Medicine, 2022) reported a prospective, multi-site evaluation of the Targeted Real-time
Early Warning System (TREWS) for sepsis, which was designed to reduce alert burden by targeting high-risk patients and
to integrate the alert into the clinical workflow. The evaluation reported an association between confirmed alerts and
improved outcomes, with the effect concentrated among patients whose alerts were confirmed and acted on.

Documented fact: the study was prospective and multi-site, and it measured outcomes rather than discrimination alone.
Inference: the result supports the claim that alert burden and workflow are part of the intervention, not separate
from it. It does not establish transportability to a different health system, and a before-and-after design carries
residual confounding from concurrent practice change.

### 9.3 Reinforcement learning for treatment policy

Komorowski and colleagues (Nature Medicine, 2018) used MIMIC-III to learn a reinforcement-learning policy for fluid
and vasopressor dosing in sepsis, and reported that the learned policy was associated with lower mortality in
off-policy evaluation. Gottesman and colleagues (Nature Medicine, 2019) documented the fragility of such evaluations:
off-policy estimates depend on untestable assumptions about the behaviour policy and on the overlap between the
observed and recommended actions.

Documented fact: the learned policy was evaluated offline, not in a randomised trial. Inference: the work is a
hypothesis generator about treatment strategies, not a causal estimate. A data team that presents an off-policy value
as an expected clinical effect has substituted an assumption for a result.

### 9.4 A benchmark that defined its own label

The PhysioNet Computing in Cardiology Challenge 2019 asked teams to predict sepsis from ICU time series across three
hospitals, with the label, the prediction window, and the utility function defined by the organisers. The benchmark
made results comparable across teams and surfaced the importance of early prediction and alert utility.

Documented fact: the challenge supplied a common label and a utility metric. Inference: the winning models did not
necessarily transport to other hospitals, because the label and the cohort were part of the task definition. A
benchmark score is a statement about the benchmark.

### 9.5 A quality measure that changed documentation

CMS SEP-1 ties lactate measurement, blood cultures, antibiotics, and fluid resuscitation to public reporting. The
evidence review underlying the measure found low-level evidence for some elements, including serial lactate and the
30 mL/kg fluid volume, and the measure requires extensive documentation.

Documented fact: SEP-1 is a reported quality measure with specified time windows and documentation requirements.
Inference: measurement changes behaviour, and some of the observed improvement may be documentation rather than
physiology. SEP-1 compliance should not be used as a surrogate for model quality or for patient benefit.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Onset anchored at ICU admission | Median time to antibiotic is implausibly short | Suspicion pair ignored | Early warning evaluates the wrong moment | Store the pair and the rule |
| Whole-stay SOFA | Score distribution is shifted high | Worst value taken over the stay | Label and features share a source; discrimination inflated | Enforce the window in code |
| Baseline SOFA assumed zero | Sepsis incidence is high in chronically ill patients | Chronic dysfunction counted as acute | Over-diagnosis and severity misclassification | Record baseline and source |
| Creatinine unit mismatch | Creatinine values cluster near 200 | Micromol/L read as mg/dL | Renal SOFA and AKI stage wrong | Range check and unit field |
| Vasopressor formula unstated | NEE differs between two reports of one cohort | Different conversion factors | Dose-response and severity comparisons invalid | Name the formula version |
| Fluid balance without a denominator | Balance reported only in litres | No weight or window | Fluid overload cannot be classified | Store weight, unit, and window |
| Tidal volume per actual weight | Tidal volumes cluster at 500 mL | Predicted body weight not computed | Injurious ventilation looks protective | Compute predicted body weight |
| Discharge treated as censoring | Discharge curve rises fastest in the sickest | Competing event ignored | Unit appears efficient because patients died | Use cumulative incidence |
| Height in inches | Predicted body weight is very low | Unit not recorded | Underventilation and wrong tidal volume | Unit field and range check |
| Informative missingness | Lactate is present mainly in sick patients | Measurement depends on state | Model learns the ordering decision | Model the measurement process |
| Alarm threshold chosen for AUROC | Alerts per patient-day is very high | Threshold not tied to capacity | Alert fatigue and missed true events | Report burden at the operating point |
| Treatment paradox | Vasopressors look harmful | Sicker patients receive more treatment | Wrong causal direction | Time-varying confounding methods |
| Duplicate ICU episodes | Cohort count exceeds unit census | Transfers create new rows | Double counting and non-independent events | Deduplicate on encounter and time |

Two red flags deserve emphasis because they are cheap to check. The first is a sepsis prevalence that is far above
the published range for the case mix; it usually means the suspicion rule is too permissive, often because
surveillance cultures are being paired with any antibiotic. The second is a SOFA component that never varies, such as
a renal score of zero for every patient, which usually means the component is not being populated at all.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The culture and antibiotic timestamps are accurate | Time zero depends on both | Charting delay, ward and ICU systems | Compare order and administration times |
| The 72-hour and 24-hour pairing windows match the intended definition | They define the episode | Local implementations differ | Recompute under alternative windows |
| SOFA components are measured within the window | The score is window-bound | Sparse sampling, late labs | Count components by time from t0 |
| Baseline organ dysfunction is known or zero | Delta SOFA defines sepsis | Chronic disease, missing history | Chart review of a sample |
| The weight used for dosing is in kilograms | Dose and urine rate are weight-normalised | Pounds entered as kilograms | Range check by age and setting |
| The NEE formula is fixed across the cohort | Dose comparison requires one scale | Multi-site formula differences | Recompute under both formulas |
| Urine output is complete | KDIGO urine staging depends on it | Missing hours, catheter changes | Compare charted and derived totals |
| Discharge and death are correctly distinguished | Competing risks require it | Transfer without outcome | Link to vital records |
| Missing labs are missing at random | Complete-case analysis requires it | Measurement driven by severity | Model the measurement process |
| The cohort is representative of the deployment site | Transportability requires it | Single-centre academic data | External validation with recalibration |

The largest limit is not in the table. Critical-care data are generated by monitoring and treatment, not by
measurement. A patient without an arterial line has fewer blood gases; a patient without a urinary catheter has no
hourly urine output; a patient who is not escalated has no deterioration event. Each absence is a function of the care
pathway, and the care pathway is a function of access and of clinician judgement. A model trained on monitored
patients therefore encodes monitoring as if it were physiology.

A second limit is that the definitions are moving. Sepsis-3 replaced an earlier consensus, KDIGO is under revision,
and the Surviving Sepsis Campaign updates on a multi-year cycle. A cohort that spans a definition change contains two
different diseases under one label. The honest approach is to report the version per case and to restrict or map,
not to pool silently.

## 12. Exercises

### Exercise 1 — Derive a suspicion time (recall and application, 25 minutes)

**Deliverable.** A table of five culture and antibiotic timestamp pairs with the classification, the time zero, and
the rule applied.

**Acceptance criteria.** Both pairing directions are handled; a pair outside the window is rejected with the reason;
a culture with no antibiotic is classified correctly; the earliest qualifying pair is selected when several exist.

### Exercise 2 — Compute a SOFA score (application, 30 minutes)

**Deliverable.** A six-row component table with the source value, the score, and the window for a supplied case.

**Acceptance criteria.** Every component names its threshold; the renal component takes the worse of creatinine and
urine output; a missing component is not scored as zero; the total is the sum of the components.

### Exercise 3 — Stage AKI under two baselines (application, 30 minutes)

**Deliverable.** A short calculation of KDIGO stage under a measured baseline and a back-calculated baseline, with the
stage shift stated.

**Acceptance criteria.** The 48-hour delta and the 7-day ratio criteria are distinguished; the urine criterion is
computed per kilogram per hour; the worst criterion determines the stage; the assumption behind the back-calculation
is named.

### Exercise 4 — Normalise a vasopressor regimen (application, 25 minutes)

**Deliverable.** A NEE calculation for a three-agent regimen under two published formulas, with the difference stated.

**Acceptance criteria.** Units are stated for every agent; the weight used is recorded; the formula version is named;
the percentage difference between formulas is computed.

### Exercise 5 — Derive ventilation variables (application, 30 minutes)

**Deliverable.** Predicted body weight, tidal volume per kilogram, and driving pressure for a supplied case, with the
formulas shown.

**Acceptance criteria.** Predicted body weight uses height and sex; the tidal volume target at 6 mL/kg is stated; the
driving pressure uses plateau pressure minus positive end-expiratory pressure; a unit error is identified if present.

### Exercise 6 — Model discharge as a competing risk (critique, 40 minutes)

**Deliverable.** A one-page comparison of a Kaplan-Meier discharge estimate and a cumulative incidence estimate on a
supplied dataset, with the direction and size of the difference.

**Acceptance criteria.** The estimand is stated for each method; the competing event is reported; the code does not
use one minus Kaplan-Meier for the cumulative incidence; the interpretation names who is misled.

### Exercise 7 — Audit unit conversions (critique, 35 minutes)

**Deliverable.** An audit of a supplied extract listing every analyte, its expected unit, its observed unit, and the
converted value, with a list of failures.

**Acceptance criteria.** Creatinine, lactate, bilirubin, weight, and vasopressor rate are covered; at least one
deliberate error is detected; the detection uses a range check rather than a visual inspection; the corrected value
is stored with the original.

### Exercise 8 — Design a deployment evaluation (design, 50 minutes)

**Deliverable.** A two-page evaluation plan for a sepsis alert, covering discrimination, calibration, alert burden,
workflow effect, and subgroup performance.

**Acceptance criteria.** The plan names the time zero and the prediction time; it reports alerts per patient-day and
the number needed to alert; it includes a site-specific calibration check; it states what the evaluation does not
establish.

## 13. Self-check answers

**1. Suspicion time.** A culture at 18:05 followed by an antibiotic at 19:10 is a valid culture-first pair, because
65 minutes is within 72 hours, and time zero is 18:05. An antibiotic at 18:05 followed by a culture at 18:50 is a
valid antibiotic-first pair, because 45 minutes is within 24 hours, and time zero is 18:05. A culture at 18:05 with an
antibiotic at 21:00 two days later is 51 hours, still within 72 hours; a gap of 80 hours is not.

**2. SOFA total.** Respiration 78/0.50 = 156, which is below 200 with support, so 3. Platelets 88 is below 100, so 2.
Bilirubin 1.6 is in 1.2–1.9, so 1. Norepinephrine 0.08 is at most 0.1, so 3. GCS 13 is in 13–14, so 1. Creatinine 2.4
scores 2 and urine output 400 mL/day scores 3, so the renal component is 3. Total 3 + 2 + 1 + 3 + 1 + 3 = 13.

**3. AKI stage.** With a baseline of 0.9 mg/dL, 2.4/0.9 = 2.67, which is stage 2 by ratio. The rise from 1.1 to 2.4 is
1.3 mg/dL within 48 hours, which meets the absolute criterion. Urine output is 400/70/24 = 0.238 mL/kg/h for 24 hours,
which is stage 3. The worst criterion gives stage 3. A back-calculated baseline of 0.76 gives 3.16 times baseline,
which is stage 3 by creatinine alone.

**4. NEE.** Norepinephrine 0.08 plus 2.5 times vasopressin 0.03 equals 0.08 + 0.075 = 0.155 micrograms per kg per
minute. Phenylephrine 1.0 alone is 1.0/10 = 0.10 under Goradia and 1.0 × 0.06 = 0.06 under Kotani. The difference is
(0.10 − 0.06)/0.10 = 40 percent.

**5. Ventilation.** Predicted body weight for a woman of 165 cm is 45.5 + 0.91 × 12.6 = 56.97 kg. A tidal volume of
500 mL is 8.8 mL/kg, above the 6 mL/kg target of 342 mL. Driving pressure is 26 − 8 = 18 cm H2O, above the 15 cm H2O
threshold.

**6. Competing risks.** A Kaplan-Meier estimate treats death as censoring, so it estimates discharge in a world where
no one dies. Because the patients who die are the ones least likely to be discharged, the Kaplan-Meier curve
overestimates discharge. A cumulative incidence function estimates the probability of discharge in the presence of
death and is the correct estimand for a flow question.

**7. Unit error.** A creatinine of 212.2 is 2.4 mg/dL multiplied by 88.4, so it is micromoles per litre. Read as
milligrams per decilitre, it exceeds 5.0 and scores 4 on the renal component instead of 2, adding 2 points to SOFA and
producing a value no laboratory would report. A range check at 20 mg/dL catches it.

**8. Deployment evaluation.** A strong plan separates discrimination (AUROC, AUPRC), calibration (slope, intercept,
Brier score), burden (alerts per patient-day, number needed to alert), workflow (time to action, override rate), and
equity (subgroup sensitivity and alert rates). A weak plan reports AUROC only and calls it validation.

**Rubric for Exercises 6 and 8.** A strong answer names the estimand, shows the arithmetic or the design, states the
counterfactual, and says what the result does not establish. A weak answer reports a number without its denominator or
asserts that a model is validated because it was deployed.

## 14. Interview questions

**1. What is time zero for sepsis, and why does it matter?**
Strong answer: the earlier of a qualifying culture and antibiotic pair, with the 72-hour and 24-hour asymmetry, because
the SOFA window and the early-warning window are both anchored there. Weak answer: "when the patient became septic",
which is not computable.

**2. A vendor reports AUROC 0.85 for sepsis. What do you ask?**
Strong answer: the label definition, the time zero, the prediction window, the prevalence, the calibration, and the
alert burden at the operating threshold. Weak answer: "what is the sensitivity", which addresses one number and not
the cohort.

**3. How do you compute SOFA, and what can go wrong?**
Strong answer: six components, worst value in a stated window, renal takes the worse of creatinine and urine, missing
is not zero, baseline matters. Weak answer: "add up the six organs", without the window or the baseline.

**4. Why normalise vasopressors?**
Strong answer: different agents have different potency and units, so a raw dose comparison compares notation; the
formula version must be named because published factors differ. Weak answer: "to make them comparable", without a
formula.

**5. How would you detect a unit conversion error in a critical-care extract?**
Strong answer: range checks per analyte, a unit field carried with every value, and a comparison of distributions
across sending laboratories. Weak answer: "check the mean", which will not reveal a bimodal unit mix.

**6. What is a competing risk in ICU data?**
Strong answer: discharge and death preclude each other, so a naive Kaplan-Meier overestimates discharge; use cumulative
incidence for flow questions and cause-specific hazards for aetiological questions. Weak answer: "patients who leave",
without the estimand.

**7. Why is creatinine a poor sole marker of AKI?**
Strong answer: it lags injury, depends on muscle mass and dilution, and requires a baseline that is often missing.
Weak answer: "it is not sensitive", without the mechanism.

**8. How do you measure alert burden?**
Strong answer: alerts per patient-day, number needed to alert, override rate, and time to action, all at the deployed
threshold. Weak answer: "count the alerts", without a denominator.

**9. What did the Epic Sepsis Model external validation show?**
Strong answer: AUROC 0.63 against a vendor claim of 0.76–0.83, 18 percent of hospitalizations alerted, 67 percent of
cases missed, poor calibration. Weak answer: "it did not work", without the numbers or the site specificity.

**10. Can reinforcement learning identify the best treatment policy?**
Strong answer: only under strong assumptions; off-policy evaluation is fragile, and the output is a hypothesis for a
trial, not a causal effect. Weak answer: "yes, if the reward is right", which ignores overlap and confounding.

**11. Which patients might a deterioration model systematically miss?**
Strong answer: those with less monitoring, atypical presentation, or language barriers, with a plan to report subgroup
alert rates and delays. Weak answer: "no one, the model is validated".

## 15. Key terms

- **Sepsis-3** — 2016 consensus definition: suspected infection plus a SOFA increase of at least 2.
- **Suspicion time** — earlier of a qualifying culture and antibiotic pair.
- **SOFA** — six-organ dysfunction score, 0–24.
- **Delta SOFA** — current score minus baseline score.
- **qSOFA** — bedside screening prompt, not a diagnosis.
- **Septic shock** — vasopressor requirement for MAP at least 65 mmHg plus lactate above 2 mmol/L.
- **APACHE IV** — severity score for ICU mortality.
- **SAPS II** — simplified severity score, version II.
- **KDIGO** — kidney guideline body; 2012 AKI definition and staging.
- **Baseline creatinine** — reference value before the acute episode.
- **Urine output criterion** — below 0.5 mL/kg/h for 6 hours or more.
- **Norepinephrine equivalent** — composite vasopressor dose normalised to norepinephrine.
- **Vasopressin** — non-catecholamine vasopressor, usually 0.03 units per minute.
- **Fluid balance** — cumulative intake minus output.
- **Fluid overload** — balance as a percentage of body weight.
- **Predicted body weight** — height- and sex-derived weight for ventilation.
- **Driving pressure** — plateau pressure minus PEEP.
- **PEEP** — positive end-expiratory pressure.
- **PaO2/FiO2** — oxygenation ratio.
- **ARDS** — acute respiratory distress syndrome, Berlin definition.
- **Competing risk** — an event that precludes the event of interest.
- **Cumulative incidence** — probability of an event in the presence of competing risks.
- **Immortal time** — person-time before an exposure could occur.
- **CMS SEP-1** — sepsis quality measure for reported hospitals.

## 16. Sources

- Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA.
  2016;315(8):801–810. https://doi.org/10.1001/jama.2016.0287 (accessed 26 October 2026).
- Evans L, et al. Surviving Sepsis Campaign: international guidelines for management of sepsis and septic shock 2021.
  Crit Care Med. 2021;49(11):e1063–e1143. https://doi.org/10.1097/CCM.0000000000005337 (accessed 26 October 2026).
- KDIGO. Clinical practice guideline for acute kidney injury. Kidney Int Suppl. 2012;2:1–138.
  https://kdigo.org/guidelines/acute-kidney-injury/ (accessed 26 October 2026).
- KDIGO. Scope of work: KDIGO 2026 clinical practice guideline for acute kidney injury and acute kidney disease.
  https://kdigo.org/wp-content/uploads/2023/10/KDIGO-AKI-Guideline_Scope-of-Work_25Oct2023_Final.pdf
  (accessed 26 October 2026).
- ARDS Definition Task Force. Acute respiratory distress syndrome: the Berlin definition. JAMA.
  2012;307(23):2526–2533. https://doi.org/10.1001/jama.2012.5669 (accessed 26 October 2026).
- Wong A, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized
  patients. JAMA Intern Med. 2021;181(8):1065–1070. https://doi.org/10.1001/jamainternmed.2021.2626
  (accessed 26 October 2026).
- Adams R, et al. Prospective, multi-site study of patient outcomes after implementation of the TREWS machine
  learning-based early warning system for sepsis. Nat Med. 2022. https://doi.org/10.1038/s41591-022-01894-0
  (accessed 26 October 2026).
- Komorowski M, et al. The artificial intelligence clinician learns optimal treatment strategies for sepsis in
  intensive care. Nat Med. 2018. https://doi.org/10.1038/s41591-018-0213-5 (accessed 26 October 2026).
- Gottesman O, et al. Guidelines for reinforcement learning in healthcare. Nat Med. 2019.
  https://doi.org/10.1038/s41591-018-0310-5 (accessed 26 October 2026).
- Goradia S, et al. Vasopressor dose equivalence: a scoping review and suggested formula. J Crit Care. 2021.
  https://doi.org/10.1016/j.jcrc.2020.11.002 (accessed 26 October 2026).
- Kotani Y, et al. An updated norepinephrine equivalent score in intensive care as a marker of shock severity.
  Crit Care. 2023;27:29. https://doi.org/10.1186/s13054-023-04322-y (accessed 26 October 2026).
- PhysioNet. MIMIC-IV. https://physionet.org/content/mimiciv/ (accessed 26 October 2026).
- PhysioNet. Early prediction of sepsis from clinical data, Challenge 2019.
  https://physionet.org/content/challenge-2019/ (accessed 26 October 2026).
- Centers for Medicare and Medicaid Services. SEP-1 measure specifications.
  https://www.cms.gov/medicare/quality/measures (accessed 26 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_pathology.md`](03_pathology.md)
- Next topic: [`05_pediatrics-and-obstetrics.md`](05_pediatrics-and-obstetrics.md)
- Cross-specialty comparison: [`07_cross-specialty-comparison.md`](07_cross-specialty-comparison.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Labs and datasets: [`../../labs/`](../../labs/)
