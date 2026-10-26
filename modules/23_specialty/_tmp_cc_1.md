# Critical Care

> Module 23 — Specialty Electives
> Topic 04 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08; helpful: 06, 14
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
| 3 | Stage acute kidney injury (AKI) by KDIGO 2012 creatinine and urine criteria, including an unknown baseline | Exercise 3; self-check 3 |
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

A critical-care analysis is a chain of timestamped measurements, and every join in
the chain is a place where the time or the unit can be wrong.

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

That rule has three consequences that a data team must implement explicitly.

First, time zero can precede the ICU admission by many hours, and in some cohorts it
precedes the hospital admission. A cohort restricted to ICU stays will therefore
truncate the early part of the window for patients who deteriorated on the ward.

Second, the pairing rule creates a dependency between two tables that are usually
owned by different systems: microbiology and pharmacy. A missing administration
timestamp does not mean no antibiotic; it may mean the drug was given on the ward and
charted in a different system.

Third, the window over which organ dysfunction is assessed extends 24 hours after
time zero. That is correct for defining the label retrospectively and fatal for
prediction if the same window is used to build features. The label may look forward;
the features may not.

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

Critical-care data is not recorded; it is cascaded. Each layer loses information and
adds latency, and a model that ignores the cascade will treat a charted value as a
measurement.

| Layer | What happens | Latency | Typical defect |
|---|---|---|---|
| Device | Monitor, ventilator, infusion pump, laboratory analyser | Seconds | Calibration drift, artefact, wrong unit at source |
| Chart | Nurse validates or edits the value into the record | Minutes to hours | Editing, carry-forward, default values |
| Observation table | Value mapped to a concept identifier and unit | Minutes | Unit conversion, duplicate rows, null flavours |
| Hourly aggregate | Values summarised into bins | Up to an hour | Mean of a maximum, minimum of a mean |
| Model feature | Aggregate transformed for the model | Batch | Window leakage, imputation that encodes the future |

The clinical consequence is that two numbers that look identical are not. A systolic
pressure of 88 mmHg charted by a nurse after review and a systolic pressure of 88
mmHg read automatically from an arterial line are different measurements with
different error properties. A pipeline that pools them has created a variable that no
clinician uses.

### 3.3 Competing risks

In ICU data, death and discharge alive compete. A patient who dies cannot later be
discharged, and a patient discharged alive cannot die in the ICU. Standard
time-to-event analysis treats the competing event as censoring, which assumes that a
patient censored at discharge would have had the same subsequent hazard as a patient
still in the ICU. That assumption is false, and it is false in a direction that
matters: the patients discharged earliest are usually the least sick.

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
| CMS SEP-1 | Severe Sepsis and Septic Shock Early Management Bundle | Used as a surrogate for model quality | Quality measure |
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
