# Cardiology

> Module 23 — Specialty Electives
> Topic 02 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a diagnostic
> authority, and not a substitute for the current ACC/AHA/HRS, ESC, or institutional
> protocol. Every threshold below names its source and version. Where a guideline has
> been superseded, the superseded and current versions are both given, because the
> difference between them is the lesson.

## 1. Why this topic exists

Cardiology is the specialty where a measurement is cheapest to produce and hardest to
make comparable. A twelve-lead electrocardiogram (ECG) costs a few dollars, a
photoplethysmographic (PPG) pulse wave costs nothing once the watch is sold, and an
echocardiogram report contains a number, the left ventricular ejection fraction (LVEF),
that three sonographers can disagree about by ten points without anyone being wrong.
The signal is continuous, the labels are adjudicated events, and the action is a drug
that causes bleeding or a procedure that costs more than a year of ordinary care.

Three structural properties make cardiology data errors unusually durable.

**The test is ubiquitous, so there is no natural control group.** Almost everyone who
enters a hospital gets an ECG, and almost everyone with a smartwatch generates a pulse
waveform. When a measurement is universal, a broken pipeline does not announce itself by
producing no data. It produces a full table of plausible numbers, and the comparison that
would expose the defect was never run.

**Deployment prevalence is unknown and usually lower than validation prevalence.** A model
validated in a cardiology clinic sees disease in one patient in three; the same model
deployed to a consumer app sees disease in one patient in two hundred. Sensitivity and
specificity are roughly stable across that move; positive predictive value is not.

**The guideline that defines the action changes on a different schedule than the model.**
A model frozen in 2019 and deployed in 2026 is scored against a moved threshold. The
2023 ACC/AHA/ACCP/HRS atrial fibrillation (AF) guideline replaced score-based
anticoagulation decisions with annual-risk-based decisions; the 2024 ESC AF guideline
removed the sex category from the score entirely. Neither change touched the ECG. Both
changed which patients get a drug.

### 1.1 The ordinary path of one case

A 71-year-old man with hypertension and diabetes buys a smartwatch. The watch emits an
irregular-rhythm notification, the app records a tachogram, the tachogram is classified,
the message produces a clinic call, the clinic orders a seven-day patch, a cardiologist
reads the report, a score is computed, and a prescription is written or not written.
Eight systems hold a partial version of one decision.

Every step in that chain has a measurable failure rate, and only two are modelling steps.
A team that tunes the classifier and ignores the confirmation rate has optimised the
smallest term in the product.

### 1.2 The signature failures

- **Prevalence-blind positive predictive value (PPV).** The headline metric is computed at
  the validation prevalence and quoted at deployment. Section 6 shows a test with 95
  percent sensitivity and 95 percent specificity producing a PPV of 8.7 percent at 0.5
  percent prevalence and 50.0 percent at 5 percent.
- **Lead reversal.** Limb electrodes are exchanged, the ECG is internally consistent, and
  the morphology is wrong. Lead I inverts, aVR and aVL swap, and an algorithm that has
  learned "negative P in lead I means ectopic atrial rhythm" now calls a normal sinus
  rhythm abnormal. The recording passes every quality check that does not look at
  Einthoven's law.
- **Device and firmware drift.** An algorithm locked to one watch model is deployed on the
  next model with a different optical sensor, a different sampling rate, and a different
  rejection rule. The model has not changed and its inputs have.
- **Guideline epoch mixing.** A registry that pools 2019 and 2026 patients reports a mean
  CHA2DS2-VASc that no single guideline would have produced, and a treatment rate that
  reflects two different thresholds.
- **Adjudication disagreement.** Major adverse cardiac events (MACE) are a committee
  judgment. Two committees reading the same charts disagree on 5 to 15 percent of events,
  so a model evaluated against committee A is not the model evaluated against committee B.
- **Verification bias.** Only patients who return the confirmatory patch are verified. The
  rest are assumed negative, which converts a participation behaviour into a test result.

### 1.3 Who is blocked when the measurement is wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | An anticoagulant that was withheld, or a bleed that was invited | Cannot see the score, the threshold, or the guideline version |
| Primary care clinician | Confidence in a notification that arrives without a confirmation pathway | Does not control the device or the algorithm |
| Cardiologist | Time reconciling a watch alert with an uninterpretable single-lead tracing | No access to the raw signal or the firmware version |
| Device maker | A clearance that does not cover the population now using the product | Intended use is fixed at submission; users are not |
| Health system | Nursing capacity consumed by alerts that convert at 10 percent | Alert volume is set by the vendor's threshold, not the clinic's capacity |
| Payer | Remote monitoring spend with no measured stroke reduction | Sees claims, not confirmation rates or adjudication rules |
| Data scientist | A model that validates at 0.91 area under the curve and fails at the first site | Inherited prevalence and label definitions were never documented |

### 1.4 What this topic covers and what it does not

This topic covers the measurement layer: how a cardiac signal becomes an interval, how an
interval becomes a label, how a label becomes an adjudicated endpoint, how an endpoint
becomes a guideline-driven action, and how each translation can fail. It does not teach
electrophysiology, pharmacology, or trial design. It assumes you have met sensitivity,
specificity, and the Kaplan-Meier estimator elsewhere and will not re-derive them.

The emphasis is deliberate. Most cardiology machine-learning curricula spend their time on
architecture and almost none on the fact that the same patient can be "AF" in one table,
"irregular rhythm" in a second, and "no AF" in a third because the three tables used three
episode definitions.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Compute CHA2DS2-VASc and CHA2DS2-VA and state which guideline version and threshold each belongs to | Exercise 1; self-check 1 |
| 2 | Convert a sensitivity and specificity pair into PPV and negative predictive value (NPV) at four deployment prevalences, showing the arithmetic | Exercise 2; self-check 2 |
| 3 | Name the three clocks in a cardiology cohort and anchor an analysis to exactly one | Exercise 3; interview question 3 |
| 4 | Diagnose limb-lead reversal and signal-quality failure from waveform features rather than from metadata | Exercise 4; self-check 4 |
| 5 | Explain how a device, firmware, or preprocessing change breaks a locked algorithm and specify the change-control evidence required | Exercise 5; interview question 6 |
| 6 | Write an outcome adjudication rule, measure inter-adjudicator agreement, and quantify the effect of disagreement on an effect estimate | Exercise 6; self-check 6 |
| 7 | Classify a case into heart failure stage A to D and LVEF category, including HFimpEF, and state the coding consequence | Exercise 7; interview question 8 |
| 8 | Place a cardiology product on the FDA device path or the clinical decision support (CDS) exemption and name the predicate or the criterion relied on | Exercise 8; interview question 10 |

Outcomes 1 and 2 are the quantitative core. They are separated because the score is a
lookup and the prevalence arithmetic is a derivation; a learner who can produce a score
but cannot state the prevalence at which it was validated has learned a table, not a
measurement system.

Outcomes 3 and 6 are about time. They are the two places where cardiology data most often
becomes unfalsifiable: an index date that varies by patient, and an event that is whatever
the committee said it was.

Outcomes 4 and 5 are about the instrument. Cardiology is the specialty where the sensor is
in the patient's pocket and the vendor ships firmware quarterly. A model is a claim about
a device as much as about a population.

Outcomes 7 and 8 connect the measurement layer to the two external systems that decide
whether the work can be used: the guideline that defines the label and the regulator that
defines the claim.

### 2.1 How the outcomes map to the module

Outcomes 1 to 3 are prerequisites for the cross-specialty comparison in topic 07, where
the same prevalence arithmetic reappears as a screening-programme question. Outcome 6 is
shared with oncology, where the equivalent object is the RECIST read, and with critical
care, where it is the sepsis label. Outcomes 7 and 8 connect to module 14 (device
regulation) and module 11 (cardiology foundations).

If you retain only one outcome, retain outcome 2. Every other error in this topic is
visible in a table of counts. A PPV computed at the wrong prevalence is invisible, because
it produces a number that looks like a performance metric and is actually a statement
about the population.

## 3. The core mental model

A cardiology analysis is a chain of conversions, and every conversion has a different
unit. The signal is millivolts at a sampling rate. The interval is milliseconds. The label
is a category assigned by a reader or a committee. The endpoint is a date. The action is a
prescription or a procedure. Confusing the units at any boundary is the defect.

```text
  [1] Signal ............ mV, 250-1000 Hz, 1-12 leads, device and firmware
        |                 failure: lead reversal, artifact, filter change
        v
  [2] Interval .......... RR (ms), PR, QRS duration, QT
        |                 failure: wrong fiducial point, missed beat
        v
  [3] Label ............. AF / not AF, rhythm class, LVEF value
        |                 failure: episode-duration threshold, reader variation
        v
  [4] Endpoint .......... MACE, HF hospitalization, stroke, death
        |                 failure: adjudication disagreement, coding drift
        v
  [5] Action ............ anticoagulate, image, implant, admit
        |                 failure: guideline epoch, threshold, contraindication
        v
  [6] Payment / claim ... code, coverage, device clearance
```

Read the diagram as a chain of unit conversions, not a pipeline of models. Step [3] is the
only step most teams model. Steps [1], [4], and [5] are where the errors that survive
review live, because they are made by people and machines outside the modelling code and
they are recorded in different systems with different identifiers.

### 3.1 The three clocks

Cardiology has three clocks running at once. They have different origins, different
resolutions, and different failure modes, and mixing them is the root of most temporal
leakage in the specialty.

| Clock | Origin | Resolution | Used for | Typical error |
|---|---|---|---|---|
| Cardiac clock | Each R peak | 1-4 ms per beat | RR intervals, AF detection, heart rate variability | Treating beat index as wall-clock time |
| Encounter clock | Test order, monitor start, admission, or notification | Minutes to days | Index date, follow-up windows, endpoint attribution | Using the report signature date as the event date |
| Guideline epoch | Guideline effective date, device firmware version, criteria version | Months to years | Threshold selection, score definition, label definition | Applying today's threshold to a 2019 cohort |

The cardiac clock is the one that makes cardiology different from every other specialty in
this module. A single ten-second ECG contains ten to twenty beats, and each beat is a
measurement whose timestamp is not a wall-clock timestamp. A model trained on beat-level
features can learn the acquisition order of the beats, which correlates with nothing
clinical and everything about the recording device.

The encounter clock is where cohort definitions are made. "Time zero" for an AF screening
study is the notification date. For a heart failure readmission study it is the discharge
date. For an ACS study it is symptom onset or first medical contact, and the two are not
interchangeable. A cohort that mixes them has no common origin and therefore no
interpretable survival curve.

The guideline epoch is the clock most teams forget to record. It is not a timestamp in the
data; it is a property of the rulebook applied at analysis time. Section 8 gives the dates.
The point here is structural: the epoch must be a column in the analytic table, not a
sentence in the methods.

### 3.2 The measurement stack

Cardiology measurements nest. A patient has encounters, an encounter has recordings, a
recording has leads, a lead has beats, and a beat has fiducial points. Each level has an
identifier and a date, and every join must respect both.

```text
  patient_id
     └── encounter_id ......... date, setting (clinic / inpatient / ambulatory)
            └── recording_id ... device, firmware, sampling rate, filter, duration
                   └── lead_id . I, II, III, aVR, aVL, aVF, V1-V6
                          └── beat_id ... R time, RR interval, morphology
```

The single most common structural defect in a cardiology warehouse is a recording table
keyed only on `patient_id`. A patient with forty ECGs over six years then has forty rows
that can be joined in any order, and the join that happens to run first becomes the
"baseline". Store the hierarchy, join on the recording identifier, and filter by date
explicitly.

### 3.3 Where the decision threshold comes from

A threshold is not a property of the model. It is a property of the model, the prevalence,
and the relative harm of the two errors. The relation is explicit:

```text
  treat if  p(AF) >  C_fp / (C_fp + C_fn)

  where C_fp = cost of a false positive (patch, visit, bleed risk, anxiety)
        C_fn = cost of a false negative (stroke, and its lifetime cost)
```

For a screening notification the ratio is not close. A false negative that produces a
stroke costs orders of magnitude more than a false positive that produces a patch and a
clinic visit, so the threshold should be low and the PPV should be expected to be poor.
That is a design choice, not a model defect, and it must be stated before the performance
table is read. A team that reports "PPV 12 percent" without stating the harm ratio has
reported a number without its interpretation.

The mirror-image error is a high-specificity model deployed in a high-prevalence clinic
because it performed well in a low-prevalence screening programme. Specificity controls
the false-positive count, and the false-positive count only matters relative to the true
positives, which is prevalence. Section 6 works the arithmetic.

### 3.4 What the model is allowed to see

```text
  admissible ....................... | decision | ............. inadmissible
  demographics, prior diagnoses,     |          |  the confirmatory patch result,
  baseline ECG and echo, current     |          |  the anticoagulation decision,
  medications, guideline-defined     |          |  later hospitalizations, the
  risk factors at the decision date  |          |  adjudicated endpoint
```

The wall is drawn at the moment the action is taken. In AF screening, the wall is the
notification. In heart failure, it is the discharge or the clinic visit. In ACS, it is
first medical contact. The common violations are the confirmatory test used as a feature,
the anticoagulation decision used as a predictor of stroke, and the diagnosis code
assigned after the workup used as a baseline comorbidity.

## 4. Definitions and vocabulary

Every term below is used later. The "common misuse" column is the part worth memorising,
because the misuse is what appears in real analytic tables.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Electrocardiogram (ECG) | Surface recording of cardiac electrical activity across leads | Called "EKG" in one table and "ECG" in another, breaking joins | mV over time; SCP-ECG or DICOM waveform |
| Photoplethysmography (PPG) | Optical measurement of pulsatile blood volume | Treated as equivalent to an ECG for rhythm classification | Arbitrary units; device-specific |
| Tachogram | Sequence of beat-to-beat intervals derived from a PPG or ECG | Treated as a diagnosis rather than a signal | ms or sample index |
| Irregular rhythm notification | Device message that the rhythm warrants confirmation | Reported as a diagnosis of AF | Device-specific event code |
| Atrial fibrillation (AF) | Supraventricular arrhythmia with absent P waves and irregularly irregular ventricular response | Applied to any irregular rhythm, including ectopy | ICD-10-CM I48.x; SNOMED CT |
| AF burden | Proportion of monitored time in AF | Reported without the monitoring duration | Percent of monitored time |
| Episode duration threshold | Minimum AF duration counted as a case | Assumed universal; devices differ (30 s, 5 min, 6 min, 24 h) | Seconds or minutes |
| Paroxysmal AF | AF that terminates spontaneously or with intervention within 7 days | Used interchangeably with "intermittent" | Clinical classification |
| Persistent AF | AF continuous beyond 7 days | Confused with permanent AF | Clinical classification |
| Verification bias | Reference standard applied only to a selected subset | Reported as if the whole cohort was verified | Proportion verified |
| Sensitivity | TP / (TP + FN) | Computed on verified patients only, without saying so | Proportion |
| Specificity | TN / (TN + FP) | Quoted without the threshold it was computed at | Proportion |
| PPV | TP / (TP + FP) | Quoted from a validation sample and applied at deployment prevalence | Proportion |
| NPV | TN / (TN + FN) | Assumed to be high because sensitivity is high | Proportion |
| Prevalence | Proportion with the condition in the population actually tested | Taken from the training set | Proportion |
| Likelihood ratio | Sensitivity / (1 - specificity) for a positive test | Replaced by "accuracy" in vendor material | Ratio |
| Lead reversal | Exchange of limb electrode positions | Detected only by human review, if at all | Lead identifiers |
| Einthoven's law | Lead II = Lead I + Lead III at every instant | Not tested in quality control | mV |
| Baseline wander | Low-frequency drift from respiration or movement | Removed by a filter that also removes ST change | Hz |
| Sampling rate | Samples per second | Downsampled without recording the anti-alias filter | Hz |
| QRS duration | Duration of ventricular depolarisation | Measured by different algorithms with different fiducial points | ms |
| RR interval | Time between successive R peaks | Converted to instantaneous heart rate and averaged, which is not the mean rate | ms |
| Heart rate variability (HRV) | Variation in RR intervals over a window | Computed on segments containing ectopy | ms, ms squared, or ratio |
| Left ventricular ejection fraction (LVEF) | Stroke volume divided by end-diastolic volume | Treated as precise to one percentage point | Percent |
| HFrEF | Heart failure with reduced ejection fraction, LVEF at or below 40 percent | Applied to any patient with a low EF and no symptoms | Percent; 2022 AHA/ACC/HFSA |
| HFmrEF | Heart failure with mildly reduced ejection fraction, LVEF 41 to 49 percent | Diagnosed on EF alone after the 2022 guideline | Percent; requires filling-pressure evidence |
| HFpEF | Heart failure with preserved ejection fraction, LVEF at or above 50 percent | Diagnosed on EF alone | Percent; requires filling-pressure evidence |
| HFimpEF | Heart failure with improved ejection fraction, prior LVEF at or below 40 percent and follow-up above 40 percent | Called "recovered EF" and therapy stopped | Percent; 2022 AHA/ACC/HFSA |
| Stage A to D | Heart failure stages: at risk, pre-HF, symptomatic, advanced | Used interchangeably with NYHA class | Ordinal A-D |
| NYHA class | Functional class I to IV for symptomatic HF | Recorded once and carried across years | Ordinal I-IV |
| NT-proBNP | N-terminal pro B-type natriuretic peptide | Compared across assays without units | pg/mL; assay-specific |
| CHA2DS2-VASc | Stroke risk score, 0 to 9 | Computed without recording the guideline version | Integer 0-9 |
| CHA2DS2-VA | 2024 ESC variant with the sex category removed, 0 to 8 | Reported as CHA2DS2-VASc | Integer 0-8 |
| HAS-BLED | Bleeding risk score, 0 to 9 | Used to withhold anticoagulation | Integer 0-9 |
| DOAC | Direct oral anticoagulant | Assumed interchangeable across agents and doses | ATC codes; drug-specific |
| MACE | Major adverse cardiac events, usually death, MI, stroke, revascularisation | Treated as one endpoint when components differ | Composite; define components |
| ACS | Acute coronary syndrome | Used as a synonym for myocardial infarction | ICD-10-CM I20-I25 |
| hs-cTn | High-sensitivity cardiac troponin | Compared across assays and sexes without cutoffs | ng/L; assay-specific 99th percentile |
| Adjudication | Committee review assigning an endpoint | Treated as ground truth | Committee charter, not a code |
| KCCQ | Kansas City Cardiomyopathy Questionnaire | Analysed as a continuous variable without its floor effects | 0-100 score |
| Implantable cardioverter-defibrillator (ICD) | Device that treats ventricular arrhythmia | Its interrogation data treated as a clean endpoint source | Manufacturer format |
| Cardiac resynchronisation therapy (CRT) | Pacing to resynchronise ventricular contraction | Pooled with ICD in device cohorts | Manufacturer format |
| Remote monitoring | Transmission of device data outside the clinic | Assumed to reduce events because it increases detection | Episode-based |
| Firmware | Embedded software controlling a device | Not recorded with the signal | Version string |
| Locked algorithm | Model whose inputs, preprocessing, and weights are frozen at clearance | Retrained in place without a new submission | Change-control record |
| 510(k) | Premarket notification showing substantial equivalence to a predicate | Treated as an efficacy finding | FDA pathway |
| De Novo | Classification request for a novel low-to-moderate risk device | Confused with 510(k) | FDA pathway |
| PMA | Premarket approval for high-risk devices | Applied to software that does not meet the bar | FDA pathway |
| Clinical decision support (CDS) | Software that meets the four statutory criteria for enforcement discretion | Claimed whenever a clinician is "in the loop" | 21st Century Cures Act |

### 4.1 Categories that look like labels but are not

Three distinctions cause more rework than any others.

First, **"AF detected" is not a diagnosis**. It is a statement about a monitoring period,
a device, and an episode-duration threshold. A 30-second episode on a seven-day patch and a
six-minute episode on a watch are different measurements of different things. A cohort
that pools them has pooled the devices.

Second, **LVEF is not a fixed attribute**. It changes with volume status, afterload,
revascularisation, and therapy, and it is measured with an operator-dependent technique
whose repeatability is roughly plus or minus five percentage points. The 2022 guideline
created HFimpEF precisely because a patient who moves from 35 to 45 percent is a different
clinical object from a patient who started at 45. Storing one LVEF per patient destroys
that distinction.

Third, **"heart failure" is not one cohort**. Stage A is a risk-factor population with no
disease. Stage D is a population under consideration for transplant or mechanical support.
The event rates differ by two orders of magnitude and the eligible interventions do not
overlap. A model trained on "HF" admissions has usually trained on stage C, and applying
it to a stage A screening population is a category error before it is a statistical one.

### 4.2 What counts as an event

| Endpoint | Event | Ascertainment source | Common defect |
|---|---|---|---|
| AF detection | First episode meeting the duration rule | Patch, device, or ECG | Duration rule not recorded |
| Stroke | Acute focal neurological deficit with imaging or clinical confirmation | Adjudication plus imaging | Included transient ischaemic attack |
| MACE | Death, MI, stroke, or revascularisation | Adjudication committee | Components weighted equally |
| HF hospitalisation | Admission with HF as primary reason | Claims plus chart | Observation stays excluded silently |
| Cardiovascular death | Death attributed to a cardiac cause | Adjudication | Attribution disagreement unmeasured |
| ICD shock | Appropriate or inappropriate device therapy | Device interrogation | Appropriate/inappropriate not separated |
| LVEF change | Absolute change at a stated interval | Repeat echo | Interval varies by patient |

> **Note:** "Last contact" is not a censoring date for a device endpoint. A patient whose
> device was last interrogated fourteen months ago may have had a shock in the gap. Censor
> at the last interrogation, and report the interrogation interval.

## 5. The mechanics, step by step

This section is the procedure for turning cardiology signals and records into a defensible
analysis. Each of the ten steps states its input, output, rationale, branch points, and edge
cases. A team that executes them in order, and documents the choices, can defend the
resulting cohort to a cardiologist, a regulator, and a payer.

### Step 1 — Fix the decision, the action, and the deployment prevalence

**Input.** A written question naming a decision, a decision-maker, an action, and the
population in which the action will be taken.

**Output.** A decision specification: the action, the threshold that triggers it, the harm
of each error, and a numeric deployment prevalence with its source.

**Rationale.** Deployment prevalence is not a property of the data you have but of the
population you serve, and it sets the PPV the buyer experiences. A model selected on the area
under the receiver operating characteristic curve (AUROC) alone has left its most
consequential characteristic unexamined.

**Procedure.**

1. Write the decision with an actor and a verb: "the programme will offer a confirmatory
   patch to adults who receive an irregular-rhythm notification."
2. Write the action that follows a confirmed positive. If no action follows, the model has
   no buyer.
3. Estimate deployment prevalence from the population that will actually be tested, and cite
   the source and date. A consumer wearable population and a cardiology clinic differ by a
   factor of ten or more.
4. State the harm ratio in the buyer's currency: clinic hours, dollars, bleeds, strokes.

**Branch point.** If the action is irreversible, the threshold must be high enough that PPV
is acceptable without confirmation. If the action is a cheap confirmatory test, the threshold
can be low and the programme should be measured on the yield of the cascade.

**Edge case.** A product cleared for one prevalence band and marketed into another. Record
the cleared population and the deployed population as separate fields.

### Step 2 — Build the measurement hierarchy

**Input.** Encounter, recording, lead, and beat tables, or their flat-file equivalent.

**Output.** A hierarchy with one row per unit at each level, explicit parent keys, and a
declared unit of analysis.

**Rationale.** Cardiology data nest deeply. A patient has encounters, an encounter has
recordings, a recording has leads, and a lead has beats. Every join must respect the level.

**Procedure.**

1. Establish the identifier chain: `patient_id`, `encounter_id`, `recording_id`, `lead_id`,
   `beat_index`.
2. Declare the unit of analysis before modelling and write it in the protocol.
3. Aggregate upward with an explicit rule. The mean RR is not the mean of instantaneous
   heart rates; compute rate as 60,000 divided by the mean RR in milliseconds.
4. Verify that no derived field mixes levels without an aggregation step in between.

**Branch point.** If the label is patient-level and the features are recording-level, the
choice is between one recording per patient (first, worst, or random) and a
multiple-instance formulation. Record the choice; the three produce different models.

**Edge case.** Duplicate recordings. Deduplicate on signal hash plus a time window, not on
the report identifier, which is regenerated on each transmission.

### Step 3 — Anchor the encounter clock

**Input.** Order dates, collection dates, report signature dates, admission and discharge
dates, notification dates.

**Output.** One time-zero field per patient with the rule that produced it, plus the
interval to every candidate event.

**Rationale.** Time zero is the contract between the data and the question. A cohort with
three time zeros has three populations and one table.

**Procedure.**

1. Name the clock: notification, test order, first medical contact, admission, discharge, or
   device implant.
2. For AF screening, time zero is the notification, not the patch result and not the
   prescription.
3. For heart failure readmission, time zero is discharge, and the window starts the next day.
4. For ACS, time zero is symptom onset if available, otherwise first medical contact; record
   which was used.
5. Verify that every feature timestamp precedes time zero. Sample one hundred rows by hand.

**Branch point.** If the question is prognostic and time zero is a clinic visit, a patient
who deteriorates before the visit never enters the cohort. That is incidence-density
sampling and it changes the estimand.

**Edge case.** Report signature lag. A patch result signed four days after the recording
shifts the event date by four days, which is 13 percent of a 30-day window. State which date
the endpoint uses.

### Step 4 — Qualify the signal

**Input.** Raw waveforms, lead identifiers, sampling rate, filter settings, calibration
pulse, device and firmware identifiers.

**Output.** A quality disposition per recording, with quality flags stored as columns.

**Rationale.** An ECG that passes visual inspection can still be wrong in a way that is
computable. Limb-lead reversal is the canonical case: the recording is clean, the rhythm is
regular, the morphology is inverted, and a model trained on morphology learns the reversal.

**Procedure.**

1. Verify the calibration pulse and the sampling rate. A record without a documented rate
   cannot be resampled consistently.
2. Test Einthoven's law: at every sample, lead II should equal lead I plus lead III within
   the noise floor. A systematic violation indicates a limb-lead problem.
3. Test the reversal signatures. In right-arm and left-arm exchange, lead I inverts (P wave
   and QRS both negative), aVR and aVL swap, and leads II and III swap. In left-arm and
   left-leg exchange, lead III inverts, aVL and aVF swap, and lead I is unchanged.
4. Detect baseline wander and muscle artifact, and record the filter applied, because a 0.67
   Hz high-pass used for monitoring removes ST information that a diagnostic filter keeps.

**Decision table for signal quality.** Six outputs, because "accept or reject" discards
information the analyst needs later.

| Observation | Action | Stored value | Consequence if mishandled |
|---|---|---|---|
| All leads present, II = I + III within noise | Accept | `good` | None |
| Lead I inverted, aVR and aVL swapped | Re-record; if unavailable, exclude from morphology models and keep for rate-only analysis | `limb_reversal_ra_la` | Inverted P and QRS learned as pathology |
| Lead III inverted, aVL and aVF swapped, lead I normal | Same disposition as above | `limb_reversal_la_ll` | Axis and inferior infarct calls wrong |
| II differs from I + III beyond the noise floor | Flag and re-record; suspect electrode impedance | `limb_inconsistent` | Silent morphology corruption |
| Artifact over more than 20 percent of the record | Keep with a flag; exclude from rhythm classification if it overlaps the only candidate episode | `artifact_partial` | False AF calls from movement |
| Fewer than 12 leads, or sampling below 250 Hz | Accept only for rate and interval summaries | `reduced_lead_set` | QRS duration and morphology unreliable |

**Branch point.** If a reversal is detected, the repair is a re-record, not a digital swap.
A software swap fixes the morphology but hides the reversal rate, which is a safety metric.

**Edge case.** A genuine right-axis deviation with a negative lead I. Require two independent
features, P-wave axis and QRS axis, before calling a reversal, and send ambiguous cases to
review.

### Step 5 — Derive intervals and measurements

**Input.** Qualified waveforms, imaging studies, and measurement tables.

**Output.** RR series, QRS duration, QT, heart rate, LVEF, and the method for each.

**Rationale.** Every derived measurement is algorithm-dependent. Two QRS detectors place the
offset fiducial point differently, and the difference is a systematic bias, not noise.

**Procedure.**

1. Detect R peaks with a documented algorithm and version. Store peak times, not only
   intervals.
2. Remove ectopic beats before computing heart rate variability, and record how many went.
3. Compute the mean RR as the arithmetic mean of the interval series, then convert to rate.
4. For LVEF, record the modality, the method, and the reader. A visual estimate and a Simpson
   biplane measurement differ by more than the treatment thresholds they are compared with.
5. Bin LVEF only at clinical boundaries: at or below 40, 41 to 49, at or above 50.

**Branch point.** If LVEF is missing, do not impute a category. A missing echo is a clinical
decision, and imputation mixes the decision with the measurement.

**Edge case.** Automated LVEF is a point estimate whose repeatability is about plus or minus
five points. A patient at 42 percent sits on a boundary and the assignment is a coin flip.

### Step 6 — Establish the label and the reference standard

**Input.** ECG reads, patch reports, device interrogations, adjudication packets, claims.

**Output.** One label per unit with the reference standard, the reader, the date, and the
agreement statistic.

**Rationale.** The label is the largest source of error in cardiology machine learning and
the one least often measured. An algorithm cannot beat the ceiling set by the disagreement
between two cardiologists reading the same strip.

**Procedure.**

1. Name the reference standard: a cardiologist over-read, a committee, a device
   interrogation, or a clinical diagnosis. Each answers a different question.
2. Define the episode rule: minimum AF duration, maximum gap within one episode, and whether
   the episode must be symptomatic.
3. Have a second reader label a random sample of at least 200 records or 10 percent. Compute
   Cohen's kappa for a binary label and quadratic weighted kappa for an ordinal one.
4. Report the agreement as the ceiling. A model at 0.88 against a reference whose
   inter-reader kappa is 0.62 has not been characterised.
5. For committee endpoints, write the charter: what counts, who decides, how ties break, and
   whether the committee sees the exposure. Blind the adjudicators.

**Branch point.** If the reference standard is a diagnosis code rather than a reading, the
label measures coding behaviour. That is legitimate for a billing question and illegitimate
for a physiological one.

**Edge case.** Adjudication disagreement is not random. Committees disagree more on
borderline troponin elevations and on events near the end of follow-up. Report disagreement
by category, not as a single kappa.

### Step 7 — Lock the instrument

**Input.** Device identifiers, firmware versions, preprocessing code, model weights.

**Output.** A locked pipeline specification with a change-control log and a revalidation
trigger.

**Rationale.** In cardiology the instrument is in the patient's pocket and the manufacturer
ships firmware updates. A model validated on one sensor generation is a different product on
the next, even though the code has not changed.

**Procedure.**

1. Record device make, model, sensor generation, and firmware version for every recording.
   If the field is absent, the recording cannot be assigned to an instrument.
2. Freeze the preprocessing chain: rate conversion, filter type and cutoff, normalisation,
   and windowing. Store it as code plus a version.
3. Define what counts as a substantial change: a sensor change, a sampling-rate change, a
   filter change, a change in the beat detector, or a retrained model.
4. Specify the evidence required after a substantial change: revalidation on recordings from
   the new device generation, with the same label definition.
5. Record every change with a date and a reason, and keep the superseded pipeline
   reproducible.

**Branch point.** If the vendor cannot supply firmware versions per recording, performance
cannot be attributed to a model change or an instrument change. Report a pooled estimate
with heterogeneity rather than a per-version claim.

**Edge case.** A firmware update that changes only the notification threshold. Discrimination
is unchanged and deployed PPV moves. This looks like a settings change and is not.

### Step 8 — Choose the operating threshold at deployment prevalence

**Input.** Sensitivity and specificity at candidate thresholds, deployment prevalence, and
the harm ratio.

**Output.** A threshold, the resulting PPV and NPV at deployment prevalence, and the expected
alert volume per thousand patients.

**Rationale.** The threshold is the only lever that converts a model into a workload. A
threshold chosen to maximise the Youden index in a validation cohort produces an alert
volume the clinic cannot staff when prevalence falls.

**Procedure.**

1. Compute sensitivity and specificity at every candidate threshold on the calibration set,
   not the test set.
2. For each threshold, compute PPV and NPV at deployment prevalence from counts, not from
   proportions.
3. Compute alerts and true positives per thousand patients at each threshold.
4. Compute confirmations per day at the programme's enrolment volume.
5. Select the threshold that satisfies the harm ratio and the staffing constraint together.
   If none does, the programme design is wrong, not the model.

**Decision table for threshold selection.** Four outputs, because the right action depends
on the setting rather than the metric.

| Deployment prevalence | PPV at sens 0.95, spec 0.95 | Permitted action | Staffing |
|---:|---:|---|---|
| 0.5 percent, consumer screening | 8.7 percent | Notification prompts confirmation only; never treat on it | 11 confirmations per true case |
| 1 percent, primary care | 16.1 percent | Patch within one week; no therapy before confirmation | 6 per true case |
| 5 percent, cardiology clinic | 50.0 percent | Same-day confirmation; consider expedited review | 2 per true case |
| 20 percent, post-stroke | 82.6 percent | Confirmation is a formality; plan the decision in parallel | 1.2 per true case |

**Branch point.** If the contract specifies a sensitivity floor, work backwards: the floor
fixes the threshold and the prevalence fixes the PPV. Present the PPV as a consequence of the
contract, not as a model property.

**Edge case.** A threshold tuned on a different case mix. Case mix changes the
sensitivity-specificity trade-off, not only the prevalence, so recompute the curve.

### Step 9 — Apply the guideline epoch and compute the action

**Input.** Score inputs, the guideline version in force at the patient's decision date, and
the patient's contraindications.

**Output.** A score, a threshold, an action, and the version identifier for each.

**Rationale.** The guideline is part of the measurement. A model that predicts "should be
anticoagulated" is predicting a rule that changed twice in five years.

**Procedure.**

1. Record the guideline and version used for each decision, keyed on the decision date rather
   than the analysis date.
2. Compute the score from its components and store the components. A stored total cannot be
   recomputed under a new scoring rule.
3. Apply the threshold from the same version. Do not mix a 2023 score with a 2019 threshold.
4. Record contraindications and patient preference separately. A patient who declined is not
   a patient who was ineligible.
5. If the cohort spans a guideline change, restrict to one epoch or stratify and report.

**Guideline epoch table for anticoagulation in AF.**

| Epoch | Score | Threshold | What changed |
|---|---|---|---|
| 2014 ACC/AHA/HRS, 2019 focused update | CHA2DS2-VASc | At least 2 | Score-based; sex category counted |
| 2023 ACC/AHA/ACCP/HRS | CHA2DS2-VASc, read as annual risk | At least 2 percent per year, about 2 in men and 3 in women | Risk-based, not score-based |
| 2024 ESC | CHA2DS2-VA | At least 2 recommended, 1 to be considered | Sex category removed |

**Branch point.** If the cohort spans the 2023 or 2024 change, a treatment-rate comparison
across the boundary measures the guideline, not the patients. Report rates and score
distributions by epoch.

**Edge case.** A patient whose score sits exactly at the threshold. The intermediate band is
a shared-decision zone, so a model trained to predict treatment there learns preference.

### Step 10 — Validate externally and monitor after deployment

**Input.** The locked pipeline, an external dataset from a different site or device
generation, and a monitoring plan.

**Output.** An external performance report, a subgroup table, and a drift specification with
triggers.

**Rationale.** Internal validation measures how well the model fits the site that built it.
The only question a buyer has is what happens elsewhere.

**Procedure.**

1. Validate on a dataset from a different institution, a different device generation, or
   both. Report discrimination, calibration, and PPV at the external prevalence.
2. Report calibration slope and intercept, not only AUROC. A model can discriminate well and
   be systematically overconfident, which is fatal for a thresholded action.
3. Report performance by age, sex, race and ethnicity, and skin phototype for PPG devices,
   with subgroup counts, not only subgroup estimates.
4. Specify monitoring: monthly notification rate, confirmation rate, and PPV among confirmed
   cases, with a review trigger for each.
5. Pre-specify the response to a drift signal: recalibrate, re-threshold, or withdraw.

**Branch point.** If external performance degrades in calibration but not discrimination,
recalibration is sufficient. If discrimination degrades, the label or the instrument changed,
and recalibration will hide it.

**Edge case.** A drift signal caused by a change in the confirmation pathway rather than the
model. If the clinic stops ordering patches for low-risk patients, measured PPV rises with no
algorithm change.
## 6. Worked example

A health system proposes an AF screening programme using a wrist-worn PPG device. The vendor
reports sensitivity 0.95 and specificity 0.95 against a seven-day patch. The programme will
enrol 100,000 adults aged 65 and over from primary care, where expected AF prevalence is 0.5
percent. The action is a confirmatory patch, then anticoagulation if AF is confirmed and the
stroke risk threshold is met.

The programme manager asks: "how many alerts will we get, and how many will be real?" That
is a PPV question at deployment prevalence, and it is not in the vendor's submission.

### 6.1 The 2x2 at deployment prevalence

Hold sensitivity and specificity fixed and move the prevalence. At 0.5 percent, 100,000
adults contain 500 with AF and 99,500 without.

| Quantity | Calculation | Value |
|---|---|---:|
| Diseased | 100,000 x 0.005 | 500 |
| Not diseased | 100,000 - 500 | 99,500 |
| True positives | 500 x 0.95 | 475 |
| False negatives | 500 - 475 | 25 |
| False positives | 99,500 x 0.05 | 4,975 |
| True negatives | 99,500 - 4,975 | 94,525 |
| PPV | 475 / (475 + 4,975) | 8.72 percent |
| NPV | 94,525 / (94,525 + 25) | 99.97 percent |

The programme generates 5,450 alerts to find 475 cases. Ten of eleven alerts are false. The
NPV of 99.97 percent sounds excellent and is almost entirely a statement about prevalence.

### 6.2 The same test at cardiology-clinic prevalence

The identical sensitivity and specificity, applied to 100,000 patients at 5 percent
prevalence, gives 5,000 diseased and 95,000 not diseased.

```text
TP  = 5,000 x 0.95 = 4,750
FP  = 95,000 x 0.05 = 4,750
PPV = 4,750 / (4,750 + 4,750) = 0.5000 -> 50.0%
NPV = 90,250 / (90,250 + 250) = 0.99724 -> 99.72%
```

Same test, same sensitivity, same specificity, and a PPV that moves from 8.72 to 50.0
percent. Nothing about the model changed. The population changed, and the calculation that
separates a defensible deployment claim from a misleading one takes four multiplications.

### 6.3 Diagnostic one: the enriched PPV

The vendor's clinical study reports a PPV of 98.2 percent. That number is real, published,
and not applicable here. It was computed among participants who had a second irregular-rhythm
detection while wearing a confirmatory patch, which selects for high arrhythmia burden. The
unconditional yield in the same study was 32.2 percent of patch wearers with AF on the patch.

Compute the programme's budget under each number, assuming each false positive costs one patch
at 150 dollars and one clinic visit at 200 dollars, so 350 dollars.

```text
At the quoted PPV of 0.982, per 1,000 alerts:
  false positives = 1,000 x (1 - 0.982) = 18
  cost            = 18 x 350 = 6,300 dollars

At the deployment PPV of 0.0872, per 1,000 alerts:
  false positives = 1,000 x (1 - 0.0872) = 912.8 -> 913
  cost            = 913 x 350 = 319,550 dollars
```

The programme budgets 6,300 dollars per thousand alerts and spends 319,550. The error is
313,250 dollars per thousand alerts, a factor of 50.7. At 5,450 alerts the first-year
difference is about 1.7 million dollars, and nurse time differs by the same factor: 18
confirmations at twenty minutes each is six hours, while 913 is 304 hours, or eight
nurse-weeks.

The diagnostic requires no new data. Take the PPV the vendor quoted, find the population in
which it was computed, and recompute at the population you serve. If prevalences differ by a
factor of ten, PPV differs by roughly the same factor.

### 6.4 Diagnostic two: verification bias in the confirmation pathway

The second diagnostic asks who was verified. In the published consumer studies roughly one in
five notified participants returned a patch: 450 of 2,161 in the Apple Heart Study, 20.8
percent. AF was present in 34 percent of those who returned.

Two analyses can be built from one study, and they differ by a factor of nearly five.

```text
Analysis A: PPV among the verified
  = 0.34 -> 34.0 percent

Analysis B: yield among all notified, treating non-returners as unverified
  = 0.208 x 0.34 = 0.0707 -> 7.1 percent
```

Analysis A answers "if a patient is verified, what is the chance of AF?" Analysis B answers
"if a patient is notified, what is the chance we confirm AF?" The programme manager needs B.

Now quantify the assumption rather than choosing an extreme. Suppose non-returners have half
the AF prevalence of returners, because returners are more likely to be symptomatic.

```text
Overall AF among notified = 0.208 x 0.34 + 0.792 x 0.17
                         = 0.0707 + 0.1346
                         = 0.2053 -> 20.5 percent

Reported 34.0 percent, corrected 20.5 percent.
Error = 13.5 percentage points, a 1.66-fold overstatement.
```

In a 100,000-person cohort with 520 notifications, the difference is 107 confirmed diagnoses
against 71, and 36 people told they had AF who did not.

The diagnostic is to compute the verification rate before any performance metric. A study
that verified 21 percent of positives has a sensitivity and a PPV that describe the 21
percent. If non-returners felt well the bias is upward; if they were too unwell it is
downward, and neither is safe to assume.

### 6.5 Vary one assumption: specificity

The vendor's specificity of 0.95 is a claim about a device in a study. Real-world specificity
falls with motion artifact, poor perfusion, and off-label wear. Hold prevalence at 0.5 percent
and sensitivity at 0.95, and move specificity.

| Specificity | False positives per 100,000 | PPV | Alerts per true case |
|---:|---:|---:|---:|
| 0.99 | 995 | 32.3 percent | 3.1 |
| 0.95 | 4,975 | 8.7 percent | 11.5 |
| 0.90 | 9,950 | 4.6 percent | 21.9 |

A five-point fall in specificity halves the PPV and doubles the confirmations per case. A
vendor reporting a five-point specificity advantage is reporting a two-fold difference in
clinic workload at deployment prevalence.

That is the varied assumption that changes the conclusion. At specificity 0.99 the programme
needs three confirmations per case and runs with a part-time nurse. At 0.90 it needs
twenty-two and cannot run without a dedicated pathway. "Is this product good enough" is
answerable only against a staffing plan.

### 6.6 The patient, carried to the decision

One notified participant is a 71-year-old man with treated hypertension and type 2 diabetes.
His patch confirms AF. The question is whether to anticoagulate.

| CHA2DS2-VASc component | Present | Points |
|---|---|---:|
| Congestive heart failure | No | 0 |
| Hypertension | Yes | 1 |
| Age 75 or older | No, age 71 | 0 |
| Diabetes mellitus | Yes | 1 |
| Stroke, transient ischaemic attack, or thromboembolism | No | 0 |
| Vascular disease | No | 0 |
| Age 65 to 74 | Yes | 1 |
| Sex category, female | No | 0 |
| Total | | 3 |

Under the 2023 ACC/AHA/ACCP/HRS guideline a score of 3 in a man exceeds the 2 percent annual
risk threshold, and anticoagulation is recommended. Under the 2024 ESC guideline CHA2DS2-VA
is also 3, because the sex point is zero for a man, which meets the recommended threshold of
2. Both agree here, which is why the version change is easy to miss.

His HAS-BLED score is 2: uncontrolled hypertension at 168 mmHg, and age above 65. A bleeding
score is not a reason to withhold anticoagulation; it is a prompt to correct modifiable
factors. The score is recorded, the pressure is treated, and the anticoagulant starts.

The cascade is where the arithmetic becomes sobering.

```text
Enrolled adults                                100,000
Notifications at 0.52 percent                      520
Patches returned at 20.8 percent                   108
AF confirmed at 34 percent                          37
Eligible for anticoagulation at 70 percent           26
Expected strokes per year at 3 percent            0.78
Strokes prevented per year, 60 percent reduction  0.47
Number needed to screen, one stroke per year   214,000
```

Every input is an assumption with a source, and the product is dominated by its two smallest
terms: the notification rate and the patch return rate. Improving sensitivity from 0.95 to
0.96 adds five detected cases per 100,000. Improving patch return from 20.8 to 30.8 percent
adds eighteen. The model is not the lever; the confirmation pathway is.

### 6.7 What the case does not establish

The case is a measurement example, not a screening recommendation. It does not show that AF
screening reduces stroke, which requires a randomised comparison with stroke as the endpoint.
It does not establish that this device performs as specified in a 65-and-over primary care
population, because the published studies enrolled younger, wealthier consumers.

| Choice | Value used here | Alternative that changes the answer | How to disclose it |
|---|---|---|---|
| Deployment prevalence | 0.5 percent | 5 percent in a clinic | State source and date |
| Operating point | 0.95, 0.95 | 0.99 specificity | Report with its threshold |
| Episode definition | Device default | 6-minute minimum | Record the device rule |
| Verification | Returners assumed representative | Returners enriched for symptoms | Report the verification rate |
| Time zero | Notification date | Patch result date | One time-zero field |
| Guideline epoch | 2023 ACC/AHA/ACCP/HRS | 2024 ESC CHA2DS2-VA | Version field per patient |
| Treatment effect | 60 percent relative reduction | Contemporary DOAC estimate | Label as an assumption |

An analysis that discloses these seven choices can be reproduced and criticised. One that
discloses none produces a number that cannot be checked.

## 7. Code, query, or calculation

The block below is standard-library Python. It implements CHA2DS2-VASc, the 2024 ESC
CHA2DS2-VA variant, HAS-BLED, and the prevalence conversion that section 6 performs by
hand. It is runnable as written with `python -I`.

**Assumptions.** Scores are computed from components the caller supplies; the function does
not infer them from text. The `predict` function holds sensitivity and specificity fixed and
varies prevalence, which is the correct counterfactual for a deployment question and the
wrong one for a case-mix question. Rounding uses banker's rounding, so a count that lands on
exactly .5 will round to the even integer; every count below is far enough from .5 that the
choice does not matter.

```python
"""Cardiology scores and prevalence-adjusted PPV. Standard library only."""

from dataclasses import dataclass


@dataclass
class Patient:
    age: int
    sex: str                  # "M" or "F"
    chf: bool = False
    hypertension: bool = False
    diabetes: bool = False
    stroke: bool = False
    vascular: bool = False


def cha2ds2_vasc(p):
    """2010 Chest derivation; scored as in the 2023 ACC/AHA/ACCP/HRS AF guideline."""
    pts = 0
    pts += 1 if p.chf else 0                                   # C
    pts += 1 if p.hypertension else 0                          # H
    pts += 2 if p.age >= 75 else 1 if p.age >= 65 else 0        # A2 / A
    pts += 1 if p.diabetes else 0                              # D
    pts += 2 if p.stroke else 0                                # S2
    pts += 1 if p.vascular else 0                              # V
    pts += 1 if p.sex == "F" else 0                            # Sc
    return pts


def cha2ds2_va(p):
    """2024 ESC variant: sex category removed, so women score one point less."""
    return cha2ds2_vasc(p) - (1 if p.sex == "F" else 0)


def has_bled(uncontrolled_htn, abnormal_renal, abnormal_liver, stroke,
             bleeding, labile_inr, age, drugs, alcohol):
    """Pisters et al., Chest 2010. Maximum 9."""
    s = 0
    s += 1 if uncontrolled_htn else 0
    s += 1 if abnormal_renal else 0
    s += 1 if abnormal_liver else 0
    s += 1 if stroke else 0
    s += 1 if bleeding else 0
    s += 1 if labile_inr else 0
    s += 1 if age > 65 else 0
    s += 1 if drugs else 0
    s += 1 if alcohol else 0
    return s


def predict(sens, spec, prevalence, n=100_000):
    """2x2 counts at a stated prevalence, with sens and spec held fixed."""
    diseased = round(n * prevalence)
    healthy = n - diseased
    tp = round(diseased * sens)
    fp = round(healthy * (1.0 - spec))
    fn = diseased - tp
    tn = healthy - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "ppv": tp / (tp + fp), "npv": tn / (tn + fn)}


print("== Patient ==")
p = Patient(age=71, sex="M", hypertension=True, diabetes=True)
print("age 71 M, hypertension, diabetes")
print("CHA2DS2-VASc (2023 ACC/AHA):", cha2ds2_vasc(p))
print("CHA2DS2-VA  (2024 ESC)     :", cha2ds2_va(p))
print("HAS-BLED (SBP 168, age >65):",
      has_bled(True, False, False, False, False, False, p.age, False, False))

print()
print("== PPV at fixed sensitivity 0.95 and specificity 0.95 ==")
print(f"{'prevalence':>10} {'diseased':>9} {'TP':>6} {'FP':>7} "
      f"{'PPV':>8} {'NPV':>8}")
for prev in (0.005, 0.01, 0.05, 0.20):
    r = predict(0.95, 0.95, prev)
    print(f"{prev:>9.1%} {round(100_000 * prev):>9} {r['tp']:>6} "
          f"{r['fp']:>7} {r['ppv']:>7.1%} {r['npv']:>8.2%}")

print()
print("== Specificity sensitivity analysis at prevalence 0.5% ==")
for spec in (0.99, 0.95, 0.90):
    r = predict(0.95, spec, 0.005)
    print(f"spec {spec:.2f}: FP={r['fp']:>6} PPV={r['ppv']:>6.2%}")
```

Expected output:

```text
== Patient ==
age 71 M, hypertension, diabetes
CHA2DS2-VASc (2023 ACC/AHA): 3
CHA2DS2-VA  (2024 ESC)     : 3
HAS-BLED (SBP 168, age >65): 2

== PPV at fixed sensitivity 0.95 and specificity 0.95 ==
prevalence  diseased     TP      FP      PPV      NPV
     0.5%       500    475    4975    8.7%   99.97%
     1.0%      1000    950    4950   16.1%   99.95%
     5.0%      5000   4750    4750   50.0%   99.72%
    20.0%     20000  19000    4000   82.6%   98.70%

== Specificity sensitivity analysis at prevalence 0.5% ==
spec 0.99: FP=   995 PPV=32.31%
spec 0.95: FP=  4975 PPV= 8.72%
spec 0.90: FP=  9950 PPV= 4.56%
```

Two details matter more than the arithmetic. First, the PPV column is computed from counts at
each prevalence, not by applying a formula to the validation proportions, so the rounding is
visible and auditable. Second, the table reproduces the values quoted in section 6 without a
separate code path, which means a reader can check the worked example against the program.

### 7.1 What the code does not do

It does not infer score components from clinical text, which is the step where most real
pipelines introduce error. It does not implement the 2023 guideline's risk modifiers, which
are qualitative and applied in shared decision-making. It does not handle the paediatric or
valvular exceptions to the score. It does not implement a calibration step, so the
`predict` function assumes the sensitivity and specificity are transportable, which is the
assumption section 11 asks you to test rather than accept.

## 8. Data, standards, and regulatory context

### 8.1 Guidelines that define the label and the action

Cardiology's clinical definitions are set by guideline committees that revise on multi-year
cycles, and the revision date is part of the measurement.

| Document | Citation | What it fixes | Date |
|---|---|---|---|
| 2022 AHA/ACC/HFSA heart failure guideline | Heidenreich et al., Circulation 2022;145:e895-e1032 | Stages A to D; HFrEF at or below 40 percent, HFmrEF 41 to 49, HFpEF at or above 50, HFimpEF as a new category | Published 1 April 2022 |
| 2023 ACC/AHA/ACCP/HRS atrial fibrillation guideline | Joglar et al., Circulation 2024;149:e1-e156 | Anticoagulation at estimated annual risk of 2 percent or more; risk-based rather than score-based | Published November 2023 |
| 2025 ACC/AHA/ACEP/NAEMSP/SCAI acute coronary syndrome guideline | Rao et al., Circulation 2025;151:e771-e862 | ACS definitions, troponin pathways, revascularisation | Published online 27 February 2025 |
| 2021 ESC heart failure guideline and 2023 focused update | McDonagh et al., Eur Heart J 2021;42:3599-3726; 2023;44:3627-3639 | European HF categories and therapy | 2021; update 25 August 2023 |
| 2023 ESC acute coronary syndrome guideline | Byrne et al., Eur Heart J 2023;44:3720-3826 | European ACS pathway | 2023 |
| 2024 ESC atrial fibrillation guideline | Van Gelder et al., Eur Heart J 2024;45:3314-3414 | CHA2DS2-VA replaces CHA2DS2-VASc; sex category removed | 30 August 2024 |

> **Note on a citation that circulates widely.** The ACC/AHA acute coronary syndrome
> guideline is frequently cited as a 2023 document. The current ACC/AHA/ACEP/NAEMSP/SCAI
> guideline is the 2025 version; the 2023 document with that scope is the ESC guideline.
> Verify the issuing body, not only the year, before quoting a threshold.

The heart failure categories illustrate why the guideline is a data structure. A patient with
a prior LVEF of 35 percent and a current LVEF of 45 percent is HFimpEF, not HFmrEF, and the
2022 guideline recommends continuing HFrEF therapy. A warehouse that stores one LVEF per
patient cannot represent that patient, and a cohort query for "HFmrEF" will include her.

### 8.2 Data standards

| Standard | Scope | Binding or convention | What it does not constrain |
|---|---|---|---|
| SCP-ECG (EN 1064) | ECG exchange format | Convention, adopted by equipment vendors | Signal quality or lead placement |
| DICOM waveform | Physiological signal storage | Convention outside imaging networks | Annotation semantics |
| HL7 FHIR Observation and DiagnosticReport | Messaging of results | Convention unless a purchaser requires it | Whether the value is correct |
| LOINC | Observation codes for ECG and echo measures | Convention; required by some payers | Measurement method |
| SNOMED CT | Clinical findings and procedures | Convention in the US, mandated in some systems | Coding completeness |
| ICD-10-CM | Diagnosis and procedure codes | Binding for billing | Physiological truth |
| AHA/ACC/HRS guideline definitions | Clinical endpoints | Convention that trials and regulators adopt | Adjudication implementation |

The distinction matters commercially. A FHIR-conformant interface can carry a QRS duration
with the wrong fiducial point and a stage group with the wrong edition. Conformance is a
statement about syntax.

### 8.3 Device regulation

An ECG analysis algorithm, an arrhythmia detector, and a heart failure risk score are
generally medical devices in the United States. The pathways are 510(k) for substantial
equivalence to a predicate, De Novo for a novel device of low-to-moderate risk, and premarket
approval for high-risk devices. A consumer product can remain outside device regulation only
under the general wellness policy, which requires that it make no diagnostic claim; a
notification that leads to anticoagulation is not wellness.

Software can be exempt from device regulation as clinical decision support only if it meets
all four statutory criteria: it does not acquire, process, or analyse a medical image or
signal; it displays, analyses, or prints medical information; it supports rather than
replaces clinician judgement; and it allows the clinician to independently review the basis
for the recommendation. An algorithm that analyses the ECG signal itself fails the first
criterion, which is why most ECG models cannot use the CDS exemption.

| Product type | Likely pathway | Evidence expected | Precedent |
|---|---|---|---|
| AF detection from PPG with notification | De Novo or 510(k) | Clinical performance study against a patch, with PPV | Consumer rhythm notifications |
| ECG analysis software for over-reading | 510(k) | Substantial equivalence plus bench performance | Established ECG analysis predicates |
| LVEF estimation from echo video | De Novo | Multi-site concordance against reference reads | Echo quantification tools |
| Anticoagulation recommendation | Device, unless all four CDS criteria are met | Clinical validation plus workflow evidence | Rarely cleared as CDS |

> **Note:** Regulatory status changes, and the FDA's position on software functions has been
> revised more than once. Confirm the current guidance before a submission. Nothing in this
> section is legal advice.

## 9. Industry practice and case evidence

The cases below are documented public events. Each separates what the publication reports
from what a data team may reasonably infer, because the transferable part is usually the
evaluation design rather than the headline metric.

**Apple Heart Study (Perez et al., New England Journal of Medicine, 2019).** Documented: 419,297
participants, 2,161 notifications (0.52 percent) over a median 117 days, 450 participants
(20.8 percent of those notified) returned an analysable ECG patch, and AF was present in 34
percent of them. PPV was 0.84 for AF on a concurrent ECG at the time of a later notification,
and 0.71 at the tachogram level. Inference: the study demonstrates that a consumer device can
generate a confirmable signal at scale, and it demonstrates the limits of that claim, because
the authors state the study was not designed to measure sensitivity or specificity and did
not establish a stroke benefit.

**Fitbit Heart Study (Lubitz et al., Circulation, 2022).** Documented: 455,699 participants,
4,728 irregular-rhythm notifications (1.0 percent), 1,057 notified participants with an
analysable patch, and AF in 32.2 percent of them. The primary PPV was 98.2 percent, computed
among 225 participants who had a second notification during patch monitoring. Exploratory
sensitivity was 67.6 percent and specificity 98.4 percent. Inference: the 98.2 percent and
the 32.2 percent are both correct and answer different questions, which is exactly the
distinction section 6.3 exploits. The authors state that the enriched population may
overstate performance for initial detection.

**AI-ECG for low ejection fraction (Attia et al., Nature Medicine, 2019).** Documented: a
convolutional network detected left ventricular systolic dysfunction from a standard 12-lead
ECG in a single-centre Mayo Clinic cohort. Inference: the model creates a screening pathway
to echocardiography, not a replacement for imaging, and the case is a template for using a
cheap test to triage an expensive one. The published performance is a single-centre estimate
on prevalent disease, and it does not transfer to a different device population without
revalidation.

**EchoNet-Dynamic (Ouyang et al., Nature, 2020).** Documented: 10,030 apical four-chamber
echocardiogram videos with ejection fraction and survival labels, used to train a video model
that estimates ejection fraction. Inference: video-native modelling avoids hand-crafted
measurement and inherits the acquisition variability of the view. Single-centre data, view
dependence, and vendor-specific acquisition restrict transportability, and the model's output
is a measurement, not a diagnosis.

**TRUST trial (Varma et al., Circulation, 2010).** Documented: automatic remote monitoring of
implantable cardioverter-defibrillators reduced in-person follow-up visits without a
significant increase in adverse events. Inference: remote data can substitute for some
scheduled visits, and the substitution requires a structured alert and clinic workflow. The
trial evaluated device follow-up, not an algorithm, which is why it remains the cleanest
evidence that the pathway rather than the classifier determines the benefit.

The pattern across the five cases is consistent. The documented facts are about detection,
agreement, and workflow. The inference that vendors make is about outcomes. Between the two
sit a confirmation pathway, a treatment decision, and a randomised comparison, and no
detection study substitutes for them.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| PPV quoted at validation prevalence | Model card has no prevalence field | PPV depends on prevalence; validation cohort is enriched | Programme understaffed by an order of magnitude | Report PPV at deployment prevalence |
| Lead reversal undetected | Lead I negative with a normal P axis | Electrodes exchanged; morphology inverted | Inverted P and QRS learned as pathology | Test Einthoven's law on every record |
| Firmware change unrecorded | Notification rate shifts at a software release | Sensor or threshold changed, model unchanged | Performance change misattributed to the population | Version every recording |
| Guideline epoch mixed | Score distribution shifts at a calendar year | Two thresholds pooled | Treatment rate measures the guideline | Store epoch per patient |
| Adjudication disagreement unmeasured | No second reader, no kappa | Committee judgement treated as truth | Effect estimates contain adjudication noise | Double-read a sample; report kappa |
| Verification bias | Confirmation rate not reported | Only returners verified | PPV describes the returners | Report verification rate and sensitivity analysis |
| Time zero drift | Median follow-up differs from the published value | Index date varies by patient | Survival and incidence not comparable | One time-zero rule; timestamp audit |
| Episode-duration rule unrecorded | AF prevalence differs from the literature by a factor | Devices use 30 s, 5 min, 6 min, or 24 h rules | Cohorts of different diseases pooled | Record the device rule per record |
| LVEF treated as precise | Categories assigned within 2 points of a boundary | Repeatability about 5 points | HFrEF and HFmrEF misclassified | Report a range near boundaries |
| Alert burden ignored | Alerts per patient-day not measured | Threshold set on discrimination only | Clinicians override the alert | Report alerts per patient-day and override rate |
| Calibration unmeasured | Only AUROC reported | Model overconfident after transport | Thresholded action fires too often | Report slope, intercept, Brier score |
| Subgroup performance unreported | No counts by skin phototype or age | PPG signal quality varies with perfusion and pigment | Systematic miss in under-represented groups | Report subgroup counts and performance |
| Composite endpoint hides harm | MACE improved, components not reported | Components have different mechanisms | A harm in one component is masked | Report each component separately |
| Device-derived endpoint censored at last contact | Interrogation interval long and variable | Events in the gap are unobserved | Event rate underestimated | Censor at last interrogation |

Two red flags deserve emphasis because they are cheap to check.

The first is a notification rate that matches the validation study. Real deployment
populations differ in age, comorbidity, and device wear time; an exact match usually means the
deployment population was selected to reproduce the study, or the rate was copied from it.

The second is a PPV reported without a verification rate. A PPV computed on 20 percent of
positives is a statement about that 20 percent. If the verification rate is not in the
document, the PPV cannot be interpreted, and the correct response is to ask for it rather
than to adjust for it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Sensitivity and specificity transport across populations | The prevalence conversion holds them fixed | Case mix changes the operating point | Recompute the curve on a local sample |
| Deployment prevalence is known | PPV is undefined without it | Enrolment criteria differ from the estimate | Pilot the first 1,000 patients and measure |
| The reference standard is stable | Labels must mean one thing | Reader drift and guideline change | Double-read a sample each year |
| Verification is representative | PPV is estimated on verified cases | Returners differ from non-returners | Compare returners and non-returners on observables |
| The device is the device that was validated | Firmware and sensor define the input | Silent hardware or firmware change | Version every recording |
| Time zero is recorded per patient | Survival needs a common origin | Index date inferred from the first row | Timestamp audit on 100 rows |
| Adjudication is blinded | Exposure knowledge biases event calls | Committee sees treatment arm | Audit the charter and the packet |
| LVEF categories are stable near boundaries | Treatment thresholds are categorical | Measurement repeatability about 5 points | Report a range within 5 points of a boundary |
| The guideline epoch is constant within a comparison | Thresholds define the action | Mid-cohort guideline revision | Cross-tab year by epoch |
| Alert volume is manageable | A model is a workload | Threshold set without a staffing model | Compute confirmations per clinic day |

The largest limit is not in the table. Cardiology data are generated by care and by consumer
behaviour, not by measurement. A patient who never wears the watch contributes nothing, a
patient who ignores the notification is never confirmed, and a patient who cannot afford the
anticoagulant is never treated. Each absence is a function of access and adherence, and a
model trained on the observed subset encodes both as if they were physiology.

A second limit is that the confirmation pathway is outside the model's control and dominates
its value. Section 6.6 shows a cascade in which a one-point sensitivity gain adds five
detections and a ten-point return-rate gain adds eighteen. Any business case that prices the
algorithm and not the pathway has priced the smaller term.

## 12. Exercises

### Exercise 1 — Score two patients under two guidelines (recall, 25 minutes)

**Deliverable.** CHA2DS2-VASc and CHA2DS2-VA for a 78-year-old woman with heart failure and
hypertension, and for a 66-year-old man with diabetes, each with the component breakdown and
the recommendation under the 2023 ACC/AHA and 2024 ESC documents.

**Acceptance criteria.** Components are listed separately; the version is named with its
publication date; the answer states whether the recommendation differs between versions and
why.

### Exercise 2 — Build a prevalence table (application, 40 minutes)

**Deliverable.** A 2x2 count table and PPV/NPV for sensitivity 0.90 and specificity 0.97 at
prevalences of 0.2, 1, 5, and 15 percent, with every count shown.

**Acceptance criteria.** Counts sum to the cohort size in every row; PPV is computed from
counts rather than from a formula on proportions; the write-up names the prevalence at which
the test becomes usable for a treat-on-result action.

### Exercise 3 — Anchor three cohorts (application, 30 minutes)

**Deliverable.** A table naming time zero, the event, the censoring rule, and the clock for
an AF screening cohort, a heart failure readmission cohort, and an ACS cohort.

**Acceptance criteria.** Each time zero is a named event rather than a table; no cohort uses
more than one clock; the censoring rule is stated in one sentence per cohort.

### Exercise 4 — Detect a lead reversal (application, 35 minutes)

**Deliverable.** A short script or checklist that tests Einthoven's law and the two limb-lead
reversal signatures on a supplied set of 12-lead records, plus a count of suspected reversals.

**Acceptance criteria.** The test is quantitative and states its noise threshold; the output
distinguishes right-arm/left-arm from left-arm/left-leg patterns; the write-up states why a
software channel swap is not the correct repair.

### Exercise 5 — Write a change-control plan (design, 40 minutes)

**Deliverable.** A one-page plan for a locked AF algorithm covering sensor change, firmware
change, threshold change, and model retraining, with the evidence required after each.

**Acceptance criteria.** Each change type has a defined trigger and a defined revalidation
set; the plan distinguishes a calibration failure from a discrimination failure; the plan
names who signs off.

### Exercise 6 — Quantify adjudication disagreement (critique, 45 minutes)

**Deliverable.** A double-read of at least 50 supplied event packets, an agreement statistic,
and a sensitivity analysis showing how the effect estimate moves under each reader's
adjudication.

**Acceptance criteria.** Agreement is reported with a confidence interval; disagreement is
broken down by event type; the sensitivity analysis reports the effect estimate under both
readings rather than an average.

### Exercise 7 — Classify heart failure cases (application, 30 minutes)

**Deliverable.** A table of ten supplied cases with stage, LVEF category including HFimpEF
where applicable, and the coding consequence of each classification.

**Acceptance criteria.** HFimpEF is distinguished from HFmrEF using the prior LVEF; filling
pressure evidence is required for HFmrEF and HFpEF; at least one case is flagged as
unclassifiable with a reason.

### Exercise 8 — Map the regulatory path (design, 30 minutes)

**Deliverable.** A one-page memo placing a described product on the 510(k), De Novo, PMA, or
CDS-exemption path, naming the intended use, the evidence required, and one precedent.

**Acceptance criteria.** The memo applies all four CDS criteria explicitly; it states what
the product does not claim; it names at least one binding guidance document with its date.

## 13. Self-check answers

**1. Scores for the two patients.** The 78-year-old woman with heart failure and hypertension
scores 1 for heart failure, 1 for hypertension, 2 for age 75 or older, and 1 for sex, for a
CHA2DS2-VASc of 5. Her CHA2DS2-VA is 4. Both are above the recommended threshold under either
guideline. The 66-year-old man with diabetes scores 1 for age 65 to 74 and 1 for diabetes, for
a CHA2DS2-VASc of 2 and a CHA2DS2-VA of 2, because the sex point is zero. Under the 2023
ACC/AHA guideline a score of 2 in a man is at the 2 percent annual risk threshold and
anticoagulation is recommended; under the 2024 ESC guideline a score of 2 is also recommended.
The versions agree on the decision and disagree on the recorded value only for the woman.

**2. Prevalence table at sensitivity 0.90 and specificity 0.97.** At 0.2 percent prevalence in
100,000 people: 200 diseased, 99,800 not diseased, 180 true positives, 20 false negatives,
2,994 false positives, 96,806 true negatives, PPV 180/3,174 = 5.67 percent, NPV 99.98 percent.
At 1 percent: 1,000 diseased, 990 true positives, 2,970 false positives, PPV 990/3,960 = 25.0
percent. At 5 percent: 5,000 diseased, 4,500 true positives, 2,850 false positives, PPV
4,500/7,350 = 61.2 percent. At 15 percent: 15,000 diseased, 13,500 true positives, 2,550 false
positives, PPV 13,500/16,050 = 84.1 percent. The test becomes usable for a treat-on-result
action at roughly 5 percent, where two of three positives are true.

**3. Time zero for the three cohorts.** AF screening: the notification date, because the
action follows the notification and the patch is the confirmation. Heart failure readmission:
the discharge date, with the window starting the next day. ACS: symptom onset if recorded,
otherwise first medical contact, and the choice must be constant within the cohort.

**4. Lead reversal detection.** Einthoven's law states that lead II equals lead I plus lead
III at every instant. A systematic violation beyond the noise floor indicates a limb-lead
problem. Right-arm and left-arm exchange inverts lead I, swaps aVR and aVL, and swaps leads II
and III. Left-arm and left-leg exchange inverts lead III, swaps aVL and aVF, and leaves lead I
unchanged. A software channel swap is not the repair because it leaves the impedance and noise
characteristics of the wrong configuration and hides the reversal rate.

**5. Change control.** A firmware change that alters the notification threshold changes the
deployed PPV without changing discrimination, so the revalidation must recompute PPV and
alert volume rather than AUROC. A sensor change alters the input distribution, so the
revalidation must include recordings from the new generation and report calibration slope. A
retrained model requires a new lock and a new external validation.

**6. Adjudication.** Report agreement as a kappa with a confidence interval, and report the
effect estimate under each reader separately. Averaging the two readings hides the
disagreement, which is the quantity a reviewer needs in order to judge whether the effect is
larger than the measurement noise.

**7. Heart failure classification.** A patient with a prior LVEF of 30 percent and a current
LVEF of 44 percent is HFimpEF, not HFmrEF, and HFrEF therapy continues. A patient at 47
percent without elevated filling pressures is not HFmrEF under the 2022 guideline, because
the category requires evidence of increased filling pressures. A patient at 52 percent with
elevated natriuretic peptides and no structural disease is not HFpEF.

**8. Regulatory mapping.** A PPG notification that leads to a confirmatory patch is a device
with a detection-aid intended use, and the evidence is a clinical performance study with PPV
against a patch. It cannot use the CDS exemption because it analyses a physiological signal,
which fails the first criterion. A model that displays a trend for a clinician to interpret,
without analysing a signal, may meet all four criteria.

**Rubric for Exercises 5 and 8.** A strong answer names the trigger, the evidence, and the
decision-maker for each change type, and states what the product does not claim. A weak answer
lists revalidation in general terms, cites a regulation without its scope, or treats a
threshold change as a settings change with no clinical consequence.

## 14. Interview questions

**1. Why can a 95 percent sensitive AF model still be wrong most of the time?**
Strong answer: PPV depends on prevalence, and at 0.5 percent prevalence a 0.95/0.95 test yields
8.7 percent; the arithmetic is shown. Weak answer: "because of false positives", with no
arithmetic and no mention of prevalence.

**2. A vendor reports 98 percent PPV. What is your first question?**
Strong answer: in which population, with what verification rate, and conditional on what
prior finding. Weak answer: "what is the confidence interval", which does not address the
estimand.

**3. What are the three clocks in a cardiology cohort?**
Strong answer: the cardiac clock at beat resolution, the encounter clock at test or admission
resolution, and the guideline epoch at multi-year resolution, with one named as primary. Weak
answer: "time zero and follow-up", which names one clock twice.

**4. How would you detect a limb-lead reversal without a human reader?**
Strong answer: Einthoven's law plus the two reversal signatures, with a noise threshold and a
human review path for ambiguous cases. Weak answer: "check the metadata", which is missing
precisely when a technician swaps electrodes.

**5. Is MACE a good endpoint for a model?**
Strong answer: only with the components reported separately, because they have different
mechanisms and different patient importance, and a composite can hide harm. Weak answer:
"yes, it is standard", which is why the problem persists.

**6. What changes when the device ships new firmware?**
Strong answer: the input distribution and possibly the notification threshold, so
discrimination, calibration, and alert volume must be re-established on the new generation.
Weak answer: "nothing, the model is the same", which confuses the code with the instrument.

**7. How do you handle a missing LVEF?**
Strong answer: do not impute a category; model the imaging decision; report the missing
fraction by subgroup and run a sensitivity analysis. Weak answer: impute the cohort mean,
which places every missing patient in a category no clinician assigned.

**8. What is HFimpEF and why does it matter for a query?**
Strong answer: prior LVEF at or below 40 percent and follow-up above 40 percent, a distinct
category that a single-LVEF warehouse cannot represent. Weak answer: "recovered EF", which is
the term the guideline replaced.

**9. Your external validation shows calibration drift but stable AUROC. What do you do?**
Strong answer: recalibrate, because the ranking is intact and the probabilities moved; check
whether the threshold change altered the case mix. Weak answer: retrain, which discards a
model that still discriminates.

**10. Can an ECG algorithm use the clinical decision support exemption?**
Strong answer: no, because it analyses a physiological signal, which fails the first of the
four criteria. Weak answer: "yes, if a clinician reviews it", which addresses only the fourth
criterion.

**11. What single field would you add to a cardiology warehouse first?**
Strong answer: the device and firmware version per recording, because without it no
performance change can be attributed. Weak answer: "more waveform features", which does not
fix the instrument record.

## 15. Key terms

- **AF** — atrial fibrillation, an irregularly irregular supraventricular rhythm.
- **AF burden** — proportion of monitored time spent in AF.
- **CHA2DS2-VASc** — stroke risk score from 0 to 9, including a sex category point.
- **CHA2DS2-VA** — 2024 ESC variant with the sex category removed.
- **HAS-BLED** — bleeding risk score from 0 to 9, used to prompt risk-factor correction.
- **HFrEF** — heart failure with LVEF at or below 40 percent.
- **HFmrEF** — heart failure with LVEF 41 to 49 percent plus elevated filling pressures.
- **HFpEF** — heart failure with LVEF at or above 50 percent plus elevated filling pressures.
- **HFimpEF** — prior LVEF at or below 40 percent with follow-up above 40 percent.
- **Stage A to D** — at risk, pre-HF, symptomatic, advanced heart failure.
- **Einthoven's law** — lead II equals lead I plus lead III at every instant.
- **PPG** — photoplethysmography, an optical pulse waveform.
- **Tachogram** — a beat-to-beat interval series.
- **Verification bias** — reference standard applied to a selected subset only.
- **PPV** — positive predictive value, true positives divided by all positives.
- **MACE** — major adverse cardiac events, a composite that must be decomposed.
- **Adjudication** — committee assignment of an endpoint under a written charter.
- **Locked algorithm** — frozen inputs, preprocessing, and weights under change control.
- **Firmware** — embedded device software whose version changes the input distribution.
- **510(k)** — premarket notification of substantial equivalence.
- **De Novo** — classification route for a novel low-to-moderate risk device.
- **PMA** — premarket approval for high-risk devices.
- **CDS** — clinical decision support meeting four statutory criteria.
- **SCP-ECG** — the EN 1064 standard for ECG exchange.
- **KCCQ** — Kansas City Cardiomyopathy Questionnaire, a symptom score.
- **HFimpEF therapy continuation** — the 2022 recommendation to keep HFrEF drugs after EF recovery.

## 16. Sources

- Heidenreich PA, et al. 2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure. Circulation. 2022;145:e895-e1032. https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063 (accessed 26 October 2026).
- Joglar JA, et al. 2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and Management of Atrial Fibrillation. Circulation. 2024;149:e1-e156. https://doi.org/10.1161/CIR.0000000000001193 (accessed 26 October 2026).
- Rao SV, et al. 2025 ACC/AHA/ACEP/NAEMSP/SCAI Guideline for the Management of Patients With Acute Coronary Syndromes. Circulation. 2025;151:e771-e862. https://doi.org/10.1161/CIR.0000000000001309 (accessed 26 October 2026).
- McDonagh TA, et al. 2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure. Eur Heart J. 2021;42:3599-3726, with the 2023 focused update at Eur Heart J. 2023;44:3627-3639. https://doi.org/10.1093/eurheartj/ehad195 (accessed 26 October 2026).
- Byrne RA, et al. 2023 ESC Guidelines for the management of acute coronary syndromes. Eur Heart J. 2023;44:3720-3826. https://academic.oup.com/eurheartj/article/44/38/3720/7243210 (accessed 26 October 2026).
- Van Gelder IC, et al. 2024 ESC Guidelines for the management of atrial fibrillation. Eur Heart J. 2024;45:3314-3414. https://doi.org/10.1093/eurheartj/ehae176 (accessed 26 October 2026).
- Perez MV, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. N Engl J Med. 2019;381:1909-1917. https://pmc.ncbi.nlm.nih.gov/articles/PMC8112605/ (accessed 26 October 2026).
- Lubitz SA, et al. Detection of atrial fibrillation in a large population using wearable devices: the Fitbit Heart Study. Circulation. 2022;146:1415-1424. https://doi.org/10.1161/CIRCULATIONAHA.122.060291 (accessed 26 October 2026).
- Attia ZI, et al. Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram. Nat Med. 2019. https://doi.org/10.1038/s41591-018-0240-2 (accessed 26 October 2026).
- Ouyang D, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020. https://doi.org/10.1038/s41586-020-2145-8 (accessed 26 October 2026).
- Varma N, et al. Efficacy and safety of automatic remote monitoring for implantable cardioverter-defibrillator follow-up: the TRUST trial. Circulation. 2010. https://doi.org/10.1161/CIRCULATIONAHA.110.937409 (accessed 26 October 2026).
- Lip GYH, et al. Refining clinical risk stratification for predicting stroke and thromboembolism in atrial fibrillation using a novel risk factor-based approach: the Euro Heart Survey on atrial fibrillation. Chest. 2010;137:263-272. https://doi.org/10.1378/chest.09-1584 (accessed 26 October 2026).
- Pisters R, et al. A novel user-friendly score (HAS-BLED) to assess 1-year risk of major bleeding in patients with atrial fibrillation. Chest. 2010;138:1093-1100. https://doi.org/10.1378/chest.10-0134 (accessed 26 October 2026).
- US Food and Drug Administration. General wellness policy for low-risk devices. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices (accessed 26 October 2026).
- US Food and Drug Administration. Clinical decision support software, guidance for industry and FDA staff. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software (accessed 26 October 2026).
- PhysioNet. PTB-XL, a large publicly available electrocardiography dataset. https://physionet.org/content/ptb-xl/ (accessed 26 October 2026).

> **Access note:** All sources were checked on 26 October 2026. Guideline editions and
> firmware versions change; confirm the version in force at the date of your analysis before
> citing a threshold.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Previous topic: [`01_oncology.md`](01_oncology.md)
- Next topic: [`03_pathology.md`](03_pathology.md)
- Cross-specialty comparison: [`07_cross-specialty-comparison.md`](07_cross-specialty-comparison.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Foundation material: [`../11_signals/`](../11_signals/)
- Device regulation: [`../14_devices_regulation/`](../14_devices_regulation/)
