# Cardiology and ECG Basics

> Module 01 — Clinical Foundations for Data Scientists
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: 02
> Version 2026.3

---

## 1. Why this topic exists

The electrocardiogram (ECG) is the most common cardiovascular test in the world. It is cheap,
non-invasive, and fast. It is also the most frequently misinterpreted waveform in clinical data
science, because the waveform is not the diagnosis: the diagnosis depends on lead placement,
calibration, the patient's baseline, the time of recording, and the clinical context.

Four industry problems recur.

**Problem 1: the 10-second strip is a sample, not a monitor.** A standard 12-lead ECG records 10
seconds. Paroxysmal atrial fibrillation that occurs between recordings is invisible. A normal ECG
does not exclude intermittent arrhythmia. A model trained on 10-second strips learns to classify
what is present in those 10 seconds, not what the patient experiences over a day.

**Problem 2: labels come from automated reads or reports.** Many datasets use the machine
interpretation stored in the ECG file. Automated reads have systematic errors, especially for
rhythm, infarction, and hypertrophy. Human over-reads are better but still variable. A model
trained on automated labels inherits their biases.

**Problem 3: acquisition differs.** Lead placement, electrode quality, body habitus, and device
filters change the waveform. A model trained on one device may fail on another even when the
underlying rhythm is the same. This is not a subtle effect; it is a common cause of transport
failure.

**Problem 4: the clinical decision is time-critical.** A correct diagnosis delivered after the
opportunity to treat is not valuable. A model that detects ST-elevation myocardial infarction
(STEMI) with high accuracy but a 20-minute latency may be worse than a simple threshold rule
that fires immediately.

This topic covers the cardiac cycle, the 12-lead layout, the systematic interpretation sequence,
common rhythms, the QT interval, artifact, and the data science failure modes specific to
waveforms.

## 2. Learning outcomes

By the end of this topic you can:

1. **Describe** the cardiac conduction pathway and the mechanical events of the cardiac cycle.
2. **Explain** the standard 12-lead layout, calibration, and paper speed.
3. **Compute** heart rate from an RR interval and **classify** rhythm, axis, and intervals.
4. **Measure** PR, QRS, and QT intervals and **compute** QTc with Bazett and Fridericia.
5. **Recognize** common rhythms and **distinguish** them from artifact.
6. **Apply** STEMI ST-elevation thresholds by lead, age, and sex.
7. **Critique** an ECG dataset for label source, lead configuration, device heterogeneity, and
   patient-level leakage.

## 3. The core mental model

The ECG is a surface recording of electrical activity produced by the heart. The signal travels
through the conduction system, and each phase produces a characteristic waveform.

```text
   SA node
      |
      v
   atria  ---------> P wave (atrial depolarization)
      |
      v
   AV node  -------> PR interval (conduction delay)
      |
      v
   His bundle -> bundle branches -> Purkinje fibers
      |
      v
   ventricles -----> QRS complex (ventricular depolarization)
      |
      v
   repolarization -> T wave (ventricular repolarization)
```

### 3.1 The cardiac cycle

```text
   diastole            systole             diastole
   |<-------->|<-------------------->|<-------->|
   filling    isovolumetric contraction     relaxation
              ejection
```

- **Diastole:** ventricles fill; the mitral and tricuspid valves are open.
- **Systole:** ventricles contract; the aortic and pulmonary valves open; blood is ejected.
- **Stroke volume:** the volume ejected per beat.
- **Cardiac output:** heart rate x stroke volume.

The ECG records electrical events, not mechanical events. Mechanical dysfunction can exist with a
normal ECG, and electrical abnormalities can exist without mechanical dysfunction.

### 3.2 The 12 leads

```text
   Limb leads:      I, II, III, aVR, aVL, aVF
   Precordial leads: V1, V2, V3, V4, V5, V6

   I    = right arm (-) to left arm (+)
   II   = right arm (-) to left leg (+)
   III  = left arm (-) to left leg (+)
   aVR  = augmented right
   aVL  = augmented left
   aVF  = augmented foot
   V1-V6 = anterior to lateral chest
```

Each lead views the heart from a different angle. An abnormality visible in one lead may be
invisible in another. A "normal" ECG requires all leads to be considered, not just lead II.

### 3.3 The paper and the grid

Standard calibration is 25 mm/s and 10 mm/mV:

```text
   1 small box = 1 mm = 0.04 s horizontally, 0.1 mV vertically
   1 large box = 5 mm = 0.20 s horizontally, 0.5 mV vertically
```

Heart rate from an RR interval:

```text
   HR = 300 / number of large boxes between R waves
   HR = 1500 / number of small boxes between R waves
```

For irregular rhythms, count the number of QRS complexes in a 10-second strip and multiply by 6.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| P wave | Atrial depolarization | Confused with artifact | mV, ms |
| PR interval | From P onset to QRS onset | Confused with PR segment | ms |
| QRS complex | Ventricular depolarization | Confused with QT | ms |
| T wave | Ventricular repolarization | Confused with U wave | mV |
| U wave | Late repolarization | Ignored or over-read | mV |
| QT interval | From QRS onset to T end | Measured on a single lead without validation | ms |
| QTc | Rate-corrected QT | Treated as measured | ms |
| RR interval | Time between R waves | Confused with heart rate | ms |
| Sinus rhythm | Rhythm originating in the SA node | Confused with normal ECG | — |
| Sinus bradycardia | Sinus rhythm below 60 beats/min | Defined differently in athletes | beats/min |
| Sinus tachycardia | Sinus rhythm above 100 beats/min | Confused with SVT | beats/min |
| Atrial fibrillation | Irregularly irregular rhythm without P waves | Confused with multifocal atrial tachycardia | — |
| Atrial flutter | Organized atrial activity, often 250-350/min | Confused with AF | beats/min |
| SVT | Supraventricular tachycardia | Used as a specific diagnosis | — |
| VT | Ventricular tachycardia | Confused with SVT with aberrancy | beats/min |
| VF | Ventricular fibrillation | Confused with asystole | — |
| First-degree AV block | PR above 200 ms | Treated as benign in all cases | ms |
| Second-degree AV block | Intermittent AV conduction | Confused with first-degree | — |
| Third-degree AV block | Complete AV dissociation | Missed when rate is normal | — |
| Bundle branch block | QRS at least 120 ms with morphology | Confused with ventricular rhythm | ms |
| Axis | Direction of ventricular depolarization | Computed without checking lead placement | degrees |
| Left axis deviation | Axis between -30 and -90 degrees | Confused with normal variant | degrees |
| Right axis deviation | Axis between +90 and +180 degrees | Confused with limb lead reversal | degrees |
| LVH | Left ventricular hypertrophy | Diagnosed by voltage alone | Voltage criteria |
| STEMI | ST-elevation myocardial infarction | Confused with pericarditis or early repolarization | — |
| NSTEMI | Non-ST-elevation myocardial infarction | Confused with unstable angina | — |
| ST elevation | Elevation of the ST segment | Measured at the wrong point | mm or mV |
| J point | Junction between QRS and ST segment | Confused with ST segment | ms |
| Early repolarization | Benign ST elevation pattern | Confused with STEMI | — |
| Pericarditis | Inflammation of the pericardium | Confused with STEMI | — |
| Artifact | Non-physiological signal | Ignored in model training | — |
| Baseline wander | Low-frequency drift | Confused with ST change | mV |
| Muscle tremor | High-frequency noise | Confused with fibrillation | — |
| Lead reversal | Misplacement of limb or precordial leads | Missed because the ECG looks plausible | — |
| Calibration | Scale of the recording | Assumed correct | mm/mV |
| Paper speed | Recording speed | Assumed 25 mm/s | mm/s |
| Automated interpretation | Machine-generated ECG read | Treated as ground truth | — |
| Over-read | Human interpretation | Treated as ground truth | — |

## 5. The mechanics, step by step

**Step 1. Verify calibration and paper speed.**

- Input: ECG header and grid.
- Operation: confirm 25 mm/s and 10 mm/mV; check the calibration pulse.
- Output: a calibration record.
- Reasoning: a recording at 50 mm/s doubles the apparent intervals; a calibration of 5 mm/mV
  halves the apparent amplitude. Both errors change every measurement.
- Branch point: some digital ECGs store samples at 500 Hz and display at a different scale; the
  header values govern.
- Analytic artifact: `sampling_rate_hz`, `paper_speed_mm_s`, `gain_mm_mv`.

**Step 2. Assess technical quality.**

- Input: the tracing.
- Operation: check baseline stability, noise, lead connectivity, and clipping.
- Output: a quality grade.
- Reasoning: artifact can mimic arrhythmia, ST change, and ischemia. A model trained without
  quality labels learns to classify noise.
- Branch point: if quality is poor, the correct output may be "uninterpretable," not a diagnosis.
- Analytic artifact: `quality_grade`, `artifact_flags`.

**Step 3. Determine the rate.**

- Input: RR intervals.
- Operation: use 300/large boxes for regular rhythms; count QRS complexes in 10 seconds and
  multiply by 6 for irregular rhythms.
- Output: beats per minute.
- Reasoning: the 300 rule assumes a regular rhythm. In atrial fibrillation, the rate varies
  beat to beat, and an average is more meaningful than a single interval.
- Branch point: a monitor may display an averaged rate that differs from the recorded strip.
- Analytic artifact: `heart_rate`, `rate_method`.

**Step 4. Determine the rhythm.**

- Input: P waves, QRS complexes, regularity, and their relationship.
- Operation: ask whether P waves are present, whether they are regular, whether each P is
  followed by a QRS, and whether the QRS is narrow or wide.
- Output: a rhythm classification.
- Reasoning: sinus rhythm requires a P wave before every QRS with a constant PR interval. An
  irregularly irregular rhythm without P waves is atrial fibrillation. A regular narrow-complex
  tachycardia without visible P waves may be SVT.
- Branch point: P waves may be hidden in the T wave; a different lead or a longer strip may
  reveal them.
- Analytic artifact: `rhythm`, `p_wave_present`, `regularity`, `qrs_width`.

**Step 5. Determine the axis.**

- Input: leads I and aVF.
- Operation: if both are positive, the axis is normal. If I is positive and aVF negative, check
  lead II to distinguish normal from left axis deviation. If I is negative and aVF positive, the
  axis is rightward.
- Output: an axis category in degrees.
- Reasoning: axis reflects the direction of ventricular depolarization. Left axis deviation can
  result from left anterior fascicular block, LVH, or inferior infarction. Right axis deviation
  can result from right ventricular hypertrophy or lateral infarction.
- Branch point: limb lead reversal can produce an abnormal axis with a otherwise normal tracing.
- Analytic artifact: `axis_degrees`, `axis_category`.

**Step 6. Measure the PR interval.**

- Input: P onset and QRS onset.
- Operation: measure in milliseconds on the lead with the clearest P wave.
- Output: PR interval.
- Reasoning: normal is 120-200 ms. A short PR suggests a pre-excitation syndrome; a long PR
  suggests first-degree AV block.
- Branch point: measurement varies with the lead and the observer; a 10 ms difference is not
  clinically meaningful.
- Analytic artifact: `pr_ms`.

**Step 7. Measure the QRS duration.**

- Input: QRS onset and offset.
- Operation: measure the widest complex in any lead.
- Output: QRS duration.
- Reasoning: normal is less than 120 ms; 120 ms or more suggests bundle branch block or
  ventricular origin. A wide QRS can mask infarction and alter the QT measurement.
- Branch point: automated measurements may differ from manual ones by 10-20 ms.
- Analytic artifact: `qrs_ms`, `bundle_branch_block`.

**Step 8. Assess chamber enlargement.**

- Input: voltage and morphology.
- Operation: apply voltage criteria (for example, Sokolow-Lyon for LVH).
- Output: a probability or category.
- Reasoning: voltage criteria are sensitive to age, sex, body habitus, and lead placement. They
  are not diagnostic. Obesity lowers voltage; young thin patients have higher voltage.
- Branch point: LVH criteria perform differently in Black and White populations; a single
  threshold is not transportable.
- Analytic artifact: `lvh_criteria`, `lvh_flag`.

**Step 9. Assess ischemia and infarction.**

- Input: ST segment, T wave, Q waves.
- Operation: measure ST elevation at the J point in contiguous leads; compare with thresholds.
- Output: an ischemia category.
- Reasoning: ST elevation in contiguous leads suggests acute injury. ST depression suggests
  ischemia or reciprocal change. Q waves suggest completed infarction. T-wave inversion is
  nonspecific.
- Branch point: early repolarization, pericarditis, and LVH can mimic STEMI.
- Analytic artifact: `st_elevation`, `st_depression`, `t_inversion`, `q_wave`.

**Step 10. Measure the QT interval and compute QTc.**

- Input: QT and RR.
- Operation: measure QT from QRS onset to T end; compute QTc with Bazett and Fridericia.
- Output: QTc in milliseconds.
- Reasoning: QT varies with heart rate. Bazett (QT / square root of RR) overcorrects at high
  rates and undercorrects at low rates. Fridericia (QT / cube root of RR) is less rate-dependent.
  The AHA/ACCF/HRS recommendations discourage Bazett for routine reporting and recommend a
  linear or power correction with age and sex adjustment.
- Branch point: in atrial fibrillation, the RR varies, so a single QTc is unreliable; average
  multiple beats or avoid correction.
- Analytic artifact: `qt_ms`, `rr_ms`, `qtc_bazett`, `qtc_fridericia`, `correction_method`.

**Step 11. Assess electrolyte and drug effects.**

- Input: morphology.
- Operation: look for peaked T waves (hyperkalemia), U waves (hypokalemia), widened QRS
  (hyperkalemia or sodium-channel blockade), and ST depression with downsloping shape (digoxin).
- Output: a set of pattern flags.
- Reasoning: these patterns are sensitive to the underlying concentration but not specific. A
  peaked T wave can be normal in a young athlete.
- Branch point: the same drug can produce different patterns at different concentrations.
- Analytic artifact: `electrolyte_pattern`, `drug_effect_pattern`.

**Step 12. Synthesize the interpretation.**

- Input: rate, rhythm, axis, intervals, morphology, and clinical context.
- Operation: combine into a structured impression.
- Output: an interpretation with a confidence and a recommendation.
- Reasoning: the ECG is one piece of evidence. A normal ECG in a patient with ongoing chest pain
  does not exclude acute coronary syndrome. The interpretation must state the context.
- Branch point: an urgent finding (STEMI, VT, complete heart block) changes the workflow.
- Analytic artifact: `impression`, `urgency`, `confidence`.

**Step 13. Compare with prior ECGs.**

- Input: current and prior tracings.
- Operation: assess change in rhythm, intervals, and morphology.
- Output: a comparison statement.
- Reasoning: a new bundle branch block or a new Q wave is more significant than a stable
  abnormality. A dataset without priors cannot reproduce this reasoning.
- Branch point: priors may be from a different device or lead configuration.
- Analytic artifact: `prior_ecg_uid`, `change_flag`.

**Step 14. Assign labels in a dataset.**

- Input: automated read, over-read, or report.
- Operation: choose the label source and record it.
- Output: a label with provenance.
- Reasoning: automated reads are available at scale but have systematic errors. Over-reads are
  better but expensive. A model trained on automated labels learns the algorithm's biases.
- Branch point: if the label source changes mid-dataset, the label distribution changes.
- Analytic artifact: `label_source`, `label_version`, `label_confidence`.

**Step 15. Handle the 10-second limitation.**

- Input: a single 10-second strip.
- Operation: state what the strip can and cannot show.
- Output: a scope statement.
- Reasoning: a 10-second strip can show the rhythm at that moment. It cannot exclude paroxysmal
  arrhythmia, transient ischemia, or intermittent block.
- Branch point: a normal strip in a patient with syncope does not exclude a dangerous arrhythmia.
- Analytic artifact: `recording_duration_s`, `scope_note`.

**Step 16. Represent the waveform properly.**

- Input: raw samples.
- Operation: store sampling rate, units, lead names, calibration, and filter settings.
- Output: a waveform record.
- Reasoning: a waveform without a sampling rate and units is not reproducible. A 500 Hz record
  resampled to 100 Hz loses high-frequency content that may matter for QRS morphology.
- Branch point: different devices use different filter settings, which change the ST segment.
- Analytic artifact: `sampling_rate_hz`, `lead_names`, `unit`, `filter_settings`.

**Step 17. Apply quality control.**

- Input: a batch of ECGs.
- Operation: check for clipping, flat lines, extreme amplitudes, and lead reversals.
- Output: a quality report.
- Reasoning: lead reversal produces a recognizable pattern: lead I inverted, aVR and aVL
  swapped, and precordial progression abnormal. A model trained on reversed leads may learn the
  reversal as a feature.
- Branch point: some reversals are subtle and require an algorithm or expert review.
- Analytic artifact: `lead_reversal_flag`, `clipping_flag`, `flatline_flag`.

**Step 18. Evaluate the model by subgroup and acquisition.**

- Input: predictions.
- Operation: stratify by device, lead configuration, sex, age, and rhythm.
- Output: a stratified performance report.
- Reasoning: ECG models often perform differently by sex and by device. Overall AUC can hide
  clinically important differences.
- Branch point: if a subgroup is small, report uncertainty.
- Analytic artifact: a stratified evaluation table.

**Step 19. Plan for urgency.**

- Input: the clinical use case.
- Operation: define the latency requirement and the action.
- Output: a deployment specification.
- Reasoning: a STEMI detection model must fire within minutes and route to the cath lab. A model
  that runs overnight in batch cannot support that decision.
- Branch point: a false positive STEMI alert can cause an unnecessary invasive procedure.
- Analytic artifact: `latency_requirement_s`, `action`, `escalation_path`.

**Step 20. Document the interpretation and its limits.**

- Input: all of the above.
- Operation: write a structured report with the method and the caveats.
- Output: a report that another analyst can reproduce.
- Reasoning: the ECG is a measurement with a method. The method belongs in the record.
- Analytic artifact: a structured ECG report with provenance.

### 5.1 Common rhythm patterns at a glance

| Rhythm | Rate | Regularity | P waves | QRS | Typical data issue |
|---|---:|---|---|---|---|
| Sinus bradycardia | Below 60 | Regular | Present, normal | Narrow | Athletic baseline, beta-blockade |
| Sinus tachycardia | Above 100 | Regular | Present, normal | Narrow | Fever, pain, anemia, sepsis |
| Atrial fibrillation | Variable | Irregularly irregular | Absent | Narrow | Paroxysmal; single strip may miss |
| Atrial flutter | Atrial 250-350 | Often regular block | Flutter waves | Narrow | Variable block mimics AF |
| AVNRT | 150-250 | Regular | Hidden | Narrow | Sudden onset and offset |
| Ventricular tachycardia | 100-250 | Regular | Dissociated | Wide | Can be confused with SVT with aberrancy |
| Ventricular fibrillation | Chaotic | Irregular | Absent | Chaotic | Immediate emergency |
| First-degree AV block | Normal | Regular | Present | Narrow | PR above 200 ms |
| Mobitz I | Variable | Irregular | Present | Narrow | Progressive PR lengthening |
| Mobitz II | Variable | Irregular | Present | Often wide | Fixed PR with dropped beats |
| Third-degree AV block | Variable | Regular | Dissociated | Wide or narrow | Complete AV dissociation |
| Premature atrial contraction | Variable | Irregular | Early, abnormal | Narrow | Common, often benign |
| Premature ventricular contraction | Variable | Irregular | Absent before beat | Wide | Common, frequency matters |

### 5.2 Artifact patterns that mimic disease

| Artifact | Appearance | Mimics | Detection |
|---|---|---|---|
| Baseline wander | Slow undulation of the baseline | ST elevation or depression | High-pass filter; inspect multiple leads |
| Muscle tremor | High-frequency irregular noise | Atrial fibrillation, flutter | Noise across all leads; relaxed patient |
| 50/60 Hz interference | Regular fine oscillation | Fine ventricular fibrillation | Notch filter; check grounding |
| Electrode loosening | Sudden baseline jump | Ectopic beat | Abrupt step in one lead |
| Lead reversal | Negative P and QRS in lead I | Right axis deviation, lateral infarct | Check P-wave axis; repeat ECG |
| Double counting | Two QRS detections per beat | Tachycardia | Compare with raw waveform |
| Motion | Large irregular swings | Ventricular tachycardia | Patient movement history |
| Clipping | Flat-topped QRS | Bundle branch block | Amplitude at system maximum |

These tables are not a substitute for training in ECG interpretation. They are a vocabulary for
recognizing what a model may be learning when the labels and the signal disagree.


## 6. Worked example

A 67-year-old woman presents to the emergency department with palpitations, mild dyspnea, and
lightheadedness that began three hours ago. She has no chest pain. Her history includes
hypertension treated with amlodipine, no diabetes, no prior stroke, no heart failure, and no
coronary disease. She takes no anticoagulant. Her blood pressure is 138/84 mmHg, her respiratory
rate is 20 breaths/min, and her oxygen saturation is 96% on room air.

### 6.1 The ECG

| Feature | Finding |
|---|---|
| Rhythm | Irregularly irregular |
| P waves | Absent |
| QRS duration | 88 ms |
| Rate | 23 QRS complexes in 10 seconds |
| Axis | Lead I positive, aVF positive |
| ST segment | No elevation or depression |
| QT | 340 ms on lead V5, T-wave end clear |
| Average RR | 0.435 s |

### 6.2 Step 1: rate

Counting QRS complexes in a 10-second strip:

```text
HR = 23 x 6 = 138 beats/min
```

The 300/large-box rule is unreliable here because the rhythm is irregular. The average RR is
60 / 138 = 0.435 s.

### 6.3 Step 2: rhythm

The rhythm is irregularly irregular with no visible P waves and a narrow QRS. This is atrial
fibrillation with a rapid ventricular response. The differential includes multifocal atrial
tachycardia (which has visible P waves of at least three morphologies) and atrial flutter with
variable block (which often shows flutter waves in inferior leads). Neither is present.

### 6.4 Step 3: axis

Lead I is positive and aVF is positive, so the axis is normal, between -30 and +90 degrees. A
normal axis does not exclude disease; it excludes major axis deviation.

### 6.5 Step 4: QT and QTc

The measured QT is 340 ms at an RR of 0.435 s. Two corrections:

```text
Bazett:     QTc = QT / sqrt(RR) = 340 / sqrt(0.435) = 340 / 0.6595 = 516 ms
Fridericia: QTc = QT / cbrt(RR) = 340 / 0.7580 = 449 ms
```

The two values differ by 67 ms, and the difference matters. Bazett overcorrects at high heart
rates, and the AHA/ACCF/HRS recommendations discourage its routine use for this reason. At a
rate of 138 beats/min, a Bazett QTc of 516 ms would suggest a high risk of torsades de pointes;
the Fridericia value of 449 ms is borderline. Neither is reliable in atrial fibrillation, where
the RR interval varies from beat to beat and the QT measurement itself is uncertain.

The correct interpretation is: "QT measurement is unreliable in atrial fibrillation at this
rate; repeat after rate control or rhythm conversion." That is a more honest statement than a
single QTc number.

### 6.6 Step 5: rate control and repeat measurement

The patient receives intravenous metoprolol. Her rate falls to 78 beats/min, and a repeat ECG
shows sinus rhythm with a QT of 380 ms.

```text
RR = 60 / 78 = 0.769 s
Bazett:     QTc = 380 / sqrt(0.769) = 380 / 0.877 = 433 ms
Fridericia: QTc = 380 / cbrt(0.769) = 380 / 0.916 = 415 ms
```

The two corrections now differ by only 18 ms. The rate dependence of Bazett is the reason. At
normal rates, the choice of correction matters less; at extreme rates, it matters a great deal.

### 6.7 Step 6: thromboembolic risk

The CHA2DS2-VASc score:

| Component | Points | This patient |
|---|---:|---:|
| Congestive heart failure | 1 | 0 |
| Hypertension | 1 | 0 |
| Age 75 or older | 2 | 0 |
| Diabetes mellitus | 1 | 0 |
| Stroke or transient ischemic attack | 2 | 0 |
| Vascular disease | 1 | 0 |
| Age 65-74 | 1 | 1 |
| Sex category female | 1 | 1 |
| **Total** |  | **2** |

The 2023 ACC/AHA/ACCP/HRS atrial fibrillation guideline recommends anticoagulation when the
annual thromboembolic risk is at least 2% per year, corresponding to a CHA2DS2-VASc of 2 or more
in men and 3 or more in women. It states that anticoagulation is reasonable when the risk is 1%
to less than 2% per year, corresponding to a score of 1 in men and 2 in women. Her score of 2
places her in the intermediate category, where anticoagulation is reasonable after a shared
decision discussion.

Female sex is treated as a risk modifier, not an independent risk factor. That is why the
threshold is one point higher in women. A woman with a score of 2 has one non-sex risk factor
(age 65-74) and is in the intermediate group.

### 6.8 Step 7: a second tracing

The same patient's husband, a 58-year-old man, presents the following week with chest pain. His
ECG shows 2 mm of ST elevation at the J point in leads V2 and V3, with reciprocal ST depression
in leads II, III, and aVF.

The Fourth Universal Definition of Myocardial Infarction (2018) sets ST-elevation thresholds:

| Lead | Threshold |
|---|---|
| V2-V3, men 40 and older | 2.0 mm or more |
| V2-V3, men under 40 | 2.5 mm or more |
| V2-V3, women | 1.5 mm or more |
| All other leads | 1.0 mm or more |

His 2.0 mm elevation in V2-V3 meets the threshold for a man aged 40 or older. The reciprocal
depression in inferior leads supports the diagnosis. This is a STEMI equivalent until proven
otherwise, and it requires immediate activation of the reperfusion pathway.

The contrast between the two cases is the point: the same family, the same clinic, two
completely different urgencies. A model that outputs "abnormal ECG" for both is not useful. The
output must distinguish the arrhythmia that needs rate control from the infarction that needs a
catheterization laboratory.

### 6.9 Step 8: the data trail

| Item | Value | Why it matters |
|---|---|---|
| Device | 12-lead ECG cart, 500 Hz | Sampling rate affects morphology |
| Lead placement | Standard | Reversal changes axis and morphology |
| Calibration | 10 mm/mV, 25 mm/s | Affects all measurements |
| Automated read | "Atrial fibrillation, rapid ventricular response" | Label source |
| Over-read | "AF with RVR, no acute ischemia" | Human label |
| QT | 340 ms | Rate-dependent |
| QTc method | Bazett and Fridericia | Must be recorded |
| Rhythm at repeat | Sinus, 78 beats/min | Context changes |

A dataset that stores only the automated read loses the over-read. A dataset that stores only
the QTc loses the method. A dataset that stores only the rhythm loses the rate. Each omission
makes the record less reproducible.

## 7. Code, query, or calculation

The block computes rate, QTc, axis category, CHA2DS2-VASc, and the STEMI threshold.

```python
"""ECG calculations for rate, QTc, axis, stroke risk, and STEMI thresholds."""

from math import sqrt


def heart_rate_from_rr(rr_s: float) -> float:
    if rr_s <= 0:
        raise ValueError("RR must be positive")
    return 60.0 / rr_s


def qtc_bazett(qt_ms: float, rr_s: float) -> float:
    return qt_ms / sqrt(rr_s)


def qtc_fridericia(qt_ms: float, rr_s: float) -> float:
    return qt_ms / (rr_s ** (1.0 / 3.0))


def axis_category(lead_i: float, lead_avf: float) -> str:
    """Simplified quadrant method. Positive deflections only.

    Normal: I+, aVF+ (0 to +90). Left: I+, aVF- (-90 to 0), but
    physiologic left axis is usually defined as -30 to -90; lead II
    is needed to distinguish. Right: I-, aVF+ (+90 to +180).
    Extreme: I-, aVF- (-90 to -180).
    """
    if lead_i >= 0 and lead_avf >= 0:
        return "normal"
    if lead_i >= 0 and lead_avf < 0:
        return "leftward (check lead II)"
    if lead_i < 0 and lead_avf >= 0:
        return "rightward"
    return "extreme"


def cha2ds2_vasc(chf: int, hypertension: int, age: int, diabetes: int,
                  stroke: int, vascular: int, female: int) -> int:
    score = 0
    score += 1 if chf else 0
    score += 1 if hypertension else 0
    score += 2 if age >= 75 else (1 if age >= 65 else 0)
    score += 1 if diabetes else 0
    score += 2 if stroke else 0
    score += 1 if vascular else 0
    score += 1 if female else 0
    return score


def stemi_threshold_mm(lead: str, age: int, female: bool) -> float:
    if lead.upper() in ("V2", "V3"):
        if female:
            return 1.5
        return 2.0 if age >= 40 else 2.5
    return 1.0


hr = 23 * 6
rr = 60.0 / hr
print(f"HR from 10-second strip: {hr} beats/min")
print(f"RR: {rr:.4f} s")
print(f"QTc Bazett: {qtc_bazett(340, rr):.1f} ms")
print(f"QTc Fridericia: {qtc_fridericia(340, rr):.1f} ms")

rr2 = 60.0 / 78.0
print()
print(f"At 78 beats/min, RR: {rr2:.4f} s")
print(f"QTc Bazett: {qtc_bazett(380, rr2):.1f} ms")
print(f"QTc Fridericia: {qtc_fridericia(380, rr2):.1f} ms")

print()
print(f"axis: {axis_category(1.0, 1.0)}")
score = cha2ds2_vasc(chf=0, hypertension=0, age=67, diabetes=0,
                     stroke=0, vascular=0, female=1)
print(f"CHA2DS2-VASc: {score}")

print()
for lead in ("V2", "V3", "II"):
    print(f"STEMI threshold {lead}, male 58: {stemi_threshold_mm(lead, 58, False)} mm")
print(f"STEMI threshold V2, female 67: {stemi_threshold_mm('V2', 67, True)} mm")

assert hr == 138
assert round(qtc_bazett(340, rr), 1) == 515.6
assert round(qtc_fridericia(340, rr), 1) == 448.8
assert score == 2
```

Expected output:

```text
HR from 10-second strip: 138 beats/min
RR: 0.4348 s
QTc Bazett: 515.6 ms
QTc Fridericia: 448.8 ms

At 78 beats/min, RR: 0.7692 s
QTc Bazett: 433.3 ms
QTc Fridericia: 414.7 ms

axis: normal
CHA2DS2-VASc: 2

STEMI threshold V2, male 58: 2.0 mm
STEMI threshold V3, male 58: 2.0 mm
STEMI threshold II, male 58: 1.0 mm
STEMI threshold V2, female 67: 1.5 mm
```

A SQL query for ECG label provenance:

```sql
-- Count ECGs by label source and rhythm, by device.
SELECT device_id,
       label_source,
       rhythm_label,
       COUNT(*) AS n_ecgs,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY device_id), 1) AS pct
FROM ecg_studies
WHERE recording_ts >= DATE '2025-01-01'
GROUP BY device_id, label_source, rhythm_label
ORDER BY device_id, n_ecgs DESC;
```

If one device contributes most of its labels from automated reads and another from over-reads, a
model trained across both will learn the label source as a feature. The query makes the
imbalance visible before modelling.

## 8. Data, standards, and regulatory context

**ECG standardization.** The AHA/ACCF/HRS recommendations for the standardization and
interpretation of the electrocardiogram were published as a multi-part series in 2007 and 2009.
Part I (2007) covers technology, lead placement, sampling, and computerized interpretation. Part
IV (2009) covers the ST segment, T and U waves, and the QT interval. Part IV states that Bazett's
formula leaves a residual correlation with heart rate and should not be used routinely, and
recommends a linear or power-function correction with age and sex adjustment. It defines
prolonged QTc as 450 ms or more in men and 460 ms or more in women, and short QT as 390 ms or
less in both sexes.

**Myocardial infarction definition.** The Fourth Universal Definition of Myocardial Infarction,
published in 2018 by the European Society of Cardiology, American College of Cardiology, American
Heart Association, and World Heart Federation, defines MI by the combination of myocardial injury
(elevated cardiac troponin) and evidence of ischemia (symptoms, ECG changes, imaging, or
angiography). It sets the ST-elevation thresholds used in section 6.8.

**Atrial fibrillation.** The 2023 ACC/AHA/ACCP/HRS guideline for the diagnosis and management of
atrial fibrillation bases anticoagulation on estimated annual thromboembolic risk rather than the
score alone. It recommends anticoagulation at a risk of 2% per year or more (CHA2DS2-VASc 2 or
more in men, 3 or more in women) and states that anticoagulation is reasonable at 1% to less than
2% per year (score 1 in men, 2 in women). It treats female sex as a risk modifier and advises
against using bleeding risk scores alone to withhold anticoagulation.

**Acute coronary syndrome.** The 2025 ACC/AHA/ACEP/NAEMSP/SCAI guideline for acute coronary
syndromes recommends high-sensitivity troponin as soon as possible, with repeat sampling at 1-2
hours for high-sensitivity assays and 3-6 hours for conventional assays when the initial value is
nondiagnostic. It endorses validated accelerated pathways, including the ESC 0/1h and 0/2h
algorithms, without formally adopting a single algorithm.

**Device standards.** Electrocardiographs are medical devices. IEC 60601-2-25 specifies
particular requirements for basic safety and essential performance of electrocardiographs,
including bandwidth, sampling, and accuracy. In the United States, the FDA regulates ECG devices
and software that interprets ECGs. A model that claims diagnostic output may be a regulated
device; a model that only prioritizes worklists may have a different regulatory pathway.

**Waveform interchange.** The HL7 annotated ECG (aECG) standard and SCP-ECG define formats for
waveform exchange. FHIR can carry waveform data in the SampledData type within an Observation.
DICOM also has waveform objects. None of these formats guarantees that the calibration and lead
names are correct; the metadata must be validated.

**Documentation.** An ECG report should include the indication, the recording conditions, the
interpretation, and the comparison with prior studies. A report without an indication is
difficult to interpret and difficult to audit.

> **Note:** This section is educational and is not medical advice. ECG interpretation and
> anticoagulation decisions require clinical training and current guidelines.

## 9. Industry practice and case evidence

**Sex-specific anticoagulation thresholds are explicit.** The 2023 atrial fibrillation guideline
sets different score thresholds for men and women because female sex is a risk modifier. A model
or a quality metric that uses a single threshold for both sexes will systematically under-treat
women at the same absolute risk. This is documented in the guideline, and it is a concrete
example of why a single cut-off is not neutral.

**Bazett's formula is discouraged for routine reporting.** The AHA/ACCF/HRS Part IV statement
documents the rate dependence of Bazett's correction and recommends alternative methods. Many
ECG machines still report Bazett by default. A dataset that carries only the machine's QTc may
therefore carry a method that the standards body does not recommend. The inference is that the
correction method should be stored and, where possible, recomputed.

**Deep learning can detect structural disease from the ECG.** Attia and colleagues reported in
*Nature Medicine* in 2019 that a convolutional neural network could identify left ventricular
systolic dysfunction from 12-lead ECGs, with performance well above chance. The study is a
single-center development and validation study. It established that the ECG contains information
about structure, not that the model is ready for screening in every population. The inference for
practice is that ECG models can target conditions beyond rhythm, but they require external
validation.

**Public ECG datasets have label limitations.** PTB-XL, published by Wagner and colleagues in
*Scientific Data* in 2020, contains more than 20,000 clinical 12-lead ECGs with standardized
diagnostic statements. The dataset is a major resource, and it is also a reminder that labels are
derived from reports and statements, not from a single adjudicated reference. A model trained on
PTB-XL labels learns the dataset's labeling conventions.

**Device and lead effects are measurable.** ECG morphology depends on lead placement, electrode
impedance, body habitus, and device filters. Studies of ECG algorithms consistently find
performance degradation across devices and sites. The documented fact is the degradation; the
inference is that device-stratified evaluation is required before deployment.

**Automated interpretation is imperfect.** Automated ECG reads are known to have error rates
that vary by rhythm and abnormality. They are useful as a second reader and as a label source at
scale, not as ground truth. A model trained on automated labels inherits both the signal and the
error.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Automated label treated as truth | Label distribution matches one vendor | Machine read used without over-read | Inherited bias | Record label source; validate |
| 10-second strip over-read | Model claims to rule out AF | Paroxysmal arrhythmia not captured | False reassurance | State scope; use monitoring data |
| Bazett used at high rate | QTc implausibly high | Overcorrection | False torsades risk | Report Fridericia or linear correction |
| QT measured in AF | Single-beat QTc varies widely | Variable RR | Unreliable value | Average beats; repeat after rate control |
| Lead reversal missed | Negative P in lead I | Limb lead misplacement | Wrong axis and ischemia | Quality check; repeat ECG |
| Calibration ignored | Intervals off by a constant factor | 50 mm/s or 5 mm/mV | Wrong measurements | Read header; verify pulse |
| Artifact labeled as arrhythmia | High-frequency noise in multiple leads | Muscle tremor, poor contact | False positive | Quality labels; filtering |
| Device shift | Performance drops at a new site | Filter and sampling differences | Transport failure | Device-stratified validation |
| Patient-level leakage | Same patient in train and test | Multiple ECGs per patient | Inflated AUC | Split by patient |
| STEMI threshold not sex-specific | Women under-detected | Threshold applied uniformly | Missed MI | Apply sex- and age-specific thresholds |
| STEMI threshold not age-specific | Men under 40 over-detected | Threshold too low | False cath lab activation | Apply age-specific thresholds |
| AF rate averaged over strip | Rate misclassified | Irregular rhythm | Wrong treatment | Report range and average |
| QRS width measured on one lead | BBB missed | Widest complex in another lead | Wrong classification | Measure across leads |
| LVH voltage criteria over-read | Young thin patients flagged | Body habitus | False hypertrophy | Use age- and sex-specific criteria |
| Alert latency too long | Model runs in batch | Workflow mismatch | No clinical value | Define latency requirement |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Calibration is standard | Measurements depend on it | 50 mm/s, 5 mm/mV | Read header; check pulse |
| Lead placement is correct | Axis and morphology depend on it | Reversal, misplacement | Quality algorithm; repeat |
| The 10-second strip represents the rhythm | Labels depend on it | Paroxysmal arrhythmia | Monitoring data; state scope |
| The label source is consistent | Training depends on it | Mixed automated and over-read | Label-source field |
| QTc correction is appropriate | Risk stratification | Rate extremes, AF | Report method; recompute |
| The patient population matches the training population | Transportability | Different prevalence, device | External validation |
| The ECG is the right test for the question | Clinical validity | Structural disease, intermittent events | Compare with echo or monitor |
| Artifact can be distinguished from signal | Model inputs | Noise that mimics arrhythmia | Quality labels; expert review |
| The model's output matches the workflow | Adoption | Batch vs real-time | Deployment specification |
| The outcome is measurable | Evaluation | No adjudicated reference | Chart review |

## 12. Exercises

**Exercise 1 — Rate and rhythm (application, 40 minutes).**
Given five RR intervals (0.60, 0.75, 1.00, 0.43, 1.20 s), compute the heart rate for each and
classify as bradycardia, normal, or tachycardia. Deliverable: a table. Acceptance criteria: rate
correct to one decimal; classification uses adult thresholds. Time: 40 minutes.

**Exercise 2 — QTc comparison (calculation, 45 minutes).**
For QT 400 ms at RR 0.60 s, compute Bazett and Fridericia. Repeat for QT 400 ms at RR 1.20 s.
Explain which correction is more stable and why. Deliverable: a calculation table and a
300-word explanation. Acceptance criteria: both methods correct; rate dependence explained.
Time: 45 minutes.

**Exercise 3 — Axis and lead reversal (critique, 40 minutes).**
An ECG shows a negative P wave in lead I, a negative QRS in lead I, and a positive QRS in aVR.
Explain the likely cause and its effect on the axis. Deliverable: a 300-word interpretation.
Acceptance criteria: identifies limb lead reversal; explains the axis artifact; proposes a
repeat. Time: 40 minutes.

**Exercise 4 — STEMI thresholds (application, 35 minutes).**
A 38-year-old woman has 1.8 mm ST elevation in V2 and V3. Does she meet the threshold? A
72-year-old man has 1.9 mm in V2 and V3. Does he? Explain. Deliverable: two short answers.
Acceptance criteria: sex and age thresholds applied correctly; the reasoning is explicit. Time:
35 minutes.

**Exercise 5 — AF risk score (calculation, 40 minutes).**
Compute CHA2DS2-VASc for a 76-year-old woman with hypertension, diabetes, and prior stroke.
State the guideline recommendation category. Deliverable: a score table and a recommendation.
Acceptance criteria: points correct; sex modifier explained; recommendation tied to annual risk.
Time: 40 minutes.

**Exercise 6 — Label audit (application, 60 minutes).**
Take 50 ECGs with both automated and over-read labels. Compute agreement by rhythm class and
identify the classes with the lowest agreement. Deliverable: a confusion matrix and a 300-word
analysis. Acceptance criteria: agreement quantified; at least two disagreement mechanisms
proposed. Time: 60 minutes.

**Exercise 7 — Quality pipeline (design, 75 minutes).**
Design a quality-control pipeline for a 12-lead ECG dataset. Include checks for calibration,
lead reversal, clipping, flatline, and noise. Deliverable: a pipeline specification with
thresholds. Acceptance criteria: each check has a rule and an action; the action distinguishes
reject, flag, and accept. Time: 75 minutes.

**Exercise 8 — Deployment plan (design, 60 minutes).**
Design a STEMI detection deployment. Specify latency, alert routing, false-positive handling, and
audit. Deliverable: a one-page plan. Acceptance criteria: latency stated in minutes; escalation
names a role; false positives have a review path; regulatory status addressed. Time: 60 minutes.

## 13. Self-check answers

**Answer to Exercise 1.** HR = 60/RR: 100, 80, 60, 139.5, 50 beats/min. The first is normal
(upper limit), the second normal, the third normal (lower limit), the fourth tachycardia, the
fifth bradycardia. Adult thresholds are below 60 for bradycardia and above 100 for tachycardia.

**Answer to Exercise 2.** At RR 0.60 s: Bazett = 400/0.7746 = 516.4 ms; Fridericia = 400/0.8434
= 474.3 ms. At RR 1.20 s: Bazett = 400/1.0954 = 365.2 ms; Fridericia = 400/1.0627 = 376.4 ms.
Bazett changes by 151 ms across the two rates; Fridericia changes by 98 ms. Fridericia is less
rate-dependent, though neither is perfect.

**Answer to Exercise 3.** A negative P wave and negative QRS in lead I, with a positive QRS in
aVR, suggest reversal of the right and left arm leads. The axis appears rightward or extreme when
it may be normal. The correct action is to repeat the ECG with verified lead placement. Treating
the reversed tracing as a true abnormality can lead to a false diagnosis of right ventricular
strain or lateral infarction.

**Answer to Exercise 4.** The 38-year-old woman: threshold for women in V2-V3 is 1.5 mm; 1.8 mm
meets it. The 72-year-old man: threshold for men 40 and older is 2.0 mm; 1.9 mm does not meet it
by the strict criterion, though it is close and should be interpreted with the clinical picture
and serial ECGs. The thresholds are criteria, not substitutes for judgment.

**Answer to Exercise 5.** Age 76 = 2 points; female = 1; hypertension = 1; diabetes = 1; prior
stroke = 2. Total = 7. This is high risk, and anticoagulation is recommended (Class 1) because
the annual risk is well above 2%. The score also illustrates that a prior stroke alone (2 points)
already places a man at the recommended threshold.

**Answer to self-check on Bazett.** Bazett divides QT by the square root of RR. At high heart
rates, RR is short, the denominator is small, and the corrected QT becomes disproportionately
large. At low rates, the opposite occurs. Fridericia uses the cube root, which reduces the rate
dependence.

**Answer to self-check on AF.** Atrial fibrillation is irregularly irregular with absent P waves.
A regular rhythm with absent P waves is not AF; it may be a junctional rhythm or a regular SVT.

**Rubric for Exercise 7.** Full credit: checks for calibration, reversal, clipping, flatline, and
noise; each has a rule and an action; actions distinguish reject, flag, and accept; thresholds
justified. Half credit: checks present but actions uniform. No credit: no quality checks.

**Rubric for Exercise 8.** Full credit: latency in minutes; named escalation role; false-positive
review; audit plan; regulatory status. Half credit: latency or escalation missing. No credit:
batch deployment for an urgent use case.

## 14. Interview questions

**1. Why is the 10-second ECG a sample rather than a monitor?**
A strong answer explains paroxysmal arrhythmia and the limits of a single recording. A weak
answer says it is short.

**2. How do you compute heart rate from an ECG?**
A strong answer gives 300/large boxes and 1500/small boxes, and the 10-second count for irregular
rhythms. A weak answer gives only one method.

**3. Why is Bazett's formula problematic?**
A strong answer explains rate dependence and the AHA recommendation. A weak answer says it is
"less accurate."

**4. What are the ST-elevation thresholds for STEMI?**
A strong answer gives the V2-V3 age- and sex-specific values and 1 mm for other leads. A weak
answer says 1 mm everywhere.

**5. How does lead reversal affect the ECG?**
A strong answer describes the negative P and QRS in lead I and the axis artifact. A weak answer
says the ECG looks abnormal.

**6. What is the difference between AF and atrial flutter?**
A strong answer describes irregularity, P waves, and flutter waves. A weak answer says both are
fast.

**7. Why is QTc unreliable in atrial fibrillation?**
A strong answer explains variable RR and uncertain T-wave end. A weak answer says it is hard to
measure.

**8. How would you split an ECG dataset?**
A strong answer splits by patient and stratifies by device. A weak answer splits by record.

**9. What does an automated ECG read contribute, and what are its limits?**
A strong answer treats it as a scalable but imperfect label source. A weak answer treats it as
ground truth.

**10. Why do ECG models fail across sites?**
A strong answer names device, filter, lead placement, and population differences. A weak answer
says "distribution shift."

**11. What is the clinical action for a STEMI alert?**
A strong answer describes cath lab activation and the time-critical pathway. A weak answer says
"treat the patient."

**12. When would you not use an ECG-based model?**
A strong answer names intermittent events, structural disease, and situations where the ECG is
not the right test. A weak answer says when accuracy is low.

## 15. Key terms

- **aVR / aVL / aVF** — augmented limb leads.
- **AF** — atrial fibrillation.
- **Artifact** — non-physiological signal.
- **AV block** — atrioventricular conduction block.
- **Axis** — direction of ventricular depolarization.
- **Bazett** — QT correction using the square root of RR.
- **Bundle branch block** — delayed ventricular conduction.
- **Calibration** — scale of the recording.
- **CHA2DS2-VASc** — stroke risk score.
- **ECG** — electrocardiogram.
- **Fridericia** — QT correction using the cube root of RR.
- **J point** — junction between QRS and ST segment.
- **Lead reversal** — misplacement of electrodes.
- **LVH** — left ventricular hypertrophy.
- **NSTEMI** — non-ST-elevation myocardial infarction.
- **P wave** — atrial depolarization.
- **PR interval** — P onset to QRS onset.
- **QRS complex** — ventricular depolarization.
- **QT interval** — QRS onset to T end.
- **QTc** — rate-corrected QT.
- **RR interval** — time between R waves.
- **Sinus rhythm** — rhythm from the SA node.
- **STEMI** — ST-elevation myocardial infarction.
- **SVT** — supraventricular tachycardia.
- **T wave** — ventricular repolarization.
- **Torsades de pointes** — polymorphic ventricular tachycardia associated with QT prolongation.
- **U wave** — late repolarization.
- **VF** — ventricular fibrillation.
- **VT** — ventricular tachycardia.
- **Waveform** — sampled electrical signal.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Standards and guidelines**

- Kligfield P, Gettes LS, Bailey JJ, et al. Recommendations for the standardization and
  interpretation of the electrocardiogram: Part I. *Journal of the American College of
  Cardiology*, 2007;49:1109-1127. https://www.jacc.org/doi/pdf/10.1016/j.jacc.2007.01.024
- Rautaharju PM, Surawicz B, Gettes LS, et al. AHA/ACCF/HRS recommendations for the
  standardization and interpretation of the electrocardiogram: Part IV: the ST segment, T and U
  waves, and the QT interval. *Journal of the American College of Cardiology*, 2009;53:982-991.
  https://www.jacc.org/doi/10.1016/j.jacc.2008.12.014
- Thygesen K, Alpert JS, Jaffe AS, et al. Fourth Universal Definition of Myocardial Infarction
  (2018). *Circulation*, 2018. https://doi.org/10.1161/CIR.0000000000000617
- Joglar JA, Chung MK, Armbruster AL, et al. 2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis
  and Management of Atrial Fibrillation. *Circulation*, 2024;149:e1-e156.
  https://doi.org/10.1161/CIR.0000000000001193
- Rao SV, et al. 2025 ACC/AHA/ACEP/NAEMSP/SCAI Guideline for the Management of Patients With
  Acute Coronary Syndromes. *Journal of the American College of Cardiology*, 2025.
  https://www.jacc.org/doi/pdf/10.1016/j.jacc.2024.11.009
- International Electrotechnical Commission. IEC 60601-2-25: Medical electrical equipment —
  Particular requirements for the basic safety and essential performance of electrocardiographs.

**Evidence**

- Attia ZI, Kapa S, Lopez-Jimenez F, et al. Screening for cardiac contractile dysfunction using
  an artificial intelligence-enabled electrocardiogram. *Nature Medicine*, 2019.
  https://doi.org/10.1038/s41591-018-0240-2
- Wagner P, Strodthoff N, Bousseljot RD, et al. PTB-XL, a large publicly available
  electrocardiography dataset. *Scientific Data*, 2020. https://doi.org/10.1038/s41597-020-0495-6

**Background reading**

- Goldberger AL, et al. *Goldberger's Clinical Electrocardiography: A Simplified Approach*, 9th
  edition.
- Thaler MS. *The Only EKG Book You'll Ever Need*, 9th edition.
- Lilly LS, et al. *Pathophysiology of Heart Disease*, 7th edition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_radiology-and-imaging-basics.md`](06_radiology-and-imaging-basics.md)
- Next topic: [`08_documentation-and-clinical-narratives.md`](08_documentation-and-clinical-narratives.md)
- Related topic: [`02_anatomy-physiology-and-vital-signs.md`](02_anatomy-physiology-and-vital-signs.md)
- Related topic: [`09_special-populations.md`](09_special-populations.md)
- Survey file: [`../../01_clinical_foundations.md`](../../course/01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- ECG quality lab: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)

