# Electrocardiography: Leads, Intervals, Rhythm, and Automated Interpretation

> Module 11 — Physiological Signals and Wearables
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals)
> Version 2026.3

## 1. Why this topic exists

Electrocardiography (ECG) is the most widely deployed physiological signal in medicine and the most widely misinterpreted by software teams. A 12-lead ECG is a set of twelve simultaneous projections of cardiac electrical activity, sampled for 10 s, interpreted through a sequence of fiducial measurements and pattern rules. A single-lead wearable ECG is one projection, often recorded for 30 s, interpreted by a much narrower set of rules. The two are not interchangeable, and the difference is not only the number of leads.

The industry problem is that ECG-derived measurements look simple and are not. A heart rate is a number. A PR interval is a number. A QT interval is a number. Each depends on electrode placement, on the filter chain, on the fiducial definition, and on the population. A team that treats these numbers as labels rather than measurements will ship an algorithm that performs well on its development set and fails on the first device with a different lead placement or a different bandwidth.

Four failures recur.

**Lead misplacement and reversal.** Limb electrodes are swapped, or a precordial lead is placed one intercostal space too high. The ECG remains electrically valid and morphologically wrong. Automated interpretation may report an inferior infarct that is an artifact of reversed leads, or may miss one that is real. The signal chain cannot detect this from morphology alone unless the algorithm checks P-wave polarity and lead consistency.

**Interval measurement without a definition.** The QT interval ends where the T wave ends, and the T wave ends where the reader says it ends. Two cardiologists measuring the same beat can differ by 20 ms to 40 ms. An algorithm that reports QT to the nearest millisecond without stating its T-end method is reporting false precision. The AHA/ACCF/HRS recommendations address this directly: use the lead with the longest QT, use the tangent method when T and U waves fuse, and treat computer-reported prolongation as a prompt for visual review rather than a diagnosis.

**Rate correction that changes the answer.** Bazett's formula overcorrects at high heart rates. At a heart rate of 80 beats per minute (bpm), a 400 ms QT gives a Bazett-corrected QT (QTc) of 462 ms and a Fridericia-corrected QT of 440 ms. One is prolonged by the common 450 ms threshold and the other is not. The choice of formula is a measurement decision that changes the clinical conclusion, and it is often left unstated.

**Single-lead overreach.** A wrist device records a lead that approximates lead I. It can detect irregular rhythm and can classify atrial fibrillation (AF) in a narrow population. It cannot diagnose an ST-elevation myocardial infarction, cannot assess bundle branch block reliably, and cannot measure QT with diagnostic accuracy. Products that blur this boundary create regulatory and clinical risk.

This topic teaches the measurement chain for ECG: how the twelve leads are formed, how each interval is defined, how quality is assessed, how rhythm is classified, and where automated interpretation stops being trustworthy. It builds directly on the sampling and filtering vocabulary of Topic 01 and sets up the feature engineering in Topic 06 and the validation logic in Topic 09.

### 1.1 Three composite failures and their costs

**The false infarction.** A patient in an urgent care clinic has a 12-lead ECG. The technologist has swapped the right and left arm electrodes. Lead I is inverted, aVR is upright, and the automated algorithm reads an inferior and lateral injury pattern. The patient is transferred to a hospital for a catheterisation that shows normal coronary arteries. The cost is a unnecessary invasive procedure, a hospital admission, and a false entry in the record. The signal chain was technically valid; the placement was wrong. A P-wave polarity check would have flagged the reversal before the interpretation.

This composite is assembled from a common pattern rather than a single published case. Lead reversal is a well-documented source of false ECG interpretation, and the detection method is standard.

**The false long QT.** A drug-safety study measures QT from a 10 s ECG at 500 Hz. The analysis measures from the R peak to the visual return of the T wave, includes a U wave, and corrects with Bazett at a heart rate of 95 bpm. The reported QTc is 480 ms, above the 450 ms threshold, and the participant is flagged for protocol review. Re-measurement from the QRS onset with the tangent method and Fridericia correction gives 440 ms. The participant is normal. The cost is a protocol deviation and a delay in a trial, caused by three measurement choices that were never written down.

**The false AF.** A consumer wearable reports an irregular rhythm during a walk. The PPG signal has motion artifact, the peak detector fires on motion spikes, and the RR intervals look irregular. The user is notified and undergoes a 14-day patch, which shows no AF. The cost is anxiety and a diagnostic workup. The signal-chain cause is that motion artifact and atrial fibrillation both produce irregular intervals, and the product did not use a quality gate strong enough to separate them.

The common thread is that each failure is a measurement or decision failure, not a modelling failure. The model did what it was asked. The chain did not support the question.

A useful way to audit an ECG product is to ask which of the three failures it is structurally exposed to. A product that records a single lead is exposed to the spatial failure. A product that reports QTc is exposed to the correction failure. A product that screens for AF with PPG is exposed to the artifact failure. The exposure is a property of the design, not of the validation dataset, and it determines which checks the validation must include.

The costs are not symmetric. The false infarction costs an invasive procedure. The false long QT costs a trial delay. The false AF costs anxiety and a diagnostic workup. All three are avoidable with checks that cost minutes: a P-wave polarity check, a QRS-onset definition, a quality gate. The asymmetry between minutes of prevention and days of consequence is the economic argument for the procedures in Section 5.

### 1.2 Why ECG is the right first deep modality

ECG is a good teaching modality because it has all the properties that make physiological data hard, in a form that is well documented.

- It is **spatial**: twelve leads measure different projections, so placement matters.
- It is **fiducial**: intervals are defined by waveform landmarks that must be detected.
- It is **rate-dependent**: the QT correction depends on heart rate.
- It is **population-dependent**: normal ranges vary by age and sex.
- It is **regulated**: there are performance standards, guidance documents, and cleared products to study.
- It is **publicly benchmarked**: PTB-XL and the PhysioNet challenges provide reproducible data.

If you can specify and audit an ECG chain, you can transfer the same habits to EEG, PPG, and respiration, which have the same structure with different physics.

### 1.3 What the rest of the module assumes

This topic assumes the sampling and filtering vocabulary of Topic 01. It does not repeat the aliasing arithmetic or the filter-family table. It assumes that a signal has a known sampling rate, a known amplitude scale, and a known quality state. If those assumptions fail, the ECG measurements in this topic are not interpretable, regardless of how good the lead detection is.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** how the twelve standard leads are derived from four limb electrodes and six precordial electrodes, and what each lead observes.
2. **Identify** limb lead reversal and precordial misplacement from P-wave polarity, lead consistency, and progression.
3. **Measure** PR, QRS, and QT intervals from a digitised ECG with stated fiducial definitions and stated uncertainty.
4. **Compute** heart rate, corrected QT by Bazett and Fridericia, and frontal axis, and explain why the three can disagree.
5. **Classify** a rhythm into sinus, atrial fibrillation, ectopic, or uninterpretable using rate, regularity, P-wave presence, and QRS width.
6. **Assess** technical quality using baseline, noise, clipping, lead-off, and QRS detectability, and decide whether a record is interpretable.
7. **Critique** an automated ECG interpretation claim for lead set, population, reference standard, and handling of unclassifiable records.
8. **Design** a validation dataset for a single-lead AF detector, including patient-level splitting and a prespecified inconclusive category.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are ordered from the spatial (leads) to the temporal (intervals) to the categorical (rhythm) to the professional (validation and critique). The order is deliberate: a rhythm classification built on a mislabelled lead or a mis-measured interval is a classification of an artifact. The spatial and temporal layers must be right before the categorical layer means anything.

Two outcomes are often skipped in industry training. Outcome 6, quality assessment, is the one that determines whether a record enters the dataset at all. Outcome 7, critiquing an interpretation claim, is the one that determines whether a product's performance statement survives regulatory review. Both are assessed by written artefacts rather than by calculations.

## 3. The core mental model

An ECG is a spatial measurement rendered as a time series. The heart's electrical activity is a moving dipole; each lead measures the projection of that dipole onto a fixed axis. The waveform is the projection over time.

```text
                    cardiac electrical field
                              |
             +----------------+----------------+
             |                                 |
        limb electrodes                   precordial electrodes
        RA, LA, LL, RL                    V1 ... V6
             |                                 |
      Einthoven / Goldberger              Wilson central terminal
             |                                 |
        I, II, III, aVR, aVL, aVF              V1 ... V6
             |                                 |
             +----------------+----------------+
                              |
                       12 simultaneous traces
                              |
                    quality control and fiducials
                              |
             P onset, QRS onset/offset, T end per lead
                              |
             intervals: PR, QRS, QT, RR; axis; morphology
                              |
                    rhythm and pattern classification
                              |
                     report with confidence and limits
```

Redraw this with the lead equations on the arrows. The spatial part is what distinguishes ECG from a single-channel signal such as PPG.

### 3.1 The lead equations

The limb leads are formed from three electrodes: right arm (RA), left arm (LA), and left leg (LL). The right leg (RL) is a reference and is not part of the signal.

- Lead I = LA - RA.
- Lead II = LL - RA.
- Lead III = LL - LA.
- aVR = RA - (LA + LL)/2, scaled.
- aVL = LA - (RA + LL)/2, scaled.
- aVF = LL - (RA + LA)/2, scaled.

The augmented leads are scaled by 1.5 relative to the bipolar leads so that their amplitudes are comparable. The six limb leads are not independent: I + III = II, and the augmented leads are linear combinations. This redundancy is what makes lead reversal detectable: a swap changes the pattern in a way that violates the equations.

The precordial leads V1 to V6 are unipolar electrodes referenced to the Wilson central terminal, the average of RA, LA, and LL. They observe the horizontal plane, from the right sternal edge (V1) to the mid-axillary line (V6). Their placement is anatomical, and small displacements change morphology, especially in V1 and V2.

### 3.2 The waveform as a sequence of fiducials

Every interval is defined by two fiducial points.

```text
        P wave        PR segment      QRS complex        ST segment     T wave
          __                              __
         /  \                            /  \          ______
   _____/    \______/\______________/\__/    \________/      \______
        |    |      |  |            |        |        |      |
      P on  P off  QRS on         QRS off   J point   T peak  T end
        |<-- PR -->|
                    |<---------- QRS --------->|
                    |<---------------- QT ---------------->|
```

- **PR interval**: P onset to QRS onset. Normal 120 ms to 200 ms.
- **QRS duration**: QRS onset to QRS offset (J point). Normal 80 ms to 100 ms, up to 120 ms in some definitions.
- **QT interval**: QRS onset to T end. Normal depends on rate and sex; corrected values above 450 ms in men and 460 ms in women are commonly called prolonged.
- **RR interval**: R peak to R peak. Heart rate = 60/RR when RR is in seconds.

The fiducials are not equally easy to detect. QRS onset and offset are high-slope events and are relatively robust. P onset is low-amplitude and often buried in noise. T end is the least reliable: the T wave has a gradual return to baseline, and a U wave may follow it.

### 3.3 The three levels of interpretation

Automated ECG interpretation operates at three levels, and they have different error profiles.

1. **Measurement**: rate, intervals, axis, amplitudes. These are numeric and can be validated against adjudicated measurements.
2. **Morphology description**: ST elevation, T inversion, Q waves, bundle branch block. These are categorical descriptions of waveform shape.
3. **Diagnosis**: myocardial infarction, ischemia, hypertrophy, arrhythmia. These are clinical inferences that combine measurements, morphology, and context.

A model that measures intervals accurately can still produce a wrong diagnosis. A model that describes morphology accurately can still produce a wrong diagnosis if the population prevalence is different. The three levels must be validated separately, and a claim at one level does not support a claim at another.

The levels also have different failure economics. A measurement error is usually detectable with a tolerance test and a reference device. A morphology error requires adjudicated examples and a statement lexicon. A diagnostic error requires clinical outcomes and a study design. A team that validates only the first level and claims the third has skipped the two hardest steps.

### 3.4 Why the reference standard matters

The reference standard for ECG interpretation is not the ECG itself. It is a cardiologist's over-read, often with imaging, laboratory, or clinical follow-up. The over-read has its own variability. Inter-reader agreement for ST elevation and for bundle branch block is moderate, not perfect. A model evaluated against a single over-read inherits that reader's thresholds and errors.

A useful discipline is to ask what the reference would have been if the test had not existed. For a screening ECG, the reference is the over-read of the same recording, which is circular in a subtle way: the over-read sees the same noisy signal the algorithm sees. For a diagnostic claim, the reference should be independent, such as imaging or follow-up, because the question is whether the ECG finding predicts the disease, not whether two readers agree on the waveform.

For rhythm, the reference may be a 12-lead ECG recorded simultaneously, a Holter monitor, or an adjudication committee. For AF, a common design is a simultaneous 12-lead ECG over-read by blinded cardiologists. The Apple ECG validation used three blinded cardiologists and a simultaneous 12-lead recording. The design is strong because the reference and the test observe the same beats at the same time. It is limited because the recording conditions are controlled and the participants are selected.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Lead | A projection of cardiac electrical activity onto an axis | Calling an electrode a lead | named I, II, III, aVR, aVL, aVF, V1-V6 |
| Electrode | A conductive contact on the skin | Confusing it with the lead it contributes to | anatomical site |
| Limb lead | Lead derived from RA, LA, LL | Assuming all limb leads are independent | I, II, III |
| Augmented lead | Scaled combination of two limb electrodes | Forgetting the 1.5 scale factor | aVR, aVL, aVF |
| Precordial lead | Unipolar chest lead referenced to Wilson terminal | Assuming placement is interchangeable | V1-V6 |
| Wilson central terminal | Average of RA, LA, LL | Treating it as ground | voltage reference |
| Frontal axis | Direction of the mean QRS vector in the frontal plane | Confusing it with the anatomical axis | degrees |
| P wave | Atrial depolarisation | Calling any atrial activity a P wave | mV, ms |
| PR interval | P onset to QRS onset | Measuring from P peak | ms |
| PR segment | P offset to QRS onset | Confusing it with the PR interval | ms |
| QRS complex | Ventricular depolarisation | Measuring only the R wave | ms |
| QRS duration | QRS onset to J point | Using the R-to-S peak distance | ms |
| J point | Junction of QRS offset and ST segment | Missing it in a slurred S wave | ms |
| ST segment | Interval between J point and T onset | Measuring ST deviation at the wrong point | mV |
| T wave | Ventricular repolarisation | Confusing it with a U wave | mV |
| U wave | Late repolarisation, often low amplitude | Counting it as part of the T wave | mV |
| QT interval | QRS onset to T end | Using a fixed offset from R | ms |
| QTc | QT corrected for heart rate | Reporting without naming the formula | ms |
| Bazett correction | QTc = QT/√RR | Using it at high heart rates without caveat | dimensionless |
| Fridericia correction | QTc = QT/RR^(1/3) | Treating it as equivalent to Bazett | dimensionless |
| RR interval | Interval between consecutive R peaks | Averaging without excluding ectopy | s or ms |
| Heart rate | Beats per minute | Reporting instantaneous rate from one interval | bpm |
| Sinus rhythm | Rhythm originating in the sinoatrial node | Calling any regular rhythm sinus | categorical |
| Atrial fibrillation | Irregularly irregular rhythm without P waves | Diagnosing from RR irregularity alone | ICD-10 I48 |
| Premature atrial contraction | Early beat from an atrial focus | Confusing it with AF | categorical |
| Premature ventricular contraction | Early wide beat from a ventricular focus | Confusing it with bundle branch block | categorical |
| Bundle branch block | Conduction delay producing a wide QRS | Diagnosing from width alone | categorical |
| Atrioventricular block | Impaired conduction from atria to ventricles | Confusing first-degree with second-degree | categorical |
| ST elevation | ST segment above baseline | Calling early repolarisation an infarct | mV |
| Low voltage | Small QRS amplitude in limb leads | Missing it because of gain settings | mV |
| Electrical alternans | Beat-to-beat amplitude alternation | Confusing it with respiration | mV |
| Calibration | 1 mV = 10 mm at standard gain | Assuming the display scale is the signal scale | mm/mV |
| Paper speed | 25 mm/s standard | Measuring intervals from a differently scaled display | mm/s |
| Over-read | Cardiologist interpretation used as reference | Treating it as error-free | categorical |
| Unclassifiable | Record that cannot support a diagnosis | Counting it as a negative | categorical |
| Einthoven triangle | I + III = II relationship | Treating limb leads as independent | mV |
| R-wave progression | Change in R/S ratio from V1 to V6 | Reading V1 alone | categorical |
| Q wave | Initial negative deflection of the QRS | Calling any negative deflection a Q wave | mV, ms |
| ST depression | ST segment below baseline | Measuring at the wrong point | mV |
| T-wave inversion | Negative T wave in a lead where it should be upright | Calling a normal variant ischemia | mV |
| Sinus arrhythmia | Respiratory variation in sinus rate | Calling it AF | bpm, ms |
| Atrial flutter | Organised atrial rhythm, often 300 bpm atrial rate | Confusing flutter waves with P waves | bpm |
| Ectopic beat | Beat from a focus outside the sinus node | Averaging it with sinus beats | categorical |
| Aberrancy | Wide QRS due to conduction rather than ventricular origin | Calling it ventricular tachycardia | categorical |
| Pre-excitation | Early ventricular activation via an accessory pathway | Missing a short PR | ms |
| Paced rhythm | Rhythm driven by a pacemaker | Missing the spike | categorical |
| Diagnostic bandwidth | 0.05-150 Hz for adult diagnostic ECG | Using a monitor bandwidth for diagnosis | Hz |
| Monitor bandwidth | Typically 0.5-40 Hz | Using it for ST measurement | Hz |
| Standard gain | 10 mm/mV | Assuming display scale equals signal scale | mm/mV |
| Standard paper speed | 25 mm/s | Measuring intervals from a scaled image | mm/s |
| Computer interpretation | Automated statement set | Treating it as an over-read | categorical |
| Agreement | Concordance between algorithm and over-read | Reporting accuracy without kappa | dimensionless |
| Kappa | Agreement beyond chance | Reporting raw agreement alone | dimensionless |
| Signal quality index | Composite measure of usability | Using it as a diagnosis | dimensionless |
| Heart rate variability | Variation in RR intervals | Computing it on a 10 s strip | ms |
| QT dispersion | Inter-lead variation in QT | Reporting it routinely | ms |

## 5. The mechanics, step by step

The procedure assumes a digital 12-lead ECG, 10 s, 500 samples/s, with leads I, II, III, aVR, aVL, aVF, V1 to V6. It scales down to a single lead by skipping the spatial steps and reducing the confidence of the conclusions.

### Step 1. Confirm the indication and the lead set

**Input:** the clinical or product question, the available lead set, and the recording duration.

**Output:** a statement of what the record can and cannot support.

**Rationale:** the lead set bounds the diagnosis. A 12-lead ECG can assess axis, bundle branch block, and regional ST changes. A single lead can assess rate and rhythm and, in some leads, ST trend. A two-lead patch can improve rhythm discrimination but cannot localise ischemia. The indication determines which measurements are required, and the lead set determines which are possible.

**Branch point:** if the question is "is this AF," a single lead with a good P-wave view is sufficient for a rhythm classification in a narrow claim. If the question is "is there an ST-elevation myocardial infarction," a 12-lead is required and the product must not claim otherwise.

**Edge case:** a 12-lead record with one failed precordial lead is not a 12-lead record for the purpose of regional diagnosis. The missing lead must be reported, and the interpretation must be limited.

**Duration and yield.** The recording duration is part of the indication. A 10 s clinic ECG captures about 11 beats at 68 bpm; a 24 h Holter captures about 98,000 beats; a 14-day patch captures about 1.4 million. The probability of capturing a paroxysmal event scales with the number of beats observed, not with the quality of the algorithm. A perfect AF detector applied to a 10 s strip will miss most paroxysmal AF, because the AF was not present during those 10 s.

**Lead sets and what they observe.** The twelve leads are not twelve independent measurements. They are twelve projections, and each region of the heart is best seen by a subset.

| Lead group | Region | Main use | Limitation |
|---|---|---|---|
| I, aVL | Lateral wall | Lateral ischemia, axis | Poor at inferior changes |
| II, III, aVF | Inferior wall | Inferior ischemia, rhythm | Affected by respiration |
| aVR | Right upper cavity | Lead reversal check, left main ischemia | Often ignored |
| V1, V2 | Septum and right ventricle | Bundle branch block, RV changes | Placement-sensitive |
| V3, V4 | Anterior wall | Anterior ischemia | Transition zone varies |
| V5, V6 | Lateral wall | Lateral ischemia, hypertrophy | Body habitus affects amplitude |

A single-lead device that records a lead I approximation observes the lateral wall and the left-to-right axis. It does not observe the inferior wall, the septum, or the right ventricle. This is the anatomical reason a single lead cannot rule out an inferior infarction.

### Step 2. Verify electrode placement and lead labels

**Input:** the lead labels, the waveform, and the patient's body habitus.

**Output:** a placement confidence flag and, when possible, a correction.

**Rationale:** a mislabelled lead is a measurement error that no algorithm can fix without recognising it. Limb lead reversal is common in busy clinics and in self-applied wearable recordings. The most frequent swap is right arm and left arm, which inverts lead I and swaps aVR and aVL. A second common swap is left arm and left leg, which swaps leads I and II and changes III.

**Detection arithmetic:** in normal sinus rhythm, the P wave is upright in lead I and lead II and inverted in aVR. If lead I has an inverted P wave and aVR has an upright P wave, RA/LA reversal is likely. If lead I is upright but lead II is inverted, LA/LL reversal is likely. The QRS axis also shifts: RA/LA reversal moves the axis rightward, often producing an apparent right axis deviation.

**Branch point:** if the reversal is detected before interpretation, correct the lead labels and re-run the measurements. If it is detected after, report the record as uninterpretable for axis and for regional diagnosis, and request a repeat.

**Edge case:** dextrocardia produces a similar pattern in lead I but is distinguished by precordial lead progression and by the clinical context. An algorithm that flags every negative P in lead I as a reversal will overcall dextrocardia.

**Reversal patterns.** The three common swaps produce distinguishable patterns.

| Swap | Lead I | Lead II | Lead III | aVR | aVL | aVF | P in I |
|---|---|---|---|---|---|---|---|
| Normal | LA-RA | LL-RA | LL-LA | RA-mean | LA-mean | LL-mean | upright |
| RA/LA | inverted | III | II | aVL | aVR | aVF | inverted |
| LA/LL | II | I | inverted | aVR | aVF | aVL | upright |
| RA/LL | inverted III | inverted II | inverted I | aVF | aVL | aVR | inverted |

The table is a guide, not a diagnosis. The confirmatory step is to compare the observed pattern with the expected pattern from the lead equations, and to repeat the recording with verified placement.

### Step 3. Check calibration and signal scale

**Input:** the digital values, the header, and any calibration pulse.

**Output:** a conversion from codes to millivolts and a confirmation that the standard gain is known.

**Rationale:** a 1 mV signal should produce a 10 mm deflection at standard gain (10 mm/mV). If the display or the export uses a different gain, the amplitudes are not comparable with thresholds. A digital record may store values in microvolts, millivolts, or ADC codes, and the header may or may not say which.

**Arithmetic:** PTB-XL stores 16-bit values at 1 µV per least significant bit (LSB). A 1 mV QRS is 1,000 codes. A record stored at 0.1526 µV per LSB, as in the Topic 01 example, would represent the same QRS as 6,554 codes. The two records have the same physiological amplitude and different digital scales.

**Branch point:** if the calibration is unknown, the record can still support rhythm and interval measurement, because those are timing-based. It cannot support amplitude-based criteria such as low voltage or ST elevation thresholds.

**Edge case:** automatic gain control may change the scale mid-record. The calibration pulse at the start does not describe the rest of the record. Check the amplitude histogram for step changes.

**Amplitude criteria that depend on calibration.** Several common criteria are amplitude-based and become meaningless if the scale is wrong.

- Low voltage in limb leads: QRS amplitude below 0.5 mV in all limb leads.
- ST elevation in V2/V3: 0.2 mV in men aged 40 or older, 0.25 mV in men younger than 40, 0.15 mV in women, with 0.1 mV in other leads.
- Left ventricular hypertrophy by Sokolow-Lyon: S in V1 plus R in V5 or V6 greater than 3.5 mV.

Each threshold is a millivolt value. If the record is stored in codes without a scale, the criterion cannot be applied. If the record is stored at 1 µV per LSB, a 0.1 mV threshold is 100 codes; at 0.1526 µV per LSB it is 655 codes. The physiological threshold is the same; the digital threshold is not.

### Step 4. Assess technical quality

**Input:** the 12 leads and the device status flags.

**Output:** a per-lead and global quality score, and a decision to interpret, flag, or reject.

**Rationale:** ECG quality is not a single property. Baseline wander, muscle noise, mains interference, clipping, lead-off, and poor R-wave detectability can coexist. A record can be adequate for rhythm and inadequate for ST analysis. Quality must be assessed per measurement, not only per record.

**Criteria:** the commonly used components are baseline wander (low-frequency power), high-frequency noise (muscle), mains interference (spectral peak), clipping (flat tops), lead-off (rail or status), and QRS detectability (signal-to-noise ratio of the detection band). A composite index can be built, but each component should remain visible.

**Branch point:** if the record is uninterpretable for the intended measurement, reject it and report why. If it is interpretable for rhythm but not for ST analysis, restrict the report.

**Edge case:** a pacemaker produces narrow spikes that a low-pass filter removes and that a QRS detector may misinterpret. A record from a paced patient needs a spike detection channel and a different quality rule.

**A composite quality index.** A simple index combines four normalised components:

```text
Q = 1 - mean(baseline_score, noise_score, mains_score, clip_score)
```

where each component is 0 for a clean signal and 1 for an unusable one. For example, define baseline_score as the ratio of power below 0.5 Hz to total power, clipped to [0,1]; noise_score as the ratio of power above 40 Hz to total power, clipped to [0,1]; mains_score as the ratio of power within ±1 Hz of the mains frequency to total power, multiplied by 3; clip_score as the fraction of samples at the rail, multiplied by 100. The weights are arbitrary and must be validated. The value of the index is that it forces each component to be named and computed, not that it produces a universal threshold.

### Step 5. Establish the time base and select beats for averaging

**Input:** the sample indices, the R peaks, and the recording start time.

**Output:** a beat list with fiducial times and a representative beat for each lead.

**Rationale:** interval measurement is more stable on an averaged beat than on a single beat, because noise averages down and the fiducials become clearer. The averaged beat must be built from beats of the same morphology; including ectopic beats in the average smears the QRS and biases the intervals.

**Arithmetic:** at 500 samples/s, 10 s contains 5,000 samples. A median beat built from 11 sinus beats reduces random noise by √11 ≈ 3.3, if the noise is independent across beats. It does not reduce baseline wander, which is correlated across beats.

**Branch point:** if the rhythm is irregular, a median beat may mix morphologies. Measure intervals on individual beats and report the median and range.

**Edge case:** a record with only two or three beats cannot support a stable median beat. Measure on the best single beat and widen the uncertainty.

**Beat selection rules.** A median beat is only as good as the beats it averages. A defensible selection rule is:

1. Exclude beats within 500 ms of the record boundary.
2. Exclude beats whose RR interval differs from the median RR by more than 20%.
3. Exclude beats whose QRS correlation with the median template is below 0.9.
4. Recompute the template and repeat once.
5. Report the number of beats included and the number excluded.

The rule is deterministic and auditable. It also makes the exclusion rate a quality metric: a record that loses half its beats to the correlation filter is a record with a rhythm problem or a noise problem, and the report should say which.

### Step 6. Identify P waves and the atrial rhythm

**Input:** the averaged or individual beats, with a lead that shows P waves well (usually II or V1).

**Output:** P-wave presence, polarity, rate, and relationship to QRS.

**Rationale:** the P wave distinguishes sinus rhythm from AF, from atrial flutter, and from junctional rhythm. It is also the fiducial for the PR interval. P-wave detection is harder than QRS detection because the amplitude is lower and the wave is broader.

**Method:** band-pass the signal at roughly 0.5 Hz to 20 Hz, search a window from 300 ms to 60 ms before each R peak, and take the maximum absolute deflection. Confirm that the deflection is consistent across beats and that its polarity matches the expected lead.

**Branch point:** if P waves are absent and the RR intervals are irregular, AF is likely. If P waves are absent and the RR intervals are regular, a junctional rhythm is possible. If P waves are present but the PR interval varies, consider AV block.

**Edge case:** a P wave can be buried in the preceding T wave at high rates. A single lead may not show it. Use multiple leads and a beat-averaged template.

**P-wave arithmetic.** A normal P wave is about 0.1 mV to 0.25 mV in amplitude and 80 ms to 110 ms in duration. At 500 samples/s, 80 ms is 40 samples. A P wave sampled at 100 Hz is 8 samples, which is enough to detect a peak but not enough to define an onset reliably. This is one reason a 500 Hz diagnostic ECG and a 100 Hz convenience export are not equivalent for PR measurement.

The P wave is best seen in lead II when the sinus node is in its normal position, and in V1 when the left atrium is enlarged. A P-wave detection algorithm that only looks at lead I will miss P waves that are isoelectric in that lead.

### Step 7. Measure the PR interval

**Input:** the P onset and the QRS onset from the same beat.

**Output:** PR interval in milliseconds, with a stated fiducial definition.

**Rationale:** the PR interval reflects conduction from the atrium to the ventricle. It is measured from the beginning of the P wave to the beginning of the QRS, not from the P peak. Measuring from the peak shortens the interval by roughly half the P duration, which can be 40 ms to 60 ms.

**Arithmetic:** if the P onset is at 3.916 s and the QRS onset is at 4.092 s, the PR interval is 176 ms. If the P peak is at 3.970 s, a peak-based measurement gives 122 ms, a 54 ms error. The error is large enough to move a borderline first-degree block into the normal range.

**Branch point:** if the PR interval is above 200 ms, first-degree AV block is present. If it is below 120 ms, consider pre-excitation or a junctional rhythm. Both conclusions require a consistent P-wave definition.

**Edge case:** if the P wave is not visible, the PR interval cannot be measured. Report it as missing, not as normal.

**First-degree block arithmetic.** First-degree AV block is a PR interval above 200 ms. At 500 samples/s, 200 ms is 100 samples. A measurement of 176 ms is 12 samples below the threshold, which is a comfortable margin. A measurement of 196 ms is 2 samples below, which is within the fiducial uncertainty of a P-onset detector. In the borderline zone, report the uncertainty and the method, not a categorical "normal."

The PR interval also varies with heart rate, with age, and with autonomic tone. A single measurement is a point estimate, not a diagnosis. Trend over time is often more informative than the absolute value.

### Step 8. Measure QRS duration and describe morphology

**Input:** the QRS onset and offset, and the morphology in all leads.

**Output:** QRS duration in milliseconds and a morphological description.

**Rationale:** QRS duration separates narrow-complex from wide-complex rhythms and is a criterion for bundle branch block. The offset is the J point, which can be slurred and hard to identify. Measuring from the R peak to the S nadir underestimates the duration.

**Arithmetic:** in the worked example, the QRS onset is at 4.092 s and the offset at 4.178 s, a duration of 86 ms. The R peak is at 4.156 s. An R-to-S-peak measurement would miss the initial and terminal low-slope components.

**Branch point:** a duration of 120 ms or more suggests a bundle branch block or a ventricular origin. A duration between 100 ms and 120 ms is indeterminate and requires morphology. A duration below 100 ms is narrow.

**Edge case:** a paced beat is wide by design. A pre-excited beat is wide because of an accessory pathway. Neither is a bundle branch block, and the morphology distinguishes them.

**Bundle branch block criteria in brief.** The width alone is not enough.

| Pattern | QRS width | V1 | V6 | Axis |
|---|---:|---|---|---|
| Normal | < 120 ms | rS | qRs | normal |
| Right bundle branch block | ≥ 120 ms | rsR' | qRS with slurred S | often normal |
| Left bundle branch block | ≥ 120 ms | QS or rS | broad R, no q | often left |
| Nonspecific intraventricular delay | ≥ 110 ms | variable | variable | variable |

The morphology in V1 and V6 is the discriminating feature. A model that uses duration alone will confuse bundle branch block with ventricular rhythm and with pacing.

### Step 9. Determine the frontal axis

**Input:** the net QRS amplitude in leads I and aVF.

**Output:** the frontal axis in degrees.

**Rationale:** the axis is the direction of the mean QRS vector. It is computed from the net area or net amplitude in two orthogonal limb leads. A normal axis is usually defined as -30° to +90°. Left axis deviation is more negative than -30°, right axis deviation is more positive than +90°.

**Arithmetic:** if lead I has a net QRS of +0.60 mV and aVF has +0.80 mV, the axis is atan2(0.80, 0.60) = 53.1°, which is normal. If lead I is -0.60 mV and aVF is +0.80 mV, the axis is 126.9°, which is right axis deviation.

**Branch point:** if lead I is isoelectric, the axis is approximately +90° or -90°, and a second lead is needed to resolve the quadrant. If both I and aVF are isoelectric, the axis is indeterminate.

**Edge case:** lead reversal changes the axis. An axis that does not match the P-wave polarity or the precordial progression should prompt a placement check.

**Axis quadrants.** The quadrant is determined by the signs of leads I and aVF.

| Lead I | aVF | Axis range | Label |
|---|---|---|---|
| + | + | 0° to +90° | normal |
| + | - | 0° to -90° | leftward; normal to -30°, left deviation beyond |
| - | + | +90° to +180° | right deviation |
| - | - | -90° to -180° | extreme axis ("northwest") |

An extreme axis is rare in a normal heart and should prompt a check for lead reversal, ventricular rhythm, or a misplaced electrode. A leftward axis between 0° and -30° is often normal, especially in older adults.

### Step 10. Measure QT and correct for rate

**Input:** the QRS onset and the T end in the lead with the longest QT.

**Output:** QT in milliseconds and QTc by a named formula.

**Rationale:** the QT interval reflects the duration of ventricular repolarisation. It is rate-dependent, so it is corrected to a standard rate. Bazett's formula is the most widely used and the most criticised: it overcorrects at high rates and undercorrects at low rates. Fridericia's formula uses the cube root of RR and is less rate-dependent in the physiological range. The AHA/ACCF/HRS recommendations prefer regression-based correction and caution against rate correction when RR variability is high.

**Arithmetic:** with QT = 398 ms and RR = 0.8824 s, QTc by Bazett is 398/√0.8824 = 423.7 ms. By Fridericia it is 398/0.8824^(1/3) = 414.9 ms. Both are below the 450 ms threshold for men and the 460 ms threshold for women.

**Branch point:** if the T wave is followed by a U wave, measure to the end of the T wave, not the U wave, or use the tangent method. If the T wave is flat or biphasic, the T end is uncertain and the QT should be reported with a wider uncertainty.

**Edge case:** in AF, the RR interval varies beat to beat. A single QTc is misleading. Report the QT at a stated RR, or use a correction that accounts for the variability, and state the limitation.

**QT uncertainty.** The QT interval has three sources of uncertainty that should be reported separately.

- **Fiducial uncertainty**: the QRS onset is usually within 5 ms; the T end is within 10 ms to 30 ms, depending on morphology and noise.
- **Lead uncertainty**: the QT varies across leads by 10 ms to 40 ms; the longest lead is used, but the choice affects the value.
- **Formula uncertainty**: Bazett and Fridericia differ by up to 40 ms at high rates.

A combined uncertainty of 20 ms to 50 ms is realistic. A report that gives QT to 1 ms without this context is overstating precision. The AHA/ACCF/HRS recommendation to validate computer-reported prolongation visually is a recognition of this uncertainty, not a comment on computer accuracy alone.

### Step 11. Classify the rhythm

**Input:** rate, regularity, P-wave presence and relationship, QRS width, and morphology.

**Output:** a rhythm category with a confidence and the criteria used.

**Rationale:** rhythm classification is a decision, not a measurement. It combines several features and a prior. The same RR irregularity can be sinus arrhythmia, AF, or frequent ectopy. The same wide QRS can be bundle branch block, ventricular tachycardia, or a paced rhythm.

**Rules of thumb:** sinus rhythm has upright P waves in I and II, a constant PR interval, and a rate of 60 bpm to 100 bpm. AF has absent P waves, an irregularly irregular RR interval, and a narrow QRS unless there is aberrancy. A premature atrial contraction has an early P wave and a narrow QRS. A premature ventricular contraction has a wide QRS and no preceding P wave.

**Branch point:** if the rhythm is irregular but P waves are present, consider sinus arrhythmia, multifocal atrial tachycardia, or frequent ectopy. If the rhythm is regular and wide, consider ventricular tachycardia, paced rhythm, or bundle branch block with a supraventricular rhythm.

**Edge case:** a short 10 s strip may not capture a paroxysmal arrhythmia. A normal 10 s strip does not rule out AF. This is why ambulatory monitoring and longer recordings exist.

**Rhythm decision table.** The table is a simplified aid, not a clinical algorithm.

| Regularity | P waves | QRS width | Likely rhythm | Confirm by |
|---|---|---|---|---|
| Regular | present, upright in II | narrow | sinus rhythm | PR consistency |
| Regular | absent | narrow | junctional | P-wave search in other leads |
| Regular | sawtooth | narrow | atrial flutter | atrial rate, block ratio |
| Irregular | absent | narrow | AF | multiple leads, RR histogram |
| Irregular | present | narrow | sinus arrhythmia, ectopy, MAT | P morphology |
| Regular | present | wide | bundle branch block, paced | morphology, spike |
| Regular | absent | wide | ventricular tachycardia | clinical context, AV dissociation |
| Irregular | present | wide | AF with aberrancy, ectopy | morphology |

The table has eight rows and at least three are dangerous if applied without context. Ventricular tachycardia, in particular, is a clinical emergency and is not a rhythm to classify from a 10 s strip by an automated rule without a validated claim.

### Step 12. Report with confidence and limitations

**Input:** the measurements, the rhythm classification, and the quality assessment.

**Output:** a structured report with values, units, reference ranges, confidence, and the limits of the lead set.

**Rationale:** a report that states "QT 398 ms" without the correction formula, the lead, and the T-end method is not reproducible. A report that states "normal ECG" from a single lead overstates what the record can support. The report is the product of the chain, and it must carry the chain's limits.

**Contents:** rate and rhythm; PR, QRS, QT, QTc with formula; axis; ST-T description; quality flags; lead set; comparison with prior records if available; and a statement of what the record cannot exclude.

**Branch point:** if the record is for a screening claim, the report should say "no evidence of AF in this recording" rather than "normal." The difference is the difference between a negative test and a diagnosis.

**Edge case:** an automated report that disagrees with the over-read should be treated as a false positive or false negative, not as an alternative interpretation. The disagreement rate is a validation metric.

**A note on confidence.** A measurement can be precise and still be wrong. The PR interval of 176 ± 6 ms is a precise estimate of the fiducial distance, but if the P onset was defined as the first deflection above 10% of the P amplitude, the estimate is conditional on that definition. The confidence statement should name the definition, not only the standard error. "PR 176 ± 6 ms (P onset at 10% threshold, lead II)" is reproducible. "PR 176 ± 6 ms" is not.

**A minimal report.** A reproducible report has the following structure:

```text
Record: 12-lead, 10 s, 500 samples/s, 1 µV/LSB
Quality: 11 of 12 leads interpretable; V3 noisy; no clipping
Rate: 68.0 bpm (mean RR 0.8824 s, 11 beats)
PR: 176 ms (P onset to QRS onset, lead II)
QRS: 86 ms (onset to J point, lead II)
QT: 398 ms (QRS onset to T end, tangent, lead II)
QTc: 423.7 ms (Bazett); 414.9 ms (Fridericia)
Axis: 53.1° (lead I +0.60 mV, aVF +0.80 mV)
Rhythm: sinus, regular, P upright in I and II
Limits: single time point; no prior comparison; V3 excluded
```

The structure forces every number to carry its method. A report that omits the method is not reproducible, and a report that omits the limits is not honest.

### Decision table for ECG quality

| Condition | Detect by | Effect on measurement | Action |
|---|---|---|---|
| Baseline wander | Low-frequency power above threshold | ST level and T end unreliable | High-pass at 0.05 Hz; re-measure |
| Muscle noise | High-frequency power, low QRS SNR | Fiducial jitter; false detections | Low-pass at 40 Hz for rhythm; flag for ST |
| Mains interference | Spectral peak at 50/60 Hz | Threshold instability | Notch; re-check |
| Clipping | Flat tops at the rail | Amplitude and ST bias | Reject the affected beats |
| Lead-off | Rail or status flag | No signal | Reject the lead |
| Limb lead reversal | P polarity and lead equations | Axis and regional diagnosis wrong | Correct labels or reject |
| Precordial misplacement | R-wave progression atypical | Regional diagnosis wrong | Flag; do not localise |
| Pacemaker | Narrow spikes before QRS | QRS duration and morphology altered | Use spike channel; different rules |
| Ectopy | Early beat with different morphology | Averaged beat smeared | Measure on individual beats |
| AF with rapid response | Irregular RR, absent P | QTc unreliable | Report QT at stated RR; no single QTc |
| Low voltage | Small QRS in limb leads | Amplitude criteria unreliable | Report; do not apply voltage criteria |
| Non-diagnostic bandwidth | Header shows 0.5-40 Hz | ST and QRS notching lost | Restrict to rhythm |

## 6. Worked example

### 6.1 The case

A 12-lead ECG is recorded for 10 s at 500 samples/s. The patient is a 54-year-old man at rest. The rhythm is regular. The record is stored at 1 µV per LSB, the PTB-XL convention. The goal is a structured measurement report: rate, PR, QRS, QT, QTc, and axis. The analysis uses lead II for the fiducials and leads I and aVF for the axis.

### 6.2 Acquisition arithmetic

- Samples: 500 samples/s × 10 s = 5,000 per lead.
- Leads: 12, so 60,000 samples per record.
- Storage: 60,000 × 2 bytes = 120 kB per record at 16 bits, before compression.
- Amplitude resolution: 1 µV per LSB. A 1 mV QRS is 1,000 codes. The 16-bit range at this scale is ±32.768 mV, which is wider than the ±5 mV used in Topic 01. The wider range accommodates electrode offsets and motion without clipping, at the cost of coarser resolution.
- Nyquist: 250 Hz. The diagnostic bandwidth of 150 Hz is preserved.

The resolution difference matters for ST measurement. A 0.05 mV ST elevation is 50 codes at 1 µV per LSB and 328 codes at 0.1526 µV per LSB. Both are measurable, but the coarser scale has less margin when the signal is noisy.

### 6.3 Reference ranges and where the case sits

| Measurement | Reference range | Case value | Position |
|---|---:|---:|---|
| Heart rate | 60-100 bpm | 68.0 bpm | mid-range |
| PR interval | 120-200 ms | 176 ms | upper half, normal |
| QRS duration | 80-100 ms | 86 ms | normal |
| QT interval | rate-dependent | 398 ms | normal for rate |
| QTc Bazett | < 450 ms men, < 460 ms women | 423.7 ms | normal |
| QTc Fridericia | < 450 ms men, < 460 ms women | 414.9 ms | normal |
| Frontal axis | -30° to +90° | 53.1° | normal |

The case is deliberately unremarkable. The teaching point is not the abnormality; it is that every normal value depends on a method, and the same record can produce an abnormal value if one method changes.

A second point is that the reference ranges are not interchangeable across sources. The PR upper limit is 200 ms in most adult references, but some use 210 ms. The QRS upper limit for normal is 100 ms in some sources and 110 ms in others. The QTc threshold is 450 ms for men and 460 ms for women in the AHA/ACCF/HRS statement, but other consensus documents use 440 ms or 470 ms. A product that hard-codes one set without citing it is making an unstated choice. The choice is defensible only if it is named.

### 6.4 Fiducial arithmetic

The detector finds 11 R peaks. The mean RR is 0.8824 s, so the heart rate is 60/0.8824 = 68.00 bpm. The fourth beat is selected for interval measurement because it is clean and not near an edge.

| Fiducial | Time (s) | Definition |
|---|---:|---|
| P onset | 3.916 | First deflection from baseline before the P peak |
| P peak | 3.970 | Maximum of the P wave |
| QRS onset | 4.092 | First high-slope deflection of the QRS |
| R peak | 4.156 | Maximum of the R wave |
| QRS offset (J point) | 4.178 | Return of the high-slope QRS to the ST segment |
| T peak | 4.365 | Maximum of the T wave |
| T end | 4.490 | Tangent intersection with baseline |

From these fiducials:

- PR = 4.092 - 3.916 = 0.176 s = 176 ms.
- QRS = 4.178 - 4.092 = 0.086 s = 86 ms.
- QT = 4.490 - 4.092 = 0.398 s = 398 ms.
- RR used for correction = 0.8824 s.

At 500 samples/s, each sample is 2 ms. The PR of 176 ms is 88 samples; the QRS of 86 ms is 43 samples; the QT of 398 ms is 199 samples. The QRS is only 43 samples wide, which is why a 100 Hz export (5 ms per sample) cannot measure it to better than 10 ms.

Beat-to-beat variability is a separate source of uncertainty. In sinus rhythm, the RR interval varies by 20 ms to 60 ms from breath to breath. The QT adapts to rate with a delay, so a single-beat QT at a single RR is not the steady-state QT. A report that measures one beat and applies one correction ignores both the variability and the adaptation.

A practical compromise is to measure the QT on the median beat and report the RR used, then repeat on three beats and report the range. If the range is wider than 20 ms, the single value is not representative.

The uncertainty on each fiducial, from the detector and from the waveform, is about 2 ms to 4 ms for QRS onset, 5 ms to 10 ms for P onset, and 10 ms to 25 ms for T end. Propagating these in quadrature:

- PR uncertainty: √(5² + 4²) ≈ 6 ms.
- QRS uncertainty: √(4² + 4²) ≈ 6 ms.
- QT uncertainty: √(4² + 25²) ≈ 25 ms.

A QT of 398 ± 25 ms overlaps the 423.7 ms QTc by construction, but the corrected value inherits the same 25 ms, so QTc is 423.7 ± 26.6 ms after dividing by √0.8824. The margin to the 450 ms threshold is about one standard deviation, not the 26 ms implied by subtracting the point estimates.

### 6.5 Correction arithmetic

Bazett: QTc = QT/√RR = 0.398/√0.8824 = 0.398/0.9394 = 0.4237 s = 423.7 ms.

Fridericia: QTc = QT/RR^(1/3) = 0.398/0.9593 = 0.4149 s = 414.9 ms.

The two formulas differ by 8.8 ms at this rate. Both are below 450 ms, so the conclusion is the same. At a higher rate the difference grows.

| Heart rate (bpm) | RR (s) | QTc Bazett (ms) | QTc Fridericia (ms) | Difference (ms) |
|---:|---:|---:|---:|---:|
| 60 | 1.0000 | 398.0 | 398.0 | 0.0 |
| 80 | 0.7500 | 459.5 | 438.2 | 21.3 |
| 100 | 0.6000 | 513.8 | 471.9 | 41.9 |

The table uses a fixed QT of 398 ms to isolate the formula effect. At 80 bpm, Bazett gives 459.5 ms and Fridericia 438.2 ms. The 450 ms threshold classifies the first as prolonged and the second as normal. The formula, not the patient, determines the label.

### 6.6 Axis arithmetic

Lead I net QRS = +0.60 mV. Lead aVF net QRS = +0.80 mV. The frontal axis is atan2(0.80, 0.60) = 53.1°. This is within the normal range of -30° to +90°.

A useful check is that the axis should be roughly perpendicular to the most isoelectric limb lead. If lead III were the most isoelectric lead, the axis would be near +30° or -150°, and the computed 53.1° would be plausible. If the computed axis and the isoelectric lead disagree by more than 15°, suspect a lead reversal or a measurement error.

### 6.7 Diagnostic 1: P-wave polarity reversal

The P wave in lead I is +0.150 mV in the normal record. In a simulated recording with the right-arm and left-arm electrodes swapped, the same fiducial index has a P wave of -0.150 mV. The QRS in lead I also inverts, and aVR becomes upright.

The diagnostic is not the negative P wave alone. Dextrocardia and an ectopic atrial rhythm can also produce a negative P in lead I. The confirmatory pattern is the combination: lead I inverted, aVR upright, and the limb lead equations inconsistent with the labels. In practice, the technologist repeats the ECG with correct placement, and the comparison confirms the swap.

The consequence of missing the reversal: the axis is reported as rightward, lead I is read as showing a lateral infarct, and aVR is read as normal. The report is wrong in three ways from one electrode swap.

The arithmetic of the reversal is worth stating. Lead I is LA - RA. Swapping the electrodes negates lead I. Lead II becomes LL - LA, which is the original lead III. Lead III becomes LL - RA, which is the original lead II. aVR and aVL swap. aVF is unchanged because it depends on LL and the average of RA and LA, which is invariant to the swap. The predicted pattern is therefore: I inverted, II and III exchanged, aVR and aVL exchanged, aVF unchanged. The observed pattern in the simulated record matches this prediction exactly, which is what confirms the reversal rather than dextrocardia.

### 6.8 Diagnostic 2: QTc formula discordance

At 68 bpm, Bazett and Fridericia agree within 9 ms. At 80 bpm, they differ by 21 ms, and the 450 ms threshold separates them. At 100 bpm, they differ by 42 ms.

The diagnostic is a simple check: compute QTc by two formulas and report both when they disagree across a clinical threshold. If the two formulas agree, the conclusion is robust to the correction choice. If they disagree, the conclusion depends on the formula, and the report should say so.

A second diagnostic is the T-end method. Measuring to the visual return to baseline rather than the tangent intersection can add 20 ms to 40 ms to the QT, which at 80 bpm adds 23 ms to 47 ms to the QTc. Two readers using two methods can disagree by more than the formula difference. The method must be stated.

The T-end arithmetic is easy to check. A T wave with a peak at 4.365 s and a tangent intersection at 4.490 s gives a T duration of 125 ms. If a U wave extends the visible return to 4.520 s, the QT increases from 398 ms to 428 ms, a 30 ms change. Corrected by Bazett at 68 bpm, the QTc moves from 423.7 ms to 455.6 ms, crossing the 450 ms threshold. A U wave that is included by the measurement method can turn a normal QT into a prolonged one.

### 6.9 Varying one assumption

Suppose the heart rate is 80 bpm instead of 68 bpm, with the same 398 ms QT. The RR is 0.75 s. Bazett gives 459.5 ms, which is prolonged by the 450 ms threshold. Fridericia gives 438.2 ms, which is normal. The clinical conclusion changes from "normal repolarisation" to "borderline prolongation" depending on the formula, with no change in the patient.

Suppose instead that the QT is measured in lead V2 rather than lead II. The T wave in V2 is often larger and its end is later, so the QT may be 10 ms to 20 ms longer. At 68 bpm, a 15 ms increase gives QTc Bazett 439.7 ms, still normal. At 80 bpm, the same 15 ms gives 477.4 ms, clearly prolonged. The lead choice interacts with the rate.

The lesson is that QTc is not a single number. It is a family of numbers indexed by formula, lead, and T-end method. A report that gives one number without the index is under-specified.

The sensitivity is summarised below. The baseline is the case value at 68 bpm with the tangent method and lead II.

| Assumption changed | Resulting QTc (ms) | Threshold status | Change from baseline |
|---|---:|---|---:|
| Baseline: Bazett, lead II, tangent | 423.7 | normal | 0.0 |
| Fridericia, same measurement | 414.9 | normal | -8.8 |
| Bazett, lead V2, +15 ms QT | 439.7 | normal | +16.0 |
| Bazett, U wave included, +30 ms QT | 455.6 | prolonged | +31.9 |
| Bazett, heart rate 80 bpm, QT fixed | 459.5 | prolonged | +35.8 |
| Fridericia, heart rate 80 bpm, QT fixed | 438.2 | normal | +14.5 |

The table has two rows that cross the threshold (455.6 and 459.5) and two that do not (439.7 and 438.2). The difference between the crossing and non-crossing rows is not the patient; it is the method. A validation that reports one method and one threshold is reporting one point from this table.

### 6.10 Conclusion of the case

The record shows a regular rhythm at 68.00 bpm, PR 176 ms, QRS 86 ms, QT 398 ms, QTc 423.7 ms by Bazett and 414.9 ms by Fridericia, and a frontal axis of 53.1°. All values are within common reference ranges. The record is interpretable for rhythm and intervals. It is not interpretable for regional ST analysis without a quality review of all 12 leads, which is outside this example.

The two diagnostics (P polarity, QTc formula discordance) catch errors that would survive a single-number report. The assumption that matters most is the correction formula at rates above 80 bpm, where the choice changes the label.

### 6.11 What the case does not show

The case is a single normal record with synthetic noise. It does not show four things that a real validation must address.

- **Abnormal morphology.** A normal QRS is easy to bound. A bundle branch block, a paced beat, or a pre-excited beat has a slurred onset and offset, and the fiducial uncertainty grows.
- **Population variation.** The reference ranges in Section 6.3 are adult ranges. Pediatric ECGs have different normal values and different rate ranges.
- **Lead quality variation.** The synthetic record has identical quality in every lead. A real record may have one good limb lead and five noisy precordial leads, and the lead used for measurement may not be the lead that was planned.
- **Clinical outcome.** The case validates measurements, not diagnoses. A normal QT does not exclude a drug effect, and a normal axis does not exclude disease.

The four omissions are the agenda for a real validation: abnormal morphology, population, lead quality, and outcome. A report that generalises from the case without addressing them is overreaching.

### 6.12 Reproducibility checklist for this case

A reader who wants to reproduce the numbers needs the following. If any item is missing, the result is not reproducible.

| Item | Value in this case |
|---|---|
| Sampling rate | 500 samples/s |
| Amplitude scale | 1 µV per LSB |
| Lead used for fiducials | II |
| Leads used for axis | I, aVF |
| P-onset definition | 10% of P amplitude above local baseline |
| QRS onset/offset definition | derivative envelope at 12% of local maximum |
| T-end definition | tangent to steepest downslope |
| QT correction | Bazett and Fridericia |
| Beat selection | fourth detected beat, clean, not at edge |
| Detector | derivative, square, 120 ms integration, 50% threshold |

The table is short because the case is simple. A real record adds lead quality, beat selection, and adjudication. The habit is the same: if the method cannot be written as a table, it cannot be reproduced.

## 7. Code, query, or calculation

The script below generates a synthetic ECG with known fiducials, detects R peaks, measures PR, QRS, and QT, computes QTc by two formulas, computes the frontal axis, and simulates a limb lead reversal. It uses only the standard library.

```python
import math

FS, DUR, HR = 500.0, 10.0, 68.0

def g(t, c, a, w):
    return a * math.exp(-0.5 * ((t - c) / w) ** 2)

def beat(t, p_amp=0.15):
    return (g(t, -0.16, p_amp, 0.025) + g(t, -0.020, -0.10, 0.008)
            + g(t, 0.0, 1.00, 0.010) + g(t, 0.022, -0.18, 0.009)
            + g(t, 0.25, 0.35, 0.055))

def synth(fs, dur, hr, invert=False):
    n = int(fs*dur); x = [0.0]*n; rr = 60.0/hr; t0 = 0.6; k = 0
    while t0 + k*rr < dur - 0.5:
        c = t0 + k*rr
        for i in range(max(0, int((c-0.4)*fs)), min(n, int((c+0.6)*fs))):
            x[i] += beat(i/fs - c)
        k += 1
    return [-v for v in x] if invert else x

def detect_r(x, fs):
    d = [0.0]*len(x)
    for i in range(1, len(x)): d[i] = x[i]-x[i-1]
    sq = [v*v for v in d]; m = int(0.12*fs); acc = 0.0; integ = [0.0]*len(x)
    for i, v in enumerate(sq):
        acc += v
        if i >= m: acc -= sq[i-m]
        integ[i] = acc/m
    thr = 0.5*max(integ); peaks = []; i = 0
    while i < len(integ):
        if integ[i] > thr:
            j = i
            while j+1 < len(integ) and integ[j+1] > thr: j += 1
            pk = max(range(i, j+1), key=lambda k: integ[k])
            lo = max(0, pk-int(0.05*fs)); hi = min(len(x), pk+int(0.05*fs))
            peaks.append(max(range(lo, hi), key=lambda k: abs(x[k])))
            i = j + int(0.2*fs)
        else:
            i += 1
    return peaks

def env(x, fs, half_ms=12):
    h = int(half_ms*fs/1000)
    d = [0.0]+[abs(x[i+1]-x[i-1]) for i in range(1, len(x)-1)]+[0.0]
    return [max(d[max(0,i-h):i+h+1]) for i in range(len(d))]

def qrs_bounds(x, r, fs):
    e = env(x, fs); lo = max(0, r-int(0.15*fs)); hi = min(len(x), r+int(0.15*fs))
    thr = 0.12*max(e[lo:hi])
    on = r
    while on > lo and e[on] > thr: on -= 1
    off = r
    while off < hi and e[off] > thr: off += 1
    return on, off

def p_peak(x, r, fs):
    lo = max(0, r-int(0.30*fs)); hi = r-int(0.07*fs)
    return max(range(lo, hi), key=lambda k: x[k]) if hi > lo else None

def p_onset(x, pk, fs):
    base = min(x[max(0,pk-int(0.08*fs)):pk])
    amp = x[pk]-base; thr = 0.10*amp
    i = pk
    while i > max(0, pk-int(0.12*fs)) and (x[i]-base) > thr:
        i -= 1
    return i

def t_end(x, r, fs):
    lo = r+int(0.12*fs); hi = min(len(x)-1, r+int(0.50*fs))
    tp = max(range(lo, hi), key=lambda k: x[k])
    best, slope = tp, 0.0
    for k in range(tp, min(hi, tp+int(0.15*fs))):
        s = x[k+1]-x[k]
        if s < slope: slope, best = s, k
    if slope == 0: return tp
    return min(best + int(round(x[best]/(-slope))), hi)

x = synth(FS, DUR, HR)
peaks = detect_r(x, FS)
rr = [(peaks[i+1]-peaks[i])/FS for i in range(len(peaks)-1)]
mrr = sum(rr)/len(rr); hr = 60/mrr
print("peaks=%d mean_RR=%.4f HR=%.2f" % (len(peaks), mrr, hr))

r = peaks[4]
on, off = qrs_bounds(x, r, FS)
p = p_peak(x, r, FS); po = p_onset(x, p, FS); te = t_end(x, r, FS)
print("r_s=%.3f on_s=%.3f off_s=%.3f P_s=%.3f T_end_s=%.3f"
      % (r/FS, on/FS, off/FS, p/FS, te/FS))
print("PR_ms=%.1f QRS_ms=%.1f QT_ms=%.1f"
      % ((on-po)/FS*1000, (off-on)/FS*1000, (te-on)/FS*1000))
qt = (te-on)/FS
print("QTcB=%.1f QTcF=%.1f" % (qt/math.sqrt(mrr)*1000, qt/(mrr**(1/3))*1000))

for h in (60, 80, 100):
    rr_h = 60/h
    print("HR%d QTcB=%.1f QTcF=%.1f"
          % (h, 400/math.sqrt(rr_h), 400/(rr_h**(1/3))))

yr = synth(FS, DUR, HR, invert=True)
pr = detect_r(yr, FS)
pn = p_peak(x, peaks[4], FS)
print("P_normal=%.3f P_reversed_at_same_index=%.3f" % (x[pn], yr[pn]))
print("axis_deg=%.1f" % math.degrees(math.atan2(0.80, 0.60)))
```

Expected output:

```text
peaks=11 mean_RR=0.8824 HR=68.00
r_s=4.156 on_s=4.092 off_s=4.178 P_s=3.970 T_end_s=4.490
PR_ms=176.0 QRS_ms=86.0 QT_ms=398.0
QTcB=423.7 QTcF=414.9
HR60 QTcB=400.0 QTcF=400.0
HR80 QTcB=461.9 QTcF=440.3
HR100 QTcB=516.4 QTcF=474.3
P_normal=0.150 P_reversed_at_same_index=-0.150
axis_deg=53.1
```

The output differs slightly from the table in Section 6.4 because the table uses a fixed QT of 398 ms and computes QTc at 80 bpm as 459.5 ms, while the script uses a fixed QT of 400 ms for the formula comparison and gets 461.9 ms. The difference is the 2 ms QT choice, not an arithmetic error. Both are shown to make the dependency explicit: state the QT you are correcting.

Three implementation notes.

The QRS bounds are found from a smoothed derivative envelope, not from the raw derivative. The raw derivative oscillates within the QRS and crosses zero at the R peak, so a naive threshold stops at the peak. The envelope fixes this by taking a local maximum over a 12 ms window.

The P onset is found by walking left from the P peak until the signal falls below 10% of the P amplitude above the local baseline. The 10% threshold is a convention, not a law; a 5% threshold shifts the onset by a few milliseconds and the PR interval by the same amount.

The T end is found by the tangent method: the steepest downslope after the T peak is extended to the baseline. For a Gaussian T wave, this lands at the true end. For a real T wave with a U wave, the steepest downslope may be on the U wave, and the method must be constrained to the T wave.

## 8. Data, standards, and regulatory context

ECG is one of the most standardised signals in medicine. The standards are mature, and the gap between a compliant device and a clinically adequate measurement is usually a gap in documentation, not in technology.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| AHA/ACCF/HRS 2007, Part I | Acquisition, bandwidth, sampling, display | Professional statement | 0.05-150 Hz diagnostic band; 500 samples/s storage |
| AHA/ACCF/HRS 2007, Part II | Diagnostic statement lexicon | Professional statement | 117 statements; a common vocabulary for reports |
| AHA/ACCF/HRS 2009, Part IV | ST segment, T and U waves, QT | Professional statement | QT definitions, correction cautions, thresholds |
| AHA/ACCF/HRS 2009, Part III | Intraventricular conduction | Professional statement | Bundle branch block criteria |
| ANSI/AAMI EC11 | Diagnostic electrocardiograph performance | Consensus standard | Frequency response and accuracy tests |
| ANSI/AAMI EC57 | Testing of cardiac rhythm and ST-segment algorithms | Consensus standard | Sensitivity and specificity reporting on annotated databases |
| IEC 60601-2-25:2011 | Diagnostic electrocardiographs | Binding when claimed | Accuracy tables for intervals and waveforms |
| IEC 60601-2-27 | Cardiac monitors with alarms | Binding when claimed | Monitoring bandwidth and alarm performance |
| IEC 60601-2-47 | Ambulatory ECG systems | Binding when claimed | Holter and patch performance |
| FDA Diagnostic ECG guidance (1998) | 510(k) content for diagnostic ECG | Guidance | EC11 conformance; excludes interpretive algorithms |
| FDA arrhythmia detector special controls (2003) | Algorithm validation | Special controls | EC57 databases; three cardiologist annotations |
| 21 CFR 870.2345 | Over-the-counter ECG software | Regulation | Clinical performance, human factors, labelling |
| HL7 FHIR R5 | ECG waveform exchange | Voluntary standard | `SampledData` and `Observation` structure |
| LOINC | Measurement codes | Voluntary; widely required | Heart rate, PR, QRS, QT codes |
| SNOMED CT | Clinical findings | Voluntary; licensed | Rhythm and morphology concepts |

### 8.1 What the AHA/ACCF/HRS recommendations actually say about QT

The Part IV statement is specific enough to be actionable, and it is often summarised incorrectly.

- The QT interval is measured from the onset of the QRS to the end of the T wave.
- The lead with the longest QT should be used, usually V2 or V3.
- When T and U waves fuse, measure in aVR or aVL, or use the tangent method.
- Computer-reported QT prolongation should be visually validated.
- Linear regression correction is preferred over Bazett's formula, which overcorrects at high heart rates.
- Rate correction should not be applied when RR variability is large, as in AF.
- A QTc of 460 ms or more in women and 450 ms or more in men is commonly considered prolonged; 390 ms or less is short.
- QT dispersion should not be included in routine reports.

The statement is a professional recommendation, not a regulation. A device can be legally marketed without following it, but a clinical reviewer will use it as the reference. The gap between the recommendation and a product's method is a question the product team should be able to answer.

### 8.2 Binding requirement versus convention

The distinction matters when a team is deciding what to document.

| Question | Binding? | Source |
|---|---|---|
| Must the device meet IEC 60601-2-25? | Only if claimed or required by the regulator | IEC |
| Must the ECG be stored at 500 samples/s? | No; it is a professional recommendation | AHA/ACCF/HRS |
| Must the QT be corrected with a regression formula? | No; it is a recommendation | AHA/ACCF/HRS |
| Must the algorithm be tested on EC57 databases? | Effectively yes for a US rhythm claim | FDA special controls |
| Must the report use a standard statement lexicon? | No, but it helps interoperability | AHA/ACCF/HRS Part II |
| Must the waveform use FHIR `SampledData`? | Only if the integration requires FHIR | HL7 |

A team that treats every convention as binding over-documents and under-tests. A team that treats every standard as optional under-documents and fails review. The table is a way to ask the question per row.

### 8.3 The regulatory pathway for ECG software

The pathway depends on the claim.

- **Diagnostic ECG device**: a 510(k) with a predicate, conformance to EC11, and performance testing. The 1998 guidance covers the device but explicitly excludes automated interpretation algorithms.
- **Arrhythmia detector and alarm**: a 510(k) under the special controls for 21 CFR 870.1025, with algorithm testing on annotated databases per EC57 and annotations by at least three qualified cardiologists.
- **Over-the-counter ECG software**: 21 CFR 870.2345, established in 2022, with clinical performance testing, a previously adjudicated dataset, human factors, and specific labelling. The Apple ECG app was classified under this framework through De Novo DEN180044 in 2018.
- **Research or wellness**: no device claim, but the general wellness policy limits the claims that can be made.

The distinction between "detects AF" and "records an ECG" is the difference between a diagnostic claim and a measurement claim. The signal chain may be identical. The evidence is not.

### 8.4 How EC57 structures algorithm validation

ANSI/AAMI EC57 is the standard that most rhythm and ST algorithm validations cite. Its structure is worth knowing because it determines what a performance claim can say.

- The algorithm is run on annotated databases, and each beat or episode is compared with the reference annotation.
- Results are reported as true positives, false positives, false negatives, and true negatives, from which sensitivity, specificity, positive predictive value, and false-positive rate are computed.
- The standard distinguishes beat-level from episode-level scoring. A detector that finds every beat but fragments an episode can look good at beat level and bad at episode level.
- The development database must not be used for validation. At least two databases are recommended, and annotations should come from at least three qualified cardiologists.
- The standard does not prescribe a single threshold for acceptable performance. The claim and the clinical context determine the threshold.

The practical consequence is that "we validated on MIT-BIH" is not a validation statement. MIT-BIH is one database, with 48 records and a specific annotation protocol. A validation needs the database versions, the annotation source, the scoring rule, and the confidence intervals.

### 8.5 What the standards do not cover

Three gaps are worth naming.

First, the standards specify performance on test waveforms and annotated databases, not on the population the product will serve. A device can meet EC11 and still fail on patients with a body habitus, skin condition, or rhythm not represented in the test set.

Second, the standards do not specify how to handle unclassifiable records. A product that reports "inconclusive" 12% of the time and a product that forces a classification in every record can both be compliant. The handling of inconclusive records changes the sensitivity and specificity that a user experiences, and it must be reported.

Third, the standards do not specify the reference standard for a new claim. For AF, the reference is usually a simultaneous 12-lead over-read. For a novel morphology claim, the reference may be imaging or a clinical outcome, and the validation design is correspondingly harder.

A fourth gap is the handling of software updates. A model that is retrained or a filter that is changed is a new device function in practice, even if the regulatory pathway treats it as a change to an existing product. The documentation must state what changed and what evidence supports the change. The FDA guidance on predetermined change control plans exists for this reason; it is discussed in Topic 08.

## 9. Industry practice and case evidence

### 9.1 The Apple ECG app

The Apple ECG app was granted De Novo classification on 14 August 2018 (DEN180044). The validation study recorded 30 s single-lead ECGs simultaneously with 12-lead ECGs in 588 eligible participants at five US sites. Three blinded cardiologists adjudicated the reference strips.

The reported performance for classifiable recordings was sensitivity 98.3% and specificity 99.6%. Including unclassifiable recordings, sensitivity was 95.5% and specificity 97.1%. About 87.8% of recordings were classifiable; 12.2% were inconclusive. The waveform was compared with lead I and passed morphology equivalence in 99.2% of recordings.

Three lessons follow. First, the performance claim is conditional on classifiability, and the inconclusive rate is part of the user experience. Second, the study was conducted with staff assistance and rested arms, so the real-world inconclusive rate is likely higher. Third, the claim is for a single-lead rhythm classification, not for a diagnostic 12-lead interpretation.

### 9.2 The Apple Heart Study

The Apple Heart Study used a smartwatch PPG-based irregular-rhythm notification, not the ECG app, in 419,297 participants. Of these, 2,161 (0.52%) received a notification. Among participants who returned ECG patches, 34% had AF confirmed. The study demonstrated feasibility at scale and also showed the downstream workload: a notification leads to a confirmatory test, and most notified participants did not have AF on the patch.

The ECG lesson is that a screening notification is not a diagnosis. The confirmatory ECG is part of the care pathway. The signal-chain lesson is that PPG artifact and RR irregularity interact: motion can produce irregular intervals that look like AF.

### 9.3 PTB-XL and public benchmarks

PTB-XL contains 21,799 clinical 12-lead ECGs from 18,869 patients, recorded at 400 Hz and upsampled to 500 Hz, with a 100 Hz version for convenience. It provides 71 SCP-ECG statements, five diagnostic superclasses, and a recommended patient-respecting 10-fold split. Folds 9 and 10 have the highest label quality and are recommended as validation and test sets.

The dataset is a benchmark, not a validation set for a product. A model trained and tested on PTB-XL has demonstrated performance on PTB-XL, a German clinical population recorded between 1989 and 1996 on Schiller devices. It has not demonstrated performance on a wrist device, a different population, or a different lead set. Cross-dataset evaluation is the minimum for a generalisation claim.

### 9.4 The PhysioNet challenges

The 2020 challenge distributed 12-lead ECGs at 500 Hz with variable record lengths and missing leads. The 2021 challenge extended the task to 12-, 6-, 3-, and 2-lead recordings. Both challenges required teams to handle reduced lead sets, which is the same problem a wearable product faces.

The documented lesson from the challenge rules is that the evaluation is per record and per patient, and that a team which ignores the lead set or the sampling rate loses performance. The inference is that lead-set robustness is a property of the training design, not of the model architecture alone.

### 9.5 Handheld and patch ECG products

Handheld single-lead recorders and ambulatory patches occupy a middle ground between the wrist ECG and the 12-lead. A handheld device records a lead I approximation when the user places it on the chest or holds it in both hands; a patch records one or two leads continuously for days.

The documented validation pattern for handheld devices is similar to the Apple design: a simultaneous 12-lead reference, blinded over-read, and a classifiable/unclassifiable split. The documented limitation is also similar: the recording conditions are controlled, and the population is selected.

The patch pattern differs. A patch is worn continuously, so the validation question is not only "does the algorithm classify a 30 s strip" but "does the system detect episodes over days, and what is the false-alert burden." The metrics shift from sensitivity and specificity per strip to episode detection rate, time to detection, and alert rate per patient-day.

The inference for product teams is that the validation metric must match the use. A per-strip sensitivity does not describe a continuous monitor, and an episode detection rate does not describe a spot-check device.

### 9.5.1 The documented versus inferred ledger

| Statement | Status |
|---|---|
| Apple ECG app: sensitivity 98.3%, specificity 99.6% on classifiable recordings | Documented in the FDA De Novo summary and Apple's validation paper |
| Apple ECG app: 87.8% of recordings classifiable | Documented in the same sources |
| Apple Heart Study: 0.52% of 419,297 participants notified | Documented in the NEJM publication |
| PTB-XL: 21,799 records, 18,869 patients, 500 Hz | Documented in the PhysioNet dataset description |
| A 10 s strip misses most paroxysmal AF | Inference from episode duration and recording length |
| Single-lead ST analysis is unreliable | Inference from spatial coverage; supported by guideline statements |
| Longer monitoring increases AF yield | Documented in ambulatory monitoring literature |
| Algorithm performance falls on a new device | Inference from domain-shift principles; magnitude is study-specific |

The ledger is not exhaustive. Its purpose is to show that the strong claims in this section are sourced, and the weak claims are labelled. A reader can check the first four rows against the cited documents and accept or reject the last four as reasoned inferences.

### 9.6 Ambulatory patch monitoring

Ambulatory ECG patches record 1 to 3 leads for days to weeks. The diagnostic yield for paroxysmal AF depends on recording duration: a 24 h Holter captures fewer episodes than a 14-day patch. The signal chain must handle motion, non-wear, battery, and data transmission. The clinical workflow must handle the alert burden.

The documented evidence is that longer monitoring increases AF detection. The inference for product teams is that the recording duration is part of the measurement specification, not an operational detail. A 30 s recording and a 14-day recording measure different things.

A patch also changes the reference standard. A 14-day patch may detect an episode that a 24 h Holter misses, so a study that compares a patch to a Holter will find more AF in the patch arm for reasons that are partly detection, not incidence. This is a form of length-time bias, and it complicates any comparison between monitoring strategies. The signal-chain lesson is that the duration and the reference must be matched when comparing two devices.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Limb lead reversal | Negative P in lead I; upright aVR | RA/LA or LA/LL swap | Wrong axis and regional diagnosis | P polarity check; repeat ECG |
| Precordial misplacement | Atypical R-wave progression | V1/V2 too high or too low | Wrong anterior diagnosis | Anatomical landmarks; flag |
| QT measured from R peak | Short QT, formula disagreement | Fiducial definition error | Underestimated prolongation | Measure from QRS onset |
| T end at visual return | Long QT, reader variability | U wave included | Overestimated prolongation | Tangent method; state method |
| Bazett at high rate | QTc rises faster than expected | Overcorrection | False prolongation | Report Fridericia or regression |
| Single-lead ST claim | ST changes in one lead only | No spatial coverage | Missed or false infarction | Restrict claim to rhythm |
| AF from RR alone | Irregular intervals with P waves | Sinus arrhythmia or ectopy | False AF | Require P-wave assessment |
| Unclassifiable counted as negative | High apparent specificity | Inconclusive records excluded | Inflated performance | Report inconclusive rate |
| Clipping on R wave | Flat-topped R | Gain or range too low | Amplitude and ST bias | Detect and reject |
| Baseline wander | ST level drifts | Respiration or motion | False ST change | 0.05 Hz high-pass; re-measure |
| Mains interference | 50/60 Hz peak | Poor electrode contact | False detections | Notch; check impedance |
| Pacemaker spike removed | No spike before wide QRS | Low-pass at 40 Hz | Missed pacing | Wideband channel |
| Noise averaged into template | Smeared QRS | Ectopy in median beat | Interval bias | Exclude ectopic beats |
| Rate from one interval | Instantaneous rate unstable | Normal variability | Misleading display | Average over 10 s or more |
| Over-reading a 10 s strip | "Normal" from a short record | Paroxysmal events missed | False reassurance | State duration and limits |
| Algorithm-human disagreement ignored | Discordance rate unknown | No over-read comparison | Unmeasured error | Report agreement and kappa |
| Wrong correction formula | QTc changes with rate faster than expected | Formula mismatch | Misclassified prolongation | Report two formulas |
| Lead-specific QT ignored | QT differs by 20 ms across leads | Single-lead measurement | Underestimated prolongation | Use the longest lead |
| Averaging across rhythm change | Template smeared | Mixed morphologies | Interval bias | Exclude ectopic beats |
| Segment padding at edges | First and last beat distorted | Zero padding or filter transient | Interval bias | Exclude boundary beats |
| Duplicate records in test set | Unrealistically high accuracy | Patient leakage | Inflated performance | Patient-level split |
| Overlapping epochs in training | Adjacent windows nearly identical | Segment leakage | Inflated performance | Split by record or patient |
| Label noise from a single reader | Disagreement with a second reader | Unmeasured reference error | Biased metrics | Adjudication; report kappa |
| Threshold tuned on test set | Optimistic operating point | Test-set leakage | Invalid claim | Prespecify threshold |

Two red flags deserve emphasis.

**An interval reported without its lead or method.** The QT in lead II and the QT in V2 can differ by 20 ms. The PR measured from the P peak and from the P onset can differ by 50 ms. A report that gives a number without the lead and the fiducial is not a measurement; it is a number that happens to be in the right units.

**A performance claim that does not state the inconclusive rate.** A sensitivity of 98% with a 12% inconclusive rate is a different product from a sensitivity of 98% with a 2% inconclusive rate. The first forces a decision on one in eight users; the second rarely does. The inconclusive rate belongs in the headline, not in a footnote.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Electrodes are correctly placed | Lead equations and axis | Reversal, misplacement | P polarity; repeat ECG |
| The lead set matches the claim | Diagnostic coverage | Single lead for a 12-lead claim | Compare claim with leads |
| Calibration is known | Amplitude thresholds | Unknown gain or units | Calibration pulse; header |
| The rhythm is stable over the record | Averaging and single QTc | Paroxysmal AF; ectopy | RR trend; beat morphology |
| P waves are visible in at least one lead | PR and rhythm | Low amplitude; buried P | Multi-lead review |
| The T end is identifiable | QT | Flat T; U wave; noise | Two methods; two readers |
| Bazett or Fridericia is appropriate | Rate correction | High rate; AF | Report both; regression |
| The reference over-read is correct | Validation | Reader variability | Multiple readers; adjudication |
| The population resembles the validation set | Generalisation | Different age, sex, comorbidity | External validation |
| The device bandwidth matches the claim | ST and morphology | 0.5-40 Hz monitor mode | Check header and mode |
| Inconclusive records are reported | Performance interpretation | Forced classification | Report rate and handling |
| A 10 s record represents the day | Screening | Paroxysmal events | Longer monitoring |
| The QRS detector is correct on every beat | Interval measurement | Ectopy, noise, pacing | Compare with adjudicated beats |
| The reference standard is independent of the test | Validation | Same reader sees both | Blind the over-read |
| The threshold transfers across populations | Decision | Different prevalence, age, sex | Recalibrate; external test |
| The algorithm's "normal" matches the clinical normal | Reporting | Different statement lexicon | Map to a standard lexicon |
| The recording environment resembles deployment | Generalisation | Clinic versus home | Free-living validation |
| The device mode is known | Interpretation | Monitor versus diagnostic bandwidth | Record mode in metadata |
| The statement lexicon is stable | Reporting and comparison | Lexicon version change | Record lexicon version |
| The P-wave detector generalises across leads | PR measurement | Low-amplitude P; lead isoelectric | Test per lead |
| The QT correction is appropriate for the rate range | QTc | Tachycardia, bradycardia | Report two formulas |

## 12. Exercises

### Exercise 1. Lead equations (recall, 20 min)

**Deliverable:** compute leads I, II, III, aVR, aVL, and aVF from RA = -0.2 mV, LA = +0.5 mV, LL = +0.9 mV at a single instant. Verify I + III = II.

**Acceptance criteria:** correct arithmetic; augmented leads scaled by 1.5; units in mV.

**Time:** 20 minutes.

### Exercise 2. Interval measurement (application, 40 min)

**Deliverable:** from the fiducials in Section 6.3, compute PR, QRS, QT, and QTc by two formulas. Then recompute PR using the P peak instead of the P onset and state the error.

**Acceptance criteria:** all intervals in ms; the peak-based error quantified; formulas named.

**Time:** 40 minutes.

### Exercise 3. Reversal detection (application, 35 min)

**Deliverable:** a decision rule that flags RA/LA reversal using P polarity in I and aVR, QRS polarity in I, and the lead equations. State the sensitivity and the false-positive sources.

**Acceptance criteria:** rule is deterministic; dextrocardia and ectopic atrial rhythm named as confounders; no reliance on a single lead.

**Time:** 35 minutes.

### Exercise 4. Quality report (critique, 45 min)

**Deliverable:** a review of a 12-lead ECG with baseline wander and muscle noise, deciding which measurements are interpretable and which are not.

**Acceptance criteria:** per-measurement decision; thresholds named; no blanket "poor quality" without a consequence.

**Time:** 45 minutes.

### Exercise 5. AF validation design (design, 60 min)

**Deliverable:** a one-page design for a single-lead AF detector validation, including population, reference standard, patient-level split, inconclusive category, and co-primary endpoints.

**Acceptance criteria:** sensitivity and specificity prespecified; inconclusive rate reported; no segment-level leakage; sample size justified.

**Time:** 60 minutes.

### Exercise 6. QTc sensitivity (application, 30 min)

**Deliverable:** a table of QTc by Bazett and Fridericia at heart rates of 50, 80, 110, and 140 bpm for a fixed QT of 400 ms. Identify the rate at which the two formulas cross the 450 ms threshold.

**Acceptance criteria:** correct arithmetic; crossing point identified; clinical implication stated.

**Time:** 30 minutes.

### Exercise 7. Single-lead limits (critique, 25 min)

**Deliverable:** a list of five diagnoses that a single lead I recording cannot reliably support, with the reason for each.

**Acceptance criteria:** each item names the missing spatial information; at least one ST and one conduction example.

**Time:** 25 minutes.

### Exercise 8. Lead-set limits (application, 30 min)

**Deliverable:** for each of the six lead groups in the Step 1 table of Section 5, state one diagnosis that the group can support and one that it cannot.

**Acceptance criteria:** each pair names the spatial information that is present and absent; no claim exceeds the lead set.

**Time:** 30 minutes.

### Exercise 9. Report template (design, 45 min)

**Deliverable:** a structured ECG report template with fields for measurement, quality, rhythm, confidence, and limitations.

**Acceptance criteria:** every numeric field has a unit and a method; quality is per measurement; limitations are explicit.

**Time:** 45 minutes.

### Marking standard

Calculation exercises are marked on the arithmetic, the units, and the stated method. A correct number with no unit is not a pass. A number with a unit but no method is a partial pass. A design exercise is marked against the criteria in the exercise and against the rubric in Section 13. A critique exercise is marked on whether each finding names a consequence, not on the number of findings.

## 13. Self-check answers

**1. What is lead II in terms of electrodes?** LL - RA.

**2. Why are augmented leads scaled by 1.5?** The augmentation increases the amplitude relative to the bipolar limb leads so that the six limb leads have comparable scale. Without the factor, aVR, aVL, and aVF would appear smaller.

**3. What is the normal PR interval?** 120 ms to 200 ms, measured from P onset to QRS onset.

**4. Why is a peak-based PR measurement wrong?** The P peak occurs after the P onset, so the interval is shortened by roughly half the P duration, commonly 40 ms to 60 ms.

**5. What is the J point?** The junction between the end of the QRS and the start of the ST segment. It is the QRS offset.

**6. Compute QTc by Bazett for QT 400 ms and RR 0.64 s.** 0.400/√0.64 = 0.400/0.8 = 0.500 s = 500 ms.

**7. Compute QTc by Fridericia for the same values.** 0.400/0.64^(1/3) = 0.400/0.8618 = 0.464 s = 464 ms.

**8. At what rate do the two formulas disagree across 450 ms for a 400 ms QT?** Bazett exceeds 450 ms when √RR < 400/450 = 0.8889, so RR < 0.7901 s, which is a rate above 75.9 bpm. Fridericia exceeds 450 ms when RR^(1/3) < 0.8889, so RR < 0.7023 s, a rate above 85.4 bpm. Between about 76 bpm and 85 bpm, Bazett calls the QT prolonged and Fridericia does not.

**9. What is the frontal axis if lead I is +0.5 mV and aVF is -0.5 mV?** atan2(-0.5, 0.5) = -45°, which is left axis deviation.

**10. Why is a negative P wave in lead I not sufficient for a lead reversal diagnosis?** Dextrocardia and an ectopic atrial rhythm can also produce it. The diagnosis requires the full pattern, including aVR polarity and the lead equations.

**11. What is the difference between sensitivity including and excluding unclassifiable records?** Excluding them raises apparent sensitivity and specificity because the hardest records are removed. Including them as failures gives the performance a user experiences. Both should be reported.

**12. Why is a 10 s ECG not a rule-out for AF?** AF is paroxysmal. A 10 s strip samples a small window, and a normal strip does not exclude episodes at other times. Longer monitoring increases detection.

**13. What does a QRS duration of 130 ms suggest?** A conduction delay, such as bundle branch block, or a ventricular origin. Morphology is needed to distinguish.

**14. Why does the lead with the longest QT get used?** The QT varies by lead, and the longest value is the most sensitive to repolarisation prolongation. Using a shorter lead can miss a prolonged QT.

**15. What is the risk of a 40 Hz low-pass for a paced ECG?** Pacemaker spikes contain high-frequency content and can be removed, so the paced rhythm may be missed or the QRS onset misidentified.

**16. What is the difference between a 12-lead ECG and a 12-channel ECG?** A 12-lead ECG is a standardised set of twelve projections with defined electrode positions and lead equations. A 12-channel ECG is any recording with twelve channels, which may be a different montage, a different reference, or a different bandwidth. The number of channels does not define the diagnostic capability.

**17. Why is the median beat useful for interval measurement?** It reduces random noise by averaging across beats while preserving the morphology that is consistent across beats. It does not remove correlated noise such as baseline wander, and it smears morphology if ectopic beats are included.

**18. What does "unclassifiable" mean in an ECG validation?** The algorithm could not produce a classification, usually because the signal was too noisy or the rhythm was outside the model's categories. It is a third outcome, not a negative result, and it must be reported separately because it changes the sensitivity and specificity a user experiences.

**19. Why is the frontal axis computed from leads I and aVF and not from I and II?** Leads I and aVF are approximately orthogonal in the frontal plane, so they span the plane and give a unique angle. Leads I and II are separated by 60°, so the computation is possible but the noise amplification is worse. Any two independent limb leads can be used if the geometry is handled correctly.

**20. What is the difference between a PR segment and a PR interval?** The PR segment is from the end of the P wave to the start of the QRS; the PR interval is from the start of the P wave to the start of the QRS. The interval includes the P duration; the segment does not.

**Rubric for Exercise 5 (AF validation).** A strong design uses a simultaneous 12-lead over-read, patient-level splitting, a prespecified inconclusive category, and reports sensitivity and specificity with confidence intervals. A weak design uses segment-level splitting or excludes unclassifiable records without reporting them.

**Rubric for Exercise 8 (report template).** A strong template has a method field for every number, a per-measurement quality decision, and a limitations section that names the lead set and duration. A weak template lists values without method or uncertainty.

## 14. Interview questions

**1. Walk me through how you would measure QT on a digital ECG.**
A strong answer names the QRS onset, the T end, the lead with the longest QT, the tangent method, the correction formula, and the uncertainty. A weak answer says "from the start of the QRS to the end of the T wave" without a method.

**2. Why can two QTc values for the same patient differ by 40 ms?**
A strong answer names the formula, the rate, the lead, and the T-end method. A weak answer says "measurement error."

**3. How do you detect limb lead reversal?**
A strong answer uses P polarity in I and aVR, QRS polarity, and the lead equations, and names the confounders. A weak answer says "look at the axis."

**4. What can a single-lead ECG diagnose?**
A strong answer limits the claim to rate and rhythm in a narrow population and names what it cannot do. A weak answer says "arrhythmias" without qualification.

**5. How would you validate an AF detector?**
A strong answer uses a simultaneous reference, patient-level splits, an inconclusive category, and reports sensitivity and specificity. A weak answer uses a public dataset and a random split.

**6. What is the difference between a measurement and a diagnosis in ECG?**
A strong answer separates intervals and amplitudes from pattern-based clinical inferences and notes that each needs different validation. A weak answer treats the automated report as a diagnosis.

**7. Why is Bazett still used if it overcorrects?**
A strong answer explains convention, guideline familiarity, and the risk of switching formulas without revalidation. A weak answer says "it is standard."

**8. How do you handle a 12-lead record with one failed lead?**
A strong answer reports the missing lead and restricts regional claims. A weak answer interpolates the lead from neighbours.

**9. What is the effect of a 0.5 Hz high-pass on ST analysis?**
A strong answer explains attenuation of low-frequency ST content and the risk of a false-normal ST. A weak answer says "it removes baseline wander."

**10. How do you measure heart rate from a 10 s strip?**
A strong answer distinguishes beat count from mean RR, states the resolution, and excludes ectopy. A weak answer counts beats and multiplies by 6 without noting the quantization.

**11. What would make you call an ECG uninterpretable?**
A strong answer names lead-off, clipping on fiducials, non-diagnostic bandwidth, and severe artifact, and ties each to the measurement it invalidates. A weak answer says "poor quality."

**12. Why does the Apple ECG claim not extend to ST analysis?**
A strong answer cites the single lead, the validated claim, and the absence of spatial coverage. A weak answer says "the FDA did not approve it."

**13. How do you decide between Bazett and Fridericia in a report?**
A strong answer reports both when they disagree across a threshold, prefers a regression-based correction when available, and avoids rate correction in AF. A weak answer picks one because it is common.

**14. What is the first thing you check when an automated ECG report looks implausible?**
A strong answer checks lead placement and quality before the algorithm, because a reversal or a noisy lead can produce a plausible but wrong report. A weak answer retrains or adjusts the threshold.

## 15. Key terms

- **Augmented lead** — scaled limb lead aVR, aVL, or aVF.
- **Bazett correction** — QTc = QT/√RR.
- **Bundle branch block** — conduction delay producing a wide QRS.
- **Calibration** — 1 mV = 10 mm at standard gain.
- **Einthoven triangle** — the limb lead relationships I + III = II.
- **Electrode** — conductive skin contact.
- **Fiducial point** — waveform landmark used for timing.
- **Fridericia correction** — QTc = QT/RR^(1/3).
- **Frontal axis** — direction of the mean QRS vector.
- **J point** — QRS offset and ST onset.
- **Lead** — projection of cardiac activity onto an axis.
- **Low voltage** — small QRS amplitude.
- **Over-read** — cardiologist interpretation used as reference.
- **P wave** — atrial depolarisation.
- **PR interval** — P onset to QRS onset.
- **Precordial lead** — chest lead V1 to V6.
- **Premature atrial contraction** — early atrial beat.
- **Premature ventricular contraction** — early ventricular beat.
- **QRS duration** — QRS onset to J point.
- **QT interval** — QRS onset to T end.
- **QTc** — rate-corrected QT.
- **RR interval** — R peak to R peak.
- **Sinus rhythm** — rhythm from the sinoatrial node.
- **ST segment** — J point to T onset.
- **T wave** — ventricular repolarisation.
- **U wave** — late repolarisation wave.
- **Unclassifiable** — record that cannot support a diagnosis.
- **Wilson central terminal** — average of RA, LA, LL.
- **Goldberger equations** — augmented lead derivations.
- **Electrical alternans** — beat-to-beat amplitude alternation.
- **R-wave progression** — change in R/S ratio across precordial leads.
- **Ectopic beat** — beat from a non-sinus focus.
- **Aberrancy** — wide QRS from conduction delay.
- **Pre-excitation** — early activation via an accessory pathway.
- **Diagnostic bandwidth** — 0.05-150 Hz for adult diagnostic ECG.
- **Kappa** — agreement beyond chance between readers or methods.
- **ST elevation** — ST segment above baseline, measured at the J point.
- **Lead reversal** — exchange of electrode positions producing a predictable pattern.
- **Monitor bandwidth** — 0.5-40 Hz, adequate for rhythm but not for ST analysis.
- **Paper speed** — 25 mm/s at standard display.
- **Signal quality index** — composite score of record usability.
- **Standard gain** — 10 mm/mV.

## 16. Sources

All URLs accessed 25 October 2026.

1. Kligfield P, Gettes LS, Bailey JJ, et al. Recommendations for the standardization and interpretation of the electrocardiogram: Part I. *J Am Coll Cardiol*. 2007;49(10):1109-1127. https://doi.org/10.1016/j.jacc.2007.01.024
2. Mason JW, Hancock EW, Gettes LS. Part II: Electrocardiography diagnostic statement list. *Circulation*. 2007;115:1325-1332. https://doi.org/10.1161/CIRCULATIONAHA.106.180201
3. Rautaharju PM, Surawicz B, Gettes LS. Part IV: The ST segment, T and U waves, and the QT interval. *Circulation*. 2009;119:e241-e250. https://doi.org/10.1161/CIRCULATIONAHA.108.191096
4. Wagner P, Strodthoff N, Bousseljot RD, et al. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*. 2020;7:154. https://doi.org/10.1038/s41597-020-0495-6
5. PhysioNet. PTB-XL v1.0.3. https://physionet.org/content/ptb-xl/1.0.3/
6. PhysioNet. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/
7. IEC 60601-2-25:2011. https://webstore.iec.ch/en/publication/2636
8. IEC 60601-2-27, cardiac monitors. https://webstore.iec.ch/
9. IEC 60601-2-47, ambulatory electrocardiographic systems. https://webstore.iec.ch/
10. ANSI/AAMI EC11 and EC57. https://webstore.ansi.org/
11. US Food and Drug Administration. Diagnostic ECG Guidance (Including Non-Alarming ST Segment Measurement). 1998. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/diagnostic-ecg-guidance-including-non-alarming-st-segment-measurement-guidance-industry
12. US Food and Drug Administration. Arrhythmia Detector and Alarm — Class II Special Controls Guidance. 2003. https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/arrhythmia-detector-and-alarm-class-ii-special-controls-guidance-document-industry-and-fda-staff
13. 21 CFR 870.2345, Electrocardiograph software for over-the-counter use. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-870/subpart-C/section-870.2345
14. US Food and Drug Administration. DEN180044, Apple ECG App. 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
15. Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *N Engl J Med*. 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
16. PhysioNet/Computing in Cardiology Challenge 2021. https://physionet.org/content/challenge-2021/
17. PhysioNet/Computing in Cardiology Challenge 2020. https://physionet.org/content/challenge-2020/
18. ANSI/AAMI EC57, Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms. https://webstore.ansi.org/
19. US Food and Drug Administration. Apple Watch Arrhythmia Detection validation summary. https://www.apple.com/healthcare/docs/site/Apple_Watch_Arrhythmia_Detection.pdf

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_signal-fundamentals.md`](01_signal-fundamentals.md)
- Next topic: [`03_eeg.md`](03_eeg.md)
- Related topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Module survey file: [`../../course/11_physiological_signals_wearables.md`](../../course/11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/) — `11_ecg_quality.ipynb`, which covers quality assessment and R-peak detection on real records.
- Related topic: [`04_ppg-and-wearables.md`](04_ppg-and-wearables.md) for the single-lead and optical counterpart.
- Standards: [IEC 60601-2-25](https://webstore.iec.ch/en/publication/2636)
- Related topic: [`09_validation.md`](09_validation.md)
- Dataset: [`https://physionet.org/content/ptb-xl/`](https://physionet.org/content/ptb-xl/)
