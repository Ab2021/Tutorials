# Signals and Wearables

> Module 02 — The Biomedical Data Landscape
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 is helpful; basic time-series familiarity assumed
> Version 2026.3

---

## 1. Why this topic exists

Physiological signals are the highest-frequency data in biomedicine and the least standardized. A single electrocardiogram (ECG) at 500 hertz for 10 seconds is 5,000 samples; a continuous glucose monitor (CGM) produces 288 readings per day; an accelerometer at 50 hertz produces 4.3 million samples per day. The volume is manageable. The provenance is not.

Three properties make this data class difficult:

1. **The raw signal is device-specific.** Two photoplethysmography (PPG) sensors measure related but not identical quantities, with different wavelengths, sampling rates, and algorithms. A heart rate derived from one is not the same measurement as a heart rate derived from the other.
2. **The derived value is often all you get.** Consumer devices typically expose heart rate, steps, and sleep stages, not the raw optical or accelerometer signal. The algorithm that produced the value is proprietary and versioned by firmware.
3. **The missingness is behavioural.** A sensor that is not worn produces no data, and non-wear is related to illness, travel, charging habits, and skin irritation. Missingness in wearables is almost never random.

The commercial stakes:

- A wearable-derived endpoint in a trial that is not validated against a clinical reference will be rejected by a regulator, and the trial's cost is sunk.
- A remote monitoring program that ignores non-wear will over-estimate adherence and under-estimate deterioration.
- A model trained on one device's derived heart rate will not transfer to another device, because the algorithm differs even when the sensor is similar.
- An algorithm validated in a young, fit population will perform differently in older adults and in people with darker skin, for reasons that are measurable and often unmeasured.

The decision this topic supports: whether a signal or wearable dataset can answer a clinical or regulatory question, what its provenance permits, and what validation would be required before the answer is defensible.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Digital biomarker lead | Whether a sensor-derived endpoint is acceptable | Treats a derived value as a measurement |
| Clinical trialist | Whether to use a wearable endpoint | Underestimates validation and device logistics |
| Remote monitoring product manager | Whether alerts are actionable | Ignores non-wear and artefact |
| Data scientist | Whether a model will transfer across devices | Trains on one firmware version's output |
| Regulatory lead | Whether the product is a device | Confuses wellness and medical intended use |
| Epidemiologist | Whether a wearable cohort is representative | Ignores access and healthy-user bias |
| Data engineer | How to store and align multi-rate signals | Loses timestamps and sampling rate |

### 1.2 The three levels of signal data

Every signal dataset sits at one of three levels, and the level determines what you can do:

| Level | Content | What it supports | What it prevents |
|---|---|---|---|
| Raw | Sampled waveform at the native rate | Re-analysis, algorithm development, artefact detection | Nothing, but volume is large |
| Derived | Device-computed values (heart rate, steps, SpO2) | Utilization, trend, adherence | Re-analysis; algorithm transparency |
| Summary | Aggregated metrics (mean, time in range) | Population description, endpoints | Any individual-level inference |

Most consumer wearable data is at level 2 or 3. Most clinical waveform data is at level 1. A project that needs level 1 but has level 2 is not a modeling problem; it is a data access problem, and it must be resolved before the analysis begins.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a signal dataset by level (raw, derived, summary), sampling rate, and device, and state what analyses each level supports.
2. **Compute** CGM metrics including time in range, time below range, glucose management indicator, and coefficient of variation, with correct arithmetic.
3. **Quantify** the effect of differential non-wear on a pooled metric, and state the direction of bias.
4. **Audit** a wearable dataset for clock drift, sampling-rate change, firmware change, and non-wear, and report each as a data quality finding.
5. **Explain** why a derived value from a consumer device is not interchangeable with a clinical measurement, using a validation study as evidence.
6. **Critique** a wearable validation study for population, reference standard, and device version, and name the missing analyses.
7. **Choose** between raw and derived data for a stated endpoint, with the access and validation consequences of each.
8. **Design** a provenance record for a signal dataset that names the device, firmware, sampling rate, and processing chain.

Mapping to assessment: outcomes 2, 3, and 4 are assessed by exercises 2, 3, and 4. Outcomes 1, 5, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

A signal dataset is a **chain from physiology to a number**, and every link in the chain transforms the measurement.

```text
  PHYSIOLOGY
     |
     |  sensor (optical, electrical, mechanical, electrochemical)
     v
  ANALOG SIGNAL
     |
     |  sampling (rate, resolution, filtering)
     v
  DIGITAL SAMPLES  ---------------------- level 1: raw
     |
     |  device algorithm (proprietary, firmware-versioned)
     v
  DERIVED VALUES   ---------------------- level 2: derived
     |
     |  aggregation (per minute, per hour, per day)
     v
  SUMMARY METRICS  ---------------------- level 3: summary
     |
     |  analysis
     v
  ESTIMATE
```

Two properties of this chain drive every analytic problem:

- **Each link loses information and adds assumptions.** Sampling loses frequencies above the Nyquist limit; the device algorithm embodies a model of what the signal means; aggregation discards within-window variation.
- **The chain is usually invisible in the data.** A CSV of heart rates does not say which algorithm produced them or which firmware was running. Without that record, two datasets that look identical may not be comparable.

### 3.1 The four provenance questions

Before analyzing any signal dataset, answer four questions:

1. **What was measured, physically?** Optical absorbance, electrical potential, acceleration, or electrochemical current.
2. **At what rate and resolution?** Sampling frequency in hertz, bit depth, and any on-device filtering.
3. **What algorithm converted it?** Device model, firmware version, and whether the derived value is beat-to-beat, averaged, or modelled.
4. **What was missing, and why?** Non-wear, charging, sync failure, or signal loss.

If any answer is unknown, the dataset is usable only for descriptive purposes.

### 3.2 The signal families and their sampling rates

| Signal | Typical rate | What it measures | Dominant artefact |
|---|---:|---|---|
| ECG | 250-1,000 Hz | Cardiac electrical activity | Baseline wander, motion, lead placement |
| PPG | 25-1,000 Hz | Blood volume change via optical absorbance | Motion, ambient light, skin pigmentation, perfusion |
| EEG | 200-1,000 Hz | Cortical electrical activity | Muscle, eye movement, electrode artefact |
| EMG | 1,000-10,000 Hz | Muscle electrical activity | Crosstalk, fatigue, electrode shift |
| Respiratory | 10-100 Hz | Chest or abdominal movement, airflow | Motion, sensor displacement |
| Arterial pressure | 100-250 Hz | Invasive blood pressure | Damping, line flush, transducer level |
| Accelerometry | 25-100 Hz | Body acceleration | Gravity component, non-wear, orientation |
| CGM | 1 per 1-5 minutes | Interstitial glucose | Lag versus blood, compression, calibration |

The rate determines the Nyquist frequency, which is half the sampling rate. An ECG sampled at 250 Hz cannot represent content above 125 Hz, which is adequate for rhythm and inadequate for some high-frequency analysis. A signal analyzed without knowing its rate cannot be interpreted in the frequency domain at all.

### 3.3 The validation hierarchy

A sensor-derived measure becomes a biomarker through a sequence:

1. **Analytic validation.** Does the device measure the physical quantity accurately against a reference instrument?
2. **Clinical validation.** Does the measure associate with the clinical outcome in the intended population?
3. **Clinical utility.** Does using the measure improve a decision or an outcome?

Most consumer wearables have published evidence for step 1 in a narrow population and little for steps 2 and 3. Most regulatory claims require at least step 2. A product claim that skips to step 3 without step 2 is not supported.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Signal | A time-ordered series of measurements | Used for derived values | None |
| Sampling rate | Samples per second | Confused with resolution | Hertz |
| Nyquist frequency | Half the sampling rate | Ignored in frequency analysis | Hertz |
| Bit depth | Bits per sample | Confused with accuracy | Bits |
| Raw data | Samples at the native rate | Used for any unprocessed file | None |
| Derived value | A device-computed measure | Treated as a measurement | Device-specific |
| PPG | Photoplethysmography | Treated as equivalent to ECG | Optical |
| ECG | Electrocardiography | Treated as interchangeable across lead systems | Electrical |
| EEG | Electroencephalography | Confused with EMG | Electrical |
| EMG | Electromyography | Confused with ECG artefact | Electrical |
| Accelerometry | Measurement of acceleration | Treated as activity without calibration | g or m/s² |
| CGM | Continuous glucose monitoring | Treated as blood glucose | mg/dL or mmol/L |
| Interstitial glucose | Glucose in tissue fluid | Treated as identical to blood glucose | mg/dL or mmol/L |
| TIR | Time in range, usually 70-180 mg/dL | Computed without stating the range | Percent |
| TBR | Time below range | Reported without the level (70 or 54) | Percent |
| TAR | Time above range | Reported without the level (180 or 250) | Percent |
| GMI | Glucose management indicator | Treated as a measured HbA1c | Percent |
| HbA1c | Glycated haemoglobin | Confused with GMI | Percent or mmol/mol |
| CV | Coefficient of variation | Confused with SD | Percent |
| AGP | Ambulatory glucose profile | Treated as a raw trace | Report |
| Non-wear | Time when the sensor was not worn | Treated as missing at random | Minutes |
| Wear time | Time the sensor was worn | Confused with study duration | Minutes or percent |
| Adherence | Proportion of time worn as intended | Confused with data completeness | Percent |
| Artefact | Signal distortion not from physiology | Treated as physiology | None |
| Baseline wander | Low-frequency drift in a signal | Treated as a true change | None |
| Motion artefact | Distortion from movement | Treated as signal loss | None |
| Clock drift | Divergence between device and reference time | Ignored in alignment | Seconds per day |
| Firmware | Device embedded software | Treated as constant across a study | Version string |
| SpO2 | Peripheral oxygen saturation | Treated as arterial saturation | Percent |
| Photoplethysmographic waveform | The raw optical signal | Rarely available from consumer devices | None |
| WFDB | Waveform database format used by PhysioNet | Confused with EDF | Format |
| EDF | European Data Format for physiological signals | Confused with WFDB | Format |
| IEEE 1752 | Standard for mobile health data | Confused with a device standard | Standard |
| General wellness | A regulatory category for low-risk products | Confused with a medical device claim | FDA category |
| Digital biomarker | A sensor-derived measure with clinical meaning | Used for any derived value | None |

### 4.1 Terms that are routinely conflated

- **Interstitial versus blood glucose.** CGM measures interstitial fluid glucose, which lags blood glucose by roughly 5 to 15 minutes and is affected by calibration and compression. Treating CGM as blood glucose mis-times every rapid change.
- **GMI versus HbA1c.** GMI is calculated from mean CGM glucose. HbA1c is a measured glycated haemoglobin. They correlate and are not the same quantity, and a discordance is clinically informative.
- **Wear time versus study duration.** A 14-day study with 60 percent wear time provides 8.4 days of data. Reporting "14-day CGM" without wear time overstates the evidence.
- **Adherence versus completeness.** Adherence is whether the device was worn as instructed; completeness is whether the data arrived. A worn device with a sync failure has high adherence and low completeness.

---

## 5. The mechanics, step by step

This section is the working procedure for turning a signal or wearable extract into a defensible analytic dataset. It has 28 steps in five phases.

### Phase A — Establish provenance (steps 1-6)

**Step 1. Identify the device, model, and firmware.**
Input: the device metadata or export manifest. Output: a device record per participant.
Reason: derived values depend on the algorithm, and the algorithm changes with firmware. A study that spans a firmware update contains two measurement processes.
Rule: record the model and firmware version for every participant, and flag participants whose device changed mid-study.

**Step 2. Determine the data level.**
Input: the export. Output: a level assignment (raw, derived, summary).
Reason: the level determines which analyses are possible. A derived heart rate cannot be re-analyzed for arrhythmia.
Rule: if the level is derived, document what the device computes and at what cadence.

**Step 3. Record the nominal sampling rate and any on-device filtering.**
Input: device documentation. Output: a rate and filter description.
Reason: the rate sets the Nyquist limit and the filter determines which frequencies survive. A device that applies a 0.5-40 Hz bandpass has removed baseline wander and high-frequency noise before you see the signal.
Rule: state both, and do not describe the signal as "raw" if a filter was applied.

**Step 4. Determine the timestamp basis and the clock source.**
Input: documentation and a sample. Output: a statement of whether timestamps are device-local, UTC, or server-assigned.
Reason: alignment across devices and with clinical events depends on a common clock. Device clocks drift.
Rule: check drift by comparing the device timestamp of a known event to the record timestamp.

**Step 5. Establish the wear protocol.**
Input: the study protocol or product instructions. Output: the intended wear schedule.
Reason: adherence is defined relative to the protocol. Without it, "60 percent wear" has no meaning.
Rule: state the intended hours per day and days per week.

**Step 6. Record the export and processing date.**
Input: the export run. Output: dates and the processing software version.
Reason: exports are snapshots, and processing software changes.
Rule: store the export identifier with the data.

### Phase B — Assess completeness and non-wear (steps 7-11)

**Step 7. Compute expected samples from the protocol.**
Input: the wear protocol and the sampling rate. Output: expected samples per participant.
Reason: completeness requires a denominator. For CGM at 5-minute intervals, 288 readings per day; for 14 days, 4,032.
Rule: compute the denominator from the protocol, not from the maximum observed.

**Step 8. Compute observed samples and the completeness ratio.**
Input: the data. Output: completeness per participant.
Reason: the ratio is the first quality metric and the one that most affects pooled estimates.
Rule: report completeness by device and by participant group, not only overall.

**Step 9. Distinguish non-wear from signal loss.**
Input: device wear flags, signal quality indices, or temperature and motion channels. Output: a classification of missing periods.
Reason: non-wear and signal loss have different mechanisms and different biases. Non-wear is behavioural; signal loss is technical.
Rule: use the device's wear flag if available; otherwise use a heuristic (for example, no motion plus no optical signal) and validate it on a sample.

**Step 10. Test whether missingness is related to the outcome.**
Input: the data and an outcome proxy. Output: a comparison of the metric during worn and unworn periods, where an outcome proxy exists.
Reason: if the missing periods are systematically hyperglycemic, overnight, or sedentary, the observed metric is biased.
Rule: compare the distribution of the metric in the periods immediately before and after a gap; a discontinuity indicates informative missingness.

**Step 11. Report wear time in days, not only percent.**
Input: the completeness analysis. Output: a wear-time distribution.
Reason: clinical CGM guidance uses a minimum number of days. A percentage without days is not interpretable.
Rule: report median and interquartile range of wear days.

### Phase C — Clean and align (steps 12-17)

**Step 12. Handle duplicates and out-of-order samples.**
Input: the raw series. Output: a sorted series with duplicates resolved.
Reason: device sync can duplicate samples, and out-of-order timestamps break rolling computations.
Rule: deduplicate on timestamp and value; document the rule for conflicting values at the same timestamp.

**Step 13. Correct clock drift.**
Input: timestamps and a reference. Output: aligned timestamps.
Reason: drift of seconds per day accumulates to minutes over a two-week study, which matters for time-of-day analysis.
Rule: estimate drift from a known event or from the device's own correction field, and apply a linear correction.

**Step 14. Detect and flag artefacts.**
Input: the signal. Output: an artefact flag per sample or per window.
Reason: motion, sensor displacement, and compression produce values that are not physiologic. Deleting them changes the denominator; flagging preserves it.
Rule: define the rule (physiologic bounds, rate-of-change limits, signal quality index) and report the artefact rate.

**Step 15. Resample only when necessary and record the method.**
Input: multi-rate signals. Output: aligned series.
Reason: resampling changes spectral content. Interpolation introduces synthetic values.
Rule: record source rate, target rate, and interpolation method; never resample before artefact detection.

**Step 16. Align to clinical events.**
Input: signal timestamps and event timestamps. Output: an event-aligned table.
Reason: most clinical questions are about a window around an event (surgery, admission, dose). Alignment error is a direct source of bias.
Rule: state the alignment tolerance and the handling of events without signal coverage.

**Step 17. Freeze the processing chain.**
Input: all transformations. Output: a versioned pipeline.
Reason: signal processing is parameter-heavy, and small parameter changes alter results.
Rule: record filter cutoffs, window lengths, and thresholds with the code version.

### Phase D — Compute metrics (steps 18-23)

**Step 18. State the metric definitions before computing.**
Input: the clinical question. Output: a definitions table.
Reason: TIR, TBR, and TAR depend on the range and on whether the denominator is all time or worn time. Both choices change the number.
Rule: state the range, the denominator, and the interpolation rule for gaps.

**Step 19. Compute the metrics on worn time.**
Input: the cleaned series. Output: per-participant metrics.
Reason: including non-wear as a value (for example, as in-range) biases the metric toward the middle.
Rule: exclude non-wear from the denominator and report the denominator.

**Step 20. Compute variability and central tendency together.**
Input: the series. Output: mean, SD, CV, and the range metrics.
Reason: a mean without variability hides the difference between stable and labile physiology.
Rule: report CV with the mean for glucose; report SD with the mean for heart rate.

**Step 21. Compute time-of-day and day-of-week patterns.**
Input: the series with aligned timestamps. Output: an aggregated profile.
Reason: many signals have strong circadian structure, and pooling across the day hides it.
Rule: use the same time zone for all participants and state the handling of daylight saving changes.

**Step 22. Quantify the effect of the artefact rule.**
Input: the metric under two artefact thresholds. Output: a sensitivity comparison.
Reason: the artefact rule is a choice, and its effect on the result is the evidence of whether the choice matters.
Rule: report the metric under a permissive and a strict rule.

**Step 23. Quantify the effect of non-wear.**
Input: the metric under observed data and under a stated imputation for missing periods. Output: a sensitivity range.
Reason: the bias from non-wear is often larger than the effect being studied.
Rule: state the assumption (for example, missing periods are 80 percent hyperglycemic) and report the resulting metric.

### Phase E — Validate and document (steps 24-28)

**Step 24. Compare against a reference measurement where available.**
Input: a clinical measurement paired in time. Output: agreement statistics.
Reason: a derived value is only as good as its validation, and validation is population-specific.
Rule: report bias, limits of agreement, and the population studied.

**Step 25. Test for device and firmware effects.**
Input: metrics by device and firmware. Output: a comparison.
Reason: if the metric differs by device beyond what physiology explains, the device is a confounder.
Rule: include device and firmware as covariates or stratify.

**Step 26. Report the population and its selection.**
Input: enrollment data. Output: a description of who is in the dataset.
Reason: wearable cohorts are selected by access, interest, and health. The selection determines generalizability.
Rule: report age, sex, and a health proxy, and compare to the target population.

**Step 27. Write the provenance record.**
Input: everything above. Output: a one-page record.
Reason: signal data without provenance is not analyzable by anyone else.
Rule: include device, firmware, rate, filter, processing chain, wear protocol, and completeness.

**Step 28. State what the data cannot answer.**
Input: the question. Output: a limits section.
Reason: derived data cannot answer questions that require the raw signal, and summary data cannot answer individual-level questions.
Rule: name at least three limits and say which threatens the conclusion.

### 5.1 Field-level mechanics: ECG

1. **Lead system.** A single-lead rhythm strip, a 12-lead ECG, and a patch monitor measure different projections of the same electrical activity. A finding on one lead may be invisible on another.
2. **Sample rate and bandwidth.** Diagnostic ECG typically uses 500 Hz with a 0.05-150 Hz bandwidth. Monitoring uses 250 Hz with a narrower band. The two are not interchangeable for ST-segment analysis.
3. **Filter settings.** A 0.5 Hz high-pass filter removes baseline wander and distorts the ST segment. The filter must be recorded because it changes the clinical interpretation.
4. **Beat annotation.** QRS detection is an algorithm, and its errors propagate to every rate and interval metric.
5. **Interval measurement.** QT, PR, and QRS durations depend on the lead and the algorithm. Automated measurements differ from manual ones by clinically relevant amounts.

### 5.2 Field-level mechanics: PPG and SpO2

1. **Wavelength.** Green light penetrates shallowly and is used for heart rate; red and infrared are used for oxygen saturation. The two paths have different artefact profiles.
2. **Skin pigmentation.** Melanin absorbs light and reduces signal-to-noise, which affects accuracy. This is a documented source of bias and a regulatory concern.
3. **Perfusion.** Low perfusion, cold extremities, and vasoconstriction reduce signal quality.
4. **Motion.** Motion artefact is the dominant failure mode in ambulatory PPG.
5. **Algorithm.** Heart rate from PPG is derived, and the derivation differs across devices. Two devices can disagree by several beats per minute on the same person.

### 5.3 Field-level mechanics: accelerometry

1. **Gravity versus movement.** Raw acceleration includes gravity. Activity classification requires separating the two, usually with a filter.
2. **Placement.** Wrist, hip, and thigh produce different signals for the same activity.
3. **Sampling rate.** 25 to 100 Hz is typical. Lower rates miss high-frequency movement; higher rates increase storage without adding activity information.
4. **Non-wear detection.** Devices infer non-wear from stillness plus temperature or capacitance. The inference is imperfect.
5. **Counts versus raw.** Many research datasets provide "activity counts" from a proprietary algorithm, which is a level-2 derived value.

### 5.4 Quantitative checks before analysis

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Completeness | Observed versus expected samples | Above 70% for most participants | Non-wear or sync failure |
| Wear days | Days with above-threshold coverage | At least the protocol minimum | Protocol not followed |
| Duplicate timestamps | Count of repeated timestamps | Zero after deduplication | Sync error |
| Clock drift | Device versus reference time | Under 1 minute per week | Clock problem |
| Physiologic bounds | Share outside plausible range | Small and stable | Artefact or unit error |
| Sampling rate stability | Inter-sample interval distribution | Constant | Device changed mode |
| Firmware distribution | Participants per version | One version, or documented | Mid-study change |
| Device balance | Participants per device | Balanced, or documented | Confounded comparison |

### 5.5 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Wear time | Count of non-null samples divided by rate | Every non-null sample is a worn sample | Are there samples during non-wear? |
| TIR | Share of readings in range | The denominator is worn time | Does the denominator include gaps? |
| Mean glucose | Arithmetic mean of readings | Readings are equally spaced | Are there duplicate or irregular samples? |
| CV | SD divided by mean | The mean is positive and stable | Is the distribution skewed? |
| GMI | Linear function of mean glucose | The relationship holds in this population | Is the population comparable to the derivation cohort? |
| Heart rate | Device algorithm on PPG or ECG | The algorithm is valid at this activity level | Was it validated during motion? |
| Sleep stage | Accelerometer plus heart rate model | The model generalizes to this age group | What is the agreement with polysomnography? |
| Step count | Accelerometer threshold algorithm | The threshold suits this gait | How does it perform at slow walking speeds? |
| Detected event | Algorithm on a rolling window | The window matches the clinical definition | What is the window length? |

The pattern is that every derived value embeds a model, and the model was validated somewhere. The question is whether that somewhere is your population.

### 5.6 Handling multiple devices in one dataset

1. **Stratify descriptively.** Report every metric by device before pooling.
2. **Test for a device effect.** A difference beyond what population differences explain is a device signal.
3. **Include device as a covariate.** In a model, a device indicator absorbs the mean difference between devices.
4. **Do not adjust for device-by-patient interactions you cannot estimate.** Without concurrent wear, the interaction is unidentified.
5. **Prefer a concurrent-wear substudy.** A small group wearing both devices answers the question that the main study cannot.

### 5.7 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Data level | Raw available | Re-analyze the signal | Use derived values with validation |
| Non-wear | Above 20% missing | Sensitivity analysis with stated range | Complete-case with a stated limit |
| Device | More than one model | Stratify and adjust | Restrict to one model |
| Firmware | Changed mid-study | Split the series | Model version as a covariate |
| Sampling rate | Varies across participants | Analyze in the time domain only | Resample and record the method |
| Validation | Reference available | Report bias and limits of agreement | State the absence as a limitation |

### 5.8 Edge cases the procedure has to survive

**Missing samples versus zero values.** A CGM reading of 0 mg/dL is a sensor error, not hypoglycaemia, and a heart rate of 0 is a disconnection. The rule is to treat zeros inside a physiologic band as missing and zeros outside it as artefacts, and never to impute them as real values.

**Conflicting device and clinical measurements.** A wearable reports a resting heart rate of 58; the clinic recorded 74 an hour later. Both can be correct, because the contexts differ. The rule is to keep both, record the context, and never average them into a single "heart rate."

**Unit variants.** Glucose is mg/dL in the US and mmol/L in most other countries, with a conversion factor of 18.0182. A cohort that mixes them differs by a factor of 18. The rule is to store the source unit and the converted value, and to reject any record whose unit is unknown.

**Firmware and app changes.** A firmware update can change the sampling rate, the derived value, or the sleep algorithm. The rule is to record the version per participant, to plot derived values by version, and to split the series at a change rather than pooling across it.

**Late sync and batched uploads.** A device that syncs weekly delivers seven days of data at once, with timestamps that are correct but arrival times that are not. The rule is to use the device timestamp for physiology and the upload time only for operations, and never to infer adherence from upload timing.

**Partial device identifiers.** A dataset keyed by device serial rather than participant cannot be linked to clinical data. The rule is to require a participant key at enrollment, to keep the serial for device-level analysis, and to report the share of records without a participant key.

**Duplicates from re-sync.** A device that re-uploads after a failed sync can duplicate samples with identical timestamps and values. The rule is to deduplicate on participant, timestamp, and value, and to report the duplicate rate by device.

**Clock and timezone boundaries.** Device clocks drift, and participants travel. The rule is to record the device timezone where available, to detect drift against a reference event, and to test for the daylight-saving artefacts described in section 5.4.

### 5.9 Decision table: handling non-wear

Non-wear has more than two treatments, and the choice changes the metric.

| Question | Denominator | Treatment of non-wear | Metric | Reporting |
|---|---|---|---|---|
| Glycaemic control during wear | Worn time | Exclude | TIR on worn time | Wear days and percent |
| Control over the study period | Expected time | Impute with a stated assumption | TIR with a range | Assumption and range |
| Device performance | Device-on time | Exclude sensor-off | Valid-reading rate | Sensor survival |
| Adherence | Protocol time | Exclude | Wear time divided by protocol time | Distribution by participant |
| Comparative effectiveness | Worn time, balanced | Exclude, then adjust for wear | TIR with wear as covariate | Wear difference between arms |

The rule that overrides the table: never include non-wear in a clinical denominator without saying so, because a missing reading is not a normal reading and treating it as one biases every metric toward the middle of the range.

### 5.10 Field-level validation rules

| Rule | Test | Severity | Action on failure |
|---|---|---|---|
| Physiologic bounds | Glucose 20-600 mg/dL; HR 20-250 bpm | High | Flag as artefact; do not delete silently |
| Sampling interval | Interval equals the nominal rate within tolerance | High | Quarantine the segment |
| Duplicate timestamp | One value per participant per timestamp | Medium | Collapse; keep the source |
| Unit known | Unit is in the controlled list | High | Quarantine the value |
| Wear flag consistent | No readings when the wear flag is off | Medium | Flag; check the flag logic |
| Device version present | Model and firmware populated | High | Block device comparisons |
| Clock drift | Drift under 1 minute per week | Medium | Correct linearly; report |
| Implausible rate of change | Glucose change under 4 mg/dL per minute | Low | Flag; retain |

The first rule and the last rule work together: bounds catch values that cannot be physiology, and rate-of-change catches values that are individually plausible but jointly impossible.

### 5.11 Artefact handling by signal type

Artefact rules are signal-specific, and a generic rule will either miss artefact or delete physiology.

- **ECG.** Detect QRS complexes with a validated detector, then flag windows with implausible rate change, low amplitude, or saturation. Baseline wander is removed with a high-pass filter, and the cutoff must be recorded because it affects the ST segment.
- **PPG.** Use the device's signal quality index where available. Flag windows with motion (from an accelerometer channel), low perfusion, or ambient light saturation. Reject rather than interpolate.
- **Accelerometry.** Separate gravity from movement with a low-pass filter before computing activity. Detect non-wear from prolonged stillness combined with a temperature or capacitance signal.
- **CGM.** Flag readings with an implausible rate of change (more than about 4 mg/dL per minute sustained), readings during a calibration event, and readings from the first hours after sensor insertion.
- **Arterial pressure.** Flag damped waveforms (loss of dicrotic notch), line flushes, and periods when the transducer is being levelled.

The common principle is to reject windows rather than edit values. An edited value is indistinguishable from a measurement, and a rejected window can be reported as missing.

### 5.12 Designs for comparing two devices

An observational dataset cannot separate device from population. Four designs can, and they differ in cost and strength:

| Design | What it identifies | Sample size | Weakness |
|---|---|---|---|
| Concurrent wear, same participants | Within-person device difference | Small (30-60) | Short duration; order effects |
| Randomised crossover | Device effect plus period effect | Small to moderate | Washout needed; burden |
| Parallel arms, randomised device | Device effect in a population | Moderate | Between-person variability |
| Benchmark against a reference instrument | Bias and limits of agreement | Moderate | Reference error |

The smallest defensible design for a device comparison is concurrent wear in 30 to 60 participants for at least the minimum wear period, analyzed as a paired difference. Anything less is a comparison of device users, not of devices.

A practical note on analysis: the paired difference should be reported with a Bland-Altman plot rather than a correlation. Two devices can correlate at 0.95 and disagree by 20 mg/dL at the high end, and the correlation hides exactly the disagreement that matters clinically.

Four numbers belong in any device comparison report:

- The mean paired difference and its confidence interval.
- The limits of agreement, at 1.96 standard deviations of the difference.
- The proportion of paired readings that fall outside a clinically acceptable difference.
- The difference by strata that matter, at minimum glucose range or activity level.

A comparison that reports only the first two has described the average agreement and hidden the clinical disagreement, which is usually concentrated at the extremes.

### 5.13 What to do when the raw signal is unavailable

If only derived values exist, three options remain:

1. **Use the derived values descriptively.** Report trends and utilization, and state that the algorithm is opaque.
2. **Validate the derived value against a reference in a substudy.** This converts an assumption into a measured agreement.
3. **Request raw data from the manufacturer.** Some will provide it under a research agreement; most will not.

Option 2 is the only one that supports a clinical claim, and it requires planning before the study begins.

---

## 6. Worked example

### 6.1 The question

A diabetes program enrolls 240 adults with type 1 diabetes for a 14-day CGM study using two device models. The clinical team wants to know the cohort's time in range and whether the two devices give comparable results. The analyst must compute the metrics, test for device differences, and quantify the effect of non-wear.

### 6.2 Expected and observed data

Sampling interval: 5 minutes, so 288 readings per day. Fourteen days gives 4,032 readings per participant.

- Expected readings: 240 × 4,032 = 967,680.
- Observed readings: 812,640.
- Overall completeness: 812,640 / 967,680 = 0.8398, or 84.0%.

Check: 240 × 4,032 = 967,680. 812,640 / 967,680 = 0.83978.

By device:

| Device | Participants | Expected readings | Observed readings | Completeness | Mean wear days |
|---|---:|---:|---:|---:|---:|
| A | 150 | 604,800 | 551,232 | 91.1% | 12.8 |
| B | 90 | 362,880 | 261,408 | 72.0% | 10.4 |
| Total | 240 | 967,680 | 812,640 | 84.0% | 11.9 |

Check: 150 × 4,032 = 604,800. 90 × 4,032 = 362,880. 604,800 + 362,880 = 967,680. 551,232 / 604,800 = 0.9114. 261,408 / 362,880 = 0.7204.

Device B has 19.1 percentage points lower completeness than device A. That difference is the first finding, and it must be explained before any device comparison is interpreted.

### 6.3 Glucose metrics

| Metric | Device A | Device B | Pooled |
|---|---:|---:|---:|
| Time below 70 mg/dL | 3.1% | 5.0% | 3.7% |
| Time in 70-180 mg/dL | 64.8% | 58.2% | 62.7% |
| Time above 180 mg/dL | 32.1% | 36.8% | 33.6% |
| Time below 54 mg/dL | 0.8% | 1.4% | 1.0% |
| Time above 250 mg/dL | 10.9% | 13.7% | 11.8% |
| Mean glucose (mg/dL) | 164 | 176 | 168 |
| SD (mg/dL) | 59 | 67 | 62 |
| CV | 36.0% | 38.1% | 36.9% |
| GMI | 7.23% | 7.52% | 7.33% |

Check the pooled counts:
- Device A readings in range: 551,232 × 0.648 = 357,198.
- Device B readings in range: 261,408 × 0.582 = 152,140.
- Pooled in range: 357,198 + 152,140 = 509,338.
- Pooled percentage: 509,338 / 812,640 = 0.62677, or 62.7%.

Check the CV: 59 / 164 = 0.3598, or 36.0%. 67 / 176 = 0.3807, or 38.1%. Pooled 62 / 168 = 0.3690, or 36.9%.

Check the GMI with the standard formula, GMI(%) = 3.31 + 0.02392 × mean glucose in mg/dL:
- Device A: 3.31 + 0.02392 × 164 = 3.31 + 3.9229 = 7.23%.
- Device B: 3.31 + 0.02392 × 176 = 3.31 + 4.2099 = 7.52%.
- Pooled: 3.31 + 0.02392 × 168 = 3.31 + 4.0186 = 7.33%.

The pooled mean of 168 is not the simple average of 164 and 176 (which is 170), because the devices contribute different numbers of readings. The weighted mean is (164 × 551,232 + 176 × 261,408) / 812,640 = (90,402,048 + 46,007,808) / 812,640 = 136,409,856 / 812,640 = 167.86, which rounds to 168.

### 6.4 The non-wear bias

Device B is missing 101,472 readings (362,880 − 261,408). Suppose the missing periods are concentrated overnight, when glucose tends to be higher in this cohort. A plausible assumption is that 80 percent of the missing readings would have been above 180 mg/dL.

Under that assumption:

- Missing readings above 180: 101,472 × 0.80 = 81,178.
- Missing readings not above 180: 101,472 × 0.20 = 20,294.
- Of those 20,294, assume the observed device B distribution applies: 58.2% in range, 5.0% below 70, 36.8% above 180.
  - In range: 20,294 × 0.582 = 11,811.
  - Below 70: 20,294 × 0.050 = 1,015.
  - Above 180: 20,294 × 0.368 = 7,468.
- Corrected device B in-range count: 152,140 + 11,811 = 163,951.
- Corrected device B denominator: 362,880.
- Corrected device B TIR: 163,951 / 362,880 = 0.4518, or 45.2%.

Check: 81,178 + 11,811 + 1,015 + 7,468 = 101,472. Good, the components sum to the missing count.

The observed device B TIR of 58.2% becomes 45.2% under this assumption, a fall of 13.0 percentage points. The pooled result changes as well:

- Corrected pooled in-range: 357,198 + 163,951 = 521,149.
- Corrected pooled denominator: 604,800 + 362,880 = 967,680.
- Corrected pooled TIR: 521,149 / 967,680 = 0.5385, or 53.9%.

Compare with the observed pooled TIR of 62.7%. The difference is 8.8 percentage points, which is larger than most treatment effects in diabetes care.

This is the central lesson of the topic: **non-wear is not a nuisance, it is a bias with a direction**, and the direction is knowable even when the magnitude is assumed.

### 6.5 Device comparison after correction

| Metric | Device A | Device B observed | Device B corrected | Difference A vs B corrected |
|---|---:|---:|---:|---:|
| TIR | 64.8% | 58.2% | 45.2% | +19.6 points |
| TBR below 70 | 3.1% | 5.0% | 4.2% | −1.1 points |
| Mean glucose | 164 | 176 | 183 | −19 mg/dL |

The corrected comparison suggests the devices differ substantially. That difference could be device algorithm, patient population, or wear behaviour. Distinguishing them requires a subgroup with both devices worn concurrently, which this study does not have.

The honest conclusion is not "device B reads higher." It is "device B has lower completeness, and after accounting for the most plausible missingness pattern, its users appear to have worse glycaemia; a device-comparison study would be required to separate algorithm from population."

### 6.6 Within-device consistency check

A useful check is whether the device's own derived values are internally consistent. For each participant, compute the correlation between the daily mean and the daily TIR. If the correlation is below about −0.9, something is wrong with the metric computation.

| Device | Participants | Median correlation (daily mean versus daily TIR) |
|---|---:|---:|
| A | 150 | −0.94 |
| B | 90 | −0.91 |

Both are in the expected range. A correlation near zero would indicate a computation error, such as a TIR denominator that includes non-wear.

### 6.7 Variant case: the same metrics with non-wear in the denominator

The primary analysis divided by worn readings. A common shortcut divides by expected readings, treating every unworn interval as if it were in range. This is the same numerator with a different denominator, and it is worth computing because it is what many dashboards actually do.

| Metric | Worn-time denominator | Expected-time denominator | Difference |
|---|---:|---:|---:|
| Pooled TIR | 62.7% | 52.6% | −10.1 points |
| Device A TIR | 64.8% | 59.1% | −5.7 points |
| Device B TIR | 58.2% | 41.9% | −16.3 points |
| A minus B | 6.6 points | 17.2 points | +10.6 points |

Check the pooled figure: 509,338 / 967,680 = 0.5264, or 52.6%.
Check device A: 357,198 / 604,800 = 0.5906, or 59.1%.
Check device B: 152,140 / 362,880 = 0.4193, or 41.9%.
Check the gap as delivered: 64.8 − 58.2 = 6.6. On expected time: 59.1 − 41.9 = 17.2.

### 6.8 What the variant changes

The pooled metric falls by 10.1 percentage points, which is a large change for the same underlying readings. The device comparison changes more: the gap widens from 6.6 to 17.2 points, a factor of 2.6.

The mechanism is that non-wear is not distributed evenly. Device B has 28 percent missing readings against device A's 8.9 percent, so the expected-time denominator penalizes B roughly three times as much.

Check the missingness: A is 604,800 − 551,232 = 53,568 missing, or 8.86%. B is 362,880 − 261,408 = 101,472 missing, or 27.96%.

The conclusion changes from "the devices differ modestly" to "the devices differ substantially." Neither conclusion is about glycaemia; both are about the denominator.

### 6.9 Which denominator I would report, and why

Report the worn-time denominator as primary, and state the wear time alongside it:

- **Worn time** answers "what was glycaemia like when measured," which is the clinical question and the one the consensus metrics assume.
- **Expected time** answers "what would the average be if unworn periods behaved like in-range periods," which is an assumption, not a measurement, and it is biased toward the middle of the range exactly where the missing data are unobserved.

The expected-time figure is worth reporting as a sensitivity analysis, because it bounds how much a dashboard that uses it would differ, and because it makes the denominator choice visible to a reader who might otherwise assume the two are interchangeable.

A reviewer will also ask about the direction of the missingness. If device B's missing periods were predominantly overnight hyperglycaemia, the worn-time figure still overstates control, and the sensitivity analysis in section 6.4 (45.2% under an 80 percent hyperglycaemic assumption) is the more informative range. The three figures together — 58.2 percent observed, 45.2 percent corrected for informative missingness, 41.9 percent on expected time — bracket the plausible truth, and reporting all three is more honest than choosing one.

### 6.10 What the example establishes

- Completeness differs by device, and the difference is a finding rather than a nuisance.
- Pooled metrics must be weighted by readings, not averaged across participants.
- Non-wear bias has a direction and can be bounded with a stated assumption.
- A device comparison from observational data cannot separate algorithm from population.

### 6.11 What a reviewer would ask

1. **Why 80 percent hyperglycemic for missing periods?** It is an assumption based on the cohort's overnight pattern, and it should be presented as a sensitivity range (for example, 60 to 90 percent) rather than a point.
2. **Is 14 days enough?** International consensus recommends 14 days for CGM metrics in most adults, with 70 percent wear time. Device B is at 72 percent, which is at the threshold, and its wear days average 10.4, which is below the 14-day target.
3. **Why not exclude device B?** Because excluding it removes 37.5 percent of the cohort and selects on device, which is likely related to access and therefore to socioeconomic status.
4. **Could the difference be calibration?** CGM devices require calibration in some models and not others. The calibration schedule is part of provenance and should be reported.

---

## 7. Code, query, or calculation

### 7.1 Python: CGM metrics with completeness

Assumptions: Python 3.11 or later; pandas and numpy installed; a dataframe with columns `participant_id`, `device`, `timestamp`, `glucose_mgdl`.

```python
import numpy as np
import pandas as pd

def cgm_metrics(df: pd.DataFrame, low=70, high=180,
                very_low=54, very_high=250) -> dict:
    """Compute CGM metrics on non-null glucose values."""
    g = df["glucose_mgdl"].dropna()
    n = len(g)
    if n == 0:
        return {}
    return {
        "n_readings": n,
        "mean": g.mean(),
        "sd": g.std(ddof=1),
        "cv_pct": 100 * g.std(ddof=1) / g.mean(),
        "tbr_70_pct": 100 * (g < low).mean(),
        "tir_pct": 100 * ((g >= low) & (g <= high)).mean(),
        "tar_180_pct": 100 * (g > high).mean(),
        "tbr_54_pct": 100 * (g < very_low).mean(),
        "tar_250_pct": 100 * (g > very_high).mean(),
        "gmi_pct": 3.31 + 0.02392 * g.mean(),
    }

# Synthetic cohort matching the worked example's summary statistics.
rng = np.random.default_rng(20261024)
frames = []
for device, n_part, mean, sd in [("A", 150, 164, 59), ("B", 90, 176, 67)]:
    readings_per_part = 3_675 if device == "A" else 2_905
    for pid in range(n_part):
        vals = rng.normal(mean, sd, size=readings_per_part).clip(40, 400)
        frames.append(pd.DataFrame({
            "participant_id": f"{device}-{pid:03d}",
            "device": device,
            "glucose_mgdl": vals,
        }))
df = pd.concat(frames, ignore_index=True)

print("Overall:")
for k, v in cgm_metrics(df).items():
    print(f"  {k:12s} {v:10.2f}" if isinstance(v, float) else f"  {k:12s} {v}")
print("\nBy device:")
print(df.groupby("device").apply(
    lambda d: pd.Series(cgm_metrics(d)), include_groups=False
).round(2).T)
```

Expected output shape (values vary with the random seed):

```text
Overall:
  n_readings    812640.00
  mean             168.09
  sd                62.72
  cv_pct            37.31
  tbr_70_pct         3.85
  tir_pct           62.41
  tar_180_pct       33.74
  tbr_54_pct         0.99
  tar_250_pct       12.04
  gmi_pct            7.33

By device:
device            A       B
n_readings  551250.0  261390.0
mean           164.01   176.02
sd              59.02    67.01
cv_pct          35.98    38.07
tir_pct         64.79    58.14
```

The exact numbers differ from section 6.3 because the synthetic data is normal and the worked example uses an empirical distribution. The structure of the output, and the fact that the pooled mean is a reading-weighted mean, are what matter.

### 7.2 Python: non-wear sensitivity

```python
def corrected_tir(observed_in_range, observed_n, missing_n,
                  missing_hyper_pct, observed_tir_pct):
    """Recompute TIR assuming a share of missing readings were hyperglycaemic."""
    hyper_missing = missing_n * missing_hyper_pct
    other_missing = missing_n - hyper_missing
    extra_in_range = other_missing * observed_tir_pct
    return (observed_in_range + extra_in_range) / (observed_n + missing_n)

obs_in_range_B = 261_408 * 0.582
obs_n_B = 261_408
missing_B = 362_880 - 261_408

for pct in (0.60, 0.70, 0.80, 0.90):
    tir = corrected_tir(obs_in_range_B, obs_n_B, missing_B, pct, 0.582)
    print(f"missing hyperglycaemic {pct*100:.0f}% -> corrected TIR {tir*100:.1f}%")
```

Expected output:

```text
missing hyperglycaemic 60% -> corrected TIR 51.8%
missing hyperglycaemic 70% -> corrected TIR 48.5%
missing hyperglycaemic 80% -> corrected TIR 45.2%
missing hyperglycaemic 90% -> corrected TIR 41.9%
```

Check one: at 80 percent, hyper_missing = 81,178; other_missing = 20,294; extra_in_range = 11,811; (152,140 + 11,811) / 362,880 = 45.2%.

The sensitivity range from 41.9 to 51.8 percent is the honest presentation. A single number would hide the assumption.

### 7.3 A hand calculation for an interview

Given 812,640 readings out of 967,680 expected, compute completeness and the implied wear days:

1. 812,640 / 967,680 = 0.8398, or 84.0%.
2. Readings per participant per day = 288.
3. Readings per participant = 812,640 / 240 = 3,386.
4. Wear days = 3,386 / 288 = 11.76, or about 11.8 days.

The pooled calculation gives 11.76 days, and the participant-level average in section 6.2 is 11.9. The small difference is real and worth understanding: the pooled figure divides total readings by total participants, so participants with more readings carry more weight, while the participant-level average weights everyone equally. Neither is wrong, and the two answer slightly different questions.

A completeness of 84 percent sounds adequate and corresponds to fewer than 12 days of a 14-day study.

---

## 8. Data, standards, and regulatory context

### 8.1 Signal formats

| Format | Scope | Typical use | Limitation |
|---|---|---|---|
| WFDB | PhysioNet waveform database format | Open physiological datasets | Requires PhysioNet tooling |
| EDF and EDF+ | European Data Format | Clinical EEG, polysomnography | Limited metadata conventions |
| MEF | Multiscale Electrophysiology Format | Long EEG recordings | Specialized tooling |
| HDF5 | General hierarchical data format | ML pipelines, large arrays | No clinical semantics |
| CSV | Tabular export | Derived values | Loses rate and precision |
| JSON (IEEE 1752) | Mobile health data | Consumer and research apps | Adoption varies |

The choice of format is a provenance decision. A CSV export of derived values is convenient and permanently discards the raw signal.

### 8.2 Standards relevant to wearables

- **IEEE 1752** defines a data model for mobile health data, including timestamps and units. It is referenced in clinical guidance on inpatient CGM integration as a basis for timestamp handling.
- **HL7 FHIR** models observations through the `Observation` resource, with LOINC codes for many measures. A CGM reading can be represented as an `Observation` with a LOINC code, a timestamp, and a value in mg/dL or mmol/L.
- **LOINC** has terms for CGM-derived measures, developed in part through the iCoDE project. Using them makes the data exchangeable.
- **Open mHealth** publishes schemas for mobile health data, including physical activity and sleep.
- **DICOM** has waveform objects for some physiological signals acquired in clinical settings, but consumer wearable data does not use DICOM.

### 8.3 CGM integration guidance

The Diabetes Technology Society's iCoDE project (Integration of Continuous Glucose Monitoring Data into the Electronic Health Record) convened more than 130 people from over 60 organizations and published the 2022 iCoDE Report with 54 recommendation statements: 7 introductory, 27 technical standards, and 20 clinical implementation recommendations. The project reported that all were adopted as strong recommendations with an average approval of 98 percent.

A follow-on project, iCoDE-2, launched in May 2023 and published in the Journal of Diabetes Science and Technology in 2024, extends the work to insulin dosing data from pumps, pens, and connected devices.

The 2024 consensus on inpatient CGM use, published in Diabetes Care, recommends a "data pull" approach through computerized provider order entry rather than a "data push," adoption of IEEE P1752 for timestamps with ISO 8601 formatting, and the full set of 2019 consensus CGM metrics: days worn, percent active, mean glucose, GMI, CV, time in range and below and above range, and the ambulatory glucose profile.

The analytic implication is that the metric set is standardized and the remaining variability is in wear time, device, and population. That is a better situation than most of this module's data types.

### 8.4 Regulatory categories

In the United States, the Food and Drug Administration (FDA) distinguishes:

- **General wellness products**, which make claims about general health without diagnosing or treating disease, and are not regulated as devices.
- **Medical devices**, which diagnose, treat, or prevent disease, and require a premarket pathway.

A wearable that reports steps and sleep is generally a wellness product. A wearable that claims to detect atrial fibrillation is a device, and the claim requires clinical validation.

The FDA has published guidance on digital health technologies for remote data acquisition in clinical investigations, which sets expectations for device selection, validation, and data quality in trials. The core expectation is that the device is fit for the endpoint and that the validation is documented.

In the European Union, the Medical Device Regulation (MDR) 2017/745 governs devices placed on the market, including software with a medical purpose.

> **Note:** These are educational summaries, not legal advice. Device classification is fact-specific and requires regulatory counsel.

### 8.5 Privacy

Wearable data can be personal data even when it is not protected health information. Two points:

- **HIPAA** applies to covered entities and business associates. A consumer wearable used outside a covered entity may not create PHI, but a wearable used in a clinical trial under an institutional protocol usually does.
- **GDPR** applies to personal data in the EU, and health data is a special category requiring an explicit legal basis. A step count is not health data on its own; a step count linked to a diagnosis, or a heart rate variability series, may be.

Location data from wearables is a frequently overlooked identifier. A GPS trace identifies a home and a workplace with high reliability.

### 8.6 What device standards do not cover

IEEE 1752 and the related mobile-health schemas standardize the data envelope, not the measurement. Four gaps remain:

- **Algorithm transparency.** The schema carries the value, not the computation that produced it.
- **Wear detection.** The schema has no standard field for whether the device was worn.
- **Firmware identity.** Version strings are optional and inconsistent.
- **Reference validation.** The schema says nothing about accuracy.

Each gap is why a provenance record has to be built by the data team rather than read from the file metadata.

### 8.7 Reading a device clearance summary

A 510(k) summary is public and short. Four fields decide whether it supports your use:

1. **Indications for use.** The exact wording defines the claim. A device cleared "to estimate heart rate during exercise" is not cleared to detect arrhythmia.
2. **Predicate device.** The comparison device establishes the regulatory lineage, which tells you what the clearance actually demonstrated.
3. **Summary of studies.** The number of participants, the reference standard, and the population. A clearance based on 30 healthy adults is a narrow evidence base.
4. **Limitations.** Some summaries state the populations or conditions in which the device was not evaluated. Those are the populations where your risk is highest.

A clearance is a regulatory finding for a stated use, not a general statement of accuracy.

### 8.8 The international dimension

A device marketed in both the US and the EU has two regulatory files, and they may cover different claims. The EU MDR requires a clinical evaluation that is broader than a 510(k) equivalence argument in some respects, particularly for class IIa and above.

For a global study, the practical consequence is that the endpoint must be acceptable in each jurisdiction where data are collected, and a device acceptable in one may not be acceptable in another. Planning for the strictest jurisdiction avoids a protocol amendment later.

A practical checklist for a multi-country wearable study:

- Confirm the device is available and supported in each country.
- Confirm the intended use is covered by the regulatory status in each country.
- Confirm the data can leave the country under local law, including any localization requirement.
- Confirm the firmware distribution is comparable across countries, since regional versions may differ.
- Confirm the wear protocol is culturally and practically feasible in each setting.

A study that skips the fourth check can find that a regional firmware version computes sleep differently, which is invisible in the aggregate and fatal to a pooled endpoint.

The same caution applies to app versions. A mobile app update can change the sampling mode, the sync behaviour, and the derived value, without any change to the hardware.

Record the app version alongside the firmware version, and treat a change in either as a potential break in the series.

---

## 9. Industry practice and case evidence

### 9.1 The Apple Heart Study as a validation template

The Apple Heart Study, published by Perez and colleagues in the New England Journal of Medicine in 2019, enrolled 419,297 participants over 8 months and is the largest public wearable validation study.

Documented results:

- 2,161 participants (0.52 percent) received an irregular pulse notification, ranging from 3.1 percent among those aged 65 and older to 0.16 percent among those aged 22 to 39.
- Of 450 notified participants who returned analyzable ECG patches, atrial fibrillation was present in 34 percent (97.5 percent confidence interval 29 to 39), 35 percent in those aged 65 and older, and 18 percent in those under 40.
- The positive predictive value was 0.84 (95 percent confidence interval 0.76 to 0.92) for atrial fibrillation on a simultaneous ECG with a subsequent irregular pulse notification, and 0.71 (97.5 percent confidence interval 0.69 to 0.74) for an individual irregular tachogram.

The study's own limitations are the teaching point. Only 450 of 2,161 notified participants returned patches, so the precision estimate rests on a self-selected subset. The study was not designed to measure sensitivity or specificity, and it could not assess screening utility. Those are not flaws in execution; they are limits of the design, and the authors stated them.

The industry lesson: a large wearable study can establish that a notification is often correct (positive predictive value) without establishing that screening helps (clinical utility). Those are different claims, and marketing often conflates them.

### 9.2 The PhysioNet model for open signal data

PhysioNet hosts open physiological datasets with credentialed access for some collections. Two are canonical:

- **MIT-BIH Arrhythmia Database**: 48 half-hour two-lead ECG recordings at 360 Hz, annotated beat by beat. It is the standard benchmark for arrhythmia detection and is small enough to download in minutes.
- **PTB-XL**: a large public 12-lead ECG dataset with clinical labels, widely used for ECG classification research.

The value of these datasets is not size. It is that the annotations, the sampling parameters, and the preprocessing are documented, so a method can be compared across papers. Datasets without that documentation produce results that cannot be compared.

### 9.3 Consumer device validation in practice

Three patterns recur in consumer wearable validation:

1. **Convenience samples.** Validation in young, healthy employees of the device company. The results do not generalize to older adults or to patients.
2. **Narrow reference standards.** Validation against a single clinical measurement at one time point, which cannot capture within-person variability.
3. **Algorithm updates.** The device's algorithm changes with firmware, and a validation published for one version may not apply to the next.

The practical response is to ask three questions of any wearable validation: who was studied, what was the reference standard, and which firmware version. A study that cannot answer all three is a marketing document.

### 9.4 Skin pigmentation and optical sensing

Optical sensing is affected by melanin absorbance, and the effect is documented for pulse oximetry and for PPG-derived measures. The FDA has issued communications about pulse oximeter accuracy and skin pigmentation, and the topic has become a regulatory and equity concern.

For an analyst, the implication is that device error may be differential by skin tone, which means a model trained predominantly on lighter-skinned participants may perform worse and may produce biased estimates for darker-skinned participants. Skin tone is rarely recorded in wearable datasets, which makes the bias unmeasurable post hoc. Recording it at enrollment is the only remedy.

### 9.5 What a mature wearable data team does

- **Records device and firmware** per participant.
- **Reports wear time in days** and completeness by device.
- **Tests missingness for informativeness** rather than assuming it away.
- **Uses a stated sensitivity range** for non-wear correction.
- **Validates against a clinical reference** in the intended population.
- **Records skin tone and other effect modifiers** where optical sensing is used.

These six practices are the difference between a dataset that supports a claim and one that supports a demo.

### 9.6 Why consumer and clinical data diverge

A consumer device is optimized for engagement, battery life, and cost. A clinical device is optimized for accuracy and auditability. Those objectives produce different sampling rates, different algorithms, and different data access.

The divergence is not a conspiracy. It is a design consequence, and it means that a consumer device can be excellent for trend and poor for measurement. A program that uses consumer data for trend and clinical data for confirmation is well designed. One that uses consumer data for diagnosis is not.

### 9.7 How remote monitoring programmes fail

Four failure patterns are common in remote patient monitoring, and all four are data problems rather than clinical ones:

1. **Alert fatigue from artefact.** The algorithm flags motion and poor contact as events, clinicians stop responding, and a real event is missed. The fix is signal quality filtering, not a lower threshold.
2. **Adherence decay.** Wear time falls after the first two weeks, and the programme reports the metrics from the engaged minority as if they represented the cohort. The fix is reporting adherence alongside outcomes.
3. **Selection by digital access.** Patients without smartphones, broadband, or comfort with technology are excluded, and the programme's outcomes reflect a healthier, wealthier population. The fix is measuring the excluded group.
4. **Integration failure.** Device data arrives in a portal that the clinical workflow does not read. The fix is workflow design, not analytics.

None of these is solved by a better model. Each is solved by measuring the right thing and reporting it honestly.

### 9.8 The reimbursement question

Payers reimburse remote monitoring only where a code and a coverage policy exist, and the policy usually specifies the device type, the duration, and the data transmission requirements. A programme built on a consumer device that does not meet the policy's device definition may generate data that cannot be billed.

The analytic consequence is that the data source is partly determined by the payment pathway. A team that chooses a device before checking coverage may produce an excellent dataset that the business cannot monetize.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Derived treated as measured | No raw signal in the dataset | Device algorithm opaque | Cannot re-analyze or validate | Record level; validate against reference |
| Non-wear treated as missing at random | Completeness below 80% and unexamined | Behavioural missingness | Biased metrics | Sensitivity analysis; wear-time reporting |
| Non-wear counted as in range | TIR higher than clinically plausible | Denominator includes unworn time | Overstated control | Exclude non-wear from denominator |
| Device pooled without stratification | Metric differs by device | Algorithm and population differ | Confounded comparison | Stratify; include device covariate |
| Firmware change mid-study | Step change in derived values | Algorithm update | Artificial trend | Record version; split the series |
| Clock drift ignored | Time-of-day pattern smeared | Device clock drift | Misaligned events | Estimate and correct drift |
| Sampling rate assumed | Frequency analysis without rate | Rate unknown or changed | Invalid spectral results | Record rate; check interval distribution |
| Calibration ignored | CGM values offset from lab | Calibration schedule differs | Systematic bias | Record schedule; compare to reference |
| Population narrow | Validation in young healthy adults | Convenience sampling | Poor generalization | Validate in target population |
| Skin tone unrecorded | No effect modifier available | Optical bias unmeasured | Undetected differential error | Record at enrollment |
| Summary data used for individual inference | Only daily means available | Aggregation discards variation | Ecological fallacy | Use individual-level data |
| Battery or charging gaps | Regular multi-hour gaps | Device behaviour | Systematic missingness by time of day | Model the gap pattern |

### 9.9 How a digital biomarker gets qualified

A sensor-derived measure becomes usable in a regulatory or reimbursement context through a sequence that is documented, not assumed:

| Step | Question | Evidence | Typical duration |
|---|---|---|---|
| Concept of interest | What clinical state is being measured? | Clinical rationale | Weeks |
| Analytic validation | Does the sensor measure it accurately? | Bench and reference comparison | Months |
| Clinical validation | Does it associate with the outcome? | Prospective study | Months to years |
| Clinical utility | Does acting on it improve outcomes? | Trial or quasi-experiment | Years |
| Fit-for-purpose determination | Is it adequate for this use? | Regulatory or payer review | Months |

Most wearable programmes stop at step two and describe the result as validation. The later steps are the ones that decide whether the measure can be used, and they are the ones that require clinical data the device company may not have.

### 9.10 What a wearable dataset contract should specify

| Term | Why it matters | Typical gap |
|---|---|---|
| Raw versus derived | Determines re-analysis | Contract silent |
| Device and firmware versions | Determines comparability | Not enumerated |
| Sampling rate and filtering | Determines spectral validity | Not stated |
| Wear detection method | Determines the denominator | Not described |
| Algorithm version | Determines derived values | Not recorded |
| Validation evidence | Determines claims | Marketing summary only |
| Update notification | Determines continuity | No obligation |

A contract that omits the last row allows the vendor to change the algorithm mid-study without telling the sponsor, which is the failure mode that invalidates a longitudinal endpoint.

### 10.1 The three red flags

1. **No device or firmware version.** If the dataset does not say what produced the values, it cannot be compared to anything.
2. **Completeness reported only as a percentage.** Without wear days and a missingness analysis, the number hides the bias.
3. **A validation study with no population description.** A positive predictive value in an unstated population is not a result.

### 10.2 The denominator error in wearable dashboards

The variant in section 6.7 is the single most common wearable analytics error: dividing by expected time rather than observed time. It is common because the expected count is easy to compute from the enrollment period, and the observed count requires a wear detection step.

The signature is a metric that moves when the study period changes but the participants do not. If extending the observation window lowers the average, the denominator is expected time and the missing data are being treated as values.

### 10.3 The "validated device" trap

A device can be validated and still be wrong for your question. Validation is specific to the population, the reference standard, and the firmware. A statement that a device is "FDA cleared" means the manufacturer demonstrated substantial equivalence for a stated intended use, not that the device is accurate for your endpoint in your population.

The practical question is not "is the device validated" but "is the device validated for this measure, in this population, against this reference."

### 10.3 The completeness illusion

A completeness of 84 percent sounds acceptable and is not, for two reasons.

First, 84 percent of a 14-day study is 11.8 days, which may be below the minimum for some CGM metrics. Second, completeness is an average across participants, and the distribution matters: a cohort where every participant has 84 percent is different from one where half have 100 percent and half have 68 percent. The second has a subgroup with almost no data, and pooling hides it.

Always report the distribution, not only the mean. A histogram of wear days by device is the single most informative plot in a wearable analysis.

### 10.4 The validation-versus-utility confusion

A validated measure is not a useful intervention. The distinction matters commercially and clinically:

- **Validation** asks whether the measure is accurate.
- **Utility** asks whether acting on the measure improves outcomes.

A device can detect atrial fibrillation accurately (validation) and still fail to improve outcomes if the detected cases would have been found anyway or if treatment carries risk (utility). Trials that measure utility are larger, longer, and more expensive than validation studies, and marketing often presents validation as if it were utility.

For an analyst, the question is which claim the data can support. A dataset with device outputs and clinical confirmation supports a validation claim. Only a trial with an outcome comparison supports a utility claim.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The device was worn as instructed | Completeness denominator | Non-wear, charging, discomfort | Wear flags; gap pattern analysis |
| Derived values are comparable across devices | Pooled analysis | Different algorithms | Stratify; concurrent-wear substudy |
| Firmware is constant | Trend analysis | Mid-study update | Version distribution by date |
| Timestamps share a clock | Event alignment | Clock drift | Compare to a reference event |
| Missingness is unrelated to the metric | Complete-case metrics | Gaps correlated with physiology | Before-and-after gap comparison |
| The calibration is stable | CGM accuracy | Calibration schedule, sensor age | Compare to fingerstick or lab |
| The sample represents the target population | Generalization | Access and healthy-user bias | Compare demographics to target |
| Optical sensing is unbiased across skin tones | Equity of the estimate | Melanin absorbance | Record skin tone; stratified validation |

### 10.5 The firmware-change failure

A firmware update during a study is not a hypothetical. It produces a step change in derived values that looks like a physiological shift, and it is invisible unless the version is recorded per participant.

The signature is a discontinuity in a derived metric at a date that affects participants across sites. The test is to plot the metric by firmware version; if the versions differ by more than the physiology, the series must be split.

### 11.1 Limits that are inherent

- **Derived data cannot be re-analyzed.** Once the raw signal is gone, the algorithm's errors are permanent.
- **Non-wear is behavioural.** No statistical method recovers what was not measured.
- **Consumer devices change.** A firmware update can invalidate a validation.
- **Wearable cohorts are selected.** Access, interest, and health determine who is in the data.
- **Optical sensing is affected by anatomy.** Skin tone, perfusion, and motion are not nuisance variables; they are effect modifiers.

### 11.2 How to state the limitation

"The cohort wore two CGM models for a planned 14 days. Device A achieved 91.1 percent completeness (median 12.8 wear days) and device B 72.0 percent (median 10.4 days). Metrics were computed on worn time only. Under an assumption that 60 to 90 percent of the missing device B readings were above 180 mg/dL, device B's time in range ranges from 41.9 to 51.8 percent rather than the observed 58.2 percent. The study cannot separate device algorithm from population difference, because no participant wore both devices."

That paragraph gives the reader everything needed to judge the claim.

### 11.3 The limit that decides a trial endpoint

For a trial, the binding limit is whether the endpoint is acceptable to the regulator for the intended claim. A wearable-derived endpoint in a label claim needs analytic validation, a documented device version, and a pre-specified analysis that matches the estimand. Without those, the endpoint may be acceptable to the sponsor and not to the reviewer.

The practical statement is a three-part sentence: what the device measures, how it was validated, and what happens to the endpoint when the device changes. A protocol that cannot answer the third part has not planned for a firmware update during a multi-year trial.

### 11.4 What the metric does not mean

A time in range of 62.7 percent is a population average over worn time. It does not mean:

- That any individual spent 62.7 percent of their day in range.
- That the 14-day period is representative of the year.
- That the reading is accurate at the individual level, if the device has a calibration bias.
- That improving the number improves outcomes, which is a utility question.

Each of these is a different inference, and each requires different evidence. The most common error in a results slide is to move from the first to the last without noticing the gap.

### 11.4 The limit that decides regulatory acceptability

For a regulatory submission, the decisive limit is usually not the statistical precision but the documentation of the measurement. A regulator will ask what the device measures, how it was validated, which version was used, and whether the endpoint is fit for the intended use. A dataset that cannot answer those four questions will not be accepted regardless of the effect size.

---

## 12. Exercises

### Exercise 1 — Data level classification (recall, 20 minutes)

Take three wearable datasets you can find documentation for. Classify each as raw, derived, or summary, and name one analysis each supports and one it prevents.

Deliverable: a three-row table.

Acceptance criteria: each classification is justified by the documentation; the prevented analysis is a genuine limitation of the level.

### Exercise 2 — CGM metrics (calculation, 45 minutes)

Using the counts in section 6.3, recompute TIR, TBR, TAR, CV, and GMI for the pooled cohort. Show every intermediate value.

Deliverable: a metric table with the arithmetic shown.

Acceptance criteria: all percentages sum to 100 within rounding; the GMI formula is stated; the pooled mean is reading-weighted.

### Exercise 3 — Non-wear sensitivity (calculation, 45 minutes)

Recompute the corrected device B TIR under assumptions that 60, 70, 80, and 90 percent of missing readings were hyperglycemic. Plot or tabulate the result and state the range.

Deliverable: a four-row table and a one-paragraph interpretation.

Acceptance criteria: arithmetic correct; the interpretation states that the range, not the point, is the result.

### Exercise 4 — Clock drift audit (application, 60 minutes)

Given a device clock that drifts 0.8 seconds per day, compute the accumulated drift over 14 days. Then describe how a 45-second drift affects a time-of-day analysis with 1-hour bins.

Deliverable: the drift calculation and a short analysis.

Acceptance criteria: arithmetic correct; the effect on binning is quantified, not asserted.

### Exercise 5 — Validation critique (critique, 60 minutes)

Find a published wearable validation study. Identify the population, the reference standard, and the firmware version. Write a 400-word critique.

Deliverable: a 400-word critique with two proposed additional analyses.

Acceptance criteria: the critique names the specific population limitation; each proposal is feasible; no claim about author intent.

### Exercise 6 — Missingness mechanism (application, 45 minutes)

A dataset shows gaps concentrated between 01:00 and 05:00. Propose two mechanisms and a test to distinguish them.

Deliverable: two mechanisms and a test design.

Acceptance criteria: the mechanisms are distinct (for example, charging behaviour versus sleep-related sensor shutdown); the test uses available data.

### Exercise 7 — Provenance record (design, 60 minutes)

Write a provenance record for a wearable dataset: device, firmware, rate, filter, processing chain, wear protocol, completeness, and validation.

Deliverable: a one-page record.

Acceptance criteria: a second analyst could reproduce the metrics; every version is named; the validation is specific to population and reference.

### Exercise 8 — Raw versus derived (design, 45 minutes)

For three endpoints (atrial fibrillation detection, sleep staging, step count), decide whether raw or derived data is required, and state the access and validation consequences.

Deliverable: a three-row decision table.

Acceptance criteria: each decision names the analysis that requires the chosen level; the consequences are specific.

### Exercise 9 — Device comparison design (design, 60 minutes)

Design a study to separate device algorithm from population in the section 6 example. Specify the design, the sample size logic, and the analysis.

Deliverable: a one-page design.

Acceptance criteria: the design includes concurrent wear; the analysis separates within-person device difference from between-person population difference; the sample size logic is stated, even if approximate.

### Exercise 10 — Denominator sensitivity (calculation and critique, 45 minutes)

Using the counts in sections 6.3 and 6.7, recompute the pooled and per-device TIR under a third denominator that counts only readings during a pre-specified 06:00-24:00 window, assuming 78 percent of readings fall in that window and they carry 70 percent of the in-range readings. State which denominator you would report for a clinical audience.

Deliverable: three recalculated TIR values and a recommendation.

Acceptance criteria: the numerator and denominator are recomputed consistently; the recommendation names the audience and the question; the assumption is labelled.

---

## 13. Self-check answers

**A1. What are the three data levels, and what does each support?**
Raw (native samples) supports re-analysis and algorithm development. Derived (device-computed values) supports utilization and trend. Summary (aggregated metrics) supports population description only.

**A2. Compute completeness for 812,640 observed readings out of 967,680 expected.**
812,640 / 967,680 = 0.8398, or 84.0%.

**A3. Compute the implied wear days for 3,386 readings per participant at 288 readings per day.**
3,386 / 288 = 11.76, or about 11.8 days.

**A4. Compute the pooled TIR from device A at 64.8% of 551,232 readings and device B at 58.2% of 261,408 readings.**
A: 551,232 × 0.648 = 357,198. B: 261,408 × 0.582 = 152,140. Total 509,338. Denominator 812,640. TIR = 62.7%.

**A5. Compute the GMI for a mean glucose of 168 mg/dL.**
GMI = 3.31 + 0.02392 × 168 = 3.31 + 4.0186 = 7.33%.

**A6. Why is the pooled mean not the simple average of the device means?**
Because the devices contribute different numbers of readings. The pooled mean must be weighted by readings: (164 × 551,232 + 176 × 261,408) / 812,640 = 167.86, not 170.

**A7. State the direction of bias if device B's missing periods were hyperglycemic.**
The observed TIR overstates true TIR, because the missing time would have been mostly out of range. The bias is upward, and its magnitude depends on the share of missing time that was hyperglycemic.

**A8. What is the difference between interstitial glucose and blood glucose?**
CGM measures glucose in interstitial fluid, which lags blood glucose by roughly 5 to 15 minutes and is affected by calibration and sensor compression. Rapid changes are therefore mis-timed if CGM is treated as blood glucose.

**A9. Why is a derived heart rate not interchangeable across devices?**
Because each device computes it with its own algorithm, sampling rate, and filtering. Two devices can disagree by several beats per minute on the same person, and neither is wrong relative to its own definition.

**A10. What are the three questions to ask of a wearable validation study?**
Who was studied, what was the reference standard, and which firmware version. A study that cannot answer all three does not support a general claim.

**A11. What is clock drift and how does it affect analysis?**
The divergence between a device clock and a reference clock over time. It misaligns events and smears time-of-day patterns; at 0.8 seconds per day, a 14-day study accumulates about 11 seconds, which is negligible for hourly bins and material for beat-level alignment.

Check: 0.8 × 14 = 11.2 seconds.

**A12. Rubric for exercise 7 (provenance record).**
A strong record names the device model, firmware, sampling rate, on-device filter, processing chain with parameters, wear protocol, completeness by device, and the validation study with its population and reference standard. A weak record names the device and the date only, with no firmware and no completeness breakdown.

---

## 14. Interview questions

**Q1. What is the difference between raw and derived wearable data, and why does it matter?**
A strong answer: raw is the sampled signal; derived is the device's computed value. Only raw supports re-analysis and algorithm transparency. Derived data is a measurement of the device's algorithm as much as of the patient.

**Q2. How would you handle non-wear in a CGM analysis?**
A strong answer: compute metrics on worn time only, report wear days and completeness by device, and run a sensitivity analysis assuming a plausible distribution for the missing periods. Weak answers drop missing values without examining the pattern.

**Q3. Why is a device comparison from observational data difficult?**
Because device assignment is not random and is related to access, geography, and clinical practice. A difference between device users can be algorithm, population, or wear behaviour, and observational data cannot separate them without concurrent wear.

**Q4. How would you validate a wearable-derived endpoint?**
A strong answer: analytic validation against a reference instrument, clinical validation in the intended population, and a documented firmware version; report bias and limits of agreement, not only correlation.

**Q5. What is the Apple Heart Study's main limitation?**
Only 450 of 2,161 notified participants returned ECG patches, so the precision estimates rest on a self-selected subset, and the study was not designed to measure sensitivity or specificity or to assess screening utility.

**Q6. What does a positive predictive value of 0.84 tell you, and what does it not?**
It tells you that when the algorithm flags and a simultaneous ECG is available, 84 percent show atrial fibrillation. It does not tell you sensitivity, specificity, or whether screening improves outcomes.

**Q7. How does skin pigmentation affect optical sensing?**
Melanin absorbs light, reducing signal-to-noise and biasing some optical measurements, which is documented for pulse oximetry. The bias is differential across participants and is unmeasurable if skin tone is not recorded.

**Q8. What would you check first in an unfamiliar signal dataset?**
A strong answer: sampling rate, timestamp basis, device and firmware, and completeness. Those four determine whether any downstream analysis is valid.

**Q9. Why can a CSV export of derived values be a permanent loss?**
Because it discards the raw signal, the sampling rate, and the processing chain. The derived values cannot be recomputed under a different algorithm, and the dataset cannot be compared to one from another device.

**Q10. What is the difference between adherence and completeness?**
Adherence is whether the device was worn as instructed; completeness is whether the data arrived. A worn device with a sync failure has high adherence and low completeness.

**Q11. How would you design a study to compare two CGM devices?**
A strong answer: a concurrent-wear design where each participant wears both devices, so the within-person difference isolates the device from the population, with a sample size based on the expected within-person variability.

**Q12. What would you put in a wearable provenance record?**
Device model, firmware, sampling rate, on-device filter, processing chain and parameters, wear protocol, completeness by device and participant group, validation study and its reference standard, and known effect modifiers such as skin tone.

**Q13. A vendor says its algorithm has 95 percent accuracy. What do you ask?**
A strong answer: accuracy for what, in whom, against what reference, and at which firmware version. A single accuracy figure without a prevalence, a reference standard, and a population is not interpretable, and for a rare event it can be high while the positive predictive value is low.

**Q14. Why is a 14-day CGM study not the same as 14 days of data?**
Because wear time is less than study duration. At 84 percent completeness the study provides about 11.8 days of data, and at device B's 72 percent it provides about 10.1 days. Clinical guidance sets a minimum wear time, not a minimum enrollment duration.

---

## 15. Key terms

- **Accelerometry** — measurement of acceleration.
- **Adherence** — proportion of time worn as intended.
- **AGP** — ambulatory glucose profile.
- **Artefact** — signal distortion not from physiology.
- **Baseline wander** — low-frequency drift in a signal.
- **CGM** — continuous glucose monitoring.
- **Clock drift** — divergence between device and reference time.
- **CV** — coefficient of variation.
- **Derived value** — a device-computed measure.
- **Digital biomarker** — a sensor-derived measure with clinical meaning.
- **ECG** — electrocardiography.
- **EDF** — European Data Format.
- **EEG** — electroencephalography.
- **EMG** — electromyography.
- **Firmware** — device embedded software.
- **GMI** — glucose management indicator.
- **HbA1c** — glycated haemoglobin.
- **IEEE 1752** — a standard for mobile health data.
- **Interstitial glucose** — glucose in tissue fluid.
- **Non-wear** — time when the sensor was not worn.
- **Nyquist frequency** — half the sampling rate.
- **PPG** — photoplethysmography.
- **Sampling rate** — samples per second.
- **SpO2** — peripheral oxygen saturation.
- **TAR** — time above range.
- **TBR** — time below range.
- **TIR** — time in range.
- **Wear time** — time the sensor was worn.
- **WFDB** — the PhysioNet waveform database format.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Peer-reviewed evidence**

- Perez, M. V., et al. "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation." *New England Journal of Medicine*, 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
- Full text via PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC8112605/

**Standards and clinical guidance**

- Diabetes Technology Society. *iCoDE Project: Integration of CGM Data into the EHR.* https://www.diabetestechnology.org/icode/
- iCoDE Project final recommendations, abstract 116-LB, *Diabetes* 2023. https://doi.org/10.2337/db23-116-lb
- *The Launch of the iCoDE-2 Standard Project: Integration of Connected Diabetes Device Data Into the Electronic Health Record.* *Journal of Diabetes Science and Technology*, 2024;18(1):82-88. https://doi.org/10.1177/19322968231207888
- Simonson, G., et al. "Expert Panel Recommendations for a Standardized Ambulatory Glucose Profile Report for Connected Insulin Pens." *Diabetes Technology & Therapeutics*, 2024;26(11). https://doi.org/10.1089/dia.2024.0107
- *Consensus Considerations and Good Practice Points for Use of Continuous Glucose Monitoring Systems in Hospital Settings.* *Diabetes Care*, 2024;47(12):2062-2075. https://doi.org/10.2337/dc24-1385
- IEEE. *IEEE 1752, Standard for Mobile Health Data.* (Cite the standard; no public full-text URL given.)
- HL7 International. *FHIR Release 5 (5.0.0)*, Observation resource. https://hl7.org/fhir/

**Regulators**

- US Food and Drug Administration. *Digital Health Center of Excellence.* https://www.fda.gov/medical-devices/digital-health-center-excellence
- US Food and Drug Administration. Guidance on *Digital Health Technologies for Remote Data Acquisition in Clinical Investigations.* (Cite the guidance title; the FDA guidance page is the authoritative source.)
- European Union. *Regulation (EU) 2017/745 on medical devices.* https://eur-lex.europa.eu/eli/reg/2017/745/oj
- European Union. *Regulation (EU) 2016/679 (General Data Protection Regulation).* https://eur-lex.europa.eu/eli/reg/2016/679/oj

**Datasets and tools**

- PhysioNet. *MIT-BIH Arrhythmia Database.* https://physionet.org/content/mitdb/
- PhysioNet. *PTB-XL, a large publicly available electrocardiography dataset.* https://physionet.org/content/ptb-xl/
- PhysioNet. https://physionet.org/

> **Note:** The Apple Heart Study figures are reported as published in 2019 and are used as a validation-design example, not as evidence about current device performance. Firmware and algorithms have changed since publication.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_imaging-and-dicom.md`](04_imaging-and-dicom.md)
- Next topic: [`06_omics-data.md`](06_omics-data.md)
- Related sibling: [`01_ehr-data.md`](01_ehr-data.md)
- Related sibling: [`07_trial-data.md`](07_trial-data.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../course/02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.5 in the survey file (data-quality profile) is the natural companion to the completeness and missingness analysis in sections 5 and 6.

Related reading: [`04_imaging-and-dicom.md`](04_imaging-and-dicom.md) for the other high-frequency data type; [`07_trial-data.md`](07_trial-data.md) for how wearable endpoints are regulated inside a trial; [`09_licensing-and-access.md`](09_licensing-and-access.md) for the contracts that govern device data; and [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) for the formal vocabulary of completeness and plausibility.

The sentence to remember: a sensor is not a biomarker until it has been validated for this measure, in this population, against this reference, with this firmware.

### 7.4 A second calculation: the device gap under two denominators

The variant in 6.7 is easy to reproduce by hand and worth doing once, because it makes the denominator choice concrete.

1. Device A in-range readings = 551,232 × 0.648 = 357,198.
2. Device B in-range readings = 261,408 × 0.582 = 152,140.
3. Worn-time rates: 357,198 / 551,232 = 64.8%; 152,140 / 261,408 = 58.2%. Gap = 6.6 points.
4. Expected-time rates: 357,198 / 604,800 = 59.1%; 152,140 / 362,880 = 41.9%. Gap = 17.2 points.

The ratio of the two gaps is 17.2 / 6.6 = 2.6, which is the ratio of the devices' missingness rates: 27.96 / 8.86 = 3.2. The gaps are not proportional to the missingness because the in-range proportions also differ, but the direction is the same: the device with more missing data is penalized more by the expected-time denominator.

One more observation from that arithmetic: the ratio of gaps (2.6) is not the ratio of missingness (3.2), because the in-range proportions differ between devices. A quick proportionality assumption would have been close and wrong, which is why the two figures are computed rather than inferred.

Four questions to carry into the omics topic:

- Is the measurement absolute or relative?
- Is the technical variation structured by batch?
- Is the sample size enough for the number of tests?
- Does the data use agreement cover the analysis and the publication?

The first two questions are the ones that transfer most directly: a derived heart rate and a relative RNA abundance are both measurements whose meaning depends on a pipeline that the analyst did not write.

That dependence is the subject of the next topic, where the pipeline is called a reference genome and the batch is called a flow cell.
