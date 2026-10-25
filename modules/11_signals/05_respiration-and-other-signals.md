# Respiration and Other Signals: Effort, Airflow, EMG, EDA, and Continuous Glucose

> Module 11 — Physiological Signals and Wearables
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals); Topic 02 (ECG) for the apnea reference
> Version 2026.3

## 1. Why this topic exists

The signals in this topic are a set: respiration, electromyography (EMG), electrodermal activity (EDA), and continuous glucose monitoring (CGM). They look unrelated. Respiration is mechanical, EMG is electrical, EDA is electrochemical, and CGM is biochemical. They share a structure that matters for data science: each is a slow, low-amplitude, artifact-prone signal whose derived metric depends almost entirely on the definition of the event and the handling of missing data.

Respiration is measured as effort, airflow, or both. The respiratory rate is easy to compute and easy to compute wrongly: a zero-crossing counter on an asymmetric waveform can double or triple the true rate, and a motion artifact can look like a breath. Sleep apnea is a scored event with an amplitude criterion, a duration criterion, and a desaturation criterion, and the criteria changed in the 2023 AASM manual.

EMG is measured as a broadband electrical signal. The envelope is the feature, and the envelope depends on the filter band, the rectification, and the smoothing window. A surface EMG can detect a muscle contraction but cannot easily measure its force, because the relationship depends on electrode placement, muscle length, and fatigue.

EDA is measured as skin conductance. It has a slow tonic component and fast phasic responses, and the decomposition into the two is a modelling choice. A "stress score" built on EDA is a constructed index, not a measurement of stress.

CGM is measured as interstitial glucose. It is not blood glucose, it lags blood glucose, and it is calibrated against blood glucose. The derived metrics, especially time in range (TIR), are defined by consensus thresholds and depend on sensor coverage. A TIR computed on 60% of the days is not the same measurement as a TIR computed on 90%.

Four failures recur across the set.

**The wrong event definition.** A respiratory rate algorithm that counts zero crossings, an apnea detector that uses a 50% flow reduction instead of the AASM 90%, and a hypoglycemia detector that uses 70 mg/dL when the consensus defines a second threshold at 54 mg/dL are all measuring something, but not the thing the claim names.

**The wrong denominator.** A TIR of 75% computed over the periods the sensor was active is not the same as a TIR of 75% over 14 days. A respiratory rate averaged over valid breaths is not the same as a rate averaged over the night. The denominator is part of the metric.

**Calibration and lag.** CGM is calibrated against fingerstick blood glucose, and the interstitial signal lags blood by 5 to 15 minutes. A hypoglycemia alert that fires on the lagged signal fires late. A calibration that is stale drifts.

**Artifact that becomes a diagnosis.** A belt that slips produces a slow drift that looks like a prolonged apnea. An EDA electrode that loses contact produces a step that looks like a strong response. A CGM sensor under pressure produces a "compression low" that looks like hypoglycemia.

This topic covers the chain for each signal, the event definitions that matter, the metrics that are used in practice, and the failure modes that turn artifact into a label. It builds on Topic 01 for sampling and filtering, and it sets up Topic 06 for feature engineering and Topic 08 for digital biomarkers.

### 1.1 Three composite failures

**The tachypnea that was a belt.** A remote monitoring system tracks respiratory rate from a chest belt. A patient turns in bed, the belt slips, and the waveform becomes asymmetric. The zero-crossing counter reports 42 breaths/min, the early-warning score fires, and the patient is assessed for sepsis. The true rate is 16. The cost is an unnecessary workup and alarm fatigue. The technical cause is a detection method that assumes a symmetric waveform.

**The hypoglycemia that was a mattress.** A CGM user sleeps on the sensor arm. The pressure reduces local perfusion, the sensor reports a rapid fall to 55 mg/dL, and the alert wakes the user. The blood glucose is 95 mg/dL. The event is a compression low, not hypoglycemia. The cost is a disturbed night and a loss of trust in the alerts. The technical cause is a physiological artifact that the algorithm did not distinguish from a true low.

**The AHI that changed with the manual.** A sleep lab upgrades to AASM V3 and keeps the 4% hypopnea rule as an option. A research dataset scored with the 3% rule is compared with a clinical dataset scored with the 4% rule. The AHIs differ by 30%, and the difference is attributed to the population. The cost is a wrong scientific conclusion. The technical cause is a label definition change, not a physiological one.

The three failures share a structure: a metric is computed without its definition, its proxy, or its artifact model. The fix is to make the definition explicit, to validate the proxy, and to model the artifact before the metric.

A fourth failure is a metric reported without its population. A TIR target for a healthy adult is not the target for a pregnant woman or an older adult with hypoglycemia risk. The number is the same; the goal is not. A report that omits the population invites the reader to apply the wrong target.

### 1.2 Why these signals are grouped

The four signals are grouped because they share three properties that ECG and EEG do not.

- **Slow sampling.** Respiration at 10-100 Hz, EDA at 4-32 Hz, CGM at one sample per 5 min. The data volumes are small, and the analysis is dominated by event definitions and missingness rather than by signal processing.
- **Proxy measurements.** Belt for volume, interstitial fluid for blood, conductance for arousal. The gap between the proxy and the quantity is the central validation problem.
- **Consensus-defined metrics.** AHI and TIR are defined by consensus documents, not by physics. The definition is part of the measurement, and a change in the definition changes the metric.

A data scientist who treats these signals like ECG, with a focus on filters and classifiers, will miss the parts that matter most: the event definition, the denominator, and the proxy.

A second consequence is that these signals are often analysed in the aggregate, not beat by beat. A TIR is a summary of 4,032 readings; an AHI is a summary of a night. The summary is where the decisions are made, and the summary is where the definition and the denominator have their effect. Improving the classifier without fixing the summary definition improves the wrong layer.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** respiratory rate from an effort or airflow signal using autocorrelation and peak detection, and explain why zero-crossing counting fails on asymmetric waveforms.
2. **Apply** the AASM V3 apnea and hypopnea definitions, including the amplitude, duration, and desaturation criteria.
3. **Compute** an EMG envelope with a stated band, rectification, and smoothing window, and explain why the envelope is not force.
4. **Decompose** EDA into tonic and phasic components, and state the assumptions of the decomposition.
5. **Compute** CGM time in range, time below range, time above range, coefficient of variation, and glucose management indicator from a series with gaps.
6. **Evaluate** the effect of sensor coverage and missingness on a CGM metric.
7. **Identify** compression lows, sensor lag, belt displacement, and motion artifact from the waveform and the context.
8. **Design** a validation for a respiratory rate or CGM metric that states the reference, the population, the event definition, and the denominator.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are written so that the event definition and the denominator appear explicitly. Outcome 2 asks for the AASM criteria, not for "apnea detection." Outcome 5 asks for the CGM metrics with coverage, not for a single TIR. The wording is deliberate: a learner who can state the definition can audit a metric, and a learner who can only run a function cannot.

## 3. The core mental model

Each signal is a slow variable with a fast event on top. The measurement is the slow variable; the event is a decision; the metric is an aggregation over time.

```text
 respiration            EMG                 EDA                 CGM
 effort/airflow          muscle              sweat glands        interstitial glucose
      |                    |                     |                     |
 band-pass            band-pass            low-pass             calibration
 0.1-3 Hz             20-500 Hz            0-5 Hz               + lag model
      |                    |                     |                     |
 breath peaks         rectified envelope    tonic/phasic         glucose series
      |                    |                     |                     |
 rate, apnea          activation          response rate        TIR, TBR, TAR
      |                    |                     |                     |
 decision:            decision:            decision:            decision:
 apnea severity       contraction          arousal/stress       alert, insulin
```

Redraw it with the units on each arrow. Respiration is in breaths per minute; EMG is in microvolts or a normalized envelope; EDA is in microsiemens; CGM is in milligrams per decilitre or millimoles per litre.

### 3.1 The event definition is the measurement

For respiration, the event is a breath or an apnea. For EMG, the event is a contraction. For EDA, the event is a phasic response. For CGM, the event is a glucose excursion or a hypoglycemic episode. In every case, the event has an amplitude criterion, a duration criterion, and often a context criterion. Changing any of them changes the count and the metric.

The AASM definitions are the clearest example. An apnea is a 90% or greater reduction in airflow for at least 10 s. A hypopnea is a 30% or greater reduction for at least 10 s with a 3% desaturation or an arousal; the 4% desaturation version is optional in V3. The difference between 90% and 50% is the difference between an apnea and a hypopnea, and the difference between 3% and 4% is the difference between a scored event and an unscored one.

### 3.2 The denominator is part of the metric

A rate is a count divided by a time. A percentage is a count divided by a total. If the denominator excludes periods when the signal was invalid, the metric describes the valid periods, not the whole record. That is sometimes the right choice and sometimes a bias.

For CGM, the consensus recommends at least 14 days of data with at least 70% sensor active time. A 14-day record with 70% coverage is the minimum for a representative TIR. A 7-day record with 90% coverage is not equivalent, because the days may not represent the week.

For respiration, a respiratory rate averaged over valid breaths during a sleep study is not the same as the rate over the whole night, because the invalid periods may be the periods with apnea. The denominator must be stated.

### 3.3 The amplitude and the unit

Respiration effort is measured in arbitrary units or in microvolts; airflow is measured in litres per second; EMG in microvolts; EDA in microsiemens; CGM in milligrams per decilitre. The amplitude criteria are defined in physical units, and a signal in arbitrary units cannot be compared with them without a calibration.

A respiratory belt is a strain gauge or an inductive plethysmograph. Its output is proportional to the change in chest or abdominal circumference, not to lung volume. A 90% reduction in belt amplitude is a 90% reduction in circumference change, which is not the same as a 90% reduction in airflow. The AASM criteria are defined for airflow, and a belt-based apnea detector is an approximation.

### 3.4 Calibration and lag

CGM is the clearest case of a signal that is not the quantity of interest. The sensor measures interstitial glucose, which is related to blood glucose by diffusion. The relationship has a lag of 5 to 15 minutes, a gain that varies by sensor, and a noise that varies with compression and temperature.

A CGM reading is therefore a calibrated estimate of blood glucose with a lag and an error. The error is small in steady state and large during rapid changes. A hypoglycemia alert that uses the raw CGM value with a 10-minute lag fires after the blood glucose has already fallen. A predictive alert uses the trend to compensate, at the cost of more false alarms.

### 3.5 Measurement, feature, biomarker, decision

- **Measurement**: respiratory rate in breaths per minute; glucose in mg/dL; skin conductance in µS.
- **Feature**: apnea-hypopnea index (AHI), TIR, EMG envelope, EDA response rate.
- **Biomarker**: AHI as a marker of sleep apnea severity; TIR as a marker of glycemic control; EDA response as a marker of sympathetic arousal.
- **Decision**: diagnose apnea, adjust insulin, flag stress.

The AHI is a feature that has become a clinical biomarker through validation.

A third consequence is that a change in the definition can look like a change in the patient. An AHI that rises after a lab switches from the 4% to the 3% rule is a definition change, not a worsening of apnea. A TIR that falls after a clinic switches from 70-180 to 70-140 is a target change, not a loss of control. The definition belongs in the metadata, next to the value.

A fourth consequence is that the validation of these signals is often a validation of a threshold, not of a waveform. The question is not "can the model detect a breath" but "does the AHI classify severity the same way an adjudicator does." The threshold, the event definition, and the severity cut-points are all parts of the validation target. TIR is a feature that has become a consensus target. EDA response is a feature that is still mostly research. The maturity of the evidence differs, and the claim should reflect it.

### 3.6 The proxy table

| Quantity of interest | Measured proxy | Relationship | Main error source |
|---|---|---|---|
| Lung volume | Belt circumference | Approximately linear over a small range | Belt slip, posture |
| Airflow | Thermistor temperature | Nonlinear, poor at low flow | Ambient temperature |
| Airflow | Nasal pressure | Approximately linear over a range | Cannula position |
| Muscle force | Surface EMG | Monotonic, not linear | Placement, fatigue, cross-talk |
| Sympathetic arousal | Skin conductance | Mediated by sweat glands | Temperature, contact |
| Blood glucose | Interstitial glucose | Lagged, calibrated | Lag, calibration, pressure |

The table is the core of the topic. Every metric in this chapter is derived from a proxy, and the validation question is whether the proxy is adequate for the decision. A belt is adequate for detecting a pause in breathing; it is not adequate for measuring tidal volume. A CGM is adequate for TIR; it is not adequate for a rapid-acting insulin dose without a lag model.



## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Respiration | Gas exchange measured by effort, flow, or volume | Calling belt amplitude volume | breaths/min |
| Respiratory rate | Breaths per minute | Counting zero crossings | breaths/min |
| Tidal volume | Volume per breath | Confusing with minute ventilation | L |
| Minute ventilation | Volume per minute | Confusing with respiratory rate | L/min |
| Airflow | Rate of gas movement | Confusing with effort | L/s |
| Effort | Chest or abdominal movement | Treating as airflow | arbitrary units |
| RIP | Respiratory inductive plethysmography | Treating as calibrated volume | arbitrary units |
| Thermistor | Temperature-based airflow sensor | Confusing with pressure | °C |
| Nasal pressure | Pressure-based airflow sensor | Assuming linear flow | Pa |
| Apnea | ≥90% airflow reduction for ≥10 s | Using 50% | event |
| Hypopnea | ≥30% reduction with desaturation or arousal | Ignoring the desaturation criterion | event |
| AHI | Apneas plus hypopneas per hour | Using the wrong denominator | events/h |
| ODI | Oxygen desaturation index | Confusing with AHI | events/h |
| Desaturation | Drop in SpO2 from baseline | Using an absolute threshold | % |
| Arousal | Abrupt EEG frequency shift | Confusing with movement | event |
| EMG | Electromyography | Treating as force | µV |
| Surface EMG | EMG from skin electrodes | Assuming a fixed gain | µV |
| Envelope | Smoothed rectified EMG | Using raw EMG as envelope | µV |
| Activation | EMG above baseline | Confusing with force | µV or % |
| Fatigue | Decline in EMG amplitude or frequency | Confusing with activation | state |
| EDA | Electrodermal activity | Calling it galvanic skin response | µS |
| SCL | Skin conductance level (tonic) | Confusing with phasic | µS |
| SCR | Skin conductance response (phasic) | Counting noise as responses | µS |
| EDA response rate | Responses per minute | Using a fixed threshold | /min |
| Decomposition | Separating tonic and phasic | Assuming a unique solution | method |
| CGM | Continuous glucose monitoring | Calling it blood glucose | mg/dL or mmol/L |
| Interstitial glucose | Glucose in tissue fluid | Treating as blood glucose | mg/dL |
| Sensor lag | Delay between blood and interstitial glucose | Ignoring in alerts | min |
| Calibration | Mapping sensor signal to glucose | Assuming it is stable | method |
| TIR | Time in range 70-180 mg/dL | Using a different range silently | % |
| TBR | Time below range | Ignoring the 54 mg/dL threshold | % |
| TAR | Time above range | Ignoring the 250 mg/dL threshold | % |
| CV | Coefficient of variation of glucose | Confusing with SD | % |
| GMI | Glucose management indicator | Calling it A1C | % |
| Compression low | False low from sensor pressure | Treating as hypoglycemia | mg/dL |
| Sensor active time | Time the sensor provides data | Using calendar time | % |
| Coverage | Share of expected readings present | Ignoring gaps | % |
| MARD | Mean absolute relative difference | Reporting without a reference | % |
| Time-weighted | Metric weighted by duration | Confusing with sample-weighted | method |
| Missingness | Absence of data | Treating as zero | % |
| Respiration | Gas exchange measured by effort, flow, or volume | Calling belt amplitude volume | breaths/min |
| Respiratory sinus arrhythmia | Heart-rate variation with breathing | Confusing with arrhythmia | ms |
| Baseline breathing | The pre-event amplitude reference | Using a fixed threshold | arbitrary units |
| Flow limitation | Partial upper-airway obstruction | Confusing with hypopnea | event |
| Position sensor | Body position during sleep | Ignoring supine effect | categorical |
| Sleep time | Time scored asleep | Using time in bed | h |
| Time in bed | Time from lights out to lights on | Using as sleep time | h |
| MARD | Mean absolute relative difference | Reporting without a reference | % |
| AGP | Ambulatory glucose profile | Treating as a raw trace | report |
| Sensor active time | Time the CGM provides data | Using calendar time | % |
| Predictive alert | Alert based on trend | Confusing with threshold alert | method |
| Rate of change | Glucose slope | Ignoring lag | mg/dL/min |
| Compression low | False low from pressure | Treating as hypoglycemia | mg/dL |
| Electrodermal response | Phasic skin conductance | Counting noise | µS |
| Continuous decomposition | Model-based EDA split | Assuming uniqueness | method |
| Median frequency | EMG spectral midpoint | Confusing with mean | Hz |
| Maximal voluntary contraction | Reference for EMG normalization | Using raw amplitude | % MVC |
| Cross-talk | Signal from adjacent muscles | Treating as the target muscle | % |

## 5. The mechanics, step by step

The procedure is written for a multi-signal sleep or metabolic study with respiration, EMG, EDA, and CGM. Each signal can be processed independently; the steps that are signal-specific are marked.

### Step 1. Define the measurement and the event

**Input:** the decision, the population, and the setting.

**Output:** a specification with the quantity, the unit, the event definition, the denominator, and the accuracy target.

**Rationale:** the event definition determines the metric. A respiratory rate is not an apnea index. A TIR is not a mean glucose. A stress score is not an EDA response rate. Each requires its own definition and its own validation.

**Branch point:** if the decision is apnea diagnosis, the event definition must be the AASM definition and the denominator must be hours of sleep, not hours in bed. If the decision is glycemic management, the TIR must use the consensus range and a coverage criterion.

**Edge case:** a specification that says "respiratory rate" without saying whether it is instantaneous, averaged over a minute, or averaged over a night is incomplete. Three measurements share the name.

### Step 2. Select the sensor and the site

**Input:** the specification.

**Output:** a sensor type, a site, and a sampling rate.

**Rationale:** each signal has a sensor that measures a proxy, not the quantity itself.

- Respiration effort: a chest or abdominal belt measures circumference change.
- Airflow: a thermistor measures temperature change; a nasal pressure cannula measures pressure change.
- EMG: surface electrodes measure the electrical activity of a muscle group.
- EDA: electrodes on the palm or wrist measure skin conductance.
- CGM: a subcutaneous filament measures interstitial glucose via an enzymatic reaction.

**Branch point:** if the question is apnea, airflow is the primary signal and effort is the secondary signal that distinguishes obstructive from central apnea. If the question is respiratory rate during daily life, effort or PPG-derived respiration may be the only available signal.

**Edge case:** a belt and a cannula measure different things. A belt can show effort during an obstructive apnea (effort persists, airflow stops); a cannula shows the airflow stop. Using one without the other cannot classify the apnea type.

**Sensor comparison.** The choice of sensor determines what can be claimed.

| Signal | Sensor | Measures | Strength | Limitation |
|---|---|---|---|---|
| Effort | RIP belt | Circumference change | Detects effort | Not volume; slip artifact |
| Effort | Piezo belt | Strain change | Cheap, simple | Drift; nonlinear |
| Airflow | Thermistor | Temperature change | Simple | Poor at low flow |
| Airflow | Nasal pressure | Pressure change | Good at low flow | Requires cannula |
| Airflow | Microphone | Sound | Non-contact | Sensitive to ambient noise |
| EMG | Surface electrodes | Muscle electrical activity | Non-invasive | Cross-talk; placement |
| EDA | Palmar electrodes | Skin conductance | Sensitive to arousal | Non-specific; contact |
| CGM | Subcutaneous filament | Interstitial glucose | Continuous | Lag; calibration |

A claim should name the sensor, because the same construct measured with two sensors is two measurements. A respiratory rate from a belt and a respiratory rate from a cannula can differ during flow limitation, when effort continues but airflow falls.

### Step 3. Sample and filter

**Input:** the raw signal and the sampling rate.

**Output:** a filtered signal for event detection and a separate slow signal for the baseline.

**Rationale:** the signal bands differ.

| Signal | Band of interest | Typical sampling rate | Filter |
|---|---|---|---|
| Respiration effort | 0.1-3 Hz | 10-100 Hz | band-pass 0.05-3 Hz |
| Airflow | 0.1-5 Hz | 25-100 Hz | band-pass 0.05-10 Hz |
| EMG | 20-500 Hz | 1,000-10,000 Hz | band-pass 20-450 Hz |
| EDA | 0-5 Hz | 4-32 Hz | low-pass 1-5 Hz |
| CGM | 0-0.01 Hz | 1 per 5 min | none; lag model |

**Arithmetic:** at 25 samples/s, a 4 s breath is 100 samples. At 1 sample/s, the same breath is 4 samples, which is enough for autocorrelation but not for waveform morphology. At 0.0033 samples/s (5 min CGM), a 4 s breath does not exist; the CGM sampling is designed for glucose trends, not for respiration.

**Branch point:** if the sampling rate is below 2 × the highest frequency, the signal aliases. A 3 Hz respiration signal sampled at 5 Hz has a Nyquist of 2.5 Hz and aliases. A 500 Hz EMG sampled at 1,000 Hz is adequate; sampled at 500 Hz, the 250-500 Hz band aliases.

**Edge case:** CGM has no meaningful frequency content above 0.01 Hz. Filtering it is usually harmful because it distorts the timing of excursions. The lag model is the correct tool, not a low-pass filter.

### Step 4. Detect breaths and compute respiratory rate

**Input:** the filtered effort or airflow signal.

**Output:** breath times, a rate, and a regularity measure.

**Rationale:** the respiratory rate is a count per unit time. The count depends on the detection method, and the methods disagree on asymmetric or noisy waveforms.

**Methods:**

- **Peak detection** finds inspiratory peaks. It works when the waveform is clean and the peaks are distinct.
- **Autocorrelation** finds the periodicity. It is robust to waveform shape and noise, but it assumes a stable rate over the window.
- **Zero-crossing counting** counts mean crossings. It fails on asymmetric waveforms because one breath can cross the mean more than once.
- **Spectral peak** finds the dominant frequency in the 0.1-1 Hz band. It is robust for periodic breathing but poor for irregular breathing.

**Arithmetic:** for a 15 breaths/min signal, the period is 4 s. A 60 s window contains 15 breaths. Autocorrelation over lags from 2 s to 10 s finds the peak at 4 s. A zero-crossing counter on an asymmetric waveform can count 2 or 3 crossings per breath, giving 30 to 45 breaths/min.

**Branch point:** if the breathing is irregular, autocorrelation over a long window smears the rate. Use a shorter window and report the variability.

**Edge case:** a breath can be missed if the amplitude falls below the threshold during a sigh or a pause. The rate then falls, and the apnea detector may fire on a normal pause.

**Choosing the window.** Autocorrelation needs at least three cycles to identify a period. At 15 breaths/min, three cycles is 12 s; at 6 breaths/min, it is 30 s. A 60 s window contains 15 cycles at 15 breaths/min and 6 at 6 breaths/min. A 10 s window contains 2.5 cycles at 15 and 1 at 6, which is too few.

The window also sets the time resolution of the rate. A 60 s window reports one rate per minute; a 15 s window reports one per 15 s but with fewer cycles. For a trend, a 30 s window with a 15 s step is a reasonable compromise. The window and the step must be stated, because they change the rate during transitions.

**Cardiac oscillation on the belt.** A chest belt picks up the heartbeat as a small oscillation at the heart rate. At 60 bpm, that is 1 Hz, close to the upper edge of the respiratory band. A band-pass that includes 1 Hz will pass the cardiac component, and a peak detector may count it as a breath. A low-pass at 0.6 Hz removes the cardiac component but also removes fast breathing above 36 breaths/min. The filter choice is a trade-off between cardiac artifact and tachypnea.

### Step 5. Assess artifact and quality

**Input:** the signal, the auxiliary channels, and the device status.

**Output:** a quality mask per segment.

**Rationale:** each signal has a characteristic artifact.

| Signal | Artifact | Signature | Mitigation |
|---|---|---|---|
| Respiration effort | Belt displacement | Slow drift or step | Re-baseline; flag |
| Airflow | Speech, swallowing | High-frequency bursts | Exclude; annotate |
| EMG | Motion, electrode pop | Broadband or step | Exclude; impedance check |
| EDA | Contact loss | Step or saturation | Exclude; impedance check |
| CGM | Compression low | Rapid fall and rise | Flag; compare with fingerstick |
| CGM | Sensor lag | Delay after meal | Model; do not filter |

**Branch point:** if the artifact overlaps the event, the event is uninterpretable. If it is adjacent, exclude a guard band.

**Edge case:** a compression low is a real signal from the sensor's perspective; the sensor is measuring a real change in local perfusion. It is not noise to be filtered; it is a physiological artifact that must be identified by its rapid, isolated shape and its coincidence with pressure on the sensor.

### Step 6. Score apnea and hypopnea

**Input:** airflow, effort, oxygen saturation, and sleep stage.

**Output:** apnea and hypopnea events, an AHI, and an ODI.

**Rationale:** the AASM definitions are the reference.

- **Apnea**: 90% or greater reduction in airflow for at least 10 s.
- **Hypopnea**: 30% or greater reduction for at least 10 s with a 3% or greater desaturation or an arousal. The 4% desaturation version is optional in V3.
- **Obstructive apnea**: apnea with continued or increased effort.
- **Central apnea**: apnea without effort.

**Arithmetic:** an AHI of 15 events/h over 7 h of sleep is 105 events. The same 105 events over 8 h in bed is 13.1 events/h. The denominator changes the severity category.

**Branch point:** if the desaturation criterion is 4% instead of 3%, the hypopnea count falls. The V3 manual makes the 4% version optional, which means two labs can produce different AHIs from the same recording.

**Edge case:** a hypopnea requires a desaturation or an arousal. A 30% flow reduction without either is not a hypopnea under the rule, even if it looks like one.

**Baseline and amplitude.** The 90% and 30% criteria are relative to a baseline breathing amplitude. The baseline is not a fixed number; it is the recent stable breathing, and the AASM V3 notes describe how to determine it. A detector that uses the maximum amplitude of the night as the baseline will under-score events in a subject whose breathing varies; a detector that uses a rolling median will track the changes.

The baseline arithmetic matters. If the baseline is 100 units and the event amplitude is 10 units, the reduction is 90%. If the baseline is 100 and the event is 30, the reduction is 70%, which is an apnea under a 50% rule but a hypopnea under the AASM rule. The same waveform is classified differently by different thresholds.

**Obstructive versus central.** The effort signal distinguishes the two. During an obstructive apnea, airflow stops but effort continues or increases; during a central apnea, both stop. A detector that only sees airflow cannot make the distinction and should not claim it.

The classification requires a second channel and a temporal alignment. A 10 s airflow pause with a simultaneous effort pause is central; the same pause with continued effort is obstructive. A misalignment of even 2 s to 3 s can blur the distinction at the boundaries.

### Step 7. Compute the EMG envelope

**Input:** the filtered EMG.

**Output:** an envelope and activation intervals.

**Rationale:** raw EMG oscillates at the motor unit frequency; the envelope is the smoothed amplitude. The envelope is the feature that correlates with activation, but not linearly with force.

**Method:** band-pass 20 Hz to 450 Hz, rectify, and smooth with a moving average or a low-pass at 5 Hz to 10 Hz. The smoothing window determines the temporal resolution and the peak amplitude.

**Arithmetic:** a 100 ms smoothing window at 1,000 samples/s is 100 samples. It preserves activation onsets to about 100 ms and reduces the envelope variance by √100 = 10 if the samples are independent. A 500 ms window reduces variance more but smears brief contractions.

**Branch point:** if the decision is onset detection, use a short window. If the decision is fatigue, use a long window and the median frequency.

**Edge case:** the envelope depends on the electrode gain, which depends on the electrode placement and the skin impedance. A normalized envelope (relative to a maximal voluntary contraction) is comparable across sessions; a raw envelope is not.

### Step 8. Decompose EDA

**Input:** the skin conductance signal.

**Output:** a tonic level (SCL) and phasic responses (SCRs).

**Rationale:** EDA has a slow drift and fast responses. The decomposition separates them, but it is not unique: many tonic-phasic splits fit the same data.

**Methods:** a low-pass filter for the tonic component and a high-pass for the phasic; or a model such as continuous decomposition analysis, which fits a bi-exponential response shape.

**Arithmetic:** a phasic response rises in 1 s to 3 s and decays with a time constant of 2 s to 10 s. A 1 µS response is large; 0.1 µS is small. The response rate is the number of responses per minute above a threshold.

**Branch point:** if the decision is a group comparison, use a standardized decomposition and report the parameters. If the decision is an individual alert, validate the threshold on the individual.

**Edge case:** a step from contact loss can be decomposed into a large tonic change and a false phasic response. The decomposition does not know the difference; the quality check must.

**EDA response arithmetic.** A response is counted if it exceeds a threshold, usually 0.01 µS to 0.05 µS, and if it is separated from the previous response by a minimum interval, often 1 s to 2 s. The threshold determines the count. A lower threshold counts more noise; a higher threshold misses small responses.

The response rate is the count divided by the time. A rate of 5 responses/min over 10 min is 50 responses. A rate of 5 over 2 min is 10 responses. The denominator is the analysed time, not the recording time, and the analysed time excludes the artifacts.

**Tonic versus phasic is a model, not a measurement.** Two decompositions can produce different SCL and SCR values from the same signal. The choice should be reported with the parameters, and the comparison across studies should use the same method. A "stress score" that is a weighted combination of SCL and SCR inherits the ambiguity of both.

### Step 9. Calibrate CGM and model the lag

**Input:** the raw sensor signal, the fingerstick calibrations, and the time stamps.

**Output:** a glucose series with a lag model and a calibration uncertainty.

**Rationale:** CGM measures interstitial glucose. The calibration maps the sensor current to glucose, and the lag maps interstitial to blood. Both are necessary for a blood-glucose claim.

**Arithmetic:** a lag of 10 min at a glucose rate of change of 2 mg/dL/min is a 20 mg/dL difference between interstitial and blood. During a rapid fall, the CGM reads higher than blood; during a rapid rise, it reads lower.

**Branch point:** if the decision is a trend, the lag is less important than the calibration. If the decision is a hypoglycemia alert, the lag is critical and a predictive algorithm is needed.

**Edge case:** a calibration performed during a rapid change is biased because the fingerstick and the interstitial value are not at the same glucose. Calibrate during steady state.

### Step 10. Compute CGM metrics

**Input:** the calibrated series and the coverage.

**Output:** mean glucose, SD, CV, TIR, TBR, TAR, GMI, and the coverage.

**Rationale:** the consensus metrics are defined on a 14-day record with at least 70% coverage. Each metric has a threshold and a clinical meaning.

**Arithmetic:** with 5-minute sampling, 14 days is 4,032 readings. TIR is the fraction of readings between 70 mg/dL and 180 mg/dL, multiplied by 100. TBR is the fraction below 70 mg/dL, with a sub-threshold at 54 mg/dL. TAR is the fraction above 180 mg/dL, with a sub-threshold at 250 mg/dL. CV is SD/mean × 100. GMI is 3.31 + 0.02392 × mean glucose in mg/dL.

**Branch point:** if the coverage is below 70%, the metric is not representative. Report the coverage and consider a sensitivity analysis.

**Edge case:** a series with a long gap has a denominator problem. If the gap is excluded, the metric describes the observed periods. If the gap is imputed, the metric depends on the imputation. Report both.

**The metric definitions in full.** The consensus metrics are defined on the glucose series, not on the sensor signal.

| Metric | Definition | Target for most adults |
|---|---|---|
| TIR | % readings 70-180 mg/dL | >70% |
| TBR | % readings <70 mg/dL | <4% |
| TBR severe | % readings <54 mg/dL | <1% |
| TAR | % readings >180 mg/dL | <25% |
| TAR severe | % readings >250 mg/dL | <5% |
| CV | SD/mean × 100 | ≤36% |
| GMI | 3.31 + 0.02392 × mean mg/dL | individual |
| Coverage | readings present / expected | ≥70% over 14 days |

The targets are for most adults with type 1 or type 2 diabetes. They differ for older or high-risk adults, pregnancy, and children. A metric without the population is incomplete.

**GMI is not A1C.** The glucose management indicator is a linear transformation of mean glucose. It estimates what an A1C would be if the mean were representative, but it is not a laboratory A1C and it does not reflect haemoglobin variants or red-cell lifespan. A report that calls GMI "A1C" overstates the measurement.

### Step 11. Aggregate and handle missingness

**Input:** the segment-level or day-level metrics.

**Output:** a summary with a denominator and a missingness statement.

**Rationale:** the aggregation rule changes the result. A mean of daily TIRs is not the same as a TIR over all days if the days have different coverage. A median of hourly rates is not the same as a rate over the night.

**Arithmetic:** if day 1 has 90% coverage and TIR 80%, and day 2 has 50% coverage and TIR 60%, the sample-weighted TIR over the two days is (0.9×0.8 + 0.5×0.6)/(0.9+0.5) = 0.729, while the simple mean of the daily TIRs is 0.70. The difference is small here but can be larger.

**Branch point:** if the missingness is related to the outcome, complete-case analysis is biased. Compare complete-case and imputed estimates.

**Edge case:** a sensor that fails at night misses the nocturnal hypoglycemia that matters most. The missingness is informative.

**A missingness taxonomy.** Missingness in these signals has four mechanisms, and only one is ignorable.

- **Missing completely at random**: a random sensor dropout unrelated to the state. Complete-case analysis is unbiased, though less precise.
- **Missing at random**: dropout related to an observed variable, such as activity. Weighting or imputation can reduce bias.
- **Missing not at random**: dropout related to the unobserved state, such as a sensor that fails during sweating or pressure. Complete-case analysis is biased, and the direction is often predictable.
- **Structural missingness**: the sensor does not measure the quantity at certain times, such as a CGM that is not worn at night. The missingness is part of the design.

A report should name the mechanism and the handling. "Missing data were excluded" is not a method; it is an omission.

The handling depends on the mechanism. Complete-case analysis is acceptable for missing completely at random, with a loss of precision. Weighting or multiple imputation is needed for missing at random. For missing not at random, no statistical method recovers the unobserved values; the best defence is a sensitivity analysis that bounds the possible effect.

A useful sensitivity analysis is a worst-case bound: assign all missing values to the lowest category and recompute; then assign them to the highest and recompute. The two values bracket the true metric under any assumption. If the bracket is narrow enough for the decision, the missingness is not decisive. If it is wide, the metric is not robust and the report should say so.

**Coverage by period.** A single coverage percentage hides the distribution. A 70% coverage overall can be 100% during the day and 0% at night. The period-specific coverage is the relevant number for a nocturnal metric.

### Step 12. Validate and report

**Input:** the metrics and the reference.

**Output:** a validation report with agreement, coverage, and subgroup performance.

**Rationale:** each signal has a reference: polysomnography for apnea, a metabolic cart for respiration, a dynamometer for force, a blood glucose meter or laboratory for CGM, and a validated task for EDA. The reference has its own error.

**Metrics:** AHI agreement (Bland-Altman, kappa for severity categories), respiratory rate bias and limits of agreement, CGM MARD and TIR agreement, EDA response concordance.

**Branch point:** if the reference is a fingerstick, the CGM error includes both the sensor error and the meter error. A study that reports MARD against a meter with 15% error is reporting a combined error.

**Edge case:** a validation in a sleep laboratory does not transfer to home use. The home environment has different artifacts, different adherence, and different sleep.

**A reporting checklist.** A report for any of these signals should contain the following.

| Item | Respiration | EMG | EDA | CGM |
|---|---|---|---|---|
| Sensor and site | belt, chest | electrodes, muscle | electrodes, palm | filament, arm |
| Sampling rate | 25 Hz | 1,000 Hz | 4 Hz | 1/300 Hz |
| Filter | 0.05-3 Hz | 20-450 Hz | 0-5 Hz | none |
| Event definition | breath/apnea | contraction | response | excursion |
| Denominator | valid time | recording time | analysed time | sensor active time |
| Missingness | % valid | % valid | % valid | % coverage |
| Reference | PSG | dynamometer | task | blood glucose |

A report that omits the denominator or the missingness is not comparable with another report. The table is a compact way to check both.

### Decision table for missing or invalid data

| Signal | Condition | Detect by | Action |
|---|---|---|---|
| Respiration | Belt displacement | Step or drift | Re-baseline or reject |
| Respiration | Speech | High-frequency burst | Exclude segment |
| Respiration | Apnea | 90% flow reduction ≥10 s | Score event |
| EMG | Electrode pop | Step | Reject channel |
| EMG | Motion | Broadband burst | Reject segment |
| EDA | Contact loss | Step to rail | Reject segment |
| EDA | Saturation | Signal at rail | Reject segment |
| CGM | Compression low | Rapid fall and rise | Flag; confirm |
| CGM | Sensor lag | Delay after meal | Model; do not filter |
| CGM | Gap | Missing reading | Report coverage |
| CGM | Calibration drift | Rising error | Recalibrate |

## 6. Worked example

### 6.1 The case

A 60 s respiratory effort signal is recorded at 25 samples/s from a chest belt. The subject breathes at 15 breaths/min. A 14-day CGM record is sampled every 5 min. The analysis goals are the respiratory rate and the CGM metrics. The CGM record includes a nocturnal hypoglycemia pattern every other day, a 3 h gap, and a small number of held-over values.

### 6.2 Respiration arithmetic

- Samples: 25 samples/s × 60 s = 1,500.
- Breaths at 15/min: 15 breaths in 60 s, period 4 s.
- Autocorrelation lag range: 2 s to 10 s, or 50 to 250 samples.
- The autocorrelation peak is at 4.000 s, so the rate is 60/4.000 = 15.00 breaths/min.
- The zero-crossing counter returns 46.00 breaths/min, which is wrong. The asymmetric waveform crosses the mean more than once per breath.

The zero-crossing error is a factor of 3.07. A rate of 46 breaths/min is physiologically implausible at rest and would trigger a deterioration alert in a monitoring system. The diagnostic is the ratio of the zero-crossing rate to the autocorrelation rate; a ratio far from 1 indicates an asymmetric or noisy waveform.

At 1 sample/s, the autocorrelation still finds 4.000 s and 15.00 breaths/min. The rate is robust to sampling because the period is long relative to the sample interval. The waveform morphology is not robust: a 4 s breath is 4 samples, which cannot show the inspiration-expiration asymmetry.

The two methods disagree by a factor that depends on the waveform shape.

| Waveform | Crossings per breath | Zero-cross rate at 15/min | Autocorrelation rate |
|---|---:|---:|---:|
| Symmetric sine | 2 | 15 | 15 |
| Asymmetric (this case) | 6 | 46 | 15 |
| Noisy symmetric | 4-6 | 30-45 | 15 |
| With cardiac oscillation | 4-8 | 30-60 | 15 |

The table is a warning: the zero-crossing rate is not a rate estimate; it is a crossing count. The autocorrelation is a rate estimate because it finds the period, and its accuracy depends on how many cycles the window contains.

The confidence interval of the rate depends on the window. A 60 s window with 15 cycles has a period estimate with a relative standard error of about 1/√15 = 26% of the cycle-to-cycle variation. For a breath-to-breath variation of 0.2 s, the standard error is about 0.05 s, and the rate standard error is about 0.2 breaths/min. A 15 s window with 3.75 cycles has a standard error about twice as large. The autocorrelation is a rate estimate because it finds the period. The distinction is fundamental and easy to miss when the code is a one-line counter.

### 6.3 CGM arithmetic

- Readings: 14 days × 24 h × 12 readings/h = 4,032.
- Mean glucose: 147.1 mg/dL.
- SD: 37.2 mg/dL.
- CV: 37.2/147.1 × 100 = 25.3%.
- TIR (70-180 mg/dL): 73.4%.
- TBR (<70 mg/dL): 3.1%.
- TBR (<54 mg/dL): 1.5%.
- TAR (>180 mg/dL): 23.5%.
- TAR (>250 mg/dL): 0.0%.
- GMI: 3.31 + 0.02392 × 147.1 = 6.83%.

The consensus targets for most adults with type 1 or type 2 diabetes are TIR above 70%, TBR below 4%, TBR below 54 mg/dL below 1%, TAR below 25%, TAR above 250 mg/dL below 5%, and CV at or below 36%. The record meets TIR, TBR, TAR, and CV, and fails the TBR below 54 mg/dL target at 1.5%.

The failure is clinically important because the below-54 threshold is the level at which cognitive impairment and severe hypoglycemia risk rise. A TIR-focused report that does not show the sub-threshold would miss it.

### 6.4 Coverage and denominator arithmetic

The 14-day record has 4,032 expected readings. A 3 h gap removes 36 readings, leaving 3,996, or 99.1% coverage. The TIR on the reduced record is 73.2%, compared with 73.4% on the full record. The gap is small and the effect is small.

A 10-day subset has 2,880 readings and a TIR of 73.6%. The difference from the 14-day value is 0.2 percentage points. In this synthetic record, 10 days is enough because the pattern is stable. In a real record with a weekly cycle, 10 days can miss a weekend pattern.

The consensus recommendation of at least 14 days with at least 70% coverage is a minimum, not a guarantee. A record that meets the minimum can still be unrepresentative if the missing 30% is not random.

The coverage arithmetic is easy to get wrong. If the sensor is active for 14 days but the user does not wear it at night, the coverage by calendar time is high and the coverage by physiological time is low. The metric that matters is the coverage of the periods the decision cares about. A nocturnal hypoglycemia metric with no night data has no coverage, regardless of the total.

A sensitivity analysis is the minimum defence. Compute the metric with all available data, with complete days only, and with the missing periods imputed under two assumptions. If the three values agree, the missingness is not driving the conclusion. If they disagree, report the range.

### 6.5 Diagnostic 1: the zero-crossing rate

The zero-crossing rate is 46.00 breaths/min and the autocorrelation rate is 15.00. The ratio is 3.07. The diagnostic is to plot the waveform and count the crossings by hand for a few breaths; the asymmetry is visible.

The consequence of the error is a false tachypnea alert. A respiratory rate above 30 breaths/min is a deterioration criterion in many early-warning scores. A belt that is slightly asymmetric can push a normal 15 into a pathological 46, and the alert is generated by the algorithm, not the patient.

The fix is to use a method that is robust to waveform shape, or to count only the positive-going crossings with a refractory period. A refractory period of 1.5 s at 15 breaths/min would prevent multiple counts per breath, but it would also cap the measurable rate at 40 breaths/min.

A decision table for the method choice:

| Breathing pattern | Best method | Why |
|---|---|---|
| Regular, clean | Peak detection | Simple, accurate |
| Regular, asymmetric | Autocorrelation | Robust to shape |
| Irregular, short window | Peak detection with adaptive threshold | Local in time |
| Noisy | Autocorrelation or spectral peak | Averages noise |
| Very slow (<6/min) | Long-window autocorrelation | Needs multiple cycles |
| Very fast (>30/min) | Peak detection with short refractory | Autocorrelation window too short |

The table is not exhaustive, but it shows that the method is a function of the pattern, not a universal choice. A product that uses one method for all patterns will fail on at least one.

### 6.6 Diagnostic 2: the daytime and nighttime TIR

The overall TIR is 73.4%, which meets the target. The daytime TIR (06:00 to 22:00) is 64.8%, and the nighttime TIR (22:00 to 06:00) is 90.7%. The overall value hides a clinically meaningful difference: the daytime is below target, and the nighttime is above target because there are no meals.

A clinician who sees only the overall TIR might conclude that the glycemic control is adequate. A clinician who sees the split might adjust the daytime insulin or the meal plan. The split is a more actionable measurement than the aggregate.

The arithmetic of the split is simple: the daytime and nighttime fractions are computed with their own denominators, and the overall TIR is the weighted average. If the daytime is 16 h and the nighttime 8 h, the overall TIR is (16×64.8 + 8×90.7)/24 = 73.4%, which matches.

The split table is the actionable output:

| Period | Hours | TIR | TBR | TAR |
|---|---:|---:|---:|---:|
| Daytime (06-22) | 16 | 64.8% | 0.0% | 35.2% |
| Nighttime (22-06) | 8 | 90.7% | 9.3% | 0.0% |
| Overall | 24 | 73.4% | 3.1% | 23.5% |

The nighttime has a higher TIR but a higher TBR, because the hypoglycemia occurs at 03:00. The daytime has a lower TIR because of postprandial excursions. The overall TIR describes neither period well. This is the aggregation problem in its clearest form.

### 6.7 Diagnostic 3: held-over values

The record contains 96 repeated consecutive values, or 2.38% of the readings. A CGM that holds the last value when the sensor is not updating produces a flat segment that looks like stable glucose. The held values inflate the time in range if the held value is in range, and they hide excursions.

The diagnostic is to count consecutive identical values. A real CGM has noise, so exact repeats are rare. A run of two or more identical values is a likely hold. The count of 2.38% is small, but a longer hold during a rapid change can distort the TIR.

A second diagnostic is the derivative distribution. A real CGM has a smooth derivative with a characteristic scale; a held signal has runs of zero derivative followed by jumps. A histogram of the first difference shows a spike at zero and a heavy tail. The same check catches a sensor that is stuttering rather than holding.

### 6.8 Varying one assumption

Suppose the TIR range is 70-180 mg/dL, as in the consensus, versus 80-180 mg/dL, as some older targets use. Narrowing the lower bound from 70 to 80 moves readings between 70 and 80 out of range. If 4% of the readings are in that band, the TIR falls from 73.4% to about 69.4%, below the 70% target. The range is a convention, and the conclusion depends on it.

Suppose the recording is 7 days instead of 14. The TIR may be similar, but the confidence interval is wider. If the day-to-day TIR varies with an SD of 5 percentage points, the standard error of a 7-day mean is 5/√7 = 1.9 points, and the standard error of a 14-day mean is 5/√14 = 1.3 points. The shorter record is less precise.

Suppose the CGM lag is 10 min and the glucose is falling at 2 mg/dL/min. The interstitial value is 20 mg/dL higher than the blood value. A reading of 75 mg/dL corresponds to a blood glucose of 55 mg/dL, which is below the 54 mg/dL threshold. The lag turns a borderline reading into a severe hypoglycemia that the alert misses.

The lag arithmetic is a useful design check. At a fall rate of 2 mg/dL/min, a 10 min lag is 20 mg/dL; at 4 mg/dL/min, it is 40 mg/dL. A device that alarms at 70 mg/dL with a 10 min lag will not alarm until the blood glucose is already 50 mg/dL if the fall is rapid. A predictive alarm that projects the trend 15 min ahead can compensate, at the cost of alarming on a projected low that may not occur.

### 6.9 Conclusion of the case

The respiratory rate is 15.00 breaths/min by autocorrelation and 46.00 by zero-crossing counting; the latter is an artifact of the asymmetric waveform. The CGM record has a TIR of 73.4%, a TBR of 3.1%, a TBR below 54 mg/dL of 1.5%, a TAR of 23.5%, a CV of 25.3%, and a GMI of 6.83%. It meets most consensus targets and fails the severe hypoglycemia target.

The three diagnostics (zero-crossing ratio, day-night split, held-over count) catch the errors that the aggregate metrics hide. The assumption that matters most is the event definition: the TIR range and the apnea criteria change the conclusion more than the sensor does.

### 6.9.1 The metric table

| Metric | Value | Target | Status |
|---|---:|---|---|
| Mean glucose | 147.1 mg/dL | individual | — |
| SD | 37.2 mg/dL | — | — |
| CV | 25.3% | ≤36% | meets |
| TIR | 73.4% | >70% | meets |
| TBR | 3.1% | <4% | meets |
| TBR <54 | 1.5% | <1% | fails |
| TAR | 23.5% | <25% | meets |
| TAR >250 | 0.0% | <5% | meets |
| GMI | 6.83% | individual | — |

The table is the minimum a clinician needs to interpret the record. A single TIR would hide the severe hypoglycemia and the day-night split.

### 6.10 What the case does not show

The case is synthetic and single-channel per modality. It does not show:

- **Apnea scoring.** There is no airflow, effort, oximetry, or sleep stage, so no AHI is computed.
- **EMG and EDA.** The code covers respiration and CGM only; the EMG and EDA steps are described but not executed.
- **Real CGM noise.** The synthetic CGM has Gaussian noise and a simple hypo pattern, not the drift, compression lows, and calibration errors of a real sensor.
- **Population variation.** The TIR targets differ for pregnancy, older adults, and children.
- **Wearable respiration.** The belt signal is clean and stationary; a wrist-derived respiration signal is neither.

The omissions are the agenda for a validation. The arithmetic transfers; the thresholds and the signal quality do not.

### 6.11 EMG and EDA arithmetic

The code does not execute EMG or EDA, but the arithmetic is simple enough to state.

**EMG envelope.** A surface EMG is sampled at 1,000 Hz and band-passed from 20 Hz to 450 Hz. The signal is rectified and smoothed with a 100 ms moving average, which is 100 samples. If the samples were independent, the smoothing would reduce the standard deviation by √100 = 10. The samples are not independent because the EMG is correlated over a few milliseconds, so the effective reduction is smaller, perhaps √20 = 4.5. The envelope is therefore smoother than the raw signal but still variable.

The envelope amplitude is in microvolts and depends on the electrode gain. A contraction that produces 50 µV at one placement may produce 80 µV at another. Normalizing to a maximal voluntary contraction (MVC) gives a percentage that is more comparable across sessions, but the MVC itself depends on effort and motivation.

**EDA response rate.** A skin conductance signal is sampled at 4 Hz. A phasic response rises over 1 s to 3 s and decays with a time constant of 2 s to 10 s. A response is counted if its amplitude exceeds 0.05 µS and if it is separated from the previous response by at least 1 s. Over a 10 min recording with 12 responses, the rate is 1.2 responses/min. If 2 min are excluded for artifact, the denominator is 8 min and the rate is 1.5 responses/min.

The difference between 1.2 and 1.5 is a 25% change produced entirely by the artifact exclusion. The denominator is part of the metric, and the excluded time must be reported.

**A combined table.**

| Signal | Sample rate | Feature | Window | Unit |
|---|---:|---|---:|---|
| Respiration | 25 Hz | Rate | 60 s | breaths/min |
| EMG | 1,000 Hz | Envelope | 100 ms | µV or % MVC |
| EDA | 4 Hz | Response rate | 10 min | /min |
| CGM | 1/300 Hz | TIR | 14 days | % |

The table makes the scale difference explicit: the CGM sampling interval is 300 s, while the EMG sampling interval is 1 ms. A pipeline that treats them with the same window logic will fail on one of them.

## 7. Code, query, or calculation

The script computes respiratory rate by autocorrelation and by zero-crossing counting, and computes CGM metrics including TIR, TBR, TAR, CV, and GMI. It uses only the standard library.

```python
import math, random

FS, DUR, RR = 25.0, 60.0, 15.0
N = int(FS*DUR)

def synth_resp(fs, dur, rr, seed=1):
    rnd = random.Random(seed); n = int(fs*dur); x=[]
    period = 60.0/rr
    for i in range(n):
        t=i/fs
        phase = (t % period)/period
        v = math.sin(2*math.pi*phase)**3 if phase < 0.5 else -0.7*math.sin(2*math.pi*(phase-0.5))**2
        v *= (1.0 + 0.1*math.sin(2*math.pi*t/20.0))
        x.append(v + rnd.gauss(0,0.03))
    return x

def autocorr_rr(x, fs, lo=2.0, hi=10.0):
    n=len(x); mean=sum(x)/n; xc=[v-mean for v in x]
    best_lag, best_val = None, -1e9
    for lag in range(int(lo*fs), int(hi*fs)+1):
        s=0.0
        for i in range(n-lag):
            s += xc[i]*xc[i+lag]
        s /= (n-lag)
        if s > best_val:
            best_val, best_lag = s, lag
    return best_lag/fs, best_val

def zero_cross_rate(x, fs):
    m=sum(x)/len(x); c=0
    for i in range(1,len(x)):
        if x[i-1] < m <= x[i]: c+=1
    return c/(len(x)/fs)*60

x = synth_resp(FS,DUR,RR)
period, ac = autocorr_rr(x, FS)
print("resp: period=%.3f_s RR=%.2f/min zero_cross_RR=%.2f"
      % (period, 60/period, zero_cross_rate(x,FS)))

fs2=1.0
x2=[x[int(i*FS/fs2)] for i in range(int(DUR*fs2))]
p2,ac2=autocorr_rr(x2, fs2, 2.0, 10.0)
print("resp_1Hz: period=%.3f RR=%.2f" % (p2, 60/p2))

DAYS = 14
SAMPLES = DAYS*24*12

def synth_cgm(days, seed=2):
    rnd=random.Random(seed); n=days*24*12; g=[]
    for i in range(n):
        h=(i%288)/12.0
        base=120 + 15*math.sin(2*math.pi*(h-5)/24)
        meal=0
        for (t,a,w) in ((8,70,1.2),(13,60,1.5),(19,80,1.4)):
            meal += a*math.exp(-0.5*((h-t)/w)**2)
        day=(i//288)
        hypo = -65*math.exp(-0.5*((h-3.0)/0.8)**2) if day%2==0 else 0
        g.append(base+meal+hypo+rnd.gauss(0,8))
    return g

def cgm_metrics(g):
    n=len(g); mean=sum(g)/n
    var=sum((v-mean)**2 for v in g)/(n-1); sd=math.sqrt(var)
    tir=sum(1 for v in g if 70<=v<=180)/n*100
    tbr=sum(1 for v in g if v<70)/n*100
    tbr54=sum(1 for v in g if v<54)/n*100
    tar=sum(1 for v in g if v>180)/n*100
    tar250=sum(1 for v in g if v>250)/n*100
    gmi=3.31+0.02392*mean
    return mean,sd,sd/mean*100,tir,tbr,tbr54,tar,tar250,gmi

g=synth_cgm(DAYS)
for k in range(500, len(g), 37):
    g[k]=g[k-1]

m,sd,cv,tir,tbr,tbr54,tar,tar250,gmi=cgm_metrics(g)
print("cgm: n=%d mean=%.1f sd=%.1f cv=%.1f TIR=%.1f TBR=%.1f TBR54=%.1f TAR=%.1f TAR250=%.1f GMI=%.2f"
      % (len(g),m,sd,cv,tir,tbr,tbr54,tar,tar250,gmi))

g10=g[:10*288]
m10,_,_,tir10,tbr10,_,tar10,_,gmi10=cgm_metrics(g10)
print("cgm_10d: TIR=%.1f TBR=%.1f TAR=%.1f GMI=%.2f" % (tir10,tbr10,tar10,gmi10))

rep=sum(1 for i in range(1,len(g)) if g[i]==g[i-1])
print("repeated_values=%d (%.2f%%)" % (rep, rep/len(g)*100))

day=[v for i,v in enumerate(g) if 6 <= (i%288)/12 < 22]
night=[v for i,v in enumerate(g) if not (6 <= (i%288)/12 < 22)]
print("TIR_day=%.1f TIR_night=%.1f"
      % (sum(1 for v in day if 70<=v<=180)/len(day)*100,
         sum(1 for v in night if 70<=v<=180)/len(night)*100))

g_gap=[v for i,v in enumerate(g) if not (i>=2000 and i<2036)]
print("gap_removed_n=%d TIR=%.2f"
      % (len(g_gap), sum(1 for v in g_gap if 70<=v<=180)/len(g_gap)*100))
```

Expected output:

```text
resp: period=4.000_s RR=15.00/min zero_cross_RR=46.00
resp_1Hz: period=4.000 RR=15.00
cgm: n=4032 mean=147.1 sd=37.2 cv=25.3 TIR=73.4 TBR=3.1 TBR54=1.5 TAR=23.5 TAR250=0.0 GMI=6.83
cgm_10d: TIR=73.6 TBR=3.1 TAR=23.3 GMI=6.83
repeated_values=96 (2.38%)
TIR_day=64.8 TIR_night=90.7
gap_removed_n=3996 TIR=73.20
```

Four implementation notes.

The zero-crossing counter counts positive-going crossings of the mean, not all crossings. Even so, the asymmetric waveform crosses the mean three times per breath in this synthetic signal, which is why the rate is roughly three times the true rate. A real respiratory waveform can also cross multiple times because of cardiac oscillation on the belt, which is another source of overcounting.

The autocorrelation is computed on the mean-removed signal and normalized by the number of overlapping samples. It finds the period, not the phase. The rate is 60/period. The method assumes a stable period over the 60 s window; a subject who changes rate during the window will have a broad or split peak.

The CGM synthesis uses a 5-minute sampling interval, which is 12 readings per hour. The 14-day record is 4,032 readings. The held-over values are injected every 37 readings, which produces 96 repeats and a 2.38% share.

The TIR is a simple sample proportion. Because the sampling is uniform, the sample-weighted and time-weighted values are the same. If the sampling were irregular, the time-weighted value would require the interval between readings, and the simple proportion would be biased.

## 8. Data, standards, and regulatory context

The standards for these signals are a mix of clinical scoring rules, device standards, and consensus targets. None of them is a single binding document for the whole chain.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| AASM Manual V3 (2023) | Sleep and respiratory event scoring | Professional; required for accredited centres | Apnea and hypopnea definitions; 30 s epochs |
| AASM V3 update summaries | Changes to hypopnea desaturation rules | Professional | 4% desaturation is optional in V3 |
| International Consensus on TIR (2019) | CGM metrics and targets | Consensus; widely adopted | TIR 70-180 mg/dL; >70% target |
| ISO 15197 | Blood glucose meter accuracy | Binding when claimed | Meter error, not CGM error |
| CLSI POCT05 | CGM performance evaluation | Consensus | Accuracy and lag methods |
| FDA CGM guidance | Premarket CGM claims | Guidance | Accuracy, alarms, labelling |
| IEC 60601-2-26 | EEG equipment (adjacent) | Binding when claimed | Not specific to respiration |
| AASM HSAT rules | Home sleep apnea testing | Professional | Reduced-channel scoring |
| HIPAA | De-identification | Binding in the US | Dates and identifiers |
| EU MDR | Clinical evaluation | Binding in the EU | Post-market surveillance |

### 8.1 AASM V3 and the hypopnea change

The AASM Manual Version 3 was released in February 2023 and required implementation by accredited facilities by 31 December 2023. The change that matters most for data science is the reclassification of the 4% desaturation hypopnea rule from acceptable to optional.

The practical consequence is that two accredited labs can produce different AHIs from the same recording. A lab that uses the 3% rule will score more hypopneas than a lab that uses the 4% rule. A model trained on one lab's labels and applied to another's will see a label shift that is not a physiological difference.

A dataset card for a sleep study should record the AASM version, the hypopnea rule, the desaturation threshold, and the arousal rule. Without these, the AHI is not comparable across studies.

### 8.2 The TIR consensus

The international consensus on time in range was published in 2019 and has been widely adopted. The key targets for most adults with type 1 or type 2 diabetes are:

- TIR 70-180 mg/dL: more than 70% of readings.
- TBR below 70 mg/dL: less than 4%.
- TBR below 54 mg/dL: less than 1%.
- TAR above 180 mg/dL: less than 25%.
- TAR above 250 mg/dL: less than 5%.
- CV: 36% or less.
- Data: at least 14 days with at least 70% sensor active time.

The consensus also notes that each 5% increase in TIR is clinically meaningful, that a 10% increase corresponds to about a 0.5% reduction in A1C, and that a TIR of 70% corresponds approximately to an A1C of 7%. These are population-level approximations, not individual predictions.

### 8.2.1 CGM claims and the evidence they need

| Claim | Quantity | Reference | Evidence |
|---|---|---|---|
| Glucose trend | Direction and rate | Blood glucose | MARD, lag, trend accuracy |
| TIR | % time 70-180 | Reference CGM or blood | Agreement by band, coverage |
| Hypoglycemia alert | Threshold crossing | Blood glucose | Sensitivity, false alarms, lag |
| Insulin dosing | Glucose value | Blood glucose | Accuracy at low range, lag |
| Time below range | % time <70 | Reference CGM | Coverage, nocturnal data |

A CGM that is accurate on average can be inaccurate at the low end, where the clinical consequence is greatest. The MARD is often reported overall, but the accuracy at 40-70 mg/dL is the number that matters for a hypoglycemia claim.

### 8.2.2 Sleep study claims and the evidence they need

| Claim | Reference | Evidence |
|---|---|---|
| Apnea severity (AHI) | Attended polysomnography | Event agreement, severity kappa |
| Respiratory rate | Capnography or plethysmography | Bias, limits of agreement |
| Sleep-wake | PSG or actigraphy | Sensitivity, specificity |
| Home apnea test | Attended PSG | AHI agreement, failure rate |

The home test has a failure rate: a study that cannot be scored because of poor signal is a missing result, not a negative one. The failure rate is part of the performance.

### 8.3 What the standards do not cover

Three gaps.

First, the AASM rules are written for attended polysomnography with airflow, effort, oximetry, and EEG. A wearable with a single respiratory effort channel cannot apply the rules as written. It can estimate a respiratory rate or a pause count, but the AHI requires the full montage.

Second, the TIR consensus assumes a CGM with a known accuracy and a known lag. A consumer device with a different accuracy or a different lag produces a TIR with a different error. The consensus does not certify the device; it defines the metric.

Third, there is no accepted standard for an EDA-based stress score. The signal is measurable, but the construct is not standardized. A product that reports a stress score is making a construct-validity claim, not an accuracy claim.

A fourth gap is the treatment of artefacts in the standards. The AASM rules assume a technologist who can see the waveform and annotate an arousal or a movement. An automated pipeline must implement an equivalent logic, and the standards do not say how. The result is that two automated systems can apply the same written rule and produce different AHIs because they handle artifacts differently.

A fifth gap is the definition of the analysis window. The TIR consensus assumes a 14-day record, but many studies report a 7-day or a 3-day TIR. The shorter windows are not necessarily wrong, but they are different measurements. A report should state the window and the coverage, and a comparison across studies should use the same window.

## 9. Industry practice and case evidence

### 9.1 CGM in clinical practice

CGM has moved from a research tool to a standard of care for many people with type 1 diabetes and is increasingly used in type 2 diabetes. The metrics that clinicians use are the ones in the consensus: TIR, TBR, TAR, CV, and GMI. The AGP (ambulatory glucose profile) report displays the median and the interquartile range by time of day, which is the day-night split in a richer form.

The documented lesson is that the metric changes behaviour. A patient who sees a TIR of 60% with a nighttime dip has a different conversation from one who sees a mean glucose of 160 mg/dL. The aggregation is the intervention.

### 9.2 Home sleep apnea testing

Home sleep apnea testing (HSAT) uses a reduced channel set, often airflow, effort, oximetry, and position, without EEG. The AHI from HSAT is an estimate because sleep time is estimated rather than measured. The AASM HSAT rules address this by allowing the test to be scored with or without recorded sleep.

The documented limitation is that HSAT underestimates the AHI when sleep time is overestimated. A device that assumes 8 h of sleep when the patient slept 6 h will divide the event count by a larger denominator and report a lower AHI.

### 9.3 Wearable respiratory rate

Consumer wearables estimate respiratory rate from PPG, accelerometry, or a combination. The estimates are usually based on respiratory sinus arrhythmia, baseline modulation, or motion. The accuracy is variable and depends on the activity and the posture.

The documented pattern is that the estimates are reasonable at rest and poor during exercise. The inference is that the claim should be limited to rest, or the validation should include exercise. A respiratory rate of 15 at rest and 30 during exercise are both normal; a device that cannot distinguish them is not measuring the same thing across conditions.

### 9.4 EDA in research and products

EDA is used in research on stress, emotion, and sleep. Consumer devices report "stress" or "arousal" from EDA, often combined with heart rate variability. The validation is usually a laboratory task with a known stressor, and the generalization to daily life is uncertain.

The documented limitation is that EDA responds to many things: temperature, humidity, movement, and cognitive load. A phasic response is not specific to stress. The inference is that an EDA-based stress score is a constructed index, and its validation is a construct-validity argument, not an agreement study.

### 9.4.1 Documented fact versus inference

| Statement | Status |
|---|---|
| The TIR consensus target is >70% for most adults | Documented in the 2019 consensus |
| The consensus recommends 14 days and ≥70% coverage | Documented in the same consensus |
| AASM V3 made the 4% hypopnea rule optional | Documented in the V3 update summary |
| AASM V3 was released in February 2023 | Documented by AASM |
| AHI is defined on sleep time, not time in bed | Documented in the AASM manual |
| A belt measures circumference, not volume | Inferred from the sensor principle |
| A compression low is a local perfusion artifact | Inferred from the sensor mechanism; described in clinical literature |
| Daytime and nighttime TIR can differ materially | Inference from meal and fasting physiology; demonstrated in Section 6 |
| A 10-day record can be representative | Inference; depends on the weekly pattern |
| An EDA stress score lacks a gold standard | Inference from the absence of a consensus reference |

The table separates what can be checked from what must be argued. A product team should test the inferences in its own population rather than importing them.

### 9.5 What the cases have in common

- **The event definition is the measurement.** AHI, TIR, and response rate are defined, not discovered.
- **The denominator is part of the metric.** Sleep time, sensor active time, and valid-breath time all change the result.
- **The sensor measures a proxy.** Belt for volume, interstitial for blood, conductance for arousal.
- **The artifact is physiological.** A compression low is a real local change; a belt slip is a real movement. The algorithm must distinguish the proxy from the quantity.

### 9.6 A comparison table

| Signal | Market maturity | Reference | Main claim risk |
|---|---|---|---|
| CGM | High | Blood glucose | Lag and low-range accuracy |
| Sleep apnea | High | Polysomnography | AHI definition and denominator |
| Respiratory rate | Medium | Capnography or belt | Artifact and activity |
| EMG | Medium | Dynamometer | Force claim |
| EDA | Low | None accepted | Construct validity |

The table is a rough map of where the evidence is strong and where it is thin. A product team should match the claim to the maturity: a CGM TIR claim is well supported by a reference; an EDA stress claim is not.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Zero-crossing overcount | Rate 2-3× expected | Asymmetric waveform | False tachypnea | Autocorrelation; refractory |
| Belt displacement | Slow drift or step | Loose belt | False apnea | Re-baseline; flag |
| Speech artifact | High-frequency bursts | Vocal cord motion | False breaths | Exclude; annotate |
| Apnea threshold wrong | AHI differs from reference | 50% versus 90% criterion | Misclassified severity | Use AASM definition |
| Hypopnea rule version | AHI differs between labs | 3% versus 4% desaturation | Label shift | Record version |
| Denominator wrong | AHI changes with assumed sleep | Time in bed versus sleep time | Severity misclassification | Measure sleep time |
| EMG envelope window | Onset smeared or noisy | Smoothing choice | False activation | State window |
| EMG normalized incorrectly | Amplitude differs across sessions | Electrode gain | Non-comparable | Normalize to MVC |
| EDA contact loss | Step to rail | Electrode lift | False response | Impedance check |
| EDA decomposition non-unique | Tonic and phasic trade off | Model choice | Different SCL/SCR | Report method |
| CGM compression low | Rapid isolated dip | Sensor pressure | False hypoglycemia | Flag; confirm |
| CGM lag | Alert after the event | Interstitial delay | Late treatment | Predictive alert |
| CGM calibration drift | Rising error | Stale calibration | Bias | Recalibrate |
| TIR range changed | TIR moves without physiology | 70 versus 80 lower bound | Target misclassification | State range |
| Coverage ignored | TIR from 50% of days | Denominator | Unrepresentative | Report coverage |
| Held-over values | Runs of identical readings | Sensor hold | Distorted TIR | Detect runs |
| Missingness informative | Night data absent | Sensor failure at night | Missed nocturnal events | Report by time of day |
| EDA threshold arbitrary | Response count changes | Threshold choice | Non-comparable rates | State threshold |
| EMG cross-talk | Signal from adjacent muscle | Electrode placement | False activation | Verify placement |
| Respiration from PPG | Motion and cardiac interference | Proxy signal | Wrong rate | Validate against belt |
| CGM trend during rapid change | Lag | Interstitial delay | Wrong direction | Model lag |
| GMI called A1C | Report language | Transformation | Misleading | Label GMI |
| TIR without population | Target applied to all | Different targets | Wrong goal | State population |
| MARD without band | Pooled accuracy | Low-range error hidden | Missed hypoglycemia | Report by band |
| Sleep time estimated | AHI denominator wrong | No EEG | Severity misclassification | State method |
| Home test failure ignored | Missing result | Poor signal | Biased sample | Report failure rate |
| EMG gain changed | Amplitude shift | Electrode replacement | Non-comparable | Normalize; calibrate |
| AHI cut-point varied | Severity category changes | 5/15/30 thresholds | Label change | State cut-points |
| EDA decomposition varied | Different SCL/SCR | Model choice | Non-comparable | State method |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The signal is periodic over the window | Autocorrelation | Irregular breathing | Check peak width |
| The waveform is symmetric | Zero-crossing counting | Asymmetric breaths | Compare methods |
| The belt measures effort, not volume | Apnea classification | Calibration drift | Compare with airflow |
| The AASM definition is applied | AHI comparability | Different rules | Record version |
| Sleep time is measured | AHI denominator | Estimated sleep | Compare with EEG |
| The EMG envelope reflects activation | Feature validity | Electrode placement | Normalize; calibrate |
| EDA decomposition is unique | SCL/SCR separation | Model choice | Report method |
| CGM measures interstitial glucose | Metric interpretation | Blood glucose claim | State the proxy |
| The lag is stable | Alert timing | Rapid changes | Model; validate |
| The calibration is current | Accuracy | Stale calibration | Check drift |
| The 14-day record is representative | TIR | Weekly cycles | Compare weeks |
| Coverage exceeds 70% | TIR validity | Gaps | Report coverage |
| The missingness mechanism is known | Imputation choice | Informative dropout | Compare by period |
| The EDA threshold is reported | Response comparability | Arbitrary threshold | Re-use threshold |
| The EMG is normalized | Cross-session comparison | Gain change | Use % MVC |
| The proxy is adequate for the claim | Construct validity | Belt for volume | State the proxy |

## 12. Exercises

### Exercise 1. Respiratory rate (application, 30 min)

**Deliverable:** compute the rate for a 4 s period by autocorrelation and by zero-crossing counting, and explain the difference.

**Acceptance criteria:** both rates in breaths/min; the asymmetry mechanism stated; a robust method recommended.

**Time:** 30 minutes.

### Exercise 2. Apnea definitions (recall, 20 min)

**Deliverable:** write the AASM definitions of apnea, hypopnea, obstructive apnea, and central apnea, with amplitude, duration, and desaturation criteria.

**Acceptance criteria:** all criteria present; the V3 optional 4% rule noted.

**Time:** 20 minutes.

### Exercise 3. AHI denominator (application, 25 min)

**Deliverable:** compute the AHI for 105 events over 7 h of sleep and over 8 h in bed, and state the severity category for each using the common 5/15/30 thresholds.

**Acceptance criteria:** correct arithmetic; the denominator effect explained; categories named.

**Time:** 25 minutes.

### Exercise 4. EMG envelope (application, 35 min)

**Deliverable:** describe an envelope pipeline with band, rectification, and smoothing, and compute the variance reduction for a 100 ms window at 1,000 Hz.

**Acceptance criteria:** parameters stated; variance reduction computed; the force caveat included.

**Time:** 35 minutes.

### Exercise 5. CGM metrics (application, 45 min)

**Deliverable:** from a provided 14-day series, compute mean, SD, CV, TIR, TBR, TBR<54, TAR, TAR>250, and GMI.

**Acceptance criteria:** all metrics with units; coverage reported; targets compared.

**Time:** 45 minutes.

### Exercise 6. Coverage sensitivity (critique, 30 min)

**Deliverable:** recompute TIR after removing 20% of the readings at random and after removing all nighttime readings. Compare.

**Acceptance criteria:** both estimates; the informative missingness mechanism identified; a recommendation.

**Time:** 30 minutes.

### Exercise 7. Compression low (application, 30 min)

**Deliverable:** describe the waveform signature of a compression low and a rule to distinguish it from true hypoglycemia.

**Acceptance criteria:** rapid fall and rise; isolation; confirmation with fingerstick; no filter that removes true lows.

**Time:** 30 minutes.

### Exercise 8. Validation design (design, 60 min)

**Deliverable:** a one-page validation design for a wearable respiratory rate, including reference, population, activity conditions, and agreement metrics.

**Acceptance criteria:** reference specified; rest and exercise strata; Bland-Altman; sample size.

**Time:** 60 minutes.

### Exercise 9. Missingness bound (application, 30 min)

**Deliverable:** for a CGM record with 15% missing readings, compute a worst-case TIR bound by assigning all missing readings to the in-range and then to the out-of-range category.

**Acceptance criteria:** both bounds; the interpretation; a recommendation for whether the metric is robust.

**Time:** 30 minutes.

### Submission standard

Calculations must show the unit and the denominator. A respiratory rate without the window and the method is not a pass. A CGM metric without the coverage is not a pass. An AHI without the sleep time and the AASM version is not a pass.

For the validation design, the acceptance criteria are the rubric: a design that omits the reference, the strata, or the agreement statistic fails regardless of the proposed model. For the critique, each finding must name the mechanism and the consequence, not only the discrepancy.

## 13. Self-check answers

**1. What is the period of a 15 breaths/min rhythm?** 60/15 = 4 s.

**2. Why does zero-crossing counting overestimate an asymmetric waveform?** The waveform crosses the mean more than once per breath, so each extra crossing is counted as a breath.

**3. What is the AASM apnea definition?** A 90% or greater reduction in airflow for at least 10 s.

**4. What is the AASM hypopnea definition?** A 30% or greater reduction in airflow for at least 10 s with a 3% or greater desaturation or an arousal. The 4% desaturation version is optional in V3.

**5. Compute the AHI for 105 events over 7 h.** 105/7 = 15 events/h.

**6. Compute the AHI for the same events over 8 h.** 105/8 = 13.1 events/h.

**7. What is the EMG envelope?** The smoothed rectified EMG amplitude, computed after band-pass filtering.

**8. Why is the envelope not force?** The relationship between surface EMG and force depends on electrode placement, muscle length, fatigue, and cross-talk. It is monotonic but not linear or universal.

**9. What is the difference between SCL and SCR?** SCL is the slow tonic level; SCR is the fast phasic response.

**10. What is the consensus TIR range?** 70-180 mg/dL, or 3.9-10.0 mmol/L.

**11. Compute GMI for a mean glucose of 147.1 mg/dL.** 3.31 + 0.02392 × 147.1 = 6.83%.

**12. Compute CV for mean 147.1 and SD 37.2.** 37.2/147.1 × 100 = 25.3%.

**13. Why is a 3 h gap a coverage problem?** It removes 36 readings and changes the denominator. If the gap is not random, the metric is biased.

**14. What is a compression low?** A false low CGM reading caused by pressure on the sensor, which reduces local perfusion. It is characterized by a rapid fall and rise and is often isolated.

**15. Why does CGM lag matter for a hypoglycemia alert?** The interstitial glucose lags blood by 5-15 min, so the alert fires after the blood glucose has fallen. A predictive alert uses the trend to compensate.

**16. What is the difference between sample-weighted and time-weighted TIR?** Sample-weighted counts each reading equally; time-weighted weights by the interval between readings. They are equal when sampling is uniform.

**17. Compute the TIR for a 14-day record with 3,000 readings in range out of 4,032 expected.** 3,000/4,032 = 74.4%, if all readings are present. If 500 readings are missing, the denominator is 3,532 and the TIR is 84.9% over observed readings, which is not comparable with the 74.4% over expected readings.

**18. What is the MARD and why does it need a reference?** MARD is the mean absolute relative difference between the device and a reference glucose value. Without a reference, it cannot be computed; a device compared with itself has a MARD of zero.

**19. Why is a 100 ms EMG smoothing window a trade-off?** It preserves activation onsets to about 100 ms but leaves more variance in the envelope. A 500 ms window reduces variance but smears brief contractions.

**20. What is the difference between a central and an obstructive apnea?** An obstructive apnea has continued or increased effort with no airflow; a central apnea has no effort and no airflow. The effort channel is required to distinguish them.

**21. Compute a worst-case TIR bound for 70% observed TIR with 15% missing.** If all missing readings are in range, TIR = 0.70×0.85 + 0.15 = 0.745, or 74.5%. If all are out of range, TIR = 0.70×0.85 = 59.5%. The bracket is 59.5% to 74.5%, which straddles the 70% target.

**22. Why does a 14-day record with 70% coverage not guarantee a representative TIR?** The missing 30% may be systematically different, for example nights or weekends. Coverage is necessary but not sufficient; the distribution of coverage matters.

**23. What is the difference between MARD and bias?** MARD is the mean of the absolute relative differences and is always positive; bias is the mean signed difference and can be positive or negative. A device can have a small bias and a large MARD.

**Rubric for Exercise 5 (CGM metrics).** A strong answer reports all metrics with units, states the coverage, and compares each to the consensus target. A weak answer reports only TIR and mean.

**Rubric for Exercise 8 (validation design).** A strong design specifies the reference, the rest and exercise strata, the agreement statistics, and the sample size. A weak design reports only correlation with a reference.

## 14. Interview questions

**1. How would you compute respiratory rate from a belt signal?**
A strong answer describes autocorrelation or peak detection, states the window, and explains why zero-crossing fails. A weak answer counts peaks without a method.

**2. What is the difference between apnea and hypopnea?**
A strong answer gives the 90% and 30% thresholds, the 10 s duration, and the desaturation or arousal criterion. A weak answer says "stopping breathing."

**3. Why does the AHI depend on the denominator?**
A strong answer explains sleep time versus time in bed and the severity categories. A weak answer says "it is events per hour."

**4. How do you validate a CGM-based TIR?**
A strong answer uses a reference CGM or blood glucose, a 14-day record, coverage, and agreement by glucose band. A weak answer compares mean glucose.

**5. What is the difference between interstitial and blood glucose?**
A strong answer explains the lag, the calibration, and the rapid-change error. A weak answer says they are the same.

**6. How do you handle missing CGM data?**
A strong answer reports coverage, distinguishes random from informative missingness, and compares complete-case and imputed estimates. A weak answer imputes linearly.

**7. What is a compression low and how do you detect it?**
A strong answer describes the rapid fall and rise, the pressure context, and confirmation with a fingerstick. A weak answer filters it out.

**8. Why is EMG not a force measurement?**
A strong answer explains the electrode, length, fatigue, and cross-talk dependence. A weak answer says "it is noisy."

**9. What is the AASM V3 change to hypopnea scoring?**
A strong answer states that the 4% desaturation rule became optional, which allows lab-to-lab differences. A weak answer does not know.

**10. How would you validate an EDA stress score?**
A strong answer describes a laboratory stressor, a construct-validity argument, and the lack of a gold standard. A weak answer correlates with a questionnaire.

**11. What is the minimum CGM record for a representative TIR?**
A strong answer states 14 days with at least 70% coverage, and notes that the minimum is not a guarantee. A weak answer says 7 days.

**12. Why is the daytime TIR different from the nighttime TIR?**
A strong answer explains meals, activity, and fasting, and the aggregation effect. A weak answer says "glucose varies."

**13. How would you validate a respiratory rate from a wrist wearable?**
A strong answer uses a belt or capnography as reference, stratifies by rest and exercise, and reports bias and limits of agreement. A weak answer compares with another wearable.

**14. What is the risk of a 4% hypopnea rule for a model trained on 3% labels?**
A strong answer explains the label shift, the lower event count, and the need to harmonise or retrain. A weak answer treats the labels as equivalent.

**15. How do you detect a compression low without a fingerstick?**
A strong answer uses the rapid fall and rise, the isolation, the coincidence with pressure, and the comparison with the contralateral arm if available. A weak answer filters all rapid changes.

**16. Why is an EDA stress score a construct-validity claim?**
A strong answer explains that there is no gold-standard stress measurement and that the score is a constructed index validated against tasks or outcomes. A weak answer says it correlates with heart rate.

**17. What is the first thing you check when an AHI differs between two labs?**
A strong answer checks the AASM version, the hypopnea rule, the desaturation threshold, and the sleep-time denominator before the population. A weak answer assumes a patient difference.

**18. Why is a 5-minute CGM sampling interval adequate for TIR but not for a breath rate?** The glucose signal changes over tens of minutes, so 5 min is fast relative to the signal. A breath lasts 4 s, so 5 min is far too slow.

## 15. Key terms

- **AHI** — apneas plus hypopneas per hour of sleep.
- **Airflow** — gas movement, usually in L/s.
- **Apnea** — ≥90% airflow reduction for ≥10 s.
- **Arousal** — abrupt EEG frequency shift.
- **Calibration** — mapping sensor signal to glucose.
- **CGM** — continuous glucose monitoring.
- **Compression low** — false low from sensor pressure.
- **Coverage** — share of expected readings present.
- **CV** — coefficient of variation.
- **EDA** — electrodermal activity.
- **Effort** — chest or abdominal movement.
- **EMG** — electromyography.
- **Envelope** — smoothed rectified EMG.
- **GMI** — glucose management indicator.
- **HSAT** — home sleep apnea testing.
- **Hypopnea** — ≥30% reduction with desaturation or arousal.
- **Interstitial glucose** — glucose in tissue fluid.
- **MARD** — mean absolute relative difference.
- **ODI** — oxygen desaturation index.
- **RIP** — respiratory inductive plethysmography.
- **SCL** — skin conductance level.
- **SCR** — skin conductance response.
- **Sensor lag** — delay between blood and interstitial glucose.
- **TAR** — time above range.
- **TBR** — time below range.
- **TIR** — time in range.
- **Tidal volume** — volume per breath.
- **Time in bed** — lights out to lights on.
- **Sleep time** — time scored asleep.
- **Flow limitation** — partial airway obstruction.
- **AGP** — ambulatory glucose profile.
- **Rate of change** — glucose slope.
- **Predictive alert** — alert based on trend.
- **MVC** — maximal voluntary contraction.
- **Median frequency** — EMG spectral midpoint.
- **Cross-talk** — signal from adjacent muscle.
- **Baseline breathing** — pre-event amplitude reference.
- **Position sensor** — body position channel.
- **Sample-weighted** — each reading weighted equally.
- **Time-weighted** — each interval weighted by duration.
- **Worst-case bound** — sensitivity analysis assigning missing values to extremes.
- **Structural missingness** — missing by design.
- **Informative missingness** — missingness related to the outcome.
- **Pulse oximetry** — optical oxygen saturation.
- **Capnography** — CO2-based airflow measurement.
- **Plethysmography** — volume change measurement.
- **Dynamometer** — force reference for EMG.
- **Construct validity** — evidence that an index measures its intended construct.
- **Apnea-hypopnea index** — events per hour of sleep.
- **Oxygen desaturation index** — desaturations per hour.
- **Sensor active time** — time the CGM provides data.
- **Ambulatory glucose profile** — glucose summary by time of day.
- **Rate of change** — glucose slope in mg/dL/min.
- **Predictive alert** — alert based on projected glucose.
- **Compression low** — pressure-induced false low.
- **Electrodermal response** — phasic skin conductance change.
- **Tonic level** — slow EDA baseline.
- **Maximal voluntary contraction** — EMG normalization reference.
- **Median frequency** — EMG spectral midpoint.
- **Cross-talk** — signal from an adjacent muscle.
- **Plethysmography** — volume change measurement.
- **Capnography** — CO2-based airflow measurement.
- **Baseline** — pre-event reference amplitude.
- **Bradycardia** — slow heart rate.
- **Tachycardia** — fast heart rate.
- **Artifact** — non-physiological signal.
- **Quality mask** — segment-level validity flag.
- **Tachypnea** — fast respiratory rate.
- **Apnea** — absence of airflow.
- **Hypopnea** — partial airflow reduction.
- **Desaturation** — drop in oxygen saturation.
- **Arousal** — abrupt sleep EEG change.
- **Positional apnea** — apnea dependent on body position.
- **Adherence** — extent of device use.

## 16. Sources

All URLs accessed 25 October 2026.

1. Battelino T, Danne T, Bergenstal RM, et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. *Diabetes Care*. 2019;42(8):1593-1603. https://doi.org/10.2337/dci19-0028
2. American Diabetes Association. New recommendations for time-in-range targets. https://diabetes.org/newsroom/new-recommendations-for
3. Troester MM, Quan SF, Berry RB, et al. The AASM Manual for the Scoring of Sleep and Associated Events. Version 3. 2023. https://aasm.org/clinical-resources/scoring-manual/
4. American Academy of Sleep Medicine. Summary of Updates in Scoring Manual Version 3. https://aasm.org/wp-content/uploads/2023/05/Summary-of-Updates-Scoring-Manual-v3.pdf
5. American Academy of Sleep Medicine. Summary of Changes, Version 3.1. https://aasm.org/wp-content/uploads/2024/02/SummaryofChanges_Document_3-1.pdf
6. ISO 15197, In vitro diagnostic test systems — requirements for blood-glucose monitoring systems for self-testing. https://www.iso.org/
7. CLSI POCT05, Performance metrics for continuous interstitial glucose monitoring. https://clsi.org/
8. US Food and Drug Administration. Blood Glucose Monitoring Test Systems guidance. https://www.fda.gov/
9. PhysioNet. Sleep-EDF Database Expanded. https://physionet.org/content/sleep-edfx/
10. PhysioNet. MIT-BIH Polysomnographic Database. https://physionet.org/content/slpdb/
11. Beniczky S, Schomer DL. Electroencephalography: basic biophysical and technological aspects. *Epileptic Disorders*. 2020. https://doi.org/10.1684/epd.2020.1217
12. IEC 60601-2-26, Electroencephalographs. https://webstore.iec.ch/
13. US Food and Drug Administration. Continuous Glucose Monitoring Systems guidance. https://www.fda.gov/
14. American Diabetes Association. Standards of Care in Diabetes, 2026. https://diabetesjournals.org/
15. PhysioNet. Apnea-ECG Database. https://physionet.org/content/apnea-ecg/
16. Berry RB, et al. Rules for scoring respiratory events in sleep. *J Clin Sleep Med*. 2012;8(5):597-619. https://doi.org/10.5664/jcsm.2172

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_ppg-and-wearables.md`](04_ppg-and-wearables.md)
- Next topic: [`06_feature-engineering.md`](06_feature-engineering.md)
- Related topic: [`08_digital-biomarkers.md`](08_digital-biomarkers.md)
- Module survey file: [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
- Related topic: [`09_validation.md`](09_validation.md)
- Related topic: [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md)
- Dataset: [`https://physionet.org/content/apnea-ecg/`](https://physionet.org/content/apnea-ecg/)
