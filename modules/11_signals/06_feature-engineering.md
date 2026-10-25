# Feature Engineering: Time Domain, Frequency Domain, Morphology, and Variability

> Module 11 — Physiological Signals and Wearables
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals), Topic 02 (ECG)
> Version 2026.3

## 1. Why this topic exists

A model does not see a signal. It sees a feature vector. The choice of features determines what the model can learn, what it cannot learn, and what it will learn by accident. In physiological data, the feature layer is where most of the scientific and regulatory content lives: a heart-rate variability (HRV) feature has a definition, a window, a unit, and a normative range, and a model that uses it is making a claim about all four.

Feature engineering is also where the most expensive mistakes are made, because they are invisible. A feature computed with the wrong window, the wrong normalization, or the wrong handling of ectopic beats looks like a number. It enters the model, the model finds a pattern, and the pattern is a property of the preprocessing rather than the physiology. The model is not wrong in a way that a held-out set will catch, because the held-out set has the same preprocessing.

Five failures recur.

**The feature that is a window.** The standard deviation of RR intervals (SDNN) depends on the recording length. A 5 min SDNN and a 24 h SDNN are different measurements with the same name. A dataset that mixes windows produces a feature that encodes the window, and a model that learns it will perform differently when the window changes.

**The feature that is an artifact.** A single ectopic beat can double the root mean square of successive differences (RMSSD), because RMSSD squares the differences. A motion artifact that creates one extra peak has the same effect. The feature is a valid computation on an invalid series, and the model learns the artifact.

**The feature that is a duplicate.** Mean RR and heart rate are reciprocals. SDNN and total spectral power are related. A feature set with many correlated features inflates the apparent dimensionality, destabilizes linear models, and makes importance rankings uninterpretable. The redundancy is not harmful in a tree model, but it is harmful in the interpretation.

**The feature that is not comparable.** A raw EMG amplitude depends on electrode gain; a raw PPG amplitude depends on skin tone and contact; a raw ECG amplitude depends on lead placement. A feature that is not normalized cannot be pooled across devices or subjects.

**The feature that has no unit.** A number without a unit cannot be checked. A feature table with a column called "HRV" and no unit, window, or method is not a dataset; it is a collection of numbers that happened to be computed.

This topic covers the feature families used in physiological data science: time domain, frequency domain, time-frequency, morphology, variability, and cross-signal. It teaches how to compute each with a stated definition, how to test robustness, and how to document a feature set so that another team can reproduce it. It builds on Topic 01 for sampling and filtering and on Topic 02 for ECG morphology; it sets up Topic 07 for representation learning and Topic 08 for biomarkers.

### 1.1 Three composite failures

**The HRV that was an ectopic beat.** A study compares HRV between a disease group and a control group. The disease group has more ectopic beats, and the analysis retains them. RMSSD is 40% higher in the disease group, and the study concludes that the disease is associated with increased vagal tone. The true association is with ectopy. The feature is arithmetically correct and biologically wrong. The cost is a published conclusion that does not replicate.

**The model that learned the device.** A team pools wearable data from three devices. One device samples PPG at 25 Hz, one at 100 Hz, and one at 50 Hz. The team computes a pulse morphology feature that depends on the sampling rate. The model learns the device identity from the feature and uses it as a shortcut for the label, because the devices were deployed in different clinics with different prevalence. The performance is good in the pooled test set and fails in a new clinic. The cost is a deployment failure that the offline metrics did not predict.

**The feature table that cannot be reproduced.** A paper reports that a 12-feature model predicts an outcome with an AUC of 0.82. The feature table is not published, the definitions are not given, and the code is not available. A replication team uses the same named features from a library and gets an AUC of 0.68. The difference is not necessarily an error; it is an under-specification. The cost is a result that cannot be checked.

The three failures share a structure: the feature is treated as a property of the signal rather than a property of the signal plus the definition. The fix is to attach the definition to the feature and to test the feature's sensitivity to the definition.

### 1.2 Why features still matter in the deep-learning era

A deep model can learn features from raw signal. It does not remove the need for feature engineering, for four reasons.

- **Small data.** Clinical datasets are often hundreds to thousands of records, not millions. A handcrafted feature set reduces the sample complexity.
- **Interpretability.** A feature such as SDNN has a physiological meaning and a normative range. A learned embedding does not.
- **Regulation.** A reviewer can evaluate a defined feature; an opaque representation is harder to assess.
- **Robustness.** A feature with a stated window and editing rule can be tested across devices; a learned representation must be revalidated.

The practical pattern is a hybrid: a small set of interpretable features plus a learned representation, with the features serving as a check on the representation. If the representation and the features disagree, the disagreement is a finding.

A feature layer also makes a model easier to monitor after deployment. A drift in SDNN or in perfusion index is interpretable; a drift in a 128-dimensional embedding is not. The feature layer gives the monitoring system a vocabulary for what changed.

The cost of the feature layer is the risk of under-specification. A learned representation is self-contained; a feature is only as good as its definition. The rest of this topic is about making the definition good enough.

A feature layer also makes the model's failure modes visible in a way that an embedding does not. If the model fails when the perfusion index is low, the feature table shows it; if the embedding fails, the cause is opaque. The visibility is worth the specification burden.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** time-domain HRV features (SDNN, RMSSD, pNN50) from an RR series with a stated editing rule and window.
2. **Compute** frequency-domain features (VLF, LF, HF, LF/HF) with a stated interpolation rate, window, and band definition.
3. **Compute** a nonlinear feature (sample entropy) and explain why it is not a substitute for the time-domain measures.
4. **Extract** morphology features (QRS duration, R amplitude, pulse rise time) with a stated fiducial definition.
5. **Normalize** features across devices and subjects, and state what the normalization removes and what it preserves.
6. **Test** a feature for robustness to window length, ectopic beats, noise, and sampling rate.
7. **Reduce** a feature set without losing interpretability, using correlation and domain knowledge.
8. **Document** a feature table with units, windows, methods, and versions so that it can be reproduced.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are ordered so that computation comes before robustness and documentation. A learner who can compute a feature but cannot say how it changes with the window has not finished the topic. The last three outcomes are the ones that make a feature set reusable.

## 3. The core mental model

A feature is a function of a signal and a set of choices. The signal is the input; the choices are the definition. The output is a number that is only meaningful with the choices attached.

```text
 signal + choices -> feature
    |         |
    |         +-- window length
    |         +-- filter band
    |         +-- editing rule
    |         +-- normalization
    |         +-- aggregation
    |
 raw waveform, RR series, or multi-channel record
    |
 feature families
    +-- time domain: mean, SD, RMSSD, pNN50, zero crossings
    +-- frequency domain: band power, peak frequency, spectral edge
    +-- time-frequency: wavelet, spectrogram
    +-- morphology: width, amplitude, rise time, notch
    +-- variability: Poincare, entropy, fractal
    +-- cross-signal: pulse transit time, RSA, coupling
    |
 feature vector -> model -> decision
```

Redraw it with the units and windows on the right. The key idea is that the feature is not a property of the signal alone; it is a property of the signal plus the definition.

### 3.1 The four questions for every feature

Before a feature enters a model, four questions should have answers.

1. **What is the unit?** Milliseconds, microvolts, square milliseconds, dimensionless. A feature without a unit is not checkable.
2. **What is the window?** 10 s, 30 s, 5 min, 24 h. The window determines the resolution and the stationarity assumption.
3. **What is the editing rule?** Are ectopic beats removed, interpolated, or retained? Is the RR series filtered? Is the feature computed on all beats or on normal-to-normal intervals only?
4. **What is the normalization?** Raw, divided by the mean, z-scored within subject, normalized to a maximal contraction, or scaled by body surface area.

A feature with all four answers is reproducible. A feature with one missing is a candidate for the "feature that is a window" or "feature that is an artifact" failure.

### 3.2 Feature families and what they capture

| Family | Examples | Captures | Sensitive to |
|---|---|---|---|
| Time domain | Mean, SD, RMSSD, pNN50 | Amplitude and beat-to-beat variation | Window, ectopy, editing |
| Frequency domain | LF, HF, LF/HF, spectral edge | Periodic content | Interpolation, window, band edges |
| Time-frequency | Wavelet energy, spectrogram | Non-stationary content | Window, scale, redundancy |
| Morphology | QRS width, R amplitude, rise time | Waveform shape | Filter, fiducial, lead |
| Variability | SD1, SD2, sample entropy | Nonlinear dynamics | Series length, parameter choice |
| Cross-signal | Pulse transit time, RSA | Coupling between systems | Alignment, lag, sampling |

The table is a map, not a taxonomy. A feature can belong to more than one family, and the same physiological property can be measured by features in several families. The choice should be driven by the decision, not by the availability of a library function.

The families also differ in cost. A time-domain feature is cheap; a wavelet decomposition is more expensive; a cross-signal coherence is expensive and requires aligned channels. The feature budget should be spent on the families that answer the question, not on the families that are easy to compute.

A useful rule is to include at least one feature from each of three perspectives: a central tendency, a dispersion, and a shape. A model that sees only means cannot distinguish a stable subject from an oscillating one; a model that sees only dispersion cannot distinguish a high baseline from a low one. The three perspectives are complementary, and each has a different sensitivity to artifact.

### 3.3 The bias-variance trade-off in features

More features are not better. Each feature adds variance to the estimate and a opportunity for leakage. A model with 500 features on 200 subjects has more parameters than data, and the feature selection becomes a source of overfitting.

The practical rule is to start with a small set of interpretable features, test their robustness, and add complexity only when the simpler set fails. A gradient-boosted model on 20 well-defined features often matches a deep model on raw signal for tabular physiological tasks, and it is easier to audit.

### 3.4 Measurement, feature, biomarker, decision

- **Measurement**: an RR interval in milliseconds, a QRS duration in milliseconds.
- **Feature**: SDNN, RMSSD, LF power, QRS width.
- **Biomarker**: a feature with a validated relationship to an outcome, such as SDNN and cardiovascular risk.
- **Decision**: flag, refer, stratify.

A feature is not a biomarker because it is significant in one study. The validation pathway in Topic 08 applies: analytical, technical, clinical, utility.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Feature | Computed descriptor of a signal | Calling raw signal a feature | unit or dimensionless |
| Feature vector | Ordered set of features for one unit | Mixing units without documentation | vector |
| RR interval | Interval between R peaks | Using PP intervals interchangeably | ms |
| NN interval | Normal-to-normal RR interval | Including ectopic beats | ms |
| SDNN | Standard deviation of NN intervals | Comparing across windows | ms |
| SDANN | SD of 5-min mean NN intervals | Using for short records | ms |
| RMSSD | Root mean square of successive differences | Using with ectopy | ms |
| pNN50 | Proportion of successive differences >50 ms | Using with ectopy | % |
| Triangular index | HRV geometric measure | Confusing with SDNN | dimensionless |
| VLF | Very-low-frequency power | Using on 5-min records | ms² |
| LF | Low-frequency power 0.04-0.15 Hz | Calling it sympathetic | ms² |
| HF | High-frequency power 0.15-0.4 Hz | Calling it vagal without caveat | ms² |
| LF/HF | Ratio of LF to HF power | Calling it sympathovagal balance | dimensionless |
| Normalized units | LF or HF / (total - VLF) | Forgetting the denominator | % |
| Total power | Power 0.0033-0.4 Hz | Using a different band | ms² |
| Interpolation | Resampling RR at a fixed rate | Ignoring the method | Hz |
| Spectral leakage | Power spreading across bins | Ignoring window | dB |
| Window | Analysis segment | Leaving length unstated | s |
| Ectopic beat | Non-sinus beat | Retaining in NN series | event |
| Editing | Removing or interpolating artifacts | Using an automatic filter | method |
| Poincare plot | SD1/SD2 plot of RR pairs | Treating as a diagnosis | ms |
| SD1 | Short-term Poincare dispersion | Confusing with RMSSD | ms |
| SD2 | Long-term Poincare dispersion | Confusing with SDNN | ms |
| Sample entropy | Regularity measure | Comparing across lengths | dimensionless |
| Approximate entropy | Regularity measure | Confusing with sample entropy | dimensionless |
| Detrended fluctuation | Fractal scaling exponent | Using without scale range | dimensionless |
| Spectral edge | Frequency below a power fraction | Reporting without fraction | Hz |
| Wavelet | Time-frequency decomposition | Using redundant bases | coefficients |
| Morphology | Waveform shape features | Confusing with rhythm | varies |
| QRS duration | Width of the QRS | Fiducial-dependent | ms |
| Rise time | Time from trough to peak | Definition-dependent | ms |
| Pulse transit time | Delay from ECG to PPG | Confusing with PWV | ms |
| RSA | Respiratory sinus arrhythmia | Confusing with HRV | ms |
| Normalization | Scaling to a reference | Hiding device differences | method |
| Z-score | (x - mean)/SD | Computing across subjects | dimensionless |
| Batch effect | Technical variation across batches | Confusing with biology | varies |
| Feature selection | Choosing a subset | Selecting on the test set | method |
| Leakage | Information from test in train | Computing features on all data | error |
| Coefficient of variation | SD/mean | Using for bounded variables | % |
| Ratio feature | Feature divided by another | Ignoring the denominator noise | dimensionless |
| Aggregation | Combining epochs | Not stating mean versus median | method |
| Fragmentation | Splitting events across windows | Double-counting | events |
| Context feature | Time, activity, posture | Leaking future information | varies |
| Lag feature | Past value as predictor | Using future values | varies |
| Rolling feature | Feature over a moving window | Look-ahead bias | varies |
| Baseline feature | Value at a reference time | Choosing baseline after outcome | varies |
| Interaction | Product of two features | Interpreting without scale | varies |
| Unit conversion | ms to s, mV to µV | Reporting inconsistent units | factor |
| Reference population | Normative comparison group | Using the wrong norm | cohort |

## 5. The mechanics, step by step

The procedure is written for a physiological feature set built from ECG, PPG, or respiration, and it generalises to EEG with different feature definitions.

### Step 1. Define the decision and the unit of analysis

**Input:** the clinical or product decision.

**Output:** the unit of analysis (beat, epoch, night, patient), the prediction horizon, and the feature families that could support it.

**Rationale:** the unit of analysis determines the window and the aggregation. A beat-level arrhythmia detector needs beat features and a short context. A nightly sleep summary needs epoch features and a night aggregation. A patient-level risk score needs a summary over the record.

**Branch point:** if the decision is event detection, the features must be local in time. If the decision is risk stratification, the features can be averaged over the record. The two require different windows and different validation splits.

**Edge case:** a decision that is measured at the patient level but trained at the epoch level needs an aggregation rule. The rule is part of the model, not an afterthought.

**The unit-of-analysis table.**

| Decision | Unit | Window | Aggregation |
|---|---|---|---|
| Arrhythmia alert | Beat | 5 s context | Event count |
| Sleep stage | 30 s epoch | 30 s | Sequence model |
| Nightly apnea | Night | Whole night | AHI |
| Glycemic control | 14 days | 5 min readings | TIR |
| Risk score | Patient | Whole record | Mean or median |
| Deterioration | Hour | 1 h | Trend |

A mismatch between the unit and the decision is a silent error. A patient-level risk score computed from a single epoch is not the same measurement as one computed from a week, and the difference is not a modelling choice.

**Leakage from the unit.** If the same patient contributes multiple epochs and the split is by epoch, the model sees the patient in both train and test. The leakage is not in the features; it is in the unit. The split must be at the unit of analysis or above.

### Step 2. Choose the window and the step

**Input:** the unit of analysis and the physiology.

**Output:** a window length, a step, and a justification.

**Rationale:** the window sets the frequency resolution and the number of independent observations. The step sets the overlap and the effective sample size.

**Arithmetic:** a 5 min window gives 1/300 = 0.0033 Hz resolution, enough for the LF and HF bands. A 10 s window gives 0.1 Hz, which cannot resolve the LF band. A 30 s window gives 0.033 Hz, marginal for LF. The Task Force recommendation is 5 min for short-term frequency-domain HRV.

**Branch point:** if the decision needs frequent updates, use a short window and time-domain features. If it needs stable spectral features, use 5 min and accept the latency.

**Edge case:** overlapping windows are not independent. A 5 min window with a 1 min step has five times as many windows as independent observations, and a naive confidence interval will be too narrow.

The effective sample size is roughly the number of non-overlapping windows, not the number of windows. A study with 1,000 overlapping windows from 200 independent 5 min segments has an effective N near 200. The distinction matters for every confidence interval and every p-value.

**Window and decision table.**

| Decision | Window | Step | Effective N |
|---|---|---|---|
| Resting HRV | 5 min | 5 min | 1 per segment |
| Sleep stage | 30 s | 30 s | 1 per epoch |
| Arrhythmia | 10 s | 5 s | 2 per 10 s |
| Trend | 5 min | 1 min | 1 per 5 min |

The table is a reminder that the step, not only the window, determines the number of observations.

### Step 3. Clean the series before computing features

**Input:** the raw peaks or RR intervals.

**Output:** an edited NN series with a documented rule.

**Rationale:** ectopic beats, missed beats, and false peaks corrupt every downstream feature. The editing rule determines which beats are included and how the gaps are handled.

**Rules:** a common approach is to remove intervals that differ from the median by more than 20%, interpolate the removed intervals, and exclude the first and last beats. The rule must be stated because different rules produce different features.

**Arithmetic:** an ectopic beat with a coupling interval of 0.45 s among intervals of 0.83 s creates two successive differences of about -0.38 s and +0.38 s. RMSSD squares these, so a single ectopic beat can add (0.38² + 0.38²)/N to the mean square. With N = 359, the contribution is 0.0008 s², which is a large fraction of a normal RMSSD² of 0.0009 s².

**Branch point:** if the decision is arrhythmia detection, the ectopic beat is the signal, not the artifact. Do not remove it. If the decision is HRV, the ectopic beat is an artifact and should be removed or interpolated.

**Edge case:** a missed beat creates an interval of about twice the true RR. A rule that removes intervals above 1.5× the median will remove it, but it will also remove a genuine compensatory pause after an ectopic beat.

**Editing rules in practice.** Four rules are common, and they produce different features.

| Rule | Action | Effect on RMSSD | Use |
|---|---|---|---|
| None | Retain all intervals | Inflated by ectopy | Arrhythmia analysis |
| Delete | Remove abnormal intervals | Reduced; gaps | Short records |
| Interpolate | Replace with local mean | Preserves length | HRV |
| Median filter | Replace with running median | Smooths | Noisy series |

The rule should be stated with the threshold that defines "abnormal," usually a deviation from the median of 20% or a deviation from the previous interval of 20%. The number of edited beats should be reported, because a high edit rate is a quality problem.

**The edit rate is a feature.** A record with 15% edited beats is not a clean HRV record, regardless of the feature values. The edit rate belongs in the quality table, and a model that uses HRV should be trained with the edit rate as a context feature or should exclude high-edit records.

### Step 4. Compute time-domain features

**Input:** the edited NN series.

**Output:** mean NN, SDNN, RMSSD, pNN50, and the triangular index.

**Rationale:** time-domain features are simple, interpretable, and robust to the spectral assumptions. They are the first features to compute and the first to report.

**Arithmetic:** for a series of N intervals with mean m:

- SDNN = √(Σ(xi - m)²/(N-1)).
- RMSSD = √(Σ(xi+1 - xi)²/(N-1)).
- pNN50 = count(|xi+1 - xi| > 50 ms)/ (N-1) × 100.

For the worked example, N = 360, mean NN = 0.8337 s, SDNN = 30.9 ms, RMSSD = 30.9 ms, pNN50 = 10.9%.

**Branch point:** if the series is short, SDNN is biased low and RMSSD is noisy. Report the window with the value.

**Edge case:** SDNN includes both short-term and long-term variation; RMSSD is dominated by short-term variation. A series with a slow trend has a high SDNN and a normal RMSSD. The two features answer different questions.

### Step 5. Compute frequency-domain features

**Input:** the edited NN series and the beat times.

**Output:** VLF, LF, HF, total power, LF/HF, and normalized units.

**Rationale:** the frequency-domain features describe the periodic content of the beat-to-beat variation. The bands are conventions from the Task Force: VLF 0.0033-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz, total 0.0033-0.4 Hz.

**Method:** interpolate the RR series at a fixed rate (often 4 Hz), detrend, apply a window, compute the periodogram, and integrate the power in each band. The interpolation rate must be at least twice the highest band edge, so 4 Hz is adequate for 0.4 Hz.

**Arithmetic:** the 5 min series at 4 Hz has 1,200 samples; zero-padded to 2,048, the frequency resolution is 4/2,048 = 0.00195 Hz. The LF band contains about 56 bins and the HF band about 128 bins. The integrated power is in ms².

**Branch point:** if the series has a strong trend, detrend before the spectrum. If the series has gaps, interpolate or use a Lomb-Scargle periodogram.

**Edge case:** the LF/HF ratio is unstable when HF is small. A ratio of 1.45 with HF 776 ms² is meaningful; a ratio of 20 with HF 20 ms² is noise. Report the components, not only the ratio.

**Interpolation and the tachogram.** The RR series is unevenly sampled: each value is separated by the interval it measures. A periodogram requires even sampling, so the series is interpolated. The interpolation rate, the method, and the detrending are all choices.

- **Linear interpolation at 4 Hz** is the common default. It preserves the LF and HF bands and is simple.
- **Cubic spline** preserves the shape better but can overshoot at sharp changes.
- **Lomb-Scargle** computes the spectrum without interpolation, at the cost of a more complex implementation.
- **Detrending** removes the mean and any slow trend. A first-order detrend removes the mean; a higher-order detrend removes slow drift that would otherwise leak into VLF.

The interpolation changes the high-frequency content. At 1 Hz, a 0.4 Hz component has 2.5 samples per cycle, which is marginal; at 4 Hz, it has 10, which is comfortable. The rate should be at least 4 times the highest band edge, not 2 times, to avoid amplitude error.

**The window function.** A rectangular window has the best nominal resolution and the most leakage. A Hann window reduces leakage and broadens the peak. For band power, the choice changes the value by a few percent; for peak frequency, it changes the location by a fraction of a bin. The window must be stated.

### Step 6. Compute nonlinear features

**Input:** the edited NN series.

**Output:** Poincare SD1/SD2, sample entropy, and optionally detrended fluctuation analysis.

**Rationale:** nonlinear features capture regularity and fractal scaling that the time and frequency domains do not. They are useful for research and for some clinical applications, but they are more sensitive to series length and parameter choice.

**Arithmetic:** SD1 = √(0.5) × SD of successive differences; SD2 = √(2 × SDNN² - 0.5 × SD of successive differences²). Sample entropy with m = 2 and r = 0.2 × SD counts matching templates of length 2 and 3.

**Branch point:** if the series is shorter than about 200 beats, sample entropy is biased. Use a longer series or a different measure.

**Edge case:** sample entropy depends on r. A change in r from 0.2 to 0.15 changes the value by 10% to 20%, which can be larger than a group difference. Report r.

### Step 7. Extract morphology features

**Input:** the filtered waveform and the fiducials.

**Output:** widths, amplitudes, rise times, and shape ratios.

**Rationale:** morphology features describe the shape of the beat or pulse, not only its timing. They are sensitive to the filter, the fiducial definition, and the lead or sensor site.

**Examples:** QRS duration from onset to J point; R amplitude from baseline to peak; PPG pulse rise time from trough to peak; PPG pulse width at half amplitude.

**Arithmetic:** a QRS duration of 86 ms at 500 samples/s is 43 samples. A filter with a 40 Hz low-pass broadens the QRS by a few milliseconds; a 150 Hz filter preserves it. The same beat measured on two devices can differ by 10 ms from the filter alone.

**Branch point:** if the morphology feature is used for diagnosis, the filter and the fiducial must match the clinical standard. If it is used as a model input, the filter must be consistent between training and deployment.

**Edge case:** a morphology feature can be dominated by a single abnormal beat. Use a median beat or a robust aggregation.

**Morphology is filter-dependent.** A QRS duration measured after a 40 Hz low-pass is longer than the same QRS measured after a 150 Hz low-pass, because the filter removes the high-frequency components that define the sharp onset and offset. The difference can be 5 ms to 15 ms, which is larger than the difference between some normal and abnormal categories.

A morphology feature should be reported with the filter, the fiducial definition, and the sampling rate. A model that uses morphology should be trained and deployed with the same chain. A model trained on 500 Hz, 150 Hz data and deployed on 100 Hz, 40 Hz data sees a different QRS width for the same heart.

**Amplitude features and gain.** An R amplitude of 1 mV can be 1,000 codes on one device and 6,554 on another. The physical value is the same; the digital value is not. A feature table should store the physical value and the scale, not the raw code.

**Normalization of morphology.** A QRS amplitude can be normalized by the total QRS area or by the lead amplitude, but the normalization changes the meaning. A ratio of R to S is scale-free; an absolute amplitude is not. The choice should match the question.

### Step 8. Compute cross-signal features

**Input:** two synchronized signals, such as ECG and PPG, or ECG and respiration.

**Output:** a delay, a coupling, or a coherence feature.

**Rationale:** cross-signal features capture the relationship between systems, such as pulse transit time or respiratory sinus arrhythmia. They require accurate time alignment and a known lag.

**Arithmetic:** pulse transit time is the delay from the ECG R peak to the PPG pulse arrival, typically 200 ms to 300 ms. A 10 ms error in either fiducial is a 3% to 5% error in the feature. Respiratory sinus arrhythmia is the variation of RR with the respiratory cycle, measured as the HF power or as the coherence between respiration and RR.

**Branch point:** if the signals are not synchronized, the delay is not measurable. Use a shared clock or a synchronization pulse.

**Edge case:** the delay depends on blood pressure and vascular stiffness, so a change in pulse transit time can reflect either. The feature is not specific without a model.

**Alignment and lag.** A cross-signal feature requires the two signals to share a clock. If they do not, the delay is the sum of the physiological delay and the clock offset, and the two cannot be separated without a synchronization event.

A simple alignment check is to compute the cross-correlation and inspect the peak. A sharp peak at a plausible delay suggests good alignment; a broad or multiple peak suggests noise or a changing relationship. The peak lag should be reported with the feature.

**Coherence and coupling.** Respiratory sinus arrhythmia can be measured as the HF power of the RR series or as the coherence between the respiratory signal and the RR series at the respiratory frequency. The two are related but not identical: HF power includes non-respiratory variation in the same band, while coherence isolates the shared component. The choice matters when the respiratory rate is near the HF band edge.

### Step 9. Normalize and scale

**Input:** the raw feature values.

**Output:** normalized values with a documented method.

**Rationale:** normalization removes device, subject, and session effects but can also remove the signal. The choice depends on the decision.

**Methods:** divide by the mean (coefficient of variation), z-score within subject, normalize to a maximal voluntary contraction, scale by body surface area, or rank-transform.

**Arithmetic:** a raw EMG amplitude of 80 µV during a contraction and 20 µV at rest has a ratio of 4. If the electrode gain changes by a factor of 2, the raw values become 160 and 40 µV, and the ratio is still 4. The ratio is robust; the raw value is not.

**Branch point:** if the decision is within-subject change, normalize within subject. If the decision is between-subject comparison, normalize to a population reference and report the reference.

**Edge case:** z-scoring within subject removes the between-subject differences that may be the signal. A model trained on within-subject z-scores cannot detect a between-subject difference in mean level.

**Normalization decision table.**

| Question | Normalization | What it removes | What it preserves |
|---|---|---|---|
| Within-subject change | Z-score within subject | Subject level | Change over time |
| Between-subject comparison | Population z-score | Population mean | Subject deviation |
| Cross-device pooling | Device-specific scaling | Device gain | Physiology |
| EMG amplitude | % MVC | Electrode gain | Activation |
| PPG amplitude | Perfusion index | Contact and skin | Pulsatile fraction |
| ECG amplitude | Ratio to QRS area | Lead gain | Shape |

A normalization that removes the effect of interest is a design error. The table forces the question: what should remain after normalization? If the answer is "the between-subject difference," a within-subject z-score is wrong.

**Batch effects.** When data are collected in batches, a feature can differ by batch for technical reasons. A batch correction can remove the effect, but it can also remove a biological difference that coincides with the batch. The correction should be tested by comparing the feature distribution before and after, and by checking whether the batch variable is confounded with the outcome.

### Step 10. Test robustness

**Input:** the feature set and the raw data.

**Output:** a robustness report for window, ectopy, noise, and sampling rate.

**Rationale:** a feature that changes materially with a defensible preprocessing choice is not a stable measurement. The test is to recompute the feature under alternative choices and to report the range.

**Tests:** window length ±50%, ectopic removal versus retention, additive noise at a stated SNR, sampling rate halved or doubled, and filter cutoff changed.

**Arithmetic:** if SDNN changes from 30.9 ms to 36.9 ms when one ectopic beat is retained, the feature is not robust to ectopy. If RMSSD changes from 30.9 to 43.0 ms, it is even less robust. The ratio of the change to the baseline is the sensitivity.

**Branch point:** if the feature is not robust, either fix the preprocessing or exclude the feature. A feature that is robust only under one arbitrary choice is a liability.

**Edge case:** robustness to noise is not the same as robustness to artifact. A feature can be stable under Gaussian noise and unstable under a motion burst.

**A robustness table.** The test should be designed before the result is seen.

| Perturbation | Range | Expected stable feature | Expected sensitive feature |
|---|---|---|---|
| Window ±50% | 2.5-7.5 min | Mean RR | SDNN, LF |
| Ectopy | 1 beat in 360 | Mean RR | RMSSD, pNN50 |
| Noise | SNR 20 dB | Mean RR | Morphology, entropy |
| Sampling | 500 → 250 Hz | Rate | QRS width, notch |
| Filter | 40 → 150 Hz | Rate | QRS width, ST |
| Interpolation | 4 → 1 Hz | Mean RR | HF power |

A feature that appears in the "stable" column for all perturbations is a candidate for a core feature. A feature that appears in the "sensitive" column needs a fixed protocol and a reported sensitivity.

**The cost of robustness testing.** Each perturbation is a full recomputation of the feature, which can be expensive for large datasets. A sample of records is enough to estimate the sensitivity; the full dataset is not required. The sample should include the range of quality, not only the clean records.

### Step 11. Select and reduce

**Input:** the robust feature set.

**Output:** a smaller set with a rationale.

**Rationale:** correlated features add variance and complicate interpretation. Selection should use domain knowledge first and statistics second, and it must be done inside the training fold to avoid leakage.

**Methods:** remove features with near-zero variance, remove one of each pair with correlation above 0.9, use domain knowledge to choose among correlated features, and use a penalized model rather than a hard selection when the sample is small.

**Branch point:** if the sample is large, keep more features and let the model regularize. If the sample is small, prefer a small, interpretable set.

**Edge case:** selecting features on the full dataset and then cross-validating leaks the test information. Selection must be inside the cross-validation loop.

**Selection methods and their assumptions.**

| Method | Assumption | Strength | Weakness |
|---|---|---|---|
| Variance filter | Low variance is uninformative | Simple | Can remove rare signal |
| Correlation filter | Correlated features are redundant | Reduces collinearity | Can remove complementary signal |
| Univariate test | Marginal association matters | Interpretable | Ignores interactions |
| L1 penalty | Sparse solution is true | Embedded in model | Unstable with correlated features |
| Tree importance | Nonlinear associations | Captures interactions | Biased toward high-cardinality |
| Domain selection | Prior knowledge is correct | Interpretable, stable | Misses novel features |

The method should be chosen before the data are seen, and the selection should be nested inside the validation. A selection that is reported as "we used the top 10 features" without saying top by what, on which data, is not reproducible.

### Step 12. Document the feature table

**Input:** the final features.

**Output:** a data dictionary with name, definition, unit, window, method, version, and known limitations.

**Rationale:** a feature table without a dictionary is not reusable. The dictionary is what makes the features comparable across studies and reproducible across teams.

**Contents:** the twelve fields from Topic 01 plus the feature-specific fields: window, editing rule, normalization, and reference population.

**Branch point:** if the feature is used in a regulated claim, the dictionary is part of the evidence. A feature without a definition cannot be validated.

**Edge case:** a feature whose definition changes between versions is a new feature. Version it with a new name, not a silent update.

**The dictionary is a contract.** If a downstream team consumes a feature table, the dictionary is the interface. A change to the definition without a version change breaks the contract silently. A version change with a migration note is a managed change.

A feature dictionary should also record what the feature is not. "RMSSD is not a measure of vagal tone in the presence of ectopy" is as important as the formula. The limitations column is part of the definition.

### Decision table for feature handling

| Condition | Feature type | Action |
|---|---|---|
| Ectopic beat present | HRV | Remove or interpolate; report rule |
| Ectopic beat present | Arrhythmia | Retain; it is the signal |
| Short record (<2 min) | Frequency-domain | Do not report LF/HF |
| Strong trend | Frequency-domain | Detrend before spectrum |
| Gaps | Frequency-domain | Interpolate or Lomb-Scargle |
| Low perfusion | PPG morphology | Exclude; morphology unreliable |
| Different devices | Any amplitude feature | Normalize; report device |
| Different sampling rates | Morphology | Resample; report rate |
| Small sample | Selection | Domain-led, nested CV |
| Regulated claim | All | Data dictionary and version |

## 6. Worked example

### 6.1 The case

A 5 min resting ECG is recorded at 500 samples/s. The subject is supine and breathing spontaneously. The R peaks are detected, the RR series is edited, and the following features are computed: mean NN, heart rate, SDNN, RMSSD, pNN50, VLF, LF, HF, total power, LF/HF, normalized LF and HF, Poincare SD1/SD2, and sample entropy. The goal is to compare the record with the Task Force reference values and to test robustness to an ectopic beat and to a shorter window.

### 6.2 Series arithmetic

- Duration: 300 s.
- Mean NN: 0.8337 s.
- Heart rate: 60/0.8337 = 71.97 bpm.
- Number of intervals: 360, which is consistent with 300/0.8337 = 359.8.
- Number of successive differences: 359.

The series is generated with a low-frequency modulation at 0.10 Hz and a high-frequency modulation at 0.25 Hz, plus Gaussian noise. The modulation amplitudes are 3.5% and 3.2% of the mean RR, which produce an LF power near the reference and an HF power slightly below it.

### 6.3 Time-domain arithmetic

| Feature | Formula | Value |
|---|---|---:|
| Mean NN | Σxi/N | 0.8337 s |
| Heart rate | 60/mean NN | 71.97 bpm |
| SDNN | √(Σ(xi-m)²/(N-1)) | 30.9 ms |
| RMSSD | √(Σ(xi+1-xi)²/(N-1)) | 30.9 ms |
| pNN50 | count(|diff|>50 ms)/(N-1) | 10.9% |

The SDNN and RMSSD are nearly equal. That is a coincidence of this record: the low-frequency and high-frequency modulations happen to produce similar total and successive variation. In many records, SDNN is larger than RMSSD because it includes the slow trends that RMSSD differences away.

The relationship between the features is worth stating:

| Feature | Includes slow variation | Includes fast variation | Sensitive to ectopy |
|---|---|---|---|
| SDNN | Yes | Yes | Moderate |
| RMSSD | No | Yes | High |
| pNN50 | No | Yes | High |
| LF power | Yes (0.04-0.15 Hz) | No | Moderate |
| HF power | No | Yes (0.15-0.4 Hz) | Moderate |

The table explains why SDNN and RMSSD can diverge. A record with a slow trend has a high SDNN and a normal RMSSD; a record with frequent ectopy has a normal SDNN and a high RMSSD. The pair is more informative than either alone.

The pNN50 of 10.9% means that 39 of the 359 successive differences exceed 50 ms. A pNN50 of 10.9% is within the range seen in healthy adults at rest, though the normative values depend on age and recording length.

### 6.4 Frequency-domain arithmetic

The RR series is interpolated at 4 Hz, detrended, windowed with a Hann window, and zero-padded to 2,048 samples. The frequency resolution is 4/2,048 = 0.00195 Hz.

| Band | Range | Power (ms²) | Normalized (%) |
|---|---|---:|---:|
| VLF | 0.0033-0.04 Hz | 24 | — |
| LF | 0.04-0.15 Hz | 1,125 | 59.2 |
| HF | 0.15-0.4 Hz | 776 | 40.8 |
| Total | 0.0033-0.4 Hz | 1,925 | — |
| LF/HF | — | 1.45 | — |

The Task Force reference for 5 min supine records is total power 3,466 ± 1,018 ms², LF 1,170 ± 416 ms², HF 975 ± 203 ms², and LF/HF about 1.5 to 2.0. The record's LF is close to the reference mean, the HF is about one standard deviation below, and the total power is about 1.5 standard deviations below. The LF/HF of 1.45 is at the low end of the reference range.

The normalized units are LF/(total - VLF) × 100 = 1,125/1,901 × 100 = 59.2% and HF/(total - VLF) × 100 = 776/1,901 × 100 = 40.8%. The two sum to 100% by construction.

The band boundaries are conventions, and the power near a boundary can move between bands.

| Boundary | Lower band | Upper band | Effect of a 0.01 Hz shift |
|---|---|---|---|
| 0.04 Hz | VLF | LF | Small |
| 0.15 Hz | LF | HF | Moderate near the boundary |
| 0.4 Hz | HF | Above | Small in this record |

A record with a strong 0.15 Hz component is sensitive to the boundary. The Task Force definitions are the reference, and a study that uses different boundaries should report them.

### 6.5 Nonlinear arithmetic

The Poincare plot has SD1 = √(0.5) × SD of successive differences and SD2 = √(2 × SDNN² - 0.5 × SDdiff²). With SDNN 30.9 ms and SDdiff 43.7 ms (the RMSSD is the root mean square, not the SD, so the SD of the differences is slightly different), SD1 is about 30.9 ms and SD2 is about 30.9 ms. The two are similar because the series is nearly symmetric in the Poincare plane.

Sample entropy with m = 2 and r = 0.2 × SD is 1.939. A higher value means less regularity. The value is in the range reported for healthy adults at rest, but the range is wide and the parameter choice matters.

### 6.6 Diagnostic 1: one ectopic beat

A single ectopic beat is inserted at beat 100, with an interval of 0.45 s. The features change as follows.

| Feature | Clean | With ectopic | Change |
|---|---:|---:|---:|
| SDNN | 30.9 ms | 36.9 ms | +19.4% |
| RMSSD | 30.9 ms | 43.0 ms | +39.2% |
| LF/HF | 1.45 | 1.10 | -24.1% |

The ectopic beat is a single event in 360 beats, yet it changes RMSSD by 39%. The mechanism is the squaring of the two large successive differences around the ectopic beat. The change is large enough to move a subject from the normal range to an abnormal range on some reference scales.

The diagnostic is to inspect the RR series for intervals outside the physiological range. A 0.45 s interval is 125 bpm, which is plausible during exercise but not at rest. The editing rule should remove it and interpolate, and the number of edited beats should be reported.

The ectopic arithmetic is worth showing. The two successive differences around the ectopic beat are approximately 0.45 - 0.83 = -0.38 s and 0.83 - 0.45 = +0.38 s. Their squares are 0.144 s² each. The normal differences have squares around 0.0002 s². The mean square over 359 differences with two outliers is therefore dominated by the outliers: (0.288 + 359×0.0002)/359 = 0.00100 s², and the square root is 0.0316 s, or 31.6 ms. The observed 43.0 ms is higher because the interpolation and the changed series add further variation.

The point is that the squared term makes RMSSD a non-robust statistic. A single outlier contributes 0.288 s² to a sum that would otherwise be 0.07 s². A robust alternative is the median of the absolute successive differences, which is insensitive to a single outlier, but it is not the standard feature and is not comparable with published values.

### 6.7 Diagnostic 2: a shorter window

The same record is analysed with the first 2 min instead of the full 5 min.

| Feature | 5 min | 2 min | Change |
|---|---:|---:|---:|
| SDNN | 30.9 ms | 30.0 ms | -2.9% |
| RMSSD | 30.9 ms | 28.8 ms | -6.8% |
| LF/HF | 1.45 | not reported | — |

The SDNN is relatively stable because the slow modulation is still captured. The RMSSD is 6.8% lower because the shorter window contains fewer samples and a different realization of the noise. The LF/HF is not reported because a 2 min window gives 0.0083 Hz resolution, which is marginal for the LF band and inadequate for a stable ratio.

The diagnostic is the window sensitivity. A feature that changes by more than 10% when the window is halved is not comparable across studies that use different windows. The Task Force recommends 5 min for short-term records, and the recommendation exists because of this sensitivity.

### 6.8 Varying one assumption

Suppose the editing rule retains the ectopic beat instead of removing it. RMSSD rises from 30.9 to 43.0 ms, a 39% change. A model trained on edited data and deployed on unedited data will see a feature distribution shifted by an amount that is larger than most physiological effects.

Suppose the interpolation rate is 1 Hz instead of 4 Hz. The Nyquist frequency is 0.5 Hz, which is above the HF band edge of 0.4 Hz, so the HF band is preserved in principle. But the interpolation at 1 Hz is coarse relative to the beat-to-beat variation, and the HF power is attenuated by the interpolation. The feature changes without a change in the signal.

Suppose the HF band is defined as 0.15-0.5 Hz instead of 0.15-0.4 Hz. The wider band includes more noise and possibly a harmonic, and the HF power rises. The LF/HF falls. The band definition is a convention, and the feature depends on it.

### 6.9 Conclusion of the case

The record has a mean NN of 0.8337 s, a heart rate of 71.97 bpm, SDNN 30.9 ms, RMSSD 30.9 ms, pNN50 10.9%, LF 1,125 ms², HF 776 ms², LF/HF 1.45, normalized LF 59.2%, normalized HF 40.8%, and sample entropy 1.939. The values are within or near the Task Force reference ranges for 5 min supine records.

The two diagnostics (ectopic beat, shorter window) show that the features are sensitive to choices that are not physiological. The assumption that matters most is the editing rule, because a single ectopic beat changes RMSSD by 39%. The second is the window, because a 2 min record is not a 5 min record.

### 6.9.1 A feature summary table

| Feature | Value | Unit | Window | Method |
|---|---:|---|---:|---|
| Mean NN | 0.8337 | s | 5 min | mean |
| Heart rate | 71.97 | bpm | 5 min | 60/mean |
| SDNN | 30.9 | ms | 5 min | SD |
| RMSSD | 30.9 | ms | 5 min | successive differences |
| pNN50 | 10.9 | % | 5 min | >50 ms |
| VLF | 24 | ms² | 5 min | 0.0033-0.04 Hz |
| LF | 1,125 | ms² | 5 min | 0.04-0.15 Hz |
| HF | 776 | ms² | 5 min | 0.15-0.4 Hz |
| Total | 1,925 | ms² | 5 min | 0.0033-0.4 Hz |
| LF/HF | 1.45 | — | 5 min | ratio |
| SD1 | 30.9 | ms | 5 min | Poincare |
| Sample entropy | 1.939 | — | 5 min | m=2, r=0.2 SD |

The table is the data dictionary in miniature. A reader can reproduce every value from the definitions and the record.

### 6.10 What the case does not show

The case is a single synthetic resting record. It does not show:

- **Age and sex effects.** HRV declines with age and differs by sex; the reference ranges are age- and sex-specific.
- **Medication effects.** Beta-blockers, antidepressants, and caffeine change HRV.
- **Non-stationarity.** Exercise, stress, and sleep change the spectrum within minutes.
- **Device effects.** A PPG-derived RR series has different timing errors from an ECG-derived series.
- **Recording length beyond 5 min.** A 24 h record has different features and different norms.
- **Respiratory coupling.** The respiratory rate is not measured, so RSA cannot be separated from other HF sources.

The omissions are the agenda for a validation. The arithmetic transfers; the normative comparison does not.

A reader who wants to extend the case should add a second subject with a different age and a different HRV, and compare the features after age adjustment. The extension requires a normative model, not a second single record.

A second extension is to add a second device with a different sampling rate. The comparison isolates the device effect from the physiology, which is the first step in a cross-device validation.

A third extension is to add a respiratory channel and compute RSA. The channel turns an HF feature into a coupling feature, and the coupling is more specific than the power. The comparison is the beginning of a normative analysis, and it requires a sample, not a single record.

A final check on the case is the unit conversion. The conversion is a one-line error that changes the interpretation by six orders of magnitude.


## 7. Code, query, or calculation

The script synthesises a 5 min RR series, computes time-domain, frequency-domain, and nonlinear features, and tests the effect of an ectopic beat and a shorter window. It uses only the standard library.

```python
import math, random

def synth_rr(n=360, mean=0.8333, lf=0.035, hf=0.032, seed=1):
    rnd=random.Random(seed); rr=[]; t=0.0
    for i in range(n):
        v = mean*(1 + lf*math.sin(2*math.pi*0.10*t)
                    + hf*math.sin(2*math.pi*0.25*t)
                    + rnd.gauss(0,0.015))
        rr.append(v); t += v
    return rr

def time_metrics(rr):
    n=len(rr); m=sum(rr)/n
    sdnn=math.sqrt(sum((v-m)**2 for v in rr)/(n-1))
    diffs=[rr[i+1]-rr[i] for i in range(n-1)]
    rmssd=math.sqrt(sum(d*d for d in diffs)/len(diffs))
    pnn50=sum(1 for d in diffs if abs(d)>0.05)/len(diffs)*100
    return m,sdnn,rmssd,pnn50

def fft(x):
    n=len(x)
    if n==1: return [complex(x[0])]
    if n&(n-1): raise ValueError("power of two")
    ev=fft(x[0::2]); od=fft(x[1::2]); out=[0j]*n
    for k in range(n//2):
        t=math.e**(-2j*math.pi*k/n)*od[k]
        out[k]=ev[k]+t; out[k+n//2]=ev[k]-t
    return out

def hann(n): return [0.5-0.5*math.cos(2*math.pi*i/(n-1)) for i in range(n)]

def freq_metrics(rr, fs=4.0):
    times=[]; t=0.0
    for v in rr:
        times.append(t); t+=v
    dur=times[-1]; n=int(dur*fs)
    y=[]; j=0
    for i in range(n):
        tt=i/fs
        while j < len(times)-2 and times[j+1] < tt: j+=1
        t0,t1=times[j],times[j+1]; v0,v1=rr[j],rr[j+1]
        y.append(v0+(v1-v0)*(tt-t0)/(t1-t0) if t1>t0 else v0)
    m=sum(y)/len(y); y=[v-m for v in y]
    N=1
    while N < len(y): N*=2
    w=hann(len(y)); sw=sum(w)
    seg=[y[i]*w[i] for i in range(len(y))]+[0.0]*(N-len(y))
    X=fft(seg); df=fs/N
    p=[0.0]*(N//2)
    for k in range(1,N//2): p[k]=2*abs(X[k])**2/(sw**2)
    def band(lo,hi): return sum(p[k] for k in range(1,len(p)) if lo<=k*df<hi)
    lf=band(0.04,0.15); hf=band(0.15,0.4)
    vlf=band(0.0033,0.04); tp=band(0.0033,0.4)
    return lf,hf,vlf,tp,df

def sampen(x,m=2,r_factor=0.2):
    n=len(x); sd=math.sqrt(sum((v-sum(x)/n)**2 for v in x)/(n-1))
    r=r_factor*sd
    def count(mm):
        c=0
        for i in range(n-mm):
            for j in range(i+1,n-mm):
                if max(abs(x[i+k]-x[j+k]) for k in range(mm))<r: c+=1
        return c
    A=count(m+1); B=count(m)
    return -math.log(A/B) if A>0 and B>0 else float('nan')

rr=synth_rr()
m,sdnn,rmssd,pnn50=time_metrics(rr)
print("beats=%d mean_RR=%.4f HR=%.2f SDNN=%.1f_ms RMSSD=%.1f_ms pNN50=%.1f%%"
      % (len(rr),m,60/m,sdnn*1000,rmssd*1000,pnn50))
lf,hf,vlf,tp,df=freq_metrics(rr)
print("LF=%.0f HF=%.0f VLF=%.0f TP=%.0f LF/HF=%.2f df=%.4f"
      % (lf*1e6,hf*1e6,vlf*1e6,tp*1e6,lf/hf,df))
print("SampEn=%.3f" % sampen(rr))

rr2=rr[:120]
m2,s2,r2,p2=time_metrics(rr2)
print("2min: SDNN=%.1f RMSSD=%.1f" % (s2*1000,r2*1000))

rr3=list(rr); rr3[100]=0.45
m3,s3,r3,p3=time_metrics(rr3)
lf3,hf3,_,_,_=freq_metrics(rr3)
print("ectopic: SDNN=%.1f RMSSD=%.1f LF/HF=%.2f" % (s3*1000,r3*1000,lf3/hf3))

print("LF_norm=%.1f HF_norm=%.1f" % (lf/(tp-vlf)*100, hf/(tp-vlf)*100))
```

Expected output:

```text
beats=360 mean_RR=0.8337 HR=71.97 SDNN=30.9_ms RMSSD=30.9_ms pNN50=10.9%
LF=1125 HF=776 VLF=24 TP=1925 LF/HF=1.45 df=0.0020
SampEn=1.939
2min: SDNN=30.0 RMSSD=28.8
ectopic: SDNN=36.9 RMSSD=43.0 LF/HF=1.10
LF_norm=59.2 HF_norm=40.8
```

Three implementation notes.

The interpolation uses the cumulative beat times and linear interpolation between adjacent RR values. This is the standard method, but it assumes that the RR value is a sample at the beat time, not an interval centred between beats. The choice shifts the series by half a beat and changes the phase, not the power.

The periodogram is normalized by the window sum squared, so the integral approximates the variance of the series. The values are in s² and are multiplied by 1e6 to report ms². A reader who forgets the conversion will report values a million times too small.

Sample entropy is O(N²) in this implementation. At 360 beats, it is fast; at 100,000 beats, it is not. A production implementation uses a KD-tree or a bucket-assisted algorithm, but the definition is the same.

The script does not implement the Poincare SD1/SD2 or the triangular index, although the text refers to them. The omission is deliberate: the code is long enough to demonstrate the main families, and the missing features are simple arithmetic on the same series. A reader who wants them can add them in a few lines.

## 8. Data, standards, and regulatory context

Feature definitions are less regulated than device performance, but they are standardized by professional consensus and by the need for comparability.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| Task Force 1996 HRV standards | Time- and frequency-domain definitions | Professional consensus | SDNN, RMSSD, LF, HF, bands |
| AASM V3 | Sleep feature definitions | Professional | Spindles, slow waves, stages |
| AHA/ACCF/HRS ECG statements | ECG morphology and intervals | Professional | QRS, QT, ST definitions |
| PhysioNet databases | Reference data and annotations | Research infrastructure | Benchmark features |
| NeuroKit2 documentation | Open implementation of features | Software documentation | Method names and defaults |
| ISO/IEC 25010 | Software quality | Voluntary | Reproducibility and maintainability |
| FDA device software guidance | Software documentation | Binding through review | Feature definitions in the submission |
| EU MDR | Clinical evaluation | Binding in the EU | Feature validity evidence |

### 8.1 The Task Force definitions

The 1996 Task Force document is the reference for HRV features. Its key definitions are:

- SDNN: standard deviation of all NN intervals.
- SDANN: standard deviation of the averages of NN intervals in 5 min segments.
- RMSSD: root mean square of successive differences.
- pNN50: proportion of successive differences greater than 50 ms.
- VLF: 0.0033-0.04 Hz.
- LF: 0.04-0.15 Hz.
- HF: 0.15-0.4 Hz.
- Total power: 0.0033-0.4 Hz.
- Normalized units: LF or HF divided by total power minus VLF, times 100.

The document also recommends 5 min records for short-term frequency-domain analysis and 24 h records for long-term time-domain analysis. It warns that SDNN is not comparable across recording lengths and that LF/HF should not be interpreted as a direct measure of sympathovagal balance.

The document's normative table is a reference, not a rule. The 5 min supine values were derived from a specific sample, and the age and sex effects are large. A product that compares a user with the table without adjustment is making an unstated assumption about the population.

### 8.2 What the standards do not cover

Three gaps.

First, the standards define the features but not the editing rule. Two laboratories can compute SDNN from the same recording with different ectopic handling and report different values. The rule must be documented, but it is not standardized.

Second, the standards do not cover device-specific features such as PPG pulse morphology or wearable-derived respiratory rate. Those features have no consensus definition, and each product defines its own.

Third, the standards do not cover the feature-to-model interface: normalization, selection, and aggregation. A feature that is standardized in isolation can be used in a way that is not.

### 8.2.1 A feature specification template

A reproducible feature specification has the following fields. The template is the answer to the "feature table that cannot be reproduced" failure.

| Field | Example | Why it matters |
|---|---|---|
| Name | rmssd_5min | Unique, versioned |
| Definition | √(mean of squared successive NN differences) | Reproducibility |
| Unit | ms | Comparability |
| Window | 300 s | Length dependence |
| Editing | Remove >20% from median; interpolate | Ectopy sensitivity |
| Input | NN series from ECG R peaks | Source signal |
| Fiducial | R peak, maximum | Timing bias |
| Filter | 0.5-40 Hz, 4th order | Morphology effects |
| Normalization | None | Scale |
| Aggregation | Mean over epochs | Patient-level score |
| Reference | Task Force 1996 | Normative comparison |
| Version | 1.2.0 | Change control |

A feature table with these fields can be reimplemented from the document. A feature table with only the name cannot.

### 8.2.2 Normative values and their limits

The Task Force reference values for 5 min supine records are a useful comparison, but they are not universal norms. They were derived from a specific population, with a specific recording protocol, and they vary with age, sex, and fitness. A feature value outside the reference range is a prompt for investigation, not a diagnosis.

The same is true for the 24 h values: SDNN 141 ± 39 ms, SDANN 127 ± 35 ms, RMSSD 27 ± 12 ms. A 24 h SDNN below 50 ms is described as highly depressed and below 100 ms as moderately depressed in the clinical literature. These thresholds are prognostic associations, not diagnostic criteria.

A product that reports "low HRV" without the window, the population, and the reference is making a comparison it cannot support.

### 8.2.3 The feature–model interface

A feature is standardized in isolation, but it enters a model through an interface that is not standardized. The interface has four parts.

- **Scaling.** A model that uses gradient descent needs features on comparable scales. A tree model does not. The scaling should be fit on the training data and applied to the test data, not fit on the full dataset.
- **Imputation.** A missing feature must be imputed or the row dropped. The imputation should be fit on the training data and should not use the outcome.
- **Encoding.** Categorical context features (device, site, activity) must be encoded. A one-hot encoding of a site with 50 levels adds 50 features and can leak site-specific prevalence.
- **Temporal order.** For time-series models, the features must be computed with only past information. A rolling feature that includes future samples is a look-ahead leak.

The interface is where training-serving skew enters. A feature computed in a batch job with a 5 min window and in a streaming job with a 30 s window is two different features with the same name. The feature store exists to prevent this, but it only works if the window and the method are part of the stored definition.

### 8.3 Regulatory treatment of features

A feature is not a device claim by itself. A model that uses features to make a diagnostic claim is the regulated object. The feature definitions are part of the software documentation, and a change in a feature definition is a change in the device function.

The distinction matters when a model is updated. A retrained model with the same features is a change in the model; a model with a new feature definition is a change in the measurement. The second is more likely to require new evidence, because the input to the model has changed.

A practical rule for change control: classify every change as a change to the data, the feature, the model, or the decision. A change to the data or the feature is upstream of the model and can invalidate the model's validation. A change to the decision threshold is downstream and can be managed with a calibration study.

The FDA guidance on predetermined change control plans is relevant here: a model that is retrained on new features is a changed device, and the change control plan should describe how feature changes are managed. A silent change in an SDNN editing rule is a change that a reviewer would want to see.

## 9. Industry practice and case evidence

### 9.1 Open-source feature libraries

NeuroKit2 provides implementations of ECG, PPG, EDA, EMG, and respiratory features, including 95 HRV indices across time, frequency, and nonlinear domains. The documentation states the method names and the sampling rate requirement, which makes the features reproducible if the version is recorded.

The practical lesson is that a library function is not a definition. `nk.hrv()` returns a table of 95 features, and the user must still know which ones are appropriate, which are redundant, and which are sensitive to the window. A library accelerates computation; it does not replace the feature specification.

### 9.2 Feature sets in public challenges

The PhysioNet challenges have used feature-based and deep-learning approaches. In the 2020 and 2021 ECG challenges, many top teams used a combination of handcrafted features and raw-signal models. The handcrafted features were often standard intervals and amplitudes, not exotic descriptors.

The documented lesson is that a small set of well-defined features is competitive when the data are limited. The inference is that feature engineering remains relevant even in the deep-learning era, particularly for small clinical datasets.

### 9.3 Wearable feature pipelines

Consumer wearable pipelines typically compute features on the device or in the cloud from PPG and accelerometry: heart rate, RR intervals, perfusion index, motion, and sleep features. The features are often proprietary, and the definitions are not published.

The documented consequence is that cross-device comparisons are difficult. Two devices that report "HRV" may use different windows, different editing rules, and different bands. A study that pools them is pooling different measurements with the same name.

### 9.3.1 What public feature sets contain

A review of public physiological feature sets shows a consistent pattern.

- **ECG**: intervals (PR, QRS, QT), amplitudes, axis, ST level, and HRV features.
- **PPG**: heart rate, RR, perfusion index, pulse amplitude, rise time, and pulse width.
- **EEG**: band powers, spectral edge, entropy, spindle and slow-wave counts.
- **Respiration**: rate, variability, apnea count, and effort amplitude.
- **CGM**: mean, SD, CV, TIR, TBR, TAR, and GMI.

The pattern is that the features are mostly simple, interpretable, and close to the clinical measurement. The exotic features (fractal dimensions, symbolic dynamics) are used in research but rarely in products. Inference: the product feature set is conservative because it must be validated and explained.

A second pattern is that the number of features in a published model is often smaller than the number computed. A study may compute 95 HRV indices and use 5 in the model. The selection is the scientific contribution, and it is often under-reported.

### 9.4 A worked feature audit for a wearable study

A wearable study plans to report HRV, sleep, and respiratory rate from a wrist device. The feature audit has four steps.

1. **List every feature and its definition.** The list contains 40 features; 12 are HRV, 8 are sleep, 6 are respiratory, and 14 are activity or context.
2. **Check the units and windows.** Two HRV features use a 5 min window and one uses a 1 min window; the 1 min feature is not comparable with the others and is removed or relabelled.
3. **Check the edit rate and the coverage.** The HRV features are computed on 82% of the nights, with a median edit rate of 3% and a maximum of 18%. The high-edit nights are flagged, and a sensitivity analysis excludes them.
4. **Check the device dependence.** The PPG-derived RR has a timing error of about 10 ms at 25 Hz sampling, which is small relative to RR but large relative to the HF band. The HF feature is reported with a device-specific correction or is not reported across devices.

The audit is not a statistical analysis; it is a documentation and comparability check. It is cheap, and it catches the failures that a model evaluation would not.

The inference is that a feature audit should precede any modelling. A model trained on incomparable features can produce a good number for the wrong reason, and the number will not replicate.

### 9.4.1 Feature engineering for a clinical trial endpoint

A trial that uses a digital endpoint has stricter requirements than a research study.

- **Pre-specified features.** The primary endpoint feature must be defined before the trial starts, with the window, the editing rule, and the aggregation.
- **Blinded computation.** The feature should be computed without knowledge of treatment assignment, or the computation should be demonstrably independent of the arm.
- **Missing data plan.** The trial protocol must state how missing feature values are handled and how the primary analysis treats them.
- **Site and device effects.** The feature must be comparable across sites and devices, or the analysis must adjust for them.
- **Version lock.** The feature code and the device firmware should be locked for the trial duration; a mid-trial update is a protocol deviation.

The documented practice in digital endpoint trials is to treat the feature like a laboratory assay: it has a method, a reference range, a quality control, and a version. The inference is that the feature, not only the model, is the endpoint.

### 9.5 Feature stores and reproducibility

A feature store is a system that computes and serves features consistently across training and inference. The main benefit is the elimination of training-serving skew: the same definition is used in both paths.

The documented practice in mature teams is to version the feature definition and the code together, and to test that the training and serving values match on a sample. A feature that is computed differently in the two paths is a silent bug that no offline metric will catch.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Window not stated | SDNN differs across studies | Length dependence | Non-comparable features | State window |
| Ectopic retained | RMSSD spikes | Squared differences | False high HRV | Edit; report count |
| Ectopic removed for arrhythmia | Event lost | Editing removes signal | Missed diagnosis | Task-specific rule |
| Interpolation too coarse | HF attenuated | Resampling | Underestimated HF | 4 Hz or higher |
| Band edge changed | LF/HF moves | Convention | Non-comparable | State bands |
| LF/HF overinterpreted | Ratio reported alone | Unstable when HF low | Misleading | Report components |
| Detrending omitted | VLF inflated | Trend leakage | Wrong band power | Detrend |
| Normalization hides signal | Between-subject effect lost | Z-score within subject | Null result | Choose normalization by question |
| Selection on full data | Optimistic performance | Leakage | Invalid model | Nested CV |
| Duplicate features | Importance unstable | Collinearity | Uninterpretable | Correlation filter |
| Morphology filter mismatch | Width differs by 10 ms | Bandwidth | Non-comparable | Harmonize filters |
| Sampling rate mismatch | Feature shift | Aliasing or resolution | Non-comparable | Resample; document |
| Unit missing | Values 1,000× off | ms versus s | Wrong thresholds | Data dictionary |
| Version drift | Feature changes silently | Code update | Training-serving skew | Version features |
| Aggregation ambiguous | Patient score varies | Mean versus median | Non-reproducible | State rule |
| Look-ahead feature | Too-good performance | Future information | Invalid model | Time-aware split |
| Baseline chosen after outcome | Group difference | Post-hoc baseline | Bias | Predefine baseline |
| Fragmentation | Event double-counted | Window overlap | Inflated count | Merge events |
| Context leakage | Site predicts label | Device or clinic shortcut | Poor deployment | Stratify by site |
| Unit conversion error | Values 1,000× off | ms versus s | Wrong threshold | Data dictionary |
| Norm mismatch | "Low HRV" | Wrong population | Misleading | State reference |
| Look-ahead in rolling window | Excellent backtest | Future samples | Invalid model | Time-aware features |
| Site as shortcut | Site predicts label | Confounded deployment | Poor external validity | Site-held-out test |
| Imputation leak | Optimistic score | Outcome used in imputation | Invalid | Fit on train only |
| Device firmware change | Feature shift | Sampling or filter change | Drift | Version and re-verify |
| Cohort change | Norms no longer apply | Different age or disease | Misleading comparison | Recompute norms |
| Overlapping windows | Too many "independent" samples | Pseudoreplication | Narrow CIs | Account for overlap |
| Feature scale mismatch | Gradient instability | Unscaled inputs | Slow or failed training | Scale in train |
| Missing feature | Row dropped | No imputation | Selection bias | Impute; report |
| Unit drift | Values shift | Code or export change | Non-comparable | Validate units |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The series is stationary within the window | Spectral features | Trends, exercise | Split window; compare |
| Ectopic beats are identifiable | Editing | Noise, aberrant conduction | Inspect intervals |
| The editing rule is appropriate | HRV validity | Arrhythmia as signal | Task-specific review |
| Interpolation preserves the band | Frequency features | Coarse rate | Compare rates |
| The band definitions are fixed | Comparability | Different conventions | State bands |
| Normalization preserves the effect | Between-subject comparison | Removing signal | Compare raw and normalized |
| The feature is robust to noise | Deployment | Motion, contact | Noise injection |
| The feature is robust to sampling | Cross-device | Different rates | Resample test |
| The window matches the reference | Normative comparison | Different length | Use reference window |
| The aggregation matches the decision | Patient-level score | Mean versus median | Predefine |
| The feature definition is versioned | Reproducibility | Silent update | Version control |
| The feature set is not leaky | Valid performance | Selection on test | Nested CV |
| The baseline is predefined | Fair comparison | Post-hoc choice | Pre-register |
| The unit is consistent | Thresholds | ms versus s | Check dictionary |
| The reference population matches | Normative claim | Different age or sex | Stratify |
| The feature is available at inference | Deployment | Future or lab-only input | Feature audit |
| The cohort matches the norm | Normative comparison | Different population | Stratify |
| The windows are independent | Inference | Overlap | Use effective N |
| The firmware is stable | Reproducibility | Update mid-study | Lock version |

## 12. Exercises

### Exercise 1. Time-domain features (application, 30 min)

**Deliverable:** compute SDNN, RMSSD, and pNN50 for a 10-interval series: 0.80, 0.82, 0.81, 0.79, 0.83, 0.80, 0.84, 0.81, 0.80, 0.82 s.

**Acceptance criteria:** formulas shown; units in ms; pNN50 denominator stated.

**Time:** 30 minutes.

### Exercise 2. Ectopic sensitivity (critique, 30 min)

**Deliverable:** insert a 0.45 s interval into the series from Exercise 1 and recompute. Explain the change.

**Acceptance criteria:** all three features recomputed; the squaring mechanism explained; an editing recommendation.

**Time:** 30 minutes.

### Exercise 3. Frequency bands (recall, 25 min)

**Deliverable:** state the Task Force VLF, LF, HF, and total bands, and compute the number of 0.002 Hz bins in each.

**Acceptance criteria:** bands correct; bin counts correct; the 5 min recommendation stated.

**Time:** 25 minutes.

### Exercise 4. Window sensitivity (application, 35 min)

**Deliverable:** compute SDNN and RMSSD for 1 min, 2 min, and 5 min segments of a synthetic series, and plot the values against window length.

**Acceptance criteria:** three windows; the trend explained; a recommendation for the decision.

**Time:** 35 minutes.

### Exercise 5. Normalization (design, 30 min)

**Deliverable:** choose a normalization for EMG amplitude in a within-subject vs between-subject study, and justify each.

**Acceptance criteria:** the two choices differ; the effect on the signal stated; the reference named.

**Time:** 30 minutes.

### Exercise 6. Feature selection (application, 45 min)

**Deliverable:** from a 20-feature table, remove correlated features and near-zero-variance features, and justify the retained set.

**Acceptance criteria:** correlation threshold stated; domain rationale for each removal; no selection on test data.

**Time:** 45 minutes.

### Exercise 7. Data dictionary (design, 40 min)

**Deliverable:** a data dictionary for five features with name, definition, unit, window, method, and limitation.

**Acceptance criteria:** all fields present; units correct; limitations specific.

**Time:** 40 minutes.

### Exercise 8. Robustness report (design, 60 min)

**Deliverable:** a one-page robustness report for one feature, testing window, ectopy, noise, and sampling rate.

**Acceptance criteria:** four tests; quantitative changes; a recommendation to keep or drop the feature.

**Time:** 60 minutes.

### Exercise 9. Normative comparison (application, 25 min)

**Deliverable:** compare a record with SDNN 25 ms and RMSSD 35 ms to the Task Force 5 min reference, and explain why the two features point in different directions.

**Acceptance criteria:** both values compared; the slow-versus-fast mechanism explained; a conclusion that does not overclaim.

**Time:** 25 minutes.

### Submission standard

Feature calculations must show the formula, the unit, the window, and the editing rule. A feature table without a dictionary is returned. A robustness report must test at least three perturbations and report the change for each. A selection exercise must state the criterion and perform the selection inside the training fold.

The standard is deliberately strict on documentation because the failure mode in this topic is under-specification, not arithmetic. An answer with the right number and the wrong definition is a wrong answer.

## 13. Self-check answers

**1. Compute the mean of 0.80, 0.82, 0.81, 0.79, 0.83, 0.80, 0.84, 0.81, 0.80, 0.82 s.** Sum = 8.12 s; mean = 0.812 s.

**2. Compute SDNN for the same series.** Deviations from 0.812: -0.012, 0.008, -0.002, -0.022, 0.018, -0.012, 0.028, -0.002, -0.012, 0.008. Squares sum to 0.002128; divided by 9 = 0.0002364; square root = 0.01538 s = 15.4 ms.

**3. Compute RMSSD.** Successive differences: 0.02, -0.01, -0.02, 0.04, -0.03, 0.04, -0.03, -0.01, 0.02. Squares sum to 0.0064; divided by 9 = 0.000711; square root = 0.02667 s = 26.7 ms.

**4. Compute pNN50.** All successive differences are below 50 ms, so pNN50 = 0%.

**5. What are the Task Force bands?** VLF 0.0033-0.04 Hz, LF 0.04-0.15 Hz, HF 0.15-0.4 Hz, total 0.0033-0.4 Hz.

**6. How many 0.002 Hz bins are in the LF band?** (0.15 - 0.04)/0.002 = 55 bins.

**7. Why is a 10 s window inadequate for LF?** The resolution is 0.1 Hz, which is wider than the LF band. The band cannot be resolved.

**8. What does RMSSD measure?** Short-term beat-to-beat variation, dominated by high-frequency (vagal) modulation.

**9. Why does one ectopic beat change RMSSD so much?** The two large successive differences around the ectopic beat are squared, and they dominate the mean square.

**10. What is SD1?** The short-term dispersion in the Poincare plot, approximately RMSSD/√2.

**11. Why is LF/HF not a direct measure of sympathovagal balance?** LF is not purely sympathetic and HF is not purely vagal; the ratio mixes both and is unstable when HF is small.

**12. What is the difference between NN and RR?** NN intervals are normal-to-normal intervals, excluding ectopic and artifact beats. RR intervals include all beats.

**13. Why must feature selection be inside cross-validation?** Selecting on the full dataset uses test information, which inflates the apparent performance.

**14. What is training-serving skew?** A difference between the feature computed at training and the feature computed at inference, often caused by different code or data pipelines.

**15. What is the minimum documentation for a feature?** Name, definition, unit, window, editing rule, normalization, method, version, and limitation.

**16. Compute the LF normalized units for LF 800 ms², HF 400 ms², VLF 50 ms².** Total = 1,250; non-VLF = 1,200; LF normalized = 800/1,200 × 100 = 66.7%; HF normalized = 400/1,200 × 100 = 33.3%.

**17. Why does a 2 min window underestimate RMSSD in this case?** Fewer differences are averaged, and the particular realization has less high-frequency variation. The estimate is noisier and, in this record, 6.8% lower.

**18. What is the difference between SD1 and RMSSD?** SD1 is the standard deviation of the projection of RR pairs onto the perpendicular to the line of identity, which is approximately RMSSD/√2. They measure the same short-term dispersion on different scales.

**19. Why is a feature with a 15% edit rate a quality problem?** A high edit rate means many beats were abnormal or misdetected. The remaining series may not represent the rhythm, and the feature values are computed on a selected subset.

**20. Compute the total power for VLF 24, LF 1,125, HF 776 ms².** 24 + 1,125 + 776 = 1,925 ms².

**21. Why does the LF/HF ratio fall when an ectopic beat is added?** The ectopic adds broadband power, which raises HF more than LF in this record, so the ratio falls from 1.45 to 1.10.

**22. What is the difference between a feature and a biomarker?** A feature is a computed descriptor; a biomarker is a feature with a validated relationship to a biological process or outcome.

**23. Why is a feature with a strong univariate association not necessarily a good model input?** The association may be confounded by age, sex, or device, and the feature may be redundant with another feature. Univariate association is a screening step, not a selection criterion.

**24. What is the effective sample size for 10 overlapping 5 min windows with a 1 min step?** Approximately two independent 5 min segments, because the 10 windows cover about 14 min of unique time. The exact value depends on the autocorrelation.

**Rubric for Exercise 8 (robustness).** A strong report tests four perturbations, reports the change for each, and makes a keep-or-drop recommendation. A weak report tests one perturbation and concludes "stable."

**Rubric for Exercise 6 (selection).** A strong selection uses a correlation threshold, a variance threshold, and domain knowledge, and is performed inside the training fold. A weak selection removes features by name without a criterion.

## 14. Interview questions

**1. How do you compute HRV from an ECG?**
A strong answer names the R-peak detection, the NN editing, the window, and the time- and frequency-domain features. A weak answer says "use an HRV library."

**2. Why is SDNN not comparable across windows?**
A strong answer explains that SDNN includes slow variation, which accumulates over longer windows. A weak answer says "it depends on the data."

**3. What do you do with an ectopic beat?**
A strong answer distinguishes the HRV task from the arrhythmia task and states the editing rule. A weak answer removes all outliers.

**4. How do you choose a feature set?**
A strong answer starts from the decision, uses interpretable features, tests robustness, and selects inside cross-validation. A weak answer uses a library's default output.

**5. What is the LF/HF ratio and what does it not mean?**
A strong answer explains the bands, the components, and the contested interpretation. A weak answer calls it sympathovagal balance.

**6. How do you normalize EMG?**
A strong answer uses maximal voluntary contraction or a reference contraction, and states what the normalization removes. A weak answer divides by the mean.

**7. Why is feature documentation a regulatory issue?**
A strong answer explains that a feature definition is part of the device function and a change is a change in the claim. A weak answer says documentation is for reproducibility only.

**8. What is training-serving skew?**
A strong answer describes the different code paths and the silent distribution shift. A weak answer says "the model sees different data."

**9. How do you test a feature for robustness?**
A strong answer perturbs the window, the editing, the noise, and the sampling rate, and reports the change. A weak answer checks the mean and SD.

**10. Why is a 5 min window recommended for frequency-domain HRV?**
A strong answer cites the 0.0033 Hz resolution and the stationarity assumption. A weak answer says "it is standard."

**11. What is sample entropy and when would you use it?**
A strong answer describes template matching, the m and r parameters, and the length sensitivity. A weak answer says it measures complexity.

**12. What is the risk of 500 features on 200 subjects?**
A strong answer explains overfitting, collinearity, and selection leakage. A weak answer says "more data is better."

**13. How would you compare HRV between two devices?**
A strong answer harmonises the R-peak detection, the editing rule, the window, and the interpolation, then compares the features on simultaneous recordings. A weak answer compares the device outputs directly.

**14. Why is the feature dictionary a regulatory artefact?**
A strong answer explains that the definition is part of the device function and a change is a change in the claim. A weak answer says it is for reproducibility only.

**15. What would you do if a feature is stable under noise but unstable under ectopy?**
A strong answer fixes the editing rule, reports the edit rate, and tests the feature on edited and unedited data. A weak answer drops the feature without diagnosing the cause.

**16. How do you compare a 5 min HRV feature with a 24 h feature?**
A strong answer says you do not; the windows measure different things and the norms differ. A weak answer compares the means.

**17. What is the first thing you check when a feature's distribution changes after a device update?**
A strong answer checks the sampling rate, filter, and window before the physiology. A weak answer retrains the model.

**18. Why is a pre-specified endpoint feature important in a trial?**
A strong answer explains that post-hoc feature choice inflates type I error and undermines the endpoint. A weak answer says it is good practice.

## 15. Key terms

- **Feature** — computed descriptor of a signal.
- **Feature vector** — ordered set of features.
- **NN interval** — normal-to-normal RR interval.
- **SDNN** — standard deviation of NN intervals.
- **RMSSD** — root mean square of successive differences.
- **pNN50** — proportion of differences above 50 ms.
- **VLF** — very-low-frequency power.
- **LF** — low-frequency power.
- **HF** — high-frequency power.
- **LF/HF** — ratio of LF to HF.
- **Normalized units** — band share of non-VLF power.
- **Interpolation** — resampling the RR series.
- **Ectopic beat** — non-sinus beat.
- **Editing rule** — rule for removing or keeping beats.
- **Poincare plot** — RR pair scatter plot.
- **SD1** — short-term Poincare dispersion.
- **SD2** — long-term Poincare dispersion.
- **Sample entropy** — regularity measure.
- **Detrended fluctuation** — fractal scaling exponent.
- **Spectral edge** — frequency below a power fraction.
- **Morphology** — waveform shape features.
- **Rise time** — trough-to-peak time.
- **Pulse transit time** — ECG-to-PPG delay.
- **RSA** — respiratory sinus arrhythmia.
- **Normalization** — scaling to a reference.
- **Z-score** — standardized value.
- **Feature selection** — choosing a subset.
- **Leakage** — test information in training.
- **Look-ahead bias** — using future information.
- **Edit rate** — share of beats removed.
- **Batch effect** — technical variation across batches.
- **Training-serving skew** — different features in training and inference.
- **Feature store** — system for consistent feature computation.
- **Aggregation rule** — how epochs combine into a patient score.
- **Normative range** — reference values for a population.
- **Poincare SD1** — short-term dispersion.
- **Poincare SD2** — long-term dispersion.
- **Approximate entropy** — regularity measure.
- **Detrended fluctuation analysis** — fractal scaling method.
- **Fragmentation** — splitting one event across windows.
- **Reference range** — values expected in a defined population.
- **Quality control** — procedures that keep a measurement stable.
- **Version lock** — fixed code and firmware for a study.
- **Primary endpoint** — the pre-specified outcome.
- **Sensitivity analysis** — re-analysis under alternative assumptions.
- **Confounding** — a third variable associated with both.
- **Stratification** — analysis within subgroups.
- **External validity** — generalization beyond the study sample.
- **Reproducibility** — same result from same data and method.
- **Pseudoreplication** — treating correlated samples as independent.
- **Effective sample size** — independent information in correlated data.
- **Site effect** — variation attributable to the collection site.
- **Protocol deviation** — departure from the pre-specified plan.
- **Assay** — a defined measurement procedure.
- **Quality control sample** — a reference used to check stability.
- **Feature budget** — the number and cost of features a model can support.
- **Signal-to-noise ratio** — signal power divided by noise power.
- **Dimensionality** — number of features in a vector.
- **Curse of dimensionality** — degradation with many features and few samples.
- **Regularization** — penalizing model complexity.
- **Training fold** — data used to fit a model.
- **Validation fold** — data used to tune.
- **Test fold** — data used once for evaluation.
- **Nested cross-validation** — selection inside each fold.
- **Pre-specification** — defining the analysis before seeing data.
- **Post-hoc** — chosen after seeing data.
- **Type I error** — false positive.
- **Power** — probability of detecting a true effect.
- **Effect size** — magnitude of a difference.
- **Confidence interval** — range of plausible values.
- **Sensitivity** — true positive rate.
- **Specificity** — true negative rate.
- **Positive predictive value** — probability of disease given a positive test.
- **Calibration** — agreement between predicted and observed risk.
- **Discrimination** — ability to rank cases above non-cases.
- **Net reclassification** — change in risk categories.
- **Decision curve** — net benefit across thresholds.
- **External validation** — testing on a separate population.
- **Internal validation** — testing within the development population.
- **Data dictionary** — definitions of every field in a dataset.

## 16. Sources

All URLs accessed 25 October 2026.

1. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*. 1996;93:1043-1065. https://doi.org/10.1161/01.CIR.93.5.1043
2. Task Force. European Heart Journal version. https://doi.org/10.1093/oxfordjournals.eurheartj.a014868
3. NeuroKit2 documentation. Heart rate variability. https://neuropsychology.github.io/NeuroKit/examples/ecg_hrv/ecg_hrv.html
4. NeuroKit2 documentation. PPG functions. https://neuropsychology.github.io/NeuroKit/functions/ppg.html
5. PhysioNet. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/
6. PhysioNet. PTB-XL. https://physionet.org/content/ptb-xl/
7. Jarrin DC, et al. Measurement fidelity of heart rate variability signal processing. *Sleep*. 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC5500286/
8. Pham T, et al. Heart rate variability in psychology: a review of HRV indices. *Sensors*. 2021. https://doi.org/10.3390/s21165389
9. AHA/ACCF/HRS. Recommendations for the standardization and interpretation of the electrocardiogram. https://doi.org/10.1016/j.jacc.2007.01.024
10. US Food and Drug Administration. Content of Premarket Submissions for Device Software Functions. 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
11. Pham T, Lau ZJ, Chen SHA, Makowski D. Heart rate variability in psychology: a review of HRV indices and an analysis tutorial. *Sensors*. 2021;21(12):3998. https://doi.org/10.3390/s21123998
12. Shaffer F, Ginsberg JP. An overview of heart rate variability metrics and norms. *Frontiers in Public Health*. 2017;5:258. https://doi.org/10.3389/fpubh.2017.00258

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_respiration-and-other-signals.md`](05_respiration-and-other-signals.md)
- Next topic: [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md)
- Related topic: [`08_digital-biomarkers.md`](08_digital-biomarkers.md)
- Module survey file: [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
- Related topic: [`09_validation.md`](09_validation.md)
- Related topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
- Dataset: [`https://physionet.org/content/mitdb/`](https://physionet.org/content/mitdb/)
