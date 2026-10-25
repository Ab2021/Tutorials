# Photoplethysmography and Wearables: Optical Sensing, SpO2, Heart Rate, and Motion

> Module 11 — Physiological Signals and Wearables
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals), Topic 02 (ECG as a reference)
> Version 2026.3

## 1. Why this topic exists

Photoplethysmography (PPG) is the optical measurement of blood volume changes in tissue. A light-emitting diode (LED) illuminates the skin, a photodetector measures the transmitted or reflected light, and the pulsatile component of that light is converted into a waveform. It is the sensor behind the heart rate on almost every wrist-worn device, the oxygen saturation on a fingertip oximeter, and a growing number of screening features for atrial fibrillation (AF), sleep, and hypertension.

PPG is also the signal where the gap between the measurement and the marketing claim is widest. A wrist PPG measures light. Heart rate is a derived feature. Oxygen saturation is a derived feature that requires a calibration model. AF is a decision built on an irregularity feature that motion can mimic. Blood pressure is a research claim that requires a calibration that the device usually does not have. The chain from light to claim has more assumptions than the ECG chain, and the assumptions are less visible to the user.

Four failures recur.

**Motion artifact misread as physiology.** A wrist PPG during walking has large, irregular deflections from wrist movement. A peak detector fires on the motion peaks, the RR intervals look irregular, and an AF detector reports an irregular rhythm. The Apple Heart Study and subsequent work showed that AF notification from PPG requires confirmation, partly because motion and ectopy can produce false positives.

**Calibration treated as universal.** Pulse oximeters convert a ratio of red and infrared absorbances into oxygen saturation using an empirical calibration curve. The curve is derived from controlled desaturation studies in human volunteers, and it is specific to the sensor, the wavelengths, the geometry, and the population. A device that reuses another device's curve, or that extends it outside the studied range, is making an unvalidated measurement. The 2025 US Food and Drug Administration draft guidance on pulse oximeters makes this explicit, with accuracy and skin-tone performance criteria.

**Skin tone and perfusion bias.** The optical path depends on melanin, skin thickness, perfusion, and body composition. A device validated in a light-skinned, warm, well-perfused population can perform differently in a darker-skinned, cold, poorly perfused population. The bias is not a model error; it is a property of the physics and the calibration. Regulatory guidance now asks for stratified evidence, not a claim of fairness.

**Averaging that hides the signal.** A consumer device reports a heart rate every second, but the value may be a 10 s average, a 30 s average, or a median of the last five estimates. The averaging window determines what the number means. A 30 s average cannot show a 10 s arrhythmia, and a 1 s estimate is too noisy to be useful. The window is part of the measurement and is often undocumented.

This topic covers the PPG chain: optical sensing, sampling, filtering, peak detection, ratio-of-ratios oximetry, motion and quality assessment, averaging, skin-tone and perfusion bias, and validation. It builds on Topic 01 for sampling and filtering and on Topic 02 for the ECG reference standard. It sets up Topic 05 for respiration and continuous glucose, Topic 08 for digital biomarkers, and Topic 09 for validation.

### 1.1 Three composite failures and their costs

**The walking false alarm.** A user walks for 20 min and the device reports an irregular rhythm notification. The PPG has motion artifact, the peak detector finds extra peaks, and the RR series contains intervals of 0.4 s next to intervals of 0.9 s. The user visits a clinic, wears a 14-day patch, and no AF is found. The cost is anxiety, a clinic visit, and a loss of trust in the device. The technical cause is the absence of a quality gate that uses the accelerometer, not a failure of the classifier.

**The reassuring oximeter.** A patient with pneumonia and dark skin has an SpO2 reading of 92% on a consumer device. The true SaO2 is 86%. The device was calibrated in a lighter-skinned population and overestimates at low saturation. The cost is a delayed escalation and a worse outcome. The technical cause is a calibration that does not transfer, not a random error.

**The blood-pressure number.** A wellness app reports a blood-pressure estimate from PPG without a cuff calibration. The number is plausible and wrong. The user changes medication based on it. The cost is a clinical harm from a measurement that was never validated as a measurement. The technical cause is a claim that exceeds the evidence.

A fourth failure is a device that reports a value it cannot support. A wrist device with a 25 Hz sampling rate reports a pulse morphology feature that requires 100 Hz; the number is computed and meaningless. The failure is not in the display but in the feature definition, which was written for a different sampling rate.

The three failures share a structure: a derived quantity is presented as if it were a direct measurement, and the assumptions behind the derivation are invisible. The fix is to make the assumptions explicit, to validate them in the intended population, and to limit the claim to what the evidence supports.

### 1.2 Why PPG is the highest-volume signal in the world

PPG is in billions of devices: wristbands, rings, earbuds, patches, and fingertip oximeters. The volume creates a statistical problem that ECG does not have. A small bias in a population becomes a large number of wrong values when the device is used by millions of people. A 1% SpO2 bias at 90% affects a clinically meaningful fraction of users. A 5% heart-rate error at rest affects a larger fraction of readings.

The volume also creates a data problem. Most PPG data are collected outside clinical settings, without a reference, and without a documented chain. A model trained on that data learns the device's quirks, not the physiology. The validation must use a reference and a documented device, and the deployment must monitor for drift.

A second consequence of volume is that the device becomes a de facto screening tool, whether or not it was designed as one. A heart-rate feature that flags an irregular rhythm will be used by users to decide whether to seek care. The product team should design the confirmation pathway even if the claim is only wellness, because the users will build one anyway.

A third consequence of volume is that small per-user harms become public-health-scale effects. A false AF notification in 0.5% of users is 500,000 notifications in a population of 100 million users. The per-user cost is anxiety and a clinic visit; the aggregate cost is a strain on confirmatory services. The product team should estimate the aggregate before launch, not after.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** the optical path of a PPG sensor and why reflectance and transmittance geometries produce different signals.
2. **Compute** heart rate from PPG peaks, with a stated averaging window and a stated peak detector.
3. **Compute** the ratio-of-ratios R and oxygen saturation from red and infrared AC/DC amplitudes, and state the calibration assumption.
4. **Calculate** the perfusion index and explain why it is not a quality score on its own.
5. **Identify** motion artifact, ambient light, poor contact, and low perfusion from the waveform and the auxiliary signals.
6. **Design** a quality gate that prevents motion from becoming an AF or heart-rate label.
7. **Evaluate** a PPG validation study for reference standard, population, skin-tone stratification, and averaging.
8. **Critique** a consumer blood-pressure or AF claim for the evidence it would require.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

The outcomes are ordered from the optical physics to the claim. Outcome 1 is physical; outcomes 2 to 5 are computational and diagnostic; outcomes 6 to 8 are professional. The order matters because a claim about SpO2 that is not grounded in the optical path and the calibration is a claim without a mechanism.

## 3. The core mental model

PPG is an optical measurement with a pulsatile and a non-pulsatile component. The non-pulsatile component is the DC level, set by tissue, bone, venous blood, and melanin. The pulsatile component is the AC level, set by arterial blood volume changes with each heartbeat. The ratio of the two, and the ratio of ratios across wavelengths, is the basis of every derived measurement.

```text
        LED (red, infrared, green)
                 |
        photons enter skin and tissue
                 |
     +-----------+------------+
     |                        |
 absorbed by melanin,     absorbed by arterial
 venous blood, tissue     blood (pulsatile)
     |                        |
     +-----------+------------+
                 |
        photodetector current
                 |
        transimpedance amplifier
                 |
        DC + AC + noise + ambient
                 |
     +-----------+------------+
     |            |            |
  DC level     AC level     quality
  (perfusion,  (pulse,      (motion,
   contact)     HR)          contact)
     |            |            |
     +-----------+------------+
                 |
        features: HR, RR, PI, R
                 |
        measurement: SpO2, HR
                 |
        decision: notify, refer, report
```

Redraw it with the two wavelengths on the left and the derived quantities on the right. The key idea is that the same optical signal supports several measurements, and each measurement uses a different part of the signal and a different calibration.

### 3.0 What the waveform looks like

A single PPG pulse has a steep systolic upstroke, a peak, a slower decline, and often a secondary wave (the dicrotic notch) produced by wave reflection.

```text
 amplitude
    ^
    |        /\  systolic peak
    |       /  \
    |      /    \      /\  dicrotic wave
    |     /      \    /  \
    |    /        \__/    \______
    |___/__________________________\______ time
        upstroke   decline   notch   decay
        (~100 ms)  (~300 ms)         (~300 ms)
```

The timing of these features depends on heart rate, vascular stiffness, and age. A young, compliant artery has a late, prominent dicrotic wave; an older, stiff artery has an earlier, smaller wave. A PPG-based feature that uses the notch is measuring vascular properties as well as cardiac timing, which is why it is a candidate biomarker and why it is device- and population-sensitive.

At 25 samples/s, the upstroke (about 100 ms) is 2.5 samples. The peak is poorly resolved and the notch is lost. At 100 samples/s, the upstroke is 10 samples and the notch is visible. The sampling rate determines which morphology features are measurable, independent of the algorithm.

The same logic applies to the pulse width. A width of 300 ms at 25 Hz is 7.5 samples; the quantization uncertainty is one sample, or 40 ms, which is 13% of the width. At 100 Hz, the uncertainty is 10 ms, or 3% of the width. A feature that needs 5% precision requires at least 67 Hz sampling. The sampling rate is derived from the feature, not from the device default.

### 3.1 Reflectance and transmittance

A fingertip oximeter works in **transmittance** mode: the LED is on one side of the finger and the detector on the other. The light passes through a known path, and the pulsatile component is a small fraction of the transmitted light. A wrist device works in **reflectance** mode: the LED and detector are on the same side, and the detector measures light backscattered from the tissue. The path is less controlled, the signal is smaller, and the sensitivity to placement and motion is higher.

The geometry determines the calibration. A transmittance finger sensor and a reflectance wrist sensor do not share a calibration curve, even if they use the same wavelengths. A company that ports a finger calibration to a wrist device without a new study has not validated the wrist measurement.

### 3.2 The DC and AC components

The DC component is the average light level. It depends on the LED current, the detector gain, the skin pigmentation, the tissue thickness, and the contact pressure. The AC component is the pulsatile variation, typically 0.5% to 5% of the DC in a good recording.

The **perfusion index** is AC/DC × 100%. It is a measure of the pulsatile fraction, not of signal quality. A high perfusion index can come from strong pulsation or from motion; a low index can come from poor perfusion or from poor contact. The index is useful as a trend and misleading as an absolute quality score.

### 3.3 The ratio of ratios

Pulse oximetry uses two wavelengths, usually red (around 660 nm) and infrared (around 940 nm). The absorption of oxygenated and deoxygenated haemoglobin differs between them. The ratio of ratios is:

```text
R = (AC_red / DC_red) / (AC_ir / DC_ir)
```

R is then mapped to oxygen saturation (SpO2) through an empirical calibration curve. A common linear approximation is SpO2 = 110 - 25R, which gives 98% at R = 0.48. The approximation is not a physical law; it is a fit to a calibration study, and it is specific to the device.

The calibration curve is the most important undocumented assumption in consumer oximetry. A device that reports SpO2 without a calibration study is reporting a number derived from someone else's curve.

### 3.4 Motion as a signal and as an artifact

Motion affects PPG in two ways. It changes the optical coupling between the sensor and the skin, producing a multiplicative change in the DC level. It also changes the tissue geometry, producing a broadband additive artifact. The result is a waveform that can have larger amplitude than the pulse and a rhythm that can mimic an arrhythmia.

Wavelength choice is a design decision with physiological consequences. Green light (around 525 nm) is absorbed strongly by haemoglobin and penetrates shallowly, which makes it good for heart rate at the wrist but poor for deep vessels. Red and infrared light penetrate more deeply, which is why they are used for SpO2. A device that uses green for heart rate and red/infrared for SpO2 has two optical paths and two calibrations, even though it has one sensor package.

An accelerometer is the usual auxiliary signal for motion detection. A quality gate that uses accelerometer variance, perfusion index, and spectral shape can reject motion segments before the peak detector sees them. The gate is not optional for a wrist device; it is the difference between a usable measurement and a random number.

### 3.5 Measurement, feature, biomarker, decision

The distinction is especially important for PPG because the marketing language blurs it.

- **Measurement**: heart rate in beats per minute (bpm), SpO2 in percent, perfusion index in percent.
- **Feature**: RR interval, RR variability, pulse amplitude, pulse width, rise time.
- **Biomarker**: a feature with a validated relationship to a state or outcome, such as RR irregularity as a marker of AF, or pulse wave velocity as a marker of arterial stiffness.
- **Decision**: notify the user of an irregular rhythm, refer for a sleep study, report a trend.

A wrist device that reports "AF" is making a decision, not a measurement. The decision requires a validated algorithm, a quality gate, and a confirmation pathway. The measurement behind it is an irregularity feature that motion can imitate.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| PPG | Photoplethysmography; optical blood volume measurement | Calling it a heart-rate sensor | arbitrary units or volts |
| LED | Light-emitting diode | Ignoring wavelength | nm |
| Photodetector | Light-to-current transducer | Confusing with LED | A/W |
| Transmittance | Light passing through tissue | Assuming it applies to wrists | geometry |
| Reflectance | Light backscattered from tissue | Assuming same calibration as transmittance | geometry |
| DC component | Average light level | Treating it as signal | V or counts |
| AC component | Pulsatile light variation | Treating it as noise | V or counts |
| Perfusion index | AC/DC × 100% | Using it as a quality score | % |
| Ratio of ratios | (ACr/DCr)/(ACir/DCir) | Calling it SpO2 | dimensionless |
| SpO2 | Peripheral oxygen saturation | Calling it SaO2 | % |
| SaO2 | Arterial oxygen saturation from blood gas | Confusing with SpO2 | % |
| Calibration curve | Empirical map from R to SpO2 | Assuming it is universal | device-specific |
| ARMS | Accuracy root mean square difference | Reporting bias alone | % SpO2 |
| Bias | Mean difference between SpO2 and SaO2 | Reporting without precision | % |
| Motion artifact | Signal from movement | Treating it as pulse | varies |
| Ambient light | External light reaching the detector | Ignoring it | lux, V |
| Contact pressure | Force between sensor and skin | Assuming more is better | N |
| Pulse amplitude | Peak-to-trough AC amplitude | Confusing with perfusion index | V |
| Pulse width | Width of the systolic peak | Using it without definition | ms |
| Pulse rise time | Time from trough to peak | Confusing with pulse width | ms |
| Dicrotic notch | Secondary wave after systole | Missing at low sampling rates | ms |
| Heart rate | Beats per minute | Reporting without averaging window | bpm |
| RR interval | Interval between pulse peaks | Calling it an ECG RR | s |
| Irregularity | Variation in RR intervals | Calling it AF | dimensionless |
| AF | Atrial fibrillation | Diagnosing from PPG alone | ICD-10 I48 |
| Quality index | Composite signal usability measure | Using it as a diagnosis | dimensionless |
| Skin tone | Melanin-related light absorption | Treating as a single category | MST, ITA |
| Fitzpatrick scale | Skin phototype classification | Using it as a colorimetry measure | I-VI |
| Monk Skin Tone | 10-point skin tone scale | Confusing with Fitzpatrick | 1-10 |
| ITA | Individual Typology Angle | Treating as equivalent to MST | degrees |
| Melanin | Pigment absorbing light | Ignoring wavelength dependence | concentration |
| Perfusion | Blood flow to tissue | Confusing with pulse amplitude | mL/min/100 g |
| Vasoconstriction | Reduced blood flow | Causing low perfusion index | state |
| Non-wear | Device off body | Treating as zero pulse | minutes |
| Averaging window | Time span of reported value | Leaving it undocumented | s |
| Sampling rate | Samples per second | Assuming 100 Hz for all devices | Hz |
| Aliasing | Folding above Nyquist | Missing dicrotic notch | Hz |
| Reference standard | ECG or arterial blood gas | Treating device as reference | categorical |
| Bland-Altman | Agreement plot | Reporting correlation instead | method |
| SpO2 range | Declared accuracy range | Extrapolating below 70% | % |
| Pulse wave | Pressure wave corresponding to the optical pulse | Treating the PPG as a pressure wave | arbitrary |
| Pulse rate | Rate from pulse peaks | Calling it heart rate without qualification | bpm |
| Pulse amplitude variability | Beat-to-beat change in AC | Confusing with perfusion index | % |
| Signal-to-noise ratio | Pulse power divided by noise power | Reporting without a band | dB |
| Wear detection | On-body status | Treating off-body as zero | boolean |
| Optical path length | Distance light travels in tissue | Assuming it is constant | mm |
| Scattering | Light redirection in tissue | Ignoring it in calibration | physics |
| Absorption coefficient | Light absorption per unit path | Treating as wavelength-independent | cm^-1 |
| Isobestic point | Wavelength with equal absorption | Using it for SpO2 | nm |
| Reflectance geometry | LED and detector on same side | Assuming transmittance calibration | geometry |
| Transmittance geometry | LED and detector opposite | Assuming reflectance calibration | geometry |
| Motion reference | Accelerometer used to model artifact | Using PPG alone | signal |
| Adaptive filter | Filter that adjusts to a reference | Assuming it removes all motion | method |
| Spectral subtraction | Removing a modelled artifact spectrum | Assuming stationarity | method |
| Template subtraction | Removing an average beat template | Assuming a stable morphology | method |
| Pulse oximetry | Optical SpO2 measurement | Confusing with blood gas | method |
| Co-oximetry | Multi-wavelength blood gas measurement | Treating as a pulse oximeter | method |
| Calibration study | Controlled desaturation study | Using a convenience sample | study design |
| Non-disparate performance | Comparable accuracy across skin tones | Claiming without stratified data | regulatory |
| Inconclusive rate | Share of readings without a result | Treating as negative | % |
| Positive predictive value | Probability of disease given a positive | Confusing with sensitivity | % |
| Prevalence | Share of population with the condition | Ignoring in PPV | % |

## 5. The mechanics, step by step

The procedure is written for a wrist or finger PPG sampled at 25 Hz to 100 Hz, with an accelerometer and a device-reported contact status. It scales to a research-grade device with raw optical channels by exposing the DC and AC components directly.

### Step 1. Define the measurement and the claim

**Input:** the product question, the intended population, and the setting.

**Output:** a measurement specification with the quantity, the unit, the averaging window, the accuracy target, and the claim level.

**Rationale:** the same optical signal supports heart rate, SpO2, AF notification, sleep staging, and blood-pressure research. Each has a different accuracy requirement and a different reference standard. A specification that says "heart rate" without a window or an accuracy target cannot be validated.

**Branch point:** if the claim is heart rate for wellness, the target may be a mean absolute error of a few bpm against ECG. If the claim is SpO2 for a medical device, the target is an ARMS threshold and a controlled desaturation study. If the claim is AF notification, the target is sensitivity and specificity against a confirmatory ECG, with a prespecified inconclusive rate.

**Edge case:** a claim that combines measurements, such as "heart rate and SpO2," needs separate validation for each. A device that is accurate for heart rate can be inaccurate for SpO2; the optical requirements are different.

**A measurement specification table.** The following fields are the minimum for a PPG claim.

| Field | Example | Why it matters |
|---|---|---|
| Quantity | Heart rate | Defines the feature and the reference |
| Unit | bpm | Prevents unit errors |
| Window | 30 s rolling | Sets latency and what can be shown |
| Population | Adults at rest | Sets the validation population |
| Condition | Rest, indoors | Sets the artifact environment |
| Reference | ECG | Sets the agreement standard |
| Accuracy target | MAE ≤ 5 bpm | Sets the pass criterion |
| Coverage | ≥ 95% of wear time | Sets the missingness expectation |

A specification without the condition and the coverage is incomplete. A device that is accurate only while the user is still, and reports nothing while the user moves, has a coverage problem that the accuracy number hides.

### Step 2. Verify sensor placement and optical coupling

**Input:** the device, the wear location, and the user instructions.

**Output:** a placement check and a contact-quality flag.

**Rationale:** the optical path depends on the sensor being in stable contact with well-perfused skin. A loose strap, a hairy wrist, a cold hand, or a sensor over bone reduces the pulsatile signal and increases motion sensitivity. The placement is part of the measurement, not an operational detail.

**Arithmetic:** the AC component in a good wrist recording is about 1% to 3% of the DC. A 1% AC on a 1 V DC is 10 mV, which is easily digitised. A 0.1% AC is 1 mV, which is at the noise floor of a low-power front end. The perfusion index is the ratio that tells you which regime you are in.

**Branch point:** if the contact is poor, the device may still report a value from noise. A contact flag that uses DC level, perfusion index, and accelerometer is more reliable than a single threshold.

**Edge case:** contact pressure can be too high. Excessive pressure compresses the vessels and reduces pulsation, producing a low perfusion index despite good contact. The optimum is light, stable contact, not maximum force.

**Placement and signal amplitude.** The optical signal depends on the tissue under the sensor.

| Site | Geometry | Typical AC/DC | Advantages | Limitations |
|---|---|---:|---|---|
| Fingertip | Transmittance | 1-5% | Strong signal, good calibration | Bulky, motion-sensitive |
| Wrist | Reflectance | 0.5-3% | Comfortable, continuous | Motion, poor contact |
| Ear | Reflectance/transmittance | 1-3% | Stable, central | Comfort, positioning |
| Forehead | Reflectance | 0.5-2% | Good for sleep | Visible, motion |
| Ring | Reflectance | 0.5-2% | Stable fit | Perfusion, size |

A fingertip transmittance signal is the easiest to measure and the easiest to calibrate. A wrist reflectance signal is the hardest to measure and the hardest to calibrate, which is why wrist SpO2 is a recent and narrow claim.

### Step 3. Check the optical drive and ambient light rejection

**Input:** the LED current, the detector gain, and the ambient-light reading.

**Output:** a documented optical configuration and an ambient-light flag.

**Rationale:** the LED current and detector gain set the DC level and the signal-to-noise ratio. Ambient light adds a DC offset and low-frequency noise. Most devices modulate the LED and use synchronous detection to reject ambient light; a device that does not will drift with room lighting.

**Arithmetic:** a fluorescent lamp modulated at 100 Hz or 120 Hz can appear in the PPG if the sampling rate is above 200 Hz and the demodulation is imperfect. At 25 Hz sampling, the same lamp aliases to 0 Hz or 5 Hz, which can look like a baseline wander. The sampling rate and the light source interact.

**Branch point:** if the ambient flag is high, the measurement is unreliable even if the waveform looks plausible. Reject or repeat.

**Edge case:** sunlight contains a broad spectrum that overlaps both the red and infrared bands. A device used outdoors can have a DC shift that saturates the detector. The saturation is not visible in the AC component if the amplifier is AC-coupled, but the DC information needed for the ratio is lost.

**LED duty cycle and power.** A PPG front end typically pulses the LED at a low duty cycle to save power and to allow synchronous detection. The duty cycle determines the peak current and the average power. A 1% duty cycle at 100 mA peak draws 1 mA average, which is manageable for a watch battery. The pulsing also creates a sampling constraint: the detector must integrate over the pulse, and the effective sampling rate is the pulse rate, not the ADC rate.

A device that increases the LED current to improve a weak signal also increases the power and the heating. Heating changes the skin perfusion and the LED wavelength, which changes the calibration. The optical design is a thermal design as well.

### Step 4. Filter and sample

**Input:** the raw optical signal and the sampling rate.

**Output:** a filtered signal for peak detection and a separate DC estimate for the ratio.

**Rationale:** peak detection needs the pulsatile band, roughly 0.5 Hz to 5 Hz for heart rates from 30 bpm to 300 bpm. The ratio needs the DC level, which a high-pass filter removes. The two paths must be separate.

**Arithmetic:** a 0.5 Hz high-pass removes baseline wander but attenuates the fundamental at 30 bpm (0.5 Hz). A 0.5 Hz cutoff is acceptable for normal rates but marginal for bradycardia. A 0.3 Hz cutoff preserves the fundamental at the cost of more wander.

At 25 samples/s, the Nyquist frequency is 12.5 Hz. The dicrotic notch contains content above 10 Hz and is attenuated. At 100 samples/s, the notch is preserved. A pulse-morphology feature computed at 25 Hz is a different feature from the same feature at 100 Hz.

**Branch point:** if the device reports only a filtered waveform, the DC is not available and the ratio cannot be recomputed. The raw optical channels are needed for a reproducible SpO2.

**Edge case:** an AC-coupled front end removes the DC by design. The device may report a "DC" estimate from a slow feedback loop, which is not the same as the raw DC. The distinction matters when the ambient light or contact changes.

**Filter design for pulse detection.** The pulse band is roughly 0.5 Hz to 5 Hz for rates from 30 bpm to 300 bpm. A band-pass from 0.5 Hz to 8 Hz preserves the fundamental and the first harmonics, including the dicrotic notch, while removing baseline wander and high-frequency noise.

The high-pass cutoff is a trade-off. A 0.5 Hz cutoff attenuates a 30 bpm fundamental (0.5 Hz) by 3 dB and a 40 bpm fundamental (0.67 Hz) by about 1.5 dB. A 0.3 Hz cutoff preserves bradycardia but admits more respiratory wander. A 1 Hz cutoff removes wander but distorts the pulse shape and the perfusion estimate.

The low-pass cutoff is also a trade-off. An 8 Hz cutoff preserves the notch at normal rates but removes the sharper features of the systolic upstroke. A 15 Hz cutoff preserves more morphology but admits more noise. The cutoff must be stated with the feature it supports.

**Zero-phase filtering for morphology.** If the feature is pulse width or rise time, a causal filter shifts the fiducial and biases the measurement. A forward-backward filter removes the phase distortion at the cost of non-causality. For real-time heart rate, a causal filter is acceptable because the rate does not depend on the absolute fiducial.

### Step 5. Compute the perfusion index and a quality gate

**Input:** the filtered signal, the DC estimate, and the accelerometer.

**Output:** a perfusion index, a motion score, and an accept/flag/reject decision.

**Rationale:** the quality gate is what prevents motion from becoming a label. It should combine at least three signals: the optical waveform, the accelerometer, and the contact status.

**Arithmetic:** the perfusion index is AC/DC × 100%. A value below 0.3% is usually poor perfusion or poor contact. A value above 10% is often motion or venous pulsation. The thresholds are device-specific and must be validated.

**Branch point:** if the accelerometer variance is high, reject regardless of the optical appearance. If the accelerometer is quiet but the optical signal is irregular, consider an arrhythmia or a sensor problem.

**Edge case:** a resting user can have a low perfusion index from cold hands. Rejecting all low-perfusion segments biases the dataset toward warm, well-perfused users, which is a selection bias, not a quality improvement. Report the rejection rate by subgroup.

**A quality gate that uses three signals.** The gate should be simple enough to audit and strong enough to stop motion. A defensible design:

1. If the accelerometer magnitude variance in a 1 s window exceeds a device-specific threshold, reject.
2. If the perfusion index is below 0.3% for more than 2 s, flag and do not report SpO2.
3. If the RR interval differs from the median by more than 30% and the accelerometer is quiet, flag for arrhythmia review rather than rejecting.
4. If the DC is within 2% of the rail, reject as saturation.
5. Otherwise, accept and compute features.

The order matters. Accelerometer first prevents motion from being interpreted as arrhythmia. Perfusion second prevents low-signal ratios from being reported. The arrhythmia branch is last because it is a physiological finding, not a quality failure.

The thresholds are starting points. Each must be validated against adjudicated segments on the target device, and the rejection rate must be reported by subgroup.

### Step 6. Detect pulse peaks and compute heart rate

**Input:** the filtered PPG and the quality mask.

**Output:** peak times, RR intervals, and a heart rate with a stated averaging window.

**Rationale:** the pulse peak is not the R peak. It occurs after the ECG R wave by a delay that depends on pulse transit time, which varies with blood pressure, age, and vascular stiffness. The PPG heart rate is therefore an estimate of the ECG rate with a timing offset, and the offset does not affect the rate if it is constant.

**Arithmetic:** at 72 bpm, the RR interval is 0.833 s. A 30 s window contains about 36 beats and 35 intervals. A 10 s window contains about 12 beats and 11 intervals. The count resolution of the 10 s window is 6 bpm per beat, as in Topic 01.

**Branch point:** if the peak detector finds extra peaks, the rate rises; if it misses peaks, the rate falls. A median filter on the RR series removes isolated errors but also removes real short intervals from ectopy. The choice depends on the decision.

**Edge case:** a dicrotic notch can be detected as a second peak if the threshold is low. A refractory period of 0.3 s to 0.4 s prevents double detection at normal rates but is too long for rates above 150 bpm.

**Peak detection methods.** Three are common for PPG.

- **Amplitude threshold with refractory period** is simple and works for clean signals. It fails when the amplitude varies with respiration or contact.
- **Adaptive threshold** tracks the signal envelope and adjusts the threshold. It handles amplitude drift but can lock onto motion.
- **Derivative-based detection** finds the steepest upstroke. It is less sensitive to baseline shift but more sensitive to noise.

A detector should report the beats it found and the beats it rejected. A detector that silently returns a rate without a beat count cannot be audited.

**Systolic peak versus maximum slope.** The systolic peak is the maximum of the pulse wave. The maximum upslope occurs earlier and is less affected by the dicrotic wave. For heart rate, either fiducial works if it is consistent. For pulse transit time, the maximum upslope is often used because it is closer to the pressure wave arrival. The choice must be stated because it changes the absolute timing by tens of milliseconds.

### Step 7. Compute the ratio of ratios and SpO2

**Input:** the red and infrared DC and AC amplitudes.

**Output:** R and SpO2 with the calibration curve and the declared range.

**Rationale:** the ratio cancels the common-mode tissue absorption and isolates the haemoglobin difference. The calibration curve maps R to SpO2. The curve is empirical and device-specific.

**Arithmetic:** with red DC 1.0 V, red AC 0.02 V, infrared DC 1.2 V, and infrared AC 0.05 V:

- Red ratio = 0.02/1.0 = 0.020.
- Infrared ratio = 0.05/1.2 = 0.04167.
- R = 0.020/0.04167 = 0.480.
- SpO2 = 110 - 25 × 0.480 = 98.0%.

A change of 0.04 in R changes SpO2 by 1 percentage point on this curve. The sensitivity of the measurement to the ratio is therefore about 25 percentage points per unit R.

**Branch point:** if the AC amplitude is at the noise floor, R is unstable. A minimum AC threshold is needed before reporting SpO2.

**Edge case:** the calibration curve is only valid over the studied range, usually 70% to 100%. Below 70%, the curve is extrapolated and the accuracy is unknown. A device that reports 60% without a study is reporting an unvalidated number.

**Where the calibration comes from.** A pulse oximeter calibration is derived from a controlled desaturation study. Healthy volunteers breathe a controlled gas mixture, their SaO2 is measured by a co-oximeter at stable plateaus, and the device's R is recorded at each plateau. The curve is fitted across the range.

The study has constraints. The range is limited by safety, usually 70% to 100%. The volunteers are healthy, so the curve may not transfer to patients with anaemia, poor perfusion, or abnormal haemoglobin. The number of participants and the number of paired samples determine the confidence in the curve.

A device that uses a generic curve from the literature is using a curve from a different sensor geometry, different wavelengths, and a different population. The error may be small in the normal range and large at the extremes. The FDA draft guidance's emphasis on device-specific studies is a response to this.

**Functional versus fractional saturation.** Pulse oximeters report functional saturation: the fraction of haemoglobin that is oxygenated, ignoring carboxyhaemoglobin and methaemoglobin. A co-oximeter reports fractional saturation, which includes all haemoglobin species. In a patient with carbon monoxide poisoning, the two differ substantially. A device that reports functional SpO2 can be normal while the patient is hypoxic. This is a physiological limitation, not a signal-processing one.

### Step 8. Detect motion and correct or reject

**Input:** the optical signal, the accelerometer, and the quality mask.

**Output:** a motion-corrected signal or a rejected segment.

**Rationale:** motion is the dominant artifact in wrist PPG. Correction methods include adaptive filtering with the accelerometer as a reference, spectral subtraction, and template subtraction. None is perfect, and rejection is often safer than correction.

**Arithmetic:** a motion artifact at 1.5 Hz with an amplitude of 1.2 V is larger than the pulse (typically 0.01 V). The artifact dominates the peak detector. A quality gate that uses accelerometer variance rejects the segment before detection.

**Branch point:** if the motion is periodic, such as walking, the artifact can be modelled and removed. If it is random, rejection is more reliable. A device that claims heart rate during exercise needs a validated motion-correction method, not a general quality gate.

**Edge case:** motion can be the signal of interest. In activity recognition, the accelerometer is the measurement, and the PPG is the artifact. The same data stream supports both, but the preprocessing is different.

**Motion correction methods and their assumptions.** Three families are common.

- **Adaptive filtering** uses the accelerometer as a reference and subtracts the component of the PPG correlated with it. It assumes a linear, stable relationship between the motion and the optical artifact. It fails when the relationship changes with contact or posture.
- **Spectral subtraction** estimates the artifact spectrum from a motion segment and subtracts it. It assumes the artifact is stationary over the segment. It fails when the motion is transient.
- **Template subtraction** models the average motion artifact and subtracts it. It assumes the artifact is repeatable. It fails for random motion.

None of the methods recovers the pulse when the artifact is larger than the pulse and broadband. In that case, rejection is the only honest option. A product that reports a value during vigorous motion without a validation is reporting an extrapolation.

### Step 9. Average and report

**Input:** the beat-level estimates and the quality mask.

**Output:** a reported value with a window, a dispersion measure, and a coverage statement.

**Rationale:** a beat-level heart rate is noisy. A reported value is usually an average or median over a window. The window determines what the value can show. A 30 s average cannot show a 10 s arrhythmia, and a 1 s value is too noisy for a trend.

**Arithmetic:** the standard error of the mean RR decreases with the square root of the number of intervals. A 30 s window has 35 intervals, so the standard error is about 1/√35 = 17% of the interval standard deviation. A 10 s window has 11 intervals, so the standard error is about 30%.

**Branch point:** if the decision is a trend, use a consistent window and report the window. If the decision is an event, use a shorter window and accept more noise.

**Edge case:** a device that reports a value every second from a 30 s rolling window produces a smooth curve with 30 s of latency. The latency is invisible in the display but matters for an alarm.

**Reporting a dispersion measure.** A single number without a dispersion is a point estimate without an uncertainty. Report the median and the interquartile range of the beat-level estimates, or the standard error of the mean. A user who sees "72 bpm" does not know whether the beat-level values were 70 to 74 or 60 to 85.

The dispersion also reveals artifact. A heart rate of 72 bpm with a beat-level range of 60 to 85 is not a stable 72; it is an average of a noisy signal. A quality gate should reject the segment or the report should flag the dispersion.

**Rounding.** A heart rate reported to the nearest integer implies an accuracy of 1 bpm. A device with a 5 bpm mean absolute error should not report to the nearest integer without a caveat; the precision of the display exceeds the precision of the measurement. Report the value and the accuracy together, or round to a level that matches the accuracy.

### Step 10. Assess skin tone and perfusion bias

**Input:** the validation dataset and the device's optical configuration.

**Output:** performance stratified by skin tone, perfusion, and body composition.

**Rationale:** melanin absorbs light in both the red and infrared bands, but not equally. The ratio of ratios partially cancels the effect, but not completely. The residual bias can be clinically meaningful, especially at low saturation.

**Arithmetic:** a 1% bias in SpO2 at 90% is clinically important because the treatment threshold is often 90% or 88%. A device that reads 91% when the true value is 90% can delay oxygen therapy. The bias is not random error; it is a systematic offset that varies by subgroup.

**Branch point:** if the validation dataset is not stratified, the performance in darker skin tones is unknown. The correct response is to collect the data, not to claim fairness from a pooled result.

**Edge case:** skin tone is not the only moderator. Perfusion, altitude, anaemia, and sensor placement also affect the measurement. A stratified analysis by skin tone alone can miss a perfusion interaction.

**Why melanin matters at two wavelengths.** Melanin absorption decreases with wavelength, so it affects red and infrared differently. The ratio of ratios assumes that the melanin contribution is common to both channels and cancels. It cancels only partially because the path lengths and the absorption coefficients differ. The residual is a bias that depends on the melanin concentration and on the sensor geometry.

The bias is largest at low saturation because the AC signal is smaller and the melanin absorption is a larger fraction of the total. The FDA draft guidance sets the between-group bias limit at 3.5% for the 70% to 85% range and 1.5% for the 85% to 100% range. The lower range allows a larger absolute bias, which reflects the greater difficulty of the measurement and the greater variability of the reference at low saturation. Inference: the limits are set to be achievable while still detecting a clinically meaningful disparity.

**Reporting bias by subgroup.** A table with one row per subgroup and columns for n, bias, precision, and ARMS is the minimum. A pooled row can be added for context, but it should not replace the subgroups. A claim of non-disparate performance requires the subgroup differences to be within the prespecified limits, not the pooled value to be small.

### Step 11. Validate against the reference

**Input:** the device output and the reference measurement.

**Output:** bias, precision, ARMS, Bland-Altman limits, and subgroup performance.

**Rationale:** the reference for heart rate is ECG; for SpO2, it is arterial blood gas SaO2 from a co-oximeter; for AF, it is a confirmatory ECG with adjudication. The reference has its own error and its own sampling.

**Arithmetic:** ARMS is the square root of the mean squared difference between SpO2 and SaO2. A device with a bias of 1% and a standard deviation of 2% has an ARMS of √(1² + 2²) = 2.24%. A device with a bias of 0% and a standard deviation of 3% has an ARMS of 3.0%. Both can meet a 4% threshold; only the second is unbiased.

**Branch point:** if the study is a controlled desaturation, the range and the number of paired samples are prespecified. If it is a convenience sample, the range is whatever the patients happened to have, and the accuracy at low saturation may be unmeasured.

**Edge case:** the reference blood gas is a point measurement, while the device is a continuous estimate. The pairing must be time-aligned, and the alignment uncertainty contributes to the difference.

**How many paired samples.** The FDA draft guidance recommends at least 3,000 paired observations across at least 150 participants. The number is not arbitrary: it provides enough observations in each saturation band and each skin-tone group to estimate the bias with a confidence interval narrow enough to test the 3% criterion.

A study with 300 pairs can estimate a pooled ARMS but cannot estimate subgroup bias with useful precision. The subgroup analysis is the reason for the large sample. A manufacturer that plans a small study should plan a pooled claim, not a non-disparate claim.

### Step 12. Report the claim with its limits

**Input:** the validation results and the intended use.

**Output:** a claim statement with the measurement, range, population, conditions, and limits.

**Rationale:** a claim that omits the population or the conditions is not reproducible and is likely to be misread. "SpO2 accuracy 2.5% ARMS" without the range, the population, and the motion condition is incomplete.

**Branch point:** if the device is for consumer wellness, the claim should avoid diagnostic language. If it is for clinical use, the claim should state the regulatory clearance and the validated population.

**Edge case:** a claim that is true at rest can be false during motion. Report both conditions, or limit the claim to rest.

**A claim checklist.** Before a claim is written, five questions should have answers.

1. What is the quantity, and is it a measurement or a decision?
2. What is the reference, and what is its error?
3. What population and condition were studied?
4. What is the accuracy, and what is the coverage?
5. What does the claim explicitly not cover?

A claim that answers all five is auditable. A claim that answers only the fourth is a number without a context, and it will be misread.

### A reference parameter set

| Stage | Parameter | Value | Justification |
|---|---|---|---|
| Optical | Wavelengths | Green for HR; red/IR for SpO2 | Penetration and haemoglobin contrast |
| Sampling | Rate | 100 Hz | 10 ms timing; notch preserved |
| Filter | Band-pass | 0.5-8 Hz, 2nd order, zero-phase | Pulse band; morphology preserved |
| Detection | Refractory | 0.4 s | Prevents notch double-count |
| Quality | Accelerometer | 1 s variance threshold | Motion gate |
| Quality | Perfusion index | 0.3% minimum | AC above noise floor |
| Ratio | DC estimate | 2 s moving average | Stable denominator |
| Ratio | AC estimate | Beat peak-to-trough | Consistent with calibration |
| Average | HR window | 30 s | Resting trend |
| Average | SpO2 window | 8 s | Responsive but stable |
| Calibration | Curve | Device-specific | Accuracy claim |

The table is a starting point for a resting wrist device. An exercise device needs a different detector, a different motion model, and a different validation. The table is a chain, not a recipe.

A chain is only reproducible if the parameters are recorded with the data. A device that stores the raw optical channels and the accelerometer but not the filter settings leaves the analyst unable to reproduce the reported heart rate. The metadata contract from Topic 01 applies here with the optical parameters added: wavelengths, LED current, detector gain, filter cutoffs, sampling rate, and calibration version.

### Decision table for PPG quality

| Condition | Detect by | Effect | Action |
|---|---|---|---|
| Motion | Accelerometer variance, broadband power | False peaks, irregular RR | Reject segment; do not label |
| Poor contact | Low DC, low perfusion index | Low SNR, unstable ratio | Reject; prompt reposition |
| Excessive pressure | Low perfusion index despite good contact | Reduced pulsation | Reject; prompt looser fit |
| Ambient light | High DC offset, low-frequency drift | Baseline wander, saturation | Reject; shield sensor |
| Low perfusion | Perfusion index below threshold | Unstable AC, noisy ratio | Flag; do not report SpO2 |
| Venous pulsation | High perfusion index, abnormal waveform | Overestimated AC | Flag; check position |
| Arrhythmia | Irregular RR with good quality | Real irregularity | Report; confirm with ECG |
| Non-wear | Accelerometer stillness, no pulse | No measurement | Mark missing |
| Saturation | DC at rail | Ratio invalid | Reject |
| Skin-tone bias | Subgroup difference in validation | Systematic offset | Report; recalibrate |

## 6. Worked example

### 6.1 The case

A wrist PPG records 30 s at 100 samples/s. The subject is at rest with a heart rate of about 72 bpm. The red channel has a DC of 1.0 V and an AC of 0.02 V. The infrared channel has a DC of 1.2 V and an AC of 0.05 V. The analysis goal is heart rate, perfusion index, and SpO2. A motion episode is then added between 10 s and 15 s to test the quality gate.

### 6.2 Acquisition arithmetic

- Samples: 100 samples/s × 30 s = 3,000.
- Nyquist: 50 Hz. The dicrotic notch content above 10 Hz is preserved; the 50 Hz mains is at Nyquist and is not usable.
- Beat count at 72 bpm: 72/60 × 30 = 36 beats, or 35 intervals.
- Peak timing resolution at 100 Hz: 10 ms. At 25 Hz: 40 ms.
- Heart-rate error from peak quantization at 25 Hz: an RR of 0.833 s becomes 0.833 ± 0.04 s, which is 72 ± 3.3 bpm. The quantization alone can produce a 3.3 bpm error.

The quantization arithmetic is worth stating exactly. If the true RR is T and the measured RR is T + δ, where δ is the timing error, the measured rate is 60/(T + δ). For T = 0.8333 s and δ = 0.04 s, the measured rate is 68.7 bpm, an error of -3.3 bpm. For δ = -0.04 s, the measured rate is 75.9 bpm, an error of +3.9 bpm. The error is asymmetric because the rate is a reciprocal.

| Sampling rate | Timing quantum | Rate at T - δ | Rate at T + δ | Error range |
|---:|---:|---:|---:|---:|
| 25 Hz | 40 ms | 75.9 bpm | 68.7 bpm | -3.3 to +3.9 |
| 50 Hz | 20 ms | 73.8 bpm | 70.3 bpm | -1.7 to +1.8 |
| 100 Hz | 10 ms | 72.9 bpm | 71.2 bpm | -0.8 to +0.9 |

A 100 Hz device has a quantization error below 1 bpm, which is adequate for a resting heart rate. A 25 Hz device has an error of up to 3.9 bpm, which is larger than the accuracy target of many products. The sampling rate is a design decision with a direct accuracy consequence.

### 6.3 Perfusion index arithmetic

- Red perfusion index = 0.02/1.0 × 100 = 2.00%.
- Infrared perfusion index = 0.05/1.2 × 100 = 4.17%.

The infrared index is higher because the infrared AC is larger relative to its DC. The two indices are not expected to be equal; the wavelengths have different absorption and different path lengths. A device that reports a single perfusion index should state which wavelength it uses.

A perfusion index of 2% is in the normal range for a warm, resting wrist. A value below 0.3% suggests poor perfusion or poor contact. A value above 10% suggests motion, venous pulsation, or a sensor problem. The index is a screening signal, not a diagnosis, and the thresholds are device-specific.

### 6.4 Ratio and SpO2 arithmetic

- Red ratio = 0.020.
- Infrared ratio = 0.04167.
- R = 0.020/0.04167 = 0.480.
- SpO2 = 110 - 25 × 0.480 = 98.0%.

The calibration curve is linear with a slope of -25% per unit R. A 0.04 change in R changes SpO2 by 1 percentage point. A 0.004 change, which is within the noise of a low-perfusion recording, changes SpO2 by 0.1 percentage point. At low perfusion, the ratio noise is larger and the SpO2 noise is correspondingly larger.

The sensitivity of SpO2 to each input is a useful check.

| Input changed by 1% | R change | SpO2 change |
|---|---:|---:|
| Red AC | +0.0048 | -0.12% |
| Red DC | -0.0048 | +0.12% |
| Infrared AC | -0.0048 | +0.12% |
| Infrared DC | +0.0048 | -0.12% |

A 1% error in any of the four amplitudes produces about a 0.12 percentage point change in SpO2 on this curve. A 10% error produces about 1.2 percentage points. The measurement is more sensitive to the AC amplitudes than to the DC amplitudes in relative terms, but the AC amplitudes are the ones most affected by perfusion and motion.

The table also shows why the ratio cancels common-mode errors: a simultaneous 1% increase in all four amplitudes leaves R unchanged. The cancellation is the reason the ratio works, and the reason it fails when the error is not common-mode.

### 6.5 Heart-rate arithmetic

The detector finds 35 peaks in the clean record. The mean RR is 0.8332 s, so the heart rate is 60/0.8332 = 72.01 bpm. The true rate is 72.0 bpm. The 0.01 bpm difference is negligible.

The 30 s window contains 35 intervals, and the standard error of the mean RR is about 17% of the interval standard deviation. If the interval standard deviation is 20 ms, the standard error is 3.4 ms, and the heart-rate standard error is about 0.29 bpm. The estimate is precise because the window is long.

A 10 s window from the same record gives 72.10 bpm, a difference of 0.09 bpm. The difference is small because the synthetic rate is constant. In a real record with respiratory sinus arrhythmia, the 10 s and 30 s estimates can differ by several bpm.

The window comparison is a table of what each window can support.

| Window | Beats at 72 bpm | Standard error factor | Latency | Suitable for |
|---:|---:|---:|---:|---|
| 1 s | 1-2 | 1.0 | low | instantaneous display |
| 10 s | 12 | 1.8 | 10 s | spot check |
| 30 s | 36 | 3.0 | 30 s | resting trend |
| 5 min | 360 | 10.0 | 5 min | clinical average |

The standard error factor is relative to a single interval. A 30 s window is about 3 times more precise than a 10 s window in standard-error terms, but it is 3 times slower to respond. The product must choose the trade-off, and the choice must be documented.

A useful check is to compute the rate at two windows and compare. If the 10 s and 30 s rates differ by more than the expected standard error, the rhythm is changing or the signal is noisy. If they agree, the rate is stable and the window choice is not affecting the conclusion. The check is cheap and catches both artifact and genuine non-stationarity.

The window comparison also reveals the difference between precision and accuracy. The 8 s and 30 s estimates agree to 0.09 bpm in this record, which means the estimate is precise. They would disagree if the rate were changing; agreement does not prove accuracy, only stability.

The accuracy comes from the reference and the calibration. A precise estimate of the wrong quantity is still wrong. The heart rate is precise to 0.1 bpm and the SpO2 is accurate only if the calibration curve is the device's own. The precision is visible in the arithmetic; the accuracy is not.

A third check is the beat count. If the number of peaks in a window is not approximately the rate times the window length, the detector is failing. At 72 bpm over 30 s, the expected count is 36 beats; a count of 35 is consistent, a count of 45 is not. The count is the simplest audit of the detector and the first thing to print.

### 6.6 Diagnostic 1: motion inflates the rate

The motion episode adds a 1.2 V, 1.5 Hz oscillation between 10 s and 15 s. The peak detector finds 37 peaks instead of 35, and the mean RR falls to 0.7869 s. The reported heart rate rises to 76.24 bpm, an error of +4.23 bpm (5.9%).

The RR series contains two intervals below 0.6 s, with a minimum of 0.480 s, which corresponds to 125 bpm. No resting adult has a 0.48 s interval during a 1.5 Hz wrist oscillation; the interval is the artifact period, not a beat.

The diagnostic is the interval histogram and the accelerometer. A cluster of intervals near 0.48 s, coincident with high accelerometer variance, is motion. A quality gate that rejects the 10 s to 15 s segment restores the rate to 72.0 bpm.

The consequence of missing the diagnostic is an AF false positive or a spurious tachycardia. Both are clinically meaningful and both are preventable with a gate.

The diagnostic comparison is summarised below.

| Check | Clean record | Motion record | Interpretation |
|---|---|---|---|
| Peak count | 35 | 37 | Two extra peaks |
| Mean RR (s) | 0.833 | 0.787 | Shortened by artifact |
| Minimum RR (s) | 0.80 | 0.48 | Artifact period |
| Intervals below 0.6 s | 0 | 2 | Physiologically implausible |
| Accelerometer variance | low | high | Confirms motion |

The table is the kind of evidence a reviewer should see before accepting a rate. A single number without the beat count and the interval distribution is not auditable.

### 6.7 Diagnostic 2: a small ratio error becomes a SpO2 error

Suppose the red AC is measured as 0.022 V instead of 0.020 V because of baseline drift in the AC estimate. The red ratio becomes 0.022, R becomes 0.528, and SpO2 becomes 110 - 25 × 0.528 = 96.8%, a drop of 1.2 percentage points from a 10% error in the AC estimate.

The diagnostic is to compute R from multiple beats and check its stability. A ratio that varies by more than 0.02 across beats is unstable, and the SpO2 should be reported with a wider uncertainty or not at all. A single-beat ratio is not a measurement; it is a sample from a noisy distribution.

The second diagnostic is the perfusion index. At a perfusion index of 2%, the AC estimate is 2% of the DC, and a 10% relative error in the AC is 0.2% of the DC. At a perfusion index of 0.3%, the same relative error is 0.03% of the DC, which is below the noise floor. The ratio becomes unreliable exactly when the perfusion is low.

### 6.8 Varying one assumption

Suppose the device samples at 25 Hz instead of 100 Hz. The peak timing resolution is 40 ms instead of 10 ms. The heart-rate error from quantization alone is up to 3.3 bpm at 72 bpm, which is larger than the 0.29 bpm standard error of the 30 s estimate. The sampling rate, not the algorithm, sets the accuracy floor.

Suppose the averaging window is 8 s instead of 30 s. The estimate uses 9 intervals instead of 35, and the standard error rises by √(35/9) = 1.97. The 8 s estimate is noisier but more responsive. For a resting trend, 30 s is better; for an exercise recovery, 8 s may be better.

Suppose the calibration slope is -20 instead of -25. At R = 0.48, SpO2 becomes 110 - 20 × 0.48 = 100.4%, which is above the physiological maximum and should be clamped to 100%. The slope is device-specific, and a slope borrowed from another device can produce impossible values at the extremes.

### 6.9 Conclusion of the case

The clean record gives a heart rate of 72.01 bpm, a red perfusion index of 2.00%, an infrared perfusion index of 4.17%, and an SpO2 of 98.0% under the stated calibration. The motion episode inflates the rate to 76.24 bpm and creates two physiologically impossible intervals; the quality gate is what prevents the inflation from reaching the report.

The two diagnostics (interval histogram plus accelerometer, ratio stability plus perfusion index) catch the two errors that would otherwise survive. The assumption that matters most is the calibration curve: a different slope changes SpO2 by more than the accuracy target. The second is the sampling rate, which sets the heart-rate accuracy floor.

### 6.9.1 The accuracy metrics for this record

| Metric | Value | Interpretation |
|---|---:|---|
| Heart rate (clean) | 72.01 bpm | Mean RR 0.8332 s over 35 intervals |
| Heart rate (motion) | 76.24 bpm | +4.23 bpm from artifact |
| Red perfusion index | 2.00% | Normal resting wrist |
| Infrared perfusion index | 4.17% | Wavelength-dependent |
| R | 0.480 | Ratio of ratios |
| SpO2 | 98.0% | Under the stated linear curve |
| Short intervals in motion | 2 | Below 0.6 s, physiologically implausible |

A reviewer should be able to reconstruct every row from the fiducials and the amplitudes. If a row cannot be reconstructed, the record is not auditable.

### 6.10 What the case does not show

The case is a single synthetic waveform. It does not show:

- **Skin-tone variation.** The synthetic signal has no melanin, no scattering change, and no calibration shift.
- **Contact variation.** The synthetic signal has a constant DC and no ambient light.
- **Exercise.** The motion episode is a pure sinusoid, not the broadband artifact of walking or running.
- **Arrhythmia.** The synthetic rhythm is regular, so the case cannot show how an ectopic beat interacts with the peak detector.
- **Calibration uncertainty.** The curve is assumed, not fitted. The real uncertainty in the curve is not represented.

The omissions are the agenda for a validation: stratified skin tone, contact and ambient conditions, the intended activity, the target arrhythmia, and a device-specific calibration. The arithmetic in the case transfers; the accuracy does not.

A useful exercise is to ask which omission would change the conclusion most. For SpO2, it is the calibration and the skin tone. For heart rate, it is the motion and the averaging window. For AF, it is the quality gate and the prevalence. Each measurement has a different sensitivity to the chain, and the validation should focus on the sensitive part.

A third exercise is to ask what the device exports. The export is the difference between a claim that can be audited and a claim that must be trusted. A raw export with timestamps and optical channels is auditable; a summary value is not.


A second exercise is to ask which omission would be invisible in a report. The answer is usually the one that is pooled away: a subgroup bias, a motion segment, or an averaging window. The visible errors are the ones that make the number implausible; the invisible ones are the ones that make it plausible and wrong.


## 7. Code, query, or calculation

The script synthesises a PPG with a known rate, detects peaks, computes heart rate and perfusion index, computes SpO2 from red and infrared AC/DC amplitudes, adds a motion episode, and shows the rate error and the short intervals.

```python
import math, random

FS, DUR, HR = 100.0, 30.0, 72.0

def g(t,c,a,w): return a*math.exp(-0.5*((t-c)/w)**2)

def ppg_beat(t):
    return g(t,0.0,1.0,0.09) + g(t,0.34,0.35,0.12)

def synth_ppg(fs,dur,hr,seed=1,motion=None):
    rnd=random.Random(seed); n=int(fs*dur); x=[1.0]*n
    rr=60.0/hr; t0=0.5; k=0
    while t0+k*rr < dur-0.5:
        c=t0+k*rr
        for i in range(max(0,int((c-0.2)*fs)), min(n,int((c+0.8)*fs))):
            x[i]+=ppg_beat(i/fs-c)
        k+=1
    for i in range(n):
        t=i/fs
        x[i]+=0.05*math.sin(2*math.pi*0.25*t)+rnd.gauss(0,0.005)
    if motion:
        a,b,amp,f = motion
        for i in range(int(a*fs), min(n,int(b*fs))):
            x[i]+=amp*math.sin(2*math.pi*f*i/fs)
    return x

def detect_peaks(x, fs, refractory=0.4):
    mn,mx=min(x),max(x); thr=mn+0.5*(mx-mn)
    peaks=[]; i=1
    while i < len(x)-1:
        if x[i]>thr and x[i]>=x[i-1] and x[i]>x[i+1]:
            if not peaks or (i-peaks[-1])/fs >= refractory:
                peaks.append(i)
            elif x[i]>x[peaks[-1]]:
                peaks[-1]=i
        i+=1
    return peaks

def hr_stats(peaks,fs):
    rr=[(peaks[i+1]-peaks[i])/fs for i in range(len(peaks)-1)]
    if not rr: return 0,0,0
    return 60/(sum(rr)/len(rr)), sum(rr)/len(rr), len(rr)

def spo2(red_dc,red_ac,ir_dc,ir_ac):
    R=(red_ac/red_dc)/(ir_ac/ir_dc)
    return 110-25*R, R

x=synth_ppg(FS,DUR,HR)
p=detect_peaks(x,FS)
hr,mrr,n=hr_stats(p,FS)
print("clean: peaks=%d HR=%.2f mean_RR=%.4f" % (len(p),hr,mrr))
print("PI_red=%.2f%% PI_ir=%.2f%%" % (0.02/1.0*100, 0.05/1.2*100))
s,R=spo2(1.0,0.02,1.2,0.05)
print("R=%.4f SpO2=%.2f%%" % (R,s))

xm=synth_ppg(FS,DUR,HR,motion=(10.0,15.0,1.2,1.5))
pm=detect_peaks(xm,FS)
hrm,mrrm,nm=hr_stats(pm,FS)
print("motion: peaks=%d HR=%.2f mean_RR=%.4f" % (len(pm),hrm,mrrm))
rrs=[(pm[i+1]-pm[i])/FS for i in range(len(pm)-1)]
short=[r for r in rrs if r<0.6]
print("short_intervals=%d min_RR=%.3f" % (len(short), min(rrs)))

pe8=[q for q in p if q/FS<=8.0]
print("HR_8s=%.2f HR_30s=%.2f" % (hr_stats(pe8,FS)[0], hr))
print("SpO2_R048=%.1f SpO2_R052=%.1f" % (110-25*0.48, 110-25*0.52))

for fs in (25.0,100.0):
    print("fs=%d peak_quantum_ms=%.0f HR_error_at_72=%.2f"
          % (fs, 1000/fs, 60/(60/72+1/fs)-72))
```

Expected output:

```text
clean: peaks=35 HR=72.01 mean_RR=0.8332
PI_red=2.00% PI_ir=4.17%
R=0.4800 SpO2=98.00%
motion: peaks=37 HR=76.24 mean_RR=0.7869
short_intervals=2 min_RR=0.480
HR_8s=72.10 HR_30s=72.01
SpO2_R048=98.0 SpO2_R052=97.0
fs=25 peak_quantum_ms=40 HR_error_at_72=-3.30
fs=100 peak_quantum_ms=10 HR_error_at_72=-0.85
```

Three implementation notes.

The peak detector uses a threshold at 50% of the range and a 0.4 s refractory period. The refractory period prevents double detection of the dicrotic notch at normal rates, but it also prevents detection of a true rate above 150 bpm. A device that claims exercise heart rate needs a shorter refractory period and a motion model.

The perfusion index is computed from fixed AC and DC amplitudes, not from the synthetic waveform. This is deliberate: the example isolates the arithmetic of the ratio from the peak detection. A real pipeline estimates the DC as a slow moving average and the AC as the peak-to-trough amplitude within each beat.

The SpO2 comparison at R = 0.48 and R = 0.52 shows the slope. The 0.04 change in R produces a 1.0 percentage point change in SpO2, which is the sensitivity of this calibration. A different device with a different slope would produce a different sensitivity.

## 8. Data, standards, and regulatory context

PPG standards are split between device safety, SpO2 accuracy, and software claims. The regulatory landscape changed materially in 2025 with the FDA draft guidance on pulse oximeters and skin tone.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| ISO 80601-2-61:2017 | Pulse oximeter safety and performance | Withdrawn, revised by the 2026 edition | SpO2 accuracy and test methods |
| ISO 80601-2-61:2026 | Revised pulse oximeter standard | Current | Check the edition in a claim |
| FDA draft guidance, Jan 2025 | SpO2 accuracy, skin-tone performance, labelling | Draft; not final | ARMS < 3%, stratified study |
| FDA recognised consensus standards | Partial recognition of ISO accuracy clause | Binding through review | FDA applies stricter ARMS |
| IEC 60601-1 | General medical electrical safety | Binding when claimed | Base safety |
| IEC 60601-1-2 | Electromagnetic compatibility | Binding when claimed | Ambient light and EMI |
| HIPAA | Protection of PPG data | Binding in the US | Continuous data are sensitive |
| EU MDR 2017/745 | Clinical evaluation | Binding in the EU | Post-market surveillance |
| HL7 FHIR | Observation exchange | Voluntary | SpO2 and heart rate codes |
| LOINC | Measurement codes | Voluntary; widely required | 2708-6 SpO2, 8867-4 heart rate |

### 8.1 SpO2 accuracy: ISO versus FDA

The ISO standard sets an accuracy root mean square difference (ARMS) of 4.0% or less over the 70% to 100% saturation range. The FDA does not recognise that clause as sufficient. Its guidance and review practice apply a stricter criterion: an ARMS below 3.0%, with a two-sided 95% confidence interval upper limit below 3.0%.

The difference is not cosmetic. A device with an ARMS of 3.8% meets the ISO clause and fails the FDA criterion. A device with an ARMS of 2.8% meets both. A manufacturer targeting the US market must design to the stricter threshold.

The 2025 draft guidance adds a second dimension: non-disparate performance across skin tones. The recommended study has at least 150 healthy, diversely pigmented participants and at least 3,000 paired SpO2 and co-oximeter SaO2 observations. Skin tone is assessed by the Monk Skin Tone scale and by colorimetry-derived Individual Typology Angle. The co-primary criteria include an overall ARMS below 3.0% and a between-group bias difference below 3.5% for the 70% to 85% range and below 1.5% for the 85% to 100% range.

The draft is not final. Its recommendations are still the best public statement of what a reviewer is likely to expect, and they are labelled as draft in any claim.

### 8.2 The accuracy statistics in a validation

A pulse oximeter validation reports more than one number, and the numbers answer different questions.

- **Bias** is the mean difference between SpO2 and SaO2. A positive bias means the device reads high.
- **Precision** is the standard deviation of the differences. It describes the spread, not the direction.
- **ARMS** is the root mean square difference. It combines bias and precision: ARMS = √(bias² + precision²).
- **Limits of agreement** are the bias ± 1.96 × precision, the range within which 95% of differences are expected.
- **Subgroup bias** is the bias within a stratum such as a skin-tone group. It can be large while the pooled bias is small.

A device with a bias of 0% and a precision of 2% has an ARMS of 2.0%. A device with a bias of 2% and a precision of 0% has an ARMS of 2.0%. The two devices have the same ARMS and different clinical behaviour: the first is unbiased but noisy, the second is precise but systematically wrong. The ARMS alone does not distinguish them.

For heart rate, the analogous statistics are the mean absolute error, the bias, the limits of agreement, and the proportion of readings within 5 bpm or 10% of the reference. The last is often the most interpretable for a consumer product.

### 8.3 Claims and the evidence they need

| Claim | Quantity | Evidence needed | Regulatory route |
|---|---|---|---|
| Wellness heart rate | Mean rate over a window | ECG agreement in the intended population | General wellness, no device claim |
| Clinical heart rate | Rate with accuracy target | ECG agreement, motion, subgroup | Device clearance |
| SpO2 | Saturation in percent | Controlled desaturation, co-oximetry, ARMS, skin tone | Device clearance |
| AF notification | Irregular rhythm alert | Confirmatory ECG, sensitivity, specificity, PPV | Device clearance |
| Sleep staging | Stage per epoch | Polysomnography agreement, per-stage metrics | Device clearance if clinical |
| Blood pressure | Pressure in mmHg | Cuff or arterial reference, calibration, validation | Device clearance; no general consumer PPG claim |
| Stress or hydration | Unspecified index | Construct validity and outcome association | Research or wellness only |

The table is a design aid, not a legal classification. The route depends on the intended use, the claims language, and the jurisdiction. The evidence column, however, is a reasonable minimum for any serious claim.

A useful regulatory question is whether the feature is a measurement or a notification. A notification can be cleared with a narrower evidence package than a measurement, because it is a prompt for confirmatory testing rather than a value used for treatment. The Apple hypertension feature is an example: it is a notification, it does not measure blood pressure, and its labelling says so.

### 8.4 What the standards do not cover

Three gaps matter for wearables.

First, the standards are written for pulse oximeters, not for wrist heart-rate monitors. A wrist device that reports heart rate has no equivalent ISO accuracy standard; the validation is usually against ECG with a mean absolute error and a Bland-Altman analysis.

Second, the standards do not cover AF detection from PPG. The claim is a software function, and the evidence is a clinical validation against a confirmatory ECG. The Apple Heart Study is the largest public example, and its design and limitations are discussed in Section 9.

Third, the standards do not cover blood pressure estimation from PPG. The claim is a research area with no cleared consumer product that measures blood pressure without a cuff. A product that reports a blood-pressure number from PPG is making a claim that requires a dedicated validation, not a heart-rate validation.

### 8.3 Regulatory status of consumer features

A heart-rate display for general wellness operates under enforcement discretion in the US, subject to the general wellness policy. An SpO2 measurement for clinical use is a device. An AF notification is a device function with a cleared or cleared-and-modified pathway. A blood-pressure measurement is a device with a dedicated pathway and, at present, no consumer PPG product with a general blood-pressure claim.

The regulatory status determines the evidence, not the technology. The same optical sensor can support a wellness heart rate and a clinical SpO2, but the validation for the two is different by orders of magnitude.

The distinction is easy to blur in product language. "Tracks your blood oxygen" is a wellness phrase; "measures SpO2" is a measurement claim; "detects hypoxia" is a diagnostic claim. Three different sentences, three different evidence packages, one sensor. A product team should choose the sentence before the study, because the sentence determines the study.

A second regulatory nuance is that a feature can be cleared for one population and not another. A device cleared for adults at rest is not cleared for children or for use during exercise. The labelling carries the limitation, and the marketing must not exceed it.

## 9. Industry practice and case evidence

### 9.1 The Apple Heart Study

The Apple Heart Study enrolled 419,297 participants and used a smartwatch PPG to detect irregular rhythms. Of the participants, 2,161 (0.52%) received a notification. Among those who returned ECG patches, 34% had AF confirmed.

The study demonstrated that a PPG-based notification can be deployed at scale and that the confirmation pathway is essential. The 34% confirmation rate implies that most notified participants did not have AF on the patch, which is expected in a low-prevalence population and is a property of the positive predictive value, not necessarily of the sensor. The study is discussed further in Topic 09.

The signal-chain lesson is that the notification rate depends on the quality gate and the irregularity feature as much as on the classifier. A gate that rejects motion reduces false notifications but also reduces sensitivity to real AF during activity.

### 9.2 The Apple hypertension notification

Apple received FDA clearance in September 2025 for a passive hypertension notification feature using PPG over 30-day windows. The feature does not measure blood pressure. In the validation study of 2,229 participants, sensitivity was 41.2% overall and 53.7% for Stage 2 hypertension, with specificity of 92.3%.

The claim is narrow and the numbers are modest. A sensitivity of 41.2% means that most hypertensive users are not notified; a specificity of 92.3% means that about 8% of non-hypertensive users receive a notification. The product communicates that it is not a blood-pressure measurement.

The signal-chain lesson is that a long window (30 days) is part of the measurement. The feature is not a single PPG beat; it is a pattern of PPG features over a month, combined with user characteristics. The validation must match the window, and a 30 s recording cannot reproduce it.

### 9.3 Consumer sleep and heart-rate devices

Consumer sleep devices that use PPG estimate heart rate and sleep-wake from the pulse and motion. Chinoy et al. compared seven devices with polysomnography and found variable agreement by device and stage. The PPG devices could estimate total sleep time but were less reliable for REM and N1.

The signal-chain lesson is the same as in Topic 03: the sensor set bounds the construct. A PPG plus accelerometer observes pulse and motion, not cortical activity. A five-stage claim from that sensor set requires validation against polysomnography, not against another wearable.

### 9.4 Pulse oximetry and skin tone

The FDA's 2025 draft guidance is a response to a documented body of evidence that pulse oximeters can overestimate SpO2 in patients with darker skin, particularly at low saturation. The mechanism is the optical path: melanin absorbs light, the AC component is smaller, and the calibration curve derived in a lighter-skinned population does not transfer exactly.

The documented evidence includes controlled desaturation studies and clinical studies. The inference for product teams is that a pooled ARMS can hide a subgroup bias. A device with an overall ARMS of 2.5% can have a 4% bias in a subgroup and still meet a pooled criterion. The stratified analysis is the only way to see it.

### 9.5 Blood-pressure research

PPG-based blood-pressure estimation is an active research area. The methods use pulse transit time, pulse wave analysis, or machine learning on the waveform. The validation problem is that the reference (cuff or arterial line) has its own error, and the calibration is subject to drift.

A second documented area is sleep and stress. Consumer devices report sleep stages and "stress" scores from PPG and accelerometry. The sleep claims can be compared with polysomnography; the stress claims usually cannot, because there is no accepted reference for a momentary stress score. A stress score is a constructed index, and its validation is a construct-validity argument, not an agreement study.

The inference for product teams is that claims should be sorted by the availability of a reference. Heart rate, SpO2, and sleep stages have references. Stress, hydration, and "readiness" often do not, and the evidence for them is weaker by construction.

The documented state is that no consumer PPG device has a general blood-pressure measurement claim. The inference is that a product claiming blood pressure from PPG without a cuff calibration and a dedicated validation is overreaching. The claim may be a trend or a notification, not a measurement.

### 9.5.1 Consumer heart-rate validation in practice

Consumer heart-rate validation usually follows a common pattern: a laboratory study with a chest-strap ECG or a 12-lead ECG as reference, a protocol with rest and exercise stages, and a Bland-Altman analysis by stage. The reported metric is often the proportion of readings within 5 bpm or 10% of the reference.

The pattern has known limitations. The laboratory protocol is not free-living, the participants are often young and fit, and the exercise is often cycling or running rather than the daily activity the product will see. A device that performs well in the laboratory can perform poorly in daily life, and a device that performs well in daily life can fail the laboratory protocol because the reference is harder to align.

Inference: the validation should include a free-living component, or the claim should be limited to the conditions studied. The two are not interchangeable.

A further limitation is the reference alignment. A chest strap and a wrist PPG do not observe the same beat at the same instant; the pulse arrives later. The delay is tens to hundreds of milliseconds and varies with posture and blood pressure. A Bland-Altman analysis that pairs beats without accounting for the delay will show a bias that is partly a timing artifact. The pairing window must be stated.

### 9.6 What the cases have in common

Four patterns recur across the public cases.

- **The notification is not the diagnosis.** Apple Heart Study notifications required patch confirmation; the Apple ECG classification is a single-lead rhythm statement, not a diagnosis. The product is designed around the confirmation pathway.
- **The window is part of the claim.** The hypertension notification uses 30-day windows; a 30 s reading cannot reproduce it. A validation that uses a different window is validating a different product.
- **The population is part of the claim.** A study in healthy volunteers does not establish performance in patients with anaemia, obesity, or poor perfusion. The claim should name the population.
- **The accuracy is conditional.** SpO2 accuracy at rest is not SpO2 accuracy during motion. A claim that omits the condition is incomplete.

These patterns are inference from the documented cases, not a regulatory rule. They are, however, the patterns a reviewer will look for.

A fifth pattern is the confirmation pathway. Every notified user needs a next step: a patch, a clinic visit, or an ECG. The product team should design that step before launch, because the notification rate determines the demand on the pathway. A device that notifies 0.5% of 10 million users creates 50,000 confirmations, which is a capacity problem for the health system even if each notification is appropriate.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Motion labeled as arrhythmia | Short RR intervals with high accelerometer variance | Motion peaks detected as beats | False AF or tachycardia | Quality gate; accelerometer |
| Poor contact | Low DC, low perfusion index | Air gap or loose strap | Noisy signal, unstable rate | Contact flag; prompt |
| Excessive pressure | Low perfusion index despite good contact | Vessel compression | Underestimated pulsation | Prompt looser fit |
| Ambient light | DC drift with room lighting | External light at detector | Baseline wander, saturation | Modulated LED; shielding |
| Low perfusion | Perfusion index below 0.3% | Cold, vasoconstriction | Unstable ratio | Flag; do not report SpO2 |
| Calibration borrowed | SpO2 values at extremes | Curve from another device | Systematic bias | Device-specific calibration |
| Extrapolation below 70% | SpO2 reported at 60% | Curve outside studied range | Unknown accuracy | Limit range; label |
| Averaging window hidden | Smooth curve with latency | 30 s rolling average | Missed events | Report window |
| Sampling too low | Dicrotic notch absent | 25 Hz sampling | Morphology features invalid | State sampling rate |
| Skin-tone bias | Subgroup difference in validation | Melanin absorption | Systematic overestimate | Stratified study |
| Venous pulsation | High perfusion index, abnormal shape | Venous blood pulsation | Overestimated AC | Reposition; flag |
| Non-wear treated as zero | Flat signal with no pulse | Device off body | Missingness bias | Wear detection |
| Reference mismatch | SpO2 compared with another oximeter | Both devices have error | Underestimated error | Use co-oximetry |
| Single-beat ratio | Unstable SpO2 | Noise in AC | Random SpO2 | Average over beats |
| AF from irregularity alone | Irregular RR with P waves | Ectopy or sinus arrhythmia | False AF | Confirmatory ECG |
| Averaging hides dispersion | Stable-looking value | Mean over noisy beats | Understated uncertainty | Report IQR or SE |
| Rounding exceeds accuracy | Integer display | False precision | Overconfidence | Match display to accuracy |
| Calibration drift with temperature | SpO2 changes with device heat | LED wavelength shift | Bias | Thermal compensation |
| Anaemia | Normal SpO2 with low oxygen content | Functional saturation | False reassurance | Clinical context |
| Carboxyhaemoglobin | Normal SpO2 in CO poisoning | Functional versus fractional | Missed toxicity | Co-oximetry |
| Sensor too tight | Low perfusion index | Vessel compression | Underestimated AC | Fit guidance |
| Sensor too loose | Motion and ambient light | Air gap | Unstable signal | Fit guidance |
| Display latency hidden | Smooth curve | Long rolling window | Missed events | Report window |
| Heart rate from one interval | Unstable display | Noisy single RR | Misleading | Average over window |
| SpO2 from one beat | Unstable value | Noisy ratio | Misleading | Average over beats |
| Calibration from another device | Plausible but biased | Curve mismatch | Systematic error | Device-specific study |
| Reference another oximeter | Small apparent error | Shared error | Underestimated error | Co-oximetry |
| Firmware changes filter | Values shift after update | Silent parameter change | Non-comparable trends | Version and re-verify |
| Wrist size affects path | Signal differs by fit | Optical geometry | Bias by body habitus | Fit range validation |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The sensor is in stable contact | Optical path | Loose strap, motion | Contact flag; perfusion index |
| The DC is measured, not reconstructed | Ratio of ratios | AC-coupled front end | Raw optical channels |
| The AC is above the noise floor | Stable ratio | Low perfusion | Perfusion index threshold |
| The calibration is device-specific | SpO2 accuracy | Borrowed curve | Calibration study |
| The calibration range covers the value | Accuracy claim | Extrapolation below 70% | Declared range |
| Motion is detected before detection | Clean peaks | Periodic motion | Accelerometer gate |
| The averaging window is documented | Interpretation | Hidden rolling average | Report window |
| The reference is co-oximetry | SpO2 validation | Another oximeter | Blood gas |
| The population matches deployment | Generalisation | Skin tone, perfusion | Stratified study |
| Skin tone is measured, not guessed | Bias analysis | Fitzpatrick as colorimetry | MST and ITA |
| The claim matches the evidence | Regulatory | Wellness claim for clinical use | Claim review |
| The device is worn | Data existence | Non-wear | Wear detection |
| The temperature is stable | LED and detector behaviour | Cold ambient, heating | Temperature log |
| The calibration is current | SpO2 accuracy | Firmware or sensor change | Re-verify after change |
| The averaging window is fixed | Comparability | Firmware changes window | Version the window |
| The display precision matches accuracy | Interpretation | Integer bpm, 5 bpm error | Report accuracy |
| The export includes raw channels | Auditability | Summary only | Irreproducible | Export policy |
| The firmware version is known | Reproducibility | Silent update | Changed behaviour | Version metadata |
| The device is calibrated after service | Accuracy | Sensor replacement | Bias | Service record |
| The wrist fit is within the validated range | Optical geometry | Very small or large wrist | Signal bias | Fit validation |
| The firmware is the validated version | Behaviour | Silent update | Changed output | Version pinning |

## 12. Exercises

### Exercise 1. Ratio of ratios (application, 30 min)

**Deliverable:** compute R and SpO2 for red DC 0.8 V, red AC 0.016 V, infrared DC 1.4 V, infrared AC 0.07 V, using the linear curve SpO2 = 110 - 25R.

**Acceptance criteria:** correct R; correct SpO2; the calibration assumption stated.

**Time:** 30 minutes.

### Exercise 2. Perfusion index (recall, 15 min)

**Deliverable:** compute the perfusion index for both channels in Exercise 1 and explain why they differ.

**Acceptance criteria:** AC/DC × 100%; the wavelength and path explanation given.

**Time:** 15 minutes.

### Exercise 3. Motion artifact (application, 35 min)

**Deliverable:** design a quality gate that uses accelerometer variance, perfusion index, and RR interval distribution. State the thresholds and the order of evaluation.

**Acceptance criteria:** each threshold has a rationale; the gate cannot turn motion into a label; edge cases (resting tremor, exercise) considered.

**Time:** 35 minutes.

### Exercise 4. Averaging (critique, 25 min)

**Deliverable:** compare 1 s, 10 s, and 30 s averaging windows for resting heart rate, exercise recovery, and AF detection.

**Acceptance criteria:** latency and noise trade-off quantified; each decision matched to a window.

**Time:** 25 minutes.

### Exercise 5. SpO2 validation design (design, 60 min)

**Deliverable:** a one-page design for a pulse oximeter validation following the FDA 2025 draft guidance, including participants, paired samples, skin-tone strata, and co-primary criteria.

**Acceptance criteria:** at least 150 participants and 3,000 pairs; MST and ITA; ARMS and bias-difference criteria; range 70-100%.

**Time:** 60 minutes.

### Exercise 6. Skin-tone bias (critique, 40 min)

**Deliverable:** explain how a pooled ARMS can hide a subgroup bias, with a numerical example.

**Acceptance criteria:** two subgroups with different bias but similar pooled ARMS; clinical consequence stated.

**Time:** 40 minutes.

### Exercise 7. AF claim (design, 45 min)

**Deliverable:** a validation outline for a PPG AF notification, including reference standard, confirmation pathway, inconclusive rate, and prevalence adjustment.

**Acceptance criteria:** PPV computed at a stated prevalence; confirmation ECG specified; no diagnosis from PPG alone.

**Time:** 45 minutes.

### Exercise 8. Sampling rate (application, 25 min)

**Deliverable:** compute the heart-rate error from 25 Hz and 100 Hz peak timing for a 72 bpm rate, and state which morphology features each rate supports.

**Acceptance criteria:** correct quantization arithmetic; dicrotic notch assessed; recommendation justified.

**Time:** 25 minutes.

### Exercise 9. Device-independent ratio audit (application, 30 min)

**Deliverable:** for a device that reports only SpO2 and not the raw optical channels, list the checks that cannot be performed and the claims that cannot be audited.

**Acceptance criteria:** at least four missing checks named; the consequence for each stated; a recommendation for what the device should export.

**Time:** 30 minutes.

### Submission standard

Calculations must show R, the perfusion index, and the SpO2 with units. A validation design must state the reference, the population, the range, the number of paired observations, and the skin-tone strata. A critique must name the clinical consequence of the bias, not only its size. A quality gate must state the order of evaluation and the action for each branch.

For the validation design, the acceptance criteria include the reference, the range, the sample size, and the skin-tone strata. A design that omits the strata fails, even if the pooled accuracy target is met. The strata are the point of the 2025 guidance.

A submission that reports an SpO2 without the calibration curve is returned. A submission that reports a heart rate without the window is returned. The standard is the same as the one used in Section 5.

## 13. Self-check answers

**1. What is the perfusion index?** AC/DC × 100%.

**2. Compute R for red ratio 0.020 and infrared ratio 0.040.** R = 0.020/0.040 = 0.50.

**3. What SpO2 does R = 0.50 give on the linear curve?** 110 - 25 × 0.50 = 97.5%.

**4. Why is a perfusion index of 0.2% a problem?** The AC is 0.2% of the DC, near the noise floor. The ratio and the SpO2 become unstable.

**5. Why is motion dangerous for an AF detector?** Motion produces peaks that are not beats, creating short RR intervals that look irregular. Without a quality gate, the algorithm labels artifact as arrhythmia.

**6. What is the difference between SpO2 and SaO2?** SpO2 is the peripheral estimate from pulse oximetry; SaO2 is the arterial saturation from a blood gas co-oximeter. The two are related but not identical.

**7. What is ARMS?** The root mean square difference between SpO2 and SaO2, in percentage points. It combines bias and precision.

**8. Compute ARMS for bias 1% and SD 2%.** √(1² + 2²) = 2.24%.

**9. Why does a 25 Hz PPG lose the dicrotic notch?** The notch contains content above 10 Hz; at 25 Hz, Nyquist is 12.5 Hz, so the notch is attenuated or aliased.

**10. What is the heart-rate error from 40 ms peak quantization at 72 bpm?** RR = 0.833 s; 0.833 + 0.040 = 0.873 s gives 68.7 bpm, a -3.3 bpm error.

**11. Why is a 30 s average unable to show a 10 s arrhythmia?** The average smooths over the event. A 10 s arrhythmia contributes one third of the window and is diluted by the surrounding normal beats.

**12. What does the FDA 2025 draft guidance require for skin tone?** At least 150 diversely pigmented participants, at least 3,000 paired observations, MST and ITA assessment, and between-group bias limits.

**13. Why can a pooled ARMS hide a subgroup bias?** A large group with small bias can dominate the pooled statistic while a small group has a large bias. The pooled value looks acceptable; the subgroup does not.

**14. What is the reference standard for SpO2?** Arterial blood gas measured by a co-oximeter, paired in time with the device reading.

**15. Why is blood pressure from PPG not a measurement?** It requires a calibration and a validation against a reference; without them, the output is an estimate with unknown error.

**16. Why can a device with a good pooled ARMS still fail a skin-tone criterion?** A large group with small bias can dominate the pooled statistic while a small group has a large bias. The pooled value looks acceptable; the between-group difference exceeds the limit.

**17. What is the difference between functional and fractional saturation?** Functional saturation counts oxygenated haemoglobin as a fraction of oxygenated plus deoxygenated haemoglobin. Fractional saturation includes carboxyhaemoglobin and methaemoglobin in the denominator. A pulse oximeter reports functional saturation.

**18. Why is the systolic peak not the best fiducial for pulse transit time?** The peak occurs after the pressure wave arrival and is shaped by wave reflection. The maximum upslope is closer to the arrival time and less affected by the dicrotic wave.

**19. What does a perfusion index of 8% suggest?** A high pulsatile fraction, which can be strong perfusion or motion or venous pulsation. It is not a quality score on its own; the waveform and the accelerometer are needed.

**20. Why is a green LED used for wrist heart rate?** Green light is strongly absorbed by haemoglobin and penetrates shallowly, so it samples the superficial vessels where the pulsatile change is largest relative to the tissue. It is less useful for SpO2 because the contrast between oxygenated and deoxygenated haemoglobin is small at green wavelengths.

**21. What is the difference between pulse rate and heart rate?** Pulse rate counts pressure or volume pulses at the periphery; heart rate counts cardiac contractions. They are usually equal, but a pulse deficit can occur in arrhythmia or poor perfusion.

**22. Why is a 30-day window part of the hypertension feature?** The feature is a pattern over a month, not a single beat. A 30 s recording cannot reproduce the window, and a validation on a different window is validating a different feature.

**23. What is the difference between bias and precision in an SpO2 study?** Bias is the mean difference from the reference; precision is the standard deviation of the differences. A device can be unbiased and imprecise, or biased and precise. ARMS combines both.

**24. Why does a 1% error in the red AC produce only a 0.12 percentage point SpO2 error?** A 1% relative change in the red AC changes the red ratio by 1%, and therefore R by 1%. R is 0.48, so 1% is 0.0048. With a slope of -25% per unit R, the SpO2 change is 25 × 0.0048 = 0.12 percentage points.

**Rubric for Exercise 5 (SpO2 validation).** A strong design follows the draft guidance's participant count, paired observations, skin-tone strata, and co-primary criteria. A weak design uses a convenience sample and reports only pooled ARMS.

**Rubric for Exercise 7 (AF claim).** A strong design uses a confirmatory ECG, reports the inconclusive rate, and computes PPV at a realistic prevalence. A weak design reports sensitivity and specificity without prevalence or confirmation.

## 14. Interview questions

**1. How does a PPG sensor measure heart rate?**
A strong answer describes the optical path, the pulsatile AC component, peak detection, and the averaging window. A weak answer says "it shines light through the skin."

**2. Why is a wrist PPG not a finger oximeter?**
A strong answer explains reflectance versus transmittance, path length, calibration, and motion sensitivity. A weak answer says "the wrist is less accurate."

**3. How would you validate an SpO2 feature?**
A strong answer uses controlled desaturation, co-oximetry, ARMS, and stratified skin tone. A weak answer compares with another oximeter.

**4. What is the ratio of ratios?**
A strong answer writes the formula, explains the cancellation, and states that the calibration is empirical. A weak answer calls it the SpO2 formula.

**5. How do you handle motion in wrist PPG?**
A strong answer uses an accelerometer gate, rejects or corrects, and validates during the intended activity. A weak answer filters the signal.

**6. Why did the Apple Heart Study need a confirmation patch?**
A strong answer explains PPV in a low-prevalence population and the need for a diagnostic reference. A weak answer says "to check the watch."

**7. What is the perfusion index and what does it not tell you?**
A strong answer defines AC/DC and explains that it is not a quality score because motion can raise it. A weak answer treats it as signal quality.

**8. How does skin tone affect PPG?**
A strong answer explains melanin absorption, reduced AC, calibration mismatch, and the FDA stratified study. A weak answer says "darker skin absorbs more light" without the consequence.

**9. What averaging window would you use for resting heart rate?**
A strong answer chooses a window based on the decision, reports the latency, and distinguishes trend from event. A weak answer says "10 seconds" without justification.

**10. Can PPG measure blood pressure?**
A strong answer describes research methods, the calibration problem, and the absence of a general consumer claim. A weak answer says yes and cites a paper.

**11. How do you compute PPV from sensitivity and specificity?**
A strong answer uses Bayes with prevalence: PPV = sens × prev / (sens × prev + (1 - spec) × (1 - prev)). A weak answer confuses PPV with sensitivity.

**12. What is the first thing you check when SpO2 is implausible?**
A strong answer checks perfusion index, contact, and ratio stability before the calibration. A weak answer adjusts the calibration.

**13. When would you reject a segment instead of correcting it?**
A strong answer rejects when the artifact is broadband and larger than the pulse, because correction would fabricate a signal. A weak answer always filters.

**14. How do you audit a heart-rate number from a device that exports only the number?**
A strong answer says you cannot audit it; you can only compare it with a reference. A weak answer assumes the number is the mean of valid beats.

**15. What is the role of the accelerometer in a PPG pipeline?**
A strong answer uses it as a motion reference for gating and, in some methods, for adaptive correction. A weak answer uses it only for step counting.

**16. What would you ask a vendor who claims SpO2 accuracy of 2% ARMS?**
A strong answer asks for the range, the population, the skin-tone strata, the number of paired observations, the reference, and the calibration study. A weak answer accepts the number.

**17. Why is the confirmation pathway part of the product?**
A strong answer explains that a notification without a next step creates anxiety and workload without a diagnosis, and that the pathway determines the clinical value. A weak answer treats the alert as the endpoint.

## 15. Key terms

- **AC component** — pulsatile optical signal.
- **ARMS** — accuracy root mean square difference.
- **Ambient light** — external light reaching the detector.
- **Bias** — mean difference from the reference.
- **Calibration curve** — empirical map from R to SpO2.
- **Co-oximeter** — blood gas device that measures SaO2.
- **Contact pressure** — force between sensor and skin.
- **DC component** — average optical level.
- **Dicrotic notch** — secondary pulse wave.
- **Fitzpatrick scale** — skin phototype classification.
- **ITA** — Individual Typology Angle.
- **LED** — light-emitting diode.
- **Monk Skin Tone** — 10-point skin tone scale.
- **Motion artifact** — movement-induced signal.
- **Non-wear** — device off body.
- **Perfusion** — blood flow to tissue.
- **Perfusion index** — AC/DC × 100%.
- **Photodetector** — light-to-current transducer.
- **PPG** — photoplethysmography.
- **Pulse transit time** — delay from ECG R to pulse.
- **Ratio of ratios** — R = (ACr/DCr)/(ACir/DCir).
- **Reflectance** — backscattered light geometry.
- **SaO2** — arterial oxygen saturation.
- **SpO2** — peripheral oxygen saturation.
- **Transmittance** — through-tissue geometry.
- **Venous pulsation** — venous contribution to AC.
- **Absorption coefficient** — light absorption per unit path length.
- **Adaptive filter** — filter that uses a reference signal.
- **Co-oximetry** — multi-wavelength blood gas measurement.
- **Functional saturation** — SpO2 as a fraction of functional haemoglobin.
- **Fractional saturation** — SaO2 including dyshaemoglobins.
- **Isobestic point** — wavelength of equal absorption.
- **Optical path length** — distance light travels in tissue.
- **PPV** — positive predictive value.
- **Prevalence** — share of the population with the condition.
- **Pulse rate** — peripheral pulse count.
- **Scattering** — light redirection in tissue.
- **Wear detection** — on-body status.
- **Signal-to-noise ratio** — pulse power divided by noise power.
- **Template subtraction** — removing an average artifact.
- **Spectral subtraction** — removing an artifact spectrum.
- **Calibration study** — controlled desaturation study.
- **Non-disparate performance** — comparable accuracy across skin tones.
- **Bias** — mean difference from the reference.
- **Precision** — standard deviation of the differences.
- **Limits of agreement** — bias ± 1.96 × precision.
- **Subgroup bias** — bias within a stratum.
- **Wear time** — time the device is on body.
- **Coverage** — share of wear time with a valid value.
- **Latency** — delay from event to reported value.
- **Duty cycle** — fraction of time the LED is on.
- **Confirmation pathway** — the test that follows a notification.
- **Notification rate** — share of users receiving an alert.
- **Readiness score** — constructed wellness index.
- **Construct validity** — evidence that an index measures its intended construct.
- **Pulse deficit** — difference between apical and peripheral beats.
- **Diffuse reflectance** — light returned from tissue.
- **Specular reflection** — light reflected at the skin surface.
- **Melanin** — pigment that absorbs light.
- **Optical geometry** — arrangement of LED and detector.
- **Skin tone** — melanin-related light absorption.
- **Saturation** — signal at the detector or ADC rail.
- **Wearable** — body-worn device.

## 16. Sources

All URLs accessed 25 October 2026.

1. ISO 80601-2-61:2017, Medical electrical equipment — Part 2-61: Particular requirements for basic safety and essential performance of pulse oximeter equipment. https://www.iso.org/standard/67963.html
2. US Food and Drug Administration. Pulse Oximeters for Medical Purposes — Non-Clinical and Clinical Performance Testing, Labeling, and Premarket Submission Recommendations. Draft guidance, January 2025. https://www.fda.gov/media/184896/download
3. US Food and Drug Administration. FDA Proposes Updated Recommendations to Help Improve Performance of Pulse Oximeters Across Skin Tones. January 2025. https://www.fda.gov/news-events/press-announcements/fda-proposes-updated-recommendations-help-improve-performance-pulse-oximeters-across-skin-tones
4. US Food and Drug Administration. Pulse Oximeters product page. https://www.fda.gov/medical-devices/products-and-medical-procedures/pulse-oximeters
5. US Food and Drug Administration. Recognized Consensus Standards, ISO 80601-2-61 entry. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfStandards/detail.cfm?standard__identification_no=37508
6. Medicines and Healthcare products Regulatory Agency. The use and regulation of pulse oximeters. https://www.gov.uk/guidance/the-use-and-regulation-of-pulse-oximeters-information-for-healthcare-professionals
7. Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *N Engl J Med*. 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
8. Apple Inc. Hypertension Notifications Validation Paper. September 2025. https://www.apple.com/health/pdf/Hypertension_Notifications_Validation_Paper_September_2025.pdf
9. Chinoy ED, Cuellar JA, Huwa KE, et al. Performance of seven consumer sleep-tracking devices compared with polysomnography. *Sleep*. 2021;44(5):zsaa291. https://doi.org/10.1093/sleep/zsaa291
10. US Food and Drug Administration. General Wellness: Policy for Low Risk Devices. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/general-wellness-policy-low-risk-devices
11. IEC 60601-1-2, electromagnetic compatibility. https://webstore.iec.ch/
12. HL7 FHIR R5 Observation. https://hl7.org/fhir/R5/observation.html
13. ISO 80601-2-61:2026, revised pulse oximeter standard. https://www.iso.org/standard/67963.html
14. IEC 60601-1, general requirements for basic safety. https://webstore.iec.ch/
15. US Food and Drug Administration. Approach for Improving the Performance Evaluation of Pulse Oximeter Devices Taking into Consideration Skin Pigmentation, Race and Ethnicity. Discussion paper, 2023. https://www.fda.gov/media/173905/download
16. ClinicalTrials.gov. Apple Electrocardiogram Clinical Validation Study, NCT03492554. https://clinicaltrials.gov/study/NCT03492554
17. Apple Inc. Using Apple Watch for Arrhythmia Detection. December 2018. https://www.apple.com/healthcare/docs/site/Apple_Watch_Arrhythmia_Detection.pdf
18. World Health Organization. Pulse oximetry training manual. https://www.who.int/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_eeg.md`](03_eeg.md)
- Next topic: [`05_respiration-and-other-signals.md`](05_respiration-and-other-signals.md)
- Related topic: [`09_validation.md`](09_validation.md)
- Module survey file: [`../../11_physiological_signals_wearables.md`](../../11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
- Related topic: [`08_digital-biomarkers.md`](08_digital-biomarkers.md)
- Related topic: [`10_deployment-on-device.md`](10_deployment-on-device.md)
- Dataset: [`https://physionet.org/`](https://physionet.org/)
