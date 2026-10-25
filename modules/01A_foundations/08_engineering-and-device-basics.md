# Engineering and Device Basics

> Module 01A — Foundations, Assumptions, and Units
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03
> Version 2026.3

## 1. Why this topic exists

A medical device is a measurement instrument with a computer attached. The measurement is produced by
a chain: a sensor converts a physical quantity into an electrical signal, an analog front end
amplifies and filters it, an analog-to-digital converter samples it, and an algorithm turns the samples
into a number or an alarm. Every link has a failure mode, and every failure mode is invisible in the
final display.

- A photoplethysmographic watch samples at 25 Hz and reports a heart rate of 72 bpm. The patient has
  atrial fibrillation with a rapid ventricular response at 150 bpm. The sampling rate is adequate for
  the average rate but the algorithm's window averages the irregularity away, and the reported value
  is both wrong and reassuring.
- An ECG is sampled at 250 Hz. The device meets the Nyquist criterion for a 100 Hz bandwidth, but the
  diagnostic standard requires 150 Hz, and pacemaker spikes and notched QRS complexes are distorted.
  The automated interpretation calls a normal rhythm.
- A 10-bit ADC with a ±5 mV input range has a resolution of 9.8 µV. The thermal noise of the electrode
  interface is larger than the signal of interest for late potentials, so the feature is unmeasurable
  regardless of the algorithm.
- A continuous glucose monitor lags interstitial glucose by about 8 minutes and samples every 5
  minutes. An alert algorithm tuned on venous blood glucose fires late, and the user concludes that
  the sensor is inaccurate when the problem is the compartment and the lag.
- A device drifts by 2% over six months. The daily quality control passes because the control material
  is measured by the same drifted device. The bias is discovered only at the next calibration.
- An alarm threshold is set to maximise sensitivity. The device alarms on 60% of monitored hours, the
  nursing staff silences it, and the one true event is missed. The failure is a human-factors failure
  caused by an engineering choice.

These are not software bugs. They are instrument-design failures: the measurement was specified
without a bandwidth, a resolution, a noise budget, or a latency requirement.

The topic matters to data scientists because the data they receive are the output of this chain. A
model trained on a device's output inherits the device's filters, its sampling, its saturation
behaviour, and its artifact rejection. When the device changes, the model's input distribution changes,
and the model fails for reasons that are invisible in the feature table.

The commercial consequence is direct. Device software is regulated under IEC 62304 and the quality
system requirements of the FDA and the EU MDR. A data scientist who cannot read a sampling
specification or a noise budget cannot contribute to a regulatory submission, and a model that cannot
state its input requirements cannot be validated.

> **Note:** This topic covers the engineering concepts that determine data quality. It does not cover
> circuit design, and it does not replace a biomedical engineer's review. The numbers are illustrative.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the signal chain from sensor to decision and name the failure mode at each stage. Assessed
   in Exercise 1.
2. **Apply** the Nyquist criterion to a stated bandwidth, choose a sampling rate with a justified
   margin, and explain why post-hoc filtering cannot remove aliasing. Assessed in Exercises 2 and 3.
3. **Compute** the resolution, quantisation noise, and signal-to-noise ratio of an ADC for a stated
   input range and bit depth. Assessed in Exercise 4.
4. **Estimate** thermal noise for a source resistance and bandwidth, and compare it with the
   quantisation floor. Assessed in Exercise 5.
5. **Distinguish** calibration, verification, drift, linearity, and hysteresis, and design a
   calibration check. Assessed in Exercise 6.
6. **Specify** latency, sampling, and bandwidth requirements for a monitoring algorithm, and explain
   how each affects the clinical decision. Assessed in Exercise 7.
7. **Critique** a device data sheet for missing specifications and unsupported claims. Assessed in
   Exercise 8.
8. **Explain** the regulatory framework for device software, including IEC 62304, ISO 14971, and the
   FDA and EU MDR requirements. Assessed in interview questions and Exercise 8.

## 3. The core mental model

The signal chain is the organising structure. Each stage transforms the signal and adds error.

```text
  PHYSIOLOGY        SENSOR          ANALOG FRONT END        ADC            PROCESSOR        OUTPUT
  source            transducer      amplifier, filter       sample,        algorithm        display,
  (mV, mmHg,        (electrode,     (gain, bandwidth,       quantise       (feature,        alarm,
   photons)         photodiode,     impedance, noise)                      classifier)      actuator
                    pressure)
      |                 |                  |                   |               |               |
   access,           contact,           noise,              aliasing,       assumptions,    latency,
   perfusion,        sensitivity,       saturation,         resolution,     failure modes,  interpret-
   motion            drift              CMRR                jitter          version         ation
```

### 3.1 The error budget

Every measurement has an error budget: a list of contributions that sum, in quadrature if independent,
to the total uncertainty.

```text
   u_total^2 = u_sensor^2 + u_analog^2 + u_quantisation^2 + u_algorithm^2 + u_calibration^2
```

The budget determines what is measurable. If the physiological signal of interest is 10 µV and the
electrode noise is 20 µV, no algorithm can recover it from a single channel. If the quantisation step
is 10 µV, the signal is at the resolution limit. The budget is the engineering equivalent of the
assumptions ledger: it states what must be true for the number to mean what it claims.

### 3.2 Sampling and the Nyquist criterion

A signal that contains frequencies up to f_max must be sampled at more than 2·f_max to be
reconstructable. Sampling below that rate folds high frequencies into low ones, and the folding is
irreversible.

```text
   f_alias = | f_signal - k * f_s |   for the integer k that brings the result below f_s/2
```

A 10 Hz signal sampled at 12 Hz appears at 2 Hz. A 10 Hz signal sampled at 8 Hz also appears at 2 Hz.
The two are indistinguishable after sampling, which is why an anti-aliasing filter must be applied
before the ADC, not after.

### 3.3 Resolution and noise

The ADC's resolution is set by the input range and the bit depth:

```text
   LSB = full-scale range / 2^N
   Quantisation SNR = 6.02 N + 1.76 dB   (for a full-scale sine)
```

A 12-bit converter over a 10 mV range has an LSB of 2.44 µV and a quantisation SNR of 74 dB. The
thermal noise of a 10 kΩ source over a 150 Hz bandwidth at 310 K is about 0.16 µV, which is well below
the quantisation floor. The design is then quantisation-limited, not noise-limited, and increasing the
bit depth would help. If the source resistance were 1 MΩ, the thermal noise would exceed the
quantisation floor and increasing the bit depth would not help.

### 3.4 The device as an instrument with a claim

A device is not only a measurement; it is a claim about a measurement. The claim has a population, a
range, a reference method, and a set of conditions. A pulse oximeter's claim is about arterial oxygen
saturation in a population with adequate perfusion and a calibration derived from a validation study.
Outside that population, the claim does not hold, even if the device is functioning as designed.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Transducer | A device that converts one form of energy to another | Calling any sensor a transducer without the conversion | none |
| Sensor | The element that responds to the measurand | Confusing with the whole device | none |
| Analog front end (AFE) | The amplification and filtering stage before the ADC | Treating it as part of the sensor | none |
| ADC | Analog-to-digital converter | Confusing with the whole acquisition system | none |
| Sampling rate (fs) | Samples per second | Confusing with the signal bandwidth | Hz |
| Nyquist frequency | Half the sampling rate | Calling it the maximum signal frequency | Hz |
| Nyquist criterion | Sample at more than twice the highest frequency | Applying it without an anti-aliasing filter | none |
| Aliasing | Folding of high frequencies into low ones | Treating it as noise removable by filtering | none |
| Anti-aliasing filter | A low-pass filter before the ADC | Placing it after the ADC | none |
| Bandwidth | The range of frequencies a system passes | Confusing with the sampling rate | Hz |
| Cutoff frequency | The frequency at which response falls by a stated amount | Treating it as a hard boundary | Hz |
| Resolution | The smallest change the system can distinguish | Confusing with accuracy | LSB or unit |
| LSB | Least significant bit, the quantisation step | Confusing with the noise floor | volts or unit |
| Quantisation noise | Error from rounding to discrete levels | Confusing with thermal noise | volts |
| Signal-to-noise ratio (SNR) | Signal power divided by noise power | Reporting it without the bandwidth | dB |
| Noise floor | The noise level in the absence of signal | Treating it as a constant across gains | volts |
| Thermal noise | Johnson noise from resistance | Confusing with flicker noise | volts |
| Shot noise | Noise from discrete charge carriers | Ignoring it in optical sensors | volts |
| Flicker noise | 1/f noise at low frequencies | Ignoring it in long recordings | volts |
| Common-mode rejection ratio (CMRR) | Ability to reject a shared input signal | Confusing with differential gain | dB |
| Input impedance | Resistance to current flow at the input | Ignoring it for high-impedance sources | Ω |
| Gain | Ratio of output to input | Confusing with amplification in dB | dimensionless |
| Saturation | Clipping when the signal exceeds the input range | Confusing with physiological saturation | volts |
| Offset | A constant added to the signal | Confusing with drift | volts |
| Drift | Change in output over time at a fixed input | Confusing with noise | unit per time |
| Linearity | Proportionality of output to input | Assuming it without testing | none |
| Hysteresis | Direction-dependent response | Ignoring it in pressure sensors | unit |
| Calibration | Establishing the response-reference relationship | Calling an adjustment a calibration | none |
| Verification | Confirming a specification is met | Confusing with calibration | none |
| Accuracy | Closeness to the true value | Confusing with precision | unit |
| Precision | Repeatability | Confusing with accuracy | unit |
| Latency | Delay from physiological event to output | Confusing with processing time only | s |
| Jitter | Variation in sample timing | Ignoring it in rhythm analysis | s |
| Duty cycle | Fraction of time a system is active | Confusing with sampling rate | fraction |
| Motion artifact | Signal corruption from movement | Treating it as random noise | none |
| Baseline wander | Low-frequency drift from respiration or movement | Removing it with a filter that distorts the signal | Hz |
| Powerline interference | 50 or 60 Hz pickup | Removing it with a notch that hides signal | Hz |
| Notch filter | A narrow band-stop filter | Using it without checking the signal band | Hz |
| Anti-aliasing | Filtering before sampling | Applying after sampling | none |
| Oversampling | Sampling well above the Nyquist rate | Confusing with averaging | none |
| Decimation | Reducing the sample rate after filtering | Decimating without filtering | none |
| Dynamic range | Range from the noise floor to full scale | Confusing with input range | dB |
| Input range | The maximum and minimum measurable input | Confusing with dynamic range | volts |
| Slew rate | Maximum rate of change of the output | Ignoring it for fast signals | V/s |
| CMRR | Common-mode rejection ratio | Confusing with SNR | dB |
| Electromagnetic compatibility (EMC) | Operation without interference | Treating it as optional | none |
| Leakage current | Current flowing to earth through the patient | Ignoring it in device design | µA |
| Isolation | Electrical separation between circuits | Confusing with insulation | none |
| Defibrillation protection | Ability to withstand a defibrillation pulse | Ignoring it in connected devices | none |
| Biocompatibility | Suitability of a material for body contact | Confusing with sterility | none |
| Sterility assurance level | Probability of a non-sterile unit | Confusing with a guarantee | probability |
| Human factors | The study of use error and usability | Treating it as training | none |
| Use error | An error arising from the design of the use environment | Blaming the user | none |
| Alarm fatigue | Desensitisation from excessive alarms | Treating it as a staff problem | none |
| Intended use | The stated clinical purpose of a device | Confusing with a marketing claim | none |
| Software as a medical device (SaMD) | Software intended for a medical purpose without hardware | Assuming it is unregulated | none |
| IEC 62304 | Medical device software life cycle standard | Confusing with ISO 14971 | none |
| ISO 14971 | Medical device risk management standard | Treating risk management as a one-time document | none |

Three clusters deserve emphasis.

**Resolution, accuracy, and precision.** Resolution is the smallest distinguishable change; accuracy is
closeness to truth; precision is repeatability. A device can have 24-bit resolution and 5% accuracy. The
resolution sets the floor for detecting change; the accuracy sets the bias; the precision sets the
noise. Confusing them leads to specifications such as "24-bit accuracy", which is a category error.

**Latency and sampling rate.** Sampling rate determines how often the signal is observed; latency
determines how long after the event the output appears. A device can sample at 500 Hz and still have a
5-second latency because of buffering and algorithm windows. For an alarm, latency is the relevant
specification, not the sampling rate.

**Intended use and claim.** The intended use defines the population, the setting, and the purpose. A
device intended for spot-checking in adults is not validated for continuous monitoring in neonates. The
claim is bounded by the intended use, and using the device outside it is off-label use, regardless of
whether the numbers look reasonable.

## 5. The mechanics, step by step

The procedure specifies and evaluates a measurement chain for a stated clinical decision.

### Step 1 — State the clinical decision and the required timeliness

**Input.** A decision such as "alert on hypoxaemia within 60 seconds."

**Output.** A decision statement with a required detection delay.

**Reasoning.** The decision determines the bandwidth, the latency, and the sampling rate. An alarm that
must fire within 60 seconds cannot use a 5-minute averaging window. A trend display that updates hourly
does not need 500 Hz sampling.

**Branch point.** If the decision is retrospective, latency requirements relax but data completeness
requirements tighten. State which.

### Step 2 — Identify the physiological signal and its bandwidth

**Input.** The measurand.

**Output.** A frequency range containing the relevant information.

**Reasoning.** The bandwidth is determined by the physiology, not by the device. Diagnostic ECG
requires 0.05 to 150 Hz; a monitoring ECG may be adequate at 0.5 to 40 Hz; photoplethysmography for
heart rate requires about 0.5 to 10 Hz; respiration is below 1 Hz. The bandwidth must include the
features the algorithm uses, not only the features a clinician sees.

**Branch point.** If the feature is a high-frequency notch (a pacemaker spike, a late potential), the
bandwidth must extend to the feature, which may be far above the visible waveform.

### Step 3 — Choose the sampling rate

**Input.** The bandwidth.

**Output.** A sampling rate above 2·f_max, with a stated margin.

**Reasoning.** The Nyquist criterion gives the minimum; practical designs use three to five times the
bandwidth to allow for filter roll-off and for algorithms that need oversampling. Diagnostic ECG at
150 Hz is sampled at 500 Hz, a ratio of 1.67 to the Nyquist rate of 300 Hz. PPG at 10 Hz is sampled at
25 Hz.

**Branch point.** If the sampling rate cannot be increased, reduce the bandwidth with an analog filter
before the ADC. Do not rely on digital filtering after sampling.

### Step 4 — Verify the anti-aliasing filter

**Input.** The sampling rate and the bandwidth.

**Output.** A filter specification with a stopband attenuation.

**Reasoning.** The anti-aliasing filter must attenuate frequencies above the Nyquist frequency before
they reach the ADC. A filter with a gradual roll-off allows high-frequency noise to alias into the
signal band. The specification must state the attenuation at the Nyquist frequency and above.

**Branch point.** If the filter is digital, it must run at a higher internal rate than the output rate;
a digital filter after decimation cannot remove aliasing that already occurred.

### Step 5 — Choose the ADC resolution

**Input.** The signal amplitude and the required resolution.

**Output.** A bit depth and an input range.

**Reasoning.** The LSB is the input range divided by 2^N. A 12-bit converter over ±5 mV gives 2.44 µV.
The required resolution is set by the smallest physiological change the decision depends on, not by the
converter's marketing specification.

**Branch point.** If the signal of interest is smaller than the LSB, increase the gain or the bit depth,
or accept that the feature is unmeasurable.

### Step 6 — Compute the quantisation noise

**Input.** The bit depth.

**Output.** A quantisation SNR.

**Reasoning.** For a full-scale sine, SNR = 6.02N + 1.76 dB. A 12-bit converter gives 74 dB, which is a
ratio of about 5,000 to 1. The quantisation noise is approximately LSB/√12 for a uniformly distributed
error.

**Branch point.** If the signal does not use the full input range, the effective SNR is lower by the
amount of the range not used. Gain staging matters.

### Step 7 — Estimate the noise sources

**Input.** The source resistance, bandwidth, and temperature.

**Output.** A thermal noise estimate.

**Reasoning.** Thermal noise is Vn = √(4kTRB). At 310 K, a 10 kΩ source over 150 Hz gives about 0.16 µV;
a 1 MΩ source gives about 1.6 µV. Electrode contact noise, amplifier noise, and flicker noise add to
this. The largest source sets the noise floor.

**Branch point.** If the thermal noise exceeds the quantisation floor, increasing the bit depth does not
improve the measurement. Reduce the source impedance or the bandwidth instead.

### Step 8 — Compute the signal-to-noise ratio

**Input.** Signal amplitude and noise amplitude.

**Output.** An SNR in dB.

**Reasoning.** SNR = 20 log10(signal/noise) for amplitudes. A 1 mV ECG with 0.16 µV noise has an SNR of
about 76 dB. The required SNR depends on the feature: QRS detection needs a few dB, late-potential
analysis needs 20 dB or more, and ST-segment measurement needs a stable baseline more than a high SNR.

**Branch point.** If the SNR is inadequate, average repeated epochs, reduce the bandwidth, or change
the sensor. Do not expect the algorithm to recover a signal below the noise floor.

### Step 9 — Set the gain and check the input range

**Input.** The signal amplitude range and the ADC range.

**Output.** A gain and a saturation check.

**Reasoning.** The gain should place the largest expected signal near, but not beyond, the ADC's full
scale. A gain that is too high clips on motion artifact; a gain that is too low wastes resolution. Both
failures are visible in the data as clipping or as a coarse staircase.

**Branch point.** If the signal range is unknown, use a conservative gain and accept lower resolution,
or use an automatic gain control with a stated settling time.

### Step 10 — Specify the calibration

**Input.** The sensor and the measurand.

**Output.** A calibration procedure with a reference and an interval.

**Reasoning.** Calibration establishes the relationship between the raw output and the reference
quantity. It requires a traceable reference, a procedure, and an interval. A device without a
calibration procedure produces numbers, not measurements.

**Branch point.** If the device is factory-calibrated and cannot be recalibrated, the calibration
interval is the device lifetime and the drift specification must be stated.

### Step 11 — Check drift over the calibration interval

**Input.** The drift specification.

**Output.** A bound on the bias at the end of the interval.

**Reasoning.** Drift is a slow change in output at a fixed input. A 2% drift over six months means that
a result at the end of the interval may be 2% away from the result at the start, independent of any
change in the patient. For a threshold decision, the drift must be small relative to the distance from
the threshold.

**Branch point.** If the drift is comparable to the decision margin, shorten the calibration interval
or add an internal reference.

### Step 12 — Check linearity across the measuring range

**Input.** A calibration curve.

**Output.** A linearity statement and a residual plot.

**Reasoning.** Many sensors are linear only over part of their range. A pulse oximeter's calibration
curve is empirical and nonlinear; a pressure sensor may saturate at high pressures. The linearity
statement must cover the range in which the device is used.

**Branch point.** If the device is used outside its linear range, the calibration is invalid. Restrict
the claim to the validated range.

### Step 13 — Check hysteresis

**Input.** Increasing and decreasing input sweeps.

**Output.** A hysteresis bound.

**Reasoning.** Hysteresis is a difference in output depending on the direction of change. It matters
for pressure sensors and for any device that is used in both increasing and decreasing conditions. A
hysteresis of 2 mmHg is negligible for a trend and important for a threshold.

**Branch point.** If hysteresis is large, calibrate in the direction of use or specify a correction.

### Step 14 — Check the time base and jitter

**Input.** The sampling clock.

**Output.** A jitter specification.

**Reasoning.** Rhythm analysis depends on the timing of samples. Jitter of even a few milliseconds can
shift a heart rate variability measure or a QT interval. The clock must be stable, and the timestamp
must be recorded with the sample, not assigned at transmission.

**Branch point.** If jitter is unknown, measure it with a known-frequency input before trusting
interval-based features.

### Step 15 — Specify the latency

**Input.** The decision and the algorithm.

**Output.** An end-to-end latency budget.

**Reasoning.** Latency is the sum of sensor response, filter group delay, buffer length, algorithm
window, processing time, and transmission. A 5-second algorithm window plus a 2-second transmission
delay gives at least 7 seconds of latency. For an alarm, the budget must be less than the clinical
tolerance.

**Branch point.** If the latency exceeds the tolerance, shorten the window (at the cost of specificity)
or change the algorithm.

### Step 16 — Design the filters

**Input.** The signal and noise bands.

**Output.** A filter specification with cutoff, order, and phase behaviour.

**Reasoning.** Filters remove noise but also distort the signal. A high-pass filter at 0.5 Hz removes
baseline wander but distorts the ST segment; a notch at 50 Hz removes powerline interference but can
remove signal content at the same frequency. Zero-phase filtering avoids phase distortion but requires
the whole record and adds latency.

**Branch point.** If the filter distorts a diagnostic feature, use a linear-phase filter with a
justified cutoff, and report the filter in the data provenance.

### Step 17 — Handle motion artifact

**Input.** The artifact characteristics.

**Output.** A detection and rejection strategy.

**Reasoning.** Motion artifact is not random noise; it is structured and often larger than the signal.
Strategies include accelerometer-based reference subtraction, signal-quality indices, and rejection
with a documented rule. Rejection reduces data completeness, which is itself a bias if it is
differential across patients.

**Branch point.** If rejection rates differ by activity level or by patient group, the cleaned dataset
is not representative. Report the rejection rate by group.

### Step 18 — Design the alarm logic

**Input.** The clinical event and the tolerated false-alarm rate.

**Output.** A threshold, a persistence rule, and a priority level.

**Reasoning.** An alarm is a classifier with a threshold. The threshold determines the trade-off
between sensitivity and the false-alarm rate. A persistence rule (the condition must hold for N
seconds) reduces false alarms but adds latency. The priority level determines whether the alarm is
audible, visible, or logged.

**Branch point.** If the false-alarm rate exceeds what staff will tolerate, the alarm will be ignored.
Measure the rate in the deployment environment, not in the bench test.

### Step 19 — Check electrical safety

**Input.** The device's connection to the patient.

**Output.** A leakage current and isolation specification.

**Reasoning.** IEC 60601-1 sets limits on patient leakage current, earth leakage current, and
dielectric strength, and requires isolation for patient-connected circuits. A device that is safe on a
bench can become unsafe with a poor earth connection, with electrosurgery, or with a defibrillation
event.

**Branch point.** If the device is used in an environment with electrosurgery or MRI, additional
particular standards apply. The general standard alone is insufficient.

### Step 20 — Check electromagnetic compatibility

**Input.** The device's environment.

**Output.** An EMC specification for emissions and immunity.

**Reasoning.** IEC 60601-1-2 sets requirements for emissions and immunity. A device that is immune in
a quiet lab can fail near a mobile phone, a electrosurgical unit, or an RFID reader. The failure may be
a transient artifact or a permanent reset.

**Branch point.** If the device is used in a home environment, the immunity requirements differ from a
hospital environment. State the intended environment.

### Step 21 — Check cybersecurity

**Input.** The device's connectivity.

**Output.** A threat model and a set of controls.

**Reasoning.** A connected device is a computer on a network. Threats include unauthorized access,
malicious firmware, denial of service, and data exfiltration. The controls include authentication,
encryption, signed updates, and network segmentation. The FDA and the EU MDR require cybersecurity
documentation for connected devices.

**Branch point.** If the device cannot be updated, a vulnerability is permanent. State the support
period and the update mechanism.

### Step 22 — Plan verification and validation

**Input.** The requirements.

**Output.** A verification plan (does the device meet the specification) and a validation plan (does it
meet the clinical need).

**Reasoning.** Verification is a bench activity against requirements; validation is a clinical activity
against the intended use. A device can pass every verification test and fail validation because the
clinical environment differs from the bench.

**Branch point.** If validation is not possible before launch, plan a post-market performance study and
state the limitation.

### Step 23 — Plan the human factors evaluation

**Input.** The use environment.

**Output.** A use-error analysis and a formative and summative evaluation.

**Reasoning.** IEC 62366-1 requires a use-engineering process. Use errors are design failures, not
user failures. The evaluation must include representative users, representative environments, and the
tasks that are most error-prone.

**Branch point.** If the device is used by lay users at home, the evaluation must include lay users.
Clinician usability does not transfer.

### Step 24 — Document the intended use and the claim

**Input.** All of the above.

**Output.** An intended-use statement with population, setting, and purpose.

**Reasoning.** The intended use bounds the claim. A device intended for spot-checking is not validated
for continuous monitoring; a device intended for adults is not validated for children. The statement is
the anchor for the regulatory review and for the data scientist's assumptions.

**Branch point.** If a new use is proposed, it requires a new validation. The existing data do not
transfer automatically.

### Step 25 — Define the data provenance

**Input.** The acquisition settings.

**Output.** Metadata stored with every record.

**Reasoning.** The sampling rate, filter settings, gain, firmware version, and calibration date
determine what the data mean. A dataset without these cannot be pooled across devices or reprocessed
after a firmware change.

**Branch point.** If the provenance is missing, the data can still be analysed descriptively but cannot
be pooled or compared across devices.

### Step 26 — Monitor post-market performance

**Input.** Field data.

**Output.** A monitoring plan and a complaint review.

**Reasoning.** Performance can change with firmware, with a new sensor lot, with a new population, or
with a new environment. Post-market surveillance is a regulatory requirement and a data-science
necessity: the model's input distribution changes when the device changes.

**Branch point.** If a firmware change alters the signal processing, the model must be revalidated.
Treat the firmware version as a feature and as a stratification variable.

### 5.1 The branch map

```text
Measurement requirement
  |
  +-- bandwidth unknown? -----------> derive from the physiology and the feature
  |
  +-- sampling below 2 fmax? -------> alias; fix the analog filter or the rate
  |
  +-- signal below LSB? ------------> gain, bit depth, or accept unmeasurable
  |
  +-- thermal noise above LSB? -----> reduce source impedance or bandwidth
  |
  +-- latency above tolerance? -----> shorten the window or change the algorithm
  |
  +-- motion artifact large? -------> detect, reject, and report rejection rates
  |
  +-- alarm rate too high? ---------> persistence rule and priority redesign
  |
  +-- firmware changed? ------------> revalidate the model
```

### 5.2 The four checks that catch most device-data errors

1. **Record the sampling rate, filter, and gain with every dataset.** Without them, the data cannot be
   reprocessed.
2. **Check the bandwidth against the feature.** A feature above the bandwidth is not present in the
   data, regardless of the algorithm.
3. **Check the resolution against the effect size.** A change smaller than the LSB cannot be detected.
4. **Check the latency against the decision.** An alarm that arrives after the decision is not an
   alarm.

## 6. Worked example

A team is designing a wearable patch that records a single-lead ECG, a photoplethysmographic (PPG)
waveform, and intermittent continuous glucose monitor (CGM) readings. The clinical decision is to alert
on a new arrhythmia within 60 seconds and to display a heart rate trend. The device must run for two
days on a 500 mAh battery.

The case works through the acquisition chain: bandwidth, sampling, aliasing, resolution, noise, data
rate, latency, and battery.

### 6.1 Bandwidth requirements

```text
ECG, diagnostic quality:      0.05 to 150 Hz
ECG, rhythm monitoring only:  0.5 to 40 Hz
PPG for heart rate:           0.5 to 10 Hz
Respiration from PPG:         0.05 to 1 Hz
CGM:                          a new value every 5 minutes (0.00333 Hz)
```

The arrhythmia decision needs the QRS morphology, not only the rate, so the diagnostic bandwidth is
appropriate. A rhythm-only device could use the 40 Hz bandwidth and save power, but it would lose the
notch and spike features that distinguish ventricular from supraventricular rhythms.

The bandwidth choice determines the sampling rate, the data rate, and the battery. It is the first
decision, and it is a clinical decision expressed as a frequency range.

### 6.2 Sampling rate

```text
ECG
  Nyquist minimum = 2 x 150 Hz = 300 Hz
  Design rate     = 500 Hz
  Ratio to Nyquist = 500/300 = 1.67

PPG
  Nyquist minimum = 2 x 10 Hz = 20 Hz
  Design rate     = 25 Hz
  Ratio to Nyquist = 25/20 = 1.25

Respiration
  Nyquist minimum = 2 x 2 Hz = 4 Hz
  Design rate     = 10 Hz
  Ratio to Nyquist = 10/4 = 2.50
```

The ECG ratio of 1.67 is adequate because the anti-aliasing filter has a sharp roll-off; the AAMI EC11
standard discusses the trade-off between the 500 Hz rate and the 150 Hz bandwidth and notes that the
sampling rate leaves little margin for the highest-frequency components. A design with a slower filter
would need a higher rate.

The PPG ratio of 1.25 is tight. It is adequate for heart rate, which is a low-frequency feature, and
inadequate for pulse-wave morphology, which contains higher harmonics. A device that claims pulse-wave
analysis at 25 Hz is over-claiming.

### 6.3 Aliasing

```text
A 10 Hz signal sampled at 12 Hz:
  f_alias = |10 - 1 x 12| = 2 Hz                (k = 1)

A 10 Hz signal sampled at 8 Hz:
  f_alias = |10 - 1 x 8| = 2 Hz                 (k = 1)

The two signals are indistinguishable after sampling except for phase.
```

The samples of the 10 Hz signal at 12 Hz are the negatives of the samples of the 2 Hz signal, which is
a 180° phase shift. The frequency and amplitude are identical. No post-hoc filter can separate them,
because the information that distinguished them was discarded at the ADC.

The clinical version of this error: a 50 Hz powerline artifact sampled at 100 Hz appears at 50 Hz, but
a 60 Hz artifact sampled at 100 Hz appears at 40 Hz and may be mistaken for a physiological rhythm. An
anti-aliasing filter at 40 Hz would have removed both.

### 6.4 Resolution and quantisation

```text
ECG input range: +/-5 mV, full scale 10 mV
Bit depth: 12

LSB = 10 mV / 2^12
    = 10 mV / 4096
    = 2.4414 uV

Quantisation SNR = 6.02 x 12 + 1.76
                 = 72.24 + 1.76
                 = 74.00 dB
```

A 2.44 µV step is small relative to a 1 mV QRS complex (about 0.24% of the amplitude) and large
relative to a 10 µV late potential (about 24%). The converter is adequate for rhythm analysis and
marginal for high-resolution ECG.

The quantisation noise is approximately LSB/√12 = 0.705 µV for a uniformly distributed error. This is
a useful comparison with the thermal noise in the next step.

### 6.5 Noise budget

```text
Thermal noise of the source
  Vn = sqrt(4 k T R B)
     = sqrt(4 x 1.380649e-23 x 310 x 10,000 x 150)
     = sqrt(2.567e-14)
     = 1.602e-7 V
     = 0.160 uV

Quantisation noise
  = 2.4414 / sqrt(12) = 0.705 uV

Amplifier voltage noise (typical, from the datasheet)
  = 1.0 uV over the band

Electrode contact noise (typical, dry electrode)
  = 5 to 20 uV over the band
```

The electrode contact noise dominates. The converter and the amplifier are not the limiting factor; the
sensor interface is. This is the most common finding in a noise budget and the most commonly missed:
teams upgrade the ADC when the problem is the electrode.

Total noise, assuming independence:

```text
  u_total = sqrt(0.160^2 + 0.705^2 + 1.0^2 + 10^2)
          = sqrt(0.0256 + 0.497 + 1.0 + 100)
          = sqrt(101.52)
          = 10.08 uV
```

The total is essentially the electrode noise. Improving the ADC from 12 to 16 bits would reduce the
quantisation term from 0.705 to 0.044 µV, changing the total from 10.08 to 10.05 µV, a 0.3% improvement
for a substantial power and cost increase.

### 6.6 Signal-to-noise ratio

```text
For a 1 mV QRS complex and 10.08 uV total noise:
  SNR = 20 log10(1000/10.08)
      = 20 log10(99.21)
      = 39.93 dB
```

A 40 dB SNR is adequate for QRS detection and marginal for ST-segment analysis, where the signal of
interest is tens of microvolts. The design is adequate for the stated decision (arrhythmia detection)
and inadequate for a diagnostic ECG claim. The claim must match the budget.

### 6.7 Data rate and storage

```text
ECG
  500 samples/s x 12 bits x 1 lead = 6,000 bit/s
  = 750 B/s
  = 2.7 MB/h
  = 64.8 MB/day
  = 129.6 MB for two days

PPG
  25 samples/s x 16 bits x 1 channel = 400 bit/s
  = 50 B/s
  = 4.32 MB/day

CGM
  1 sample per 5 min x 16 bits = 0.053 bit/s, negligible
```

The ECG dominates. A two-day record is about 130 MB uncompressed, which is manageable on the device and
on a phone but not trivial to transmit over a low-power link. The design choice is to store
continuously and transmit summaries, or to store only detected episodes. The first preserves the data
for reanalysis; the second preserves battery and bandwidth.

### 6.8 Latency budget

```text
Sensor response            0.1 s
Anti-aliasing filter group delay  0.05 s
Algorithm window           4.0 s
Processing                 0.2 s
Transmission to phone      1.0 s
Phone processing           0.3 s
Alert display              0.1 s
----------------------------------------
Total                      5.75 s
```

The budget is within the 60-second decision requirement with a large margin. If the algorithm window
were increased to 30 seconds to improve specificity, the total would be about 31.8 seconds, still
within the requirement but with less margin for a slow network.

The latency is dominated by the algorithm window, not by the electronics. This is typical: the
clinical requirement should be allocated to the window first, and the electronics are usually fast
enough.

### 6.9 Battery

```text
Capacity 500 mAh, average draw 10 mA
  Runtime = 500/10 = 50 h = 2.08 days
```

The runtime is just above the two-day requirement, with no margin for cold, aging, or a higher
transmission duty cycle. A design with 20% margin would need about 12 mA average draw or a 600 mAh
battery. The calculation is a reminder that the sampling and transmission choices are power choices.

### 6.10 CGM sampling and lag

```text
Sampling interval 5 min -> 0.00333 Hz
Nyquist 0.001667 Hz -> fastest resolvable period 10 min

Interstitial lag: about 8 min
Total delay from a blood glucose change to a displayed value: about 13 min
```

The CGM cannot resolve a glucose change faster than about 10 minutes, and it lags the blood by about
8 minutes. An alert algorithm tuned on venous blood glucose will fire late by roughly the lag plus half
the sampling interval. The error is not in the sensor's accuracy; it is in the sampling and the
compartment.

### 6.11 What the example demonstrates

| Stage | Specification | Consequence |
|---|---|---|
| Bandwidth | 0.05–150 Hz ECG | Determines the sampling rate and the claim |
| Sampling | 500 Hz ECG, 25 Hz PPG | PPG adequate for rate, not morphology |
| Aliasing | 10 Hz at 12 Hz → 2 Hz | Irreversible without an analog filter |
| Resolution | 2.44 µV LSB | Adequate for QRS, marginal for late potentials |
| Noise | 10.08 µV total, electrode-dominated | ADC upgrade would not help |
| SNR | 39.9 dB | Adequate for arrhythmia, not diagnostic ECG |
| Data rate | 130 MB for two days | Storage and transmission trade-off |
| Latency | 5.75 s | Within the 60 s requirement |
| Battery | 50 h | Just meets two days, no margin |
| CGM | 10 min resolution, 8 min lag | Alert timing constrained |

## 7. Code, query, or calculation

The script computes the sampling, aliasing, quantisation, noise, data-rate, latency, and battery
figures used in section 6.

```python
"""Topic 08 verification: sampling, aliasing, quantisation, noise, data rate."""
from math import sqrt, log10, sin, pi

print("8.1 Nyquist and sampling")
for name, fmax, fs in (("ECG diagnostic", 150.0, 500.0),
                       ("PPG", 10.0, 25.0),
                       ("EEG", 100.0, 250.0),
                       ("respiration", 2.0, 10.0)):
    nyq = 2 * fmax
    print(f"  {name:16s} fmax={fmax:6.1f} Hz  Nyquist={nyq:6.1f} Hz  fs={fs:6.1f} Hz  ratio={fs/nyq:.2f}")

print("\n8.2 Aliasing")
f_sig, f_s = 10.0, 12.0
f_alias = abs(f_sig - round(f_sig / f_s) * f_s)
print(f"  signal {f_sig} Hz sampled at {f_s} Hz -> alias {f_alias} Hz")
f_sig2, f_s2 = 10.0, 8.0
f_alias2 = abs(f_sig2 - round(f_sig2 / f_s2) * f_s2)
print(f"  signal {f_sig2} Hz sampled at {f_s2} Hz -> alias {f_alias2} Hz")

print("\n8.3 Quantisation")
fs_range = 0.010  # volts, +/-5 mV
bits = 12
lsb = fs_range / (2 ** bits)
snr_q = 6.02 * bits + 1.76
print(f"  range = {fs_range*1000:.1f} mV, bits = {bits}")
print(f"  LSB = {lsb*1e6:.3f} uV")
print(f"  quantisation SNR = {snr_q:.2f} dB")

print("\n8.4 Thermal noise")
k = 1.380649e-23
T = 310.0
R = 10000.0
B = 150.0
vn = sqrt(4 * k * T * R * B)
print(f"  sqrt(4kTRB) = {vn*1e6:.3f} uV at R={R:.0f} ohm, B={B:.0f} Hz, T={T:.0f} K")
signal = 1e-3
print(f"  SNR for 1 mV signal = {20*log10(signal/vn):.2f} dB")

print("\n8.5 Data rate")
fs_ecg, bits_ecg, leads = 500, 12, 12
bps = fs_ecg * bits_ecg * leads
print(f"  ECG: {fs_ecg} Hz x {bits_ecg} bit x {leads} leads = {bps} bit/s")
print(f"       = {bps/8:.0f} B/s = {bps/8*3600/1e6:.1f} MB/h = {bps/8*86400/1e6:.1f} MB/day")
fs_ppg, bits_ppg = 25, 16
bps_ppg = fs_ppg * bits_ppg
print(f"  PPG: {fs_ppg} Hz x {bits_ppg} bit = {bps_ppg} bit/s = {bps_ppg/8:.0f} B/s")

print("\n8.6 CGM sampling and lag")
interval_min = 5.0
fs_cgm = 1 / (interval_min * 60)
print(f"  interval = {interval_min:.0f} min -> {fs_cgm:.5f} Hz")
print(f"  Nyquist = {fs_cgm/2:.6f} Hz -> fastest resolvable period = {1/(fs_cgm/2)/60:.1f} min")

print("\n8.7 Battery life")
capacity_mah, draw_ma = 500.0, 10.0
print(f"  {capacity_mah:.0f} mAh / {draw_ma:.0f} mA = {capacity_mah/draw_ma:.1f} h")

print("\n8.8 Aliased sine: amplitude preserved, frequency false")
fs = 12.0
for f in (2.0, 10.0):
    samples = [sin(2 * pi * f * n / fs) for n in range(12)]
    print(f"  f={f:4.1f} Hz first six samples: " + " ".join(f"{x:+.3f}" for x in samples[:6]))
```

**Expected output.**

```text
8.1 Nyquist and sampling
  ECG diagnostic   fmax= 150.0 Hz  Nyquist= 300.0 Hz  fs= 500.0 Hz  ratio=1.67
  PPG              fmax=  10.0 Hz  Nyquist=  20.0 Hz  fs=  25.0 Hz  ratio=1.25
  EEG              fmax= 100.0 Hz  Nyquist= 200.0 Hz  fs= 250.0 Hz  ratio=1.25
  respiration      fmax=   2.0 Hz  Nyquist=   4.0 Hz  fs=  10.0 Hz  ratio=2.50

8.2 Aliasing
  signal 10.0 Hz sampled at 12.0 Hz -> alias 2.0 Hz
  signal 10.0 Hz sampled at 8.0 Hz -> alias 2.0 Hz

8.3 Quantisation
  range = 10.0 mV, bits = 12
  LSB = 2.441 uV
  quantisation SNR = 74.00 dB

8.4 Thermal noise
  sqrt(4kTRB) = 0.160 uV at R=10000 ohm, B=150 Hz, T=310 K
  SNR for 1 mV signal = 75.90 dB

8.5 Data rate
  ECG: 500 Hz x 12 bit x 12 leads = 72000 bit/s
       = 9000 B/s = 32.4 MB/h = 777.6 MB/day
  PPG: 25 Hz x 16 bit = 400 bit/s = 50 B/s

8.6 CGM sampling and lag
  interval = 5 min -> 0.00333 Hz
  Nyquist = 0.001667 Hz -> fastest resolvable period = 10.0 min

8.7 Battery life
  500 mAh / 10 mA = 50.0 h

8.8 Aliased sine: amplitude preserved, frequency false
  f= 2.0 Hz first six samples: +0.000 +0.866 +0.866 +0.000 -0.866 -0.866
  f=10.0 Hz first six samples: +0.000 -0.866 -0.866 +0.000 +0.866 +0.866
```

**Assumptions stated by the script.**

1. The ECG data rate assumes a single lead and no compression. The section 6 example is single-lead;
   the 12-lead figure in the script is a separate illustration.
2. The thermal noise formula assumes a resistive source at 310 K and a brick-wall 150 Hz bandwidth. A
   real filter has a gradual roll-off and the integrated noise is slightly different.
3. The quantisation SNR formula assumes a full-scale sine wave; for physiological signals that do not
   fill the range, the effective SNR is lower.
4. The CGM lag of 8 minutes is a typical value from the literature, not a specification of a particular
   device.
5. The battery calculation ignores temperature, aging, and the transmission duty cycle, all of which
   reduce runtime.

## 8. Data, standards, and regulatory context

### 8.1 The core device standards

| Standard | Scope | Binding status |
|---|---|---|
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General requirements for basic safety and essential performance of medical electrical equipment | Referenced by regulators; FDA recognises edition 3.2 with US national differences |
| IEC 60601-1-2 | Electromagnetic compatibility | Collateral standard to 60601-1 |
| IEC 60601-1-8 | Alarm systems | Collateral standard to 60601-1 |
| IEC 62304 | Medical device software life cycle | Referenced by FDA and EU MDR |
| ISO 14971 | Application of risk management to medical devices | Referenced by regulators |
| IEC 62366-1 | Usability engineering | Referenced by regulators |
| ISO 10993 | Biological evaluation of medical devices | Referenced by regulators |
| ISO 80601-2-61 | Pulse oximeter equipment | Particular standard |
| ANSI/AAMI EC11 | Diagnostic electrocardiographic devices | US consensus standard |
| ANSI/AAMI EC57 | Testing of cardiac rhythm and ST-segment algorithms | US consensus standard |
| EU MDR 2017/745 | Medical device regulation | Binding in the EU |
| 21 CFR Part 820 | Quality system regulation | Binding in the US |

**IEC 60601-1** is the foundational standard. Edition 3.2 is the consolidated version of the 2005
edition with amendments 1 (2012) and 2 (2020), published in August 2020. It covers electrical safety,
mechanical safety, and essential performance, and it is supplemented by collateral standards for EMC
(60601-1-2), alarms (60601-1-8), and particular standards for specific device types.

The standard does not cover every device. In vitro diagnostic equipment outside its definition falls
under IEC 61010, implantable device parts under ISO 14708, and medical gas pipeline systems under
ISO 7396-1. A device that is claimed to comply must state which standards and which editions.

**IEC 62304** defines the software life cycle processes: planning, requirements, architecture, detailed
design, implementation, integration, verification, and maintenance. It assigns software safety classes
A, B, and C based on the possible harm from a software failure. A data scientist contributing to device
software works within this process, and a model that is part of the device is software under the
standard.

**ISO 14971** defines risk management: hazard identification, risk estimation, risk evaluation, risk
control, and residual risk. The risk file is the document that connects a failure mode to a harm and to
a control. A model that can produce a false negative is a hazard under this framework.

### 8.2 ECG-specific standards

**ANSI/AAMI EC11** specifies the frequency response of diagnostic electrocardiographs. The standard's
methods include a 0.67–40 Hz band, a 40–100 Hz band, and a 100–150 Hz band with +10/−30% tolerance,
plus a 1.5 mV, 20 ms triangular-wave test. The device must meet method A plus D, or A plus B plus C.
The standard also discusses the interaction between a 500 Hz sampling rate and the 150 Hz bandwidth,
noting that the sampling rate leaves little margin for the highest-frequency content and that a higher
rate would be needed for a stricter attenuation requirement.

The FDA's *Diagnostic ECG Guidance* (1998) recommends demonstrating substantial equivalence through
conformance to EC11, or to EC38 for ST-segment devices, and lists bandwidth, input dynamic range, and
analog versus digital implementation as key comparison characteristics.

**ANSI/AAMI EC57** specifies the testing of cardiac rhythm and ST-segment algorithms against annotated
databases. It is the standard against which an arrhythmia detector's performance is measured, and it
defines the sensitivity and positive predictive value metrics that a claim must report.

### 8.3 Software as a medical device

Software that is intended for a medical purpose without being part of a hardware device is regulated as
software as a medical device (SaMD). The IMDRF risk categorisation framework classifies SaMD by the
seriousness of the condition and the significance of the information to the decision. The FDA's
guidance on clinical decision support and the EU MDR's classification rules determine whether a
particular algorithm is a device.

The practical consequence for a data scientist: a model that predicts deterioration and is used to
triage patients is likely a device; a model that is used for internal quality improvement may not be.
The determination depends on the intended use, not on the algorithm.

### 8.4 What the standards do not constrain

- They do not specify the clinical decision. The intended use comes from the clinical need.
- They do not guarantee performance in a new population or environment. Validation is separate.
- They do not make a device's output a diagnosis. A measurement is an input to a clinical judgement.
- They do not cover every failure mode. The risk analysis must be device-specific.
- They do not replace post-market surveillance. Performance can change with firmware, sensors, and
  users.

> **Note:** The standards cited are sold documents. The summaries here are from public front matter and
> secondary descriptions and should be checked against the purchased standard before use in a
> regulatory submission.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Device manufacturers** maintain a requirement specification, a risk file, a verification plan, and a
validation plan. The requirement specification includes the sampling rate, bandwidth, resolution, and
latency, and each requirement is traceable to a test. A design review checks that the requirements are
consistent with the intended use.

**Hospital clinical engineering departments** perform incoming inspections, calibration checks, and
preventive maintenance. They maintain an inventory with firmware versions, because a firmware change
can alter the data and because a recall applies to specific versions.

**Data science teams working with device data** record the device model, firmware, sampling rate, and
filter settings in the data dictionary. When these are missing, the team restricts the analysis to
within-device comparisons and states the limitation.

**Regulators** review the risk file, the software life cycle documentation, and the clinical evidence.
The FDA's 510(k) and De Novo pathways and the EU MDR's conformity assessment both require a
demonstration that the device meets its claims in its intended population.

### 9.2 Case evidence

**The FDA infusion pump initiative.** The 2010 white paper reported about 56,000 adverse event reports
and 87 recalls between 2005 and 2009, and identified software defects, user-interface issues, and
unit ambiguity as causes of over- and under-infusion. The initiative led to requirements for
manufacturers and to a programme of static code analysis. It is a documented case of software and
human-factors failures in a device class.

**Alarm fatigue.** The Joint Commission has identified alarm fatigue as a sentinel event category and
issued a national patient safety goal requiring hospitals to establish alarm management policies. The
mechanism is straightforward: an alarm with a high false-positive rate is silenced, and a true event is
then missed. The engineering cause is a threshold chosen for sensitivity without a persistence rule or
a priority hierarchy.

**Pulse oximetry and skin pigmentation.** The FDA's 2021 safety communication noted that pulse oximeter
accuracy can be affected by skin pigmentation, poor circulation, and other factors, and advised
clinicians to be aware of the limitation. The Sjoding et al. (*NEJM* 2020) analysis found that occult
hypoxaemia was roughly three times more frequent in Black patients than in White patients at the same
oximeter readings. The device meets its specification on average and is biased in a subgroup, which is
a validation-population failure as much as a sensor failure.

**Wearable atrial fibrillation detection.** Consumer devices that detect irregular rhythms use PPG and
single-lead ECG with algorithms validated against adjudicated recordings. The validation populations
and the recording conditions differ from the real-world use environment, and the performance in a
young healthy validation cohort does not transfer to an older population with more comorbidity and more
motion artifact. The Apple Heart Study and subsequent studies illustrate both the promise and the
generalisation problem; the documented lesson is that a device claim is bounded by its validation
population.

> **Note:** The infusion pump figures, the Joint Commission alarm goal, and the pulse oximetry finding
> are documented in the cited public sources. The wearable example is described generically because
> specific performance figures vary by device and study.

### 9.3 The practical pattern

Teams that avoid device-data failures do four things:

1. They record the acquisition settings with the data, not in a separate document.
2. They check the bandwidth and resolution against the feature, not against the device's marketing.
3. They measure latency end to end, including transmission and display.
4. They treat a firmware change as a data change and revalidate.

Each practice is a control on a specific failure mode, and each is checkable without a circuit
diagram.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Sampling below twice the bandwidth | A 500 Hz ECG with a 400 Hz analog bandwidth | Aliasing | High frequencies appear as low ones | Anti-aliasing filter before the ADC |
| 2 | No anti-aliasing filter | Digital filtering only | Aliasing already occurred at the ADC | Irreversible corruption | Specify an analog filter with a stopband |
| 3 | Resolution below the effect size | A 10 µV feature with a 20 µV LSB | Quantisation | The feature is unmeasurable | Increase gain or bit depth, or change the sensor |
| 4 | ADC upgraded when the sensor dominates | A 24-bit converter on a noisy electrode | Noise budget misread | Cost and power with no benefit | Compute the budget before the upgrade |
| 5 | Clipping on motion artifact | Flat-topped waveforms in the record | Input range exceeded | Lost data and false features | Conservative gain; saturation detection |
| 6 | Filter distorts a diagnostic feature | ST segment changes with filter setting | High-pass cutoff too high | Misinterpretation | Use a filter validated for the feature; report the setting |
| 7 | Notch filter removes signal | A 50 Hz notch removes a 50 Hz component | Narrow band-stop in the signal band | Loss of diagnostic content | Justify the notch; check the signal band |
| 8 | Latency exceeds the decision window | An alarm that fires after the event | Buffer, window, transmission | No clinical value | Budget the latency end to end |
| 9 | Drift within the calibration interval | A slow bias in the QC chart | Sensor aging | Threshold misclassification | Shorter interval; internal reference |
| 10 | Calibration against a non-traceable reference | A calibration certificate with no chain | Metrological traceability missing | Values not comparable | Traceable reference; certificate review |
| 11 | Firmware change unrecorded | A step change in the data at a version boundary | Signal processing changed | Model input distribution changed | Store the firmware version; revalidate |
| 12 | Rejection rate differs by group | Fewer usable epochs in active patients | Motion artifact rejection | Selection bias | Report rejection rates by group |
| 13 | Alarm threshold set for sensitivity only | Alarms on most of the monitoring time | No persistence rule | Alarm fatigue, missed events | Persistence, priority, and measured false-alarm rate |
| 14 | Device used outside its intended population | Adults' device on children | Validation population mismatch | Unknown accuracy | Restrict use; validate in the population |
| 15 | EMC failure in the real environment | Artifacts near electrosurgery or phones | Immunity insufficient | Transient or permanent malfunction | IEC 60601-1-2 testing in the intended environment |
| 16 | Cybersecurity vulnerability unpatched | An unsupported firmware version | No update mechanism | Data breach or device takeover | Support period; signed updates |
| 17 | Data pooled across devices | Two models with different filters | Acquisition settings differ | Spurious site effects | Include device and firmware as covariates |

Three rows deserve emphasis.

**Row 4, noise budget.** The intuition that "more bits is better" is wrong when the sensor dominates.
The calculation in section 6.5 shows that a 16-bit converter improves the total noise by 0.3% when the
electrode contributes 10 µV. The budget directs the investment to the sensor, where it matters.

**Row 12, differential rejection.** Motion artifact rejection removes epochs, and the epochs removed
are not random: they are the ones with movement, which correlate with activity, agitation, and
sometimes with the clinical event. A model trained on the surviving epochs learns a selected
population. The rejection rate must be reported by group, and the missingness must be treated as
informative.

**Row 17, pooling.** Two devices with different filters produce different waveforms from the same
patient. Pooling them creates a site effect that a model can learn as if it were biology. Device model
and firmware version belong in the data dictionary and in the model's stratification.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The signal bandwidth is known and bounded | Sampling and filter design | An unknown high-frequency feature | Spectral analysis of a representative record |
| 2 | The anti-aliasing filter is effective above Nyquist | Aliasing prevention | A gradual roll-off | Inject a known high-frequency signal and check the alias |
| 3 | The ADC range covers the signal | No clipping | Motion artifact or a high-gain setting | Saturation counters; visual inspection |
| 4 | The noise is stationary over the recording | Averaging and SNR calculations | Movement, electrode drying | Noise estimate by epoch |
| 5 | The calibration is traceable and current | Measurement validity | An expired or untraceable calibrator | Certificate audit |
| 6 | The drift is small relative to the decision margin | Threshold decisions | Aging, temperature | Periodic reference measurement |
| 7 | The device is used within its intended population | Claim validity | Off-label use | Check the intended-use statement |
| 8 | The firmware version is stable during the analysis | Pooling and reproducibility | An update mid-study | Version field in the data |
| 9 | The artifact rejection is not differential | Representative data | Rejection correlated with group | Rejection rate by group |
| 10 | The latency meets the decision requirement | Alarm utility | Buffering, network | End-to-end timing test |
| 11 | The electrical environment matches the test environment | Safety and EMC | Electrosurgery, MRI, home use | Site-specific assessment |
| 12 | The device's output is a measurement, not a diagnosis | Interpretation | Treating an estimate as a diagnosis | Check the intended use and the claim |
| 13 | The post-market performance is stable | Continued validity | Component changes, new users | Post-market surveillance |
| 14 | The data provenance is complete | Reprocessing and pooling | Missing acquisition settings | Data dictionary audit |

The limits of this topic:

- It does not cover circuit design, transducer physics, or materials science in depth.
- It does not cover imaging devices (CT, MRI, ultrasound), which have their own physics and standards.
- It does not cover the full IEC 60601 series or the full FDA submission process.
- It does not validate any specific device. It provides the questions a validation must answer.
- It does not cover therapeutic devices such as infusion pumps in detail, only their data and safety
  aspects.

## 12. Exercises

### Exercise 1 — The signal chain (recall and application)

For each stage of the chain (sensor, analog front end, ADC, processor, output), name one failure mode
and one data signature that would reveal it.

**Deliverable.** A five-row table.

**Acceptance criteria.** Sensor: contact or drift, signature is a slow bias or a flat signal; AFE: noise
or saturation, signature is a raised noise floor or clipping; ADC: aliasing or quantisation, signature
is a false low frequency or a staircase; processor: algorithm assumption, signature is a systematic
error at a boundary; output: latency or display error, signature is a delay or a unit mismatch.

**Time.** 25 minutes.

### Exercise 2 — Choose a sampling rate (calculation)

An EEG feature requires content up to 70 Hz. Compute the Nyquist minimum, propose a design rate with a
justified margin, and state the anti-aliasing filter cutoff.

**Deliverable.** Two rates and a filter specification.

**Acceptance criteria.** Nyquist minimum 140 Hz; design rate at least 250 Hz (a ratio of 1.79 or more);
filter cutoff at or below 70 Hz with a stopband by 125 Hz; the justification references filter
roll-off, not only the Nyquist number.

**Time.** 20 minutes.

### Exercise 3 — Aliasing (calculation)

A 60 Hz interference signal is sampled at 50 Hz. Compute the alias frequency. Then compute the alias
for a 60 Hz signal sampled at 100 Hz and explain why the anti-aliasing filter is still required at
100 Hz.

**Deliverable.** Two alias frequencies and an explanation.

**Acceptance criteria.** At 50 Hz, alias = |60 − 50| = 10 Hz; at 100 Hz, alias = |60 − 100| = 40 Hz;
the explanation states that a 40 Hz alias can contaminate a physiological band and that the filter
must remove 60 Hz before sampling.

**Time.** 20 minutes.

### Exercise 4 — ADC resolution (calculation)

A 16-bit ADC has an input range of ±10 mV. Compute the LSB and the quantisation SNR. Compare with a
12-bit ADC over the same range.

**Deliverable.** Four values and a comparison.

**Acceptance criteria.** 16-bit LSB 0.305 µV, SNR 98.08 dB; 12-bit LSB 4.883 µV, SNR 74.00 dB; the
comparison states that 16 bits gives 16 times finer resolution and 24 dB more SNR, and that the benefit
is realised only if the noise floor is below the LSB.

**Time.** 25 minutes.

### Exercise 5 — Noise budget (calculation)

A sensor has a source resistance of 1 MΩ and a bandwidth of 100 Hz at 310 K. Compute the thermal noise.
Then compute the total noise if the amplifier contributes 3 µV and the electrode contributes 8 µV, and
state which component dominates.

**Deliverable.** Two values and a conclusion.

**Acceptance criteria.** Thermal noise 1.308 µV; total sqrt(1.308² + 3² + 8²) = 8.62 µV; the electrode
dominates; the conclusion states that reducing the amplifier noise would have little effect.

**Time.** 25 minutes.

### Exercise 6 — Calibration and drift (application)

A glucose meter has a 2% drift over its six-month calibration interval. The clinical threshold is
7.0 mmol/L. A patient's true value is 7.2 mmol/L. Compute the range of displayed values at the start
and end of the interval and state whether the drift can change the classification.

**Deliverable.** Two ranges and a conclusion.

**Acceptance criteria.** Start 7.2 ± 0.144 = 7.06 to 7.34; end 7.2 ± 0.288 = 6.91 to 7.49 if the drift
is one-sided and additive; the conclusion states that the lower end crosses 7.0 and that the drift can
change the classification.

**Time.** 25 minutes.

### Exercise 7 — Latency budget (design)

Design a latency budget for an alarm that must fire within 30 seconds of a physiological event. Include
sensor, filter, window, processing, transmission, and display. State which component dominates and what
you would change if the budget were exceeded.

**Deliverable.** A six-row budget and a recommendation.

**Acceptance criteria.** Total under 30 s; the window is identified as the dominant term; the
recommendation shortens the window or changes the algorithm, with the specificity trade-off stated.

**Time.** 30 minutes.

### Exercise 8 — Critique a data sheet (critique)

A device data sheet states: "24-bit resolution, 1,000 Hz sampling, ±5% accuracy, Bluetooth connectivity,
AI-powered insights." Identify the missing specifications and the claims that cannot be evaluated, and
list five questions you would ask the vendor.

**Deliverable.** A one-page critique with a question list.

**Acceptance criteria.** Missing: bandwidth, input range, noise, latency, calibration interval,
intended population, validation study; unevaluable: "AI-powered insights" without a defined output and
validation; questions cover bandwidth, noise, latency, calibration, population, and algorithm
validation.

**Time.** 40 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Stage | Failure mode | Data signature |
|---|---|---|
| Sensor | Poor contact or drift | Flat signal, slow bias, high impedance |
| Analog front end | Noise or saturation | Raised noise floor, clipped peaks |
| ADC | Aliasing or quantisation | False low frequency, staircase steps |
| Processor | Algorithm assumption violated | Systematic error at a boundary |
| Output | Latency or unit error | Delayed alarm, implausible value |

The signature is the practical diagnostic. A staircase pattern in a continuous signal is almost always
quantisation; a flat-topped waveform is almost always saturation; a slow bias is almost always drift or
calibration.

### Answer to Exercise 2

```text
Nyquist minimum = 2 x 70 Hz = 140 Hz

Design rate: 250 Hz
  Ratio to Nyquist = 250/140 = 1.79

Anti-aliasing filter: cutoff at or below 70 Hz, with the stopband
reaching adequate attenuation by 125 Hz (the Nyquist frequency of 250 Hz).
```

The ratio of 1.79 is adequate for a filter with a moderate roll-off. A sharper filter would allow a
lower rate; a gentler filter would require a higher one. The filter specification is part of the
sampling decision, not an afterthought.

### Answer to Exercise 3

```text
60 Hz sampled at 50 Hz:
  f_alias = |60 - 1 x 50| = 10 Hz

60 Hz sampled at 100 Hz:
  f_alias = |60 - 1 x 100| = 40 Hz
```

At 100 Hz, the 60 Hz interference aliases to 40 Hz, which is inside the EEG band and can be mistaken
for gamma activity. The anti-aliasing filter is required even when the sampling rate is above the
interference frequency, because the alias falls in the signal band. This is the most common
misunderstanding of the Nyquist criterion: it constrains the bandwidth, not the interference
frequency.

### Answer to Exercise 4

```text
16-bit, +/-10 mV (20 mV full scale)
  LSB = 20 mV / 65,536 = 0.3052 uV
  SNR = 6.02 x 16 + 1.76 = 96.32 + 1.76 = 98.08 dB

12-bit, +/-10 mV
  LSB = 20 mV / 4,096 = 4.883 uV
  SNR = 6.02 x 12 + 1.76 = 74.00 dB
```

The 16-bit converter has 16 times finer resolution and 24 dB more SNR. The benefit is realised only if
the analog noise floor is below the 12-bit LSB. If the electrode noise is 10 µV, the extra bits resolve
noise more finely, not signal.

### Answer to Exercise 5

```text
Thermal noise
  Vn = sqrt(4 x 1.380649e-23 x 310 x 1e6 x 100)
     = sqrt(1.711e-12)
     = 1.308e-6 V
     = 1.308 uV

Total
  u = sqrt(1.308^2 + 3^2 + 8^2)
    = sqrt(1.711 + 9 + 64)
    = sqrt(74.711)
    = 8.64 uV
```

The electrode contributes 8 µV of the 8.64 µV total. Reducing the amplifier noise from 3 µV to 1 µV
would change the total to sqrt(1.711 + 1 + 64) = 8.17 µV, a 5% improvement. Reducing the electrode
noise would be far more effective.

### Answer to Exercise 6

```text
True value 7.2 mmol/L, drift 2%

At the start of the interval (no drift):
  7.2 x 0.02 = 0.144
  range 7.06 to 7.34 mmol/L

At the end of the interval (2% one-sided):
  7.2 x 0.04 = 0.288 if the drift adds to the error
  range 6.91 to 7.49 mmol/L
```

The lower end of the end-of-interval range crosses the 7.0 mmol/L threshold. The drift can change the
classification for a patient whose true value is close to the threshold. The clinical response is to
shorten the calibration interval or to use a device with a smaller drift specification for
threshold-based decisions.

### Answer to Exercise 7

| Component | Budget |
|---|---:|
| Sensor response | 0.5 s |
| Anti-aliasing filter group delay | 0.1 s |
| Algorithm window | 12.0 s |
| Processing | 0.3 s |
| Transmission | 2.0 s |
| Display | 0.2 s |
| **Total** | **15.1 s** |

The window dominates. If the budget were exceeded, shortening the window from 12 s to 6 s would save 6 s
and reduce the total to about 9 s, at the cost of more false alarms because the algorithm would have
less evidence. The trade-off must be measured, not assumed.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Missing specifications | 30% | Bandwidth, input range, noise, latency, calibration interval, population |
| Unevaluable claims | 25% | "AI-powered insights" without a defined output or validation |
| Question quality | 30% | Specific, answerable questions about the six missing items |
| Intended-use framing | 15% | The claim is bounded by the population and the setting |

A weak critique says the data sheet is "vague" without naming the specifications that are missing or
the questions that would resolve them.

## 14. Interview questions

### Question 1 — "What is the Nyquist criterion and why does it matter?"

**Strong answer.** "A signal must be sampled at more than twice its highest frequency to be
reconstructable. Sampling below that folds high frequencies into low ones, and the folding cannot be
undone by post-hoc filtering. The practical consequence is that an anti-aliasing filter must precede
the ADC."

**Weak answer.** "Sample at twice the frequency." The criterion is about bandwidth, and the filter
placement is the operational part.

### Question 2 — "A device samples at 1,000 Hz. Is that enough for an ECG?"

**Strong answer.** "It depends on the bandwidth and the feature. For diagnostic ECG with a 150 Hz
bandwidth, 1,000 Hz gives a ratio of 3.3 to the Nyquist minimum, which is ample. For a 40 Hz monitoring
bandwidth, it is more than enough. Sampling rate alone does not establish adequacy; the bandwidth and
the anti-aliasing filter do."

**Weak answer.** "Yes, 1,000 Hz is very fast." The number is meaningless without the bandwidth it is
meant to support.

### Question 3 — "What is the difference between resolution and accuracy?"

**Strong answer.** "Resolution is the smallest distinguishable change, set by the ADC and the noise
floor. Accuracy is closeness to the true value, set by calibration and bias. A device can have 24-bit
resolution and 5% accuracy. Resolution sets the detection floor; accuracy sets the bias."

**Weak answer.** "They are both about precision." Resolution, accuracy, and precision are three
different properties.

### Question 4 — "Why can a 24-bit ADC be worse than a 12-bit ADC?"

**Strong answer.** "It is not worse, but the extra bits may be useless. If the electrode noise is 10 µV
and the 12-bit LSB is 2.4 µV, the measurement is sensor-limited. A 24-bit converter resolves the noise
more finely but does not improve the signal. It also costs power and money, which matters for a
wearable."

**Weak answer.** "More bits are always better." Bits beyond the noise floor are not information.

### Question 5 — "What is aliasing and why can't you filter it out afterwards?"

**Strong answer.** "Aliasing is the folding of frequencies above the Nyquist frequency into the band
below it. After sampling, the aliased component is indistinguishable from a genuine low-frequency
signal, so no filter can separate them. The only fix is to remove the high frequencies before the
ADC."

**Weak answer.** "It is noise that a low-pass filter can remove." The aliased component is inside the
passband, so a low-pass filter removes the signal with it.

### Question 6 — "How do you specify latency for an alarm?"

**Strong answer.** "End to end: sensor response, filter delay, algorithm window, processing,
transmission, and display. The window usually dominates. The budget must be less than the clinical
tolerance, and it must be measured in the deployment environment, not estimated from the processor
speed."

**Weak answer.** "The processor is fast, so latency is low." The processor is rarely the bottleneck.

### Question 7 — "What does IEC 62304 require?"

**Strong answer.** "A software life cycle: planning, requirements, architecture, detailed design,
implementation, integration, verification, and maintenance. It assigns safety classes A, B, and C based
on the possible harm from a software failure, and it requires traceability from requirements to tests.
A model embedded in a device is software under the standard."

**Weak answer.** "It is the medical device software standard." The life cycle and the safety
classification are the substance.

### Question 8 — "What is the difference between verification and validation?"

**Strong answer.** "Verification asks whether the device meets its specification; validation asks
whether it meets the clinical need in the intended use environment. A device can pass every
verification test and fail validation because the bench differs from the ward."

**Weak answer.** "They are the same thing with different names." They answer different questions and
require different evidence.

### Question 9 — "Why is alarm fatigue an engineering problem?"

**Strong answer.** "Because the alarm rate is set by the threshold and the persistence rule, which are
design choices. A threshold chosen for maximum sensitivity produces a high false-alarm rate, staff
silence the alarm, and a true event is missed. The fix is a measured false-alarm rate, a persistence
rule, and a priority hierarchy."

**Weak answer.** "Because staff ignore alarms." That describes the behaviour, not the cause; the cause
is the alarm design.

### Question 10 — "What acquisition metadata would you require with a device dataset?"

**Strong answer.** "Device model, firmware version, sampling rate, bandwidth and filter settings, gain,
input range, calibration date, sensor lot, and the intended-use population. Without these, the data
cannot be pooled across devices, reprocessed after a firmware change, or interpreted against a
specification."

**Weak answer.** "The device model and the date." That is not enough to interpret the signal or to
detect a processing change.

### Question 11 — "How would you handle a firmware update in the middle of a study?"

**Strong answer.** "Treat it as a data change. Record the version, compare the signal characteristics
before and after, and revalidate any model that uses the data. If the signal processing changed, the
model's input distribution changed, and the performance estimate from the pre-update period may not
hold."

**Weak answer.** "Note it in the limitations." A note is not a control; the comparison and the
revalidation are.

### Question 12 — "Why does pulse oximetry fail in some patients?"

**Strong answer.** "The device estimates arterial saturation from an optical signal using a calibration
derived in a validation population. Bias varies with skin pigmentation, perfusion, haematocrit, motion,
and ambient light. A device can meet its specification on average and be biased in a subgroup, which is
a validation-population failure as much as a sensor failure."

**Weak answer.** "Because the sensor is inaccurate." It is accurate enough on average; the problem is
the subgroup and the conditions.

## 15. Key terms

- **ADC** — analog-to-digital converter.
- **Aliasing** — folding of high frequencies into low ones.
- **Alarm fatigue** — desensitisation from excessive alarms.
- **Analog front end (AFE)** — amplification and filtering before the ADC.
- **Anti-aliasing filter** — a low-pass filter before the ADC.
- **Bandwidth** — the range of frequencies a system passes.
- **Baseline wander** — low-frequency drift from respiration or movement.
- **Biocompatibility** — suitability of a material for body contact.
- **Calibration** — establishing the response-reference relationship.
- **CMRR** — common-mode rejection ratio.
- **Cutoff frequency** — the frequency at which response falls by a stated amount.
- **Cybersecurity** — protection of a connected device from unauthorized access.
- **Data provenance** — the acquisition settings recorded with the data.
- **Decimation** — reducing the sample rate after filtering.
- **Defibrillation protection** — ability to withstand a defibrillation pulse.
- **Drift** — change in output over time at a fixed input.
- **Duty cycle** — fraction of time a system is active.
- **Dynamic range** — range from the noise floor to full scale.
- **EMC** — electromagnetic compatibility.
- **Flicker noise** — 1/f noise at low frequencies.
- **Gain** — ratio of output to input.
- **Human factors** — the study of use error and usability.
- **Hysteresis** — direction-dependent response.
- **IEC 60601-1** — general requirements for medical electrical equipment.
- **IEC 62304** — medical device software life cycle standard.
- **IEC 62366-1** — usability engineering standard.
- **Input impedance** — resistance to current flow at the input.
- **Input range** — the maximum and minimum measurable input.
- **Intended use** — the stated clinical purpose of a device.
- **Isolation** — electrical separation between circuits.
- **ISO 14971** — medical device risk management standard.
- **Jitter** — variation in sample timing.
- **Latency** — delay from physiological event to output.
- **Leakage current** — current flowing to earth through the patient.
- **Linearity** — proportionality of output to input.
- **LSB** — least significant bit, the quantisation step.
- **Motion artifact** — signal corruption from movement.
- **Noise floor** — the noise level in the absence of signal.
- **Notch filter** — a narrow band-stop filter.
- **Nyquist criterion** — sample at more than twice the highest frequency.
- **Nyquist frequency** — half the sampling rate.
- **Oversampling** — sampling well above the Nyquist rate.
- **Powerline interference** — 50 or 60 Hz pickup.
- **Quantisation noise** — error from rounding to discrete levels.
- **Resolution** — the smallest change the system can distinguish.
- **SaMD** — software as a medical device.
- **Sampling rate** — samples per second.
- **Saturation** — clipping when the signal exceeds the input range.
- **Sensor** — the element that responds to the measurand.
- **Shot noise** — noise from discrete charge carriers.
- **Signal-to-noise ratio (SNR)** — signal power divided by noise power.
- **Slew rate** — maximum rate of change of the output.
- **Sterility assurance level** — probability of a non-sterile unit.
- **Thermal noise** — Johnson noise from resistance.
- **Transducer** — a device that converts one form of energy to another.
- **Use error** — an error arising from the design of the use environment.
- **Verification** — confirming a specification is met.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Device standards

- IEC 60601-1:2005+AMD1:2012+AMD2:2020 CSV, *Medical electrical equipment — Part 1: General
  requirements for basic safety and essential performance*. https://webstore.iec.ch/en/publication/67497
- IEC 60601-1-2:2014+AMD1:2020, *Electromagnetic disturbances — Requirements and tests*.
  https://webstore.iec.ch/
- IEC 60601-1-8:2006+AMD1:2012+AMD2:2020, *General requirements, tests and guidance for alarm systems*.
  https://webstore.iec.ch/
- IEC 62304:2006+AMD1:2015, *Medical device software — Software life cycle processes*.
  https://webstore.iec.ch/
- ISO 14971:2019, *Medical devices — Application of risk management to medical devices*.
  https://www.iso.org/standard/72704.html
- IEC 62366-1:2015+AMD1:2020, *Application of usability engineering to medical devices*.
  https://webstore.iec.ch/
- ISO 10993-1:2018, *Biological evaluation of medical devices — Part 1: Evaluation and testing within a
  risk management process*. https://www.iso.org/standards.html
- ISO 80601-2-61:2017, *Medical electrical equipment — Particular requirements for pulse oximeter
  equipment*. https://www.iso.org/standards.html
- ANSI/AAMI EC11:1991/(R)2001, *Diagnostic electrocardiographic devices*.
  https://array.aami.org/
- ANSI/AAMI EC57:2012/(R)2020, *Testing and reporting performance results of cardiac rhythm and
  ST segment measurement algorithms*. https://array.aami.org/

### Regulatory guidance

- US Food and Drug Administration, *Diagnostic ECG Guidance (Including Non-Alarming ST Segment
  Measurement)*, 1998.
  https://www.fda.gov/files/medical%20devices/published/Guidance-for-Industry--Diagnostic-ECG-Guidance-(Including-Non-Alarming-ST-Segment-Measurement)-(PDF-Version).PDF
- US Food and Drug Administration, *White Paper: Infusion Pump Improvement Initiative*, 2010.
  https://www.fda.gov/medical-devices/infusion-pumps/white-paper-infusion-pump-improvement-initiative
- US Food and Drug Administration, *Safety Communication: Pulse Oximeter Accuracy and Limitations*, 2021.
  https://www.fda.gov/medical-devices/safety-communications/pulse-oximeter-accuracy-and-limitations-fda-safety-communication
- Regulation (EU) 2017/745 on medical devices (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
- 21 CFR Part 820, *Quality System Regulation*. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820
- International Medical Device Regulators Forum, *Software as a Medical Device: Possible Framework for
  Risk Categorization*. https://www.imdrf.org/

### Clinical evidence and guidance

- Kligfield, P. et al., "Recommendations for the Standardization and Interpretation of the
  Electrocardiogram: Part I", *Journal of the American College of Cardiology* 49(10), 1109–1127, 2007.
  https://doi.org/10.1016/j.jacc.2007.01.024
- Bailey, J. J. et al., "Recommendations for standardization and specifications in automated
  electrocardiography: bandwidth and digital signal processing", *Circulation* 81(2), 730–739, 1990.
  https://doi.org/10.1161/01.CIR.81.2.730
- Sjoding, M. W. et al., "Racial Bias in Pulse Oximetry Measurement", *New England Journal of Medicine*
  383, 2477–2478, 2020. https://doi.org/10.1056/NEJMc2029240
- The Joint Commission, *National Patient Safety Goal on Alarm Management*.
  https://www.jointcommission.org/

> **Note:** IEC, ISO, and AAMI documents are sold standards; the URLs are product or catalogue pages.
> Summaries here are from public front matter and secondary descriptions and should be checked against
> the purchased standard before use in a regulatory submission.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`07_epidemiology-basics.md`](07_epidemiology-basics.md)
- Next topic: [`09_computing-basics.md`](09_computing-basics.md)
- Units and uncertainty: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Dimensional checks of composite units: [`04_dimensional-analysis.md`](04_dimensional-analysis.md)
- Physiological signals and timescales: [`05_biological-basics.md`](05_biological-basics.md)
- Assumption recording: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `11_ecg_quality.ipynb` covers ECG signal quality and
  `14B_quantization.ipynb` covers resolution and quantisation.
