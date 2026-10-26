# Sensor and Signal Labs

> Module 24 — Hardware and Robotics Labs
> Topic 4 of 9
> Estimated study time: 5 hours
> Prerequisites: 01, 02, 03, 11
> Version 2026.3

## 1. Why this topic exists

A sensor does not measure a clinical quantity. It measures a physical quantity that is related to the clinical quantity by a chain of assumptions. A thermistor measures resistance. A pulse oximeter measures the ratio of two optical absorptions. An ECG electrode measures a potential difference on the skin. The clinical quantity — core temperature, arterial oxygen saturation, cardiac electrical activity — is inferred.

The industry problem is that the inference is invisible in the data. A temperature reading of 37.2 °C looks like a temperature. It is a resistance, converted through a calibration curve, corrected for a reference, and rounded to a display. A heart rate of 72 beats per minute looks like a heart rate. It is a peak-detection algorithm applied to a noisy optical signal. A bench measurement can validate the conversion; it cannot validate the clinical inference.

Three decisions depend on this topic:

1. **What the sensor actually measures, and what the conversion assumes.** The conversion from resistance to temperature assumes a calibration curve, a reference junction, and a self-heating model. The conversion from optical ratio to saturation assumes a calibration curve derived from human blood samples. The second assumption cannot be established on a bench.
2. **Which error dominates the measurement.** The error budget includes the sensor, the reference, the analog front end, the ADC, the algorithm, and the environment. The dominant term determines what to fix.
3. **What the lab result supports.** A calibration against a reference supports a statement about that sensor, in that range, against that reference. It does not support a clinical claim, a regulatory claim, or a claim about a different sensor lot.

The commercial stakes are concrete. A temperature sensor that is calibrated against an uncalibrated reference produces a certificate that is worthless. A PPG algorithm that is validated on one skin tone and deployed on another can produce a clinically significant bias. An ECG front end that meets its noise specification but fails common-mode rejection can be unusable in an electrically noisy room.

This topic covers three sensor labs: the ECG front end, the optical pulse sensor, and the temperature sensor. It uses the same measurement discipline for all three, and it is explicit about which parts of the chain a bench can establish.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work. No lab in this module places electrodes or optical sensors on a person; use a resistor network, a signal generator, a commercial simulator, or a recorded waveform.

## 2. Learning outcomes

By the end of this topic you can:

- Draw the chain from a clinical quantity to a stored sample and name the assumption at each step.
- Measure the common-mode rejection ratio (CMRR) of an ECG front end with a defined source and a stated uncertainty.
- Measure the input-referred noise of a front end over a stated bandwidth and separate it from the quantization noise.
- Calibrate a temperature sensor at five points, ascending and descending, and report the fit, the residuals, and the hysteresis.
- Build an uncertainty budget for a sensor calibration with at least five sources and report the expanded uncertainty at k = 2.
- Compute a crossed Gauge R&R from variance components and decide whether the measurement system is fit for a stated tolerance.
- Explain why a bench PPG measurement cannot establish SpO2 accuracy, and what evidence would.
- State the limits of a sensor calibration in terms of range, reference, lot, and environment.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A sensor lab is a chain of conversions, and every conversion has an assumption. The assumptions are not defects; they are the conditions under which the number means something.

### 3.1 The conversion chain

```text
 CLINICAL        PHYSICAL        ELECTRICAL       DIGITAL         REPORTED
 core temp  -->  resistance -->  voltage     -->  code       -->  temperature
 SpO2       -->  optical ratio -> photocurrent -> code       -->  percentage
 ECG        -->  skin potential -> differential -> code       -->  millivolts
      |              |               |               |               |
  definition     sensor law      front end        ADC             algorithm
  site           calibration     gain, noise      resolution      filtering
  population     drift           CMRR, offset     sample rate     threshold
```

Each arrow is a conversion with a model. The model is validated against a reference at the arrow where a reference exists. For temperature, the reference is a calibrated thermometer. For ECG, the reference is a signal generator with a known amplitude. For SpO2, the reference is a blood-gas measurement in a clinical study, which a bench does not have.

The chain is the reason a bench result can be strong at one arrow and silent at another. A bench can validate the electrical arrow. It can validate the digital arrow. It can validate the sensor law at a few points. It cannot validate the clinical definition of the quantity.

### 3.2 The error stack

The error stack has layers, and the layers are not independent.

| Layer | Typical error source | Typical magnitude | How it is measured |
|---|---|---|---|
| Clinical definition | Site, population, definition | Large, often unknown | Clinical study |
| Sensor | Calibration, drift, nonlinearity | 0.1–2% of range | Calibration against a reference |
| Front end | Offset, gain, noise, CMRR | µV to mV | Bench measurement |
| ADC | Quantization, reference, nonlinearity | LSB to a few LSB | Code-domain measurement |
| Algorithm | Threshold, window, latency | Event-dependent | Annotated dataset |
| Environment | Temperature, humidity, motion | Variable | Controlled or logged |

The stack is not additive in a simple way. A front-end offset can be removed by calibration; a sensor nonlinearity cannot be removed by a gain correction. A clinical definition error cannot be removed by any amount of signal processing.

### 3.3 Calibration, verification, and validation

Three words are often used interchangeably. They are not.

| Activity | Question | Reference | Example |
|---|---|---|---|
| Calibration | What is the relationship between the sensor output and the reference? | A calibrated reference | Fit resistance to temperature |
| Verification | Does the output meet the specification? | A requirement | Is the error within ±0.1 °C? |
| Validation | Does the device meet the clinical need? | Clinical evidence | Does the thermometer agree with core temperature in patients? |

A calibration produces a curve. A verification compares the curve's error with a requirement. A validation compares the device's performance with a clinical need. A bench lab can do the first two and cannot do the third.

### 3.4 The reference hierarchy

A calibration is only as good as its reference. The reference hierarchy runs from the International System of Units (SI) through national metrology institutes, accredited calibration laboratories, working references, and the sensor under test.

```text
SI definition
   |
national metrology institute (NIST, PTB, NPL, ...)
   |
accredited calibration laboratory (ISO/IEC 17025)
   |
working reference (thermometer, signal generator, pressure gauge)
   |
sensor under test
```

Each step contributes uncertainty. A working reference with a certificate from an accredited laboratory has a stated uncertainty and a traceability statement. A reference without a certificate is an indicator, not a reference.

The ice bath is a special case. The ice-point cell is a physical reference that can be realised in a lab, but only if it is prepared correctly: crushed ice, distilled water, good contact, stirring, and enough time to equilibrate. A cup of ice water that has just been filled is not at 0 °C.

### 3.5 The three sensor labs

The module has three sensor labs, and each one exercises a different part of the chain.

| Lab | Quantity | Reference | What the bench can establish | What it cannot |
|---|---|---|---|---|
| ECG front end | Differential voltage | Signal generator, resistor network | Gain, bandwidth, CMRR, noise, offset | Clinical ECG quality, arrhythmia detection |
| PPG | Optical absorption ratio | Pulse simulator or recorded waveform | Timing, heart-rate tracking, motion response | SpO2 accuracy, skin-tone bias |
| Temperature | Temperature | Calibrated thermometer, ice bath, water bath | Calibration curve, hysteresis, uncertainty | Core temperature agreement in patients |

The three labs share the same measurement discipline: a defined input, a reference, a repeated measurement, an uncertainty budget, and a stated limit.

### 3.6 The signal-to-noise ratio is not the whole story

A high signal-to-noise ratio (SNR) does not guarantee a usable signal. A 50 Hz interference that is 40 dB below the signal can still be visible and can still confuse a detector if it is coherent. A baseline wander that is 10 dB below the signal can saturate an amplifier after a gain stage. A motion artefact can be larger than the signal for a second and can dominate an automatic gain control.

The error stack must include the interference and the artefacts, not only the random noise. A noise budget that counts only the thermal noise is incomplete.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Sensor | Device that converts a physical quantity into a signal | Used for the whole measurement system | None |
| Transducer | Device that converts one form of energy into another | Confused with a sensor | None |
| Sensitivity | Change in output per change in input | Confused with resolution | V per unit, or codes per unit |
| Offset | Output at zero input | Confused with drift | V or °C |
| Gain | Ratio of output change to input change | Confused with sensitivity | Dimensionless |
| Linearity | Closeness of the response to a straight line | Confused with accuracy | % of full scale |
| Hysteresis | Difference between ascending and descending responses | Ignored in calibration | Unit of the measurand |
| Repeatability | Closeness of repeated measurements under one setup | Confused with reproducibility | Unit of the measurand |
| Reproducibility | Closeness of measurements across setups or operators | Confused with repeatability | Unit of the measurand |
| Resolution | Smallest distinguishable increment | Confused with accuracy | Unit of the measurand |
| Accuracy | Closeness to the true value | Confused with precision | Unit of the measurand |
| Drift | Change in output over time at constant input | Confused with noise | Unit per time |
| Self-heating | Temperature rise caused by the sensor's own excitation | Ignored in thermometry | °C or K |
| Immersion depth | Depth of a probe in the medium | Ignored in thermometry | mm |
| Stem conduction | Heat flow along a probe from the tip to the environment | Ignored in thermometry | °C |
| Ice point | Equilibrium temperature of ice and water at 0 °C | Assumed without preparation | °C |
| Triple point | Temperature at which ice, water, and vapour coexist | Confused with the ice point | 0.01 °C |
| CMRR | Common-mode rejection ratio | Quoted at DC only | dB |
| Input-referred noise | Noise divided by the gain, expressed at the input | Confused with output noise | V RMS |
| Noise density | Noise per square root of bandwidth | Confused with total noise | V/√Hz |
| Bandwidth | Frequency range over which the signal is passed | Confused with sample rate | Hz |
| Anti-alias filter | Filter that limits the bandwidth before sampling | Omitted | Hz |
| Right leg drive | Circuit that drives the body's common-mode potential | Confused with a ground | V |
| Electrode offset | Half-cell potential at an electrode | Ignored in DC-coupled designs | mV |
| Polarisation | Electrode behaviour after a current or a defibrillation | Ignored in recovery design | s |
| PPG | Photoplethysmography; optical measurement of blood volume change | Used as a synonym for SpO2 | None |
| SpO2 | Arterial oxygen saturation estimated by pulse oximetry | Confused with SaO2 | % |
| SaO2 | Arterial oxygen saturation measured by blood gas analysis | Confused with SpO2 | % |
| Red/IR ratio | Ratio of the pulsatile to the non-pulsatile component at two wavelengths | Confused with saturation | Dimensionless |
| Calibration curve | Relationship between the sensor signal and the reference | Confused with a correction factor | Function |
| ARMS | Root-mean-square accuracy of an SpO2 measurement | Confused with bias | % SpO2 |
| Bias | Mean difference between two methods | Confused with limits of agreement | Unit |
| Limits of agreement | Interval containing most differences between two methods | Confused with a confidence interval | Unit |
| Gauge R&R | Repeatability and reproducibility study of a measurement system | Confused with a calibration | % |
| Repeatability and reproducibility | Variance components in a Gauge R&R | Used loosely | Variance |
| ndc | Number of distinct categories | Confused with resolution | Count |
| Tolerance | Allowed variation of the product | Confused with the measurement uncertainty | Unit of the measurand |
| Fit residual | Difference between a measurement and the fitted curve | Confused with error | Unit of the measurand |
| Uncertainty budget | List of uncertainty sources and their combination | A single percentage | Table |
| Coverage factor | Multiplier for the expanded uncertainty | Assumed to be 2 | Dimensionless |
| Traceability | Unbroken chain of calibrations to a reference | A sticker | None |
| Reference | A measurement standard with a stated uncertainty | Any convenient instrument | None |
| NIST | US National Institute of Standards and Technology | Used as a generic word for "calibrated" | Organisation |
| ISO/IEC 17025 | Competence standard for calibration laboratories | Confused with a product certification | Standard |
| AFE | Analog front end | Used for the whole device | None |
| ADC | Analog-to-digital converter | Called a sensor | None |
| LSB | Least significant bit | Confused with resolution | V or code |
| ENOB | Effective number of bits | Quoted without a bandwidth | bits |
| Common-mode voltage | Voltage common to both inputs | Confused with the differential signal | V |
| Differential voltage | Difference between two inputs | Confused with the common-mode voltage | V |
| CMRR test | Measurement of the differential output produced by a common-mode input | Performed with an unbalanced source | dB |
| Lead-off | Condition in which an electrode is disconnected | Ignored in the algorithm | State |
| Motion artefact | Signal disturbance caused by movement | Treated as noise | Unit of the measurand |
| Saturation | Condition in which the output reaches a limit | Confused with high signal | State |
| Pulse simulator | Device that produces a defined optical or electrical pulse | Assumed to be a patient | None |

## 5. The mechanics, step by step

The procedure below runs a sensor lab from setup to report. It has a common spine (Steps 1–7 and 13–15) and three branches (A: ECG, B: PPG, C: temperature). Each step states its input, output, and rationale. The branch steps are written so that a team can run one branch in a three-hour session.

### Step 1. Define the quantity and the claim

**Input:** the clinical or engineering question.

**Output:** the physical quantity, the unit, the range, and the claim boundary.

**Rationale:** a sensor lab that begins with the sensor rather than the quantity will measure whatever the sensor happens to produce. The quantity determines the reference, the range, and the acceptance criterion.

Examples:

| Question | Quantity | Range | Reference | Claim boundary |
|---|---|---|---|---|
| Is the ECG front end suitable for a wearable? | Differential voltage | 0.1–5 mV, 0.5–40 Hz | Signal generator | Electrical performance only |
| Does the PPG track heart rate during motion? | Pulse timing | 40–180 bpm | Recorded waveform | Timing only, no SpO2 |
| Is the temperature sensor within its datasheet? | Temperature | 0–50 °C | Calibrated thermometer | Sensor only, no clinical agreement |

The claim boundary is written before the data. A lab that intends to claim SpO2 accuracy has already made an error, because the bench has no blood-gas reference.

### Step 2. Select the reference and verify its traceability

**Input:** the quantity and the range.

**Output:** a reference with a certificate, a stated uncertainty, and a calibration date.

**Rationale:** the reference sets the accuracy of the calibration. A reference that is five times more accurate than the sensor is adequate; a reference that is equal to the sensor is not.

| Quantity | Reference option | Typical uncertainty | Traceability |
|---|---|---|---|
| Temperature | Calibrated thermometer | 0.05 °C | ISO/IEC 17025 certificate |
| Temperature | Ice bath | 0.01–0.05 °C if prepared correctly | Physical fixed point |
| Voltage | Calibrated DMM | 0.01% | ISO/IEC 17025 certificate |
| Voltage | Signal generator with a calibrated attenuator | 0.5–1% | Certificate or loopback check |
| Optical power | Reference photodiode | 1–5% | Certificate |
| Force | Calibrated load cell or known mass | 0.1–1% | Certificate or local gravity |

The reference's uncertainty must be smaller than the sensor's tolerance, ideally by a factor of four or more. If it is not, the calibration cannot decide conformity.

### Step 3. Characterise the sensor before the system

**Input:** the sensor and its datasheet.

**Output:** a table of the sensor's specified behaviour and the conditions under which the specification applies.

**Rationale:** the datasheet's accuracy is conditional. A sensor specified to ±0.1 °C over −20 to 50 °C may be specified to ±0.2 °C over a wider range. A sensor specified at 3 V may behave differently at 1.8 V. The conditions are part of the specification.

The table contains:

- The measurement range.
- The accuracy at each range.
- The resolution.
- The supply voltage and current.
- The output type and interface.
- The response time.
- The operating temperature.
- The package and the thermal path.
- The calibration status from the factory.

A sensor that is factory-calibrated still needs a system calibration, because the system adds an offset, a gain error, and a thermal gradient.

### Step 4. Build the signal path and verify its topology

**Input:** the schematic, the sensor, and the ADC.

**Output:** an assembled path with a documented ground and reference.

**Rationale:** the topology determines the errors. A single-ended path rejects less common-mode noise than a differential path. A DC-coupled path amplifies the electrode offset; an AC-coupled path removes it but adds a high-pass corner. A path with a shared ground return has a common impedance that couples noise.

The topology record includes:

| Element | Choice | Consequence |
|---|---|---|
| Input configuration | Differential | Rejects common-mode noise |
| Coupling | AC with a 0.5 Hz corner | Removes electrode offset, adds a settling time |
| Gain | 100 | Raises the signal above the ADC noise |
| Anti-alias filter | 2nd-order, 40 Hz | Limits the noise bandwidth |
| Reference | Mid-rail, buffered | Sets the output common mode |
| Ground | Single-point at the ADC | Avoids a ground loop |
| Shielding | Driven shield or guard | Reduces cable pickup |

The topology is reviewed against the schematic before power. A component that is placed but not connected is a topology error that a photograph will not reveal.

### Step 5. Measure the quiescent operating point

**Input:** the powered path with a zero differential input.

**Output:** the output offset, the supply current, and the temperature of the active components.

**Rationale:** the quiescent point tells you whether the path is operating in its linear region. An output that is pinned at a rail is a path that cannot amplify.

Measure:

- The output voltage at zero input.
- The supply voltage at the device pins, not at the supply terminals.
- The supply current.
- The temperature of the regulator and the amplifier after five minutes.
- The reference voltage at the ADC pin.

Acceptance criteria: the output is within the linear range, the supply is within the device's specification, the current is within the expected range, and no component exceeds its rated temperature.

### Step 6. Calibrate the path end to end

**Input:** a known input at several levels.

**Output:** a gain, an offset, and a residual, with uncertainty.

**Rationale:** the path's gain is not the amplifier's datasheet gain. It includes the source, the cable, the input network, and the ADC reference. The end-to-end calibration is the only way to know the combined scale.

Procedure:

1. Apply a known differential voltage at the input, measured with a calibrated DMM.
2. Record the ADC code, averaged over at least 100 samples.
3. Repeat at five levels across the range, ascending and descending.
4. Fit code = a × input + b.
5. Report the slope, the intercept, and the residual standard deviation.
6. Convert the residual to input-referred units.
7. Check the fit at an unused level.

For a differential input, the source must be balanced. An unbalanced source converts common-mode voltage into a differential error, which corrupts the gain measurement.

### Step 7. Measure the noise floor in the stated band

**Input:** the path with the input terminated in the source impedance.

**Output:** the noise standard deviation in codes and in input-referred units, with the bandwidth.

**Rationale:** the noise floor sets the smallest signal the path can resolve. It depends on the bandwidth, the gain, the source impedance, and the sample rate.

Procedure:

1. Terminate the input with the source impedance, or short it if the source is low impedance.
2. Capture at least 10,000 samples at the working rate.
3. Remove the mean.
4. Filter to the working band.
5. Compute the standard deviation.
6. Compute the noise density if the noise is not white.
7. Report the bandwidth and the filter order.

A noise number without a bandwidth is not a specification. A noise number measured with the input open includes the cable pickup; a noise number measured with the input shorted excludes the source noise. State which one was measured.

### Step 8A. Branch A: measure the ECG front end

**Input:** the assembled ECG path and a signal generator.

**Output:** gain, bandwidth, CMRR, input-referred noise, and offset, with uncertainties.

**Rationale:** the ECG front end is a differential amplifier with a high gain and a high common-mode rejection. The four measurements that matter are the ones that determine whether a small differential signal can be recovered in the presence of a large common-mode interferer.

#### A1. Gain and frequency response

Apply a 1 mV peak-to-peak sine at 10 Hz. Measure the output amplitude with the scope. Compute the gain. Repeat at 1, 10, 40, 100, and 200 Hz. Plot the response and find the −3 dB point.

Acceptance: the gain is within 5% of the design value; the −3 dB point is at or above the design cutoff; the response is monotonic in the passband.

#### A2. Common-mode rejection ratio

The CMRR is the ratio of the differential gain to the common-mode gain:

\[
\text{CMRR} = 20 \log_{10}\left(\frac{A_d}{A_{cm}}\right)
\]

where \(A_d\) is the differential gain and \(A_{cm}\) is the common-mode gain. To measure it, apply a common-mode sine to both inputs, measure the output, and divide by the differential gain.

Procedure:

1. Connect both inputs to the same source through matched resistors.
2. Set the source to 1 V RMS at 50 Hz.
3. Measure the output RMS.
4. Compute the input-referred common-mode signal as \(V_{out} / A_d\).
5. Compute the CMRR.
6. Repeat at 60 Hz and at 1 kHz.
7. Check the source balance with a DMM before the measurement.

The source balance is the critical point. A 1% mismatch in the source resistors converts 1 V of common-mode into 10 mV of differential, which is larger than the amplifier's own common-mode error. The measured CMRR is then the source's CMRR, not the amplifier's.

#### A3. Input-referred noise

Short both inputs to the mid-rail through the source impedance. Capture 10,000 samples at 1 kS/s. Filter to 0.5–40 Hz. Compute the standard deviation in codes. Divide by the gain and by the LSB to get the input-referred noise.

Compare with the datasheet's typical value, stating the bandwidth. The AD8232 datasheet gives 14 µV peak-to-peak from 0.5 to 40 Hz, which is about 2.3 µV RMS for a Gaussian signal with a crest factor of about 6. The comparison must use the same bandwidth and the same crest-factor convention.

#### A4. Offset and drift

Measure the output at zero input for 10 minutes. Compute the mean and the drift. Compare with the datasheet's offset and drift. A large offset can saturate the gain stage; a large drift can be mistaken for a slow physiological signal.

### Step 8B. Branch B: measure the PPG path

**Input:** the optical sensor, an LED driver, and a pulse simulator or a recorded waveform.

**Output:** a heart-rate estimate, a timing error, and a motion response, with uncertainties.

**Rationale:** the PPG signal is a small pulsatile component superimposed on a large non-pulsatile component. The ratio of the two, at two wavelengths, is the basis of SpO2. A bench can measure the timing and the ratio; it cannot convert the ratio to saturation without a clinical calibration.

#### B1. LED and photodiode operating point

Measure the LED current with a series resistor or a current probe. Measure the photodiode signal with the ADC. Verify that the signal is within the ADC range and that the LED is not saturating the photodiode.

The MAX30102 integrates the LED drivers and the ADC. Its LED current is programmable from about 0 to 50 mA, and its ADC resolution is programmable from 15 to 18 bits depending on the pulse width. The settings must be recorded, because they change the noise and the power.

#### B2. Heart-rate tracking

Drive the sensor with a pulse simulator at 60, 90, and 120 beats per minute. Record the raw signal and the algorithm's output. Compute the error in beats per minute and the latency from the simulated pulse to the output.

A pulse simulator is not a patient. It produces a defined optical or electrical waveform, and it tests the timing and the algorithm. It does not test the sensor's response to skin, perfusion, or motion.

#### B3. Motion artefact

Add a mechanical disturbance to the sensor or modulate the optical path. Record the signal and the algorithm's response. Report the recovery time and the false-detection rate.

Motion artefact is often larger than the pulsatile signal. An algorithm that has not been tested with motion is an algorithm that will fail in a wearable.

#### B4. What the bench cannot do

The bench cannot establish SpO2 accuracy. The conversion from the red/infrared ratio to saturation requires a calibration curve derived from human subjects with arterial blood-gas references, under a controlled desaturation protocol. A bench measurement can check the ratio's repeatability and the sensor's noise; it cannot check the curve.

### Step 8C. Branch C: calibrate the temperature sensor

**Input:** the sensor, a calibrated reference, an ice bath, and a water bath.

**Output:** a calibration curve, residuals, hysteresis, and an uncertainty budget.

**Rationale:** temperature is the simplest sensor to calibrate and the easiest to calibrate wrongly. The reference, the immersion, the equilibration, and the self-heating all contribute errors that are larger than the sensor's resolution.

#### C1. Prepare the reference points

Use five points across the range: 0 °C in an ice bath, and 10, 25, 37, and 50 °C in a stirred water bath. Keep the water below 60 °C. Handle the bath with tongs and safety glasses.

The ice bath is prepared with crushed ice and distilled water, stirred, and allowed to equilibrate for at least 10 minutes. The thermometer is immersed to the specified depth, away from the walls and the bottom.

#### C2. Take the readings

At each point, record the reference and the sensor, ascending and descending, with three repeats. Randomise the order within each direction to avoid a systematic drift being mistaken for hysteresis.

Record:

- The reference temperature.
- The sensor's raw value.
- The time.
- The bath temperature stability.
- The immersion depth.
- The ambient temperature.

#### C3. Fit and inspect the residuals

Fit reference = a × raw + b. Plot the residuals against the reference. A curve in the residuals indicates nonlinearity; a slope in the residuals indicates a gain error; a step indicates a change in the setup.

Report the slope, the intercept, the residual standard deviation, and the largest residual.

#### C4. Compute the hysteresis

For each point, compute the difference between the ascending and descending means. The maximum difference is the hysteresis. Compare it with the tolerance.

#### C5. Build the uncertainty budget

Combine the reference uncertainty, the repeatability, the resolution, the fit residual, the drift since calibration, and the self-heating. Combine in quadrature and report the expanded uncertainty at k = 2.

The budget must state the distribution assumed for each Type B term. A rectangular distribution is common for a tolerance; a normal distribution is common for a certificate.

### Step 9. Run a Gauge R&R if the decision is about a tolerance

**Input:** the measurement system, ten parts or levels, three operators, three trials.

**Output:** the variance components, the %GRR, and the number of distinct categories.

**Rationale:** a calibration tells you the curve. A Gauge R&R tells you whether the measurement system can distinguish parts within the tolerance. They answer different questions.

The design is crossed: every operator measures every part, three times, in random order, blind to the part identity. The analysis uses the mean squares from a two-way analysis of variance with interaction.

The variance components are:

| Source | Mean square | Variance component |
|---|---|---|
| Repeatability | MS_e | \(\sigma_e^2 = MS_e\) |
| Operator × part | MS_op | \(\sigma_{op}^2 = (MS_{op} - MS_e)/n\) |
| Operator | MS_o | \(\sigma_o^2 = (MS_o - MS_{op})/(p n)\) |
| Part | MS_p | \(\sigma_p^2 = (MS_p - MS_{op})/(o n)\) |

Then:

\[
\sigma_{GRR} = \sqrt{\sigma_e^2 + \sigma_o^2 + \sigma_{op}^2}
\]

\[
\%GRR = \frac{6 \sigma_{GRR}}{\text{tolerance}} \times 100
\]

\[
ndc = 1.41 \frac{\sigma_p}{\sigma_{GRR}}
\]

The AIAG heuristic treats under 10% as acceptable, 10–30% as conditional, and over 30% as unacceptable. The heuristic is a starting point, not a decision rule; the decision must be justified for the clinical or manufacturing context.

### Step 10. Compare with the datasheet, not with a memory

**Input:** the measured values and the datasheet.

**Output:** a comparison table with the conditions and the verdict.

**Rationale:** a datasheet value is conditional. The comparison must match the conditions: supply voltage, gain, bandwidth, temperature, and load.

| Parameter | Datasheet | Condition | Measured | Condition | Verdict |
|---|---:|---|---:|---|---|
| CMRR | 80 dB min | DC–60 Hz, gain 100 | 78.0 dB | 50 Hz, gain 100 | Fail |
| Noise | 14 µV p-p | 0.5–40 Hz | 13.7 µV RMS | 0.5–40 Hz | Compare convention |
| Gain | 100 | ±1% resistors | 99.2 | 10 Hz | Within tolerance |
| Offset | ±300 mV input range | — | 1.2 mV | Zero input | Within range |

A verdict that compares a peak-to-peak datasheet value with an RMS measurement is not a comparison. The conventions must match.

### Step 11. Record the environmental conditions

**Input:** the lab environment.

**Output:** the temperature, the humidity, the supply voltage, and any interference source.

**Rationale:** a sensor measurement is a measurement of the environment as well as the sensor. A temperature calibration at 18 °C and a use at 35 °C are different conditions.

Record the ambient temperature and humidity, the supply voltage at the device, the presence of a switching supply or a radio, the warm-up time, and the time since power-on.

### Step 12. State what the result does not prove

**Input:** the claim and the evidence.

**Output:** a limitations paragraph with at least three items.

**Rationale:** a sensor result is easy to over-extend. The limitations paragraph is the boundary.

Typical limitations:

- One sensor, one lot, one board, one reference with its own uncertainty.
- A bench input, not a physiological source.
- A temperature range narrower than the clinical range.
- No clinical population, long-term drift data, EMC testing, or safety testing.

### Step 13. Review, report, and archive

**Input:** the protocol, the data, and the analysis.

**Output:** a review record, a report, and an archive that a reviewer can reopen.

**Rationale:** the review catches the errors that the operator cannot see: an unbalanced source, a missing calibration, a point taken before equilibration. The report and the archive are the durable outputs.

The review checks the reference, the balance, the equilibration, the fit, the residuals, the uncertainty, and the limitations. The report contains the quantity, the sensor, the reference, the traceability, the setup, the conditions, the calibration curve, the residuals, the hysteresis, the budget, the Gauge R&R where applicable, the datasheet comparison, the verdict, and the limitations. The archive contains the raw readings, the reference values, the analysis script, the certificate, and the hashes.

### 5.16 Acceptance criteria for a sensor lab

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Quantity and claim boundary are written | Charter | Rewrite |
| 2 | Reference has a certificate and an uncertainty | Certificate | Replace |
| 3 | Reference uncertainty is at least 4× smaller than the tolerance | Budget | Replace |
| 4 | Sensor datasheet conditions are recorded | Datasheet table | Complete |
| 5 | Topology is documented with the ground return | Schematic | Redraw |
| 6 | Quiescent point is in the linear range | Measurement | Fix |
| 7 | End-to-end calibration is reported | Fit | Recalibrate |
| 8 | Noise is reported with a bandwidth | Noise log | Re-measure |
| 9 | CMRR source is balanced | Balance check | Rebuild |
| 10 | Calibration is ascending and descending | Data table | Repeat |
| 11 | Hysteresis is computed | Table | Compute |
| 12 | Uncertainty budget has at least five sources | Budget | Extend |
| 13 | Gauge R&R is computed from variance components | Table | Recompute |
| 14 | Datasheet comparison matches conditions | Comparison table | Correct |
| 15 | Limitations name at least three items | Report | Complete |

### 5.17 Branch points and stop points

**Branch A: the sensor is patient-connected.** Stop. A patient connection requires an applied-part classification, isolation, and a leakage test under qualified supervision. The lab uses a resistor network or a simulator instead.

**Branch B: the claim is SpO2 accuracy.** Stop. The claim requires a clinical calibration study with arterial blood-gas references. The bench can measure the ratio and the timing, not the accuracy.

**Branch C: the water bath exceeds 60 °C.** Stop. Use a lower point or a supervised bath with a lid and a thermometer. Hot water is a burn hazard, and a bath above 60 °C is outside this course's safety boundary.

**Branch D: the reference has no certificate.** Stop. An uncalibrated reference makes the calibration a comparison between two unknown sensors. Replace the reference or label the result as indicative.

## 6. Worked example

### 6.1 The question

A team is calibrating a digital temperature sensor for a wearable prototype. The sensor is specified to ±0.1 °C over −20 to 50 °C. The team wants to know whether the sensor, as assembled on the prototype board, meets that specification over 0–50 °C, and whether the measurement system can support a ±0.1 °C tolerance decision.

### 6.2 The setup

| Item | Value | Note |
|---|---|---|
| Sensor | Digital temperature sensor, 16-bit | 0.0078 °C resolution |
| Interface | I2C, 400 kHz | Read once per second |
| Reference | Calibrated thermometer | ±0.05 °C, 0.01 °C resolution |
| Reference certificate | Accredited laboratory, 2026-02 | Traceable |
| Bath | Stirred water bath, 0–50 °C | Stability ±0.02 °C |
| Ice bath | Crushed ice and distilled water | Stirred, 10-minute equilibration |
| Immersion | 80 mm | Above the minimum mark |
| Ambient | 22.4 °C, 41% relative humidity | Logged |
| Supply | 3.30 V from a linear regulator | Measured at the sensor |
| Warm-up | 10 minutes | Before the first point |

The sensor is on a break-out board with a 100 mm cable. The board is in still air, not in the bath, for the first test. This is a deliberate choice: the team wants to know the sensor's accuracy, not the probe assembly's. The distinction matters because a probe assembly's thermal response depends on its mass, its insulation, and its immersion, none of which are properties of the sensor alone.

### 6.3 The readings

Five points, ascending and descending, three repeats each. The table shows the mean of the three repeats at each point.

| Point | Direction | Reference (°C) | Sensor mean (°C) | Difference (°C) |
|---:|---|---:|---:|---:|
| 1 | Ascending | 0.02 | 0.000 | −0.020 |
| 2 | Ascending | 10.01 | 9.988 | −0.022 |
| 3 | Ascending | 25.00 | 24.982 | −0.018 |
| 4 | Ascending | 37.00 | 36.989 | −0.011 |
| 5 | Ascending | 50.00 | 49.994 | −0.006 |
| 5 | Descending | 50.00 | 50.006 | +0.006 |
| 4 | Descending | 37.00 | 37.012 | +0.012 |
| 3 | Descending | 25.00 | 25.008 | +0.008 |
| 2 | Descending | 10.01 | 10.002 | −0.008 |
| 1 | Descending | 0.02 | 0.014 | −0.006 |

The sensor reads low on the ascending pass and high on the descending pass. That is a hysteresis loop of about 0.02–0.03 °C, consistent with the thermal lag of the package.

### 6.4 The fit

The team fits reference = a × sensor + b using all 30 readings, not the ten means. The result is:

\[
a = 0.99967, \quad b = +0.0146\ \text{°C}
\]

The residual standard deviation is 0.0100 °C, and the largest residual is 0.0153 °C. The slope is 0.033% below unity, which is negligible. The intercept is +0.015 °C, which is the sensor's offset at the ice point.

The residuals do not show a curve. A straight line is adequate over this range. A quadratic term would reduce the residual by less than the uncertainty, so it is not justified.

### 6.5 The hysteresis

The hysteresis at each point is the difference between the ascending and descending means:

| Point | Ascending (°C) | Descending (°C) | Hysteresis (°C) |
|---:|---:|---:|---:|
| 1 | 0.000 | 0.014 | 0.014 |
| 2 | 9.988 | 10.002 | 0.014 |
| 3 | 24.982 | 25.008 | 0.026 |
| 4 | 36.989 | 37.012 | 0.023 |
| 5 | 49.994 | 50.006 | 0.012 |

The maximum hysteresis is 0.026 °C at 25 °C. It is 26% of the ±0.1 °C tolerance, which is significant but not disqualifying.

### 6.6 The uncertainty budget

| Source | Value | Distribution | Divisor | Standard uncertainty (°C) |
|---|---:|---|---:|---:|
| Reference certificate | 0.05 | Normal, k = 2 | 2 | 0.0250 |
| Reference resolution | 0.01 | Rectangular | √3 | 0.0058 |
| Repeatability of the mean | 0.010 | Normal | √3 | 0.0058 |
| Sensor resolution | 0.0078 | Rectangular | √12 | 0.0023 |
| Fit residual | 0.010 | Normal | 1 | 0.0100 |
| Drift since calibration | 0.010 | Rectangular | √3 | 0.0058 |
| Self-heating | 0.004 | Rectangular | √3 | 0.0023 |
| **Combined** | | | | **0.0289** |
| **Expanded, k = 2** | | | | **0.0578** |

The combined standard uncertainty is 0.0289 °C, and the expanded uncertainty is 0.058 °C. The dominant term is the reference certificate (0.025 °C), followed by the fit residual (0.010 °C).

The budget does not include the bath's spatial non-uniformity, which the team measured separately as 0.015 °C between the reference and the sensor location. If that term is added:

\[
u_c = \sqrt{0.0289^2 + (0.015/\sqrt{3})^2} = \sqrt{0.000835 + 0.000075} = 0.0302\ \text{°C}
\]

The expanded uncertainty becomes 0.060 °C. The conclusion does not change, but the budget is more honest.

### 6.7 Does the sensor meet its specification?

The specified accuracy is ±0.1 °C. The largest observed deviation from the reference, after the offset correction, is 0.0153 °C. The expanded uncertainty is 0.0604 °C.

| Quantity | Value | Unit | Comparison |
|---|---:|---|---|
| Largest residual | 0.0153 | °C | — |
| Specification | 0.100 | °C | Within |
| Expanded uncertainty (k = 2) | 0.0604 | °C | — |
| Upper confidence bound on the error | 0.0757 | °C | Within |
| Verdict | | | Pass, with 0.0243 °C margin |

The upper bound is the largest residual plus the expanded uncertainty, 0.0153 + 0.0604 = 0.0757 °C. It is below 0.1 °C, so the sensor passes. The margin is 0.0243 °C, which is small. A second board or a second lot could fail.

The pass is a statement about this sensor, on this board, at these five points, against this reference. It is not a statement about the product.

### 6.8 Is the measurement system fit for a ±0.1 °C tolerance?

The tolerance is ±0.1 °C, so the tolerance width is 0.2 °C. The Gauge R&R uses the standard deviation of the measurement system, not the expanded uncertainty. From the budget, the combined standard uncertainty including the spatial term is 0.0302 °C, which is the system's standard deviation for this design.

\[
\%GRR = \frac{6 \times 0.0302}{0.2} \times 100 = 90.6\%
\]

The AIAG heuristic calls anything over 30% unacceptable. The measurement system consumes 91% of the tolerance width, which means it cannot reliably distinguish a conforming unit from a non-conforming one at the tolerance limit.

This is the central lesson of the example. The sensor passes its specification, and the measurement system is still not fit for a tolerance decision at that specification. The pass is a comparison of a mean against a datasheet; the tolerance decision requires discrimination at the limit, and the reference is not accurate enough.

### 6.9 What would fix it

| Change | Effect on u_c | Effect on %GRR | Cost |
|---|---:|---:|---|
| Better reference, ±0.02 °C | 0.0197 °C | 59% | Higher |
| More repeats at each point | Small; repeatability is not dominant | Small | Time |
| Better fit model | Reduces 0.010 to ~0.005 | 87% | Analysis |
| Narrower tolerance, ±0.2 °C | Unchanged | 45% | Product change |
| Fixed-point calibration at 0 °C and 50 °C | Removes the reference certificate term at those points | Depends | Procedure |

The most effective change is a better reference. The repeatability is already small, and more repeats will not fix a reference that is the dominant term.

### 6.10 Two checks that would invalidate the result

**Check 1: the ice bath was not at 0 °C.** An ice bath that is not stirred, has too little ice, or has not equilibrated can be 0.2–0.5 °C above the ice point. The reference would then read 0.3 °C instead of 0.0 °C, and the offset at point 1 would be wrong by 0.3 °C, which is three times the specification. The check is to verify the ice bath with a second thermometer and to confirm that the ice is crushed and the water is distilled. A bath that has been topped up with tap water is not a fixed point.

**Check 2: the reference thermometer's immersion depth was insufficient.** A thermometer with a 5 mm stem and an 80 mm immersion has a stem conduction error that depends on the stem temperature and the immersion depth. If the stem is at 22 °C and the bath is at 50 °C, the error can be 0.05–0.2 °C. The check is to compare the reading at two immersion depths. If the reading changes with depth, the immersion is insufficient, and the calibration is invalid.

### 6.11 A varied condition: the reference is an uncalibrated thermometer

The team repeats the analysis with a second thermometer that has no certificate. The datasheet claims ±0.5 °C, and the team assumes a rectangular distribution.

| Condition | Reference uncertainty | u_c (°C) | U (k = 2) | %GRR | Verdict |
|---|---:|---:|---:|---:|---|
| Calibrated reference | 0.025 | 0.0302 | 0.0604 | 90.6% | Marginal |
| Uncalibrated reference | 0.289 | 0.2894 | 0.579 | 868% | Unusable |

The change is a factor of about 10 in the uncertainty. The sensor still passes its specification if the comparison is made against the uncalibrated reference, because the observed difference is small. But the measurement system is now useless for a tolerance decision, and the pass is meaningless.

This is the difference between a calibration and a comparison. A calibration has a traceable reference and a stated uncertainty. A comparison has two sensors and no truth.

### 6.12 What the example demonstrates

The arithmetic closes: a slope of 0.99967, an intercept of +0.0146 °C, a residual standard deviation of 0.0100 °C, a maximum hysteresis of 0.026 °C, a combined standard uncertainty of 0.0302 °C, an expanded uncertainty of 0.0604 °C, and a %GRR of 90.6% against a ±0.1 °C tolerance.

The example demonstrates the difference between a sensor specification and a measurement-system capability. The sensor meets its datasheet. The system cannot support a tolerance decision at that datasheet limit. A team that reports only "the sensor is within ±0.1 °C" has told half the story.

The half that is missing is the decision. A specification is a statement about a component under stated conditions. A decision is a statement about a unit relative to a limit, and it requires a measurement system whose uncertainty is small relative to the distance from the limit.

The same arithmetic applies to an ECG amplitude, a PPG timing, or a force reading. In each case, the question is not "is the number within the specification" but "can this system tell a conforming unit from a non-conforming one." The %GRR is one way to answer the second question.

## 7. Code, query, or calculation

The program below performs the calibration fit, the hysteresis calculation, the uncertainty budget, and a crossed Gauge R&R. It uses only the Python standard library. The calibration data are embedded, and the Gauge R&R data are generated deterministically from a fixed seed with known variance components, so the program runs without hardware.

```python
import math
import random
import statistics

# --- 1. Calibration data: (point, direction, reference, sensor) ----------
DATA = [
    (1, "asc", 0.02, 0.000), (1, "asc", 0.02, 0.001), (1, "asc", 0.02, -0.001),
    (2, "asc", 10.01, 9.988), (2, "asc", 10.01, 9.989), (2, "asc", 10.01, 9.987),
    (3, "asc", 25.00, 24.982), (3, "asc", 25.00, 24.983), (3, "asc", 25.00, 24.981),
    (4, "asc", 37.00, 36.989), (4, "asc", 37.00, 36.990), (4, "asc", 37.00, 36.988),
    (5, "asc", 50.00, 49.994), (5, "asc", 50.00, 49.995), (5, "asc", 50.00, 49.993),
    (5, "desc", 50.00, 50.006), (5, "desc", 50.00, 50.007), (5, "desc", 50.00, 50.005),
    (4, "desc", 37.00, 37.012), (4, "desc", 37.00, 37.013), (4, "desc", 37.00, 37.011),
    (3, "desc", 25.00, 25.008), (3, "desc", 25.00, 25.009), (3, "desc", 25.00, 25.007),
    (2, "desc", 10.01, 10.002), (2, "desc", 10.01, 10.003), (2, "desc", 10.01, 10.001),
    (1, "desc", 0.02, 0.014), (1, "desc", 0.02, 0.015), (1, "desc", 0.02, 0.013),
]

def ols(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    resid = [y - (a * x + b) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    return a, b, sd, max(abs(r) for r in resid), resid

xs = [r[3] for r in DATA]
ys = [r[2] for r in DATA]
a, b, sd, max_res, resid = ols(xs, ys)
print("calibration fit: reference = a*sensor + b")
print(f"a              = {a:.5f}")
print(f"b              = {b:+.4f} C")
print(f"residual sd    = {sd:.4f} C")
print(f"max |residual| = {max_res:.4f} C")

# --- 2. Hysteresis -------------------------------------------------------
by_point = {}
for point, direction, ref, sensor in DATA:
    by_point.setdefault(point, {"asc": [], "desc": [], "ref": ref})
    by_point[point][direction].append(sensor)

print("\nhysteresis")
print("point  asc      desc     diff")
hyst = []
for point in sorted(by_point):
    asc = statistics.fmean(by_point[point]["asc"])
    desc = statistics.fmean(by_point[point]["desc"])
    diff = desc - asc
    hyst.append(diff)
    print(f"{point:5d}  {asc:8.4f} {desc:8.4f} {diff:+8.4f}")
print(f"max hysteresis = {max(abs(h) for h in hyst):.4f} C")

# --- 3. Uncertainty budget ----------------------------------------------
def rect(a):
    return a / math.sqrt(3)

def rect12(a):
    return math.sqrt(a * a / 12)

u_ref = 0.05 / 2           # certificate, normal k=2
u_ref_res = rect(0.01)
u_rep = sd / math.sqrt(3)  # mean of three repeats
u_sensor_res = rect12(0.0078)
u_fit = sd
u_drift = rect(0.010)
u_self = rect(0.004)
u_spatial = rect(0.015)

components = {
    "reference certificate": u_ref,
    "reference resolution": u_ref_res,
    "repeatability": u_rep,
    "sensor resolution": u_sensor_res,
    "fit residual": u_fit,
    "drift": u_drift,
    "self-heating": u_self,
}
u_c = math.sqrt(sum(v * v for v in components.values()))
u_c_spatial = math.sqrt(u_c * u_c + u_spatial * u_spatial)

print("\nuncertainty budget")
print(f"{'source':24s}{'u (C)':>10s}")
for k, v in components.items():
    print(f"{k:24s}{v:10.4f}")
print(f"{'combined':24s}{u_c:10.4f}")
print(f"{'with spatial':24s}{u_c_spatial:10.4f}")
print(f"{'expanded k=2':24s}{2*u_c_spatial:10.4f}")

# --- 4. Gauge R&R: 10 parts x 3 operators x 3 trials ---------------------
# Known components used to generate the data (C):
SIGMA_PART = 0.300
SIGMA_OP = 0.015
SIGMA_INT = 0.010
SIGMA_E = 0.020
rng = random.Random(20261026)

parts = [rng.gauss(0.0, SIGMA_PART) for _ in range(10)]
ops = [0.0, 0.01, -0.01]
rows = []
for p in range(10):
    for o in range(3):
        inter = rng.gauss(0.0, SIGMA_INT)
        for t in range(3):
            y = parts[p] + ops[o] + inter + rng.gauss(0.0, SIGMA_E)
            rows.append((p, o, t, y))

def cell_means(rows):
    cells = {}
    for p, o, t, y in rows:
        cells.setdefault((p, o), []).append(y)
    return {k: statistics.fmean(v) for k, v in cells.items()}

pm = {}
for p, o, t, y in rows:
    pm.setdefault(p, []).append(y)
pm = {k: statistics.fmean(v) for k, v in pm.items()}
om = {}
for p, o, t, y in rows:
    om.setdefault(o, []).append(y)
om = {k: statistics.fmean(v) for k, v in om.items()}
gm = statistics.fmean([r[3] for r in rows])
cm = cell_means(rows)

p_n, o_n, t_n = 10, 3, 3
ss_p = o_n * t_n * sum((pm[p] - gm) ** 2 for p in pm)
ss_o = p_n * t_n * sum((om[o] - gm) ** 2 for o in om)
ss_po = t_n * sum((cm[(p, o)] - pm[p] - om[o] + gm) ** 2
                  for p in range(p_n) for o in range(o_n))
ss_e = sum((y - cm[(p, o)]) ** 2 for p, o, t, y in rows)

df_p, df_o = p_n - 1, o_n - 1
df_po = df_p * df_o
df_e = p_n * o_n * (t_n - 1)
ms_p = ss_p / df_p
ms_o = ss_o / df_o
ms_po = ss_po / df_po
ms_e = ss_e / df_e

var_e = ms_e
var_po = max(0.0, (ms_po - ms_e) / t_n)
var_o = max(0.0, (ms_o - ms_po) / (p_n * t_n))
var_p = max(0.0, (ms_p - ms_po) / (o_n * t_n))
grr = math.sqrt(var_e + var_po + var_o)
ndc = 1.41 * math.sqrt(var_p) / grr
tol_width = 0.2

print("\ngauge R&R (generated data)")
print(f"MS_part={ms_p:.5f}  MS_op={ms_o:.5f}  MS_int={ms_po:.5f}  MS_e={ms_e:.5f}")
print(f"sigma_e   = {math.sqrt(var_e):.4f} C")
print(f"sigma_op  = {math.sqrt(var_o):.4f} C")
print(f"sigma_int = {math.sqrt(var_po):.4f} C")
print(f"sigma_part= {math.sqrt(var_p):.4f} C")
print(f"sigma_GRR = {grr:.4f} C")
print(f"%GRR      = {6*grr/tol_width*100:.1f}%")
print(f"ndc       = {ndc:.2f}")
```

Expected output:

```text
calibration fit: reference = a*sensor + b
a              = 0.99967
b              = +0.0146 C
residual sd    = 0.0100 C
max |residual| = 0.0153 C

hysteresis
point  asc      desc     diff
    1    0.0000   0.0140  +0.0140
    2    9.9880  10.0020  +0.0140
    3   24.9820  25.0080  +0.0260
    4   36.9890  37.0120  +0.0230
    5   49.9940  50.0060  +0.0120
max hysteresis = 0.0260 C

uncertainty budget
source                       u (C)
reference certificate       0.0250
reference resolution        0.0058
repeatability               0.0058
sensor resolution           0.0023
fit residual                0.0100
drift                       0.0058
self-heating                0.0023
combined                    0.0289
with spatial                0.0302
expanded k=2                0.0604

gauge R&R (generated data)
MS_part=0.37264  MS_op=0.00177  MS_int=0.00037  MS_e=0.00044
sigma_e   = 0.0210 C
sigma_op  = 0.0068 C
sigma_int = 0.0000 C
sigma_part= 0.2034 C
sigma_GRR = 0.0221 C
%GRR      = 66.2%
ndc       = 13.00
```

The calibration fit reproduces the worked example: a slope of 0.99967, an intercept of +0.0146 °C, and a residual standard deviation of 0.0100 °C. The hysteresis table matches the ten means in Section 6.5. The uncertainty budget gives 0.0289 °C without the spatial term and 0.0302 °C with it, and an expanded uncertainty of 0.0604 °C.

The Gauge R&R section is generated from known variance components. The mean squares and the recovered components will differ from the generating values because the sample is small and the random draw is finite. The program prints the values so that the reader can compare them with the generating values. A real study would use measured data, not generated data, and the variance components would be the evidence.

The %GRR from the calibration budget, 88.2%, is computed separately in Section 6.8. The Gauge R&R section computes a different %GRR, from the generated crossed study. The two are not the same quantity: the first uses the calibration uncertainty, and the second uses the variance components of the measurement system in a repeated-measures design. Both are reported, and the difference is explained.

## 8. Data, standards, and regulatory context

### 8.1 ECG standards

IEC 60601-2-27:2011 is the particular standard for electrocardiographic monitoring equipment. It applies to monitors used in hospitals, ambulances, air transport, and in-hospital telemetry, and it excludes home-use ECG monitors, ambulatory Holter monitors, fetal heart rate monitors, and pulse plethysmographic devices.

The standard specifies the performance of the electrocardiograph channel, including the input impedance, the common-mode rejection, the patient leakage current, the frequency response, the noise, and the recovery from a defibrillation discharge. The exact clauses and limits are in the purchased text.

ANSI/AAMI EC11:1991 (R2007) is the US standard for diagnostic electrocardiographic devices. It covers devices intended for diagnostic contour analysis, not monitoring. ANSI/AAMI EC13 covers cardiac monitors, heart rate meters, and alarms.

The distinction matters because a wearable ECG is not automatically a diagnostic electrocardiograph. The applicable standard depends on the intended use, the alarm function, and the clinical claim.

### 8.2 Pulse oximetry standards

ISO 80601-2-61 is the particular standard for pulse oximeters. It specifies that SpO2 accuracy shall be a root-mean-square difference (ARMS) of no more than 4.0% over the range of 70% to 100% SaO2 for the 2017 edition, with a declared range and a modified Bland–Altman plot.

The FDA's 2013 guidance on pulse oximeter premarket submissions recommends stricter typical ARMS values by sensor type: 3.0% for transmittance wrap or clip sensors, 3.5% for ear clip, and 3.5% for reflectance. The FDA does not recognise the standard's 4% clause for all sensors because it conflicts with the guidance.

The third edition of ISO 80601-2-61 was in final development in 2026. The reported changes include a reduced maximum permissible ARMS, a differential-bias requirement, larger and more diverse participant cohorts, and expanded reporting. A lab must check the current edition before citing a limit.

A bench cannot establish ARMS. The measurement requires human subjects, controlled desaturation, and arterial blood-gas sampling. A bench can measure the ratio's repeatability, the sensor's noise, and the timing; it cannot measure the accuracy of the saturation estimate.

### 8.3 Clinical thermometer standards

ISO 80601-2-56 is the particular standard for clinical thermometers for body temperature measurement. The second edition was published in 2017, with Amendment 1 in 2018. A third edition was in final approval in 2026. The standard covers the basic safety and essential performance of electrical clinical thermometers, and it excludes screening thermographs.

ASTM E1112 is the standard specification for electronic thermometers for intermittent determination of patient temperature. It covers the display range, the accuracy, the environmental conditions, the resolution, the cleaning durability, the drop resistance, and the contact-part nontoxicity.

In the United States, 21 CFR 880.2910 classifies clinical electronic thermometers as Class II, exempt from 510(k) subject to conditions, including testing to FDA-recognised editions of the relevant standards.

A bench calibration against a calibrated thermometer is a component of the evidence for these standards. It is not the whole of the evidence. The standard also requires environmental testing, cleaning, drop, and clinical performance.

### 8.4 The Gauge R&R reference

The AIAG Measurement Systems Analysis reference manual, fourth edition (2010), defines the average-and-range method and the analysis-of-variance method for a Gauge R&R. The manual's acceptance heuristic is under 10% acceptable, 10–30% conditional, and over 30% unacceptable, measured against either the process variation or the tolerance.

The manual uses a 6σ multiplier in the fourth edition, equivalent to 99.73% of the distribution. Earlier editions used 5.15σ, equivalent to 99%. A report must state which multiplier was used.

The number of distinct categories is \(ndc = 1.41 \times PV/GRR\). The manual treats 5 or more as desirable.

### 8.5 Uncertainty and traceability

The GUM (JCGM 100:2008) defines the uncertainty evaluation. ISO/IEC 17025:2017 Clause 6.5 requires metrological traceability, and Clause 7.8.4 requires the calibration certificate to state the uncertainty and the traceability.

A reference thermometer with a certificate from an accredited laboratory has a stated uncertainty and a traceability chain. A thermometer without a certificate has neither, even if its display is stable and its readings are plausible.

### 8.6 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| IEC 60601-2-27 | Particular standard | ECG monitoring equipment | A wearable prototype |
| ANSI/AAMI EC11 | US standard | Diagnostic electrocardiographs | A monitoring device |
| ISO 80601-2-61 | Particular standard | Pulse oximeters | A bench PPG sensor |
| ISO 80601-2-56 | Particular standard | Clinical thermometers | A temperature sensor on a breakout board |
| ASTM E1112 | Standard specification | Electronic thermometers | A sensor component |
| AIAG MSA 4th edition | Industry reference | Gauge R&R practice where adopted | A research measurement |
| JCGM 100:2008 | International guidance | Uncertainty statements | A rough estimate |

## 9. Industry practice and case evidence

### 9.1 How sensor vendors specify accuracy

A sensor datasheet's accuracy is a statement about a population of units under defined conditions. The conditions usually include the supply voltage, the temperature range, the load, and the measurement time. A sensor specified to ±0.1 °C over −20 to 50 °C may be specified to ±0.2 °C over a wider range, and the transition is not always smooth.

The TMP117 datasheet, for example, specifies ±0.1 °C from −20 to 50 °C, ±0.15 °C from −40 to 70 °C, ±0.2 °C from −40 to 100 °C, and ±0.25 °C from −55 to 125 °C. It also states that each unit is tested on a production setup that is NIST traceable and verified with equipment calibrated to ISO/IEC 17025 accredited standards. The datasheet does not issue a per-device calibration certificate with a measurement uncertainty; the specification is a min/max limit.

The distinction matters for a lab. A factory-tested sensor is not the same as a calibrated sensor with a certificate. The factory test supports the datasheet limit; the certificate supports a measurement uncertainty for a specific unit.

### 9.2 Why SpO2 cannot be bench-calibrated

The SpO2 value is derived from the ratio of the pulsatile to the non-pulsatile optical absorption at two wavelengths. The relationship between the ratio and arterial saturation is empirical. It depends on the sensor geometry, the wavelengths, the tissue, the perfusion, the skin pigmentation, and the haemoglobin species.

A calibration curve is therefore derived from human subjects with arterial blood-gas references, under a controlled desaturation protocol. The curve is specific to the sensor and the algorithm. A bench measurement with a pulse simulator or a filter can check the electronics and the timing, but it cannot check the curve.

The clinical consequence is documented. The FDA and the UK Medicines and Healthcare products Regulatory Agency (MHRA) have both warned that pulse oximeters can overestimate oxygen saturation in people with darker skin pigmentation. The mechanism is understood in general terms: the calibration curves were derived from populations that under-represented dark skin, and the optical path differs. The exact bias varies by device. A bench lab cannot reproduce or correct this bias.

### 9.3 Temperature calibration in industry

A production temperature calibration typically uses a dry-block calibrator or a stirred bath, a reference thermometer, and a fixture that controls the immersion depth. The calibration is performed at several points, and the residuals are checked against the tolerance.

The ice point is used as a fixed point because it is reproducible without a certificate, if it is prepared correctly. The triple point of water is more reproducible but requires a sealed cell and a controlled environment, which is beyond a teaching lab.

The self-heating error is a function of the excitation current and the thermal resistance. A thermistor with a 100 µA excitation and a 1 mW/°C dissipation constant has a self-heating of \(I^2 R / 1\ \text{mW/°C}\). A digital sensor with a low duty cycle has a much smaller self-heating.

### 9.4 ECG front-end practice

An ECG front end is designed around the electrode offset, the common-mode interference, and the safety isolation. The electrode offset can be hundreds of millivolts, which is why the front end is usually AC-coupled or has a DC restoration loop. The common-mode interference from the mains can be several volts, which is why the common-mode rejection and the right-leg drive are critical.

The AD8232 is a single-lead heart-rate monitor front end with a gain of 100, a common-mode rejection ratio of 80 dB minimum from DC to 60 Hz, an input-referred noise of 14 µV peak-to-peak from 0.5 to 40 Hz, and a supply current of 170 µA typical. These are the numbers a lab compares against.

A bench measurement of CMRR is only as good as the source balance. A 1% resistor mismatch in the source network converts common-mode to differential and limits the measurable CMRR to about 46 dB, regardless of the amplifier.

### 9.5 Inference

It is an inference, not a documented finding, that teams which measure the reference uncertainty before the calibration make better conformity decisions. The reasoning is that the reference term often dominates the budget, and a team that has not quantified it cannot know whether the calibration can decide the tolerance. The inference is consistent with the worked example, but it has not been tested across teams.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Reference uncalibrated | No certificate, only a datasheet | Unknown reference error | Meaningless calibration | Use a traceable reference |
| 2 | Ice bath not prepared | Ice floating, water cloudy | Not at 0 °C | Offset error | Crush ice, stir, equilibrate |
| 3 | Immersion too shallow | Reading changes with depth | Stem conduction | Temperature error | Observe the immersion mark |
| 4 | Self-heating ignored | Sensor warm to the touch | Excitation power | Positive bias | Low duty cycle, calculate |
| 5 | CMRR source unbalanced | 1% resistors in the source | Common-mode to differential conversion | CMRR limited by the source | Match resistors, measure balance |
| 6 | Noise quoted without bandwidth | "14 µV" with no band | Different noise bandwidth | Incomparable numbers | State the band and filter |
| 7 | Peak-to-peak compared with RMS | Datasheet p-p, measured RMS | Convention mismatch | False pass or fail | Convert with a stated crest factor |
| 8 | PPG calibrated on the bench | A curve fitted to a simulator | No blood-gas reference | Invalid SpO2 claim | Label as timing only |
| 9 | Skin-tone bias ignored | Validation on one population | Optical path difference | Clinical bias | Diverse clinical study |
| 10 | Hysteresis not measured | Only ascending data | Thermal lag | Offset depends on direction | Ascending and descending |
| 11 | Fit extrapolated beyond the range | A curve used outside the points | Nonlinearity | Wrong reading | State the range |
| 12 | Repeatability confused with reproducibility | One operator, one day | Operator and setup effects | Understated variation | Crossed Gauge R&R |
| 13 | Tolerance and uncertainty confused | %GRR computed from U | Wrong divisor | Wrong fitness decision | Use the variance components |
| 14 | Datasheet conditions ignored | Comparing at a different gain | Conditional specification | False verdict | Match the conditions |
| 15 | One lot treated as the population | A single sensor | Lot-to-lot variation | Overgeneralised claim | Multiple lots |
| 16 | Clinical claim from a bench | "Accurate to 1 bpm" | Bench is not a patient | Regulatory and ethical problem | State the boundary |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The reference is traceable and in calibration | The calibration has a truth | Expired or missing certificate | Read the certificate |
| 2 | The reference uncertainty is much smaller than the tolerance | The calibration can decide conformity | Reference comparable to the sensor | Compare the uncertainties |
| 3 | The bath is at the reference temperature | The set point is the reference | Spatial gradient, insufficient equilibration | Measure at two locations |
| 4 | The sensor is at the bath temperature | The reading represents the bath | Self-heating, poor immersion | Compare at two excitation levels |
| 5 | The fit model is adequate over the range | The curve is used for conversion | Nonlinearity outside the points | Plot residuals; add points |
| 6 | The source is balanced for the CMRR test | The measurement reflects the amplifier | Resistor mismatch | Measure the source balance |
| 7 | The noise is stationary | The standard deviation is meaningful | Drift, bursts, warm-up | Plot the trend |
| 8 | The PPG simulator represents the optical signal | The timing test is meaningful | Different waveform, no skin | Compare with a recorded waveform |
| 9 | The Gauge R&R parts span the tolerance | The variance components are estimable | Parts clustered | Select parts across the range |
| 10 | The operators are representative | The reproducibility term is meaningful | One trained operator | Include the intended users |
| 11 | The environment is controlled | The calibration is repeatable | Temperature and humidity variation | Log and bound |
| 12 | The result is about one sensor | The claim is bounded | A different lot or board | Test multiple units |

## 12. Exercises

### Exercise 1. Draw the conversion chain (recall and application)

**Deliverable:** a diagram of the chain from core temperature to a displayed value, with the assumption at each arrow and the reference that could validate it.

**Acceptance criteria:** at least five arrows; each arrow names a model or a calibration; each reference is identified as available or unavailable on a bench; the diagram distinguishes the clinical definition from the sensor law.

**Time:** 30 minutes.

### Exercise 2. Measure CMRR (application)

**Deliverable:** a CMRR measurement at 50 Hz for an ECG front end, with the source balance check, the gain, the output voltage, and the uncertainty.

**Acceptance criteria:** the source is balanced to better than 0.1%; the gain is measured at the same frequency; the CMRR is computed from the measured values; the uncertainty includes the source, the gain, and the meter; the result is compared with the datasheet condition.

**Time:** 60 minutes.

### Exercise 3. Temperature calibration (application)

**Deliverable:** a five-point calibration, ascending and descending, with the fit, the residuals, the hysteresis, and the uncertainty budget.

**Acceptance criteria:** the reference has a certificate; the ice bath is prepared and verified; the fit reports the slope, intercept, and residual SD; the hysteresis is computed at each point; the budget has at least five sources and an expanded uncertainty at k = 2.

**Time:** 90 minutes.

### Exercise 4. Gauge R&R (calculation)

**Deliverable:** a crossed Gauge R&R with 10 parts, 3 operators, and 3 trials, with the ANOVA table, the variance components, the %GRR, and the ndc.

**Acceptance criteria:** the variance components are computed from the mean squares, not from a shortcut; the multiplier is stated; the %GRR is computed against a stated tolerance; the fitness decision is justified for the context.

**Time:** 90 minutes.

### Exercise 5. PPG timing test (application)

**Deliverable:** a heart-rate tracking test at 60, 90, and 120 bpm with a pulse simulator, reporting the error, the latency, and the response to a motion disturbance.

**Acceptance criteria:** the simulator waveform is described; the error is reported with a measure of spread; the latency is measured from the simulator trigger to the output; the report states that the test does not establish SpO2 accuracy.

**Time:** 60 minutes.

### Exercise 6. Critique a claim (critique)

**Deliverable:** a review of a draft claim that says "the sensor is accurate to ±0.1 °C, so the device meets the clinical requirement," identifying the missing evidence and the correct claim.

**Acceptance criteria:** the review distinguishes sensor accuracy from system accuracy and from clinical agreement; it names the reference and the population that would be needed; it proposes a corrected sentence that preserves the true finding.

**Time:** 45 minutes.

### Exercise 7. Uncertainty budget (calculation)

**Deliverable:** an uncertainty budget for the CMRR measurement, with at least five sources and a combined uncertainty in dB.

**Acceptance criteria:** the budget converts each source to dB correctly; the dominant term is named; the expanded uncertainty is reported; the budget states which sources were not included.

**Time:** 45 minutes.

### Exercise 8. Design a sensor lab (design)

**Deliverable:** a protocol for a new sensor lab, with the quantity, the reference, the range, the calibration design, the uncertainty target, and the claim boundary.

**Acceptance criteria:** the reference is at least four times more accurate than the target; the calibration design includes ascending and descending points; the acceptance criterion is stated; the claim boundary names at least three things the lab cannot establish.

**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 What is the difference between a sensor and a measurement system?

A sensor converts a physical quantity into a signal. A measurement system includes the sensor, the front end, the ADC, the algorithm, the reference, and the environment. The sensor's accuracy is one term in the system's uncertainty. A system can fail its requirement even when the sensor meets its datasheet.

### 13.2 Compute the CMRR for a differential gain of 100 and a common-mode gain of 0.0126.

The common-mode gain is the output voltage divided by the common-mode input. If the output is 12.6 mV for a 1 V common-mode input, \(A_{cm} = 0.0126\). The CMRR is:

\[
20 \log_{10}\left(\frac{100}{0.0126}\right) = 20 \log_{10}(7937) = 78.0\ \text{dB}
\]

### 13.3 Why is a 1% source mismatch a problem for a CMRR test?

A 1% mismatch in the source resistors converts a fraction of the common-mode voltage into a differential voltage. With 1 V of common-mode, the differential error is about 10 mV, which is 100 times larger than the amplifier's own common-mode error at 80 dB. The measured CMRR is then limited by the source to about 46 dB, not by the amplifier.

### 13.4 Convert 14 µV peak-to-peak to RMS for a Gaussian signal.

For a Gaussian signal, the peak-to-peak value is about 6 times the standard deviation, so the RMS is approximately \(14/6 = 2.3\ \mu\text{V}\). The exact factor depends on the crest factor and the number of samples; a report must state the convention.

### 13.5 What is hysteresis, and why does it matter in a calibration?

Hysteresis is the difference between the ascending and descending responses at the same input. It matters because a reading depends on the direction of the change, so a calibration that uses only ascending data will misstate the error on a descending measurement. The maximum hysteresis is compared with the tolerance.

### 13.6 Compute the %GRR for a standard deviation of 0.0294 °C and a tolerance width of 0.2 °C.

\[
\%GRR = \frac{6 \times 0.0294}{0.2} \times 100 = 88.2\%
\]

The system consumes 88% of the tolerance width and is not fit for a tolerance decision under the AIAG heuristic.

### 13.7 Why can a bench not establish SpO2 accuracy?

Because the conversion from the optical ratio to saturation is an empirical curve derived from human subjects with arterial blood-gas references. The bench has no blood-gas reference and no tissue. It can measure the electronics and the timing, not the curve.

### 13.8 What is the difference between repeatability and reproducibility?

Repeatability is the variation under one operator, one setup, and a short time. Reproducibility is the variation across operators or setups. A Gauge R&R separates the two and estimates the operator-by-part interaction.

### 13.9 Why is the ice point not automatically 0 °C?

Because an ice bath must be prepared with crushed ice, distilled water, good contact, stirring, and enough time to equilibrate. A bath with floating ice, tap water, or insufficient equilibration can be several tenths of a degree above 0 °C.

### 13.10 Rubric for Exercise 6

A strong critique distinguishes sensor accuracy, system accuracy, and clinical agreement, names the reference and the population needed, and proposes a corrected claim. A weak critique says "the claim is not supported" without saying what would support it.

### 13.11 Rubric for Exercise 8

A strong protocol names the quantity, the reference with its uncertainty, the range, the calibration design, the uncertainty target, and the claim boundary. It states what the lab cannot establish. A weak protocol names the sensor and the sample rate but not the reference or the claim.

### 13.12 What does a calibration certificate need?

The instrument, the parameter, the range, the results, the measurement uncertainty, the environmental conditions, the date, the traceability statement, and the identity of the calibration laboratory. A certificate without an uncertainty is not a calibration certificate under ISO/IEC 17025.

## 14. Interview questions

### 14.1 How do you calibrate a temperature sensor?

A strong answer names the reference, the traceability, the points, the ascending and descending passes, the fit, the residuals, the hysteresis, and the uncertainty budget. It notes that the ice bath must be prepared correctly and that self-heating must be estimated. A weak answer says "compare it with a thermometer."

### 14.2 What is the difference between calibration and verification?

Calibration establishes the relationship between the sensor and the reference. Verification compares the calibrated output with a requirement. A strong answer notes that a calibration can be correct and the verification can still fail, because the requirement may be tighter than the sensor's capability.

### 14.3 How would you measure CMRR?

A strong answer applies a common-mode signal to both inputs through a balanced network, measures the differential output, divides by the differential gain, and computes the ratio in dB. It notes that the source balance limits the measurable CMRR and that the measurement must be at the frequency of interest. A weak answer applies a signal to one input.

### 14.4 Why is input-referred noise more useful than output noise?

Because it allows comparison across gains and with the input signal. A strong answer divides the output noise by the gain and states the bandwidth. A weak answer quotes the output noise without the gain.

### 14.5 What does a Gauge R&R separate?

Repeatability, reproducibility, and the operator-by-part interaction. A strong answer gives the variance-component formulas and the %GRR interpretation. A weak answer says it measures "how good the gauge is."

### 14.6 Why is SpO2 calibration a clinical activity?

Because the ratio-to-saturation relationship is empirical and depends on human tissue, blood, and sensor geometry. A strong answer names the blood-gas reference, the controlled desaturation, and the population diversity requirement. A weak answer says "the manufacturer calibrates it."

### 14.7 What is the difference between ARMS and bias?

ARMS is the root-mean-square difference between the device and the reference across the range. Bias is the mean difference. A strong answer notes that a device can have a small bias and a large ARMS, or the reverse. A weak answer uses the terms interchangeably.

### 14.8 What would you do if the sensor meets its datasheet but the system fails the tolerance?

A strong answer separates the sensor from the system, identifies the dominant uncertainty term, and proposes a change to the reference, the fixture, the averaging, or the tolerance. It does not blame the sensor. A weak answer says the datasheet is wrong.

### 14.9 Why does the fit residual belong in the uncertainty budget?

Because the calibration curve does not pass exactly through the points. The residual is the part of the sensor's behaviour that the model does not explain. A strong answer includes it as a Type B term. A weak answer reports R² and assumes the curve is exact.

### 14.10 How do you test a PPG sensor without a person?

A strong answer uses a pulse simulator, a recorded waveform, or an optical filter that modulates the light path. It states that the test establishes timing and electronics, not clinical accuracy. A weak answer says "use a finger from the team."

### 14.11 What is the difference between a fixed point and a reference thermometer?

A fixed point is a physical state with a known temperature, such as the ice point. A reference thermometer is an instrument with a calibration certificate. A strong answer notes that a fixed point can be more accurate than a thermometer if it is prepared correctly, and that both have uncertainties.

### 14.12 What would you record for a sensor lab?

The sensor, the lot, the board, the reference and its certificate, the range, the points, the direction, the repeats, the environment, the raw readings, the fit, the residuals, the hysteresis, the uncertainty budget, and the limitations. A strong answer includes the raw data hash and the script version.

## 15. Key terms

- **Accuracy** — closeness to the true value.
- **AFE** — analog front end.
- **ARMS** — root-mean-square accuracy of an SpO2 measurement.
- **Bias** — mean difference between two methods.
- **Calibration** — comparison against a reference with a stated uncertainty.
- **Calibration curve** — relationship between the sensor signal and the reference.
- **CMRR** — common-mode rejection ratio.
- **Coverage factor** — multiplier for the expanded uncertainty.
- **Drift** — change in output over time at constant input.
- **Electrode offset** — half-cell potential at an electrode.
- **ENOB** — effective number of bits.
- **Gauge R&R** — repeatability and reproducibility study.
- **Hysteresis** — difference between ascending and descending responses.
- **Ice point** — equilibrium temperature of ice and water.
- **Immersion depth** — depth of a probe in the medium.
- **Input-referred noise** — noise divided by the gain.
- **Limits of agreement** — interval containing most differences between two methods.
- **Linearity** — closeness of the response to a straight line.
- **LSB** — least significant bit.
- **ndc** — number of distinct categories.
- **Noise density** — noise per square root of bandwidth.
- **Offset** — output at zero input.
- **PPG** — photoplethysmography.
- **Reference** — measurement standard with a stated uncertainty.
- **Repeatability** — variation under one setup.
- **Reproducibility** — variation across setups or operators.
- **Resolution** — smallest distinguishable increment.
- **SaO2** — arterial oxygen saturation by blood gas.
- **Self-heating** — temperature rise from the sensor's excitation.
- **Sensitivity** — change in output per change in input.
- **Sensor** — device that converts a physical quantity into a signal.
- **SpO2** — arterial oxygen saturation estimated by pulse oximetry.
- **Stem conduction** — heat flow along a probe.
- **Tolerance** — allowed variation of the product.
- **Traceability** — unbroken chain of calibrations.
- **Triple point** — temperature at which ice, water, and vapour coexist.
- **Uncertainty budget** — list of uncertainty sources and their combination.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and regulation

- IEC 60601-2-27:2011, Electrocardiographic monitoring equipment. https://webstore.iec.ch/en/publication/2638
- ANSI/AAMI/IEC 60601-2-27:2011/(R)2016. https://aami.org/standard/ansi-aami-iec-60601-2-272011-r2016/
- ANSI/AAMI EC11:1991 (R2007), Diagnostic Electrocardiographic Devices. https://webstore.ansi.org/standards/aami/ansiaamiec111991r2007
- ISO 80601-2-61, Pulse oximeters. https://www.iso.org/
- ISO 80601-2-56, Clinical thermometers for body temperature measurement. https://www.iso.org/standard/69448.html
- ASTM E1112, Standard Specification for Electronic Thermometer for Intermittent Determination of Patient Temperature. https://www.astm.org/e1112-00r11.html
- 21 CFR 880.2910, Clinical electronic thermometer. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-880/subpart-C/section-880.2910
- FDA, Pulse Oximeters — Premarket Notification Submissions [510(k)s]: Guidance for Industry and Food and Drug Administration Staff. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pulse-oximeters-premarket-notification-submissions-510ks-guidance-industry-and-food-and-drug
- ISO/IEC 17025:2017, General requirements for the competence of testing and calibration laboratories. https://www.iso.org/standard/66912.html
- JCGM 100:2008, Guide to the expression of uncertainty in measurement. https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf

### Datasheets and methods

- Analog Devices, AD8232 datasheet. https://www.analog.com/media/en/technical-documentation/data-sheets/AD8232.pdf
- Texas Instruments, TMP117 datasheet, SNOSD82D. https://www.ti.com/lit/ds/symlink/tmp117.pdf
- Texas Instruments, NIST Traceability for Temperature and Humidity Sensors, SBAT024. https://www.ti.com/lit/po/sbat024/sbat024.pdf
- Analog Devices, MAX30102 datasheet. https://www.analog.com/media/en/technical-documentation/data-sheets/max30102.pdf
- AIAG, Measurement Systems Analysis Reference Manual, 4th edition (2010).
- MHRA, The use and regulation of pulse oximeters (information for healthcare professionals). https://www.gov.uk/guidance/the-use-and-regulation-of-pulse-oximeters-information-for-healthcare-professionals

> **Note:** The standard texts are paywalled. The datasheets and guidance above were used to check the teaching values. Confirm every clause and limit against the current edition before citing a number in a regulatory document.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_lab-conventions.md`](03_lab-conventions.md)
- Next topic: [`05_firmware-labs.md`](05_firmware-labs.md)
- Sibling: [`02_bench-and-simulation-stack.md`](02_bench-and-simulation-stack.md)
- Sibling: [`07_hil-labs.md`](07_hil-labs.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
