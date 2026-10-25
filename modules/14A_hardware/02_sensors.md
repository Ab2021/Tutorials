# Sensors

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 02 of 14
> Estimated study time: 4 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

Every medical measurement begins with a physical interaction. An electrode converts an ionic potential into an electronic one. A photodiode converts light into current. A diaphragm converts pressure into strain. A thermistor converts temperature into resistance. The sensor is where the clinical quantity first becomes a signal, and it is where the largest and least visible errors enter the chain.

The industry problem is that sensor specifications are written for engineers and read by data scientists as if they were guarantees. A datasheet says "accuracy ±3%" and the data scientist assumes that every value is within 3% of truth. The specification usually means something narrower: a root-mean-square difference across a study population, under controlled conditions, with a particular reference method, excluding motion, low perfusion, and interfering substances. The gap between the specification and the clinical reality is where device performance is lost.

Three decisions depend on sensor literacy:

1. **Whether a measurement is fit for the intended decision.** A pulse oximeter with a 3% root-mean-square error may be adequate for triage and inadequate for titrating oxygen in a patient with a target range of 88–92%. The same number supports different decisions.
2. **How to stratify and interpret error.** A sensor that performs well on average may fail a subgroup because of skin tone, perfusion, body size, or ambient light. A pooled accuracy number hides the subgroup that matters.
3. **What to record and what to ask for.** Sensor type, site, lot, calibration, and signal-quality flags are not metadata decoration. They are the variables that explain why two records of the same physiology differ.

This topic covers the main sensing principles used in medical devices: biopotential electrodes, optical sensors, pressure and flow sensors, temperature sensors, motion sensors, chemical and biochemical sensors, and imaging detectors. It does not attempt to make the reader a sensor physicist. It builds the ability to read a sensor specification, identify its assumptions, predict its failure modes, and design a validation that would reveal them.

A useful framing: a sensor is a transducer plus a set of promises. The transducer converts energy; the promises define the conditions under which the conversion is accurate. Most of this topic is about the promises.

The commercial context reinforces the point. Sensor cost is often a small fraction of device cost, but sensor error can dominate clinical error. A team that spends its engineering budget on the user interface and the model while accepting the sensor's typical specification has optimised the visible part of the product and left the invisible part unmanaged. The failure appears later, as a subgroup gap, a field complaint, or a validation that cannot be reproduced.

## 2. Learning outcomes

By the end of this topic you can:

- Match a clinical measurand to a sensing principle and state why the alternatives are worse for that application.
- Read a sensor datasheet and separate sensitivity, resolution, accuracy, drift, and cross-sensitivity, with units and test conditions for each.
- Compute the noise floor of an optical or resistive sensor from first principles and compare it with the signal of interest.
- Explain the assumptions behind pulse oximetry and continuous glucose monitoring, and predict how skin tone, perfusion, motion, and lag change the reported value.
- Design a calibration and drift-check plan for a sensor that cannot be recalibrated in the field.
- Identify the sensor-level causes of a subgroup performance gap and propose a stratified validation.
- Evaluate the biocompatibility and sterilization constraints that change sensor performance over the product life.
- Critique a claim that a sensor "measures" a quantity when it estimates it through a model.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A sensor is a map from a physical domain to an electrical domain, with a sensitivity, a range, a bandwidth, and an error model. The map is rarely one-to-one. It is usually a response to one quantity that is also sensitive to several others, and the device's job is to make the intended quantity dominate.

```text
   PHYSICAL DOMAIN                       ELECTRICAL DOMAIN
   ---------------                       -----------------
   quantity x(t)  ---->  transducer  ---->  y(t) = S*x(t) + b + n(t) + sum_i C_i*z_i(t)
                         (sensitivity S,
                          offset b,
                          noise n,
                          cross-sensitivities C_i to confounders z_i)
        |                                          |
        |                                          |
   confounders:                             conditioning:
   temperature, humidity,                   bridge, bias, TIA,
   motion, light, flow,                     filtering, ADC
   interferents, age
```

The equation in the diagram is the whole topic in one line:

\[
y(t) = S\,x(t) + b + n(t) + \sum_i C_i\,z_i(t)
\]

- \(S\) is the sensitivity, with a tolerance and a temperature coefficient.
- \(b\) is the offset, which may drift with time, temperature, and sterilization.
- \(n(t)\) is the noise, which sets the resolution.
- The sum is the cross-sensitivity to confounders: temperature, humidity, motion, ambient light, interferents, and ageing.

A good sensor design makes \(S\) large, \(b\) stable, \(n\) small, and \(C_i\) small for every confounder that varies in use. A good validation measures each term separately, because a pooled accuracy number mixes them.

### 3.1 The six sensing principles

Most medical sensors use one of six physical mechanisms. The mechanism determines the noise floor, the bandwidth, the cross-sensitivities, and the failure modes.

| Principle | Physical effect | Medical examples | Dominant noise | Characteristic failure |
|---|---|---|---|---|
| Electrochemical | Charge transfer at an electrode-electrolyte interface | ECG, EEG, EMG, glucose oxidase sensors, ISFET | Electrode impedance, flicker noise | Offset, drift, contact loss |
| Optical | Absorption, scattering, fluorescence, or emission of light | Pulse oximetry, PPG, fluorescence immunoassay, imaging | Shot noise, ambient light, detector dark current | Motion, skin tone, ambient light, perfusion |
| Piezoresistive | Resistance change with strain | Pressure catheters, bridge sensors, strain gauges | Bridge thermal noise, 1/f noise | Offset drift, hysteresis, temperature |
| Capacitive / MEMS | Capacitance change with displacement | Pressure sensors, accelerometers, microphones | Brownian noise, charge noise | Humidity, stiction, offset |
| Piezoelectric | Charge generation under stress | Ultrasound transducers, contact microphones | Thermal-mechanical noise | Coupling, bandwidth, depoling |
| Thermal | Heat transport or temperature-dependent resistance | Thermistors, RTDs, thermocouples, flow sensors | Johnson noise, self-heating | Site, equilibration, perfusion |

The principle is not a detail. An optical sensor and an electrochemical sensor measuring the same nominal quantity have different error structures, different calibration models, and different failure modes. A data scientist who pools them because the column name is the same will build a model that learns the device rather than the physiology.

### 3.2 The sensor error stack

Errors add in layers. At the bottom is the fundamental noise of the transduction. Above it is the conditioning noise. Above that is the calibration uncertainty. Above that is the physiological and environmental variability that the sensor cannot distinguish from the signal.

| Layer | Example | Can be reduced by |
|---|---|---|
| Fundamental transduction noise | Shot noise in a photodiode | More light, longer integration, larger area |
| Conditioning noise | Amplifier voltage noise, resistor thermal noise | Better amplifier, lower impedance, more gain |
| Quantisation | ADC step | More bits or more gain, within range |
| Calibration uncertainty | Reference method error, curve fit | Better reference, more calibration points |
| Drift | Electrode ageing, LED degradation | Periodic recalibration, reference channel |
| Physiological confounders | Venous pulsation, skin tone, motion | Better signal processing, multiple wavelengths, quality indices |
| Environmental confounders | Ambient light, temperature, humidity | Shielding, compensation, controlled use |

The layers are not equally important. In a well-designed optical sensor, the fundamental shot noise may be 20 dB below the motion artefact; reducing shot noise further buys nothing. In a poorly designed electrode, the amplifier noise may dominate the electrode noise; improving the electrode buys nothing. The first task is to identify the dominant layer.

### 3.3 What "accuracy" means for a sensor

Accuracy is not one number. For a sensor, at least five different quantities are commonly reported:

| Quantity | Definition | What it answers |
|---|---|---|
| Sensitivity | Change in output per unit input | How large is the signal? |
| Resolution | Smallest detectable change | How small a change can be seen? |
| Precision | Spread of repeated measurements | How repeatable is the value? |
| Bias | Mean difference from reference | Is the value systematically off? |
| Accuracy (often A<sub>rms</sub>) | Root-mean-square difference from reference | How far from truth, on average, in a population? |
| Total error | Worst-case combination of bias and variability | What is the guaranteed bound? |

A sensor can have good precision and poor accuracy, or good accuracy on average and poor performance in a subgroup. The pulse oximeter case in Section 9 is the canonical example: a pooled A<sub>rms</sub> of about 4% can hide a larger error in a subgroup.

### 3.4 The calibration model is part of the sensor

Many sensors do not measure the clinical quantity directly. A pulse oximeter measures a ratio of optical absorptions and maps it to oxygen saturation through an empirical calibration curve derived from human studies. A continuous glucose monitor measures a current from an enzymatic reaction and maps it to interstitial glucose through a calibration model. A blood-pressure cuff measures cuff pressure oscillations and maps them to systolic and diastolic values through an envelope algorithm.

In each case, the sensor output is an input to a model, and the model has its own assumptions and its own error. The clinical value is the output of the whole chain, not the transducer alone. A data scientist who treats the clinical value as a raw measurement will misattribute model error to physiology.

### 3.5 A sensor selection scorecard

A selection decision is easier to defend when the criteria are written before the candidates are compared. The scorecard below is a starting point; the weights depend on the application.

| Criterion | Question | Weight in a wearable | Weight in an implant |
|---|---|---:|---:|
| Clinical validity | Does the principle measure the intended quantity? | High | High |
| Accuracy in the target population | Has it been validated in the relevant subgroups? | High | High |
| Sensitivity and resolution | Is the signal large enough for the noise floor? | High | Medium |
| Cross-sensitivity | How many confounders must be compensated? | High | High |
| Power | Does it fit the energy budget? | High | High |
| Size and mass | Does it fit the form factor? | Medium | High |
| Biocompatibility | Is the material acceptable for the contact duration? | Medium | High |
| Sterilization | Does it survive the reprocessing method? | Medium | High |
| Calibration burden | Can the calibration be maintained in use? | High | High |
| Cost and supply | Is it available for the product life? | Medium | High |
| Regulatory precedent | Is there a cleared predicate or a standard? | High | High |

The scorecard is not a substitute for engineering. It is a way to make the trade-offs explicit, so that a later reviewer can see why a sensor was chosen and what was sacrificed. A decision record that says "we chose the optical sensor because it was smaller" is not a decision record; it is a preference.

### 3.6 Three questions before trusting a sensor

Before using a sensor value in an analysis, ask:

1. **What intermediate quantity does the sensor actually measure?** If the answer is "the clinical quantity", the answer is probably wrong. A pulse oximeter measures an optical ratio; a CGM measures a current; a cuff measures pressure oscillations. The intermediate quantity tells you which confounders matter.
2. **Which confounders vary on the timescale of the measurement?** A confounder that is constant within a patient and a session can be absorbed by calibration. A confounder that changes within a beat or a minute cannot.
3. **Who was in the calibration and validation cohort?** If the cohort does not include the patient in front of you, the accuracy claim is an extrapolation. The size of the extrapolation is a clinical judgement, not a statistical one.

The three questions do not replace a validation. They tell you which validation to ask for.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Sensor | Device that converts a physical quantity into a signal, including packaging and conditioning | Used interchangeably with transducer | None |
| Transducer | Element that performs the energy conversion | Called a sensor | None |
| Measurand | The quantity intended to be measured | Confused with the sensor output | Physical unit |
| Sensitivity | Output change per unit input change | Confused with resolution | V/unit, Ω/°C, A/W |
| Resolution | Smallest change that produces a detectable output change | Confused with accuracy | Unit of measurand |
| Accuracy | Closeness to the true value | Used as a synonym for precision | Unit or % |
| Precision | Closeness of repeated measurements | Used as a synonym for accuracy | Standard deviation |
| Bias | Systematic difference from a reference | Ignored in pooled accuracy | Unit or % |
| Drift | Change in output over time at constant input | Confused with noise | Unit/year, %/year |
| Offset | Output at zero input | Confused with drift | Unit or V |
| Hysteresis | Difference between increasing and decreasing input paths | Ignored in calibration | % of full scale |
| Linearity | Deviation from a straight-line response | Confused with accuracy | % of full scale |
| Cross-sensitivity | Response to a non-target quantity | Ignored in validation | Unit per unit confounder |
| Selectivity | Ability to respond only to the target | Confused with sensitivity | Ratio |
| Time constant | Time to reach 63% of a step response | Confused with response time | s |
| Response time | Time to reach a stated fraction of a step | Quoted without the fraction | s |
| Bandwidth | Frequency range over which the sensor responds | Confused with sampling rate | Hz |
| Noise floor | Noise power in the absence of signal | Confused with resolution | Unit RMS or density |
| Shot noise | Noise from discrete charge carriers | Ignored in optical sensors | A RMS |
| Dark current | Current in a photodetector with no light | Ignored in low-light designs | A |
| Responsivity | Photocurrent per unit optical power | Confused with quantum efficiency | A/W |
| Perfusion index | Pulsatile component as a fraction of the DC optical signal | Used as a universal quality metric | % |
| A<sub>rms</sub> | Root-mean-square difference from a reference | Treated as a maximum error | % or unit |
| MARD | Mean absolute relative difference | Treated as a complete accuracy measure | % |
| Lag | Delay between a change in the true quantity and the reported value | Ignored in alignment | s or min |
| Calibration | Establishing the mapping from output to reference values | Treated as a one-time factory step | Procedure and date |
| Recalibration | Periodic re-establishment of the mapping | Assumed unnecessary for sealed sensors | Interval |
| Biocompatibility | Compatibility of a material with the body | Confused with sterility | ISO 10993 category |
| Sterilization | Killing or removing microorganisms | Confused with disinfection | Method and dose |
| Shelf life | Time over which the sensor remains within specification | Confused with use life | Months |
| Use life | Time over which the sensor performs after opening or implantation | Confused with shelf life | Hours to years |
| Cross-talk | Interference between channels or sensors | Confused with cross-sensitivity | dB or % |
| Artefact | Signal component not arising from the measurand | Used loosely for any error | None |
| Signal quality index | A computed measure of artefact or reliability | Treated as a diagnosis | Unitless |
| Reference method | The comparator used to establish accuracy | Assumed error-free | Method and uncertainty |

## 5. The mechanics, step by step

The procedure below is a sensor selection and characterisation method. It applies to a new sensor design and to the evaluation of a sensor already inside a device. Each step states its input, output, and rationale. The edge cases required by the assignment are mapped to the steps and summarised in Section 5.13.

### Step 1. Define the measurand and the clinical decision

**Input:** the clinical question, the decision it supports, and the population.

**Output:** a requirement with range, resolution, accuracy, response time, and operating conditions.

**Rationale:** the same physical quantity can require different sensors at different performance levels. Heart rate for a fitness band can tolerate a 5 bpm error averaged over 30 s; heart rate for a neonatal monitor may require beat-to-beat detection with a much shorter delay. The decision determines the requirement, and the requirement determines the sensor.

Write the requirement in the form: quantity, range, resolution, accuracy with a statistical definition, response time, and environment. For example: "SpO₂ 70–100%, resolution 1%, A<sub>rms</sub> ≤ 3.0% against arterial blood gas, response time ≤ 30 s, operating temperature 15–40 °C, motion present." A requirement without a statistical definition is not testable.

### Step 2. Map the physics to an observable

**Input:** the measurand and the available physical effects.

**Output:** a candidate sensing principle and the intermediate quantity that is actually measured.

**Rationale:** almost no medical sensor measures the clinical quantity directly. It measures an intermediate quantity that is related to the clinical quantity through physics and a calibration model. For SpO₂, the intermediate is the ratio of pulsatile to steady optical absorption at two wavelengths. For glucose, it is an enzymatic current. For blood pressure, it is a pressure or an oscillation amplitude. Naming the intermediate quantity is the first step toward naming the confounders.

**Branch point:** if the intermediate quantity is also affected by a confounder that varies in use, the sensor needs a second channel, a reference measurement, or a compensation model. A single-wavelength optical sensor cannot distinguish oxygen saturation from blood volume; a two-wavelength sensor can, under assumptions.

### Step 3. Choose the sensing principle

**Input:** the candidate principles, the requirement, and the environment.

**Output:** a selected principle with a written justification and the rejected alternatives.

**Rationale:** the choice is a trade across sensitivity, noise, bandwidth, power, size, biocompatibility, and cost. The decision table below summarises the common trades.

| Requirement | Electrochemical | Optical | Piezoresistive | Capacitive MEMS | Piezoelectric | Thermal |
|---|---|---|---|---|---|---|
| Direct biopotential | Best | No | No | No | No | No |
| Oxygen saturation | No | Best | No | No | No | No |
| Invasive pressure | Possible | Possible | Best | Good | No | No |
| Airway flow | No | Possible | Good | Good | Possible | Good |
| Temperature | No | IR only | No | No | No | Best |
| Motion | No | Possible | Possible | Best | No | No |
| Implantable glucose | Best | Possible | No | No | No | No |
| Low power | Good | Moderate | Good | Good | Good | Moderate |
| Sterilizable | Material-dependent | Window-dependent | Good | Good | Good | Good |

A principle that is best on one row may be unacceptable on another. An optical sensor is the best choice for SpO₂ but a poor choice for an implantable glucose sensor that must operate for months in fibrous tissue. A piezoresistive sensor is excellent for pressure but sensitive to temperature and hysteresis.

### Step 4. Characterise the transfer function

**Input:** a prototype sensor and a reference stimulus.

**Output:** sensitivity with tolerance, offset, linearity, hysteresis, and the operating range.

**Rationale:** the transfer function is the sensor's contract. It must be measured, not assumed from the datasheet, because the datasheet's conditions may differ from the application. Measure at least five points across the range, in both directions, at three temperatures. Fit a line or a polynomial and record the residuals. A sensor with a 1% nonlinearity cannot support a 0.5% accuracy claim without a nonlinear calibration.

For a pressure sensor, the transfer function includes the effects of the tubing, the fluid, and the stopcock. For an optical sensor, it includes the LED spectrum, the detector responsivity, and the tissue optics. The sensor is a system, not a component.

### Step 5. Characterise cross-sensitivity

**Input:** the list of confounders from Step 2.

**Output:** a cross-sensitivity coefficient for each confounder, with the range over which it was tested.

**Rationale:** cross-sensitivity is the error that survives calibration. A pressure sensor calibrated at 25 °C may drift by 0.2% per °C; over a 15–40 °C range, that is a 5% error. A pulse oximeter calibrated on light skin may have a different calibration on dark skin. A glucose sensor responds to acetaminophen as well as glucose. Each coefficient must be measured against a changing confounder while the measurand is held constant.

The test matrix is the product of the confounders and their ranges. It is expensive, which is why it is often incomplete. A data scientist should ask which confounders were tested and which were assumed.

### Step 6. Measure the noise floor and resolution

**Input:** a quiet stimulus at a fixed value and a recording of the output.

**Output:** noise spectral density, RMS noise in the signal band, and the smallest detectable change.

**Rationale:** resolution is not the same as the number of bits in the datasheet. It is the smallest change that can be distinguished from noise with a stated confidence. For a photodiode, the noise floor is dominated by shot noise, which depends on the DC photocurrent. For a bridge, it is thermal noise from the bridge resistance. For an electrode, it is the electrode-electrolyte noise plus the amplifier noise.

Measure the noise with the sensor in its final packaging and at its final bandwidth. A noise measurement on an open bench, with a different cable and no tissue, can be optimistic by a factor of two or more.

### Step 7. Characterise the dynamic response

**Input:** a step or swept stimulus.

**Output:** time constant, response time, bandwidth, and group delay.

**Rationale:** a sensor that is accurate at steady state may be wrong during change. A thermistor has a thermal time constant of seconds; a glucose sensor has a diffusion lag of minutes; a pressure sensor can have a resonance in the kilohertz range. The dynamic response matters whenever the clinical decision depends on a trend or an alarm.

For a first-order sensor, the time constant is the time to reach 63% of a step; the 90% time is 2.3 times the time constant. A sensor with a 10 s time constant cannot support a 5 s alarm delay, regardless of the electronics.

### Step 8. Define the calibration model and its uncertainty

**Input:** the transfer function, the cross-sensitivities, and a reference method.

**Output:** a calibration equation, a calibration interval, and an uncertainty budget.

**Rationale:** the calibration model converts the sensor output to the clinical quantity. It has coefficients with uncertainty, and it may have a nonlinear form. The uncertainty budget must include the reference method, the fit residuals, the drift between calibrations, and the cross-sensitivity that is not compensated.

A sensor that is calibrated once at the factory and never again needs a drift budget that covers the entire service life. A sensor that is calibrated by the user needs a procedure that a user can follow correctly, and a check that detects a failed calibration.

### Step 9. Design the interface and the sampling

**Input:** the sensor output impedance, the signal bandwidth, and the converter.

**Output:** a conditioning circuit and a sampling rate.

**Rationale:** the sensor's output impedance determines the conditioning. A high-impedance electrode needs a buffer; a low-impedance bridge needs a bridge amplifier; a photodiode needs a transimpedance amplifier. The sampling rate must exceed twice the highest frequency the conditioning passes, and the anti-alias filter must precede the converter.

For an optical sensor, the interface includes the LED driver. The LED current determines the optical power, the photocurrent, and the shot noise. Increasing the current improves the SNR until the detector or the tissue saturates, or until the power budget or the thermal limit is reached.

### Step 10. Assess biocompatibility, sterilization, and shelf life

**Input:** the intended contact duration and tissue, and the reprocessing method.

**Output:** a material selection, a sterilization validation plan, and a shelf-life claim.

**Rationale:** a sensor that contacts the patient must be biocompatible for the contact category and must survive the sterilization method. Both can change the sensor's performance. Autoclaving can change a polymer's modulus, gamma irradiation can discolour an optical window, ethylene oxide can leave residues, and repeated cleaning can degrade an electrode. The performance specification must be verified after the worst-case sterilization and ageing.

ISO 10993-1 provides the framework for biological evaluation by contact category and duration. The sensor's materials, not only its electronics, determine the evidence required.

### Step 11. Verify and validate

**Input:** the characterised sensor and the requirement.

**Output:** verification evidence against the specification and validation evidence in the intended population.

**Rationale:** verification tests the sensor against its written specification in the laboratory. Validation tests whether the sensor supports the clinical decision in the intended population and environment. A sensor can pass verification and fail validation because the specification omitted a confounder. The validation must be stratified by the subgroups that the confounders predict: skin tone, perfusion, age, body size, and ambient conditions.

### Step 12. Plan the lifecycle

**Input:** the field performance data and the failure history.

**Output:** a monitoring plan, a recalibration or replacement interval, and a change-control procedure.

**Rationale:** sensors age. LEDs degrade, electrodes dry, membranes foul, and references drift. The lifecycle plan defines when the sensor is no longer within specification and how that is detected. It also defines what happens when a sensor lot changes or a supplier changes a material: the calibration may change, and the change must be verified before it reaches patients.

### 5.13 Edge cases and where they enter

| Edge case | Sensor context | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | Photodiode TIA or ADC | DC optical term plus gain exceeds rail | Output at rail; flat-topped PPG | Loss of pulsatile signal |
| Common-mode rejection failure | Electrodes | Impedance mismatch converts common-mode to differential | 50/60 Hz tone; CMRR test | Interference mistaken for signal |
| Electrode offset | Biopotential electrodes | Half-cell potential, up to ±300 mV | Open-circuit voltage; input range check | Saturation; low-gain compromise |
| Clock jitter | Sampled optical or pressure signals | Aperture error in the converter | SNR versus input frequency | Degraded high-frequency content |
| Brown-out | LED driver or processor | Supply drop; LED current collapses | Driver fault flag; optical output drop | Invalid readings; reset |
| Watchdog reset loop | Sensor hub firmware | Repeated resets; lost samples | Reset counter; boot log | Data gaps; device unavailable |
| Thermal throttling | High-power optical or compute module | LED or processor reduces output to limit temperature | Temperature log; output drift | Reduced SNR; missed deadlines |
| Firmware update failure | Sensor hub | Incomplete write; calibration coefficients lost | Version and checksum check | Wrong calibration; device bricked |
| Calibration drift | All sensors | Ageing, fouling, membrane degradation | Reference check; trend of zero | Systematic bias in all values |

The table is a bridge between this topic and the rest of the module. A sensor failure rarely presents as a sensor failure; it presents as a data problem, a model problem, or a clinical problem. Naming the mechanism is what allows the right team to fix it.

### 5.14 Decision table: when a sensor needs a second channel

| Condition | Single channel sufficient? | Second channel or reference | Why |
|---|---|---|---|
| Confounder is stable and known | Yes | No | Calibration can absorb it |
| Confounder varies within a measurement | No | Yes, measure the confounder | Compensation requires an estimate |
| Two analytes with overlapping response | No | Yes, a selective channel or separation | A single response cannot distinguish them |
| Motion artefact dominates | No | Yes, an accelerometer or optical reference | Motion is not physiological signal |
| Ambient light varies | No | Yes, a dark or reference channel | Ambient adds to the signal |
| LED output degrades | No | Yes, a monitor photodiode | Source drift mimics signal drift |
| Tissue contact varies | No | Yes, an impedance or contact sensor | Contact changes the optical path |
| Reference method is intermittent | Yes, with interpolation | No, if the drift is slow | Calibration between references |

The pattern is general: if a confounder varies on the timescale of the measurement and cannot be held constant, the sensor needs a way to observe it. A calibration curve cannot compensate for a variable it never sees.

### 5.15 Principle-specific mechanics

The twelve steps are generic. The following subsections give the mechanism and the numbers for the principles a biomedical data scientist is most likely to encounter.

#### 5.15.1 Biopotential electrodes

An electrode is an electrochemical half-cell. At the metal-electrolyte interface, charge is transferred by oxidation or reduction, and the potential across the interface depends on the electrode material, the electrolyte, and the current. A Ag/AgCl electrode with a hydrogel is the clinical standard because its half-cell potential is stable and its interface is reversible. A dry electrode replaces the gel with a conductive polymer or a metal contact; it is reusable and comfortable, but its contact impedance is higher and more variable.

| Electrode type | Typical contact impedance at 10 Hz | Offset | Main failure |
|---|---:|---:|---|
| Ag/AgCl with hydrogel | 5–20 kΩ | ±300 mV possible | Drying, motion, skin irritation |
| Dry conductive polymer | 50–500 kΩ | Larger and less stable | Motion, contact loss |
| Capacitive (insulated) | >1 MΩ at low frequency | Set by amplifier bias | Motion, gap, low-frequency cutoff |
| Textile | 10–100 kΩ, variable | Variable | Washing, fibre compression |

The contact impedance is not a nuisance parameter. It converts the amplifier's current noise into voltage noise, it forms a divider with the amplifier's input impedance, and it changes with motion. A dry electrode at 500 kΩ with an amplifier current noise of 1 pA/√Hz produces 500 nV/√Hz, which over a 100 Hz bandwidth is 5 µV RMS, comparable to the ECG signal itself.

The electrode offset is a half-cell potential plus a skin potential. It can exceed 300 mV, which is why IEC 60601-2-25 requires an electrocardiograph to tolerate ±300 mV while recording a ±5 mV signal. The offset is not a biological signal; it is an electrochemical state. It drifts with temperature, sweat, and gel dehydration.

#### 5.15.2 Optical sensors

The optical chain is: source, tissue, detector, conditioning. Each element has a noise contribution and a drift mechanism.

**Source.** An LED's radiant flux is approximately proportional to current, with a temperature coefficient of about −0.1% to −0.3% per °C at constant current, depending on the material. The peak wavelength also shifts with temperature, typically 0.05–0.2 nm/°C. For a two-wavelength oximeter, the shift changes the effective absorption coefficient and therefore the calibration. The LED's output degrades with age; a monitor photodiode or a periodic reference measurement detects the change.

**Tissue.** The optical path includes skin, fat, bone, and blood. Melanin absorbs strongly at short wavelengths and less at near-infrared. At 525 nm, the absorption by melanin is much larger than at 660 or 940 nm, which is why green-wavelength PPG is more sensitive to skin tone than infrared. The pulsatile component comes from arterial volume changes; the steady component comes from tissue, venous blood, and bone. The ratio of pulsatile to steady is the perfusion index.

**Detector.** A photodiode's responsivity depends on wavelength and temperature. Its dark current doubles roughly every 8–10 °C, which matters in a warm device. Shot noise from the DC photocurrent sets the fundamental optical noise floor. A transimpedance amplifier converts current to voltage; its feedback resistor contributes thermal noise, and its input bias current contributes a DC offset.

**Ambient light.** Room light, surgical lights, and sunlight add a DC and a modulated component. A fluorescent or LED source modulated at 100 or 120 Hz can fall inside the signal band after aliasing if the sampling rate is low. The standard mitigations are optical filtering, modulated LED drive with synchronous detection, and a dark channel that measures the ambient contribution when the LED is off.

**Ratio-of-ratios.** Pulse oximetry estimates the ratio:

\[
R = \frac{(\text{AC}/\text{DC})_{\text{red}}}{(\text{AC}/\text{DC})_{\text{IR}}}
\]

and maps \(R\) to SpO₂ through an empirical calibration curve. The curve is derived from human desaturation studies, not from first principles. Its shape depends on the wavelengths, the optical geometry, and the population. A change in any of these changes the curve, which is why a new sensor geometry requires a new clinical calibration.

#### 5.15.3 Pressure and flow sensors

A piezoresistive pressure sensor is a Wheatstone bridge on a diaphragm. The bridge output for a given pressure is:

\[
V_{\text{out}} = V_{\text{ex}} \times S \times P
\]

where \(V_{\text{ex}}\) is the excitation voltage, \(S\) is the sensitivity in mV/V per unit pressure, and \(P\) is the applied pressure. For a 5 V excitation and a 10 mV full-scale output at 300 mmHg, the sensitivity is 33.3 µV/mmHg. The bridge resistance of 350 Ω produces 2.45 nV/√Hz of thermal noise; over a 100 Hz bandwidth, that is 24.5 nV RMS, or 0.0007 mmHg. The noise is not the limit. The limits are offset drift, temperature coefficient, hysteresis, and the mechanical interface.

A pressure channel needs a zeroing step. The zero is established with the transducer open to atmosphere at the level of the patient. If the transducer is below or above the patient, a hydrostatic offset of 0.74 mmHg per centimetre of water column appears. A 10 cm height difference is 7.4 mmHg, which is clinically significant for a mean arterial pressure. The zeroing procedure and the transducer height are part of the measurement, not setup details.

Flow is often measured as a differential pressure across a restriction. The relationship is square-law:

\[
Q = k \sqrt{\Delta P}
\]

so the sensitivity falls at low flow and the square-root inversion amplifies noise near zero. A ventilator's flow sensor can drift silently and change the delivered tidal volume. Condensation in the tubing, secretions on the sensor, and a partial obstruction change the restriction coefficient \(k\), which is why some devices use a fixed-orifice design and others use a hot-wire or ultrasonic principle.

#### 5.15.4 Chemical and biochemical sensors

An enzymatic glucose sensor measures a current produced by the oxidation of glucose at an electrode. The current is limited by diffusion at high glucose and by enzyme kinetics at low glucose, following a Michaelis-Menten-like curve. The sensor's output is therefore nonlinear, and the calibration is a curve, not a single sensitivity.

Three error mechanisms dominate:

1. **Lag.** Glucose must diffuse from blood to interstitial fluid, react with the enzyme, and be processed by the electronics. The physiological component is about 5–10 minutes, the sensor reaction about 1–2 minutes, and the filtering about 3–9 minutes, giving a total of roughly 8–15 minutes in commercial devices.
2. **Fouling.** Proteins and cells adsorb on the membrane, reducing the diffusion rate. The sensitivity falls over the wear period, which is why many sensors require periodic calibration or have a factory calibration that assumes a known degradation curve.
3. **Interferents.** Acetaminophen, ascorbic acid, and other electroactive substances can be oxidized at the electrode and add to the current. The magnitude depends on the sensor's membrane and applied potential.

An ISFET measures pH or ion concentration through a field-effect transistor with an ion-selective membrane. Its advantage is small size and fast response; its weaknesses are drift, encapsulation, and sensitivity to light and temperature.

#### 5.15.5 Temperature and motion

A negative-temperature-coefficient thermistor is characterized by the Steinhart-Hart equation:

\[
\frac{1}{T} = A + B \ln R + C (\ln R)^3
\]

A 10 kΩ NTC thermistor with a beta of 3,950 K has a sensitivity of about −4.4%/°C at 25 °C, which is large compared with the measurement noise. The limits are self-heating, equilibration time, and site. A thermistor with 1 mW of self-heating in still air can read 0.1–0.5 °C high. A probe that is not in good contact with the tissue reads a mixture of tissue and ambient temperature.

An inertial measurement unit combines an accelerometer, a gyroscope, and sometimes a magnetometer. The accelerometer measures specific force, including gravity; the gyroscope measures angular rate with a bias that drifts; the magnetometer measures the earth's field but is disturbed by nearby metal and currents. Sensor fusion estimates orientation by combining the three, but the fusion assumes a known initial orientation and a stable magnetic environment. In a hospital, the magnetic environment is not stable.

### 5.16 Calibration in practice

Calibration is the process that turns a sensor output into a clinical quantity. There are five common models, and each has a different failure mode.

| Model | How it works | Failure mode | Evidence required |
|---|---|---|---|
| Factory calibration | Coefficients fixed at manufacture | Lot-to-lot variation; ageing | Process validation; sample testing |
| User calibration | User applies a known reference | User error; wrong reference | Usability validation; error detection |
| Automatic calibration | Device self-calibrates at startup or periodically | Drift between calibrations; hidden failure | Built-in reference; fault detection |
| Reference channel | A second sensor measures a confounder | Reference drift; added cost | Cross-calibration; stability data |
| Model-based calibration | A physiological model infers the quantity | Model misspecification; population bias | Clinical validation; subgroup analysis |

A two-point calibration determines a line; a multipoint calibration determines a curve and reveals nonlinearity. For a pressure channel, a five-point calibration from 0 to 300 mmHg, repeated in increasing and decreasing directions, gives the slope, the intercept, the hysteresis, and the residuals. If the residuals exceed the requirement, the channel needs a nonlinear fit or a hardware change.

The uncertainty of the calibrated value includes the reference, the fit, the drift, and the resolution. A pressure channel calibrated against a 0.05% reference, with 0.02% repeatability, 0.01% resolution, and 0.1% drift over the service interval, has a combined standard uncertainty of about 0.115% of reading and an expanded uncertainty of 0.23% at 95% coverage. If the requirement is 1%, the margin is a factor of four; if it is 0.2%, the channel fails.

The calibration interval is a risk decision, not a manufacturing convenience. A sensor that drifts 0.1% per year and is calibrated every two years has a 0.2% drift budget before any other error. Halving the interval halves the drift term but doubles the service cost. The trade should be explicit in the risk file.

### 5.17 Designing the validation study

The validation study is where the sensor's promises are tested against the population it will serve. A useful design has six elements:

1. **A reference method with known uncertainty.** The reference must be better than the sensor by a factor that makes the comparison meaningful. A pulse oximeter compared with another pulse oximeter measures agreement, not accuracy.
2. **A representative population, stratified by the confounders.** For SpO₂, the strata include skin tone, perfusion, and oxygen saturation range. For CGM, they include diabetes type, glucose range, and rate of change.
3. **Paired measurements at the same time.** A comparison of means across groups is not a validation; it is a group-level association.
4. **A pre-specified analysis.** Define A<sub>rms</sub>, bias, limits of agreement, and subgroup criteria before seeing the data. A post hoc subgroup analysis is a hypothesis, not evidence.
5. **A sample size that supports the subgroup claims.** A study with 20 participants cannot support a claim about a subgroup that is 10% of the population.
6. **A data-management plan that preserves the raw signal and the device configuration.** Without these, a discrepant result cannot be investigated.

The FDA's January 2025 draft pulse oximeter guidance illustrates the scale: it recommends at least 150 participants, at least 3,000 paired SpO₂/SaO₂ observations, at least 30% of observations in each saturation decile, and non-disparity criteria across skin-tone groups. Those numbers are not arbitrary; they are the sample size needed to detect a clinically meaningful subgroup difference with confidence.

### 5.18 Reading a sensor datasheet in ten minutes

A datasheet is organised around the manufacturer's test conditions, not the customer's use. Read it in this order:

| Section | What to extract | What is often missing |
|---|---|---|
| Features | The headline specification | The conditions behind it |
| Absolute maximum ratings | The range beyond which damage occurs | Whether operation near the limit is reliable |
| Electrical characteristics | Sensitivity, offset, noise, drift, supply range | Typical versus maximum distinction |
| Test conditions | Temperature, supply, load, stimulus | The reference method and its uncertainty |
| Typical performance curves | Behaviour across temperature and range | Unit-to-unit variation |
| Application information | Recommended circuit and layout | The assumptions behind the recommendations |
| Package and handling | Thermal resistance, moisture sensitivity | Long-term reliability under sterilization |
| Qualification | Standards claimed | Which clauses were tested and which were not |

A specification is only usable when the value, the tolerance, the conditions, and the test method are all present. If any of the four is missing, the number is a marketing statement. The most common omission is the distinction between typical and maximum: a design that uses the typical value will pass on the bench and fail on the worst unit.

The same discipline applies to a sensor module that integrates the transducer and the electronics. The module's accuracy may be specified after calibration, at a stated temperature, with a stated averaging time. The data scientist who receives only the module's output cannot tell which of those conditions held.

### 5.19 Propagating sensor error to the clinical quantity

A sensor error becomes a clinical error through the sensitivity of the clinical quantity to the sensor output. For a pressure channel with a 300 mmHg full scale, a 0.5% full-scale accuracy is 1.5 mmHg. A temperature coefficient of 0.2% of full scale per °C, over a 10 °C change from the calibration temperature, adds 6 mmHg before compensation and 1.8 mmHg after a first-order compensation that removes 70% of the drift. The combined error, assuming the terms are independent, is:

\[
e_{\text{total}} = \sqrt{1.5^2 + 1.8^2 + 0.5^2} = 2.4\ \text{mmHg}
\]

where the third term is the resolution and repeatability. A mean arterial pressure target of ±5 mmHg can tolerate a 2.4 mmHg measurement error, but not much more. If the temperature compensation is removed, the error becomes √(1.5² + 6² + 0.5²) = 6.2 mmHg, which exceeds the clinical tolerance.

The arithmetic is simple, but the input values are not. The 0.2%/°C coefficient is a datasheet maximum for one sensor; a different sensor may be 0.05%/°C or 0.5%/°C. The compensation's effectiveness depends on the temperature sensor's accuracy and on the thermal lag between the sensor and the fluid. A data scientist who receives pressure data without the sensor temperature cannot reproduce this budget, which is why the temperature channel belongs in the record even when it is not used by the model.

## 6. Worked example

### 6.1 The requirement

A wearable reflectance pulse oximeter must estimate heart rate and SpO₂ from a finger or wrist. The requirement for this example is limited to the optical front end:

- Green LED at 525 nm for heart rate, with a radiant flux of 8 mW at 20 mA.
- Photodiode responsivity 0.35 A/W at 525 nm.
- Optical coupling: 2% of the emitted light reaches the tissue; 0.5% of the emitted light returns to the photodiode.
- Perfusion index: 2% of the DC photocurrent is pulsatile.
- Signal bandwidth: 0.5–50 Hz for the pulsatile component.
- Transimpedance amplifier (TIA) gain: 100 kΩ.
- ADC: 16 bits, 3.0 V reference, single-ended.
- Ambient light at the photodiode: equivalent to 1 µW of green light, modulated at 100 Hz by a fluorescent ballast.

### 6.2 The optical and electrical budget

The optical power at the photodiode is:

\[
P_{\text{pd}} = 8\ \text{mW} \times 0.005 = 40\ \text{µW}
\]

The DC photocurrent is:

\[
I_{\text{DC}} = 40\ \text{µW} \times 0.35\ \text{A/W} = 14\ \text{µA}
\]

The pulsatile current at a 2% perfusion index is:

\[
I_{\text{AC}} = 14\ \text{µA} \times 0.02 = 280\ \text{nA}
\]

The shot noise of the DC photocurrent over a 50 Hz bandwidth is:

\[
i_{\text{shot}} = \sqrt{2 q I_{\text{DC}} B} = \sqrt{2 \times 1.602 \times 10^{-19} \times 14 \times 10^{-6} \times 50} = 15.0\ \text{pA RMS}
\]

The shot-noise-limited SNR is:

\[
\text{SNR}_{\text{shot}} = 20 \log_{10}\left(\frac{280\ \text{nA}}{15.0\ \text{pA}}\right) = 85.4\ \text{dB}
\]

The TIA converts the current to voltage with a gain of 100 kΩ:

\[
V_{\text{DC}} = 14\ \text{µA} \times 100\ \text{kΩ} = 1.40\ \text{V}
\]

\[
V_{\text{AC}} = 280\ \text{nA} \times 100\ \text{kΩ} = 28.0\ \text{mV}
\]

The DC output of 1.40 V leaves 1.60 V to the 3.0 V rail, which is adequate for the LED and ambient terms only if they are small. The ambient equivalent of 1 µW produces:

\[
I_{\text{amb}} = 1\ \text{µW} \times 0.35\ \text{A/W} = 0.35\ \text{µA}
\]

\[
V_{\text{amb}} = 0.35\ \text{µA} \times 100\ \text{kΩ} = 35\ \text{mV}
\]

The total DC output is 1.435 V, which still fits. A brighter ambient source, or a larger TIA gain, would not.

The ADC has a 3.0 V reference and 16 bits, so the LSB is:

\[
\text{LSB} = \frac{3.0}{2^{16}} = 45.8\ \text{µV}
\]

The pulsatile signal of 28.0 mV occupies:

\[
N_{\text{codes}} = \frac{28.0\ \text{mV}}{45.8\ \text{µV}} = 611\ \text{codes}
\]

The quantisation noise of an ideal converter is LSB/√12 = 13.2 µV RMS. The quantisation SNR for the 28 mV signal is:

\[
\text{SNR}_{q} = 20 \log_{10}\left(\frac{28.0\ \text{mV}}{13.2\ \text{µV}}\right) = 66.5\ \text{dB}
\]

The optical shot noise, referred to the TIA output, is 15.0 pA × 100 kΩ = 1.50 µV RMS, which is far below the quantisation noise. The converter, not the optics, limits the single-sample SNR. The TIA resistor's thermal noise is:

\[
e_{R} = \sqrt{4 k T R} = \sqrt{4 \times 1.38 \times 10^{-23} \times 310 \times 100\,000} = 40.7\ \text{nV}/\sqrt{\text{Hz}}
\]

Over the 78.5 Hz noise bandwidth of a first-order 50 Hz filter, that is 367 nV RMS, or 3.7 pA at the input. It is negligible compared with the 15 pA shot noise. The design is shot-noise-limited in the optical domain and quantisation-limited in the digital domain.

### 6.3 Diagnostic 1: motion artefact, not noise, sets the limit

A motion artefact of 5% of the DC photocurrent is 700 nA, which is 2.5 times the 280 nA pulsatile signal. The signal-to-artefact ratio is:

\[
\text{SAR} = 20 \log_{10}\left(\frac{280}{700}\right) = -8.0\ \text{dB}
\]

In the presence of that motion, the pulsatile signal is below the artefact. No amount of averaging over a single pulse recovers it, because the artefact is correlated with the motion and may be within the same band. The design response is not a better ADC; it is motion rejection: a shorter wavelength with better tissue penetration, an accelerometer for adaptive filtering, a mechanical design that stabilises contact, and a quality index that rejects the segment.

This diagnostic reveals a design flaw in the requirement, not in the electronics. The requirement said "motion present" but the budget was computed for a stationary subject. A budget that omits the dominant artefact is optimistic even when every number in it is correct.

### 6.4 Diagnostic 2: the DC term saturates the TIA

The DC photocurrent of 14 µA at a TIA gain of 100 kΩ produces 1.40 V. If the LED current is increased to 40 mA to improve the SNR, the DC photocurrent doubles to 28 µA and the output reaches 2.80 V, leaving only 0.20 V of headroom. A modest ambient contribution or a change in tissue coupling then saturates the amplifier.

The saturation is visible as a flat top on the PPG waveform and as a constant code at the ADC. The firmware may report a valid signal because the AC coupling after the TIA removes the DC level, and the remaining signal looks plausible. This is a dangerous failure: the AC-coupled output can still show a waveform, but it is distorted by the clipping.

The fix is to remove the DC before the TIA, or to use a lower TIA gain with a second stage, or to modulate the LED and use a synchronous detector. Each has a cost: DC removal adds a high-pass corner and settling time; a lower gain raises the quantisation noise; modulation adds complexity and power.

### 6.5 Vary one assumption: perfusion falls to 0.3%

Change only the perfusion index, from 2% to 0.3%, which is plausible in a cold finger or a patient with peripheral vascular disease. The pulsatile current becomes 42 nA, and the shot-noise-limited SNR becomes:

\[
\text{SNR}_{\text{shot}} = 20 \log_{10}\left(\frac{42\ \text{nA}}{15.0\ \text{pA}}\right) = 68.9\ \text{dB}
\]

The TIA output is 4.2 mV, or 92 ADC codes. The quantisation SNR becomes:

\[
\text{SNR}_{q} = 20 \log_{10}\left(\frac{4.2\ \text{mV}}{13.2\ \text{µV}}\right) = 50.1\ \text{dB}
\]

The motion artefact is unchanged at 700 nA, so the signal-to-artefact ratio collapses to:

\[
\text{SAR} = 20 \log_{10}\left(\frac{42}{700}\right) = -24.4\ \text{dB}
\]

The design that worked at a 2% perfusion index is unusable at 0.3%. The consequence is not a gradual degradation; it is a transition from a reliable measurement to a signal dominated by artefact. A validation that enrolled only warm, well-perfused subjects would not reveal the transition.

The fix options are: increase the LED power, which raises the DC term and the saturation risk; increase the TIA gain, which does the same; use a longer wavelength with deeper penetration, which changes the calibration; or reject the segment and report no value, which is safer than reporting a wrong value. The last option is the one most often omitted from the design requirement.

### 6.6 What the example demonstrates

The arithmetic closes: 40 µW at the photodiode, 14 µA DC, 280 nA pulsatile, 15 pA shot noise, 85.4 dB shot-limited SNR, 28 mV at the TIA, 612 codes, 66.5 dB quantisation SNR. The two diagnostics reveal different flaws: the first is an omitted confounder (motion), the second is a headroom assumption (DC term plus gain). The sensitivity analysis shows that a perfusion change of a factor of 6.7 turns a comfortable design into an unusable one.

The method generalises. Write the optical or electrical budget, identify the dominant noise or artefact, test the assumption that the dominant term depends on, and state the consequence. A sensor datasheet gives the sensitivity; the budget gives the engineering.

### 6.7 Closing the power budget

The optical duty cycle is 2%: the LED is on for 100 µs at 200 Hz. The average LED power is:

\[
P_{\text{LED}} = 3.3\ \text{V} \times 20\ \text{mA} \times 0.02 = 1.32\ \text{mW}
\]

The TIA, ADC, and microcontroller add approximately 3.3 mW, 1.0 mW, and 5.0 mW, giving a total of 10.62 mW. A 200 mAh lithium-polymer cell at 3.7 V stores 2,664 J. The runtime is:

\[
t = \frac{2\,664\ \text{J}}{10.6 \times 10^{-3}\ \text{J/s} \times 86\,400\ \text{s/day}} = 2.91\ \text{days}
\]

The power budget closes with a 2.90-day runtime, which is adequate for a device that is charged twice a week but not for a 30-day patch. The dominant term is the microcontroller, not the LED. A design that halves the LED current to save power would reduce the photocurrent and the SNR, but would save only 0.66 mW, or 6% of the total. The engineering effort should go to the microcontroller's sleep schedule, not to the optical source.

This is a common pattern: the component that dominates the signal budget is not the component that dominates the power budget. Optimising the wrong one is a predictable way to spend a quarter without changing the product.
## 7. Code, query, or calculation

The script reproduces the optical, electrical, and power budget in Section 6 with the Python standard library. It is deliberately explicit: every intermediate quantity is printed so that a reader can check the arithmetic and substitute their own sensor values.

**Assumptions.** The LED flux, coupling, responsivity, perfusion index, bandwidth, TIA gain, converter, and duty cycle are the values stated in Section 6.1. The model ignores ambient light except where stated, treats shot noise as the only optical noise, and assumes a first-order 50 Hz filter for the TIA noise bandwidth.

```python
import math

q = 1.602176634e-19
k = 1.380649e-23
T = 310.0

p_led_mw = 8.0
coupling = 0.005
responsivity = 0.35
pi = 0.02
bw = 50.0
rtia = 100e3
vref = 3.0
bits = 16

p_pd_uw = p_led_mw * 1e3 * coupling
i_dc = p_pd_uw * 1e-6 * responsivity
i_ac = i_dc * pi
i_shot = math.sqrt(2 * q * i_dc * bw)
snr_shot = 20 * math.log10(i_ac / i_shot)
v_dc = i_dc * rtia
v_ac = i_ac * rtia
lsb = vref / 2**bits
codes = v_ac / lsb
snr_q = 20 * math.log10(v_ac / (lsb / math.sqrt(12)))
e_r = math.sqrt(4 * k * T * rtia)
e_r_bw = e_r * math.sqrt(math.pi / 2 * bw)
i_r = e_r_bw / rtia
p_led = 3.3 * 20e-3 * 0.02 * 1e3
p_total = p_led + 3.3 + 1.0 + 5.0
runtime = (0.200 * 3.7 * 3600) / (p_total * 1e-3 * 86400)

print("OPTICAL BUDGET")
print(f"  optical power at photodiode {p_pd_uw:8.2f} uW")
print(f"  DC photocurrent             {i_dc*1e6:8.2f} uA")
print(f"  pulsatile current (PI=2%)   {i_ac*1e9:8.1f} nA")
print(f"  shot noise (50 Hz)          {i_shot*1e12:8.2f} pA RMS")
print(f"  shot-limited SNR            {snr_shot:8.2f} dB")
print()
print("TRANSIMPEDANCE AND ADC")
print(f"  TIA DC output               {v_dc:8.3f} V")
print(f"  TIA AC output               {v_ac*1e3:8.2f} mV")
print(f"  ADC LSB                     {lsb*1e6:8.2f} uV")
print(f"  codes for AC signal         {codes:8.0f}")
print(f"  quantisation SNR            {snr_q:8.2f} dB")
print(f"  TIA resistor noise          {e_r_bw*1e9:8.2f} nV RMS ({i_r*1e12:.2f} pA)")
print()
print("MOTION AND PERFUSION")
i_motion = i_dc * 0.05
print(f"  motion current (5% DC)      {i_motion*1e9:8.1f} nA")
print(f"  signal-to-artefact          {20*math.log10(i_ac/i_motion):8.2f} dB")
for pi2 in (0.003, 0.02):
    ac2 = i_dc * pi2
    print(f"  PI={pi2*100:4.1f}%: AC {ac2*1e9:6.1f} nA, ADC {ac2*rtia/lsb:5.0f} codes, "
          f"SAR {20*math.log10(ac2/i_motion):6.2f} dB")
print()
print("POWER")
print(f"  LED average                 {p_led:8.2f} mW")
print(f"  total                       {p_total:8.2f} mW")
print(f"  runtime (200 mAh)           {runtime:8.2f} days")
```

**Expected output.**

```text
OPTICAL BUDGET
  optical power at photodiode    40.00 uW
  DC photocurrent                14.00 uA
  pulsatile current (PI=2%)      280.0 nA
  shot noise (50 Hz)             14.98 pA RMS
  shot-limited SNR               85.43 dB

TRANSIMPEDANCE AND ADC
  TIA DC output                  1.400 V
  TIA AC output                  28.00 mV
  ADC LSB                        45.78 uV
  codes for AC signal              612
  quantisation SNR               66.52 dB
  TIA resistor noise            366.69 nV RMS (3.67 pA)

MOTION AND PERFUSION
  motion current (5% DC)         700.0 nA
  signal-to-artefact             -7.96 dB
  PI= 0.3%: AC   42.0 nA, ADC    92 codes, SAR -24.44 dB
  PI= 2.0%: AC  280.0 nA, ADC   612 codes, SAR  -7.96 dB

POWER
  LED average                     1.32 mW
  total                          10.62 mW
  runtime (200 mAh)               2.90 days
```

**What the output shows.**

The shot-limited SNR is 85.4 dB, but the quantisation-limited SNR is 66.5 dB. The converter, not the optics, limits the single-sample measurement. Raising the TIA gain to 300 kΩ would improve the quantisation SNR by 9.5 dB but would push the DC output to 4.2 V, beyond the 3.0 V rail. The design is caught between the DC term and the quantisation step.

The motion calculation is the important one. A 5% motion artefact is 2.5 times the pulsatile signal, so the signal-to-artefact ratio is −8.0 dB even though the optical SNR is 85 dB. The measurement is not noise-limited; it is artefact-limited. The correct engineering response is motion rejection, not a quieter amplifier.

The power budget closes at 10.62 mW and a 2.90-day runtime. The microcontroller is 47% of the total, the TIA 31%, the LED 12%, and the converter 9%. A 50% reduction in LED current saves 6% of the power and costs 6 dB of optical SNR. That is a bad trade; the same effort spent on the microcontroller's duty cycle would save more.

**Sensitivity to the coupling assumption.** The 0.5% coupling is the least certain number in the model. If the actual coupling is 0.25%, the photocurrent halves, the shot noise falls by √2, and the shot-limited SNR falls by 3 dB, but the quantisation SNR falls by 6 dB because the signal at the TIA halves. A design that has only 6 dB of quantisation margin cannot tolerate a factor-of-two coupling error. The coupling must be measured on the actual mechanical design, not assumed from a similar product.

### 7.1 Reproducing the pressure error budget

The same script pattern applies to a pressure channel. The calculation separates the calibration accuracy, the temperature drift before and after compensation, and the resolution.

```python
import math

fs = 300.0                 # full scale, mmHg
acc = 0.005 * fs           # 0.5% of full scale
tcoef = 0.002 * fs         # 0.2% of full scale per degC
dt = 10.0                  # temperature change from calibration, degC
raw = tcoef * dt
comp = raw * 0.3           # residual after 70% compensation
res = 0.5                  # resolution and repeatability, mmHg

total = math.sqrt(acc**2 + comp**2 + res**2)
total_nc = math.sqrt(acc**2 + raw**2 + res**2)

print(f"accuracy {acc:.2f} mmHg")
print(f"drift raw {raw:.2f} mmHg")
print(f"residual after 70% compensation {comp:.2f} mmHg")
print(f"total with compensation {total:.2f} mmHg")
print(f"total without compensation {total_nc:.2f} mmHg")
```

**Expected output.**

```text
accuracy 1.50 mmHg
drift raw 6.00 mmHg
residual after 70% compensation 1.80 mmHg
total with compensation 2.40 mmHg
total without compensation 6.20 mmHg
```

The compensated total of 2.40 mmHg fits a ±5 mmHg clinical tolerance with a factor of two margin. The uncompensated total of 6.20 mmHg does not. The difference between the two is the value of the temperature compensation, and the compensation's own uncertainty is not included; if the temperature sensor is accurate to ±1 °C and the thermal lag is 30 s, the residual drift is larger than the 1.8 mmHg calculated here. A budget that treats compensation as perfect is optimistic.

## 8. Data, standards, and regulatory context

Sensor performance is regulated through a combination of particular standards, collateral standards, and guidance. The table separates the binding instruments from the consensus standards that become binding when referenced.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| ISO 80601-2-61:2017 | Pulse oximeter equipment | Particular standard; 2017 edition withdrawn 2026-04-10 and revised by ISO 80601-2-61:2026 | SpO₂ accuracy, display, alarm, labelling |
| ISO 80601-2-61:2026 | Pulse oximeter equipment, revised | Current particular standard | Updated accuracy and validation requirements |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment safety | Consensus standard; binding when referenced | Applied parts, leakage current, isolation |
| IEC 60601-1-2:2014+AMD1:2020 | Electromagnetic disturbances | Collateral standard | Immunity of sensor and cable to RF, ESD, magnetic fields |
| ISO 10993-1:2025 | Biological evaluation of medical devices | Consensus standard | Material selection by contact category and duration |
| ISO 11135:2014 | Ethylene oxide sterilization | Consensus standard | Residuals, validation, packaging |
| ISO 11137 series | Radiation sterilization | Consensus standard | Dose setting and validation |
| ISO 17665-1:2006 | Moist heat sterilization | Consensus standard | Autoclave cycles and validation |
| FDA pulse oximeter guidance (draft, January 2025) | SpO₂ accuracy and diverse pigmentation | Draft guidance | Study design, sample size, non-disparity criteria |
| FDA iCGM special controls | Integrated continuous glucose monitoring | Binding through classification regulation | Accuracy, lag, interferents, cybersecurity, labelling |
| EU MDR 2017/745 | Medical devices in the EU | Binding regulation | Clinical evaluation, technical file, post-market surveillance |

### 8.1 Pulse oximetry: the standard and the guidance diverge

ISO 80601-2-61:2017 clause 201.12.1.101.1 requires an A<sub>rms</sub> of no more than 4.0% over the 70–100% saturation range, with the accuracy stated in the instructions for use. The standard notes that only about two-thirds of measurements fall within ±A<sub>rms</sub>, which is a property of the root-mean-square definition, not a defect.

The FDA does not recognise that clause in full. Its 2013 guidance recommended tighter values for premarket submissions: A<sub>rms</sub> ≤3.0% for transmittance wrap or clip sensors, and ≤3.5% for ear clip and reflectance sensors. The January 2025 draft guidance goes further: A<sub>rms</sub> <3% with a 95% confidence upper limit below 3% for all sensor types, at least 150 participants, at least 3,000 paired observations, at least 30% of observations in each saturation decile, and non-disparity criteria across skin-tone groups. The draft also proposes that the largest difference in mean bias between skin-tone groups be below 3.5% for 70–85% saturation and below 1.5% for 85–100%.

The gap between a 4.0% standard and a 3.0% guidance is not a rounding difference. A device that meets the standard may not meet the guidance, and a device that meets the guidance may still fail a subgroup. The data scientist should ask which criterion the label claims and which population was studied.

### 8.2 Continuous glucose monitoring

An integrated CGM is regulated as a device with special controls. The evidence package includes accuracy against a reference method across the measuring range, performance during rapid glucose change, lag, interference from acetaminophen and other substances, performance at altitude and temperature, cybersecurity, and labelling that states the conditions of use. The accuracy metric is usually MARD, but MARD is sensitive to the study design, the glucose range, the rate of change, and the calibration scheme. A MARD of 10% in one study and 14% in another may reflect the populations and protocols rather than a difference in the sensor.

The regulatory question is not "what is the MARD?" but "what decision does the label support?" A CGM labelled for non-adjunctive use, meaning the user can make treatment decisions without a fingerstick confirmation, has a different evidence burden from a CGM labelled as an adjunct. The label, not the datasheet, defines the claim.

### 8.3 Biocompatibility and sterilization

A sensor that contacts the patient must be evaluated to ISO 10993-1. The evaluation depends on the contact category (surface, external communicating, implant) and the duration (limited, prolonged, long-term). A single-use finger clip is a surface device with limited contact; an implanted glucose sensor is an implant with long-term contact. The evidence for the latter is much larger: cytotoxicity, sensitization, irritation, acute systemic toxicity, material-mediated pyrogenicity, and, for long-term implants, carcinogenicity and reproductive toxicity endpoints may be relevant.

Sterilization can change sensor performance. Gamma irradiation can darken an optical window and change its transmission. Ethylene oxide can leave residuals that affect a membrane. Autoclaving can warp a polymer housing and change the optical alignment. The verification must be performed after the worst-case sterilization cycle, not before it.

### 8.4 What the standards do not specify

The standards do not specify the calibration curve for a particular sensor geometry, the algorithm that computes a heart rate from a PPG, the quality index that rejects motion, or the method for handling missing data. They specify the performance that must be demonstrated, not the design that achieves it. A device can be compliant and still produce data that are unsuitable for a model, because the compliance tests use controlled conditions and the model sees real ones.

### 8.5 What the standards imply for the data record

Each standard creates a data obligation, even when the standard does not say so explicitly.

| Standard | Data the team should retain | Why |
|---|---|---|
| ISO 80601-2-61 | Sensor type, site, perfusion index, motion flag, SpO₂ value and quality | Reproduce the accuracy claim and stratify by condition |
| IEC 60601-1 | Applied-part classification, leakage test result, isolation configuration | Explain noise and safety-related filtering |
| IEC 60601-1-2 | Immunity test levels, cable configuration, worst-case frequencies | Interpret interference-related artefacts |
| ISO 10993-1 | Material, contact duration, sterilization cycle count | Explain performance changes after reprocessing |
| FDA CGM special controls | Calibration scheme, wear time, interferent exposure, lag correction | Support the labelled claim and post-market analysis |
| EU MDR | Clinical evaluation population, PMS plan, vigilance data | Support the conformity assessment and surveillance |

The pattern is that the regulatory evidence and the data provenance are the same information viewed from two angles. A team that records the sensor lot, site, calibration, and quality flags for its own analysis is also assembling the evidence it will need for a submission. A team that records nothing can still pass a bench test, but it cannot explain a field result.

A second implication is that the standards constrain the *claim*, not the *data*. A device can be cleared for a 70–100% saturation range and still produce values outside that range; the values are not necessarily valid, and the label says so. A dataset that includes values outside the labelled range must either exclude them or treat them as extrapolations. The same applies to a CGM labelled for a specific wear duration: a reading from day 15 of a 10-day sensor is not supported by the label, even if the device still displays a number.

## 9. Industry practice and case evidence

### 9.1 Pulse oximetry and skin tone

A 2024 meta-analysis reported pooled A<sub>rms</sub> of 3.96% for light skin, 4.71% for medium skin, and 4.15% for dark skin, with mean bias of 0.70%, 0.27%, and 1.27%. The pattern is not a simple monotonic degradation; the medium group has the largest A<sub>rms</sub> and the dark group the largest bias. The FDA's January 2025 draft guidance responds to the evidence with stricter validation and non-disparity criteria.

The sensor-level mechanism is plausible: melanin absorbs light, especially at shorter wavelengths, which reduces the DC and pulsatile signals and changes the effective ratio-of-ratios. The magnitude depends on the wavelengths, the optical geometry, and the calibration population. The relative contribution of optics and calibration is device-specific and is not public.

**Documented fact:** the meta-analysis numbers and the draft guidance are public. **Inference:** the mechanism is well established in tissue optics, but the attribution of the bias to a particular design choice is an inference.

### 9.2 Continuous glucose monitoring: lag and MARD

Modern factory-calibrated CGMs report MARDs in the range of roughly 9–14%, with substantial variation across studies and glucose ranges. The lag decomposes into a physiological component of about 5–10 minutes for interstitial fluid, a sensor reaction component of about 1–2 minutes, and a filtering component of about 3–9 minutes, giving a total of roughly 8–15 minutes in commercial devices. The lag is not a constant; it depends on the rate of change, the sensor's diffusion path, and the filtering algorithm.

For a data scientist, the consequence is that a CGM value aligned with a blood glucose value without lag correction will show a bias that depends on the rate of change. During a rapid rise, the CGM reads lower; during a rapid fall, it reads higher. A model trained on unaligned pairs learns the lag as if it were physiology.

**Documented fact:** the lag decomposition and the MARD ranges are reported in peer-reviewed reviews. **Inference:** the exact lag for a specific sensor must be estimated from paired data; the published range is a prior, not a specification.

### 9.3 Dry electrodes and contact impedance

Dry electrodes are attractive for long-wear ECG and EEG because they avoid gel and skin preparation. Their contact impedance is higher and more variable than wet Ag/AgCl, often in the range of 50–500 kΩ at 10 Hz compared with 5–20 kΩ for a good wet electrode. The higher impedance increases the thermal noise, converts amplifier current noise into voltage noise, and makes the signal more sensitive to motion.

The validation literature for dry electrodes is smaller than for wet electrodes, and the cohorts are often healthy volunteers. A device that works well in a laboratory with cooperative subjects may fail in a population with dry skin, tremor, or poor electrode placement. The contact impedance is a first-order signal-quality variable, not a secondary detail.

**Documented fact:** the impedance ranges are representative values from the literature and from electrode datasheets; they vary by electrode and subject. **Inference:** the performance gap in a clinical population is inferred from the mechanism and from the limited validation data.

### 9.4 New detector architectures

Butterfly Network's iQ3, cleared in January 2024, uses a capacitive micromachined ultrasonic transducer (CMUT) integrated on an application-specific integrated circuit. The architecture offers bandwidth and integration advantages, but CMUTs face output-pressure and bias-voltage challenges. Siemens Healthineers' photon-counting CT detectors use cadmium zinc telluride or cadmium telluride to provide energy discrimination, with challenges in charge trapping and calibration. In both cases, the detector physics changes the noise model, the calibration, and the reconstruction assumptions.

A data scientist working with images from these devices should not assume that the noise is stationary or that the reconstruction is linear. The detector's energy response and the calibration of the spectral channels are part of the measurement model.

**Documented fact:** the clearances and the detector materials are public. **Inference:** the effect on a specific reconstruction or model is device- and protocol-specific.

### 9.5 What good sensor practice looks like

Teams that avoid sensor surprises tend to:

- Read the datasheet's test conditions before the headline number.
- Measure the transfer function and the cross-sensitivities on the actual mechanical and tissue interface.
- Include the dominant artefact in the budget, not only the electronic noise.
- Stratify validation by the confounders that the mechanism predicts.
- Record the sensor lot, site, calibration, and quality flags with every measurement.
- Re-verify after sterilization, ageing, and any supplier change.
- Treat a change in sensor lot as a change in the measurement until proven otherwise.

None of these is exotic. They are the difference between a specification and a measurement.

### 9.6 Supply chain and lifecycle

A sensor's performance depends on its supply chain. An LED die from a different bin can have a different peak wavelength and radiant flux. A hydrogel from a different lot can have a different chloride concentration and a different offset. A photodiode from a different foundry can have a different dark current. None of these changes may be visible in the product's part number, and all of them can change the calibration.

The lifecycle plan should include: incoming inspection of critical sensor parameters, lot tracking through production, a change-control procedure that treats a material or supplier change as a design change, and a post-market comparison of performance by lot. The last item is often the only way to detect a slow drift that is within each lot's specification but changes the population mean.

A clinically excellent device can fail because a critical sensor is end-of-life or unavailable in a jurisdiction. The mitigation is a last-time buy, a qualified second source, or a redesign that is planned before the shortage. The data team's interest is direct: a sensor change can invalidate a model that was trained on the previous sensor's data.

### 9.7 Comparing two published accuracy studies

Two studies can report the same metric and reach different conclusions. Before comparing them, check:

- **The population.** Age, disease severity, skin tone, perfusion, and body size affect the error.
- **The reference.** A study against arterial blood gas is not comparable with one against another pulse oximeter.
- **The protocol.** Motion, temperature, altitude, and probe site change the error.
- **The analysis.** A<sub>rms</sub> across all pairs is not the same as the mean of per-subject A<sub>rms</sub> values.
- **The exclusion rules.** Excluding poor-quality segments can make a device look better than it is in use.

A meta-analysis that pools studies without checking these dimensions produces a number with a wide and poorly understood confidence interval. The number is not useless, but it is not a specification for a particular device in a particular patient.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Optical saturation | Flat-topped PPG; output at rail | DC term plus gain exceeds the TIA range | Loss of pulsatile signal | DC cancellation; lower gain; modulated drive |
| Motion artefact | Signal-to-artefact ratio below 0 dB during movement | Tissue deformation changes the optical path | False rate or false desaturation | Accelerometer; quality index; mechanical stabilisation |
| Ambient light interference | 100/120 Hz tone; reading changes with room light | Ambient adds to the photocurrent | Bias in ratio and rate | Optical filter; synchronous detection; dark channel |
| Skin-tone bias | Subgroup bias in validation | Melanin absorption changes the ratio | Disparate accuracy | Multi-wavelength; stratified calibration; non-disparity validation |
| Low perfusion | Weak pulsatile signal; high variability | Reduced arterial volume change | Unreliable reading | Perfusion index; signal-quality rejection; warming |
| Electrode offset | Saturation; baseline shift | Half-cell potential plus skin potential | Loss of signal | Headroom; offset servo; AC coupling |
| Contact impedance rise | Increased noise; 50/60 Hz interference | Gel drying, motion, poor adhesion | Poor SNR | Impedance monitoring; electrode replacement |
| Pressure offset drift | Zero shifts after warm-up | Thermal coefficient, hysteresis | Bias in pressure or flow | Zeroing; temperature compensation; periodic check |
| Flow sensor obstruction | Reduced flow reading; increased resistance | Secretions, condensation, debris | Under-delivery of tidal volume | Heated wire; filters; occlusion alarms |
| Glucose sensor fouling | Sensitivity decline over wear | Protein adsorption, membrane degradation | Bias in glucose | Factory calibration curve; wear limit; recalibration |
| Glucose interferent | Unexplained rise after medication | Electroactive substance oxidized at the electrode | False hyperglycaemia | Membranes; interferent testing; labelling |
| Thermistor self-heating | Reading high in still air | Excitation current dissipates power | Bias in temperature | Low excitation; pulsed measurement; calibration |
| IMU drift | Orientation error grows over time | Gyroscope bias integration | Incorrect gait or posture | Fusion with accelerometer; periodic reset |
| Sterilization degradation | Performance shift after reprocessing | Material change, window discolouration | Out-of-specification sensor | Post-sterilization verification; cycle limits |
| Lot-to-lot variation | Shift after a supplier change | Process variation in the transducer | Unexplained data drift | Incoming inspection; lot tracking; change control |
| Reference channel drift | Compensation changes over time | Reference sensor ages differently | Hidden bias | Periodic cross-check; reference replacement |
| Optical window contamination | Gradual signal loss; cleaning does not restore | Finger oils, dust, scratches | Reduced SNR; calibration shift | Cleaning validation; window replacement |

The most dangerous failures are the ones that produce a plausible value. A saturated PPG can still produce a heart rate from the remaining signal; a drifting pressure sensor can still produce a number within the display range; a fouled glucose sensor can still produce a curve. The red flag is not always an error message. It is a change in the relationship between the sensor and an independent reference.

### 10.1 Ten red flags in sensor data

1. A saturation rate that differs by more than a factor of two between sites.
2. A signal-quality index that improves after a firmware update with no hardware change.
3. A feature distribution shift that coincides with a new sensor lot.
4. A PPG waveform whose DC level drifts while the AC amplitude is stable.
5. A pressure channel whose zero moves after every warm-up.
6. A glucose trace that rises after a medication with no change in intake.
7. A temperature channel that reads high only when the device is charging.
8. A heart-rate series with periodic gaps at the radio reconnect interval.
9. A SpO₂ distribution with a hard edge at a value that suggests a display or clamp limit.
10. A subgroup error that is absent in the bench data but present in the field data.

Each red flag is a prompt to check the chain, not a conclusion. The check is usually cheap: compare the sensor with an independent reference, inspect the configuration, and stratify by the suspected confounder.
## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The sensor's sensitivity is constant over its range | A single calibration slope is used | Nonlinearity, saturation, enzyme kinetics | Five-point calibration in both directions |
| The offset is stable between calibrations | A fixed zero is subtracted | Ageing, temperature, sterilization, gel drying | Periodic zero check; drift trend |
| Cross-sensitivities are small or compensated | The intended quantity dominates the output | Temperature, humidity, ambient light, interferents | Confounder sweep at fixed measurand |
| The optical coupling is known | The photocurrent and SNR depend on it | Mechanical fit, tissue type, sensor placement | Measure on the actual interface |
| The perfusion index is representative | The pulsatile amplitude is predicted from it | Cold, peripheral disease, vasoconstriction | Stratify validation by perfusion |
| The skin-tone distribution is representative | The calibration curve is applied to the population | calibration cohort skew | Stratified validation; non-disparity analysis |
| The reference method is accurate | Accuracy is defined against it | Reference error, sampling delay, operator error | Reference uncertainty budget; paired samples |
| The sensor is used at the labelled site | The calibration is site-specific | Off-label placement, poor technique | Usability testing; site logging |
| The device is used within its environmental range | Temperature and humidity affect the sensor | Home, ambulance, operating room extremes | Environmental testing; use logs |
| The sterilization cycle is the validated one | Material performance is cycle-dependent | Reprocessing beyond the validated count | Cycle tracking; post-sterilization testing |
| The firmware and calibration coefficients match the validated version | The measurement model is fixed | Updates, service replacement, rollback | Version and checksum verification |
| The patient population matches the validation cohort | Accuracy generalises | Age, comorbidity, medications, body size | Subgroup analysis; external validation |
| The reference method is time-aligned | Paired comparison is valid | Sampling delay, different compartments | Time-stamped paired sampling |
| The sensor is not reused beyond its validated life | Performance is maintained | Cleaning, wear, drift | Cycle tracking; end-of-life test |
| The cleaning agent is compatible | Housing and window survive | Solvent attack, crazing | Cleaning validation; inspection |
| The electromagnetic environment matches the test | Immunity is adequate | Electrosurgery, nearby radios | In-use EMC testing; logging |

The table is a ledger of promises. Each row is a condition that the sensor's accuracy claim depends on, and each row has a test that could falsify it. A validation that does not address the rows that vary in the intended population is not a validation of the intended use; it is a validation of the laboratory.

Two limits deserve separate mention.

**The first is the difference between a sensor and a measurement.** A sensor produces a signal; a measurement is a value with an uncertainty and a context. The same sensor can produce a good measurement in one context and a bad one in another. A pulse oximeter on a warm, well-perfused finger in a quiet room is a different measurement from the same device on a cold, moving finger in an ambulance. The device did not change; the context did.

**The second is the difference between a population average and an individual.** An A<sub>rms</sub> of 3% describes a distribution, not a guarantee for the next patient. A clinician who treats a single reading as if it were within 3% of truth is making an inference that the statistic does not support. The data scientist who reports only the average makes the same inference at the population level.

A third limit is that the sensor models in this topic are simplified. The optical model ignores scattering and the layered structure of skin; the pressure model ignores the compliance of tubing and the dynamics of the fluid; the glucose model ignores the spatial distribution of the enzyme and the diffusion boundary layer. The simplifications are adequate for budgeting and for reasoning about failure modes, not for designing a sensor from scratch. A specialist uses a more detailed model and a physical prototype.

A fourth limit is that the validation evidence is usually reported for a population, not for the individual in front of the clinician. A sensor with a 3% population A<sub>rms</sub> can be 10% wrong for a particular patient, and the device cannot tell which patient it is. The clinical response is to treat a single reading as one piece of evidence, not as a measurement of truth, and to confirm with an independent method when the decision is consequential.

A fifth limit is that the sensor's failure modes are not independent. A cold finger reduces perfusion, increases contact impedance, and changes the optical path at the same time. A model that treats these as separate covariates will underestimate the combined effect. The validation should include the combinations that occur in use, not only the single-confounder sweeps.

### 11.1 Assumption audit questions

Use these questions in a design review or a data review:

- Which assumption in the table would change the clinical decision if it were false?
- Which assumption has no test, only a belief?
- Which assumption was true for the bench but is false for the home?
- Which assumption depends on a supplier or a lot?
- Which assumption is encoded in the firmware but not in the documentation?
- Which assumption would a new firmware version change?
- Which assumption is specific to the population of the validation study?
- Which assumption is specific to the environment of the validation study?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback if it fails?

A review that cannot answer the last two questions has identified risks without controls.

## 12. Exercises

The eight exercises move from matching principles to designing a validation. Each states a deliverable, acceptance criteria, and a time box. They assume access to a calculator or Python and a public sensor datasheet.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Match principle to measurand | Table with rejection reasons |
| 2 | Read a specification | Structured extract |
| 3 | Compute an optical budget | Calculation table |
| 4 | Diagnose an artefact | Ratio and proposals |
| 5 | Design a calibration plan | One-page plan |
| 6 | Stratify a validation | Protocol outline |
| 7 | Evaluate a CGM claim | Five questions |
| 8 | Design a provenance record | Metadata table |

### Exercise 1. Match principle to measurand (recall, 20 minutes)

For each measurand, name two candidate sensing principles and one reason to reject each: (a) arterial oxygen saturation, (b) invasive blood pressure, (c) core temperature, (d) interstitial glucose, (e) joint angle.

**Deliverable:** a five-row table with principle, advantage, and rejection reason.

**Acceptance criteria:** at least two principles per row; the rejection reason is physical, not commercial; units where relevant.

**Time:** 20 minutes.

### Exercise 2. Read a datasheet (application, 30 minutes)

Choose a commercial sensor datasheet with an accuracy specification. Extract the value, tolerance, test conditions, reference method, and whether the number is typical or maximum. Write a one-paragraph assessment of whether the specification supports a claim of 1% accuracy in a moving patient.

**Deliverable:** a structured extract and a paragraph.

**Acceptance criteria:** all five elements present; the distinction between typical and maximum is explicit; the conclusion follows from the conditions.

**Time:** 30 minutes.

### Exercise 3. Compute an optical budget (application, 40 minutes)

An infrared LED emits 12 mW at 940 nm, with 1.5% coupling to the tissue and 0.4% returning to a photodiode with 0.6 A/W responsivity. The perfusion index is 1.5%, the bandwidth is 40 Hz, the TIA gain is 50 kΩ, and the ADC is 16 bits with a 3.3 V reference. Compute the DC and pulsatile photocurrents, the shot noise, the shot-limited SNR, the TIA output, the ADC codes, and the quantisation SNR. State which limit dominates.

**Deliverable:** a calculation table and a two-sentence conclusion.

**Acceptance criteria:** correct units; shot noise computed from the DC current; quantisation noise from LSB/√12; the dominant limit identified.

**Time:** 40 minutes.

### Exercise 4. Diagnose a motion artefact (critique, 30 minutes)

A wrist PPG reports a heart rate of 140 bpm while the patient is walking, but 72 bpm at rest. The optical SNR at rest is 80 dB, and the motion artefact is 8% of the DC photocurrent. Estimate the signal-to-artefact ratio at a 2% perfusion index and explain why a better ADC would not fix the problem. Propose two design changes.

**Deliverable:** a calculation and two proposals with costs.

**Acceptance criteria:** signal-to-artefact ratio correct; the explanation distinguishes noise from artefact; each proposal states its effect on power, size, or cost.

**Time:** 30 minutes.

### Exercise 5. Design a calibration plan (design, 35 minutes)

A disposable pressure sensor is factory-calibrated and used for 72 hours. Design a calibration and drift-check plan that does not require the user to calibrate. Include the reference, the interval, the acceptance criterion, and the action if the check fails.

**Deliverable:** a one-page plan.

**Acceptance criteria:** the plan covers zero, span, temperature, and drift; the acceptance criterion is quantitative; the failure action is safe.

**Time:** 35 minutes.

### Exercise 6. Stratify a validation (design, 40 minutes)

A pulse oximeter is intended for home use in adults with chronic lung disease. Design the validation strata and the minimum sample size rationale. Include skin tone, perfusion, saturation range, and motion. State the primary analysis and the subgroup criteria.

**Deliverable:** a protocol outline with a stratification table.

**Acceptance criteria:** strata follow the confounders; the sample size is justified; the analysis is pre-specified; the non-disparity criterion is explicit.

**Time:** 40 minutes.

### Exercise 7. Evaluate a CGM claim (critique, 30 minutes)

A CGM is advertised with a MARD of 9.5%. Write five questions that would determine whether the number supports non-adjunctive use in a patient with type 1 diabetes.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover study population, glucose range, rate of change, calibration, and lag; no question is answerable by the headline number alone.

**Time:** 30 minutes.

### Exercise 8. Design a sensor provenance record (design, 35 minutes)

Design a metadata schema for a wearable optical sensor. Include at least 15 fields covering device, sensor, calibration, placement, environment, and quality. Explain how each field would help diagnose a subgroup performance gap.

**Deliverable:** a table with field, type, example, and diagnostic use.

**Acceptance criteria:** covers lot, site, skin-tone category if ethically collected, perfusion index, ambient light, motion, and firmware; each field has a stated use.

**Time:** 35 minutes.

## 13. Self-check answers

**Answer 1 (Exercise 1).** (a) SpO₂: optical ratio-of-ratios, because it can distinguish oxy- and deoxyhemoglobin; electrochemical, rejected because it is invasive and not continuous. (b) Invasive blood pressure: piezoresistive bridge, because it is linear and small; ultrasonic, rejected because it needs coupling and is not invasive. (c) Core temperature: thermistor, because it is stable and inexpensive; infrared, rejected because it measures surface temperature and depends on emissivity and site. (d) Interstitial glucose: electrochemical enzymatic, because it is selective and continuous; optical, rejected because the signal is weak and non-specific in tissue. (e) Joint angle: encoder, because it is direct and precise; IMU, rejected because it drifts and measures segment orientation, not joint angle.

**Answer 2 (Exercise 2).** A specification of "±2% of reading" without a range, temperature, or reference is not a specification. A usable version states: "±2% of reading from 10% to 100% of full scale, at 23 ± 2 °C, against a calibrated reference with 0.1% uncertainty, 95% of readings, excluding motion." Even that version does not support a 1% claim in a moving patient, because motion is excluded and the population is not the patient population.

**Answer 3 (Exercise 3).** Optical power at the photodiode = 12 mW × 0.004 = 48 µW. DC photocurrent = 48 µW × 0.6 A/W = 28.8 µA. Pulsatile = 28.8 µA × 0.015 = 432 nA. Shot noise = √(2 × 1.602e-19 × 28.8e-6 × 40) = 19.2 pA RMS. Shot-limited SNR = 20 log10(432 nA / 19.2 pA) = 87.0 dB. TIA output DC = 28.8 µA × 50 kΩ = 1.44 V; AC = 432 nA × 50 kΩ = 21.6 mV. LSB = 3.3 / 65,536 = 50.35 µV; codes = 21.6 mV / 50.35 µV = 429. Quantisation SNR = 20 log10(21.6 mV / (50.35 µV/√12)) = 63.4 dB. The converter limits the single-sample SNR; the optics are 23.6 dB quieter. The DC output of 1.44 V leaves 1.86 V to the 3.3 V rail, so there is headroom for ambient.

**Answer 4 (Exercise 4).** At a 2% perfusion index, the pulsatile signal is 2% of DC and the motion artefact is 8% of DC, so the signal-to-artefact ratio is 20 log10(2/8) = −12.0 dB. A better ADC reduces quantisation noise, but the artefact is 4 times the signal and lies in the same band; the converter is not the limit. Two design changes: add an accelerometer and use adaptive filtering to subtract the motion-correlated component, which costs power and complexity; or improve the mechanical coupling and use a quality index that rejects motion segments, which costs coverage but is safer. A third option, a longer wavelength with deeper penetration, may reduce motion sensitivity but changes the calibration.

**Answer 5 (Exercise 5).** The plan: factory calibration at three temperatures and five pressures; a zero check at startup against atmosphere; a span check against a built-in reference or a known excitation; a drift limit of 1% of full scale over 72 hours; a temperature compensation curve; and a fail-safe action that flags the sensor and stops the infusion or alarm if the zero check fails. The user is not asked to calibrate; the device performs the checks automatically. The acceptance criterion is quantitative and tied to the clinical requirement.

**Answer 6 (Exercise 6).** Strata: skin tone (at least three groups, with at least 30% of observations in each), perfusion index (low versus normal), saturation range (70–85%, 85–92%, 92–100%), and motion (still versus moving). Minimum sample: at least 150 participants and 3,000 paired observations, with at least 30% in each saturation decile, following the FDA draft guidance. Primary analysis: A<sub>rms</sub> with a 95% confidence interval, and bias by stratum. Subgroup criterion: the largest difference in mean bias below 3.5% for 70–85% and below 1.5% for 85–100%, and the upper confidence limit of A<sub>rms</sub> below 3% overall.

**Answer 7 (Exercise 7).** Questions: (1) What was the study population, and did it include type 1 diabetes with rapid glucose change? (2) What was the glucose range, and how many points were in the hypoglycaemic range? (3) What was the rate of change, and was lag corrected? (4) Was the device factory-calibrated or user-calibrated, and how was calibration handled in the analysis? (5) Were interferents and compression lows excluded, and how? A MARD of 9.5% does not by itself support non-adjunctive use; the label and the study design do.

**Answer 8 (Exercise 8).** Fields: device model; firmware; sensor lot; sensor type; site; wear duration; calibration coefficients and date; skin-tone category if collected with consent; perfusion index; ambient light level; motion index; signal-quality flag; LED current; detector temperature; battery state; and the raw optical channels. Each field supports a diagnostic: lot for a manufacturing shift, site for placement effects, perfusion for low-flow bias, ambient for light interference, and firmware for a configuration change.

**Rubric for the design exercises.** A strong design states the requirement, the confounders, the test, the acceptance criterion, and the action on failure. A weak design lists best practices without quantities or without a failure action.

**Numeric answer key.**

| Item | Expected value | Tolerance for full credit |
|---|---:|---|
| Exercise 3: DC photocurrent | 28.8 µA | ±0.5 µA |
| Exercise 3: pulsatile current | 432 nA | ±10 nA |
| Exercise 3: shot noise | 19.2 pA RMS | ±1 pA |
| Exercise 3: shot-limited SNR | 87.0 dB | ±0.5 dB |
| Exercise 3: ADC codes | 429 | ±10 codes |
| Exercise 3: quantisation SNR | 63.4 dB | ±0.5 dB |
| Exercise 4: signal-to-artefact | −12.0 dB | ±0.5 dB |
| Exercise 7: pressure total with compensation | 2.40 mmHg | ±0.1 mmHg |

A value outside the tolerance but with a correct method receives partial credit. A value inside the tolerance with no units receives partial credit only. The purpose of the key is to make marking consistent, not to replace the reasoning.

**Common mistakes in the calculation exercises.**

- Using the LED power at the photodiode as the emitted power.
- Forgetting the responsivity when converting optical power to current.
- Computing shot noise from the AC current instead of the DC current.
- Using the signal bandwidth instead of the noise bandwidth for the TIA resistor.
- Converting the TIA output to codes with the wrong reference voltage.
- Comparing the shot-limited SNR with the quantisation SNR and choosing the larger as the system SNR.
- Treating a 5% motion artefact as a 5% signal error rather than as an additive term.
- Applying a temperature coefficient to the reading instead of to full scale.

Each mistake changes the answer by a factor or a few decibels, which is enough to change a design decision.

## 14. Interview questions

**1. What is the difference between a sensor and a transducer?**
A strong answer defines the transducer as the energy-conversion element and the sensor as the transducer plus packaging and conditioning, and gives an example. A weak answer uses the terms interchangeably.

**2. Why can a pulse oximeter be accurate on average and biased in a subgroup?**
A strong answer explains the calibration cohort, the optical effect of melanin, and the difference between pooled and stratified accuracy. A weak answer says "skin tone affects light" without connecting it to the calibration and the validation.

**3. What limits a PPG measurement: noise or artefact?**
A strong answer computes the shot-limited SNR and the signal-to-artefact ratio and shows that motion usually dominates. A weak answer assumes the amplifier noise is the limit.

**4. How would you validate a new electrode?**
A strong answer specifies the reference, the population, the contact-impedance measurement, the motion protocol, and the stratified analysis. A weak answer compares it with another electrode on a few volunteers.

**5. What is A<sub>rms</sub>, and what does it not tell you?**
A strong answer defines it as the root-mean-square difference from a reference and notes that it is a population statistic, not a maximum or an individual guarantee. A weak answer treats it as a worst-case error.

**6. Why does CGM lag matter for a model?**
A strong answer decomposes the lag into physiological, sensor, and filtering components and explains that unaligned pairs bias the model by the rate of change. A weak answer says the sensor is "slow" without quantifying or aligning.

**7. How does sterilization change a sensor?**
A strong answer names material and optical effects, the need for post-sterilization verification, and the cycle limit. A weak answer says sterilization is "validated" without connecting it to performance.

**8. When would you add a second sensing channel?**
A strong answer identifies a confounder that varies on the measurement timescale and cannot be held constant, and names the reference channel. A weak answer adds channels for redundancy without a confounder model.

**9. How do you read a datasheet's accuracy specification?**
A strong answer extracts the value, tolerance, conditions, reference, and typical-versus-maximum status. A weak answer quotes the headline number.

**10. What would make you reject a sensor for a clinical application?**
A strong answer names an unacceptable failure mode, a subgroup gap, a calibration that cannot be maintained, or a biocompatibility or sterilization limitation. A weak answer rejects on price or size alone.

**11. How do you compare two sensors that report the same quantity?**
A strong answer compares the intermediate quantity, the calibration cohort, the confounders, the reference method, and the subgroup results, and uses paired measurements. A weak answer compares the headline accuracy numbers.

**12. What is the role of the reference method in a validation?**
A strong answer states that the reference must be more accurate than the sensor by a meaningful factor, time-aligned, and independent, and that its uncertainty must be in the budget. A weak answer treats the reference as ground truth.

## 15. Key terms

- **A<sub>rms</sub>** — root-mean-square difference between a measured and reference value, in % or units.
- **Biocompatibility** — compatibility of a material with the body for a stated contact and duration.
- **Bias** — systematic difference between a measured and reference value.
- **Calibration** — establishing the mapping from sensor output to a reference quantity.
- **CMUT** — capacitive micromachined ultrasonic transducer.
- **Cross-sensitivity** — response to a non-target quantity.
- **Dark current** — photodetector current with no illumination.
- **Drift** — change in output over time at constant input.
- **Hysteresis** — difference between increasing and decreasing input paths.
- **ISFET** — ion-sensitive field-effect transistor.
- **Lag** — delay between a change in the true quantity and the reported value.
- **MARD** — mean absolute relative difference; a CGM accuracy metric.
- **Measurand** — the quantity intended to be measured.
- **MEMS** — micro-electromechanical system.
- **Noise floor** — noise power in the absence of signal.
- **Perfusion index** — pulsatile fraction of the DC optical signal.
- **Photoplethysmography (PPG)** — optical measurement of pulsatile blood volume.
- **Piezoresistive** — resistance change with mechanical strain.
- **Precision** — closeness of repeated measurements to each other.
- **Responsivity** — photocurrent per unit optical power, in A/W.
- **Sensitivity** — output change per unit input change.
- **Shot noise** — noise from discrete charge carriers, √(2qIB).
- **Signal quality index** — computed measure of artefact or reliability.
- **SpO₂** — peripheral oxygen saturation estimated by pulse oximetry.
- **Sterilization** — killing or removing microorganisms from a device.
- **Thermistor** — temperature-dependent resistor.
- **Time constant** — time to reach 63% of a step response.
- **Transducer** — element that converts one form of energy into another.
- **Transimpedance amplifier (TIA)** — current-to-voltage converter.
- **Use life** — period over which a sensor performs within specification after use begins.
- **Linearity** — deviation of the sensor response from a straight line.
- **Selectivity** — ability to respond to the target rather than to interferents.
- **Shelf life** — period over which an unused sensor remains within specification.
- **Cross-talk** — interference between channels or sensors.
- **Reference method** — the comparator used to establish accuracy.
- **Response time** — time to reach a stated fraction of a step change.
- **Offset** — output at zero input.
- **Sensitivity tolerance** — the allowed variation of sensitivity across units and conditions.
- **Dark channel** — a measurement made with the source off to estimate ambient light.
- **Perfusion** — blood flow to the tissue, which affects the pulsatile optical signal.
- **Vasoconstriction** — narrowing of blood vessels, which reduces perfusion and PPG amplitude.
- **Emissivity** — efficiency of a surface in emitting infrared radiation; affects IR thermometry.
- **Michaelis-Menten kinetics** — enzyme rate model that becomes nonlinear at high substrate concentration.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards and regulation**

1. ISO 80601-2-61:2017, *Medical electrical equipment — Part 2-61: Particular requirements for pulse oximeter equipment*. https://www.iso.org/standard/67963.html
2. ISO 80601-2-61:2026, revised pulse oximeter standard (published 2026; the 2017 edition was withdrawn on 10 April 2026). ISO catalogue entry for the 2017 edition above; consult ISO for the 2026 edition.
3. IEC 60601-1:2005+AMD1:2012+AMD2:2020, *General requirements for basic safety and essential performance*. https://webstore.iec.ch/en/publication/26169
4. IEC 60601-1-2:2014+AMD1:2020, *Electromagnetic disturbances — Requirements and tests*. https://webstore.iec.ch/en/publication/67554
5. ISO 10993-1:2025, *Biological evaluation of medical devices — Part 1: Evaluation and testing within a risk management process*. https://www.iso.org/standard/87505.html
6. ISO 11135:2014, *Sterilization of health-care products — Ethylene oxide*. https://www.iso.org/standard/56137.html
7. ISO 11137-1:2006+AMD1:2013, *Sterilization of health care products — Radiation*. https://www.iso.org/standard/33952.html
8. ISO 17665-1:2006, *Sterilization of health care products — Moist heat*. https://www.iso.org/standard/43187.html
9. U.S. Food and Drug Administration, *Pulse Oximeters for Medical Purposes: Non-Clinical and Clinical Performance Testing, Labeling, and Premarket Submission Recommendations*, draft guidance, January 2025. https://www.fda.gov/media/184896/download
10. U.S. Food and Drug Administration, *Pulse Oximeters — Premarket Notification Submissions [510(k)s]*, guidance, 2013. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pulse-oximeters-premarket-notification-submissions-510ks-guidance-industry-and-food-and-drug
11. Regulation (EU) 2017/745 on medical devices. https://eur-lex.europa.eu/eli/reg/2017/745/oj

**Clinical and technical evidence**

12. "Accuracy of Pulse Oximetry by Skin Pigmentation: Systematic Review and Meta-analysis," *Journal of Medical Internet Research*, 2024. https://www.jmir.org/2024/1/e62769
13. L. Heinemann et al., "Benefits and Limitations of MARD as a Performance Parameter for Continuous Glucose Monitoring in the Interstitial Space," *Journal of Diabetes Science and Technology*, 2020. https://journals.sagepub.com/doi/full/10.1177/1932296819855670
14. D. B. Keenan et al., "Delays in Minimally Invasive Continuous Glucose Monitoring Devices: A Review of Current Technology," *Journal of Diabetes Science and Technology*, 2009. https://ncbi.nlm.nih.gov/pmc/articles/PMC2769894/
15. A. S. Y. et al., "Selecting the Appropriate Continuous Glucose Monitoring System — a Practical Approach," 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC5954591/
16. Butterfly Network, Butterfly iQ3 launch and FDA clearance, January 2024. https://www.butterflynetwork.com/press-releases/butterfly-network-launches-butterfly-iq3
17. Siemens Healthineers, Naeotom Alpha photon-counting CT FDA clearance. https://www.siemens-healthineers.com/en-us/press-room/press-releases/naeotom-family-fda-clearance

**Background**

18. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
19. R. W. Boyd, *Nonlinear Optics*, 3rd ed., Academic Press, 2008 (for optical background).
20. Texas Instruments, *ADS129x datasheet*, SBAS502C, 2020 (example of integrated biopotential front end). https://www.ti.com/lit/ds/symlink/ads1292.pdf
21. U.S. Food and Drug Administration, *Medical Device Development Tools*, and the 510(k) database for device-specific summaries. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm
22. European Commission, *Medical Device Regulation (EU) 2017/745*, harmonised standards list. https://ec.europa.eu/growth/tools-databases/nando/
23. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
24. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
25. World Health Organization, *Medical device technical series: pulse oximeters*, 2021. https://www.who.int/publications/i/item/9789240023396
26. ISO 14117:2019, *Active implantable medical devices — Electromagnetic compatibility*. https://www.iso.org/standard/65097.html
27. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
28. U.S. Food and Drug Administration, *Design Considerations and Pre-market Submission Recommendations for Interoperable Medical Devices*, guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/design-considerations-and-premarket-submission-recommendations-interoperable-medical-devices

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_the-signal-chain.md`](01_the-signal-chain.md)
- Next topic: [`03_analog-front-ends.md`](03_analog-front-ends.md)
- Related: [`04_adcs-and-signal-conditioning.md`](04_adcs-and-signal-conditioning.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on PPG and skin tone: see Lab 14A.2 in the survey file.
- Lab on ECG quality: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)
- Lab on signal-chain simulation: see Lab 14A.1 in the survey file.
- Lab on device teardown: see Lab 14A.6 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
