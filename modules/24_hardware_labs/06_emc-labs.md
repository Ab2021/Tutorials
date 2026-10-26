# EMC Labs

> Module 24 — Hardware and Robotics Labs
> Topic 6 of 9
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 14A
> Version 2026.3

## 1. Why this topic exists

Electromagnetic compatibility (EMC) is the property of a device to work in its electromagnetic environment without disturbing other devices. It has two halves: emissions, which are what the device sends out, and immunity, which is what the device tolerates. A medical device must satisfy both.

The industry problem is that EMC is treated as a test at the end of a project rather than a design property. A device that works on the bench fails radiated emissions because of a cable, fails immunity because of a filter that was omitted, or passes at 3 V/m and fails at 10 V/m in the home. Each failure costs a board revision, a re-test, and a schedule delay.

Three decisions depend on this topic:

1. **Which environment and which test levels apply.** A professional healthcare device and a home healthcare device face different immunity levels. A device used in both must meet the harsher level.
2. **Which coupling mechanism dominates.** A conducted disturbance enters through a cable; a radiated disturbance enters through a field; a transient enters through a discharge. The mitigation depends on the mechanism.
3. **What a pre-compliance result establishes.** A pre-compliance scan finds weaknesses before an accredited booking. It is not a compliance test, and a pass in a teaching lab has no regulatory standing.

The commercial stakes are concrete. A late EMC redesign can consume a quarter of engineering time. A test-house booking that has to be repeated costs a week and a five-figure sum. A field failure caused by immunity can be a recall.

This topic builds the EMC mental model, the lab procedure for a safe tabletop experiment, and the pre-compliance plan. It assumes the safety boundary from Topic 1 and the measurement discipline from Topics 2 and 3.

The lab is deliberately modest. It does not use an anechoic chamber, an ESD gun, or a calibrated field probe. It uses a signal generator, a coupling capacitor, a scope, and a spectrum analyser or a near-field probe. The modest setup is enough to demonstrate the mechanisms, and it is honest about what it cannot establish.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work. An ESD generator, an electrical fast transient generator, or any radio-frequency power source is used only under qualified supervision in a shared laboratory, with a written test plan and a shielded enclosure where required. Do not radiate radio frequency outside a shielded enclosure or without authorisation.

## 2. Learning outcomes

By the end of this topic you can:

- Explain the four coupling mechanisms and identify the dominant one in a given bench setup.
- State the IEC 60601-1-2 immunity levels for the professional and home healthcare environments and the environment each applies to.
- Distinguish a pre-compliance measurement from an accredited type test and state what each can establish.
- Characterise a victim circuit with a baseline noise floor, a transfer function, and a susceptibility threshold.
- Measure the rectified offset produced by a conducted disturbance and compare it with the signal of interest.
- Quantify the effect of a ferrite bead, a common-mode choke, a shielded cable, and a transient suppressor.
- Model an ESD current waveform and compute its peak and its energy into a defined load.
- Write a pre-compliance plan with borrowed limits, a test setup, a margin target, and redesign triggers.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

EMC is a source-path-victim problem. The source generates energy, the path couples it, and the victim responds. A mitigation acts on one of the three.

### 3.1 The coupling mechanisms

```text
  SOURCE                     PATH                       VICTIM
  switching supply  --->  conducted (galvanic)  --->  ADC input
  radio             --->  radiated (field)      --->  cable, PCB trace
  ESD               --->  capacitive discharge  --->  enclosure, connector
  motor             --->  inductive coupling    --->  loop area
  two circuits      --->  common impedance      --->  shared ground return
```

| Mechanism | Coupling path | Frequency range | Typical mitigation |
|---|---|---|---|
| Conducted, differential | Wires carrying the disturbance | 150 kHz–30 MHz | Differential filter, choke |
| Conducted, common mode | Cable to earth capacitance | 150 kHz–30 MHz | Common-mode choke, Y-capacitor |
| Radiated, field to cable | Antenna effect of a cable | 30 MHz–2.7 GHz | Shield, ferrite, shorten cable |
| Radiated, field to PCB | Trace or loop antenna | 30 MHz–2.7 GHz | Ground plane, loop area reduction |
| Capacitive | Nearby conductor | 1 kHz–1 GHz | Shield, distance, guard |
| Inductive | Nearby current loop | 1 kHz–100 MHz | Twisted pair, loop area, distance |
| Common impedance | Shared ground or supply return | DC–100 MHz | Separate returns, star point, plane |
| ESD | Direct discharge or field | DC–1 GHz | TVS, enclosure, ground path |

The mechanism determines the fix. A ferrite bead on a differential pair does not attenuate a common-mode current. A shielded cable with a pigtail ground is an antenna. A Y-capacitor that fixes common-mode emissions can violate the leakage-current limit.

### 3.2 The frequency map

EMC is a frequency problem. The same component behaves differently across the spectrum.

| Frequency | Dominant issue | Component behaviour |
|---|---|---|
| 50/60 Hz | Magnetic field, leakage current | Transformer core, Y-capacitor |
| 10–150 kHz | Harmonics, flicker | Inductor impedance, capacitor ESR |
| 150 kHz–30 MHz | Conducted emissions and immunity | Filter parasitics, cable resonance |
| 30 MHz–1 GHz | Radiated emissions and immunity | PCB traces, cable antennas, seams |
| 1–6 GHz | Radiated immunity, wireless coexistence | Enclosure apertures, absorber |

A filter that works at 150 kHz may be useless at 30 MHz because the capacitor becomes inductive and the inductor becomes capacitive. The self-resonant frequency of each component is part of the design.

### 3.3 The immunity performance criteria

An immunity test does not require the device to be unaffected. It requires the device to meet a defined performance criterion during and after the test.

| Criterion | Requirement | Example |
|---|---|---|
| A | Normal performance during the exposure | ECG waveform unchanged |
| B | Temporary degradation, self-recovering | Display flickers, then returns |
| C | Temporary degradation requiring operator intervention | Alarm requires reset |
| D | Loss of function requiring repair | Device damaged |

The criterion must be defined before the test. A device that resets during the test and recovers has met a B criterion and failed an A criterion. The choice of criterion is a clinical decision, not an EMC decision.

### 3.4 The pre-compliance boundary

Pre-compliance testing is early, in-house, margin-seeking testing. It uses the same physics as a compliance test, at a lower cost and with less control.

| Aspect | Pre-compliance | Accredited type test |
|---|---|---|
| Location | Engineering bench | Accredited laboratory |
| Equipment | LISN, spectrum analyser, near-field probe, CDN | Calibrated chamber, antennas, generators |
| Calibration | Bench calibration, often not accredited | Accredited calibration |
| Limits | Borrowed from the standard | The standard's limits |
| Configuration | Representative | Worst case, defined by the standard |
| Result | Margin estimate | Pass or fail with a report |
| Regulatory standing | None | Evidence for conformity |

A pre-compliance result is useful for finding a 6 dB failure before the chamber. It is not useful for claiming compliance.

### 3.5 The design review points

EMC is cheapest to fix at the schematic, then at the layout, then at the enclosure, then at the test. The cost rises by roughly an order of magnitude at each stage.

```text
schematic --> layout --> enclosure --> pre-compliance --> accredited test
   low cost                                            high cost
```

The review at each stage asks the same questions: what are the sources, what are the paths, what are the victims, and what is the evidence that the mitigation works?

### 3.6 The three quantities of a pre-compliance scan

A scan produces three quantities: the ambient, the device emission, and the limit. The margin is the difference between the emission and the limit, after subtracting the ambient contribution.

\[
E_{\text{device}} = \sqrt{E_{\text{measured}}^2 - E_{\text{ambient}}^2}
\]

The subtraction is in power, not in decibels. A measured level that is 3 dB above the ambient is only 1.2 dB above the device's true emission, and the uncertainty is large. A measurement within 6 dB of the ambient is not a reliable emission measurement.

### 3.7 The configuration is part of the result

An EMC measurement is not a property of the device alone. It is a property of the device, the cable, the enclosure, the operating mode, and the test setup.

| Variable | Effect on the measurement | How to control it |
|---|---|---|
| Cable length | Changes the resonance and the coupling | Fix the length and record it |
| Cable routing | Changes the loop area | Fix the route with a template |
| Enclosure | Changes the shielding and the seams | Test the final enclosure |
| Operating mode | Changes the sources | Test the worst case |
| Display state | Changes the clock and the driver | Test the brightest and the dimmest |
| Radio state | Changes the emissions | Test transmit and receive |
| Supply | Changes the switching harmonics | Test the worst-case supply |
| Temperature | Changes the oscillator and the filter | Test at the extremes |

A result that does not name these variables is not reproducible. A result that names them but tests only one configuration is a best case.

### 3.8 The cost curve of an EMC fix

The cost of an EMC fix rises with the stage at which it is found.

| Stage | Typical fix | Relative cost | Schedule impact |
|---|---|---|---|
| Schematic | Add a filter, choose a slower clock | 1 | None |
| Layout | Move the filter, change the ground | 3 | Days |
| Enclosure | Add a gasket, change the seam | 10 | Weeks |
| Pre-compliance | Add a component, re-spin the board | 30 | Weeks |
| Accredited test | Re-spin, re-book, re-test | 100 | Months |
| Field | Recall or field correction | 1000 | Months to years |

The numbers are illustrative, not measured. The pattern is the point: the earlier the fix, the cheaper it is.

### 3.9 An immunity test is a stress test, not a performance test

An immunity test does not measure how well the device performs. It measures whether the device meets a defined criterion while it is stressed. The distinction matters because a device can pass the test with a degraded performance that is still within the criterion, and can fail the test with a performance that is clinically acceptable but outside the criterion.

The criterion must therefore be written in clinical terms, not in electrical terms. "The ECG waveform remains visible and the heart rate does not change by more than 5 bpm" is a criterion. "The output does not saturate" is an electrical condition that may or may not correspond to the clinical requirement.

The test level and the criterion are a pair. A high level with a weak criterion is not a strong test, and a low level with a strict criterion may be stricter than the standard requires.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| EMC | Electromagnetic compatibility | Used as a synonym for emissions | None |
| EMI | Electromagnetic interference | Confused with EMC | None |
| Emissions | Energy a device sends into the environment | Confused with immunity | dBµV or dBµV/m |
| Immunity | Ability to tolerate electromagnetic disturbance | Confused with emissions | V/m or V |
| Susceptibility | Tendency to respond to a disturbance | Confused with immunity | Units of the victim |
| Coupling | Transfer of energy from a source to a victim | Assumed to be one mechanism | None |
| Conducted | Coupling through a conductor | Confused with radiated | dBµV |
| Radiated | Coupling through an electromagnetic field | Confused with conducted | dBµV/m |
| Common mode | Current or voltage common to both conductors | Confused with differential | A or V |
| Differential mode | Current or voltage between two conductors | Confused with common mode | A or V |
| Common-mode choke | Component that impedes common-mode current | Assumed to impede differential | Ω at frequency |
| Ferrite bead | Lossy component that absorbs high-frequency energy | Assumed to be a pure inductor | Ω at frequency |
| Y-capacitor | Capacitor from line to earth | Assumed to be a filter capacitor | nF, leakage limit |
| X-capacitor | Capacitor across the line | Confused with Y-capacitor | µF |
| TVS | Transient voltage suppressor | Assumed to protect against all transients | V, W |
| LISN | Line impedance stabilisation network | Called a filter | Ω, dB |
| CDN | Coupling/decoupling network | Confused with a LISN | Ω, dB |
| Near-field probe | Small antenna for locating a source | Assumed to measure field strength | dBµV, relative |
| Spectrum analyser | Instrument that measures amplitude versus frequency | Assumed to be calibrated for absolute field | dBµV |
| Resolution bandwidth | Analyser filter width | Confused with the signal bandwidth | Hz |
| Quasi-peak detector | Detector weighted by pulse repetition rate | Confused with peak | dBµV |
| Peak detector | Detector that captures the maximum | Confused with quasi-peak | dBµV |
| Average detector | Detector that averages the signal | Confused with RMS | dBµV |
| Ambient | Background electromagnetic level | Ignored in a scan | dBµV |
| Margin | Difference between emission and limit | Confused with uncertainty | dB |
| Pre-compliance | In-house margin testing | Presented as compliance | None |
| Type test | Accredited conformity test | Confused with pre-compliance | None |
| ESD | Electrostatic discharge | Treated only as a device-damage issue | kV, A |
| EFT | Electrical fast transient | Confused with surge | kV |
| Surge | Slow high-energy transient | Confused with EFT | kV |
| Burst | Repetitive transient packet | Confused with a single pulse | ms |
| Dwell time | Time at each test frequency | Confused with sweep time | s |
| Modulation | Amplitude or frequency variation of the test signal | Ignored in the test plan | % or Hz |
| Electrosurgery | High-frequency surgical energy | Assumed to be a radio signal | 200 kHz–3 MHz |
| Rectification | Conversion of an AC disturbance to DC by a nonlinear junction | Ignored in analog design | V |
| Blanking | Temporary disablement of a measurement during a disturbance | Confused with filtering | ms |
| Shielding effectiveness | Attenuation of a shield | Quoted without frequency | dB |
| Transfer impedance | Shield coupling parameter | Ignored in cable selection | Ω/m |
| Ground loop | Loop formed by multiple ground connections | Ignored in cable routing | Area |
| Pigtail | Short unshielded connection from a shield to ground | Assumed to be adequate | mm |

## 5. The mechanics, step by step

The procedure below runs a safe tabletop EMC lab. It has a preparation phase, a characterisation phase, an injection phase, a mitigation phase, and a planning phase. Steps that require an ESD generator, an EFT generator, or a radio-frequency source are marked as supervised-only.

### Step 1. Define the device and the environment

**Input:** the device function, the intended use, and the use location.

**Output:** an environment classification and a list of applicable test categories.

**Rationale:** the environment determines the immunity levels. A home healthcare device must meet 10 V/m radiated RF, while a professional healthcare device must meet 3 V/m. A device used in both must meet the harsher level.

| Environment | Radiated RF immunity | Conducted RF immunity | ESD | Magnetic field |
|---|---:|---:|---:|---:|
| Professional healthcare facility | 3 V/m, 80 MHz–2.7 GHz | 3 V RMS, 150 kHz–80 MHz | ±8 kV contact, ±15 kV air | 30 A/m |
| Home healthcare | 10 V/m, 80 MHz–2.7 GHz | 3 V RMS, 150 kHz–80 MHz, 6 V in ISM bands | ±8 kV contact, ±15 kV air | 30 A/m |

The levels are from IEC 60601-1-2 Edition 4.1. The exact table, the modulation, and the exceptions must be checked against the current edition. A device with a radio transmitter has additional requirements.

### Step 2. Build the applicable test list

**Input:** the environment and the device's ports.

**Output:** a table of test categories, standards, levels, and performance criteria.

**Rationale:** the test list is the scope of the pre-compliance plan. A category that is omitted is a weakness that will not be found.

| Category | Standard | Level (professional) | Level (home) | Port |
|---|---|---|---|---|
| ESD | IEC 61000-4-2 | ±8 kV contact, ±15 kV air | Same | Enclosure |
| Radiated RF | IEC 61000-4-3 | 3 V/m | 10 V/m | Enclosure |
| EFT | IEC 61000-4-4 | ±2 kV | ±2 kV | Mains and signal |
| Surge | IEC 61000-4-5 | ±1 kV differential, ±2 kV common | Same | Mains |
| Conducted RF | IEC 61000-4-6 | 3 V RMS | 3 V RMS, 6 V ISM | Mains and signal |
| Magnetic field | IEC 61000-4-8 | 30 A/m | 30 A/m | Enclosure |
| Voltage dips | IEC 61000-4-11 | Defined dips | Defined dips | Mains |
| Proximity field | IEC 61000-4-39 | 65 A/m at 134 kHz, 7.5 A/m at 13.56 MHz | Plus 8 A/m at 30 kHz | Enclosure |

The table is a teaching summary. The current edition's Table 4, Table 5, Table 8, Table 9, and Table 11 are the normative source, and the applicability conditions are part of the requirement.

### Step 3. Build the victim circuit

**Input:** a representative analog input, a cable, and a ground return.

**Output:** a victim that is visible, probeable, and representative.

**Rationale:** a pre-compliance measurement is only as good as the victim. A victim that is not representative produces a margin that does not transfer to the product.

The victim contains:

- A 12-bit ADC input with a 10 kΩ source impedance.
- A 10 cm unshielded lead, representative of a sensor cable.
- A ground return that is visible and probeable.
- A test point at the ADC input and at the cable entry.
- A way to disconnect the lead without moving the board.

The victim is not a product. It is a physical model of the coupling path.

### Step 4. Measure the ambient

**Input:** the victim, powered but not stimulated, and the measurement instrument.

**Output:** an ambient spectrum and a time-domain noise floor.

**Rationale:** the ambient sets the detection limit. A device emission that is below the ambient cannot be measured without a shielded room or a different method.

Procedure:

1. Connect the probe or the near-field probe.
2. Measure the spectrum from 150 kHz to 1 GHz with the device off.
3. Measure again with the device on but idle.
4. Record the ambient peaks and their frequencies.
5. Record the resolution bandwidth, the detector, and the sweep time.

The ambient in a teaching lab is often dominated by the mains, the lighting, the wireless access points, and the computers. A measurement at a frequency where the ambient is high is not a measurement of the device.

### Step 5. Characterise the victim's baseline

**Input:** the victim, powered, with a quiet input.

**Output:** the noise floor, the transfer function, and the susceptibility threshold.

**Rationale:** the baseline is the reference for every mitigation comparison. Without it, a change cannot be attributed to the mitigation.

Procedure:

1. Terminate the input with the source impedance.
2. Capture 10 s of data at 1 kS/s.
3. Compute the spectrum and the time-domain standard deviation.
4. Inject a small sine at the input and measure the gain at each frequency.
5. Increase the injected amplitude until the output distorts, and record the threshold.
6. Repeat with the lead connected and disconnected.

The difference between the connected and disconnected cases is the coupling of the lead. A lead that increases the noise floor by 6 dB is a lead that will pick up interference.

### Step 6. Inject a conducted disturbance at a safe level

**Input:** the victim, a signal generator, and a coupling network.

**Output:** the rectified offset and the recovery time at each frequency and amplitude.

**Rationale:** a low-level injection demonstrates the mechanism without the hazards of a full EMC test. The mechanism is the same: a nonlinear junction rectifies the disturbance and shifts the baseline.

Procedure:

1. Set the generator to a sine at 1 kHz and 1 V RMS.
2. Couple the signal into the lead through a 10 nF capacitor.
3. Measure the ADC input with a scope and the digitised output.
4. Increase the frequency in decades from 1 kHz to 10 MHz.
5. At each frequency, record the DC shift, the AC amplitude, and the recovery time.
6. Reduce the amplitude until the shift disappears, and record the threshold.
7. Repeat with the lead shielded, with a ferrite bead, and with a common-mode choke.

The generator's output must be below the victim's damage threshold. A 1 V RMS sine into a 10 kΩ input is 100 µW, which is safe for the input. A 10 V RMS sine may not be.

**Stop point:** do not use an EFT generator or an ESD gun without supervision. The generators produce kilovolt transients that can damage the victim, the instrument, and the operator.

### Step 6A. Run a pre-compliance emissions scan

**Input:** the powered victim, a LISN or a near-field probe, and a spectrum analyser.

**Output:** a spectrum with the top offenders, their frequencies, and their margins against a borrowed limit.

**Rationale:** the immunity experiment shows how the device responds to a disturbance. The emissions scan shows how the device disturbs others. The two are different mechanisms and need different measurements.

Procedure:

1. Connect the device to the mains through a LISN, if the device has a mains port. For a battery device, use a near-field probe.
2. Set the analyser to the conducted range, 150 kHz to 30 MHz, with a 9 kHz resolution bandwidth.
3. Use the peak detector for a fast scan, then the quasi-peak detector at the highest peaks.
4. Record the ambient with the device off.
5. Identify the top five peaks and their frequencies.
6. Compare each peak with the borrowed limit and compute the margin.
7. Move the near-field probe along the board and the cables to locate the source.
8. Record the probe position, the cable routing, and the operating mode.

The near-field probe is not calibrated. It measures relative field, not absolute field strength. It is useful for locating a source, not for demonstrating compliance. The LISN measurement is closer to the standard, but it is still a bench measurement unless the laboratory, the calibration, and the configuration meet the standard.

### Step 7. Model the ESD waveform

**Input:** the standard's waveform parameters and the victim's input impedance.

**Output:** the peak current, the rise time, the decay time, and the energy into a defined load.

**Rationale:** the ESD waveform is fast and high-current. A model shows why a small capacitor and a short ground path matter.

The IEC 61000-4-2 contact-discharge waveform is commonly modelled as a double exponential:

\[
i(t) = I_{pk}\left(1 - e^{-t/t_r}\right) e^{-t/t_d}
\]

with a peak current of about 30 A for an 8 kV contact discharge into a 2 Ω load, a rise time of about 0.8 ns, and a decay time of about 60 ns. The exact waveform and the calibration are defined in the standard.

The energy into a resistive load is:

\[
E = \int i(t)^2 R\, dt
\]

For a 50 Ω load, the energy is larger than for a 2 Ω load because the voltage is higher, even though the current is lower. The exact partitioning depends on the source and the load, and the standard defines the calibration load. The model is a teaching approximation, not a calibration.

### Step 8. Model the rectification

**Input:** the disturbance waveform and the nonlinear junction.

**Output:** the DC shift produced by the rectification.

**Rationale:** a semiconductor junction rectifies an AC disturbance. The rectified current charges a capacitor and shifts the baseline. The shift is often larger than the signal.

A simple model is a diode with a forward voltage \(V_f\) and a junction capacitance \(C_j\), fed through a source impedance \(R_s\), with a load capacitance \(C_L\). The rectified voltage is approximately:

\[
V_{\text{shift}} \approx \frac{V_{pk} - V_f}{1 + R_s C_L \omega}
\]

for a half-wave rectifier, with the frequency dependence coming from the RC time constant. The model is crude; it shows the mechanism and the direction of the effect.

### Step 9. Compare the shift with the signal

**Input:** the measured or modelled shift and the signal amplitude.

**Output:** a ratio and a verdict.

**Rationale:** an EMC defect matters when the disturbance is comparable to the signal. A 1 mV shift on a 1 V signal is negligible; a 1 mV shift on a 1 mV signal is a failure.

| Condition | Shift | Signal | Ratio | Verdict |
|---|---:|---:|---:|---|
| Baseline | 2 µV | 1 mV | 0.002 | Negligible |
| 1 V RMS at 1 MHz, unshielded | 120 mV at ADC, 1.2 mV input-referred | 1 mV | 1.2 | Fail |
| With ferrite bead | 18 mV at ADC, 0.18 mV input-referred | 1 mV | 0.18 | Marginal |
| With shielded cable and choke | 6 mV at ADC, 0.06 mV input-referred | 1 mV | 0.06 | Pass |

The input-referred shift is the ADC shift divided by the gain. The comparison must use the same units.

### Step 10. Test the mitigations

**Input:** the victim and a set of mitigation components.

**Output:** a table of insertion loss and residual error for each mitigation.

**Rationale:** a mitigation is only useful if it reduces the disturbance without damaging the signal. The table quantifies both.

| Mitigation | Insertion loss at 1 MHz | Residual shift | Signal bandwidth | Cost | Board area |
|---|---:|---:|---:|---|---|
| Ferrite bead | 12 dB | 0.18 mV | Unchanged | Low | Small |
| Common-mode choke | 20 dB | 0.06 mV | Unchanged | Medium | Medium |
| Shielded cable | 25 dB | 0.04 mV | Unchanged | High | None |
| RC filter, 10 nF | 18 dB | 0.08 mV | 1.6 kHz | Low | Small |
| TVS diode | Clamps | 0.10 mV | Slight capacitance | Low | Small |

The RC filter reduces the signal bandwidth. For an ECG with a 40 Hz band, a 1.6 kHz corner is acceptable; for a signal with content at 10 kHz, it is not. The mitigation must be compatible with the signal.

### Step 11. Compare the mitigation with the mechanism

**Input:** the measured effect and the coupling mechanism.

**Output:** a statement of which mechanism the mitigation addresses.

**Rationale:** a mitigation that does not address the dominant mechanism is a coincidence. The ferrite bead addresses a common-mode current on the cable; the RC filter addresses a differential disturbance at the input. If the dominant mechanism is a radiated field into the PCB loop, neither will help much.

### Step 12. Write the pre-compliance plan

**Input:** the environment, the test list, the victim, the measurements, and the mitigations.

**Output:** a plan with the setup, the limits, the expected margin, the in-house versus accredited split, and the redesign triggers.

**Rationale:** the plan is the deliverable. It must state what the lab can and cannot establish.

The plan contains:

- The intended environment and the applicable categories.
- The victim description and the coupling path.
- The instruments and their calibration status.
- The limits borrowed from the standard, with the edition.
- The expected margin and the uncertainty.
- The tests performed in-house and the tests reserved for the accredited laboratory.
- The redesign triggers: a margin below a defined threshold, a failure at a defined level, or a change in the enclosure or the cable.
- The statement that the result is not a compliance result.

### Step 13. Review the plan against the standard

**Input:** the plan and the standard's scope.

**Output:** a review record with gaps.

**Rationale:** a plan that omits a category is a plan that will miss a failure.

The review checks the environment, the categories, the levels, the modulation, the performance criteria, the ports, the cable configuration, and the operating modes. It also checks the calibration of the instruments and the accreditation of the laboratory.

### Step 14. Record and archive

**Input:** the data, the analysis, and the plan.

**Output:** a record that another person can follow.

**Rationale:** an EMC result is a comparison between configurations. The record must preserve the configuration, the instrument settings, and the raw data.

### 5.15 Acceptance criteria for the EMC lab

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Environment is classified | Environment table | Reclassify |
| 2 | Test categories are listed | Test list | Complete |
| 3 | Victim is documented | Schematic and photo | Document |
| 4 | Ambient is measured | Spectrum | Measure |
| 5 | Baseline is measured | Noise floor | Measure |
| 6 | Injection level is stated | Generator setting | State |
| 7 | Rectified shift is measured | Scope and ADC data | Measure |
| 8 | Shift is compared with the signal | Ratio table | Compute |
| 9 | At least three mitigations are compared | Mitigation table | Extend |
| 10 | Insertion loss and signal effect are both reported | Table | Complete |
| 11 | ESD model states its assumptions | Model note | State |
| 12 | Plan states it is not compliance | Plan | Correct |
| 13 | Accredited tests are identified | Plan | Complete |
| 14 | Raw data and settings are archived | Record | Archive |

### 5.15a Test record fields

An EMC measurement is a comparison between configurations. The record must preserve the configuration as well as the number.

| Field | Example | Why it matters |
|---|---|---|
| Standard and edition | IEC 60601-1-2:2014+AMD1:2020 | The levels and methods differ by edition |
| Test category | Conducted immunity, 1 MHz | Identifies the method |
| Level | 1 V RMS at the victim | The stress |
| Modulation | CW, no modulation | The test signal |
| Victim configuration | Gain 100, 10 kΩ source, 10 cm lead | The coupling path |
| Cable routing | Straight, 2 cm above the bench | The loop area |
| Instrument | Generator, scope, analyser, probe | The measurement chain |
| Calibration | Generator 1%, scope 3% | The uncertainty |
| Ambient | 50 Hz, Wi-Fi, lighting | The detection limit |
| Operating mode | Idle, display on, radio off | The source state |
| Result | 120 mV shift at the ADC | The measurement |
| Verdict | Fail against 0.1 mV input-referred | The decision |
| Limitation | Bench injection, not a type test | The boundary |

### 5.16 Branch points and stop points

**Branch A: the device has a radio transmitter.** The emissions tests include the intentional radiator, and the immunity test includes the radio's own field. The plan must add the radio-specific requirements and the coexistence tests.

**Branch B: the device is implanted or patient-connected.** The EMC test requires a patient simulator and a defined configuration. Stop; this is outside the course's safety boundary.

**Branch C: the disturbance source is an electrosurgical unit.** The unit is a high-power radio-frequency source. Do not operate one outside a clinical environment. Model the interference, and use a recorded waveform if one is available.

**Stop point: any test that requires a mains connection, an ESD gun, an EFT generator, or a radio-frequency power amplifier.** These tests require qualified supervision and a written plan. The lab procedure in this topic is limited to low-level injection and simulation.

## 6. Worked example

### 6.1 The question

A team is preparing a pre-compliance plan for a wearable ECG front end. The device is intended for the home healthcare environment. The team wants to know how a conducted disturbance at 1 MHz couples into the ADC input, whether the rectified shift is significant compared with a 1 mV ECG signal, and whether a ferrite bead and a shielded cable reduce the shift.

### 6.2 The victim

| Item | Value | Note |
|---|---|---|
| ADC | 12-bit, 3.3 V reference | 0.806 mV per code |
| Gain | 100 | Input-referred LSB 8.06 µV |
| Source impedance | 10 kΩ | Representative of a dry electrode |
| Lead | 10 cm unshielded | Representative of a sensor cable |
| Baseline noise | 2.1 µV RMS, 0.5–40 Hz | Measured |
| Injection | 1 V RMS sine, 1 kHz–10 MHz | Through a 10 nF capacitor |
| Scope | 100 MHz, 8-bit | 3% vertical accuracy |
| Signal generator | 10 MHz, 50 Ω | 1% amplitude accuracy |
| Ambient | 50 Hz and harmonics, Wi-Fi | Measured |

### 6.2a The test configuration

The configuration is part of the result. A shift measured with a different cable, a different probe, or a different operating mode is a different experiment.

| Item | Configuration | Why it matters |
|---|---|---|
| Device state | Idle, display on, radio off | The clock and the display are noise sources |
| Lead length | 10 cm | The lead is the coupling path |
| Lead routing | Straight, 2 cm above the bench | The return path and the loop area |
| Source impedance | 10 kΩ | Limits the rectified current |
| Coupling capacitor | 10 nF | Sets the injection impedance |
| Generator | 50 Ω output, 1 V RMS | The source amplitude |
| Scope probe | 10×, 100 MHz | Loading and bandwidth |
| Bench | Wood, no ground plane | The reference for the coupling |
| Ambient | 50 Hz, lighting, Wi-Fi | The detection limit |

A configuration that is not recorded cannot be repeated. The first question after a surprising result is always whether the configuration was the same.

### 6.3 The baseline

With the input terminated and the lead connected, the ADC input noise is 2.1 µV RMS over 0.5–40 Hz. With the lead disconnected, the noise is 1.7 µV RMS. The difference is the lead's pickup, which is 1.2 µV RMS in quadrature.

The baseline is the reference. A shift of 2 µV or less is within the noise; a shift of 100 µV is 48 times the noise.

### 6.4 The injection result

The generator is set to 1 V RMS and swept from 1 kHz to 10 MHz. The ADC input is measured with the scope, and the digitised output is recorded.

| Frequency | ADC shift, unshielded | Input-referred shift | Ratio to 1 mV signal | Note |
|---:|---:|---:|---:|---|
| 1 kHz | 4 mV | 40 µV | 0.04 | Below the noise |
| 10 kHz | 9 mV | 90 µV | 0.09 | Visible |
| 100 kHz | 28 mV | 280 µV | 0.28 | Significant |
| 1 MHz | 120 mV | 1.2 mV | 1.2 | Larger than the signal |
| 3 MHz | 96 mV | 0.96 mV | 0.96 | Still large |
| 10 MHz | 41 mV | 0.41 mV | 0.41 | Decreasing |

The shift peaks around 1 MHz. The peak is consistent with the lead's resonance and the rectification at the input junction. The shift at 1 MHz is 1.2 mV input-referred, which is larger than the 1 mV ECG signal.

### 6.4a Why the response peaks at 1 MHz

The shift rises from 4 mV at 1 kHz to 120 mV at 1 MHz, then falls to 41 mV at 10 MHz. The shape is the product of three frequency-dependent factors.

| Factor | Low frequency | High frequency | Effect on the shift |
|---|---|---|---|
| Lead coupling | Small | Large, until resonance | Rises with frequency |
| Rectification efficiency | Depends on the junction | Falls as the junction capacitance shunts | Falls at high frequency |
| Input network | Passes the disturbance | Shunts it to ground | Falls at high frequency |

The peak is where the rising coupling and the falling rectification cross. The frequency is a property of the lead length, the input capacitance, and the junction, not a universal constant. A different lead length would move the peak.

This is why a single-frequency test is not enough. A device that passes at 1 MHz may fail at 3 MHz, and a mitigation that works at 1 MHz may be ineffective at 10 MHz. The scan must cover the range, and the worst case must be identified.

### 6.5 The uncertainty

The scope's vertical accuracy is 3% of reading, which at 120 mV is 3.6 mV. The generator's amplitude accuracy is 1%, which contributes 1% to the input-referred shift. The gain uncertainty is 1%.

The combined relative uncertainty is:

\[
\frac{u}{V} = \sqrt{0.03^2 + 0.01^2 + 0.01^2} = 0.0332
\]

The expanded uncertainty at k = 2 is 6.6% of 120 mV, or 8.0 mV. The shift is 120 ± 8 mV, and the input-referred shift is 1.20 ± 0.08 mV.

The uncertainty does not change the conclusion. The shift is larger than the signal by a factor of 1.2, and the uncertainty is 7% of the shift.

### 6.6 The mitigation comparison

The team adds a ferrite bead, then a shielded cable, then a common-mode choke, and repeats the 1 MHz measurement.

| Configuration | ADC shift | Input-referred | Insertion loss | Signal bandwidth | Verdict |
|---|---:|---:|---:|---:|---|
| Unshielded | 120 mV | 1.20 mV | 0 dB | 40 Hz | Fail |
| Ferrite bead | 30 mV | 0.30 mV | 12 dB | 40 Hz | Marginal |
| Shielded cable | 21 mV | 0.21 mV | 15 dB | 40 Hz | Marginal |
| Shielded cable plus choke | 7 mV | 0.07 mV | 25 dB | 40 Hz | Pass |

The insertion loss is computed as:

\[
\text{IL} = 20 \log_{10}\left(\frac{V_{\text{unmitigated}}}{V_{\text{mitigated}}}\right)
\]

For the ferrite bead, \(20 \log_{10}(120/30) = 12.0\ \text{dB}\). For the shielded cable, \(20 \log_{10}(120/21) = 15.1\ \text{dB}\). For the combination, \(20 \log_{10}(120/7) = 24.7\ \text{dB}\).

The signal bandwidth is unchanged because the mitigations are common-mode and shielding measures, not differential filters. A differential RC filter would reduce the shift further but would also reduce the signal bandwidth.

### 6.7 Does the mitigation meet the requirement?

The team's internal requirement is that an injected disturbance at the home healthcare level must not shift the baseline by more than 10% of the smallest signal of interest, which is 0.1 mV input-referred.

| Configuration | Input-referred shift | Requirement | Margin | Verdict |
|---|---:|---:|---:|---|
| Unshielded | 1.20 mV | 0.10 mV | −1.10 mV | Fail |
| Ferrite bead | 0.30 mV | 0.10 mV | −0.20 mV | Fail |
| Shielded cable | 0.21 mV | 0.10 mV | −0.11 mV | Fail |
| Shielded cable plus choke | 0.07 mV | 0.10 mV | +0.03 mV | Pass |

The combination passes, with a margin of 0.03 mV, which is smaller than the measurement uncertainty of 0.08 mV. The pass is not robust. The team should add a differential filter or reduce the source impedance before claiming the requirement is met.

### 6.7a Why the margin is not the whole story

The combination passes by 3.1 dB, but the measurement uncertainty is 6.6% of the shift, which is about 0.6 dB. The pass is statistically real but practically fragile. Three other factors make it fragile.

| Factor | Effect | Why it matters |
|---|---|---|
| Unit-to-unit variation | ±3 dB in coupling | One board may fail |
| Cable variation | ±6 dB with routing | The installed cable may be worse |
| Temperature and humidity | Small for this mechanism | Not the main risk |
| Source impedance variation | Factor of 5 per decade | A dry electrode is worse |
| Production tolerances | ±2 dB in the choke | The margin may disappear |

A 3 dB margin is not enough for a production design. The usual target is 6 dB, and for a safety-related immunity function, more. The combination is a proof of the mechanism, not a design margin.

### 6.8 Two checks that would invalidate the result

**Check 1: the injection level was not calibrated at the victim.** The generator's 1 V RMS setting is the voltage at the generator's output, not at the victim's input. The 10 nF coupling capacitor and the cable impedance form a divider, and the actual voltage at the input may be different. If the actual level is 0.5 V instead of 1 V, the shift is half, and the conclusion changes. The check is to measure the voltage at the victim's input with a high-impedance probe, not at the generator's output.

**Check 2: the scope's bandwidth and probe loading changed the measurement.** The 120 mV shift is a low-frequency baseline movement, but the rectification transient has high-frequency content. A scope with a 100 MHz bandwidth and a 10× probe is adequate for the baseline; it may not capture the transient that causes it. If the transient is missed, the mechanism is inferred rather than measured. The check is to capture the transient with a wideband probe and to compare the baseline shift with the transient's integral.

### 6.9 A varied condition: the source impedance is reduced from 10 kΩ to 1 kΩ

The team replaces the 10 kΩ source with a 1 kΩ source and repeats the 1 MHz injection.

| Condition | Source impedance | ADC shift | Input-referred | Ratio to 1 mV | Verdict |
|---|---:|---:|---:|---:|---|
| Baseline | 10 kΩ | 120 mV | 1.20 mV | 1.20 | Fail |
| Varied | 1 kΩ | 24 mV | 0.24 mV | 0.24 | Marginal |
| Varied plus shield and choke | 1 kΩ | 2 mV | 0.02 mV | 0.02 | Pass |

The shift falls by a factor of 5 when the source impedance falls by a factor of 10. The rectified current is limited by the source impedance, so a lower impedance reduces the voltage developed across the input. The result moves in the direction the model predicts.

The lesson is that the sensor's source impedance is part of the EMC design. A dry electrode with a high impedance is more susceptible than a wet electrode with a low impedance, and a design that works with one may fail with the other.

### 6.10 What the example demonstrates

The arithmetic closes: a 1 V RMS injection at 1 MHz produces a 120 mV shift at the ADC, which is 1.2 mV input-referred, with an expanded uncertainty of 0.08 mV. A ferrite bead gives 12 dB of insertion loss, a shielded cable gives 15 dB, and the combination gives 25 dB, reducing the shift to 0.07 mV. A 1 kΩ source reduces the shift by a factor of 5.

The example demonstrates the difference between a mitigation and a fix. The ferrite bead reduces the shift by 12 dB but does not meet the requirement. The combination meets the requirement with a margin smaller than the uncertainty. A robust design would change the source impedance or add a differential filter, not rely on a marginal common-mode measure.

## 7. Code, query, or calculation

The program below computes the CISPR 11 Class B conducted limit, the insertion loss of a mitigation, the ESD waveform peak and energy, and the ambient subtraction. It uses only the Python standard library.

```python
import math

# --- 1. CISPR 11 Class B conducted emission limit, mains port -----------
def cispr11_class_b_qp(f_hz):
    """Quasi-peak limit in dBuV for 150 kHz to 30 MHz."""
    f_mhz = f_hz / 1e6
    if 0.15 <= f_mhz < 0.5:
        # Linear in log frequency from 66 to 56 dBuV.
        frac = math.log10(f_mhz / 0.15) / math.log10(0.5 / 0.15)
        return 66.0 - 10.0 * frac
    if 0.5 <= f_mhz < 5.0:
        return 56.0
    if 5.0 <= f_mhz <= 30.0:
        return 60.0
    raise ValueError("outside 150 kHz to 30 MHz")

print("CISPR 11 Class B conducted limit")
for f in (150e3, 300e3, 500e3, 1e6, 5e6, 10e6, 30e6):
    print(f"{f/1e3:8.0f} kHz  {cispr11_class_b_qp(f):5.1f} dBuV")

# --- 2. Margin and ambient subtraction ----------------------------------
def margin(emission_dbuv, limit_dbuv):
    return limit_dbuv - emission_dbuv

def subtract_ambient(measured_dbuv, ambient_dbuv):
    """Power subtraction; returns None if the ambient exceeds the measurement."""
    m = 10 ** (measured_dbuv / 10.0)
    a = 10 ** (ambient_dbuv / 10.0)
    if m <= a:
        return None
    return 10 * math.log10(m - a)

print("\nmeasured emission and margin")
for f, measured, ambient in ((150e3, 72.0, 40.0), (300e3, 64.0, 39.0),
                             (1e6, 44.0, 38.0), (10e6, 42.0, 36.0)):
    limit = cispr11_class_b_qp(f)
    dev = subtract_ambient(measured, ambient)
    print(f"{f/1e3:8.0f} kHz  measured {measured:5.1f}  ambient {ambient:5.1f}  "
          f"device {dev:5.1f}  limit {limit:5.1f}  margin {margin(dev, limit):+5.1f} dB")

# --- 3. Insertion loss ---------------------------------------------------
def insertion_loss(v_unmitigated, v_mitigated):
    return 20 * math.log10(v_unmitigated / v_mitigated)

print("\ninsertion loss at 1 MHz")
for name, v in (("ferrite", 30.0), ("shield", 21.0), ("shield+choke", 7.0)):
    print(f"{name:14s} {insertion_loss(120.0, v):5.1f} dB  "
          f"residual {v/100:.3f} mV input-referred")

# --- 4. ESD double-exponential model -----------------------------------
def esd_current(t, ipk=30.0, tr=0.8e-9, td=60e-9):
    return ipk * (1 - math.exp(-t / tr)) * math.exp(-t / td)

def esd_energy(r_load=2.0, t_end=300e-9, steps=200000):
    dt = t_end / steps
    e = 0.0
    peak = 0.0
    t_peak = 0.0
    for i in range(steps):
        t = i * dt
        cur = esd_current(t)
        e += cur * cur * r_load * dt
        if cur > peak:
            peak, t_peak = cur, t
    return peak, t_peak, e

peak, t_peak, energy = esd_energy()
print("\nESD model (IEC 61000-4-2-like, teaching approximation)")
print(f"peak current = {peak:.1f} A at {t_peak*1e9:.1f} ns")
print(f"energy into 2 ohm = {energy*1e3:.1f} mJ")
print(f"charge (approx) = {sum(esd_current(i*300e-9/20000) for i in range(20000))*300e-9/20000*1e6:.1f} uC")

# --- 5. Rectified shift estimate ---------------------------------------
def required_attenuation(v_before_mv, v_target_mv):
    return 20 * math.log10(v_before_mv / v_target_mv)

print("\nrequired attenuation to reach 0.10 mV input-referred")
for name, v in (("ferrite", 0.30), ("shield", 0.21), ("shield+choke", 0.07)):
    req = required_attenuation(1.20, 0.10)
    print(f"{name:14s} residual {v:4.2f} mV  needs {req:5.1f} dB  "
          f"has {required_attenuation(1.20, v):5.1f} dB  "
          f"margin {required_attenuation(1.20, v)-req:+5.1f} dB")
```

Expected output:

```text
CISPR 11 Class B conducted limit
     150 kHz   66.0 dBuV
     300 kHz   60.2 dBuV
     500 kHz   56.0 dBuV
    1000 kHz   56.0 dBuV
    5000 kHz   60.0 dBuV
   10000 kHz   60.0 dBuV
   30000 kHz   60.0 dBuV

measured emission and margin
     150 kHz  measured  72.0  ambient  40.0  device  72.0  limit  66.0  margin  -6.0 dB
     300 kHz  measured  64.0  ambient  39.0  device  64.0  limit  60.2  margin  -3.7 dB
    1000 kHz  measured  44.0  ambient  38.0  device  42.7  limit  56.0  margin +13.3 dB
   10000 kHz  measured  42.0  ambient  36.0  device  40.7  limit  60.0  margin +19.3 dB

insertion loss at 1 MHz
ferrite         12.0 dB  residual 0.300 mV input-referred
shield          15.1 dB  residual 0.210 mV input-referred
shield+choke    24.7 dB  residual 0.070 mV input-referred

ESD model (IEC 61000-4-2-like, teaching approximation)
peak current = 27.9 A at 3.5 ns
energy into 2 ohm = 0.1 mJ
charge (approx) = 1.8 uC

required attenuation to reach 0.10 mV input-referred
ferrite        residual 0.30 mV  needs  21.6 dB  has  12.0 dB  margin  -9.5 dB
shield         residual 0.21 mV  needs  21.6 dB  has  15.1 dB  margin  -6.4 dB
shield+choke   residual 0.07 mV  needs  21.6 dB  has  24.7 dB  margin  +3.1 dB
```

The program's limit interpolation gives 60.2 dBµV at 300 kHz and 56.0 dBµV at 500 kHz, which matches the CISPR 11 Class B shape used in the teaching example. The ambient subtraction shows that a 44 dBµV measurement with a 38 dBµV ambient is 42.7 dBµV after subtraction, a 1.3 dB correction. A 44 dBµV measurement with a 43 dBµV ambient would be 37.0 dBµV, a 7 dB correction, and the measurement would be unreliable because a 1 dB ambient error moves the result by several dB.

The ESD model gives a peak current of 27.9 A at 3.5 ns, a charge of about 1.8 µC, and an energy of about 0.1 mJ into 2 Ω. The peak is lower than the 30 A nominal because the discrete time step does not land exactly on the peak and the double exponential is a teaching approximation. The energy is small in absolute terms, but it is delivered in nanoseconds into a junction with a tiny thermal mass, which is why a discharge can damage a component without heating the board.

The required-attenuation table is the decision aid. The unmitigated shift is 1.20 mV, the target is 0.10 mV, and the required attenuation is 21.6 dB. The ferrite bead and the shield each fall short, and the combination exceeds the requirement by 3.1 dB. The 3.1 dB margin is smaller than the measurement uncertainty, which is why the worked example calls the pass marginal.

## 8. Data, standards, and regulatory context

### 8.1 The EMC standard family

IEC 60601-1-2 is the collateral standard for electromagnetic compatibility of medical electrical equipment. Edition 4.1 (2014 + Amendment 1:2020) specifies immunity test levels by intended environment. The standard references the IEC 61000-4 series for the test methods.

The IEC 61000-4 series defines the test methods:

| Standard | Method | Typical levels |
|---|---|---|
| IEC 61000-4-2 | Electrostatic discharge | ±8 kV contact, ±15 kV air |
| IEC 61000-4-3 | Radiated radio-frequency field | 3 V/m or 10 V/m, 80 MHz–2.7 GHz |
| IEC 61000-4-4 | Electrical fast transient | ±2 kV on signal lines, ±4 kV on mains |
| IEC 61000-4-5 | Surge | ±1 kV differential, ±2 kV common |
| IEC 61000-4-6 | Conducted radio-frequency | 3 V RMS, 150 kHz–80 MHz |
| IEC 61000-4-8 | Power-frequency magnetic field | 30 A/m |
| IEC 61000-4-11 | Voltage dips and interruptions | Defined dips |
| IEC 61000-4-39 | Proximity magnetic fields | 65 A/m at 134 kHz, 7.5 A/m at 13.56 MHz |

The levels in the table are the commonly quoted medical levels. The exact level depends on the port, the environment, and the edition. A test plan must cite the table and the edition.

### 8.2 Emissions standards

CISPR 11 is the emissions standard for industrial, scientific, and medical equipment. It classifies equipment by group and class.

- Group 1: equipment that does not intentionally generate radio-frequency energy for its own function.
- Group 2: equipment that intentionally generates radio-frequency energy, such as a radio transmitter.
- Class A: equipment for use in an industrial environment.
- Class B: equipment for use in a residential environment, including home healthcare.

The conducted emission limits apply from 150 kHz to 30 MHz on the mains port. The radiated limits apply from 30 MHz to 1 GHz, with extensions above 1 GHz for some equipment. The Class B limits are stricter than Class A.

In the United States, the Federal Communications Commission (FCC) Part 15 rules apply to unintentional radiators. A medical device may need to satisfy both the FDA-recognised IEC 60601-1-2 and the FCC rules.

### 8.3 The immunity performance criteria

IEC 60601-1-2 defines performance criteria for immunity. The criteria are defined by the manufacturer, based on the device's intended use and the risk analysis. The standard gives examples, but the manufacturer must state the criterion for each test.

A device that resets during an ESD test and recovers may meet a criterion B, which allows temporary degradation with self-recovery. A device that loses a stored setting may fail a criterion C. The criterion is a clinical decision.

### 8.4 Electrosurgery

Electrosurgical units (ESUs) generate radio-frequency energy, typically in the range of a few hundred kilohertz to a few megahertz, to cut and coagulate tissue. The output is modulated, and the waveform is not a clean sine. The interference couples into an ECG front end through the patient leads, the cables, and the enclosure.

The mechanisms are rectification at semiconductor junctions, baseline shift, and saturation. The mitigations are input filtering, shielding, physical separation, and blanking during the active phase. A blanking circuit disables the measurement for a defined interval, which reduces the interference but also loses the signal during that interval.

The exact ESU frequencies and modulation depend on the device and the mode. A lab should model the interference with a stated waveform and label it as a model, not a measurement of a specific ESU.

### 8.5 Pre-compliance versus compliance

A pre-compliance test is not a compliance test. The differences are the laboratory's accreditation, the calibration of the equipment, the control of the configuration, and the reporting. A pre-compliance result can guide a design; it cannot support a conformity claim.

The FDA recognises IEC 60601-1-2 Edition 4.1 with a partial recognition that excludes certain clauses. The recognition is recorded in the FDA's Recognized Consensus Standards database, and the exclusions must be checked for the specific device.

### 8.6 CISPR 11 groups and classes

CISPR 11 classifies equipment along two axes. The group describes whether the equipment intentionally generates radio-frequency energy for its own function. The class describes the environment for which the equipment is intended.

| Classification | Meaning | Example | Emission limits |
|---|---|---|---|
| Group 1 | Does not intentionally generate RF for its function | Most medical devices | Lower |
| Group 2 | Intentionally generates RF | A radio-frequency therapy device | Higher, with a defined fundamental |
| Class A | Industrial environment | Laboratory equipment | Less strict |
| Class B | Residential environment, including home healthcare | Home infusion pump | Stricter |

A home healthcare device is normally Group 1 Class B. A device with a radio is Group 1 or Group 2 depending on the radio's function, and the radio's intentional emissions are assessed separately from the unintentional emissions.

The conducted limits apply to the mains port; the radiated limits apply to the enclosure. A battery-powered device has no mains port, but it still has an enclosure, and it may have a charging port that is tested as a mains port.

### 8.7 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| IEC 60601-1-2 | Collateral standard | Medical electrical equipment EMC | A bench experiment |
| IEC 61000-4-2 to -4-11 | Test method standards | Accredited EMC testing | A low-level injection |
| CISPR 11 | Emissions standard | Medical equipment emissions | A pre-compliance scan |
| FCC Part 15 | US regulation | Unintentional radiators in the US | A device outside the US |
| ISO 14971 | Risk management standard | Medical device risk file | A lab hazard list |
| FDA recognition database | Regulatory recognition | Premarket submissions | A teaching lab |

## 9. Industry practice and case evidence

### 9.1 How a test house runs an EMC test

An accredited EMC laboratory works from a test plan derived from the standard and the device's intended use. The laboratory:

- Reviews the device, the cables, and the operating modes.
- Configures the device in its worst-case mode.
- Measures the emissions with a LISN and a spectrum analyser, or in a chamber with an antenna.
- Applies the immunity tests with calibrated generators and antennas.
- Records the performance criterion and the observed behaviour.
- Issues a report with the results, the conditions, and the deviations.

The laboratory's accreditation covers specific standards and specific tests. A laboratory accredited for CISPR 11 may not be accredited for IEC 61000-4-39.

### 9.2 Pre-compliance in industry

A device company typically runs a pre-compliance scan at the engineering bench before booking the chamber. The scan uses a LISN, a spectrum analyser, and a near-field probe to find the frequencies where the margin is small. The goal is to find the 6 dB failure before the chamber does.

The scan is not a substitute for the chamber. It cannot calibrate the field, it cannot reproduce the chamber's uniformity, and it cannot produce an accredited report. It can find the obvious problems and save a booking.

### 9.3 Design practices that reduce EMC risk

The practices that reduce EMC risk are mostly layout and architecture:

- A solid ground plane with short return paths.
- A split between analog and digital sections, with a single connection point.
- A filter at the connector, not at the far end of the board.
- A shield bonded at the connector, not with a pigtail.
- A clock tree with the lowest feasible edge rate.
- A switching supply with a controlled layout and a shielded inductor.
- A cable schedule with the length, the shield, and the connector defined.

The practices are cheap at the schematic stage and expensive at the enclosure stage.

### 9.4 Electrosurgery interference in practice

ECG monitors used during electrosurgery must tolerate a high-frequency disturbance that is much larger than the ECG. The monitors use input filters, shielding, and blanking. The blanking interval is a trade-off: a longer blank loses more signal, and a shorter blank lets more interference through.

The clinical risk is that the monitor displays a misleading rhythm during the blanking interval. The mitigation is a clear indication that the signal is blanked, not a fabricated waveform.

### 9.5 What a failure report contains

When an accredited test fails, the laboratory's report is the starting point for the fix. A useful report contains:

- The test category, the standard, and the level.
- The frequency or the discharge point at which the failure occurred.
- The device configuration and the operating mode.
- The observed behaviour and the performance criterion.
- The measurement trace or the capture.
- The calibration status of the test equipment.
- The laboratory's judgement of the cause, if it offers one.

A report that says only "failed radiated immunity at 3 V/m" is not enough to fix the problem. The frequency, the orientation, and the cable configuration are what identify the coupling path.

### 9.6 Questions a design review should answer

An EMC design review at the schematic and layout stages asks:

1. What are the intentional and unintentional sources, and at what frequencies?
2. What are the cables, their lengths, their shields, and their connectors?
3. Where does the current return, and is the return path close to the source?
4. Where is the filter, and is it at the connector?
5. Is the shield bonded at the connector, or is there a pigtail?
6. What is the enclosure material, and where are the seams and apertures?
7. What is the ESD path from each accessible surface?
8. What is the worst-case operating mode, and why?
9. What is the margin target, and what evidence supports it?
10. What would trigger a redesign before the chamber booking?

A review that answers these questions has a chance of passing the first time. A review that answers only the first has a filter and a hope.

### 9.7 What an EMC engineer does first

An experienced EMC engineer does not start with a filter. The first week is usually spent on the architecture:

- Locating the high-frequency sources and their return paths.
- Reviewing the cable schedule and the connector pinout.
- Checking the ground strategy and the enclosure seams.
- Choosing the switching frequencies to avoid the bands of interest.
- Defining the worst-case operating mode and the test configuration.

The filter is a late-stage refinement. The architecture determines whether a filter can work. A board with a split ground plane and a long cable will need a more expensive filter than a board with a solid plane and a short cable, and the filter may not be enough.

This is the practical meaning of the cost curve. The engineer's time in the first week is cheap; the re-spin after the chamber is not.

### 9.8 Inference

It is an inference, not a documented finding, that teams which measure the ambient before the device emission make fewer false failures. The reasoning is that a 44 dBµV measurement with a 43 dBµV ambient is mostly ambient, and a team that does not subtract it will try to fix a device that is not the source. The inference is consistent with the practice in accredited laboratories, but it has not been tested in this course.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Pre-compliance called compliance | Report says "passed EMC" | No accreditation | Regulatory rejection | State the boundary |
| 2 | Ambient not measured | A peak that appears with the device off | Ambient mistaken for emission | Wasted redesign | Measure ambient first |
| 3 | Victim not characterised | No baseline, no transfer function | Unknown susceptibility | Uninterpretable result | Characterise first |
| 4 | Injection level not measured at the victim | Generator setting used as the level | Divider and cable loss | Wrong margin | Probe at the victim |
| 5 | Mitigation tested without the signal | Filter reduces the signal too | Bandwidth loss | Functional failure | Measure both |
| 6 | Ferrite used on a differential pair | No effect on the disturbance | Wrong mode | False fix | Identify the mode |
| 7 | Shield with a pigtail | Shield current on the pigtail | Antenna effect | Worse emission | Bond at the connector |
| 8 | Y-capacitor violates leakage | Leakage current above the limit | Trade-off ignored | Safety failure | Check the leakage budget |
| 9 | ESD path missing | Discharge through the signal ground | No low-impedance path | Damage, reset | Add a TVS and a ground path |
| 10 | Test at one operating mode | Only idle tested | Worst-case mode missed | Field failure | Test all modes |
| 11 | Cable configuration not recorded | Different cable in the chamber | Coupling path changed | Non-reproducible | Record the cable schedule |
| 12 | ESD gun uncalibrated | Discharge waveform not verified | Wrong test level | Invalid result | Calibrate the generator |
| 13 | RF radiated outside a shield | Interference to nearby equipment | Uncontrolled field | Regulatory and safety issue | Use a shielded enclosure |
| 14 | Filter placed far from the connector | Noise bypasses the filter | Layout inductance | Reduced attenuation | Place at the connector |
| 15 | Ground loop through the scope | Probe ground on a live node | Earth loop | Damage, shock | Differential probe |
| 16 | Mode stirred but not documented | Different results between runs | Position dependence | Unreliable margin | Fix the position and record it |
| 17 | Spectrum analyser overloaded | A high peak that appears everywhere | Front-end compression | False emission | Add attenuation, check the mixer level |
| 18 | Quasi-peak compared with a peak limit | A margin that changes with the detector | Detector mismatch | Wrong verdict | Match the detector to the limit |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The victim represents the product | The margin transfers | A different layout or cable | Compare with the product |
| 2 | The ambient is stable | It can be subtracted | A nearby transmitter turns on | Monitor the ambient during the test |
| 3 | The injection level is known at the victim | The margin depends on it | Cable loss, divider | Probe at the victim |
| 4 | The disturbance is the dominant mechanism | The mitigation is appropriate | A different coupling path | Change one variable at a time |
| 5 | The mitigation does not change the signal | The function is preserved | A filter with a low corner | Measure the signal bandwidth |
| 6 | The ESD model is representative | It shows the mechanism | A different waveform | Compare with the standard's waveform |
| 7 | The rectification model is adequate | It shows the direction | A different junction | Measure the shift |
| 8 | The test level is below the damage threshold | The victim survives | A higher level | Use a limiter or a simulator |
| 9 | The results are repeatable | The comparison is valid | Position, cable, mode | Repeat the baseline |
| 10 | The standard's edition is current | The limits are correct | A new edition | Check the edition |
| 11 | The laboratory is accredited for the test | The result is accepted | A scope mismatch | Check the accreditation |
| 12 | The device is not patient-connected | The lab is safe | A patient connection | Stop and escalate |
| 13 | The analyser is not overloaded | The spectrum is valid | A high peak compresses the front end | False emission | Check the mixer level |
| 14 | The detector matches the limit | The margin is comparable | Peak versus quasi-peak | Wrong verdict | Use the specified detector |

## 12. Exercises

### Exercise 1. Classify the coupling (recall and application)

**Deliverable:** a table of five bench disturbances, each classified by mechanism, with the evidence and the mitigation.

**Acceptance criteria:** each row names the source, the path, and the victim; the mechanism is one of conducted, radiated, capacitive, inductive, common impedance, or ESD; the mitigation addresses the mechanism; at least one row is a common-impedance path.

**Time:** 30 minutes.

### Exercise 2. Build the test list (application)

**Deliverable:** an applicable-test table for a home healthcare ECG device, with the standard, the level, the port, and the performance criterion.

**Acceptance criteria:** the environment is stated; at least seven categories are listed; each level is cited to an edition; each performance criterion is defined; the table states which tests cannot be performed in-house.

**Time:** 45 minutes.

### Exercise 3. Measure the ambient and the baseline (application)

**Deliverable:** an ambient spectrum and a time-domain noise floor for a victim circuit, with the instrument settings.

**Acceptance criteria:** the ambient is measured with the device off and on; the resolution bandwidth and detector are stated; the baseline is measured with the input terminated; the difference between the connected and disconnected lead is reported.

**Time:** 45 minutes.

### Exercise 4. Injection and rectification (application)

**Deliverable:** a sweep of the rectified shift from 1 kHz to 10 MHz at a safe injection level, with the input-referred shift and the ratio to a 1 mV signal.

**Acceptance criteria:** the injection level is measured at the victim, not at the generator; the shift is measured in ADC codes and converted to input-referred units; the uncertainty is stated; the peak frequency is identified.

**Time:** 60 minutes.

### Exercise 5. Mitigation comparison (critique)

**Deliverable:** a comparison of three mitigations by insertion loss, residual shift, signal bandwidth, cost, and board area, with a recommendation.

**Acceptance criteria:** at least three mitigations are measured or modelled; the insertion loss is computed from the measured values; the effect on the signal is reported; the recommendation addresses the dominant mechanism.

**Time:** 60 minutes.

### Exercise 6. ESD model (calculation)

**Deliverable:** a double-exponential ESD model with the peak current, the rise time, the decay time, and the energy into 2 Ω and 50 Ω.

**Acceptance criteria:** the model is stated; the integration is shown; the peak and the energy are reported with units; the model is labelled as an approximation, not a calibration.

**Time:** 45 minutes.

### Exercise 7. Pre-compliance plan (design)

**Deliverable:** a two-page plan for a home healthcare device, with the environment, the categories, the victim, the instruments, the borrowed limits, the expected margin, the in-house and accredited split, and the redesign triggers.

**Acceptance criteria:** the plan states that it is not a compliance result; it names the accredited tests; it includes the cable schedule and the operating modes; it defines a margin threshold for redesign.

**Time:** 90 minutes.

### Exercise 8. Critique a claim (critique)

**Deliverable:** a review of a draft statement that says "the device passed EMC pre-compliance, so it is ready for the FDA," identifying the missing evidence and the correct claim.

**Acceptance criteria:** the review distinguishes pre-compliance from a type test; it names the accreditation, the standard, and the configuration that would be needed; it proposes a corrected sentence.

**Time:** 30 minutes.

## 13. Self-check answers

### 13.1 What are the four coupling mechanisms?

Conducted, radiated, capacitive, and inductive. A fifth, common impedance, is a conducted mechanism through a shared return. The mechanism determines the mitigation: a filter for conducted, a shield for radiated, a guard or distance for capacitive, a twisted pair or loop-area reduction for inductive.

### 13.2 What is the difference between emissions and immunity?

Emissions are the energy the device sends into the environment. Immunity is the device's ability to tolerate energy from the environment. A device can pass one and fail the other; the tests are separate.

### 13.3 Why is 10 V/m harder than 3 V/m?

Because the field strength is higher by a factor of 3.33, which is 10.46 dB. A device with a 6 dB margin at 3 V/m has a −4.46 dB margin at 10 V/m.

### 13.4 Compute the margin for a 64 dBµV emission against a 60.2 dBµV limit.

The margin is \(60.2 - 64.0 = -3.8\ \text{dB}\). The emission exceeds the limit by 3.8 dB, which is a failure.

### 13.5 Why is ambient subtraction done in power, not in decibels?

Because the measured level is the sum of the device and the ambient powers. Subtracting decibels would subtract logarithms, which is not the same as subtracting powers. The correct operation is \(10\log_{10}(10^{M/10} - 10^{A/10})\).

### 13.6 What is the insertion loss of a mitigation that reduces 120 mV to 30 mV?

\[
20 \log_{10}(120/30) = 20 \log_{10}(4) = 12.0\ \text{dB}
\]

### 13.7 Why does a ferrite bead not fix a differential disturbance?

Because a ferrite bead is placed around both conductors and impedes the common-mode current. A differential disturbance flows in opposite directions on the two conductors, so the bead's impedance is not in the differential path. A differential filter is needed.

### 13.8 What is the difference between a pre-compliance test and a type test?

A pre-compliance test is an in-house, margin-seeking measurement. A type test is an accredited conformity test under the standard's method, with calibrated equipment and a controlled configuration. A pre-compliance result can guide a design; it cannot support a compliance claim.

### 13.9 Why does the rectified shift peak at a specific frequency?

Because the coupling and the rectification both depend on frequency. The cable's coupling increases with frequency, the rectifier's efficiency depends on the junction and the load, and the RC time constant rolls off at high frequency. The peak is where the product is largest.

### 13.10 Rubric for Exercise 5

A strong comparison reports the insertion loss, the residual shift, the signal bandwidth, the cost, and the board area for each mitigation, and recommends one that addresses the dominant mechanism. A weak comparison reports only the insertion loss and ignores the effect on the signal.

### 13.11 Rubric for Exercise 7

A strong plan states the environment, the categories, the victim, the instruments, the limits with the edition, the margin, the in-house and accredited split, and the redesign triggers. It states that it is not a compliance result. A weak plan lists tests without levels or criteria.

### 13.12 Why is a near-field probe not a calibrated field probe?

Because it measures a relative voltage induced in a small loop or dipole, not a calibrated field strength. Its sensitivity depends on the probe geometry, the orientation, and the distance. It is useful for locating a source and comparing positions, not for demonstrating a field level.

### 13.13 Why does a lower source impedance reduce the rectified shift?

Because the rectified current is limited by the source impedance. A lower impedance allows more current for a given disturbance voltage, but the voltage developed across the input by the rectified current is also lower. The net effect in the worked example is a reduction by a factor of 5 for a tenfold reduction in impedance.

### 13.14 Why is the performance criterion a clinical decision?

Because the criterion defines what the device must do during a disturbance, and that is a statement about the clinical use. A monitor that blanks its display during electrosurgery may meet an electrical criterion and fail a clinical one. The clinician, the risk file, and the intended use determine the criterion; the EMC engineer implements it.

### 13.15 Rubric for Exercise 8

A strong critique distinguishes pre-compliance from a type test, names the missing accreditation and configuration, and proposes a corrected sentence that preserves the true finding. A weak critique says "the claim is wrong" without saying what would make it right.

## 14. Interview questions

### 14.1 What is the difference between conducted and radiated immunity?

Conducted immunity couples a disturbance through a cable or a connector. Radiated immunity couples it through an electromagnetic field. A strong answer names the test methods (IEC 61000-4-6 and IEC 61000-4-3) and the frequency ranges. A weak answer says one is wired and one is wireless.

### 14.2 Why does a home healthcare device need a higher immunity level?

Because the home environment has more consumer transmitters, more uncontrolled sources, and less separation between the device and the user. A strong answer cites the 10 V/m level and the 3 V/m professional level. A weak answer says "home is worse" without a number.

### 14.3 What is common-mode noise, and how do you attenuate it?

Common-mode noise is a current or voltage common to both conductors, usually referenced to earth. It is attenuated with a common-mode choke, a Y-capacitor, or a shield. A strong answer distinguishes it from differential noise and notes that a ferrite bead is a common-mode component. A weak answer says "use a filter."

### 14.4 Why is a shield with a pigtail ineffective?

Because the pigtail is an inductance at high frequency, so the shield current develops a voltage across it and the shield becomes an antenna. A strong answer says the shield must be bonded at the connector with a low-impedance connection. A weak answer says "the shield is grounded."

### 14.5 What does a LISN do?

It provides a defined impedance to the mains port and isolates the device from the mains supply, so that the conducted emission can be measured reproducibly. A strong answer notes that it is not a filter and that its impedance is defined by the standard. A weak answer calls it a filter.

### 14.6 Why can an ESD event reset a device without damaging it?

Because the discharge couples into a signal or reset line, causing a transient that the logic interprets as a valid edge. A strong answer names the coupling path and the mitigation (TVS, filter, layout). A weak answer says "the device is sensitive."

### 14.7 What is the difference between a peak and a quasi-peak detector?

A peak detector captures the maximum amplitude. A quasi-peak detector weights the signal by its pulse repetition rate, so a repetitive pulse reads higher than a single pulse but lower than the peak. A strong answer notes that the standard specifies the detector. A weak answer says they are the same.

### 14.8 Why does electrosurgery corrupt an ECG?

Because the high-frequency energy is rectified at semiconductor junctions in the front end, producing a DC shift and saturation. A strong answer names the mechanisms and the mitigations (filtering, shielding, blanking). A weak answer says "interference."

### 14.9 What is the first thing you measure in an EMC pre-compliance test?

The ambient. A strong answer says the ambient sets the detection limit and must be subtracted in power. A weak answer says "the emission."

### 14.10 What does a pre-compliance result not prove?

It does not prove compliance, because the laboratory is not accredited for the test, the equipment is not calibrated to the standard, and the configuration is not the worst case. A strong answer states the boundary. A weak answer says it proves the device is "probably fine."

### 14.11 How would you reduce a radiated emission at 300 MHz?

A strong answer identifies the source (a cable, a seam, a clock harmonic), measures the near field, and applies a fix to the path or the source: bond the shield, close the seam, shorten the cable, slow the clock edge, or add a ferrite. A weak answer says "add a capacitor."

### 14.12 What would you do if the pre-compliance scan passed but the accredited test failed?

A strong answer compares the configurations, the calibration, the ambient, and the test levels; it treats the difference as a configuration or measurement issue before assuming the device changed. It does not blame the laboratory without evidence.

## 15. Key terms

- **Ambient** — background electromagnetic level.
- **Blank** — temporary disablement of a measurement during a disturbance.
- **Burst** — repetitive transient packet.
- **CDN** — coupling/decoupling network.
- **CISPR 11** — emissions standard for industrial, scientific, and medical equipment.
- **Common mode** — current or voltage common to both conductors.
- **Common-mode choke** — component that impedes common-mode current.
- **Conducted** — coupling through a conductor.
- **Coupling** — transfer of energy from a source to a victim.
- **Differential mode** — current or voltage between two conductors.
- **EFT** — electrical fast transient.
- **EMC** — electromagnetic compatibility.
- **EMI** — electromagnetic interference.
- **Emissions** — energy a device sends into the environment.
- **ESD** — electrostatic discharge.
- **Ferrite bead** — lossy component that absorbs high-frequency energy.
- **Ground loop** — loop formed by multiple ground connections.
- **Immunity** — ability to tolerate electromagnetic disturbance.
- **Insertion loss** — attenuation of a mitigation, in dB.
- **LISN** — line impedance stabilisation network.
- **Margin** — difference between emission and limit.
- **Near-field probe** — small antenna for locating a source.
- **Pigtail** — short unshielded shield connection.
- **Pre-compliance** — in-house margin testing.
- **Quasi-peak** — detector weighted by pulse repetition rate.
- **Radiated** — coupling through an electromagnetic field.
- **Rectification** — conversion of AC to DC by a nonlinear junction.
- **Resolution bandwidth** — analyser filter width.
- **Shielding effectiveness** — attenuation of a shield.
- **Surge** — slow high-energy transient.
- **Susceptibility** — tendency to respond to a disturbance.
- **Transfer impedance** — shield coupling parameter.
- **TVS** — transient voltage suppressor.
- **Type test** — accredited conformity test.
- **Y-capacitor** — capacitor from line to earth.
- **Class A** — emissions class for industrial environments.
- **Class B** — emissions class for residential and home healthcare environments.
- **Group 1** — equipment that does not intentionally generate RF for its function.
- **Group 2** — equipment that intentionally generates RF for its function.
- **Dwell time** — time at each test frequency.
- **Performance criterion** — defined behaviour the device must meet during a test.
- **Detector** — circuit that converts the analyser's intermediate frequency to a displayed level.
- **Front-end overload** — compression of the analyser input by a strong signal.
- **Loop area** — area enclosed by a current path, which sets its magnetic coupling.
- **Resonance** — frequency at which a cable or structure amplifies the coupling.
- **Insertion loss** — attenuation introduced by a component or a filter.
- **Ground plane** — continuous conductor that provides a low-impedance return path.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and regulation

- IEC 60601-1-2:2014+AMD1:2020, Electromagnetic disturbances — Requirements and tests. https://webstore.iec.ch/en/publication/67554
- IEC 61000-4-2:2025, Electrostatic discharge immunity test. https://webstore.iec.ch/
- IEC 61000-4-3, Radiated, radio-frequency, electromagnetic field immunity test. https://webstore.iec.ch/
- IEC 61000-4-4, Electrical fast transient/burst immunity test. https://webstore.iec.ch/
- IEC 61000-4-5, Surge immunity test. https://webstore.iec.ch/
- IEC 61000-4-6, Immunity to conducted disturbances induced by radio-frequency fields. https://webstore.iec.ch/
- IEC 61000-4-8, Power frequency magnetic field immunity test. https://webstore.iec.ch/
- IEC 61000-4-11, Voltage dips, short interruptions and voltage variations immunity tests. https://webstore.iec.ch/
- IEC 61000-4-39, Radiated fields in close proximity. https://webstore.iec.ch/
- CISPR 11, Industrial, scientific and medical equipment — Radio-frequency disturbance characteristics. https://webstore.iec.ch/
- FCC, 47 CFR Part 15, Radio frequency devices. https://www.ecfr.gov/current/title-47/chapter-I/subchapter-A/part-15
- FDA, Recognized Consensus Standards database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfStandards/search.cfm

### Test methods and guidance

- STMicroelectronics, AN3353, IEC 61000-4-2 standard testing. https://www.st.com/resource/en/application_note/dm00023467-iec-61000-4-2-standard-testing-stmicroelectronics.pdf
- IEC 61000-4-2:2025 preview. https://assets.vde-verlag.de/iec-normen/preview-pdf/info_iec61000-4-2%7Bed3.0%7Db.pdf
- openEMS documentation. https://openems.de/
- ngspice documentation. https://ngspice.sourceforge.io/
- ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
- IEC 60601-1-2:2014+AMD1:2020, consolidated edition, IEC webstore. https://webstore.iec.ch/en/publication/67554
- CISPR 11:2015+AMD1:2016+AMD2:2019, Industrial, scientific and medical equipment. https://webstore.iec.ch/
- IEC 61000-4-3:2020, Radiated, radio-frequency, electromagnetic field immunity test. https://webstore.iec.ch/

> **Note:** The standard texts are paywalled. The application notes and previews above were used to check the teaching values. Confirm every clause, level, and table against the current edition before citing a number in a test plan or a regulatory document.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_firmware-labs.md`](05_firmware-labs.md)
- Next topic: [`07_hil-labs.md`](07_hil-labs.md)
- Sibling: [`01_safety-and-standards.md`](01_safety-and-standards.md)
- Sibling: [`04_sensor-and-signal-labs.md`](04_sensor-and-signal-labs.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
