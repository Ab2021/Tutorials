# The Signal Chain

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 01 of 14
> Estimated study time: 4 hours
> Prerequisites: 01A
> Version 2026.3

## 1. Why this topic exists

A data scientist usually meets a medical measurement as a column in a file. The column has a name, a unit, and a timestamp. It looks like a fact. It is the last surviving value of a chain of physical and electronic transformations, each of which had a range, a bandwidth, a noise floor, a delay, and a failure mode. When the model behaves strangely in deployment, the cause is often not the model. It is a stage of that chain that changed, saturated, or was never characterised.

The industry problem is attribution. A sepsis alert fires late. An arrhythmia classifier misses a wide-complex beat. A wearable heart-rate estimate degrades when the patient stands. A ventilator's tidal-volume delivery drifts. In each case the team can spend weeks tuning the algorithm before someone asks whether the input signal still contains the information the algorithm was trained on. The cost of asking late is a delayed launch, a failed verification study, or a recall.

Three decisions are blocked without signal-chain literacy:

1. **What data to request.** A vendor offers a device-computed heart rate, a filtered waveform, and a raw waveform. The three have different information content and different error structure. Choosing the wrong one determines what questions can be answered later.
2. **How to interpret model error.** A model that performs well on a public dataset and poorly on device data may be seeing a different anti-alias filter, a different gain, a different electrode, or a different patient population. Without a chain model, these are indistinguishable.
3. **What evidence a regulator will expect.** A claim about a measurement requires evidence that the measurement chain was verified across its specified range, including saturation, offset, interference, and drift. IEC 60601-2-25, for example, does not test an algorithm; it tests the electrocardiograph's response to defined inputs.

The topic is not a circuit-design course. It is a course in reading a device as a sequence of information-preserving and information-destroying operations, and in knowing which quantities must be recorded so that a later analysis can distinguish a physiological change from an instrument change.

A useful discipline: for every signal you analyse, write the chain from the patient to the number, and annotate each arrow with the quantity that could change without anyone updating the data dictionary. Most surprises live on those arrows.

## 2. Learning outcomes

By the end of this topic you can:

- Draw a medical device signal chain from transducer to clinical output and annotate each stage with its input range, output range, bandwidth, and noise contribution.
- Compute an input-referred noise budget in microvolts RMS from source impedance, amplifier voltage noise, current noise, and bandwidth, and compare it with the quantisation step of the analogue-to-digital converter (ADC).
- Convert a signal-to-noise ratio (SNR) into an effective number of bits (ENOB) and explain why a 24-bit ADC can deliver a 9-bit measurement.
- Calculate the residual common-mode interference produced by a source-impedance mismatch and decide whether the limiting factor is the amplifier's common-mode rejection ratio (CMRR) or the electrodes.
- Allocate gain so that a specified differential offset and signal fit inside the amplifier and ADC input range, and identify the offset at which the chain saturates.
- Build a latency budget from filter group delay, conversion time, processing time, and transmission time, and state whether a claimed real-time behaviour is achievable.
- List the provenance fields that must accompany a waveform so that a later analyst can detect a firmware, sensor-lot, or calibration change.
- Critique a vendor accuracy claim by asking for the test conditions, the subgroup results, and the stage of the chain at which the accuracy was measured.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A signal chain is a composition of functions. The patient produces a physical quantity \(x(t)\). Each stage applies a transfer function \(H_i\), adds noise \(n_i\), and imposes a range limit and a bandwidth limit. The device output is:

\[
y(t) = \mathrm{clip}\left(H_n * \cdots * H_1 * x(t) + \sum_i n_i(t)\right)
\]

The clip is not invertible. Neither is an anti-alias filter that removed a band before sampling, nor a lossy compression step, nor a downsampling step. Everything else is, in principle, invertible if the transfer function is known. The practical question is whether the transfer function was recorded.

```text
 Patient            Sensor / electrode        Analog front end              ADC                 Processor            Output
+---------+        +------------------+       +------------------+       +-------------+      +-------------+      +-----------+
| source  |  --->  | transduction     | --->  | amplify, filter, | --->  | sample,     | ---> | algorithm,  | ---> | display,  |
| 1 mV    |        | Z_s, offset      |       | protect, drive   |       | quantise    |      | pack, send  |      | alarm,    |
| 10 kOhm |        | +/-300 mV        |       | CMRR, noise      |       | ENOB, jitter|      | latency     |      | actuator  |
+---------+        +------------------+       +------------------+       +-------------+      +-------------+      +-----------+
     |                      |                        |                        |                    |                    |
     v                      v                        v                        v                    v                    v
  physiology           contact, motion,         clipping, CMRR          aliasing,             dropped             alarm
  bandwidth            drift, saturation         degradation             quantisation           frames              fatigue
```

Read the diagram in two directions.

**Forward (causality).** Each stage can only pass what the previous stage delivered. A 0.5 Hz high-pass filter at the front end removes the slow ST-segment information before any algorithm sees it. A gain that saturates on a 300 mV electrode offset removes the QRS entirely. A 100 Hz sampling rate aliases 50 Hz interference into a 50 Hz artefact that looks physiological. No downstream model can recover the lost information; it can only learn a correlation that may not hold on another device.

**Backward (diagnosis).** When an output is wrong, walk backward and ask at which stage the expected value first diverged from the observed value. If the ADC code is already clipped, the algorithm is not at fault. If the ADC code is clean but the digital filter removed the band, the algorithm is being asked an impossible question. If the digital signal is clean but the timestamp is wrong, the model may be correct and the integration wrong.

Three quantities travel through every stage and must be tracked separately:

| Quantity | Question | Typical failure |
|---|---|---|
| Amplitude | Does the signal fit, with headroom, at every stage? | Saturation on offset or motion |
| Time | Is the bandwidth adequate, and what is the group delay? | Aliasing; a "real-time" alert that arrives late |
| Provenance | Which device, firmware, gain, filter, and calibration produced this number? | Apparent model drift that is a configuration change |

The amplitude view is a budget in microvolts. The time view is a budget in milliseconds. The provenance view is a data dictionary that is complete enough to reconstruct both budgets later.

A compact way to remember the chain: **transduce, protect, amplify, filter, sample, compute, transmit, display**. The order matters. Protection before amplification prevents damage; amplification before filtering improves the noise figure only if the amplifier's own noise is low enough; filtering before sampling is mandatory; timestamping before transmission preserves causality.

### 3.1 The information-loss hierarchy

Not all losses are equal.

| Loss | Recoverable later? | Why |
|---|---|---|
| Additive white noise | Partly, by averaging if the signal is repeated | Averaging reduces uncorrelated noise by the square root of the number of averages, at the cost of time |
| Quantisation | No, below one least significant bit (LSB) | The code is the only record |
| Clipping | No | The original amplitude is not represented |
| Aliasing | No | Different frequencies become indistinguishable |
| Linear filtering | Yes, if the transfer function is known and the signal was not clipped | Deconvolution amplifies noise, so recovery is limited |
| Lossy compression | No, beyond the compression error | The discarded coefficients are gone |
| Downsampling without filtering | No | Aliasing again |
| Wrong timestamp | No, unless a reference clock was also recorded | Causality and latency cannot be reconstructed reliably |

This hierarchy explains a practical rule: ask for the rawest data the device can legally export, plus the device-computed values, plus the configuration. The raw data preserves the option to re-derive; the computed values preserve the vendor's intended interpretation; the configuration preserves the ability to tell which is which.

### 3.2 A note on units

Every stage has a unit and a reference point. A noise voltage can be referred to the amplifier input or to the ADC input; the two differ by the gain. A signal can be expressed in microvolts, in ADC codes, or in physical units after calibration. Most analysis errors in this area come from mixing reference points. Write the reference point next to every number: "0.85 µV RMS input-referred, 131 Hz noise bandwidth" is a specification; "0.85 µV" is not.

### 3.3 Two archetypes: electrical and optical

The same chain logic applies to very different physics, but the dominant noise and failure modes differ.

| Aspect | ECG (electrical) | PPG (optical) |
|---|---|---|
| Measurand | Potential difference at the skin | Change in light absorption |
| Source amplitude | 0.5–5 mV | 0.1–2% modulation of a large DC level |
| Dominant interference | 50/60 Hz common mode, motion | Ambient light, motion, perfusion changes |
| Offset | ±300 mV electrode potential | Large DC optical background |
| Front-end challenge | High input impedance, CMRR | LED current control, ambient rejection, dark current |
| Sampling | 250–1,000 Hz | 25–100 Hz |
| Key confounder | Electrode contact and placement | Skin tone, venous pulsation, pressure |
| Typical data failure | Saturation, baseline wander | Dropouts, motion artefact, calibration bias |

An optical chain often has a larger DC term than the signal of interest, so the front end must either subtract the ambient level or use a modulated source and a synchronous detector. A team that applies an ECG-style high-pass filter to a PPG signal can remove the respiratory modulation that carries clinical information. The physics changes the filter, not only the sensor.

### 3.4 The two-question test

Before trusting any device-derived number, ask two questions.

**Question 1: what was the last stage at which the quantity was still in its original physical form?** If the answer is "the display", the number has passed through every stage and accumulated every error. If the answer is "a raw waveform file", the number is closer to the physics and further from the vendor's interpretation. Neither is automatically better; the choice depends on the question. For signal-quality research, raw is better. For a clinical workflow study, the device-computed value is the thing the clinician actually saw.

**Question 2: what would this number look like if the patient were unchanged but the device changed?** If the answer is "the same", the number is robust to configuration. If the answer is "different", the configuration is part of the measurement and must be recorded. Most device-derived features fail this test, which is why provenance is not optional.

The two questions take about a minute and prevent the most common category error in device data science: treating a processed value as a physical fact.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Signal chain | Ordered set of stages from measurand to clinical output | Used to mean only the analogue circuit | None |
| Measurand | The physical quantity intended to be measured | Confused with the sensor output | Physical unit |
| Transducer | Device that converts one form of energy into another | Called a sensor when it also contains conditioning | None |
| Sensor | Transducer plus the packaging and, often, conditioning | Treated as an ideal source | None |
| Sensitivity | Change in output per unit change in input | Confused with resolution | V per unit, or codes per unit |
| Offset | Output present when the input is zero | Confused with drift | V or µV |
| Drift | Change in offset or sensitivity over time or temperature | Called noise | µV/°C, µV/year, %/year |
| Gain | Ratio of output to input | Quoted without a reference point | Dimensionless, or V/V |
| Differential signal | Difference between two inputs | Assumed to reject all interference | V |
| Common-mode signal | Average of two inputs relative to a reference | Assumed to be zero in the body | V |
| CMRR | Ratio of differential gain to common-mode gain | Quoted at DC only | dB |
| Input impedance | Impedance presented by the amplifier input | Confused with source impedance | Ω, often with parallel capacitance |
| Source impedance | Impedance of the sensor and its connection | Assumed resistive and constant | Ω |
| Thermal noise | Voltage noise from random charge motion in a conductor | Called "white noise" without a bandwidth | nV/√Hz |
| Current noise | Equivalent input current noise of an amplifier | Ignored with high source impedance | pA/√Hz |
| Flicker noise | Noise with power spectral density rising at low frequency | Ignored in low-bandwidth designs | nV/√Hz at a stated frequency |
| SNR | Signal power divided by noise power | Quoted without a bandwidth | dB |
| SINAD | Signal power divided by all noise and distortion | Treated as equal to SNR | dB |
| ENOB | Effective number of bits implied by SINAD | Confused with the ADC's marketed resolution | bits |
| LSB | Smallest representable step of a converter | Confused with accuracy | V |
| Full-scale range | Input range that maps to the converter's code range | Assumed to equal the supply voltage | V |
| Quantisation noise | Error from mapping a continuous value to discrete codes | Assumed to be the only noise | V RMS |
| Aliasing | Folding of frequencies above half the sampling rate into lower frequencies | Confused with filtering | None |
| Nyquist frequency | Half the sampling rate | Called the sampling rate | Hz |
| Anti-alias filter | Filter before sampling that attenuates above Nyquist | Placed after the ADC in firmware | Hz, order, attenuation |
| Bandwidth | Range of frequencies passed within a stated tolerance | Quoted as a single number without tolerance | Hz |
| Group delay | Delay of the envelope of a narrowband signal through a filter | Confused with phase delay | s |
| Latency | Time from a physical event to the output | Measured only in software | ms |
| Jitter | Variation in sampling instant | Ignored for low-frequency signals | s RMS, or ps |
| Saturation | Condition where the output no longer follows the input | Called "noise" | V, or codes |
| Baseline wander | Low-frequency movement of the signal baseline | Removed without recording the filter | V, Hz |
| Calibration | Establishing the relationship between output and a reference | Treated as a one-time factory step | Procedure and date |
| Accuracy | Closeness of the measured value to the true value | Confused with precision | Bias plus uncertainty |
| Precision | Closeness of repeated measurements to each other | Confused with accuracy | Standard deviation |
| Typical value | Value expected for a majority of units under stated conditions | Used as a design guarantee | Stated conditions |
| Maximum value | Value guaranteed not to be exceeded under stated conditions | Ignored in the noise budget | Stated conditions |
| Applied part | Part of the device that comes into physical contact with the patient | Confused with any patient-connected cable | Classification B, BF, CF |
| Patient leakage current | Current flowing through the patient to earth | Confused with touch current | µA |
| MOPP | Means of patient protection | Treated as a single component | Insulation, impedance, distance |
| Driven right leg | Circuit that actively drives the patient reference to reduce common-mode voltage | Assumed to eliminate motion artefact | V, Hz |
| Settling time | Time for the output to reach a stated accuracy after a step | Ignored after a filter change | s |
| Headroom | Margin between the expected signal and the range limit | Assumed to be large | V or % of range |

## 5. The mechanics, step by step

The procedure below turns a clinical requirement into a verified chain. It is written for an engineer or data scientist who must specify, review, or diagnose a device, not for a circuit designer who is laying out a board. Each step states its input, its output, and why it exists. Branch points are marked, and the edge cases listed in the assignment are mapped to the steps where they occur.

### Step 1. State the measurand and the clinical decision

**Input:** a clinical question, a use environment, and a decision that the measurement must support.

**Output:** a requirement table with range, resolution, accuracy, bandwidth, latency, and operating conditions.

**Rationale:** every later choice is a trade against these numbers. A heart-rate monitor and a diagnostic electrocardiograph use the same electrode but different bandwidths, different offset tolerances, and different evidence. Without a written requirement, the chain will be optimised for the wrong quantity.

Write the requirement as a range with a tolerance, not a single number. "Heart rate 30–300 beats per minute (bpm), ±5 bpm or ±5%, whichever is greater, over 30 s" is testable. "Accurate heart rate" is not.

### Step 2. Characterise the source

**Input:** the sensor datasheet, the electrode or probe interface, and the clinical environment.

**Output:** source amplitude, source impedance (magnitude and phase), source offset, source bandwidth, and the common-mode voltage relative to earth.

**Rationale:** the source sets the noise floor that no amplifier can improve. A 10 kΩ electrode at 37 °C produces about 13 nV/√Hz of thermal noise. An amplifier with 5 nV/√Hz adds little; one with 50 nV/√Hz dominates. The source impedance also converts the amplifier's current noise into voltage noise, and mismatch between the two source impedances converts common-mode interference into differential signal.

For a biopotential electrode, the source is not a resistor. It is an electrochemical half-cell with a DC offset, a frequency-dependent impedance, and a noise contribution that depends on the electrode area and gel. A Ag/AgCl electrode on prepared skin may show 5–20 kΩ at 10 Hz; a dry electrode may show 100 kΩ to 1 MΩ, with a much larger imaginary component. The offset can be ±300 mV or more. IEC 60601-2-25 requires an electrocardiograph to tolerate ±300 mV of differential and common-mode DC offset while recording a ±5 mV signal, which is a useful benchmark even outside diagnostic ECG.

### Step 3. Choose the transduction and estimate its noise

**Input:** the source characterisation and the requirement table.

**Output:** a sensor choice, its sensitivity with tolerance, its noise-equivalent input, and its own bandwidth.

**Rationale:** the sensor's noise-equivalent input is the smallest change it can resolve, expressed at its input. For a photoplethysmographic (PPG) sensor, the relevant noise is optical shot noise and detector noise, which depend on the light level, the LED current, and the integration time. For a pressure sensor, it is the diaphragm and bridge noise. For an electrode, the transduction is essentially perfect at the electrode-skin interface; the noise and offset come from the interface, not from the metal.

At this step, decide whether the sensor includes conditioning. A digital pressure sensor with an internal ADC may have a specified total error band, which is easier to use than a raw bridge but may hide the internal filter and update rate. A raw bridge gives more control but requires you to design the conditioning.

### Step 4. Protect the input

**Input:** the expected and fault conditions: defibrillation, electrosurgery, electrostatic discharge (ESD), accidental connection to mains, and patient movement.

**Output:** a protection topology and a verification plan.

**Rationale:** protection components add leakage, capacitance, and noise. A series resistor that limits defibrillation current also forms a low-pass filter with the input capacitance and contributes thermal noise. A transient-voltage suppressor has leakage current that counts toward the patient leakage limit. The protection must be designed before the gain, because it changes the source impedance and the noise.

IEC 60601-1 sets patient leakage limits that depend on the applied-part classification: for a type CF applied part, 10 µA in normal condition and 50 µA in single-fault condition at 50/60 Hz; for type B and BF, 100 µA and 500 µA. These limits constrain the isolation and protection design and, indirectly, the filter components that can be connected to the patient.

**Branch point:** if the device is battery-powered and isolated, the leakage path may be small enough that protection can be simple. If the device connects to mains-powered equipment or to a hospital network with a conductive path, the isolation barrier becomes the dominant design constraint. The decision table later in this section covers the common topologies.

### Step 5. Allocate gain and check headroom

**Input:** the source amplitude, the offset, the protection network, and the ADC full-scale range.

**Output:** a gain, an input range, and a headroom calculation.

**Rationale:** the gain must make the smallest signal of interest larger than the ADC noise and quantisation, while keeping the largest expected input, including offset, inside the amplifier and ADC range. These two requirements conflict. The resolution is to remove the offset before applying gain, or to choose a gain low enough that the offset fits and rely on a higher-resolution ADC.

The headroom equation is:

\[
V_{\text{out,max}} = G \times (V_{\text{signal,max}} + V_{\text{offset,max}} + V_{\text{cm,residual}}) \le V_{\text{rail}} - V_{\text{dropout}}
\]

For a 300 mV offset and a 2.5 V rail, the maximum gain without offset removal is about 8, leaving no room for signal or dropout. A gain of 6 gives 1.8 V, leaving 0.7 V. A gain of 100 requires offset removal to below about 20 mV before gain.

**Decision table: offset management**

| Condition | Option | Consequence |
|---|---|---|
| Offset small relative to range, high-resolution ADC available | DC coupling, low gain | Simplest; quantisation may limit resolution |
| Offset large, diagnostic bandwidth required | DC servo or offset digital-to-analogue converter (DAC) | Preserves low-frequency content; adds components and noise |
| Offset large, monitoring bandwidth acceptable | AC coupling at 0.5 Hz or higher | Removes offset; loses slow components; settling after a step |
| Offset large, electrode impedance high | Active electrode or buffer at the electrode | Reduces cable and motion effects; adds power and size |
| Offset varies slowly and is predictable | Periodic offset calibration with input shorted | Requires a switch; switch leakage and charge injection |
| Offset exceeds range intermittently | Range switching or saturation detection | Must be flagged; saturated samples are not usable |

### Step 6. Filter for bandwidth and anti-aliasing

**Input:** the required signal bandwidth, the sampling rate, and the interference environment.

**Output:** a filter topology, corner frequencies, order, and the resulting group delay.

**Rationale:** the filter defines what the algorithm can see. A high-pass at 0.5 Hz removes baseline wander but also removes the ST segment, which is diagnostic information. A low-pass at 40 Hz removes muscle noise but also removes high-frequency QRS notches that some algorithms use. An anti-alias filter must attenuate frequencies above the Nyquist frequency before sampling; a digital filter after the ADC cannot remove an alias that has already folded into the band.

The order of the filter is a trade between roll-off and group delay. A first-order low-pass at 150 Hz has a group delay of about 1/(2π×150) = 1.06 ms at low frequency; a fourth-order Butterworth at the same corner has a group delay of roughly 2.6 ms, and a steeper elliptic filter has more. For a real-time alarm, the group delay is part of the latency budget.

**Branch point:** if the signal of interest is narrowband and the interference is at a known frequency, a notch filter can help, but a notch at 50/60 Hz also removes signal content at that frequency and can ring after a transient. If the interference is broadband, increase the order of the anti-alias filter or raise the sampling rate.

### Step 7. Choose the sampling rate and converter resolution

**Input:** the filter output bandwidth, the required latency, and the resolution requirement.

**Output:** a sampling rate, a converter architecture, and an expected ENOB.

**Rationale:** the sampling rate must exceed twice the highest frequency that the anti-alias filter passes with meaningful amplitude. In practice, choose the rate so that the filter attenuation at Nyquist is adequate for the largest out-of-band interferer. For a 150 Hz diagnostic bandwidth and a 500 Hz sample rate, the Nyquist frequency is 250 Hz; the anti-alias filter must attenuate content between 150 and 250 Hz enough that it does not alias into the diagnostic band.

Resolution is not the same as accuracy. A 24-bit converter with 0.85 µV RMS input-referred noise and a 2.4 V reference at gain 6 has an LSB of about 47.7 nV at the input, and an ENOB of about 8.4 bits. The extra bits below the noise floor are useful for averaging and for preserving small signals in a quiet moment, but they do not represent real resolution on a single sample.

**Branch point:** if the signal is slow and the environment is noisy, a delta-sigma converter with a high oversampling ratio can provide both resolution and anti-aliasing, at the cost of latency. If the signal is fast and multi-channel, a successive-approximation register (SAR) converter or a parallel array may be necessary.

### Step 8. Build the noise budget

**Input:** source impedance, amplifier voltage and current noise, resistor values, bandwidth, and converter noise.

**Output:** total input-referred noise in µV RMS, SNR, and ENOB.

**Rationale:** noise contributions add in power, not amplitude, if they are uncorrelated. The input-referred total is:

\[
e_{\text{in,total}} = \sqrt{e_{\text{source}}^2 + e_{\text{amp}}^2 + (i_{\text{amp}} Z_s)^2 + e_{\text{resistor}}^2 + e_{\text{ADC}}^2}
\]

Each term must be integrated over the same noise bandwidth. For a first-order low-pass with corner \(f_c\), the noise bandwidth is \((\pi/2) f_c\), not \(f_c\). For a higher-order filter, the ratio approaches the filter's noise-bandwidth factor, which is between 1.0 and 1.6 for common responses.

The budget must distinguish typical from maximum. A design that uses the amplifier's typical 20 nV/√Hz and ignores the maximum 35 nV/√Hz will pass on the bench and fail on the worst unit. The same applies to resistor tolerance, reference accuracy, and electrode offset.

### Step 9. Check the timing and latency budget

**Input:** filter group delay, converter conversion time, processing time, transmission time, and display refresh.

**Output:** end-to-end latency with a tolerance, and the fraction attributable to each stage.

**Rationale:** "real-time" is not a specification. A latency budget makes it one. For an arrhythmia alarm, the clinically relevant delay may be seconds; for a closed-loop insulin pump, it may be tens of milliseconds. The budget must include the time between samples for a periodic measurement, the algorithm's window length, and the network's retry behaviour.

Group delay is not the only delay. A finite-impulse-response (FIR) filter with \(N\) taps at sample rate \(f_s\) has a delay of \((N-1)/(2 f_s)\). A median filter of width \(W\) samples has a delay of about \(W/2\) samples. An algorithm that requires a 10 s window cannot produce an output before 10 s, regardless of the processor speed.

### Step 10. Digitise, timestamp, and package

**Input:** the converter output, the device clock, and the configuration.

**Output:** a data stream with sample values, sample timestamps, and metadata.

**Rationale:** a sample without a timestamp is an ordinal, not a time series. Clock drift, network delay, and buffering can reorder or delay samples. Record both the device time and the receiving system time, and record the clock source. If the device uses a crystal with ±20 ppm accuracy, the time error accumulates at 1.7 ms per day; over a 30-day patch, that is 52 ms, which may matter for cross-signal alignment.

Package the metadata with the signal: device model, serial or pseudonymous identifier, firmware version, sampling rate, gain, filter settings, calibration date, sensor lot, battery state, and signal-quality flags. Without these, a later change in the data is indistinguishable from a change in the patient.

### Step 11. Calibrate and verify

**Input:** the chain design and a reference standard.

**Output:** calibration coefficients, a verification record, and a statement of uncertainty.

**Rationale:** calibration converts the digital code to a physical unit and establishes the uncertainty. Verification demonstrates that the chain meets its specification across the stated range, including the edges. A chain that is verified only at mid-scale may fail at the extremes, where offset, gain error, and nonlinearity are largest.

The verification plan should include: zero input, full-scale input, offset at the specified maximum, a step response for settling, a frequency sweep for bandwidth, a common-mode sweep for CMRR, and a power-supply sweep for rejection. For a device with a firmware filter, repeat the verification after any filter change.

### Step 12. Validate in the intended environment

**Input:** the verified chain and a clinical or simulated use case.

**Output:** evidence that the chain performs in the presence of real interference, motion, and user behaviour.

**Rationale:** bench verification uses clean signals. The clinical environment adds electrosurgery, mobile phones, patient movement, and poor electrode contact. A chain that meets its noise specification on the bench may fail in the electromagnetic environment of an operating room. IEC 60601-1-2 defines the immunity tests; the device must pass them, and the data team should know which test levels were applied.

The validation ladder in the survey file runs from bench to phantom to animal or cadaver to human factors to clinical pilot to pivotal study to post-market surveillance. Each step answers a different question, and none substitutes for the next.

### 5.1 Edge cases and where they enter

| Edge case | Stage | Mechanism | Detection | Consequence if missed |
|---|---|---|---|---|
| Saturation | 5, 7 | Output reaches rail; code clamps | Code histogram at extremes; flat-topped waveform | Lost QRS or pressure peak; false algorithm confidence |
| Common-mode rejection failure | 2, 4, 6 | Source impedance mismatch converts common-mode to differential | Inject common-mode signal; measure output | 50/60 Hz interference mistaken for signal |
| Electrode offset | 2, 5 | Half-cell potential plus skin potential | Measure open-circuit voltage; check input range | Saturation; low-gain compromise |
| Clock jitter | 7, 10 | Sampling instant varies; aperture error | Measure SNR vs input frequency; phase noise | Degraded high-frequency SNR; timing errors |
| Brown-out | 10, 11 | Supply drops below minimum; converter or processor resets | Supply monitor log; code reset flags | Corrupted samples; lost calibration |
| Watchdog reset loop | 10 | Firmware fails to kick watchdog; repeated resets | Reset counter; boot log | Gaps in data; device unavailable |
| Thermal throttling | 10 | Processor reduces clock to stay within temperature limit | Die temperature log; latency distribution | Dropped frames; missed deadlines |
| Firmware update failure | 10, 11 | Incomplete write; signature check fails; rollback fails | Version check; boot count | Device bricked or running unvalidated code |
| Calibration drift | 11 | Sensor aging, temperature, sterilization | Periodic reference measurement; trend of zero | Bias in all downstream values |

The assignment requires these edge cases to be addressed explicitly. A design review that does not name them is incomplete, even if the nominal chain is well engineered.

### 5.2 Reference points and the gain cascade

A common source of confusion is the reference point of a noise or signal specification. The same physical noise can be 0.85 µV at the electrode, 5.1 µV at the amplifier output with a gain of 6, and 47.7 nV per LSB at the converter input. When combining stages, convert every term to one reference point, usually the input, before adding.

The conversion rules are:

- Voltage noise at the output divided by the gain gives input-referred voltage noise.
- Current noise multiplied by the source impedance gives input-referred voltage noise.
- ADC voltage noise divided by the gain gives input-referred voltage noise.
- Quantisation step at the ADC input divided by the gain gives the input-referred LSB.

If a stage has a gain of \(G_1\) and the next stage has a gain of \(G_2\), the input-referred noise of the second stage is its output noise divided by \(G_1 G_2\). This is why the first stage's noise dominates: later stages are divided by the accumulated gain.

### 5.3 The decision table for front-end topology

| Requirement | Topology | Strength | Limitation |
|---|---|---|---|
| Lowest noise with low source impedance | Transformer or low-noise bipolar input | Very low voltage noise | Current noise rises; not for high impedance |
| High source impedance, low frequency | Complementary metal-oxide-semiconductor (CMOS) or junction-gate field-effect transistor (JFET) input | Low current noise | Higher voltage noise; input capacitance |
| Large offset, wide bandwidth | Instrumentation amplifier plus DC servo | Preserves DC to low frequency | Servo adds noise and settling time |
| Large offset, monitoring only | AC coupling | Simple, removes offset | Loses slow signal; step recovery |
| Multi-channel biopotential | Integrated AFE with per-channel gain | Matched channels, small size | Fixed filter and gain options |
| Very low power, wearable | Chopper-stabilised amplifier | Low offset and flicker noise | Charge injection; clock artefacts |
| High common-mode, isolated | Isolation amplifier or digital isolator | Breaks leakage path | Cost, power, propagation delay |
| Fast transient protection | Series resistor plus clamp | Survives defibrillation | Adds noise and capacitance |

The topology choice is not a matter of fashion. It follows from the source impedance, the offset, the bandwidth, and the leakage limit. A wearable with a dry electrode and a 1 MΩ source cannot use the same front end as a diagnostic ECG with a 10 kΩ wet electrode, even though both measure the same physiological signal.

### 5.4 What "verified" means at the chain level

Verification is not a single test. For a chain, it is a matrix of input conditions and measured outputs, with acceptance criteria derived from the requirement table. A minimal matrix has:

- Input amplitude: zero, 10% of full scale, full scale, and 120% of full scale to test clipping.
- Input offset: zero, positive maximum, negative maximum, and a step between them.
- Frequency: the low corner, a mid-band frequency, the high corner, and a frequency above Nyquist to test aliasing.
- Common-mode: zero, positive maximum, and a 50/60 Hz sweep.
- Supply: minimum, nominal, and maximum, plus a brown-out transient.
- Temperature: the operating range extremes, soaked to equilibrium.
- Time: a soak long enough to reveal drift, and a power cycle to reveal settling.

Each cell produces a pass or fail and a margin. The margins are the useful part: a design that passes at 0.1 dB of margin is not robust, even though it passes. Report the margin, not only the verdict.

### 5.5 Noise bandwidth and integration arithmetic

A noise spectral density in nV/√Hz becomes an RMS voltage only after multiplication by the square root of the noise bandwidth. Using the signal bandwidth instead of the noise bandwidth understates the noise. For a first-order low-pass with corner \(f_c\), the noise bandwidth is \((\pi/2) f_c = 1.571 f_c\). For higher-order Butterworth responses, the factor approaches values between 1.0 and 1.6:

| Filter order and type | Noise-bandwidth factor | 150 Hz noise bandwidth |
|---|---:|---:|
| First-order low-pass | 1.571 | 235.6 Hz |
| Second-order Butterworth | 1.111 | 166.7 Hz |
| Third-order Butterworth | 1.047 | 157.1 Hz |
| Fourth-order Butterworth | 1.026 | 153.9 Hz |
| Eighth-order Butterworth | 1.007 | 151.0 Hz |
| Brick-wall ideal | 1.000 | 150.0 Hz |

The table has a practical consequence. A first-order filter at 150 Hz integrates 57% more noise than a brick-wall filter at the same corner. If the noise budget was calculated with 150 Hz but the circuit uses a first-order filter, the predicted noise is low by 1.96 dB. For a 1 µV budget, that is a 24% underestimate.

When several filters are cascaded, the combined noise bandwidth is not the product of the factors. Compute the combined transfer function, or simulate it, or measure it. A common approximation for a cascade of identical first-order sections is not accurate enough for a noise budget at the 10% level.

The same arithmetic applies to current noise. A current noise density of 1 pA/√Hz flowing through 10 kΩ produces 10 nV/√Hz. Through 1 MΩ, it produces 1 µV/√Hz. This is why a high-impedance electrode requires a low-current-noise amplifier, and why adding a 1 MΩ resistor to set a 0.05 Hz corner can destroy the noise budget even though the resistor's own thermal noise is only 128 nV/√Hz.

### 5.6 Latency arithmetic

Latency is the sum of delays, not the delay of the slowest stage. A representative chain for an ECG monitor is:

| Stage | Delay | Basis |
|---|---:|---|
| Anti-alias filter, fourth-order at 150 Hz | 2.6 ms | Group delay at low frequency |
| Delta-sigma converter, 500 samples per second | 2.0 ms | Conversion and decimation |
| FIR filter, 101 taps at 500 samples per second | 100 ms | \((N-1)/(2 f_s)\) |
| Median filter, 5 samples | 4 ms | Half the window |
| Algorithm window | 2,000 ms | 10 beats at 60 bpm |
| Network transmission and retry | 150 ms | Measured 95th percentile |
| Display refresh | 33 ms | 30 frames per second |
| **Total** | **2,291 ms** | Sum of the rows |

The algorithm window dominates. No amount of processor acceleration reduces a 2 s window to 200 ms. If the clinical requirement is a 1 s alarm delay, the algorithm must use a shorter window or a different feature, which changes its sensitivity and specificity. The latency budget makes that trade visible before the algorithm is written.

For a closed-loop device, the relevant delay may be the phase lag at the control bandwidth, not the total delay. A 10 ms delay at a 1 Hz control bandwidth contributes \(2\pi \times 1 \times 0.010 = 0.063\) radians, or 3.6 degrees, which is negligible. The same 10 ms at a 20 Hz bandwidth contributes 72 degrees, which can destabilise a loop. Always state latency in the context of the bandwidth at which it matters.

### 5.7 Calibration and uncertainty arithmetic

Calibration establishes a linear or polynomial mapping from code to physical unit. A two-point calibration determines gain and offset:

\[
\text{value} = \frac{\text{code} - c_0}{c_1 - c_0} \times (x_1 - x_0) + x_0
\]

where \(c_0\) and \(c_1\) are the codes at reference values \(x_0\) and \(x_1\). The uncertainty of the calibrated value includes the uncertainty of the references, the repeatability of the codes, the resolution of the converter, and the stability of the chain between calibrations.

A useful approximation for the combined standard uncertainty is:

\[
u_c = \sqrt{u_{\text{ref}}^2 + u_{\text{repeat}}^2 + u_{\text{res}}^2 + u_{\text{drift}}^2}
\]

For a pressure channel calibrated against a 0.05% reference, with 0.02% repeatability, 0.01% resolution, and 0.1% drift over the service interval, the combined standard uncertainty is 0.115% of reading. The expanded uncertainty at 95% coverage, using a coverage factor of 2, is 0.23%. If the clinical requirement is 1%, the channel passes with a factor of four margin; if it is 0.2%, it fails.

Calibration drift is not a single number. It has a temperature coefficient, a time coefficient, and a hysteresis component. The time coefficient is often specified per year, but the device may be calibrated every two years. The budget must use the actual service interval, not the datasheet's first year.

### 5.8 Supply rejection and grounding

A chain shares a power supply, a ground reference, and an enclosure with other circuits. Supply noise couples into the signal path through the amplifier's finite power-supply rejection ratio (PSRR), through the reference, and through the ground impedance. A switching regulator at 2 MHz with 20 mV of ripple, feeding an amplifier with 60 dB PSRR at that frequency, produces 20 µV of input-referred noise. If the signal is 1 mV, that is 2% of the signal, which may be acceptable; if the signal is 10 µV, it is 200%.

Grounding is a design choice, not a default. A single-point ground for the analogue section, a separate return for the digital section, and a star connection at the supply entry reduce the shared impedance. A ground loop between the device and the patient, through the mains earth and the electrode cable, can inject current into the patient and add interference. The isolation barrier breaks the loop; the driven-right-leg circuit reduces the common-mode voltage that remains.

The driven-right-leg circuit senses the common-mode voltage on the patient and drives it back through a resistor. Its effectiveness depends on the loop gain and the stability of the feedback. A poorly compensated driven-right-leg loop can oscillate at a few kilohertz, adding a signal that looks like muscle noise. The loop must be tested with the actual electrode impedances, not with a resistive simulator.

### 5.9 Reading a chain backwards from data

A data scientist often receives data without a chain diagram. The following signatures help infer what happened. They are heuristics, not proof; confirm each with the device configuration.

| Signature in the data | Likely stage | Confirm by |
|---|---|---|
| Flat-topped peaks at a consistent code value | Saturation at amplifier or ADC | Code histogram; count of samples at full scale |
| Quantisation steps visible in a supposedly continuous signal | Low gain or missing bits | Autocorrelation; check gain and bit depth |
| A pure tone at exactly 50 or 60 Hz | Common-mode interference or ground loop | Check electrode impedance; test with a battery-powered source |
| A tone that changes frequency with the mains | Aliased interference | Compare with sample rate; check anti-alias filter |
| Baseline wander that correlates with respiration | High-pass corner too low or motion | Check filter setting; inspect raw versus filtered |
| Periodic gaps at a fixed interval | Buffer overrun or radio retry | Firmware log; packet sequence numbers |
| A sudden change in noise floor at a known date | Firmware or gain change | Version metadata; service records |
| Identical values repeated for several samples | Communication stall or sample-and-hold | Timestamp deltas; sequence numbers |
| A slow drift that reverses after calibration | Sensor drift or reference drift | Calibration log; reference measurement |
| A signal that looks plausible but is uncorrelated with the patient | Wrong channel mapping or units | Cross-check with an independent sensor |

The most dangerous signature is the last one. A plausible but wrong signal passes every quality check that looks only at morphology. Cross-checking against an independent measurement, even a coarse one, is the only reliable defence.

### 5.10 The chain in a regulatory submission

A regulatory submission describes the chain in a specific format. The design description states the intended use, the sensors, the signal conditioning, the conversion, the processing, and the output. The verification section provides evidence that each stage meets its specification. The risk file links each stage to its failure modes and the controls that reduce them. The software documentation identifies the requirements, the architecture, and the tests that trace to each requirement.

The data scientist's contribution to this package is usually the algorithm and its validation, but the algorithm's performance claim depends on the chain description. If the submission claims a heart-rate accuracy of ±5 bpm, the evidence must show that the input chain preserves the information needed to compute heart rate, that the algorithm was tested on that chain, and that the test population covers the intended use. A model validated on a public dataset with a different sampling rate and filter does not support the claim, even if the model is excellent.

A useful test of a chain description is whether another engineer could rebuild the measurement from it. If the description omits the gain, the filter corner, the sampling rate, or the calibration procedure, it is not a specification; it is a block diagram.

## 6. Worked example

### 6.1 The requirement

A wearable cardiac monitor must record a single-lead electrocardiogram for 24 hours. The clinical requirement is:

- Signal: QRS amplitude 0.5–2.0 mV peak-to-peak at the electrode.
- Diagnostic bandwidth: 0.5–150 Hz (a monitoring bandwidth, not the 0.05 Hz diagnostic standard).
- Electrode offset: up to ±300 mV.
- Interference: up to 1 V RMS common-mode at 50 Hz from the environment.
- Converter: 24-bit delta-sigma, 2.4 V reference, differential input, 500 samples per second.
- Supply: 3.0 V single cell, with a 2.5 V analogue rail.

The design uses an integrated biopotential front end with a gain of 6, an input-referred noise of 5.6 µV peak-to-peak at 500 samples per second and a 131 Hz bandwidth (typical), a minimum CMRR of 105 dB at 50 Hz, and an input bias current noise of 1 pA/√Hz. The electrode source impedance is 10 kΩ per electrode, matched to within 0.3%.

### 6.2 The headroom calculation

The input range of the front end at a gain of 6 and a 2.4 V reference is approximately:

\[
V_{\text{in,range}} = \pm \frac{V_{\text{REF}}}{G} = \pm \frac{2.4}{6} = \pm 400\ \text{mV}
\]

The worst-case input is the offset plus the signal plus a margin for motion:

\[
V_{\text{in,max}} = 300\ \text{mV} + 2\ \text{mV} + 20\ \text{mV} = 322\ \text{mV}
\]

The headroom is \(400 - 322 = 78\ \text{mV}\), or 19.5% of the range. This is adequate but not generous. A 500 mV offset, which can occur with a poorly prepared electrode or a different gel, would exceed the range and saturate the channel.

### 6.3 The noise budget

The noise bandwidth of a first-order 150 Hz low-pass is:

\[
B_n = \frac{\pi}{2} \times 150 = 235.6\ \text{Hz}
\]

The source thermal noise from two 10 kΩ electrodes in series is:

\[
e_{\text{source}} = \sqrt{4 k T R} = \sqrt{4 \times 1.38 \times 10^{-23} \times 310 \times 20\,000} = 18.5\ \text{nV}/\sqrt{\text{Hz}}
\]

Over the noise bandwidth:

\[
E_{\text{source}} = 18.5 \times \sqrt{235.6} = 284\ \text{nV RMS}
\]

The amplifier's voltage noise is not specified directly in the summary; the 5.6 µV peak-to-peak figure is the total input-referred noise in the 131 Hz bandwidth. Convert it to RMS using the crest factor for Gaussian noise, approximately 6.6 for a 10 s interval:

\[
E_{\text{total,131}} = \frac{5.6}{6.6} = 0.85\ \text{µV RMS}
\]

Scaling to the 235.6 Hz noise bandwidth of the 150 Hz filter:

\[
E_{\text{total,235}} = 0.85 \times \sqrt{\frac{235.6}{131}} = 1.14\ \text{µV RMS}
\]

This total already includes the amplifier's voltage noise, the current noise into the specified source, and the converter noise under the datasheet test conditions. The source thermal noise computed above (0.284 µV) is part of that total, not an additional term. Adding it again would double-count.

For a design budget, decompose the total into its contributors. The datasheet's noise table gives 0.9 µV RMS at 500 samples per second and a gain of 6, which is consistent with the 0.85 µV estimate. Allocate:

| Contributor | Input-referred RMS | Share of power |
|---|---:|---:|
| Electrode and cable thermal noise | 0.28 µV | 6% |
| Front-end voltage noise | 0.80 µV | 49% |
| Front-end current noise into 10 kΩ | 0.20 µV | 3% |
| Reference and supply noise | 0.35 µV | 10% |
| Converter quantisation and thermal noise | 0.50 µV | 19% |
| Residual common-mode at 50 Hz (10 mV ambient) | 0.30 µV | 7% |
| Unallocated margin | 0.25 µV | 5% |
| **Total (root sum square)** | **1.13 µV** | **100%** |

The root sum square of the rows is 1.13 µV, which matches the scaled datasheet total of 1.14 µV to within rounding. The budget closes. The largest single contributor is the front-end voltage noise, which is why the first amplifier matters.

The common-mode row deserves a note. It is computed for a 10 mV ambient common-mode voltage, which is a quiet clinical environment, not the 1 V immunity test level. At 1 V, the same chain produces 30 µV, which would dominate the budget. The budget must state the interference level it assumes; a noise budget without an interference assumption is incomplete.

### 6.4 SNR and effective resolution

The signal is 1.0 mV peak-to-peak, or 0.354 mV RMS for a sine of that amplitude. The SNR is:

\[
\text{SNR} = 20 \log_{10}\left(\frac{0.354\ \text{mV}}{1.13\ \text{µV}}\right) = 20 \log_{10}(313) = 49.9\ \text{dB}
\]

The effective number of bits is:

\[
\text{ENOB} = \frac{49.9 - 1.76}{6.02} = 8.0\ \text{bits}
\]

The converter has 24 bits. The measurement has 8 effective bits. The remaining 16 bits are below the noise floor and are useful only for averaging or for preserving small signals when the noise is lower than the worst case. This is the single most important number in the example: a 24-bit ADC does not give a 24-bit measurement.

The input-referred LSB is:

\[
\text{LSB}_{\text{in}} = \frac{2 \times 2.4}{6 \times 2^{24}} = 47.7\ \text{nV}
\]

The noise is 1.13 µV, or 23.7 LSB RMS. The converter is not the limiting factor; the analogue front end is. If the design doubled the gain to 12, the input range would fall to ±200 mV, which would saturate on the 300 mV offset. The gain cannot be increased without removing the offset.

### 6.5 Diagnostic 1: the offset is closer than it looks

A bench test with a signal generator set to 1 mV peak-to-peak and zero offset produces a clean waveform. A test with a 300 mV offset produces a waveform whose positive peaks are flattened. The firmware reports a lead-off flag intermittently.

The first diagnostic is to plot the ADC code histogram. With a 300 mV offset and a 2 mV signal, the mean code sits at 75% of the positive range. The signal peaks at 75.5% and the negative peaks at 74.5%. The channel is not clipped, but the headroom is only 78 mV, and a 20 mV motion artefact plus a 50 mV offset drift would consume it. The histogram shows a truncated distribution at the high end when the patient moves.

The design flaw is not the gain. It is the assumption that the offset is static and below 300 mV. A 400 mV offset, which is within the range of some electrodes, would clip. The fix is either a lower gain (4, giving ±600 mV) or an offset servo. A gain of 4 raises the input-referred noise by a factor of 1.5, to about 1.7 µV RMS, and lowers the SNR by 3.5 dB. The trade is explicit.

### 6.6 Diagnostic 2: the CMRR is not the CMRR you think

The datasheet says 105 dB minimum CMRR at 50 Hz. A 1 V common-mode signal should therefore produce at most:

\[
V_{\text{cm,residual}} = \frac{1\ \text{V}}{10^{105/20}} = 5.6\ \text{µV}
\]

The measured residual is 30 µV, five times larger. The amplifier is not at fault. The front end's common-mode input impedance is 10 MΩ, and the two electrode impedances differ by 300 Ω, which is 3% of the nominal 10 kΩ. The mismatch forms a divider that converts common-mode voltage into differential voltage:

\[
V_{\text{diff}} \approx V_{\text{cm}} \times \frac{\Delta Z}{Z_{\text{in,cm}}} = 1\ \text{V} \times \frac{300}{10\,000\,000} = 30\ \text{µV}
\]

The effective system CMRR is therefore 90.5 dB, not 105 dB. The amplifier's own 5.6 µV contribution is small compared with the 30 µV from the mismatch. Improving the amplifier to 120 dB would change the residual from 30 µV to about 30 µV. Reducing the electrode mismatch to 30 Ω (0.3%) would reduce it to about 3 µV, at which point the amplifier's CMRR begins to matter.

This is a general lesson: in a biopotential chain, the CMRR specification is an amplifier property, while the rejection achieved is a system property. The system property depends on the electrode impedances, the amplifier's input impedance, the cables, the patient, and the environment. A design review that quotes only the amplifier's CMRR has not characterised the system.

### 6.7 Vary one assumption: the offset rises to 500 mV

Change only the electrode offset, from 300 mV to 500 mV, and keep the gain at 6. The input range is ±400 mV. The offset alone exceeds the range by 100 mV, so the channel saturates for the entire recording. The ADC output sits at the positive rail, and the firmware's lead-off detection may or may not flag it.

The consequence is not a 20% reduction in signal quality. It is a total loss of signal for the affected lead. If the device has a single lead, the recording is unusable. If the algorithm receives the saturated code as a valid sample, it may compute a heart rate from the noise at the rail, or it may report asystole.

The fix options are:

1. Reduce the gain to 4, accepting a 3.5 dB SNR loss.
2. Add a DC servo that removes the offset before the gain stage, preserving the gain but adding a high-pass corner and settling time.
3. Add a coarse offset DAC that subtracts a known offset, with a residual range of ±100 mV.
4. Detect saturation and flag the sample, so that the algorithm does not treat it as physiological.

Option 4 is mandatory in any case. Saturation detection is not a substitute for headroom, but it is the difference between a known data gap and a silent false reading.

### 6.8 What the example demonstrates

The arithmetic closes: the noise budget sums to 1.13 µV RMS, the SNR is 49.9 dB, the ENOB is 8.0 bits, and the headroom is 78 mV. The two diagnostics reveal different flaws: one is a range assumption, the other is a system-level rejection assumption. The sensitivity analysis shows that a single changed assumption, the offset, moves the design from marginal to unusable.

The general method is to write the budget, find the dominant term, test the assumption that the term depends on, and state the consequence of that assumption failing. A design review that reports only the nominal SNR has not done the work.

### 6.9 Closing the time and energy budget

The chain must also fit the time and energy constraints. The 24-hour recording requires the device to store 500 samples per second per channel, at 3 bytes per sample, which is 1,500 bytes per second, or 129.6 megabytes per day before compression. A 128 megabyte flash device is therefore insufficient for the raw stream with metadata; the design needs either compression, a larger memory, or a lower sample rate for storage with a separate event buffer.

The front end draws 335 µW per channel, or 0.67 mW for two channels. The microcontroller, flash, and Bluetooth radio add approximately 12 mW average. The total is 12.67 mW, or 12.67 mJ per second, which is 1,095 J per day. A 300 mAh lithium-ion cell at 3.7 V stores 3,996 J, so the nominal runtime is 3.65 days. A 20% degradation from cold temperature and ageing reduces that to 2.92 days, which is above the 24-hour requirement but not by the factor that a first-pass estimate suggested. The energy budget closes with a margin of 2.92:1, and the margin is stated rather than assumed.

The latency of the stored recording is less critical than the latency of the live alarm. If the device also streams to a phone, the 2.3 s latency computed earlier applies. The stored record can be timestamped at acquisition, so an offline analysis can align it with other signals even if the stream was delayed. This is why the timestamp must be attached at the converter, not at the network layer.

## 7. Code, query, or calculation

The script below reproduces every number in Section 6 using only the Python standard library. It is written to be read as well as run: each block corresponds to one step of the budget.

**Assumptions.** Skin temperature 310 K; two 10 kΩ electrodes in series; first-order 150 Hz noise bandwidth; the datasheet's typical total input-referred noise is not used directly, because the budget is built from components and then compared with the datasheet. The component values are illustrative and must be replaced with the values for the chosen front end.

```python
import math

k = 1.380649e-23          # Boltzmann constant, J/K
T = 310.0                 # skin temperature, K
R_source = 20_000.0       # two 10 kOhm electrodes in series, ohm
BW = 150.0                # signal bandwidth, Hz
NBW = math.pi / 2 * BW    # first-order noise bandwidth, Hz


def rss(values):
    return math.sqrt(sum(v * v for v in values))


def db20(x):
    return 20.0 * math.log10(x)


e_density = math.sqrt(4 * k * T * R_source)      # V/sqrt(Hz)
e_source = e_density * math.sqrt(NBW)            # V RMS

budget_uv = {
    "electrode thermal": e_source * 1e6,
    "front-end voltage": 0.80,
    "front-end current": 0.20,
    "reference and supply": 0.35,
    "converter": 0.50,
    "residual common-mode": 0.30,
    "margin": 0.25,
}
total_uv = rss(budget_uv.values())

vref = 2.4
gain = 6.0
bits = 24
lsb_in = (2 * vref) / (gain * 2 ** bits)
signal_pp_mv = 1.0
signal_rms_uv = signal_pp_mv * 1e3 / (2 * math.sqrt(2))
snr_db = db20(signal_rms_uv / total_uv)
enob = (snr_db - 1.76) / 6.02

cm = 1.0
cmrr_amp_db = 105.0
zin_cm = 10e6
delta_z = 300.0
v_amp_residual = cm / (10 ** (cmrr_amp_db / 20))
v_mismatch = cm * delta_z / zin_cm
cmrr_system_db = db20(cm / v_mismatch)

print("SOURCE NOISE")
print(f"  density            {e_density*1e9:8.2f} nV/sqrt(Hz)")
print(f"  noise bandwidth    {NBW:8.1f} Hz")
print(f"  RMS                {e_source*1e6:8.3f} uV")
print()
print("NOISE BUDGET (input-referred, uV RMS)")
for name, value in budget_uv.items():
    share = 100 * value**2 / total_uv**2
    print(f"  {name:24s} {value:6.2f}  {share:5.1f}%")
print(f"  {'TOTAL (RSS)':24s} {total_uv:6.2f}  100.0%")
print()
print("RESOLUTION")
print(f"  input-referred LSB {lsb_in*1e9:8.3f} nV")
print(f"  noise in LSB       {total_uv*1e-6/lsb_in:8.2f} LSB RMS")
print(f"  signal RMS         {signal_rms_uv:8.1f} uV")
print(f"  SNR                {snr_db:8.2f} dB")
print(f"  ENOB               {enob:8.2f} bits")
print()
print("COMMON-MODE REJECTION")
print(f"  amplifier residual {v_amp_residual*1e6:8.2f} uV")
print(f"  mismatch residual  {v_mismatch*1e6:8.2f} uV")
print(f"  system CMRR        {cmrr_system_db:8.2f} dB")
print()
print("HEADROOM")
v_range_mv = vref / gain * 1e3
v_worst_mv = 300 + 2 + 20
print(f"  input range        +/-{v_range_mv:6.1f} mV")
print(f"  worst-case input   {v_worst_mv:6.1f} mV")
print(f"  headroom           {v_range_mv-v_worst_mv:6.1f} mV")
print()
print("LATENCY")
stages_ms = {
    "anti-alias filter": 2.6,
    "converter": 2.0,
    "FIR 101 taps": 100.0,
    "median filter": 4.0,
    "algorithm window": 2000.0,
    "network": 150.0,
    "display": 33.0,
}
for name, value in stages_ms.items():
    print(f"  {name:20s} {value:7.1f} ms")
print(f"  {'TOTAL':20s} {sum(stages_ms.values()):7.1f} ms")
print()
print("ENERGY")
front_end_mw = 2 * 335e-3
system_mw = front_end_mw + 12.0
j_per_day = system_mw * 1e-3 * 86400
battery_j = 0.300 * 3.7 * 3600
print(f"  system power       {system_mw:8.2f} mW")
print(f"  energy per day     {j_per_day:8.1f} J")
print(f"  battery energy     {battery_j:8.1f} J")
print(f"  nominal runtime    {battery_j/j_per_day:8.2f} days")
print(f"  aged runtime       {0.8*battery_j/j_per_day:8.2f} days")
```

**Expected output.**

```text
SOURCE NOISE
  density               18.50 nV/sqrt(Hz)
  noise bandwidth       235.6 Hz
  RMS                   0.284 uV

NOISE BUDGET (input-referred, uV RMS)
  electrode thermal          0.28    6.3%
  front-end voltage          0.80   49.8%
  front-end current          0.20    3.1%
  reference and supply       0.35    9.5%
  converter                  0.50   19.4%
  residual common-mode       0.30    7.0%
  margin                     0.25    4.9%
  TOTAL (RSS)                1.13  100.0%

RESOLUTION
  input-referred LSB   47.684 nV
  noise in LSB          23.78 LSB RMS
  signal RMS            353.6 uV
  SNR                   49.88 dB
  ENOB                   7.99 bits

COMMON-MODE REJECTION
  amplifier residual     5.62 uV
  mismatch residual     30.00 uV
  system CMRR           90.46 dB

HEADROOM
  input range        +/- 400.0 mV
  worst-case input    322.0 mV
  headroom             78.0 mV

LATENCY
  anti-alias filter        2.6 ms
  converter                2.0 ms
  FIR 101 taps           100.0 ms
  median filter            4.0 ms
  algorithm window      2000.0 ms
  network                150.0 ms
  display                 33.0 ms
  TOTAL                 2291.6 ms

ENERGY
  system power          12.67 mW
  energy per day       1094.7 J
  battery energy       3996.0 J
  nominal runtime        3.65 days
  aged runtime           2.92 days
```

The output demonstrates four points:

1. The electrode thermal noise is only 6.3% of the total. Improving the electrode alone would not change the SNR meaningfully; the front-end voltage noise is the dominant term at 49.8%.
2. The converter contributes 19.4% of the noise power, which is more than the electrode. A 24-bit converter is not automatically quiet; its input-referred noise must be in the budget.
3. The effective resolution is 7.99 bits, not 24. Reporting 24 bits would overstate the measurement by a factor of about one million in amplitude.
4. The latency is 2.29 s, dominated by the algorithm window. A processor upgrade would not change it.

**A note on the crest factor.** The conversion from peak-to-peak noise to RMS uses a factor of 6.6, which is appropriate for Gaussian noise observed over a 10 s interval. A shorter interval has a lower crest factor and a smaller peak-to-peak value for the same RMS. When comparing a datasheet's peak-to-peak noise with an RMS budget, state the interval and the crest factor. A factor of 6 to 8 is common; a factor of 3.3 is sometimes used for a 1 s interval, which changes the RMS estimate by a factor of two.

### 7.1 Averaging and the limits of post-processing

Averaging repeated beats is the most common way to recover resolution below the single-sample noise floor. If the noise is uncorrelated between beats and the signal is identical, averaging \(N\) beats raises the SNR by \(10\log_{10}N\) dB and the ENOB by \(0.5\log_2 N\) bits.

```python
import math

def enob(snr_db):
    return (snr_db - 1.76) / 6.02

base_snr = 49.88
print("AVERAGING GAIN (uncorrelated noise assumed)")
print(" N   SNR dB   ENOB bits   noise reduction dB")
for n in (1, 4, 16, 64, 256):
    gain = 10 * math.log10(n)
    print(f"{n:4d} {base_snr + gain:8.2f} {enob(base_snr + gain):10.2f} {gain:18.2f}")
```

**Expected output.**

```text
AVERAGING GAIN (uncorrelated noise assumed)
 N   SNR dB   ENOB bits   noise reduction dB
   1    49.88       7.99               0.00
   4    55.90       8.99               6.02
  16    61.92       9.99              12.04
  64    67.94      10.99              18.06
 256    73.96      11.99              24.08
```

Sixteen averages gain two bits. That sounds like a way to turn an 8-bit measurement into a 10-bit one, but the assumption is strong. The signal must be identical across beats, which is false for a physiological waveform that changes with respiration and posture. The noise must be uncorrelated across beats, which is false for 50 Hz interference that is coherent with the recording. And the averaging must be aligned to the beat, which requires a detection step whose own errors propagate. In practice, averaging improves the estimate of a stable feature, not the resolution of a changing waveform.

The crest-factor sensitivity is equally important. The same 1.13 µV RMS noise has a peak-to-peak value of 3.73 µV at a crest factor of 3.3 and 9.04 µV at a crest factor of 8.0. A datasheet that quotes 5.6 µV peak-to-peak over 10 s and a designer who converts it with a factor of 3.3 will disagree by a factor of two. Always state the convention.

## 8. Data, standards, and regulatory context

The signal chain is governed by standards that constrain the hardware, the software, and the evidence. The table below separates what is binding in a given jurisdiction from what is a consensus standard that becomes binding only when referenced.

| Instrument | Scope | Status | What it constrains in the chain |
|---|---|---|---|
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General safety of medical electrical equipment | Consensus standard; binding when referenced by regulation or contract | Applied-part classification, patient leakage current, isolation, protection against excessive temperatures |
| IEC 60601-1-2:2014+AMD1:2020 | Electromagnetic disturbances | Collateral standard | Emissions and immunity test levels; cable and enclosure design; filter choices |
| IEC 60601-1-8:2006+AMD2:2020 | Alarm systems | Collateral standard | Alarm condition delay and alarm signal generation delay; disclosure requirements |
| IEC 60601-2-25:2011 | Particular requirements for electrocardiographs | Particular standard | Input dynamic range, DC offset tolerance, frequency response, accuracy of derived values |
| IEC 60601-2-27:2011 | Particular requirements for ECG monitoring equipment | Particular standard | Dynamic range, offset tolerance, heart-rate accuracy, alarm performance |
| ISO 14971:2019 | Application of risk management to medical devices | Consensus standard | Hazard analysis, risk controls, residual risk, production and post-production monitoring |
| IEC 62304:2006+AMD1:2015 | Medical device software life-cycle processes | Consensus standard | Software safety classification, requirements, architecture, verification, configuration management |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | Use specification, use-related risk, formative and summative evaluation |
| ISO 13485:2016 | Quality management systems | Consensus standard | Design controls, document control, purchasing, CAPA |
| FDA 21 CFR 820 | Quality system regulation (U.S.) | Binding regulation | Design controls, CAPA, complaints, production |
| EU MDR 2017/745 | Medical device regulation (EU) | Binding regulation | Clinical evaluation, technical documentation, post-market surveillance |
| FDA guidance on premarket cybersecurity | Device cybersecurity | Guidance; expectations enforced through review | Threat model, security testing, software bill of materials, update plan |

### 8.1 What IEC 60601-1 actually constrains

IEC 60601-1 classifies applied parts as type B, BF, or CF according to the degree of protection against electric shock. The classification determines the patient leakage current limits. For alternating current at 50/60 Hz, the limits in clause 8.7.3 are 100 µA normal condition and 500 µA single-fault condition for type B and BF applied parts, and 10 µA normal condition and 50 µA single-fault condition for type CF applied parts. Direct-current limits are lower: 10 µA and 50 µA for B/BF and CF respectively.

These numbers matter to the data scientist for an indirect reason. The leakage limit constrains the isolation, the protection components, and the capacitance that can be connected to the patient. A design that adds a large filter capacitor to reduce 50 Hz interference may fail the leakage test. A design that adds a series resistor to survive defibrillation may add thermal noise and form a low-pass filter with the cable capacitance. The filter you see in the data is partly a safety artefact.

The standard also constrains temperatures of applied parts, mechanical strength, and resistance to cleaning and sterilization. A sensor that meets its electrical specification after autoclaving may not meet it after 50 cycles. The chain includes the sterilisation history.

### 8.2 What IEC 60601-2-25 adds for ECG

IEC 60601-2-25:2011 clause 201.12.4.107.2 requires an electrocardiograph to record a ±5 mV input signal while tolerating differential and common-mode DC offsets of ±300 mV, with the recorded amplitude changing by no more than 5% across the recording width. The frequency response requirement in Table 201.107 extends from 0.05 Hz to at least 150 Hz for diagnostic ECG.

These requirements explain the gain and offset choices in Section 6. A front end with a gain of 6 and a 2.4 V reference has a ±400 mV input range, which covers the ±300 mV offset plus a ±5 mV signal with 95 mV to spare. A front end with a gain of 100 would not, unless it removed the offset first. The standard does not prescribe the gain; it prescribes the input condition, and the gain follows.

The standard also specifies tests for input impedance, gain accuracy, noise, and common-mode rejection. The test conditions matter: a CMRR measured with 10 kΩ source impedances is not the same as one measured with 1 kΩ or 100 kΩ. When a datasheet quotes a CMRR, read the test condition before comparing it with another device.

### 8.3 What IEC 60601-1-2 adds for interference

IEC 60601-1-2:2014+AMD1:2020 sets immunity test levels that differ by environment. Electrostatic discharge is 8 kV contact and 15 kV air in both environments. Radiated radio-frequency immunity from 80 MHz to 2.7 GHz is 3 V/m in a professional healthcare facility and 10 V/m in a home healthcare environment, with 80% amplitude modulation at 1 kHz. Conducted radio-frequency immunity is 3 V RMS from 150 kHz to 80 MHz, with 6 V RMS in the industrial, scientific, and medical bands. Power-frequency magnetic field immunity is 30 A/m at 50/60 Hz. Amendment 1 adds proximity magnetic field tests at 30 kHz, 134.2 kHz, and 13.56 MHz for devices used near wireless power transfer and near-field communication.

A device that passes these tests can still show interference in use. The standard defines a test level, not a guarantee of signal quality. A 3 V/m field at a particular frequency may be below the level that saturates an unprotected front end but above the level that produces a visible artefact in a low-amplitude channel. The data team should know which tests were performed, at which levels, and with which cable configuration.

### 8.4 What IEC 60601-1-8 says about delay, and what it does not

A widespread misreading holds that IEC 60601-1-8 imposes a 10 s maximum alarm delay. It does not. Clause 6.4.1 requires disclosure in the instructions for use when the sum of the maximum alarm condition delay and the maximum alarm signal generation delay exceeds 10 s, or when the sum of the mean delays exceeds 5 s. The clause is a transparency requirement, not a pass/fail timing limit. Particular standards, such as those for ventilators or infusion pumps, may impose tighter limits, and a risk analysis may justify a shorter delay for a specific alarm.

The distinction matters for data science. If a device's instructions for use disclose a mean delay of 8 s, an alert that arrives 8 s after the physiological event is behaving as labelled, not malfunctioning. A model that assumes instantaneous input is inconsistent with the device's own specification.

### 8.5 What the standards do not cover

Standards constrain the device, not the analysis. They do not specify:

- Which algorithm should process the signal.
- How the signal should be sampled for a retrospective study.
- Whether a model trained on one device will generalise to another.
- How to annotate artefacts or handle missing data.
- How to report subgroup performance.

Those are the responsibility of the data team, and they are where most of the methodological risk lies. A device can be fully compliant and still produce data that are unsuitable for a model, because the compliance testing used clean signals and the model sees real ones.

### 8.6 The regulatory pathway in one paragraph

In the United States, a device is classified into class I, II, or III according to risk. Most monitoring devices are class II and reach the market through the 510(k) pathway, which requires substantial equivalence to a legally marketed predicate device. Some novel devices use the De Novo pathway, and high-risk devices use premarket approval. In the European Union, the Medical Device Regulation requires a clinical evaluation, a technical file, and conformity assessment by a notified body for most class IIa and higher devices. Both jurisdictions require risk management to ISO 14971 and software lifecycle evidence to IEC 62304, although the precise recognition differs. A clearance is not a clinical outcome study; it is a determination that the device meets a regulatory standard for its intended use.

## 9. Industry practice and case evidence

### 9.1 Integrated front ends are the default for biopotentials

Most wearable and portable ECG designs use an integrated analog front end rather than a discrete instrumentation amplifier. A representative part, the Texas Instruments ADS1292, integrates two 24-bit delta-sigma converters, a programmable gain amplifier, a driven-right-leg amplifier, lead-off detection, and a bias amplifier in a single package. Its datasheet (SBAS502C, April 2020) specifies 8 µV peak-to-peak input-referred noise at a gain of 6 and a 150 Hz bandwidth, 335 µW per channel, a minimum CMRR of 105 dB and a typical CMRR of 120 dB at 50/60 Hz, and an input bias current of ±200 pA at 25 °C.

The integration changes the data scientist's problem. The gain, filter, and lead-off behaviour are configured by register writes, not by visible components. The same silicon can produce different data in two products because the firmware sets different register values. The configuration must be captured with the data, or the provenance is incomplete.

**Documented fact:** the datasheet numbers above are from the manufacturer's revision C. **Inference:** the prevalence of this architecture in wearables is inferred from its feature set and from teardown reports, not from a census of products.

### 9.2 The pulse oximeter bias case

Pulse oximetry is a signal chain in which the sensor, the tissue, and the calibration curve are all part of the measurement. A 2024 meta-analysis reported pooled accuracy root-mean-square (A<sub>rms</sub>) values of 3.96% for light skin, 4.71% for medium skin, and 4.15% for dark skin, with mean bias of 0.70%, 0.27%, and 1.27% respectively. The Food and Drug Administration's January 2025 draft guidance proposes stricter validation across diverse skin pigmentation and non-disparity criteria.

The chain interpretation is direct. The optical path includes melanin absorption, which changes the ratio of red to infrared signals. The calibration curve was derived from a reference dataset that may under-represent dark skin. The device can meet a headline accuracy specification and still fail a subgroup, because the specification was computed on a pooled population. A data scientist who receives only the pooled number cannot detect the gap.

**Documented fact:** the meta-analysis and the draft guidance are public. **Inference:** the mechanism by which melanin changes the ratio is well established in optics, but the relative contribution of calibration versus optical path in any specific device is not public.

### 9.3 Continuous glucose monitoring and lag

Continuous glucose monitors estimate interstitial glucose from an enzymatic electrochemical sensor. The interstitial compartment lags blood glucose by roughly 5–15 minutes, depending on the sensor and the direction of change. Compression of the sensor site can produce a false low. Acetaminophen and other interferents can bias the reading depending on the sensor chemistry. Published mean absolute relative differences cluster around 9–14% across systems.

For a data scientist, the CGM value is a model output with a known lag and a state-dependent error. Aligning it with a blood glucose reference without accounting for the lag produces an apparent bias that depends on the rate of change. A model that predicts hypoglycaemia from CGM must be trained with the same lag it will experience at inference.

**Documented fact:** lag, compression lows, and interferents are described in the clinical literature and in product labelling. **Inference:** the exact lag distribution for a given sensor is not usually published; it must be estimated from paired data.

### 9.4 Surgical robot recalls show the chain extends into software

In 2025, Virtual Incision's MIRA system was recalled for a software issue that could interrupt or lag motion between the surgeon's controllers and the robot arms. Quantum Surgical's EPIONE was recalled for a software bug that could misalign the robotic arm relative to the planned trajectory. A CMR Surgical Versius console recall involved Secure Boot not being enabled.

These are not sensor failures in the analogue sense. They are failures of the digital portion of the chain: the command path, the coordinate transform, and the boot integrity check. They demonstrate that the signal chain for a robot includes the controller, the network, and the firmware, and that a latency or coordinate error can have procedural consequences.

**Documented fact:** the recalls are listed in the FDA recall database and in manufacturer notices. **Inference:** the root-cause analyses are not public; the descriptions above are limited to the recall notices.

### 9.5 What good practice looks like

Organisations that manage device data well tend to do the same things:

- They request the rawest signal the device can export, plus the device-computed values, plus the configuration.
- They record firmware version, gain, filter setting, sampling rate, and calibration date with every record.
- They test the chain at the edges, not only at mid-scale.
- They keep a bench record of the device's response to a known input, so that a later change can be detected.
- They treat a firmware update as a change to the measurement, not only to the software.
- They separate the question "is the device accurate?" from "is the model accurate on this device's data?"

None of these practices requires a laboratory. They require a data dictionary and the discipline to fill it.

### 9.6 Where inference is unavoidable

Some of the most useful quantities are not public: the internal filter coefficients of a commercial device, the exact calibration curve for a pulse oximeter, the noise correlation between channels, and the firmware's handling of a saturated sample. In these cases, the honest approach is to state the inference, describe the experiment that would test it, and avoid presenting the inference as a specification. A model that depends on an unverified internal filter is a model with an unquantified risk.

### 9.7 Reading a vendor specification

A vendor specification is a claim about a population of units under stated conditions. Read it in four parts: the value, the tolerance, the conditions, and the test method. "0.5% accuracy" without a range, a temperature, and a reference method is not a specification. "±2 mmHg from 0 to 300 mmHg at 23 ± 5 °C against a calibrated reference, 95% of readings" is a specification.

The same discipline applies to software claims. "AI-powered" does not describe a function; "detects atrial fibrillation from a 30 s single-lead ECG with a sensitivity of 95% in a stated population" does. A claim that cannot be tested cannot be verified, and an unverifiable claim should not enter a design review as if it were evidence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Front-end saturation on electrode offset | Flat-topped waveform; codes at full scale | Gain times offset exceeds rail | Loss of signal; false asystole or false Normal | Headroom calculation; offset servo; saturation flag |
| Common-mode rejection degraded by electrode mismatch | 50/60 Hz tone that varies with patient movement | Mismatch converts common-mode to differential | Interference mistaken for signal | Buffer at electrode; match impedances; check system CMRR |
| Anti-alias filter omitted or set too high | High-frequency noise folded into the band | Sampling below twice the highest input frequency | Irreversible corruption | Filter before ADC; verify with a frequency sweep |
| Quantisation limited by low gain | Visible steps in a small signal | Signal occupies few codes | Poor resolution; biased features | Raise gain or use a higher-resolution converter with offset removal |
| Reference noise dominates | Output noise changes with supply or temperature | Reference feeds the converter's full-scale | Gain error and noise | Low-noise reference; decoupling; ratiometric design |
| Clock jitter | SNR falls at high input frequency | Aperture uncertainty | Degraded high-frequency content | Low-jitter clock; lower input bandwidth |
| Latency exceeds the clinical requirement | Alarm arrives after the event | Filter delay, window length, network | Alert is clinically useless | Latency budget; shorter window; edge processing |
| Dropped frames | Gaps at regular intervals | Buffer overrun or radio retry | Missing data; false trends | Sequence numbers; buffer sizing; backpressure |
| Brown-out corrupts samples | Reset flags; abrupt code jumps | Supply falls below minimum | Invalid data; lost calibration | Supply monitor; brown-out reset; non-volatile state |
| Watchdog reset loop | Repeated boot messages; gaps | Firmware fails to service watchdog | Device unavailable | Reset cause logging; fail-safe mode; watchdog timeout review |
| Thermal throttling | Latency increases after warm-up | Processor reduces clock | Missed deadlines | Thermal design; workload profiling; temperature logging |
| Firmware update failure | Version mismatch; boot failure | Incomplete write; signature failure; no rollback | Device bricked or unvalidated code | A/B partitions; signed images; rollback; update verification |
| Calibration drift | Slow bias versus reference | Sensor aging, temperature, sterilisation | Systematic error in all values | Periodic calibration; drift budget; reference checks |
| Wrong channel mapping | Plausible signal uncorrelated with the patient | Connector or firmware mapping error | Misattributed measurement | Cross-check; channel identification test |
| Timestamp offset | Signals misaligned by a constant | Clock drift or time-zone handling | Incorrect causality and features | Record clock source; synchronise; log offsets |

Two red flags deserve emphasis. First, a waveform that looks clean is not necessarily correct: a saturated channel can produce a flat line that a morphology check accepts. Second, a noise specification quoted without a bandwidth is not a specification. Both are cheap to detect at review time and expensive to discover in a clinical study.

### 10.1 Prioritising the list

The table is not a checklist to apply uniformly. Failures differ in severity, frequency, and detectability, and the effort should follow the risk. A useful prioritisation asks three questions:

1. **What is the worst credible harm?** A saturation that hides asystole is more severe than a timestamp offset of 10 ms, even though both corrupt the data.
2. **How often does it occur in the intended population?** Electrode offset saturation may be rare in a supervised clinic and common in a home wearable.
3. **How would we know?** A failure with no detector is more dangerous than one that raises a flag, because it can persist silently for the life of the device.

The combination of high severity and low detectability deserves a design change, not a monitoring plan. A failure that is severe but reliably detected can be handled with an alarm and a workflow. A failure that is mild but undetectable is usually acceptable. The dangerous quadrant is severe and silent, and it is where a data scientist should spend review time.

### 10.2 The data-side red flags

Some failures are visible only in the data, not in the device. A sudden change in the distribution of a feature, with no change in the patient population, is a configuration-change signal. A saturation rate that differs between sites is a training or electrode signal. A missingness pattern that aligns with shift changes is a workflow signal. None of these is proof of a hardware fault, but each is a reason to check the chain before retraining the model.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The source impedance is known and stable | Noise and CMRR calculations depend on it | Dry electrodes, motion, gel drying, cable damage | Measure impedance at the electrode; log it |
| The electrode offset is within the specified range | Gain and headroom are sized for it | Poor preparation, different gel, offset drift | Measure open-circuit offset; add saturation detection |
| Noise sources are uncorrelated | Root-sum-square addition is valid | Shared reference, supply coupling, crosstalk | Measure with inputs shorted and with a common source |
| The noise bandwidth equals the signal bandwidth times a factor | Converts density to RMS | Higher-order filters, digital filtering, averaging | Measure the transfer function; simulate |
| The datasheet typical applies to the design | Budget uses typical values | Process variation, temperature, supply | Use maximum values for worst-case; test multiple units |
| The ADC's input-referred noise is negligible | Converter is treated as ideal | Low gain, high resolution, internal buffer noise | Measure code histogram with a shorted input |
| The anti-alias filter attenuates above Nyquist | Aliasing is avoided | Filter tolerance, component drift, layout parasitics | Sweep above Nyquist; inject a tone |
| The clock is stable enough | Timestamps and sampling are accurate | Crystal tolerance, temperature, ageing | Compare with a reference clock over 24 h |
| The firmware version is the validated version | Behaviour matches the evidence | Auto-update, field service, rollback | Record version; verify at ingest; alert on change |
| The clinical population matches the validation population | Accuracy generalises | Skin tone, perfusion, body size, comorbidity | Subgroup analysis; stratified validation |
| The environment matches the test environment | EMC and thermal performance hold | Home use, ambulance, operating room | Test at the intended immunity levels; log environment |
| The operator uses the device as intended | Human factors are adequate | Off-label placement, poor training | Usability testing; use-error analysis |

The most consequential assumption is usually the one that is not written down. A team that assumes the electrode offset is below 300 mV because the datasheet says so has confused a test condition with a guarantee. A team that assumes the firmware is unchanged because no one mentioned an update has confused a process with a fact.

A second limit is scope. This topic describes the chain at the level needed to reason about data. It does not cover circuit layout, electromagnetic compatibility design, or the detailed test methods of IEC 60601-1. Those require a specialist. The data scientist's contribution is to ask the questions that determine whether the data are fit for the intended analysis, and to record the answers.

A third limit is that the numbers in the worked example are illustrative. The structure of the budget is general; the values are not. A different front end, a different electrode, or a different bandwidth changes every term. Recompute, do not copy.

A fourth limit is the treatment of noise as stationary and Gaussian. Real device noise includes flicker noise at low frequency, interference that is not Gaussian, and occasional transients from motion or electrosurgery. The RMS budget is a useful summary for a stationary environment; it does not describe the tail of the distribution, which is what an alarm threshold depends on. If the alarm threshold is set at five times the RMS noise, the false-alarm rate depends on the tail, not on the RMS. Measure the tail, or model it explicitly.

A fifth limit is that the chain is treated as linear until the clipping stage. Real front ends have gain compression before hard clipping, and real sensors have hysteresis and nonlinearity. A signal that is 80% of full scale may already be compressed by 1%. For a measurement that claims 0.1% accuracy, the compression matters. Characterise the transfer function across the range, not only at the point used for calibration.

A sixth limit is that the analysis assumes the configuration is static. Devices change mode: a monitor may switch gain when the patient moves, a ventilator may change its flow sensor range, a robot may switch control mode. Each switch is a change in the transfer function. The data record must include the mode, or the analysis will pool data from different chains and attribute the difference to the patient.

These limits do not make the budget useless. They define the conditions under which it is valid, which is what an engineering claim requires. A budget without stated limits is a calculation; a budget with stated limits is evidence.

A final distinction is between verification and validation, which are often conflated in data discussions. Verification asks whether the chain meets its written specification: the gain is 6.0 ± 0.1, the bandwidth is 150 Hz ± 10%, the noise is below 1.2 µV RMS. Validation asks whether the chain measures what it is intended to measure in the intended use environment: the heart rate is accurate for the patients and conditions in the labelled use. A chain can be fully verified and still fail validation, because the specification was wrong. The data scientist usually sees the validation evidence and not the verification evidence; asking for both is a useful habit.

## 12. Exercises

The eight exercises move from description to calculation to diagnosis to design. They are intended to be done in order, because Exercise 2 supplies the budget that Exercises 3 and 7 modify. Each exercise states a deliverable, acceptance criteria, and a time box. The time boxes assume access to a spreadsheet or Python and a device manual.

**How to use the exercises.** Do not look at Section 13 before attempting the calculation. The answers are written to show the method, not only the result, and reading them first removes the benefit. For the design exercises, there is no single correct answer; the rubric in Section 13 is the standard.

### Exercise 1. Draw and annotate a chain (recall, 20 minutes)

Choose a device you can find a public manual for: a home blood-pressure monitor, a wearable heart-rate monitor, or a hospital thermometer. Draw the chain from transducer to display. Annotate each stage with its input range, output range, and one failure mode.

**Deliverable:** a one-page diagram with a caption.

**Acceptance criteria:** at least five stages; units on every range; one failure mode per stage; no unsupported internal details.

**Time:** 20 minutes.

### Exercise 2. Compute a noise budget (application, 40 minutes)

A pressure channel uses a bridge with a 350 Ω source impedance, an amplifier with 15 nV/√Hz voltage noise and 2 pA/√Hz current noise, a 100 Hz first-order low-pass filter, and a 16-bit converter with a 2.5 V reference and a gain of 50. Compute the input-referred noise, the input-referred LSB, the SNR for a 10 mV signal, and the ENOB.

**Deliverable:** a spreadsheet or script with the intermediate values and a one-paragraph conclusion.

**Acceptance criteria:** correct noise bandwidth; current noise converted through the source impedance; LSB referred to the input; SNR and ENOB consistent with the noise; units on every number.

**Time:** 40 minutes.

### Exercise 3. Diagnose a saturation (application, 30 minutes)

An ECG channel with a gain of 100 and a ±2.5 V input range produces a flat-topped waveform. The electrode offset is measured at 28 mV. Explain why the channel saturates, calculate the maximum offset that would not saturate, and propose two design changes with their costs.

**Deliverable:** a short memo with the calculation and the trade-off.

**Acceptance criteria:** correct headroom arithmetic; two changes; each change states its effect on noise or bandwidth; no recommendation without a cost.

**Time:** 30 minutes.

### Exercise 4. Critique a CMRR claim (critique, 25 minutes)

A vendor states "CMRR > 100 dB" for a wearable ECG patch. Write five questions that would determine whether the claim is relevant to a system-level interference problem.

**Deliverable:** five questions with a one-sentence rationale each.

**Acceptance criteria:** questions cover test condition, source impedance, frequency, input impedance, and system-level measurement; no question is answerable by a marketing brochure alone.

**Time:** 25 minutes.

### Exercise 5. Build a latency budget (application, 30 minutes)

An algorithm uses a 4 s window, a 201-tap FIR filter at 250 samples per second, a 5-sample median filter, a 20 ms network, and a 60 Hz display. The processor adds 15 ms. Compute the end-to-end latency and identify the two largest contributors. Propose one change that reduces latency by at least 50% and state its effect on the algorithm.

**Deliverable:** a table and a short recommendation.

**Acceptance criteria:** FIR delay formula correct; all stages included; the proposed change is feasible and its effect is stated.

**Time:** 30 minutes.

### Exercise 6. Design a provenance record (design, 35 minutes)

Design a metadata schema for a waveform record from a wearable device. Include at least 15 fields, their types, and why each is needed to detect a change in the measurement.

**Deliverable:** a table with field, type, example, and purpose.

**Acceptance criteria:** covers device identity, firmware, sampling, gain, filter, calibration, sensor lot, timestamps, quality flags, and environment; each field has a stated use.

**Time:** 35 minutes.

### Exercise 7. Test an assumption (critique, 40 minutes)

Take the worked example in Section 6 and change the electrode mismatch from 300 Ω to 30 Ω. Recompute the system CMRR and the residual at 1 V common mode. Then change the amplifier CMRR from 105 dB to 120 dB and recompute. Which change matters more, and why?

**Deliverable:** a calculation and a three-sentence answer.

**Acceptance criteria:** correct arithmetic; the answer distinguishes amplifier CMRR from system CMRR; the conclusion is supported by the numbers.

**Time:** 40 minutes.

### Exercise 8. Write a verification matrix (design, 45 minutes)

Write a verification matrix for the chain in Section 6. Include input amplitude, offset, frequency, common-mode, supply, and temperature conditions. State the acceptance criterion and the expected margin for each cell.

**Deliverable:** a matrix with at least 12 cells.

**Acceptance criteria:** covers the edges, not only nominal; each cell has a criterion and a margin; the saturated and aliased cases are included.

**Time:** 45 minutes.

## 13. Self-check answers

**Answer 1 (Exercise 1).** A home blood-pressure monitor chain: cuff and pressure sensor (0–300 mmHg, ±3 mmHg), analogue front end (gain 50, 0.5–5 Hz), 16-bit converter (2.5 V reference), microcontroller (deflation control and oscillometric envelope), display (systolic, diastolic, pulse). Failure modes: cuff leak, motion artefact, quantisation at low pressure, firmware averaging error, user posture. The exact internal values are not public; the diagram should mark inferred stages as inferred.

**Answer 2 (Exercise 2).** Noise bandwidth = 1.571 × 100 = 157.1 Hz. Source thermal noise for 350 Ω at 310 K = √(4 × 1.38e-23 × 310 × 350) = 2.45 nV/√Hz; over 157.1 Hz = 30.7 nV RMS. Amplifier voltage noise = 15 × 12.53 = 188 nV RMS. Current noise = 2 pA/√Hz × 350 Ω = 0.7 nV/√Hz, negligible. Total ≈ √(30.7² + 188²) = 190.5 nV RMS. Input-referred LSB = 5 V / (50 × 65,536) = 1.526 µV. The LSB is eight times the noise, so the converter limits resolution, not the amplifier. SNR for 10 mV RMS = 20 log10(0.01 / 190.5e-9) = 94.4 dB; ENOB = (94.4 − 1.76)/6.02 = 15.4 bits. Note that this ENOB is close to the converter's 16 bits only because the signal is defined as RMS and the noise is small; the converter's own noise and distortion set the practical ceiling.

**Answer 3 (Exercise 3).** With a gain of 100 and a ±2.5 V range, the maximum input is 25 mV. A 28 mV offset exceeds it by 3 mV, so the channel saturates. The maximum offset that would not saturate is 25 mV minus the signal amplitude. Two changes: reduce the gain to 50, which doubles the range to ±50 mV but raises the input-referred noise by 6 dB if the amplifier noise dominates; or add a DC servo that removes the offset, which preserves the gain but adds a high-pass corner and settling time. A third option, an offset DAC, preserves both but adds complexity and calibration.

**Answer 4 (Exercise 4).** Questions: (1) At what source impedance and frequency was the CMRR measured? (2) Was it measured at the amplifier input or at the system level with electrodes? (3) What is the amplifier's common-mode input impedance, and how does it interact with electrode mismatch? (4) What was the common-mode amplitude and waveform? (5) Does the claim hold across the operating temperature and supply range? A strong answer to any of these requires a test report, not a datasheet headline.

**Answer 5 (Exercise 5).** FIR delay = (201 − 1)/(2 × 250) = 0.400 s. Median delay ≈ 2 samples / 250 = 8 ms. Algorithm window = 4.0 s. Network = 20 ms. Display = 16.7 ms. Processor = 15 ms. Total ≈ 4.46 s. The two largest are the algorithm window (4.0 s) and the FIR filter (0.4 s). Reducing the window to 1.5 s cuts 2.5 s, or 56%, but reduces the number of beats available for rate estimation and may increase variance. The FIR could be shortened, but that changes the filter response and must be re-verified.

**Answer 6 (Exercise 6).** Fields: device model; serial or pseudonymous ID; firmware version; hardware revision; sampling rate; bit depth; gain; filter type and corner; calibration date and coefficients; sensor lot; electrode type and site; acquisition start time in device clock; receive time in host clock; clock offset estimate; battery state; signal-quality flags; saturation count; temperature. Each field answers a question that would otherwise be answered by guessing.

**Answer 7 (Exercise 7).** With 30 Ω mismatch and 10 MΩ input impedance, Vdiff = 1 V × 30/10e6 = 3 µV, system CMRR = 110.5 dB. With 300 Ω, 30 µV and 90.5 dB. Changing the amplifier from 105 dB to 120 dB changes its own residual from 5.6 µV to 1 µV, but the 30 µV mismatch term still dominates. The mismatch change matters more by a factor of ten. The lesson: fix the dominant term, not the specification that looks impressive.

**Answer 8 (Exercise 8).** A complete matrix includes: zero input at three temperatures; 1 mV and 5 mV signals at 1 Hz, 40 Hz, and 150 Hz; +300 mV and −300 mV differential offset; +300 mV and −300 mV common-mode offset; 1 V RMS 50 Hz common mode; supply at 2.7 V, 3.0 V, and 3.3 V; a 100 mV supply step; a 250 Hz tone above Nyquist; and a 24 h soak. Acceptance criteria follow the requirement table: gain error < 2%, offset < 10 µV, CMRR > 90 dB system, no aliasing above 1 LSB, and no drift above 5 µV over 24 h. Each cell should report the measured value and the margin, not only pass or fail.

**Rubric for the design exercises.** A strong design names the dominant term, quantifies the trade, states the assumption that could break it, and proposes a test. A weak design lists components without quantities, recommends a change without a cost, or presents a typical value as a guarantee.

**Common calculation errors and how they are marked.**

| Error | Example | Marking consequence |
|---|---|---|
| Using signal bandwidth as noise bandwidth | 150 Hz instead of 235.6 Hz | 1.96 dB underestimate; half credit if the method is otherwise correct |
| Adding noise amplitudes instead of powers | 0.8 + 0.5 = 1.3 µV instead of 0.94 µV | Wrong total; no credit for the total |
| Omitting current noise into a high source impedance | Ignoring 1 pA/√Hz × 1 MΩ | Large underestimate; half credit |
| Mixing input and output reference points | Adding ADC noise without dividing by gain | Order-of-magnitude error; no credit for the budget |
| Using typical values for a worst-case claim | 20 nV/√Hz typical instead of 35 nV/√Hz maximum | The design is not worst-case; state the distinction for credit |
| Quoting ENOB from the marketed bit count | "24 bits" | Conceptual error; no credit for the resolution claim |
| Treating peak-to-peak as RMS | 5.6 µV peak-to-peak used directly | Factor 6.6 error; no credit unless the conversion is shown |
| Ignoring the offset in the headroom | Gain chosen from signal amplitude alone | Saturation in the field; no credit for the gain choice |

The marking rule is consistent: method first, arithmetic second, units third. A correct method with an arithmetic slip loses less than an incorrect method with a right answer. A number without a unit loses one mark on every occurrence.

## 14. Interview questions

The questions below are grouped by the four skills this topic develops: describing the chain, quantifying it, diagnosing it, and judging evidence. A strong answer contains a mechanism, a number with a unit, and a consequence. An answer that contains only vocabulary is not sufficient for a hardware-facing role.

| Skill | Questions | What a strong answer demonstrates |
|---|---|---|
| Describe | 1, 5 | Correct order, correct terms, awareness of information loss |
| Quantify | 2, 3, 4, 8 | A budget that closes, with units and tolerances |
| Diagnose | 6, 10, 12 | A specific test and a specific failure signature |
| Judge evidence | 7, 9, 11 | Separation of specification, clearance, and clinical evidence |

**1. Walk me through a medical device signal chain.**
A strong answer names the stages in order, gives an example of the quantity at each stage, and identifies where information can be lost. A weak answer lists components without explaining what each does or how it fails.

**2. What is CMRR, and why does it matter?**
A strong answer defines it as the ratio of differential to common-mode gain, gives a dB value, and explains that system CMRR depends on source-impedance matching and input impedance, not only on the amplifier. A weak answer quotes a datasheet number without test conditions.

**3. Why can a 24-bit ADC give an 8-bit measurement?**
A strong answer distinguishes resolution from ENOB, computes ENOB from SNR, and identifies the dominant noise source. A weak answer assumes more bits always mean more accuracy.

**4. How do you choose a gain for a biopotential front end?**
A strong answer writes the headroom equation, includes the electrode offset, and explains the trade between gain and input range. A weak answer picks a gain because it is common or because the datasheet recommends it.

**5. What is the difference between signal bandwidth, noise bandwidth, and sampling rate?**
A strong answer defines each, gives the first-order noise-bandwidth factor, and explains the Nyquist condition. A weak answer uses the three terms interchangeably.

**6. How would you detect that a channel is saturated?**
A strong answer checks the code histogram, looks for flat-topped waveforms, counts samples at full scale, and cross-checks with an independent signal. A weak answer relies on a visual inspection of the waveform alone.

**7. Why does a firmware update matter to a data scientist?**
A strong answer treats it as a change to the measurement transfer function and requires version metadata and re-verification. A weak answer treats it as an IT event unrelated to the data.

**8. How do you budget latency for a real-time alert?**
A strong answer lists filter delay, conversion, window length, processing, network, and display, and identifies the dominant term. A weak answer quotes the processor speed or the algorithm's inference time only.

**9. What does IEC 60601-1-8 require for alarm delay?**
A strong answer states that it requires disclosure above a threshold, not a hard 10 s limit, and notes that particular standards may be stricter. A weak answer repeats the common misconception that 10 s is a legal maximum.

**10. When would you stop trusting a device dataset?**
A strong answer names a provenance gap, an unexplained configuration change, a saturation rate that biases the cohort, or a subgroup performance gap, and proposes a specific check. A weak answer says "when the model performs badly" without distinguishing data from model failure.

**11. How would you compare two devices measuring the same quantity?**
A strong answer compares the chain, the test conditions, the subgroups, and the time alignment, and uses paired measurements. A weak answer compares headline accuracy numbers from two datasheets.

**12. What is the most common signal-chain mistake you have seen?**
A strong answer names a specific, checkable mistake, such as quoting typical noise as a maximum or ignoring electrode offset in the gain calculation, and explains the consequence. A weak answer is generic.

## 15. Key terms

- **Aliasing** — folding of frequencies above the Nyquist frequency into lower frequencies; irreversible.
- **Analog front end (AFE)** — the circuit that amplifies, filters, and conditions a sensor signal before conversion.
- **Applied part** — the part of a device that contacts the patient; classified B, BF, or CF.
- **Brown-out** — a supply voltage drop below the operating minimum, often causing a reset.
- **Clipping** — output limiting at a rail; the input amplitude is no longer represented.
- **CMRR** — common-mode rejection ratio; differential gain divided by common-mode gain, in dB.
- **Common-mode voltage** — the average of two input voltages relative to a reference.
- **Crest factor** — peak value divided by RMS value; used to convert peak-to-peak noise to RMS.
- **Driven right leg (DRL)** — a circuit that drives the patient reference to reduce common-mode voltage.
- **Effective number of bits (ENOB)** — the resolution implied by the measured SNR or SINAD.
- **Electrode offset** — the DC potential at an electrode-skin interface, often hundreds of millivolts.
- **Group delay** — the delay of a signal's envelope through a filter.
- **Headroom** — the margin between the expected signal and the range limit.
- **Input-referred noise** — noise converted to the input of a stage by dividing by the gain.
- **Jitter** — variation in the sampling instant; degrades high-frequency SNR.
- **Latency** — time from a physical event to the output.
- **Least significant bit (LSB)** — the smallest step of a converter, in volts at its input.
- **Noise bandwidth** — the bandwidth of an ideal brick-wall filter that would pass the same noise power.
- **Nyquist frequency** — half the sampling rate.
- **Patient leakage current** — current flowing through the patient to earth; limited by IEC 60601-1.
- **Provenance** — the record of device, configuration, and processing that produced a value.
- **Quantisation noise** — error from mapping a continuous value to discrete codes; q/√12 RMS for an ideal converter.
- **Saturation** — the condition in which a stage can no longer follow its input.
- **Signal-to-noise ratio (SNR)** — signal power divided by noise power, in dB.
- **Source impedance** — the impedance of the sensor and its connection.
- **Thermal noise** — voltage noise from random charge motion; √(4kTR) in V/√Hz.
- **Timestamp** — the recorded time of a sample, with a stated clock source.
- **Typical value** — a value expected for most units, not a guarantee.
- **Anti-alias filter** — a filter placed before sampling to attenuate frequencies above Nyquist.
- **Differential input** — an input that responds to the difference between two terminals.
- **Input bias current** — the current flowing into an amplifier's input; converts to noise through the source impedance.
- **Settling time** — the time for an output to reach a stated accuracy after a step.
- **Transfer function** — the mathematical relationship between a stage's input and output.
- **Full-scale range** — the input range that maps to a converter's entire code range.
- **Noise floor** — the noise power present in a measurement in the absence of signal.
- **Reference electrode** — the electrode that defines the patient's reference potential.
- **Signal conditioning** — amplification, filtering, isolation, and other processing before conversion.
- **Common-mode input impedance** — the impedance seen by a common-mode source at the amplifier input.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Standards and regulation**

1. IEC 60601-1:2005+AMD1:2012+AMD2:2020, *Medical electrical equipment — Part 1: General requirements for basic safety and essential performance*. https://webstore.iec.ch/en/publication/26169
2. IEC 60601-1-2:2014+AMD1:2020, *Electromagnetic disturbances — Requirements and tests*. https://webstore.iec.ch/en/publication/67554
3. IEC 60601-1-8:2006+AMD2:2020, *Alarm systems — General requirements, tests and guidance*. https://webstore.iec.ch/en/publication/26539
4. IEC 60601-2-25:2011, *Particular requirements for electrocardiographs*. https://webstore.iec.ch/en/publication/26368
5. IEC 60601-2-27:2011, *Particular requirements for electrocardiographic monitoring equipment*. https://webstore.iec.ch/en/publication/26400
6. ISO 14971:2019, *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
7. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
8. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
9. ISO 13485:2016, *Medical devices — Quality management systems*. https://www.iso.org/standard/59752.html
10. U.S. Food and Drug Administration, *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions* (guidance). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket
11. Regulation (EU) 2017/745 on medical devices. https://eur-lex.europa.eu/eli/reg/2017/745/oj

**Datasheets and application notes**

12. Texas Instruments, *ADS129x Low-Power, 2-Channel, 24-Bit Analog Front-End for Biopotential Measurements*, SBAS502C, April 2020. https://www.ti.com/lit/ds/symlink/ads1292.pdf
13. Analog Devices, *MT-001: Taking the Mystery out of the Infamous Formula, SNR = 6.02N + 1.76 dB*, Rev. A. https://www.analog.com/media/en/training-seminars/tutorials/MT-001.pdf
14. Analog Devices, *MT-229: Quantization Noise: An Expanded Derivation of SNR = 6.02N + 1.76 dB*. https://www.analog.com/media/en/training-seminars/tutorials/MT-229.pdf

**Clinical and device evidence**

15. "Accuracy of Pulse Oximetry by Skin Pigmentation: Systematic Review and Meta-analysis," *Journal of Medical Internet Research*, 2024. https://www.jmir.org/2024/1/e62769
16. U.S. Food and Drug Administration, draft guidance on pulse oximeter accuracy and diverse pigmentation, January 2025. https://www.fda.gov/media/184896/download
17. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
18. Intuitive Surgical, da Vinci 5 FDA clearance announcement, March 2024. https://isrg.intuitive.com/news-releases/news-release-details/intuitive-announces-fda-clearance-fifth-generation-robotic
19. Medtronic, Hugo FDA clearance announcement, December 2025. https://news.medtronic.com/2025-12-03-Medtronic-announces-FDA-clearance-of-Hugo-TM-robotic-assisted-surgery-system-for-urologic-surgical-procedures

**Background**

20. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009. ISBN 978-0-471-67622-5.
21. ISO 80601-2-61:2017, *Particular requirements for pulse oximeters*. https://www.iso.org/standard/67963.html
22. IEC 60601-1-11:2015+AMD1:2020, *Requirements for medical electrical equipment and medical electrical systems used in the home healthcare environment*. https://webstore.iec.ch/en/publication/67384
23. U.S. Food and Drug Administration, 510(k) premarket notification database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm
24. AAMI, *ANSI/AAMI ES60601-1:2005/(R)2012 and A1:2012, C1:2009/(R)2012 and A2:2010/(R)2012*, consolidated text. https://www.aami.org/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_sensors.md`](02_sensors.md)
- Analogue depth: [`03_analog-front-ends.md`](03_analog-front-ends.md)
- Conversion depth: [`04_adcs-and-signal-conditioning.md`](04_adcs-and-signal-conditioning.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on ECG quality: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)
- Lab on quantization: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab on signal-chain simulation: see Lab 14A.1 in the survey file.
- Lab on hardware-in-the-loop testing: see Lab 14A.7 in the survey file.
- Device architecture teardown: see Lab 14A.6 in the survey file.
- Lab index: [`../../labs/`](../../labs/)

- Lab on quantisation: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
