# Analog Front Ends

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 03 of 14
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

The analog front end (AFE) is the stage between the sensor and the converter. It amplifies a small, high-impedance, interference-prone signal into a large, low-impedance, band-limited one. It is also the stage where most of the difficult trade-offs in a medical device are made: noise versus power, gain versus input range, bandwidth versus settling, common-mode rejection versus input impedance, and protection versus signal fidelity.

The industry problem is that AFE behaviour is invisible in the data. A model receives a digitised waveform and has no direct evidence of the gain, the filter, the reference, or the common-mode loop that produced it. When the waveform changes, the team cannot tell whether the patient changed or the AFE changed. Three decisions depend on AFE literacy:

1. **Whether a signal is limited by the sensor or the electronics.** A 2 µV noise floor may come from the electrode, the amplifier, or the reference. The fix is different in each case.
2. **Why interference appears and why a filter did or did not remove it.** Common-mode interference is not the same as differential noise; a differential filter cannot remove a common-mode voltage that has already been converted to differential by an impedance mismatch.
3. **What a gain or filter change does to the model.** A gain change alters the quantisation and the saturation point; a filter change alters the phase and the group delay. Both can invalidate a model trained on the previous configuration.

This topic covers the AFE as a system: input protection, topology, gain, filtering, common-mode control, noise, stability, and verification. It is written for a reader who must specify, review, or diagnose an AFE, not for a reader who is laying out a printed circuit board. The arithmetic is the point: an AFE that is not budgeted is an AFE that will be discovered in the field.

A useful framing: the AFE is a negotiation between the source and the converter. The source has an impedance, an offset, and a noise floor. The converter has a range, a resolution, and a sampling rate. The AFE's job is to make the source fit the converter without adding more noise than the source already has, and without letting the source's faults damage the converter or the patient.

## 2. Learning outcomes

By the end of this topic you can:

- Choose an AFE topology from the source impedance, signal amplitude, offset, bandwidth, and power budget.
- Compute the input-referred noise of an instrumentation amplifier including voltage noise, current noise, resistor noise, and gain.
- Explain why a discrete difference amplifier with 1% resistors has a common-mode rejection ratio of about 40 dB regardless of the operational amplifier.
- Design a driven-right-leg or guard circuit and estimate its effect on the common-mode voltage.
- Allocate gain and offset so that the signal, the electrode offset, and the interference all fit within the amplifier and converter ranges.
- Compute the settling time of a high-pass filter after a step and decide whether the corner is compatible with the clinical requirement.
- Identify the stability, layout, and protection issues that turn a good schematic into a noisy board.
- Write a verification plan for an AFE that covers gain, bandwidth, CMRR, noise, offset, and saturation.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

An AFE is a chain of three functions: **protect, amplify, filter**. Around them sits a fourth function, **reference**, which defines the common-mode operating point. The functions are not independent. Protection adds series resistance, which adds noise and forms a filter with input capacitance. Gain reduces the relative contribution of later stages but reduces the input range. Filtering removes interference but adds delay and can destabilise a feedback loop.

```text
        PROTECT                 AMPLIFY                  FILTER                SAMPLE
 patient --> R_series --> [clamp] --> [in-amp / PGA] --> [high-pass] --> [low-pass] --> ADC
   |             |             |              |                  |               |
   |             |             |              |                  |               |
   +--> leakage  +--> noise    +--> leakage   +--> noise         +--> group      +--> alias
        limit         and           and           and                delay
                      capacitance   capacitance   offset
                                      |
                                      +--> REFERENCE / RLD / GUARD
```

The model has three quantities to track at every node:

| Quantity | At the input | At the amplifier output | At the converter |
|---|---|---|---|
| Signal | 0.5–5 mV | Gain × signal | Codes |
| Offset | ±300 mV electrode | Gain × residual offset | Must not saturate |
| Common mode | 1 V or more | Rejected by CMRR | Residual differential |
| Noise | Source plus amplifier | Amplifier plus source × gain | Quantisation plus noise |
| Bandwidth | 0.5–150 Hz | Filter corners | Nyquist |

The central equation of the AFE is the noise budget:

\[
e_{\text{in}} = \sqrt{e_s^2 + e_n^2 + (i_n Z_s)^2 + e_R^2}
\]

where \(e_s\) is the source noise, \(e_n\) the amplifier voltage noise, \(i_n\) the amplifier current noise, \(Z_s\) the source impedance, and \(e_R\) the noise of the gain-setting and protection resistors. Every term is referred to the amplifier input and integrated over the same noise bandwidth.

The second central equation is the common-mode conversion:

\[
V_{\text{diff,residual}} \approx V_{\text{cm}} \left(\frac{1}{\text{CMRR}_{\text{amp}}} + \frac{\Delta Z}{Z_{\text{in,cm}}}\right)
\]

The first term is the amplifier's own rejection; the second is the conversion caused by source-impedance mismatch. In a biopotential AFE, the second term usually dominates. This is why improving the amplifier's CMRR from 100 dB to 120 dB often changes nothing, while matching the electrodes changes everything.

### 3.1 The three AFE archetypes

| Archetype | Topology | Best for | Weakness |
|---|---|---|---|
| Integrated biopotential AFE | In-amp + PGA + delta-sigma + RLD on one die | ECG, EEG, EMG, wearable | Fixed gain and filter options; register-configured |
| Discrete instrumentation amplifier | Three-op-amp or two-op-amp in-amp + external filter | Custom bandwidth, high CMRR, flexible gain | Layout-sensitive; more components; more noise |
| Difference amplifier | One op-amp and four resistors | Low-cost, low-channel-count, moderate CMRR | Resistor mismatch limits CMRR to about 40 dB at 1% |
| Chopper / zero-drift | Modulated amplifier | Low offset, low flicker noise, DC measurements | Charge injection; clock artefacts; higher complexity |
| Isolation amplifier | Modulated barrier | Patient isolation, high common mode | Propagation delay; power; cost |

The archetype determines the failure modes. An integrated AFE fails through register configuration and internal filter assumptions. A discrete in-amp fails through layout, resistor matching, and stability. A difference amplifier fails through resistor tolerance. A chopper fails through clock feedthrough. An isolation amplifier fails through delay and power.

### 3.2 The gain–range–noise triangle

Gain cannot be chosen independently of range and noise. For a fixed converter full scale \(V_{\text{FS}}\), the input-referred quantisation step is:

\[
\text{LSB}_{\text{in}} = \frac{V_{\text{FS}}}{G \times 2^N}
\]

Increasing \(G\) reduces the input-referred LSB but also reduces the maximum input:

\[
V_{\text{in,max}} = \frac{V_{\text{FS}}}{G}
\]

The maximum input must exceed the signal plus the offset plus the interference headroom. For a 300 mV electrode offset and a 2.4 V converter range, the maximum gain is 8 if the offset is not removed. With a DC servo or AC coupling, the offset is removed and the gain can be much higher, but the servo adds noise and settling time.

The triangle has no free corner. A design that increases gain to improve resolution must either reduce the offset, reduce the signal range, or accept a lower saturation threshold. The choice is a clinical one: which failure is worse, a noisy measurement or a saturated one?

### 3.3 What the AFE does not fix

An AFE cannot recover information that the sensor did not produce. It cannot remove an interference that is already differential. It cannot correct a nonlinear sensor. It cannot compensate for a changing electrode impedance unless it measures it. It cannot make a saturated input unsaturate. The AFE's job is to preserve the signal and reject what can be rejected; the rest is the sensor's or the algorithm's problem.

A common mistake is to specify a better AFE when the problem is the electrode. A 120 dB CMRR amplifier does not help if the electrode mismatch converts 1 V of common mode into 100 µV of differential signal. The AFE sees that 100 µV as signal; no amount of amplifier rejection removes it.

### 3.4 The ten-line AFE review

A compact review of any AFE can be written in ten lines:

1. Source impedance at the lowest and highest frequency.
2. Worst-case differential and common-mode input.
3. Protection network and its noise and bandwidth cost.
4. Gain and offset-removal scheme.
5. High-pass corner and settling time.
6. Low-pass corner and attenuation at Nyquist.
7. Effective CMRR with the actual mismatch.
8. Input-referred noise and the dominant term.
9. Phase margin with the actual load.
10. Leakage path and applied-part classification.

If any line is blank, the review is incomplete. The lines are deliberately quantitative; a review that fills them with adjectives has not answered them.

### 3.5 The two failure questions

For every AFE, ask two questions that the block diagram cannot answer.

**What happens when the input exceeds the range?** A well-designed AFE saturates gracefully and reports the saturation; a poorly designed one saturates silently or, worse, produces a plausible waveform from the remaining signal. The answer determines whether the data can be trusted at the edges.

**What happens when the common-mode voltage changes?** A well-designed AFE keeps the input inside its common-mode range and rejects the change; a poorly designed one loses gain, oscillates, or injects current into the patient. The answer determines whether the device is safe and usable in a moving patient.

The questions are not rhetorical. Each has a test, and the test should be in the verification plan.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| AFE | Analog front end; the conditioning between sensor and converter | Used to mean only the amplifier | None |
| Instrumentation amplifier (in-amp) | Differential amplifier with high input impedance and set gain | Confused with an operational amplifier | Gain V/V |
| Operational amplifier (op-amp) | High-gain differential voltage amplifier | Used as a complete in-amp | Gain-bandwidth product Hz |
| Common-mode voltage | Average of the two inputs relative to the reference | Assumed to be zero | V |
| Differential voltage | Difference between the two inputs | Confused with the signal of interest | V |
| CMRR | Common-mode rejection ratio | Quoted without test conditions | dB |
| Input impedance | Impedance between each input and the reference, and between inputs | Confused with source impedance | Ω, often with capacitance |
| Source impedance | Impedance of the sensor and cable | Assumed resistive | Ω |
| Input bias current | Current flowing into the amplifier inputs | Ignored with high source impedance | A |
| Input offset voltage | Differential voltage needed to zero the output | Confused with sensor offset | V |
| Offset drift | Change in offset with temperature | Ignored in budget | V/°C |
| Voltage noise density | Input-referred voltage noise per root hertz | Quoted without a frequency | nV/√Hz |
| Current noise density | Input-referred current noise per root hertz | Ignored with high source impedance | pA/√Hz |
| Flicker noise | 1/f noise at low frequency | Confused with drift | nV/√Hz at a frequency |
| Gain bandwidth | Frequency at which the open-loop gain falls to unity | Confused with closed-loop bandwidth | Hz |
| Slew rate | Maximum rate of change of the output | Ignored for small signals | V/µs |
| Settling time | Time to reach a stated accuracy after a step | Confused with rise time | s |
| Stability | Freedom from oscillation under the actual load | Assumed from the datasheet | Phase margin degrees |
| Phase margin | Margin from 180° at the unity-gain crossing | Ignored in filter design | degrees |
| PSRR | Power-supply rejection ratio | Confused with CMRR | dB |
| RLD | Driven-right-leg circuit; drives the patient reference | Assumed to eliminate interference | V, Hz |
| Guard | A conductor driven to the common-mode voltage to reduce leakage | Confused with a shield | V |
| DC servo | Feedback loop that removes a DC offset | Confused with AC coupling | Hz, settling time |
| AC coupling | High-pass filter that removes DC | Assumed to preserve slow signals | Hz |
| Anti-alias filter | Low-pass filter before sampling | Placed after the converter | Hz, order |
| Headroom | Margin between signal and rail | Ignored in gain selection | V or % |
| Clamp | Protection device that limits voltage | Assumed to be transparent | V, capacitance |
| Leakage current | Current through the patient to earth | Confused with bias current | µA |
| MOPP | Means of patient protection | Treated as a single part | Insulation, impedance |
| Chopper | Amplifier that modulates to reduce offset and flicker noise | Assumed to be noise-free | Hz, charge injection |

## 5. The mechanics, step by step

The procedure below designs or reviews an AFE from the source to the converter. Each step states its input, output, and rationale. Branch points and edge cases are identified, and Section 5.13 maps the required edge cases to the steps.

### Step 1. Characterise the source

**Input:** the sensor, the electrode or probe, and the cable.

**Output:** source amplitude, source impedance magnitude and phase, source offset, source noise, and the common-mode voltage relative to earth.

**Rationale:** the source determines the achievable noise floor and the required input impedance. A 10 kΩ electrode at 310 K produces about 13 nV/√Hz of thermal noise; an amplifier with 8 nV/√Hz adds little, while one with 50 nV/√Hz dominates. A 1 MΩ dry electrode produces 128 nV/√Hz and converts a 1 pA/√Hz current noise into 1 µV/√Hz, which is why high-impedance sources need low-current-noise amplifiers.

Measure the source impedance at the frequencies of interest, not only at DC. An electrode's impedance falls with frequency because of its double-layer capacitance; a cable's impedance rises with frequency because of its inductance. A source that looks like 10 kΩ at 1 Hz may look like 1 kΩ at 1 kHz.

### Step 2. Choose the topology

**Input:** the source characterisation and the requirement.

**Output:** a topology, a package, and a justification.

**Rationale:** the topology follows from the source and the environment. A high-impedance biopotential source needs an in-amp with high input impedance and low current noise. A low-impedance bridge needs a difference amplifier or an in-amp with low voltage noise. A high-common-mode environment needs an isolation amplifier or a driven guard. A low-offset DC measurement needs a chopper or a zero-drift amplifier.

**Branch point:** if the source impedance is above about 100 kΩ and the bandwidth is below 100 Hz, current noise usually dominates; choose a CMOS or JFET input. If the source impedance is below 1 kΩ and the bandwidth is above 1 kHz, voltage noise usually dominates; choose a bipolar input. If both are high, a chopper or a transformer may be the only option.

### Step 3. Protect the input

**Input:** the fault conditions and the leakage limit.

**Output:** a protection network and a verification plan.

**Rationale:** protection components are in the signal path. A series resistor limits current into the clamp but adds thermal noise and forms a low-pass filter with the input capacitance. A transient-voltage suppressor has capacitance and leakage. A clamp diode has reverse leakage and junction capacitance. The protection must be designed with the noise budget, not after it.

For a defibrillation-protected ECG input, the series resistors may be 10–100 kΩ, which is comparable to the electrode impedance. Their thermal noise is 13–40 nV/√Hz, which can dominate a low-noise amplifier. The protection resistor is therefore a noise source, not a transparent element.

### Step 4. Set the input common-mode operating point

**Input:** the amplifier's input common-mode range and the source's common-mode voltage.

**Output:** a bias network, a reference, and a common-mode control scheme.

**Rationale:** an in-amp has a finite input common-mode range. If the patient's common-mode voltage exceeds it, the amplifier saturates or its CMRR collapses. The bias network defines the operating point, and the driven-right-leg circuit actively pulls the patient to a safe, centred voltage.

The bias network also provides a path for the input bias current. Without it, the input floats and the output drifts to a rail. The bias resistors must be large enough not to load the source and small enough to sink the bias current. A 10 MΩ bias resistor with a 1 nA bias current produces a 10 mV offset; with a 1 pA bias current, 10 µV.

### Step 5. Allocate gain and offset

**Input:** the signal range, the offset, the converter range, and the noise budget.

**Output:** a gain, an offset-removal scheme, and a headroom calculation.

**Rationale:** the gain must make the signal large relative to the converter noise without saturating on the offset. The headroom equation is:

\[
G \times (V_{\text{signal,max}} + V_{\text{offset,residual}} + V_{\text{cm,residual}}) \le V_{\text{rail}} - V_{\text{dropout}}
\]

If the offset is removed by AC coupling, the high-pass corner must be low enough to preserve the clinical signal. For diagnostic ECG, the corner is 0.05 Hz; for monitoring, 0.5 Hz; for heart-rate-only, 1 Hz or higher. Each choice removes information.

**Decision table: offset removal**

| Offset | Bandwidth need | Option | Cost |
|---|---|---|---|
| < 10 mV | Any | DC coupling | None |
| 10–300 mV | DC to 0.05 Hz | DC servo | Noise, settling, complexity |
| 10–300 mV | 0.5 Hz and above | AC coupling | Loss of slow signal; step recovery |
| > 300 mV | Any | Coarse offset DAC plus servo | Calibration, complexity |
| Intermittent | Any | Saturation detection plus range switching | Data gaps; switching transients |

### Step 6. Set the high-pass corner

**Input:** the lowest clinical frequency and the offset step magnitude.

**Output:** a corner frequency, a filter order, and a settling-time estimate.

**Rationale:** the high-pass corner determines what slow information survives and how quickly the AFE recovers from a step. A first-order high-pass at \(f_c\) has a time constant \(\tau = 1/(2\pi f_c)\). At 0.05 Hz, \(\tau = 3.18\) s, and the settling to 1% takes about 4.6τ = 14.6 s. At 0.5 Hz, \(\tau = 0.318\) s, and the 1% settling is 1.46 s.

The step can be an electrode movement, a defibrillation, or a gain change. If the clinical requirement is to detect a rhythm within 5 s of a shock, a 0.05 Hz corner with a 14.6 s settling is incompatible. The filter choice is a clinical decision, not only an engineering one.

### Step 7. Set the low-pass and anti-alias filter

**Input:** the highest clinical frequency, the sampling rate, and the interference environment.

**Output:** a corner, an order, and an attenuation at Nyquist.

**Rationale:** the low-pass filter must pass the clinical band and attenuate the frequencies that would alias. For a 150 Hz clinical band and a 500 Hz sampling rate, the Nyquist frequency is 250 Hz. A fourth-order Butterworth at 150 Hz gives about 24 dB of attenuation at 250 Hz, which may be enough for a 1 V interferer but not for a 10 V one. The required attenuation depends on the largest out-of-band signal and the acceptable alias level.

The filter also adds group delay. A fourth-order Butterworth at 150 Hz has a group delay of about 2.6 ms at low frequency; a steeper elliptic filter has more. The delay is part of the latency budget.

### Step 8. Design the common-mode control

**Input:** the measured common-mode voltage, the CMRR requirement, and the safety limit.

**Output:** a driven-right-leg or guard circuit with a loop gain and a stability analysis.

**Rationale:** the driven-right-leg circuit senses the common-mode voltage and drives it back through a resistor. The loop gain determines the reduction: a loop gain of 40 dB reduces the common-mode voltage by a factor of 100. The loop must be stable with the actual electrode impedances and cable capacitance; a poorly compensated loop oscillates.

The safety limit constrains the current that can be injected into the patient. A driven-right-leg resistor of 1 MΩ with a 1 V drive produces 1 µA, which is well below the 10 µA type CF limit but may be too small to control the common mode in a high-impedance system. The design trade is between control authority and patient current.

### Step 9. Compute the noise budget

**Input:** the source, the amplifier, the resistors, and the bandwidth.

**Output:** input-referred noise, SNR, and ENOB.

**Rationale:** the budget identifies the dominant term. For a biopotential AFE, the source thermal noise, the amplifier voltage noise, the current noise into the source, and the protection resistor noise are the usual terms. The reference and supply noise enter through the amplifier's PSRR and the converter's reference.

Use maximum values for a worst-case design and typical values for a nominal estimate. Report both. A design that meets the specification only with typical values is not a design; it is a hope.

### Step 10. Check stability and drive capability

**Input:** the amplifier's open-loop output impedance, the load, and the feedback network.

**Output:** a phase margin and a compensation network, or a documented decision to use a different amplifier.

**Rationale:** an operational amplifier driving a capacitive load can oscillate. A filter with a high Q can peak and ring. A chopper can produce clock feedthrough. The stability analysis must use the actual load, including the cable and the converter's input capacitance, not the datasheet's test load.

A phase margin below 45° produces visible overshoot and ringing; below 30°, the circuit may oscillate. The fix is a series isolation resistor, a feedback capacitor, or a different amplifier. Each fix changes the bandwidth or the noise.

### Step 11. Layout and ground

**Input:** the schematic, the board stack-up, and the mechanical enclosure.

**Output:** a layout with separated analog and digital returns, guarded inputs, and a defined shield connection.

**Rationale:** an AFE is a circuit, not a schematic. A 10 mm trace can add inductance; a shared ground can couple digital current into the analog reference; a shield connected at both ends can create a ground loop. The layout must be designed with the same care as the schematic.

The most common layout failure is a ground loop between the patient, the cable shield, and the mains earth. The loop injects current into the patient and adds interference. The isolation barrier breaks the loop; the shield should be connected at one end, usually the device end, unless the cable is designed for both-end connection with a chassis ground.

### Step 12. Verify and document

**Input:** the prototype and the requirement.

**Output:** a verification matrix, a noise measurement, and a configuration record.

**Rationale:** verification must cover the edges, not only the nominal point. Measure gain at 0.5 Hz, 10 Hz, and the high corner; measure CMRR at 50/60 Hz with matched and mismatched source impedances; measure noise with the input shorted and with a 10 kΩ source; measure settling after a 300 mV step; measure the supply rejection at the switching frequency.

Document the gain, the filter corners, the reference, the RLD configuration, and the firmware settings. Without the record, a later change cannot be distinguished from a patient change.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 5, 7 | Gain times offset exceeds the rail | Code histogram; flat-topped waveform | Loss of signal; false reading |
| Common-mode rejection failure | 2, 4, 8 | Source mismatch or RLD failure | Inject common mode; measure output | Interference mistaken for signal |
| Electrode offset | 1, 5 | Half-cell potential | Open-circuit voltage; range check | Saturation; gain compromise |
| Clock jitter | 7, 10 | Sampling aperture variation | SNR versus input frequency | Degraded high-frequency content |
| Brown-out | 10, 12 | Supply falls below minimum | Supply monitor; reset flags | Invalid samples; lost calibration |
| Watchdog reset loop | 12 | Firmware fails to service watchdog | Reset counter | Data gaps |
| Thermal throttling | 9, 10 | Power dissipation raises temperature | Temperature log; offset drift | Drift; reduced bandwidth |
| Firmware update failure | 12 | Configuration registers lost or changed | Version and register readback | Wrong gain or filter |
| Calibration drift | 9, 12 | Ageing, temperature, sterilization | Reference check; trend | Systematic bias |

### 5.14 Stability checklist

| Check | Criterion | If it fails |
|---|---|---|
| Phase margin with the actual load | ≥ 45° | Add isolation resistor or compensation |
| Capacitive load | Below the datasheet's maximum, or isolated | Series resistor; buffer |
| Filter Q | Below 0.707 for Butterworth, or damped | Change topology; add damping |
| RLD loop phase margin | ≥ 45° with electrode impedance | Reduce loop gain; add compensation |
| Supply decoupling | Low impedance at the switching frequency | Add capacitor; change layout |
| Input bias path | DC path exists for both inputs | Add bias resistors |
| Input common-mode range | Covers the worst-case patient voltage | Change reference or topology |
| Output swing | Covers the converter's full scale | Reduce gain or use a rail-to-rail amplifier |
| Protection clamp leakage | Below the leakage budget | Change clamp; add guard |
| Thermal rise | Below the specified maximum | Reduce power; add copper |

The checklist is not exhaustive, but it covers the failures that appear most often in design reviews. A review that signs off on the schematic without the load, the layout, and the supply is not a stability review.

### 5.15 Noise sources in detail

An AFE noise budget is only as good as its inventory of noise sources. The table below lists the sources that matter in a biopotential or bridge AFE.

| Source | Formula or density | Typical value | Mitigation |
|---|---|---|---|
| Resistor thermal noise | √(4kTR) | 12.8 nV/√Hz at 10 kΩ, 128 nV/√Hz at 1 MΩ | Lower resistance; cool; limit bandwidth |
| Amplifier voltage noise | Datasheet e_n | 8 nV/√Hz (AD8422), 100 nV/√Hz (AD8232 at 1 kHz) | Choose a low-noise amplifier; raise gain early |
| Amplifier current noise | Datasheet i_n | 0.09 pA/√Hz (AD8422), 1 pA/√Hz (typical CMOS) | Lower source impedance; choose JFET/CMOS |
| Flicker noise | e_n at a stated frequency, 1/f | 10–100 nV/√Hz at 10 Hz | Chop; avoid DC; high-pass |
| Shot noise | √(2qIB) | pA to nA in photodiodes | More light; shorter integration |
| Reference noise | Datasheet | 1–10 µV RMS | Filter; decouple; ratiometric design |
| Supply noise | PSRR × ripple | 20 µV from a 2 MHz switcher | LDO; filter; layout |
| Interference | Coupled voltage or current | mV to V | Shield; RLD; filter; layout |
| Quantisation | LSB/√12 | µV to nV | More resolution or gain |

The relative importance changes with source impedance. At 10 kΩ, a 8 nV/√Hz amplifier and a 0.09 pA/√Hz current noise contribute 8 nV/√Hz and 0.9 nV/√Hz respectively, so voltage noise dominates. At 1 MΩ, the same current noise contributes 90 nV/√Hz, which is eleven times the voltage noise. The amplifier that is best at 10 kΩ is not the best at 1 MΩ.

The noise gain is not always the signal gain. A non-inverting amplifier has a noise gain of \(1 + R_f/R_g\), which equals the signal gain. A difference amplifier or an in-amp can have a noise gain that differs from the differential gain, especially at high frequency where the input capacitance and the source impedance form a pole. The noise gain peaking at high frequency can dominate the output noise even when the signal band is low.

The reference is part of the noise budget. A 2.5 V reference with 10 µV RMS of noise, divided by a gain of 100, contributes 0.1 µV at the input, which may be negligible; the same reference with 100 µV RMS contributes 1 µV, which is not. The reference's low-frequency noise and its thermal drift both matter.

### 5.16 Common-mode rejection in a real system

The amplifier's CMRR is measured with a defined source imbalance, usually 1 kΩ. In a real system, the imbalance is the electrode mismatch, which varies with contact, motion, and time. The effective CMRR is approximately:

\[
\text{CMRR}_{\text{eff}} = 20 \log_{10}\left(\frac{1}{\frac{1}{\text{CMRR}_{\text{amp}}} + \frac{\Delta Z}{Z_{\text{in,cm}}}}\right)
\]

For an amplifier with 100 dB CMRR and a 10 MΩ common-mode input impedance, the mismatch term dominates when \(\Delta Z / Z_{\text{in,cm}} > 10^{-5}\), or \(\Delta Z > 100\ \Omega\). Most electrode pairs exceed that.

| Mismatch ΔZ | Z_in,cm = 10 MΩ | Z_in,cm = 1 GΩ |
|---:|---:|---:|
| 10 Ω | 120 dB | 160 dB |
| 100 Ω | 100 dB | 140 dB |
| 1 kΩ | 80 dB | 120 dB |
| 10 kΩ | 60 dB | 100 dB |

The table shows why input impedance matters more than the amplifier's CMRR specification in a high-impedance system. An amplifier with 120 dB CMRR but a 10 MΩ input impedance gives 80 dB effective CMRR with a 1 kΩ mismatch. An amplifier with 100 dB CMRR and a 1 GΩ input impedance gives 120 dB effective CMRR with the same mismatch. The second amplifier is better for the application even though its datasheet number is lower.

The driven-right-leg loop changes the common-mode voltage, not the CMRR. If the loop reduces the common mode by 40 dB, the residual interference falls by 40 dB, regardless of the amplifier's CMRR. A design with a modest CMRR and a good RLD can outperform a design with an excellent CMRR and no RLD.

Cable shielding is part of the common-mode path. A shield connected at both ends creates a ground loop; connected at one end, it reduces electric-field coupling but not magnetic coupling. A driven shield, held at the common-mode voltage, reduces the cable capacitance seen by the source and the leakage current. The shield connection is a design decision with safety implications, not a cable accessory.

### 5.17 Filter realisations and their trade-offs

The filter topology determines the noise, the delay, the component sensitivity, and the stability.

| Topology | Order | Q control | Noise | Delay | Best for |
|---|---|---:|---|---|---|
| Passive RC | 1 | None | Thermal noise of R | Low | Anti-alias pre-filter; low cost |
| Sallen-Key | 2 | Component ratio | Amplifier noise | Moderate | General low-pass; easy |
| Multiple feedback | 2 | Component ratio | Higher | Moderate | Narrowband; inverting |
| State-variable | 2 | Independent | Higher | Moderate | Tunable corner and Q |
| Switched-capacitor | 2+ | Clock ratio | Aliasing of clock | High | Programmable corner; integrated |
| Digital FIR | Any | Designed | Quantisation | (N−1)/(2fs) | Linear phase; precise |
| Digital IIR | Any | Designed | Quantisation | Frequency-dependent | Efficient sharp filters |

A Sallen-Key low-pass with a high Q peaks near the corner, which can amplify interference at the corner frequency. A Butterworth response (Q = 0.707) is maximally flat and is the usual choice for a measurement filter. A Bessel response has a more constant group delay and is preferable when the waveform shape matters. A Chebyshev response has a steeper roll-off but ripples in the passband.

The filter's components have tolerances. A Sallen-Key with 5% resistors and 10% capacitors has a corner that varies by more than 10% and a Q that can peak or overdamp. For a production device, either use 1% components or trim the filter in test. A filter that is correct on the bench and wrong in production is a common source of unit-to-unit data differences.

### 5.18 Power, thermal, and supply rejection

The AFE's supply is not ideal. A switching regulator at 2 MHz with 20 mV of ripple, feeding an amplifier with 60 dB PSRR at 2 MHz, produces 20 µV of input-referred noise. A low-dropout regulator with 40 dB PSRR at 2 MHz reduces the ripple to 0.2 mV, and the amplifier's PSRR reduces it further. The cascade of regulators and filters is part of the noise budget.

The thermal design matters for two reasons. First, the amplifier's offset and bias current drift with temperature; a 0.3 µV/°C offset drift over a 20 °C rise is 6 µV, comparable to the noise. Second, the patient-applied part has a temperature limit; the AFE's power dissipation contributes to the applied-part temperature. A 170 µA AFE at 3 V dissipates 0.51 mW, which is negligible; a 10 mA AFE at 5 V dissipates 50 mW, which is not.

The supply sequence matters. An amplifier powered before its reference can drive its output to a rail and latch; a converter powered before its reference can produce invalid codes. The power management integrated circuit (PMIC) should sequence the rails and hold the converter in reset until the references are stable. A brown-out during a firmware update can leave the configuration registers in an unknown state, which is why the update must be atomic and verified.

### 5.19 Protection arithmetic

| Protection element | Typical value | Noise or error contribution | Check |
|---|---|---|---|
| Series resistor | 10–100 kΩ | Thermal noise 13–40 nV/√Hz | Compare with amplifier noise |
| Clamp diode | 0.1–10 nA leakage | Offset across source impedance | Leakage × Z_s < offset budget |
| TVS diode | 10–100 pF | Forms a pole with R_series | Pole above the signal band |
| Capacitor to ground | 1–10 nF | Reduces bandwidth; increases settling | Corner above the clinical band |
| Isolation barrier | 1–5 kV | Propagation delay; leakage | Delay in latency budget; leakage in safety budget |

A 100 kΩ series resistor with a 10 nF capacitor forms a pole at 159 Hz, which is inside a 150 Hz ECG band. The protection network has just changed the signal. A 10 kΩ resistor and a 1 nF capacitor form a pole at 15.9 kHz, which is safely above the band but still limits the response to a defibrillation transient. The protection design and the signal design are the same design.

### 5.20 Input impedance and bias current

| Source impedance | 1 pA bias current offset | 100 pA bias current offset | 1 nA bias current offset |
|---:|---:|---:|---:|
| 10 kΩ | 10 nV | 1 µV | 10 µV |
| 100 kΩ | 100 nV | 10 µV | 100 µV |
| 1 MΩ | 1 µV | 100 µV | 1 mV |
| 10 MΩ | 10 µV | 1 mV | 10 mV |

The table is a reminder that bias current becomes an offset through the source impedance. A 1 nA bias current into a 1 MΩ dry electrode produces 1 mV, which is comparable to an ECG signal. The bias current must be in the offset budget, not only in the noise budget.

### 5.21 Gain error budget

| Source of gain error | Typical value | Effect on a 1 mV signal | Calibration option |
|---|---:|---:|---|
| Resistor tolerance | 0.1% | 1 µV | Factory calibration |
| Amplifier gain error | 0.04–3.5% | 0.4–35 µV | Factory calibration |
| Gain drift | 1–12 ppm/°C | 0.012–0.14 µV/°C | Temperature compensation |
| Reference tolerance | 0.05–1% | 0.5–10 µV | Ratiometric or calibrated |
| Source loading | Depends on Z_in | Depends | Buffer; higher Z_in |
| Temperature coefficient of source | Sensor-specific | Sensor-specific | Sensor compensation |

The gain error is usually calibrated at manufacture, but the drift is not. A 12 ppm/°C gain drift over a 20 °C change is 240 ppm, or 0.024%, which is 0.24 µV on a 1 mV signal. That is small compared with the 2.13 µV noise, but it is systematic, so it does not average away.

### 5.22 Ten design review questions for an AFE

1. What is the source impedance at the lowest and highest frequencies of interest?
2. What is the worst-case differential and common-mode input, including fault conditions?
3. What is the gain, and where does the offset go before the gain stage?
4. What is the high-pass corner, and what is the settling time after a 300 mV step?
5. What is the low-pass corner, and what is the attenuation at Nyquist?
6. What is the effective CMRR with the actual electrode mismatch, not the datasheet's test condition?
7. What is the input-referred noise budget, and which term dominates?
8. What is the phase margin with the actual cable and converter load?
9. What is the leakage current path, and how is it limited?
10. What is recorded so that a later change in gain or filter can be detected?

A review that cannot answer all ten has not reviewed the AFE; it has reviewed the block diagram.

### 5.23 Chopper and zero-drift artefacts

A chopper-stabilised amplifier reduces offset and flicker noise by modulating the input to a higher frequency, amplifying, and demodulating. The technique is effective for low-frequency, high-impedance sources, but it introduces artefacts:

- **Clock feedthrough** at the chopping frequency, which appears as a tone in the output.
- **Charge injection** at the input switches, which produces a small current pulse and an offset that depends on the source impedance.
- **Intermodulation** between the chopping frequency and an interference tone, which can fold into the signal band.
- **Settling** after a step, which is limited by the chopper's filter.

A chopper is not a free improvement. It trades offset and flicker noise for clock artefacts and complexity. The choice depends on whether the dominant error is offset drift or broadband noise.

### 5.24 Measuring AFE noise correctly

A noise measurement is easy to do badly. The procedure below gives a result that can be compared with the budget.

1. Terminate the input with the actual source impedance, not a short, unless the specification is input-referred voltage noise.
2. Shield the measurement from ambient interference; use a battery-powered supply if possible.
3. Set the gain and filter to the production configuration.
4. Record at least 10 s of data at the production sampling rate.
5. Remove the mean and compute the RMS in the signal band, not the full band.
6. Compute the noise spectral density and check for 1/f, mains, and chopper tones.
7. Repeat at three temperatures and at the minimum and maximum supply.
8. Compare with the budget and identify the dominant term.

A measurement with a shorted input gives the amplifier's voltage noise but hides the current noise and the source thermal noise. A measurement with a 10 kΩ resistor gives the sum but includes the resistor's own noise, which must be subtracted in power, not amplitude. A measurement in a 1 MHz band includes interference that the production filter removes. State the conditions with the result.

## 6. Worked example

### 6.1 The requirement

A single-lead wearable ECG uses an integrated heart-rate AFE with the following datasheet values:

- Fixed gain: 100 V/V.
- Input-referred noise: 14 µV peak-to-peak over 0.5–40 Hz (typical).
- CMRR: 80 dB minimum, 86 dB typical, from DC to 60 Hz.
- Supply current: 170 µA at 3.0 V.
- Input bias current: 1 pA typical.
- High-pass corner: set by an external capacitor to 0.5 Hz.
- Low-pass corner: set by an external capacitor to 40 Hz.
- Driven-right-leg amplifier: integrated, with a 1 MΩ series resistor.

The source is a pair of Ag/AgCl electrodes with 10 kΩ each, matched to within 100 Ω. The common-mode voltage before the RLD is 1.0 V RMS at 50 Hz. The AFE drives the right leg with a loop gain that reduces the common-mode voltage by a factor of 100 at 50 Hz.

### 6.2 The noise budget

The noise bandwidth of the 0.5–40 Hz band is approximately the low-pass noise bandwidth minus the high-pass contribution. For a first-order high-pass at 0.5 Hz and a first-order low-pass at 40 Hz, the effective noise bandwidth is close to 1.57 × 40 = 62.8 Hz, with a small correction for the high-pass. Use 62.8 Hz.

The source thermal noise for two 10 kΩ electrodes is:

\[
e_s = \sqrt{4 k T \times 20\,000} = 18.5\ \text{nV}/\sqrt{\text{Hz}}
\]

Over 62.8 Hz:

\[
E_s = 18.5 \times \sqrt{62.8} = 147\ \text{nV RMS}
\]

The amplifier's 14 µV peak-to-peak noise over 0.5–40 Hz converts to RMS with a crest factor of 6.6:

\[
E_n = \frac{14}{6.6} = 2.12\ \text{µV RMS}
\]

The amplifier's current noise is not specified in the summary; assume 1 pA/√Hz, which into 10 kΩ gives 10 nV/√Hz, or 79 nV RMS over 62.8 Hz. The protection resistors are 10 kΩ each, adding the same 147 nV RMS as the electrodes.

The total input-referred noise is:

\[
E_{\text{total}} = \sqrt{0.147^2 + 2.12^2 + 0.079^2 + 0.147^2} = 2.14\ \text{µV RMS}
\]

The amplifier dominates: 98% of the noise power. Improving the electrodes would not help; the AFE's own noise is the limit.

For a 1.0 mV peak-to-peak ECG, the RMS signal is 0.354 mV, and the SNR is:

\[
\text{SNR} = 20 \log_{10}\left(\frac{354\ \text{µV}}{2.14\ \text{µV}}\right) = 44.4\ \text{dB}
\]

The ENOB is:

\[
\text{ENOB} = \frac{44.4 - 1.76}{6.02} = 7.1\ \text{bits}
\]

A 24-bit converter behind this AFE would deliver about 7 effective bits for a single beat. Averaging over many beats improves the estimate of a stable feature, but not the resolution of a changing waveform.

### 6.3 The common-mode budget

The amplifier's CMRR of 80 dB means that a 1 V common-mode voltage produces:

\[
V_{\text{cm,amp}} = \frac{1\ \text{V}}{10^{80/20}} = 100\ \text{µV}
\]

at the input, or 10 mV at the output with a gain of 100. That is ten times the 1 mV ECG signal. Without common-mode control, the AFE would be unusable.

The driven-right-leg loop reduces the common-mode voltage by a factor of 100, from 1.0 V to 10 mV. The residual at the input is:

\[
V_{\text{cm,residual}} = \frac{10\ \text{mV}}{10^{80/20}} = 1.0\ \text{µV}
\]

or 100 µV at the output, which is below the 2.13 µV input-referred noise. The RLD does not need to be perfect; it needs to reduce the common mode to the point where the amplifier's CMRR brings it below the noise floor.

The electrode mismatch of 100 Ω in 10 kΩ is 1%. The input impedance of the integrated AFE is high, so the mismatch conversion is small; assume a common-mode input impedance of 10 MΩ. The mismatch produces:

\[
V_{\text{diff,mismatch}} = V_{\text{cm}} \times \frac{100}{10\,000\,000} = 10\ \text{mV} \times 1 \times 10^{-5} = 0.1\ \text{µV}
\]

which is negligible compared with the 1.0 µV from the amplifier's CMRR. At the original 1 V common mode, the mismatch would produce 10 µV, which would dominate. The RLD reduces both terms.

### 6.4 Diagnostic 1: the RLD is disconnected

A service technician replaces the electrode cable and forgets to connect the right-leg lead. The common-mode voltage returns to 1.0 V. The amplifier's CMRR produces 100 µV at the input, and the mismatch produces 10 µV. The total interference is about 100 µV, which is 10% of the 1 mV signal and 47 times the 2.13 µV noise.

The ECG still looks plausible: the QRS is visible, but the baseline is thick with 50 Hz. A heart-rate algorithm may still detect beats, with occasional false counts. The failure is not a flat line; it is a degraded signal that passes a superficial check.

The diagnostic is to measure the common-mode voltage at the input, or to inject a known common-mode signal and measure the output. A quick check is to touch the patient's right leg: if the interference changes, the RLD is not working. A proper check measures the CMRR with the RLD enabled and disabled.

### 6.5 Diagnostic 2: the high-pass corner is too low for the settling requirement

The high-pass capacitor is changed from 0.5 Hz to 0.05 Hz to preserve the ST segment. The time constant becomes:

\[
\tau = \frac{1}{2\pi \times 0.05} = 3.18\ \text{s}
\]

After a 300 mV electrode step, the settling to 1% takes about 4.6τ = 14.6 s. A motion artefact or a defibrillation produces a step that takes 15 s to recover. During that time, the AFE output is saturated or offset, and the heart-rate algorithm sees a false flat line or a false rate.

The design flaw is not the corner itself; it is the combination of the corner with the clinical requirement for a short recovery. A diagnostic ECG may accept the 15 s settling because the clinician can wait; a wearable monitor that alarms on pause or fibrillation cannot. The requirement must state the recovery time, and the filter must be chosen to meet it.

### 6.6 Vary one assumption: the electrode mismatch rises to 1 kΩ

Change only the mismatch, from 100 Ω to 1 kΩ (10%). With the RLD active and a 10 mV common-mode voltage, the mismatch conversion is:

\[
V_{\text{diff,mismatch}} = 10\ \text{mV} \times \frac{1000}{10\,000\,000} = 1.0\ \text{µV}
\]

This is now equal to the amplifier's CMRR residual of 1.0 µV. The total common-mode interference is about 1.4 µV, still below the 2.13 µV noise, but no longer negligible. If the RLD fails and the common mode returns to 1 V, the mismatch term becomes 100 µV, which dominates the signal.

The consequence is that the design's interference margin depends on the electrode matching, not only on the RLD. A validation that uses well-matched electrodes in the lab will not reveal the sensitivity. The fix is to specify the electrode matching, measure it in production, or increase the input impedance.

### 6.7 Power, thermal, and battery

The AFE draws 170 µA at 3.0 V, or 0.51 mW. The microcontroller and radio add about 8.0 mW, for a total of 8.51 mW. The daily energy is:

\[
E_{	ext{day}} = 8.51 	imes 10^{-3} 	imes 86\,400 = 735\ 	ext{J/day}
\]

A 150 mAh cell at 3.7 V stores 1,998 J, giving a runtime of 2.72 days. The AFE is 6% of the power; the radio and microcontroller are 94%. A design that reduces the AFE current by half saves 0.26 mW and extends the runtime by 0.08 days. A design that reduces the radio duty cycle by half saves 4 mW and extends the runtime by 1.2 days. The engineering effort should follow the budget.

The thermal rise of the AFE is small. A 4 mm × 4 mm package with a junction-to-ambient thermal resistance of 50 °C/W and 0.51 mW dissipation rises 0.026 °C above ambient. The rise is negligible compared with the offset drift of 0.3 µV/°C, which over a 20 °C ambient change is 6 µV. The dominant thermal effect is the ambient change, not self-heating.

The battery voltage also changes the reference. A 3.7 V cell falls to 3.0 V at end of discharge; if the AFE's reference is derived from the battery, the gain changes by 19%. A ratiometric design compensates by using the same reference for the converter and the amplifier, but it does not compensate for the amplifier's own gain error. A separate low-noise reference, with its own dropout, is the safer choice for a measurement that claims accuracy across the battery life.

### 6.8 What the example demonstrates

The arithmetic closes: 2.13 µV RMS input-referred noise, 44.4 dB SNR, 7.1 ENOB, 1.0 µV CMRR residual with the RLD active, and 14.6 s settling at 0.05 Hz. The two diagnostics reveal different flaws: one is a common-mode control failure, the other is a filter-clinical mismatch. The sensitivity analysis shows that a tenfold change in electrode mismatch moves the interference from negligible to comparable with the noise.

The general method is the same as in the previous topics: write the budget, identify the dominant term, test the assumption behind it, and state the consequence. For an AFE, the dominant term is often not the one the datasheet highlights.

### 6.9 Verification matrix

| Test | Condition | Acceptance criterion | Margin to record |
|---|---|---|---|
| Gain | 1 mV at 10 Hz | 100 ± 1 V/V | Gain error in % |
| Bandwidth | Sweep 0.1–100 Hz | −3 dB at 0.5 and 40 Hz | Corner frequencies |
| CMRR | 1 V at 50 Hz, matched electrodes | > 80 dB effective | dB |
| CMRR with mismatch | 1 V at 50 Hz, 1 kΩ mismatch | > 60 dB effective | dB |
| Noise | Input shorted, 0.5–40 Hz | < 2.5 µV RMS | µV RMS |
| Offset | Zero input, 25 °C | < 10 µV input-referred | µV |
| Settling | 300 mV step, 0.5 Hz corner | < 2 s to 1% | s |
| Saturation | 400 mV offset | Flag asserted, no false output | Pass or fail |
| Supply rejection | 100 mV at 2 MHz | < 5 µV input-referred | µV |
| Leakage | 250 V AC, patient lead | < 10 µA type CF | µA |

The matrix is deliberately small. A verification plan with fifty tests and no margins is less useful than one with ten tests that state the margin. The regulator will ask for the evidence behind the claim; the margin is the evidence that the design is not on the edge.

### 6.10 Residual risk

The AFE design in this example leaves three residual risks. First, the 80 dB CMRR depends on the RLD; a cable failure removes the margin. Second, the 0.5 Hz high-pass removes the ST segment, so the device cannot be used for diagnostic ST analysis. Third, the 2.13 µV noise limits the ENOB to 7.1 bits, so a single-beat analysis cannot resolve features below about 6 µV. These risks are acceptable for a heart-rate monitor and unacceptable for a diagnostic electrocardiograph. The device's intended use determines whether the residual risk is acceptable, not the engineering team's preference.
## 7. Code, query, or calculation

The script reproduces the noise, common-mode, settling, and power results in Section 6. The unit conversions are explicit because mixing microvolts and volts is the most common arithmetic error in an AFE budget.

**Assumptions.** Two 10 kΩ electrodes in series, 310 K, a first-order 40 Hz low-pass noise bandwidth, an amplifier noise of 14 µV peak-to-peak converted with a crest factor of 6.6, a current noise of 1 pA/√Hz, a CMRR of 80 dB, a 10 MΩ common-mode input impedance, and a 150 mAh battery.

```python
import math

k = 1.380649e-23
T = 310.0
R_source = 20_000.0
BW = 1.5708 * 40.0

e_s_density = math.sqrt(4 * k * T * R_source)
e_s_uv = e_s_density * math.sqrt(BW) * 1e6
e_amp = 14.0 / 6.6
i_n = 1e-12
e_current_uv = i_n * 10_000 * math.sqrt(BW) * 1e6
e_prot_uv = e_s_uv
total = math.sqrt(e_s_uv**2 + e_amp**2 + e_current_uv**2 + e_prot_uv**2)
snr = 20 * math.log10(353.6 / total)
enob = (snr - 1.76) / 6.02

cmrr = 80.0
cm = 1.0
res_1v = cm / 10**(cmrr/20)
res_10mv = 0.010 / 10**(cmrr/20)
zin = 10e6
mismatch = 1000.0
res_mismatch = 0.010 * mismatch / zin

def settle(fc, fraction=0.01):
    tau = 1 / (2 * math.pi * fc)
    return -math.log(fraction) * tau

p_afe = 170e-6 * 3.0 * 1e3
p_total = p_afe + 8.0
j_day = p_total * 1e-3 * 86400
battery = 0.150 * 3.7 * 3600

print("NOISE BUDGET")
print(f"  source density        {e_s_density*1e9:7.2f} nV/sqrt(Hz)")
print(f"  noise bandwidth       {BW:7.2f} Hz")
print(f"  source RMS            {e_s_uv:7.3f} uV")
print(f"  amplifier RMS         {e_amp:7.3f} uV")
print(f"  current-noise RMS     {e_current_uv:7.3f} uV")
print(f"  protection RMS        {e_prot_uv:7.3f} uV")
print(f"  total RMS             {total:7.3f} uV")
print(f"  SNR (1 mVpp)          {snr:7.2f} dB")
print(f"  ENOB                  {enob:7.2f} bits")
print()
print("COMMON MODE")
print(f"  residual at 1 V       {res_1v*1e6:7.2f} uV")
print(f"  residual at 10 mV     {res_10mv*1e6:7.2f} uV")
print(f"  mismatch at 1 kOhm    {res_mismatch*1e6:7.2f} uV")
print()
print("SETTLING TO 1%")
for fc in (0.05, 0.5, 1.0):
    print(f"  corner {fc:4.2f} Hz: {settle(fc):7.2f} s")
print()
print("POWER")
print(f"  AFE power             {p_afe:7.3f} mW")
print(f"  total power           {p_total:7.3f} mW")
print(f"  energy per day        {j_day:7.1f} J")
print(f"  runtime (150 mAh)     {battery/j_day:7.2f} days")
```

**Expected output.**

```text
NOISE BUDGET
  source density          18.50 nV/sqrt(Hz)
  noise bandwidth         62.83 Hz
  source RMS              0.147 uV
  amplifier RMS           2.121 uV
  current-noise RMS       0.079 uV
  protection RMS          0.147 uV
  total RMS               2.133 uV
  SNR (1 mVpp)            44.39 dB
  ENOB                     7.08 bits

COMMON MODE
  residual at 1 V        100.00 uV
  residual at 10 mV        1.00 uV
  mismatch at 1 kOhm       1.00 uV

SETTLING TO 1%
  corner 0.05 Hz:   14.66 s
  corner 0.50 Hz:    1.47 s
  corner 1.00 Hz:    0.73 s

POWER
  AFE power               0.510 mW
  total power             8.510 mW
  energy per day          735.3 J
  runtime (150 mAh)        2.72 days
```

The output shows three things that are easy to miss in a hand calculation.

First, the total noise is 2.133 µV, and the amplifier contributes 2.121 µV, or 98.9% of the noise power. The source, the protection resistors, and the current noise together contribute 1.1%. Improving the electrode or the protection network would not change the SNR. The only way to improve the measurement is a quieter amplifier or a narrower bandwidth.

Second, the RLD reduces the common-mode voltage from 1 V to 10 mV, and the residual falls from 100 µV to 1 µV. The 1 µV residual is comparable to the 1 µV mismatch term at a 1 kΩ mismatch. The two terms are independent and add in root-sum-square, giving 1.4 µV, still below the 2.13 µV noise. The design has a common-mode margin of about 1.5:1, which is adequate but not generous.

Third, the high-pass corner dominates the settling time. At 0.05 Hz, the 1% settling is 14.66 s; at 0.5 Hz, it is 1.47 s. A clinical requirement for a 2 s recovery rules out the 0.05 Hz corner, regardless of how good the rest of the AFE is.

### 7.1 Effective CMRR versus mismatch

The following script prints the effective CMRR for an amplifier with 120 dB CMRR and a 10 MΩ common-mode input impedance, across four mismatch values. It shows why the amplifier specification is not the system specification.

```python
import math

cmrr_amp = 120.0
zin = 10e6
for dz in (10, 100, 1000, 10_000):
    term_amp = 1 / 10**(cmrr_amp / 20)
    term_mismatch = dz / zin
    effective = 20 * math.log10(1 / (term_amp + term_mismatch))
    print(f"mismatch {dz:6.0f} ohm: effective CMRR {effective:6.1f} dB")
```

**Expected output.**

```text
mismatch     10 ohm: effective CMRR  120.0 dB
mismatch    100 ohm: effective CMRR  100.0 dB
mismatch   1000 ohm: effective CMRR   80.0 dB
mismatch  10000 ohm: effective CMRR   60.0 dB
```

At a 1 kΩ mismatch, the system CMRR is 80 dB, not 120 dB. The amplifier's 120 dB contributes 0.0001% to the conversion, while the mismatch contributes 0.01%. Improving the amplifier further would change nothing.

**A note on the crest factor.** The 14 µV peak-to-peak noise is converted to RMS with a factor of 6.6. If the datasheet's interval is shorter, the crest factor is lower and the RMS is higher. For example, a 5.6 µV peak-to-peak value with a crest factor of 5.0 is 1.12 µV RMS, while the same value with a crest factor of 6.6 is 0.85 µV RMS. A factor-of-1.3 difference in noise is a 2.3 dB difference in SNR. Always state the convention.

## 8. Data, standards, and regulatory context

The AFE is constrained by particular standards that test its input behaviour, and by general standards that test its safety and electromagnetic immunity. The table separates the instruments and the clauses that matter most.

| Instrument | Scope | Status | What it constrains in the AFE |
|---|---|---|---|
| IEC 60601-2-25:2011 | Electrocardiographs | Particular standard | Input dynamic range ±5 mV, DC offset ±300 mV, frequency response 0.05–150 Hz, CMRR, input impedance |
| IEC 60601-2-27:2011 | ECG monitoring equipment | Particular standard | Dynamic range, offset tolerance, heart-rate accuracy, alarm performance |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General medical electrical equipment | Consensus standard; binding when referenced | Patient leakage current, applied-part classification, isolation, protection |
| IEC 60601-1-2:2014+AMD1:2020 | Electromagnetic disturbances | Collateral standard | Immunity to ESD, RF, magnetic fields, transients |
| IEC 60601-1-8:2006+AMD2:2020 | Alarm systems | Collateral standard | Alarm condition delay, signal generation delay, disclosure |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis for saturation, interference, leakage |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Firmware that configures gain and filter |
| ISO 80416-1 / IEC 60417 | Graphical symbols | Convention | Input labels and connector marking |

### 8.1 What IEC 60601-2-25 tests

IEC 60601-2-25:2011 clause 201.12.4.107.2 requires the electrocardiograph to record a ±5 mV input signal while tolerating differential and common-mode DC offsets of ±300 mV, with the recorded amplitude changing by no more than 5% across the recording width. The frequency response in Table 201.107 extends from 0.05 Hz to at least 150 Hz for diagnostic ECG.

The standard also specifies tests for input impedance, gain accuracy, noise, and common-mode rejection. The CMRR test is performed with a defined source imbalance, usually 1 kΩ, and the result is reported as a ratio. The test condition matters: an amplifier that achieves 100 dB with a 1 kΩ imbalance may achieve 80 dB with a 10 kΩ imbalance, which is closer to a real electrode pair.

The standard does not prescribe an amplifier topology or a gain. It prescribes the input conditions and the output accuracy. The design is free to meet them in any way, as long as the evidence supports the claim.

### 8.2 What IEC 60601-1 constrains

IEC 60601-1 classifies applied parts as type B, BF, or CF and sets patient leakage limits. For alternating current at 50/60 Hz, the limits in clause 8.7.3 are 100 µA normal and 500 µA single-fault for B and BF, and 10 µA normal and 50 µA single-fault for CF. The AFE's input protection, bias network, and isolation determine the leakage path.

The leakage limit constrains the components that can connect to the patient. A bias resistor to the supply rail provides a leakage path; an isolation barrier breaks it. A protection capacitor to ground provides a path for AC leakage; a series resistor limits it. The AFE designer must know the applied-part classification before choosing the bias network.

The standard also constrains the applied-part temperature and the resistance to cleaning and sterilization. An AFE that dissipates 0.5 mW is not a thermal concern; an AFE that dissipates 500 mW in a hand-held probe may be.

### 8.3 What IEC 60601-1-2 adds

IEC 60601-1-2:2014+AMD1:2020 sets immunity levels by environment. Electrostatic discharge is 8 kV contact and 15 kV air. Radiated RF immunity is 3 V/m in a professional healthcare facility and 10 V/m in a home healthcare environment, from 80 MHz to 2.7 GHz, with 80% amplitude modulation at 1 kHz. Conducted RF is 3 V RMS from 150 kHz to 80 MHz, with 6 V RMS in the ISM bands. Power-frequency magnetic field immunity is 30 A/m.

An AFE can pass these tests and still show interference in use. The test levels are not the same as the worst-case environment, and the test setup is not the same as the clinical setup. A 3 V/m field at a frequency where the cable resonates can produce a larger voltage at the input than the test implies. The data team should know which tests were performed and which cable configuration was used.

### 8.4 What the standards do not cover

The standards do not specify the gain, the filter topology, the RLD loop gain, the PCB layout, or the firmware configuration. They specify the performance that must be demonstrated. A device can be compliant and still produce data that are unsuitable for a model, because the compliance tests use clean signals and the model sees real ones.

The standards also do not cover the data interface. A device that meets IEC 60601-2-25 can still export a filtered waveform with a different bandwidth from the one it used for its diagnostic claim. The exported data must be described by its own configuration, not by the standard's test conditions.

### 8.5 The regulatory pathway

In the United States, an ECG monitor is usually a class II device and reaches the market through the 510(k) pathway, with a predicate device and a substantial-equivalence argument. In the European Union, it is usually class IIa or IIb under the Medical Device Regulation, with a clinical evaluation and a notified-body assessment. Both jurisdictions require risk management to ISO 14971 and software lifecycle evidence to IEC 62304 for the firmware that configures the AFE. A clearance is not a clinical outcome study; it is a determination that the device meets the regulatory standard for its intended use.

### 8.6 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 60601-2-25 | Gain, bandwidth, CMRR test result, input impedance | Reproduce the diagnostic claim |
| IEC 60601-1 | Applied-part type, leakage test result, isolation | Explain safety-related filtering |
| IEC 60601-1-2 | Immunity levels, cable configuration, worst-case frequency | Interpret interference artefacts |
| ISO 14971 | Hazard analysis for saturation and interference | Link AFE failure to harm |
| IEC 62304 | Firmware version, register configuration | Detect a configuration change |
| EU MDR | Clinical evaluation population, PMS data | Support conformity and surveillance |

The record is the same information the data team needs for provenance. A team that records the gain, corners, and firmware for its own analysis is also assembling regulatory evidence.

## 9. Industry practice and case evidence

### 9.1 Integrated AFEs dominate wearables

A representative integrated AFE, the Analog Devices AD8232, integrates a fixed-gain instrumentation amplifier, a two-pole high-pass filter, a three-pole low-pass filter, a driven-right-leg amplifier, and lead-off detection in a single package. Its datasheet (Rev. D, 2020) specifies a fixed gain of 100 V/V, a CMRR of 80 dB minimum and 86 dB typical from DC to 60 Hz, an input-referred noise of 14 µV peak-to-peak over 0.5–40 Hz, a supply current of 170 µA, and an operating supply of 2.0–3.5 V. The ±300 mV electrode offset is rejected by a DC-blocking integrator, so the gain and the high-pass filtering happen in one stage.

The integration changes the data scientist's problem. The gain is fixed at 100, the high-pass corner is set by an external capacitor, and the low-pass corner is set by another. The same silicon can produce different data in two products because the capacitors differ. The configuration must be recorded with the data, or the provenance is incomplete.

**Documented fact:** the datasheet values above are from the manufacturer's Rev. D. **Inference:** the prevalence of this architecture in wearables is inferred from its feature set and from teardown reports, not from a census of products.

### 9.2 Discrete instrumentation amplifiers remain the reference

The Analog Devices AD8422 is a representative discrete in-amp. Its datasheet (Rev. C, November 2022) specifies a CMRR of 150 dB minimum at a gain of 1000 with a 1 kΩ source imbalance, an input voltage noise of 8 nV/√Hz maximum at 1 kHz, an input current noise of 90 fA/√Hz, an offset drift of 0.3 µV/°C maximum, and a quiescent current of 368 µA maximum. The gain is set by a single resistor from 1 to 1000.

A discrete in-amp can outperform an integrated AFE on noise and CMRR, but it requires an external filter, a reference, a protection network, and a layout. The choice between integrated and discrete is a trade between performance and integration. A wearable with a 10 kΩ electrode and a 40 Hz bandwidth does not need an 8 nV/√Hz amplifier; the electrode noise is 18.5 nV/√Hz, and the integrated AFE's 2 µV RMS is adequate. A diagnostic ECG with a 0.05 Hz bandwidth and a low-noise requirement may need the discrete solution.

**Documented fact:** the datasheet values are from Rev. C. **Inference:** the suitability for a specific application depends on the source impedance and bandwidth, which must be computed.

### 9.3 The CMRR specification is not the system CMRR

A recurring industry issue is the gap between an amplifier's CMRR specification and the rejection achieved in a system. The specification is measured with a defined source imbalance; the system has electrodes, cables, connectors, and a patient. The mismatch term in Section 5.16 usually dominates.

The practical consequence is that a design review that quotes the amplifier's CMRR without the electrode mismatch is incomplete. The review should state the effective CMRR at the expected mismatch and the common-mode voltage, and should include a test with mismatched electrodes. A test with matched resistors in a simulator will pass a design that fails in use.

**Documented fact:** the mismatch mechanism is standard circuit theory. **Inference:** the magnitude in a specific device depends on the electrode and input impedance, which must be measured.

### 9.4 Dry electrodes change the AFE requirement

Dry electrodes have higher and more variable contact impedance than wet Ag/AgCl, often 50–500 kΩ at 10 Hz compared with 5–20 kΩ. The higher impedance increases the thermal noise, converts the amplifier's current noise into voltage noise, and makes the input bias current a larger offset. An AFE designed for a wet electrode may be inadequate for a dry one, even if the signal amplitude is the same.

The fix is a higher input impedance, a lower current noise, and a bias network that does not load the source. A CMOS or JFET input is usually preferable. The trade is higher voltage noise and a greater sensitivity to input capacitance.

**Documented fact:** the impedance ranges are representative values from the literature. **Inference:** the performance gap in a clinical population is inferred from the mechanism and from limited validation data.

### 9.5 What good AFE practice looks like

Teams that avoid AFE surprises tend to:

- Measure the source impedance and mismatch before choosing the amplifier.
- Compute the effective CMRR with the actual mismatch, not the datasheet's test condition.
- Include the protection network in the noise budget.
- State the high-pass corner and the settling time together.
- Test with mismatched electrodes and with the RLD disabled.
- Record the gain, filter corners, reference, and firmware configuration.
- Re-verify after any component or firmware change.

The practices are not exotic. They are the difference between an AFE that works on the bench and one that works in the field.

### 9.6 A note on vendor evaluation boards

Vendor evaluation boards are useful for measuring an AFE's noise and bandwidth, but they are not the product. The board's layout, cable, and power supply differ from the product's, and the board's default configuration may not be the production configuration. A measurement on an evaluation board is an upper bound on performance, not a validation.

The same applies to the vendor's application notes. They describe a circuit that worked for the vendor's test conditions. The source impedance, the cable, and the interference environment in the product may be different. The application note is a starting point for the design, not evidence that the design meets the specification.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Saturation on electrode offset | Flat-topped waveform; codes at rail | Gain times offset exceeds range | Loss of signal | DC servo; lower gain; headroom check |
| Common-mode interference | 50/60 Hz tone; changes with patient movement | RLD failure or electrode mismatch | False rhythm; poor SNR | RLD test; electrode matching; input impedance |
| RLD oscillation | High-frequency oscillation; unstable baseline | Poor loop compensation | Added noise; possible saturation | Reduce loop gain; compensate; test with cable |
| High-pass settling | Baseline shift after motion | Corner too low for the recovery requirement | False asystole or pause | Choose corner from recovery requirement |
| Anti-alias filter too high | High-frequency noise folds into band | Sampling below twice the input bandwidth | Irreversible corruption | Lower corner; raise sampling rate |
| Protection resistor noise | Noise higher than the amplifier specification | Series resistance in the input path | Reduced SNR | Lower resistance; low-noise resistor; budget |
| Input bias current offset | Offset varies with source impedance | Bias current times source impedance | Saturation or bias | Low-bias amplifier; bias compensation |
| Reference noise | Output noise changes with reference | Reference feeds the converter and amplifier | Gain error and noise | Low-noise reference; filter; ratiometric |
| Supply coupling | Noise at the switching frequency | Finite PSRR; shared ground | Interference | LDO; decoupling; layout |
| Capacitive load instability | Overshoot; ringing; oscillation | Output stage driving cable capacitance | Invalid signal | Isolation resistor; buffer |
| Leakage current exceedance | Safety test failure | Bias or protection path to patient | Regulatory failure; patient risk | Isolation; current-limiting resistors |
| Firmware configuration loss | Gain or filter changes after update | Registers reset to default | Data mismatch | Verify configuration; checksum; record |

The most dangerous failure is the one that produces a plausible waveform. A 50 Hz interference that is filtered by the display can still corrupt the algorithm; a saturated channel that is AC-coupled can still show a QRS-like shape. The red flag is a change in the relationship between the AFE output and an independent reference, not necessarily an error message.
### 10.1 Diagnosing an AFE from data

| Observation | Likely AFE cause | Confirm by |
|---|---|---|
| Flat-topped peaks at a fixed code | Saturation at the amplifier or converter | Code histogram; input range calculation |
| 50/60 Hz tone that varies with posture | RLD or electrode mismatch | Disable RLD; measure common mode |
| Baseline that drifts with respiration | High-pass corner too low | Change corner; compare raw and filtered |
| Noise that falls when the electrode is pressed | Contact impedance or bias path | Measure impedance; check bias resistor |
| A tone at the chopper frequency | Clock feedthrough | Change chopping frequency; inspect spectrum |
| Distortion at high signal amplitude | Output swing or slew limit | Reduce gain; measure THD versus amplitude |
| A sudden gain change after service | Configuration register reset | Read back registers; check firmware |
| Noise that appears only on battery | Ground loop or charger coupling | Test on battery; inspect shield connection |
| Offset that changes with temperature | Offset drift or bias current drift | Temperature sweep; record offset |
| A signal that looks like the previous channel | Multiplexer or configuration error | Inject a known signal per channel |

The table is a starting point, not a diagnosis. The confirm step is what separates an inference from a finding.

### 10.2 Ten red flags in AFE data

1. A noise floor that changes when the device is touched.
2. A 50/60 Hz tone whose amplitude depends on the patient's posture.
3. A baseline that recovers exponentially after every movement.
4. A signal that clips at a code value that is not the converter's full scale.
5. A channel whose noise is lower than the datasheet's typical value.
6. A gain that changes after a firmware update.
7. A filter corner that moves with temperature.
8. A common-mode test that passes with resistors but fails with electrodes.
9. A leakage current that is close to the limit at one supply voltage.
10. A configuration register that reads back differently from what was written.

Each red flag is a prompt to measure, not a conclusion. The measurement is usually cheap and always cheaper than a field failure.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The source impedance is known and stable | Noise and CMRR depend on it | Dry electrodes, motion, cable damage | Measure at the electrode |
| The electrode mismatch is small | Effective CMRR depends on it | Poor contact, different gel, motion | Measure mismatch; inject common mode |
| The amplifier noise is the dominant term | Budget is simplified | Low-noise amplifier, high source impedance | Measure with input shorted and with source |
| The current noise is negligible | It is omitted or approximated | High source impedance | Measure with a high-impedance source |
| The CMRR specification applies to the system | It is used in the budget | Source imbalance, cable, RLD | Measure effective CMRR |
| The high-pass corner is compatible with the recovery requirement | Filter choice | Motion, defibrillation, clinical need | Step response test |
| The anti-alias filter attenuates above Nyquist | Aliasing is avoided | Component tolerance, layout | Sweep above Nyquist |
| The amplifier is stable with the actual load | No oscillation | Cable capacitance, filter Q | Phase margin measurement |
| The protection network is transparent | It is omitted from the budget | Series resistance, clamp capacitance | Noise and bandwidth measurement |
| The supply is quiet enough | PSRR is adequate | Switching regulator, shared ground | Spectrum with a quiet input |
| The leakage path meets the applied-part limit | Safety | Bias network, protection capacitor | Leakage current test |
| The configuration is recorded | Provenance | Firmware update, service | Read back registers; version log |

The dominant assumption in most AFE designs is that the electrode mismatch is small. It is small in a laboratory with prepared skin and matched electrodes; it is not small in a moving patient with dry skin. A design that is validated only in the laboratory has validated the assumption, not the product.

A second limit is that the AFE models in this topic are linear. Real amplifiers compress before they clip, and real filters ring after a transient. A signal that is 80% of full scale may already be compressed; a filter that is 20% from its corner may already have significant phase shift. The linear model is adequate for budgeting, not for verifying a precision claim.

A third limit is that the noise budget assumes stationary, uncorrelated noise. Interference is neither. A 50 Hz tone is correlated with itself and can be removed by a notch filter, but the notch also removes signal at 50 Hz and rings after a transient. The RMS budget does not capture the tail of the interference distribution, which is what an alarm threshold depends on.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no test, only a datasheet?
- Which assumption was true on the bench but is false in the home?
- Which assumption depends on the electrode or the cable?
- Which assumption is encoded in a register that can be reset?
- Which assumption is specific to the validation population?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A design review that cannot answer the last two has risks without controls.

### 11.2 A note on the model's scope

The AFE model in this topic is a linear, time-invariant, small-signal model. It is adequate for noise, gain, and CMRR at a fixed operating point. It is not adequate for:

- Large-signal behaviour near the rails, where gain compresses.
- Transient behaviour after a defibrillation or an ESD event.
- Stability with a complex load, which requires a loop analysis.
- Thermal behaviour with a duty-cycled load.
- Long-term drift, which requires reliability data.

Each of these needs a different model, a simulation, or a measurement. The linear model tells you where to look; it does not replace the test.

## 12. Exercises

### Exercise 1. Choose a topology (recall, 20 minutes)

For each source, choose an AFE topology and justify it in one sentence: (a) 10 kΩ wet ECG electrode, 0.5–40 Hz; (b) 1 MΩ dry EEG electrode, 0.5–30 Hz; (c) 350 Ω bridge, DC–100 Hz; (d) 1 pF piezoelectric sensor, 1–10 kHz.

**Deliverable:** a four-row table with topology and justification.

**Acceptance criteria:** the justification names the source impedance and the dominant noise; no topology is chosen without a reason.

**Time:** 20 minutes.

### Exercise 2. Compute a noise budget (application, 40 minutes)

An AFE uses a 5 kΩ source, an amplifier with 10 nV/√Hz voltage noise and 0.5 pA/√Hz current noise, a 1 kΩ protection resistor, and a 100 Hz first-order low-pass. Compute the input-referred noise, the SNR for a 2 mV peak-to-peak signal, and the ENOB. State which term dominates.

**Deliverable:** a calculation table and a two-sentence conclusion.

**Acceptance criteria:** correct noise bandwidth; current noise converted through the source; protection resistor included; units on every term.

**Time:** 40 minutes.

### Exercise 3. Diagnose a CMRR failure (application, 30 minutes)

A system with a 120 dB amplifier CMRR, a 10 MΩ input impedance, and a 2 kΩ electrode mismatch shows 50 µV of 50 Hz interference at the input when the common mode is 1 V. Explain why, compute the effective CMRR, and propose two fixes.

**Deliverable:** a calculation and two proposals with costs.

**Acceptance criteria:** mismatch term computed; effective CMRR correct; each fix states its effect on input impedance, noise, or cost.

**Time:** 30 minutes.

### Exercise 4. Set a high-pass corner (application, 25 minutes)

A monitor must recover from a 300 mV motion step within 3 s to 1%. Choose a first-order high-pass corner and compute the settling time. Then state what clinical information is lost at that corner.

**Deliverable:** a calculation and a one-paragraph trade-off.

**Acceptance criteria:** settling formula correct; corner chosen from the requirement; the lost information is named.

**Time:** 25 minutes.

### Exercise 5. Design a protection network (design, 35 minutes)

Design an input protection network for a wearable ECG that must survive 8 kV contact ESD and a defibrillation event, while keeping the input-referred noise below 3 µV RMS over 0.5–40 Hz. State the component values, the noise contribution, and the bandwidth effect.

**Deliverable:** a schematic description and a budget.

**Acceptance criteria:** protection and noise are traded explicitly; the bandwidth effect is calculated; the leakage path is identified.

**Time:** 35 minutes.

### Exercise 6. Write a verification matrix (design, 40 minutes)

Write a verification matrix for the AFE in Section 6. Include gain, bandwidth, CMRR with matched and mismatched electrodes, noise, offset, settling, saturation, supply rejection, and leakage. State the acceptance criterion and the margin for each.

**Deliverable:** a matrix with at least 10 rows.

**Acceptance criteria:** covers the edges; each row has a criterion and a margin; the mismatched CMRR and the RLD-disabled case are included.

**Time:** 40 minutes.

### Exercise 7. Analyse a settling failure (critique, 30 minutes)

A wearable monitor with a 0.05 Hz high-pass corner alarms on pause after a motion artefact. Explain the mechanism, estimate the false-alarm duration, and propose a design change that preserves the ST segment.

**Deliverable:** a memo with the calculation and the proposal.

**Acceptance criteria:** settling time correct; the mechanism connects the filter to the alarm; the proposal is technically feasible.

**Time:** 30 minutes.

### Exercise 8. Design a data record (design, 30 minutes)

Design a configuration record for an AFE that would allow a later analyst to detect a gain, filter, or reference change. Include at least 12 fields and explain the diagnostic use of each.

**Deliverable:** a table with field, type, example, and diagnostic use.

**Acceptance criteria:** covers gain, corners, reference, RLD, protection, firmware, and calibration; each field has a stated use.

**Time:** 30 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Topology selection | Table with justifications |
| 2 | Noise budget | Calculation table |
| 3 | CMRR diagnosis | Effective CMRR and fixes |
| 4 | Filter choice | Settling calculation |
| 5 | Protection design | Schematic and budget |
| 6 | Verification | Matrix |
| 7 | Failure analysis | Memo |
| 8 | Provenance | Configuration table |

## 13. Self-check answers

**Answer 1 (Exercise 1).** (a) Integrated biopotential AFE or discrete in-amp with a 10 kΩ source and a 40 Hz bandwidth; the source noise is 13 nV/√Hz, so an 8–20 nV/√Hz amplifier is adequate. (b) CMOS or JFET in-amp, because the 1 MΩ source converts current noise into voltage noise; a bipolar input with 1 pA/√Hz would contribute 1 µV/√Hz. (c) Difference amplifier or in-amp with low voltage noise, because the 350 Ω bridge has low thermal noise and the signal is small. (d) Charge amplifier, because the piezoelectric source is capacitive and the charge is the signal; a voltage amplifier would be dominated by cable capacitance.

**Answer 2 (Exercise 2).** Noise bandwidth = 1.571 × 100 = 157.1 Hz. Source thermal noise for 5 kΩ at 310 K = √(4 × 1.38e-23 × 310 × 5000) = 9.25 nV/√Hz; over 157.1 Hz = 116 nV RMS. Amplifier voltage noise = 10 × 12.53 = 125 nV RMS. Current noise = 0.5 pA/√Hz × 5 kΩ = 2.5 nV/√Hz → 31 nV RMS. Protection resistor 1 kΩ = 4.14 nV/√Hz → 52 nV RMS. Total = √(116² + 125² + 31² + 52²) = 178 nV RMS. SNR for 2 mV peak-to-peak (0.707 mV RMS) = 20 log10(707/0.178) = 72.0 dB. ENOB = (72.0 − 1.76)/6.02 = 11.7 bits. The amplifier voltage noise and the source thermal noise are comparable; the current noise is negligible at 5 kΩ.

**Answer 3 (Exercise 3).** The mismatch term is V_cm × ΔZ/Z_in = 1 V × 2000/10e6 = 200 µV. The amplifier's own CMRR at 120 dB contributes 1 µV. The measured 50 µV is lower than 200 µV because the actual mismatch or the effective input impedance differs, but the mechanism is the mismatch, not the amplifier. The effective CMRR is 20 log10(1/50e-6) = 86 dB. Fixes: increase the input impedance to 1 GΩ, which reduces the mismatch term to 2 µV; or match the electrodes to 100 Ω, which reduces it to 10 µV. A third fix, a buffer at the electrode, increases the input impedance and reduces cable effects.

**Answer 4 (Exercise 4).** For 1% settling, t = 4.6τ = 4.6/(2πf_c). Setting t = 3 s gives f_c = 4.6/(2π × 3) = 0.244 Hz. A 0.25 Hz corner gives t = 2.93 s. At 0.25 Hz, the ST segment and slow baseline changes are attenuated; the monitor is suitable for rhythm and rate, not for diagnostic ST analysis. A 0.5 Hz corner would settle in 1.47 s but remove more of the ST segment.

**Answer 5 (Exercise 5).** A protection network: 10 kΩ series resistors on each input, clamp diodes to the rails, a 1 nF capacitor to ground after the resistors, and a transient-voltage suppressor at the connector. The 10 kΩ resistors contribute 12.8 nV/√Hz each, or 90 nV RMS over 62.8 Hz; the 1 nF capacitor with 10 kΩ forms a pole at 15.9 kHz, above the 40 Hz band. The clamp leakage of 1 nA into 10 kΩ is 10 µV, which is within the offset budget if the amplifier's offset is trimmed. The leakage path to the patient is limited by the 10 kΩ resistors and the isolation barrier.

**Answer 6 (Exercise 6).** The matrix should include: gain at 10 Hz (100 ± 1 V/V); bandwidth −3 dB at 0.5 and 40 Hz; CMRR with matched electrodes > 80 dB; CMRR with 1 kΩ mismatch > 60 dB; noise < 2.5 µV RMS; offset < 10 µV; settling < 2 s; saturation flag at 400 mV; supply rejection < 5 µV; leakage < 10 µA. Each row records the measured value and the margin.

**Answer 7 (Exercise 7).** At 0.05 Hz, τ = 3.18 s, and the 1% settling is 14.7 s. After a motion step, the baseline is outside the alarm threshold for most of that time, so a pause or asystole alarm can fire falsely. The fix is a 0.5 Hz corner for the alarm path and a separate 0.05 Hz path for the ST analysis, or a motion detector that suspends the alarm during recovery. The first preserves the ST segment but doubles the AFE; the second preserves a single AFE but requires a reliable motion signal.

**Answer 8 (Exercise 8).** Fields: gain; high-pass corner; low-pass corner; filter order; reference voltage; RLD enable and gain; protection configuration; input bias network; firmware version; calibration date; offset trim; noise measurement at last verification. Each field supports a diagnostic: a gain change explains a scale shift, a corner change explains a morphology change, a reference change explains a bias, and a firmware change explains a configuration reset.

**Rubric for the design exercises.** A strong design states the requirement, the source, the topology, the noise budget, the failure modes, and the verification. A weak design names components without quantities or omits the protection and leakage path.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 2: source noise | 116 nV RMS | ±5 nV |
| Exercise 2: total noise | 178 nV RMS | ±10 nV |
| Exercise 2: SNR | 72.0 dB | ±0.5 dB |
| Exercise 2: ENOB | 11.7 bits | ±0.1 bit |
| Exercise 3: mismatch term | 200 µV | ±10 µV |
| Exercise 3: effective CMRR | 86 dB | ±1 dB |
| Exercise 4: corner for 3 s | 0.244 Hz | ±0.02 Hz |
| Exercise 4: settling at 0.25 Hz | 2.93 s | ±0.1 s |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 5 | 10 | 20 |
| 2 | 5 | 25 | 10 | 40 |
| 3 | 5 | 15 | 10 | 30 |
| 4 | 5 | 10 | 10 | 25 |
| 5 | 5 | 20 | 10 | 35 |
| 6 | 5 | 20 | 15 | 40 |
| 7 | 5 | 15 | 10 | 30 |
| 8 | 5 | 10 | 15 | 30 |

The time boxes assume a reader who has finished the topic. A reader who has not should add 30 minutes for the first calculation.

## 14. Interview questions

**1. What does an analog front end do?**
A strong answer names protection, amplification, filtering, and referencing, and explains that the stages interact. A weak answer says it "amplifies the signal".

**2. Why is a discrete difference amplifier limited to about 40 dB CMRR?**
A strong answer explains that the resistor mismatch sets the common-mode gain and that 1% resistors give about 40 dB regardless of the op-amp. A weak answer attributes the limit to the op-amp.

**3. What is the difference between amplifier CMRR and system CMRR?**
A strong answer explains the source-impedance mismatch term and the input impedance, and gives an example. A weak answer quotes the datasheet number.

**4. How do you choose the gain?**
A strong answer writes the headroom equation, includes the offset, and trades gain against range and noise. A weak answer picks a gain because it is common.

**5. Why does a high-pass filter cause a settling problem?**
A strong answer gives the time constant and the 1% settling formula, and connects it to a clinical recovery requirement. A weak answer says the filter "takes time".

**6. What does a driven-right-leg circuit do?**
A strong answer explains that it senses the common-mode voltage and drives it back to reduce it, and that the loop must be stable and current-limited. A weak answer says it "reduces noise".

**7. How does protection affect noise?**
A strong answer explains that series resistance adds thermal noise and forms a pole with capacitance, and that the protection must be in the budget. A weak answer treats protection as transparent.

**8. When would you use a chopper amplifier?**
A strong answer names low offset, low flicker noise, and high source impedance, and notes the clock artefacts. A weak answer says it is "more accurate".

**9. How do you verify an AFE?**
A strong answer covers gain, bandwidth, CMRR with mismatch, noise, offset, settling, saturation, supply rejection, and leakage. A weak answer tests only the gain.

**10. What would make you reject an AFE design?**
A strong answer names an unstable loop, a leakage failure, a saturation that cannot be detected, or a CMRR that depends on unrealistic electrode matching. A weak answer rejects on price alone.

**11. How do you measure effective CMRR?**
A strong answer injects a known common-mode signal with matched and mismatched source impedances, measures the differential output, and computes the ratio. A weak answer reads the datasheet.

**12. Why does protection design affect signal quality?**
A strong answer explains series resistance noise, capacitance, and the pole it forms, and shows the trade with the bandwidth. A weak answer treats protection as independent of the signal path.

## 15. Key terms

- **AFE** — analog front end.
- **Anti-alias filter** — low-pass filter before sampling.
- **Bias current** — current flowing into an amplifier input.
- **Chopper** — amplifier that modulates to reduce offset and flicker noise.
- **CMRR** — common-mode rejection ratio.
- **Common-mode voltage** — average of the two inputs relative to the reference.
- **DC servo** — feedback loop that removes a DC offset.
- **Differential signal** — difference between two inputs.
- **Driven right leg (RLD)** — circuit that drives the patient reference.
- **Flicker noise** — 1/f noise at low frequency.
- **Gain bandwidth** — frequency at which open-loop gain is unity.
- **Guard** — conductor driven to the common-mode voltage.
- **Headroom** — margin between signal and rail.
- **Input impedance** — impedance presented by the amplifier input.
- **Instrumentation amplifier** — differential amplifier with high input impedance.
- **Isolation barrier** — barrier that breaks the leakage path.
- **Leakage current** — current through the patient to earth.
- **MOPP** — means of patient protection.
- **Noise gain** — gain seen by the noise sources, which may differ from the signal gain.
- **Phase margin** — stability margin at the unity-gain crossing.
- **PSRR** — power-supply rejection ratio.
- **Sallen-Key** — second-order filter topology.
- **Settling time** — time to reach a stated accuracy after a step.
- **Slew rate** — maximum output rate of change.
- **Source impedance** — impedance of the sensor and cable.
- **Stability** — freedom from oscillation.
- **Thermal noise** — noise from random charge motion.
- **Transimpedance amplifier** — current-to-voltage converter.
- **Voltage noise density** — input-referred noise per root hertz.
- **Common-mode input impedance** — impedance seen by a common-mode source.
- **Difference amplifier** — one-op-amp differential amplifier with four resistors.
- **Noise bandwidth** — bandwidth of an ideal filter passing the same noise power.
- **Reference** — voltage that defines the converter's full scale and the amplifier's output centre.
- **Transient-voltage suppressor (TVS)** — clamp used for protection.
- **Zero-drift amplifier** — amplifier with very low offset drift, often chopper-stabilised.
- **Common-mode range** — input voltage range over which the amplifier's CMRR is maintained.
- **Dielectric absorption** — capacitor memory effect that causes settling tails.
- **Input capacitance** — capacitance at the amplifier input, which forms a pole with the source.
- **Noise figure** — degradation of SNR through a stage, in dB.
- **Overvoltage protection** — circuitry that limits input voltage under fault.
- **Resistor matching** — ratio accuracy that sets a difference amplifier's CMRR.
- **Signal gain** — gain applied to the differential signal.
- **Supply sequencing** — order in which rails are powered.
- **Thermal drift** — change in a parameter with temperature.
- **Unity-gain bandwidth** — frequency at which the open-loop gain is one.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Datasheets**

1. Analog Devices, *AD8232: Single-Lead, Heart Rate Monitor Front End*, Rev. D, 2020. https://www.analog.com/media/en/technical-documentation/data-sheets/AD8232.pdf
2. Analog Devices, *AD8422: High Performance, Low Power, Rail-to-Rail Instrumentation Amplifier*, Rev. C, November 2022. https://www.analog.com/media/en/technical-documentation/data-sheets/AD8422.pdf
3. Texas Instruments, *ADS129x Low-Power, 2-Channel, 24-Bit Analog Front-End*, SBAS502C, 2020. https://www.ti.com/lit/ds/symlink/ads1292.pdf
4. Analog Devices, *MT-001: Taking the Mystery out of SNR = 6.02N + 1.76 dB*. https://www.analog.com/media/en/training-seminars/tutorials/MT-001.pdf

**Standards and regulation**

5. IEC 60601-2-25:2011, *Particular requirements for electrocardiographs*. https://webstore.iec.ch/en/publication/26368
6. IEC 60601-2-27:2011, *Particular requirements for electrocardiographic monitoring equipment*. https://webstore.iec.ch/en/publication/26400
7. IEC 60601-1:2005+AMD1:2012+AMD2:2020, *General requirements for basic safety and essential performance*. https://webstore.iec.ch/en/publication/26169
8. IEC 60601-1-2:2014+AMD1:2020, *Electromagnetic disturbances — Requirements and tests*. https://webstore.iec.ch/en/publication/67554
9. IEC 60601-1-8:2006+AMD2:2020, *Alarm systems*. https://webstore.iec.ch/en/publication/26539
10. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
11. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
12. Regulation (EU) 2017/745 on medical devices. https://eur-lex.europa.eu/eli/reg/2017/745/oj

**Background**

13. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
14. S. Franco, *Design with Operational Amplifiers and Analog Integrated Circuits*, 4th ed., McGraw-Hill, 2014.
15. A. S. Sedra and K. C. Smith, *Microelectronic Circuits*, 8th ed., Oxford University Press, 2019.
16. Texas Instruments, *Analog Engineer's Circuit Cookbook: Amplifiers*, 2019. https://www.ti.com/lit/pdf/slyy137
17. WhaleTeq, *AN_11: How to Measure the Amplitude Linearity, Dynamic Range and DC Offset Tolerance*, application note. https://www.whaleteq.com/data/adownload/files/1672134706493523860.pdf
18. Analog Devices, *AD8232 evaluation board user guide*. https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/eval-ad8232.html
19. Texas Instruments, *ECG front-end design with the ADS1298*, application report. https://www.ti.com/lit/pdf/sbaa175
20. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, chapter on biopotential amplifiers.
21. IEC 60601-1-11:2015+AMD1:2020, *Requirements for medical electrical equipment used in the home healthcare environment*. https://webstore.iec.ch/en/publication/67384
22. ISO 80601-2-61:2017, *Particular requirements for pulse oximeter equipment* (for the optical AFE analogy). https://www.iso.org/standard/67963.html
23. Analog Devices, *Linear Circuit Design Handbook*, 2008. https://www.analog.com/en/resources/technical-books/linear-circuit-design-handbook.html
24. B. Carter, *Op Amps for Everyone*, 5th ed., Newnes, 2017.
25. IEC 61000-4-2:2008, *Electrostatic discharge immunity test*. https://webstore.iec.ch/en/publication/4208

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_sensors.md`](02_sensors.md)
- Next topic: [`04_adcs-and-signal-conditioning.md`](04_adcs-and-signal-conditioning.md)
- Related: [`01_the-signal-chain.md`](01_the-signal-chain.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on ECG quality: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)
- Lab on signal-chain simulation: see Lab 14A.1 in the survey file.
- Lab on hardware-in-the-loop testing: see Lab 14A.7 in the survey file.
- Lab on device teardown: see Lab 14A.6 in the survey file.
- Lab index: [`../../labs/`](../../labs/)

### Appendix note

The worked example uses an integrated AFE with a fixed gain of 100. A discrete design with a gain of 1000 would reduce the input-referred noise contribution of the later stages but would reduce the input range to 2.4 mV, which would saturate on the electrode offset unless a servo were added. The trade is not between a better and a worse amplifier; it is between two different allocations of the same range and noise budget. The right choice depends on which signal the device must preserve and which failure it must avoid.

The same reasoning applies to the filter corners. A 0.5 Hz corner is not "better" than a 0.05 Hz corner; it is faster to settle and removes more slow signal. A 40 Hz corner is not "better" than a 150 Hz corner; it removes more muscle noise and more diagnostic content. The requirement decides, and the requirement is clinical, not electrical. A design review that argues about the corner without naming the clinical information it protects has not framed the decision correctly.

Quantities to record with every AFE dataset:

- Gain and gain tolerance.
- High-pass corner and order.
- Low-pass corner and order.
- Reference voltage and tolerance.
- RLD configuration and loop gain.
- Protection network values.
- Input bias network.
- Supply voltage and regulator type.
- Firmware version and register map.
- Calibration date and coefficients.
- Noise measurement at last verification.
- Leakage test result and applied-part type.
