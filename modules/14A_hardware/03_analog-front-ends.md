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

or 100 µV at the output, which is below the 2.14 µV input-referred noise. The RLD does not need to be perfect; it needs to reduce the common mode to the point where the amplifier's CMRR brings it below the noise floor.

The electrode mismatch of 100 Ω in 10 kΩ is 1%. The input impedance of the integrated AFE is high, so the mismatch conversion is small; assume a common-mode input impedance of 10 MΩ. The mismatch produces:

\[
V_{\text{diff,mismatch}} = V_{\text{cm}} \times \frac{100}{10\,000\,000} = 10\ \text{mV} \times 1 \times 10^{-5} = 0.1\ \text{µV}
\]

which is negligible compared with the 1.0 µV from the amplifier's CMRR. At the original 1 V common mode, the mismatch would produce 10 µV, which would dominate. The RLD reduces both terms.

### 6.4 Diagnostic 1: the RLD is disconnected

A service technician replaces the electrode cable and forgets to connect the right-leg lead. The common-mode voltage returns to 1.0 V. The amplifier's CMRR produces 100 µV at the input, and the mismatch produces 10 µV. The total interference is about 100 µV, which is 10% of the 1 mV signal and 47 times the 2.14 µV noise.

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

This is now equal to the amplifier's CMRR residual of 1.0 µV. The total common-mode interference is about 1.4 µV, still below the 2.14 µV noise, but no longer negligible. If the RLD fails and the common mode returns to 1 V, the mismatch term becomes 100 µV, which dominates the signal.

The consequence is that the design's interference margin depends on the electrode matching, not only on the RLD. A validation that uses well-matched electrodes in the lab will not reveal the sensitivity. The fix is to specify the electrode matching, measure it in production, or increase the input impedance.

### 6.7 Power, thermal, and battery

The AFE draws 170 µA at 3.0 V, or 0.51 mW. The microcontroller and radio add about 8.0 mW, for a total of 8.51 mW. The daily energy is:

\[
E_{	ext{