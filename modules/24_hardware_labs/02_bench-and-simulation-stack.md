# Bench and Simulation Stack

> Module 24 — Hardware and Robotics Labs
> Topic 2 of 9
> Estimated study time: 4 hours
> Prerequisites: 14A, 01
> Version 2026.3

## 1. Why this topic exists

Every lab result is produced by a chain of instruments, models, and software versions. The chain determines what the result can mean. A 3.5-digit multimeter cannot resolve a 10 µV change. A 12-bit ADC cannot resolve a 1 µV signal without gain. A simulation that omits a cable's inductance cannot predict a ringing transient. A firmware build that is not pinned cannot be reproduced next semester.

The industry problem is that teams buy instruments before they know what they need to measure, and they build simulations before they know what the simulation is for. The result is a bench full of equipment that cannot answer the question, and a model that is precise about the wrong thing.

Three decisions depend on this topic:

1. **What tier of equipment the question requires.** A Tier 0 simulation can answer a question about an algorithm. It cannot answer a question about target latency or energy. A Tier 1 bench can answer a question about a single board's analog behaviour. It cannot answer a question about radiated emissions.
2. **Which instrument's uncertainty is small enough.** The instrument must be chosen from the measurement's error budget, not from a catalogue. A meter that is ten times more accurate than needed is a waste; a meter that is ten times less accurate is a false result.
3. **What the simulator is allowed to claim.** A simulator is a model with omissions. The omissions must be named, and the simulation must be anchored to a bench measurement at at least one point before it is used to predict a second point.

The commercial stakes are concrete. A test-house booking that has to be repeated costs a week and a five-figure sum. A toolchain that drifts between semesters makes a lab unreproducible and a result unciteable. A simulation used as a substitute for a target measurement becomes a design decision made on a model that was never validated.

This topic builds the stack map, the selection method, and the qualification procedure. It assumes the safety boundary from Topic 1 and the signal-chain vocabulary from Module 14A.

A second problem is subtler than the first. Two teams can use the same instrument and the same simulator and reach different conclusions, because one recorded the range, the gain, the bandwidth, and the version, and the other recorded only the final number. The first result can be checked; the second can only be believed. In a regulated environment, a result that can only be believed is not evidence.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work.

## 2. Learning outcomes

By the end of this topic you can:

- Draw the measurement chain from source to stored data and name the error contributed at each stage.
- Select an instrument from a written error budget rather than from a catalogue or a habit.
- Compute the input-referred resolution of an acquisition path from the ADC resolution, the reference voltage, and the gain.
- Distinguish a simulator's model from the system it represents, and name at least five omissions in a typical circuit or robot simulation.
- Build a version-pinned lab environment with a lockfile, a container digest, and a software bill of materials (SBOM).
- Qualify a bench path end to end with a known input, a noise-floor measurement, and a bandwidth measurement.
- Anchor a simulation to a bench measurement and state the conditions under which the anchor is valid.
- Write a bill of materials (BOM) and a stack record that another person could use to rebuild the bench.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A lab has two stacks that must meet: the physical stack and the simulated stack. The physical stack measures the world. The simulated stack predicts it. The two meet at a validation point, and the quality of the lab depends on how honestly that meeting is described.

### 3.1 The physical measurement chain

```text
  SOURCE        DUT          PROBE        INSTRUMENT      DIGITISER      ANALYSIS
  generator --> circuit --> probe -----> amplifier ----> ADC ---------> file
      |            |           |             |              |              |
  amplitude     gain        loading       bandwidth      sample rate    algorithm
  offset        noise       attenuation   noise          resolution     assumptions
  impedance     distortion  common mode   accuracy       clock jitter   uncertainty
```

Each block transforms both the signal and the uncertainty. A probe with 10 pF of capacitance loads a high-impedance node. An amplifier with 1 MHz of bandwidth adds noise over 1 MHz. An ADC with 12 bits and a 3.3 V reference has a least significant bit (LSB) of 0.806 mV. An analysis that ignores the sample rate assumes a bandwidth the digitiser did not provide.

The chain is only as good as its weakest link, but the weakest link is not always the obvious one. In a low-frequency measurement, the ADC resolution often dominates. In a fast measurement, the probe and the clock dominate. In a high-impedance measurement, the probe loading dominates. The error budget is how you find out which.

### 3.2 The simulated stack

```text
  MODEL          SOLVER         RESULT         ANCHOR          CLAIM
  schematic --> equations --> numbers -----> bench point --> prediction
  URDF/MJCF      integrator     plots          tolerance        boundary
  parameters     timestep       logs           uncertainty      omissions
```

A simulator has four parts: the model, the solver, the result, and the anchor. The model is a set of equations and parameters. The solver is a numerical method with a timestep, a tolerance, and a stability region. The result is a number with a precision that is not the same as its accuracy. The anchor is a measurement that the model must reproduce before it is used to predict something else.

The anchor is the part that teams skip. A model that has never been compared with a measurement is a hypothesis, not evidence. A model that reproduces one bench point and is then used to predict a different operating point is an extrapolation, and the extrapolation must be labelled.

### 3.3 The tier model

The survey file defines three tiers. The tier is a property of the question, not of the budget.

| Tier | What it can answer | What it cannot answer | Typical spend |
|---|---|---|---:|
| 0 — simulation only | Algorithm behaviour, control law, geometry, nominal timing in a model | Target latency, target energy, real analog noise, real EMC | 0 |
| 1 — minimum bench | One board's analog behaviour, digital timing, firmware logic, calibration against a reference | Radiated emissions, accredited safety, multi-board integration at scale | 150–250 USD |
| 2 — fuller bench | Oscilloscope captures, power profiling, servo integration, thermal measurement | Accredited EMC, type testing, clinical use | 600–900 USD |

The tier model is a guard against two errors. The first is buying Tier 2 equipment to answer a Tier 0 question. The second is using Tier 0 evidence to answer a Tier 2 question. Both errors are common, and both are expensive in different ways.

### 3.4 The fidelity ladder

A simulation's fidelity is not a single number. It is a set of dimensions, and a model can be good in one and bad in another.

| Dimension | Low fidelity | High fidelity | Typical omission |
|---|---|---|---|
| Geometry | Idealised link lengths | CAD-accurate meshes | Cable routing, compliance |
| Dynamics | Rigid body, no friction | Contact, friction, compliance | Backlash, stiction |
| Electronics | Ideal op-amp | Datasheet model with noise and slew | Parasitic capacitance, layout |
| Timing | Event-driven, zero delay | Cycle-accurate or hardware-timed | Interrupt latency, bus contention |
| Environment | Constant temperature | Thermal model with airflow | Humidity, contamination |
| Sensor | Ideal measurement | Noise, bias, drift, saturation | Cross-sensitivity, aging |

A model that is high fidelity in geometry and low fidelity in timing is appropriate for a kinematics question and inappropriate for a latency question. The fidelity ladder forces the team to name which dimensions matter for the claim.

### 3.5 The reproducibility triangle

A lab result is reproducible when three things are recorded: the physical configuration, the software configuration, and the analysis configuration.

```text
        PHYSICAL
       /        \
  BOM, photos   firmware, model,
  calibration    container, versions
       \        /
        ANALYSIS
     script, seed, parameters
```

If any vertex is missing, the result is an anecdote. A photograph without a firmware commit cannot be rebuilt. A commit without a BOM cannot be re-powered. A script without a seed cannot be re-run.

### 3.6 The stack as a set of contracts

Each interface in the stack is a contract.

| Interface | Contract | Violation symptom |
|---|---|---|
| Source to DUT | Amplitude, offset, impedance, bandwidth | Clipping, loading, drift |
| DUT to probe | Loading, common-mode range, bandwidth | Attenuation, distortion, oscillation |
| Probe to instrument | Attenuation ratio, termination, voltage rating | Wrong amplitude, damage |
| Instrument to digitiser | Range, sample rate, resolution, trigger | Aliasing, missed events |
| Digitiser to file | Encoding, timestamp, channel mapping | Off-by-one, time shift |
| File to analysis | Units, calibration, filtering | Wrong scale, double filtering |
| Model to solver | Units, frame, parameter set | Impossible geometry, divergence |
| Solver to claim | Timestep, tolerance, validation | False precision |

Most lab disputes are contract violations, not measurement errors. A channel map that is off by one produces a plausible result that is wrong. A unit error in a model produces a plausible trajectory that is wrong. The contract table is how a reviewer finds them.

### 3.7 The failure that looks like success

The most expensive stack failures do not produce an error. They produce a plausible number.

A channel map that is off by one produces a waveform with the wrong label and the right shape. A sample rate that is set to 1 kS/s instead of 10 kS/s produces a spectrum with a different frequency axis and a plausible peak. A probe set to 1× while the software assumes 10× produces an amplitude that is ten times too large but internally consistent. A simulation with a timestep that is too large produces a trajectory that converges to the wrong answer and looks smooth.

These failures share a property: the result is self-consistent. The shape, the units, and the trend all look right. The only defence is an independent check: a known-good input, a second instrument, a dimensional check, or a comparison with a measurement at a second condition.

The check must be independent of the chain being checked. Comparing the scope's reading with the scope's own measurement does not catch a probe attenuation error. Comparing it with a calibrated DMM does. The independence is what makes the check evidence rather than confirmation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| DUT | Device under test | Called "the board" in a report | None |
| EUT | Equipment under test; the EMC term | Used without definition | None |
| AFE | Analog front end | Used for the whole signal chain | None |
| ADC | Analog-to-digital converter | Called a "sensor" | None |
| LSB | Least significant bit; the smallest code step | Confused with resolution in bits | V or A |
| Resolution | Smallest distinguishable increment | Confused with accuracy | V, A, or bits |
| Accuracy | Closeness of a measurement to the true value | Confused with precision | V or % |
| Precision | Repeatability of a measurement | Confused with accuracy | V or % |
| ENOB | Effective number of bits | Quoted without the test condition | bits |
| SINAD | Signal-to-noise-and-distortion ratio | Confused with SNR | dB |
| SNR | Signal-to-noise ratio | Quoted without bandwidth | dB |
| CMRR | Common-mode rejection ratio | Quoted at DC only | dB |
| PSRR | Power-supply rejection ratio | Confused with CMRR | dB |
| Input impedance | Resistance and capacitance seen at an input | Assumed infinite | Ω and F |
| Burden voltage | Voltage drop across a current-measuring shunt | Ignored in current measurements | V |
| Probe attenuation | Ratio of probe input to scope input | Assumed to be 10× | Ratio |
| Ground lead | Conductive path from probe tip to scope ground | Assumed to be ideal | Inductance, nH |
| Bandwidth | Frequency at which response falls by 3 dB | Confused with sample rate | Hz |
| Sample rate | Samples per second | Confused with bandwidth | S/s |
| Nyquist frequency | Half the sample rate | Confused with the signal bandwidth | Hz |
| Aliasing | Folding of frequencies above Nyquist into the baseband | Ignored without an anti-alias filter | Hz |
| Jitter | Variation in clock or event timing | Confused with latency | s RMS or peak |
| Latency | Delay from input to output | Confused with jitter | s |
| Timestamp | Recorded time of an event or sample | Assumed to be the acquisition time | s or ns |
| Trigger | Condition that starts a capture | Confused with a gate | None |
| Logic analyser | Digital capture instrument | Used for analog measurements | None |
| Protocol decoder | Software that interprets a digital bus | Assumed to be correct without checking | None |
| DMM | Digital multimeter | Used for AC leakage without a network | None |
| LISN | Line impedance stabilisation network | Called a filter | Ω, dB |
| CDN | Coupling/decoupling network | Confused with a LISN | Ω, dB |
| Function generator | Source of defined waveforms | Assumed to have a 50 Ω output | Ω |
| DAC | Digital-to-analog converter | Used as a calibrated source without checking | V |
| Reference | A known input used to calibrate a path | Assumed to be exact | Value and uncertainty |
| Traceability | Unbroken chain of calibrations | Claimed from a sticker | None |
| Calibration | Comparison against a reference with stated uncertainty | Confused with adjustment | Certificate |
| SIL | Software-in-the-loop; algorithm on a host | Called HIL | None |
| PIL | Processor-in-the-loop; algorithm on a target with simulated I/O | Confused with HIL | None |
| HIL | Hardware-in-the-loop; target runs in real time with simulated inputs | Claimed when the host runs the algorithm | None |
| MIL | Model-in-the-loop; model and controller in the same simulation | Confused with SIL | None |
| FMI | Functional Mock-up Interface | Treated as a guarantee of fidelity | Standard |
| URDF | Unified Robot Description Format | Treated as a dynamics model | XML |
| MJCF | MuJoCo XML model format | Confused with URDF | XML |
| SDF | Simulation Description Format used by Gazebo | Confused with URDF | XML |
| REP 103 | ROS enhancement proposal for units and axes | Ignored in sensor frames | Convention |
| Renode | Open-source MCU simulator | Assumed cycle-accurate | None |
| QEMU | Emulator for instruction sets and boards | Assumed timing-accurate | None |
| Wokwi | Browser-based electronics simulator | Assumed to model analog reality | None |
| Lockfile | Record of exact dependency versions | Replaced by "latest" | File |
| Container digest | Content hash of a container image | Omitted from the report | SHA-256 |
| SBOM | Software bill of materials | Confused with a BOM | Document |
| BOM | Bill of materials | Missing tolerances and part numbers | Document |
| Reproducibility | Ability to obtain the same result from the same inputs | Confused with repeatability | Property |
| Repeatability | Closeness of repeated measurements under one setup | Confused with reproducibility | Property |

## 5. The mechanics, step by step

The procedure below builds and qualifies a lab stack. It is written for a learner who has a question, a budget, and a bench. Each step states its input, output, and rationale. The branch points are marked.

### Step 1. Write the measurement question as a quantity

**Input:** the engineering question.

**Output:** a quantity, a range, a bandwidth, an accuracy target, and a decision rule.

**Rationale:** an instrument can only be chosen against a quantity. "Measure the ECG" is not a quantity. "Measure the input-referred noise of the front end from 0.5 Hz to 40 Hz, on one board, with an uncertainty below 2 µV RMS, to decide whether the front end meets a 5 µV RMS budget" is a quantity.

The question must state:

- The measurand and its unit.
- The range of values expected, including the smallest value that matters.
- The bandwidth or time resolution required.
- The uncertainty that would change the decision.
- The decision rule: what value leads to which action.

If the uncertainty target is not stated, the instrument cannot be selected. If the decision rule is not stated, the measurement cannot fail.

### Step 2. Choose the tier from the question, not the budget

**Input:** the question from Step 1.

**Output:** a tier and a justification.

**Rationale:** the tier determines what evidence is admissible. A target-latency question requires a target or a cycle-accurate simulator. A geometry question can be answered at Tier 0. A thermal question requires a bench and a temperature sensor.

| Question type | Minimum tier | Why |
|---|---|---|
| Does the algorithm detect the event in a recorded signal? | 0 | The signal is already recorded |
| Does the firmware sample at 500 Hz with acceptable jitter? | 1 | Timing is a target property |
| Does the front end meet a noise budget? | 1 | Noise is a physical property |
| Does the robot reach the target pose? | 0 or 1 | Kinematics can be simulated; compliance needs a bench |
| Does the device meet radiated emissions? | Accredited lab | Pre-compliance is screening, not compliance |
| Does the model predict the measured trajectory? | 0 plus 1 | The anchor requires a bench point |

A Tier 0 result that is presented as a Tier 1 result is the most common overclaim in student labs. The fix is a label on every figure: "simulated" or "measured on target."

### Step 3. Build the error budget before choosing instruments

**Input:** the quantity, the range, and the uncertainty target.

**Output:** an error budget with one row per error source and a combined uncertainty.

**Rationale:** the budget tells you which instrument matters. A budget that is dominated by the ADC resolution does not need a six-digit meter. A budget that is dominated by the reference does not need a better ADC.

| Source | Type | Estimate | Distribution | Standard uncertainty | Share |
|---|---|---:|---|---:|---:|
| ADC quantization | B | LSB/√12 | Rectangular | LSB/√12 | Computed |
| ADC noise | A | Measured SD | Normal | SD/√n | Computed |
| Reference voltage | B | Datasheet | Rectangular | Value/√3 | Computed |
| Gain resistor tolerance | B | Datasheet | Rectangular | Value/√3 | Computed |
| Source amplitude | B | Generator spec | Rectangular | Value/√3 | Computed |
| Probe loading | B | Calculated | Rectangular | Value/√3 | Computed |
| Repeatability | A | Measured SD | Normal | SD/√n | Computed |

The budget is iterative. A first pass uses datasheet values. A second pass replaces the largest terms with measurements. The iteration stops when the combined uncertainty is below the target or when the remaining terms are all below one third of the largest.

### Step 4. Select instruments from the budget

**Input:** the error budget and the tier.

**Output:** a list of instruments with model, range, resolution, accuracy, and the budget term each addresses.

**Rationale:** the instrument's specification must be compared with the budget term in the same units. A meter with a 0.5% accuracy on a 3.3 V range contributes 16.5 mV, which may be negligible for a 1 V signal and dominant for a 10 mV signal.

A selection table makes the reasoning visible.

| Instrument | Key specification | Budget term | Verdict |
|---|---|---|---|
| Handheld DMM, 3.5 digit | 10 mV resolution, 0.5% + 3 counts | Reference check at 1 V | Adequate |
| Bench DMM, 5.5 digit | 1 µV resolution, 0.015% | Millivolt-level calibration | Required |
| USB scope, 2 channel, 100 MHz | 8-bit, 1 MΩ, 15 pF | Waveform shape at 1 kHz | Adequate |
| Bench scope, 4 channel, 200 MHz | 8-bit, 1 MΩ, 12 pF | Edge timing at 1 MHz | Required |
| Logic analyser, 8 channel, 24 MHz | 24 MS/s, 3.3 V threshold | UART and SPI decode | Adequate |
| Power monitor, INA219 | 12-bit, 0.5% max, 320 mV FSR | Current above 10 mA | Adequate with shunt check |
| Power monitor, PPK2 | 200 nA–1 A, 100 kS/s | Sleep current below 1 mA | Required for low power |
| Signal generator, 10 MHz | 1 mVpp–10 Vpp, 50 Ω | Frequency response | Adequate with termination |

The table is not a shopping list. It is an argument. Each row says why the instrument is sufficient or why it is not.

### Step 5. Decide what to simulate and what to measure

**Input:** the question and the tier.

**Output:** a split of the work into simulated and measured parts, with the anchor points named.

**Rationale:** simulation is cheap for parameter sweeps and expensive for fidelity. Measurement is expensive per unit and cheap for reality. The split should put the parameter sweep in simulation and the anchor in measurement.

A useful split:

| Task | Simulated | Measured | Anchor |
|---|---|---|---|
| Filter response versus component tolerance | Yes | One point at the nominal values | Cutoff frequency |
| ADC noise versus sample rate | Yes | One point at each rate | Noise SD in codes |
| Robot reach versus joint limits | Yes | One pose on the physical arm | End-effector position |
| Control-loop stability | Yes | One step response on the target | Overshoot and settling time |
| Power consumption | Partly | Yes, on the target | Current in each state |
| EMC emissions | Partly | Pre-compliance scan | One frequency with a known source |

The anchor is not a formality. It is the evidence that the model's parameters are in the right range.

### Step 6. Build the physical path

**Input:** the schematic, the BOM, and the instruments.

**Output:** an assembled path with a documented connection diagram.

**Rationale:** a measurement path is a circuit. It has loading, impedance, and return currents. The connection diagram must show the source, the DUT, the probe, the instrument, and the ground return.

Rules:

- Draw the ground return as a conductor, not as a symbol.
- Mark every impedance that is above 1 kΩ or below 10 Ω.
- Mark every capacitance that is above 10 pF in a high-impedance node.
- Show the probe's attenuation and input impedance.
- Show the instrument's input impedance and range.
- Show the source's output impedance and offset.
- Show the cable length and shielding.

The diagram is reviewed by a second person before power. A path that cannot be drawn cannot be debugged.

### Step 7. Calibrate the path end to end

**Input:** a known input and the assembled path.

**Output:** a gain, an offset, and a residual, with uncertainty.

**Rationale:** the datasheet's gain is not the path's gain. The path includes the source, the cable, the probe, and the digitiser. The end-to-end calibration is the only way to know the combined scale factor.

Procedure:

1. Apply a known DC level at the input, measured with a calibrated DMM.
2. Record the digitiser code, averaged over at least 100 samples.
3. Repeat at five levels across the range, ascending and descending.
4. Fit code = a × input + b.
5. Report the slope in codes per volt, the intercept in codes, and the residual standard deviation in codes.
6. Convert the residual to input-referred units by dividing by the slope.
7. Check the fit at a level that was not used in the fit.

The calibration is valid only for the range, the gain, the temperature, and the source impedance used. A change in any of these requires a check.

### Step 8. Measure the noise floor

**Input:** the path with a quiet input, preferably a short at the input connector.

**Output:** a noise standard deviation in codes and in input-referred units, with a bandwidth.

**Rationale:** the noise floor sets the smallest signal the path can resolve. It must be measured with the input terminated in the same impedance as the real source, because the source's impedance contributes noise.

Procedure:

1. Terminate the input with the source impedance, or short it if the source is low impedance.
2. Set the sample rate and the gain to the values that will be used.
3. Capture at least 10,000 samples.
4. Remove the mean.
5. Compute the standard deviation.
6. Compute the power spectral density if the noise is not white.
7. Report the bandwidth over which the noise was measured.
8. Repeat with the input open to see the difference.

The noise floor is not a single number. It depends on the bandwidth, the gain, the source impedance, and the sample rate. A noise specification without these conditions is not a specification.

### Step 9. Measure the bandwidth

**Input:** a swept sine or a step at the input.

**Output:** a cutoff frequency and a roll-off, with uncertainty.

**Rationale:** the bandwidth determines which signals the path can pass and which it attenuates. It also determines the noise bandwidth, which is why it belongs in the noise report.

Procedure:

1. Set the source to a sine at a low frequency, with an amplitude that does not clip.
2. Measure the output amplitude at each frequency, using the same instrument settings.
3. Sweep in logarithmically spaced steps from one decade below the expected cutoff to one decade above.
4. Find the frequency at which the amplitude falls by 3 dB.
5. Check the roll-off against the expected order.
6. Repeat at a second amplitude to check linearity.

The measurement must state the source amplitude, the load, and the instrument's bandwidth. A scope with a 100 MHz bandwidth is adequate for a 10 kHz filter; a scope with a 10 MHz bandwidth is not adequate for a 1 MHz filter.

### Step 10. Build the simulation with named parameters

**Input:** the model, the parameter values, and the solver settings.

**Output:** a model file, a parameter file, and a list of omissions.

**Rationale:** a simulation without named parameters cannot be reviewed. A parameter that is buried in a script is a parameter that cannot be changed and a result that cannot be reproduced.

The model record contains:

- The geometry or the netlist, with units.
- The parameter values, with tolerances and sources.
- The solver, the timestep, the tolerance, and the integration method.
- The initial conditions.
- The boundary conditions.
- The omissions, with a reason for each.
- The version of the simulator.

A model that does not list its omissions is claiming a fidelity it has not established.

### Step 11. Run the anchor comparison

**Input:** the bench measurement and the simulation at the same conditions.

**Output:** a difference, an uncertainty, and a verdict.

**Rationale:** the anchor comparison is the only evidence that the model is connected to reality. The comparison must be at the same conditions: same input, same load, same temperature, same bandwidth.

The comparison table:

| Quantity | Measured | Simulated | Difference | Combined uncertainty | Verdict |
|---|---:|---:|---:|---:|---|
| Cutoff frequency | 1,612 Hz | 1,591 Hz | 21 Hz | 18 Hz | Within uncertainty |
| Passband gain | 0.997 | 1.000 | −0.003 | 0.006 | Within uncertainty |
| Overshoot | 4.8% | 4.3% | 0.5 pp | 0.6 pp | Within uncertainty |
| Settling time | 3.2 ms | 3.0 ms | 0.2 ms | 0.15 ms | Marginal |
| Peak current | 41.2 mA | 40.0 mA | 1.2 mA | 1.0 mA | Marginal |

A verdict of "within uncertainty" is not a proof of the model. It is evidence that the model is not obviously wrong at this point. The extrapolation to another point is a separate claim.

### Step 12. Pin the software environment

**Input:** the analysis scripts, the firmware, the simulator, and the libraries.

**Output:** a lockfile, a container digest, an SBOM, and a build command.

**Rationale:** a result that depends on "the latest version" cannot be reproduced. Version drift is silent: the same script gives a different answer after an update.

The record contains:

- The operating system and version.
- The interpreter and version.
- The package versions, in a lockfile.
- The container image digest, if a container is used.
- The firmware commit and the build flags.
- The model hash and the conversion flags.
- The simulator version and the physics engine.
- The SBOM, with licences.

A container is not automatically reproducible. A container that installs packages at run time is not pinned. The digest must be recorded, and the build must be reproducible from the digest.

### Step 13. Run a failure drill on the stack

**Input:** the assembled stack.

**Output:** a list of failure modes with a detection method and a recovery action.

**Rationale:** a stack that has never failed is a stack whose failure modes are unknown. The drill finds the failures before the experiment does.

Drill at least:

- Disconnect the source.
- Short the input.
- Set the gain to the wrong value.
- Set the sample rate below the signal bandwidth.
- Use the wrong probe attenuation.
- Run the simulation with a timestep that is too large.
- Run the analysis with the wrong channel map.
- Change the units in one file.

For each, record how the failure appears and how it is detected. A channel swap that produces a plausible waveform is the most dangerous failure because it does not look like a failure.

### Step 14. Record the stack and hand it over

**Input:** all of the above.

**Output:** a stack record that another person can use to rebuild the bench.

**Rationale:** the stack is part of the result. A result without a stack record is a claim without a method.

The stack record contains:

- The measurement question and the decision rule.
- The tier and the justification.
- The error budget.
- The instrument list with IDs, ranges, resolutions, and calibration status.
- The connection diagram and the photographs.
- The calibration record.
- The noise and bandwidth measurements.
- The model, parameters, solver settings, and omissions.
- The anchor comparison.
- The lockfile, container digest, SBOM, and build command.
- The failure drill.
- The limitations.

### 5.15 Acceptance criteria for the stack

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Question is a quantity with a decision rule | Written question | Rewrite |
| 2 | Tier is justified | Tier table | Re-tier |
| 3 | Error budget has at least six sources | Budget table | Extend |
| 4 | Instruments are selected against the budget | Selection table | Reselect |
| 5 | Physical path is drawn with ground return | Diagram | Redraw |
| 6 | End-to-end calibration is reported | Fit and residuals | Recalibrate |
| 7 | Noise floor is measured with a stated bandwidth | Noise log | Re-measure |
| 8 | Bandwidth is measured | Sweep | Re-measure |
| 9 | Model parameters are named and sourced | Parameter file | Complete |
| 10 | Omissions are listed | Omission table | List |
| 11 | Anchor comparison is within uncertainty or explained | Comparison table | Investigate |
| 12 | Environment is pinned | Lockfile and digest | Pin |
| 13 | Failure drill is recorded | Drill table | Run |
| 14 | Stack record is complete | Record | Complete |

### 5.16 Branch points

The procedure has three branch points that change the work.

**Branch A: the question is a target-timing question.** If the claim is about latency, jitter, or energy on the target, the stack must include the target. A host simulation cannot substitute. The branch adds a logic analyser, a power monitor, and a cycle counter.

**Branch B: the question is an EMC question.** If the claim is about emissions or immunity, the stack must include a characterised victim and a defined injection. The branch adds a LISN, a CDN, or a near-field probe, and it changes the claim from "compliance" to "pre-compliance."

**Branch C: the question is a clinical question.** If the claim is about a patient population, the stack is no longer a bench stack. The branch stops the lab and hands the question to a clinical study with the appropriate approvals.

### 5.17 What the stack cannot establish

The stack produces measurements and simulations. It does not produce:

- A type test.
- A clinical validation.
- A safety-rated function.
- A regulatory submission.
- A guarantee that another unit behaves the same.

The stack record is evidence about one configuration. The limitations section is what makes it usable.

## 6. Worked example

### 6.1 The question

A team is building a 12-bit acquisition path for an ECG front end. The ADC has a 3.3 V nominal reference, a programmable gain amplifier set to 100, and a sample rate of 1 kS/s. The team wants to know the input-referred noise floor and the effective number of bits (ENOB), and whether the path meets a 5 µV RMS input-referred noise budget over the 0.5–40 Hz band.

The simulation predicts a quantization-limited noise floor of 2.34 µV RMS. The bench measurement is intended to test that prediction.

### 6.2 The setup

| Item | Value | Note |
|---|---|---|
| ADC | 12-bit successive approximation, on an STM32 board | Internal reference |
| Reference voltage | 3.296 V measured with a 5.5-digit DMM | 0.015% + 2 µV |
| Gain | 100, set by two 1% resistors | Nominal |
| Sample rate | 1 kS/s, timer-driven | Crystal-derived |
| Input | Shorted to mid-rail at the ADC pin | Source noise excluded |
| Capture length | 10,000 samples | 10 s |
| Analysis band | 0.5–40 Hz, Butterworth 4th order, zero-phase | Applied offline |
| Temperature | 23.1 °C | Logged |

The reference voltage is measured because the nominal 3.3 V is not the actual value. The input is shorted at the ADC pin because the question is about the ADC and the digitiser, not about the source.

### 6.3 The measurement

The capture is analysed in Python. The mean is removed, the samples are filtered to the 0.5–40 Hz band, and the standard deviation is computed.

| Quantity | Value | Unit |
|---|---:|---|
| Mean code | 2047.6 | codes |
| Standard deviation, raw | 2.41 | codes |
| Standard deviation, filtered | 1.70 | codes |
| Full-scale range | 4095 | codes |
| LSB size | 0.805 mV | V |
| Input-referred LSB | 8.05 µV | V |
| Input-referred noise, filtered | 13.7 | µV RMS |

The filtered standard deviation is lower than the raw value because the filter removes the out-of-band noise. This is correct for the stated band, but the raw value is the one that matters for aliasing and for the ADC's own noise specification.

The input-referred noise is:

\[
v_n = \sigma_{\text{code}} \times \frac{V_{\text{ref}}}{2^{12}-1} \div G
= 1.70 \times \frac{3.296}{4095} \div 100
= 1.70 \times 8.05\ \mu\text{V} = 13.7\ \mu\text{V RMS}
\]

### 6.4 The ENOB

The ENOB is computed from the signal-to-noise ratio for a full-scale sine. The full-scale sine has an RMS amplitude of:

\[
V_{\text{fs,rms}} = \frac{3.296}{2\sqrt{2}} = 1.165\ \text{V}
\]

The noise in codes is 1.70 codes, which is 1.70 × 0.805 mV = 1.369 mV RMS at the ADC input. The SNR is:

\[
\text{SNR} = 20 \log_{10}\left(\frac{1.165}{1.369 \times 10^{-3}}\right) = 20 \log_{10}(851) = 58.6\ \text{dB}
\]

The ENOB is:

\[
\text{ENOB} = \frac{\text{SNR} - 1.76}{6.02} = \frac{58.6 - 1.76}{6.02} = 9.44\ \text{bits}
\]

The path has 12 bits of resolution and 9.44 bits of effective performance in the 0.5–40 Hz band. The difference is the noise and distortion that the ideal quantizer does not have.

### 6.4a What the ENOB does and does not mean

The ENOB is a summary of the signal-to-noise-and-distortion ratio under a defined test condition. It is not a guarantee that every signal is resolved to 9.44 bits. A small signal near the noise floor is resolved to less. A signal at a frequency where the amplifier distorts is resolved to less. A signal that is sampled at a different rate has a different noise bandwidth and a different ENOB.

The number is useful for comparing two configurations under the same conditions. It is not useful for predicting the performance of a different signal.

### 6.5 The uncertainty

The standard deviation of 1.70 codes is estimated from 10,000 samples. The relative standard uncertainty of a standard deviation estimate is approximately \(1/\sqrt{2(n-1)}\):

\[
\frac{u_\sigma}{\sigma} = \frac{1}{\sqrt{2 \times 9999}} = 0.00707
\]

So the standard uncertainty of the code-domain noise is 0.012 codes, and the expanded uncertainty at k = 2 is 0.024 codes, or 1.4%.

The reference voltage uncertainty is 0.015% of 3.296 V plus 2 µV, which is 0.494 mV + 0.002 mV = 0.496 mV, rectangular, standard uncertainty 0.286 mV, or 0.0087% relative.

The gain uncertainty is set by two 1% resistors. For a non-inverting amplifier with gain \(1 + R_f/R_g\), the sensitivity to each resistor is not 1. A conservative bound is 2% relative, rectangular, standard uncertainty 1.15%.

The combined relative uncertainty of the input-referred noise is dominated by the gain:

\[
\frac{u_{v_n}}{v_n} = \sqrt{0.014^2 + 0.000087^2 + 0.0115^2} = 0.0182
\]

So the input-referred noise is 13.7 µV ± 0.5 µV (k = 2). The budget is dominated by the gain tolerance, not by the noise estimate.

### 6.6 Does the path meet the budget?

The budget is 5 µV RMS input-referred over 0.5–40 Hz. The measured value is 13.7 µV RMS. The path fails the budget by a factor of 2.7.

| Quantity | Value | Unit | Comparison |
|---|---:|---|---|
| Measured input-referred noise | 13.7 | µV RMS | — |
| Budget | 5.0 | µV RMS | Fail by 8.7 µV |
| Expanded uncertainty (k = 2) | 0.5 | µV RMS | Cannot explain the gap |
| Simulated quantization noise | 2.34 | µV RMS | Underpredicts by 5.9× |

The gap is too large to be explained by the uncertainty. The simulation's ideal-quantizer model is not the dominant noise source. The bench measurement is telling the team something the simulation did not include.

### 6.6a Where the extra noise comes from

The ideal quantization noise for a 12-bit converter is:

\[
v_q = \frac{\text{LSB}}{\sqrt{12}} = \frac{8.05\ \mu\text{V}}{3.464} = 2.32\ \mu\text{V RMS}
\]

The measured 13.7 µV is 5.9 times larger. The difference must come from the ADC's own input noise, the reference noise, the amplifier noise, or the board's digital coupling. A quick check separates them:

| Test | Change | Measured noise | Interpretation |
|---|---|---:|---|
| Baseline | Input shorted at the ADC pin | 13.7 µV | Total |
| Reference bypassed with 10 µF | Added capacitor | 12.9 µV | Reference noise is small |
| Amplifier input shorted | Gain stage disconnected | 4.1 µV | ADC and board contribute 4.1 µV |
| Board in reset, ADC clock off | No conversion | 0.8 µV | Digital coupling contributes 3.3 µV |
| Battery supply instead of USB | Supply change | 11.2 µV | Supply coupling contributes 2.5 µV |

The table is diagnostic, not definitive. It shows that the amplifier and the board together account for most of the excess. The next step is to measure the amplifier's input-referred noise separately and compare it with the datasheet.

This is the value of the anchor comparison. It did not prove the model; it found the model's missing term.

### 6.7 Two checks that would invalidate the result

**Check 1: the input was shorted at the ADC pin, not at the source.** The measurement excludes the source impedance and the cable. If the real source has a 10 kΩ impedance, its thermal noise at 23 °C over a 40 Hz bandwidth is:

\[
v_{n,\text{source}} = \sqrt{4 k_B T R \Delta f}
= \sqrt{4 \times 1.381 \times 10^{-23} \times 296 \times 10^4 \times 40}
= 2.56\ \mu\text{V RMS}
\]

The source would add 2.56 µV in quadrature, raising the total to \(\sqrt{13.7^2 + 2.56^2} = 13.9\ \mu\text{V}\), a 1.8% increase. The conclusion would not change, but the reported number would be wrong for the real source. The check is to repeat the measurement with the source impedance connected.

**Check 2: the filter's passband and stopband were not verified.** The 0.5–40 Hz Butterworth filter was applied offline with a zero-phase implementation. If the filter's actual cutoff is lower than 40 Hz, the measured noise is lower than the true in-band noise. If the filter's stopband is inadequate, the measured noise includes out-of-band content. A filter that is not verified is an assumption, not a measurement. The check is to apply a known white-noise input and measure the filter's transfer function, or to compute the equivalent noise bandwidth from the coefficients and compare it with the nominal 40 Hz.

### 6.8 A varied condition: the sample rate is raised to 10 kS/s

The team repeats the measurement at 10 kS/s, keeping the same analog bandwidth and the same offline filter.

| Condition | Sample rate | Raw SD | Filtered SD | Input-referred noise | ENOB |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 kS/s | 2.41 codes | 1.70 codes | 13.7 µV | 9.44 bits |
| Varied | 10 kS/s | 6.83 codes | 1.72 codes | 13.8 µV | 9.43 bits |

The raw standard deviation rises by a factor of 2.83, which is close to the square root of 10, as expected for white noise sampled over a wider band. The filtered standard deviation is essentially unchanged, because the offline filter removes the additional out-of-band noise. The input-referred in-band noise is stable at 13.7–13.8 µV.

The result moves in the way the model predicts for the raw data and does not move for the filtered data. That is a consistency check on the analysis. It also shows why a noise specification must state the bandwidth: the same path has a raw noise of 2.41 codes at 1 kS/s and 6.83 codes at 10 kS/s, and both are correct.

### 6.8a What would change the conclusion

The conclusion is that the path fails a 5 µV budget. Three changes would alter it.

| Change | Effect on the measured noise | Effect on the conclusion |
|---|---|---|
| Reduce the gain from 100 to 10 | Input-referred noise rises by 10× if the ADC noise dominates | Fails by more; the gain is not the problem |
| Increase the gain from 100 to 1000 | Input-referred noise falls until the amplifier's own noise dominates | May pass, but the amplifier noise sets a floor |
| Average 16 consecutive samples | Noise falls by 4× if the noise is white and the signal is slow | May pass, but the averaging adds latency and does not remove interference |
| Use a 16-bit ADC | Quantization noise falls by 16×, but the analog noise remains | Fails; the bottleneck is analog, not resolution |

The table is the reason a noise budget must be built from the analog chain, not from the converter's bit count. A 16-bit converter with a noisy front end is a 9-bit measurement system with a higher price.

### 6.9 What the example demonstrates

The arithmetic closes: an LSB of 8.05 µV, an ideal quantization noise of 2.32 µV, a measured in-band noise of 13.7 µV, an ENOB of 9.44 bits, and an uncertainty of 0.5 µV dominated by the gain tolerance. The path fails a 5 µV budget by a factor of 2.7.

The example also demonstrates the stack method. The simulation was useful because it predicted the quantization floor and was wrong about the total. The bench measurement was useful because it found the missing term. The anchor comparison was useful because it turned a disagreement into a diagnostic table. A simulation without the anchor would have produced a confident, wrong answer.

## 7. Code, query, or calculation

The program below reproduces the worked example and adds a Monte Carlo check of the ideal quantizer. It uses only the Python standard library. The input is a synthetic capture with a known noise level, so the program runs without hardware. The same functions accept a real capture loaded from a CSV.

```python
import math
import random
import statistics

VREF = 3.296          # measured reference, V
BITS = 12
GAIN = 100.0
FS = 1000.0           # sample rate, S/s
N = 10000             # samples
SEED = 20261026

def quantize(v, vref=VREF, bits=BITS):
    lsb = vref / (2**bits - 1)
    code = round(v / lsb)
    return max(0, min(2**bits - 1, code))

def simulate(n=N, sigma_v=0.0, seed=SEED):
    """Simulate a mid-rail input with Gaussian noise, then quantize."""
    rng = random.Random(seed)
    lsb = VREF / (2**BITS - 1)
    mid = VREF / 2
    codes = []
    for _ in range(n):
        v = mid + rng.gauss(0.0, sigma_v)
        codes.append(quantize(v))
    return codes

def stats(codes):
    m = statistics.fmean(codes)
    sd = statistics.stdev(codes)
    return m, sd

def quantization_error_sd(n=N, cycles=101.0):
    """Drive a full-scale sine and measure the quantizer error in codes."""
    lsb = VREF / (2**BITS - 1)
    errs = []
    for i in range(n):
        v = VREF / 2 + 0.49 * VREF * math.sin(2 * math.pi * cycles * i / n)
        errs.append((v - quantize(v) * lsb) / lsb)
    return statistics.stdev(errs)

lsb = VREF / (2**BITS - 1)
sd_q = quantization_error_sd()
print(f"LSB                    = {lsb*1e3:.4f} mV")
print(f"ideal quantization sd  = {sd_q:.4f} codes")
print(f"ideal input-referred   = {sd_q*lsb/GAIN*1e6:.2f} uV RMS")

# --- measured case: solve for the analog sigma that yields 1.70 codes ---
# The benchmark capture had a filtered sd of 1.70 codes. The quantizer
# adds its own variance, so the analog sigma must be slightly smaller.
target_codes = 1.70
lo, hi = 0.0, 3.0
for _ in range(60):
    mid = (lo + hi) / 2
    sd_try = stats(simulate(sigma_v=mid * lsb))[1]
    if sd_try < target_codes:
        lo = mid
    else:
        hi = mid
sigma_codes = (lo + hi) / 2
measured = simulate(sigma_v=sigma_codes * lsb)
m_m, sd_m = stats(measured)
print(f"model analog sigma     = {sigma_codes*lsb*1e6:.2f} uV at ADC input")
print(f"quantized sd           = {sd_m:.3f} codes")

# --- ENOB from code-domain noise ---
fs_rms = VREF / (2 * math.sqrt(2))
noise_v = sd_m * lsb
snr_db = 20 * math.log10(fs_rms / noise_v)
enob = (snr_db - 1.76) / 6.02
print(f"full-scale sine RMS    = {fs_rms:.4f} V")
print(f"noise at ADC input     = {noise_v*1e3:.4f} mV RMS")
print(f"SNR                    = {snr_db:.2f} dB")
print(f"ENOB                   = {enob:.2f} bits")

# --- input-referred noise and uncertainty ---
vn_uv = sd_m * lsb / GAIN * 1e6
u_stat = vn_uv / math.sqrt(2 * (N - 1))
u_ref = 0.000087 * vn_uv
u_gain = 0.02 / math.sqrt(3) * vn_uv
u_c = math.hypot(math.hypot(u_stat, u_ref), u_gain)
print(f"input-referred noise   = {vn_uv:.2f} uV RMS")
print(f"u_stat                 = {u_stat:.4f} uV")
print(f"u_ref                  = {u_ref:.4f} uV")
print(f"u_gain                 = {u_gain:.3f} uV")
print(f"combined u_c           = {u_c:.3f} uV")
print(f"expanded U (k=2)       = {2*u_c:.3f} uV")

# --- budget check ---
budget = 5.0
verdict = "PASS" if vn_uv <= budget else "FAIL"
print(f"budget {budget:.1f} uV RMS  ->  {verdict} (margin {budget-vn_uv:+.1f} uV)")

# --- sample-rate scaling check for white noise ---
# Measured in the worked example: raw sd 2.41 codes at 1 kS/s,
# 6.83 codes at 10 kS/s, same analog band limit.
raw_1k = 2.41
raw_10k_measured = 6.83
predicted_10k = raw_1k * math.sqrt(10)
print(f"predicted raw sd at 10 kS/s = {predicted_10k:.2f} codes")
print(f"measured raw sd at 10 kS/s  = {raw_10k_measured:.2f} codes")
print(f"measured ratio              = {raw_10k_measured/raw_1k:.3f} (sqrt(10) = 3.162)")
```

Expected output:

```text
LSB                    = 0.8049 mV
ideal quantization sd  = 0.2870 codes
ideal input-referred   = 2.31 uV RMS
model analog sigma     = 1337.57 uV at ADC input
quantized sd           = 1.700 codes
full-scale sine RMS    = 1.1653 V
noise at ADC input     = 1.3682 mV RMS
SNR                    = 58.61 dB
ENOB                   = 9.44 bits
input-referred noise   = 13.68 uV RMS
u_stat                 = 0.0967 uV
u_ref                  = 0.0012 uV
u_gain                 = 0.158 uV
combined u_c           = 0.185 uV
expanded U (k=2)       = 0.371 uV
budget 5.0 uV RMS  ->  FAIL (margin -8.7 uV)
predicted raw sd at 10 kS/s = 7.62 codes
measured raw sd at 10 kS/s  = 6.83 codes
measured ratio              = 2.834 (sqrt(10) = 3.162)
```

The program's ideal quantization standard deviation is 0.287 codes, which is close to the analytic LSB/√12 = 0.2898 codes; the small difference is the finite record and the non-integer number of sine cycles. The measured case is solved to reproduce 1.70 codes, so the agreement is by design; the purpose is to show the arithmetic that converts codes to microvolts and to ENOB. The uncertainty is dominated by the gain tolerance, as in the worked example.

The last three lines check the sample-rate scaling. For white noise, the raw standard deviation scales with the square root of the bandwidth, so a tenfold increase in sample rate would produce a factor of 3.162. The measured factor in the worked example was 6.83/2.41 = 2.83, which is lower than 3.162 because the analog front end band-limits the noise before the ADC. This is a useful diagnostic: a ratio close to 3.16 indicates white noise at the ADC input; a ratio below it indicates analog bandwidth limiting; a ratio above it indicates a digital or clock-related noise source.

## 8. Data, standards, and regulatory context

### 8.1 Instrument safety standards

Bench instruments are governed by IEC 61010-1, the safety standard for electrical equipment for measurement, control, and laboratory use. Edition 3.0 was published in 2010, with Amendment 1 in 2016. The measurement-category requirements are in IEC 61010-2-030.

The category rating is a property of the instrument's input circuit, not of the instrument's price. A CAT II 300 V meter is adequate for a bench supplied by a SELV adapter. It is not adequate for a CAT III distribution board. The test leads must have a rating at least equal to the instrument's.

In the United States, the relevant product standard is UL 61010-1, and the National Recognised Testing Laboratory (NRTL) mark indicates that a sample has been tested. A mark is not a calibration certificate, and it does not cover a modified instrument.

### 8.2 Calibration and traceability

ISO/IEC 17025:2017 is the competence standard for testing and calibration laboratories. Clause 6.5 requires metrological traceability through a documented unbroken chain of calibrations, each contributing to the measurement uncertainty. Clause 7.8.4 specifies the content of a calibration certificate, including the measurement uncertainty, the environmental conditions, and a statement of traceability.

The practical consequence for a lab stack is that an instrument's calibration status has three parts: the certificate covers the range and parameter used, the uncertainty is stated, and the traceability chain is documented. A sticker with a date is not enough.

The Guide to the Expression of Uncertainty in Measurement (JCGM 100:2008, commonly called the GUM) defines the Type A and Type B evaluation, the combined standard uncertainty, and the expanded uncertainty with a coverage factor. The worked example uses the GUM method: Type A for the repeatability, Type B for the datasheet and tolerance terms, combined in quadrature, expanded at k = 2.

### 8.3 ADC test methods

The Institute of Electrical and Electronics Engineers (IEEE) publishes standards for digitising waveform recorders (IEEE 1057) and for analog-to-digital converter terminology and test methods (IEEE 1241). These define the sine-fit, the histogram, and the spectral methods for measuring ENOB, SINAD, and noise.

A lab that reports ENOB should state which method it used. The worked example uses a noise-only estimate, not a full sine-fit. That is adequate for a noise budget and inadequate for a datasheet comparison.

### 8.4 Simulation credibility

ASME V&V 40 is a standard for assessing the credibility of computational models used in medical device design. It defines the context of use, the model risk, and the credibility activities. It does not prescribe a single method; it requires an argument proportionate to the risk.

The Food and Drug Administration (FDA) has published guidance on computational modelling and on physiologically closed-loop control. The closed-loop guidance, issued in September 2023, specifically addresses hardware-in-the-loop testing and requires that the patient model not introduce delays that change the system's behaviour. That requirement is the regulatory expression of the stack principle: the test rig is part of the measurement.

### 8.5 Robotics software conventions

ROS 2 uses the conventions in REP 103 (units and axes) and REP 105 (coordinate frames for mobile platforms). REP 103 specifies SI units, a right-handed coordinate system, x forward, y left, z up, and quaternion order x, y, z, w. REP 105 defines the frame tree from earth to map to odom to base_link.

ROS 2 Jazzy Jalisco was released on 23 May 2024 as a long-term-support distribution. Its recommended simulator pairing is Gazebo Harmonic, bridged through the ros_gz packages. The release notes name the supported platforms and the new features. A lab that pins Jazzy must also pin the Gazebo version and the bridge version, because the bridge is a separate package with its own release cadence.

The Unified Robot Description Format (URDF) describes links, joints, and geometry. It is not a dynamics model. MuJoCo uses its own XML format (MJCF), and Gazebo uses the Simulation Description Format (SDF). A conversion between formats can change the inertia, the joint damping, and the contact model. The differences must be recorded.

### 8.6 Software supply chain

A software bill of materials (SBOM) lists the components in a build, with versions and licences. The FDA's cybersecurity guidance treats an SBOM as part of the premarket submission for cyber devices, and IEC 81001-5-1 defines security activities across the health software lifecycle.

For a lab, the SBOM is a practical tool: it tells the next person what was in the build. A lockfile is the minimum; an SBOM is the reviewable form.

### 8.7 File formats and metadata

A measurement is only as useful as its file. The file must carry enough metadata to interpret the samples without the person who recorded them.

| Field | Why it is needed | Common omission |
|---|---|---|
| Sample rate | Converts sample index to time | Stored in the filename only |
| Time base | Absolute time of the first sample | No timestamp at all |
| Channel map | Which physical signal is in which column | Assumed from the column order |
| Units | Converts counts to volts or amperes | Raw codes with no scale |
| Scale factor | Gain and reference used | Applied later and forgotten |
| Range | Instrument setting | Lost when the instrument is reconfigured |
| Filter | Any analog or digital filtering | Applied twice or not at all |
| Trigger | What started the capture | Unknown pre-trigger history |
| Operator | Who recorded it | Missing |
| Instrument ID | Which unit produced it | Missing |
| Checksum | Detects corruption | Rarely included |

The raw file should be immutable. The processed file should be reproducible from the raw file and a script. A CSV with a header comment is adequate for a small capture; a binary format with a schema is better for a long one. The choice matters less than the metadata.

A SHA-256 hash of the raw file belongs in the report. The hash is not a security measure in a teaching lab; it is a way to prove that the analysed file is the recorded file.

### 8.8 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| IEC 61010-1 | Product safety standard | Lab instruments | A custom DUT |
| ISO/IEC 17025 | Laboratory competence standard | Accredited calibration and testing | A student lab |
| JCGM 100:2008 | International guidance | Uncertainty statements in metrology | A rough estimate |
| IEEE 1057 / 1241 | Engineering standards | ADC and recorder test methods | A noise budget |
| ASME V&V 40 | Consensus standard | Model credibility for medical devices | A teaching simulation |
| REP 103 / 105 | ROS conventions | ROS coordinate frames and units | Non-ROS software |
| IEC 81001-5-1 | International standard | Health software security lifecycle | A prototype build |

## 9. Industry practice and case evidence

### 9.1 How device companies build a bench

A device company typically separates three benches: an engineering bench, a verification bench, and a production bench.

- The **engineering bench** is flexible. It has a scope, a supply, a function generator, and a set of probes. It is used for debug and exploration. Its results are not verification evidence.
- The **verification bench** is controlled. It has calibrated instruments, a written procedure, a fixture, and a data-collection script. Its results go into the design history file.
- The **production bench** is fast and repeatable. It has a fixture, a tester, and a pass/fail limit. It runs on every unit.

The same measurement can be made on all three benches, and only the verification bench produces evidence that a reviewer will accept. The difference is not the instrument quality; it is the control of the configuration, the calibration, and the procedure.

### 9.2 The cost of a repeated test-house booking

An accredited EMC test is booked by the day, with a chamber and a technician. A repeat booking costs the day rate plus the schedule delay. The industry practice is to run a pre-compliance scan on an engineering bench before the booking, fix the obvious failures, and reserve the chamber for the confirmation.

The pre-compliance scan is not a compliance test. It uses a LISN, a spectrum analyser, and a near-field probe, and it measures relative changes. Its value is in finding the 6 dB failure before the chamber finds it.

### 9.3 Open-source and commercial simulation

The robotics stack has both open-source and commercial options. Gazebo and MuJoCo are open-source physics simulators with different model formats and solver behaviour. Pinocchio is an open-source rigid-body dynamics library. MoveIt 2 is an open-source motion-planning framework. Commercial platforms such as dSPACE and Speedgoat provide real-time hardware and validated I/O for hardware-in-the-loop work.

The choice is not about which is better in general. It is about which is adequate for the question and which can be validated. A lab that uses Gazebo for a contact-rich manipulation question may find that the contact model dominates the result; a lab that uses MuJoCo for a kinematics question is using more fidelity than it needs.

### 9.4 The reproducibility problem in research labs

A widely discussed problem in computational research is that published results often cannot be reproduced because the code, the data, or the environment is missing or has changed. The lab response is the stack record: a lockfile, a container digest, a BOM, and a script that runs from raw data to the reported figure.

The record does not guarantee reproducibility. It makes the attempt possible. A reviewer who can rebuild the environment can find the difference; a reviewer who cannot is left with a claim.

### 9.5 Instrument practice in industry

Test-equipment manufacturers publish application notes on the errors that matter: probe loading, ground-lead inductance, burden voltage, and the difference between resolution and accuracy. The notes are practical because the errors are common.

A useful habit from industry practice is the "known-good" check: before measuring an unknown, measure a known. A 1 kHz sine of known amplitude through the same probe and scope confirms the attenuation, the termination, and the channel map. The check takes a minute and catches the most common setup errors.

### 9.6 What a due-diligence reviewer checks in a stack record

A technical due-diligence reviewer who is handed a lab result usually starts with the stack, not the number. The questions below are a composite of common review practice; the list is an inference from how verification evidence is audited, not a published checklist.

1. What exactly was measured, on which unit, and at which revision?
2. Which instrument produced the number, and what is its calibration status?
3. What is the uncertainty, and which term dominates it?
4. Was the result compared with a requirement, and was the requirement written before the test?
5. Is the raw data retained, and can the analysed file be traced to it?
6. Is the software environment pinned, and can the analysis be rerun?
7. What does the result not prove?
8. If a simulation is involved, where is the anchor measurement?
9. If the claim is about the target, was the target in the loop?
10. What changed between the version in the report and the version in the repository?

A record that answers these questions survives a review. A record that answers only the first two is a claim with a number attached.

### 9.7 Inference

It is an inference, not a documented finding, that teams which write the error budget before buying instruments make fewer repeat purchases. The reasoning is that the budget names the parameter, and the parameter names the instrument. The inference is consistent with the tier model in the survey file, but it has not been tested in this course.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Instrument chosen before the question | "We have a scope, so let's use it" | No error budget | Wrong measurement | Write the quantity first |
| 2 | Resolution confused with accuracy | "It is 12-bit, so it is accurate to 1 part in 4096" | Quantization is not the only error | False precision | Separate the budget terms |
| 3 | Probe loading ignored | Amplitude drops when the probe is connected | Capacitive and resistive loading | Wrong amplitude and bandwidth | Measure with and without the probe |
| 4 | Ground lead too long | Ringing on fast edges | Lead inductance | False transient | Short ground, tip-and-barrel |
| 5 | Sample rate below signal bandwidth | Aliased tone that looks real | No anti-alias filter | Wrong spectrum | Filter before sampling |
| 6 | Simulation used as evidence | Figure labelled "result" with no bench point | No anchor | Confident wrong answer | Anchor comparison |
| 7 | Model omissions unstated | "The model is accurate" | Hidden assumptions | Overclaim | Omission table |
| 8 | Unpinned toolchain | "It worked last semester" | Version drift | Unreproducible result | Lockfile and digest |
| 9 | Wrong channel map | Plausible but shifted signal | Off-by-one in the file | Wrong attribution | Known-good check |
| 10 | Burden voltage ignored | Current reading lower than expected | Shunt drop reduces DUT voltage | Wrong power | Measure the DUT voltage |
| 11 | Calibration out of range | Certificate covers a different range | Extrapolated correction | Unknown error | Check the range |
| 12 | Simulation timestep too large | Energy grows or oscillates | Instability | Wrong trajectory | Timestep study |
| 13 | Units mixed in a model | Robot flies apart | Metres versus millimetres | Invalid result | Unit check in the model |
| 14 | Container not pinned | Build downloads "latest" | Supply-chain drift | Unreproducible build | Digest and lockfile |
| 15 | No known-good check | First measurement is the experiment | Setup error undetected | Wasted run | Measure a known first |
| 16 | BOM missing tolerances | Replacement part changes behaviour | Component variation | Irreproducible bench | BOM with tolerances |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The instrument's specification applies at the used range and temperature | The budget uses it | Wrong range, temperature drift | Read the certificate; check the range |
| 2 | The source impedance is known | It affects loading and noise | A high-impedance sensor, a long cable | Measure the impedance |
| 3 | The probe's loading is negligible or corrected | The path is assumed not to change the DUT | High-impedance node, high frequency | Measure with and without |
| 4 | The ADC's reference is the measured value | The LSB calculation depends on it | Reference drift, load regulation | Measure the reference at the ADC pin |
| 5 | The noise is stationary over the capture | The standard deviation is meaningful | Drift, interference bursts, warm-up | Plot the trend; repeat |
| 6 | The filter's response is as designed | The band-limited noise depends on it | Coefficient errors, numerical issues | Measure the filter response |
| 7 | The simulator's parameters are in the right range | The anchor depends on it | Wrong inertia, wrong capacitance | Compare with a measurement |
| 8 | The anchor point is representative | The extrapolation depends on it | Nonlinear behaviour away from the anchor | Add a second anchor |
| 9 | The software environment is pinned | Reproducibility depends on it | Floating dependencies, unpinned container | Rebuild from the lockfile |
| 10 | The model's units are consistent | The result depends on it | Mixed units, wrong frame | Unit check; dimensional analysis |
| 11 | The tier matches the claim | The evidence is admissible | Tier 0 result presented as Tier 1 | Label every figure |
| 12 | The instrument is within calibration | The uncertainty is valid | Expired certificate, wrong range | Check the register |

## 12. Exercises

### Exercise 1. Write a measurement question (application)

**Deliverable:** one measurement question for the ECG front end, stated as a quantity with a range, a bandwidth, an uncertainty target, and a decision rule.

**Acceptance criteria:** the question names the unit, the smallest value that matters, the bandwidth, the uncertainty that would change the decision, and the action that follows from each outcome; a partner can select an instrument from the question alone.

**Time:** 30 minutes.

### Exercise 2. Error budget (calculation)

**Deliverable:** an error budget for a 1 mV, 10 Hz signal measured through a 12-bit ADC with a 3.3 V reference and a gain of 10, including at least six sources.

**Acceptance criteria:** each source is classified as Type A or Type B; the combined uncertainty is computed in the correct unit; the dominant term is named; the budget is compared with a stated target and a verdict is given.

**Time:** 45 minutes.

### Exercise 3. Instrument selection (critique)

**Deliverable:** a selection table for three instruments, with the budget term each addresses and a verdict of adequate or not adequate.

**Acceptance criteria:** the table uses the same units as the budget; at least one instrument is rejected with a quantitative reason; the rejection is not based on price or brand; the accepted instrument has a stated range and resolution.

**Time:** 30 minutes.

### Exercise 4. Simulate versus measure (design)

**Deliverable:** a plan for anchoring a simulation of an RC low-pass filter to a bench measurement, including the component tolerances, the source impedance, the probe loading, and the acceptance criterion.

**Acceptance criteria:** the plan states the cutoff frequency, the tolerance range, the measurement instrument, and the comparison table; it names two omissions in the simulation; it states what the anchor does not prove.

**Time:** 45 minutes.

### Exercise 5. Known-good check (application)

**Deliverable:** a procedure for a known-good check of a scope, a probe, and a signal generator, with expected values and a pass criterion.

**Acceptance criteria:** the procedure checks amplitude, frequency, attenuation, and channel mapping; it uses a known source; it can be completed in under five minutes; it records the result in the lab log.

**Time:** 30 minutes.

### Exercise 6. Failure drill (application)

**Deliverable:** a table of eight stack failures, with the symptom, the detection method, and the recovery action.

**Acceptance criteria:** at least one failure is a channel-map error; at least one is a sample-rate error; at least one is a simulation-settings error; each detection method is a measurement or a check, not an assumption.

**Time:** 45 minutes.

### Exercise 7. Stack record (design)

**Deliverable:** a stack record for a completed lab, including the question, tier, error budget, instrument list, calibration, model, anchor, lockfile, digest, and limitations.

**Acceptance criteria:** a partner can rebuild the bench from the record; every instrument has an ID and a calibration status; the model has named parameters and omissions; the limitations section names at least three things the stack cannot establish.

**Time:** 90 minutes.

### Exercise 8. Tier dispute (critique)

**Deliverable:** a written response to a team that wants to present a simulation-only latency result as a target measurement, with the specific claims that would change and the evidence that would be needed.

**Acceptance criteria:** the response names the difference between SIL, PIL, and HIL; it identifies the claims that the simulation can and cannot support; it proposes a minimal Tier 1 experiment; it does not simply refuse.

**Time:** 30 minutes.

## 13. Self-check answers

### 13.1 What is the difference between resolution and accuracy?

Resolution is the smallest increment an instrument can display or a converter can encode. Accuracy is how close the reading is to the true value. A 12-bit converter with a 3.3 V reference has a resolution of 0.805 mV and an accuracy that depends on the reference, the gain, the linearity, and the noise. A high-resolution instrument can be inaccurate; a low-resolution instrument can be accurate within its step.

### 13.2 Compute the LSB of a 12-bit ADC with a 3.3 V reference.

\[
\text{LSB} = \frac{3.3}{2^{12} - 1} = \frac{3.3}{4095} = 0.8059\ \text{mV}
\]

With a gain of 100, the input-referred LSB is 8.06 µV.

### 13.3 Why is the ideal quantization noise LSB/√12?

The quantization error is modelled as a uniform random variable between −LSB/2 and +LSB/2. The variance of a uniform distribution over an interval of width LSB is LSB²/12, so the standard deviation is LSB/√12. This assumes the input is not correlated with the sampling clock and that the quantizer does not overload.

### 13.4 What does ENOB mean, and what does it not mean?

ENOB is the number of bits of an ideal converter that would have the same signal-to-noise-and-distortion ratio as the measured converter under the stated test condition. It summarises the combined effect of noise and distortion. It does not mean that every signal is resolved to that many bits, and it does not transfer to a different bandwidth, amplitude, or sample rate.

### 13.5 Why does the raw noise scale with the square root of the sample rate?

For white noise, the power spectral density is constant. Doubling the sample rate doubles the bandwidth, so the noise power doubles and the standard deviation increases by √2. If the analog front end limits the bandwidth below the Nyquist frequency, the increase is smaller. If the noise is not white, the scaling is different.

### 13.6 What is the difference between SIL, PIL, and HIL?

SIL runs the algorithm on a host computer with simulated inputs. PIL runs the compiled algorithm on the target processor with simulated inputs, often with instrumentation. HIL runs the target in real time with a simulated plant and real I/O, so the target's timing and interfaces are exercised. Only HIL supports a claim about target timing.

### 13.7 What must a stack record contain to be reproducible?

The physical configuration (BOM, photographs, calibration), the software configuration (lockfile, container digest, firmware commit, SBOM), and the analysis configuration (script, parameters, seed). It must also contain the question, the tier, the error budget, the anchor comparison, and the limitations.

### 13.8 A scope probe has a 15 pF input capacitance. What is its impedance at 1 MHz?

\[
Z = \frac{1}{2\pi f C} = \frac{1}{2\pi \times 10^6 \times 15 \times 10^{-12}} = 10.6\ \text{k}\Omega
\]

At 1 MHz, a 15 pF probe is a 10.6 kΩ load. On a 100 kΩ node, that is a 9.6% loading error. The probe's resistance, typically 10 MΩ, is negligible by comparison at this frequency.

### 13.9 Why is a container not automatically reproducible?

A container image can install packages at build time from a network repository, so the same Dockerfile can produce different images on different days. The image digest pins the result, but only if the digest is recorded and the image is available. A build that fetches "latest" is not pinned even if it runs inside a container.

### 13.10 Rubric for Exercise 4

A strong plan states the cutoff frequency and its tolerance range, names the source impedance and the probe capacitance, gives a comparison table with a combined uncertainty, and lists at least two simulation omissions. A weak plan gives a single simulated number and a single measured number with no uncertainty and no omissions.

### 13.11 Rubric for Exercise 7

A strong stack record can be used by a partner to rebuild the bench without asking questions. It has instrument IDs, calibration status, a wiring diagram, a model with named parameters, an anchor comparison, a lockfile, a digest, and a limitations section. A weak record lists the software versions but not the hardware, or reports a result without the calibration record.

### 13.12 Rubric for Exercise 8

A strong response distinguishes SIL, PIL, and HIL, names the latency claim that requires HIL, proposes a minimal experiment with a logic analyser and a GPIO toggle, and states what the simulation can still contribute. A weak response either refuses the simulation entirely or accepts it as equivalent to a target measurement.

### 13.13 What is the difference between a lockfile and an SBOM?

A lockfile records the exact versions of the dependencies that were resolved for a build, so the environment can be recreated. An SBOM records the components in the build, with versions and licences, in a form that can be reviewed and shared. The lockfile is for the builder; the SBOM is for the reviewer and the security process. A lab needs both, and neither replaces the container digest.

A lockfile without a digest pins the packages but not the base image. An SBOM without a lockfile lists the components but not the resolution. A digest without either pins the artefact but not the licence or the vulnerability status. The three are complementary records of the same build.

## 14. Interview questions

### 14.1 How do you choose an instrument for a measurement?

A strong answer starts from the measurand, the range, the bandwidth, and the uncertainty that would change the decision. It builds an error budget, compares each instrument's specification with the budget term, and selects the least expensive instrument that satisfies the budget. A weak answer starts from a brand or a price.

### 14.2 What is burden voltage, and why does it matter?

Burden voltage is the voltage drop across the shunt or the ammeter used to measure current. It reduces the voltage available to the DUT and can change the DUT's behaviour. A strong answer gives an example: a 0.1 Ω shunt at 100 mA drops 10 mV, which is negligible for a 5 V rail and significant for a 1.2 V rail.

### 14.3 Why can a 12-bit ADC have only 9 effective bits?

Because the effective number of bits includes noise, distortion, reference error, and nonlinearity, not only quantization. A strong answer computes the ideal quantization noise, compares it with the measured noise, and identifies the dominant extra term. A weak answer says "the ADC is cheap."

### 14.4 What is the difference between a simulation and a measurement?

A simulation evaluates a model; a measurement evaluates the world through an instrument. A strong answer notes that the model has omissions and the instrument has uncertainty, and that the two must be compared at an anchor point before either is used to predict the other. A weak answer says a simulation is "less accurate."

### 14.5 How do you validate a simulation?

A strong answer names the anchor comparison: same input, same conditions, same bandwidth, with a combined uncertainty and a verdict. It notes that one anchor does not validate extrapolation, and that a second anchor at a different operating point is needed for a stronger claim. A weak answer says "it matches the literature."

### 14.6 What is the difference between jitter and latency?

Latency is the mean delay from input to output. Jitter is the variation around that delay. A strong answer notes that a system can have a stable latency with high jitter, or a variable latency with low jitter, and that both matter for a control loop. A weak answer uses the terms interchangeably.

### 14.7 How would you check a scope and probe before a measurement?

A strong answer applies a known sine from a signal generator, checks the amplitude against the generator setting, checks the frequency, checks the attenuation ratio, and confirms the channel map. It records the result. A weak answer says "I would press autoset."

### 14.8 What is an SBOM, and why does a lab need one?

An SBOM is a list of the software components in a build, with versions and licences. A lab needs one to know what was in the build, to reproduce it, and to assess a vulnerability. A strong answer connects it to the FDA cybersecurity guidance and IEC 81001-5-1. A weak answer calls it a list of libraries.

### 14.9 What is the difference between Tier 0 and Tier 1 evidence?

Tier 0 is simulation only; Tier 1 includes a physical bench. A strong answer says Tier 0 can support an algorithmic or geometric claim and cannot support a claim about target timing, target energy, or real analog noise. A weak answer says Tier 1 is "better."

### 14.10 How do you handle a simulation that disagrees with the bench?

A strong answer treats the disagreement as a diagnostic: it checks the anchor conditions, the units, the parameters, and the instrument, then isolates the missing term with a targeted experiment. It does not adjust the model to fit the data without recording the change. A weak answer either trusts the simulation or discards it.

### 14.11 What does "reproducible" mean in a lab?

A strong answer says another person can rebuild the physical, software, and analysis configuration and obtain a result within the stated uncertainty. It notes that reproducibility is a property of the whole record, not of the code alone. A weak answer says "the code is on GitHub."

### 14.12 What would you do if a tool's "latest" version broke your analysis?

A strong answer pins the previous version, records the change, and reruns the analysis to confirm the result. It then investigates whether the new version changes a default that affects the result, and documents the decision. A weak answer upgrades and reports the new number without checking.

## 15. Key terms

- **AFE** — analog front end; the circuit that conditions a sensor signal.
- **Aliasing** — folding of frequencies above Nyquist into the baseband.
- **Anchor comparison** — measurement used to check a model at a defined condition.
- **Bandwidth** — frequency at which the response falls by 3 dB.
- **BOM** — bill of materials, with part numbers and tolerances.
- **Burden voltage** — voltage drop across a current-measuring shunt.
- **Calibration** — comparison against a reference with a stated uncertainty.
- **CDN** — coupling/decoupling network.
- **CMRR** — common-mode rejection ratio.
- **Container digest** — content hash that pins a container image.
- **DMM** — digital multimeter.
- **DUT** — device under test.
- **ENOB** — effective number of bits.
- **EUT** — equipment under test.
- **FMI** — Functional Mock-up Interface.
- **GUM** — Guide to the Expression of Uncertainty in Measurement.
- **HIL** — hardware-in-the-loop.
- **Input impedance** — resistance and capacitance seen at an input.
- **Jitter** — variation in event timing.
- **Latency** — mean delay from input to output.
- **LISN** — line impedance stabilisation network.
- **Lockfile** — record of exact dependency versions.
- **LSB** — least significant bit.
- **MIL** — model-in-the-loop.
- **MJCF** — MuJoCo XML model format.
- **Nyquist frequency** — half the sample rate.
- **PIL** — processor-in-the-loop.
- **Probe attenuation** — ratio of probe input to instrument input.
- **PSRR** — power-supply rejection ratio.
- **REP 103** — ROS convention for units and axes.
- **Reproducibility** — ability to obtain the same result from the same inputs.
- **Resolution** — smallest distinguishable increment.
- **SBOM** — software bill of materials.
- **SDF** — Simulation Description Format used by Gazebo.
- **SIL** — software-in-the-loop.
- **SINAD** — signal-to-noise-and-distortion ratio.
- **SNR** — signal-to-noise ratio.
- **Stack record** — document that allows a bench to be rebuilt.
- **Traceability** — unbroken chain of calibrations.
- **Trigger** — condition that starts a capture.
- **URDF** — Unified Robot Description Format.
- **Zero-phase filter** — filter applied forward and backward so that it introduces no phase shift.
- **Noise floor** — the standard deviation of the output with a quiet input, over a stated bandwidth.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and metrology

- IEC 61010-1:2010+AMD1:2016, Safety requirements for electrical equipment for measurement, control, and laboratory use. https://webstore.iec.ch/en/publication/4279
- IEC 61010-2-030, Particular requirements for testing and measuring circuits. https://webstore.iec.ch/
- ISO/IEC 17025:2017, General requirements for the competence of testing and calibration laboratories. https://www.iso.org/standard/66912.html
- JCGM 100:2008, Evaluation of measurement data — Guide to the expression of uncertainty in measurement. https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf
- NIST, Metrological Traceability: Frequently Asked Questions and NIST Policy. https://www.nist.gov/metrology/metrological-traceability
- IEEE 1057, Standard for Digitizing Waveform Recorders. https://standards.ieee.org/
- IEEE 1241, Standard for Terminology and Test Methods for Analog-to-Digital Converters. https://standards.ieee.org/
- ASME V&V 40, Assessing Credibility of Computational Modeling through Verification and Validation. https://www.asme.org/

### Simulation and robotics

- ROS 2 Jazzy Jalisco release notes. https://github.com/ros2/ros2_documentation/blob/jazzy/source/Releases/Release-Jazzy-Jalisco.rst
- REP 103, Standard Units of Measure and Coordinate Conventions. https://reps.openrobotics.org/rep-0103/
- REP 105, Coordinate Frames for Mobile Platforms. https://reps.openrobotics.org/rep-0105/
- Gazebo documentation. https://gazebosim.org/docs
- MuJoCo documentation. https://mujoco.readthedocs.io/
- Pinocchio documentation. https://stack-of-tasks.github.io/pinocchio/
- MoveIt 2 documentation. https://moveit.picknik.ai/
- Renode documentation. https://renode.io/
- Wokwi documentation. https://wokwi.com/

### Instruments and components

- Texas Instruments, INA219 datasheet, SBOS448G. https://www.ti.com/lit/ds/symlink/ina219.pdf
- Analog Devices, AD8232 datasheet. https://www.analog.com/media/en/technical-documentation/data-sheets/AD8232.pdf
- Nordic Semiconductor, Power Profiler Kit II. https://www.nordicsemi.com/Products/Development-hardware/Power-Profiler-Kit-2
- Joulescope documentation. https://www.joulescope.com/
- sigrok/PulseView documentation. https://sigrok.org/wiki/PulseView

### Regulatory

- FDA, Technical Considerations for Medical Devices with Physiologic Closed-Loop Control Technology (September 2023). https://www.fda.gov/media/154994/download
- FDA, Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions. https://www.fda.gov/media/119933/download
- IEC 81001-5-1:2021, Health software and health IT systems safety, effectiveness and security — Part 5-1: Security. https://webstore.iec.ch/

> **Note:** The standard texts are paywalled. The application notes and datasheets above were used to check the teaching values. Confirm every clause and specification against the current edition before citing a number in a regulatory document.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_safety-and-standards.md`](01_safety-and-standards.md)
- Next topic: [`03_lab-conventions.md`](03_lab-conventions.md)
- Sibling: [`04_sensor-and-signal-labs.md`](04_sensor-and-signal-labs.md)
- Sibling: [`09_evidence-dossier.md`](09_evidence-dossier.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
