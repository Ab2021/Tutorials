# Hardware-in-the-Loop Labs

> Module 24 — Hardware, Robotics Labs
> Topic 7 of 9
> Estimated study time: 5 hours
> Prerequisites: 03, 04, 05
> Version 2026.3

## 1. Why this topic exists

A bench test answers a question about one block. A hardware-in-the-loop (HIL) test answers a question about the integrated system in real time. The device under test (DUT) runs its real firmware, on its real processor, with real inputs and outputs, while a model or a signal generator supplies the world around it.

The industry problem is that integration defects appear only when the blocks run together. A detector that works on a recorded signal fails when the sample arrives late. A watchdog that works in a unit test fails when the health task is starved. A safe state that works on the bench fails when the actuator driver keeps the pin high through a reset. These defects are not in the blocks; they are in the timing and the interfaces between them.

Three decisions depend on this topic:

1. **Whether the evidence is HIL or something weaker.** Software-in-the-loop (SIL) runs the algorithm on a host. Processor-in-the-loop (PIL) runs it on the target with simulated inputs. HIL runs the target in real time with real I/O. Only HIL supports a claim about target timing.
2. **Whether the test rig changes the behaviour it measures.** A signal generator, a DAC, a plant model, and a logging path all add delay. If the delay is not characterised, the measured latency is the rig's latency, not the device's.
3. **Whether the scenarios cover the risk file.** A HIL suite that tests only nominal rhythms is a demonstration. A HIL suite that injects the faults named in the risk file is evidence.

The commercial stakes are concrete. A HIL escape can become a field correction or a recall. A latency claim that was measured on a desktop becomes a design decision made on the wrong number. A fault that was never injected becomes a surprise in the field.

The rig is also a place where a team can fool itself. A scenario that is too clean, a latency that excludes the logging, a fault that was never actually applied, and a p99 computed from 20 events all produce a report that looks better than the device. The conventions in this topic are designed to make those self-deceptions visible.

This topic builds the HIL loop, the timing budget, the fault-injection matrix, and the traceability matrix. It assumes the firmware lab from Topic 5 and the sensor lab from Topic 4.

A useful way to read the topic is as a sequence of questions: what is the claim, what is the rig, what is the delay, what is the fault, what is the evidence, and what is missing. Each section answers one of them.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work. The HIL "patient" is a resistor network, a signal generator, or a recorded waveform, never a person.

## 2. Learning outcomes

By the end of this topic you can:

- Distinguish MIL, SIL, PIL, and HIL, and state which claims each can support.
- Draw the HIL loop with the plant model, the signal conditioning, the DUT, and the capture, and name the delay at each stage.
- Measure the rig's own latency with a loopback and correct the reported latency for it.
- Build a scenario suite that covers the faults named in a risk file, with expected behaviour for each.
- Measure end-to-end latency at p50, p95, and p99, with a confidence interval and the number of runs.
- Measure false alarms per hour and fault-detection latency, and compare them with written requirements.
- Build a traceability matrix that links each requirement to a risk control, a test case, and an evidence file.
- State what a HIL result does not prove, including usability, clinical benefit, and EMC compliance.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A HIL rig is a closed loop with a real device in it. The loop has a plant model, a signal path, the DUT, and a capture path. Each element adds delay, and the delays determine what the measurement means.

### 3.1 The HIL loop

```text
  scenario --> plant model --> signal conditioning --> DUT
      ^                                                  |
      |                                                  v
      +------------- capture <---------------------- output
                                                         |
                                                    actuator model
```

The DUT sees a signal that represents the world. The world is updated from the DUT's output. The loop is closed, so the DUT's behaviour changes the input it receives. That is what makes HIL different from a replay: a replay is open-loop, and the DUT cannot affect the signal.

A closed loop also means that a delay in the rig can destabilise the loop or change the DUT's behaviour. The rig's delay is part of the system being tested, and it must be measured and reported.

### 3.2 The timing budget

The end-to-end latency is the sum of the delays along the loop.

| Stage | What it includes | Typical magnitude | How to measure |
|---|---|---|---|
| Sensor | Sampling, conversion, filtering | 1–10 ms | DUT GPIO or known input |
| Transport | Bus, queue, radio | 0.1–100 ms | Timestamp at both ends |
| Processing | Algorithm, window, decision | 1–50 ms | GPIO toggle |
| Actuator | DAC, driver, mechanical | 0.1–20 ms | Scope or position sensor |
| Capture | Logging, timestamping | 0.01–1 ms | Loopback |

The requirement is on the end-to-end latency, not on one stage. A stage that is fast and a stage that is slow can produce the same total, and the total is what the clinician or the controller sees.

### 3.3 The real-time property

A HIL rig is real time if it produces each output before the next input is due, every time, with a bounded jitter. A rig that is fast on average and late occasionally is not real time.

```text
  period T:   |---------|---------|---------|
  deadline:   |---->    |---->    |---->    |
  actual:     |--->     |------>  |-->      |
                        ^ late
```

The late sample is the one that matters. A rig with a 1 ms mean and a 20 ms tail is a rig that will occasionally feed the DUT a stale input.

The real-time property is verified with a timer-driven loop and an external capture. A `sleep()` loop is not real time, because the operating system can delay the wake-up by an unbounded amount.

### 3.4 The fault-injection points

A fault can be injected at any interface in the loop.

| Injection point | Example | What it tests | What it does not test |
|---|---|---|---|
| Signal | Open lead, saturation, noise | Detection and response | The physical connector |
| Digital | Stuck bit, dropped sample, delayed sample | Protocol and algorithm robustness | The analog front end |
| Power | Brown-out, ripple | Reset and safe state | The battery chemistry |
| Timing | Clock drift, delayed command | Deadline behaviour | The network |
| Configuration | Corrupted parameter, wrong version | Validation and rollback | The user interface |
| Actuator | Stuck output, limited torque | Safe state and recovery | The mechanical structure |

The injection point determines what the test proves. A digital stuck bit does not prove that an open electrode is detected; the analog path is different.

### 3.5 The evidence chain

A HIL result is evidence only if it is linked to a requirement and a risk control.

```text
  hazard --> risk control --> requirement --> test case --> result --> evidence file
```

A test case without a requirement is an experiment. A result without an evidence file is an anecdote. The traceability matrix is the map from the hazard to the file.

### 3.6 The coverage question

Coverage is not a percentage of code. It is a statement about which hazards were exercised.

| Coverage type | Question | Limitation |
|---|---|---|
| Code coverage | Which lines ran? | A line can run without the hazard being present |
| Requirement coverage | Which requirements were tested? | A requirement can be tested at one condition |
| Risk coverage | Which risk controls were exercised? | A control can be exercised without the fault |
| Fault coverage | Which faults were injected? | The fault list may be incomplete |
| Scenario coverage | Which clinical scenarios were run? | The population is not represented |

A HIL report should state the coverage it claims and the coverage it does not. A code-coverage number without a risk-coverage statement is a weak claim.

### 3.7 The two clocks

A HIL rig has at least two clocks: the clock that produces the samples and the clock that records the events. If they are not the same, the latency includes their offset.

| Clock | Typical source | Drift | Use |
|---|---|---|---|
| Sample clock | DUT timer or DAC | Crystal, tens of ppm | Sample timing |
| Capture clock | Logic analyser | Crystal, tens of ppm | Event timing |
| Host clock | Operating system | NTP, milliseconds | Logging only |
| Wall clock | Real-time clock | Minutes per month | Human reference |

The host clock is not suitable for a latency measurement. It can jump backwards when NTP corrects it, and its resolution is often 1 ms or worse. The capture clock is suitable if its offset from the sample clock is measured.

The offset is measured by a common event: a GPIO toggle that both instruments capture. The difference between the two timestamps is the offset, and it is subtracted from the latency.

### 3.8 A scenario is a hypothesis

A scenario is not a recording of reality. It is a hypothesis about the input the device will see, expressed as a signal and an expected behaviour.

| Scenario element | Hypothesis | Evidence needed |
|---|---|---|
| Rhythm | The device will see this rhythm | Clinical data or a cited database |
| Noise | The noise has this spectrum and amplitude | Measurement or a cited model |
| Motion | The artefact has this shape and duration | Measurement or a cited model |
| Timing | The samples arrive at this rate | The rig's clock |
| Fault | This fault is plausible | The risk file |

A scenario that is not grounded in evidence is a guess. The guess may still be useful for exploration, but it is not evidence of performance in the field.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| HIL | Hardware-in-the-loop; target runs in real time with simulated inputs | Claimed when the host runs the algorithm | None |
| SIL | Software-in-the-loop; algorithm on a host | Called HIL | None |
| PIL | Processor-in-the-loop; algorithm on the target with simulated I/O | Called HIL | None |
| MIL | Model-in-the-loop; model and controller in one simulation | Called SIL | None |
| DUT | Device under test | Called "the board" | None |
| Plant model | Model of the system the DUT controls | Confused with the device model | None |
| Real time | Producing each output before the next input is due | "Fast enough on average" | None |
| Determinism | Same input, same timing, same output | Confused with speed | None |
| Jitter | Variation in period or latency | Confused with drift | Time |
| Latency | Delay from input to output | Confused with jitter | Time |
| End-to-end latency | Delay from the physical input to the physical output | One stage only | Time |
| Transport delay | Delay in a bus, queue, or network | Ignored in the budget | Time |
| Actuator delay | Delay from command to motion | Ignored in the budget | Time |
| Loopback | Signal path that returns the output to the input for timing | Confused with feedback | Time |
| Scenario | Defined input sequence and expected behaviour | A single waveform | None |
| Seed | Initial value for a random process | Omitted | Integer |
| Replay | Open-loop playback of a recorded signal | Called closed-loop | None |
| Fault injection | Deliberate introduction of a fault | Confused with a test failure | None |
| Oracle | Rule that decides pass or fail | A human judgement | None |
| Pass/fail criterion | Pre-defined condition for acceptance | Decided after the run | None |
| False alarm | Alarm without the target condition | Any alarm | Count per hour |
| False alarm rate | False alarms per unit time | Count per run | 1/h |
| Detection latency | Time from fault to detection | Time from fault to recovery | Time |
| Recovery time | Time from detection to safe state or normal operation | Detection latency | Time |
| Safe state | State in which the risk is acceptable | Any stopped state | State |
| Traceability matrix | Map from requirement to test to evidence | A list of tests | Table |
| Coverage | Extent to which the hazards were exercised | Code coverage only | % or statement |
| Model credibility | Evidence that a model is adequate for its context | Model accuracy | Argument |
| Context of use | The decision the model supports | The model's purpose in general | Statement |
| V&V 40 | ASME standard for model credibility | A software test standard | Standard |
| PCLC | Physiologic closed-loop control | A generic control loop | Regulation |
| Real-time operating system | OS with bounded scheduling latency | A general-purpose OS | None |
| Monotonic clock | Clock that never goes backwards | Wall-clock time | s |
| Timestamp | Recorded time of an event | Assumed to be the event time | s |
| Log | Record of events and data | A print statement | File |
| Evidence file | File that supports a test result | Any output | File |
| Requirement ID | Unique identifier for a requirement | A description | String |
| Risk control | Action that reduces risk | A test | None |
| Hazard | Potential source of harm | A fault | None |

## 5. The mechanics, step by step

The procedure below builds and runs a HIL rig. It assumes a DUT with firmware, a signal path from a DAC or a generator to the DUT's analog input, a capture path from the DUT's output to a logic analyser or a host, and a scenario runner.

### Step 1. Write the requirements and the claim boundary

**Input:** the device function, the risk file, and the clinical or control context.

**Output:** a requirement table with detection, latency, false-alarm, and safe-state requirements, and a claim boundary.

**Rationale:** a HIL test without a requirement cannot fail. The requirement table is the oracle for every scenario.

| ID | Requirement | Value | Verification |
|---|---|---:|---|
| HL-01 | Sensitivity on the annotated set | ≥ 0.90 | Scenario suite |
| HL-02 | Positive predictive value | ≥ 0.85 | Scenario suite |
| HL-03 | False alarms | ≤ 1 per hour | 30-hour run |
| HL-04 | End-to-end latency, p99 | ≤ 30 ms | 1,000 events |
| HL-05 | Lead-off detection | ≤ 250 ms | Fault injection |
| HL-06 | Saturation detection | ≤ 150 ms | Fault injection |
| HL-07 | Safe state after fault | ≤ 2 s | Fault injection |
| HL-08 | Every risk control exercised | 100% | Traceability matrix |

The claim boundary states what the HIL result does not prove: usability, clinical benefit, EMC compliance, and performance in the intended population.

### Step 2. Select the scenarios from the risk file

**Input:** the risk file, the intended use, and the signal characteristics.

**Output:** a scenario list with the expected behaviour and the linked hazard.

**Rationale:** the scenarios are the test cases. A scenario that is not linked to a hazard is an experiment; a hazard that is not covered is an untested risk.

The suite should include:

- Normal sinus rhythm at 60, 72, and 90 bpm.
- Sinus bradycardia and tachycardia.
- Atrial fibrillation.
- Premature ventricular contractions.
- Baseline wander.
- 50/60 Hz interference.
- Muscle noise.
- Motion artifact.
- Lead-off.
- Saturation.
- Dropout.
- Stuck ADC bit.
- Clock drift.
- Delayed sample.
- Corrupted configuration.
- Wrong firmware version.

At least 30 scenarios and three seeds per scenario are a reasonable minimum for a teaching lab. A clinical claim needs a much larger and more representative set.

### Step 3. Calibrate the signal path

**Input:** the DAC or generator, the attenuator, the buffer, and the DUT's ADC.

**Output:** a gain, an offset, and a noise floor for the path, with uncertainty.

**Rationale:** the DUT must see the voltage the scenario specifies. A path with a 5% gain error feeds a 5% wrong signal, and the detector's performance is measured against the wrong input.

Procedure:

1. Apply a DC level at the DAC and measure the DUT's ADC code.
2. Repeat at five levels across the range.
3. Fit code = a × voltage + b.
4. Measure the noise floor with the DAC at mid-scale.
5. Measure the path's bandwidth with a sine sweep.
6. Verify that a 1 mV input at the DUT corresponds to the intended code.

The path's gain and offset are recorded in the manifest. A change in the attenuator or the buffer is a change in the test.

### Step 4. Verify that the loop is real time

**Input:** the scenario runner and the DAC update.

**Output:** a measured update period, jitter, and missed deadlines.

**Rationale:** a HIL claim depends on the runner producing samples on time. A runner that uses `sleep()` or a general-purpose operating system timer is not real time.

Procedure:

1. Toggle a GPIO at each DAC update.
2. Capture 10,000 updates with a logic analyser.
3. Compute the period, the jitter, and the number of intervals above the deadline.
4. Repeat with the logging enabled and with the display updating.
5. Record the worst case.

A single missed deadline in 10,000 updates is a 0.01% miss rate, which may be acceptable for a demonstration and unacceptable for a safety claim. The acceptance criterion must be stated.

### Step 5. Measure the rig's own latency

**Input:** a loopback from the DAC output to the capture input, bypassing the DUT.

**Output:** the rig's delay, with uncertainty.

**Rationale:** the measured end-to-end latency includes the rig. If the rig's delay is not characterised, the report attributes it to the device.

Procedure:

1. Connect the DAC output directly to the capture input.
2. Generate a step at the DAC.
3. Measure the time from the DAC update to the captured edge.
4. Repeat 100 times and compute the mean and the standard deviation.
5. Record the delay and its uncertainty.

A rig delay of 2–5 ms is common for a USB DAQ and a host-side logger. The delay must be subtracted only if it is additive and independent of the DUT's behaviour. If the rig's delay depends on the DUT's output, the correction is not valid.

### Step 5A. Align the clocks

**Input:** the scenario timestamps, the DUT timestamps, and the capture timestamps.

**Output:** a common time base and a measured offset.

**Rationale:** a latency measurement is a difference between two clocks. If the clocks are not aligned, the difference includes the offset.

Procedure:

1. Use a monotonic clock for each timestamp source.
2. Record the offset between the host clock and the capture clock at the start and the end of the run.
3. If the offset drifts, correct it linearly or discard the run.
4. Record the timestamp source in the log.
5. Do not use wall-clock time for a latency measurement; an NTP correction can move it backwards.

A run with a clock offset of 10 ms produces a latency error of 10 ms, which is larger than the margin in many requirements. The alignment is part of the measurement, and it must be repeated after any change to the rig or the host.

A useful check is to inject a step at a known time and confirm that both the scenario log and the capture show it at the same instant. If the step appears at different times, the clocks are not aligned.

### Step 6. Build the scenario runner

**Input:** the scenario definitions and the calibrated path.

**Output:** a runner that renders each scenario at the specified rate, streams it with a hardware timer, and logs the DUT's output.

**Rationale:** the runner is the instrument. Its timing, its buffer size, and its logging must be designed, not improvised.

The runner records:

- The scenario name and the seed.
- The sample rate and the buffer size.
- The DAC code for each sample.
- The DUT's output and the timestamp.
- The fault flags and the device state.
- The host time and the monotonic time.

A runner that logs to a file on the same thread that updates the DAC can miss a deadline. The logging must be buffered or moved to another thread.

### Step 7. Run the nominal scenarios first

**Input:** the runner and the nominal scenarios.

**Output:** a baseline result for each nominal scenario.

**Rationale:** the nominal run establishes that the rig works before the faults are injected. A fault result on a rig that has not passed the nominal case is uninterpretable.

The nominal run checks:

- The signal amplitude and rate at the DUT.
- The detector's output on a clean signal.
- The latency on a clean signal.
- The absence of false alarms.

### Step 8. Inject the faults

**Input:** the fault list and the injection points.

**Output:** a capture for each fault, showing that the fault was applied and how the DUT responded.

**Rationale:** a fault that was not demonstrated is a fault that may not have been applied. The capture is the evidence.

Procedure:

1. For an open lead, insert a 1 MΩ series resistor or open the connection.
2. For a short to mid-rail, connect the input to mid-rail through a low resistor.
3. For saturation, drive the input to the rail.
4. For a stuck bit, mask one bit in the digital path.
5. For a dropped sample, remove one sample from the stream.
6. For a delayed sample, hold the sample for a defined number of periods.
7. For a corrupted configuration, write an invalid parameter.
8. For a wrong version, flash an older image.

Each injection is captured at the electrical or digital interface, not only inferred from the DUT's output.

### Step 9. Measure latency and jitter

**Input:** the timestamps and the DUT's output.

**Output:** the latency distribution, the p50, p95, and p99, and a confidence interval.

**Rationale:** a mean latency hides the tail. The tail is what a control loop and a safety monitor see.

Procedure:

1. Timestamp the input event at the source.
2. Timestamp the output event at the DUT or the capture.
3. Compute the difference for each event.
4. Sort and compute the percentiles.
5. Compute a confidence interval for the p99, using a bootstrap or an order-statistic method.
6. Report the number of events and the number of runs.

A p99 from 100 events is the second-largest value, which is not a stable estimate. A p99 from 1,000 events is the tenth-largest, which is better. The report must state the sample size.

### Step 10. Measure the false-alarm rate

**Input:** a long nominal run, ideally 24–30 hours.

**Output:** the number of false alarms and the rate per hour.

**Rationale:** a false alarm rate is a count per unit time, not a count per run. A run that is 10 minutes long cannot estimate a rate of 1 per hour.

Procedure:

1. Run the nominal scenario for at least 24 hours, or use a accelerated replay if the signal is stationary.
2. Count every alarm.
3. Classify each alarm as true, false, or indeterminate.
4. Compute the rate and a confidence interval.
5. Report the duration and the number of alarms.

A count of 3 in 30 hours is 0.1 per hour. The confidence interval is wide because the count is small. A normal approximation gives 0.1 ± 0.11 per hour, which includes zero; an exact Poisson interval is similar in width. The report should say that the estimate is imprecise.

### Step 11. Build the traceability matrix

**Input:** the requirements, the risk controls, the test cases, and the results.

**Output:** a matrix with one row per requirement and a link to the evidence.

**Rationale:** the matrix is the argument that the tests cover the risks. A test that is not in the matrix is not part of the evidence.

| Requirement | Risk control | Test case | Result | Evidence file |
|---|---|---|---|---|
| HL-01 | Detect arrhythmia | SCN-01 to SCN-12 | Sensitivity 0.93 | results/sens.json |
| HL-02 | Avoid false alarm | SCN-01 to SCN-12 | PPV 0.91 | results/ppv.json |
| HL-03 | Alarm fatigue | RUN-30H | 0.1/h | results/far.json |
| HL-04 | Timely response | LAT-1000 | p99 19.6 ms | results/lat.csv |
| HL-05 | Lead-off | FLT-LEAD | 180 ms | captures/lead_off.png |
| HL-06 | Saturation | FLT-SAT | 95 ms | captures/sat.png |
| HL-07 | Safe state | FLT-ALL | 1.2 s | captures/safe.png |
| HL-08 | Coverage | Matrix review | 100% | traceability.csv |

The matrix is reviewed against the risk file. A risk control that is not in the matrix is an untested control.

### Step 12. Analyse and report

**Input:** the raw logs, the captures, and the matrix.

**Output:** a report with the results, the uncertainty, the verdict, and the limitations.

**Rationale:** the report is the deliverable. It must be readable by a person who was not in the room, and it must state what the result does not prove.

The report contains:

- The intended use and the claim boundary.
- The requirements and the acceptance criteria.
- The rig architecture and the timing budget.
- The calibration and the rig latency.
- The scenario suite and the seeds.
- The results with uncertainty.
- The fault-injection captures.
- The traceability matrix.
- The limitations.

### Step 13. Review and archive

**Input:** the report, the logs, and the matrix.

**Output:** a review record and an archive.

**Rationale:** the review catches the gaps that the team cannot see: a scenario that was not run, a fault that was not captured, a requirement without a result.

The review checks the coverage, the calibration, the real-time evidence, the uncertainty, and the limitations.

### 5.15 Acceptance criteria for the HIL lab

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Requirements are written and numbered | Requirement table | Rewrite |
| 2 | Claim boundary is stated | Report | Add |
| 3 | At least 30 scenarios | Scenario list | Extend |
| 4 | Three seeds per scenario | Logs | Repeat |
| 5 | Signal path is calibrated | Fit | Recalibrate |
| 6 | Runner is real time | Jitter capture | Fix |
| 7 | Rig latency is measured | Loopback | Measure |
| 8 | Faults are demonstrated | Captures | Inject |
| 9 | Latency is reported at p50, p95, p99 | Table | Compute |
| 10 | Sample size is stated | Report | Add |
| 11 | False-alarm rate has a duration | Report | Extend run |
| 12 | Safe state is verified | Capture | Fix |
| 13 | Traceability matrix is complete | Matrix | Complete |
| 14 | Limitations are stated | Report | Add |

### 5.16 Branch points

**Branch A: the algorithm runs on a host, not the target.** The result is SIL, not HIL. Label it accordingly and do not claim target timing.

**Branch B: the plant model is a replay.** The result is open-loop. It can test detection but not closed-loop stability.

**Branch C: the fault cannot be injected electrically.** Inject it digitally and state the limitation. A digital fault does not prove the analog path.

**Branch D: the actuator is connected.** Use a dummy load first, then the actuator under supervision. Verify the safe state with the actuator disconnected, then with the load.

## 6. Worked example

### 6.1 The question

A team has built a HIL rig for an ECG arrhythmia detector. The detector runs on a Cortex-M board. The rig streams scenarios through a 12-bit DAC and an attenuator to the DUT's analog front end. The team wants to know whether the detector meets the latency requirement, the false-alarm requirement, and the lead-off detection requirement, and what the rig itself contributes to the measurement.

### 6.2 The rig

| Item | Value | Note |
|---|---|---|
| DUT | Cortex-M4 at 80 MHz | Firmware commit 5c1d9a2 |
| Algorithm | Pan-Tompkins-like QRS detector | 8-sample moving average, adaptive threshold |
| DAC | 12-bit, 500 kS/s | External, SPI |
| Attenuator | 100:1 | 3.3 V to 33 mV range |
| ADC | 12-bit, 500 S/s | DUT internal |
| Signal bandwidth | 0.5–40 Hz | Analog front end |
| Capture | Logic analyser, 24 MHz | 41.67 ns resolution |
| Host | Linux, PREEMPT_RT | Not used for the timing claim |
| Scenario rate | 500 Hz | 2 ms per sample |
| Seeds | 3 per scenario | 20261026, 20261027, 20261028 |

The host runs the scenario and logs the results. The DAC is driven by the DUT's SPI, not by the host, so the sample clock is not subject to host scheduling.

### 6.2a The rig architecture

| Element | Implementation | Delay contribution | Evidence |
|---|---|---|---|
| Scenario store | Host file, 16-bit PCM | None during the run | Hash of the file |
| DAC | External 12-bit, SPI | 0.2 ms | Datasheet, loopback |
| Attenuator | 100:1, 1% resistors | Negligible | Measured gain |
| DUT input | ADC, 500 S/s | 2 ms sampling | GPIO capture |
| Algorithm | 8-sample window | 16 ms window | GPIO capture |
| Output flag | GPIO | < 1 µs | Logic analyser |
| Capture | Logic analyser, 24 MHz | 0.04 µs resolution | Instrument spec |
| Logger | Host, buffered | Not in the loop | N/A |

The table is the timing budget in the form the report should present it. The sum of the measured contributions is compared with the end-to-end measurement, and the difference is the unexplained delay.

### 6.3 The rig latency

The team connects the DAC output directly to the capture input and measures the delay from the DAC update to the captured edge.

| Statistic | Value | Unit |
|---|---:|---|
| Mean loopback delay | 2.8 | ms |
| Standard deviation | 0.3 | ms |
| Minimum | 2.3 | ms |
| Maximum | 3.4 | ms |
| Number of runs | 100 | — |

The 2.8 ms is the sum of the DAC settling, the capture buffer, and the timestamp alignment. It is not the DUT's latency. The team will report both the raw and the corrected values.

### 6.4 The latency result

The team runs 200 events with a clean signal and records the time from the DAC update to the detector's output flag.

| Statistic | Raw | Corrected for rig | Unit |
|---|---:|---:|---|
| p50 | 12.2 | 9.4 | ms |
| p90 | 15.7 | 12.9 | ms |
| p95 | 16.4 | 13.6 | ms |
| p99 | 19.6 | 16.8 | ms |
| Maximum | 21.0 | 18.2 | ms |
| Mean | 12.5 | 9.7 | ms |
| Standard deviation | 2.5 | 2.5 | ms |

The corrected p99 is 16.8 ms, against a requirement of 30 ms. The margin is 13.2 ms. The raw p99 is 19.6 ms, which also meets the requirement, so the conclusion does not depend on the correction. The correction matters for the margin, not for the verdict.

### 6.5 The uncertainty of the p99

The p99 is estimated from 200 events. The 99th percentile is the 198th order statistic, and its uncertainty is not the same as the uncertainty of the mean.

A bootstrap resample of the 200 values gives a 95% interval of 17.4 to 21.0 ms for the raw p99. The interval is asymmetric and wide because the tail is bounded by the maximum and only a few observations lie above the 95th percentile. The corrected interval is 14.6 to 18.2 ms.

The width of the interval is a warning about the sample size. A p99 estimated from 200 events has a 3.6 ms interval, which is 18% of the point estimate. A larger sample would narrow it.

The timing instrument contributes a resolution of 41.67 ns and a timebase error of 50 ppm, which is negligible at 20 ms. The rig-delay uncertainty is 0.3 ms, which is larger than the bootstrap width. The combined uncertainty of the corrected p99 is about 0.4 ms.

### 6.6 The false-alarm rate

The team runs the nominal scenario for 30 hours. The detector raises 3 alarms that are not associated with an arrhythmia.

\[
\text{FAR} = \frac{3}{30} = 0.1\ \text{per hour}
\]

The requirement is 1 per hour. The point estimate passes with a factor of 10 margin. The confidence interval is wide:

\[
\text{FAR}_{95\%} \approx 0.1 \pm 1.96 \frac{\sqrt{3}}{30} = 0.1 \pm 0.11
\]

The interval is 0 to 0.21 per hour, which includes zero. The estimate is imprecise because the count is small. A longer run would narrow the interval, but the duration is already 30 hours.

The team should report the interval, not only the point estimate. A claim of "0.1 per hour" without the interval overstates the precision.

### 6.6a What counts as a false alarm

The three alarms in the 30-hour run were classified by a reviewer who listened to the event window and read the annotation.

| Alarm | Time | Reference rhythm | Classification | Reason |
|---:|---|---|---|---|
| 1 | 04:12 | Sinus, 68 bpm | False | Motion artefact |
| 2 | 11:37 | Sinus, 72 bpm | False | Baseline step |
| 3 | 22:05 | Sinus, 65 bpm | False | Noise spike |
| 4 | 27:41 | PVC | True | Correct detection |

The table shows the classification rule and the evidence. A false alarm is not an alarm the team dislikes; it is an alarm without the target condition, confirmed against the reference.

The classification is a judgement, and it should be made by a person who did not tune the detector. A team that both tunes and adjudicates can unconsciously reclassify the inconvenient events.

### 6.7 The fault-detection results

| Fault | Injected at | Detection | Requirement | Verdict |
|---|---|---:|---:|---|
| Lead-off | Series 1 MΩ in the input | 180 ms | 250 ms | Pass |
| Saturation | Input driven to the rail | 95 ms | 150 ms | Pass |
| Stuck ADC bit | Bit 5 masked in the digital path | 240 ms | 250 ms | Pass, marginal |
| Dropped sample | One sample removed | 8 ms | 50 ms | Pass |
| Delayed sample | 20 ms hold | 22 ms | 50 ms | Pass |
| Corrupted configuration | Invalid threshold written | 1.2 s | 2 s | Pass |

The stuck-bit detection is marginal. The detector notices the flat signal after 240 ms, which is close to the 250 ms requirement. A different bit or a different signal could exceed the requirement.

### 6.8 Two checks that would invalidate the result

**Check 1: the DAC update is not timer-driven.** If the scenario runner uses a host `sleep()` or a general-purpose timer, the update interval can be delayed by tens of milliseconds, and the DUT receives a jittery signal. The measured latency would then include the host's scheduling, not the DUT's processing. The check is to capture the DAC update with the logic analyser and compute the jitter. If the jitter is comparable to the latency requirement, the run is not a valid timing test.

**Check 2: the plant model adds delay that is not part of the device.** The rig's 2.8 ms loopback delay is characterised, but the scenario renderer may add a further delay between the scenario time and the DAC update. If the renderer buffers 10 samples before writing, the effective delay is 20 ms at 500 Hz. The check is to compare the scenario timestamp with the DAC update timestamp for a step input. If the step appears 20 ms late, the renderer's buffer is part of the latency and must be corrected or removed.

### 6.9 A varied condition: the sample rate is reduced to 250 Hz

The team repeats the latency measurement at 250 Hz, keeping the same algorithm and the same scenario.

| Condition | Sample rate | p50 | p95 | p99 | Requirement | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 500 Hz | 12.2 ms | 16.4 ms | 19.6 ms | 30 ms | Pass |
| Varied | 250 Hz | 15.1 ms | 21.8 ms | 27.9 ms | 30 ms | Pass, marginal |

The latency rises because the algorithm's window is defined in samples, so a lower sample rate means a longer window in time. The p99 rises from 19.6 to 27.9 ms, which is within the requirement but close to it.

The varied condition shows the trade-off between sample rate and latency. A lower rate reduces the data and the power, and increases the latency. The choice depends on the clinical requirement, not on the convenience of the firmware.

### 6.9a What the marginal results mean

Two results are marginal: the stuck-bit detection at 240 ms against 250 ms, and the 250 Hz p99 at 27.9 ms against 30 ms. A marginal result is not a failure, but it is not a comfortable pass.

| Result | Point estimate | Requirement | Margin | Interval or variation | Decision |
|---|---:|---:|---:|---|---|
| Stuck-bit detection | 240 ms | 250 ms | 10 ms | One bit, one signal | Retest with other bits |
| 250 Hz p99 | 27.9 ms | 30 ms | 2.1 ms | p99 interval width about 4 ms | Treat as unproven |
| Lead-off detection | 180 ms | 250 ms | 70 ms | Three repeats | Pass |
| False-alarm rate | 0.1/h | 1/h | 0.9/h | 0 to 0.21/h | Pass, imprecise |

A margin that is smaller than the measurement interval is not evidence of compliance. The 250 Hz p99 of 27.9 ms with a 4 ms interval could be 31.9 ms, which would fail. The 500 Hz result is more robust because its margin is larger than its interval.

The table is the honest form of the result. A report that says "all requirements met" without showing the margins and the intervals has hidden the two results that need attention.

### 6.10 What the example demonstrates

The arithmetic closes: a rig delay of 2.8 ms, a raw p99 of 19.6 ms, a corrected p99 of 16.8 ms, and a requirement of 30 ms. The false-alarm rate is 0.1 per hour with a 95% interval of 0 to 0.21. The lead-off detection is 180 ms against a 250 ms requirement.

The example demonstrates the difference between a device measurement and a rig measurement. The 2.8 ms rig delay is small compared with the 19.6 ms latency, but it is not negligible for the margin. A rig with a 20 ms delay would dominate the result, and the corrected latency would be the only meaningful number.

The example also demonstrates the difference between a point estimate and an interval. The false-alarm rate passes by a factor of 10, but the interval includes zero. The claim should be "the observed rate is 0.1 per hour, with a wide interval because only three events were observed," not "the rate is 0.1 per hour."

## 7. Code, query, or calculation

The program below analyses the latency log, computes the bootstrap interval for p99, computes the false-alarm interval, computes a Wilson interval for sensitivity, and checks the traceability coverage. It uses only the Python standard library.

```python
import math
import random
import statistics

# --- 1. Latency log in milliseconds (200 events, sorted for display) ----
LATENCIES_MS = [
    7.3, 8.0, 8.0, 8.3, 8.3, 8.5, 8.7, 8.9, 9.0, 9.1,
    9.1, 9.1, 9.2, 9.2, 9.3, 9.3, 9.4, 9.4, 9.5, 9.5,
    9.5, 9.5, 9.6, 9.6, 9.6, 9.6, 9.8, 10.0, 10.0, 10.0,
    10.0, 10.0, 10.1, 10.1, 10.1, 10.2, 10.2, 10.2, 10.2, 10.3,
    10.3, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.5, 10.5,
    10.5, 10.6, 10.6, 10.7, 10.7, 10.7, 10.7, 10.8, 10.8, 10.8,
    10.8, 10.9, 10.9, 10.9, 11.0, 11.0, 11.1, 11.1, 11.1, 11.2,
    11.3, 11.3, 11.3, 11.3, 11.4, 11.4, 11.4, 11.4, 11.5, 11.5,
    11.5, 11.5, 11.5, 11.5, 11.6, 11.6, 11.6, 11.8, 11.8, 11.8,
    11.9, 11.9, 11.9, 11.9, 12.0, 12.0, 12.1, 12.1, 12.1, 12.2,
    12.2, 12.2, 12.2, 12.2, 12.3, 12.3, 12.3, 12.3, 12.4, 12.4,
    12.4, 12.5, 12.5, 12.6, 12.6, 12.6, 12.8, 12.8, 12.8, 12.8,
    12.9, 12.9, 13.0, 13.0, 13.1, 13.1, 13.2, 13.2, 13.2, 13.2,
    13.2, 13.3, 13.3, 13.5, 13.5, 13.5, 13.5, 13.6, 13.7, 13.8,
    13.8, 13.8, 13.8, 13.8, 13.9, 13.9, 13.9, 13.9, 13.9, 14.0,
    14.0, 14.0, 14.0, 14.1, 14.2, 14.3, 14.3, 14.3, 14.4, 14.4,
    14.4, 14.4, 14.5, 14.5, 14.5, 14.7, 15.0, 15.1, 15.2, 15.2,
    15.3, 15.3, 15.3, 15.3, 15.3, 15.5, 15.7, 15.7, 15.7, 15.7,
    15.8, 15.8, 15.9, 16.0, 16.0, 16.0, 16.1, 16.1, 16.3, 16.4,
    16.6, 16.7, 17.4, 17.7, 18.4, 19.3, 19.3, 19.6, 20.2, 21.0,
]

def percentile(xs, p):
    ys = sorted(xs)
    k = math.ceil(p / 100 * len(ys))
    return ys[k - 1]

def summarise(xs):
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "sd": statistics.stdev(xs),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "max": max(xs),
    }

s = summarise(LATENCIES_MS)
print("latency summary (ms)")
print(f"n={s['n']} mean={s['mean']:.2f} sd={s['sd']:.2f}")
print(f"p50={s['p50']:.1f} p90={s['p90']:.1f} p95={s['p95']:.1f} "
      f"p99={s['p99']:.1f} max={s['max']:.1f}")

# --- 2. Bootstrap 95% CI for p99 ---------------------------------------
def bootstrap_p99(xs, n_resamples=4000, seed=20261026):
    rng = random.Random(seed)
    n = len(xs)
    stats_ = []
    for _ in range(n_resamples):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        stats_.append(percentile(sample, 99))
    stats_.sort()
    return stats_[int(0.025 * n_resamples)], stats_[int(0.975 * n_resamples)]

lo, hi = bootstrap_p99(LATENCIES_MS)
print(f"p99 bootstrap 95% CI = ({lo:.1f}, {hi:.1f}) ms")

# --- 3. Correct for the rig delay --------------------------------------
RIG_DELAY_MS = 2.8
RIG_DELAY_U_MS = 0.3
corrected = {k: (v - RIG_DELAY_MS if k not in ("n",) else v) for k, v in s.items()}
print("\ncorrected for rig delay")
print(f"p50={corrected['p50']:.1f} p95={corrected['p95']:.1f} "
      f"p99={corrected['p99']:.1f} max={corrected['max']:.1f}")
print(f"rig delay {RIG_DELAY_MS:.1f} +/- {RIG_DELAY_U_MS:.1f} ms")
print(f"corrected p99 CI = ({lo-RIG_DELAY_MS:.1f}, {hi-RIG_DELAY_MS:.1f}) ms")

REQ_MS = 30.0
print(f"requirement {REQ_MS:.0f} ms -> "
      f"{'PASS' if s['p99'] <= REQ_MS else 'FAIL'}, "
      f"margin {REQ_MS - s['p99']:.1f} ms raw / "
      f"{REQ_MS - corrected['p99']:.1f} ms corrected")

# --- 4. False-alarm rate with a normal-approximation interval ----------
alarms, hours = 3, 30.0
rate = alarms / hours
se = math.sqrt(alarms) / hours
print("\nfalse alarms (normal approximation; small count)")
print(f"rate = {rate:.3f} per hour")
print(f"95% CI = ({max(0.0, rate - 1.96*se):.3f}, {rate + 1.96*se:.3f}) per hour")

# --- 5. Wilson interval for sensitivity ---------------------------------
def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

tp, fn = 930, 70
p = tp / (tp + fn)
lo_w, hi_w = wilson(p, tp + fn)
print("\nsensitivity")
print(f"p={p:.3f}  95% Wilson CI=({lo_w:.3f}, {hi_w:.3f})  n={tp+fn}")

# --- 6. Traceability coverage ------------------------------------------
REQUIREMENTS = {
    "HL-01": "SCN-01..12",
    "HL-02": "SCN-01..12",
    "HL-03": "RUN-30H",
    "HL-04": "LAT-200",
    "HL-05": "FLT-LEAD",
    "HL-06": "FLT-SAT",
    "HL-07": "FLT-ALL",
    "HL-08": None,
}
covered = sum(1 for v in REQUIREMENTS.values() if v)
print("\ntraceability")
print(f"requirements={len(REQUIREMENTS)} covered={covered} "
      f"coverage={covered/len(REQUIREMENTS)*100:.0f}%")
for k, v in REQUIREMENTS.items():
    print(f"{k}: {'COVERED' if v else 'GAP'} ({v})")
```

Expected output:

```text
latency summary (ms)
n=200 mean=12.46 sd=2.51
p50=12.2 p90=15.7 p95=16.4 p99=19.6 max=21.0
p99 bootstrap 95% CI = (17.4, 21.0) ms

corrected for rig delay
p50=9.4 p95=13.6 p99=16.8 max=18.2
rig delay 2.8 +/- 0.3 ms
corrected p99 CI = (14.6, 18.2) ms
requirement 30 ms -> PASS, margin 10.4 ms raw / 13.2 ms corrected

false alarms (normal approximation; small count)
rate = 0.100 per hour
95% CI = (0.000, 0.213) per hour

sensitivity
p=0.930  95% Wilson CI=(0.912, 0.944)  n=1000

traceability
requirements=8 covered=7 coverage=88%
HL-01: COVERED (SCN-01..12)
HL-02: COVERED (SCN-01..12)
HL-03: COVERED (RUN-30H)
HL-04: COVERED (LAT-200)
HL-05: COVERED (FLT-LEAD)
HL-06: COVERED (FLT-SAT)
HL-07: COVERED (FLT-ALL)
HL-08: GAP (None)
```

The percentile function uses the nearest-rank method. With 200 values, the 99th percentile is the 198th value, which is 19.6 ms. The bootstrap interval is 19.3 to 21.0 ms, which is asymmetric because the tail is bounded by the maximum.

The corrected values subtract the 2.8 ms rig delay from every percentile. The subtraction is valid only if the delay is additive, independent of the DUT's behaviour, and measured on the same run. The report must state those assumptions.

The false-alarm interval uses a normal approximation and is labelled as such. With three events, the approximation is rough, and an exact Poisson method would be better. The interval includes zero, which is a warning that the estimate is imprecise.

The traceability section shows a coverage gap: HL-08, the requirement that every risk control is exercised, has no test. That is a deliberate example of a matrix that looks complete until it is read. A coverage of 88% is not a pass; it is a finding.

## 8. Data, standards, and regulatory context

### 8.1 The closed-loop guidance

The FDA's guidance on physiologically closed-loop control (PCLC) technology, issued in September 2023, addresses devices that use a physiological sensor to adjust therapy. The guidance discusses HIL testing and requires that the patient model be integrated so that it minimally affects the system's functionality and does not alter the realistic nature of the test, for example by introducing additional delays.

The guidance also requires characterising the actuator transfer mechanism and the signal generator, including their accuracy and time delays, and accounting for them in the test plan and the analysis. It identifies latency as a device hazard and as a sensor and actuator specification.

This is the regulatory expression of the rig-delay problem. A HIL result that does not characterise the rig's delay is not a valid measurement of the device's latency.

### 8.2 Model credibility

ASME V&V 40 is a standard for assessing the credibility of computational models used in medical device design. It defines the context of use, the model risk, and the credibility activities. The level of credibility should be proportionate to the risk.

A plant model used to test a nominal rhythm detector is a low-risk model. A plant model used to test a closed-loop insulin controller is a high-risk model, and the credibility evidence is correspondingly larger.

### 8.3 Software lifecycle

IEC 62304:2006 + Amendment 1:2015 defines the software lifecycle for medical device software. The HIL test is a system-level verification activity. The test cases, the results, and the traceability are part of the software verification record.

The software safety class determines the rigour. A Class C device needs unit, integration, and system verification, with a documented traceability from requirements to tests.

### 8.4 Risk management

ISO 14971:2019 requires the manufacturer to identify hazards, estimate and evaluate the risk, implement controls, and evaluate the residual risk. The HIL suite is the evidence that the controls were exercised.

The fault list should be derived from the risk file, not from the imagination of the test engineer. A fault that is not in the risk file is an untested risk, and a fault in the risk file that is not in the suite is a gap.

### 8.5 Related standards

IEC 60601-1-10 covers the development of physiologic closed-loop controllers in medical electrical equipment. It addresses the controller's response to sensor failure, the means to verify the control, and the information for use.

IEC 62366-1 covers usability engineering. A HIL test does not establish usability. A formative or summative usability study with representative users is a separate activity.

IEC 60601-1-2 covers EMC. A HIL test does not establish EMC compliance. The HIL rig may itself be an EMC source, and the DUT's behaviour in the rig is not the same as its behaviour in an electromagnetic field.

### 8.6 HIL data management

A HIL run produces a large amount of data: the scenario, the input stream, the DUT output, the timestamps, the fault flags, and the captures. The data must be organised so that a reviewer can find the evidence for a specific test.

| Data | Size | Retention | Naming |
|---|---|---|---|
| Scenario definition | Small | Permanent | `scn_<id>_<seed>.yaml` |
| Input stream | Large | Permanent or regenerable | `in_<id>_<seed>.bin` |
| DUT output | Medium | Permanent | `out_<id>_<seed>.csv` |
| Timestamps | Medium | Permanent | In the output file |
| Captures | Large | Permanent for faults | `cap_<fault>.png` |
| Analysis | Small | Permanent | `analysis_<run>.py` |
| Summary | Small | Permanent | `result_<id>.json` |

The input stream can be regenerated from the scenario and the seed, so it can be stored as a scenario plus a seed rather than a large file. The regeneration must be verified against the hash for at least one run.

The naming convention is part of the traceability. A file named `out_07_20261026.csv` can be linked to a scenario and a seed without opening it.

### 8.7 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| FDA PCLC guidance | Guidance | Premarket submissions for PCLC devices | A teaching HIL rig |
| ASME V&V 40 | Consensus standard | Model credibility for medical devices | A simple plant model |
| IEC 62304 | International standard | Medical device software | A prototype |
| ISO 14971 | International standard | Medical device risk management | A lab hazard list |
| IEC 60601-1-10 | Collateral standard | Closed-loop controllers | A demonstration |
| IEC 62366-1 | International standard | Usability engineering | A HIL test |

## 9. Industry practice and case evidence

### 9.1 How industry uses HIL

HIL is standard practice in automotive, aerospace, and increasingly in medical devices. The rig runs the real controller against a simulated plant, so that faults can be injected without risk to a person or a machine.

A medical HIL rig typically includes:

- A real-time computer running the plant model.
- Signal conditioning that matches the DUT's electrical interface.
- Fault-insertion units on the signal and power lines.
- A data-acquisition system that records the DUT's outputs and the rig's inputs.
- A test manager that runs the scenarios and records the results.

The rig is calibrated and maintained like an instrument. Its delays are measured and recorded.

### 9.2 Published HIL examples

Karunagaran, Sahoo, and Fujita described an FPGA-based HIL testbed for an insulin pump in 2015. The testbed combined event-driven and time-driven operation and injected faults such as data corruption, bit flips, and incorrect signal assertions. The event-based triggers made the tests resilient to task jitter and software changes.

Hottenbacher and Haase described a HIL environment for networked medical devices using the IEEE 11073 service-oriented device connectivity standard in 2023. The environment tested communication under high load and built on prior work measuring the latency of IEEE 11073 stacks.

These examples show the pattern: a real-time plant, a fault-insertion path, and a measurement of timing and communication. The lab in this topic is a smaller version of the same pattern.

### 9.3 What test engineers measure

A HIL test engineer measures four things:

- **Functional correctness:** does the DUT produce the right output for the scenario?
- **Timing:** does it produce the output before the deadline, every time?
- **Fault behaviour:** does it detect the fault and reach the safe state?
- **Coverage:** which requirements and risks were exercised?

A report that covers only the first is a demonstration. A report that covers all four is evidence.

### 9.4 The rig as a source of error

The rig is not a neutral observer. It adds delay, noise, and impedance. It can also change the DUT's behaviour by loading an output or by injecting a ground loop.

The industry practice is to characterise the rig before the DUT is connected: a loopback for the delay, a noise floor for the signal path, and a calibration for the gain and offset. The characterisation is repeated after any change to the rig.

### 9.5 What a HIL report reviewer checks

A reviewer who was not in the room checks the report against the evidence, not against the conclusion.

1. Is the claim HIL, SIL, or PIL, and is it labelled?
2. Are the requirements written before the results?
3. Is the rig's delay measured, and is the correction justified?
4. Is the runner real time, and is the jitter captured?
5. Are the scenarios linked to hazards?
6. Is the fault applied, and is the injection captured?
7. Is the latency a distribution with a sample size?
8. Is the false-alarm rate a rate with a duration?
9. Is the safe state verified through the reset?
10. Is the traceability matrix complete, including the gaps?
11. Are the limitations stated?

A report that answers these questions is reviewable. A report that answers only the conclusion is a demo.

### 9.6 Why a HIL rig is not a clinical simulator

A HIL rig simulates the electrical and timing interface, not the patient. A plant model can reproduce a rhythm, a pressure waveform, or a glucose response, but it cannot reproduce the variability of a population, the effect of comorbidity, or the clinician's interpretation.

The distinction matters for the claim. A HIL result can support a statement about the device's behaviour in the tested scenarios. It cannot support a statement about clinical benefit, and it cannot replace a clinical investigation.

A team that treats the plant model as a patient has confused the map with the territory. The model is a tool for exercising the device; the clinical evidence is a separate activity.

This is also why a HIL report should not use clinical language for its results. "The detector achieved a sensitivity of 0.93 on the scenario set" is a HIL finding. "The device detects arrhythmias" is a clinical claim, and it needs clinical evidence.

### 9.7 Inference

It is an inference, not a documented finding, that teams which measure the rig delay before the DUT latency produce more accurate latency claims. The reasoning is that the rig delay is often a significant fraction of the total, and a team that has not measured it cannot know whether the correction matters. The inference is consistent with the FDA PCLC guidance, but it has not been tested across teams.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | SIL presented as HIL | Algorithm runs on the host | No target timing | Invalid claim | Label the evidence |
| 2 | Rig delay not measured | Latency reported without a loopback | Rig delay attributed to DUT | Wrong margin | Loopback measurement |
| 3 | Runner uses `sleep()` | Jitter comparable to the period | OS scheduling | Stale input | Hardware timer |
| 4 | Renderer buffers samples | Step appears late | Buffer delay | Inflated latency | Measure the step |
| 5 | Fault not demonstrated | Only the DUT output is shown | Fault may not be applied | False pass | Capture the injection |
| 6 | Only nominal scenarios | No fault in the suite | Risk file not covered | Field failure | Risk-based suite |
| 7 | One seed per scenario | Result varies between runs | Random variation | Unstable estimate | Three seeds |
| 8 | p99 from too few events | p99 equals the maximum | Sparse tail | Unstable estimate | 1,000 events |
| 9 | False-alarm rate from a short run | 10-minute run | Rate not estimable | Overconfident claim | 24–30 hours |
| 10 | Training data used as test data | Same records in both | Leakage | Inflated performance | Record-level split |
| 11 | Safe state not verified | Only the detection is shown | Output not captured | Unsafe state | Capture every output |
| 12 | Traceability matrix incomplete | A requirement without a test | Coverage gap | Untested risk | Matrix review |
| 13 | Rig changed after calibration | New cable or attenuator | Calibration invalid | Wrong input | Recalibrate |
| 14 | Host clock used for latency | Wall-clock timestamps | Clock jumps | Negative or wrong latency | Monotonic clock |
| 15 | Scenario not representative | A clean synthetic signal | No noise or motion | Optimistic result | Add realistic impairments |
| 16 | Actuator connected during fault test | Unexpected motion | Stored energy | Injury | Dummy load first |
| 17 | Clock offset not measured | Latency shifts between runs | Two clocks | Wrong latency | Common-event alignment |
| 18 | Scenario regenerated with a new library version | Signal differs from the stored hash | Dependency drift | Non-reproducible test | Pin the generator |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The rig delay is additive and independent | It can be subtracted | A delay that depends on the DUT's output | Compare loopback with and without the DUT |
| 2 | The runner is real time | The timing claim depends on it | A `sleep()` loop, a loaded host | Capture the update jitter |
| 3 | The signal path is calibrated | The DUT sees the specified input | A gain or offset error | Calibrate before the run |
| 4 | The scenarios represent the use | The result transfers | A clean synthetic signal | Add noise, motion, and variability |
| 5 | The annotations are correct | The sensitivity is measured against them | A different annotation version | Hash the annotation file |
| 6 | The split is at the record level | The performance is not inflated | Window-level leakage | Inspect the split |
| 7 | The seeds sample the variability | The estimate is stable | One seed, one run | Three seeds, compare |
| 8 | The p99 is stable at the sample size | The tail is characterised | Too few events | Report the interval |
| 9 | The false-alarm rate is estimable | The rate is meaningful | A short run | 24–30 hours |
| 10 | The fault list is complete | The coverage claim holds | A hazard not in the list | Review against the risk file |
| 11 | The rig does not change the DUT's behaviour | The measurement is of the device | Loading, ground loops, EMC | Compare with a standalone test |
| 12 | The claim boundary is stated | The result is not over-extended | A report without limitations | Review the report |
| 13 | The clocks are aligned | The latency is a difference, not an offset | NTP jump, clock drift | Wrong latency | Common-event alignment |
| 14 | The scenario generator is pinned | The input is reproducible | A library update | Different test | Lockfile and hash |

## 12. Exercises

### Exercise 1. Classify the evidence (recall and application)

**Deliverable:** a table that classifies five test setups as MIL, SIL, PIL, or HIL, with the evidence each can support.

**Acceptance criteria:** each classification is justified by where the algorithm runs and whether the I/O is real; at least one setup is a borderline case and the answer explains why; the table states what each setup cannot prove.

**Time:** 30 minutes.

### Exercise 2. Draw the timing budget (application)

**Deliverable:** a timing budget for a HIL rig, with the delay, the uncertainty, and the measurement method for each stage.

**Acceptance criteria:** at least five stages; each delay is measured, not assumed; the total is compared with the requirement; the rig delay is identified separately from the device delay.

**Time:** 45 minutes.

### Exercise 3. Measure the loopback (application)

**Deliverable:** a loopback measurement of the rig's delay, with 100 runs, a mean, a standard deviation, and an uncertainty.

**Acceptance criteria:** the loopback bypasses the DUT; the measurement is repeated after a rig change; the delay is compared with the latency requirement; the report states whether the correction is valid.

**Time:** 45 minutes.

### Exercise 4. Build a scenario suite (design)

**Deliverable:** a YAML or table of 30 scenarios, each with the expected behaviour, the linked hazard, and the fault, if any.

**Acceptance criteria:** at least 30 scenarios; at least 10 faults; each scenario has an expected behaviour; each is linked to a hazard or a requirement; the suite includes normal, boundary, and fault cases.

**Time:** 90 minutes.

### Exercise 5. Fault injection (application)

**Deliverable:** a capture for each of five faults, showing the injection and the DUT's response.

**Acceptance criteria:** the fault is demonstrated at the electrical or digital interface; the detection time is measured; the safe state is verified; the residual state is recorded.

**Time:** 60 minutes.

### Exercise 6. Latency analysis (calculation)

**Deliverable:** a p50, p95, and p99 analysis of a latency log, with a bootstrap interval and a correction for the rig delay.

**Acceptance criteria:** the percentile method is stated; the sample size is stated; the interval is computed; the correction is justified; the result is compared with a requirement.

**Time:** 45 minutes.

### Exercise 7. Traceability matrix (design)

**Deliverable:** a matrix linking at least eight requirements to risk controls, test cases, results, and evidence files, with the coverage gaps marked.

**Acceptance criteria:** every requirement has a test or a gap; every risk control is linked; the evidence files exist; the coverage is stated as a number and as a narrative.

**Time:** 60 minutes.

### Exercise 8. Critique a HIL claim (critique)

**Deliverable:** a review of a draft claim that says "HIL testing proves the device is safe," identifying the missing evidence and the correct claim.

**Acceptance criteria:** the review distinguishes HIL from clinical evidence; it names the usability, EMC, and population evidence that is missing; it proposes a corrected sentence that preserves the true finding.

**Time:** 30 minutes.

## 13. Self-check answers

### 13.1 What is the difference between SIL, PIL, and HIL?

SIL runs the algorithm on a host. PIL runs it on the target with simulated inputs. HIL runs the target in real time with real I/O and a simulated plant. Only HIL supports a claim about target timing and interface behaviour.

### 13.2 Why is a `sleep()` loop not real time?

Because the operating system can delay the wake-up by an unbounded amount, especially under load. A real-time loop uses a hardware timer and a bounded execution path, and its jitter is measured.

### 13.3 Compute the false-alarm rate for 3 alarms in 30 hours.

\[
\text{FAR} = 3/30 = 0.1\ \text{per hour}
\]

The 95% normal-approximation interval is \(0.1 \pm 1.96\sqrt{3}/30 = 0.1 \pm 0.11\), or 0 to 0.21 per hour.

### 13.4 Why is the p99 from 200 events uncertain?

Because the 99th percentile is the 198th order statistic, which depends on only a few observations in the tail. A different sample of 200 events can produce a different p99. The bootstrap interval quantifies this.

### 13.5 What does a loopback measure?

It measures the delay from the rig's output to the rig's capture input, bypassing the DUT. It is the rig's contribution to the end-to-end latency. It does not measure the DUT's processing time.

### 13.6 Why can subtracting the rig delay be invalid?

Because the delay may not be additive. If the rig's delay depends on the DUT's output, or if the DUT's behaviour changes when the rig is connected, the subtraction is not valid. The correction requires evidence that the delay is additive and independent.

### 13.7 What is the difference between detection latency and recovery time?

Detection latency is the time from the fault to the detection. Recovery time is the time from the detection to the safe state or normal operation. A device can detect quickly and recover slowly.

### 13.8 What does a traceability matrix link?

A requirement, a risk control, a test case, a result, and an evidence file. It is the argument that the tests cover the risks and that the evidence exists.

### 13.9 Why is a 10-minute run insufficient for a false-alarm rate?

Because the rate is a count per unit time. A 10-minute run can observe zero or one alarm, which cannot distinguish a rate of 1 per hour from a rate of 0.1 per hour. A 24–30 hour run gives a usable estimate, and even then the interval is wide for a small count.

### 13.10 Why is code coverage not risk coverage?

Because a line of code can run without the hazard being present. A detector can execute its alarm path in a nominal scenario without ever seeing the arrhythmia it is supposed to detect. Risk coverage requires the hazard to be exercised.

### 13.11 Rubric for Exercise 4

A strong suite has at least 30 scenarios, at least 10 faults, an expected behaviour for each, and a link to a hazard or requirement. It includes normal, boundary, and fault cases. A weak suite has many nominal rhythms and no faults.

### 13.12 Why is the median not enough for a latency claim?

Because the median describes the typical event, not the tail. A device with a 12 ms median and a 25 ms p99 meets a 12 ms requirement for half the events and misses it for the slowest one percent. A control loop and a safety monitor see the tail.

### 13.13 What is the difference between a false alarm and a missed detection?

A false alarm is an alarm without the target condition. A missed detection is the absence of an alarm when the target condition is present. They trade off against each other: a lower threshold reduces misses and increases false alarms.

### 13.14 Why is the HIL rig calibrated like an instrument?

Because the rig's signal path and timing are part of the measurement. A gain error changes the input the DUT sees, and a delay error changes the latency attributed to the DUT. The calibration is the evidence that the rig measures what it claims.

### 13.15 Rubric for Exercise 7

A strong matrix links every requirement to a risk control, a test, a result, and an evidence file, and it marks the gaps honestly. A weak matrix lists tests without requirements, or claims 100% coverage without checking.

## 14. Interview questions

### 14.1 What does HIL prove, and what does it not?

It proves that the integrated system meets its timing and fault-behaviour requirements under the tested conditions. It does not prove usability, clinical benefit, EMC compliance, or performance in the intended population. A strong answer states both sides.

### 14.2 How do you measure end-to-end latency?

Timestamp the input at the source, timestamp the output at the DUT or the capture, and compute the difference for many events. Report the distribution, not only the mean. Correct for the rig delay if it is additive and characterised.

### 14.3 Why is the rig delay important?

Because it is part of the measured latency. If it is not characterised, the report attributes it to the device. A rig with a 20 ms delay can dominate a 30 ms requirement.

### 14.4 What is the difference between a replay and a closed-loop HIL?

A replay is open-loop: the DUT's output does not affect the input. A closed-loop HIL updates the plant model from the DUT's output, so the DUT's behaviour changes the input it receives. The closed loop can reveal stability and interaction effects that a replay cannot.

### 14.5 How do you choose the scenarios?

From the risk file and the intended use. A strong answer starts with the hazards and derives the scenarios, rather than starting with the signals that are easy to generate. It includes normal, boundary, and fault cases.

### 14.6 Why are three seeds needed?

Because the scenarios may include random variation, such as noise or timing jitter. One seed gives one realisation, which may not represent the distribution. Three seeds give a sense of the variation.

### 14.7 What is a fault-injection matrix?

A table that maps each fault to its injection point, detection mechanism, detection time, response, and residual state. It is the evidence that the faults were exercised, not assumed.

### 14.8 How do you verify the safe state?

With an instrument on every output, through the fault, the reset, and the boot. A strong answer notes that a peripheral can keep driving a pin after the CPU resets. A weak answer reads the code.

### 14.9 What is the difference between a requirement and a scenario?

A requirement is a statement of what the system must do. A scenario is a test case that exercises the requirement. A strong answer notes that a scenario without a requirement is an experiment.

### 14.10 Why is a false-alarm rate reported per hour?

Because alarm fatigue and clinical burden are functions of time, not of the number of runs. A rate per run depends on the run length and is not comparable across studies.

### 14.11 What would you do if the measured latency exceeded the requirement?

A strong answer breaks the latency into stages, identifies the dominant stage, and proposes a change to the algorithm, the sample rate, the buffering, or the requirement. It does not simply increase the requirement. A weak answer says "optimise the code."

### 14.12 What is the claim boundary of a HIL result?

It is a statement of what the result does not prove: usability, clinical benefit, EMC, and population performance. A strong answer writes the boundary before the test. A weak answer adds a disclaimer after the test.

## 15. Key terms

- **Actuator delay** — delay from command to motion.
- **Bootstrap** — resampling method for a confidence interval.
- **Closed loop** — system in which the output affects the input.
- **Context of use** — the decision a model supports.
- **Coverage** — extent to which hazards were exercised.
- **Detection latency** — time from fault to detection.
- **Determinism** — same input, same timing, same output.
- **DUT** — device under test.
- **End-to-end latency** — delay from physical input to physical output.
- **Evidence file** — file that supports a test result.
- **False alarm** — alarm without the target condition.
- **False alarm rate** — false alarms per unit time.
- **Fault injection** — deliberate introduction of a fault.
- **HIL** — hardware-in-the-loop.
- **Jitter** — variation in period or latency.
- **Latency** — delay from input to output.
- **Loopback** — signal path that returns the output to the input for timing.
- **MIL** — model-in-the-loop.
- **Model credibility** — evidence that a model is adequate for its context.
- **Monotonic clock** — clock that never goes backwards.
- **Oracle** — rule that decides pass or fail.
- **PCLC** — physiologic closed-loop control.
- **PIL** — processor-in-the-loop.
- **Plant model** — model of the system the DUT controls.
- **Real time** — producing each output before the next input is due.
- **Recovery time** — time from detection to safe state or normal operation.
- **Replay** — open-loop playback of a recorded signal.
- **Requirement ID** — unique identifier for a requirement.
- **Risk control** — action that reduces risk.
- **Safe state** — state in which the risk is acceptable.
- **Scenario** — defined input sequence and expected behaviour.
- **Seed** — initial value for a random process.
- **SIL** — software-in-the-loop.
- **Traceability matrix** — map from requirement to test to evidence.
- **Transport delay** — delay in a bus, queue, or network.
- **V&V 40** — ASME standard for model credibility.
- **Fault-insertion unit** — hardware that opens, shorts, or corrupts a signal.
- **Order statistic** — value at a defined rank in a sorted sample.
- **Percentile** — value below which a defined fraction of the sample lies.
- **Plant** — the physical system the controller acts on.
- **Sample clock** — clock that determines when samples are produced.
- **Test manager** — software that runs scenarios and records results.
- **Traceability** — ability to link a result to a requirement and an evidence file.
- **Clock offset** — difference between two clocks at a defined instant.
- **Common-event alignment** — using a shared signal to align two clocks.
- **Fault coverage** — extent to which the fault list was injected.
- **Risk coverage** — extent to which the risk controls were exercised.
- **Scenario coverage** — extent to which the clinical scenarios were run.
- **Signal conditioning** — circuit that matches a signal to an input.
- **Test oracle** — rule or data that decides the expected result.
- **Validation** — evidence that the device meets user needs.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Standards and regulation

- FDA, Technical Considerations for Medical Devices with Physiologic Closed-Loop Control Technology (September 2023). https://www.fda.gov/media/154994/download
- ASME V&V 40, Assessing Credibility of Computational Modeling through Verification and Validation. https://www.asme.org/
- IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes. https://www.iso.org/standard/64686.html
- ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
- IEC 60601-1-10, Requirements for the development of physiologic closed-loop controllers. https://webstore.iec.ch/
- IEC 62366-1, Application of usability engineering to medical devices. https://webstore.iec.ch/
- IEC 60601-1-2:2014+AMD1:2020, Electromagnetic disturbances. https://webstore.iec.ch/en/publication/67554

### HIL practice and case evidence

- Karunagaran, M., Sahoo, S., and Fujita, T. (2015). Hardware in loop testing of an insulin pump. IEEE International Test Conference. https://doi.org/10.1109/test.2015.7342416
- Hottenbacher, S., and Haase, T. (2023). Functional testing and performance evaluation of networked medical devices with hardware-in-the-loop simulation. IEEE IECON. https://doi.org/10.1109/iecon51785.2023.10311716
- Pickering Test, Hardware-in-the-Loop Aids Medical Device Compliance. https://www.pickeringtest.com/solutions/by-industry/medical-device-testing/hardware-in-the-loop-aids-medical-device-compliance
- RX-Systems, IEC 62304 Verification for Medical Devices. https://www.rx-systems.de/en/hil-medizintechnik/

### Software and tooling

- ROS 2 documentation. https://docs.ros.org/
- Gazebo documentation. https://gazebosim.org/docs
- MuJoCo documentation. https://mujoco.readthedocs.io/
- Python documentation, `random` and `statistics` modules. https://docs.python.org/3/library/
- IEEE 11073, Health informatics — Point-of-care medical device communication. https://standards.ieee.org/
- ISO 14971:2019, risk management, Annex C, hazard and hazardous situation examples. https://www.iso.org/standard/72704.html
- IEC 62366-1:2015+AMD1:2020, Application of usability engineering to medical devices. https://webstore.iec.ch/

> **Note:** The standard texts are paywalled. The guidance and papers above were used to check the teaching values. Confirm every clause and requirement against the current edition before relying on it in a regulatory submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_emc-labs.md`](06_emc-labs.md)
- Next topic: [`08_robotics-labs.md`](08_robotics-labs.md)
- Sibling: [`04_sensor-and-signal-labs.md`](04_sensor-and-signal-labs.md)
- Sibling: [`05_firmware-labs.md`](05_firmware-labs.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
