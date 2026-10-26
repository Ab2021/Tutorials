# Validation on Target Hardware

> Module 14B — Efficient AI and On-Device Inference
> Topic 10 of 11
> Estimated study time: 4 hours
> Prerequisites: 02, 08
> Version 2026.3

## 1. Why this topic exists

A measurement on a workstation is a hypothesis about the device. Validation on target hardware is the experiment that turns the hypothesis into evidence. The gap between the two is not small: it is routinely a factor of two in latency, a factor of three in energy, and a difference in which subgroup fails.

The gap has three sources.

**The environment.** A workstation has a fan, a stable supply, and no enclosure. A device has a battery, a sealed case, and a skin-temperature limit. The same model that runs at 45 ms on a bench can run at 65 ms after twenty minutes in a pocket.

**The runtime.** A workstation runs a desktop runtime with a full operator set and a large memory allocator. A device runs a stripped runtime with a static arena, a subset of operators, and a different kernel selection. The graph that runs on the workstation is not the graph that runs on the device.

**The data.** A test set is a curated sample. A device sees sensor noise, motion artifact, poor contact, and inputs that the test set never contained. The model's accuracy on the test set is an upper bound on its accuracy in the field, not an estimate of it.

Validation on target hardware is the discipline of closing these gaps with measurement. It requires a rig that injects real data, a protocol that controls the clock and thermal state, a statistical plan that can support the claim, and a report that states the conditions.

Three industry failures recur.

**The cold-burst validation.** Latency is measured for 30 seconds after boot, at room temperature, with the debugger attached. The device passes. In the field, after ten minutes, the clock drops and the p99 exceeds the deadline. The validation measured the best case and reported it as the case.

**The accuracy-only validation.** The team runs the held-out set through the device and confirms that accuracy matches the workstation. The latency, energy, and thermal behavior are never measured, and the device fails its battery or deadline requirement after launch.

**The underpowered non-inferiority claim.** The team compares the compressed model with the baseline on 200 cases, finds no significant difference, and declares equivalence. The confidence interval spans four percentage points, and the claim is not supported by the data. The test was not designed to answer the question.

This topic teaches the rig, the protocol, the statistics, and the report. It assumes the device architecture from Topic 08 and the compression from Topics 03–07.

### 1.1 The four gaps between bench and field

| Gap | Bench | Field | Test that closes it |
|---|---|---|---|
| Thermal | 22 °C, open air | 40 °C, sealed case | Soak at ambient |
| Power | Bench supply | Battery, end-of-life voltage | Battery simulator; rail monitor |
| Runtime | Desktop operator set | Stripped runtime, static arena | HIL with production firmware |
| Data | Curated test set | Noise, artifact, poor contact | Real sensor replay |

A validation that closes only one gap is a partial validation. The report should name the gaps it closed and the ones it did not.

### 1.2 Three claims a validation must separate

- **The device meets the claim.** A statement about the measured configuration, with intervals.
- **The device will meet the claim in the field.** A statement about the population and the environment, which requires an extrapolation.
- **The device is safe.** A clinical and risk judgement, which no measurement alone can establish.

A report that conflates the three over-claims. A report that separates them lets the reader see which part is evidence and which part is judgement.

## 2. Learning outcomes

By the end of this topic you can:

1. **Design** a hardware-in-the-loop rig that injects real sensor data and measures latency, energy, and temperature.
2. **Measure** p50, p95, and p99 latency with a stated warm-up, clock state, and sample size.
3. **Integrate** a power trace to obtain energy per inference, with the inclusion set and uncertainty stated.
4. **Run** a thermal soak at the specified ambient and identify the time to throttle.
5. **Compute** the sample size for a non-inferiority claim on target hardware.
6. **Explain** what an emulator cannot tell you and how to bound the gap.
7. **Report** the results so that another engineer can reproduce them.
8. **Decide** whether the device meets its claim, and state the residual uncertainty.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A rig diagram with instrumentation |
| 2 | Exercise 2 | Percentiles with conditions |
| 3 | Exercise 3 | An energy calculation with uncertainty |
| 4 | Exercise 4 | A soak result and throttle time |
| 5 | Exercise 5 and Section 6 | A sample-size calculation |
| 6 | Exercise 6 | An emulator gap analysis |
| 7 | Exercise 7 | A reproducible report |
| 8 | Exercise 8 | A decision with residual uncertainty |

## 3. The core mental model

Validation is a chain from the claim to the evidence. Each link has a condition that must be controlled.

```text
   claim: p99 ≤ 55 ms, energy ≤ 1.5 mJ, sensitivity ≥ 0.925
        │
        ▼
   conditions: hardware, runtime, ambient, battery, enclosure, input data
        │
        ▼
   rig: data injection, clock control, power measurement, temperature control
        │
        ▼
   protocol: warm-up, sample size, percentile method, inclusion set
        │
        ▼
   statistics: point estimate, confidence interval, non-inferiority test
        │
        ▼
   decision: pass, fail, or inconclusive
```

A break in any link makes the conclusion unsupported. A perfect rig with a 30-sample test cannot support a p99 claim. A perfect statistical plan on a cold bench cannot support a field latency claim.

### 3.1 The four measurement domains

| Domain | What is measured | Instrument | Main confounder |
|---|---|---|---|
| Latency | Time per inference | GPIO marker, timer, trace | Clock state, warm-up |
| Energy | Joules per inference | Power monitor, shunt | Inclusion set, sampling rate |
| Thermal | Temperature and clock over time | Thermocouple, on-die sensor | Ambient, enclosure |
| Accuracy | Decisions on real inputs | Held-out set through the device | Input distribution, preprocessing |

The four are measured together, because they interact. A latency measurement at a low clock is an energy measurement at a low power; a thermal measurement is a latency measurement over time.

### 3.2 The p99 problem

The 99th percentile is a tail statistic, and tail statistics need many samples.

- With 100 samples, the p99 is the second-largest value, and its uncertainty is enormous.
- With 1,000 samples, the p99 is the tenth-largest value, and the uncertainty is still wide.
- With 10,000 samples, the p99 is estimated with a bootstrap interval of roughly ±0.3 ms in the Section 6 case.

A p99 claim based on 100 samples is not a p99 claim; it is a maximum claim with a small sample. The sample size must be stated, and the interval must be reported.

The same logic applies to p99.9. A claim at p99.9 needs at least 10,000 samples to have ten observations in the tail, and preferably 100,000.

### 3.3 Warm-up and clock state

The first inference after boot is not representative. It includes:

- Model load and memory allocation.
- Cache fill.
- Clock ramp from the low-power state.
- Runtime initialization and kernel selection.

A warm-up protocol runs a fixed number of inferences before measurement, or waits until the latency distribution stabilizes. The number is device-specific; 10–100 inferences is common for a small model.

The clock state is the second confounder. A device that has been idle may run at a boost clock for the first few milliseconds and then settle. A device that has been busy may be throttled. The measurement must state whether the clock was pinned, and if not, must record it.

| State | Latency | When it occurs |
|---|---|---|
| Cold | 3–10× warm | First inference after boot |
| Boost | 0.8–1.0× warm | First few milliseconds after idle |
| Warm | Baseline | Steady state |
| Throttled | 1.2–1.5× warm | After sustained load |

### 3.4 Energy per inference

Energy is the integral of power over the inference window:

```text
E = ∫ P(t) dt ≈ Σ P_i × Δt
```

The measurement requires a power monitor with a sampling rate high enough to capture the inference. For a 45 ms inference, a 1 kHz monitor gives 45 samples, which is adequate for a total but not for the shape. A 10 kHz monitor gives 450 samples and reveals the phases.

The inclusion set matters as much as the instrument. Does the energy include:

- The sensor and its analog front end?
- The preprocessing?
- The radio?
- The display?
- The sleep current between inferences?

A number without the inclusion set is not comparable with another number. State it.

### 3.5 Thermal soak

A thermal soak runs the device at its clinical duty cycle, in its enclosure, at the specified ambient, until the temperature and latency stabilize. The outputs are:

- The time to throttle.
- The steady-state clock and latency.
- The case or skin temperature.
- The recovery time after the load stops.

The soak duration must match the clinical use. A device that runs continuously needs a multi-hour soak; a device that runs for 30 seconds per hour may never reach steady state.

### 3.6 What an emulator cannot tell you

An emulator models arithmetic and memory access, not physics.

| Emulator models | Emulator does not model |
|---|---|
| Operator semantics | Clock frequency and DVFS |
| Approximate cycle counts | Cache and bus contention |
| Memory size | Thermal throttling |
| Some operator support | Kernel selection by the runtime |
| Numerical results | Power draw and battery behaviour |
| | Sensor noise and analog front end |
| | Enclosure and skin temperature |

An emulator is useful for functional testing and for early feasibility. It is not evidence for a latency, energy, or thermal claim. The gap must be measured, not assumed.

### 3.7 The measurement is an experiment

A validation is an experiment with a hypothesis (the device meets the claim), a treatment (the device configuration), a control (the reference or baseline), and a measurement (the protocol). The statistical design is what makes the experiment informative.

An experiment without a control answers "what is the value" but not "is it different". An experiment without a pre-specified analysis answers a question chosen after the data. An experiment without a power calculation may be unable to answer the question at all.

### 3.8 The cost of a validation

A full validation is expensive: a rig, a thermal chamber, hundreds of device-hours, and a statistical analysis. The cost is justified when the claim is clinical and the consequence of a miss is a hazard.

A cheaper validation can be appropriate for an early design decision, provided the report labels it as a screening test and not as evidence. The label is what prevents a screening result from being cited as a validation years later.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| HIL | Hardware-in-the-loop | Confused with a simulator | Method |
| Rig | The test fixture and instrumentation | Confused with the device | System |
| p50 | Median latency | Reported as the latency | ms |
| p95 | 95th percentile | Confused with p99 | ms |
| p99 | 99th percentile | Estimated from too few samples | ms |
| Percentile method | Interpolation or nearest-rank rule | Assumed identical across tools | Method |
| Bootstrap interval | Resampling-based confidence interval | Confused with a normal interval | Interval |
| Warm-up | Inferences before measurement | Omitted under time pressure | Count |
| Cold start | First inference after boot | Mixed into warm statistics | State |
| Boost clock | Temporary higher clock | Assumed to persist | Hz |
| DVFS | Dynamic voltage and frequency scaling | Ignored in latency claims | Mechanism |
| Throttling | Clock reduction under thermal load | Assumed not to occur | Mechanism |
| Thermal soak | Sustained load until stabilization | Confused with a short test | Test |
| Time to throttle | Interval before clock reduction | Not measured | s |
| Steady state | Stable temperature and latency | Assumed after a few minutes | State |
| Power monitor | Instrument measuring voltage and current | Confused with a fuel gauge | Instrument |
| Shunt resistor | Current-sense resistor | Assumed lossless | Ω |
| Sampling rate | Samples per second | Assumed adequate | Hz |
| Inclusion set | What the energy measurement includes | Unstated | Set |
| Energy per inference | Joules for one inference | Confused with average power | mJ |
| Average power | Energy divided by time | Confused with peak power | mW |
| Duty cycle | Fraction of time active | Confused with utilization | Proportion |
| Ambient | Surrounding temperature | Confused with case temperature | °C |
| Case temperature | Temperature of the enclosure | Confused with skin temperature | °C |
| Skin temperature | Temperature at the patient contact | Assumed equal to case | °C |
| Non-inferiority | Not worse by more than a margin | Confused with superiority | Claim |
| Margin | Largest acceptable difference | Chosen after results | Proportion |
| Power | Probability of detecting a true difference | Confused with energy | Proportion |
| Alpha | Type I error rate | Confused with power | Proportion |
| Sample size | Number of observations | Chosen by convenience | Count |
| Paired design | Same inputs for both models | Confused with unpaired | Design |
| Emulator | Software model of the device | Confused with the device | Tool |
| Functional test | Correctness test | Confused with performance test | Test |
| Trace | Recorded timing of events | Confused with a log | Artifact |
| GPIO marker | Pin toggled at an event | Assumed instantaneous | Signal |
| Instrumentation overhead | Effect of measurement on the system | Ignored | ms |
| Reproducibility | Ability to repeat the result | Confused with repeatability | Property |

Two definitions need emphasis.

**p99 is not a maximum.** It is the value below which 99% of observations fall. A p99 of 48 ms is compatible with a maximum of 54 ms, and the maximum is what a single patient may experience. Report both if the clinical consequence depends on the worst case.

**Energy per inference is not average power.** A device that draws 24 mW for 45 ms and 3 µW for the rest of the cycle has a very different average power from a device that draws 24 mW continuously. The energy per inference is the integral over the active window; the average power is the integral over the duty cycle.

### 4.1 The units of a validation report

| Quantity | Unit | Typical precision |
|---|---|---|
| Latency | ms | 0.1 ms |
| Energy | mJ | 0.001 mJ |
| Power | mW | 0.01 mW |
| Temperature | °C | 0.1 °C |
| Sensitivity | proportion | 0.001 |
| Sample size | cases | integer |
| Margin | percentage points | 0.1 pp |

Reporting more precision than the instrument supports is a defect, not a virtue. A latency of 48.4432 ms from a timer with 0.1 ms resolution is a false precision.

## 5. The mechanics, step by step

The procedure produces a validated measurement package. It assumes the device, the runtime, and the clinical claim.

### Step 1. Restate the claim and its conditions

**Input:** The clinical claim and the constraint ledger.

**Output:** A measurement claim with a metric, a value, a percentile, and a condition.

**Rationale:** A claim that cannot be measured cannot be validated. The restatement makes the measurement target explicit.

Examples:

- "p99 inference latency ≤ 55 ms on the production board, warm, batch 1, at 22 °C ambient and 40 °C ambient."
- "Energy per inference ≤ 1.5 mJ including sensor, preprocessing, and inference, excluding radio."
- "Sensitivity ≥ 0.925 on the held-out set, at the fixed operating threshold, overall and in each subgroup."

**Branch point:** If the claim has no condition, the measurement cannot be interpreted. Add the condition before measuring.

**Edge case:** A claim that is conditional on the battery state must state the state. A latency measured at 3.0 V may not hold at 2.4 V.

### Step 2. Build the HIL rig

**Input:** The device, the sensor interface, and the instrumentation.

**Output:** A rig that injects real data and measures latency, energy, and temperature.

**Rationale:** The rig determines what can be measured. A rig that cannot inject real data measures the model on synthetic inputs; a rig without a power monitor cannot measure energy.

The rig components:

| Component | Purpose | Notes |
|---|---|---|
| Host | Generates or replays sensor data | Stores the held-out set |
| DAC or digital interface | Injects the signal | Must match the sensor's range |
| GPIO marker | Marks inference start and end | Read by an oscilloscope or logic analyzer |
| Power monitor | Measures voltage and current | 1–10 kHz, shunt on the battery rail |
| Thermocouple | Measures case temperature | Attached at the skin contact |
| Thermal chamber | Controls ambient | 22 °C and 40 °C |
| Battery simulator | Controls supply voltage | Tests brownout and end-of-life |
| Logic analyzer | Captures markers and bus traffic | Timestamps at µs resolution |

**Branch point:** If the sensor is analog, the injection must match its range and impedance. A digital injection that bypasses the analog front end measures a different pipeline.

**Edge case:** The GPIO marker itself costs time. Toggle it at the earliest and latest points, and measure the overhead with an empty inference. Subtract it or include it consistently.

### Step 3. Define the warm-up and clock protocol

**Input:** The device's boot and power states.

**Output:** A protocol that states how many warm-up inferences, whether the clock is pinned, and how the clock is recorded.

**Rationale:** Cold and boost states are not representative. The protocol removes them or records them.

The protocol:

1. Boot the device and wait for the self-test.
2. Run 50 warm-up inferences (or until the p99 changes by less than 2% between blocks).
3. Pin the clock if the device allows it; otherwise record the clock at 1 Hz during measurement.
4. Measure for the planned number of samples.
5. Repeat at the specified ambient temperatures.

**Branch point:** If the clock cannot be pinned, the measurement includes DVFS. Report the clock distribution and the latency conditional on the clock.

**Edge case:** A device that sleeps between inferences must be woken for each measurement, and the wake time is part of the latency. A measurement that keeps the device awake measures a different product.

### Step 4. Sample latency and compute percentiles

**Input:** The latency samples and the percentile method.

**Output:** p50, p95, p99, p99.9, maximum, and a confidence interval for p99.

**Rationale:** The percentile is a summary of a distribution, and the summary is only as good as the sample.

The method:

```text
sort the samples ascending
p = (n - 1) × percentile
lower = floor(p), upper = ceil(p)
value = x[lower] + (x[upper] - x[lower]) × (p - lower)
```

The bootstrap interval:

1. Resample the samples with replacement, 1,000 times.
2. Compute p99 for each resample.
3. Take the 2.5th and 97.5th percentiles of the resampled values.

**Branch point:** If the sample size is below 1,000, report the maximum and the p95, and state that p99 is not estimable with useful precision.

**Edge case:** A bimodal distribution (some inputs hit a fallback) has a p99 that depends on the mix. Report the modes and the input conditions that produce them.

### Step 5. Measure cold start and wake

**Input:** The device and a power switch.

**Output:** Cold-start latency, wake latency, and the time to first correct output.

**Rationale:** A device that must respond after power-on is bound by cold start, not warm latency.

The measurement:

- Power off, wait, power on, and time to the first output.
- Put the device to sleep, wake it, and time to the first output.
- Repeat 30 times and report the distribution.

**Branch point:** If cold start exceeds the availability requirement, the device needs a different boot design, not a faster model.

**Edge case:** A cold start that includes a sensor warm-up (an optical front end, for example) may be dominated by the sensor, not the model.

### Step 6. Measure energy with a power monitor

**Input:** The power trace and the inference window.

**Output:** Energy per inference, average power, and the inclusion set.

**Rationale:** Energy is the integral of power. The window and the inclusion set determine the number.

The method:

1. Place a shunt resistor on the battery rail.
2. Sample voltage and current at 1–10 kHz.
3. Align the trace with the GPIO markers.
4. Integrate power over the window.
5. Repeat 100 times and report the mean and standard deviation.
6. State the inclusion set.

**Branch point:** If the monitor's sampling rate is too low to capture the inference, the integral is an average over the window and misses the peaks. Increase the rate or lengthen the window.

**Edge case:** The shunt resistor adds a voltage drop that can affect the device, especially at low battery. Use a small shunt and a high-gain amplifier, and verify that the device behaves the same with and without the shunt.

### Step 7. Run the thermal soak

**Input:** The device in its enclosure, the thermal chamber, and the clinical duty cycle.

**Output:** Temperature and latency over time, time to throttle, steady-state values.

**Rationale:** The field constraint is sustained performance. A device that meets its deadline for 30 seconds and misses it after 10 minutes has not met its deadline.

The soak:

1. Place the device in the enclosure at the specified ambient.
2. Run the clinical duty cycle for at least 60 minutes, preferably 4–24 hours.
3. Record case temperature, clock, and latency at 1 Hz.
4. Identify the time at which the p99 first exceeds the cold p99 by more than the tolerance.
5. Record the steady-state values and the recovery time.

**Branch point:** If the device throttles before the end of the clinical duty cycle, the design must reduce sustained power, improve the thermal path, or relax the latency claim.

**Edge case:** A device that is idle most of the time may never throttle in the field even if it throttles in a continuous soak. Match the soak to the duty cycle.

### Step 8. Validate accuracy and calibration on the device

**Input:** The held-out set and the device's output.

**Output:** Accuracy, calibration, and subgroup results from the device, not the workstation.

**Rationale:** The device's preprocessing, quantization, and runtime can change the outputs. The device's results are the ones that matter.

The procedure:

1. Run the held-out set through the device, using the same input path as deployment.
2. Record the raw outputs and the decisions at the fixed threshold.
3. Compare with the workstation reference.
4. Compute accuracy, calibration, and subgroup metrics.
5. Inspect the disagreements.

**Branch point:** If the device's accuracy differs from the workstation's by more than the numerical tolerance, the export or the preprocessing changed the model. Investigate before proceeding.

**Edge case:** A device that applies a different filter or scaling than the workstation is a different model. Record the preprocessing as part of the configuration.

### Step 9. Size the sample for the non-inferiority claim

**Input:** The margin, the expected difference, the variance, and the power.

**Output:** The number of cases per group.

**Rationale:** A non-inferiority claim needs enough cases to exclude the margin. A small sample produces a wide interval and an inconclusive result.

The formula for two proportions:

```text
n per group = (z_(1-α) + z_(1-β))² × [p1(1-p1) + p2(1-p2)] / (margin - |p1 - p2|)²
```

For a one-sided α of 0.025, power 0.8, p1 = 0.942, p2 = 0.938, and a margin of 0.015:

```text
z = 1.96 + 0.8416 = 2.8016; z² = 7.849
numerator = 7.849 × (0.0546 + 0.0578) = 0.882
denominator = (0.015 - 0.004)² = 0.000121
n = 0.882 / 0.000121 = 7,289 per group
```

**Branch point:** If the required sample exceeds the available data, the margin must be relaxed, the expected difference reduced, or the claim changed. Relaxing the margin after seeing the data is not an option.

**Edge case:** A paired design can reduce the required sample because the same cases are used for both models. The formula above assumes unpaired; a paired analysis should use the variance of the paired differences.

### Step 10. Report, decide, and record

**Input:** All measurements and the statistical analysis.

**Output:** A validation report and a decision.

**Rationale:** The report is the evidence. A decision without a report is an opinion.

The report includes:

- The claim and its conditions.
- The device configuration: board, firmware, runtime, model hash.
- The rig and the instrumentation.
- The warm-up and clock protocol.
- The sample sizes and percentile method.
- The latency, energy, and thermal results, with intervals.
- The accuracy, calibration, and subgroup results from the device.
- The emulator gap analysis.
- The decision and the residual uncertainty.

**Branch point:** If a result is inconclusive, the decision is "inconclusive", not "pass". An inconclusive result is a finding that requires more data or a different design.

**Edge case:** A result that passes at 22 °C and fails at 40 °C is a fail for a device intended for both. Report both and decide on the worse case.

### 5.1 Decision table for validation outcomes

| Observation | Likely cause | Decision |
|---|---|---|
| p99 within claim at 22 °C and 40 °C | Design meets target | Pass |
| p99 within claim at 22 °C, exceeds at 40 °C | Thermal throttling | Fail, or restrict the ambient claim |
| Cold start exceeds availability | Boot design | Fail, or change the availability claim |
| Energy within claim but battery life short | Sleep current or radio | Investigate the system, not the model |
| Accuracy matches workstation, subgroup fails | Subgroup data or calibration | Fail the subgroup claim |
| Accuracy differs from workstation | Export or preprocessing | Investigate before deciding |
| p99 interval crosses the claim | Too few samples | Inconclusive; collect more |
| Bimodal latency | Fallback or input-dependent path | Report the modes; fix the path |

### 5.2 Edge cases in target validation

**Outlier channels.** A latency outlier can come from a single input that triggers a slow path. Correlate the latency samples with the input characteristics, not only with time.

**Per-tensor versus per-channel scales.** A per-channel model may be slower on a runtime that implements per-channel dequantization in the inner loop. Measure both if the runtime supports both.

**Activation clipping.** Clipping can change the saturation rate and, through it, the accuracy on tails. Measure the saturation rate on the device, not only the accuracy.

**Operator fallback.** Fallback produces a bimodal latency distribution. The p99 is dominated by the fallback path. Report the fraction of inputs that hit it.

**Unsupported operators.** A model that loads but falls back may pass accuracy and fail latency. Check the partition on the device.

**Zero-point drift.** A device that runs for months may drift. The validation is a snapshot; the monitoring plan must cover drift.

**Calibration-set bias.** The device's calibration data is fixed at build time. A validation on a different population measures a different model.

**Thermal throttling.** The soak is the only way to see it. A short validation cannot.

### 5.3 Measurement uncertainty and replicates

Every measurement has uncertainty, and a validation report should state it.

- **Latency:** the timer resolution, the marker overhead, and the run-to-run variation. Report the bootstrap interval for the percentile, not only the point estimate.
- **Energy:** the shunt tolerance, the monitor's gain error, and the sampling rate. Report the mean and standard deviation over at least 100 inferences.
- **Temperature:** the thermocouple accuracy and the placement. Report the location, because case and skin temperatures differ.
- **Accuracy:** the sampling error of the held-out set, expressed as a confidence interval.

Replicates are the defense against a single anomalous run. Measure at least three sessions on different days, or with different input order, and compare the distributions. A result that changes between sessions is a result that depends on a condition the protocol did not control.

| Quantity | Minimum replicates | Reported statistic |
|---|---|---|
| Latency | 3 sessions × 1,000 samples | p50, p95, p99 with intervals |
| Energy | 100 inferences | Mean and standard deviation |
| Thermal | 1 soak of clinical duration | Time to throttle, steady state |
| Accuracy | 1 held-out evaluation | Point estimate and 95% CI |

### 5.4 When the device fails

A failed validation is not the end of the project; it is a diagnosis. The response depends on which claim failed.

| Failed claim | First diagnosis | Likely remedy |
|---|---|---|
| p99 at 22 °C | Model or runtime | Optimize the graph or reduce MACs |
| p99 at 40 °C only | Thermal | Reduce sustained power or improve cooling |
| Cold start | Boot design | Lazy initialization, smaller image |
| Energy | Inclusion set or radio | Move the decision on-device; reduce transmissions |
| Accuracy | Export or preprocessing | Compare intermediate outputs; fix the export |
| Subgroup | Calibration data | Stratified calibration; per-channel scales |
| Calibration | Activation clipping | Re-tune the range; recalibrate |

The remedy should be tested with the same protocol, not a shortened one. A fix that passes a 30-second test has not been validated.

### 5.5 A validation checklist

- The claim states the metric, value, percentile, and conditions.
- The device configuration is recorded.
- The rig injects real data through the deployment path.
- The warm-up and clock protocol are stated.
- The sample size is justified by a power or precision calculation.
- The percentile method and interval are stated.
- The energy inclusion set and monitor rate are stated.
- The thermal soak matches the clinical duty cycle.
- Accuracy is measured on the device, not the workstation.
- The statistical design matches the claim.
- The report states the residual uncertainty.
- The raw data is retained.


## 6. Worked example

This is a constructed case. The device and the measurements are illustrative, and the arithmetic is reproduced in Section 7.

### 6.1 The claim and the device

The claim is p99 inference latency at most 55 ms, energy at most 1.5 mJ per inference including sensor and preprocessing but excluding radio, and sensitivity at least 0.925 overall and at least 0.915 in each subgroup.

The device is the ECG patch: Cortex-M4F at 64 MHz, 256 KiB SRAM, 1 MiB flash, 41,216-parameter INT8 model, 41,816 bytes.

### 6.2 Scale and zero point by hand

The first convolution has 40 weights with a minimum of -0.061 and a maximum of 0.412.

```text
Affine: S = 0.473 / 255 = 0.00185490; Z = round(0.061 / S) = 33
q(0.412) = 255; q(-0.061) = 0

Symmetric: S = 0.412 / 127 = 0.00324409; Z = 0
```

The per-value error is bounded by half a step: 0.000927 in affine, 0.001622 in symmetric. That bound is what the numerical comparison tests.

### 6.3 Memory

| Representation | Bytes | KiB |
|---|---:|---:|
| FP32 weights | 164,864 | 161.0 |
| INT8 per-channel | 41,816 | 40.8 |
| Peak SRAM (weights plus activations) | 136 KiB | 136.0 |

The memory measurement on the device is the runtime high-water mark, 136 KiB, not the model file size. The two differ by a factor of 3.3.

### 6.4 Latency at 22 °C

The protocol runs 50 warm-up inferences, pins the clock at 64 MHz, and collects 10,000 samples on real sensor data.

| Statistic | Value (ms) |
|---|---:|
| p50 | 41.97 |
| p95 | 46.30 |
| p99 | 48.44 |
| p99.9 | 51.19 |
| Maximum | 53.71 |
| Mean | 42.06 |

The p99 is 48.44 ms, which is 6.56 ms below the 55 ms claim. The maximum is 53.71 ms, still below the claim, but only by 1.29 ms.

The bootstrap 95% interval for p99, from 400 resamples, is [48.17, 48.69]. The interval is narrow because the sample is large. With 1,000 samples, the same interval would be roughly three times wider.

### 6.5 Cold start and wake

| Measurement | Value | Condition |
|---|---:|---|
| Cold start to first output | 1,240 ms | Power-on, model load, self-test |
| Wake to first output | 38 ms | Sleep, sensor restore, inference |
| Warm inference | 48.44 ms p99 | Steady state |

Cold start is 1.24 s, which is not part of the inference claim but is part of the availability requirement. If the device must alert within 2 seconds of power-on, it passes; if within 1 second, it fails.

### 6.6 Energy per inference

The power monitor samples at 1 kHz on the battery rail. The inference window is 45.7 ms, giving 46 samples. The mean power is 24.34 mW.

```text
E = 24.34 mW × 46 ms = 1.120 mJ
```

With the preprocessing included, the cycle energy is:

```text
E_preprocess = 3.0 V × 0.0045 A × 0.050 s = 0.675 mJ
E_total = 1.120 + 0.675 = 1.795 mJ
```

Against the 1.5 mJ claim, the inference-only energy passes (1.120) and the sensor-plus-preprocessing energy fails (1.795). The claim's inclusion set decides the verdict, which is why the inclusion set must be stated.

The monitor's uncertainty is ±3%, or ±0.033 mJ on the inference. The claim's margin is larger than the uncertainty, so the verdict is not sensitive to the instrument.

### 6.7 Thermal soak

The device runs its clinical duty cycle (one inference every 2 seconds) in its enclosure at 40 °C ambient.

| Time (min) | Case temperature (°C) | Clock (MHz) | p99 (ms) |
|---:|---:|---:|---:|
| 0 | 26 | 64 | 48.4 |
| 5 | 34 | 64 | 48.9 |
| 10 | 39 | 64 | 49.8 |
| 12 | 41 | 48 | 60.8 |
| 20 | 42 | 48 | 61.2 |
| 60 | 42 | 48 | 61.4 |

The device throttles at 12 minutes, and the steady-state p99 is 61.4 ms, which exceeds the 55 ms claim by 6.4 ms. The cold measurement passed and the sustained measurement failed.

The throttle factor is 64/48 = 1.333, and the latency ratio is 61.4 / 48.4 = 1.269, slightly less than the clock ratio because some of the latency is sensor-bound and does not scale with the clock.

### 6.8 Accuracy, calibration, and subgroups on the device

| Metric | Workstation reference | Device | Difference |
|---|---:|---:|---:|
| Sensitivity | 0.942 | 0.938 | -0.4 pp |
| Specificity | 0.961 | 0.958 | -0.3 pp |
| ECE | 0.021 | 0.029 | +0.008 |
| Slope | 0.97 | 0.93 | -0.04 |
| Worst subgroup | 0.931 | 0.926 | -0.5 pp |

The device matches the workstation within the quantization tolerance. The subgroup difference is -0.5 pp, which passes the -1.0 pp margin with a 95% CI of [-1.4, +0.4].

### 6.9 Sample size for the non-inferiority claim

The margin is 1.5 pp, the expected difference is 0.4 pp, p1 = 0.942, p2 = 0.938, one-sided α = 0.025, power 0.8.

```text
z = 1.96 + 0.8416 = 2.8016; z² = 7.849
p1(1-p1) = 0.942 × 0.058 = 0.05464
p2(1-p2) = 0.938 × 0.062 = 0.05816
numerator = 7.849 × 0.1128 = 0.8854
denominator = (0.015 - 0.004)² = 0.000121
n = 0.8854 / 0.000121 = 7,317 per group
```

The available held-out set has 8,000 positives per group, so the study is adequately powered. If the margin were 1.0 pp, the sample would be 24,592 per group, which exceeds the available data and would make the claim inconclusive.

If the expected difference were zero, the sample would be 3,812 per group. The difference between 3,812 and 7,317 is the price of assuming a 0.4 pp loss rather than hoping for none.

### 6.10 Diagnostic one: outlier-driven scale collapse

The fully connected layer's weights have a median absolute value of 0.0610 and a maximum of 1.2912. The symmetric scale is 0.01016693, or 16.7% of the median.

```text
step / median = 0.1667
weights rounded to zero = 1,411 (4.3%)
distinct levels used = 102 of 255
SQNR = 30.00 dB
```

On the device, the consequence is a small accuracy loss that is visible in the tail, not in the mean. The diagnostic is included in the validation report because it explains why the device's sensitivity differs from the workstation's.

### 6.11 Diagnostic two: per-channel skew

The 512 channels have ranges from 0.2756 to 2.3202, a ratio of 8.4. Per-channel scales raise SQNR to 44.67 dB. The device uses per-channel weights, so the skew is controlled; the diagnostic confirms that the metadata is present and the scales are applied on the correct axis.

A layout or export error that applies the per-channel scales to the wrong axis would produce a large accuracy loss, not a subtle one. The validation would catch it, but the diagnostic explains where to look.

### 6.12 Varying one assumption

**Margin: 1.5 pp to 1.0 pp.** The required sample rises from 7,317 to 24,592 per group. The available data is insufficient, so the claim becomes inconclusive. The options are to accept the wider margin, collect more data, or use a paired design to reduce the variance.

**Ambient: 22 °C to 40 °C.** The steady-state p99 rises from 48.4 to 61.4 ms because the clock drops from 64 to 48 MHz. The claim fails at 40 °C. The options are to reduce the model's work, improve the thermal path, or restrict the ambient claim.

**Sample size: 10,000 to 1,000.** The p99 point estimate remains about 48.4 ms, but the bootstrap interval widens from [48.17, 48.69] to roughly [47.5, 49.4]. The claim still passes, but the evidence is weaker, and a p99.9 claim would not be supportable.

**Energy inclusion: inference only to inference plus preprocessing.** The measured energy rises from 1.120 to 1.795 mJ, and the claim changes from pass to fail. The inclusion set is a definition, and the definition decides the verdict.

### 6.13 What the case does not show

- **The production units.** The case is one device. Unit-to-unit variation in flash, sensors, and batteries can move the margins.
- **The field input distribution.** The held-out set is curated. Motion artifact and poor contact are under-represented.
- **The long-term drift.** The validation is a snapshot; the monitoring plan must cover months.
- **The battery end-of-life.** The measurements are at nominal voltage; a 2.4 V cutoff can change the clock and the latency.
- **The regulatory decision.** The report supports a decision; it does not make it.


## 7. Code, query, or calculation

The script computes percentiles, a bootstrap interval for p99, a non-inferiority sample size, and energy from a power trace. It reproduces Section 6.

```python
import math
import random


def percentile(sorted_values, p):
    """Linear interpolation percentile, p in [0, 1]."""
    if not sorted_values:
        raise ValueError("empty sample")
    k = (len(sorted_values) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_values[int(k)]
    return (sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower]) * (k - lower))


def bootstrap_p99(values, resamples=400, seed=9):
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = sorted(rng.choices(values, k=len(values)))
        estimates.append(percentile(sample, 0.99))
    estimates.sort()
    return estimates[int(0.025 * resamples)], estimates[int(0.975 * resamples)]


def sample_size(p1, p2, margin, alpha=0.025, power=0.8):
    z_alpha = 1.96
    z_beta = 0.8416
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (margin - abs(p1 - p2)) ** 2
    return math.ceil(numerator / denominator)


def energy_mj(power_samples_mw, sample_interval_s):
    return sum(power_samples_mw) * sample_interval_s
```

Latency:

```python
random.seed(3)
latencies = sorted(round(random.lognormvariate(math.log(42), 0.06), 2)
                   for _ in range(10_000))
for p in (0.50, 0.95, 0.99, 0.999):
    print("p%.1f = %.2f ms" % (p * 100, percentile(latencies, p)))
print("max = %.2f ms" % latencies[-1])
low, high = bootstrap_p99(latencies)
print("p99 95%% interval = [%.2f, %.2f]" % (low, high))
```

Expected output:

```text
p50.0 = 41.97 ms
p95.0 = 46.30 ms
p99.0 = 48.44 ms
p99.9 = 51.19 ms
max = 53.71 ms
p99 95% interval = [48.17, 48.69]
```

Sample size:

```python
print("n margin 1.5pp =", sample_size(0.942, 0.938, 0.015))
print("n margin 1.0pp =", sample_size(0.942, 0.938, 0.010))
print("n margin 1.5pp, no loss =", sample_size(0.942, 0.942, 0.015))
```

Expected output:

```text
n margin 1.5pp = 7317
n margin 1.0pp = 24592
n margin 1.5pp, no loss = 3812
```

Energy:

```python
random.seed(2)
power = [24.6 + random.gauss(0, 0.8) for _ in range(46)]
print("mean power = %.2f mW" % (sum(power) / len(power)))
print("energy = %.3f mJ" % energy_mj(power, 0.001))
print("with preprocessing = %.3f mJ" % (energy_mj(power, 0.001) + 0.675))
```

Expected output:

```text
mean power = 24.34 mW
energy = 1.120 mJ
with preprocessing = 1.795 mJ
```

Quantization diagnostics:

```python
def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def symmetric_scale(values, bits=8):
    return max(abs(v) for v in values) / (2 ** (bits - 1) - 1)


def quantize_symmetric(values, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    return [min(qmax, max(qmin, round_half_away(v / scale))) for v in values]


def dequantize(codes, scale):
    return [q * scale for q in codes]


def sqnr(original, reconstructed):
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    return 10 * math.log10(signal / noise) if noise else float("inf")


random.seed(11)
weights = [[round(random.gauss(0, 0.09), 4) for _ in range(64)]
           for _ in range(512)]
weights[7] = [round(v * 6.0, 4) for v in weights[7]]
weights[7][3] = 0.62
flat = [v for row in weights for v in row]
scale = symmetric_scale(flat, 8)
codes = quantize_symmetric(flat, scale, 8)
median = sorted(abs(v) for v in flat)[len(flat) // 2]
print("scale %.8f step/median %.4f" % (scale, scale / median))
print("zeros", sum(1 for q in codes if q == 0), "levels", len(set(codes)))
print("sqnr %.2f" % sqnr(flat, dequantize(codes, scale)))
```

Expected output:

```text
scale 0.01016693 step/median 0.1667
zeros 1411 levels 102
sqnr 30.00
```

Assumptions: the latency sample is synthetic lognormal; the bootstrap uses 400 resamples; the sample-size formula is unpaired and normal-approximate; the power trace is 46 samples at 1 kHz; and the preprocessing energy is 0.675 mJ.

## 8. Data, standards, and regulatory context

Validation on target hardware is where the standards become concrete. The claim, the conditions, and the evidence are all regulated artefacts.

### 8.1 IEC 62304 and verification

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79, requires verification that the software requirements are met. The validation report is the verification evidence for the performance requirements.

The standard implies:

- The requirement must be written before the test.
- The test must be under configuration management.
- The result must be traceable to the requirement.
- The runtime, model, and board versions must be recorded.

A latency report without a requirement is a measurement, not verification.

### 8.2 IEC 60601-1 and environmental conditions

IEC 60601-1, the general standard for basic safety and essential performance of medical electrical equipment, requires that essential performance be maintained under the environmental and electrical conditions the standard specifies. If the latency is part of essential performance, the validation must include the relevant conditions: supply variation, electromagnetic disturbance, and the specified ambient range.

A validation at 22 °C on a bench supply does not cover a device intended for 5–40 °C on a battery.

### 8.3 ISO 14971 and the risk of a missed deadline

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and controlled. A latency overrun is a hazard when it delays an alarm. The validation report is evidence that the control (a faster model, a fallback, a watchdog) works.

The risk file should link the hazard to the validation result and to the monitoring that would detect a drift.

### 8.4 GMLP and clinically relevant conditions

The FDA, Health Canada, and the United Kingdom Medicines and Healthcare products Regulatory Agency published ten guiding principles for Good Machine Learning Practice in October 2021. Principle 8 is directly relevant: testing should demonstrate device performance during clinically relevant conditions.

A test on a curated set at room temperature is not a clinically relevant condition. A test with real sensor data, motion artifact, and the specified ambient is closer. The validation report should state which conditions were covered and which were not.

### 8.5 FDA PCCP and the acceptance criteria

The FDA's final guidance on Predetermined Change Control Plans was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022. The modification protocol in a PCCP specifies the performance evaluation, including the metrics, margins, and conditions.

The validation report is the evidence that the protocol's criteria were met. A report that uses a different margin or a different condition from the protocol does not satisfy the plan.

### 8.6 EU framework

In the European Union, Regulation (EU) 2017/745 governs the device, and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk systems. The Annex I applicability date was originally 2 August 2027; Regulation (EU) 2026/1744 (the Digital Omnibus on AI), in force 27 July 2026, moves it to 2 August 2028. Verify the current date before relying on it.

### 8.7 What is binding and what is convention

| Item | Status | Consequence |
|---|---|---|
| IEC 62304 | Binding where required | Report is verification evidence |
| IEC 60601-1 | Binding where required | Environmental conditions apply |
| ISO 14971 | Binding where required | Latency overrun is a hazard |
| GMLP principle 8 | Non-binding guidance | Clinically relevant conditions |
| FDA PCCP guidance | Non-binding; statute binding | Protocol criteria must be met |
| Percentile method | Convention | Must be stated |
| Bootstrap interval | Convention | Must be stated |
| Sample-size formula | Convention | Must be stated |

### 8.8 The validation report template

| Section | Content |
|---|---|
| Claim | Metric, value, percentile, conditions |
| Configuration | Board, firmware, runtime, model hash |
| Rig | Injection, markers, power monitor, chamber |
| Protocol | Warm-up, clock, sample size, ambient |
| Latency | p50, p95, p99, max, interval |
| Energy | J per inference, inclusion set, uncertainty |
| Thermal | Time to throttle, steady state, recovery |
| Accuracy | Sensitivity, specificity, ECE, subgroups |
| Statistics | Margin, power, sample size, result |
| Decision | Pass, fail, or inconclusive, with residual uncertainty |

### 8.9 What a reviewer will ask

- Show the claim and the conditions, in writing, before the results.
- Show the rig and the instrumentation, including the marker overhead.
- Show the warm-up and clock protocol.
- Show the sample size and the interval.
- Show the thermal soak and the time to throttle.
- Show the energy inclusion set and the monitor rate.
- Show the device's accuracy, not only the workstation's.
- Show the power calculation for the non-inferiority claim.
- Show the emulator gap analysis.
- Show the raw data and the analysis script.

### 8.10 What the standards do not say

- No standard specifies a percentile, a sample size, or a bootstrap method.
- No standard requires a thermal soak, although essential performance implies the conditions.
- No standard defines the energy inclusion set.
- No standard says how to extrapolate from a validation sample to a population.
- No standard requires an emulator gap analysis.

The absence means the report must justify its protocol, not cite one.

## 9. Industry practice and case evidence

### 9.1 What vendors report and what they do not

Vendor benchmark reports, including MLPerf, specify the scenario, the quality target, and the system. They usually do not report the thermal soak, the enclosure, or the battery state. The result is a number that is valid for the vendor's conditions and not transferable to a product.

MLPerf Tiny is closer to the device case because it reports energy per inference on microcontroller-class hardware. The v1.2 results, published in April 2024, include 91 performance results and 18 energy results. The energy figures (for example, 1,375.6 µJ per keyword-spotting inference on a Cortex-M7 board at 280 MHz) establish the order of magnitude, not the value for a clinical model.

### 9.2 Independent thermal evidence

Independent studies of mobile sustained performance are consistently less optimistic than peak specifications. One study of a Pixel 2 reported throttling beginning after 151 seconds, with the clock reduced from 2.46 GHz to 1.19 GHz at a 48 °C skin temperature, and frame-rate jank increasing by up to 146%. Another found run-to-run variance of up to 56% in a popular benchmark's sub-scores, and a score increase of more than 60% when the device was chilled to 4 °C.

These are measurements of phones, not medical devices. The inference, labelled as such, is that any fanless device with a skin-temperature limit will throttle, and the only question is when.

### 9.3 What the evidence supports

| Claim | Status |
|---|---|
| p99 requires a large sample for a stable estimate | Statistical fact |
| Thermal throttling occurs in fanless devices | Independent measurements |
| Cold start is much slower than warm inference | Consequence of model load and cache fill |
| Energy per inference depends on the inclusion set | Definitional |
| Emulators do not model thermal or power | Tool limitation |
| A 10,000-sample latency test supports p99 | Depends on the interval width; not universal |
| A 200-case non-inferiority test supports a 1.5 pp margin | Not supported; underpowered |

### 9.4 Why medical validation differs

**The claim is clinical.** A benchmark reports a score; a device reports a sensitivity in a population. The statistical design must match the claim.

**The conditions are specified.** A device has an intended environment, a battery, and an enclosure. The validation must cover them.

**The evidence is audited.** The report must be reproducible, with versions and raw data.

**The failure is a hazard.** A missed deadline is not a slow model; it is a delayed alarm, which is a risk-file item.

### 9.5 What a good validation report looks like

A report that a reviewer can act on has five properties.

- **It states the claim and the conditions** before the results, so the result can be judged against a target.
- **It separates measurement from inference**, so the reader knows which statements are data and which are interpretation.
- **It reports intervals**, not only point estimates, so the uncertainty is visible.
- **It names the gaps**, so the reader knows what was not tested.
- **It reaches a decision**, including "inconclusive", rather than leaving the reader to infer one.

A report that presents a table of numbers without a claim is a data dump. A report that presents a conclusion without conditions is an assertion.

### 9.6 The difference between a benchmark and a validation

A benchmark asks "what can this system do under benchmark conditions". A validation asks "does this device meet its claim under its conditions". The first is comparable across systems; the second is specific to a product.

A team that cites a benchmark as a validation has confused the two. A team that runs a validation and reports it with benchmark-style precision has also confused them, because the validation's uncertainty is part of its meaning.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Cold-burst validation | 30-second test | Thermal throttling | Field latency worse | Soak at ambient |
| Too few samples for p99 | 100 samples | Tail under-sampled | Unstable estimate | 10,000 samples |
| Warm-up omitted | First inference included | Model load, cache fill | Inflated latency | Fixed warm-up |
| Clock not recorded | No clock trace | DVFS | Uninterpretable result | Pin or log clock |
| Energy inclusion unstated | Number without a set | Definition ambiguity | Wrong verdict | State the set |
| Power monitor too slow | 100 Hz sampling | Missed peaks | Underestimated energy | 1–10 kHz |
| Emulator treated as evidence | Desktop result | No thermal or power model | Field miss | HIL test |
| Accuracy validated on workstation only | Device outputs untested | Export or preprocessing change | Wrong field accuracy | Run the device |
| Underpowered non-inferiority | Wide interval | Small sample | Inconclusive claim | Power calculation |
| Margin chosen after results | Margin equals observed | Rationalization | Invalid claim | Pre-specify |
| Subgroup omitted | Overall only | Aggregate masks harm | Equity harm | Pre-specify subgroups |
| Ambient not specified | "Room temperature" | Uncontrolled condition | Unreproducible | Specify 22/40 °C |
| Battery voltage not controlled | Bench supply | End-of-life behaviour | Latency miss | Battery simulator |
| Warm-up count arbitrary | No stabilization check | Cold samples included | Inflated p99 | Check stabilization |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The rig injects real sensor data | Representative latency and accuracy | Synthetic inputs | Compare with field data |
| The clock is pinned or recorded | Interpretable latency | DVFS | Log at 1 Hz |
| The sample is large enough | Stable p99 | Small sample | Bootstrap interval |
| The ambient is controlled | Reproducible thermal | Chamber failure | Log ambient |
| The power monitor is fast enough | Accurate energy | Low sampling rate | Compare rates |
| The inclusion set is stated | Comparable energy | Ambiguity | Document |
| The device matches production | Transfer of results | Prototype board | Test production units |
| The held-out set represents deployment | Valid accuracy | Site shift | Stratified audit |
| The statistical design matches the claim | Valid inference | Wrong test | Pre-specify |
| The margin is clinically justified | Meaningful claim | Arbitrary margin | Clinical review |
| The soak matches the duty cycle | Realistic thermal | Continuous vs intermittent | Match the use |
| The emulator gap is bounded | Early feasibility | No measurement | HIL comparison |
| The battery voltage is controlled | Predicts end-of-life | Bench supply | Battery simulator |
| Warm-up is verified | Stable distribution | Arbitrary count | Block comparison |

The central limit is that validation is a sample. The device will see inputs, temperatures, and battery states that the validation did not. The report should state the coverage and the extrapolation, not only the result.

A second limit is that the statistical claim and the clinical claim are different. A non-inferiority test can show that a difference is small; it cannot show that the device is safe for a population that was not in the sample.

## 12. Exercises

**Exercise 1 — Rig design (design, 40 minutes).**
Design a HIL rig for a wearable ECG device, including data injection, latency markers, power measurement, and temperature control.
*Deliverable:* a diagram and a component table.
*Acceptance criteria:* real sensor data path; marker overhead addressed; power monitor rate; chamber specified.
*Time:* 40 minutes.

**Exercise 2 — Percentiles (calculation, 35 minutes).**
Given 20 sorted latencies, compute p50, p95, and p99 by the interpolation method, and state whether p99 is meaningful with 20 samples.
*Deliverable:* a calculation and a judgement.
*Acceptance criteria:* interpolation correct; p99 identified as the largest values; sample-size limitation stated.
*Time:* 35 minutes.

**Exercise 3 — Energy (calculation, 30 minutes).**
A power trace has 50 samples at 1 kHz with a mean of 30 mW. Compute the energy and the energy with preprocessing (0.5 mJ). State the inclusion set.
*Deliverable:* a calculation.
*Acceptance criteria:* 1.5 mJ; 2.0 mJ with preprocessing; inclusion set stated.
*Time:* 30 minutes.

**Exercise 4 — Thermal soak (analysis, 35 minutes).**
A device throttles at 8 minutes and its p99 rises from 50 to 64 ms. Compute the throttle factor and decide whether a 55 ms claim holds.
*Deliverable:* a calculation and a verdict.
*Acceptance criteria:* factor 1.28; claim fails; options stated.
*Time:* 35 minutes.

**Exercise 5 — Sample size (calculation, 40 minutes).**
Compute the sample size for a non-inferiority margin of 1.2 pp, expected difference 0.3 pp, p1 = 0.94, one-sided α = 0.025, power 0.8.
*Deliverable:* a calculation.
*Acceptance criteria:* formula stated; n computed; units cases per group.
*Time:* 40 minutes.

**Exercise 6 — Emulator gap (critique, 30 minutes).**
List five things an emulator cannot tell you and propose a measurement for each.
*Deliverable:* a five-row table.
*Acceptance criteria:* each row has a measurement; thermal, power, and kernel selection included.
*Time:* 30 minutes.

**Exercise 7 — Report (design, 50 minutes).**
Write the validation report for the Section 6 device using the template in Section 8.
*Deliverable:* a two-page report.
*Acceptance criteria:* claim and conditions; configuration; results with intervals; decision and residual uncertainty.
*Time:* 50 minutes.

**Exercise 8 — Decision (design, 30 minutes).**
Given a device that passes at 22 °C and fails at 40 °C, decide whether to ship. State the conditions, the residual risk, and the monitoring.
*Deliverable:* a one-page decision.
*Acceptance criteria:* decision tied to the intended environment; residual risk stated; monitoring proposed.
*Time:* 30 minutes.

## 13. Self-check answers

**A1. Why p99 needs a large sample.**
Because it is a tail quantile. With n samples, only about n/100 observations are above p99, so the estimate depends on a few values. A 95% interval for p99 with 1,000 samples is roughly three times wider than with 10,000. With 100 samples, the p99 is effectively the second-largest value and is not a stable estimate.

**A2. Energy for Exercise 3.**
50 samples at 1 kHz is 50 ms. E = 30 mW × 50 ms = 1.5 mJ. With preprocessing, 2.0 mJ. The inclusion set is inference only, then inference plus preprocessing.

**A3. Throttle factor for Exercise 4.**
64 / 50 = 1.28. The p99 of 64 ms exceeds the 55 ms claim, so the claim fails at the throttled state. Options: reduce sustained power, improve the thermal path, or restrict the ambient claim.

**A4. Sample size for Exercise 5.**
z = 2.8016, z² = 7.849. p1(1-p1) = 0.94 × 0.06 = 0.0564; p2(1-p2) = 0.94 × 0.06 (approximately) = 0.0564. Numerator = 7.849 × 0.1128 = 0.8854. Denominator = (0.012 - 0.003)² = 0.000081. n = 0.8854 / 0.000081 = 10,931 per group.

**A5. Why cold start is not warm latency.**
Cold start includes model load, memory allocation, cache fill, clock ramp, and runtime initialization. It can be 10–30 times the warm inference. A device that must respond after power-on is bound by cold start, not by the steady-state number.

**A6. Why the energy inclusion set matters.**
Because a number that includes the radio is not comparable with one that excludes it. In the case, inference-only energy is 1.112 mJ and inference plus preprocessing is 1.787 mJ; the claim's verdict changes with the definition. The inclusion set is part of the measurement, not a footnote.

**A7. What an emulator cannot model.**
Clock frequency and DVFS, cache and bus contention, thermal throttling, kernel selection, power draw, sensor noise, and enclosure temperature. It can model operator semantics and approximate cycle counts.

**A8. Why a soak is required.**
Because thermal throttling appears after minutes, and a short test never reaches steady state. The soak is the only way to measure the field latency.

**A9. Rubric for Exercise 1 (rig design).**
A strong design injects real sensor data through the analog front end or an equivalent interface, marks latency with a GPIO and measures the marker overhead, uses a 1–10 kHz power monitor on the battery rail, and controls ambient with a chamber. A weak design uses synthetic inputs and a desktop timer.

**A10. Rubric for Exercise 7 (report).**
A strong report states the claim and conditions, the configuration, the rig, the protocol, the results with intervals, the thermal soak, the device accuracy, the statistical design, and the decision with residual uncertainty. A weak report gives a latency number and an accuracy number.

**A11. Why is a benchmark not a validation?**
A benchmark measures a system under benchmark conditions and is comparable across systems. A validation measures a device against its own claim, under its own conditions, with its own population. A benchmark can inform a design; it cannot support a claim.

**A12. What makes a validation reproducible?**
The configuration record, the rig description, the protocol, the raw samples, the analysis script, and the versions of the runtime and model. A colleague with these can repeat the result; without them, the result is an anecdote.

The configuration record, the rig description, the protocol, the raw samples, the analysis script, and the versions of the runtime and model. A colleague with these can repeat the result; without them, the result is an anecdote.

**A13. Why is a screening test not evidence?**
Because it usually closes only one gap (for example, functional correctness) and lacks the sample size, the thermal coverage, or the statistical design of a validation. It is useful for a design decision, but it cannot support a claim.

## 14. Interview questions

**Q1. Why is p99 more important than p50?**
A strong answer connects the tail to the deadline and to the fraction of patients exposed, and notes that p99 needs a large sample. A weak answer says p99 is conservative.

**Q2. How do you measure energy per inference?**
A strong answer describes the shunt, the sampling rate, the integration window, the inclusion set, and the uncertainty. A weak answer multiplies power by time from a datasheet.

**Q3. What is thermal throttling and how do you test it?**
A strong answer describes the clock reduction, the soak at ambient, the time to throttle, and the steady state. A weak answer says the device gets hot.

**Q4. What can an emulator not tell you?**
A strong answer lists clock, cache, thermal, power, kernel selection, and sensor effects. A weak answer says it is less accurate.

**Q5. How large a sample do you need for a p99 claim?**
A strong answer says it depends on the interval width and the variability, and gives an order of magnitude (thousands to tens of thousands). A weak answer gives a fixed number.

**Q6. Why is cold start measured separately?**
A strong answer lists model load, cache fill, clock ramp, and runtime init, and connects it to the availability requirement. A weak answer says it is slower.

**Q7. How do you size a non-inferiority study?**
A strong answer states the margin, the expected difference, the variance, alpha, and power, and computes n. A weak answer says "enough cases".

**Q8. What conditions must a latency claim state?**
A strong answer lists hardware, runtime, batch, percentile, warm or cold, ambient, and clock. A weak answer says the model and the device.

**Q9. How do you validate accuracy on the device?**
A strong answer runs the held-out set through the device's real input path and compares decisions and calibration with the reference. A weak answer runs the workstation model.

**Q10. When is a result inconclusive?**
A strong answer says when the confidence interval crosses the claim or the margin, and prescribes more data or a different design. A weak answer treats inconclusive as a pass.

**Q11. Why does the ambient temperature matter?**
A strong answer explains the clock reduction and the latency increase, and gives the soak as the test. A weak answer says the device is specified for indoor use.

**Q12. What would you reject in a validation report?**
A strong answer rejects a cold-only test, an unstated inclusion set, a p99 from too few samples, an unrecorded clock, a missing thermal soak, or a margin chosen after the results. A weak answer rejects only a failed accuracy.

**Q13. What does a thermal soak add to a latency test?**
A strong answer says it reveals throttling, the time to throttle, and the steady-state latency, none of which appear in a short test. A weak answer says it makes the device hot.

**Q14. How would you report an inconclusive result?**
A strong answer states the interval that crossed the claim, the reason (sample size, variance, or design), and the additional evidence required. A weak answer reports the point estimate and omits the uncertainty.

## 15. Key terms

- **Ambient** — surrounding temperature.
- **Bootstrap interval** — resampling-based confidence interval.
- **Case temperature** — temperature of the enclosure.
- **Cold start** — first inference after boot.
- **Duty cycle** — fraction of time active.
- **DVFS** — dynamic voltage and frequency scaling.
- **Emulator** — software model of the device.
- **Energy per inference** — joules for one inference.
- **GPIO marker** — pin toggled at an event.
- **HIL** — hardware-in-the-loop.
- **Inclusion set** — what the energy measurement includes.
- **Instrumentation overhead** — effect of measurement on the system.
- **Margin** — largest acceptable difference.
- **Non-inferiority** — not worse by more than a margin.
- **p50** — median latency.
- **p95** — 95th percentile.
- **p99** — 99th percentile.
- **Paired design** — same inputs for both models.
- **Percentile method** — interpolation or nearest-rank rule.
- **Power** — probability of detecting a true difference.
- **Power monitor** — instrument measuring voltage and current.
- **Reproducibility** — ability to repeat the result.
- **Rig** — test fixture and instrumentation.
- **Sample size** — number of observations.
- **Sampling rate** — samples per second.
- **Shunt resistor** — current-sense resistor.
- **Skin temperature** — temperature at the patient contact.
- **Steady state** — stable temperature and latency.
- **Thermal soak** — sustained load until stabilization.
- **Throttling** — clock reduction under thermal load.
- **Time to throttle** — interval before clock reduction.
- **Trace** — recorded timing of events.
- **Warm-up** — inferences before measurement.
- **Boost clock** — temporary higher clock after idle.
- **Paired analysis** — comparison using matched inputs.
- **Residual uncertainty** — uncertainty remaining after the decision.
- **Sampling error** — error from observing a sample rather than a population.
- **Type II error** — failing to detect a true difference.
- **Confidence level** — the long-run coverage of an interval.
- **Null hypothesis** — the hypothesis tested by a statistical test.
- **One-sided test** — a test with the rejection region on one side.
- **Effect size** — the magnitude of a difference.
- **Power analysis** — the calculation of sample size or detectable effect.
- **Raw data** — the unprocessed observations behind a summary.

## 16. Sources

All URLs accessed 25 October 2026.

**Benchmarks**
- MLCommons, "MLPerf Tiny v1.2 Results", April 2024. https://mlcommons.org/2024/04/mlperf-tiny-v1-2-results/
- MLCommons, `tiny_results_v1.2` summary. https://github.com/mlcommons/tiny_results_v1.2/blob/main/summary.csv
- MLCommons, "MLPerf Inference: Edge". https://mlcommons.org/benchmarks/inference-edge/

**Thermal measurement**
- Bantock, D., et al., "Mitigating Interactive Performance Degradation from Mobile Device Thermal Throttling". https://eprints.soton.ac.uk/439709/1/tus_accepted_paper.pdf
- Guo, Y., et al., "Freeze It If You Can: Challenges and Future Directions in Mobile Benchmarking", HotMobile 2017. https://yaoguopku.github.io/papers/Guo-HotMobile-17.pdf

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- IEC 60601-1. https://www.iec.ch/
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, Health Canada, MHRA, "Good Machine Learning Practice", October 2021. https://www.fda.gov/media/153486/download
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

**Runtime and hardware**
- Arm, "CMSIS-NN". https://arm-software.github.io/CMSIS-NN/latest/
- Google AI Edge, "LiteRT for Microcontrollers overview". https://developers.google.com/edge/litert/microcontrollers/overview

> **Note:** The Section 6 latency and power samples are synthetic. MLPerf Tiny figures are vendor-submitted. Thermal studies are independent academic measurements on phones, not medical devices.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`09_medical-and-regulatory-constraints.md`](09_medical-and-regulatory-constraints.md)
- Next topic: [`11_economics-of-compression.md`](11_economics-of-compression.md)
- Related topic: [`02_target-hardware-and-performance.md`](02_target-hardware-and-performance.md)
- Related topic: [`08_on-device-architecture.md`](08_on-device-architecture.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
