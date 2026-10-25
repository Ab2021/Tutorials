# Deployment on Device: Latency, Power, Memory, Quantization, and Updates

> Module 11 — Physiological Signals and Wearables
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (signal fundamentals), Topic 06 (feature engineering), Topic 07 (deep learning)
> Version 2026.3

## 1. Why this topic exists

A model that runs in a notebook is not a product. A product runs on a device with a fixed battery, a fixed memory, a fixed clock, and a user who expects an answer in less than a second. The constraints are not implementation details; they change the model, the features, and sometimes the claim.

Four failures recur.

**The model that did not fit.** A team trains a 12-layer CNN with 1.2 million parameters and an AUROC of 0.94. The target device has 256 kB of flash and 64 kB of RAM. The model does not fit, and the quantized version loses 4 AUROC points. The team had validated a model that could not be shipped.

**The battery that did not last.** A device runs inference every second on a 100 mAh battery. The model takes 40 ms at 25 mW, which seems small, but the sensor and the radio dominate. The device lasts 18 hours instead of 7 days, and the user stops wearing it. The power budget was never computed.

**The latency that broke the decision.** An arrhythmia alert has a 2 s latency budget. The model runs in the cloud, the radio round trip is 3 s, and the alert arrives after the event. The architecture did not match the clinical requirement.

**The update that changed the model.** A firmware update retrains the model on new data. The accuracy improves on the new data and degrades on an older device. No one revalidated the change, and the product is now a different device from the one that was cleared.

This topic covers the deployment chain: latency, power, memory, quantization, pruning, benchmarking, thermal limits, privacy, update strategy, monitoring, and on-device validation. It builds on the model and validation topics and closes the module.

### 1.1 Three composite failures

**The model that did not fit.** A team trains a 1.2 million parameter CNN and validates it on a laptop. The target device has 256 kB of flash and 64 kB of RAM. The model is quantized, pruned, and compressed, and the final version has 4 AUROC points less than the validated model. The product ships with a model that was never validated, because the validated model was not deployable.

**The battery that lasted a day.** A device runs a 40 ms inference every second and a sensor continuously. The team measures the inference energy, finds it small, and ships. The sensor draws 5.5 mW continuously, the radio draws 6.9 mW for 10% of the time, and the battery lasts 2.3 days instead of the promised 7. The model was never the problem; the system was.

**The update that changed the claim.** A firmware update retrains the model on new data and improves the accuracy on the new device. The old device sees a different distribution, and the accuracy falls. The change was not validated on the old device, and the product is now two products with one label.

The three failures share a structure: the deployment budget and the change control are treated as engineering details rather than as parts of the claim. The model is validated in one environment and shipped in another.

### 1.2 Why deployment belongs in the data science curriculum

A data scientist who ignores deployment will build models that cannot ship. The constraints are predictable, and they can be designed into the model from the start:

- the input length determines the activation memory,
- the channel count determines the first-layer compute,
- the architecture determines the operator support,
- the quantization determines the accuracy and the memory,
- the update strategy determines the regulatory path.

A model that is designed for the device is not a compromise; it is a model that can be validated, shipped, and monitored.

A fourth failure is a deployment that cannot be diagnosed. The device misses a deadline, the log has no timing data, and the team cannot tell whether the cause is the model, the clock, or the radio. The observability is part of the design, not an afterthought.

## 2. Learning outcomes

By the end of this topic you can:

1. **Specify** the latency, power, memory, and privacy requirements for a device deployment.
2. **Compute** the parameter memory, activation memory, and compute budget of a 1D CNN.
3. **Quantize** a model to int8 and estimate the accuracy and memory trade-off.
4. **Benchmark** latency and energy on the target hardware, not on a laptop.
5. **Design** a power budget that includes sensing, compute, radio, and sleep.
6. **Choose** between edge, cloud, and hybrid processing for a stated decision.
7. **Plan** a model update with versioning, validation, and a predetermined change control plan.
8. **Monitor** a deployed model for drift, calibration, and device mix.

Each outcome maps to an exercise in Section 12 and a self-check item in Section 13.

## 3. The core mental model

A deployed signal pipeline is a chain of budgets. Each stage consumes time, energy, and memory, and the budgets are shared.

```text
 sensor -> analog front end -> ADC -> DSP -> model -> decision -> radio/display
   |            |              |       |        |         |            |
 power        power          power   latency  latency   latency      power
   |            |              |       |        |         |            |
 memory       memory         buffer  buffer   weights   state        buffer
```

Redraw it with the four budgets on every arrow: milliseconds, millijoules, kilobytes, and bytes per second. A design that exceeds any budget fails, regardless of the model's accuracy.

### 3.1 The four budgets

- **Latency**: the time from the end of the acquisition window to the decision. A 2 s window plus a 40 ms inference plus a 100 ms radio is 2.14 s, not 40 ms.
- **Energy**: the sum of sensing, compute, radio, and sleep. The compute is often a small fraction; the sensor and the radio dominate.
- **Memory**: the weights, the activations, the buffers, and the firmware. Flash and RAM are separate budgets.
- **Bandwidth**: the bytes per second sent to the cloud. A raw 500 Hz, 16-bit ECG is 1,000 bytes/s per lead; a feature vector is tens of bytes.

### 3.2 Edge, cloud, and hybrid

| Architecture | Latency | Power | Privacy | Model size | Update |
|---|---|---|---|---|---|
| Edge | Low | Device | High | Small | OTA |
| Cloud | Network | Radio | Transfer | Large | Server |
| Hybrid | Mixed | Both | Partial | Medium | Both |

The choice is driven by the decision. A fall detection needs edge latency. A nightly sleep report can be cloud. A screening alert can be hybrid: edge detection, cloud confirmation.

### 3.3 Quantization

Quantization maps floating-point weights and activations to a smaller integer representation. The common scheme is symmetric int8 with a per-tensor scale.

```text
 q = round(x / scale),  scale = max(|x|) / 127
 x_approx = q * scale
```

The error is bounded by half the scale per value. The memory falls by 4× from float32 to int8. The compute can be faster because integer arithmetic is cheaper on many microcontrollers.

The cost is accuracy. A model with a small dynamic range quantizes well; a model with outliers quantizes poorly. The scale should be chosen from the training distribution, and the accuracy should be measured after quantization, not assumed.

### 3.4 The power budget

The energy per inference is the power times the latency. The daily energy is the energy per inference times the number of inferences. The battery life is the battery energy divided by the daily energy.

A 100 mAh, 3.7 V battery stores 1,332 J. A device that uses 1 mJ per inference at 0.5 inferences/s uses 43.2 J/day, which is 3.2% of the battery per day, or about 31 days of compute. If the sensor and radio use 10 mW continuously, they use 864 J/day, and the battery lasts 1.5 days. The compute is not the problem; the always-on sensor is.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Edge inference | Computation on the device | Confusing with cloud | architecture |
| Latency | Time from input to output | Reporting only model time | ms |
| Throughput | Inferences per second | Confusing with latency | 1/s |
| Energy | Power times time | Reporting power only | mJ |
| Power | Energy per second | Confusing with energy | mW |
| Battery capacity | Stored charge | Confusing with energy | mAh |
| Battery energy | Capacity times voltage | Forgetting voltage | J |
| Duty cycle | Active fraction | Ignoring sleep | % |
| Flash | Non-volatile storage | Confusing with RAM | kB |
| RAM | Volatile working memory | Confusing with flash | kB |
| Arena | Preallocated memory pool | Using malloc | kB |
| Activation | Intermediate tensor | Confusing with weight | kB |
| Quantization | Reducing numerical precision | Assuming no accuracy loss | int8 |
| Scale | Quantization step | Confusing with zero point | real |
| Zero point | Offset for asymmetric quantization | Ignoring | integer |
| Pruning | Removing weights | Assuming speedup | % |
| Distillation | Training a small model from a large one | Confusing with pruning | method |
| Operator | A compute primitive | Assuming all are supported | op |
| Kernel | Implemented operator | Confusing with convolution kernel | code |
| CMSIS-NN | Arm's neural network library | Assuming portability | library |
| TFLM | TensorFlow Lite for Microcontrollers | Assuming full TF support | runtime |
| MAC | Multiply-accumulate | Confusing with FLOP | operation |
| MOPS | Million operations per second | Confusing with MIPS | 1e6/s |
| Thermal limit | Maximum temperature | Ignoring | °C |
| OTA | Over-the-air update | Assuming atomicity | method |
| Rollback | Reverting an update | Ignoring | method |
| Secure boot | Verified firmware | Ignoring | method |
| SBOM | Software bill of materials | Treating as optional | document |
| Privacy | Protection of data | Confusing with security | property |
| Differential privacy | Noise-based privacy | Confusing with encryption | method |
| On-device validation | Testing on the target | Confusing with laptop | method |
| Device mix | Distribution of devices | Ignoring in monitoring | population |
| Watchdog | Hardware timer that resets the device | Ignoring | timer |
| High-water mark | Peak memory use | Using static estimate | kB |
| Ping-pong buffer | Double buffer for streaming | Single buffer | method |
| Operator fusion | Combining operations | Ignoring | method |
| Memory planner | Offline allocation | Using malloc | method |
| Quantization-aware training | Training with quantization | Assuming post-training is enough | method |
| Per-channel scale | Scale per output channel | Per-tensor only | method |
| Mixed precision | Float for sensitive layers | All int8 | method |
| Pruning | Removing weights | Assuming speedup | method |
| Structured pruning | Removing channels | Unstructured sparsity | method |
| Sparsity | Fraction of zero weights | Confusing with pruning | % |
| Effective throughput | Real operations per second | Peak throughput | op/s |
| Clock throttling | Reduced clock under load | Ignoring | Hz |
| Duty cycle | Active fraction | Ignoring sleep | % |
| Sleep current | Current in low-power mode | Ignoring | µA |
| Wake-up energy | Energy to wake | Ignoring | mJ |
| Battery derating | Capacity loss with temperature | Ignoring | % |
| Surface temperature | Device skin temperature | Ignoring | °C |
| Secure element | Hardware key storage | Software key only | hardware |
| Signed image | Authenticated firmware | Unsigned update | method |
| Rollback partition | Previous firmware | No recovery | method |
| Atomic update | All-or-nothing write | Partial update | method |
| Telemetry | Device-reported metrics | Ignoring | data |
| Re-identification | Identifying a person from data | Assuming aggregation is enough | risk |
| WCET | Worst-case execution time | Using the mean | ms |
| Jitter | Variation in latency | Ignoring the tail | ms |
| Interrupt latency | Delay before an ISR runs | Ignoring | µs |
| Cache miss | Memory access penalty | Ignoring in WCET | cycles |
| Wait state | Flash access delay | Ignoring | cycles |
| A/B partition | Two firmware slots | Single slot | method |
| Boot counter | Counts failed boots | No rollback trigger | count |
| Fuel gauge | Battery state estimator | Voltage only | % |

## 5. The mechanics, step by step

The procedure is written for a wearable or embedded device with a microcontroller or a low-power applications processor. It assumes the model and the features are already validated offline.

### Step 1. Specify the deployment requirements

**Input:** the decision, the user, and the setting.

**Output:** a requirements table with latency, energy, memory, bandwidth, privacy, and update constraints.

**Rationale:** the requirements are the design target. A model that meets the accuracy target and misses the latency target is not deployable.

**Contents:** maximum latency, maximum energy per inference or per day, flash and RAM limits, radio bandwidth, data retention, and update frequency.

**Branch point:** if the decision is an alert, the latency is the sum of the window, the inference, and the notification. If the decision is a nightly report, the latency can be hours.

**Edge case:** a requirement that says "real time" without a number is not a requirement. "Within 2 s of the end of the window" is a requirement.

**The requirements table.**

| Requirement | Target | Why |
|---|---|---|
| Latency | < 2 s from window end | Alert utility |
| Energy | < 50 J/day compute | Battery life |
| Flash | < 200 kB | Device |
| RAM | < 48 kB | Device |
| Bandwidth | < 1 kB/s | Radio power |
| Privacy | Raw data on device | Trust and law |
| Update | Signed, atomic, rollback | Safety |

The table is the design contract. A model that misses one row is not deployable, even if it meets the others.

### Step 2. Choose the compute platform

**Input:** the requirements and the available hardware.

**Output:** a microcontroller, an applications processor, or a cloud backend.

**Rationale:** the platform determines the achievable model size and latency. A Cortex-M4 at 64 MHz has a different budget from a Cortex-A53.

**Comparison:**

| Platform | Clock | RAM | Typical use | Model size |
|---|---|---|---|---|
| Cortex-M0+ | 48 MHz | 16-64 kB | Simple features | < 20 kB |
| Cortex-M4F | 64-96 MHz | 64-256 kB | Small CNN | 20-100 kB |
| Cortex-M7 | 200-480 MHz | 256 kB-1 MB | Larger CNN | 100-500 kB |
| Cortex-A | 1+ GHz | GB | Cloud-like | MB |

**Branch point:** if the model is larger than the flash, the platform is wrong or the model must be compressed.

**Edge case:** a platform with a neural accelerator can run a larger model at lower energy, but the toolchain and the operator support may be limited.

**Platform selection table.**

| Constraint | M0+ | M4F | M7 | Applications processor |
|---|---|---|---|---|
| Flash | 64-256 kB | 256 kB-1 MB | 1-2 MB | GB |
| RAM | 8-32 kB | 64-256 kB | 256 kB-1 MB | GB |
| CNN | Tiny | Small | Medium | Large |
| Energy | Lowest | Low | Medium | High |
| Use | Features | Wearable CNN | Hub | Phone/cloud |

The table is a starting point. The choice should be driven by the model and the battery, not by the vendor's benchmark.

### Step 3. Compute the memory budget

**Input:** the model architecture and the input size.

**Output:** the weight memory, the activation memory, the buffer memory, and the total.

**Rationale:** flash holds the weights and the code; RAM holds the activations and the buffers. The two are separate and both can be the limiting factor.

**Arithmetic:** for the 3,362-parameter model, int8 weights are 3.28 kB and float32 weights are 13.13 kB. The largest activation is 3,750 × 8 = 30,000 values; at int8, that is 29.3 kB. A double buffer for the ping-pong scheme is 58.6 kB, plus the weights, for 61.9 kB of arena. A device with 64 kB of RAM is at the limit.

**Branch point:** if the activation is larger than the RAM, reduce the input length, the channel count, or the stride.

**Edge case:** the framework itself uses memory. TFLM's interpreter and kernels can use 10 kB to 30 kB, which must be added to the budget.

**Partitioning the memory.** Flash and RAM are separate budgets, and each has fixed consumers.

| Region | Flash | RAM | Contents |
|---|---:|---:|---|
| Bootloader | 8-32 kB | 0 | Secure boot, rollback |
| Application code | 30-100 kB | 0 | Drivers, radio stack |
| Runtime | 10-30 kB | 2-8 kB | Interpreter, kernels |
| Weights | 3-500 kB | 0 | Quantized parameters |
| Arena | 0 | 20-100 kB | Activations, scratch |
| Stack | 0 | 2-8 kB | Call frames |
| Heap | 0 | 0 in production | Avoided |
| Logging buffer | 0 | 1-4 kB | On-device events |

The table is a budget, not a suggestion. A device with 64 kB of RAM that allocates 61.9 kB to the arena has 2.1 kB for the stack, the runtime, and the logging buffer, which is not enough. The arena must shrink or the device must change.

The flash budget is usually less tight than the RAM budget, but the update image doubles the requirement: an A/B partition scheme needs room for two images, so a 256 kB flash with a 200 kB application has no room for a rollback copy.

**Static versus dynamic allocation.** A production firmware should avoid `malloc` after initialization. A preallocated arena, a fixed stack, and a fixed logging buffer make the memory use deterministic and auditable. Dynamic allocation introduces fragmentation, which can fail after hours even when the high-water mark is below the limit.

### Step 4. Quantize the model

**Input:** the trained float model and a calibration set.

**Output:** an int8 model, a scale per tensor, and an accuracy estimate.

**Rationale:** quantization reduces memory and can reduce latency. The accuracy loss depends on the model and the data.

**Method:** post-training quantization with a representative calibration set; quantization-aware training if the loss is too large.

**Arithmetic:** with weights from a Gaussian distribution with standard deviation 0.1, the maximum absolute weight is about 0.36, the scale is 0.36/127 = 0.00284, and the RMS quantization error is 8.2 × 10⁻⁴. The signal-to-quantization-noise ratio depends on the weight distribution, not only on the bit count.

**Branch point:** if the accuracy loss is unacceptable, use quantization-aware training or keep the sensitive layers in float.

**Edge case:** the first and last layers are often kept in float because they are sensitive to quantization. The memory saving is smaller, but the accuracy is preserved.

**Quantization comparison.**

| Scheme | Bits | Memory | Accuracy | Use |
|---|---:|---:|---|---|
| Float32 | 32 | 4× | Baseline | Training |
| Float16 | 16 | 2× | Near baseline | Some accelerators |
| Int8 | 8 | 1× | Small loss | MCU |
| Int4 | 4 | 0.5× | Larger loss | Research |

The table shows the trade-off. Int8 is the default for microcontrollers because the hardware supports it and the accuracy loss is usually small. Int4 is attractive for memory but requires careful calibration and may not be supported.

**Quantized inference arithmetic.** An int8 convolution does not multiply the quantized values directly. Each input and weight tensor has a scale, and the product is accumulated in int32:

```text
 acc = sum_i (q_x[i] - z_x) * (q_w[i] - z_w)
 y = acc * (s_x * s_w / s_y) + z_y
```

The multiplier `s_x * s_w / s_y` is a floating-point constant that the runtime applies once per output, often as a fixed-point multiply and shift. The int32 accumulator has enough range for the products if the tensor lengths are moderate; a long accumulation can overflow, which is why some runtimes use int16 inputs with int32 accumulation and others restrict the channel count.

The arithmetic matters for two reasons. First, the zero point `z` is not zero for asymmetric tensors, so a naive `sum(q_x * q_w)` is wrong. Second, the requantization step rounds, and the rounding error accumulates across layers. A model with many layers and small activations can lose more accuracy than the weight error suggests.

### Step 5. Optimize the compute

**Input:** the quantized model and the target platform.

**Output:** an optimized model with a measured latency.

**Rationale:** the theoretical MAC count is not the latency. The latency depends on the operator support, the memory access, and the cache.

**Methods:** use the platform's optimized kernels (CMSIS-NN on Arm), fuse operators, avoid transposes, and reduce the input length.

**Arithmetic:** 3.75 million MACs at 100 MOPS is 37.5 ms; at 500 MOPS it is 7.5 ms. The same model on two platforms can differ by 5×.

**Branch point:** if the latency is too high, reduce the MACs or increase the clock. The first is usually better for energy.

**Edge case:** a model with many small operators can be slower than a model with fewer large operators, because the per-operator overhead dominates.

**Latency is not one number.** The end-to-end latency has five components, and the deadline applies to the sum plus the jitter.

| Component | Typical | Variability | Source |
|---|---:|---|---|
| Acquisition window | 2,000 ms | fixed | design |
| Preprocessing | 2-10 ms | low | DSP |
| Inference | 7.5-75 ms | medium | clock, cache |
| Decision logic | < 1 ms | low | firmware |
| Notification | 10-100 ms | high | radio, display |

The worst-case execution time (WCET) is the maximum over the components, not the mean. A 37.5 ms mean inference with a 10 ms cache-miss penalty and a 5 ms interrupt can be 52.5 ms at the 99th percentile. A deadline of 50 ms is missed by the tail, even though the mean meets it.

Determinism matters more than speed for an alarm. A model that runs in 20 ms on average and 80 ms occasionally is worse than a model that runs in 40 ms every time, because the alarm deadline is a maximum, not an average.

**Interrupt latency and blocking.** The inference can be delayed by a higher-priority interrupt, a flash write, or a radio event. The firmware should either mask the long operations during the inference or budget the delay. A flash write can block the CPU for milliseconds on some parts.

**Cache and memory access.** A Cortex-M7 with a cache can run the same model 2× faster than the same core without a cache, and the timing can vary with the data. A WCET analysis should include the cache-miss case.

### Step 6. Measure the energy

**Input:** the optimized model and a power measurement setup.

**Output:** the energy per inference and the daily energy.

**Rationale:** the energy is the integral of power over time. A model that runs fast at high power can use more energy than a slower model at low power.

**Arithmetic:** at 25 mW active and 7.5 ms, the energy is 0.188 mJ. At 37.5 ms, it is 0.938 mJ. The five-times latency difference is a five-times energy difference because the power is constant.

**Branch point:** if the energy is too high, reduce the active time, the active power, or the number of inferences.

**Edge case:** the sleep current is not zero. A device that wakes every second pays a wake-up energy that can exceed the inference energy.

**Energy comparison table.**

| Mode | Power | Duration | Energy |
|---|---:|---:|---:|
| Inference | 25 mW | 37.5 ms | 0.94 mJ |
| Sensor | 5.5 mW | 2 s | 11 mJ |
| Radio | 6.9 mW | 200 ms | 1.38 mJ |
| Sleep | 8.1 µW | 1.8 s | 0.015 mJ |
| Wake-up | 25 mW | 5 ms | 0.125 mJ |

The table shows that the sensor is the largest single contributor in a 2 s cycle. The inference is about 7% of the cycle energy, and the radio is about 10%. A model optimization that halves the inference saves 3.5% of the cycle.

### Step 7. Build the full power budget

**Input:** the sensing, compute, radio, and sleep currents.

**Output:** a daily energy and a battery life estimate.

**Rationale:** the compute is often a small part of the total. The sensor and the radio dominate.

**Arithmetic:** a 100 mAh, 3.7 V battery stores 1,332 J. A sensor at 5.5 mW continuous uses 475.2 J/day. A radio at 6.9 mW for 10% of the day uses 59.6 J/day. A compute at 0.938 mJ per inference at 0.5 inferences/s uses 40.5 J/day. The sleep current adds 0.7 J/day. The total is 576.0 J/day, and the battery lasts 2.31 days.

**Branch point:** if the battery life is too short, duty-cycle the sensor or the radio before optimizing the model.

**Edge case:** the battery capacity derates with temperature and age. A 100 mAh battery at 0 °C delivers less than at 25 °C.

### Step 8. Check the thermal limit

**Input:** the power and the device geometry.

**Output:** a temperature estimate.

**Rationale:** a device that runs continuously at high power heats the skin and the battery. The thermal limit is a safety and comfort constraint.

**Arithmetic:** a 25 mW processor in a 5 g device with a specific heat of 1 J/(g·K) and no cooling heats at 0.005 K/s, or 18 K/hour. The actual heat loss limits the rise, but the estimate shows the order of magnitude.

**Branch point:** if the temperature rise is too high, reduce the duty cycle or the power.

**Edge case:** the skin contact temperature is regulated for safety; the device surface should not exceed the applicable limit.

**Thermal mitigation table.**

| Mitigation | Effect | Cost |
|---|---|---|
| Duty cycle | Less heat | Less data |
| Lower clock | Less heat | More latency |
| Heat spreader | Lower surface temperature | Size and cost |
| Air gap | Lower skin temperature | Comfort |
| Thermal shutdown | Safety | Data loss |

The mitigation is a design choice. A device that runs continuously at high power needs a thermal design, not only a firmware fix.

### Step 9. Design the update strategy

**Input:** the model, the firmware, and the validation.

**Output:** an update plan with versioning, signing, rollback, and revalidation.

**Rationale:** a model update is a device change. The change must be validated and, for a regulated device, authorized.

**Contents:** the version scheme, the signing key, the rollback image, the validation protocol, and the communication to the user.

**Branch point:** if the update changes the model's performance, it is a new device function and needs evidence. A predetermined change control plan can pre-authorize the change.

**Edge case:** a partial update can leave the device in an inconsistent state. The update should be atomic and recoverable.

**Watchdog and fail-safe behaviour.** A deployed model must fail visibly and safely.

| Failure | Detection | Safe action |
|---|---|---|
| Inference overruns | Watchdog | Reset; log; retry |
| Repeated reset | Boot counter | Roll back image |
| Sensor invalid | Quality flag | Suppress output |
| Model output out of range | Range check | Suppress; log |
| Low battery | Fuel gauge | Reduce duty cycle |
| Thermal limit | Temperature sensor | Throttle; warn |

The fail-safe action should be defined for each failure, and the device should not fail silently. A monitor that stops reporting because the model crashed is worse than a monitor that reports "unavailable."

**Update checklist.**

| Item | Question | Evidence |
|---|---|---|
| Version | Is the version recorded | Manifest |
| Signature | Is the image signed | Signature check |
| Atomicity | Can a power loss recover | Fault injection |
| Rollback | Is there a previous image | Partition table |
| Validation | Was the new model tested | Report |
| Communication | Is the user informed | Release note |
| Monitoring | Is the new version tracked | Telemetry |

The checklist is the minimum for a regulated update. A missing row is a risk that the reviewer will find.

### Step 10. Protect privacy

**Input:** the data, the architecture, and the regulations.

**Output:** a data-flow diagram and a privacy design.

**Rationale:** continuous physiological data are sensitive. The design should minimize collection, retention, and transfer.

**Methods:** process on the device, send features rather than raw signal, encrypt at rest and in transit, and delete raw data after the decision.

**Arithmetic:** a raw 500 Hz, 16-bit ECG is 1,000 bytes/s per lead, or 86.4 MB/day per lead. A feature vector at 1 Hz is a few kilobytes per day. The difference is four orders of magnitude.

**Branch point:** if the raw signal is needed for a future model, the retention is a privacy decision, not only a storage decision.

**Edge case:** a feature can be re-identifying. Heart-rate patterns and gait can be biometric. Aggregation is not anonymization.

**Data minimization table.**

| Data | Retention | Transfer | Risk |
|---|---|---|---|
| Raw signal | Minutes to hours | No | High |
| Features | Days to months | Maybe | Medium |
| Decision | Months | Yes | Low |
| Telemetry | Months | Yes | Low |
| Identifiers | Not stored | No | High |

The table is a privacy design, not a legal opinion. The retention and transfer should be justified by the decision, not by the convenience of the pipeline.

A device that sends raw signal for a cloud model has a different privacy posture from a device that sends a decision. The architecture and the privacy design are the same decision.

**On-device logging without PHI.** A device needs logs for debugging and audit, and the logs must not become a repository of protected health information.

| Log field | Keep | Why |
|---|---|---|
| Timestamp | Yes, relative | Debug timing |
| Model version | Yes | Traceability |
| Quality flags | Yes | Failure analysis |
| Feature values | Aggregated | Debug without identity |
| Raw waveform | No, unless consented | Re-identifiable |
| User identifier | Hashed or none | Privacy |
| Location | No | Not needed |

A rolling buffer of the last few minutes of signal can be useful for a crash diagnosis, but it should be encrypted, time-limited, and deleted after upload. A log that stores raw ECG indefinitely is a privacy liability, not an engineering convenience. A privacy review that starts after the architecture is a review of a decision that has already been made.

The same is true of the power budget: a review that starts after the model is trained is a review of a model that may not fit.

### Step 11. Validate on the device

**Input:** the model and the production firmware.

**Output:** an on-device validation report with latency, energy, accuracy, and failure modes.

**Rationale:** a model that is accurate on a laptop can fail on the device because of quantization, operator differences, or timing. The validation must run on the target.

**Contents:** the accuracy on a held-out set, the latency distribution, the energy per inference, the memory high-water mark, and the failure modes (watchdog, thermal, memory).

**Branch point:** if the on-device accuracy differs from the offline accuracy, the difference is a deployment bug, not a model property.

**Edge case:** the device clock may be throttled under thermal load, which increases latency and changes the duty cycle.

### Step 12. Monitor after deployment

**Input:** the deployed model and the telemetry.

**Output:** a monitoring dashboard with drift, calibration, device mix, and failure rates.

**Rationale:** a deployed model is exposed to new devices, new populations, and new firmware. The monitoring is the only way to detect a change before it becomes a safety issue.

**Metrics:** input distribution, score distribution, positive rate, calibration on labelled samples, device mix, firmware mix, and error rates.

**Branch point:** if a metric shifts, the response is an investigation, not an automatic retrain.

**Edge case:** the telemetry itself is data. The monitoring should not collect more than it needs.

**Monitoring metrics.**

| Metric | What it detects | Frequency |
|---|---|---|
| Input mean and SD | Data drift | Daily |
| Score distribution | Model drift | Daily |
| Positive rate | Prevalence shift | Daily |
| Calibration | Probability drift | Weekly |
| Device mix | New hardware | Weekly |
| Firmware mix | New software | Weekly |
| Error and reset rate | Reliability | Continuous |

The table is the minimum dashboard. A shift in the input mean is an early warning; a shift in the calibration is a late warning. The two together distinguish a data problem from a model problem.

### Decision table for architecture

| Requirement | Edge | Cloud | Hybrid |
|---|---|---|---|
| Latency < 100 ms | Yes | No | Yes |
| Model > 10 MB | No | Yes | Partial |
| No data transfer | Yes | No | Partial |
| Frequent updates | OTA | Server | Both |
| Low device cost | Yes | No | Partial |
| High accuracy | Limited | Yes | Yes |

## 6. Worked example

### 6.1 The case

A 1D CNN with 3,362 parameters classifies 30 s single-lead ECG at 250 samples/s. The target is a Cortex-M4F at 64 MHz with 256 kB of flash and 64 kB of RAM, a 100 mAh battery, and a 2 s decision latency requirement. The model is quantized to int8, benchmarked, and budgeted.

### 6.2 Memory arithmetic

| Item | Float32 | Int8 |
|---|---:|---:|
| Weights | 13.13 kB | 3.28 kB |
| Largest activation | 117.2 kB | 29.3 kB |
| Double buffer | 234.4 kB | 58.6 kB |
| Arena (weights + double buffer) | 247.5 kB | 61.9 kB |

The float32 model does not fit in 64 kB of RAM and barely fits in 256 kB of flash if the activations are streamed. The int8 model fits in 64 kB of RAM with about 2 kB to spare, before the framework overhead. The framework can use 10 kB to 30 kB, so the int8 model is at the limit and may not fit with a full runtime.

The arithmetic is:

- Largest activation: 3,750 × 8 = 30,000 values.
- At int8: 30,000 bytes = 29.3 kB.
- Double buffer: 58.6 kB.
- Weights: 3.28 kB.
- Arena: 61.9 kB.

A device with 64 kB of RAM has 2.1 kB left for the stack, the buffers, and the framework. That is not enough. The model needs a smaller activation, a single buffer, or a larger device.

The fit table makes the options explicit.

| Option | Activation | Arena | Fit 64 kB |
|---|---:|---:|---|
| 30 s, 8 channels | 29.3 kB | 61.9 kB | Marginal |
| 30 s, single buffer | 29.3 kB | 32.6 kB | Yes |
| 10 s, double buffer | 9.8 kB | 22.9 kB | Yes |
| 30 s, 16 channels | 58.6 kB | 120.5 kB | No |

The single buffer removes the ping-pong but complicates the streaming. The shorter window reduces the memory and the context. The wider model is not deployable on this device.

### 6.3 Latency arithmetic

At 3.75 million MACs:

| Platform throughput | Latency |
|---:|---:|
| 50 MOPS | 75.0 ms |
| 100 MOPS | 37.5 ms |
| 500 MOPS | 7.5 ms |
| 1,000 MOPS | 3.75 ms |

The 2 s latency requirement is met at any of these throughputs, because the window dominates. The model latency is not the constraint; the window is.

The latency budget is:

- Acquisition window: 2,000 ms.
- Inference: 37.5 ms at 100 MOPS.
- Decision and notification: 50 ms.
- Total: 2,087.5 ms.

The total exceeds 2 s by 87.5 ms. The requirement should be interpreted as "within 2 s of the end of the window," which the design meets, or the window should be shortened.

### 6.4 Energy arithmetic

At 25 mW active:

| Latency | Energy per inference |
|---:|---:|
| 7.5 ms | 0.188 mJ |
| 37.5 ms | 0.938 mJ |
| 75.0 ms | 1.875 mJ |

At 0.5 inferences/s (one every 2 s), the daily energy is:

- 0.188 mJ × 0.5 × 86,400 = 8.1 J/day.
- 0.938 mJ × 0.5 × 86,400 = 40.5 J/day.
- 1.875 mJ × 0.5 × 86,400 = 81.0 J/day.

The battery stores 1,332 J. The compute alone would last 164 days at the lowest energy and 16 days at the highest. The compute is not the limiting factor.

### 6.5 Full power budget

| Component | Power | Duty | Daily energy |
|---|---:|---:|---:|
| PPG/ECG sensor | 5.5 mW | 100% | 475 J |
| MCU active | 25 mW | 1.9% | 40.5 J |
| Radio | 6.9 mW | 10% | 59.6 J |
| Sleep | 8.1 µW | 98% | 0.7 J |
| Total | — | — | 576.0 J |

The battery lasts 1,332/576.0 = 2.31 days. The sensor is 82% of the energy. Reducing the sensor duty cycle from 100% to 50% saves 237.5 J/day and extends the battery to 3.9 days. Optimizing the model from 0.938 mJ to 0.188 mJ saves 32.4 J/day and extends the battery to 2.45 days. The sensor is the lever.

The budget reconciles to the total:

| Component | J/day | Share |
|---|---:|---:|
| Sensor | 475.2 | 82.5% |
| Radio | 59.6 | 10.3% |
| MCU | 40.5 | 7.0% |
| Sleep | 0.7 | 0.1% |
| Total | 576.0 | 100% |

The components sum to the total. A budget that does not reconcile to within a few percent has a missing term or a unit error.

### 6.6.1 The deadline diagnostic

The offline latency is 37.5 ms at 100 MOPS, which meets a 50 ms deadline with 12.5 ms of margin. On the device, the 99th-percentile latency is 64 ms, and the deadline is missed.

The components of the difference are:

| Source | Effect | Latency |
|---|---:|---:|
| Nominal at 64 MHz | — | 37.5 ms |
| Clock throttled to 48 MHz | × 64/48 | 50.0 ms |
| Cache miss on the weights | +10% | 55.0 ms |
| Interrupt during inference | +5 ms | 60.0 ms |
| Flash wait states | +8% | 64.8 ms |

The diagnostic is to measure the 99th percentile on the target, at the throttled clock, with the worst-case input, and with the radio active. A mean latency measured on a laptop or a development board is not the WCET.

The consequence of missing the diagnostic is a missed deadline at the tail. At 43,200 inferences/day, a 1% miss rate is 432 late decisions per day. For an arrhythmia alarm, that is 432 alarms that arrive after the event.

The fix is to design for the WCET: reduce the model, shorten the window, raise the clock, or move the inference to a phone. The mean is not the design target.

### 6.6 Quantization arithmetic

The weights are Gaussian with standard deviation 0.1. The maximum absolute weight is about 0.36, the scale is 0.00284, and the RMS error is 8.2 × 10⁻⁴ per weight. The relative error is about 0.8% of the standard deviation.

The accuracy loss is not determined by the weight error alone. It depends on how the error propagates through the network. A model with a small margin between classes can lose more accuracy than a model with a large margin. The post-quantization accuracy must be measured, not inferred from the weight error.

### 6.7 Diagnostic 1: the memory high-water mark

The static budget says 61.9 kB, which fits in 64 kB. The runtime high-water mark is 68 kB, which does not. The difference is the framework, the stack, and the temporary buffers.

The diagnostic is to measure the high-water mark on the device, not to trust the static estimate. A device that crashes at 68 kB is a device that does not fit, even if the arithmetic says it does.

The fix is to reduce the activation (shorter window, fewer channels), use a single buffer, or choose a device with more RAM.

### 6.8 Diagnostic 2: the duty-cycle error

The power budget assumes a 1.9% MCU duty cycle, which is 37.5 ms every 2 s. The actual duty cycle is 4.8% because the MCU also runs the sensor driver and the radio stack. The daily energy rises from 40.5 J to 102 J, and the battery life falls from 2.31 days to 2.1 days.

The diagnostic is to measure the actual active time, not the inference time. The inference is a fraction of the MCU work.

### 6.9 Varying one assumption

Suppose the window is 10 s instead of 30 s. The activation falls by a factor of 3, the memory falls to about 20 kB, and the model fits comfortably. The latency falls, but the decision quality may fall because the model sees less context.

Suppose the device is a Cortex-M0+ at 48 MHz with 16 kB of RAM. The int8 model does not fit. The model must be reduced to a feature-based classifier, or the device must be upgraded.

Suppose the battery is 50 mAh instead of 100 mAh. The compute life halves, but the sensor still dominates. The battery life falls to about 1.2 days, and the product is not viable without a sensor duty cycle.

### 6.10 Conclusion of the case

The int8 model fits the flash and is at the limit of the RAM. The latency is dominated by the window, not the inference. The energy is dominated by the sensor, not the model. The two diagnostics (memory high-water mark, duty-cycle error) show where the static budget misleads.

The assumption that matters most is the sensor duty cycle, because it is 82% of the energy. The second is the RAM high-water mark, because it determines whether the device runs at all.

### 6.10 What the case does not show

The case is a static budget from a single model. It does not show:

- **The measured high-water mark.** The static estimate is 61.9 kB; the runtime may be higher.
- **The measured on-device accuracy.** The quantization error is estimated from the weights, not measured on the test set.
- **The thermal behaviour.** The junction temperature under continuous load is not modelled.
- **The radio protocol.** The connection interval and the payload size are not specified.
- **The update path.** The image size and the rollback partition are not budgeted.
- **The device mix.** Only one device and one firmware version are considered.

Each omission is a measurement that a real deployment must make. The arithmetic transfers; the fit does not.

A third diagnostic is the clock. The case assumes 64 MHz, but the device may throttle to 32 MHz under thermal load or low battery. The latency doubles, the duty cycle rises, and the energy per inference rises. The budget should be computed at the throttled clock, not the nominal one.

A fourth diagnostic is the framework. The case budgets the model and the arena but not the interpreter, the kernels, and the stack. A TFLM runtime can add 10 kB to 30 kB, which is the difference between fitting and not fitting in 64 kB.

A fifth diagnostic is the sensor. The case uses 5.5 mW, but the sensor's power depends on the LED current, the sampling rate, and the duty cycle. A sensor at 10 mW doubles the dominant term and halves the battery life.

A sixth diagnostic is the wake-up. The case ignores the wake-up energy, which can exceed the inference energy if the device wakes frequently. The measured energy, not the modeled energy, is the budget.

A seventh diagnostic is the battery derating. A 100 mAh battery at 0 °C delivers less than at 25 °C, and the capacity falls with age. The battery life should be quoted for the worst-case temperature and the end of life.

An eighth diagnostic is the memory fragmentation. A runtime that allocates and frees can fragment the heap and fail after hours or days, even if the high-water mark is below the limit. The arena in the case is preallocated to avoid fragmentation; a runtime that uses malloc does not have that guarantee.

A ninth diagnostic is the watchdog. A long inference can trigger the watchdog and reset the device. The watchdog should be fed during the inference or the inference should be chunked.

A tenth diagnostic is the radio. The case budgets 6.9 mW at 10% duty, but a connection that retries can double the radio time. The radio is the second-largest term, and it is the one most sensitive to the environment.

An eleventh diagnostic is the firmware image size. The update image includes the model, the runtime, and the application. A model that fits the flash may not fit the update partition, which is often half the flash.

A twelfth diagnostic is the test coverage. The on-device validation should include the worst-case input, the longest window, and the lowest battery. A test that runs only on clean, full-battery data does not exercise the failure modes.

The case's twelve diagnostics are the checklist for a deployment review. A product that passes all twelve has a deployment plan; a product that fails one has a risk.

The list is not exhaustive, but it covers the failures that are predictable before launch.

## 7. Code, query, or calculation

The script computes the memory, latency, energy, battery life, quantization error, and ring buffer for the case. It uses only the standard library.

```python
import math, random

params=3362
macs=3751280
print("int8_kb=%.2f fp32_kb=%.2f" % (params/1024, params*4/1024))

a1=3750*8; a2=1875*16; a3=938*32
print("act_kb=%.1f %.1f %.1f peak=%.1f" % (a1/1024,a2/1024,a3/1024,max(a1,a2,a3)/1024))
print("arena_kb=%.1f" % ((max(a1,a2)*2 + params)/1024))

for mops in (50,100,500,1000):
    print("mops=%d latency_ms=%.2f" % (mops, macs/(mops*1e6)*1000))

for lat in (7.5,37.5):
    e=25e-3*lat/1000
    print("lat=%.1fms energy_mJ=%.3f" % (lat,e*1000))

cap_j=0.1*3.7*3600
print("battery_J=%.0f" % cap_j)
for e_mj in (0.19,0.94,1.79):
    n=cap_j/(e_mj/1000)
    print("e=%.2fmJ inferences=%.0f days_at_0.5Hz=%.1f"
          % (e_mj,n,n/(0.5*86400)))

random.seed(1)
w=[random.gauss(0,0.1) for _ in range(10000)]
mx=max(abs(x) for x in w); scale=mx/127
q=[max(-127,min(127,round(x/scale))) for x in w]
dq=[x*scale for x in q]
mse=sum((w[i]-dq[i])**2 for i in range(len(w)))/len(w)
print("scale=%.5f mse=%.3e rmse=%.3e" % (scale,mse,math.sqrt(mse)))

fs=100; sec=2; ch=2
buf=fs*sec*ch*2
print("ring_buffer_bytes=%d kB=%.1f" % (buf,buf/1024))
```

Expected output:

```text
int8_kb=3.28 fp32_kb=13.13
act_kb=29.3 29.3 29.3 peak=29.3
arena_kb=61.9
mops=50 latency_ms=75.03
mops=100 latency_ms=37.51
mops=500 latency_ms=7.50
mops=1000 latency_ms=3.75
lat=7.5ms energy_mJ=0.188
lat=37.5ms energy_mJ=0.938
battery_J=1332
e=0.19mJ inferences=7010526 days_at_0.5Hz=162.3
e=0.94mJ inferences=1417021 days_at_0.5Hz=32.8
e=1.79mJ inferences=744134 days_at_0.5Hz=17.2
scale=0.00284 mse=6.762e-07 rmse=8.223e-04
ring_buffer_bytes=800 kB=0.8
```

Three implementation notes.

The activation memory assumes that each layer's output is materialized. An inference engine can reuse buffers, so the peak is the maximum of the layers, not the sum. The arena is the maximum plus the double buffer and the weights.

The energy assumes a constant 25 mW during the inference. A real processor has a sleep current, a wake-up energy, and a voltage regulator efficiency. The estimate is a lower bound.

The quantization error is computed on a Gaussian weight distribution, not on the trained weights. A trained model has outliers and a different distribution; the error must be measured after quantization.

A second block computes the full power budget, including the sensor, the MCU, the radio, and the sleep current.

```python
sensor_mw = 5.5
mcu_mw = 25.0
radio_mw = 6.9
sleep_mw = 8.1e-3
d_mcu = 0.019
d_radio = 0.10
d_sleep = 0.98
day = 86400
cap_j = 0.1*3.7*3600

e_sensor = sensor_mw*1e-3*day
e_mcu = mcu_mw*1e-3*d_mcu*day
e_radio = radio_mw*1e-3*d_radio*day
e_sleep = sleep_mw*1e-3*d_sleep*day
total = e_sensor + e_mcu + e_radio + e_sleep

print("e_sensor=%.1f e_mcu=%.1f e_radio=%.1f e_sleep=%.2f"
      % (e_sensor, e_mcu, e_radio, e_sleep))
print("total=%.1f battery_life_days=%.2f" % (total, cap_j/total))
print("average_mw=%.2f" % (total/day*1000))
```

Expected output:

```text
e_sensor=475.2 e_mcu=41.0 e_radio=59.6 e_sleep=0.69
total=576.5 battery_life_days=2.31
average_mw=6.67
```

The output shows that the sensor is 82% of the daily energy, the radio 10%, the MCU 7%, and the sleep 0.1%. A model optimization that halves the MCU energy saves 3.5% of the daily energy. A sensor duty cycle of 50% saves 237.6 J, or 41%. The system design, not the model, determines the battery life.

## 8. Data, standards, and regulatory context

Deployment is regulated as software in a medical device. The standards cover the lifecycle, the risk, the cybersecurity, and the change management.

| Source | What it constrains | Status | Practical consequence |
|---|---|---|---|
| IEC 62304 | Software lifecycle | Consensus; often required | Development and maintenance process |
| ISO 14971 | Risk management | Consensus; often required | Hazard analysis |
| FDA premarket software guidance (2023) | Software documentation | Binding through review | Architecture, V&V, SBOM |
| FDA cybersecurity guidance | Security | Binding through review | Threat model, updates |
| FDA PCCP guidance (2024/2025) | Model updates | Binding through review | Pre-authorized changes |
| IEC 62443 | Industrial cybersecurity | Voluntary | Security controls |
| ISO/IEC 27001 | Information security | Voluntary | Management system |
| EU MDR | Clinical evaluation | Binding in the EU | PMS and vigilance |
| EU AI Act | AI transparency | Binding in the EU | Governance and oversight |

### 8.1 The software lifecycle

IEC 62304 defines software safety classes A, B, and C, based on the risk of harm. A physiological monitor with an alert is often class B or C. The class determines the required activities: planning, requirements, architecture, detailed design, unit verification, integration testing, and system testing.

The lifecycle is not a waterfall. The standard allows iterative development, but the artefacts must exist. A team that develops iteratively and documents at the end is not compliant.

### 8.1.1 The software safety class

IEC 62304 assigns a safety class based on the risk of harm from a software failure.

| Class | Harm | Example | Activities |
|---|---|---|---|
| A | No injury | Logging | Basic |
| B | Non-serious injury | Wellness trend | Intermediate |
| C | Serious injury or death | Arrhythmia alarm | Full |

A physiological monitor with an alarm is often class B or C. The class determines the required artefacts, the reviews, and the testing. A team that starts with class A documentation and discovers class C at review will redo the work.

The class is a property of the intended use, not of the algorithm. A model that is accurate and an alarm that is wrong can be class C.

### 8.2 Cybersecurity and updates

The FDA cybersecurity guidance requires a threat model, a software bill of materials, and a plan for security updates. A device that cannot be updated is a device with a growing vulnerability.

The update mechanism must be authenticated, atomic, and recoverable. A signed image, a rollback partition, and a version check are the minimum. An unsigned update is a remote code execution vulnerability.

### 8.3 Change control

A model update is a change to the device function. The change must be validated and, for a regulated device, authorized. The predetermined change control plan is the mechanism for planned changes.

The plan should describe the modification, the protocol, the acceptance criteria, and the monitoring. A change outside the plan requires a new submission.

### 8.3.1 The PCCP components for a deployment

A predetermined change control plan for a model update has three parts.

| Part | Content | Deployment question |
|---|---|---|
| Description | What changes | Model, input, or threshold |
| Protocol | How it is validated | Data, metrics, acceptance |
| Impact | Benefits and risks | Bias, safety, performance |

The description must be specific. "Retrain the model" is not a description; "retrain the AF classifier on data from the same device family, with the same architecture, and a prespecified accuracy criterion" is.

The protocol must include the data management, the retraining practice, the evaluation, and the update procedure. The impact assessment must include the subgroup performance and the monitoring.

### 8.4 What the standards do not cover

Three gaps.

First, the standards do not specify the latency or energy requirement. A device can be compliant and too slow for the decision. The requirement is a clinical and product decision.

Second, the standards do not specify the on-device validation method. A team can validate on a laptop and ship a quantized model without measuring the on-device accuracy. The gap is a validation gap.

Third, the standards do not specify the telemetry. A device can be deployed without monitoring. The monitoring is a post-market obligation in some jurisdictions and a good practice everywhere.

A fourth gap is the definition of "real time." The standards do not say what latency is acceptable for an alert. The clinical requirement must come from the use case, not from the standard.

A fifth gap is the on-device model version. A device can report a performance number without reporting the model version that produced it. The version should be in the telemetry and in the label.

| Gap | Question the standard does not answer | Who must answer |
|---|---|---|
| Latency | How fast is fast enough | Clinical and product |
| Energy | How long must the battery last | Product |
| On-device validation | How to test the target | Engineering |
| Telemetry | What to monitor | Post-market |
| Model version | How to trace a result | Quality system |

The table is a reminder that the standards are a floor, not a complete design.

A deployment that meets the standards and misses the deadline has met the floor and failed the product. The requirements table in Section 5, Step 1 is the product-specific layer that the standards do not provide.

## 9. Industry practice and case evidence

### 9.1 Q-PPG and microcontroller inference

The Q-PPG work deployed PPG-based heart-rate models on an STM32WB55 (Cortex-M4 at 64 MHz) with a MAX30101 sensor and a 370 mAh battery. The reported active power was 25 mW for the MCU, 5.5 mW for the sensor, and 6.9 mW for the connected radio. The inference energy ranged from 1.39 mJ to 47.65 mJ depending on the model size, with the smallest model at 1.79 mJ. The best quantized model had a mean absolute error of 4.41 bpm.

The documented lesson is that the sensing and radio dominate the energy at low duty cycles. The inference is a fraction of the total, and the model size affects the energy but not as much as the always-on sensor.

### 9.2 TFLite Micro and keyword spotting

The TFLite Micro runtime has a small footprint: about 9 kB for a convolutional reference model and about 13 kB for a keyword-spotting model on a Cortex-M4. A keyword-spotting application can run in about 70 kB on a Cortex-M7 with a 12 ms inference.

The documented lesson is that the runtime is not the bottleneck for small models; the model and the buffers are. A team that budgets 64 kB for the model and forgets the runtime will not fit.

### 9.3 Consumer wearable power budgets

Reference designs for wearables use 50 mAh to 200 mAh batteries and average currents of a few milliamps. A 100 mAh battery at 1 mA lasts about 92 hours. The sensor and the radio are the largest consumers; the compute is small when duty-cycled.

The inference for a product team is that the battery life is a system property, not a model property. The model can be optimized, but the sensor duty cycle is the larger lever.

### 9.4 On-device ECG classification

Consumer ECG devices perform the classification on the device or on the paired phone. The on-device path has a latency and privacy advantage; the phone path has a compute advantage. The architecture is chosen for the claim and the user experience.

The documented pattern is that the classification is often on the phone, not the watch, because the phone has more compute and a larger battery. The inference is that the "edge" can be the phone, not only the sensor.

### 9.4.1 A deployment comparison table

| Product | Compute | Battery | Model | Latency |
|---|---|---|---|---|
| Wrist PPG HR | MCU | 100-370 mAh | Tiny CNN | 1-2 s |
| Single-lead ECG | MCU/phone | 200-300 mAh | Small CNN | 30 s |
| Sleep wearable | MCU | 100-200 mAh | Features/CNN | Minutes |
| Clinical monitor | Applications | Mains | Larger | < 1 s |
| Cloud screening | Server | N/A | Large | Seconds |

The table is illustrative of the categories, not of specific products. The point is that the compute platform follows the product class, and the model must follow the platform.

### 9.4.2 A worked comparison of edge and cloud

A 30 s ECG classification can run on a phone or in the cloud.

| Item | Phone | Cloud |
|---|---|---|
| Model size | 100 MB | 1 GB |
| Latency | 50 ms | 2-3 s |
| Energy | Phone battery | Radio + server |
| Privacy | Data on phone | Data transferred |
| Update | App store | Server |
| Offline | Yes | No |

The phone is the better edge for this task because it has more compute than the watch and more privacy than the cloud. The inference is that "edge" is a spectrum, not a binary.

### 9.4.3 The deployment evidence table

| Claim | Evidence | Where |
|---|---|---|
| Latency | End-to-end timing | On-device |
| Energy | Power measurement | Bench |
| Memory | High-water mark | On-device |
| Accuracy | Held-out test | On-device |
| Security | Threat model | Review |
| Update | Fault injection | Bench |
| Privacy | Data-flow review | Review |

The table is the deployment evidence that a reviewer will ask for. An offline accuracy number is not enough.

The evidence should be collected on the production firmware, not on a development build. A development build can differ in the compiler flags, the optimization level, and the enabled features.

### 9.5 What the cases have in common

- **The sensor dominates the energy.** The model is a small fraction at low duty cycles.
- **The runtime has a footprint.** The framework memory is part of the budget.
- **The platform determines the latency.** The same model can differ by 5× across platforms.
- **The update is a regulated change.** The model version is part of the device.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Model does not fit | Flash or RAM overflow | Too many parameters | No deployment | Compress; smaller model |
| Quantization loss | Accuracy drop | Scale mismatch | Claim invalid | QAT; mixed precision |
| Latency over budget | Window plus inference | Long window | Missed alert | Shorten window; edge |
| Battery too short | Daily recharge | Sensor always on | Non-adherence | Duty cycle |
| Thermal limit | Device warm | Continuous compute | Comfort and safety | Duty cycle; heat sink |
| Runtime overhead | High-water mark | Framework memory | Crash | Budget the runtime |
| Operator unsupported | Slow fallback | Missing kernel | Latency | Use supported ops |
| Update brick | No rollback | Partial write | Device unusable | Atomic update |
| Unsigned update | No signature | Security | Compromise | Secure boot |
| Privacy leak | Raw data sent | Over-collection | Regulatory and trust | Minimize; encrypt |
| Drift undetected | No telemetry | No monitoring | Silent failure | Monitoring plan |
| Device mix shift | New device | Distribution | Performance change | Stratify telemetry |
| Clock throttling | Latency varies | Thermal | Missed deadline | Measure under load |
| Watchdog reset | Device reboots | Long inference | Data loss | Feed watchdog; chunk |
| Memory fragmentation | Fails after hours | malloc/free | Crash | Arena |
| Clock throttle | Latency doubles | Thermal | Missed deadline | Budget at throttled clock |
| Update image too large | Update fails | Partition size | No update | Compress; A/B partition |
| Battery derating | Shorter life in cold | Temperature | Non-adherence | Worst-case budget |
| Radio retry | Higher energy | Poor connection | Battery drain | Backoff |
| Wrong model version | Results differ | Version mismatch | Unverified | Version check |
| Telemetry privacy | Data over-collection | Over-monitoring | Compliance | Minimize |
| Tail latency | Mean meets, p99 misses | Jitter | Missed deadline | WCET budget |
| Interrupt blocking | Late inference | ISR priority | Deadline miss | Mask or budget |
| Log PHI | Raw signal retained | Convenience | Privacy breach | Retention policy |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The model fits the flash | Deployment | Too many parameters | Build and flash |
| The model fits the RAM | Runtime | Activation size | High-water mark |
| The runtime fits the budget | Execution | Framework overhead | Measure |
| The quantization is accurate | Claim | Distribution shift | On-device test |
| The latency meets the decision | Utility | Window plus inference | End-to-end timing |
| The energy meets the battery | Adherence | Sensor and radio | Power measurement |
| The thermal limit is respected | Safety | Continuous compute | Temperature test |
| The update is atomic | Recovery | Power loss | Fault injection |
| The update is authenticated | Security | Unsigned image | Secure boot test |
| The privacy design is adequate | Compliance | Re-identification | Data-flow review |
| The monitoring detects drift | Safety | No telemetry | Simulated drift |
| The device mix is known | Monitoring | New devices | Telemetry audit |
| The update partition fits | Update | Image too large | Partition test |
| The clock is stable under load | Latency | Thermal throttle | Stress test |
| The watchdog is fed | Reliability | Long inference | Timing test |
| The battery is derated | Life estimate | Cold, age | Worst-case test |
| The telemetry is minimized | Privacy | Over-collection | Data-flow review |
| The WCET meets the deadline | Alert utility | Mean only | Tail measurement |
| The logs exclude PHI | Privacy | Raw signal retained | Log audit |
| The A/B partition fits | Update | Image too large | Partition check |

## 12. Exercises

### Exercise 1. Memory budget (application, 30 min)

**Deliverable:** compute the int8 weights, the largest activation, and the arena for a model with 10,000 parameters and a largest activation of 20,000 values.

**Acceptance criteria:** weights, activation, double buffer, arena; the fit against 64 kB and 128 kB.

**Time:** 30 minutes.

### Exercise 2. Latency (application, 20 min)

**Deliverable:** compute the latency for 5 million MACs at 50, 100, and 500 MOPS.

**Acceptance criteria:** three values; the platform that meets a 20 ms budget.

**Time:** 20 minutes.

### Exercise 3. Energy and battery (application, 30 min)

**Deliverable:** compute the daily energy and battery life for a 200 mAh battery, a 5 mW sensor, a 20 mW MCU at 2% duty, and a 10 mW radio at 5% duty.

**Acceptance criteria:** all components; total; battery life in days.

**Time:** 30 minutes.

### Exercise 4. Quantization (application, 30 min)

**Deliverable:** compute the scale and RMS error for weights with a maximum absolute value of 0.5 and a Gaussian standard deviation of 0.15.

**Acceptance criteria:** scale; RMS error; the relative error.

**Time:** 30 minutes.

### Exercise 5. Architecture choice (design, 35 min)

**Deliverable:** choose edge, cloud, or hybrid for three decisions: fall detection, nightly sleep report, AF screening.

**Acceptance criteria:** latency, privacy, and power justified for each.

**Time:** 35 minutes.

### Exercise 6. Update plan (design, 40 min)

**Deliverable:** an update plan with versioning, signing, rollback, and revalidation for a model change.

**Acceptance criteria:** atomicity, authentication, rollback, and evidence; the PCCP link.

**Time:** 40 minutes.

### Exercise 7. Privacy design (design, 30 min)

**Deliverable:** a data-flow diagram for a wearable that minimizes collection and transfer.

**Acceptance criteria:** raw versus feature; retention; encryption; re-identification risk.

**Time:** 30 minutes.

### Exercise 8. On-device validation (design, 45 min)

**Deliverable:** a plan to measure accuracy, latency, energy, and memory on the target device.

**Acceptance criteria:** all four metrics; the high-water mark; the thermal condition.

**Time:** 45 minutes.

### Exercise 9. Deadline analysis (application, 25 min)

**Deliverable:** given a mean inference of 30 ms, a 20% clock throttle, a 10 ms interrupt, and a 50 ms deadline, decide whether the design meets the deadline at the tail.

**Acceptance criteria:** the throttled latency computed; the interrupt added; the margin or miss stated; a mitigation proposed.

**Time:** 25 minutes.

### Submission standard

A deployment budget must show the memory, the latency, the energy, and the battery life, with the assumptions. A quantization exercise must show the scale and the error. An update plan must include signing, atomicity, rollback, and revalidation.

An answer that reports only the model latency is incomplete. An answer that reports only the MCU current is incomplete. The system budget is the requirement.

## 13. Self-check answers

**1. What is the int8 size of 10,000 parameters?** 10,000 bytes = 9.77 kB.

**2. What is the float32 size?** 40,000 bytes = 39.06 kB.

**3. What is the arena for a 20,000-value activation and 10,000 weights?** 20,000 × 2 + 10,000 = 50,000 bytes = 48.8 kB.

**4. Compute the latency for 5 million MACs at 100 MOPS.** 5e6/100e6 = 0.05 s = 50 ms.

**5. Compute the energy at 20 mW for 50 ms.** 20e-3 × 0.05 = 1 mJ.

**6. Compute the battery energy for 200 mAh at 3.7 V.** 0.2 × 3.7 × 3600 = 2,664 J.

**7. Compute the daily energy for a 5 mW sensor.** 5e-3 × 86,400 = 432 J.

**8. Compute the battery life for 432 J/day and 2,664 J.** 6.17 days.

**9. What is the quantization scale for a maximum weight of 0.5?** 0.5/127 = 0.00394.

**10. What is the maximum quantization error?** Half the scale, 0.00197.

**11. Why is the sensor often the largest energy consumer?** It runs continuously, while the compute is duty-cycled.

**12. Why is the window part of the latency?** The decision cannot be made until the window is complete.

**13. Why is the runtime memory part of the budget?** The framework uses RAM and flash in addition to the model.

**14. Why is an update a regulated change?** It changes the device function and may change the performance.

**15. Why is on-device validation necessary?** Quantization, operators, and timing can change the accuracy and latency.

**16. What is the privacy advantage of edge processing?** Raw data stay on the device; only features or decisions are transmitted.

**Rubric for Exercise 5 (architecture).** A strong answer matches the latency and privacy requirement to the architecture and states the trade-off. A weak answer chooses cloud for everything.

**Rubric for Exercise 6 (update plan).** A strong plan has signed images, atomic updates, rollback, and revalidation. A weak plan says "push an update."

## 14. Interview questions

**1. How do you decide between edge and cloud?**
A strong answer uses latency, privacy, power, and model size. A weak answer says edge is faster.

**2. How do you fit a model on a microcontroller?**
A strong answer quantizes, prunes, reduces the input, and measures the high-water mark. A weak answer says "use a smaller model."

**3. What is the latency budget?**
A strong answer includes the window, the inference, the decision, and the notification. A weak answer reports only the inference.

**4. Why does quantization lose accuracy?**
A strong answer explains the scale, the outliers, and the propagation. A weak answer says "fewer bits."

**5. How do you measure energy?**
A strong answer integrates power over the duty cycle and includes the sensor and radio. A weak answer measures the MCU current only.

**6. What is the biggest energy consumer in a wearable?**
A strong answer says the sensor or the radio, depending on the duty cycle, and shows the budget. A weak answer says the processor.

**7. How do you update a deployed model safely?**
A strong answer uses signed, atomic updates, rollback, and revalidation. A weak answer says OTA.

**8. What is a PCCP?**
A strong answer describes the pre-authorized change plan. A weak answer says it is a regulatory document.

**9. Why is on-device accuracy different from offline accuracy?**
A strong answer cites quantization, operator differences, and timing. A weak answer says it should be the same.

**10. How do you monitor a deployed model?**
A strong answer monitors input, score, calibration, device mix, and failures. A weak answer monitors accuracy only.

**11. What is the privacy risk of raw signal transmission?**
A strong answer explains re-identification and the data volume. A weak answer says encryption solves it.

**12. How do you handle thermal limits?**
A strong answer duty-cycles, reduces power, and measures the surface temperature. A weak answer ignores it.

**13. What is WCET and why is the mean not enough?**
A strong answer defines the worst-case execution time and explains that a deadline is a maximum. A weak answer says the average is representative.

**14. How do you log on a device without storing PHI?**
A strong answer keeps timestamps, versions, and flags, aggregates features, and excludes raw signal or identifiers. A weak answer stores everything for debugging.

**15. What happens when an inference overruns?**
A strong answer uses a watchdog, a defined safe action, and a rollback after repeated resets. A weak answer says the device restarts.

## 15. Key terms

- **Edge inference** — computation on the device.
- **Latency** — time from input to output.
- **Throughput** — inferences per second.
- **Energy** — power times time.
- **Power** — energy per second.
- **Battery capacity** — stored charge.
- **Duty cycle** — active fraction.
- **Flash** — non-volatile storage.
- **RAM** — working memory.
- **Arena** — preallocated memory pool.
- **Activation** — intermediate tensor.
- **Quantization** — reduced precision.
- **Scale** — quantization step.
- **Zero point** — asymmetric offset.
- **Pruning** — removing weights.
- **Distillation** — small model from large.
- **Operator** — compute primitive.
- **Kernel** — implemented operator.
- **CMSIS-NN** — Arm neural network library.
- **TFLM** — TensorFlow Lite for Microcontrollers.
- **MAC** — multiply-accumulate.
- **MOPS** — million operations per second.
- **Thermal limit** — maximum temperature.
- **OTA** — over-the-air update.
- **Rollback** — revert an update.
- **Secure boot** — verified firmware.
- **SBOM** — software bill of materials.
- **Privacy** — protection of data.
- **On-device validation** — testing on target.
- **Device mix** — distribution of devices.
- **Observability** — ability to diagnose behaviour.
- **Fail-safe** — defined action on failure.
- **Safe state** — output suppressed or defaulted.
- **Log retention** — how long logs are kept.

## 16. Sources

All URLs accessed 25 October 2026.

1. David R, Duke J, Jain A, et al. TensorFlow Lite Micro: embedded machine learning for TinyML systems. *Proceedings of Machine Learning and Systems*. 2021. https://arxiv.org/abs/2010.08678
2. TensorFlow. Micro speech example. https://android.googlesource.com/platform/external/tensorflow/+/60c6e0991e4/tensorflow/lite/experimental/micro/examples/micro_speech/
3. Zhang Y, Suda N, Lai L, Chandra V. Hello Edge: keyword spotting on microcontrollers. 2017. https://arxiv.org/abs/1711.07128
4. Q-PPG: energy-efficient PPG-based heart rate monitoring on wearable devices. *IEEE Transactions on Biomedical Circuits and Systems*. 2022. https://arxiv.org/abs/2203.14907
5. Analog Devices. MAXREFDES1044 reference design. https://www.analog.com/en/resources/reference-designs/maxrefdes1044.html
6. Analog Devices. Need clinical-grade PPG from your wearable? https://www.analog.com/en/resources/technical-articles/need-clinical-grade-ppg-from-your-wearable.html
7. Texas Instruments. TIDA-010029 reference design. https://www.ti.com/lit/ug/tidueg6a/tidueg6a.pdf
8. IEC 62304, Medical device software — software life cycle processes. https://webstore.iec.ch/
9. ISO 14971, Application of risk management to medical devices. https://www.iso.org/
10. US Food and Drug Administration. Content of Premarket Submissions for Device Software Functions. 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
11. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions. 2024/2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_validation.md`](09_validation.md)
- Related topic: [`07_deep-learning-for-signals.md`](07_deep-learning-for-signals.md)
- Related topic: [`04_ppg-and-wearables.md`](04_ppg-and-wearables.md)
- Module survey file: [`../../course/11_physiological_signals_wearables.md`](../../course/11_physiological_signals_wearables.md)
- Lab: [`../../labs/`](../../labs/)
