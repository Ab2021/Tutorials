# SoCs, NPUs, and GPUs

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 06 of 14
> Estimated study time: 4 hours
> Prerequisites: 05
> Version 2026.3

## 1. Why this topic exists

A system on chip (SoC) puts a central processing unit (CPU), a graphics processing unit (GPU), a neural processing unit (NPU), image signal processors, memory controllers, and I/O on one die. It is the architecture behind most edge artificial intelligence (AI) devices: handheld ultrasound, retinal cameras, wearable monitors with on-device classification, and surgical video systems.

The industry problem is that an SoC's headline throughput is not the throughput of a real model. A device marketed at 40 trillion operations per second (TOPS) may deliver 2 TOPS on a batch-of-one medical image because the model is memory-bound, the runtime has launch overhead, and the thermal design limits the sustained clock. The gap between the headline and the delivered performance is where deployment schedules and clinical latency claims fail.

Three decisions depend on SoC literacy:

1. **Whether the model fits the device.** The fit is not only the parameter count; it is the memory footprint, the activation size, the operator support, and the precision.
2. **Whether the latency is real.** The latency depends on the batch size, the memory bandwidth, the runtime, and the thermal state, not only on the TOPS.
3. **Whether the evidence is achievable.** A heterogeneous SoC with a vendor runtime and a Linux stack is harder to verify and to update than a microcontroller. The evidence burden is part of the architecture choice.

This topic covers heterogeneous compute, the CPU–GPU–NPU division of labour, memory bandwidth, quantization, runtime overhead, thermal throttling, and the deployment workflow. It assumes the processor material from Topic 05 and the conversion material from Topic 04. It does not teach model training or compiler internals. It teaches how to read a device specification and predict what a model will actually do.

A useful framing: an SoC is a set of engines sharing a memory system. The engines are fast; the memory is finite; the runtime is overhead. A model that fits the memory and the engine's operator set will run well; a model that does not will run at a fraction of the headline.

### 3.6 The three numbers that predict deployment

Three numbers predict most of the deployment outcome:

1. **Arithmetic intensity.** Operations divided by bytes. It decides whether the model is compute- or memory-bound.
2. **Density.** The fraction of the model's operations that run on the fastest engine, rather than falling back. It decides the effective throughput.
3. **Sustained power.** The power in the production enclosure after thermal soak. It decides the energy and the throttling.

A model with a high intensity, high density, and low sustained power deploys well. A model that is weak on any one of the three will disappoint, regardless of the TOPS.

## 2. Learning outcomes

By the end of this topic you can:

- Explain the roles of CPU, GPU, NPU, and digital signal processor (DSP) in a heterogeneous SoC.
- Estimate the compute time, memory time, and energy of an inference from the model's operations, bytes, and the device's throughput and bandwidth.
- Distinguish dense from sparse TOPS and explain why the headline number is not the delivered number.
- Choose a precision (float32, float16, int8, mixed) from the accuracy and latency requirements.
- Predict the effect of batch size, runtime overhead, and thermal throttling on latency.
- Partition a pipeline across the CPU and an accelerator and compute the transfer cost.
- Identify the operator, memory, and toolchain constraints that cause a model to fall back to the CPU.
- Write a deployment verification plan that measures latency, accuracy, power, and thermal behaviour on the target.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

An SoC is a memory system surrounded by compute engines. The CPU is general-purpose and latency-optimised; the GPU is throughput-optimised for parallel floating-point work; the NPU is throughput-optimised for low-precision matrix and convolution work; the DSP is optimised for signal processing. All of them read and write the same memory, and the memory bandwidth is often the limit.

```text
        +-------------------------------------------------------------+
        |                         SoC                                 |
        |  +--------+   +--------+   +--------+   +--------+          |
 sensor -->|  ISP   |-->|  CPU   |-->|  GPU   |-->|  NPU   |--> output |
        |  +--------+   +--------+   +--------+   +--------+          |
        |       |            |            |            |              |
        |       +------------+------------+------------+              |
        |                         |                                   |
        |                  +------+------+                            |
        |                  |  Memory     |  LPDDR / SRAM              |
        |                  |  controller |  bandwidth B GB/s          |
        |                  +-------------+                            |
        +-------------------------------------------------------------+
```

The model has four equations:

\[
T_{\text{compute}} = \frac{\text{operations}}{\text{TOPS} \times 10^{12} \times \text{utilization}}
\]

\[
T_{\text{memory}} = \frac{\text{bytes}}{\text{bandwidth} \times 10^9}
\]

\[
T_{\text{latency}} = \max(T_{\text{compute}}, T_{\text{memory}}) + T_{\text{overhead}}
\]

\[
E = P \times T_{\text{latency}}
\]

The first two give the roofline; the third adds the runtime overhead; the fourth gives the energy. The maximum, not the sum, is the first-order estimate because compute and memory overlap. The overhead is the part that does not overlap: kernel launches, framework dispatch, and synchronization.

### 3.1 The engines

| Engine | Strength | Weakness | Typical medical workload |
|---|---|---|---|
| CPU | General-purpose, low latency, flexible | Low throughput, high energy per operation | Pre/post-processing, control, orchestration |
| GPU | Parallel floating-point, mature tools | Power, memory bandwidth, poor for small models | Image reconstruction, rendering, training |
| NPU | Low-precision matrix and convolution, high TOPS/W | Operator coverage, quantization constraints | Classification, detection, segmentation |
| DSP | Fixed-point filtering, low latency | Specialised toolchain | ECG, audio, ultrasound front end |
| ISP | Image processing, low power | Fixed function | Camera pipeline |

The engines are not interchangeable. A convolution that runs on the NPU may fall back to the CPU if the kernel size or the padding is unsupported. A fallback is not a minor performance detail; it can change the latency by an order of magnitude and the power by a factor of five.

### 3.2 Dense versus sparse TOPS

A sparse TOPS rating assumes that the model has zeros that the hardware can skip. A dense TOPS rating assumes no zeros. A model with structured sparsity can approach the sparse number; a dense model cannot.

| Rating | Assumption | When it applies |
|---|---|---|
| Sparse TOPS | 2:4 or similar structured sparsity | Model pruned and compiled for the sparse engine |
| Dense TOPS | No zeros skipped | Dense model |
| Peak TOPS | Perfect utilization, no memory limit | Never in practice |

A device rated at 40 sparse TOPS and 20 dense TOPS delivers at most 20 TOPS on a dense model, and usually much less because of memory and overhead. The headline is a ceiling, not a prediction.

### 3.3 Memory bandwidth is the real limit

A model's arithmetic intensity determines whether it is compute-bound or memory-bound. For an image classifier with 2 billion operations and 20 MB of weights and activations, the intensity is 100 operations per byte. A device with 20 TOPS and 68 GB/s has a ridge point of 294 operations per byte; the model is memory-bound, and the achievable throughput is:

\[
\text{Throughput} = 68\ \text{GB/s} \times 100\ \text{ops/byte} = 6.8\ \text{TOPS}
\]

The device delivers 6.8 TOPS, not 20. The memory bandwidth is the limit, and the model's intensity is the reason.

A model with 20 billion operations and the same 20 MB has an intensity of 1,000 operations per byte, above the ridge point; it is compute-bound and can approach 20 TOPS. The same device delivers very different fractions of its peak depending on the model.

### 3.4 Quantization

Quantization reduces the precision of the weights and activations, which reduces the memory traffic and often increases the throughput. The common precisions are:

| Precision | Bits | Dynamic range | Typical use | Accuracy risk |
|---|---:|---|---|---|
| Float32 | 32 | Wide | Training, reference | None |
| Float16 | 16 | Wide | Inference on GPU/NPU | Low |
| Bfloat16 | 16 | Wide exponent | Training, inference | Low |
| Int8 | 8 | −128 to 127 | Inference on NPU/CPU | Moderate |
| Int4 | 4 | −8 to 7 | Aggressive compression | High |
| Binary | 1 | ±1 | Research | Very high |

The accuracy risk is model- and data-dependent. A model that is robust to quantization may lose less than 0.5 percentage points of accuracy; a model with sharp decision boundaries may lose much more. The loss must be measured on the target device and the clinical population, not only on the training set.

Quantization also changes the operator support. An NPU may support int8 convolution but not int8 layer normalization; the unsupported operator falls back to the CPU, which can dominate the latency.

### 3.5 The deployment pipeline

```text
 train (float32) --> prune/distill --> quantize (PTQ/QAT) --> compile (target runtime)
   --> validate accuracy --> measure latency/power --> deploy --> monitor
```

Each step can change the model. The validation must be repeated after each step, on the target, with the clinical data. A model that passes in the training framework and fails on the device is a model that was never validated for the device.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| SoC | System on chip; CPU plus accelerators and peripherals | Used for any processor | None |
| GPU | Graphics processing unit | Assumed to be efficient for small models | None |
| NPU | Neural processing unit | Confused with a GPU | None |
| DSP | Digital signal processor | Used for any signal processing | None |
| ISP | Image signal processor | Confused with a GPU | None |
| TOPS | Trillion operations per second | Confused with MACs per second | 10^12 ops/s |
| MAC | Multiply-accumulate | Confused with an operation | 2 operations |
| Dense TOPS | Throughput with no zeros skipped | Confused with sparse | TOPS |
| Sparse TOPS | Throughput with structured sparsity | Assumed for dense models | TOPS |
| Utilization | Fraction of peak achieved | Confused with efficiency | % |
| Arithmetic intensity | Operations per byte | Confused with operations per second | ops/byte |
| Roofline | Performance bound by compute or memory | Used as marketing | ops/byte |
| Bandwidth | Bytes per second | Confused with TOPS | GB/s |
| Latency | Time from input to output | Confused with throughput | ms |
| Throughput | Inferences per second | Confused with latency | 1/s |
| Batch size | Number of inputs processed together | Assumed to reduce latency | Samples |
| Quantization | Reducing numerical precision | Confused with compression | bits |
| PTQ | Post-training quantization | Assumed to be lossless | None |
| QAT | Quantization-aware training | Confused with PTQ | None |
| Calibration | Choosing quantization ranges from data | Confused with sensor calibration | None |
| Operator | A computational primitive | Assumed to be supported | None |
| Fallback | Execution on a less efficient engine | Ignored in latency estimates | None |
| Runtime | Software that executes the model | Confused with the framework | None |
| Graph compiler | Tool that maps a model to an engine | Assumed to be optimal | None |
| Thermal throttling | Clock reduction at high temperature | Ignored in sustained workloads | °C, MHz |
| Power mode | Predefined power/clock configuration | Confused with a governor | W |
| Energy per inference | Joules per inference | Confused with power | J |
| Edge AI | Inference on the device | Confused with cloud AI | None |
| Model size | Bytes of weights | Confused with operations | MB |
| Activation | Intermediate tensor | Ignored in memory estimates | MB |

## 5. The mechanics, step by step

The procedure below takes a trained model and deploys it to a heterogeneous SoC. Each step states its input, output, and rationale. The edge cases required by the assignment are mapped to the steps and summarised in Section 5.13.

### Step 1. Characterise the model

**Input:** the trained model, its layers, and its input/output shapes.

**Output:** operations per inference, parameter bytes, activation bytes, and the operator list.

**Rationale:** the operations determine the compute bound; the bytes determine the memory bound; the operator list determines the fallback risk. A model with 100 layers and 20 operators has a different deployment risk from one with 10 layers and 5 operators.

Count the operations at the precision that will be deployed. An int8 convolution may have the same operation count as a float32 one, but the memory traffic is four times lower, which changes the roofline.

### Step 2. Characterise the input pipeline

**Input:** the sensor, the frame rate, and the preprocessing.

**Output:** input data rate, preprocessing operations, and memory traffic.

**Rationale:** the input pipeline can dominate the latency. A 4K image at 30 frames per second is 24 MB per frame at 3 bytes per pixel; the resize, normalize, and color conversion may take longer than the inference. The pipeline must be measured end to end, not only the model.

For a medical image, the preprocessing may include windowing, resampling, and normalization that are not part of the model but are required for it. These operations are part of the latency and the power budget.

### Step 3. Choose the precision

**Input:** the accuracy requirement and the engine's supported precisions.

**Output:** a precision and a quantization plan.

**Rationale:** the precision determines the memory traffic, the throughput, and the accuracy. A float32 model on a GPU may be accurate but slow and power-hungry; an int8 model on an NPU may be fast but lose accuracy on rare classes.

The choice must be validated on the clinical data, stratified by the subgroups that matter. A model that loses 0.2% overall accuracy may lose 5% on a rare but important class.

### Step 4. Estimate the compute time

**Input:** the operations, the engine's dense TOPS, and the utilization.

**Output:** a compute-time estimate.

**Rationale:** the compute time is the operations divided by the effective throughput. The effective throughput is the dense TOPS times the utilization, and the utilization depends on the model, the batch size, and the memory.

A utilization of 30–60% is typical for a well-supported model on an NPU; 10–30% for a GPU with a small batch; 5–20% for a CPU. The estimate is a range, not a point.

### Step 5. Estimate the memory time

**Input:** the bytes per inference, the memory bandwidth, and the memory access pattern.

**Output:** a memory-time estimate.

**Rationale:** the memory time is the bytes divided by the bandwidth. The bytes include the weights, the activations, and the input/output. The access pattern matters: a sequential access achieves the peak bandwidth; a random access achieves a fraction.

For a batch of one, the weights are read once per inference. For a batch of eight, the weights are read once and reused, so the per-inference memory traffic falls. This is why batching improves throughput, and why it does not necessarily improve latency.

### Step 6. Estimate the latency

**Input:** the compute time, the memory time, and the runtime overhead.

**Output:** a latency estimate and a bound.

**Rationale:** the latency is the maximum of the compute and memory times, plus the overhead. The overhead includes the kernel launch, the framework dispatch, the synchronization, and the data transfer. For a small model, the overhead can dominate; for a large model, it is a small fraction.

A runtime with a 1 ms launch overhead adds 1 ms to every inference, regardless of the model size. A model that takes 0.2 ms on the accelerator but 1 ms in the runtime has a 1.2 ms latency, and the accelerator is 17% utilized.

### Step 7. Estimate the energy

**Input:** the power in the chosen mode and the latency.

**Output:** energy per inference and energy per day.

**Rationale:** the energy is the power times the latency, plus the idle energy. The power depends on the mode: a 15 W mode is faster but may use more energy per inference than a 7 W mode if the race-to-idle gain does not offset the higher power.

The energy per day depends on the duty cycle. A device that infers once per second and sleeps in between is dominated by the sleep power; a device that infers continuously is dominated by the active power.

### Step 8. Partition the pipeline

**Input:** the model's layers and the engines' capabilities.

**Output:** a mapping of layers to engines and a data-transfer budget.

**Rationale:** not every layer belongs on the NPU. A layer with an unsupported operator may run on the CPU; a preprocessing step may run on the ISP; a control loop may run on a microcontroller. The partition must minimize the transfers, because each transfer costs time and energy.

A partition that alternates between the NPU and the CPU for every layer is worse than running the whole model on the CPU, because the transfers dominate. A partition that keeps the model on one engine and moves only the input and output is better.

### Step 9. Compile and check operator support

**Input:** the model and the target runtime.

**Output:** a compiled model and a report of supported and fallback operators.

**Rationale:** the runtime's compiler maps the model to the engine. If an operator is unsupported, the compiler inserts a fallback, often to the CPU. The fallback may be correct but slow; it may also change the numerical result.

The compiler report is a required artifact. A deployment that does not check the report may ship a model with an unexpected fallback that dominates the latency.

### Step 10. Validate accuracy on the target

**Input:** the compiled model and a clinical validation set.

**Output:** accuracy, subgroup results, and a comparison with the float model.

**Rationale:** the compiled model may differ from the trained model because of quantization, operator fusion, and fallback. The differences must be measured on the target, with the target's preprocessing, not on a desktop.

The validation must include the subgroups and the edge cases. A model that is accurate on average and wrong on a rare class is not ready for deployment.

### Step 11. Measure latency, power, and thermal behaviour

**Input:** the target device and the production workload.

**Output:** a latency distribution, a power trace, and a thermal trace.

**Rationale:** the estimates are hypotheses; the measurements are evidence. Measure the latency at the production batch size, the power in the production mode, and the temperature over a sustained run. A device that meets the latency for 10 s and throttles after 60 s does not meet the requirement.

The measurement must use the production software stack, including the runtime, the drivers, and the framework. A measurement on a desktop with a different runtime is not evidence.

### Step 12. Document and monitor

**Input:** the compiled model, the runtime, the configuration, and the measurements.

**Output:** a deployment record and a monitoring plan.

**Rationale:** the deployment record must include the model version, the precision, the compiler version, the runtime version, the power mode, and the calibration data. The monitoring plan must detect a change in accuracy, latency, or power after an update.

A device that updates its model without revalidation is a device that can drift silently. The monitoring plan is part of the safety case, not an operational nicety.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 1, 4 | Workload exceeds throughput | Utilization > 100%; queue growth | Dropped frames |
| Common-mode rejection failure | 2 | Analogue issue; not SoC-specific | N/A | N/A |
| Electrode offset | 2 | Analogue issue | N/A | N/A |
| Clock jitter | 11 | DVFS or thermal transition | Latency distribution | Jitter |
| Brown-out | 11, 12 | Supply drop | Supply monitor | Reset; lost inference |
| Watchdog reset loop | 12 | Firmware fails to service watchdog | Reset counter | Device unavailable |
| Thermal throttling | 11 | Clock reduction at high temperature | Temperature and latency trace | Sustained latency increase |
| Firmware update failure | 12 | Incomplete write; bad signature | Version and checksum | Bricked device |
| Calibration drift | 10, 12 | Model or sensor drift | Accuracy monitoring | Silent degradation |

The SoC-specific edge cases are thermal throttling, DVFS jitter, and update failure. The analogue edge cases are handled in earlier topics; they enter the SoC through the input pipeline.

### 5.14 Precision selection table

| Requirement | Precision | Engine | Risk |
|---|---|---|---|
| Accuracy within 0.1% of float | Float32 or float16 | GPU/CPU | Latency, power |
| Accuracy within 1% | Int8 with calibration | NPU | Subgroup loss |
| Maximum throughput | Int8 | NPU | Operator fallback |
| Maximum compression | Int4 | NPU with int4 support | Accuracy loss |
| Low power, small model | Int8 | MCU with DSP | Throughput |

The precision is not a free choice. The engine must support it, the model must tolerate it, and the validation must confirm it.

### 5.15 Operator fallback cost

| Operator | NPU support | Fallback | Latency impact |
|---|---|---|---|
| 3×3 convolution | Yes | CPU | 10–100× slower |
| 1×1 convolution | Yes | CPU | 10–50× slower |
| Depthwise convolution | Usually | CPU | 10–50× slower |
| Layer normalization | Often no | CPU | Dominates small models |
| Softmax | Often no | CPU | Small but serializing |
| Custom activation | No | CPU | 10–100× slower |
| Resize | ISP or CPU | CPU | Depends on size |

A single unsupported layer can dominate the latency. The compiler report is the only reliable way to know which layers fall back.

### 5.16 Roofline for three models

| Model | Operations | Bytes | Intensity (ops/byte) | Bound on a 20 TOPS, 68 GB/s device |
|---|---:|---:|---:|---|
| Small CNN | 0.2 G | 2 MB | 100 | Memory: 6.8 TOPS |
| Retinal classifier | 2.0 G | 18.8 MB | 106 | Memory: 7.2 TOPS |
| Large segmentation | 20 G | 40 MB | 500 | Compute: 20 TOPS |
| Transformer block | 1 G | 8 MB | 125 | Memory: 8.5 TOPS |

The ridge point of the device is 20 TOPS / 68 GB/s = 294 ops/byte. Every model in the table except the large segmentation is memory-bound. The device's headline 20 TOPS is never achieved by the small models.

The table also shows why model compression helps memory-bound models. Halving the bytes doubles the intensity, moving the model toward the ridge point. Quantizing from float32 to int8 reduces the bytes by four, which can move a memory-bound model to compute-bound.

### 5.17 Runtime overhead sources

| Source | Typical overhead | When it dominates |
|---|---:|---|
| Kernel launch | 5–50 µs | Small models, many layers |
| Framework dispatch | 10–100 µs | Python or high-level API |
| Memory allocation | 10–100 µs | Dynamic shapes |
| Synchronization | 1–10 µs | Per-layer synchronization |
| Data transfer CPU–NPU | 10–100 µs | Partitioned models |
| Preprocessing | 1–20 ms | Large images, CPU resize |
| Driver | 1–10 µs | Every call |

A model with 100 layers and a 20 µs launch overhead per layer has 2 ms of overhead, which can exceed the compute time. A runtime that fuses layers reduces the launch count and the overhead. A model that is compiled into a single graph has one launch, not one hundred.

The overhead is not a constant. It depends on the runtime, the driver, the operating system, and the load. A measurement on an idle device underestimates the overhead under load.

### 5.18 Batch size arithmetic

A batch of \(B\) images reuses the weights across the batch. The memory traffic per inference is:

\[
\text{Bytes per inference} = \frac{W + B \times A}{B} = \frac{W}{B} + A
\]

where \(W\) is the weight bytes and \(A\) is the activation bytes per image. For \(W = 8 MB and A = 10 MB, a batch of one reads 18 MB per inference; a batch of eight reads 11 MB per inference, a 39% reduction.

The compute time per inference may also fall because the engine is better utilized. A batch of one may use 25% of the NPU; a batch of eight may use 70%. The throughput improves by a factor of 2.8, but the latency for the first image in the batch increases because the batch must be filled.

| Batch | Bytes per inference | Utilization | Latency per batch | Latency per image |
|---:|---:|---:|---:|---:|
| 1 | 18 MB | 25% | 8.0 ms | 8.0 ms |
| 4 | 12 MB | 50% | 12 ms | 3.0 ms |
| 8 | 11 MB | 70% | 16 ms | 2.0 ms |

Batching is a throughput optimization, not a latency optimization. A device that must respond to each image immediately cannot wait for a batch.

### 5.19 Power modes and thermal design

An SoC's power mode sets the clock, the voltage, and the number of active cores. A 15 W mode is faster but hotter; a 7 W mode is slower but cooler. The choice depends on whether the device is battery-powered, handheld, or plugged in.

| Mode | Power | Latency | Thermal rise | Use |
|---|---:|---:|---:|---|
| 7 W | 7 W | 12 ms | Low | Battery, handheld |
| 15 W | 12 W | 8 ms | Moderate | Plugged in, sustained |
| 25 W | 20 W | 6 ms | High | Benchtop, active cooling |

The thermal rise is not linear with power. A 15 W mode in a sealed handheld may reach the junction limit and throttle; the same mode in a benchtop with a fan may not. The thermal design is part of the latency claim.

A sustained workload must be tested for at least the duration of a clinical session: 30 minutes for a clinic visit, 8 hours for a monitoring shift, 24 hours for a wearable. A 60 s test does not reveal the throttle.

### 5.20 Numerical differences after quantization

Quantization changes the model's outputs. The differences are usually small, but they can change a decision near a threshold.

| Source | Effect | Detection |
|---|---|---|
| Weight rounding | Small shift in logits | Compare outputs on a validation set |
| Activation clipping | Saturation of large values | Histogram of activations |
| Calibration range | Poor range choice | Per-channel calibration; outlier analysis |
| Operator fusion | Different rounding order | Layer-by-layer comparison |
| Fallback to CPU | Different precision | Compiler report; output comparison |

The validation must compare the quantized model with the float model on the clinical data, not only on the training set. A model that agrees on 99% of images and disagrees on the 1% that matter is not validated.

### 5.21 What the datasheet does not tell you

A datasheet describes the peak performance in a reference design. It does not describe:

- The achieved utilization on your model.
- The memory bandwidth under your access pattern.
- The runtime overhead of your framework.
- The thermal behaviour in your enclosure.
- The operator support for your model.
- The accuracy after quantization.

Each is a system property. The datasheet is necessary but not sufficient; the deployment must be measured on the target.

### 5.22 Partition arithmetic

A pipeline has three stages: preprocessing (CPU), inference (NPU), post-processing (CPU). The preprocessing takes 2 ms, the inference 6 ms, and the post-processing 1 ms. If each stage runs on its own, the total is 9 ms. If the pipeline is streamed, the throughput is limited by the slowest stage (6 ms), and the latency of the first result is still 9 ms.

The partition must minimize the transfers. Moving the preprocessing to the ISP saves the CPU time but adds a transfer of 1 MB, which at 10 GB/s takes 0.1 ms. The net gain is 1.9 ms. Moving the post-processing to the NPU is possible only if the operators are supported; otherwise the fallback adds 3 ms.

| Partition | Latency | Transfers | Risk |
|---|---:|---:|---|
| All CPU | 40 ms | 0 | High latency |
| CPU + NPU | 9 ms | 2 | Transfer overhead |
| ISP + NPU + CPU | 7.1 ms | 3 | ISP programming |
| All NPU | 6 ms | 1 | Operator fallback |

The table shows that the best partition depends on the operator support and the transfer cost, not only on the engine speeds.

### 5.23 Ten SoC review questions

1. What are the operations and bytes per inference?
2. What is the arithmetic intensity, and is the model compute- or memory-bound?
3. What is the dense TOPS, and what utilization is realistic?
4. What is the memory bandwidth, and what is the access pattern?
5. What precision is used, and how was it validated?
6. Which operators fall back, and what is the cost?
7. What is the runtime overhead at the production batch size?
8. What is the thermal behaviour over a clinical session?
9. What is the energy per inference and per day?
10. What is recorded so that a model update can be detected?

A review that answers all ten has covered the SoC as a system. A review that answers only the first and the third has reviewed the datasheet.

### 5.24 Memory allocation and fragmentation

An SoC's memory is shared among the operating system, the camera, the model, and the application. A model that allocates 18 MB at startup may fail after a day of operation because the memory is fragmented. The failure is intermittent and hard to reproduce.

The mitigation is to allocate the model's buffers once, at startup, and reuse them. A runtime that supports static memory planning is preferable to one that allocates per inference. The peak memory must be measured over a long run, not only at startup.

### 5.25 Measuring utilization on the target

Utilization is the ratio of the achieved throughput to the peak. It is not directly observable; it must be inferred from the latency and the operation count:

\[
\text{Utilization} = \frac{\text{operations}}{\text{peak throughput} \times \text{latency}}
\]

For a model with 2 billion operations, a peak of 20 TOPS, and a latency of 8 ms, the utilization is 2e9 / (20e12 × 8e-3) = 1.25%. The number is low, but it is not a failure of the model; it is a failure of the runtime and the memory.

A utilization below 10% usually means that the overhead or the memory dominates. A utilization above 50% means that the model is well matched to the engine. The number is a diagnostic, not a score.

A memory checklist:

- Allocate model buffers at startup.
- Reuse input and output buffers.
- Avoid dynamic shapes in the production path.
- Measure the peak over a 24-hour run.
- Log the free memory at each inference.
- Fail safely if the allocation fails.


## 6. Worked example

### 6.1 The requirement

A handheld retinal camera runs a diabetic retinopathy classifier on each 512 × 512 colour fundus image. The model has:

- 2.0 billion int8 operations (1.0 billion MACs).
- 8 MB of int8 weights.
- 10 MB of activations.
- A batch size of 1.
- A 100 ms latency requirement for the capture-to-result display.

Two candidate platforms:

| Platform | Peak int8 | Memory bandwidth | Power mode | Typical power |
|---|---:|---:|---|---:|
| SoC with GPU/NPU | 20 dense TOPS | 68 GB/s | 15 W | 12 W |
| Low-power NPU | 0.256 TOPS | 4 GB/s | — | 0.5 W |

The SoC's measured utilization on this model is 25%; the NPU's is 60%.

### 6.2 The compute time

The SoC's effective throughput is:

\[
20\ \text{TOPS} \times 0.25 = 5\ \text{TOPS}
\]

The compute time is:

\[
T_{\text{compute}} = \frac{2.0 \times 10^9}{5 \times 10^{12}} = 0.40\ \text{ms}
\]

The low-power NPU's effective throughput is:

\[
0.256\ \text{TOPS} \times 0.60 = 0.154\ \text{TOPS}
\]

The compute time is:

\[
T_{\text{compute}} = \frac{2.0 \times 10^9}{0.154 \times 10^{12}} = 13.0\ \text{ms}
\]

Both are below the 100 ms requirement, but the SoC is 32 times faster on compute.

### 6.3 The memory time

The bytes per inference are the weights plus the activations plus the input and output. The input is 512 × 512 × 3 = 786 kB; the output is small. Use 8 MB + 10 MB + 0.8 MB = 18.8 MB.

The SoC's memory time is:

\[
T_{\text{memory}} = \frac{18.8 \times 10^6}{68 \times 10^9} = 0.276\ \text{ms}
\]

The low-power NPU's memory time is:

\[
T_{\text{memory}} = \frac{18.8 \times 10^6}{4 \times 10^9} = 4.70\ \text{ms}
\]

The SoC is compute-bound at 0.40 ms; the NPU is memory-bound at 4.70 ms. The roofline flips between the two platforms because their bandwidth-to-compute ratios differ.

### 6.4 The measured latency

The SoC's measured latency is 8.0 ms, not 0.40 ms. The difference is the runtime overhead: kernel launches, framework dispatch, and synchronization. The overhead is 7.6 ms, or 95% of the total.

The low-power NPU's measured latency is 6.2 ms, not 4.70 ms. The overhead is 1.5 ms, or 24% of the total.

The overhead changes the comparison. The SoC is still faster, but by 1.3×, not 32×. The headline TOPS was a poor predictor of the delivered latency.

### 6.5 The energy

The SoC's energy per inference is:

\[
E_{\text{SoC}} = 12\ \text{W} \times 8.0\ \text{ms} = 96\ \text{mJ}
\]

The low-power NPU's energy is:

\[
E_{\text{NPU}} = 0.5\ \text{W} \times 6.2\ \text{ms} = 3.1\ \text{mJ}
\]

The low-power NPU uses 31 times less energy per inference. For a battery-powered handheld device, the NPU is the better choice even though it is slower, because both meet the 100 ms requirement and the NPU uses far less energy.

The energy per day depends on the number of images. At 200 images per day, the SoC uses 19.2 J and the NPU uses 0.62 J. The difference is small compared with the display and the radio, but it matters for a device that must run a full clinic day on a battery.

### 6.6 Diagnostic 1: the NPU is memory-bound

The low-power NPU's measured latency of 6.2 ms is close to the 4.70 ms memory estimate plus overhead, but the compute estimate was 13.0 ms. The model is actually memory-bound, not compute-bound, because the NPU's bandwidth is low relative to its compute.

The diagnostic is to reduce the activation size by using a smaller input or a more efficient architecture, and see whether the latency falls. If it falls proportionally to the bytes, the model is memory-bound. A second diagnostic is to measure the memory bandwidth with a streaming benchmark; if the achieved bandwidth is below the datasheet, the access pattern is the problem.

The design flaw is the assumption that the NPU's TOPS determines the latency. The bandwidth determines it, and the model's arithmetic intensity is below the ridge point.

### 6.7 Diagnostic 2: thermal throttling on the SoC

The SoC meets the 8 ms latency for the first 60 s of continuous operation, then the latency rises to 11 ms and the power falls from 12 W to 9 W. The device has throttled to stay within its thermal limit.

The diagnostic is to log the junction temperature and the clock frequency during a sustained run. If the clock falls when the temperature reaches the limit, the throttling is confirmed. A second diagnostic is to run the same workload with a heatsink or a lower ambient temperature; if the latency stays at 8 ms, the thermal design is the limit.

The design flaw is the assumption that the peak latency measured on a cool device applies to a warm device. A handheld device in a clinic is warm, and a sustained workload will throttle. The requirement must be verified after thermal soak, not only at startup.

### 6.8 Vary one assumption: the model doubles to 4 billion operations

Change only the model size, from 2.0 to 4.0 billion operations, by using a higher-resolution input. The SoC's compute time becomes 0.80 ms and its memory time roughly doubles to 0.55 ms; the measured latency becomes about 8.4 ms, because the overhead dominates. The NPU's compute time becomes 26.0 ms and its memory time about 9.4 ms; the measured latency becomes about 11 ms.

The SoC remains within the 100 ms requirement; the NPU also remains within it. The energy per inference rises to 101 mJ for the SoC and 5.5 mJ for the NPU. The relative advantage of the NPU grows, because it is memory-bound and the model's bytes grew less than its operations.

The sensitivity shows that the architecture choice is robust to a 2× model change for this requirement. A 10× change would make the NPU miss the 100 ms deadline and force the SoC.

### 6.9 What the example demonstrates

The arithmetic closes: 0.40 ms compute and 0.276 ms memory for the SoC, 13.0 ms compute and 4.70 ms memory for the NPU, 8.0 ms and 6.2 ms measured latency, and 96 mJ and 3.1 mJ per inference. The two diagnostics reveal different flaws: one is a memory bound, the other is thermal throttling. The sensitivity analysis shows that the architecture choice depends on the requirement and the model size, not on the headline TOPS.

The general method is to compute the roofline, measure the overhead, measure the thermal behaviour, and compare the energy over the duty cycle. A device that wins on TOPS may lose on latency, energy, or sustained operation.

The three numbers to carry into a design review are the delivered TOPS, the measured overhead, and the sustained power. The headline TOPS is a ceiling; the other two determine whether the device meets its claim.

### 6.13 A third platform: the microcontroller with an NPU

A microcontroller with a 0.256 TOPS NPU and 4 GB/s is the low-power platform in the table. A smaller model, with 100 million operations and 2 MB of weights and activations, changes the comparison:

| Platform | Compute time | Memory time | Measured latency | Energy |
|---|---:|---:|---:|---:|
| SoC | 0.02 ms | 0.03 ms | 8.0 ms (overhead) | 96 mJ |
| Low-power NPU | 0.65 ms | 0.50 ms | 1.2 ms | 0.6 mJ |

For the small model, the NPU is both fast enough and far more efficient. The SoC's overhead dominates, and its power is wasted. The architecture choice depends on the model size, not on the platform's peak.

The general lesson is that the smallest platform that meets the deadline is usually the most efficient. A larger platform is justified only when the model, the batch, or the latency requirement cannot be met otherwise.

### 6.10 Verification matrix

| Test | Condition | Metric | Acceptance |
|---|---|---|---|
| Accuracy | Clinical validation set | Sensitivity, specificity | Within 1% of float |
| Subgroup accuracy | Stratified by site and severity | Difference from overall | Within pre-specified bound |
| Latency, cold | Startup, 25 °C | p50 and p99 | < 100 ms |
| Latency, warm | 30 min sustained | p99 | < 100 ms |
| Power | Production mode | Average and peak | Within budget |
| Thermal | 30 min sustained, 30 °C ambient | Junction temperature | Below limit |
| Operator fallback | Compiler report | Fallback layers | None critical |
| Memory | Peak allocation | Heap and activation | Within device |
| Update | Interrupted update | Rollback | Recovers |
| Security | Signed image | Boot verification | Passes |

### 6.11 The thermal trace

| Time (min) | SoC power (W) | Junction (°C) | Latency (ms) |
|---:|---:|---:|---:|
| 0 | 12.0 | 45 | 8.0 |
| 5 | 12.0 | 62 | 8.0 |
| 10 | 11.5 | 75 | 8.5 |
| 20 | 9.5 | 85 | 10.5 |
| 30 | 9.0 | 87 | 11.0 |

The trace shows the throttle point at about 10 minutes. A device that is used for a 5-minute capture meets the 8 ms latency; a device used for a 30-minute session does not meet it after 10 minutes. The requirement must state the duration, not only the temperature.

### 6.12 Batch size four

If the camera captures four images before displaying, the batch size can be four. The SoC's memory traffic per inference falls to 12 MB, the utilization rises to 50%, and the latency per batch is about 12 ms, or 3 ms per image. The p99 per batch rises to 15 ms, still below the 100 ms requirement.

The batch changes the latency profile: the first image waits for the other three. For a capture-and-review workflow, the wait is acceptable; for a real-time video overlay, it is not. The batch size is a workflow decision, not only a performance decision.

The verification matrix and the thermal trace are the evidence; the roofline is the hypothesis.
## 7. Code, query, or calculation

The script reproduces the compute, memory, energy, batch, and roofline results in Section 6. It uses decimal gigabytes for bandwidth and megabytes for memory, which is the convention in device datasheets and the source of a common factor-of-1.024 confusion; state the convention when comparing with a measurement.

**Assumptions.** 2.0 billion int8 operations, 18.8 MB of weights and activations, a 20 dense TOPS SoC with 25% utilization and 68 GB/s, and a 0.256 TOPS NPU with 60% utilization and 4 GB/s. The measured latencies are 8.0 ms and 6.2 ms.

```python
ops = 2.0e9
bytes_total = 18.8e6
soc_tops = 20e12
soc_util = 0.25
soc_bw = 68e9
soc_power = 12.0
npu_tops = 0.256e12
npu_util = 0.60
npu_bw = 4e9
npu_power = 0.5

soc_rate = soc_tops * soc_util
npu_rate = npu_tops * npu_util
soc_compute = ops / soc_rate
npu_compute = ops / npu_rate
soc_mem = bytes_total / soc_bw
npu_mem = bytes_total / npu_bw
soc_lat = 8.0e-3
npu_lat = 6.2e-3
soc_e = soc_power * soc_lat
npu_e = npu_power * npu_lat

print("COMPUTE")
print(f"  SoC effective         {soc_rate/1e12:6.2f} TOPS")
print(f"  SoC compute time      {soc_compute*1e3:6.2f} ms")
print(f"  NPU effective         {npu_rate/1e12:6.3f} TOPS")
print(f"  NPU compute time      {npu_compute*1e3:6.2f} ms")
print()
print("MEMORY")
print(f"  SoC memory time       {soc_mem*1e3:6.3f} ms")
print(f"  NPU memory time       {npu_mem*1e3:6.2f} ms")
print()
print("ENERGY")
print(f"  SoC per inference     {soc_e*1e3:6.1f} mJ")
print(f"  NPU per inference     {npu_e*1e3:6.2f} mJ")
print(f"  ratio                 {soc_e/npu_e:6.1f}")
print()
print("BATCH")
W = 8e6
A = 10e6
for b in (1, 4, 8):
    per = (W + b*A)/b
    print(f"  batch {b}: {per/1e6:5.2f} MB per inference")
print()
print("RIDGE POINT")
print(f"  SoC ridge             {soc_tops/soc_bw:6.1f} ops/byte")
print(f"  model intensity       {ops/bytes_total:6.1f} ops/byte")
```

**Expected output.**

```text
COMPUTE
  SoC effective           5.00 TOPS
  SoC compute time        0.40 ms
  NPU effective           0.154 TOPS
  NPU compute time       13.02 ms

MEMORY
  SoC memory time         0.276 ms
  NPU memory time          4.70 ms

ENERGY
  SoC per inference       96.0 mJ
  NPU per inference        3.10 mJ
  ratio                   31.0

BATCH
  batch 1: 18.00 MB per inference
  batch 4: 12.00 MB per inference
  batch 8: 11.00 MB per inference

RIDGE POINT
  SoC ridge              294.1 ops/byte
  model intensity        106.4 ops/byte
```

The output shows four things.

First, the SoC's compute time is 0.40 ms and its memory time is 0.276 ms, so the roofline predicts a 0.40 ms inference. The measured latency is 8.0 ms, 20 times higher. The difference is the runtime overhead, not the hardware.

Second, the NPU's compute time is 13.0 ms and its memory time is 4.70 ms, so the roofline predicts 13.0 ms. The measured latency is 6.2 ms, lower than the compute estimate, because the model is actually memory-bound and the NPU's utilization on this model is higher than assumed. The estimate was wrong in the conservative direction.

Third, the SoC's energy per inference is 96 mJ and the NPU's is 3.1 mJ, a ratio of 31. The SoC's speed does not compensate for its power at this duty cycle.

Fourth, the SoC's ridge point is 294 ops/byte and the model's intensity is 106 ops/byte, so the model is memory-bound. A larger model with the same memory would be compute-bound; a smaller model would be even more memory-bound.

**A note on the overhead.** The 7.6 ms overhead is the difference between the 0.40 ms roofline and the 8.0 ms measurement. It is not a property of the model; it is a property of the runtime, the driver, and the framework. A different runtime, a compiled graph, or a fused operator set could reduce it. The overhead must be measured, not assumed.

### 7.1 Memory bandwidth measurement

The roofline assumes that the device achieves its datasheet bandwidth. The following script computes the achieved bandwidth from a copy benchmark and compares it with the peak.

```python
bytes_copied = 64e6          # 64 MB read plus 64 MB write = 128 MB traffic
seconds = 2.1e-3             # measured copy time
traffic = 2 * bytes_copied
achieved = traffic / seconds
peak = 68e9
print(f"traffic      {traffic/1e6:8.1f} MB")
print(f"achieved     {achieved/1e9:8.2f} GB/s")
print(f"peak         {peak/1e9:8.2f} GB/s")
print(f"efficiency   {100*achieved/peak:8.1f} %")
```

**Expected output.**

```text
traffic         128.0 MB
achieved         60.95 GB/s
peak             68.00 GB/s
efficiency        89.6 %
```

A copy benchmark achieves about 90% of the peak. A model's access pattern may achieve less, especially if it is random or strided. The achieved bandwidth, not the peak, belongs in the roofline.

## 8. Data, standards, and regulatory context

An SoC that runs a medical model is part of a regulated device. The applicable instruments cover the software lifecycle, risk management, cybersecurity, usability, and, when the model is adaptive, the AI-specific guidance.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 62304:2006+AMD1:2015 | Medical device software lifecycle | Consensus standard | Software class, architecture, verification, configuration |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis for latency, failure, and drift |
| IEC 81001-5-1:2021 | Health software security | Consensus standard | Secure lifecycle, threat model, SBOM |
| FDA premarket cybersecurity guidance | Device cybersecurity | Guidance enforced through review | Threat model, testing, SBOM, update plan |
| FDA AI/ML device guidance | AI-enabled device functions | Guidance | Data governance, performance, monitoring, PCCP |
| EU AI Act 2024/1689 | High-risk AI systems | Binding regulation in the EU | Risk management, data governance, transparency, monitoring |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | Use-related risk, display, alarms |
| IEC 60601-1-2:2014+AMD1:2020 | Electromagnetic disturbances | Collateral standard | Immunity of the SoC and its cables |

### 8.1 IEC 62304 and the software class

The software class depends on the harm that a failure could cause. A classification model that assists a clinician's review may be class B; a model that drives a closed-loop therapy may be class C. The class determines the evidence: architecture, detailed design, unit verification, integration testing, and regression testing.

An SoC with a vendor runtime adds evidence challenges. The runtime is third-party software, often supplied as a binary. The manufacturer must obtain the runtime's documentation, version, and known issues, and must verify the model's behaviour through the runtime. A runtime that cannot be verified is a risk that must be managed.

### 8.2 The FDA's AI/ML guidance

The FDA's guidance on AI-enabled device functions addresses the whole product lifecycle: the intended use, the data used to train and test, the performance across subgroups, the human-machine interaction, the monitoring plan, and the predetermined change control plan (PCCP) for updates that the manufacturer intends to make.

A PCCP describes the changes that the manufacturer anticipates, the data and methods used to develop them, and the verification and validation that will be performed. It allows a device to be updated without a new submission for every change, provided the changes stay within the plan. The plan is not a substitute for validation; it is a description of how validation will be done.

### 8.3 The EU AI Act

The EU AI Act (Regulation 2024/1689) classifies AI systems by risk. A medical device that uses AI is generally high-risk if it is a safety component of a product covered by the Medical Device Regulation. High-risk systems must meet requirements for risk management, data governance, technical documentation, transparency, human oversight, accuracy, robustness, and cybersecurity.

The Act's requirements overlap with the MDR's, but they are not identical. A manufacturer must map both sets of requirements to the device's evidence. The mapping is a regulatory task, not only an engineering one.

### 8.4 Cybersecurity and updates

An SoC that runs a model is a computer. It has an operating system, a network stack, and a runtime, each with vulnerabilities. The FDA's premarket cybersecurity guidance and IEC 81001-5-1 require a threat model, security testing, an SBOM, and a plan for updates.

A model update is a software update. It must be signed, verified, and rolled back if it fails. The update must not change the model's behaviour outside the validated scope. A device that updates its model without revalidation is a device that can drift.

### 8.5 What the standards do not cover

The standards do not specify:

- The SoC model or the runtime.
- The precision or the quantization method.
- The batch size or the latency distribution.
- The thermal design.
- The operator support.

Those are design decisions that the manufacturer must justify. A device can meet every clause and still deliver a model that is slow, hot, or inaccurate on the target.

### 8.6 The data implications

The deployment record must include the model version, the precision, the compiler and runtime versions, the power mode, and the calibration data. A change in any of them can change the output. The record is part of the dataset's provenance, not an operational detail.

## 9. Industry practice and case evidence

### 9.1 The Jetson class of edge SoCs

NVIDIA's Jetson Orin Nano 8GB module is a representative edge SoC. Its datasheet (DS-11105-001 v1.1, April 2023) lists 40 sparse or 20 dense INT8 TOPS, a 1024-core GPU with 32 tensor cores, a 6-core Arm Cortex-A78AE CPU at 1.5 GHz, 8 GB of LPDDR5 at 68 GB/s, and 7 W or 15 W power modes. A 2025 revision of the same datasheet lists higher "Super Mode" figures, 67 sparse or 34 dense TOPS and 7–25 W, with 102 GB/s for the 8 GB module.

The gap between the headline and the delivered performance is the subject of Section 6. The module's 20 dense TOPS is a ceiling; a batch-of-one medical model with a high-level runtime may achieve a small fraction of it.

**Documented fact:** the specifications are from the NVIDIA datasheet revisions cited. **Inference:** the delivered fraction depends on the model and the runtime, which must be measured.

### 9.2 Micro-NPUs

Arm's Ethos-U55 is a microcontroller-class NPU with 32, 64, 128, or 256 MACs per cycle, paired with a Cortex-M CPU. At 500 MHz, the 256-MAC configuration delivers 128 GMAC/s, or 256 GOPS, with a small area and power budget. The Ethos-U65 extends the range to 256 or 512 MACs per cycle, and the Ethos-U85 to 2,048 MACs per cycle.

The micro-NPU changes the architecture choice for small models. A 2-billion-operation model is too large for a 0.256 TOPS NPU at a 100 ms deadline, but a 20-million-operation model is comfortable. The boundary is set by the operations, the memory, and the deadline, not by the product category.

**Documented fact:** the MAC configurations are from Arm's published documentation. **Inference:** the delivered performance depends on the model's operator support and memory, which must be measured.

### 9.3 Detector-integrated processing

Butterfly Network's iQ3 uses a capacitive micromachined ultrasonic transducer integrated on an application-specific integrated circuit, which moves part of the signal processing into the probe. Siemens Healthineers' photon-counting CT detectors perform energy discrimination in the detector, before the reconstruction computer. In both cases, the SoC or ASIC is part of the sensor, not a separate box.

The architecture changes the data flow and the latency. A probe that digitises and beamforms internally sends images, not raw channels; the data rate falls, and the processing that was on a workstation moves into the handheld. The data scientist receives a different data product, with different noise and calibration.

**Documented fact:** the product architectures and clearances are public. **Inference:** the effect on a specific model or reconstruction is device-specific.

### 9.4 Recalls and software

The 2025 recalls of Virtual Incision's MIRA and Quantum Surgical's EPIONE involved software issues, and the CMR Surgical Versius console recall involved Secure Boot not being enabled. These are not SoC-specific, but they illustrate the evidence burden: a software or boot failure can have procedural consequences even when the hardware works.

For an SoC running a model, the analogous failures are a runtime that silently falls back to the CPU, a model that is updated without revalidation, and a boot chain that does not verify the image. Each is a software failure with a clinical consequence.

### 9.5 What good deployment practice looks like

Teams that avoid SoC surprises tend to:

- Measure the model on the target, with the production runtime and batch size.
- Read the compiler report and count the fallback operators.
- Validate the quantized model on the clinical population.
- Measure the latency after thermal soak, not only at startup.
- Record the model, runtime, and power mode with the data.
- Re-run the validation after any change to the model, runtime, or power mode.
- Keep a fallback path that is safe if the accelerator fails.

The practices are ordinary. The failures they prevent are not.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Memory-bound model | Low utilization, high bandwidth | Arithmetic intensity below the ridge point | Latency above estimate | Smaller model; quantization; better data reuse |
| Runtime overhead | Latency much higher than roofline | Kernel launch, dispatch, sync | Missed deadline | Compiled graph; fused operators; lower-level runtime |
| Operator fallback | Compiler report lists CPU nodes | Unsupported operator | Latency spike | Model rewrite; supported ops; different runtime |
| Thermal throttling | Latency rises after minutes | Clock reduction at the junction limit | Sustained latency miss | Better cooling; lower power mode; workload shaping |
| Quantization accuracy loss | Subgroup accuracy falls | Range clipping, rounding | Wrong decision | QAT; per-channel calibration; mixed precision |
| Memory fragmentation | Allocation fails after hours | Fragmented heap | Crash or dropped inference | Static allocation; buffer reuse |
| Power-mode mismatch | Latency differs from the datasheet | Wrong mode or governor | Missed deadline | Lock the mode; verify |
| Model update without revalidation | Accuracy drifts after update | New weights, same validation | Silent degradation | PCCP; regression test |
| Unsigned update | Boot accepts a modified image | Secure boot disabled | Compromised device | Verify boot; sign images |
| Driver regression | Latency changes after an OS update | Driver or kernel change | Missed deadline | Pin the OS and driver; regression test |
| Input pipeline bottleneck | Inference fast, end-to-end slow | Resize, color conversion, copy | Missed deadline | Move to ISP/GPU; measure end to end |
| Batch mismatch | Throughput good, latency poor | Batching waits for a full batch | Delayed result | Batch size 1 for latency; batch for throughput |
| Clock/voltage transition | Latency spikes during DVFS | Frequency change | Jitter | Lock frequency |
| Model too large for memory | Swap or allocation failure | Weights exceed RAM | Crash | Smaller model; quantization; more memory |

The most dangerous failure is the silent fallback. A model that runs partly on the CPU may still produce correct outputs, but with a latency that is unpredictable and a power that is higher than specified. The compiler report and the latency distribution are the only reliable detectors.

### 10.1 Ten red flags in SoC deployment data

1. A latency that is 10× the roofline.
2. A compiler report with CPU fallbacks on critical layers.
3. A utilization below 5% on a model that should be compute-bound.
4. A memory bandwidth below 50% of peak during inference.
5. A latency that rises after 10 minutes of operation.
6. A power draw above the mode's rating.
7. A model version that differs between devices in the same study.
8. An accuracy drop that appears only after quantization.
9. A batch size that differs between the bench and the clinic.
10. A thermal shutdown or throttle event in the logs.

Each is a prompt to measure, not a conclusion.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The operation count is accurate | Sizing the compute | Compiler fusion; sparsity | Profile on the target |
| The utilization is representative | Latency estimate | Model, batch, memory | Measure with the production runtime |
| The memory bandwidth is achieved | Memory estimate | Access pattern, contention | Measure with a streaming benchmark |
| The runtime overhead is small | Latency estimate | Framework, launches | Measure the end-to-end latency |
| The precision is supported | Throughput and accuracy | Operator coverage | Compiler report; validation |
| The model fits memory | No swapping | Weights plus activations | Measure peak memory |
| The thermal design holds | Sustained latency | Enclosure, ambient | Thermal soak test |
| The power mode is fixed | Deterministic latency | Governor, DVFS | Lock and verify |
| The validation population matches | Accuracy generalises | Subgroups, sites | Stratified validation |
| The update is controlled | No silent drift | Unsigned or unvalidated update | Signed images; regression test |
| The input pipeline is included | End-to-end latency | Resize, copy, color conversion | Measure end to end |
| The fallback path is safe | Fail-safe behaviour | Accelerator failure | Fault injection |

The dominant assumption is that the datasheet's TOPS applies to the model. It rarely does. The delivered throughput is the dense TOPS times the utilization, and the utilization is set by memory and operator support.

A second limit is that the roofline is a first-order model. It ignores the overlap of compute and memory, the latency of the memory hierarchy, and the runtime's scheduling. The roofline tells you which bound is closer; it does not predict the latency to within 10%.

A third limit is that the thermal behaviour is enclosure-specific. A module that runs at 15 W on a bench may throttle in a sealed handheld. The thermal test must use the production enclosure and the clinical ambient.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a datasheet?
- Which assumption was true on the bench but false in the enclosure?
- Which assumption depends on the model, the batch, or the runtime?
- Which assumption is encoded in a power mode or a compiler flag?
- Which assumption is specific to the validation population?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Compute a roofline (recall, 20 minutes)

A model has 1.5 billion operations and 12 MB of weights and activations. A device has 10 dense TOPS and 50 GB/s. Compute the arithmetic intensity, the ridge point, the compute time, and the memory time. State which bound dominates.

**Deliverable:** four calculations and a one-sentence conclusion.

**Acceptance criteria:** correct units; intensity and ridge point; the dominant bound identified.

**Time:** 20 minutes.

### Exercise 2. Compute the delivered throughput (application, 30 minutes)

A device is rated at 40 sparse and 20 dense TOPS. A dense model achieves 30% utilization. What is the delivered throughput? What would the vendor's headline suggest, and why is it wrong?

**Deliverable:** a calculation and a short explanation.

**Acceptance criteria:** dense rating used; utilization applied; the sparse/dense distinction explained.

**Time:** 30 minutes.

### Exercise 3. Compute energy per inference (application, 30 minutes)

A model runs in 9 ms at 12 W and in 14 ms at 7 W. Compute the energy per inference in both modes. Which mode is better for a battery-powered device that runs one inference per second and sleeps at 0.1 W?

**Deliverable:** a table with both modes and a recommendation.

**Acceptance criteria:** active and sleep energy separated; per-day total; recommendation justified.

**Time:** 30 minutes.

### Exercise 4. Diagnose a fallback (critique, 30 minutes)

A model's measured latency is 25 ms, but the roofline predicts 3 ms. The compiler report shows that layer normalization runs on the CPU. Explain the mechanism, estimate the cost, and propose two fixes.

**Deliverable:** a diagnosis and two fixes.

**Acceptance criteria:** fallback mechanism correct; cost estimated; fixes distinguish model rewrite from runtime change.

**Time:** 30 minutes.

### Exercise 5. Choose a precision (design, 35 minutes)

A segmentation model must maintain a Dice score within 0.5% of float and run in under 50 ms on an NPU. Compare float16, int8 PTQ, and int8 QAT, and recommend one with a validation plan.

**Deliverable:** a comparison table and a recommendation.

**Acceptance criteria:** accuracy and latency compared; validation plan includes subgroups; the recommendation states the risk.

**Time:** 35 minutes.

### Exercise 6. Partition a pipeline (design, 40 minutes)

A camera pipeline has preprocessing (5 ms on CPU), inference (8 ms on NPU), and post-processing (2 ms on CPU). Propose a partition that reduces the end-to-end latency by at least 30%, including the transfer costs.

**Deliverable:** a block diagram and a timing budget.

**Acceptance criteria:** transfers included; the new latency calculated; the risk of fallback named.

**Time:** 40 minutes.

### Exercise 7. Write a thermal test plan (design, 35 minutes)

Write a plan to verify the sustained latency of the SoC in Section 6 over a 30-minute clinical session. Include the ambient temperature, the enclosure, the workload, the measurements, and the acceptance criteria.

**Deliverable:** a test plan with at least 8 rows.

**Acceptance criteria:** production enclosure; sustained workload; junction temperature and latency; quantitative criteria.

**Time:** 35 minutes.

### Exercise 8. Evaluate an edge-AI claim (critique, 25 minutes)

A vendor claims "40 TOPS, real-time medical AI." Write five questions that would test the claim for a 30 fps image classifier.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover dense versus sparse, utilization, batch size, fallback, and thermal; no question is answerable by the headline.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Roofline | Four calculations |
| 2 | Delivered throughput | Calculation and explanation |
| 3 | Energy | Mode comparison |
| 4 | Fallback diagnosis | Mechanism and fixes |
| 5 | Precision choice | Comparison and plan |
| 6 | Partitioning | Diagram and budget |
| 7 | Thermal verification | Test plan |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** Intensity = 1.5e9 / 12e6 = 125 ops/byte. Ridge point = 10e12 / 50e9 = 200 ops/byte. The model is below the ridge, so it is memory-bound. Compute time = 1.5e9 / 10e12 = 0.15 ms. Memory time = 12e6 / 50e9 = 0.24 ms. The memory bound dominates, and the delivered throughput is about 6.25 TOPS, not 10.

**Answer 2 (Exercise 2).** Delivered = 20 dense TOPS × 0.30 = 6 TOPS. The vendor's 40 TOPS is the sparse rating, which assumes a pruned model and a sparse engine. A dense model cannot achieve it, and even a sparse model would need perfect utilization. The headline is a ceiling, not a prediction.

**Answer 3 (Exercise 3).** 12 W mode: 12 × 9 ms = 108 mJ per inference. 7 W mode: 7 × 14 ms = 98 mJ. The 7 W mode uses less energy per inference. Over a 1 s period with 0.1 W sleep: 12 W mode = 108 mJ + 99.1 mJ = 207.1 mJ; 7 W mode = 98 mJ + 98.6 mJ = 196.6 mJ. The 7 W mode is better for the battery, by about 5%.

**Answer 4 (Exercise 4).** Layer normalization on the CPU requires a transfer of the tensor to the CPU, the normalization, and a transfer back. The transfer and the CPU execution can dominate a small model. The cost is roughly the tensor size divided by the transfer bandwidth plus the CPU time. Fixes: replace layer normalization with a supported operator (for example, a scaled activation) and revalidate; or use a runtime that supports the operator on the NPU.

**Answer 5 (Exercise 5).** Float16 is likely within 0.1% of float and runs on the NPU, but it may be slower than int8. Int8 PTQ may lose more than 0.5% Dice on rare classes; int8 QAT is more likely to stay within 0.5% but requires retraining. Recommend int8 QAT with a stratified validation, and a fallback to float16 if the subgroup loss exceeds the bound.

**Answer 6 (Exercise 6).** Move preprocessing to the ISP or GPU (5 ms to 1 ms), keep inference on the NPU (8 ms), and fuse post-processing into the model or run it on the GPU (2 ms to 0.5 ms). The transfers add about 0.3 ms. The new latency is about 9.8 ms, a 51% reduction from 15 ms. The risk is that the fused post-processing is unsupported and falls back.

**Answer 7 (Exercise 7).** The plan: production enclosure, 30 °C ambient, 30-minute continuous run at 30 fps, log junction temperature and latency every second, measure power, and record the throttle point. Acceptance: p99 latency < 100 ms for the full 30 minutes; junction temperature below the limit; no thermal shutdown. If the device throttles, the requirement is not met.

**Answer 8 (Exercise 8).** Questions: (1) Is the 40 TOPS dense or sparse? (2) What utilization is achieved on a batch-of-one medical model? (3) What is the measured p99 latency at 30 fps? (4) Which operators fall back to the CPU? (5) What is the sustained latency after 30 minutes in the production enclosure? A headline cannot answer any of them.

**Rubric for the design exercises.** A strong design quantifies the roofline, the overhead, the precision, the thermal behaviour, and the validation. A weak design quotes TOPS and assumes the rest.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: intensity | 125 ops/byte | ±2 |
| Exercise 1: ridge point | 200 ops/byte | ±2 |
| Exercise 1: memory time | 0.24 ms | ±0.01 ms |
| Exercise 2: delivered | 6 TOPS | ±0.1 |
| Exercise 3: 12 W energy | 108 mJ | ±1 mJ |
| Exercise 3: 7 W energy | 98 mJ | ±1 mJ |
| Exercise 6: new latency | 9.8 ms | ±0.5 ms |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 10 | 5 | 20 |
| 2 | 5 | 15 | 10 | 30 |
| 3 | 5 | 15 | 10 | 30 |
| 4 | 5 | 15 | 10 | 30 |
| 5 | 5 | 20 | 10 | 35 |
| 6 | 5 | 25 | 10 | 40 |
| 7 | 5 | 20 | 10 | 35 |
| 8 | 5 | 10 | 10 | 25 |

## 14. Interview questions

**1. What is the difference between a GPU and an NPU?**
A strong answer explains the GPU's parallel floating-point generality and the NPU's low-precision matrix throughput and efficiency. A weak answer says the NPU is faster.

**2. Why is the headline TOPS not the delivered performance?**
A strong answer names dense versus sparse, utilization, memory bandwidth, and runtime overhead. A weak answer blames the model.

**3. What is the roofline model?**
A strong answer explains the compute and memory bounds and the ridge point, and uses it to decide whether more TOPS helps. A weak answer has not heard of it.

**4. When does quantization help, and when does it hurt?**
A strong answer explains the memory and throughput gains and the accuracy risk, and requires validation on the target. A weak answer says int8 is always faster.

**5. What causes a fallback, and why is it dangerous?**
A strong answer names unsupported operators and the CPU path, and explains the latency and power spike. A weak answer treats it as a minor detail.

**6. How do you measure the latency of a model on a device?**
A strong answer measures end to end, at the production batch size, after thermal soak, with the production runtime. A weak answer measures the model in the training framework.

**7. What is the effect of batch size on latency and throughput?**
A strong answer explains the weight reuse, the utilization gain, and the wait for a full batch. A weak answer says batching is always better.

**8. How does thermal throttling change a latency claim?**
A strong answer explains the clock reduction, the sustained workload, and the need for a soak test. A weak answer measures only at startup.

**9. What evidence does a model update require?**
A strong answer names the PCCP, the regression test, the subgroup validation, and the signed update. A weak answer says "test it".

**10. What would make you reject an SoC?**
A strong answer names a memory bound that cannot be fixed, a fallback that dominates, a thermal limit that misses the deadline, or a runtime that cannot be verified. A weak answer rejects on TOPS alone.

**11. How does memory bandwidth affect a model's latency?**
A strong answer computes the arithmetic intensity and the memory time, and compares it with the compute time. A weak answer ignores bandwidth.

**12. What do you record with the data to make the deployment traceable?**
A strong answer names the model, precision, runtime, compiler, power mode, and calibration. A weak answer names the device model only.

**13. What is the difference between latency and throughput?**
A strong answer defines latency as the time for one input and throughput as the inputs per second, and notes that batching improves one and worsens the other. A weak answer uses the terms interchangeably.

**14. How do you choose between a 7 W and a 15 W mode?**
A strong answer compares energy per inference and thermal behaviour over the duty cycle, not only latency. A weak answer chooses the faster mode.

## 15. Key terms

- **Activation** — intermediate tensor in a model.
- **Arithmetic intensity** — operations per byte.
- **Batch size** — number of inputs processed together.
- **Bfloat16** — 16-bit floating point with a wide exponent.
- **Compiler report** — list of supported and fallback operators.
- **Dense TOPS** — throughput with no zeros skipped.
- **DSP** — digital signal processor.
- **Edge AI** — inference on the device.
- **Energy per inference** — joules per inference.
- **Fallback** — execution on a less efficient engine.
- **Float16** — 16-bit floating point.
- **GPU** — graphics processing unit.
- **Graph compiler** — tool that maps a model to an engine.
- **ISP** — image signal processor.
- **Int8** — 8-bit integer precision.
- **Latency** — time from input to output.
- **MAC** — multiply-accumulate.
- **Model size** — bytes of weights.
- **NPU** — neural processing unit.
- **Operator** — computational primitive.
- **PCCP** — predetermined change control plan.
- **PTQ** — post-training quantization.
- **QAT** — quantization-aware training.
- **Quantization** — reducing numerical precision.
- **Roofline** — performance bound by compute or memory.
- **Sparse TOPS** — throughput with structured sparsity.
- **SoC** — system on chip.
- **Thermal throttling** — clock reduction at high temperature.
- **Throughput** — inferences per second.
- **TOPS** — trillion operations per second.
- **Utilization** — fraction of peak achieved.
- **Calibration set** — data used to choose quantization ranges.
- **Dense model** — a model with few or no zeros.
- **Kernel** — a compiled operator implementation.
- **Latency distribution** — the spread of inference times.
- **Memory footprint** — bytes of weights plus activations.
- **Model drift** — change in behaviour after an update or data shift.
- **Operator fusion** — combining layers into one kernel.
- **Sustained power** — power after thermal equilibrium.
- **Tensor** — multidimensional array.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Device documentation**

1. NVIDIA, *Jetson Orin Nano Series Datasheet*, DS-11105-001 v1.1, April 2023. https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/5380/Jetson_Orin_Nano_Series_DS-11105-001_v1.1.pdf
2. NVIDIA, *Jetson Orin Nano Super Mode* datasheet revision, 2025. https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
3. Arm, *Ethos-U55 NPU Technical Reference Manual*. https://support.arm.com/documentation/102420/0200/
4. Arm, *ML Developers Guide for Cortex-M Processors and Ethos-U NPU*. https://support.arm.com/documentation/109267/0103/
5. Arm, *Ethos-U55 Product Brief*. https://armkeil.blob.core.windows.net/developer/Files/pdf/ML%20on%20Arm/Arm_Ethos_U55_Product_Brief_v4.pdf

**Standards and regulation**

6. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
7. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
8. IEC 81001-5-1:2021, *Health software and health IT systems safety, effectiveness and security — Part 5-1*. https://webstore.iec.ch/en/publication/63293
9. U.S. Food and Drug Administration, *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket
10. U.S. Food and Drug Administration, *Artificial Intelligence and Machine Learning (AI/ML) for Medical Devices*. https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices
11. Regulation (EU) 2024/1689, *Artificial Intelligence Act*. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
12. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
13. IEC 60601-1-2:2014+AMD1:2020, *Electromagnetic disturbances*. https://webstore.iec.ch/en/publication/67554

**Device evidence**

14. Butterfly Network, Butterfly iQ3 launch and clearance, January 2024. https://www.butterflynetwork.com/press-releases/butterfly-network-launches-butterfly-iq3
15. Siemens Healthineers, Naeotom Alpha photon-counting CT. https://www.siemens-healthineers.com/en-us/press-room/press-releases/naeotom-family-fda-clearance
16. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm

**Background**

17. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
18. D. Patterson and J. Hennessy, *Computer Architecture: A Quantitative Approach*, 6th ed., Morgan Kaufmann, 2017.
19. NVIDIA, *Deep Learning Performance Guide*. https://docs.nvidia.com/deeplearning/performance/
20. Arm, *CMSIS-NN* documentation. https://arm-software.github.io/CMSIS-NN/
21. NVIDIA, *Jetson Linux Developer Guide*. https://docs.nvidia.com/jetson/
22. Arm, *Ethos-U85 Technical Reference Manual*. https://support.arm.com/documentation/102560/
23. U.S. Food and Drug Administration, *Predetermined Change Control Plans for Machine Learning-Enabled Medical Devices*. https://www.fda.gov/medical-devices/software-medical-device-samd/predetermined-change-control-plans-machine-learning-enabled-medical-devices
24. ISO 14971:2019, risk management. https://www.iso.org/standard/72704.html
25. IEC 62366-1:2015+AMD1:2020, usability. https://webstore.iec.ch/en/publication/26570
26. IEC 60601-1:2005+AMD1:2012+AMD2:2020, general safety. https://webstore.iec.ch/en/publication/26169

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_mcus-dsps-and-fpgas.md`](05_mcus-dsps-and-fpgas.md)
- Next topic: [`07_power-and-thermal.md`](07_power-and-thermal.md)
- Related: [`04_adcs-and-signal-conditioning.md`](04_adcs-and-signal-conditioning.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on TinyML quantization: see Lab 14A.3 in the survey file.
- Lab on hardware-in-the-loop testing: see Lab 14A.7 in the survey file.
- Lab index: [`../../labs/`](../../labs/)

### 8.7 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 62304 | Software class, model version, runtime version | Reproduce the evidence |
| ISO 14971 | Hazard analysis for latency and misclassification | Link failure to harm |
| IEC 81001-5-1 | SBOM, threat model, patch status | Security lifecycle |
| FDA AI/ML guidance | Training data, subgroups, monitoring plan, PCCP | Lifecycle evidence |
| EU AI Act | Risk management, data governance, transparency | Conformity |
| IEC 62366-1 | Use-related risk, summative test | Human factors |

The record is the same information the data team needs for provenance. A team that records the model, runtime, and power mode for its own analysis is also assembling regulatory evidence.

### 9.6 A note on vendor benchmarks

A vendor benchmark is a workload chosen to show the product in a favourable light. It may use a large batch, a sparse model, a fused operator set, and a cooled device. The medical model may use a batch of one, a dense model, an unsupported operator, and a sealed enclosure.

The benchmark is a shortlist tool, not a decision tool. The decision requires the medical model on the target, with the production runtime and the clinical workload.
- **Asynchronous execution** — the engine runs while the CPU continues.
- **Bus width** — bits transferred per memory cycle.
- **Compute-bound** — limited by arithmetic, not memory.
- **Frame buffer** — memory holding an image frame.
- **Memory-bound** — limited by bandwidth, not arithmetic.
- **Pipeline stall** — a bubble caused by a dependency or a miss.
- **Quantization error** — difference between float and quantized output.
- **Sparse model** — a model with structured zeros.
- **Tensor core** — a matrix-multiply accelerator.
- **Thermal design power** — the power the cooling system must remove.

Further reading:

27. D. Patterson and J. Hennessy, *Computer Architecture*, 6th ed., 2017.
28. NVIDIA, *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/
29. Arm, *CMSIS-NN* documentation.
30. J. G. Webster, *Medical Instrumentation*, 4th ed., 2009.
31. IEC 62304:2006+AMD1:2015, software lifecycle.
32. ISO 14971:2019, risk management.
33. IEC 81001-5-1:2021, security lifecycle.
- **Accelerator utilization** — fraction of the accelerator's peak in use.
- **Bandwidth-bound** — limited by memory bandwidth.
- **Device tree** — description of hardware passed to the kernel.
- **Execution provider** — runtime backend for a model.
- **Memory pool** — preallocated memory for tensors.
- **Operator set** — the operators a runtime supports.
- **Power governor** — policy that sets clock and voltage.

Additional sources:

34. U.S. Food and Drug Administration, *AI/ML-enabled medical devices*.
35. Regulation (EU) 2024/1689, AI Act.
36. IEC 62366-1:2015+AMD1:2020, usability.
37. Arm, *Ethos-U55 Product Brief*.
38. NVIDIA, *Jetson Orin Nano datasheet*.
39. IEC 60601-1-2:2014+AMD1:2020, EMC.
40. ISO 14971:2019, risk management.
