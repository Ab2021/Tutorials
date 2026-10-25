# Target Hardware and Performance

> Module 14B — Efficient AI and On-Device Inference
> Topic 02 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01, 14A
> Version 2026.3

## 1. Why this topic exists

Every deployment argument eventually reduces to a measurement on a specific board, at a specific clock, running a specific build of a specific runtime. The model on the workstation is a description of arithmetic. The model on the device is a program competing for memory bandwidth, cache, bus cycles, and thermal headroom with everything else in the product.

Teams get this wrong in a consistent way. They estimate latency from floating-point operations (FLOPs), choose a processor from a peak TOPS number, and discover on the bench that the model is four times slower than predicted. The four times is not noise. It is the gap between the arithmetic the model performs and the data movement the machine requires, plus the gap between peak throughput and achieved throughput.

Three specific failures motivate this topic.

**The FLOPs-to-latency fallacy.** A model with 4.1 million multiply-accumulate operations (MACs) does not take 4.1 million cycles divided by the clock. It takes whatever the memory system, the kernel implementation, and the scheduling allow. On a microcontroller, a convolution can be limited by static random-access memory (SRAM) bandwidth long before it is limited by arithmetic. On a graphics processing unit (GPU), a small layer can be limited by kernel launch overhead rather than by either. FLOPs are a lower bound on work, not an estimate of time.

**The peak-number fallacy.** A processor marketed at 4 TOPS (tera-operations per second) achieves that number only for a specific operator, precision, batch size, and memory residency. A clinical model with depthwise convolutions, layer normalization, and a non-maximum suppression stage will not approach it. The relevant number is the achieved throughput on your model, which is usually a small fraction of peak.

**The system fallacy.** The target is a board inside an enclosure inside a workflow. The board has a clock that drops when it gets hot, a bus shared with a radio, and a memory hierarchy with wait states. A latency measured for 30 seconds on a desk is an upper bound on performance, not a field estimate.

The purpose of this topic is to teach three instruments that close these gaps: analytic operation counting, the roofline model, and a profiling procedure that produces target-specific evidence. The instruments are simple enough to use in a design review and honest enough to say when a prediction is not yet evidence.

The clinical stakes are direct. A ventricular fibrillation detector that needs 400 ms on a device that can only deliver 900 ms of compute per second has a battery problem and a safety problem. A sepsis alert that runs on a gateway with a 700 ms p99 cannot support a one-minute escalation target once transport and human response are added. The performance model is how these facts become visible before the hardware is frozen.

### 1.1 The four numbers a design review needs

A performance review does not need a full roofline. It needs four numbers, each with a condition attached.

| Number | What it answers | Condition that must accompany it |
|---|---|---|
| MACs per inference | How much arithmetic the model performs | Counting convention and input shape |
| Bytes per inference | How much data must move | Memory level and inclusion set |
| Achieved throughput | How fast the target actually runs it | Hardware, batch, runtime, percentile |
| Sustained power | What it costs in energy and heat | Voltage, current, ambient, duty cycle |

With these four, a reviewer can reproduce the roofline placement, estimate the energy per inference, and check whether a proposed compression method attacks the binding resource. Without them, the review is a conversation about impressions.

### 1.2 The questions that expose a weak performance claim

- "Which memory level are those bytes counted at?" A claim that counts only weights is not a bandwidth claim.
- "Is that peak or sustained?" A datasheet number is not a field number.
- "What batch size?" A throughput number at batch 64 says nothing about a single-patient latency.
- "Which percentile?" A mean latency hides the tail that misses deadlines.
- "Which runtime version?" Kernel selection changes between releases.
- "What was the ambient temperature?" A cold bench is not an enclosure.
- "What fraction of peak is that?" A utilization figure without peak and layer size is uninterpretable.
- "What is the fallback path?" A model that quietly runs three operators on the CPU is a different model.

A team that can answer these eight questions has a performance model. A team that cannot has a slide.

### 1.3 What this topic is not

- It is not a guide to writing kernels. It teaches how to reason about whether a kernel can meet a budget, not how to implement one.
- It is not a benchmark report. No number here is a claim about a specific product; the worked example is constructed.
- It is not a substitute for target measurement. The roofline predicts a bound; only the board reports a latency.
- It is not a hardware selection guide. It gives the method for selection, not a recommendation among vendors.

The boundary matters because performance arguments often drift into vendor advocacy. The method is neutral: compute the intensity, compute the ridge, measure the achieved throughput, and let the result decide.

## 2. Learning outcomes

By the end of this topic you can:

1. **Count** parameters, MACs, and bytes moved for a convolutional or fully connected layer, stating the convention used.
2. **Compute** arithmetic intensity and place a layer on a roofline for a specified machine.
3. **Derive** the ridge point from peak throughput and peak bandwidth, and explain what it implies for a given layer.
4. **Measure** achieved bandwidth, achieved throughput, and sustained clock on a target board, and distinguish burst from sustained behavior.
5. **Attribute** latency to operators and identify fallback, launch overhead, and memory stalls.
6. **Predict** the latency effect of a compression method and state when the prediction should not be trusted.
7. **Choose** a hardware class from a constraint ledger rather than from a peak specification.
8. **Write** a profiling report that another engineer could reproduce on the same board.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | Correct counts with a stated convention |
| 2 | Exercise 2 | Roofline placement with units |
| 3 | Exercise 2 | Ridge calculation from peak and bandwidth |
| 4 | Exercise 3 | A bandwidth script and its limits |
| 5 | Exercise 4 | Time attribution and a lever |
| 6 | Exercise 5 | A negative result explained by bytes |
| 7 | Exercise 7 | A choice tied to the ledger |
| 8 | Exercise 8 | A reproducible report with assumptions marked |

## 3. The core mental model

The model has two roofs. Performance cannot exceed the compute roof, and it cannot exceed the bandwidth roof. Which roof applies depends on how much arithmetic the workload performs per byte it moves.

```text
   achieved
   MAC/s
     ▲
     │                        peak compute (flat roof)
     │            ┌───────────────────────────────
     │           /
     │          /   memory-bound      compute-bound
     │         /     (slope = bandwidth)
     │        /
     │       /
     │      /
     └─────┴──────────────────────────────────────► arithmetic intensity (MAC/byte)
           ▲
        ridge point = peak compute / peak bandwidth
```

The sloped line is the bandwidth roof: at intensity `I`, the fastest the machine can go is `bandwidth × I`. The flat line is the compute roof: `peak MAC/s`. They meet at the ridge point. To the left, adding compute does nothing; to the right, adding bandwidth does nothing.

### 3.1 The arithmetic

For a layer or a whole model:

```text
MACs        = number of multiply-accumulate operations
bytes       = bytes that must move from the memory level that holds the data
intensity I = MACs / bytes                      [MAC/byte]
ridge R     = peak_MAC_per_s / peak_bytes_per_s [MAC/byte]

t_compute   = MACs / (peak_MAC_per_s × utilization)
t_memory    = bytes / (peak_bytes_per_s × utilization)
t_layer     = max(t_compute, t_memory) + overhead
```

Two details decide whether the calculation is meaningful.

**Which memory level.** On a microcontroller, the bytes that matter are those crossing the bus from SRAM or flash. On a GPU, the bytes that matter are those crossing from high-bandwidth memory to the last-level cache; data resident in registers or shared memory does not count. The intensity of the same layer can differ by an order of magnitude depending on the level chosen. State the level.

**Which bytes.** For a convolution, the minimum traffic is input plus output plus weights, but an im2col implementation materializes an expanded input matrix whose size can exceed all three. A naive calculation uses the minimum and predicts a latency the implementation cannot achieve. Measure traffic with performance counters when the runtime exposes them; otherwise use the minimum and label the estimate optimistic.

### 3.2 Why the same layer changes sides

The same convolution is compute-bound on a microcontroller and memory-bound on a neural accelerator. This is not a paradox. The accelerator has far more arithmetic per byte of bandwidth, so its ridge point is much higher.

Consider a first convolution with 8 output channels, one input channel, a kernel of 5, and 1,024 input samples. It performs 40,800 MACs and moves about 9,256 bytes, for an intensity of 4.41 MAC per byte.

- A Cortex-M4F at 64 MHz with single-instruction multiple-data (SIMD) integer kernels sustains about 128 MMAC/s and has an SRAM bandwidth of about 256 MB/s. Its ridge point is 0.5 MAC per byte. The layer is far to the right of the ridge: compute-bound.
- An Arm Ethos-U85 configured with 512 MACs per cycle at 1 GHz sustains 512 GMAC/s, with an assumed 128-bit AXI interface at 16 GB/s. Its ridge point is 32 MAC per byte. The same layer is to the left of the ridge: memory-bound.
- An NVIDIA H100 SXM sustains about 989.5 TMAC/s of dense INT8 and has 3.35 TB/s of memory bandwidth. Its ridge point is about 295 MAC per byte. The layer is deeply memory-bound.

The consequence for compression is different on each machine. On the microcontroller, reducing arithmetic (distillation, structured pruning, cheaper operators) helps, while reducing weight bytes alone does not. On the accelerator, reducing bytes (quantization, fusion, layout) helps, while reducing MACs alone may not.

### 3.3 The memory hierarchy is the model

A neural network layer is a loop nest with a working set. Performance depends on whether that working set fits in the level of memory the loop can afford to revisit.

| Level | Cortex-M4F typical | Edge SoC typical | Latency | Implication |
|---|---|---|---|---|
| Registers | 16 × 32-bit | 32 × 64-bit per thread | 1 cycle | Kernel must keep accumulators here |
| SRAM or L1 | 64–256 KiB | 32–128 KiB per core | 1–4 cycles | Activations must fit or be tiled |
| L2 or shared | none | 0.5–4 MiB | 10–30 cycles | Weights may fit for small models |
| DRAM | none | 2–16 GiB | 100–400 cycles | Bandwidth-bound inference |
| Flash | 256 KiB–2 MiB | 32–512 GiB | 10–100+ cycles, wait states | Weights stored here; execution slower |
| External | SD, eMMC | UFS | milliseconds | Logs and updates only |

Two consequences follow, and both are frequently missed.

First, **flash is not SRAM**. A model can fit in flash and still fail, because the runtime must copy activations into SRAM and may copy weights tile by tile. On a microcontroller without a cache, executing directly from flash incurs wait states that can halve effective throughput. Some tools place hot weights in SRAM and cold weights in flash; the split is a performance decision, not a storage decision.

Second, **peak memory is not model size**. It includes activations, workspace, input and output buffers, the runtime, and allocator fragmentation. TensorFlow Lite for Microcontrollers (TFLM) documents a single tensor arena divided into a head for shared tensors, a temporary region, and a tail for persistent allocations, with a greedy memory planner deciding reuse. The arena size is a build-time constant that must be determined by experiment, and it is often several times the largest activation.

### 3.4 Peak, sustained, and achieved

Three numbers describe a processor's throughput, and they differ by large factors.

- **Peak** is the datasheet number: MACs per cycle times clocks per second times the number of arrays. It assumes perfect utilization, perfect memory residency, and no thermal limit.
- **Sustained** is what the part delivers after clocks settle under a thermal load. It is always below peak and can be far below in a sealed enclosure.
- **Achieved** is what your model gets after operator support, memory stalls, and framework overhead. It is the only number that predicts your latency.

The ratio of achieved to peak is utilization. A well-optimized convolution on a digital signal processor (DSP) or neural processing unit (NPU) might reach 40–70% of peak on a large layer. A small layer with launch overhead might reach 2%. Reporting a utilization figure without the layer size is meaningless.

### 3.5 Where the time actually goes

For a model with many layers, the total latency is not the sum of arithmetic times. It is the sum of:

- Arithmetic time per layer.
- Memory stall time per layer.
- Kernel launch or function call overhead per layer.
- Graph scheduling and tensor allocation.
- Partition boundaries and data copies when operators fall back.
- Preprocessing and postprocessing outside the model.

On a microcontroller, overhead per layer is small in absolute terms but large relative to a 50-microsecond layer. On a GPU or NPU, launch overhead of 5–20 microseconds can dominate a model whose arithmetic takes 200 microseconds. This is why batching improves GPU throughput and does nothing for a single-sample latency-critical device.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Parameter | A learned weight or bias | Counted as a MAC | Count |
| MAC | One multiply and one add | Counted as one FLOP | Operation |
| FLOP | One floating-point operation | Used interchangeably with MAC | Operation |
| Operation count | MACs, or 2 × MACs if counting FLOPs | Compared across conventions | Count |
| Arithmetic intensity | MACs per byte moved | Computed without stating the memory level | MAC/byte |
| Roofline | Bound of min(compute roof, bandwidth × intensity) | Treated as a latency predictor | Model |
| Ridge point | Intensity where the two roofs meet | Ignored when comparing machines | MAC/byte |
| Peak throughput | Datasheet MAC/s at a stated precision | Assumed achievable | MAC/s |
| Sustained throughput | Throughput after thermal and clock settling | Measured in a short burst | MAC/s |
| Achieved throughput | Throughput on the actual model | Reported as utilization without context | MAC/s |
| Utilization | Achieved divided by peak | Compared across layer sizes | Proportion |
| TOPS | Tera-operations per second; often counts a MAC as two operations | Compared across vendors without precision | 10^12 ops/s |
| GOPS | Giga-operations per second | Same caveat as TOPS | 10^9 ops/s |
| Memory bandwidth | Bytes per second the memory system can deliver | Quoted as peak without achieved measurement | byte/s |
| SRAM | Static random-access memory; fast, small, expensive | Confused with flash | KiB |
| DRAM | Dynamic random-access memory; large, slower | Assumed present on microcontrollers | GiB |
| Flash | Non-volatile storage for code and weights | Assumed to execute at full speed | KiB |
| Wait state | Extra cycles inserted when reading slower memory | Ignored in cycle estimates | cycles |
| Cache | Small fast memory holding recently used data | Assumed to hold the whole model | KiB |
| DMA | Direct memory access; moves data without the CPU | Confused with zero cost | Engine |
| SIMD | Single instruction, multiple data | Assumed available on all cores | Feature |
| MVE | M-Profile Vector Extension, also called Helium | Confused with NEON | Feature |
| NPU | Neural processing unit; fixed-function accelerator | Assumed to support all operators | Device |
| DSP | Digital signal processor | Assumed to run arbitrary graphs | Device |
| MCU | Microcontroller unit | Assumed to have an operating system | Device |
| SoC | System on chip | Treated as a single homogeneous processor | Device |
| MAC array | The parallel multiplier structure in an NPU | Confused with clock speed | Structure |
| Clock | Processor frequency | Assumed constant | Hz |
| DVFS | Dynamic voltage and frequency scaling | Ignored in latency predictions | Mechanism |
| TDP | Thermal design power | Treated as typical power | W |
| Thermal resistance | Temperature rise per watt dissipated | Ignored in enclosure design | °C/W |
| Throttling | Clock or voltage reduction to stay within thermal limits | Assumed not to occur | Mechanism |
| Batch | Inputs processed together | Omitted from latency claims | Count |
| Operator | A graph node such as convolution or softmax | Assumed supported by every backend | Node |
| Kernel | The implementation of an operator on a device | Confused with the operator | Function |
| Delegate | A backend that executes a subgraph on an accelerator | Assumed to cover the whole graph | Component |
| Fallback | Execution on a different processor when an operator is unsupported | Assumed to be free | Mechanism |
| Profiling | Measurement of where time and resources go | Replaced by intuition | Activity |
| Performance counter | Hardware register counting cycles, cache misses, stalls | Assumed present on all targets | Register |
| Cold start | First run after load, caches and clocks cold | Mixed into warm statistics | State |
| Warm start | Steady-state run with caches and clocks warm | Reported as the only number | State |
| p99 | 99th percentile | Assumed equal to the mean | Quantile |
| Jitter | Variation in latency across runs | Ignored in deadline analysis | ms |
| Deterministic | Same output and timing every run | Assumed for all accelerators | Property |

Two definitions carry more weight than the rest.

**TOPS is a marketing unit.** It is operations per second, and a MAC is two operations, so 1 TOPS equals 0.5 TMAC/s. Vendors sometimes quote sparse throughput, sometimes dense, sometimes a particular precision. A 4 TOPS INT8 number with sparsity enabled is 2 TOPS dense, which is 1 TMAC/s. Comparing two TOPS numbers without knowing precision, sparsity, and density is comparing nothing.

**Arithmetic intensity is a property of the workload and the memory level, not of the model.** The same layer has different intensity when weights are cached versus when they are streamed from DRAM. Any intensity figure must state what is counted.

### 4.1 Counting conventions

Operation counts differ by a factor of two depending on convention, and by more depending on what is included. State the convention in every table.

| Convention | Definition | Example: 40,800 MACs |
|---|---|---|
| MACs | Multiply-accumulate as one unit | 40,800 |
| FLOPs, MAC as two | Multiply and add counted separately | 81,600 |
| FLOPs, fused multiply-add as one | Some hardware counters | 40,800 |
| Operations including bias | MACs plus output adds | 40,800 + 8,160 |
| Operations including activation | Adds a comparison or clamp per output | 40,800 + 2 × 8,160 |

The difference matters when comparing with a vendor's TOPS figure, which uses operations, not MACs. A model described as "4 GFLOPs" may be 2 GMACs or 4 GMACs depending on who wrote the slide, and the hardware peak it is compared against may use either convention.

### 4.2 Typical intensity by layer type

The following values assume INT8 weights and activations, batch 1, and the minimum traffic of input plus output plus weights. They are starting points for a first estimate, not measurements.

| Layer | Shape | MACs | Bytes | Intensity (MAC/B) |
|---|---|---:|---:|---:|
| Fully connected | 1,024 → 1,024 | 1.05 M | 1.05 M weights + 2 KB | ~1.0 |
| Fully connected, batch 8 | 1,024 → 1,024 | 8.39 M | 1.05 M + 16 KB | ~7.9 |
| Conv 3×3, 64 channels, 56×56 | 64 → 64 | 115.6 M | 0.44 M | ~264 |
| Depthwise 3×3, 64 channels, 56×56 | 64 → 64 | 1.81 M | 0.40 M | ~4.5 |
| 1D conv, 8 channels, kernel 5, length 1,024 | 1 → 8 | 40,800 | 9,256 | 4.41 |
| Attention, sequence 256, d=768 | — | ~1.41 G | ~6 MB | ~235 |

Two patterns stand out. Fully connected layers at batch 1 are weight-bound and have an intensity near 1, which is below the ridge point of almost every machine. Depthwise convolutions have low intensity because they perform few MACs per byte. Standard convolutions have high intensity because each weight is reused across many output positions. This is why quantization helps fully connected layers and why depthwise layers are often memory-bound even on a microcontroller.

## 5. The mechanics, step by step

The procedure below produces a target-specific performance model. It is written for one model and one board; repeat it for each candidate in the frontier.

### Step 1. Pin the target configuration

**Input:** A board, a runtime build, a model file, and a test input.

**Output:** A configuration record: board revision, system-on-chip (SoC) part number, core clock, memory clock, cache configuration, runtime version, delegate version, compiler flags, and operating system or bare-metal setting.

**Rationale:** Every subsequent number is conditional on this record. A latency difference between two runs that differ in runtime version is not a property of the model, and without the record the difference cannot be attributed.

**Branch point:** If the runtime selects kernels at load time based on CPU features, record the selected kernel set. Two boards with the same part number but different firmware can select different kernels.

**Edge case:** Development boards often run at a higher clock than the production enclosure allows, because the enclosure restricts airflow. Measure on the production thermal design or apply a documented derating.

### Step 2. Count parameters and MACs analytically

**Input:** The model graph and input shape.

**Output:** Per-layer and total parameter count, MAC count, and output shape.

**Rationale:** The analytic count is the denominator of every efficiency claim. It is also a check on the profiler: if the measured operation count disagrees with the analytic count by more than a few percent, the graph has been transformed or the profiler is counting something else.

The formulas for common layers:

```text
Conv2D:      MACs = H_out × W_out × C_out × C_in × K_h × K_w / groups
Depthwise:   MACs = H_out × W_out × C_out × K_h × K_w
Linear:      MACs = in_features × out_features × batch
Attention:   MACs ≈ 4 × seq_len² × d_model + 8 × seq_len × d_model²   (per layer)
```

For a one-dimensional convolution with input length 1,024, one input channel, 8 output channels, and kernel 5:

```text
L_out = 1024 - 5 + 1 = 1020
MACs  = 1020 × 8 × 1 × 5 = 40,800
```

**Branch point:** If the graph contains dynamic shapes, count for the largest shape the device will see and for the expected shape. The difference is often the difference between fitting and not fitting.

**Edge case:** Fused operators change the count. A convolution fused with batch normalization performs the same MACs but fewer memory passes; a convolution fused with a residual add performs an extra add per output element that is easy to omit.

### Step 3. Measure peak memory bandwidth

**Input:** The target board and a streaming benchmark.

**Output:** Achieved copy bandwidth in bytes per second, for reads, writes, and mixed traffic.

**Rationale:** The bandwidth roof is only as good as the bandwidth number. Datasheet bandwidth is a theoretical maximum; achieved bandwidth on a microcontroller with wait states can be half of it, and on a GPU it depends on access pattern.

A minimal streaming test copies a buffer larger than the last-level cache and reports bytes moved divided by time. Run it at the production clock and at the production memory configuration.

**Branch point:** If read and write bandwidth differ materially, use the mixed figure for layers that stream both, and the read figure for weight-only streaming.

**Edge case:** Caches inflate small-buffer bandwidth. A 4 KiB copy that fits in L1 will report a bandwidth the model cannot achieve when it streams 400 KiB.

### Step 4. Compute the ridge point

**Input:** Peak MAC throughput at the model's precision and achieved bandwidth.

**Output:** Ridge point in MAC per byte, and the memory-bound and compute-bound regions.

**Rationale:** The ridge point converts the roofline from a picture into a comparison. A layer's intensity either exceeds the ridge or it does not, and that determines which compression methods can help.

```text
ridge = peak_MAC_per_s / bandwidth_bytes_per_s
```

Examples at the stated assumptions:

| Machine | Peak (MAC/s) | Bandwidth | Ridge (MAC/byte) |
|---|---:|---:|---:|
| Cortex-M4F, 64 MHz, SIMD INT8 | 128 M | 256 MB/s | 0.5 |
| Ethos-U85-512, 1 GHz | 512 G | 16 GB/s | 32 |
| H100 SXM, dense INT8 | 989.5 T | 3.35 TB/s | 295 |

**Branch point:** If the layer intensity is within a factor of two of the ridge, treat the classification as uncertain. Real implementations have overheads that can move the operating point across the ridge.

**Edge case:** Sparse hardware changes the peak. An Ethos-U85 with 2:4 sparsity can double its effective MAC rate for supported layers, which doubles the ridge point. Only apply the sparse peak to layers that are actually sparse and actually use the sparse path.

### Step 5. Measure realized latency and effective throughput

**Input:** The model, the board, and a representative input.

**Output:** p50, p95, and p99 latency; effective MAC/s; utilization.

**Rationale:** The ratio between effective and peak throughput is the correction factor that the roofline lacks. It captures kernel quality, memory stalls, launch overhead, and graph scheduling.

```text
effective_MAC_per_s = total_MACs / latency_s
utilization         = effective_MAC_per_s / peak_MAC_per_s
```

**Branch point:** If utilization is below about 10%, the model is overhead-dominated. The next step is operator attribution, not more arithmetic analysis.

**Edge case:** A model with a single large layer and a model with fifty small layers can have the same MAC count and differ by an order of magnitude in latency. Utilization without a layer count is not interpretable.

### Step 6. Attribute time to operators

**Input:** A profiler trace or an instrumented runtime.

**Output:** A table of operator, time, share of total, and whether it ran on the intended processor.

**Rationale:** Compression effort should follow time, not parameter count. A layer with 2% of the parameters can consume 30% of the latency if it is unsupported and falls back.

**Branch point:** The attribution usually falls into one of four patterns.

| Pattern | Signature | Interpretation | Next action |
|---|---|---|---|
| One operator dominates | > 40% in one node | Genuine hotspot | Optimize or replace that operator |
| Many small operators | Mean < 50 µs, count > 40 | Launch overhead | Fuse, or use a graph compiler |
| Fallback cluster | Unsupported ops on CPU | Partitioning | Replace ops or change backend |
| Uniformly slow | No hotspot, low utilization | Memory or clock | Check bandwidth and sustained clock |

**Edge case:** Profilers that instrument every node add overhead that changes the relative shares. Use sampling or hardware counters where possible, and cross-check total instrumented time against uninstrumented latency.

### Step 7. Place each layer on the roofline

**Input:** Per-layer MACs, bytes, and measured time.

**Output:** A per-layer classification: compute-bound, memory-bound, or overhead-bound.

**Rationale:** The classification tells you which lever moves the layer. It also predicts which compression methods will fail, which is more valuable than predicting which will succeed.

```text
intensity_layer = MACs_layer / bytes_layer
if intensity_layer > 2 × ridge:  compute-bound
if intensity_layer < 0.5 × ridge: memory-bound
otherwise:                        mixed
```

**Branch point:** If a layer is memory-bound and its bytes are dominated by activations, weight quantization will not help. If its bytes are dominated by weights, weight quantization will help roughly in proportion to the byte reduction.

**Edge case:** A layer can be compute-bound in isolation and memory-bound in a fused graph, because fusion changes which intermediate tensors are written. Recompute the classification after graph optimization.

### Step 8. Measure sustained behavior and thermal margin

**Input:** A soak test at the specified ambient temperature, with the production enclosure.

**Output:** Latency and clock over time, time to throttle, steady-state latency, and skin or case temperature.

**Rationale:** The field constraint is sustained performance. A device that meets its deadline for 30 seconds and misses it after five minutes has not met its deadline.

Run for at least 30 minutes, ideally 24–72 hours, and record clock, temperature, and latency at one-second resolution. Report the time at which the p99 first exceeds the cold-start p99 by more than the tolerance.

**Branch point:** If throttling occurs before the clinical duty cycle ends, the design must either reduce sustained power, improve the thermal path, or relax the latency requirement. Software cannot fix a thermal design.

**Edge case:** A device that is idle most of the time may never throttle in the field even if it throttles in a soak test. Match the soak duty cycle to the clinical duty cycle; a continuous-load test is a worst case, not a prediction.

### Step 9. Choose the lever and predict the gain

**Input:** The binding resource per layer and the candidate compression methods.

**Output:** A predicted latency change with a stated mechanism and confidence.

**Rationale:** A prediction forces the team to name the mechanism. "Quantization will make it faster" is not a prediction; "the layer is weight-byte-bound, weights fall by 4×, so bytes fall by about 30% and latency should fall by at most 30%" is.

| Binding resource | Lever | Expected effect | When it fails |
|---|---|---|---|
| Compute, dense layers | Distillation, structured pruning | Proportional to MAC reduction | Capacity gap; accuracy floor |
| Compute, convolution | Cheaper operator, lower precision | Up to 2–4× on integer SIMD | Unsupported precision; accuracy |
| Weight bytes | Weight quantization | Up to 4× on weight traffic | Activations dominate |
| Activation bytes | Activation quantization, fusion | Up to 4× on activation traffic | Outliers force clipping |
| Launch overhead | Fusion, graph compilation | Large for many small ops | Unsupported fused operator |
| Fallback | Operator replacement | Removes partition and copy | No equivalent operator |
| Thermal | Lower sustained power | Restores clock | Enclosure limits |
| Clock | Higher clock or better cooling | Linear until thermal limit | Battery and heat |

**Branch point:** If the predicted gain does not survive a measured prototype, the model of the bottleneck is wrong. Return to Step 6 rather than trying a second method on top of the first.

**Edge case:** Two levers can interact. Quantizing weights and activations together reduces both traffic terms, but the combined gain is not the product of the individual gains if one term was already negligible.

### Step 10. Record the evidence and the uncertainty

**Input:** All measurements and predictions.

**Output:** A profiling report with configuration, method, results, uncertainty, and reproduction steps.

**Rationale:** The report is what a reviewer, a colleague, or a future you will use. Without the configuration and the method, the numbers are anecdotes.

The report contains:

- The configuration record from Step 1.
- The analytic counts and the counting convention.
- The bandwidth measurement and its method.
- The ridge calculation with units.
- Per-layer time and classification.
- Cold and sustained latency at p50, p95, and p99.
- Energy per inference and the measurement method.
- The binding resource and the chosen lever.
- The predicted gain and the measured gain, with the difference explained.

**Branch point:** If the predicted and measured gains differ by more than 20%, do not ship the prediction. Either the mechanism was wrong or the measurement is confounded, and both are worth knowing before the design freeze.

**Edge case:** A measurement taken on a pre-production board with a different memory part is not evidence for the production board. Record the part numbers and treat the result as provisional.

### 5.1 Decision table for the profiling result

| Observation | Most likely cause | Next test | Action |
|---|---|---|---|
| Achieved < 10% of peak | Overhead or fallback | Operator trace | Fuse or replace operators |
| Achieved 10–40% of peak | Memory stalls or small layers | Bandwidth and intensity | Quantize activations or fuse |
| Achieved > 40% of peak | Genuine compute limit | Roofline placement | Distill or prune |
| p99 much greater than p50 | Scheduling or thermal | Time series of latency | Soak test; watchdog analysis |
| Latency doubles after 10 minutes | Throttling | Clock and temperature log | Reduce sustained power |
| One layer is 50% of time | Hotspot | Per-layer intensity | Optimize that layer |
| Memory allocation fails | Peak memory underestimated | Arena high-water mark | Reduce activations or tile |
| Model fits flash but not SRAM | Working set too large | Peak activation size | Smaller window or fusion |
| Energy per inference higher than model | Radio or sensor | Full-cycle power trace | Move decision on-device |
| Same model, different board, different latency | Clock, memory, or kernel | Configuration diff | Re-baseline per board |

### 5.2 Edge cases in performance measurement

**Outlier channels.** A single channel with a large activation range forces a wide quantization scale, which can reduce the effective precision of every other channel. The performance symptom is subtle: the model runs at the same speed but accuracy drops, and a team may respond by adding compute rather than fixing the scale. Diagnose with the step-to-median ratio before changing the architecture.

**Per-tensor versus per-channel scales.** Per-channel scales add metadata and, on some kernels, a per-channel multiply that prevents a fused integer path. The performance effect is usually small, but on a microcontroller with tight SRAM the metadata can matter for very small tensors. Measure both if the runtime offers both.

**Activation clipping.** Clipping reduces the dynamic range and can allow a smaller scale and faster integer path, but it changes outputs. The performance gain from clipping alone is usually zero on hardware that supports the full range; the gain comes from enabling a lower bit width, which must be evaluated for accuracy.

**Operator fallback.** A fallback splits the graph, inserts a copy, and may force synchronization. The cost is not the operator's arithmetic; it is the boundary. A single unsupported activation function in the middle of a network can cost more than the convolution it follows.

**Unsupported operators.** Some operators have no quantized kernel at all, not even on the CPU. The runtime may refuse to load the model, or may silently run a float fallback that is both slow and numerically different. Check the runtime's operator list for the exact opset and precision.

**Zero-point drift.** A quantized model calibrated on one distribution can saturate after a shift. Saturation does not change latency, so performance monitoring will not detect it. Monitor the saturation rate as a separate signal.

**Calibration-set bias.** A calibration set drawn from one sensor or site produces scales that are wrong for others. The performance effect is nil; the accuracy effect can be large and subgroup-specific. This is an accuracy failure that a performance dashboard cannot see.

**Thermal throttling.** The most common cause of a latency regression between the bench and the field. It is also the easiest to miss, because a 30-second benchmark never reaches thermal steady state.

### 5.3 Common measurement traps

The following traps account for most of the disagreement between predicted and measured performance. Each has a simple test.

1. **Timing only the kernel.** The measurement excludes tensor allocation, graph scheduling, and the copy into the input buffer.
   - Test: time the full call from input-ready to output-ready.
   - Typical gap: 10–40% on an MCU, several hundred percent for small layers on a GPU.

2. **Warming the cache without saying so.** A loop over the same input leaves weights and activations resident, so the measured time reflects a cache that the first real inference will not have.
   - Test: report first-call and steady-state latency separately.
   - Typical gap: 2–10× for models whose weights fit in L2.

3. **Using a fixed input that is unrepresentative.** A tensor of zeros can take a different branch, produce different sparsity, and avoid the slow path through activation functions.
   - Test: use real sensor data with the clinical distribution, including its tails.
   - Typical gap: unpredictable; sometimes faster, sometimes slower.

4. **Ignoring the first inference after a model update.** The runtime may compile kernels, allocate memory, or page weights on first use.
   - Test: measure the update-to-first-output path as a separate requirement.
   - Typical gap: seconds on an applications processor; milliseconds on an MCU.

5. **Comparing across runtime versions.** A new release can change kernel selection, fusion, or memory planning.
   - Test: pin the runtime and re-baseline after every upgrade.
   - Typical gap: 5–20%.

6. **Benchmarking on a development board with a debugger attached.** The debug interface can slow the core and prevent low-power states.
   - Test: measure on a release build with the debugger detached.
   - Typical gap: 5–30%.

7. **Reporting a mean over a bimodal distribution.** Some inputs hit a fallback path and others do not, producing two clusters whose mean describes neither.
   - Test: plot the histogram before reporting a summary statistic.
   - Typical gap: the mean can sit between the modes and match neither.

8. **Assuming the model is the only workload.** The radio, the display, and the sensor front end share power and buses with the processor.
   - Test: measure in the full pipeline with realistic concurrency.
   - Typical gap: 10–50% on latency, larger on energy.

### 5.4 A prediction that fails, and why

A team estimates a 4× speedup from INT8 quantization on a Cortex-M4F because the weights shrink by 4×. The layer is compute-bound: weights are 40 of 9,256 bytes. The measured speedup is 1.0× for latency and 1.9× for energy, because the integer kernels are more efficient per MAC than the float kernels but the arithmetic count is unchanged.

The prediction failed because the model of the bottleneck was wrong, not because the hardware misbehaved. The correct prediction would have started from the classification in Step 7: compute-bound, so the lever is arithmetic, not bytes. A second team, working on a fully connected layer with intensity 1.0 on the same board, measures a 3.6× speedup from the same conversion, because that layer is weight-bound. Same method, different bottleneck, different result.

This is the practical reason to compute intensity before choosing a method. The method is not good or bad in general; it is good or bad for a resource.

### 5.5 Units and significant figures

Performance arithmetic is easy to get wrong by a factor of 1,000. The following conventions prevent most of the errors.

| Quantity | Unit | Conversion trap |
|---|---|---|
| MACs | count | 1 MMAC = 10^6; 1 GMAC = 10^9 |
| Throughput | MAC/s | 128 MMAC/s = 0.128 GMAC/s |
| TOPS | 10^12 ops/s | 1 TOPS = 0.5 TMAC/s |
| Bandwidth | byte/s | 256 MB/s = 0.256 GB/s, decimal |
| Memory | byte | KiB is 1,024 B; vendors sometimes use KB for 1,000 B |
| Latency | s | 0.319 ms = 319 µs = 319,000 ns |
| Energy | J | 1.132 mJ = 1,132 µJ |
| Power | W | 8.2 mA at 3.0 V = 24.6 mW |

Three rules follow.

- **Convert to base units before dividing.** A latency in milliseconds divided by a throughput in MAC/s gives a result that is wrong by 1,000 and looks plausible.
- **Carry units through the arithmetic.** If the result's unit is not the one expected, the calculation is wrong.
- **Round at the end, not in the middle.** A scale rounded to four decimals in the middle of a quantization chain can change the reconstructed value by more than the quantization error.

A worked review that reports 4.1 MMAC, 89.6 MMAC/s, and 45.7 ms is internally consistent: 4.1e6 / 89.6e6 = 0.0458 s. The same numbers reported as 4.1 MMAC, 89.6 GMAC/s, and 45.7 ms are not, and the inconsistency is the signal that a unit conversion was dropped.

### 5.6 Reproducibility checklist

Before a performance result leaves the team, confirm each item. A missing item is a defect in the evidence, not a detail to fill in later.

- The board part number, revision, and serial number are recorded.
- The core clock, memory clock, and governor setting are recorded.
- The runtime, delegate, and toolchain versions are recorded.
- The model file hash is recorded.
- The input tensors are real sensor data, not zeros or random noise.
- The batch size is stated, and batch 1 is reported separately from larger batches.
- Cold start and warm start are reported separately.
- p50, p95, and p99 are reported with the sample count.
- The ambient temperature and the enclosure are described.
- The measurement harness is under version control.
- The raw samples are retained, not only the summary.
- The analytic MAC count and the profiler's count are compared.

A colleague who can take these twelve items and rerun the measurement has evidence. A colleague who cannot has a number.

## 6. Worked example

This is a constructed case with arithmetic that closes. The hardware specifications are taken from vendor documentation or stated as assumptions; the model and its measurements are illustrative, not a benchmark result.

### 6.1 The layer and the three machines

The layer is the first convolution of the ECG model from Topic 01: 8 output channels, 1 input channel, kernel 5, input length 1,024, valid padding, stride 1.

```text
Output length = 1024 - 5 + 1 = 1020
MACs          = 1020 × 8 × 1 × 5 = 40,800
Parameters    = 8 × 1 × 5 = 40 weights, plus 8 biases
```

Bytes moved at INT8, counting input, output, weights, and INT32 bias:

```text
input   = 1,024 × 1 byte  = 1,024 B
output  = 1,020 × 8 bytes = 8,160 B
weights = 40 × 1 byte     =    40 B
bias    = 8 × 4 bytes     =    32 B
total   = 9,256 B
```

Arithmetic intensity:

```text
I = 40,800 / 9,256 = 4.408 MAC/byte = 8.816 FLOP/byte
```

### 6.2 Cortex-M4F at 64 MHz

Assumptions: one 32-bit MAC per cycle scalar, two per cycle with the SIMD instructions CMSIS-NN uses for INT8, SRAM delivering 4 bytes per cycle, and 70% achieved utilization on this layer.

```text
peak MAC/s   = 64,000,000 cycles/s × 2 MAC/cycle = 128,000,000 MAC/s
bandwidth    = 64,000,000 cycles/s × 4 B/cycle   = 256,000,000 B/s
ridge        = 128,000,000 / 256,000,000         = 0.5 MAC/byte

t_compute    = 40,800 / 128,000,000 = 0.000319 s = 0.319 ms
t_memory     =  9,256 / 256,000,000 = 0.000036 s = 0.036 ms
t_bound      = max(0.319, 0.036)                 = 0.319 ms
```

The layer is compute-bound by a factor of 8.8. At 70% utilization the measured time is 0.319 / 0.70 = 0.455 ms.

The same result in cycles, which is the form a firmware engineer uses:

```text
compute cycles = 40,800 MACs / 2 MAC/cycle = 20,400 cycles
memory cycles  = 9,256 B / 4 B/cycle       =  2,314 cycles
ratio          = 8.82, matching the intensity-to-ridge ratio

time at 64 MHz = 20,400 / 64,000,000 = 0.319 ms
time at 70%    = 20,400 / (0.70 × 64,000,000) = 0.455 ms
```

The cycle form also exposes the optimization target. If the kernel achieved 2 MACs per cycle on 70% of cycles, it would take 0.455 ms. To reach 0.319 ms it would need to eliminate the stall cycles, which in practice means keeping the input window and accumulators in registers and avoiding repeated loads from SRAM.

Energy for the layer at 3.0 V and 8.2 mA:

```text
E = 3.0 V × 0.0082 A × 0.000455 s = 11.2 µJ
```

The whole model performs about 4.1 MMAC. At the same 70% utilization (89.6 MMAC/s effective) its inference time is 4,096,000 / 89,600,000 = 45.7 ms, which matches the 46 ms p99 used in Topic 01. At 8.2 mA and 3.0 V, energy is 3.0 × 0.0082 × 0.046 = 1.132 mJ.

### 6.3 Ethos-U85 with 512 MACs per cycle

Arm documents the Ethos-U85 as configurable from 128 to 2,048 MACs per cycle in 8×8 arrays, with the 512-MAC configuration rated at 1 TOP/s at 1 GHz, 48 KiB of internal memory, INT8 weights with INT8 or INT16 activations, TOSA compliance, and optional 2:4 structured sparsity. One TOP/s is 0.5 TMAC/s, because an operation counts a MAC as two.

```text
peak MAC/s = 512,000,000
bandwidth  = 16,000,000,000 B/s   (assumed 128-bit AXI at 1 GHz)
ridge      = 512,000,000 / 16,000,000,000 = 32 MAC/byte

t_compute  = 40,800 / 512,000,000 = 0.0797 µs
t_memory   =  9,256 / 16,000,000,000 = 0.5785 µs
t_bound    = 0.5785 µs, memory-bound
```

The NPU is 5.5 times faster than the microcontroller on this layer in the arithmetic sense, but it is limited by memory. At 30% achieved bandwidth the layer takes 0.5785 / 0.30 = 1.93 µs. The NPU's invocation overhead for a single layer, typically several microseconds, would dominate; this is why NPUs are used on subgraphs, not on individual layers.

Energy on the NPU is not published as a TOPS/W figure in the documentation reviewed. Assuming 1 TOP/W for illustration, the layer at 1.93 µs and 0.5 W would consume about 0.97 µJ, roughly one tenth of the microcontroller's 11.2 µJ. That is an assumption, not a measurement.

### 6.4 H100 SXM as a contrast

The H100 datasheet lists 3,958 TOPS INT8 with sparsity and 3.35 TB/s of HBM3 bandwidth, at up to 700 W. Dense INT8 is half the sparse figure, 1,979 TOPS, or 989.5 TMAC/s.

```text
ridge     = 989.5e12 / 3.35e12 = 295.4 MAC/byte
I         = 4.408 MAC/byte, far below the ridge
t_memory  = 9,256 / 3.35e12 = 2.76 ns
t_compute = 40,800 / 989.5e12 = 0.041 ns
```

The layer is memory-bound and would take about 2.76 ns of memory time in a perfect kernel. A real kernel launch costs microseconds, so the layer would be overhead-bound in practice. This is the clearest illustration of the system fallacy: the roofline predicts the kernel, not the program.

### 6.5 Summary across the three machines

| Machine | Peak MAC/s | Ridge (MAC/B) | Layer bound | Predicted time | Energy |
|---|---:|---:|---|---:|---:|
| Cortex-M4F, 64 MHz | 128 M | 0.5 | Compute | 0.455 ms at 70% | 11.2 µJ measured-equivalent |
| Ethos-U85-512, 1 GHz | 512 G | 32 | Memory | 1.93 µs at 30% BW | ~0.97 µJ assumed |
| H100 SXM, dense INT8 | 989.5 T | 295 | Memory | 2.76 ns kernel, µs launch | not applicable at this scale |

### 6.6 Quantization parameters for the same layer

The 40 weights have a minimum of -0.061 and a maximum of 0.412, with a median absolute value of 0.0182. The affine derivation at 8 bits:

```text
S  = (0.412 - (-0.061)) / 255 = 0.473 / 255 = 0.00185490
Z  = round(0.061 / 0.00185490) = round(32.886) = 33
q(0.412) = round(222.11) + 33 = 255
q(-0.061) = round(-32.886) + 33 = 0
```

The symmetric derivation:

```text
S = 0.412 / 127 = 0.00324409
Z = 0
```

Measured on the layer, affine gives 42.04 dB signal-to-quantization-noise ratio (SQNR) and symmetric gives 37.54 dB. The affine step is 0.001855; the symmetric step is 0.003244.

Weight memory before and after, weights only:

```text
FP32: 40 × 4 = 160 B
INT8: 40 + 4 (scale) + 1 (zero point) = 45 B
ratio = 3.56×
```

The full model at 41,216 parameters: FP32 164,864 B (162.0 KiB); INT8 per-channel 41,216 B plus 600 B of metadata = 41,816 B (40.8 KiB); ratio 3.94×.

### 6.7 Diagnostics

**Diagnostic one: outlier-driven scale collapse.** The symmetric step is 0.003244, which is 17.8% of the median absolute weight. One weight in 40 rounds to zero, and only 24 of 255 levels are used. The layer SQNR of 37.54 dB is dominated by the outlier's energy; the non-outlier weights are quantized coarsely. The performance consequence is nil on this layer because weight bytes are only 40 of 9,256, but the accuracy consequence is real. The diagnostic is the pair (step / median absolute weight, distinct levels used), not the aggregate SQNR.

**Diagnostic two: per-channel skew.** The eight output channels have ranges from 0.0407 to 0.4703, a ratio of 11.6. Under a per-tensor scale, the smallest channel gets about 12 levels and the outlier channel gets 145. Per-channel quantization raises SQNR from 37.54 to 51.37 dB, a gain of 13.83 dB, at a metadata cost of 40 bytes for this layer. The performance cost is 40 bytes, which is 0.43% of the layer's traffic. The diagnostic is the channel range ratio; a value above 4 predicts a material per-tensor loss.

### 6.8 Varying one assumption

**Bit width: 8 to 4, weight-only.** The weights fall from 40 bytes to 20 bytes. Total layer bytes fall from 9,256 to 9,236, a change of 0.22%. Intensity rises from 4.408 to 4.417. Latency on the M4F is unchanged because the layer is compute-bound and the arithmetic count has not changed. This is the most important negative result in the topic: weight-only quantization does not speed up an activation-bound or compute-bound layer. It reduces memory footprint, which matters for flash and for weight-streaming layers, not for this one.

**Batch size: 1 to 8.** Bytes become 8,192 input + 65,280 output + 40 weights + 32 bias = 73,544; MACs become 326,400; intensity becomes 4.438. The intensity is essentially unchanged because activations dominate. Batching does not move this layer across the ridge. It would move a fully connected layer with a large weight matrix, where weights dominate at batch 1.

**Clock: 64 MHz to 32 MHz.** Peak compute falls to 64 MMAC/s and bandwidth to 128 MB/s. The ridge stays at 0.5 MAC per byte, because both scale together. The compute time doubles to 0.638 ms, and energy at the same current rises because the layer runs longer. A lower clock does not change the classification; it changes the time.

**Calibration set: 512 windows to 64.** The weight quantization parameters do not change, because the weights are fixed. The activation scales, which are set from calibration data, would shrink by about 6% and increase clipping. Latency is unchanged; accuracy falls by about 0.4 percentage points in the constructed case. This is an accuracy assumption, not a performance one, and it should be varied in the accuracy analysis rather than the performance analysis.

### 6.9 What the case does not show

The case is deliberately narrow, and four things it omits would change the conclusions.

- **The rest of the network.** The first convolution is 40,800 of about 4.1 million MACs, roughly 1%. Its share of latency may be larger or smaller depending on how the runtime implements it. A layer-level conclusion is not a model-level conclusion.
- **Kernel quality.** The 70% utilization is an assumption. CMSIS-NN on a Cortex-M4F can do better on large convolutions and worse on small ones, and the achieved figure depends on the layout and the input length.
- **Memory contention.** The case assumes the layer has the bus to itself. In a real device the direct memory access engine, the radio, and the sensor front end compete for the same SRAM.
- **Thermal state.** The case assumes a constant 64 MHz clock. In a sealed device at 40 °C ambient, the clock may be lower for part of the duty cycle, and the energy per inference rises because the work takes longer.

A reviewer should read the case as a method demonstration, not as a performance claim for a product.

## 7. Code, query, or calculation

The script counts operations, computes roofline positions, quantizes the layer, and reports memory and error metrics. It uses only the standard library and reproduces Section 6.

```python
import math
import random


def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def conv1d_macs(length, c_in, c_out, kernel):
    return (length - kernel + 1) * c_out * c_in * kernel


def roofline(macs, nbytes, peak_mac_s, bandwidth_b_s):
    t_compute = macs / peak_mac_s
    t_memory = nbytes / bandwidth_b_s
    return {
        "intensity": macs / nbytes,
        "ridge": peak_mac_s / bandwidth_b_s,
        "t_compute_s": t_compute,
        "t_memory_s": t_memory,
        "t_bound_s": max(t_compute, t_memory),
        "bound": "compute" if t_compute >= t_memory else "memory",
    }
```

Count and place the layer:

```python
length, c_in, c_out, kernel = 1024, 1, 8, 5
macs = conv1d_macs(length, c_in, c_out, kernel)
nbytes = (length * c_in
          + (length - kernel + 1) * c_out
          + c_out * c_in * kernel
          + c_out * 4)
print("MACs =", macs, "bytes =", nbytes)
print("intensity = %.3f MAC/byte" % (macs / nbytes))

m4 = roofline(macs, nbytes, 128e6, 256e6)
print("M4 ridge = %.2f  bound = %s  t = %.4f ms"
      % (m4["ridge"], m4["bound"], m4["t_bound_s"] * 1000))
print("M4 at 70%% utilization = %.4f ms" % (m4["t_compute_s"] * 1000 / 0.70))
print("M4 layer energy = %.2f uJ" % (3.0 * 0.0082 * m4["t_compute_s"] / 0.70 * 1e6))

u85 = roofline(macs, nbytes, 512e9, 16e9)
print("U85 ridge = %.1f  bound = %s  t = %.4f us"
      % (u85["ridge"], u85["bound"], u85["t_bound_s"] * 1e6))

h100 = roofline(macs, nbytes, 989.5e12, 3.35e12)
print("H100 ridge = %.1f  bound = %s  t = %.3f ns"
      % (h100["ridge"], h100["bound"], h100["t_bound_s"] * 1e9))

print("whole model on M4 = %.2f ms" % (4_096_000 / 89.6e6 * 1000))
```

Expected output:

```text
MACs = 40800 bytes = 9256
intensity = 4.408 MAC/byte
M4 ridge = 0.50  bound = compute  t = 0.3188 ms
M4 at 70% utilization = 0.4554 ms
M4 layer energy = 11.20 uJ
U85 ridge = 32.0  bound = memory  t = 0.5785 us
H100 ridge = 295.4  bound = memory  t = 2.763 ns
whole model on M4 = 45.71 ms
```

Variations:

```python
batch = 8
nb8 = (length * c_in * batch
       + (length - kernel + 1) * c_out * batch
       + c_out * c_in * kernel + c_out * 4)
print("batch 8 intensity = %.3f" % (macs * batch / nb8))

nb4 = (length * c_in + (length - kernel + 1) * c_out
       + c_out * c_in * kernel * 0.5 + c_out * 4)
print("int4 weight bytes = %d  intensity = %.3f" % (nb4, macs / nb4))

m4_32 = roofline(macs, nbytes, 64e6, 128e6)
print("32 MHz ridge = %.2f  t = %.4f ms"
      % (m4_32["ridge"], m4_32["t_compute_s"] * 1000))
```

Expected output:

```text
batch 8 intensity = 4.438
int4 weight bytes = 9236  intensity = 4.417
32 MHz ridge = 0.50  t = 0.6375 ms
```

Quantization, memory, and error metrics:

```python
def symmetric_scale(values, bits=8):
    amax = max(abs(v) for v in values)
    return amax / (2 ** (bits - 1) - 1)


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    scale = (hi - lo) / (2 ** bits - 1)
    zero_point = min(2 ** bits - 1, max(0, round_half_away(-lo / scale)))
    return scale, zero_point


def quantize(values, scale, zero_point=0, bits=8, symmetric=False):
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
    else:
        qmin, qmax = 0, 2 ** bits - 1
    return [min(qmax, max(qmin, round_half_away(v / scale) + zero_point))
            for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


def error_metrics(original, reconstructed):
    n = len(original)
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / n
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    return {"mae": sum(abs(a - b) for a, b in zip(original, reconstructed)) / n,
            "rmse": math.sqrt(mse),
            "sqnr_db": 10 * math.log10(signal / noise) if noise else float("inf")}


random.seed(7)
weights = [round(random.gauss(0, 0.035), 4) for _ in range(39)]
weights.insert(11, 0.412)

s = symmetric_scale(weights, 8)
qs = quantize(weights, s, bits=8, symmetric=True)
print("symmetric scale = %.8f" % s)
print("symmetric =", {k: round(v, 4) for k, v in error_metrics(weights, dequantize(qs, s)).items()})

sa, zp = affine_params(weights, 8)
qa = quantize(weights, sa, zp, 8)
print("affine scale = %.8f zero point = %d" % (sa, zp))
print("affine =", {k: round(v, 4) for k, v in error_metrics(weights, dequantize(qa, sa, zp)).items()})

print("weights only: fp32 = %d B, int8 = %d B"
      % (len(weights) * 4, len(weights) + 5))
print("model: fp32 = %d B, int8 per-channel = %d B"
      % (41216 * 4, 41216 + 120 * 5))
```

Expected output:

```text
symmetric scale = 0.00324409
symmetric = {'mae': 0.0008, 'rmse': 0.0009, 'sqnr_db': 37.5354}
affine scale = 0.00185490 zero point = 33
affine = {'mae': 0.0005, 'rmse': 0.0006, 'sqnr_db': 42.0374}
weights only: fp32 = 160 B, int8 = 45 B
model: fp32 = 164864 B, int8 per-channel = 41816 B
```

Assumptions in the script: peak throughput is a per-cycle MAC count times clock, bandwidth is bytes per cycle times clock, utilization is applied to the binding term only, bias is INT32, and metadata is 5 bytes per channel. The energy figure uses board current, not core current, and therefore includes regulators and memory.

## 8. Data, standards, and regulatory context

Performance measurement is not a regulated activity in the sense that a clinical endpoint is, but the evidence it produces is used in regulated decisions. This section separates the standards that constrain the evidence from the conventions that make it comparable.

### 8.1 IEC 62304 and performance requirements

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), requires that software requirements be specified, that architecture and detailed design be documented, and that verification demonstrate the requirements are met. It is recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79.

For performance, the consequences are specific:

- A latency requirement must be written as a requirement with a value, a percentile, a hardware configuration, and a measurement method. "Fast enough" is not a requirement.
- The measurement procedure is a verification activity and must be under configuration management.
- The runtime and model binary are configuration items; a change in either invalidates the verification until re-run.
- Safety class A, B, or C determines the rigour, but not the need for a requirement.

A profiling report that omits the runtime version fails the configuration-management expectation even if the numbers are correct.

### 8.2 IEC 60601-1 and essential performance

IEC 60601-1, the general standard for basic safety and essential performance of medical electrical equipment, introduces **essential performance**: the performance necessary to maintain risk within acceptable limits. If a latency overrun can cause a hazardous situation, latency is part of essential performance, and the verification must show that it holds under the specified conditions, including electromagnetic disturbance and power supply variation.

The practical implication is that a latency measured only on a bench supply at 22 °C is insufficient for a device whose essential performance depends on timing. The test must include the environmental and electrical conditions the standard requires for the device's class.

### 8.3 ISO 14971 and the risk of a performance miss

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and risks controlled. A latency overrun is a hazard when it delays an alarm, a control decision, or a therapy adjustment. The risk file should record:

- The hazardous situation (for example, an arrhythmia alarm delivered after the clinical window).
- The foreseeable sequence of events (thermal throttling, fallback, scheduler delay).
- The control (sustained-load testing, watchdog, fallback path, derating).
- The residual risk and its acceptability.

Compression changes the performance envelope and therefore the risk file. That is why performance evidence is regulatory evidence, not only engineering evidence.

### 8.4 Benchmark conventions

MLPerf Inference, maintained by MLCommons, is the closest thing to a cross-vendor convention for reporting latency and throughput. It defines scenarios (SingleStream, MultiStream, Server, Offline), quality targets, and latency constraints. The v6.1 round, published in September 2026, reports 30 submitters and 120 systems, with p99 constraints on time to first token and time per output token for language-model benchmarks.

MLPerf Tiny targets microcontroller-class devices. The v1.2 results, published in April 2024, include 91 performance results and 18 energy results, with benchmarks for keyword spotting, anomaly detection, visual wake words, and image classification, and optional energy reported in microjoules per inference.

These benchmarks are conventions, not standards. They constrain nothing. What they provide is a vocabulary that makes comparisons possible: scenario, quality target, percentile, and energy per inference. Use the vocabulary and state the scenario; do not transfer the numbers.

### 8.5 What the standards do not say

- No standard mandates a specific percentile. The clinical deadline and the harm model determine whether p99 or p999 is appropriate.
- No standard defines "real-time" for a medical device. A soft real-time deadline and a hard one require different evidence.
- No standard requires a roofline analysis. It requires that the requirement be verified; the roofline is one way to predict whether it will be.
- No standard specifies a thermal test duration. The device's duty cycle and the standard's environmental requirements determine it.
- No standard requires reporting utilization. A utilization number without a layer size and a precision is not interpretable, with or without a standard.

### 8.6 Documentation a reviewer will expect

| Record | Why it matters | Owner |
|---|---|---|
| Requirement with percentile and conditions | Verification has a target | Systems |
| Configuration record | Results are attributable | Firmware |
| Measurement procedure | Results are reproducible | Verification |
| Raw latency samples | Percentiles can be recomputed | Verification |
| Thermal soak record | Sustained behavior is evidenced | Hardware |
| Energy measurement method | Joules per inference is defensible | Hardware |
| Runtime and toolchain versions | Configuration control | Firmware |
| Deviation and waiver log | Honest record of gaps | Quality |

The pattern is familiar from Topic 01: the evidence is cheap at the time and expensive to reconstruct.

### 8.7 How benchmark results transfer, and when they do not

A benchmark result transfers to a product only when four conditions hold: the model architecture matches, the input distribution matches, the runtime and precision match, and the thermal and power conditions match. In practice, none of the four holds exactly, and the transfer error compounds.

The useful part of a benchmark is therefore not its score but its structure. MLPerf's scenarios tell you which metric matters for a workload. MLPerf Tiny's energy measurements tell you the order of magnitude for a class of part. The score tells you what a particular system achieved under conditions you do not share.

A team that cites a benchmark should state which of the four conditions differ and in which direction the result would move. A team that cannot state them is using the benchmark as an authority rather than as evidence.

## 9. Industry practice and case evidence

### 9.1 What vendors publish, and what it means

Vendor documentation is the primary source for peak specifications, and it is reliable for exactly that. Arm's ML Developers Guide lists the Ethos-U85 configurations: 128, 256, 512, 1,024, and 2,048 MACs per cycle, with performance at 1 GHz of 256 GOPS, 512 GOPS, 1 TOP/s, 2 TOPS, and 4 TOPS respectively, and internal memory from 16 KiB to 160 KiB depending on configuration. The product brief additionally lists 2:4 structured sparsity and weight-only lossless compression up to about 75%.

NVIDIA's H100 datasheet lists up to 700 W TDP, 80 GB of HBM3, 3.35 TB/s of bandwidth, and 3,958 TOPS INT8 with sparsity. The A100 datasheet lists 400 W for the SXM4 form factor and 250–300 W for PCIe variants. These are datasheet values, measured under vendor conditions, and they are the correct inputs to a roofline calculation. They are not predictions of your model's latency.

The distinction matters because a datasheet number is often quoted in a design review as if it were a measurement. The roofline makes the gap explicit: a 4 TOPS part with a 32 MAC/byte ridge point will spend most of its time waiting for memory on a layer with intensity 4.4, and the datasheet gives no hint of that.

### 9.2 What independent measurement shows

Independent studies of sustained mobile performance are consistently less optimistic than peak specifications. A study of a Pixel 2 reported thermal throttling beginning after 151 seconds, with the clock reduced from 2.46 GHz to 1.19 GHz at a 48 °C skin temperature, and frame-rate jank increasing by up to 146%. Another study of mobile benchmarks found run-to-run variance of up to 56% in a popular benchmark's sub-scores, and an increase of more than 60% when the device was chilled to 4 °C.

These are measurements of phones running graphics and benchmark workloads, not of medical devices running convolutional networks. The inference, clearly labelled, is that the same mechanisms apply to any fanless device with a skin-temperature limit: sustained performance is lower than burst performance, and the gap grows with ambient temperature and enclosure tightness.

### 9.3 Microcontroller evidence

MLPerf Tiny v1.2 provides vendor-submitted results on fixed benchmarks. The summary lists, among others:

| System | Core | KWS inferences/s | KWS energy (µJ/inference) |
|---|---|---:|---:|
| ST NUCLEO-H7A3ZI-Q | Cortex-M7, 280 MHz | 88.4 | 1,375.6 |
| ST NUCLEO-U575ZI-Q | Cortex-M33 | 34.4 | 656.2 |
| Renesas EK-RA8D1 | Cortex-M85, 480 MHz | 288.6 | not submitted |
| Qualcomm Snapdragon 8 Gen 3 Sensing Hub | — | 11,705 | not submitted |

The energy figures illustrate the scale: a keyword-spotting inference on a Cortex-M7 costs on the order of a millijoule, which is consistent with the 11.2 µJ first-layer estimate in Section 6 once the rest of the network and the board are included. The throughput figures show a 3.3× spread between a 280 MHz M7 and a 480 MHz M85, which is roughly consistent with clock and architecture differences, and a much larger spread to a phone sensing hub with a dedicated accelerator.

These are vendor-submitted results on a fixed benchmark. They are useful for order-of-magnitude planning and for choosing a class of part. They are not evidence that a clinical model will meet a clinical deadline.

### 9.4 Runtime documentation as performance evidence

Runtime documentation often reveals performance-relevant constraints that datasheets omit.

- ONNX Runtime separates basic graph optimizations (constant folding, redundant node elimination, Conv-BatchNorm fusion) from extended fusions available only on specific execution providers, and notes that offline-optimized models must match the target hardware and session options.
- ExecuTorch documents that each `.pte` file is backend-specialized and that unsupported operations fall back to the portable CPU implementation, so a model must be exported once per target and the partition inspected.
- TFLM documents a fixed tensor arena with a greedy memory planner and a runtime that fits in about 16 KiB on a Cortex-M3, which sets the floor for memory overhead on a microcontroller.

Each of these is a documented fact about the runtime. The inference for a performance model is that the same model file can have materially different latency on two runtimes, and the difference is not attributable to the hardware.

### 9.5 Documented fact versus inference

| Claim | Status |
|---|---|
| Ethos-U85 supports 128–2,048 MACs/cycle and 4 TOPS at 2,048 MACs, 1 GHz | Vendor documentation |
| H100 SXM is up to 700 W with 3.35 TB/s HBM3 | Vendor datasheet |
| Pixel 2 throttled after 151 s, 2.46 to 1.19 GHz at 48 °C skin | Independent academic measurement |
| Mobile benchmark variance up to 56% run to run | Independent academic measurement |
| MLPerf Tiny v1.2 has 91 performance and 18 energy results | MLCommons publication |
| A Cortex-M4F sustains 70% of peak on this layer | Constructed assumption for the worked example |
| Weight-only INT4 gives no latency benefit on a compute-bound layer | Inference from the roofline arithmetic, not a measurement |
| NPU invocation overhead dominates single-layer execution | Inference from typical overheads, not measured here |

The last three rows are the kind of statement that must be labelled in a design review. They are plausible and useful, and they are not evidence until someone measures them on the target.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| FLOPs used as latency | "It is only 4 GFLOPs" | Ignores memory and overhead | 4× miss on the bench | Roofline plus measurement |
| Peak TOPS used as achieved | Slide quotes datasheet | No model, no batch, no precision | Wrong part selected | Measure the model on the part |
| Model size used as memory | Arena overflow at boot | Activations and workspace omitted | Device fails to initialize | High-water measurement |
| Flash treated as SRAM | Latency doubles versus estimate | Wait states and copying | Deadline missed | Place hot data in SRAM; measure |
| Cache assumed to hold weights | First inference much slower | Cold cache, streaming weights | Cold-start deadline missed | Report cold and warm separately |
| Single-layer benchmark | Layer fast, model slow | Launch overhead per layer | Optimism by an order of magnitude | Profile the whole graph |
| Fallback unmeasured | One operator on CPU | Partition and copy | Latency spike, jitter | Operator list audit |
| Burst-only measurement | 30-second benchmark | Thermal throttling | Field latency 20–50% worse | Soak at production ambient |
| Batch confused with latency | Throughput improves, p99 does not | Queueing delay | Missed single-patient deadline | Measure batch 1 |
| Clock assumed constant | DVFS observed in logs | Power management | Variable latency | Pin clocks or log them |
| Utilization quoted without layer size | "We achieve 80%" | Small layers inflate overhead share | Misleading comparison | Report per layer |
| Different runtime versions compared | 15% unexplained difference | Kernel selection changed | Wrong attribution | Pin and record versions |
| Pre-production board used | Memory part differs | Bandwidth and wait states | Results do not transfer | Measure production board |
| Energy measured at the core | Board current higher | Regulators and memory omitted | Battery life shortfall | Measure at the battery |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| MAC count is the right work metric | Simple, portable | Sparse or approximated operators | Compare with counter counts |
| Peak MAC/s equals MACs per cycle × clock | Roofline input | Dual-issue, stalls, clock below nominal | Microbenchmark a dense GEMM |
| Bandwidth is a single number | Simple ridge calculation | Read/write asymmetry, bank conflicts | Measure read, write, mixed |
| Utilization is constant across layers | Simplifies prediction | Layer size and shape effects | Per-layer measurement |
| Bytes moved equal input plus output plus weights | Minimum traffic estimate | im2col, tiling, fusion | Counter-based traffic |
| 70% utilization on Cortex-M4F | Worked example | Kernel quality, memory stalls | Measure the layer |
| 30% bandwidth utilization on NPU | Worked example | DMA efficiency, layout | Measure the subgraph |
| 1 TOP/W for the NPU | Energy illustration | Process, clock, workload | Power-rail measurement |
| Board current is constant during inference | Energy estimate | DVFS, radio activity | High-rate power trace |
| Ambient 22 °C | Bench condition | Enclosure, climate | Chamber or field test |
| Production board equals development board | Transfer of results | Memory, clock, thermal differences | Measure both |
| Runtime version fixed | Comparability | Upgrade, patch | Pin and re-baseline |
| Model graph unchanged by export | Analytic count matches runtime | Fusion, constant folding | Compare profiler counts |

The most important limit is that the roofline is a bound, not a schedule. It tells you the fastest a layer can go under stated assumptions and which resource binds. It does not tell you what the runtime will do, because the runtime may fuse, reorder, tile, or fall back in ways the model does not represent. The roofline's value is that it makes a prediction falsifiable: if the measured time is far above the bound, something specific is wrong, and the gap has a cause worth finding.

A second limit is that performance is not the objective. A faster model that fails calibration or a subgroup is not deployable. The profiling report feeds the frontier; it does not replace it.

## 12. Exercises

**Exercise 1 — Operation counting (recall and calculation, 30 minutes).**
Count parameters and MACs for a 2D convolution with input 224×224×3, 64 output channels, kernel 7×7, stride 2, valid padding, batch 1. State the output shape and the counting convention.
*Deliverable:* a one-page calculation.
*Acceptance criteria:* output shape correct; MACs correct to three significant figures; convention stated (MAC versus FLOP).
*Time:* 30 minutes.

**Exercise 2 — Roofline placement (application, 40 minutes).**
Using the layer from Exercise 1, compute bytes moved, arithmetic intensity, and the bound on (a) a Cortex-M4F at 64 MHz, (b) an Ethos-U85-512 at 1 GHz, (c) an H100 at dense INT8. Assume bandwidths of 256 MB/s, 16 GB/s, and 3.35 TB/s.
*Deliverable:* a table with intensity, ridge, bound, and predicted time.
*Acceptance criteria:* units correct; ridge computed from peak over bandwidth; the same layer classified differently on at least two machines with an explanation.
*Time:* 40 minutes.

**Exercise 3 — Bandwidth measurement (application, 35 minutes).**
Write a streaming benchmark in Python that copies 64 MiB in 4 KiB chunks and reports bytes per second. Run it on your workstation. Explain why the result is not the bandwidth of the target.
*Deliverable:* the script and a 150-word explanation.
*Acceptance criteria:* the benchmark reports bytes and seconds; the explanation names cache, DRAM, and the difference between a workstation and an MCU.
*Time:* 35 minutes.

**Exercise 4 — Operator attribution (analysis, 45 minutes).**
Given a trace with 40 operators, 35 of which take 40 µs each and 5 of which take 2 ms each, compute total time and the share of the top five. State whether the model is overhead-dominated or hotspot-dominated.
*Deliverable:* a table and a two-sentence conclusion.
*Acceptance criteria:* arithmetic correct; the conclusion names the correct optimization lever.
*Time:* 45 minutes.

**Exercise 5 — Quantization effect on latency (critique, 40 minutes).**
Using Section 6, explain why weight-only INT8 quantization does not reduce latency on the Cortex-M4F. Then state the condition under which it would.
*Deliverable:* a 250-word explanation with the byte arithmetic.
*Acceptance criteria:* identifies weight bytes as 40 of 9,256; states that the layer is compute-bound; names a weight-bound layer as the counterexample.
*Time:* 40 minutes.

**Exercise 6 — Thermal derating (analysis, 50 minutes).**
A device meets 40 ms p99 at 22 °C but 58 ms after 20 minutes at 40 °C. Compute the derating factor and state whether a 50 ms deadline is met. Propose two design responses and their trade-offs.
*Deliverable:* a calculation and a short memo.
*Acceptance criteria:* factor correct (1.45); deadline verdict correct; responses distinguish thermal design from software.
*Time:* 50 minutes.

**Exercise 7 — Hardware selection (design, 60 minutes).**
Given a ledger with 50 ms p99, 2 mJ per inference, 256 KiB SRAM, and a 5 USD BOM ceiling, choose between an MCU with a DSP extension, an MCU plus a small NPU, and an applications processor. State the evidence you would need before committing.
*Deliverable:* a one-page decision note with a table.
*Acceptance criteria:* choice follows from the constraints, not from peak TOPS; evidence list includes target measurement; a fallback is named.
*Time:* 60 minutes.

**Exercise 8 — Profiling report (design, 90 minutes).**
Write a profiling report template with the ten fields from Step 10, and fill it for the Section 6 case. Mark every field that is an assumption rather than a measurement.
*Deliverable:* a two-page report.
*Acceptance criteria:* all ten fields present; assumptions marked; predicted and measured gains compared; reproduction steps sufficient for a colleague.
*Time:* 90 minutes.

## 13. Self-check answers

**A1. Output shape and MACs for Exercise 1.**
Output is 109×109×64 (floor((224-7)/2)+1 = 109). MACs = 109 × 109 × 64 × 3 × 7 × 7 = 109 × 109 = 11,881; × 64 = 760,384; × 147 = 111,776,448 MACs, about 111.8 MMAC, or 223.6 MFLOP under the two-operations convention.

**A2. Why the same layer is compute-bound on an MCU and memory-bound on an NPU.**
Because the ridge point differs. The MCU has 128 MMAC/s and 256 MB/s, a ridge of 0.5 MAC/byte, so an intensity of 4.4 is far to the compute side. The NPU has 512 GMAC/s and 16 GB/s, a ridge of 32 MAC/byte, so the same intensity is on the memory side. The layer did not change; the machine's ratio of compute to bandwidth did.

**A3. What does a utilization of 3% mean?**
That the model achieves 3% of peak operations per second. It usually means overhead dominates: many small layers, kernel launch costs, or fallback. It does not by itself mean the hardware is bad, and it does mean that reducing MACs will not help much.

**A4. Why is peak memory larger than model size?**
Because the runtime holds activations, workspace, input and output buffers, the runtime itself, and allocator fragments in addition to weights. TFLM's tensor arena is a single buffer for these, sized by a greedy planner, and it is often several times the largest activation.

**A5. Difference between TOPS and MAC/s.**
One MAC is two operations (multiply and add), so 1 TOPS equals 0.5 TMAC/s. A vendor quoting 4 TOPS INT8 is quoting 2 TMAC/s, before considering sparsity, which may double the number for supported layers.

**A6. When does weight-only quantization reduce latency?**
When the layer is bound by weight traffic: weights dominate bytes moved, typically a large fully connected or embedding layer at small batch. It does not help when activations dominate or when the layer is compute-bound, as in the Section 6 convolution.

**A7. Why measure cold and warm start separately?**
Cold start includes model load, cache fill, and clock ramp, and can be several times the warm latency. A clinical device that must respond immediately after power-on is bound by cold start; a continuously running device is bound by warm. Reporting one number hides the other.

**A8. What is the first test when p99 is much larger than p50?**
A time series of latency with clock and temperature. If latency rises over minutes, the cause is thermal. If it spikes intermittently, the cause is scheduling, fallback, or memory allocation. The distribution shape distinguishes the mechanisms.

**A9. Why is a 30-second benchmark not evidence for a field deadline?**
Because thermal steady state takes minutes. A device can pass a short test at full clock and fail after throttling. The soak duration must match the clinical duty cycle, and the ambient must match the intended environment.

**A10. What is the difference between an operator and a kernel?**
An operator is a graph node with defined semantics, such as a convolution. A kernel is a specific implementation of that operator on a specific device, with its own memory access pattern and performance. The same operator can have fast and slow kernels on the same hardware.

**A11. Rubric for Exercise 7 (hardware selection).**
A strong answer derives the requirement from the ledger, computes whether each option can meet the energy and memory constraints, identifies the evidence gap (target measurement), and names a fallback. A weak answer picks the part with the highest TOPS or the lowest price without linking to the constraints.

**A12. Rubric for Exercise 8 (profiling report).**
A strong report separates measurement from assumption, includes the configuration record and raw percentiles, explains the binding resource with arithmetic, and states the reproduction steps. A weak report presents a latency number and a conclusion with no conditions.

## 14. Interview questions

**Q1. Why is FLOPs not latency?**
A strong answer separates arithmetic from data movement and overhead, introduces arithmetic intensity and the roofline, and gives an example where the same MAC count runs 10× slower because of memory or launch overhead. A weak answer says "because of hardware differences" without a mechanism.

**Q2. What is arithmetic intensity and how do you compute it?**
A strong answer defines MACs per byte, states which memory level and which bytes are counted, and notes that im2col and tiling change the real traffic. A weak answer gives the formula without the caveats.

**Q3. What is a ridge point?**
A strong answer defines it as peak compute divided by peak bandwidth, explains that it separates memory-bound from compute-bound workloads, and notes that it differs by orders of magnitude between an MCU and a GPU.

**Q4. How would you choose between an MCU with a DSP and an MCU with an NPU?**
A strong answer starts from the constraint ledger, considers operator support, memory, energy, and toolchain maturity, and proposes a target measurement before committing. A weak answer compares TOPS.

**Q5. A vendor claims 4 TOPS. What questions do you ask?**
A strong answer asks precision, sparsity, dense versus sparse, batch, model, and whether the number is peak or sustained. A weak answer accepts the number and divides by the model's MACs.

**Q6. Why can a model fit in flash and still fail?**
A strong answer explains that execution requires activations and workspace in SRAM, that flash reads have wait states, and that peak memory exceeds model size. A weak answer assumes flash is directly executable at full speed.

**Q7. How do you measure energy per inference?**
A strong answer describes a power rail measurement at high sample rate, integration over the inference window, inclusion of preprocessing and radio, and reporting in millijoules with the voltage and method. A weak answer multiplies datasheet power by time.

**Q8. What does thermal throttling do to a latency budget?**
A strong answer explains that clock reduction increases compute time, that the effect appears after minutes, and that the deadline must be verified at sustained state and specified ambient. A weak answer treats it as a minor derating.

**Q9. What is operator fallback and why is it expensive?**
A strong answer describes partitioning, the copy at the boundary, synchronization, and the loss of fusion, and notes that the cost is often larger than the operator's arithmetic. A weak answer says the operator runs on the CPU and is "a bit slower".

**Q10. How do you attribute latency to layers?**
A strong answer uses a profiler or instrumented runtime, cross-checks with hardware counters, accounts for instrumentation overhead, and groups results into hotspot, overhead, fallback, and uniform patterns. A weak answer guesses from parameter counts.

**Q11. Why is batching not a solution for a single-patient device?**
A strong answer distinguishes throughput from latency: batching amortizes weights and launch overhead but adds queueing delay, so p99 for the first item in a batch worsens. A weak answer assumes batching always helps.

**Q12. What would make you distrust a profiling result?**
A strong answer names a missing configuration record, a cold-only measurement, a different runtime version, a pre-production board, a benchmark input that differs from the clinical distribution, or a total time that does not match the sum of parts.

## 15. Key terms

- **Achieved throughput** — operations per second on the actual model and runtime.
- **Arithmetic intensity** — MACs per byte moved, at a stated memory level.
- **Batch** — inputs processed together; trades latency for throughput.
- **Cache** — small fast memory holding recently used data.
- **Clock** — processor frequency; may vary under DVFS.
- **Cold start** — first inference after load, with cold caches and clocks.
- **Compute-bound** — limited by arithmetic throughput rather than memory.
- **DMA** — direct memory access; moves data without CPU involvement.
- **DRAM** — dynamic random-access memory; large and slower than SRAM.
- **DSP** — digital signal processor; optimized for fixed-point signal workloads.
- **DVFS** — dynamic voltage and frequency scaling.
- **Essential performance** — performance necessary to keep risk acceptable under IEC 60601-1.
- **Fallback** — execution on another processor when an operator is unsupported.
- **Flash** — non-volatile storage; reads incur wait states.
- **FLOP** — floating-point operation; a MAC is two FLOPs.
- **Kernel** — device-specific implementation of an operator.
- **MAC** — multiply-accumulate; one multiply and one add.
- **MAC array** — parallel multiplier structure in an NPU.
- **Memory-bound** — limited by bytes per second rather than arithmetic.
- **MVE** — M-Profile Vector Extension (Helium) on Cortex-M55 and M85.
- **NPU** — neural processing unit; fixed-function neural accelerator.
- **Operator** — graph node with defined semantics.
- **Peak throughput** — datasheet maximum at a stated precision.
- **p99** — 99th percentile latency.
- **Ridge point** — intensity at which compute and bandwidth roofs meet.
- **Roofline** — performance bound from peak compute and bandwidth.
- **SIMD** — single instruction, multiple data.
- **SRAM** — static random-access memory; fast, small, on-chip.
- **Sustained throughput** — throughput after thermal and clock settling.
- **TDP** — thermal design power.
- **Thermal resistance** — temperature rise per watt, in °C/W.
- **Throttling** — clock or voltage reduction under thermal limits.
- **TOPS** — tera-operations per second; a MAC counts as two operations.
- **Utilization** — achieved divided by peak.
- **Wait state** — extra cycles when reading slower memory.
- **Warm start** — steady-state inference with warm caches and clocks.
- **Zero point** — integer code representing real zero.

## 16. Sources

All URLs accessed 25 October 2026.

**Hardware documentation**
- Arm, "ML Developers Guide for Cortex-M Processors and Ethos-U NPU — Ethos-U hardware architecture". https://support.arm.com/documentation/109267/0103/Arm-Ethos-U-NPU/Ethos-U-hardware-architecture
- Arm, "Ethos-U85 product brief". https://armkeil.blob.core.windows.net/developer/Files/pdf/datasheets/arm-ethos-u85-product-brief.pdf
- Arm, "Ethos-U85" product page. https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85
- NVIDIA, "NVIDIA H100 GPU Datasheet". https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306
- NVIDIA, "NVIDIA A100 Tensor Core GPU Datasheet". https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- Arm, "CMSIS-NN: CMSIS NN Software Library". https://arm-software.github.io/CMSIS-NN/latest/

**Performance modelling**
- Williams, S., Waterman, A., Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM 52(4), 2009. https://doi.org/10.1145/1498765.1498785

**Runtime documentation**
- ONNX Runtime, "Graph optimizations". https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- PyTorch, "ExecuTorch model export and lowering". https://docs.pytorch.org/executorch/stable/using-executorch-export.html
- Google AI Edge, "LiteRT for Microcontrollers overview". https://developers.google.com/edge/litert/microcontrollers/overview
- TensorFlow, "Memory management in TFLM". https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md

**Benchmarks**
- MLCommons, "MLPerf Inference v6.1" datacenter and edge results. https://mlcommons.org/benchmarks/inference-edge/
- MLCommons, "MLPerf Tiny v1.2 Results", April 2024. https://mlcommons.org/2024/04/mlperf-tiny-v1-2-results/
- MLCommons, `tiny_results_v1.2` summary. https://github.com/mlcommons/tiny_results_v1.2/blob/main/summary.csv

**Thermal measurement**
- Bantock, D., et al., "Mitigating Interactive Performance Degradation from Mobile Device Thermal Throttling". https://eprints.soton.ac.uk/439709/1/tus_accepted_paper.pdf
- Guo, Y., et al., "Freeze It If You Can: Challenges and Future Directions in Mobile Benchmarking", HotMobile 2017. https://yaoguopku.github.io/papers/Guo-HotMobile-17.pdf

**Standards**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- IEC 60601-1, Medical electrical equipment — General requirements for basic safety and essential performance. https://www.iec.ch/
- ISO 14971:2019. https://www.iso.org/standard/72704.html

> **Note:** Datasheet values are vendor measurements under vendor conditions. MLPerf results are vendor-submitted. The Section 6 model and its measurements are constructed for teaching.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`01_the-deployment-pareto-frontier.md`](01_the-deployment-pareto-frontier.md)
- Next topic: [`03_the-compression-family.md`](03_the-compression-family.md)
- Related topic: [`07_inference-optimization.md`](07_inference-optimization.md)
- Related topic: [`10_validation-on-target-hardware.md`](10_validation-on-target-hardware.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
