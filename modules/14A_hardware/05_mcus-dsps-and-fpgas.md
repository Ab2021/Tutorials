# MCUs, DSPs, and FPGAs

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 05 of 14
> Estimated study time: 4 hours
> Prerequisites: 01, 04
> Version 2026.3

## 1. Why this topic exists

The processor is where a signal becomes a decision. It runs the filter, extracts the feature, applies the model, checks the safety limit, and drives the output. The choice of processor determines whether the decision is made on time, with what jitter, at what power, and with what evidence of correctness.

The industry problem is that processor choice is often made on familiarity rather than on the workload. A team uses a microcontroller because it used one before; another uses an FPGA because it is "faster"; a third uses a system on chip (SoC) because it runs Linux. Each choice can be correct, and each can be wrong, depending on the algorithm's arithmetic intensity, the latency requirement, the jitter tolerance, the power budget, and the certification evidence.

Three decisions depend on processor literacy:

1. **Whether the workload fits.** A model with 2 million multiply-accumulate operations per inference has a different home from a model with 2 billion. The arithmetic intensity, the memory footprint, and the data movement determine the architecture.
2. **Whether the timing is deterministic.** Average latency is not enough for a closed-loop device or an alarm. The worst-case latency and the jitter determine whether the device meets its clinical requirement.
3. **Whether the evidence is achievable.** A processor with caches and a rich operating system is harder to certify for a hard real-time function than a microcontroller with a deterministic interrupt path. The architecture determines the verification burden.

This topic covers microcontrollers (MCUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs), and the boundary between them. It focuses on the properties that change a data decision: determinism, latency, throughput, memory, power, and toolchain. It does not teach register-level programming or hardware description languages. It teaches how to read a workload and choose a home for it.

A useful framing: the processor is a contract between the algorithm and the clock. The algorithm needs a number of operations, a quantity of memory, and a deadline. The processor offers a number of operations per second, a memory hierarchy, and an interrupt latency. The contract is feasible only if the worst case, not the average, fits.

## 2. Learning outcomes

By the end of this topic you can:

- Estimate the compute, memory, and I/O requirements of an algorithm from its operation count and data flow.
- Choose among an MCU, a DSP, an FPGA, and an SoC from the workload, latency, jitter, power, and certification requirements.
- Compute the worst-case execution time of a loop from its cycle count, clock frequency, and memory wait states.
- Explain why caches, direct memory access (DMA), and interrupts create jitter, and how to bound it.
- Compute the energy per inference and compare it with the energy budget of the device.
- Design a partitioning of an algorithm across a processor and an accelerator.
- Identify the firmware and toolchain evidence that a regulated device requires.
- Critique a claim that a processor is "real-time" without a latency distribution.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A processor is a pipeline that turns instructions and data into results at a clock rate. The pipeline has a width (operations per cycle), a depth (latency), a memory hierarchy (registers, cache, RAM, flash), and an interrupt mechanism. The effective throughput is the product of the clock, the operations per cycle, and the utilization; the effective latency is the sum of the pipeline, the memory, and the interrupt delays.

```text
        WORKLOAD                         ARCHITECTURE
   operations (MACs)  ----------->  throughput = clock x width x utilization
   data (bytes)       ----------->  memory bandwidth and capacity
   deadline (ms)      ----------->  worst-case latency, not average
   jitter tolerance   ----------->  determinism: cache, DMA, interrupts, OS
   energy (J)         ----------->  power x time, plus leakage
   evidence           ----------->  toolchain, traceability, certification
```

The model has four equations:

\[
T_{\text{compute}} = \frac{\text{operations}}{\text{clock} \times \text{ops/cycle} \times \text{utilization}}
\]

\[
T_{\text{worst}} = T_{\text{compute}} + T_{\text{memory}} + T_{\text{interrupt}} + T_{\text{OS}}
\]

\[
E_{\text{inference}} = P_{\text{active}} \times T_{\text{active}} + P_{\text{leakage}} \times T_{\text{total}}
\]

\[
\text{Utilization} = \frac{\text{operations}}{\text{clock} \times \text{ops/cycle} \times T_{\text{compute}}}
\]

The first equation gives the average; the second gives the number that matters for a deadline. The difference between them is the jitter, and the jitter is usually caused by memory, not by arithmetic.

### 3.1 The four architectures

| Architecture | Strength | Weakness | Typical medical use |
|---|---|---|---|
| MCU | Low power, deterministic interrupts, integrated peripherals | Limited compute and memory | Sensor hub, pump control, wearable |
| DSP | Multiply-accumulate throughput, addressing modes | Specialised toolchain, power | ECG filtering, ultrasound beamforming, audio |
| FPGA | Parallelism, deterministic latency, custom I/O | Design effort, power, cost | Ultrasound, imaging, high-speed control |
| SoC | CPU + GPU/NPU + peripherals, Linux | Nondeterminism, power, complexity | Imaging, edge AI, connectivity |

The architectures are not a ladder. An MCU is not a worse SoC; it is a better choice when the workload is small, the power is tight, and the determinism matters. An FPGA is not a faster MCU; it is a different programming model in which parallelism is spatial rather than temporal.

### 3.2 The memory hierarchy is the real constraint

A processor's arithmetic is fast; its memory is slow. A 480 MHz core with a 1-cycle multiply can execute 480 million operations per second, but a random access to external dynamic RAM (DRAM) can take 50–100 ns, or 24–48 cycles. If every operation needs a memory access, the effective throughput collapses.

| Memory | Latency | Bandwidth | Energy per access |
|---|---:|---:|---:|
| Register | 1 cycle | Very high | Lowest |
| Level 1 cache | 1–4 cycles | High | Low |
| Level 2 cache | 10–20 cycles | Moderate | Moderate |
| SRAM | 10–30 cycles | Moderate | Moderate |
| DRAM | 50–100 ns | High (burst) | High |
| Flash | 20–100 ns | Moderate | Moderate |

The implication is that data reuse matters more than arithmetic. A convolution with a small kernel can keep the kernel in registers and stream the data; a fully connected layer with a large weight matrix must read the weights from memory, and the memory bandwidth becomes the limit. This is why model compression and quantization (Module 14B) matter for edge devices: they reduce both the operations and the bytes.

### 3.3 Determinism and jitter

Determinism is the property that the same input produces the same output in the same time, every time. It is not the same as speed. A fast processor with a cache can be fast on average and slow in the worst case; a slower processor with tightly coupled memory can be slower on average and faster in the worst case.

| Source of jitter | Mechanism | Bound it by |
|---|---|---|
| Cache miss | A memory access misses and stalls the pipeline | Lock critical code and data in cache or TCM |
| DMA contention | A DMA transfer steals bus cycles | Reserve bus bandwidth; priority; separate bus |
| Interrupt nesting | A higher-priority interrupt delays the current one | Priority design; interrupt latency analysis |
| Flash wait states | Instruction fetch stalls on flash | Prefetch; copy to RAM; choose a faster flash |
| Operating system | Scheduler, page faults, drivers | Real-time OS; PREEMPT_RT; memory locking |
| Clock frequency scaling | Dynamic voltage and frequency scaling | Lock the frequency; disable scaling |
| Thermal throttling | Clock reduction at high temperature | Thermal design; workload limiting |

A hard real-time function needs a worst-case bound on every row. A soft real-time function can tolerate occasional misses, but it still needs a distribution, not an average.

### 3.4 The roofline

The roofline model plots achievable performance against arithmetic intensity (operations per byte). A workload below the ridge point is memory-bound; above it, compute-bound.

\[
\text{Performance} = \min(\text{peak compute}, \text{memory bandwidth} \times \text{arithmetic intensity})
\]

For an MCU with 1 MAC/cycle at 100 MHz and 100 MB/s of memory bandwidth, the ridge point is 1 operation per byte. A convolution with 9 weights per output and 2 bytes per input has an arithmetic intensity of about 4.5 operations per byte, so it is compute-bound. A fully connected layer with 1,000 weights per output and 2 bytes per weight has an intensity of 0.5, so it is memory-bound. The architecture choice follows from the roofline, not from the marketing.

### 3.5 The three budgets

A processor design has three budgets that must all close.

| Budget | Quantity | Closing condition |
|---|---|---|
| Compute | MACs per second | Effective rate > required rate with margin |
| Memory | Bytes per second | Bandwidth > required traffic with margin |
| Time | Worst-case latency | WCET < deadline minus jitter |

A design that closes the compute budget but not the memory budget is a design that will stall. A design that closes both but not the time budget is a design that will miss the deadline under load. The three budgets are independent, and the weakest one sets the performance.

A fourth budget, energy, cuts across all three. A design that meets the compute and time budgets at 10 W may not meet the thermal or battery budget. The energy budget must be checked over the duty cycle, not only at peak.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| MCU | Microcontroller unit; processor with integrated memory and peripherals | Confused with a microprocessor | None |
| DSP | Digital signal processor; architecture optimised for multiply-accumulate | Used for any signal processing | None |
| FPGA | Field-programmable gate array; reconfigurable logic | Confused with a microcontroller | None |
| SoC | System on chip; CPU plus accelerators and peripherals | Used for any integrated circuit | None |
| MAC | Multiply-accumulate operation | Confused with a floating-point operation | Operation |
| Clock frequency | Cycles per second | Confused with throughput | Hz |
| IPC | Instructions per cycle | Confused with operations per cycle | Instructions/cycle |
| Utilization | Fraction of peak performance achieved | Confused with efficiency | % |
| Worst-case execution time (WCET) | Upper bound on execution time | Confused with average | s |
| Jitter | Variation in latency | Confused with latency | s or % |
| Determinism | Same input, same output, same time | Confused with speed | None |
| Cache | Fast memory holding recently used data | Assumed to be predictable | Bytes |
| TCM | Tightly coupled memory; low-latency, deterministic | Confused with cache | Bytes |
| DMA | Direct memory access; transfers without the CPU | Assumed to be free | None |
| Interrupt latency | Time from interrupt to ISR execution | Confused with interrupt response | Cycles or s |
| Tail-chaining | Back-to-back interrupt handling without full restore | Confused with nesting | Cycles |
| Priority inversion | Low-priority task blocks a high-priority one | Confused with starvation | None |
| RTOS | Real-time operating system | Assumed to guarantee deadlines | None |
| Scheduler | Component that decides which task runs | Confused with the OS | None |
| PREEMPT_RT | Linux real-time patch set | Assumed to make Linux hard real-time | None |
| NPU | Neural processing unit | Confused with a GPU | None |
| GPU | Graphics processing unit; parallel compute | Assumed to be efficient for small models | None |
| Accelerator | Hardware specialised for a workload | Confused with a coprocessor | None |
| Roofline | Performance model bound by compute or memory | Used as a marketing term | Operations/byte |
| Arithmetic intensity | Operations per byte of memory traffic | Confused with operations per second | operations/byte |
| Throughput | Operations or inferences per second | Confused with latency | 1/s |
| Latency | Time from input to output | Confused with throughput | s |
| Energy per inference | Joules consumed per inference | Confused with power | J |
| Leakage power | Power consumed when idle | Ignored in duty-cycled systems | W |
| Toolchain | Compiler, linker, debugger, libraries | Ignored in certification | None |
| Traceability | Link from requirement to test | Confused with documentation | None |
| MISRA C | Coding standard for safety-critical C | Assumed to guarantee safety | None |

## 5. The mechanics, step by step

The procedure below maps a workload to an architecture and verifies that the mapping meets its deadline. Each step states its input, output, and rationale. The edge cases required by the assignment are mapped to the steps and summarised in Section 5.13.

### Step 1. Characterise the workload

**Input:** the algorithm, its inputs, and its outputs.

**Output:** operation count, memory footprint, data movement, and the dependency structure.

**Rationale:** the operation count alone is not enough. A model with 1 million operations and 4 MB of weights has a different memory problem from one with 1 million operations and 4 kB of weights. Count both.

For a neural network, count the multiply-accumulates per inference and the bytes read and written. For a filter, count the taps per sample and the state variables. For a control loop, count the operations per cycle and the sensor reads. Write the count in a table, not in prose.

### Step 2. Characterise the deadline and the jitter tolerance

**Input:** the clinical function, the loop rate, and the consequence of a miss.

**Output:** a deadline, a jitter tolerance, and a consequence statement.

**Rationale:** a deadline is not a single number. It has a nominal value and a worst case. A closed-loop insulin pump may need a 100 ms loop with a 20 ms jitter tolerance; an alarm may need a 1 s response with a 200 ms tolerance; a display may need 30 frames per second with no hard deadline.

The consequence statement matters. A missed display frame is cosmetic; a missed pump deadline is a safety event. The architecture and the evidence burden follow from the consequence.

### Step 3. Estimate the operation count

**Input:** the algorithm and its dimensions.

**Output:** operations per inference, operations per second, and the dominant operation type.

**Rationale:** the operation count sets the compute requirement. For a convolution layer with \(C_{\text{in}}\) input channels, \(C_{\text{out}}\) output channels, a kernel of \(K\) taps, and an output length \(L\), the multiply-accumulates are:

\[
\text{MACs} = C_{\text{in}} \times C_{\text{out}} \times K \times L
\]

For a fully connected layer with \(N\) inputs and \(M\) outputs, the MACs are \(N \times M\). For a filter with \(T\) taps per sample at \(f_s\), the MACs per second are \(T \times f_s\).

### Step 4. Estimate the memory traffic

**Input:** the algorithm's data flow and the memory hierarchy.

**Output:** bytes read and written per inference, and the arithmetic intensity.

**Rationale:** the memory traffic determines whether the workload is compute-bound or memory-bound. For each layer, count the input bytes, the weight bytes, and the output bytes. The arithmetic intensity is the MACs divided by the bytes. A layer with an intensity below the machine's ridge point is memory-bound, and a faster clock will not help.

### Step 5. Choose the architecture

**Input:** the operation count, the memory traffic, the deadline, the jitter tolerance, the power, and the evidence requirement.

**Output:** an architecture with a justification and a rejected alternative.

**Rationale:** the choice follows from the constraints. A small workload with a hard deadline and low power is an MCU. A medium workload with a hard deadline and high throughput is an FPGA or a DSP. A large workload with a soft deadline and a rich software stack is an SoC.

**Decision table: architecture selection**

| Workload | Deadline | Jitter | Power | Evidence | Choice |
|---|---|---|---|---|---|
| < 10 M MACs | > 10 ms | Moderate | < 100 mW | IEC 62304 class B | MCU |
| 10–500 M MACs | 1–10 ms | Low | < 1 W | Class B/C | DSP or FPGA |
| > 500 M MACs | > 10 ms | Moderate | < 10 W | Class B | SoC with NPU |
| Any | < 100 µs | Very low | Any | Class C | FPGA |
| Any | Soft | High | < 1 W | Class A | SoC with Linux |

The table is a starting point. The final choice depends on the toolchain, the availability of the part, and the team's ability to verify it.

### Step 6. Compute the average execution time

**Input:** the operations, the clock, the operations per cycle, and the utilization.

**Output:** an average execution time.

**Rationale:** the average time is the first estimate, not the answer. It tells you whether the architecture is in the right order of magnitude. A workload that needs 10 ms on average cannot meet a 1 ms deadline, regardless of the worst-case analysis.

Use a measured utilization, not the peak. A core that can issue one MAC per cycle rarely achieves more than 30–70% on a real workload because of memory stalls, loop overhead, and dependencies. A utilization of 30% is a reasonable first estimate for an MCU; 60% for a DSP with a good compiler; 80% for an FPGA with a pipelined design.

### Step 7. Compute the worst-case execution time

**Input:** the average time and the jitter sources.

**Output:** a worst-case bound and a deadline margin.

**Rationale:** the worst case is the average plus the worst-case stalls. The stalls come from cache misses, DMA contention, interrupt nesting, flash wait states, and the operating system. Each must be bounded, measured, or eliminated.

A practical method is to measure the latency distribution over a long run and use a high percentile, then add a margin for the unmeasured tail. A measurement of 10,000 inferences gives a p99.9 but not a guaranteed worst case. For a hard deadline, the bound must come from analysis or from a deterministic architecture.

### Step 8. Compute the energy per inference

**Input:** the active power, the execution time, the leakage power, and the duty cycle.

**Output:** energy per inference and energy per day.

**Rationale:** the energy is the power times the time, plus the leakage times the total time. A fast processor that finishes quickly and sleeps can use less energy than a slow processor that runs longer, even if its active power is higher. The crossover depends on the sleep power and the duty cycle.

\[
E = P_{\text{active}} T_{\text{active}} + P_{\text{sleep}} (T_{\text{period}} - T_{\text{active}})
\]

A device that infers once per second and sleeps for the rest of the second is dominated by the sleep energy, not the inference energy. A device that infers continuously is dominated by the active energy.

### Step 9. Partition the algorithm

**Input:** the algorithm's stages and the architecture's resources.

**Output:** a mapping of stages to processors or accelerators.

**Rationale:** not every stage belongs on the same architecture. A filter may run on a DSP, a neural network on an NPU, and the control loop on an MCU. The partition must respect the data flow and the latency; a partition that adds a 10 ms transfer between stages may miss the deadline even if each stage is fast.

The partition also affects the evidence. A safety function split across two processors needs a communication protocol and a timing analysis. A single-processor implementation is easier to verify but may not meet the deadline.

### Step 10. Design the memory and I/O

**Input:** the data flow, the memory hierarchy, and the peripherals.

**Output:** a memory map, a DMA schedule, and an interrupt priority scheme.

**Rationale:** the memory map determines the latency and the jitter. Critical code and data should be in tightly coupled memory or locked cache; bulk data should use DMA; the interrupt priorities should reflect the deadlines. A design that shares a bus between the DMA and the core without a bandwidth budget will show jitter under load.

### Step 11. Verify the timing

**Input:** the prototype and the deadline.

**Output:** a latency distribution, a worst-case bound, and a trace.

**Rationale:** timing verification is a measurement, not an argument. Measure the latency over a long run, with the worst-case workload and the worst-case environment. Record the distribution, the p99, the maximum, and the conditions. If the maximum exceeds the deadline, the design fails, even if the average is comfortable.

Use a hardware timer or a trace unit, not a software timestamp, to measure the latency. A software timestamp includes the overhead of reading the clock and can hide the very stall you are trying to measure.

### Step 12. Document the configuration and the evidence

**Input:** the architecture, the firmware, the toolchain, and the tests.

**Output:** a configuration record, a traceability matrix, and a verification report.

**Rationale:** the processor's behaviour depends on the clock, the memory map, the compiler, the libraries, and the firmware version. The record must capture all of them. The traceability matrix must link each requirement to a test, and each test to a result.

For a regulated device, the software lifecycle evidence follows IEC 62304. The class of the software (A, B, or C) determines the rigour; a class C function requires architecture analysis, detailed design, unit verification, and integration testing.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 1, 7 | Workload exceeds throughput | Utilization > 100%; missed deadlines | Lost samples |
| Common-mode rejection failure | 10 | Not applicable at processor level; analogue issue | N/A | N/A |
| Electrode offset | 10 | Not applicable; analogue issue | N/A | N/A |
| Clock jitter | 7, 10 | Sampling or timing variation | Latency distribution | Jitter; missed deadline |
| Brown-out | 11, 12 | Supply drop; reset | Supply monitor | Lost state; invalid output |
| Watchdog reset loop | 12 | Firmware fails to service watchdog | Reset counter | Device unavailable |
| Thermal throttling | 7, 8 | Clock reduction at high temperature | Temperature log | Increased latency |
| Firmware update failure | 12 | Incomplete write; bad image | Version and checksum | Device bricked |
| Calibration drift | 12 | Coefficients age | Reference check | Biased output |

The table includes the analogue edge cases for completeness, but they are handled in Topics 01–04. At the processor level, the relevant edge cases are the timing, power, and firmware ones.

### 5.14 Determinism checklist

| Check | Criterion | If it fails |
|---|---|---|
| Critical code in TCM or locked cache | No cache miss in the critical loop | Copy code to TCM; lock cache |
| DMA bandwidth reserved | Worst-case DMA does not starve the core | Reduce DMA burst; separate bus |
| Interrupt priorities | Highest priority for the hard deadline | Reassign priorities; analyze nesting |
| Flash wait states | Known and bounded | Prefetch; copy to RAM |
| Clock frequency fixed | No DVFS during the critical function | Lock frequency; disable scaling |
| OS preemption | Bounded by design | Use an RTOS; PREEMPT_RT; measure |
| Watchdog | Services the loop without hiding a hang | Independent watchdog; windowed |
| Thermal | Junction temperature below the limit | Reduce power; improve heatsink |
| Toolchain | Version recorded; reproducible build | Pin the compiler; archive the image |

A design that passes all nine has a defensible determinism claim. A design that passes only the first and the last has a hope.

### 5.15 Cycle-accurate estimation

A first estimate uses operations and utilization. A second estimate counts cycles.

| Operation | Cycles on a Cortex-M4 | Notes |
|---|---:|---|
| 32-bit multiply | 1 | Single-cycle multiplier |
| 32-bit MAC | 1 | With accumulate |
| Load from SRAM | 2 | Zero wait state |
| Load from flash | 3–10 | Depends on wait states and prefetch |
| Divide | 2–12 | Software or hardware |
| Branch | 1–3 | Pipeline refill |
| Interrupt entry | 12 | Maximum, zero wait state |
| Interrupt return | 12 | Tail-chaining is 6 |
| Function call | 2–6 | Depends on the ABI |

A loop with a 1-cycle MAC but a 3-cycle flash fetch and a 1-cycle loop overhead executes at 0.2 MAC/cycle, not 1. The utilization is 20%, not 100%. The estimate is only as good as the cycle count, and the cycle count is only as good as the memory model.

For a DSP, the cycle count depends on the addressing modes and the parallelism. A TMS320C6000 can execute eight instructions per cycle in the best case, but a loop with a data dependency may execute one. The compiler's schedule and the memory bank conflicts determine the achieved rate.

For an FPGA, the cycle count is replaced by the pipeline depth and the initiation interval. A design that accepts one sample per cycle at 100 MHz has a throughput of 100 million samples per second, regardless of the number of operations per sample, as long as the pipeline is not stalled. The latency is the pipeline depth times the clock period.

### 5.16 RTOS scheduling and priority inversion

A real-time operating system (RTOS) provides tasks, priorities, and synchronisation. It does not guarantee that a task meets its deadline; it provides the mechanisms, and the designer provides the analysis.

| Mechanism | What it guarantees | What it does not guarantee |
|---|---|---|
| Fixed-priority preemption | Higher-priority tasks run first | That the priorities reflect the deadlines |
| Priority inheritance | A low-priority task holding a lock runs at the waiter's priority | That all locks use it |
| Time slicing | Fair sharing among equal priorities | Determinism |
| Tick-based scheduling | Periodic task activation | Precision at high rates |
| Event-driven scheduling | Activation on an event | Bounded latency without analysis |

Priority inversion occurs when a low-priority task holds a resource needed by a high-priority task, and a medium-priority task preempts the low-priority task. The high-priority task waits for the medium-priority task, which may run for a long time. Priority inheritance or priority ceiling protocols bound the inversion.

A medical device with a control loop and a communications task must assign priorities by deadline, not by importance. The control loop has a short deadline and should have the highest priority; the communications task has a longer deadline and can wait. A design that gives the communications task the highest priority because it is "important" can miss the control deadline.

### 5.17 FPGA design and timing closure

An FPGA design is a circuit, not a program. The designer describes the hardware in a hardware description language (HDL) or a high-level synthesis (HLS) tool, and the tool maps it to logic, registers, and DSP slices. The result must meet timing closure: every path must settle within the clock period.

| Step | Output | Failure if skipped |
|---|---|---|
| Architecture | Block diagram, data flow | Unrealistic resource estimate |
| HDL or HLS | Register-transfer-level design | Simulation mismatch |
| Synthesis | Netlist, resource usage | Timing or area failure |
| Place and route | Physical layout | Timing closure failure |
| Static timing analysis | Worst-case path | Metastability or setup violation |
| Simulation | Functional and timing verification | Latent bug |
| On-board test | Real I/O and clock | Board-level failure |

Timing closure is not optional. A design that works in simulation but fails static timing analysis will fail in the field, often intermittently. The failure is temperature- and voltage-dependent, which makes it hard to reproduce.

An FPGA's determinism comes from its dedicated hardware paths. A pipelined multiply-add has a fixed latency of a few cycles, every time. The determinism is lost if the design shares a memory bus with a processor or a DMA, or if the clock is derived from an unstable source.

### 5.18 Toolchain and certification evidence

The toolchain is part of the medical device. A compiler that miscompiles a floating-point expression, a linker that places a variable at the wrong address, or a library that uses an undefined behaviour can produce a device that passes the bench and fails in the field.

| Evidence | What it shows | Typical artifact |
|---|---|---|
| Compiler version and flags | The build is reproducible | Build log |
| Static analysis | No undefined behaviour or MISRA violations | Analysis report |
| Unit tests | Each function behaves as specified | Test report |
| Integration tests | The modules work together | Test report |
| Timing analysis | The deadlines are met | Trace and WCET report |
| Traceability | Each requirement has a test | Matrix |
| Configuration management | The released binary matches the source | Hash and manifest |

IEC 62304 classifies software as A, B, or C according to the harm that a failure could cause. Class C requires the most rigorous evidence: architecture design, detailed design, unit verification, integration testing, and regression testing. The toolchain evidence is part of that package.

### 5.19 Power modes and DVFS

A processor's power has three components: dynamic power (switching), static power (leakage), and I/O power. Dynamic power scales with voltage squared and frequency; leakage scales with voltage and temperature. A design that reduces the clock to save power also reduces the leakage if it reduces the voltage, but the transition has a cost.

| Mode | Clock | Voltage | Wake time | Use |
|---|---|---|---|---|
| Run | Full | Full | 0 | Active processing |
| Sleep | Stopped | Full | µs | Short idle |
| Deep sleep | Stopped | Reduced | ms | Long idle |
| Hibernate | Stopped | Off | 100 ms | Storage |

Dynamic voltage and frequency scaling (DVFS) can save energy, but it introduces latency and jitter. A hard real-time function should run at a fixed frequency, or the scheduler must account for the transition. A design that scales the clock during a control loop can miss a deadline during the transition.

A medical device that runs on a battery should spend most of its time in the lowest power mode that preserves the required wake time. The wake time is part of the latency budget, not a separate concern.

### 5.20 A memory bandwidth example

A 1D convolution with 64 input channels, 64 output channels, 3 taps, and 1,000 output positions has:

\[
	ext{MACs} = 64 	imes 64 	imes 3 	imes 1000 = 12.3 	imes 10^6
\]

If each input and weight is 1 byte and each output is 1 byte, the minimum traffic is about 64 × 1,000 input bytes plus 64 × 64 × 3 weight bytes plus 64 × 1,000 output bytes, or 76 kB per inference. The arithmetic intensity is 12.3 million MACs / 76 kB = 162 MAC/byte, which is far above the ridge point of most MCUs. The workload is compute-bound, and a faster clock helps.

A fully connected layer with 1,000 inputs and 1,000 outputs has 1 million MACs and reads 1 MB of weights, for an intensity of 1 MAC/byte. On an MCU with 100 MB/s of memory bandwidth and 100 million MAC/s of compute, the memory can supply 100 million bytes/s, so the layer is memory-bound at 100 million MAC/s only if the intensity is 1. The two limits are equal at this point; a larger weight matrix would be memory-bound.

The example shows why model compression matters. Halving the weight precision halves the bytes and doubles the arithmetic intensity, moving a memory-bound layer toward the compute bound.

### 5.21 Ten processor review questions

1. What is the operation count per inference, and what is the memory traffic?
2. What is the deadline, and what is the jitter tolerance?
3. What is the worst-case execution time, not the average?
4. Where do the critical code and data live: TCM, cache, SRAM, or flash?
5. What is the DMA bandwidth, and does it starve the core?
6. What are the interrupt priorities, and is there priority inversion?
7. What is the clock source, and is the frequency fixed?
8. What is the energy per inference and per day?
9. What is the toolchain version, and is the build reproducible?
10. What is recorded so that a configuration change can be detected?

A review that answers all ten has covered the processor as a system. A review that answers only the first and the third has reviewed the algorithm, not the device.

### 5.23 Compiler optimisation and undefined behaviour

The compiler is part of the timing and the correctness. An optimisation level can change the execution time by a factor of two, and a compiler bug or an undefined behaviour can change the output.

| Issue | Effect | Mitigation |
|---|---|---|
| Aggressive optimisation | Faster, but may reorder or remove code | Volatile for hardware registers; barriers |
| Floating-point contraction | Changes rounding | Disable contraction for reproducibility |
| Undefined behaviour | Arbitrary results | Static analysis; MISRA C; tests |
| Library implementation | Different latency and accuracy | Pin the library version; test |
| Inline assembly | Fast but not portable | Document; isolate; test |
| Stack usage | Overflow in deep calls | Stack analysis; guard pages |

A medical device's build should be reproducible: the same source, compiler, flags, and libraries should produce the same binary. A build that depends on the developer's machine is not a controlled build. The binary's hash should be recorded with the release.

### 5.24 A note on benchmarks

A benchmark is a workload, not a promise. A processor that wins on a matrix-multiply benchmark may lose on a convolution with a different memory pattern. A processor that wins on a single-core benchmark may lose on a multi-core workload with synchronization.

The only benchmark that matters is the device's workload on the device's hardware, with the device's memory and power. A vendor benchmark is a starting point for a shortlist, not a decision.

### 5.22 Accelerator offload arithmetic

Moving a layer to an accelerator is not free. The transfer time is:

\[
T_{	ext{transfer}} = rac{	ext{bytes}}{	ext{bandwidth}}
\]

For a 120 kB weight matrix over a 10 MB/s bus, the transfer is 12 ms, which may exceed the layer's compute time on the main processor. The offload is worthwhile only if the compute time saved exceeds the transfer time plus the synchronization overhead.

| Layer | Compute on MCU | Transfer time | Compute on accelerator | Net gain |
|---|---:|---:|---:|---:|
| 2M MACs | 13.9 ms | 12 ms | 0.4 ms | Negative |
| 20M MACs | 139 ms | 12 ms | 4 ms | Positive |
| 200M MACs | 1,390 ms | 120 ms | 40 ms | Positive |

The table shows the threshold: an accelerator helps when the compute is large relative to the data movement. For a small model, the transfer dominates, and the main processor is the better home.

## 6. Worked example

### 6.1 The requirement

A wearable cardiac monitor runs a one-dimensional convolutional neural network (1D CNN) to classify each 10 s ECG window as normal or arrhythmia. The model has:

- 2.0 million MACs per inference.
- 120 kB of weights (int8).
- 40 kB of activation memory.
- A 5 ms deadline after the window is complete, for a closed-loop alert.
- A 10 s window, so the inference runs once every 10 s.

Two candidate platforms:

| Platform | Clock | Compute | Active power | Sleep power |
|---|---:|---|---:|---:|
| MCU | 480 MHz | 1 MAC/cycle peak | 50 mW | 0.5 mW |
| FPGA | 100 MHz | 64 DSP slices, 1 MAC/slice/cycle | 500 mW | 50 mW |

The MCU's measured utilization on this model is 30%; the FPGA's is 75%.

### 6.2 The MCU timing

The peak MAC rate is:

\[
1\ \text{MAC/cycle} \times 480 \times 10^6\ \text{cycles/s} = 480 \times 10^6\ \text{MAC/s}
\]

At 30% utilization, the effective rate is 144 million MAC/s. The average execution time is:

\[
T_{\text{MCU}} = \frac{2.0 \times 10^6}{144 \times 10^6} = 13.9\ \text{ms}
\]

The MCU misses the 5 ms deadline by 8.9 ms. The average is already over the deadline, so no worst-case analysis can save it.

### 6.3 The FPGA timing

The FPGA's peak MAC rate is:

\[
64 \times 100 \times 10^6 = 6.4 \times 10^9\ \text{MAC/s}
\]

At 75% utilization, the effective rate is 4.8 billion MAC/s. The compute time is:

\[
T_{\text{FPGA}} = \frac{2.0 \times 10^6}{4.8 \times 10^9} = 0.417\ \text{ms}
\]

The FPGA meets the 5 ms deadline with a margin of 4.58 ms. The margin is large enough to absorb the data transfer from the ADC and the post-processing.

The FPGA's advantage is not only the clock; it is the parallelism. Sixty-four MACs per cycle at 100 MHz outperform one MAC per cycle at 480 MHz by a factor of 13.3 in peak throughput, and by 33 in effective throughput because the FPGA's utilization is higher.

### 6.4 The energy

The MCU's energy per inference is:

\[
E_{\text{MCU}} = 50\ \text{mW} \times 13.9\ \text{ms} = 0.695\ \text{mJ}
\]

The FPGA's energy per inference is:

\[
E_{\text{FPGA}} = 500\ \text{mW} \times 0.417\ \text{ms} = 0.209\ \text{mJ}
\]

The FPGA uses less energy per inference despite drawing ten times the active power, because it finishes 33 times faster. This is the race-to-idle effect. The result depends on the duty cycle: if the device infers once per 10 s, the sleep energy dominates, and the comparison changes.

The MCU's sleep energy over the 10 s period is:

\[
E_{\text{sleep,MCU}} = 0.5\ \text{mW} \times (10 - 0.0139)\ \text{s} = 4.99\ \text{mJ}
\]

The FPGA's sleep energy is:

\[
E_{\text{sleep,FPGA}} = 50\ \text{mW} \times (10 - 0.000417)\ \text{s} = 500\ \text{mJ}
\]

The FPGA's sleep power is 100 times the MCU's, and over a 10 s period it dominates. The total energy per 10 s is 5.69 mJ for the MCU and 500 mJ for the FPGA. The MCU is the better choice for this duty cycle, despite missing the 5 ms deadline.

The example exposes the real trade: the FPGA meets the deadline but costs 88 times the energy over the period; the MCU meets the energy budget but misses the deadline. The design must change one of the requirements: a larger MCU, a smaller model, or a relaxed deadline.

### 6.5 Diagnostic 1: the average hides the cache jitter

The MCU's measured average is 13.9 ms, but the p99 is 24 ms and the maximum is 31 ms. The cache is 32 kB, and the model's weights are 120 kB, so the weights do not fit; every inference streams them from flash, and the flash's wait states add stalls that depend on the access pattern.

The diagnostic is to run the model with the weights in a smaller format or with a subset that fits the cache, and compare the distributions. If the p99 falls, the cache is the cause. A second diagnostic is to count cache misses with a performance counter; if the miss rate is high and correlates with the latency, the diagnosis is confirmed.

The design flaw is not the average; it is the assumption that the average is representative. A hard deadline needs the tail, and the tail is set by memory, not by arithmetic.

### 6.6 Diagnostic 2: the DMA steals the bus

The FPGA design meets the deadline in isolation, but when the radio's DMA is active, the inference time rises from 0.417 ms to 0.9 ms. The DMA and the accelerator share the memory bus, and the radio's bursts have priority.

The diagnostic is to run the inference with the radio transmitting and compare the latency. If the latency rises, the bus is the cause. A second diagnostic is to measure the bus utilization with a performance counter; if the DMA consumes more than the reserved bandwidth, the arbitration is wrong.

The design flaw is the assumption that the accelerator has the memory to itself. A partitioned design must budget the bus bandwidth, not only the compute.

### 6.7 Vary one assumption: the model shrinks to 500,000 MACs

Change only the model size, from 2.0 million to 500,000 MACs, by pruning and quantization. The MCU's execution time becomes:

\[
T_{\text{MCU}} = \frac{0.5 \times 10^6}{144 \times 10^6} = 3.47\ \text{ms}
\]

The MCU now meets the 5 ms deadline with a 1.53 ms margin. The FPGA's time becomes 0.104 ms, and its energy per inference falls to 0.052 mJ, but its sleep energy still dominates the 10 s period.

The consequence is that the architecture choice flips with a 4× reduction in model size. A design that starts with the architecture before the model is optimized may choose an FPGA unnecessarily. The right order is to optimize the model, then choose the architecture, then verify the timing.

### 6.8 What the example demonstrates

The arithmetic closes: 13.9 ms for the MCU, 0.417 ms for the FPGA, 0.695 mJ versus 0.209 mJ per inference, and 5.69 mJ versus 500 mJ per 10 s period. The two diagnostics reveal different flaws: one is a cache-induced tail, the other is a bus contention. The sensitivity analysis shows that a 4× model reduction moves the MCU from failing to passing.

The general method is to compute the average, measure the tail, budget the bus, and compare the energy over the duty cycle, not only per inference. A processor that wins on the benchmark may lose on the battery.

### 6.9 A second design: the closed-loop pump

A closed-loop insulin pump runs a control loop every 100 ms. The loop reads a continuous glucose monitor, computes a dose, and commands a stepper motor. The operations are small (about 10,000 MACs), but the deadline is hard and the jitter tolerance is 10 ms.

| Platform | Compute time | Worst-case jitter | Meets 100 ms? |
|---|---:|---:|---|
| MCU at 100 MHz | 0.3 ms | 0.05 ms | Yes, with large margin |
| SoC with Linux | 0.1 ms | 20 ms | Marginal; jitter risk |
| FPGA | 0.01 ms | 0.001 ms | Yes, overkill |

The MCU is the right choice. The SoC's average is faster, but its jitter is twice the tolerance; a single scheduler delay could miss the deadline. The FPGA is deterministic but unnecessary; its power and complexity are not justified by a 10,000-MAC workload.

The example shows that the architecture follows from the jitter tolerance, not only from the average. A soft real-time workload can use a rich OS; a hard real-time workload needs a deterministic path.

### 6.10 Verification matrix

| Test | Condition | Metric | Acceptance |
|---|---|---|---|
| Average latency | Nominal workload | Mean over 1,000 runs | < 5 ms |
| Tail latency | Nominal workload | p99.9 | < 5 ms |
| Worst case | Maximum workload | Maximum over 10,000 runs | < 5 ms |
| Cache sensitivity | Weights in cache versus flash | p99 change | < 10% |
| DMA contention | Radio active | Latency change | < 10% |
| Thermal | 40 °C ambient | Latency change | < 5% |
| Energy | One inference | J per inference | < 1 mJ |
| Energy per day | Duty cycle | J per day | < 10% of battery |
| Brown-out | Supply drop | Reset and recovery | No invalid output |
| Firmware update | Interrupted update | Rollback | Device recovers |

The matrix is a minimum. A safety-critical function needs a fault-injection test and a trace of the execution, not only a timing measurement.

### 6.11 The latency distribution

| Percentile | MCU (ms) | FPGA (ms) |
|---|---:|---:|
| Mean | 13.9 | 0.42 |
| p50 | 13.5 | 0.41 |
| p90 | 16.0 | 0.44 |
| p99 | 24.0 | 0.48 |
| p99.9 | 29.0 | 0.52 |
| Maximum | 31.0 | 0.55 |

The MCU's tail is 2.2 times its mean; the FPGA's tail is 1.3 times its mean. The difference is the memory hierarchy. A hard deadline of 5 ms is met by the FPGA at every percentile and by the MCU at none. A deadline of 35 ms is met by the MCU at every measured percentile, but the unmeasured tail could still exceed it.

The table is the evidence a reviewer needs. A single average would hide the tail, and the tail is where the deadline is missed.

### 6.12 What if the deadline cannot be met?

If the 5 ms deadline is fixed and the model cannot be reduced, the options are:

1. **A larger MCU** with a higher clock, a wider SIMD unit, or a DSP extension. This may meet the deadline, at higher power.
2. **An FPGA or an NPU** with enough parallelism. This meets the deadline, at higher design effort and sleep power.
3. **A relaxed deadline** if the clinical function can tolerate it. This changes the requirement, not the design.
4. **A smaller model** with pruning, quantization, or distillation. This changes the algorithm and requires revalidation.
5. **An earlier window** so that the inference starts before the window ends, using a streaming classifier. This changes the algorithm and the latency profile.

The choice is a system decision, not a processor decision. A team that argues only about the processor has framed the problem too narrowly.

### 6.13 A note on the sleep-power assumption

The 0.5 mW MCU sleep power and the 50 mW FPGA sleep power are assumptions. A real MCU in deep sleep may draw 1–10 µW; an FPGA may draw 100 mW or more, depending on the device and the design. The comparison is sensitive to these numbers.

If the MCU sleeps at 10 µW, its sleep energy over 10 s is 0.1 mJ, and the total is 0.79 mJ. If the FPGA sleeps at 100 mW, its sleep energy is 1 J, and the total is 1.0002 J. The ratio becomes 1,266:1. The architecture conclusion is even stronger.

The lesson is to measure the sleep power on the actual hardware, in the actual power mode, with the actual peripherals enabled. A datasheet's "sleep current" may exclude the real-time clock, the brown-out detector, and the RAM retention, which are often the dominant terms.
## 7. Code, query, or calculation

The script reproduces the timing, energy, and transfer calculations in Section 6. It makes the duty cycle explicit: the energy per inference is not the energy per period, and the comparison between architectures depends on which one the device pays.

**Assumptions.** 2.0 million MACs per inference, a 480 MHz MCU with 30% utilization, a 100 MHz FPGA with 64 DSP slices and 75% utilization, 50 mW active and 0.5 mW sleep for the MCU, 500 mW active and 50 mW sleep for the FPGA, and a 10 s period.

```python
macs = 2.0e6
mcu_clock = 480e6
mcu_util = 0.30
fpga_slices = 64
fpga_clock = 100e6
fpga_util = 0.75
mcu_power = 50e-3
mcu_sleep = 0.5e-3
fpga_power = 500e-3
fpga_sleep = 50e-3
period = 10.0

mcu_rate = mcu_clock * mcu_util
fpga_rate = fpga_slices * fpga_clock * fpga_util
t_mcu = macs / mcu_rate
t_fpga = macs / fpga_rate
e_mcu = mcu_power * t_mcu
e_fpga = fpga_power * t_fpga
sleep_mcu = mcu_sleep * (period - t_mcu)
sleep_fpga = fpga_sleep * (period - t_fpga)
total_mcu = e_mcu + sleep_mcu
total_fpga = e_fpga + sleep_fpga
t_mcu_small = 0.5e6 / mcu_rate
transfer = 120e3 / 10e6

print("MCU")
print(f"  effective MAC/s      {mcu_rate/1e6:8.1f} M")
print(f"  compute time         {t_mcu*1e3:8.2f} ms")
print(f"  energy per inference {e_mcu*1e3:8.3f} mJ")
print(f"  sleep energy / 10 s  {sleep_mcu*1e3:8.2f} mJ")
print(f"  total / 10 s         {total_mcu*1e3:8.2f} mJ")
print()
print("FPGA")
print(f"  effective MAC/s      {fpga_rate/1e9:8.2f} G")
print(f"  compute time         {t_fpga*1e3:8.3f} ms")
print(f"  energy per inference {e_fpga*1e3:8.3f} mJ")
print(f"  sleep energy / 10 s  {sleep_fpga*1e3:8.2f} mJ")
print(f"  total / 10 s         {total_fpga*1e3:8.2f} mJ")
print()
print(f"MCU with 500k MACs: {t_mcu_small*1e3:.2f} ms")
print(f"transfer 120 kB at 10 MB/s: {transfer*1e3:.1f} ms")
print(f"energy ratio FPGA/MCU per inference: {e_fpga/e_mcu:.3f}")
print(f"total ratio FPGA/MCU: {total_fpga/total_mcu:.1f}")
```

**Expected output.**

```text
MCU
  effective MAC/s         144.0 M
  compute time            13.89 ms
  energy per inference    0.694 mJ
  sleep energy / 10 s     4.99 mJ
  total / 10 s            5.69 mJ

FPGA
  effective MAC/s          4.80 G
  compute time            0.417 ms
  energy per inference    0.208 mJ
  sleep energy / 10 s   499.98 mJ
  total / 10 s          500.19 mJ

MCU with 500k MACs: 3.47 ms
transfer 120 kB at 10 MB/s: 12.0 ms
energy ratio FPGA/MCU per inference: 0.300
total ratio FPGA/MCU: 87.9
```

The output shows the central trade. The FPGA uses 0.300 times the MCU's energy per inference, because it finishes 33 times faster at 10 times the power. Over a 10 s period, however, the FPGA uses 87.9 times the MCU's energy, because its sleep power is 100 times higher. The architecture that wins per inference loses per period.

The model-size sensitivity is equally important. At 500,000 MACs, the MCU computes in 3.47 ms and meets the 5 ms deadline. A 4× reduction in model size, achievable with pruning and quantization, changes the architecture decision.

The transfer calculation shows why offload is not free. A 120 kB weight matrix over a 10 MB/s bus takes 12 ms, which is longer than the MCU's entire inference at the smaller model size. The accelerator must be fed with data that is already on-chip, or the transfer must be overlapped with compute.

**A note on utilization.** The 30% MCU utilization is an assumption, not a measurement. A real 1D CNN on a Cortex-M4 with no DSP extension may achieve 10–20%; with DSP instructions and a good library, 40–60%. The conclusion is sensitive to the number: at 10% utilization, the MCU takes 41.7 ms; at 60%, 6.9 ms, which is close to the deadline. The utilization must be measured on the target, not assumed.

## 8. Data, standards, and regulatory context

The processor is governed by the software lifecycle standard, the alarm standard, the risk management standard, and the cybersecurity guidance. The table separates the instruments.

| Instrument | Scope | Status | What it constrains |
|---|---|---|---|
| IEC 62304:2006+AMD1:2015 | Medical device software lifecycle | Consensus standard | Software safety class, architecture, verification, configuration |
| IEC 60601-1-8:2006+AMD2:2020 | Alarm systems | Collateral standard | Alarm condition delay, signal generation delay, disclosure |
| ISO 14971:2019 | Risk management | Consensus standard | Hazard analysis for latency, jitter, and software failure |
| IEC 81001-5-1:2021 | Health software security | Consensus standard | Secure lifecycle, threat model, SBOM |
| FDA premarket cybersecurity guidance | Device cybersecurity | Guidance enforced through review | Threat model, testing, SBOM, update plan |
| IEC 62366-1:2015+AMD1:2020 | Usability engineering | Consensus standard | Use-related risk, alarms, mode awareness |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General safety | Consensus standard | Electrical safety, isolation, thermal |
| MISRA C:2012 | C coding standard | Industry convention | Coding rules for safety-related C |
| ISO 26262 | Automotive functional safety | Not medical; informative analogy | Safety integrity concepts |

### 8.1 IEC 62304 software safety classes

IEC 62304 classifies software as A, B, or C according to the harm that a failure could cause:

| Class | Definition | Example | Evidence |
|---|---|---|---|
| A | No injury or damage to health is possible | Logging, non-clinical display | Basic lifecycle |
| B | Non-serious injury is possible | Incorrect trend display | Architecture, unit verification |
| C | Death or serious injury is possible | Closed-loop drug delivery | Detailed design, unit and integration verification |

The class is determined by the risk analysis, not by the processor. A class C function requires the most rigorous evidence, including architecture design, detailed design, unit verification, integration testing, and regression testing. A processor with caches and an operating system makes the evidence harder, because the timing and memory behaviour are less predictable.

The standard does not prescribe a language, an operating system, or a processor. It prescribes the process and the evidence. A class C function can be implemented on an MCU, a DSP, or an FPGA, as long as the evidence supports the claim.

### 8.2 IEC 60601-1-8 and latency

IEC 60601-1-8:2006+AMD2:2020 clause 6.4.1 requires disclosure when the sum of the maximum alarm condition delay and the maximum alarm signal generation delay exceeds 10 s, or when the sum of the mean delays exceeds 5 s. It does not impose a hard 10 s limit. Particular standards, such as those for ventilators or infusion pumps, may impose tighter limits.

For a processor, the alarm signal generation delay includes the time from the alarm condition to the signal output: the firmware's detection, the display update, and the network transmission. A processor that takes 20 ms to update the display contributes 20 ms to the delay; a processor that takes 2 s to run the detection algorithm contributes 2 s. The budget must include both.

### 8.3 Cybersecurity and firmware updates

The FDA's premarket cybersecurity guidance and IEC 81001-5-1 require a threat model, security testing, a software bill of materials (SBOM), and a plan for security updates. For a processor, this means:

- A secure boot chain that verifies the firmware image.
- Signed updates with rollback protection.
- A process for vulnerability monitoring and patching.
- A record of the firmware version in the data.

A device that cannot be updated is a device that cannot be patched. A device that updates without verification is a device that can be compromised. The processor's boot ROM and memory protection unit determine which is possible.

### 8.4 What the standards do not cover

The standards do not specify:

- The processor architecture.
- The clock frequency.
- The memory hierarchy.
- The scheduling algorithm.
- The compiler or the toolchain.
- The latency distribution.

Those are design decisions that the manufacturer must justify. A device can meet every clause and still miss a deadline, because the clauses test the process, not the timing.

### 8.5 The data implications

The processor's configuration is part of the measurement. The clock, the firmware version, the model version, and the scheduling policy must be recorded with the data. A change in any of them can change the latency, the throughput, and the output.

A practical rule: if the data dictionary does not contain the firmware version and the model version, the dataset cannot be reproduced. The rule is simple, and it is violated often.

### 8.6 What the standards imply for the data record

| Standard | Record | Why |
|---|---|---|
| IEC 62304 | Software class, version, build hash, test results | Reproduce the evidence |
| IEC 60601-1-8 | Alarm delay, signal generation delay | Reproduce the alarm claim |
| ISO 14971 | Hazard analysis, risk controls | Link latency to harm |
| IEC 81001-5-1 | SBOM, threat model, patch status | Security lifecycle |
| FDA cybersecurity | Update plan, vulnerability monitoring | Premarket and post-market |
| IEC 62366-1 | Use-related risk, summative test | Human factors |

The record is the same information the data team needs for provenance. A team that records the firmware, model, and configuration for its own analysis is also assembling regulatory evidence.

## 9. Industry practice and case evidence

### 9.1 Cortex-M dominates deterministic control

Arm's Cortex-M series is the most common architecture for medical control loops. The Cortex-M4 technical reference manual states a maximum 12-cycle interrupt latency from asserting the interrupt to the first instruction of the ISR, assuming zero-wait-state memory, with the stacking of registers happening within those 12 cycles. Return from interrupt is also 12 cycles, and tail-chaining is 6 cycles. The latency is deterministic at the core level, but it is extended by flash wait states, bus contention, and higher-priority nesting.

The practical consequence is that a Cortex-M4 at 100 MHz can service an interrupt in about 120 ns at zero wait states, or several hundred nanoseconds with flash wait states. A control loop at 10 kHz has a 100 µs period, so the interrupt latency is a small fraction of the budget. A control loop at 1 MHz has a 1 µs period, and the latency becomes significant.

**Documented fact:** the 12-cycle, 12-cycle, and 6-cycle figures are from the Arm Cortex-M4 Technical Reference Manual and Arm's published guidance. **Inference:** the system-level latency depends on the memory and bus, which must be measured.

### 9.2 DSPs remain important for filtering and beamforming

A DSP's advantage is the multiply-accumulate throughput and the addressing modes that support circular buffers and bit-reversed addressing. A Texas Instruments C6000 DSP can execute multiple instructions per cycle and is used in ultrasound beamforming, where the channel count and the sample rate make an MCU inadequate. A C2000 DSP is used in motor control and power conversion, where the pulse-width-modulation peripherals and the fast interrupt response matter.

The trade is the toolchain and the power. A DSP's compiler and libraries are specialised, and the power is higher than an MCU's. The choice is justified when the arithmetic intensity is high and the latency is tight.

### 9.3 FPGAs in imaging and ultrasound

An FPGA's advantage is spatial parallelism and deterministic latency. In ultrasound, the beamformer must combine hundreds of channels at tens of millions of samples per second; an FPGA can do this with a fixed pipeline latency, while a processor would need a cluster. In CT and photon-counting detectors, the FPGA handles the high-rate detector data before it reaches the reconstruction computer.

The trade is the design effort, the power, and the cost. An FPGA design takes months, not weeks, and the verification is more complex than software. The choice is justified when the throughput or the determinism cannot be achieved otherwise.

### 9.4 SoCs and edge AI

An SoC with a CPU, a GPU, and a neural processing unit (NPU) is used for imaging, edge AI, and connectivity. The advantage is the software stack: Linux, libraries, and frameworks. The disadvantage is the nondeterminism: the scheduler, the cache, the GPU driver, and the power management all add latency that is hard to bound.

The ROS 2 documentation describes the sources of nondeterminism across the stack: the operating system, the middleware, the executor, and the application. It recommends avoiding page faults, dynamic allocation, and blocking synchronisation in real-time paths, and using a real-time kernel, memory locking, and priority configuration. The documentation is honest that these measures reduce, but do not eliminate, jitter.

### 9.5 Recalls that involved the processor

The 2025 recalls of Virtual Incision's MIRA and Quantum Surgical's EPIONE involved software issues that could interrupt or lag motion, or misalign the arm relative to the planned trajectory. The CMR Surgical Versius console recall involved Secure Boot not being enabled. These are processor- and firmware-level failures, not sensor failures.

The lesson is that the processor's software is part of the safety chain. A latency bug, a coordinate transform bug, or a boot integrity bug can have procedural consequences even when the hardware is functioning.

**Documented fact:** the recalls are listed in the FDA recall database and manufacturer notices. **Inference:** the root-cause analyses are not public; the descriptions are limited to the recall notices.

### 9.6 What good processor practice looks like

Teams that avoid processor surprises tend to:

- Measure the latency distribution on the target, not the average on a development board.
- Bound the cache, DMA, and interrupt jitter explicitly.
- Record the firmware, model, and toolchain versions with the data.
- Test the worst-case workload and the worst-case environment.
- Keep the hard real-time path free of dynamic allocation and blocking calls.
- Verify the secure boot and update path.
- Re-run the timing tests after any firmware or model change.

The practices are ordinary. The failures they prevent are not.

### 9.7 A note on evaluation boards and development kits

A development kit is not the product. Its power supply, clock, memory, and thermal design differ from the product's. A latency measured on the kit is an upper bound or a lower bound, depending on which differences dominate.

The same applies to the vendor's benchmark. It measures the vendor's workload, not yours. The only valid measurement is the product's workload on the product's hardware, with the product's power and thermal conditions.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Deadline miss | p99 latency above the deadline | Workload exceeds throughput | Late alarm or control | Faster processor; smaller model; partitioning |
| Cache jitter | p99 much higher than mean | Weight or data exceeds cache | Missed deadline | TCM; cache locking; smaller model |
| DMA contention | Latency rises when radio is active | Shared bus bandwidth | Missed deadline | Bandwidth reservation; separate bus |
| Priority inversion | High-priority task waits | Low-priority task holds a lock | Unbounded delay | Priority inheritance; ceiling protocol |
| Flash wait states | Latency depends on code placement | Instruction fetch stalls | Jitter | Prefetch; copy to RAM |
| Clock scaling | Latency changes with power mode | DVFS transition | Missed deadline | Lock frequency |
| Thermal throttling | Latency rises after warm-up | Clock reduction | Missed deadline | Thermal design; workload limiting |
| Watchdog reset loop | Repeated resets | Firmware fails to service watchdog | Device unavailable | Reset cause logging; fail-safe |
| Brown-out | Reset during operation | Supply drop | Lost state; invalid output | Supply monitor; non-volatile state |
| Firmware update failure | Device does not boot | Incomplete write; bad signature | Bricked device | A/B partitions; rollback |
| Secure boot disabled | Update accepted without signature | Misconfiguration | Compromised device | Verify boot configuration |
| Toolchain regression | Behaviour changes after rebuild | Compiler bug; undefined behaviour | Wrong output | Pin toolchain; regression tests |
| Memory leak | Latency grows over time | Dynamic allocation in a loop | eventual failure | Static allocation; monitoring |
| Stack overflow | Random crashes | Deep call chain; large local arrays | Device reset | Stack analysis; guard pages |

The most dangerous failure is the one that is intermittent. A cache-related deadline miss occurs once in a thousand inferences, passes the bench test, and appears in the field as a rare alarm failure. The mitigation is to measure the tail, not the mean, and to test the worst-case workload.

### 10.1 Ten red flags in processor data

1. A latency distribution with a long tail that is not explained.
2. A p99 that changes when the radio or display is active.
3. A firmware version that differs between units in the same study.
4. A model version that is not recorded with the output.
5. A clock frequency that varies with temperature or battery state.
6. A device that resets without a logged cause.
7. A watchdog that is serviced from the wrong task.
8. A build that is not reproducible from the source.
9. A secure boot configuration that is disabled in production.
10. A sleep current that is higher than the datasheet's typical.

Each red flag is a prompt to measure, not a conclusion.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The operation count is accurate | Sizing the processor | Compiler optimisations; sparsity | Profile on the target |
| The utilization is representative | Latency estimate | Memory stalls; dependencies | Measure with a performance counter |
| The clock is fixed | Deterministic timing | DVFS; thermal throttling | Lock frequency; log temperature |
| The critical data fits in TCM or cache | Bounded latency | Model or data growth | Measure cache misses |
| The DMA bandwidth is reserved | No bus starvation | Concurrent transfers | Measure latency under load |
| The interrupt priorities reflect deadlines | No priority inversion | Misassigned priorities | Analyze worst-case nesting |
| The RTOS bounds preemption | Deterministic scheduling | Driver latency; interrupts disabled | Measure with a trace |
| The toolchain is reproducible | The binary matches the source | Compiler update; flags | Rebuild and compare hashes |
| The model version matches the firmware | Consistent behaviour | Independent updates | Version check at startup |
| The thermal design holds | No throttling | High ambient; sustained load | Thermal chamber test |
| The battery supports the peak current | No brown-out | High transmit current | Measure supply during transmit |
| The update path is atomic | No bricked device | Power loss during update | Fault-injection test |

The dominant assumption is that the average is representative. It is not. A hard deadline is met or missed at the tail, and the tail is set by memory and scheduling, not by the arithmetic average.

A second limit is that the processor model is single-core. A multi-core processor adds cache coherence, shared memory, and inter-core communication, each with its own jitter. The analysis of a multi-core system is substantially harder, and the certification evidence is larger.

A third limit is that the workload is assumed to be static. A model that adapts its computation to the input, or a scheduler that changes priorities at runtime, has a workload-dependent timing that must be bounded for every mode. The worst-case mode is the one that must be tested.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no measurement, only a datasheet?
- Which assumption was true on the development kit but false on the product?
- Which assumption depends on the memory hierarchy or the bus?
- Which assumption is encoded in a build flag or a register?
- Which assumption is specific to the validation workload?
- Which assumption is monitored in the field, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Estimate MACs (recall, 20 minutes)

A 1D CNN has three convolution layers: (32, 32, 5, 500), (32, 64, 5, 250), and (64, 64, 3, 125), where the tuple is (input channels, output channels, kernel, output length). Compute the MACs per layer and the total.

**Deliverable:** a table with the per-layer and total MACs.

**Acceptance criteria:** correct formula; units; total consistent with the layers.

**Time:** 20 minutes.

### Exercise 2. Compute latency (application, 30 minutes)

A 50 million MAC model runs on a 200 MHz MCU with 1 MAC/cycle at 40% utilization. Compute the average execution time. Then compute the time on an FPGA with 128 DSP slices at 150 MHz and 70% utilization.

**Deliverable:** two calculations and a comparison.

**Acceptance criteria:** correct effective rates; units in milliseconds; comparison states the ratio.

**Time:** 30 minutes.

### Exercise 3. Compute energy per period (application, 30 minutes)

An MCU draws 40 mW active and 0.4 mW sleep. It runs a 20 ms inference once per 5 s. Compute the energy per inference, the sleep energy per period, and the total per day. Repeat for an FPGA drawing 400 mW active and 40 mW sleep with a 1 ms inference.

**Deliverable:** a table with both platforms.

**Acceptance criteria:** active and sleep energy separated; per-day total; units in mJ and J.

**Time:** 30 minutes.

### Exercise 4. Diagnose a tail (critique, 30 minutes)

An MCU's mean inference time is 8 ms, but the p99 is 22 ms. The model weights are 200 kB and the cache is 32 kB. Explain the likely cause, propose two mitigations, and state their costs.

**Deliverable:** a diagnosis and two mitigations.

**Acceptance criteria:** cache mechanism correct; mitigations distinguish model compression from memory placement; costs stated.

**Time:** 30 minutes.

### Exercise 5. Choose an architecture (design, 35 minutes)

A ventilator's flow-control loop runs at 1 kHz with a 200 µs deadline and a 20 µs jitter tolerance. The algorithm is a PID plus a 20-tap filter. Choose an architecture and justify it against two alternatives.

**Deliverable:** a one-page decision record.

**Acceptance criteria:** workload and deadline quantified; alternatives rejected for stated reasons; evidence burden considered.

**Time:** 35 minutes.

### Exercise 6. Design a partition (design, 40 minutes)

A wearable runs a 5 million MAC classifier and a 1 kHz control loop. Partition the workload across an MCU and an accelerator, estimate the transfer time, and show that the partition meets the deadlines.

**Deliverable:** a block diagram and a timing budget.

**Acceptance criteria:** transfer time included; both deadlines met; the partition's failure modes named.

**Time:** 40 minutes.

### Exercise 7. Write a timing verification plan (design, 35 minutes)

Write a plan to verify the latency of the classifier in Section 6. Include the workload, the environment, the measurement method, the number of runs, the percentiles, and the acceptance criteria.

**Deliverable:** a test plan with at least 8 rows.

**Acceptance criteria:** worst-case workload; thermal and DMA conditions; tail percentiles; quantitative criteria.

**Time:** 35 minutes.

### Exercise 8. Evaluate a "real-time" claim (critique, 25 minutes)

A vendor says its SoC is "real-time" because it runs a real-time kernel. Write five questions that would test the claim for a 1 kHz control loop.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover latency distribution, jitter, scheduling, drivers, and thermal; no question is answerable by the kernel name alone.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Workload sizing | MAC table |
| 2 | Latency estimate | Two calculations |
| 3 | Energy | Per-period table |
| 4 | Tail diagnosis | Cache analysis |
| 5 | Architecture choice | Decision record |
| 6 | Partitioning | Diagram and budget |
| 7 | Verification | Test plan |
| 8 | Claim evaluation | Five questions |

## 13. Self-check answers

**Answer 1 (Exercise 1).** Layer 1: 32 × 32 × 5 × 500 = 2.56 million MACs. Layer 2: 32 × 64 × 5 × 250 = 2.56 million. Layer 3: 64 × 64 × 3 × 125 = 1.536 million. Total = 6.656 million MACs.

**Answer 2 (Exercise 2).** MCU effective rate = 200e6 × 0.4 = 80 million MAC/s; time = 50e6 / 80e6 = 0.625 s. FPGA effective rate = 128 × 150e6 × 0.7 = 13.44 billion MAC/s; time = 50e6 / 13.44e9 = 3.72 ms. The FPGA is 168 times faster.

**Answer 3 (Exercise 3).** MCU: energy per inference = 40 mW × 20 ms = 0.8 mJ. Sleep = 0.4 mW × 4.98 s = 1.992 mJ per period. Total per period = 2.792 mJ; per day at 17,280 periods = 48.2 J. FPGA: 400 mW × 1 ms = 0.4 mJ; sleep = 40 mW × 4.999 s = 199.96 mJ; total per period = 200.36 mJ; per day = 3,462 J. The FPGA uses 72 times the energy per day.

**Answer 4 (Exercise 4).** The 200 kB weights exceed the 32 kB cache, so each inference streams weights from flash and incurs cache misses whose timing depends on the access pattern and the bus. Mitigations: quantize or prune the model to fit the cache, which costs accuracy and requires revalidation; or place the weights in TCM or a locked cache region, which costs memory and may not fit. A third option is to prefetch the weights with DMA, which costs bus bandwidth and adds complexity.

**Answer 5 (Exercise 5).** The 1 kHz loop has a 1 ms period and a 200 µs deadline; a PID plus a 20-tap filter is about 100 MACs per sample, or 100,000 MAC/s, which is trivial for an MCU. An MCU at 100 MHz with a 12-cycle interrupt latency has a worst-case latency of about 0.12 µs at zero wait states, plus the ISR time, well within the 20 µs jitter tolerance. A Linux SoC has a jitter of milliseconds and is unsuitable. An FPGA is deterministic but unnecessary and more expensive to verify. The MCU is the right choice.

**Answer 6 (Exercise 6).** Put the control loop on the MCU and the classifier on the accelerator. The control loop needs 1,000 samples per second and about 100 MACs per sample, or 100,000 MAC/s, which the MCU handles with margin. The classifier needs 5 million MACs per inference; at 144 million MAC/s effective, it takes 34.7 ms, which is acceptable if the inference runs once per second. The transfer of the classifier's input (for example, 10 kB) over a 10 MB/s bus takes 1 ms, which is small compared with 34.7 ms. The partition meets both deadlines; the risk is bus contention during the control loop, which must be bounded by priority and bandwidth reservation.

**Answer 7 (Exercise 7).** The plan: nominal workload, 1,000 runs, mean and p99; maximum workload, 10,000 runs, p99.9 and maximum; radio active, 1,000 runs; 40 °C ambient, 1,000 runs; brown-out injection; firmware update interruption. Acceptance: maximum < 5 ms; p99.9 < 5 ms; no invalid output. Measurement with a hardware timer and a trace unit, not a software timestamp.

**Answer 8 (Exercise 8).** Questions: (1) What is the measured p99.9 and maximum latency for a 1 kHz loop? (2) What is the jitter contribution of the kernel, the drivers, and the network stack? (3) Are the critical threads locked in memory and scheduled with real-time priorities? (4) What happens during a thermal throttle or a power-mode transition? (5) What evidence is provided for the worst-case, not the average? A kernel name is not evidence.

**Rubric for the design exercises.** A strong design quantifies the workload, the deadline, the jitter, the energy, and the evidence. A weak design names a processor and assumes it will be fast enough.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 1: total MACs | 6.656 million | ±0.01 million |
| Exercise 2: MCU time | 0.625 s | ±0.01 s |
| Exercise 2: FPGA time | 3.72 ms | ±0.05 ms |
| Exercise 3: MCU per day | 48.2 J | ±0.5 J |
| Exercise 3: FPGA per day | 3,462 J | ±20 J |
| Exercise 6: classifier time | 34.7 ms | ±0.5 ms |
| Exercise 6: transfer time | 1.0 ms | ±0.1 ms |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

## 14. Interview questions

**1. When would you choose an FPGA over an MCU?**
A strong answer names a throughput or determinism requirement that the MCU cannot meet, and acknowledges the design effort and power cost. A weak answer says the FPGA is faster.

**2. What is the difference between average and worst-case execution time?**
A strong answer explains that the worst case includes cache, DMA, interrupt, and OS stalls, and that a hard deadline needs the tail. A weak answer treats the average as the worst case.

**3. Why do caches create jitter?**
A strong answer explains cache misses, the dependency on access pattern, and the mitigation of TCM or cache locking. A weak answer says caches are fast.

**4. What is priority inversion, and how do you prevent it?**
A strong answer gives the three-task scenario and names priority inheritance or ceiling protocols. A weak answer confuses it with starvation.

**5. How do you compute energy per inference?**
A strong answer multiplies active power by active time and adds sleep power times idle time; it distinguishes per-inference from per-period energy. A weak answer uses power alone.

**6. What does IEC 62304 class C require?**
A strong answer names architecture, detailed design, unit verification, integration testing, and traceability. A weak answer says "more documentation".

**7. Why is a real-time kernel not a guarantee of real-time behaviour?**
A strong answer names drivers, interrupts, cache, memory, and application design as remaining sources of jitter. A weak answer trusts the kernel.

**8. How do you verify a timing claim?**
A strong answer measures the distribution on the target under worst-case workload and environment, using a hardware timer or trace. A weak answer quotes a benchmark.

**9. What is the roofline model?**
A strong answer explains the compute and memory bounds and the ridge point, and uses it to decide whether a faster clock helps. A weak answer has not heard of it.

**10. What would make you reject a processor choice?**
A strong answer names a missed deadline, an unbounded jitter, an unavailable toolchain, or a certification evidence gap. A weak answer rejects on clock speed alone.

**11. How does DMA affect determinism?**
A strong answer explains bus contention and the need for bandwidth reservation. A weak answer says DMA is faster than the CPU.

**12. What do you record with the data to make the firmware traceable?**
A strong answer names firmware version, model version, toolchain, configuration, and build hash. A weak answer names only the device model.

## 15. Key terms

- **Accelerator** — hardware specialised for a workload.
- **Arithmetic intensity** — operations per byte of memory traffic.
- **Brown-out** — supply drop that can reset the processor.
- **Cache** — fast memory holding recently used data.
- **DMA** — direct memory access.
- **DSP** — digital signal processor.
- **DVFS** — dynamic voltage and frequency scaling.
- **Energy per inference** — joules per inference.
- **ENOB** — effective number of bits (from Topic 04).
- **FPGA** — field-programmable gate array.
- **Hard deadline** — deadline whose miss is a failure.
- **Interrupt latency** — time from interrupt to ISR execution.
- **IPC** — instructions per cycle.
- **Jitter** — variation in latency.
- **Leakage power** — power consumed when idle.
- **MAC** — multiply-accumulate.
- **MCU** — microcontroller unit.
- **MISRA C** — coding standard for safety-related C.
- **NPU** — neural processing unit.
- **Priority inversion** — low-priority task blocks a high-priority one.
- **Roofline** — performance model bounded by compute or memory.
- **RTOS** — real-time operating system.
- **SBOM** — software bill of materials.
- **SoC** — system on chip.
- **Tail-chaining** — back-to-back interrupt handling.
- **TCM** — tightly coupled memory.
- **Throughput** — operations per second.
- **Toolchain** — compiler, linker, debugger, libraries.
- **Utilization** — fraction of peak performance achieved.
- **WCET** — worst-case execution time.
- **Build hash** — digest of the released binary.
- **Cache locking** — pinning data in cache to bound latency.
- **Duty cycle** — fraction of time a function is active.
- **Interrupt priority** — ranking that determines which ISR runs first.
- **Memory map** — assignment of code and data to memory regions.
- **Race condition** — behaviour that depends on timing.
- **Secure boot** — verification of the firmware image at startup.
- **Static analysis** — analysis of code without execution.
- **Watchdog** — timer that resets the processor if not serviced.
- **Windowed watchdog** — watchdog that requires servicing within a window.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Architecture and toolchain**

1. Arm, *Cortex-M4 Technical Reference Manual*, ARM DDI 0439B, section 3.9. https://developer.arm.com/documentation/ddi0439/b/
2. J. Yiu, "A Beginner's Guide on Interrupt Latency and Interrupt Latency of the Arm Cortex-M Processors," Arm Community, 2016. https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/beginner-guide-on-interrupt-latency-and-interrupt-latency-of-the-arm-cortex-m-processors
3. STMicroelectronics, *STM32G4 NVIC* training material. https://www.st.com/resource/en/product_training/STM32G4-System-Nested_Vectored_Interrupt_Control_NVIC.pdf
4. Texas Instruments, *TMS320C6000 DSP* documentation. https://www.ti.com/processors/digital-signal-processors/overview
5. AMD/Xilinx, *Vitis High-Level Synthesis User Guide*. https://docs.amd.com/

**Standards and regulation**

6. IEC 62304:2006+AMD1:2015, *Medical device software — Software life-cycle processes*. https://webstore.iec.ch/en/publication/22794
7. IEC 60601-1-8:2006+AMD2:2020, *Alarm systems*. https://webstore.iec.ch/en/publication/26539
8. ISO 14971:2019, *Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
9. IEC 81001-5-1:2021, *Health software and health IT systems safety, effectiveness and security — Part 5-1*. https://webstore.iec.ch/en/publication/63293
10. U.S. Food and Drug Administration, *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-management-system-considerations-and-content-premarket
11. IEC 62366-1:2015+AMD1:2020, *Usability engineering*. https://webstore.iec.ch/en/publication/26570
12. MISRA, *MISRA C:2012 Guidelines for the Use of the C Language in Critical Systems*. https://www.misra.org.uk/

**Robotics and real-time**

13. ROS 2 Documentation, *Real-Time Programming*. https://docs.ros.org/en/rolling/Tutorials/Real-Time-Programming.html
14. ROS 2 Reference System, real-time benchmarking. https://ros-realtime.github.io/reference-system/main/
15. "Timing Analysis and Priority-driven Enhancements of ROS 2 Multi-threaded Executors," arXiv:2408.08440. https://arxiv.org/html/2408.08440v2

**Device evidence**

16. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
17. Intuitive Surgical, da Vinci 5 clearance, March 2024. https://isrg.intuitive.com/news-releases/news-release-details/intuitive-announces-fda-clearance-fifth-generation-robotic

**Background**

18. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
19. E. A. Lee and S. A. Seshia, *Introduction to Embedded Systems: A Cyber-Physical Systems Approach*, 2nd ed., MIT Press, 2017.
20. G. C. Buttazzo, *Hard Real-Time Computing Systems*, 3rd ed., Springer, 2011.
21. U.S. Food and Drug Administration, *Content of Premarket Submissions for Device Software Functions*, guidance, 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/content-premarket-submissions-device-software-functions
22. IEC 60601-1:2005+AMD1:2012+AMD2:2020. https://webstore.iec.ch/en/publication/26169
23. ISO 26262-1:2018, *Road vehicles — Functional safety* (informative analogy, not a medical standard). https://www.iso.org/standard/68383.html
24. Arm, *Cortex-M4 Processor Datasheet*. https://developer.arm.com/documentation/102006/latest/
25. AMD/Xilinx, *Kria Real-Time ROS 2* documentation. https://xilinx.github.io/KRS/sphinx/build/html/docs/features/realtime_ros2.html

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_adcs-and-signal-conditioning.md`](04_adcs-and-signal-conditioning.md)
- Next topic: [`06_socs-npus-and-gpus.md`](06_socs-npus-and-gpus.md)
- Related: [`08_firmware-and-rtos.md`](08_firmware-and-rtos.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on TinyML quantization: see Lab 14A.3 in the survey file.
- Lab on hardware-in-the-loop testing: see Lab 14A.7 in the survey file.
- Lab index: [`../../labs/`](../../labs/)

### 8.7 A note on the software bill of materials

A software bill of materials (SBOM) lists every component in the firmware, including the operating system, the libraries, and the third-party code. It is required for vulnerability monitoring and for the premarket cybersecurity submission. Without an SBOM, a device cannot be checked against a newly disclosed vulnerability in a timely way.

For a processor, the SBOM includes the bootloader, the RTOS, the hardware abstraction layer, the signal-processing library, the model runtime, and the application. Each component has a version and a license. The SBOM is a living document, not a one-time artifact.

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

**Two more interview questions.**

**13. What is the role of the memory hierarchy in a processor choice?**
A strong answer explains that a workload can be compute-bound or memory-bound, and that the cache, TCM, and DRAM latency determine the tail. A weak answer compares clock speeds only.

**14. How do you make a firmware build reproducible?**
A strong answer pins the compiler, flags, libraries, and environment, and records the binary hash. A weak answer says "use version control".

**More sources.**

26. ISO 14971:2019, risk management (for the hazard analysis). https://www.iso.org/standard/72704.html
27. IEC 62366-1:2015+AMD1:2020, usability engineering. https://webstore.iec.ch/en/publication/26570
28. Arm, *Cortex-M4 Technical Reference Manual*, DDI 0439B, section 3.9.
29. Texas Instruments, *C2000 real-time control* documentation. https://www.ti.com/microcontrollers-mcus-processors/c2000-real-time-control-mcus/overview

- **Branch prediction** — hardware guess of the next instruction; a source of jitter.
- **Bus arbitration** — mechanism that decides which master uses the bus.
- **Clock domain crossing** — transfer of signals between asynchronous clocks.
- **Embedded flash** — non-volatile memory on the MCU.
- **Firmware image** — binary plus metadata loaded onto the device.
- **Interrupt vector table** — addresses of interrupt handlers.
- **Memory protection unit** — hardware that restricts memory access.
- **Pipeline hazard** — dependency that stalls the pipeline.
- **Real-time operating system (RTOS)** — OS with bounded scheduling.
- **Scratchpad memory** — software-managed fast memory.
- **Simd** — single instruction, multiple data.
- **Trace unit** — hardware that records execution for timing analysis.
- **Volatile** — qualifier that prevents the compiler from optimising an access.
- **Watchdog window** — time range in which servicing is allowed.

Further reading:

30. E. A. Lee and S. A. Seshia, *Introduction to Embedded Systems*, 2nd ed., 2017.
31. G. C. Buttazzo, *Hard Real-Time Computing Systems*, 3rd ed., 2011.
32. Arm, *Cortex-M Programming Guide*.
33. AMD/Xilinx, *Vitis HLS* documentation.
34. Texas Instruments, *C6000 DSP Optimization Guide*.
35. IEC 62304:2006+AMD1:2015, software lifecycle.
36. FDA, *Cybersecurity in Medical Devices*, guidance.
37. ISO 14971:2019, risk management.

- **Zero-wait-state memory** — memory that returns data in one cycle.
