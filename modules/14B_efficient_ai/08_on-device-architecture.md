# On-Device Architecture

> Module 14B — Efficient AI and On-Device Inference
> Topic 08 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 02, 07
> Version 2026.3

## 1. Why this topic exists

A compressed model is not a product. The product is a device that boots, reads a sensor, runs the model within a deadline, reports a result, survives a power loss, and can be updated without becoming unsafe. The architecture that surrounds the model determines whether any of that happens.

Teams that come from a machine-learning background tend to treat the runtime as a library call. Teams that come from embedded systems tend to treat the model as a blob. Both perspectives miss the same thing: the model and the runtime share memory, time, and failure modes, and the architecture must allocate all three deliberately.

Three industry problems follow.

**The memory surprise at integration.** A model that fits in flash and in the simulator fails on the device because the runtime's tensor arena, the sensor buffers, the stack, and the communication buffers together exceed SRAM. The failure appears at initialization, after the hardware is built, when the fix is expensive.

**The update surprise.** A model update that works on a bench device bricks a field device because the power failed mid-write and there was no rollback. Or the update succeeds but the new model is incompatible with the firmware's preprocessing, and the device produces plausible but wrong outputs.

**The observability surprise.** A device that runs correctly in the lab fails in the field, and there is no record of what happened because logging was disabled to save flash, or because the log contained protected health information and was never enabled. The failure is invisible and the fix is a guess.

On-device architecture is the discipline of designing the whole path: sensor, driver, direct memory access (DMA), preprocessing, model, postprocessing, output, watchdog, fallback, update, and telemetry. The model is one component, and it is often not the one that fails.

This topic teaches the architecture, the memory and storage budget, the runtime and format choices, the update path, and the validation that a device needs before it can be shipped.

### 1.1 Five architectural decisions

| Decision | Options | Consequence if wrong |
|---|---|---|
| Runtime | TFLM, ExecuTorch, ONNX Runtime, vendor SDK | Model cannot run or runs slowly |
| Format | `.tflite`, `.pte`, `.onnx`, `.gguf` | Conversion loses operators or accuracy |
| Memory model | Static arena, dynamic allocation | Boot failure or fragmentation |
| Update model | A/B, single slot, no field update | Bricking or no update path |
| Failure model | Watchdog, timeout, fallback, safe state | Unsafe behavior in the field |

The decisions are made early and are expensive to reverse. A runtime chosen after the model is frozen constrains the model; a memory model chosen after the board is built constrains the firmware; an update model chosen after launch cannot be retrofitted.

### 1.2 What this topic is not

- It is not a firmware engineering course. It teaches the architecture that firmware must implement, not the implementation.
- It is not a security audit. It names the security requirements that the architecture must satisfy.
- It is not a substitute for the clinical risk analysis. The safe state is a clinical decision.

The boundary matters because architecture reviews often drift into implementation or into regulatory strategy. This topic stays on the interface: what the model needs, what the device provides, and how the two fail together.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the end-to-end on-device data path and name the timing and memory budget of each stage.
2. **Budget** flash and SRAM separately, including the runtime arena, buffers, stack, and update slots.
3. **Choose** a runtime and model format from the target's constraints and the team's toolchain.
4. **Design** a watchdog, timeout, and fallback path that fails safe.
5. **Design** a signed update with A/B slots, rollback, and version reporting.
6. **Specify** telemetry that supports monitoring without exposing protected health information.
7. **Validate** boot, update, power loss, and soak behavior on the target.
8. **Document** the device's software and model versions in a reproducible record.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A data path with budgets |
| 2 | Exercise 2 | Flash and SRAM tables |
| 3 | Exercise 3 | A runtime choice with a rationale |
| 4 | Exercise 4 | A watchdog and fallback design |
| 5 | Exercise 5 | An update state machine |
| 6 | Exercise 6 | A telemetry schema with no PHI |
| 7 | Exercise 7 | A test plan with power-loss cases |
| 8 | Exercise 8 | A version and SBOM record |

## 3. The core mental model

The device is a pipeline with a control plane and a data plane.

```text
   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌───────────┐
   │ sensor   │──►│ driver   │──►│ DMA buffer│──►│preprocess│──►│  model    │
   └──────────┘   └──────────┘   └───────────┘   └─────────┘   └─────┬─────┘
                                                                     │
                                                                     ▼
   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌───────────┐
   │  output  │◄──│ postproc │◄──│  result   │◄──│ runtime │◄──│  logits   │
   └──────────┘   └──────────┘   └───────────┘   └─────────┘   └───────────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ control plane: watchdog, timeout, fallback, version, OTA     │
   └──────────────────────────────────────────────────────────────┘
```

The data plane must meet the deadline. The control plane must keep the device safe when the data plane does not.

### 3.1 The two budgets

**Flash** holds code, the model, configuration, logs, and the update machinery. It is persistent and cheap per byte, but it is slow to read and it wears out after many write cycles.

**SRAM** holds the runtime arena, activations, buffers, stack, and control structures. It is fast and expensive per byte, and it is lost on power-off.

The two budgets are not interchangeable. A model that fits in flash can fail because the arena does not fit in SRAM. A device with spare SRAM cannot use it to store a model that is too large for flash. The budgets must be planned separately and both must have headroom.

| Resource | Typical wearable | Typical handheld | Typical edge box |
|---|---:|---:|---:|
| Flash | 256 KiB–2 MiB | 8–128 GiB | 32–512 GiB |
| SRAM | 64–512 KiB | 2–16 GiB | 16–128 GiB |
| Clock | 16–200 MHz | 1–3 GHz | 1–3 GHz |
| Power | µW–mW | 0.5–5 W | 5–75 W |

### 3.2 The runtime contract

A runtime is defined by a contract, not by a logo. The contract has six parts.

- **Inputs:** shape, dtype, layout, range, and whether they are static.
- **Outputs:** shape, dtype, layout, and whether they are probabilities or logits.
- **Operators:** the set the runtime implements, at the precision and shape the model uses.
- **Memory:** the arena size, alignment, and whether allocation is static.
- **Timing:** the latency distribution on the target, not the host.
- **Failure:** what happens when an operator is unsupported, an allocation fails, or a deadline is missed.

A runtime that satisfies five of the six is not a candidate. The sixth will fail in the field.

### 3.3 The model format is part of the architecture

A model format is not a neutral container. It determines what the runtime can load, what it can fuse, what metadata it carries, and how large the artifact is.

| Format | Container | Typical runtime | Strength | Constraint |
|---|---|---|---|---|
| `.tflite` | FlatBuffer | LiteRT / TFLM | Mature microcontroller path | Operator coverage |
| `.onnx` | Protobuf | ONNX Runtime | Broad model support | Runtime-specific quantization |
| `.pte` | FlatBuffer with segments | ExecuTorch | PyTorch-native, backend-specialized | Maturing ecosystem |
| `.ptd` | Companion data file | ExecuTorch | Separates weights from program | Must be paired with `.pte` |
| `.plan` / engine | Proprietary | TensorRT | NVIDIA optimization | NVIDIA-specific |
| `.gguf` | Custom container | llama.cpp | Quantized LLM inference | LLM-specific |

The choice is constrained by the team's training framework, the target's runtime, and the operators the model uses. A format that the runtime can load but cannot accelerate is a format that produces a slow device.

### 3.8 The artifact is not the checkpoint

A training checkpoint contains optimizer state, gradients, and a framework-specific graph. A deployment artifact contains a graph, weights, and metadata in a runtime-specific format. The conversion between them can change the graph, the precision, and the operator set.

The conversion should be treated as a build step with its own tests: the exported artifact is loaded on the target, run on a reference input, and compared with the framework output. A conversion that succeeds is not a conversion that preserves behavior.

### 3.4 Failure is a design input

Every device will fail. The architecture decides whether it fails safe.

- **Watchdog:** a timer that resets the device if the main loop does not check in. It converts a hang into a reboot, which is better than a silent freeze.
- **Timeout:** a per-inference deadline. If the model does not produce a result in time, the device uses the fallback.
- **Fallback:** a simpler rule or model that produces a safe output when the main path is unavailable.
- **Safe state:** the output the device presents when it cannot compute. For an alarm device, the safe state is usually an alarm rather than silence; for a therapy device, it is usually a controlled stop.
- **Log:** a record of the failure that survives a reboot, without protected health information.

A device without a watchdog, a timeout, and a defined safe state is not architecturally complete, regardless of its model's accuracy.

### 3.7 Boot and wake

Boot and wake are different paths with different latency and different failure modes.

- **Cold boot** runs the bootloader, verifies the image, initializes the runtime, allocates the arena, and starts the application. It can take hundreds of milliseconds to seconds.
- **Wake from sleep** restores the clock and the peripherals and resumes the application. It can take milliseconds.
- **Warm restart** after a watchdog reset reinitializes the application but not the bootloader.

A clinical device that must respond immediately after power-on is bound by cold boot. A device that sleeps between inferences is bound by wake time. Both must be measured; a device that meets its inference deadline but takes 2 seconds to boot has not met its availability requirement.

The boot path also determines the update behavior: the bootloader must verify the image before jumping to it, and it must decide which slot to boot when both are marked valid or invalid.

### 3.5 The update path

An update is a sequence of states, and each transition can fail.

```text
   idle ──► download ──► verify ──► install ──► reboot ──► self-test ──► commit
              │            │           │           │           │
              ▼            ▼           ▼           ▼           ▼
            abort        abort       abort      rollback    rollback
```

The design requirements:

- **Signed bundle:** the device verifies the signature before installing.
- **A/B slots:** the new image is written to the inactive slot, so the running image is never modified.
- **Rollback:** if the self-test fails, the device boots the previous slot.
- **Anti-rollback:** the device refuses an image with a version lower than the installed one, to prevent a downgrade attack.
- **Power-loss safety:** an interrupted write leaves the inactive slot invalid, and the bootloader boots the valid slot.
- **Version reporting:** the device reports the firmware and model versions after boot.

### 3.6 Telemetry without PHI

Monitoring needs data; privacy forbids identifying data. The resolution is to send features and events, not raw signals or identifiers.

| Data | Send | Do not send |
|---|---|---|
| Model version | Yes | — |
| Firmware version | Yes | — |
| Inference count | Yes | — |
| Latency histogram | Yes | — |
| Saturation rate | Yes | — |
| Alert count | Yes | — |
| Raw ECG | No | Identifiers, timestamps, waveform |
| Patient identifiers | No | Any |
| Device location | No | Precise location |

The schema should be reviewed by privacy and security before the device ships, not after a incident.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Runtime | Software that executes a model on a device | Confused with the model | Component |
| Model format | Serialized representation of a model | Confused with the framework checkpoint | Format |
| FlatBuffer | Binary serialization format with random access | Confused with Protocol Buffers | Format |
| Protobuf | Binary serialization format | Confused with FlatBuffer | Format |
| Arena | Preallocated memory for tensors | Confused with heap | Buffer |
| Static allocation | Memory reserved at compile time | Confused with stack allocation | Technique |
| Dynamic allocation | Memory reserved at runtime | Assumed safe on an MCU | Technique |
| Fragmentation | Unusable free memory between allocations | Ignored in peak-memory estimates | Bytes |
| DMA | Direct memory access | Assumed free of contention | Engine |
| Double buffering | Two buffers alternating between producer and consumer | Confused with ping-pong in networking | Technique |
| Watchdog | Timer that resets the device if not serviced | Confused with a timeout | Component |
| Timeout | Deadline for an operation | Confused with a watchdog | ms |
| Fallback | Alternative path when the main path fails | Confused with operator fallback | Architecture |
| Safe state | Output presented when computation is unavailable | Assumed to be silence | State |
| OTA | Over-the-air update | Confused with a firmware flash | Process |
| A/B slots | Two firmware or model partitions | Confused with dual boot | Layout |
| Rollback | Reverting to a previous image | Confused with a factory reset | Process |
| Anti-rollback | Refusing older versions | Confused with rollback | Security |
| Secure boot | Verifying the boot image's signature | Confused with signed updates | Security |
| Signed bundle | Update package with a cryptographic signature | Assumed encrypted | Artifact |
| SBOM | Software bill of materials | Confused with a version string | Document |
| PHI | Protected health information | Confused with device telemetry | Data class |
| Telemetry | Operational data sent from the device | Assumed to be non-sensitive | Data |
| Soak test | Sustained load test | Confused with a benchmark | Test |
| Power-loss test | Test of behavior when power is removed | Often omitted | Test |
| Boot time | Time from power-on to ready | Confused with wake time | ms |
| Wake time | Time from sleep to ready | Confused with boot time | ms |
| Duty cycle | Fraction of time active | Confused with utilization | Proportion |
| Version string | Identifier for firmware or model | Assumed unique | String |
| Configuration item | Item under version control | Confused with a file | Quality record |
| Determinism | Same output every run | Assumed for all runtimes | Property |
| Fault injection | Deliberately causing failures in test | Confused with fuzzing | Test |
| Brownout | Supply voltage below the operating threshold | Confused with power-off | Condition |
| Bootloader | Code that loads the application | Confused with firmware | Component |
| Partition | A flash region | Confused with a graph partition | Region |
| Boot time | Time from power-on to ready | Confused with wake time | ms |
| Wake time | Time from sleep to ready | Confused with boot time | ms |
| Brownout | Supply below the operating threshold | Confused with power-off | Condition |
| Fault injection | Deliberately causing failures in test | Confused with fuzzing | Test |

Two terms deserve emphasis.

**Watchdog versus timeout.** A watchdog resets the whole device when the software stops servicing it. A timeout abandons a single operation. A device needs both: the timeout to use the fallback, and the watchdog to recover from a hang that the timeout cannot detect.

**Safe state is a clinical decision.** For an arrhythmia alarm, failing to alarm may be worse than a false alarm, so the safe state may be to alarm. For an insulin pump, the safe state may be to stop delivery and alert. The choice belongs to the clinical risk analysis, not to the firmware team.

## 5. The mechanics, step by step

The procedure produces an architecture with a memory budget and a validated failure behavior. It assumes a target board, a model, and a clinical requirement.

### Step 1. Write the runtime contract

**Input:** The model, the clinical requirement, and the target board.

**Output:** A one-page contract with inputs, outputs, operators, memory, timing, and failure behavior.

**Rationale:** The contract is the interface between the model team and the firmware team. Without it, each assumes the other has handled a constraint.

The contract includes:

- Input tensor shape, dtype, layout, and range.
- Output tensor shape, dtype, and interpretation.
- The operator list, with precision and shape constraints.
- The arena size and alignment.
- The latency budget at p99, with the measurement condition.
- The behavior when an operator is unsupported, an allocation fails, or the deadline is missed.

**Branch point:** If the contract cannot be written because a term is unknown, the architecture is not ready. Resolve the unknown before selecting a runtime.

**Edge case:** A model with dynamic shapes needs a shape contract, not only a tensor contract. The set of supported shapes is part of the interface.

### Step 2. Choose the runtime and the format

**Input:** The contract and the candidate runtimes.

**Output:** A runtime, a format, and a version.

**Rationale:** The runtime determines the operator coverage, the memory model, and the update path. The format determines what the runtime can load and optimize.

The decision:

| Constraint | Candidate runtime | Format |
|---|---|---|
| Cortex-M with < 256 KiB SRAM | LiteRT for Microcontrollers, CMSIS-NN | `.tflite` |
| Cortex-M with an Ethos-U NPU | ExecuTorch with the Ethos-U backend, or vendor toolchain | `.pte` |
| Mobile CPU/GPU/NPU | ExecuTorch, ONNX Runtime Mobile | `.pte`, `.onnx` |
| NVIDIA edge GPU | TensorRT | `.plan` |
| Local language model | llama.cpp, MLC-LLM | `.gguf` |
| Multi-vendor server | ONNX Runtime | `.onnx` |

**Branch point:** If the model uses an operator that no candidate runtime supports, change the operator or the model before choosing the runtime. A runtime that cannot execute the model is not a runtime for this product.

**Edge case:** A runtime that supports the operator in float but not in INT8 is not a candidate for a quantized model. Check the precision-specific support list.

### Step 3. Lay out flash

**Input:** The firmware size, the model size, the update strategy, and the log requirement.

**Output:** A flash map with offsets, sizes, and headroom.

**Rationale:** Flash is a hard constraint and an update mechanism. The layout determines whether A/B updates are possible and how much room remains for model growth.

A layout for a 1 MiB device:

| Region | Size (KiB) | Purpose |
|---|---:|---|
| Bootloader | 32 | Secure boot, update, rollback |
| Firmware A | 300 | Running image |
| Firmware B | 300 | Update target |
| Model A | 41 | Running model |
| Model B | 41 | Update target |
| Configuration | 8 | Calibration, thresholds, versions |
| Logs | 16 | Persistent event log |
| **Total** | **738** | 286 KiB free |

**Branch point:** If the total exceeds flash, the options are: remove A/B for the model (risking a bricked device), compress the firmware, reduce the log, or choose a larger part. The first is a safety decision, not a storage decision.

**Edge case:** Flash write granularity and wear leveling affect the log region. A log that writes every inference will wear out the sector; use a ring buffer and a wear-leveled region.

### Step 4. Lay out SRAM

**Input:** The arena requirement, the sensor buffers, the stack, and the runtime overhead.

**Output:** An SRAM map with the high-water mark and headroom.

**Rationale:** SRAM is the constraint that fails at initialization. A map makes the failure visible before the hardware is built.

An SRAM map for a 256 KiB device:

| Region | Size (KiB) | Purpose |
|---|---:|---|
| Tensor arena | 96 | Activations, workspace |
| Sensor DMA buffers | 16 | Double buffer, 8 KiB each |
| Preprocess buffer | 8 | Filtered window |
| Stack | 8 | Main and interrupt |
| Runtime control | 4 | Graphs, tensors, metadata |
| Log ring | 4 | Recent events |
| **Total** | **136** | 120 KiB free (47%) |

**Branch point:** If the total exceeds SRAM, reduce the input window, fuse operators to reduce intermediates, move buffers to a smaller type, or choose a larger part. The arena is usually the largest term and the easiest to reduce.

**Edge case:** The stack must be sized for the worst-case interrupt nesting, not the average. An interrupt that fires during a deep call chain can overflow a stack that is adequate in normal operation.

### Step 5. Design the data path

**Input:** The sensor interface, the preprocessing steps, and the model input contract.

**Output:** A data path with buffer ownership and timing at each stage.

**Rationale:** The data path determines the end-to-end latency and the memory traffic. A model that is fast in isolation can be slow because the sensor buffer is copied three times before it reaches the model.

The design:

- The sensor writes to a DMA buffer, not to a CPU-managed buffer.
- The DMA buffer is double-buffered so that the next sample arrives while the current one is processed.
- Preprocessing reads from the DMA buffer and writes to the model input buffer, with no intermediate copies.
- The model reads from the input buffer and writes to the output buffer.
- Postprocessing reads the output and produces the result.

**Branch point:** If a copy can be removed by changing the layout, do it before optimizing the model. A copy of 8 KiB at 1 GB/s is 8 µs, which is small; a copy through a slow bus can be milliseconds.

**Edge case:** Cache coherency. On a device with a data cache, a DMA write may not be visible to the CPU until the cache is invalidated. A missing invalidate produces stale data that looks like a model failure.

### Step 6. Add the watchdog, timeout, and fallback

**Input:** The deadline, the failure modes, and the safe state.

**Output:** A control-plane design with thresholds and actions.

**Rationale:** The device must fail safe, and the failure must be detectable. A watchdog without a timeout reboots on a slow inference; a timeout without a fallback produces no output.

The design:

| Condition | Detection | Action |
|---|---|---|
| Inference exceeds deadline | Timeout | Use fallback, log, continue |
| Main loop stalls | Watchdog | Reset, boot, self-test |
| Sensor silent | DMA completion timeout | Use last valid window or alarm |
| Memory allocation fails | Initialization check | Boot fallback firmware or safe state |
| Model output invalid | Range check | Use fallback, log |
| Power below threshold | Brownout detector | Safe state, controlled shutdown |

**Branch point:** If the fallback is a simpler model, it must be resident and tested. A fallback that must be downloaded is not a fallback.

**Edge case:** A watchdog that is serviced by an interrupt can hide a stalled main loop. Service it from the main loop, or use a windowed watchdog.

### Step 7. Design the update path

**Input:** The update mechanism, the security requirements, and the flash layout.

**Output:** A state machine with verification, rollback, and version reporting.

**Rationale:** An update is the most dangerous operation a device performs. It rewrites the code that keeps the device safe.

The state machine:

1. **Idle:** the device runs the active slot.
2. **Download:** the bundle is received into the inactive slot, with a hash check per block.
3. **Verify:** the signature is checked against the device's public key.
4. **Install:** the bootloader marks the inactive slot as pending.
5. **Reboot:** the bootloader boots the pending slot.
6. **Self-test:** the firmware checks the model, the sensor, and a known input.
7. **Commit:** the pending slot becomes active; the previous slot becomes the fallback.
8. **Rollback:** if the self-test fails, the bootloader boots the previous slot and marks the pending slot invalid.

**Branch point:** If the device cannot support A/B slots, the update must be performed by a bootloader that can recover from a partial write, or the device must not support field updates at all.

**Edge case:** A power loss during the commit step can leave both slots marked valid or both invalid. The bootloader must resolve the state deterministically, usually by a monotonic counter and a validity flag written last.

### Step 8. Design telemetry and logging

**Input:** The monitoring questions and the privacy constraints.

**Output:** A schema, a retention policy, and a transport plan.

**Rationale:** Monitoring is how a deployed model is kept safe. A device that cannot report its version, its error rate, and its saturation rate cannot be monitored.

The schema includes:

- Firmware and model versions.
- Inference count and latency histogram.
- Fallback count and reason.
- Saturation rate per layer.
- Alert count, not patient data.
- Power and thermal events.

**Branch point:** If the transport is intermittent, buffer events in a ring and send summaries. A device that loses events when offline cannot support a monitoring claim.

**Edge case:** A log that contains a timestamp and a device identifier can be re-identified in a small population. Aggregate before transmission, or use a rotating identifier.

### Step 9. Validate boot, update, power loss, and soak

**Input:** The device, the test fixtures, and the failure cases.

**Output:** A test report with pass or fail per case.

**Rationale:** The failure modes are not exercised by normal operation. They must be induced.

The test cases:

- Cold boot from each slot.
- Update with a valid bundle.
- Update with a corrupted bundle.
- Update with an invalid signature.
- Power loss at each stage of the update.
- Rollback after a failed self-test.
- Watchdog reset during inference.
- Sensor timeout during inference.
- Brownout during a flash write.
- 72-hour soak at the specified ambient.

**Branch point:** If a power-loss case leaves the device unrecoverable, the update design is not complete. Do not ship with a known bricking case.

**Edge case:** A test that passes on one board revision may fail on another with a different flash part. Test the production configuration.

### Step 10. Record versions, SBOM, and configuration

**Input:** All artifacts and test results.

**Output:** A configuration record and an SBOM.

**Rationale:** A device is a configuration of hardware, firmware, runtime, model, and data. Without the record, a field failure cannot be reproduced.

The record includes:

- Board revision and component part numbers.
- Bootloader, firmware, and runtime versions.
- Model hash and format version.
- Calibration data version.
- Build toolchain and flags.
- SBOM with third-party components and licenses.
- Test report with dates and results.
- Change classification for each update.

**Branch point:** If a component in the SBOM has a known vulnerability, the device needs a remediation plan before shipment, not after disclosure.

**Edge case:** A model file that is identical but loaded by a different runtime version is a different configuration. Record both.

### 5.1 The boot sequence

The boot sequence is a state machine with verification at each step.

```text
   power-on
      │
      ▼
   bootloader ──► verify bootloader signature (if secure boot)
      │
      ▼
   select slot ──► check validity flag and version counter
      │
      ▼
   verify image ──► hash and signature
      │
      ├── fail ──► boot the other slot ──► fail ──► safe state
      ▼
   initialize runtime ──► allocate arena ──► fail ──► safe state
      │
      ▼
   self-test ──► sensor, model, known input
      │
      ├── fail ──► rollback
      ▼
   application
```

Each step has a failure path, and each failure path must be reachable in test. A boot sequence without a defined failure path is a sequence that will hang in the field.

The self-test is the most important step for a model update. It runs the model on a known input and checks the output against an expected range. A model that loads but produces a constant output passes the load check and fails the self-test, which is exactly the distinction the device needs.

### 5.2 A fault-injection matrix

Fault injection is how the architecture's failure paths are tested. The matrix below lists the faults and the expected behavior.

| Fault | Injection | Expected behavior |
|---|---|---|
| Power loss during download | Cut supply at 50% | Inactive slot invalid, active slot boots |
| Power loss during install | Cut supply at 90% | Bootloader detects invalid flag, boots active |
| Corrupted bundle | Flip a byte | Hash check fails, update aborts |
| Invalid signature | Sign with wrong key | Verification fails, update aborts |
| Model load failure | Corrupt the model header | Self-test fails, rollback |
| Watchdog stall | Block the main loop | Reset, boot, self-test |
| Sensor timeout | Disconnect the sensor | Fallback or alarm |
| Brownout | Lower supply below threshold | Safe state, controlled shutdown |
| Flash wear | Cycle the log region | Wear leveling, no data loss |

### 5.3 Decision table for architecture choices

| Constraint | Choice | Alternative | Trade-off |
|---|---|---|---|
| SRAM < 128 KiB | TFLM with a static arena | Vendor runtime with a smaller arena | Operator coverage |
| NPU available | ExecuTorch with a delegate, or vendor SDK | CPU-only | Portability |
| Needs A/B updates | Two firmware and model slots | Single slot with a recovery bootloader | Flash cost |
| No network | Local logging with manual export | Telemetry | Monitoring delay |
| Battery-powered | Duty-cycled inference | Continuous inference | Latency |
| Real-time deadline | Static allocation, watchdog | Dynamic runtime | Flexibility |
| Regulated | Traceable toolchain and SBOM | Fast iteration | Schedule |

### 5.4 Edge cases in on-device architecture

**Outlier channels.** A model with an outlier channel may need a larger arena if the runtime keeps a per-channel buffer. The memory effect is small but real on a tight device.

**Per-tensor versus per-channel scales.** Per-channel scales add metadata to the model file and, on some runtimes, a per-channel buffer in SRAM. On a 41 KiB model the metadata is 600 bytes; on a device with 120 KiB free, that is negligible, but on a smaller device it matters.

**Activation clipping.** Clipping can reduce the arena if it allows a smaller activation type, but the accuracy effect must be evaluated. The memory gain is usually small compared with the risk.

**Operator fallback.** A fallback requires the CPU to have a buffer for the tensor and the code for the operator. On a device with no spare flash, the fallback code can be the difference between fitting and not fitting.

**Unsupported operators.** A model that loads but falls back at runtime needs both the accelerator and the CPU path in flash. The flash budget must include both.

**Zero-point drift.** A device that runs for months may see sensor drift that shifts the activation distribution. The saturation rate should be logged and monitored; a drift that saturates the quantizer is invisible in accuracy until it is large.

**Calibration-set bias.** The calibration data is part of the device configuration. A device calibrated on one site and deployed at another carries a bias that the architecture cannot detect without telemetry.

**Thermal throttling.** A device in a sealed enclosure may throttle after minutes. The watchdog and timeout must be set for the sustained latency, not the burst latency, or the fallback will trigger in normal operation.

## 6. Worked example

This is a constructed case. The device, memory map, and measurements are illustrative and chosen so that the arithmetic closes.

### 6.1 The device

The ECG patch uses a Cortex-M4F at 64 MHz with 256 KiB SRAM and 1 MiB flash, powered by a CR2032 coin cell rated 235 mAh at 3.0 V. It runs a 41,216-parameter INT8 model of 41,816 bytes and a 4.2 ms sensor-to-alert pipeline segment.

The clinical requirement is a 250 ms end-to-end alert latency, a 7-day battery life, and a 64 KiB model budget with room for an update slot.

### 6.2 Scale and zero point by hand

The model's first convolution has 40 weights with a minimum of -0.061 and a maximum of 0.412. The affine parameters at 8 bits:

```text
S = (0.412 - (-0.061)) / 255 = 0.473 / 255 = 0.00185490
Z = round(0.061 / 0.00185490) = round(32.886) = 33
```

Endpoint checks:

```text
q(0.412) = round(222.11) + 33 = 255
q(-0.061) = round(-32.886) + 33 = 0
```

The symmetric parameters:

```text
S = 0.412 / 127 = 0.00324409
Z = 0
```

The affine step is 1.75 times finer than the symmetric step, and the affine SQNR is 42.04 dB against 37.54 dB. On this layer, affine is the better choice; on the model as a whole, weights are symmetric per-channel and activations are asymmetric per-tensor.

### 6.3 Flash budget

| Region | Size (KiB) | Notes |
|---|---:|---|
| Bootloader | 32 | Secure boot, update, rollback |
| Firmware A | 300 | Application |
| Firmware B | 300 | Update target |
| Model A | 41 | 41,816 B rounded up |
| Model B | 41 | Update target |
| Configuration | 8 | Thresholds, calibration, versions |
| Logs | 16 | Ring buffer |
| **Total** | **738** | of 1,024 KiB |

Free flash: 1,024 - 738 = 286 KiB, or 27.9%. The free space is the margin for model growth: a model can grow from 41 KiB to 184 KiB in each slot before the layout fails, which is more than four times its current size.

The A/B model slots cost 41 KiB, or 4.0% of flash. The cost is the price of a safe update. A single-slot design would save 41 KiB and risk a bricked device on a power loss during update.

### 6.4 SRAM budget

| Region | Size (KiB) | Notes |
|---|---:|---|
| Tensor arena | 96 | Activations, workspace |
| Sensor DMA buffers | 16 | Two 8 KiB buffers |
| Preprocess buffer | 8 | Filtered window |
| Stack | 8 | Worst-case interrupt nesting |
| Runtime control | 4 | Graph and tensor metadata |
| Log ring | 4 | Recent events |
| **Total** | **136** | of 256 KiB |

Free SRAM: 120 KiB, or 46.9%. The headroom is deliberate: the runtime may allocate scratch buffers that are not visible in the static plan, and a firmware update may add a feature that needs memory.

The tensor arena is 70.6% of the used SRAM. If the model's activations grew by 20%, the arena would need 115 KiB and the total would be 155 KiB, still within 256 KiB. If they doubled, the total would be 232 KiB, leaving 24 KiB, which is too little headroom for a production device.

### 6.5 Latency budget

| Stage | Time (ms) | Notes |
|---|---:|---|
| Sensor acquisition and DMA | 32 | 256 Hz, 8-sample window |
| Preprocessing | 18 | Filter, normalize, segment |
| Inference | 45.7 | p99, warm |
| Postprocessing | 4 | Threshold and refractory logic |
| BLE transport | 90 | p99 |
| Human acknowledgement | 20 | Display and vibration |
| **Total** | **209.7** | of 250 ms |

The margin is 40.3 ms, or 16.1%. The model is 21.8% of the total. The transport is 42.9%, which is why a model-only optimization has limited effect on the end-to-end deadline.

### 6.6 Energy and battery life

```text
E_inference = 3.0 V × 0.0082 A × 0.0457 s = 1.124 mJ
E_cycle     = 1.124 + 0.675 (preprocess) + 3.240 (radio) = 5.039 mJ
```

At one cycle every 2 seconds, the average power is 5.039 mJ / 2 s = 2.520 mW, and the average current is 0.840 mA. Adding 3 µA of sleep current gives 0.843 mA.

```text
ideal life = 235 mAh / 0.843 mA = 278.8 h = 11.6 days
derated    = 11.6 × 0.66 = 7.7 days
```

The 0.66 factor is the capacity derating at 0 °C. The device meets the 7-day requirement with 0.7 days of margin, which is thin but positive. The radio is 64.3% of the cycle energy; the model is 22.3%.

### 6.7 Accuracy and calibration

| Configuration | Sensitivity | ECE | Slope | Worst subgroup |
|---|---:|---:|---:|---:|
| FP32 reference | 0.942 | 0.021 | 0.97 | 0.931 |
| INT8 on device | 0.938 | 0.029 | 0.93 | 0.926 |
| INT8 after OTA | 0.938 | 0.029 | 0.93 | 0.926 |

The OTA does not change the model; it changes the firmware that runs it. The accuracy is identical because the model and the runtime are identical. If the OTA also updated the model, the accuracy would need to be re-evaluated, and the change would need to be classified.

### 6.8 Diagnostic one: outlier-driven scale collapse

The fully connected layer's weights have a median absolute value of 0.0610 and a maximum of 1.2912. The symmetric per-tensor scale is 0.01016693, which is 16.7% of the median.

```text
step / median = 0.1667
weights rounded to zero = 1,411 of 32,768 (4.3%)
distinct levels used = 102 of 255
SQNR = 30.00 dB
```

The aggregate SQNR is acceptable and the zeroed fraction is small, but the level count shows that most of the integer range is unused. The remedy is per-channel quantization, which raises SQNR to 44.67 dB and costs 2,560 bytes of metadata.

On this device, 2,560 bytes is 6.1% of the 41,816-byte model and 0.24% of flash. The metadata is affordable, and per-channel is the right choice.

### 6.9 Diagnostic two: per-channel skew

The 512 channels have ranges from 0.2756 to 2.3202, a ratio of 8.4. Under a per-tensor scale, a typical channel receives about 28 levels. Per-channel scales give every channel 255 levels.

The skew is a property of the weights. It does not change with the device architecture, the runtime, or the update path. It does determine whether the model fits its accuracy budget, which is why it belongs in the architecture review: a model that needs per-channel scales needs the metadata in flash and, on some runtimes, a per-channel buffer in SRAM.

### 6.10 Varying one assumption

**Model size: 41 KiB to 70 KiB.** Both slots become 70 KiB, and the flash total becomes 738 - 82 + 140 = 796 KiB, leaving 228 KiB. The layout still fits. At 184 KiB per slot, the total would be 1,024 KiB and the layout would fail; the maximum model size with A/B slots is therefore about 184 KiB, not the naive 286 KiB free.

**Arena: 96 KiB to 160 KiB.** The SRAM total becomes 200 KiB, leaving 56 KiB. That is still positive, but the headroom is only 21.9%, which is thin for a production device with runtime scratch allocations. A 160 KiB arena is a warning sign, not an immediate failure.

**Update without A/B slots.** The model slot shrinks from 82 KiB to 41 KiB, saving 41 KiB of flash, but a power loss during the update leaves the device without a valid model. The saving is not worth the risk for a device that cannot be physically recovered.

**Calibration set: 512 to 64 windows.** The activation ranges shrink by about 6%, clipping increases, and sensitivity falls by 0.4 points to 0.934. The flash and SRAM budgets do not change. The calibration data is part of the configuration partition, and its size (8 KiB) is unchanged, but its quality is not.

**Bit width: 8 to 4 for weights.** The model falls to about 21 KiB, and both slots fall to 42 KiB total, saving 40 KiB of flash. The first-layer SQNR falls to 12.65 dB, and sensitivity falls by about 3.1 points to 0.907, below the floor. The flash saving is real and the accuracy cost is unacceptable.

### 6.11 The decision

The architecture fits: 738 KiB of 1,024 KiB of flash, 136 KiB of 256 KiB of SRAM, and a 209.7 ms pipeline against a 250 ms deadline. The margins are 27.9% flash, 46.9% SRAM, and 16.1% latency. The narrowest margin is latency, and the largest term in latency is the radio, not the model.

The architecture is therefore not constrained by the model. A model update can grow to 184 KiB per slot before the flash layout fails, which is four and a half times the current size. The next constraint to bind is latency: a model that grows in MACs will consume the 40.3 ms margin before it consumes the flash margin.

The decision is to ship the current model with the A/B update path, the 96 KiB arena, and the 136 KiB SRAM map, and to set a model-size trigger: if the model exceeds 120 KiB per slot, the flash headroom falls below 12.5% and the layout must be revisited.

The SRAM trigger is an arena above 128 KiB, at which point the total exceeds 168 KiB and the headroom falls below 35%. The latency trigger is a model above 55 ms, at which point the pipeline margin falls below 10%.

This is the characteristic shape of a device architecture decision: not "does it fit", but "which margin binds first, and what is the trigger for revisiting the design".

### 6.12 What the case does not show

- **The production variation.** Three units from different lots can differ in flash size, sensor gain, and battery capacity. The margins must absorb that variation.
- **The thermal steady state.** The latency budget is measured warm at 22 °C; a sealed device at 40 °C may have a longer p99.
- **The failure paths.** The case describes the normal path. The fault-injection matrix in Section 5 is the evidence for the failure paths.
- **The update payload.** The case assumes a 41 KiB model over BLE. A larger model changes the download time and the user experience.
- **The manufacturing test.** A device that passes in the lab may fail in production because of a solder joint or a calibration step. That test is outside this topic but inside the product.

## 7. Code, query, or calculation

The script budgets flash and SRAM, computes the model artifact size, checks the update layout, and implements the quantization core. It reproduces Section 6.

```python
def kib(nbytes):
    return nbytes / 1024


def flash_map(firmware_kib=300, model_bytes=41_816, boot_kib=32,
              config_kib=8, log_kib=16, total_kib=1024, ab=True):
    model_kib = kib(model_bytes)
    slots = 2 if ab else 1
    used = boot_kib + 2 * firmware_kib + slots * model_kib + config_kib + log_kib
    return {"model_kib": model_kib, "slots": slots,
            "used_kib": used, "free_kib": total_kib - used,
            "free_pct": 100 * (total_kib - used) / total_kib}
```

Budget the device:

```python
flash = flash_map()
print("model KiB = %.1f" % flash["model_kib"])
print("flash used = %.1f KiB free = %.1f KiB (%.1f%%)"
      % (flash["used_kib"], flash["free_kib"], flash["free_pct"]))

sram = {"arena": 96, "dma": 16, "preprocess": 8, "stack": 8,
        "runtime": 4, "log": 4}
print("sram used =", sum(sram.values()), "KiB free =", 256 - sum(sram.values()),
      "KiB (%.1f%%)" % (100 * (256 - sum(sram.values())) / 256))

model_kib = flash["model_kib"]
max_model = (1024 - 32 - 600 - 8 - 16) / 2
print("max model per slot with A/B = %.1f KiB" % max_model)
print("max model without A/B = %.1f KiB" % (max_model * 2))
```

Expected output:

```text
model KiB = 40.8
flash used = 737.8 KiB free = 286.2 KiB (27.9%)
sram used = 136 KiB free = 120 KiB (46.9%)
max model per slot with A/B = 184.0 KiB
max model without A/B = 368.0 KiB
```

Latency and energy:

```python
stages = {"sensor": 32, "preprocess": 18, "inference": 45.7,
          "postprocess": 4, "ble": 90, "human": 20}
total = sum(stages.values())
print("pipeline ms =", total, "margin =", 250 - total,
      "(%.1f%%)" % (100 * (250 - total) / 250))
print("model share = %.1f%%" % (100 * stages["inference"] / total))

e_inf = 3.0 * 0.0082 * 0.0457
e_cycle = e_inf + 0.675 + 3.240
print("E inference = %.3f mJ, E cycle = %.3f mJ" % (e_inf, e_cycle))
current_ma = e_cycle / 2.0 / 3.0
life_days = 235 / (current_ma + 0.003) / 24
print("avg current = %.3f mA, ideal life = %.1f days, derated = %.1f days"
      % (current_ma + 0.003, life_days, life_days * 0.66))
```

Expected output:

```text
pipeline ms = 209.7 margin = 40.3 (16.1%)
model share = 21.8%
E inference = 1.124 mJ, E cycle = 5.039 mJ
avg current = 0.843 mA, ideal life = 11.6 days, derated = 7.7 days
```

Update flash math:

```python
for model_kib in (40.8, 70, 120, 184):
    f = flash_map(model_bytes=model_kib * 1024)
    print("model %5.1f KiB -> used %6.1f free %6.1f (%.1f%%)"
          % (model_kib, f["used_kib"], f["free_kib"], f["free_pct"]))
```

Expected output:

```text
model  40.8 KiB -> used  737.8 free  286.2 (27.9%)
model  70.0 KiB -> used  796.2 free  227.8 (22.2%)
model 120.0 KiB -> used  896.2 free  127.8 (12.5%)
model 184.0 KiB -> used 1024.2 free   -0.2 (-0.0%)
```

Quantization diagnostics:

```python
import math
import random


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

Assumptions: the model is 41,816 bytes; the firmware is 300 KiB per slot; the bootloader is 32 KiB; configuration is 8 KiB; logs are 16 KiB; the arena is 96 KiB; the cycle runs every 2 seconds; the board draws 8.2 mA during inference; and the CR2032 derates to 66% at 0 °C.

## 8. Data, standards, and regulatory context

On-device architecture is the most heavily standardized part of the module, because it touches electrical safety, software lifecycle, cybersecurity, and usability. The runtime and format specifications are vendor documents; the safety and lifecycle requirements are standards.

### 8.1 IEC 62304

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79, requires the software architecture, detailed design, unit verification, integration testing, and configuration management.

For this topic, the standard implies:

- The runtime and bootloader are software items with their own requirements and verification.
- The model is a software item, even if it was produced by a training pipeline.
- The memory and timing budgets are design constraints that must be verified.
- The update mechanism is a software item with its own failure-mode analysis.
- The SBOM is part of the configuration record.

A device whose architecture exists only in a slide does not meet the design-documentation expectation, even if it works.

### 8.2 IEC 60601-1

IEC 60601-1, the general standard for basic safety and essential performance of medical electrical equipment, constrains the electrical and environmental design. For software, the relevant concept is **essential performance**: the performance necessary to keep risk acceptable.

If the device's essential performance depends on the model's output within a deadline, the architecture's watchdog, timeout, and fallback are part of the essential-performance design. They must be verified under the environmental conditions the standard requires, including supply variation and electromagnetic disturbance.

### 8.3 ISO 14971

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and controlled. The architecture introduces hazards that the model alone does not:

- A watchdog reset during an alert.
- A fallback that produces a false negative.
- A failed update that leaves the device inoperative.
- A sensor timeout that is interpreted as a normal reading.
- A log that leaks protected health information.

Each is a foreseeable hazard with a control and a residual-risk judgement. The controls are architectural: the watchdog, the safe state, the A/B slots, the timeout behavior, and the telemetry schema.

### 8.4 Cybersecurity

The FDA's cybersecurity guidance for medical devices, and the IEC 62443 series for industrial automation and control systems, establish expectations for secure boot, signed updates, vulnerability monitoring, and an SBOM. The exact obligations depend on the jurisdiction and the device class.

For this topic, the minimum architecture is:

- Secure boot with a verified root of trust.
- Signed update bundles with anti-rollback.
- An SBOM with third-party components and versions.
- A vulnerability monitoring process.
- Encrypted storage of any patient data.
- Least-privilege access to debug interfaces in production.

A device that ships with an open debug port and an unsigned update path has an architecture defect, regardless of its model.

### 8.5 FDA PCCP

The FDA's final guidance on Predetermined Change Control Plans for AI-enabled device software functions was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022.

The architecture determines what a PCCP can authorize. A device with A/B slots and a signed update path can implement a planned model update; a device without them cannot, even if the plan describes one. The plan and the architecture must be consistent.

### 8.6 EU framework

In the European Union, Regulation (EU) 2017/745 governs the device, and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk systems. The Annex I applicability date was originally 2 August 2027; the AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves it to 2 August 2028. Verify the current date before relying on it.

### 8.7 What is binding and what is convention

| Item | Status | Consequence |
|---|---|---|
| IEC 62304 | Binding where required | Runtime, bootloader, model are software items |
| IEC 60601-1 | Binding where required | Essential performance includes timing |
| ISO 14971 | Binding where required | Architectural hazards enter the risk file |
| IEC 62443 / FDA cybersecurity | Binding or expected | Secure boot, signed updates, SBOM |
| FDA PCCP guidance | Non-binding; statute binding | Update path must match the plan |
| FlatBuffer and Protobuf | Vendor/runtime convention | Format compatibility |
| A/B slots | Engineering convention | Not required, but expected for field updates |
| Telemetry schema | Convention | Privacy review required |

### 8.8 The documentation a device needs

| Record | Content |
|---|---|
| Architecture description | Data path, control path, memory map |
| Runtime contract | Inputs, outputs, operators, memory, timing, failure |
| Flash and SRAM maps | Offsets, sizes, headroom |
| Update state machine | States, transitions, rollback |
| Security architecture | Boot, signing, keys, debug access |
| SBOM | Components, versions, licenses, vulnerabilities |
| Test report | Boot, update, power loss, soak, fault injection |
| Configuration record | Board, firmware, runtime, model, calibration |

### 8.9 What the standards do not say

- No standard specifies an arena size, a watchdog period, or a slot layout.
- No standard requires A/B updates, although cybersecurity guidance expects recoverable updates.
- No standard defines the content of a telemetry schema beyond privacy law.
- No standard says whether a firmware update that does not change the model is a device change; that depends on the effect.

The absence means the architecture must justify its choices with its own risk analysis, not with a citation.

### 8.10 What a reviewer will ask

- Show the flash map and the SRAM map, with headroom.
- Show the arena high-water mark, not the model size.
- Show the update state machine and the power-loss tests.
- Show the watchdog period and the p99 it is derived from.
- Show the fallback and the safe state.
- Show the telemetry schema and the privacy review.
- Show the SBOM and the vulnerability process.
- Show the configuration record for the exact build under review.

A team that can produce these eight items has an architecture package. A team that can produce a block diagram has a proposal.

## 9. Industry practice and case evidence

### 9.1 Runtime documentation

The runtime projects document their memory and format models in enough detail to design against.

- LiteRT for Microcontrollers documents a single tensor arena with a head, a temporary region, and a tail, and a greedy memory planner that reuses buffers. The runtime fits in about 16 KiB on a Cortex-M3, and the arena size is determined by experiment. The documentation includes recording APIs that print the allocation plan, which is the practical way to size the arena.
- ExecuTorch documents a `.pte` file as a flatbuffer with an optional extended header and optional appended segments, with a companion `.ptd` file for weights when program-data separation is used. Each `.pte` is backend-specialized, and unsupported nodes fall back to the portable CPU.
- ONNX Runtime documents graph optimization levels and execution providers, and warns that offline-optimized models must match the target hardware.
- llama.cpp documents GGUF tensor encodings: Q8_0 uses 32-weight blocks with an FP16 scale at 8.5 bits per weight, and Q4_K uses 256-weight superblocks with 6-bit sub-scales at 4.5 bits per weight. These are format facts, not performance claims.

These are documented facts about the runtimes. The inference for a device architecture is that the runtime's memory model and format are design inputs, not implementation details.

### 9.2 Update architecture in practice

A/B update with rollback is standard practice in mobile and embedded systems, and the pattern is well documented in vendor and platform guidance. The essential elements are a bootloader that can select between slots, a validity flag written after a successful self-test, and an anti-rollback counter.

The medical-device-specific addition is the evidence: the update path must be validated for power loss at every stage, and the change must be classified against the approved plan. A consumer device can recover from a failed update by asking the user to retry; a medical device may not have that option.

### 9.3 What the evidence supports

| Claim | Status |
|---|---|
| TFLM fits in about 16 KiB on a Cortex-M3 | Vendor documentation |
| The TFLM arena is sized by experiment | Vendor documentation |
| ExecuTorch `.pte` files are backend-specialized | Vendor documentation |
| GGUF Q4_K uses 4.5 bits per weight | Format documentation |
| A/B updates with rollback prevent bricking | Established engineering practice |
| A watchdog prevents silent hangs | Established practice, with caveats |
| Telemetry without PHI is sufficient for monitoring | Not established; depends on the schema and population |
| An update path validated on one board works on another | Not supported; test the production configuration |

### 9.4 What the evidence does not establish

- That a memory map validated on one board works on another with a different flash part.
- That an update validated with a successful case survives a power loss at every stage.
- That a watchdog period that works at 22 °C works at 40 °C, where the p99 is longer.
- That telemetry that is de-identified at launch remains so when the population changes.
- That a runtime upgrade preserves the memory plan and the operator support.

Each is a hypothesis that the test plan must address, not an assumption the architecture can carry.

### 9.5 Why medical devices differ

**The device may be implanted or worn continuously.** A reboot is not a neutral event; it may interrupt monitoring or therapy. The watchdog is a safety mechanism, not a convenience.

**The update may change a clinical claim.** A firmware update that changes preprocessing or thresholds is a device change, even if the model file is identical.

**The evidence must survive the update.** A test report for version 1.0 does not cover version 1.1 unless the change is analyzed and the affected tests are repeated.

**The privacy obligation is continuous.** Telemetry that is acceptable at launch may become unacceptable when the population or the data source changes.

### 9.6 A note on procurement

The architecture is also a procurement document. The flash and SRAM sizes, the runtime, the update path, and the security features determine which part can be bought and at what price. A team that specifies the model first and the part second often discovers that the only part with enough SRAM is also the part that breaks the bill of materials.

The practical sequence is to set the memory and security requirements from the clinical and update requirements, then choose the smallest part that meets them with headroom.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Arena too small | Initialization failure | Runtime scratch | Device does not boot | Measure high-water mark |
| Stack overflow | Intermittent crashes | Deep interrupt nesting | Reset loop | Size for worst case |
| DMA coherency bug | Stale sensor data | Missing cache invalidate | Wrong outputs | Invalidate after DMA |
| Watchdog too tight | Resets during slow inference | Sustained latency | Lost alerts | Set from p99, not p50 |
| No timeout | Hang blocks alerts | Main loop stall | Missed deadline | Per-inference timeout |
| Fallback not resident | Fallback fails when needed | Code not in flash | Unsafe state | Include fallback in image |
| Update bricks device | Power-loss test fails | No rollback | Field failure | A/B slots and bootloader |
| Signature not verified | Update accepts any bundle | No secure boot | Malicious update | Verify before install |
| Anti-rollback missing | Old image accepted | Downgrade attack | Vulnerability exposure | Monotonic counter |
| Log contains PHI | Privacy review flags it | Raw signal logged | Regulatory breach | Log features, not signals |
| Version not reported | Field failure cannot be reproduced | No version string | Long investigation | Report at boot |
| SBOM missing | Vulnerability response delayed | Unknown components | Prolonged exposure | Maintain SBOM |
| Flash wear | Log region fails | Write amplification | Data loss | Ring buffer; wear leveling |
| Brownout during write | Corrupted slot | Supply collapse | Boot failure | Brownout detector; atomic flags |
| Boot time exceeds availability requirement | Device slow to start | Large image, slow flash | Missed first alert | Measure cold boot; optimize init |
| Wake latency not measured | Battery life or latency off | Peripheral restore time | Missed deadline | Measure wake separately |
| Self-test absent | Bad model runs | No known-input check | Silent wrong output | Add a boot self-test |
| Debug port left open | Production access | No fuse or lock | Data extraction | Disable in production |
| Keys stored in flash | Key extraction | No secure element | Update forgery | Use a secure element |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The model size is stable | Flash budget | Model growth | Recompute after each model change |
| The arena size is sufficient | SRAM budget | Runtime scratch, fragmentation | High-water measurement |
| The firmware fits in 300 KiB | Layout | Feature growth | Check the map file |
| The update path is atomic | Safe update | Power loss at a bad moment | Fault injection |
| The watchdog period exceeds p99 latency | No false resets | Thermal throttling | Soak test |
| The fallback is resident and tested | Safe state | Flash pressure | Fault injection |
| The sensor buffer is coherent | Correct data | Missing cache invalidate | Compare DMA and CPU buffers |
| The telemetry schema is de-identified | Privacy | Small populations | Privacy review |
| The board revision matches production | Transfer of results | Different flash or SRAM | Test the production board |
| The runtime version is fixed | Reproducibility | Upgrade | Pin and re-validate |
| The battery model is accurate | Battery life | Temperature, pulse drain | Bench measurement |
| The calibration data is representative | Accuracy | Site shift | Stratified audit |
| Cold boot meets the availability requirement | Device usable at power-on | Large image, slow verification | Measure cold boot |
| Wake time is measured separately | Predicts latency | Peripheral restore | Measure wake |
| The self-test exercises the model | Detects bad updates | Load-only check | Fault injection |
| Debug access is disabled in production | Prevents extraction | Open port | Verify fuses |
| Keys are protected | Prevents forgery | Keys in plain flash | Inspect the design |

The central limit is that the architecture is validated on a small number of devices, while the field contains many. Unit-to-unit variation in flash, sensors, and batteries can move the margins. The validation should include at least three units from different production lots, and the margins should be large enough to absorb the variation.

A second limit is that the update path is validated by fault injection, which is never exhaustive. The test plan should cover the states and transitions, not every possible timing; the bootloader's design should be simple enough that the uncovered cases are provably safe.

## 12. Exercises

**Exercise 1 — Data path (design, 35 minutes).**
Draw the data path for a wearable ECG patch from electrode to alert, naming the buffer ownership and the time budget at each stage.
*Deliverable:* a diagram with a table of stages and times.
*Acceptance criteria:* every stage has an owner and a time; the total is compared with the deadline; at least one copy is identified as removable.
*Time:* 35 minutes.

**Exercise 2 — Memory budget (calculation, 40 minutes).**
A device has 512 KiB flash and 128 KiB SRAM. Firmware is 180 KiB per slot, the bootloader is 24 KiB, configuration is 4 KiB, logs are 8 KiB, and the model is 30 KiB. The arena is 48 KiB, DMA buffers 12 KiB, stack 6 KiB, runtime 3 KiB. Compute the flash and SRAM headroom with A/B slots.
*Deliverable:* two tables and the headroom percentages.
*Acceptance criteria:* flash used = 24 + 360 + 60 + 4 + 8 = 456 KiB, free 56 KiB (10.9%); SRAM used = 69 KiB, free 59 KiB (46.1%).
*Time:* 40 minutes.

**Exercise 3 — Runtime choice (analysis, 30 minutes).**
A model uses depthwise convolution, GELU, and layer normalization on a Cortex-M4. Choose a runtime and state the operator risk.
*Deliverable:* a recommendation with a support analysis.
*Acceptance criteria:* names the unsupported operators; proposes replacements or fallbacks; states the memory implication of the fallback code.
*Time:* 30 minutes.

**Exercise 4 — Watchdog and fallback (design, 35 minutes).**
Design the watchdog period, the inference timeout, and the fallback for a device with p50 45 ms, p99 70 ms, and a 250 ms deadline.
*Deliverable:* a table of thresholds and actions.
*Acceptance criteria:* watchdog period exceeds p99 with margin; timeout is below the deadline; fallback is resident and defined.
*Time:* 35 minutes.

**Exercise 5 — Update state machine (design, 45 minutes).**
Draw the update state machine with download, verify, install, reboot, self-test, commit, and rollback. Identify the power-loss behavior at each transition.
*Deliverable:* a state diagram and a table.
*Acceptance criteria:* every transition has a defined recovery; the invalid slot is detectable; the commit is atomic.
*Time:* 45 minutes.

**Exercise 6 — Telemetry schema (design, 30 minutes).**
Design a telemetry schema for monitoring model drift without PHI. List the fields, the aggregation, and the retention.
*Deliverable:* a schema table.
*Acceptance criteria:* no identifiers or raw signals; includes version, latency, saturation, and alert counts; retention stated.
*Time:* 30 minutes.

**Exercise 7 — Test plan (design, 45 minutes).**
Write a test plan for boot, update, power loss, watchdog, sensor timeout, and soak. State the pass criteria.
*Deliverable:* a test matrix.
*Acceptance criteria:* every failure mode has a test; pass criteria are observable; the production configuration is specified.
*Time:* 45 minutes.

**Exercise 8 — Configuration record (design, 40 minutes).**
Write the configuration record and a minimal SBOM for the Section 6 device.
*Deliverable:* a one-page record.
*Acceptance criteria:* board, bootloader, firmware, runtime, model, calibration versions; SBOM components; test report reference.
*Time:* 40 minutes.

## 13. Self-check answers

**A1. Flash and SRAM for Exercise 2.**
Flash: bootloader 24 + firmware 2×180 = 360 + model 2×30 = 60 + config 4 + logs 8 = 456 KiB; free 56 KiB, or 10.9%. SRAM: arena 48 + DMA 12 + stack 6 + runtime 3 = 69 KiB; free 59 KiB, or 46.1%.

**A2. Why A/B slots are worth the flash.**
Because a power loss during an update to a single slot can leave the device without a valid image. A/B slots ensure that the running image is never modified, so the device can always boot. The cost is one extra firmware and model slot; the benefit is a device that cannot be bricked by a routine update.

**A3. Why the watchdog period must exceed p99 latency.**
Because the watchdog is serviced by the main loop, and a slow inference delays the service. If the period is set from p50, a normal p99 inference will reset the device. Set it from the sustained p99 with margin, and use a separate inference timeout for the deadline.

**A4. What is DMA coherency and why does it matter?**
A DMA engine writes to memory without the CPU's cache being updated. If the CPU reads a cached copy, it sees stale data. The fix is to invalidate the cache region after the DMA completes, or to use non-cached memory for the buffer. A missing invalidate produces plausible but wrong sensor data.

**A5. What belongs in an SBOM?**
Third-party software components, their versions, licenses, and known vulnerabilities, plus the runtime, the bootloader, and any cryptography library. The model is a configuration item, not usually an SBOM component, but its version and hash belong in the configuration record.

**A6. Why is the safe state a clinical decision?**
Because the consequence of a false negative and a false positive differ by device. For an alarm, failing to alarm may be worse, so the safe state is to alarm. For a therapy, a false delivery may be worse, so the safe state is to stop. The firmware cannot choose this without the clinical risk analysis.

**A7. What does an A/B update cost in flash?**
One extra firmware slot and one extra model slot. For 300 KiB firmware and 41 KiB model, that is 341 KiB, or 33% of a 1 MiB device. The cost is the price of rollback.

**A8. Why must the update path be tested with power loss?**
Because the update writes to flash, and a power loss during a write can leave the slot invalid. The bootloader must detect the invalid slot and boot the valid one. A test that only checks a successful update does not exercise the failure.

**A9. Why does a model update need a self-test rather than a load check?**
A load check proves the runtime can parse the file. A self-test proves the model produces a plausible output on a known input. A corrupted or mismatched model can load successfully and produce a constant or wrong output, which the load check cannot detect.

**A10. What is the difference between cold boot and wake?**
Cold boot runs the bootloader, verifies the image, initializes the runtime, and allocates memory. Wake restores clocks and peripherals and resumes the application. Cold boot is measured in hundreds of milliseconds to seconds; wake in milliseconds. A device that sleeps between inferences is bound by wake, not boot.

**A11. Rubric for Exercise 5 (state machine).**
A strong state machine defines every state and transition, marks the commit as atomic, defines the invalid-slot detection, and specifies the rollback trigger. A weak design lists the happy path only.

**A12. Rubric for Exercise 8 (configuration record).**
A strong record names the board revision, bootloader, firmware, runtime, model hash, calibration version, and SBOM components, and references the test report. A weak record gives a firmware version and a model file name.

## 14. Interview questions

**Q1. What is the difference between flash and SRAM budgets?**
A strong answer says flash holds persistent code and model, SRAM holds the runtime arena and buffers, and a model can fit in flash and fail in SRAM. A weak answer treats memory as one budget.

**Q2. Why does a device need a watchdog and a timeout?**
A strong answer distinguishes the device-level reset from the operation-level deadline and explains the fallback. A weak answer uses the terms interchangeably.

**Q3. What is an A/B update?**
A strong answer describes the inactive slot, the bootloader selection, the self-test, and the rollback. A weak answer calls it a backup.

**Q4. How do you size the tensor arena?**
A strong answer measures the runtime high-water mark, adds headroom for scratch and fragmentation, and iterates. A weak answer sums the tensor sizes.

**Q5. What is DMA coherency?**
A strong answer explains the cache invalidation requirement and the stale-data failure. A weak answer says DMA is fast.

**Q6. What belongs in telemetry for a medical device?**
A strong answer lists versions, latency, saturation, fallback counts, and alert counts, and excludes raw signals and identifiers. A weak answer proposes sending the raw signal for analysis.

**Q7. How do you validate an update path?**
A strong answer tests valid, corrupted, and unsigned bundles, power loss at each stage, rollback, and anti-rollback. A weak answer tests a successful update.

**Q8. Why is the runtime version part of the configuration?**
A strong answer says kernel selection and memory planning can change, so the same model can behave differently. A weak answer treats the runtime as infrastructure.

**Q9. What is the safe state for an alarm device?**
A strong answer says it depends on the clinical risk analysis, and for many alarms it is to alarm rather than to stay silent. A weak answer assumes silence is safe.

**Q10. How would you reduce SRAM usage?**
A strong answer reduces the input window, fuses operators, shrinks buffers, and measures the arena, in that order. A weak answer suggests a larger part.

**Q11. What is an SBOM and why does it matter?**
A strong answer lists components, versions, and vulnerabilities, and connects it to incident response. A weak answer calls it a dependency list.

**Q12. Why is the boot path part of the clinical latency budget?**
A strong answer distinguishes cold boot, wake, and warm restart, and explains that a device that must alert immediately after power-on is bound by boot, not inference. A weak answer considers only inference time.

**Q13. What does a fault-injection matrix test?**
A strong answer lists power loss, corruption, bad signatures, sensor loss, watchdog stalls, and brownout, and states the expected behavior for each. A weak answer tests only the happy path.

**Q14. When would you reject an architecture?**
A strong answer rejects on no rollback, no watchdog, no safe state, insufficient headroom, unverified coherency, or a telemetry schema that leaks PHI. A weak answer rejects only on memory.

## 15. Key terms

- **A/B slots** — two firmware or model partitions for safe updates.
- **Anti-rollback** — refusing older image versions.
- **Arena** — preallocated memory for tensors.
- **Bootloader** — code that loads and verifies the application.
- **Brownout** — supply voltage below the operating threshold.
- **Configuration item** — item under version control.
- **DMA** — direct memory access.
- **Double buffering** — two buffers alternating between producer and consumer.
- **Fallback** — alternative path when the main path fails.
- **FlatBuffer** — binary serialization with random access.
- **Fragmentation** — unusable free memory between allocations.
- **Model format** — serialized representation of a model.
- **OTA** — over-the-air update.
- **Partition** — flash region or graph subgraph.
- **PHI** — protected health information.
- **Protobuf** — binary serialization format.
- **Rollback** — reverting to a previous image.
- **Runtime** — software that executes a model.
- **Safe state** — output presented when computation is unavailable.
- **SBOM** — software bill of materials.
- **Secure boot** — verifying the boot image's signature.
- **Signed bundle** — update package with a cryptographic signature.
- **Soak test** — sustained load test.
- **Static allocation** — memory reserved at compile time.
- **Telemetry** — operational data sent from the device.
- **Timeout** — deadline for an operation.
- **Watchdog** — timer that resets the device if not serviced.
- **Wake time** — time from sleep to ready.
- **Boot time** — time from power-on to ready.
- **Cold boot** — full boot through the bootloader.
- **Warm restart** — restart after a watchdog reset.
- **Self-test** — boot check of sensor, model, and known input.
- **Load check** — verification that the model loads, not that it works.
- **Secure element** — protected key storage and cryptographic operations.
- **Watchdog window** — a service window that detects both too-early and too-late checks.
- **Ring buffer** — fixed-size log that overwrites the oldest entries.

## 16. Sources

All URLs accessed 25 October 2026.

**Runtime documentation**
- Google AI Edge, "LiteRT for Microcontrollers overview". https://developers.google.com/edge/litert/microcontrollers/overview
- TensorFlow, "Memory management in TFLM". https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md
- PyTorch, "ExecuTorch `.pte` file format". https://docs.pytorch.org/executorch/stable/pte-file-format.html
- PyTorch, "ExecuTorch backends". https://docs.pytorch.org/executorch/main/backends-section.html
- ONNX Runtime, "Graph optimizations". https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- ggml-org, "Tensor Encoding Schemes", llama.cpp wiki. https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- IEC 60601-1. https://www.iec.ch/
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, "Cybersecurity in Medical Devices". https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

**Quantization and performance**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877
- Arm, "CMSIS-NN". https://arm-software.github.io/CMSIS-NN/latest/

> **Note:** The Section 6 memory maps and measurements are constructed for teaching. Runtime memory models are vendor documentation for specific versions.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`07_inference-optimization.md`](07_inference-optimization.md)
- Next topic: [`09_medical-and-regulatory-constraints.md`](09_medical-and-regulatory-constraints.md)
- Related topic: [`02_target-hardware-and-performance.md`](02_target-hardware-and-performance.md)
- Related topic: [`10_validation-on-target-hardware.md`](10_validation-on-target-hardware.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
